"""AstrBot 对抗人格守护插件（GAN 式双 AI 对抗）。

架构：
- B（生成器）= AstrBot 主 AI，负责真实对话；
- A（判别器）= 判官 AI，读取人格提示词，审查 B 的每次输出是否遵守规则；
- 违规时执行 快改→重生成 的对抗修复循环，并以奖惩账本 +
  上下文反馈注入模拟强化学习，让 B 在后续对话中持续纠偏。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

try:
    # v4.24.1+ 提供的临时内容块机制：只参与本轮请求，不写入会话历史、不破坏缓存
    from astrbot.core.agent.message import TextPart
except Exception:  # pragma: no cover - 兼容旧版本
    TextPart = None

from .core.judge import (
    JUDGE_SYSTEM_PROMPT,
    Verdict,
    _content_to_text,
    build_history_block,
    build_judge_user_prompt,
    format_violations,
    is_pass,
    parse_verdict,
)
from .core.ledger import Ledger
from .core.memory import MemoryStore
from .core.rules import (
    EXTRACT_SYSTEM_PROMPT,
    RulesStore,
    build_extract_prompt,
    parse_rules,
    render_rules_block,
    rule_id_of,
)
from .core.summarizer import (
    SUMMARY_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    build_review_prompt,
    build_summary_prompt,
    parse_review,
    parse_summary,
    resolve_entry,
)
from .core.repair import (
    ACTION_FAST_FIX_THEN_REGEN,
    ACTION_RECORD_ONLY,
    run_adversarial_repair,
)

PLUGIN_NAME = "astrbot_plugin_adversarial_guard"
PENDING_TTL = 600  # 请求快照保留时长（秒）
PENDING_CAP = 200  # 快照字典容量上限
PERSONA_CACHE_CAP = 100  # 每个 umo 缓存的最近人格提示词条数

HELP_TEXT = (
    "对抗人格守护（判官A × 主AI B × 记忆C）\n"
    "/guard - 查看状态与帮助\n"
    "/guard on|off - 本会话开关审查\n"
    "/guard stats - 查看奖惩统计与近 7 天趋势\n"
    "/guard rules - 查看本会话规则库（判官逐条对照）\n"
    "/guard patch - 根据高频违规生成人格补丁建议（管理员）\n"
    "/guard memory - 查看本会话长期记忆（C总结+A复审的教训）\n"
    "/guard forget <序号|all> - 删除长期记忆（管理员）\n"
    "/guard test <文本> - 让判官A审查指定文本\n"
    "/guard reset - 清空本会话账本（管理员）"
)


def _resolve_data_dir() -> str:
    """优先使用 AstrBot 的插件数据目录，失败则退回相对路径。"""
    try:
        from astrbot.api.star import StarTools

        return str(StarTools.get_data_dir(PLUGIN_NAME))
    except Exception:
        return os.path.join("data", "plugin_data", PLUGIN_NAME)


def _is_admin(event: AstrMessageEvent) -> bool:
    fn = getattr(event, "is_admin", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            pass
    return getattr(event, "role", "") == "admin"


class AdversarialGuard(Star):
    """双 AI 对抗守护：on_llm_request 捕获人格并注入反馈，on_llm_response 审查修复。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        data_dir = _resolve_data_dir()
        os.makedirs(data_dir, exist_ok=True)
        self.ledger = Ledger(
            path=os.path.join(data_dir, "ledger.json"),
            decay=self._fget("reward_decay", 0.8),
            window=max(1, self._iget("feedback_window", 5, 1)),
        )
        # v1.1.0：长期记忆（C 总结 + A 复审后的教训条目）
        self.memory = MemoryStore(
            path=os.path.join(data_dir, "memory.json"),
            max_entries=self._iget("memory_max_entries", 20, 1, 100),
        )
        # 后台总结任务集合（插件卸载时等待收尾）
        self._summary_tasks: set = set()
        # v1.2.0：规则库 + 审查缓存 + 人工反馈冷却
        self.rules_store = RulesStore(path=os.path.join(data_dir, "rules.json"))
        self._review_cache: dict[tuple, dict] = {}  # (persona_hash, reply_hash) -> verdict dict
        self._last_reply_ts: dict[str, float] = {}  # umo -> 最近一次被审回复时间
        self._fb_last: dict[str, float] = {}  # umo -> 上次人工反馈记账时间
        self._pending: dict[int, dict] = {}  # id(event) -> 请求快照
        self._last_persona: dict[str, str] = {}  # umo -> 最近一次人格提示词

    # ================================================================ 配置
    def _refresh_ledger_params(self) -> None:
        self.ledger.set_params(
            self._fget("reward_decay", 0.8),
            max(1, self._iget("feedback_window", 5, 1)),
        )

    def _action(self) -> str:
        v = str(self.config.get("violation_action") or ACTION_FAST_FIX_THEN_REGEN)
        return (
            v
            if v in (ACTION_FAST_FIX_THEN_REGEN, "regen_only", ACTION_RECORD_ONLY)
            else ACTION_FAST_FIX_THEN_REGEN
        )

    def _threshold(self) -> int:
        return self._iget("score_threshold", 70, 0, 100)

    def _timeout(self) -> float:
        return max(3.0, self._fget("judge_timeout", 30.0))

    def _max_judge_chars(self) -> int:
        return max(200, self._iget("max_judge_chars", 4000, 200, 10**7))

    def _iget(self, key: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
        try:
            # 注意：不能用 `or default`——显式配置 0（如抽样率 0、熔断阈值 0）会被 falsy 吞掉
            raw = self.config.get(key, default)
            val = int(raw)
        except (TypeError, ValueError):
            val = int(default)
        if lo is not None:
            val = max(lo, val)
        if hi is not None:
            val = min(hi, val)
        return val

    def _fget(self, key: str, default: float) -> float:
        try:
            raw = self.config.get(key, default)
            return float(raw)
        except (TypeError, ValueError):
            return float(default)

    def _enabled_for(self, event: AstrMessageEvent) -> bool:
        if not bool(self.config.get("enable", True)):
            return False
        toggle = self.ledger.get_toggle(event.unified_msg_origin)
        return True if toggle is None else toggle

    # ============================================================ Provider
    async def _get_main_provider(self, event: AstrMessageEvent):
        getter = getattr(self.context, "get_using_provider_async", None)
        if getter is not None:
            try:
                return await getter(umo=event.unified_msg_origin)
            except Exception:
                pass
        legacy = getattr(self.context, "get_using_provider", None)
        if callable(legacy):
            try:
                return legacy()
            except Exception:
                return None
        return None

    async def _get_judge_provider(self, event: AstrMessageEvent):
        pid = str(self.config.get("judge_provider_id") or "").strip()
        if pid:
            try:
                prov = self.context.get_provider_by_id(pid)
                if prov is not None:
                    return prov
            except Exception:
                logger.warning(f"[{PLUGIN_NAME}] 判官 provider_id '{pid}' 无效，回退主提供商")
        return await self._get_main_provider(event)

    async def _chat_text(
        self,
        provider,
        system_prompt: str,
        prompt: str,
        contexts: list | None = None,
        *,
        use_judge_model: bool,
    ) -> str:
        kwargs = {
            "prompt": prompt,
            "contexts": contexts or [],
            "system_prompt": system_prompt,
        }
        model = str(self.config.get("judge_model") or "").strip()
        if use_judge_model and model:
            kwargs["model"] = model
        resp = await provider.text_chat(**kwargs)
        return (getattr(resp, "completion_text", "") or "").strip()

    # ============================================================== 钩子
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """捕获本轮人格提示词/上下文快照，并把奖惩反馈注入请求。"""
        try:
            if not self._enabled_for(event):
                return
            self._refresh_ledger_params()
            prompt_text = self._extract_prompt(req)
            contexts = list(getattr(req, "contexts", None) or [])
            persona = getattr(req, "system_prompt", "") or ""

            # v1.2.0：豁免名单（指令前缀/用户/正则）——命中则本轮不审查（不存快照）
            if self._is_exempt_from_review(event, prompt_text):
                return

            # 先存干净快照（供判官使用），再做注入，避免判官看到反馈文本本身
            self._pending[id(event)] = {
                "persona": persona,
                "contexts": contexts,
                "prompt": prompt_text,
                "ts": time.time(),
            }
            self._gc_pending()

            umo = event.unified_msg_origin
            if persona:
                self._last_persona[umo] = persona
                if len(self._last_persona) > PERSONA_CACHE_CAP:
                    self._last_persona.pop(next(iter(self._last_persona)))
                # v1.2.0：规则库为空或 persona 变化 → 后台重新提取（不阻塞）
                if (
                    bool(self.config.get("rules_extract_enabled", True))
                    and self.rules_store.is_stale(umo, persona)
                ):
                    self._schedule_rules_extract(event, persona)

            # v1.1.0：长期记忆注入（C 总结、A 复审过的教训条目），先于窗口反馈
            if bool(self.config.get("memory_injection", True)):
                memory_text = MemoryStore.render_entries(self.memory.entries(umo))
                if memory_text:
                    self._inject_text(req, persona, memory_text)

            if not bool(self.config.get("feedback_injection", True)):
                return
            feedback = self.ledger.injection_text(umo, self._threshold())
            if not feedback:
                return
            self._inject_text(req, persona, feedback)
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] on_llm_request 异常(已忽略): {e}")

    @staticmethod
    def _inject_text(req: ProviderRequest, persona: str, text: str) -> None:
        """把附加文本注入请求：优先临时内容块（不写入历史、不破坏缓存），旧版回退 system_prompt。"""
        parts = getattr(req, "extra_user_content_parts", None)
        if parts is not None and TextPart is not None:
            parts.append(TextPart(text=text).mark_as_temp())
        elif persona is not None:
            req.system_prompt = persona + "\n" + text

    # ================================================== v1.2.0 豁免/抽样/缓存
    def _is_exempt_from_review(self, event: AstrMessageEvent, prompt_text: str) -> bool:
        """豁免名单：指令前缀 / 用户 ID / 正则 任一命中则本轮不审查。"""
        text = (getattr(event, "message_str", "") or "").strip()
        prefixes = [
            x.strip() for x in str(self.config.get("exempt_prefixes", "") or "").replace("，", ",").split(",")
            if x.strip()
        ]
        for p in prefixes:
            if text.startswith(p) or prompt_text.startswith(p):
                return True
        users = {
            x.strip() for x in str(self.config.get("exempt_users", "") or "").replace("，", ",").split(",")
            if x.strip()
        }
        if users:
            try:
                if str(event.get_sender_id() or "") in users:
                    return True
            except Exception:
                pass
        patterns_raw = str(self.config.get("exempt_patterns", "") or "").replace("，", ",")
        for p in [x.strip() for x in patterns_raw.split(",") if x.strip()]:
            try:
                if re.search(p, text) or re.search(p, prompt_text):
                    return True
            except re.error:
                continue
        return False

    def _should_review_this_round(self) -> bool:
        """审查抽样：review_sample_rate 0-100（默认 100 全量）。"""
        rate = self._iget("review_sample_rate", 100, 0, 100)
        if rate >= 100:
            return True
        if rate <= 0:
            return False
        return random.random() * 100 < rate

    @staticmethod
    def _reply_hash(text: str, persona: str) -> tuple:
        return (
            hashlib.md5((persona or "").encode("utf-8")).hexdigest(),
            hashlib.md5((text or "").encode("utf-8")).hexdigest(),
        )

    def _cache_get_verdict(self, reply: str, persona: str) -> Verdict | None:
        if not bool(self.config.get("review_cache_enabled", True)):
            return None
        import time as _t
        entry = self._review_cache.get(self._reply_hash(reply, persona))
        if not entry:
            return None
        if _t.time() - entry["ts"] > 3600:
            self._review_cache.pop(self._reply_hash(reply, persona), None)
            return None
        v = entry["verdict"]
        logger.info(f"[{PLUGIN_NAME}] 审查缓存命中，复用裁决 score={v.score}")
        return v

    def _cache_put_verdict(self, reply: str, persona: str, verdict: Verdict) -> None:
        if not bool(self.config.get("review_cache_enabled", True)):
            return
        if len(self._review_cache) > 200:
            self._review_cache.clear()
        self._review_cache[self._reply_hash(reply, persona)] = {
            "ts": time.time(),
            "verdict": verdict,
        }

    # ================================================== v1.2.0 修复熔断
    def _effective_action_with_fuse(self, umo: str) -> str:
        """修复熔断：连续违规达到阈值后，冷却期内强制 record_only，避免修复风暴。"""
        action = self._action()
        threshold = self._iget("fuse_threshold", 0, 0, 100)
        if threshold <= 0:
            return action
        fused_until = self.ledger.fused_until(umo)
        if fused_until > time.time():
            return ACTION_RECORD_ONLY
        if self.ledger.violate_streak(umo) >= threshold:
            minutes = self._iget("fuse_cooldown_minutes", 10, 1, 1440)
            self.ledger.set_fused_until(umo, time.time() + minutes * 60)
            logger.warning(
                f"[{PLUGIN_NAME}] 连续违规 {threshold} 次，修复熔断 {minutes} 分钟（强制 record_only）"
            )
            return ACTION_RECORD_ONLY
        return action

    # ================================================== v1.2.0 规则库提取
    def _schedule_rules_extract(self, event: AstrMessageEvent, persona: str):
        """后台提取人格规则清单（幂等：进行中不重复调度）。"""
        umo = event.unified_msg_origin
        if getattr(self, "_rules_extracting", None) == umo:
            return
        self._rules_extracting = umo
        task = asyncio.create_task(self._extract_rules(event, persona))
        self._summary_tasks.add(task)
        task.add_done_callback(self._summary_tasks.discard)

    async def _extract_rules(self, event: AstrMessageEvent, persona: str):
        umo = event.unified_msg_origin
        try:
            prompt = build_extract_prompt(persona)
            try:
                raw = await asyncio.wait_for(
                    self._chat_text_with(event, "summarizer", EXTRACT_SYSTEM_PROMPT, prompt),
                    timeout=self._timeout(),
                )
            except Exception as e:
                logger.debug(f"[{PLUGIN_NAME}] 规则提取调用失败: {e}")
                return
            rules = parse_rules(raw)
            if not rules:
                logger.debug(f"[{PLUGIN_NAME}] 人格中未提取到明文规则，本会话使用模糊对照")
                self.rules_store.set_rules(umo, persona, [])
                return
            self.rules_store.set_rules(umo, persona, rules)
            logger.info(f"[{PLUGIN_NAME}] 规则库已提取 {len(rules)} 条（{umo}）")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[{PLUGIN_NAME}] 规则提取任务异常（已忽略）: {e}")
        finally:
            if getattr(self, "_rules_extracting", None) == umo:
                self._rules_extracting = None

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """审查 B 的最终回复；违规时触发对抗修复并记账。"""
        # 注意：流式 chunk / 工具中间态都会触发本钩子，此时不消费快照，留给最终响应
        snap = self._pending.get(id(event))
        if snap is None:
            return
        try:
            if getattr(resp, "is_chunk", False):
                return
            if getattr(resp, "role", "assistant") != "assistant":
                self._pending.pop(id(event), None)
                return
            if getattr(resp, "tools_call_name", None):
                snap["ts"] = time.time()  # Agent 多轮工具调用期间刷新 TTL
                return
            if not self._enabled_for(event):
                self._pending.pop(id(event), None)
                return
            reply = (getattr(resp, "completion_text", "") or "").strip()
            if not reply:
                self._pending.pop(id(event), None)
                return
            # 正式进入审查流程：消费快照
            snap = self._pending.pop(id(event))

            umo = event.unified_msg_origin
            threshold = self._threshold()
            persona = snap["persona"]
            user_msg = snap["prompt"]

            # v1.2.0：审查抽样（省 token）
            if not self._should_review_this_round():
                logger.debug(f"[{PLUGIN_NAME}] 本轮抽样未命中，跳过审查（rate={self._iget('review_sample_rate', 100, 0, 100)}）")
                return

            history_block = build_history_block(
                snap["contexts"],
                exclude_prompt=user_msg,
                max_turns=max(0, self._iget("history_turns_for_judge", 6, 0)),
            )

            # v1.2.0：审查缓存（同 persona + 同回复复用裁决）
            verdict = self._cache_get_verdict(reply, persona)
            if verdict is None:
                verdict = await self._judge_once(event, persona, history_block, user_msg, reply)
                if verdict is not None:
                    self._cache_put_verdict(reply, persona, verdict)
            if verdict is None:
                logger.warning(f"[{PLUGIN_NAME}] 判官不可用或输出无法解析，放行原文")
                return
            self._last_reply_ts[umo] = time.time()

            ok = is_pass(verdict, threshold)
            delta = self.ledger.record_check(
                umo, ok, verdict.score, [] if ok else verdict.violations, threshold
            )
            if ok:
                logger.info(f"[{PLUGIN_NAME}] 审查通过 score={verdict.score} delta=+{delta}")
                return

            logger.info(
                f"[{PLUGIN_NAME}] 检测到违规 score={verdict.score} delta={delta}\n"
                f"{format_violations(verdict.violations)}"
            )

            # v1.2.0：修复熔断（连续违规达阈值 → 冷却期强制 record_only）
            result = await run_adversarial_repair(
                chat_fn=self._make_chat_fn(event, snap),
                judge_fn=lambda text: self._judge_once(
                    event, persona, history_block, user_msg, text
                ),
                action=self._effective_action_with_fuse(umo),
                persona=persona,
                contexts=snap["contexts"],
                user_msg=user_msg,
                original_reply=reply,
                first_verdict=verdict,
                threshold=threshold,
                max_regen_rounds=max(0, self._iget("max_regen_rounds", 2, 0)),
            )

            final = (result.final_text or "").strip()
            if final and final != reply:
                resp.completion_text = final
            self.ledger.record_repair(umo, result.ok, result.method)
            logger.info(f"[{PLUGIN_NAME}] 修复完成 method={result.method} ok={result.ok}")
            if bool(self.config.get("notify_on_fix", False)) and final != reply:
                # 区分"已通过复检的修正"与"仅替换为更优版本但仍未过审"，避免误导用户
                notice = (
                    f"[守护] 已由判官A修正本次回复（method={result.method}）"
                    if result.ok
                    else f"[守护] 已尝试修正本次回复，但仍未完全通过审查（method={result.method}）"
                )
                await self._send_notice(event, notice)

            # v1.1.0：后台异步总结（C 生成教训 → A 三维复审 → 写入长期记忆）
            if (
                bool(self.config.get("memory_enabled", True))
                and self.ledger is not None
            ):
                task = asyncio.create_task(
                    self._summarize_and_store(
                        event, verdict_dict=self._verdict_to_dict(verdict),
                        repair_info={
                            "method": result.method,
                            "ok": result.ok,
                            "attempts": result.attempts,
                        },
                        reward=self.ledger.get_reward(umo),
                    )
                )
                self._summary_tasks.add(task)
                task.add_done_callback(self._summary_tasks.discard)
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] on_llm_response 异常(已忽略): {e}")

    # ================================================== 长期记忆：总结与复审
    @staticmethod
    def _verdict_to_dict(v) -> dict:
        """把 Verdict 裁决压缩为可序列化摘要（供总结/复审提示词使用）。"""
        return {
            "compliant": bool(v.compliant),
            "score": int(v.score),
            "violations": [
                {
                    "rule": x.get("rule", ""),
                    "evidence": x.get("evidence", ""),
                    "severity": x.get("severity", 1),
                }
                for x in (v.violations or [])
            ],
        }

    def _refresh_memory_params(self) -> None:
        self.memory.set_max_entries(self._iget("memory_max_entries", 20, 1, 100))

    async def _chat_text_with(
        self, event: AstrMessageEvent, provider_kind: str,
        system_prompt: str, prompt: str,
    ) -> str:
        """按 provider_kind（judge/summarizer）选择模型并调用，返回文本。

        - summarizer：优先 summarizer_provider_id/summarizer_model，
          未配置时回退判官提供商；
        - judge：判官提供商 + judge_model。
        """
        if provider_kind == "summarizer":
            pid = str(self.config.get("summarizer_provider_id") or "").strip()
            model = str(self.config.get("summarizer_model") or "").strip()
            provider = None
            if pid:
                try:
                    provider = self.context.get_provider_by_id(pid)
                except Exception:
                    provider = None
                if provider is None:
                    logger.warning(
                        f"[{PLUGIN_NAME}] 总结 provider_id '{pid}' 无效，回退判官/主提供商"
                    )
            if provider is None:
                provider = await self._get_judge_provider(event)
        else:
            provider = await self._get_judge_provider(event)
            model = str(self.config.get("judge_model") or "").strip()
        if provider is None:
            raise RuntimeError("无可用提供商")
        kwargs = {
            "prompt": prompt,
            "contexts": [],
            "system_prompt": system_prompt,
        }
        if model:
            kwargs["model"] = model
        resp = await provider.text_chat(**kwargs)
        return (getattr(resp, "completion_text", "") or "").strip()

    async def _summarize_and_store(
        self, event: AstrMessageEvent, verdict_dict: dict,
        repair_info: dict, reward: float,
    ) -> None:
        """后台任务：C 总结教训 → A 三维复审（忠实/简洁/隐私）→ 写入长期记忆。

        任一环节失败均静默丢弃本轮（fail-open），绝不影响正常聊天。
        """
        umo = event.unified_msg_origin
        try:
            self._refresh_memory_params()
            timeout = self._timeout()
            # ---- 1) C：生成教训草稿 ----
            prompt = build_summary_prompt(verdict_dict, repair_info, reward)
            try:
                raw = await asyncio.wait_for(
                    self._chat_text_with(event, "summarizer", SUMMARY_SYSTEM_PROMPT, prompt),
                    timeout=timeout,
                )
            except Exception as e:
                logger.debug(f"[{PLUGIN_NAME}] 总结模型调用失败（本轮不写记忆）: {e}")
                return
            draft = parse_summary(raw)
            if draft is None:
                logger.debug(f"[{PLUGIN_NAME}] 总结输出不可用（keep=false 或解析失败），本轮不写记忆")
                return
            draft["meta"] = {"score": verdict_dict.get("score")}

            # ---- 2) A：三维复审（忠实性/简洁性/隐私） ----
            entry: dict | None
            if bool(self.config.get("summary_review_enabled", True)):
                review_prompt = build_review_prompt(verdict_dict, repair_info, draft)
                try:
                    raw = await asyncio.wait_for(
                        self._chat_text_with(event, "judge", REVIEW_SYSTEM_PROMPT, review_prompt),
                        timeout=timeout,
                    )
                except Exception as e:
                    logger.debug(f"[{PLUGIN_NAME}] 记忆复审调用失败，信任草稿写入: {e}")
                    review = None
                else:
                    review = parse_review(raw)
                    if review is None:
                        logger.debug("[{PLUGIN_NAME}] 复审输出解析失败，本轮不写记忆")
                        return
                entry = resolve_entry(draft, review)
                if entry is None:
                    logger.info(f"[{PLUGIN_NAME}] 教训草稿被复审驳回（{review.get('reason', '未说明')}），已丢弃")
                    return
                if entry.get("meta", {}).get("corrected"):
                    logger.info(f"[{PLUGIN_NAME}] 教训草稿经复审修正后写入")
            else:
                entry = resolve_entry(draft, None)

            # ---- 3) 写入长期记忆 ----
            saved = self.memory.add(
                umo, rule=entry["rule"], lesson=entry["lesson"], meta=entry.get("meta"),
            )
            logger.info(
                f"[{PLUGIN_NAME}] 长期记忆已更新：{saved['rule']}（累计 {saved['count']} 次）"
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[{PLUGIN_NAME}] 长期记忆总结任务异常（已忽略）: {e}")

    # =========================================================== 内部流程
    async def _judge_once(
        self,
        event: AstrMessageEvent,
        persona: str,
        history_block: str,
        user_msg: str,
        reply: str,
    ) -> Verdict | None:
        """调用判官 A 审查一次回复；任何失败返回 None。

        v1.2.0：会话规则库存在时，附加 [规则清单] 块供判官逐条对照（R1/R2/...）。
        """
        provider = await self._get_judge_provider(event)
        if provider is None:
            return None
        limit = self._max_judge_chars()
        judged_reply = reply if len(reply) <= limit else reply[:limit] + "...(截断)"
        user_prompt = build_judge_user_prompt(persona, history_block, user_msg, judged_reply)
        rules = self.rules_store.get(event.unified_msg_origin)
        rules_block = render_rules_block(rules)
        if rules_block:
            # 规则清单插在人格提示词之后，判官可逐条对照
            marker = "[近几轮对话]"
            if marker in user_prompt:
                user_prompt = user_prompt.replace(
                    marker, rules_block + "\n\n" + marker, 1
                )
            else:
                user_prompt = user_prompt + "\n\n" + rules_block
        try:
            raw = await asyncio.wait_for(
                self._chat_text(provider, JUDGE_SYSTEM_PROMPT, user_prompt, use_judge_model=True),
                timeout=self._timeout(),
            )
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 判官调用失败: {e}")
            return None
        return parse_verdict(raw)

    def _make_chat_fn(self, event: AstrMessageEvent, snap: dict):
        """构造重生成用的主 AI(B) 调用函数（始终用主模型，不带判官模型覆盖）。"""

        async def chat_fn(prompt: str) -> str:
            provider = await self._get_main_provider(event)
            if provider is None:
                raise RuntimeError("无可用的主 AI 提供商")
            return await asyncio.wait_for(
                self._chat_text(
                    provider,
                    snap["persona"],
                    prompt,
                    contexts=snap["contexts"],
                    use_judge_model=False,
                ),
                timeout=self._timeout(),
            )

        return chat_fn

    async def _send_notice(self, event: AstrMessageEvent, text: str):
        try:
            await event.send(event.make_result().message(text))
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 发送提示失败: {e}")

    @staticmethod
    def _extract_prompt(req: ProviderRequest) -> str:
        p = (getattr(req, "prompt", "") or "").strip()
        if p:
            return p[:8000]
        contexts = getattr(req, "contexts", None) or []
        for m in reversed(contexts):
            if isinstance(m, dict) and m.get("role") == "user":
                t = _content_to_text(m.get("content"))
                if t:
                    return t[:8000]
        return ""

    def _gc_pending(self):
        now = time.time()
        stale = [k for k, v in self._pending.items() if now - v["ts"] > PENDING_TTL]
        for k in stale:
            self._pending.pop(k, None)
        while len(self._pending) > PENDING_CAP:
            self._pending.pop(next(iter(self._pending)))

    # ============================================================= 指令
    @filter.event_message_type(EventMessageType.ALL, priority=1)
    async def human_feedback_listener(self, event: AstrMessageEvent):
        """v1.2.0 人工反馈奖惩：用户口头正/负反馈计入纪律分（模拟 RLHF）。

        仅在 bot 最近 120 秒内产生过被审回复时生效；每会话 60 秒冷却防刷。
        """
        try:
            if not bool(self.config.get("human_feedback_enabled", False)):
                return
            # 过滤 bot 自己的消息（bot 回复中含反馈词时不得误记账）
            try:
                self_id = str(event.get_self_id() or "")
                sender_id = str(event.get_sender_id() or "")
                if self_id and sender_id and self_id == sender_id:
                    return
            except Exception:
                pass
            if self._pending.get(id(event)):
                return  # 本消息自身会触发 LLM（命令/唤醒），不算反馈
            text = (getattr(event, "message_str", "") or "").strip()
            if not text or len(text) > 30:
                return
            umo = event.unified_msg_origin
            last = self._last_reply_ts.get(umo, 0)
            now = time.time()
            if now - last > 120:
                return
            if now - self._fb_last.get(umo, 0) < 60:
                return
            neg_words = [
                x.strip() for x in str(self.config.get("feedback_negative_words", "不对,说错了,答错了,不是这样,胡说")).replace("，", ",").split(",")
                if x.strip()
            ]
            pos_words = [
                x.strip() for x in str(self.config.get("feedback_positive_words", "说得好,答对了,对了,不错,很棒")).replace("，", ",").split(",")
                if x.strip()
            ]
            positive = None
            if any(w in text for w in neg_words):
                positive = False
            elif any(w in text for w in pos_words):
                positive = True
            if positive is None:
                return
            self._fb_last[umo] = now
            delta = self.ledger.record_human_feedback(umo, positive)
            kind = "正面" if positive else "负面"
            logger.info(f"[{PLUGIN_NAME}] 人工{kind}反馈（{text[:20]}）delta={delta}")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[{PLUGIN_NAME}] 人工反馈处理异常（已忽略）: {e}")

    @filter.command("guard")
    async def guard_command(self, event: AstrMessageEvent):
        """/guard 主指令：查看状态或管理审查开关。"""
        parts = (event.message_str or "").split()
        sub = parts[1].lower() if len(parts) > 1 else ""
        umo = event.unified_msg_origin
        self._refresh_ledger_params()

        if sub in ("on", "off"):
            self.ledger.set_toggle(umo, sub == "on")
            state = "开启" if sub == "on" else "关闭"
            yield event.plain_result(f"对抗人格守护已在当前会话{state}。")
            return
        if sub == "stats":
            text = f"会话奖惩统计：\n{self.ledger.stats_text(umo)}"
            fb = self.ledger.human_feedback_stats(umo)
            if fb.get("pos") or fb.get("neg"):
                text += f"\n人工反馈：正面 {fb['pos']} 次 / 负面 {fb['neg']} 次"
            fused = self.ledger.fused_until(umo)
            if fused > time.time():
                text += f"\n🔥 修复熔断中：剩余约 {int((fused - time.time()) / 60) + 1} 分钟"
            trend = self.ledger.daily_trend(umo, 7)
            max_checked = max((c for _, c, _ in trend), default=0)
            if max_checked > 0:
                lines = ["", "📈 近 7 天审查趋势（█ 审查 / ▲ 违规率）"]
                for label, checked, violated in trend:
                    bars = max(1, round(checked / max_checked * 15)) if checked else 0
                    rate = f"{violated / checked * 100:.0f}%" if checked else "-"
                    lines.append(
                        f"{label} {'█' * bars} {checked}次 违规率{rate}"
                    )
                text += "\n" + "\n".join(lines)
            yield event.plain_result(text)
            return
        if sub == "rules":
            rules = self.rules_store.get(umo)
            if not rules:
                yield event.plain_result(
                    "本会话暂无规则库。与 bot 对话一次后自动从人格提示词提取（rules_extract_enabled）。\n"
                    "无明文规则的人格可能提取为空，此时判官使用模糊对照。"
                )
                return
            lines = [f"本会话规则库（共 {len(rules)} 条，判官逐条对照）："]
            lines.extend(f"R{i}. {r}" for i, r in enumerate(rules, 1))
            yield event.plain_result("\n".join(lines))
            return
        if sub == "patch":
            if not _is_admin(event):
                yield event.plain_result("该操作需要管理员权限。")
                return
            async for r in self._run_patch(event, umo):
                yield r
            return
        if sub == "memory":
            self._refresh_memory_params()
            entries = self.memory.entries(umo)
            if not entries:
                yield event.plain_result("本会话暂无长期记忆条目。违规经总结与复审后会自动积累。")
                return
            ranked = sorted(
                entries,
                key=lambda e: (-int(e.get("count", 1)), -float(e.get("ts", 0))),
            )
            lines = [f"本会话长期记忆（共 {len(ranked)} 条，注入时按次数取前 8 条）："]
            for i, e in enumerate(ranked, 1):
                corrected = "（复审修正版）" if (e.get("meta") or {}).get("corrected") else ""
                lines.append(
                    f"{i}. {e.get('rule', '')} ×{e.get('count', 1)}{corrected}\n"
                    f"   └ {e.get('lesson', '')}"
                )
            yield event.plain_result("\n".join(lines))
            return
        if sub == "forget":
            if not _is_admin(event):
                yield event.plain_result("该操作需要管理员权限。")
                return
            arg = parts[2] if len(parts) > 2 else ""
            if arg == "all":
                self.memory.clear(umo)
                yield event.plain_result("已清空本会话的全部长期记忆。")
                return
            if arg.isdigit():
                entries = sorted(
                    self.memory.entries(umo),
                    key=lambda e: (-int(e.get("count", 1)), -float(e.get("ts", 0))),
                )
                idx = int(arg)
                if idx < 1 or idx > len(entries):
                    yield event.plain_result(f"序号超出范围（1-{len(entries)}），可先发送 /guard memory 查看。")
                    return
                target = entries[idx - 1]
                # remove 按 entries 存储顺序索引；此处按展示序号定位到存储序号
                store_idx = self.memory.entries(umo).index(target) + 1
                self.memory.remove(umo, store_idx)
                yield event.plain_result(f"已删除长期记忆：{target.get('rule', '')}")
                return
            yield event.plain_result("用法：/guard forget <序号|all>（序号见 /guard memory）")
            return
        if sub == "reset":
            if not _is_admin(event):
                yield event.plain_result("该操作需要管理员权限。")
                return
            self.ledger.reset(umo)
            yield event.plain_result("已清空本会话的奖惩账本。")
            return
        if sub == "test":
            text = (event.message_str or "").split(maxsplit=2)
            target = text[2].strip() if len(text) > 2 else ""
            if not target:
                yield event.plain_result("用法：/guard test <待审查文本>")
                return
            async for r in self._run_test(event, target):
                yield r
            return

        toggle = self.ledger.get_toggle(umo)
        state = "全局默认" if toggle is None else ("开启" if toggle else "关闭")
        status = (
            f"{HELP_TEXT}\n\n"
            f"当前状态：{'启用' if self._enabled_for(event) else '停用'}（会话级：{state}）\n"
            f"策略：{self._action()}｜阈值：{self._threshold()}｜重生成上限：{max(0, self._iget('max_regen_rounds', 2, 0))}\n"
            f"判官模型：{str(self.config.get('judge_model') or '').strip() or '跟随主AI'}"
        )
        yield event.plain_result(status)

    async def _run_patch(self, event: AstrMessageEvent, umo: str):
        """v1.2.0 人格补丁建议：基于高频违规 + 长期记忆，由 C 模型生成建议追加的人格条款。

        治本思路：把规则写明确比事后修复更有效。输出建议文本供管理员自行合并到人格提示词。
        """
        persona = self._last_persona.get(umo, "")
        rule_counts = {}
        try:
            s = self.ledger._data.get("sessions", {}).get(umo) or {}
            rule_counts = dict(s.get("rule_counts") or {})
        except Exception:
            pass
        memory_entries = self.memory.entries(umo)
        if not rule_counts and not memory_entries:
            yield event.plain_result("暂无违规记录，无需生成人格补丁。")
            return
        ranked = sorted(rule_counts.items(), key=lambda kv: -kv[1])[:8]
        lessons = [f"- {e.get('lesson')}" for e in memory_entries[:6] if e.get("lesson")]
        payload = {
            "persona": persona[:4000],
            "top_violations": [{"rule": r, "count": c} for r, c in ranked],
            "memory_lessons": lessons,
        }
        system = (
            "你是人格提示词顾问。给定一段 AI 人格提示词与该 AI 的高频违规记录，"
            "请生成 3~6 条「建议追加到人格提示词末尾的补充条款」，用于从源头杜绝这些违规。"
            "要求：每条一句话、明确可判定（禁止/必须句式）、与已有条款不重复；"
            "只输出一个 JSON 对象：{\"patches\": [\"条款1\", ...]}，禁止 markdown 代码块与额外文字。"
        )
        prompt = json.dumps(payload, ensure_ascii=False)
        try:
            raw = await asyncio.wait_for(
                self._chat_text_with(event, "summarizer", system, prompt),
                timeout=self._timeout(),
            )
        except Exception as e:
            yield event.plain_result(f"补丁生成失败（模型调用异常）：{e}")
            return
        data = None
        try:
            text = re.sub(r"^```[a-zA-Z]*\s*", "", raw.strip())
            text = re.sub(r"\s*```$", "", text).strip()
            start, end = text.find("{"), text.rfind("}")
            data = json.loads(text[start : end + 1]) if start != -1 and end > start else None
        except Exception:
            data = None
        patches = data.get("patches") if isinstance(data, dict) else None
        if not isinstance(patches, list) or not patches:
            yield event.plain_result("未能生成有效补丁建议，请稍后重试。")
            return
        lines = ["🩹 人格补丁建议（追加到人格提示词末尾）：", ""]
        for i, p in enumerate(patches[:6], 1):
            lines.append(f"{i}. {str(p).strip()}")
        lines.append("")
        lines.append("ℹ️ 复制以上条款合并到人格提示词保存后，规则库将自动重新提取。")
        yield event.plain_result("\n".join(lines))

    async def _run_test(self, event: AstrMessageEvent, target: str):
        """用手动输入的文本模拟一次判官审查（以本会话最近的人格提示词为基准）。"""
        persona = self._last_persona.get(event.unified_msg_origin, "")
        if not persona:
            yield event.plain_result(
                "本会话还没有 LLM 请求记录，判官无人格提示词可对照。请先与机器人对话一次。"
            )
            return
        verdict = await self._judge_once(event, persona, "", "(手动测试)", target)
        if verdict is None:
            yield event.plain_result("判官调用失败或输出无法解析，请检查日志。")
            return
        passed = verdict.compliant and verdict.score >= self._threshold()
        lines = [
            f"裁决：{'【合规·通过】' if passed else '【违规】'}｜分数：{verdict.score}",
        ]
        if verdict.violations:
            lines.append(format_violations(verdict.violations))
        if verdict.rewrite:
            lines.append(f"建议改写：\n{verdict.rewrite}")
        yield event.plain_result("\n".join(lines))

    async def terminate(self):
        """插件卸载/停用时等待后台总结任务收尾并持久化账本与长期记忆。"""
        try:
            tasks = [t for t in self._summary_tasks if not t.done()]
            if tasks:
                await asyncio.wait(tasks, timeout=10)
        except Exception:
            pass
        try:
            self.ledger.save()
        except Exception:
            pass
        try:
            self.memory.save()
        except Exception:
            pass
