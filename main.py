"""AstrBot 对抗人格守护插件（GAN 式双 AI 对抗）。

架构：
- B（生成器）= AstrBot 主 AI，负责真实对话；
- A（判别器）= 判官 AI，读取人格提示词，审查 B 的每次输出是否遵守规则；
- 违规时执行 快改→重生成 的对抗修复循环，并以奖惩账本 +
  上下文反馈注入模拟强化学习，让 B 在后续对话中持续纠偏。
"""

from __future__ import annotations

import asyncio
import os
import time

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
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
    "对抗人格守护（判官A × 主AI B）\n"
    "/guard - 查看状态与帮助\n"
    "/guard on|off - 本会话开关审查\n"
    "/guard stats - 查看奖惩统计\n"
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
            val = int(self.config.get(key) or default)
        except (TypeError, ValueError):
            val = int(default)
        if lo is not None:
            val = max(lo, val)
        if hi is not None:
            val = min(hi, val)
        return val

    def _fget(self, key: str, default: float) -> float:
        try:
            return float(self.config.get(key) or default)
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

            if not bool(self.config.get("feedback_injection", True)):
                return
            feedback = self.ledger.injection_text(umo, self._threshold())
            if not feedback:
                return
            parts = getattr(req, "extra_user_content_parts", None)
            if parts is not None and TextPart is not None:
                parts.append(TextPart(text=feedback).mark_as_temp())
            elif persona is not None:
                # 旧版本兜底：追加到 system_prompt（可能影响提示词缓存）
                req.system_prompt = persona + "\n" + feedback
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] on_llm_request 异常(已忽略): {e}")

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
            history_block = build_history_block(
                snap["contexts"],
                exclude_prompt=user_msg,
                max_turns=max(0, self._iget("history_turns_for_judge", 6, 0)),
            )

            verdict = await self._judge_once(event, persona, history_block, user_msg, reply)
            if verdict is None:
                logger.warning(f"[{PLUGIN_NAME}] 判官不可用或输出无法解析，放行原文")
                return

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

            result = await run_adversarial_repair(
                chat_fn=self._make_chat_fn(event, snap),
                judge_fn=lambda text: self._judge_once(
                    event, persona, history_block, user_msg, text
                ),
                action=self._action(),
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
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] on_llm_response 异常(已忽略): {e}")

    # =========================================================== 内部流程
    async def _judge_once(
        self,
        event: AstrMessageEvent,
        persona: str,
        history_block: str,
        user_msg: str,
        reply: str,
    ) -> Verdict | None:
        """调用判官 A 审查一段回复；任何失败返回 None。"""
        provider = await self._get_judge_provider(event)
        if provider is None:
            return None
        limit = self._max_judge_chars()
        judged_reply = reply if len(reply) <= limit else reply[:limit] + "...(截断)"
        user_prompt = build_judge_user_prompt(persona, history_block, user_msg, judged_reply)
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
            yield event.plain_result(f"会话奖惩统计：\n{self.ledger.stats_text(umo)}")
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
        """插件卸载/停用时持久化账本。"""
        try:
            self.ledger.save()
        except Exception:
            pass
