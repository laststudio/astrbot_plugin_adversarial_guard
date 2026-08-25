"""奖惩账本（RL 模拟核心）。

以会话（unified_msg_origin）为键，持久化记录：
- 纪律分 reward：带衰减的滚动奖惩分
- recent：最近 N 次审查结果快照（用于反馈注入窗口）
- rule_counts：各违规规则累计次数
- 各类计数器

数据存储于 AstrBot data/plugin_data 目录，避免插件更新时被覆盖。
"""

from __future__ import annotations

import json
import os
import threading
import time

HISTORY_CAP = 50  # 每会话保留的审查历史条数上限
RECENT_CAP = 20  # 反馈窗口原始队列容量（取尾部 window 条参与统计）
MAX_RULES_IN_INJECT = 6  # 注入文本中最多列出的违规规则条数


class Ledger:
    def __init__(self, path: str, decay: float = 0.8, window: int = 5):
        self.path = path
        self.decay = min(1.0, max(0.0, float(decay)))
        self.window = max(1, int(window))
        self._lock = threading.Lock()
        self._data = {"sessions": {}}
        self.load()

    # ------------------------------------------------------------------ 基础
    def _session(self, umo: str) -> dict:
        sessions = self._data.setdefault("sessions", {})
        s = sessions.get(umo)
        if s is None:
            s = {
                "reward": 0.0,
                "recent": [],  # [{"t": ts, "ok": bool, "rules": [..], "score": int}]
                "rule_counts": {},  # rule -> count
                "total_checked": 0,
                "total_violated": 0,
                "total_fixed": 0,
                "total_regen_success": 0,
            }
            sessions[umo] = s
        return s

    def set_params(self, decay: float, window: int) -> None:
        self.decay = min(1.0, max(0.0, float(decay)))
        self.window = max(1, int(window))

    # ------------------------------------------------------------------ 记账
    def record_check(self, umo: str, ok: bool, score: int, violations: list) -> float:
        """记录一次审查结果。违规按 severity 合计扣分；合规按分数盈余加分。

        返回本次变动 delta。
        """
        total_sev = sum(int(v.get("severity", 1)) for v in violations)
        if ok:
            delta = round(max(0.0, (int(score) - 70) / 10), 2)
        else:
            delta = -max(1, min(10, total_sev if total_sev > 0 else 1))
        rules = []
        for v in violations or []:
            r = str(v.get("rule") or "").strip()
            if r and r not in rules:
                rules.append(r)

        with self._lock:
            s = self._session(umo)
            s["reward"] = round(s["reward"] * self.decay + delta, 2)
            s["total_checked"] += 1
            if not ok:
                s["total_violated"] += 1
                for r in rules:
                    s["rule_counts"][r] = s["rule_counts"].get(r, 0) + 1
            s["recent"].append({"t": time.time(), "ok": ok, "rules": rules, "score": int(score)})
            del s["recent"][:-RECENT_CAP]
            self.save()
        return delta

    def record_repair(self, umo: str, success: bool, method: str) -> None:
        """记录一次修复尝试的最终结果（不重复计分，仅统计与更新最近快照）。"""
        with self._lock:
            s = self._session(umo)
            if success:
                s["total_fixed"] += 1
                if method == "regen":
                    s["total_regen_success"] += 1
                # 修复成功给予小幅奖励，鼓励对抗循环收敛
                s["reward"] = round(s["reward"] + 1.0, 2)
            if s["recent"]:
                s["recent"][-1]["fixed"] = success
                s["recent"][-1]["method"] = method
            hist_cap_guard(s)
            self.save()

    # ------------------------------------------------------------- 反馈注入
    def injection_text(self, umo: str, threshold: int) -> str | None:
        """生成注入后续请求的 RL 反馈块；无近期违规时返回 None。"""
        with self._lock:
            s = self._session(umo)
            tail = s["recent"][-self.window :]
            counts = dict(s.get("rule_counts", {}))
            reward = s["reward"]
        bad_rules: list[str] = []
        for item in tail:
            for r in item.get("rules", []):
                if r not in bad_rules:
                    bad_rules.append(r)
        if not bad_rules:
            return None
        counts = {r: c for r, c in counts.items() if r in bad_rules}
        lines = [
            f"- {r}（累计出现 {counts.get(r, 1)} 次）" for r in bad_rules[:MAX_RULES_IN_INJECT]
        ]
        tone = "你的纪律分为正，请继续保持" if reward >= 0 else "你的纪律分为负，必须在本轮纠正"
        return (
            "<guard_feedback>\n"
            "[奖惩系统·判官A给你的反馈]\n"
            "你在最近几轮对话中出现过以下违反人格设定的行为：\n" + "\n".join(lines) + "\n"
            f"当前纪律分：{reward}（越高越合规）。{tone}，逐条遵守人格提示词中的全部规则。\n"
            "</guard_feedback>"
        )

    # ---------------------------------------------------------------- 查询
    def stats_text(self, umo: str, top_n: int = 5) -> str:
        with self._lock:
            s = self._session(umo)
            reward = s["reward"]
            checked = s["total_checked"]
            violated = s["total_violated"]
            fixed = s["total_fixed"]
            regen_ok = s["total_regen_success"]
            rule_counts = dict(s.get("rule_counts", {}))
        rate = f"{violated / checked * 100:.1f}%" if checked else "N/A"
        lines = [
            f"纪律分：{reward}",
            f"累计审查：{checked} 次｜违规：{violated} 次（{rate}）",
            f"修复成功：{fixed} 次（其中重生成收敛 {regen_ok} 次）",
        ]
        if rule_counts:
            lines.append(f"高频违规 Top{top_n}：")
            ranked = sorted(rule_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
            lines.extend(f"- {r} ×{c}" for r, c in ranked)
        return "\n".join(lines)

    def reset(self, umo: str) -> None:
        with self._lock:
            self._data.get("sessions", {}).pop(umo, None)
            self.save()

    # ------------------------------------------------------------ 会话开关
    def set_toggle(self, umo: str, enabled: bool) -> None:
        with self._lock:
            toggles = self._data.setdefault("toggles", {})
            toggles[umo] = bool(enabled)
            self.save()

    def get_toggle(self, umo: str) -> bool | None:
        """返回会话级开关状态；None 表示未设置（跟随全局）。"""
        with self._lock:
            return self._data.get("toggles", {}).get(umo)

    # ------------------------------------------------------------ 持久化
    def save(self) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as e:
            import logging

            logging.getLogger("astrbot").warning(f"[adversarial_guard] 账本保存失败: {e}")

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._data = data
                self._data.setdefault("sessions", {})
                self._data.setdefault("toggles", {})
        except (OSError, json.JSONDecodeError) as e:
            import logging

            logging.getLogger("astrbot").warning(f"[adversarial_guard] 账本加载失败，已重置: {e}")


def hist_cap_guard(s: dict) -> None:
    """控制 recent 队列长度（record 内已裁剪，这里兜底防御异常增长）。"""
    if len(s.get("recent", [])) > RECENT_CAP * 4:
        del s["recent"][:-RECENT_CAP]
