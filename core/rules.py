"""显式规则库（v1.2.0 新增）。

把人格提示词中的模糊规则用 LLM 一次性提取为编号清单（R1/R2/...），
判官审查时逐条对照打勾，violations.rule 引用规则编号，显著提升判官准确率。

- 按会话隔离存储（rules.json），记录 persona 哈希用于变化检测；
- persona 变化（哈希不同）→ 视为过期，由 main.py 触发后台重新提取；
- 原子写盘、损坏重置，与 ledger/memory 风格一致。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time

MAX_RULES = 30
MAX_RULE_CHARS = 150

# 提取规则的提示词（送总结/判官类便宜模型即可）
EXTRACT_SYSTEM_PROMPT = """你是"规则管理员"。给你一段 AI 的人格提示词，请把其中所有**明确的、可判定遵守与否**的要求提取为编号规则清单。

要求：
1. 只提取明文规则（格式要求、语气要求、内容边界、必须做/不得做的事项），忽略纯背景设定与性格描述；
2. 每条规则一句话、独立可判定（判官能据此直接判断一条回复是否违反）；
3. 最多 30 条；没有可提取的明文规则时输出空数组；
4. 只输出一个 JSON 对象，禁止 markdown 代码块，禁止额外文字。格式：
{"rules": ["规则1", "规则2", ...]}"""


def persona_hash(persona: str) -> str:
    return hashlib.md5((persona or "").encode("utf-8")).hexdigest()[:12]


def build_extract_prompt(persona: str) -> str:
    p = (persona or "").strip()[:8000] or "（空）"
    return f"[人格提示词]\n{p}\n\n请依照系统指令输出规则清单 JSON。"


def parse_rules(raw: str) -> list[str]:
    """容错解析提取输出，返回规则文本列表（0~30 条）。"""
    if not raw:
        return []
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    rules_raw = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(rules_raw, list):
        return []
    out: list[str] = []
    for r in rules_raw[:MAX_RULES]:
        t = str(r or "").strip()[:MAX_RULE_CHARS]
        if t and t not in out:
            out.append(t)
    return out


def render_rules_block(rules: list[str]) -> str:
    """渲染为送审判官的规则清单块（R1/R2/... 编号）。"""
    if not rules:
        return ""
    lines = ["[规则清单] 审查时逐条对照下列编号规则，violations.rule 请引用编号与规则摘要："]
    for i, r in enumerate(rules, 1):
        lines.append(f"R{i}. {r}")
    return "\n".join(lines)


def rule_id_of(rules: list[str], rule_text: str) -> str:
    """根据判官返回的 violations.rule 文本模糊匹配规则编号（如 R3）。"""
    t = (rule_text or "").strip()
    if not t:
        return ""
    m = re.match(r"^R(\d+)", t)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= len(rules):
            return f"R{idx}"
    # 模糊包含匹配
    for i, r in enumerate(rules, 1):
        if r in t or t in r:
            return f"R{i}"
    return ""


class RulesStore:
    """按会话保存规则清单与 persona 哈希。"""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict = {"sessions": {}}
        self.load()

    def _session(self, umo: str) -> dict:
        sessions = self._data.setdefault("sessions", {})
        s = sessions.get(umo)
        if s is None:
            s = {"rules": [], "persona_hash": "", "updated_at": 0.0}
            sessions[umo] = s
        return s

    def get(self, umo: str) -> list[str]:
        with self._lock:
            s = self._data.get("sessions", {}).get(umo)
            if not s:
                return []
            return [str(r) for r in s.get("rules", [])]

    def is_stale(self, umo: str, persona: str) -> bool:
        """规则库不存在或 persona 已变化时返回 True（需要重新提取）。"""
        with self._lock:
            s = self._data.get("sessions", {}).get(umo)
        if not s or not s.get("rules"):
            return True
        return str(s.get("persona_hash", "")) != persona_hash(persona)

    def set_rules(self, umo: str, persona: str, rules: list[str]) -> list[str]:
        with self._lock:
            s = self._session(umo)
            s["rules"] = [str(r)[:MAX_RULE_CHARS] for r in rules[:MAX_RULES]]
            s["persona_hash"] = persona_hash(persona)
            s["updated_at"] = time.time()
            self.save()
        return s["rules"]

    def clear(self, umo: str) -> None:
        with self._lock:
            self._data.get("sessions", {}).pop(umo, None)
            self.save()

    def save(self) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as e:
            import logging

            logging.getLogger("astrbot").warning(
                f"[adversarial_guard] 规则库保存失败: {e}"
            )

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._data = data
                self._data.setdefault("sessions", {})
        except (OSError, json.JSONDecodeError) as e:
            import logging

            logging.getLogger("astrbot").warning(
                f"[adversarial_guard] 规则库加载失败，已重置: {e}"
            )
            self._data = {"sessions": {}}
