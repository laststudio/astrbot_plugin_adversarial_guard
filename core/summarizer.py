"""总结模型(C) 与 判官复审(A) 的提示词构建与解析（v1.1.0 新增）。

流程：审查/修复完成后，C 把本轮违规与修复结果浓缩为一条「教训」；
随后 A 从 忠实性 / 简洁性 / 隐私过滤 三个维度复审该教训，
通过（或给出修正版）后由 main.py 写入长期记忆。

所有解析均带容错（剥离代码围栏、截取花括号段），解析失败返回 None，
由上层按 fail-open 原则丢弃本轮记忆，绝不影响正常聊天。
"""

from __future__ import annotations

import json
import re

MAX_INPUT_CHARS = 3000

# ------------------------------------------------------------------ C：总结
SUMMARY_SYSTEM_PROMPT = """你是"记忆管理员C"，负责把 AI 对话合规系统的审查结果浓缩为可长期保存的行为教训。

你将收到一段 JSON 输入，包含：
- verdict：判官A对本轮 AI 回复的裁决（是否合规、分数、违规列表）
- repair：修复处理结果（method=fast_fix/regen/best_effort/original，ok=是否通过复审）
- reward：当前纪律分

你的任务：提炼一条值得长期记住的「教训」，帮助生成 AI 在后续对话中规避同类问题。

要求：
1. rule：用一句话概括被违反的人格规则本身（不是违规现象）。例如人格禁止 markdown 而回复用了加粗，rule 应为"禁止使用任何 markdown 语法"，而非"使用了加粗"。
2. lesson：一句可执行的规避指令（不超过 60 字），直接告诉生成 AI 该怎么做。例如："回复中不得出现 **加粗**、#标题、- 列表等任何 markdown 符号，用纯文本自然表述"。
3. keep：若本轮内容没有长期保存价值（例如违规过于琐碎、或属于一次性误判），设为 false；否则 true。
4. 只输出一个 JSON 对象，禁止 markdown 代码块，禁止任何额外文字。格式：
{"rule": "...", "lesson": "...", "keep": true}"""


def build_summary_prompt(verdict_dict: dict, repair_info: dict, reward: float) -> str:
    """组装送交总结模型(C)的用户消息。"""
    payload = {
        "verdict": verdict_dict,
        "repair": repair_info,
        "reward": reward,
    }
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS] + "...(截断)"
    return f"{text}\n\n请依照系统指令输出教训 JSON。"


def parse_summary(raw: str) -> dict | None:
    """解析 C 的输出：{"rule": str, "lesson": str, "keep": bool}。

    解析失败 / keep=false / 字段非法 返回 None（上层丢弃）。
    """
    data = _extract_json(raw)
    if not isinstance(data, dict):
        return None
    keep = bool(data.get("keep", True))
    if not keep:
        return None
    rule = str(data.get("rule") or "").strip()
    lesson = str(data.get("lesson") or "").strip()
    if not rule or not lesson:
        return None
    return {"rule": rule, "lesson": lesson, "keep": True}


# ------------------------------------------------------------------ A：复审
REVIEW_SYSTEM_PROMPT = """你是"判官A"兼任的记忆审计员。记忆管理员C 刚刚把一条「教训」草稿提交给你，准备写入生成 AI(B) 的长期记忆。你负责在写入前把好最后一道关。

你将收到：
[原始裁决] 判官A（你）本轮的真实裁决 JSON
[修复结果] 对抗修复的处理摘要
[待写入条目] C 提炼的教训草稿 {"rule", "lesson"}

请从三个维度审查：
1. 忠实性：教训必须严格来源于原始裁决中列出的违规，不得捏造、夸大或引入裁决中不存在的问题；rule 应对应人格规则本身而非违规现象。
2. 简洁性：lesson 必须是一句可执行的规避指令（≤60字），不得空话套话。
3. 隐私：不得包含用户 QQ 号、昵称、手机号、地址等任何个人敏感信息。

输出要求：只输出一个 JSON 对象，禁止 markdown 代码块，禁止额外文字。格式：
{"pass": true, "reason": "", "fixed_lesson": "", "fixed_rule": ""}
- pass=true 表示草稿可直接写入；
- pass=false 时：若问题是轻微不忠实或含隐私，请在 fixed_lesson/fixed_rule 给出修正后的版本（修正版将被采纳写入）；若草稿完全不可用（捏造事实），fixed_lesson 留空表示丢弃。"""


def build_review_prompt(verdict_dict: dict, repair_info: dict, draft: dict) -> str:
    """组装送交判官(A)复审的用户消息。"""
    payload = {
        "原始裁决": verdict_dict,
        "修复结果": repair_info,
        "待写入条目": draft,
    }
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS] + "...(截断)"
    return f"{text}\n\n请依照系统指令输出审计 JSON。"


def parse_review(raw: str) -> dict | None:
    """解析 A 的复审输出：{"pass", "reason", "fixed_lesson", "fixed_rule"}。

    解析失败返回 None（上层按丢弃处理）。
    """
    data = _extract_json(raw)
    if not isinstance(data, dict):
        return None
    passed = bool(data.get("pass"))
    reason = str(data.get("reason") or "").strip()
    fixed_lesson = str(data.get("fixed_lesson") or "").strip()
    fixed_rule = str(data.get("fixed_rule") or "").strip()
    return {
        "pass": passed,
        "reason": reason,
        "fixed_lesson": fixed_lesson,
        "fixed_rule": fixed_rule,
    }


def resolve_entry(draft: dict, review: dict | None) -> dict | None:
    """根据复审结论决定最终写入条目。

    - review 为 None（复审不可用/关闭）：信任草稿直接写入；
    - review.pass=True：写入草稿；
    - review.pass=False 且给出 fixed_lesson：采纳修正版（meta.corrected=True）；
    - review.pass=False 且无修正版：丢弃（返回 None）。
    """
    if review is None:
        return dict(draft)
    if review.get("pass"):
        return dict(draft)
    fixed_lesson = str(review.get("fixed_lesson") or "").strip()
    if not fixed_lesson:
        return None
    entry = dict(draft)
    entry["lesson"] = fixed_lesson
    fixed_rule = str(review.get("fixed_rule") or "").strip()
    if fixed_rule:
        entry["rule"] = fixed_rule
    entry["meta"] = dict(draft.get("meta") or {})
    entry["meta"]["corrected"] = True
    return entry


# ------------------------------------------------------------------ 内部
def _extract_json(raw) -> dict | None:
    """从模型输出中容错提取 JSON 对象（剥围栏、截花括号段）。"""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None
