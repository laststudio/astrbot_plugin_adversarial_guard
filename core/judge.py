"""判官 AI（A）模块。

负责构建审查提示词、解析判官返回的严格 JSON 裁决。
判官的输入 = 人格提示词 + 近几轮上下文 + 本轮用户消息 + B 的待审回复。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# 送审判官时人格提示词的最大长度（保护 token 开销）
MAX_PERSONA_CHARS = 8000
# 上下文历史中单条消息的最大长度
MAX_HISTORY_MSG_CHARS = 300

JUDGE_SYSTEM_PROMPT = """你是"判官A"，一个严格、客观、铁面无私的AI输出合规审查器。你的对手是另一个AI（"B"），它负责在给定的人格设定下回复用户。B 经常遗忘或无视人格设定中的规则，你的任务就是抓住它的每一次违规。

你将收到四部分内容：
[人格提示词] B 必须遵守的全部设定与规则
[近几轮对话] 提供语境
[本轮用户消息] 触发本次回复的输入
[待审回复] B 的输出，即你的审查对象

审查维度（逐条对照人格提示词）：
1. 格式规则：如禁止 markdown 语法（**加粗**、#标题、```代码块、- 列表等）、禁止表情符号、字数/句数限制、禁止分点等。此类明文格式规则必须严格执行，出现即违规。
2. 语气与风格：人设口吻、自称、称呼习惯、说话方式是否一致。
3. 内容边界：是否涉及人格设定禁止的话题、是否越界扮演其他身份、是否做出无法兑现的现实承诺。
4. 指令遵循：人格提示词中任何明确的"要求做/不得做"事项。

评分标准（0-100 整数）：
- 95-100：完全合规，无任何瑕疵
- 80-94：基本合规，仅有极轻微的可忽略偏差
- 60-79：存在明显问题（如轻微语气偏离、少量多余符号）
- 0-59：严重违规（明确违反明文规则、格式完全不符、越界等）

裁决规则：
- compliant 仅在回复没有任何违反人格提示词明文规则之处时为 true；只要存在任一明文规则被违反，无论多小都必须为 false。
- 每条违规给出 rule（对应人格中的规则原文摘要）、evidence（回复中的原句证据）、severity（1-5，5最严重）、suggestion（修复建议）。
- rewrite 字段：当 compliant=false 时，给出修复了全部违规、保持原意与人设口吻、可直接发送给用户的完整替换文本；compliant=true 时留空字符串。
- 你只做审查与改写，不回答用户的问题本身。

输出要求：只输出一个 JSON 对象，禁止使用任何 markdown 代码块标记，禁止输出 JSON 以外的任何文字。格式：
{"compliant": false, "score": 42, "violations": [{"rule": "...", "evidence": "...", "severity": 3, "suggestion": "..."}], "rewrite": "..."}"""


@dataclass
class Verdict:
    """一次审查的裁决结果。"""

    compliant: bool = True
    score: int = 100
    violations: list = field(default_factory=list)
    rewrite: str = ""


def build_judge_user_prompt(
    persona: str,
    history_block: str,
    user_msg: str,
    reply: str,
) -> str:
    """组装送审判官的用户消息。"""
    persona = (persona or "").strip()[:MAX_PERSONA_CHARS] or "（未提供人格提示词）"
    history_block = history_block.strip() or "（无历史记录）"
    user_msg = (user_msg or "").strip()[:MAX_PERSONA_CHARS] or "（空）"
    parts = [
        "[人格提示词]",
        persona,
        "",
        "[近几轮对话]",
        history_block,
        "",
        "[本轮用户消息]",
        user_msg,
        "",
        "[待审回复]",
        reply,
        "",
        "请依照系统指令输出裁决 JSON。",
    ]
    return "\n".join(parts)


def parse_verdict(raw: str) -> Verdict | None:
    """解析判官输出的 JSON，带容错（剥离代码围栏、截取花括号段）。

    解析失败返回 None，由上层决定放行。
    """
    if not raw:
        return None
    text = raw.strip()
    # 剥离可能的 ```json 围栏
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
    if not isinstance(data, dict):
        return None

    score = _to_int(data.get("score"), default=100)
    score = max(0, min(100, score))
    violations = _normalize_violations(data.get("violations"))
    rewrite = data.get("rewrite")
    rewrite = rewrite.strip() if isinstance(rewrite, str) else ""
    compliant = bool(data.get("compliant"))
    # 自洽性兜底：判官列出了违规条目时，以违规为准（与判官指令"存在明文违规即 false"一致）
    if violations and compliant:
        compliant = False
    if not violations and not compliant and score >= 90:
        compliant = True
    return Verdict(compliant=compliant, score=score, violations=violations, rewrite=rewrite)


def is_pass(verdict: Verdict, threshold: int) -> bool:
    """综合判定是否通过：需同时满足判官裁定合规且分数不低于阈值。"""
    return verdict.compliant and verdict.score >= threshold


def format_violations(violations: list, prefix: str = "- ") -> str:
    """将违规列表渲染为人类可读的多行文本。"""
    lines = []
    for v in violations:
        rule = v.get("rule", "未知规则")
        evidence = v.get("evidence", "")
        severity = v.get("severity", 1)
        line = f"{prefix}违反「{rule}」"
        if evidence:
            ev = evidence if len(evidence) <= 60 else evidence[:57] + "..."
            line += f"：证据「{ev}」"
        line += f"（严重度{severity}）"
        lines.append(line)
    return "\n".join(lines)


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_violations(raw) -> list:
    """规范化违规条目结构，过滤非法字段。"""
    result = []
    if not isinstance(raw, list):
        return result
    for item in raw[:20]:
        if not isinstance(item, dict):
            continue
        sev = _to_int(item.get("severity"), default=1)
        result.append(
            {
                "rule": str(item.get("rule") or "未说明规则")[:200],
                "evidence": str(item.get("evidence") or "")[:500],
                "severity": max(1, min(5, sev)),
                "suggestion": str(item.get("suggestion") or "")[:500],
            }
        )
    return result


def build_history_block(contexts: list, exclude_prompt: str, max_turns: int) -> str:
    """把会话上下文渲染为判官可读的历史文本块。

    contexts 元素形如 {"role": "user"/"assistant", "content": str 或部件列表}。
    若末尾恰好是本轮用户消息（与 exclude_prompt 相同），则剔除避免重复。
    """
    msgs = []
    for m in contexts or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        if role not in ("user", "assistant"):
            continue
        content = _content_to_text(m.get("content"))
        if not content:
            continue
        msgs.append((role, content))

    if exclude_prompt:
        while msgs and msgs[-1][0] == "user" and msgs[-1][1].strip() == exclude_prompt.strip():
            msgs.pop()

    tail = msgs[-max_turns:] if max_turns > 0 else msgs
    lines = []
    for role, content in tail:
        who = "用户" if role == "user" else "AI"
        c = (
            content
            if len(content) <= MAX_HISTORY_MSG_CHARS
            else content[: MAX_HISTORY_MSG_CHARS - 3] + "..."
        )
        lines.append(f"{who}: {c}")
    return "\n".join(lines)


def _content_to_text(content) -> str:
    """兼容 content 为纯字符串或 OpenAI 风格部件列表两种形态。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "\n".join(t for t in texts if t).strip()
    return ""
