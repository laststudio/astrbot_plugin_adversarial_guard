"""对抗修复循环：快改（fast_fix）→ 重生成降级（regen）→ 择优放行。

依赖注入设计：本模块不直接接触 AstrBot API，
chat_fn / judge_fn 由 main.py 提供，便于独立测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .judge import Verdict, format_violations

ACTION_FAST_FIX_THEN_REGEN = "fast_fix_then_regen"
ACTION_REGEN_ONLY = "regen_only"
ACTION_RECORD_ONLY = "record_only"


@dataclass
class RepairResult:
    final_text: str
    final_verdict: Verdict | None
    ok: bool  # 最终文本是否通过判官
    method: str  # original / fast_fix / regen / best_effort
    attempts: list = field(default_factory=list)  # [{source, score, ok}]


def build_regen_prompt(user_msg: str, violations: list, round_no: int, max_rounds: int) -> str:
    """构建带驳回通知的重生成提示词（追加在原始用户消息之后）。"""
    block = (
        f"[系统驳回通知·第{round_no}/{max_rounds}次]\n"
        "你上一次的回复已被合规判官A驳回，原因如下：\n"
        f"{format_violations(violations)}\n\n"
        "请重新生成本轮回复，要求：\n"
        "1. 逐条修复上述全部违规；\n"
        "2. 完整保持人格设定（语气、身份、格式规则）与回答内容的有用性；\n"
        "3. 直接输出新回复正文，禁止解释修改过程，禁止提及本驳回通知。"
    )
    base = (user_msg or "").strip() or "（用户消息为空）"
    return f"{base}\n\n{block}"


async def run_adversarial_repair(
    chat_fn,
    judge_fn,
    *,
    action: str,
    persona: str,
    contexts: list,
    user_msg: str,
    original_reply: str,
    first_verdict: Verdict,
    threshold: int,
    max_regen_rounds: int,
) -> RepairResult:
    """执行修复策略，返回最终应采用的文本。

    chat_fn(prompt: str) -> str          调用主 AI(B)，内部已带上原 system_prompt 与上下文
    judge_fn(text: str) -> Verdict|None  调用判官 A 审查任意文本
    """
    candidates: list[tuple[int, str, Verdict | None]] = []  # (priority, text, verdict)

    if action == ACTION_RECORD_ONLY:
        return RepairResult(original_reply, first_verdict, False, "original", [])

    # ---------- 第 1 步：快改（采用判官给出的 rewrite 并复检） ----------
    if action == ACTION_FAST_FIX_THEN_REGEN:
        rewrite = (first_verdict.rewrite or "").strip()
        if rewrite:
            v = await judge_fn(rewrite)
            passed = v is not None and (v.compliant and v.score >= threshold)
            candidates.append((1, rewrite, v))
            attempts = [{"source": "fast_fix", "score": v.score if v else None, "ok": passed}]
            if passed:
                return RepairResult(rewrite, v, True, "fast_fix", attempts)
            # 快改未通过 → 降级重生成
            last_violations = v.violations if v else first_verdict.violations
        else:
            last_violations = first_verdict.violations
            attempts = []
    else:
        last_violations = first_verdict.violations
        attempts = []

    # ---------- 第 2 步：重生成对抗循环 ----------
    best: tuple[int, str, Verdict | None] | None = None
    for i in range(1, max(0, int(max_regen_rounds)) + 1):
        prompt = build_regen_prompt(
            user_msg, last_violations or first_verdict.violations, i, max_regen_rounds
        )
        try:
            reply = (await chat_fn(prompt)).strip()
        except Exception:
            break  # 生成失败则终止循环，走择优兜底
        if not reply:
            continue
        v = await judge_fn(reply)
        passed = v is not None and (v.compliant and v.score >= threshold)
        cand = (0, reply, v)  # 重生成结果优先级高于快改
        candidates.append(cand)
        attempts.append({"source": f"regen_{i}", "score": v.score if v else None, "ok": passed})
        if passed:
            return RepairResult(reply, v, True, "regen", attempts)
        if v is not None and (best is None or best[2] is None or v.score > best[2].score):
            best = cand
        if v is not None and v.violations:
            last_violations = v.violations

    # ---------- 第 3 步：择优兜底 ----------
    pool = [c for c in candidates if c[1]]
    if not pool:
        return RepairResult(original_reply, first_verdict, False, "original", attempts)

    def sort_key(c):
        prio, _text, verdict = c
        score = verdict.score if verdict else -1
        return (-score, prio)

    top = min(pool, key=sort_key)
    top_score = top[2].score if top[2] else -1
    orig_score = first_verdict.score
    # 原回复分数更高时保留原文（修复反而变差的情形）
    if orig_score > top_score + 5:
        return RepairResult(original_reply, first_verdict, False, "original", attempts)
    method = "best_effort"
    return RepairResult(top[1], top[2], False, method, attempts)
