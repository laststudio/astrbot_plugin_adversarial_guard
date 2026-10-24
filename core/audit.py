"""违规审计日志（v1.3.0 新增）。

以 JSONL（每行一条 JSON）追加每次违规的完整裁决与修复过程，
供事后审计与误判追溯。文件超过 max_bytes 时滚动为 .1 备份。

设计要点：
- 追加写 + 每行独立 JSON，进程崩溃最多损坏最后一行（读取时自动跳过）；
- tail() 反向读取最近 N 条，仅在内存中保留必要字段，控制开销；
- enabled=False 时全部操作为空操作（零开销）。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

logger = logging.getLogger("astrbot")

# 单条记录中各文本字段的最大长度
MAX_TEXT_CHARS = 300


def _clip(text, limit: int = MAX_TEXT_CHARS) -> str:
    t = str(text or "").strip()
    return t if len(t) <= limit else t[: limit - 3] + "..."


class AuditLog:
    """违规审计日志：JSONL 追加 + 滚动 + 尾部读取。"""

    def __init__(self, path: str, max_bytes: int = 1024 * 1024, enabled: bool = True):
        self.path = path
        self.max_bytes = max(64 * 1024, int(max_bytes))
        self.enabled = bool(enabled)
        self._lock = threading.Lock()

    def set_params(self, max_bytes: int, enabled: bool) -> None:
        self.max_bytes = max(64 * 1024, int(max_bytes))
        self.enabled = bool(enabled)

    # ---------------------------------------------------------------- 写入
    def record(
        self,
        umo: str,
        *,
        score: int,
        ok: bool,
        violations: list,
        repair: dict | None = None,
        reply: str = "",
        user_msg: str = "",
    ) -> None:
        """追加一条违规审计记录（仅在违规 ok=False 时由上层调用）。"""
        if not self.enabled:
            return
        entry = {
            "ts": round(time.time(), 2),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "umo": str(umo),
            "score": int(score),
            "violations": [
                {
                    "rule": _clip(v.get("rule", ""), 200),
                    "evidence": _clip(v.get("evidence", ""), 300),
                    "severity": int(v.get("severity", 1) or 1),
                }
                for v in (violations or [])[:10]
                if isinstance(v, dict)
            ],
            "repair": {
                "method": str((repair or {}).get("method", "")),
                "ok": bool((repair or {}).get("ok", False)),
                "attempts": int((repair or {}).get("attempts", 0) or 0),
            },
            "reply": _clip(reply),
            "user_msg": _clip(user_msg),
        }
        with self._lock:
            try:
                self._rotate_if_needed()
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError as e:
                logger.warning(f"[adversarial_guard] 审计日志写入失败: {e}")

    def _rotate_if_needed(self) -> None:
        if not os.path.exists(self.path):
            return
        if os.path.getsize(self.path) < self.max_bytes:
            return
        backup = self.path + ".1"
        try:
            if os.path.exists(backup):
                os.remove(backup)
            os.replace(self.path, backup)
        except OSError:
            pass  # 滚动失败不阻塞写入（继续追加原文件）

    # ---------------------------------------------------------------- 读取
    def tail(self, n: int = 20, umo: str | None = None) -> list[dict]:
        """读取最近 n 条记录（可按会话过滤）。文件不存在返回空列表。"""
        if not self.enabled or not os.path.exists(self.path):
            return []
        n = max(1, min(100, int(n)))
        out: list[dict] = []
        try:
            with open(self.path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue  # 跳过损坏行
            if not isinstance(entry, dict):
                continue
            if umo is not None and entry.get("umo") != umo:
                continue
            out.append(entry)
            if len(out) >= n:
                break
        return out

    def render(self, entries: list[dict]) -> str:
        """把 tail() 结果渲染为人类可读文本。"""
        if not entries:
            return "暂无审计记录。"
        lines = [f"最近 {len(entries)} 条违规审计："]
        for e in entries:
            vio = e.get("violations") or []
            first_rule = vio[0].get("rule", "?") if vio else "?"
            more = f" 等{len(vio)}项" if len(vio) > 1 else ""
            repair = e.get("repair") or {}
            flag = "✅" if repair.get("ok") else "❌"
            lines.append(
                f"[{e.get('time', '?')}] {e.get('umo', '?')}\n"
                f"  分数{e.get('score', '?')}｜违反「{first_rule}」{more}｜"
                f"修复 {repair.get('method', '-')} {flag}"
            )
            if e.get("reply"):
                lines.append(f"  回复: {e['reply']}")
        return "\n".join(lines)
