"""长期记忆存取（v1.1.0 新增）。

以会话（unified_msg_origin）为键，持久化保存由总结模型(C)生成、
经判官(A)复审通过后的「教训」条目。主 AI(B) 每轮请求前读取本会话
条目并以 <guard_memory> 块注入，作为跨轮次的长期行为记忆。

设计要点：
- 同一 rule 的教训不无限堆积：再次出现时更新 lesson / 提升次数（教训演进而非重复）；
- 条目数量按会话上限裁剪，超出时优先淘汰最旧；
- 原子写盘（tmp + os.replace），损坏/非法结构时自动重置，不影响运行；
- 线程锁保护（后台总结任务与主流程可能并发读写）。
"""

from __future__ import annotations

import json
import os
import threading
import time

DEFAULT_MAX_ENTRIES = 20
MAX_LESSON_CHARS = 120
MAX_RULE_CHARS = 200


class MemoryStore:
    """长期记忆存储：按会话保存经复审的教训条目。"""

    def __init__(self, path: str, max_entries: int = DEFAULT_MAX_ENTRIES):
        self.path = path
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._data: dict = {"sessions": {}}
        self.load()

    # ---------------------------------------------------------------- 基础
    def set_max_entries(self, n: int) -> None:
        self.max_entries = max(1, int(n))

    def _session(self, umo: str) -> dict:
        sessions = self._data.setdefault("sessions", {})
        s = sessions.get(umo)
        if s is None:
            s = {"entries": [], "updated_at": 0.0}
            sessions[umo] = s
        return s

    # ---------------------------------------------------------------- 读写
    def entries(self, umo: str) -> list[dict]:
        """返回本会话当前全部教训条目（副本，避免调用方误改内部状态）。"""
        with self._lock:
            s = self._data.get("sessions", {}).get(umo)
            if not s:
                return []
            return [dict(e) for e in s.get("entries", []) if isinstance(e, dict)]

    def add(self, umo: str, rule: str, lesson: str, meta: dict | None = None) -> dict:
        """写入/合并一条教训。

        同一 rule 已存在时：更新 lesson、次数 +1、时间戳刷新（保留首次 seen_at）。
        新条目插入队尾；总条数超过 max_entries 时淘汰最旧。
        返回写入后的条目。
        """
        rule = str(rule or "").strip()[:MAX_RULE_CHARS] or "未说明规则"
        lesson = str(lesson or "").strip()[:MAX_LESSON_CHARS]
        now = time.time()
        with self._lock:
            s = self._session(umo)
            entries: list = s.setdefault("entries", [])
            target = None
            for e in entries:
                if isinstance(e, dict) and e.get("rule") == rule:
                    target = e
                    break
            if target is not None:
                target["lesson"] = lesson or target.get("lesson", "")
                target["count"] = int(target.get("count", 1)) + 1
                target["ts"] = now
                if meta:
                    target.setdefault("meta", {}).update(meta)
                entry = target
            else:
                entry = {
                    "rule": rule,
                    "lesson": lesson,
                    "count": 1,
                    "ts": now,
                    "seen_at": now,
                }
                if meta:
                    entry["meta"] = dict(meta)
                entries.append(entry)
            # 容量裁剪：保留最新的 max_entries 条
            entries.sort(key=lambda e: e.get("ts", 0) if isinstance(e, dict) else 0)
            while len(entries) > self.max_entries:
                entries.pop(0)
            s["updated_at"] = now
            self.save()
        return dict(entry)

    def remove(self, umo: str, index: int) -> bool:
        """按 1-based 序号删除一条教训（与展示顺序一致）。"""
        with self._lock:
            s = self._data.get("sessions", {}).get(umo)
            if not s:
                return False
            entries: list = s.get("entries", [])
            if index < 1 or index > len(entries):
                return False
            entries.pop(index - 1)
            s["updated_at"] = time.time()
            self.save()
            return True

    def clear(self, umo: str) -> None:
        with self._lock:
            sessions = self._data.get("sessions", {})
            if umo in sessions:
                sessions.pop(umo, None)
                self.save()

    # ---------------------------------------------------------------- 渲染
    @staticmethod
    def render_entries(umo_entries: list[dict], max_lines: int = 8) -> str:
        """把教训条目渲染为注入给主 AI(B) 的 <guard_memory> 文本块。

        无条目返回空串。按次数降序取前 max_lines 条，避免提示词膨胀。
        """
        if not umo_entries:
            return ""
        ranked = sorted(
            [e for e in umo_entries if isinstance(e, dict)],
            key=lambda e: (-int(e.get("count", 1)), -float(e.get("ts", 0))),
        )[: max(1, max_lines)]
        lines = [
            "<guard_memory>",
            ("[长期记忆·判官A确认的行为教训] 你在过去对话中曾反复出现以下问题，"
             "本轮回复务必逐条规避："),
        ]
        for i, e in enumerate(ranked, 1):
            rule = str(e.get("rule", ""))
            lesson = str(e.get("lesson", ""))
            cnt = int(e.get("count", 1))
            line = f"{i}. {rule}（已出现 {cnt} 次）"
            if lesson and lesson != rule:
                line += f"：{lesson}"
            lines.append(line)
        lines.append("</guard_memory>")
        return "\n".join(lines)

    # ---------------------------------------------------------------- 持久化
    def save(self) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as e:
            import logging

            logging.getLogger("astrbot").warning(
                f"[adversarial_guard] 长期记忆保存失败: {e}"
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
                f"[adversarial_guard] 长期记忆加载失败，已重置: {e}"
            )
            self._data = {"sessions": {}}
