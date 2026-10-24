"""长期记忆存取（v1.1.0 新增，v1.3.0 增加全局层）。

以会话（unified_msg_origin）为键，持久化保存由总结模型(C)生成、
经判官(A)复审通过后的「教训」条目。主 AI(B) 每轮请求前读取本会话
条目并以 <guard_memory> 块注入，作为跨轮次的长期行为记忆。

v1.3.0 新增全局层：
- 会话内某教训反复出现（次数达晋升阈值）→ 自动晋升到全局层（__global__），
  之后所有会话注入时均可见——多群部署时教训互通、学得更快；
- 注入合并时同 rule 以会话层为准（会话层更新即覆盖全局旧版）；
- /guard forget global <序号|all> 可管理全局层（管理员）。

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
DEFAULT_PROMOTE_THRESHOLD = 3
MAX_LESSON_CHARS = 120
MAX_RULE_CHARS = 200

# 全局层的伪会话键
GLOBAL_KEY = "__global__"


class MemoryStore:
    """长期记忆存储：按会话保存经复审的教训条目（v1.3.0 含全局层）。"""

    def __init__(
        self,
        path: str,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        promote_threshold: int = DEFAULT_PROMOTE_THRESHOLD,
    ):
        self.path = path
        self.max_entries = max(1, int(max_entries))
        # 会话内累计次数达到该阈值时，教训晋升到全局层（<=0 表示关闭晋升）
        self.promote_threshold = max(0, int(promote_threshold))
        self._lock = threading.Lock()
        self._data: dict = {"sessions": {}}
        self.load()

    # ---------------------------------------------------------------- 基础
    def set_max_entries(self, n: int) -> None:
        self.max_entries = max(1, int(n))

    def set_promote_threshold(self, n: int) -> None:
        self.promote_threshold = max(0, int(n))

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
            # v1.3.0：达到晋升阈值 → 写入全局层（多会话共享教训）
            if (
                self.promote_threshold > 0
                and entry.get("rule")
                and int(entry.get("count", 1)) >= self.promote_threshold
            ):
                self._promote_locked(entry)
            self.save()
        return dict(entry)

    def _promote_locked(self, entry: dict) -> bool:
        """把教训写入全局层（调用方必须已持有锁）。返回是否为新增。"""
        rule = entry.get("rule")
        g = self._session(GLOBAL_KEY)
        g_entries: list = g.setdefault("entries", [])
        target = None
        for e in g_entries:
            if isinstance(e, dict) and e.get("rule") == rule:
                target = e
                break
        now = time.time()
        if target is not None:
            # 全局层合并：lesson 取最新，count 取全局累计 +1
            target["lesson"] = entry.get("lesson") or target.get("lesson", "")
            target["count"] = int(target.get("count", 1)) + 1
            target["ts"] = now
            target.setdefault("meta", {})["from_sessions"] = (
                int(target.get("meta", {}).get("from_sessions", 1)) + 1
            )
            return False
        g_entries.append(
            {
                "rule": rule,
                "lesson": entry.get("lesson", ""),
                "count": 1,
                "ts": now,
                "seen_at": now,
                "meta": {"from_sessions": 1},
            }
        )
        g_entries.sort(key=lambda e: e.get("ts", 0) if isinstance(e, dict) else 0)
        while len(g_entries) > max(self.max_entries, 30):
            g_entries.pop(0)
        g["updated_at"] = now
        return True

    # ------------------------------------------------------------ 全局层（v1.3.0）
    def global_entries(self) -> list[dict]:
        with self._lock:
            s = self._data.get("sessions", {}).get(GLOBAL_KEY)
            if not s:
                return []
            return [dict(e) for e in s.get("entries", []) if isinstance(e, dict)]

    def entries_merged(self, umo: str) -> list[dict]:
        """合并视图：全局层 + 本会话层，同 rule 以会话层为准。供注入使用。"""
        with self._lock:
            merged: dict[str, dict] = {}
            g = self._data.get("sessions", {}).get(GLOBAL_KEY) or {}
            for e in g.get("entries", []):
                if isinstance(e, dict) and e.get("rule"):
                    d = dict(e)
                    d["global"] = True
                    merged[d["rule"]] = d
            s = self._data.get("sessions", {}).get(umo) or {}
            for e in s.get("entries", []):
                if isinstance(e, dict) and e.get("rule"):
                    d = dict(e)
                    d["global"] = d["rule"] in merged  # 会话层覆盖全局时仍标记来源
                    merged[d["rule"]] = d
            return list(merged.values())

    def remove_global(self, index: int) -> bool:
        """按 1-based 序号删除一条全局教训。"""
        with self._lock:
            s = self._data.get("sessions", {}).get(GLOBAL_KEY)
            if not s:
                return False
            entries: list = s.get("entries", [])
            if index < 1 or index > len(entries):
                return False
            entries.pop(index - 1)
            s["updated_at"] = time.time()
            self.save()
            return True

    def clear_global(self) -> None:
        with self._lock:
            sessions = self._data.get("sessions", {})
            if GLOBAL_KEY in sessions:
                sessions.pop(GLOBAL_KEY, None)
                self.save()

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
            scope = "🌐" if e.get("global") else ""
            line = f"{i}. {scope}{rule}（已出现 {cnt} 次）"
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
