# -*- coding: utf-8 -*-
"""v1.1.0 长期记忆功能本地测试：MemoryStore、总结/复审解析、注入渲染。

不依赖 AstrBot 运行时，纯本地单测。
运行：python -m unittest test_local -v
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.memory import MemoryStore  # noqa: E402
from core.summarizer import (  # noqa: E402
    build_review_prompt,
    build_summary_prompt,
    parse_review,
    parse_summary,
    resolve_entry,
)


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="guard_mem_")
        self.path = os.path.join(self.tmp.name, "memory.json")

    def _store(self, max_entries=20):
        return MemoryStore(path=self.path, max_entries=max_entries)

    def test_add_and_entries(self):
        m = self._store()
        m.add("umo1", rule="禁止markdown", lesson="用纯文本表述")
        entries = m.entries("umo1")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["rule"], "禁止markdown")
        self.assertEqual(entries[0]["count"], 1)

    def test_same_rule_merges(self):
        m = self._store()
        m.add("umo1", rule="禁止markdown", lesson="v1")
        m.add("umo1", rule="禁止markdown", lesson="v2 更明确")
        entries = m.entries("umo1")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["count"], 2)
        self.assertEqual(entries[0]["lesson"], "v2 更明确")

    def test_different_rules_separate(self):
        m = self._store()
        m.add("u", rule="规则A", lesson="a")
        m.add("u", rule="规则B", lesson="b")
        self.assertEqual(len(m.entries("u")), 2)

    def test_session_isolation(self):
        m = self._store()
        m.add("u1", rule="A", lesson="a")
        m.add("u2", rule="B", lesson="b")
        self.assertEqual(len(m.entries("u1")), 1)
        self.assertEqual(len(m.entries("u2")), 1)

    def test_max_entries_eviction_oldest(self):
        m = self._store(max_entries=3)
        import time
        for i in range(5):
            m.add("u", rule=f"规则{i}", lesson=f"l{i}")
            time.sleep(0.01)
        entries = m.entries("u")
        self.assertEqual(len(entries), 3)
        rules = [e["rule"] for e in entries]
        # 最旧的两条被淘汰
        self.assertNotIn("规则0", rules)
        self.assertNotIn("规则1", rules)
        self.assertIn("规则4", rules)

    def test_persistence_roundtrip(self):
        m = self._store()
        m.add("u", rule="A", lesson="a")
        m2 = self._store()
        self.assertEqual(len(m2.entries("u")), 1)
        self.assertEqual(m2.entries("u")[0]["rule"], "A")

    def test_corrupted_file_resets(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{broken json!!!")
        m = self._store()
        self.assertEqual(m.entries("u"), [])
        m.add("u", rule="A", lesson="a")  # 重置后可正常使用
        self.assertEqual(len(m.entries("u")), 1)

    def test_remove_by_index(self):
        m = self._store()
        m.add("u", rule="A", lesson="a")
        m.add("u", rule="B", lesson="b")
        self.assertTrue(m.remove("u", 1))
        rules = [e["rule"] for e in m.entries("u")]
        self.assertNotIn("A", rules)
        self.assertFalse(m.remove("u", 99))

    def test_clear(self):
        m = self._store()
        m.add("u", rule="A", lesson="a")
        m.clear("u")
        self.assertEqual(m.entries("u"), [])


class TestRenderEntries(unittest.TestCase):
    def test_empty_returns_empty(self):
        self.assertEqual(MemoryStore.render_entries([]), "")

    def test_block_format(self):
        entries = [
            {"rule": "禁止markdown", "lesson": "用纯文本", "count": 3, "ts": 1},
        ]
        text = MemoryStore.render_entries(entries)
        self.assertIn("<guard_memory>", text)
        self.assertIn("</guard_memory>", text)
        self.assertIn("禁止markdown", text)
        self.assertIn("已出现 3 次", text)
        self.assertIn("用纯文本", text)

    def test_ranked_by_count_max_lines(self):
        entries = [
            {"rule": f"规则{i}", "lesson": f"l{i}", "count": i, "ts": i}
            for i in range(1, 12)
        ]
        text = MemoryStore.render_entries(entries, max_lines=8)
        # 次数最高的 8 条（11..4）入选，次数低的 3 条淘汰
        self.assertIn("规则11", text)
        self.assertIn("规则4", text)
        self.assertNotIn("规则3", text)


class TestParseSummary(unittest.TestCase):
    def test_valid(self):
        raw = '{"rule": "禁止markdown", "lesson": "用纯文本", "keep": true}'
        d = parse_summary(raw)
        self.assertIsNotNone(d)
        self.assertEqual(d["rule"], "禁止markdown")

    def test_fenced_json(self):
        raw = '```json\n{"rule": "A", "lesson": "B", "keep": true}\n```'
        self.assertIsNotNone(parse_summary(raw))

    def test_keep_false_returns_none(self):
        self.assertIsNone(parse_summary('{"rule": "A", "lesson": "B", "keep": false}'))

    def test_missing_fields(self):
        self.assertIsNone(parse_summary('{"rule": "", "lesson": "B", "keep": true}'))
        self.assertIsNone(parse_summary('not json at all'))
        self.assertIsNone(parse_summary(''))


class TestParseReviewAndResolve(unittest.TestCase):
    def test_parse_pass(self):
        d = parse_review('{"pass": true, "reason": "", "fixed_lesson": "", "fixed_rule": ""}')
        self.assertTrue(d["pass"])

    def test_parse_fail_with_fix(self):
        d = parse_review(json.dumps({
            "pass": False, "reason": "含QQ号",
            "fixed_lesson": "脱敏后的教训", "fixed_rule": "规则",
        }, ensure_ascii=False))
        self.assertFalse(d["pass"])
        self.assertEqual(d["fixed_lesson"], "脱敏后的教训")

    def test_resolve_review_none_trusts_draft(self):
        draft = {"rule": "A", "lesson": "B"}
        self.assertEqual(resolve_entry(draft, None), draft)

    def test_resolve_pass(self):
        draft = {"rule": "A", "lesson": "B"}
        review = {"pass": True, "fixed_lesson": "", "fixed_rule": ""}
        self.assertEqual(resolve_entry(draft, review), draft)

    def test_resolve_adopts_fixed(self):
        draft = {"rule": "A", "lesson": "B"}
        review = {"pass": False, "fixed_lesson": "B修正", "fixed_rule": "A2"}
        entry = resolve_entry(draft, review)
        self.assertEqual(entry["lesson"], "B修正")
        self.assertEqual(entry["rule"], "A2")
        self.assertTrue(entry["meta"]["corrected"])

    def test_resolve_discard_when_no_fix(self):
        draft = {"rule": "A", "lesson": "B"}
        review = {"pass": False, "fixed_lesson": "", "fixed_rule": ""}
        self.assertIsNone(resolve_entry(draft, review))


class TestPrompts(unittest.TestCase):
    def test_summary_prompt_contains_sections(self):
        p = build_summary_prompt(
            {"compliant": False, "score": 42, "violations": [{"rule": "R", "evidence": "E", "severity": 3}]},
            {"method": "fast_fix", "ok": True, "attempts": []},
            -3.5,
        )
        self.assertIn("verdict", p)
        self.assertIn("repair", p)
        self.assertIn("reward", p)

    def test_review_prompt_contains_draft(self):
        p = build_review_prompt(
            {"compliant": False, "score": 42, "violations": []},
            {"method": "regen", "ok": False},
            {"rule": "A", "lesson": "B"},
        )
        self.assertIn("待写入条目", p)
        self.assertIn("A", p)

    def test_long_input_truncated(self):
        p = build_summary_prompt({"v": "x" * 5000}, {}, 0)
        self.assertLessEqual(len(p), 3200)


if __name__ == "__main__":
    unittest.main(verbosity=1)
