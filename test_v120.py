# -*- coding: utf-8 -*-
"""v1.2.0 本地测试：规则库提取/存取、ledger 趋势/熔断/人工反馈、抽样与缓存。"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.rules import (  # noqa: E402
    RulesStore,
    build_extract_prompt,
    parse_rules,
    persona_hash,
    render_rules_block,
    rule_id_of,
)
from core.ledger import Ledger  # noqa: E402


class TestParseRules(unittest.TestCase):
    def test_valid(self):
        rules = parse_rules('{"rules": ["禁止markdown", "自称小云", "不超过三句"]}')
        self.assertEqual(len(rules), 3)
        self.assertEqual(rules[0], "禁止markdown")

    def test_fenced(self):
        self.assertEqual(len(parse_rules('```json\n{"rules": ["A"]}\n```')), 1)

    def test_empty_rules(self):
        self.assertEqual(parse_rules('{"rules": []}'), [])

    def test_garbage(self):
        self.assertEqual(parse_rules("not json"), [])
        self.assertEqual(parse_rules(""), [])

    def test_dedupe_and_cap(self):
        raw = json.dumps({"rules": [f"规则{i}" for i in range(40)] + ["规则1"]}, ensure_ascii=False)
        rules = parse_rules(raw)
        self.assertEqual(len(rules), 30)


class TestRulesStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="guard_rules_")
        self.path = os.path.join(self.tmp.name, "rules.json")

    def test_set_get_stale(self):
        s = RulesStore(self.path)
        persona = "你是小云，禁止markdown"
        self.assertTrue(s.is_stale("u", persona))  # 不存在 → stale
        s.set_rules("u", persona, ["禁止markdown"])
        self.assertFalse(s.is_stale("u", persona))
        # persona 变化 → stale
        self.assertTrue(s.is_stale("u", "另一个人格"))

    def test_persistence(self):
        s = RulesStore(self.path)
        s.set_rules("u", "人格P", ["R1", "R2"])
        s2 = RulesStore(self.path)
        self.assertEqual(s2.get("u"), ["R1", "R2"])

    def test_corrupted_resets(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{broken")
        s = RulesStore(self.path)
        self.assertEqual(s.get("u"), [])


class TestRenderAndMatch(unittest.TestCase):
    def test_render_block(self):
        text = render_rules_block(["禁止markdown", "自称小云"])
        self.assertIn("[规则清单]", text)
        self.assertIn("R1. 禁止markdown", text)
        self.assertIn("R2. 自称小云", text)

    def test_render_empty(self):
        self.assertEqual(render_rules_block([]), "")

    def test_rule_id_match(self):
        rules = ["禁止markdown", "自称小云", "不超过三句"]
        self.assertEqual(rule_id_of(rules, "R2"), "R2")
        self.assertEqual(rule_id_of(rules, "R2 自称小云"), "R2")
        self.assertEqual(rule_id_of(rules, "违反了「自称小云」相关规则"), "R2")
        self.assertEqual(rule_id_of(rules, "完全无关"), "")

    def test_persona_hash_stable(self):
        self.assertEqual(persona_hash("abc"), persona_hash("abc"))
        self.assertNotEqual(persona_hash("abc"), persona_hash("abd"))


class TestExtractPrompt(unittest.TestCase):
    def test_contains_persona(self):
        p = build_extract_prompt("你是小云")
        self.assertIn("[人格提示词]", p)
        self.assertIn("你是小云", p)


class TestLedgerV120(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="guard_led_")
        self.ledger = Ledger(path=os.path.join(self.tmp.name, "l.json"), decay=0.8, window=5)

    def test_streak_and_fuse(self):
        self.assertEqual(self.ledger.violate_streak("u"), 0)
        self.ledger.record_check("u", False, 40, [{"rule": "r", "severity": 2}], 70)
        self.ledger.record_check("u", False, 40, [{"rule": "r", "severity": 2}], 70)
        self.assertEqual(self.ledger.violate_streak("u"), 2)
        self.ledger.record_check("u", True, 95, [], 70)
        self.assertEqual(self.ledger.violate_streak("u"), 0)

    def test_fuse_set_get(self):
        import time
        self.assertEqual(self.ledger.fused_until("u"), 0.0)
        self.ledger.set_fused_until("u", time.time() + 600)
        self.assertGreater(self.ledger.fused_until("u"), time.time())

    def test_human_feedback(self):
        self.ledger.record_human_feedback("u", positive=False)
        self.ledger.record_human_feedback("u", positive=True)
        self.ledger.record_human_feedback("u", positive=True)
        stats = self.ledger.human_feedback_stats("u")
        self.assertEqual(stats, {"pos": 2, "neg": 1})
        # 负反馈 -2 ×1 + 正反馈 +1 ×2 = 0
        self.assertAlmostEqual(self.ledger.get_reward("u"), 0.0)

    def test_daily_trend(self):
        self.ledger.record_check("u", True, 90, [], 70)
        self.ledger.record_check("u", False, 40, [{"rule": "r", "severity": 1}], 70)
        trend = self.ledger.daily_trend("u", 7)
        self.assertEqual(len(trend), 7)
        today = trend[-1]
        self.assertEqual(today[1], 2)   # checked
        self.assertEqual(today[2], 1)   # violated
        # 其余天为零
        for _, c, v in trend[:-1]:
            self.assertEqual((c, v), (0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=1)