"""v1.2.0 main 层逻辑测试：豁免名单、抽样、缓存、熔断、人工反馈 self 过滤。

不连接真实 LLM，全部 mock。
"""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astrbot_plugin_adversarial_guard.main import AdversarialGuard

from core.judge import Verdict


def make_plugin(config=None):
    """绕过 Star.__init__ 构造插件实例（与上游测试风格一致）。"""
    tmp = tempfile.mkdtemp(prefix="guard_main_test_")
    with mock.patch.object(AdversarialGuard, "__init__", return_value=None):
        p = AdversarialGuard(None, config or {})
    p.config = dict(config or {})
    p.data_dir = tmp
    from core.ledger import Ledger
    from core.memory import MemoryStore
    from core.rules import RulesStore

    p.ledger = Ledger(path=os.path.join(tmp, "ledger.json"), decay=0.8, window=5)
    p.memory = MemoryStore(path=os.path.join(tmp, "memory.json"), max_entries=20)
    p.rules_store = RulesStore(path=os.path.join(tmp, "rules.json"))
    p._summary_tasks = set()
    p._review_cache = {}
    p._last_reply_ts = {}
    p._fb_last = {}
    p._pending = {}
    p._last_persona = {}
    return p


class FakeEvent:
    def __init__(self, message_str="", sender_id="u1", self_id="bot001",
                 umo="default:GroupMessage:123", at_wake=False):
        self.message_str = message_str
        self._sid = sender_id
        self._self_id = self_id
        self.unified_msg_origin = umo
        self.is_at_or_wake_command = at_wake

    def get_sender_id(self):
        return self._sid

    def get_self_id(self):
        return self._self_id

    def get_group_id(self):
        return "123"

    def is_admin(self):
        return False


class TestExempt(unittest.TestCase):
    def test_prefix_exempt(self):
        p = make_plugin({"exempt_prefixes": "/af,/status"})
        ev = FakeEvent(message_str="/af status")
        self.assertTrue(p._is_exempt_from_review(ev, "/af status"))

    def test_user_exempt(self):
        p = make_plugin({"exempt_users": "999"})
        ev = FakeEvent(message_str="随便聊聊", sender_id="999")
        self.assertTrue(p._is_exempt_from_review(ev, "随便聊聊"))

    def test_pattern_exempt(self):
        p = make_plugin({"exempt_patterns": "紧急|公告"})
        ev = FakeEvent(message_str="这是紧急通知")
        self.assertTrue(p._is_exempt_from_review(ev, "这是紧急通知"))

    def test_no_exempt_by_default(self):
        p = make_plugin()
        ev = FakeEvent(message_str="普通消息")
        self.assertFalse(p._is_exempt_from_review(ev, "普通消息"))

    def test_invalid_pattern_ignored(self):
        p = make_plugin({"exempt_patterns": "[bad,"})
        ev = FakeEvent(message_str="任意消息")
        self.assertFalse(p._is_exempt_from_review(ev, "任意消息"))


class TestSamplingAndCache(unittest.TestCase):
    def test_sample_rate_full(self):
        p = make_plugin({"review_sample_rate": 100})
        self.assertTrue(all(p._should_review_this_round() for _ in range(20)))

    def test_sample_rate_zero(self):
        p = make_plugin({"review_sample_rate": 0})
        self.assertFalse(any(p._should_review_this_round() for _ in range(20)))

    def test_cache_roundtrip(self):
        p = make_plugin()
        v = Verdict(compliant=False, score=40)
        p._cache_put_verdict("回复文本", "人格", v)
        got = p._cache_get_verdict("回复文本", "人格")
        self.assertIsNotNone(got)
        self.assertEqual(got.score, 40)

    def test_cache_disabled(self):
        p = make_plugin({"review_cache_enabled": False})
        v = Verdict(compliant=False, score=40)
        p._cache_put_verdict("回复文本", "人格", v)
        self.assertIsNone(p._cache_get_verdict("回复文本", "人格"))

    def test_cache_key_includes_persona(self):
        p = make_plugin()
        p._cache_put_verdict("同回复", "人格A", Verdict(score=50))
        self.assertIsNone(p._cache_get_verdict("同回复", "人格B"))


class TestFuse(unittest.TestCase):
    def test_fuse_disabled(self):
        p = make_plugin({"fuse_threshold": 0})
        # 连续违规也不熔断
        for _ in range(5):
            p.ledger.record_check("u", False, 40, [{"severity": 2}], 70)
        self.assertEqual(p._effective_action_with_fuse("u"), p._action())

    def test_fuse_triggers_and_cooldown(self):
        import time
        p = make_plugin({"fuse_threshold": 2, "fuse_cooldown_minutes": 10})
        p.ledger.record_check("u", False, 40, [{"severity": 2}], 70)
        # 第 1 次违规：未达阈值，正常策略
        self.assertNotEqual(p._effective_action_with_fuse("u"), "record_only")
        p.ledger.record_check("u", False, 40, [{"severity": 2}], 70)
        # 第 2 次连续违规：熔断
        self.assertEqual(p._effective_action_with_fuse("u"), "record_only")
        self.assertGreater(p.ledger.fused_until("u"), time.time())
        # 冷却期内持续 record_only
        self.assertEqual(p._effective_action_with_fuse("u"), "record_only")

    def test_fuse_expires(self):
        import time
        p = make_plugin({"fuse_threshold": 2})
        # 仅 1 次违规（streak=1 < threshold=2），手动设置已过期的熔断时间
        p.ledger.record_check("u", False, 40, [], 70)
        p.ledger.set_fused_until("u", time.time() - 1)  # 已过期
        self.assertNotEqual(p._effective_action_with_fuse("u"), "record_only")


class TestHumanFeedbackSelfFilter(unittest.TestCase):
    def _run_listener(self, config, sender_id, self_id="bot001", text="这样不对"):
        p = make_plugin(config)
        ev = FakeEvent(message_str=text, sender_id=sender_id, self_id=self_id)
        run = asyncio.run
        run(p.human_feedback_listener(ev))
        return p, ev

    def test_bot_self_message_ignored(self):
        p, _ = self._run_listener(
            {"human_feedback_enabled": True}, sender_id="bot001", self_id="bot001"
        )
        stats = p.ledger.human_feedback_stats("default:GroupMessage:123")
        self.assertEqual(stats, {"pos": 0, "neg": 0})

    def test_user_negative_feedback_counted(self):
        p = make_plugin({
            "human_feedback_enabled": True,
            "feedback_negative_words": "不对",
        })
        p._last_reply_ts["default:GroupMessage:123"] = __import__("time").time()
        ev = FakeEvent(message_str="不对", sender_id="user1")
        asyncio.run(p.human_feedback_listener(ev))
        stats = p.ledger.human_feedback_stats("default:GroupMessage:123")
        self.assertEqual(stats["neg"], 1)

    def test_disabled_no_count(self):
        p = make_plugin({"human_feedback_enabled": False})
        p._last_reply_ts["default:GroupMessage:123"] = __import__("time").time()
        ev = FakeEvent(message_str="不对", sender_id="user1")
        asyncio.run(p.human_feedback_listener(ev))
        self.assertEqual(p.ledger.human_feedback_stats("default:GroupMessage:123"), {"pos": 0, "neg": 0})

    def test_no_recent_reply_no_count(self):
        p = make_plugin({"human_feedback_enabled": True})
        ev = FakeEvent(message_str="不对", sender_id="user1")
        asyncio.run(p.human_feedback_listener(ev))
        self.assertEqual(p.ledger.human_feedback_stats("default:GroupMessage:123"), {"pos": 0, "neg": 0})


class TestIggetZeroValue(unittest.TestCase):
    """回归：显式配置 0 不得被 falsy 回退吞掉（v1.2.0 修复上游 _iget/_fget 的 `or` 坑）。"""

    def test_sample_rate_zero_respected(self):
        p = make_plugin({"review_sample_rate": 0})
        self.assertEqual(p._iget("review_sample_rate", 100, 0, 100), 0)

    def test_fuse_threshold_zero_respected(self):
        p = make_plugin({"fuse_threshold": 0})
        self.assertEqual(p._iget("fuse_threshold", 0, 0, 100), 0)

    def test_decay_zero_respected(self):
        p = make_plugin({"reward_decay": 0.0})
        self.assertEqual(p._fget("reward_decay", 0.8), 0.0)

    def test_dirty_values_fallback(self):
        p = make_plugin({"review_sample_rate": "abc", "reward_decay": None})
        self.assertEqual(p._iget("review_sample_rate", 100, 0, 100), 100)
        self.assertEqual(p._fget("reward_decay", 0.8), 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=1)
