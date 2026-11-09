"""v4 步骤 3：性别/年龄推断单元测试（不联网）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import inference  # noqa: E402

CFG = {
    "casting": {
        "gender": {"llm_confidence_cap": 0.70, "llm_default_confidence": 0.60,
                   "unknown_confidence": 0.20, "review": 0.60, "min_gap": 0.20,
                   "weight": {
            "llm": 0.55, "pronoun": 0.05, "pronoun_max": 0.25,
            "title": 0.20, "self_reference": 0.10, "name_hint": 0.10,
            "conflict_penalty": 0.20}},
        "age": {
            "stage_order": ["少年", "青年", "中年", "老年"],
            "default_stage": "中年",
            "bands": {"少年": [0, 17], "青年": [18, 34],
                      "中年": [35, 54], "老年": [55, 120]},
            "advance_min_age": {"少年": 20, "青年": 37, "中年": 57},
            "auto_apply_confidence": 0.75,
            "evidence_default_confidence": 0.50,
            "flashback_confidence_factor": 0.50,
            "evidence_log_cap": 12,
            "consecutive_evidence": 2, "cooldown_chapters": 3,
            "flashback_keywords": ["回忆", "当年"],
            "use_real_age": False,
            "transition": {"chapters": 2},
        },
    }
}


class GenderTest(unittest.TestCase):
    def test_male_evidence(self):
        r = inference.resolve_gender("周律", {"gender": "男",
                                          "persona": "他拍案而起，本官倒要看看，公子好大的胆子"}, CFG)
        self.assertEqual(r["gender"], "男")
        self.assertGreaterEqual(r["confidence"], 0.6)

    def test_female_title(self):
        r = inference.resolve_gender("苏姑娘", {"gender": "女",
                                            "persona": "她低头，自称妾身，见过夫人"}, CFG)
        self.assertEqual(r["gender"], "女")

    def test_no_evidence_unknown(self):
        r = inference.resolve_gender("路人甲", {}, CFG)
        self.assertEqual(r["gender"], "未知")
        self.assertLess(r["confidence"], 0.6)

    def test_conflict_low_confidence(self):
        r = inference.resolve_gender("林某", {"gender": "男",
                                          "persona": "她自称妾身，却被人称作公子"}, CFG)
        self.assertLess(r["confidence"], 0.6)

    def test_llm_confidence_capped(self):
        r = inference.resolve_gender("某人", {"gender": "男", "gender_confidence": 1.0}, CFG)
        self.assertLess(r["confidence"], 0.85)


def profile(stage="少年", **extra):
    base = {"age_stage": stage, "first_seen_chapter": 1, "age_evidence_log": []}
    base.update(extra)
    return base


class AgeTest(unittest.TestCase):
    def test_under_buffer_holds(self):
        r = inference.decide_age(profile("少年"), {"age": 18, "stage": "青年", "confidence": 0.9}, 4, CFG)
        self.assertEqual(r["action"], "hold")
        self.assertIn("20", r["reason"])

    def test_over_buffer_advances(self):
        p = profile("少年", stage_entered_chapter=1)
        r = inference.decide_age(p, {"age": 20, "stage": "青年", "confidence": 0.9}, 4, CFG)
        self.assertEqual(r["action"], "advance")
        self.assertEqual(r["target"], "青年")

    def test_real_age_switch(self):
        p = profile("少年", book_real_age=True, stage_entered_chapter=1)
        cfg_real = {"casting": {"age": dict(CFG["casting"]["age"], use_real_age=True)}}
        r = inference.decide_age(p, {"age": 18, "stage": "青年", "confidence": 0.9}, 4, cfg_real)
        self.assertEqual(r["action"], "advance")

    def test_no_number_requires_two_chapters(self):
        p = profile("少年", stage_entered_chapter=1)
        r = inference.decide_age(p, {"age": None, "stage": "青年", "confidence": 0.8}, 3, CFG)
        self.assertEqual(r["action"], "hold")
        p["age_evidence_log"] = [{"chapter": 3, "stage": "青年", "confidence": 0.8}]
        r = inference.decide_age(p, {"age": None, "stage": "青年", "confidence": 0.8}, 4, CFG)
        self.assertEqual(r["action"], "advance")

    def test_cooldown_blocks(self):
        p = profile("青年", stage_entered_chapter=5)
        r = inference.decide_age(p, {"age": 40, "stage": "中年", "confidence": 0.9}, 6, CFG)
        self.assertEqual(r["action"], "hold")
        self.assertIn("冷却", r["reason"])

    def test_cap_one_band(self):
        p = profile("少年", stage_entered_chapter=1)
        r = inference.decide_age(p, {"age": 60, "stage": "老年", "confidence": 0.9}, 5, CFG)
        self.assertEqual(r["target"], "青年")

    def test_flashback_halves(self):
        p = profile("青年", stage_entered_chapter=1)
        r = inference.decide_age(p, {"age": 50, "stage": "中年", "confidence": 0.8,
                                     "evidence": "回忆当年他已五十岁"}, 5, CFG)
        self.assertEqual(r["action"], "hold")

    def test_no_downgrade(self):
        r = inference.decide_age(profile("中年"), {"age": 20, "stage": "青年", "confidence": 0.9}, 6, CFG)
        self.assertEqual(r["action"], "hold")
        self.assertIn("降级", r["reason"])


class TransitionTest(unittest.TestCase):
    def test_blend_half_then_full(self):
        p = {"transition": {"start_chapter": 5, "chapters": 2,
                            "from": {"rate": 6, "pitch": 0, "volume": 0},
                            "to": {"rate": 0, "pitch": -4, "volume": 0}}}
        first = inference.blend_transition(p, 5, CFG)
        self.assertAlmostEqual(first["rate"], 3.0)
        self.assertAlmostEqual(first["pitch"], -2.0)
        second = inference.blend_transition(p, 6, CFG)
        self.assertAlmostEqual(second["rate"], 0.0)
        self.assertEqual(inference.blend_transition(p, 7, CFG), {})

    def test_no_transition(self):
        self.assertIsNone(inference.blend_transition({"transition": None}, 5, CFG))


if __name__ == "__main__":
    unittest.main()
