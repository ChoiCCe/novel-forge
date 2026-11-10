"""A/B 参数合成的最小单测（不联网、不合成）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import prosody  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def base_profile(temperament: str, stage: str, emotion: str = "平静") -> dict:
    return {"temperament": temperament, **prosody.persona_base(temperament, stage, CFG)}


class PersonaTest(unittest.TestCase):
    def test_temperament_plus_age(self):
        base = prosody.persona_base("急躁", "少年", CFG)
        self.assertEqual(base["rate"], "+11%")
        self.assertEqual(base["pitch"], "+7Hz")

    def test_calm_middle_aged_is_negative(self):
        base = prosody.persona_base("沉稳", "中年", CFG)
        self.assertEqual(base["rate"], "-6%")
        self.assertEqual(base["pitch"], "-3Hz")


class CombineTest(unittest.TestCase):
    """v3 compose_prosody：6 组场景都必须落在夹紧范围内。"""

    def assert_clamped(self, composed):
        low, high = prosody.clamp_bounds("rate", CFG)
        self.assertGreaterEqual(composed["numeric"]["rate"], low)
        self.assertLessEqual(composed["numeric"]["rate"], high)
        low, high = prosody.clamp_bounds("pitch", CFG)
        self.assertGreaterEqual(composed["numeric"]["pitch"], low)
        self.assertLessEqual(composed["numeric"]["pitch"], high)
        low, high = prosody.clamp_bounds("volume", CFG)
        self.assertGreaterEqual(composed["numeric"]["volume"], low)
        self.assertLessEqual(composed["numeric"]["volume"], high)

    def test_calm_narration(self):
        composed = prosody.compose_prosody(
            {"rate": "-4%", "pitch": "+0Hz", "volume": "+0%"}, "平静",
            "他沿着长街慢慢走回家，屋檐上的雨还没停。", CFG)
        self.assert_clamped(composed)
        self.assertEqual(composed["formatted"], {"rate": "-4%", "pitch": "+0Hz", "volume": "+0%"})

    def test_anger_short_sentence(self):
        # 周律 #14 的真实组合：急躁少年 + 愤怒
        profile = base_profile("急躁", "少年")
        composed = prosody.compose_prosody(profile, "愤怒", "气死我了！气死我了！", CFG)
        self.assert_clamped(composed)
        self.assertGreater(composed["raw"]["rate"], composed["numeric"]["rate"])
        self.assertGreaterEqual(prosody.parse_value(composed["formatted"]["rate"], "%"), 14)
        self.assertGreaterEqual(prosody.parse_value(composed["formatted"]["pitch"], "Hz"), 4)
        self.assertGreaterEqual(prosody.parse_value(composed["formatted"]["volume"], "%"), 4)

    def test_anger_long_sentence(self):
        profile = base_profile("急躁", "少年")
        text = "他知道自己已经退无可退，可偏偏还要在这群人面前把话说完，胸口的火压不住。"
        composed = prosody.compose_prosody(profile, "愤怒", text, CFG)
        self.assert_clamped(composed)
        self.assertGreater(composed["numeric"]["rate"], 0)

    def test_sadness(self):
        profile = base_profile("中性", "青年")
        text = "他说完这句话以后，屋里再没有人开口，窗外的雨一直下到天亮。"
        composed = prosody.compose_prosody(profile, "悲伤", text, CFG)
        self.assert_clamped(composed)
        self.assertLess(composed["numeric"]["rate"], 0)

    def test_majesty(self):
        profile = base_profile("沉稳", "中年")
        text = "他抬眼扫过众人，一字一句道：“此事，就这么定了。”"
        composed = prosody.compose_prosody(profile, "威严", text, CFG)
        self.assert_clamped(composed)

    def test_burst(self):
        profile = base_profile("豪爽", "中年")
        composed = prosody.compose_prosody(profile, "惊讶", "什么？！", CFG)
        self.assert_clamped(composed)

    def test_short_sentence_anger_faster_than_long(self):
        short = prosody.emotion_delta("愤怒", 5, CFG)
        long = prosody.emotion_delta("愤怒", 30, CFG)
        self.assertEqual(short["variant"], "short")
        self.assertGreater(short["rate"], long["rate"])


class TextTest(unittest.TestCase):
    def test_number_to_chinese(self):
        self.assertEqual(prosody.normalize_text("他今年18岁，1980年入京。"),
                         "他今年十八岁，一九八零年入京。")

    def test_split_respects_quotes(self):
        text = "他压低声音说“你，到底，想做什么”，然后转身，门外传来脚步，雨声渐密。"
        clauses = prosody.split_clauses(text, CFG)
        self.assertGreater(len(clauses), 1)
        self.assertIn("想做什么”，然后转身，", "".join(clauses))

    def test_plan_pieces_clause_pauses(self):
        text = "此事绝不能就这么拖下去，否则等他们缓过神来，再想动手就难了。"
        pieces = prosody.plan_pieces(text, CFG)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(prosody._has_speech(piece["text"]) for piece in pieces))
        self.assertTrue(any(piece["pause_before_ms"] >= 120 for piece in pieces[1:]))
        self.assertEqual("".join(piece["text"] for piece in pieces), text)

    def test_pause_template(self):
        template = prosody.pause_template(CFG)
        self.assertEqual(template["chapter_start_ms"], 800)
        self.assertEqual(prosody.transition_pause("narration", "dialogue", CFG), 350)
        self.assertEqual(prosody.transition_pause("dialogue", "narration", CFG), 250)
        self.assertEqual(prosody.transition_pause("dialogue", "dialogue", CFG), 500)


if __name__ == "__main__":
    unittest.main()
