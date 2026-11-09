"""A/B 参数合成与导演层兜底的最小单测（不联网、不合成）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import director, prosody  # noqa: E402

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
            "他沿着长街慢慢走回家，屋檐上的雨还没停。", prosody.default_direction(), CFG)
        self.assert_clamped(composed)
        self.assertEqual(composed["formatted"], {"rate": "-4%", "pitch": "+0Hz", "volume": "+0%"})

    def test_anger_short_sentence(self):
        # 周律 #14 的真实组合：急躁少年 + 愤怒 + fast/loud
        profile = base_profile("急躁", "少年")
        direction = {"pace": "fast", "emphasis": ["气死我了"], "pause_after_ms": 200, "volume": "loud"}
        composed = prosody.compose_prosody(profile, "愤怒", "气死我了！气死我了！", direction, CFG)
        self.assert_clamped(composed)
        self.assertGreater(composed["raw"]["rate"], composed["numeric"]["rate"])
        # v4 验收：愤怒语速终值 16~18%，pitch 5~7Hz，volume 6~8%
        self.assertGreaterEqual(prosody.parse_value(composed["formatted"]["rate"], "%"), 16)
        self.assertLessEqual(prosody.parse_value(composed["formatted"]["rate"], "%"), 18)
        self.assertGreaterEqual(prosody.parse_value(composed["formatted"]["pitch"], "Hz"), 5)
        self.assertGreaterEqual(prosody.parse_value(composed["formatted"]["volume"], "%"), 6)

    def test_anger_long_sentence(self):
        profile = base_profile("急躁", "少年")
        text = "他知道自己已经退无可退，可偏偏还要在这群人面前把话说完，胸口的火压不住。"
        composed = prosody.compose_prosody(profile, "愤怒", text, prosody.default_direction(), CFG)
        self.assert_clamped(composed)
        self.assertGreater(composed["numeric"]["rate"], 0)

    def test_sadness(self):
        profile = base_profile("中性", "青年")
        text = "他说完这句话以后，屋里再没有人开口，窗外的雨一直下到天亮。"
        composed = prosody.compose_prosody(profile, "悲伤", text,
                                           prosody.default_direction() | {"pace": "slow"}, CFG)
        self.assert_clamped(composed)
        self.assertLess(composed["numeric"]["rate"], 0)

    def test_majesty(self):
        profile = base_profile("沉稳", "中年")
        text = "他抬眼扫过众人，一字一句道：“此事，就这么定了。”"
        composed = prosody.compose_prosody(profile, "威严", text,
                                           prosody.default_direction() | {"pace": "slow", "volume": "soft"}, CFG)
        self.assert_clamped(composed)

    def test_burst(self):
        profile = base_profile("豪爽", "中年")
        composed = prosody.compose_prosody(
            profile, "惊讶", "什么？！",
            {"pace": "burst", "emphasis": [], "pause_after_ms": 300, "volume": "loud"}, CFG)
        self.assert_clamped(composed)

    def test_short_sentence_anger_faster_than_long(self):
        profile = base_profile("中性", "青年")
        short = prosody.emotion_delta("愤怒", 5, CFG)
        long = prosody.emotion_delta("愤怒", 30, CFG)
        self.assertEqual(short["variant"], "short")
        self.assertGreater(short["rate"], long["rate"])

    def test_emphasis_right_after_open_quote_has_no_empty_piece(self):
        # 强调词紧跟开引号时，子切分不能留下纯标点的无声片段（edge 会 No audio）
        text = "‘挨打的不是你！’周律怒声打断，抬手揉揉眼边，‘关键是咽不下这口气。’"
        direction = {"pace": "normal", "emphasis": ["挨打"], "pause_after_ms": 200,
                     "volume": "loud"}
        pieces = prosody.plan_pieces(text, direction, CFG)
        self.assertTrue(all(p["text"].strip("‘’“”") for p in pieces))
        self.assertTrue(all(prosody._has_speech(p["text"]) for p in pieces))
        self.assertEqual("".join(p["text"] for p in pieces), text)


class TextTest(unittest.TestCase):
    def test_number_to_chinese(self):
        self.assertEqual(prosody.normalize_text("他今年18岁，1980年入京。"),
                         "他今年十八岁，一九八零年入京。")

    def test_split_respects_quotes(self):
        text = "他压低声音说“你，到底，想做什么”，然后转身，门外传来脚步，雨声渐密。"
        clauses = prosody.split_clauses(text, CFG)
        self.assertGreater(len(clauses), 1)
        self.assertIn("想做什么”，然后转身，", "".join(clauses))

    def test_plan_pieces_emphasis_pause_and_slow(self):
        text = "此事，绝不能，再拖下去。"
        direction = {"pace": "normal", "emphasis": ["绝不能"], "pause_after_ms": 0, "volume": "normal"}
        pieces = prosody.plan_pieces(text, direction, CFG)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(any(piece["pause_before_ms"] >= 80 for piece in pieces[1:]))
        slowed = [piece for piece in pieces if piece["text"].startswith("绝不能")]
        self.assertTrue(slowed and slowed[0]["rate_pct"] < 0)

    def test_pause_template(self):
        template = prosody.pause_template(CFG)
        self.assertEqual(template["chapter_start_ms"], 800)
        self.assertEqual(prosody.transition_pause("narration", "dialogue", CFG), 350)
        self.assertEqual(prosody.transition_pause("dialogue", "narration", CFG), 250)
        self.assertEqual(prosody.transition_pause("dialogue", "dialogue", CFG), 500)


class DirectorTest(unittest.TestCase):
    def segments(self):
        return [SimpleNamespace(type="narration", speaker="旁白", emotion="平静", text="甲"),
                SimpleNamespace(type="dialogue", speaker="楼温", emotion="愤怒", text="乙")]

    def test_missing_index_falls_back(self):
        items, covered = director._items_from_payload(
            {"directions": [{"index": 1, "pace": "fast", "emphasis": ["甲"],
                             "pause_after_ms": 900, "volume": "loud"}]},
            2, 1200)
        self.assertEqual(covered, {1})
        self.assertEqual(items[0]["pace"], "fast")
        self.assertEqual(items[1], prosody.default_direction())

    def test_bad_values_are_clamped(self):
        items, _ = director._items_from_payload(
            {"directions": [{"index": 1, "pace": "urgent", "emphasis": "甲",
                             "pause_after_ms": 9999, "volume": "max"}]}, 1, 1200)
        self.assertEqual(items[0]["pace"], "normal")
        self.assertEqual(items[0]["volume"], "normal")
        self.assertEqual(items[0]["pause_after_ms"], 1200)

    def test_index_keyed_object_and_punctuation(self):
        items, covered = director._items_from_payload(
            {"1": {"pace": "slow", "emphasis": ["气死我了！"]},
             "2": {"pace": "burst", "volume": "loud"}}, 2, 1200)
        self.assertEqual(covered, {1, 2})
        self.assertEqual(items[0]["emphasis"], ["气死我了"])
        self.assertEqual(items[1]["pace"], "burst")

    def test_two_failures_mark_fallback(self):
        class BadLLM:
            calls = 0

            def chat_json(self, messages, **kwargs):
                BadLLM.calls += 1
                raise ValueError("not json")

        result = director.run_director(self.segments(), BadLLM(), CFG, "第一章")
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(BadLLM.calls, 2)
        self.assertEqual(result["items"], [prosody.default_direction()] * 2)

    def test_emphasis_outside_text_is_dropped(self):
        segments = [SimpleNamespace(type="dialogue", speaker="周律", emotion="愤怒",
                                    text="气死我了！气死我了！")]
        items = [{"pace": "fast", "emphasis": ["气死我了", "弑君改天"],
                  "pause_after_ms": 0, "volume": "normal"}]
        cleaned, dropped = director.validate_emphasis(items, segments)
        self.assertEqual(cleaned[0]["emphasis"], ["气死我了"])
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["word"], "弑君改天")
        self.assertEqual(dropped[0]["index"], 1)

    def test_gating_sends_only_clue_segments(self):
        segments = [
            SimpleNamespace(type="narration", speaker="旁白", emotion="平静",
                            text="他沿着长街慢慢走回家，屋檐上的雨还没停。"),
            SimpleNamespace(type="dialogue", speaker="楼温", emotion="平静",
                            text="今日朝中议事，诸公先听我一言。"),
            SimpleNamespace(type="narration", speaker="旁白", emotion="平静", text="天亮了。"),
            SimpleNamespace(type="dialogue", speaker="周律", emotion="愤怒", text="你给我站住！"),
        ]
        est = director.estimate(segments, CFG)
        self.assertEqual(est["total"], 4)
        self.assertEqual(est["sent"], 2)  # 短促句 + 愤怒对白
        self.assertEqual(est["default_count"], 2)

    def test_nothing_eligible_skips_llm(self):
        segments = [SimpleNamespace(type="narration", speaker="旁白", emotion="平静",
                                    text="他沿着长街慢慢走回家，屋檐上的雨还没停。")]
        calls = {"n": 0}

        class CountingLLM:
            def chat_json(self, messages, **kwargs):
                calls["n"] += 1
                return {"directions": []}

        result = director.run_director(segments, CountingLLM(), CFG, "第一章")
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["sent"], 0)
        self.assertEqual(result["default_count"], 1)
        self.assertEqual(calls["n"], 0)


if __name__ == "__main__":
    unittest.main()
