"""阶段 0：累计出场统计与分档判定（无绑定副作用）。"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import appearance  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def record(index: int, segments: list[dict]) -> dict:
    return {"book_id": "book_a", "chapter_index": index,
            "segments": segments}


def seg(speaker: str, text: str, kind: str = "dialogue") -> dict:
    return {"type": kind, "speaker": speaker, "text": text, "emotion": "平静"}


class AggregateTest(unittest.TestCase):
    def test_empty_book_does_not_crash(self):
        result = appearance.aggregate_book([])
        self.assertEqual(result["chapter_count"], 0)
        self.assertEqual(result["total_chars"], 0)
        self.assertEqual(result["chapters"], [])
        self.assertEqual(result["roles"], {})

    def test_totals_share_idle_and_chapters(self):
        records = [
            record(0, [seg("甲", "一二三"), seg("乙", "四"), seg("旁白", "五六", "narration")]),
            record(1, [seg("甲", "七八"), seg("丙", "九")]),
        ]
        result = appearance.aggregate_book(records)
        self.assertEqual(result["chapter_count"], 2)
        self.assertEqual(result["total_chars"], 7)  # 旁白不计，空格不计
        jia = result["roles"]["甲"]
        self.assertEqual(jia["chapters_seen"], 2)
        self.assertEqual(jia["total_lines"], 2)
        self.assertEqual(jia["total_chars"], 5)
        self.assertAlmostEqual(jia["share"], 5 / 7, places=3)
        self.assertEqual(jia["idle_chapters"], 0)
        yi = result["roles"]["乙"]
        self.assertEqual(yi["idle_chapters"], 1)
        self.assertNotIn("旁白", result["roles"])

    def test_chapter_stats_ignores_narration(self):
        stats = appearance.chapter_stats(record(0, [seg("甲", "好"), seg("旁白", "叙述", "narration")]))
        self.assertEqual(stats, {"甲": {"lines": 1, "chars": 1}})


class RegistryApplyTest(unittest.TestCase):
    def test_apply_does_not_touch_bindings(self):
        registry = {
            "identities": {"甲": {"canonical_name": "甲", "books": ["book_a"]}},
            "character_aliases": {"甲": "甲"},
            "voice_bindings": {"lock:甲|青年": {"character_id": "甲",
                                               "voice": "zh-CN-YunxiNeural"}},
        }
        before = copy.deepcopy(registry["voice_bindings"])
        aggregate = appearance.aggregate_book([record(0, [seg("甲", "一二")])])
        result = appearance.apply_to_registry(registry, "book_a", aggregate, ["随从"])
        self.assertEqual(result["unresolved"], [])
        self.assertEqual(registry["identities"]["甲"]["appearance_by_book"]["book_a"]["total_chars"], 2)
        self.assertEqual(registry["voice_bindings"], before)
        self.assertIn("book_a", registry["appearance_meta"]["books"])

    def test_unmatched_speaker_is_reported_not_created(self):
        registry = {"identities": {}, "character_aliases": {}, "voice_bindings": {}}
        aggregate = appearance.aggregate_book([record(0, [seg("陌生人", "一二")])])
        result = appearance.apply_to_registry(registry, "book_a", aggregate, ["随从"])
        self.assertEqual(result["unresolved"], ["陌生人"])
        self.assertEqual(registry["identities"], {})

    def test_generic_name_falls_back_to_existing_identity(self):
        registry = {"identities": {"随从@unknown": {"canonical_name": "随从", "generic": True,
                                                   "books": []}},
                    "character_aliases": {}}
        cid, how = appearance.resolve_identity("随从", "book_a", registry, ["随从"])
        self.assertEqual(cid, "随从@unknown")
        self.assertIn(how, ("canonical", "generic_name"))


class TierTest(unittest.TestCase):
    def cfg(self):
        cfg = copy.deepcopy(CFG)
        conf = cfg["casting"]["voice_rules"]["tiering"]
        conf.update({"enabled": True, "main_min_chapters": 2,
                     "main_min_lines": 40, "main_min_share": 0.08})
        reclaim = cfg["casting"]["voice_rules"]["reclaim"]
        reclaim.update({"enabled": True, "idle_chapters": 4,
                        "min_lines_keep": 20, "max_share": 0.05})
        return cfg

    def test_main_requires_chapters_and_lines_or_share(self):
        cfg = self.cfg()
        stat = {"chapters_seen": 2, "total_lines": 41, "share": 0.02}
        self.assertEqual(appearance.tier_of(stat, cfg), "main")
        stat = {"chapters_seen": 1, "total_lines": 200, "share": 0.5}
        self.assertEqual(appearance.tier_of(stat, cfg), "support")
        stat = {"chapters_seen": 2, "total_lines": 10, "share": 0.09}
        self.assertEqual(appearance.tier_of(stat, cfg), "main")
        stat = {"chapters_seen": 2, "total_lines": 9, "share": 0.01}
        self.assertEqual(appearance.tier_of(stat, cfg), "minor")

    def test_generic_is_always_generic(self):
        cfg = self.cfg()
        stat = {"chapters_seen": 99, "total_lines": 999, "share": 0.9}
        self.assertEqual(appearance.tier_of(stat, cfg, generic=True), "generic")

    def test_dormant_only_for_low_frequency_minor(self):
        cfg = self.cfg()
        stat = {"idle_chapters": 4, "total_lines": 5, "share": 0.01}
        self.assertTrue(appearance.dormant_candidate(stat, cfg, "minor"))
        self.assertFalse(appearance.dormant_candidate(stat, cfg, "main"))
        self.assertFalse(appearance.dormant_candidate(stat, cfg, "support"))
        self.assertFalse(appearance.dormant_candidate(
            {"idle_chapters": 3, "total_lines": 5, "share": 0.01}, cfg, "minor"))
        self.assertFalse(appearance.dormant_candidate(
            {"idle_chapters": 9, "total_lines": 20, "share": 0.01}, cfg, "minor"))
        self.assertFalse(appearance.dormant_candidate(
            {"idle_chapters": 9, "total_lines": 5, "share": 0.06}, cfg, "minor"))


if __name__ == "__main__":
    unittest.main()
