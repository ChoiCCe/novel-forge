"""v4 步骤 2：迁移与分层绑定的单元测试（不联网）。"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import casting  # noqa: E402


CATALOG = json.loads((ROOT / "voices" / "catalog.edge.json").read_text(encoding="utf-8"))
REG = json.loads((ROOT / "voices" / "registry.json").read_text(encoding="utf-8"))


def v1_sample():
    return {
        "characters": {
            "周律_男_中年": {"name": "周律", "gender": "男", "age_stage": "中年",
                             "temperament": "急躁", "rate": "+4%", "pitch": "+2Hz",
                             "volume": "+0%", "first_seen_chapter": 1},
            "周律_男_青年": {"name": "周律", "gender": "男", "age_stage": "青年",
                             "temperament": "急躁", "rate": "+4%", "pitch": "+2Hz",
                             "volume": "+0%", "first_seen_chapter": 1},
            "同学_男_少年": {"name": "同学", "gender": "男", "age_stage": "少年",
                             "temperament": "中性", "rate": "+0%", "pitch": "+0Hz",
                             "volume": "+0%"},
        },
        "locks": {
            "周律|中年": {"edge_voice": "zh-CN-YunxiaNeural", "stage": "中年"},
        },
        "voice_changes": [],
        "voice_map_version": "edge-v5",
    }


class RegistryV2Test(unittest.TestCase):
    def test_live_registry_is_v2_only(self):
        self.assertEqual(REG["schema_version"], 2)
        for key in ("characters", "locks", "voice_changes"):
            self.assertNotIn(key, REG)
        locks = [item for item in REG["voice_bindings"].values()
                 if item.get("scope") == "character_stage" and item.get("locked")]
        self.assertGreaterEqual(len(locks), 3)


class MigrateTest(unittest.TestCase):
    def test_zero_loss_and_stage_merge(self):
        reg, report = casting.migrate_v1(v1_sample(), cfg={})
        self.assertTrue(report["zero_loss"])
        self.assertEqual(reg["schema_version"], 2)
        self.assertEqual(len(reg["identities"]["周律"]["stage_profiles"]), 2)
        self.assertEqual(report["counts"]["characters"], 3)
        self.assertEqual(report["counts"]["stages"], 3)

    def test_generic_identity_scoped(self):
        reg, _ = casting.migrate_v1(v1_sample(), cfg={})
        self.assertIn("同学@unknown", reg["identities"])
        self.assertTrue(reg["identities"]["同学@unknown"]["generic"])

    def test_locks_become_bindings(self):
        reg, _ = casting.migrate_v1(v1_sample(), cfg={})
        binding = reg["voice_bindings"]["lock:周律|中年"]
        self.assertEqual(binding["character_id"], "周律")
        self.assertEqual(binding["scope"], "character_stage")
        self.assertTrue(binding["locked"])

    def test_idempotent(self):
        reg, _ = casting.migrate_v1(v1_sample(), cfg={})
        again, report = casting.migrate_v1(reg, cfg={})
        self.assertTrue(report["skipped"])
        self.assertIs(again, reg)


class PriorityTest(unittest.TestCase):
    def setUp(self):
        self.reg = copy.deepcopy(REG)

    def resolve(self, name, stage, book_id="b1"):
        return casting.resolve_effective(self.reg, name=name, age_stage=stage, book_id=book_id,
                                          config_voice="zh-CN-YunyangNeural", catalog=CATALOG)

    def test_manual_book_stage_beats_global(self):
        self.reg["voice_bindings"] = {
            "global": {"id": "global", "scope": "global", "voice": "zh-CN-YunjianNeural",
                       "source": "manual", "priority": 55},
            "bs": {"id": "bs", "scope": "book_character_stage", "book_id": "b1",
                   "character_id": "周律", "age_stage": "中年", "voice": "zh-CN-YunxiNeural",
                   "source": "manual", "priority": 100},
        }
        self.assertEqual(self.resolve("周律", "中年")["binding_id"], "bs")

    def test_character_stage_beats_older_manual_global(self):
        self.reg["voice_bindings"] = {
            "global": {"id": "global", "scope": "global", "voice": "zh-CN-YunjianNeural",
                       "source": "manual", "priority": casting.PRIORITY["global"]},
            "cs": {"id": "cs", "scope": "character_stage", "character_id": "周律",
                   "age_stage": "中年", "voice": "zh-CN-YunxiNeural",
                   "source": "migrated", "priority": casting.PRIORITY["character_stage"]},
        }
        self.assertEqual(self.resolve("周律", "中年")["binding_id"], "cs")

    def test_wrong_stage_falls_back_to_global(self):
        self.reg["voice_bindings"] = {
            "global": {"id": "global", "scope": "global", "voice": "zh-CN-YunjianNeural",
                       "source": "manual", "priority": casting.PRIORITY["global"]},
            "cs": {"id": "cs", "scope": "character_stage", "character_id": "周律",
                   "age_stage": "少年", "voice": "zh-CN-YunxiaNeural",
                   "source": "migrated", "priority": casting.PRIORITY["character_stage"]},
        }
        self.assertEqual(self.resolve("周律", "中年")["binding_id"], "global")

    def test_chain_is_traceable(self):
        self.reg["voice_bindings"] = {
            "global": {"id": "global", "scope": "global", "voice": "zh-CN-YunjianNeural",
                       "source": "manual", "priority": casting.PRIORITY["global"]},
            "cs": {"id": "cs", "scope": "character_stage", "character_id": "周律",
                   "age_stage": "中年", "voice": "zh-CN-YunxiNeural",
                   "source": "migrated", "priority": casting.PRIORITY["character_stage"]},
        }
        res = self.resolve("周律", "中年")
        self.assertEqual(len(res["chain"]), 2)
        self.assertEqual(res["chain"][0]["id"], "cs")
        self.assertEqual(res["source"], "migrated")

    def test_unavailable_binding_voice_resolved(self):
        self.reg["voice_bindings"] = {
            "old": {"id": "old", "scope": "character_stage", "character_id": "周律",
                    "age_stage": "中年", "voice": "zh-CN-YunyeNeural",
                    "source": "manual", "priority": 50},
        }
        res = self.resolve("周律", "中年")
        self.assertEqual(res["voice"], "zh-CN-YunjianNeural")
        self.assertTrue(res["fallback"])


if __name__ == "__main__":
    unittest.main()
