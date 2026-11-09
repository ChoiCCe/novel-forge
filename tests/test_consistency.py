"""v4 步骤 4：一致性（撞声/漂移）单元测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import consistency  # noqa: E402

CFG = {"casting": {"consistency": {
    "generic_names": ["同学", "随从", "路人", "士兵"],
    "role_weight": {"main_min_lines": 80, "support_min_lines": 10,
                    "weights": {"main": 1.0, "support": 0.5,
                                "minor": 0.2, "generic": 0.1}},
    "collision": {"hint_score": 0.5, "warn_score": 1.0, "block_auto_score": 2.0,
                  "narrator_overlap_score": 1.0, "allow_group_exemption": True},
}}}


def p(name, voice="zh-CN-YunxiaNeural", lines=0, stage="少年", generic=False):
    return {"name": name, "edge_voice": voice, "line_count": lines,
            "age_stage": stage, "generic": generic}


class CollisionTest(unittest.TestCase):
    def test_chapter1_style_four_boys_is_error(self):
        cast = {
            "周律_男_少年": p("周律", lines=100),
            "同学_男_少年": p("同学", lines=20),
            "楼础_男_少年": p("楼础", lines=20),
            "马维_男_少年": p("马维", lines=20),
        }
        rep = consistency.collision_report(cast, "zh-CN-XiaoxiaoNeural", CFG)
        item = rep["by_voice"]["zh-CN-YunxiaNeural"]
        self.assertAlmostEqual(item["score"], 2.5)
        self.assertEqual(item["level"], "error")
        self.assertFalse(consistency.auto_recommend_allowed("zh-CN-YunxiaNeural", rep))

    def test_same_character_stages_not_counted(self):
        cast = {"周律_少年": p("周律", lines=100), "周律_中年": p("周律", lines=100, stage="中年")}
        rep = consistency.collision_report(cast, "", CFG)
        self.assertEqual(rep["by_voice"]["zh-CN-YunxiaNeural"]["score"], 0.0)
        self.assertEqual(rep["by_voice"]["zh-CN-YunxiaNeural"]["level"], "ok")

    def test_generic_group_exemption(self):
        cast = {f"同学{i}": p("同学", lines=5, generic=True) for i in range(25)}
        rep = consistency.collision_report(cast, "", CFG)
        item = rep["by_voice"]["zh-CN-YunxiaNeural"]
        self.assertGreaterEqual(item["score"], 2.0)
        self.assertEqual(item["level"], "warn")
        self.assertTrue(consistency.auto_recommend_allowed("zh-CN-YunxiaNeural", rep))

    def test_narrator_overlap_warns(self):
        cast = {"周律": p("周律", "zh-CN-XiaoxiaoNeural", lines=100)}
        rep = consistency.collision_report(cast, "zh-CN-XiaoxiaoNeural", CFG)
        types = [w["type"] for w in rep["warnings"]]
        self.assertIn("narrator_collision", types)
        self.assertEqual(rep["by_voice"]["zh-CN-XiaoxiaoNeural"]["level"], "ok")


class DriftTest(unittest.TestCase):
    def test_manual_binding_never_drifts(self):
        reg = {"voice_bindings": {
            "manual": {"source": "manual", "voice": "zh-CN-YunxiaNeural", "character_id": "周律"},
        }}
        out = consistency.drift_report(reg, lambda b: {"voice": "zh-CN-YunxiNeural"})
        self.assertEqual(out, [])

    def test_migrated_binding_reports_diff(self):
        reg = {"voice_bindings": {
            "old": {"source": "migrated", "voice": "zh-CN-YunxiaNeural",
                    "character_id": "周律", "age_stage": "少年", "scope": "character_stage"},
        }}
        out = consistency.drift_report(reg, lambda b: {"voice": "zh-CN-YunxiNeural"})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["diff"]["voice"][0], "zh-CN-YunxiaNeural")
        self.assertEqual(out[0]["actions"], ["apply", "keep", "ignore"])


if __name__ == "__main__":
    unittest.main()
