"""v4 步骤 1：声线目录与失效声线平替的单元测试（不联网）。"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import voices_catalog  # noqa: E402


def catalog() -> dict:
    return json.loads((ROOT / "voices" / "catalog.edge.json").read_text(encoding="utf-8"))


class ResolveTest(unittest.TestCase):
    def test_unavailable_voices_mapped(self):
        cat = catalog()
        r = voices_catalog.resolve_voice("zh-CN-YunyeNeural", cat)
        self.assertTrue(r["fallback"])
        self.assertEqual(r["voice"], "zh-CN-YunjianNeural")
        self.assertEqual(r["legacy_voice"], "zh-CN-YunyeNeural")

        r = voices_catalog.resolve_voice("zh-CN-XiaoshuangNeural", cat)
        self.assertEqual(r["voice"], "zh-CN-XiaoyiNeural")
        self.assertEqual(r["overrides"].get("pitch"), "+10Hz")

    def test_available_voice_untouched(self):
        r = voices_catalog.resolve_voice("zh-CN-YunxiaNeural", catalog())
        self.assertEqual(r["voice"], "zh-CN-YunxiaNeural")
        self.assertFalse(r["fallback"])

    def test_hidden_voice_is_available(self):
        # Xiaoxuan 不在 list-voices，但实测可合成，目录必须标为可用
        self.assertTrue(voices_catalog.is_available("zh-CN-XiaoxuanNeural", catalog()))


class DialectTest(unittest.TestCase):
    def test_filtered_by_gender(self):
        cat = catalog()
        special = {"东北": {"voice": "zh-CN-liaoning-XiaobeiNeural", "gender": "女"}}
        self.assertEqual(voices_catalog.special_voice("东北口音", "男", special, cat), "")
        self.assertEqual(voices_catalog.special_voice("东北口音", "女", special, cat),
                         "zh-CN-liaoning-XiaobeiNeural")

    def test_legacy_string_special_config(self):
        cat = catalog()
        special = {"陕西": "zh-CN-shaanxi-XiaoniNeural"}
        self.assertEqual(voices_catalog.special_voice("陕西话", "男", special, cat), "")
        self.assertEqual(voices_catalog.special_voice("陕西话", "女", special, cat),
                         "zh-CN-shaanxi-XiaoniNeural")


if __name__ == "__main__":
    unittest.main()
