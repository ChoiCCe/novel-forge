"""旁白策略与定稿锁的单元测试（不联网、不写 registry）。"""
from __future__ import annotations

import copy
import tempfile
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import (caster, dialect, director, narrator_lock, script, tts_router,  # noqa: E402
                  voice_policy, voices_catalog)

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
NARRATOR = narrator_lock.effective(CFG)


def clone_cfg():
    return copy.deepcopy(CFG)


def profile(name, rate, pitch="+0Hz", volume="+0%", voice="zh-CN-XiaoxiaoNeural",
            lines=3, source="auto", binding="lock:test"):
    return {
        "key": name, "name": name, "age_stage": "青年", "edge_voice": voice,
        "rate": rate, "pitch": pitch, "volume": volume, "line_count": lines,
        "voice_source": source, "binding_id": binding,
    }


def narrator(rate=None, pitch=None, volume=None):
    return {"key": "旁白", "name": "旁白", "edge_voice": NARRATOR["voice"],
            "rate": rate or NARRATOR["rate"], "pitch": pitch or NARRATOR["pitch"],
            "volume": volume or NARRATOR["volume"],
            "voice_source": "config", "binding_id": None}


class PolicySwitchTest(unittest.TestCase):
    def test_default_is_independent_first(self):
        self.assertEqual(voice_policy.policy_of(CFG), "independent_first")

    def test_forbid_fallback_when_strategy_missing(self):
        cfg = clone_cfg()
        cfg["casting"]["voice_rules"].pop("strategy")
        cfg["casting"]["voice_rules"].pop("narrator_voice_policy", None)
        cfg["casting"]["voice_rules"]["forbid_narrator_voice"] = True
        self.assertEqual(voice_policy.policy_of(cfg), "forbid")

    def test_forbid_report_marks_same_voice_not_ok(self):
        cfg = clone_cfg()
        cfg["casting"]["voice_rules"]["strategy"] = "forbid"
        cast = {"a": profile("甲", "-6%")}
        report = voice_policy.plan(cast, narrator(), cfg)
        self.assertEqual(report["policy"], "forbid")
        self.assertEqual(report["same_voice_roles"], 1)
        self.assertEqual(report["compliant"], 0)
        self.assertEqual(report["adjustments"], [])

    def test_label_of(self):
        self.assertEqual(voice_policy.label_of("zh-CN-XiaoyiNeural",
                                               "zh-CN-XiaoxiaoNeural"), "independent")
        self.assertEqual(voice_policy.label_of("zh-CN-XiaoxiaoNeural",
                                               "zh-CN-XiaoxiaoNeural"), "voice_borrowed")
        self.assertEqual(voice_policy.label_of("zh-CN-XiaoxiaoNeural",
                                               "zh-CN-XiaoxiaoNeural",
                                               unknown=True), "unknown_fallback")


class ThresholdTest(unittest.TestCase):
    def test_single_dimension_exact_min_still_fails_distance(self):
        # rate 差 4pp（刚好 min_rate_pp），但整体距离 1.0 < 1.4
        a = voice_policy.params({"rate": "-6%", "pitch": "+0Hz", "volume": "+0%"})
        b = voice_policy.params({"rate": "-10%", "pitch": "+0Hz", "volume": "+0%"})
        self.assertAlmostEqual(voice_policy.distance(a, b, CFG), 1.0, places=2)
        self.assertFalse(voice_policy.pair_ok(a, b, CFG))

    def test_two_dimensions_pass(self):
        # rate 4pp + pitch 4Hz：距离 sqrt(1+1)=1.41 ≥ 1.4
        a = voice_policy.params({"rate": "-6%", "pitch": "+0Hz", "volume": "+0%"})
        b = voice_policy.params({"rate": "-10%", "pitch": "-4Hz", "volume": "+0%"})
        self.assertTrue(voice_policy.pair_ok(a, b, CFG))

    def test_auto_fix_clustered_roles(self):
        cfg = clone_cfg()
        cast = {
            "甲": profile("甲", "-10%", lines=5),
            "乙": profile("乙", "-11%", lines=4),
            "丙": profile("丙", "-9%", lines=3),
        }
        report = voice_policy.plan(cast, narrator(), cfg)
        self.assertEqual(report["same_voice_roles"], 3)
        self.assertEqual(report["borrowed_roles"], 3)
        self.assertEqual(report["param_fix"], 3)
        self.assertEqual(report["unresolved"], 0)
        params = {row["key"]: row for row in report["rows"]}
        fixed = [{"rate": params[key]["rate"], "pitch": params[key]["pitch"],
                  "volume": params[key]["volume"]} for key in ("甲", "乙", "丙")]
        numeric = [voice_policy.params(item) for item in fixed]
        narrator_params = voice_policy.params(narrator())
        for item in numeric:
            self.assertTrue(voice_policy.pair_ok(item, narrator_params, cfg))
        for i in range(len(numeric)):
            for j in range(i + 1, len(numeric)):
                self.assertTrue(voice_policy.pair_ok(numeric[i], numeric[j], cfg))
        self.assertTrue(all(params[key]["param_fix"] for key in params))
        self.assertTrue(all(params[key]["label"] == "voice_borrowed" for key in params))

    def test_manual_conflict_not_auto_fixed(self):
        cfg = clone_cfg()
        cast = {"甲": profile("甲", NARRATOR["rate"], source="manual")}
        report = voice_policy.plan(cast, narrator(), cfg)
        self.assertEqual(report["manual_conflict"], 1)
        self.assertEqual(report["param_fix"], 0)
        self.assertTrue(report["rows"][0]["manual_conflict"])


class NarratorLockTest(unittest.TestCase):
    def test_unlocked_passes(self):
        cfg = clone_cfg()
        cfg["casting"]["narrator"]["locked"] = False
        state = narrator_lock.check(cfg)
        self.assertFalse(state["locked"])

    def test_locked_mismatch_reports_field(self):
        cfg = clone_cfg()
        values = narrator_lock.normalize(narrator_lock.effective(cfg))
        cfg["casting"]["narrator"].update({
            "locked": True,
            "fingerprint": narrator_lock.fingerprint(values),
            "locked_values": values,
        })
        self.assertTrue(narrator_lock.check(cfg)["locked"])
        cfg["caster"]["narrator_rate"] = "-10%"
        with self.assertRaises(narrator_lock.NarratorLockError) as ctx:
            narrator_lock.check(cfg)
        self.assertIn("rate", str(ctx.exception))
        self.assertIn(values["rate"], str(ctx.exception))
        self.assertIn("-10%", str(ctx.exception))

    def test_override_allows_changed_config(self):
        cfg = clone_cfg()
        values = narrator_lock.normalize(narrator_lock.effective(cfg))
        cfg["casting"]["narrator"].update({
            "locked": True,
            "fingerprint": narrator_lock.fingerprint(values),
            "locked_values": values,
        })
        cfg["caster"]["narrator_pitch"] = "-3Hz"
        state = narrator_lock.check(cfg, override=True)
        self.assertTrue(state["override"])
        self.assertIn("pitch", state["diff"])

    def test_lock_without_fingerprint_errors(self):
        cfg = clone_cfg()
        cfg["casting"]["narrator"]["locked"] = True
        cfg["casting"]["narrator"]["fingerprint"] = ""
        with self.assertRaises(narrator_lock.NarratorLockError):
            narrator_lock.check(cfg)


class UnknownSpeakerTest(unittest.TestCase):
    def test_unknown_uses_narrator_voice_with_own_params(self):
        profile_ = tts_router.unknown_speaker_profile(CFG, narrator(), "张三")
        self.assertEqual(profile_["edge_voice"], "zh-CN-XiaoxiaoNeural")
        self.assertEqual(profile_["rate"], "-2%")
        self.assertEqual(profile_["pitch"], "-2Hz")
        self.assertEqual(profile_["volume"], "+0%")
        self.assertEqual(profile_["prosody_source"], "unknown_speaker")
        self.assertEqual(profile_["key"], "unknown:张三")


class DialectEvidenceTest(unittest.TestCase):
    def test_strong_keyword_creates_evidence(self):
        record = dialect.detect_evidence("他一口粤语，改不过来", CFG)
        self.assertEqual(record["dialect"], "粤语")
        self.assertGreaterEqual(record["confidence"], 0.6)
        self.assertIn("粤语", record["evidence"])
        self.assertEqual(record["method"], "keyword_scan")

    def test_weak_keyword_below_threshold_is_no_evidence(self):
        # “台湾”只是弱证据 0.50，低于 min_confidence 0.60
        self.assertIsNone(dialect.detect_evidence("他来自台湾", CFG))

    def test_persona_requires_quoted_evidence(self):
        self.assertIsNone(dialect.from_persona(
            {"dialect": "粤语", "dialect_confidence": 0.9}, CFG))
        record = dialect.from_persona(
            {"dialect": "粤语", "dialect_confidence": 0.9,
             "dialect_evidence": "他说一口粤语"}, CFG)
        self.assertIsNotNone(record)
        self.assertEqual(record["method"], "llm_persona")

    def test_manual_evidence_is_valid(self):
        record = dialect.manual_evidence("台湾腔", "WebUI 指定", 3)
        self.assertTrue(dialect.valid(record, CFG))
        self.assertEqual(record["method"], "manual")
        self.assertEqual(record["chapter"], 3)


class LadderTest(unittest.TestCase):
    """四级阶梯：空池时也绝不跨到方言声线。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="novel-forge-test-")
        self.cfg = clone_cfg()
        self.cfg["casting"]["catalog"]["path"] = str(ROOT / "voices" / "catalog.edge.json")
        rules = self.cfg["casting"]["voice_rules"]
        rules["pool_source"] = "static"
        rules["voice_pools"] = {stage: {"男": [], "女": [], "未知": []}
                                for stage in ("少年", "青年", "中年", "老年")}
        self.caster = caster.Caster(self.cfg, None, Path(self.tmp.name))
        self.caster._active_cids = set()

    def tearDown(self):
        self.tmp.cleanup()

    def _ident(self, name, gender="男", generic=False):
        ident = self.caster._identity(name, create=True, gender=gender)
        ident["gender"] = gender
        ident["generic"] = generic
        self.caster._active_cids.add(ident["id"])
        return ident

    def _bind(self, ident, stage, voice):
        return self.caster._write_binding(ident["id"], stage, voice,
                                          source="auto", updated_by="test")

    def test_no_evidence_never_accented_when_pool_empty(self):
        ident = self._ident("无籍贯角色")
        alloc = self.caster._allocate_ladder(ident, "男", "青年", "普通口音")
        self.assertEqual(alloc["level"], 3)
        self.assertEqual(alloc["voice"], self.caster.narrator_voice)
        self.assertEqual(alloc["accent"], "")

    def test_weak_evidence_still_never_accented(self):
        ident = self._ident("疑似台湾角色", gender="女")
        alloc = self.caster._allocate_ladder(ident, "女", "青年", "她来自台湾")
        self.assertEqual(alloc["level"], 3)
        self.assertEqual(alloc["voice"], self.caster.narrator_voice)
        self.assertEqual(ident.get("dialect_profile"), None)

    def test_evidence_selects_accented_voice_and_records_evidence(self):
        ident = self._ident("台客", gender="女")
        alloc = self.caster._allocate_ladder(ident, "女", "青年", "她一口台湾腔，改不过来")
        self.assertEqual(alloc["level"], 1)
        self.assertTrue(voices_catalog.is_accented(alloc["voice"], self.caster.catalog))
        self.assertEqual(alloc["accent"], "台湾腔")
        self.assertTrue(ident["dialect_profile"]["evidence"])
        self.assertEqual(ident["dialect_profile"]["method"], "keyword_scan")

    def test_reuse_non_narrator_before_borrowing_narrator(self):
        rules = self.cfg["casting"]["voice_rules"]
        rules["voice_pools"]["青年"]["男"] = ["zh-CN-YunxiNeural"]
        first = self._ident("甲")
        alloc_a = self.caster._allocate_ladder(first, "男", "青年")
        self.assertEqual(alloc_a["level"], 1)
        self._bind(first, "青年", alloc_a["voice"])
        second = self._ident("乙")
        alloc_b = self.caster._allocate_ladder(second, "男", "青年")
        self.assertEqual(alloc_b["level"], 2)
        self.assertEqual(alloc_b["voice"], alloc_a["voice"])

    def test_unknown_or_generic_is_level4(self):
        ident = self._ident("随从", generic=True)
        alloc = self.caster._allocate_ladder(ident, "男", "青年")
        self.assertEqual(alloc["level"], 4)
        self.assertEqual(alloc["voice"], self.caster.narrator_voice)

    def test_binding_level_recomputes_after_sharing_ends(self):
        first = self._ident("甲")
        second = self._ident("乙")
        voice = "zh-CN-YunxiNeural"
        bind_a = self._bind(first, "青年", voice)
        self._bind(second, "青年", voice)
        self.assertEqual(self.caster._binding_level(first, bind_a), 2)
        bind_a["voice"] = "zh-CN-YunyangNeural"
        self.assertEqual(self.caster._binding_level(first, bind_a), 1)


class LLMParamTest(unittest.TestCase):
    def test_schema_exposes_no_final_prosody_params(self):
        import json
        payload = json.dumps([script.SEGMENTS_SCHEMA, director.DIRECTIONS_SCHEMA],
                             ensure_ascii=False)
        for key in ("rate", "pitch", "volume_pct", "hz"):
            self.assertNotIn(key, payload.lower())
        self.assertNotIn("最终参数", payload)

    def test_prompts_forbid_numeric_params(self):
        self.assertIn("禁止输出 rate/pitch/volume", script.SYSTEM_PROMPT)
        self.assertIn("禁止输出 rate/pitch/volume", director.DIRECTOR_SYSTEM)


if __name__ == "__main__":
    unittest.main()
