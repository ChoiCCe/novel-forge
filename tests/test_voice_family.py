"""三层声线模型：family 迁移 / 年龄曲线 / overlay / 频率分配抢占 / reclaim / LLM 闸门。"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import tts_router, voice_family as vf  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def cfg_on(**overrides) -> dict:
    """在配置副本里把三层模型的开关打开；不写回 config.yaml。"""
    cfg = copy.deepcopy(CFG)
    rules = cfg.setdefault("casting", {}).setdefault("voice_rules", {})
    for key in ("family", "age_curve", "overlay", "allocation", "reclaim", "llm"):
        rules.setdefault(key, {})["enabled"] = True
    rules["allocation"].setdefault("preempt", {})["enabled"] = True
    for key, value in overrides.items():
        block, _, field = key.partition("__")
        rules.setdefault(block, {})[field] = value
    return cfg


def cfg_relax(**overrides) -> dict:
    """三层模型 + 同族近龄放宽的配置副本。"""
    cfg = cfg_on()
    rules = cfg["casting"]["voice_rules"]
    rules["same_family_relax"] = dict(rules.get("same_family_relax") or {})
    rules["same_family_relax"]["enabled"] = True
    rules["same_family_relax"].update(overrides)
    return cfg


def flatten(value) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, sub in value.items():
            keys.append(str(key))
            keys.extend(flatten(sub))
    elif isinstance(value, list):
        for sub in value:
            keys.extend(flatten(sub))
    return keys


def identity(cid: str, stage: str, books=("book_x",), note: str = "") -> dict:
    return {"id": cid, "canonical_name": cid, "books": list(books),
            "stage_profiles": {stage: {"age_note": note}}}


def binding(cid: str, stage: str, voice: str, rate="+0%", pitch="+0Hz",
            volume="+0%") -> dict:
    return {"id": f"lock:{cid}|{stage}", "character_id": cid, "age_stage": stage,
            "voice": voice, "rate": rate, "pitch": pitch, "volume": volume}


def family(cid: str, voice: str, state="owned", anchor_age=26, params=None,
           **extra) -> dict:
    row = {"id": f"family:{cid}", "character_id": cid, "book_id": "book_x",
           "voice": voice, "anchor_stage": "青年", "anchor_age": anchor_age,
           "anchor_params": params or {"rate": 0.0, "pitch": 0.0, "volume": 0.0},
           "state": state, "lease": None, "promotion": None}
    row.update(extra)
    return row


def stat(chars=10, share=0.1, recent=0.1, last=0) -> dict:
    return {"chars_all": chars, "share_all": share, "share_recent": recent,
            "last_chapter": last, "chapters_seen": [last]}


class ConfigTest(unittest.TestCase):
    def test_defaults_off_and_new_keys(self):
        cfg = vf.config(CFG)
        for block in ("family", "age_curve", "overlay", "allocation", "reclaim", "llm"):
            self.assertFalse(cfg[block]["enabled"], block)
        self.assertFalse(cfg["allocation"]["preempt"]["enabled"])
        self.assertEqual(cfg["overlay"]["reader_perspective"], "reader")
        self.assertAlmostEqual(cfg["allocation"]["w_all"], 0.7)
        self.assertAlmostEqual(cfg["allocation"]["w_recent"], 0.3)
        self.assertEqual(cfg["allocation"]["recent_window_chapters"], 20)
        self.assertEqual(cfg["reclaim"]["idle_chapters"], 5)

    def test_removed_switch_keys_absent(self):
        keys = flatten(CFG)
        for gone in ("one_switch_allowed", "minor_to_adult", "adult_to_old",
                     "switch_priority", "switch_min_age", "switch_requires_manual",
                     "derived"):
            self.assertNotIn(gone, keys)


class FrequencyTest(unittest.TestCase):
    def test_weighted_score(self):
        cfg = cfg_on()
        scores = vf.frequency_scores(
            {"A": stat(share=0.6, recent=0.4), "B": stat(share=0.4, recent=0.6)}, cfg)
        self.assertAlmostEqual(scores["A"]["score"], 0.7 * 0.6 + 0.3 * 0.4, places=6)
        self.assertAlmostEqual(scores["B"]["score"], 0.7 * 0.4 + 0.3 * 0.6, places=6)
        self.assertGreater(scores["A"]["score"], scores["B"]["score"])
        # 累计 70% 的权重让“全书主线”压过“近期临时戏份”
        self.assertAlmostEqual(scores["A"]["score"] - scores["B"]["score"], 0.08, places=6)

    def test_cast_order_is_frequency_desc(self):
        scores = {"A": {"score": 0.2}, "B": {"score": 0.5}}
        cast = {"b": {"character_id": "B"}, "a": {"character_id": "A"}}
        self.assertEqual(vf.cast_order(cast, scores), ["b", "a"])


class MigrateTest(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_on()
        self.registry = {
            "identities": {"A": identity("A", "青年"), "B": identity("B", "中年")},
            "voice_bindings": {
                "lock:A|青年": binding("A", "青年", "zh-CN-YunxiNeural", "-2%", "+1Hz"),
                "lock:B|中年": binding("B", "中年", "zh-CN-YunyangNeural", "-3%", "-2Hz", "-1%"),
            },
        }

    def test_migrate_idempotent_and_readonly_on_dry_run(self):
        before = json.dumps(self.registry, ensure_ascii=False, sort_keys=True)
        dry = vf.migrate_families(self.registry, "book_x", {}, self.registry["identities"],
                                  self.cfg, 0, dry_run=True)
        self.assertEqual(len(dry["created"]), 2)
        self.assertEqual(json.dumps(self.registry, ensure_ascii=False, sort_keys=True), before)
        first = vf.migrate_families(self.registry, "book_x", {}, self.registry["identities"],
                                    self.cfg, 0, dry_run=False)
        self.assertEqual(len(first["created"]), 2)
        second = vf.migrate_families(self.registry, "book_x", {}, self.registry["identities"],
                                     self.cfg, 0, dry_run=False)
        self.assertEqual(second["created"], [])
        self.assertEqual(sorted(second["existing"]), ["A", "B"])
        # 旧锁保留，供旧章复现
        self.assertIn("lock:A|青年", self.registry["voice_bindings"])

    def test_migrated_params_equal_old_binding(self):
        vf.migrate_families(self.registry, "book_x", {}, self.registry["identities"],
                            self.cfg, 0, dry_run=False)
        fam = vf.current_family(self.registry, self.cfg, "A")
        self.assertEqual(fam["voice"], "zh-CN-YunxiNeural")
        self.assertEqual(fam["anchor_params"],
                         vf.binding_params(self.registry["voice_bindings"]["lock:A|青年"]))
        # 锚点年龄渲染必须逐字等于旧绑定（零跳变）
        render = vf.render_chapter(self.registry, "book_x", 0, "A", fam["anchor_age"], self.cfg)
        self.assertEqual(render["params"], fam["anchor_params"])
        self.assertEqual(render["anchor_age"], 26)

    def test_stage_note_digit_age_wins(self):
        self.registry["identities"]["B"]["stage_profiles"]["中年"]["age_note"] = "四十岁"
        vf.migrate_families(self.registry, "book_x", {}, self.registry["identities"],
                            self.cfg, 0, dry_run=False)
        self.assertEqual(vf.current_family(self.registry, self.cfg, "B")["anchor_age"], 46)


class CurveTest(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_on()

    def test_points_and_monotonic(self):
        self.assertEqual(vf.curve_point(14, self.cfg),
                         {"rate_pct": 3.0, "pitch_hz": 4.0, "volume_pct": 0.0})
        self.assertEqual(vf.curve_point(100, self.cfg),
                         {"rate_pct": -12.0, "pitch_hz": -4.0, "volume_pct": -5.0})
        previous = vf.curve_point(14, self.cfg)
        for age in range(15, 101):
            now = vf.curve_point(age, self.cfg)
            self.assertLessEqual(now["rate_pct"], previous["rate_pct"] + 1e-9)
            self.assertLessEqual(now["pitch_hz"], previous["pitch_hz"] + 1e-9)
            self.assertLessEqual(now["volume_pct"], previous["volume_pct"] + 1e-9)
            previous = now

    def test_pitch_never_dips_below_minus_four(self):
        for age in (68, 85, 120):
            params = vf.curve_params(age, 26, {"rate": 0, "pitch": 0, "volume": 0}, self.cfg)
            self.assertGreaterEqual(params["pitch"], -4.0)

    def test_clamp_on_extreme_anchor(self):
        params = vf.curve_params(68, 26, {"rate": -40, "pitch": -30, "volume": -20}, self.cfg)
        self.assertEqual(params["rate"], -25.0)
        self.assertEqual(params["pitch"], -4.0)
        self.assertEqual(params["volume"], -10.0)

    def test_per_chapter_caps(self):
        target = {"rate": -10.0, "pitch": -4.0, "volume": -3.0}
        one = vf.smooth_params({"rate": 0, "pitch": 0, "volume": 0}, target, 1, None, self.cfg)
        self.assertAlmostEqual(one["rate"], -0.25)
        self.assertAlmostEqual(one["pitch"], -0.5)
        self.assertAlmostEqual(one["volume"], -0.2)
        three = vf.smooth_params({"rate": 0, "pitch": 0, "volume": 0}, target, 3, None, self.cfg)
        self.assertAlmostEqual(three["rate"], -0.75)
        self.assertAlmostEqual(three["pitch"], -1.5)
        self.assertAlmostEqual(three["volume"], -0.6)

    def test_jump_reason_bypasses_caps(self):
        jump = vf.smooth_params({"rate": 0, "pitch": 0, "volume": 0},
                                {"rate": -10.0, "pitch": -4.0, "volume": -3.0},
                                1, "time_skip", self.cfg)
        self.assertEqual(jump["rate"], -10.0)
        self.assertEqual(jump["pitch"], -4.0)
        bad = vf.smooth_params({"rate": 0, "pitch": 0, "volume": 0},
                               {"rate": -10.0, "pitch": -4.0, "volume": -3.0},
                               1, "random", self.cfg)
        self.assertAlmostEqual(bad["rate"], -0.25)


class OverlayTest(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_on()

    def test_expiry_and_no_slot(self):
        registry = {}
        row = vf.add_overlay(registry, "A", {"rate": -6, "pitch": -3, "volume": -4},
                             3, self.cfg, reason="whisper", scope="chapter", dry_run=False)
        self.assertEqual(row["until_chapter"], 3)
        self.assertEqual(row["perspective"], "reader")
        self.assertNotIn("voice_families", registry)
        self.assertEqual(len(vf.active_overlays(registry, "A", 3)), 1)
        self.assertEqual(vf.active_overlays(registry, "A", 4), [])
        removed = vf.expire_overlays(registry, 4, dry_run=False)
        self.assertEqual(len(removed), 1)
        self.assertEqual(registry["voice_overlays"], {})

    def test_clamp_and_stack_with_lent_family(self):
        registry = {"voice_families": {"family:A": family("A", "zh-CN-YunxiNeural", "lent")}}
        row = vf.add_overlay(registry, "A", {"rate": -30, "pitch": -9},
                             5, self.cfg, reason="disguise", scope="arc",
                             until_chapter=8, dry_run=False)
        self.assertEqual(row["deltas"]["rate"], -12.0)
        self.assertEqual(row["deltas"]["pitch"], -6.0)
        params = vf.apply_overlay({"rate": 0, "pitch": 0, "volume": 0},
                                  vf.active_overlays(registry, "A", 6), self.cfg)
        self.assertEqual(params["rate"], -12.0)
        # overlay 不占槽位，也不改 family 状态/声线
        self.assertEqual(registry["voice_families"]["family:A"]["state"], "lent")
        self.assertEqual(registry["voice_families"]["family:A"]["voice"], "zh-CN-YunxiNeural")

    def test_overlay_does_not_survive_without_expiry_call(self):
        registry = {}
        vf.add_overlay(registry, "A", {"rate": -4}, 10, self.cfg,
                       reason="injury", dry_run=False)
        self.assertEqual(vf.active_overlays(registry, "A", 11), [])
        self.assertEqual(len(registry["voice_overlays"]), 1)  # 需显式到期清理


class PreemptTest(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_on()
        self.registry = {
            "voice_families": {
                "family:A": family("A", ""),
                "family:B": family("B", "zh-CN-YunxiNeural"),
            },
            "book_settings": {"book_x": {}},
        }
        self.stats = {"A": stat(share=0.50, recent=0.50),
                      "B": stat(share=0.20, recent=0.20)}

    def run_chapter(self, chapter, tiers=None, stats=None):
        plan = vf.plan_preemptions(self.registry, "book_x", stats or self.stats,
                                   chapter, self.cfg, "zh-CN-XiaoxiaoNeural",
                                   tiers=tiers or {"A": "support", "B": "support"})
        vf.apply_preemptions(self.registry, "book_x", plan, self.cfg, dry_run=False)
        return plan

    def test_only_boundary(self):
        plan = vf.plan_preemptions(self.registry, "book_x", self.stats, 0, self.cfg,
                                   "zh-CN-XiaoxiaoNeural",
                                   tiers={"A": "support", "B": "support"}, boundary=False)
        self.assertEqual(plan["preemptions"], [])

    def test_main_never_preempts_nor_is_preempted(self):
        self.assertEqual(self.run_chapter(0, tiers={"A": "main", "B": "support"})["preemptions"], [])
        self.assertEqual(self.run_chapter(0, tiers={"A": "support", "B": "main"})["preemptions"], [])

    def test_confirm_then_apply_and_audit(self):
        first = self.run_chapter(0)
        self.assertEqual([row["action"] for row in first["preemptions"]], ["pending"])
        self.assertEqual(self.registry["voice_families"]["family:B"]["state"], "owned")
        second = self.run_chapter(1)
        actions = [row["action"] for row in second["preemptions"]]
        self.assertEqual(actions, ["preempt"])
        self.assertEqual(self.registry["voice_families"]["family:A"]["voice"], "zh-CN-YunxiNeural")
        self.assertEqual(self.registry["voice_families"]["family:A"]["state"], "owned")
        self.assertEqual(self.registry["voice_families"]["family:B"]["voice"], "")
        self.assertEqual(self.registry["voice_families"]["family:B"]["state"], "shared")
        audit = self.registry["family_audit"][-1]
        self.assertEqual((audit["winner"], audit["loser"]), ("A", "B"))
        self.assertAlmostEqual(audit["gap"], 0.30)
        self.assertEqual(audit["ratio"], 2.5)

    def test_overlap_gate_never_flip_flops(self):
        stats = {"A": stat(share=0.22, recent=0.22), "B": stat(share=0.20, recent=0.20)}
        for chapter in range(0, 6):
            plan = self.run_chapter(chapter, stats=stats)
            self.assertEqual(plan["preemptions"], [], f"chapter {chapter}")

    def test_ratio_gate_blocks(self):
        stats = {"A": stat(share=0.30, recent=0.30), "B": stat(share=0.22, recent=0.22)}
        plan = self.run_chapter(0, stats=stats)
        self.assertEqual(plan["preemptions"], [])

    def test_cooldown_after_execution(self):
        self.registry["book_settings"]["book_x"]["preempt_tracker"] = {
            "A->B": {"streak": 2, "last_chapter": 4, "executed_chapter": 4}}
        plan = self.run_chapter(5)
        self.assertEqual(plan["preemptions"], [])

    def test_lent_incumbent_skipped_by_state_machine(self):
        self.registry["voice_families"]["family:B"]["state"] = "lent"
        plan = self.run_chapter(0)
        self.assertEqual(plan["preemptions"], [])


class RelaxTest(unittest.TestCase):
    """同族近龄放宽：只降“同声线 + 年龄接近”的距离阈值，并保留绝对下限。"""

    def setUp(self):
        # 楼础 18 岁 / 马维 14 岁，同一 Yunxia 声线；距离约 0.86（线上判撞）
        self.left = {"rate": -2.0, "pitch": -1.3, "volume": 0.0}
        self.right = {"rate": -5.0, "pitch": 0.0, "volume": -2.0}

    def verdict(self, cfg, age_a=18, age_b=14, voice_a="zh-CN-YunxiaNeural",
                voice_b="zh-CN-YunxiaNeural", left=None, right=None):
        return vf.collision_verdict(left or self.left, right or self.right,
                                    voice_a=voice_a, voice_b=voice_b,
                                    age_a=age_a, age_b=age_b, cfg=cfg)

    def test_disabled_keeps_strict_threshold(self):
        result = self.verdict(cfg_on())
        self.assertEqual(result["rule"], "base")
        self.assertEqual(result["threshold"], 1.4)
        self.assertFalse(result["ok"])

    def test_close_age_same_voice_relaxed(self):
        result = self.verdict(cfg_relax())
        self.assertEqual(result["rule"], "same_family_close_age")
        self.assertAlmostEqual(result["threshold"], 0.75)
        self.assertAlmostEqual(result["base_threshold"], 1.4)
        self.assertTrue(result["relaxed"])
        self.assertEqual(result["age_gap"], 4)
        self.assertTrue(result["ok"])

    def test_curve_ramps_back_between_six_and_twelve(self):
        ramp = self.verdict(cfg_relax(), age_a=18, age_b=10)   # gap 8
        self.assertAlmostEqual(ramp["threshold"], 0.967, places=3)
        self.assertFalse(ramp["ok"])
        far = self.verdict(cfg_relax(), age_a=30, age_b=14)    # gap 16
        self.assertEqual(far["rule"], "base")
        self.assertEqual(far["threshold"], 1.4)

    def test_different_voice_never_relaxed(self):
        result = self.verdict(cfg_relax(), voice_b="zh-CN-YunyangNeural")
        self.assertEqual(result["rule"], "base")
        self.assertEqual(result["threshold"], 1.4)
        self.assertFalse(result["ok"])

    def test_unknown_age_never_relaxed(self):
        result = self.verdict(cfg_relax(), age_b=None)
        self.assertEqual(result["rule"], "base")
        self.assertFalse(result["ok"])

    def test_floor_blocks_identical_params(self):
        cfg = cfg_relax(relaxed_distance=0.2, floor_distance=0.5)
        same = {"rate": -2.0, "pitch": 0.0, "volume": 0.0}
        result = self.verdict(cfg, age_a=18, age_b=18, left=same, right=dict(same))
        self.assertEqual(result["threshold"], 0.5)   # 0.2 被 floor 抬回 0.5
        self.assertEqual(result["distance"], 0.0)
        self.assertFalse(result["ok"])

    def test_near_identical_still_blocked(self):
        cfg = cfg_relax()
        left = {"rate": -2.0, "pitch": 0.0, "volume": 0.0}
        right = {"rate": -2.4, "pitch": -0.4, "volume": 0.0}
        result = self.verdict(cfg, age_a=18, age_b=18, left=left, right=right)
        self.assertLess(result["distance"], result["threshold"])
        self.assertFalse(result["ok"])

    def test_audit_row_written(self):
        registry = {}
        verdict = self.verdict(cfg_relax())
        row = vf.record_collision_audit(registry, verdict, chapter=1,
                                        left="楼础", right="马维", book_id="book_x")
        self.assertEqual(row["rule"], "same_family_close_age")
        self.assertEqual(row["age_gap"], 4)
        self.assertAlmostEqual(row["threshold"], 0.75)
        self.assertAlmostEqual(row["base_threshold"], 1.4)
        self.assertTrue(row["accepted"])
        self.assertEqual(row["floor"], 0.5)
        self.assertEqual(registry["collision_audit"][0]["chapter"], 1)


class ReclaimTest(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_on()

    def test_threshold_floor_and_double_gate(self):
        stat_row = stat(chars=5, share=0.01, last=0)
        retired = {"status": "retired", "confidence": 0.9, "evidence_quote": "退场"}
        self.assertTrue(vf.dormant_allowed(stat_row, retired, 5, self.cfg)["allowed"])
        self.assertFalse(vf.dormant_allowed(stat_row, retired, 4, self.cfg)["allowed"])
        self.assertFalse(vf.dormant_allowed(stat(chars=20, share=0.01, last=0),
                                            retired, 5, self.cfg)["allowed"])
        self.assertFalse(vf.dormant_allowed(stat(chars=5, share=0.05, last=0),
                                            retired, 5, self.cfg)["allowed"])

    def test_low_confidence_treated_hidden(self):
        stat_row = stat(chars=5, share=0.01, last=0)
        verdict = vf.dormant_allowed(
            stat_row, {"status": "retired", "confidence": 0.5,
                       "evidence_quote": "退场"}, 5, self.cfg)
        self.assertFalse(verdict["allowed"])
        self.assertEqual(verdict["status"], "hidden")

    def test_hidden_and_likely_return_never_lend(self):
        for status in ("hidden", "likely_return", "present"):
            verdict = vf.dormant_allowed(
                stat(chars=5, share=0.01, last=0),
                {"status": status, "confidence": 0.9, "evidence_quote": "退场"},
                9, self.cfg)
            self.assertFalse(verdict["allowed"], status)

    def test_idle_floor_cannot_be_configured_lower(self):
        cfg = cfg_on(reclaim__idle_chapters=2)
        verdict = vf.dormant_allowed(stat(chars=5, share=0.01, last=0),
                                     {"status": "retired", "confidence": 0.9,
                                      "evidence_quote": "退场"}, 3, cfg)
        self.assertEqual(verdict["threshold"], 4)
        self.assertFalse(verdict["allowed"])

    def test_transition_priority(self):
        cfg = cfg_on(reclaim__promotion={"enabled": True})
        fam = {"state": "lent", "lease": {"until_chapter": 5, "lessee": "B"}}
        self.assertEqual(vf.family_transition(fam, 3, cfg, owner_returned=True)["action"], "buyback")
        self.assertEqual(vf.family_transition(fam, 6, cfg)["action"], "lease_expire")
        self.assertEqual(vf.family_transition({"state": "dormant"}, 7, cfg)["action"], "lend")
        promo = {"state": "lent", "retirement": {"status": "retired"},
                 "lease": {"until_chapter": 10, "lease_count": 3, "active_chapters": 12}}
        self.assertEqual(vf.family_transition(promo, 8, cfg)["action"], "promote")


class LLMGateTest(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_on()
        self.body = "楼础18岁了，他握紧了拳头。"
        self.record = {"chapter_index": 1,
                       "segments": [{"type": "narration", "speaker": "旁白",
                                     "text": self.body}]}

    def valid_payload(self):
        return {"judgments": [{
            "character_id": "楼础", "chapter": 1, "age_range": [18, 18],
            "timbre_descriptor": {"register": "mid", "texture": "clear"},
            "confidence": 0.9, "evidence_quote": "楼础18岁了"}]}

    def test_voice_id_rejected(self):
        payload = self.valid_payload()
        payload["judgments"][0]["voice_id"] = "zh-CN-YunxiNeural"
        result = vf.validate_age_payload(payload, self.body, {"楼础"}, self.cfg)
        self.assertFalse(result["ok"])
        self.assertEqual(result["errors"][0]["code"], "VOICE_ID_IMMUTABLE")

    def test_valid_payload_passes(self):
        result = vf.validate_age_payload(self.valid_payload(), self.body, {"楼础"}, self.cfg)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["rows"]), 1)

    def test_evidence_mismatch_rejected(self):
        payload = self.valid_payload()
        payload["judgments"][0]["evidence_quote"] = "原文里没有这句"
        result = vf.validate_age_payload(payload, self.body, {"楼础"}, self.cfg)
        self.assertFalse(result["ok"])
        self.assertEqual(result["errors"][0]["code"], "EVIDENCE_MISMATCH")

    def test_rejected_output_falls_back_with_pending_review(self):
        class FakeClient:
            def __init__(self, payload):
                self.payload = payload
                self.calls = 0

            def chat_json(self, *args, **kwargs):
                self.calls += 1
                return self.payload

        payload = self.valid_payload()
        payload["judgments"][0]["voice"] = "zh-CN-YunxiNeural"
        registry = {}
        result = vf.judge_ages(FakeClient(payload), self.record, {"楼础"},
                               self.cfg, registry=registry, dry_run=False)
        self.assertEqual(result["status"], "fallback")
        self.assertIn(18, [row["age_range"][0] for row in result["rows"]])
        codes = {row["code"] for row in result["pending"]}
        self.assertIn("VOICE_ID_IMMUTABLE", codes)
        self.assertIn("RULE_FALLBACK", codes)
        self.assertTrue(all(row["status"] == "pending_review"
                            for row in registry["pending_review"]))

    def test_low_confidence_goes_pending(self):
        payload = self.valid_payload()
        payload["judgments"][0]["confidence"] = 0.3
        result = vf.validate_age_payload(payload, self.body, {"楼础"}, self.cfg)
        self.assertEqual([row["code"] for row in result["errors"]], ["CONFIDENCE_PENDING"])
        self.assertFalse(vf.judge_ages(object(), self.record, {"楼础"}, self.cfg)["status"] == "ok")


class CacheKeyTest(unittest.TestCase):
    def test_param_changes_produce_new_cache_entry(self):
        voice = {"engine": "edge", "edge_voice": "zh-CN-YunxiNeural",
                 "rate": "+0%", "pitch": "+0Hz", "volume": "+0%"}
        base = tts_router.cache_key("edge", voice, "你好")
        self.assertEqual(base, tts_router.cache_key("edge", dict(voice), "你好"))
        for field, value in (("rate", "+5%"), ("pitch", "-3Hz"), ("volume", "-4%")):
            changed = dict(voice, **{field: value})
            self.assertNotEqual(base, tts_router.cache_key("edge", changed, "你好"), field)


class ScanBookTest(unittest.TestCase):
    def test_scan_counts_dialogue_and_recent_window(self):
        cfg = cfg_on()
        with tempfile.TemporaryDirectory() as tmp:
            book_dir = Path(tmp) / "book_x"
            book_dir.mkdir()
            for chapter in range(4):
                text = "一二三四五" * (chapter + 1)
                record = {
                    "book_id": "book_x", "chapter_index": chapter,
                    "cast": {"speakers": {"甲": "甲_男_青年"},
                             "cast": {"甲_男_青年": {"name": "甲",
                                                    "character_id": "甲",
                                                    "age_band": "青年"}}},
                    "segments": [{"type": "dialogue", "speaker": "甲", "text": text},
                                 {"type": "narration", "speaker": "旁白", "text": "旁白不算"}],
                }
                (book_dir / f"ch{chapter:03d}.json").write_text(
                    json.dumps(record, ensure_ascii=False), encoding="utf-8")
            identities = {"甲": identity("甲", "青年")}
            stats = vf.scan_book(book_dir, 3, identities, cfg)
            row = stats["甲"]
            self.assertEqual(row["chars_all"], 5 * (1 + 2 + 3 + 4))
            # 近 20 章窗口覆盖全部 4 章
            self.assertEqual(row["chars_recent"], row["chars_all"])
            self.assertEqual(row["chapters_seen"], [0, 1, 2, 3])
            self.assertEqual(row["stage_chars"], {"青年": row["chars_all"]})


if __name__ == "__main__":
    unittest.main()
