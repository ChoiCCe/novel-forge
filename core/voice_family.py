"""声线三层模型：Identity / Voice Family / Rendition。

- **Identity**：角色 cid，全书唯一，不可变。
- **Voice Family**：`family:{cid}`，一条基声线，一生锁定；唯一占槽位、唯一可外借/归还。
- **Rendition**：由 `params{rate,pitch,volume}` 组成，随年龄走连续曲线；伪装等临时效果
  走 overlay 叠加层，不写绑定、不占槽位、到期自动失效。

硬约束：
1. LLM 碰不到 `voice`（schema 里没有 voice_id；出现即 `VOICE_ID_IMMUTABLE` 整条拒绝）。
2. 曲线是**相对锚点**的：`Δ = C(age) - C(anchor_age)`，迁移当天零跳变。
3. 年龄只做近似：pitch 方向收窄到 ±4Hz（实测再低会让听感“发胖”），主要靠语速与音量。
4. 分配按出场频率，不按先来后到；抢占只在章边界、`main` 既不抢也不被抢、有重叠门槛与冷却。
5. 默认全部关闭；关闭时所有函数只读/只返回计划，不产生副作用。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

STAGES = ("少年", "青年", "中年", "老年")
STAGE_MID_AGE = {"少年": 14, "青年": 26, "中年": 46, "老年": 68}
RETIRE_STATUS = ("retired", "hidden", "likely_return", "present")

DEFAULT_CONFIG = {
    "family": {
        "enabled": False,
        "key_prefix": "family:",
        "anchor_by": "stage_chars",
        "anchor_tiebreak": "earliest",
        "legacy_mode": "readonly",
        "frontier": "last_rendered_chapter",
    },
    # 实测（outputs/novel-forge/v7_voice_age）：pitch 每降 1Hz，谱质心约降 7Hz；
    # 降 11Hz 时质心 −6.5%，听感被判“胖”。因此下压上限收到 −4Hz（质心 −0.8%）。
    "age_curve": {
        "enabled": False,
        "anchors": [
            {"stage": "少年", "age": 14, "rate_pct": 3.0, "pitch_hz": 4.0, "volume_pct": 0.0},
            {"stage": "青年", "age": 26, "rate_pct": 0.0, "pitch_hz": 0.0, "volume_pct": 0.0},
            {"stage": "中年", "age": 46, "rate_pct": -4.0, "pitch_hz": -2.0, "volume_pct": -1.0},
            {"stage": "老年", "age": 68, "rate_pct": -10.0, "pitch_hz": -4.0, "volume_pct": -3.0},
            {"stage": "高龄", "age": 85, "rate_pct": -12.0, "pitch_hz": -4.0, "volume_pct": -5.0},
        ],
        "max_pitch_per_chapter_hz": 0.5,
        "max_rate_per_chapter_pp": 0.25,
        "max_volume_per_chapter_pct": 0.2,
        "jump_reasons": ["time_skip", "flashback_frame", "age_reveal"],
        "clamp": {"rate_pct": [-25.0, 15.0], "pitch_hz": [-4.0, 8.0], "volume_pct": [-10.0, 10.0]},
        "timbre_bias": {
            "register": {"low": -1.0, "mid": 0.0, "high": 1.0},
            "texture": {"hoarse": {"rate_pct": -2.0, "pitch_hz": -1.0},
                        "warm": {"rate_pct": 0.0, "pitch_hz": 0.0},
                        "bright": {"rate_pct": 1.0, "pitch_hz": 1.0},
                        "clear": {"rate_pct": 1.0, "pitch_hz": 0.0}},
        },
    },
    "overlay": {
        "enabled": False,
        "reader_perspective": "reader",  # 默认读者视角：保留本人 family，只叠 overlay
        "default_until": "chapter_end",
        "max_chapters": 60,
        "clamp": {"rate_pct": [-12.0, 12.0], "pitch_hz": [-6.0, 6.0], "volume_pct": [-10.0, 10.0]},
        "reasons": ["disguise", "impersonation", "possession", "injury", "whisper",
                    "shout", "drunk", "sick", "cry", "chorus", "nonhuman", "flashback_age"],
    },
    "allocation": {
        "enabled": False,
        "w_all": 0.7,                    # 累计占比 70%：全书重要度
        "w_recent": 0.3,                 # 近 20 章占比 30%：当前戏份
        "recent_window_chapters": 20,
        "preempt": {
            "enabled": False,
            "min_score_gap": 0.08,       # 频率分绝对差达到 8 个百分点才允许抢
            "min_score_ratio": 1.5,      # 且 challenger ≥ 1.5× incumbent
            "confirm_chapters": 2,       # 连续 2 章超标才执行，防横跳
            "cooldown_chapters": 10,     # 同一对角色 10 章内不重复抢
            "max_per_chapter": 2,
            "exclude_tiers": ["main"],
            "demote_to": "shared",
        },
    },
    "reclaim": {
        "enabled": False,
        "scope": "family",
        "idle_chapters": 5,
        "idle_chapters_floor": 4,
        "min_lines_keep": 20,
        "max_share": 0.05,
        "dormant_to_lent_after": 1,
        "lease_chapters": 20,
        "buyback_first": True,
        "promotion": {
            "enabled": False,
            "min_leases": 3,
            "min_active_chapters": 12,
            "approval": "manual",
            "require_llm_status": "retired",
            "irreversible": True,
        },
    },
    "llm": {
        "enabled": False,
        "min_confidence": 0.60,
        "auto_apply_confidence": 0.80,
        "retry_with_errors": 1,
        "fallback": "rule",
        "cache": True,
    },
    # 同族近龄放宽：只对“同一条声线 + 年龄接近”下调距离阈值，
    # 不同声线 / 跨代仍按 differentiate.min_distance 严格判定。
    "same_family_relax": {
        "enabled": False,
        "same_voice_only": True,
        "max_age_gap": 6,          # 同辈：差 ≤6 岁
        "ramp_age_gap": 12,        # 6–12 岁线性回收到原阈值
        "relaxed_distance": 0.75,
        "floor_distance": 0.50,    # 绝对下限，防止“完全同声”
    },
}

AGE_SCHEMA = {
    "type": "object",
    "required": ["judgments"],
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["character_id", "chapter", "age_range",
                             "timbre_descriptor", "confidence", "evidence_quote"],
                "properties": {
                    "character_id": {"type": "string"},
                    "chapter": {"type": "integer", "minimum": 0},
                    "age_range": {
                        "type": "array", "minItems": 2, "maxItems": 2,
                        "items": {"type": "integer", "minimum": 0, "maximum": 120},
                    },
                    "timbre_descriptor": {
                        "type": "object",
                        "required": ["register", "texture"],
                        "properties": {
                            "register": {"enum": ["low", "mid", "high"]},
                            "texture": {"enum": ["bright", "warm", "hoarse", "clear"]},
                        },
                    },
                    "jump_reason": {"enum": ["time_skip", "flashback_frame", "age_reveal"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_quote": {"type": "string"},
                },
            },
        }
    },
}


# ---------------------------------------------------------------- 配置

def config(cfg: dict) -> dict:
    raw = ((cfg.get("casting") or {}).get("voice_rules") or {})
    merged: dict = {}
    for key, default in DEFAULT_CONFIG.items():
        if isinstance(default, dict):
            block = {**default}
            for sub, sub_default in default.items():
                if isinstance(sub_default, dict):
                    block[sub] = {**sub_default, **(raw.get(key) or {}).get(sub, {})}
                else:
                    block[sub] = (raw.get(key) or {}).get(sub, sub_default)
            merged[key] = block
        else:
            merged[key] = raw.get(key, default)
    return merged


def enabled(cfg: dict) -> bool:
    return bool(config(cfg)["family"]["enabled"])


def allocation_enabled(cfg: dict) -> bool:
    return bool(config(cfg)["allocation"]["enabled"])


# ---------------------------------------------------------------- 基础工具

def parse_num(value, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").replace("%", "").replace("Hz", "").strip()
    try:
        return float(text)
    except ValueError:
        return default


def fmt_pct(value: float) -> str:
    return f"{value:+.1f}%".replace(".0%", "%")


def fmt_hz(value: float) -> str:
    return f"{value:+.1f}Hz".replace(".0Hz", "Hz")


def binding_params(binding: dict) -> dict:
    return {"rate": parse_num(binding.get("rate")),
            "pitch": parse_num(binding.get("pitch")),
            "volume": parse_num(binding.get("volume"))}


def family_key(cfg: dict, cid: str) -> str:
    return f"{config(cfg)['family']['key_prefix']}{cid}"


def current_family(registry: dict, cfg: dict, cid: str) -> dict:
    return dict((registry.get("voice_families") or {}).get(family_key(cfg, cid)) or {})


def name_index(identities: dict, book_id: str) -> dict:
    index: dict[str, str] = {}
    for cid, ident in (identities or {}).items():
        books = ident.get("books") or []
        if books and book_id not in books:
            continue
        index[cid] = cid
        if ident.get("canonical_name"):
            index[str(ident["canonical_name"])] = cid
        for alias in ident.get("aliases") or []:
            index[str(alias)] = cid
    return index


# ---------------------------------------------------------------- 出场统计

_HASH_CACHE: dict[tuple[str, int, int], dict] = {}


def scan_book(book_dir: Path, upto_chapter: int, identities: dict,
              cfg: dict) -> dict:
    """扫描 scripts/ch*.json，返回每个角色的累计与近 N 章台词字数。

    台词只统计 dialogue 段；旁白不计。`stage_chars` 供 family 选锚点使用。
    """
    window = int(config(cfg)["allocation"]["recent_window_chapters"])
    book_dir = Path(book_dir)
    files = sorted(book_dir.glob("ch*.json"))
    mtime = max((int(p.stat().st_mtime) for p in files), default=0)
    cache_key = (str(book_dir), int(upto_chapter), mtime, window)
    if cache_key in _HASH_CACHE:
        return json.loads(json.dumps(_HASH_CACHE[cache_key]))
    index = name_index(identities, book_dir.name)
    stats: dict[str, dict] = {}
    recent_start = max(0, upto_chapter - window + 1)
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        chapter = int(record.get("chapter_index") or 0)
        if chapter > upto_chapter:
            continue
        cast = record.get("cast") or {}
        speakers = cast.get("speakers") or {}
        profiles = cast.get("cast") or {}
        speaker_cid: dict[str, str] = {}
        for raw, key in speakers.items():
            profile = profiles.get(key) or {}
            cid = profile.get("character_id") or index.get(str(raw)) or str(raw)
            if cid:
                speaker_cid[str(raw)] = cid
        stage_by_cid: dict[str, str] = {}
        for key, profile in profiles.items():
            cid = profile.get("character_id") or index.get(str(profile.get("name")) or "")
            if cid and profile.get("age_band"):
                stage_by_cid[cid] = str(profile["age_band"])
        for seg in record.get("segments") or []:
            if seg.get("type") != "dialogue":
                continue
            raw = str(seg.get("speaker") or "")
            cid = speaker_cid.get(raw) or index.get(raw)
            if not cid:
                continue
            chars = len(str(seg.get("text") or ""))
            row = stats.setdefault(cid, {"chars_all": 0, "chars_recent": 0,
                                         "stage_chars": {}, "first_chapter": chapter,
                                         "last_chapter": chapter, "chapters_seen": []})
            row["chars_all"] += chars
            if chapter >= recent_start:
                row["chars_recent"] += chars
            row["last_chapter"] = max(row["last_chapter"], chapter)
            row["first_chapter"] = min(row["first_chapter"], chapter)
            if chapter not in row["chapters_seen"]:
                row["chapters_seen"].append(chapter)
            stage = stage_by_cid.get(cid)
            if stage:
                row["stage_chars"][stage] = row["stage_chars"].get(stage, 0) + chars
    total_all = sum(row["chars_all"] for row in stats.values()) or 1
    total_recent = sum(row["chars_recent"] for row in stats.values()) or 1
    for row in stats.values():
        row["share_all"] = round(row["chars_all"] / total_all, 6)
        row["share_recent"] = round(row["chars_recent"] / total_recent, 6)
        row["idle_chapters"] = max(0, upto_chapter - row["last_chapter"])
    _HASH_CACHE[cache_key] = json.loads(json.dumps(stats))
    return stats


def frequency_scores(stats: dict, cfg: dict) -> dict:
    """频率分 = 0.7×累计占比 + 0.3×近 20 章占比。

    累计占比反映全书地位（跨几百章仍是主线）；近 20 章反映当前弧线戏份。
    两者相加避免“早期主角长期隐身仍霸占声线”和“临时龙套突然顶掉主角”。
    """
    block = config(cfg)["allocation"]
    w_all, w_recent = float(block["w_all"]), float(block["w_recent"])
    rows = {}
    for cid, row in (stats or {}).items():
        score = w_all * float(row.get("share_all") or 0) \
            + w_recent * float(row.get("share_recent") or 0)
        rows[cid] = {"score": round(score, 6),
                     "share_all": float(row.get("share_all") or 0),
                     "share_recent": float(row.get("share_recent") or 0)}
    return rows


def tier_for(cid: str, stats: dict, cfg: dict, appearance: dict | None = None) -> str:
    if appearance:
        for role in appearance.get("roles") or []:
            if role.get("character_id") == cid:
                return str(role.get("tier") or "generic")
    tier_cfg = ((cfg.get("casting") or {}).get("voice_rules") or {}).get("tiering") or {}
    row = (stats or {}).get(cid) or {}
    chapters = len(row.get("chapters_seen") or [])
    lines = int(row.get("chars_all") or 0)
    share = float(row.get("share_all") or 0)
    if chapters >= int(tier_cfg.get("main_min_chapters", 2)) \
            and (lines >= int(tier_cfg.get("main_min_lines", 40))
                 or share >= float(tier_cfg.get("main_min_share", 0.08))):
        return "main"
    if chapters >= 2 or share >= 0.02:
        return "support"
    return "minor"


# ---------------------------------------------------------------- 曲线 / overlay

def _anchors(cfg: dict) -> list[dict]:
    rows = config(cfg)["age_curve"]["anchors"]
    rows = sorted(rows, key=lambda item: float(item.get("age") or 0))
    return rows


def curve_point(age: float, cfg: dict) -> dict:
    rows = _anchors(cfg)
    if not rows:
        return {"rate_pct": 0.0, "pitch_hz": 0.0, "volume_pct": 0.0}
    if age <= float(rows[0]["age"]):
        return {"rate_pct": float(rows[0]["rate_pct"]),
                "pitch_hz": float(rows[0]["pitch_hz"]),
                "volume_pct": float(rows[0]["volume_pct"])}
    if age >= float(rows[-1]["age"]):
        return {"rate_pct": float(rows[-1]["rate_pct"]),
                "pitch_hz": float(rows[-1]["pitch_hz"]),
                "volume_pct": float(rows[-1]["volume_pct"])}
    for left, right in zip(rows, rows[1:]):
        if float(left["age"]) <= age <= float(right["age"]):
            span = float(right["age"]) - float(left["age"]) or 1.0
            t = (age - float(left["age"])) / span
            return {"rate_pct": float(left["rate_pct"]) + (float(right["rate_pct"]) - float(left["rate_pct"])) * t,
                    "pitch_hz": float(left["pitch_hz"]) + (float(right["pitch_hz"]) - float(left["pitch_hz"])) * t,
                    "volume_pct": float(left["volume_pct"]) + (float(right["volume_pct"]) - float(left["volume_pct"])) * t}
    return {"rate_pct": 0.0, "pitch_hz": 0.0, "volume_pct": 0.0}


def _clamp_params(params: dict, cfg: dict, block: str = "age_curve") -> dict:
    limits = config(cfg)[block]["clamp"]
    out = {}
    for name, key in (("rate", "rate_pct"), ("pitch", "pitch_hz"), ("volume", "volume_pct")):
        low, high = limits[key]
        out[name] = min(max(float(params.get(name) or 0), float(low)), float(high))
    return out


def curve_params(age: float, anchor_age: float, anchor_params: dict, cfg: dict,
                 timbre: dict | None = None) -> dict:
    """相对曲线：锚点参数 + (C(age) − C(anchor_age)) + 音色偏置，再夹紧。"""
    now = curve_point(float(age), cfg)
    base = curve_point(float(anchor_age), cfg)
    params = {
        "rate": float(anchor_params.get("rate") or 0) + now["rate_pct"] - base["rate_pct"],
        "pitch": float(anchor_params.get("pitch") or 0) + now["pitch_hz"] - base["pitch_hz"],
        "volume": float(anchor_params.get("volume") or 0) + now["volume_pct"] - base["volume_pct"],
    }
    if timbre:
        bias = config(cfg)["age_curve"]["timbre_bias"]
        params["pitch"] += float((bias["register"] or {}).get(timbre.get("register"), 0))
        texture = (bias["texture"] or {}).get(timbre.get("texture"), {})
        params["rate"] += float(texture.get("rate_pct") or 0)
        params["pitch"] += float(texture.get("pitch_hz") or 0)
    return _clamp_params(params, cfg)


def smooth_params(previous: dict, target: dict, chapters_delta: int,
                  jump_reason: str | None, cfg: dict) -> dict:
    """单调曲线 + 每章上限；明确时间跳跃（jump_reason 合法）才允许一次跨过去。"""
    block = config(cfg)["age_curve"]
    if jump_reason and jump_reason in block["jump_reasons"]:
        return _clamp_params(target, cfg)
    steps = max(1, int(chapters_delta or 1))
    caps = {"rate": float(block["max_rate_per_chapter_pp"]) * steps,
            "pitch": float(block["max_pitch_per_chapter_hz"]) * steps,
            "volume": float(block["max_volume_per_chapter_pct"]) * steps}
    out = {}
    for name in ("rate", "pitch", "volume"):
        prev = float((previous or {}).get(name) or 0)
        want = float((target or {}).get(name) or 0)
        delta = want - prev
        cap = caps[name]
        out[name] = prev + min(max(delta, -cap), cap)
    return _clamp_params(out, cfg)


def anchor_age_for(ident: dict, stage: str, cfg: dict) -> int:
    profile = ((ident or {}).get("stage_profiles") or {}).get(stage) or {}
    note = str(profile.get("age_note") or "")
    match = re.search(r"(\d{1,3})\s*岁", note)
    if match:
        return int(match.group(1))
    return int(STAGE_MID_AGE.get(stage, 26))


def choose_family_stage(ident: dict, stage_chars: dict, cfg: dict) -> str | None:
    profile_stages = list(((ident or {}).get("stage_profiles") or {}).keys())
    if stage_chars:
        ranked = sorted(stage_chars.items(),
                        key=lambda item: (-int(item[1]), STAGES.index(item[0])
                                          if item[0] in STAGES else 99))
        return ranked[0][0]
    if profile_stages:
        return sorted(profile_stages,
                      key=lambda s: STAGES.index(s) if s in STAGES else 99)[0]
    return None


def overlay_until(start_chapter: int, until_chapter: int | None, scope: str, cfg: dict) -> int:
    block = config(cfg)["overlay"]
    default_until = str(block.get("default_until") or "chapter_end")
    if until_chapter is not None:
        end = int(until_chapter)
    elif scope == "scene" or default_until == "chapter_end":
        end = int(start_chapter)
    else:
        end = int(start_chapter) + int(block["max_chapters"]) - 1
    end = min(end, int(start_chapter) + int(block["max_chapters"]) - 1)
    return max(end, int(start_chapter))


def add_overlay(registry: dict, cid: str, deltas: dict, start_chapter: int,
                cfg: dict, reason: str = "disguise", scope: str = "chapter",
                until_chapter: int | None = None, disguised_as: str = "",
                dry_run: bool = True) -> dict:
    block = config(cfg)["overlay"]
    if reason not in block["reasons"]:
        reason = "disguise"
    end = overlay_until(start_chapter, until_chapter, scope, cfg)
    row = {
        "id": f"overlay:{cid}|{start_chapter}|{reason}",
        "character_id": cid,
        "reason": reason,
        "scope": scope,
        "start_chapter": int(start_chapter),
        "until_chapter": end,
        "disguised_as": disguised_as,
        "perspective": block["reader_perspective"],
        "deltas": _clamp_params({k: v for k, v in deltas.items()}, cfg, "overlay"),
        "created_at": f"chapter:{start_chapter}",
    }
    if not dry_run:
        overlays = registry.setdefault("voice_overlays", {})
        overlays[row["id"]] = row
    return row


def expire_overlays(registry: dict, chapter: int, dry_run: bool = True) -> list[dict]:
    removed: list[dict] = []
    overlays = registry.get("voice_overlays") or {}
    for key, row in list(overlays.items()):
        if int(row.get("until_chapter") or 0) < int(chapter):
            removed.append(dict(row))
            if not dry_run:
                overlays.pop(key, None)
    return removed


def active_overlays(registry: dict, cid: str, chapter: int) -> list[dict]:
    rows = []
    for row in (registry.get("voice_overlays") or {}).values():
        if row.get("character_id") != cid:
            continue
        if int(row.get("start_chapter") or 0) <= chapter <= int(row.get("until_chapter") or 0):
            rows.append(dict(row))
    return sorted(rows, key=lambda item: item.get("id"))


def apply_overlay(params: dict, overlays: list[dict], cfg: dict) -> dict:
    out = dict(params)
    for row in overlays:
        delta = row.get("deltas") or {}
        out["rate"] = float(out.get("rate") or 0) + float(delta.get("rate") or 0)
        out["pitch"] = float(out.get("pitch") or 0) + float(delta.get("pitch") or 0)
        out["volume"] = float(out.get("volume") or 0) + float(delta.get("volume") or 0)
    return _clamp_params(out, cfg, "overlay")


# ---------------------------------------------------------------- family 迁移

def stage_params(ident: dict, stage: str) -> dict:
    """阶段档案里的有效基准参数（caster 写入的 persona 结果）。

    旧锁的 rate/pitch/volume 常为 null（参数来自阶段档案），迁移时必须以档案兜底，
    否则 family 锚点会被清零，和线上现版产生无谓跳变。
    """
    profile = ((ident or {}).get("stage_profiles") or {}).get(stage) or {}
    return {key: parse_num(profile.get(key)) for key in ("rate", "pitch", "volume")}


def migrate_families(registry: dict, book_id: str, stats: dict, identities: dict,
                     cfg: dict, chapter: int, dry_run: bool = True) -> dict:
    """把 `lock:{cid}|{stage}` 收敛成 `family:{cid}`；旧锁保留只读，参数零跳变。"""
    block = config(cfg)["family"]
    families = registry.setdefault("voice_families", {}) if not dry_run \
        else dict(registry.get("voice_families") or {})
    bindings = registry.get("voice_bindings") or {}
    index = name_index(identities, book_id)
    created, skipped, existing = [], [], []
    for cid, ident in (identities or {}).items():
        books = ident.get("books") or []
        if books and book_id not in books:
            continue
        key = f"{block['key_prefix']}{cid}"
        if key in families:
            existing.append(cid)
            continue
        row = (stats or {}).get(cid) or {}
        stage = choose_family_stage(ident, row.get("stage_chars") or {}, cfg)
        if not stage:
            skipped.append({"character_id": cid, "reason": "no_stage"})
            continue
        lock_key = f"lock:{cid}|{stage}"
        binding = bindings.get(lock_key)
        if not binding or not binding.get("voice"):
            # 回退：任取一条该角色的旧锁
            candidates = [(k, v) for k, v in bindings.items()
                          if v.get("character_id") == cid and v.get("voice")]
            if not candidates:
                skipped.append({"character_id": cid, "reason": "no_binding", "stage": stage})
                continue
            lock_key, binding = sorted(
                candidates, key=lambda item: str(item[1].get("age_stage") or "")).pop()
            stage = str(binding.get("age_stage") or stage)
        params = binding_params(binding)
        if not any(params.values()):
            params = stage_params(ident, stage)
        family = {
            "id": key,
            "character_id": cid,
            "book_id": book_id,
            "voice": binding["voice"],
            "anchor_stage": stage,
            "anchor_age": anchor_age_for(ident, stage, cfg),
            "anchor_params": params,
            "state": "owned",
            "source_binding": lock_key,
            "created_at_chapter": int(chapter),
            "lease": None,
            "promotion": None,
            "updated_by": "family_migrate",
            "updated_at_chapter": int(chapter),
        }
        families[key] = family
        created.append(family)
    if not dry_run:
        registry["voice_families"] = families
        registry.setdefault("family_audit", []).append(
            {"type": "migrate", "book_id": book_id, "chapter": int(chapter),
             "created": [row["character_id"] for row in created],
             "skipped": skipped, "legacy_mode": block["legacy_mode"]})
    return {"families": families, "created": created, "skipped": skipped,
            "existing": existing}


# ---------------------------------------------------------------- 退场 / 状态机

def retirement_effective(judgment: dict, cfg: dict) -> str:
    """低置信度按 hidden；证据定位不到按 invalid（一律不放行 dormant）。"""
    if not judgment:
        return "invalid"
    status = str(judgment.get("status") or "")
    if status not in RETIRE_STATUS:
        return "invalid"
    if float(judgment.get("confidence") or 0) < float(config(cfg)["llm"]["min_confidence"]):
        return "hidden"
    if not str(judgment.get("evidence_quote") or "").strip():
        return "invalid"
    if status == "present":
        return "hidden"
    return status


def dormant_allowed(stat: dict, retirement: dict, chapter: int, cfg: dict) -> dict:
    block = config(cfg)["reclaim"]
    idle = max(0, int(chapter) - int((stat or {}).get("last_chapter") or 0))
    floor = max(int(block["idle_chapters_floor"]), 4)
    need = max(int(block["idle_chapters"]), floor)
    verdict = retirement_effective(retirement, cfg)
    reasons = []
    if idle < need:
        reasons.append(f"idle {idle} < {need}")
    if int((stat or {}).get("chars_all") or 0) >= int(block["min_lines_keep"]):
        reasons.append("lines_gate")
    if float((stat or {}).get("share_all") or 0) >= float(block["max_share"]):
        reasons.append("share_gate")
    if verdict != "retired":
        reasons.append(f"retirement:{verdict}")
    if str(block.get("scope") or "family") != "family":
        reasons.append("scope")
    return {"allowed": not reasons, "idle": idle, "threshold": need,
            "status": verdict, "reasons": reasons}


def family_transition(family: dict, chapter: int, cfg: dict,
                      owner_returned: bool = False) -> dict:
    """按 `回购 > 转正审批 > 租约续期 > 新外借` 顺序推进单个 family。"""
    block = config(cfg)["reclaim"]
    state = str(family.get("state") or "owned")
    lease = family.get("lease") or {}
    if lease and owner_returned:
        return {"action": "buyback", "state": "owned",
                "reason": "原主回归，租约期内也无条件回购"}
    if lease and chapter > int(lease.get("until_chapter") or chapter):
        return {"action": "lease_expire", "state": "dormant",
                "reason": "租约到期且原主未归"}
    promotion = family.get("promotion") or {}
    promo_cfg = block.get("promotion") or {}
    if promo_cfg.get("enabled") and lease and int(lease.get("lease_count") or 0) >= int(promo_cfg["min_leases"]) \
            and int(lease.get("active_chapters") or 0) >= int(promo_cfg["min_active_chapters"]) \
            and str((family.get("retirement") or {}).get("status")) == str(promo_cfg["require_llm_status"]):
        return {"action": "promote", "state": "promoted",
                "reason": "长期租用者转正（需人工审批）"}
    if state == "dormant":
        return {"action": "lend", "state": "lent",
                "reason": "dormant 可外借"}
    return {"action": "keep", "state": state, "reason": ""}


def lease_family(family: dict, lessee: str, chapter: int, cfg: dict) -> dict:
    lease_len = int(config(cfg)["reclaim"]["lease_chapters"])
    lease = family.get("lease") or {}
    return {"action": "lend", "state": "lent",
            "lease": {"lessee": lessee, "start_chapter": int(chapter),
                      "until_chapter": int(chapter) + lease_len,
                      "lease_count": int(lease.get("lease_count") or 0) + 1,
                      "active_chapters": int(lease.get("active_chapters") or 0)}}


# ---------------------------------------------------------------- 频率分配 / 抢占

def _holders(registry: dict, cfg: dict, book_id: str) -> dict:
    rows = {}
    for key, family in (registry.get("voice_families") or {}).items():
        if family.get("book_id") not in (None, book_id):
            continue
        rows[family.get("character_id")] = dict(family, _key=key)
    return rows


def _needy(family: dict | None, narrator: str) -> bool:
    if not family:
        return True
    return str(family.get("state")) in ("shared", "borrowed") \
        or not family.get("voice") or str(family.get("voice")) == narrator


def plan_preemptions(registry: dict, book_id: str, stats: dict, chapter: int,
                     cfg: dict, narrator: str, tiers: dict | None = None,
                     boundary: bool = True) -> dict:
    """章边界上的频率优先抢占计划；不写 registry。

    - `main` 既不抢也不被抢；
    - 差距同时满足绝对分与倍数门槛；
    - 连续 `confirm_chapters` 章达标才执行，同一对角色有冷却；
    - incumbent 若在 lent/promoted，状态机优先，跳过。
    """
    block = config(cfg)["allocation"]
    pre = block["preempt"]
    if not boundary or not block["enabled"] or not pre["enabled"]:
        return {"preemptions": [], "tracker": {}, "note": "disabled"}
    scores = frequency_scores(stats, cfg)
    holders = _holders(registry, cfg, book_id)
    tracker = (((registry.get("book_settings") or {}).get(book_id) or {})
               .get("preempt_tracker") or {})
    tracker = {key: dict(value) for key, value in tracker.items()}
    decisions = []
    taken: set[str] = set()
    excluded = set(pre.get("exclude_tiers") or ["main"])
    for cid, score_row in sorted(scores.items(), key=lambda item: -item[1]["score"]):
        if len(decisions) >= int(pre["max_per_chapter"]):
            break
        tier = (tiers or {}).get(cid) or tier_for(cid, stats, cfg)
        if tier in excluded:
            continue
        family = holders.get(cid)
        if not _needy(family, narrator):
            continue
        candidates = []
        for other, other_family in holders.items():
            if other == cid or other in taken:
                continue
            other_tier = (tiers or {}).get(other) or tier_for(other, stats, cfg)
            if other_tier in excluded:
                continue
            if str(other_family.get("state")) != "owned":
                continue
            if not other_family.get("voice") or other_family.get("voice") == narrator:
                continue
            gap = score_row["score"] - float(scores.get(other, {}).get("score") or 0)
            ratio = min(score_row["score"] / max(float(scores.get(other, {}).get("score") or 0), 1e-9),
                        999.0)
            if gap < float(pre["min_score_gap"]) or ratio < float(pre["min_score_ratio"]):
                continue
            candidates.append((gap, ratio, other, other_family))
        if not candidates:
            for key in list(tracker):
                if key.startswith(f"{cid}->"):
                    tracker.pop(key, None)
            continue
        gap, ratio, other, other_family = max(candidates, key=lambda item: (item[0], item[2]))
        track_key = f"{cid}->{other}"
        row = tracker.get(track_key) or {"streak": 0, "last_chapter": -9999}
        if chapter - int(row.get("last_chapter") or -9999) < int(pre["cooldown_chapters"]) \
                and int(row.get("executed_chapter") or -9999) > -9999:
            continue
        row["streak"] = int(row.get("streak") or 0) + 1
        row["last_chapter"] = int(chapter)
        tracker[track_key] = row
        if row["streak"] < int(pre["confirm_chapters"]):
            decisions.append({"action": "pending", "chapter": int(chapter),
                              "challenger": cid, "incumbent": other,
                              "challenger_score": score_row["score"],
                              "incumbent_score": scores.get(other, {}).get("score", 0),
                              "gap": round(gap, 4), "ratio": round(ratio, 3),
                              "streak": row["streak"]})
            continue
        taken.add(other)
        row["executed_chapter"] = int(chapter)
        decisions.append({"action": "preempt", "chapter": int(chapter),
                          "challenger": cid, "incumbent": other,
                          "challenger_score": score_row["score"],
                          "incumbent_score": scores.get(other, {}).get("score", 0),
                          "gap": round(gap, 4), "ratio": round(ratio, 3),
                          "streak": row["streak"],
                          "voice": other_family.get("voice"),
                          "demote_to": pre.get("demote_to", "shared")})
    return {"preemptions": decisions, "tracker": tracker,
            "scores": scores, "note": "chapter_boundary"}


def apply_preemptions(registry: dict, book_id: str, plan: dict, cfg: dict,
                      dry_run: bool = True) -> list[dict]:
    """执行抢占：winner 拿 voice，loser 降级共享池；只写 family 与审计，不动音频。"""
    applied = []
    families = registry.get("voice_families") or {}
    for row in plan.get("preemptions") or []:
        if row.get("action") != "preempt":
            continue
        winner = families.get(family_key(cfg, row["challenger"]))
        loser = families.get(family_key(cfg, row["incumbent"]))
        if winner is None and not dry_run:
            winner = registry["voice_families"].setdefault(
                family_key(cfg, row["challenger"]),
                {"id": family_key(cfg, row["challenger"]),
                 "character_id": row["challenger"], "book_id": book_id,
                 "voice": "", "anchor_stage": "", "anchor_age": 26,
                 "anchor_params": {"rate": 0.0, "pitch": 0.0, "volume": 0.0},
                 "state": "owned", "source_binding": "", "created_at_chapter": row["chapter"],
                 "lease": None, "promotion": None})
        if loser is None:
            continue
        record = dict(row)
        record["previous_voice"] = winner.get("voice") if winner else ""
        if not dry_run:
            winner["voice"] = loser.get("voice")
            winner["state"] = "owned"
            winner["preempted_from"] = row["incumbent"]
            winner["updated_at_chapter"] = row["chapter"]
            loser["voice"] = ""
            loser["state"] = str(row.get("demote_to") or "shared")
            loser["demoted_by"] = row["challenger"]
            loser["updated_at_chapter"] = row["chapter"]
            registry.setdefault("family_audit", []).append(
                {"type": "preempt", "book_id": book_id, "chapter": row["chapter"],
                 "winner": row["challenger"], "loser": row["incumbent"],
                 "winner_score": row["challenger_score"],
                 "loser_score": row["incumbent_score"],
                 "gap": row["gap"], "ratio": row["ratio"],
                 "previous_voice": record["previous_voice"],
                 "new_voice": loser.get("voice")})
            book = registry.setdefault("book_settings", {}).setdefault(book_id, {})
            book["preempt_tracker"] = plan.get("tracker") or {}
        applied.append(record)
    if not dry_run and not applied:
        book = registry.setdefault("book_settings", {}).setdefault(book_id, {})
        book["preempt_tracker"] = plan.get("tracker") or {}
    return applied


def cast_order(cast: dict, scores: dict) -> list[str]:
    """按频率分降序给 cast key 排序；只改分配顺序，不改内容。"""
    def score(key: str) -> float:
        profile = cast.get(key) or {}
        cid = profile.get("character_id") or profile.get("name") or key
        return float((scores.get(cid) or {}).get("score") or 0)

    return sorted(cast.keys(), key=lambda key: (-score(key), key))


# ---------------------------------------------------------------- 撞声放宽

def relax_threshold(age_gap: float | None, cfg: dict) -> tuple[float, float, str]:
    """同族近龄的距离阈值曲线，返回 (阈值, 原阈值, 规则名)。

    曲线：gap ≤ max_age_gap 用 relaxed_distance；max_gap..ramp_gap 线性回升到原阈值；
    超过 ramp_gap 完全恢复原阈值。最终被 floor_distance 兜底。
    """
    rules = config(cfg)["same_family_relax"]
    from . import voice_policy
    base = float(voice_policy.thresholds(cfg)["distance"])
    if not rules["enabled"] or age_gap is None:
        return base, base, "base"
    relaxed = float(rules["relaxed_distance"])
    floor = float(rules["floor_distance"])
    max_gap = float(rules["max_age_gap"])
    ramp_gap = float(rules["ramp_age_gap"])
    if age_gap <= max_gap:
        value = relaxed
    elif age_gap <= ramp_gap and ramp_gap > max_gap:
        ratio = (age_gap - max_gap) / (ramp_gap - max_gap)
        value = relaxed + (base - relaxed) * ratio
    else:
        return base, base, "base"
    value = max(value, floor)
    return min(value, base), base, "same_family_close_age"


def collision_verdict(params_a: dict, params_b: dict, *, voice_a: str, voice_b: str,
                      age_a: int | None, age_b: int | None, cfg: dict) -> dict:
    """判定一对角色是否撞声，同族近龄时按曲线放宽并给出审计字段。

    params_* 走 voice_policy.deltas 所需的三维数值；voice_* 是最终 edge 声线。
    """
    from . import voice_policy
    t = voice_policy.thresholds(cfg)
    diff = voice_policy.deltas(params_a, params_b)
    dist = voice_policy.distance(params_a, params_b, cfg)
    same_voice = bool(voice_a) and voice_a == voice_b
    rules = config(cfg)["same_family_relax"]
    age_gap = abs(int(age_a) - int(age_b)) if age_a is not None and age_b is not None else None
    if same_voice and rules["enabled"] and rules["same_voice_only"]:
        threshold, base, rule = relax_threshold(age_gap, cfg)
    else:
        threshold, base, rule = t["distance"], t["distance"], "base"
    # 放宽只作用于“距离”这一维；仍然要求至少一维存在可听差异，
    # 且距离必须 ≥ 阈值（阈值已被 floor 兜底，参数完全相同 = 0 必判撞声）。
    any_dim = any(diff[key] > 0 for key in ("rate", "pitch", "volume"))
    ok = any_dim and dist >= threshold
    return {
        "ok": ok,
        "distance": round(dist, 3),
        "threshold": round(threshold, 3),
        "base_threshold": round(base, 3),
        "rule": rule,
        "age_gap": age_gap,
        "same_voice": same_voice,
        "delta": diff,
        "relaxed": threshold < base - 1e-9,
        "floor": float(rules["floor_distance"]),
    }


def record_collision_audit(registry: dict, verdict: dict, *, chapter: int,
                           left: str, right: str, book_id: str = "") -> dict:
    """把一次放宽判定写进 registry 审计（默认关闭时不写）。"""
    row = {"type": "collision_relax", "book_id": book_id, "chapter": chapter,
           "a": left, "b": right, "age_gap": verdict.get("age_gap"),
           "distance": verdict.get("distance"),
           "base_threshold": verdict.get("base_threshold"),
           "threshold": verdict.get("threshold"),
           "rule": verdict.get("rule"), "accepted": bool(verdict.get("ok")),
           "floor": verdict.get("floor")}
    registry.setdefault("collision_audit", []).append(row)
    return row


# ---------------------------------------------------------------- LLM 判定

def has_voice_id(payload) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key) in ("voice_id", "voice", "edge_voice"):
                return True
            if has_voice_id(value):
                return True
    elif isinstance(payload, list):
        return any(has_voice_id(item) for item in payload)
    return False


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def evidence_ok(quote: str, body: str) -> bool:
    return bool(str(quote or "").strip()) and _norm(quote) in _norm(body)


def validate_age_payload(payload: dict, body: str, known_cids: set[str],
                         cfg: dict) -> dict:
    from .llm import validate_schema
    errors = []
    if has_voice_id(payload):
        return {"ok": False, "rows": [],
                "errors": [{"code": "VOICE_ID_IMMUTABLE",
                            "detail": "输出出现 voice_id/voice 字段，整条拒绝"}]}
    schema_error = validate_schema(payload, AGE_SCHEMA)
    if schema_error:
        return {"ok": False, "rows": [],
                "errors": [{"code": "SCHEMA_INVALID", "detail": schema_error}]}
    rows = []
    for index, item in enumerate(payload.get("judgments") or []):
        item = dict(item)
        cid = str(item.get("character_id") or "")
        if cid not in known_cids:
            errors.append({"code": "UNKNOWN_CHARACTER", "index": index,
                           "detail": cid})
            continue
        low, high = item.get("age_range") or [0, 0]
        if int(low) > int(high) or int(high) > 120:
            errors.append({"code": "AGE_RANGE_INVALID", "index": index,
                           "detail": f"{low}-{high}"})
            continue
        if not evidence_ok(str(item.get("evidence_quote") or ""), body):
            errors.append({"code": "EVIDENCE_MISMATCH", "index": index,
                           "detail": str(item.get("evidence_quote") or "")[:40]})
            continue
        confidence = float(item.get("confidence") or 0)
        if confidence < float(config(cfg)["llm"]["min_confidence"]):
            errors.append({"code": "CONFIDENCE_PENDING", "index": index,
                           "detail": confidence})
        rows.append(item)
    blocking = [err for err in errors if err["code"] != "CONFIDENCE_PENDING"]
    return {"ok": not blocking, "rows": rows, "errors": errors}


def rule_age_judgments(body: str, chapter: int) -> list[dict]:
    """规则回落：只认明确的“N 岁”；判不出来就返回空，不猜。"""
    rows = []
    for match in re.finditer(r"([^\s。！？\d]{0,12})(\d{1,3})\s*岁([^。！？]{0,12})", body):
        age = int(match.group(2))
        if not 0 <= age <= 120:
            continue
        rows.append({"character_id": "", "chapter": int(chapter),
                     "age_range": [age, age],
                     "timbre_descriptor": {"register": "mid", "texture": "warm"},
                     "confidence": 0.99, "source": "rule",
                     "evidence_quote": match.group(0)})
    return rows


def judge_ages(client, record: dict, known_cids: set[str], cfg: dict,
               registry: dict | None = None, dry_run: bool = True) -> dict:
    """LLM 年龄轨迹 → 校验 → 失败带错误重试 → 规则回落 + pending_review。"""
    block = config(cfg)["llm"]
    chapter = int(record.get("chapter_index") or 0)
    body = "\n".join(str(seg.get("text") or "") for seg in record.get("segments") or [])
    if not block.get("enabled"):
        return {"status": "disabled", "rows": [], "pending": []}
    messages = [
        {"role": "system", "content":
         "你只输出角色的年龄轨迹。禁止输出 voice_id/voice/edge_voice；只允许 JSON。"
         "age_range 是本章内该角色出现的年龄区间（回忆与现时并存时取完整范围）。"},
        {"role": "user", "content":
         f"角色：{sorted(known_cids)}\n章节：{chapter}\n原文：\n{body[:int(block.get('max_chars', 6000))]}"},
    ]
    payload, errors = None, []
    for attempt in range(int(block.get("retry_with_errors", 1)) + 1):
        try:
            payload = client.chat_json(messages, schema=AGE_SCHEMA, schema_name="voice_age")
        except Exception as err:  # noqa: BLE001 - 链路失败统一回落
            errors = [{"code": "LLM_UNAVAILABLE_FALLBACK", "detail": str(err)}]
            payload = None
            break
        checked = validate_age_payload(payload, body, known_cids, cfg)
        errors = checked["errors"]
        if checked["ok"] and not any(err["code"] == "CONFIDENCE_PENDING" for err in errors):
            if not dry_run and registry is not None:
                registry.setdefault("voice_age_judgments", []).append(
                    {"chapter": chapter, "rows": checked["rows"],
                     "source": "llm", "model": "chain"})
            return {"status": "ok", "rows": checked["rows"], "pending": []}
        if attempt < int(block.get("retry_with_errors", 1)):
            messages.append({"role": "user", "content":
                             "上次输出被拒，请只修正并重发 JSON："
                             + json.dumps(errors, ensure_ascii=False)})
    if payload is not None:
        checked = validate_age_payload(payload, body, known_cids, cfg)
        rows, errors = checked["rows"], checked["errors"]
    else:
        rows = []
    fallback = rule_age_judgments(body, chapter)
    pending = []
    for err in errors:
        pending.append({"kind": "voice_age", "chapter": chapter,
                        "code": err.get("code"), "detail": err.get("detail"),
                        "source": "llm", "status": "pending_review"})
    if fallback:
        for row in fallback:
            pending.append({"kind": "voice_age", "chapter": chapter,
                            "code": "RULE_FALLBACK", "detail": row.get("evidence_quote"),
                            "source": "rule", "status": "pending_review"})
    if not dry_run and registry is not None and pending:
        registry.setdefault("pending_review", []).extend(pending)
    return {"status": "fallback", "rows": fallback, "pending": pending,
            "errors": errors}


# ---------------------------------------------------------------- 渲染

def render_chapter(registry: dict, book_id: str, chapter: int, cid: str,
                   age: float, cfg: dict) -> dict:
    family = current_family(registry, cfg, cid)
    if not family:
        return {"character_id": cid, "family": None, "params": None,
                "overlays": [], "note": "no_family"}
    params = curve_params(age, float(family.get("anchor_age") or age),
                          family.get("anchor_params") or {}, cfg)
    overlays = active_overlays(registry, cid, chapter)
    params = apply_overlay(params, overlays, cfg) if overlays else params
    return {"character_id": cid, "family": family.get("voice"),
            "anchor_age": family.get("anchor_age"),
            "params": params, "overlays": overlays}
