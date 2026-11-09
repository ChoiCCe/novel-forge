"""声线阶梯与同声线参数区分策略。

四级阶梯（casting.voice_rules.strategy=independent_first）：
1) 独立声线 + 独特参数（一对一声线）；
2) 复用某个非旁白声线 + 不同参数；
3) 旁白声线 + 不同参数（最后手段）；
4) 未知/不明确角色 → 旁白声线 + 独立参数档。

本模块只做纯计算：根据 Caster 产出的 profile/binding 判定阶梯级别、来源标签、
同一“声线+参数”组合冲突、复用组与自动微调方案。声线分配与方言证据闸门在 Caster。

判定：同一时间不允许两个角色（含旁白）使用完全相同的“声线+参数”组合；
rate/pitch/volume 至少一维达到阈值，且归一化整体距离达标。
参数阈值来自 casting.voice_rules.differentiate，依据 tools/param_diff_probe.py 的实测。
"""
from __future__ import annotations

import math

from . import prosody

POLICY_INDEPENDENT = "independent_first"
POLICY_DIFFERENTIATE = "differentiate"
POLICY_FORBID = "forbid"
_POLICIES = (POLICY_INDEPENDENT, POLICY_DIFFERENTIATE, POLICY_FORBID)

LABEL_INDEPENDENT = "independent"
LABEL_SHARED = "voice_shared"
LABEL_BORROWED = "voice_borrowed"
LABEL_UNKNOWN = "unknown_fallback"
LABEL_PARAM_FIX = "param_fix"

LEVEL_NAMES = {
    1: LABEL_INDEPENDENT,
    2: LABEL_SHARED,
    3: LABEL_BORROWED,
    4: LABEL_UNKNOWN,
}


def policy_of(cfg: dict) -> str:
    """strategy 优先；缺失时按旧键回退（differentiate / forbid）。"""
    rules = (cfg.get("casting") or {}).get("voice_rules") or {}
    strategy = str(rules.get("strategy") or "").strip()
    if strategy in _POLICIES:
        return strategy
    policy = str(rules.get("narrator_voice_policy") or "").strip()
    if policy in _POLICIES:
        return policy
    return POLICY_FORBID if rules.get("forbid_narrator_voice") else POLICY_DIFFERENTIATE


def label_of(voice: str, narrator_voice: str, *, unknown: bool = False,
             shared: bool = False) -> str:
    if unknown:
        return LABEL_UNKNOWN
    if voice and voice == narrator_voice:
        return LABEL_BORROWED
    if shared:
        return LABEL_SHARED
    return LABEL_INDEPENDENT


def thresholds(cfg: dict) -> dict:
    rules = (cfg.get("casting") or {}).get("voice_rules") or {}
    diff = rules.get("differentiate") or {}
    return {
        "rate": float(diff.get("min_rate_pp", 4)),
        "pitch": float(diff.get("min_pitch_hz", 4)),
        "volume": float(diff.get("min_volume_pct", 6)),
        "distance": float(diff.get("min_distance", 1.4)),
    }


def params(profile: dict) -> dict:
    return {
        "rate": prosody.parse_value(profile.get("rate", "+0%"), "%"),
        "pitch": prosody.parse_value(profile.get("pitch", "+0Hz"), "Hz"),
        "volume": prosody.parse_value(profile.get("volume", "+0%"), "%"),
    }


def deltas(a: dict, b: dict) -> dict:
    return {key: abs(a[key] - b[key]) for key in ("rate", "pitch", "volume")}


def distance(a: dict, b: dict, cfg: dict) -> float:
    """按阈值归一化后的欧氏距离：单维刚好达标时距离为 1.0。"""
    t = thresholds(cfg)
    diff = deltas(a, b)
    return math.sqrt(
        (diff["rate"] / t["rate"]) ** 2
        + (diff["pitch"] / t["pitch"]) ** 2
        + (diff["volume"] / t["volume"]) ** 2
    )


def pair_ok(a: dict, b: dict, cfg: dict) -> bool:
    """至少一维达标，且归一化整体距离达标。"""
    t = thresholds(cfg)
    diff = deltas(a, b)
    any_dim = any(diff[key] >= t[key] for key in ("rate", "pitch", "volume"))
    return any_dim and distance(a, b, cfg) >= t["distance"]


def _candidate_lattice(cfg: dict) -> list[dict]:
    """确定性的候选增量格，按归一化距离从近到远排序。"""
    t = thresholds(cfg)
    rate_steps = [0, -4, 4, -6, 6, -8, 8, -10, 10, -12, 12]
    pitch_steps = [0, -2, 2, -4, 4, -6, 6]
    volume_steps = [0, -3, 3, -6, 6, -9, 9]
    lattice: list[dict] = []
    for rate in rate_steps:
        for pitch in pitch_steps:
            for volume in volume_steps:
                if not (rate or pitch or volume):
                    continue
                score = (abs(rate) / t["rate"] + abs(pitch) / t["pitch"]
                         + abs(volume) / t["volume"])
                lattice.append({"rate": rate, "pitch": pitch, "volume": volume,
                                "score": round(score, 4)})
    lattice.sort(key=lambda item: (item["score"], item["rate"], item["pitch"], item["volume"]))
    return lattice


def _within_clamp(value: dict, cfg: dict) -> bool:
    for key in ("rate", "pitch", "volume"):
        low, high = prosody.clamp_bounds(key, cfg)
        if not low <= value[key] <= high:
            return False
    return True


def _format(value: dict) -> dict:
    return {
        "rate": prosody.format_value(value["rate"], "%"),
        "pitch": prosody.format_value(value["pitch"], "Hz"),
        "volume": prosody.format_value(value["volume"], "%"),
    }


def unknown_variants(cfg: dict, narrator: dict, count: int) -> list[dict]:
    """未知说话人参数档：每人一档，全部与旁白、与彼此达到区分阈值。

    第一档取 casting.unknown_speaker 的配置值；不足时从确定性增量格里补，
    仍然夹紧在 prosody clamp 内。返回已格式化的 rate/pitch/volume 字典列表。
    """
    base = params(narrator)
    override = (cfg.get("casting") or {}).get("unknown_speaker") or {}
    wanted = dict(base)
    for key in ("rate", "pitch", "volume"):
        if override.get(key):
            wanted[key] = prosody.parse_value(str(override[key]),
                                              "%" if key != "pitch" else "Hz")
    narrator_params = params(narrator)
    variants: list[dict] = []
    used = [narrator_params]
    lattice = _candidate_lattice(cfg)
    for index in range(max(1, int(count))):
        if index == 0:
            chosen = wanted
        else:
            chosen = None
            for candidate in lattice:
                value = {key: wanted[key] + candidate[key]
                         for key in ("rate", "pitch", "volume")}
                if not _within_clamp(value, cfg):
                    continue
                if all(pair_ok(value, item, cfg) for item in used):
                    chosen = value
                    break
        if chosen is None:
            chosen = dict(wanted)
        if not _within_clamp(chosen, cfg):
            chosen = dict(wanted)
        variants.append(_format(chosen))
        used.append(chosen)
    return variants


def _member_rows(cast: dict, narrator_voice: str, cfg: dict) -> list[dict]:
    rows = []
    for key, profile in (cast or {}).items():
        voice = profile.get("edge_voice") or ""
        source = str(profile.get("voice_source") or "config")
        unknown = source in ("unknown_speaker", "unknown_fallback") \
            or int(profile.get("ladder_level") or 0) == 4 \
            or bool(profile.get("generic"))
        dialect_profile = profile.get("dialect_profile") or {}
        rows.append({
            "key": key,
            "name": profile.get("name") or key,
            "stage": profile.get("age_stage") or profile.get("age_band"),
            "gender": profile.get("gender") or "未知",
            "voice": voice,
            "unknown": unknown,
            "shared": int(profile.get("ladder_level") or 0) == 2,
            "ladder_level": int(profile.get("ladder_level") or 0),
            "line_count": int(profile.get("line_count") or 0),
            "source": source,
            "binding_id": profile.get("binding_id"),
            "manual": source.lower() == "manual",
            "accent": profile.get("accent") or "",
            "dialect": dialect_profile.get("dialect") or "",
            "dialect_evidence": dialect_profile.get("evidence") or "",
            "dialect_method": dialect_profile.get("method") or "",
            "dialect_confidence": dialect_profile.get("confidence") or 0,
            "params": params(profile),
            "profile": profile,
        })
    # 稳定排序：主要角色优先占位，其余按 key
    rows.sort(key=lambda item: (-item["line_count"], item["key"]))
    return rows


def _assign_labels(rows: list[dict], narrator_voice: str) -> None:
    counts: dict[str, int] = {}
    for item in rows:
        counts[item["voice"]] = counts.get(item["voice"], 0) + 1
    for item in rows:
        if item["unknown"]:
            level = 4
        elif item["voice"] == narrator_voice:
            level = 3
        elif item["ladder_level"] == 2 or counts.get(item["voice"], 0) > 1:
            level = 2
        else:
            level = 1
        item["level"] = level
        item["label"] = LEVEL_NAMES[level]


def plan(cast: dict, narrator: dict, cfg: dict) -> dict:
    """评估并按需生成自动微调方案；adjustments 由调用方写入 binding（param_fix）。"""
    policy = policy_of(cfg)
    t = thresholds(cfg)
    narrator_params = params(narrator)
    narrator_voice = narrator.get("edge_voice") or ""
    rows = _member_rows(cast, narrator_voice, cfg)
    _assign_labels(rows, narrator_voice)
    report = {
        "policy": policy,
        "thresholds": t,
        "narrator": {"voice": narrator_voice, **_format(narrator_params), "params": narrator_params},
        "same_voice_roles": 0,
        "independent_roles": 0,
        "shared_roles": 0,
        "borrowed_roles": 0,
        "unknown_fallback_roles": 0,
        "compliant": 0,
        "param_fix": 0,
        "unresolved": 0,
        "manual_conflict": 0,
        "combo_conflicts": 0,
        "conflict_details": [],
        "reuse_groups": [],
        "narrator_group": [],
        "dialect_rows": [],
        "dialect_enabled": False,
        "nearest_to_narrator": None,
        "rows": [],
        "adjustments": [],
    }
    if policy == POLICY_FORBID:
        for item in rows:
            item["same_voice"] = item["voice"] == narrator_voice
            item["ok"] = not item["same_voice"]
            item["distance_to_narrator"] = round(
                distance(item["params"], narrator_params, cfg), 2)
        report["rows"] = [_public_row(item) for item in rows]
        report["same_voice_roles"] = sum(1 for item in rows if item["same_voice"])
        report["compliant"] = sum(1 for item in rows if item["ok"])
        report["independent_roles"] = sum(1 for item in rows
                                          if item["label"] == LABEL_INDEPENDENT)
        report["borrowed_roles"] = report["same_voice_roles"]
        report["unknown_fallback_roles"] = sum(1 for item in rows
                                               if item["label"] == LABEL_UNKNOWN)
        return report

    groups: dict[str, list[dict]] = {}
    for item in rows:
        groups.setdefault(item["voice"], []).append(item)
    lattice = _candidate_lattice(cfg)
    max_attempts = int(((cfg.get("casting") or {}).get("voice_rules") or {})
                       .get("differentiate", {}).get("max_attempts", 120))

    for voice, members in groups.items():
        used: list[dict] = []
        if voice == narrator_voice:
            used.append({"name": "旁白", "params": narrator_params})
        for item in members:
            item["same_voice"] = voice == narrator_voice
            base = item["params"]
            conflicts = [u for u in used if not pair_ok(base, u["params"], cfg)]
            item["conflicts"] = [u["name"] for u in conflicts]
            if not conflicts:
                item["ok"] = True
                item["fixed"] = False
                used.append({"name": item["name"], "params": base})
                continue
            if item["manual"]:
                # 人工锁定不自动改，只报冲突
                item["ok"] = False
                item["fixed"] = False
                item["manual_conflict"] = True
                item["unresolved"] = True
                used.append({"name": item["name"], "params": base})
                continue
            chosen = None
            for candidate in lattice[:max_attempts]:
                value = {key: base[key] + candidate[key] for key in ("rate", "pitch", "volume")}
                if not _within_clamp(value, cfg):
                    continue
                if all(pair_ok(value, u["params"], cfg) for u in used):
                    chosen = value
                    break
            if chosen is None:
                item.update({"ok": False, "fixed": False, "unresolved": True})
                used.append({"name": item["name"], "params": base})
                continue
            item["fixed"] = True
            item["ok"] = True
            item["fixed_params"] = _format(chosen)
            item["params"] = chosen
            report["adjustments"].append({
                "key": item["key"],
                "name": item["name"],
                "voice": voice,
                "binding_id": item["binding_id"],
                "from": _format(base),
                "to": item["fixed_params"],
                "source": LABEL_PARAM_FIX,
            })
            used.append({"name": item["name"], "params": chosen})

    # 复用组：同一非旁白声线被 2 个以上角色使用
    for voice, members in groups.items():
        if voice == narrator_voice or len(members) < 2:
            continue
        pair_distances = [
            distance(a["params"], b["params"], cfg)
            for index, a in enumerate(members) for b in members[index + 1:]
        ]
        report["reuse_groups"].append({
            "voice": voice,
            "accent": next((m["accent"] for m in members if m["accent"]), ""),
            "count": len(members),
            "members": [{"name": m["name"], "params": _format(m["params"]),
                         "level": m["level"]} for m in members],
            "min_pair_distance": round(min(pair_distances), 2) if pair_distances else 0.0,
            "ok": all(m.get("ok") for m in members),
        })

    narrator_members = [
        {"name": item["name"], "params": _format(item["params"]),
         "label": item["label"]}
        for item in rows if item["voice"] == narrator_voice
    ]
    report["narrator_group"] = narrator_members

    for item in rows:
        pub = _public_row(item)
        pub["distance_to_narrator"] = round(distance(item["params"], narrator_params, cfg), 2)
        pub["conflicts"] = item.get("conflicts", [])
        report["rows"].append(pub)

    # 组合唯一性硬校验：“声线+参数”完全相同的组合必须为 0
    combos: dict[tuple, list[str]] = {}
    for item in rows:
        values = _format(item["params"])
        key = (item["voice"], values["rate"], values["pitch"], values["volume"])
        combos.setdefault(key, []).append(item["name"])
    narrator_key = (narrator_voice, _format(narrator_params)["rate"],
                    _format(narrator_params)["pitch"],
                    _format(narrator_params)["volume"])
    combos.setdefault(narrator_key, []).append("旁白")
    duplicates = [names for names in combos.values() if len(names) > 1]
    report["combo_conflicts"] = sum(len(names) - 1 for names in duplicates)
    report["conflict_details"] = duplicates

    distances = [(item["name"], distance(item["params"], narrator_params, cfg))
                 for item in rows]
    if distances:
        nearest = min(distances, key=lambda pair: pair[1])
        report["nearest_to_narrator"] = {
            "name": nearest[0], "distance": round(nearest[1], 2)}

    report["dialect_rows"] = [
        {"name": item["name"], "voice": item["voice"], "label": item["label"],
         "dialect": item["dialect"], "accent": item["accent"],
         "evidence": item["dialect_evidence"], "method": item["dialect_method"],
         "confidence": item["dialect_confidence"]}
        for item in rows if item["dialect"] or item["accent"]
    ]
    report["dialect_enabled"] = bool(report["dialect_rows"])

    report["same_voice_roles"] = sum(1 for item in rows
                                    if item["label"] == LABEL_BORROWED)
    report["independent_roles"] = sum(1 for item in rows
                                      if item["label"] == LABEL_INDEPENDENT)
    report["shared_roles"] = sum(1 for item in rows
                                 if item["label"] == LABEL_SHARED)
    report["borrowed_roles"] = report["same_voice_roles"]
    report["unknown_fallback_roles"] = sum(1 for item in rows
                                           if item["label"] == LABEL_UNKNOWN)
    report["compliant"] = sum(1 for item in rows if item.get("ok"))
    report["param_fix"] = sum(1 for item in rows if item.get("fixed"))
    report["unresolved"] = sum(1 for item in rows if item.get("unresolved"))
    report["manual_conflict"] = sum(1 for item in rows if item.get("manual_conflict"))
    return report


def _public_row(item: dict) -> dict:
    values = item.get("fixed_params") or _format(item["params"])
    return {
        "key": item["key"],
        "name": item["name"],
        "stage": item["stage"],
        "gender": item["gender"],
        "voice": item["voice"],
        "label": item["label"],
        "level": item.get("level"),
        "same_voice": item.get("level") == 3,
        "rate": values["rate"],
        "pitch": values["pitch"],
        "volume": values["volume"],
        "source": item["source"],
        "accent": item.get("accent") or "",
        "dialect": item.get("dialect") or "",
        "dialect_evidence": item.get("dialect_evidence") or "",
        "ok": bool(item.get("ok")),
        "fixed": bool(item.get("fixed")),
        "param_fix": bool(item.get("fixed")),
        "unresolved": bool(item.get("unresolved")),
        "manual_conflict": bool(item.get("manual_conflict")),
        "distance_to_narrator": 0.0,
        "conflicts": [],
    }
