"""v4 步骤 4：一致性检查（撞声 / 漂移 / 角色权重）。

纯函数，不改 registry；WebUI 与 caster 只负责展示和决策。
"""
from __future__ import annotations

def role_weight(profile: dict, cfg: dict) -> float:
    """权重全部来自 config.yaml -> casting.consistency.role_weight。"""
    conf = ((cfg.get("casting") or {}).get("consistency") or {}).get("role_weight") or {}
    weights = conf["weights"]
    main_min = int(conf["main_min_lines"])
    support_min = int(conf["support_min_lines"])
    lines = int(profile.get("line_count") or 0)
    if profile.get("generic"):
        return float(weights["generic"])
    if lines >= main_min:
        return float(weights["main"])
    if lines >= support_min:
        return float(weights["support"])
    return float(weights["minor"])


def collision_report(cast: dict, narrator_voice: str, cfg: dict) -> dict:
    """按“同声线其他角色权重之和”计算撞声分。

    返回 {by_voice: {voice: {score, level, members}}, warnings: [...]}。
    同一角色的不同阶段共用声线不计数；旁白同声线额外 +1.0，仅警告。
    """
    groups: dict[str, list[dict]] = {}
    for key, profile in (cast or {}).items():
        voice = profile.get("edge_voice")
        if not voice:
            continue
        groups.setdefault(voice, []).append({
            "key": key,
            "name": profile.get("name") or key,
            "stage": profile.get("age_stage") or profile.get("age_band"),
            "weight": role_weight(profile, cfg),
            "generic": bool(profile.get("generic")),
        })

    by_voice: dict[str, dict] = {}
    warnings: list[dict] = []
    ccfg = ((cfg.get("casting") or {}).get("consistency") or {}).get("collision") or {}
    warn_at = float(ccfg["warn_score"])
    block_at = float(ccfg["block_auto_score"])
    hint_at = float(ccfg["hint_score"])
    narrator_score = float(ccfg["narrator_overlap_score"])
    group_exempt = bool(ccfg["allow_group_exemption"])

    for voice, members in groups.items():
        weights_by_name: dict[str, float] = {}
        for m in members:
            # 普通角色按本名合并（不同阶段不重复计数）；泛称按条目各自计数
            ident = m["key"] if m["generic"] else m["name"]
            weights_by_name[ident] = max(weights_by_name.get(ident, 0.0), m["weight"])
        # 同一角色不同阶段只计一次；分数为所有共用该声线的角色权重之和
        score = sum(weights_by_name.values()) if len(weights_by_name) > 1 else 0.0
        level = "ok"
        if score >= block_at:
            level = "error"
        elif score >= warn_at:
            level = "warn"
        elif score >= hint_at:
            level = "hint"
        if level == "error" and group_exempt and all(m["generic"] for m in members):
            level = "warn"
        by_voice[voice] = {"score": round(score, 2), "level": level, "members": members}
        if level != "ok":
            warnings.append({
                "type": "collision",
                "voice": voice,
                "score": round(score, 2),
                "level": level,
                "message": f"{len(members)} 个角色共用 {voice}，撞声分 {score:.1f}",
                "members": [m["name"] for m in members],
                "exempted": level == "warn" and score >= block_at,
            })

    if narrator_voice and narrator_voice in by_voice:
        item = by_voice[narrator_voice]
        item["narrator_overlap"] = True
        warnings.append({
            "type": "narrator_collision",
            "voice": narrator_voice,
            "score": item["score"] + narrator_score,
            "level": "warn",
            "message": f"旁白与 {len(item['members'])} 个角色共用 {narrator_voice}，仅警告",
            "members": [m["name"] for m in item["members"]],
        })
    return {"by_voice": by_voice, "warnings": warnings}


def auto_recommend_allowed(voice: str, report: dict) -> bool:
    """撞声分 >=2.0 时禁止自动推荐；手动仍可强制。"""
    item = (report or {}).get("by_voice", {}).get(voice)
    return not (item and item.get("level") == "error")


def drift_report(registry: dict, recommend, cfg: dict | None = None) -> list[dict]:
    """对 migrated/auto 绑定比较“当前配置推荐值”，只提示，不自动改写。

    recommend(binding) -> {"voice","rate","pitch","volume"} 或 None。
    手动 binding 永不漂移。
    """
    out = []
    for bid, binding in (registry.get("voice_bindings") or {}).items():
        if binding.get("source") not in ("migrated", "auto"):
            continue
        rec = recommend(binding)
        if not rec or not rec.get("voice"):
            continue
        # 空值表示“沿用配置推荐”，不算漂移；只比较显式写死的字段
        diff = {k: (binding.get(k), rec.get(k)) for k in ("voice", "rate", "pitch", "volume")
                if binding.get(k) and binding.get(k) != rec.get(k)}
        if not diff:
            continue
        out.append({
            "type": "drift",
            "binding_id": bid,
            "scope": binding.get("scope"),
            "character_id": binding.get("character_id"),
            "age_stage": binding.get("age_stage"),
            "current": {k: binding.get(k) for k in ("voice", "rate", "pitch", "volume")},
            "recommended": rec,
            "diff": diff,
            "message": "配置推荐值已变化，是否应用新推荐？",
            "actions": ["apply", "keep", "ignore"],
        })
    return out
