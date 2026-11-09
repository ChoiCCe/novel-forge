"""v4 数据模型：身份层 / 阶段层 / 绑定层 + 分层优先级解析。

运行时 registry **只使用 v2 结构**：
- `schema_version: 2`
- `identities`: 身份层（id -> 身份 + stage_profiles）
- `character_aliases`: 别名 -> identity id
- `voice_bindings`: 覆盖层（id -> binding）
- `voice_events`: 变更历史
- `narrator` / `book_settings`: 旁白与单书配置

`characters` / `locks` / `voice_changes` 已删除，运行时不读取也不镜像。
`migrate_v1` 仅保留给离线 `tools/migrate_registry.py` 使用，任何运行时路径都不调用它。
"""
from __future__ import annotations

import time

# 优先级：数值越大越优先
PRIORITY = {
    "book_character_stage": 100,
    "book_character": 95,
    "character_stage": 90,
    "character": 85,
    "book_stage": 80,
    "book": 70,
    "global_stage": 60,
    "global": 55,
    "lock": 50,
    "config": 40,
    "inferred": 30,
    "fallback": 10,
}

SOURCE_RANK = {"manual": 1, "copy": 1, "migrated": 2, "auto": 3, "fallback": 4}

TITLE_PREFIXES = ("大将军", "将军", "大帅", "老爷", "公子", "先生", "大人")

DEFAULT_GENERIC = (
    "同学", "随从", "路人", "士兵", "一个孩子", "大的孩子", "掌柜", "小二",
    "宫女", "侍卫", "家丁", "宾客", "学生",
)


def strip_title(name: str) -> str:
    """去掉称谓前缀；保留至少 2 字，避免把“大将军X”截成单字。"""
    for prefix in TITLE_PREFIXES:
        if name.startswith(prefix) and len(name) - len(prefix) >= 2:
            return name[len(prefix):]
    return name


def is_generic(name: str, generic_names=DEFAULT_GENERIC) -> bool:
    base = strip_title(name or "")
    return base in set(generic_names)


def identity_id(name: str, book_id: str | None = None, generic_names=DEFAULT_GENERIC) -> str:
    """稳定身份 ID：普通角色用归一化本名；泛称加 @book_id，避免跨书串身份。"""
    base = strip_title(name or "")
    if is_generic(base, generic_names):
        return f"{base}@{book_id or 'unknown'}"
    return base


def _parse_key(key: str) -> tuple[str, str, str]:
    parts = key.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return key, "未知", "未知"


# ---------------------------------------------------------------------------- 迁移
def migrate_v1(reg: dict, *, cfg: dict | None = None, now: str | None = None,
               backup: str = "", book_id: str | None = None) -> tuple[dict, dict]:
    """v1 -> v2。幂等：schema_version>=2 时原样返回并标记 skipped。

    返回 (new_registry, report)；report 含计数、合并、性别冲突、失效声线。
    """
    now = now or time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if int(reg.get("schema_version", 1)) >= 2:
        return reg, {"skipped": True, "reason": "already_schema_v2"}

    age_cfg = ((cfg or {}).get("casting", {}) or {}).get("age", {}) or {}
    default_stage = str(age_cfg.get("default_stage") or "未知")
    generic_names = tuple(((cfg or {}).get("casting", {}).get("consistency", {})
                           or {}).get("generic_names") or DEFAULT_GENERIC)
    chars: dict = reg.get("characters", {}) or {}
    locks: dict = reg.get("locks", {}) or {}
    out = dict(reg)
    identities: dict[str, dict] = {}
    aliases: dict[str, str] = {}
    bindings: dict[str, dict] = {}
    events: list[dict] = []
    merged: list[dict] = []
    gender_conflicts: list[dict] = []
    unavailable: list[dict] = []

    for key, entry in chars.items():
        name = entry.get("name") or _parse_key(key)[0]
        cid = identity_id(name, book_id, generic_names)
        ident = identities.get(cid)
        if ident is None:
            ident = {
                "id": cid,
                "canonical_name": strip_title(name),
                "aliases": [],
                "gender": entry.get("gender", "未知"),
                "gender_source": "migrated",
                "gender_confidence": 0.7 if entry.get("gender") in ("男", "女") else 0.3,
                "gender_confirmed": False,
                "generic": is_generic(name, generic_names),
                "persona": entry.get("persona", ""),
                "books": [],
                "stage_profiles": {},
                "created_at": now,
                "updated_at": now,
            }
            identities[cid] = ident
        else:
            if ident.get("gender") and entry.get("gender") and entry["gender"] != ident["gender"]:
                gender_conflicts.append({"character_id": cid, "existing": ident["gender"],
                                         "incoming": entry["gender"], "source_key": key})
            merged.append({"character_id": cid, "key": key})
        if name != strip_title(name) and name not in ident["aliases"]:
            ident["aliases"].append(name)
        aliases[name] = cid
        aliases[strip_title(name)] = cid

        stage = entry.get("age_stage") or entry.get("age_band") or default_stage
        profile = {
            "temperament": entry.get("temperament", "中性"),
            "rate": entry.get("rate", "+0%"),
            "pitch": entry.get("pitch", "+0Hz"),
            "volume": entry.get("volume", "+0%"),
            "prosody_source": entry.get("prosody_source", "temperament"),
            "first_seen_chapter": entry.get("first_seen_chapter"),
            "last_seen_chapter": entry.get("last_seen_chapter"),
            "age_note": entry.get("age_note", ""),
            "age_confidence": entry.get("age_confidence"),
            "age_source": entry.get("updated_by", "migrated"),
            "locked": False,
            "transition": None,
            "checked_chapters": entry.get("checked_chapters", []),
            "legacy_key": key,
            "legacy_edge_voice": entry.get("legacy_edge_voice", ""),
        }
        ident["stage_profiles"][stage] = profile

    # 旧锁 -> 绑定层：保留原声线；manual 来源高于 migrated
    for lock_key, lock in locks.items():
        if "|" not in lock_key:
            name, stage = lock_key, lock.get("stage") or default_stage
        else:
            name, stage = lock_key.split("|", 1)
        cid = aliases.get(name) or identity_id(name, book_id, generic_names)
        source = "manual" if str(lock.get("updated_by", "")).lower() in ("manual", "webui") else "migrated"
        bid = f"lock:{lock_key}"
        legacy_voice = lock.get("legacy_edge_voice") or ""
        bindings[bid] = {
            "id": bid,
            "scope": "character_stage",
            "book_id": None,
            "character_id": cid,
            "age_stage": stage,
            "voice": lock.get("edge_voice", ""),
            "rate": None, "pitch": None, "volume": None,
            "source": source,
            "locked": True,
            "priority": PRIORITY["character_stage"],
            "note": "v1 历史锁定迁移",
            "legacy_voice": legacy_voice,
            "created_at": now,
            "updated_by": lock.get("updated_by", "migrated"),
        }
        if cid in identities and stage in identities[cid]["stage_profiles"]:
            identities[cid]["stage_profiles"][stage]["locked"] = True
        if legacy_voice and lock.get("edge_voice"):
            unavailable.append({"binding_id": bid, "from": legacy_voice, "to": lock["edge_voice"]})

    for i, ev in enumerate(reg.get("voice_changes", []) or []):
        events.append({
            "id": f"evt:migrated:{i}",
            "book_id": ev.get("book_id"),
            "chapter": ev.get("chapter"),
            "character_id": identity_id(ev.get("name", ""), book_id, generic_names),
            "from_stage": ev.get("from_stage"),
            "to_stage": ev.get("to_stage"),
            "from_voice": ev.get("from_voice"),
            "to_voice": ev.get("to_voice"),
            "reason": ev.get("evidence", ""),
            "evidence": ev.get("evidence", ""),
            "confidence": ev.get("confidence"),
            "source": "migrated",
            "status": "applied",
            "created_at": now,
        })

    narrator_cfg = ((cfg or {}).get("casting", {}) or {}).get("narrator", {}) or {}
    edge_cfg = ((cfg or {}).get("tts", {}).get("engines", {}) or {}).get("edge", {}) or {}
    caster_cfg = (cfg or {}).get("caster", {}) or {}
    narrator = {
        "global": {
            "voice": narrator_cfg.get("default_voice") or edge_cfg.get("narrator_voice"),
            "gender_preference": narrator_cfg.get("gender_preference", "auto"),
            "rate": narrator_cfg.get("rate") or caster_cfg.get("narrator_rate") or "+0%",
            "pitch": narrator_cfg.get("pitch") or caster_cfg.get("narrator_pitch") or "+0Hz",
            "volume": narrator_cfg.get("volume") or caster_cfg.get("narrator_volume") or "+0%",
            "source": "config",
            "locked": False,
        },
        "books": {},
    }

    out["schema_version"] = 2
    out["voice_map_version"] = reg.get("voice_map_version", "edge-v5")
    out["identities"] = identities
    out["character_aliases"] = aliases
    out["voice_bindings"] = bindings
    out["voice_events"] = events
    out["narrator"] = narrator
    out["migration"] = {
        "from": 1,
        "at": now,
        "backup": backup,
        "counts": {"characters": len(chars), "identities": len(identities),
                   "stages": sum(len(v["stage_profiles"]) for v in identities.values()),
                   "bindings": len(bindings), "events": len(events)},
    }

    report = {
        "skipped": False,
        "counts": out["migration"]["counts"],
        "merged": merged,
        "gender_conflicts": gender_conflicts,
        "unavailable_voices": unavailable,
        "generic_identities": [cid for cid, v in identities.items() if v.get("generic")],
        "zero_loss": len(chars) == sum(len(v["stage_profiles"]) for v in identities.values()),
    }
    return out, report


# ---------------------------------------------------------------------------- 解析
def _binding_matches(binding: dict, *, book_id: str | None, cid: str, stage: str | None) -> bool:
    scope = binding.get("scope", "")
    if not binding.get("voice"):
        return False
    b_book = binding.get("book_id")
    b_char = binding.get("character_id")
    b_stage = binding.get("age_stage")
    if scope == "global":
        return True
    if scope == "global_stage":
        return b_stage is None or b_stage == stage
    if scope == "book":
        return b_book == book_id
    if scope == "book_stage":
        return b_book == book_id and (b_stage is None or b_stage == stage)
    if scope in ("book_character", "book_character_stage"):
        return b_book == book_id and b_char == cid and (
            scope == "book_character" or b_stage is None or b_stage == stage)
    if scope == "character":
        return b_char == cid
    if scope == "character_stage":
        return b_char == cid and (b_stage is None or b_stage == stage)
    return False


def _sort_key(binding: dict) -> tuple:
    return (
        binding.get("priority") or PRIORITY.get(binding.get("scope", ""), 0),
        -SOURCE_RANK.get(binding.get("source", "auto"), 3),
        str(binding.get("updated_at") or binding.get("created_at") or ""),
    )


def resolve_effective(registry: dict, *, name: str, age_stage: str | None,
                      book_id: str | None = None, gender: str = "",
                      config_voice: str = "", catalog: dict | None = None,
                      generic_names=DEFAULT_GENERIC) -> dict:
    """按分层优先级解析当前生效声线，并返回可追溯来源与候选链。

    - 绑定命中：voice/韵律取 binding，`source` 为 binding.source；
    - 无绑定：使用 config_voice（配置映射，priority 40）；
    - 有绑定但声线在目录中不可用：自动平替，并在 chain 里标注。
    """
    from . import voices_catalog

    generic = is_generic(name, generic_names)
    cid = (registry.get("character_aliases") or {}).get(name) or identity_id(
        name, book_id if generic else None, generic_names)

    bindings = (registry.get("voice_bindings") or {}).values()
    candidates = [b for b in bindings if _binding_matches(
        b, book_id=book_id, cid=cid, stage=age_stage)]
    candidates.sort(key=_sort_key, reverse=True)

    chain = [{"id": b.get("id"), "scope": b.get("scope"), "voice": b.get("voice"),
              "source": b.get("source"), "priority": b.get("priority") or PRIORITY.get(b.get("scope", ""))}
             for b in candidates]

    if candidates:
        winner = dict(candidates[0])
        voice = winner.get("voice", "")
        resolved = voices_catalog.resolve_voice(voice, catalog) if catalog else {
            "voice": voice, "fallback": False, "note": "", "overrides": {}}
        return {
            "voice": resolved["voice"],
            "rate": winner.get("rate"),
            "pitch": winner.get("pitch"),
            "volume": winner.get("volume"),
            "source": winner.get("source", "auto"),
            "binding_id": winner.get("id"),
            "priority": winner.get("priority") or PRIORITY.get(winner.get("scope", ""), 0),
            "chain": chain,
            "fallback": resolved.get("fallback", False),
            "note": resolved.get("note", ""),
        }

    fallback_note = ""
    voice = config_voice
    if voice and catalog:
        resolved = voices_catalog.resolve_voice(voice, catalog)
        voice = resolved["voice"]
        fallback_note = resolved.get("note", "")
        source = "fallback" if resolved.get("fallback") else "config"
    else:
        source = "config" if voice else "fallback"
    return {
        "voice": voice,
        "rate": None, "pitch": None, "volume": None,
        "source": source,
        "binding_id": None,
        "priority": PRIORITY["fallback"] if source == "fallback" else PRIORITY["config"],
        "chain": chain + [{"id": None, "scope": "config", "voice": config_voice,
                           "source": source, "priority": PRIORITY["config"]}],
        "fallback": source == "fallback",
        "note": fallback_note,
    }
