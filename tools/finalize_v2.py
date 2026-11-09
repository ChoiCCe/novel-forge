"""v4 收尾：把 registry 从“v2 + legacy 镜像”收口为纯 v2（一次性、幂等）。

步骤：
1. 备份 registry 到 voices/registry.v4-pre-legacy-drop.<时间>.bak.json（已存在则不覆盖）；
2. 把 legacy characters 的字段补进 identities / stage_profiles（v2 是唯一真源）；
3. 校验/补建 6 条 lock -> voice_bindings，并统一 priority/source/character_id；
4. voice_changes -> voice_events（仅当 voice_events 为空）；
5. 删除 characters / locks / voice_changes 三个 key；
6. 输出报告到 outputs/v4_finalize_report.json。

本工具只做离线收口，不被运行时调用；运行时不读取任何 legacy 字段。
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.casting import PRIORITY, identity_id, strip_title  # noqa: E402

REG = ROOT / "voices" / "registry.json"
REPORT = ROOT / "outputs" / "v4_finalize_report.json"

STAGE_FIELDS = (
    "temperament", "rate", "pitch", "volume", "prosody_source",
    "first_seen_chapter", "last_seen_chapter", "age_note", "age_confidence",
    "updated_by", "updated_by_chapter", "checked_chapters", "legacy_edge_voice",
    "voice_repair", "transition", "stage_entered_chapter", "age_evidence_log",
    "age_hold_reason", "engine", "ref_wav", "ref_text", "class_name",
)
IDENT_FIELDS = (
    "gender", "gender_confidence", "gender_evidence", "gender_source",
    "gender_confirmed", "generic", "persona", "voice_style", "default_emotion",
)


def _parse_key(key: str, default_stage: str) -> tuple[str, str]:
    parts = key.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0], parts[2]
    return key, default_stage


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def main() -> int:
    raw = REG.read_text(encoding="utf-8")
    reg = json.loads(raw)
    backup = None
    if any(key in reg for key in ("characters", "locks", "voice_changes")):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = ROOT / "voices" / f"registry.v4-pre-legacy-drop.{stamp}.bak.json"
        if backup.exists():
            backup = backup.with_name(backup.stem + "-1.json")
        shutil.copy2(REG, backup)
    else:
        backups = sorted((ROOT / "voices").glob("registry.v4-pre-legacy-drop.*.bak.json"))
        backup = backups[-1] if backups else None

    identities = reg.setdefault("identities", {})
    aliases = reg.setdefault("character_aliases", {})
    bindings = reg.setdefault("voice_bindings", {})
    default_stage = str(((reg.get("migration") or {}).get("default_stage")) or "中年")
    chars = reg.pop("characters", {}) or {}
    locks = reg.pop("locks", {}) or {}
    legacy_changes = reg.pop("voice_changes", []) or []

    def cid_for(name: str) -> str:
        return aliases.get(name) or identity_id(name)

    enriched_profiles = 0
    for key, entry in chars.items():
        name = entry.get("name") or _parse_key(key, default_stage)[0]
        stage = entry.get("age_stage") or entry.get("age_band") or _parse_key(key, default_stage)[1]
        cid = cid_for(name)
        ident = identities.get(cid)
        if ident is None:
            ident = {
                "id": cid, "canonical_name": strip_title(name), "aliases": [],
                "gender": entry.get("gender", "未知"), "generic": False,
                "books": [], "stage_profiles": {}, "created_at": _now(), "updated_at": _now(),
            }
            identities[cid] = ident
        for field in IDENT_FIELDS:
            if field in entry and entry[field] not in (None, ""):
                ident[field] = entry[field]
        if name != ident["canonical_name"] and name not in ident.setdefault("aliases", []):
            ident["aliases"].append(name)
        aliases[name] = cid
        aliases[ident["canonical_name"]] = cid
        profile = ident.setdefault("stage_profiles", {}).setdefault(stage, {})
        profile["legacy_key"] = key
        for field in STAGE_FIELDS:
            if field in entry and entry[field] is not None:
                profile[field] = entry[field]
        profile.setdefault("checked_chapters", [])
        profile.setdefault("transition", None)
        profile.setdefault("age_source", entry.get("updated_by", "migrated"))
        enriched_profiles += 1

    lock_report = []
    for lock_key, lock in locks.items():
        if "|" in lock_key:
            name, stage = lock_key.split("|", 1)
        else:
            name, stage = lock_key, lock.get("stage") or default_stage
        cid = cid_for(name)
        bid = f"lock:{name}|{stage}"
        voice = lock.get("edge_voice") or ""
        existing = bindings.get(bid) or {}
        branch = (identities.get(cid) or {}).get("stage_profiles", {}).get(stage, {})
        legacy_voice = (lock.get("legacy_edge_voice")
                        or branch.get("legacy_edge_voice")
                        or existing.get("legacy_voice", ""))
        binding = {
            "id": bid,
            "scope": "character_stage",
            "book_id": None,
            "character_id": cid,
            "age_stage": stage,
            "voice": voice or existing.get("voice", ""),
            "rate": existing.get("rate"),
            "pitch": existing.get("pitch"),
            "volume": existing.get("volume"),
            "source": "manual" if str(lock.get("updated_by", "")).lower() in ("manual", "webui")
            else "migrated",
            "locked": True,
            "priority": PRIORITY["character_stage"],
            "note": existing.get("note") or "v1 历史锁定迁移",
            "legacy_voice": legacy_voice,
            "created_at": existing.get("created_at") or lock.get("created_at") or _now(),
            "updated_at": _now(),
            "updated_by": lock.get("updated_by", "migrated"),
        }
        bindings[bid] = binding
        if cid in identities and stage in identities[cid].get("stage_profiles", {}):
            identities[cid]["stage_profiles"][stage]["locked"] = True
        lock_report.append({
            "lock_key": lock_key, "binding_id": bid, "character_id": cid,
            "voice_match": not voice or binding["voice"] == voice,
            "voice": binding["voice"],
        })

    # 幂等补全：即使 legacy locks 已删除，也从阶段档案回填历史声线与修复原因
    current_locks = []
    for bid, binding in bindings.items():
        if not str(bid).startswith("lock:"):
            continue
        cid = binding.get("character_id") or ""
        stage = binding.get("age_stage") or default_stage
        branch = (identities.get(cid) or {}).get("stage_profiles", {}).get(stage, {})
        if not binding.get("legacy_voice") and branch.get("legacy_edge_voice"):
            binding["legacy_voice"] = branch["legacy_edge_voice"]
        current_locks.append({
            "lock_key": f"{cid}|{stage}", "binding_id": bid, "character_id": cid,
            "voice_match": True, "voice": binding.get("voice", ""),
        })
    if not lock_report:
        lock_report = current_locks

    events = reg.setdefault("voice_events", [])
    if not events and legacy_changes:
        for index, item in enumerate(legacy_changes):
            name = item.get("name", "")
            events.append({
                "id": f"evt:migrated:{index}",
                "book_id": item.get("book_id"),
                "chapter": item.get("chapter"),
                "character_id": cid_for(name),
                "from_stage": item.get("from_stage"),
                "to_stage": item.get("to_stage"),
                "from_voice": item.get("from_voice"),
                "to_voice": item.get("to_voice"),
                "reason": item.get("evidence", ""),
                "evidence": item.get("evidence", ""),
                "confidence": item.get("confidence"),
                "source": "migrated",
                "status": "applied",
                "created_at": _now(),
            })

    reg["schema_version"] = 2
    reg["legacy_dropped_at"] = _now()
    (reg.setdefault("migration", {}))["legacy_dropped_at"] = reg["legacy_dropped_at"]

    total_profiles = sum(len(v.get("stage_profiles", {})) for v in identities.values())
    report = {
        "at": reg["legacy_dropped_at"],
        "backup": str(backup) if backup else None,
        "legacy_counts": {"characters": len(chars), "locks": len(locks),
                          "voice_changes": len(legacy_changes)},
        "v2_counts": {
            "identities": len(identities),
            "stage_profiles": sum(len(v.get("stage_profiles", {})) for v in identities.values()),
            "voice_bindings": len(bindings),
            "voice_events": len(events),
        },
        "enriched_profiles": enriched_profiles,
        "locks_migrated": sum(1 for item in lock_report if item["voice_match"]),
        "locks_total": len(lock_report),
        "locks": lock_report,
        "zero_loss": True if not chars else enriched_profiles == total_profiles,
        "legacy_keys_remaining": [key for key in ("characters", "locks", "voice_changes")
                                  if key in reg],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp = REG.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(REG)
    print(json.dumps({key: report[key] for key in (
        "backup", "legacy_counts", "v2_counts", "locks_migrated", "locks_total",
        "zero_loss", "legacy_keys_remaining")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
