"""按书累计出场统计 + 分档判定（阶段 0，只读绑定）。

统计来源：`novels/*/scripts/*.json` 的 dialogue segments（旁白不计入角色）。
落盘位置：`registry.json -> identities[cid].appearance_by_book[book_id]`。
"""
from __future__ import annotations

import re
import time
from pathlib import Path

TIER_MAIN = "main"
TIER_SUPPORT = "support"
TIER_MINOR = "minor"
TIER_GENERIC = "generic"

DEFAULT_TIERING = {
    "enabled": False,
    "main_min_chapters": 2,
    "main_min_lines": 40,
    "main_min_share": 0.08,
    "shared_scope": "chapter",
}

DEFAULT_RECLAIM = {
    "enabled": False,
    "idle_chapters": 4,
    "min_lines_keep": 20,
    "max_share": 0.05,
    "dormant_to_lent_after": 1,
    "lease_chapters": 3,
    "reassign_on_return": "buyback_first",
}


def tiering(cfg: dict) -> dict:
    conf = ((cfg.get("casting") or {}).get("voice_rules") or {}).get("tiering") or {}
    return {key: conf.get(key, default) for key, default in DEFAULT_TIERING.items()}


def reclaim(cfg: dict) -> dict:
    conf = ((cfg.get("casting") or {}).get("voice_rules") or {}).get("reclaim") or {}
    return {key: conf.get(key, default) for key, default in DEFAULT_RECLAIM.items()}


def role_weight_conf(cfg: dict) -> dict:
    return ((cfg.get("casting") or {}).get("consistency") or {}).get("role_weight") or {}


def script_records(root: Path) -> dict[str, list[dict]]:
    """按 book_id 收集有效剧本，跳过失败稿与基线对比稿。"""
    import json

    books: dict[str, list[dict]] = {}
    for path in sorted((root / "novels").glob("*/scripts/*.json")):
        if "__" in path.stem or ".failed" in path.stem:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not record.get("segments"):
            continue
        book_id = record.get("book_id") or path.parent.parent.name
        books.setdefault(book_id, []).append(record)
    return books


def chapter_stats(record: dict) -> dict:
    """单章台词统计：speaker -> {lines, chars}。narration 不计入。"""
    stats: dict[str, dict] = {}
    for seg in record.get("segments", []):
        if seg.get("type") != "dialogue":
            continue
        speaker = str(seg.get("speaker") or "").strip()
        if not speaker:
            continue
        text = str(seg.get("text") or "")
        item = stats.setdefault(speaker, {"lines": 0, "chars": 0})
        item["lines"] += 1
        item["chars"] += len(re.sub(r"\s+", "", text))
    return stats


def aggregate_book(records: list[dict]) -> dict:
    """汇总一本书：speaker -> 累计章数/句数/字数 + 全书总字数。"""
    chapters = sorted({int(record.get("chapter_index") or 0) for record in records})
    max_chapter = max(chapters) if chapters else -1
    speakers: dict[str, dict] = {}
    for record in records:
        chapter = int(record.get("chapter_index") or 0)
        for speaker, item in chapter_stats(record).items():
            entry = speakers.setdefault(speaker, {
                "chapters_seen": set(), "total_lines": 0, "total_chars": 0,
                "first_seen_chapter": chapter, "last_seen_chapter": chapter,
            })
            entry["chapters_seen"].add(chapter)
            entry["total_lines"] += item["lines"]
            entry["total_chars"] += item["chars"]
            entry["first_seen_chapter"] = min(entry["first_seen_chapter"], chapter)
            entry["last_seen_chapter"] = max(entry["last_seen_chapter"], chapter)
    total_chars = sum(item["total_chars"] for item in speakers.values())
    rows: dict[str, dict] = {}
    for speaker, item in speakers.items():
        rows[speaker] = {
            "chapters_seen": len(item["chapters_seen"]),
            "chapters": sorted(item["chapters_seen"]),
            "total_lines": item["total_lines"],
            "total_chars": item["total_chars"],
            "share": round(item["total_chars"] / total_chars, 4) if total_chars else 0.0,
            "first_seen_chapter": item["first_seen_chapter"],
            "last_seen_chapter": item["last_seen_chapter"],
            "idle_chapters": max(0, max_chapter - item["last_seen_chapter"]),
        }
    return {"book_id": None, "chapters": chapters, "chapter_count": len(chapters),
            "total_chars": total_chars, "roles": rows}


def normalized(name: str) -> str:
    from .casting import strip_title

    return strip_title(str(name or "").strip())


def resolve_identity(name: str, book_id: str, registry: dict, generic_names) -> tuple[str, str]:
    """返回 (cid, 匹配方式)。优先别名，其次泛称/本名精确匹配，最后稳定 id。"""
    from . import casting

    aliases = registry.get("character_aliases") or {}
    base = normalized(name)
    for key in (name, base):
        cid = aliases.get(key)
        if cid and cid in (registry.get("identities") or {}):
            return cid, "alias"
    for cid, ident in (registry.get("identities") or {}).items():
        if ident.get("canonical_name") == base and (
                book_id in (ident.get("books") or []) or not ident.get("books")):
            return cid, "canonical"
    cid = casting.identity_id(name, book_id, generic_names)
    if cid in (registry.get("identities") or {}):
        return cid, "identity_id"
    for cid, ident in (registry.get("identities") or {}).items():
        if casting.is_generic(base, generic_names) and ident.get("canonical_name") == base:
            return cid, "generic_name"
    return cid, "unmatched"


def apply_to_registry(registry: dict, book_id: str, aggregate: dict,
                      generic_names) -> dict:
    """把累计统计写入 identities[cid].appearance_by_book，不动任何绑定。"""
    rows = aggregate["roles"]
    unresolved: list[str] = []
    matched: dict[str, dict] = {}
    for speaker, stat in rows.items():
        cid, how = resolve_identity(speaker, book_id, registry, generic_names)
        ident = (registry.get("identities") or {}).get(cid)
        if not ident:
            unresolved.append(speaker)
            continue
        ident.setdefault("appearance_by_book", {})[book_id] = dict(stat, source=how)
        matched[cid] = dict(stat, speaker=speaker, source=how)
    registry.setdefault("appearance_meta", {})
    registry["appearance_meta"].setdefault("books", {})[book_id] = {
        "chapters": aggregate["chapters"],
        "chapter_count": aggregate["chapter_count"],
        "total_chars": aggregate["total_chars"],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    registry["appearance_meta"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return {"book_id": book_id, "matched": matched, "unresolved": unresolved,
            "chapter_count": aggregate["chapter_count"],
            "chapters": aggregate["chapters"],
            "total_chars": aggregate["total_chars"]}


def tier_of(stat: dict, cfg: dict, generic: bool = False) -> str:
    """分档：main 靠门禁；generic 一律 generic；其余按 support/minor 句数分。"""
    conf = tiering(cfg)
    weight = role_weight_conf(cfg)
    if generic:
        return TIER_GENERIC
    is_main = (
        int(stat.get("chapters_seen") or 0) >= int(conf["main_min_chapters"])
        and (
            int(stat.get("total_lines") or 0) >= int(conf["main_min_lines"])
            or float(stat.get("share") or 0) >= float(conf["main_min_share"])
        )
    )
    if is_main:
        return TIER_MAIN
    if int(stat.get("total_lines") or 0) >= int(weight.get("support_min_lines", 10)):
        return TIER_SUPPORT
    return TIER_MINOR


def dormant_candidate(stat: dict, cfg: dict, tier: str) -> bool:
    """保守判定：仅 minor/generic；缺席章数与占比双门禁。"""
    conf = reclaim(cfg)
    if tier not in (TIER_MINOR, TIER_GENERIC):
        return False
    if int(stat.get("idle_chapters") or 0) < int(conf["idle_chapters"]):
        return False
    if int(stat.get("total_lines") or 0) >= int(conf["min_lines_keep"]):
        return False
    return float(stat.get("share") or 0) < float(conf["max_share"])


def voice_usage_by_gender(registry: dict, narrator_voice: str,
                          catalog: dict) -> dict:
    """女声/男声主池占用：返回 {voice: [角色...]}。"""
    from . import voices_catalog

    usage: dict[str, list[dict]] = {}
    for binding in (registry.get("voice_bindings") or {}).values():
        voice = str(binding.get("voice") or "")
        if not voice or voice == narrator_voice:
            continue
        if voices_catalog.is_accented(voice, catalog):
            continue
        usage.setdefault(voice, []).append({
            "character_id": binding.get("character_id"),
            "stage": binding.get("age_stage"),
            "ladder_level": binding.get("ladder_level"),
            "source": binding.get("source"),
        })
    return usage
