"""阶段 0：只读分档报告（模拟阶段 A，不写绑定、不合成音频）。

用法：
    python tools/tier_report.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import appearance, voices_catalog  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
REGISTRY = ROOT / CFG["paths"]["voices"] / "registry.json"


def binding_of(registry: dict, cid: str) -> dict:
    for binding in (registry.get("voice_bindings") or {}).values():
        if binding.get("character_id") == cid:
            return binding
    return {}


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    catalog = voices_catalog.load_catalog(
        (CFG.get("casting") or {}).get("catalog", {}).get("path", "voices/catalog.edge.json"),
        ROOT)
    narrator_voice = ((CFG.get("tts") or {}).get("engines", {}).get("edge") or {}).get(
        "narrator_voice") or CFG["casting"]["narrator"]["default_voice"]
    tier_conf = appearance.tiering(CFG)
    reclaim_conf = appearance.reclaim(CFG)
    out_dir = (ROOT / CFG["paths"]["outputs"] / "v5_appearance").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"mode": "dry_run", "wrote_bindings": False,
               "tiering": tier_conf, "reclaim": reclaim_conf, "books": {}}

    for book_id, meta in (registry.get("appearance_meta", {}).get("books") or {}).items():
        rows = []
        for cid, ident in (registry.get("identities") or {}).items():
            stat = (ident.get("appearance_by_book") or {}).get(book_id)
            if not stat:
                continue
            tier = appearance.tier_of(stat, CFG, bool(ident.get("generic")))
            binding = binding_of(registry, cid)
            rows.append({
                "character_id": cid, "name": ident.get("canonical_name"),
                "gender": ident.get("gender"), "generic": bool(ident.get("generic")),
                "stage": binding.get("age_stage") or "", "tier": tier,
                "chapters_seen": stat["chapters_seen"], "total_lines": stat["total_lines"],
                "total_chars": stat["total_chars"], "share": stat["share"],
                "idle_chapters": stat["idle_chapters"],
                "voice": binding.get("voice") or "", "ladder_level": binding.get("ladder_level"),
                "simulated": "independent" if tier == "main" else "shared",
                "dormant_candidate": appearance.dormant_candidate(stat, CFG, tier),
            })
        rows.sort(key=lambda item: (-item["total_chars"], item["name"]))
        tiers = defaultdict(int)
        for row in rows:
            tiers[row["tier"]] += 1
        # 按 性别×年龄段 估算槽位需求
        groups = defaultdict(lambda: {"current_voices": set(), "main": 0, "shared": 0})
        for row in rows:
            if not row["voice"] or row["voice"] == narrator_voice:
                continue
            if row["stage"] == "":
                continue
            key = (row["gender"] or "未知", row["stage"])
            groups[key]["current_voices"].add(row["voice"])
            if row["tier"] == "main":
                groups[key]["main"] += 1
            else:
                groups[key]["shared"] += 1
        group_rows = []
        freed_total = 0
        for (gender, stage), item in sorted(groups.items()):
            capacity = len((catalog.get("pool_capacity", {}).get("main", {})
                            .get(stage, {}).get(gender, []))) if gender in ("男", "女") else 0
            needed = item["main"] + (1 if item["shared"] else 0)
            freed = max(0, len(item["current_voices"]) - needed)
            freed_total += freed
            group_rows.append({"gender": gender, "stage": stage,
                               "current_distinct": len(item["current_voices"]),
                               "main": item["main"], "shared": item["shared"],
                               "capacity": capacity, "needed": needed, "freed": freed,
                               "voices": sorted(item["current_voices"])})
        # 女声主池占用
        female_pool = set()
        for stage, buckets in (catalog.get("pool_capacity", {}).get("main", {}) or {}).items():
            if stage.startswith("_") or not isinstance(buckets, dict):
                continue
            female_pool.update(buckets.get("女") or [])
        female_usage = appearance.voice_usage_by_gender(registry, narrator_voice, catalog)
        female_rows = []
        for voice in sorted(female_pool):
            holders = female_usage.get(voice, [])
            names = []
            for holder in holders:
                ident = (registry.get("identities") or {}).get(holder["character_id"]) or {}
                names.append(f"{ident.get('canonical_name') or holder['character_id']}"
                             f"({holder.get('stage')}/L{holder.get('ladder_level')})")
            female_rows.append({"voice": voice, "holders": names, "count": len(names)})
        idle_dist = defaultdict(int)
        for row in rows:
            if row["tier"] in ("minor", "generic"):
                idle_dist[row["idle_chapters"]] += 1
        dormant_by_n = {n: sum(1 for row in rows if row["tier"] in ("minor", "generic")
                               and row["idle_chapters"] >= n
                               and row["total_lines"] < int(reclaim_conf["min_lines_keep"])
                               and row["share"] < float(reclaim_conf["max_share"]))
                        for n in (2, 3, 4, 5, 6)}
        payload["books"][book_id] = {
            "chapter_count": meta.get("chapter_count"), "roles": rows,
            "tiers": dict(tiers), "slot_groups": group_rows,
            "freed_slots": freed_total, "female_pool": female_rows,
            "idle_distribution": dict(sorted(idle_dist.items())),
            "dormant_by_n": dormant_by_n,
        }
        print(f"\n=== {book_id} | {meta.get('chapter_count')} 章 | 模拟分档（不写绑定）===")
        print(f"{'角色':<12}{'档位':<8}{'章':>3}{'句':>5}{'占比':>7}{'缺席':>5}"
              f"{'现声线':<24}{'级别':>4}{'模拟':<10}{'dormant':>8}")
        for row in rows:
            print(f"{row['name']:<12}{row['tier']:<8}{row['chapters_seen']:>3}{row['total_lines']:>5}"
                  f"{row['share'] * 100:>6.1f}%{row['idle_chapters']:>5}"
                  f"{(row['voice'] or '—'):<24}{row['ladder_level'] or 0:>4}"
                  f"{row['simulated']:<10}{('是' if row['dormant_candidate'] else '否'):>8}")
        print(f"档位：main {tiers['main']} / support {tiers['support']} / "
              f"minor {tiers['minor']} / generic {tiers['generic']}；"
              f"模拟可腾出槽位 {freed_total}")
        for item in group_rows:
            print(f"  {item['gender']}{item['stage']}: 现有 {item['current_distinct']} 条声线 "
                  f"→ 需要 {item['needed']}（独立 {item['main']} + 共享池 "
                  f"{1 if item['shared'] else 0}），容量 {item['capacity']}，可释放 {item['freed']}")
        print("女声主池占用：")
        for item in female_rows:
            print(f"  {item['voice']}: {item['count']} 人 "
                  f"{'、'.join(item['holders']) if item['holders'] else '（空闲）'}")
        print(f"minor/generic 缺席分布：{dict(sorted(idle_dist.items()))}")
        print(f"按 N 的 dormant 候选数：{dormant_by_n}")

    target = out_dir / "tier_report.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n只读报告：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
