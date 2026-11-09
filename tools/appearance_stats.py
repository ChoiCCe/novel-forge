"""阶段 0：按书累计出场统计（只改 identities.appearance_by_book，不动绑定）。

用法：
    python tools/appearance_stats.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import appearance  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
REGISTRY = ROOT / CFG["paths"]["voices"] / "registry.json"


def generic_names() -> list[str]:
    conf = ((CFG.get("casting") or {}).get("consistency") or {})
    return list(conf.get("generic_names") or [])


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    books = appearance.script_records(ROOT)
    out_dir = (ROOT / CFG["paths"]["outputs"] / "v5_appearance").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "tiering": appearance.tiering(CFG), "books": {}}
    for book_id, records in books.items():
        aggregate = appearance.aggregate_book(records)
        aggregate["book_id"] = book_id
        result = appearance.apply_to_registry(registry, book_id, aggregate, generic_names())
        rows = []
        for cid, stat in result["matched"].items():
            ident = registry["identities"][cid]
            tier = appearance.tier_of(stat, CFG, bool(ident.get("generic")))
            rows.append({"character_id": cid, "name": ident.get("canonical_name"),
                         "speaker": stat["speaker"], "tier": tier,
                         "gender": ident.get("gender"), "generic": bool(ident.get("generic")),
                         "chapters_seen": stat["chapters_seen"], "total_lines": stat["total_lines"],
                         "total_chars": stat["total_chars"], "share": stat["share"],
                         "last_seen_chapter": stat["last_seen_chapter"],
                         "idle_chapters": stat["idle_chapters"], "match": stat["source"]})
        rows.sort(key=lambda item: (-item["total_chars"], item["name"]))
        payload["books"][book_id] = {
            "chapter_count": result["chapter_count"], "chapters": result["chapters"],
            "total_chars": result["total_chars"], "unresolved": result["unresolved"],
            "roles": rows,
        }
        print(f"\n=== {book_id} | {result['chapter_count']} 章 | 台词 {result['total_chars']} 字 ===")
        print(f"{'角色':<12}{'档位':<8}{'章':>3}{'句':>5}{'字数':>7}{'占比':>7}"
              f"{'末章':>5}{'缺席':>5}{'匹配':>10}")
        for row in rows:
            print(f"{row['name']:<12}{row['tier']:<8}{row['chapters_seen']:>3}"
                  f"{row['total_lines']:>5}{row['total_chars']:>7}{row['share'] * 100:>6.1f}%"
                  f"{row['last_seen_chapter']:>5}{row['idle_chapters']:>5}{row['match']:>10}")
        if result["unresolved"]:
            print(f"未匹配身份：{'、'.join(result['unresolved'])}")
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    target = out_dir / "appearance.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n统计写入 registry：identities[*].appearance_by_book")
    print(f"报告：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
