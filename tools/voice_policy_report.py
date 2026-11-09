"""按真实章节剧本评估 narrator_voice_policy，输出角色差异表。

用法：
    python tools/voice_policy_report.py ch000 ch001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import tts_router, voice_policy  # noqa: E402
from core.caster import Caster  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def find_script(chapter: str) -> Path:
    pattern = f"novels/*/scripts/{chapter}.json"
    found = sorted(ROOT.glob(pattern))
    return found[0]


def unknown_speakers(record: dict, speakers: dict, cfg: dict, narrator: dict) -> list[dict]:
    names: list[str] = []
    for seg in record.get("segments", []):
        if seg.get("type") == "dialogue" and seg.get("speaker") not in speakers:
            if seg.get("speaker") not in names:
                names.append(seg.get("speaker"))
    variants = voice_policy.unknown_variants(cfg, narrator, len(names))
    rows = []
    for slot, name in enumerate(names):
        profile = tts_router.unknown_speaker_profile(
            cfg, narrator, name, slot=slot, variants=variants)
        rows.append({"speaker": name, "voice": profile.get("edge_voice"),
                     "rate": profile.get("rate"), "pitch": profile.get("pitch"),
                     "volume": profile.get("volume"), "source": "unknown_speaker",
                     "segments": sum(1 for seg in record.get("segments", [])
                                     if seg.get("type") == "dialogue" and seg.get("speaker") == name)})
    return rows


def print_report(chapter: str, report: dict, unknown: list[dict]) -> None:
    t = report["thresholds"]
    print(f"\n=== {chapter} | policy={report['policy']} | "
          f"阈值 rate≥{t['rate']:g}pp pitch≥{t['pitch']:g}Hz volume≥{t['volume']:g}% "
          f"distance≥{t['distance']:g} ===")
    print(f"旁白 {report['narrator']['voice']} {report['narrator']['rate']}/"
          f"{report['narrator']['pitch']}/{report['narrator']['volume']}")
    header = (f"{'角色':<10}{'级别':<4}{'来源':<14}{'声线':<26}"
              f"{'rate':>7}{'pitch':>7}{'vol':>6}{'旁白距':>7}{'达标':>5}{'微调':>5}")
    print(header)
    labels = {"independent": "独立", "voice_shared": "复用非旁白",
              "voice_borrowed": "借旁白", "unknown_fallback": "未知兜底"}
    for row in sorted(report["rows"], key=lambda item: (int(item.get("level") or 9), item["name"])):
        print(f"{row['name']:<10}{row.get('level', ''):<4}{labels.get(row['label'], row['label']):<14}"
              f"{row['voice']:<26}{row['rate']:>7}{row['pitch']:>7}{row['volume']:>6}"
              f"{row['distance_to_narrator']:>7.2f}{('是' if row['ok'] else '否'):>5}"
              f"{('是' if row['param_fix'] else '否'):>5}")
    if unknown:
        print("未知说话人（第 4 级：旁白声线 + 独立参数档）：")
        for item in unknown:
            print(f"  {item['speaker']:<10}{'4':<4}{'未知兜底':<14}{item['voice']:<26}{item['rate']:>7}"
                  f"{item['pitch']:>7}{item['volume']:>6}{item['segments']:>4} 段")
    if report.get("reuse_groups"):
        print("同声线复用组（第 2 级，需参数拉开）：")
        for group in report["reuse_groups"]:
            members = "、".join(f"{m['name']}({m['params']['rate']}/{m['params']['pitch']})"
                                for m in group["members"])
            print(f"  {group['voice']:<26} {group['count']} 人 最小距离 {group['min_pair_distance']:.2f} "
                  f"{'达标' if group['ok'] else '未达标'}  {members}")
    if report.get("narrator_group"):
        members = "、".join(f"{m['name']}({m['params']['rate']}/{m['params']['pitch']})"
                            for m in report["narrator_group"])
        print(f"旁白声线组（第 3/4 级）：{members}")
    nearest = report.get("nearest_to_narrator")
    if nearest:
        print(f"与旁白最近距离：{nearest['name']} {nearest['distance']:.2f}")
    if report.get("dialect_rows"):
        print("方言声线（证据授权）：")
        for item in report["dialect_rows"]:
            print(f"  {item['name']:<10}{item['dialect']}/{item['voice']} "
                  f"置信度 {float(item.get('confidence') or 0):.2f} 依据：{item['evidence'][:40]}")
    else:
        print("方言声线未启用（本批次无角色证据）")
    print(f"独立 {report['independent_roles']} | 复用非旁白 {report.get('shared_roles', 0)} | "
          f"借旁白 {report['borrowed_roles']} | 未知兜底 {report['unknown_fallback_roles']} | "
          f"差异达标 {report['compliant']} | 自动微调 {report['param_fix']} | "
          f"未收敛 {report['unresolved']} | 人工冲突 {report['manual_conflict']} | "
          f"组合冲突 {report.get('combo_conflicts', 0)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="同声线参数区分报告")
    parser.add_argument("chapters", nargs="+", help="章节文件前缀，如 ch000 ch001")
    parser.add_argument("--json", default="", help="报告输出路径（默认 outputs 下）")
    args = parser.parse_args()
    caster = Caster(CFG, None, ROOT)
    out_dir = (ROOT / CFG["paths"]["outputs"] / "v5_narrator_policy").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {"policy": voice_policy.policy_of(CFG), "chapters": {}}
    for chapter in args.chapters:
        path = find_script(chapter)
        record = json.loads(path.read_text(encoding="utf-8"))
        cast_data = record.get("cast") or {}
        caster.ensure_registered(cast_data)
        cast_data = caster.refresh_cast(cast_data, record.get("chapter_index"),
                                        record.get("book_id"))
        report = cast_data.get("voice_policy") or {}
        unknown = unknown_speakers(record, cast_data.get("speakers") or {}, CFG,
                                   cast_data.get("narrator") or {})
        print_report(chapter, report, unknown)
        bundle["chapters"][chapter] = {"script": str(path), "report": report, "unknown": unknown}
    target = Path(args.json) if args.json else out_dir / "voice_policy_report.json"
    target.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
