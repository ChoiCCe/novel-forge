"""构建中文 edge 声线目录：逐条试合成 + 口音分类 + 池容量表。

用法：
    python tools/voice_catalog_build.py            # 复用已有成功结果，只补未测过的
    python tools/voice_catalog_build.py --force    # 全部重新试合成
    python tools/voice_catalog_build.py --dry-run  # 只打印将探测的声线

规则：
- 以 list-voices 的中文 locale（zh-CN / zh-TW / zh-HK / zh-CN-liaoning / zh-CN-shaanxi）为准；
- XiaoxuanNeural 不在 list-voices 里但可合成，单独补入；
- 试合成失败写入 unavailable，不静默删除；
- zh-TW / zh-HK / 地方口音一律 accented=true，默认不进主池。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import dialect  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
CATALOG = ROOT / "voices" / "catalog.edge.json"
PROBE_TEXT = "这一句用来检查声线是否可以正常合成。"
STAGES = ["少年", "青年", "中年", "老年"]

# 已知声线的听感年龄/可演绎阶段；stages 是“能演的范围”，不是感知年龄本身。
KNOWN = {
    "zh-CN-XiaoxiaoNeural": {"age": 30, "stages": ["青年", "中年"],
                             "temperaments": ["中性", "沉稳", "阴柔"]},
    "zh-CN-XiaoyiNeural": {"age": 20, "stages": ["少年", "青年", "中年"],
                           "temperaments": ["急躁", "阴柔", "中性"]},
    "zh-CN-XiaoxuanNeural": {"age": 55, "stages": ["中年", "老年"],
                             "temperaments": ["沉稳", "威严", "阴沉"]},
    "zh-CN-YunxiaNeural": {"age": 14, "stages": ["少年", "青年"],
                           "temperaments": ["中性", "急躁", "阴沉"]},
    "zh-CN-YunxiNeural": {"age": 24, "stages": ["少年", "青年", "中年"],
                          "temperaments": ["中性", "急躁", "阴柔"]},
    "zh-CN-YunyangNeural": {"age": 45, "stages": ["青年", "中年", "老年"],
                            "temperaments": ["沉稳", "威严", "阴沉"]},
    "zh-CN-YunjianNeural": {"age": 48, "stages": ["青年", "中年", "老年"],
                            "temperaments": ["豪爽", "威严", "急躁"]},
    "zh-CN-liaoning-XiaobeiNeural": {"age": 30, "stages": ["青年", "中年"],
                                     "temperaments": ["豪爽", "中性"]},
    "zh-CN-shaanxi-XiaoniNeural": {"age": 30, "stages": ["青年", "中年"],
                                   "temperaments": ["豪爽", "中性"]},
    "zh-HK-HiuGaaiNeural": {"age": 28, "stages": ["青年", "中年"]},
    "zh-HK-HiuMaanNeural": {"age": 32, "stages": ["青年", "中年"]},
    "zh-HK-WanLungNeural": {"age": 35, "stages": ["青年", "中年"]},
    "zh-TW-HsiaoChenNeural": {"age": 28, "stages": ["青年", "中年"]},
    "zh-TW-HsiaoYuNeural": {"age": 24, "stages": ["少年", "青年"]},
    "zh-TW-YunJheNeural": {"age": 30, "stages": ["青年", "中年"]},
}
EXTRA_VOICES = ["zh-CN-XiaoxuanNeural"]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


async def list_zh() -> list[dict]:
    import edge_tts
    voices = await edge_tts.list_voices()
    rows = [v for v in voices if str(v.get("Locale") or "").lower().startswith("zh")]
    rows.sort(key=lambda item: item.get("ShortName", ""))
    return rows


async def probe_one(voice: str, attempts: int = 3) -> dict:
    import edge_tts
    started = time.time()
    error_type, error = "", ""
    for attempt in range(attempts):
        try:
            with tempfile.TemporaryDirectory(prefix="vf-catalog-") as tmp:
                mp3 = Path(tmp) / "probe.mp3"
                await edge_tts.Communicate(PROBE_TEXT, voice).save(str(mp3))
                if mp3.exists() and mp3.stat().st_size > 0:
                    return {"available": True, "error_type": "", "error": "",
                            "probe_ms": int((time.time() - started) * 1000)}
                error_type = "empty_audio"
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            error_type = ("no_audio" if "NoAudioReceived" in name or "NoAudioReceived" in str(exc)
                          else name)
            error = str(exc)[:200]
        if attempt < attempts - 1:
            await asyncio.sleep(2)
    return {"available": False, "error_type": error_type, "error": error,
            "probe_ms": int((time.time() - started) * 1000)}


def build_entry(voice: str, gender: str, locale: str) -> dict:
    meta = KNOWN.get(voice) or {}
    tag = dialect.tag_for_locale(locale)
    return {
        "id": voice,
        "gender": gender,
        "locale": locale,
        "perceived_age": int(meta.get("age") or 30),
        "stages": list(meta.get("stages") or ["青年", "中年"]),
        "temperaments": list(meta.get("temperaments") or []),
        "accented": bool(tag),
        "dialect": tag,
        "available": False,
        "status": "unverified",
        "error_type": "",
        "error": "",
        "probe_ms": 0,
        "checked_at": "",
    }


def capacity(voices: list[dict], narrator_voice: str) -> dict:
    def ids(rows):
        return [row["id"] for row in rows]

    main_all = [row for row in voices
                if row.get("available") and not row.get("accented")]
    main = [row for row in main_all if row["id"] != narrator_voice]
    accented = [row for row in voices if row.get("available") and row.get("accented")]
    table = {"main": {}, "with_narrator": {}, "accented": {}}
    for stage in STAGES:
        table["main"][stage] = {}
        table["with_narrator"][stage] = {}
        for gender in ("男", "女"):
            rows = [row for row in main if row.get("gender") == gender
                    and stage in (row.get("stages") or [])]
            table["main"][stage][gender] = ids(rows)
            table["with_narrator"][stage][gender] = ids(
                [row for row in main_all if row.get("gender") == gender
                 and stage in (row.get("stages") or [])])
    for tag in sorted({row.get("dialect") for row in accented if row.get("dialect")}):
        table["accented"][tag] = {}
        for gender in ("男", "女"):
            rows = [row for row in accented if row.get("dialect") == tag
                    and row.get("gender") == gender]
            table["accented"][tag][gender] = ids(rows)
    table["main"]["_totals"] = {"男": len({v for stage in STAGES
                                            for v in table["main"][stage]["男"]}),
                                "女": len({v for stage in STAGES
                                          for v in table["main"][stage]["女"]})}
    return table


def print_capacity(table: dict) -> None:
    print("\n=== 主池（非方言，不含旁白）容量 ===")
    print(f"{'阶段':<6}{'男':<6}{'女':<6}")
    for stage in STAGES:
        print(f"{stage:<6}{len(table['main'][stage]['男']):<6}{len(table['main'][stage]['女']):<6}")
    print(f"主池独立声线总数：男 {table['main']['_totals']['男']} / "
          f"女 {table['main']['_totals']['女']}")
    print("\n=== 方言池（仅证据授权可用） ===")
    if not table["accented"]:
        print("无")
    for tag, rows in table["accented"].items():
        print(f"{tag}: 男 {len(rows.get('男') or [])} / 女 {len(rows.get('女') or [])} "
              f"{rows.get('男', []) + rows.get('女', [])}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="构建中文 edge 声线目录")
    parser.add_argument("--force", action="store_true", help="全部重新试合成")
    parser.add_argument("--dry-run", action="store_true", help="只打印将探测的声线")
    parser.add_argument("--out", default=str(CATALOG))
    args = parser.parse_args()

    rows = await list_zh()
    listed = {row["ShortName"]: row for row in rows}
    for voice in EXTRA_VOICES:
        listed.setdefault(voice, {"ShortName": voice, "Gender": "Female", "Locale": "zh-CN"})
    old = load_json(Path(args.out))
    old_entries = {item.get("id"): item for item in old.get("voices", []) if item.get("id")}

    voices: list[dict] = []
    for voice, row in sorted(listed.items()):
        gender = "女" if str(row.get("Gender", "")).lower().startswith("female") else "男"
        stage_row = build_entry(voice, gender, str(row.get("Locale") or ""))
        previous = old_entries.get(voice) or {}
        if not args.force and previous.get("available"):
            stage_row.update({
                "available": True, "status": previous.get("status", "verified"),
                "probe_ms": previous.get("probe_ms", 0),
                "checked_at": previous.get("checked_at", ""),
                "error_type": previous.get("error_type", ""),
                "error": previous.get("error", ""),
            })
        elif previous:
            stage_row["checked_at"] = previous.get("checked_at", "")
        voices.append(stage_row)

    todo = [row for row in voices if args.force or not row.get("available")]
    print(f"中文声线 {len(voices)} 条；本次探测 {len(todo)} 条：")
    for row in todo:
        print(f"  {row['id']} ({row['gender']}, {row['locale']})")
    if args.dry_run:
        return 0

    for row in todo:
        result = await probe_one(row["id"])
        row.update(result)
        row["status"] = "verified" if result["available"] else "unavailable"
        row["checked_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        mark = "OK  " if result["available"] else "FAIL"
        print(f"{mark} {row['id']} {result['probe_ms']}ms {result.get('error_type', '')}")

    available = [row for row in voices if row.get("available")]
    unavailable = [row for row in voices if not row.get("available")]
    table = capacity(voices, str(CFG["tts"]["engines"]["edge"]["narrator_voice"]))
    catalog = {
        "version": f"edge-zh-{datetime.now().strftime('%Y%m%d')}",
        "verified_at": datetime.now().strftime("%Y-%m-%d"),
        "probe_method": "edge-tts 逐条试合成（tools/voice_catalog_build.py，每条最多 3 次）",
        "probe_text": PROBE_TEXT,
        "locales": {row["locale"]: sum(1 for item in voices if item["locale"] == row["locale"])
                    for row in voices},
        "voices": voices,
        "unavailable": [{"id": row["id"], "locale": row["locale"],
                         "error_type": row.get("error_type", ""),
                         "error": row.get("error", "")} for row in unavailable],
        "pool_capacity": table,
        "fallbacks": old.get("fallbacks") or {},
    }
    target = Path(args.out)
    target.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n可用 {len(available)} / 不可用 {len(unavailable)}")
    if unavailable:
        print("不可用：")
        for row in unavailable:
            print(f"  {row['id']}  {row.get('error_type')} {row.get('error')}")
    print_capacity(table)
    print(f"\n目录：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
