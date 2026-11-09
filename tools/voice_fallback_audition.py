"""v4 第 0 步：失效声线的平替试听生成。

XiaoshuangNeural（女少年）与 YunyeNeural（男老年）实测 NoAudioReceived，
本工具用可用声线 + 韵律偏移做近似，并与“原生可用声线基准”并排生成，
供人工盲听选择。

输出：outputs/voice_fallback_audition/*.mp3 + fallback_manifest.json
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT.parent / "outputs" / "voice_fallback_audition"

# 试听文本：前半句是角色口吻，后半句是叙述，能听出音色年龄感
AUDITION_TEXT = "二叔，您都这个年纪了，还跟他们较什么劲呢？我扶您回屋歇着吧。"

CANDIDATES = [
    # 女少年（原生 Xiaoshuang 不可用）
    {"group": "女少年", "label": "A_xiaoyi_p0", "voice": "zh-CN-XiaoyiNeural",
     "rate": "+0%", "pitch": "+0Hz", "volume": "+0%", "note": "原生 Xiaoyi，未抬音高（对照）"},
    {"group": "女少年", "label": "B_xiaoyi_p6", "voice": "zh-CN-XiaoyiNeural",
     "rate": "+0%", "pitch": "+6Hz", "volume": "+0%", "note": "Xiaoyi 抬 6Hz，轻度少年感"},
    {"group": "女少年", "label": "C_xiaoyi_p10", "voice": "zh-CN-XiaoyiNeural",
     "rate": "+2%", "pitch": "+10Hz", "volume": "+0%", "note": "Xiaoyi 抬 10Hz + 2%，更接近少女"},
    {"group": "女少年", "label": "D_xiaoxiao_p8", "voice": "zh-CN-XiaoxiaoNeural",
     "rate": "+2%", "pitch": "+8Hz", "volume": "+0%", "note": "Xiaoxiao 抬 8Hz（备选，易与旁白撞声）"},
    # 男老年（原生 Yunye 不可用）
    {"group": "男老年", "label": "A_yunjian_base", "voice": "zh-CN-YunjianNeural",
     "rate": "-8%", "pitch": "-4Hz", "volume": "+0%", "note": "Yunjian 默认偏威严（对照）"},
    {"group": "男老年", "label": "B_yunjian_slow", "voice": "zh-CN-YunjianNeural",
     "rate": "-12%", "pitch": "-6Hz", "volume": "-2%", "note": "Yunjian 降速降调，老年近似"},
    {"group": "男老年", "label": "C_yunyang_slow", "voice": "zh-CN-YunyangNeural",
     "rate": "-12%", "pitch": "-6Hz", "volume": "-2%", "note": "Yunyang 降速降调，沉稳老者"},
    {"group": "男老年", "label": "D_yunyang_base", "voice": "zh-CN-YunyangNeural",
     "rate": "-8%", "pitch": "-4Hz", "volume": "+0%", "note": "Yunyang 轻度放慢（对照）"},
    # 原生可用基准
    {"group": "基准", "label": "REF_xiaoshuang_attempt", "voice": "zh-CN-XiaoshuangNeural",
     "rate": "+0%", "pitch": "+0Hz", "volume": "+0%", "note": "原生女少年：预期失败，留证"},
    {"group": "基准", "label": "REF_yunye_attempt", "voice": "zh-CN-YunyeNeural",
     "rate": "+0%", "pitch": "+0Hz", "volume": "+0%", "note": "原生男老年：预期失败，留证"},
]


async def synth_one(item: dict) -> dict:
    out = OUT_DIR / f"{item['group']}_{item['label']}.mp3"
    rec = dict(item, file=str(out), ok=False, bytes=0, error="")
    try:
        comm = edge_tts.Communicate(AUDITION_TEXT, item["voice"],
                                    rate=item["rate"], pitch=item["pitch"], volume=item["volume"])
        await comm.save(str(out))
        rec["bytes"] = out.stat().st_size if out.exists() else 0
        rec["ok"] = rec["bytes"] > 0
    except Exception as err:  # noqa: BLE001
        rec["error"] = f"{type(err).__name__}: {err}"[:300]
        out.unlink(missing_ok=True)
    return rec


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for item in CANDIDATES:
        records.append(await synth_one(item))
        await asyncio.sleep(0.5)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "text": AUDITION_TEXT,
        "items": records,
    }
    (OUT_DIR / "fallback_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [r for r in records if r["ok"]]
    print(f"生成 {len(ok)}/{len(records)} 条试听 -> {OUT_DIR}")
    for r in records:
        print(("OK  " if r["ok"] else "FAIL") + f" {r['label']:<22} {r['voice']:<28} "
              f"{r['rate']}/{r['pitch']}/{r['volume']} {r['error'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
