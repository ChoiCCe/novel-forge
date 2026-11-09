"""同一非旁白声线的参数区分度实测。

用法：
    python tools/param_diff_probe.py

对同一条台词做多组 rate/pitch/volume 变体，测量时长、响度、F0，
并生成 A(+8%/+2Hz) 与 B(-8%/-3Hz) 的对照音频，用于确认
voice_rules.differentiate 的阈值（rate/pitch/volume/整体距离）不是拍脑袋。
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import voice_policy  # noqa: E402
from core.engines.edge_engine import EdgeEngine  # noqa: E402
from core.export import write_chapter_mp3  # noqa: E402
from core.postprocess import concat_with_pauses, wav_info  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
SAMPLE_RATE = int(CFG["tts"]["sample_rate"])
VOICE = "zh-CN-YunxiNeural"
TEXT = "这一句用来检查同一条声线在不同参数下，能不能听出区别。"

VARIANTS = [
    ("base", "+0%", "+0Hz", "+0%"),
    ("rate+4", "+4%", "+0Hz", "+0%"),
    ("rate+8", "+8%", "+0Hz", "+0%"),
    ("rate-8", "-8%", "+0Hz", "+0%"),
    ("pitch+2", "+0%", "+2Hz", "+0%"),
    ("pitch+4", "+0%", "+4Hz", "+0%"),
    ("pitch-3", "+0%", "-3Hz", "+0%"),
    ("volume+6", "+0%", "+0Hz", "+6%"),
    ("volume+12", "+0%", "+0Hz", "+12%"),
    ("compareA", "+8%", "+2Hz", "+0%"),
    ("compareB", "-8%", "-3Hz", "+0%"),
]


def rms_db(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    if data.size == 0:
        return -120.0
    value = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)) / 32768.0)
    return 20.0 * np.log10(max(value, 1e-6))


def f0_stats(path: Path) -> dict:
    """自相关测 F0：60–400Hz，取有声帧中位数。"""
    with wave.open(str(path), "rb") as handle:
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    signal = data.astype(np.float64) / 32768.0
    frame = int(0.04 * SAMPLE_RATE)
    hop = int(0.02 * SAMPLE_RATE)
    low, high = int(SAMPLE_RATE / 400), int(SAMPLE_RATE / 60)
    values: list[float] = []
    for start in range(0, max(0, len(signal) - frame), hop):
        chunk = signal[start:start + frame]
        if np.sqrt(np.mean(chunk ** 2)) < 0.02:
            continue
        chunk = chunk - chunk.mean()
        corr = np.correlate(chunk, chunk, mode="full")[frame - 1:]
        corr /= max(corr[0], 1e-9)
        peak = int(np.argmax(corr[low:high])) + low
        if corr[peak] >= 0.3:
            values.append(SAMPLE_RATE / peak)
    if not values:
        return {"f0_median": 0.0, "f0_std": 0.0, "voiced_frames": 0}
    return {"f0_median": round(float(np.median(values)), 2),
            "f0_std": round(float(np.std(values)), 2), "voiced_frames": len(values)}


def build_engine() -> EdgeEngine:
    edge_cfg = dict(CFG["tts"]["engines"]["edge"], _ffmpeg=CFG["postprocess"].get("ffmpeg_path"))
    return EdgeEngine(edge_cfg, SAMPLE_RATE, ROOT)


def main() -> int:
    out_dir = (ROOT / CFG["paths"]["outputs"] / "v5_param_diff").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = build_engine()
    rows: list[dict] = []
    paths: dict[str, Path] = {}
    for tag, rate, pitch, volume in VARIANTS:
        target = out_dir / f"probe_{tag}.wav"
        if not target.exists():
            engine.synthesize(TEXT, {"edge_voice": VOICE, "rate": rate,
                                     "pitch": pitch, "volume": volume}, "平静", target)
        paths[tag] = target
        info = wav_info(target)
        stats = f0_stats(target)
        row = {"tag": tag, "voice": VOICE, "rate": rate, "pitch": pitch,
               "volume": volume, "duration_ms": info["duration_ms"],
               "rms_db": round(rms_db(target), 2), "path": str(target), **stats}
        rows.append(row)
        print(f"{tag:<10}{rate:>6}{pitch:>7}{volume:>6} 时长 {row['duration_ms']:>5}ms "
              f"响度 {row['rms_db']:>6.2f}dB F0 {row['f0_median']:>6.2f}Hz")

    base = next(row for row in rows if row["tag"] == "base")
    for row in rows:
        row["vs_base"] = {
            "duration_pct": round((row["duration_ms"] - base["duration_ms"])
                                  / max(base["duration_ms"], 1) * 100, 2),
            "rms_db": round(row["rms_db"] - base["rms_db"], 2),
            "f0_hz": round(row["f0_median"] - base["f0_median"], 2),
        }

    compare = {row["tag"]: row for row in rows}
    a, b = compare["compareA"], compare["compareB"]
    merged = out_dir / "compare_same_voice.wav"
    concat_with_pauses([paths["compareA"], paths["compareB"]], [0, 700],
                       merged, SAMPLE_RATE)
    mp3 = out_dir / "compare_same_voice.mp3"
    write_chapter_mp3(merged, mp3, CFG)

    cfg = CFG
    a_params = voice_policy.params(a)
    b_params = voice_policy.params(b)
    distance = voice_policy.distance(a_params, b_params, cfg)
    ok = voice_policy.pair_ok(a_params, b_params, cfg)
    payload = {
        "voice": VOICE, "text": TEXT, "sample_rate": SAMPLE_RATE,
        "thresholds": voice_policy.thresholds(cfg),
        "variants": rows,
        "compare": {"A": a, "B": b, "distance": round(distance, 3),
                    "pair_ok": ok, "wav": str(merged), "mp3": str(mp3)},
        "basis": {
            "rate": "4pp 约等于 4% 语速/时长差；同类嗓音 A/B 的语速可辨阈约 5%，"
                    "再叠加语调差即可稳定分辨",
            "pitch": "edge pitch 以 Hz 偏移直接改 F0；4Hz 对 120Hz 男声约 3.3% 基频差，"
                     "实测 A/B 的 f0_hz 差需 ≥4Hz",
            "volume": "6% 幅度约 0.5dB；实测 A/B 响度差需 ≥0.4dB 才不被环境噪声掩盖",
            "distance": "把 rate/pitch/volume 按阈值归一化后取欧氏距离；1.4 表示"
                        "至少一维明显超阈、另一维也有可见偏移",
        },
    }
    target = out_dir / "param_diff.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nA/B 距离 {distance:.2f}（阈值 {voice_policy.thresholds(cfg)['distance']:.2f}）"
          f" 达标={ok}")
    print(f"对照音频：{merged}\n对照 MP3：{mp3}\n数据：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
