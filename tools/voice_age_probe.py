"""edge-tts 年龄曲线可行性实测（只读探针）。

目的：回答“同一基声线能否靠 rate/pitch/volume 表达少年→青年→中年→老年”。
- 逐条合成同一段文本，参数网格覆盖 pitch / rate 的实际可用区间；
- 转 16k 单声道 wav，用自相关测 F0、谱质心、语速（字/秒）；
- 输出 mp3 试听样本与 JSON 报告；**不写 registry、不改绑定、不合成正式音频**。

用法：
    python tools/voice_age_probe.py --voice zh-CN-YunxiNeural --out outputs/novel-forge/v7_voice_age
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
import time
import wave
from array import array
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEXT = "窗外雨声淅沥，半梦半醒间，徐宝心恍惚回到了故国。"
SAMPLE_RATE = 16000
MIN_INTERVAL_S = 1.0

PITCH_GRID = [-40, -20, -12, -8, -4, 0, 4, 8, 12, 20, 40]
RATE_GRID = [-50, -30, -20, -12, -6, 0, 6, 12, 20, 30, 50]
STAGE_GRID = [
    ("少年", "zh-CN-YunxiNeural", 8.0, 8.0, 0.0),
    ("青年", "zh-CN-YunxiNeural", 0.0, 0.0, 0.0),
    ("中年", "zh-CN-YunxiNeural", -6.0, -5.0, 0.0),
    ("老年", "zh-CN-YunxiNeural", -16.0, -11.0, 0.0),
]
CONTROL = ("老年_换声线对照", "zh-CN-YunjianNeural", 0.0, 0.0, 0.0)


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def synth(voice: str, rate: float, pitch: float, volume: float, out_mp3: Path) -> None:
    import edge_tts
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    if out_mp3.exists() and out_mp3.stat().st_size > 0:
        return
    communicate = edge_tts.Communicate(
        TEXT, voice, rate=f"{rate:+.0f}%", pitch=f"{pitch:+.0f}Hz", volume=f"{volume:+.0f}%")
    asyncio.run(communicate.save(str(out_mp3)))
    if not out_mp3.exists() or out_mp3.stat().st_size == 0:
        raise RuntimeError("edge-tts 返回空音频")


def to_wav(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(src), "-ar", str(SAMPLE_RATE), "-ac", "1",
                    "-c:a", "pcm_s16le", str(dst)], check=True)


def _frames(samples: np.ndarray, size: int, hop: int):
    for start in range(0, len(samples) - size + 1, hop):
        yield samples[start:start + size]


def analyze(wav_path: Path) -> dict:
    with wave.open(str(wav_path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
        sr = handle.getframerate()
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    frame, hop = int(sr * 0.040), int(sr * 0.010)
    min_lag, max_lag = int(sr / 400), int(sr / 60)
    f0_values, centroids, energies = [], [], []
    for chunk in _frames(samples, frame, hop):
        energy = float(np.sqrt(np.mean(chunk ** 2)))
        if energy < 0.01:
            continue
        energies.append(energy)
        windowed = chunk - chunk.mean()
        corr = np.correlate(windowed, windowed, mode="full")[frame - 1:]
        peak = corr[0] or 1e-9
        segment = corr[min_lag:max_lag]
        if len(segment) == 0:
            continue
        lag = int(np.argmax(segment)) + min_lag
        strength = float(corr[lag] / peak)
        if strength >= 0.45:
            f0_values.append(sr / lag)
        spectrum = np.abs(np.fft.rfft(windowed * np.hanning(len(windowed))))
        freqs = np.fft.rfftfreq(len(windowed), 1 / sr)
        total = float(spectrum.sum()) or 1e-9
        centroids.append(float((spectrum * freqs).sum() / total))
    voiced_ms = int(len(energies) * hop * 1000 / sr)
    f0 = float(np.median(f0_values)) if f0_values else 0.0
    return {
        "duration_ms": int(len(samples) * 1000 / sr),
        "voiced_ms": voiced_ms,
        "f0_median_hz": round(f0, 1),
        "f0_p10_hz": round(float(np.percentile(f0_values, 10)), 1) if f0_values else 0.0,
        "f0_p90_hz": round(float(np.percentile(f0_values, 90)), 1) if f0_values else 0.0,
        "f0_frames": len(f0_values),
        "centroid_hz": round(float(np.median(centroids)), 1) if centroids else 0.0,
        "chars_per_second": round(len(TEXT) / (voiced_ms / 1000), 2) if voiced_ms else 0.0,
        "rms": round(float(np.median(energies)), 4) if energies else 0.0,
    }


def run_case(name: str, voice: str, rate: float, pitch: float, volume: float,
             out_dir: Path) -> dict:
    stem = f"{name}_{voice.split('-')[-1].replace('Neural', '')}_{rate:+.0f}%_{pitch:+.0f}Hz"
    mp3 = out_dir / f"{stem}.mp3"
    wav = out_dir / f"{stem}.wav"
    synth(voice, rate, pitch, volume, mp3)
    to_wav(mp3, wav)
    metrics = analyze(wav)
    time.sleep(MIN_INTERVAL_S)
    return {"name": name, "voice": voice, "rate_pct": rate, "pitch_hz": pitch,
            "volume_pct": volume, "mp3": str(mp3), **metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="zh-CN-YunxiNeural")
    parser.add_argument("--out", default="outputs/novel-forge/v7_voice_age")
    args = parser.parse_args()
    out_dir = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"text": TEXT, "sample_rate": SAMPLE_RATE, "voice": args.voice,
               "pitch_grid": [], "rate_grid": [], "stages": []}

    for pitch in PITCH_GRID:
        results["pitch_grid"].append(
            run_case(f"pitch{pitch:+.0f}", args.voice, 0, pitch, 0, out_dir))
    for rate in RATE_GRID:
        results["rate_grid"].append(
            run_case(f"rate{rate:+.0f}", args.voice, rate, 0, 0, out_dir))
    for name, voice, rate, pitch, volume in STAGE_GRID:
        results["stages"].append(run_case(name, voice, rate, pitch, volume, out_dir))
    name, voice, rate, pitch, volume = CONTROL
    results["stages"].append(run_case(name, voice, rate, pitch, volume, out_dir))

    report = out_dir / "age_probe.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告：{report}")
    for row in results["pitch_grid"] + results["rate_grid"] + results["stages"]:
        print(f"{row['name']:<14} {row['voice']:<22} rate={row['rate_pct']:+.0f}% "
              f"pitch={row['pitch_hz']:+.0f}Hz  F0={row['f0_median_hz']:.1f}Hz "
              f"语速={row['chars_per_second']:.2f}字/秒 时长={row['duration_ms']}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
