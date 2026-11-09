"""长章峰值内存对比：旧全量 concat vs 流式 concat demuxer。

用法（默认依次跑四个子进程并输出对比表）：
    python tools/mem_probe.py
    python tools/mem_probe.py --seconds 3600
    python tools/mem_probe.py --mode stream-long --dir <临时目录>
"""
from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import postprocess  # noqa: E402


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_rss_mb() -> float:
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    fn = ctypes.windll.kernel32.K32GetProcessMemoryInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), ctypes.c_ulong]
    fn.restype = ctypes.c_long
    process = ctypes.windll.kernel32.GetCurrentProcess()
    if not fn(process, ctypes.byref(counters), ctypes.sizeof(counters)):
        raise RuntimeError("K32GetProcessMemoryInfo 调用失败")
    return counters.PeakWorkingSetSize / 1024 / 1024


def _silence(path: Path, seconds: int, sample_rate: int) -> None:
    """分秒写入，避免生成阶段自身制造大块内存。"""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        chunk = b"\x00\x00" * sample_rate
        for _ in range(int(seconds)):
            handle.writeframes(chunk)


def make_fixtures(folder: Path, seconds: int, segment_seconds: int, sample_rate: int) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    short = folder / f"seg_{segment_seconds}s.wav"
    long = folder / f"seg_{seconds}s.wav"
    if not short.exists():
        _silence(short, segment_seconds, sample_rate)
    if not long.exists():
        _silence(long, seconds, sample_rate)
    return {"short": short, "long": long}


def build_jobs(seconds: int, segment_seconds: int, source: Path) -> list:
    count = max(1, seconds // segment_seconds)
    return [SimpleNamespace(
        index=i + 1, chapter_index=1, type="narration", text=f"第 {i + 1} 段用于内存压测",
        speaker="旁白", emotion="平静", pause_after_ms=0, audio_path=str(source),
        audio_tag="", direction={}, engine="edge", voice={},
    ) for i in range(count)]


def run_mode(mode: str, seconds: int, segment_seconds: int, work: Path) -> dict:
    import yaml

    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    sr = int(cfg["tts"]["sample_rate"])
    fixtures = make_fixtures(work / "fixtures", seconds, segment_seconds, sr)
    if mode == "old-long":
        jobs = build_jobs(seconds, seconds, fixtures["long"])
        out = work / "out_old_long.wav"
        started = time.perf_counter()
        postprocess.concat_jobs(jobs, out, cfg)
    elif mode == "stream-long":
        jobs = build_jobs(seconds, seconds, fixtures["long"])
        out = work / "out_stream_long.wav"
        started = time.perf_counter()
        postprocess.concat_jobs_streaming(jobs, out, cfg)
    elif mode == "old-many":
        jobs = build_jobs(seconds, segment_seconds, fixtures["short"])
        out = work / "out_old_many.wav"
        started = time.perf_counter()
        postprocess.concat_jobs(jobs, out, cfg)
    else:
        jobs = build_jobs(seconds, segment_seconds, fixtures["short"])
        out = work / "out_stream_many.wav"
        started = time.perf_counter()
        postprocess.concat_jobs_streaming(jobs, out, cfg)
    elapsed = time.perf_counter() - started
    payload = {
        "mode": mode, "segments": len(jobs), "seconds": seconds,
        "peak_rss_mb": round(peak_rss_mb(), 1), "elapsed_s": round(elapsed, 2),
        "out_mb": round(out.stat().st_size / 1024 / 1024, 1),
    }
    out.unlink(missing_ok=True)
    return payload


def compare(seconds: int, segment_seconds: int) -> int:
    work = Path(tempfile.mkdtemp(prefix="novel-forge-mem-"))
    results = []
    try:
        for mode in ("old-many", "stream-many", "old-long", "stream-long"):
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--mode", mode,
                 "--seconds", str(seconds), "--segment-seconds", str(segment_seconds),
                 "--dir", str(work)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr)
                return 1
            results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"长章内存压测：目标 {seconds}s，分段 {segment_seconds}s")
    print(f"{'模式':<14}{'段数':>5}{'时长(s)':>9}{'峰值RSS(MB)':>13}{'耗时(s)':>9}{'产物(MB)':>10}")
    for item in results:
        print(f"{item['mode']:<14}{item['segments']:>5}{item['seconds']:>9}"
              f"{item['peak_rss_mb']:>13}{item['elapsed_s']:>9}{item['out_mb']:>10}")
    base = {item["mode"]: item for item in results}
    delta = base["old-long"]["peak_rss_mb"] - base["stream-long"]["peak_rss_mb"]
    print(f"\n单段超长场景：流式比旧全量少 {delta:.1f} MB；"
          f"多分段场景：{base['old-many']['peak_rss_mb']:.1f} vs {base['stream-many']['peak_rss_mb']:.1f} MB")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="长章 concat 峰值内存对比")
    parser.add_argument("--mode", choices=["old-many", "stream-many", "old-long", "stream-long"])
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--segment-seconds", type=int, default=10)
    parser.add_argument("--dir", default="")
    args = parser.parse_args()
    if not args.mode:
        return compare(args.seconds, args.segment_seconds)
    work = Path(args.dir) if args.dir else Path(tempfile.mkdtemp(prefix="novel-forge-mem-"))
    result = run_mode(args.mode, args.seconds, args.segment_seconds, work)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
