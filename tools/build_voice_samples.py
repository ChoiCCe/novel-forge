"""为所有 available=true 的 edge 声线生成固定样本，供门户横向试听。

用法：
    python tools/build_voice_samples.py            # 跳过已存在
    python tools/build_voice_samples.py --force    # 全部重生成

输出：voices/samples/<voice>.wav + voices/samples/manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import postprocess, prosody, voices_catalog  # noqa: E402
from core.engines.edge_engine import EdgeEngine  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
SAMPLE_TEXT = "夜色从山脊后慢慢落下来，城里的灯火一盏一盏亮起，街巷深处传来更鼓声，远处有人在低声说话。"
OUT_DIR = ROOT / "voices" / "samples"
CATALOG = voices_catalog.load_catalog("voices/catalog.edge.json", ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="edge 声线样本库")
    parser.add_argument("--force", action="store_true", help="已存在的样本也重新生成")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edge_cfg = dict(CFG["tts"]["engines"]["edge"], _ffmpeg=CFG["postprocess"].get("ffmpeg_path"))
    engine = EdgeEngine(edge_cfg, int(CFG["tts"]["sample_rate"]), ROOT)
    engine.global_cfg = CFG
    entries, failures = [], []
    started = time.time()
    for item in CATALOG.get("voices", []):
        voice = item.get("id") or item.get("voice")
        if not item.get("available") or not voice:
            continue
        target = OUT_DIR / f"{voice}.wav"
        if target.exists() and not args.force:
            entries.append(describe(item, target))
            continue
        try:
            engine.synthesize(SAMPLE_TEXT, {"edge_voice": voice}, "平静", target)
            postprocess.trim_edge_silence(target, int(CFG["tts"]["sample_rate"]))
            entries.append(describe(item, target))
            print(f"[OK] {voice} {entries[-1]['duration_ms']/1000:.2f}s")
        except Exception as err:  # 单条失败不阻塞整批
            failures.append({"voice": voice, "error": f"{type(err).__name__}: {err}"})
            target.unlink(missing_ok=True)
            print(f"[FAIL] {voice} {err}")
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "catalog_version": str(CFG["tts"]["engines"]["edge"].get("voice_map_version", "")),
        "sample_text": SAMPLE_TEXT,
        "chars": prosody.spoken_len(SAMPLE_TEXT),
        "count": len(entries),
        "failed": len(failures),
        "voices": entries,
        "failures": failures,
        "elapsed_s": round(time.time() - started, 2),
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n样本 {len(entries)} 条，失败 {len(failures)} 条，"
          f"耗时 {manifest['elapsed_s']}s -> {OUT_DIR / 'manifest.json'}")
    return 0 if not failures else 1


def describe(item: dict, target: Path) -> dict:
    info = postprocess.wav_info(target)
    return {
        "voice": item.get("id") or item.get("voice"),
        "gender": item.get("gender"),
        "stages": item.get("stages") or [],
        "sample": target.name,
        "duration_ms": info["duration_ms"],
        "chars": prosody.spoken_len(SAMPLE_TEXT),
        "chars_per_sec": round(prosody.spoken_len(SAMPLE_TEXT) / (info["duration_ms"] / 1000), 2)
        if info["duration_ms"] else 0,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
