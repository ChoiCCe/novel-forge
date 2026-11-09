"""一次性参数试听 CLI：合成一条 wav 后立即退出，供门户按需调用。

用法：
    python tools/voice_preview.py --voice zh-CN-XiaoxiaoNeural \
        --rate -10% --pitch -3Hz --volume +0% --text "..." --out <path>
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import postprocess, prosody  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
DEFAULT_TEXT = "夜色从山脊后慢慢落下来，城里的灯火一盏一盏亮起，街巷深处传来更鼓声。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一次性 edge 参数试听")
    parser.add_argument("--voice", required=True, help="edge 声线 id")
    parser.add_argument("--rate", default="+0%", help="语速，单位 %%（如 -10%%）")
    parser.add_argument("--pitch", default="+0Hz", help="音高，单位 Hz（如 -3Hz）")
    parser.add_argument("--volume", default="+0%", help="音量，单位 %%（如 +0%%）")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="试听文本")
    parser.add_argument("--out", default="", help="输出 wav 路径；默认临时文件")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import edge_tts
    from core.postprocess import convert_to_wav

    started = time.time()
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="voice-preview-")) / "preview.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="voice-preview-mp3-") as tmp:
        mp3 = Path(tmp) / "preview.mp3"
        asyncio.run(edge_tts.Communicate(
            args.text, args.voice, rate=args.rate, pitch=args.pitch,
            volume=args.volume).save(str(mp3)))
        if not mp3.exists() or mp3.stat().st_size == 0:
            raise RuntimeError("edge-tts 返回空音频")
        convert_to_wav(mp3, out, int(CFG["tts"]["sample_rate"]),
                       CFG["postprocess"].get("ffmpeg_path"))
    postprocess.trim_edge_silence(out, int(CFG["tts"]["sample_rate"]))
    duration = postprocess.wav_info(out)["duration_ms"]
    chars = prosody.spoken_len(args.text)
    print(f"path={out}")
    print(f"duration_ms={duration} chars={chars} "
          f"chars_per_sec={chars / (duration / 1000) if duration else 0:.2f} "
          f"elapsed_s={time.time() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
