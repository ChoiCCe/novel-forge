"""edge-tts 引擎（联网，V1 正式方案：旁白 + 对白）。

情绪只用 rate/pitch/volume 做韵律近似，edge 不支持 SSML express-as。
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from .base import BaseEngine


class EdgeEngine(BaseEngine):
    name = "edge"

    def is_available(self, voice: dict) -> tuple[bool, str]:
        try:
            import edge_tts  # noqa: F401
        except Exception as err:
            return False, f"未安装 edge-tts：{err}"
        return True, ""

    def synthesize(self, text: str, voice: dict, emotion: str, out_path: Path) -> Path:
        import edge_tts

        from ..postprocess import convert_to_wav

        voice_id = voice.get("edge_voice")
        if not voice_id:
            raise ValueError("edge 合成任务缺少 edge_voice（应由选角/旁白配置提供）")
        rate = voice.get("rate", "+0%")
        pitch = voice.get("pitch", "+0Hz")
        volume = voice.get("volume", "+0%")
        with tempfile.TemporaryDirectory(prefix="novel-forge-edge-") as tmp:
            mp3 = Path(tmp) / "segment.mp3"
            communicate = edge_tts.Communicate(text, voice_id, rate=rate, pitch=pitch, volume=volume)

            def _send():
                asyncio.run(communicate.save(str(mp3)))

            global_cfg = getattr(self, "global_cfg", None)
            if global_cfg:
                from .. import tts_scheduler
                tts_scheduler.run(_send, global_cfg, self.root)
            else:
                _send()
            if not mp3.exists() or mp3.stat().st_size == 0:
                raise RuntimeError("edge-tts 返回空音频")
            convert_to_wav(mp3, out_path, self.sample_rate, self.cfg.get("_ffmpeg"))
        return out_path
