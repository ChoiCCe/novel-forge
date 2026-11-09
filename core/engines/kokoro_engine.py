"""Kokoro 可选引擎（中文 lang_code='z'，24kHz）。

本机未安装 kokoro 包，默认 disabled；装上后把 config.tts.engines.kokoro.enabled 置 true 即可，
路由与缓存逻辑无需改动。
"""
from __future__ import annotations

from pathlib import Path

from .base import BaseEngine, write_float_wav


class KokoroEngine(BaseEngine):
    name = "kokoro"

    def __init__(self, cfg: dict, sample_rate: int, project_root: Path):
        super().__init__(cfg, sample_rate, project_root)
        self._pipe = None

    def is_available(self, voice: dict) -> tuple[bool, str]:
        try:
            import kokoro  # noqa: F401
        except Exception as err:
            return False, f"未安装 kokoro：{err}"
        return True, ""

    def _pipeline(self):
        if self._pipe is None:
            from kokoro import KPipeline

            self._pipe = KPipeline(
                lang_code=self.cfg.get("lang_code", "z"),
                repo_id=self.cfg.get("repo_id", "hexgrad/Kokoro-82M"),
            )
        return self._pipe

    def synthesize(self, text: str, voice: dict, emotion: str, out_path: Path) -> Path:
        import numpy as np

        pipeline = self._pipeline()
        voice_name = voice.get("kokoro_voice") or self.cfg.get("default_voice", "zf_xiaobei")
        speed = float(voice.get("speed", self.cfg.get("speed", 1.0)))
        pieces = [np.asarray(audio, dtype="float32")
                  for _, _, audio in pipeline(text, voice=voice_name, speed=speed)
                  if audio is not None]
        if not pieces:
            raise RuntimeError("kokoro 未返回音频")
        return write_float_wav(np.concatenate(pieces), 24000, out_path)
