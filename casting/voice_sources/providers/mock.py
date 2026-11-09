# -*- coding: utf-8 -*-
"""假 provider：本地合成，不连任何外部服务，用来验证接口链路。

产出确定性的类人声（谐波 + 三个共振峰 + 音节包络），16k 单声道 16bit。
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ..protocol import VoiceSpec

SR = 16000


class MockProvider:
    name = "mock"

    def __init__(self) -> None:
        self._voices = [
            VoiceSpec(
                id="mock-LingxiNeural", name="灵汐（虚拟）", lang="zh", locale="zh-CN",
                gender="女", age="青年", source="mock", tags=("demo", "virtual"),
            )
        ]
        self.probe_text = ""

    def list_voices(self) -> list[VoiceSpec]:
        return list(self._voices)

    def probe(self, voice: VoiceSpec, text: str, out_wav: Path) -> Path:
        if voice.id != "mock-LingxiNeural":
            raise KeyError(voice.id)
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        _synth(out_wav, text)
        return out_wav


def _synth(path: Path, text: str, f0: float = 332.0) -> None:
    duration = min(6.0, max(3.0, len(text) * 0.12))
    n = int(duration * SR)
    t = np.arange(n) / SR
    pitch = f0 * (1 - 0.05 * t / duration) * (1 + 0.004 * np.sin(2 * np.pi * 4.7 * t))
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    rng = np.random.default_rng(7)
    formants = ((760.0, 180.0, 1.0), (1450.0, 220.0, 0.5), (3000.0, 320.0, 0.15))
    y = np.zeros(n)
    for k in range(1, 56):
        fk = k * f0
        if fk > 7000:
            break
        amp = (1.0 / k ** 1.15) * (0.18 + sum(
            g * np.exp(-0.5 * ((fk - f) / bw) ** 2) for f, bw, g in formants))
        y += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    y *= 0.55 + 0.45 * np.sin(2 * np.pi * 3.2 * t) ** 2
    y *= np.minimum(1.0, np.minimum(t / 0.08, (duration - t) / 0.12))
    y *= 0.12 / max(float(np.sqrt(np.mean(y ** 2))), 1e-9)
    y = np.clip(y, -0.99, 0.99)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes((y * 32767).astype(np.int16).tobytes())


provider = MockProvider()
