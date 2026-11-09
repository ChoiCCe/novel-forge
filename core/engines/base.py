"""TTS 引擎统一抽象：synthesize(text, voice_profile, emotion) -> wav 文件。

新增引擎 = 本包加一个文件 + config 注册，不动路由与管线。
"""
from __future__ import annotations

import array
import wave
from pathlib import Path


class EngineNotReady(RuntimeError):
    """依赖/模型/参考音频缺失，调用方应走降级链。"""


class BaseEngine:
    name = "base"

    def __init__(self, cfg: dict, sample_rate: int, project_root: Path):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.root = project_root

    def is_available(self, voice: dict) -> tuple[bool, str]:
        return True, ""

    def synthesize(self, text: str, voice: dict, emotion: str, out_path: Path) -> Path:
        raise NotImplementedError


def write_float_wav(samples, sample_rate: int, out_path: Path) -> Path:
    """把 [-1, 1] 浮点采样写成 16-bit PCM wav。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
        payload = (np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    except Exception:
        buf = array.array("h")
        for value in samples:
            buf.append(int(max(-1.0, min(1.0, float(value))) * 32767))
        payload = buf.tobytes()
    with wave.open(str(out_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(payload)
    return out_path
