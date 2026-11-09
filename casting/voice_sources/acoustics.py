# -*- coding: utf-8 -*-
"""声学测量：F0 + 80–6000Hz 音色包络。

两套包络口径，阈值不通用：

- ``primary``：与 ``casting/voices.py`` 完全同一套算法（线性 FFT bin 上取 log10
  平均谱，不做平滑、不去均值）。判重阈值 ``<0.02`` + 半音 ``<0.4``，近音
  复合距离阈值 ``<0.18`` —— 沿用 pipeline 文档里的既有口径，保证和现有
  ``voice_pool.json`` 里的数值可直接对照。
- ``centered``：v12 实测口径（80–6000Hz 几何网格、9 点平滑、减去均值），
  对整体响度/斜率不敏感，更适合看“是不是同一个人”。同一人跨档波动实测
  0.04–0.086，取约 2 倍余量，近音阈值 ``<0.18``。

``primary`` 不触发近音但 ``centered`` 触发时，同样判“近音（仅备用）”。
背景里的 Yunyang/Yunjian ``0.129`` 就是 ``centered`` 口径的实测值。
"""
from __future__ import annotations

import functools
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BAND = (80.0, 6000.0)
CENTERED_GRID = np.geomspace(80.0, 6000.0, 300)


@dataclass(frozen=True)
class VoiceFeatures:
    f0: float
    f0_p25: float
    f0_p75: float
    env_primary: np.ndarray | None
    env_centered: np.ndarray | None
    sample_rate: int
    duration_s: float

    def as_dict(self) -> dict:
        return {
            "f0": round(self.f0, 2),
            "f0_p25": round(self.f0_p25, 2),
            "f0_p75": round(self.f0_p75, 2),
            "sample_rate": self.sample_rate,
            "duration_s": round(self.duration_s, 2),
        }


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as fh:
        if fh.getnchannels() != 1 or fh.getsampwidth() != 2:
            raise ValueError(f"{path} 不是 16bit 单声道 wav")
        sr = fh.getframerate()
        data = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16)
    return data.astype(np.float64) / 32768.0, sr


def _frames(x: np.ndarray, sr: int, win: float = 0.04, hop: float = 0.01) -> np.ndarray:
    n, h = int(win * sr), int(hop * sr)
    if len(x) < n:
        return np.zeros((0, n))
    return np.lib.stride_tricks.sliding_window_view(x, n)[::h]


def extract(path: str | Path) -> VoiceFeatures:
    """测量一条声线的 F0 与两套音色包络（同一文件只测一次）。"""
    p = Path(path).resolve()
    st = p.stat()
    return _extract_cached(str(p), st.st_mtime_ns, st.st_size)


@functools.lru_cache(maxsize=64)
def _extract_cached(path_str: str, mtime_ns: int, size: int) -> VoiceFeatures:
    return _measure(Path(path_str))


def _measure(path: Path) -> VoiceFeatures:
    x, sr = read_wav(path)
    duration = len(x) / sr
    frames = _frames(x, sr)
    if len(frames) == 0:
        raise ValueError(f"{path} 太短，无法测量")

    lo, hi = int(sr / 400), int(sr / 60)
    window = np.hanning(frames.shape[1])
    freqs = np.fft.rfftfreq(frames.shape[1], 1 / sr)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    f0s, primary, centered = [], [], []
    for fr in frames:
        if math.sqrt(float(np.mean(fr ** 2))) < 0.02:
            continue
        c = fr - fr.mean()
        corr = np.correlate(c, c, mode="full")[len(c) - 1:]
        corr /= max(corr[0], 1e-9)
        seg = corr[lo:hi]
        if seg.size:
            peak = int(np.argmax(seg)) + lo
            if corr[peak] >= 0.3:
                f0s.append(sr / peak)
        mag = np.abs(np.fft.rfft(fr * window)) + 1e-9
        primary.append(np.log10(mag[band]))
        interp = np.interp(CENTERED_GRID, freqs, np.log10(mag))
        centered.append(np.convolve(interp, np.ones(9) / 9, mode="same"))

    if not f0s or not primary:
        raise ValueError(f"{path} 没有检测到稳定浊音帧")
    f0 = np.array(f0s)
    env_primary = np.mean(np.array(primary), axis=0)
    env_centered = np.mean(np.array(centered), axis=0)
    env_centered = env_centered - env_centered.mean()
    return VoiceFeatures(
        f0=float(np.median(f0)),
        f0_p25=float(np.percentile(f0, 25)),
        f0_p75=float(np.percentile(f0, 75)),
        env_primary=env_primary,
        env_centered=env_centered,
        sample_rate=sr,
        duration_s=duration,
    )


def semitone(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 99.0
    return abs(12 * math.log2(a / b))


def env_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or len(a) != len(b):
        return 1.0
    return float(np.sqrt(np.mean((a - b) ** 2)))


def timbre_distance(a: VoiceFeatures, b: VoiceFeatures) -> float:
    """主口径复合距离：包络 + 0.25 × min(半音差, 4) / 4。"""
    return env_distance(a.env_primary, b.env_primary) + 0.25 * min(semitone(a.f0, b.f0), 4) / 4


def centered_distance(a: VoiceFeatures, b: VoiceFeatures) -> float:
    return env_distance(a.env_centered, b.env_centered)
