# -*- coding: utf-8 -*-
"""离线 provider：回放已落盘的探测 wav，不重新合成。

用于把历史探测音频、授权音色样音直接接进入池校验；也用于无网络环境复验。
``probe`` 只接受与录制时相同的文本，避免拿不同句子比较 F0/包络。
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..protocol import VoiceSpec


class CachedWavProvider:
    name = "cached-wav"

    def __init__(self, text: str, entries: dict[str, tuple[VoiceSpec, Path]],
                 name: str = "cached-wav") -> None:
        self.name = name
        self.probe_text = text
        self._entries = entries

    def list_voices(self) -> list[VoiceSpec]:
        return [spec for spec, _ in self._entries.values()]

    def probe(self, voice: VoiceSpec, text: str, out_wav: Path) -> Path:
        if text != self.probe_text:
            raise ValueError("缓存音频的文本与请求文本不一致，不能用于声学比较")
        src = Path(self._entries[voice.id][1])
        if not src.exists():
            raise FileNotFoundError(src)
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out_wav)
        return out_wav
