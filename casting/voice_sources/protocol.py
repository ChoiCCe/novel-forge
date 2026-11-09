# -*- coding: utf-8 -*-
"""声线供给契约（provider protocol）。

接入一条新声线只需要声明三件事，**不需要**实现任何分配逻辑：

1. ``list_voices()`` 返回 ``VoiceSpec``：名称、语言、地区、性别、年龄段；
2. ``probe(voice, text, out_wav)`` 用给定固定文本合成一段单声道 wav（16bit），
   失败直接抛异常；
3. 注册 provider（``registry.register`` 或 CLI ``--provider module:attr``）。

入池判重、音色距离、跨地区拒绝、池子合并、重跑分配、回退判断全部由
``voice_sources`` 统一完成，provider 只负责“能不能出声”。

id 约定
-------
``VoiceSpec.id`` 用 ``-`` 分段，最后一段是短名（如 ``zh-CN-XiaoyiNeural`` →
``Xiaoyi``）。分配引擎把短名当作“基声线”键，因此同一池内短名不能重复；
新声线的 id 建议写成 ``{source}-{短名}Neural``，例如 ``mock-LingxiNeural``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

AGE_BANDS = ("少年", "青年", "中年", "老年")
GENDERS = ("男", "女")

# 与 casting/voices.py 的探测文本保持一致：只有同一句文本、同一采样率下的
# F0 / 包络才可直接比较。
DEFAULT_PROBE_TEXT = (
    "这一句用来检查声线是否可以正常合成，语气平稳，不带情绪，也不刻意加快或放慢。"
)


def short_name(voice_id: str) -> str:
    """基声线短名：与 casting/engine.build_bases 的取法一致。"""
    return voice_id.split("-")[-1].replace("Neural", "")


@dataclass(frozen=True)
class VoiceSpec:
    """一条声线的静态声明。年龄只是声明值，实际分配年龄由 F0 排序推导。"""

    id: str
    name: str
    lang: str                 # "zh" / "en" / "ja"
    locale: str               # "zh-CN"；必须与目标池的 base_region 完全一致
    gender: str               # "男" / "女"
    age: str                  # 少年 / 青年 / 中年 / 老年
    source: str = "custom"    # "edge" / "license:xx" / "mock" ...
    tags: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if not self.id or "-" not in self.id:
            raise ValueError(f"声线 id 需要形如 source-NameNeural: {self.id!r}")
        if self.gender not in GENDERS:
            raise ValueError(f"{self.id}: gender 必须是 {GENDERS}，收到 {self.gender!r}")
        if self.age not in AGE_BANDS:
            raise ValueError(f"{self.id}: age 必须是 {AGE_BANDS}，收到 {self.age!r}")
        if not self.locale.lower().startswith(self.lang.lower()):
            raise ValueError(f"{self.id}: locale {self.locale!r} 与 lang {self.lang!r} 不符")

    @property
    def region(self) -> str:
        return "-".join(self.locale.split("-")[:2])


@runtime_checkable
class VoiceProvider(Protocol):
    """最小 provider 契约。"""

    name: str

    def list_voices(self) -> Sequence[VoiceSpec]:
        ...

    def probe(self, voice: VoiceSpec, text: str, out_wav: Path) -> Path:
        """用 text 合成 voice 的音频到 out_wav；失败抛异常，不要返回空文件。"""
        ...
