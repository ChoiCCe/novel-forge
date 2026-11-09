# -*- coding: utf-8 -*-
"""provider 注册表。

用法::

    from casting.voice_sources import register, available

    class MyProvider: ...
    register(MyProvider())

    for provider, spec in available("zh"):
        print(provider.name, spec.id)

命令行也可以不改代码接入::

    python -m casting.voice_sources intake --run <上次产物> --out <目录> \\
        --provider my_package.my_module:provider
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from .protocol import VoiceProvider, VoiceSpec


@dataclass(frozen=True)
class ProviderSpec:
    provider: VoiceProvider
    spec: VoiceSpec


class VoiceRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, VoiceProvider] = {}

    def register(self, provider: VoiceProvider, *, replace: bool = False) -> VoiceProvider:
        name = getattr(provider, "name", "") or type(provider).__name__
        if not name:
            raise ValueError("provider 必须有 name")
        if not hasattr(provider, "list_voices") or not hasattr(provider, "probe"):
            raise TypeError(f"{name}: provider 必须实现 list_voices / probe")
        if name in self._providers and not replace:
            raise ValueError(f"provider 已注册：{name}（需要覆盖请传 replace=True）")
        self._providers[name] = provider
        return provider

    def get(self, name: str) -> VoiceProvider:
        return self._providers[name]

    def providers(self) -> list[VoiceProvider]:
        return list(self._providers.values())

    def available(self, lang: str | None = None) -> list[ProviderSpec]:
        out: list[ProviderSpec] = []
        for provider in self._providers.values():
            for spec in provider.list_voices():
                spec.validate()
                if lang is None or spec.lang.lower() == lang.lower():
                    out.append(ProviderSpec(provider, spec))
        return out

    def clear(self) -> None:
        self._providers.clear()


DEFAULT_REGISTRY = VoiceRegistry()


def register(provider: VoiceProvider, *, replace: bool = False) -> VoiceProvider:
    return DEFAULT_REGISTRY.register(provider, replace=replace)


def registry() -> VoiceRegistry:
    return DEFAULT_REGISTRY


def available(lang: str | None = None) -> list[ProviderSpec]:
    return DEFAULT_REGISTRY.available(lang)


def load_provider(target: str) -> VoiceProvider:
    """按 ``module:attr`` 导入 provider 实例（供命令行使用）。"""
    if ":" not in target:
        raise ValueError(f"--provider 需要 module:attr 形式，收到 {target!r}")
    module_name, attr = target.split(":", 1)
    module = importlib.import_module(module_name)
    provider = getattr(module, attr)
    if isinstance(provider, type):
        provider = provider()
    register(provider)
    return provider
