# -*- coding: utf-8 -*-
"""声线准入与池子维护。

旧的 intake / plans / options / demo / __main__（窗口分配重跑、方案 diff、
回收窗口选项）已删除。这里只保留声线探测、准入、入池这些与分配方案无关的
基础设施；默认准入路径不做音频级判重/近音判定（见 admission.admit 的
``acoustic_gate``）。
"""

import os

# edge/onnx 相关依赖在部分环境会申请大块内存失败；必须在 import numpy 之前设置。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from .admission import Thresholds, admit, make_pool, merge_into_pool, probe_specs
from .protocol import DEFAULT_PROBE_TEXT, VoiceProvider, VoiceSpec, short_name
from .registry import ProviderSpec, available, load_provider, register, registry

__all__ = [
    "DEFAULT_PROBE_TEXT", "ProviderSpec", "Thresholds", "VoiceProvider", "VoiceSpec",
    "short_name", "admit", "available", "load_provider", "make_pool",
    "merge_into_pool", "probe_specs", "register", "registry",
]
