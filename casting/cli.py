# -*- coding: utf-8 -*-
"""人名挖掘用的 LLM 客户端辅助。

旧的“一体化窗口分配 + 章边界换色”CLI（run 子命令）已删除。
声线分配以 outputs/novel-forge/v17_cast_official/voice_map.json 为准。

保留 ``_name_client``：outputs 下的历史人名挖掘脚本仍会
``from casting.cli import _name_client``，它与人名投票有关，不属于旧分配方案。
"""
from __future__ import annotations


def _name_client(args):
    """专用 LLM 客户端：只在本进程内调高并发/降低间隔，不改 config.yaml。"""
    import threading

    import yaml

    from core.llm import LLMClient
    from .llm_resolve.client import CONFIG_PATH, PROJECT_ROOT

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    client = LLMClient(cfg, PROJECT_ROOT)
    n = max(2, int(args.name_concurrency))
    client.min_interval_s = 0.25
    client.scheduler_concurrency = n
    client.global_concurrency = n
    client._global_sem = threading.Semaphore(n)
    client.per_model_concurrency = 1
    client._model_sems = {}
    client.max_retries = 0
    client.model_attempts = 2
    client.timeout = 90.0
    client.max_total_s = 200.0
    return client


def main(argv=None) -> int:
    raise SystemExit(
        "旧窗口分配 CLI 已删除。声线分配结果见 "
        "outputs/novel-forge/v17_cast_official/voice_map.json"
    )
