# -*- coding: utf-8 -*-
"""LLM 客户端适配与成本统计。

复用项目 ``core.llm.LLMClient`` 的模型链、冷却与降级逻辑，不复制路由代码。
只额外记录：调用次数、prompt/completion 字符数、估算 token、耗时、实际尝试的模型。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CACHE_VERSION = "v14-llm-resolve-v1"
CACHE_DIR = PROJECT_ROOT.parent / "outputs" / "novel-forge" / "v14_llm_resolve_cache"
_CACHE_ENABLED = os.environ.get("NOVEL_FORGE_LLM_CACHE", "1") != "0"

_client = None
_client_lock = threading.Lock()


def get_client():
    """懒加载项目 LLMClient（读取 config.yaml，不改任何默认值）。"""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            import yaml  # 项目已依赖

            from core.llm import LLMClient

            cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
            _client = LLMClient(cfg, PROJECT_ROOT)
    return _client


def set_client(client) -> None:
    """测试或调用方注入自定义客户端。"""
    global _client
    _client = client


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def estimate_tokens(text: str) -> int:
    """粗估 token：CJK 约 1 字 1 token，其他约 4 字符 1 token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return max(1, cjk + other // 4)


@dataclass
class Usage:
    calls: int = 0
    ok: int = 0
    failed: int = 0
    cache_hits: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    est_prompt_tokens: int = 0
    est_completion_tokens: int = 0
    elapsed_s: float = 0.0
    models: Counter = field(default_factory=Counter)
    errors: list[dict] = field(default_factory=list)

    def record_ok(self, prompt_text: str, completion_text: str,
                  elapsed_s: float, models: list[str] | None = None) -> None:
        self.calls += 1
        self.ok += 1
        self.prompt_chars += len(prompt_text)
        self.completion_chars += len(completion_text)
        self.est_prompt_tokens += estimate_tokens(prompt_text)
        self.est_completion_tokens += estimate_tokens(completion_text)
        self.elapsed_s += float(elapsed_s)
        for model in models or []:
            self.models[str(model)] += 1

    def record_fail(self, prompt_text: str, elapsed_s: float, error: str,
                    models: list[str] | None = None) -> None:
        self.calls += 1
        self.failed += 1
        self.prompt_chars += len(prompt_text)
        self.est_prompt_tokens += estimate_tokens(prompt_text)
        self.elapsed_s += float(elapsed_s)
        for model in models or []:
            self.models[str(model)] += 1
        if len(self.errors) < 50:
            self.errors.append({"error": str(error)[:400]})

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def merge(self, other: "Usage") -> None:
        self.calls += other.calls
        self.ok += other.ok
        self.failed += other.failed
        self.cache_hits += other.cache_hits
        self.prompt_chars += other.prompt_chars
        self.completion_chars += other.completion_chars
        self.est_prompt_tokens += other.est_prompt_tokens
        self.est_completion_tokens += other.est_completion_tokens
        self.elapsed_s += other.elapsed_s
        self.models.update(other.models)
        for err in other.errors:
            if len(self.errors) < 50:
                self.errors.append(err)

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "ok": self.ok,
            "failed": self.failed,
            "cache_hits": self.cache_hits,
            "prompt_chars": self.prompt_chars,
            "completion_chars": self.completion_chars,
            "est_prompt_tokens": self.est_prompt_tokens,
            "est_completion_tokens": self.est_completion_tokens,
            "est_total_tokens": self.est_prompt_tokens + self.est_completion_tokens,
            "elapsed_s": round(self.elapsed_s, 3),
            "models": dict(self.models),
            "errors": list(self.errors[:10]),
            "token_note": "token 为字符数估算（CJK 1 字≈1 token，其他 4 字符≈1 token）",
        }


def _model_call_counts(client) -> dict[str, int]:
    stats = getattr(client, "stats", None) or {}
    models = stats.get("models") or {}
    out = {}
    for name, info in models.items():
        try:
            out[str(name)] = int(info.get("calls", 0))
        except Exception:
            continue
    return out


def _cache_key(messages: list[dict], schema: dict, schema_name: str,
               model_tag: str = "") -> str:
    payload = {
        "version": CACHE_VERSION,
        "model": model_tag,
        "schema_name": schema_name,
        "schema": schema,
        "messages": messages,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str):
    if not _CACHE_ENABLED:
        return None
    path = CACHE_DIR / key[:2] / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("data")
    except Exception:
        return None


def _cache_put(key: str, data) -> None:
    if not _CACHE_ENABLED:
        return
    try:
        folder = CACHE_DIR / key[:2]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{key}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"data": data}, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def call_json(client, messages: list[dict], schema: dict, schema_name: str,
              usage: Usage) -> tuple[Any | None, str | None]:
    """调用一次强制 JSON；成功返回 (data, None)，失败返回 (None, reason)。

    这里吞掉所有异常，保证上层可以降级而不是中断流水线。
    """
    prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
    model_tag = ",".join(getattr(client, "models", None) or [])
    key = _cache_key(messages, schema, schema_name, model_tag)
    cached = _cache_get(key)
    if cached is not None:
        usage.record_cache_hit()
        return cached, None
    before = _model_call_counts(client)
    t0 = time.monotonic()
    try:
        data = client.chat_json(
            messages,
            temperature=0.0,
            schema=schema,
            schema_name=schema_name,
        )
    except Exception as exc:  # noqa: BLE001 - 降级边界
        elapsed = time.monotonic() - t0
        after = _model_call_counts(client)
        tried = [m for m, c in after.items() if c > before.get(m, 0)]
        reason = f"{type(exc).__name__}: {exc}"
        usage.record_fail(prompt_text, elapsed, reason, tried)
        return None, reason

    elapsed = time.monotonic() - t0
    after = _model_call_counts(client)
    tried = [m for m, c in after.items() if c > before.get(m, 0)]
    completion_text = json.dumps(data, ensure_ascii=False) if data is not None else ""
    usage.record_ok(prompt_text, completion_text, elapsed, tried)
    if data is not None:
        _cache_put(key, data)
    return data, None
