"""统一 TTS 调度器（v4 工程化约束 4）。

所有 edge-tts 请求必须经过这里：并发上限 + 相邻请求最小间隔。
参数全部来自 `config.yaml` 的 `tts.scheduler`，可被单书设置覆盖
（`voices/registry.json` -> `book_settings[book_id]`）。

设计要点：
- 并发默认 2、最大 2；`concurrency` 会被夹紧到 `[1, max_concurrency]`。
- 间隔按“相邻请求开始时间”计算，而不是请求结束时间；并发 >1 时仍保证请求起点有间距。
- 渲染线程用 `book_context(book_id)` 标明当前书，engine 内部自动读取该书的覆盖值。
"""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

log = logging.getLogger("novel-forge.tts.scheduler")

_ACTIVE_BOOK: ContextVar[str | None] = ContextVar("novel_forge_tts_book", default=None)
_BOOK_KEYS = ("tts_concurrency", "tts_min_interval_s")


def _cfg_block(cfg: dict) -> dict:
    return ((cfg.get("tts") or {}).get("scheduler") or {})


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _registry_path(cfg: dict, root: Path) -> Path:
    voices = (cfg.get("paths") or {}).get("voices", "voices")
    return (root / voices / "registry.json").resolve()


def book_overrides(cfg: dict, root: Path, book_id: str | None) -> dict:
    """读取单书覆盖；registry 缺失时用全局默认，损坏时告警并退回全局默认。"""
    if not book_id:
        return {}
    path = _registry_path(cfg, root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as err:
        log.warning("读取单书 TTS 覆盖失败，退回全局默认：%s（%s）", path, err)
        return {}
    return dict(((data.get("book_settings") or {}).get(book_id) or {}))


def resolve_params(cfg: dict, root: Path, book_id: str | None = None) -> dict:
    """合并全局默认 + 单书覆盖，返回最终调度参数。"""
    base = _cfg_block(cfg)
    max_concurrency = max(1, _as_int(base.get("max_concurrency"), 2))
    concurrency = _as_int(base.get("concurrency"), 2)
    min_interval = _as_float(base.get("min_interval_s"), 3.0)
    enabled = bool(base.get("enabled", True))
    book = book_overrides(cfg, root, book_id)
    if book.get("tts_concurrency") is not None:
        concurrency = _as_int(book.get("tts_concurrency"), concurrency)
    if book.get("tts_min_interval_s") is not None:
        min_interval = _as_float(book.get("tts_min_interval_s"), min_interval)
    concurrency = max(1, min(concurrency, max_concurrency))
    return {
        "enabled": enabled,
        "concurrency": concurrency,
        "max_concurrency": max_concurrency,
        "min_interval_s": max(0.0, min_interval),
        "book_id": book_id or "",
        "source": "book" if book else "global",
    }


@contextmanager
def book_context(book_id: str | None):
    token = _ACTIVE_BOOK.set(book_id or None)
    try:
        yield
    finally:
        _ACTIVE_BOOK.reset(token)


class TTSScheduler:
    """进程级调度器；改参数时重建内部信号量。"""

    def __init__(self, cfg: dict, root: Path):
        self.cfg = cfg
        self.root = Path(root)
        self._params = resolve_params(cfg, self.root, None)
        self._sem = threading.BoundedSemaphore(self._params["concurrency"])
        self._interval_lock = threading.Lock()
        self._last_start = 0.0
        self._stats_lock = threading.Lock()
        self._stats = {"calls": 0, "wait_s": 0.0, "failures": 0}

    def params(self, book_id: str | None = None) -> dict:
        return resolve_params(self.cfg, self.root, book_id or _ACTIVE_BOOK.get())

    def _ensure(self, book_id: str | None) -> dict:
        params = resolve_params(self.cfg, self.root, book_id or _ACTIVE_BOOK.get())
        if params != self._params:
            self._params = params
            self._sem = threading.BoundedSemaphore(params["concurrency"])
            log.info("TTS 调度参数更新：并发 %s，间隔 %.2fs（%s）",
                     params["concurrency"], params["min_interval_s"], params["source"])
        return params

    def run(self, fn):
        params = self._ensure(None)
        if not params["enabled"]:
            return fn()
        started = time.monotonic()
        with self._sem:
            with self._interval_lock:
                wait = params["min_interval_s"] - (time.monotonic() - self._last_start)
                if started > self._last_start and wait > 0:
                    time.sleep(wait)
                self._last_start = time.monotonic()
            waited = time.monotonic() - started
            with self._stats_lock:
                self._stats["calls"] += 1
                self._stats["wait_s"] += waited
            try:
                return fn()
            except Exception:
                with self._stats_lock:
                    self._stats["failures"] += 1
                raise

    def snapshot(self, book_id: str | None = None) -> dict:
        with self._stats_lock:
            stats = dict(self._stats)
        stats["params"] = self.params(book_id)
        return stats


_SCHEDULER: TTSScheduler | None = None
_INIT_LOCK = threading.Lock()


def get_scheduler(cfg: dict, root: Path) -> TTSScheduler:
    global _SCHEDULER
    if _SCHEDULER is None or Path(_SCHEDULER.root) != Path(root):
        with _INIT_LOCK:
            if _SCHEDULER is None or Path(_SCHEDULER.root) != Path(root):
                _SCHEDULER = TTSScheduler(cfg, root)
    _SCHEDULER.cfg = cfg
    return _SCHEDULER


def run(fn, cfg: dict, root: Path):
    """统一入口：engine 只调这一个函数，不直接碰信号量。"""
    return get_scheduler(cfg, root).run(fn)
