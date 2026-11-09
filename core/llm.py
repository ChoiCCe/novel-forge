"""OpenAI 兼容 LLM 客户端：快速失败换模型、健康度排序、冷却与并发/限流控制。"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from openai import OpenAI

log = logging.getLogger("novel-forge.llm")
T = TypeVar("T")
R = TypeVar("R")


class LLMError(RuntimeError):
    pass


class LLMChainError(LLMError):
    """模型链路全部失败，携带每个模型的失败原因。"""

    def __init__(self, failures: list[dict]):
        self.failures = failures
        detail = "；".join(f"{item['model']}: {item['error']}" for item in failures)
        super().__init__(f"模型链路全部失败（{detail}）")


RETRYABLE_KINDS = ("timeout", "server", "connection")


def error_status(err: Exception) -> int | None:
    status = getattr(err, "status_code", None)
    if status is None:
        status = getattr(getattr(err, "response", None), "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def classify_error(err: Exception) -> str:
    """把异常归一成可决策的错误类型；路由只按类型决定“重试一次”还是“立刻换模型”。"""
    status = error_status(err)
    text = str(err).lower()
    if status == 429 or "rate limit" in text or "429" in text:
        return "rate_limit"
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status in (400, 422):
        return "bad_request"
    if status is not None and 500 <= status < 600:
        return "server"
    if isinstance(err, TimeoutError) or "timed out" in text or "timeout" in text:
        return "timeout"
    if "connection" in text or "network" in text or "proxy" in text:
        return "connection"
    return "other"


def load_env(path: str | Path) -> dict[str, str]:
    """轻量 .env 读取；不覆盖进程里已有的同名环境变量。"""
    env: dict[str, str] = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in env.items():
        os.environ.setdefault(key, value)
    return env


def _strip_wrappers(text: str) -> str:
    text = text.replace("\ufeff", "").strip()
    # DeepSeek-R1 / Qwen 系列常见推理段标记
    text = re.sub(r"<think>[\s\S]*?<｜end▁of▁thinking｜>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|begin_of_thought\|>[\s\S]*?<\|end_of_thought\|>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(思考过程|推理过程)[:：][\s\S]*?(?=\{)", "", text)
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _balanced_slices(text: str) -> list[str]:
    """扫描所有顶层 {} / [] 片段，供“取最后一个 JSON 块”使用。"""
    slices: list[str] = []
    i = 0
    while i < len(text):
        if text[i] not in "{[":
            i += 1
            continue
        opener = text[i]
        closer = "}" if opener == "{" else "]"
        depth, in_string, escaped = 0, False, False
        for j in range(i, len(text)):
            ch = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    slices.append(text[i:j + 1])
                    i = j + 1
                    break
        else:
            i += 1
    return slices


def validate_schema(data: Any, schema: dict) -> str:
    """极简 JSON Schema 校验（type/properties/required/items/enum/anyOf）。

    不引入新依赖；返回空串表示通过，否则返回错误描述。
    """
    if not schema:
        return ""
    if "anyOf" in schema:
        if not any(validate_schema(data, sub) == "" for sub in schema["anyOf"] or []):
            return "不匹配 anyOf 中的任一分支"
    expected = schema.get("type")
    if expected == "object" and not isinstance(data, dict):
        return f"顶层应为 object，实际 {type(data).__name__}"
    if expected == "array" and not isinstance(data, list):
        return f"顶层应为 array，实际 {type(data).__name__}"
    if expected == "string" and not isinstance(data, str):
        return "应为 string"
    if expected == "integer" and (not isinstance(data, int) or isinstance(data, bool)):
        return "应为 integer"
    if expected == "number" and not isinstance(data, (int, float)):
        return "应为 number"
    if expected == "boolean" and not isinstance(data, bool):
        return "应为 boolean"
    if "enum" in schema and data not in schema["enum"]:
        return f"取值 {data!r} 不在 enum {schema['enum']}"
    if isinstance(data, dict):
        for key in schema.get("required") or []:
            if key not in data:
                return f"缺少字段 {key}"
        for key, sub in (schema.get("properties") or {}).items():
            if key in data:
                err = validate_schema(data[key], sub)
                if err:
                    return f"{key}: {err}"
    if isinstance(data, list) and schema.get("items"):
        for i, item in enumerate(data):
            err = validate_schema(item, schema["items"])
            if err:
                return f"[{i}]: {err}"
    return ""


def extract_json(text: str) -> Any:
    """解析 LLM 输出：容忍 think 标签、代码围栏、尾逗号、多 JSON 块（优先最后一个）。"""
    cleaned = _strip_wrappers(text)
    candidates = _balanced_slices(cleaned)
    if not candidates:
        candidates = [cleaned]
    # R1 常在推理里给示例 JSON，真正答案在最后；因此倒序优先
    ordered = list(reversed(candidates)) + candidates
    errors: list[str] = []
    for candidate in ordered:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as err:
            errors.append(str(err))
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(repaired)
        except (ValueError, SyntaxError):
            pass
    raise LLMError(f"无法解析 JSON：{cleaned[:200]}（{errors[:2]}）")


class LLMClient:
    """所有模块共享一个客户端；OpenAI SDK 本身线程安全。"""

    def __init__(self, cfg: dict, project_root: Path):
        env_cfg = cfg.get("llm", {})
        env_file = (project_root / cfg["paths"]["env_file"]).resolve()
        load_env(env_file)

        def pick(key_cfg: str, default: str) -> str:
            env_key = env_cfg.get(key_cfg)
            return os.environ.get(env_key, default) if env_key else default

        api_key = os.environ.get(env_cfg.get("api_key_env", "ZHIPU_API_KEY"), "")
        if not api_key:
            raise LLMError(f"缺少 API Key，请在 {env_file} 设置 {env_cfg.get('api_key_env')}")
        base_url = pick("base_url_env", env_cfg["base_url"])
        chain = [str(m) for m in (env_cfg.get("model_chain") or []) if str(m).strip()]
        if not chain:
            raise LLMError("config llm.model_chain 为空")

        routing = env_cfg.get("routing") or {}
        cool = routing.get("cool_down") or {}
        backoff = routing.get("backoff") or {}
        probe = routing.get("probe") or {}
        scheduler = env_cfg.get("scheduler") or {}
        self.timeout = float(env_cfg.get("timeout_s", 45))
        self.max_retries = int(env_cfg.get("max_retries", 1))
        self.model_attempts = int(routing.get("model_attempts", env_cfg.get("model_attempts", 1)))
        self.max_total_s = float(routing.get("max_total_s", 105))
        self.allow_thinking = bool(routing.get("allow_thinking", False))
        self.cooldown_failures = int(cool.get("failures", 3))
        self.cooldown_s = float(cool.get("seconds", 600))
        self.rate_limit_cooldown_s = float(cool.get("rate_limit_seconds", 60))
        self.backoff_base = float(backoff.get("base_s", 1.5))
        self.backoff_max = float(backoff.get("max_s", 20))
        self.probe_on_start = bool(probe.get("on_start", True))
        self.probe_ttl = float(probe.get("ttl_s", 300))
        self.probe_timeout = float(probe.get("sample_timeout_s", 8))
        # LLM 调度器：全局并发硬上限 1 + 相邻请求最小间隔，跨供应商共用
        self.scheduler_concurrency = max(1, min(int(scheduler.get("concurrency", 1)), 1))
        self.scheduler_limit = int(scheduler.get("max_concurrency", 1))
        self.min_interval_s = max(0.0, float(scheduler.get("min_interval_s", 1.0)))
        self.per_model_concurrency = max(1, min(int(scheduler.get("per_model_concurrency", 1)), 1))
        self.global_concurrency = self.scheduler_concurrency
        self.concurrency = self.scheduler_concurrency
        self.disable_thinking = bool(env_cfg.get("disable_thinking", True))
        self.json_mode = bool(env_cfg.get("response_format_json", True))
        structured = env_cfg.get("structured_output") or {}
        self.structured_enabled = bool(structured.get("enabled", False))
        self.prefer_schema = bool(structured.get("prefer_schema", False))
        self.validate_output = bool(structured.get("validate", False))
        self._json_disabled_models: set[str] = set()
        self._schema_disabled_models: set[str] = set()
        self._stats_lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._pace_lock = threading.Lock()
        self._next_request_at = 0.0
        self._probed_at = 0.0
        self._global_sem = threading.Semaphore(self.global_concurrency)
        self._model_sems: dict[str, threading.Semaphore] = {}

        # 多供应商：LLM.specs 决定候选表；未配置 providers 时退化为单一硅基通道
        self.specs: list[dict] = []
        self._clients: dict[str, OpenAI] = {}
        provider_order = [str(p) for p in (env_cfg.get("provider_order") or [])]
        providers = env_cfg.get("providers") or {}
        if providers:
            names = provider_order + [n for n in providers if n not in provider_order]
            for name in names:
                spec = providers.get(name) or {}
                if not spec.get("enabled", True):
                    continue
                key_env = spec.get("api_key_env", "")
                key_value = os.environ.get(key_env, "") if key_env else ""
                if not key_value:
                    log.warning("供应商 %s 缺少 %s，跳过", name, key_env)
                    continue
                p_base = os.environ.get(spec.get("base_url_env", ""), "") or spec.get(
                    "base_url", base_url)
                models = [str(m) for m in (spec.get("models") or []) if str(m).strip()]
                if spec.get("use_model_chain"):
                    models = chain + [m for m in models if m not in chain]
                if not models:
                    log.warning("供应商 %s 未配置 models，跳过", name)
                    continue
                self._clients[name] = OpenAI(api_key=key_value, base_url=p_base,
                                            timeout=self.timeout, max_retries=0)
                spec_thinking = {str(m) for m in (spec.get("thinking_models") or [])}
                for model in models:
                    self.specs.append({"model": model, "provider": name,
                                       "thinking": model in spec_thinking})
        else:
            # 兼容旧配置：没有 providers 时退化为单一供应商（默认智谱）
            self._clients["zhipu"] = OpenAI(api_key=api_key, base_url=base_url,
                                            timeout=self.timeout, max_retries=0)
            for model in chain:
                self.specs.append({"model": model, "provider": "zhipu", "thinking": False})
        if not self.specs:
            raise LLMError("没有可用的 LLM 供应商/模型")
        self.models = [item["model"] for item in self.specs]
        self._model_provider = {item["model"]: item["provider"] for item in self.specs}
        self.thinking_models = {str(m) for m in (env_cfg.get("thinking_models") or [])}
        for spec in providers.values():
            self.thinking_models |= {str(m) for m in (spec.get("thinking_models") or [])}
        for item in self.specs:
            if item["thinking"]:
                self.thinking_models.add(item["model"])
        self.model_weight = {str(k): float(v) for k, v in (env_cfg.get("model_weight") or {}).items()}
        for spec in providers.values():
            for k, v in (spec.get("model_weight") or {}).items():
                self.model_weight.setdefault(str(k), float(v))
        self.stats = {
            "json_calls": 0, "json_success": 0, "chain_failures": 0,
            "model_switches": 0, "rate_limited": 0,
            "schema_calls": 0, "schema_pass": 0, "schema_fail": 0,
            "schema_switch": 0, "schema_unsupported": 0,
            "models": {model: {"calls": 0, "success": 0, "failures": 0, "last_error": "",
                               "provider": self._model_provider.get(model, "")}
                       for model in self.models},
        }
        # 健康度：滑动窗口（最近 20 次）+ 延迟 EWMA + 冷却截止时间
        self.health: dict[str, dict] = {
            model: {"calls": 0, "success": 0, "failures": 0, "consecutive_failures": 0,
                    "cooldown_until": 0.0, "latency_ewma": 0.0, "last_kind": "",
                    "last_error": "", "rate_limited": 0, "window": []}
            for model in self.models
        }
        self.client = self._clients[self._model_provider.get(self.models[0], "zhipu")]

    def _client(self, model: str) -> OpenAI:
        provider = self._model_provider.get(model)
        if provider is None:
            # 禁止静默回落到第一个供应商：模型名写错时会被另一方以“模型不存在”误判
            raise LLMError(f"模型 {model} 未在任何供应商配置中声明（检查 model_chain / providers.models）")
        return self._clients[provider]

    def _pace(self) -> None:
        """跨供应商的最小请求间隔：同一时刻只允许一个 LLM 请求进入。"""
        if self.min_interval_s <= 0:
            return
        with self._pace_lock:
            now = time.monotonic()
            wait = self._next_request_at - now
            if wait > 0:
                time.sleep(wait)
            self._next_request_at = time.monotonic() + self.min_interval_s

    def _bump(self, key: str, amount: int = 1) -> None:
        with self._stats_lock:
            self.stats[key] = self.stats.get(key, 0) + amount

    def _health(self, model: str) -> dict:
        return self.health.setdefault(
            model, {"calls": 0, "success": 0, "failures": 0, "consecutive_failures": 0,
                    "cooldown_until": 0.0, "latency_ewma": 0.0, "last_kind": "",
                    "last_error": "", "rate_limited": 0, "window": []})

    def _count_call(self, model: str) -> None:
        with self._stats_lock:
            entry = self.stats["models"].setdefault(
                model, {"calls": 0, "success": 0, "failures": 0, "last_error": ""})
            entry["calls"] += 1
            health = self._health(model)
            health["calls"] += 1

    def _record(self, model: str, ok: bool, latency: float, error: str = "",
                kind: str = "", learn_latency: bool = True) -> None:
        """记录一次模型级结果：驱动动态排序、冷却与连续失败计数。"""
        with self._stats_lock:
            entry = self.stats["models"].setdefault(
                model, {"calls": 0, "success": 0, "failures": 0, "last_error": ""})
            health = self._health(model)
            health["window"].append((bool(ok), max(0.01, latency)))
            if len(health["window"]) > 20:
                health["window"] = health["window"][-20:]
            if latency > 0 and learn_latency:
                health["latency_ewma"] = (health["latency_ewma"] * 0.7 + latency * 0.3
                                          if health["latency_ewma"] else latency)
            elif latency > 0:
                health["probe_latency"] = latency
            if ok:
                entry["success"] += 1
                health["success"] += 1
                health["consecutive_failures"] = 0
                health["cooldown_until"] = 0.0
                health["last_error"] = ""
                health["last_kind"] = ""
                return
            entry["failures"] += 1
            entry["last_error"] = error
            health["failures"] += 1
            health["last_error"] = error
            health["last_kind"] = kind
            health["consecutive_failures"] += 1
            if kind == "rate_limit":
                health["rate_limited"] += 1
                self.stats["rate_limited"] = self.stats.get("rate_limited", 0) + 1
                health["cooldown_until"] = max(health["cooldown_until"],
                                               time.monotonic() + self.rate_limit_cooldown_s)
            elif (health["consecutive_failures"] >= self.cooldown_failures
                  and kind in RETRYABLE_KINDS + ("auth", "not_found", "bad_request", "other")):
                now = time.monotonic()
                if health["cooldown_until"] <= now:
                    log.warning("模型 %s 连续失败 %s 次（%s），冷却 %.0fs",
                                model, health["consecutive_failures"], kind, self.cooldown_s)
                health["cooldown_until"] = max(health["cooldown_until"], now + self.cooldown_s)

    @contextmanager
    def _model_slot(self, model: str):
        """全局并发 + 单模型并发双重闸门，避免打爆硅基的账号/单模型配额。"""
        with self._stats_lock:
            sem = self._model_sems.get(model)
            if sem is None:
                sem = threading.Semaphore(max(1, self.per_model_concurrency))
                self._model_sems[model] = sem
        sem.acquire()
        self._global_sem.acquire()
        try:
            yield
        finally:
            self._global_sem.release()
            sem.release()

    def _score(self, model: str) -> float:
        """成功率高、延迟低、质量权重高的模型优先；无业务延迟时按静态顺序并列。"""
        health = self._health(model)
        recent = health["window"]
        rate = (sum(1 for ok, _ in recent if ok) / len(recent)) if recent else 0.6
        # 只信业务请求的延迟：探活是极小请求，TTFT 不能代表 900 字块的真实耗时
        latency = health["latency_ewma"] or 1.0
        weight = self.model_weight.get(model, 1.0)
        return (rate * weight) / (latency + 1.0)

    def _ordered_models(self, prefer: str | None = None) -> list[str]:
        now = time.monotonic()
        candidates = list(self.models)
        if not self.allow_thinking:
            candidates = [m for m in candidates if m not in self.thinking_models]
        if not candidates:
            raise LLMError("model_chain 中只有思考模型，但 routing.allow_thinking=false")
        ready: list[str] = []
        cooling: list[tuple[float, str]] = []
        for model in candidates:
            health = self._health(model)
            if health["cooldown_until"] > now:
                cooling.append((health["cooldown_until"], model))
            else:
                if health["cooldown_until"]:
                    log.info("模型 %s 冷却结束，恢复试探", model)
                    health["cooldown_until"] = 0.0
                    health["consecutive_failures"] = 0
                ready.append(model)
        ready.sort(key=lambda m: (-self._score(m), self.models.index(m)))
        if prefer and prefer in ready:
            ready.remove(prefer)
            ready.insert(0, prefer)
        if not ready:
            # 全部冷却：挑最早恢复的那个，宁可试一次也不让整章卡死
            ready = [min(cooling, key=lambda item: item[0])[1]]
            log.warning("全部模型冷却中，提前试探 %s", ready[0])
        return ready

    def _probe_one(self, model: str) -> None:
        if model in self.thinking_models and not self.allow_thinking:
            return
        start = time.monotonic()
        self._count_call(model)
        try:
            with self._model_slot(model):
                self._pace()
                self._client(model).chat.completions.create(
                    model=model, max_tokens=8, temperature=0.0,
                    messages=[{"role": "user", "content": "只回复 ok"}],
                    timeout=min(self.probe_timeout, self.timeout),
                    extra_body={"thinking": {"type": "disabled"}} if self.disable_thinking else None,
                )
            self._record(model, True, time.monotonic() - start, learn_latency=False)
        except Exception as err:
            self._record(model, False, time.monotonic() - start,
                         f"{type(err).__name__}: {err}", classify_error(err),
                         learn_latency=False)

    def _ensure_warm(self) -> None:
        """按 TTL 做一次轻量探活；探活只覆盖非思考模型，避免 45s 级等待。"""
        if not self.probe_on_start:
            return
        with self._probe_lock:
            now = time.monotonic()
            if now - self._probed_at < self.probe_ttl:
                return
            self._probed_at = now
        targets = [m for m in self.models
                   if m not in self.thinking_models or self.allow_thinking]
        if not targets:
            return
        workers = max(1, min(len(targets), self.global_concurrency))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(self._probe_one, targets))
        log.info("模型探活完成：%s", ", ".join(
            f"{m}={self._health(m)['latency_ewma']:.2f}s" for m in targets))

    def json_success_rate(self) -> float:
        return self.stats["json_success"] / self.stats["json_calls"] if self.stats["json_calls"] else 0.0

    def schema_pass_rate(self) -> float:
        return self.stats["schema_pass"] / self.stats["schema_calls"] if self.stats["schema_calls"] else 0.0

    def stats_snapshot(self) -> dict:
        with self._stats_lock:
            data = {key: value for key, value in self.stats.items() if key != "models"}
        data["schema_pass_rate"] = round(self.schema_pass_rate(), 4)
        data["json_success_rate"] = round(self.json_success_rate(), 4)
        return data

    def health_snapshot(self) -> list[dict]:
        """每模型：调用数、成功率、平均延迟、切换/429 次数、最后错误、剩余冷却。"""
        now = time.monotonic()
        rows: list[dict] = []
        with self._stats_lock:
            for model in self.models:
                health = self._health(model)
                calls = health["calls"]
                rows.append({
                    "model": model,
                    "calls": calls,
                    "success": health["success"],
                    "failures": health["failures"],
                    "success_rate": round(health["success"] / calls, 4) if calls else 0.0,
                    "avg_latency_s": round(health["latency_ewma"], 2),
                    "probe_latency_s": round(health.get("probe_latency", 0.0), 2),
                    "consecutive_failures": health["consecutive_failures"],
                    "rate_limited": health["rate_limited"],
                    "cooldown_remaining_s": round(max(0.0, health["cooldown_until"] - now), 1),
                    "last_kind": health["last_kind"],
                    "last_error": health["last_error"][:160],
                })
        return rows

    def _request_once(self, model: str, messages: list[dict], temperature: float,
                      max_tokens: int | None, schema: dict | None = None,
                      schema_name: str = "payload", timeout: float | None = None) -> str:
        kwargs: dict[str, Any] = {"model": model, "temperature": temperature, "messages": messages}
        self._last_reasoning = False
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if timeout is not None:
            kwargs["timeout"] = max(1.0, timeout)
        if self.disable_thinking and not (model in self.thinking_models and self.allow_thinking):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        use_json_mode = self.json_mode and model not in self._json_disabled_models
        schema_mode = bool(self.structured_enabled and self.prefer_schema and schema
                           and model not in self._schema_disabled_models)
        if schema_mode:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": False, "schema": schema},
            }
        elif use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        self._count_call(model)
        client = self._client(model)
        try:
            self._pace()
            response = client.chat.completions.create(**kwargs)
        except Exception as err:
            status = getattr(err, "status_code", None)
            text = str(err).lower()
            unsupported = status in (400, 404, 422) or "response_format" in text \
                or "json_schema" in text or "json_object" in text
            if schema_mode and unsupported:
                log.warning("%s 不支持 json_schema，退回 json_object：%s", model, err)
                self._schema_disabled_models.add(model)
                self._bump("schema_unsupported")
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                else:
                    kwargs.pop("response_format", None)
                self._pace()
                response = client.chat.completions.create(**kwargs)
            elif use_json_mode and unsupported:
                # 部分模型/网关不支持 response_format，仅对该模型退回纯提示词
                log.warning("%s 不支持 response_format，退回纯提示词：%s", model, err)
                self._json_disabled_models.add(model)
                kwargs.pop("response_format", None)
                self._pace()
                response = client.chat.completions.create(**kwargs)
            else:
                raise
        message = response.choices[0].message
        content = (message.content or "").strip()
        reasoning = getattr(message, "reasoning_content", None)
        if not reasoning and getattr(message, "model_extra", None):
            reasoning = (message.model_extra or {}).get("reasoning_content")
        self._last_reasoning = bool(reasoning) and not content
        if not content and reasoning:
            content = str(reasoning).strip()
        if not content:
            raise LLMError("content 与 reasoning_content 均为空")
        return content

    def _request_model(self, model: str, messages: list[dict], temperature: float,
                       schema: dict | None, schema_name: str, deadline: float) -> str:
        """单模型请求：最多额外重试 1 次，只对超时/5xx/连接错误重试；429 立即上抛换模型。"""
        attempts = max(1, self.max_retries + 1)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                with self._model_slot(model):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.5:
                        raise LLMError(f"单块总耗时预算 {self.max_total_s:.0f}s 已用尽")
                    timeout = min(self.timeout, remaining)
                    return self._request_once(model, messages, temperature, None,
                                              schema=schema, schema_name=schema_name,
                                              timeout=timeout)
            except Exception as err:
                last_error = err
                kind = classify_error(err)
                if kind == "rate_limit" or attempt >= attempts:
                    raise
                if kind in RETRYABLE_KINDS:
                    wait = min(self.backoff_base ** attempt, self.backoff_max)
                    log.warning("%s 第 %s/%s 次请求失败（%s，%.1fs），%.1fs 后重试",
                                model, attempt, attempts, kind, timeout, wait)
                    time.sleep(wait)
                    continue
                raise
        raise last_error if last_error else LLMError("未知请求失败")

    def _try_model(self, model: str, messages: list[dict], temperature: float,
                   schema: dict | None, schema_name: str, deadline: float, accept) -> Any:
        history = list(messages)
        last_error: Exception | None = None
        for attempt in range(1, max(1, self.model_attempts) + 1):
            try:
                content = self._request_model(model, history, temperature, schema,
                                              schema_name, deadline)
                data = extract_json(content)
                if schema and self.validate_output:
                    self._bump("schema_calls")
                    schema_error = validate_schema(data, schema)
                    if schema_error:
                        self._bump("schema_fail")
                        raise ValueError(f"schema 校验失败：{schema_error}")
                    self._bump("schema_pass")
                if accept is not None and not accept(data):
                    raise ValueError("JSON 内容不符合阶段要求（空结果）")
                return data
            except Exception as err:
                last_error = err
                log.warning("%s 第 %s/%s 次 JSON 尝试失败：%s", model, attempt,
                            self.model_attempts, err)
                if attempt < self.model_attempts:
                    hint = ("上次输出未通过 schema 校验。请严格按 JSON Schema 输出，"
                            "不要解释、推理或 markdown。") if "schema" in str(err) else \
                           "上次输出无法解析。请只输出一个 JSON 对象：以 { 开头、} 结尾，不要任何解释、推理或 markdown。"
                    history = history + [{"role": "user", "content": hint}]
        raise last_error if last_error else LLMError("未知请求失败")

    def chat_json(self, messages: list[dict], temperature: float = 0.0,
                  prefer: str | None = None, accept=None,
                  schema: dict | None = None, schema_name: str = "payload") -> Any:
        """按“健康度 + 成功率 + 延迟”动态排序尝试模型；失败快速切换，不做长尾重试。

        - 非思考模型优先；思考模型仅在 allow_thinking=true 时参与；
        - 单块总预算 max_total_s 封顶，最坏不超过“2 个模型 × 单次超时”的量级；
        - schema/JSON 不合法视为本模型失败，立即换下一个；全部失败抛 LLMChainError。
        """
        self._bump("json_calls")
        self._ensure_warm()
        deadline = time.monotonic() + self.max_total_s
        use_schema = bool(self.structured_enabled and schema)
        models = self._ordered_models(prefer)
        failures: list[dict] = []
        switched_after_schema = False
        for index, model in enumerate(models):
            if time.monotonic() >= deadline - 0.5:
                failures.append({"model": model, "error": "单块总耗时预算已用尽，未尝试"})
                break
            start = time.monotonic()
            try:
                data = self._try_model(model, messages, temperature,
                                       schema if use_schema else None, schema_name,
                                       deadline, accept)
                self._record(model, True, time.monotonic() - start)
                self._bump("json_success")
                return data
            except Exception as err:
                elapsed = time.monotonic() - start
                kind = classify_error(err)
                model_error = f"{type(err).__name__}: {err}"
                if "schema 校验失败" in str(err):
                    switched_after_schema = True
                self._record(model, False, elapsed, model_error, kind)
                failures.append({"model": model, "error": model_error,
                                 "kind": kind, "latency_s": round(elapsed, 2)})
                if switched_after_schema:
                    self._bump("schema_switch")
                    switched_after_schema = False
                if index + 1 < len(models):
                    self._bump("model_switches")
                    log.warning("模型 %s 失败（%s，%.2fs），切换下一个", model, kind, elapsed)
        self._bump("chain_failures")
        raise LLMChainError(failures)

    def probe_models(self, models: list[str] | None = None) -> list[dict]:
        """逐个模型发最小请求，返回可用性/延迟/是否思考模型，用于跑批前确认权限/下架情况。"""
        results: list[dict] = []
        for model in (models or self.models):
            start = time.time()
            try:
                reply = self._request_once(
                    model, [{"role": "user", "content": "只输出：{\"ok\":true}"}],
                    0.0, 512, timeout=self.timeout)
                results.append({
                    "model": model, "ok": True, "latency_s": round(time.time() - start, 2),
                    "reply": reply[:80].replace("\n", " "), "error": "",
                    "thinking": bool(getattr(self, "_last_reasoning", False)),
                })
            except Exception as err:
                results.append({
                    "model": model, "ok": False, "latency_s": round(time.time() - start, 2),
                    "reply": "", "error": f"{type(err).__name__}: {err}",
                    "thinking": None, "kind": classify_error(err),
                })
        return results

    def map(self, fn: Callable[[T], R], items: Iterable[T], workers: int | None = None) -> list[R]:
        """小并发上限的保序 map；异常原样抛出，便于定位具体章节。"""
        items = list(items)
        workers = max(1, min(workers or self.concurrency, self.concurrency, len(items) or 1))
        if workers == 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items))
