"""声线守门测试：对 config.yaml 与 voices/catalog.edge.json 实际引用的每个声线合成一句固定短句。

只读配置、只写临时音频（测完即删），不修改 registry / catalog / 缓存。
任一声线失败即以非零码退出，适合作为 edge-tts 升级后的守门检查。

用法：
    python scripts/voice_smoke.py
    python scripts/voice_smoke.py --timeout 90
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXED_SENTENCE = "这是一句用于声线健康检查的固定短句，今天天气不错。"


def _load_yaml(path: Path) -> dict:
    import yaml  # requirements.txt 已包含 PyYAML

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_json(path: Path) -> dict:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_voices(cfg: dict, catalog: dict) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """返回 (声线 -> 引用来源列表, 跳过项列表)。跳过的只记录原因，不参与合成。"""
    refs: dict[str, list[str]] = {}

    def add(voice: str | None, source: str) -> None:
        if isinstance(voice, str) and voice.strip():
            refs.setdefault(voice.strip(), []).append(source)

    edge = cfg.get("tts", {}).get("engines", {}).get("edge", {})
    add(edge.get("narrator_voice"), "tts.engines.edge.narrator_voice")
    add(edge.get("narrator_voice_alt"), "tts.engines.edge.narrator_voice_alt")
    for key, value in (edge.get("voice_map") or {}).items():
        add(value, f"tts.engines.edge.voice_map.{key}")
    for stage, by_gender in (edge.get("age_stage_voice_map") or {}).items():
        for gender, value in (by_gender or {}).items():
            add(value, f"tts.engines.edge.age_stage_voice_map.{stage}.{gender}")
    for keyword, item in (edge.get("special") or {}).items():
        if isinstance(item, dict):
            add(item.get("voice"), f"tts.engines.edge.special.{keyword}")

    casting = cfg.get("casting", {})
    add(casting.get("narrator", {}).get("default_voice"), "casting.narrator.default_voice")
    add(casting.get("gender", {}).get("unknown_voice"), "casting.gender.unknown_voice")

    skipped: dict[str, str] = {}
    for entry in catalog.get("voices", []):
        voice = entry.get("id")
        if not voice:
            continue
        if entry.get("available"):
            # catalog 里已由 config 引用的声线会合并到同一行，来源里补一条 catalog
            add(voice, "catalog.voices")
        else:
            skipped[voice] = entry.get("error") or entry.get("status") or "catalog 标记不可用"

    for source_voice, fallback in (catalog.get("fallbacks") or {}).items():
        if isinstance(fallback, dict):
            add(fallback.get("voice"), f"catalog.fallbacks.{source_voice}")
            skipped.setdefault(source_voice,
                               f"catalog 标记不可用（由 fallback 覆盖 -> {fallback.get('voice')}）")
    return refs, list(skipped.items())


async def _synthesize(voice: str, timeout_s: float) -> None:
    import edge_tts

    with tempfile.TemporaryDirectory(prefix="novel-forge-smoke-") as tmp:
        out = Path(tmp) / "smoke.mp3"
        communicate = edge_tts.Communicate(FIXED_SENTENCE, voice)
        await asyncio.wait_for(communicate.save(str(out)), timeout=timeout_s)
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError("edge-tts 返回空音频")


def classify(error: str) -> str:
    text = (error or "").lower()
    if "noaudio" in text:
        return "no_audio"
    if "403" in text or "forbidden" in text:
        return "forbidden"
    if "429" in text or "rate" in text:
        return "rate_limited"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "connection" in text or "ssl" in text or "network" in text or "resolve" in text:
        return "network"
    return "engine"


def _short(message: str, limit: int = 60) -> str:
    message = " ".join((message or "").split())
    return message[:limit] + ("…" if len(message) > limit else "")


def main() -> int:
    parser = argparse.ArgumentParser(description="edge 声线是否可用（升级后守门测试）")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--catalog", default=str(PROJECT_ROOT / "voices" / "catalog.edge.json"))
    parser.add_argument("--timeout", type=float, default=60.0, help="单条合成超时秒数")
    args = parser.parse_args()

    config_path = Path(args.config)
    catalog_path = Path(args.catalog)
    cfg = _load_yaml(config_path)
    catalog = _load_json(catalog_path)
    refs, skipped = collect_voices(cfg, catalog)
    if not refs:
        print("没有从 config/catalog 中收集到任何声线，检查配置路径。")
        return 1

    print(f"固定短句：{FIXED_SENTENCE}")
    print(f"声线数量：{len(refs)}（跳过 {len(skipped)} 条 catalog 已标记不可用）\n")

    header = f"{'声线':<32} {'结果':<6} {'耗时(ms)':>10}  {'来源 / 错误'}"
    print(header)
    print("-" * len(header))

    failures = 0
    for voice in sorted(refs):
        started = time.perf_counter()
        status, detail = "OK", "、".join(dict.fromkeys(refs[voice]))
        try:
            asyncio.run(_synthesize(voice, args.timeout))
        except Exception as err:  # noqa: BLE001 - 守门脚本要完整列出所有失败
            elapsed = int((time.perf_counter() - started) * 1000)
            failures += 1
            status = "FAIL"
            detail = f"{classify(str(err))} · {_short(str(err))}"
            print(f"{voice:<32} {status:<6} {elapsed:>10}  {detail}")
            continue
        elapsed = int((time.perf_counter() - started) * 1000)
        print(f"{voice:<32} {status:<6} {elapsed:>10}  {detail}")

    for voice, reason in skipped:
        print(f"{voice:<32} {'SKIP':<6} {0:>10}  catalog 不可用，不参与：{reason}")

    print(f"\n汇总：成功 {len(refs) - failures}/{len(refs)}，失败 {failures}，跳过 {len(skipped)}")
    if failures:
        print("存在失败声线：按 docs/edge-tts-update.md 第 6 节排查，必要时回滚 edge-tts 版本。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
