"""v4 第 0 步：edge 声线可用性审计。

做三件事：
1. 从 config.yaml / voices/registry.json 收集所有被引用的声线及引用位置；
2. 拉边缘 list-voices 做差集，再对每个被引用声线逐条试合成（短句，多次重试）；
3. 把“已下架/仅预览”与“临时网络失败”分开，输出 JSON/Markdown 报告。

用法：
    python tools/voice_audit.py             # 审计
    python tools/voice_audit.py --no-probe  # 只做列表差集，不试合成
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import edge_tts  # noqa: E402
import yaml  # noqa: E402

OUT_DIR = ROOT.parent / "outputs" / "voice_fallback_audition"
PROBE_TEXT = "这一句用来检查声线是否可以正常合成，听到声音就说明可用。"
TRIES = 3


def collect_refs(cfg: dict, reg: dict) -> dict[str, list[str]]:
    """收集声线 -> 引用来源列表。"""
    refs: dict[str, list[str]] = {}

    def add(voice, where):
        if isinstance(voice, str) and voice.startswith("zh-") and voice.endswith("Neural"):
            refs.setdefault(voice, [])
            if where not in refs[voice]:
                refs[voice].append(where)

    edge_cfg = cfg.get("tts", {}).get("engines", {}).get("edge", {})
    add(edge_cfg.get("narrator_voice"), "config:tts.engines.edge.narrator_voice")
    add(edge_cfg.get("narrator_voice_alt"), "config:tts.engines.edge.narrator_voice_alt")
    for stage, m in (edge_cfg.get("age_stage_voice_map") or {}).items():
        for gender, voice in (m or {}).items():
            add(voice, f"config:age_stage_voice_map.{stage}.{gender}")
    for key, voice in (edge_cfg.get("voice_map") or {}).items():
        add(voice, f"config:voice_map.{key}")
    for key, voice in (edge_cfg.get("special") or {}).items():
        add(voice, f"config:special.{key}")

    for key, binding in (reg.get("voice_bindings") or {}).items():
        add(binding.get("voice"), f"registry:voice_bindings[{key}].voice")
        add(binding.get("legacy_voice"), f"registry:voice_bindings[{key}].legacy_voice")
    for ev in (reg.get("voice_events") or []):
        add(ev.get("from_voice"), "registry:voice_events.from_voice")
        add(ev.get("to_voice"), "registry:voice_events.to_voice")

    casting = cfg.get("casting") or {}
    narr = casting.get("narrator") or {}
    add(narr.get("default_voice"), "config:casting.narrator.default_voice")
    add((casting.get("gender") or {}).get("unknown_voice"), "config:casting.gender.unknown_voice")
    for k, v in ((casting.get("catalog") or {}).get("fallbacks") or {}).items():
        add(k, "config:casting.catalog.fallbacks(key)")
        add(v, "config:casting.catalog.fallbacks(value)")

    return refs


def classify(err: Exception) -> str:
    name = type(err).__name__
    msg = str(err)[:300]
    if "NoAudioReceived" in name or "NoAudioReceived" in msg:
        return "no_audio"          # 服务端不下发音频：声线下架/仅预览
    if "403" in msg or "Forbidden" in msg or "401" in msg:
        return "http_auth"         # 鉴权/封禁，需重试观察
    if "Invalid response status" in msg:
        return "http_status"
    if "Timeout" in name or "timed out" in msg.lower():
        return "timeout"
    if "ClientConnector" in name or "Cannot connect" in msg or "SSL" in name:
        return "network"
    if "empty" in msg.lower() or "空音频" in msg:
        return "empty_audio"
    return "other"


async def probe(voice: str) -> dict:
    last = None
    for attempt in range(1, TRIES + 1):
        t0 = time.time()
        try:
            with tempfile.TemporaryDirectory(prefix="vf-audit-") as tmp:
                mp3 = Path(tmp) / "p.mp3"
                comm = edge_tts.Communicate(PROBE_TEXT, voice)
                await comm.save(str(mp3))
                size = mp3.stat().st_size if mp3.exists() else 0
                if size == 0:
                    raise RuntimeError("empty audio file")
                return {"ok": True, "bytes": size, "elapsed_ms": int((time.time() - t0) * 1000),
                        "attempts": attempt, "error_type": "", "error": ""}
        except Exception as err:  # noqa: BLE001
            last = {"ok": False, "bytes": 0, "elapsed_ms": int((time.time() - t0) * 1000),
                    "attempts": attempt, "error_type": classify(err), "error": str(err)[:300]}
            if attempt < TRIES:
                await asyncio.sleep(2)
    return last


async def main_async(args) -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    reg = json.loads((ROOT / "voices" / "registry.json").read_text(encoding="utf-8"))
    refs = collect_refs(cfg, reg)

    listed = await edge_tts.list_voices()
    zh = [v for v in listed if str(v.get("Locale", "")).startswith("zh")]
    listed_ids = {v["ShortName"] for v in listed}
    zh_ids = {v["ShortName"] for v in zh}

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "probe_text": PROBE_TEXT,
        "tries": TRIES,
        "list_voices_total": len(listed),
        "list_voices_zh": len(zh),
        "zh_list": sorted(zh_ids),
        "referenced": {},
        "missing_from_list": [],
        "not_in_zh_list_but_global": [],
        "ok": [],
        "failed_permanent": [],
        "failed_transient": [],
    }

    for voice, where in sorted(refs.items()):
        entry = {"voice": voice, "refs": where, "in_list": voice in listed_ids, "in_zh_list": voice in zh_ids}
        if not entry["in_list"]:
            report["missing_from_list"].append(voice)
        if entry["in_list"] and not entry["in_zh_list"]:
            report["not_in_zh_list_but_global"].append(voice)
        if args.no_probe:
            entry["probe"] = {"skipped": True}
        else:
            entry["probe"] = await probe(voice)
        report["referenced"][voice] = entry
        if entry.get("probe", {}).get("ok"):
            report["ok"].append(voice)
        elif not entry.get("probe", {}).get("skipped"):
            if entry["probe"]["error_type"] in ("no_audio", "empty_audio"):
                report["failed_permanent"].append(voice)
            else:
                report["failed_transient"].append(voice)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# edge 声线可用性审计", "",
             f"- 生成时间：{report['generated_at']}",
             f"- list-voices：全部 {len(listed)} 条，中文 {len(zh)} 条",
             f"- 被引用声线：{len(refs)} 条；可用 {len(report['ok'])}，"
             f"永久失败 {len(report['failed_permanent'])}，临时失败 {len(report['failed_transient'])}",
             "", "| 声线 | 列表内 | 试合成 | 错误类型 | 引用位置 |", "|---|---:|---:|---|---|"]
    for voice, entry in report["referenced"].items():
        p = entry.get("probe") or {}
        if p.get("skipped"):
            status = "未探测"
        elif p.get("ok"):
            status = f"可用（{p['bytes']}B / {p['elapsed_ms']}ms）"
        else:
            status = f"失败 ×{p.get('attempts')}"
        lines.append(f"| `{voice}` | {'是' if entry['in_list'] else '否'} | {status} | "
                     f"{p.get('error_type') or '-'} | {', '.join(entry['refs'])} |")
    (OUT_DIR / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-probe", action="store_true")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
