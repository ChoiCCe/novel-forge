"""旁白参数定稿锁：配置指纹校验。

锁写在 config.yaml -> casting.narrator：
    locked: true
    fingerprint: "<sha256>"
    locked_values: {voice, rate, pitch, volume}

启动时只比较配置层（tts.engines.edge.narrator_voice / caster.narrator_*），
不比较 registry 的单书覆盖；锁定后改动任一字段都会报错并列出差异。
"""
from __future__ import annotations

import hashlib
import json

from . import prosody

FIELDS = ("voice", "rate", "pitch", "volume")


class NarratorLockError(RuntimeError):
    """旁白配置与锁定指纹不一致。"""


def effective(cfg: dict) -> dict:
    """与 Caster.narrator_profile 的配置层取值保持一致。"""
    casting = cfg.get("casting") or {}
    narrator = casting.get("narrator") or {}
    caster = cfg.get("caster") or {}
    edge = (((cfg.get("tts") or {}).get("engines") or {}).get("edge") or {})
    voice = edge.get("narrator_voice") or narrator.get("default_voice") or ""
    return {
        "voice": str(voice),
        "rate": str(caster.get("narrator_rate") or narrator.get("rate") or "+0%"),
        "pitch": str(caster.get("narrator_pitch") or narrator.get("pitch") or "+0Hz"),
        "volume": str(caster.get("narrator_volume") or narrator.get("volume") or "+0%"),
    }


def normalize(values: dict) -> dict:
    return {
        "voice": str(values.get("voice") or ""),
        "rate": prosody.format_value(prosody.parse_value(values.get("rate"), "%"), "%"),
        "pitch": prosody.format_value(prosody.parse_value(values.get("pitch"), "Hz"), "Hz"),
        "volume": prosody.format_value(prosody.parse_value(values.get("volume"), "%"), "%"),
    }


def fingerprint(values: dict) -> str:
    payload = json.dumps(normalize(values), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def diff(expected: dict, current: dict) -> dict:
    left, right = normalize(expected), normalize(current)
    return {key: {"locked": left[key], "current": right[key]}
            for key in FIELDS if left[key] != right[key]}


def check(cfg: dict, override: bool = False) -> dict:
    """启动校验。未锁定时直接返回；锁定时不一致抛 NarratorLockError。"""
    narrator = (cfg.get("casting") or {}).get("narrator") or {}
    current = effective(cfg)
    result = {"locked": bool(narrator.get("locked")), "override": False,
              "fingerprint": fingerprint(current), "values": normalize(current), "diff": {}}
    if not result["locked"]:
        return result
    stored = str(narrator.get("fingerprint") or "").strip()
    if not stored:
        raise NarratorLockError(
            "casting.narrator.locked=true 但缺少 fingerprint；"
            "请用 `python tools/narrator_lock.py --lock` 重新锁定，或改为 locked=false")
    locked_values = narrator.get("locked_values") or {}
    result["diff"] = diff(locked_values, current) if locked_values else {}
    if result["fingerprint"] == stored:
        return result
    differences = result["diff"] or {"fingerprint": {"locked": stored,
                                                     "current": result["fingerprint"]}}
    if override:
        result["override"] = True
        return result
    lines = []
    for key, item in differences.items():
        if key == "fingerprint":
            lines.append("指纹不一致（locked_values 缺失，无法逐项对比）")
        else:
            lines.append(f"{key}: 锁定值 {item['locked']} -> 当前 {item['current']}")
    raise NarratorLockError(
        "旁白参数已锁定，但配置被改动：" + "；".join(lines)
        + "。如确认要改，请加 --override-narrator-lock，或先运行 tools/narrator_lock.py --lock 重新定稿")
