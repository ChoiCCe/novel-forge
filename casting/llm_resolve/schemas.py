# -*- coding: utf-8 -*-
"""强制 JSON Schema 与语义校验。

JSON Schema 先由 ``core.llm.chat_json`` 做结构校验；本模块再做一层语义校验，
因为“结构合法”不等于“可以采纳”：证据必须逐字定位、canonical 必须在名单内、
不允许出现声线字段、年龄数值必须被剥离。
"""
from __future__ import annotations

from typing import Any

ALIAS_SCHEMA = {
    "type": "object",
    "required": ["merge", "alias", "canonical", "confidence", "evidence_quotes"],
    "properties": {
        "merge": {"type": "boolean"},
        "alias": {"type": "string"},
        "canonical": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence_quotes": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
}

SPEAKER_SCHEMA = {
    "type": "object",
    "required": ["resolved", "speaker", "confidence", "evidence_quotes"],
    "properties": {
        "resolved": {"type": "boolean"},
        "speaker": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence_quotes": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
}

_MULTI_TURN = {
    "type": "object",
    "required": ["quote", "speaker", "confidence", "evidence"],
    "properties": {
        "quote": {"type": "string"},
        "speaker": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
}

_MULTI_AGE = {
    "type": "object",
    "required": ["quote", "kind"],
    "properties": {
        "quote": {"type": "string"},
        "kind": {"type": "string", "enum": ["number", "title"]},
    },
}

_MULTI_ENTITY = {
    "type": "object",
    "required": ["name", "aliases", "gender", "gender_evidence",
                 "age_evidence", "turns"],
    "properties": {
        "name": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}},
        "gender": {"type": "string", "enum": ["男", "女", "未知"]},
        "gender_evidence": {"type": "array", "items": {"type": "string"}},
        "age_evidence": {"type": "array", "items": _MULTI_AGE},
        "turns": {"type": "array", "items": _MULTI_TURN},
    },
}

MULTILINGUAL_SCHEMA = {
    "type": "object",
    "required": ["entities", "unresolved"],
    "properties": {
        "entities": {"type": "array", "items": _MULTI_ENTITY},
        "unresolved": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["quote", "reason"],
                "properties": {
                    "quote": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}

FORBIDDEN_KEY_PARTS = (
    "voice", "voice_id", "speaker_id", "rate", "pitch", "volume", "声线",
    "音色", "参数值",
)


def strip_forbidden(obj: Any, path: str = "") -> list[str]:
    """递归移除声线相关字段；返回被移除的路径，便于报告违规。"""
    removed: list[str] = []
    if isinstance(obj, dict):
        for key in list(obj):
            low = str(key).lower()
            if any(part in low for part in FORBIDDEN_KEY_PARTS):
                removed.append(f"{path}.{key}" if path else str(key))
                obj.pop(key, None)
                continue
            removed.extend(strip_forbidden(obj[key], f"{path}.{key}" if path else str(key)))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            removed.extend(strip_forbidden(item, f"{path}[{i}]"))
    return removed


def clamp_confidence(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    if num != num:  # NaN
        return 0.0
    return max(0.0, min(1.0, num))


def as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out
