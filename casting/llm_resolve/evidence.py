# -*- coding: utf-8 -*-
"""证据逐字定位。

硬约束：evidence_quotes 必须是原文的连续子串。只允许两种无损校正：
1) 剥掉最外层成对的引号/括号/空白；
2) 忽略引号字符本身（“”‘’「」『』"'），因为模型常把外层引号抄丢或抄错，
   但文字内容仍逐字一致。
不做同义改写、模糊匹配或省略号补全。
"""
from __future__ import annotations

_WRAPPERS = "\"'“”‘’「」『』《》〈〉[]【】()（） \t\r\n\u3000"
_QUOTE_CHARS = "“”‘’「」『』\"'`"


def _strip_quotes(text: str) -> str:
    return text.translate({ord(ch): None for ch in _QUOTE_CHARS})


def locate_verbatim(text: str, quote: str) -> int:
    """返回 quote 在 text 中的起始下标；定位不到返回 -1。"""
    if not isinstance(text, str) or not isinstance(quote, str):
        return -1
    q = quote.strip()
    if not q:
        return -1
    pos = text.find(q)
    if pos >= 0:
        return pos
    # 仅剥掉最外层包装符号后再试一次（LLM 常把引号一起抄进来）
    stripped = q.strip(_WRAPPERS)
    if stripped and stripped != q:
        pos = text.find(stripped)
        if pos >= 0:
            return pos
    # 忽略引号字符本身后再定位（内容仍要求连续）
    norm_q = _strip_quotes(q)
    if norm_q:
        norm_text = _strip_quotes(text)
        pos = norm_text.find(norm_q)
        if pos >= 0:
            return pos
    return -1


def split_evidence(quotes, text: str) -> tuple[list[str], list[str]]:
    """返回 (可定位, 不可定位)。保持原始顺序与内容。"""
    valid: list[str] = []
    invalid: list[str] = []
    for quote in quotes or []:
        if not isinstance(quote, str) or not quote.strip():
            continue
        (valid if locate_verbatim(text, quote) >= 0 else invalid).append(quote)
    return valid, invalid


def evidence_hits(quote: str, text: str) -> bool:
    return locate_verbatim(text, quote) >= 0
