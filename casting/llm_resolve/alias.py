# -*- coding: utf-8 -*-
"""任务一：人名别名合并。

原则：宁可漏合并，不可误合并。两个真实角色共现、同姓、同尾字、同辈分
都不构成合并理由；证据不能逐字定位时整条作废。
"""
from __future__ import annotations

import re
from typing import Iterable

from .client import Usage, call_json, get_client
from .evidence import split_evidence
from .schemas import (ALIAS_SCHEMA, as_str_list, clamp_confidence,
                      strip_forbidden)

GENERIC_TITLES = {
    "公子", "姑娘", "大人", "先生", "小姐", "将军", "掌柜", "前辈",
    "老爷子", "老丈", "夫人", "太太", "老爷", "陛下", "殿下", "阁下",
}

# 明确指向“同一人”的字面标记。只有双方都在已知名单里时才用于防误合并。
IDENTITY_MARKERS = re.compile(
    r"(名叫|本名|原名|真名|绰号|外号|人称|称他|称她|称作|叫作|叫做|"
    r"就是|同一人|同一人|改名|化名|自称|又称|别名|冒充|顶替|冒名|"
    r"用这个名字|对外称|冒用)"
)

SYSTEM_PROMPT = """你是中文小说人物称谓消歧器。只做一件事：判断“待判定称谓”是否指代“已知人名”中的某一个。
硬规则：
1. 只输出 JSON，不要输出解释性正文。
2. 宁可漏合并，不可误合并。两个真实角色即使同姓、同尾字、同时出现、关系亲密，也绝不能合并。
3. canonical 必须从“已知人名”中原样选择；不确定就 merge=false，canonical 留空。
4. evidence_quotes 必须从【原文片段】逐字复制连续子串，不得改写、缩写、拼接、加省略号；至少一条要能直接支持“同一人”。
5. 敬称/绰号（如“楼公子”“薛祖”）可以合并；仅凭姓氏相同或称谓相近不行。
6. 如果【待判定称谓】已经在【已知人名】中，而原文表明它使用了另一个名字
   （如“我叫徐础”“从此改名”），canonical 必须填那个被使用的新名字，不能填它自己。
7. 规则层结果仅供参考；原文证据与规则层冲突时，以原文为准。
8. merge=true 时 canonical 必须不同于 alias，否则不要合并。
9. 禁止输出任何 voice_id、声线名、音色、参数值。
输出字段：merge(bool), alias(string), canonical(string), confidence(0~1), evidence_quotes(string[]), reason(string)。"""


def _normalize_names(names: Iterable) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in names or []:
        if isinstance(item, dict):
            name = item.get("canonical") or item.get("name")
        else:
            name = item
        if isinstance(name, str) and name.strip():
            name = name.strip()
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _messages(paragraph: str, names: list[str], candidate: str,
              rule_canonical: str | None) -> list[dict]:
    # 名单太长时截断，避免单次 prompt 失控；按长度优先保留长名（更具体）
    trimmed = sorted(names, key=lambda n: (-len(n), n))[:400]
    names_text = "、".join(trimmed) if trimmed else "（空）"
    user = (
        f"【原文片段】\n{paragraph}\n\n"
        f"【已知人名】\n{names_text}\n\n"
        f"【待判定称谓】\n{candidate}\n\n"
        f"【规则层当前结果】\n{rule_canonical or '无'}\n\n"
        f"请判断“{candidate}”是否就是已知人名中某一个的别名、敬称、绰号或同一人。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _result(status: str, *, alias: str, canonical: str, merged: bool,
            confidence: float, evidence: list[str], usage: Usage,
            reason: str = "", degraded: bool = False,
            invalid_evidence: list[str] | None = None,
            guard: str = "") -> dict:
    return {
        "task": "alias",
        "status": status,
        "degraded": degraded,
        "alias": alias,
        "canonical": canonical,
        "merged": merged,
        "confidence": round(confidence, 4),
        "evidence_quotes": evidence,
        "invalid_evidence": invalid_evidence or [],
        "guard": guard,
        "reason": reason,
        "usage": usage.as_dict(),
    }


def resolve_alias(paragraph: str, names, candidate: str, *,
                  client=None, rule_canonical: str | None = None,
                  min_confidence: float = 0.8,
                  usage: Usage | None = None) -> dict:
    """判定 ``candidate`` 是否合并到 ``names`` 中的某个正名。"""
    usage = usage or Usage()
    names_list = _normalize_names(names)
    candidate = (candidate or "").strip()

    if not paragraph or not candidate:
        return _result("invalid", alias=candidate, canonical=rule_canonical or "",
                       merged=False, confidence=0.0, evidence=[],
                       usage=usage, reason="empty paragraph or candidate")
    if candidate in GENERIC_TITLES:
        return _result("no_merge", alias=candidate, canonical="",
                       merged=False, confidence=0.0, evidence=[], usage=usage,
                       reason="generic title")

    try:
        llm = client or get_client()
    except Exception as exc:  # noqa: BLE001
        return _result("pending", alias=candidate, canonical=rule_canonical or "",
                       merged=False, confidence=0.0, evidence=[], usage=usage,
                       reason=f"llm unavailable: {type(exc).__name__}: {exc}",
                       degraded=True)

    data, error = call_json(llm, _messages(paragraph, names_list, candidate,
                                           rule_canonical),
                            ALIAS_SCHEMA, "llm_resolve_alias", usage)
    if data is None:
        return _result("pending", alias=candidate, canonical=rule_canonical or "",
                       merged=False, confidence=0.0, evidence=[], usage=usage,
                       reason=error or "llm call failed", degraded=True)

    if not isinstance(data, dict):
        return _result("invalid", alias=candidate, canonical=rule_canonical or "",
                       merged=False, confidence=0.0, evidence=[], usage=usage,
                       reason="non-object response")

    violated = strip_forbidden(data)
    alias = str(data.get("alias") or "").strip()
    canonical = str(data.get("canonical") or "").strip()
    confidence = clamp_confidence(data.get("confidence"))
    merge = bool(data.get("merge"))
    evidence = as_str_list(data.get("evidence_quotes"))
    reason = str(data.get("reason") or "")

    valid, invalid = split_evidence(evidence, paragraph)
    if invalid:
        # 硬约束：有一条定位不到，整条作废
        return _result("invalid", alias=candidate, canonical=rule_canonical or "",
                       merged=False, confidence=confidence, evidence=valid,
                       usage=usage, invalid_evidence=invalid,
                       reason="evidence not verbatim: " + " | ".join(invalid)[:300],
                       guard="invalid_evidence")

    if alias and alias != candidate:
        return _result("invalid", alias=candidate, canonical=rule_canonical or "",
                       merged=False, confidence=confidence, evidence=valid,
                       usage=usage, reason=f"alias mismatch: {alias}",
                       guard="alias_mismatch")

    if not merge:
        return _result("no_merge", alias=candidate, canonical="",
                       merged=False, confidence=confidence, evidence=valid,
                       usage=usage, reason=reason or "llm declined merge",
                       guard="llm_no_merge")

    # 以下 merge=True：逐条硬校验
    if canonical not in names_list:
        return _result("invalid", alias=candidate, canonical=rule_canonical or "",
                       merged=False, confidence=confidence, evidence=valid,
                       usage=usage,
                       reason=f"canonical not in name list: {canonical}",
                       guard="canonical_not_in_names")
    if canonical == candidate:
        return _result("invalid", alias=candidate, canonical=rule_canonical or "",
                       merged=False, confidence=confidence, evidence=valid,
                       usage=usage, reason="canonical equals alias",
                       guard="self_merge")
    if not valid:
        return _result("invalid", alias=candidate, canonical=rule_canonical or "",
                       merged=False, confidence=confidence, evidence=[],
                       usage=usage, reason="merge without locatable evidence",
                       guard="no_evidence")
    if not any((candidate in q) or (canonical in q) for q in valid):
        return _result("invalid", alias=candidate, canonical=rule_canonical or "",
                       merged=False, confidence=confidence, evidence=valid,
                       usage=usage,
                       reason="evidence mentions neither alias nor canonical",
                       guard="weak_evidence")

    # 防误合并：双方都在真实人名名单内时，必须有字面同一人标记，且置信度更高
    both_real = candidate in names_list and canonical in names_list
    if both_real:
        has_marker = any(IDENTITY_MARKERS.search(q) for q in valid)
        if not has_marker and confidence < 0.95:
            return _result("low_confidence", alias=candidate,
                           canonical=rule_canonical or "", merged=False,
                           confidence=confidence, evidence=valid, usage=usage,
                           reason="both names are real characters and no identity marker",
                           guard="real_characters_guard")
        min_confidence = max(min_confidence, 0.9)

    if confidence < min_confidence:
        return _result("low_confidence", alias=candidate, canonical="",
                       merged=False, confidence=confidence, evidence=valid,
                       usage=usage,
                       reason=f"confidence {confidence:.2f} < {min_confidence:.2f}",
                       guard="low_confidence")

    note = reason
    if violated:
        note = (note + " | " if note else "") + "stripped: " + ",".join(violated)
    return _result("merged", alias=candidate, canonical=canonical, merged=True,
                   confidence=confidence, evidence=valid, usage=usage,
                   reason=note, guard="merged")


def resolve_alias_batch(cases, *, client=None, min_confidence: float = 0.8,
                        usage: Usage | None = None) -> list[dict]:
    """批量判定。``cases`` 每项至少含 paragraph / names / candidate。"""
    usage = usage or Usage()
    results = []
    for case in cases or []:
        result = resolve_alias(
            case.get("paragraph", ""),
            case.get("names", []),
            case.get("candidate", ""),
            client=client,
            rule_canonical=case.get("rule_canonical"),
            min_confidence=min_confidence,
            usage=usage,
        )
        result["id"] = case.get("id")
        results.append(result)
    return results
