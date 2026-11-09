# -*- coding: utf-8 -*-
"""任务二：低归因 / 引号识别失败的说话人归属。

上下文从待判定处向两侧逐级扩大（±1 → ±3 → ±5 …），上下文累计不超过
1000 字（不含待判定段本身）。到上限仍判不出就返回 unresolved，不猜。
低置信度只记录，不采纳。
"""
from __future__ import annotations

from typing import Iterable

from .client import Usage, call_json, get_client
from .evidence import split_evidence
from .schemas import SPEAKER_SCHEMA, as_str_list, clamp_confidence, strip_forbidden

DEFAULT_RADII = (1, 3, 5, 7, 9, 12, 16, 20, 30, 50)
MAX_CONTEXT_CHARS = 1000

NON_SPEAKERS = {
    "旁白", "敘述", "叙述", "narrator", "unknown", "未知", "不确定",
    "他", "她", "它", "他们", "她们", "众人", "有人", "某人",
}

LANG_HINT = {
    "zh": "中文小说：引号可能是“”、「」、『』；注意“道/说/问/答/笑道”等叙述动词。",
    "en": "English fiction: dialogue may use \"...\" or “...”; dialogue tags "
          "(said/asked/replied/cried) can precede or follow the quote; watch pronouns.",
    "ja": "日本語小説：「」『』が会話、地の文の「と言った／と答えた／と尋ねた」も手がかり。",
}

SYSTEM_PROMPT = """你是小说台词说话人判定器。只输出 JSON。
硬规则：
1. 只根据给定上下文判断，不得引入外部知识。
2. 证据必须从上下文逐字复制连续子串，不得改写、拼接、加省略号。
3. 判不出来就 resolved=false、speaker=""，不要猜；宁可不判，不可错判。
4. speaker 必须是角色名；不要返回“他/她/旁白/未知”这类代词或占位词。
5. 若待判定段没有引号但含自由间接引语/转述，可以判定；纯叙述则 resolved=false。
6. 禁止输出任何 voice_id、声线名、音色、参数值；禁止判断年龄。
输出字段：resolved(bool), speaker(string), confidence(0~1), evidence_quotes(string[]), reason(string)。"""


def _normalize_characters(characters: Iterable) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in characters or []:
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


def build_context(paragraphs: list, index: int, radius: int,
                  max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[int], int, bool]:
    """构造 ±radius 段上下文。

    返回 ``(context_text, used_indices, context_chars, truncated)``。
    ``context_chars`` 不含待判定段；``None`` 视为场景边界，``""`` 跳过。
    """
    left: list[int] = []
    right: list[int] = []
    for d in range(1, radius + 1):
        j = index - d
        if j < 0 or paragraphs[j] is None:
            break
        if paragraphs[j]:
            left.append(j)
    for d in range(1, radius + 1):
        j = index + d
        if j >= len(paragraphs) or paragraphs[j] is None:
            break
        if paragraphs[j]:
            right.append(j)

    def ctx_chars() -> int:
        return sum(len(paragraphs[j]) + 1 for j in left + right)

    raw_chars = ctx_chars()
    truncated = raw_chars > max_chars
    # 从最远端开始裁剪，直到上下文不超过预算
    while (left or right) and ctx_chars() > max_chars:
        ldist = index - left[0] if left else 10 ** 9
        rdist = right[-1] - index if right else 10 ** 9
        if ldist >= rdist:
            left.pop(0)
        else:
            right.pop()

    used = sorted(left + [index] + right)
    parts = []
    for j in used:
        if j == index:
            tag = "【待判定】"
        elif j < index:
            tag = "【上文】"
        else:
            tag = "【下文】"
        parts.append(f"{tag}{paragraphs[j]}")
    context = "\n".join(parts)
    return context, used, ctx_chars(), truncated


def _messages(context: str, characters: list[str], lang: str) -> list[dict]:
    names_text = "、".join(characters[:400]) if characters else "（未提供）"
    hint = LANG_HINT.get(lang, LANG_HINT["zh"])
    user = (
        f"【语言/文本类型】\n{hint}\n\n"
        f"【已知角色名】\n{names_text}\n\n"
        f"【上下文（【待判定】为需要归因的段落）】\n{context}\n\n"
        "请判断【待判定】段落中的台词/自由间接引语是谁说的。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _result(status: str, *, speaker: str | None, confidence: float,
            evidence: list[str], usage: Usage, window_chars: int,
            radius: int, attempts: int, history: list[dict],
            reason: str = "", degraded: bool = False,
            speaker_new: bool = False, context: str = "") -> dict:
    return {
        "task": "speaker",
        "status": status,
        "degraded": degraded,
        "speaker": speaker,
        "speaker_new": speaker_new,
        "confidence": round(confidence, 4),
        "evidence_quotes": evidence,
        "window_chars": window_chars,
        "radius": radius,
        "attempts": attempts,
        "history": history,
        "reason": reason,
        "context_preview": context[:600],
        "usage": usage.as_dict(),
    }


def resolve_speaker(paragraphs: list, index: int, characters, *,
                    lang: str = "zh",
                    max_context_chars: int = MAX_CONTEXT_CHARS,
                    min_confidence: float = 0.7,
                    client=None, rule_speaker: str | None = None,
                    radii=DEFAULT_RADII,
                    usage: Usage | None = None) -> dict:
    usage = usage or Usage()
    if not isinstance(paragraphs, list) or not (0 <= index < len(paragraphs)):
        return _result("invalid", speaker=None, confidence=0.0, evidence=[],
                       usage=usage, window_chars=0, radius=0, attempts=0,
                       history=[], reason="bad paragraphs/index")
    target = paragraphs[index]
    if not target:
        return _result("invalid", speaker=None, confidence=0.0, evidence=[],
                       usage=usage, window_chars=0, radius=0, attempts=0,
                       history=[], reason="empty target paragraph")

    chars = _normalize_characters(characters)
    try:
        llm = client or get_client()
    except Exception as exc:  # noqa: BLE001
        return _result("pending", speaker=rule_speaker, confidence=0.0,
                       evidence=[], usage=usage, window_chars=0, radius=0,
                       attempts=0, history=[],
                       reason=f"llm unavailable: {type(exc).__name__}: {exc}",
                       degraded=True)

    history: list[dict] = []
    previous_used: tuple[int, ...] | None = None
    last_status = "unresolved"
    last_conf = 0.0
    last_window = 0
    last_radius = 0
    last_context = ""
    last_evidence: list[str] = []
    last_reason = ""
    speaker_new = False
    attempts = 0

    for radius in radii:
        context, used, ctx_chars, truncated = build_context(
            paragraphs, index, radius, max_context_chars)
        used_key = tuple(used)
        if used_key == previous_used:
            continue
        previous_used = used_key
        attempts += 1
        last_window, last_radius, last_context = ctx_chars, radius, context

        data, error = call_json(llm, _messages(context, chars, lang),
                                SPEAKER_SCHEMA, "llm_resolve_speaker", usage)
        if data is None:
            return _result("pending", speaker=rule_speaker, confidence=0.0,
                           evidence=[], usage=usage, window_chars=ctx_chars,
                           radius=radius, attempts=attempts, history=history,
                           reason=error or "llm call failed", degraded=True)
        if not isinstance(data, dict):
            last_status, last_reason = "invalid", "non-object response"
            history.append({"radius": radius, "window_chars": ctx_chars,
                            "status": last_status, "reason": last_reason})
            break

        strip_forbidden(data)
        resolved = bool(data.get("resolved"))
        speaker = str(data.get("speaker") or "").strip()
        confidence = clamp_confidence(data.get("confidence"))
        evidence = as_str_list(data.get("evidence_quotes"))
        reason = str(data.get("reason") or "")
        last_conf, last_reason = confidence, reason

        if not resolved or not speaker or speaker in NON_SPEAKERS:
            last_status = "unresolved"
            history.append({"radius": radius, "window_chars": ctx_chars,
                            "status": last_status, "speaker": None,
                            "confidence": confidence, "reason": reason})
            if truncated:
                break
            continue

        valid, invalid = split_evidence(evidence, context)
        if invalid or not valid:
            last_status = "invalid"
            last_reason = ("evidence not verbatim: " + " | ".join(invalid)[:300]
                           if invalid else "resolved without evidence")
            history.append({"radius": radius, "window_chars": ctx_chars,
                            "status": last_status, "speaker": speaker,
                            "confidence": confidence, "invalid": invalid,
                            "reason": last_reason})
            if truncated:
                break
            continue

        in_list = speaker in chars
        if not in_list and context.find(speaker) < 0:
            last_status = "invalid"
            last_reason = f"speaker not in character list or context: {speaker}"
            history.append({"radius": radius, "window_chars": ctx_chars,
                            "status": last_status, "speaker": speaker,
                            "confidence": confidence, "reason": last_reason})
            if truncated:
                break
            continue

        speaker_new = not in_list
        last_evidence = valid
        if confidence < min_confidence:
            last_status = "low_confidence"
            history.append({"radius": radius, "window_chars": ctx_chars,
                            "status": last_status, "speaker": speaker,
                            "confidence": confidence, "reason": reason})
            if truncated:
                break
            continue

        history.append({"radius": radius, "window_chars": ctx_chars,
                        "status": "resolved", "speaker": speaker,
                        "confidence": confidence, "reason": reason})
        return _result("resolved", speaker=speaker, confidence=confidence,
                       evidence=valid, usage=usage, window_chars=ctx_chars,
                       radius=radius, attempts=attempts, history=history,
                       reason=reason, speaker_new=speaker_new, context=context)

    status = last_status if last_status in ("low_confidence", "invalid") else "unresolved"
    return _result(status, speaker=None, confidence=last_conf,
                   evidence=last_evidence, usage=usage,
                   window_chars=last_window, radius=last_radius,
                   attempts=attempts, history=history,
                   reason=last_reason or "unresolved within context budget",
                   context=last_context)
