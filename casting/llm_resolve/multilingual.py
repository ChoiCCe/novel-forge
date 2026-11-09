# -*- coding: utf-8 -*-
"""任务三：跨语言正文的角色识别与对话归属。

不是把中文规则翻译过去，而是让 LLM 直接按目标语言的引号体系、称谓体系、
叙述动词做角色识别；本模块负责统一实体表、逐字证据校验、年龄段规则推导。

年龄硬约束：LLM 只允许摘录原文里明确的数字/称谓证据，不判断年龄数值；
age_band 由 ``derive_age_band`` 规则推导。
"""
from __future__ import annotations

import re
from typing import Any

from .client import Usage, call_json, get_client
from .evidence import locate_verbatim, split_evidence
from .schemas import MULTILINGUAL_SCHEMA, as_str_list, clamp_confidence, strip_forbidden

LANG_HINTS = {
    "en": (
        "English fiction. Dialogue may use \"...\" or “...”. Dialogue tags "
        "(said / asked / replied / cried / murmured) may precede or follow the quote. "
        "Titles such as Mr., Mrs., Miss, Lady, Sir are part of how characters address "
        "each other. Do not translate names."
    ),
    "ja": (
        "日本語小説。「」『』が会話。話者は「〜と言った／と答えた／と尋ねた」や"
        "敬称（さん／様／君／ちゃん／先生）から判断する。名前は日本語のまま出力する。"
    ),
    "zh": (
        "中文小说。引号可能是“”、「」、『』；叙述动词如道/说/问/答/笑道。"
    ),
}

SYSTEM_PROMPT = """你是跨语言小说角色与台词归属分析器。只输出 JSON。
任务：直接阅读目标语言正文，识别角色实体，并把每一句台词归到说话人。
硬规则：
1. 只依据原文，不得引入外部知识；不得翻译或改写人名。
2. 每条 quote / evidence 必须从原文逐字复制连续子串，不得改写、拼接、加省略号。
3. gender 只能是“男”“女”“未知”。判不出来就填“未知”，不要猜；不确定时宁可用“未知”。
4. 年龄：不要判断年龄，不要输出年龄数值、年龄段或“看起来多少岁”。
   只允许摘录原文中明确出现年龄数字或明确年龄称谓的短语，填入 age_evidence，
   kind 取 "number"（如“十七岁”“seventeen years old”“十七歳”）或
   "title"（如“老者”“少年”“old man”“老人”）。
5. 不确定说话人的台词放进 unresolved，不要瞎猜。
6. 禁止输出任何 voice_id、声线名、音色、参数值。
输出结构：
{"entities":[{"name","aliases":[],"gender","gender_evidence":[],
"age_evidence":[{"quote","kind"}],
"turns":[{"quote","speaker","confidence","evidence":[]}]}],
"unresolved":[{"quote","reason"}]}"""


_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_EN_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

AGE_TITLE_WORDS = (
    ("老年", ("老年", "老者", "老人", "老翁", "老太太", "老头", "老妇",
              "年迈", "白发", "花甲", "古稀", "old man", "old woman",
              "elderly", "aged", "おじいさん", "おばあさん", "老人", "老女", "翁")),
    ("中年", ("中年", "壮年", "middle-aged", "middle aged", "中年")),
    ("青年", ("青年", "年轻", "后生", "年轻人", "young man", "young woman",
              "youth", "若者", "青年")),
    ("少年", ("少年", "少女", "男孩", "女孩", "孩子", "小孩", "孩童",
              "boy", "girl", "child", "children", "子供", "坊や", "小僧")),
)


def _cn_number(text: str) -> int | None:
    if not text:
        return None
    if "十" not in text:
        if len(text) == 1 and text in _CN_DIGITS:
            return _CN_DIGITS[text]
        return None
    parts = text.split("十")
    if len(parts) != 2:
        return None
    left, right = parts
    tens = _CN_DIGITS.get(left, 1) if left else 1
    ones = _CN_DIGITS.get(right, 0) if right else 0
    return tens * 10 + ones


def _parse_age_number(quote: str) -> int | None:
    q = quote.strip()
    m = re.search(r"(\d{1,3})\s*(?:岁|歳|周岁|years?\s*old|year-old|yo\b)",
                  q, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"([一二两三四五六七八九十]{1,4})\s*(?:岁|歳|周岁)", q)
    if m:
        return _cn_number(m.group(1))
    for word, num in _EN_NUMBERS.items():
        if re.search(rf"\b{re.escape(word)}\b[^.]{{0,20}}(?:years?\s*old|year-old)",
                     q, re.IGNORECASE):
            return num
    m = re.search(r"aged?\s+(\d{1,3})", q, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _band_from_age(age: int) -> str:
    if age < 18:
        return "少年"
    if age < 36:
        return "青年"
    if age < 60:
        return "中年"
    return "老年"


def derive_age_band(age_evidence) -> str:
    """由“原文明确证据”推导年龄段；没有证据就是未知。"""
    if not isinstance(age_evidence, list):
        return "未知"
    numbers: list[int] = []
    titles: list[str] = []
    for item in age_evidence:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote") or "")
        kind = str(item.get("kind") or "")
        if not quote:
            continue
        if kind == "number" or re.search(r"\d", quote):
            age = _parse_age_number(quote)
            if age is not None and 0 < age < 130:
                numbers.append(age)
                continue
        if kind == "title" or not numbers:
            low = quote.lower()
            for band, words in AGE_TITLE_WORDS:
                if any(w.lower() in low for w in words):
                    titles.append(band)
                    break
    if numbers:
        # 多数投票，冲突时取中位数孩子的规则：数值优先于称谓
        counts: dict[int, int] = {}
        for age in numbers:
            counts[age] = counts.get(age, 0) + 1
        age = max(counts, key=lambda k: (counts[k], -k))
        return _band_from_age(age)
    if titles:
        counts: dict[str, int] = {}
        for band in titles:
            counts[band] = counts.get(band, 0) + 1
        if len(counts) == 1:
            return titles[0]
        return max(counts, key=counts.get)
    return "未知"


def _chunks(text: str, max_chars: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in paras:
        while len(para) > max_chars:
            head, para = para[:max_chars], para[max_chars:]
            if buf:
                chunks.append("\n".join(buf))
                buf, size = [], 0
            chunks.append(head)
        if size + len(para) + 1 > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks or ([text] if text else [])


def _messages(chunk: str, lang: str) -> list[dict]:
    hint = LANG_HINTS.get(lang, LANG_HINTS["zh"])
    user = f"【语言与文本类型】\n{hint}\n\n【正文】\n{chunk}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _valid_age_evidence(items, chunk: str) -> tuple[list[dict], list[str]]:
    valid, invalid = [], []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote") or "").strip()
        kind = str(item.get("kind") or "").strip()
        if kind not in ("number", "title") or not quote:
            invalid.append(quote or "<empty>")
            continue
        if locate_verbatim(chunk, quote) < 0:
            invalid.append(quote)
            continue
        valid.append({"quote": quote, "kind": kind})
    return valid, invalid


def _parse_chunk(data: dict, chunk: str) -> dict:
    """把单块 LLM 输出校验成可合并结构。"""
    violations: list[str] = []
    invalid_evidence: list[str] = []
    low_confidence: list[dict] = []
    entities: list[dict] = []
    unresolved: list[dict] = []

    raw_entities = data.get("entities") if isinstance(data, dict) else []
    for raw in raw_entities or []:
        if not isinstance(raw, dict):
            continue
        # 年龄数值/年龄段一律不采纳：由规则层重新推导
        for key in ("age", "age_years", "age_value", "age_band", "age_range"):
            if key in raw:
                raw.pop(key, None)
                violations.append(f"stripped_age_field:{key}")
        violations.extend(strip_forbidden(raw))
        name = str(raw.get("name") or "").strip()
        if not name:
            continue

        aliases = []
        for alias in as_str_list(raw.get("aliases")):
            if alias == name:
                continue
            if locate_verbatim(chunk, alias) >= 0:
                aliases.append(alias)
            else:
                invalid_evidence.append(alias)

        gender = str(raw.get("gender") or "未知").strip()
        if gender not in ("男", "女", "未知"):
            gender = "未知"
        g_valid, g_invalid = split_evidence(as_str_list(raw.get("gender_evidence")), chunk)
        invalid_evidence.extend(g_invalid)
        if gender != "未知" and not g_valid:
            gender = "未知"  # 没有逐字证据就不采纳性别，交给旁白

        age_valid, a_invalid = _valid_age_evidence(raw.get("age_evidence"), chunk)
        invalid_evidence.extend(a_invalid)

        turns = []
        for turn in raw.get("turns") or []:
            if not isinstance(turn, dict):
                continue
            violations.extend(strip_forbidden(turn))
            quote = str(turn.get("quote") or "").strip()
            speaker = str(turn.get("speaker") or "").strip()
            conf = clamp_confidence(turn.get("confidence"))
            ev_valid, ev_invalid = split_evidence(as_str_list(turn.get("evidence")), chunk)
            invalid_evidence.extend(ev_invalid)
            if not quote or not speaker:
                continue
            if locate_verbatim(chunk, quote) < 0:
                invalid_evidence.append(quote[:120])
                continue
            item = {"quote": quote, "speaker": speaker,
                    "confidence": round(conf, 4), "evidence": ev_valid}
            if conf < 0.6:
                low_confidence.append(item)
            else:
                turns.append(item)

        entities.append({
            "name": name,
            "aliases": aliases,
            "gender": gender,
            "gender_evidence": g_valid,
            "age_evidence": age_valid,
            "turns": turns,
        })

    for item in (data.get("unresolved") if isinstance(data, dict) else []) or []:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if quote and locate_verbatim(chunk, quote) >= 0:
            unresolved.append({"quote": quote, "reason": reason})

    return {
        "entities": entities,
        "unresolved": unresolved,
        "low_confidence_turns": low_confidence,
        "invalid_evidence": invalid_evidence,
        "violations": violations,
    }


def _merge_entities(parts: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for part in parts:
        for ent in part.get("entities", []):
            name = ent["name"]
            if name not in merged:
                merged[name] = {
                    "name": name,
                    "aliases": list(ent["aliases"]),
                    "gender": ent["gender"],
                    "gender_evidence": list(ent["gender_evidence"]),
                    "age_evidence": list(ent["age_evidence"]),
                    "turns": list(ent["turns"]),
                    "_gender_votes": {"男": 0, "女": 0},
                }
                if ent["gender"] in ("男", "女"):
                    merged[name]["_gender_votes"][ent["gender"]] += 1
            else:
                base = merged[name]
                for alias in ent["aliases"]:
                    if alias not in base["aliases"]:
                        base["aliases"].append(alias)
                for quote in ent["gender_evidence"]:
                    if quote not in base["gender_evidence"]:
                        base["gender_evidence"].append(quote)
                for ev in ent["age_evidence"]:
                    if ev not in base["age_evidence"]:
                        base["age_evidence"].append(ev)
                if ent["gender"] in ("男", "女"):
                    base["_gender_votes"][ent["gender"]] += 1
                have = {(t["quote"], t["speaker"]) for t in base["turns"]}
                for turn in ent["turns"]:
                    if (turn["quote"], turn["speaker"]) not in have:
                        base["turns"].append(turn)

    # 别名指向正名：若某个实体名是别人的别名，合并过去
    alias_owner: dict[str, str] = {}
    for ent in merged.values():
        for alias in ent["aliases"]:
            alias_owner.setdefault(alias, ent["name"])
    for name in list(merged):
        owner = alias_owner.get(name)
        if owner and owner != name and owner in merged:
            src, dst = merged.pop(name), merged[owner]
            for alias in src["aliases"]:
                if alias not in dst["aliases"]:
                    dst["aliases"].append(alias)
            for quote in src["gender_evidence"]:
                if quote not in dst["gender_evidence"]:
                    dst["gender_evidence"].append(quote)
            for ev in src["age_evidence"]:
                if ev not in dst["age_evidence"]:
                    dst["age_evidence"].append(ev)
            for k in ("男", "女"):
                dst["_gender_votes"][k] += src["_gender_votes"][k]
            have = {(t["quote"], t["speaker"]) for t in dst["turns"]}
            for turn in src["turns"]:
                if (turn["quote"], turn["speaker"]) not in have:
                    dst["turns"].append(turn)
            for turn in dst["turns"]:
                if turn["speaker"] == name:
                    turn["speaker"] = owner

    out = []
    for ent in merged.values():
        votes = ent.pop("_gender_votes")
        if votes["男"] > votes["女"]:
            ent["gender"] = "男"
        elif votes["女"] > votes["男"]:
            ent["gender"] = "女"
        else:
            ent["gender"] = "未知"
        ent["age_band"] = derive_age_band(ent["age_evidence"])
        out.append(ent)
    out.sort(key=lambda e: (-len(e["turns"]), e["name"]))
    return out


def _empty_result(lang: str, status: str, reason: str, usage: Usage,
                  degraded: bool = False, **extra) -> dict:
    out = {
        "task": "document",
        "lang": lang,
        "status": status,
        "degraded": degraded,
        "entities": [],
        "unresolved": [],
        "low_confidence_turns": [],
        "invalid_evidence": [],
        "violations": [],
        "chunks": 0,
        "failures": [],
        "reason": reason,
        "usage": usage.as_dict(),
    }
    out.update(extra)
    return out


def resolve_document(text: str, lang: str = "zh", *, max_chars: int = 6000,
                     client=None, usage: Usage | None = None) -> dict:
    usage = usage or Usage()
    if not text or not text.strip():
        return _empty_result(lang, "invalid", "empty text", usage)
    try:
        llm = client or get_client()
    except Exception as exc:  # noqa: BLE001
        return _empty_result(
            lang, "pending", f"llm unavailable: {type(exc).__name__}: {exc}",
            usage, degraded=True)

    chunks = _chunks(text, max_chars)
    parts: list[dict] = []
    failures: list[str] = []
    for i, chunk in enumerate(chunks):
        data, error = call_json(llm, _messages(chunk, lang),
                                MULTILINGUAL_SCHEMA,
                                f"llm_resolve_document_{lang}", usage)
        if data is None:
            failures.append(f"chunk{i}: {error}")
            continue
        if not isinstance(data, dict):
            failures.append(f"chunk{i}: non-object response")
            continue
        parts.append(_parse_chunk(data, chunk))

    if not parts:
        return _empty_result(lang, "pending", "all chunks failed", usage,
                             degraded=True, chunks=len(chunks),
                             failures=failures)

    entities = _merge_entities(parts)
    unresolved: list[dict] = []
    low: list[dict] = []
    invalid: list[str] = []
    violations: list[str] = []
    for part in parts:
        for item in part["unresolved"]:
            if item not in unresolved:
                unresolved.append(item)
        low.extend(part["low_confidence_turns"])
        invalid.extend(part["invalid_evidence"])
        violations.extend(part["violations"])

    status = "ok" if not failures else "partial"
    return {
        "task": "document",
        "lang": lang,
        "status": status,
        "degraded": bool(failures),
        "entities": entities,
        "unresolved": unresolved,
        "low_confidence_turns": low,
        "invalid_evidence": invalid[:50],
        "violations": violations[:50],
        "chunks": len(chunks),
        "failures": failures,
        "reason": "" if status == "ok" else "some chunks failed",
        "usage": usage.as_dict(),
    }
