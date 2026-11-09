"""LLM 导演层：每章独立调用一次，给每个 segment 输出表演指令。

输出只包含四项：pace / emphasis / pause_after_ms / volume。
失败策略：缺失 index 用默认值兜底；JSON 解析连续失败两次则整章退回纯 A 参数，
由调用方把 director 状态写成 fallback 并进入报告。
"""
from __future__ import annotations

import logging
import re

from .prosody import PACES, VOLUMES, default_direction, spoken_len

log = logging.getLogger("novel-forge.director")

DIRECTOR_SYSTEM = """你是中文有声书的表演导演。只输出一个 JSON 对象，不要解释、推理或 markdown。
你的任务：为每个片段决定表演节奏、重读词、停顿与音量。
规则：
1) 必须覆盖输入中出现的每一个 index，不能遗漏；index 只用于对齐，不要改动文本。
2) pace：slow | normal | fast | burst。紧张/冲突用 fast 或 burst，抒情/沉重用 slow。
3) emphasis：需要重读的词或短语数组。原文没有明确该重读的词就返回空数组 []；
   禁止推测、改写或补写原文里没有的词；每段最多 3 个，每个不超过 6 字。
4) pause_after_ms：该段结束后的额外停顿，0-1200 的整数；只在情绪转折、重要揭示处给 200-800，普通叙述用 0。
5) volume：soft | normal | loud；不要整章一律 loud。
6) 旁白以叙事清晰为主；对白要逐条判断，情绪明显的对白至少给 pace 或 volume 一个非 normal 值。
7) emphasis 只写原文里逐字出现的词或短语，不要带标点；系统会做子串校验，对不上会被丢弃。
8) 只给方向（pace/volume/emphasis/pause），禁止输出 rate/pitch/volume 百分比或 Hz 数值；最终参数由程序映射、加权并夹紧。"""


DIRECTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "directions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "pace": {"type": "string", "enum": list(PACES)},
                    "emphasis": {"type": "array", "items": {"type": "string"}},
                    "pause_after_ms": {"type": "integer"},
                    "volume": {"type": "string", "enum": list(VOLUMES)},
                },
                "required": ["index"],
            },
        }
    },
    "required": ["directions"],
}


def _prompt(chapter_title: str, segments: list, indexes: list[int] | None = None) -> str:
    """indexes 为原始 1-based index；门控后只把有线索的片段送审。"""
    indexed = [(index, seg) for index, seg in enumerate(segments, 1)
               if indexes is None or index in indexes]
    count = len(indexed)
    lines = [
        f"章节：{chapter_title}",
        f"本次只送审 {count} 个片段。directions 数组只需覆盖下列 index，"
        "不得输出未列出的 index；没有明显线索的片段系统会直接用默认值。",
        "片段列表（index|type|speaker|emotion|text）：",
    ]
    for index, seg in indexed:
        kind = "narration" if getattr(seg, "type", "") == "narration" else "dialogue"
        speaker = getattr(seg, "speaker", "旁白")
        emotion = getattr(seg, "emotion", "平静")
        text = str(getattr(seg, "text", "")).replace("\n", " ")
        lines.append(f"{index}|{kind}|{speaker}|{emotion}|{text}")
    lines.append(
        '输出格式：{"directions":['
        '{"index":1,"pace":"slow|normal|fast|burst","emphasis":[],'
        '"pause_after_ms":0,"volume":"soft|normal|loud"}, ...]}')
    return "\n".join(lines)


def normalize_item(item, max_pause: int) -> dict | None:
    if not isinstance(item, dict):
        return None
    try:
        index = int(item.get("index"))
    except (TypeError, ValueError):
        return None
    if index <= 0:
        return None
    pace = str(item.get("pace") or "normal").strip().lower()
    if pace not in PACES:
        pace = "normal"
    volume = str(item.get("volume") or "normal").strip().lower()
    if volume not in VOLUMES:
        volume = "normal"
    try:
        pause = int(float(item.get("pause_after_ms") or 0))
    except (TypeError, ValueError):
        pause = 0
    pause = max(0, min(int(max_pause), pause))
    raw = item.get("emphasis")
    if isinstance(raw, str):
        raw = [raw]
    emphasis: list[str] = []
    for word in raw if isinstance(raw, list) else []:
        word = str(word).strip().strip("，。！？；：、…—\"“”「」『』")
        if word and len(word) <= 12 and word not in emphasis:
            emphasis.append(word)
        if len(emphasis) >= 3:
            break
    return {"pace": pace, "emphasis": emphasis, "pause_after_ms": pause, "volume": volume}


def _norm_for_match(text: str) -> str:
    """去掉标点/空白/符号后用于 emphasis 子串校验；保留中英文与数字。"""
    return re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", str(text or ""))


def validate_emphasis(items: list[dict], segments: list) -> tuple[list[dict], list[dict]]:
    """强调词硬校验：去掉标点后必须是该段原文的子串，否则丢弃并记录。"""
    dropped: list[dict] = []
    for index, item in enumerate(items):
        if index >= len(segments):
            break
        text = str(getattr(segments[index], "text", ""))
        norm_text = _norm_for_match(text)
        kept: list[str] = []
        for word in list(item.get("emphasis") or []):
            norm_word = _norm_for_match(word)
            if norm_word and norm_word in norm_text:
                kept.append(word)
            else:
                dropped.append({
                    "index": index + 1,
                    "word": str(word),
                    "text": text[:60],
                })
        item["emphasis"] = kept
    return items, dropped


def _gate_conf(cfg: dict) -> dict:
    return ((cfg.get("script", {}) or {}).get("director", {}) or {}).get("gating", {}) or {}


def _emotion_intensity(emotion: str, gate: dict) -> float:
    table = gate.get("emotion_intensity") or {}
    try:
        return float(table.get(emotion, 0.0))
    except (TypeError, ValueError):
        return 0.0


def needs_director(seg, cfg: dict) -> bool:
    """只送有明显表演线索的段落：省 token，避免模型对平淡旁白硬编演出。"""
    gate = _gate_conf(cfg)
    if not gate.get("enabled", True):
        return True
    emotion = str(getattr(seg, "emotion", "") or "平静")
    neutral = set(gate.get("neutral_emotions") or ["平静", "默认"])
    if emotion not in neutral:
        return True
    text = str(getattr(seg, "text", "") or "")
    punct = str(gate.get("punct_chars") or "！？!?…")
    if any(ch in text for ch in punct):
        return True
    short_chars = int(gate["short_sentence_chars"])
    if spoken_len(text) <= short_chars:
        return True
    if str(getattr(seg, "type", "")) == "dialogue":
        threshold = float(gate["dialogue_intensity_min"])
        if _emotion_intensity(emotion, gate) >= threshold:
            return True
    return False


def eligible_indexes(segments: list, cfg: dict) -> list[int]:
    return [index for index, seg in enumerate(segments, 1) if needs_director(seg, cfg)]


def estimate(segments: list, cfg: dict) -> dict:
    """WebUI / 报告用：预计送审段数、总段数、默认兜底段数。"""
    total = len(segments)
    sent = len(eligible_indexes(segments, cfg))
    return {"sent": sent, "total": total, "default_count": total - sent}


def _items_from_payload(data, count: int, max_pause: int,
                        allowed: set[int] | None = None) -> tuple[list[dict], set[int]]:
    if isinstance(data, dict):
        raw = data.get("directions") or data.get("items")
        if raw is None:
            # 兼容 {"1": {...}, "2": {...}} 这种按 index 的对象
            numeric = [key for key in data if str(key).strip().isdigit()]
            raw = {key: data[key] for key in numeric} if numeric else []
        if isinstance(raw, dict):
            raw = [dict(value, index=key) if isinstance(value, dict) else {"index": key}
                   for key, value in raw.items()]
    else:
        raw = data
    items = [default_direction() for _ in range(count)]
    covered: set[int] = set()
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if index <= 0 or index > count:
            continue
        if allowed is not None and index not in allowed:
            continue
        item = normalize_item(entry, max_pause)
        if item is None:
            continue
        items[index - 1] = item
        covered.add(index)
    return items, covered


def run_director(segments: list, llm, cfg: dict, chapter_title: str, enabled: bool | None = None) -> dict:
    """返回导演结果 + 门控统计 + 被丢弃的强调词。"""
    conf = cfg.get("script", {}).get("director", {})
    if enabled is None:
        enabled = bool(conf.get("enabled", True))
    count = len(segments)
    base = {"sent": 0, "total": count, "default_count": count,
            "eligible": [], "dropped_emphasis": []}
    if not enabled:
        return {"enabled": False, "status": "disabled", "items": [default_direction() for _ in range(count)],
                "error": "", "attempts": 0, "covered": 0, **base}
    if count == 0:
        return {"enabled": True, "status": "ok", "items": [], "error": "", "attempts": 0,
                "covered": 0, **base}

    eligible = eligible_indexes(segments, cfg)
    sent = len(eligible)
    base.update({"sent": sent, "default_count": count - sent, "eligible": eligible})
    if sent == 0:
        log.info("[%s] 导演层：门控后 0/%s 段有表演线索，跳过 LLM 调用", chapter_title, count)
        return {"enabled": True, "status": "skipped", "items": [default_direction() for _ in range(count)],
                "error": "", "attempts": 0, "covered": 0, **base}
    if llm is None:
        return {"enabled": True, "status": "fallback", "items": [default_direction() for _ in range(count)],
                "error": "LLM 不可用", "attempts": 0, "covered": 0, **base}

    max_pause = int(conf["pause_after_max_ms"])
    prompt = _prompt(chapter_title, segments, indexes=eligible)
    allowed = set(eligible)
    last_error = ""
    for attempt in (1, 2):
        try:
            data = llm.chat_json([
                {"role": "system", "content": DIRECTOR_SYSTEM},
                {"role": "user", "content": prompt},
            ], schema=DIRECTIONS_SCHEMA, schema_name="directions")
            items, covered = _items_from_payload(data, count, max_pause, allowed=allowed)
            items, dropped = validate_emphasis(items, segments)
            missing = sent - len(covered)
            status = "ok" if missing == 0 else "partial"
            log.info("[%s] 导演层：门控送审 %s/%s（默认 %s），覆盖 %s/%s（%s），丢弃强调词 %s",
                     chapter_title, sent, count, count - sent, len(covered), sent, status, len(dropped))
            return {
                "enabled": True, "status": status, "items": items,
                "error": "" if missing == 0 else f"{missing} 个送审 index 缺失，已用默认值兜底",
                "attempts": attempt, "covered": len(covered),
                "sent": sent, "total": count, "default_count": count - sent,
                "eligible": eligible, "dropped_emphasis": dropped,
            }
        except Exception as err:
            last_error = f"{type(err).__name__}: {err}"
            log.warning("[%s] 导演层第 %s 次失败：%s", chapter_title, attempt, err)
    log.error("[%s] 导演层两次失败，整章退回纯参数模式（director=fallback）：%s", chapter_title, last_error)
    return {"enabled": True, "status": "fallback", "items": [default_direction() for _ in range(count)],
            "error": last_error, "attempts": 2, "covered": 0, **base}


def from_record(record: dict, count: int, *, max_pause: int) -> list[dict]:
    """从剧本快照取导演结果；没有时返回全默认，保证 tasks 一定能构建。"""
    block = (record or {}).get("director") or {}
    items = block.get("items") or []
    output = [default_direction() for _ in range(count)]
    for index, item in enumerate(items[:count]):
        if isinstance(item, dict):
            output[index] = {
                "pace": item.get("pace") if item.get("pace") in PACES else "normal",
                "emphasis": list(item.get("emphasis") or []),
                "pause_after_ms": max(0, min(int(max_pause), int(item.get("pause_after_ms") or 0))),
                "volume": item.get("volume") if item.get("volume") in VOLUMES else "normal",
            }
    return output


def sample_for_report(segments: list, items: list[dict], limit: int = 4) -> list[dict]:
    """给报告用的导演输出样例：挑带 pace/emphasis/pause 的片段，不足再补普通片段。"""
    picked: list[dict] = []
    for index, (seg, item) in enumerate(zip(segments, items), 1):
        interesting = (item.get("pace") not in (None, "normal")
                       or item.get("emphasis") or item.get("pause_after_ms")
                       or item.get("volume") not in (None, "normal"))
        if interesting:
            picked.append({
                "index": index, "type": getattr(seg, "type", ""),
                "speaker": getattr(seg, "speaker", ""), "emotion": getattr(seg, "emotion", ""),
                "text": str(getattr(seg, "text", ""))[:40], **item,
            })
        if len(picked) >= limit:
            break
    if len(picked) < limit:
        for index, (seg, item) in enumerate(zip(segments, items), 1):
            if any(row["index"] == index for row in picked):
                continue
            picked.append({
                "index": index, "type": getattr(seg, "type", ""),
                "speaker": getattr(seg, "speaker", ""), "emotion": getattr(seg, "emotion", ""),
                "text": str(getattr(seg, "text", ""))[:40], **item,
            })
            if len(picked) >= limit:
                break
    return picked
