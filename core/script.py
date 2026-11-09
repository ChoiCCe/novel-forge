"""剧本生成：原文 -> 旁白/对白片段（含说话人、情绪标签），再做一次 LLM 校对。

设计要点：
- 章节内按顺序切块，块间带最近 N 条上下文（N=config.script.context_segments），
  保证“他/她”指代可追踪；
- 不做规则切分降级：LLM 链路失败直接抛 ScriptError，由 main 标记该章节失败；
- 校对只做“合并碎段 + 剥离重复引述标签”，覆盖率明显变差时自动回退。
"""
from __future__ import annotations

import json
import logging
import re

from .coverage import measure
from .segments import SEGMENTS_SCHEMA, Segment

log = logging.getLogger("novel-forge.script")

SPEECH_VERB = r"(?:说|道|问|答|喊|叫|笑|叹|骂|嘀咕|开口|自夸|提醒|解释|喝道|答道|问道|低声|沉声|大声|轻声)"


class ScriptError(RuntimeError):
    """剧本阶段失败（LLM 链路全失败 / 返回空片段 / 覆盖率不达标）。"""

EMOTION_SYNONYMS = {
    "高兴": "喜悦", "开心": "喜悦", "欢喜": "喜悦", "激动": "喜悦",
    "生气": "愤怒", "恼怒": "愤怒", "暴怒": "愤怒",
    "伤心": "悲伤", "悲凉": "悲伤", "哀伤": "悲伤",
    "害怕": "恐惧", "惊恐": "恐惧", "慌张": "恐惧",
    "吃惊": "惊讶", "震惊": "惊讶", "诧异": "惊讶",
    "着急": "焦急", "急切": "焦急", "紧迫": "焦急",
    "嘲讽": "讽刺", "讥讽": "讽刺", "嘲弄": "讽刺",
    "冷淡": "冷漠", "冰冷": "冷漠",
    "温柔": "温和", "轻柔": "温和", "平和": "平静", "普通": "平静",
    "严肃": "威严", "庄重": "威严", "无奈": "无奈",
}


SYSTEM_PROMPT = """你是中文有声小说台本切分器。只输出一个 JSON 对象，不要解释、推理或 markdown。
schema：{"segments":[{"type":"narration|dialogue","speaker":"...","emotion":"...","text":"..."}]}
规则：
1) text 必须逐字引用原文的连续片段；禁止改写、缩写、扩写、调序、增删；原文全部内容都要覆盖。
2) 引号内真正说出口的话独立为 dialogue；引号仅表强调/术语时保留 narration；旁白 speaker=旁白。
3) 说话动词（如“他说/道”）留在旁白；对白 speaker 取角色名单，判断不了填“未知角色”。
4) 每段不超过 120 字，按句末标点切；emotion 只能取给定标签，不要全部填“平静”。
5) 自查：segments 的 text 顺序拼接应与原文一致（标点可略有差异）。
6) 只给语义标签（type/speaker/emotion），禁止输出 rate/pitch/volume 等数字型韵律参数；最终参数由程序映射并夹紧。"""


def normalize_emotion(value: str, allowed: list[str]) -> str:
    value = (value or "").strip()
    for synonym, target in EMOTION_SYNONYMS.items():
        if synonym in value:
            return target if target in allowed else "平静"
    for emotion in allowed:
        if emotion in value:
            return emotion
    return "平静"


def extract_names(text: str, limit: int = 15) -> list[str]:
    """从“XX说/道”里启发式抽角色名，用于首章没有历史档案时的 LLM 提示。"""
    stop = {"他", "她", "我", "它", "众人", "大家", "所有人", "有人", "两人", "自己", "对方", "这人", "那人"}
    counts: dict[str, int] = {}
    for match in re.finditer(rf"([\u4e00-\u9fa5]{{2,4}}){SPEECH_VERB}", text):
        name = match.group(1)
        for prefix in ("向", "对", "朝", "跟", "和", "冲", "又", "也", "便", "就", "才", "却", "更", "还", "都", "正", "在", "刚", "曾"):
            if name.startswith(prefix) and len(name) - len(prefix) >= 2:
                name = name[len(prefix):]
        if name not in stop and len(name) >= 2:
            counts[name] = counts.get(name, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0])))
    return [name for name, _ in ranked[:limit]]


def split_chunks(text: str, limit: int) -> list[str]:
    units: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= limit:
            units.append(paragraph)
            continue
        sentences = re.split(r"(?<=[。！？；…])", paragraph)
        current = ""
        for sentence in sentences:
            if not sentence:
                continue
            if len(sentence) > limit:
                if current:
                    units.append(current)
                    current = ""
                units.extend(sentence[i:i + limit] for i in range(0, len(sentence), limit))
            elif len(current) + len(sentence) <= limit:
                current += sentence
            else:
                units.append(current)
                current = sentence
        if current:
            units.append(current)

    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > limit:
            chunks.append(current)
            current = unit
        else:
            current = f"{current}\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


def strip_outer_quotes(text: str) -> str:
    """对白正文去掉最外层引号，避免 TTS 把引号读成停顿；引号只是标注，不参与发音。"""
    text = text.strip()
    if text[:1] in ("“", "「", "『"):
        text = text[1:]
    for i, ch in enumerate(text):
        if ch in ("”", "」", "』"):
            text = text[:i] + text[i + 1:]
            break
    return text.strip()


def parse_segments(data, allowed: list[str], max_chars: int,
                   narrator_name: str = "旁白") -> list[Segment]:
    if isinstance(data, dict):
        data = data.get("segments", [])
    if not isinstance(data, list):
        return []
    segments: list[Segment] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        raw_type = str(item.get("type", "")).strip().lower()
        # GLM 等模型偶尔把 schema 值写成中文，做同义词归一，避免整章被误判为旁白
        kind = "dialogue" if raw_type in ("dialogue", "对白", "对话", "台词") else "narration"
        if raw_type and raw_type not in ("dialogue", "对白", "对话", "台词", "narration", "旁白", "叙述"):
            log.warning("未知 type=%r，按旁白处理", raw_type)
        speaker = str(item.get("speaker") or ("未知角色" if kind == "dialogue" else narrator_name)).strip()
        emotion = normalize_emotion(str(item.get("emotion", "")), allowed)
        if kind == "dialogue":
            text = strip_outer_quotes(text)
        segments.extend(split_long(
            Segment(kind, speaker if kind == "dialogue" else narrator_name, emotion, text), max_chars))
    return segments


def enforce_narrator_speakers(segments: list[Segment], narrator_name: str = "旁白",
                              mode: str = "fix") -> tuple[list[Segment], dict]:
    """写盘前语料级校验：type=narration 的 speaker 必须等于配置旁白名。

    mode=fix 就地改写并返回证据；mode=error 发现违例直接抛 ScriptError。
    返回 (segments, report)，report 含 checked/fixed/items，供日志与剧本快照留档。
    """
    items: list[dict] = []
    checked = 0
    for index, seg in enumerate(segments):
        if seg.type != "narration":
            continue
        checked += 1
        if seg.speaker == narrator_name:
            continue
        items.append({"index": index, "from": seg.speaker, "text": seg.text[:60]})
        if mode == "fix":
            seg.speaker = narrator_name
    report = {"narrator": narrator_name, "mode": mode,
              "checked": checked, "fixed": len(items), "items": items[:20]}
    if items and mode == "error":
        detail = "；".join(f"#{item['index']} speaker={item['from']!r}" for item in items[:5])
        raise ScriptError(f"旁白段 speaker 不等于“{narrator_name}”：{len(items)} 处（{detail}）")
    return segments, report


def _chunk_prompt(chunk: str, index: int, total: int, roster: list[str], context: list[Segment],
                  emotions: list[str], context_limit: int) -> str:
    context_json = json.dumps([s.to_dict() for s in context[-context_limit:]], ensure_ascii=False) if context else "[]"
    return (
        f"可用情绪标签：{'|'.join(emotions)}\n"
        f"角色名单：{'、'.join(roster) or '（暂无，按原文推断）'}\n"
        f"最近台本上下文（只作参考，不要重复输出）：{context_json}\n\n"
        f"原文第 {index}/{total} 块（请完整切分）：\n{chunk}"
    )


def _call_chunk(chunk: str, index: int, total: int, roster: list[str], context: list[Segment], llm, cfg: dict) -> list[Segment]:
    """单块切分：只调用一次模型链，失败直接抛错，不产出规则切分的低质量剧本。

    模型链内部的“单模型 JSON 重试 / 换下一个模型”由 LLMClient.chat_json 负责。
    """
    allowed = cfg["script"]["emotions"]
    max_chars = int(cfg["script"].get("max_segment_chars", 150))
    narrator_name = str(cfg["script"].get("narrator_name", "旁白"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _chunk_prompt(
            chunk, index, total, roster, context, allowed,
            int(cfg["script"]["context_segments"]))},
    ]

    def accept(data) -> bool:
        # 空片段视为失败，触发同模型纠错与下一个模型，而不是直接判整章失败
        return bool(parse_segments(data, allowed, max_chars, narrator_name))

    try:
        data = llm.chat_json(messages, accept=accept, schema=SEGMENTS_SCHEMA,
                             schema_name="segments")
    except Exception as err:
        raise ScriptError(f"块 {index}/{total} LLM 链路失败：{err}") from err
    segments = parse_segments(data, allowed, max_chars, narrator_name)
    if not segments:
        raise ScriptError(f"块 {index}/{total} 返回空片段，拒绝产出空剧本")
    report = measure(segments, chunk)
    log.info("块 %s/%s：%s 段，覆盖率 %.1f%%，精确率 %.1f%%",
             index, total, len(segments), report.ratio * 100, report.precision * 100)
    if report.ratio < float(cfg["script"]["quality"]["low_coverage_warn"]):
        log.warning("块 %s/%s 覆盖率偏低（%.1f%%），交给章节级补写", index, total, report.ratio * 100)
    return segments


def chapter_to_segments(text: str, llm, cfg: dict, known_names: list[str] | None = None) -> list[Segment]:
    limit = int(cfg["script"].get("chunk_chars", 1200))
    chunks = split_chunks(text, limit)
    roster = list(dict.fromkeys((known_names or []) + extract_names(text)))
    segments: list[Segment] = []
    for index, chunk in enumerate(chunks, 1):
        try:
            chunk_segments = _call_chunk(chunk, index, len(chunks), roster, segments, llm, cfg)
        except ScriptError as err:
            # 保留该章已完成的块，供失败报告与断点续跑排查
            err.partial_segments = list(segments)
            raise
        segments.extend(chunk_segments)
        for seg in chunk_segments:
            if seg.type == "dialogue" and seg.speaker not in ("未知角色", "旁白") and seg.speaker not in roster:
                roster.append(seg.speaker)
    return merge_adjacent(segments, int(cfg["script"].get("max_segment_chars", 150)))


def merge_adjacent(segments: list[Segment], limit: int | None = None) -> list[Segment]:
    """合并相邻同角色同情绪碎段；传入 limit 时保证合并后不超过单段上限。"""
    merged: list[Segment] = []
    for seg in segments:
        if (merged and merged[-1].type == seg.type and merged[-1].speaker == seg.speaker
                and merged[-1].emotion == seg.emotion and not seg.repaired and not merged[-1].repaired
                and (limit is None or len(merged[-1].text) + len(seg.text) <= limit)):
            merged[-1].text += seg.text
        else:
            merged.append(Segment(seg.type, seg.speaker, seg.emotion, seg.text, seg.repaired))
    return merged


def split_long(seg: Segment, limit: int) -> list[Segment]:
    if len(seg.text) <= limit:
        return [seg]
    parts = [p for p in re.split(r"(?<=[。！？；…])", seg.text) if p]
    if not parts:
        return [seg]
    output: list[Segment] = []
    current = ""
    for part in parts:
        if len(current) + len(part) <= limit:
            current += part
        else:
            if current:
                output.append(Segment(seg.type, seg.speaker, seg.emotion, current, seg.repaired))
            while len(part) > limit:
                output.append(Segment(seg.type, seg.speaker, seg.emotion, part[:limit], seg.repaired))
                part = part[limit:]
            current = part
    if current:
        output.append(Segment(seg.type, seg.speaker, seg.emotion, current, seg.repaired))
    return output or [seg]


PROOF_SYSTEM = """你是台本校对器。只输出一个 JSON 对象，不要解释或推理。
只做两件事：1) 合并同说话人、同情绪、相邻且语义连续的碎段；2) 删除旁白里与紧邻对白重复的“XX说/道/问/喊”标签。
禁止改写、扩写、缩写、调序或新增其他内容；不得删除对白、描写或信息。
schema：{"segments":[{"type":"narration|dialogue","speaker":"...","emotion":"...","text":"..."}]}"""


def proofread(segments: list[Segment], source: str, llm, cfg: dict) -> tuple[list[Segment], bool]:
    """返回 (片段, 是否采纳校对)。覆盖率和精确率同时变差则回退。"""
    if not segments:
        return segments, False
    before = measure(segments, source)
    payload = json.dumps([s.to_dict() for s in segments], ensure_ascii=False)
    if len(payload) > int(cfg["script"]["proofread_payload_max_chars"]):  # 超长章节跳过，避免上下文溢出
        log.info("章节台本过长（%s 字符），跳过 LLM 校对", len(payload))
        return segments, False
    allowed = cfg["script"]["emotions"]
    max_chars = int(cfg["script"].get("max_segment_chars", 150))
    narrator_name = str(cfg["script"].get("narrator_name", "旁白"))
    try:
        data = llm.chat_json([
            {"role": "system", "content": PROOF_SYSTEM},
            {"role": "user", "content": payload},
        ], prefer=cfg["llm"].get("review_model") or None,
            schema=SEGMENTS_SCHEMA, schema_name="segments")
        candidate = parse_segments(data, allowed, max_chars, narrator_name)
    except Exception as err:
        log.warning("台本校对失败，保留原结果：%s", err)
        return segments, False
    if not candidate:
        return segments, False
    after = measure(candidate, source)
    before_dlg = sum(len(s.text) for s in segments if s.type == "dialogue")
    after_dlg = sum(len(s.text) for s in candidate if s.type == "dialogue")
    # 校对只能合并/删标签，不能把对白吞成旁白（GLM 曾出现整章对白比例归零）
    quality = cfg["script"]["quality"]
    keeps_dialogue = before_dlg == 0 or after_dlg >= before_dlg * float(quality["proofread_keep_dialogue"])
    accepted = (after.ratio >= before.ratio - float(quality["proofread_ratio_slack"])
                and after.precision >= float(quality["proofread_precision_min"])
                and keeps_dialogue)
    log.info("校对：覆盖率 %.2f%% -> %.2f%%，精确率 %.2f%% -> %.2f%%，对白字符 %s -> %s（%s）",
             before.ratio * 100, after.ratio * 100, before.precision * 100, after.precision * 100,
             before_dlg, after_dlg, "采纳" if accepted else "回退")
    return (candidate, True) if accepted else (segments, False)
