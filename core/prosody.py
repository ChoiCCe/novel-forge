"""韵律参数合成：Persona 基础 + 情绪 delta。

设计原则：
- 只做数值计算，不碰引擎调用；TTS 路由与 WebUI 试听都复用这里的函数；
- 所有参数先解析成数值再相加，最后统一格式化，避免 "+0%" 与 "-12%" 字符串拼接出错；
- 同一角色同阶段的基础值由 registry 锁死，本模块不负责持久化。
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import yaml

log = logging.getLogger("novel-forge.prosody")

TEMPERAMENTS = ("沉稳", "急躁", "阴柔", "豪爽", "阴沉", "中性")

_TEMPERAMENT_HINTS = (
    ("急躁", ("急躁", "暴躁", "火爆", "性急", "冲动", "易怒", "气盛", "激进",
              "愤慨", "尖锐", "挑衅", "直率", "爽快")),
    ("阴沉", ("阴沉", "阴冷", "阴鸷", "森然", "城府", "狠戾", "冷厉", "狡猾")),
    ("阴柔", ("阴柔", "温婉", "柔婉", "娇柔", "轻柔", "细腻", "缠绵", "软糯")),
    ("豪爽", ("豪爽", "爽朗", "豪迈", "粗犷", "坦率", "开阔")),
    ("沉稳", ("沉稳", "沉着", "冷静", "稳重", "从容", "克制", "老练", "低沉",
              "威严", "理性", "渊博", "恭敬", "温和")),
)

_OPEN_QUOTES = set("“「『‘")
_CLOSE_QUOTES = set("”」』’")
_CLAUSE_PUNCT = set("，,；;：:—–－。！？!?…")
_DIGITS = "零一二三四五六七八九"


# ---------------------------------------------------------------------------- 数值解析
def parse_value(value, kind: str = "%") -> float:
    """把 "+12%" / "-3Hz" / 12 解析成数值；unit 由 kind 决定。"""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    text = text.replace("％", "%").replace("hz", "Hz")
    text = text.replace("%", "").replace("Hz", "").replace("hz", "")
    try:
        return float(text)
    except ValueError:
        return 0.0


def format_value(number: float, kind: str = "%") -> str:
    if kind == "Hz":
        rounded = int(round(float(number)))
        sign = "+" if rounded >= 0 else ""
        return f"{sign}{rounded}Hz"
    rounded = int(round(number))
    sign = "+" if rounded >= 0 else ""
    return f"{sign}{rounded}%"


def _clamp(number: float, bounds) -> float:
    low, high = bounds if isinstance(bounds, (list, tuple)) and len(bounds) == 2 else (-999, 999)
    return max(float(low), min(float(high), float(number)))


def clamp_bounds(kind: str, cfg: dict) -> tuple[float, float]:
    """读取某一维度的硬夹紧范围；缺失即配置错误，不再保留代码内默认值。"""
    pros = cfg["tts"]["prosody"]
    bounds = pros[f"{kind}_clamp"]
    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
        raise KeyError(f"tts.prosody.{kind}_clamp 必须是 [low, high]")
    low, high = float(bounds[0]), float(bounds[1])
    return (min(low, high), max(low, high))


def clamp_value(number: float, kind: str, cfg: dict) -> float:
    low, high = clamp_bounds(kind, cfg)
    return max(low, min(high, float(number)))


def _soft_limit(number: float, cap_pos: float, cap_neg: float | None = None) -> float:
    """软饱和：按正负分别取 cap，避免放宽正向时连带削弱负向情绪。

    v3 用 max(|low|,|high|) 共用 cap；v4 rate_clamp 改为 [-15, +24] 后，
    若仍共用 cap，负向悲伤会被压得更狠或更松，因此必须分开。
    """
    cap = cap_pos if number >= 0 else (cap_pos if cap_neg is None else cap_neg)
    if cap <= 0:
        return 0.0
    return cap * math.tanh(float(number) / cap)


def _add_style(total: dict, style: dict | None, factor: float = 1.0) -> dict:
    style = style or {}
    total["rate"] += parse_value(style.get("rate", "+0%"), "%") * factor
    total["pitch"] += parse_value(style.get("pitch", "+0Hz"), "Hz") * factor
    total["volume"] += parse_value(style.get("volume", "+0%"), "%") * factor
    return total


# ---------------------------------------------------------------------------- Persona 基础
def infer_temperament(text: str) -> str:
    """旧档案没有 temperament 时，从 persona/voice_style 文案里保守推断。"""
    for name, hints in _TEMPERAMENT_HINTS:
        if any(word in (text or "") for word in hints):
            return name
    return "中性"


def persona_base(temperament: str, stage: str, cfg: dict) -> dict:
    """性格映射 + 年龄段偏移，返回 {"rate","pitch","volume"} 字符串。"""
    pros = cfg["tts"].get("prosody", {})
    tmap = pros.get("temperament_map", {})
    amap = pros.get("age_offsets", {})
    total = {"rate": 0.0, "pitch": 0.0, "volume": 0.0}
    _add_style(total, tmap.get(temperament) or tmap.get("中性"))
    _add_style(total, amap.get(stage) or amap.get("青年"))
    return {
        "rate": format_value(total["rate"], "%"),
        "pitch": format_value(total["pitch"], "Hz"),
        "volume": format_value(total["volume"], "%"),
    }


# ---------------------------------------------------------------------------- 情绪 / pace
def spoken_len(text: str) -> int:
    """按“字”计数：标点与空白不算，避免短促句判定被标点撑大。"""
    return sum(1 for ch in (text or "") if ch.isalnum())


def emotion_delta(emotion: str, text_len: int, cfg: dict) -> dict:
    """情绪基础 delta + 短促句/长句修正；返回数值 delta 与命中的修正标签。"""
    pros = cfg["tts"].get("prosody", {})
    style = cfg["tts"].get("emotion_style", {}).get(emotion) or cfg["tts"].get(
        "emotion_style", {}).get("平静", {})
    total = {"rate": 0.0, "pitch": 0.0, "volume": 0.0}
    _add_style(total, style)
    variant = "none"
    short_limit = int(pros["short_sentence_chars"])
    long_limit = int(pros["long_sentence_chars"])
    variants = pros.get("emotion_variants", {})
    if text_len <= short_limit:
        extra = (variants.get("short") or {}).get(emotion)
        if extra:
            _add_style(total, extra)
            variant = "short"
    elif text_len > long_limit:
        extra = (variants.get("long") or {}).get(emotion)
        if extra:
            _add_style(total, extra)
            variant = "long"
    total["variant"] = variant
    return total


def emotion_weight(emotion: str, cfg: dict) -> float:
    """情绪强度权重（0-1）：0 表示该情绪只作为标签、不改变 Persona 基础值。"""
    comp = cfg["tts"]["prosody"]["compose"]
    weights = comp["emotion_weight"]
    return max(0.0, min(1.0, float(weights.get(emotion, comp["emotion_weight_default"]))))


def compose_prosody(profile: dict, emotion: str, text: str, cfg: dict) -> dict:
    """v3 最终韵律：加权融合 + 软饱和 + 硬夹紧。

    raw = persona × w_persona + emotion_delta × w_emotion

    与 v2 的区别：
    - v2 是四层直接相加，周律第 14 段落到 +37%/+12Hz/+12%，听感像喊；
    - v3 中 persona 仍是基础（权重 1.0），但情绪只按强度权重生效（愤怒 0.45），
      再经软饱和压到 (-15%, +15%) 开区间内，最后硬夹紧兜底。
    """
    comp = cfg["tts"].get("prosody", {}).get("compose") or {}
    persona = {
        "rate": parse_value(profile.get("rate", "+0%"), "%"),
        "pitch": parse_value(profile.get("pitch", "+0Hz"), "Hz"),
        "volume": parse_value(profile.get("volume", "+0%"), "%"),
    }
    emo = emotion_delta(emotion, spoken_len(text), cfg)
    persona_w = float(comp["persona_weight"])
    emo_w = emotion_weight(emotion, cfg)

    raw = {
        kind: persona[kind] * persona_w + emo[kind] * emo_w
        for kind in ("rate", "pitch", "volume")
    }
    margin = max(0.0, float(comp["soft_limit_margin"]))
    margin_neg = max(0.0, float(comp["soft_limit_margin_negative"]))
    soft: dict[str, float] = {}
    final: dict[str, float] = {}
    for kind in ("rate", "pitch", "volume"):
        low, high = clamp_bounds(kind, cfg)
        cap_pos = max(0.0, high - margin)
        cap_neg = max(0.0, abs(low) - margin_neg)
        soft[kind] = _soft_limit(raw[kind], cap_pos, cap_neg)
        final[kind] = clamp_value(soft[kind], kind, cfg)

    formatted = {
        kind: format_value(final[kind], "Hz" if kind == "pitch" else "%")
        for kind in ("rate", "pitch", "volume")
    }
    # 只有“显示值”真的被限幅改变时才算 clamped，避免所有值都因 tanh 标成 true
    clamped = {
        kind: format_value(raw[kind], "Hz" if kind == "pitch" else "%") != formatted[kind]
        for kind in ("rate", "pitch", "volume")
    }
    return {
        "formatted": formatted,
        "numeric": final,
        "raw": raw,
        "soft": soft,
        "clamped": clamped,
        "weights": {"persona": persona_w, "emotion": emo_w},
        "persona": persona,
        "emotion_delta": {kind: emo[kind] for kind in ("rate", "pitch", "volume")},
        "emotion_variant": emo["variant"],
    }


def combine_voice(profile: dict, emotion: str, text: str, cfg: dict) -> dict:
    """兼容旧调用：只返回格式化后的 rate/pitch/volume；计算全部走 compose_prosody。"""
    return compose_prosody(profile, emotion, text, cfg)["formatted"]


# ---------------------------------------------------------------------------- 文本规整
def int_to_chinese(number: int) -> str:
    if number < 0:
        return str(number)
    if number < 10:
        return _DIGITS[number]
    if number < 20:
        return "十" + (_DIGITS[number - 10] if number % 10 else "")
    if number < 100:
        tens, unit = divmod(number, 10)
        return _DIGITS[tens] + "十" + (_DIGITS[unit] if unit else "")
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        head = _DIGITS[hundreds] + "百"
        if not rest:
            return head
        if rest < 10:
            return head + "零" + _DIGITS[rest]
        return head + int_to_chinese(rest)
    if number < 10000:
        thousands, rest = divmod(number, 1000)
        head = _DIGITS[thousands] + "千"
        if not rest:
            return head
        if rest < 10:
            return head + "零" + _DIGITS[rest]
        return head + int_to_chinese(rest)
    return str(number)


def _year_to_chinese(digits: str) -> str:
    """年份按位读：1980 -> 一九八零。"""
    return "".join(_DIGITS[int(ch)] for ch in digits) + "年"


def normalize_text(text: str) -> str:
    """只转换年龄与年份的阿拉伯数字，避免误伤对白里的数字。"""
    text = re.sub(r"(\d{2,4})\s*年", lambda m: _year_to_chinese(m.group(1)), text)
    text = re.sub(r"(\d{1,3})\s*岁", lambda m: int_to_chinese(int(m.group(1))) + "岁", text)
    return text


def load_lexicon(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as err:
        log.warning("lexicon.yaml 解析失败，忽略：%s", err)
        return {}
    if isinstance(data, dict) and isinstance(data.get("replacements"), dict):
        data = data["replacements"]
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if key and value}


def apply_lexicon(text: str, lexicon: dict[str, str]) -> str:
    if not lexicon:
        return text
    # 长词优先，避免“重耳”被“重”先替换
    for source in sorted(lexicon, key=len, reverse=True):
        text = text.replace(source, lexicon[source])
    return text


def prepare_text(text: str, cfg: dict, lexicon: dict[str, str] | None = None) -> str:
    """合成专用文本：数字规整 + 专名替换；剧本原文保持不变。"""
    return apply_lexicon(normalize_text(text or ""), lexicon or {})


# ---------------------------------------------------------------------------- 子句切分
def split_clauses(text: str, cfg: dict) -> list[str]:
    """超过长句阈值时按逗号/分号/破折号切子句；引号内部不切。"""
    text = (text or "").strip()
    limit = int(cfg["tts"].get("prosody", {}).get("long_sentence_chars", 25))
    if len(text) <= limit:
        return [text] if text else []
    clauses: list[str] = []
    buffer: list[str] = []
    depth = 0
    for ch in text:
        if ch in _OPEN_QUOTES:
            depth += 1
        elif ch in _CLOSE_QUOTES:
            depth = max(0, depth - 1)
        buffer.append(ch)
        if depth == 0 and ch in _CLAUSE_PUNCT:
            piece = "".join(buffer).strip()
            if piece:
                clauses.append(piece)
            buffer = []
    tail = "".join(buffer).strip()
    if tail:
        clauses.append(tail)
    # 纯标点（如收尾的“…”）edge 会返回 No audio，并到前一个子句；仍无内容则丢弃
    merged: list[str] = []
    for piece in clauses:
        if _has_speech(piece):
            merged.append(piece)
        elif merged:
            merged[-1] += piece
    return merged or ([text] if _has_speech(text) else [])


def _has_speech(text: str) -> bool:
    return any(ch.isalnum() for ch in text or "")


def clause_pause_ms(boundary: int, count: int, cfg: dict) -> int:
    low, high = cfg["tts"].get("prosody", {}).get("clause_pause_ms", [120, 200])
    low, high = int(low), int(high)
    if count <= 2:
        return high
    ratio = max(0.0, min(1.0, (boundary - 1) / max(1, count - 1)))
    return int(round(low + (high - low) * ratio))


def plan_pieces(text: str, cfg: dict) -> list[dict]:
    """把一段文本拆成“最小合成单元”（按子句），返回 [{"text", "pause_before_ms"}]。"""
    clauses = split_clauses(text, cfg)
    pieces: list[dict] = []
    for ci, clause in enumerate(clauses):
        piece_text = clause.strip()
        if not _has_speech(piece_text):
            continue
        pause = 0 if ci == 0 else clause_pause_ms(ci, len(clauses), cfg)
        pieces.append({"text": piece_text, "pause_before_ms": int(pause)})
    return pieces or [{"text": text, "pause_before_ms": 0}]


# ---------------------------------------------------------------------------- 停顿模板
def pause_template(cfg: dict) -> dict:
    pros = cfg["tts"].get("prosody", {})
    template = pros.get("pause_template") or {}
    return {
        "chapter_start_ms": int(template.get("chapter_start_ms", 0)),
        "narration_to_dialogue_ms": int(template.get(
            "narration_to_dialogue_ms", cfg["postprocess"].get("pause_ms_dialogue", 200))),
        "dialogue_to_narration_ms": int(template.get(
            "dialogue_to_narration_ms", cfg["postprocess"].get("pause_ms_narration", 120))),
        "segment_gap_ms": int(template.get("segment_gap_ms", 200)),
        "chapter_end_ms": int(template.get("chapter_end_ms", 0)),
    }


def transition_pause(prev_type: str, cur_type: str, cfg: dict) -> int:
    template = pause_template(cfg)
    if prev_type == "narration" and cur_type == "dialogue":
        return template["narration_to_dialogue_ms"]
    if prev_type == "dialogue" and cur_type == "narration":
        return template["dialogue_to_narration_ms"]
    return template["segment_gap_ms"]
