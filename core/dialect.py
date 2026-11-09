"""方言/口音声线的证据闸门。

硬约束：方言声线（zh-TW 台湾腔、zh-HK 粤语/港式、东北/陕西口音等）只能用于
“确有方言证据”的角色；没有证据时一律不得使用，即使主池耗尽也不能拿来兜底。

证据来源（写入 identity.dialect_profile，便于复核）：
1. 原文明确写出籍贯/地域/方言特征（keywords 扫描，保留原文片段）；
2. 角色档案 dialect/region 字段（LLM 抽取时必须引用原文片段）；
3. 用户手动指定（source=manual，置信度 1.0）。

证据缺失、原文片段为空或置信度低于 casting.dialect.min_confidence 时，一律按“无证据”处理。
"""
from __future__ import annotations

DEFAULT_KEYWORDS = {
    "台湾腔": {"strong": ["台湾腔", "台湾口音", "台普"], "weak": ["台北", "台湾"]},
    "粤语": {"strong": ["粤语", "广东话", "港式", "香港口音", "岭南口音", "粤人"],
             "weak": ["香港", "广东", "岭南"]},
    "东北": {"strong": ["东北口音", "东北腔", "一口东北话", "关外口音"],
             "weak": ["东北", "关外"]},
    "陕西": {"strong": ["陕西口音", "西北口音", "秦腔", "关中口音"],
             "weak": ["陕西", "关中", "西北"]},
}

STRONG_CONFIDENCE = 0.85
WEAK_CONFIDENCE = 0.50
MANUAL_CONFIDENCE = 1.0


def config(cfg: dict) -> dict:
    return (cfg.get("casting") or {}).get("dialect") or {}


def enabled(cfg: dict) -> bool:
    return bool(config(cfg).get("enabled", True))


def min_confidence(cfg: dict) -> float:
    try:
        return float(config(cfg).get("min_confidence", 0.60))
    except (TypeError, ValueError):
        return 0.60


def keyword_table(cfg: dict) -> dict[str, dict]:
    raw = config(cfg).get("keywords") or DEFAULT_KEYWORDS
    table: dict[str, dict] = {}
    for tag, spec in raw.items():
        if isinstance(spec, dict):
            table[str(tag)] = {
                "strong": [str(item) for item in (spec.get("strong") or [])],
                "weak": [str(item) for item in (spec.get("weak") or [])],
            }
        elif isinstance(spec, list):
            table[str(tag)] = {"strong": [str(item) for item in spec], "weak": []}
    return table or DEFAULT_KEYWORDS


def accented_locales(cfg: dict) -> tuple[str, ...]:
    locales = config(cfg).get("accented_locales") or [
        "zh-TW", "zh-HK", "zh-CN-liaoning", "zh-CN-shaanxi"]
    return tuple(str(item) for item in locales)


def tag_for_locale(locale: str) -> str:
    locale = str(locale or "").lower()
    if locale.startswith("zh-tw"):
        return "台湾腔"
    if locale.startswith("zh-hk"):
        return "粤语"
    if "liaoning" in locale:
        return "东北"
    if "shaanxi" in locale:
        return "陕西"
    return ""


def is_accented_entry(entry: dict, cfg: dict) -> bool:
    """catalog 条目是否属于方言/口音声线。"""
    if not entry:
        return False
    if entry.get("accented") or entry.get("dialect"):
        return True
    locale = str(entry.get("locale") or "").lower()
    return any(locale.startswith(item.lower()) for item in accented_locales(cfg))


def normalize_tag(value: str, cfg: dict) -> str:
    """把自由文本（粤人/岭南口音/台式国语...）归一到四个方言标签之一。"""
    text = str(value or "").strip()
    if not text:
        return ""
    table = keyword_table(cfg)
    if text in table:
        return text
    for tag, spec in table.items():
        for word in spec.get("strong", []) + spec.get("weak", []):
            if word and word in text:
                return tag
    return ""


def _snippet(text: str, start: int, end: int, limit: int = 60) -> str:
    left = max(0, start)
    while left > 0 and text[left - 1] not in "。！？\n；":
        left -= 1
    right = min(len(text), end)
    while right < len(text) and text[right] not in "。！？\n；":
        right += 1
    piece = text[left:right].strip()
    return piece[:limit]


def detect_evidence(text: str, cfg: dict) -> dict | None:
    """从原文/人设文本里扫描方言证据；strong=0.85，weak=0.50（低于阈值视为无证据）。"""
    if not enabled(cfg):
        return None
    body = str(text or "")
    if not body:
        return None
    best: dict | None = None
    for tag, spec in keyword_table(cfg).items():
        for word in spec.get("strong", []):
            index = body.find(word)
            if index >= 0:
                candidate = {
                    "dialect": tag,
                    "accent": tag,
                    "evidence": _snippet(body, index, index + len(word)),
                    "confidence": STRONG_CONFIDENCE,
                    "method": "keyword_scan",
                    "keyword": word,
                }
                if best is None or candidate["confidence"] > best["confidence"]:
                    best = candidate
        for word in spec.get("weak", []):
            index = body.find(word)
            if index >= 0:
                candidate = {
                    "dialect": tag,
                    "accent": tag,
                    "evidence": _snippet(body, index, index + len(word)),
                    "confidence": WEAK_CONFIDENCE,
                    "method": "keyword_scan",
                    "keyword": word,
                }
                if best is None or candidate["confidence"] > best["confidence"]:
                    best = candidate
    return best if best and best["confidence"] >= min_confidence(cfg) else None


def from_persona(persona: dict, cfg: dict) -> dict | None:
    """LLM 人设里的 dialect/region 字段；必须带原文 evidence，置信度不足按无证据。"""
    if not enabled(cfg) or not isinstance(persona, dict):
        return None
    evidence = str(persona.get("dialect_evidence") or "").strip()
    raw_tag = normalize_tag(str(persona.get("dialect") or ""), cfg) \
        or normalize_tag(str(persona.get("region") or ""), cfg)
    if not raw_tag or not evidence:
        return None
    try:
        confidence = float(persona.get("dialect_confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < min_confidence(cfg):
        return None
    return {
        "dialect": raw_tag,
        "accent": raw_tag,
        "region": str(persona.get("region") or "").strip(),
        "evidence": evidence[:120],
        "confidence": confidence,
        "method": "llm_persona",
        "keyword": raw_tag,
    }


def valid(record: dict | None, cfg: dict) -> bool:
    if not enabled(cfg) or not isinstance(record, dict):
        return False
    tag = normalize_tag(str(record.get("dialect") or record.get("accent") or ""), cfg)
    evidence = str(record.get("evidence") or "").strip()
    try:
        confidence = float(record.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return bool(tag and evidence and confidence >= min_confidence(cfg))


def normalize(record: dict | None, cfg: dict) -> dict | None:
    """校验并标准化证据；不合法返回 None（调用方按“无证据”处理）。"""
    if not valid(record, cfg):
        return None
    tag = normalize_tag(str(record.get("dialect") or record.get("accent") or ""), cfg)
    try:
        confidence = float(record.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "dialect": tag,
        "accent": tag,
        "region": str(record.get("region") or "").strip(),
        "evidence": str(record.get("evidence") or "").strip()[:120],
        "confidence": confidence,
        "method": str(record.get("method") or "keyword_scan"),
        "keyword": str(record.get("keyword") or ""),
        "chapter": record.get("chapter"),
    }


def manual_evidence(dialect_tag: str, evidence: str = "WebUI 手动指定方言声线",
                    chapter: int | None = None) -> dict:
    """用户在 UI 手动指定：视为置信度 1.0 的证据，但仍落盘 evidence 以便复核。"""
    return {
        "dialect": str(dialect_tag),
        "accent": str(dialect_tag),
        "region": "",
        "evidence": evidence or "WebUI 手动指定方言声线",
        "confidence": MANUAL_CONFIDENCE,
        "method": "manual",
        "keyword": "",
        "chapter": chapter,
    }
