"""LLM 声线事件：语义判定层（LLM）+ 确定性落地层（代码）。

分工：
- LLM 只负责“这一章里谁的声音该变、为什么变、证据是哪句”，输出严格 JSON；
- 代码负责校验、置信度闸门、去重、写入 `registry.voice_events`、把交换类事件
  落成 binding 覆盖（arc/persistent 才写 binding，scene/chapter 只在合成时覆盖）。

硬约束：
1. LLM 永远不直接改绑定；所有绑定变更都由 `apply_events()` 做，且可 `revert_event()`。
2. 事件必须带 `evidence_quote`，且该句必须能在本章原文中找到（空白归一后包含匹配），
   否则整条丢弃（`evidence_mismatch`），防止模型编造。
3. 默认 `enabled: false`，关闭时抽取与落地都不执行。
4. 同一章重跑按 `event_key` 幂等：已存在的事件不会重复写入，也不会抖动覆盖。
"""
from __future__ import annotations

import hashlib
import json
import re
import time

from . import prosody

# ---------------------------------------------------------------- 分类与策略

TYPE_DISGUISE = "disguise"
TYPE_CAMOUFLAGE = "voice_camouflage"
TYPE_SOFT_MOD = "soft_mod"
TYPE_GENDER_DISGUISE = "gender_disguise"
TYPE_INJURY = "injury_voice"
TYPE_POSSESSION = "possession"
TYPE_TRANSFORMATION = "transformation"
TYPE_IMPERSONATION = "impersonation"
TYPE_FLASHBACK = "flashback"
TYPE_CHORUS = "chorus"
TYPE_TWIN = "twin_duplicate"
TYPE_NONHUMAN = "nonhuman_voice"
TYPE_MUTE = "mute"
TYPE_RESTORE = "restore"
TYPE_BASELINE = "voice_baseline"

# voice_mode 默认值 / 默认作用域；swap 类的事件在类型层面就允许换声线。
STRATEGY = {
    TYPE_DISGUISE: {"mode": "same", "scope": "arc", "swap": True},
    TYPE_CAMOUFLAGE: {"mode": "soft_mod", "scope": "chapter", "swap": False},
    TYPE_SOFT_MOD: {"mode": "soft_mod", "scope": "scene", "swap": False},
    TYPE_GENDER_DISGUISE: {"mode": "swap_to", "scope": "arc", "swap": True},
    TYPE_INJURY: {"mode": "soft_mod", "scope": "arc", "swap": False},
    TYPE_POSSESSION: {"mode": "swap_to", "scope": "arc", "swap": True},
    TYPE_TRANSFORMATION: {"mode": "swap_to", "scope": "scene", "swap": True},
    TYPE_IMPERSONATION: {"mode": "soft_mod", "scope": "scene", "swap": False},
    TYPE_FLASHBACK: {"mode": "soft_mod", "scope": "scene", "swap": False},
    TYPE_CHORUS: {"mode": "soft_mod", "scope": "scene", "swap": False},
    TYPE_TWIN: {"mode": "swap_to", "scope": "arc", "swap": True},
    TYPE_NONHUMAN: {"mode": "soft_mod", "scope": "arc", "swap": False},
    TYPE_MUTE: {"mode": "same", "scope": "arc", "swap": False},
    TYPE_RESTORE: {"mode": "same", "scope": "arc", "swap": False},
    TYPE_BASELINE: {"mode": "same", "scope": "arc", "swap": False},
}

VOICE_MODES = ("same", "soft_mod", "swap_to")
SCOPES = ("scene", "chapter", "arc")

# 同一章内多个 swap 事件并存时的确定性优先级（数字大者优先）。
SWAP_PRIORITY = {
    TYPE_POSSESSION: 90,
    TYPE_TWIN: 80,
    TYPE_TRANSFORMATION: 70,
    TYPE_GENDER_DISGUISE: 60,
    TYPE_INJURY: 50,
    TYPE_DISGUISE: 40,
}

DEFAULT_CONFIG = {
    "enabled": False,
    "auto_apply_confidence": 0.85,
    "pending_review_confidence": 0.55,
    "chunk_chars": 1600,
    "max_segments_per_chunk": 40,
    "prior_events_chapters": 3,
    "max_events_per_chapter": 12,
    "max_events_per_character": 1,
    "allow_swap_to": True,
    "allow_soft_mod": True,
    "allow_arc_scope": True,
    "allow_gender_swap": True,
    "prompt_version": "e1",
    "params": {"max_rate_pct": 25, "max_pitch_hz": 12, "max_volume_pct": 15},
}

# ---------------------------------------------------------------- JSON Schema

_NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "integer"}]}

EVENTS_SCHEMA = {
    "type": "object",
    "required": ["events"],
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "character_id", "voice_mode", "scope",
                             "start_chapter", "confidence", "evidence_quote"],
                "properties": {
                    "type": {"enum": list(STRATEGY.keys())},
                    "character_id": {"type": "string"},
                    "voice_mode": {"enum": list(VOICE_MODES)},
                    "target_voice_owner": {"type": "string"},
                    "scope": {"enum": list(SCOPES)},
                    "start_chapter": {"type": "integer"},
                    "end_chapter": {"type": "integer"},
                    "persistent": {"type": "boolean"},
                    "params": {
                        "type": "object",
                        "properties": {
                            "rate_pct": {"type": "number"},
                            "pitch_hz": {"type": "number"},
                            "volume_pct": {"type": "number"},
                        },
                    },
                    "confidence": {"type": "number"},
                    "evidence_quote": {"type": "string"},
                    "disguised_as": {"type": "string"},
                    "reveal_policy": {"enum": ["reader", "concealed"]},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}

# ---------------------------------------------------------------- 配置读取


def config(cfg: dict) -> dict:
    raw = ((cfg.get("casting") or {}).get("voice_rules") or {}).get("voice_events") or {}
    merged = dict(DEFAULT_CONFIG)
    for key, default in DEFAULT_CONFIG.items():
        if key == "params":
            merged[key] = {**default, **(raw.get("params") or {})}
        else:
            merged[key] = raw.get(key, default)
    return merged


def enabled(cfg: dict) -> bool:
    return bool(config(cfg)["enabled"])


# ---------------------------------------------------------------- 规范化


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def event_key(event: dict) -> str:
    material = "|".join(str(event.get(key) or "") for key in (
        "book_id", "start_chapter", "type", "character_id", "target_voice_owner",
        "voice_mode", "scope", "end_chapter", "evidence_quote"))
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, float(value or 0)))


# 证据线索 -> 默认韵律偏移。只在 LLM 没给 params 时兜底，保证“判定了类型就一定可落地”。
# 顺序即优先级：先命中更具体的动作。
_CUE_PARAMS = (
    (("耳语", "低声", "小声", "压低", "窃窃", "低语"), (0, -8, -8)),
    (("尖叫", "吼", "怒吼", "呐喊", "大叫", "大声", "高声", "嚷", "喊道", "叫道"),
     (12, 2, 8)),
    (("哭", "哽咽", "抽泣", "泣", "泪"), (-8, 3, -4)),
    (("笑道", "笑着说", "笑出声", "嘿嘿", "哈哈", "嘻嘻", "噗嗤"), (8, 3, 2)),
    (("不耐", "怒道", "骂", "怒喝", "生气", "喝斥"), (8, 2, 4)),
    (("醉", "酒意", "醉熏", "醉醺", "酒后"), (4, 1, 2)),
    (("虚弱", "受伤", "疼痛", "无力", "重伤", "喘息"), (-10, -2, -6)),
    (("颤声", "颤抖着", "声音颤", "哆嗦"), (-3, 2, -4)),
    (("沙哑", "嘶哑", "哑着", "嗓子哑", "嘶"), (-4, -3, -3)),
    (("冷冷", "阴沉", "沉声", "冷淡"), (-6, -2, -2)),
    (("轻柔", "柔声", "轻声", "温和", "恭顺", "细声"), (-4, 1, -2)),
    (("模仿", "学腔", "腔调", "口音"), (2, 2, 0)),
    (("回忆", "当年", "小时候", "多年前", "往事", "梦中", "想起", "那时"),
     (-3, 2, -1)),
)

_FLASHBACK_MARKERS = ("回忆", "当年", "小时候", "多年前", "往事", "梦中", "想起",
                      "记得", "那时", "幼时", "过去")

# 只有在原文明确描写“声音/说话方式”时才允许参数级事件；面部表情、情绪词不算。
_AUDIBLE_CUES = (
    "声音", "嗓音", "语声", "声调", "语调", "语气", "口音", "腔调",
    "低声", "高声", "大声", "小声", "沉声", "尖声", "尖叫", "吼", "嚷",
    "耳语", "低语", "喃喃", "沙哑", "嘶哑", "哑着", "颤声", "颤抖着",
    "哽咽", "哭腔", "哭道", "哭诉", "笑道", "笑着说", "笑出声", "嘿嘿",
    "哈哈", "嘻嘻", "噗嗤", "醉熏", "醉醺", "酒意", "虚弱", "气息",
    "嗓子", "喉咙", "喊道", "叫道", "喝道", "叹道", "声嘶", "轻柔",
    "柔声", "轻声", "细声", "恭顺", "不耐", "怒道", "骂道", "怒喝",
)

# LLM 给的数值超过这个“合理听感包络”时，不用它，改用证据线索的默认值。
_PARAM_ENVELOPE = (12.0, 6.0, 8.0)


def _audible(text: str) -> bool:
    body = str(text or "")
    return any(cue in body for cue in _AUDIBLE_CUES)


def _is_narration_evidence(text: str) -> bool:
    """证据若以引号开头，基本是台词本身，不能用来判断“声音怎么变”。"""
    body = str(text or "").lstrip()
    return bool(body) and body[0] not in "“‘\"「『'‘"


def _cue_params(text: str) -> tuple[float, float, float] | None:
    body = str(text or "")
    for words, params in _CUE_PARAMS:
        if any(word in body for word in words):
            return params
    return None


def normalize_event(raw: dict, *, chapter: int, book_id: str, text_body: str,
                    identities: dict, conf: dict) -> tuple[dict | None, str]:
    """把 LLM 原始事件规范成可落地事件；返回 (event, error)。"""
    if not isinstance(raw, dict):
        return None, "invalid_event"
    kind = str(raw.get("type") or "")
    if kind not in STRATEGY:
        return None, f"unknown_type:{kind}"
    cid = str(raw.get("character_id") or "").strip()
    if cid not in identities:
        # LLM 常写名字而不是 id：按 canonical_name / 别名回查
        resolved = ""
        for key, ident in identities.items():
            if ident.get("canonical_name") == cid:
                resolved = key
                break
        if resolved:
            cid = resolved
    if not cid or cid not in identities:
        return None, "unknown_character"
    evidence = str(raw.get("evidence_quote") or "").strip()
    if not evidence:
        return None, "missing_evidence"
    if _norm_text(evidence) not in _norm_text(text_body):
        return None, "evidence_mismatch"

    strategy = STRATEGY[kind]
    mode = str(raw.get("voice_mode") or strategy["mode"])
    if mode not in VOICE_MODES:
        return None, f"unknown_voice_mode:{mode}"
    if mode == "swap_to" and not conf["allow_swap_to"]:
        return None, "swap_to_disabled"
    if mode == "soft_mod" and not conf["allow_soft_mod"]:
        return None, "soft_mod_disabled"

    scope = str(raw.get("scope") or strategy["scope"])
    if scope not in SCOPES:
        return None, f"unknown_scope:{scope}"
    if scope == "arc" and not conf["allow_arc_scope"]:
        scope = "chapter"

    start = int(raw.get("start_chapter") or chapter)
    persistent = bool(raw.get("persistent", False))
    end = 0 if persistent else int(raw.get("end_chapter") or 0)
    if not persistent and end and end < start:
        end = start

    # 类型强制：参数级类型（伪装腔、软调、失声、幻术、齐声、非人）不允许换声线。
    if not strategy["swap"] and kind not in (TYPE_RESTORE, TYPE_MUTE, TYPE_BASELINE):
        mode = "soft_mod"
    if kind == TYPE_FLASHBACK:
        if not any(marker in evidence for marker in _FLASHBACK_MARKERS):
            return None, "flashback_without_marker"
        mode, scope = "soft_mod", "scene"
    if kind == TYPE_RESTORE:
        mode = "same"
    if kind == TYPE_BASELINE:
        mode = "same"
        if not (_is_narration_evidence(evidence)
                and any(word in evidence for word in ("声音", "嗓音", "声线", "语调"))):
            return None, "baseline_not_allowed"
    target = str(raw.get("target_voice_owner") or "").strip()
    if mode == "swap_to" and not target:
        return None, "missing_target_voice_owner"

    params_raw = raw.get("params") or {}
    limits = conf["params"]
    llm_params = (
        _clamp(params_raw.get("rate_pct", 0), float(limits["max_rate_pct"])),
        _clamp(params_raw.get("pitch_hz", 0), float(limits["max_pitch_hz"])),
        _clamp(params_raw.get("volume_pct", 0), float(limits["max_volume_pct"])),
    )
    params = {"rate_pct": llm_params[0], "pitch_hz": llm_params[1],
              "volume_pct": llm_params[2]}
    if mode == "soft_mod":
        if not _audible(evidence):
            return None, "no_audible_cue"
        within = all(abs(value) <= limit for value, limit in zip(llm_params, _PARAM_ENVELOPE))
        if not within or not any(abs(value) >= 0.5 for value in llm_params):
            # LLM 未给数值或给到极端值：按证据里的动作线索取安全默认值
            fallback = _cue_params(evidence)
            if fallback is None:
                return None, "no_cue_for_soft_mod"
            params = {"rate_pct": fallback[0], "pitch_hz": fallback[1],
                      "volume_pct": fallback[2]}

    if kind == TYPE_DISGUISE:
        reveal = str(raw.get("reveal_policy") or "reader")
        if reveal == "concealed":
            if not target:
                return None, "missing_target_voice_owner"
            mode = "swap_to"
        else:
            mode = "same"

    confidence = float(raw.get("confidence") or 0.0)
    event = {
        "event_key": "",
        "book_id": book_id,
        "chapter": chapter,
        "type": kind,
        "character_id": cid,
        "name": (identities.get(cid) or {}).get("canonical_name") or cid,
        "voice_mode": mode,
        "target_voice_owner": target,
        "scope": scope,
        "start_chapter": start,
        "end_chapter": end,
        "persistent": persistent,
        "params": params,
        "confidence": round(confidence, 3),
        "evidence_quote": evidence,
        "disguised_as": str(raw.get("disguised_as") or ""),
        "reveal_policy": str(raw.get("reveal_policy") or "reader"),
        "reason": str(raw.get("reason") or ""),
        "no_audio_effect": mode == "same" and not any(
            abs(value) >= 0.5 for value in params.values()),
    }
    event["event_key"] = event_key(event)
    return event, ""


def gate(event: dict, conf: dict) -> str:
    """返回 applied / pending_review / dropped。"""
    confidence = float(event.get("confidence") or 0)
    if confidence >= float(conf["auto_apply_confidence"]):
        return "applied"
    if confidence >= float(conf["pending_review_confidence"]):
        return "pending_review"
    return "dropped"


# ---------------------------------------------------------------- 章节输入压缩

_CUE_WORDS = (
    "声音", "低声", "压低", "尖", "沙哑", "嘶", "哑", "哽咽", "哭腔", "哭", "笑",
    "醉", "虚弱", "受伤", "吼", "喊", "嚷", "尖叫", "耳语", "喃喃", "颤抖", "冷",
    "粗", "柔", "假声", "男声", "女声", "口音", "学", "模仿", "伪装", "易容",
    "替身", "附身", "夺舍", "灵魂", "幻术", "变身", "齐声", "合唱", "回忆",
    "当年", "小时候", "失声", "换声", "变声", "嗓音",
)


def compact_text(text: str, limit: int = 120) -> str:
    """旁白压缩：保留开头，若含声音线索词则保留线索前后窗口。"""
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(body) <= limit:
        return body
    for word in _CUE_WORDS:
        index = body.find(word)
        if index >= 0:
            left = max(0, index - limit // 3)
            return ("…" if left else "") + body[left:left + limit] + "…"
    return body[:limit] + "…"


def chapter_chunks(record: dict, conf: dict) -> list[list[dict]]:
    """按段落把章节切成若干块；优先在旁白处断开，控制上下文长度。"""
    segments = list(record.get("segments") or [])
    chunk_chars = int(conf["chunk_chars"])
    max_segments = int(conf["max_segments_per_chunk"])
    chunks: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for seg in segments:
        text = str(seg.get("text") or "")
        if current and (size + len(text) > chunk_chars or len(current) >= max_segments):
            chunks.append(current)
            current, size = [], 0
        current.append(seg)
        size += len(text)
    if current:
        chunks.append(current)
    return chunks


def cast_brief(registry: dict, book_id: str, speakers: set[str],
               max_rows: int = 60) -> list[dict]:
    rows = []
    extras = []
    for cid, ident in (registry.get("identities") or {}).items():
        books = ident.get("books") or []
        if books and book_id not in books:
            continue
        name = ident.get("canonical_name") or cid
        stage = ""
        for stage_name, branch in (ident.get("stage_profiles") or {}).items():
            if branch.get("last_seen_chapter") is not None:
                stage = stage_name
        row = {
            "character_id": cid,
            "name": name,
            "gender": ident.get("gender") or "未知",
            "age_stage": stage,
            "persona": ident.get("persona") or "",
            "generic": bool(ident.get("generic")),
        }
        # 说话人必给；其他角色只给少量，供 swap_to 的 target 回查
        if cid in speakers or name in speakers:
            rows.append(row)
        elif not row["generic"]:
            extras.append(row)
    room = max(0, max_rows - len(rows))
    return rows + extras[:room]


def prior_events_summary(registry: dict, book_id: str, chapter: int, limit: int) -> list[dict]:
    rows = []
    for event in registry.get("voice_events") or []:
        if event.get("book_id") not in (book_id, None):
            continue
        if str(event.get("source") or "") != "llm_event":
            continue
        if int(event.get("chapter") or 0) >= chapter:
            continue
        rows.append({"chapter": event.get("chapter"), "type": event.get("type"),
                     "character_id": event.get("character_id"),
                     "voice_mode": event.get("voice_mode"),
                     "scope": event.get("scope"), "evidence": event.get("evidence_quote")})
    return rows[-max(1, limit):]


SYSTEM_PROMPT = """你是小说的“声线导演”。只做一件事：判断本章里哪些角色的声音需要变化，并输出 JSON。

判定原则：
1. 只有在原文有明确线索时才输出事件；没有线索就返回 {"events": []}，不要推测。
2. evidence_quote 必须逐字引用原文中的一句话（可以是一小段），不得改写。
3. 临时状态（耳语、压低嗓音、醉酒、虚弱、受伤、喊叫、哭腔、冷笑、模仿腔调）用
   voice_mode=soft_mod，params 给出 rate_pct / pitch_hz / volume_pct；只有原文
   明确写到“声音/嗓音/语调/语速/音量”或明确的发声动作时才输出，面部表情、
   情绪词（悲戚、严肃、沉思）不算声音线索。数值建议 -10~+10 之间，不要顶格。
4. 中长期状态（女扮男装、伤残失声、附身/夺舍、灵魂互换、幻术伪装、双胞胎/分身）
   才允许 voice_mode=swap_to，且必须给 target_voice_owner。
5. 身份伪装/易容/替身默认按“读者视角”：voice_mode=same，让读者能认出原角色；
   仅当叙事刻意向读者隐瞒身份（原文明确写成“陌生声音/听不出是谁”）时，
   才用 voice_mode=swap_to 并把 reveal_policy 设为 concealed。
6. type=flashback 仅当证据里出现“回忆/当年/小时候/多年前/往事/梦中/想起”这类
   明确标记时才可用；否则不要标成 flashback。
7. 声音恢复正常用 type=restore，voice_mode=same。
7b. 角色首次出场时对声音的静态描写（如“声音轻柔而恭顺”）是配音属性，用
    type=voice_baseline，只记录、不改变合成；不要当成临时变化或换声。
8. swap_to 的 target_voice_owner 可写角色名/character_id，或用
   pool:opposite_gender（异性池）、self:derived（同角色派生音）。
9. character_id 必须来自给出的角色清单，不要造 id。
10. 同一角色每章最多 2 条事件；不要为普通对白输出事件，只保留前后差异明显、
    对听感有实质影响的。
11. confidence 取 0~1：证据明确且类型清楚给 0.9 左右；需要推断给 0.6~0.75；
    纯猜测不要输出。

示例（只学边界，不要照抄）：
- “他压低声音说” → type=soft_mod, scope=scene
- “楼温喝得醉熏熏，自以为小声地说” → type=soft_mod（醉酒），不是 flashback
- “声音轻柔而恭顺”（角色首次出场）→ type=voice_baseline，只记录
- “周律不耐烦地说” → type=soft_mod, scope=scene
- “回忆中，少年时的他说” → type=flashback, scope=scene
- “她忽然换成男声” → type=gender_disguise, voice_mode=swap_to

只输出一个 JSON 对象，不要解释、推理或 markdown。"""


def build_messages(record: dict, registry: dict, book_id: str, chapter: int,
                   chunk: list[dict], cast: list[dict], prior: list[dict],
                   conf: dict) -> list[dict]:
    lines = [f"第 {chapter} 章：{record.get('title') or record.get('label') or ''}", "",
             "角色清单（character_id | 名字 | 性别 | 年龄段 | 人设）："]
    for row in cast:
        lines.append(f"- {row['character_id']} | {row['name']} | {row['gender']} | "
                     f"{row['age_stage'] or '未知'} | {row['persona'][:40]}")
    if prior:
        lines += ["", "前几章已生效的声线事件（供连续性判断，不要重复输出）："]
        for item in prior:
            lines.append(f"- 第{item['chapter']}章 {item['type']} {item['character_id']} "
                         f"{item['voice_mode']}/{item['scope']}：{item['evidence'][:50]}")
    lines += ["", "本章片段（序号 | 类型 | 说话人 | 文本）："]
    for index, seg in enumerate(chunk):
        text = compact_text(seg.get("text"), 140)
        lines.append(f"{index} | {seg.get('type')} | {seg.get('speaker')} | {text}")
    lines += ["", f"只输出 JSON：{{\"events\": [...]}}，最多 {int(conf['max_events_per_chapter'])} 条。"]
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


# ---------------------------------------------------------------- 抽取


def extract_chapter(client, cfg: dict, record: dict, registry: dict,
                    book_id: str, chapter: int) -> dict:
    """一章一次抽取（内部按块调用 LLM）；返回事件 + 拒绝原因 + 统计。"""
    conf = config(cfg)
    identities = registry.get("identities") or {}
    speakers = {str(seg.get("speaker") or "") for seg in record.get("segments") or []
                if seg.get("type") == "dialogue"}
    cast = cast_brief(registry, book_id, speakers)
    prior = prior_events_summary(registry, book_id, chapter,
                                 int(conf["prior_events_chapters"]))
    body = "".join(str(seg.get("text") or "") for seg in record.get("segments") or [])
    chunks = chapter_chunks(record, conf)
    events: list[dict] = []
    rejected: list[dict] = []
    errors: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        messages = build_messages(record, registry, book_id, chapter, chunk,
                                  cast, prior, conf)
        try:
            payload = client.chat_json(messages, schema=EVENTS_SCHEMA,
                                       schema_name="voice_events")
        except Exception as err:  # 单块失败不影响其他块
            errors.append(f"{type(err).__name__}: {err}")
            continue
        for raw in (payload.get("events") or [])[:int(conf["max_events_per_chapter"])]:
            event, error = normalize_event(raw, chapter=chapter, book_id=book_id,
                                           text_body=body, identities=identities,
                                           conf=conf)
            if error:
                rejected.append({"raw": raw, "error": error})
                continue
            if event["event_key"] in seen:
                continue
            seen.add(event["event_key"])
            events.append(event)
    cap = int(conf["max_events_per_character"])
    if cap > 0 and events:
        grouped: dict[str, list[tuple[int, dict]]] = {}
        for index, event in enumerate(events):
            grouped.setdefault(event["character_id"], []).append((index, event))
        kept: list[tuple[int, dict]] = []
        for items in grouped.values():
            items.sort(key=lambda pair: (-float(pair[1].get("confidence") or 0), pair[0]))
            kept.extend(items[:cap])
            for _, extra in items[cap:]:
                rejected.append({"raw": {"character_id": extra["character_id"],
                                         "type": extra["type"],
                                         "evidence_quote": extra["evidence_quote"]},
                                 "error": "character_event_cap"})
        events = [event for _, event in sorted(kept, key=lambda pair: pair[0])]
    return {"events": events, "rejected": rejected, "chunks": len(chunks),
            "errors": errors}


# ---------------------------------------------------------------- 确定性落地


def _binding_for(registry: dict, cid: str) -> tuple[str, dict] | tuple[None, None]:
    best = None
    for bid, binding in (registry.get("voice_bindings") or {}).items():
        if binding.get("character_id") != cid or not binding.get("voice"):
            continue
        chapter = int(binding.get("updated_at_chapter") or 0)
        if best is None or chapter >= best[0]:
            best = (chapter, bid, binding)
    if best is None:
        return None, None
    return best[1], best[2]


def target_voice(registry: dict, owner: str) -> str:
    if not owner or owner.startswith("pool:") or owner.startswith("self:"):
        return ""
    for bid, binding in (registry.get("voice_bindings") or {}).items():
        if binding.get("character_id") == owner or binding.get("id") == owner:
            return str(binding.get("voice") or "")
    return ""


def _apply_binding_overlay(registry: dict, event: dict) -> dict:
    bid, binding = _binding_for(registry, event["character_id"])
    if binding is None:
        return {"overlay": False, "note": "no_binding"}
    voice = target_voice(registry, event["target_voice_owner"])
    if not voice:
        return {"overlay": False, "note": "target_voice_unavailable"}
    binding["event_override"] = {
        "event_key": event["event_key"],
        "type": event["type"],
        "end_chapter": event["end_chapter"],
        "persistent": event["persistent"],
        "prev": {"voice": binding.get("voice"),
                 "rate": binding.get("rate"),
                 "pitch": binding.get("pitch"),
                 "volume": binding.get("volume")},
    }
    binding["voice"] = voice
    binding["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    binding["updated_by"] = "llm_event"
    return {"overlay": True, "binding_id": bid, "voice": voice}


def apply_events(registry: dict, events: list[dict], chapter: int, book_id: str,
                 dry_run: bool = False) -> dict:
    """把规范事件写入 registry；幂等；返回统计。dry_run 时不落盘。"""
    log = registry.setdefault("voice_events", [])
    existing = {str(item.get("event_key") or "") for item in log}
    summary = {"applied": 0, "pending_review": 0, "dropped": 0, "recorded": 0,
               "skipped": 0, "overlays": 0, "events": []}
    for event in events:
        key = event.get("event_key") or event_key(event)
        if key in existing:
            summary["skipped"] += 1
            continue
        status = event.get("status") or "applied"
        if status not in ("applied", "pending_review", "recorded"):
            status = "applied"
        record = dict(event)
        record.update({
            "id": f"ve:{key}",
            "kind": event.get("type"),
            "source": "llm_event",
            "status": status,
            "from_voice": None,
            "to_voice": None,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        if status == "applied" and not dry_run and event.get("voice_mode") == "swap_to" \
                and (event.get("scope") == "arc" or event.get("persistent")):
            overlay = _apply_binding_overlay(registry, event)
            if overlay.get("overlay"):
                record["to_voice"] = overlay.get("voice")
                record["binding_id"] = overlay.get("binding_id")
                summary["overlays"] += 1
        if not dry_run:
            log.append(record)
            existing.add(key)
        summary[status] += 1
        summary["events"].append({**record, "voice_mode": event.get("voice_mode"),
                                  "chapter": chapter, "book_id": book_id})
    return summary


def apply_extraction(registry: dict, extraction: dict, cfg: dict, chapter: int,
                     book_id: str, dry_run: bool = False) -> dict:
    """按置信度闸门 + 模式开关给事件定级，再落地。"""
    conf = config(cfg)
    events = []
    dropped: list[dict] = []
    for event in extraction.get("events") or []:
        status = gate(event, conf)
        if status == "dropped":
            dropped.append({"event_key": event["event_key"], "type": event["type"],
                            "character_id": event["character_id"],
                            "confidence": event["confidence"]})
            continue
        if status == "applied" and event.get("no_audio_effect"):
            status = "recorded"
        events.append({**event, "status": status})
    summary = apply_events(registry, events, chapter, book_id, dry_run=dry_run)
    summary["dropped"] += len(dropped)
    summary["dropped_events"] = dropped
    return summary


def revert_event(registry: dict, event_key_value: str, reason: str = "") -> dict:
    """撤销事件：交换类恢复 binding 快照；未生效的 pending 直接标记驳回。"""
    target = None
    for item in registry.get("voice_events") or []:
        if str(item.get("event_key") or "") == event_key_value:
            target = item
            break
    if target is None:
        return {"ok": False, "error": "event_not_found"}
    bid, binding = _binding_for(registry, target.get("character_id") or "")
    restored = False
    if binding is not None and (binding.get("event_override") or {}).get("event_key") == event_key_value:
        prev = (binding.get("event_override") or {}).get("prev") or {}
        for key in ("voice", "rate", "pitch", "volume"):
            if prev.get(key) is not None:
                binding[key] = prev[key]
        binding.pop("event_override", None)
        binding["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        binding["updated_by"] = "llm_event_revert"
        restored = True
    previous = target.get("status")
    target["status"] = "reverted"
    target["revert_reason"] = reason
    target["reverted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return {"ok": True, "previous_status": previous, "restored_binding": restored,
            "binding_id": bid}


# ---------------------------------------------------------------- 合成时覆盖


def _window(event: dict, chapter: int) -> bool:
    start = int(event.get("start_chapter") or 0)
    if chapter < start:
        return False
    end = int(event.get("end_chapter") or 0)
    if end and chapter > end:
        return False
    return True


def resolve_overrides(registry: dict, book_id: str, chapter: int, cid: str) -> dict:
    """返回本章该角色的覆盖：voice（可空）+ 参数偏移 + 事件元数据。"""
    active = []
    for event in registry.get("voice_events") or []:
        if str(event.get("status") or "") != "applied":
            continue
        if event.get("character_id") != cid:
            continue
        if event.get("book_id") not in (book_id, None):
            continue
        if not _window(event, chapter):
            continue
        active.append(event)
    if not active:
        return {"voice": "", "rate_pct": 0.0, "pitch_hz": 0.0, "volume_pct": 0.0,
                "events": []}
    swaps = [item for item in active if item.get("voice_mode") == "swap_to"]
    softs = [item for item in active if item.get("voice_mode") == "soft_mod"]
    voice = ""
    if swaps:
        swaps.sort(key=lambda item: (SWAP_PRIORITY.get(item.get("type"), 0),
                                     float(item.get("confidence") or 0)), reverse=True)
        voice = target_voice(registry, swaps[0].get("target_voice_owner") or "")
    params = {"rate_pct": 0.0, "pitch_hz": 0.0, "volume_pct": 0.0}
    for item in softs:
        for key in params:
            params[key] += float((item.get("params") or {}).get(key) or 0)
    limits = DEFAULT_CONFIG["params"]
    params["rate_pct"] = _clamp(params["rate_pct"], limits["max_rate_pct"])
    params["pitch_hz"] = _clamp(params["pitch_hz"], limits["max_pitch_hz"])
    params["volume_pct"] = _clamp(params["volume_pct"], limits["max_volume_pct"])
    return {"voice": voice, **params,
            "events": [{"type": item.get("type"), "scope": item.get("scope"),
                        "event_key": item.get("event_key")} for item in active]}


def apply_to_cast(cast_data: dict, registry: dict, book_id: str, chapter: int,
                  cfg: dict) -> dict:
    """在合成前把事件覆盖写进 cast profile（不改绑定、不改音频缓存键）。"""
    if not enabled(cfg):
        return {"enabled": False, "changed": 0}
    changed = 0
    for profile in (cast_data.get("cast") or {}).values():
        cid = profile.get("character_id") or ""
        if not cid:
            continue
        override = resolve_overrides(registry, book_id, chapter, cid)
        if not override["events"]:
            continue
        if override["voice"]:
            profile["edge_voice"] = override["voice"]
        base = {key: prosody.parse_value(profile.get(key) or default,
                                        "Hz" if key == "pitch" else "%")
                for key, default in (("rate", "+0%"), ("pitch", "+0Hz"), ("volume", "+0%"))}
        merged = {
            "rate": base["rate"] + override["rate_pct"],
            "pitch": base["pitch"] + override["pitch_hz"],
            "volume": base["volume"] + override["volume_pct"],
        }
        profile["rate"] = prosody.format_value(
            prosody.clamp_value(merged["rate"], "%", cfg), "%")
        profile["pitch"] = prosody.format_value(
            prosody.clamp_value(merged["pitch"], "Hz", cfg), "Hz")
        profile["volume"] = prosody.format_value(
            prosody.clamp_value(merged["volume"], "%", cfg), "%")
        profile["voice_event"] = override["events"]
        changed += 1
    return {"enabled": True, "changed": changed}


def pending_events(registry: dict, book_id: str | None = None) -> list[dict]:
    rows = []
    for event in registry.get("voice_events") or []:
        if str(event.get("status") or "") != "pending_review":
            continue
        if book_id and event.get("book_id") not in (book_id, None):
            continue
        rows.append(event)
    return rows


def dump_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
