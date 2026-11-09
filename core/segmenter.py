"""确定性剧本切分器：引号硬边界 + 上下文归属 + 重叠窗口多遍投票。

把“整章交给模型猜说话人”换成可复现的分层流程：

1) 词法层：按中文引号（“”「」『』‘’）切单元，任何单元不跨越引号边界；
   引号内再套引号时单独成单元并记录 level（2 = 转述层）。
   只表强调/术语的短引号（如 一口一个“础弟”）回收为旁白，不产生对白段。
2) 归属层（优先级从高到低，全部确定性）：
   a1 引号前最近一句的叙述动词主语（“楼础笑道：”）
   a2 同段续说（同段已有归属 + 中间叙述没有说话动词）
   a3 引号前叙述里最近的具名角色（“马维咬牙切齿，”）
   b  引号后紧跟的叙述（“……。”马维笑道）
   c2 称呼语回指（上一句对白里的“础弟/马兄”指向谁）
   c1 邻接律（A-B-A 交替，取最近出现过的另一个说话人）
   d  代词消解（他/她/其 -> 句中主语）
   e  以上全失败 -> unattributed（不默认塞“未知角色”）
3) 投票层：只把 unattributed（以及可选的“仅靠推断规则”的段落）交给 LLM，
   用重叠窗口（默认窗 12 段 / 步长 4 段，每段被 2-3 个窗口覆盖）多遍识别，
   按最小同意比例（默认 2/3）采纳；达不到就标 disputed，不硬选。

本模块只做纯计算，不读配置、不写文件、不碰 registry。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 词法层

QUOTE_PAIRS = {"“": "”", "「": "」", "『": "』", "‘": "’"}
QUOTE_OPEN = set(QUOTE_PAIRS)
QUOTE_CLOSE = {close: open_ for open_, close in QUOTE_PAIRS.items()}

# 说话动词：长的排前面，正则交替按先后命中
SPEECH_VERBS = [
    "拱手笑道", "正色道", "讪讪道", "插口道", "拍胸道", "低声道", "沉声道",
    "大声道", "轻声道", "冷笑道", "微笑道", "大笑道", "笑着说道",
    "说道", "问道", "答道", "笑道", "喝道", "叫道", "叹道", "骂道",
    "回道", "应道", "心道", "暗道", "追问", "开口", "插口", "解释",
    "提醒", "自夸", "嘀咕", "说", "道", "问", "答", "喊", "叫", "叹", "骂", "嚷", "吼",
]
_VERB_RE = re.compile("|".join(re.escape(v) for v in SPEECH_VERBS))
# 单字“道/说”容易误吞“知道/味道/说服/说明”等，命中后看相邻字
_VERB_FALSE_PREFIX = {"知", "味", "难", "报", "通", "铁", "街", "渠", "轨", "称", "说"}
_VERB_FALSE_SUFFIX = {"说": ("服", "明", "法", "话", "书", "谎", "辞", "理", "客")}

# 代词
PRONOUNS = {"他", "她", "其", "我", "你"}

# 宾语/介词标记：指着郭时风道 -> 郭时风是宾语，不是主语
_OBJECT_PREFIX = ("指着", "看着", "望着", "对着", "朝向", "看向", "转向",
                  "向", "朝", "对", "跟", "和", "与", "替", "给", "问", "敬")

# 称呼后缀（用于把“础弟/马兄”映射回角色）
_VOCATIVE_PREFIX = ("老", "小", "阿")

STOP_NAMES = {
    "觉得", "有些", "十分", "非常", "点头", "摇头", "摆手", "探身", "开口",
    "解释", "提醒", "自夸", "嘀咕", "大笑", "冷笑", "微笑", "笑道", "说道",
    "问道", "答道", "心道", "暗道", "咬牙", "切齿", "低声", "沉声", "轻声",
    "大声", "正色", "讪讪", "拱手", "拍胸", "插口", "又吃", "这才", "于是",
}


@dataclass
class Unit:
    index: int
    para: int
    kind: str                 # narration | dialogue | term
    level: int                # 引号层级：0 旁白，1 外层对白，2 转述层
    text: str                 # 去引号后的正文
    quote: str = ""           # 原始引号片段（含引号），便于人工核对
    start: int = 0
    end: int = 0
    speaker: str = ""
    rule: str = ""            # a1|a2|a3|b|c1|c2|d|e
    evidence: str = ""
    source: str = ""          # rule | llm | llm_override
    agreement: float | None = None
    votes: list = field(default_factory=list)
    status: str = ""          # decided | inferred | unattributed | disputed
    embedded: bool = False    # 转述层：并入外层对白朗读，不单独出行
    parent: int | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index, "para": self.para, "kind": self.kind,
            "level": self.level, "text": self.text, "quote": self.quote,
            "start": self.start, "end": self.end,
            "speaker": self.speaker, "rule": self.rule, "evidence": self.evidence,
            "source": self.source, "status": self.status,
            "agreement": self.agreement, "votes": self.votes,
            "embedded": self.embedded, "parent": self.parent,
        }


def parse_paragraph(text: str, para: int, base: int = 0) -> list[Unit]:
    """把一段原文切成不跨引号边界的单元，并保留原始引号片段。"""
    units: list[Unit] = []
    buf: list[str] = []
    stack: list[str] = []
    starts: list[int] = []
    buf_start = base
    cursor = base

    def flush(depth: int) -> None:
        nonlocal buf
        body = "".join(buf).strip()
        if body:
            units.append(Unit(index=len(units), para=para,
                              kind=("dialogue" if depth > 0 else "narration"),
                              level=depth, text=body, start=buf_start, end=cursor))
        buf = []

    for ch in text:
        if ch in QUOTE_OPEN:
            flush(len(stack))
            stack.append(ch)
            starts.append(cursor)
            cursor += 1
            buf_start = cursor
            continue
        if ch in QUOTE_CLOSE and stack and QUOTE_CLOSE[ch] == stack[-1]:
            flush(len(stack))
            if units and starts:
                open_pos = starts.pop()
                units[-1].quote = text[open_pos: cursor + 1]
            stack.pop()
            cursor += 1
            buf_start = cursor
            continue
        if not buf:
            buf_start = cursor
        buf.append(ch)
        cursor += 1
    flush(len(stack))
    return units


def _is_term(unit: Unit, siblings: list[Unit], pos: int) -> bool:
    """短引号只在表强调/术语时回收为旁白（如 一口一个“础弟”）。"""
    if unit.level != 1 or len(unit.text) > 10:
        return False
    if re.search(r"[。！？…；，,]", unit.text):
        return False
    before = [u for u in siblings[:pos] if u.kind != "dialogue"]
    after = [u for u in siblings[pos + 1:] if u.kind != "dialogue"]
    if not before or not after:
        return False
    prev_text = before[-1].text.rstrip()
    if prev_text.endswith(("：", ":")):
        return False
    if re.search(r"(说|道|问|答|喊|叫|笑|叹|骂|嚷|吼)$", prev_text.rstrip("，, ")):
        return False
    return True


def merge_narration(units: list[Unit]) -> list[Unit]:
    merged: list[Unit] = []
    for unit in units:
        if (merged and merged[-1].kind == "narration" and unit.kind == "narration"
                and merged[-1].para == unit.para):
            merged[-1].text += unit.text
            merged[-1].end = unit.end
        else:
            merged.append(unit)
    for index, unit in enumerate(merged):
        unit.index = index
    return merged


# ---------------------------------------------------------------- 角色表

def name_aliases(name: str) -> set[str]:
    """生成称呼别名：楼础 -> 础弟/楼兄/楼础；郭时风 -> 郭兄/时风/时风兄。"""
    aliases = {name}
    if len(name) >= 2:
        aliases.add(name[-1] + "弟")
        aliases.add(name[0] + "兄")
        aliases.add(name[-2:] + "弟")
        if len(name) == 3:
            aliases.add(name[1:])
            aliases.add(name[1:] + "兄")
            aliases.add(name[0] + "公子")
            aliases.add(name[0] + "先生")
    return {item for item in aliases if item and item != name[0]}


def _cjk_only(text: str) -> bool:
    return bool(text) and all("\u4e00" <= ch <= "\u9fff" for ch in text)


def extract_roster(text: str, seed: list[str] | None = None) -> list[str]:
    """角色表 = 外部种子（registry/cast）+ 原文里高频出现的“X说/道”主语。"""
    seed = [name for name in (seed or []) if _cjk_only(name) and len(name) >= 2]
    counts: dict[str, int] = {}
    for match in _VERB_RE.finditer(text):
        head = text[max(0, match.start() - 6):match.start()]
        for size in (3, 2):
            if len(head) < size:
                continue
            name = head[-size:]
            if not _cjk_only(name) or name in STOP_NAMES:
                continue
            if name[0] in "的了是在和与就跟对向":
                continue
            if any(name.endswith(suffix) for suffix in ("道", "说", "问", "答", "笑")):
                continue
            counts[name] = counts.get(name, 0) + 1
            break
    frequent = [name for name, count in counts.items() if count >= 2]
    return list(dict.fromkeys(seed + sorted(frequent, key=lambda n: -counts[n])))


def _occurrences(text: str, name: str) -> list[int]:
    out, pos = [], text.find(name)
    while pos >= 0:
        out.append(pos)
        pos = text.find(name, pos + 1)
    return out


def nearest_name_before(text: str, pos: int, roster: list[str],
                        exclude: str = "") -> tuple[str, int] | None:
    best = None
    for name in roster:
        if name == exclude:
            continue
        for alias in name_aliases(name):
            hit = text.rfind(alias, 0, pos)
            if hit >= 0 and (best is None or hit > best[1]):
                best = (name, hit)
    return best


def first_name_in(text: str, roster: list[str]) -> str:
    best, best_pos = "", len(text) + 1
    for name in roster:
        for alias in name_aliases(name):
            hit = text.find(alias)
            if 0 <= hit < best_pos:
                best, best_pos = name, hit
    return best


def _last_sentence(text: str) -> str:
    parts = [item for item in re.split(r"(?<=[。！？…；])", text) if item.strip()]
    return parts[-1] if parts else text


def _first_sentence(text: str) -> str:
    parts = [item for item in re.split(r"(?<=[。！？…；])", text) if item.strip()]
    return parts[0] if parts else text


def _clause_before(text: str, pos: int) -> str:
    cut = max(text.rfind(ch, 0, pos) for ch in "，,；;：:。！？!? ")
    return text[cut + 1:pos]


def _is_object(clause: str, name: str) -> bool:
    """clause 里紧挨 name 之前是不是宾语/介词标记。"""
    hit = clause.rfind(name)
    if hit < 0:
        return False
    head = clause[:hit]
    return any(head.endswith(marker) for marker in _OBJECT_PREFIX)


def _verb_subject(sentence: str, roster: list[str]) -> tuple[str, str, str]:
    """返回 (speaker, rule, evidence)；找不到返回 ("", "", "")。"""
    verbs = [m for m in _VERB_RE.finditer(sentence)]
    if not verbs:
        return "", "", ""
    verb = verbs[-1]
    if verb.group(0) in ("道", "说"):
        prev = sentence[verb.start() - 1] if verb.start() > 0 else ""
        nxt = sentence[verb.end()] if verb.end() < len(sentence) else ""
        if prev in _VERB_FALSE_PREFIX or nxt in _VERB_FALSE_SUFFIX.get(verb.group(0), ()):
            return "", "", ""
    clause = _clause_before(sentence, verb.start())
    found = nearest_name_before(clause, len(clause), roster)
    while found and _is_object(clause, found[0]):
        clause = clause[:found[1]]
        found = nearest_name_before(clause, len(clause), roster)
    if found:
        return found[0], "a1", sentence.strip()
    if any(ch in clause for ch in PRONOUNS):
        first = first_name_in(sentence, roster)
        if first:
            return first, "d", sentence.strip()
    first = first_name_in(sentence, roster)
    if first:
        return first, "a1", sentence.strip()
    return "", "", ""


def _has_speech_verb(text: str) -> bool:
    for match in _VERB_RE.finditer(text):
        if match.group(0) in ("道", "说"):
            prev = text[match.start() - 1] if match.start() > 0 else ""
            nxt = text[match.end()] if match.end() < len(text) else ""
            if prev in _VERB_FALSE_PREFIX or nxt in _VERB_FALSE_SUFFIX.get(match.group(0), ()):
                continue
        return True
    return False


# ---------------------------------------------------------------- 归属层

def _prev_narration(siblings: list[Unit], pos: int) -> Unit | None:
    for unit in reversed(siblings[:pos]):
        if unit.kind != "dialogue":
            return unit
        if unit.para != siblings[pos].para:
            break
    return None


def _next_narration(siblings: list[Unit], pos: int) -> Unit | None:
    for unit in siblings[pos + 1:]:
        if unit.kind != "dialogue":
            return unit
        if unit.para != siblings[pos].para:
            break
    return None


def _vocative(text: str, roster: list[str], exclude: str) -> str:
    for name in roster:
        if name == exclude:
            continue
        for alias in sorted(name_aliases(name) - {name}, key=len, reverse=True):
            if len(alias) >= 2 and alias in text:
                return name
    return ""


def attribute(units: list[Unit], roster: list[str]) -> list[Unit]:
    """按 a1/a2/a3/b/c2/c1/d/e 的优先级给对白单元定说话人。"""
    last_speaker_by_para: dict[int, str] = {}
    last_dialogue_speaker = ""
    recent_speakers: list[str] = []
    for pos, unit in enumerate(units):
        if unit.kind != "dialogue":
            continue
        # 嵌套引号：沿用外层对白说话人（转述按同一人在读）
        if unit.level >= 2:
            unit.speaker = last_speaker_by_para.get(unit.para, "") or last_dialogue_speaker
            unit.rule = "nest"
            unit.source = "rule"
            unit.status = "decided"
            continue

        speaker, rule, evidence = "", "", ""
        prev = _prev_narration(units, pos)
        nxt = _next_narration(units, pos)

        # a1：引号前最近一句的叙述动词主语
        if prev is not None and prev.para == unit.para:
            speaker, rule, evidence = _verb_subject(_last_sentence(prev.text), roster)

        # a2：同段续说（已归属 + 中间叙述没有说话动词）
        if not speaker:
            anchor = last_speaker_by_para.get(unit.para)
            if anchor and (prev is None or prev.para != unit.para
                           or not _has_speech_verb(prev.text)):
                speaker, rule, evidence = anchor, "a2", prev.text.strip() if prev else ""

        # a3：引号前叙述里最近的具名角色
        if not speaker and prev is not None and prev.para == unit.para:
            found = nearest_name_before(prev.text, len(prev.text), roster)
            if found:
                speaker, rule, evidence = found[0], "a3", prev.text.strip()

        # b：引号后紧跟的叙述（“……。”马维笑道）
        if not speaker and nxt is not None and nxt.para == unit.para:
            sentence = _first_sentence(nxt.text)
            head = sentence.lstrip("，, ")
            for name in roster:
                if head.startswith(name):
                    speaker, rule, evidence = name, "b", sentence.strip()
                    break
            if not speaker:
                speaker, rule, evidence = _verb_subject(sentence, roster)
                if rule:
                    rule = "b"

        # c2：称呼语回指（上一句对白在叫谁）
        if not speaker and last_dialogue_speaker:
            speaker = _vocative(units[pos - 1].text if pos else "", roster, last_dialogue_speaker)
            if speaker:
                rule = "c2"
                evidence = units[pos - 1].text[:40]

        # c1：邻接律 A-B-A
        if not speaker and last_dialogue_speaker:
            for name in reversed(recent_speakers):
                if name != last_dialogue_speaker:
                    speaker, rule = name, "c1"
                    evidence = last_dialogue_speaker
                    break

        if not speaker:
            unit.speaker = ""
            unit.rule = "e"
            unit.status = "unattributed"
            continue

        unit.speaker = speaker
        unit.rule = rule
        unit.source = "rule"
        unit.evidence = evidence
        unit.status = "decided" if rule in ("a1", "a3", "b") else "inferred"
        last_speaker_by_para[unit.para] = speaker
        last_dialogue_speaker = speaker
        recent_speakers.append(speaker)
        if len(recent_speakers) > 24:
            recent_speakers.pop(0)
    return units


# ---------------------------------------------------------------- 装配

def _split_long(text: str, limit: int) -> list[str]:
    if limit <= 0 or len(text) <= limit:
        return [text]
    parts = re.split(r"(?<=[。！？；…])", text)
    out, current = [], ""
    for part in parts:
        if not part:
            continue
        if len(current) + len(part) <= limit:
            current += part
        else:
            if current:
                out.append(current)
            current = part
    if current:
        out.append(current)
    return out or [text]


def fold_nested(units: list[Unit]) -> tuple[list[Unit], list[tuple[Unit, Unit]]]:
    """把同级对白碎片与转述层并回外层对白，转述层仍单独留档（不重复朗读）。"""
    out: list[Unit] = []
    nested: list[tuple[Unit, Unit]] = []
    i = 0
    while i < len(units):
        unit = units[i]
        if unit.kind == "dialogue" and unit.level == 1:
            text = unit.text
            j = i + 1
            while j < len(units):
                nxt = units[j]
                if nxt.para != unit.para or nxt.kind != "dialogue":
                    break
                if nxt.level >= 2:
                    nxt.embedded = True
                    nested.append((nxt, unit))
                text += nxt.text
                j += 1
            unit.text = text
            out.append(unit)
            i = j
            continue
        out.append(unit)
        i += 1
    return out, nested


def segment_chapter(text: str, seed_names: list[str] | None = None,
                    max_chars: int = 150, narrator: str = "旁白") -> dict:
    """返回 {"units": [...], "segments": [...], "roster": [...], "paragraphs": n}。"""
    roster = extract_roster(text, seed_names)
    units: list[Unit] = []
    offset = 0
    paragraphs = [line for line in text.split("\n")]
    for para_index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            offset += len(paragraph) + 1
            continue
        para_units = parse_paragraph(paragraph, para_index, offset)
        units.extend(para_units)
        offset += len(paragraph) + 1

    # 术语引号回收为旁白
    by_para: dict[int, list[Unit]] = {}
    for unit in units:
        by_para.setdefault(unit.para, []).append(unit)
    for unit in units:
        siblings = by_para[unit.para]
        pos = next(i for i, item in enumerate(siblings) if item is unit)
        if unit.kind == "dialogue" and _is_term(unit, siblings, pos):
            unit.kind = "narration"
            unit.level = 0
    units = merge_narration([unit for unit in units if unit.kind != "term"])
    units, nested = fold_nested(units)
    for position, unit in enumerate(units):
        unit.index = position
        if unit.kind == "dialogue" and unit.level == 1:
            unit.quote = f"“{unit.text}”"
    for child, parent in nested:
        child.parent = parent.index
    units = attribute(units, roster)

    segments: list[dict] = []
    for unit in units:
        if unit.kind == "dialogue":
            speakers = [unit.speaker or ""]
            kind = "dialogue"
        else:
            speakers = [narrator]
            kind = "narration"
        for piece in _split_long(unit.text, max_chars):
            segments.append({
                "type": kind,
                "speaker": speakers[0],
                "emotion": "平静",
                "text": piece,
                "unit": unit.index,
                "level": unit.level,
                "rule": unit.rule,
                "status": unit.status,
            })
    return {"units": [unit.to_dict() for unit in units], "segments": segments,
            "nested": [child.to_dict() for child, _ in nested],
            "roster": roster, "paragraphs": len([p for p in paragraphs if p.strip()])}


# ---------------------------------------------------------------- 投票层

def build_windows(units: list[Unit], size: int = 12, step: int = 4) -> list[dict]:
    """流动窗口：窗 size 段、步长 step 段，每段被 2-3 个窗口覆盖。"""
    windows = []
    total = len(units)
    start = 0
    while start < total:
        end = min(total, start + size)
        windows.append({"id": f"{start}-{end}", "start": start, "end": end})
        if end >= total:
            break
        start += max(1, step)
    return windows


def render_window(units: list[Unit], window: dict, targets: set[int]) -> str:
    lines = []
    for unit in units[window["start"]:window["end"]]:
        mark = "★" if unit.index in targets else " "
        if unit.kind == "dialogue":
            tag = f"[{unit.index}|对白|L{unit.level}]"
            body = f"“{unit.text}”"
        else:
            tag = f"[{unit.index}|旁白]"
            body = unit.text
        lines.append(f"{mark}{tag} {body}")
    return "\n".join(lines)


VOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "speaker": {"type": "string"},
                    "confidence": {"type": "number"},
                    "why": {"type": "string"},
                },
                "required": ["index", "speaker", "confidence"],
            },
        }
    },
    "required": ["assignments"],
}


def vote_prompt(units: list[Unit], window: dict, targets: set[int],
                roster: list[str]) -> str:
    return (
        "你是中文有声小说台本的说话人归属器。只输出一个 JSON 对象，不要解释。\n"
        f"候选角色：{'、'.join(roster)}\n"
        "标记说明：每行 `[序号|旁白]` 是旁白；`[序号|对白|L层级]` 是引号内对白，"
        "L1 为外层对白，L2 为引号里的转述层。\n"
        "只判定行首带 ★ 的段落，其余段落仅作上下文，不要输出它们的归属。\n"
        f"本轮必须判定的序号：{sorted(targets)}。每个序号都要有一条结果，"
        "判不出就填 \"UNKNOWN\"，不要省略任何一条。\n"
        "判定规则：只有在上下文有明确指向（叙述动词、称呼、邻接交替）时才选候选角色；"
        "无法确定必须输出 \"UNKNOWN\"，禁止猜测。\n"
        '输出格式：{"assignments":[{"index":整数,"speaker":"候选角色或UNKNOWN",'
        '"confidence":0到1,"why":"不超过20字的依据"}]}\n\n'
        f"{render_window(units, window, targets)}"
    )


def validate_votes(data, window: dict, targets: set[int],
                   roster: list[str]) -> tuple[dict, list[str], list[int], set[int]]:
    """校验单遍返回：speaker 必须在候选表或 UNKNOWN。

    模型偶尔会顺手判上下文段（不在 ★ 列表）：这些条目直接忽略，只计入 extra。
    模型漏判（少列了 ★ 段）不算错误：该遍对这些段没有投票，其余投票仍然有效；
    漏判集合单独返回，由调用方决定是否重试补齐。
    """
    errors: list[str] = []
    extra: list[int] = []
    votes: dict[int, dict] = {}
    items = data.get("assignments") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return {}, ["返回不是数组"], [], set()
    for item in items:
        if not isinstance(item, dict):
            errors.append("条目不是对象")
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            errors.append(f"index 非法：{item.get('index')!r}")
            continue
        if index not in targets:
            extra.append(index)
            continue
        speaker = str(item.get("speaker") or "").strip()
        if speaker != "UNKNOWN" and speaker not in roster:
            errors.append(f"index {index} 的 speaker {speaker!r} 不在候选表")
            continue
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        votes[index] = {"speaker": speaker, "confidence": max(0.0, min(1.0, confidence)),
                        "why": str(item.get("why") or "")[:60], "window": window["id"]}
    return votes, errors, extra, targets - set(votes)


def tally(votes_by_pass: list[dict], min_agree: float = 2 / 3,
          min_passes: int = 2) -> dict[int, dict]:
    """多遍投票：一致即采纳；达到最小同意比例采纳；否则 disputed。

    覆盖不足（有效遍数 < min_passes）一律标 disputed——宁可交人工复核，
    也不拿单遍结论当共识。
    """
    result: dict[int, dict] = {}
    indexes = sorted({index for group in votes_by_pass for index in group})
    for index in indexes:
        rounds = [group[index] for group in votes_by_pass if index in group]
        counts: dict[str, int] = {}
        for round_ in rounds:
            counts[round_["speaker"]] = counts.get(round_["speaker"], 0) + 1
        winner, hits = max(counts.items(), key=lambda kv: kv[1])
        agreement = hits / len(rounds) if rounds else 0.0
        result[index] = {
            "speaker": "" if winner == "UNKNOWN" else winner,
            "agreement": round(agreement, 3),
            "passes": len(rounds),
            "votes": [{"speaker": item["speaker"], "confidence": item["confidence"],
                       "why": item["why"], "window": item["window"]} for item in rounds],
            "disputed": winner == "UNKNOWN" or agreement < min_agree
                        or len(rounds) < min_passes,
        }
    return result


def resolve_with_llm(units: list[Unit], llm, roster: list[str], *,
                     targets: set[int] | None = None, size: int = 12, step: int = 4,
                     min_agree: float = 2 / 3, retries: int = 1) -> dict:
    """只对 targets（默认 unattributed）做重叠窗口多遍识别；返回 {index: 裁决}。"""
    targets = set(targets if targets is not None else
                  [unit.index for unit in units if unit.status == "unattributed"])
    if not targets or not roster:
        return {"votes": {}, "calls": 0, "errors": [], "windows": 0}
    windows = [w for w in build_windows(units, size, step)
               if any(w["start"] <= i < w["end"] for i in targets)]
    # 边界补窗：章节首尾的单元天然只落在一个窗口里，补一个上下文长度不同的窗口，
    # 保证每个待判单元至少被 2 遍覆盖（否则 min_passes 会直接判 disputed）。
    total = len(units)
    seen = {(w["start"], w["end"]) for w in windows}
    for index in sorted(targets):
        if sum(1 for w in windows if w["start"] <= index < w["end"]) >= 2:
            continue
        if index < size // 2 + step:
            span = (0, min(total, size + step))
        elif index >= total - size // 2 - step:
            span = (max(0, total - size - step), total)
        else:
            head = max(0, min(index - size // 2, total - size))
            span = (head, min(total, head + size))
        if span not in seen:
            seen.add(span)
            windows.append({"id": f"{span[0]}-{span[1]}",
                            "start": span[0], "end": span[1]})
    windows.sort(key=lambda w: w["start"])
    votes_by_pass: list[dict] = []
    extras: list[int] = []
    errors: list[str] = []
    calls = 0
    for window in windows:
        inside = {i for i in targets if window["start"] <= i < window["end"]}
        if not inside:
            continue
        votes: dict[int, dict] = {}
        problems: list[str] = []
        best: tuple[dict, list[int], set[int]] | None = None
        for _ in range(retries + 1):
            calls += 1
            try:
                data = llm.chat_json(
                    [{"role": "system", "content": "只输出一个 JSON 对象，不要解释或 markdown。"},
                     {"role": "user", "content": vote_prompt(units, window, inside, roster)}],
                    schema=VOTE_SCHEMA, schema_name="assignments")
            except Exception as err:  # 单窗口失败不拖垮整章，交给下一窗口
                problems = [f"调用失败：{err}"]
                continue
            votes, problems, extra, missing = validate_votes(data, window, inside, roster)
            if problems:
                continue
            best = (votes, extra, missing)
            if not missing:  # 完整一遍，不必重试
                break
        if best is None:
            errors.append(f"窗口 {window['id']} 校验失败：{'；'.join(problems[:3])}")
            continue
        votes, extra, missing = best
        extras.extend(extra)
        if missing:
            errors.append(f"窗口 {window['id']} 部分漏判 {sorted(missing)}（该遍仍有效）")
        if votes:
            votes_by_pass.append(votes)
    return {"votes": tally(votes_by_pass, min_agree), "calls": calls,
            "errors": errors, "windows": len(windows),
            "passes": len(votes_by_pass), "extra_indexes": len(extras)}


def apply_votes(units: list[Unit], verdicts: dict, *, override_inferred: bool = False) -> dict:
    """把投票结果写回单元。

    默认只采纳 unattributed 的残余段；对仅靠推断规则（a2/c1/c2/d）判定的段，
    投票只作审计：一致就记 agreement，不一致记 rule_llm_conflict 交人工复核，
    不静默推翻确定性规则（实测 LLM 在推断段上更容易被前文带偏）。
    """
    stats = {"adopted": 0, "disputed": 0, "overrode": 0, "conflict": 0,
             "agree": 0, "conflict_items": []}
    for unit in units:
        verdict = verdicts.get(unit.index)
        if not verdict:
            continue
        if verdict["disputed"]:
            unit.status = "disputed"
            unit.agreement = verdict["agreement"]
            unit.votes = verdict["votes"]
            stats["disputed"] += 1
            continue
        speaker = verdict["speaker"]
        if not speaker:
            unit.status = "disputed"
            unit.votes = verdict["votes"]
            stats["disputed"] += 1
            continue
        if unit.status in ("unattributed", ""):
            unit.speaker = speaker
            unit.source = "llm"
            unit.rule = "llm"
            unit.status = "decided"
            stats["adopted"] += 1
        elif unit.status == "inferred":
            if unit.speaker == speaker:
                stats["agree"] += 1
            elif override_inferred:
                unit.speaker = speaker
                unit.source = "llm_override"
                unit.status = "decided"
                stats["overrode"] += 1
            else:
                unit.status = "rule_llm_conflict"
                stats["conflict"] += 1
                stats["conflict_items"].append({
                    "index": unit.index, "text": unit.text[:60],
                    "rule": unit.rule, "rule_speaker": unit.speaker,
                    "llm_speaker": speaker, "agreement": verdict["agreement"],
                    "votes": verdict["votes"],
                })
        else:
            if unit.speaker and unit.speaker != speaker:
                stats["conflict"] += 1
                stats["conflict_items"].append({
                    "index": unit.index, "text": unit.text[:60],
                    "rule": unit.rule, "rule_speaker": unit.speaker,
                    "llm_speaker": speaker, "agreement": verdict["agreement"],
                    "votes": verdict["votes"],
                })
            else:
                stats["agree"] += 1
        unit.agreement = verdict["agreement"]
        unit.votes = verdict["votes"]
    return stats


def stats(units: list[Unit]) -> dict:
    dialogues = [unit for unit in units if unit.kind == "dialogue"]
    attributed = [unit for unit in dialogues if unit.speaker]
    by_rule: dict[str, int] = {}
    for unit in dialogues:
        by_rule[unit.rule or "e"] = by_rule.get(unit.rule or "e", 0) + 1
    same_pairs = []
    for prev, cur in zip(dialogues, dialogues[1:]):
        if prev.speaker and prev.speaker == cur.speaker:
            same_pairs.append({"prev": prev.index, "cur": cur.index,
                               "speaker": cur.speaker, "rule": cur.rule,
                               "gap": cur.index - prev.index - 1})
    adjacent = [item for item in same_pairs if item["gap"] <= 1]
    return {
        "dialogue_units": len(dialogues),
        "attributed": len(attributed),
        "unattributed": len(dialogues) - len(attributed),
        "attribution_rate": round(len(attributed) / len(dialogues), 4) if dialogues else 1.0,
        "by_rule": dict(sorted(by_rule.items())),
        "same_speaker_pairs": len(same_pairs),
        "same_speaker_adjacent": len(adjacent),
        "same_speaker_far": len(same_pairs) - len(adjacent),
        "same_speaker_inferred": sum(1 for item in same_pairs if item["rule"] in ("a2", "c1", "c2", "d")),
        "same_speaker_detail": same_pairs,
    }
