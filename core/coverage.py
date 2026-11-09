"""覆盖率校验与缺口回喂补写。

覆盖率只在“内容字符”（汉字/字母/数字）上比对，忽略空白与标点：
- 原文 -> 剧本拼接文本，用 difflib 对齐；
- 未匹配的原文区间就是缺口，带上下文回喂 LLM 补写；
- 补写若仍不合格，就把缺口原文原样加成旁白，保证不漏字。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .segments import SEGMENTS_SCHEMA, Segment

log = logging.getLogger("novel-forge.coverage")

CONTENT_RE = re.compile(r"[0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff]")


def content_only(text: str) -> str:
    return "".join(ch.lower() if "A" <= ch <= "Z" else ch for ch in CONTENT_RE.findall(text))


def _content_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    mapping: list[int] = []
    for i, ch in enumerate(text):
        if CONTENT_RE.match(ch):
            chars.append(ch.lower() if "A" <= ch <= "Z" else ch)
            mapping.append(i)
    return "".join(chars), mapping


@dataclass
class Gap:
    start: int       # 原文 raw 起点
    end: int         # 原文 raw 终点（不含）
    text: str        # 原文缺口
    script_pos: int  # 对应剧本内容坐标（插入点=该位置之后）
    content_len: int


@dataclass
class CoverageReport:
    ratio: float
    precision: float
    matched: int
    total: int
    script_len: int
    gaps: list[Gap] = field(default_factory=list)

    def brief(self) -> dict:
        return {
            "ratio": round(self.ratio, 4),
            "precision": round(self.precision, 4),
            "matched": self.matched,
            "total": self.total,
            "gap_count": len(self.gaps),
            "gap_chars": sum(g.content_len for g in self.gaps),
        }


def measure(segments: list, source: str) -> CoverageReport:
    """segments: 任何带 .text 属性的对象列表（避免与 script.py 循环导入）。"""
    script = "".join(seg.text for seg in segments)
    script_norm, _ = _content_with_map(script)
    source_norm, source_map = _content_with_map(source)
    if len(source_norm) > 100_000:
        raise ValueError("coverage.measure 只接受单章文本（>10万字符会耗尽 difflib 内存），请按章调用")
    if not source_norm:
        return CoverageReport(1.0, 1.0, 0, 0, len(script_norm))

    matcher = SequenceMatcher(None, script_norm, source_norm, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    gaps: list[Gap] = []
    for tag, a1, a2, b1, b2 in matcher.get_opcodes():
        # a=剧本, b=原文：原文多出的部分对应 insert/replace；delete 是剧本多写（影响 precision）
        if tag in ("replace", "insert") and b2 > b1:
            raw_start = source_map[b1]
            raw_end = source_map[b2 - 1] + 1
            gap_text = source[raw_start:raw_end].strip()
            if gap_text:
                gaps.append(Gap(
                    start=raw_start, end=raw_end, text=gap_text,
                    script_pos=a2, content_len=len(content_only(gap_text)),
                ))

    total = len(source_norm)
    script_len = len(script_norm)
    return CoverageReport(
        ratio=matched / total,
        precision=matched / script_len if script_len else 1.0,
        matched=matched,
        total=total,
        script_len=script_len,
        gaps=gaps,
    )


def _insert_index(segments: list, script_pos: int) -> int:
    if script_pos <= 0:
        return 0
    cursor = 0
    for i, seg in enumerate(segments):
        cursor += len(content_only(seg.text))
        if cursor >= script_pos:
            return i + 1
    return len(segments)


def _split_gap(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts = re.split(r"(?<=[。！？；…])", text)
    chunks, current = [], ""
    for part in parts:
        if not part:
            continue
        if len(current) + len(part) <= limit:
            current += part
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks or [text]


def _repair_once(gap: Gap, chunk: str, source: str, llm, cfg: dict, roster: list[str]) -> list:
    """让 LLM 只补写缺口片段；输出不合格时原样落成旁白。"""
    emotions = cfg["script"]["emotions"]
    before = source[max(0, gap.start - 80):gap.start]
    after = source[gap.end:gap.end + 80]
    prompt = (
        "你是中文有声小说台本的补写器。只补写下面“缺失片段”，不要输出前后文，不要改写。\n"
        f"角色名单：{'、'.join(roster) or '无'}\n"
        f"前文（仅理解用）：{before}\n"
        f"缺失片段：{chunk}\n"
        f"后文（仅理解用）：{after}\n"
        '只输出 JSON：{"segments":[{"type":"narration|dialogue","speaker":"旁白或角色名",'
        f'"emotion":"{"|".join(emotions)}","text":"缺失片段中的原文"}}]}}'
    )
    try:
        data = llm.chat_json([
            {"role": "system", "content": "只输出一个 JSON 对象，不要解释、推理或 markdown。"},
            {"role": "user", "content": prompt},
        ], prefer=cfg["llm"].get("repair_model") or None,
            schema=SEGMENTS_SCHEMA, schema_name="segments")
        items = data.get("segments", []) if isinstance(data, dict) else data
        repaired = [
            Segment(
                type="dialogue" if item.get("type") == "dialogue" else "narration",
                speaker=str(item.get("speaker") or "未知角色") if item.get("type") == "dialogue" else "旁白",
                emotion=str(item.get("emotion") or "平静"),
                text=str(item.get("text", "")).strip(),
                repaired=True,
            )
            for item in items if isinstance(item, dict) and str(item.get("text", "")).strip()
        ]
        if repaired:
            check = measure(repaired, chunk)
            accept = cfg["coverage"]
            if (check.ratio >= float(accept["repair_accept_ratio"])
                    and check.precision >= float(accept["repair_accept_precision"])):
                return repaired
        log.warning("缺口补写不合格（%s），改用原文旁白兜底：%s", gap.text[:20], chunk[:30])
    except Exception as err:
        log.warning("缺口补写失败（%s），改用原文旁白兜底：%s", err, chunk[:30])
    return [Segment("narration", "旁白", "平静", chunk, repaired=True)]


def repair(segments: list, source: str, llm, cfg: dict, roster: list[str]) -> tuple[list, CoverageReport, int]:
    """从右往左插入补写片段，避免脚本坐标位移。返回 (新片段, 报告, 实际轮数)。"""
    cov_cfg = cfg["coverage"]
    threshold = float(cov_cfg["threshold"])
    max_rounds = int(cov_cfg["max_rounds"])
    max_gaps = int(cov_cfg["max_gaps_per_round"])
    gap_limit = int(cov_cfg["gap_max_chars"])
    min_gap = int(cov_cfg["min_gap_chars"])

    report = measure(segments, source)
    rounds = 0
    while rounds < max_rounds and report.ratio < threshold and report.gaps:
        candidates = [g for g in report.gaps if g.content_len >= min_gap]
        candidates.sort(key=lambda g: g.content_len, reverse=True)
        selected = candidates[:max_gaps]
        selected.sort(key=lambda g: g.script_pos, reverse=True)  # 从右往左插

        for gap in selected:
            pieces = []
            for chunk in _split_gap(gap.text, gap_limit):
                pieces.extend(_repair_once(gap, chunk, source, llm, cfg, roster))
            index = _insert_index(segments, gap.script_pos)
            segments[index:index] = pieces
        rounds += 1
        report = measure(segments, source)
        log.info("覆盖率补写第 %s 轮：%.2f%%（阈值 %.1f%%）", rounds, report.ratio * 100, threshold * 100)
    return segments, report, rounds
