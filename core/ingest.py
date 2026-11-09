"""章节切分与清洗：把一本小说的纯文本变成结构化章节。

参考 ebook2audiobook 的 `第X章 / 楔子` 标题正则，但额外处理：
- 知轩藏书等站点广告行、全角空格、分隔线；
- BOM / GB18030 兜底解码；
- `min_chapter_chars` 以下的误匹配标题并入下一章。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Chapter:
    book_id: str
    index: int
    label: str          # "楔子" / "第一章"
    title: str          # 完整标题行
    text: str           # 清洗后的正文（段落间保留单个换行）
    char_count: int     # 去空白后的字符数，用于覆盖率分母
    line_count: int

    def to_dict(self) -> dict:
        return asdict(self)


_SEPARATOR_RE = re.compile(r"^[=\-_*＊·—\s　]{6,}$")
_SPACE_RE = re.compile(r"[\s　]+")


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("无法解码文件，请确认是 UTF-8 或 GBK 文本")


def make_book_id(path: Path) -> str:
    """书名 + 路径短哈希：同名不同书不会互相覆盖。"""
    stem = path.stem
    match = re.search(r"《(.+?)》", stem)
    title = match.group(1) if match else stem
    title = re.sub(r"[\\/:*?\"<>|《》]", "", title)[:40].strip()
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:6]
    return f"{title}_{digest}"


def clean_text(raw_text: str, drop_patterns: list[str]) -> list[str]:
    """逐行清洗：去广告、去分隔线、统一空白；返回保留的段落列表。"""
    patterns = [re.compile(p, re.IGNORECASE) for p in drop_patterns]
    lines: list[str] = []
    for line in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.replace("\u3000", " ").strip()
        if not stripped:
            if lines and lines[-1] == "":
                continue
            lines.append("")
            continue
        if _SEPARATOR_RE.match(stripped):
            continue
        if any(p.search(stripped) for p in patterns):
            continue
        lines.append(stripped)
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def split_chapters(path: str | Path, cfg: dict) -> list[Chapter]:
    """按标题正则切章；无匹配时整本作为一章，避免静默丢内容。"""
    path = Path(path)
    raw = path.read_bytes()
    text = _decode(raw)
    heading_re = re.compile(cfg["heading_pattern"])
    min_chars = int(cfg.get("min_chapter_chars", 200))
    lines = clean_text(text, cfg.get("drop_line_patterns", []))

    heads: list[tuple[int, str, str, str]] = []
    for i, line in enumerate(lines):
        match = heading_re.match(line)
        if not match:
            continue
        label, rest = match.group(1), match.group(2).strip()
        title = f"{label} {rest}".strip()
        heads.append((i, label, rest, title))

    book_id = make_book_id(path)
    chapters: list[Chapter] = []
    if not heads:
        body = "\n".join(line for line in lines if line)
        chapters.append(Chapter(
            book_id=book_id, index=0, label="全文", title=path.stem,
            text=body, char_count=len(_SPACE_RE.sub("", body)), line_count=len(lines),
        ))
        return chapters

    for k, (start, label, rest, title) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        body_lines = [line for line in lines[start + 1:end] if line]
        body = "\n".join(body_lines)
        char_count = len(_SPACE_RE.sub("", body))
        if char_count < min_chars and k + 1 < len(heads):
            # 误匹配（目录行/空章）并入下一章正文，避免丢字
            heads[k + 1] = (start, heads[k + 1][1], heads[k + 1][2], heads[k + 1][3])
            continue
        chapters.append(Chapter(
            book_id=book_id, index=len(chapters), label=label, title=title,
            text=body, char_count=char_count, line_count=len(body_lines),
        ))
    return chapters
