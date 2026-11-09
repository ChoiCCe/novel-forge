"""导出：M4B（含章节标记）与基础 SRT。

SRT 采用“分段文本 + 实际音频时长”的粗对齐；精确对轴需要 ASR 强制对齐，放到 P2。
"""
from __future__ import annotations

import logging
import re
import shutil
import wave
from pathlib import Path

from .postprocess import resolve_ffmpeg, _run, wav_info

log = logging.getLogger("novel-forge.export")


def _ts(ms: int) -> str:
    ms = max(0, int(ms))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def write_srt(entries: list[dict], path: str | Path, offset_ms: int = 0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, item in enumerate(entries, 1):
        start = item["start_ms"] + offset_ms
        end = start + max(1, item["duration_ms"])
        text = str(item["text"]).strip()
        lines.extend([str(i), f"{_ts(start)} --> {_ts(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _escape_metadata(value: str) -> str:
    return (value.replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;")
            .replace("#", "\\#").replace("\n", " "))


def safe_chapter_name(title: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(title)).strip().strip(".")
    return cleaned[:80] or "未命名"


def chapter_mp3_name(index: int, title: str) -> str:
    """ABS 可直接入库的零填充命名：第001章 名实.mp3（去掉标题自带的“第X章”）。"""
    body = re.sub(r"^第[0-9零一二三四五六七八九十百千]+章\s*", "", str(title)).strip()
    return f"第{index:03d}章 {safe_chapter_name(body)}.mp3"


def write_chapter_mp3(wav_path: str | Path, out_path: str | Path, cfg: dict) -> Path:
    """单章 MP3：默认 48kbps 单声道，流式编码，不载入 PCM。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg(cfg["postprocess"].get("ffmpeg_path"))
    bitrate = cfg["export"].get("mp3_bitrate", "48k")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(wav_path), "-vn", "-c:a", "libmp3lame", "-b:a", bitrate,
        "-ac", "1", "-ar", str(cfg["tts"]["sample_rate"]), str(out_path),
    ])
    return out_path


def _concat_quote(path: str | Path) -> str:
    text = str(Path(path).resolve()).replace("\\", "/")
    return "'" + text.replace("'", "'\\''") + "'"


def write_m4b_from_wavs(wav_paths: list[str | Path], chapters: list[dict],
                        out_path: str | Path, cfg: dict, title: str,
                        pause_ms: int = 0) -> Path:
    """书级 M4B：concat demuxer 流式封装 + 章节标记，不载入整本 PCM。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".ffmeta.txt")
    lines = [";FFMETADATA1", f"title={_escape_metadata(title)}", "artist=novel-forge"]
    for chapter in chapters:
        lines.extend([
            "[CHAPTER]", "TIMEBASE=1/1000",
            f"START={int(chapter['start_ms'])}",
            f"END={max(int(chapter['end_ms']), int(chapter['start_ms']) + 1)}",
            f"title={_escape_metadata(chapter['title'])}",
        ])
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sample_rate = int(cfg["tts"]["sample_rate"])
    quiet_dir = out_path.parent / f".silence_{out_path.stem}"
    entries: list[Path] = []
    for i, path in enumerate(wav_paths):
        if i > 0 and pause_ms > 0:
            quiet = quiet_dir / f"gap_{i:05d}.wav"
            quiet.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(quiet), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(b"\x00\x00" * int(sample_rate * pause_ms / 1000))
            entries.append(quiet)
        entries.append(Path(path))
    list_path = out_path.with_suffix(".concat.txt")
    list_path.write_text("".join(f"file {_concat_quote(item)}\n" for item in entries),
                         encoding="utf-8")
    ffmpeg = resolve_ffmpeg(cfg["postprocess"].get("ffmpeg_path"))
    bitrate = cfg["export"].get("m4b_bitrate", "64k")
    try:
        _run([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-i", str(meta_path), "-map", "0:a", "-map_metadata", "1",
            "-c:a", "aac", "-b:a", bitrate, "-movflags", "+faststart", "-f", "mp4",
            str(out_path),
        ])
    finally:
        list_path.unlink(missing_ok=True)
        shutil.rmtree(quiet_dir, ignore_errors=True)
        meta_path.unlink(missing_ok=True)
    log.info("书级 M4B 已生成：%s（%.1f MB）", out_path, out_path.stat().st_size / 1024 / 1024)
    return out_path


def write_m4b(wav_path: str | Path, chapters: list[dict], out_path: str | Path,
              cfg: dict, title: str) -> Path:
    """chapters: [{"title","start_ms","end_ms"}]。"""
    wav_path = Path(wav_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".ffmeta.txt")
    lines = [";FFMETADATA1", f"title={_escape_metadata(title)}", "artist=novel-forge"]
    for chapter in chapters:
        lines.extend([
            "[CHAPTER]", "TIMEBASE=1/1000",
            f"START={int(chapter['start_ms'])}",
            f"END={max(int(chapter['end_ms']), int(chapter['start_ms']) + 1)}",
            f"title={_escape_metadata(chapter['title'])}",
        ])
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ffmpeg = resolve_ffmpeg(cfg["postprocess"].get("ffmpeg_path"))
    bitrate = cfg["export"].get("m4b_bitrate", "64k")
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(wav_path), "-i", str(meta_path),
        "-map", "0:a", "-map_metadata", "1",
        "-c:a", "aac", "-b:a", bitrate, "-movflags", "+faststart", "-f", "mp4",
        str(out_path),
    ])
    log.info("M4B 已生成：%s（%.1f MB）", out_path, out_path.stat().st_size / 1024 / 1024)
    return out_path


def m4b_duration_ms(wav_path: str | Path) -> int:
    return wav_info(wav_path)["duration_ms"]
