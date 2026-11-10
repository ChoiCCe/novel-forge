"""多引擎路由 + 内容哈希缓存 + 断点续跑。

缓存 key = sha256(engine + text_hash + voice_profile_hash)，改一句只重跑一句；
同一条音频先落 cache/audio，再复制到 novels/{book}/audio/chXXX/ 供断点续跑。
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .engines.base import BaseEngine
from .engines.cosyvoice_engine import CosyVoiceEngine
from .engines.edge_engine import EdgeEngine
from .engines.kokoro_engine import KokoroEngine
from .engines.sherpa_engine import SherpaEngine
from .postprocess import concat_with_pauses, trim_edge_silence, wav_info
from . import prosody, tts_scheduler, voice_policy

log = logging.getLogger("novel-forge.tts")

# 合成链路版本：裁静音等行为变化时递增，让旧缓存自动失效
SYNTH_VERSION = 3

ENGINE_CLASSES = {"edge": EdgeEngine, "sherpa": SherpaEngine,
                  "cosyvoice": CosyVoiceEngine, "kokoro": KokoroEngine}
VOICE_HASH_KEYS = ("engine", "edge_voice", "kokoro_voice", "rate", "pitch", "volume", "sid",
                   "speed", "ref_wav", "ref_text", "class_name")


@dataclass
class TTSJob:
    index: int
    chapter_index: int
    chapter_title: str
    type: str
    speaker: str
    speaker_key: str
    emotion: str
    text: str
    requested_engine: str
    engine: str
    voice: dict
    speech_text: str = ""
    clause_count: int = 1
    piece_count: int = 1
    degraded: bool = False
    cache_key: str = ""
    audio_path: str | None = None
    duration_ms: int = 0
    start_ms: int = 0
    status: str = "pending"
    error: str = ""
    fail_count: int = 0
    last_error_type: str = ""
    last_error_at: str = ""

    def to_dict(self) -> dict:
        return {
            "index": self.index, "type": self.type, "speaker": self.speaker,
            "speaker_key": self.speaker_key, "emotion": self.emotion, "text": self.text,
            "speech_text": self.speech_text, "clause_count": self.clause_count,
            "piece_count": self.piece_count,
            "engine": self.engine, "requested_engine": self.requested_engine,
            "degraded": self.degraded, "cache_key": self.cache_key,
            "audio_path": self.audio_path, "duration_ms": self.duration_ms,
            "status": self.status, "error": self.error,
            "fail_count": self.fail_count, "last_error_type": self.last_error_type,
            "last_error_at": self.last_error_at,
        }


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def voice_fingerprint(voice: dict) -> str:
    payload = {key: voice.get(key) for key in VOICE_HASH_KEYS}
    return _sha(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def cache_key(engine: str, voice: dict, text: str, salt: str = "") -> str:
    """严格按“engine + 文本 hash + 音色 hash + 切分盐”组合，避免串音色或旧参数误命中。"""
    payload = f"{engine}|{_sha(text)}|{voice_fingerprint(voice)}"
    if salt:
        payload += f"|{salt}"
    return _sha(payload)


def prosody_salt(cfg: dict) -> str:
    """把切分/停顿配置写进缓存盐：改停顿配置也应重新合成。"""
    pros = cfg["tts"].get("prosody", {})
    payload = {
        "long": pros.get("long_sentence_chars"), "clause": pros.get("clause_pause_ms"),
        "rate_clamp": pros.get("rate_clamp"),
        "synth": SYNTH_VERSION,
    }
    return _sha(json.dumps(payload, ensure_ascii=False, sort_keys=True))


class EngineRegistry:
    def __init__(self, cfg: dict, project_root: Path):
        self.cfg = cfg
        self.root = project_root
        self.sample_rate = int(cfg["tts"]["sample_rate"])
        self.ffmpeg = cfg["postprocess"].get("ffmpeg_path")
        self.engines: dict[str, BaseEngine] = {}
        for name, engine_cfg in cfg["tts"]["engines"].items():
            if not engine_cfg.get("enabled"):
                continue
            engine_cfg = dict(engine_cfg, _ffmpeg=self.ffmpeg)
            cls = ENGINE_CLASSES[engine_cfg["type"]]
            engine = cls(engine_cfg, self.sample_rate, project_root)
            # 引擎只拿到自身配置段；调度器需要完整配置（tts.scheduler 与单书覆盖）
            engine.global_cfg = cfg
            self.engines[name] = engine
        self._availability: dict[tuple, tuple[bool, str]] = {}
        lex = cfg["tts"].get("lexicon")
        if lex:
            lex_path = Path(lex)
            if not lex_path.is_absolute():
                lex_path = project_root / lex_path
            self.lexicon = prosody.load_lexicon(lex_path)
        else:
            self.lexicon = {}

    def available(self, name: str, voice: dict) -> tuple[bool, str]:
        key = (name, voice.get("ref_wav"))
        if key not in self._availability:
            engine = self.engines.get(name)
            self._availability[key] = engine.is_available(voice) if engine else (False, "引擎未启用")
        return self._availability[key]

    def resolve(self, preferred: str, fallbacks: list[str], voice: dict) -> tuple[str, str]:
        for name in dict.fromkeys([preferred] + list(fallbacks)):
            if not name:
                continue
            ok, reason = self.available(name, voice)
            if ok:
                return name, ""
        return "", "、".join(f"{n}:{self.available(n, voice)[1]}" for n in dict.fromkeys([preferred] + list(fallbacks)))


def unknown_speaker_profile(cfg: dict, narrator: dict, speaker: str = "",
                            slot: int = 0, variants: list[dict] | None = None) -> dict:
    """没有角色档案的说话人：旁白声线 + 独立参数档（四级阶梯第 4 级）。

    多个未知说话人同章出现时，按首次出现顺序领取不同参数档，避免“声线+参数”组合重复。
    """
    profile = dict(narrator)
    values: dict = {}
    if variants and 0 <= slot < len(variants):
        values = variants[slot]
    else:
        override = (cfg.get("casting") or {}).get("unknown_speaker") or {}
        for key in ("rate", "pitch", "volume"):
            if override.get(key):
                values[key] = str(override[key])
    for key in ("rate", "pitch", "volume"):
        if values.get(key):
            profile[key] = str(values[key])
    profile["prosody_source"] = "unknown_speaker"
    profile["voice_source"] = "unknown_fallback"
    profile["ladder_level"] = 4
    profile["key"] = f"unknown:{speaker}" if speaker else "unknown_speaker"
    profile["name"] = speaker or "未知说话人"
    return profile


def build_jobs(segments: list, cast_data: dict, chapter_index: int, chapter_title: str, cfg: dict,
               registry: EngineRegistry, single_voice: bool = False) -> list[TTSJob]:
    narrator = cast_data["narrator"]
    narrator_name = str(narrator.get("name") or "旁白")
    speakers = cast_data.get("speakers", {})
    cast = cast_data.get("cast", {})
    narration_fb = cfg["tts"].get("narration_fallbacks", ["edge"])
    dialogue_fb = cfg["tts"].get("dialogue_fallbacks", ["edge"])

    # 未知说话人按首次出现顺序领取独立参数档，避免同章重复“旁白声线+参数”组合
    unknown_order: list[str] = []
    if not single_voice:
        for seg in segments:
            if seg.type != "dialogue":
                continue
            key = speakers.get(seg.speaker)
            if key and cast.get(key):
                continue
            if seg.speaker not in unknown_order:
                unknown_order.append(seg.speaker)
    unknown_slots = {name: index for index, name in enumerate(unknown_order)}
    unknown_variants = (voice_policy.unknown_variants(cfg, narrator, len(unknown_order))
                        if unknown_order else [])

    jobs: list[TTSJob] = []
    unknown_speakers: set[str] = set()
    narrator_speaker_mismatch: list[int] = []
    for seg_index, seg in enumerate(segments):
        degraded = False
        if seg.type == "narration" and seg.speaker != narrator_name:
            narrator_speaker_mismatch.append(seg_index)
        if single_voice:
            # 对照组：全章统一旁白声线，只保留情绪韵律
            profile, requested, fallbacks = narrator, narrator["engine"], narration_fb
        elif seg.type == "dialogue":
            key = speakers.get(seg.speaker)
            profile = cast.get(key) if key else None
            if profile is None:
                # 未知角色降级旁白声线，但参数走独立档位，避免与旁白同参
                slot = unknown_slots.get(seg.speaker, 0)
                profile, degraded = unknown_speaker_profile(
                    cfg, narrator, seg.speaker, slot=slot, variants=unknown_variants), True
                if seg.speaker not in unknown_speakers:
                    unknown_speakers.add(seg.speaker)
                    log.warning("[%s] 说话人“%s”没有角色档案，降级为旁白声线（独立参数档 %s/%s/%s）",
                                chapter_title, seg.speaker, profile.get("rate"),
                                profile.get("pitch"), profile.get("volume"))
            requested = profile.get("engine", narrator["engine"])
            fallbacks = dialogue_fb
        else:
            profile, requested, fallbacks = narrator, narrator["engine"], narration_fb

        # Persona 基础 + 情绪（按强度权重），由 compose_prosody 合成
        voice = dict(profile)
        voice.update(prosody.combine_voice(profile, seg.emotion, seg.text, cfg))
        speech_text = prosody.prepare_text(seg.text, cfg, registry.lexicon)
        salt = prosody_salt(cfg)
        clause_count = len(prosody.split_clauses(speech_text, cfg))
        piece_count = len(prosody.plan_pieces(speech_text, cfg))

        resolved, reason = registry.resolve(requested, fallbacks, voice)
        if resolved:
            voice["engine"] = resolved
            degraded = degraded or resolved != requested
        else:
            voice["engine"] = requested

        jobs.append(TTSJob(
            index=len(jobs), chapter_index=chapter_index, chapter_title=chapter_title,
            type=seg.type, speaker=narrator_name if seg.type == "narration" else seg.speaker,
            speaker_key=(narrator.get("key") or narrator_name) if seg.type == "narration"
            else profile.get("key", narrator_name),
            emotion=seg.emotion, text=seg.text,
            speech_text=speech_text,
            clause_count=clause_count, piece_count=piece_count,
            requested_engine=requested, engine=resolved, voice=voice, degraded=degraded,
            cache_key=cache_key(resolved, voice, speech_text, salt) if resolved else "",
            status="pending" if resolved else "failed",
            error="" if resolved else reason,
        ))
    if narrator_speaker_mismatch:
        log.warning("[%s] %s 段 type=narration 的 speaker 不是“%s”（段号 %s），"
                    "渲染层仍强制使用旁白声线", chapter_title, len(narrator_speaker_mismatch),
                    narrator_name, narrator_speaker_mismatch[:8])
    return jobs


def synthesize_job(engine, job: TTSJob, voice: dict, cfg: dict, dest: Path,
                    chapter_dir: Path) -> None:
    """把一段文本按子句切分后分别合成，再按静音拼成单段音频。

    子句间 120-200ms；合成结果仍是一段音频，时长自然包含子段与静音，时间轴按整段累计。
    """
    text = job.speech_text or job.text
    pieces = prosody.plan_pieces(text, cfg)
    job.piece_count = len(pieces)
    sample_rate = int(cfg["tts"]["sample_rate"])
    if len(pieces) == 1:
        engine.synthesize(pieces[0]["text"], voice, job.emotion, dest)
        trim_edge_silence(dest, sample_rate)
        return
    parts: list[Path] = []
    pauses: list[int] = []
    try:
        for i, piece in enumerate(pieces):
            part = chapter_dir / f".part_{job.index:04d}_{i:02d}.wav"
            engine.synthesize(piece["text"], voice, job.emotion, part)
            trim_edge_silence(part, sample_rate)
            parts.append(part)
            pauses.append(int(piece["pause_before_ms"]))
        concat_with_pauses(parts, pauses, dest, sample_rate)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)


def classify_error(message: str) -> str:
    """把引擎错误归类为可展示的短类型，供 WebUI 标注。"""
    text = str(message or "")
    lower = text.lower()
    if "noaudi" in lower or "no audio" in lower or "空音频" in text or "empty" in lower:
        return "no_audio"
    if "403" in text or "forbidden" in lower:
        return "forbidden"
    if "429" in text or "too many requests" in lower or "rate limit" in lower:
        return "rate_limited"
    if "timeout" in lower or "timed out" in lower or "超时" in text:
        return "timeout"
    if "connect" in lower or "network" in lower or "dns" in lower or "ssl" in lower:
        return "network"
    return "engine"


def _render_one(job: TTSJob, book_dir: Path, cfg: dict, registry: EngineRegistry,
                cache_root: Path, retries: int, force: bool, audio_tag: str,
                stats: dict, stats_lock: threading.Lock, on_progress) -> None:
    """合成单个任务；stats 与进度回调由调用方加锁串行化。"""
    suffix = f"_{audio_tag}" if audio_tag else ""
    chapter_dir = book_dir / "audio" / f"ch{job.chapter_index:03d}{suffix}"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    fallbacks = cfg["tts"].get(("dialogue_fallbacks" if job.type == "dialogue" else "narration_fallbacks"), ["edge"])
    chain = list(dict.fromkeys([job.engine] + list(fallbacks)))
    done = False
    last_error = ""

    for engine_name in chain:
        ok, reason = registry.available(engine_name, job.voice)
        if not ok:
            last_error = reason
            continue
        key = (cache_key(engine_name, job.voice, job.speech_text or job.text,
                         prosody_salt(cfg)) or job.cache_key)
        filename = f"{job.index:04d}_{job.type}_{engine_name}_{key[:8]}.wav"
        dest = chapter_dir / filename
        cached = cache_root / key[:2] / f"{key}.wav"
        voice = dict(job.voice, engine=engine_name)

        if not force and dest.exists():
            job.audio_path, job.engine, job.cache_key = str(dest), engine_name, key
            job.duration_ms = wav_info(dest)["duration_ms"]
            job.status = "cached"
            job.fail_count, job.last_error_type, job.last_error_at = 0, "", ""
            job.error = ""
            job.degraded = job.degraded or engine_name != job.requested_engine
            with stats_lock:
                stats["cached"] += 1
            done = True
            break
        if not force and cached.exists():
            shutil.copy2(cached, dest)
            job.audio_path, job.engine, job.cache_key = str(dest), engine_name, key
            job.duration_ms = wav_info(dest)["duration_ms"]
            job.status = "cache_hit"
            job.fail_count, job.last_error_type, job.last_error_at = 0, "", ""
            job.error = ""
            job.degraded = job.degraded or engine_name != job.requested_engine
            with stats_lock:
                stats["cache_hit"] += 1
            done = True
            break

        tmp = chapter_dir / f".part_{job.index:04d}.wav"
        for attempt in range(retries + 1):
            try:
                synthesize_job(registry.engines[engine_name], job, voice, cfg, tmp, chapter_dir)
                cached.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tmp, cached)
                shutil.copy2(tmp, dest)
                tmp.unlink(missing_ok=True)
                job.audio_path, job.engine, job.cache_key = str(dest), engine_name, key
                job.duration_ms = wav_info(dest)["duration_ms"]
                job.status = "ok" if engine_name == job.requested_engine else "fallback_ok"
                job.degraded = job.degraded or engine_name != job.requested_engine
                job.error = ""
                job.fail_count, job.last_error_type, job.last_error_at = 0, "", ""
                with stats_lock:
                    stats["ok"] += 1
                done = True
                break
            except Exception as err:
                last_error = str(err)
                log.warning("分段 %s/%s %s 第 %s 次失败：%s",
                            job.index, job.chapter_index, engine_name, attempt + 1, err)
        if done:
            break

    if not done:
        job.status = "failed"
        job.error = last_error or "无可用引擎"
        job.fail_count = int(job.fail_count or 0) + 1
        job.last_error_type = classify_error(job.error)
        job.last_error_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with stats_lock:
            stats["failed"] += 1
        log.error("分段 %s 合成失败（连续第 %s 次，%s）：%s",
                  job.index, job.fail_count, job.last_error_type, job.error)
    if on_progress:
        with stats_lock:
            on_progress(job, stats)


def render_jobs(jobs: list[TTSJob], book_dir: Path, cfg: dict, registry: EngineRegistry,
                limits: dict, force: bool = False, audio_tag: str = "",
                on_progress=None, should_stop=None, book_id: str | None = None) -> dict:
    """按小样限额合成；已完成/命中缓存的分段跳过（断点续跑）。

    TTS 请求统一经过 core.tts_scheduler：并发与间隔来自 config.yaml + 单书覆盖。
    并发为 1 时与旧行为完全一致（顺序执行）；>1 时用线程池，进度回调串行落库。
    on_progress(job, stats) 每个任务结束后回调一次，供 WebUI 落库进度；
    should_stop() 返回 True 时在当前任务结束后退出（暂停/取消），未完成的任务保持 pending。
    """
    cache_root = (registry.root / cfg["paths"]["cache"] / "audio").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    retries = int(cfg["tts"].get("retries", 2))
    max_failures = int(cfg["tts"]["max_segment_failures"])
    stats = {"ok": 0, "cached": 0, "cache_hit": 0, "skipped_limit": 0, "failed": 0,
             "failed_skipped": 0, "bytes": 0, "stopped": 0}
    seen = {"narration": 0, "dialogue": 0}
    stats_lock = threading.Lock()
    effective_book = book_id or book_dir.name
    params = tts_scheduler.resolve_params(cfg, registry.root, effective_book)
    concurrency = params["concurrency"] if params["enabled"] else 1

    queue: list[TTSJob] = []
    for job in jobs:
        if should_stop and should_stop():
            stats["stopped"] = 1
            break
        if job.status == "failed" and int(job.fail_count or 0) >= max_failures and not force:
            # 连续失败达到配置阈值：停止自动重试，保留失败现场供 WebUI 标注/手动重试
            stats["failed_skipped"] += 1
            if on_progress:
                on_progress(job, stats)
            continue
        if job.status == "failed":
            job.status = "pending"
        limit = limits.get(job.type)
        # 语义：None = 不限额；0 = 本次跳过该类；正整数 = 最多合成多少段
        if limit is not None and seen[job.type] >= int(limit):
            job.status = "skipped_limit"
            stats["skipped_limit"] += 1
            if on_progress:
                on_progress(job, stats)
            continue
        seen[job.type] += 1
        queue.append(job)

    args = (book_dir, cfg, registry, cache_root, retries, force, audio_tag,
            stats, stats_lock, on_progress)
    with tts_scheduler.book_context(effective_book):
        if concurrency <= 1 or len(queue) <= 1:
            for job in queue:
                if should_stop and should_stop():
                    stats["stopped"] = 1
                    break
                _render_one(job, *args)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(contextvars.copy_context().run, _render_one, job, *args)
                           for job in queue]
                for future in futures:
                    try:
                        future.result()
                    except Exception as err:  # noqa: BLE001
                        log.error("并发合成任务异常：%s", err)
                        with stats_lock:
                            stats["failed"] += 1
    return stats


def prune_audio(chapter_dir: Path, jobs: list[TTSJob]) -> int:
    """删除本章 audio 目录里未被当前任务引用的旧 hash 文件，避免目录越滚越大。

    cache/audio 中的正式缓存不删除，误删后下次运行会从缓存秒级恢复。
    """
    chapter_dir = Path(chapter_dir)
    if not chapter_dir.exists():
        return 0
    keep = {Path(job.audio_path).name for job in jobs if job.audio_path}
    removed = 0
    for path in chapter_dir.iterdir():
        if path.is_file() and path.name not in keep and path.suffix in (".wav", ".part"):
            path.unlink(missing_ok=True)
            removed += 1
    if removed:
        log.info("清理孤儿音频 %s 个：%s", removed, chapter_dir)
    return removed
