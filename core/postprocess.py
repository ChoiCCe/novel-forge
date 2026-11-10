"""音频后处理：格式统一、响度归一化、按序拼接与时间轴。

所有中间音频最终统一为 16-bit / 单声道 / 24kHz WAV，拼接用 Python wave 完成，
避免 Windows 下 ffmpeg concat 列表中中文路径转义问题；只有 loudnorm 与 M4B 封装调用 ffmpeg。
"""
from __future__ import annotations

import array
import json
import logging
import shutil
import subprocess
import wave
from pathlib import Path

log = logging.getLogger("novel-forge.postprocess")


def resolve_ffmpeg(explicit: str | None = None) -> str:
    if explicit and Path(explicit).exists():
        return explicit
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError("找不到 ffmpeg：可 pip install imageio-ffmpeg 或配置 postprocess.ffmpeg_path")


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败：{result.stderr.strip()[-400:]}")


def convert_to_wav(src: str | Path, dst: str | Path, sample_rate: int, ffmpeg: str | None = None) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run([
        resolve_ffmpeg(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", str(dst),
    ])
    return dst


def normalize_loudness(src: str | Path, dst: str | Path, cfg: dict) -> Path:
    pp = cfg["postprocess"]
    filter_str = pp.get("loudnorm", "I=-16:TP=-1.5:LRA=11")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".norm.wav")
    _run([
        resolve_ffmpeg(pp.get("ffmpeg_path")), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-af", f"loudnorm={filter_str}",
        "-ar", str(cfg["tts"]["sample_rate"]), "-ac", "1", "-c:a", "pcm_s16le", str(tmp),
    ])
    tmp.replace(dst)
    return dst


def normalize_loudness_two_pass(src: str | Path, dst: str | Path, cfg: dict) -> Path:
    """逐章两遍 loudnorm：先 measure 再 apply，避免一次性整本处理。"""
    pp = cfg["postprocess"]
    filter_str = pp.get("loudnorm", "I=-16:TP=-1.5:LRA=11")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg(pp.get("ffmpeg_path"))
    try:
        output = _run_output([
            ffmpeg, "-hide_banner", "-i", str(src),
            "-af", f"loudnorm={filter_str}:print_format=json", "-f", "null", "-",
        ])
        measured = json.loads(output[output.rindex("{"):output.rindex("}") + 1])
        second = (f"loudnorm={filter_str}:measured_I={measured['input_i']}"
                  f":measured_LRA={measured['input_lra']}"
                  f":measured_TP={measured['input_tp']}"
                  f":measured_thresh={measured['input_thresh']}"
                  f":offset={measured.get('target_offset', 0)}:linear=true")
        _run([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src), "-af", second,
            "-ar", str(cfg["tts"]["sample_rate"]), "-ac", "1", "-c:a", "pcm_s16le", str(dst),
        ])
        return dst
    except Exception as err:
        log.warning("两遍 loudnorm 失败，退回单遍：%s", err)
        return normalize_loudness(src, dst, cfg)


def concat_files_streaming(paths: list[str | Path], out_path: str | Path, pause_ms: int,
                           sample_rate: int, cfg: dict) -> int:
    """多章流式拼接（章间静音），返回总时长 ms；不载入整本 PCM。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quiet_dir = out_path.parent / f".silence_{out_path.stem}"
    quiet_cache: dict[int, Path] = {}
    entries: list[Path] = []
    for i, path in enumerate(paths):
        if i > 0 and pause_ms > 0:
            entries.append(_silence_cached(quiet_dir, pause_ms, sample_rate, quiet_cache))
        entries.append(Path(path))
    _ffmpeg_concat(entries, out_path, sample_rate, cfg)
    shutil.rmtree(quiet_dir, ignore_errors=True)
    total = sum(wav_info(path)["duration_ms"] for path in paths) + pause_ms * max(0, len(paths) - 1)
    return total


def wav_info(path: str | Path) -> dict:
    with wave.open(str(path), "rb") as handle:
        return {
            "channels": handle.getnchannels(),
            "sampwidth": handle.getsampwidth(),
            "framerate": handle.getframerate(),
            "frames": handle.getnframes(),
            "duration_ms": int(handle.getnframes() * 1000 / handle.getframerate()),
        }


def _frames(path: str | Path, sample_rate: int) -> bytes:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise RuntimeError(f"音频格式不是单声道16bit：{path}")
        if handle.getframerate() != sample_rate:
            raise RuntimeError(f"采样率不一致（{handle.getframerate()} != {sample_rate}）：{path}")
        return handle.readframes(handle.getnframes())


def _silence(sample_rate: int, ms: int) -> bytes:
    return b"\x00\x00" * int(sample_rate * ms / 1000)


def trim_edge_silence(path: str | Path, sample_rate: int, threshold: int = 60) -> int:
    """裁掉单个合成单元首尾的数字静音，返回裁掉的毫秒数。

    edge 每段输出常带 150-900ms 首尾静音；不裁的话，子句停顿和段间停顿会叠加到
    模板值的 2-3 倍，时间轴虽然没错但听感拖沓。阈值 60 约 -55dBFS，只裁纯静音。
    """
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        channels, width = handle.getnchannels(), handle.getsampwidth()
    if channels != 1 or width != 2:
        return 0
    samples = array.array("h")
    samples.frombytes(frames)
    start = 0
    while start < len(samples) and abs(samples[start]) < threshold:
        start += 1
    if start >= len(samples):
        return 0  # 整段静音，保持原样，避免产出 0 帧文件
    end = len(samples) - 1
    while end > start and abs(samples[end]) < threshold:
        end -= 1
    if start == 0 and end == len(samples) - 1:
        return 0
    removed_ms = (start + (len(samples) - 1 - end)) * 1000 // sample_rate
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples[start:end + 1].tobytes())
    return removed_ms


def streaming_enabled(cfg: dict) -> bool:
    return bool(cfg.get("postprocess", {}).get("streaming", True))


def _run_output(cmd: list[str]) -> str:
    """执行并返回合并后的 stderr/stdout；用于 loudnorm 第一遍测量。"""
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败：{result.stderr.strip()[-400:]}")
    return (result.stdout or "") + (result.stderr or "")


def _silence_wav(sample_rate: int, ms: int, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(sample_rate * ms / 1000)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)
    return path


def _silence_cached(quiet_dir: Path, ms: int, sample_rate: int, cache: dict) -> Path:
    """同一停顿长度只落盘一次，避免几百个临时静音文件拖慢导出。"""
    if ms not in cache:
        cache[ms] = _silence_wav(sample_rate, ms, quiet_dir / f"silence_{ms:05d}.wav")
    return cache[ms]


def _concat_quote(path: Path) -> str:
    """ffmpeg concat 列表的路径写法：正斜杠 + 单引号转义。"""
    text = str(Path(path).resolve()).replace("\\", "/")
    return "'" + text.replace("'", "'\\''") + "'"


def _ffmpeg_concat(entries: list[Path], out_path: Path, sample_rate: int, cfg: dict,
                   codec_args: list[str] | None = None) -> None:
    """用 concat demuxer 拼接同格式 wav，内存占用与总时长无关。"""
    list_path = out_path.with_suffix(out_path.suffix + ".concat.txt")
    list_path.write_text("".join(f"file {_concat_quote(item)}\n" for item in entries),
                         encoding="utf-8")
    ffmpeg = resolve_ffmpeg(cfg["postprocess"].get("ffmpeg_path"))
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", str(list_path)]
    cmd += codec_args or ["-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le"]
    cmd += [str(out_path)]
    try:
        _run(cmd)
    finally:
        list_path.unlink(missing_ok=True)


def plan_jobs(jobs: list, cfg: dict) -> list[dict]:
    """只读 wav 头计算时间轴，不拼接、不载入 PCM。"""
    from .prosody import pause_template, transition_pause

    template = pause_template(cfg)
    jobs = [job for job in jobs if job.audio_path]
    timeline: list[dict] = []
    cursor = template["chapter_start_ms"]
    for i, job in enumerate(jobs):
        if i > 0:
            pause = transition_pause(jobs[i - 1].type, job.type, cfg)
            cursor += pause
        duration_ms = wav_info(job.audio_path)["duration_ms"]
        timeline.append({
            "job_index": job.index, "start_ms": cursor, "duration_ms": duration_ms,
            "text": job.text, "speaker": job.speaker, "type": job.type, "emotion": job.emotion,
        })
        cursor += duration_ms
    return timeline


def concat_jobs_streaming(jobs: list, out_path: str | Path, cfg: dict) -> list[dict]:
    """常数内存版 concat_jobs：静音落盘 + ffmpeg concat demuxer。"""
    from .prosody import pause_template, transition_pause

    sample_rate = int(cfg["tts"]["sample_rate"])
    template = pause_template(cfg)
    jobs = [job for job in jobs if job.audio_path]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timeline = plan_jobs(jobs, cfg)

    quiet_dir = out_path.parent / f".silence_{out_path.stem}"
    quiet_cache: dict[int, Path] = {}
    entries: list[Path] = []
    if template["chapter_start_ms"] > 0:
        entries.append(_silence_cached(quiet_dir, template["chapter_start_ms"], sample_rate, quiet_cache))
    for i, job in enumerate(jobs):
        if i > 0:
            pause = transition_pause(jobs[i - 1].type, job.type, cfg)
            if pause > 0:
                entries.append(_silence_cached(quiet_dir, pause, sample_rate, quiet_cache))
        entries.append(Path(job.audio_path))
    if template["chapter_end_ms"] > 0:
        entries.append(_silence_cached(quiet_dir, template["chapter_end_ms"], sample_rate, quiet_cache))
    _ffmpeg_concat(entries, out_path, sample_rate, cfg)
    shutil.rmtree(quiet_dir, ignore_errors=True)
    return timeline


def concat_jobs(jobs: list, out_path: str | Path, cfg: dict) -> list[dict]:
    """按 job.index 顺序拼接已合成音频；返回每个 job 在成品里的时间轴。

    停顿模板由 tts.prosody.pause_template 提供：章首/旁白↔对白/段间/章末；
    所有静音都计入 cursor，SRT 与音频不会错位。
    """
    from .prosody import pause_template, transition_pause

    sample_rate = int(cfg["tts"]["sample_rate"])
    template = pause_template(cfg)
    jobs = [job for job in jobs if job.audio_path]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timeline: list[dict] = []
    cursor = 0
    with wave.open(str(out_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        if template["chapter_start_ms"] > 0:
            output.writeframes(_silence(sample_rate, template["chapter_start_ms"]))
            cursor += template["chapter_start_ms"]
        for i, job in enumerate(jobs):
            if i > 0:
                pause = transition_pause(jobs[i - 1].type, job.type, cfg)
                output.writeframes(_silence(sample_rate, pause))
                cursor += pause
            frames = _frames(job.audio_path, sample_rate)
            output.writeframes(frames)
            duration_ms = int(len(frames) / 2 * 1000 / sample_rate)
            timeline.append({
                "job_index": job.index, "start_ms": cursor, "duration_ms": duration_ms,
                "text": job.text, "speaker": job.speaker, "type": job.type, "emotion": job.emotion,
            })
            cursor += duration_ms
        if template["chapter_end_ms"] > 0:
            output.writeframes(_silence(sample_rate, template["chapter_end_ms"]))
            cursor += template["chapter_end_ms"]
    return timeline


def concat_with_pauses(paths: list[str | Path], pause_before: list[int],
                       out_path: str | Path, sample_rate: int) -> int:
    """按“每段前的静音”拼接（子句/强调词边界用）；返回总时长 ms。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_ms = 0
    with wave.open(str(out_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        for i, path in enumerate(paths):
            pause = int(pause_before[i] if i < len(pause_before) else 0)
            if pause > 0:
                output.writeframes(_silence(sample_rate, pause))
                total_ms += pause
            frames = _frames(path, sample_rate)
            output.writeframes(frames)
            total_ms += int(len(frames) / 2 * 1000 / sample_rate)
    return total_ms


def concat_files(paths: list[str | Path], out_path: str | Path, pause_ms: int, sample_rate: int) -> int:
    """拼接多个同格式 wav（章节间使用），返回总时长 ms。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_ms = 0
    with wave.open(str(out_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        for i, path in enumerate(paths):
            if i > 0:
                output.writeframes(_silence(sample_rate, pause_ms))
                total_ms += pause_ms
            frames = _frames(path, sample_rate)
            output.writeframes(frames)
            total_ms += int(len(frames) / 2 * 1000 / sample_rate)
    return total_ms
