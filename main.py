#!/usr/bin/env python3
"""novel-forge CLI：txt 进 -> 剧本 -> 分段音频 -> 响度归一 -> M4B/SRT。

示例：
    python main.py --book book.txt --chapters 0-1
    python main.py --book book.txt --chapters 0 --script-only
    python main.py --book book.txt --chapters 0-1 --full          # 不限流，整章合成
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import export, ingest, narrator_lock, postprocess, script, tts_router  # noqa: E402
from core.caster import Caster  # noqa: E402
from core.llm import LLMClient, LLMError  # noqa: E402
from core.pipeline import process_chapter, save_chapter_script  # noqa: E402

log = logging.getLogger("novel-forge")


def load_config() -> dict:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_chapters(spec: str, count: int) -> list[int]:
    if spec.strip().lower() in ("all", "*"):
        return list(range(count))
    indexes: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indexes.extend(range(int(start), int(end) + 1))
        else:
            indexes.append(int(part))
    unique = sorted(set(indexes))
    invalid = [i for i in unique if i < 0 or i >= count]
    if invalid:
        raise SystemExit(f"章节下标越界：{invalid}，本书共 {count} 章（0-{count - 1}）")
    return unique


def run_audio(cfg: dict, chapter, jobs: list, book_dir: Path, limits: dict, force: bool,
              audio_tag: str = "", *, registry) -> dict:
    stats = tts_router.render_jobs(jobs, book_dir, cfg, registry, limits, force=force,
                                   audio_tag=audio_tag)
    usable = [job for job in jobs if job.audio_path]
    result = {"stats": stats, "timeline": [], "wav": None, "srt": None, "duration_ms": 0,
              "skipped_incremental": False}
    if not usable:
        log.warning("[%s] 没有可用音频", chapter.title)
        return result

    suffix = f"_{audio_tag}" if audio_tag else ""
    output_dir = book_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = output_dir / f"ch{chapter.index:03d}{suffix}_raw.wav"
    final = output_dir / f"ch{chapter.index:03d}{suffix}.wav"
    srt_path = output_dir / f"ch{chapter.index:03d}{suffix}.srt"
    mp3_dir = output_dir / (cfg["export"].get("mp3_dir", "mp3") + (f"_{audio_tag}" if audio_tag else ""))
    mp3_path = mp3_dir / export.chapter_mp3_name(chapter.index, chapter.title)
    script_path = book_dir / "scripts" / f"ch{chapter.index:03d}.json"
    marker = output_dir / f"ch{chapter.index:03d}{suffix}.export.json"
    script_hash = hashlib.sha1(script_path.read_bytes()).hexdigest() if script_path.exists() else ""
    marker_data = {}
    if marker.exists():
        try:
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            marker_data = {}

    # 增量：成品存在 + 剧本未变 + 本次全部命中缓存 → 不重跑导出
    reusable = (final.exists() and not force
                and stats.get("ok", 0) == 0 and stats.get("failed", 0) == 0
                and stats.get("failed_skipped", 0) == 0
                and bool(script_hash) and marker_data.get("script_sha1") == script_hash)
    if reusable:
        duration = postprocess.wav_info(final)["duration_ms"]
        timeline = postprocess.plan_jobs(usable, cfg)
        result.update({
            "timeline": timeline, "wav": str(final),
            "srt": str(srt_path) if srt_path.exists() else None,
            "duration_ms": duration, "skipped_incremental": True,
        })
        log.info("[%s] 命中增量：跳过重新导出（%s）", chapter.title, final.name)
    else:
        if postprocess.streaming_enabled(cfg):
            timeline = postprocess.concat_jobs_streaming(usable, raw, cfg)
        else:
            timeline = postprocess.concat_jobs(usable, raw, cfg)
        if cfg["postprocess"].get("loudnorm_two_pass", True):
            postprocess.normalize_loudness_two_pass(raw, final, cfg)
        else:
            postprocess.normalize_loudness(raw, final, cfg)
        raw.unlink(missing_ok=True)
        result.update({
            "timeline": timeline, "wav": str(final), "duration_ms": postprocess.wav_info(final)["duration_ms"],
        })
        if cfg["export"].get("srt"):
            export.write_srt(timeline, srt_path)
            result["srt"] = str(srt_path)

    if cfg["export"].get("mp3", True):
        if not reusable or not mp3_path.exists():
            export.write_chapter_mp3(final, mp3_path, cfg)
        result["mp3"] = str(mp3_path)
    # 单章 M4B 仍保留（带 tag 或不带 tag 均按 export.m4b）
    if cfg["export"].get("m4b", True):
        marker = [{"title": chapter.title, "start_ms": 0, "end_ms": result["duration_ms"]}]
        m4b_file = output_dir / f"ch{chapter.index:03d}{suffix}.m4b"
        result["m4b"] = str(export.write_m4b(final, marker, m4b_file, cfg, chapter.title))
        m4b_file.with_suffix(".ffmeta.txt").unlink(missing_ok=True)
    if script_hash:
        (output_dir / f"ch{chapter.index:03d}{suffix}.export.json").write_text(
            json.dumps({"script_sha1": script_hash, "duration_ms": result["duration_ms"],
                        "audio_tag": audio_tag}, ensure_ascii=False),
            encoding="utf-8")
    return result


def write_chapter_status(book_dir: Path, chapter, status: str, payload: dict) -> Path:
    """每章完成即写状态文件，供外层按章扫描触发。"""
    path = book_dir / "output" / f"ch{chapter.index:03d}.status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "book_id": chapter.book_id,
        "chapter_index": chapter.index,
        "title": chapter.title,
        "status": status,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    data.update(payload)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_demo(cfg: dict, book_id: str, chapter_runs: list[dict], book_dir: Path,
               tag: str = "") -> dict:
    """把各章成品拼成一个短 demo，并写章节标记 M4B + 偏移后的 SRT。"""
    runs = [run for run in chapter_runs if (run.get("audio") or {}).get("wav")]
    if not runs:
        return {}
    outputs_dir = (PROJECT_ROOT / cfg["paths"]["outputs"]).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    first, last = runs[0]["chapter"].index, runs[-1]["chapter"].index
    suffix = f"_{tag}" if tag else ""
    stem = f"demo_{book_id}_ch{first}-{last}{suffix}"
    demo_wav = outputs_dir / f"{stem}.wav"
    pause = int(cfg["postprocess"].get("pause_ms_between_chapters", 800))
    sr = int(cfg["tts"]["sample_rate"])
    wavs = [run["audio"]["wav"] for run in runs]
    entries: list[dict] = []
    markers: list[dict] = []
    offset = 0
    for i, run in enumerate(runs):
        if i > 0:
            offset += pause
        marker_start = offset
        for entry in run["audio"]["timeline"]:
            entries.append(dict(entry, start_ms=entry["start_ms"] + offset))
        offset += postprocess.wav_info(run["audio"]["wav"])["duration_ms"]
        markers.append({"title": run["chapter"].title, "start_ms": marker_start, "end_ms": offset})

    if postprocess.streaming_enabled(cfg):
        total = postprocess.concat_files_streaming(wavs, demo_wav, pause, sr, cfg)
    else:
        import wave
        total = 0
        with wave.open(str(demo_wav), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sr)
            for i, path in enumerate(wavs):
                if i > 0:
                    output.writeframes(b"\x00\x00" * int(sr * pause / 1000))
                    total += pause
                with wave.open(path, "rb") as src:
                    output.writeframes(src.readframes(src.getnframes()))
                total += postprocess.wav_info(path)["duration_ms"]

    srt = export.write_srt(entries, outputs_dir / f"{stem}.srt") if cfg["export"].get("srt") else None
    m4b_path = outputs_dir / f"{stem}.m4b"
    if postprocess.streaming_enabled(cfg):
        m4b = export.write_m4b_from_wavs(wavs, markers, m4b_path, cfg,
                                         runs[0]["chapter"].book_id, pause_ms=pause)
    else:
        m4b = export.write_m4b(demo_wav, markers, m4b_path, cfg, runs[0]["chapter"].book_id)
    if cfg["export"].get("book_m4b", False):
        book_dir.joinpath("output").mkdir(parents=True, exist_ok=True)
        export.write_m4b_from_wavs(wavs, markers,
                                   book_dir / "output" / f"book_{book_id}{suffix}.m4b",
                                   cfg, runs[0]["chapter"].book_id, pause_ms=pause)
    return {"wav": str(demo_wav), "srt": str(srt) if srt else None, "m4b": str(m4b), "duration_ms": total}


def probe_models(cfg: dict) -> int:
    """逐个探测 model_chain 中的模型，结果落盘并打印；有不可用模型返回 1。"""
    llm = LLMClient(cfg, PROJECT_ROOT)
    results = llm.probe_models()
    out = (PROJECT_ROOT / cfg["paths"]["cache"] / "model_probe.json").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 模型可用性探测 ===")
    for item in results:
        if item["ok"]:
            print(f"[可用] {item['model']} | {item['latency_s']}s | {item['reply']}")
        else:
            print(f"[不可用] {item['model']} | {item['latency_s']}s | {item['error']}")
    print(f"探测结果：{out}")
    return 0 if all(item["ok"] for item in results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="小说 txt -> 台本 -> 音频 -> M4B")
    parser.add_argument("--book", default=None, help="小说 txt 路径（--probe-models 时可省略）")
    parser.add_argument("--chapters", default=None,
                        help="章节下标，如 0-1 / 0,2 / all；仅给 --force-* 时默认不额外选中第 0 章")
    parser.add_argument("--chapter", type=int, default=None, help="只处理单章（等价于 --chapters N）")
    parser.add_argument("--force-chapter", type=int, default=None,
                        help="只强制重跑指定单章（含剧本与 TTS），其他章命中原有产物")
    parser.add_argument("--force-tts-chapter", type=int, default=None,
                        help="只强制重做指定单章的 TTS/导出，剧本缓存与 cast 保持不变（不调用 LLM）")
    parser.add_argument("--book-id", default="",
                        help="覆盖自动生成的 book_id（重做时复用原书目录，避免产物与门户脱节）")
    parser.add_argument("--script-only", action="store_true", help="只生成剧本与任务列表，不合成")
    parser.add_argument("--skip-tts", action="store_true", help="同 --script-only")
    parser.add_argument("--probe-models", action="store_true", help="只探测 model_chain 各模型可用性")
    parser.add_argument("--show-cast", action="store_true", help="打印当前角色表、年龄阶段与换音色历史")
    parser.add_argument("--webui", action="store_true", help="启动本地 WebUI（127.0.0.1）")
    parser.add_argument("--port", type=int, default=8765, help="WebUI 端口，默认 8765")
    parser.add_argument("--no-proofread", action="store_true", help="跳过二次校对")
    parser.add_argument("--limit-narration", type=int, default=None,
                        help="本次最多合成多少段旁白（0=全部跳过，默认取 config）")
    parser.add_argument("--limit-dialogue", type=int, default=None,
                        help="本次最多合成多少句对白（0=全部跳过，默认取 config）")
    parser.add_argument("--full", action="store_true", help="不限额，合成所选章节全部分段")
    parser.add_argument("--force", action="store_true", help="忽略剧本/音频缓存，全部重跑")
    parser.add_argument("--workers", type=int, default=None, help="章节并发上限（默认 config 值）")
    parser.add_argument("--no-demo", action="store_true", help="不拼 demo M4B")
    parser.add_argument("--single-voice", action="store_true", help="对照组：全章统一旁白声线")
    parser.add_argument("--prune-audio", action="store_true", help="合成后删除本章旧 hash 的孤儿音频")
    parser.add_argument("--demo-tag", default="", help="demo 文件名后缀，避免覆盖，如 single")
    parser.add_argument("--audio-tag", default="", help="音频/字幕文件名后缀，如 v2（不覆盖旧成品）")
    parser.add_argument("--override-narrator-lock", action="store_true",
                        help="旁白已锁定但配置被改动时，仍允许本次运行")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    cfg = load_config()
    lock_state = narrator_lock.check(cfg, override=args.override_narrator_lock)
    if lock_state.get("override"):
        log.warning("旁白参数已锁定但配置被改动，按 --override-narrator-lock 继续：%s",
                    lock_state.get("diff") or "fingerprint 不一致")
    if args.webui:
        from webui.server import serve
        serve(PROJECT_ROOT, cfg, args.port)
        return
    if args.show_cast:
        print(Caster(cfg, None, PROJECT_ROOT).show_cast_text())
        return
    if args.probe_models:
        raise SystemExit(probe_models(cfg))
    if not args.book:
        parser.error("--book 必填（仅 --probe-models 可省略）")

    if args.no_proofread:
        cfg["script"]["proofread"] = False
    if args.full:
        limits = {"narration": None, "dialogue": None}
    else:
        sample = cfg["tts"].get("sample_limits", {})
        limits = {
            "narration": args.limit_narration if args.limit_narration is not None else int(sample.get("narration", 0)),
            "dialogue": args.limit_dialogue if args.limit_dialogue is not None else int(sample.get("dialogue", 0)),
        }

    registry = tts_router.EngineRegistry(cfg, PROJECT_ROOT)

    book_path = Path(args.book)
    if not book_path.is_absolute():
        book_path = (PROJECT_ROOT / book_path).resolve()
    if not book_path.exists():
        raise SystemExit(f"找不到小说：{book_path}")

    chapters = ingest.split_chapters(book_path, cfg["ingest"])
    base_spec = args.chapters
    if (base_spec is None and args.chapter is None and args.force_chapter is None
            and args.force_tts_chapter is None):
        base_spec = "0"  # 兼容旧用法：不指定任何章节参数时只跑第 0 章
    selected = ([chapters[i] for i in parse_chapters(base_spec, len(chapters))]
                if base_spec else [])
    forced: set[int] = set()
    tts_forced: set[int] = set()
    if args.chapter is not None:
        selected = [chapters[i] for i in parse_chapters(str(args.chapter), len(chapters))]
    if args.force_chapter is not None:
        forced = set(parse_chapters(str(args.force_chapter), len(chapters)))
    if args.force_tts_chapter is not None:
        tts_forced = set(parse_chapters(str(args.force_tts_chapter), len(chapters)))
    have = {ch.index for ch in selected}
    selected.extend(chapters[i] for i in sorted((forced | tts_forced) - have))
    selected.sort(key=lambda ch: ch.index)
    if not selected:
        raise SystemExit("未选择任何章节")

    book_id = args.book_id or chapters[0].book_id
    if any(sep in book_id for sep in ("/", "\\")) or book_id in (".", ".."):
        raise SystemExit(f"--book-id 非法：{book_id}")
    for chapter in chapters:
        chapter.book_id = book_id
    book_dir = (PROJECT_ROOT / cfg["paths"]["novels"] / book_id).resolve()
    raw_dir = book_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "chapters.json").write_text(
        json.dumps([ch.to_dict() for ch in chapters], ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("《%s》共 %s 章，本次处理：%s", book_id, len(chapters),
             ", ".join(f"{ch.index}:{ch.title}" for ch in selected))

    llm = LLMClient(cfg, PROJECT_ROOT)
    caster = Caster(cfg, llm, PROJECT_ROOT)
    known_names = caster.registry_names()
    for chapter in selected:
        known_names.extend(script.extract_names(chapter.text, limit=8))
    known_names = list(dict.fromkeys(known_names))

    def work(chapter):
        """逐章隔离失败：单章抛错不影响其他章节，失败原因原样带回。"""
        try:
            return {"ok": True, "chapter": chapter,
                    "result": process_chapter(chapter, cfg, llm, caster, known_names,
                                              args.force or chapter.index in forced,
                                              args.single_voice,
                                              registry=registry)}
        except Exception as err:
            failures = getattr(err, "failures", [])
            partial = getattr(err, "partial_segments", [])
            log.error("[%s] 剧本失败：%s", chapter.title, err)
            for item in failures:
                log.error("  - %s: %s", item["model"], item["error"])
            return {"ok": False, "chapter": chapter, "error": f"{type(err).__name__}: {err}",
                    "failures": failures, "partial_segments": [s.to_dict() for s in partial]}

    workers = max(1, min(args.workers or cfg["llm"]["concurrency"], len(selected)))
    t0 = time.time()
    processed = llm.map(work, selected, workers=workers)
    log.info("剧本阶段结束，用时 %.1fs", time.time() - t0)

    chapter_runs: list[dict] = []
    limits_left = dict(limits)
    failed = 0
    for outcome in processed:
        chapter = outcome["chapter"]
        script_path = book_dir / "scripts" / f"ch{chapter.index:03d}.json"
        if not outcome["ok"]:
            failed += 1
            fail_path = book_dir / "scripts" / f"ch{chapter.index:03d}.failed.json"
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            fail_path.write_text(json.dumps({
                "book_id": book_id,
                "chapter_index": chapter.index,
                "title": chapter.title,
                "status": "failed",
                "error": outcome["error"],
                "model_failures": outcome["failures"],
                "partial_segments": outcome["partial_segments"],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            chapter_runs.append({"chapter": chapter, "failed": True, "error": outcome["error"],
                                 "failures": outcome["failures"], "script_path": str(fail_path), "audio": {}})
            write_chapter_status(book_dir, chapter, "failed",
                                 {"error": outcome["error"], "script": str(fail_path),
                                  "failures": outcome["failures"]})
            continue

        item = outcome["result"]
        save_chapter_script(cfg, chapter, item["record"], item["jobs"], book_dir)
        log.info("[%s] 剧本已保存：%s", chapter.title, script_path)
        run = {"chapter": chapter, "failed": False, "script": item, "audio": {},
               "script_path": str(script_path)}
        if not (args.script_only or args.skip_tts):
            audio_tag = args.audio_tag or ("single" if args.single_voice else "")
            audio = run_audio(cfg, chapter, item["jobs"], book_dir, limits_left,
                              args.force or chapter.index in forced or chapter.index in tts_forced,
                              audio_tag=audio_tag, registry=registry)
            if args.prune_audio:
                folder = f"ch{chapter.index:03d}" + (f"_{audio_tag}" if audio_tag else "")
                tts_router.prune_audio(book_dir / "audio" / folder, item["jobs"])
            used = {kind: sum(1 for job in item["jobs"]
                              if job.type == kind and job.status in ("ok", "cached", "cache_hit", "fallback_ok"))
                    for kind in ("narration", "dialogue")}
            for kind in limits_left:
                if limits_left[kind] is not None:
                    limits_left[kind] = max(0, limits_left[kind] - used[kind])
            run["audio"] = audio
            save_chapter_script(cfg, chapter, item["record"], item["jobs"], book_dir)
        audio_info = run["audio"]
        write_chapter_status(book_dir, chapter,
                             "ok" if audio_info.get("wav") else "script_only",
                             {"script": str(script_path),
                              "segments": len(item["segments"]), "jobs": len(item["jobs"]),
                              "wav": audio_info.get("wav"), "mp3": audio_info.get("mp3"),
                              "srt": audio_info.get("srt"),
                              "duration_ms": audio_info.get("duration_ms", 0),
                              "stats": audio_info.get("stats", {})})
        chapter_runs.append(run)

    demo = {}
    if not (args.script_only or args.skip_tts) and not args.no_demo:
        demo = build_demo(cfg, book_id, chapter_runs, book_dir,
                          tag=args.demo_tag or args.audio_tag or ("single" if args.single_voice else ""))

    summary = {
        "book_id": book_id,
        "book_path": str(book_path),
        "model_chain": list(cfg["llm"]["model_chain"]),
        "llm_stats": llm.stats,
        "chapters": [
            {
                "index": run["chapter"].index,
                "title": run["chapter"].title,
                "status": "failed" if run.get("failed") else "ok",
                "error": run.get("error"),
                "failures": run.get("failures", []),
                "segments": len(run["script"]["segments"]) if run.get("script") else 0,
                "coverage": run["script"]["record"]["coverage"] if run.get("script") else {},
                "jobs": len(run["script"]["jobs"]) if run.get("script") else 0,
                "audio": run["audio"].get("stats", {}),
                "wav": run["audio"].get("wav"),
                "srt": run["audio"].get("srt"),
                "duration_ms": run["audio"].get("duration_ms", 0),
            }
            for run in chapter_runs
        ],
        "demo": demo,
        "limits": limits,
        "failed_chapters": failed,
    }
    summary_path = book_dir / "output" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== novel-forge 运行摘要 ===")
    for chapter in summary["chapters"]:
        if chapter["status"] == "failed":
            print(f"[{chapter['index']}] {chapter['title']} | 失败：{chapter['error']}")
            continue
        cov = chapter["coverage"]
        print(f"[{chapter['index']}] {chapter['title']} | 片段 {chapter['segments']} | "
              f"覆盖率 {cov['final_ratio'] * 100:.2f}% | 音频 {chapter['audio']} | "
              f"{chapter['duration_ms'] / 1000:.1f}s")
    stats = llm.stats
    print(f"LLM JSON 解析成功 {stats['json_success']}/{stats['json_calls']}，链路失败 {stats['chain_failures']} 次")
    for model, entry in stats["models"].items():
        print(f"  {model}: 调用 {entry['calls']}，成功 {entry['success']}，失败 {entry['failures']}"
              + (f"，最后错误 {entry['last_error']}" if entry["last_error"] else ""))
    if demo:
        print(f"demo M4B: {demo['m4b']}（{demo['duration_ms'] / 1000:.1f}s）")
    print(f"剧本/汇总: {book_dir}")
    if failed:
        print(f"有 {failed} 个章节失败，可用 --chapters 重跑（已完成的章节会命中缓存）")
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except (LLMError, RuntimeError, UnicodeDecodeError) as err:
        log.error("运行失败：%s", err)
        sys.exit(1)
