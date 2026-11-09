"""后台任务执行器：单 worker，状态全部落 SQLite，支持暂停/继续/取消/失败重试。

复用 core 管线（core.pipeline.process_chapter / tts_router.render_jobs / postprocess / export），
不复制业务逻辑，也不反向依赖 CLI 入口；只额外做四段进度落库与单章 M4B 封装。
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path

from core import export, ingest, postprocess, tts_router
from core.caster import Caster
from core.llm import LLMClient
from core.pipeline import process_chapter, save_chapter_script
from core.script import extract_names

log = logging.getLogger("novel-forge.webui.runner")


def sync_cast(db, caster: Caster, book_id: str) -> None:
    """把 registry.json 的 v2 阶段档案镜像到 SQLite，供界面显示/编辑。"""
    rows = []
    for entry in caster.iter_profiles():
        rows.append({
            "key": entry.get("key", ""),
            "name": entry.get("name", ""),
            "gender": entry.get("gender", ""),
            "age_stage": entry.get("age_stage") or entry.get("age_band") or caster.default_stage,
            "edge_voice": entry.get("edge_voice", ""),
            "rate": entry.get("rate", "+0%"),
            "pitch": entry.get("pitch", "+0Hz"),
            "volume": entry.get("volume", "+0%"),
            "temperament": entry.get("temperament", "中性"),
            "prosody_source": entry.get("prosody_source", "temperament"),
            "age_note": entry.get("age_note", ""),
            "age_confidence": entry.get("age_confidence"),
            "first_seen_chapter": entry.get("first_seen_chapter"),
            "last_seen_chapter": entry.get("last_seen_chapter"),
            "updated_by": entry.get("updated_by", ""),
            "persona": entry.get("persona", ""),
        })
    db.replace_cast(book_id, rows)
    events = []
    for event in caster.voice_events():
        if event.get("book_id") not in (None, book_id):
            continue
        events.append({
            "chapter": event.get("chapter"),
            "name": event.get("name") or event.get("character_id", ""),
            "from_stage": event.get("from_stage", ""),
            "to_stage": event.get("to_stage", ""),
            "from_voice": event.get("from_voice", ""),
            "to_voice": event.get("to_voice", ""),
            "evidence": event.get("evidence", ""),
            "confidence": event.get("confidence"),
            "updated_by": event.get("source", ""),
        })
    db.replace_voice_changes(book_id, events)


class TaskRunner:
    def __init__(self, db, cfg: dict, root: Path, caster: Caster, llm: LLMClient):
        self.db = db
        self.cfg = cfg
        self.root = root
        self.caster = caster
        self.llm = llm
        self.registry = tts_router.EngineRegistry(cfg, root)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- 生命周期
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="forge-task-runner", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            task_id = self.db.claim_next_task()
            if task_id is None:
                self._stop.wait(1.0)
                continue
            try:
                self._run_task(task_id)
            except Exception as err:  # 兜底：任务级异常不拖垮 worker
                log.exception("任务 %s 执行异常", task_id)
                self.db.set_task_status(task_id, "failed", f"{type(err).__name__}: {err}",
                                        finished=True)

    def _control(self, task_id: int) -> str:
        status = self.db.task_status(task_id)
        if status in ("paused", "cancelled", "missing"):
            return "cancel" if status in ("cancelled", "missing") else "pause"
        return "run"

    # ---------------------------------------------------------------- 任务
    def _run_task(self, task_id: int) -> None:
        task = self.db.task(task_id)
        params = json.loads(task.get("params") or "{}")
        self.db.set_task_status(task_id, "running")
        book_id = task["book_id"]
        book_dir = (self.root / self.cfg["paths"]["novels"] / book_id).resolve()
        (book_dir / "output").mkdir(parents=True, exist_ok=True)

        known_names = list(self.caster.registry_names())
        for row in task["chapters"]:
            chapter = self.db.chapter(book_id, row["chapter_idx"])
            if chapter:
                known_names.extend(extract_names(chapter.get("text", ""), limit=8))
        known_names = list(dict.fromkeys(known_names))

        for row in task["chapters"]:
            if row["export_status"] == "done":
                continue
            if self._control(task_id) != "run":
                return
            outcome = self._run_chapter(task_id, task, row, params, book_dir, known_names)
            if outcome is True:
                self.db.bump_task_counts(task_id, done=1)
            elif outcome is False:
                self.db.bump_task_counts(task_id, failed=1)
            else:  # 暂停或取消
                return
            sync_cast(self.db, self.caster, book_id)

        current = self.db.task(task_id)
        status = "done" if current["failed_chapters"] == 0 else "failed"
        self.db.set_task_status(task_id, status, finished=True)
        log.info("任务 %s 结束：%s（成功 %s / 失败 %s）", task_id, status,
                 current["done_chapters"], current["failed_chapters"])

    def _run_chapter(self, task_id: int, task: dict, row: dict, params: dict,
                     book_dir: Path, known_names: list[str]) -> bool | None:
        book_id = task["book_id"]
        idx = int(row["chapter_idx"])
        full = self.db.chapter(book_id, idx)
        if not full:
            self.db.update_task_chapter(task_id, idx, script_status="failed", error="章节不存在")
            return False
        chapter = ingest.Chapter(
            book_id=book_id, index=idx, label=full.get("label", ""), title=full["title"],
            text=full.get("text", ""), char_count=int(full.get("char_count", 0)),
            line_count=full.get("text", "").count("\n") + 1,
        )
        force = bool(params.get("force"))
        single_voice = bool(params.get("single_voice"))
        director_flag = params.get("director")
        director_enabled = None if director_flag is None else bool(director_flag)

        # ---- 剧本
        self.db.update_task_chapter(task_id, idx, script_status="running", detail="生成剧本…",
                                    error="")
        try:
            result = process_chapter(chapter, self.cfg, self.llm, self.caster,
                                     known_names, force, single_voice,
                                     director_enabled=director_enabled,
                                     registry=self.registry)
        except Exception as err:
            log.error("[%s] 剧本失败：%s", chapter.title, err)
            self.db.update_task_chapter(task_id, idx, script_status="failed", coverage_status="failed",
                                        error=f"{type(err).__name__}: {err}", detail="剧本失败")
            return False

        record = result["record"]
        jobs = result["jobs"]
        coverage = record.get("coverage", {})
        save_chapter_script(self.cfg, chapter, record, jobs, book_dir)
        prior = {int(item["job_index"]): item for item in self.db.segments(book_id, idx)}
        for job in jobs:
            old = prior.get(job.index)
            if old and old.get("text") == job.text and old.get("status") == "failed":
                job.fail_count = int(old.get("fail_count") or 0)
                job.last_error_type = old.get("last_error_type") or ""
                job.last_error_at = old.get("last_error_at") or ""
                job.status = "failed"
                job.error = old.get("error") or ""
        self.db.replace_segments(book_id, idx, [job.to_dict() for job in jobs])
        self.db.update_task_chapter(
            task_id, idx, script_status="done", coverage_status="done",
            coverage_ratio=coverage.get("final_ratio"), tts_total=len(jobs),
            detail=f"剧本 {len(result['segments'])} 段 / {len(jobs)} 任务，覆盖率 "
                   f"{coverage.get('final_ratio', 0) * 100:.2f}%"
                   + (f"，导演 {result['director'].get('status')}"
                      if result.get("director", {}).get("enabled") else "，导演关闭"))
        sync_cast(self.db, self.caster, book_id)

        # ---- 合成（限额 None = 全量）
        total = len(jobs)
        self.db.update_task_chapter(task_id, idx, tts_status="running", tts_done=0,
                                    tts_total=total, detail=f"合成 0/{total}")

        def on_progress(job, stats):
            done_count = (stats["ok"] + stats["cached"] + stats["cache_hit"]
                          + stats["failed"] + stats["failed_skipped"] + stats["skipped_limit"])
            detail = (f"合成 {done_count}/{total}：新合成 {stats['ok']}，"
                      f"命中 {stats['cached'] + stats['cache_hit']}，"
                      f"失败 {stats['failed']}，达上限跳过 {stats['failed_skipped']}")
            self.db.update_task_chapter(task_id, idx, tts_done=done_count, detail=detail)
            self.db.upsert_segment(book_id, idx, job.to_dict())

        stats = tts_router.render_jobs(
            jobs, book_dir, self.cfg, self.registry,
            {"narration": None, "dialogue": None}, force=force,
            on_progress=on_progress,
            should_stop=lambda: self._control(task_id) != "run",
        )
        if stats.get("stopped"):
            self.db.update_task_chapter(task_id, idx, tts_status="pending",
                                        detail=f"已暂停，完成 {stats['ok'] + stats['cached'] + stats['cache_hit']}/{total}")
            return None
        usable = [job for job in jobs if job.audio_path]
        if not usable:
            self.db.update_task_chapter(task_id, idx, tts_status="failed",
                                        error="没有可用音频", detail=f"合成失败 {stats['failed']} 段")
            return False
        tts_failed = int(stats.get("failed", 0))
        skipped_failed = int(stats.get("failed_skipped", 0))
        self.db.update_task_chapter(
            task_id, idx, tts_status="done",
            detail=(f"合成 {len(usable)}/{total} 段（失败 {tts_failed}，"
                    f"连续失败达上限 {skipped_failed}）"),
            error=("" if not (tts_failed + skipped_failed)
                   else f"{tts_failed + skipped_failed} 段合成失败，"
                        f"其中 {skipped_failed} 段已达重试上限，可在分段表手动重试"))

        # ---- 导出
        if self._control(task_id) != "run":
            self.db.update_task_chapter(task_id, idx, tts_status="done")
            return None
        self.db.update_task_chapter(task_id, idx, export_status="running", detail="拼接/归一化/导出…")
        script_path = book_dir / "scripts" / f"ch{chapter.index:03d}.json"
        final_path = book_dir / "output" / f"ch{chapter.index:03d}.wav"
        marker_path = book_dir / "output" / f"ch{chapter.index:03d}.export.json"
        script_hash = hashlib.sha1(script_path.read_bytes()).hexdigest() if script_path.exists() else ""
        try:
            marker_data = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.exists() else {}
        except (json.JSONDecodeError, OSError):
            marker_data = {}
        reuse = (final_path.exists() and not force
                 and stats.get("ok", 0) == 0
                 and not (tts_failed + skipped_failed)
                 and bool(script_hash) and marker_data.get("script_sha1") == script_hash)
        try:
            audio = self._export_chapter(book_id, chapter, jobs, book_dir, reuse=reuse)
        except Exception as err:
            log.error("[%s] 导出失败：%s", chapter.title, err)
            self.db.update_task_chapter(task_id, idx, export_status="failed",
                                        error=f"{type(err).__name__}: {err}")
            return False
        self.db.update_task_chapter(
            task_id, idx, export_status="done",
            detail=(f"音频 {audio['duration_ms'] / 1000:.1f}s / WAV "
                    f"{audio['wav_size'] / 1024 / 1024:.2f}MB / MP3 "
                    f"{audio['mp3_size'] / 1024:.0f}KB / M4B "
                    f"{audio['m4b_size'] / 1024 / 1024:.2f}MB"
                    + ("；命中增量，跳过重新导出" if audio.get("reused") else "")
                    + (f"；覆盖 {len(audio['overwritten'])} 个旧文件"
                       if audio.get("overwritten") else "")))
        return True

    @staticmethod
    def _overwrite_info(path: Path) -> dict | None:
        """覆盖导出前记录旧文件信息，写入日志便于回溯被替换的成品。"""
        if not path.exists():
            return None
        stat = path.stat()
        old_mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        now_text = time.strftime("%Y-%m-%d %H:%M:%S")
        log.info("导出覆盖 %s（旧文件 mtime=%s size=%sB，覆盖时间=%s）",
                 path, old_mtime, stat.st_size, now_text)
        return {"path": str(path), "mtime": old_mtime, "size": stat.st_size,
                "overwritten_at": now_text}

    def _export_chapter(self, book_id: str, chapter, jobs: list, book_dir: Path,
                        reuse: bool = False) -> dict:
        cfg = self.cfg
        usable = [job for job in jobs if job.audio_path]
        output_dir = book_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        raw = output_dir / f"ch{chapter.index:03d}_raw.wav"
        final = output_dir / f"ch{chapter.index:03d}.wav"
        srt_path = output_dir / f"ch{chapter.index:03d}.srt"
        m4b_path = output_dir / f"ch{chapter.index:03d}.m4b"
        mp3_dir = output_dir / cfg["export"].get("mp3_dir", "mp3")
        mp3_path = mp3_dir / export.chapter_mp3_name(chapter.index, chapter.title)
        overwritten: list[dict] = []

        if reuse:
            timeline = postprocess.plan_jobs(usable, cfg)
            duration_ms = postprocess.wav_info(final)["duration_ms"]
            if cfg["export"].get("srt") and not srt_path.exists():
                export.write_srt(timeline, srt_path)
            if cfg["export"].get("mp3", True) and not mp3_path.exists():
                export.write_chapter_mp3(final, mp3_path, cfg)
            if cfg["export"].get("m4b", True) and not m4b_path.exists():
                export.write_m4b(final, [{"title": chapter.title, "start_ms": 0,
                                          "end_ms": duration_ms}], m4b_path, cfg, chapter.title)
            log.info("[%s] 命中增量：跳过重新导出（%s）", chapter.title, final.name)
        else:
            overwritten = [info for info in (self._overwrite_info(final),
                                             self._overwrite_info(srt_path),
                                             self._overwrite_info(m4b_path),
                                             self._overwrite_info(mp3_path))
                           if info]
            if postprocess.streaming_enabled(cfg):
                timeline = postprocess.concat_jobs_streaming(usable, raw, cfg)
            else:
                timeline = postprocess.concat_jobs(usable, raw, cfg)
            if cfg["postprocess"].get("loudnorm_two_pass", True):
                postprocess.normalize_loudness_two_pass(raw, final, cfg)
            else:
                postprocess.normalize_loudness(raw, final, cfg)
            raw.unlink(missing_ok=True)
            duration_ms = postprocess.wav_info(final)["duration_ms"]
            if cfg["export"].get("srt"):
                export.write_srt(timeline, srt_path)
            if cfg["export"].get("mp3", True):
                export.write_chapter_mp3(final, mp3_path, cfg)
            if cfg["export"].get("m4b", True):
                export.write_m4b(final, [{"title": chapter.title, "start_ms": 0,
                                          "end_ms": duration_ms}], m4b_path, cfg, chapter.title)
                m4b_path.with_suffix(".ffmeta.txt").unlink(missing_ok=True)

        result = {
            "wav": str(final), "wav_size": final.stat().st_size,
            "srt": str(srt_path) if srt_path.exists() else None,
            "srt_size": srt_path.stat().st_size if srt_path.exists() else 0,
            "m4b": str(m4b_path) if m4b_path.exists() else None,
            "m4b_size": m4b_path.stat().st_size if m4b_path.exists() else 0,
            "mp3": str(mp3_path) if mp3_path.exists() else None,
            "mp3_size": mp3_path.stat().st_size if mp3_path.exists() else 0,
            "duration_ms": duration_ms,
            "overwritten": overwritten,
            "reused": reuse,
        }
        self.db.record_output(book_id, chapter.index, "wav", str(final),
                              result["wav_size"], duration_ms)
        if result["srt"]:
            self.db.record_output(book_id, chapter.index, "srt", result["srt"],
                                  result["srt_size"], duration_ms)
        if result["m4b"]:
            self.db.record_output(book_id, chapter.index, "m4b", result["m4b"],
                                  result["m4b_size"], duration_ms)
        if result["mp3"]:
            self.db.record_output(book_id, chapter.index, "mp3", result["mp3"],
                                  result["mp3_size"], duration_ms)
        script_path = book_dir / "scripts" / f"ch{chapter.index:03d}.json"
        if script_path.exists():
            marker = output_dir / f"ch{chapter.index:03d}.export.json"
            marker.write_text(json.dumps({
                "script_sha1": hashlib.sha1(script_path.read_bytes()).hexdigest(),
                "duration_ms": duration_ms,
            }, ensure_ascii=False), encoding="utf-8")
        return result
