"""章节管线：剧本生成 → 导演层 → TTS 任务构建。

从 CLI 入口抽出，供 main.py 与 webui.runner 共用。引擎注册表由调用方显式注入，
不再依赖 `main.ENGINE_REGISTRY` 这类模块级可变全局状态。
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from . import coverage, script, tts_router

ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger("novel-forge.pipeline")


def _script_cache_path(cfg: dict, text_hash: str) -> Path:
    root = (ROOT / cfg["paths"]["cache"] / "scripts").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root / text_hash[:2] / f"{text_hash}.json"


def _text_hash(chapter, cfg: dict) -> str:
    """章节内容 + 模型链 + 提示词版本 + 覆盖率阈值 共同决定剧本缓存。"""
    payload = (
        f"{chapter.text}|{json.dumps(cfg['llm']['model_chain'], ensure_ascii=False)}"
        f"|{cfg['llm'].get('repair_model')}|{cfg['llm'].get('review_model')}"
        f"|{cfg['script'].get('prompt_version')}|{cfg['coverage']['threshold']}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cached(cfg: dict, text_hash: str, force: bool) -> dict | None:
    path = _script_cache_path(cfg, text_hash)
    if force or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("剧本缓存损坏，忽略：%s", path)
        return None


def _save_cached(cfg: dict, text_hash: str, record: dict) -> None:
    path = _script_cache_path(cfg, text_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _segments_from_dicts(items: list[dict]) -> list:
    return [
        script.Segment(
            type=item.get("type", "narration"),
            speaker=item.get("speaker", "旁白"),
            emotion=item.get("emotion", "平静"),
            text=item.get("text", ""),
            repaired=bool(item.get("repaired")),
        )
        for item in items if item.get("text")
    ]


def process_chapter(chapter, cfg: dict, llm, caster, known_names: list[str], force: bool,
                    single_voice: bool = False, reuse_cast: bool = True, *, registry) -> dict:
    """生成单章剧本；任何环节失败都抛错（不做规则降级），由调用方标记章节失败。

    缓存命中默认只读（reuse_cast=True）：跳过 ensure_registered / refresh_cast /
    年龄推进，不写 registry、不改声线绑定；只有未命中缓存（确实重建本章）时才允许副作用。
    """
    threshold = float(cfg["coverage"]["threshold"])
    min_precision = float(cfg["coverage"].get("min_precision", 0.0))
    text_hash = _text_hash(chapter, cfg)
    cached = _load_cached(cfg, text_hash, force)
    if cached:
        ratio = float(cached.get("coverage", {}).get("final_ratio", 0))
        if cached.get("segments") and ratio >= threshold:
            segments = _segments_from_dicts(cached["segments"])
            cast_data = cached["cast"]
            if not reuse_cast:
                caster.ensure_registered(cast_data)
                cast_data = caster.refresh_cast(cast_data, chapter.index, chapter.book_id)  # 换引擎/换声线映射后仍用当前配置
            record = cached
            # 旧快照里可能残留已移除的导演层结果，读入即丢弃，避免继续写回磁盘
            record.pop("director", None)
            log.info("[%s] 命中剧本缓存（%s 段，覆盖率 %.2f%%）", chapter.title, len(segments), ratio * 100)
        else:
            log.warning("[%s] 剧本缓存覆盖率 %.2f%% 低于阈值 %.1f%%，重新生成",
                        chapter.title, ratio * 100, threshold * 100)
            cached = None
    if not cached:
        t0 = time.time()
        segments = script.chapter_to_segments(chapter.text, llm, cfg, known_names)
        before = coverage.measure(segments, chapter.text)
        log.info("[%s] 初版剧本 %s 段，覆盖率 %.2f%%", chapter.title, len(segments), before.ratio * 100)

        roster = list(dict.fromkeys(known_names + script.extract_names(chapter.text) + caster.registry_names()))
        segments, repaired_report, rounds = coverage.repair(segments, chapter.text, llm, cfg, roster)

        proofread_ok = False
        if cfg["script"].get("proofread", True):
            segments, proofread_ok = script.proofread(segments, chapter.text, llm, cfg)
        segments = script.merge_adjacent(segments, int(cfg["script"].get("max_segment_chars", 150)))
        final_report = coverage.measure(segments, chapter.text)

        # 写盘/合成前的硬校验：不达标直接失败，避免低质量剧本流入音频
        if final_report.ratio < threshold:
            raise script.ScriptError(
                f"[{chapter.title}] 覆盖率 {final_report.ratio * 100:.2f}% 未达阈值 {threshold * 100:.1f}%"
                f"（缺口 {final_report.brief()['gap_chars']} 字符），拒绝写盘")
        if min_precision and final_report.precision < min_precision:
            raise script.ScriptError(
                f"[{chapter.title}] 精确率 {final_report.precision * 100:.2f}% 低于 {min_precision * 100:.1f}%"
                "（剧本含原文之外的内容），拒绝写盘")

        cast_data = caster.build(chapter.title, segments, chapter.book_id, chapter.index)
        record = {
            "book_id": chapter.book_id,
            "chapter_index": chapter.index,
            "label": chapter.label,
            "title": chapter.title,
            "text_hash": text_hash,
            "model_chain": list(cfg["llm"]["model_chain"]),
            "prompt_version": cfg["script"].get("prompt_version"),
            "segments": [seg.to_dict() for seg in segments],
            "coverage": {
                "initial_ratio": round(before.ratio, 4),
                "final_ratio": round(final_report.ratio, 4),
                "precision": round(final_report.precision, 4),
                "threshold": threshold,
                "repair_rounds": rounds,
                "proofread_applied": proofread_ok,
                "gap_chars": final_report.brief()["gap_chars"],
            },
            "cast": cast_data,
            "elapsed_s": round(time.time() - t0, 1),
        }
        _save_cached(cfg, text_hash, record)
        log.info("[%s] 最终 %s 段，覆盖率 %.2f%%，校对=%s，耗时 %.1fs",
                 chapter.title, len(segments), final_report.ratio * 100,
                 "采纳" if proofread_ok else "未采纳/跳过", record["elapsed_s"])

    # 写盘/合成前的语料级校验：narration 段的 speaker 必须是配置旁白名，
    # 防止旧缓存或人工编辑把旁白段指到某个角色，导致渲染层拿人物声线读旁白。
    narrator_name = str(cfg["script"].get("narrator_name", "旁白"))
    guard_mode = str(cfg["script"].get("narrator_guard", "fix"))
    segments, narration_guard = script.enforce_narrator_speakers(segments, narrator_name, guard_mode)
    record["narration_guard"] = narration_guard
    if narration_guard["fixed"]:
        log.warning("[%s] 旁白段 speaker 强制修正 %s/%s 段（期望“%s”）：%s",
                    chapter.title, narration_guard["fixed"], narration_guard["checked"],
                    narrator_name,
                    "；".join(f"#{i['index']} {i['from']}->{narrator_name}"
                             for i in narration_guard["items"][:3]))
        record["segments"] = [seg.to_dict() for seg in segments]
        _save_cached(cfg, text_hash, record)

    # 年龄推进：只在本章有年龄/时间证据时调一次轻量抽取；失败或证据不足沿用上一阶段
    if not reuse_cast:
        cast_data = caster.update_ages(chapter, segments, cast_data)
    record["cast"] = cast_data
    if not cached:
        _save_cached(cfg, text_hash, record)

    # 语音/字幕按更细粒度切分（旧缓存里可能存在超长合并段），保证单段不是几十秒的巨块
    max_chars = int(cfg["script"].get("max_segment_chars", 150))
    tts_segments = [piece for seg in segments for piece in script.split_long(seg, max_chars)]

    jobs = tts_router.build_jobs(tts_segments, cast_data, chapter.index, chapter.title,
                                 cfg, registry, single_voice=single_voice)
    return {"record": record, "segments": segments, "cast": cast_data, "jobs": jobs}


def save_chapter_script(cfg: dict, chapter, record: dict, jobs: list, book_dir: Path) -> Path:
    scripts_dir = book_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload["jobs"] = [job.to_dict() for job in jobs]
    payload["char_count"] = chapter.char_count
    path = scripts_dir / f"ch{chapter.index:03d}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
