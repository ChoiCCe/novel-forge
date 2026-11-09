"""本地 WebUI 服务：stdlib http.server + JSON API + 静态单页，只绑 127.0.0.1。

启动：python main.py --webui [--port 8765]
不引入 FastAPI/uvicorn，避免新增依赖；浏览器轮询任务进度，无 WebSocket。
"""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import re
import sys
import threading
import urllib.parse
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import casting, director as director_mod, ingest, postprocess, prosody, tts_router, tts_scheduler  # noqa: E402
from core.caster import Caster  # noqa: E402
from core.llm import LLMClient  # noqa: E402
from webui.db import Database, now  # noqa: E402
from webui.runner import TaskRunner, sync_cast  # noqa: E402

log = logging.getLogger("novel-forge.webui")

KIND_LABEL = {"wav": "WAV", "srt": "SRT", "m4b": "M4B"}


def effective_view(app: "App", profile: dict, emotion: str | None = None) -> dict:
    """角色编辑页展示的“生效参数”：与成品/试听共用同一套 compose_prosody。"""
    emotion = emotion or profile.get("default_emotion") or "平静"
    composed = prosody.compose_prosody(
        profile, emotion, "这是一段用于展示最终参数的参考语句。",
        prosody.default_direction(), app.cfg)
    return {
        "emotion": emotion,
        "rate": composed["formatted"]["rate"],
        "pitch": composed["formatted"]["pitch"],
        "volume": composed["formatted"]["volume"],
        "raw": {kind: round(composed["raw"][kind], 2) for kind in ("rate", "pitch", "volume")},
        "clamped": composed["clamped"],
    }


class HttpError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------- 应用状态
class App:
    def __init__(self, root: Path, cfg: dict, port: int):
        self.root = root
        self.cfg = cfg
        self.port = port
        self.data_dir = root / "data"
        self.import_dir = root / "imports"
        self.preview_dir = self.data_dir / "previews"
        self.static_dir = Path(__file__).resolve().parent / "static"
        for folder in (self.data_dir, self.import_dir, self.preview_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.data_dir / "forge.db")
        self.llm = LLMClient(cfg, root)
        self.caster = Caster(cfg, self.llm, root)
        self.registry = tts_router.EngineRegistry(cfg, root)
        self.runner = TaskRunner(self.db, cfg, root, self.caster, self.llm)
        self.books_dir = (root / cfg["paths"]["novels"]).resolve()
        self.outputs_dir = (root / cfg["paths"]["outputs"]).resolve()
        self.voices = self._voice_options()
        self._apply_settings()
        self.reset = self.db.reset_interrupted()
        if self.reset:
            log.info("上次有 %s 个任务中断，已标记为暂停（可在界面点击继续）", self.reset)
        self.runner.start()

    def _voice_options(self) -> list[str]:
        """只列 catalog 中 available=true 的声线；失效声线不进下拉（v4 步骤 1）。"""
        catalog = getattr(self.caster, "catalog", {}) or {}
        available = [v.get("id") for v in catalog.get("voices", []) if v.get("available")]
        if available:
            return sorted(set(available))
        edge = self.cfg["tts"]["engines"]["edge"]
        names = {edge.get("narrator_voice"), edge.get("narrator_voice_alt")}
        names.update(edge.get("voice_map", {}).values())
        for by_gender in edge.get("age_stage_voice_map", {}).values():
            names.update(by_gender.values())
        return sorted(name for name in names if isinstance(name, str) and name)

    def book_dir(self, book_id: str) -> Path:
        return self.books_dir / book_id

    # ---------------------------------------------------------------- 运行时设置
    def settings_view(self) -> dict:
        return {
            "director": {
                "enabled": bool(self.cfg["script"].get("director", {}).get("enabled", True)),
                "prompt_version": self.cfg["script"].get("director", {}).get("prompt_version", "d1"),
            },
            "pause_template": prosody.pause_template(self.cfg),
            "temperaments": list(prosody.TEMPERAMENTS),
            "paces": list(prosody.PACES),
            "volumes": list(prosody.VOLUMES),
            "emotions": list(self.cfg["tts"].get("emotion_style", {}).keys()),
        }

    def _apply_settings(self) -> None:
        """把 SQLite 里的界面设置叠加到运行时配置（不改 config.yaml）。"""
        stored = self.db.setting("director_enabled", "")
        if stored in ("0", "1"):
            self.cfg.setdefault("script", {}).setdefault("director", {})["enabled"] = stored == "1"
        raw = self.db.setting("pause_template", "")
        if raw:
            try:
                template = json.loads(raw)
            except json.JSONDecodeError:
                template = {}
            current = self.cfg.setdefault("tts", {}).setdefault("prosody", {})
            merged = dict(current.get("pause_template") or {})
            for key in ("chapter_start_ms", "narration_to_dialogue_ms",
                        "dialogue_to_narration_ms", "segment_gap_ms", "chapter_end_ms"):
                if key in template:
                    merged[key] = max(0, min(5000, int(template[key])))
            current["pause_template"] = merged

    def save_settings(self, body: dict) -> dict:
        if "director_enabled" in body:
            enabled = bool(body["director_enabled"])
            self.db.set_setting("director_enabled", "1" if enabled else "0")
            self.cfg.setdefault("script", {}).setdefault("director", {})["enabled"] = enabled
        if isinstance(body.get("pause_template"), dict):
            allowed = {key: max(0, min(5000, int(value)))
                       for key, value in body["pause_template"].items()
                       if key in ("chapter_start_ms", "narration_to_dialogue_ms",
                                  "dialogue_to_narration_ms", "segment_gap_ms", "chapter_end_ms")}
            current = self.cfg.setdefault("tts", {}).setdefault("prosody", {})
            merged = dict(current.get("pause_template") or {})
            merged.update(allowed)
            current["pause_template"] = merged
            self.db.set_setting("pause_template", json.dumps(merged, ensure_ascii=False))
        return self.settings_view()


APP: App | None = None


# ---------------------------------------------------------------------------- 工具
def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\r\n]", "_", name).strip()
    return cleaned[:120] or "upload.txt"


def _detect_encoding(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "unknown"


def _book_title(path: Path) -> str:
    match = re.search(r"《(.+?)》", path.stem)
    return match.group(1) if match else path.stem[:40]


def _media_url(path: str | Path) -> str:
    return "/media?path=" + urllib.parse.quote(str(path), safe="")


def _duration_of(path: Path) -> int | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            return int(handle.getnframes() * 1000 / handle.getframerate())
    except Exception:
        return None


def _task_view(task: dict) -> dict:
    """给前端的任务视图：补齐四段百分比。"""
    chapters = []
    for row in task.get("chapters", []):
        tts_total = int(row.get("tts_total") or 0)
        tts_done = int(row.get("tts_done") or 0)
        stages = {
            "script": row.get("script_status"),
            "coverage": row.get("coverage_status"),
            "tts": row.get("tts_status"),
            "export": row.get("export_status"),
        }
        percent = {
            "script": 100 if stages["script"] == "done" else 0,
            "coverage": 100 if stages["coverage"] == "done" else 0,
            "tts": round(tts_done * 100 / tts_total, 1) if tts_total else 0,
            "export": 100 if stages["export"] == "done" else 0,
        }
        chapters.append({**row, "stages": stages, "percent": percent})
    return {**task, "chapters": chapters}


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    if not length:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8")) if raw else {}


# ---------------------------------------------------------------------------- HTTP 处理
class Handler(BaseHTTPRequestHandler):
    server_version = "novel-forge-webui/1.0"

    def log_message(self, fmt: str, *args) -> None:  # 降低噪音，只写 debug
        log.debug("%s - %s", self.address_string(), fmt % args)

    # ---------------------------------------------------------------- 输出
    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"ok": False, "error": message}, status)

    def _bytes(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, download: bool = False) -> None:
        if not path.exists() or not path.is_file():
            self._error("文件不存在", 404)
            return
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() in (".m4b", ".m4a"):
            ctype = "audio/mp4"
        start, end = 0, size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header:
            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = 206
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition",
                             f'attachment; filename="{urllib.parse.quote(path.name)}"')
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = handle.read(min(262144, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    # ---------------------------------------------------------------- 路由
    def do_GET(self) -> None:
        try:
            self._route_get()
        except HttpError as err:
            self._error(str(err), err.status)
        except Exception as err:
            log.exception("GET %s 失败", self.path)
            self._error(f"{type(err).__name__}: {err}", 500)

    def do_POST(self) -> None:
        try:
            self._route_post()
        except HttpError as err:
            self._error(str(err), err.status)
        except Exception as err:
            log.exception("POST %s 失败", self.path)
            self._error(f"{type(err).__name__}: {err}", 500)

    def do_PATCH(self) -> None:
        try:
            self._route_patch()
        except HttpError as err:
            self._error(str(err), err.status)
        except Exception as err:
            log.exception("PATCH %s 失败", self.path)
            self._error(f"{type(err).__name__}: {err}", 500)

    def _route_get(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        app = APP
        assert app is not None

        if path == "/":
            self._file(app.static_dir / "index.html")
            return
        if path.startswith("/static/"):
            self._file(app.static_dir / path[len("/static/"):])
            return
        if path == "/media":
            target = Path(query.get("path", [""])[0])
            try:
                resolved = target.resolve()
            except OSError:
                self._error("非法路径", 400)
                return
            allowed = [app.books_dir, app.outputs_dir, app.preview_dir]
            if not any(resolved == root or root in resolved.parents for root in allowed):
                self._error("无权访问该文件", 403)
                return
            self._file(resolved, download=query.get("download", ["0"])[0] == "1")
            return
        if path == "/api/health":
            edge_ok, edge_reason = app.registry.available("edge", {"edge_voice": app.caster.narrator_voice})
            try:
                ffmpeg = postprocess.resolve_ffmpeg(app.cfg["postprocess"].get("ffmpeg_path"))
            except Exception as err:
                ffmpeg = f"不可用：{err}"
            self._json({"ok": True, "edge": edge_ok, "edge_reason": edge_reason, "ffmpeg": ffmpeg,
                        "db": str(app.db.path)})
            return
        if path == "/api/voices":
            catalog = getattr(app.caster, "catalog", {}) or {}
            self._json({"ok": True, "voices": app.voices, "stages": list(app.caster.stage_order),
                        "catalog": catalog.get("voices", []),
                        "unavailable": [v["id"] for v in catalog.get("voices", [])
                                        if not v.get("available")],
                        "narrator": app.caster.narrator_voice})
            return
        if path == "/api/settings":
            self._json({"ok": True, **app.settings_view()})
            return
        if path == "/api/books":
            self._json({"ok": True, "books": app.db.list_books()})
            return
        if path == "/api/tasks":
            book_id = query.get("book_id", [None])[0]
            self._json({"ok": True, "tasks": app.db.tasks(book_id, limit=20)})
            return

        parts = [part for part in path.split("/") if part]
        if len(parts) >= 3 and parts[1] == "books":
            book_id = urllib.parse.unquote(parts[2])
            if len(parts) == 3:
                summary = app.db.book_summary(book_id)
                self._json({"ok": True, "book": app.db.get_book(book_id), "summary": summary})
                return
            if len(parts) == 4 and parts[3] == "chapters":
                offset = int(query.get("offset", ["0"])[0])
                limit = min(500, int(query.get("limit", ["100"])[0]))
                search = query.get("q", [""])[0]
                self._json({
                    "ok": True,
                    "chapters": app.db.chapters(book_id, offset, limit, search),
                    "summary": app.db.book_summary(book_id),
                })
                return
            if len(parts) == 4 and parts[3] == "cast": 
                sync_cast(app.db, app.caster, book_id)
                rows = self._scope_cast(app, book_id, app.db.cast_rows(book_id))
                changes = app.db.voice_changes(book_id, 30)
                if rows:
                    names = {row["name"] for row in rows}
                    changes = [item for item in changes if item["name"] in names]
                narrator = app.caster.narrator_profile(book_id)
                cast_map = {row["key"]: row for row in rows}
                collisions = app.caster.check_collisions(cast_map, narrator.get("edge_voice"))
                self._json({
                    "ok": True,
                    "cast": rows,
                    "changes": changes,
                    "narrator": self._narrator_view(app, book_id, narrator),
                    "collisions": collisions,
                    "drift": app.caster.check_drift(),
                    "book_options": app.caster.book_options(book_id),
                    "tts": tts_scheduler.resolve_params(app.cfg, app.root, book_id),
                    "voices": app.voices, "stages": list(app.caster.stage_order),
                    "unavailable": [v["id"] for v in (getattr(app.caster, "catalog", {}) or {})
                                    .get("voices", []) if not v.get("available")],
                })
                return
            if len(parts) == 4 and parts[3] == "cast-export":
                bindings = [b for b in (app.caster.registry.get("voice_bindings") or {}).values()
                            if b.get("book_id") in (None, book_id) or b.get("scope") != "book"]
                self._json({"ok": True, "book_id": book_id,
                            "narrator": self._narrator_view(app, book_id),
                            "bindings": bindings,
                            "book_options": app.caster.book_options(book_id)})
                return
            if len(parts) == 4 and parts[3] == "director-estimate":
                self._json({"ok": True, **self._director_estimate(app, book_id)})
                return
            if len(parts) == 4 and parts[3] == "outputs":
                self._json({"ok": True, "outputs": self._list_outputs(app, book_id)})
                return
            if len(parts) == 6 and parts[3] == "chapters" and parts[5] == "segments":
                idx = int(parts[4])
                self._json({"ok": True, "chapter_idx": idx,
                            "max_failures": int(app.cfg["tts"]["max_segment_failures"]),
                            "segments": app.db.segments(book_id, idx)})
                return
        if len(parts) == 3 and parts[1] == "books" and parts[2] == "summary":
            self._json({"ok": True})
            return
        if len(parts) == 2 and parts[1] == "task":
            self._error("缺少任务 id", 400)
            return
        if len(parts) >= 2 and parts[1] == "tasks":
            task_id = int(parts[2])
            task = app.db.task(task_id)
            if not task:
                self._error("任务不存在", 404)
                return
            self._json({"ok": True, "task": _task_view(task)})
            return
        self._error("未知接口", 404)

    def _route_post(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        app = APP
        assert app is not None
        parts = [part for part in path.split("/") if part]

        if path == "/api/books/upload":
            filename = _safe_name(query.get("filename", ["upload.txt"])[0])
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if not raw:
                self._error("空文件", 400)
                return
            target = app.import_dir / filename
            target.write_bytes(raw)
            book_id = self._import_book(app, target, raw)
            self._json({"ok": True, "book_id": book_id, "book": app.db.get_book(book_id)})
            return
        if path == "/api/books/import":
            body = _read_json(self)
            source = Path(body.get("path", ""))
            if not source.is_absolute():
                source = (app.root / source).resolve()
            if not source.exists():
                self._error(f"找不到文件：{source}", 404)
                return
            # 就地导入：保留原路径，book_id 与已有 scripts/audio/output 目录一致
            book_id = self._import_book(app, source, source.read_bytes())
            self._json({"ok": True, "book_id": book_id, "book": app.db.get_book(book_id)})
            return
        if path.startswith("/api/books/") and path.endswith("/selection"):
            book_id = urllib.parse.unquote(path[len("/api/books/"):-len("/selection")])
            body = _read_json(self)
            app.db.set_selection(book_id, body.get("mode", "ids"), body.get("ids") or [])
            self._json({"ok": True, "summary": app.db.book_summary(book_id)})
            return
        if path == "/api/tasks":
            body = _read_json(self)
            book_id = body.get("book_id") or ""
            if not app.db.get_book(book_id):
                self._error("书籍不存在", 404)
                return
            idxs = body.get("chapter_idxs")
            rows = app.db.selected_chapters(book_id, idxs)
            if not rows:
                self._error("没有选中章节", 400)
                return
            if idxs:
                app.db.set_selection(book_id, "ids", idxs)
            params = {
                "force": bool(body.get("force")),
                "single_voice": bool(body.get("single_voice")),
            }
            if "director" in body:
                params["director"] = bool(body.get("director"))
            task_id = app.db.create_task(book_id, rows, params)
            self._json({"ok": True, "task_id": task_id, "task": _task_view(app.db.task(task_id))})
            return
        if path == "/api/preview":
            body = _read_json(self)
            result = self._preview(app, body)
            self._json({"ok": True, **result})
            return
        if path == "/api/preview/segment":
            body = _read_json(self)
            result = self._preview_segment(app, body)
            self._json({"ok": True, **result})
            return
        if path == "/api/preview/narrator":
            body = _read_json(self)
            result = self._preview_narrator(app, body)
            self._json({"ok": True, **result})
            return
        if path == "/api/narrator":
            body = _read_json(self)
            book_id = body.get("book_id") or None
            scope = body.get("scope") or "book"
            try:
                narrator = app.caster.set_narrator(
                    book_id, body.get("edge_voice") or body.get("voice"),
                    body.get("rate"), body.get("pitch"), body.get("volume"), scope)
            except ValueError as err:
                self._error(str(err), 400)
                return
            self._json({"ok": True, "narrator": self._narrator_view(app, book_id, narrator)})
            return
        if path == "/api/book-options":
            body = _read_json(self)
            book_id = body.get("book_id") or ""
            if not app.db.get_book(book_id):
                self._error("书籍不存在", 404)
                return
            key = body.get("key") or "real_age"
            value = body.get("value")
            if body.get("remove"):
                if key not in ("tts_concurrency", "tts_min_interval_s"):
                    self._error("该选项不支持清除", 400)
                    return
                with app.caster.lock:
                    settings = app.caster.registry.get("book_settings", {}).get(book_id, {})
                    settings.pop(key, None)
                    app.caster._save_registry()
                self._json({"ok": True, "book_options": app.caster.book_options(book_id),
                            "tts": tts_scheduler.resolve_params(app.cfg, app.root, book_id)})
                return
            if key == "real_age":
                value = bool(value)
            elif key == "tts_concurrency":
                params = tts_scheduler.resolve_params(app.cfg, app.root, book_id)
                try:
                    value = max(1, min(int(value), params["max_concurrency"]))
                except (TypeError, ValueError):
                    self._error("并发必须是整数", 400)
                    return
            elif key == "tts_min_interval_s":
                try:
                    value = max(0.0, float(value))
                except (TypeError, ValueError):
                    self._error("间隔必须是数字", 400)
                    return
            else:
                self._error("未知选项", 400)
                return
            options = app.caster.set_book_option(book_id, key, value)
            self._json({"ok": True, "book_options": options,
                        "tts": tts_scheduler.resolve_params(app.cfg, app.root, book_id)})
            return
        if (len(parts) == 8 and parts[1] == "books" and parts[3] == "chapters"
                and parts[5] == "segments" and parts[7] == "retry"):
            book_id = urllib.parse.unquote(parts[2])
            idx = int(parts[4])
            job_index = int(parts[6])
            if not app.db.get_book(book_id):
                self._error("书籍不存在", 404)
                return
            chapter = app.db.chapter(book_id, idx)
            if not chapter:
                self._error("章节不存在", 404)
                return
            if not app.db.segment(book_id, idx, job_index):
                self._error("分段不存在", 404)
                return
            app.db.reset_segment_failures(book_id, idx, job_index)
            body = _read_json(self)
            params = {"force": bool(body.get("force", False)),
                      "single_voice": bool(body.get("single_voice", False))}
            task_id = app.db.create_task(book_id, [chapter], params)
            self._json({"ok": True, "task_id": task_id,
                        "segment": app.db.segment(book_id, idx, job_index),
                        "task": _task_view(app.db.task(task_id))})
            return
        if path == "/api/cast/binding":
            body = _read_json(self)
            result = self._cast_binding(app, body)
            self._json({"ok": True, **result})
            return
        if path == "/api/cast/import":
            body = _read_json(self)
            result = self._cast_import(app, body)
            self._json({"ok": True, **result})
            return
        if len(parts) == 4 and parts[1] == "tasks":
            task_id = int(parts[2])
            action = parts[3]
            task = app.db.task(task_id)
            if not task:
                self._error("任务不存在", 404)
                return
            if action == "pause":
                if task["status"] in ("running", "queued"):
                    app.db.set_task_status(task_id, "paused")
                self._json({"ok": True, "task": _task_view(app.db.task(task_id))})
                return
            if action == "resume":
                if task["status"] in ("paused", "failed", "cancelled"):
                    app.db.set_task_status(task_id, "queued", finished=False)
                self._json({"ok": True, "task": _task_view(app.db.task(task_id))})
                return
            if action == "cancel":
                app.db.set_task_status(task_id, "cancelled", finished=True)
                self._json({"ok": True, "task": _task_view(app.db.task(task_id))})
                return
            if action == "retry":
                failed = app.db.failed_chapter_idx(task_id)
                if not failed:
                    self._error("该任务没有失败章节", 400)
                    return
                rows = [app.db.chapter(task["book_id"], idx) for idx in failed]
                rows = [row for row in rows if row]
                params = json.loads(task.get("params") or "{}")
                new_id = app.db.create_task(task["book_id"], rows, params)
                self._json({"ok": True, "task_id": new_id, "task": _task_view(app.db.task(new_id))})
                return
        self._error("未知接口", 404)

    def _route_patch(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        app = APP
        assert app is not None
        parts = [part for part in path.split("/") if part]

        if path == "/api/settings":
            body = _read_json(self)
            self._json({"ok": True, **app.save_settings(body)})
            return
        if len(parts) == 3 and parts[1] == "chapters":
            body = _read_json(self)
            row = app.db.update_chapter(int(parts[2]), body.get("title"), body.get("selected"))
            if not row:
                self._error("章节不存在", 404)
                return
            self._json({"ok": True, "chapter": row, "summary": app.db.book_summary(row["book_id"])})
            return
        if len(parts) >= 4 and parts[1] == "cast":
            book_id = urllib.parse.unquote(parts[2])
            key = urllib.parse.unquote("/".join(parts[3:]))
            body = _read_json(self)
            rows = {row["key"]: row for row in app.db.cast_rows(book_id)}
            current = rows.get(key)
            if not current:
                self._error("角色不存在（先跑一章生成角色表）", 404)
                return
            name = current["name"]
            old_stage = current["age_stage"]
            new_stage = body.get("age_stage") or old_stage
            voice = body.get("edge_voice") or None
            rate = body.get("rate") or None
            pitch = body.get("pitch") or None
            volume = body.get("volume") or None
            temperament = body.get("temperament") or None
            if new_stage != old_stage:
                updated = app.caster.set_stage(name, old_stage, new_stage, voice, rate, pitch,
                                               volume, chapter_index=current.get("last_seen_chapter"))
                if not updated:
                    self._error("阶段变更失败（角色名/阶段不匹配）", 400)
                    return
                if temperament:
                    updated = app.caster.set_voice(name, new_stage, None, None, None, None,
                                                   temperament=temperament)
            else:
                updated = app.caster.set_voice(name, old_stage, voice, rate, pitch, volume,
                                               temperament=temperament)
            sync_cast(app.db, app.caster, book_id)
            rows = self._scope_cast(app, book_id, app.db.cast_rows(book_id))
            changes = app.db.voice_changes(book_id, 30)
            if rows:
                names = {row["name"] for row in rows}
                changes = [item for item in changes if item["name"] in names]
            self._json({"ok": True, "cast": rows, "changes": changes, "updated": updated,
                        "collisions": app.caster.check_collisions(
                            {row["key"]: row for row in rows}),
                        "drift": app.caster.check_drift()})
            return
        self._error("未知接口", 404)

    # ---------------------------------------------------------------- 业务辅助
    def _scope_cast(self, app: App, book_id: str, rows: list[dict]) -> list[dict]:
        """只展示本书已有剧本里出现过的角色，避免把其他书的全局锁混进来。"""
        keys: set[str] = set()
        scripts_dir = app.book_dir(book_id) / "scripts"
        if scripts_dir.exists():
            for path in scripts_dir.glob("ch*.json"):
                if not re.fullmatch(r"ch\d{3}\.json", path.name):
                    continue  # 跳过 __glm_baseline 等实验快照
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                block = data.get("cast") or {}
                keys.update((block.get("cast") or {}).keys())
                keys.update((block.get("speakers") or {}).values())
        selected = [row for row in rows if not keys or row["key"] in keys]
        characters = {profile["key"]: profile for profile in app.caster.iter_profiles()}
        history: dict[str, list[dict]] = {}
        for entry in characters.values():
            stage = entry.get("age_stage") or entry.get("age_band") or app.caster.default_stage
            history.setdefault(entry.get("name", ""), []).append({
                "stage": stage,
                "key": entry.get("key"),
                "voice": entry.get("edge_voice"),
                "first_seen_chapter": entry.get("first_seen_chapter"),
                "last_seen_chapter": entry.get("last_seen_chapter"),
                "updated_by": entry.get("updated_by"),
                "age_note": entry.get("age_note") or "",
                "age_confidence": entry.get("age_confidence"),
            })
        for items in history.values():
            items.sort(key=lambda item: (app.caster.stage_order.index(item["stage"])
                                         if item["stage"] in app.caster.stage_order
                                         else app.caster.stage_order.index(app.caster.default_stage),
                                         item.get("first_seen_chapter") or 0))
        out = []
        for row in selected:
            entry = characters.get(row["key"], {}) or app.caster.profile_by_key(row["key"]) or {}
            lock = app.caster.binding_for(row["name"], row["age_stage"], book_id) or {}
            enriched = dict(row)
            enriched.update({
                "voice_source": entry.get("voice_source") or ("locked" if lock else "config"),
                "binding_id": entry.get("binding_id"),
                "voice_priority": entry.get("voice_priority"),
                "voice_chain": (entry.get("voice_chain") or [])[:4],
                "gender_confidence": entry.get("gender_confidence"),
                "gender_evidence": entry.get("gender_evidence") or "",
                "gender_confirmed": entry.get("gender_confirmed"),
                "gender_source": entry.get("gender_source") or "",
                "age_evidence_log": (entry.get("age_evidence_log") or [])[-6:],
                "age_history": history.get(row["name"], []),
                "transition": entry.get("transition"),
                "age_hold_reason": entry.get("age_hold_reason") or "",
                "stage_entered_chapter": entry.get("stage_entered_chapter"),
                "legacy_edge_voice": entry.get("legacy_edge_voice") or "",
                "voice_repair": entry.get("voice_repair") or "",
                "locked": bool(lock),
                "lock_updated_by": lock.get("updated_by") or "",
                "line_count": entry.get("line_count") or 0,
                "generic": bool(entry.get("generic")),
            })
            enriched["effective"] = effective_view(app, enriched)
            out.append(enriched)
        return out

    def _narrator_view(self, app: App, book_id: str | None, narrator: dict | None = None) -> dict:
        narrator = narrator or app.caster.narrator_profile(book_id)
        return {
            "name": narrator.get("name") or "旁白",
            "age_stage": narrator.get("display_label") or narrator.get("age_stage", ""),
            "edge_voice": narrator.get("edge_voice"),
            "legacy_edge_voice": narrator.get("legacy_edge_voice"),
            "voice_source": narrator.get("voice_source"),
            "rate": narrator["rate"], "pitch": narrator["pitch"], "volume": narrator["volume"],
            "book_id": book_id, "effective": effective_view(app, narrator),
        }

    def _cast_binding(self, app: App, body: dict) -> dict:
        """显式创建/删除单书例外绑定；撞声 >=2.0 需 force 才允许写入。"""
        book_id = body.get("book_id") or ""
        key = body.get("key") or ""
        rows = {row["key"]: row for row in app.db.cast_rows(book_id)}
        current = rows.get(key)
        if not current:
            raise HttpError("角色不存在（先跑一章生成角色表）", 404)
        name = current["name"]
        stage = body.get("age_stage") or current["age_stage"]
        scope = body.get("scope") or "book_character_stage"
        if scope not in ("book_character_stage", "character_stage"):
            raise HttpError("scope 只能是 book_character_stage 或 character_stage", 400)
        if scope == "book_character_stage" and not book_id:
            raise HttpError("缺少 book_id", 400)
        cid = (app.caster.registry.get("character_aliases") or {}).get(name) or name
        bid = f"{scope}:{book_id or '*'}:{cid}:{stage}"
        if body.get("remove"):
            removed = app.caster.remove_binding(bid)
            sync_cast(app.db, app.caster, book_id)
            scoped = self._scope_cast(app, book_id, app.db.cast_rows(book_id))
            return {"removed": removed, "binding_id": bid,
                    "cast": scoped, "collisions": app.caster.check_collisions(
                        {row["key"]: row for row in scoped})}
        voice = body.get("edge_voice") or current["edge_voice"]
        cast_map = {row["key"]: row for row in self._scope_cast(
            app, book_id, app.db.cast_rows(book_id))}
        report = app.caster.check_collisions(cast_map)
        level = (report.get("by_voice", {}).get(voice) or {}).get("level")
        if level == "error" and not body.get("force"):
            raise HttpError(
                f"{voice} 撞声分已达阻断线，自动推荐被禁止；确认要强制请带 force=true", 409)
        binding = app.caster.upsert_binding({
            "id": bid, "scope": scope, "book_id": book_id or None,
            "character_id": cid, "age_stage": stage, "voice": voice,
            "rate": body.get("rate") or None, "pitch": body.get("pitch") or None,
            "volume": body.get("volume") or None,
            "source": "manual", "locked": bool(body.get("locked")),
            "priority": casting.PRIORITY[scope],
            "note": body.get("note") or "WebUI 显式单书例外",
            "updated_by": "webui",
        })
        sync_cast(app.db, app.caster, book_id)
        scoped = self._scope_cast(app, book_id, app.db.cast_rows(book_id))
        return {"binding": binding, "forced": bool(body.get("force")),
                "cast": scoped,
                "collisions": app.caster.check_collisions(
                    {row["key"]: row for row in scoped})}

    def _cast_import(self, app: App, body: dict) -> dict:
        """从其他书复制角色声线：显式写成目标书的单书例外绑定，冲突逐条列出。"""
        target = body.get("book_id") or ""
        source = body.get("source_book_id") or ""
        if not app.db.get_book(target) or not app.db.get_book(source):
            raise HttpError("目标书或来源书不存在", 404)
        if target == source:
            raise HttpError("来源书与目标书相同", 400)
        source_rows = self._scope_cast(app, source, app.db.cast_rows(source))
        target_rows = self._scope_cast(app, target, app.db.cast_rows(target))
        by_name = {}
        for row in source_rows:
            rank = row.get("last_seen_chapter") or 0
            if row["name"] not in by_name or rank >= (by_name[row["name"]].get("last_seen_chapter") or 0):
                by_name[row["name"]] = row
        created, skipped, conflicts = [], [], []
        for row in target_rows:
            orig = by_name.get(row["name"])
            if not orig:
                skipped.append({"name": row["name"], "reason": "来源书无此角色"})
                continue
            cid = (app.caster.registry.get("character_aliases") or {}).get(row["name"]) or row["name"]
            bid = f"book_character_stage:{target}:{cid}:{row['age_stage']}"
            existing = (app.caster.registry.get("voice_bindings") or {}).get(bid)
            if existing:
                skipped.append({"name": row["name"], "reason": "目标书已有显式例外"})
                continue
            binding = app.caster.upsert_binding({
                "id": bid, "scope": "book_character_stage", "book_id": target,
                "character_id": cid, "age_stage": row["age_stage"],
                "voice": orig.get("edge_voice"), "rate": None, "pitch": None, "volume": None,
                "source": "copy", "locked": False,
                "priority": casting.PRIORITY["book_character_stage"],
                "note": f"从《{source}》复制",
                "updated_by": "webui",
            })
            created.append({"name": row["name"], "binding_id": bid,
                            "voice": binding.get("voice"),
                            "differs": orig.get("edge_voice") != row.get("edge_voice")})
            if orig.get("edge_voice") != row.get("edge_voice"):
                conflicts.append({"name": row["name"], "target_voice": row.get("edge_voice"),
                                  "source_voice": orig.get("edge_voice"),
                                  "binding_id": bid})
        sync_cast(app.db, app.caster, target)
        scoped = self._scope_cast(app, target, app.db.cast_rows(target))
        return {"created": created, "skipped": skipped, "conflicts": conflicts,
                "cast": scoped,
                "collisions": app.caster.check_collisions(
                    {row["key"]: row for row in scoped})}

    def _director_estimate(self, app: App, book_id: str) -> dict:
        """预计送审段数：按当前门控阈值扫已选章节的已存分段（不调用模型）。"""
        sent = total = 0
        missing = 0
        chapters = app.db.selected_chapters(book_id)
        for chapter in chapters:
            rows = app.db.segments(book_id, chapter["idx"], 1000)
            if not rows:
                missing += 1
                continue
            segments = [SimpleNamespace(type=row.get("type", ""), speaker=row.get("speaker", ""),
                                        emotion=row.get("emotion", "平静"), text=row.get("text", ""))
                        for row in rows]
            est = director_mod.estimate(segments, app.cfg)
            sent += est["sent"]
            total += est["total"]
        return {"sent": sent, "total": total, "default_count": total - sent,
                "chapters": len(chapters), "missing_chapters": missing}

    def _import_book(self, app: App, target: Path, raw: bytes) -> str:
        encoding = _detect_encoding(raw)
        chapters = ingest.split_chapters(target, app.cfg["ingest"])
        book_id = chapters[0].book_id
        app.db.upsert_book(book_id, _book_title(target), str(target), encoding,
                           [chapter.to_dict() for chapter in chapters])
        sync_cast(app.db, app.caster, book_id)
        log.info("导入《%s》%s 章（%s）", _book_title(target), len(chapters), encoding)
        return book_id

    def _preview(self, app: App, body: dict) -> dict:
        book_id = body.get("book_id") or ""
        key = body.get("key") or ""
        rows = {row["key"]: row for row in app.db.cast_rows(book_id)}
        profile = rows.get(key)
        if not profile:
            raise HttpError("角色不存在", 404)
        text = (body.get("text") or "").strip()
        if not text:
            name = profile["name"]
            for row in app.db.segments(book_id, int(profile.get("last_seen_chapter") or 1), 500):
                if row["type"] == "dialogue" and row["speaker"] == name and len(row["text"]) <= 90:
                    text = row["text"]
                    break
        if not text:
            text = f"我是{profile['name']}，这是一段试听，用于确认声线与年龄阶段是否合适。"
        text = text[:160]
        entry = app.caster.profile_by_key(key) or {}
        emotion = body.get("emotion") or entry.get("default_emotion") or "平静"
        direction = self._direction_from_body(app, body)
        voice_profile = dict(profile)
        for field in ("rate", "pitch", "volume"):
            if body.get(field):
                voice_profile[field] = body[field]
        if body.get("temperament") in prosody.TEMPERAMENTS:
            voice_profile["temperament"] = body["temperament"]
        composed = prosody.compose_prosody(voice_profile, emotion, text, direction, app.cfg)
        voice = {
            "engine": "edge",
            "edge_voice": body.get("edge_voice") or profile["edge_voice"] or entry.get("edge_voice"),
            **composed["formatted"],
        }
        job = tts_router.TTSJob(
            index=0, chapter_index=int(profile.get("last_seen_chapter") or 0),
            chapter_title="试听", type="dialogue", speaker=profile["name"],
            speaker_key=key, emotion=emotion, text=text, direction=direction,
            requested_engine="edge", engine="edge", voice=voice,
            speech_text=prosody.prepare_text(text, app.cfg, app.registry.lexicon),
        )
        return self._synth_preview(app, job, voice, f"cast:{key}", composed)

    def _preview_segment(self, app: App, body: dict) -> dict:
        """单段即时试听：用已存剧本的 direction + 当前角色参数，不整章重跑。"""
        book_id = body.get("book_id") or ""
        chapter_idx = int(body.get("chapter_idx") or 0)
        job_index = int(body.get("job_index") or 0)
        rows = app.db.segments(book_id, chapter_idx, 1000)
        row = next((item for item in rows if int(item.get("job_index", -1)) == job_index), None)
        if not row:
            raise HttpError("分段不存在（先跑一次该章或重新生成剧本）", 404)
        cast = {item["key"]: item for item in app.db.cast_rows(book_id)}
        if row["type"] == "dialogue":
            profile = cast.get(row.get("speaker_key")) or next(
                (item for item in cast.values() if item["name"] == row["speaker"]), None)
            profile = dict(profile or {
                "key": row.get("speaker_key", ""), "name": row["speaker"],
                "edge_voice": app.cfg["tts"]["engines"]["edge"].get("narrator_voice"),
                "rate": "+0%", "pitch": "+0Hz", "volume": "+0%", "temperament": "中性",
            })
        else:
            profile = dict(app.caster.narrator_profile())
        stored = self._stored_direction(app, book_id, chapter_idx, job_index)
        direction = self._direction_from_body(app, body, fallback=stored)
        emotion = body.get("emotion") or row.get("emotion") or "平静"
        text = (body.get("text") or row.get("text") or "").strip()[:300]
        for field in ("rate", "pitch", "volume"):
            if body.get(field):
                profile[field] = body[field]
        if body.get("temperament") in prosody.TEMPERAMENTS:
            profile["temperament"] = body["temperament"]
        composed = prosody.compose_prosody(profile, emotion, text, direction, app.cfg)
        voice = {
            "engine": "edge",
            "edge_voice": body.get("edge_voice") or profile.get("edge_voice"),
            **composed["formatted"],
        }
        job = tts_router.TTSJob(
            index=job_index, chapter_index=chapter_idx, chapter_title="试听",
            type=row["type"], speaker=row["speaker"], speaker_key=profile.get("key", ""),
            emotion=emotion, text=text, direction=direction,
            requested_engine="edge", engine="edge", voice=voice,
            speech_text=prosody.prepare_text(text, app.cfg, app.registry.lexicon),
        )
        return self._synth_preview(app, job, voice,
                                  f"seg:{book_id}:{chapter_idx}:{job_index}", composed)

    def _preview_narrator(self, app: App, body: dict) -> dict:
        """旁白试听：用本书旁白覆盖 + 临时参数合成，不依赖已存分段。"""
        book_id = body.get("book_id") or None
        profile = dict(app.caster.narrator_profile(book_id))
        if body.get("edge_voice"):
            profile["edge_voice"] = body["edge_voice"]
        for field in ("rate", "pitch", "volume"):
            if body.get(field):
                profile[field] = body[field]
        text = (body.get("text") or "这一章的开头，风从山口吹下来，带着一点凉意。")[:200]
        emotion = body.get("emotion") or profile.get("default_emotion") or "平静"
        direction = self._direction_from_body(app, body)
        composed = prosody.compose_prosody(profile, emotion, text, direction, app.cfg)
        voice = {"engine": "edge", "edge_voice": profile.get("edge_voice"),
                 **composed["formatted"]}
        job = tts_router.TTSJob(
            index=0, chapter_index=0, chapter_title="旁白试听", type="narration",
            speaker="旁白", speaker_key="", emotion=emotion, text=text, direction=direction,
            requested_engine="edge", engine="edge", voice=voice,
            speech_text=prosody.prepare_text(text, app.cfg, app.registry.lexicon),
        )
        return self._synth_preview(app, job, voice, f"narrator:{book_id or 'global'}",
                                  composed)

    def _synth_preview(self, app: App, job, voice: dict, label: str,
                       composed: dict | None = None) -> dict:
        engine = app.registry.engines.get("edge")
        if engine is None:
            raise RuntimeError("edge 引擎未启用")
        payload = json.dumps({
            "label": label, "voice": {key: voice.get(key) for key in
                                      ("edge_voice", "rate", "pitch", "volume")},
            "speech": job.speech_text, "direction": job.direction,
        }, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        out = app.preview_dir / f"{digest}.wav"
        if not out.exists():
            tmp_dir = app.preview_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tts_router.synthesize_job(engine, job, voice, app.cfg, out, tmp_dir)
        return {
            "url": _media_url(out), "duration_ms": _duration_of(out) or 0,
            "text": job.text, "voice": voice.get("edge_voice"),
            "rate": voice.get("rate"), "pitch": voice.get("pitch"),
            "volume": voice.get("volume"), "emotion": job.emotion,
            "direction": job.direction,
            "composed": {
                "raw": {kind: round(composed["raw"][kind], 2) for kind in ("rate", "pitch", "volume")},
                "weights": composed["weights"],
                "clamped": composed["clamped"],
                "emotion_variant": composed["emotion_variant"],
            } if composed else {},
        }

    def _stored_direction(self, app: App, book_id: str, chapter_idx: int, job_index: int) -> dict:
        path = app.book_dir(book_id) / "scripts" / f"ch{chapter_idx:03d}.json"
        if not path.exists():
            return prosody.default_direction()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return prosody.default_direction()
        for item in data.get("jobs") or []:
            if int(item.get("index", -1)) == job_index and isinstance(item.get("direction"), dict):
                return dict(prosody.default_direction(), **item["direction"])
        return prosody.default_direction()

    def _direction_from_body(self, app: App, body: dict, fallback: dict | None = None) -> dict:
        base = dict(prosody.default_direction(), **(fallback or {}))
        raw = body.get("direction")
        if isinstance(raw, dict):
            base.update(raw)
        for key in ("pace", "volume", "pause_after_ms", "emphasis"):
            if key in body:
                base[key] = body[key]
        item = director_mod.normalize_item({"index": 1, **base},
                                           int(app.cfg["script"].get("director", {})
                                               .get("pause_after_max_ms", 1200)))
        return item or prosody.default_direction()

    def _list_outputs(self, app: App, book_id: str) -> list[dict]:
        rows = {(row["chapter_idx"], row["kind"]): row for row in app.db.outputs(book_id)}
        items: list[dict] = []
        output_dir = app.book_dir(book_id) / "output"
        if output_dir.exists():
            for path in sorted(output_dir.iterdir()):
                if path.suffix.lower() not in (".wav", ".srt", ".m4b"):
                    continue
                kind = path.suffix.lower().lstrip(".")
                match = re.search(r"ch(\d+)", path.name)
                chapter_idx = int(match.group(1)) if match else -1
                record = rows.get((chapter_idx, kind))
                items.append({
                    "name": path.name,
                    "kind": kind,
                    "kind_label": KIND_LABEL.get(kind, kind.upper()),
                    "chapter_idx": chapter_idx,
                    "size": path.stat().st_size,
                    "duration_ms": (record or {}).get("duration_ms") or _duration_of(path) or 0,
                    "url": _media_url(path),
                    "download": _media_url(path) + "&download=1",
                })
        for path in sorted(app.outputs_dir.glob(f"demo_{book_id}_*")):
            if path.suffix.lower() not in (".m4b", ".srt", ".wav"):
                continue
            kind = path.suffix.lower().lstrip(".")
            duration = _duration_of(path)
            if duration is None and kind in ("m4b", "srt"):
                sibling = path.with_suffix(".wav")
                duration = _duration_of(sibling)
            items.append({
                "name": path.name, "kind": kind, "kind_label": KIND_LABEL.get(kind, kind.upper()),
                "chapter_idx": -1, "size": path.stat().st_size,
                "duration_ms": duration or 0,
                "url": _media_url(path), "download": _media_url(path) + "&download=1",
            })
        return items


# ---------------------------------------------------------------------------- 启动
def serve(root: Path, cfg: dict, port: int = 8765) -> None:
    global APP
    APP = App(root, cfg, port)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    print(f"\nWebUI 已启动：http://127.0.0.1:{port}  （仅本机访问，Ctrl+C 退出）")
    print(f"SQLite：{APP.db.path}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n正在退出…")
    finally:
        APP.runner.shutdown()
        httpd.server_close()
