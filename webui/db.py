"""SQLite 持久化：书、章节、分段、任务、角色锁定、进度、成品索引。

设计约束：
- 只用标准库 sqlite3，不引入 ORM；每次操作独立连接（WAL 模式），可跨线程调用；
- 章节正文入库，服务重启后不依赖内存字典；
- 角色锁定的“真源”仍是 voices/registry.json（跨书复用），SQLite 里按书镜像一份供界面编辑与展示。
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  source_path TEXT NOT NULL,
  encoding TEXT DEFAULT '',
  total_chars INTEGER DEFAULT 0,
  chapter_count INTEGER DEFAULT 0,
  created_at TEXT,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS chapters (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  book_id TEXT NOT NULL,
  idx INTEGER NOT NULL,
  label TEXT DEFAULT '',
  title TEXT NOT NULL,
  text TEXT DEFAULT '',
  char_count INTEGER DEFAULT 0,
  selected INTEGER DEFAULT 1,
  est_duration_ms INTEGER DEFAULT 0,
  updated_at TEXT,
  UNIQUE(book_id, idx)
);
CREATE INDEX IF NOT EXISTS idx_chapters_book ON chapters(book_id, idx);

CREATE TABLE IF NOT EXISTS segments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  book_id TEXT NOT NULL,
  chapter_idx INTEGER NOT NULL,
  job_index INTEGER NOT NULL,
  type TEXT DEFAULT 'narration',
  speaker TEXT DEFAULT '旁白',
  speaker_key TEXT DEFAULT '',
  emotion TEXT DEFAULT '平静',
  text TEXT DEFAULT '',
  engine TEXT DEFAULT '',
  voice TEXT DEFAULT '',
  cache_key TEXT DEFAULT '',
  audio_path TEXT DEFAULT '',
  duration_ms INTEGER DEFAULT 0,
  status TEXT DEFAULT 'pending',
  error TEXT DEFAULT '',
  fail_count INTEGER DEFAULT 0,
  last_error_type TEXT DEFAULT '',
  last_error_at TEXT DEFAULT '',
  updated_at TEXT,
  UNIQUE(book_id, chapter_idx, job_index)
);

CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  book_id TEXT NOT NULL,
  kind TEXT DEFAULT 'chapters',
  params TEXT DEFAULT '{}',
  status TEXT DEFAULT 'queued',
  error TEXT DEFAULT '',
  total_chapters INTEGER DEFAULT 0,
  done_chapters INTEGER DEFAULT 0,
  failed_chapters INTEGER DEFAULT 0,
  created_at TEXT,
  started_at TEXT,
  finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, id);

CREATE TABLE IF NOT EXISTS task_chapters (
  task_id INTEGER NOT NULL,
  chapter_idx INTEGER NOT NULL,
  title TEXT DEFAULT '',
  script_status TEXT DEFAULT 'pending',
  coverage_status TEXT DEFAULT 'pending',
  tts_status TEXT DEFAULT 'pending',
  export_status TEXT DEFAULT 'pending',
  tts_done INTEGER DEFAULT 0,
  tts_total INTEGER DEFAULT 0,
  coverage_ratio REAL,
  detail TEXT DEFAULT '',
  error TEXT DEFAULT '',
  updated_at TEXT,
  PRIMARY KEY (task_id, chapter_idx)
);

CREATE TABLE IF NOT EXISTS cast (
  book_id TEXT NOT NULL,
  key TEXT NOT NULL,
  name TEXT DEFAULT '',
  gender TEXT DEFAULT '',
  age_stage TEXT DEFAULT '',
  edge_voice TEXT DEFAULT '',
  rate TEXT DEFAULT '+0%',
  pitch TEXT DEFAULT '+0Hz',
  volume TEXT DEFAULT '+0%',
  temperament TEXT DEFAULT '中性',
  prosody_source TEXT DEFAULT 'temperament',
  age_note TEXT DEFAULT '',
  age_confidence REAL,
  first_seen_chapter INTEGER,
  last_seen_chapter INTEGER,
  updated_by TEXT DEFAULT '',
  persona TEXT DEFAULT '',
  updated_at TEXT,
  PRIMARY KEY (book_id, key)
);

CREATE TABLE IF NOT EXISTS voice_changes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  book_id TEXT NOT NULL,
  chapter_idx INTEGER,
  name TEXT DEFAULT '',
  from_stage TEXT DEFAULT '',
  to_stage TEXT DEFAULT '',
  from_voice TEXT DEFAULT '',
  to_voice TEXT DEFAULT '',
  evidence TEXT DEFAULT '',
  confidence REAL,
  updated_by TEXT DEFAULT '',
  created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_voice_changes_book ON voice_changes(book_id, id);

CREATE TABLE IF NOT EXISTS outputs (
  book_id TEXT NOT NULL,
  chapter_idx INTEGER NOT NULL,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  size_bytes INTEGER DEFAULT 0,
  duration_ms INTEGER DEFAULT 0,
  created_at TEXT,
  PRIMARY KEY (book_id, chapter_idx, kind)
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT DEFAULT ''
);
"""


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.init()

    # ---------------------------------------------------------------- 基础
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def init(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # 旧库升级：补 cast 的新列（SQLite 不支持 ADD COLUMN IF NOT EXISTS）
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(cast)")}
            for name, ddl in (
                ("volume", "TEXT DEFAULT '+0%'"),
                ("temperament", "TEXT DEFAULT '中性'"),
                ("prosody_source", "TEXT DEFAULT 'temperament'"),
            ):
                if name not in columns:
                    conn.execute(f"ALTER TABLE cast ADD COLUMN {name} {ddl}")
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(segments)")}
            for name, ddl in (
                ("fail_count", "INTEGER DEFAULT 0"),
                ("last_error_type", "TEXT DEFAULT ''"),
                ("last_error_at", "TEXT DEFAULT ''"),
            ):
                if name not in columns:
                    conn.execute(f"ALTER TABLE segments ADD COLUMN {name} {ddl}")

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def _row(self, sql: str, params: tuple = ()) -> dict | None:
        rows = self._rows(sql, params)
        return rows[0] if rows else None

    def _write(self, sql: str, params: tuple = ()) -> int:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(sql, params)
            return cursor.lastrowid or 0

    # ---------------------------------------------------------------- 书与章节
    def upsert_book(self, book_id: str, title: str, source_path: str, encoding: str,
                    chapters: list[dict]) -> None:
        total_chars = sum(int(item["char_count"]) for item in chapters)
        stamp = now()
        with self._lock, self._connect() as conn:
            existing = conn.execute("SELECT id FROM books WHERE id=?", (book_id,)).fetchone()
            if existing:
                conn.execute(
                    "UPDATE books SET title=?, source_path=?, encoding=?, total_chars=?, "
                    "chapter_count=?, updated_at=? WHERE id=?",
                    (title, source_path, encoding, total_chars, len(chapters), stamp, book_id))
                old = {row["idx"]: row for row in conn.execute(
                    "SELECT idx, title, selected FROM chapters WHERE book_id=?", (book_id,))}
            else:
                conn.execute(
                    "INSERT INTO books(id,title,source_path,encoding,total_chars,chapter_count,"
                    "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                    (book_id, title, source_path, encoding, total_chars, len(chapters), stamp, stamp))
                old = {}
            for item in chapters:
                kept = old.get(item["index"])
                title_value = kept["title"] if kept and kept["title"] else item["title"]
                selected = kept["selected"] if kept else 1
                est = int(int(item["char_count"]) / 4.7 * 1000)
                conn.execute(
                    "INSERT INTO chapters(book_id,idx,label,title,text,char_count,selected,"
                    "est_duration_ms,updated_at) VALUES(?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(book_id,idx) DO UPDATE SET label=excluded.label,"
                    "title=excluded.title,text=excluded.text,char_count=excluded.char_count,"
                    "est_duration_ms=excluded.est_duration_ms,updated_at=excluded.updated_at",
                    (book_id, item["index"], item["label"], title_value, item["text"],
                     int(item["char_count"]), selected, est, stamp))

    def list_books(self) -> list[dict]:
        return self._rows(
            "SELECT b.*, (SELECT COUNT(*) FROM tasks t WHERE t.book_id=b.id) AS task_count "
            "FROM books b ORDER BY b.updated_at DESC, b.id")

    def get_book(self, book_id: str) -> dict | None:
        return self._row("SELECT * FROM books WHERE id=?", (book_id,))

    def book_summary(self, book_id: str) -> dict:
        row = self._row(
            "SELECT COUNT(*) AS chapter_count, COALESCE(SUM(char_count),0) AS total_chars, "
            "COALESCE(SUM(CASE WHEN selected=1 THEN char_count ELSE 0 END),0) AS selected_chars, "
            "COALESCE(SUM(CASE WHEN selected=1 THEN est_duration_ms ELSE 0 END),0) AS est_duration_ms, "
            "COALESCE(SUM(selected),0) AS selected_count FROM chapters WHERE book_id=?", (book_id,))
        return row or {}

    def chapters(self, book_id: str, offset: int = 0, limit: int = 100, query: str = "") -> list[dict]:
        sql = ("SELECT id, idx, label, title, char_count, selected, est_duration_ms "
               "FROM chapters WHERE book_id=?")
        params: list = [book_id]
        if query:
            sql += " AND (title LIKE ? OR CAST(idx AS TEXT)=?)"
            params += [f"%{query}%", query]
        sql += " ORDER BY idx LIMIT ? OFFSET ?"
        params += [limit, offset]
        return self._rows(sql, tuple(params))

    def chapter(self, book_id: str, idx: int) -> dict | None:
        return self._row("SELECT * FROM chapters WHERE book_id=? AND idx=?", (book_id, idx))

    def update_chapter(self, chapter_row_id: int, title: str | None = None,
                       selected: int | None = None) -> dict | None:
        fields, params = [], []
        if title is not None:
            fields.append("title=?")
            params.append(title.strip() or "未命名章节")
        if selected is not None:
            fields.append("selected=?")
            params.append(1 if selected else 0)
        if not fields:
            return self._row("SELECT * FROM chapters WHERE id=?", (chapter_row_id,))
        params.append(now())
        params.append(chapter_row_id)
        self._write(f"UPDATE chapters SET {', '.join(fields)}, updated_at=? WHERE id=?", tuple(params))
        return self._row("SELECT * FROM chapters WHERE id=?", (chapter_row_id,))

    def set_selection(self, book_id: str, mode: str, ids: list[int] | None = None) -> None:
        stamp = now()
        if mode == "all":
            self._write("UPDATE chapters SET selected=1, updated_at=? WHERE book_id=?", (stamp, book_id))
        elif mode == "none":
            self._write("UPDATE chapters SET selected=0, updated_at=? WHERE book_id=?", (stamp, book_id))
        elif mode == "invert":
            self._write("UPDATE chapters SET selected=1-selected, updated_at=? WHERE book_id=?",
                        (stamp, book_id))
        elif mode == "ids":
            with self._lock, self._connect() as conn:
                conn.execute("UPDATE chapters SET selected=0, updated_at=? WHERE book_id=?", (stamp, book_id))
                for row_id in ids or []:
                    conn.execute("UPDATE chapters SET selected=1, updated_at=? WHERE id=? AND book_id=?",
                                 (stamp, int(row_id), book_id))

    def selected_chapters(self, book_id: str, idxs: list[int] | None = None) -> list[dict]:
        if idxs:
            rows: list[dict] = []
            for idx in idxs:
                item = self.chapter(book_id, int(idx))
                if item:
                    rows.append(item)
            return rows
        return self._rows(
            "SELECT * FROM chapters WHERE book_id=? AND selected=1 ORDER BY idx", (book_id,))

    # ---------------------------------------------------------------- 角色锁定镜像
    def replace_cast(self, book_id: str, rows: list[dict]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM cast WHERE book_id=?", (book_id,))
            conn.executemany(
                "INSERT INTO cast(book_id,key,name,gender,age_stage,edge_voice,rate,pitch,volume,"
                "temperament,prosody_source,age_note,"
                "age_confidence,first_seen_chapter,last_seen_chapter,updated_by,persona,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(book_id, r.get("key", ""), r.get("name", ""), r.get("gender", ""),
                  r.get("age_stage", ""), r.get("edge_voice", ""), r.get("rate", "+0%"),
                  r.get("pitch", "+0Hz"), r.get("volume", "+0%"),
                  r.get("temperament", "中性"), r.get("prosody_source", "temperament"),
                  r.get("age_note", ""), r.get("age_confidence"),
                  r.get("first_seen_chapter"), r.get("last_seen_chapter"),
                  r.get("updated_by", ""), r.get("persona", ""), now()) for r in rows])

    def cast_rows(self, book_id: str) -> list[dict]:
        return self._rows(
            "SELECT * FROM cast WHERE book_id=? ORDER BY name, age_stage", (book_id,))

    def replace_voice_changes(self, book_id: str, rows: list[dict]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM voice_changes WHERE book_id=?", (book_id,))
            conn.executemany(
                "INSERT INTO voice_changes(book_id,chapter_idx,name,from_stage,to_stage,from_voice,"
                "to_voice,evidence,confidence,updated_by,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                [(book_id, r.get("chapter"), r.get("name", ""), r.get("from_stage", ""),
                  r.get("to_stage", ""), r.get("from_voice", ""), r.get("to_voice", ""),
                  r.get("evidence", ""), r.get("confidence"), r.get("updated_by", ""), now())
                 for r in rows])

    def voice_changes(self, book_id: str, limit: int = 50) -> list[dict]:
        return self._rows(
            "SELECT * FROM voice_changes WHERE book_id=? ORDER BY id DESC LIMIT ?", (book_id, limit))

    # ---------------------------------------------------------------- 分段
    def replace_segments(self, book_id: str, chapter_idx: int, jobs: list[dict]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM segments WHERE book_id=? AND chapter_idx=?",
                         (book_id, chapter_idx))
            conn.executemany(
                "INSERT INTO segments(book_id,chapter_idx,job_index,type,speaker,speaker_key,emotion,"
                "text,engine,voice,cache_key,audio_path,duration_ms,status,error,fail_count,"
                "last_error_type,last_error_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(book_id, chapter_idx, j.get("index", 0), j.get("type", ""), j.get("speaker", ""),
                  j.get("speaker_key", ""), j.get("emotion", ""), j.get("text", ""),
                  j.get("engine", ""), j.get("voice", {}).get("edge_voice", ""),
                  j.get("cache_key", ""), j.get("audio_path") or "", int(j.get("duration_ms", 0)),
                  j.get("status", ""), j.get("error", ""), int(j.get("fail_count", 0) or 0),
                  j.get("last_error_type", ""), j.get("last_error_at", ""), now()) for j in jobs])

    def upsert_segment(self, book_id: str, chapter_idx: int, job: dict) -> None:
        self._write(
            "INSERT INTO segments(book_id,chapter_idx,job_index,type,speaker,speaker_key,emotion,"
            "text,engine,voice,cache_key,audio_path,duration_ms,status,error,fail_count,"
            "last_error_type,last_error_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(book_id,chapter_idx,job_index) DO UPDATE SET engine=excluded.engine,"
            "voice=excluded.voice,cache_key=excluded.cache_key,audio_path=excluded.audio_path,"
            "duration_ms=excluded.duration_ms,status=excluded.status,error=excluded.error,"
            "fail_count=excluded.fail_count,last_error_type=excluded.last_error_type,"
            "last_error_at=excluded.last_error_at,updated_at=excluded.updated_at",
            (book_id, chapter_idx, job.get("index", 0), job.get("type", ""), job.get("speaker", ""),
             job.get("speaker_key", ""), job.get("emotion", ""), job.get("text", ""),
             job.get("engine", ""), job.get("voice", {}).get("edge_voice", ""),
             job.get("cache_key", ""), job.get("audio_path") or "", int(job.get("duration_ms", 0)),
             job.get("status", ""), job.get("error", ""), int(job.get("fail_count", 0) or 0),
             job.get("last_error_type", ""), job.get("last_error_at", ""), now()))

    def segments(self, book_id: str, chapter_idx: int, limit: int = 500) -> list[dict]:
        return self._rows(
            "SELECT * FROM segments WHERE book_id=? AND chapter_idx=? ORDER BY job_index LIMIT ?",
            (book_id, chapter_idx, limit))

    def segment(self, book_id: str, chapter_idx: int, job_index: int) -> dict | None:
        return self._row(
            "SELECT * FROM segments WHERE book_id=? AND chapter_idx=? AND job_index=?",
            (book_id, chapter_idx, job_index))

    def reset_segment_failures(self, book_id: str, chapter_idx: int, job_index: int) -> int:
        """手动重试：清零连续失败计数并恢复 pending；成功一次后 render 会自动清零。"""
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "UPDATE segments SET fail_count=0,last_error_type='',last_error_at='',"
                "status='pending',error='',updated_at=? "
                "WHERE book_id=? AND chapter_idx=? AND job_index=?",
                (now(), book_id, chapter_idx, job_index))
            return cursor.rowcount or 0

    # ---------------------------------------------------------------- 任务
    def create_task(self, book_id: str, chapter_rows: list[dict], params: dict) -> int:
        import json
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO tasks(book_id,kind,params,status,total_chapters,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (book_id, "chapters", json.dumps(params, ensure_ascii=False), "queued",
                 len(chapter_rows), now()))
            task_id = cursor.lastrowid
            conn.executemany(
                "INSERT INTO task_chapters(task_id,chapter_idx,title,updated_at) VALUES(?,?,?,?)",
                [(task_id, int(row["idx"]), row["title"], now()) for row in chapter_rows])
            return int(task_id)

    def task(self, task_id: int) -> dict | None:
        row = self._row("SELECT * FROM tasks WHERE id=?", (task_id,))
        if row:
            row["chapters"] = self.task_chapters(task_id)
        return row

    def tasks(self, book_id: str | None = None, limit: int = 20) -> list[dict]:
        if book_id:
            return self._rows("SELECT * FROM tasks WHERE book_id=? ORDER BY id DESC LIMIT ?",
                              (book_id, limit))
        return self._rows("SELECT * FROM tasks ORDER BY id DESC LIMIT ?", (limit,))

    def task_chapters(self, task_id: int) -> list[dict]:
        return self._rows(
            "SELECT * FROM task_chapters WHERE task_id=? ORDER BY chapter_idx", (task_id,))

    def set_task_status(self, task_id: int, status: str, error: str = "",
                        started: bool = False, finished: bool = False) -> None:
        fields = ["status=?", "error=?"]
        params: list = [status, error]
        if started:
            fields.append("started_at=?")
            params.append(now())
        if finished:
            fields.append("finished_at=?")
            params.append(now())
        params.append(task_id)
        self._write(f"UPDATE tasks SET {', '.join(fields)} WHERE id=?", tuple(params))

    def task_status(self, task_id: int) -> str:
        row = self._row("SELECT status FROM tasks WHERE id=?", (task_id,))
        return row["status"] if row else "missing"

    def bump_task_counts(self, task_id: int, done: int = 0, failed: int = 0) -> None:
        self._write("UPDATE tasks SET done_chapters=done_chapters+?, "
                    "failed_chapters=failed_chapters+? WHERE id=?",
                    (done, failed, task_id))

    def claim_next_task(self) -> int | None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT id FROM tasks WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
            if not row:
                conn.execute("ROLLBACK")
                return None
            conn.execute("UPDATE tasks SET status='running', started_at=?, error='' WHERE id=?",
                         (now(), row["id"]))
            conn.execute("COMMIT")
            return int(row["id"])

    def reset_interrupted(self) -> int:
        """服务重启：把上次停在 running 的任务标记为 paused，可一键继续。"""
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT id FROM tasks WHERE status='running'").fetchall()
            for row in rows:
                conn.execute("UPDATE tasks SET status='paused' WHERE id=?", (row["id"],))
                conn.execute("UPDATE task_chapters SET script_status='pending' "
                             "WHERE task_id=? AND script_status='running'", (row["id"],))
                conn.execute("UPDATE task_chapters SET tts_status='pending', detail='服务重启，待继续' "
                             "WHERE task_id=? AND tts_status='running'", (row["id"],))
                conn.execute("UPDATE task_chapters SET export_status='pending' "
                             "WHERE task_id=? AND export_status='running'", (row["id"],))
            return len(rows)

    def update_task_chapter(self, task_id: int, chapter_idx: int, **fields) -> None:
        if not fields:
            return
        fields["updated_at"] = now()
        sets = ", ".join(f"{key}=?" for key in fields)
        params = list(fields.values()) + [task_id, chapter_idx]
        self._write(f"UPDATE task_chapters SET {sets} WHERE task_id=? AND chapter_idx=?", tuple(params))

    def failed_chapter_idx(self, task_id: int) -> list[int]:
        rows = self._rows(
            "SELECT chapter_idx FROM task_chapters WHERE task_id=? AND "
            "(script_status='failed' OR tts_status='failed' OR export_status='failed')", (task_id,))
        return [int(row["chapter_idx"]) for row in rows]

    # ---------------------------------------------------------------- 成品索引
    def record_output(self, book_id: str, chapter_idx: int, kind: str, path: str,
                      size_bytes: int, duration_ms: int) -> None:
        self._write(
            "INSERT INTO outputs(book_id,chapter_idx,kind,path,size_bytes,duration_ms,created_at) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(book_id,chapter_idx,kind) DO UPDATE SET "
            "path=excluded.path,size_bytes=excluded.size_bytes,duration_ms=excluded.duration_ms,"
            "created_at=excluded.created_at",
            (book_id, chapter_idx, kind, path, size_bytes, duration_ms, now()))

    def outputs(self, book_id: str) -> list[dict]:
        return self._rows(
            "SELECT * FROM outputs WHERE book_id=? ORDER BY chapter_idx, kind", (book_id,))

    # ---------------------------------------------------------------- 运行时设置
    def setting(self, key: str, default: str = "") -> str:
        row = self._row("SELECT value FROM settings WHERE key=?", (key,))
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        self._write(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))

    def settings(self) -> dict:
        return {row["key"]: row["value"] for row in self._rows("SELECT key,value FROM settings")}
