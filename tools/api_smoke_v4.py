"""v4 WebUI API 冒烟：不依赖浏览器，验证旁白覆盖 / 真实年龄开关 / 单书例外 / 复制预检。

用法：python tools/api_smoke_v4.py [--base http://127.0.0.1:8799]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

BOOK_OVERRIDE_RATE = "-6%"


def call(base: str, path: str, payload: dict | None = None, method: str | None = None):
    url = base.rstrip("/") + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8") or "{}")


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(("PASS " if ok else "FAIL ") + label + (f"  [{detail}]" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8799")
    args = parser.parse_args()
    base = args.base
    passed = True

    status, data = call(base, "/api/health")
    passed &= check("健康检查", status == 200 and data.get("edge"), str(data.get("edge_reason")))

    status, data = call(base, "/api/books")
    books = data.get("books") or []
    passed &= check("书库非空", bool(books), f"{len(books)} 本")
    if not books:
        return 1
    book = books[0]["id"]

    status, cast_data = call(base, f"/api/books/{urllib.parse.quote(book)}/cast")
    data = cast_data
    rows = data.get("cast") or []
    voices = data.get("voices") or []
    passed &= check("角色表", status == 200 and bool(rows), f"{len(rows)} 行")
    passed &= check("旁白字段", bool(data.get("narrator", {}).get("edge_voice")))
    passed &= check("撞声报告", "by_voice" in (data.get("collisions") or {}),
                    f"{len((data.get('collisions') or {}).get('warnings', []))} 条提示")
    passed &= check("失效声线清单", bool(data.get("unavailable")),
                    ",".join(data.get("unavailable") or []))

    if not rows:
        return 1

    status, data = call(base, "/api/narrator", {
        "book_id": book, "scope": "book", "edge_voice": "zh-CN-XiaoxiaoNeural",
        "rate": BOOK_OVERRIDE_RATE, "pitch": "+0Hz", "volume": "+0%",
    })
    narrator = data.get("narrator") or {}
    passed &= check("旁白本书覆盖", status == 200 and narrator.get("rate") == BOOK_OVERRIDE_RATE,
                    f"rate={narrator.get('rate')} source={narrator.get('voice_source')}")
    passed &= check("旁白回显 book_id", narrator.get("book_id") == book, str(narrator.get("book_id")))

    status, data = call(base, "/api/book-options", {
        "book_id": book, "key": "real_age", "value": True})
    passed &= check("真实年龄开关 ON", status == 200 and data.get("book_options", {}).get("real_age") is True)
    status, data = call(base, "/api/book-options", {
        "book_id": book, "key": "real_age", "value": False})
    passed &= check("真实年龄开关 OFF", status == 200 and data.get("book_options", {}).get("real_age") is False)

    row = rows[0]
    alt_voice = next(v for v in voices if v != row.get("edge_voice"))
    payload = {"book_id": book, "key": row["key"], "edge_voice": alt_voice,
               "scope": "book_character_stage", "force": True}
    status, data = call(base, "/api/cast/binding", payload)
    passed &= check("单书例外创建", status == 200 and data.get("binding", {}).get("voice") == alt_voice,
                    f"{row['name']} -> {alt_voice}")
    payload["remove"] = True
    status, data = call(base, "/api/cast/binding", payload)
    passed &= check("单书例外删除", status == 200 and data.get("removed") is True)

    status, data = call(base, "/api/book-options", {
        "book_id": book, "key": "tts_concurrency", "value": 2})
    tts = data.get("tts") or {}
    passed &= check("单书 TTS 并发", status == 200 and tts.get("concurrency") == 2,
                    f"并发={tts.get('concurrency')} 上限={tts.get('max_concurrency')}")
    status, data = call(base, "/api/book-options", {
        "book_id": book, "key": "tts_min_interval_s", "value": 1.5})
    tts = data.get("tts") or {}
    passed &= check("单书 TTS 间隔", status == 200 and tts.get("min_interval_s") == 1.5,
                    f"间隔={tts.get('min_interval_s')} 来源={tts.get('source')}")
    for key in ("tts_concurrency", "tts_min_interval_s"):
        status, data = call(base, "/api/book-options",
                            {"book_id": book, "key": key, "remove": True})
    passed &= check("单书 TTS 清除", status == 200)

    status, data = call(base, "/api/cast/import", {"book_id": book, "source_book_id": book})
    passed &= check("同书复制被拒", status == 400, str(data.get("error")))

    status, data = call(base, f"/api/books/{urllib.parse.quote(book)}/cast-export")
    passed &= check("导出快照", status == 200 and "bindings" in data,
                    f"{len(data.get('bindings') or [])} 条绑定")

    print("RESULT", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
