"""标记旧旁白参数合成、需要按 C3 重合成的章节（只标注，不重跑）。

用法：
    python tools/mark_narrator_stale.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import narrator_lock  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
TARGET = narrator_lock.effective(CFG)


def narration_segments(record: dict) -> int:
    return sum(1 for seg in record.get("segments", [])
               if seg.get("type") in ("narration", "旁白"))


def audio_artifacts(book_dir: Path, chapter: str) -> list[str]:
    found: list[str] = []
    out_dir = book_dir / "output"
    if out_dir.exists():
        for item in sorted(out_dir.iterdir()):
            if item.is_file() and item.stem.startswith(chapter) \
                    and item.suffix.lower() in (".wav", ".mp3", ".m4b", ".srt"):
                found.append(str(item.relative_to(ROOT)))
    audio_dir = book_dir / "audio" / chapter
    if audio_dir.exists():
        wavs = sorted(audio_dir.glob("*.wav"))
        if wavs:
            found.append(f"{audio_dir.relative_to(ROOT)}/（{len(wavs)} 段缓存音频）")
    return found


def main() -> int:
    chapters: list[dict] = []
    for script in sorted((ROOT / "novels").glob("*/scripts/*.json")):
        if "__" in script.stem or ".failed" in script.stem:
            continue
        record = json.loads(script.read_text(encoding="utf-8"))
        narrator = (record.get("cast") or {}).get("narrator") or {}
        if not narrator:
            continue
        current = narrator_lock.normalize({
            "voice": narrator.get("edge_voice") or narrator.get("voice"),
            "rate": narrator.get("rate"), "pitch": narrator.get("pitch"),
            "volume": narrator.get("volume"),
        })
        diff = narrator_lock.diff(TARGET, current)
        artifacts = audio_artifacts(script.parent.parent, script.stem)
        chapters.append({
            "book_id": record.get("book_id") or script.parent.parent.name,
            "chapter": script.stem,
            "script": str(script.relative_to(ROOT)),
            "snapshot": current,
            "target": narrator_lock.normalize(TARGET),
            "diff": diff,
            "narration_segments": narration_segments(record),
            "artifacts": artifacts,
            "stale": bool(diff),
            "needs_resynth": bool(diff) and bool(artifacts),
        })

    stale = [item for item in chapters if item["stale"]]
    need = [item for item in chapters if item["needs_resynth"]]
    payload = {
        "target": narrator_lock.normalize(TARGET),
        "fingerprint": narrator_lock.fingerprint(TARGET),
        "summary": {"chapters": len(chapters), "stale": len(stale),
                    "needs_resynth": len(need)},
        "note": "只标注不重跑；缓存 key 含旁白参数指纹，C3 合成不会命中旧参数缓存。",
        "chapters": chapters,
    }
    out_dir = (ROOT / CFG["paths"]["outputs"] / "narrator_c3").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "narrator_c3_stale.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    books: dict[str, list[dict]] = {}
    for item in chapters:
        books.setdefault(item["book_id"], []).append(item)
    for book_id, items in books.items():
        book_dir = next((ROOT / "novels").glob(f"{book_id}*/"), None)
        if not book_dir:
            continue
        (book_dir / "output").mkdir(parents=True, exist_ok=True)
        (book_dir / "output" / "narrator_stale.json").write_text(
            json.dumps({"target": payload["target"], "chapters": items},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    for item in chapters:
        flag = "需重合成" if item["needs_resynth"] else ("参数不同(无音频)" if item["stale"] else "已是 C3")
        detail = "、".join(f"{k}: {v['locked']}->{v['current']}" for k, v in item["diff"].items())
        print(f"{item['book_id']}/{item['chapter']:<6} {flag:<14} 旁白段 {item['narration_segments']:>3}  "
              f"产物 {len(item['artifacts'])}  {detail}")
    print(f"\n汇总：{len(stale)} 章参数非 C3，其中 {len(need)} 章已有音频需重合成")
    print(f"报告：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
