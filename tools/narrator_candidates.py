"""生成旁白候选 A/B 试听包（同一短段 + 同一长段，逐候选合成）。

用法：
    python tools/narrator_candidates.py

输出：outputs/novel-forge/v5_narrator_candidates/*.wav + index.html + index.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import narrator_lock, postprocess, prosody  # noqa: E402
from core.caster import Caster  # noqa: E402
from core.tts_router import EngineRegistry, TTSJob, synthesize_job  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

SHORT_TEXT = "大将军发怒的时候必须有人来劝，否则的话，事后他会更愤怒，迁怒于当时在场的所有人。"
LONG_TEXT = ("无论做什么，十八岁都该是有所成就的时候了，更有个别幸运儿，未满十岁就凭父祖的功业"
             "获得爵位，十三四岁领受尊贵而清闲的官职，没有意外的话，在十八岁之前将能手握实权，"
             "参与议事治国，若能表现突出，早晚会被提拔为国家砥柱大臣。")
CANDIDATES = [
    ("C0", "基线（当前）", "-6%", "+0Hz", "+0%"),
    ("C1", "候选一", "-8%", "-2Hz", "+0%"),
    ("C2", "候选二", "-10%", "-3Hz", "+0%"),
    ("C3", "候选三", "-12%", "-4Hz", "+0%"),
]


def make_job(index: int, text: str, voice: dict, lexicon: dict) -> TTSJob:
    return TTSJob(
        index=index, chapter_index=1, chapter_title="旁白候选", type="narration",
        speaker="旁白", speaker_key="旁白", emotion="平静", text=text,
        requested_engine="edge", engine="edge", voice=dict(voice),
        speech_text=prosody.prepare_text(text, CFG, lexicon),
    )


def main() -> int:
    out_dir = (ROOT / CFG["paths"]["outputs"] / "v5_narrator_candidates").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = EngineRegistry(CFG, ROOT)
    engine = registry.engines["edge"]
    base = narrator_lock.normalize(narrator_lock.effective(CFG))
    rows = []
    for cid, label, rate, pitch, volume in CANDIDATES:
        voice = {"engine": "edge", "edge_voice": base["voice"],
                 "rate": rate, "pitch": pitch, "volume": volume}
        if cid == "C0":
            voice["rate"] = base["rate"]
            voice["pitch"] = base["pitch"]
            voice["volume"] = base["volume"]
        for kind, text in (("short", SHORT_TEXT), ("long", LONG_TEXT)):
            name = f"{cid}_{kind}.wav"
            dest = out_dir / name
            job = make_job(len(rows), text, voice, registry.lexicon)
            synthesize_job(engine, job, voice, CFG, dest, out_dir)
            duration = postprocess.wav_info(dest)["duration_ms"]
            chars = prosody.spoken_len(job.speech_text)
            rows.append({
                "id": cid, "label": label, "kind": kind, "file": name,
                "rate": voice["rate"], "pitch": voice["pitch"], "volume": voice["volume"],
                "voice": voice["edge_voice"], "chars": chars,
                "duration_ms": duration,
                "chars_per_sec": round(chars / (duration / 1000), 2) if duration else 0,
                "text": text,
            })
            print(f"{cid} {kind:<6} {duration/1000:6.2f}s {chars:>4}字 "
                  f"{rows[-1]['chars_per_sec']:.2f}字/秒  {name}")
    index = {"baseline": base, "short_text": SHORT_TEXT, "long_text": LONG_TEXT,
             "candidates": rows}
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    write_html(out_dir, rows, base)
    print(f"\n试听包：{out_dir}")
    return 0


def write_html(out_dir: Path, rows: list[dict], base: dict) -> None:
    parts = []
    for row in rows:
        parts.append(f"""
        <tr><td>{row['id']} {row['label']}</td><td>{row['kind']}</td>
        <td>{row['rate']} / {row['pitch']} / {row['volume']}</td>
        <td>{row['duration_ms']/1000:.2f}s</td><td>{row['chars']}</td>
        <td>{row['chars_per_sec']:.2f}</td>
        <td><audio controls preload="none" src="{row['file']}"></audio></td></tr>""")
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>旁白候选 A/B 试听</title>
<style>body{{font:15px/1.6 system-ui,'Microsoft YaHei',sans-serif;margin:32px;color:#222}}
h1{{font-size:20px}}table{{border-collapse:collapse;width:100%;max-width:1000px}}
td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}
audio{{width:230px}}small{{color:#666}}</style>
<h1>旁白候选 A/B 试听</h1>
<p>基线：{base['voice']} {base['rate']} / {base['pitch']} / {base['volume']}。
每条候选使用同一段约 40 字旁白与同一段 108 字旁白；时长含流水线的子句停顿，字/秒按实际朗读字数计算。</p>
<table><tr><th>编号</th><th>段型</th><th>rate / pitch / volume</th><th>时长</th>
<th>字数</th><th>字/秒</th><th>试听</th></tr>{''.join(parts)}</table>
</html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
