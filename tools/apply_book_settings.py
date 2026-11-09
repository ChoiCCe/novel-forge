"""把门户保存的「按书参数」落到 registry.json。

门户进程不直接写 /opt/novel-forge；worker（root）在每次生成前调用本工具，
因此改动只对下一次生成生效，已生成的章节不受影响。

用法：
    python tools/apply_book_settings.py --book-id <book_id> --settings /path/book.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.caster import Caster  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def _clean(value) -> str:
    return str(value or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="应用门户按书旁白/角色参数")
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--settings", required=True, help="门户保存的 settings JSON")
    args = parser.parse_args()

    payload = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    caster = Caster(CFG, None, ROOT)

    narrator = payload.get("narrator") or {}
    if any(_clean(narrator.get(key)) for key in ("voice", "rate", "pitch", "volume")):
        desired = {key: _clean(narrator.get(key)) or None
                   for key in ("voice", "rate", "pitch", "volume")
                   if _clean(narrator.get(key))}
        current = caster.narrator_profile(args.book_id)
        if all(current.get(key) == value for key, value in desired.items()):
            print(f"narrator: unchanged, skip write")
        else:
            caster.set_narrator(
                args.book_id,
                voice=_clean(narrator.get("voice")) or None,
                rate=_clean(narrator.get("rate")) or None,
                pitch=_clean(narrator.get("pitch")) or None,
                volume=_clean(narrator.get("volume")) or None,
                scope="book",
            )
            print(f"narrator: {narrator}")

    for key, item in (payload.get("characters") or {}).items():
        if not isinstance(item, dict):
            continue
        name = _clean(item.get("name")) or key.split("|")[0]
        stage = _clean(item.get("age_stage")) or (key.split("|")[-1] if "|" in key else "")
        voice = _clean(item.get("voice")) or None
        rate = _clean(item.get("rate")) or None
        pitch = _clean(item.get("pitch")) or None
        volume = _clean(item.get("volume")) or None
        if not (voice or rate or pitch or volume):
            continue
        ident = caster._identity(name, args.book_id, create=True,
                                 gender=_clean(item.get("gender")) or "未知")
        if ident is None:
            print(f"skip: {name}")
            continue
        if not stage:
            stage = caster.default_stage
        ident.setdefault("stage_profiles", {}).setdefault(stage, {})
        desired = {"edge_voice": voice, "rate": rate, "pitch": pitch, "volume": volume}
        desired = {key: value for key, value in desired.items() if value}
        current = caster._profile_view(ident, stage)
        if all(current.get(key) == value for key, value in desired.items()):
            print(f"character: {name}|{stage} unchanged, skip write")
            continue
        caster.set_voice(name, stage, edge_voice=voice, rate=rate, pitch=pitch,
                         volume=volume, updated_by="manual")
        print(f"character: {name}|{stage} -> {voice or '-'} {rate or '-'} {pitch or '-'} {volume or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
