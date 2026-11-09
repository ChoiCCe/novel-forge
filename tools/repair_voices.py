"""v4：修复 registry v2 中失效/性别错配的 edge 声线引用（幂等、先备份）。

- 声线唯一存放在 `voice_bindings[*].voice`；失效声线 -> catalog 平替；
- 自动/迁移产生的锁若性别与角色不符，换成同性别声线（manual/webui 保留反串）；
- 原始值保留在 `legacy_voice`，替代原因写 `voice_repair`；
- 备份写到 voices/registry.v4-pre-repair.<日期>.bak.json（已存在则复用，不覆盖）；
- 只读 registry，不碰音频缓存。
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.caster import infer_gender  # noqa: E402

REG = ROOT / "voices" / "registry.json"
CAT = ROOT / "voices" / "catalog.edge.json"
CFG = ROOT / "config.yaml"


def main() -> int:
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    default_stage = str(cfg["casting"]["age"]["default_stage"])
    catalog = json.loads(CAT.read_text(encoding="utf-8"))
    fallbacks = catalog.get("fallbacks", {})
    voices = {v["id"]: v for v in catalog.get("voices", [])}
    reg = json.loads(REG.read_text(encoding="utf-8"))

    bak = ROOT / "voices" / f"registry.v4-pre-repair.{time.strftime('%Y%m%d')}.bak.json"
    if not bak.exists():
        shutil.copy2(REG, bak)
        print(f"备份 -> {bak}")

    repaired = []

    def fix(obj: dict, where: str, voice_field: str = "voice") -> None:
        voice = obj.get(voice_field)
        if not voice:
            return
        entry = voices.get(voice)
        if entry and entry.get("available", True):
            return
        fb = fallbacks.get(voice) or {}
        target = fb.get("voice") if isinstance(fb, dict) else fb
        if not target:
            print(f"[WARN] {where} 引用 {voice} 不可用且无平替")
            return
        obj.setdefault("legacy_voice", voice)
        obj[voice_field] = target
        obj["voice_repair"] = (fb.get("note") if isinstance(fb, dict) else "") or "v4 失效声线平替"
        repaired.append({"where": where, "from": voice, "to": target})

    bindings = reg.get("voice_bindings") or {}
    for key, binding in bindings.items():
        fix(binding, f"voice_bindings[{key}]")

    stage_map = (((cfg.get("tts") or {}).get("engines") or {}).get("edge", {})
                 .get("age_stage_voice_map") or {})
    identities = reg.get("identities") or {}
    alias_gender: dict[str, str] = {}
    for ident in identities.values():
        if ident.get("canonical_name") and ident.get("gender") in ("男", "女"):
            alias_gender[ident["canonical_name"]] = ident["gender"]
    for alias, cid in (reg.get("character_aliases") or {}).items():
        gender = identities.get(cid, {}).get("gender")
        if gender in ("男", "女"):
            alias_gender[alias] = gender

    def target_voice(gender: str, stage: str) -> str:
        candidate = (stage_map.get(stage) or {}).get(gender) or ""
        fb = fallbacks.get(candidate) or {}
        return (fb.get("voice") if isinstance(fb, dict) else candidate) or candidate

    def fix_gender(binding: dict, where: str, name: str, stage: str, hint: str = "") -> None:
        voice = binding.get("voice")
        if not voice or str(binding.get("updated_by", "")).lower() in ("manual", "webui"):
            return
        gender = hint if hint in ("男", "女") else alias_gender.get(name) or infer_gender(name)
        if gender not in ("男", "女"):
            return
        voice_gender = (voices.get(voice) or {}).get("gender", "未知")
        if voice_gender not in ("男", "女") or voice_gender == gender:
            return
        target = target_voice(gender, stage)
        if not target:
            print(f"[WARN] {where} 性别 {gender} 阶段 {stage} 无同性别声线")
            return
        original = binding.get("legacy_voice") or voice
        binding["legacy_voice"] = original
        binding["voice"] = target
        binding["voice_repair"] = f"性别不匹配（{gender}）：{original} → {target}"
        repaired.append({"where": where, "from": original, "to": target,
                         "reason": "gender_mismatch"})

    for key, binding in bindings.items():
        ident = identities.get(binding.get("character_id") or "") or {}
        fix_gender(binding, f"voice_bindings[{key}]", ident.get("canonical_name", ""),
                   binding.get("age_stage") or default_stage, ident.get("gender", ""))

    for index, event in enumerate(reg.get("voice_events") or []):
        for field in ("from_voice", "to_voice"):
            voice = event.get(field)
            entry = voices.get(voice)
            if voice and entry and not entry.get("available", True):
                fb = fallbacks.get(voice) or {}
                event[f"{field}_legacy"] = voice
                event[field] = (fb.get("voice") if isinstance(fb, dict) else fb) or voice

    reg["voice_map_version"] = "edge-v5"
    reg["voice_audit"] = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "unavailable": [{"voice": v, "replacement": (fb.get("voice") if isinstance(fb, dict) else fb),
                         "error": voices.get(v, {}).get("error", "")} for v, fb in fallbacks.items()],
        "repaired_count": len(repaired),
        "note": "原值保存在 legacy_voice；音频缓存未改动",
    }

    tmp = REG.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(REG)
    print(json.dumps({"repaired": repaired, "count": len(repaired)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
