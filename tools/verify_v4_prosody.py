"""v4 步骤 6 验收：愤怒韵律 + 旁白 -4%/-6% 同段时长对比。

- 数值部分直接调用 compose_prosody，验收“最终值”，不是看配置数字；
- 时长部分固定 Xiaoxiao/同文本/同参数，只改 rate，输出 WAV 并测真实时长。
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import edge_tts  # noqa: E402
import yaml  # noqa: E402
from core import postprocess, prosody  # noqa: E402

OUT = ROOT.parent / "outputs" / "v4_prosody_ab"
NARRATION = {
    "short": "他抬起头，看了一眼天色。",
    "normal": ("长安城的暮色来得比往年早，楼础站在城门口，看着远处那队人马缓缓走近，"
               "心里忽然生出一种说不清道不明的预感，仿佛有什么事情要在今夜发生。"),
    "long": ("那年冬天，周律第一次见到闻人学究，是在太学最里面的那间书阁，"
             "老人正伏在案前抄写一卷旧书，窗外的雪落了一层又一层，"
             "屋里只点着一盏快要燃尽的油灯，把两个人的影子拉得很长很长。"
             "他站在门口没有出声，等了足足半个时辰，才听见老人头也不抬地问了一句，"
             "你是谁家的孩子，这么晚了还在这里做什么；周律答得很慢，"
             "说自己是从城南来的，想借一本书看看，可那本书其实早就被人借走了。"),
}


def load_cfg(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def profile_for(reg: dict, name: str, stage: str) -> dict:
    ident = None
    for candidate in (reg.get("identities") or {}).values():
        if name in ([candidate.get("canonical_name")] + list(candidate.get("aliases") or [])):
            ident = candidate
            break
    if not ident:
        raise KeyError(f"identity not found: {name}")
    profile = dict((ident.get("stage_profiles") or {}).get(stage) or {})
    profile["name"] = ident.get("canonical_name", name)
    profile["age_stage"] = stage
    binding = (reg.get("voice_bindings") or {}).get(f"lock:{ident.get('id')}|{stage}") or {}
    profile["edge_voice"] = binding.get("voice")
    return profile


def angry_check() -> dict:
    reg = json.loads((ROOT / "voices" / "registry.json").read_text(encoding="utf-8"))
    text = "气死我了！气死我了！"
    out = {}
    for name, stage_name in (("周律", "少年"), ("周律", "中年")):
        stage_key = f"{name}_男_{stage_name}"
        profile = profile_for(reg, name, stage_name)
        stage = {}
        for label, cfg_file in (("v3", "config.v3.bak.yaml"), ("v4", "config.yaml")):
            cfg = load_cfg(cfg_file)
            res = prosody.compose_prosody(profile, "愤怒", text, cfg)
            stage[label] = {
                "profile_base": {k: profile.get(k) for k in ("rate", "pitch", "volume")},
                "final": res["formatted"],
                "raw": {k: round(v, 2) for k, v in res["raw"].items()},
                "weights": res["weights"],
                "clamped": res["clamped"],
            }
        out[stage_key] = stage
    return out


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as fh:
        return fh.getnframes() / float(fh.getframerate())


async def synth(text: str, rate: str, dst: Path) -> float:
    mp3 = dst.with_suffix(".mp3")
    await edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural", rate=rate).save(str(mp3))
    postprocess.convert_to_wav(mp3, dst, 24000)
    return wav_duration(dst)


async def narration_check() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, text in NARRATION.items():
        d4 = await synth(text, "-4%", OUT / f"nar_{name}_v3_rate-4.wav")
        d6 = await synth(text, "-6%", OUT / f"nar_{name}_v4_rate-6.wav")
        rows[name] = {
            "chars": len(text),
            "v3_seconds": round(d4, 3),
            "v4_seconds": round(d6, 3),
            "ratio": round(d6 / d4, 4) if d4 else None,
            "expected_ratio": 1.021,  # (1/0.94)/(1/0.96)
            "delta_pct": round((d6 / d4 - 1) * 100, 2) if d4 else None,
        }
    return rows


async def main() -> int:
    t = time.time()
    angry = angry_check()
    prev_path = OUT / "verify.json"
    prev = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else {}
    if "--skip-narration" in sys.argv:
        narr = prev.get("narration", {})
    else:
        narr = await narration_check()
    result = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "angry": angry, "narration": narr}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verify.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"耗时 {time.time() - t:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
