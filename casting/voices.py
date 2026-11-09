# -*- coding: utf-8 -*-
"""声线池可信表：自动枚举 + 实测判重 + 音色距离 + 可用性分类。

设计约束：
- 不读人工维护的 catalog；每次按当前 edge 服务实际返回的声线重算。
- 探测结果按 (voice, text, rate, pitch, volume) 缓存，重复跑不重复合成。
- 判重与距离基于同文本同参数的合成音频，不依赖声线名称。
- 跨语言/跨地区：主池只取目标 locale；同语言其他地区标 cross_region，
  同地区口音标 minor_only，默认都不进主池。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import wave
from pathlib import Path

import numpy as np

PROBE_TEXT = "这一句用来检查声线是否可以正常合成，语气平稳，不带情绪，也不刻意加快或放慢。"
SR = 16000


# --------------------------------------------------------------------------- 枚举
def enumerate_voices(lang: str = "zh") -> list[dict]:
    """返回当前 edge 服务可用的目标语言声线（含 region 与 gender）。"""
    import edge_tts

    async def _run():
        return await edge_tts.list_voices()

    rows = asyncio.run(_run())
    out = []
    for r in rows:
        locale = r.get("Locale", "")
        if not locale.lower().startswith(lang.lower()):
            continue
        out.append({
            "id": r.get("ShortName", ""),
            "gender": "女" if r.get("Gender") == "Female" else "男",
            "locale": locale,
            "region": "-".join(locale.split("-")[:2]),
        })
    return sorted(out, key=lambda x: x["id"])


# --------------------------------------------------------------------------- 合成
def _synth(text: str, voice: str, dst: Path, rate="+0%", pitch="+0Hz", volume="+0%") -> Path:
    import edge_tts
    from core.postprocess import convert_to_wav

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 1000:
        return dst
    tmp = dst.with_suffix(".part.mp3")
    last = None
    for _ in range(3):
        try:
            comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
            asyncio.run(comm.save(str(tmp)))
            if tmp.exists() and tmp.stat().st_size > 0:
                convert_to_wav(tmp, dst, SR)
                tmp.unlink(missing_ok=True)
                return dst
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"edge 合成失败 {voice}: {last}")


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as fh:
        data = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16)
    return data.astype(np.float64) / 32768.0


def _frames(x: np.ndarray, win=0.04, hop=0.01) -> np.ndarray:
    n, h = int(win * SR), int(hop * SR)
    if len(x) < n:
        return np.zeros((0, n))
    return np.lib.stride_tricks.sliding_window_view(x, n)[::h]


def _features(x: np.ndarray) -> dict:
    frames = _frames(x)
    if len(frames) == 0:
        return {"f0": 0.0, "f0_p25": 0.0, "f0_p75": 0.0, "env": None}
    lo, hi = int(SR / 400), int(SR / 60)
    window = np.hanning(frames.shape[1])
    freqs = np.fft.rfftfreq(frames.shape[1], 1 / SR)
    band = (freqs >= 80) & (freqs <= 6000)
    f0s, envs = [], []
    for fr in frames:
        if math.sqrt(float(np.mean(fr ** 2))) < 0.02:
            continue
        c = fr - fr.mean()
        corr = np.correlate(c, c, mode="full")[len(c) - 1:]
        corr /= max(corr[0], 1e-9)
        seg = corr[lo:hi]
        if seg.size:
            peak = int(np.argmax(seg)) + lo
            if corr[peak] >= 0.3:
                f0s.append(SR / peak)
        spec = np.abs(np.fft.rfft(fr * window)) + 1e-9
        envs.append(np.log10(spec[band]))
    if not f0s:
        return {"f0": 0.0, "f0_p25": 0.0, "f0_p75": 0.0, "env": None}
    f0 = np.array(f0s)
    return {
        "f0": float(np.median(f0)),
        "f0_p25": float(np.percentile(f0, 25)),
        "f0_p75": float(np.percentile(f0, 75)),
        "env": np.mean(np.array(envs), axis=0),
    }


def _semitone(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 99.0
    return abs(12 * math.log2(a / b))


def _env_dist(a, b) -> float:
    if a is None or b is None or len(a) != len(b):
        return 1.0
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --------------------------------------------------------------------------- 主流程
def build_pool(lang: str = "zh", cache_dir: str | Path = "outputs/novel-forge/casting_voices",
               narrator: str | None = None, probe: bool = True) -> dict:
    """枚举 + 实测，返回 voice_pool：每条声线的分类、距离、近音组。"""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    voices = enumerate_voices(lang)
    base = voices[0]["region"] if voices else lang

    rows = []
    for v in voices:
        row = {**v, "status": "unknown", "distance": None, "nearest": None,
               "near_group": None, "duplicate_of": None, "scope": "main", "notes": []}
        if v["locale"] != base:
            same_lang = v["locale"].split("-")[0] == base.split("-")[0]
            row["scope"] = "cross_region" if same_lang else "minor_only"
            row["notes"].append("地区不同，默认不进主池")
        if probe:
            key = hashlib.md5(f"{v['id']}|{PROBE_TEXT}|{lang}".encode()).hexdigest()[:16]
            dst = cache / f"{key}.wav"
            try:
                _synth(PROBE_TEXT, v["id"], dst)
                feats = _features(_read_wav(dst))
                row.update(feats)
                row["status"] = "ok"
                row["probe_wav"] = str(dst)
            except Exception as exc:  # noqa: BLE001
                row["status"] = "unavailable"
                row["error"] = str(exc)[:160]
        rows.append(row)

    ok = [r for r in rows if r["status"] == "ok" and r["scope"] == "main"]
    # 判重（同性别内）+ 音色距离
    for i, a in enumerate(ok):
        for b in ok[i + 1:]:
            if a["gender"] != b["gender"]:
                continue
            st = _semitone(a["f0"], b["f0"])
            d = _env_dist(a["env"], b["env"])
            if d < 0.02 and st < 0.4:
                keep, drop = (a, b) if len(a["id"]) <= len(b["id"]) else (b, a)
                drop["duplicate_of"] = keep["id"]
                drop["notes"].append("实测与保留项同音")
        best, best_d = None, 9.0
        for b in ok:
            if b is a or b["gender"] != a["gender"]:
                continue
            st = _semitone(a["f0"], b["f0"])
            d = _env_dist(a["env"], b["env"]) + 0.25 * min(st, 4) / 4
            if d < best_d:
                best, best_d = b, d
        a["distance"] = round(best_d, 3) if best else None
        a["nearest"] = best["id"] if best else None

    # 近音组：距离 < 0.18 视为需要警告的同质声线
    parent = {r["id"]: r["id"] for r in ok}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in ok:
        if r["nearest"] and r["distance"] is not None and r["distance"] < 0.18:
            ra, rb = find(r["id"]), find(r["nearest"])
            if ra != rb:
                parent[rb] = ra
    groups: dict[str, list[str]] = {}
    for r in ok:
        groups.setdefault(find(r["id"]), []).append(r["id"])
    near_groups = [sorted(v) for v in groups.values() if len(v) > 1]
    for gi, g in enumerate(near_groups):
        for r in ok:
            if r["id"] in g:
                r["near_group"] = gi

    # 分类：duplicate → backup；cross_region → disabled；其余独立可用
    for r in rows:
        if r["status"] == "unavailable":
            r["usable"] = False
            r["class"] = "unavailable"
        elif r["duplicate_of"]:
            r["usable"] = False
            r["class"] = "backup"
            r["notes"].append(f"仅作备份，等价于 {r['duplicate_of']}")
        elif r["scope"] == "cross_region":
            r["usable"] = False
            r["class"] = "disabled"
        elif r["scope"] == "minor_only":
            r["usable"] = True
            r["class"] = "minor_only"
        else:
            r["usable"] = True
            r["class"] = "independent"
    for r in rows:
        r.pop("env", None)

    usable = [r for r in rows if r["usable"] and r["class"] == "independent"]
    if narrator is not None:
        for r in rows:
            r["is_narrator"] = r["id"] == narrator
    else:
        for r in rows:
            r["is_narrator"] = False

    return {
        "lang": lang,
        "base_region": base,
        "probe": probe,
        "probe_text": PROBE_TEXT if probe else "",
        "narrator": narrator,
        "voices": rows,
        "main_pool": [r["id"] for r in rows
                      if r["usable"] and r["class"] == "independent" and not r["is_narrator"]],
        "narrator_voice": narrator,
        "near_groups": near_groups,
        "counts": {
            "enumerated": len(rows),
            "independent": sum(1 for r in rows if r["class"] == "independent"),
            "minor_only": sum(1 for r in rows if r["class"] == "minor_only"),
            "backup": sum(1 for r in rows if r["class"] == "backup"),
            "disabled": sum(1 for r in rows if r["class"] == "disabled"),
            "unavailable": sum(1 for r in rows if r["class"] == "unavailable"),
        },
    }


def choose_narrator(pool: dict, demand: dict[str, int]) -> str:
    """旁白选声：优先从“供大于求”的性别里挑，其次挑音高最稳的。

    demand 是各性别需要独立声线的角色数。旁白独占一条声线，
    所以应让稀缺性别（供 < 求）尽量留给角色。
    """
    usable = [r for r in pool["voices"] if r["usable"] and r["class"] == "independent"]
    if not usable:
        raise RuntimeError("没有可用声线，无法选旁白")
    supply = {}
    for r in usable:
        supply[r["gender"]] = supply.get(r["gender"], 0) + 1

    def volatility(r):
        return (r["f0_p75"] - r["f0_p25"]) / max(r["f0"], 1.0)

    def surplus(r):
        # 每条声线按 2 人容量计，减去该性别角色需求
        return supply.get(r["gender"], 0) * 2 - demand.get(r["gender"], 0)

    return min(usable, key=lambda r: (-surplus(r), volatility(r)))["id"]


def load_or_build(path: str | Path, lang: str = "zh", rebuild: bool = False, probe=True) -> dict:
    p = Path(path)
    if p.exists() and not rebuild:
        return json.loads(p.read_text(encoding="utf-8"))
    pool = build_pool(lang=lang, probe=probe)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")
    return pool
