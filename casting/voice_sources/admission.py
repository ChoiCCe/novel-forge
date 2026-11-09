# -*- coding: utf-8 -*-
"""准入校验：可用性 → 跨地区 → 判重 → 音色距离 → 分级。

三关全自动，输出带数值依据的可信分级表：

- ``independent`` 独立可用：地区一致、探测成功、不与池内任何同语言同性别
  声线判重、最近邻复合距离 ≥0.18 且 centered 距离 ≥0.18；
- ``backup`` 仅备用：与池内某条实测同音（包络 <0.02 且半音差 <0.4），
  或与最近邻过近（复合距离 <0.18，或 centered <0.18）；
- ``disabled`` 禁用：探测失败、语言/地区不符、短名冲突。

判重与距离都用**同一条固定探测文本 + 同一采样率**的合成音频，不依赖声线名称。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from . import acoustics
from .protocol import VoiceSpec, short_name


@dataclass(frozen=True)
class Thresholds:
    duplicate_env: float = 0.02
    duplicate_semitone: float = 0.4
    near: float = 0.18
    near_centered: float = 0.18


@dataclass
class ProbedVoice:
    spec: VoiceSpec
    provider: str
    probe_text: str
    wav: Path | None = None
    features: acoustics.VoiceFeatures | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.features is not None and self.error is None


# --------------------------------------------------------------------------- 探测
def probe_specs(items, workdir: str | Path, text: str) -> list[ProbedVoice]:
    """items: iterable of registry.ProviderSpec。逐个调用 provider.probe 并测量。"""
    out: list[ProbedVoice] = []
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    for item in items:
        spec: VoiceSpec = item.spec
        dst = work / f"{spec.id}.wav"
        probed = ProbedVoice(spec=spec, provider=item.provider.name,
                             probe_text=text, wav=dst)
        try:
            result = item.provider.probe(spec, text, dst)
            if result is not None:
                dst = Path(result)
                probed.wav = dst
            if not dst.exists() or dst.stat().st_size < 1000:
                raise RuntimeError("provider 未产出有效 wav")
            probed.features = acoustics.extract(dst)
        except Exception as exc:  # noqa: BLE001
            probed.error = f"{type(exc).__name__}: {exc}"[:200]
        out.append(probed)
    return out


# --------------------------------------------------------------------------- 参考池
def _load_reference(pool: dict) -> dict[str, acoustics.VoiceFeatures]:
    feats: dict[str, acoustics.VoiceFeatures] = {}
    for row in pool.get("voices", []):
        wav = row.get("probe_wav")
        if not wav or not Path(wav).exists():
            continue
        try:
            feats[row["id"]] = acoustics.extract(wav)
        except Exception:  # noqa: BLE001
            continue
    return feats


def make_pool(lang: str, base_region: str, probe_text: str, rows: list[dict]) -> dict:
    """把一组声线行（id/gender/locale/probe_wav）组装成 pool 结构。"""
    voices = []
    for r in rows:
        voices.append({
            "id": r["id"], "gender": r["gender"], "locale": r.get("locale", base_region),
            "region": "-".join(r.get("locale", base_region).split("-")[:2]),
            "status": "ok", "scope": "main", "class": "independent", "usable": True,
            "distance": None, "nearest": None, "near_group": None, "duplicate_of": None,
            "notes": [], "probe_wav": r["probe_wav"], "is_narrator": False,
            **{k: v for k, v in r.items()
               if k not in ("id", "gender", "locale", "probe_wav")},
        })
    return {
        "lang": lang, "base_region": base_region, "probe": True, "probe_text": probe_text,
        "narrator": None, "voices": voices,
        "main_pool": [v["id"] for v in voices],
        "narrator_voice": None, "near_groups": [],
        "counts": {"enumerated": len(voices), "independent": len(voices),
                   "minor_only": 0, "backup": 0, "disabled": 0, "unavailable": 0},
    }


# --------------------------------------------------------------------------- 准入
def admit(candidates: list[ProbedVoice], pool: dict,
          thresholds: Thresholds | None = None, *, acoustic_gate: bool = False) -> dict:
    """声线准入。

    默认 ``acoustic_gate=False``：不按音频距离判重/判近音，也不再产生 backup。
    分配器只负责分配，不评判两条声线像不像；距离只作为观测值记录，
    不影响 verdict。需要旧行为时显式传 ``acoustic_gate=True``。
    """
    th = thresholds or Thresholds()
    refs = _load_reference(pool)
    ref_rows = {row["id"]: row for row in pool.get("voices", [])}
    existing_short = {short_name(v["id"]): v["id"] for v in pool.get("voices", [])}
    seen_short: dict[str, str] = {}

    results = []
    for cand in candidates:
        spec = cand.spec
        item = {
            "id": spec.id, "name": spec.name, "source": spec.source,
            "provider": cand.provider, "declared": {
                "lang": spec.lang, "locale": spec.locale, "region": spec.region,
                "gender": spec.gender, "age": spec.age,
            },
            "probe": {"ok": cand.ok, "error": cand.error,
                      "wav": str(cand.wav) if cand.wav else None,
                      **(cand.features.as_dict() if cand.features else {})},
            "checks": {}, "duplicate_of": None, "nearest": None,
            "verdict": "disabled", "usable": False, "reasons": [],
        }
        results.append(item)

        if not cand.ok:
            item["reasons"].append(f"可用性探测失败：{cand.error}")
            item["checks"]["availability"] = {"ok": False}
            continue
        item["checks"]["availability"] = {"ok": True}

        # 跨语言 / 跨地区：直接拒绝
        region_ok = spec.lang.lower() == str(pool.get("lang", "")).lower() \
            and spec.locale == pool.get("base_region")
        item["checks"]["region"] = {
            "ok": region_ok, "lang": spec.lang, "locale": spec.locale,
            "pool_lang": pool.get("lang"), "pool_region": pool.get("base_region"),
        }
        if not region_ok:
            item["reasons"].append(
                f"语言/地区不符：{spec.locale} ≠ 目标池 {pool.get('base_region')}")
            continue

        short = short_name(spec.id)
        if short in existing_short or short in seen_short:
            other = existing_short.get(short) or seen_short.get(short)
            item["reasons"].append(f"基声线短名冲突：{short} 已被 {other} 占用")
            item["checks"]["short_name"] = {"ok": False, "short": short}
            continue
        seen_short[short] = spec.id
        item["checks"]["short_name"] = {"ok": True, "short": short}

        same_gender = [(rid, f) for rid, f in refs.items()
                       if ref_rows.get(rid, {}).get("gender") == spec.gender
                       and ref_rows.get(rid, {}).get("locale") == pool.get("base_region")]

        # 默认路径：不按音频相似度判重/判近音，只记录最近邻作为观测值。
        if not acoustic_gate:
            best = None
            for rid, f in same_gender:
                d = acoustics.timbre_distance(cand.features, f)
                c = acoustics.centered_distance(cand.features, f)
                e = acoustics.env_distance(cand.features.env_primary, f.env_primary)
                st = acoustics.semitone(cand.features.f0, f.f0)
                if best is None or d < best["distance"]:
                    best = {"id": rid, "distance": round(d, 3),
                            "env_primary": round(e, 4), "env_centered": round(c, 4),
                            "semitone": round(st, 2)}
            item["nearest"] = best
            item["checks"]["duplicate"] = {"ok": True, "duplicates": [], "gated": False}
            item["checks"]["near"] = {
                "ok": True, "gated": False,
                "distance": best["distance"] if best else None,
                "env_centered": best["env_centered"] if best else None,
            }
            item["verdict"] = "independent"
            item["usable"] = True
            item["reasons"].append("默认路径不按音频相似度判重/判近音（acoustic_gate=False）")
            continue

        sr_bad = [rid for rid, f in same_gender if f.sample_rate != cand.features.sample_rate]
        if sr_bad:
            item["reasons"].append(
                f"采样率与参考池不一致（{cand.features.sample_rate} vs "
                f"{refs[sr_bad[0]].sample_rate}），不可比")
            continue
        if not same_gender:
            item["reasons"].append("池内没有同性别参考声线，按独立可用处理")
            item["verdict"] = "independent"
            item["usable"] = True
            item["checks"]["duplicate"] = {"ok": True, "duplicates": []}
            item["checks"]["near"] = {"ok": True, "distance": None}
            continue

        # 判重
        dups = []
        for rid, f in same_gender:
            env = acoustics.env_distance(cand.features.env_primary, f.env_primary)
            st = acoustics.semitone(cand.features.f0, f.f0)
            if env < th.duplicate_env and st < th.duplicate_semitone:
                dups.append({"id": rid, "env_primary": round(env, 4),
                             "semitone": round(st, 2)})
        item["checks"]["duplicate"] = {"ok": not dups, "duplicates": dups,
                                       "env_threshold": th.duplicate_env,
                                       "semitone_threshold": th.duplicate_semitone}
        if dups:
            dups.sort(key=lambda x: x["env_primary"])
            item["duplicate_of"] = dups[0]["id"]
            item["verdict"] = "backup"
            item["reasons"].append(
                f"与 {dups[0]['id']} 实测同音：包络 {dups[0]['env_primary']} < "
                f"{th.duplicate_env}，半音差 {dups[0]['semitone']} < "
                f"{th.duplicate_semitone}；判为同一条，只能备用")

        # 最近邻 / 近音
        best = None
        for rid, f in same_gender:
            d = acoustics.timbre_distance(cand.features, f)
            c = acoustics.centered_distance(cand.features, f)
            e = acoustics.env_distance(cand.features.env_primary, f.env_primary)
            st = acoustics.semitone(cand.features.f0, f.f0)
            if best is None or d < best["distance"]:
                best = {"id": rid, "distance": round(d, 3),
                        "env_primary": round(e, 4), "env_centered": round(c, 4),
                        "semitone": round(st, 2)}
        item["nearest"] = best
        near = best is not None and (best["distance"] < th.near
                                     or best["env_centered"] < th.near_centered)
        item["checks"]["near"] = {
            "ok": not near, "threshold": th.near,
            "centered_threshold": th.near_centered,
            "distance": best["distance"] if best else None,
            "env_centered": best["env_centered"] if best else None,
        }
        if near and not dups:
            item["verdict"] = "backup"
            item["reasons"].append(
                f"与最近邻 {best['id']} 过近：复合距离 {best['distance']} < {th.near} "
                f"或 centered {best['env_centered']} < {th.near_centered}；只能备用")
        elif not dups:
            item["verdict"] = "independent"
            item["usable"] = True
            item["reasons"].append(
                f"最近邻 {best['id']} 复合距离 {best['distance']}、centered "
                f"{best['env_centered']}，均不低于近音阈值")

    summary = {"independent": 0, "backup": 0, "disabled": 0}
    for r in results:
        summary[r["verdict"]] += 1
    return {
        "lang": pool.get("lang"), "base_region": pool.get("base_region"),
        "probe_text": candidates[0].probe_text if candidates else "",
        "thresholds": {
            "duplicate_env": th.duplicate_env,
            "duplicate_semitone": th.duplicate_semitone,
            "near": th.near, "near_centered": th.near_centered,
        },
        "reference": {"voices": len(pool.get("voices", [])),
                      "measured": len(refs)},
        "candidates": results, "summary": summary,
        "acoustic_gate": acoustic_gate,
        "caliber_note": ("primary=线性 FFT bin 的 log10 包络 RMS（与 casting/voices 一致）；"
                         "centered=80–6000Hz 几何网格 + 9 点平滑 + 去均值（v12 口径）"),
    }


# --------------------------------------------------------------------------- 入池
def merge_into_pool(pool: dict, report: dict) -> dict:
    """把独立/备用声线并入池子，并重算近音组与容量。禁用项只留在报告里。"""
    merged = copy.deepcopy(pool)
    rows = merged.setdefault("voices", [])
    added = []
    for cand in report.get("candidates", []):
        if cand["verdict"] not in ("independent", "backup"):
            continue
        d = cand["declared"]
        row = {
            "id": cand["id"], "gender": d["gender"], "locale": d["locale"],
            "region": d["region"], "status": "ok",
            "distance": (cand.get("nearest") or {}).get("distance"),
            "nearest": (cand.get("nearest") or {}).get("id"),
            "near_group": None, "duplicate_of": cand.get("duplicate_of"),
            "scope": "main", "notes": list(cand.get("reasons", [])),
            "f0": cand["probe"].get("f0"), "f0_p25": cand["probe"].get("f0_p25"),
            "f0_p75": cand["probe"].get("f0_p75"),
            "probe_wav": cand["probe"].get("wav"),
            "usable": cand["verdict"] == "independent",
            "class": cand["verdict"], "is_narrator": False,
            "declared_age": d["age"], "source": cand["source"],
            "provider": cand["provider"],
            "admission": {
                "verdict": cand["verdict"], "reasons": cand.get("reasons", []),
                "duplicate_of": cand.get("duplicate_of"), "nearest": cand.get("nearest"),
                "checks": cand.get("checks", {}),
            },
        }
        rows.append(row)
        added.append(row)

    # 近音组：独立可用 + 同性别内复合距离 <0.18 用并查集连组
    feats = _load_reference(merged)
    parent = {r["id"]: r["id"] for r in rows
              if r.get("usable") and r.get("class") == "independent"
              and r["id"] in feats}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    gender_of = {r["id"]: r.get("gender") for r in rows}
    ids = list(parent)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if gender_of.get(a) != gender_of.get(b):
                continue
            if acoustics.timbre_distance(feats[a], feats[b]) < 0.18:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    groups: dict[str, list[str]] = {}
    for rid in ids:
        groups.setdefault(find(rid), []).append(rid)
    near_groups = [sorted(v) for v in groups.values() if len(v) > 1]
    for gi, g in enumerate(near_groups):
        for r in rows:
            if r["id"] in g:
                r["near_group"] = gi
    merged["near_groups"] = near_groups

    merged["main_pool"] = [r["id"] for r in rows
                           if r.get("usable") and r.get("class") == "independent"
                           and not r.get("is_narrator")]
    merged["counts"] = {
        "enumerated": len(rows),
        "independent": sum(1 for r in rows if r.get("class") == "independent"),
        "minor_only": sum(1 for r in rows if r.get("class") == "minor_only"),
        "backup": sum(1 for r in rows if r.get("class") == "backup"),
        "disabled": sum(1 for r in rows if r.get("class") == "disabled"),
        "unavailable": sum(1 for r in rows if r.get("class") == "unavailable"),
    }
    merged["voice_sources"] = sorted({r.get("provider") for r in added if r.get("provider")})
    merged["admitted"] = [r["id"] for r in added]
    return merged
