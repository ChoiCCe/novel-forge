"""edge 声线目录：可用性、平替、性别过滤（v4 步骤 1）。

设计要点：
- `list-voices` 不可信（Xiaoxuan 不在列表里却能合成），因此目录以“试合成结果”为准，
  `available` 字段是审计后的静态结论；运行时另有可选探测缓存。
- 失效声线不在 registry/config 里静默删除，而是保留原始引用并在这里映射到 `fallbacks`，
  调用方可以记录 `legacy_voice` 与替代原因。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

log = logging.getLogger("novel-forge.voices")

DEFAULT_CATALOG = "voices/catalog.edge.json"
PROBE_TEXT = "这一句用来检查声线是否可以正常合成。"


def load_catalog(path: str | Path | None = None, root: Path | None = None) -> dict:
    if path is None:
        path = DEFAULT_CATALOG
    p = Path(path)
    if not p.is_absolute():
        p = (root or Path(__file__).resolve().parents[1]) / p
    if not p.exists():
        return {"voices": [], "fallbacks": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        log.warning("声线目录解析失败：%s", err)
        return {"voices": [], "fallbacks": {}}


def _index(catalog: dict) -> dict[str, dict]:
    return {v.get("id"): v for v in catalog.get("voices", []) if v.get("id")}


def entry(voice: str, catalog: dict) -> dict:
    return _index(catalog).get(voice) or {}


def is_available(voice: str, catalog: dict) -> bool:
    entry = _index(catalog).get(voice)
    return bool(entry.get("available")) if entry else False


def voice_gender(voice: str, catalog: dict) -> str:
    entry = _index(catalog).get(voice)
    return (entry or {}).get("gender", "未知")


def voice_stages(voice: str, catalog: dict) -> list[str]:
    entry = _index(catalog).get(voice) or {}
    stages = entry.get("stages") or []
    return [str(item) for item in stages]


def voice_locale(voice: str, catalog: dict) -> str:
    return str(entry(voice, catalog).get("locale") or "")


def is_accented(voice: str, catalog: dict) -> bool:
    """方言/口音声线：catalog 显式标记优先，locale 前缀兜底。"""
    data = entry(voice, catalog)
    if data.get("accented") or data.get("dialect"):
        return True
    locale = str(data.get("locale") or "").lower()
    return any(locale.startswith(prefix)
               for prefix in ("zh-tw", "zh-hk", "zh-cn-liaoning", "zh-cn-shaanxi"))


def accent_tag(voice: str, catalog: dict) -> str:
    data = entry(voice, catalog)
    if data.get("dialect"):
        return str(data["dialect"])
    locale = str(data.get("locale") or "").lower()
    if locale.startswith("zh-tw"):
        return "台湾腔"
    if locale.startswith("zh-hk"):
        return "粤语"
    if "liaoning" in locale:
        return "东北"
    if "shaanxi" in locale:
        return "陕西"
    return ""


def enumerate_voices(catalog: dict, *, available_only: bool = True,
                     include_accented: bool = True) -> list[str]:
    result: list[str] = []
    for item in catalog.get("voices", []):
        voice = item.get("id")
        if not voice:
            continue
        if available_only and not item.get("available"):
            continue
        if not include_accented and is_accented(voice, catalog):
            continue
        result.append(voice)
    return result


def fallback_for(voice: str, catalog: dict) -> dict | None:
    fb = (catalog.get("fallbacks") or {}).get(voice)
    if isinstance(fb, str):
        return {"voice": fb}
    return fb


def resolve_voice(voice: str, catalog: dict) -> dict:
    """返回 {voice, legacy_voice, fallback(bool), note}。

    只做一次性替换；平替本身若也不可用则继续沿链走，最多 3 跳。
    """
    original = voice
    note = ""
    seen = {voice}
    for _ in range(3):
        if not voice:
            break
        if is_available(voice, catalog):
            break
        fb = fallback_for(voice, catalog)
        if not fb or not fb.get("voice") or fb["voice"] in seen:
            note = f"{voice} 不可用且无平替"
            break
        note = fb.get("note") or note
        seen.add(fb["voice"])
        voice = fb["voice"]
    return {
        "voice": voice,
        "legacy_voice": original if voice != original else "",
        "fallback": voice != original,
        "note": note if voice != original else "",
        "overrides": {k: fb[k] for k in ("rate", "pitch", "volume")
                      if isinstance(fb, dict) and fb.get(k) is not None} if voice != original else {},
    }


def special_voice(persona_text: str, gender: str, special_cfg: dict | None,
                  catalog: dict | None = None) -> str:
    """方言/口音声线：必须与角色性别一致，否则返回空串（硬性别闸）。

    special_cfg 兼容两种写法：
      - 旧：{东北: "zh-CN-liaoning-XiaobeiNeural"}
      - 新：{东北: {voice: "...", gender: "女"}}
    """
    if not special_cfg:
        return ""
    for keyword, spec in special_cfg.items():
        if not keyword or keyword not in (persona_text or ""):
            continue
        if isinstance(spec, str):
            voice, required = spec, ""
        else:
            voice, required = spec.get("voice", ""), spec.get("gender", "")
        if not voice:
            continue
        if required and gender not in ("男", "女"):
            continue
        if required and gender != required:
            log.info("方言声线 %s 性别不符（角色 %s），跳过", voice, gender)
            continue
        if catalog:
            vg = voice_gender(voice, catalog)
            if vg in ("男", "女") and gender in ("男", "女") and vg != gender:
                log.info("方言声线 %s 目录性别为 %s，角色为 %s，跳过", voice, vg, gender)
                continue
        return voice
    return ""


def probe_voices(voices: list[str], cache_path: Path, ttl_days: int = 7,
                 force: bool = False) -> dict[str, dict]:
    """对声线逐条试合成，结果缓存到 cache_path；失败重试 2 次。

    仅用于工具/后台；启动流程默认读缓存，避免拖慢 WebUI。
    """
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    ttl = ttl_days * 86400
    now = time.time()
    todo = [v for v in voices if force or v not in cache
            or now - float(cache[v].get("checked_at", 0)) > ttl]
    if todo:
        results = asyncio.run(_probe_many(todo))
        cache.update(results)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return {v: cache.get(v, {}) for v in voices}


async def _probe_many(voices: list[str]) -> dict[str, dict]:
    import edge_tts

    out: dict[str, dict] = {}
    for voice in voices:
        ok, err_type, err = False, "", ""
        for attempt in range(3):
            try:
                with _tempfile() as tmp:
                    mp3 = Path(tmp) / "p.mp3"
                    await edge_tts.Communicate(PROBE_TEXT, voice).save(str(mp3))
                    ok = mp3.exists() and mp3.stat().st_size > 0
                    if ok:
                        break
                    err_type = "empty_audio"
            except Exception as exc:  # noqa: BLE001
                name = type(exc).__name__
                err_type = "no_audio" if "NoAudioReceived" in name or "NoAudioReceived" in str(exc) else name
                err = str(exc)[:200]
            if attempt < 2:
                await asyncio.sleep(2)
        out[voice] = {"available": ok, "error_type": "" if ok else err_type,
                      "error": "" if ok else err, "checked_at": time.time()}
    return out


class _tempfile:
    def __enter__(self):
        import tempfile
        self._ctx = tempfile.TemporaryDirectory(prefix="vf-probe-")
        return self._ctx.__enter__()

    def __exit__(self, *args):
        return self._ctx.__exit__(*args)
