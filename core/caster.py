"""选角与音色锁定（v4：registry 只使用 v2 结构）。

角色 key 归一化为 `名字_性别_年龄段`（如 `楼温_男_中年`）。
用年龄段而不是精确年龄，是为了同一角色在不同章节被 LLM 估成 43/45 岁时仍能命中同一条档案。

持久化只写 v2：
- 身份与阶段档案：`identities[*].stage_profiles[*]`
- 声线与覆盖：`voice_bindings[*]`（每个阶段一条 `lock:{character_id}|{stage}` 全局锁）
- 变更历史：`voice_events`
不再读写 `characters` / `locks` / `voice_changes`。
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

from . import casting
from . import consistency
from . import dialect
from . import inference
from . import prosody
from . import voice_policy
from . import voices_catalog
from .casting import strip_title
from .prosody import TEMPERAMENTS, infer_temperament, persona_base

log = logging.getLogger("novel-forge.caster")

AGE_KEYWORDS = re.compile(
    r"岁|年龄|年|载|春秋|登基|即位|转眼|多年|翌年|次年|数月|日后|长成|长大|成年")

STAGE_FIELDS = (
    "temperament", "rate", "pitch", "volume", "prosody_source",
    "age_note", "age_confidence", "age_source", "first_seen_chapter",
    "last_seen_chapter", "updated_by", "updated_by_chapter", "checked_chapters",
    "legacy_edge_voice", "voice_repair", "transition", "stage_entered_chapter",
    "age_evidence_log", "age_hold_reason", "engine", "ref_wav", "ref_text",
    "class_name", "line_count",
)
IDENT_FIELDS = (
    "gender", "gender_confidence", "gender_evidence", "gender_source",
    "gender_confirmed", "generic", "persona", "voice_style", "default_emotion",
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def canonicalize_names(names: list[str]) -> dict[str, str]:
    """把包含已知短名的长名归并到短名（大将军楼温 -> 楼温）。"""
    stripped = {name: strip_title(name) for name in names}
    bases = set(stripped.values())
    mapping: dict[str, str] = {}
    for raw, base in stripped.items():
        candidates = [b for b in bases if b != base and len(b) >= 2 and b in base]
        mapping[raw] = max(candidates, key=len) if candidates else base
    return mapping


def _parse_age(value, cfg: dict) -> int:
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    digits = re.search(r"\d+", text)
    if digits:
        return int(digits.group(0))
    chinese = {"十": 10, "二十": 20, "三十": 30, "四十": 40, "五十": 50,
               "六十": 60, "七十": 70, "八十": 80, "九十": 90, "百": 100}
    for word, number in chinese.items():
        if word in text:
            return number
    return int(inference._value(cfg, ("casting", "age", "unknown_age_default")))


def age_band(age: int, cfg: dict) -> str:
    """按 config.yaml 的 casting.age.bands 分档，不在代码里写死 18/35/55。"""
    return inference.band_for_age(age, cfg)


def infer_gender(name: str) -> str:
    if any(word in name for word in ("夫人", "公主", "娘娘", "嬷嬷", "婢", "女", "娘", "姐", "妹", "姑", "婆", "太后")):
        return "女"
    if any(word in name for word in ("公子", "将军", "先生", "郎", "兄", "爷", "王", "帝", "僧", "道士", "大人")):
        return "男"
    return "未知"


def normalize_key(name: str, gender: str, band: str) -> str:
    safe = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", name) or "未知角色"
    gender = gender if gender in ("男", "女") else "未知"
    return f"{safe}_{gender}_{band}"


class Caster:
    """持有全局音色注册表；build 只负责本章选角，写入受锁保护。"""

    def __init__(self, cfg: dict, llm, project_root: Path):
        self.cfg = cfg
        self.llm = llm
        self.root = project_root
        self.voices_dir = (project_root / cfg["paths"]["voices"]).resolve()
        self.refs_dir = (project_root / cfg["tts"]["engines"]["cosyvoice"]["refs_dir"]).resolve()
        self.registry_path = self.voices_dir / "registry.json"
        self.lock = threading.Lock()
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
        self.stage_order = inference.stages(cfg)
        self.default_stage = str(inference._value(cfg, ("casting", "age", "default_stage")))
        self.generic_names = tuple(
            ((cfg.get("casting") or {}).get("consistency") or {}).get("generic_names")
            or casting.DEFAULT_GENERIC)
        self.age_schema = {
            "type": "object",
            "properties": {
                "ages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {},
                            "stage": {"type": "string", "enum": list(self.stage_order)},
                            "evidence": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["name"],
                    },
                }
            },
            "required": ["ages"],
        }
        edge_cfg = cfg["tts"]["engines"]["edge"]
        self.voice_map = edge_cfg.get("voice_map", {})
        self.stage_voice_map = edge_cfg.get("age_stage_voice_map", {})
        self.special_voices = edge_cfg.get("special", {})
        self.narrator_voice = (edge_cfg.get("narrator_voice")
                               or cfg["casting"]["narrator"]["default_voice"])
        # 声线阶梯：independent_first（默认）/ forbid / differentiate（旧）
        voice_rules = (cfg.get("casting", {}).get("voice_rules") or {})
        self.voice_policy = voice_policy.policy_of(cfg)
        self.forbid_narrator_voice = self.voice_policy == voice_policy.POLICY_FORBID
        self.voice_pools = voice_rules.get("voice_pools") or {}
        self.pool_source = str(voice_rules.get("pool_source") or "static")
        self.reuse_non_narrator = bool(voice_rules.get("reuse_non_narrator", True))
        self._active_cids: set[str] | None = None
        self.narrator_alternates = [str(v) for v in (voice_rules.get("narrator_alternates") or [])
                                    if str(v).strip()]
        catalog_path = (cfg.get("casting", {}).get("catalog", {}) or {}).get("path", "voices/catalog.edge.json")
        self.catalog = voices_catalog.load_catalog(catalog_path, project_root)
        self.voice_map_version = str(edge_cfg.get("voice_map_version", "v1"))
        # 惰性同步：缓存命中路径必须纯读，只有真正重建章节时才允许写 registry。
        self._voice_map_synced = False

    def sync_voice_map_version(self) -> None:
        """显式/重建路径调用：对齐 registry 的 voice_map_version 与旧档案。"""
        with self.lock:
            if self._voice_map_synced:
                return
            self._sync_voice_map_version()
            self._voice_map_synced = True

    # ------------------------------------------------------------------ registry 基础
    def _load_registry(self) -> dict:
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text(encoding="utf-8"))
                if int(data.get("schema_version", 1)) < 2:
                    raise RuntimeError(
                        "registry.json 仍是 v1；请先运行 `python tools/migrate_registry.py` "
                        "完成离线迁移，运行时不提供 legacy 兼容读取。")
                for key in ("identities", "character_aliases", "voice_bindings",
                            "voice_events", "book_settings"):
                    data.setdefault(key, {})
                data.setdefault("narrator", {})
                return data
            except json.JSONDecodeError:
                log.warning("音色注册表损坏，重建：%s", self.registry_path)
        return {
            "version": 2, "schema_version": 2, "identities": {},
            "character_aliases": {}, "voice_bindings": {}, "voice_events": [],
            "narrator": {"global": {}, "books": {}}, "book_settings": {},
        }

    def _save_registry(self) -> None:
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.registry, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.registry_path)

    def _identities(self) -> dict:
        return self.registry.setdefault("identities", {})

    def _aliases(self) -> dict:
        return self.registry.setdefault("character_aliases", {})

    def _bindings(self) -> dict:
        return self.registry.setdefault("voice_bindings", {})

    def _identity(self, name: str, book_id: str | None = None, create: bool = False,
                  gender: str = "未知") -> dict | None:
        """按姓名/别名取身份；create=True 时补建，并登记别名。"""
        aliases = self._aliases()
        canonical = strip_title(name or "")
        cid = aliases.get(name) or aliases.get(canonical)
        if cid is None:
            cid = casting.identity_id(name, book_id, self.generic_names)
        ident = self._identities().get(cid)
        if ident is None:
            if not create:
                return None
            ident = {
                "id": cid,
                "canonical_name": canonical or name,
                "aliases": [],
                "gender": gender,
                "gender_confidence": 0.0,
                "gender_evidence": "",
                "gender_source": "auto",
                "gender_confirmed": False,
                "generic": casting.is_generic(name, self.generic_names),
                "persona": "",
                "voice_style": "",
                "default_emotion": "平静",
                "books": [],
                "stage_profiles": {},
                "created_at": _now(),
                "updated_at": _now(),
            }
            self._identities()[cid] = ident
        aliases[name] = cid
        aliases[canonical] = cid
        if name != ident.get("canonical_name") and name not in ident.setdefault("aliases", []):
            ident["aliases"].append(name)
        if book_id and book_id not in ident.setdefault("books", []):
            ident["books"].append(book_id)
        return ident

    def _branch(self, name: str, stage: str) -> dict:
        """身份 + 阶段档案（调用方需自行 _store_profile）。"""
        ident = self._identity(name, create=True)
        return {
            "ident": ident,
            "stage": stage,
            "profile": ident.setdefault("stage_profiles", {}).setdefault(stage, {}),
        }

    def _lock_id(self, cid: str, stage: str) -> str:
        return f"lock:{cid}|{stage}"

    def _write_binding(self, cid: str, stage: str, voice: str, *, source: str,
                       updated_by: str, legacy_voice: str = "",
                       ladder_level: int = 0, accent: str = "") -> dict:
        bindings = self._bindings()
        bid = self._lock_id(cid, stage)
        existing = bindings.get(bid) or {}
        bindings[bid] = {
            "id": bid,
            "scope": "character_stage",
            "book_id": None,
            "character_id": cid,
            "age_stage": stage,
            "voice": voice,
            "rate": existing.get("rate"),
            "pitch": existing.get("pitch"),
            "volume": existing.get("volume"),
            "source": source,
            "locked": True,
            "priority": casting.PRIORITY["character_stage"],
            "note": existing.get("note") or "角色阶段锁",
            "legacy_voice": legacy_voice or existing.get("legacy_voice", ""),
            "ladder_level": int(ladder_level or existing.get("ladder_level") or 0),
            "accent": accent or existing.get("accent") or "",
            "created_at": existing.get("created_at") or _now(),
            "updated_at": _now(),
            "updated_by": updated_by,
        }
        return bindings[bid]

    def _source_for(self, updated_by: str) -> str:
        value = str(updated_by or "").lower()
        if value in ("manual", "webui"):
            return "manual"
        if value == "migrated":
            return "migrated"
        if value == "llm":
            return "llm"
        return "auto"

    def _profile_view(self, ident: dict, stage: str) -> dict:
        """把 v2 身份/阶段/绑定组装成调用方使用的扁平 profile（不写盘）。"""
        stage = stage if stage in self.stage_order else self.default_stage
        branch = ident.get("stage_profiles", {}).get(stage, {}) or {}
        gender = ident.get("gender") or "未知"
        name = ident.get("canonical_name") or ident.get("id") or ""
        key = normalize_key(name, gender, stage)
        binding = self._bindings().get(self._lock_id(ident.get("id", ""), stage)) or {}
        persona = str(ident.get("persona") or f"{gender}{stage}角色")
        voice_style = str(ident.get("voice_style") or "")
        voice = binding.get("voice") or self._voice_for_stage(gender, stage, persona + voice_style)
        resolved = voices_catalog.resolve_voice(voice, self.catalog)
        ref_wav = branch.get("ref_wav")
        ref_text = branch.get("ref_text")
        if ref_wav is None:
            candidate = self.refs_dir / f"{key}.wav"
            ref_wav = str(candidate) if candidate.exists() else None
        if ref_text is None:
            candidate = self.refs_dir / f"{key}.txt"
            ref_text = candidate.read_text(encoding="utf-8").strip() if candidate.exists() else None
        return {
            "key": key,
            "name": name,
            "character_id": ident.get("id"),
            "gender": gender,
            "gender_confidence": ident.get("gender_confidence", 0.0),
            "gender_evidence": ident.get("gender_evidence", ""),
            "gender_source": ident.get("gender_source", "auto"),
            "gender_confirmed": bool(ident.get("gender_confirmed")),
            "generic": bool(ident.get("generic")),
            "persona": persona,
            "voice_style": voice_style,
            "default_emotion": ident.get("default_emotion") or "平静",
            "age_band": stage,
            "age_stage": stage,
            "temperament": branch.get("temperament") or "中性",
            "prosody_source": branch.get("prosody_source") or "temperament",
            "age_note": branch.get("age_note", ""),
            "age_confidence": branch.get("age_confidence"),
            "age_source": branch.get("age_source"),
            "first_seen_chapter": branch.get("first_seen_chapter"),
            "last_seen_chapter": branch.get("last_seen_chapter"),
            "updated_by": branch.get("updated_by", "migrated"),
            "updated_by_chapter": branch.get("updated_by_chapter"),
            "checked_chapters": list(branch.get("checked_chapters") or []),
            "transition": branch.get("transition"),
            "stage_entered_chapter": branch.get("stage_entered_chapter"),
            "age_evidence_log": list(branch.get("age_evidence_log") or []),
            "age_hold_reason": branch.get("age_hold_reason", ""),
            "legacy_edge_voice": binding.get("legacy_voice") or branch.get("legacy_edge_voice", ""),
            "voice_repair": branch.get("voice_repair") or resolved.get("note", ""),
            "engine": branch.get("engine") or self.cfg["caster"]["dialogue_engine"],
            "edge_voice": resolved["voice"],
            "voice_source": binding.get("source") or "config",
            "ladder_level": int(binding.get("ladder_level") or 0),
            "accent": binding.get("accent") or "",
            "dialect_profile": dict(ident.get("dialect_profile") or {}),
            "binding_id": binding.get("id"),
            "voice_priority": binding.get("priority") or casting.PRIORITY["config"],
            "voice_chain": [],
            "locked": bool(binding),
            "rate": branch.get("rate") or "+0%",
            "pitch": branch.get("pitch") or "+0Hz",
            "volume": branch.get("volume") or "+0%",
            "voice": 0,
            "sid": 0,
            "speed": 1.0,
            "ref_wav": ref_wav,
            "ref_text": ref_text,
            "class_name": branch.get("class_name") or "角色",
            "line_count": branch.get("line_count", 0),
        }

    def _store_profile(self, profile: dict, book_id: str | None = None) -> dict:
        """把扁平 profile 的身份字段/阶段字段写回 v2；声线只由 _locked_voice 写 binding。"""
        name = profile.get("name", "")
        stage = profile.get("age_stage") or profile.get("age_band") or self.default_stage
        ident = self._identity(name, book_id, create=True,
                               gender=profile.get("gender", "未知"))
        if ident is None:
            return profile
        for field in IDENT_FIELDS:
            if field in profile:
                ident[field] = profile[field]
        branch = ident.setdefault("stage_profiles", {}).setdefault(stage, {})
        for field in STAGE_FIELDS:
            if field in profile:
                branch[field] = profile[field]
        branch["locked"] = bool(profile.get("locked"))
        ident["updated_at"] = _now()
        return profile

    def iter_profiles(self) -> list[dict]:
        """所有身份的所有阶段档案（含历史阶段），用于 WebUI/同步。"""
        rows: list[dict] = []
        for ident in self._identities().values():
            stages = sorted(ident.get("stage_profiles", {}).keys(),
                            key=lambda s: inference.band_index(s, self.cfg))
            for stage in stages:
                rows.append(self._profile_view(ident, stage))
        return rows

    def profile_history(self, name: str) -> list[dict]:
        ident = self._identity(name)
        if not ident:
            return []
        stages = sorted(ident.get("stage_profiles", {}).keys(),
                        key=lambda s: inference.band_index(s, self.cfg))
        return [self._profile_view(ident, stage) for stage in stages]

    def profile_by_key(self, key: str) -> dict | None:
        for profile in self.iter_profiles():
            if profile.get("key") == key:
                return profile
        return None

    def binding_for(self, name: str, stage: str | None = None,
                    book_id: str | None = None) -> dict | None:
        ident = self._identity(name, book_id)
        if not ident:
            return None
        stages = [stage] if stage else list(ident.get("stage_profiles", {}).keys())
        for item in stages:
            binding = self._bindings().get(self._lock_id(ident["id"], item))
            if binding:
                return dict(binding, age_stage=item)
        return None

    def voice_events(self) -> list[dict]:
        return list(self.registry.get("voice_events") or [])

    def registry_names(self) -> list[str]:
        names: list[str] = []
        for ident in self._identities().values():
            for value in [ident.get("canonical_name")] + list(ident.get("aliases") or []):
                if value and value not in names:
                    names.append(value)
        return names

    # ------------------------------------------------------------------ 声线解析
    def _static_pool(self, gender: str, stage: str) -> list[str]:
        stage_cfg = self.voice_pools.get(stage) or {}
        pool = stage_cfg.get(gender) or ([] if gender in ("男", "女")
                                         else stage_cfg.get("未知") or [])
        result: list[str] = []
        for name in pool:
            resolved = voices_catalog.resolve_voice(name, self.catalog)["voice"]
            if not resolved or resolved == self.narrator_voice:
                continue
            if not voices_catalog.is_available(resolved, self.catalog):
                continue
            if voices_catalog.is_accented(resolved, self.catalog):
                continue
            voice_gender = voices_catalog.voice_gender(resolved, self.catalog)
            if gender in ("男", "女") and voice_gender in ("男", "女") \
                    and voice_gender != gender:
                continue
            if resolved not in result:
                result.append(resolved)
        return result

    def _catalog_pool(self, gender: str, stage: str, *, accented: bool = False,
                      accent: str = "") -> list[str]:
        """从 catalog 动态取池：available、性别/阶段兼容、排除旁白；accented 需显式开启。"""
        result: list[str] = []
        for item in self.catalog.get("voices") or []:
            voice = item.get("id")
            if not voice or not item.get("available") or voice == self.narrator_voice:
                continue
            is_accented = voices_catalog.is_accented(voice, self.catalog)
            if accented != is_accented:
                continue
            if accented and accent \
                    and voices_catalog.accent_tag(voice, self.catalog) != accent:
                continue
            voice_gender = item.get("gender") or "未知"
            if gender in ("男", "女") and voice_gender in ("男", "女") \
                    and voice_gender != gender:
                continue
            stages = [str(v) for v in (item.get("stages") or [])]
            if stage and stage in self.stage_order and stages and stage not in stages:
                continue
            result.append(voice)
        return result

    def _pool_voices(self, gender: str, stage: str, *, accented: bool = False,
                     accent: str = "") -> list[str]:
        """该 stage/gender 的声线池。

        catalog 模式下：静态 voice_pools 只提供优先顺序，实际可用集合以 catalog 为准；
        accented=True 时只返回方言声线（调用方必须先校验角色证据）。
        """
        if accented:
            return self._catalog_pool(gender, stage, accented=True, accent=accent)
        if self.pool_source != "catalog":
            return self._static_pool(gender, stage)
        ordered = self._static_pool(gender, stage)
        rest = self._catalog_pool(gender, stage, accented=False)
        result: list[str] = []
        for voice in ordered + rest:
            if voice not in result:
                result.append(voice)
        return result

    def _used_voices(self, exclude_cid: str | None = None) -> set[str]:
        return set(self._voice_usage(exclude_cid))

    def _voice_usage(self, exclude_cid: str | None = None) -> dict[str, int]:
        usage: dict[str, int] = {}
        active = getattr(self, "_active_cids", None)
        for key, binding in self._bindings().items():
            if exclude_cid and key.startswith(f"lock:{exclude_cid}|"):
                continue
            cid = binding.get("character_id") or ""
            if active is not None and cid not in active:
                continue
            voice = binding.get("voice")
            if voice:
                usage[voice] = usage.get(voice, 0) + 1
        return usage

    def _allocate_ladder(self, ident: dict | None, gender: str, stage: str,
                         persona_text: str = "", chapter_index: int | None = None,
                         exclude_cid: str | None = None, updated_by: str = "auto") -> dict:
        """四级阶梯：独立声线 → 复用非旁白 → 旁白 → 未知兜底。"""
        if ident is not None and (ident.get("generic") or gender == "未知"):
            return {"voice": self.narrator_voice, "level": 4,
                    "label": voice_policy.LABEL_UNKNOWN, "accent": "",
                    "dialect_evidence": {}, "reason": "未知/不明确角色"}
        evidence = self._dialect_evidence(ident, persona_text, chapter_index) if ident else {}
        accent = str(evidence.get("accent") or "")
        allowed_accent = self._pool_voices(gender, stage, accented=True, accent=accent) \
            if accent else []
        main = self._pool_voices(gender, stage)
        ordered = allowed_accent + main
        # 去重且保留“方言优先”顺序
        candidates: list[str] = []
        for voice in ordered:
            if voice not in candidates:
                candidates.append(voice)
        used = self._voice_usage(exclude_cid)
        for voice in candidates:
            if voice not in used:
                level = 1
                return {"voice": voice, "level": level,
                        "label": voice_policy.LABEL_INDEPENDENT,
                        "accent": accent if voice in allowed_accent else "",
                        "dialect_evidence": evidence,
                        "reason": "独立声线" if not accent else "方言证据授权的独立声线"}
        if self.reuse_non_narrator and candidates:
            usage = used
            voice = min(candidates, key=lambda item: (
                usage.get(item, 0), candidates.index(item)))
            return {"voice": voice, "level": 2, "label": voice_policy.LABEL_SHARED,
                    "accent": accent if voice in allowed_accent else "",
                    "dialect_evidence": evidence, "reason": "复用非旁白声线"}
        return {"voice": self.narrator_voice, "level": 3,
                "label": voice_policy.LABEL_BORROWED, "accent": "",
                "dialect_evidence": evidence, "reason": "非旁白池已耗尽"}

    def _allocate_independent_voice(self, gender: str, stage: str, persona_text: str = "",
                                    exclude_cid: str | None = None) -> str:
        """兼容入口：只返回第一级（未占用的非方言独立声线），池耗尽返回空串。"""
        used = self._used_voices(exclude_cid)
        for candidate in self._pool_voices(gender, stage):
            if candidate not in used:
                return candidate
        return ""

    def _dialect_evidence(self, ident: dict | None, persona_text: str = "",
                          chapter_index: int | None = None) -> dict:
        """读取/扫描方言证据；无证据或置信度不足返回空字典（绝不兜底到方言声线）。"""
        if ident is None or not dialect.enabled(self.cfg):
            return {}
        current = dialect.normalize(ident.get("dialect_profile"), self.cfg)
        if current:
            return current
        found = dialect.detect_evidence(persona_text, self.cfg)
        if not found:
            return {}
        found["chapter"] = chapter_index
        ident["dialect_profile"] = found
        ident["updated_at"] = _now()
        log.info("方言证据：%s -> %s（%s）", ident.get("canonical_name"),
                 found.get("dialect"), str(found.get("evidence"))[:40])
        return found

    def _record_dialect_persona(self, ident: dict, persona: dict, persona_text: str,
                                chapter_index: int | None) -> bool:
        """LLM 人设证据优先；没有则退回原文关键词扫描。已有合法证据不覆盖。"""
        if ident is None or dialect.normalize(ident.get("dialect_profile"), self.cfg):
            return False
        record = dialect.from_persona(persona, self.cfg)
        if not record:
            record = dialect.detect_evidence(persona_text, self.cfg)
        if not record:
            return False
        record["chapter"] = chapter_index
        ident["dialect_profile"] = record
        ident["updated_at"] = _now()
        log.info("方言证据落盘：%s -> %s (%s)", ident.get("canonical_name"),
                 record.get("dialect"), record.get("method"))
        return True

    def set_dialect_evidence(self, name: str, dialect_tag: str, evidence: str,
                             chapter_index: int | None = None,
                             confidence: float = 1.0, source: str = "manual") -> dict | None:
        """手动写入方言证据（UI/测试）；无原文片段直接拒绝，避免“口头指定”无据可查。"""
        if not str(evidence or "").strip():
            return None
        with self.lock:
            ident = self._identity(name, create=False)
            if ident is None:
                return None
            tag = dialect.normalize_tag(dialect_tag, self.cfg)
            if not tag:
                return None
            record = {
                "dialect": tag, "accent": tag, "region": "",
                "evidence": str(evidence).strip()[:120],
                "confidence": float(confidence), "method": source,
                "keyword": "", "chapter": chapter_index,
            }
            if dialect.normalize(record, self.cfg) is None:
                return None
            ident["dialect_profile"] = record
            ident["updated_at"] = _now()
            self._save_registry()
            return record

    def _voice_for_stage(self, gender: str, stage: str, persona_text: str = "") -> str:
        """性别 + 年龄阶段 → edge 声线。

        方言声线不在这里做关键词兜底：必须由 _allocate_ladder 拿到角色证据后才可选。
        失效声线（如 Xiaoshuang/Yunye）统一映射到 catalog 的平替。
        """
        stage = stage if stage in self.stage_order else self.default_stage
        by_gender = self.stage_voice_map.get(stage, {})
        if gender in ("男", "女") and by_gender.get(gender):
            candidate = by_gender[gender]
        elif by_gender.get("未知"):
            candidate = by_gender["未知"]
        elif stage == "少年":
            candidate = self.voice_map.get("少年") or self.narrator_voice
        else:
            category = f"{gender}{stage}" if gender in ("男", "女") else "未知"
            candidate = self.voice_map.get(category) or self.voice_map.get("未知") or self.narrator_voice
        return self._avoid_narrator_voice(
            voices_catalog.resolve_voice(candidate, self.catalog)["voice"],
            gender, stage)

    def _avoid_narrator_voice(self, voice: str, gender: str, stage: str) -> str:
        """forbid 策略：角色声线不得等于旁白声线，命中时按替代清单换同性别声线。

        differentiate 策略下不做替换，改由 apply_voice_policy 用参数拉开差异。
        """
        if self.voice_policy != voice_policy.POLICY_FORBID:
            return voice
        if not voice or voice != self.narrator_voice:
            return voice
        for candidate in self.narrator_alternates:
            chosen = voices_catalog.resolve_voice(candidate, self.catalog)["voice"]
            if not chosen or chosen == self.narrator_voice:
                continue
            chosen_gender = voices_catalog.voice_gender(chosen, self.catalog)
            if gender in ("男", "女") and chosen_gender in ("男", "女") and chosen_gender != gender:
                continue
            log.info("角色声线与旁白相同（%s），改用 %s", voice, chosen)
            return chosen
        log.warning("角色声线与旁白相同（%s），且未配置可用替代声线", voice)
        return voice

    def _binding_level(self, ident: dict, binding: dict) -> int:
        """阶梯级别：4 未知/不明确、3 借旁白、2 复用非旁白、1 独立。

        级别按“本章同时在场”的实际占用重算，不信任历史 stored 值：
        角色从共用声线迁走后，旧 level=2 必须能自动降回 1。
        """
        manual = str(binding.get("updated_by") or "").lower() in ("manual", "webui")
        if not manual and (ident.get("generic") or ident.get("gender") == "未知"):
            return 4
        voice = binding.get("voice") or ""
        if voice == self.narrator_voice:
            return 3
        active = getattr(self, "_active_cids", None)
        stored = int(binding.get("ladder_level") or 0)
        if active is None and stored in (1, 2):
            # 跨章全局同步没有“同时在场”语义，保留本章作用域内算出的级别
            return stored
        bid = binding.get("id") or ""
        for other_id, other in self._bindings().items():
            if other_id == bid or other.get("voice") != voice:
                continue
            cid = other.get("character_id") or ""
            if active is not None and cid not in active:
                continue
            return 2
        return 1

    def _recompute_levels(self) -> None:
        """在当前作用域内重算所有在场角色的阶梯级别（bindings 只存一处）。"""
        active = getattr(self, "_active_cids", None)
        if active is None:
            return
        for binding in self._bindings().values():
            cid = binding.get("character_id") or ""
            if cid not in active:
                continue
            ident = self._identities().get(cid)
            if not ident:
                continue
            level = self._binding_level(ident, binding)
            if int(binding.get("ladder_level") or 0) != level:
                binding["ladder_level"] = level

    def _migrate_binding(self, binding: dict, alloc: dict, source: str = "") -> None:
        binding["previous_voice"] = binding.get("voice")
        binding["voice"] = alloc["voice"]
        binding["ladder_level"] = int(alloc.get("level") or 0)
        binding["accent"] = str(alloc.get("accent") or "")
        binding["source"] = source or f"ladder_{alloc.get('label') or 'auto'}"
        binding["updated_by"] = binding["source"]
        binding["updated_at"] = _now()

    def _locked_voice(self, name: str, gender: str, stage: str, persona_text: str = "",
                      create: bool = False, chapter_index: int | None = None,
                      age_note: str = "", age_confidence: float | None = None,
                      updated_by: str = "auto", book_id: str | None = None) -> str:
        ident = self._identity(name, book_id, create=create, gender=gender)
        if ident is None:
            return self._voice_for_stage(gender, stage, persona_text)
        cid = ident["id"]
        if getattr(self, "_active_cids", None) is not None:
            self._active_cids.add(cid)
        had_dialect = bool(dialect.normalize(ident.get("dialect_profile"), self.cfg))
        binding = self._bindings().get(self._lock_id(cid, stage))
        if binding and binding.get("voice"):
            changed = False
            manual = str(binding.get("updated_by", "")).lower() in ("manual", "webui")
            if not manual:
                # 每章刷新时补扫证据；已落盘的合法证据不会重复写
                if self._dialect_evidence(ident, persona_text, chapter_index):
                    if not had_dialect:
                        changed = True
                    had_dialect = True
            resolved = voices_catalog.resolve_voice(binding["voice"], self.catalog)
            if resolved["fallback"]:
                # 保留原始声线，显式记录替代原因，不静默覆盖
                binding.setdefault("legacy_voice", binding["voice"])
                binding["voice"] = resolved["voice"]
                binding["updated_at"] = _now()
                changed = True
            accented = voices_catalog.is_accented(binding["voice"], self.catalog)
            if not manual and accented:
                evidence = self._dialect_evidence(ident, persona_text, chapter_index)
                if not evidence:
                    # 方言声线必须由证据授权；证据撤回/缺失时迁回非方言链路
                    alloc = self._allocate_ladder(
                        ident, gender, stage, persona_text, chapter_index,
                        exclude_cid=cid, updated_by=updated_by)
                    self._migrate_binding(binding, alloc, "dialect_evidence_revoked")
                    changed = True
                    accented = False
            if not manual:
                unknown_like = bool(ident.get("generic")) or gender == "未知"
                if unknown_like and binding["voice"] != self.narrator_voice:
                    # 第 4 级：未知/不明确角色统一回旁白声线 + 独立参数档
                    alloc = self._allocate_ladder(
                        ident, gender, stage, persona_text, chapter_index,
                        exclude_cid=cid, updated_by=updated_by)
                    self._migrate_binding(binding, alloc, "ladder_unknown")
                    changed = True
                elif not unknown_like and binding["voice"] != self.narrator_voice \
                        and not accented and getattr(self, "_active_cids", None) is not None:
                    # 阶梯自愈：声线被其他角色共用且本 stage 还有空闲声线时，迁到第 1 级
                    # 只在“本章同时在场”作用域内判断，跨章全局同步不重排声线
                    usage = self._voice_usage(exclude_cid=cid)
                    if usage.get(binding["voice"], 0) >= 1:
                        free = [voice for voice in self._pool_voices(gender, stage)
                                if voice not in usage]
                        if free:
                            alloc = {"voice": free[0], "level": 1,
                                     "label": voice_policy.LABEL_INDEPENDENT, "accent": ""}
                            self._migrate_binding(binding, alloc, "ladder_independent")
                            changed = True
                voice_gender = voices_catalog.voice_gender(binding["voice"], self.catalog)
                if (gender in ("男", "女") and voice_gender in ("男", "女")
                        and voice_gender != gender):
                    alloc = self._allocate_ladder(
                        ident, gender, stage, persona_text, chapter_index,
                        exclude_cid=cid, updated_by=updated_by)
                    self._migrate_binding(binding, alloc, "voice_gender_fix")
                    changed = True
                if (self.voice_policy == voice_policy.POLICY_INDEPENDENT
                        and binding["voice"] == self.narrator_voice):
                    # 阶梯自愈：只要还有非旁白可用（空闲优先，其次复用），就迁出旁白
                    alloc = self._allocate_ladder(
                        ident, gender, stage, persona_text, chapter_index,
                        exclude_cid=cid, updated_by=updated_by)
                    if alloc["voice"] != self.narrator_voice:
                        self._migrate_binding(binding, alloc)
                        changed = True
                if (self.voice_policy in (voice_policy.POLICY_DIFFERENTIATE,
                                          voice_policy.POLICY_INDEPENDENT)
                        and str(binding.get("source")) == "collision_fix"):
                    # 旧的一刀切撞声替换结果作废：恢复角色自然声线，改由参数差异化处理
                    natural = self._voice_for_stage(gender, stage, persona_text)
                    if natural and natural != binding["voice"]:
                        binding["previous_voice"] = binding["voice"]
                        binding["voice"] = natural
                        binding["source"] = "policy_restore"
                        binding["updated_by"] = "policy_restore"
                        binding["updated_at"] = _now()
                        changed = True
                    elif natural:
                        # 自然声线本来就与旁白不同：保留声线，只清掉旧的 collision_fix 标记
                        binding["source"] = "policy_reeval"
                        binding["updated_by"] = "policy_reeval"
                        binding["updated_at"] = _now()
                        changed = True
                fixed = self._avoid_narrator_voice(binding["voice"], gender, stage)
                if fixed != binding["voice"]:
                    binding.setdefault("legacy_voice", binding["voice"])
                    binding["voice"] = fixed
                    binding["source"] = "collision_fix"
                    binding["updated_by"] = "collision_fix"
                    binding["updated_at"] = _now()
                    changed = True
            else:
                # 手动指定方言声线 = 用户授权证据，补录来源便于复核
                if voices_catalog.is_accented(binding["voice"], self.catalog) and not had_dialect:
                    ident["dialect_profile"] = dialect.manual_evidence(
                        voices_catalog.accent_tag(binding["voice"], self.catalog),
                        f"WebUI 手动指定声线 {binding['voice']}", chapter_index)
                    changed = True
            level = self._binding_level(ident, binding)
            if int(binding.get("ladder_level") or 0) != level:
                binding["ladder_level"] = level
                changed = True
            accent_tag = voices_catalog.accent_tag(binding["voice"], self.catalog) \
                if voices_catalog.is_accented(binding["voice"], self.catalog) else ""
            if str(binding.get("accent") or "") != accent_tag:
                binding["accent"] = accent_tag
                changed = True
            if not had_dialect and dialect.normalize(ident.get("dialect_profile"), self.cfg):
                changed = True
            if changed:
                self._save_registry()
            return binding["voice"]
        if self.voice_policy == voice_policy.POLICY_INDEPENDENT:
            alloc = self._allocate_ladder(
                ident, gender, stage, persona_text, chapter_index,
                exclude_cid=cid, updated_by=updated_by)
            voice = alloc["voice"]
            if create:
                self._write_binding(cid, stage, voice, source=self._source_for(updated_by),
                                    updated_by=updated_by, ladder_level=alloc["level"],
                                    accent=alloc["accent"])
        else:
            voice = self._voice_for_stage(gender, stage, persona_text)
            if create:
                self._write_binding(cid, stage, voice, source=self._source_for(updated_by),
                                    updated_by=updated_by)
        if not had_dialect and dialect.normalize(ident.get("dialect_profile"), self.cfg):
            self._save_registry()
        return voice

    def _sync_profile(self, profile: dict, book_id: str | None = None) -> bool:
        """按当前配置对齐引擎、年龄阶段与 edge 声线；声线取角色阶段锁。"""
        changed = False
        engine = self.cfg["caster"]["dialogue_engine"]
        if profile.get("engine") != engine:
            profile["engine"] = engine
            changed = True
        stage = profile.get("age_stage") or profile.get("age_band") or self.default_stage
        if profile.get("age_stage") != stage:
            profile["age_stage"] = stage
            changed = True
        if "age_note" not in profile:
            profile["age_note"] = ""
            changed = True
        profile.setdefault("age_confidence", None)
        profile.setdefault("first_seen_chapter", None)
        profile.setdefault("last_seen_chapter", None)
        profile.setdefault("updated_by", "auto")
        profile.setdefault("checked_chapters", [])
        if profile.get("temperament") not in TEMPERAMENTS:
            profile["temperament"] = infer_temperament(
                f"{profile.get('persona', '')}{profile.get('voice_style', '')}")
            changed = True
        if profile.get("prosody_source") != "manual" and not profile.get("transition"):
            base = persona_base(profile["temperament"], stage, self.cfg)
            for key in ("rate", "pitch", "volume"):
                if profile.get(key) != base[key]:
                    profile[key] = base[key]
                    changed = True
            if profile.get("prosody_source") != "temperament":
                profile["prosody_source"] = "temperament"
                changed = True
        else:
            profile.setdefault("volume", "+0%")
        voice = self._locked_voice(
            profile.get("name", ""), profile.get("gender", "未知"), stage,
            f"{profile.get('persona', '')}{profile.get('voice_style', '')}",
            book_id=book_id)
        if voice and profile.get("edge_voice") != voice:
            old = profile.get("edge_voice")
            if old:
                repaired = voices_catalog.resolve_voice(old, self.catalog)
                if repaired["fallback"]:
                    profile.setdefault("legacy_edge_voice", old)
                    profile["voice_repair"] = repaired["note"]
            profile["edge_voice"] = voice
            changed = True
        ident = self._identity(profile.get("name", ""), book_id)
        binding = (self._bindings().get(self._lock_id(ident["id"], stage)) or {}) if ident else {}
        if binding.get("legacy_voice"):
            profile.setdefault("legacy_edge_voice", binding["legacy_voice"])
            if profile.get("prosody_source") != "manual" and not profile.get("transition"):
                fallback = voices_catalog.fallback_for(binding["legacy_voice"], self.catalog) or {}
                for key in ("rate", "pitch", "volume"):
                    if fallback.get(key):
                        unit = "Hz" if key == "pitch" else "%"
                        current = prosody.parse_value(profile.get(key) or "0", unit)
                        delta = prosody.parse_value(fallback[key], unit)
                        profile[key] = prosody.format_value(current + delta, unit)
                        changed = True
        changed |= self._apply_binding(profile, book_id)
        return changed

    def _apply_binding(self, profile: dict, book_id: str | None) -> bool:
        """应用绑定层覆盖，并把生效来源写进档案（可追溯）。"""
        name = profile.get("name", "")
        stage = profile.get("age_stage") or self.default_stage
        resolved = casting.resolve_effective(
            self.registry, name=name, age_stage=stage, book_id=book_id,
            gender=profile.get("gender", ""), config_voice=profile.get("edge_voice", ""),
            catalog=self.catalog, generic_names=self.generic_names)
        changed = False
        if resolved.get("binding_id"):
            if resolved.get("voice") and profile.get("edge_voice") != resolved["voice"]:
                profile["edge_voice"] = resolved["voice"]
                changed = True
            for key in ("rate", "pitch", "volume"):
                if resolved.get(key) and profile.get(key) != resolved[key]:
                    profile[key] = resolved[key]
                    changed = True
        source = resolved.get("source", "config")
        if profile.get("voice_source") != source:
            profile["voice_source"] = source
            changed = True
        profile["binding_id"] = resolved.get("binding_id")
        profile["voice_priority"] = resolved.get("priority")
        profile["voice_chain"] = resolved.get("chain", [])[:4]
        profile["locked"] = bool(resolved.get("binding_id"))
        return changed

    def apply_voice_policy(self, cast_data: dict, book_id: str | None = None) -> dict:
        """按 narrator_voice_policy 评估同声线角色，并把微调写进 binding（source=param_fix）。

        调用方需自行持有 self.lock（refresh_cast 已持有；build 内单独加锁）。
        """
        cast = cast_data.get("cast") or {}
        narrator = cast_data.get("narrator") or self.narrator_profile(book_id)
        report = voice_policy.plan(cast, narrator, self.cfg)
        cast_data["voice_policy"] = report
        if report.get("policy") == voice_policy.POLICY_FORBID:
            return report
        changed = False
        for adj in report.get("adjustments") or []:
            profile = cast.get(adj.get("key")) or {}
            if not profile:
                continue
            stage = profile.get("age_stage") or profile.get("age_band") or self.default_stage
            bid = adj.get("binding_id")
            binding = self._bindings().get(bid) if bid else None
            if binding is None:
                cid = profile.get("character_id") or ""
                if not cid:
                    continue
                binding = self._write_binding(
                    cid, stage, profile.get("edge_voice") or "",
                    source="param_fix", updated_by="param_fix")
                bid = binding["id"]
            for key in ("rate", "pitch", "volume"):
                binding[key] = adj["to"][key]
            binding["source"] = "param_fix"
            binding["updated_by"] = "param_fix"
            binding["updated_at"] = _now()
            binding["note"] = "同声线参数区分自动微调"
            profile.update(adj["to"])
            profile["voice_source"] = "param_fix"
            profile["binding_id"] = bid
            changed = True
        if changed:
            self._save_registry()
        return report

    def _sync_voice_map_version(self) -> None:
        """声线映射表升级时一次性重刷旧档案，之后跨章继续锁定不再漂移。"""
        changed = False
        if self.registry.get("voice_map_version") != self.voice_map_version:
            self.registry["voice_map_version"] = self.voice_map_version
            changed = True
        for profile in self.iter_profiles():
            if self._sync_profile(profile):
                self._store_profile(profile)
                changed = True
        if changed:
            self._save_registry()

    # ------------------------------------------------------------------ 人设
    def _persona_prompt(self, chapter_title: str, samples: dict[str, list[str]], emotions: list[str]) -> str:
        lines = [f"章节：{chapter_title}", "请为以下角色建立配音档案，性别和年龄按原文推断："]
        for name, quotes in samples.items():
            lines.append(f"- {name}：{' / '.join(quotes[:3])}")
        lines.append(
            '只输出 JSON：{"personas":[{"name":"角色名","gender":"男|女|未知",'
            '"gender_evidence":"原文性别依据（代词/称谓），没有就空","gender_confidence":0到1,'
            '"age":整数或null,"age_evidence":"原文年龄依据",'
            '"temperament":"' + "|".join(TEMPERAMENTS) + '",'
            f'"persona":"20字以内的音色与性格","voice_style":"说话风格","default_emotion":"{"|".join(emotions)}",'
            '"dialect":"粤语|台湾腔|东北|陕西|空","region":"籍贯/地域或空",'
            '"dialect_evidence":"原文里支持方言/地域的逐字片段，没有就空","dialect_confidence":0到1}]}'
        )
        lines.append("temperament 只能取给定值；拿不准就填“中性”。")
        lines.append("性别没有原文依据时 gender=未知、gender_confidence=0，不要瞎猜。")
        lines.append("方言/地域没有原文依据时 dialect=空、dialect_evidence=空、dialect_confidence=0，"
                     "不要仅凭名字或语气猜测。")
        return "\n".join(lines)

    PERSONA_SCHEMA = {
        "type": "object",
        "properties": {
            "personas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "gender": {"type": "string", "enum": ["男", "女", "未知"]},
                        "gender_evidence": {"type": "string"},
                        "gender_confidence": {"type": "number"},
                        "age": {},
                        "age_evidence": {"type": "string"},
                        "temperament": {"type": "string"},
                        "persona": {"type": "string"},
                        "voice_style": {"type": "string"},
                        "default_emotion": {"type": "string"},
                        "dialect": {"type": "string"},
                        "region": {"type": "string"},
                        "dialect_evidence": {"type": "string"},
                        "dialect_confidence": {"type": "number"},
                    },
                    "required": ["name"],
                },
            }
        },
        "required": ["personas"],
    }

    def _ask_personas(self, chapter_title: str, samples: dict[str, list[str]]) -> dict[str, dict]:
        emotions = self.cfg["script"]["emotions"]
        try:
            data = self.llm.chat_json([
                {"role": "system", "content": "你是中文有声书的配音导演，只输出 JSON。"},
                {"role": "user", "content": self._persona_prompt(chapter_title, samples, emotions)},
            ], schema=self.PERSONA_SCHEMA, schema_name="personas")
            items = data.get("personas", []) if isinstance(data, dict) else data
            return {str(item.get("name", "")).strip(): item for item in items if isinstance(item, dict)}
        except Exception as err:
            log.warning("Persona 生成失败，使用姓名/称谓推断：%s", err)
            return {}

    def _build_profile(self, name: str, persona: dict, chapter_index: int | None = None,
                       book_id: str | None = None) -> dict:
        guessed = inference.resolve_gender(name, persona, self.cfg)
        gender = guessed["gender"] if guessed["gender"] in ("男", "女") else "未知"
        age_value = _parse_age(persona.get("age"), self.cfg)
        band = age_band(age_value, self.cfg)
        persona_text = f"{persona.get('persona', '')}{persona.get('voice_style', '')}"
        age_note = f"LLM 推断 {age_value} 岁" if persona.get("age") else ""
        confidence = float(inference._value(
            self.cfg, ("casting", "age", "evidence_default_confidence"))) if age_note else None

        with self.lock:
            ident = self._identity(name, book_id, create=True, gender=gender)
            if ident is None:
                raise RuntimeError(f"无法建立身份：{name}")
            existing = ident.get("stage_profiles", {}).get(band)
            if existing:
                # 音色与阶段档案锁定：跨章节/跨书复用，只更新出现章节与声线
                self._update_identity(ident, guessed, persona)
                self._record_dialect_persona(ident, persona, persona_text, chapter_index)
                profile = self._profile_view(ident, band)
                if not profile.get("first_seen_chapter") and chapter_index is not None:
                    profile["first_seen_chapter"] = chapter_index
                profile["last_seen_chapter"] = chapter_index
                profile["edge_voice"] = self._locked_voice(
                    name, gender, band, persona_text, create=True,
                    chapter_index=chapter_index, updated_by="persona", book_id=book_id)
                self._sync_profile(profile, book_id)
                self._store_profile(profile, book_id)
                self._save_registry()
                return profile

            self._update_identity(ident, guessed, persona)
            self._record_dialect_persona(ident, persona, persona_text, chapter_index)
            temperament = str(persona.get("temperament") or "").strip()
            if temperament not in TEMPERAMENTS:
                temperament = infer_temperament(persona_text)
            base = persona_base(temperament, band, self.cfg)
            profile = {
                "key": normalize_key(name, gender, band),
                "name": name,
                "character_id": ident["id"],
                "gender": gender,
                "gender_confidence": guessed["confidence"],
                "gender_evidence": guessed["evidence"],
                "gender_source": guessed["source"],
                "gender_confirmed": guessed["confidence"] >= float(
                    self.cfg["casting"]["gender"]["auto_confirm"]),
                "generic": casting.is_generic(name, self.generic_names),
                "age_band": band,
                "age_stage": band,
                "temperament": temperament,
                "prosody_source": "temperament",
                "age_note": age_note,
                "age_confidence": confidence,
                "first_seen_chapter": chapter_index,
                "last_seen_chapter": chapter_index,
                "updated_by": "persona",
                "checked_chapters": [],
                "persona": str(persona.get("persona") or f"{gender}{band}角色").strip(),
                "voice_style": str(persona.get("voice_style") or "").strip(),
                "default_emotion": str(persona.get("default_emotion") or "平静").strip(),
                "engine": self.cfg["caster"]["dialogue_engine"],
                "rate": base["rate"],
                "pitch": base["pitch"],
                "volume": base["volume"],
                "class_name": "角色",
            }
            profile["edge_voice"] = self._locked_voice(
                name, gender, band, persona_text, create=True,
                chapter_index=chapter_index, age_note=age_note,
                age_confidence=confidence, updated_by="persona", book_id=book_id)
            self._sync_profile(profile, book_id)
            self._store_profile(profile, book_id)
            self._save_registry()
            return profile

    def _update_identity(self, ident: dict, guessed: dict, persona: dict) -> None:
        old_conf = float(ident.get("gender_confidence") or 0.0)
        if guessed.get("gender") in ("男", "女") and (
                guessed.get("confidence", 0.0) >= old_conf or ident.get("gender") == "未知"):
            for field in ("gender", "gender_confidence", "gender_evidence",
                          "gender_source", "gender_confirmed"):
                if field in guessed:
                    ident[field] = guessed[field]
        for field in ("persona", "voice_style", "default_emotion"):
            if persona.get(field) and not ident.get(field):
                ident[field] = str(persona[field]).strip()

    def _latest_profile(self, name: str, exclude_key: str = "") -> dict | None:
        """同名角色里取“最后出现章节最晚、阶段最靠后”的档案。"""
        ident = self._identity(name)
        if not ident:
            return None
        rows = [profile for profile in self.profile_history(name)
                if profile.get("key") != exclude_key]
        if not rows:
            return None
        return max(rows, key=lambda item: (
            item.get("last_seen_chapter") or 0,
            inference.band_index(item.get("age_stage") or self.default_stage, self.cfg)))

    def set_stage(self, name: str, from_stage: str, to_stage: str, edge_voice: str | None = None,
                  rate: str | None = None, pitch: str | None = None, volume: str | None = None,
                  chapter_index: int | None = None,
                  evidence: str = "WebUI 手动指定阶段") -> dict | None:
        """人工确认进入下一阶段：新建该阶段档案与锁，并记录 voice_events。"""
        if from_stage not in self.stage_order or to_stage not in self.stage_order or from_stage == to_stage:
            return None
        with self.lock:
            ident = self._identity(name, create=False)
            if not ident or from_stage not in ident.get("stage_profiles", {}):
                return None
            old = self._profile_view(ident, from_stage)
            chapter = chapter_index if chapter_index is not None else old.get("last_seen_chapter")
            new_profile = dict(old)
            new_profile.update({
                "key": normalize_key(name, old.get("gender", "未知"), to_stage),
                "age_band": to_stage,
                "age_stage": to_stage,
                "first_seen_chapter": old.get("first_seen_chapter") or chapter,
                "last_seen_chapter": chapter,
                "updated_by": "manual",
                "updated_by_chapter": chapter,
                "age_confidence": 1.0,
                "age_note": evidence,
            })
            temperament = new_profile.get("temperament") or infer_temperament(
                f"{new_profile.get('persona', '')}{new_profile.get('voice_style', '')}")
            new_profile["temperament"] = temperament
            if rate or pitch or volume:
                if rate:
                    new_profile["rate"] = rate
                if pitch:
                    new_profile["pitch"] = pitch
                if volume:
                    new_profile["volume"] = volume
                new_profile["prosody_source"] = "manual"
            elif old.get("prosody_source") != "manual":
                base = persona_base(temperament, to_stage, self.cfg)
                new_profile.update({"rate": base["rate"], "pitch": base["pitch"],
                                    "volume": base["volume"], "prosody_source": "temperament"})
            voice = edge_voice or self._locked_voice(
                name, new_profile.get("gender", "未知"), to_stage,
                f"{new_profile.get('persona', '')}{new_profile.get('voice_style', '')}",
                create=True, chapter_index=chapter, age_note=evidence,
                age_confidence=1.0, updated_by="manual")
            if edge_voice:
                self._write_binding(ident["id"], to_stage, edge_voice, source="manual",
                                    updated_by="manual")
            new_profile["edge_voice"] = voice
            self._sync_profile(new_profile, None)
            self._store_profile(new_profile)
            if chapter is not None:
                branch = ident["stage_profiles"].get(from_stage)
                if branch:
                    branch["last_seen_chapter"] = max(0, chapter - 1)
            self._append_event(name, old.get("edge_voice"), voice, from_stage, to_stage,
                              chapter, evidence, 1.0, "manual", ident)
            self._save_registry()
            return new_profile

    def set_voice(self, name: str, stage: str, edge_voice: str | None = None,
                  rate: str | None = None, pitch: str | None = None, volume: str | None = None,
                  temperament: str | None = None,
                  updated_by: str = "manual") -> dict | None:
        """改声线/韵律：同时更新阶段档案与角色阶段锁，避免下次运行被旧锁覆盖。"""
        with self.lock:
            ident = self._identity(name, create=False)
            if not ident or stage not in ident.get("stage_profiles", {}):
                return None
            branch = ident["stage_profiles"][stage]
            profile = self._profile_view(ident, stage)
            old_voice = profile.get("edge_voice")
            if temperament in TEMPERAMENTS and temperament != profile.get("temperament"):
                branch["temperament"] = temperament
                base = persona_base(temperament, stage, self.cfg)
                branch.update({"rate": base["rate"], "pitch": base["pitch"],
                               "volume": base["volume"], "prosody_source": "temperament"})
            elif rate or pitch or volume:
                if rate:
                    branch["rate"] = rate
                if pitch:
                    branch["pitch"] = pitch
                if volume:
                    branch["volume"] = volume
                branch["prosody_source"] = "manual"
            branch["updated_by"] = updated_by
            if edge_voice:
                self._write_binding(ident["id"], stage, edge_voice,
                                    source=self._source_for(updated_by),
                                    updated_by=updated_by)
            profile = self._profile_view(ident, stage)
            self._sync_profile(profile)
            self._store_profile(profile)
            if edge_voice and old_voice and old_voice != edge_voice:
                self._append_event(name, old_voice, edge_voice, stage, stage,
                                  profile.get("last_seen_chapter") or 0,
                                  "WebUI 手动调整声线", 1.0, updated_by, ident)
            self._save_registry()
            return profile

    def narrator_profile(self, book_id: str | None = None) -> dict:
        engine = self.cfg["caster"]["narration_engine"]
        override: dict = {}
        registry_narrator = self.registry.get("narrator") or {}
        cfg_narrator = self.cfg["casting"]["narrator"]
        global_cfg = registry_narrator.get("global") or {}
        # 迁移自 v1 的旧值（source=config）不覆盖 config.yaml，避免 v4 的 -6% 被旧 -4% 压回
        if global_cfg.get("source") != "config" or global_cfg.get("updated_by"):
            override.update(global_cfg)
        if book_id:
            override.update((registry_narrator.get("books") or {}).get(book_id) or {})
        voice = override.get("voice") or self.narrator_voice
        resolved = voices_catalog.resolve_voice(voice, self.catalog)
        rate = str(override.get("rate") or self.cfg["caster"]["narrator_rate"])
        pitch = str(override.get("pitch") or self.cfg["caster"]["narrator_pitch"])
        volume = str(override.get("volume") or self.cfg["caster"]["narrator_volume"])
        stage = str(cfg_narrator["age_stage"])
        gender = {"female": "女", "male": "男"}.get(
            str(cfg_narrator.get("gender_preference", "auto")), "未知")
        return {
            "key": "旁白",
            "name": "旁白",
            "gender": gender,
            "display_label": f"{gender if gender != '未知' else ''}{stage}",
            "age_band": stage,
            "age_stage": stage,
            "temperament": "中性",
            "prosody_source": "fixed",
            "persona": f"自然清晰的书播{gender if gender != '未知' else ''}声",
            "voice_style": "叙述感、克制",
            "default_emotion": self.cfg["caster"].get("narrator_emotion", "平静"),
            "engine": engine,
            "edge_voice": resolved["voice"],
            "legacy_edge_voice": voice if resolved["fallback"] else "",
            "voice_repair": resolved["note"],
            "voice_source": "manual" if override.get("voice") else "config",
            "book_id": book_id,
            "voice": 0,
            "sid": 0,
            "speed": 1.0,
            "rate": rate,
            "pitch": pitch,
            "volume": volume,
            "ref_wav": None,
            "ref_text": None,
            "class_name": "旁白",
        }

    def set_narrator(self, book_id: str | None, voice: str | None = None,
                     rate: str | None = None, pitch: str | None = None,
                     volume: str | None = None, scope: str = "book") -> dict:
        """旁白覆盖：scope=book 只影响本书，scope=global 影响全部书，scope=config 清除覆盖。"""
        with self.lock:
            narrator_cfg = self.registry.setdefault("narrator", {})
            target: dict = {}
            if scope == "global":
                target = narrator_cfg.setdefault("global", {})
            elif scope == "book":
                if not book_id:
                    raise ValueError("缺少 book_id")
                target = narrator_cfg.setdefault("books", {}).setdefault(book_id, {})
            if scope == "config":
                if book_id:
                    narrator_cfg.get("books", {}).pop(book_id, None)
                else:
                    narrator_cfg.pop("global", None)
            else:
                if voice:
                    resolved = voices_catalog.resolve_voice(voice, self.catalog)
                    if resolved["fallback"]:
                        raise ValueError(f"{voice} 不可用：{resolved['note']}")
                    target["voice"] = voice
                for key, value in (("rate", rate), ("pitch", pitch), ("volume", volume)):
                    if value:
                        target[key] = value
                target["updated_at"] = _now()
                target["updated_by"] = "webui"
            self._save_registry()
            return self.narrator_profile(book_id)

    def book_options(self, book_id: str | None) -> dict:
        if not book_id:
            return {}
        return dict((self.registry.get("book_settings") or {}).get(book_id) or {})

    def set_book_option(self, book_id: str, key: str, value) -> dict:
        with self.lock:
            options = self.registry.setdefault("book_settings", {}).setdefault(book_id, {})
            options[key] = value
            options["updated_at"] = _now()
            self._save_registry()
            return dict(options)

    def check_collisions(self, cast: dict, narrator_voice: str = "") -> dict:
        """撞声检查：>=2.0 阻断自动推荐，1.0~2.0 警告，群像可豁免。"""
        narrator_voice = narrator_voice or self.narrator_profile().get("edge_voice", "")
        return consistency.collision_report(cast, narrator_voice, self.cfg)

    def check_drift(self) -> list[dict]:
        """配置推荐值 vs 已生效绑定：manual 永不漂移，只提示 migrated/auto。"""
        identities = self.registry.get("identities") or {}

        def recommend(binding: dict) -> dict | None:
            ident = identities.get(binding.get("character_id") or "") or {}
            stage = binding.get("age_stage") or self.default_stage
            branch = (ident.get("stage_profiles") or {}).get(stage) or {}
            gender = ident.get("gender") or "未知"
            base = persona_base(branch.get("temperament") or "中性", stage, self.cfg)
            return {"voice": self._voice_for_stage(gender, stage), **base}

        return consistency.drift_report(self.registry, recommend, self.cfg)

    def upsert_binding(self, binding: dict) -> dict:
        """写入一条覆盖绑定（单书例外/手动指定）；下次解析立即生效。"""
        with self.lock:
            bid = binding.get("id") or ""
            if not bid:
                raise ValueError("缺少绑定 id")
            bindings = self._bindings()
            existing = bindings.get(bid) or {}
            binding["created_at"] = existing.get("created_at") or _now()
            binding["updated_at"] = _now()
            bindings[bid] = binding
            self._save_registry()
            return binding

    def remove_binding(self, binding_id: str) -> bool:
        with self.lock:
            if binding_id in self._bindings():
                self._bindings().pop(binding_id, None)
                self._save_registry()
                return True
            return False

    # ------------------------------------------------------------------ 选角入口
    def build(self, chapter_title: str, segments: list, book_id: str,
              chapter_index: int | None = None) -> dict:
        """返回 {"speakers": {原文名: key}, "cast": {key: profile}, "narrator": profile}。"""
        self.sync_voice_map_version()
        raw_names = [seg.speaker for seg in segments
                     if seg.type == "dialogue" and seg.speaker not in ("旁白", "未知角色")]
        aliases = canonicalize_names(list(dict.fromkeys(raw_names)))
        samples: dict[str, list[str]] = {}
        for seg in segments:
            if seg.type != "dialogue" or seg.speaker in ("旁白", "未知角色"):
                continue
            name = aliases.get(seg.speaker, seg.speaker)
            samples.setdefault(name, [])
            if len(samples[name]) < 3:
                samples[name].append(seg.text[:60])

        line_counts: dict[str, int] = {}
        for seg in segments:
            if seg.type == "dialogue" and seg.speaker not in ("旁白", "未知角色"):
                name = aliases.get(seg.speaker, seg.speaker)
                line_counts[name] = line_counts.get(name, 0) + 1

        personas = self._ask_personas(chapter_title, samples) if samples else {}
        speakers: dict[str, str] = {}
        cast: dict[str, dict] = {}
        self._active_cids = set()
        for name in samples:
            profile = self._build_profile(name, personas.get(name, {}), chapter_index, book_id)
            profile["line_count"] = line_counts.get(name, 0)
            profile["generic"] = casting.is_generic(name, self.generic_names)
            self._store_profile(profile, book_id)
            cast[profile["key"]] = profile
        self._recompute_levels()
        if cast:
            self._save_registry()
        self._active_cids = None
        key_by_name = {profile["name"]: profile["key"] for profile in cast.values()}
        for raw_name in raw_names:  # 原始称谓与归并名都指向同一 key，路由才能命中
            speakers[raw_name] = key_by_name[aliases.get(raw_name, raw_name)]

        result = {"speakers": speakers, "cast": cast,
                  "narrator": self.narrator_profile(book_id), "book_id": book_id}
        with self.lock:
            self.apply_voice_policy(result, book_id)
        book_path = (self.root / self.cfg["paths"]["cache"] / "voices" / f"{book_id}.json").resolve()
        book_path.parent.mkdir(parents=True, exist_ok=True)
        book_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def ensure_registered(self, cast_data: dict) -> None:
        """缓存命中时补登全局注册表，保证换书仍能复用同一音色。"""
        self.sync_voice_map_version()
        with self.lock:
            changed = False
            for profile in cast_data.get("cast", {}).values():
                ident = self._identity(profile.get("name", ""), create=False)
                stage = profile.get("age_stage") or profile.get("age_band") or self.default_stage
                if ident is None or stage not in ident.get("stage_profiles", {}):
                    profile = dict(profile)
                    profile.setdefault("age_stage", stage)
                    profile.setdefault("age_note", "")
                    profile.setdefault("age_confidence", None)
                    profile.setdefault("first_seen_chapter", None)
                    profile.setdefault("last_seen_chapter", None)
                    profile.setdefault("updated_by", "cache")
                    profile.setdefault("checked_chapters", [])
                    self._store_profile(profile)
                    changed = True
                    if profile.get("edge_voice"):
                        ident = self._identity(profile.get("name", ""), create=True)
                        if ident and self._lock_id(ident["id"], stage) not in self._bindings():
                            self._write_binding(ident["id"], stage, profile["edge_voice"],
                                                source="auto", updated_by="cache")
            if changed:
                self._save_registry()

    def refresh_cast(self, cast_data: dict, chapter_index: int | None = None,
                     book_id: str | None = None) -> dict:
        """缓存命中时用当前配置覆盖剧本快照里的旧声线，防止换引擎/换映射后漂移。"""
        self.sync_voice_map_version()
        with self.lock:
            changed = False
            key_map: dict[str, str] = {}
            # 声线“独立/复用”按本章同时出现的角色计算，跨章不复用判断
            # 先一次性收齐在场角色，避免按 dict 顺序边处理边扩张导致级别随顺序漂移
            self._active_cids = set()
            for profile in cast_data.get("cast", {}).values():
                ident = self._identity(profile.get("name", ""), book_id, create=False)
                if ident:
                    self._active_cids.add(ident["id"])
            for key, profile in list(cast_data.get("cast", {}).items()):
                name = profile.get("name", "")
                ident = self._identity(name, book_id, create=False)
                if ident:
                    self._active_cids.add(ident["id"])
                stage = profile.get("age_stage") or profile.get("age_band") or self.default_stage
                branch = (ident or {}).get("stage_profiles", {}).get(stage)
                if not ident or branch is None:
                    latest = self._latest_profile(name)
                    if latest:
                        key_map[key] = latest["key"]
                        continue
                    copy = dict(profile)
                    copy.setdefault("age_stage", stage)
                    copy.setdefault("updated_by", "cache")
                    copy.setdefault("checked_chapters", [])
                    self._store_profile(copy, book_id)
                    ident = self._identity(name, book_id, create=True)
                    changed = True
                    branch = (ident or {}).get("stage_profiles", {}).get(stage) if ident else None
                entry = self._profile_view(ident, stage)
                if chapter_index is not None:
                    entry["last_seen_chapter"] = chapter_index
                    if not entry.get("first_seen_chapter"):
                        entry["first_seen_chapter"] = chapter_index
                    changed = True
                if entry.get("edge_voice"):
                    self._locked_voice(entry.get("name", ""), entry.get("gender", "未知"),
                                       stage, create=True, chapter_index=chapter_index,
                                       updated_by="cache", book_id=book_id)
                self._sync_profile(entry, book_id)
                self._store_profile(entry, book_id)
                key_map[key] = entry["key"]
            self._recompute_levels()
            self._active_cids = None
            for profile in self.iter_profiles():
                if self._sync_profile(profile, book_id):
                    self._store_profile(profile, book_id)
                    changed = True
            if changed:
                self._save_registry()
            cast: dict[str, dict] = {}
            for key, profile in cast_data.get("cast", {}).items():
                target = key_map.get(key, key)
                latest = self.profile_by_key(target)
                cast[target] = latest or dict(profile)
            cast_data["cast"] = cast
            cast_data["speakers"] = {raw: key_map.get(key, key)
                                     for raw, key in cast_data.get("speakers", {}).items()}
            cast_data["narrator"] = self.narrator_profile(book_id)
            self.apply_voice_policy(cast_data, book_id)
        return cast_data

    # ------------------------------------------------------------------ 年龄推进
    def _age_prompt(self, chapter_title: str, text: str, names: list[str]) -> str:
        return "\n".join([
            f"章节：{chapter_title}",
            f"本章出场角色：{'、'.join(names)}",
            "只从下章原文里找“明确的年龄或时间推进证据”（如“十八岁”“三年后”“登基十年”），"
            "没有证据的角色不要输出。只输出 JSON：",
            '{"ages":[{"name":"角色名","age":整数或null,"stage":"'
            + "|".join(self.stage_order) + '或null",'
            '"evidence":"原文依据，20字以内","confidence":0到1}]}',
            "原文：",
            text[:4000],
        ])

    def extract_ages(self, chapter_title: str, text: str, cast_data: dict,
                     chapter_index: int | None = None) -> dict[str, dict] | None:
        """轻量年龄抽取：只查本章出场角色，已核对过本章的直接跳过。

        返回 None 表示这次抽取失败（不要据此写 checked_chapters，下次还能重试）。
        """
        if self.llm is None or chapter_index is None or not AGE_KEYWORDS.search(text):
            return {}
        pending = []
        for profile in cast_data.get("cast", {}).values():
            checked = profile.get("checked_chapters") or []
            if chapter_index not in checked:
                pending.append(profile.get("name", ""))
        names = [name for name in dict.fromkeys(pending) if name]
        if not names:
            return {}
        try:
            data = self.llm.chat_json(
                [{"role": "system", "content": "你是中文小说的时间线标注员，只输出 JSON。"},
                 {"role": "user", "content": self._age_prompt(chapter_title, text, names)}],
                prefer=self.cfg["llm"].get("review_model"),
                schema=self.age_schema, schema_name="ages",
            )
            items = data.get("ages", []) if isinstance(data, dict) else data
        except Exception as err:
            log.warning("[%s] 年龄抽取失败，沿用上一阶段：%s", chapter_title, err)
            return None
        result: dict[str, dict] = {}
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or name not in names:
                continue
            age = item.get("age")
            stage = item.get("stage")
            if stage not in self.stage_order:
                stage = age_band(int(age), self.cfg) if isinstance(age, (int, float)) else None
            fallback_conf = float(inference._value(
                self.cfg, ("casting", "age", "evidence_default_confidence")))
            try:
                confidence = float(item.get("confidence", fallback_conf))
            except (TypeError, ValueError):
                confidence = fallback_conf
            result[name] = {
                "name": name,
                "age": int(age) if isinstance(age, (int, float)) else None,
                "stage": stage,
                "evidence": str(item.get("evidence", "")).strip()[:40],
                "confidence": max(0.0, min(1.0, confidence)),
            }
        return result

    def _apply_transition(self, profile: dict, chapter_index: int) -> None:
        """推进“同声线跨阶段”的渐变，结束后恢复目标阶段基础韵律。"""
        blend = inference.blend_transition(profile, chapter_index, self.cfg)
        if blend is None:
            return
        tr = profile.get("transition") or {}
        if not blend:
            base = persona_base(profile.get("temperament", "中性"),
                                tr.get("to_stage") or profile.get("age_stage") or self.default_stage, self.cfg)
            profile.update(base)
            profile["transition"] = None
            return
        for key in ("rate", "pitch", "volume"):
            profile[key] = prosody.format_value(blend[key], "Hz" if key == "pitch" else "%")

    def _append_event(self, name: str, from_voice: str | None, to_voice: str | None,
                      from_stage: str, to_stage: str, chapter, evidence: str,
                      confidence: float, source: str, ident: dict | None = None) -> None:
        events = self.registry.setdefault("voice_events", [])
        ident = ident or self._identity(name)
        events.append({
            "id": f"evt:{_now()}:{len(events)}",
            "book_id": None,
            "chapter": chapter,
            "name": name,
            "character_id": (ident or {}).get("id") or casting.identity_id(name),
            "from_stage": from_stage,
            "to_stage": to_stage,
            "from_voice": from_voice,
            "to_voice": to_voice,
            "reason": evidence,
            "evidence": evidence,
            "confidence": confidence,
            "source": source,
            "status": "applied",
            "created_at": _now(),
        })

    def apply_age_evidence(self, cast_data: dict, evidence: dict[str, dict],
                           chapter_index: int | None = None) -> dict:
        """把年龄证据写进档案；只有证据足够明确且越过缓冲线才升阶段换声线。"""
        if chapter_index is None:
            return cast_data
        with self.lock:
            cast = cast_data.get("cast", {})
            changed = False
            for key, profile in list(cast.items()):
                name = profile.get("name", "")
                info = evidence.get(name)
                self._apply_transition(profile, chapter_index)
                profile.setdefault("checked_chapters", [])
                if chapter_index not in profile["checked_chapters"]:
                    profile["checked_chapters"].append(chapter_index)
                profile["last_seen_chapter"] = chapter_index
                if info:
                    profile["age_note"] = info["evidence"] or profile.get("age_note", "")
                    profile["age_confidence"] = info["confidence"]
                    profile["updated_by"] = "llm"
                    profile["updated_by_chapter"] = chapter_index
                changed = True
                self._store_profile(profile, cast_data.get("book_id"))
                if not info or not info.get("stage"):
                    continue
                current = profile.get("age_stage") or profile.get("age_band") or self.default_stage
                decision = inference.decide_age(profile, info, chapter_index, self.cfg)
                profile["age_evidence_log"] = inference.log_evidence(
                    profile, info, chapter_index, self.cfg)
                if decision["action"] != "advance":
                    profile["age_hold_reason"] = decision["reason"]
                    self._store_profile(profile, cast_data.get("book_id"))
                    continue
                target = decision["target"]
                old_key = key
                new_key = normalize_key(name, profile.get("gender", "未知"), target)
                old_voice = profile.get("edge_voice")
                ident = self._identity(name, cast_data.get("book_id"), create=True)
                if ident and target in ident.get("stage_profiles", {}):
                    new_profile = self._profile_view(ident, target)
                    new_profile["age_note"] = info["evidence"]
                    new_profile["age_confidence"] = info["confidence"]
                    new_profile["first_seen_chapter"] = (
                        new_profile.get("first_seen_chapter") or chapter_index)
                else:
                    new_profile = dict(profile)
                    new_profile.update({
                        "key": new_key,
                        "age_band": target,
                        "age_stage": target,
                        "age_note": info["evidence"],
                        "age_confidence": info["confidence"],
                        "first_seen_chapter": chapter_index,
                        "last_seen_chapter": chapter_index,
                        "checked_chapters": list(profile.get("checked_chapters", [])),
                    })
                new_profile["updated_by"] = "llm"
                new_profile["updated_by_chapter"] = chapter_index
                new_voice = self._locked_voice(
                    name, new_profile.get("gender", "未知"), target,
                    f"{new_profile.get('persona', '')}{new_profile.get('voice_style', '')}",
                    create=True, chapter_index=chapter_index,
                    age_note=info["evidence"], age_confidence=info["confidence"],
                    updated_by="llm", book_id=cast_data.get("book_id"))
                new_profile["edge_voice"] = new_voice
                new_profile["stage_entered_chapter"] = chapter_index
                new_profile["age_evidence_log"] = list(profile.get("age_evidence_log", []))
                base = persona_base(new_profile.get("temperament", "中性"), target, self.cfg)
                transition_cfg = ((self.cfg.get("casting") or {}).get("age") or {}).get("transition") or {}
                chapters = int(transition_cfg.get("chapters") or 0)
                if new_voice == old_voice and decision.get("transition") and chapters > 0:
                    # 同一声线跨阶段：按配置章数渐变，避免听感突变
                    new_profile["transition"] = {
                        "from_stage": current,
                        "to_stage": target,
                        "start_chapter": chapter_index,
                        "chapters": chapters,
                        "from": {k: prosody.parse_value(profile.get(k, base[k]),
                                                        "Hz" if k == "pitch" else "%")
                                 for k in ("rate", "pitch", "volume")},
                        "to": {k: prosody.parse_value(base[k], "Hz" if k == "pitch" else "%")
                               for k in ("rate", "pitch", "volume")},
                    }
                    self._apply_transition(new_profile, chapter_index)
                else:
                    new_profile["transition"] = None
                    new_profile.update(base)
                old_branch = (ident or {}).get("stage_profiles", {}).get(current)
                if old_branch:
                    old_branch["last_seen_chapter"] = chapter_index - 1
                self._sync_profile(new_profile, cast_data.get("book_id"))
                self._store_profile(new_profile, cast_data.get("book_id"))
                cast.pop(old_key, None)
                cast[new_key] = new_profile
                for raw, speaker_key in cast_data.get("speakers", {}).items():
                    if speaker_key == old_key:
                        cast_data["speakers"][raw] = new_key
                self._append_event(name, old_voice, new_voice, current, target,
                                   chapter_index, info["evidence"], info["confidence"],
                                   "llm", ident)
                log.info("[第%s章] %s 年龄阶段 %s -> %s，声线 %s -> %s（依据：%s）",
                         chapter_index, name, current, target, old_voice, new_voice,
                         info["evidence"])
            if changed:
                self._save_registry()
            cast_data["cast"] = cast
        return cast_data

    def update_ages(self, chapter, segments: list, cast_data: dict) -> dict:
        """章节入口：抽取年龄证据并应用；失败时保持原阶段。"""
        real_age = bool(self.book_options(getattr(chapter, "book_id", None)).get("real_age"))
        for profile in cast_data.get("cast", {}).values():
            profile["book_real_age"] = real_age
        evidence = self.extract_ages(chapter.title, chapter.text, cast_data, chapter.index)
        if evidence is None:
            return cast_data
        return self.apply_age_evidence(cast_data, evidence, chapter.index)

    def show_cast_text(self) -> str:
        """`--show-cast` 的输出：角色/阶段/声线/锁定章节/依据/变更历史。"""
        entries = self.iter_profiles()

        def stage_index(entry: dict) -> int:
            stage = entry.get("age_stage") or entry.get("age_band") or self.default_stage
            return inference.band_index(stage, self.cfg)

        current: dict[str, dict] = {}
        for entry in entries:
            name = entry.get("name", "")
            rank = (entry.get("last_seen_chapter") or 0, stage_index(entry))
            if name not in current or rank >= (
                    current[name].get("last_seen_chapter") or 0, stage_index(current[name])):
                current[name] = entry
        rows = list(current.values())
        rows.sort(key=lambda e: (e.get("name", ""), stage_index(e)))
        header = (f"{'角色':<8}{'阶段':<6}{'性格':<6}{'rate':<7}{'pitch':<8}{'volume':<8}"
                  f"{'声线':<28}{'首见章':<8}{'最后章':<8}{'置信度':<8}{'来源':<8}依据")
        lines = ["=== 当前角色表 ===", header, "-" * 116]
        narrator = self.narrator_profile()
        lines.append(f"{narrator['name']:<8}{narrator['display_label']:<6}{'中性':<6}{narrator['rate']:<7}{narrator['pitch']:<8}{narrator['volume']:<8}"
                     f"{self.narrator_voice:<28}{'—':<8}{'—':<8}{'—':<8}{'config':<8}固定女声旁白")
        for entry in rows:
            stage = entry.get("age_stage") or entry.get("age_band") or self.default_stage
            conf = entry.get("age_confidence")
            conf_text = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
            note = entry.get("age_note") or "—"
            lines.append(
                f"{entry.get('name', ''):<8}{stage:<6}{entry.get('temperament', '中性'):<6}"
                f"{entry.get('rate', '+0%'):<7}{entry.get('pitch', '+0Hz'):<8}"
                f"{entry.get('volume', '+0%'):<8}{entry.get('edge_voice', ''):<28}"
                f"{str(entry.get('first_seen_chapter') or '—'):<8}"
                f"{str(entry.get('last_seen_chapter') or '—'):<8}"
                f"{conf_text:<8}{entry.get('updated_by', '—'):<8}{note}")
        changes = self.voice_events()
        lines += ["", f"=== 换音色历史（{len(changes)} 条）==="]
        if not changes:
            lines.append("（暂无。只有年龄证据越过缓冲线才会升阶段换声线。）")
        for item in changes:
            lines.append(
                f"第{item.get('chapter')}章  {item.get('character_id') or item.get('name')}  "
                f"{item.get('from_stage')}→{item.get('to_stage')}  "
                f"{item.get('from_voice')} → {item.get('to_voice')}  "
                f"依据：{item.get('evidence')}（置信度 {item.get('confidence')}）")
        bindings = [item for item in self._bindings().values()
                    if item.get("scope") == "character_stage" and item.get("locked")]
        lines += ["", f"=== 音色锁（角色阶段）{len(bindings)} 条 ==="]
        for item in sorted(bindings, key=lambda b: str(b.get("id"))):
            lines.append(
                f"{item.get('id', ''):<34} {item.get('voice', ''):<28} "
                f"来源 {item.get('source', '—'):<9} {item.get('updated_by', '—')}")
        return "\n".join(lines)
