"""LLM 声线规划：档案推断 + 整书/整章规划 + 确定性校验 + 迟滞变更预算。

设计原则（与旧“声线事件抽取”不同）：
- LLM 只产出**建议**：角色档案、声线分配、退场判定；
- 代码负责**强制规范**：schema 枚举、catalog 白名单、池容量、撞声、参数区间、
  变更预算、结构性事件门槛、主角保护、dormant/lent 冲突、失败回落；
- 任何 LLM 输出都不能直接写绑定；只有 `apply_plan()` 在 `enabled=true` 时才落库，
  且只落“校验通过 + 差异足够大 + 迟滞满足”的变更，其余进 `pending_review`。

默认 `enabled: false`，关闭时所有函数只读、不产生副作用。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time

TIERS = ("main", "support", "minor", "generic")
GENDERS = ("男", "女", "未知")
AGE_BANDS = ("少年", "青年", "中年", "老年", "未知")
RETIRE_STATUS = ("retired", "hidden", "likely_return")
# `present` 仅用于“本章被提及/在场”的辅助输出，不参与 dormant 判定
RETIRE_STATUS_ALL = RETIRE_STATUS + ("present",)

STRUCTURAL_REASONS = (
    "age_band_cross", "gender_correction", "identity_reveal",
    "disguise_lift", "exit_return", "post_rebind_adjust", "user_override",
)
PARAM_REASONS = ("param_audible_change", "collision_repair", "pool_rebalance")
REASON_CODES = STRUCTURAL_REASONS + PARAM_REASONS

DEFAULT_CONFIG = {
    "enabled": False,
    "profile": {"enabled": False, "min_confidence": 0.60,
                "auto_apply_confidence": 0.80, "max_chars": 6000},
    "plan": {"enabled": False, "min_confidence": 0.65,
             "auto_apply_confidence": 0.85, "max_assignments": 60},
    "retirement": {"enabled": False, "min_confidence": 0.70,
                   "low_confidence_status": "hidden",
                   "suppress_dormant": ["hidden", "likely_return"]},
    "hysteresis": {
        "enabled": True,
        "param_threshold": 1.25,        # 收紧后：≈ 撞声下限的 2.5 倍
        "release_band": 0.75,           # 迟滞释放带
        "consecutive_chapters": 2,      # 参数变更需连续 2 章超标
        "window_chapters": 20,
        "max_changes_per_character": 2,
        "min_chapters_between_changes": 3,
        "main_param_max_pct": 8,
        "main_param_max_hz": 4,
        "main_param_max_volume": 5,
        "main_extra_approval": True,
        "max_cumulative_rate_pct": 12,
        "max_cumulative_pitch_hz": 6,
        "max_cumulative_volume_pct": 8,
    },
    "retry_with_errors": 1,
    "fallback_to_caster": True,
    "cache": True,
}

# 撞声下限（与 casting.consistency.differentiate 一致）
COLLISION_BASE = {"rate": 4.0, "pitch": 4.0, "volume": 6.0}
COLLISION_MIN_DISTANCE = 1.4
# 参数差异归一化：用 2 倍撞声下限做分母，差异分 1.0 = 单维刚好 2 倍下限
PARAM_SCALE = {"rate": 8.0, "pitch": 8.0, "volume": 12.0}


# ---------------------------------------------------------------- 配置 / 目录

def config(cfg: dict) -> dict:
    raw = ((cfg.get("casting") or {}).get("voice_rules") or {}).get("voice_plan") or {}
    merged = dict(DEFAULT_CONFIG)
    for key, default in DEFAULT_CONFIG.items():
        if isinstance(default, dict):
            merged[key] = {**default, **(raw.get(key) or {})}
        else:
            merged[key] = raw.get(key, default)
    return merged


def enabled(cfg: dict) -> bool:
    return bool(config(cfg)["enabled"])


def load_catalog(cfg: dict) -> dict:
    path = ((cfg.get("casting") or {}).get("catalog") or {}).get("path") or "voices/catalog.edge.json"
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / path).read_text(encoding="utf-8"))
    rows = {}
    for entry in data.get("voices") or []:
        if entry.get("available") is False:
            continue
        rows[str(entry.get("id"))] = entry
    return rows


def voice_ids(cfg: dict) -> list[str]:
    return sorted(load_catalog(cfg).keys())


def voice_fits(entry: dict, gender: str, age_band: str) -> bool:
    if not entry:
        return False
    if entry.get("gender") not in (gender, "未知", None) and gender != "未知":
        return False
    stages = entry.get("stages") or []
    return (not stages) or age_band in stages or age_band == "未知"


# ---------------------------------------------------------------- JSON Schema

PROFILE_SCHEMA = {
    "type": "object",
    "required": ["profiles"],
    "properties": {
        "profiles": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["character_id", "gender", "age_band", "aliases",
                             "speech_style", "confidence", "evidence_quote",
                             "evidence_chapter"],
                "properties": {
                    "character_id": {"type": "string"},
                    "gender": {"enum": list(GENDERS)},
                    "age_band": {"enum": list(AGE_BANDS)},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "speech_style": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence_quote": {"type": "string"},
                    "evidence_chapter": {"type": "integer"},
                },
            },
        }
    },
}


def plan_schema(cfg: dict) -> dict:
    """生产用规划 schema：物理上没有 voice_id，LLM 只能改参数。"""
    return {
        "type": "object",
        "required": ["assignments"],
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["character_id", "params", "reason_code",
                                 "confidence", "evidence_quote"],
                    "properties": {
                        "character_id": {"type": "string"},
                        "params": {
                            "type": "object",
                            "required": ["rate_pct", "pitch_hz", "volume_pct"],
                            "properties": {
                                "rate_pct": {"type": "number", "minimum": -16,
                                             "maximum": 25},
                                "pitch_hz": {"type": "number", "minimum": -8,
                                             "maximum": 8},
                                "volume_pct": {"type": "number", "minimum": -10,
                                               "maximum": 10},
                            },
                        },
                        "reason_code": {"enum": list(REASON_CODES)},
                        "confidence": {"type": "number"},
                        "evidence_quote": {"type": "string"},
                    },
                },
            },
            "kept_character_ids": {"type": "array", "items": {"type": "string"}},
        },
    }


RETIREMENT_SCHEMA = {
    "type": "object",
    "required": ["judgments"],
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["character_id", "status", "confidence",
                             "evidence_quote"],
                "properties": {
                    "character_id": {"type": "string"},
                    "status": {"enum": list(RETIRE_STATUS_ALL)},
                    "confidence": {"type": "number"},
                    "evidence_quote": {"type": "string"},
                    "expected_return_hint": {"type": "string"},
                },
            },
        }
    },
}


# ---------------------------------------------------------------- 基础工具

def _norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _evidence_ok(quote: str, body: str) -> bool:
    return bool(quote) and _norm(quote) in _norm(body)


def _hash(payload: dict) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                        .encode("utf-8")).hexdigest()[:16]


def compact_text(text: str, limit: int = 160) -> str:
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    return body if len(body) <= limit else body[:limit] + "…"


def chapter_body(record: dict, max_chars: int) -> str:
    parts = []
    used = 0
    for seg in record.get("segments") or []:
        text = str(seg.get("text") or "")
        if not text:
            continue
        if used + len(text) > max_chars:
            break
        parts.append((f"[{seg.get('speaker')}] " if seg.get("type") == "dialogue" else "")
                     + text)
        used += len(text)
    return "\n".join(parts)


def identity_index(registry: dict, book_id: str) -> dict:
    rows = {}
    for cid, ident in (registry.get("identities") or {}).items():
        books = ident.get("books") or []
        if books and book_id not in books:
            continue
        rows[cid] = ident
    return rows


def name_index(identities: dict) -> dict:
    index = {}
    for cid, ident in identities.items():
        index[cid] = cid
        if ident.get("canonical_name"):
            index[str(ident["canonical_name"])] = cid
        for alias in ident.get("aliases") or []:
            index[str(alias)] = cid
    return index


def current_binding(registry: dict, cid: str) -> tuple[str | None, dict]:
    best = None
    for bid, binding in (registry.get("voice_bindings") or {}).items():
        if binding.get("character_id") != cid:
            continue
        order = int(binding.get("updated_at_chapter") or 0)
        if best is None or order >= best[0]:
            best = (order, bid, binding)
    return (best[1], best[2]) if best else (None, {})


def parse_pct(value, suffix: str = "") -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "0").replace("%", "").replace("Hz", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def binding_params(binding: dict) -> dict:
    return {"rate": parse_pct(binding.get("rate")),
            "pitch": parse_pct(binding.get("pitch")),
            "volume": parse_pct(binding.get("volume"))}


def params_of(assignment: dict) -> dict:
    return {"rate": float(assignment.get("rate_pct") or 0),
            "pitch": float(assignment.get("pitch_hz") or 0),
            "volume": float(assignment.get("volume_pct") or 0)}


# ---------------------------------------------------------------- 差异分 / 迟滞

def param_diff_score(current: dict, proposed: dict) -> float:
    """归一化差异分：1.0 ≈ 单维达到撞声下限的 2 倍。"""
    total = 0.0
    for key, scale in PARAM_SCALE.items():
        delta = float(proposed.get(key) or 0) - float(current.get(key) or 0)
        total += (delta / scale) ** 2
    return round(math.sqrt(total), 3)


def collision_distance(a: dict, b: dict) -> float:
    total = 0.0
    for key, floor in COLLISION_BASE.items():
        total += ((float(a.get(key) or 0) - float(b.get(key) or 0)) / floor) ** 2
    return round(math.sqrt(total), 3)


def has_voice_id(payload) -> bool:
    """递归检查 LLM 输出里是否出现 voice_id / voice 字段（物理禁止）。"""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in ("voice_id", "voice", "voiceid"):
                return True
            if has_voice_id(value):
                return True
    elif isinstance(payload, list):
        return any(has_voice_id(item) for item in payload)
    return False


def change_decision(cid: str, current: dict, assignment: dict, history: list[dict],
                    cfg: dict) -> dict:
    """决定一条参数规划是否落库；返回 {decision, code, score}。

    decision ∈ adopt / pending / reject。声线本身永远不由 LLM 决定。
    """
    conf = config(cfg)["hysteresis"]
    proposed = params_of(assignment)
    current_params = binding_params(current)
    reason_code = str(assignment.get("reason_code") or "")
    score = param_diff_score(current_params, proposed)

    if conf["enabled"]:
        if score < conf["release_band"]:
            return {"decision": "reject", "code": "param_below_release_band",
                    "score": score}
        if score < conf["param_threshold"] and reason_code != "post_rebind_adjust":
            return {"decision": "pending", "code": "param_in_hysteresis_band",
                    "score": score}

    if current.get("tier") == "main":
        if (abs(proposed["rate"] - current_params["rate"]) > conf["main_param_max_pct"]
                or abs(proposed["pitch"] - current_params["pitch"]) > conf["main_param_max_hz"]
                or abs(proposed["volume"] - current_params["volume"]) > conf["main_param_max_volume"]):
            return {"decision": "reject", "code": "main_param_delta_exceeded",
                    "score": score}
        if conf["main_extra_approval"]:
            return {"decision": "pending", "code": "main_param_manual",
                    "score": score}

    anchor = binding_params(current.get("params_anchor") or current)
    cumulative = {
        "rate": abs(proposed["rate"] - anchor["rate"]),
        "pitch": abs(proposed["pitch"] - anchor["pitch"]),
        "volume": abs(proposed["volume"] - anchor["volume"]),
    }
    if (cumulative["rate"] > conf["max_cumulative_rate_pct"]
            or cumulative["pitch"] > conf["max_cumulative_pitch_hz"]
            or cumulative["volume"] > conf["max_cumulative_volume_pct"]):
        return {"decision": "reject", "code": "cumulative_drift_exceeded",
                "score": score}

    if reason_code == "post_rebind_adjust" and conf["enabled"]:
        pass  # 人工改绑后的首次微调可立即采纳，不受连续章数限制
    else:
        streak = 1
        for item in reversed(history):
            if item.get("character_id") != cid:
                continue
            if (item.get("decision") == "adopt"
                    and float(item.get("score") or 0) >= conf["param_threshold"]):
                streak += 1
            else:
                break
        need = int(conf["consecutive_chapters"])
        if streak < need:
            return {"decision": "pending", "code": "param_needs_consecutive_chapters",
                    "score": score, "streak": streak, "need": need}

    window = int(conf["window_chapters"])
    recent = [item for item in history
              if item.get("character_id") == cid
              and item.get("decision") == "adopt"
              and abs(int(item.get("chapter") or 0) - int(assignment.get("chapter") or 0)) <= window]
    if len(recent) >= int(conf["max_changes_per_character"]):
        return {"decision": "reject", "code": "change_budget_exceeded", "score": score}
    if recent:
        last = max(int(item.get("chapter") or 0) for item in recent)
        if int(assignment.get("chapter") or 0) - last < int(conf["min_chapters_between_changes"]):
            return {"decision": "reject", "code": "change_too_frequent", "score": score}
    return {"decision": "adopt", "code": reason_code or "param_audible_change",
            "score": score}


# ---------------------------------------------------------------- 校验器

def retired_effective(retirement: dict | None, cfg: dict) -> str:
    """退场判定的有效值：低置信度/缺证据一律按 hidden（保守）。"""
    conf = config(cfg)["retirement"]
    if not retirement:
        return ""
    if retirement.get("_body") and not _evidence_ok(
            retirement.get("evidence_quote", ""), retirement.get("_body", "")):
        return "invalid"
    status = str(retirement.get("status") or "")
    if status not in RETIRE_STATUS_ALL:
        return "invalid"
    if float(retirement.get("confidence") or 0) < float(conf["min_confidence"]):
        return str(conf["low_confidence_status"])
    return status


def mentioned_in_chapter(cid: str, record: dict, identities: dict) -> bool:
    """被提及但无台词也算在场：名字/别名出现在本章任意段落。"""
    ident = identities.get(cid) or {}
    names = [str(ident.get("canonical_name") or cid)] + [str(x) for x in ident.get("aliases") or []]
    body = "".join(str(seg.get("text") or "") for seg in record.get("segments") or [])
    return any(name and name in body for name in names)


def idle_after_chapter(stat: dict, record: dict, cid: str, identities: dict) -> int:
    if mentioned_in_chapter(cid, record, identities):
        return 0
    return int(stat.get("idle_chapters") or 0)


def dormant_allowed(stat: dict, retirement: dict | None, cfg: dict) -> dict:
    """N=5 + 双门禁 + 退场判定；hidden/likely_return 一律不 dormant。"""
    reclaim = ((cfg.get("casting") or {}).get("voice_rules") or {}).get("reclaim") or {}
    idle = int(reclaim.get("idle_chapters", 5))
    floor = int(reclaim.get("idle_chapters_floor", 4))
    idle = max(floor, idle)
    min_lines = int(reclaim.get("min_lines_keep", 20))
    max_share = float(reclaim.get("max_share", 0.05))
    status = retired_effective(retirement, cfg)
    if status in ("hidden", "likely_return"):
        return {"dormant": False, "code": f"retirement_{status}"}
    if status == "retired":
        return {"dormant": True, "code": "retirement_retired"}
    if status == "present":
        return {"dormant": False, "code": "retirement_present"}
    # LLM 不可用或无效 → 规则回落
    if int(stat.get("idle_chapters") or 0) >= idle \
            and int(stat.get("total_lines") or 0) < min_lines \
            and float(stat.get("share") or 0) < max_share:
        return {"dormant": True, "code": "rule_fallback"}
    return {"dormant": False, "code": "rule_not_met"}


def promotion_decision(lease: dict, original_stat: dict, retirement: dict | None,
                       cfg: dict) -> dict:
    """长期租用者转正：buyback 优先，其次转正审批；转正后不可逆。"""
    rules = ((cfg.get("casting") or {}).get("voice_rules") or {}).get("reclaim") or {}
    promo = rules.get("promotion") or {}
    if not promo.get("enabled", False):
        return {"promote": False, "code": "promotion_disabled"}
    leases = int(lease.get("lease_count") or 0)
    active = int(lease.get("active_chapters") or 0)
    idle = int(original_stat.get("idle_chapters") or 0)
    status = retired_effective(retirement, cfg)
    if leases < int(promo.get("min_leases", 3)):
        return {"promote": False, "code": "promotion_not_enough_leases"}
    if active < int(promo.get("min_active_chapters", 12)):
        return {"promote": False, "code": "promotion_not_enough_active"}
    if idle < max(int(rules.get("idle_chapters", 5)), int(rules.get("idle_chapters_floor", 4))):
        return {"promote": False, "code": "promotion_original_not_absent"}
    if promo.get("require_llm_status", "retired") and status != "retired":
        return {"promote": False, "code": "promotion_requires_retired"}
    if str(promo.get("approval", "manual")) == "manual":
        return {"promote": True, "code": "promotion_pending_manual"}
    return {"promote": True, "code": "promotion_auto"}


def validate_assignment(assignment: dict, ctx: dict) -> list[dict]:
    """单条规划的结构化校验；ctx 含 catalog/bindings/chat/config/chapter。"""
    errors = []
    cid = str(assignment.get("character_id") or "")
    if cid not in ctx["identities"]:
        return [{"code": "unknown_character", "character_id": cid}]
    entry = ctx["catalog"].get(str(assignment.get("voice_id") or ""))
    if entry is None:
        errors.append({"code": "voice_not_in_catalog", "character_id": cid,
                       "voice_id": assignment.get("voice_id")})
    else:
        if entry.get("accented"):
            ident = ctx["identities"].get(cid) or {}
            has_evidence = bool(ident.get("dialect_profile")) or any(
                tag in str(assignment.get("evidence_quote") or "")
                for tag in ("粤语", "台湾腔", "东北", "陕西", "口音"))
            if not has_evidence:
                errors.append({"code": "accented_requires_evidence", "character_id": cid})
        if entry.get("id") == ctx.get("narrator_voice"):
            errors.append({"code": "narrator_voice_forbidden", "character_id": cid})
        if not voice_fits(entry, str(assignment.get("gender") or "未知"),
                          str(assignment.get("age_band") or "未知")):
            errors.append({"code": "voice_gender_age_mismatch", "character_id": cid})
    if not _evidence_ok(assignment.get("evidence_quote", ""), ctx["body"]):
        errors.append({"code": "evidence_mismatch", "character_id": cid})
    _bid, current = current_binding(ctx["registry"], cid)
    if current:
        if str(current.get("status") or "") == "lent":
            errors.append({"code": "lent_conflict", "character_id": cid})
        elif str(current.get("status") or "") == "dormant" \
                and str(assignment.get("reason_code")) != "exit_return":
            errors.append({"code": "dormant_conflict", "character_id": cid})
        if current.get("tier") == "main" and str(assignment.get("tier")) != "main":
            errors.append({"code": "main_tier_downgrade", "character_id": cid})
    return errors


def validate_plan(assignments: list[dict], ctx: dict) -> dict:
    accepted, rejected = [], []
    for assignment in assignments:
        errors = validate_assignment(assignment, ctx)
        if errors:
            rejected.append({"assignment": assignment, "errors": errors})
        else:
            accepted.append(assignment)
    # 池容量：同 gender×age_band 的不同 voice 数不得超过 pool_capacity
    capacity = ctx.get("capacity") or {}
    buckets: dict[tuple, set] = {}
    for assignment in accepted:
        key = (str(assignment.get("gender")), str(assignment.get("age_band")), "main")
        buckets.setdefault(key, set()).add(str(assignment.get("voice_id")))
    for (gender, age, pool), voices in buckets.items():
        slot = ((capacity.get(pool) or {}).get(age) or {}).get(gender) or []
        limit = len(slot) or 0
        if limit and len(voices) > limit:
            for assignment in accepted:
                if (str(assignment.get("gender")), str(assignment.get("age_band")),
                        "main") == (gender, age, pool):
                    rejected.append({"assignment": assignment,
                                     "errors": [{"code": "pool_capacity_exceeded",
                                                 "character_id": assignment.get("character_id"),
                                                 "limit": limit, "used": len(voices)}]})
            accepted = [item for item in accepted if not (
                (str(item.get("gender")), str(item.get("age_band")), "main")
                == (gender, age, pool))]
    # 同章撞声：同一声线两名角色参数距离必须 ≥ min_distance
    by_voice: dict[str, list[dict]] = {}
    for assignment in accepted:
        by_voice.setdefault(str(assignment.get("voice_id")), []).append(assignment)
    for voice, rows in by_voice.items():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                distance = collision_distance(params_of(rows[i]), params_of(rows[j]))
                if distance < COLLISION_MIN_DISTANCE:
                    rejected.append({"assignment": rows[j],
                                     "errors": [{"code": "collision_below_threshold",
                                                 "character_id": rows[j].get("character_id"),
                                                 "voice_id": voice,
                                                 "distance": distance}]})
                    accepted.remove(rows[j])
    return {"accepted": accepted, "rejected": rejected}


# ---------------------------------------------------------------- Prompt

PROFILE_SYSTEM = """你是小说角色档案分析器。只输出 JSON。

任务：根据给出的章节原文，为每个有台词或被提及的角色推断：
- gender：男/女/未知（依据亲属称谓、代词、身份词，如“母亲/公主/夫人/将军/他/她”）；
- age_band：少年(0-17)/青年(18-34)/中年(35-54)/老年(55+)/未知；
- aliases：原文中出现过的其他叫法；
- speech_style：一句话描述说话风格；
- evidence_quote：必须逐字引用原文中能支撑判断的一句话；
- evidence_chapter：该句所在章号；
- confidence：0~1。

没有证据就填“未知”，不要猜。只输出一个 JSON 对象。"""

PLAN_SYSTEM = """你是整本书的配音参数规划师。只输出 JSON。

硬边界：你无权更换声线。voice_id 由既有绑定固定（首次分配由 caster 规则或人工完成），
你的输出里不允许出现 voice_id / voice 字段，出现即整条作废。

规则：
1. 默认沿用当前参数，不要为了“更合适”改。正常章节的期望变更是 0 条。
2. 只对已经在 voice_bindings 里有声线的角色输出参数建议；没有绑定的角色不要输出。
3. 只有当原文对“声音/嗓音/语调/语速/音量”有明确描写，且听感差异足够大时才输出；
   面部表情、情绪词不算。
4. reason_code：普通可听差异用 param_audible_change；人工改绑后立即微调用
   post_rebind_adjust；其余结构性理由只用于审计说明，不能借此换声。
5. main 档的角色参数改动幅度要小，超出上限会被校验器拒绝。
6. 保持不变的角色放进 kept_character_ids。
7. 每条建议必须给 evidence_quote（逐字引用原文）与 confidence。
8. 参数范围：rate_pct -16~25，pitch_hz -8~8，volume_pct -10~10。

只输出一个 JSON 对象。"""

RETIRE_SYSTEM = """你是小说角色去向判定器。只输出 JSON。

对给出的角色判断本章之后的状态：
- retired：明确退场/死亡/确认不再出现（唯一可外借声线的状态）；
- hidden：作者刻意藏线，暂时没有戏但可能回来；
- likely_return：有明确伏笔/预告表明会回来；
- present：本章仍在场（有台词或被提及）。
evidence_quote 必须逐字引用原文；判断不了就填 hidden，不要编造。
只输出一个 JSON 对象。"""


def profile_messages(records: list[dict], identities: dict, cfg: dict) -> list[dict]:
    max_chars = int(config(cfg)["profile"]["max_chars"])
    lines = ["角色清单（id | 名字 | 现有性别 | 现有年龄段）："]
    for cid, ident in identities.items():
        lines.append(f"- {cid} | {ident.get('canonical_name') or cid} | "
                     f"{ident.get('gender') or '未知'} | "
                     f"{(ident.get('current_stage') or '未知')}")
    lines += ["", "章节原文（只给有台词/被提及的部分）："]
    used = 0
    for record in records:
        body = chapter_body(record, max_chars)
        lines.append(f"\n=== 第 {record.get('chapter_index')} 章 {record.get('title')} ===")
        lines.append(body)
        used += len(body)
        if used >= max_chars:
            break
    return [{"role": "system", "content": PROFILE_SYSTEM},
            {"role": "user", "content": "\n".join(lines)}]


def plan_messages(records: list[dict], identities: dict, profiles: list[dict],
                  catalog: dict, registry: dict, cfg: dict, chapter: int,
                  book_id: str) -> list[dict]:
    lines = ["声线目录（voice_id | 性别 | 适用年龄段 | 是否方言）："]
    for vid, entry in catalog.items():
        lines.append(f"- {vid} | {entry.get('gender')} | "
                     f"{'/'.join(entry.get('stages') or [])} | "
                     f"{'方言' if entry.get('accented') else '通用'}")
    lines += ["", "角色档案（id | 性别 | 年龄段 | 档位 | 累计章/句/占比）："]
    for cid, ident in identities.items():
        prof = next((item for item in profiles if item.get("character_id") == cid), {})
        appearance = (ident.get("appearance_by_book") or {}).get(book_id, {})
        lines.append(
            f"- {cid} | {prof.get('gender') or ident.get('gender') or '未知'} | "
            f"{prof.get('age_band') or '未知'} | {ident.get('tier') or 'generic'} | "
            f"{appearance.get('chapters_seen', 0)}/{appearance.get('total_lines', 0)}/"
            f"{round(float(appearance.get('share') or 0) * 100, 1)}%")
    lines += ["", "当前绑定（id | 声线 | rate/pitch/volume | 状态）："]
    for cid in identities:
        _bid, binding = current_binding(registry, cid)
        lines.append(f"- {cid} | {binding.get('voice') or '无'} | "
                     f"{binding.get('rate') or '+0%'}/{binding.get('pitch') or '+0Hz'}/"
                     f"{binding.get('volume') or '+0%'} | {binding.get('status') or 'locked'}")
    lines += ["", f"规划章号：{chapter}"]
    for record in records:
        lines.append(f"\n=== 第 {record.get('chapter_index')} 章 {record.get('title')} ===")
        lines.append(chapter_body(record, int(config(cfg)["profile"]["max_chars"]) // 2))
    return [{"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": "\n".join(lines)}]


def retire_messages(record: dict, identities: dict, cfg: dict) -> list[dict]:
    body = chapter_body(record, int(config(cfg)["profile"]["max_chars"]))
    lines = [f"第 {record.get('chapter_index')} 章 {record.get('title')}", "",
             "角色清单（id | 名字）："]
    for cid, ident in identities.items():
        lines.append(f"- {cid} | {ident.get('canonical_name') or cid}")
    lines += ["", "本章原文：", body]
    return [{"role": "system", "content": RETIRE_SYSTEM},
            {"role": "user", "content": "\n".join(lines)}]


# ---------------------------------------------------------------- 抽取与落地

def infer_profiles(client, records: list[dict], identities: dict, cfg: dict) -> dict:
    payload = client.chat_json(profile_messages(records, identities, cfg),
                               schema=PROFILE_SCHEMA, schema_name="voice_profiles")
    body = "".join(chapter_body(record, 10 ** 9) for record in records)
    idx = name_index(identities)
    accepted, rejected = [], []
    for raw in payload.get("profiles") or []:
        cid = str(raw.get("character_id") or "")
        cid = idx.get(cid, cid)
        if cid not in identities:
            rejected.append({"raw": raw, "code": "unknown_character"})
            continue
        if not _evidence_ok(raw.get("evidence_quote", ""), body):
            rejected.append({"raw": raw, "code": "profile_evidence_mismatch"})
            continue
        item = dict(raw)
        item["character_id"] = cid
        item["_evidence_ok"] = True
        accepted.append(item)
    return {"profiles": accepted, "rejected": rejected}


def propose_plan(client, records: list[dict], identities: dict, profiles: list[dict],
                 catalog: dict, registry: dict, cfg: dict, chapter: int,
                 book_id: str) -> dict:
    payload = client.chat_json(
        plan_messages(records, identities, profiles, catalog, registry, cfg, chapter,
                      book_id),
        schema=plan_schema(cfg), schema_name="voice_plan")
    assignments = []
    for raw in payload.get("assignments") or []:
        item = dict(raw)
        item["chapter"] = chapter
        assignments.append(item)
    return {"assignments": assignments,
            "kept": payload.get("kept_character_ids") or []}


def judge_retirement(client, record: dict, identities: dict, cfg: dict) -> dict:
    payload = client.chat_json(retire_messages(record, identities, cfg),
                               schema=RETIREMENT_SCHEMA, schema_name="retirement")
    body = chapter_body(record, 10 ** 9)
    judgments = []
    for raw in payload.get("judgments") or []:
        item = dict(raw)
        cid = str(item.get("character_id") or "")
        cid = name_index(identities).get(cid, cid)
        item["character_id"] = cid
        item["chapter"] = int(record.get("chapter_index") or 0)
        item["_body"] = body
        status = retired_effective(item, cfg)
        item["effective_status"] = status
        judgments.append(item)
    return {"judgments": judgments}


def plan_diff(assignments: list[dict], registry: dict) -> dict:
    """规划 vs 现有绑定逐条对比。"""
    rows = []
    for assignment in assignments:
        cid = str(assignment.get("character_id") or "")
        bid, current = current_binding(registry, cid)
        proposed = params_of(assignment)
        cur_params = binding_params(current)
        voice_changed = str(assignment.get("voice_id")) != str(current.get("voice") or "")
        rows.append({
            "character_id": cid,
            "binding_id": bid,
            "from_voice": current.get("voice") or "",
            "to_voice": assignment.get("voice_id"),
            "voice_changed": voice_changed,
            "from_params": cur_params,
            "to_params": proposed,
            "param_score": param_diff_score(cur_params, proposed),
            "reason_code": assignment.get("reason_code"),
            "evidence_quote": assignment.get("evidence_quote"),
            "confidence": assignment.get("confidence"),
            "type": "voice_swap" if voice_changed else "param_only",
        })
    return {"rows": rows, "voice_swaps": sum(1 for r in rows if r["voice_changed"]),
            "param_only": sum(1 for r in rows if not r["voice_changed"])}


def apply_plan(registry: dict, assignments: list[dict], decisions: list[dict],
               cfg: dict, dry_run: bool = True) -> dict:
    """落库：adopt 写 binding + voice_events；pending 进 pending_review；reject 记审计。"""
    if dry_run:
        return {"dry_run": True, "adopted": 0, "pending": 0, "rejected": 0}
    log = registry.setdefault("voice_events", [])
    pending = registry.setdefault("pending_review", [])
    audit = registry.setdefault("voice_plan_audit", [])
    adopted = 0
    for assignment, decision in zip(assignments, decisions):
        cid = assignment.get("character_id")
        record = {
            "id": f"vp:{_hash(assignment)}", "book_id": assignment.get("book_id"),
            "chapter": assignment.get("chapter"), "character_id": cid,
            "voice_id": assignment.get("voice_id"),
            "reason_code": assignment.get("reason_code"),
            "evidence_quote": assignment.get("evidence_quote"),
            "confidence": assignment.get("confidence"),
            "decision": decision.get("decision"), "code": decision.get("code"),
            "score": decision.get("score"), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if decision.get("decision") == "adopt":
            bid, binding = current_binding(registry, cid)
            if binding:
                binding["prev"] = {key: binding.get(key) for key in
                                   ("voice", "rate", "pitch", "volume", "tier")}
                binding["voice"] = assignment.get("voice_id")
                binding["rate"] = f"{assignment.get('rate_pct', 0):+.0f}%"
                binding["pitch"] = f"{assignment.get('pitch_hz', 0):+.0f}Hz"
                binding["volume"] = f"{assignment.get('volume_pct', 0):+.0f}%"
                binding["tier"] = assignment.get("tier")
                binding["updated_by"] = "llm_plan"
                binding["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            log.append(record)
            adopted += 1
        elif decision.get("decision") == "pending":
            pending.append(record)
        audit.append(record)
    return {"dry_run": False, "adopted": adopted,
            "pending": sum(1 for d in decisions if d.get("decision") == "pending"),
            "rejected": sum(1 for d in decisions if d.get("decision") == "reject")}


def dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
