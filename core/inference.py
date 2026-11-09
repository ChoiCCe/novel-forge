"""v4 步骤 3：性别/年龄推断与成长防抖。

所有阈值、阶段边界、防抖章数都来自 `config.yaml -> casting`，代码不保留数值默认值。
"""
from __future__ import annotations

import re

MALE_PRONOUN = re.compile(r"他|他的|他自己")
FEMALE_PRONOUN = re.compile(r"她|她的|她自己")
MALE_TITLE = re.compile(r"老爷|公子|先生|大人|掌柜|将军|少爷|公子爷|大叔|老伯|大哥")
FEMALE_TITLE = re.compile(r"夫人|姑娘|小姐|娘子|嫂子|婆婆|丫头|姑娘家|大姐")
MALE_SELF = re.compile(r"在下|小人|本官|老夫|某家|小生|晚辈")
FEMALE_SELF = re.compile(r"妾身|奴家|小女子|民女|奴婢|老身")
# 名字不足以决定性别，只做极弱的提示
FEMALE_NAME_HINT = re.compile(r"[娘姑姐妹婆嫂母妃]$")


def _section(cfg: dict, *path: str) -> dict:
    cur = cfg
    for key in path:
        cur = (cur or {}).get(key) or {}
        if not isinstance(cur, dict):
            raise KeyError("配置缺失：casting." + ".".join(path))
    return cur


def _value(cfg: dict, path: tuple[str, ...]):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError("配置缺失：" + ".".join(path))
        cur = cur[key]
    return cur


def _num(cfg: dict, path: tuple[str, ...]) -> float:
    try:
        return float(_value(cfg, path))
    except (TypeError, ValueError) as err:
        raise KeyError("配置不是数字：" + ".".join(path)) from err


def stages(cfg: dict) -> tuple[str, ...]:
    return tuple(_value(cfg, ("casting", "age", "stage_order")))


def bands(cfg: dict) -> dict:
    return _value(cfg, ("casting", "age", "bands"))


def band_for_age(age: int, cfg: dict) -> str:
    """按 config 的 age.bands 把年龄落进阶段。"""
    band_map = bands(cfg)
    for stage in stages(cfg):
        low, high = band_map[stage]
        if low <= age <= high:
            return stage
    return str(_value(cfg, ("casting", "age", "default_stage")))


def band_index(stage: str, cfg: dict) -> int:
    order = stages(cfg)
    return order.index(stage) if stage in order else order.index(
        _value(cfg, ("casting", "age", "default_stage")))


def _count(pattern, text: str) -> int:
    return len(pattern.findall(text or ""))


def resolve_gender(name: str, persona: dict, cfg: dict) -> dict:
    """加权推断性别，返回 {gender, confidence, evidence, source}。"""
    gcfg = _section(cfg, "casting", "gender")
    w = gcfg["weight"]
    text = " ".join(str(persona.get(k, "")) for k in
                    ("gender_evidence", "persona", "voice_style", "evidence"))
    llm_gender = persona.get("gender")
    llm_gender = llm_gender if llm_gender in ("男", "女") else ""
    conf_raw = persona.get("gender_confidence")
    try:
        llm_conf = float(conf_raw) if conf_raw not in (None, "") else float(gcfg["llm_default_confidence"])
    except (TypeError, ValueError):
        llm_conf = float(gcfg["llm_default_confidence"])
    llm_conf = min(llm_conf, float(gcfg["llm_confidence_cap"]))

    w_llm = float(w["llm"])
    w_pron = float(w["pronoun"])
    pron_max = float(w["pronoun_max"])
    w_title = float(w["title"])
    w_self = float(w["self_reference"])
    w_name = float(w["name_hint"])
    penalty = float(w["conflict_penalty"])

    m_pron, f_pron = _count(MALE_PRONOUN, text), _count(FEMALE_PRONOUN, text)
    m_title, f_title = _count(MALE_TITLE, text), _count(FEMALE_TITLE, text)
    m_self, f_self = _count(MALE_SELF, text), _count(FEMALE_SELF, text)
    name_female = bool(FEMALE_NAME_HINT.search(name or ""))

    score_m = score_f = 0.0
    bits: list[str] = []
    if llm_gender == "男":
        score_m += w_llm * llm_conf
    elif llm_gender == "女":
        score_f += w_llm * llm_conf
    if m_pron:
        score_m += min(pron_max, w_pron * m_pron)
        bits.append(f"男代词×{m_pron}")
    if f_pron:
        score_f += min(pron_max, w_pron * f_pron)
        bits.append(f"女代词×{f_pron}")
    if m_title:
        score_m += w_title
        bits.append("男称谓")
    if f_title:
        score_f += w_title
        bits.append("女称谓")
    if m_self:
        score_m += w_self
        bits.append("男自称")
    if f_self:
        score_f += w_self
        bits.append("女自称")
    if name_female:
        score_f += w_name
        bits.append("名字偏女")

    if score_m > 0 and score_f > 0:
        score_m -= penalty / 2
        score_f -= penalty / 2
        bits.append("男女证据冲突")
        score_m, score_f = max(0.0, score_m), max(0.0, score_f)

    if score_m == 0 and score_f == 0:
        return {"gender": "未知", "confidence": float(gcfg["unknown_confidence"]),
                "source": "unknown", "evidence": "无性别证据，待确认"}

    score_m, score_f = min(1.0, score_m), min(1.0, score_f)
    gap = abs(score_m - score_f)
    best = max(score_m, score_f)
    review = float(gcfg["review"])
    min_gap = float(gcfg["min_gap"])
    if best < review or gap < min_gap:
        return {
            "gender": "未知",
            "confidence": round(best, 3),
            "source": "low_confidence",
            "evidence": ("、".join(bits) + f"（证据不足：{best:.2f}，差 {gap:.2f}）").strip(),
        }
    gender = "男" if score_m > score_f else "女"
    return {
        "gender": gender,
        "confidence": round(best, 3),
        "source": "llm+heuristic" if llm_gender else "heuristic",
        "evidence": "、".join(bits) or (f"LLM 判断{llm_gender}" if llm_gender else ""),
    }


def decide_age(profile: dict, info: dict, chapter: int, cfg: dict) -> dict:
    """根据一条年龄证据决定是否升阶段，执行全部防抖规则。

    返回 {action, target, reason, transition}；不修改 profile（日志由调用方写）。
    """
    acfg = _section(cfg, "casting", "age")
    order = stages(cfg)
    default_stage = str(acfg["default_stage"])
    band_map = bands(cfg)
    current = profile.get("age_stage") or profile.get("age_band") or default_stage
    if current not in order:
        current = default_stage
    target = info.get("stage") if info.get("stage") in order else None
    try:
        confidence = float(info.get("confidence", acfg["evidence_default_confidence"]))
    except (TypeError, ValueError):
        confidence = float(acfg["evidence_default_confidence"])
    evidence = str(info.get("evidence", ""))
    age = info.get("age")
    auto = float(acfg["auto_apply_confidence"])
    use_real_age = bool(acfg["use_real_age"]
                        and ((profile.get("book_real_age") is True) or acfg.get("force_all_books")))

    if target is None:
        return {"action": "hold", "target": None, "reason": "缺少目标阶段", "transition": False}
    if band_index(target, cfg) < band_index(current, cfg):
        return {"action": "hold", "target": None, "reason": "拒绝降级", "transition": False}
    if target == current:
        return {"action": "hold", "target": None, "reason": "阶段未变", "transition": False}

    flash = any(k in evidence for k in acfg["flashback_keywords"])
    if flash:
        confidence *= float(acfg["flashback_confidence_factor"])

    # 单章最多升一档：目标跨级时压到下一档
    idx = band_index(current, cfg) + 1
    jump_capped = False
    if band_index(target, cfg) > idx:
        target = order[idx]
        jump_capped = True

    cooldown = int(acfg["cooldown_chapters"])
    entered = profile.get("stage_entered_chapter") or profile.get("first_seen_chapter")
    if isinstance(entered, int) and chapter - entered < cooldown:
        return {"action": "hold", "target": None,
                "reason": f"冷却中（{chapter - entered}/{cooldown} 章）", "transition": False}

    if confidence < auto:
        return {"action": "hold", "target": None,
                "reason": f"置信度不足 {confidence:.2f}<{auto}", "transition": False}

    if isinstance(age, (int, float)):
        # 真实年龄模式：越过当前阶段上界即升阶（如 17→18）；否则走缓冲线
        need = int(band_map[current][1]) + 1 if use_real_age else int(acfg["advance_min_age"][current])
        if age < need:
            return {"action": "hold", "target": None,
                    "reason": f"{age} 岁未越过 {current}→{target} 缓冲线 {need}", "transition": False}
        reason = (f"{age} 岁 ≥ {need}（{'真实年龄' if use_real_age else '缓冲线'}）"
                  + (f"；跨级压到 {target}" if jump_capped else "")
                  + ("；含回忆标记已降权" if flash else ""))
        return {"action": "advance", "target": target, "reason": reason,
                "transition": bool(acfg["transition"]["chapters"])}

    # 无数字年龄：需要连续 N 章同类证据
    need_chapters = int(acfg["consecutive_evidence"])
    log = profile.get("age_evidence_log") or []
    prev = log[-1] if log else None
    consecutive = bool(
        prev and prev.get("stage") == target and prev.get("chapter") == chapter - 1
        and prev.get("confidence", 0) >= auto)
    if not consecutive:
        return {"action": "hold", "target": None,
                "reason": f"无数字年龄，需连续 {need_chapters} 章同类证据", "transition": False}
    return {"action": "advance", "target": target, "reason": "连续两章无数字年龄证据",
            "transition": bool(acfg["transition"]["chapters"])}


def log_evidence(profile: dict, info: dict, chapter: int, cfg: dict) -> list[dict]:
    cap = int(_value(cfg, ("casting", "age", "evidence_log_cap")))
    log = profile.setdefault("age_evidence_log", [])
    log.append({
        "chapter": chapter,
        "age": info.get("age"),
        "stage": info.get("stage"),
        "confidence": round(float(info.get("confidence")), 3),
        "evidence": str(info.get("evidence", ""))[:60],
    })
    del log[:-cap]
    return log


def blend_transition(profile: dict, chapter: int, cfg: dict) -> dict | None:
    """同声线跨阶段的渐变；返回混合后的 rate/pitch/volume，结束后返回 {}。"""
    tr = profile.get("transition")
    if not isinstance(tr, dict):
        return None
    chapters = int(tr.get("chapters") or _value(cfg, ("casting", "age", "transition", "chapters")))
    elapsed = chapter - int(tr.get("start_chapter", chapter)) + 1
    if elapsed > chapters:
        return {}
    t = min(1.0, elapsed / float(chapters))
    out = {}
    for key in ("rate", "pitch", "volume"):
        a = float(tr.get("from", {}).get(key, 0.0))
        b = float(tr.get("to", {}).get(key, 0.0))
        out[key] = a + (b - a) * t
    return out
