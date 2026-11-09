# -*- coding: utf-8 -*-
"""LLM 归因模块：人名别名合并、低归因说话人归属、跨语言角色识别。

本包独立于 ``casting.corpus / engine / cli``，只暴露函数，由调用方接入。
不做声线分配、不改 config 默认值、不写 registry / v11_* / v13_*。

接口（v1）
---------------------------------------------------------------
公共返回结构
    ::

        {
          "task": "alias" | "speaker" | "document",
          "status": "merged" | "no_merge" | "resolved" | "unresolved"
                    | "pending" | "invalid" | "low_confidence" | "ok" | "partial",
          "degraded": bool,          # LLM 不可用/超时/非法 schema 时为 True
          "reason": str,
          "usage": {"calls", "prompt_chars", "completion_chars",
                    "est_prompt_tokens", "est_completion_tokens",
                    "est_total_tokens", "elapsed_s", "models"},
          ...
        }

1) 人名别名合并
    ``resolve_alias(paragraph, names, candidate, *, client=None,
                    rule_canonical=None, min_confidence=0.8) -> dict``

    - ``paragraph``: 原文片段（字符串）
    - ``names``: 已有人名列表（canonical）
    - ``candidate``: 待判定称谓，如 ``"楼公子"``
    - ``rule_canonical``: 规则层当前结论，用于降级回退
    - 返回 ``{alias, canonical, merged, confidence, evidence_quotes, ...}``
    - ``evidence_quotes`` 必须全部能在 ``paragraph`` 中逐字定位；有一条定位
      不到，整条判 ``invalid``，``merged=False``。
    - ``merged=True`` 需同时满足：``canonical`` 在 ``names`` 中、confidence
      ≥ 阈值、至少一条 evidence、evidence 含称谓或正名。
    - 宁可漏合并，不可误合并：两个真实角色共现/同姓/同尾字都不是合并理由。

    ``resolve_alias_batch(cases, *, client=None) -> list[dict]``
    ``cases`` 为 ``[{"paragraph", "names", "candidate", "rule_canonical"?, "id"?}]``。

2) 低归因 / 引号识别失败的说话人归属
    ``resolve_speaker(paragraphs, index, characters, *, lang="zh",
                       max_context_chars=1000, min_confidence=0.7,
                       client=None, rule_speaker=None,
                       radii=None) -> dict``

    - ``paragraphs``: 同一章的段落列表；``None`` 表示场景边界，``""`` 跳过。
    - ``index``: 待判定段落下标。
    - ``characters``: 角色名列表，或 ``[{"name"/"canonical": ...}]``。
    - 上下文从 ``index`` 向两侧逐级扩大：±1 → ±3 → ±5 …，累计（不含待判定段）
      不超过 ``max_context_chars``（默认 1000）。到 1000 字仍判不出 →
      ``status="unresolved"``，``speaker=None``，不猜。
    - 低置信度只记录在 ``history``，不采纳。
    - 返回 ``{speaker, confidence, evidence_quotes, window_chars, radius,
      attempts, history, ...}``。
    - 允许返回名单外但原文中逐字出现的新名字（``speaker_new=True``），
      避免人名挖掘漏项时整条失效。

3) 跨语言正文
    ``resolve_document(text, lang, *, max_chars=6000, client=None) -> dict``

    - ``lang``: ``"en"`` / ``"ja"`` / ``"zh"``（其他语言走通用提示）。
    - 返回统一角色实体表：``entities[]``，每项含
      ``name / aliases / gender / gender_evidence / age_band / age_evidence /
      turns[]``；``turns`` 的每条含 ``quote / speaker / confidence / evidence``。
    - 只输出证据与判断，不输出任何 voice_id、声线名、参数值。
    - 年龄数值不交给 LLM 判断：``age_evidence`` 只收原文里明确的数字/称谓，
      ``age_band`` 由本模块的规则函数 ``derive_age_band`` 推导。
    - 性别判不出返回 ``"未知"``，由下游走旁白代读，这里不兜底猜。

降级
----
LLM 不可用、超时、链路全失败、返回非法 schema 时：
- 别名 → ``status="pending"``，``merged=False``，``canonical=rule_canonical``；
- 归属 → ``status="pending"``，``speaker=rule_speaker``；
- 跨语言 → ``status="pending"`` / ``"partial"``，已完成的分块保留。
任何情况都不抛异常中断流水线（调用参数错误除外）。

可复现
------
- 默认 ``temperature=0``，所有名单/实体/台词按固定键排序，不依赖 dict 顺序；
- 每次调用按 ``sha256(版本 + 模型 + schema + messages)`` 落盘缓存，默认目录
  ``outputs/novel-forge/v14_llm_resolve_cache``（原子写入）；
- 同一输入二次运行直接命中缓存，返回与首次完全一致的结构。
  关掉缓存（``NOVEL_FORGE_LLM_CACHE=0``）时，可复现只由 ``temperature=0``
  保证；实测供应商仍可能对相同 prompt 给出不同答案，因此默认开启缓存。
"""

from .alias import resolve_alias, resolve_alias_batch
from .attribution import resolve_speaker, build_context, DEFAULT_RADII
from .multilingual import resolve_document, derive_age_band
from .client import Usage, get_client, set_client, estimate_tokens
from .eval_support import load_annotations, resolve_annotation_needle

__version__ = "1.0.0"

__all__ = [
    "resolve_alias",
    "resolve_alias_batch",
    "resolve_speaker",
    "build_context",
    "DEFAULT_RADII",
    "resolve_document",
    "derive_age_band",
    "Usage",
    "get_client",
    "set_client",
    "estimate_tokens",
    "load_annotations",
    "resolve_annotation_needle",
]
