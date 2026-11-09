# -*- coding: utf-8 -*-
"""1.2.0 人名挖掘：按上下文窗口切分，交给多个真实不同的模型投票。

主判不再是规则分词。规则层（``castmine`` 旧实现）只在 LLM 完全不可用时兜底。

判定单位
--------
一个“窗口”= 同一章内连续的若干段，累计不超过 ``window_chars``（默认 1300），
且至少含一处引号对白或“言说动词+冒号”。窗口内的段落就是**判定目标**，不改上下文。

投票（初筛）
------------
- 初始两个模型必须是模型链里两个不同的模型；
- 一个窗口里可能有多处引号、多个说话人：判定单位是**整个窗口**，但按“词”计票。
  某词被某模型列进 ``speakers``，该词得一张 name 票；列进 ``non_persons`` 得一张
  non_name 票；模型没提到该词 = 弃权，不是反对票。
- 两票一致且唯一最高 → 直接采纳（记 stage1）。
- 冲突（同一词既得 name 又得 non_name）不扩窗，**换判据**：让模型回答“该词在这个
  位置有没有可定位的言说证据”，并要求逐字抄回原句片段。片段必须能在窗口原文里
  逐字定位、包含该词、且含言说动词或称呼语。给不出可定位证据的不算有效判断；
  最多 3 次尝试，3 次都拿不出 → 直接判非角色，不再加模型。
- 无冲突但未定（只有一个模型提到该词）→ 依次加入池内其它模型加票。

机械验收（采纳前）
------------------
投票只是初筛，不能自证。最终采纳前对全书做一次证据扫描：

- 判为角色 → 全书必须至少有一处可定位的“某道/某说/某问”或引号归属证据，
  否则降为泛称，不进角色位；
- 判为非角色 → 若全书存在可定位言说证据，必须翻案转角色。

验收只校验 LLM 给的证据是否真实存在、是否含言说标记；不引入新的分词规则，
也不用词表判断是不是人名。

可复现
------
结果按 ``sha256(版本 + 提示词版本 + 模型 + schema + messages)`` 落盘缓存，
默认目录 ``outputs/novel-forge/v15_name_vote_cache``；同输入二次运行直接命中缓存。
供应商即使 temperature=0 也可能给出不同答案，因此可复现性依赖缓存。

边界
----
- 不输出声线、参数、年龄；schema 与语义层都会剥离/拒绝这些字段。
- 性别仍由 corpus 的规则证据链判定，判不出就是未知，走旁白代读。
- 引号缺失/错配的窗口标 ``seg_quality=degraded``，提示词明确告知模型，
  这类窗口强制投票（本流程本来就全量投票）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .corpus import (
    QUOTE_STYLES, SPEECH_ALT, STOP_NAMES, FUNC_CHARS, quote_spans,
)
from .llm_resolve.evidence import locate_verbatim

PROMPT_VERSION = "name-vote-v2"
SCHEMA_NAME = "name_vote_entities"
CACHE_VERSION = "v15-name-vote-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT.parent / "outputs" / "novel-forge" / "v15_name_vote_cache"
_CACHE_ENABLED = os.environ.get("NOVEL_FORGE_NAME_CACHE", "1") != "0"

DEFAULT_WINDOW_CHARS = 1300
# 免费/弱模型一次吃不下 4 个窗口，可用环境变量把打包数降到 1（不改变判定单位，
# 也不是抽样：每个窗口仍然全量送判）。
_BATCH = max(1, int(os.environ.get("NOVEL_FORGE_NAME_BATCH", "4") or 4))
BATCH_SIZE = _BATCH        # 每个请求打包若干个窗口；窗口仍是独立判定单位
EVIDENCE_BATCH = _BATCH    # 证据轮每个请求打包同样数量
MAX_EVIDENCE_ATTEMPTS = 3  # 冲突词最多换 3 次找证据；全失败判非角色
ARB_BATCH = _BATCH         # 仲裁仍用原窗口上下文
RATE_LIMIT_RETRIES = 5     # 429 只退避重试，不写失败缓存
RATE_LIMIT_BACKOFF = 5.0   # 5s / 10s / 20s / 40s

# 默认模型池：全部来自 config.yaml 的 model_chain（排除思考模型），顺序固定。
# 前两个是初始投票模型；池内其余模型既用于冲突词的证据轮，也用于无冲突但
# 未定词条的加票。加票一直加到出现唯一最高票或池耗尽，池耗尽才 unresolved。
# 2026-11-06 实测：硅基流动全账户 402；智谱 glm-4.7-flash 持续 429。
# 可用池只有两个真正不同的模型：glm-4-flash 做初筛，glm-4.5-flash 做质量兜底
# （必须关 thinking，否则 reasoning_content 吃光 token、正文为空）。
DEFAULT_MODELS = [
    "glm-4-flash",
    "glm-4.5-flash",
]

PRONOUNS = {
    "他", "她", "它", "他们", "她们", "它们", "我", "你", "您", "我们", "你们",
    "咱们", "自己", "对方", "大家", "众人", "有人", "某人", "旁人", "别人",
    "此人", "那人", "这人", "谁", "什么", "怎么", "为何", "这里", "那里",
}

BATCH_SCHEMA = {
    "type": "object",
    "required": ["results"],
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["window_id", "speakers"],
                "properties": {
                    "window_id": {"type": "string"},
                    "speakers": {
                        "type": "array", "items": {"type": "string"},
                        "maxItems": 16, "uniqueItems": True,
                    },
                    "non_persons": {
                        "type": "array", "items": {"type": "string"},
                        "maxItems": 16, "uniqueItems": True,
                    },
                },
            },
        },
    },
}

SYSTEM_PROMPT = """你是小说人物识别器。只输出 JSON，不要解释。

任务：在给定窗口里找出**在对话中开口的人物名称**，并指出看起来像人名、其实不是人物指称的词。

什么算人物名（speakers）
1. 原文逐字出现的姓名、名、字、号、绰号，如“沈耽”“楼温”“徐公子”，必须是窗口内的连续子串。
2. 全书固定指代同一个人的封号、爵位、尊称，如“齐王”“宁王”“徐先生”“金圣女”；
   判断标准是它在这个窗口里明确指某一个人，而不是泛指一类人。
3. 必须是真正开口的人：出现在“X道/说/问/答/喊/喝道”的主语位置，或紧贴引号对白的提示语里。
   只在叙述里被提到、没有开口的人不要列。

什么不是人物名
- 地名、国家、朝代、州郡、城池、军队、组织、家族、门派：吴国、晋军、楼家、范门；
- 物品、招式、功法、法宝、称号：降世棒、槊法、百目神幡；
- 泛指的职务、身份、群体、临时称呼：官兵、使者、刺客、车夫、老兵、幕僚、小吏、侍卫、
  众人、士兵、女子、汉子、老者、少女、少年、妇人。**即使它们出现在“X道：”的主语位置，
  也只是无名角色，不要当人名**；
- 代词：他、她、他们、此人、对方。
  以上这类词只在 non_persons 里列出（只列看起来像人名的，如“官兵”“吴国”“空间通道”“厉害”）。

其他规则
4. 同一个名字只出现一次，必须去重；同一个人有全名和简称时只列最完整的常用形式
   （有“张释清”就不列“释清”）。每个窗口最多 12 个，严禁逐句罗列说话人。
5. 引号可能缺失、错配或中英混用；靠段落边界与“道/说/问/答/喊/叫道”等言说动词判断。
6. 拿不准的不要猜；只列最有把握的 3 个以内。
7. 禁止输出声线、音色、voice_id、任何参数值，禁止判断年龄。

输出 JSON：{"results":[{"window_id":"W1","speakers":[],"non_persons":[]}]}"""

EVIDENCE_SCHEMA = {
    "type": "object",
    "required": ["results"],
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["window_id", "term", "has_evidence"],
                "properties": {
                    "window_id": {"type": "string"},
                    "term": {"type": "string"},
                    "has_evidence": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
            },
        },
    },
}

EVIDENCE_PROMPT = """你是小说对话归属校验器。只输出 JSON，不要解释。

任务：对给出的每个待查词，判断它在对应窗口里**有没有可定位的言说证据**。
不要判断“这个词像不像人名”，只找证据。

有效证据只有两类：
1. 该词紧跟言说动词并带冒号或引号，例如“沈耽道：”“沈耽笑道：“……”“沈耽问道：”；
2. 该词作为称呼语出现在引号内的对白里，例如“沈兄，你……”中的“沈兄”。

要求：
- evidence 必须从窗口原文**逐字**抄一段相邻文字，不得改写、补全、加省略号、跨段拼接；
- 片段必须包含该词；
- 找不到就 has_evidence=false，evidence 留空字符串；
- 窗口里的引号可能缺失或错配，按段落边界与“道/说/问/答/喝道/笑道”等动词判断；
- 只给证据，不要输出声线、参数、年龄。

输出 JSON：{"results":[{"window_id":"E1","term":"沈耽","has_evidence":true,"evidence":"沈耽道：“……”"}]}"""


# --------------------------------------------------------------------------- 基础工具
def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF


def _valid_term(term: str) -> bool:
    if not (2 <= len(term) <= 4):
        return False
    if not all(_is_cjk(ch) for ch in term):
        return False
    if term in PRONOUNS or term in STOP_NAMES:
        return False
    if any(ch in FUNC_CHARS for ch in term):
        return False
    return True


def _norm(term: Any) -> str:
    return re.sub(r"\s+", "", str(term or "")).strip()


# “某词 + 短修饰 + 言说动词 + 冒号/引号”才算可定位归属，例如
# 沈耽道： / 沈耽笑道：“ / 沈耽沉声道： / 沈耽说：“
# 不把句号算进来，避免“沈耽知道。”里的“道”被当成言说动词。
_SPEECH_TAIL_RE = re.compile(
    r"^[^。！？\n“”\"「」『』]{0,8}?(?:" + SPEECH_ALT + r")(?=[：:。“”\"「」『』])"
)


def _speech_after(text: str, pos: int) -> bool:
    """从 pos 开始看，是否马上出现“言说动词 + 冒号/引号”。"""
    return bool(_SPEECH_TAIL_RE.match(text[pos:]))


def _has_speech_evidence(text: str, term: str) -> bool:
    """text 里是否存在 term 的任一次出现，紧跟言说动词并带冒号/引号。"""
    start = 0
    while True:
        pos = text.find(term, start)
        if pos < 0:
            return False
        if _speech_after(text, pos + len(term)):
            return True
        start = pos + 1


def _evidence_ok(term: str, quote: str, text: str) -> bool:
    """机械校验 LLM 给的证据片段：逐字存在、含该词、含言说动词或称呼语。"""
    if not isinstance(quote, str):
        return False
    q = quote.strip()
    if not q or term not in q:
        return False
    if locate_verbatim(text, q) < 0:
        return False
    if _has_speech_evidence(q, term):
        return True
    # 称呼语：引号里的“某词”紧跟呼语停顿，如“沈兄，你听我说”
    if re.search(r"[“\"「『]" + re.escape(term) + r"[，,！!？?、：:]", q):
        return True
    return False


def _cache_key(model: str, messages: list[dict], schema: dict) -> str:
    payload = {
        "version": CACHE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "schema_name": SCHEMA_NAME,
        "schema": schema,
        "messages": messages,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_FAIL_SENTINEL = {"__name_vote_failed__": True}


def _cache_get(key: str):
    if not _CACHE_ENABLED:
        return None
    path = CACHE_DIR / key[:2] / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8")).get("data")
        # 旧版本留下的负缓存一律视为未命中：超时/schema 失败是可重试的，
        # 不能让一次临时失败永久冻结该窗口。
        return None if data == _FAIL_SENTINEL else data
    except Exception:
        return None


def _cache_put(key: str, data) -> None:
    if not _CACHE_ENABLED:
        return
    try:
        folder = CACHE_DIR / key[:2]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{key}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"data": data}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


# --------------------------------------------------------------------------- 切窗
def quote_quality(text: str, style: str) -> tuple[str, list[str]]:
    """检测引号缺失/错配/混用；返回 (ok|degraded, issues)。"""
    op, cl = QUOTE_STYLES[style]
    n_open, n_close = text.count(op), text.count(cl)
    issues: list[str] = []
    if n_open != n_close:
        issues.append("unbalanced")
    if n_open and not n_close:
        issues.append("unclosed")
    if n_close > n_open:
        issues.append("orphan_close")
    if style == "cn" and ('"' in text or "'" in text):
        issues.append("ascii_mixed")
    if style == "ascii" and ("“" in text or "「" in text or "『" in text):
        issues.append("fullwidth_mixed")
    if re.search(r"(?:" + SPEECH_ALT + r")[：:]", text) and n_open == 0:
        issues.append("unquoted_dialogue")
    if n_open >= 4 and n_open * 100 / max(len(text), 1) > 12:
        issues.append("dense")
    return ("degraded" if issues else "ok"), issues


def build_units(chapters, style: str, window_chars: int = DEFAULT_WINDOW_CHARS) -> list[dict]:
    """同章内按段落切窗，只保留含对白的窗口；目标段固定。"""
    units: list[dict] = []
    for ch in chapters:
        paras = ch["paras"]
        start = None
        size = 0
        for pi, text in enumerate(paras):
            if not text:
                if start is not None and size > 0:
                    _flush_unit(units, ch, paras, start, pi, style)
                    start, size = None, 0
                continue
            if start is None:
                start, size = pi, 0
            size += len(text) + 1
            if size >= window_chars:
                _flush_unit(units, ch, paras, start, pi + 1, style)
                start, size = None, 0
        if start is not None:
            _flush_unit(units, ch, paras, start, len(paras), style)
    return units


def _flush_unit(units, ch, paras, start, end, style) -> None:
    texts = [t for t in paras[start:end] if t]
    if not texts:
        return
    body = "\n".join(texts)
    if not any(quote_spans(t, style) for t in texts) and \
            not re.search(r"(?:" + SPEECH_ALT + r")[：:]", body):
        return
    quality, issues = quote_quality(body, style)
    units.append({
        "uid": f"ch{ch['idx']:04d}u{len(units):05d}",
        "chapter": ch["idx"],
        "title": ch.get("title", ""),
        "pi_start": start,
        "pi_end": end,
        "text": body,
        "chars": len(body),
        "seg_quality": quality,
        "issues": issues,
    })


# --------------------------------------------------------------------------- 调用与解析
def _batch_messages(batch: list[dict], degraded: bool) -> list[dict]:
    warn = ("注意：部分窗口引号可能缺失、错配或中英混用；对白可能没有引号，"
            "请结合段落边界与言说动词判断。") if degraded else ""
    parts = []
    for i, unit in enumerate(batch, 1):
        parts.append(f"【窗口 W{i}】\n{unit['text']}")
    user = (
        f"{warn}\n"
        "下面有多个相互独立的窗口。请**逐个窗口**列出有名字的说话人，"
        "以及看起来像人名但实际不是人物指称的词。\n"
        "无名角色（如“一名士兵”“那汉子”“车夫”）不要列进 speakers；"
        "只有职务/身份/群体的词放进 non_persons。\n"
        "名字必须逐字出现在该窗口内，不得跨窗口引用。\n"
        "results 必须按窗口顺序返回，window_id 填 W1、W2 …，一个窗口一条记录。\n\n"
        + "\n\n".join(parts)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _evidence_messages(batch: list[dict]) -> list[dict]:
    """batch 元素：{key, text, term}。一次校验多个“窗口+待查词”。"""
    parts = []
    for i, item in enumerate(batch, 1):
        parts.append(f"【待查 E{i}】词：{item['term']}\n窗口原文：\n{item['text']}")
    user = (
        "下面每个待查项都是独立判断，词与窗口一一对应。"
        "请逐个判断该词在该窗口里有没有可定位的言说证据，并逐字抄回片段。\n"
        "results 按 E1、E2 … 顺序返回，每项一条记录。\n\n"
        + "\n\n".join(parts)
    )
    return [
        {"role": "system", "content": EVIDENCE_PROMPT},
        {"role": "user", "content": user},
    ]


class ModelRunner:
    """固定模型的批量调用 + 缓存。失败写负缓存，保证重跑结果一致。"""

    def __init__(self, client, pool: list[str], timeout_s: float = 120.0):
        self.client = client
        self.pool = list(pool)
        self.timeout_s = timeout_s
        self.failures: Counter = Counter()
        self.calls = 0
        self.cache_hits = 0
        self._lock = threading.Lock()

    def ask_batch(self, model: str, batch: list[dict]) -> dict[str, dict]:
        """一次请求判多个窗口；返回 {uid: answer}，缺的窗口不回填。"""
        degraded = any(u["seg_quality"] == "degraded" for u in batch)
        messages = _batch_messages(batch, degraded)
        key = _cache_key(model, messages, BATCH_SCHEMA)
        cached = _cache_get(key)
        if cached is not None:
            with self._lock:
                self.cache_hits += 1
            data = cached
        else:
            data = self._invoke(model, messages, BATCH_SCHEMA, "name_vote_batch", key)
        out: dict[str, dict] = {}
        if not isinstance(data, dict):
            return out
        results = data.get("results")
        if not isinstance(results, list):
            return out
        for item in results:
            if not isinstance(item, dict):
                continue
            wid = str(item.get("window_id") or "").strip().upper().lstrip("W")
            if not wid.isdigit():
                continue
            idx = int(wid) - 1
            if 0 <= idx < len(batch):
                out[batch[idx]["uid"]] = item
        return out

    def ask_evidence(self, model: str, batch: list[dict]) -> dict[str, dict]:
        """一次校验多个“窗口+待查词”；返回 {key: answer}。"""
        messages = _evidence_messages(batch)
        key = _cache_key(model, messages, EVIDENCE_SCHEMA)
        cached = _cache_get(key)
        if cached is not None:
            with self._lock:
                self.cache_hits += 1
            data = cached
        else:
            data = self._invoke(model, messages, EVIDENCE_SCHEMA,
                                "name_vote_evidence", key)
        out: dict[str, dict] = {}
        if not isinstance(data, dict):
            return out
        results = data.get("results")
        if not isinstance(results, list):
            return out
        for item in results:
            if not isinstance(item, dict):
                continue
            wid = str(item.get("window_id") or "").strip().upper().lstrip("E")
            if not wid.isdigit():
                continue
            idx = int(wid) - 1
            if 0 <= idx < len(batch):
                out[batch[idx]["key"]] = item
        return out

    def _invoke(self, model: str, messages: list[dict], schema: dict,
                schema_name: str, key: str) -> dict | None:
        from core.llm import classify_error

        with self._lock:
            self.calls += 1
        for attempt in range(RATE_LIMIT_RETRIES):
            try:
                data = self.client._try_model(
                    model, messages, 0.0, schema, schema_name,
                    time.monotonic() + self.timeout_s, None)
                break
            except Exception as err:  # noqa: BLE001 - 单模型失败不应中断整本书
                text = str(err)
                kind = classify_error(err)
                if kind == "rate_limit" and attempt + 1 < RATE_LIMIT_RETRIES:
                    time.sleep(RATE_LIMIT_BACKOFF * (2 ** attempt))
                    continue
                # 余额/权限问题不是“这个窗口判不出”，继续跑只会把整本书写进负缓存
                if kind == "auth" or "402" in text or "insufficient" in text.lower():
                    raise RuntimeError(
                        f"LLM 账户余额或权限不足，已停止人名投票（{model}）：{text[:160]}"
                    ) from err
                with self._lock:
                    self.failures[model] += 1
                # 超时/限流/schema 失败都是可重试的，不写负缓存；
                # 否则一次临时失败会在后续所有运行里被永久冻结。
                return None
        else:
            with self._lock:
                self.failures[model] += 1
            return None
        if not isinstance(data, dict):
            with self._lock:
                self.failures[model] += 1
            return None
        _cache_put(key, data)
        return data


TITLE_PREFIX = (
    "大将军", "将军", "公子", "姑娘", "大人", "先生", "夫人", "小姐",
    "掌柜", "师父", "师傅", "老爷", "少爷", "老板", "队长", "校尉",
)

GROUP_TERMS = {
    "声音", "众人", "大家", "所有人", "人群", "士兵", "孩子", "孩子们",
    "最大的孩子", "其他孩子", "其他的人", "几个人", "旁人", "来人",
}


def _normalize_name(raw: Any) -> str:
    """去空白、去前置敬称；群体词/代词/泛指不算人名。"""
    name = _norm(raw)
    name = name.strip("“”\"'「」『』（）()【】[]，。！？；：、 ")
    if not name or name in GROUP_TERMS or name.endswith("们"):
        return ""
    for pre in TITLE_PREFIX:
        if name.startswith(pre) and len(name) - len(pre) >= 2:
            name = name[len(pre):]
            break
    return name


def parse_answer(data: dict, context: str) -> dict:
    """把一次模型输出规范化成 {term: {"label": name|non_name, "evidence": [...]}}。"""
    out: dict[str, dict] = {}
    for raw in data.get("speakers") or []:
        name = _normalize_name(raw)
        if not _valid_term(name) or locate_verbatim(context, name) < 0:
            continue
        out[name] = {"label": "name", "evidence": [name], "aliases": []}
    for raw in data.get("non_persons") or []:
        term = _normalize_name(raw)
        if not _valid_term(term) or locate_verbatim(context, term) < 0:
            continue
        if term in out:
            continue  # 同一窗口既列为说话者又列为非人物，按“说话者”优先
        out[term] = {"label": "non_name", "evidence": [term], "aliases": []}
    return out


def _unique_max(labels: list[str]) -> tuple[str | None, Counter]:
    counts = Counter(labels)
    if not counts:
        return None, counts
    top = counts.most_common()
    if len(top) == 1:
        return top[0][0], counts
    if top[0][1] > top[1][1]:
        return top[0][0], counts
    return None, counts


def _decided(labels: list[str], voters: set[str]) -> bool:
    """唯一最高票 + 至少两个不同模型明确表态，才算初筛出结果。

    只有一个模型提到该词（另一个弃权）不算“两个模型一致”，继续加票。
    """
    return len(voters) >= 2 and _unique_max(labels)[0] is not None


# --------------------------------------------------------------------------- 主流程
def mine_names_voted(book: dict, *, models: list[str] | None = None,
                     window_chars: int = DEFAULT_WINDOW_CHARS,
                     concurrency: int = 6,
                     client=None, timeout_s: float | None = None) -> dict:
    """LLM 投票式人名挖掘；返回与 ``mine_names`` 兼容的结构。"""
    if timeout_s is None:
        timeout_s = float(os.environ.get("NOVEL_FORGE_NAME_TIMEOUT", "60"))
    chapters = book["chapters"]
    style = book["primary_style"]
    pool = [m for m in (models or DEFAULT_MODELS)]
    if client is not None:
        charset = set(getattr(client, "models", []) or [])
        pool = [m for m in pool if not charset or m in charset]
    if len(pool) < 2:
        raise RuntimeError("可用的不同模型少于 2 个，无法投票")

    runner = ModelRunner(client, pool, timeout_s=timeout_s)
    units = build_units(chapters, style, window_chars)
    unit_by_uid = {u["uid"]: u for u in units}

    state = {
        u["uid"]: {"votes": defaultdict(list), "voters": defaultdict(set),
                   "aliases": defaultdict(list), "got": 0,
                   "ev_tried": defaultdict(set), "ev_resolved": {}}
        for u in units
    }
    settle_votes: dict[tuple[str, str], tuple[int, int]] = {}
    models_used: Counter = Counter()

    def absorb(batch: list[dict], replies: dict[str, dict]) -> None:
        """把一次批量回复并入共享状态；只在主线程调用，保证顺序确定。"""
        for model, reply in replies.items():
            if not reply:
                continue
            models_used[model] += 1
            for unit in batch:
                item = reply.get(unit["uid"])
                if not item:
                    continue
                st = state[unit["uid"]]
                st["got"] += 1
                # 逐字校验永远用目标段：上下文里的名字不能算进本窗口
                parsed = parse_answer(item, unit["text"])
                for term, rec in parsed.items():
                    st["votes"][term].append(rec["label"])
                    st["voters"][term].add(model)
                    st["aliases"][term].extend(rec.get("aliases", []))

    def run_round(pending: list[dict], models: list[str], batch_size: int) -> None:
        if not pending:
            return
        batches = [pending[i:i + batch_size]
                   for i in range(0, len(pending), batch_size)]
        workers = max(1, min(concurrency, len(batches)))

        def work(batch):
            out = {m: runner.ask_batch(m, batch) for m in models}
            # 免费模型偶发超时/schema 失败：同一批缺哪几个窗口就立刻补一次，
            # 避免把一次临时失败记成“该模型弃权”。
            for m in models:
                missing = [u for u in batch if not out[m].get(u["uid"])]
                if missing:
                    again = runner.ask_batch(m, missing)
                    if again:
                        out[m].update(again)
            return out

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for batch, replies in zip(batches, ex.map(work, batches)):
                absorb(batch, replies)

    def term_done(uid: str, term: str) -> bool:
        st = state[uid]
        if term in st["ev_resolved"]:
            return True
        return _decided(st["votes"][term], st["voters"][term])

    def term_conflict(uid: str, term: str) -> bool:
        """两个模型明确给出了相反标签（name vs non_name）。"""
        st = state[uid]
        if term in st["ev_resolved"]:
            return False
        labels = st["votes"][term]
        return len(set(labels)) > 1 and not term_done(uid, term)

    def unit_done(uid: str) -> bool:
        st = state[uid]
        return st["got"] > 0 and all(term_done(uid, t) for t in st["votes"])

    def conflicts() -> list[tuple[str, str]]:
        """当前所有未解决的冲突 (uid, term)，按窗口顺序。"""
        out = []
        for u in units:
            for term in sorted(state[u["uid"]]["votes"]):
                if term_conflict(u["uid"], term):
                    out.append((u["uid"], term))
        return out

    def snapshot() -> None:
        """记录每个词第一次达到“两模型唯一最高”时的票数与表态模型数。"""
        for u in units:
            st = state[u["uid"]]
            for term in st["votes"]:
                key = (u["uid"], term)
                if key in settle_votes or term in st["ev_resolved"]:
                    continue
                if _decided(st["votes"][term], st["voters"][term]):
                    settle_votes[key] = (
                        len(st["voters"][term]),
                        len(st["votes"][term]),
                        _unique_max(st["votes"][term])[0],
                    )

    # ---- 第一轮：两个不同模型，全量窗口
    run_round(units, [pool[0], pool[1]], BATCH_SIZE)
    snapshot()
    settled_after_1 = sum(1 for u in units if unit_done(u["uid"]))

    # ---- 冲突 → 换判据：找可定位言说证据，不扩窗
    def run_evidence(pairs: list[tuple[str, str]]) -> None:
        """冲突词最多换 MAX_EVIDENCE_ATTEMPTS 个模型找证据；拿到即判角色。

        三次都拿不出可定位证据 → 直接判非角色，不再往池子里加模型。
        """
        pending = [(uid, term) for uid, term in pairs
                   if term not in state[uid]["ev_resolved"]]
        if not pending:
            return
        for _ in range(MAX_EVIDENCE_ATTEMPTS):
            if not pending:
                break
            # 质量兜底优先：证据轮是最终判定，先用 glm-4.5-flash 这类质量模型；
            # 同一词不重复试同一个模型，试过两票仍无证据才按非角色收口。
            ordered = list(reversed(pool))
            groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for uid, term in pending:
                tried = state[uid]["ev_tried"][term]
                pick = next((m for m in ordered if m not in tried), None)
                if pick is None:
                    continue
                tried.add(pick)
                groups[pick].append((uid, term))
            if not groups:
                break
            for model, todo in groups.items():
                items = [{"key": f"{uid}\x1f{term}", "uid": uid, "term": term,
                          "text": unit_by_uid[uid]["text"]} for uid, term in todo]
                batches = [items[i:i + EVIDENCE_BATCH]
                           for i in range(0, len(items), EVIDENCE_BATCH)]
                workers = max(1, min(concurrency, len(batches)))

                def work(batch):
                    return runner.ask_evidence(model, batch)

                with ThreadPoolExecutor(max_workers=workers) as ex:
                    for batch, reply in zip(batches, ex.map(work, batches)):
                        for it in batch:
                            uid, term = it["uid"], it["term"]
                            item = reply.get(it["key"]) or {}
                            quote = str(item.get("evidence") or "")
                            ok = bool(item.get("has_evidence")) and \
                                _evidence_ok(term, quote, it["text"])
                            st = state[uid]
                            if ok:
                                st["ev_resolved"][term] = {
                                    "result": "name", "quote": quote,
                                    "model": model,
                                    "attempts": len(st["ev_tried"][term]),
                                }
                            elif len(st["ev_tried"][term]) >= MAX_EVIDENCE_ATTEMPTS:
                                st["ev_resolved"][term] = {
                                    "result": "non_name", "quote": "",
                                    "model": model,
                                    "attempts": len(st["ev_tried"][term]),
                                }
            pending = [(uid, term) for uid, term in pending
                       if term not in state[uid]["ev_resolved"]]
        # 池里没有更多模型可试时，剩余冲突按非角色收口
        for uid, term in pending:
            st = state[uid]
            st["ev_resolved"][term] = {
                "result": "non_name", "quote": "", "model": "",
                "attempts": len(st["ev_tried"][term]),
            }

    run_evidence(conflicts())

    # ---- 仲裁：无冲突但只有一个模型表态，依次加入池内其它模型加票；
    #      新出现的冲突立即转证据轮，不再继续加票。
    arbitration_units = 0
    for model in pool[2:]:
        pending = [u for u in units
                   if state[u["uid"]]["got"] > 0
                   and any(not term_done(u["uid"], t)
                           for t in state[u["uid"]]["votes"])]
        if not pending:
            break
        if arbitration_units == 0:
            arbitration_units = len(pending)
        run_round(pending, [model], ARB_BATCH)
        snapshot()
        new_pairs = conflicts()
        if new_pairs:
            run_evidence(new_pairs)

    evidence_counts: Counter = Counter()
    evidence_attempts: Counter = Counter()
    for st in state.values():
        for rec in st["ev_resolved"].values():
            evidence_counts[rec["result"]] += 1
            evidence_attempts[rec["attempts"]] += 1

    rows: list[dict] = []
    term_units: dict[str, Counter] = defaultdict(Counter)
    alias_votes: dict[str, Counter] = defaultdict(Counter)
    for u in units:
        st = state[u["uid"]]
        degraded = u["seg_quality"] == "degraded"
        for term in sorted(st["votes"]):
            labels = st["votes"][term]
            decision, counts = _unique_max(labels)
            if not _decided(labels, st["voters"][term]):
                decision = "uncertain"
            ev = st["ev_resolved"].get(term)
            if ev:
                decision = ev["result"]
            # 票池只有两个真正不同的模型时，1-1 分裂拿不到第三票。
            # 这里只记录“分歧确实发生且无法再加人”，不拿同一模型重复调用冒充第二票。
            split_no_third = bool(len(pool) < 3 and len(set(labels)) > 1)
            term_units[term][decision] += 1
            rows.append({
                "term": term, "decision": decision, "votes": dict(counts),
                "votes_total": len(labels), "voters": len(st["voters"][term]),
                "unit": u["uid"], "chapter": u["chapter"], "degraded": degraded,
                "evidence": (ev or {}).get("quote", ""),
                "insufficient_voters": split_no_third,
            })
            for al in set(st["aliases"].get(term, [])):
                alias_votes[al][term] += 1

    stats = {
        "units": len(units),
        "degraded_units": sum(1 for u in units if u["seg_quality"] == "degraded"),
        "decisions": len(rows),
        "decided_at_votes": Counter(v[0] for v in settle_votes.values()),
        "decided_at_ballots": Counter(v[1] for v in settle_votes.values()),
        # 仅两个初始模型一致（同标签、恰好 2 个表态模型）即采纳的词。
        # 这是“一致但可能错”的分母，用于和 1.1 规则层交叉核对。
        "two_model_name_terms": sorted({
            t for (_, t), (nv, _, lab) in settle_votes.items()
            if nv == 2 and lab == "name"}),
        "two_model_non_name_terms": sorted({
            t for (_, t), (nv, _, lab) in settle_votes.items()
            if nv == 2 and lab == "non_name"}),
        "stage1_units": settled_after_1,
        # 冲突换判据：证据轮通过/失败，以及各用了几次
        "evidence_name": evidence_counts["name"],
        "evidence_non_name": evidence_counts["non_name"],
        "evidence_attempts": dict(sorted(evidence_attempts.items())),
        "arbitration_units": arbitration_units,
        "pool_exhausted": sum(1 for u in units if state[u["uid"]]["got"] > 0
                              and not unit_done(u["uid"])),
        "unresolved": sum(1 for r in rows if r["decision"] == "uncertain"),
        "unresolved_at_votes": Counter(
            len(state[u["uid"]]["voters"][t])
            for u in units for t in state[u["uid"]]["votes"]
            if not term_done(u["uid"], t)),
        # 两票池下 1-1 分裂、又没有第三个不同模型可加：单独统计，不重复调用冒充第二票
        "insufficient_voters": sum(1 for r in rows if r["insufficient_voters"]),
        "insufficient_voters_terms": sorted({r["term"] for r in rows
                                             if r["insufficient_voters"]}),
        "insufficient_voters_resolved": sum(
            1 for r in rows if r["insufficient_voters"]
            and r["decision"] in ("name", "non_name")),
        "insufficient_voters_unresolved": sum(
            1 for r in rows if r["insufficient_voters"]
            and r["decision"] not in ("name", "non_name")),
        "failed_units": sum(1 for u in units if state[u["uid"]]["got"] == 0),
        "models_used": models_used,
    }
    stats["model_failures"] = Counter(runner.failures)
    stats["cache_hits"] = runner.cache_hits
    stats["llm_calls"] = runner.calls
    return _assemble(book, units, rows, term_units,
                     alias_votes, stats, pool)


def _scan_counts(chapters, terms: list[str]) -> tuple[dict[str, int], dict[str, list[int]]]:
    if not terms:
        return {}, {}
    pat = re.compile("|".join(re.escape(t) for t in sorted(terms, key=lambda x: (-len(x), x))))
    counts: Counter = Counter()
    chapters_seen: dict[str, set] = defaultdict(set)
    for ch in chapters:
        idx = ch["idx"]
        for text in ch["paras"]:
            if not text:
                continue
            for m in pat.finditer(text):
                counts[m.group(0)] += 1
                chapters_seen[m.group(0)].add(idx)
    return dict(counts), {k: sorted(v) for k, v in chapters_seen.items()}


def _scan_evidence(chapters, terms: list[str]) -> dict[str, bool]:
    """全书一次扫描：每个词是否存在可定位的言说/引号归属证据。

    只做“词后紧跟言说动词+冒号/引号”的机械匹配，不判断词本身像不像人名。
    """
    if not terms:
        return {}
    pat = re.compile("|".join(re.escape(t) for t in sorted(terms, key=lambda x: (-len(x), x))))
    found: dict[str, bool] = {t: False for t in terms}
    remaining = set(terms)
    for ch in chapters:
        if not remaining:
            break
        for text in ch["paras"]:
            if not text or not remaining:
                continue
            for m in pat.finditer(text):
                term = m.group(0)
                if term in remaining and _speech_after(text, m.end()):
                    found[term] = True
                    remaining.discard(term)
                    if not remaining:
                        break
    return found


def _assemble(book, units, rows, term_units, alias_votes,
              stats, pool) -> dict:
    chapters = book["chapters"]
    style = book["primary_style"]
    quote_total = sum(len(quote_spans(t, style)) for ch in chapters
                      for t in ch["paras"] if t)
    min_count = max(3, int(quote_total * 0.00015))
    terms = sorted(term_units)
    counts, chap_map = _scan_counts(chapters, terms)
    has_evidence = _scan_evidence(chapters, terms)

    # 词级汇总：name 单元 > non_name 单元才进角色位；平票或都 uncertain 的记为
    # unresolved（保留为 candidate，不生成角色）。
    # 采纳前过机械验收：判角色但全书无“某道/某说/某问”证据 → 降为泛称；
    # 判非角色但全书存在可定位言说证据 → 翻案转角色。
    role: list[str] = []
    candidates: list[dict] = []
    unresolved_terms: list[str] = []
    rejected_terms: list[str] = []
    demoted_terms: list[str] = []
    promoted_terms: list[str] = []
    for term in terms:
        c = term_units[term]
        n_name, n_non = c["name"], c["non_name"]
        if counts.get(term, 0) < 2:
            continue
        ev = has_evidence.get(term, False)
        if n_name > n_non and n_name >= 1:
            if ev:
                role.append(term)
            else:
                demoted_terms.append(term)
                candidates.append({
                    "term": term, "count": counts.get(term, 0),
                    "chapters": len(chap_map.get(term, [])),
                    "evidence": {"tag": 0, "modifier": 0, "subject": 0, "vocative": 0,
                                 "identity": False, "degree": 0, "adverbial": 0},
                    "reason": f"机械验收：全书无可定位言说证据，降为泛称"
                              f"（name {n_name} / non_name {n_non}）",
                })
        else:
            if n_non > n_name:
                if ev:
                    # 全书有可定位“某道/某说/某问”证据，必须翻案
                    promoted_terms.append(term)
                    role.append(term)
                else:
                    rejected_terms.append(term)
                    candidates.append({
                        "term": term, "count": counts.get(term, 0),
                        "chapters": len(chap_map.get(term, [])),
                        "evidence": {"tag": 0, "modifier": 0, "subject": 0,
                                     "vocative": 0, "identity": False,
                                     "degree": 0, "adverbial": 0},
                        "reason": f"LLM 判定非人物（name {n_name} / non_name {n_non}）",
                    })
            else:
                unresolved_terms.append(term)
                candidates.append({
                    "term": term, "count": counts.get(term, 0),
                    "chapters": len(chap_map.get(term, [])),
                    "evidence": {"tag": 0, "modifier": 0, "subject": 0,
                                 "vocative": 0, "identity": False,
                                 "degree": 0, "adverbial": 0},
                    "reason": f"LLM 票数无唯一多数（name {n_name} / non_name {n_non}）",
                })

    # 别名：同一 alias→canonical 至少 2 票且唯一最高才合并
    alias: dict[str, str] = {}
    role_set = set(role)
    for al, cnt in sorted(alias_votes.items()):
        if al not in role_set:
            continue
        canon, tally = _unique_max([k for k, v in cnt.items() for _ in range(v)])
        if canon and canon in role_set and canon != al and tally[canon] >= 2 \
                and list(tally.values()).count(tally[canon]) == 1:
            alias[al] = canon

    def canon(n: str) -> str:
        seen = set()
        while n in alias and n not in seen:
            seen.add(n)
            n = alias[n]
        return n

    gaz: dict[str, dict] = {}
    role_ids = {canon(t) for t in role}
    for term in terms:
        cnt = counts.get(term, 0)
        if cnt < min_count and term not in role:
            continue
        gaz[term] = {
            "count": cnt,
            "canonical": canon(term),
            "chapters": chap_map.get(term, []),
        }
        if canon(term) not in role_ids:
            gaz[term]["candidate"] = True

    merges = [{"from": a, "to": b, "shared": "", "counts": [counts.get(a, 0), counts.get(b, 0)],
               "chapter_overlap": 0.0}
              for a, b in sorted(alias.items())]

    total_decisions = stats["decisions"] or 1
    stats_out = {
        "prompt_version": PROMPT_VERSION,
        "models": pool,
        "units": stats["units"],
        "degraded_units": stats["degraded_units"],
        "failed_units": stats["failed_units"],
        "decisions": stats["decisions"],
        "decisions_at_votes": dict(sorted(stats["decided_at_votes"].items())),
        "decisions_at_ballots": dict(sorted(stats["decided_at_ballots"].items())),
        "stage1_units": stats["stage1_units"],
        "stage1_rate": round(stats["stage1_units"] / max(stats["units"], 1), 4),
        "evidence_name_units": stats["evidence_name"],
        "evidence_non_name_units": stats["evidence_non_name"],
        "evidence_attempts": stats["evidence_attempts"],
        "arbitration_units": stats["arbitration_units"],
        "arbitration_rate": round(stats["arbitration_units"] / max(stats["units"], 1), 4),
        "pool_exhausted_units": stats["pool_exhausted"],
        "unresolved_decisions": stats["unresolved"],
        "unresolved_at_votes": dict(sorted(stats["unresolved_at_votes"].items())),
        "unresolved_rate": round(stats["unresolved"] / total_decisions, 4),
        "unresolved_terms": len(unresolved_terms),
        "rejected_terms": len(rejected_terms),
        # 两票池下拿不到第三票的分歧：单列，避免和“模型池耗尽”混在一起
        "insufficient_voters_decisions": stats["insufficient_voters"],
        "insufficient_voters_rate": round(stats["insufficient_voters"] / total_decisions, 4),
        "insufficient_voters_terms": stats["insufficient_voters_terms"],
        "insufficient_voters_resolved": stats["insufficient_voters_resolved"],
        "insufficient_voters_unresolved": stats["insufficient_voters_unresolved"],
        "acceptance_demoted": sorted(demoted_terms),
        "acceptance_promoted": sorted(promoted_terms),
        "acceptance_demoted_terms": len(demoted_terms),
        "acceptance_promoted_terms": len(promoted_terms),
        "batches": (stats["units"] + BATCH_SIZE - 1) // BATCH_SIZE,
        "batch_size": BATCH_SIZE,
        "models_used": dict(sorted(stats["models_used"].items())),
        "role_terms": len(role_ids),
        "two_model_name_terms": stats["two_model_name_terms"],
        "two_model_non_name_terms": stats["two_model_non_name_terms"],
        # 两模型一致但错的直接度量：
        # - 两票都说 name、全书却无任何可定位言说证据 → 误入
        # - 两票都说 non_name、全书却有可定位言说证据 → 误删
        "two_model_name_wrong": sorted(
            t for t in stats["two_model_name_terms"] if not has_evidence.get(t, False)),
        "two_model_non_name_wrong": sorted(
            t for t in stats["two_model_non_name_terms"] if has_evidence.get(t, False)),
    }
    consistent_total = (len(stats["two_model_name_terms"])
                        + len(stats["two_model_non_name_terms"]))
    consistent_wrong = (len(stats_out["two_model_name_wrong"])
                        + len(stats_out["two_model_non_name_wrong"]))
    stats_out["two_model_consistent_decisions"] = consistent_total
    stats_out["two_model_consistent_wrong"] = consistent_wrong
    stats_out["two_model_consistent_wrong_rate"] = round(
        consistent_wrong / consistent_total, 4) if consistent_total else 0.0
    runtime = {
        "llm_calls": stats["llm_calls"],
        "cache_hits": stats["cache_hits"],
        "model_failures": dict(sorted(stats["model_failures"].items())),
    }
    return {
        "gazetteer": gaz,
        "merges": merges,
        "candidate_terms": candidates,
        "quote_total": quote_total,
        "raw_candidates": len(terms),
        "kept": len(gaz),
        "name_mining": "llm_vote",
        "vote_stats": stats_out,
        "name_runtime": runtime,
        "vote_rows": rows,
        "unresolved_terms": sorted(unresolved_terms),
        "degraded_units": [{"uid": u["uid"], "chapter": u["chapter"],
                            "issues": u["issues"]}
                           for u in units if u["seg_quality"] == "degraded"],
    }
