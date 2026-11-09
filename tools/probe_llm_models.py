#!/usr/bin/env python3
"""逐个探测候选 LLM：是否存在、是否可用、延迟、是否思考模型、JSON 是否合法。

只读：不发业务请求，不写产物；用于选路前的实测确认。

用法：
    python tools/probe_llm_models.py
    python tools/probe_llm_models.py --models glm-4-flash glm-4.5-flash
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from openai import OpenAI  # noqa: E402

from core.llm import extract_json, load_env, validate_schema  # noqa: E402
from core.script import SYSTEM_PROMPT  # noqa: E402
from core.segments import SEGMENTS_SCHEMA  # noqa: E402

DEFAULT_CANDIDATES = [
    # 2026-11-06：硅基流动整账户 402，候选表切到智谱可用免费型号。
    "glm-4-flash",
    "glm-4.5-flash",
]

SCHEMA = SEGMENTS_SCHEMA

PROMPT = """角色名单：他、老掌柜
原文第 1/1 块（请完整切分）：
他推门进来，低声道：“事情办妥了。”
老掌柜抬眼看了看他，没有说话，只是把手里的账册合上。"""


def main() -> int:
    parser = argparse.ArgumentParser(description="探测候选模型的可用性/延迟/是否思考模型")
    parser.add_argument("--models", nargs="*", default=None, help="候选模型 ID 列表")
    parser.add_argument("--provider", default="zhipu", choices=["zhipu"],
                        help="从 config.llm.providers 读取对应 key/base_url")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", default="", help="可选：结果 JSON 落盘路径")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    load_env((ROOT / cfg["paths"]["env_file"]).resolve())
    spec = ((cfg["llm"].get("providers") or {}).get(args.provider) or {})
    if spec:
        key = os.environ[spec["api_key_env"]]
        base = os.environ.get(spec.get("base_url_env", ""), "") or spec["base_url"]
    else:
        key = os.environ[cfg["llm"]["api_key_env"]]
        base = os.environ.get(cfg["llm"].get("base_url_env", ""), cfg["llm"]["base_url"])
    client = OpenAI(api_key=key, base_url=base, timeout=args.timeout, max_retries=0)
    candidates_default = DEFAULT_CANDIDATES

    existing: set[str] = set()
    try:
        existing = {m.id for m in client.models.list().data}
        print(f"供应商 {args.provider} 可见模型 {len(existing)} 个")
    except Exception as err:
        print(f"[warn] /models 列举失败（该网关可能不支持）：{err}")

    candidates = args.models or candidates_default
    rows: list[dict] = []
    for model in candidates:
        row = {"model": model, "listed": model in existing, "ok": False, "status": "",
               "latency_s": 0.0, "thinking": None, "json_valid": False, "error": ""}
        if existing and not row["listed"]:
            row["status"] = "404(not listed)"
            rows.append(row)
            print(f"{model:<46} 不存在")
            continue
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": PROMPT}],
                temperature=0.0, max_tokens=512,
                extra_body={"thinking": {"type": "disabled"}},
                response_format={"type": "json_object"},
            )
            row["latency_s"] = round(time.time() - start, 2)
            message = resp.choices[0].message
            content = (message.content or "").strip()
            reasoning = getattr(message, "reasoning_content", None) or (
                (getattr(message, "model_extra", None) or {}).get("reasoning_content"))
            row["thinking"] = bool(reasoning)
            row["thinking_len"] = len(str(reasoning or ""))
            row["ok"] = True
            row["status"] = "200"
            try:
                data = extract_json(content or str(reasoning or ""))
                row["json_valid"] = validate_schema(data, SCHEMA) == ""
                row["segments"] = len(data.get("segments", [])) if isinstance(data, dict) else 0
            except Exception as err:
                row["json_error"] = str(err)[:120]
        except Exception as err:
            row["latency_s"] = round(time.time() - start, 2)
            status = getattr(err, "status_code", None)
            row["status"] = str(status or type(err).__name__)
            row["error"] = str(err)[:160].replace("\n", " ")
        rows.append(row)
        mark = "OK " if row["ok"] else "FAIL"
        thinking = {True: "思考", False: "非思考", None: "-"}[row["thinking"]]
        print(f"{mark} {model:<46} {row['status']:<16} {row['latency_s']:>7.2f}s "
              f"{thinking:<4} json={'Y' if row['json_valid'] else 'N'}"
              + (f" | {row['error']}" if row["error"] else ""))

    print("\n| 模型 | 存在 | 可用 | 状态码 | 延迟(s) | 思考模型 | JSON 合法 | 备注 |")
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        thinking = {True: "是", False: "否", None: "-"}[row["thinking"]]
        note = row["error"] or (f"segments={row.get('segments', 0)}" if row["ok"] else
                                ("未被账号模型列表收录" if not row["listed"] else ""))
        print(f"| `{row['model']}` | {'是' if row['listed'] else '否'} | "
              f"{'是' if row['ok'] else '否'} | {row['status']} | {row['latency_s']} | "
              f"{thinking} | {'是' if row['json_valid'] else '否'} | {note} |")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
