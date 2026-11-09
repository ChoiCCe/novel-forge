"""旁白参数定稿锁工具。

用法：
    python tools/narrator_lock.py --status   # 查看当前值与锁状态
    python tools/narrator_lock.py --lock     # 按当前配置写入 fingerprint + locked_values
    python tools/narrator_lock.py --unlock   # 解除锁定

只改 config.yaml -> casting.narrator 下的 locked / fingerprint / locked_values，
保留文件其余内容与原注释。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import narrator_lock  # noqa: E402

CONFIG = ROOT / "config.yaml"
_KEYS = ("locked", "fingerprint", "locked_values")


def load() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _rewrite(text: str, updates: dict) -> str:
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if re.match(r"^  narrator:\s*$", line))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"^  [A-Za-z_]", lines[i]):
            end = i
            break
    block = []
    skip_nested = False
    for line in lines[start + 1:end]:
        if re.match(r"^    (locked|fingerprint|locked_values):", line):
            skip_nested = "locked_values" in line
            continue
        if skip_nested:
            if re.match(r"^      ", line):
                continue
            skip_nested = False
        block.append(line)
    if block and not block[-1].endswith("\n"):
        block[-1] += "\n"
    block.append(f"    locked: {str(updates['locked']).lower()}\n")
    block.append(f'    fingerprint: "{updates["fingerprint"]}"\n')
    values = updates.get("locked_values") or {}
    if values:
        block.append("    locked_values:\n")
        for key in ("voice", "rate", "pitch", "volume"):
            block.append(f'      {key}: "{values.get(key, "")}"\n')
    else:
        block.append("    locked_values: {}\n")
    return "".join(lines[:start + 1] + block + lines[end:])


def write(updates: dict) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    CONFIG.write_text(_rewrite(text, updates), encoding="utf-8")


def show(cfg: dict) -> None:
    narrator = (cfg.get("casting") or {}).get("narrator") or {}
    current = narrator_lock.normalize(narrator_lock.effective(cfg))
    print("=== 旁白参数 ===")
    for key in narrator_lock.FIELDS:
        print(f"{key}: {current[key]}")
    print(f"fingerprint: {narrator_lock.fingerprint(current)}")
    print(f"locked: {str(bool(narrator.get('locked'))).lower()}")
    if narrator.get("locked"):
        diff = narrator_lock.diff(narrator.get("locked_values") or {}, current)
        if diff:
            print("与锁定值不一致：")
            for key, item in diff.items():
                print(f"  {key}: {item['locked']} -> {item['current']}")
        else:
            print("与锁定值一致")


def main() -> int:
    parser = argparse.ArgumentParser(description="旁白参数定稿锁")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="查看当前值与锁状态（默认）")
    group.add_argument("--lock", action="store_true", help="按当前配置锁定")
    group.add_argument("--unlock", action="store_true", help="解除锁定")
    args = parser.parse_args()
    cfg = load()
    if args.lock:
        values = narrator_lock.normalize(narrator_lock.effective(cfg))
        write({"locked": True,
               "fingerprint": narrator_lock.fingerprint(values),
               "locked_values": values})
        print("已锁定：", values)
        return 0
    if args.unlock:
        write({"locked": False, "fingerprint": "", "locked_values": {}})
        print("已解除锁定")
        return 0
    show(load())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
