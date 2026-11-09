"""只读依赖检查：对照 requirements.txt 输出已安装/最新版本，不执行任何升级。

用法：
  python tools/deps_check.py                 # 只查本地安装
  python tools/deps_check.py --online        # 额外查询 PyPI 最新版（只读网络请求）
  python tools/deps_check.py --json          # 机器可读输出
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_requirements(path: Path) -> list[str]:
    names: list[str] = []
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~;\[\s]", line, maxsplit=1)[0].strip()
        if name and name not in names:
            names.append(name)
    return names


def installed_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def latest_version(name: str, timeout: float = 8.0) -> str:
    url = f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str((payload.get("info") or {}).get("version") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description="只读依赖检查（不会升级）")
    parser.add_argument("--online", action="store_true", help="查询 PyPI 最新版本")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    parser.add_argument("--requirements", default=str(ROOT / "requirements.txt"))
    args = parser.parse_args()

    rows = []
    for name in parse_requirements(Path(args.requirements)):
        current = installed_version(name)
        latest = ""
        if args.online:
            try:
                latest = latest_version(name)
            except Exception as err:  # noqa: BLE001
                latest = f"查询失败：{err}"
        rows.append({"name": name, "installed": current or "(未安装)", "latest": latest or "—",
                     "update_available": bool(latest and current and latest != current)})

    if args.json:
        print(json.dumps({"requirements": args.requirements, "online": args.online,
                          "packages": rows}, ensure_ascii=False, indent=2))
        return 0
    print(f"{'包':<28}{'已安装':<16}{'最新':<16}可升级")
    print("-" * 68)
    for row in rows:
        print(f"{row['name']:<28}{row['installed']:<16}{row['latest']:<16}"
              f"{'是' if row['update_available'] else '否'}")
    print("\n提示：本脚本只读；升级请按 docs/deps-update.md / docs/edge-tts-update.md 手动执行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
