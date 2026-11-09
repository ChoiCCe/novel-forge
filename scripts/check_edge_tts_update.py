"""只读检查 edge-tts 是否有新版本：不安装、不升级、不写远端。

输出：当前版本 / PyPI 最新 / GitHub 最新 release / 是否有更新 / 变更链接。
用法：
    python scripts/check_edge_tts_update.py
    python scripts/check_edge_tts_update.py --json
"""
from __future__ import annotations

import importlib.metadata as metadata
import json
import re
import sys
import urllib.error
import urllib.request

PYPI_URL = "https://pypi.org/pypi/edge-tts/json"
GITHUB_RELEASE_URL = "https://api.github.com/repos/rany2/edge-tts/releases/latest"
CHANGELOG_URL = "https://github.com/rany2/edge-tts/releases"
TIMEOUT_S = 15
PACKAGE = "edge-tts"


def installed_version() -> str:
    try:
        return metadata.version(PACKAGE)
    except metadata.PackageNotFoundError:
        return ""


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "novel-forge-edge-tts-check",
    })
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def _version_key(text: str) -> tuple:
    """把 7.3.1 / 7.3.0b1 粗略转成可比较元组，预发布版本排在正式版之后。"""
    if not text:
        return ()
    numeric = [int(part) for part in re.findall(r"\d+", text)]
    return tuple(numeric)


def _is_newer(latest: str, current: str) -> bool:
    if not latest:
        return False
    if not current:
        return True
    return _version_key(latest) > _version_key(current)


def collect() -> dict:
    current = installed_version()
    result = {
        "package": PACKAGE,
        "current": current,
        "latest_pypi": "",
        "latest_github": "",
        "has_update": False,
        "changelog_url": CHANGELOG_URL,
        "pypi_url": "https://pypi.org/project/edge-tts/",
        "errors": [],
    }
    try:
        payload = _fetch_json(PYPI_URL)
        result["latest_pypi"] = payload.get("info", {}).get("version", "")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
        result["errors"].append(f"PyPI 查询失败：{err}")
    try:
        payload = _fetch_json(GITHUB_RELEASE_URL)
        result["latest_github"] = (payload.get("tag_name") or "").lstrip("v")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
        result["errors"].append(f"GitHub 查询失败：{err}")
    newest = max((result["latest_pypi"], result["latest_github"]), key=_version_key)
    result["latest"] = newest
    result["has_update"] = _is_newer(newest, current)
    return result


def main(argv: list[str]) -> int:
    if "--json" in argv:
        print(json.dumps(collect(), ensure_ascii=False, indent=2))
        return 0

    data = collect()
    print("edge-tts 版本检查（只读，不会升级）")
    print(f"  当前版本     : {data['current'] or '未安装'}")
    print(f"  PyPI 最新    : {data['latest_pypi'] or '查询失败'}")
    print(f"  GitHub 最新  : {data['latest_github'] or '查询失败'}")
    print(f"  是否有更新   : {'是' if data['has_update'] else '否'}")
    print(f"  变更链接     : {data['changelog_url']}")
    print(f"  PyPI 项目页  : {data['pypi_url']}")
    for message in data["errors"]:
        print(f"  [警告] {message}")
    if data["has_update"]:
        print("\n升级前先备份版本号，升级后必须运行：python scripts/voice_smoke.py")
        print("完整步骤见 docs/edge-tts-update.md")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
