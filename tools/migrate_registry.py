"""v4 步骤 2：registry v1 -> v2 迁移（默认 dry-run）。

用法：
    python tools/migrate_registry.py              # 只打印报告，不改文件
    python tools/migrate_registry.py --apply      # 备份后原子写入
    python tools/migrate_registry.py --rollback voices/registry.v1.2026....bak.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from core import casting  # noqa: E402

REG = ROOT / "voices" / "registry.json"
CFG = ROOT / "config.yaml"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _backup(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bak = path.with_name(f"{path.stem}.v1.{stamp}.bak{path.suffix}")
    shutil.copy2(path, bak)
    return bak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="执行迁移（默认 dry-run）")
    ap.add_argument("--rollback", metavar="BACKUP", help="用备份文件覆盖 registry.json")
    args = ap.parse_args()

    if args.rollback:
        bak = Path(args.rollback)
        if not bak.exists():
            print(f"备份不存在：{bak}")
            return 1
        shutil.copy2(bak, REG)
        print(f"已回滚 -> {REG}（来源 {bak}）")
        return 0

    reg = _load(REG)
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    bak_path = ""
    if args.apply and int(reg.get("schema_version", 1)) < 2:
        # config 首次大改也留一份备份
        cfg_bak = ROOT / "config.v3.bak.yaml"
        if not cfg_bak.exists():
            shutil.copy2(CFG, cfg_bak)
            print(f"config 备份 -> {cfg_bak}")
        bak_path = str(_backup(REG))

    new_reg, report = casting.migrate_v1(reg, cfg=cfg, backup=bak_path)

    print(json.dumps({k: v for k, v in report.items() if k != "merged"}, ensure_ascii=False, indent=2))
    if report.get("merged"):
        print(f"合并到已有身份的条目：{len(report['merged'])}（示例 {report['merged'][:3]}）")

    if not args.apply:
        print("\n[dry-run] 未写入。加 --apply 执行。")
        return 0
    if not bak_path:
        print("\n已是 v2，无需迁移。")
        return 0

    _write_atomic(REG, new_reg)
    after = _load(REG)
    summary = {
        "schema_version": after.get("schema_version"),
        "identities": len(after.get("identities", {})),
        "bindings": len(after.get("voice_bindings", {})),
        "events": len(after.get("voice_events", [])),
        "characters_legacy_kept": len(after.get("characters", {})),
        "backup": bak_path,
    }
    print("\n[applied]", json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
