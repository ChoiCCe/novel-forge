# 依赖更新流程（只读检查 → 手动升级 → 验证 → 回滚）

> 原则：**不自动升级**。任何升级都由人执行，并按本文件逐包操作。
> 适用范围：`requirements.txt` 中除 `edge-tts` 以外的依赖；`edge-tts` 另有更细的流程见 `docs/edge-tts-update.md`。

## 0. 前置：记录当前状态

```powershell
python --version
python -m pip --version
python tools/deps_check.py --json > outputs/deps_before.json
python tools/deps_check.py --online
python -m pip freeze > outputs/pip-freeze-before.txt
```

把 `outputs/deps_before.json` 与 `pip-freeze-before.txt` 作为回滚基线（`outputs/` 不入库）。

## 1. 检查上游

```powershell
python -m pip list --outdated --format=columns
# 单包精确查询（含历史版本）
python -m pip index versions edge-tts
python tools/deps_check.py --online
```

上游入口：
- PyPI 项目页：https://pypi.org/project/<包名>/
- GitHub Releases：在项目仓库 `releases` 页查看破坏性变更（重点看 `BREAKING`、`Changed`、`Removed`）。

## 2. 升级（逐包、一次一个）

```powershell
# 1) 先只下载，不安装，确认体积与平台轮子
python -m pip download --no-deps --dest outputs/wheels <包名>==<目标版本>

# 2) 安装指定精确版本（不要用 -U 裸升级）
python -m pip install <包名>==<目标版本>

# 3) 立即验证
python -m unittest discover -s tests -q
python scripts/voice_smoke.py            # 若涉及 TTS
python main.py --webui --port 8799       # 人工点一次健康检查/角色页
```

验证通过后，把 `requirements.txt` 中该包锁定为**精确版本**（`包名==x.y.z`），再进行下一个包。

## 3. 版本锁定约定

- `requirements.txt` 只写精确版本 `==`；不写 `>=`、`~=`。
- 间接依赖不手工锁定；升级主依赖后用 `pip freeze` 对照，确认没有意外的大版本跃迁。
- 每次升级在提交信息里写：`deps: <包名> x.y.z -> a.b.c`。

## 4. 回滚

```powershell
# 回到升级前记录的版本
python -m pip install <包名>==<旧版本>

# 若 requirements.txt 已改，先还原该文件再安装
python -m pip install -r requirements.txt

# 验证
python -m unittest discover -s tests -q
python scripts/voice_smoke.py
```

回滚后保留 `outputs/pip-freeze-before.txt` 作为证据；若新版本导致音频缓存不兼容，清空对应 `cache/audio` 条目即可（缓存 key 含引擎版本盐）。

## 5. 不建议做的事

- 不要 `pip install -U -r requirements.txt` 一把梭。
- 不要在未跑单测与声线守门测试前继续叠加第二个升级。
- 不要把 `outputs/`、`cache/`、音频、密钥提交到仓库。
