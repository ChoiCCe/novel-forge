# edge-tts 更新流程（检查 → 升级 → 守门测试 → 回滚）

> `edge-tts` 依赖微软 Edge 免费朗读端点，声线可用性与返回格式都可能在版本间变化。
> 原则：**不自动升级**；每次升级都必须先记录旧版本，升级后跑声线守门测试（`scripts/voice_smoke.py`）。
> 通用依赖流程见 `docs/deps-update.md`，本文件只覆盖 `edge-tts`。

## 1. 查最新版本

```powershell
# PyPI 上发布的版本（含历史）
python -m pip index versions edge-tts

# 只读检查脚本：当前版本 / PyPI 最新 / GitHub 最新 release / 是否有更新 / 变更链接
python scripts/check_edge_tts_update.py
python scripts/check_edge_tts_update.py --json > outputs/edge-tts-check.json
```

手动入口：
- PyPI 项目页：https://pypi.org/project/edge-tts/
- PyPI JSON：https://pypi.org/pypi/edge-tts/json
- GitHub Releases：https://github.com/rany2/edge-tts/releases
- 变更日志：https://github.com/rany2/edge-tts/blob/master/README.md

## 2. 比对已装版本

```powershell
edge-tts --version
python -c "import importlib.metadata as m; print(m.version('edge-tts'))"
```

两个命令应一致。若命令行入口不可用（PATH 未刷新），以 `importlib.metadata` 为准。

## 3. 升级（逐版本、可回滚）

```powershell
# 0) 记录旧版本与依赖基线（回滚要用）
python -m pip freeze > outputs/pip-freeze-before-edge-tts.txt
python -c "import importlib.metadata as m; print(m.version('edge-tts'))"

# 1) 只下载目标版本，确认轮子与本机 Python 匹配
python -m pip download --no-deps --dest outputs/wheels edge-tts==<目标版本>

# 2) 安装精确版本（不要用 -U 裸升级）
python -m pip install edge-tts==<目标版本>
```

升级完成后：

1. 把 `requirements.txt` 中 `edge-tts>=7` 改成**精确版本** `edge-tts==<目标版本>`；
2. 在提交信息里写 `deps: edge-tts x.y.z -> a.b.c`；
3. 继续第 4 节验证。

## 4. 升级后必做验证清单

- [ ] `python -c "import edge_tts; print(edge_tts.__version__)"` 无导入错误
- [ ] `python scripts/voice_smoke.py` **全部声线成功**（任一失败即非零退出，不得带病上线）
- [ ] `python -m unittest discover -s tests -q` 全绿
- [ ] `python main.py --webui --port 8799` 启动后 `/api/health` 显示 `edge 就绪`
- [ ] WebUI 角色页任选一角色试听，确认有音频且时长正常
- [ ] 单章小样合成（`--chapters 1 --audio-tag smoke`），试听 3 段旁白 + 3 句对白
- [ ] 若 `voices/catalog.edge.json` 中 `available` 状态发生变化，同步更新 catalog 的
      `version / verified_at / probe_method / available / error / replaced_by`

只有全部通过，才把新版本作为基线；否则按第 5 节回滚。

## 5. 回滚

```powershell
# 用第 3 步记录的旧版本号
python -m pip install edge-tts==<旧版本>

# 若 requirements.txt 已改，先还原该文件再安装
python -m pip install -r requirements.txt

# 复验
python scripts/voice_smoke.py
python -m unittest discover -s tests -q
```

若新版本合成出的音频与旧缓存不兼容（时长/内容异常），清理对应 `cache/audio` 条目即可；
缓存 key 含引擎与韵律版本盐，旧缓存不会与新参数串用。

## 6. 检查频率与触发条件

**建议频率**：每周一次只读检查（`scripts/check_edge_tts_update.py`），或每次开始长篇合成前检查一次。

**立即检查的触发条件**：

- `scripts/voice_smoke.py` 出现任一失败，或同一时段失败率明显上升（连续 3 次 `NoAudioReceived` / `403` / 超时）；
- WebUI 分段状态出现“失败 3 次（已停止自动重试）”且错误类型集中在 `no_audio / forbidden / rate_limited`；
- 微软调整免费端点后社区集中反馈某声线失效（GitHub Issues 出现同类报告）；
- 更换网络环境或代理后大量合成超时。

**不要**自动创建 Windows 计划任务 / cron；本文件只给建议频率，由人决定何时执行检查与升级。
