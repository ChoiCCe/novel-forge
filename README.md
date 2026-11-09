# novel-forge

[English](README.en.md) | 中文

**把一本中文小说 TXT 自动做成多角色有声书：文本切章 → 剧本 → 选角配音 → 响度归一 → M4B / MP3 / SRT。**

Turn a Chinese novel TXT into a multi-character audiobook: chapter splitting → script → voice casting → loudness normalization → M4B / MP3 / SRT.

## 解决什么问题

- 人工给几百个角色分配声线、逐章校对说话人，成本高且容易前后不一致。
- 免费 TTS 声线数量有限，角色一多就撞声；需要一套可解释的复用规则。
- 长篇里角色会变老、改名、用敬称；声线不能每章乱跳。
- 只想先看剧本、只想跑一章试听，不想每次整本重跑。

## 核心设计

- **声线池**：从 `voices/catalog.edge.json` 动态取可用声线，按性别与年龄段建池。
- **性别硬约束**：男角色不会拿到女声，女角色不会拿到男声；判不出性别时走旁白代读。
- **年龄就近**：角色升阶段时优先选同龄段声线，参数（语速/音高/音量）随阶段渐变。
- **身份归一**：别名、敬称、绰号归并到同一个 `character_id`，跨书复用声线。
- **旁白兜底**：声线池耗尽或说话人不明时，退到旁白声线 + 独立参数档，而不是随机换声。
- **层级约束**：任何时刻不允许两个角色（含旁白）使用完全相同的「声线 + 参数」。
- **可缓存**：剧本、片段音频、选角结果都按 hash 落盘，增量重跑只补缺口。

## 快速开始

要求：Python 3.10+，Windows / macOS / Linux；合成需要联网调用 edge-tts。

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy config.example.yaml config.yaml
copy voices\registry.example.json voices\registry.json   # 可选：带 3 条示例角色
```

在仓库根目录创建 `.env`，只放 key，不要提交：

```ini
ZHIPU_API_KEY=YOUR_API_KEY_HERE
```

`config.yaml` 默认使用智谱 OpenAI 兼容接口；换其他供应商时改 `llm.providers` 与 `llm.model_chain` 即可。

放一本自己的小说（TXT，UTF-8 或 GBK）：

```powershell
python main.py --book books\your_novel.txt --chapters 0 --script-only
python main.py --book books\your_novel.txt --chapters 0-1 --full
python main.py --webui
```

章节下标从 0 开始。常用参数：

| 参数 | 作用 |
|---|---|
| `--chapters 0-3` / `--chapter 0` | 选择章节 |
| `--script-only` | 只生成剧本，不合成 |
| `--full` | 不限制旁白/对白采样条数 |
| `--force` | 忽略缓存全部重跑 |
| `--port 8765` | WebUI 端口 |

## 目录结构

```
novel-forge/
├── main.py                 CLI 入口（剧本 / 合成 / WebUI）
├── config.example.yaml     配置模板（复制为 config.yaml）
├── lexicon.yaml            多音字与专名发音替换表
├── requirements.txt
├── casting/                文本侧：切章、人名挖掘、别名与说话人 LLM 归属、声线探测
├── core/                   流水线：选角、剧本、TTS 路由、后期、M4B 导出
├── tools/                  维护与诊断脚本
├── webui/                  本地 WebUI（导入、编排、试听）
├── scripts/                edge-tts 升级检查与声线冒烟测试
├── tests/                  单元测试（不联网）
├── docs/                   依赖与 edge-tts 升级流程
└── voices/
    ├── catalog.edge.json   已探测的 edge 声线目录
    └── registry.example.json  注册表 schema + 3 条示例角色
```

运行时生成、不入库：`cache/`、`novels/`、`outputs/`、`voices/registry.json`、`.env`。

## 测试

```powershell
copy config.example.yaml config.yaml
copy voices\registry.example.json voices\registry.json
python -m unittest discover -s tests
```

## 已知限制

- **需要自备文本**：仓库不含任何小说正文，请使用自己拥有版权或已获授权的内容。
- **女声池很小**：普通话可用女声只有 3 条，旁白占用 1 条后仅剩 2 条；女性角色多时必然复用。
- **同声线复用是设计而非缺陷**：先求独立声线，池耗尽后用参数拉开距离，不会为了“不重复”牺牲可用性。
- **edge-tts 是免费端点**：声线可用性可能随微软调整变化，升级流程见 `docs/edge-tts-update.md`。
- **中文优先**：分章、人名、称谓规则主要针对中文小说。

## License

PolyForm Noncommercial License 1.0.0，见 [LICENSE](LICENSE)。个人学习、研究、非营利用途可免费使用；任何商业用途需另行授权。
