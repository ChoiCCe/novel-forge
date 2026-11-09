# novel-forge

English | [中文](README.md)

**Turn a Chinese novel TXT into a multi-character audiobook: chapter splitting → script → voice casting → loudness normalization → M4B / MP3 / SRT.**

把一本中文小说 TXT 自动做成多角色有声书：文本切章 → 剧本 → 选角配音 → 响度归一 → M4B / MP3 / SRT。

## Why it exists

- Assigning voices to hundreds of characters and checking speakers chapter by chapter is slow and inconsistent.
- Free TTS catalogs are small, so characters collide; you need explainable reuse rules.
- In a long book, characters age, change names and use honorifics; voices must not jump around.
- You often want a script-only pass or a one-chapter preview without re-running the whole book.

## Core design

- **Voice pools** — build pools from the verified `voices/catalog.edge.json`, grouped by gender and age stage.
- **Hard gender constraint** — male characters never get a female voice and vice versa; unknown gender is read by the narrator.
- **Age proximity** — when a character ages, prefer voices of the matching stage and shift prosody gradually.
- **Identity normalization** — aliases, honorifics and nicknames merge into one `character_id` reused across books.
- **Narrator fallback** — when the pool is exhausted or the speaker is unknown, fall back to the narrator voice with independent prosody instead of a random voice.
- **No identical pairs** — no two characters (including the narrator) may share the exact same voice + prosody combination.
- **Cache friendly** — scripts, segment audio and casting results are content-hashed; reruns only fill gaps.

## Quick start

Requires Python 3.10+, Windows / macOS / Linux, and network access for edge-tts.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy config.example.yaml config.yaml
copy voices\registry.example.json voices\registry.json   # optional: 3 sample characters
```

Create `.env` in the repo root (never commit it):

```ini
ZHIPU_API_KEY=YOUR_API_KEY_HERE
```

`config.yaml` defaults to the Zhipu OpenAI-compatible endpoint. To use another provider, edit `llm.providers` and `llm.model_chain`.

Add your own novel (TXT, UTF-8 or GBK):

```powershell
python main.py --book books\your_novel.txt --chapters 0 --script-only
python main.py --book books\your_novel.txt --chapters 0-1 --full
python main.py --webui
```

Chapter indexes start at 0. Common flags:

| Flag | Purpose |
|---|---|
| `--chapters 0-3` / `--chapter 0` | Select chapters |
| `--script-only` | Generate the script only |
| `--full` | Do not cap narration/dialogue samples |
| `--force` | Ignore caches and rerun everything |
| `--port 8765` | WebUI port |

## Layout

```
novel-forge/
├── main.py                 CLI entry (script / audio / WebUI)
├── config.example.yaml     Config template (copy to config.yaml)
├── lexicon.yaml            Pronunciation replacements
├── requirements.txt
├── casting/                Text side: chapters, name mining, LLM alias/speaker resolution, voice probing
├── core/                   Pipeline: casting, script, TTS routing, postprocess, M4B export
├── tools/                  Maintenance and diagnostics
├── webui/                  Local WebUI (import, orchestrate, preview)
├── scripts/                edge-tts update check and voice smoke test
├── tests/                  Offline unit tests
├── docs/                   Dependency and edge-tts upgrade guides
└── voices/
    ├── catalog.edge.json   Verified edge voice catalog
    └── registry.example.json  Registry schema + 3 sample characters
```

Generated at runtime and ignored by git: `cache/`, `novels/`, `outputs/`, `voices/registry.json`, `.env`.

## Tests

```powershell
copy config.example.yaml config.yaml
copy voices\registry.example.json voices\registry.json
python -m unittest discover -s tests
```

## Known limitations

- **Bring your own text** — this repo contains no novel text. Use content you own or are licensed to process.
- **Tiny female pool** — only 3 Mandarin female voices are available; the narrator takes one, leaving 2 for characters, so reuse is unavoidable.
- **Same-voice reuse is intentional** — independent voices come first; when the pool runs out, prosody is used to separate characters.
- **edge-tts is a free endpoint** — voice availability may change; see `docs/edge-tts-update.md`.
- **Chinese-first** — chapter, name and honorific rules target Chinese novels.

## License

MIT, see [LICENSE](LICENSE).
