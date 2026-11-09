#!/usr/bin/env python3
"""sherpa-onnx 中文女声旁白候选：下载、试听合成、清单与合并音轨。

只下载女声/可作旁白的模型，全部集中在 outputs/sherpa_voices_audition/。
用法：
    python tools/sherpa_audition.py --download
    python tools/sherpa_audition.py --synth
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = (PROJECT_ROOT.parent / "outputs" / "sherpa_voices_audition").resolve()
MODEL_DIR = OUT_DIR / "models"
TEXT = (
    "身为当朝大将军之子，楼础却不是幸运儿中的一员，身上无爵无官，十八岁生日更是过得平淡无奇，"
    "连他本人也是快到中午时才突然想起今天是什么日子。楼础对此倒是不怎么在意，只在心里感叹韶华易逝，"
    "岁月如流水一般滔滔不绝，自己的一生怕是都将随波逐流，再无任何变化，年纪轻轻就生出一缕沧桑感来。"
)
SR = 24000
# GitHub release 直连在部分网络会超时，用 gh-proxy 镜像；只影响下载，模型来源不变
PROXY = "https://gh-proxy.com/"

MODELS = [
    {"key": "melo_zh_en", "dir": "vits-melo-tts-zh_en",
     "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2",
     "speakers": [0], "note": "MeloTTS 中英混合，单女声"},
    {"key": "aishell3", "dir": "vits-zh-aishell3",
     "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-aishell3.tar.bz2",
     "speakers": [83, 144, 7, 59, 93], "note": "AISHELL-3 多人声，按基频 195-229Hz 选 5 个女声"},
    {"key": "theresa", "dir": "vits-zh-hf-theresa",
     "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-hf-theresa.tar.bz2",
     "speakers": [0], "note": "HF 女声 theresa"},
    {"key": "eula", "dir": "vits-zh-hf-eula",
     "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-hf-eula.tar.bz2",
     "speakers": [0], "note": "HF 女声 eula"},
    {"key": "keqing", "dir": "vits-zh-hf-keqing",
     "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-hf-keqing.tar.bz2",
     "speakers": [0], "note": "HF 女声 keqing"},
    {"key": "bronya", "dir": "vits-zh-hf-bronya",
     "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-hf-bronya.tar.bz2",
     "speakers": [0], "note": "HF 女声 bronya"},
]


def _download(url: str, dest: Path, meta_path: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        print(f"[跳过] 已存在 {dest.name} ({dest.stat().st_size / 1048576:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    cmd = ["curl.exe", "-L", "--retry", "5", "--retry-delay", "2",
           "--connect-timeout", "20", "-o", str(tmp)]
    if tmp.exists() and tmp.stat().st_size > 0:
        cmd += ["-C", "-"]
    cmd.append(PROXY + url)
    subprocess.run(cmd, check=True)
    tmp.replace(dest)
    elapsed = time.time() - t0
    size = dest.stat().st_size
    record = {"name": dest.name, "url": url, "bytes": size,
              "download_s": round(elapsed, 1),
              "speed_mbps": round(size / 1048576 / max(elapsed, 0.1), 2)}
    # 每个模型单独一个元数据文件，避免并行下载时互相覆盖
    (OUT_DIR / f"download_{dest.stem}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[下载] {dest.name} {size / 1048576:.1f} MB，{elapsed:.1f}s，{record['speed_mbps']} MB/s")


UNNEEDED = (".pth", ".int8.onnx", "README.md", "LICENSE")



def cleanup(folder: Path) -> None:
    """删掉训练检查点/文档/int8 副本/未使用的规则包，只留推理必需文件。"""
    for path in folder.rglob("*"):
        if path.is_file() and (path.name in ("README.md", "LICENSE", "rule.far")
                               or path.name.endswith(UNNEEDED)):
            path.unlink(missing_ok=True)


def download(only: str = "") -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    meta = OUT_DIR / "download_info.json"
    keys = {part.strip() for part in only.split(",") if part.strip()}
    selected = [item for item in MODELS if not keys or item["key"] in keys]
    for item in selected:
        archive = MODEL_DIR / f"{item['dir']}.tar.bz2"
        _download(item["url"], archive, meta)
        target = MODEL_DIR / item["dir"]
        if target.exists():
            print(f"[解压] 跳过，已存在 {target.name}")
        else:
            t0 = time.time()
            with tarfile.open(archive, "r:bz2") as tar:
                tar.extractall(MODEL_DIR)
            print(f"[解压] {target.name} {time.time() - t0:.1f}s")
        cleanup(target)
        archive.unlink(missing_ok=True)  # 解压后删压缩包，避免双份占用
    total = 0
    for item in selected:
        folder = MODEL_DIR / item["dir"]
        size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        total += size
        print(f"\n--- {folder.name}  {size / 1048576:.1f} MB ---")
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                print(f"  {path.relative_to(folder)}  {path.stat().st_size / 1048576:.2f} MB")
    print(f"\n模型总体积：{total / 1073741824:.2f} GB")


def _find_model_files(folder: Path) -> dict:
    onnx = [p for p in folder.glob("*.onnx") if ".int8." not in p.name]
    if not onnx:
        raise FileNotFoundError(f"缺 onnx：{folder}")
    onnx.sort(key=lambda p: (p.name != "model.onnx", p.name))
    files = {"model": onnx[0], "tokens": folder / "tokens.txt"}
    if (folder / "lexicon.txt").exists():
        files["lexicon"] = folder / "lexicon.txt"
    if (folder / "dict").is_dir():
        files["dict_dir"] = folder / "dict"
    if (folder / "espeak-ng-data").is_dir():
        files["data_dir"] = folder / "espeak-ng-data"
    fsts = [folder / name for name in ("phone.fst", "date.fst", "number.fst") if (folder / name).exists()]
    files["rule_fsts"] = ",".join(str(p) for p in fsts)
    return files


def build_tts(folder: Path):
    import sherpa_onnx

    files = _find_model_files(folder)
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(files["model"]), tokens=str(files["tokens"]),
        lexicon=str(files.get("lexicon", "")), dict_dir=str(files.get("dict_dir", "")),
        data_dir=str(files.get("data_dir", "")),
    )
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=1, provider="cpu"),
        rule_fsts=files["rule_fsts"], max_num_sentences=1,
    )
    t0 = time.time()
    tts = sherpa_onnx.OfflineTts(config)
    return tts, time.time() - t0


def _f0_median(samples, sr: int) -> float:
    """自相关估计基频中位数，只用于筛女声，不做声纹判断。"""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32)
    frame, hop = int(sr * 0.04), int(sr * 0.02)
    lo, hi = int(sr / 320), int(sr / 70)
    values = []
    for start in range(0, len(x) - frame, hop):
        seg = x[start:start + frame]
        seg = seg - seg.mean()
        if float(np.sqrt((seg ** 2).mean())) < 0.01:
            continue
        corr = np.correlate(seg, seg, mode="full")[frame - 1:]
        lag = lo + int(np.argmax(corr[lo:hi]))
        if corr[lag] > 0.3 * corr[0]:
            values.append(sr / lag)
    return float(np.median(values)) if values else 0.0


def probe_aishell3() -> None:
    folder = MODEL_DIR / "vits-zh-aishell3"
    tts, load_s = build_tts(folder)
    print(f"[aishell3] 加载 {load_s:.1f}s")
    results = []
    for sid in range(174):
        try:
            audio = tts.generate("这是音色测试。", sid=sid, speed=1.0)
        except Exception:
            break
        f0 = _f0_median(audio.samples, audio.sample_rate)
        results.append({"sid": sid, "f0_hz": round(f0, 1),
                        "duration_s": round(len(audio.samples) / audio.sample_rate, 2)})
    (OUT_DIR / "aishell3_probe.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    female = [r for r in results if 165 <= r["f0_hz"] <= 320]
    female.sort(key=lambda r: r["f0_hz"], reverse=True)
    print(f"[aishell3] 探测 {len(results)} 个 speaker，基频女声候选 {len(female)} 个")
    for item in female[:12]:
        print(f"  sid={item['sid']:3d} f0={item['f0_hz']}Hz")


def load_download_info() -> dict:
    info: dict[str, dict] = {}
    for path in OUT_DIR.glob("download_*.json"):
        if path.name == "download_info.json":
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        info[rec["name"].replace(".tar.bz2", "")] = rec
    legacy = OUT_DIR / "download_info.json"
    if legacy.exists():
        for rec in json.loads(legacy.read_text(encoding="utf-8")):
            info.setdefault(rec["name"].replace(".tar.bz2", ""), rec)
    return info


def synth_one(key: str, start_index: int, only_sid: int | None = None) -> None:
    """一个进程只加载一个模型、只合成一个 speaker。

    多个 sherpa 模型同进程连续加载会崩溃；同一模型连续多次 generate 在本机会
    出现 OpenBLAS/bad allocation，因此按 speaker 拆成独立进程。
    """
    import sherpa_onnx

    sys.path.insert(0, str(PROJECT_ROOT))
    from core.postprocess import convert_to_wav, wav_info

    item = next(m for m in MODELS if m["key"] == key)
    folder = MODEL_DIR / item["dir"]
    if not folder.exists():
        raise SystemExit(f"模型未下载：{folder}")
    info = load_download_info()
    size_mb = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1048576
    tts, load_s = build_tts(folder)
    tts.generate("测试。", sid=item["speakers"][0], speed=1.0)  # 预热，不计时
    entries = []
    speakers = [only_sid] if only_sid is not None else item["speakers"]
    for offset, sid in enumerate(speakers):
        name = f"{start_index + offset:02d}_{item['key']}_s{sid}.wav"
        raw = OUT_DIR / f".raw_{name}"
        t0 = time.time()
        audio = tts.generate(TEXT, sid=sid, speed=1.0)
        synth_s = time.time() - t0
        sherpa_onnx.write_wave(str(raw), audio.samples, audio.sample_rate)
        out = convert_to_wav(raw, OUT_DIR / name, SR)
        raw.unlink(missing_ok=True)
        dur_s = wav_info(out)["duration_ms"] / 1000
        entries.append({
            "file": name, "model": item["key"], "dir": item["dir"], "sid": sid,
            "native_sr": audio.sample_rate, "model_mb": round(size_mb, 1),
            "load_s": round(load_s, 1),
            "download_s": info.get(item["dir"], {}).get("download_s"),
            "synth_s": round(synth_s, 2), "audio_s": round(dur_s, 2),
            "chars": len(TEXT), "chars_per_s": round(len(TEXT) / synth_s, 1),
            "rtf": round(synth_s / dur_s, 3), "note": item["note"],
        })
        print(f"[合成] {name} {synth_s:.2f}s 音频 {dur_s:.1f}s "
              f"{len(TEXT) / synth_s:.0f} 字/秒 RTF {synth_s / dur_s:.2f}")
    part_name = f"part_{key}_s{only_sid}.json" if only_sid is not None else f"part_{key}.json"
    (OUT_DIR / part_name).write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def synth_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    index = 1
    for item in MODELS:
        folder = MODEL_DIR / item["dir"]
        if not folder.exists() or not item.get("speakers"):
            print(f"[跳过] 未下载或未指定 speaker：{item['dir']}")
            continue
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        for sid in item["speakers"]:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                     "--synth-one", item["key"], "--sid", str(sid),
                                     "--start-index", str(index)],
                                    env=env)
            part = OUT_DIR / f"part_{item['key']}_s{sid}.json"
            if result.returncode != 0 or not part.exists():
                # 本机 commit 接近上限时个别大模型会 OOM，跳过但不中断整批试听
                print(f"[跳过] {item['key']} sid={sid} 合成失败（内存不足）")
                index += 1
                continue
            entries = json.loads(part.read_text(encoding="utf-8"))
            manifest.extend(entries)
            index += len(entries)
    write_outputs(manifest)


def collect_manifest() -> None:
    """只汇总已有 part_*.json，避免重复合成；排序按 MODELS 与 speakers 顺序。"""
    manifest: list[dict] = []
    for item in MODELS:
        for sid in item["speakers"]:
            part = OUT_DIR / f"part_{item['key']}_s{sid}.json"
            if part.exists():
                manifest.extend(json.loads(part.read_text(encoding="utf-8")))
    if not manifest:
        raise SystemExit("没有可汇总的 part_*.json，请先运行 --synth")
    write_outputs(manifest)


def write_outputs(manifest: list[dict]) -> None:
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    write_manifest(manifest)
    concat_auditions([item["file"] for item in manifest])


def write_manifest(manifest: list[dict]) -> None:
    path = OUT_DIR / "manifest.md"
    edge_row = ""
    edge_wav = OUT_DIR / "00_edge_Xiaoxiao.wav"
    if edge_wav.exists():
        with wave.open(str(edge_wav), "rb") as handle:
            dur = handle.getnframes() / handle.getframerate()
        edge_row = (f"| 00_edge_Xiaoxiao.wav | edge-tts | Xiaoxiao | 24000 Hz | — | — | "
                    f"4.8s（含联网） | {dur:.2f}s | {len(TEXT) / 4.8:.1f} | — |")
    lines = [
        "# sherpa-onnx 中文女声旁白候选试听清单", "",
        f"统一文本（{len(TEXT)} 字符）：{TEXT}", "",
        f"统一输出：单声道 16-bit / {SR} Hz（sherpa 原生采样率见下表，已重采样）；"
        "本清单按文件名顺序试听。", "",
        "| 文件 | 模型 | speaker | 原生采样率 | 模型体积 | 下载耗时 | 合成耗时 | 音频时长 | 字符/秒 | RTF |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in manifest:
        lines.append(
            f"| {item['file']} | {item['model']} | {item['sid']} | {item['native_sr']} Hz | "
            f"{item['model_mb']} MB | {item['download_s']}s | {item['synth_s']}s | "
            f"{item['audio_s']}s | {item['chars_per_s']} | {item['rtf']} |")
    lines += [
        "", "## 对比基准", "",
        "`00_edge_Xiaoxiao.wav`：edge-tts `zh-CN-XiaoxiaoNeural`，同样的文本、同样的 24 kHz 输出。", "",
        "| 文件 | 引擎 | speaker | 原生采样率 | 模型体积 | 下载耗时 | 合成耗时 | 音频时长 | 字符/秒 | RTF |",
        "|---|---|---|---|---|---|---|---|---|---|",
        edge_row,
        "", "说明：",
        "- RTF = 合成耗时 / 音频时长，越小越快；字符/秒含标点。",
        "- aishell3 原生 8 kHz，人声偏“电话音质”，只适合做音色筛选参考，不适合成品旁白。",
        "- `vits-zh-hf-bronya` 已下载但本机合成时触发 onnxruntime 内存分配失败（commit 接近上限），"
        "本轮未生成试听；模型仍保留，内存宽松时可补跑。",
        "- edge 的合成耗时含联网请求，不适合与本地 RTF 直接横比，只作听感基准。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    write_xlsx(manifest)


def write_xlsx(manifest: list[dict]) -> None:
    """Excel 版清单：一页表格，便于试听时打分排序。"""
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment

    columns = [("文件", "file", 26), ("模型", "model", 14), ("speaker", "sid", 9),
               ("原生采样率(Hz)", "native_sr", 14), ("模型体积(MB)", "model_mb", 13),
               ("下载耗时(s)", "download_s", 13), ("合成耗时(s)", "synth_s", 13),
               ("音频时长(s)", "audio_s", 13), ("字符/秒", "chars_per_s", 11),
               ("RTF", "rtf", 9), ("备注", "note", 40)]
    wb = Workbook()
    ws = wb.active
    ws.title = "试听清单"
    for col, (title, _, width) in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col, value=title)
        cell.font = Font(bold=True)
        ws.column_dimensions[cell.column_letter].width = width
    for row, item in enumerate(manifest, start=2):
        for col, (_, key, _) in enumerate(columns, start=1):
            value = item.get(key)
            ws.cell(row=row, column=col, value=value).alignment = Alignment(vertical="center")
    edge_wav = OUT_DIR / "00_edge_Xiaoxiao.wav"
    if edge_wav.exists():
        with wave.open(str(edge_wav), "rb") as handle:
            dur = handle.getnframes() / handle.getframerate()
        row = len(manifest) + 2
        values = ["00_edge_Xiaoxiao.wav", "edge-tts", "Xiaoxiao", SR, "—", "—", 4.8,
                  round(dur, 2), round(len(TEXT) / 4.8, 1), "—", "联网合成，作听感基准"]
        for col, value in enumerate(values, start=1):
            ws.cell(row=row, column=col, value=value)
    note_row = len(manifest) + 5
    ws.cell(row=note_row, column=1,
            value="统一文本 138 字符；输出 24kHz/mono/16bit；aishell3 原生 8kHz；bronya 因内存不足未合成。")
    wb.save(OUT_DIR / "manifest.xlsx")


def concat_auditions(files: list[str]) -> None:
    """按顺序拼接试听文件，中间 1 秒静音，方便一次听完。

    逐块读写，避免一次性把整段 PCM 读进内存（低内存机器上会 MemoryError）。
    """
    paths = [OUT_DIR / name for name in files]
    out = OUT_DIR / "audition_all.wav"
    chunk = SR  # 每次 1 秒
    with wave.open(str(out), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(SR)
        for i, path in enumerate(paths):
            if i:
                dst.writeframes(b"\x00\x00" * SR)
            with wave.open(str(path), "rb") as src:
                while True:
                    data = src.readframes(chunk)
                    if not data:
                        break
                    dst.writeframes(data)
    print(f"[拼接] {out}（{len(paths)} 个候选）")


def edge_reference() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from core.engines.edge_engine import EdgeEngine

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = EdgeEngine({"_ffmpeg": None}, SR, PROJECT_ROOT)
    out = OUT_DIR / "00_edge_Xiaoxiao.wav"
    t0 = time.time()
    engine.synthesize(TEXT, {"edge_voice": "zh-CN-XiaoxiaoNeural", "rate": "+0%",
                             "pitch": "+0Hz", "volume": "+0%"}, "平静", out)
    print(f"[edge] 00_edge_Xiaoxiao.wav 生成完成，{time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--probe-aishell", action="store_true")
    parser.add_argument("--synth", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--synth-one", default="")
    parser.add_argument("--sid", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--edge-ref", action="store_true")
    args = parser.parse_args()
    if args.synth_one:
        synth_one(args.synth_one, args.start_index, args.sid)
        raise SystemExit(0)
    if args.download:
        download(args.only)
    if args.probe_aishell:
        probe_aishell3()
    if args.synth:
        synth_all()
    if args.manifest:
        collect_manifest()
    if args.edge_ref:
        edge_reference()
