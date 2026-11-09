"""CosyVoice2 引擎：优先复用 third_party 的 /generate HTTP 服务，也保留进程内方式。

服务端契约（来自 third_party/Muliti-Role_Cosyvoice2/src/tts/server/server.py）：
POST /generate
{"text": "...", "prompts": {"default": {"audio_path": "...", "text": "...", "class_name": "角色|旁白"}}}
返回 audio/wav 字节流。
"""
from __future__ import annotations

import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from .base import BaseEngine, EngineNotReady


class CosyVoiceEngine(BaseEngine):
    name = "cosyvoice"

    def __init__(self, cfg: dict, sample_rate: int, project_root: Path):
        super().__init__(cfg, sample_rate, project_root)
        self._probe_result: bool | None = None

    def _probe_http(self) -> bool:
        if self._probe_result is not None:
            return self._probe_result
        parsed = urlparse(self.cfg.get("url", ""))
        host, port = parsed.hostname, parsed.port or 80
        try:
            with socket.create_connection((host, port), timeout=1.5):
                self._probe_result = True
        except OSError:
            self._probe_result = False
        return self._probe_result

    def is_available(self, voice: dict) -> tuple[bool, str]:
        if not self.cfg.get("enabled"):
            return False, "config 未启用（缺 torch / CosyVoice2 权重）"
        ref = voice.get("ref_wav")
        ref_text = voice.get("ref_text")
        if not ref or not Path(ref).exists():
            return False, f"缺少参考音频 {ref or '(未配置)'}"
        if not ref_text:
            return False, f"缺少与参考音频逐字对应的文本：{Path(ref).with_suffix('.txt')}"
        if self.cfg.get("mode", "http") == "http" and not self._probe_http():
            return False, f"HTTP 服务不可达：{self.cfg.get('url')}"
        return True, ""

    def synthesize(self, text: str, voice: dict, emotion: str, out_path: Path) -> Path:
        if self.cfg.get("mode", "http") == "in_process":
            return self._in_process(text, voice, out_path)
        return self._http(text, voice, out_path)

    def _http(self, text: str, voice: dict, out_path: Path) -> Path:
        import requests

        payload = {
            "text": text,
            "prompts": {
                "default": {
                    "audio_path": voice["ref_wav"],
                    "text": voice.get("ref_text", ""),
                    "class_name": voice.get("class_name", "角色"),
                }
            },
        }
        response = requests.post(self.cfg["url"], json=payload, timeout=float(self.cfg.get("timeout_s", 600)))
        if response.status_code != 200 or not response.content:
            raise EngineNotReady(f"CosyVoice HTTP {response.status_code}: {response.text[:200]}")
        with tempfile.TemporaryDirectory(prefix="novel-forge-cosy-") as tmp:
            wav_path = Path(tmp) / "cosy.wav"
            wav_path.write_bytes(response.content)
            from ..postprocess import convert_to_wav
            convert_to_wav(wav_path, out_path, self.sample_rate, self.cfg.get("_ffmpeg"))
        return out_path

    def _in_process(self, text: str, voice: dict, out_path: Path) -> Path:
        """进程内推理；需要 torch/torchaudio 与 CosyVoice2-0.5B 权重，V1 未验证。"""
        import torch
        import torchaudio
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav
        from .base import write_float_wav

        model_dir = Path(self.cfg.get("in_process_model_dir", ""))
        if not model_dir.is_absolute():
            model_dir = (self.root / model_dir).resolve()
        if not model_dir.exists():
            raise EngineNotReady(f"缺少 CosyVoice2 权重目录：{model_dir}")
        model = CosyVoice2(str(model_dir), load_jit=False, load_trt=False, load_vllm=False,
                           fp16=False, class_name=voice.get("class_name"))
        prompt = load_wav(voice["ref_wav"], 16000)
        chunks = [item["tts_speech"] for item in model.inference_zero_shot(
            text, voice.get("ref_text", ""), prompt, stream=False)]
        if not chunks:
            raise EngineNotReady("CosyVoice2 返回空音频")
        tensor = torch.cat(chunks, dim=-1)
        raw = out_path.with_suffix(".raw.wav")
        torchaudio.save(str(raw), tensor, model.sample_rate)
        from ..postprocess import convert_to_wav
        convert_to_wav(raw, out_path, self.sample_rate, self.cfg.get("_ffmpeg"))
        raw.unlink(missing_ok=True)
        return out_path
