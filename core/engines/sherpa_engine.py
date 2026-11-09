"""sherpa-onnx 本地中文 VITS 引擎（CPU 友好）。

模型目录需包含 model.onnx（或任意 .onnx）与 tokens.txt；中文 VITS 通常还需要 lexicon。
模型未就绪时 is_available 返回 False，路由自动降级到 edge 占位引擎。
"""
from __future__ import annotations

from pathlib import Path

from .base import BaseEngine, EngineNotReady


class SherpaEngine(BaseEngine):
    name = "sherpa"

    def __init__(self, cfg: dict, sample_rate: int, project_root: Path):
        super().__init__(cfg, sample_rate, project_root)
        self.model_dir = Path(cfg.get("model_dir", ""))
        if not self.model_dir.is_absolute():
            self.model_dir = (project_root / self.model_dir).resolve()
        self._tts = None

    def _model_files(self) -> list[Path]:
        if not self.model_dir.is_dir():
            return []
        return sorted(self.model_dir.glob("*.onnx"))

    def is_available(self, voice: dict) -> tuple[bool, str]:
        if not self.cfg.get("enabled"):
            return False, "config 未启用"
        try:
            import sherpa_onnx  # noqa: F401
        except Exception as err:
            return False, f"未安装 sherpa-onnx：{err}"
        if not self._model_files():
            return False, f"缺少模型：{self.model_dir}/*.onnx"
        if not (self.model_dir / "tokens.txt").exists():
            return False, f"缺少 tokens.txt：{self.model_dir}"
        return True, ""

    def _load(self):
        if self._tts is not None:
            return self._tts
        import sherpa_onnx

        model_path = self._model_files()[0]
        lexicon = self.model_dir / "lexicon.txt"
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model_path),
                    lexicon=str(lexicon) if lexicon.exists() else "",
                    tokens=str(self.model_dir / "tokens.txt"),
                    data_dir=str(self.model_dir / "espeak-ng-data")
                    if (self.model_dir / "espeak-ng-data").exists() else "",
                    dict_dir=str(self.model_dir / "dict") if (self.model_dir / "dict").exists() else "",
                ),
                num_threads=int(self.cfg.get("num_threads", 2)),
                provider="cpu",
            ),
            max_num_sentences=1,
        )
        self._tts = sherpa_onnx.OfflineTts(config)
        return self._tts

    def synthesize(self, text: str, voice: dict, emotion: str, out_path: Path) -> Path:
        tts = self._load()
        sid = int(voice.get("sid", self.cfg.get("sid", 0)))
        speed = float(voice.get("speed", self.cfg.get("speed", 1.0)))
        audio = tts.generate(text, sid=sid, speed=speed)
        if not audio.samples:
            raise EngineNotReady("sherpa 返回空音频")
        raw = out_path.with_suffix(".raw.wav")
        from .base import write_float_wav
        write_float_wav(audio.samples, audio.sample_rate, raw)
        from ..postprocess import convert_to_wav
        convert_to_wav(raw, out_path, self.sample_rate, self.cfg.get("_ffmpeg"))
        raw.unlink(missing_ok=True)
        return out_path
