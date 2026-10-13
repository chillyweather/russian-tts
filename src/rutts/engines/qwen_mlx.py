from __future__ import annotations

import importlib.metadata
from typing import Any

from rutts.engines.base import EngineConfig, SynthesisResult
from rutts.engines.mlx_common import (
    cached_revision,
    clear_mlx_cache,
    concat_segments,
    resolve_revision,
)
from rutts.narrator import NarratorProfile

MODEL_FAMILY = "Qwen3-TTS-12Hz-1.7B-Base"
PRECISION_REPOS = {
    "bf16": f"mlx-community/{MODEL_FAMILY}-bf16",
    "8bit": f"mlx-community/{MODEL_FAMILY}-8bit",
}
LANGUAGE = "russian"
GENERATION_SETTINGS: dict[str, Any] = {
    "lang_code": LANGUAGE,
    "temperature": 0.9,
    "max_tokens": 4096,
    "split_pattern": "\n",
    "decoded_length": "full",
}
LICENSE = "Apache-2.0"


def _keep_full_decoded_length(model: Any) -> None:
    """Stop mlx-audio 0.5.8 cutting the end of the audio.

    Its tokenizer counts frames whose first code is non-zero as the valid length, but 0 is an
    ordinary code, so every such frame removes 80 ms from the end of the utterance.
    """
    import mlx.core as mx

    tokenizer = model.speech_tokenizer
    decode = tokenizer.decode

    def decode_full(codes: Any) -> tuple[Any, Any]:
        wav, _ = decode(codes)
        return wav, mx.array([wav.shape[-1]] * wav.shape[0])

    tokenizer.decode = decode_full


class QwenMlxEngine:
    def __init__(self, precision: str = "bf16") -> None:
        if precision not in PRECISION_REPOS:
            raise ValueError(f"unsupported precision '{precision}' (use bf16 or 8bit)")
        self.precision = precision
        self.repo = PRECISION_REPOS[precision]
        self._model: Any = None

    @property
    def config(self) -> EngineConfig:
        return EngineConfig(
            engine_id="qwen-mlx",
            model_repo=self.repo,
            model_revision=cached_revision(self.repo),
            precision=self.precision,
            runtime_name="mlx-audio",
            runtime_version=importlib.metadata.version("mlx-audio"),
            license=LICENSE,
            settings=dict(GENERATION_SETTINGS),
        )

    def load(self) -> None:
        from mlx_audio.tts.utils import load_model

        self._model = load_model(self.repo)
        _keep_full_decoded_length(self._model)
        resolve_revision(self.repo)

    def unload(self) -> None:
        self._model = None
        clear_mlx_cache()

    def synthesize(self, text: str, narrator: NarratorProfile) -> SynthesisResult:
        if self._model is None:
            raise RuntimeError("engine is not loaded; call load() first")
        segments = list(
            self._model.generate(
                text=text,
                ref_audio=str(narrator.reference_audio),
                ref_text=narrator.transcript,
                lang_code=LANGUAGE,
                temperature=GENERATION_SETTINGS["temperature"],
                max_tokens=GENERATION_SETTINGS["max_tokens"],
                split_pattern=GENERATION_SETTINGS["split_pattern"],
                verbose=False,
            )
        )
        return concat_segments(segments)
