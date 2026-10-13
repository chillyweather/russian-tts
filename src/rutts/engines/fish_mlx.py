from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import soundfile as sf

from rutts.engines.base import EngineConfig, SynthesisResult
from rutts.engines.mlx_common import (
    cached_revision,
    clear_mlx_cache,
    concat_segments,
    resolve_revision,
)
from rutts.narrator import NarratorProfile

PRECISION = "8bit"
REPO = "mlx-community/fish-audio-s2-pro-8bit"
LICENSE = "Fish Audio Research License (non-commercial)"
GENERATION_SETTINGS: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 30,
    "max_tokens": 1024,
    "chunk_length": 300,
    "conversion_provenance": (
        "Community mlx-community 8-bit conversion of fishaudio/s2-pro; "
        "not the hosted S2.1 release."
    ),
    "reference_differences": (
        "ref_audio is passed as a mono mx.array resampled to the model's codec rate, "
        "not a file path; ref_text is the narrator transcript; no language code is passed "
        "(the model infers language from the text, so lang_code and split_pattern do not apply); "
        "chunk_length is a byte budget for internal batching."
    ),
}


class FishMlxEngine:
    def __init__(self, precision: str = PRECISION) -> None:
        if precision != PRECISION:
            raise ValueError(f"unsupported precision '{precision}' for fish-mlx (use 8bit)")
        self.precision = precision
        self.repo = REPO
        self._model: Any = None

    @property
    def config(self) -> EngineConfig:
        return EngineConfig(
            engine_id="fish-mlx",
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
                ref_audio=_reference_array(self._model.sample_rate, narrator.reference_audio),
                ref_text=narrator.transcript,
                temperature=GENERATION_SETTINGS["temperature"],
                top_p=GENERATION_SETTINGS["top_p"],
                top_k=GENERATION_SETTINGS["top_k"],
                max_tokens=GENERATION_SETTINGS["max_tokens"],
                chunk_length=GENERATION_SETTINGS["chunk_length"],
                verbose=False,
            )
        )
        return concat_segments(segments)


def _reference_array(codec_rate: int, path: Path) -> Any:
    import mlx.core as mx
    from mlx_audio.utils import resample_audio

    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = mx.array(samples.mean(axis=1))
    if rate != codec_rate:
        audio = resample_audio(audio, rate, codec_rate)
    return audio
