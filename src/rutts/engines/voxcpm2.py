from __future__ import annotations

import importlib.metadata
from typing import Any

from rutts.engines.base import EngineConfig, SynthesisResult
from rutts.engines.mlx_common import (
    UNRESOLVED_REVISION,
    clear_mlx_cache,
    concat_segments,
    resolve_revision,
)
from rutts.narrator import NarratorProfile

PRECISION_REPOS = {
    "bf16": "mlx-community/VoxCPM2-bf16",
    "8bit": "mlx-community/VoxCPM2-8bit",
}
LICENSE = "Apache-2.0"
GENERATION_SETTINGS: dict[str, Any] = {
    "inference_timesteps": 10,
    "cfg_value": 2.0,
    "max_tokens": 2000,
    "runtime_note": (
        "MLX port used instead of the official PyTorch MPS path; "
        "the official MPS path remains unmeasured on this machine."
    ),
    "reference_differences": (
        "ref_audio is passed as a file path that mlx-audio encodes and resamples itself; "
        "reference-only cloning mode is used, so ref_text is not used and no prompt_audio "
        "continuation is sent; there is no language code (the model is multilingual and "
        "infers language from the text)."
    ),
}


class VoxCpm2Engine:
    def __init__(self, precision: str = "bf16") -> None:
        if precision not in PRECISION_REPOS:
            raise ValueError(f"unsupported precision '{precision}' for voxcpm2 (use bf16 or 8bit)")
        self.precision = precision
        self.repo = PRECISION_REPOS[precision]
        self._revision = UNRESOLVED_REVISION
        self._model: Any = None

    @property
    def config(self) -> EngineConfig:
        return EngineConfig(
            engine_id="voxcpm2",
            model_repo=self.repo,
            model_revision=self._revision,
            precision=self.precision,
            runtime_name="mlx-audio",
            runtime_version=importlib.metadata.version("mlx-audio"),
            license=LICENSE,
            settings=dict(GENERATION_SETTINGS),
        )

    def load(self) -> None:
        from mlx_audio.tts.utils import load_model

        self._model = load_model(self.repo)
        self._revision = resolve_revision(self.repo)

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
                inference_timesteps=GENERATION_SETTINGS["inference_timesteps"],
                cfg_value=GENERATION_SETTINGS["cfg_value"],
                max_tokens=GENERATION_SETTINGS["max_tokens"],
            )
        )
        return concat_segments(segments)
