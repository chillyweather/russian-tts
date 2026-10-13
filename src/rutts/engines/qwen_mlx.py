from __future__ import annotations

import importlib.metadata
import re
from typing import Any

import numpy as np

from rutts.engines.base import EngineConfig, SynthesisResult
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
}
LICENSE = "Apache-2.0"
UNRESOLVED_REVISION = "unresolved"
_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")


class QwenMlxEngine:
    def __init__(self, precision: str = "bf16") -> None:
        if precision not in PRECISION_REPOS:
            raise ValueError(f"unsupported precision '{precision}' (use bf16 or 8bit)")
        self.precision = precision
        self.repo = PRECISION_REPOS[precision]
        self._revision = UNRESOLVED_REVISION
        self._model: Any = None

    @property
    def config(self) -> EngineConfig:
        return EngineConfig(
            engine_id="qwen-mlx",
            model_repo=self.repo,
            model_revision=self._revision,
            precision=self.precision,
            runtime_name="mlx-audio",
            runtime_version=importlib.metadata.version("mlx-audio"),
            license=LICENSE,
            settings=dict(GENERATION_SETTINGS),
        )

    def load(self) -> None:
        from huggingface_hub import snapshot_download
        from mlx_audio.tts.utils import load_model

        self._model = load_model(self.repo)
        snapshot = snapshot_download(self.repo, local_files_only=True)
        commit = snapshot.rstrip("/").split("/")[-1]
        if not _COMMIT_HASH.match(commit):
            raise RuntimeError(f"could not resolve a commit hash for {self.repo}: {snapshot}")
        self._revision = commit

    def unload(self) -> None:
        self._model = None

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
        if not segments:
            raise RuntimeError("engine returned no audio segments")
        rates = {int(segment.sample_rate) for segment in segments}
        if len(rates) != 1:
            raise RuntimeError(f"engine returned mixed sample rates: {sorted(rates)}")
        samples = np.concatenate(
            [np.asarray(segment.audio, dtype=np.float32).reshape(-1) for segment in segments]
        )
        return SynthesisResult(samples=samples, sample_rate=rates.pop())
