from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from rutts.narrator import NarratorProfile


@dataclass(frozen=True)
class EngineConfig:
    engine_id: str
    model_repo: str
    model_revision: str
    precision: str
    runtime_name: str
    runtime_version: str
    license: str
    settings: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SynthesisResult:
    samples: np.ndarray
    sample_rate: int


class SynthesisEngine(Protocol):
    @property
    def config(self) -> EngineConfig: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def synthesize(self, text: str, narrator: NarratorProfile) -> SynthesisResult: ...


EngineFactory = Callable[..., SynthesisEngine]

ENGINE_FACTORIES: dict[str, EngineFactory] = {}


def register_engine(engine_id: str, factory: EngineFactory) -> None:
    ENGINE_FACTORIES[engine_id] = factory


def get_engine(engine_id: str, **options: Any) -> SynthesisEngine:
    if engine_id not in ENGINE_FACTORIES:
        known = ", ".join(sorted(ENGINE_FACTORIES)) or "none"
        raise ValueError(f"unknown engine '{engine_id}' (available: {known})")
    return ENGINE_FACTORIES[engine_id](**options)
