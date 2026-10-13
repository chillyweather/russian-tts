from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from rutts.atomic import atomic_path, atomic_write_text
from rutts.engines.base import SynthesisEngine
from rutts.narrator import NarratorProfile
from rutts.preview import audio_problems
from rutts.resources import measure

SMOKE_TEXT = "Короткая проверка памяти перед полным прогоном."
SMOKE_FILE = "smoke.json"
SMOKE_AUDIO = "smoke.wav"


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def run_smoke(engine: SynthesisEngine, narrator: NarratorProfile, out_dir: Path) -> dict[str, Any]:
    if (out_dir / SMOKE_FILE).exists():
        raise FileExistsError(f"smoke record already exists; choose a new --out directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "text": SMOKE_TEXT,
        "narrator": {"name": narrator.name, "reference_sha256": narrator.reference_sha256},
        "status": "failed",
        "feasible": False,
        "error": None,
        "load_seconds": 0.0,
        "render_seconds": 0.0,
        "audio_duration_seconds": None,
        "file": None,
        "resources": {"load": {}, "render": {}},
    }
    try:
        with measure() as load_sample:
            try:
                engine.load()
            except Exception as exc:
                record["error"] = _describe(exc)
        record["load_seconds"] = load_sample.elapsed_seconds
        record["resources"]["load"] = load_sample.metrics

        if record["error"] is None:
            with measure() as render_sample:
                try:
                    result = engine.synthesize(SMOKE_TEXT, narrator)
                    samples = np.asarray(result.samples, dtype=np.float32).reshape(-1)
                    record["error"] = audio_problems(samples)
                    if samples.size > 0:
                        _write_audio(out_dir, samples, result.sample_rate, record)
                except Exception as exc:
                    record["error"] = _describe(exc)
            record["render_seconds"] = render_sample.elapsed_seconds
            record["resources"]["render"] = render_sample.metrics
    finally:
        engine.unload()

    config = engine.config
    record["engine"] = {"config": config.to_dict(), "fingerprint": config.fingerprint()}
    record["status"] = "ok" if record["error"] is None else "failed"
    record["feasible"] = record["status"] == "ok"
    atomic_write_text(out_dir / SMOKE_FILE, _dump(record))
    return record


def _write_audio(out_dir: Path, samples: np.ndarray, sample_rate: int, record: dict[str, Any]) -> None:
    with atomic_path(out_dir / SMOKE_AUDIO) as temp:
        sf.write(str(temp), samples, sample_rate, format="WAV", subtype="FLOAT")
    record["file"] = SMOKE_AUDIO
    record["audio_duration_seconds"] = samples.size / sample_rate


def _dump(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, indent=2) + "\n"
