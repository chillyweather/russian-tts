from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rutts.engines.fish_mlx import FishMlxEngine
from rutts.narrator import create_profile, load_profile
from tests.conftest import write_reference_wav


@pytest.mark.local_model
def test_real_fish_preview_produces_audio(tmp_path: Path) -> None:
    reference = write_reference_wav(tmp_path / "reference.wav", seconds=5.0)
    profile = create_profile(
        tmp_path / "narrator",
        name="Local check",
        audio=reference,
        transcript="Это тестовая фраза для опорной записи.",
        provenance="Synthetic tone; local model check only",
    )
    narrator = load_profile(profile.directory)
    engine = FishMlxEngine()
    engine.load()
    try:
        result = engine.synthesize("Привет, это проверка локальной озвучки.", narrator)
    finally:
        engine.unload()
    assert result.samples.size > 0
    assert np.isfinite(result.samples).all()
    assert len(engine.config.model_revision) == 40
    assert result.samples.size / result.sample_rate > 0.5
