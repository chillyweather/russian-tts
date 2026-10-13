from __future__ import annotations

from pathlib import Path

import pytest

from rutts.engines import base
from rutts.engines.fish_mlx import FishMlxEngine
from rutts.engines.voxcpm2 import VoxCpm2Engine
from rutts.narrator import NarratorProfile


def _narrator() -> NarratorProfile:
    return NarratorProfile(
        name="n",
        reference_audio=Path("reference.wav"),
        transcript="Тест.",
        language="ru",
        provenance="test",
        reference_sha256="0" * 64,
        directory=Path("."),
    )


def test_fish_and_voxcpm2_are_registered_by_default() -> None:
    import rutts.engines  # noqa: F401

    assert base.ENGINE_FACTORIES["fish-mlx"] is FishMlxEngine
    assert base.ENGINE_FACTORIES["voxcpm2"] is VoxCpm2Engine


def test_fish_defaults_to_8bit_community_repo() -> None:
    config = base.get_engine("fish-mlx").config
    assert config.engine_id == "fish-mlx"
    assert config.model_repo == "mlx-community/fish-audio-s2-pro-8bit"
    assert config.precision == "8bit"
    assert config.license == "Fish Audio Research License (non-commercial)"
    assert config.model_revision == "unresolved"
    assert config.settings["conversion_provenance"] == (
        "Community mlx-community 8-bit conversion of fishaudio/s2-pro; "
        "not the hosted S2.1 release."
    )
    assert "ref_audio" in str(config.settings["reference_differences"])
    assert "no language code" in str(config.settings["reference_differences"])


@pytest.mark.parametrize("precision", ["bf16", "4bit", "fp16"])
def test_fish_rejects_precisions_other_than_8bit(precision: str) -> None:
    with pytest.raises(ValueError, match=f"unsupported precision '{precision}' for fish-mlx .*use 8bit"):
        FishMlxEngine(precision=precision)


def test_fish_synthesize_before_load_raises() -> None:
    with pytest.raises(RuntimeError, match="call load"):
        FishMlxEngine().synthesize("Привет.", _narrator())


def test_voxcpm2_defaults_to_bf16_mlx_repo_with_honest_runtime_note() -> None:
    config = base.get_engine("voxcpm2").config
    assert config.engine_id == "voxcpm2"
    assert config.model_repo == "mlx-community/VoxCPM2-bf16"
    assert config.precision == "bf16"
    assert config.runtime_name == "mlx-audio"
    assert config.license == "Apache-2.0"
    note = str(config.settings["runtime_note"])
    assert "MLX port used instead of the official PyTorch MPS path" in note
    assert "unmeasured" in note


def test_voxcpm2_accepts_8bit_and_rejects_unknown_precision() -> None:
    assert VoxCpm2Engine(precision="8bit").config.model_repo == "mlx-community/VoxCPM2-8bit"
    with pytest.raises(ValueError, match="unsupported precision 'fp16' for voxcpm2 .*use bf16 or 8bit"):
        VoxCpm2Engine(precision="fp16")


def test_voxcpm2_synthesize_before_load_raises() -> None:
    with pytest.raises(RuntimeError, match="call load"):
        VoxCpm2Engine().synthesize("Привет.", _narrator())


def test_voxcpm2_records_reference_differences() -> None:
    differences = str(VoxCpm2Engine().config.settings["reference_differences"])
    assert "file path" in differences
    assert "ref_text is not used" in differences
