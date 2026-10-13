from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from rutts.engines import base
from rutts.engines.base import EngineConfig, get_engine
from rutts.engines.qwen_mlx import PRECISION_REPOS, QwenMlxEngine
from rutts.preview import load_suite
from rutts.resources import UNAVAILABLE, measure
from rutts import resources
from tests.conftest import FakeEngine


def _config(**overrides: object) -> EngineConfig:
    base_config = EngineConfig(
        engine_id="qwen-mlx",
        model_repo="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        model_revision="a" * 40,
        precision="bf16",
        runtime_name="mlx-audio",
        runtime_version="0.5.8",
        license="Apache-2.0",
        settings={"temperature": 0.9},
    )
    return replace(base_config, **overrides)  # type: ignore[arg-type]


def test_fingerprint_matches_canonical_json_hash() -> None:
    config = _config()
    canonical = (
        '{"engine_id":"qwen-mlx","license":"Apache-2.0","model_repo":'
        '"mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16","model_revision":"' + "a" * 40 + '",'
        '"precision":"bf16","runtime_name":"mlx-audio","runtime_version":"0.5.8",'
        '"settings":{"temperature":0.9}}'
    )
    assert config.fingerprint() == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_fingerprint_is_stable_and_sensitive_to_settings() -> None:
    assert _config().fingerprint() == _config().fingerprint()
    assert _config().fingerprint() != _config(precision="8bit").fingerprint()
    assert _config().fingerprint() != _config(settings={"temperature": 0.7}).fingerprint()


def test_qwen_precisions_map_to_mlx_community_repos() -> None:
    assert PRECISION_REPOS == {
        "bf16": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        "8bit": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    }


def test_qwen_engine_reports_unresolved_revision_before_load() -> None:
    config = QwenMlxEngine(precision="8bit").config
    assert config.model_revision == "unresolved"
    assert config.license == "Apache-2.0"
    assert config.precision == "8bit"


def test_qwen_engine_rejects_unknown_precision() -> None:
    with pytest.raises(ValueError):
        QwenMlxEngine(precision="fp16")


def test_get_engine_resolves_registered_engine(fake_engine_factory: list[FakeEngine]) -> None:
    engine = get_engine("fake", precision="bf16")
    assert isinstance(engine, FakeEngine)
    assert engine.precision == "bf16"


def test_get_engine_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="unknown engine 'nope'"):
        get_engine("nope")


def test_qwen_is_registered_by_default() -> None:
    import rutts.engines  # noqa: F401

    assert "qwen-mlx" in base.ENGINE_FACTORIES


def test_unavailable_metrics_are_marked_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_command(command: list[str]) -> None:
        return None

    monkeypatch.setattr(resources, "_run", no_command)
    monkeypatch.setattr(resources, "_mlx_peak_bytes", lambda: UNAVAILABLE)
    with measure() as sample:
        pass
    assert sample.metrics["swap_used_bytes"] == UNAVAILABLE
    assert sample.metrics["memory_free_percent"] == UNAVAILABLE
    assert sample.metrics["mlx_peak_memory_bytes"] == UNAVAILABLE
    assert isinstance(sample.metrics["peak_rss_bytes"], int)
    assert sample.elapsed_seconds >= 0


def test_swap_and_pressure_are_parsed_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = {
        "sysctl": "vm.swapusage: total = 11264.00M  used = 10109.50M  free = 1154.50M  (encrypted)\n",
        "memory_pressure": "System-wide memory free percentage: 53%\n",
    }
    monkeypatch.setattr(resources, "_run", lambda command: outputs[command[0]])
    metrics = resources._snapshot()
    assert metrics["swap_used_bytes"] == int(10109.5 * 1024**2)
    assert metrics["memory_free_percent"] == 53


def test_suite_has_sixteen_excerpts_covering_required_categories() -> None:
    excerpts = load_suite()
    assert 12 <= len(excerpts) <= 20
    categories = {excerpt.category for excerpt in excerpts}
    required = {
        "prose", "dialogue", "emotional", "long_syntax", "questions", "names",
        "numerals", "abbreviations", "stress_homograph", "yo_ambiguity",
    }
    assert required <= categories
    assert len({excerpt.id for excerpt in excerpts}) == len(excerpts)


def test_suite_excerpts_are_short_russian_passages() -> None:
    for excerpt in load_suite():
        enders = excerpt.text.count(".") + excerpt.text.count("!") + excerpt.text.count("?")
        assert 1 <= enders <= 6, excerpt.id


def test_qwen_engine_reports_the_cached_revision_before_load() -> None:
    from huggingface_hub import constants

    commit = "c" * 40
    repo_dir = Path(constants.HF_HUB_CACHE) / "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-8bit"
    (repo_dir / "snapshots" / commit).mkdir(parents=True)
    (repo_dir / "snapshots" / commit / "config.json").write_text("{}")
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(commit)

    assert QwenMlxEngine(precision="8bit").config.model_revision == commit
