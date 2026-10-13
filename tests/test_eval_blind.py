from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from rutts.cli import main
from tests.eval_support import (
    ENGINE_A,
    ENGINE_B,
    EXCERPTS,
    make_narrator,
    read_json,
    register_fake,
    run_preview_cli,
    two_candidates,
    write_excerpts,
)

Build = Callable[..., tuple[Path, Path]]


def _prepare(candidates: list[Path], out: Path, seed: int = 7, attempt: str = "first") -> int:
    return main(
        [
            "eval", "prepare", "--candidates", *map(str, candidates), "--out", str(out),
            "--seed", str(seed), "--attempt", attempt,
        ]
    )


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def test_prepare_writes_anonymous_blind_file_without_identities(two_candidates: Build, tmp_path: Path) -> None:
    a, b = two_candidates()
    out = tmp_path / "eval"
    assert _prepare([a, b], out) == 0

    blind_text = (out / "blind.json").read_text(encoding="utf-8")
    fingerprints = [read_json(a / "results.json")["engine"]["fingerprint"], read_json(b / "results.json")["engine"]["fingerprint"]]
    for forbidden in [*fingerprints, ENGINE_A, ENGINE_B, "fake/model", "bf16", "fp16", str(a), str(b), "Fixture narrator"]:
        assert forbidden not in blind_text

    blind = read_json(out / "blind.json")
    assert blind["seed"] == 7
    assert [item["id"] for item in blind["excerpts"]] == [
        "dialogue_evening", "names_meeting", "numerals_birth", "prose_morning", "question_truth",
    ]
    assert all(sorted(item["labels"]) == ["A", "B"] for item in blind["excerpts"])
    assert len(blind["pairs"]) == 5
    assert all(pair["labels"] == ["A", "B"] for pair in blind["pairs"])
    for item in blind["excerpts"]:
        for label in item["labels"]:
            assert (out / "samples" / item["id"] / f"{label}.wav").is_file()


def test_key_maps_each_label_to_its_candidate(two_candidates: Build, tmp_path: Path) -> None:
    a, b = two_candidates()
    out = tmp_path / "eval"
    _prepare([a, b], out)

    key = read_json(out / "key.json")
    fingerprint_by_dir = {
        str(a.resolve()): read_json(a / "results.json")["engine"]["fingerprint"],
        str(b.resolve()): read_json(b / "results.json")["engine"]["fingerprint"],
    }
    assert len(key["samples"]) == 10
    for entry in key["samples"]:
        assert entry["engine_fingerprint"] == fingerprint_by_dir[entry["candidate_dir"]]
        assert entry["attempt"] == 1
        assert entry["file"] == f"samples/{entry['excerpt_id']}/{entry['label']}.wav"


def test_same_seed_gives_same_labels_and_other_seed_can_differ(two_candidates: Build, tmp_path: Path) -> None:
    a, b = two_candidates()
    _prepare([a, b], tmp_path / "one", seed=11)
    _prepare([a, b], tmp_path / "two", seed=11)
    _prepare([a, b], tmp_path / "three", seed=12)

    def labelling(directory: Path) -> list[tuple[str, str, int]]:
        return [
            (item["excerpt_id"], item["label"], item["candidate"])
            for item in read_json(directory / "key.json")["samples"]
        ]

    assert labelling(tmp_path / "one") == labelling(tmp_path / "two")
    assert labelling(tmp_path / "one") != labelling(tmp_path / "three")


def test_loudness_is_matched_by_pure_gain(two_candidates: Build, tmp_path: Path) -> None:
    a, b = two_candidates()
    out = tmp_path / "eval"
    _prepare([a, b], out)

    key = read_json(out / "key.json")
    assert key["loudness"]["target_lowered"] is False
    rms_values: dict[str, list[float]] = {}
    for entry in key["samples"]:
        source, _ = sf.read(Path(entry["candidate_dir"]) / entry["source_file"], dtype="float32")
        scaled, rate = sf.read(out / entry["file"], dtype="float32")
        source_rate = sf.info(Path(entry["candidate_dir"]) / entry["source_file"]).samplerate
        assert rate == source_rate
        assert scaled.shape == source.shape
        gain = 10 ** (entry["gain_db"] / 20)
        np.testing.assert_allclose(scaled, source * gain, atol=1e-6)
        rms_values.setdefault(entry["excerpt_id"], []).append(_rms(scaled))

    for excerpt_id, values in rms_values.items():
        assert len(values) == 2
        assert abs(values[0] - values[1]) / max(values) < 0.01, excerpt_id


def test_clipping_lowers_common_target_for_the_whole_set(two_candidates: Build, tmp_path: Path) -> None:
    a, b = two_candidates()
    # A quiet body with one full-scale click: raising it to the mean level would clip.
    peaky = np.full(24000, 0.05, dtype=np.float32)
    peaky[12000] = 0.9
    sf.write(str(a / "dialogue_evening" / "attempt-1.wav"), peaky, 24000, subtype="FLOAT")
    out = tmp_path / "eval"
    assert _prepare([a, b], out) == 0

    key = read_json(out / "key.json")
    assert key["loudness"]["target_lowered"] is True
    for entry in key["samples"]:
        scaled, _ = sf.read(out / entry["file"], dtype="float32")
        assert float(np.max(np.abs(scaled))) <= 1.0
    peaky_entry = next(
        e for e in key["samples"] if e["excerpt_id"] == "dialogue_evening" and e["candidate_dir"] == str(a.resolve())
    )
    peaky_output, _ = sf.read(out / peaky_entry["file"], dtype="float32")
    assert float(np.max(np.abs(peaky_output))) == pytest.approx(1.0, abs=1e-6)


def test_unavailable_excerpt_is_listed_and_not_sampled(
    tmp_path: Path, reference_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    narrator = make_narrator(tmp_path, reference_wav)
    excerpts = write_excerpts(tmp_path / "excerpts.json")
    register_fake(monkeypatch, ENGINE_A, precision="bf16")
    register_fake(monkeypatch, ENGINE_B, precision="fp16", fail_texts=frozenset({EXCERPTS[0]["text"]}))
    a = tmp_path / "preview-a"
    b = tmp_path / "preview-b"
    run_preview_cli(ENGINE_A, "bf16", narrator, excerpts, a)
    run_preview_cli(ENGINE_B, "fp16", narrator, excerpts, b, retries=1)

    out = tmp_path / "eval"
    assert _prepare([a, b], out) == 0
    blind = read_json(out / "blind.json")
    assert "prose_morning" not in [item["id"] for item in blind["excerpts"]]
    assert [item["excerpt_id"] for item in blind["unavailable"]] == ["prose_morning"]
    assert "first attempt failed" in blind["unavailable"][0]["reason"]
    key = read_json(out / "key.json")
    assert key["unavailable"][0]["excerpt_id"] == "prose_morning"


def test_final_attempt_uses_the_retry_that_succeeded(
    tmp_path: Path, reference_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    narrator = make_narrator(tmp_path, reference_wav)
    excerpts = write_excerpts(tmp_path / "excerpts.json")
    register_fake(monkeypatch, ENGINE_A, precision="bf16", raise_on=frozenset({1}))
    register_fake(monkeypatch, ENGINE_B, precision="fp16")
    a = tmp_path / "preview-a"
    b = tmp_path / "preview-b"
    run_preview_cli(ENGINE_A, "bf16", narrator, excerpts, a, retries=1)
    run_preview_cli(ENGINE_B, "fp16", narrator, excerpts, b, retries=1)

    first = tmp_path / "first"
    final = tmp_path / "final"
    _prepare([a, b], first, attempt="first")
    _prepare([a, b], final, attempt="final")
    first_key = read_json(first / "key.json")
    final_key = read_json(final / "key.json")
    assert first_key["unavailable"][0]["excerpt_id"] == "prose_morning"
    assert final_key["unavailable"] == []
    attempts = {
        (entry["excerpt_id"], entry["candidate_dir"]): entry["attempt"] for entry in final_key["samples"]
    }
    assert attempts[("prose_morning", str(a.resolve()))] == 2


def test_prepare_refuses_to_overwrite_an_evaluation(two_candidates: Build, tmp_path: Path) -> None:
    a, b = two_candidates()
    out = tmp_path / "eval"
    assert _prepare([a, b], out) == 0
    before = (out / "blind.json").read_text(encoding="utf-8")
    assert _prepare([a, b], out) == 2
    assert (out / "blind.json").read_text(encoding="utf-8") == before


def test_prepare_with_one_candidate_still_writes_samples(two_candidates: Build, tmp_path: Path) -> None:
    a, _ = two_candidates()
    out = tmp_path / "eval"
    assert _prepare([a], out) == 0
    blind: dict[str, Any] = read_json(out / "blind.json")
    assert all(item["labels"] == ["A"] for item in blind["excerpts"])
    assert blind["pairs"] == []
