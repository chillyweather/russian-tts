from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rutts.cli import main
from tests.eval_support import (
    ENGINE_A,
    ENGINE_B,
    EXCERPTS,
    make_narrator,
    read_json,
    register_fake,
    run_preview_cli,
    two_candidates,  # noqa: F401
    write_excerpts,
)

Build = Callable[..., tuple[Path, Path]]
CSV_HEADER = "kind,excerpt_id,label,naturalness,phrasing,pronunciation,voice_stability,comfort,note,preference"


def _prepare(candidates: list[Path], out: Path) -> None:
    assert main(["eval", "prepare", "--candidates", *map(str, candidates), "--out", str(out), "--seed", "5"]) == 0


def _report(eval_dir: Path) -> dict[str, Any]:
    assert main(["eval", "report", str(eval_dir)]) == 0
    return read_json(eval_dir / "report.json")


def _import(eval_dir: Path, tmp_path: Path, rows: list[str]) -> None:
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("\n".join([CSV_HEADER, *rows]) + "\n", encoding="utf-8")
    assert main(["eval", "import-scores", str(eval_dir), "--file", str(csv_path)]) == 0


def _candidate(report: dict[str, Any], index: int) -> dict[str, Any]:
    return next(item for item in report["candidates"] if item["candidate"] == index)


@pytest.fixture
def evaluation(two_candidates: Build, tmp_path: Path) -> Path:
    a, b = two_candidates()
    out = tmp_path / "eval"
    _prepare([a, b], out)
    return out


def test_incomplete_report_names_every_unscored_sample_and_pair(evaluation: Path, tmp_path: Path) -> None:
    blind = read_json(evaluation / "blind.json")
    first = blind["excerpts"][0]
    _import(evaluation, tmp_path, [f"sample,{first['id']},A,5,5,5,5,5,,"])

    report = _report(evaluation)
    assert report["status"] == "incomplete"
    assert report["samples_scored"] == 1
    unscored = report["unscored"]["samples"]
    expected_samples = [
        {"excerpt_id": excerpt["id"], "label": label}
        for excerpt in blind["excerpts"]
        for label in excerpt["labels"]
        if not (excerpt["id"] == first["id"] and label == "A")
    ]
    assert unscored == expected_samples
    assert len(report["unscored"]["preferences"]) == len(blind["pairs"])
    assert "Unscored" in (evaluation / "report.md").read_text(encoding="utf-8")


def test_complete_report_gives_means_minimums_and_pairwise_counts(evaluation: Path, tmp_path: Path) -> None:
    blind = read_json(evaluation / "blind.json")
    key = read_json(evaluation / "key.json")
    rows: list[str] = []
    for excerpt in blind["excerpts"]:
        owner_of = {
            item["label"]: item["candidate"]
            for item in key["samples"]
            if item["excerpt_id"] == excerpt["id"]
        }
        # Candidate 0 always scores 5 and candidate 1 always scores 3 for naturalness.
        for label in excerpt["labels"]:
            natural = 5 if owner_of[label] == 0 else 3
            rows.append(f"sample,{excerpt['id']},{label},{natural},4,4,4,4,,")
        winner = next(label for label, owner in owner_of.items() if owner == 0)
        rows.append(f"pair,{excerpt['id']},A/B,,,,,,,{winner}")
    _import(evaluation, tmp_path, rows)

    report = _report(evaluation)
    assert report["status"] == "complete"
    assert report["unscored"] == {"samples": [], "preferences": []}
    first = _candidate(report, 0)
    second = _candidate(report, 1)
    assert first["means"]["naturalness"] == 5.0
    assert second["means"]["naturalness"] == 3.0
    assert second["minimums"]["naturalness"] == 3
    assert first["means"]["comfort"] == 4.0
    assert first["scored_samples"] == 5
    assert first["pairwise"] == [{"opponent": 1, "wins": 5, "losses": 0, "ties": 0}]
    assert second["pairwise"] == [{"opponent": 0, "wins": 0, "losses": 5, "ties": 0}]
    assert "Status: **complete**" in (evaluation / "report.md").read_text(encoding="utf-8")


def test_first_attempt_failures_are_reported_apart_from_failures_after_retries(
    tmp_path: Path, reference_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    narrator = make_narrator(tmp_path, reference_wav)
    excerpts = write_excerpts(tmp_path / "excerpts.json")
    register_fake(monkeypatch, ENGINE_A, precision="bf16", raise_on=frozenset({1}))
    register_fake(monkeypatch, ENGINE_B, precision="fp16", fail_texts=frozenset({EXCERPTS[1]["text"]}))
    a = tmp_path / "preview-a"
    b = tmp_path / "preview-b"
    run_preview_cli(ENGINE_A, "bf16", narrator, excerpts, a, retries=1)
    run_preview_cli(ENGINE_B, "fp16", narrator, excerpts, b, retries=1)
    out = tmp_path / "eval"
    _prepare([a, b], out)

    report = _report(out)
    recovered = _candidate(report, 0)["failure_rates"]
    assert recovered["first_attempt"] == {"failed": 1, "excerpts": 5, "rate": 0.2}
    assert recovered["after_retry_budget"] == {"failed": 0, "excerpts": 5, "rate": 0.0}
    always_failing = _candidate(report, 1)["failure_rates"]
    assert always_failing["first_attempt"] == {"failed": 1, "excerpts": 5, "rate": 0.2}
    assert always_failing["after_retry_budget"] == {"failed": 1, "excerpts": 5, "rate": 0.2}
    assert _candidate(report, 0)["retry_budget"] == 1
    # Under the default first-attempt rule both first-attempt failures make their excerpt unavailable.
    assert sorted(item["excerpt_id"] for item in report["unavailable"]) == sorted([EXCERPTS[0]["id"], EXCERPTS[1]["id"]])
    assert _candidate(report, 0)["unavailable_excerpts"] == 1
    assert _candidate(report, 1)["unavailable_excerpts"] == 1


def test_resource_observations_come_from_the_preview_measurements(evaluation: Path) -> None:
    report = _report(evaluation)
    for candidate in report["candidates"]:
        assert isinstance(candidate["resources"]["peak_rss_bytes"], int)
        assert candidate["resources"]["peak_rss_bytes"] > 0
    assert report["chapter_trials"] == []
