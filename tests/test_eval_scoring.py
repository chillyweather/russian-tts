from __future__ import annotations

import io
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rutts.cli import main
from rutts.evaluation.scoring import run_session
from tests.eval_support import read_json, two_candidates, write_json  # noqa: F401

Build = Callable[..., tuple[Path, Path]]
SCORES = ["5", "4", "4", "3", "5"]


def _prepare(candidates: list[Path], out: Path) -> None:
    assert main(["eval", "prepare", "--candidates", *map(str, candidates), "--out", str(out), "--seed", "3"]) == 0


def _sample_script(labels: list[str], pairs: list[list[str]], scores: list[str]) -> list[str]:
    lines: list[str] = []
    for label in labels:
        lines += [*scores, f"note {label}"]
    for _ in pairs:
        lines.append("A")
    return lines


def _full_script(blind: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for excerpt in blind["excerpts"]:
        pairs = [pair["labels"] for pair in blind["pairs"] if pair["excerpt_id"] == excerpt["id"]]
        lines += _sample_script(excerpt["labels"], pairs, SCORES)
    return lines


def _run(eval_dir: Path, lines: list[str], play: list[Path] | None = None) -> bool:
    stdin = io.StringIO("".join(line + "\n" for line in lines))
    played = play if play is not None else []
    return run_session(eval_dir, stdin=stdin, stdout=io.StringIO(), play=played.append)


@pytest.fixture
def prepared(two_candidates: Build, tmp_path: Path) -> Path:
    a, b = two_candidates()
    out = tmp_path / "eval"
    _prepare([a, b], out)
    return out


def test_session_scores_every_sample_and_pair_and_saves(prepared: Path) -> None:
    blind = read_json(prepared / "blind.json")
    played: list[Path] = []
    assert _run(prepared, _full_script(blind), played) is True

    scores = read_json(prepared / "scores.json")
    first = blind["excerpts"][0]["id"]
    assert scores["samples"][first]["A"] == {
        "naturalness": 5,
        "phrasing": 4,
        "pronunciation": 4,
        "voice_stability": 3,
        "comfort": 5,
        "note": "note A",
    }
    assert scores["preferences"][first] == {"A/B": "A"}
    assert len(played) == 10
    assert played[0] == prepared / "samples" / first / "A.wav"


def test_out_of_range_and_malformed_input_is_asked_again(prepared: Path) -> None:
    first = read_json(prepared / "blind.json")["excerpts"][0]["id"]
    lines = ["0", "6", "4.5", "", "four", "4", "4", "4", "4", "4", "", "5", "5", "5", "5", "5", "", "="]
    stdout = io.StringIO()
    stdin = io.StringIO("".join(line + "\n" for line in lines))
    run_session(prepared, stdin=stdin, stdout=stdout, play=lambda _path: None)

    scores = read_json(prepared / "scores.json")
    assert scores["samples"][first]["A"]["naturalness"] == 4
    assert scores["samples"][first]["B"]["naturalness"] == 5
    assert scores["preferences"][first] == {"A/B": "="}
    assert stdout.getvalue().count("whole number from 1 to 5") == 5


def test_preference_must_name_a_label_or_equals(prepared: Path) -> None:
    blind = read_json(prepared / "blind.json")
    excerpt = blind["excerpts"][0]
    lines = _sample_script(excerpt["labels"], [], SCORES)
    lines += ["C", "a/b", "b"]
    _run(prepared, lines)
    scores = read_json(prepared / "scores.json")
    assert scores["preferences"][excerpt["id"]] == {"A/B": "B"}


def test_resume_after_interruption_keeps_completed_samples(prepared: Path) -> None:
    blind = read_json(prepared / "blind.json")
    full = _full_script(blind)
    first_session: list[Path] = []
    # Stop after the first sample is complete, before the second sample starts.
    assert _run(prepared, full[:6], first_session) is False
    assert read_json(prepared / "scores.json")["samples"].keys() == {blind["excerpts"][0]["id"]}

    second_session: list[Path] = []
    assert _run(prepared, full[6:], second_session) is True
    first_id = blind["excerpts"][0]["id"]
    assert prepared / "samples" / first_id / "A.wav" not in second_session
    assert len(set(first_session + second_session)) == 10
    assert len(read_json(prepared / "scores.json")["samples"]) == len(blind["excerpts"])


def test_partial_sample_is_never_saved(prepared: Path) -> None:
    blind = read_json(prepared / "blind.json")
    first = blind["excerpts"][0]["id"]
    assert _run(prepared, ["5", "4", "3"]) is False
    assert not (prepared / "scores.json").exists() or first not in read_json(prepared / "scores.json")["samples"]


def test_session_never_reads_the_identity_key(prepared: Path) -> None:
    (prepared / "key.json").write_text("this is not json", encoding="utf-8")
    blind = read_json(prepared / "blind.json")
    assert _run(prepared, _full_script(blind)) is True


def test_no_play_option_never_calls_the_player(
    prepared: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    blind = read_json(prepared / "blind.json")

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("afplay must not run under --no-play")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(line + "\n" for line in _full_script(blind))))
    assert main(["eval", "score", str(prepared), "--no-play"]) == 0
    assert "Sample A" in capsys.readouterr().out
    assert read_json(prepared / "scores.json")["preferences"]


def test_import_scores_merges_valid_rows(prepared: Path, tmp_path: Path) -> None:
    blind = read_json(prepared / "blind.json")
    rows = ["kind,excerpt_id,label,naturalness,phrasing,pronunciation,voice_stability,comfort,note,preference"]
    for excerpt in blind["excerpts"]:
        for label in excerpt["labels"]:
            rows.append(f"sample,{excerpt['id']},{label},5,4,4,3,5,from paper,")
        rows.append(f"pair,{excerpt['id']},A/B,,,,,,,B")
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    assert main(["eval", "import-scores", str(prepared), "--file", str(csv_path)]) == 0
    scores = read_json(prepared / "scores.json")
    first = blind["excerpts"][0]["id"]
    assert scores["samples"][first]["B"]["voice_stability"] == 3
    assert scores["samples"][first]["B"]["note"] == "from paper"
    assert scores["preferences"][first] == {"A/B": "B"}


def test_import_scores_rejects_the_whole_file_on_any_bad_row(
    prepared: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blind = read_json(prepared / "blind.json")
    excerpt = blind["excerpts"][0]
    rows = [
        "kind,excerpt_id,label,naturalness,phrasing,pronunciation,voice_stability,comfort,note,preference",
        f"sample,{excerpt['id']},A,9,4,4,3,5,,",
    ]
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    assert main(["eval", "import-scores", str(prepared), "--file", str(csv_path)]) == 2
    assert "naturalness" in capsys.readouterr().err
    assert not (prepared / "scores.json").exists()
