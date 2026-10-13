from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rutts.book.project import install_narrator, load_project
from rutts.book.text_import import import_text
from rutts.cli import main
from tests.eval_support import ENGINE_A, ENGINE_B, read_json, two_candidates  # noqa: F401

Build = Callable[..., tuple[Path, Path]]
FIXTURE = Path(__file__).parent / "fixtures" / "book_sample.txt"
CSV_HEADER = "kind,excerpt_id,label,naturalness,phrasing,pronunciation,voice_stability,comfort,note,preference"
PASSING = {"naturalness": 5, "phrasing": 4, "pronunciation": 4, "voice_stability": 4, "comfort": 4}
MIDDLING = {"naturalness": 4, "phrasing": 4, "pronunciation": 4, "voice_stability": 4, "comfort": 4}
FLAT_THREE = {"naturalness": 3, "phrasing": 3, "pronunciation": 3, "voice_stability": 3, "comfort": 3}


def _prepare(candidates: list[Path], out: Path) -> None:
    assert main(["eval", "prepare", "--candidates", *map(str, candidates), "--out", str(out), "--seed", "9"]) == 0


def _score(eval_dir: Path, tmp_path: Path, values: dict[int, dict[str, int]], winner: int = 0) -> None:
    """Score every sample from the candidate's scores; the candidate ``winner`` is preferred in each pair."""
    blind = read_json(eval_dir / "blind.json")
    key = read_json(eval_dir / "key.json")
    rows: list[str] = []
    for excerpt in blind["excerpts"]:
        owner_of = {
            item["label"]: item["candidate"] for item in key["samples"] if item["excerpt_id"] == excerpt["id"]
        }
        for label in excerpt["labels"]:
            score = values[owner_of[label]]
            rows.append(
                f"sample,{excerpt['id']},{label},{score['naturalness']},{score['phrasing']},"
                f"{score['pronunciation']},{score['voice_stability']},{score['comfort']},,"
            )
        if len(excerpt["labels"]) == 2:
            preferred = next(label for label, owner in owner_of.items() if owner == winner)
            rows.append(f"pair,{excerpt['id']},A/B,,,,,,,{preferred}")
    csv_path = tmp_path / f"scores-{eval_dir.name}.csv"
    csv_path.write_text("\n".join([CSV_HEADER, *rows]) + "\n", encoding="utf-8")
    assert main(["eval", "import-scores", str(eval_dir), "--file", str(csv_path)]) == 0


def _decide(eval_dir: Path, *extra: str) -> int:
    return main(["eval", "decide", str(eval_dir), *extra])


def _decision(eval_dir: Path) -> dict[str, Any]:
    return read_json(eval_dir / "decision.json")


def _render_and_trial(book: Path, engine: str, precision: str, label: str, out: Path, force: bool) -> None:
    command = ["book", "render", str(book), "--chapter", "c001", "--engine", engine, "--precision", precision, "--target", "short"]
    if force:
        command.append("--force-new-run")
    assert main(command) == 0
    assert main(["book", "assemble", str(book), "--chapter", "c001", "--target", "short"]) == 0
    assert main(
        ["eval", "chapter-trial", str(book), "--chapter", "c001", "--target", "short", "--label", label, "--out", str(out)]
    ) == 0


@pytest.fixture
def evaluation(two_candidates: Build, tmp_path: Path) -> Path:
    """Two candidates, prepared, with a chapter trial for each rendered by the real book pipeline."""
    a, b = two_candidates()
    out = tmp_path / "eval"
    _prepare([a, b], out)
    book = tmp_path / "book"
    import_text(FIXTURE, book, title="Книга", author="", chapter_regex=None)
    install_narrator(load_project(book), tmp_path / "narrator")
    _render_and_trial(book, ENGINE_A, "bf16", "quiet", out, force=False)
    _render_and_trial(book, ENGINE_B, "fp16", "loud", out, force=True)
    return out


def test_refuses_with_fewer_than_two_candidates(two_candidates: Build, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, _ = two_candidates()
    out = tmp_path / "eval"
    _prepare([a], out)
    _score(out, tmp_path, {0: PASSING})

    assert _decide(out) == 1
    decision = _decision(out)
    assert decision["status"] == "no_decision"
    assert any("fewer than two candidates are feasible" in gap for gap in decision["gaps"])
    assert "OmniVoice" in decision["next_step"] and "F5-TTS" in decision["next_step"]
    assert "no decision" in capsys.readouterr().err


def test_refuses_when_scores_are_missing(two_candidates: Build, tmp_path: Path) -> None:
    a, b = two_candidates()
    out = tmp_path / "eval"
    _prepare([a, b], out)

    assert _decide(out) == 1
    decision = _decision(out)
    assert decision["status"] == "no_decision"
    assert any("scores are missing" in gap for gap in decision["gaps"])
    assert not (out / "decision.md").exists()


def test_refuses_when_no_chapter_trial_exists_for_the_top_two(two_candidates: Build, tmp_path: Path) -> None:
    a, b = two_candidates()
    out = tmp_path / "eval"
    _prepare([a, b], out)
    _score(out, tmp_path, {0: PASSING, 1: MIDDLING})

    assert _decide(out) == 1
    decision = _decision(out)
    assert decision["status"] == "no_decision"
    assert sum("no chapter trial exists" in gap for gap in decision["gaps"]) == 2


def test_refuses_when_no_candidate_passes_the_quality_gates(evaluation: Path, tmp_path: Path) -> None:
    _score(evaluation, tmp_path, {0: FLAT_THREE, 1: FLAT_THREE})

    assert _decide(evaluation) == 1
    decision = _decision(evaluation)
    assert decision["status"] == "no_decision"
    assert any("none passes the quality gates" in gap for gap in decision["gaps"])
    assert "naturalness mean 3.0 is below 4" in decision["gaps"][0]


def test_decision_succeeds_with_passing_scores_and_chapter_trials(evaluation: Path, tmp_path: Path) -> None:
    _score(evaluation, tmp_path, {0: PASSING, 1: MIDDLING}, winner=0)

    assert _decide(evaluation) == 0
    decision = _decision(evaluation)
    assert decision["status"] == "decided"
    assert decision["gaps"] == []
    assert decision["target_revision"] is None

    chosen = decision["chosen"]
    assert chosen["candidate"] == 0
    assert chosen["model_repo"] == "fake/model"
    assert chosen["model_revision"] == "0" * 40
    assert chosen["precision"] == "bf16"
    assert chosen["runtime_version"] == "1.0"
    assert chosen["settings"] == {"tone_hz": 440}
    assert chosen["narrator"]["provenance"] == "Test recording, public domain"
    candidate_dir = Path(read_json(evaluation / "key.json")["candidates"][0]["directory"])
    results = read_json(candidate_dir / "results.json")
    assert chosen["narrator"]["reference_sha256"] == results["narrator"]["reference_sha256"]
    assert decision["evidence"]["chapter_trial"]["label"] == "quiet"
    assert decision["evidence"]["means"]["naturalness"] == 5.0
    assert decision["evidence"]["pairwise"] == [{"opponent": 1, "wins": 5, "losses": 0, "ties": 0}]
    assert decision["remaining_limitations"]

    markdown = (evaluation / "decision.md").read_text(encoding="utf-8")
    assert "fake/model" in markdown
    assert "## Remaining limitations" in markdown
    assert "Test recording, public domain" in markdown


def test_target_revision_requires_the_explicit_flag(evaluation: Path, tmp_path: Path) -> None:
    _score(evaluation, tmp_path, {0: FLAT_THREE, 1: FLAT_THREE})
    assert _decide(evaluation) == 1

    reason = "Listener accepts 3/5 naturalness for this suite; revisit after the second round"
    assert _decide(evaluation, "--approve-target-revision", reason) == 0
    decision = _decision(evaluation)
    assert decision["status"] == "decided"
    assert decision["target_revision"]["reason"] == reason
    assert "naturalness" in " ".join(decision["target_revision"]["waived_gates"])
    assert reason in (evaluation / "decision.md").read_text(encoding="utf-8")


def test_empty_target_revision_reason_is_refused(evaluation: Path, tmp_path: Path) -> None:
    _score(evaluation, tmp_path, {0: FLAT_THREE, 1: FLAT_THREE})
    assert _decide(evaluation, "--approve-target-revision", "  ") == 2
    assert not (evaluation / "decision.json").exists()
    assert not (evaluation / "decision.md").exists()
