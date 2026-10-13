from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rutts.book.project import install_narrator, load_project
from rutts.book.text_import import import_text
from rutts.cli import main
from tests.conftest import write_reference_wav
from tests.eval_support import make_narrator, read_json, register_fake, write_json

FIXTURE = Path(__file__).parent / "fixtures" / "book_sample.txt"
MEMORY_KEYS = ("peak_rss_bytes", "mlx_peak_memory_bytes", "swap_used_bytes", "memory_free_percent")


@pytest.fixture
def book(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    narrator = make_narrator(tmp_path, write_reference_wav(tmp_path / "source.wav"))
    directory = tmp_path / "book"
    import_text(FIXTURE, directory, title="Книга", author="", chapter_regex=None)
    install_narrator(load_project(directory), narrator)
    register_fake(monkeypatch, "fake_a", precision="bf16")
    return directory


def _render(book: Path, engine: str, precision: str, *, force: bool = False) -> int:
    command = [
        "book", "render", str(book), "--chapter", "c001", "--engine", engine,
        "--precision", precision, "--target", "short",
    ]
    if force:
        command.append("--force-new-run")
    return main(command)


def _assemble(book: Path) -> int:
    return main(["book", "assemble", str(book), "--chapter", "c001", "--target", "short"])


def _trial(book: Path, out: Path, label: str, target: str = "short") -> int:
    return main(
        ["eval", "chapter-trial", str(book), "--chapter", "c001", "--target", target, "--label", label, "--out", str(out)]
    )


def _manifest(book: Path) -> dict[str, Any]:
    return read_json(book / "chunks" / "c001" / "short" / "manifest.json")


def test_trial_records_measured_audio_render_time_and_derived_rates(book: Path, tmp_path: Path) -> None:
    assert _render(book, "fake_a", "bf16") == 0
    assert _assemble(book) == 0
    out = tmp_path / "eval"
    assert _trial(book, out, "fast") == 0

    manifest = _manifest(book)
    ok = [chunk for chunk in manifest["chunks"] if chunk["status"] == "ok"]
    finished = sum(chunk["actual_duration_seconds"] for chunk in ok)
    rendered = sum(chunk["render_seconds"] for chunk in manifest["chunks"])
    trial = read_json(out / "chapter_trials" / "fast.json")

    assert trial["label"] == "fast"
    assert trial["chapter_id"] == "c001"
    assert trial["target"] == "short"
    assert trial["engine_fingerprint"] == manifest["engine_fingerprint"]
    assert trial["chunks"] == len(manifest["chunks"]) == len(ok)
    assert trial["finished_audio_seconds"] == pytest.approx(finished)
    assert trial["finished_audio_seconds"] == pytest.approx(0.5 * len(ok))
    assert trial["render_seconds"] == pytest.approx(rendered)
    assert trial["retries"] == 0
    assert trial["failed_chunks"] == []
    assert trial["assembly_complete"] is True
    assert trial["completed"] is True
    hours = trial["finished_audio_seconds"] / 3600
    assert trial["render_hours_per_audio_hour"] == pytest.approx(rendered / trial["finished_audio_seconds"])
    assert trial["retries_per_audio_hour"] == pytest.approx(0 / hours)


def test_trial_reports_memory_observations_from_the_manifest(book: Path, tmp_path: Path) -> None:
    assert _render(book, "fake_a", "bf16") == 0
    assert _assemble(book) == 0
    out = tmp_path / "eval"
    assert _trial(book, out, "fast") == 0

    chunks = [chunk for chunk in _manifest(book)["chunks"] if chunk["status"] == "ok"]
    trial = read_json(out / "chapter_trials" / "fast.json")
    peak = max(chunk["resources"]["peak_rss_bytes"] for chunk in chunks)
    assert trial["memory"]["peak_rss_bytes"] == peak
    assert trial["memory"]["peak_rss_bytes"] > 0
    mlx_values = [chunk["resources"]["mlx_peak_memory_bytes"] for chunk in chunks]
    assert trial["memory"]["mlx_peak_memory_bytes"] == (
        max(mlx_values) if all(isinstance(value, int) for value in mlx_values) else "unavailable"
    )
    assert set(trial["memory"]) == set(MEMORY_KEYS)


def test_metrics_missing_from_the_manifest_are_unavailable(book: Path, tmp_path: Path) -> None:
    assert _render(book, "fake_a", "bf16") == 0
    assert _assemble(book) == 0
    manifest = _manifest(book)
    for chunk in manifest["chunks"]:
        chunk.pop("resources", None)
    write_json(book / "chunks" / "c001" / "short" / "manifest.json", manifest)

    out = tmp_path / "eval"
    assert _trial(book, out, "bare") == 0
    trial = read_json(out / "chapter_trials" / "bare.json")
    assert trial["memory"] == {key: "unavailable" for key in MEMORY_KEYS}


def test_failed_chunk_makes_the_trial_incomplete(book: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    register_fake(monkeypatch, "fake_c", precision="bf16", raise_on=frozenset({2, 3, 4}))
    assert _render(book, "fake_c", "bf16") == 1
    assert _assemble(book) == 1
    out = tmp_path / "eval"
    assert _trial(book, out, "flaky") == 0

    manifest = _manifest(book)
    failed = [chunk["id"] for chunk in manifest["chunks"] if chunk["status"] == "failed"]
    assert failed
    trial = read_json(out / "chapter_trials" / "flaky.json")
    assert trial["failed_chunks"] == failed
    assert trial["assembly_complete"] is False
    assert trial["completed"] is False


def test_trial_refuses_a_target_that_was_not_rendered(book: Path, tmp_path: Path) -> None:
    assert _render(book, "fake_a", "bf16") == 0
    assert _assemble(book) == 0
    assert _trial(book, tmp_path / "eval", "wrong", target="long") == 2


def test_trial_refuses_to_overwrite_a_label(book: Path, tmp_path: Path) -> None:
    assert _render(book, "fake_a", "bf16") == 0
    assert _assemble(book) == 0
    out = tmp_path / "eval"
    assert _trial(book, out, "fast") == 0
    assert _trial(book, out, "fast") == 2


def test_notes_record_user_findings_for_a_trial(book: Path, tmp_path: Path) -> None:
    assert _render(book, "fake_a", "bf16") == 0
    assert _assemble(book) == 0
    out = tmp_path / "eval"
    assert _trial(book, out, "fast") == 0

    code = main(
        [
            "eval", "chapter-notes", str(out), "--label", "fast",
            "--stress-errors", "2", "--omissions", "0", "--distracting-joins", "1",
            "--notes", "Stress on ‘замок’ wrong in chunk 4",
        ]
    )
    assert code == 0
    notes = read_json(out / "chapter_notes" / "fast.json")
    assert notes["label"] == "fast"
    assert notes["stress_errors"] == 2
    assert notes["omissions"] == 0
    assert notes["distracting_joins"] == 1
    assert notes["notes"] == "Stress on ‘замок’ wrong in chunk 4"


def test_notes_require_counts_and_an_existing_trial(book: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _render(book, "fake_a", "bf16") == 0
    assert _assemble(book) == 0
    out = tmp_path / "eval"
    assert _trial(book, out, "fast") == 0
    with pytest.raises(SystemExit):
        main(["eval", "chapter-notes", str(out), "--label", "fast", "--stress-errors", "1"])
    assert main(
        [
            "eval", "chapter-notes", str(out), "--label", "unknown",
            "--stress-errors", "0", "--omissions", "0", "--distracting-joins", "0",
        ]
    ) == 2
    assert "no chapter trial" in capsys.readouterr().err
    assert not (out / "chapter_notes" / "unknown.json").exists()
    assert json.loads((out / "chapter_trials" / "fast.json").read_text(encoding="utf-8"))["label"] == "fast"
