from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rutts.book.project import BookProject, install_narrator, load_project, write_json
from rutts.book.text_import import import_text
from rutts.cli import main
from rutts.engines import base
from rutts.resources import UNAVAILABLE
from tests.conftest import FakeEngine
from tests.test_book_render import profile_dir  # noqa: F401  (shared fixture)

CH1 = "Первая фраза первой главы. Вторая фраза."
CH2A = (
    "Он шёл по длинной улице, которая тянулась от старого вокзала к реке, мимо облупленных домов с тёмными "
    "окнами и закрытых магазинов, где на витринах ещё висели выцветшие афиши прошлого лета. Он думал о письме, "
    "полученном накануне, о словах, которые в нём были написаны, и о человеке, который их написал и которого он "
    "не видел уже много лет."
)
CH2B = (
    "Каждый шаг казался тяжелее предыдущего, потому что дорога заканчивалась у старой пристани, где его ждали "
    "давно, с тех самых пор, как перестали верить в его возвращение. Теперь уже поздно что-либо менять, поздно "
    "говорить и поздно просить прощения у тех, кого он оставил в городе."
)
CH3 = "Третья глава. Конец."
BOOK_TEXT = f"""Глава 1. Утро

{CH1}

Глава 2. Дорога

{CH2A}

{CH2B}

Глава 3. Пристань

{CH3}

Глава 4. Архив

Этой главы не будет в книге.
"""


def calm_sample() -> dict[str, Any]:
    return {
        "peak_rss_bytes": 1024**3,
        "mlx_peak_memory_bytes": UNAVAILABLE,
        "swap_used_bytes": 0,
        "memory_free_percent": 60,
    }


@pytest.fixture(autouse=True)
def calm_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never depend on the real machine's memory state; pressure tests override this."""
    monkeypatch.setattr("rutts.resources.sample", calm_sample)


@pytest.fixture
def install_engine(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeEngine], None]:
    def install(engine: FakeEngine) -> None:
        monkeypatch.setitem(base.ENGINE_FACTORIES, "fake", lambda **options: engine)

    return install


@pytest.fixture
def job_book(tmp_path: Path, profile_dir: Path) -> BookProject:
    source = tmp_path / "book.txt"
    source.write_text(BOOK_TEXT, encoding="utf-8")
    out = tmp_path / "book"
    import_text(source, out, title="Книга", author="", chapter_regex=None)
    project = load_project(out)
    install_narrator(project, profile_dir)
    assert main(["book", "exclude", str(out), "--chapter", "c004", "--reason", "archive notes"]) == 0
    return project


def _generate(book: BookProject, *extra: str) -> int:
    return main(["book", "generate", str(book.directory), "--engine", "fake", "--target", "short", *extra])


def _job_file(book: BookProject, name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((book.directory / "job" / name).read_text(encoding="utf-8"))
    return document


def test_chapters_and_chunks_reach_the_engine_in_source_order(
    job_book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    engine = FakeEngine()
    install_engine(engine)
    assert _generate(job_book) == 0
    assert engine.received == [CH1, CH2A, CH2B, CH3]


def test_engine_is_loaded_once_for_the_whole_multi_chapter_job(
    job_book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    engine = FakeEngine()
    install_engine(engine)
    assert _generate(job_book) == 0
    assert engine.loads == 1
    assert engine.unloads == 1
    assert engine.loaded is False


def test_excluded_chapter_is_skipped_and_listed_with_its_reason(
    job_book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    engine = FakeEngine()
    install_engine(engine)
    assert _generate(job_book) == 0
    assert not any("Этой главы" in text for text in engine.received)
    coverage = _job_file(job_book, "coverage.json")
    assert coverage["complete"] is True
    excluded = [segment for segment in coverage["segments"] if not segment["included"]]
    assert excluded and {segment["chapter_id"] for segment in excluded} == {"c004"}
    assert {segment["exclusion_reason"] for segment in excluded} == {"archive notes"}
    assert coverage["totals"]["excluded"] == len(excluded)


def _records(book: BookProject, chapter_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = _job_file_path(book, chapter_id)["chunks"]
    return records


def _job_file_path(book: BookProject, chapter_id: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(
        (book.target_dir(chapter_id, "short") / "manifest.json").read_text(encoding="utf-8")
    )
    return document


def _state(book: BookProject) -> dict[str, Any]:
    return _job_file(book, "state.json")


def test_cancelled_job_resumes_across_the_chapter_boundary(
    job_book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    install_engine(FakeEngine(interrupt_on=frozenset({3})))
    assert _generate(job_book) == 130
    assert _state(job_book)["status"] == "interrupted"
    assert [record["status"] for record in _records(job_book, "c002")] == ["ok", "missing"]
    assert "resume" in capsys.readouterr().err

    resumed = FakeEngine()
    install_engine(resumed)
    assert _generate(job_book) == 0
    assert resumed.received == [CH2B, CH3]
    assert resumed.loads == 1
    state = _state(job_book)
    assert state["status"] == "completed"
    assert state["invocations"] == 2
    assert _job_file(job_book, "coverage.json")["complete"] is True


def test_failing_middle_chapter_is_incomplete_while_later_chapters_render(
    job_book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    engine = FakeEngine(fail_texts=frozenset({CH2A, CH2B}))
    install_engine(engine)
    assert _generate(job_book) == 1
    assert engine.received == [CH1, CH2A, CH2A, CH2A, CH2B, CH2B, CH2B, CH3]
    state = _state(job_book)
    assert state["status"] == "incomplete"
    assert state["chapters"]["c002"]["status"] == "incomplete"
    assert state["chapters"]["c002"]["failed_chunks"] == ["c002-k0001", "c002-k0002"]
    assert state["chapters"]["c003"]["status"] == "completed"
    assert [item["chunk"] for item in state["recent_failures"]] == ["c002-k0001", "c002-k0002"]
    out = capsys.readouterr().out
    assert "incomplete c002 Глава 2. Дорога: failed chunks c002-k0001, c002-k0002" in out

    coverage = _job_file(job_book, "coverage.json")
    assert coverage["complete"] is False
    assert coverage["uncovered_segments"] == ["c002-s0002", "c002-s0003"]
    assert coverage["totals"]["uncovered"] == 2


def test_retry_failed_later_completes_the_incomplete_chapter(
    job_book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    install_engine(FakeEngine(fail_texts=frozenset({CH2A, CH2B})))
    assert _generate(job_book) == 1

    untouched = FakeEngine()
    install_engine(untouched)
    assert _generate(job_book) == 1
    assert untouched.received == []

    retried = FakeEngine()
    install_engine(retried)
    assert _generate(job_book, "--retry-failed") == 0
    assert retried.received == [CH2A, CH2B]
    assert _state(job_book)["status"] == "completed"
    assert _job_file(job_book, "coverage.json")["complete"] is True


ALL_TEXTS = [CH1, CH2A, CH2B, CH3]


def test_fully_valid_book_never_loads_the_engine(
    job_book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    install_engine(FakeEngine())
    assert _generate(job_book) == 0
    again = FakeEngine()
    install_engine(again)
    assert _generate(job_book) == 0
    assert again.received == []
    assert again.loads == 0
    assert again.unloads == 0
    assert _state(job_book)["status"] == "completed"


def test_changed_configuration_is_refused_then_accepted_with_full_invalidation(
    job_book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    install_engine(FakeEngine(precision="bf16"))
    assert _generate(job_book) == 0

    refused = FakeEngine(precision="fp16")
    install_engine(refused)
    assert _generate(job_book, "--precision", "fp16") == 2
    err = capsys.readouterr().err
    assert "engine.precision: 'bf16' -> 'fp16'" in err
    assert refused.received == [] and refused.loads == 0

    accepted = FakeEngine(precision="fp16")
    install_engine(accepted)
    assert _generate(job_book, "--precision", "fp16", "--accept-config-change") == 0
    assert accepted.received == ALL_TEXTS
    job = _job_file(job_book, "job.json")
    assert job["engine"]["precision"] == "fp16"
    assert [entry["configuration"]["engine"]["precision"] for entry in job["superseded_configs"]] == ["bf16"]
    superseded = sorted(path.name for path in (job_book.target_dir("c002", "short") / "superseded").iterdir())
    assert superseded == ["c002-k0001.wav", "c002-k0002.wav"]


def test_time_left_is_shown_only_after_three_rendered_chunks(
    job_book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    install_engine(FakeEngine(interrupt_on=frozenset({3})))
    _generate(job_book)
    early = capsys.readouterr().out
    assert "c001-k0001" in early and "c002-k0001" in early
    assert "estimate" not in early
    assert main(["book", "job-status", str(job_book.directory)]) == 0
    assert "estimate" not in capsys.readouterr().out

    install_engine(FakeEngine())
    _generate(job_book)
    lines = [line for line in capsys.readouterr().out.splitlines() if " chunk " in line]
    # the first run rendered two chunks, so the first chunk of this run is the third rendered in the job
    assert lines[0].startswith("c002 ") and "c002-k0002" in lines[0]
    assert "estimate" in lines[0]
    assert "(estimate from observed throughput)" in lines[0]
    throughput = _state(job_book)["throughput"]
    assert throughput["rendered_chunks"] == 4
    assert throughput["audio_seconds"] > 0
    assert throughput["render_seconds_per_audio_second"] is not None


def test_memory_pressure_stops_with_exit_three_and_resumes_cleanly(
    job_book: BookProject, install_engine: Callable[[FakeEngine], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}

    def swapping() -> dict[str, Any]:
        calls["count"] += 1
        sample = calm_sample()
        if calls["count"] >= 5:
            sample["swap_used_bytes"] = 6 * 1024**3
        return sample

    monkeypatch.setattr("rutts.resources.sample", swapping)
    install_engine(FakeEngine())
    assert _generate(job_book) == 3

    state = _state(job_book)
    assert state["status"] == "stopped_resource_pressure"
    assert [record["status"] for record in _records(job_book, "c002")] == ["ok", "missing"]
    diagnostics = sorted((job_book.directory / "job" / "diagnostics").glob("*.json"))
    assert len(diagnostics) == 1
    report = json.loads(diagnostics[0].read_text(encoding="utf-8"))
    assert report["chunk"] == "c002-k0002"
    assert report["thresholds"]["max_swap_gb"] == 4.0
    assert report["samples"] and report["samples"][-1]["measurements"]["swap_used_bytes"] == 6 * 1024**3
    for chapter_id in ("c001", "c002"):
        for record in _records(job_book, chapter_id):
            if record["status"] == "ok":
                assert (job_book.target_dir(chapter_id, "short") / record["audio_file"]).is_file()

    monkeypatch.setattr("rutts.resources.sample", calm_sample)
    resumed = FakeEngine()
    install_engine(resumed)
    assert _generate(job_book) == 0
    assert resumed.received == [CH2B, CH3]


def test_engine_memory_error_stops_the_job_with_diagnostics(
    job_book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    install_engine(FakeEngine(memory_error_on=frozenset({3})))
    assert _generate(job_book) == 3
    assert _state(job_book)["status"] == "stopped_resource_pressure"
    report_path = next((job_book.directory / "job" / "diagnostics").glob("*.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "out of memory" in report["last_error"]
    assert report["chunk"] == "c002-k0002"
    assert [record["status"] for record in _records(job_book, "c002")] == ["ok", "missing"]


def test_unavailable_metrics_never_stop_the_job(
    job_book: BookProject, install_engine: Callable[[FakeEngine], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    unknown: dict[str, Any] = {
        "peak_rss_bytes": UNAVAILABLE,
        "mlx_peak_memory_bytes": UNAVAILABLE,
        "swap_used_bytes": UNAVAILABLE,
        "memory_free_percent": UNAVAILABLE,
    }
    monkeypatch.setattr("rutts.resources.sample", lambda: dict(unknown))
    install_engine(FakeEngine())
    assert _generate(job_book) == 0
    lines = (job_book.directory / "job" / "resources.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    first = json.loads(lines[0])
    assert "swap_used_bytes" in first["unavailable"]
    assert first["reasons"] == []


def test_project_passes_portability_and_survives_a_move(
    job_book: BookProject, install_engine: Callable[[FakeEngine], None], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    install_engine(FakeEngine())
    assert _generate(job_book, "--assemble") == 0
    assert main(["book", "check-portable", str(job_book.directory)]) == 0
    capsys.readouterr()

    moved = tmp_path / "elsewhere" / "moved-book"
    moved.parent.mkdir()
    shutil.move(str(job_book.directory), str(moved))
    assert main(["book", "check-portable", str(moved)]) == 0
    resumed = FakeEngine()
    install_engine(resumed)
    assert main(["book", "generate", str(moved), "--engine", "fake"]) == 0
    assert resumed.received == [] and resumed.loads == 0
    assert _state_at(moved)["status"] == "completed"
    assert _job_file_at(moved, "coverage.json")["complete"] is True


def _state_at(directory: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((directory / "job" / "state.json").read_text(encoding="utf-8"))
    return document


def _job_file_at(directory: Path, name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((directory / "job" / name).read_text(encoding="utf-8"))
    return document


def test_check_portable_reports_absolute_and_escaping_paths(job_book: BookProject, capsys: pytest.CaptureFixture[str]) -> None:
    write_json(job_book.directory / "job" / "leak.json", {"audio": "/Users/someone/cache/model.bin", "x_file": "../../outside.wav"})
    assert main(["book", "check-portable", str(job_book.directory)]) == 1
    out = capsys.readouterr().out
    assert "job/leak.json: audio stores an absolute path" in out
    assert "job/leak.json: x_file points outside the project" in out


def test_chapter_selection_runs_in_source_order_and_later_selection_completes_the_book(
    job_book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    from_second = FakeEngine()
    install_engine(from_second)
    assert _generate(job_book, "--from", "c002") == 1
    assert from_second.received == [CH2A, CH2B, CH3]
    assert _state(job_book)["chapters"]["c001"]["status"] == "pending"

    first = FakeEngine()
    install_engine(first)
    assert _generate(job_book, "--chapters", "c001") == 0
    assert first.received == [CH1]
    assert _state(job_book)["status"] == "completed"
