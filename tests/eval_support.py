from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rutts.cli import main
from rutts.engines import base
from rutts.narrator import create_profile
from tests.conftest import FakeEngine, write_reference_wav

TRANSCRIPT = "Это тестовая фраза для опорной записи."

EXCERPTS = [
    {"id": "prose_morning", "category": "prose", "text": "Утро выдалось тихим и прохладным. Над рекой поднимался лёгкий туман."},
    {"id": "dialogue_evening", "category": "dialogue", "text": "— Ты придёшь сегодня вечером? — спросила она. — Не знаю, — ответил он."},
    {"id": "question_truth", "category": "questions", "text": "Почему же никто не сказал ему правду? Неужели все так считали?"},
    {"id": "numerals_birth", "category": "numerals", "text": "Он родился 3 марта 1954 года в 7 часов 45 минут утра."},
    {"id": "names_meeting", "category": "names", "text": "Анна Сергеевна Белова встретила Фёдора Игнатьевича у вокзала."},
]


def write_excerpts(path: Path, items: list[dict[str, str]] | None = None) -> Path:
    path.write_text(json.dumps({"version": 1, "excerpts": items or EXCERPTS}, ensure_ascii=False), encoding="utf-8")
    return path


def make_narrator(tmp_path: Path, reference_wav: Path) -> Path:
    directory = tmp_path / "narrator"
    create_profile(
        directory,
        name="Fixture narrator",
        audio=reference_wav,
        transcript=TRANSCRIPT,
        provenance="Test recording, public domain",
    )
    return directory


def register_fake(monkeypatch: pytest.MonkeyPatch, name: str, **options: Any) -> None:
    """Register an engine id whose instances are FakeEngines built with ``options``."""

    def factory(**overrides: Any) -> FakeEngine:
        return FakeEngine(**{**options, **overrides})

    monkeypatch.setitem(base.ENGINE_FACTORIES, name, factory)


def run_preview_cli(
    engine: str,
    precision: str,
    narrator: Path,
    excerpts: Path,
    out: Path,
    retries: int = 1,
) -> None:
    code = main(
        [
            "preview", "--engine", engine, "--precision", precision, "--narrator", str(narrator),
            "--excerpts", str(excerpts), "--out", str(out), "--retries", str(retries),
        ]
    )
    assert code in (0, 1), f"preview exited {code}"


def read_json(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


ENGINE_A = "fake_quiet"
ENGINE_B = "fake_loud"


@pytest.fixture
def two_candidates(
    tmp_path: Path, reference_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., tuple[Path, Path]]:
    """Return a factory building two real preview directories with different fingerprints and loudness."""
    narrator = make_narrator(tmp_path, reference_wav)
    excerpts = write_excerpts(tmp_path / "excerpts.json")
    register_fake(monkeypatch, ENGINE_A, precision="bf16", amplitude=0.05)
    register_fake(monkeypatch, ENGINE_B, precision="fp16", amplitude=0.4)

    def build(first: Path | None = None, second: Path | None = None) -> tuple[Path, Path]:
        a = first or tmp_path / "preview-a"
        b = second or tmp_path / "preview-b"
        run_preview_cli(ENGINE_A, "bf16", narrator, excerpts, a)
        run_preview_cli(ENGINE_B, "fp16", narrator, excerpts, b)
        return a, b

    return build
