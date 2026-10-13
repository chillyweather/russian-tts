from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rutts.atomic import atomic_write_text
from rutts.validation import ValidationError

SCHEMA_VERSION = 1
BLIND_FILE = "blind.json"
KEY_FILE = "key.json"
SCORES_FILE = "scores.json"
REPORT_JSON = "report.json"
REPORT_MD = "report.md"
DECISION_JSON = "decision.json"
DECISION_MD = "decision.md"
SAMPLES_DIR = "samples"
TRIALS_DIR = "chapter_trials"
NOTES_DIR = "chapter_notes"
RESULTS_FILE = "results.json"
MANIFEST_FILE = "manifest.json"
CATEGORIES = ("naturalness", "phrasing", "pronunciation", "voice_stability", "comfort")
UNAVAILABLE = "unavailable"


def write_json(path: Path, document: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(document, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationError([f"file is missing: {path}"])
    try:
        document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError([f"file is corrupt: {path} ({exc})"]) from exc
    return document


def load_blind(eval_dir: Path) -> dict[str, Any]:
    return read_json(eval_dir / BLIND_FILE)


def load_key(eval_dir: Path) -> dict[str, Any]:
    return read_json(eval_dir / KEY_FILE)


def load_scores(eval_dir: Path) -> dict[str, Any]:
    path = eval_dir / SCORES_FILE
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "samples": {}, "preferences": {}}
    return read_json(path)


def pair_key(first: str, second: str) -> str:
    return "/".join(sorted([first, second]))
