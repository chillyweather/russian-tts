from __future__ import annotations

import json
import re
from pathlib import Path
from collections.abc import Callable
from typing import Any, TextIO

from rutts.book.project import CHAPTERS_DIR, BookProject, read_json, sha256_file, sha256_text
from rutts.book.render import MANIFEST_FILE
from rutts.validation import ValidationError

SCORE_FIELDS = ("naturalness", "sustained_comfort", "pronunciation", "completeness", "voice_continuity")
SCORE_LABELS = {
    "naturalness": "narration naturalness",
    "sustained_comfort": "sustained comfort",
    "pronunciation": "pronunciation",
    "completeness": "completeness",
    "voice_continuity": "voice continuity",
}
PLAYBACK_CHECKS = {
    "opened": "did the M4B open",
    "metadata_correct": "was the metadata correct (title, author, chapter titles)",
    "chapter_navigation": "did chapter navigation work",
    "audio_played": "did the audio play without missing sections",
}
# Unresolved findings of these types in the acceptance chapter block the listening gate (PRD section 6).
GATED_TYPES = frozenset({"pronunciation", "omission", "addition", "repetition", "join"})
SCORES_DIR = "acceptance/listening"
INPUT_ENDED = "input ended before every answer was given; nothing was recorded"


def gate_failures(scores: dict[str, int]) -> list[str]:
    """PRD gates: naturalness and sustained comfort at least 4, no category below 3."""
    failures: list[str] = []
    for name in ("naturalness", "sustained_comfort"):
        if scores[name] < 4:
            failures.append(f"{name} {scores[name]} is below 4")
    for name in SCORE_FIELDS:
        if scores[name] < 3:
            failures.append(f"{name} {scores[name]} is below 3")
    return failures


def _score(text: str) -> int:
    if not re.fullmatch(r"[1-5]", text):
        raise ValueError("enter a whole number from 1 to 5")
    return int(text)


def _yes_no(text: str) -> bool:
    value = text.lower()
    if value in ("y", "yes"):
        return True
    if value in ("n", "no"):
        return False
    raise ValueError("answer y or n")


def _text(text: str) -> str:
    return text


def _non_empty(text: str) -> str:
    if not text.strip():
        raise ValueError("enter a non-empty value")
    return text.strip()


def _ask(stdin: TextIO, stdout: TextIO, prompt: str, parse: Callable[[str], Any]) -> Any:
    while True:
        stdout.write(prompt)
        stdout.flush()
        line = stdin.readline()
        if line == "":
            raise ValidationError([INPUT_ENDED])
        try:
            return parse(line.strip())
        except ValueError as exc:
            stdout.write(f"  {exc}\n")


def ask_scores(stdin: TextIO, stdout: TextIO) -> dict[str, Any]:
    stdout.write("Score each category from 1 (poor) to 5 (excellent) after listening to the whole chapter.\n")
    scores = {name: _ask(stdin, stdout, f"  {SCORE_LABELS[name]} (1-5): ", _score) for name in SCORE_FIELDS}
    listened = _ask(stdin, stdout, "  did you listen to the whole chapter without stopping? (y/n): ", _yes_no)
    notes = _ask(stdin, stdout, "  notes (Enter to skip): ", _text)
    return {"scores": scores, "listened_whole_chapter": listened, "notes": notes}


def _valid_score(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5


def validate_answers(document: object) -> list[str]:
    if not isinstance(document, dict):
        return ["scores file must be a JSON object"]
    problems: list[str] = []
    scores = document.get("scores")
    if not isinstance(scores, dict):
        problems.append("scores must be an object with the five categories")
    else:
        for name in SCORE_FIELDS:
            if not _valid_score(scores.get(name)):
                problems.append(f"{name} must be a whole number from 1 to 5")
        problems += [f"unknown category {key!r}" for key in scores if key not in SCORE_FIELDS]
    if not isinstance(document.get("listened_whole_chapter"), bool):
        problems.append("listened_whole_chapter must be true or false")
    if not isinstance(document.get("notes", ""), str):
        problems.append("notes must be text")
    return problems


def load_scores_file(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError([f"scores file {path} is not usable: {exc}"]) from exc
    problems = validate_answers(document)
    if problems:
        raise ValidationError(problems)
    return {
        "scores": {name: int(document["scores"][name]) for name in SCORE_FIELDS},
        "listened_whole_chapter": document["listened_whole_chapter"],
        "notes": document.get("notes", ""),
    }


def ask_playback(stdin: TextIO, stdout: TextIO, player: str) -> dict[str, Any]:
    version = _ask(stdin, stdout, f"version of {player} (type it): ", _non_empty)
    checks = {
        key: _ask(stdin, stdout, f"  {question}? (y/n): ", _yes_no) for key, question in PLAYBACK_CHECKS.items()
    }
    return {"player": player, "player_version": version, "checks": checks}


def chapter_signature(project: BookProject, chapter_id: str, target: str) -> str | None:
    """Hash of the chapter's chunk audio and master; a listening review is bound to it."""
    manifest_path = project.target_dir(chapter_id, target) / MANIFEST_FILE
    if not manifest_path.is_file():
        return None
    records = read_json(manifest_path)["chunks"]
    master_sha: str | None = None
    report_path = project.directory / CHAPTERS_DIR / f"{chapter_id}.json"
    if report_path.is_file():
        master = project.directory / read_json(report_path)["master_file"]
        master_sha = sha256_file(master) if master.is_file() else None
    chunks = [[record["id"], record["audio_sha256"] if record["status"] == "ok" else None] for record in records]
    return sha256_text(json.dumps({"master": master_sha, "chunks": chunks}, sort_keys=True))


def copy_scores_file(project: BookProject, chapter_id: str, source: Path) -> dict[str, str]:
    """Keep a copy of the user's scores file inside the record, with its original hash."""
    relative = f"listening/{chapter_id}-scores.json"
    destination = project.directory / SCORES_DIR / f"{chapter_id}-scores.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    return {"path": relative, "sha256": sha256_file(destination), "source_name": source.name}
