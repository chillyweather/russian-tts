from __future__ import annotations

import csv
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from rutts.evaluation.common import (
    CATEGORIES,
    SAMPLES_DIR,
    SCHEMA_VERSION,
    SCORES_FILE,
    load_blind,
    load_scores,
    pair_key,
    write_json,
)
from rutts.validation import ValidationError

SCORE_HINT = "enter a whole number from 1 to 5"
IMPORT_COLUMNS = ("kind", "excerpt_id", "label", *CATEGORIES, "note", "preference")


class _Stopped(Exception):
    """The listener ended the session or the input stream closed."""


def afplay(path: Path) -> None:
    subprocess.run(["afplay", str(path)], check=False)


def _parse_score(text: str) -> int:
    if not re.fullmatch(r"[1-5]", text):
        raise ValueError(SCORE_HINT)
    return int(text)


def _parse_preference(text: str, labels: list[str]) -> str:
    value = text.upper()
    if value == "=" or value in labels:
        return value
    raise ValueError(f"enter {', '.join(labels)} or = for no preference")


def _ask(stdin: TextIO, stdout: TextIO, prompt: str, parse: Callable[[str], Any]) -> Any:
    while True:
        stdout.write(prompt)
        stdout.flush()
        line = stdin.readline()
        if line == "":
            raise _Stopped()
        try:
            return parse(line.strip())
        except ValueError as exc:
            stdout.write(f"  {exc}\n")


def _save(eval_dir: Path, scores: dict[str, Any]) -> None:
    write_json(
        eval_dir / SCORES_FILE,
        {"schema_version": SCHEMA_VERSION, "samples": scores["samples"], "preferences": scores["preferences"]},
    )


def run_session(
    eval_dir: Path,
    *,
    stdin: TextIO,
    stdout: TextIO,
    play: Callable[[Path], None] | None,
) -> bool:
    """Ask for scores and preferences; returns True only when every sample and pair is scored."""
    blind = load_blind(eval_dir)
    scores = load_scores(eval_dir)
    scores.setdefault("samples", {})
    scores.setdefault("preferences", {})
    excerpts = blind["excerpts"]
    try:
        for position, excerpt in enumerate(excerpts, start=1):
            excerpt_id = excerpt["id"]
            stdout.write(f"\n[{position}/{len(excerpts)}] {excerpt_id} ({excerpt['category']})\n")
            stdout.write(f"{excerpt['text']}\n")
            for label in excerpt["labels"]:
                if label in scores["samples"].get(excerpt_id, {}):
                    continue
                sample = eval_dir / SAMPLES_DIR / excerpt_id / f"{label}.wav"
                stdout.write(f"Sample {label}: {sample}\n")
                if play is not None:
                    play(sample)
                record: dict[str, Any] = {}
                for category in CATEGORIES:
                    record[category] = _ask(stdin, stdout, f"  {category} (1-5): ", _parse_score)
                record["note"] = _ask(stdin, stdout, "  note (Enter to skip): ", lambda text: text)
                scores["samples"].setdefault(excerpt_id, {})[label] = record
                _save(eval_dir, scores)
            pairs = [pair["labels"] for pair in blind["pairs"] if pair["excerpt_id"] == excerpt_id]
            for first, second in pairs:
                key = pair_key(first, second)
                if key in scores["preferences"].get(excerpt_id, {}):
                    continue
                options = [first, second]
                preference = _ask(
                    stdin,
                    stdout,
                    f"  preference {first} or {second} (= for none): ",
                    lambda text: _parse_preference(text, options),
                )
                scores["preferences"].setdefault(excerpt_id, {})[key] = preference
                _save(eval_dir, scores)
    except (_Stopped, KeyboardInterrupt):
        stdout.write("\nsession stopped; completed answers are saved and the session can be resumed\n")
        return False
    stdout.write("\nall samples and preferences are scored\n")
    return True


def import_scores(eval_dir: Path, csv_path: Path) -> dict[str, int]:
    """Merge scores written down elsewhere; the whole file is refused if any row is invalid."""
    blind = load_blind(eval_dir)
    scores = load_scores(eval_dir)
    if not csv_path.is_file():
        raise ValidationError([f"scores file not found: {csv_path}"])
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in IMPORT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValidationError([f"scores file is missing column(s): {', '.join(missing)}"])
        rows = list(reader)

    excerpts = {item["id"]: item for item in blind["excerpts"]}
    problems: list[str] = []
    new_samples: dict[str, dict[str, Any]] = {}
    new_preferences: dict[str, dict[str, str]] = {}
    for number, row in enumerate(rows, start=2):
        where = f"line {number}"
        excerpt_id = (row["excerpt_id"] or "").strip()
        label = (row["label"] or "").strip().upper()
        kind = (row["kind"] or "").strip().lower()
        if excerpt_id not in excerpts:
            problems.append(f"{where}: unknown excerpt '{excerpt_id}'")
            continue
        if kind == "sample":
            if label not in excerpts[excerpt_id]["labels"]:
                problems.append(f"{where}: '{label}' is not a sample of {excerpt_id}")
                continue
            if label in scores["samples"].get(excerpt_id, {}) or label in new_samples.get(excerpt_id, {}):
                problems.append(f"{where}: {excerpt_id} {label} is already scored")
                continue
            values: dict[str, Any] = {}
            for category in CATEGORIES:
                raw = (row[category] or "").strip()
                try:
                    values[category] = _parse_score(raw)
                except ValueError:
                    problems.append(f"{where}: {category} must be a whole number from 1 to 5, got '{raw}'")
            if all(category in values for category in CATEGORIES):
                values["note"] = (row["note"] or "").strip()
                new_samples.setdefault(excerpt_id, {})[label] = values
        elif kind == "pair":
            parts = label.split("/")
            options = [part for part in parts if part]
            valid_pairs = {
                pair_key(*pair["labels"]) for pair in blind["pairs"] if pair["excerpt_id"] == excerpt_id
            }
            if len(options) != 2 or pair_key(*options) not in valid_pairs:
                problems.append(f"{where}: '{label}' is not a comparison pair of {excerpt_id}")
                continue
            key = pair_key(*options)
            if key in scores["preferences"].get(excerpt_id, {}) or key in new_preferences.get(excerpt_id, {}):
                problems.append(f"{where}: {excerpt_id} {key} is already scored")
                continue
            preference = (row["preference"] or "").strip().upper()
            try:
                new_preferences.setdefault(excerpt_id, {})[key] = _parse_preference(preference, options)
            except ValueError as exc:
                problems.append(f"{where}: {exc}")
        else:
            problems.append(f"{where}: kind must be 'sample' or 'pair', got '{kind}'")
    if problems:
        raise ValidationError(problems)

    for excerpt_id, labelled in new_samples.items():
        scores["samples"].setdefault(excerpt_id, {}).update(labelled)
    for excerpt_id, pairs in new_preferences.items():
        scores["preferences"].setdefault(excerpt_id, {}).update(pairs)
    _save(eval_dir, scores)
    return {
        "samples": sum(len(labelled) for labelled in new_samples.values()),
        "preferences": sum(len(pairs) for pairs in new_preferences.values()),
    }
