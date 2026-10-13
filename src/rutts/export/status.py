from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rutts.book.project import BookProject, read_json, sha256_file
from rutts.export.build import components, included_ids, record_path
from rutts.export.validate import ExpectedChapter, check, coverage_checks, m4b_checks, mp3_checks

LABELS = {
    "chapters": "chapter order or titles changed",
    "audio": "chapter audio or masters changed",
    "metadata": "book metadata changed",
    "settings": "processing settings changed",
}


@dataclass(frozen=True)
class Status:
    state: str  # missing | ready | stale | invalid
    changes: tuple[str, ...]
    record: dict[str, Any] | None


def export_status(project: BookProject, name: str, requested: dict[str, Any] | None = None) -> Status:
    """Compare an export with the project. ``requested`` holds settings to compare; None compares recorded settings."""
    path = record_path(project, name)
    if not path.is_file():
        return Status("missing", (), None)
    record = read_json(path)
    if record["status"] != "ready":
        return Status("invalid", tuple(item["name"] for item in record["validation"]["checks"] if not item["ok"]), record)
    recorded = record["input"]["components"]
    settings = requested if requested is not None else recorded["settings"]
    current = components(project, included_ids(project), settings["target"], settings)
    changes: list[str] = []
    for key, label in LABELS.items():
        if current[key] != recorded[key]:
            if key == "audio":
                changed = [cid for cid in current["audio"] if current["audio"][cid] != recorded["audio"].get(cid)]
                changes.append(f"{label}: {', '.join(changed)}")
            else:
                changes.append(label)
    for entry in record["files"]:
        file_path = project.directory / entry["path"]
        if not file_path.is_file() or sha256_file(file_path) != entry["sha256"]:
            changes.append(f"output file changed or missing: {entry['path']}")
    return Status("stale" if changes else "ready", tuple(changes), record)


def recheck_export(project: BookProject, record: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-run the validation gate against the files on disk for a recorded export."""
    rate = int(record["processing"]["sample_rate"]["output"])
    chapters = [ExpectedChapter(item["title"], item["start_ms"] / 1000) for item in record["chapters"]]
    total = sum(int(item["samples"]) for item in record["chapters"]) / rate
    ids = [item["id"] for item in record["chapters"]]
    target = str(record["target"])
    checks: list[dict[str, Any]] = []
    metadata = record["metadata"]
    mp3_number = 0
    for entry in record["files"]:
        path = project.directory / entry["path"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            checks.append(check(f"{entry['path']} unchanged since export", False, "file changed or missing"))
            continue
        if entry["format"] == "m4b":
            checks += m4b_checks(path, chapters, total, metadata["title"], metadata["author"])
        elif entry["format"] == "mp3":
            mp3_number += 1
            checks += mp3_checks(path, chapters[mp3_number - 1].title, metadata["album"], metadata["author"], mp3_number, len(chapters))
    checks += coverage_checks(project, ids, target, reviewed=record["kind"] == "reviewed")
    return checks
