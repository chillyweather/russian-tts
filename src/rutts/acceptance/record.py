from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from rutts.book.project import BookProject, read_json, write_json

RECORD_FILE = "acceptance/record.json"
REPORT_FILE = "acceptance/acceptance.md"
SCHEMA_VERSION = 1
EVIDENCED = "evidenced"
NOT_EVIDENCED = "not_evidenced"
FAILED = "failed"
SECTIONS = ("configuration", "offline", "coverage", "listening", "exercises", "resources", "output")
ACCEPTED = "accepted"
INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class Section:
    status: str
    evidence: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def overall(sections: dict[str, Section]) -> str:
    statuses = [sections[name].status for name in SECTIONS]
    if FAILED in statuses:
        return FAILED
    if all(status == EVIDENCED for status in statuses):
        return ACCEPTED
    return INCOMPLETE


def empty_inputs() -> dict[str, Any]:
    return {"offline": [], "listening": None, "playback": None}


def load_inputs(project: BookProject) -> dict[str, Any]:
    path = project.directory / RECORD_FILE
    if not path.is_file():
        return empty_inputs()
    return dict(read_json(path)["inputs"])


def save(project: BookProject, sections: dict[str, Section], inputs: dict[str, Any]) -> dict[str, Any]:
    """Persist the inputs and the freshly evaluated sections; returns the record document."""
    stamp = now()
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": stamp,
        "overall": overall(sections),
        "sections": {
            name: {
                "status": section.status,
                "evidence": section.evidence,
                "missing": section.missing,
                "updated_at": stamp,
            }
            for name, section in sections.items()
        },
        "inputs": inputs,
    }
    write_json(project.directory / RECORD_FILE, document)
    return document

