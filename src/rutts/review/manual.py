from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from rutts.book.project import BookProject, read_json, write_json
from rutts.review.decisions import require_reason
from rutts.validation import ValidationError

MANUAL_FILE = "review/manual.json"
SCHEMA_VERSION = 1
KINDS = ("stress", "pronunciation", "prosody", "join", "voice", "other")
STATES = ("resolved", "dismissed")


def load_manual(project: BookProject) -> list[dict[str, Any]]:
    path = project.directory / MANUAL_FILE
    return list(read_json(path)["findings"]) if path.is_file() else []


def _save(project: BookProject, findings: list[dict[str, Any]]) -> None:
    write_json(project.directory / MANUAL_FILE, {"schema_version": SCHEMA_VERSION, "findings": findings})


def _find(project: BookProject, finding_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings = load_manual(project)
    for finding in findings:
        if finding["id"] == finding_id:
            return findings, finding
    raise ValidationError([f"no manual finding {finding_id}"])


def word_span(text: str, word: str) -> tuple[int, int]:
    matches = list(re.finditer(rf"(?<!\w){re.escape(word)}(?!\w)", text))
    if not matches:
        raise ValidationError([f"word {word!r} is not in the chunk's narration text"])
    return matches[0].start(), matches[0].end()


def flag(
    project: BookProject,
    *,
    chapter_id: str,
    target: str,
    chunk_id: str,
    kind: str,
    note: str,
    audio_sha256: str | None,
    chunk_fingerprint: str | None,
    chunk_text: str,
    word: str | None = None,
    time: float | None = None,
) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValidationError([f"unknown kind '{kind}' (use one of: {', '.join(KINDS)})"])
    if word and time is not None:
        raise ValidationError(["pass either --word or --time, not both"])
    text = require_reason(note)
    finding: dict[str, Any] = {
        "id": f"m{len(load_manual(project)) + 1:03d}",
        "chapter_id": chapter_id,
        "target": target,
        "chunk_id": chunk_id,
        "kind": kind,
        "note": text,
        "char_start": None,
        "char_end": None,
        "word": None,
        "time": None,
        "audio_sha256": audio_sha256,
        "chunk_fingerprint": chunk_fingerprint,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "state": "open",
        "reason": None,
    }
    if word:
        start, end = word_span(chunk_text, word)
        finding.update(word=word, char_start=start, char_end=end)
    if time is not None:
        if time < 0:
            raise ValidationError(["--time must not be negative"])
        finding["time"] = float(time)
    findings = load_manual(project)
    findings.append(finding)
    _save(project, findings)
    return finding


def set_state(project: BookProject, finding_id: str, state: str, reason: str | None) -> dict[str, Any]:
    if state not in STATES:
        raise ValidationError([f"unknown state '{state}'"])
    text = require_reason(reason)
    findings, finding = _find(project, finding_id)
    finding.update(state=state, reason=text)
    _save(project, findings)
    return finding


def reopen(project: BookProject, finding_id: str) -> dict[str, Any]:
    findings, finding = _find(project, finding_id)
    finding.update(state="open", reason=None)
    _save(project, findings)
    return finding


def needs_relisten(finding: dict[str, Any], current_audio_sha256: str | None) -> bool:
    """Open findings whose chunk audio changed since they were flagged; they stay open until closed explicitly."""
    return bool(finding["state"] == "open" and finding["audio_sha256"] != current_audio_sha256)
