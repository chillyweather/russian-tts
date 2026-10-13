from __future__ import annotations

import difflib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rutts.book.assemble import planned_chunks
from rutts.book.chunking import TARGET_SECONDS
from rutts.book.project import CHAPTERS_DIR, BookProject, read_json, sha256_text
from rutts.book.render import MANIFEST_FILE, PLAN_FILE, SUPERSEDED_DIR
from rutts.review import decisions as decision_store
from rutts.review import manual as manual_store
from rutts.validation import ValidationError
from rutts.verification.verify import findings_dir, load_findings

CONTENT_FAILURE_TYPES = frozenset(
    {
        "omission",
        "addition",
        "substitution",
        "repetition",
        "missing_audio",
        "missing_chunk",
        "content_failure",
        "transcription_failed",
    }
)
MISSING_CHUNK = "missing_chunk"
CONTENT_FAILURE = "content_failure"
OPEN_STATUSES = frozenset({"open", "stale"})


@dataclass(frozen=True)
class QueueItem:
    id: str
    lineage: str
    origin: str
    chapter_id: str
    target: str
    chunk_id: str
    type: str
    severity: str
    expected: str
    heard: str
    reason: str
    chars: tuple[int, int] | None
    audio_seconds: tuple[float, float] | None
    audio_file: str | None
    audio_sha256: str | None
    chunk_fingerprint: str | None
    verifier_fingerprint: str | None
    current: bool
    engine_id: str
    segment_ids: tuple[str, ...]
    source_text: str
    narration_text: str
    engine_text: str
    master_offset_seconds: float | None
    history: dict[str, Any]
    note: str = ""
    status: str = "open"
    decision: dict[str, Any] | None = None
    needs_relisten: bool = False

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES

    @property
    def is_content_failure(self) -> bool:
        return self.type in CONTENT_FAILURE_TYPES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Chapter:
    project: BookProject
    chapter_id: str
    target: str
    target_dir: Path
    records: dict[str, dict[str, Any]]
    chunk_order: list[str]
    plan: dict[str, dict[str, Any]]
    engine_texts: dict[str, str]
    source_segments: dict[str, str]
    offsets: dict[str, float]
    engine_id: str


def _digest(*parts: Any) -> str:
    return sha256_text(json.dumps(list(parts), ensure_ascii=False, separators=(",", ":")))[:12]


def marked_narration(source: str, narration: str) -> str:
    """Narration text with words that differ from the source wrapped in [[ ]]."""
    if not source or source == narration:
        return narration
    before, after = source.split(), narration.split()
    out: list[str] = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if op == "equal":
            out += after[j1:j2]
        elif j2 > j1:
            out.append("[[" + " ".join(after[j1:j2]) + "]]")
    return " ".join(out)


def scopes(project: BookProject, chapters: Sequence[str] | None, target: str | None) -> list[tuple[str, str]]:
    """Rendered (chapter, target) pairs, in book order."""
    source = project.read_source()
    chapter_ids = list(chapters) if chapters else [chapter.id for chapter in source.chapters if chapter.included]
    pairs: list[tuple[str, str]] = []
    for chapter_id in chapter_ids:
        source.chapter(chapter_id)
        targets = [target] if target else list(TARGET_SECONDS)
        for name in targets:
            if (project.target_dir(chapter_id, name) / MANIFEST_FILE).is_file():
                pairs.append((chapter_id, name))
    return pairs


def _master_offsets(project: BookProject, chapter_id: str, target: str) -> dict[str, float]:
    path = project.directory / CHAPTERS_DIR / f"{chapter_id}.json"
    if not path.is_file():
        return {}
    report = read_json(path)
    if report["target"] != target:
        return {}
    return {str(entry["id"]): float(entry["start_seconds"]) for entry in report["chunks"] if entry["start_seconds"] is not None}


def _load_chapter(project: BookProject, chapter_id: str, target: str) -> _Chapter:
    target_dir = project.target_dir(chapter_id, target)
    manifest = read_json(target_dir / MANIFEST_FILE)
    records = {str(record["id"]): record for record in manifest["chunks"]}
    plan = {str(chunk["id"]): chunk for chunk in read_json(target_dir / PLAN_FILE)["chunks"]}
    engine_texts = {chunk.id: chunk.text for chunk in planned_chunks(project, chapter_id, target, manifest)}
    source = project.read_source().chapter(chapter_id)
    return _Chapter(
        project=project,
        chapter_id=chapter_id,
        target=target,
        target_dir=target_dir,
        records=records,
        chunk_order=list(records),
        plan=plan,
        engine_texts=engine_texts,
        source_segments={segment.id: segment.text for segment in source.segments},
        offsets=_master_offsets(project, chapter_id, target),
        engine_id=str(manifest["engine"]["engine_id"]),
    )


def _segment_ids(chapter: _Chapter, chunk_id: str) -> tuple[str, ...]:
    return tuple(str(item) for item in chapter.plan.get(chunk_id, {}).get("narration_segment_ids", []))


def _texts(chapter: _Chapter, chunk_id: str) -> tuple[str, str, str]:
    planned = chapter.plan.get(chunk_id, {})
    narration = str(planned.get("text", ""))
    source = " ".join(
        chapter.source_segments[segment_id]
        for segment_id in planned.get("narration_segment_ids", [])
        if segment_id in chapter.source_segments
    )
    return source, marked_narration(source, narration), chapter.engine_texts.get(chunk_id, "")


def _history(chapter: _Chapter, record: dict[str, Any]) -> dict[str, Any]:
    superseded_dir = chapter.target_dir / SUPERSEDED_DIR
    superseded = sorted(path.name for path in superseded_dir.glob(f"{record['id']}.*")) if superseded_dir.is_dir() else []
    return {
        "attempts": list(record.get("attempts", [])),
        "events": list(record.get("history", [])),
        "superseded_audio": superseded,
    }


def _chunk_state(chapter: _Chapter, record: dict[str, Any]) -> str | None:
    """'failed' or 'missing' when the chunk has no valid audio for its current narration; None when it is usable."""
    if record["status"] == "failed":
        return "failed"
    if record["status"] != "ok":
        return "missing"
    if sha256_text(chapter.engine_texts.get(str(record["id"]), "")) != record.get("text_sha256"):
        return "missing"
    return None


def chunk_states(project: BookProject, chapter_id: str, target: str) -> dict[str, str | None]:
    chapter = _load_chapter(project, chapter_id, target)
    return {chunk_id: _chunk_state(chapter, record) for chunk_id, record in chapter.records.items()}


def _chunk_item(chapter: _Chapter, record: dict[str, Any], state: str) -> QueueItem:
    chunk_id = str(record["id"])
    source, narration, engine_text = _texts(chapter, chunk_id)
    if state == "failed":
        kind = CONTENT_FAILURE
        attempts = record.get("attempts", [])
        error = attempts[-1].get("error") if attempts else None
        reason = f"failed after {len(attempts)} attempt(s)" + (f": {error}" if error else "")
    else:
        kind = MISSING_CHUNK
        reason = "no valid audio for the current narration; run book regenerate"
    audio_sha = record.get("audio_sha256")
    return QueueItem(
        id=_digest(chapter.chapter_id, chapter.target, chunk_id, kind, audio_sha),
        lineage=_digest(chapter.chapter_id, chapter.target, chunk_id, kind, "chunk"),
        origin="chunk",
        chapter_id=chapter.chapter_id,
        target=chapter.target,
        chunk_id=chunk_id,
        type=kind,
        severity="major",
        expected=source,
        heard="",
        reason=reason,
        chars=None,
        audio_seconds=None,
        audio_file=record.get("audio_file"),
        audio_sha256=audio_sha,
        chunk_fingerprint=record.get("fingerprint"),
        verifier_fingerprint=None,
        current=True,
        engine_id=chapter.engine_id,
        segment_ids=_segment_ids(chapter, chunk_id),
        source_text=source,
        narration_text=narration,
        engine_text=engine_text,
        master_offset_seconds=None,
        history=_history(chapter, record),
    )


def _finding_items(chapter: _Chapter) -> list[QueueItem]:
    if not findings_dir(chapter.project, chapter.chapter_id, chapter.target).is_dir():
        return []
    items: list[QueueItem] = []
    for document in load_findings(chapter.project, chapter.chapter_id, chapter.target):
        record = chapter.records.get(str(document["chunk_id"]))
        if record is None or _chunk_state(chapter, record) is not None:
            continue
        current = (
            document.get("audio_sha256") == record.get("audio_sha256")
            and document.get("chunk_fingerprint") == record.get("fingerprint")
        )
        chunk_id = str(document["chunk_id"])
        source, narration, engine_text = _texts(chapter, chunk_id)
        for finding in document["findings"]:
            location = finding["location"]
            chars = (int(location["chars"][0]), int(location["chars"][1])) if location.get("chars") else None
            seconds = (float(location["audio_seconds"][0]), float(location["audio_seconds"][1])) if location.get("audio_seconds") else None
            lineage = _digest(chapter.chapter_id, chapter.target, chunk_id, finding["type"], chars, seconds)
            offset = chapter.offsets.get(chunk_id)
            items.append(
                QueueItem(
                    id=_digest(lineage, document.get("audio_sha256")),
                    lineage=lineage,
                    origin="automatic",
                    chapter_id=chapter.chapter_id,
                    target=chapter.target,
                    chunk_id=chunk_id,
                    type=str(finding["type"]),
                    severity=str(finding["severity"]),
                    expected=str(finding.get("expected", "")),
                    heard=str(finding.get("heard", "")),
                    reason=_reason(finding),
                    chars=chars,
                    audio_seconds=seconds,
                    audio_file=record.get("audio_file"),
                    audio_sha256=document.get("audio_sha256"),
                    chunk_fingerprint=document.get("chunk_fingerprint"),
                    verifier_fingerprint=document.get("verifier_fingerprint"),
                    current=current,
                    engine_id=chapter.engine_id,
                    segment_ids=_segment_ids(chapter, chunk_id),
                    source_text=source,
                    narration_text=narration,
                    engine_text=engine_text,
                    master_offset_seconds=None if offset is None or seconds is None else round(offset + seconds[0], 3),
                    history=_history(chapter, record),
                )
            )
    return items


def _reason(finding: dict[str, Any]) -> str:
    evidence = finding.get("evidence") or {}
    detail = evidence.get("reason") if isinstance(evidence, dict) else None
    text = f"{finding['type']}: expected {finding.get('expected', '')!r}, heard {finding.get('heard', '')!r}"
    return f"{text}; {detail}" if detail else text


def _manual_items(chapter: _Chapter) -> list[QueueItem]:
    items: list[QueueItem] = []
    for finding in manual_store.load_manual(chapter.project):
        if finding["chapter_id"] != chapter.chapter_id or finding["target"] != chapter.target:
            continue
        chunk_id = str(finding["chunk_id"])
        record = chapter.records.get(chunk_id, {})
        source, narration, engine_text = _texts(chapter, chunk_id)
        time = finding.get("time")
        chars = (finding["char_start"], finding["char_end"]) if finding.get("char_start") is not None else None
        items.append(
            QueueItem(
                id=str(finding["id"]),
                lineage=str(finding["id"]),
                origin="manual",
                chapter_id=chapter.chapter_id,
                target=chapter.target,
                chunk_id=chunk_id,
                type=str(finding["kind"]),
                severity="manual",
                expected=str(finding.get("word") or ""),
                heard="",
                reason=str(finding["note"]),
                chars=chars,
                audio_seconds=None if time is None else (float(time), float(time)),
                audio_file=record.get("audio_file"),
                audio_sha256=finding.get("audio_sha256"),
                chunk_fingerprint=finding.get("chunk_fingerprint"),
                verifier_fingerprint=None,
                current=True,
                engine_id=chapter.engine_id,
                segment_ids=_segment_ids(chapter, chunk_id),
                source_text=source,
                narration_text=narration,
                engine_text=engine_text,
                master_offset_seconds=None,
                history=_history(chapter, record) if record else {"attempts": [], "events": [], "superseded_audio": []},
                note=str(finding["note"]),
                status=str(finding["state"]),
                needs_relisten=manual_store.needs_relisten(finding, record.get("audio_sha256")),
            )
        )
    return items


def _with_decision(item: QueueItem, decisions: dict[str, dict[str, Any]]) -> QueueItem:
    if item.origin == "manual":
        return item
    decision = decisions.get(item.lineage)
    if not item.current:
        return replace(item, status="open", decision=decision)
    if decision is None:
        return item
    if decision_store.decision_is_valid(decision, item):
        return replace(item, status=str(decision["state"]), decision=decision)
    return replace(item, status="stale", decision=decision)


def _chapter_items(chapter: _Chapter) -> list[QueueItem]:
    items: list[QueueItem] = []
    for chunk_id in chapter.chunk_order:
        record = chapter.records[chunk_id]
        state = _chunk_state(chapter, record)
        if state is not None:
            items.append(_chunk_item(chapter, record, state))
    items += _finding_items(chapter)
    items += _manual_items(chapter)
    return items


def build_queue(
    project: BookProject, chapters: Sequence[str] | None = None, target: str | None = None
) -> list[QueueItem]:
    """Review items derived from disk state; only decisions and manual findings are stored."""
    decisions = decision_store.load_decisions(project)
    items: list[QueueItem] = []
    for chapter_id, chapter_target in scopes(project, chapters, target):
        chapter = _load_chapter(project, chapter_id, chapter_target)
        items += [_with_decision(item, decisions) for item in _chapter_items(chapter)]
    return items


def find_item(project: BookProject, item_id: str) -> QueueItem:
    for item in build_queue(project):
        if item.id == item_id:
            return item
    raise ValidationError([f"no review item {item_id}; run book review list to see current items"])
