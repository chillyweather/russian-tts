from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from rutts.book.chunking import limits_for_target, plan_chunks
from rutts.book.project import NARRATION_FILE, BookProject, NarrationDocument, SourceDocument, read_json, write_json
from rutts.pronunciation.apply import (
    KINDS,
    MATCH_MODES,
    DictionaryEntry,
    NarrationBuild,
    OverrideEntry,
    build_narration,
    compile_dictionary_entry,
    remap_overrides,
    resolve_span,
)
from rutts.pronunciation.markup import CONVENTIONS, read_markup, require_valid_replacement, write_markup
from rutts.validation import ValidationError

DICTIONARY_FILE = "narration/dictionary.json"
OVERRIDES_FILE = "narration/overrides.json"
HISTORY_FILE = "narration/history.json"
TARGETS = ("short", "long")
ID_PREFIXES = {"dictionary": "d", "override": "o"}


@dataclass
class Corrections:
    dictionary: list[dict[str, Any]] = field(default_factory=list)
    overrides: list[dict[str, Any]] = field(default_factory=list)
    markup: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> Corrections:
        return copy.deepcopy(self)


def _optional_entries(directory: Any, name: str) -> list[dict[str, Any]]:
    path = directory / name
    return list(read_json(path)["entries"]) if path.is_file() else []


def load_corrections(project: BookProject) -> Corrections:
    directory = project.directory
    return Corrections(
        dictionary=_optional_entries(directory, DICTIONARY_FILE),
        overrides=_optional_entries(directory, OVERRIDES_FILE),
        markup=read_markup(directory),
        history=_optional_entries(directory, HISTORY_FILE),
    )


def _write_corrections(project: BookProject, corrections: Corrections) -> None:
    write_json(project.directory / DICTIONARY_FILE, {"schema_version": 1, "entries": corrections.dictionary})
    write_json(project.directory / OVERRIDES_FILE, {"schema_version": 1, "entries": corrections.overrides})
    write_markup(project.directory, corrections.markup)
    write_json(project.directory / HISTORY_FILE, {"schema_version": 1, "entries": corrections.history})


def _collection(corrections: Corrections, target: str) -> list[dict[str, Any]]:
    return {"dictionary": corrections.dictionary, "override": corrections.overrides, "markup": corrections.markup}[target]


def _state(corrections: Corrections, target: str, entry_id: str) -> dict[str, Any] | None:
    return next((item for item in _collection(corrections, target) if item["id"] == entry_id), None)


def _put(corrections: Corrections, target: str, entry_id: str, state: dict[str, Any] | None) -> None:
    collection = _collection(corrections, target)
    collection[:] = [item for item in collection if item["id"] != entry_id]
    if state is not None:
        collection.append(dict(state))
        collection.sort(key=lambda item: str(item["id"]))


def _next_id(corrections: Corrections, target: str) -> str:
    """Ids are never reused, so the highest id ever recorded for the target is the reference."""
    used = [int(str(item["id"])[1:]) for item in _collection(corrections, target)]
    for record in corrections.history:
        if record["target"] != target:
            continue
        for key in ("before", "after"):
            if record[key] is not None:
                used.append(int(str(record[key]["id"])[1:]))
    return f"{ID_PREFIXES[target]}{max(used, default=0) + 1:03d}"


def _record(
    corrections: Corrections,
    action: str,
    target: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    reverts_seq: int | None = None,
) -> None:
    corrections.history.append(
        {
            "seq": len(corrections.history) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "action": action,
            "target": target,
            "entry": after if after is not None else before,
            "reverts_seq": reverts_seq,
            "before": before,
            "after": after,
        }
    )


def _change(corrections: Corrections, action: str, target: str, entry_id: str, state: dict[str, Any] | None) -> None:
    before = _state(corrections, target, entry_id)
    _put(corrections, target, entry_id, state)
    _record(corrections, action, target, before, state)


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValidationError([f"unknown {name} '{value}' (use one of: {', '.join(choices)})"])


def add_dictionary(
    corrections: Corrections,
    *,
    pattern: str,
    replacement: str,
    match: str = "word",
    kind: str = "other",
    note: str = "",
    case_sensitive: bool = False,
) -> DictionaryEntry:
    if not pattern.strip():
        raise ValidationError(["dictionary pattern is empty"])
    if not replacement.strip():
        raise ValidationError(["dictionary replacement is empty"])
    _check_choice("match", match, MATCH_MODES)
    _check_choice("kind", kind, KINDS)
    require_valid_replacement(replacement)
    entry = DictionaryEntry(_next_id(corrections, "dictionary"), pattern, replacement, match, case_sensitive, kind, note)
    compile_dictionary_entry(entry)
    _change(corrections, "add", "dictionary", entry.id, asdict(entry))
    return entry


def add_override(
    corrections: Corrections,
    source: SourceDocument,
    *,
    segment_id: str,
    find: str,
    occurrence: int | None,
    replacement: str,
    kind: str = "other",
    note: str = "",
) -> OverrideEntry:
    if not replacement.strip():
        raise ValidationError(["override replacement is empty"])
    _check_choice("kind", kind, KINDS)
    require_valid_replacement(replacement)
    start, end, segment = resolve_span(source, segment_id, find, occurrence)
    entry = OverrideEntry(
        id=_next_id(corrections, "override"),
        segment_id=segment_id,
        start=start,
        end=end,
        original=find,
        replacement=replacement,
        kind=kind,
        note=note,
        location={"line_start": segment.line_start, "line_end": segment.line_end},
    )
    _change(corrections, "add", "override", entry.id, asdict(entry))
    return entry


def remove_entry(corrections: Corrections, target: str, entry_id: str) -> None:
    if _state(corrections, target, entry_id) is None:
        raise ValidationError([f"no {target} entry with id {entry_id}"])
    _change(corrections, "remove", target, entry_id, None)


def set_markup(corrections: Corrections, engine_id: str, convention: str, *, verified_by_listening: bool) -> None:
    _check_choice("convention", convention, CONVENTIONS)
    if convention != "none" and not verified_by_listening:
        raise ValidationError(
            [
                f"convention '{convention}' may only be enabled after listening to markup-test output; "
                "pass --verified-by-listening"
            ]
        )
    state = {"id": engine_id, "convention": convention, "verified_by_listening": verified_by_listening}
    _change(corrections, "add", "markup", engine_id, state)


def revert(corrections: Corrections, seq: int) -> None:
    matches = [record for record in corrections.history if record["seq"] == seq]
    if not matches:
        raise ValidationError([f"no history entry #{seq}"])
    original = matches[0]
    target = str(original["target"])
    entry_id = str(original["entry"]["id"])
    current = _state(corrections, target, entry_id)
    _put(corrections, target, entry_id, original["before"])
    _record(corrections, "revert", target, current, original["before"], reverts_seq=seq)


def _build(project: BookProject, corrections: Corrections) -> NarrationBuild:
    """Build narration from source; overrides are remapped in ``corrections`` first."""
    source = project.read_source()
    overrides = remap_overrides(source, [OverrideEntry.from_dict(item) for item in corrections.overrides])
    corrections.overrides = [asdict(item) for item in overrides]
    return build_narration(
        source,
        [DictionaryEntry.from_dict(item) for item in corrections.dictionary],
        overrides,
    )


def preview(project: BookProject, corrections: Corrections) -> NarrationBuild:
    """Narration that ``corrections`` would produce; nothing is written."""
    return _build(project, corrections.snapshot())


def commit(project: BookProject, corrections: Corrections) -> NarrationBuild:
    """Persist corrections and rebuild narration.json from source."""
    build = _build(project, corrections)
    _write_corrections(project, corrections)
    write_json(project.directory / NARRATION_FILE, build.document.to_dict())
    return build


def rebuild_narration(project: BookProject) -> NarrationBuild:
    return commit(project, load_corrections(project))


def _planned_texts(document: NarrationDocument, source: SourceDocument, chapter_id: str, target: str) -> dict[str, str]:
    chapter = source.chapter(chapter_id)
    if not chapter.included:
        return {}
    narration = {segment.id: segment for segment in document.segments}
    segments = [narration[segment.id] for segment in chapter.segments if segment.included]
    return {chunk.id: chunk.text for chunk in plan_chunks(segments, target, limits_for_target(target))}


def invalidated_chunks(project: BookProject, old: NarrationDocument, new: NarrationDocument) -> dict[str, list[str]]:
    """Chunk ids per target whose planned narration text differs between two narration documents."""
    source = project.read_source()
    result: dict[str, list[str]] = {}
    for target in TARGETS:
        changed: list[str] = []
        for chapter in source.chapters:
            before = _planned_texts(old, source, chapter.id, target)
            after = _planned_texts(new, source, chapter.id, target)
            changed.extend(
                chunk_id for chunk_id in sorted(set(before) | set(after)) if before.get(chunk_id) != after.get(chunk_id)
            )
        result[target] = changed
    return result
