from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rutts.book.project import NarrationDocument, NarrationSegment, SourceDocument, SourceSegment
from rutts.validation import ValidationError

MATCH_MODES = ("word", "phrase", "regex")
KINDS = ("name", "stress", "numeral", "date", "abbreviation", "initials", "foreign", "other")
CONTEXT_CHARS = 30


@dataclass(frozen=True)
class DictionaryEntry:
    id: str
    pattern: str
    replacement: str
    match: str
    case_sensitive: bool
    kind: str
    note: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DictionaryEntry:
        return cls(
            id=str(data["id"]),
            pattern=str(data["pattern"]),
            replacement=str(data["replacement"]),
            match=str(data["match"]),
            case_sensitive=bool(data["case_sensitive"]),
            kind=str(data["kind"]),
            note=str(data["note"]),
        )


@dataclass(frozen=True)
class OverrideEntry:
    id: str
    segment_id: str
    start: int
    end: int
    original: str
    replacement: str
    kind: str
    note: str
    location: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OverrideEntry:
        return cls(
            id=str(data["id"]),
            segment_id=str(data["segment_id"]),
            start=int(data["start"]),
            end=int(data["end"]),
            original=str(data["original"]),
            replacement=str(data["replacement"]),
            kind=str(data["kind"]),
            note=str(data["note"]),
            location=dict(data["location"]),
        )


@dataclass(frozen=True)
class Change:
    origin: str
    entry_id: str
    segment_id: str
    start: int
    end: int
    original: str
    replacement: str
    kind: str
    before: str
    after: str

    @property
    def key(self) -> tuple[str, str, int, int, str]:
        return (self.origin, self.entry_id, self.start, self.end, self.replacement)

    def describe(self) -> str:
        return (
            f"{self.segment_id}: {self.before}[{self.original} -> {self.replacement}]{self.after} "
            f"({self.origin} {self.entry_id}, {self.kind})"
        )


@dataclass(frozen=True)
class Stale:
    override_id: str
    segment_id: str
    reason: str

    def describe(self) -> str:
        return f"stale override {self.override_id} ({self.segment_id}): {self.reason}"


@dataclass(frozen=True)
class NarrationBuild:
    document: NarrationDocument
    changes: list[Change]
    stale: list[Stale]


def _word_pattern(body: str) -> str:
    return rf"(?<!\w){body}(?!\w)"


def compile_dictionary_entry(entry: DictionaryEntry) -> re.Pattern[str]:
    flags = 0 if entry.case_sensitive else re.IGNORECASE
    if entry.match == "regex":
        body = entry.pattern
    elif entry.match in ("word", "phrase"):
        words = entry.pattern.split() if entry.match == "phrase" else [entry.pattern]
        body = _word_pattern(r"\s+".join(re.escape(word) for word in words))
    else:
        raise ValidationError([f"unknown match mode: {entry.match}"])
    try:
        return re.compile(body, flags)
    except re.error as exc:
        raise ValidationError([f"dictionary pattern {entry.pattern!r} is not a valid regex: {exc}"]) from exc


def _overlaps(start: int, end: int, claimed: list[tuple[int, int]]) -> bool:
    return any(start < other_end and other_start < end for other_start, other_end in claimed)


def _context(text: str, start: int, end: int) -> tuple[str, str]:
    before = text[max(0, start - CONTEXT_CHARS) : start]
    after = text[end : end + CONTEXT_CHARS]
    return ("…" if start - CONTEXT_CHARS > 0 else "") + before, after + ("…" if end + CONTEXT_CHARS < len(text) else "")


def _override_problem(override: OverrideEntry, segment: SourceSegment | None) -> str | None:
    if segment is None:
        return "its source segment no longer exists"
    if not 0 <= override.start < override.end <= len(segment.text):
        return "its span is outside the source segment"
    if segment.text[override.start : override.end] != override.original:
        return f"source text at this span is no longer {override.original!r}"
    return None


def build_narration(
    source: SourceDocument,
    dictionary: list[DictionaryEntry],
    overrides: list[OverrideEntry],
) -> NarrationBuild:
    """Derive narration from source: overrides claim their spans first, then dictionary matches fill the rest."""
    segments = {segment.id: segment for chapter in source.chapters for segment in chapter.segments}
    compiled = [(entry, compile_dictionary_entry(entry)) for entry in dictionary]
    changes: list[Change] = []
    stale: list[Stale] = []
    narration: list[NarrationSegment] = []
    for chapter in source.chapters:
        for segment in chapter.segments:
            claimed: list[tuple[int, int]] = []
            found: list[tuple[int, int, str, str, str, str, str]] = []
            for cut_start, cut_end in segment.narration_cuts:
                claimed.append((cut_start, cut_end))
                found.append((cut_start, cut_end, segment.text[cut_start:cut_end], "", "footnote", f"{segment.id}-marker", "footnote"))
            for override in overrides:
                if override.segment_id != segment.id:
                    continue
                problem = _override_problem(override, segments.get(override.segment_id))
                if problem is None and _overlaps(override.start, override.end, claimed):
                    problem = "its span overlaps another override"
                if problem is not None:
                    stale.append(Stale(override.id, override.segment_id, problem))
                    continue
                claimed.append((override.start, override.end))
                found.append((override.start, override.end, segment.text[override.start : override.end], override.replacement, "override", override.id, override.kind))
            for entry, pattern in compiled:
                for match in pattern.finditer(segment.text):
                    if match.end() == match.start() or _overlaps(match.start(), match.end(), claimed):
                        continue
                    claimed.append((match.start(), match.end()))
                    found.append((match.start(), match.end(), match.group(0), entry.replacement, "dictionary", entry.id, entry.kind))
            found.sort()
            pieces: list[str] = [segment.narration_prefix]
            position = 0
            for start, end, original, replacement, origin, entry_id, kind in found:
                before, after = _context(segment.text, start, end)
                changes.append(Change(origin, entry_id, segment.id, start, end, original, replacement, kind, before, after))
                pieces.append(segment.text[position:start])
                pieces.append(replacement)
                position = end
            pieces.append(segment.text[position:])
            narration.append(
                NarrationSegment(
                    id=segment.id,
                    source_segment_id=segment.id,
                    chapter_id=chapter.id,
                    kind=segment.kind,
                    text="".join(pieces),
                )
            )
    return NarrationBuild(NarrationDocument(segments=narration), changes, stale)


def _location_key(location: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, repr(value)) for key, value in location.items()))


def remap_overrides(source: SourceDocument, overrides: list[OverrideEntry]) -> list[OverrideEntry]:
    """Follow each override's source segment by its line location after splits and merges."""
    by_location = {
        _location_key(segment.location): segment.id
        for chapter in source.chapters
        for segment in chapter.segments
    }
    remapped: list[OverrideEntry] = []
    for override in overrides:
        segment_id = by_location.get(_location_key(override.location), override.segment_id)
        remapped.append(OverrideEntry(**{**override.__dict__, "segment_id": segment_id}))
    return remapped


def resolve_span(source: SourceDocument, segment_id: str, find: str, occurrence: int | None) -> tuple[int, int, SourceSegment]:
    """Locate ``find`` inside one source segment; ambiguity needs an explicit occurrence number."""
    chapter_index, segment_index = source.segment_location(segment_id)
    segment = source.chapters[chapter_index].segments[segment_index]
    if not find:
        raise ValidationError(["--find must not be empty"])
    starts = [index for index in range(len(segment.text)) if segment.text.startswith(find, index)]
    if not starts:
        raise ValidationError([f"{find!r} does not occur in {segment_id}"])
    if occurrence is None:
        if len(starts) > 1:
            raise ValidationError(
                [f"{find!r} occurs {len(starts)} times in {segment_id}; pass --occurrence 1..{len(starts)}"]
            )
        occurrence = 1
    if not 1 <= occurrence <= len(starts):
        raise ValidationError([f"--occurrence {occurrence} is out of range; {find!r} occurs {len(starts)} time(s)"])
    start = starts[occurrence - 1]
    return start, start + len(find), segment


def format_changes(changes: list[Change]) -> list[str]:
    return [change.describe() for change in changes]
