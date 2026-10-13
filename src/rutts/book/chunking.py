from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from rutts.book.project import NarrationSegment
from rutts.validation import ValidationError

TARGET_SECONDS: dict[str, tuple[float, float]] = {"short": (20.0, 40.0), "long": (45.0, 90.0)}
DEFAULT_MAX_CHARS = 600
CHARS_PER_SECOND = 14.0
SENTENCE_END = re.compile(r"[.!?…]+[»\"”)]*\s+")
CLAUSE_MARKERS = (";:", "—", ",")
OPENERS = "«“"
CLOSERS = "»”"


@dataclass(frozen=True)
class ChunkLimits:
    max_chars: int = DEFAULT_MAX_CHARS
    chars_per_second: float = CHARS_PER_SECOND

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkLimits:
        return cls(max_chars=int(data["max_chars"]), chars_per_second=float(data["chars_per_second"]))


@dataclass(frozen=True)
class ChunkSpan:
    segment_id: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class Chunk:
    id: str
    spans: list[ChunkSpan]
    text: str
    boundary_after: str
    estimated_seconds: float
    target: str

    @property
    def narration_segment_ids(self) -> list[str]:
        ids: list[str] = []
        for span in self.spans:
            if span.segment_id not in ids:
                ids.append(span.segment_id)
        return ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "narration_segment_ids": self.narration_segment_ids,
            "spans": [span.to_dict() for span in self.spans],
            "text": self.text,
            "boundary_after": self.boundary_after,
            "estimated_seconds": self.estimated_seconds,
            "target": self.target,
        }


@dataclass(frozen=True)
class _Atom:
    segment_id: str
    start: int
    end: int
    paragraph_end: bool = False


def _aim_range(target: str, limits: ChunkLimits) -> tuple[int, int]:
    if target not in TARGET_SECONDS:
        known = ", ".join(sorted(TARGET_SECONDS))
        raise ValidationError([f"unknown chunk target '{target}' (available: {known})"])
    low_seconds, high_seconds = TARGET_SECONDS[target]
    high = min(round(high_seconds * limits.chars_per_second), limits.max_chars)
    low = min(round(low_seconds * limits.chars_per_second), high)
    return low, high


def _has_letter(text: str) -> bool:
    return any(char.isalpha() for char in text)


def _inside_quotes(text: str) -> list[bool]:
    flags: list[bool] = []
    depth = 0
    ascii_open = False
    for char in text:
        if char in OPENERS:
            depth += 1
        elif char in CLOSERS:
            depth = max(depth - 1, 0)
        elif char == '"':
            ascii_open = not ascii_open
        flags.append(depth > 0 or ascii_open)
    return flags


def _sentence_spans(text: str, quoted: list[bool]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_END.finditer(text):
        end = match.end()
        if quoted[end - 1]:
            continue
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _clause_cut(
    text: str, start: int, end: int, max_chars: int, quoted: list[bool]
) -> int | None:
    limit = min(end - 1, start + max_chars)
    for markers in CLAUSE_MARKERS:
        for index in range(limit - 1, start, -1):
            if (
                text[index] in markers
                and text[index + 1].isspace()
                and not quoted[index]
                and _has_letter(text[start : index + 1])
            ):
                return index + 1
    return None


def _space_cut(text: str, start: int, end: int, max_chars: int) -> int:
    limit = min(end, start + max_chars)
    for index in range(limit - 1, start, -1):
        if text[index].isspace() and _has_letter(text[start:index]):
            return index
    return start + max_chars


def _split_long(
    text: str, start: int, end: int, max_chars: int, quoted: list[bool]
) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    while end - start > max_chars:
        cut = _clause_cut(text, start, end, max_chars, quoted)
        if cut is None:
            cut = _space_cut(text, start, end, max_chars)
        pieces.append((start, cut))
        start = cut
    pieces.append((start, end))
    return pieces


def _paragraph_atoms(segment: NarrationSegment, max_chars: int) -> list[_Atom]:
    text = segment.text
    quoted = _inside_quotes(text)
    atoms = [
        _Atom(segment.id, piece_start, piece_end)
        for start, end in _sentence_spans(text, quoted)
        for piece_start, piece_end in _split_long(text, start, end, max_chars, quoted)
    ]
    if atoms:
        atoms[-1] = replace(atoms[-1], paragraph_end=True)
    return atoms


def _render(atoms: Sequence[_Atom], texts: dict[str, str]) -> tuple[list[ChunkSpan], str]:
    merged: list[ChunkSpan] = []
    for atom in atoms:
        if merged and merged[-1].segment_id == atom.segment_id and merged[-1].end == atom.start:
            merged[-1] = ChunkSpan(atom.segment_id, merged[-1].start, atom.end)
        else:
            merged.append(ChunkSpan(atom.segment_id, atom.start, atom.end))
    spans: list[ChunkSpan] = []
    for span in merged:
        raw = texts[span.segment_id][span.start : span.end]
        start = span.start + len(raw) - len(raw.lstrip())
        end = span.end - (len(raw) - len(raw.rstrip()))
        if start < end:
            spans.append(ChunkSpan(span.segment_id, start, end))
    text = " ".join(texts[span.segment_id][span.start : span.end] for span in spans)
    return spans, text


def _boundary_for(atom: _Atom) -> str:
    return "paragraph" if atom.paragraph_end else "sentence"


def plan_chunks(
    segments: Sequence[NarrationSegment],
    target: str,
    limits: ChunkLimits = ChunkLimits(),
) -> list[Chunk]:
    """Group narration segments into chunks that are sent to the engine in order.

    Headings and scene breaks force a boundary and are never part of a chunk text.
    """
    low, high = _aim_range(target, limits)
    if not segments:
        return []
    chapter_id = segments[0].chapter_id
    texts = {segment.id: segment.text for segment in segments}
    emitted: list[tuple[list[_Atom], str]] = []
    buffer: list[_Atom] = []

    def length(atoms: Sequence[_Atom]) -> int:
        return len(_render(atoms, texts)[1])

    def flush(boundary: str) -> None:
        if buffer:
            emitted.append((list(buffer), boundary))
            buffer.clear()
        elif emitted:
            emitted[-1] = (emitted[-1][0], boundary)

    def paragraph_cut() -> int | None:
        for index in range(len(buffer) - 2, -1, -1):
            if buffer[index].paragraph_end and length(buffer[: index + 1]) >= low:
                return index
        return None

    def add(atom: _Atom) -> None:
        buffer.append(atom)
        while len(buffer) > 1 and length(buffer) > high:
            cut = paragraph_cut()
            if cut is None:
                emitted.append((buffer[:-1], _boundary_for(buffer[-2])))
                del buffer[:-1]
            else:
                emitted.append((buffer[: cut + 1], "paragraph"))
                del buffer[: cut + 1]

    for segment in segments:
        if segment.kind == "scene_break":
            flush("scene")
        elif segment.kind == "heading":
            flush("paragraph")
        elif segment.kind == "paragraph":
            for atom in _paragraph_atoms(segment, limits.max_chars):
                add(atom)
        else:
            raise ValidationError([f"unknown segment kind: {segment.kind}"])
    flush("chapter_end")

    chunks: list[Chunk] = []
    for atoms, boundary in emitted:
        spans, text = _render(atoms, texts)
        if not text:
            continue
        chunks.append(
            Chunk(
                id=f"{chapter_id}-k{len(chunks) + 1:04d}",
                spans=spans,
                text=text,
                boundary_after=boundary,
                estimated_seconds=round(len(text) / limits.chars_per_second, 1),
                target=target,
            )
        )
    return chunks


def plan_document(
    chapter_id: str, target: str, limits: ChunkLimits, chunks: Sequence[Chunk]
) -> dict[str, Any]:
    low, high = TARGET_SECONDS[target]
    return {
        "schema_version": 1,
        "chapter_id": chapter_id,
        "target": target,
        "target_seconds": {"min": low, "max": high},
        "limits": limits.to_dict(),
        "duration_basis": "estimated from characters at chars_per_second; not measured",
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
