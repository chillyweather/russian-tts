from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rutts.atomic import atomic_write_text
from rutts.narrator import PROFILE_FILE, TRANSCRIPT_FILE, NarratorProfile, load_profile
from rutts.validation import ValidationError

SCHEMA_VERSION = 1
PROJECT_FILE = "project.json"
ORIGINAL_FILE = "source/original.txt"
SOURCE_FILE = "source/source.json"
NARRATION_FILE = "narration/narration.json"
NARRATOR_DIR = "narrator"
CHUNKS_DIR = "chunks"
CHAPTERS_DIR = "chapters"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, document: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(document, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationError([f"book file is missing: {path}"])
    try:
        document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError([f"book file is corrupt: {path} ({exc})"]) from exc
    return document


@dataclass
class SourceSegment:
    id: str
    chapter_id: str
    kind: str
    text: str
    line_start: int
    line_end: int
    included: bool = True
    exclusion_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "chapter_id": self.chapter_id,
            "kind": self.kind,
            "text": self.text,
            "location": {"line_start": self.line_start, "line_end": self.line_end},
            "included": self.included,
            "exclusion_reason": self.exclusion_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSegment:
        location = data["location"]
        reason = data["exclusion_reason"]
        return cls(
            id=str(data["id"]),
            chapter_id=str(data["chapter_id"]),
            kind=str(data["kind"]),
            text=str(data["text"]),
            line_start=int(location["line_start"]),
            line_end=int(location["line_end"]),
            included=bool(data["included"]),
            exclusion_reason=None if reason is None else str(reason),
        )


@dataclass
class SourceChapter:
    id: str
    title: str
    order: int
    included: bool
    segments: list[SourceSegment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "order": self.order,
            "included": self.included,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceChapter:
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            order=int(data["order"]),
            included=bool(data["included"]),
            segments=[SourceSegment.from_dict(item) for item in data["segments"]],
        )


@dataclass
class SourceDocument:
    chapters: list[SourceChapter]

    def chapter_index(self, chapter_id: str) -> int:
        for index, chapter in enumerate(self.chapters):
            if chapter.id == chapter_id:
                return index
        raise ValidationError([f"unknown chapter: {chapter_id}"])

    def chapter(self, chapter_id: str) -> SourceChapter:
        return self.chapters[self.chapter_index(chapter_id)]

    def segment_location(self, segment_id: str) -> tuple[int, int]:
        for chapter_index, chapter in enumerate(self.chapters):
            for segment_index, segment in enumerate(chapter.segments):
                if segment.id == segment_id:
                    return chapter_index, segment_index
        raise ValidationError([f"unknown segment: {segment_id}"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "chapters": [chapter.to_dict() for chapter in self.chapters],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceDocument:
        return cls(chapters=[SourceChapter.from_dict(item) for item in data["chapters"]])


@dataclass
class NarrationSegment:
    id: str
    source_segment_id: str
    chapter_id: str
    kind: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_segment_id": self.source_segment_id,
            "chapter_id": self.chapter_id,
            "kind": self.kind,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NarrationSegment:
        return cls(
            id=str(data["id"]),
            source_segment_id=str(data["source_segment_id"]),
            chapter_id=str(data["chapter_id"]),
            kind=str(data["kind"]),
            text=str(data["text"]),
        )


@dataclass
class NarrationDocument:
    segments: list[NarrationSegment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NarrationDocument:
        return cls(segments=[NarrationSegment.from_dict(item) for item in data["segments"]])


def renumber(chapters: list[SourceChapter]) -> None:
    """Derive chapter and segment ids from reading order."""
    for order, chapter in enumerate(chapters, start=1):
        chapter.order = order
        chapter.id = f"c{order:03d}"
        for index, segment in enumerate(chapter.segments, start=1):
            segment.chapter_id = chapter.id
            segment.id = f"{chapter.id}-s{index:04d}"


def narration_from_source(source: SourceDocument) -> NarrationDocument:
    return NarrationDocument(
        segments=[
            NarrationSegment(
                id=segment.id,
                source_segment_id=segment.id,
                chapter_id=chapter.id,
                kind=segment.kind,
                text=segment.text,
            )
            for chapter in source.chapters
            for segment in chapter.segments
        ]
    )


@dataclass(frozen=True)
class BookProject:
    directory: Path
    meta: dict[str, Any]

    @property
    def title(self) -> str:
        return str(self.meta["title"])

    @property
    def author(self) -> str:
        return str(self.meta["author"])

    @property
    def narrator_dir(self) -> Path:
        return self.directory / NARRATOR_DIR

    def chapter_dir(self, chapter_id: str) -> Path:
        return self.directory / CHUNKS_DIR / chapter_id

    def target_dir(self, chapter_id: str, target: str) -> Path:
        return self.chapter_dir(chapter_id) / target

    def read_source(self) -> SourceDocument:
        return SourceDocument.from_dict(read_json(self.directory / SOURCE_FILE))

    def read_narration(self) -> NarrationDocument:
        return NarrationDocument.from_dict(read_json(self.directory / NARRATION_FILE))

    def save_source(self, source: SourceDocument) -> None:
        write_json(self.directory / SOURCE_FILE, source.to_dict())

    def narrator(self) -> NarratorProfile:
        return load_profile(self.narrator_dir)

    def narrator_sha256(self) -> str:
        return sha256_file(self.narrator_dir / PROFILE_FILE)

    def planning_segments(self, chapter_id: str) -> list[NarrationSegment]:
        """Narration segments of a chapter that take part in synthesis, in reading order."""
        source = self.read_source()
        chapter = source.chapter(chapter_id)
        if not chapter.included:
            return []
        narration = {segment.id: segment for segment in self.read_narration().segments}
        return [narration[segment.id] for segment in chapter.segments if segment.included]


def create_project(directory: Path, meta: dict[str, Any], source: SourceDocument) -> BookProject:
    write_json(directory / PROJECT_FILE, {"schema_version": SCHEMA_VERSION, **meta})
    project = BookProject(directory=directory, meta=meta)
    project.save_source(source)
    write_json(directory / NARRATION_FILE, narration_from_source(source).to_dict())
    return project


def load_project(directory: Path) -> BookProject:
    if not (directory / PROJECT_FILE).is_file():
        raise ValidationError([f"not a book project (no {PROJECT_FILE}): {directory}"])
    meta = read_json(directory / PROJECT_FILE)
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError([f"unsupported book project schema in {directory}"])
    return BookProject(directory=directory, meta=meta)


def install_narrator(project: BookProject, profile_dir: Path) -> NarratorProfile:
    profile = load_profile(profile_dir)
    destination = project.narrator_dir
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in (PROFILE_FILE, TRANSCRIPT_FILE, profile.reference_audio.name):
        shutil.copyfile(profile_dir / name, destination / name)
    return load_profile(destination)
