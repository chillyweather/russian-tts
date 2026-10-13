from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rutts.book.project import (
    ORIGINAL_FILE,
    PROJECT_FILE,
    BookProject,
    SourceChapter,
    SourceDocument,
    SourceSegment,
    create_project,
    renumber,
    sha256_bytes,
)
from rutts.validation import ValidationError, raise_for, text_problems

DEFAULT_MARKER = re.compile(r"(?:Глава|ГЛАВА|Часть|ЧАСТЬ)\s+\S.*|#+\s+\S.*")
SCENE_BREAK = re.compile(r"([*\-_=~#])(?:\s*\1){2,}")


@dataclass
class _Block:
    kind: str
    text: str
    line_start: int
    line_end: int


@dataclass
class _Section:
    title: str
    blocks: list[_Block] = field(default_factory=list)


def read_input(path: Path) -> tuple[str, bytes]:
    if not path.is_file():
        raise ValidationError([f"input file not found: {path}"])
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError([f"input file is not valid UTF-8: {path} ({exc})"]) from exc
    text = text.removeprefix("﻿")
    raise_for(text_problems(text, "book text"))
    return text, data


def _compile_marker(extra: str | None) -> re.Pattern[str] | None:
    if extra is None:
        return None
    try:
        return re.compile(extra)
    except re.error as exc:
        raise ValidationError([f"invalid --chapter-regex: {exc}"]) from exc


def _is_marker(line: str, extra: re.Pattern[str] | None) -> bool:
    if DEFAULT_MARKER.fullmatch(line):
        return True
    return extra is not None and extra.match(line) is not None


def _marker_title(line: str) -> str:
    return re.sub(r"^#+\s*", "", line) if line.startswith("#") else line


def _parse_sections(text: str, extra: re.Pattern[str] | None) -> list[_Section]:
    sections = [_Section(title="")]
    current = sections[0]
    paragraph: list[str] = []
    start = end = 0

    def close_paragraph() -> None:
        if paragraph:
            body = "\n".join(paragraph).strip()
            current.blocks.append(_Block("paragraph", body, start, end))
            paragraph.clear()

    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            close_paragraph()
        elif _is_marker(stripped, extra):
            close_paragraph()
            current = _Section(title=_marker_title(stripped))
            current.blocks.append(_Block("heading", stripped, number, number))
            sections.append(current)
        elif SCENE_BREAK.fullmatch(stripped):
            close_paragraph()
            current.blocks.append(_Block("scene_break", stripped, number, number))
        else:
            if not paragraph:
                start = number
            paragraph.append(line)
            end = number
    close_paragraph()
    return [section for section in sections if section.blocks]


def parse_source(text: str, chapter_regex: str | None) -> SourceDocument:
    sections = _parse_sections(text, _compile_marker(chapter_regex))
    chapters = [
        SourceChapter(
            id="",
            title=section.title,
            order=0,
            included=True,
            segments=[
                SourceSegment(
                    id="",
                    chapter_id="",
                    kind=block.kind,
                    text=block.text,
                    line_start=block.line_start,
                    line_end=block.line_end,
                )
                for block in section.blocks
            ],
        )
        for section in sections
    ]
    renumber(chapters)
    return SourceDocument(chapters=chapters)


def import_text(
    source_file: Path,
    out: Path,
    *,
    title: str | None,
    author: str,
    chapter_regex: str | None,
) -> BookProject:
    _compile_marker(chapter_regex)
    text, data = read_input(source_file)
    if (out / PROJECT_FILE).exists():
        raise ValidationError([f"a book project already exists in {out}"])
    source = parse_source(text, chapter_regex)
    original = out / ORIGINAL_FILE
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(data)
    meta = {
        "title": title or source_file.stem,
        "author": author,
        "source_format": "text",
        "original_filename": source_file.name,
        "original_sha256": sha256_bytes(data),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return create_project(out, meta, source)
