from __future__ import annotations

from rutts.book.project import BookProject, SourceChapter, SourceDocument, renumber
from rutts.pronunciation.store import rebuild_narration
from rutts.validation import ValidationError

DEFAULT_REASON = "excluded by user"


def _save(project: BookProject, source: SourceDocument) -> None:
    """Persist the source, then re-derive narration so corrections follow the new segment ids."""
    project.save_source(source)
    rebuild_narration(project)


def _require_title(title: str) -> str:
    cleaned = title.strip()
    if not cleaned:
        raise ValidationError(["chapter title is empty"])
    return cleaned


def split_chapter(project: BookProject, at_segment_id: str, title: str) -> None:
    """Start a new chapter at ``at_segment_id``; the rest of its chapter moves into it."""
    source = project.read_source()
    chapter_index, segment_index = source.segment_location(at_segment_id)
    chapter = source.chapters[chapter_index]
    if segment_index == 0:
        raise ValidationError(
            [f"segment {at_segment_id} already starts {chapter.id}; choose a later segment"]
        )
    tail = SourceChapter(
        id="",
        title=_require_title(title),
        order=0,
        included=True,
        segments=chapter.segments[segment_index:],
    )
    chapter.segments = chapter.segments[:segment_index]
    source.chapters.insert(chapter_index + 1, tail)
    renumber(source.chapters)
    _save(project, source)


def merge_chapter(project: BookProject, chapter_id: str) -> None:
    """Append a chapter's segments to the previous chapter."""
    source = project.read_source()
    index = source.chapter_index(chapter_id)
    if index == 0:
        raise ValidationError([f"{chapter_id} has no previous chapter to merge into"])
    merged = source.chapters.pop(index)
    source.chapters[index - 1].segments.extend(merged.segments)
    renumber(source.chapters)
    _save(project, source)


def rename_chapter(project: BookProject, chapter_id: str, title: str) -> None:
    source = project.read_source()
    source.chapter(chapter_id).title = _require_title(title)
    _save(project, source)


def set_chapter_included(
    project: BookProject, chapter_id: str, included: bool, reason: str | None = None
) -> None:
    source = project.read_source()
    chapter = source.chapter(chapter_id)
    chapter.included = included
    chapter.exclusion_reason = None if included else (reason or DEFAULT_REASON)
    _save(project, source)


def set_segment_included(
    project: BookProject, segment_id: str, included: bool, reason: str | None = None
) -> None:
    source: SourceDocument = project.read_source()
    chapter_index, segment_index = source.segment_location(segment_id)
    segment = source.chapters[chapter_index].segments[segment_index]
    segment.included = included
    segment.exclusion_reason = None if included else (reason or DEFAULT_REASON)
    _save(project, source)
