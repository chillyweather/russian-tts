from __future__ import annotations

import re
from pathlib import Path

import pytest

from rutts.book.chunking import ChunkLimits, plan_chunks
from rutts.book.project import NarrationSegment, load_project
from rutts.book.text_import import import_text
from rutts.validation import ValidationError

FIXTURE = Path(__file__).parent / "fixtures" / "book_sample.txt"


def _seg(segment_id: str, kind: str, text: str, chapter: str = "c001") -> NarrationSegment:
    return NarrationSegment(
        id=segment_id, source_segment_id=segment_id, chapter_id=chapter, kind=kind, text=text
    )


@pytest.fixture
def fixture_segments(tmp_path: Path) -> list[NarrationSegment]:
    import_text(FIXTURE, tmp_path / "book", title=None, author="", chapter_regex=None)
    project = load_project(tmp_path / "book")
    return project.planning_segments("c002")


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def test_chunk_texts_reproduce_narration_in_order(fixture_segments: list[NarrationSegment]) -> None:
    chunks = plan_chunks(fixture_segments, "short", ChunkLimits())
    joined = _squash("".join(chunk.text for chunk in chunks))
    expected = _squash("".join(s.text for s in fixture_segments if s.kind == "paragraph"))
    assert joined == expected


def test_every_chunk_respects_hard_limit(fixture_segments: list[NarrationSegment]) -> None:
    for target in ("short", "long"):
        chunks = plan_chunks(fixture_segments, target, ChunkLimits(max_chars=600))
        assert chunks
        assert all(len(chunk.text) <= 600 for chunk in chunks)


def test_long_sentence_is_split_at_clause_boundaries(fixture_segments: list[NarrationSegment]) -> None:
    long_segment = next(s for s in fixture_segments if len(s.text) > 800)
    chunks = [c for c in plan_chunks(fixture_segments, "long", ChunkLimits(max_chars=600))]
    pieces = [
        (span.start, span.end)
        for chunk in chunks
        for span in chunk.spans
        if span.segment_id == long_segment.id
    ]
    assert len(pieces) >= 2
    assert pieces[0][0] == 0
    assert pieces[-1][1] == len(long_segment.text.rstrip())
    for (_, end), (start, _) in zip(pieces, pieces[1:], strict=False):
        assert long_segment.text[end:start].strip() == ""
    for start, end in pieces[:-1]:
        cut = long_segment.text[:end].rstrip()
        assert cut[-1] in ";:,—", cut[-30:]


def test_chunk_spans_index_exact_text(fixture_segments: list[NarrationSegment]) -> None:
    by_id = {s.id: s for s in fixture_segments}
    for chunk in plan_chunks(fixture_segments, "short", ChunkLimits()):
        pieces = [by_id[span.segment_id].text[span.start : span.end] for span in chunk.spans]
        assert " ".join(pieces) == chunk.text


def test_quoted_speech_is_not_split_inside(fixture_segments: list[NarrationSegment]) -> None:
    chunks = plan_chunks(fixture_segments, "short", ChunkLimits())
    holder = next(c for c in chunks if "Нет." in c.text)
    assert "«Нет. Не сейчас», — сказал он." in holder.text


def test_dialogue_dash_stays_with_its_line() -> None:
    body = "Он говорил, что вернётся, " * 30
    segment = _seg("c001-s0001", "paragraph", "— " + body.strip() + " — конец.")
    chunks = plan_chunks([segment], "long", ChunkLimits(max_chars=200))
    assert len(chunks) >= 2
    assert chunks[0].text.startswith("— Он говорил")
    assert all(chunk.text != "—" for chunk in chunks)


def test_scene_break_and_heading_force_boundaries_and_are_not_sent(
    fixture_segments: list[NarrationSegment],
) -> None:
    all_chapter_one = [
        _seg("c001-s0001", "heading", "Глава 1. Утро"),
        _seg("c001-s0002", "paragraph", "Первое предложение. Второе предложение."),
        _seg("c001-s0003", "scene_break", "* * *"),
        _seg("c001-s0004", "paragraph", "Третье предложение."),
    ]
    chunks = plan_chunks(all_chapter_one, "short", ChunkLimits())
    assert [c.boundary_after for c in chunks] == ["scene", "chapter_end"]
    texts = " ".join(c.text for c in chunks)
    assert "* * *" not in texts
    assert "Глава 1. Утро" not in texts


def test_chunk_ids_are_chapter_scoped_and_ordered(fixture_segments: list[NarrationSegment]) -> None:
    chunks = plan_chunks(fixture_segments, "short", ChunkLimits())
    assert [c.id for c in chunks] == [f"c002-k{i:04d}" for i in range(1, len(chunks) + 1)]


def test_estimated_seconds_come_from_fourteen_chars_per_second(
    fixture_segments: list[NarrationSegment],
) -> None:
    for chunk in plan_chunks(fixture_segments, "short", ChunkLimits()):
        assert chunk.estimated_seconds == round(len(chunk.text) / 14.0, 1)
        assert chunk.target == "short"


def test_unknown_target_is_rejected(fixture_segments: list[NarrationSegment]) -> None:
    with pytest.raises(ValidationError, match="unknown chunk target"):
        plan_chunks(fixture_segments, "medium", ChunkLimits())


def test_project_helper_builds_planning_segments_without_excluded_text(tmp_path: Path) -> None:
    import_text(FIXTURE, tmp_path / "book", title=None, author="", chapter_regex=None)
    project = load_project(tmp_path / "book")
    segments = project.planning_segments("c001")
    assert [s.kind for s in segments] == ["heading", "paragraph", "scene_break", "paragraph"]
