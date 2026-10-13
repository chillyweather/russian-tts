from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources as package_resources
from typing import Any

from rutts.assist.base import StressAssistant
from rutts.book.project import BookProject, SourceChapter, SourceSegment, sha256_text, write_json
from rutts.pronunciation.apply import DictionaryEntry, OverrideEntry, compile_dictionary_entry, remap_overrides
from rutts.pronunciation.markup import RUSSIAN_VOWELS, STRESS_MARK
from rutts.pronunciation.store import load_corrections
from rutts.validation import ValidationError

SUGGESTIONS_DIR = "narration/suggestions"
HOMOGRAPHS_FILE = "homographs_ru.json"
SPOKEN_KINDS = ("paragraph", "heading")
CONTEXT_CHARS = 30
STATES = ("proposed", "accepted", "rejected")
KINDS = ("yo", "stress")
WORD = re.compile(r"[А-Яа-яЁё]+")
YO_FROM = {"е": "ё", "Е": "Ё"}


@dataclass
class Result:
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    unaligned: list[str] = field(default_factory=list)
    skipped_user_override: list[dict[str, Any]] = field(default_factory=list)


def chapter_sha256(chapter: SourceChapter) -> str:
    return sha256_text(json.dumps(chapter.to_dict(), ensure_ascii=False, sort_keys=True))


def load_homographs() -> frozenset[str]:
    text = package_resources.files("rutts").joinpath("data", HOMOGRAPHS_FILE).read_text(encoding="utf-8")
    return frozenset(str(word) for word in json.loads(text)["words"])


def align(source: str, annotated: str) -> tuple[set[int], set[int]] | None:
    """Match assistant output to the source character by character.

    Only two differences are allowed: a '+' stress hint directly before a Russian vowel, and е -> ё.
    Returns (hint vowel positions, ё positions) in source coordinates, or None when anything else differs.
    """
    hints: set[int] = set()
    yo: set[int] = set()
    pending = False
    i = 0
    for char in annotated:
        if char == STRESS_MARK and not (i < len(source) and source[i] == STRESS_MARK):
            if pending:
                return None
            pending = True
            continue
        if i >= len(source):
            return None
        expected = source[i]
        if char == expected:
            pass
        elif expected in YO_FROM and char == YO_FROM[expected]:
            yo.add(i)
        else:
            return None
        if pending:
            if expected not in RUSSIAN_VOWELS:
                return None
            hints.add(i)
            pending = False
        i += 1
    if pending or i != len(source):
        return None
    return hints, yo


def _with_yo(word: str, start: int, yo: list[int], enabled: bool) -> list[str]:
    chars = list(word)
    if enabled:
        for position in yo:
            chars[position - start] = YO_FROM[chars[position - start]]
    return chars


def _context(text: str, start: int, end: int) -> tuple[str, str]:
    before = text[max(0, start - CONTEXT_CHARS) : start]
    after = text[end : end + CONTEXT_CHARS]
    return ("…" if start - CONTEXT_CHARS > 0 else "") + before, after + ("…" if end + CONTEXT_CHARS < len(text) else "")


def _suggestion(
    chapter_id: str, number: int, kind: str, segment: SourceSegment, start: int, end: int, replacement: str
) -> dict[str, Any]:
    before, after = _context(segment.text, start, end)
    return {
        "id": f"{chapter_id}-g{number:03d}",
        "kind": kind,
        "segment_id": segment.id,
        "start": start,
        "end": end,
        "original": segment.text[start:end],
        "replacement": replacement,
        "context_before": before,
        "context_after": after,
        "state": "proposed",
        "override_id": None,
    }


def compute(
    project: BookProject,
    chapter_id: str,
    assistant: StressAssistant,
    *,
    stress_scope: str,
    yo_on: bool,
    own_overrides: frozenset[str] = frozenset(),
) -> Result:
    source = project.read_source()
    chapter = source.chapter(chapter_id)
    corrections = load_corrections(project)
    overrides = [
        item
        for item in remap_overrides(source, [OverrideEntry.from_dict(raw) for raw in corrections.overrides])
        if item.id not in own_overrides
    ]
    dictionary = [(entry, compile_dictionary_entry(entry)) for entry in map(DictionaryEntry.from_dict, corrections.dictionary)]
    homographs = load_homographs()
    result = Result()
    number = 0
    for segment in chapter.segments:
        if not segment.included or segment.kind not in SPOKEN_KINDS:
            continue
        aligned = align(segment.text, assistant.annotate(segment.text))
        if aligned is None:
            result.unaligned.append(segment.id)
            continue
        hints, yo_positions = aligned
        for match in WORD.finditer(segment.text):
            start, end = match.span()
            word = match.group(0)
            word_hints = sorted(position for position in hints if start <= position < end)
            word_yo = sorted(position for position in yo_positions if start <= position < end)
            if len(word_hints) > 1 or (not word_hints and not word_yo):
                continue
            wants_stress = bool(word_hints) and (stress_scope == "all" or word.lower() in homographs)
            if not wants_stress and not (word_yo and yo_on):
                continue
            skip = _user_precedence(segment, start, end, overrides, dictionary)
            if skip is not None:
                result.skipped_user_override.append(
                    {"segment_id": segment.id, "start": start, "end": end, "original": word, "reason": skip}
                )
                continue
            chars = _with_yo(word, start, word_yo, yo_on)
            if wants_stress:
                chars.insert(word_hints[0] - start, STRESS_MARK)
                kind = "stress"
            else:
                kind = "yo"
            number += 1
            result.suggestions.append(_suggestion(chapter_id, number, kind, segment, start, end, "".join(chars)))
    return result


def _user_precedence(
    segment: SourceSegment,
    start: int,
    end: int,
    overrides: list[OverrideEntry],
    dictionary: list[tuple[DictionaryEntry, Any]],
) -> str | None:
    for override in overrides:
        if override.segment_id == segment.id and override.start < end and start < override.end:
            return "user occurrence override"
    for _, pattern in dictionary:
        for found in pattern.finditer(segment.text):
            if found.start() < end and start < found.end():
                return "user dictionary entry"
    return None


def suggestions_path(project: BookProject, chapter_id: str) -> Any:
    return project.directory / SUGGESTIONS_DIR / f"{chapter_id}.json"


def read_suggestions(project: BookProject, chapter_id: str) -> dict[str, Any] | None:
    path = suggestions_path(project, chapter_id)
    if not path.is_file():
        return None
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def write_suggestions(project: BookProject, chapter_id: str, document: dict[str, Any]) -> None:
    suggestions_path(project, chapter_id).parent.mkdir(parents=True, exist_ok=True)
    write_json(suggestions_path(project, chapter_id), document)


def is_stale(project: BookProject, document: dict[str, Any]) -> bool:
    chapter = project.read_source().chapter(str(document["chapter_id"]))
    return bool(document["source_sha256"] != chapter_sha256(chapter))


def carry_forward(previous: dict[str, Any] | None, current: dict[str, Any], source_sha256: str) -> None:
    """Keep decisions for identical proposals when the source has not changed."""
    if previous is None or previous["source_sha256"] != source_sha256:
        return
    decided = {
        (item["segment_id"], item["start"], item["end"], item["kind"], item["replacement"]): item
        for item in previous["suggestions"]
        if item["state"] != "proposed"
    }
    for item in current["suggestions"]:
        earlier = decided.get((item["segment_id"], item["start"], item["end"], item["kind"], item["replacement"]))
        if earlier is not None:
            item["state"] = earlier["state"]
            item["override_id"] = earlier["override_id"]


def find_suggestion(project: BookProject, suggestion_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    chapter_id = suggestion_id.split("-")[0]
    document = read_suggestions(project, chapter_id)
    if document is None:
        raise ValidationError([f"no suggestions for {chapter_id}; run rutts book suggest first"])
    for item in document["suggestions"]:
        if item["id"] == suggestion_id:
            return document, item
    raise ValidationError([f"no suggestion with id {suggestion_id}"])


def occurrence_of(segment_text: str, start: int, find: str) -> int:
    return sum(1 for index in range(start + 1) if segment_text.startswith(find, index))
