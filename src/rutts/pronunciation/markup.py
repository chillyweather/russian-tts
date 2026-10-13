from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rutts.atomic import atomic_path
from rutts.book.chunking import Chunk
from rutts.validation import ValidationError

MARKUP_FILE = "narration/markup.json"
STRESS_MARK = "+"
ACUTE = "́"
CONVENTIONS = ("none", "plus", "acute")
RUSSIAN_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
HINT = re.compile(rf"\+(?=[{RUSSIAN_VOWELS}])")
HINT_VOWEL = re.compile(rf"\+([{RUSSIAN_VOWELS}])")


@dataclass(frozen=True)
class MarkupSupport:
    convention: str
    verified: bool


# Default per engine. Nothing is verified by ear yet, so every engine starts without hints.
ENGINE_MARKUP: dict[str, MarkupSupport] = {
    "qwen-mlx": MarkupSupport("none", verified=False),
    "fish-mlx": MarkupSupport("none", verified=False),
    "voxcpm2": MarkupSupport("none", verified=False),
}


def validate_replacement(text: str) -> list[str]:
    """Reject markup that the engine would speak literally or that has no defined meaning."""
    problems: list[str] = []
    if "<" in text or ">" in text:
        problems.append(f"angle brackets are not supported in replacements (SSML-like markup): {text!r}")
    if ACUTE in text:
        problems.append(f"use '+' before the stressed vowel instead of a combining accent: {text!r}")
    for word in text.split():
        if word.count(STRESS_MARK) > 1:
            problems.append(f"more than one stress mark in word {word!r}")
        for index, char in enumerate(word):
            if char != STRESS_MARK:
                continue
            following = word[index + 1 : index + 2]
            if not following or following not in RUSSIAN_VOWELS:
                problems.append(f"stress mark '+' must be directly before a Russian vowel: {word!r}")
    return problems


def require_valid_replacement(text: str) -> None:
    problems = validate_replacement(text)
    if problems:
        raise ValidationError(problems)


def has_hints(text: str) -> bool:
    return HINT.search(text) is not None


def translate(text: str, convention: str) -> str:
    if convention == "plus":
        return text
    if convention == "acute":
        return HINT_VOWEL.sub(lambda match: match.group(1) + ACUTE, text)
    if convention == "none":
        return HINT.sub("", text)
    raise ValidationError([f"unknown pronunciation convention: {convention}"])


def translate_chunks(chunks: list[Chunk], convention: str) -> list[Chunk]:
    """The text an engine receives; fingerprints are computed on these texts."""
    return [replace(chunk, text=translate(chunk.text, convention)) for chunk in chunks]


def hint_chunk_ids(chunks: list[Chunk], convention: str) -> list[str]:
    """Chunks whose stress hints are dropped because the engine does not support them."""
    if convention != "none":
        return []
    return [chunk.id for chunk in chunks if has_hints(chunk.text)]


def read_markup(directory: Path) -> list[dict[str, Any]]:
    path = directory / MARKUP_FILE
    if not path.is_file():
        return []
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return list(document["entries"])


def write_markup(directory: Path, entries: list[dict[str, Any]]) -> None:
    with atomic_path(directory / MARKUP_FILE) as temp:
        temp.write_text(json.dumps({"schema_version": 1, "entries": entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def effective_convention(engine_id: str, settings: list[dict[str, Any]]) -> str:
    for setting in settings:
        if setting["id"] == engine_id:
            return str(setting["convention"])
    return ENGINE_MARKUP.get(engine_id, MarkupSupport("none", verified=False)).convention
