from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 60.0
SILENCE_THRESHOLD = 1e-4
MIN_CYRILLIC_RATIO = 0.7


class ValidationError(Exception):
    def __init__(self, messages: Sequence[str]) -> None:
        self.messages = list(messages)
        super().__init__("; ".join(self.messages))


def raise_for(problems: Sequence[str]) -> None:
    if problems:
        raise ValidationError(problems)


def _cyrillic_problems(text: str, label: str) -> list[str]:
    if not text.strip():
        return [f"{label} is empty"]
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return [f"{label} contains no letters"]
    cyrillic = sum(1 for char in letters if "Ѐ" <= char <= "ӿ")
    if cyrillic / len(letters) < MIN_CYRILLIC_RATIO:
        return [
            f"{label} must be predominantly Russian (Cyrillic); "
            f"only {cyrillic} of {len(letters)} letters are Cyrillic"
        ]
    return []


def text_problems(text: str, label: str = "text") -> list[str]:
    return _cyrillic_problems(text, label)


def transcript_problems(transcript: str) -> list[str]:
    return _cyrillic_problems(transcript, "transcript")


def read_text_file(path: Path) -> str:
    if not path.is_file():
        raise ValidationError([f"text file not found: {path}"])
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError([f"text file is not valid UTF-8: {path} ({exc})"]) from exc


def reference_audio_problems(path: Path) -> list[str]:
    if not path.is_file():
        return [f"reference audio not found: {path}"]
    try:
        info = sf.info(str(path))
        samples, _ = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        return [f"reference audio cannot be decoded: {path} ({exc})"]
    duration = info.frames / info.samplerate
    problems: list[str] = []
    if duration < MIN_REFERENCE_SECONDS:
        problems.append(
            f"reference audio is {duration:.2f} s long; it must be at least "
            f"{MIN_REFERENCE_SECONDS:g} s"
        )
    elif duration > MAX_REFERENCE_SECONDS:
        problems.append(
            f"reference audio is {duration:.2f} s long; it must be at most "
            f"{MAX_REFERENCE_SECONDS:g} s"
        )
    if samples.size == 0 or float(np.max(np.abs(samples))) < SILENCE_THRESHOLD:
        problems.append(f"reference audio is silent: {path}")
    return problems
