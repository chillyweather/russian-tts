from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rutts.atomic import atomic_write_text
from rutts.validation import (
    ValidationError,
    raise_for,
    reference_audio_problems,
    transcript_problems,
)

PROFILE_FILE = "profile.json"
TRANSCRIPT_FILE = "transcript.txt"
REFERENCE_STEM = "reference"
LANGUAGE = "ru"


@dataclass(frozen=True)
class NarratorProfile:
    name: str
    reference_audio: Path
    transcript: str
    language: str
    provenance: str
    reference_sha256: str
    directory: Path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_profile(
    directory: Path,
    *,
    name: str,
    audio: Path,
    transcript: str,
    provenance: str,
) -> NarratorProfile:
    problems: list[str] = []
    if not name.strip():
        problems.append("narrator name is empty")
    if not provenance.strip():
        problems.append("provenance is empty; record where the reference recording came from")
    problems += transcript_problems(transcript)
    problems += reference_audio_problems(audio)
    if (directory / PROFILE_FILE).exists():
        problems.append(f"a narrator profile already exists in {directory}")
    raise_for(problems)

    directory.mkdir(parents=True, exist_ok=True)
    reference_name = REFERENCE_STEM + audio.suffix.lower()
    reference_path = directory / reference_name
    shutil.copyfile(audio, reference_path)
    atomic_write_text(directory / TRANSCRIPT_FILE, transcript)
    document: dict[str, Any] = {
        "name": name,
        "language": LANGUAGE,
        "provenance": provenance,
        "reference_file": reference_name,
        "reference_sha256": _sha256_file(reference_path),
        "transcript_file": TRANSCRIPT_FILE,
        "transcript_sha256": _sha256_text(transcript),
    }
    atomic_write_text(directory / PROFILE_FILE, json.dumps(document, ensure_ascii=False, indent=2))
    return load_profile(directory)


def load_profile(directory: Path) -> NarratorProfile:
    profile_path = directory / PROFILE_FILE
    if not profile_path.is_file():
        raise ValidationError([f"no narrator profile found in {directory} (missing {PROFILE_FILE})"])
    try:
        document = json.loads(profile_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError([f"narrator profile is corrupt: {profile_path} ({exc})"]) from exc

    reference_path = directory / str(document["reference_file"])
    transcript_path = directory / str(document["transcript_file"])
    problems: list[str] = []
    if not reference_path.is_file():
        problems.append(f"narrator reference audio is missing: {reference_path}")
    if not transcript_path.is_file():
        problems.append(f"narrator transcript is missing: {transcript_path}")
    raise_for(problems)

    if _sha256_file(reference_path) != document["reference_sha256"]:
        problems.append(
            f"narrator reference audio changed since the profile was created: {reference_path}"
        )
    transcript = transcript_path.read_text(encoding="utf-8")
    if _sha256_text(transcript) != document["transcript_sha256"]:
        problems.append(f"narrator transcript changed since the profile was created: {transcript_path}")
    raise_for(problems)

    return NarratorProfile(
        name=str(document["name"]),
        reference_audio=reference_path,
        transcript=transcript,
        language=str(document["language"]),
        provenance=str(document["provenance"]),
        reference_sha256=str(document["reference_sha256"]),
        directory=directory,
    )
