from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from rutts.engines.mlx_common import clear_mlx_cache, resolve_revision

WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
WHISPER_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
RUSSIAN = "ru"
WHISPER_LICENSE = "MIT"


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class Transcript:
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "words": [word.to_dict() for word in self.words]}


@dataclass(frozen=True)
class TranscriberConfig:
    transcriber_id: str
    model_repo: str
    model_revision: str
    runtime_name: str
    runtime_version: str
    language: str
    settings: dict[str, Any] = field(default_factory=dict)
    license: str = "unverified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Transcriber(Protocol):
    @property
    def config(self) -> TranscriberConfig: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def transcribe(self, path: Path) -> Transcript: ...


TranscriberFactory = Callable[..., Transcriber]

TRANSCRIBER_FACTORIES: dict[str, TranscriberFactory] = {}


def register_transcriber(transcriber_id: str, factory: TranscriberFactory) -> None:
    TRANSCRIBER_FACTORIES[transcriber_id] = factory


def get_transcriber(transcriber_id: str, **options: Any) -> Transcriber:
    if transcriber_id not in TRANSCRIBER_FACTORIES:
        known = ", ".join(sorted(TRANSCRIBER_FACTORIES)) or "none"
        raise ValueError(f"unknown transcriber '{transcriber_id}' (available: {known})")
    return TRANSCRIBER_FACTORIES[transcriber_id](**options)


class WhisperMlxTranscriber:
    """Whisper large v3 turbo through mlx-audio, forced to Russian with word timestamps."""

    def __init__(self) -> None:
        self._model: Any = None

    @property
    def config(self) -> TranscriberConfig:
        revision = resolve_revision(WHISPER_REPO)
        if revision != WHISPER_REVISION:
            raise RuntimeError(
                f"cached {WHISPER_REPO} is revision {revision}, pinned revision is {WHISPER_REVISION}"
            )
        return TranscriberConfig(
            transcriber_id="whisper-mlx",
            model_repo=WHISPER_REPO,
            model_revision=revision,
            runtime_name="mlx-audio",
            runtime_version=metadata.version("mlx-audio"),
            language=RUSSIAN,
            settings={"word_timestamps": True, "task": "transcribe"},
            license=WHISPER_LICENSE,
        )

    def load(self) -> None:
        from mlx_audio.stt import load

        self._model = load(WHISPER_REPO, revision=WHISPER_REVISION)

    def unload(self) -> None:
        self._model = None
        clear_mlx_cache()

    def transcribe(self, path: Path) -> Transcript:
        if self._model is None:
            raise RuntimeError("transcriber is not loaded")
        output = self._model.generate(str(path), language=RUSSIAN, word_timestamps=True)
        words: list[Word] = []
        for segment in output.segments:
            for item in segment.get("words", []):
                text = str(item["word"]).strip()
                if text:
                    words.append(Word(text=text, start=float(item["start"]), end=float(item["end"])))
        return Transcript(words=words)


register_transcriber("whisper-mlx", WhisperMlxTranscriber)
