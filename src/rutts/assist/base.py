from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class AssistantConfig:
    assistant_id: str
    package: str
    package_version: str
    model: str
    model_sha256: str
    settings: dict[str, Any] = field(default_factory=dict)
    license: str = "unverified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class StressAssistant(Protocol):
    @property
    def config(self) -> AssistantConfig: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def annotate(self, text: str) -> str:
        """Return ``text`` with '+' before stressed vowels and ё restored; no other change is allowed."""
        ...


AssistantFactory = Callable[..., StressAssistant]

ASSISTANT_FACTORIES: dict[str, AssistantFactory] = {}


def register_assistant(assistant_id: str, factory: AssistantFactory) -> None:
    ASSISTANT_FACTORIES[assistant_id] = factory


def get_assistant(assistant_id: str, **options: Any) -> StressAssistant:
    if assistant_id not in ASSISTANT_FACTORIES:
        known = ", ".join(sorted(ASSISTANT_FACTORIES)) or "none"
        raise ValueError(f"unknown assistant '{assistant_id}' (available: {known})")
    return ASSISTANT_FACTORIES[assistant_id](**options)
