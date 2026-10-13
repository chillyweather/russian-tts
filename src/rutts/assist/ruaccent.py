"""RUAccent adapter. Written against RUAccent's documented API and NOT verified against the real package."""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
from typing import Any

from rutts.assist.base import AssistantConfig, register_assistant
from rutts.validation import ValidationError

PACKAGE = "ruaccent"
DEFAULT_OMOGRAPH_MODEL_SIZE = "turbo"
WEIGHT_SUFFIXES = (".onnx", ".safetensors", ".bin", ".pt", ".pth")
INSTALL_HINT = (
    "RUAccent is not installed in this environment; install it with 'pip install ruaccent' "
    "(see the 'Stress and ё suggestions' section of README.md)"
)


def model_files_sha256(package_dir: Path) -> str:
    """Digest over the weight files under the package directory, or 'unavailable' when none are found."""
    weights = sorted(path for path in package_dir.rglob("*") if path.is_file() and path.suffix in WEIGHT_SUFFIXES)
    if not weights:
        return "unavailable"
    digest = hashlib.sha256()
    for path in weights:
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _license() -> str:
    try:
        value = metadata.metadata(PACKAGE).get("License")
    except metadata.PackageNotFoundError:
        return "unverified"
    return str(value).strip() if value and str(value).strip() else "unverified"


class RUAccentAssistant:
    def __init__(self, omograph_model_size: str = DEFAULT_OMOGRAPH_MODEL_SIZE) -> None:
        self.omograph_model_size = omograph_model_size
        self._accentizer: Any = None

    @property
    def config(self) -> AssistantConfig:
        try:
            version = metadata.version(PACKAGE)
        except metadata.PackageNotFoundError as exc:
            raise ValidationError([INSTALL_HINT]) from exc
        package_dir = self._package_dir()
        return AssistantConfig(
            assistant_id="ruaccent",
            package=PACKAGE,
            package_version=version,
            model=f"ruaccent-omograph-{self.omograph_model_size}",
            model_sha256=model_files_sha256(package_dir) if package_dir is not None else "unavailable",
            settings={"omograph_model_size": self.omograph_model_size, "use_dictionary": True},
            license=_license(),
        )

    @staticmethod
    def _package_dir() -> Path | None:
        try:
            import ruaccent
        except ImportError:
            return None
        file = getattr(ruaccent, "__file__", None)
        return Path(file).parent if file else None

    def load(self) -> None:
        try:
            from ruaccent import RUAccent
        except ImportError as exc:
            raise ValidationError([INSTALL_HINT]) from exc
        accentizer = RUAccent()
        try:
            accentizer.load(omograph_model_size=self.omograph_model_size, use_dictionary=True)
        except Exception as exc:
            raise ValidationError(
                [f"RUAccent models could not be loaded ({exc}); check the ruaccent installation and its model files"]
            ) from exc
        self._accentizer = accentizer

    def unload(self) -> None:
        self._accentizer = None

    def annotate(self, text: str) -> str:
        if self._accentizer is None:
            raise RuntimeError("assistant is not loaded")
        return str(self._accentizer.process_all(text))


register_assistant("ruaccent", RUAccentAssistant)
