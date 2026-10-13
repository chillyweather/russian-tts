from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from rutts.validation import ValidationError


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ValidationError([f"{name} is not installed or not on PATH"])
    return path


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def ffmpeg_version() -> str:
    output = run([require_tool("ffmpeg"), "-version"]).stdout
    return output.splitlines()[0] if output else "unknown"


def probe(path: Path) -> dict[str, Any]:
    result = run([require_tool("ffprobe"), "-v", "error", "-show_format", "-show_chapters", "-of", "json", str(path)])
    if result.returncode != 0:
        raise ValidationError([f"ffprobe cannot read {path.name}: {result.stderr.strip()[:300]}"])
    document: dict[str, Any] = json.loads(result.stdout)
    return document
