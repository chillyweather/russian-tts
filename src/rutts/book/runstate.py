from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rutts.atomic import atomic_path

RUN_FILE = "run.json"
RECENT_FAILURES = 5


def format_elapsed(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def read_run_state(directory: Path) -> dict[str, Any] | None:
    path = directory / RUN_FILE
    if not path.is_file():
        return None
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


@dataclass
class RunTracker:
    """Keeps run.json current: counts, elapsed time and the most recent failures."""

    directory: Path
    chapter_id: str
    target: str
    total: int
    reused: int
    previous: dict[str, Any] | None

    def __post_init__(self) -> None:
        self._started = time.perf_counter()
        self._cumulative_before = float((self.previous or {}).get("cumulative_seconds", 0.0))
        self._failures: list[dict[str, str]] = list((self.previous or {}).get("recent_failures", []))
        self.current_chunk: str | None = None

    def elapsed(self) -> float:
        return time.perf_counter() - self._started

    def record_failure(self, chunk_id: str, error: str) -> None:
        self._failures = [*self._failures, {"chunk": chunk_id, "error": error}][-RECENT_FAILURES:]

    def save(self, records: list[dict[str, Any]], status: str) -> dict[str, Any]:
        elapsed = self.elapsed()
        document: dict[str, Any] = {
            "status": status,
            "chapter": self.chapter_id,
            "target": self.target,
            "current_chunk": self.current_chunk,
            "total_chunks": self.total,
            "completed": sum(1 for record in records if record["status"] == "ok"),
            "reused": self.reused,
            "remaining": sum(1 for record in records if record["status"] == "missing"),
            "failed": sum(1 for record in records if record["status"] == "failed"),
            "elapsed_seconds": round(elapsed, 3),
            "cumulative_seconds": round(self._cumulative_before + elapsed, 3),
            "recent_failures": self._failures,
        }
        with atomic_path(self.directory / RUN_FILE) as temp:
            temp.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return document


def status_lines(chapter_id: str, target: str, records: list[dict[str, Any]], run: dict[str, Any] | None) -> list[str]:
    """Human-readable progress for one chapter and target, computed from the manifest and run.json."""
    completed = sum(1 for record in records if record["status"] == "ok")
    remaining = sum(1 for record in records if record["status"] == "missing")
    failed = [record for record in records if record["status"] == "failed"]
    review = [record["id"] for record in records if record.get("needs_review")]
    state = run["status"] if run is not None else "no run recorded"
    lines = [
        f"{chapter_id} {target}: {state}",
        f"  completed {completed} of {len(records)} chunk(s); remaining {remaining}; "
        f"failed {len(failed)}; reused {run['reused'] if run else 0} in the last run",
    ]
    if run is not None:
        current = run.get("current_chunk") or "none"
        lines.append(f"  current chapter {run['chapter']}, last chunk {current}")
        lines.append(
            f"  elapsed this run {format_elapsed(run['elapsed_seconds'])}; "
            f"cumulative {format_elapsed(run['cumulative_seconds'])}"
        )
    lines.append(f"  needs review: {', '.join(review) if review else 'none'}")
    failures = run["recent_failures"] if run is not None else []
    if failures:
        lines.append("  recent failures:")
        lines.extend(f"    {item['chunk']}: {item['error']}" for item in failures)
    else:
        lines.append("  recent failures: none")
    return lines
