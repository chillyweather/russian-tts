from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rutts.book.project import CHAPTERS_DIR, BookProject, read_json
from rutts.book.render import MANIFEST_FILE
from rutts.export.tools import probe, require_tool, run
from rutts.validation import ValidationError

CHAPTER_TOLERANCE_S = 0.05
DURATION_TOLERANCE_S = 0.25
DURATION_TOLERANCE_RATIO = 0.001


@dataclass(frozen=True)
class ExpectedChapter:
    title: str
    start_seconds: float


def check(name: str, ok: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def decode_check(path: Path) -> dict[str, Any]:
    result = run([require_tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(path), "-f", "null", "-"])
    errors = result.stderr.strip()
    return check("decodes without errors", result.returncode == 0 and not errors, errors[:300])


def m4b_checks(path: Path, chapters: list[ExpectedChapter], total_seconds: float, title: str, author: str) -> list[dict[str, Any]]:
    checks = [decode_check(path)]
    try:
        document = probe(path)
    except ValidationError as exc:
        return checks + [check("ffprobe reads the file", False, "; ".join(exc.messages))]
    found = document.get("chapters", [])
    titles = [str(item.get("tags", {}).get("title", "")) for item in found]
    checks.append(check("chapter count", len(found) == len(chapters), f"expected {len(chapters)}, found {len(found)}"))
    checks.append(check("chapter order and titles", titles == [item.title for item in chapters], "; ".join(titles)))
    starts = [float(item.get("start_time", "nan")) for item in found]
    starts_ok = len(starts) == len(chapters) and all(
        abs(start - expected.start_seconds) <= CHAPTER_TOLERANCE_S for start, expected in zip(starts, chapters, strict=False)
    )
    checks.append(check("chapter start times", starts_ok, f"tolerance {CHAPTER_TOLERANCE_S}s"))
    tags = document.get("format", {}).get("tags", {})
    checks.append(check("title tag", tags.get("title", "") == title, str(tags.get("title", ""))))
    checks.append(check("author tag", tags.get("artist", "") == author, str(tags.get("artist", ""))))
    duration = float(document.get("format", {}).get("duration", "nan"))
    tolerance = max(DURATION_TOLERANCE_S, DURATION_TOLERANCE_RATIO * total_seconds)
    checks.append(check("duration", abs(duration - total_seconds) <= tolerance, f"{duration:.3f}s vs {total_seconds:.3f}s"))
    return checks


def mp3_checks(path: Path, title: str, album: str, author: str, track: int, total: int) -> list[dict[str, Any]]:
    checks = [decode_check(path)]
    try:
        tags = probe(path).get("format", {}).get("tags", {})
    except ValidationError as exc:
        return checks + [check("ffprobe reads the file", False, "; ".join(exc.messages))]
    checks.append(check("title tag", tags.get("title", "") == title, str(tags.get("title", ""))))
    checks.append(check("album and artist tags", tags.get("album", "") == album and tags.get("artist", "") == author))
    checks.append(check("track tag", tags.get("track", "") == f"{track}/{total}", str(tags.get("track", ""))))
    return checks


def coverage_checks(project: BookProject, chapter_ids: list[str], target: str, reviewed: bool) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for chapter_id in chapter_ids:
        report = read_json(project.directory / CHAPTERS_DIR / f"{chapter_id}.json")
        manifest = read_json(project.target_dir(chapter_id, target) / MANIFEST_FILE)
        reported_ok = {item["id"] for item in report["chunks"] if item["status"] == "ok"}
        needed = {item["id"] for item in manifest["chunks"] if item["status"] == "ok"}
        absent = sorted(needed - reported_ok)
        checks.append(check(f"{chapter_id} chunk coverage", not absent, ", ".join(absent)))
        if reviewed:
            gaps = list(report["missing_chunks"]) + list(report["failed_chunks"])
            checks.append(check(f"{chapter_id} no missing or failed chunks", not gaps, ", ".join(gaps)))
    return checks


def ok(checks: list[dict[str, Any]]) -> bool:
    return all(item["ok"] for item in checks)
