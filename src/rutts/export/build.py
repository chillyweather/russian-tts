from __future__ import annotations

import json
import os
import re
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from rutts.atomic import atomic_write_text
from rutts.book.project import CHAPTERS_DIR, BookProject, read_json, sha256_file, sha256_text, write_json
from rutts.book.render import MANIFEST_FILE, resolve_target
from rutts.export import loudness
from rutts.export.tools import ffmpeg_version, require_tool, run
from rutts.export.validate import ExpectedChapter, check, coverage_checks, m4b_checks, mp3_checks, ok
from rutts.review.eligibility import Eligibility, export_eligibility

EXPORTS_DIR = "exports"
RECORD_FILE = "export.json"
BUILDS_FILE = "builds.json"
WORK_DIR = "work"
DEFAULT_NAME = "audiobook"
DRAFT_SUFFIX = " (ЧЕРНОВИК)"
SCHEMA_VERSION = 1
KINDS = ("reviewed", "draft")
CODECS = {"m4b": "aac", "mp3": "libmp3lame"}


@dataclass(frozen=True)
class ExportSettings:
    name: str = DEFAULT_NAME
    bitrate: str = "96k"
    loudness: bool = True
    threshold_db: float = 2.0
    target_dbfs: float = -20.0


@dataclass(frozen=True)
class Outcome:
    status: str  # ready | invalid | refused
    messages: tuple[str, ...] = ()
    record: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Processed:
    chapters: list[tuple[str, str, np.ndarray]]
    rate: int
    processing: dict[str, Any]


def export_dir(project: BookProject, name: str) -> Path:
    return project.directory / EXPORTS_DIR / name


def record_path(project: BookProject, name: str) -> Path:
    return export_dir(project, name) / RECORD_FILE


def included_ids(project: BookProject) -> list[str]:
    return [chapter.id for chapter in project.read_source().chapters if chapter.included]


def settings_document(kind: str, target: str, formats: tuple[str, ...], settings: ExportSettings) -> dict[str, Any]:
    return {
        "kind": kind,
        "target": target,
        "formats": list(formats),
        "bitrate": settings.bitrate,
        "codecs": {name: CODECS[name] for name in formats},
        "loudness": {
            "enabled": settings.loudness,
            "method": loudness.METHOD if settings.loudness else None,
            "threshold_db": settings.threshold_db,
            "target_dbfs": settings.target_dbfs,
        },
    }


def _chapter_audio(project: BookProject, chapter_id: str, target: str) -> dict[str, Any]:
    report_path = project.directory / CHAPTERS_DIR / f"{chapter_id}.json"
    manifest_path = project.target_dir(chapter_id, target) / MANIFEST_FILE
    report = read_json(report_path) if report_path.is_file() else None
    records: list[dict[str, Any]] = read_json(manifest_path)["chunks"] if manifest_path.is_file() else []
    master = project.directory / report["master_file"] if report else None
    return {
        "master_sha256": sha256_file(master) if master is not None and master.is_file() else None,
        "chunk_signature_sha256": sha256_text(json.dumps(report["chunk_signature"], sort_keys=True)) if report else None,
        "chunks": [[record["id"], record.get("audio_sha256") if record["status"] == "ok" else None] for record in records],
    }


def components(project: BookProject, chapter_ids: list[str], target: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Everything an export depends on; its signature is the hash of this document."""
    titles = {chapter.id: chapter.title for chapter in project.read_source().chapters}
    return {
        "chapters": [[chapter_id, titles[chapter_id]] for chapter_id in chapter_ids],
        "audio": {chapter_id: _chapter_audio(project, chapter_id, target) for chapter_id in chapter_ids},
        "metadata": {"title": project.title, "author": project.author},
        "settings": settings,
    }


def signature_of(document: dict[str, Any]) -> str:
    return sha256_text(json.dumps(document, sort_keys=True, ensure_ascii=False))


def _append_build(directory: Path, record: dict[str, Any]) -> None:
    """Every build, in order, so later acceptance evidence can see a rebuilt export superseding an earlier one."""
    path = directory / BUILDS_FILE
    entries: list[dict[str, Any]] = read_json(path)["entries"] if path.is_file() else []
    entries.append(
        {
            "seq": len(entries) + 1,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": record["kind"],
            "status": record["status"],
            "target": record["target"],
            "signature": record["input"]["signature"],
        }
    )
    write_json(path, {"schema_version": SCHEMA_VERSION, "entries": entries})


def _has_master(project: BookProject, chapter_id: str) -> bool:
    report_path = project.directory / CHAPTERS_DIR / f"{chapter_id}.json"
    return report_path.is_file() and (project.directory / read_json(report_path)["master_file"]).is_file()


def _lossless_snapshot(project: BookProject, chapter_ids: list[str], target: str) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for chapter_id in chapter_ids:
        report_path = project.directory / CHAPTERS_DIR / f"{chapter_id}.json"
        if report_path.is_file():
            master = project.directory / read_json(report_path)["master_file"]
            if master.is_file():
                snapshot[master.relative_to(project.directory).as_posix()] = sha256_file(master)
        manifest_path = project.target_dir(chapter_id, target) / MANIFEST_FILE
        if manifest_path.is_file():
            for record in read_json(manifest_path)["chunks"]:
                path = project.target_dir(chapter_id, target) / str(record["audio_file"])
                if record["status"] == "ok" and path.is_file():
                    snapshot[path.relative_to(project.directory).as_posix()] = sha256_file(path)
    return snapshot


def _resample(source: Path, rate: int, destination: Path) -> np.ndarray:
    result = run([require_tool("ffmpeg"), "-v", "error", "-nostdin", "-y", "-i", str(source), "-ar", str(rate), "-c:a", "pcm_f32le", str(destination)])
    if result.returncode != 0:
        raise RuntimeError(f"resampling failed: {result.stderr.strip()[:300]}")
    samples, _ = sf.read(str(destination), dtype="float32", always_2d=False)
    return np.asarray(samples, dtype=np.float32).reshape(-1)


def _process(project: BookProject, chapter_ids: list[str], titles: dict[str, str], settings: ExportSettings, work: Path) -> _Processed:
    sources: dict[str, tuple[Path, np.ndarray, int]] = {}
    for chapter_id in chapter_ids:
        path = project.directory / read_json(project.directory / CHAPTERS_DIR / f"{chapter_id}.json")["master_file"]
        samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
        sources[chapter_id] = (path, np.asarray(samples, dtype=np.float32).reshape(-1), int(rate))
    source_rates = {chapter_id: rate for chapter_id, (_, _, rate) in sources.items()}
    counts = Counter(source_rates.values())
    out_rate = max(counts, key=lambda rate: (counts[rate], rate))
    resampled: list[dict[str, Any]] = []
    audio: dict[str, np.ndarray] = {}
    for chapter_id, (path, samples, rate) in sources.items():
        if rate != out_rate:
            samples = _resample(path, out_rate, work / f"{chapter_id}.resampled.wav")
            resampled.append({"id": chapter_id, "from": rate, "to": out_rate})
        audio[chapter_id] = samples
    measurements = {chapter_id: loudness.measure(samples, out_rate) for chapter_id, samples in audio.items()}
    median: float | None = None
    if settings.loudness:
        median, gains = loudness.plan(measurements, settings.threshold_db, settings.target_dbfs)
    else:
        gains = [loudness.Gain(chapter_id, measurement, False, 0.0, 0.0) for chapter_id, measurement in measurements.items()]
    by_id = {gain.chapter_id: gain for gain in gains}
    rows: list[dict[str, Any]] = []
    chapters: list[tuple[str, str, np.ndarray]] = []
    for chapter_id, samples in audio.items():
        gain = by_id[chapter_id]
        processed = loudness.apply_gain(samples, gain.applied_db)
        assert len(processed) == len(samples)
        sf.write(str(work / f"{chapter_id}.wav"), processed, out_rate, subtype="FLOAT")
        rows.append(
            {
                "id": chapter_id,
                "measured_gated_rms_dbfs": gain.measurement.gated_rms_dbfs,
                "sample_peak_dbfs": gain.measurement.sample_peak_dbfs,
                "corrective": gain.corrective,
                "requested_gain_db": gain.requested_db,
                "applied_gain_db": gain.applied_db,
                "reduced_for_peak": gain.reduced_for_peak,
                "reduction_db": gain.reduction_db,
                "output_peak_dbfs": loudness.sample_peak_dbfs(processed),
            }
        )
        chapters.append((chapter_id, titles[chapter_id], processed))
    processing: dict[str, Any] = {
        "loudness": {
            "enabled": settings.loudness,
            "method": loudness.METHOD if settings.loudness else None,
            "target_dbfs": settings.target_dbfs,
            "threshold_db": settings.threshold_db,
            "median_gated_rms_dbfs": median,
            "chapters": rows,
        },
        "sample_rate": {"source": source_rates, "output": out_rate},
        "resampling": {
            "used": bool(resampled),
            "resampler": "ffmpeg swresample (default resampler)" if resampled else None,
            "chapters": resampled,
        },
        "channels": 1,
        "bitrate": settings.bitrate,
    }
    return _Processed(chapters=chapters, rate=out_rate, processing=processing)


def _escape(value: str) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", value)


def ffmetadata(title: str, author: str, chapters: list[tuple[str, int, int]]) -> str:
    lines = [";FFMETADATA1", f"title={_escape(title)}"]
    if author:
        lines += [f"artist={_escape(author)}", f"album_artist={_escape(author)}"]
    lines += [f"album={_escape(title)}", "genre=Audiobook"]
    for chapter_title, start_ms, end_ms in chapters:
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start_ms}", f"END={end_ms}", f"title={_escape(chapter_title)}"]
    return "\n".join(lines) + "\n"


def _safe_name(title: str) -> str:
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", title).strip()


def _error(result: Any) -> str | None:
    if result.returncode == 0:
        return None
    return result.stderr.strip()[:300] or "encoder failed"


def _encode_m4b(combined: Path, metadata: Path, out: Path, bitrate: str) -> str | None:
    return _error(
        run(
            [
                require_tool("ffmpeg"), "-v", "error", "-nostdin", "-y",
                "-i", str(combined), "-i", str(metadata),
                "-map_metadata", "1", "-map_chapters", "1", "-map", "0:a",
                "-c:a", "aac", "-b:a", bitrate, "-ac", "1",
                "-f", "mp4", "-movflags", "+faststart", str(out),
            ]
        )
    )


def _encode_mp3(source: Path, out: Path, bitrate: str, tags: dict[str, str]) -> str | None:
    argv = [require_tool("ffmpeg"), "-v", "error", "-nostdin", "-y", "-i", str(source), "-c:a", "libmp3lame", "-b:a", bitrate, "-ac", "1", "-f", "mp3"]
    for key, value in tags.items():
        argv += ["-metadata", f"{key}={value}"]
    return _error(run([*argv, str(out)]))


def _temp(directory: Path, suffix: str) -> Path:
    return directory / f".{uuid.uuid4().hex}.tmp{suffix}"


def _file_entry(project: BookProject, format_name: str, path: Path) -> dict[str, Any]:
    return {
        "format": format_name,
        "path": path.relative_to(project.directory).as_posix(),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def build_export(
    project: BookProject,
    target: str | None,
    kind: str,
    formats: tuple[str, ...],
    settings: ExportSettings,
) -> Outcome:
    included = [chapter for chapter in project.read_source().chapters if chapter.included]
    if not included:
        return Outcome("refused", ("no included chapters to export",))
    ids = [chapter.id for chapter in included]
    titles = {chapter.id: chapter.title for chapter in included}
    resolved = resolve_target(project, ids[0], target)
    elig: Eligibility = export_eligibility(project, ids, resolved)
    gaps: list[str] = []
    if kind == "reviewed":
        if not elig.reviewed.eligible:
            return Outcome("refused", elig.reviewed.reasons)
        build_ids = ids
    else:
        if not elig.draft.eligible:
            return Outcome("refused", elig.draft.reasons)
        build_ids = [chapter_id for chapter_id in ids if _has_master(project, chapter_id)]
        gaps = list(elig.draft.reasons) + [f"{chapter_id}: skipped, no assembled master" for chapter_id in ids if chapter_id not in build_ids]
        if not build_ids:
            return Outcome("refused", ("no chapter has an assembled master to export",))

    draft = kind == "draft"
    document = components(project, ids, resolved, settings_document(kind, resolved, formats, settings))
    signature = signature_of(document)
    before = _lossless_snapshot(project, ids, resolved)
    directory = export_dir(project, settings.name)
    work = directory / WORK_DIR
    work.mkdir(parents=True, exist_ok=True)
    processed = _process(project, build_ids, titles, settings, work)
    combined = work / "book.wav"
    sf.write(str(combined), np.concatenate([samples for _, _, samples in processed.chapters]), processed.rate, subtype="FLOAT")

    rate = processed.rate
    chapters: list[dict[str, Any]] = []
    position = 0
    for chapter_id, title, samples in processed.chapters:
        start = position
        position += len(samples)
        chapters.append(
            {
                "id": chapter_id,
                "title": title,
                "samples": len(samples),
                "start_ms": round(start * 1000 / rate),
                "end_ms": round(position * 1000 / rate),
            }
        )
    expected = [ExpectedChapter(item["title"], item["start_ms"] / 1000) for item in chapters]
    total_seconds = position / rate
    title = project.title + (DRAFT_SUFFIX if draft else "")
    stem = settings.name + (".draft" if draft else "")

    checks: list[dict[str, Any]] = coverage_checks(project, build_ids, resolved, reviewed=not draft)
    errors: list[str] = []
    staged: list[tuple[Path, Path, str]] = []
    metadata_file = work / "ffmetadata.txt"
    atomic_write_text(metadata_file, ffmetadata(title, project.author, [(item["title"], item["start_ms"], item["end_ms"]) for item in chapters]))
    if "m4b" in formats:
        temp = _temp(directory, ".m4b")
        error = _encode_m4b(combined, metadata_file, temp, settings.bitrate)
        if error:
            errors.append(f"m4b encoder: {error}")
        else:
            checks += m4b_checks(temp, expected, total_seconds, title, project.author)
            staged.append((temp, directory / f"{stem}.m4b", "m4b"))
    mp3_names: list[str] = []
    if "mp3" in formats:
        for number, (chapter_id, chapter_title, _) in enumerate(processed.chapters, start=1):
            name = f"{number:02d} - {_safe_name(chapter_title)}{'.draft' if draft else ''}.mp3"
            mp3_names.append(name)
            temp = _temp(directory, ".mp3")
            tags = {"title": chapter_title, "album": title, "artist": project.author, "track": f"{number}/{len(processed.chapters)}"}
            error = _encode_mp3(work / f"{chapter_id}.wav", temp, settings.bitrate, {key: value for key, value in tags.items() if value})
            if error:
                errors.append(f"mp3 encoder for {chapter_id}: {error}")
            else:
                checks += mp3_checks(temp, chapter_title, title, project.author, number, len(processed.chapters))
                staged.append((temp, directory / name, "mp3"))

    after = _lossless_snapshot(project, ids, resolved)
    changed = sorted(path for path in before if after.get(path) != before[path])
    checks.append(check("lossless chunks and masters unchanged", not changed and bool(before), ", ".join(changed)))
    if errors:
        checks.append(check("encoders succeeded", False, "; ".join(errors)))
    valid = ok(checks)

    files: list[dict[str, Any]] = []
    if valid:
        for temp, final, format_name in staged:
            os.replace(temp, final)
            files.append(_file_entry(project, format_name, final))
        if "mp3" in formats:
            playlist = directory / f"{stem}.m3u8"
            atomic_write_text(playlist, "\n".join(mp3_names) + "\n")
            files.append(_file_entry(project, "m3u8", playlist))
    else:
        for temp, _, _ in staged:
            temp.unlink(missing_ok=True)

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": settings.name,
        "kind": kind,
        "status": "ready" if valid else "invalid",
        "target": resolved,
        "formats": list(formats),
        "files": files,
        "processing": {**processed.processing, "ffmpeg": ffmpeg_version(), "codecs": {name: CODECS[name] for name in formats}},
        "chapters": chapters,
        "metadata": {"title": title, "author": project.author, "album": title, "genre": "Audiobook", "date": None, "cover": None},
        "gaps": gaps,
        "eligibility": {"reviewed": asdict(elig.reviewed), "draft": asdict(elig.draft)},
        "validation": {"ok": valid, "checks": checks},
        "lossless": {"checked_files": len(before), "retained": not changed and bool(before), "changed": changed},
        "input": {"signature": signature, "components": document},
        "work_dir": work.relative_to(project.directory).as_posix(),
    }
    write_json(directory / RECORD_FILE, record)
    _append_build(directory, record)
    if not valid:
        failed = [f"{item['name']}: {item['detail']}".rstrip(": ") for item in checks if not item["ok"]] + errors
        return Outcome("invalid", tuple(failed), record)
    return Outcome("ready", tuple(gaps), record)
