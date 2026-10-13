from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rutts import resources
from rutts.book.assemble import assemble_chapter, chunk_status
from rutts.book.chunking import Chunk, limits_for_target, plan_chunks
from rutts.book.project import SCHEMA_VERSION, BookProject, SourceChapter, SourceDocument, read_json, sha256_text, write_json
from rutts.book.render import MANIFEST_FILE, ConfigChangeRequired, StopRequested, is_fatal, record_fingerprint, render_chapter
from rutts.book.runstate import format_elapsed
from rutts.engines.base import EngineConfig, SynthesisEngine, SynthesisResult
from rutts.narrator import NarratorProfile
from rutts.pronunciation.markup import effective_convention, read_markup, translate_chunks
from rutts.resources import UNAVAILABLE
from rutts.validation import ValidationError

JOB_DIR = "job"
JOB_FILE = "job/job.json"
STATE_FILE = "job/state.json"
COVERAGE_FILE = "job/coverage.json"
RESOURCES_FILE = "job/resources.jsonl"
DIAGNOSTICS_DIR = "job/diagnostics"
DEFAULT_POLICY: dict[str, Any] = {"retries": 2, "max_swap_gb": 4.0, "min_free_percent": 5.0, "max_rss_gb": 13.0}
CONFIG_KEYS = ("engine", "narrator", "target", "limits", "policy")
RECENT_FAILURES = 10
ESTIMATE_AFTER_CHUNKS = 3
GIB = 1024**3
STOPPED = "stopped_resource_pressure"
_THROUGHPUT_KEYS = ("rendered_chunks", "render_seconds", "rendered_characters", "audio_seconds")


@dataclass(frozen=True)
class JobRequest:
    target: str | None = None
    chapters: tuple[str, ...] | None = None
    from_chapter: str | None = None
    retries: int | None = None
    retry_failed: bool = False
    accept_config_change: bool = False
    assemble: bool = False
    max_swap_gb: float | None = None
    min_free_percent: float | None = None
    max_rss_gb: float | None = None


@dataclass(frozen=True)
class JobResult:
    status: str
    exit_code: int
    state: dict[str, Any]
    diagnostics: Path | None


class EngineSession:
    """Loads the engine on first use and keeps it loaded until the job closes.

    Per-chapter load and unload calls made by ``render_chapter`` become no-ops, so the job loads once.
    """

    def __init__(self, engine: SynthesisEngine) -> None:
        self._engine = engine
        self._loaded = False

    @property
    def config(self) -> EngineConfig:
        return self._engine.config

    def load(self) -> None:
        if not self._loaded:
            self._engine.load()
            self._loaded = True

    def unload(self) -> None:
        """Deliberately a no-op: the job unloads once, in close()."""

    def close(self) -> None:
        if self._loaded:
            self._engine.unload()
            self._loaded = False

    def synthesize(self, text: str, narrator: NarratorProfile) -> SynthesisResult:
        self.load()
        return self._engine.synthesize(text, narrator)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _merge_policy(request: JobRequest, base: dict[str, Any]) -> dict[str, Any]:
    requested: dict[str, Any] = {
        "retries": request.retries,
        "max_swap_gb": request.max_swap_gb,
        "min_free_percent": request.min_free_percent,
        "max_rss_gb": request.max_rss_gb,
    }
    return {key: base[key] if value is None else value for key, value in requested.items()}


def _configuration(project: BookProject, engine: SynthesisEngine, target: str, policy: dict[str, Any]) -> dict[str, Any]:
    profile = project.narrator()
    return {
        "engine": engine.config.to_dict(),
        "engine_fingerprint": engine.config.fingerprint(),
        "narrator": {
            "name": profile.name,
            "language": profile.language,
            "provenance": profile.provenance,
            "reference_sha256": profile.reference_sha256,
            "sha256": project.narrator_sha256(),
        },
        "target": target,
        "limits": limits_for_target(target).to_dict(),
        "policy": policy,
    }


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(f"{prefix}.{key}" if prefix else str(key), item))
        return flat
    return {prefix: value}


def _differences(recorded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    before = _flatten("", {key: recorded.get(key) for key in CONFIG_KEYS})
    after = _flatten("", {key: current.get(key) for key in CONFIG_KEYS})
    return [
        f"{key}: {before.get(key)!r} -> {after.get(key)!r}"
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    ]


def _open_job(
    project: BookProject, configuration: dict[str, Any], accept_config_change: bool
) -> dict[str, Any]:
    """Create job/job.json on first use; refuse a different configuration unless it is accepted."""
    path = project.directory / JOB_FILE
    if not path.is_file():
        document = {"schema_version": SCHEMA_VERSION, "created_at": _now(), "superseded_configs": [], **configuration}
        write_json(path, document)
        return document
    recorded = read_json(path)
    differences = _differences(recorded, configuration)
    if not differences:
        return recorded
    if not accept_config_change:
        raise ConfigChangeRequired(
            [
                "job configuration differs from the job record: " + "; ".join(differences),
                "pass --accept-config-change to supersede the job configuration; affected chunks are invalidated "
                "and regenerated",
            ]
        )
    superseded = [
        *recorded["superseded_configs"],
        {"superseded_at": _now(), "configuration": {key: recorded[key] for key in CONFIG_KEYS}},
    ]
    document = {"schema_version": SCHEMA_VERSION, "created_at": recorded["created_at"], "superseded_configs": superseded, **configuration}
    write_json(path, document)
    return document


def _select(source: SourceDocument, request: JobRequest) -> list[SourceChapter]:
    if request.chapters:
        wanted = set(request.chapters)
        for chapter_id in sorted(wanted):
            source.chapter(chapter_id)
        return [chapter for chapter in source.chapters if chapter.id in wanted]
    if request.from_chapter:
        return source.chapters[source.chapter_index(request.from_chapter) :]
    return list(source.chapters)


def _plan(project: BookProject, chapter_id: str, target: str, engine_id: str) -> list[Chunk]:
    convention = effective_convention(engine_id, read_markup(project.directory))
    planned = plan_chunks(project.planning_segments(chapter_id), target, limits_for_target(target))
    return translate_chunks(planned, convention)


def _records(directory: Path) -> dict[str, dict[str, Any]]:
    path = directory / MANIFEST_FILE
    if not path.is_file():
        return {}
    return {str(record["id"]): record for record in read_json(path)["chunks"]}


def _chunk_state(
    chunk: Chunk, record: dict[str, Any] | None, directory: Path, fingerprint: str, narrator_sha256: str
) -> str:
    """ok only when the audio verifies and was rendered under the job's current configuration."""
    status = chunk_status(chunk, record, directory)
    if status == "ok" and record is not None:
        fresh = record_fingerprint(sha256_text(chunk.text), fingerprint, narrator_sha256)
        if record.get("fingerprint") != fresh:
            return "missing"
    return status


def _scrub(text: str, project_dir: Path) -> str:
    return text.replace(str(project_dir), "<project>")


class _Job:
    """Observes each chunk of the chapters it renders and keeps job/state.json, resources and progress current."""

    def __init__(
        self,
        project: BookProject,
        engine: SynthesisEngine,
        target: str,
        policy: dict[str, Any],
        sampler: Callable[[], dict[str, Any]],
        progress: Callable[[str], None],
        previous: dict[str, Any] | None,
    ) -> None:
        self.project = project
        self.directory = project.directory / JOB_DIR
        self.target = target
        self.policy = policy
        self.sampler = sampler
        self.progress = progress
        self.source = project.read_source()
        self.narration = {segment.id: segment for segment in project.read_narration().segments}
        self.engine_id = engine.config.engine_id
        self.fingerprint = engine.config.fingerprint()
        self.narrator_sha256 = project.narrator_sha256()
        self.limits = limits_for_target(target)
        self.plans = {
            chapter.id: _plan(project, chapter.id, target, self.engine_id)
            for chapter in self.source.chapters
            if chapter.included
        }
        self.positions = {chapter.id: index for index, chapter in enumerate(self.source.chapters, start=1)}
        self.started = time.perf_counter()
        prior = previous or {}
        self.cumulative_before = float(prior.get("cumulative_seconds", 0.0))
        self.invocations = int(prior.get("invocations", 0)) + 1
        self.throughput: dict[str, float] = {
            "rendered_chunks": 0,
            "render_seconds": 0.0,
            "rendered_characters": 0,
            "audio_seconds": 0.0,
            **{key: value for key, value in prior.get("throughput", {}).items() if key in _THROUGHPUT_KEYS},
        }
        self.failures: list[dict[str, str]] = list(prior.get("recent_failures", []))[-RECENT_FAILURES:]
        self.resources: dict[str, Any] = dict(
            prior.get("resources")
            or {"samples": 0, "unavailable_samples": 0, "max_swap_used_bytes": None, "min_memory_free_percent": None, "max_peak_rss_bytes": None}
        )
        self.swap_baseline: int | None = None
        self.low_free_streak = 0
        self.samples: list[dict[str, Any]] = []
        self.chapter_id: str | None = None
        self.chunk: Chunk | None = None
        self.errors: dict[str, list[str]] = {}
        self.stop_reasons: list[str] = []

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def begin_chapter(self, chapter_id: str) -> None:
        self.chapter_id = chapter_id
        chapter = self.source.chapter(chapter_id)
        self.progress(
            f"chapter {chapter_id} ({self.positions[chapter_id]}/{len(self.source.chapters)}) {chapter.title}: "
            f"{len(self.plans[chapter_id])} chunk(s)"
        )

    # survey and documents

    def survey(self) -> dict[str, dict[str, dict[str, Any]]]:
        states: dict[str, dict[str, dict[str, Any]]] = {}
        for chapter_id, plan in self.plans.items():
            directory = self.project.target_dir(chapter_id, self.target)
            records = _records(directory)
            states[chapter_id] = {}
            for chunk in plan:
                record = records.get(chunk.id)
                state = _chunk_state(chunk, record, directory, self.fingerprint, self.narrator_sha256)
                states[chapter_id][chunk.id] = {
                    "status": state,
                    "audio_sha256": record["audio_sha256"] if state == "ok" and record else None,
                }
        return states

    def chapter_complete(self, chapter_id: str) -> bool:
        return all(entry["status"] == "ok" for entry in self.survey()[chapter_id].values())

    def coverage(self, survey: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
        segments: list[dict[str, Any]] = []
        for chapter in self.source.chapters:
            plan = self.plans.get(chapter.id, [])
            for segment in chapter.segments:
                included = chapter.included and segment.included
                if included:
                    reason = None
                elif not segment.included:
                    reason = segment.exclusion_reason
                else:
                    reason = chapter.exclusion_reason or "chapter excluded"
                chunks = [
                    {"id": chunk.id, **survey[chapter.id][chunk.id]}
                    for chunk in plan
                    if segment.id in chunk.narration_segment_ids
                ] if included else []
                narratable = included and segment.kind == "paragraph" and any(
                    char.isalpha() for char in self.narration[segment.id].text
                )
                covered = bool(chunks) and all(entry["status"] == "ok" for entry in chunks)
                segments.append(
                    {
                        "segment_id": segment.id,
                        "chapter_id": chapter.id,
                        "kind": segment.kind,
                        "included": included,
                        "exclusion_reason": reason,
                        "narratable": narratable,
                        "covered": covered,
                        "chunks": chunks,
                    }
                )
        uncovered = [item["segment_id"] for item in segments if item["narratable"] and not item["covered"]]
        return {
            "schema_version": SCHEMA_VERSION,
            "target": self.target,
            "complete": not uncovered,
            "totals": {
                "segments": len(segments),
                "included": sum(1 for item in segments if item["included"]),
                "covered": sum(1 for item in segments if item["included"] and item["covered"]),
                "uncovered": len(uncovered),
                "excluded": sum(1 for item in segments if not item["included"]),
            },
            "uncovered_segments": uncovered,
            "segments": segments,
        }

    def _estimate(self, remaining_chars: int) -> dict[str, Any]:
        rendered = int(self.throughput["rendered_chunks"])
        if rendered < ESTIMATE_AFTER_CHUNKS or self.throughput["rendered_characters"] <= 0:
            return {"available": False, "remaining_seconds": None, "basis": None}
        per_character = self.throughput["render_seconds"] / self.throughput["rendered_characters"]
        return {
            "available": True,
            "remaining_seconds": round(per_character * remaining_chars, 1),
            "basis": f"estimate from observed throughput over {rendered} chunk(s) rendered in this job",
        }

    def document(self, status: str) -> dict[str, Any]:
        survey = self.survey()
        chapters: dict[str, dict[str, Any]] = {}
        totals = {"total": 0, "ok": 0, "failed": 0, "remaining": 0}
        remaining_chars = 0
        for chapter in self.source.chapters:
            if not chapter.included:
                chapters[chapter.id] = {
                    "title": chapter.title,
                    "status": "excluded",
                    "reason": chapter.exclusion_reason,
                    "total": 0,
                    "ok": 0,
                    "failed": 0,
                    "remaining": 0,
                    "failed_chunks": [],
                }
                continue
            plan = self.plans[chapter.id]
            states = survey[chapter.id]
            ok = sum(1 for chunk in plan if states[chunk.id]["status"] == "ok")
            failed = [chunk.id for chunk in plan if states[chunk.id]["status"] == "failed"]
            remaining_chars += sum(len(chunk.text) for chunk in plan if states[chunk.id]["status"] != "ok")
            if ok == len(plan):
                row_status = "completed"
            elif chapter.id == self.chapter_id and status in ("interrupted", STOPPED):
                row_status = status
            elif failed or chapter.id in self.errors:
                row_status = "incomplete"
            else:
                row_status = "pending"
            chapters[chapter.id] = {
                "title": chapter.title,
                "status": row_status,
                "reason": "; ".join(self.errors[chapter.id]) if chapter.id in self.errors else None,
                "total": len(plan),
                "ok": ok,
                "failed": len(failed),
                "remaining": len(plan) - ok - len(failed),
                "failed_chunks": failed,
            }
            totals["total"] += len(plan)
            totals["ok"] += ok
            totals["failed"] += len(failed)
            totals["remaining"] += len(plan) - ok - len(failed)
        rendered_chars = self.throughput["rendered_characters"]
        audio = self.throughput["audio_seconds"]
        throughput = {
            **{key: self.throughput[key] for key in _THROUGHPUT_KEYS},
            "render_seconds_per_audio_second": round(self.throughput["render_seconds"] / audio, 4) if audio else None,
            "seconds_per_character": round(self.throughput["render_seconds"] / rendered_chars, 6) if rendered_chars else None,
        }
        elapsed = self.elapsed()
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "target": self.target,
            "engine": self.engine_id,
            "engine_fingerprint": self.fingerprint,
            "current_chapter": self.chapter_id,
            "current_chunk": self.chunk.id if self.chunk is not None else None,
            "invocations": self.invocations,
            "elapsed_seconds": round(elapsed, 3),
            "cumulative_seconds": round(self.cumulative_before + elapsed, 3),
            "chunks": totals,
            "chapters": chapters,
            "recent_failures": self.failures[-RECENT_FAILURES:],
            "throughput": throughput,
            "estimate": self._estimate(remaining_chars),
            "resources": self.resources,
            "stop": {"reasons": self.stop_reasons} if self.stop_reasons else None,
        }

    def write_state(self, status: str) -> dict[str, Any]:
        document = self.document(status)
        write_json(self.project.directory / STATE_FILE, document)
        return document

    # resources and progress

    def _sample(self, phase: str) -> dict[str, Any]:
        measurements = dict(self.sampler())
        unavailable = [key for key, value in measurements.items() if value == UNAVAILABLE]
        self.resources["samples"] += 1
        if unavailable:
            self.resources["unavailable_samples"] += 1
        swap = measurements.get("swap_used_bytes")
        if isinstance(swap, int):
            self.resources["max_swap_used_bytes"] = max(swap, self.resources["max_swap_used_bytes"] or 0)
        free = measurements.get("memory_free_percent")
        if isinstance(free, int):
            current = self.resources["min_memory_free_percent"]
            self.resources["min_memory_free_percent"] = free if current is None else min(free, current)
        rss = measurements.get("peak_rss_bytes")
        if isinstance(rss, int):
            self.resources["max_peak_rss_bytes"] = max(rss, self.resources["max_peak_rss_bytes"] or 0)
        self.samples.append(
            {"chapter": self.chapter_id, "chunk": self.chunk.id if self.chunk else None, "phase": phase, "measurements": measurements}
        )
        return measurements

    def _pressure(self, sample: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        swap = sample.get("swap_used_bytes")
        if isinstance(swap, int):
            # Swap already in use by other apps is not ours; judge growth since the job started.
            if self.swap_baseline is None:
                self.swap_baseline = swap
            growth = swap - self.swap_baseline
            if growth > self.policy["max_swap_gb"] * GIB:
                reasons.append(
                    f"swap grew by {growth / GIB:.2f} GB since the job started, "
                    f"exceeding {self.policy['max_swap_gb']:g} GB"
                )
        free = sample.get("memory_free_percent")
        if isinstance(free, int) and free < self.policy["min_free_percent"]:
            self.low_free_streak += 1
        else:
            self.low_free_streak = 0
        if self.low_free_streak >= 2:
            reasons.append(
                f"free memory {free}% is below {self.policy['min_free_percent']:g}% on two consecutive samples"
            )
        rss = sample.get("peak_rss_bytes")
        if isinstance(rss, int) and rss > self.policy["max_rss_gb"] * GIB:
            reasons.append(f"process peak RSS {rss / GIB:.2f} GB exceeds {self.policy['max_rss_gb']:g} GB")
        return reasons

    def _log(self, chunk: Chunk, before: dict[str, Any], after: dict[str, Any] | None, reasons: list[str]) -> None:
        unavailable = sorted({key for sample in (before, after) if sample for key, value in sample.items() if value == UNAVAILABLE})
        line = {
            "chapter": self.chapter_id,
            "chunk": chunk.id,
            "before": before,
            "after": after,
            "unavailable": unavailable,
            "reasons": reasons,
        }
        path = self.project.directory / RESOURCES_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _stop(self, reasons: list[str]) -> None:
        self.stop_reasons = reasons
        raise StopRequested(STOPPED, reasons)

    def before_chunk(self, chunk: Chunk) -> None:
        self.chunk = chunk
        before = self._sample("before")
        reasons = self._pressure(before)
        if reasons:
            self._log(chunk, before, None, reasons)
            self._stop(reasons)

    def after_chunk(self, chunk: Chunk, record: dict[str, Any], seconds: float) -> None:
        after = self._sample("after")
        reasons = self._pressure(after)
        before = self.samples[-2]["measurements"]
        if record["status"] == "ok":
            self.throughput["rendered_chunks"] += 1
            self.throughput["render_seconds"] += float(record["render_seconds"])
            self.throughput["rendered_characters"] += len(chunk.text)
            self.throughput["audio_seconds"] += float(record["actual_duration_seconds"])
        else:
            error = str(record["attempts"][-1]["error"])
            self.failures = [*self.failures, {"chapter": self.chapter_id or "", "chunk": chunk.id, "error": error}][
                -RECENT_FAILURES:
            ]
        self.chunk = None
        self._log(chunk, before, after, reasons)
        document = self.write_state("running")
        self.progress(self._line(chunk, record, seconds, document))
        if reasons:
            self._stop(reasons)

    def _line(self, chunk: Chunk, record: dict[str, Any], seconds: float, document: dict[str, Any]) -> str:
        cid = str(self.chapter_id)
        chapter = document["chapters"][cid]
        plan = self.plans[cid]
        position = next(index for index, item in enumerate(plan, start=1) if item.id == chunk.id)
        book = document["chunks"]
        line = (
            f"{cid} ({self.positions[cid]}/{len(self.source.chapters)} {chapter['title']}) "
            f"{chunk.id} chunk {position}/{len(plan)} {record['status']} {seconds:.1f}s | "
            f"chapter {chapter['ok']}/{chapter['total']} done, {chapter['remaining']} remaining | "
            f"book {book['ok']}/{book['total']} done, {book['remaining']} remaining | "
            f"failed {book['failed']} | elapsed {format_elapsed(self.cumulative_before + self.elapsed())}"
        )
        estimate = document["estimate"]
        if estimate["available"]:
            line += (
                f" | estimate {format_elapsed(estimate['remaining_seconds'])} remaining "
                "(estimate from observed throughput)"
            )
        return line

    def diagnostics(self, status: str, reasons: list[str], error: BaseException | None) -> Path:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.project.directory / DIAGNOSTICS_DIR / f"{stamp}.json"
        document = {
            "status": status,
            "stopped_at": _now(),
            "reasons": reasons,
            "thresholds": {key: self.policy[key] for key in ("max_swap_gb", "min_free_percent", "max_rss_gb")},
            "chapter": self.chapter_id,
            "chunk": self.chunk.id if self.chunk is not None else None,
            "last_error": None if error is None else _scrub(f"{type(error).__name__}: {error}", self.project.directory),
            "samples": self.samples,
        }
        write_json(path, document)
        return path



def run_job(
    project: BookProject,
    engine: SynthesisEngine,
    request: JobRequest,
    *,
    sampler: Callable[[], dict[str, Any]] | None = None,
    progress: Callable[[str], None] = print,
) -> JobResult:
    """Render the requested chapters of a book in source order, resuming any work already valid."""
    state_path = project.directory / STATE_FILE
    previous_state = read_json(state_path) if state_path.is_file() else None
    previous_job = read_json(project.directory / JOB_FILE) if (project.directory / JOB_FILE).is_file() else None
    base_policy = previous_job["policy"] if previous_job else DEFAULT_POLICY
    base_target = previous_job["target"] if previous_job else "short"
    target = request.target or base_target
    policy = _merge_policy(request, base_policy)
    if policy["retries"] < 0:
        raise ValidationError(["--retries must be zero or positive"])
    configuration = _configuration(project, engine, target, policy)
    _open_job(project, configuration, request.accept_config_change)

    selected = _select(project.read_source(), request)
    for chapter in selected:
        if not chapter.included:
            progress(f"skipped {chapter.id} {chapter.title}: excluded ({chapter.exclusion_reason or 'no reason recorded'})")
    runnable = [chapter for chapter in selected if chapter.included]

    job = _Job(project, engine, target, policy, sampler or resources.sample, progress, previous_state)
    session = EngineSession(engine)
    status = "running"
    error: BaseException | None = None
    try:
        job.write_state("running")
        for chapter in runnable:
            job.begin_chapter(chapter.id)
            try:
                render_chapter(
                    project,
                    chapter.id,
                    session,
                    target,
                    limits=job.limits,
                    accept_config_change=request.accept_config_change,
                    retries=int(policy["retries"]),
                    retry_failed=request.retry_failed,
                    observer=job,
                )
            except ConfigChangeRequired:
                raise
            except ValidationError as exc:
                job.errors[chapter.id] = list(exc.messages)
                continue
            if request.assemble and job.chapter_complete(chapter.id):
                assemble_chapter(project, chapter.id, target)
        status = "finished"
    except KeyboardInterrupt:
        status = "interrupted"
    except StopRequested as exc:
        status = exc.status
        job.stop_reasons = exc.reasons
    except Exception as exc:
        if is_fatal(exc):
            status = STOPPED
            job.stop_reasons = [f"engine memory exhaustion: {exc}"]
            error = exc
        else:
            status = "failed"
            error = exc
    finally:
        session.close()
        survey = job.survey()
        coverage = job.coverage(survey)
        write_json(project.directory / COVERAGE_FILE, coverage)
        if status == "finished":
            status = "completed" if coverage["complete"] else "incomplete"
        diagnostics = job.diagnostics(status, job.stop_reasons, error) if status == STOPPED else None
        state = job.write_state(status)
    if status == "failed" and error is not None:
        raise error
    exit_codes = {"completed": 0, "incomplete": 1, STOPPED: 3, "interrupted": 130}
    return JobResult(status=status, exit_code=exit_codes[status], state=state, diagnostics=diagnostics)
