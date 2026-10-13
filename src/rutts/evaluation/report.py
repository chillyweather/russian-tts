from __future__ import annotations

from pathlib import Path
from typing import Any

from rutts.atomic import atomic_write_text
from rutts.evaluation.common import (
    CATEGORIES,
    REPORT_JSON,
    REPORT_MD,
    RESULTS_FILE,
    SCHEMA_VERSION,
    TRIALS_DIR,
    NOTES_DIR,
    UNAVAILABLE,
    load_blind,
    load_key,
    load_scores,
    pair_key,
    read_json,
    write_json,
)

RESOURCE_KEYS = ("peak_rss_bytes", "mlx_peak_memory_bytes", "swap_used_bytes", "memory_free_percent")


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _rate(failed: int, total: int) -> float | None:
    return round(failed / total, 4) if total else None


def _resources(results: dict[str, Any]) -> dict[str, Any]:
    observations = [
        attempt["resources"]
        for item in results["excerpts"]
        for attempt in item["attempts"]
        if isinstance(attempt.get("resources"), dict)
    ]
    summary: dict[str, Any] = {}
    for key in RESOURCE_KEYS:
        values = [item[key] for item in observations if isinstance(item.get(key), int)]
        if not values:
            summary[key] = UNAVAILABLE
        elif key == "memory_free_percent":
            summary[key] = min(values)
        else:
            summary[key] = max(values)
    renders = [
        attempt["render_seconds"]
        for item in results["excerpts"]
        for attempt in item["attempts"]
        if attempt["status"] == "ok"
    ]
    summary["mean_render_seconds_per_ok_attempt"] = _mean(renders) if renders else UNAVAILABLE
    return summary


def _failure_rates(results: dict[str, Any]) -> dict[str, Any]:
    excerpts = results["excerpts"]
    first_failed = sum(1 for item in excerpts if item["attempts"][0]["status"] == "failed")
    after_failed = sum(1 for item in excerpts if item["status"] == "failed")
    total = len(excerpts)
    return {
        "first_attempt": {"failed": first_failed, "excerpts": total, "rate": _rate(first_failed, total)},
        "after_retry_budget": {"failed": after_failed, "excerpts": total, "rate": _rate(after_failed, total)},
    }


def _trials(eval_dir: Path) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for path in sorted((eval_dir / TRIALS_DIR).glob("*.json")):
        trial = read_json(path)
        notes_path = eval_dir / NOTES_DIR / f"{trial['label']}.json"
        trial["manual_notes"] = read_json(notes_path) if notes_path.is_file() else None
        trials.append(trial)
    return trials


def build_report(eval_dir: Path) -> dict[str, Any]:
    blind = load_blind(eval_dir)
    key = load_key(eval_dir)
    scores = load_scores(eval_dir)
    sample_candidate = {(item["excerpt_id"], item["label"]): item["candidate"] for item in key["samples"]}
    sample_scores: dict[tuple[str, str], dict[str, Any]] = {
        (excerpt_id, label): record
        for excerpt_id, labelled in scores.get("samples", {}).items()
        for label, record in labelled.items()
    }

    unscored_samples: list[dict[str, str]] = []
    scored_samples = 0
    for excerpt in blind["excerpts"]:
        for label in excerpt["labels"]:
            record = sample_scores.get((excerpt["id"], label))
            if record is None or any(category not in record for category in CATEGORIES):
                unscored_samples.append({"excerpt_id": excerpt["id"], "label": label})
            else:
                scored_samples += 1

    preference_of: dict[tuple[str, str], str] = {}
    unscored_pairs: list[dict[str, Any]] = []
    for pair in blind["pairs"]:
        excerpt_id, labels = pair["excerpt_id"], pair["labels"]
        value = scores.get("preferences", {}).get(excerpt_id, {}).get(pair_key(*labels))
        if value is None:
            unscored_pairs.append({"excerpt_id": excerpt_id, "labels": labels})
        else:
            preference_of[(excerpt_id, pair_key(*labels))] = value

    candidates: list[dict[str, Any]] = []
    trials = _trials(eval_dir)
    for entry in key["candidates"]:
        index = entry["candidate"]
        results_path = Path(entry["directory"]) / RESULTS_FILE
        results = read_json(results_path)
        mine = [
            sample_scores[(excerpt_id, label)]
            for (excerpt_id, label), owner in sample_candidate.items()
            if owner == index and (excerpt_id, label) in sample_scores
        ]
        means = {
            category: _mean([float(record[category]) for record in mine if category in record])
            for category in CATEGORIES
        }
        minimums = {
            category: min([record[category] for record in mine if category in record], default=None)
            for category in CATEGORIES
        }
        category_means = [value for value in means.values() if value is not None]
        candidates.append(
            {
                "candidate": index,
                "directory": entry["directory"],
                "engine_fingerprint": entry["engine_fingerprint"],
                "engine": entry["engine"],
                "narrator": {
                    "name": entry["narrator"]["name"],
                    "provenance": entry["narrator"]["provenance"],
                    "reference_sha256": entry["narrator"]["reference_sha256"],
                },
                "scored_samples": len(mine),
                "means": means,
                "minimums": minimums,
                "overall_mean": _mean(category_means) if len(category_means) == len(CATEGORIES) else None,
                "pairwise": _pairwise(index, sample_candidate, preference_of),
                "failure_rates": _failure_rates(results),
                "retry_budget": results["retry_budget"],
                "unavailable_excerpts": sum(
                    1 for item in key["unavailable"] if any(c["candidate"] == index for c in item["candidates"])
                ),
                "resources": _resources(results),
                "chapter_trials": [trial["label"] for trial in trials if trial["engine_fingerprint"] == entry["engine_fingerprint"]],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if not unscored_samples and not unscored_pairs else "incomplete",
        "seed": blind["seed"],
        "attempt_mode": blind["attempt_mode"],
        "excerpts_compared": len(blind["excerpts"]),
        "samples_total": len(key["samples"]),
        "samples_scored": scored_samples,
        "unscored": {"samples": unscored_samples, "preferences": unscored_pairs},
        "candidates": candidates,
        "unavailable": key["unavailable"],
        "chapter_trials": trials,
    }


def _pairwise(index: int, owner: dict[tuple[str, str], int], preference_of: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, int]] = {}
    for (excerpt_id, pair), value in preference_of.items():
        first_label, second_label = pair.split("/")
        first = owner.get((excerpt_id, first_label))
        second = owner.get((excerpt_id, second_label))
        if first is None or second is None or index not in (first, second):
            continue
        opponent = second if first == index else first
        row = rows.setdefault(opponent, {"wins": 0, "losses": 0, "ties": 0})
        if value == "=":
            row["ties"] += 1
        elif owner.get((excerpt_id, value)) == index:
            row["wins"] += 1
        else:
            row["losses"] += 1
    return [{"opponent": opponent, **row} for opponent, row in sorted(rows.items())]


def write_report(eval_dir: Path) -> dict[str, Any]:
    report = build_report(eval_dir)
    write_json(eval_dir / REPORT_JSON, report)
    atomic_write_text(eval_dir / REPORT_MD, render_markdown(report))
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Blind listening report",
        "",
        f"- Status: **{report['status']}**",
        f"- Excerpts compared: {report['excerpts_compared']}; samples scored "
        f"{report['samples_scored']} of {report['samples_total']}",
        f"- Attempt rule: {report['attempt_mode']} (seed {report['seed']})",
    ]
    unscored = report["unscored"]
    if report["status"] == "incomplete":
        lines += ["", "## Unscored", ""]
        lines += [f"- sample {item['excerpt_id']} {item['label']}" for item in unscored["samples"]]
        lines += [f"- preference {item['excerpt_id']} {'/'.join(item['labels'])}" for item in unscored["preferences"]]
    lines += ["", "## Candidates", "", "| # | Scored | " + " | ".join(f"{c} mean / min" for c in CATEGORIES) + " | Overall |"]
    lines.append("| --- " * (len(CATEGORIES) + 3) + "|")
    for candidate in report["candidates"]:
        cells = [
            f"{_fmt(candidate['means'][c])} / {_fmt(candidate['minimums'][c])}" for c in CATEGORIES
        ]
        lines.append(
            f"| {candidate['candidate']} | {candidate['scored_samples']} | "
            + " | ".join(cells)
            + f" | {_fmt(candidate['overall_mean'])} |"
        )
    lines += ["", "## Failures (first attempt vs after retry budget)", ""]
    for candidate in report["candidates"]:
        first = candidate["failure_rates"]["first_attempt"]
        after = candidate["failure_rates"]["after_retry_budget"]
        lines.append(
            f"- Candidate {candidate['candidate']}: first attempt {first['failed']}/{first['excerpts']} "
            f"({_fmt(first['rate'])}); after retry budget of {candidate['retry_budget']} "
            f"{after['failed']}/{after['excerpts']} ({_fmt(after['rate'])}); "
            f"{candidate['unavailable_excerpts']} excerpt(s) unavailable for comparison"
        )
    lines += ["", "## Pairwise preferences", ""]
    for candidate in report["candidates"]:
        for row in candidate["pairwise"]:
            lines.append(
                f"- Candidate {candidate['candidate']} vs {row['opponent']}: "
                f"{row['wins']} win(s), {row['losses']} loss(es), {row['ties']} tie(s)"
            )
    lines += ["", "## Resources (preview renders)", ""]
    for candidate in report["candidates"]:
        resources = candidate["resources"]
        lines.append(
            f"- Candidate {candidate['candidate']}: peak RSS {_fmt(resources['peak_rss_bytes'])} bytes, "
            f"MLX peak {_fmt(resources['mlx_peak_memory_bytes'])} bytes, swap {_fmt(resources['swap_used_bytes'])} "
            f"bytes, free memory {_fmt(resources['memory_free_percent'])}%, mean render "
            f"{_fmt(resources['mean_render_seconds_per_ok_attempt'])} s"
        )
    lines += ["", "## Chapter trials", ""]
    if not report["chapter_trials"]:
        lines.append("- none recorded")
    for trial in report["chapter_trials"]:
        lines.append(
            f"- {trial['label']} ({trial['chapter_id']}, {trial['target']}): completed {trial['completed']}, "
            f"finished audio {_fmt(trial['finished_audio_seconds'])} s, render {_fmt(trial['render_seconds'])} s, "
            f"failed chunks {len(trial['failed_chunks'])}, render hours per audio hour "
            f"{_fmt(trial['render_hours_per_audio_hour'])}, retries per audio hour "
            f"{_fmt(trial['retries_per_audio_hour'])}"
        )
        notes = trial["manual_notes"]
        if notes is None:
            lines.append("  - manual listening notes: not recorded")
        else:
            lines.append(
                f"  - manual listening: stress errors {notes['stress_errors']}, omissions {notes['omissions']}, "
                f"distracting joins {notes['distracting_joins']}; {notes['notes']}"
            )
    return "\n".join(lines) + "\n"

