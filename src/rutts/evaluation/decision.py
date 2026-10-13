from __future__ import annotations

from pathlib import Path
from typing import Any

from rutts.atomic import atomic_write_text
from rutts.evaluation.common import CATEGORIES, DECISION_JSON, DECISION_MD, SCHEMA_VERSION, UNAVAILABLE, write_json
from rutts.evaluation.report import build_report

NATURALNESS_MIN = 4.0
COMFORT_MIN = 4.0
CATEGORY_FLOOR = 3.0
SECOND_ROUND = (
    "Evaluate a second-round candidate (OmniVoice or F5-TTS Russian Base v2) on the same suite "
    "before choosing a default."
)


def _memory_recorded(trial: dict[str, Any]) -> bool:
    return any(bool(value != UNAVAILABLE) for value in trial["memory"].values())


def _feasible(candidate: dict[str, Any]) -> bool:
    return bool(candidate["scored_samples"] > 0 and candidate["resources"]["peak_rss_bytes"] != UNAVAILABLE)


def _quality_failures(candidate: dict[str, Any]) -> list[str]:
    means = candidate["means"]
    failures: list[str] = []
    if means["naturalness"] is None or means["naturalness"] < NATURALNESS_MIN:
        failures.append(f"naturalness mean {means['naturalness']} is below {NATURALNESS_MIN:g}")
    if means["comfort"] is None or means["comfort"] < COMFORT_MIN:
        failures.append(f"comfort mean {means['comfort']} is below {COMFORT_MIN:g}")
    for category in CATEGORIES:
        if means[category] is None or means[category] < CATEGORY_FLOOR:
            failures.append(f"{category} mean {means[category]} is below {CATEGORY_FLOOR:g}")
    return failures


def _wins(candidate: dict[str, Any]) -> int:
    return sum(int(row["wins"]) for row in candidate["pairwise"])


def _rank_key(candidate: dict[str, Any]) -> tuple[int, float, float]:
    overall = candidate["overall_mean"] if candidate["overall_mean"] is not None else 0.0
    render = candidate["resources"]["mean_render_seconds_per_ok_attempt"]
    render_seconds = float(render) if render != UNAVAILABLE else float("inf")
    return (-_wins(candidate), -float(overall), render_seconds)


def _trial_gap(candidate: dict[str, Any], trials: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """Return the completed trial for a candidate, or the reason none qualifies."""
    mine = [trial for trial in trials if trial["engine_fingerprint"] == candidate["engine_fingerprint"]]
    if not mine:
        return None, f"no chapter trial exists for candidate {candidate['candidate']}"
    for trial in mine:
        if trial["completed"] is True and _memory_recorded(trial):
            return trial, ""
    trial = mine[0]
    if trial["failed_chunks"]:
        reason = f"{len(trial['failed_chunks'])} failed chunk(s)"
    elif trial["completed"] is not True:
        reason = "the chapter did not assemble completely"
    else:
        reason = "no memory observation was recorded (the rendered manifest has no resource measurements)"
    return None, f"chapter trial '{trial['label']}' for candidate {candidate['candidate']} is not usable: {reason}"


def _refuse(eval_dir: Path, gaps: list[str], *, ranking: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "no_decision",
        "gaps": gaps,
        "next_step": SECOND_ROUND,
        "ranking": ranking or [],
    }
    write_json(eval_dir / DECISION_JSON, document)
    stale = eval_dir / DECISION_MD
    stale.unlink(missing_ok=True)
    return document


def decide(eval_dir: Path, *, target_revision: str | None = None) -> dict[str, Any]:
    report = build_report(eval_dir)
    candidates: list[dict[str, Any]] = report["candidates"]
    if report["status"] == "incomplete":
        unscored = report["unscored"]
        return _refuse(
            eval_dir,
            [
                f"scores are missing: {len(unscored['samples'])} sample score set(s) and "
                f"{len(unscored['preferences'])} preference(s) are unscored; finish scoring first"
            ],
        )

    gaps: list[str] = []
    feasible = [candidate for candidate in candidates if _feasible(candidate)]
    ranked = sorted(feasible, key=_rank_key)
    ranking = [
        {"candidate": item["candidate"], "wins": _wins(item), "overall_mean": item["overall_mean"]} for item in ranked
    ]
    if len(feasible) < 2:
        gaps.append(f"fewer than two candidates are feasible ({len(feasible)} of {len(candidates)})")
        return _refuse(eval_dir, gaps, ranking=ranking)

    top = ranked[:2]
    trials: list[dict[str, Any]] = report["chapter_trials"]
    usable: dict[int, dict[str, Any]] = {}
    for candidate in top:
        trial, reason = _trial_gap(candidate, trials)
        if trial is None:
            gaps.append(reason)
        else:
            usable[candidate["candidate"]] = trial
    if len(usable) == 2:
        first, second = (usable[item["candidate"]] for item in top)
        if (first["chapter_id"], first["target"]) != (second["chapter_id"], second["target"]):
            gaps.append("the chapter trials for the top two candidates are for different chapters or targets")

    passing = [
        candidate
        for candidate in top
        if not _quality_failures(candidate) or target_revision is not None
    ]
    if not passing:
        failing = "; ".join(f"candidate {item['candidate']}: {', '.join(_quality_failures(item))}" for item in top)
        gaps.append(f"none passes the quality gates ({failing})")
    if gaps:
        return _refuse(eval_dir, gaps, ranking=ranking)

    chosen = passing[0]
    trial = usable[chosen["candidate"]]
    failures = _quality_failures(chosen)
    config = chosen["engine"]
    decision: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "decided",
        "chosen": {
            "candidate": chosen["candidate"],
            "engine_fingerprint": chosen["engine_fingerprint"],
            "engine_id": config["engine_id"],
            "model_repo": config["model_repo"],
            "model_revision": config["model_revision"],
            "precision": config["precision"],
            "runtime_name": config["runtime_name"],
            "runtime_version": config["runtime_version"],
            "license": config["license"],
            "settings": config["settings"],
            "narrator": chosen["narrator"],
        },
        "evidence": {
            "seed": report["seed"],
            "attempt_mode": report["attempt_mode"],
            "excerpts_compared": report["excerpts_compared"],
            "samples_scored": chosen["scored_samples"],
            "means": chosen["means"],
            "minimums": chosen["minimums"],
            "overall_mean": chosen["overall_mean"],
            "pairwise": chosen["pairwise"],
            "failure_rates": chosen["failure_rates"],
            "resources": chosen["resources"],
            "ranking": ranking,
            "chapter_trial": {key: trial[key] for key in trial if key != "manual_notes"},
            "manual_notes": trial["manual_notes"],
        },
        "gates": {
            "naturalness_mean_at_least": NATURALNESS_MIN,
            "comfort_mean_at_least": COMFORT_MIN,
            "category_mean_at_least": CATEGORY_FLOOR,
            "failures": failures,
        },
        "target_revision": None,
        "remaining_limitations": _limitations(report, chosen, trial, failures, target_revision),
        "gaps": [],
    }
    if target_revision is not None:
        decision["target_revision"] = {"reason": target_revision, "waived_gates": failures}
    write_json(eval_dir / DECISION_JSON, decision)
    atomic_write_text(eval_dir / DECISION_MD, _markdown(decision))
    return decision


def _limitations(
    report: dict[str, Any],
    chosen: dict[str, Any],
    trial: dict[str, Any],
    failures: list[str],
    target_revision: str | None,
) -> list[str]:
    limits = [
        f"The comparison covers {report['excerpts_compared']} excerpt(s) from the suite; "
        "it is not a full-book listening test.",
        f"The chapter evidence covers one {trial['target']} chapter ({trial['chapter_id']}); "
        "other chapters, dialogue-heavy text and long-run voice drift are not measured.",
        f"The voice is one reference recording ({chosen['narrator']['name']}); "
        "other narrators were not compared unless listed elsewhere.",
    ]
    if report["unavailable"]:
        limits.append(
            f"{len(report['unavailable'])} excerpt(s) had no usable attempt and count as failures "
            f"under the {report['attempt_mode']} attempt rule."
        )
    notes = trial["manual_notes"]
    if notes is None:
        limits.append("Manual listening notes for the chapter trial were not recorded; stress, omission and join checks are unevidenced.")
    else:
        limits.append(
            f"Manual chapter review: {notes['stress_errors']} stress error(s), {notes['omissions']} omission(s), "
            f"{notes['distracting_joins']} distracting join(s). {notes['notes']}".strip()
        )
    memory = trial["memory"]
    limits.append(
        f"Chapter memory observations: peak RSS {memory['peak_rss_bytes']} bytes, MLX peak "
        f"{memory['mlx_peak_memory_bytes']} bytes, swap {memory['swap_used_bytes']} bytes, "
        f"free memory {memory['memory_free_percent']}%."
    )
    if target_revision is not None:
        waived = ", ".join(failures) if failures else "none were needed"
        limits.append(f"Quality targets were revised by approval; waived gate failures: {waived}.")
    return limits


def _markdown(decision: dict[str, Any]) -> str:
    chosen = decision["chosen"]
    evidence = decision["evidence"]
    lines = [
        "# Model and narrator decision",
        "",
        "## Chosen configuration",
        "",
        f"- Model repository: {chosen['model_repo']}",
        f"- Model revision: {chosen['model_revision']}",
        f"- Precision: {chosen['precision']}",
        f"- Engine: {chosen['engine_id']} ({chosen['runtime_name']} {chosen['runtime_version']})",
        f"- License of the artifact: {chosen['license']}",
        f"- Settings: {chosen['settings']}",
        f"- Engine fingerprint: {chosen['engine_fingerprint']}",
        "",
        "## Narrator",
        "",
        f"- Name: {chosen['narrator']['name']}",
        f"- Provenance: {chosen['narrator']['provenance']}",
        f"- Reference sha256: {chosen['narrator']['reference_sha256']}",
        "",
        "## Evidence",
        "",
        f"- Samples scored: {evidence['samples_scored']} of {evidence['excerpts_compared']} excerpt(s) compared",
        f"- Means: {evidence['means']}",
        f"- Minimums: {evidence['minimums']}",
        f"- Failure rates: {evidence['failure_rates']}",
        f"- Resources: {evidence['resources']}",
        f"- Chapter trial: {evidence['chapter_trial']}",
        f"- Manual chapter notes: {evidence['manual_notes']}",
        "",
        "## Gates",
        "",
        f"- Naturalness and comfort means at least {NATURALNESS_MIN:g}; every category mean at least "
        f"{CATEGORY_FLOOR:g}",
        f"- Failures: {', '.join(decision['gates']['failures']) or 'none'}",
    ]
    if decision["target_revision"] is not None:
        revision = decision["target_revision"]
        lines += ["", "## Target revision", "", f"Approved reason (verbatim): {revision['reason']}"]
        lines.append(f"Waived gates: {', '.join(revision['waived_gates']) or 'none'}")
    lines += ["", "## Remaining limitations", ""]
    lines += [f"- {item}" for item in decision["remaining_limitations"]]
    return "\n".join(lines) + "\n"
