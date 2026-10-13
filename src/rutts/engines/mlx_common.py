from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from rutts.engines.base import SynthesisResult

UNRESOLVED_REVISION = "unresolved"
_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")


def resolve_revision(repo: str) -> str:
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo, local_files_only=True)
    commit = snapshot.rstrip("/").split("/")[-1]
    if not _COMMIT_HASH.match(commit):
        raise RuntimeError(f"could not resolve a commit hash for {repo}: {snapshot}")
    return commit


def cached_revision(repo: str) -> str:
    """Revision of the locally cached snapshot, known without loading the model."""
    try:
        return resolve_revision(repo)
    except Exception:
        return UNRESOLVED_REVISION


def concat_segments(segments: Sequence[Any]) -> SynthesisResult:
    if not segments:
        raise RuntimeError("engine returned no audio segments")
    rates = {int(segment.sample_rate) for segment in segments}
    if len(rates) != 1:
        raise RuntimeError(f"engine returned mixed sample rates: {sorted(rates)}")
    samples = np.concatenate(
        [np.asarray(segment.audio, dtype=np.float32).reshape(-1) for segment in segments]
    )
    return SynthesisResult(samples=samples, sample_rate=rates.pop())


def clear_mlx_cache() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        return
    mx.clear_cache()
