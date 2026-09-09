"""Small, dependency-light provenance manifest for reportable ML runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root,
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def build_run_manifest(
    *,
    experiment_group: str,
    dataset: str,
    seed: int,
    split_counts: dict[str, int],
    preprocessing: dict[str, Any],
    normalization_stats: dict[str, Any],
    synthetic: bool,
    project_root: Path,
    dataset_details: dict[str, Any],
) -> dict[str, Any]:
    """Return a JSON-serializable manifest without raw signals or labels."""
    return {
        "experiment_group": experiment_group,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(project_root),
        "dataset": dataset,
        "synthetic": synthetic,
        "research_claim_boundary": (
            "Synthetic run: validates software integration only; not clinical or "
            "dataset-performance evidence."
            if synthetic else "Public-dataset run: see dataset and split artifacts."
        ),
        "seed": seed,
        "split_counts": split_counts,
        "dataset_details": dataset_details,
        "preprocessing": preprocessing,
        "preprocessing_sha256": canonical_json_hash(preprocessing),
        "normalization_stats": normalization_stats,
        "normalization_sha256": canonical_json_hash(normalization_stats),
        "test_used_for_tuning": False,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
