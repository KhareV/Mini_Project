"""Participant-level IID and controlled non-IID client partitioning."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def partition_manifest(manifest: pd.DataFrame, n_clients: int, mode: str = "iid",
                       seed: int = 42) -> dict[str, pd.DataFrame]:
    """Assign complete participants to deterministic client manifests."""
    if n_clients < 1 or mode not in {"iid", "label_skew"}:
        raise ValueError("n_clients must be positive and mode must be iid or label_skew")
    required = {"participant_id", "label_int"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"manifest requires columns {sorted(required)}")
    frame = manifest.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    participants = frame.groupby("participant_id")["label_int"].agg(
        dominant=lambda s: int(round(float(np.mean(s)))),
        count="size",
    ).reset_index()
    rng = np.random.RandomState(seed)
    if mode == "iid":
        order = participants["participant_id"].to_numpy()
        rng.shuffle(order)
    else:
        # Alternate dominant-label groups while retaining deterministic order.
        groups = [participants[participants.dominant == label]["participant_id"].to_numpy()
                  for label in sorted(participants.dominant.unique())]
        for group in groups:
            rng.shuffle(group)
        order = np.array([pid for pair in zip(*map(list, groups)) for pid in pair], dtype=object)
        leftovers = [pid for group in groups for pid in group if pid not in set(order)]
        order = np.concatenate([order, np.asarray(leftovers, dtype=object)])
    assignments = {f"client_{index:03d}": [] for index in range(n_clients)}
    for index, participant in enumerate(order):
        assignments[f"client_{index % n_clients:03d}"].append(str(participant))
    result = {}
    for client, pids in assignments.items():
        rows = frame[frame.participant_id.isin(pids)].copy().reset_index(drop=True)
        rows["client_id"] = client
        result[client] = rows
    assigned = pd.concat(result.values(), ignore_index=True) if result else frame.iloc[:0]
    if set(assigned.participant_id) != set(frame.participant_id):
        raise AssertionError("partition dropped participants")
    if assigned.groupby("participant_id")["client_id"].nunique().max() != 1:
        raise AssertionError("participant leakage across clients")
    return result


def partition_digest(partitions: dict[str, pd.DataFrame]) -> str:
    """Stable digest for a persisted client assignment."""
    lines = []
    for client in sorted(partitions):
        for pid in sorted(partitions[client].participant_id.unique()):
            lines.append(f"{client}:{pid}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()
