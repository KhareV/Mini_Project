"""Deterministic FedAvg aggregation primitives."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def weighted_average_state_dicts(updates: list[tuple[Mapping[str, torch.Tensor], int]]) -> dict:
    """Average model tensors weighted by client sample counts."""
    if not updates or any(count <= 0 for _, count in updates):
        raise ValueError("at least one update with positive sample count is required")
    keys = set(updates[0][0])
    if any(set(state) != keys for state, _ in updates):
        raise ValueError("client state dictionaries have different keys")
    total = sum(count for _, count in updates)
    averaged = {}
    for key in keys:
        reference = updates[0][0][key]
        if not torch.is_floating_point(reference):
            averaged[key] = reference.clone()
            continue
        value = torch.zeros_like(reference)
        for state, count in updates:
            value += state[key] * (count / total)
        averaged[key] = value
    return averaged
