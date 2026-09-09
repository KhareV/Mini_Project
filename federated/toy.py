"""Tiny deterministic FedAvg convergence experiment for contract validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyFedAvgResult:
    global_values: tuple[float, ...]
    target: float


def run_toy_fedavg(client_targets: list[float], client_samples: list[int],
                   rounds: int = 10, local_steps: int = 5, lr: float = 0.2,
                   initial: float = 0.0) -> ToyFedAvgResult:
    if not client_targets or len(client_targets) != len(client_samples):
        raise ValueError("targets and sample counts must be non-empty and equal")
    if rounds < 1 or local_steps < 1 or lr <= 0 or any(n <= 0 for n in client_samples):
        raise ValueError("invalid toy FedAvg configuration")
    global_value = float(initial)
    history = [global_value]
    weights = np.asarray(client_samples, dtype=float)
    weights /= weights.sum()
    for _ in range(rounds):
        local_values = []
        for target in client_targets:
            value = global_value
            for _ in range(local_steps):
                value += lr * (float(target) - value)
            local_values.append(value)
        global_value = float(np.dot(weights, local_values))
        history.append(global_value)
    target = float(np.dot(np.asarray(client_targets, dtype=float), weights))
    return ToyFedAvgResult(tuple(history), target)
