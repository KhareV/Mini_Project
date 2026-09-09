"""Timestamp-based alignment primitives for future multimodal experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SynchronizedSignals:
    timestamps: np.ndarray
    values: dict[str, np.ndarray]
    presence: dict[str, np.ndarray]
    sampling_rate: float


def synchronize_modalities(modalities: dict[str, tuple[np.ndarray, np.ndarray]],
                           target_fs: float) -> SynchronizedSignals:
    """Interpolate timestamped 1-D modalities onto one union time grid.

    Values outside a modality's observed interval remain NaN and are marked
    absent in the presence mask; no extrapolation is silently invented.
    """
    if target_fs <= 0 or not modalities:
        raise ValueError("target_fs and modalities are required")
    validated = {}
    for name, (timestamps, samples) in modalities.items():
        t = np.asarray(timestamps, dtype=float).reshape(-1)
        x = np.asarray(samples, dtype=float).reshape(-1)
        if len(t) != len(x) or len(t) < 2 or not np.isfinite(t).all() or not np.isfinite(x).all():
            raise ValueError(f"invalid samples for modality {name}")
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"timestamps for modality {name} must be strictly increasing")
        validated[name] = (t, x)
    start = min(t[0] for t, _ in validated.values())
    end = max(t[-1] for t, _ in validated.values())
    step = 1.0 / target_fs
    grid = np.arange(start, end + step / 2, step)
    values, presence = {}, {}
    for name, (t, x) in validated.items():
        mask = (grid >= t[0]) & (grid <= t[-1])
        aligned = np.full(grid.shape, np.nan, dtype=float)
        aligned[mask] = np.interp(grid[mask], t, x)
        values[name] = aligned
        presence[name] = mask
    return SynchronizedSignals(grid, values, presence, float(target_fs))
