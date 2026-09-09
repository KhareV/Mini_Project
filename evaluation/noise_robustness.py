"""Deterministic SNR corruption/evaluation helpers for NSTDB experiments."""

from __future__ import annotations

import numpy as np


def add_awgn_at_snr(signal: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Add seeded white Gaussian noise at a requested signal-to-noise ratio."""
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("signal must be non-empty and finite")
    rng = np.random.RandomState(seed)
    signal_power = float(np.mean(values.astype(float) ** 2))
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    return values + rng.normal(0.0, np.sqrt(noise_power), len(values)).astype(np.float32)


def evaluate_snr_curve(model, signals: list[np.ndarray], labels: np.ndarray,
                       snr_levels: tuple[float, ...] = (24, 18, 12, 6, 0, -6),
                       source_fs: int = 250, seed: int = 42) -> list[dict]:
    """Return locked-model probabilities for each SNR without model fitting."""
    labels = np.asarray(labels, dtype=int)
    if len(signals) != len(labels):
        raise ValueError("signals and labels must have equal length")
    results = []
    for snr in snr_levels:
        probabilities = [
            model.predict(add_awgn_at_snr(signal, snr, seed + index), source_fs).probability_abnormal
            for index, signal in enumerate(signals)
        ]
        results.append({"snr_db": float(snr), "n_samples": len(probabilities),
                        "labels": labels.tolist(), "probabilities": probabilities})
    return results
