"""Validation-only decision threshold selection for binary classifiers."""

from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared probability error for a binary outcome."""
    y_true, y_prob = _validate(y_true, y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Return non-empty equal-width reliability bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    y_true, y_prob = _validate(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for index in range(n_bins):
        mask = ((y_prob >= edges[index]) & (y_prob < edges[index + 1]))
        if index == n_bins - 1:
            mask |= y_prob == 1.0
        if not mask.any():
            continue
        bins.append({
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": int(mask.sum()),
            "mean_probability": float(np.mean(y_prob[mask])),
            "observed_frequency": float(np.mean(y_true[mask])),
        })
    return bins


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    """Sample-weighted absolute calibration gap across reliability bins."""
    y_true, y_prob = _validate(y_true, y_prob)
    bins = reliability_bins(y_true, y_prob, n_bins=n_bins)
    return float(sum((item["count"] / len(y_true)) * abs(
        item["mean_probability"] - item["observed_frequency"]
    ) for item in bins))


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(y_prob):
        raise ValueError("labels and probabilities must be non-empty and equal length")
    if not np.isfinite(y_true).all() or not np.isfinite(y_prob).all():
        raise ValueError("labels and probabilities must be finite")
    if not np.isin(y_true, [0.0, 1.0]).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("binary labels and probabilities in [0, 1] are required")
    return y_true, y_prob


def select_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Select the F1-maximising threshold using validation data only.

    Candidate cut points are 0, 1, and every unique finite validation score.
    Ties prefer the threshold closest to 0.5 for a stable, unsurprising model
    default. The returned threshold must never be selected from test scores.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must be equal-length vectors")
    if len(y_true) == 0 or not np.isfinite(y_prob).all():
        raise ValueError("validation labels/scores must be non-empty and finite")
    candidates = np.unique(np.concatenate(([0.0, 1.0], np.clip(y_prob, 0.0, 1.0))))
    best = (-1.0, 0.5, 0)
    for threshold in candidates:
        pred = (y_prob >= threshold).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        f1 = 0.0 if 2 * tp + fp + fn == 0 else (2 * tp) / (2 * tp + fp + fn)
        candidate = (f1, -abs(float(threshold) - 0.5), -int(threshold * 1_000_000))
        if candidate > (best[0], -abs(best[1] - 0.5), -int(best[1] * 1_000_000)):
            best = (f1, float(threshold), 0)
    return best[1], best[0]
