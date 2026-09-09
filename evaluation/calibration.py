"""Validation-only decision threshold selection for binary classifiers."""

from __future__ import annotations

import numpy as np


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
