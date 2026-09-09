"""Quality and plausibility contracts for PPG and SpO2 modalities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PPGQuality:
    score: float
    label: str
    usable: bool
    reasons: tuple[str, ...]


def assess_ppg(raw: np.ndarray, filtered: np.ndarray,
               pulse_intervals_s: np.ndarray | None = None) -> PPGQuality:
    raw = np.asarray(raw, dtype=float).reshape(-1)
    filtered = np.asarray(filtered, dtype=float).reshape(-1)
    reasons: list[str] = []
    if len(raw) == 0 or len(filtered) == 0:
        return PPGQuality(0.0, "POOR", False, ("missing_signal",))
    if not np.isfinite(raw).all() or not np.isfinite(filtered).all():
        reasons.append("non_finite_samples")
    if np.std(filtered) < 1e-8:
        reasons.append("flatline")
    amplitude_score = min(1.0, float(np.std(filtered)) / (np.std(raw) + 1e-8))
    interval_score = 1.0
    if pulse_intervals_s is not None:
        intervals = np.asarray(pulse_intervals_s, dtype=float).reshape(-1)
        if len(intervals) == 0 or not np.isfinite(intervals).all():
            reasons.append("missing_pulse_intervals")
            interval_score = 0.0
        else:
            bpm = 60.0 / intervals
            valid = (bpm >= 30.0) & (bpm <= 220.0)
            interval_score = float(np.mean(valid))
            if interval_score < 0.8:
                reasons.append("implausible_pulse_rate")
    finite_score = 0.0 if "non_finite_samples" in reasons else 1.0
    score = round(100.0 * (0.35 * finite_score + 0.30 * amplitude_score + 0.35 * interval_score), 3)
    usable = bool(not reasons and score >= 60.0)
    label = "EXCELLENT" if score >= 85 else "GOOD" if score >= 70 else "FAIR" if score >= 60 else "POOR"
    return PPGQuality(score, label, usable, tuple(reasons))


@dataclass(frozen=True)
class SpO2Quality:
    value: float | None
    confidence: float
    trend: float
    state: str
    usable: bool


def assess_spo2(series: np.ndarray, sample_interval_s: float = 1.0) -> SpO2Quality:
    values = np.asarray(series, dtype=float).reshape(-1)
    if values.size == 0:
        return SpO2Quality(None, 0.0, 0.0, "UNRELIABLE", False)
    valid = np.isfinite(values) & (values >= 70.0) & (values <= 100.0)
    if not valid.any():
        return SpO2Quality(None, 0.0, 0.0, "UNRELIABLE", False)
    clean = values[valid]
    confidence = float(np.mean(valid))
    trend = 0.0
    if len(clean) > 1:
        trend = float((clean[-1] - clean[0]) / ((len(clean) - 1) * sample_interval_s))
    value = float(np.median(clean))
    usable = confidence >= 0.8
    return SpO2Quality(value, confidence, trend, "GOOD" if usable else "UNRELIABLE", usable)
