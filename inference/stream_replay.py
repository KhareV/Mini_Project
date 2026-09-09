"""Deterministic fixed-window streaming adapter for offline/device replay."""

from __future__ import annotations

from typing import List

import numpy as np

from .ecg_model_v1 import ECGModelV1, ECGPrediction


class ECGStreamReplayer:
    """Buffer samples and emit MODEL_V1 predictions at a fixed stride."""

    def __init__(self, model: ECGModelV1, source_fs: int,
                 window_seconds: float = 10.0, stride_seconds: float = 5.0):
        if source_fs <= 0 or window_seconds <= 0 or stride_seconds <= 0:
            raise ValueError("sampling rate and window durations must be positive")
        if stride_seconds > window_seconds:
            raise ValueError("stride cannot exceed window duration")
        self.model = model
        self.source_fs = int(source_fs)
        self.window_samples = int(round(source_fs * window_seconds))
        self.stride_samples = int(round(source_fs * stride_seconds))
        self._buffer = np.empty(0, dtype=np.float32)
        self.windows_emitted = 0

    def ingest(self, samples: np.ndarray) -> List[ECGPrediction]:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return []
        if not np.isfinite(values).all():
            raise ValueError("stream samples must be finite")
        self._buffer = np.concatenate((self._buffer, values))
        predictions: List[ECGPrediction] = []
        while len(self._buffer) >= self.window_samples:
            window = self._buffer[:self.window_samples].copy()
            predictions.append(self.model.predict(window, self.source_fs))
            self._buffer = self._buffer[self.stride_samples:]
            self.windows_emitted += 1
        return predictions

    @property
    def buffered_samples(self) -> int:
        return int(self._buffer.size)

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)
        self.windows_emitted = 0
