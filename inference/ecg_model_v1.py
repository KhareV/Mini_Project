"""Offline replay wrapper for the frozen centralized ECG model."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from models.ecg_cnn import ECGCNN1D
from preprocessing.ecg import ECGPreprocessor, NormalizationStats


@dataclass(frozen=True)
class ECGPrediction:
    label: str
    probability_abnormal: float
    threshold: float
    model_version: str
    preprocessing_version: str

    def to_dict(self) -> dict:
        return asdict(self)


class ECGModelV1:
    """Load and run MODEL_V1 with its matching normalization artifact."""

    def __init__(self, model, preprocessor: ECGPreprocessor, threshold: float = 0.5,
                 device: str = "cpu"):
        self.model = model.to(device).eval()
        self.preprocessor = preprocessor
        self.threshold = float(threshold)
        self.device = device
        self.model_version = model.config.model_version

    @classmethod
    def load(cls, checkpoint_path: str, normalization_path: str,
             threshold: float = 0.5, device: str = "cpu") -> "ECGModelV1":
        model, checkpoint = ECGCNN1D.load_checkpoint(checkpoint_path, device=device)
        stats = NormalizationStats.load(normalization_path)
        expected = model.config.preprocessing_version
        if stats.preprocessing_version != expected:
            raise ValueError(
                f"Preprocessing mismatch: model={expected}, stats={stats.preprocessing_version}"
            )
        preprocessor = ECGPreprocessor(
            normalization_stats=stats, require_normalization_stats=True
        )
        return cls(model, preprocessor, threshold=threshold, device=device)

    def _window(self, signal: np.ndarray, source_fs: int) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        result = self.preprocessor.process(signal, source_fs, record_id="replay")
        if not result.is_valid:
            raise ValueError(f"Invalid ECG signal: {result.validation_notes}")
        length = int(self.model.config.input_length)
        if len(result.signal) >= length:
            window = result.signal[:length]
        else:
            window = np.pad(result.signal, (0, length - len(result.signal)))
        return window.astype(np.float32, copy=False)

    @torch.no_grad()
    def predict(self, signal: np.ndarray, source_fs: int) -> ECGPrediction:
        window = self._window(signal, source_fs)
        tensor = torch.from_numpy(window).reshape(1, 1, -1).to(self.device)
        probability = float(torch.softmax(self.model(tensor), dim=-1)[0, 1].item())
        return ECGPrediction(
            label="ABNORMAL" if probability >= self.threshold else "NORMAL",
            probability_abnormal=probability,
            threshold=self.threshold,
            model_version=self.model_version,
            preprocessing_version=self.preprocessor.config_dict["preprocessing_version"],
        )

