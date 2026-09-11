"""Hosted inference for the validated standalone ECG MODEL_V1."""
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from models.ecg_cnn import ECGCNN1D

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "models/MODEL_V1.pt"
CALIBRATION = ROOT / "reports/MODEL_V1_validation_calibration.json"


class ECGInferenceService:
    def __init__(self):
        self.model = None

    def status(self) -> Dict[str, object]:
        return {"ready": CHECKPOINT.exists() and CALIBRATION.exists(), "deployment_eligible": True, "clinical_use_eligible": False, "model_version": "MODEL_V1", "input_contract": "canonical float ECG: 2500 samples, 10 seconds at 250 Hz", "checkpoint": str(CHECKPOINT.relative_to(ROOT))}

    def _load(self):
        if self.model is None:
            self.model, _ = ECGCNN1D.load_checkpoint(CHECKPOINT, "cpu")
            self.model.eval()
            self.calibration = json.loads(CALIBRATION.read_text())

    @torch.no_grad()
    def predict(self, ecg: List[float], quality: float = 1.0) -> Dict[str, object]:
        started = time.perf_counter()
        if len(ecg) != 2500:
            raise ValueError("ecg must contain exactly 2500 canonical samples")
        values = np.asarray(ecg, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("ecg contains non-finite values")
        if not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be between 0 and 1")
        metadata = {"model_version": "MODEL_V1", "quality": quality, "preprocessing_version": "1.0.0", "deployment_eligible": True, "clinical_use_eligible": False, "medical_disclaimer": "Research output only; not a diagnosis."}
        if quality < 0.5:
            return {"prediction": "UNRELIABLE_SIGNAL", "model_confidence": None, "reason": "ECG quality below locked 0.50 gate", "latency_ms": round((time.perf_counter()-started)*1000, 2), **metadata}
        self._load()
        raw = float(self.model.predict_proba(torch.tensor(values).view(1, 1, -1))[0, 1])
        raw = min(max(raw, 1e-6), 1-1e-6)
        logit = math.log(raw / (1-raw))
        calibrated = 1 / (1 + math.exp(-(self.calibration["coefficient"] * logit + self.calibration["intercept"])))
        threshold = self.calibration["locked_threshold"]
        return {"prediction": "POTENTIALLY_ABNORMAL_PATTERN" if calibrated >= threshold else "NORMAL_MONITORED_PATTERN", "model_confidence": calibrated, "raw_model_score": raw, "threshold": threshold, "latency_ms": round((time.perf_counter()-started)*1000, 2), **metadata}


ecg_inference_service = ECGInferenceService()
