"""Inference bridge for the locked P1+P2 research model.

This service accepts already standardized 10-second BIDMC-compatible windows;
raw hardware ingestion is deliberately not part of this pre-hardware gate.
"""
from pathlib import Path
import sys
import json
import math
import time
from typing import Dict, List

import torch


ROOT = Path(__file__).resolve().parents[2]
P2_ROOT = ROOT / "continuous-health-monitor"
CHECKPOINT = P2_ROOT / "experiments/results/multimodal_p1_integrated/checkpoint.pt"
P1_CHECKPOINT = ROOT / "models/MODEL_V1.pt"
CALIBRATION = P2_ROOT / "experiments/results/multimodal_p1_integrated/calibration.json"


class MultimodalInferenceService:
    def __init__(self):
        self.model = None

    def status(self) -> Dict[str, object]:
        return {
            "ready": CHECKPOINT.exists() and P1_CHECKPOINT.exists() and CALIBRATION.exists(),
            "model_version": "MODEL_V1 + multimodal_p1_integrated",
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "input_contract": "canonical Person 1 ECG (2500 samples at 250 Hz) plus preprocessed PPG (1250 samples at 125 Hz)",
        }

    def _load(self):
        if self.model is not None:
            return
        if not CHECKPOINT.exists() or not P1_CHECKPOINT.exists():
            raise FileNotFoundError("locked P1 and multimodal checkpoints are required")
        p2_path = str(P2_ROOT)
        if p2_path not in sys.path:
            sys.path.insert(0, p2_path)
        from ml.models.p1_ecg_adapter import Person1ECGEncoderAdapter
        from ml.models.ppg_encoder import PPGEncoder
        from ml.models.fusion_model import MultimodalFusionModel

        checkpoint = torch.load(CHECKPOINT, map_location="cpu")
        config = checkpoint["model_config"]
        ecg = Person1ECGEncoderAdapter.from_checkpoint(str(P1_CHECKPOINT), embedding_dim=config["ecg_embedding_dim"])
        ppg = PPGEncoder(in_channels=config["ppg_input_channels"], embedding_dim=config["ppg_embedding_dim"])
        self.model = MultimodalFusionModel(
            ecg, ppg, tabular_dim=config["tabular_dim"], hidden_dims=config["hidden_dims"],
            num_classes=config["num_classes"], quality_aware=checkpoint["quality_aware"],
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        with open(CALIBRATION) as handle:
            self.calibration = json.load(handle)

    @torch.no_grad()
    def predict(self, ecg: List[float], ppg: List[float], hr: float, spo2: float,
                presence_mask: List[float], quality_scores: List[float]) -> Dict[str, object]:
        started = time.perf_counter()
        self._load()
        if len(ecg) != 2500 or len(ppg) != 1250:
            raise ValueError("ecg must contain 2500 canonical samples and ppg must contain 1250 preprocessed samples")
        if len(presence_mask) != 4 or len(quality_scores) != 4:
            raise ValueError("presence_mask and quality_scores must each contain four values")
        available_quality = [quality for present, quality in zip(presence_mask, quality_scores) if present]
        quality = min(available_quality) if available_quality else 0.0
        metadata = {"model_version": "MODEL_V1 + multimodal_p1_integrated", "preprocessing_version": self.calibration["preprocessing_version"], "quality": quality, "available_modalities": [name for name, present in zip(["ecg", "ppg", "hr", "spo2"], presence_mask) if present], "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        if not presence_mask[0] or quality < 0.5:
            return {"prediction": "UNRELIABLE_SIGNAL", "model_confidence": None, "reason": "ECG unavailable or an available modality has insufficient quality", **metadata, "medical_disclaimer": "Research output only; not a diagnosis."}
        tabular = torch.tensor([[(hr - 80.0) / 20.0, (spo2 - 95.0) / 5.0]], dtype=torch.float32)
        logits = self.model(
            torch.tensor(ecg, dtype=torch.float32).view(1, 1, -1),
            torch.tensor(ppg, dtype=torch.float32).view(1, 1, -1), tabular,
            torch.tensor(presence_mask, dtype=torch.float32).view(1, 4),
            torch.tensor(quality_scores, dtype=torch.float32).view(1, 4),
        )
        raw_probability = float(torch.softmax(logits, dim=-1)[0, 1])
        coefficient = self.calibration["coefficients"][0]
        calibrated = 1.0 / (1.0 + math.exp(-(coefficient * raw_probability + self.calibration["intercept"][0])))
        metadata["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return {"prediction": "POTENTIALLY_ABNORMAL_PATTERN" if calibrated >= self.calibration["locked_threshold"] else "NORMAL_MONITORED_PATTERN", "model_confidence": calibrated, "raw_model_score": raw_probability, "threshold": self.calibration["locked_threshold"], **metadata, "medical_disclaimer": "Research output only; not a diagnosis."}


multimodal_inference_service = MultimodalInferenceService()
