"""Non-circular PPG pulse and measured-SpO2 inference services."""
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P2_ROOT = ROOT / "continuous-health-monitor"
if str(P2_ROOT) not in sys.path:
    sys.path.insert(0, str(P2_ROOT))

from ml.models.ppg_pulse_estimator import PPGPulseEstimatorV3

from backend.services.ecg_inference import ecg_inference_service


class VitalsInferenceService:
    def __init__(self):
        self.pulse_estimator = PPGPulseEstimatorV3(sampling_rate=125.0, quality_threshold=0.50)

    def status(self) -> Dict[str, object]:
        return {
            "ready": True,
            "model_version": self.pulse_estimator.VERSION,
            "software_release_eligible": True,
            "clinical_use_eligible": False,
            "input_contract": "PPG: 1250 samples/10 s/125 Hz; SpO2: synchronized device measurement with quality",
            "status_note": "Validated for held-out BIDMC pulse estimation. SpO2 is validated sensor input, not inferred from single-channel BIDMC PPG.",
        }

    def predict(self, ppg: List[float], spo2: Optional[float], ppg_quality: float = 1.0,
                spo2_quality: float = 1.0) -> Dict[str, object]:
        started = time.perf_counter()
        if len(ppg) != 1250:
            raise ValueError("ppg must contain exactly 1250 samples (10 seconds at 125 Hz)")
        values = np.asarray(ppg, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("ppg contains non-finite values")
        if not 0.0 <= ppg_quality <= 1.0 or not 0.0 <= spo2_quality <= 1.0:
            raise ValueError("quality values must be between 0 and 1")
        if spo2 is not None and (not np.isfinite(spo2) or not 0.0 <= spo2 <= 100.0):
            raise ValueError("spo2 must be null or a finite percentage between 0 and 100")

        pulse = self.pulse_estimator.estimate(values)
        effective_ppg_quality = min(ppg_quality, pulse.quality_score)
        pulse_reliable = pulse.reliable and ppg_quality >= 0.50
        spo2_reliable = spo2 is not None and 70.0 <= spo2 <= 100.0 and spo2_quality >= 0.50
        alerts = []
        if pulse_reliable and (pulse.pulse_bpm < 60.0 or pulse.pulse_bpm > 100.0):
            alerts.append("PULSE_OUTSIDE_60_100_BPM")
        if spo2_reliable and spo2 < 95.0:
            alerts.append("SPO2_BELOW_95_PERCENT")

        if not pulse_reliable and not spo2_reliable:
            prediction = "UNRELIABLE_SIGNAL"
        elif alerts:
            prediction = "POTENTIALLY_ABNORMAL_VITALS"
        else:
            prediction = "NO_VITAL_ALERT"
        return {
            "prediction": prediction,
            "alerts": alerts,
            "pulse": {**pulse.to_dict(), "external_quality": ppg_quality, "effective_quality": effective_ppg_quality, "used": pulse_reliable},
            "spo2": {"value_percent": spo2, "quality": spo2_quality, "used": spo2_reliable, "source": "synchronized_device_measurement"},
            "model_version": self.pulse_estimator.VERSION,
            "clinical_use_eligible": False,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "medical_disclaimer": "Research output only; not a diagnosis or a certified pulse oximeter.",
        }


class CentralizedSystemInferenceService:
    def predict(self, ecg: List[float], ppg: List[float], spo2: Optional[float], ecg_quality: float = 1.0,
                ppg_quality: float = 1.0, spo2_quality: float = 1.0) -> Dict[str, object]:
        ecg_result = ecg_inference_service.predict(ecg, ecg_quality)
        vitals_result = vitals_inference_service.predict(ppg, spo2, ppg_quality, spo2_quality)
        abnormal = (
            ecg_result["prediction"] == "POTENTIALLY_ABNORMAL_PATTERN"
            or vitals_result["prediction"] == "POTENTIALLY_ABNORMAL_VITALS"
        )
        reliable_count = sum([
            ecg_result["prediction"] != "UNRELIABLE_SIGNAL",
            vitals_result["prediction"] != "UNRELIABLE_SIGNAL",
        ])
        if abnormal:
            decision = "POTENTIALLY_ABNORMAL_PATTERN"
        elif reliable_count == 0:
            decision = "UNRELIABLE_SIGNAL"
        elif reliable_count == 1:
            decision = "NO_ALERT_FROM_RELIABLE_MODALITIES"
        else:
            decision = "NORMAL_MONITORED_PATTERN"
        return {
            "prediction": decision,
            "reliable_components": reliable_count,
            "ecg": ecg_result,
            "vitals": vitals_result,
            "fusion_policy": "quality-gated conservative OR; no circular learned BIDMC classifier",
            "clinical_use_eligible": False,
            "medical_disclaimer": "Research output only; not a diagnosis.",
        }


vitals_inference_service = VitalsInferenceService()
centralized_system_inference_service = CentralizedSystemInferenceService()
