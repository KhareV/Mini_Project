"""Replay fixed held-out BIDMC examples through the released vitals service."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.vitals_inference import vitals_inference_service, centralized_system_inference_service


def stable_output(value):
    """Remove runtime-only timings so the evidence file is byte reproducible."""
    if isinstance(value, dict):
        return {key: stable_output(item) for key, item in value.items() if key != "latency_ms"}
    if isinstance(value, list):
        return [stable_output(item) for item in value]
    return value


def main():
    samples = json.loads((ROOT / "continuous-health-monitor/data/processed/bidmc_test_windows.json").read_text())
    examples = []
    for index in (0, 300, 500):
        sample = samples[index]
        output = vitals_inference_service.predict(
            sample["ppg"], sample["spo2_mean"], sample["ppg_quality"], 1.0
        )
        examples.append({
            "index": index,
            "recording_id": sample["recording_id"],
            "subject_id": sample["subject_id"],
            "timestamp_start": sample["timestamp_start"],
            "reference_pulse_bpm": sample["pulse_mean"],
            "reference_spo2_percent": sample["spo2_mean"],
            "service_output": stable_output(output),
        })
    report = {
        "release": "CENTRALIZED_SYSTEM_V2",
        "source": "fixed examples from the patient-held-out BIDMC test partition",
        "endpoint_equivalent": "POST /model/vitals/infer",
        "examples": examples,
        "integrated_example": stable_output(centralized_system_inference_service.predict(
            samples[0]["ecg"], samples[0]["ppg"], samples[0]["spo2_mean"],
            samples[0]["ecg_quality"], samples[0]["ppg_quality"], 1.0,
        )),
    }
    output = ROOT / "reports/CENTRALIZED_SYSTEM_V2_SAMPLE_PREDICTIONS.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
