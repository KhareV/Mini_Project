"""Deterministic software-only end-to-end replay through the serving boundary."""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.services.streaming_inference import StreamingInferenceService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="continuous-health-monitor/data/processed/bidmc_test_windows.json")
    parser.add_argument("--output", default="reports/bidmc_stream_replay.json")
    parser.add_argument("--max-windows", type=int, default=100)
    args = parser.parse_args()
    samples = json.loads(Path(args.input).read_text())[:args.max_windows]
    stream = StreamingInferenceService()
    counts, transitions = Counter(), Counter()
    for sample in samples:
        window = {
            "ecg": sample["ecg"], "ppg": sample["ppg"], "hr": sample["hr_mean"], "spo2": sample["spo2_mean"],
            "presence_mask": [float(sample[key]) for key in ("ecg_present", "ppg_present", "hr_present", "spo2_present")],
            "quality_scores": [sample["ecg_quality"], sample["ppg_quality"], 1.0, 1.0],
        }
        output = stream.ingest(str(sample["participant_id"]), [window])
        decision = output["decisions"][0]
        counts[decision["prediction"]] += 1
        if decision["transition"]:
            transitions[decision["transition"]] += 1
    report = {"source": "BIDMC held-out test split", "windows_replayed": len(samples), "decision_counts": dict(counts), "event_transitions": dict(transitions), "software_only": True, "hardware_integration": False}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
