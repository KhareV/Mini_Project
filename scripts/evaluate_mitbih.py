"""Evaluate locked MODEL_V1 on MIT-BIH without fitting or tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.mitbih import MITBIHDataset
from evaluation.metrics import compute_metrics
from inference.ecg_model_v1 import ECGModelV1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/mitbih")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--threshold", type=float, required=True,
                        help="Locked validation threshold; never tune on MIT-BIH")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = MITBIHDataset(args.data_dir, preprocessing_version="1.1.0")
    model = ECGModelV1.load(args.checkpoint, args.normalization, args.threshold)
    windows = dataset.build_windows(window_seconds=10, stride_seconds=5, target_fs=250)
    signals, labels = [], []
    cache = {}
    for _, row in windows.iterrows():
        if args.max_windows and len(labels) >= args.max_windows:
            break
        record_id = row["record_id"]
        if record_id not in cache:
            signal, fs, _, _ = dataset.load_signal_with_annotations(record_id)
            if fs != 250:
                signal = resample(signal, int(len(signal) * 250 / fs)).astype(np.float32)
            cache[record_id] = signal
        signal = cache[record_id]
        segment = signal[int(row["start_sample"]):int(row["end_sample"])]
        prediction = model.predict(segment, 250)
        signals.append(prediction.probability_abnormal)
        labels.append(int(row["label_int"]))
    probabilities = np.asarray(signals)
    y_true = np.asarray(labels)
    y_pred = (probabilities >= args.threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, probabilities, split="external",
                              model_name="MODEL_V1", dataset="MIT-BIH_1.0.0",
                              notes="Locked PTB-XL model; no MIT-BIH fitting or tuning")
    result = {"dataset": "MIT-BIH_1.0.0", "windows": len(labels),
              "threshold": args.threshold, "metrics": metrics.to_dict()}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
