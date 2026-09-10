"""Fit the standalone ECG decision rule on validation data only, then test once.

The checkpoint is never changed.  This script writes a separately versioned
calibration/threshold artifact so a 0.5 default cannot be mistaken for a
validated operating point.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.ptbxl import PTBXLDataset
from evaluation.metrics import compute_metrics
from models.ecg_cnn import ECGCNN1D, ECGWindowDataset
from preprocessing.ecg import ECGPreprocessor
from training.train_classical import patient_level_split_synthetic
from training.train_ecg_cnn import prepare_windows


def predict(model, X):
    loader = DataLoader(ECGWindowDataset(X, np.zeros(len(X), dtype=np.int64)), batch_size=128)
    values = []
    with torch.no_grad():
        for x, _ in loader:
            values.extend(model.predict_proba(x)[:, 1].tolist())
    return np.asarray(values)


def ece(y, p, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p >= low) & ((p < high) if high < 1 else (p <= high))
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def logit(probabilities):
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/MODEL_V1.pt")
    parser.add_argument("--cache", default="data/cache/ptbxl_selection_windows_v1.npz")
    parser.add_argument("--output", default="reports/MODEL_V1_validation_calibration.json")
    parser.add_argument("--test-output", default="reports/canonical_ecg_calibrated_test.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache = np.load(ROOT / args.cache)
    X_val, y_val = cache["X_val"], cache["y_val"]
    ds = PTBXLDataset(ROOT / "data/raw/ptbxl")
    manifest = ds.generate_splits(ds.build_manifest(), seed=args.seed)
    records = {r.record_id: r for r in ds.load_all_records() if r.is_valid}
    ordered_rows = [row for _, row in manifest.iterrows() if row.record_id in records]
    ordered_ids = [str(row.record_id) for row in ordered_rows]
    ordered_labels = np.asarray([int(row.label_int) for row in ordered_rows])
    ordered_pids = [row.participant_id for row in ordered_rows]
    split = patient_level_split_synthetic(ordered_pids, ordered_labels, seed=args.seed)
    duplicate_audit = json.loads((ROOT / "reports/PTBXL_CROSS_SPLIT_DUPLICATE_AUDIT.json").read_text())
    excluded = duplicate_audit["cross_split_excluded_record_ids"]
    val_ids = np.asarray(ordered_ids, dtype=object)[split["val"]]
    val_keep = ~np.isin(val_ids, excluded["val"])
    X_val, y_val = X_val[val_keep], y_val[val_keep]
    model, _ = ECGCNN1D.load_checkpoint(ROOT / args.checkpoint, "cpu")
    raw_val = predict(model, X_val)
    platt = LogisticRegression(random_state=args.seed, solver="lbfgs")
    platt.fit(logit(raw_val).reshape(-1, 1), y_val)
    val_prob = platt.predict_proba(logit(raw_val).reshape(-1, 1))[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    threshold = float(max(thresholds, key=lambda t: f1_score(y_val, val_prob >= t)))
    calibration = {
        "checkpoint": args.checkpoint,
        "fit_split": "PTB-XL validation only",
        "seed": args.seed,
        "method": "Platt logistic regression on clipped-logit abnormal probability",
        "coefficient": float(platt.coef_[0, 0]),
        "intercept": float(platt.intercept_[0]),
        "locked_threshold": threshold,
        "validation": {
            "n": int(len(y_val)), "f1": round(float(f1_score(y_val, val_prob >= threshold)), 4),
            "auroc": round(float(roc_auc_score(y_val, val_prob)), 4),
            "brier": round(float(brier_score_loss(y_val, val_prob)), 6), "ece_10": round(ece(y_val, val_prob), 6),
        },
        "test_accessed_for_selection": False,
    }
    (ROOT / args.output).write_text(json.dumps(calibration, indent=2) + "\n")

    signals, labels, pids = [], [], []
    for row in ordered_rows:
        record = records.get(row.record_id)
        if record is not None:
            signal, _ = ds.load_signal(record)
            signals.append(signal); labels.append(int(row.label_int)); pids.append(row.participant_id)
    split = patient_level_split_synthetic(pids, np.asarray(labels), seed=args.seed)
    X_test, y_test = prepare_windows(np.asarray(signals)[split["test"]], np.asarray(labels)[split["test"]], ECGPreprocessor(), source_fs=500)
    test_ids = np.asarray(ordered_ids, dtype=object)[split["test"]]
    test_keep = ~np.isin(test_ids, excluded["test"])
    X_test, y_test = X_test[test_keep], y_test[test_keep]
    raw_test = predict(model, X_test)
    test_prob = platt.predict_proba(logit(raw_test).reshape(-1, 1))[:, 1]
    metrics = compute_metrics(y_test, test_prob >= threshold, test_prob, split="test", model_name="CENTRAL_ECG_MODEL_V1", dataset="PTB-XL", notes="Validation-only Platt calibration and locked threshold.").to_dict()
    payload = {"calibration": args.output, "metrics": metrics, "n_test": int(len(y_test)), "test_used_after_lock": True}
    (ROOT / args.test_output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"calibration": calibration, "test": payload}, indent=2))


if __name__ == "__main__":
    main()
