"""The single permitted PTB-XL test evaluation after validation-only selection."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.ptbxl import PTBXLDataset
from evaluation.metrics import compute_metrics
from models.ecg_cnn import ECGCNN1D, ECGWindowDataset
from preprocessing.ecg import ECGPreprocessor
from training.train_classical import patient_level_split_synthetic
from training.train_ecg_cnn import prepare_windows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="experiments/ecg_model_selection/C1_baseline_reproduction/best_checkpoint.pt")
    p.add_argument("--destination", default="models/candidates/C1_validation_selected_nonrelease.pt")
    p.add_argument("--no-copy", action="store_true", help="Evaluate the supplied canonical checkpoint in place.")
    p.add_argument("--output", default="reports/central_ecg_final_test.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    ds = PTBXLDataset(ROOT / "data/raw/ptbxl")
    manifest = ds.generate_splits(ds.build_manifest(), seed=args.seed)
    records = {item.record_id: item for item in ds.load_all_records() if item.is_valid}
    sigs, labels, pids = [], [], []
    for _, row in manifest.iterrows():
        item = records.get(row.record_id)
        if item is not None:
            signal, _ = ds.load_signal(item); sigs.append(signal); labels.append(int(row.label_int)); pids.append(row.participant_id)
    split = patient_level_split_synthetic(pids, np.asarray(labels), seed=args.seed)
    X_test, y_test = prepare_windows(np.asarray(sigs)[split["test"]], np.asarray(labels)[split["test"]], ECGPreprocessor(), source_fs=500)
    model, checkpoint = ECGCNN1D.load_checkpoint(args.checkpoint, "cpu")
    loader = DataLoader(ECGWindowDataset(X_test, y_test), batch_size=128)
    probs = []
    with torch.no_grad():
        for x, _ in loader: probs.extend(model.predict_proba(x)[:, 1].tolist())
    probs = np.asarray(probs)
    metrics = compute_metrics(y_test, probs >= .5, probs, split="test", model_name="CENTRAL_ECG_MODEL_V1", dataset="PTB-XL").to_dict()
    destination = ROOT / args.destination
    if args.no_copy:
        destination = ROOT / args.checkpoint
    else:
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(args.checkpoint, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    payload = {"selection": "validation F1 only; test evaluated once after C1/C2/C3 registry closed", "checkpoint": str(destination.relative_to(ROOT)), "sha256": digest, "metrics": metrics, "n_test": int(len(y_test)), "source_epoch": checkpoint.get("epoch")}
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))

if __name__ == "__main__": main()
