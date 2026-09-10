"""Validation-only ECG improvement campaign.

The sealed PTB-XL test split is loaded solely to preserve the existing split
indices; it is neither passed to training nor evaluated by this program.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.ptbxl import PTBXLDataset
from preprocessing.ecg import ECGPreprocessor
from training.train_classical import patient_level_split_synthetic
from training.train_ecg_cnn import prepare_windows, train_cnn
from models.ecg_cnn import ECGCNNConfig


def load_data(dataset_root: Path, cache: Path, seed: int):
    if cache.exists():
        loaded = np.load(cache, allow_pickle=True)
        return tuple(loaded[name] for name in ("X_train", "y_train", "X_val", "y_val"))
    ds = PTBXLDataset(data_dir=dataset_root)
    manifest = ds.generate_splits(ds.build_manifest(), seed=seed)
    records = {record.record_id: record for record in ds.load_all_records() if record.is_valid}
    signals, labels, pids = [], [], []
    for _, row in manifest.iterrows():
        record = records.get(row["record_id"])
        if record is None:
            continue
        signal, _ = ds.load_signal(record)
        signals.append(signal); labels.append(int(row["label_int"])); pids.append(row["participant_id"])
    signals, labels = np.asarray(signals), np.asarray(labels)
    split = patient_level_split_synthetic(pids, labels, seed=seed)
    preprocessor = ECGPreprocessor()
    X_train, y_train = prepare_windows(signals[split["train"]], labels[split["train"]], preprocessor, source_fs=500)
    X_val, y_val = prepare_windows(signals[split["val"]], labels[split["val"]], preprocessor, source_fs=500)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    return X_train, y_train, X_val, y_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl", default="data/raw/ptbxl")
    parser.add_argument("--output", default="experiments/ecg_model_selection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=18)
    args = parser.parse_args()
    output = ROOT / args.output
    X_train, y_train, X_val, y_val = load_data(ROOT / args.ptbxl, ROOT / "data/cache/ptbxl_selection_windows_v1.npz", args.seed)
    trials = [
        ("C1_baseline_reproduction", ECGCNNConfig(model_version="C1", dropout=.4), .001, 1e-4, .02),
        ("C2_regularized", ECGCNNConfig(model_version="C2", dropout=.5), .0005, 5e-4, .01),
        ("C3_wider_features", ECGCNNConfig(model_version="C3", conv_channels=[48, 96, 192, 256], fc_hidden=160, dropout=.45), .0005, 2e-4, .01),
    ]
    registry = []
    for name, config, lr, weight_decay, noise in trials:
        result = train_cnn(X_train, y_train, X_val, y_val, None, None, output / name, "ptbxl", args.seed, 48, args.epochs, lr, 6, "auto", config, weight_decay, noise, False)
        registry.append({"experiment_id": name, "best_val_f1": result["best_val_f1"], "checkpoint": result["best_checkpoint"], "model_config": result["config"], "lr": lr, "weight_decay": weight_decay, "augment_noise_std": noise, "test_accessed": False})
    registry.sort(key=lambda item: item["best_val_f1"], reverse=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "registry.json").write_text(json.dumps({"selection_metric": "validation F1", "test_accessed": False, "trials": registry, "winner": registry[0]}, indent=2) + "\n")
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
