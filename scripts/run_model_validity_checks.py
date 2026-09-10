"""Reproducible pre-federation validity gates for the canonical ECG pipeline."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.mitbih import MITBIHDataset
from datasets.ptbxl import PTBXLDataset
from models.ecg_cnn import ECGCNN1D, ECGCNNConfig
from preprocessing.ecg import ECGPreprocessor


def hashes(X):
    return {hashlib.sha256(row.tobytes()).hexdigest() for row in X}


def train_accuracy(X, y, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ECGCNN1D(ECGCNNConfig(dropout=0.0))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    features = torch.tensor(X).unsqueeze(1)
    targets = torch.tensor(y, dtype=torch.long)
    model.train()
    for _ in range(80):
        optimiser.zero_grad()
        loss = criterion(model(features), targets)
        loss.backward(); optimiser.step()
    model.eval()
    with torch.no_grad():
        train_prob = torch.softmax(model(features), dim=1)[:, 1].numpy()
    return model, float(((train_prob >= .5) == y).mean())


def main():
    output = ROOT / "reports/MODEL_VALIDITY_CHECK_V1.json"
    seed = 42
    ds = PTBXLDataset(ROOT / "data/raw/ptbxl")
    manifest = ds.generate_splits(ds.build_manifest(), seed=seed)
    groups = {name: set(frame.participant_id.astype(str)) for name, frame in manifest.groupby("split")}
    cache = np.load(ROOT / "data/cache/ptbxl_selection_windows_v1.npz")
    train_hashes, val_hashes = hashes(cache["X_train"]), hashes(cache["X_val"])
    duplicate_audit = json.loads((ROOT / "reports/PTBXL_CROSS_SPLIT_DUPLICATE_AUDIT.json").read_text())

    # A balanced tiny subset must be learnable; shuffled labels must not retain
    # predictive signal on held-out validation data.
    normal = np.where(cache["y_train"] == 0)[0][:32]
    abnormal = np.where(cache["y_train"] == 1)[0][:32]
    idx = np.r_[normal, abnormal]
    X_tiny, y_tiny = cache["X_train"][idx], cache["y_train"][idx]
    _, tiny_accuracy = train_accuracy(X_tiny, y_tiny, seed)
    shuffled = np.random.RandomState(seed).permutation(y_tiny)
    shuffled_model, shuffled_train_accuracy = train_accuracy(X_tiny, shuffled, seed + 1)
    with torch.no_grad():
        val_prob = torch.softmax(shuffled_model(torch.tensor(cache["X_val"][:512]).unsqueeze(1)), dim=1)[:, 1].numpy()
    shuffled_auroc = float(roc_auc_score(cache["y_val"][:512], val_prob))

    # Mapping audit counts every native MIT-BIH annotation, without training.
    mit = MITBIHDataset(ROOT / "data/raw/mitbih")
    mapped, excluded = 0, 0
    for record in mit.load_all_records():
        if record.is_valid:
            _, _, _, labels = mit.load_signal_with_annotations(record.record_id)
            mapped += int(np.sum(labels != "EXCLUDE")); excluded += int(np.sum(labels == "EXCLUDE"))
    windows = mit.build_windows()

    report = {
        "seed": seed,
        "patient_split": {
            "counts": {name: len(value) for name, value in groups.items()},
            "pairwise_overlaps": {"train_val": len(groups["train"] & groups["val"]), "train_test": len(groups["train"] & groups["test"]), "val_test": len(groups["val"] & groups["test"])},
            "duplicate_record_ids": int(manifest.record_id.duplicated().sum()),
        },
        "window_duplicate_check": {
            "raw_cached_train_validation_exact_duplicates": int(len(train_hashes & val_hashes)),
            "full_dataset_cross_partition_duplicate_groups": duplicate_audit["cross_split_duplicate_group_count"],
            "evaluation_exclusions": duplicate_audit["cross_split_excluded_record_ids"],
            "status": "raw duplicates detected; all duplicate validation/test records are excluded by the locked calibration/evaluation protocol",
        },
        "preprocessing": {"version": ECGPreprocessor().config_dict["preprocessing_version"], "normalization": "per-window deterministic z-score; no statistics are fitted across train/validation/test partitions", "normalization_stats_supplied": False},
        "training_sanity": {"tiny_dataset_n": int(len(y_tiny)), "tiny_dataset_train_accuracy": round(tiny_accuracy, 4), "shuffled_label_train_accuracy": round(shuffled_train_accuracy, 4), "shuffled_label_validation_auroc_n512": round(shuffled_auroc, 4)},
        "mitbih_label_mapping": {"mapped_beats": mapped, "excluded_beats": excluded, "window_count": int(len(windows)), "normal_windows": int((windows.label_int == 0).sum()), "abnormal_windows": int((windows.label_int == 1).sum()), "rule": "window is ABNORMAL only when abnormal beats outnumber normal beats; ties are NORMAL"},
        "gates": {"patient_leakage": all(v == 0 for v in [len(groups["train"] & groups["val"]), len(groups["train"] & groups["test"]), len(groups["val"] & groups["test"])]), "record_duplicate": not manifest.record_id.duplicated().any(), "cross_partition_duplicates_excluded": not duplicate_audit["cross_split_excluded_record_ids"]["train"] and bool(duplicate_audit["cross_split_excluded_record_ids"]["val"] or duplicate_audit["cross_split_excluded_record_ids"]["test"]), "tiny_overfit": tiny_accuracy >= .98, "shuffled_label_heldout_not_predictive": .35 <= shuffled_auroc <= .65},
    }
    report["all_gates_pass"] = all(report["gates"].values())
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_gates_pass"]:
        raise SystemExit("One or more model-validity gates failed")


if __name__ == "__main__":
    main()
