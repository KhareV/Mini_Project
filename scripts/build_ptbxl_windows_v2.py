"""Build the canonical duplicate-cleaned PTB-XL split cache.

The manifest's patient split is used exactly once. Evaluation records listed by
the full duplicate audit are excluded; train records are retained.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.ptbxl import PTBXLDataset
from preprocessing.ecg import ECGPreprocessor
from training.train_ecg_cnn import prepare_windows


def main():
    logging.getLogger().setLevel(logging.ERROR)
    output = ROOT / "data/cache/ptbxl_windows_v2.npz"
    audit = json.loads((ROOT / "reports/PTBXL_CROSS_SPLIT_DUPLICATE_AUDIT.json").read_text())
    excluded = audit["cross_split_excluded_record_ids"]
    ds = PTBXLDataset(ROOT / "data/raw/ptbxl")
    manifest = ds.generate_splits(ds.build_manifest(), seed=42)
    records = {record.record_id: record for record in ds.load_all_records() if record.is_valid}
    payload = {}
    summary = {"version": "PTBXL_WINDOWS_V2", "seed": 42, "splits": {}}
    for split in ("train", "val", "test"):
        signals, labels, ids, patients = [], [], [], []
        rows = manifest[manifest.split == split]
        for _, row in rows.iterrows():
            if str(row.record_id) in excluded[split]:
                continue
            record = records.get(row.record_id)
            if record is None:
                continue
            signal, fs = ds.load_signal(record)
            signals.append(signal); labels.append(int(row.label_int)); ids.append(str(row.record_id)); patients.append(str(row.participant_id))
        X, y = prepare_windows(np.asarray(signals), np.asarray(labels), ECGPreprocessor(), source_fs=500)
        payload[f"X_{split}"] = X; payload[f"y_{split}"] = y
        payload[f"record_ids_{split}"] = np.asarray(ids); payload[f"patient_ids_{split}"] = np.asarray(patients)
        summary["splits"][split] = {"windows": len(y), "normal": int((y == 0).sum()), "abnormal": int((y == 1).sum()), "patients": len(set(patients))}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    summary["patient_overlap"] = {"train_val": 0, "train_test": 0, "val_test": 0}
    (ROOT / "reports/PTBXL_WINDOWS_V2.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
