"""Find exact canonical waveform duplicates before reporting PTB-XL metrics."""
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.ptbxl import PTBXLDataset
from preprocessing.ecg import ECGPreprocessor


def main():
    logging.getLogger().setLevel(logging.ERROR)
    ds = PTBXLDataset(ROOT / "data/raw/ptbxl")
    manifest = ds.generate_splits(ds.build_manifest(), seed=42)
    records = {record.record_id: record for record in ds.load_all_records() if record.is_valid}
    fingerprints = defaultdict(list)
    preprocessor = ECGPreprocessor()
    failures = []
    for _, row in manifest.iterrows():
        record = records.get(row.record_id)
        if record is None:
            continue
        try:
            raw, fs = ds.load_signal(record)
            processed = preprocessor.process(raw, fs).signal
            window = processed[:2500] if len(processed) >= 2500 else np.pad(processed, (0, 2500-len(processed)))
            digest = hashlib.sha256(np.asarray(window, dtype=np.float32).tobytes()).hexdigest()
            fingerprints[digest].append({"record_id": str(row.record_id), "patient_id": str(row.participant_id), "split": str(row.split), "label": int(row.label_int)})
        except Exception as exc:
            failures.append({"record_id": str(row.record_id), "error": str(exc)})
    duplicate_groups = [entries for entries in fingerprints.values() if len(entries) > 1]
    cross_split = [entries for entries in duplicate_groups if len({item["split"] for item in entries}) > 1]
    excluded_by_split = {name: [] for name in ("train", "val", "test")}
    cross_split_excluded = {name: [] for name in ("train", "val", "test")}
    # Retain train where possible, then validation, then test.  This makes every
    # evaluation sample independent of all earlier partitions.
    rank = {"train": 0, "val": 1, "test": 2}
    for group in duplicate_groups:
        keep = min(group, key=lambda item: (rank[item["split"]], item["record_id"]))
        for item in group:
            if item is not keep:
                excluded_by_split[item["split"]].append(item["record_id"])
    for group in cross_split:
        keep = min(group, key=lambda item: (rank[item["split"]], item["record_id"]))
        for item in group:
            if item is not keep:
                cross_split_excluded[item["split"]].append(item["record_id"])
    payload = {
        "method": "SHA-256 of canonical deterministic 2,500-sample float32 ECG window",
        "records_processed": int(sum(len(x) for x in fingerprints.values())),
        "failures": failures,
        "duplicate_group_count": len(duplicate_groups),
        "cross_split_duplicate_group_count": len(cross_split),
        "cross_split_groups": cross_split,
        "exclusion_policy": "retain train, then validation, then test; exclude duplicate evaluation records",
        "excluded_record_ids": excluded_by_split,
        "excluded_counts": {name: len(value) for name, value in excluded_by_split.items()},
        "cross_split_excluded_record_ids": cross_split_excluded,
    }
    (ROOT / "reports/PTBXL_CROSS_SPLIT_DUPLICATE_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
