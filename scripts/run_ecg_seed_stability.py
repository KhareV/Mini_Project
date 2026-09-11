"""Repeat the frozen ECG training recipe across fixed seeds."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ecg_cnn import ECGCNNConfig
from training.train_ecg_cnn import train_cnn


def main():
    data = np.load(ROOT / "data/cache/ptbxl_windows_v2.npz")
    seeds = [42, 123, 456, 789, 2026]
    rows = []
    for seed in seeds:
        metrics, _ = train_cnn(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            data["X_test"], data["y_test"],
            ROOT / "experiments/ecg_seed_stability_v2" / f"seed_{seed}",
            dataset="PTB-XL_V2", seed=seed, batch_size=64, max_epochs=3,
            lr=0.001, patience=3, device="cpu",
            model_config=ECGCNNConfig(model_version="ECG_SEED_STABILITY_V2"),
            weight_decay=1e-4, augment_noise_std=0.02,
        )
        rows.append({"seed": seed, **metrics.to_dict()})
    summary = {"protocol": "Frozen 3-epoch ECG recipe; duplicate-cleaned fixed split; no seed selected from test", "runs": rows}
    for key in ("f1", "auroc", "auprc", "specificity", "recall"):
        values = np.asarray([row[key] for row in rows])
        summary[f"{key}_mean"] = round(float(values.mean()), 4)
        summary[f"{key}_std"] = round(float(values.std(ddof=1)), 4)
    (ROOT / "reports/ECG_SEED_STABILITY_V2.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
