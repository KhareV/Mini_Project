"""Train ROBUST_ECG_V1 solely from PTB-XL training windows.

MIT-BIH and NSTDB are deliberately absent from this program.  Validation F1
selects the checkpoint; evaluation is performed separately after training.
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.ecg_cnn import ECGCNNConfig
from training.train_ecg_cnn import train_cnn


def main():
    cache = np.load(ROOT / "data/cache/ptbxl_selection_windows_v1.npz")
    output = ROOT / "experiments/robust_ecg_v1"
    result = train_cnn(
        cache["X_train"], cache["y_train"], cache["X_val"], cache["y_val"], None, None,
        output, "ptbxl", 42, 48, 14, .0005, 6, "auto",
        ECGCNNConfig(model_version="ROBUST_ECG_V1", dropout=.45), 2e-4, .05, False, True,
    )
    (output / "selection.json").write_text(json.dumps({"candidate": "ROBUST_ECG_V1", "selection_metric": "validation F1", "best_val_f1": result["best_val_f1"], "checkpoint": result["best_checkpoint"], "training_sources": ["PTB-XL train only", "synthetic Gaussian noise", "synthetic baseline wander", "synthetic dropout"], "excluded_sources": ["PTB-XL validation", "PTB-XL test", "MIT-BIH", "NSTDB"], "test_accessed": False}, indent=2) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__": main()
