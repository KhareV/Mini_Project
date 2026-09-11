"""Run required majority/logistic/random-forest baselines on real PTB-XL."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.ecg_features import ECGFeatureExtractor
from training.train_classical import run_logistic_regression, run_majority_baseline, run_random_forest


def matrix(X):
    extractor = ECGFeatureExtractor(fs=250)
    rows = []
    for signal in X:
        rows.append(extractor.extract(signal).feature_values)
    return np.nan_to_num(np.asarray(rows, dtype=np.float32)), extractor.FEATURE_NAMES


def main():
    cache_path = ROOT / "data/cache/ptbxl_windows_v2.npz"
    feature_path = ROOT / "data/cache/ptbxl_classical_features_v2.npz"
    windows = np.load(cache_path)
    if feature_path.exists():
        f = np.load(feature_path); X_train, X_val, X_test = f["X_train"], f["X_val"], f["X_test"]
    else:
        X_train, names = matrix(windows["X_train"]); X_val, _ = matrix(windows["X_val"]); X_test, _ = matrix(windows["X_test"])
        np.savez_compressed(feature_path, X_train=X_train, X_val=X_val, X_test=X_test, feature_names=np.asarray(names))
    y_train, y_val, y_test = windows["y_train"], windows["y_val"], windows["y_test"]
    out = ROOT / "experiments/ptbxl_real_baselines_v2"
    majority = run_majority_baseline(y_train, y_val, y_test, out / "majority", dataset="PTB-XL_V2")
    logistic, _, _ = run_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, out / "logistic", dataset="PTB-XL_V2", seed=42)
    forest, _ = run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, out / "random_forest", dataset="PTB-XL_V2", seed=42)
    report = {"dataset": "PTB-XL duplicate-cleaned V2", "majority": majority["test"], "logistic_regression": logistic["test"], "random_forest": forest["test"]}
    (ROOT / "reports/PTBXL_REAL_BASELINES_V2.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
