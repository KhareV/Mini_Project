"""Export auditable TP/TN/FP/FN examples from the locked ECG model."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.ecg_cnn import ECGCNN1D


def main():
    data = np.load(ROOT / "data/cache/ptbxl_windows_v2.npz")
    X, y, ids = data["X_test"], data["y_test"], data["record_ids_test"]
    model, _ = ECGCNN1D.load_checkpoint(ROOT / "models/MODEL_V1.pt", "cpu")
    raw = []
    loader = DataLoader(TensorDataset(torch.tensor(X).unsqueeze(1)), batch_size=128)
    with torch.no_grad():
        for (batch,) in loader: raw.extend(model.predict_proba(batch)[:, 1].tolist())
    raw = np.asarray(raw)
    cal = json.loads((ROOT / "reports/MODEL_V1_validation_calibration.json").read_text())
    clipped = np.clip(raw, 1e-6, 1 - 1e-6); logits = np.log(clipped / (1-clipped))
    prob = 1 / (1 + np.exp(-(cal["coefficient"] * logits + cal["intercept"])))
    pred = (prob >= cal["locked_threshold"]).astype(int)
    masks = {"true_positive": (y == 1) & (pred == 1), "true_negative": (y == 0) & (pred == 0), "false_positive": (y == 0) & (pred == 1), "false_negative": (y == 1) & (pred == 0)}
    selected = {}
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for axis, (name, mask) in zip(axes, masks.items()):
        candidates = np.where(mask)[0]
        index = int(candidates[np.argmax(np.abs(prob[candidates] - cal["locked_threshold"]))])
        selected[name] = {"record_id": str(ids[index]), "true_label": int(y[index]), "predicted_label": int(pred[index]), "raw_probability": round(float(raw[index]), 6), "calibrated_probability": round(float(prob[index]), 6), "threshold": cal["locked_threshold"]}
        axis.plot(np.arange(2500) / 250.0, X[index], linewidth=.7)
        axis.set_title(f"{name.replace('_',' ').title()} | {ids[index]} | p={prob[index]:.3f}")
        axis.set_ylabel("normalized")
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    (ROOT / "reports/ecg_prediction_examples.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "reports/ecg_prediction_examples.png", dpi=150); plt.close(fig)
    payload = {"checkpoint": "models/MODEL_V1.pt", "calibration": "reports/MODEL_V1_validation_calibration.json", "selection": "most decision-distant example in each confusion category; illustrative, not a metric", "examples": selected}
    (ROOT / "reports/ECG_PREDICTION_EXAMPLES_V1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
