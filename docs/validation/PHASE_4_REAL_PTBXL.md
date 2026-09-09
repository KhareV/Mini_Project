# Phase 4 — Internal `MODEL_V1` evidence on supplied PTB-XL

Date: 2026-09-09  
Source: local PTB-XL `1.0.1`, 500 Hz, Lead I  
Preprocessing: frozen `1.1.0`, 250 Hz canonical, train-only normalization  
Seed: 42; batch size 128; early stopping patience 3

## Gate result

**PASS for the internal centralized-model gate.** The model was selected using validation F1 only. The test split was evaluated once after checkpoint selection and threshold calibration.

| Metric | Test result |
|---|---:|
| AUROC | 0.8746 |
| AUPRC | 0.9187 |
| Accuracy | 0.7878 |
| F1 | 0.8150 |
| Sensitivity | 0.8155 |
| Specificity | 0.7506 |
| Decision threshold | 0.554572 |

Training stopped at epoch 4; the best validation checkpoint was epoch 1 (validation F1 0.8225, AUROC 0.8839). This is an internal public-dataset result, not a clinical claim. External robustness (MIT-BIH/NSTDB), calibration drift, replay equivalence, and edge constraints are still required before freezing a deployment model.

Artifacts are in `experiments/phase4_ptbxl_v1/E04_ecg_cnn/`. Checkpoints are ignored by Git; the committed JSON metrics, run manifest, and normalization statistics preserve the reproducible evidence contract.
