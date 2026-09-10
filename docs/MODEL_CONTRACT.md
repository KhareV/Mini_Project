# Centralized model contract for Person 3

Person 3 must federate only the explicitly selected target model. No raw data, calibration fitting, model selection, or hardware behavior is part of this handoff.

## Primary FL reference: `CENTRAL_ECG_MODEL_V1`

- Initial checkpoint: `models/MODEL_V1.pt` (SHA-256 recorded in `CENTRALIZED_SYSTEM_V1.md`).
- Architecture: `models.ecg_cnn.ECGCNN1D`, config embedded in the checkpoint.
- Input: `float32`, shape `[batch, 1, 2500]`; one canonically preprocessed ECG lead at 250 Hz.
- Output: shape `[batch, 2]`, logits ordered `[NORMAL, ABNORMAL]`.
- Label mapping: `0 = NORMAL`, `1 = ABNORMAL`.
- Loss/default centralized optimizer: weighted cross-entropy; AdamW/Adam compatible, but FL experiments must record their exact optimizer and scheduler.
- Evaluation: `scripts/evaluate_selected_ecg.py` for PTB-XL and `scripts/evaluate_mitbih_external.py` for external-only reporting.
- Preprocessing: `preprocessing.ecg.ECGPreprocessor`, version 1.0.0; no fitting on validation, test, or client evaluation data.

## Secondary model: quality-aware fusion

- Checkpoint: `continuous-health-monitor/experiments/results/multimodal_p1_integrated/checkpoint.pt`.
- It consumes canonical ECG, PPG, HR, SpO2, presence mask, and quality scores. It is not the primary FL target because its current held-out F1/AUROC do not exceed the canonical ECG reference.
- Its Platt calibration is centrally fixed in `calibration.json`; calibration is not a federated parameter and must not be aggregated.

## Non-release artifacts

`models/candidates/C1_validation_selected_nonrelease.pt` and all `experiments/ecg_model_selection/` candidates are experimental evidence only. They must not be chosen by filename similarity.
