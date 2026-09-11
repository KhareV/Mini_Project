# Centralized model contract for Person 3

Person 3 receives one validated trainable target and one deterministic sensor
component. No raw data, calibration fitting, model selection, or hardware
behavior is part of this handoff.

## Track A — standalone ECG reference: `CENTRAL_ECG_MODEL_V1`

- Initial checkpoint: `models/MODEL_V1.pt` (SHA-256 recorded in `CENTRALIZED_SYSTEM_V2.md`).
- Architecture: `models.ecg_cnn.ECGCNN1D`, config embedded in the checkpoint.
- Input: `float32`, shape `[batch, 1, 2500]`; one canonically preprocessed ECG lead at 250 Hz.
- Output: shape `[batch, 2]`, logits ordered `[NORMAL, ABNORMAL]`.
- Label mapping: `0 = NORMAL`, `1 = ABNORMAL`.
- Operating point: apply the validation-only Platt calibration in `reports/MODEL_V1_validation_calibration.json`, then the locked threshold `0.37`. The checkpoint itself is unchanged.
- Exclude the duplicate test records enumerated in `reports/PTBXL_CROSS_SPLIT_DUPLICATE_AUDIT.json` from PTB-XL validation/test reporting. Never fit calibration or select a threshold on those reports.
- Loss/default centralized optimizer: weighted cross-entropy; AdamW/Adam compatible, but FL experiments must record their exact optimizer and scheduler.
- Evaluation: `scripts/evaluate_selected_ecg.py` for PTB-XL and `scripts/evaluate_mitbih_external.py` for external-only reporting.
- Preprocessing: `preprocessing.ecg.ECGPreprocessor`, version 1.0.0; no fitting on validation, test, or client evaluation data.

## Track B — PPG pulse component: `PPG_PULSE_ESTIMATOR_V3`

- Implementation: `continuous-health-monitor/ml/models/ppg_pulse_estimator.py`.
- It consumes one 1,250-sample PPG window and outputs pulse, uncertainty/quality,
  and reliability. It has no trainable parameters and is not federated.
- Preserve its 0.50 quality gate and fixed pulse alert range of 60–100 BPM.
- SpO2 remains a synchronized device measurement pending red/IR hardware
  calibration. Do not invent a learned SpO2 checkpoint from single-channel BIDMC.

## Rejected learned fusion track

The historical quality-aware BIDMC checkpoint must not be federated as a
validated predictor. Its contemporaneous endpoint was circular; the
non-circular future-alert replacement failed to generalize. The architecture
may be reused only after a suitable independently labelled cohort exists.

## Required reporting

- Start ECG FL runs from the listed checkpoint and record its SHA-256.
- Compare each federated round with the frozen ECG centralized baseline, using identical labels, preprocessing, and held-out protocol.
- Report F1, AUROC, AUPRC, sensitivity/recall, specificity, calibration/threshold, client count, client split, rounds, aggregation method, and optimizer.
- Keep ECG classification and PPG pulse-estimation metrics separate. A
  cross-component performance claim requires a future common clinical endpoint.

## Non-release artifacts

`models/candidates/C1_validation_selected_nonrelease.pt` and all `experiments/ecg_model_selection/` candidates are experimental evidence only. They must not be chosen by filename similarity.
