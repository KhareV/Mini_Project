# Centralized model contract for Person 3

Person 3 receives two explicit, frozen experiment targets. No raw data, calibration fitting, model selection, or hardware behavior is part of this handoff. The tracks must use separately reported centralized and federated metrics; PTB-XL ECG and BIDMC multimodal scores are not directly comparable.

## Track A — standalone ECG reference: `CENTRAL_ECG_MODEL_V1`

- Initial checkpoint: `models/MODEL_V1.pt` (SHA-256 recorded in `CENTRALIZED_SYSTEM_V1.md`).
- Architecture: `models.ecg_cnn.ECGCNN1D`, config embedded in the checkpoint.
- Input: `float32`, shape `[batch, 1, 2500]`; one canonically preprocessed ECG lead at 250 Hz.
- Output: shape `[batch, 2]`, logits ordered `[NORMAL, ABNORMAL]`.
- Label mapping: `0 = NORMAL`, `1 = ABNORMAL`.
- Loss/default centralized optimizer: weighted cross-entropy; AdamW/Adam compatible, but FL experiments must record their exact optimizer and scheduler.
- Evaluation: `scripts/evaluate_selected_ecg.py` for PTB-XL and `scripts/evaluate_mitbih_external.py` for external-only reporting.
- Preprocessing: `preprocessing.ecg.ECGPreprocessor`, version 1.0.0; no fitting on validation, test, or client evaluation data.

## Track B — quality-aware multimodal classifier: `QUALITY_AWARE_MULTIMODAL_V1`

- Checkpoint: `continuous-health-monitor/experiments/results/multimodal_p1_integrated/checkpoint.pt`.
- It consumes canonical ECG, PPG, HR, SpO2, presence mask, and quality scores, and returns `NORMAL`, `POTENTIALLY_ABNORMAL`, or `UNRELIABLE_SIGNAL` through the locked quality rule.
- The embedded P1 ECG backbone remains frozen and identical on every client. Federate only the fusion model's trainable PPG encoder, ECG projection, tabular encoder, and fusion-head parameters. Do not unfreeze or aggregate the P1 backbone in this track.
- Its Platt calibration is centrally fixed in `calibration.json`; calibration is not a federated parameter and must not be aggregated.
- Preserve the locked 0.28 decision threshold and the `quality < 0.50 => UNRELIABLE_SIGNAL` rule during client and server evaluation.

## Required reporting for both tracks

- Start every FL run from the listed checkpoint and record its SHA-256.
- Compare each federated round with that track's own frozen centralized baseline, using identical labels, preprocessing, and held-out protocol.
- Report F1, AUROC, AUPRC, sensitivity/recall, specificity, calibration/threshold, client count, client split, rounds, aggregation method, and optimizer.
- Keep Track A PTB-XL/MIT-BIH/NSTDB evaluation separate from Track B BIDMC evaluation. A cross-track conclusion requires a future common dataset and label protocol.

## Non-release artifacts

`models/candidates/C1_validation_selected_nonrelease.pt` and all `experiments/ecg_model_selection/` candidates are experimental evidence only. They must not be chosen by filename similarity.
