# MODEL_VALIDITY_CHECK_V1

The canonical ECG checkpoint passed implementation sanity checks: zero patient
overlap, no duplicate record IDs, 100% memorization of a balanced 64-window
tiny set, and shuffled-label held-out AUROC of 0.5804 (near chance). The full
machine-readable evidence is in `reports/MODEL_VALIDITY_CHECK_V1.json`.

The full PTB-XL scan found eight exact waveform duplicate groups that crossed
partitions. This is a data issue, not a patient-ID split failure. The locked
evaluation protocol removes the affected validation/test records before
calibration or final test reporting; the exclusion list is in
`reports/PTBXL_CROSS_SPLIT_DUPLICATE_AUDIT.json`.

The standalone ECG result for current use is therefore the duplicate-cleaned,
validation-calibrated result in `canonical_ecg_calibrated_test.json`. The
original 0.50-threshold result remains available as a historical baseline.
