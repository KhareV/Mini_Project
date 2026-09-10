# MIT-BIH external validation record

The earlier prose report in this location was stale and must not be used.
The authoritative direct evaluation of `models/MODEL_V1.pt` is
`canonical_ecg_mitbih_external.json`:

- F1: 0.4630
- AUROC: 0.7204
- Sensitivity: 0.9990
- Specificity: 0.2579
- Windows: 16,559 (12,541 normal / 4,018 abnormal)

MIT-BIH is external-only: it was not used for fitting, threshold selection,
or model selection. The label audit is recorded in
`MODEL_VALIDITY_CHECK_V1.json`: supported annotated beats map to the binary
schema, excluded annotation symbols are discarded, and a window is abnormal
only when abnormal beats outnumber normal beats.

This is evidence of substantial domain shift and an overly sensitive fixed
ECG decision rule on MIT-BIH; it is not evidence of wearable performance.
