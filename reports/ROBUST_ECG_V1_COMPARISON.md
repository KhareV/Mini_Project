# ROBUST_ECG_V1 comparison

`ROBUST_ECG_V1` is a separate research candidate. It was trained only on PTB-XL training windows with synthetic Gaussian noise, synthetic baseline wander, amplitude variation, and short dropout. It did not train on PTB-XL validation/test, MIT-BIH, NSTDB, or any copied NSTDB waveform.

| Evaluation | Frozen `MODEL_V1` | `ROBUST_ECG_V1` | Interpretation |
| --- | ---: | ---: | --- |
| PTB-XL test F1 | 0.8202 | 0.8348 | Candidate improves clean held-out F1. |
| PTB-XL test AUROC | 0.9012 | 0.9000 | Essentially unchanged/slightly lower. |
| MIT-BIH F1 | 0.4630 | 0.4563 | Candidate is lower. |
| MIT-BIH AUROC | 0.7204 | 0.7688 | Candidate improves ranking discrimination. |
| NSTDB +24 dB specificity | 42.25% | 44.34% | Candidate improves. |
| NSTDB +6 dB specificity | 16.48% | 9.96% | Candidate regresses. |
| NSTDB 0 dB specificity | 1.17% | 4.61% | Candidate improves. |
| NSTDB -6 dB specificity | 0.18% | 7.44% | Candidate improves substantially, but remains inadequate for trusted classification. |

## Release decision

Do not promote this candidate to the centralized release. The benefits are not uniform across external/noise conditions and MIT-BIH F1 is lower. `models/MODEL_V1.pt` remains the canonical baseline for Person 3. `ROBUST_ECG_V1` remains available only for a separately pre-registered robustness study.

Artifacts: `experiments/robust_ecg_v1/`, `reports/robust_ecg_v1_test.json`, `reports/robust_ecg_v1_mitbih_external.json`, and `reports/robust_ecg_v1_noise_robustness.json`.
