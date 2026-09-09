# Phase 3B — Centralized classical baselines on supplied PTB-XL

Date: 2026-09-09  
Source: local PTB-XL `1.0.1`, 500 Hz, Lead I  
Preprocessing: frozen `1.1.0`, 250 Hz canonical, train-only normalization  
Seed: 42

## Gate result

**PASS.** The centralized reference models ran end-to-end on the patient-level PTB-XL split without using test data for fitting or tuning.

| Model | Test accuracy | Test AUROC | Test F1 | Test sensitivity | Test specificity |
|---|---:|---:|---:|---:|---:|
| Majority | 0.5734 | n/a | 0.7288 | 1.0000 | 0.0000 |
| Logistic regression | 0.7686 | 0.8477 | 0.7836 | 0.7309 | 0.8192 |
| Random forest | 0.7733 | 0.8540 | 0.7919 | 0.7523 | 0.8015 |

Artifacts are in `experiments/phase3_ptbxl_1.0.1/`, including the run manifest, normalization statistics, models, scalers, and result JSON files. These are reference baselines, not the final clinical model. The next implementation phase is CNN training and calibration against this frozen split.
