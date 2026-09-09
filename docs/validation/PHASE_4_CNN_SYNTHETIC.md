# Phase 4 — CNN contract validation

Date: 2026-09-09  
Run: `python3 training/train_ecg_cnn.py --synthetic --epochs 2 --patience 2 --batch-size 64 --output-dir experiments/phase4_synthetic`

## Gate result

**PASS for the implementation contract; not a research result.** The CNN now uses the same 1.1.0 preprocessing contract as the classical baselines, fits normalization on training patients only, records a run manifest, keeps the generated patient split instead of re-splitting records, and selects the decision threshold from validation F1 only. The two-epoch synthetic smoke run produced train/validation/test windows of 350/75/75 and wrote checkpoints, metrics, normalization statistics, and provenance.

Synthetic ECGs are intentionally easy to separate and are useful only for software validation. Phase 4 remains open for the real PTB-XL training run, threshold calibration on validation data, and the locked `MODEL_V1` evidence package.
