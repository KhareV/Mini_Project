# Phase 3 Validation — Centralized Classical Baselines

Date: 2026-09-09
Status: `VALIDATED_ON_SYNTHETIC_FIXTURES`

## Scope

This phase validates the centralized classical-model workflow, not physiological classification performance. The synthetic generator intentionally makes its two labels easy to distinguish through heart-rate and noise changes. Its scores are therefore software-integration evidence only.

## Implemented contract

- Participant-level train/validation/test split before preprocessing.
- Training-only, saved normalization artifact.
- Strict preprocessing for all training, validation, and test signals.
- Named 42-feature extraction with SQI input.
- Majority, logistic-regression, and random-forest baselines.
- Per-run provenance manifest: Git commit, seed, split counts, preprocessing configuration hash, normalization hash, and explicit synthetic/research boundary.
- Explicit result status `SOFTWARE_VALIDATION_ONLY` in each synthetic baseline result file.
- Checked class probabilities: finite, binary, and summing to one.

## Reproducible run

```bash
python3 training/train_classical.py \
  --synthetic --n-normal 120 --n-abnormal 80 --seed 42 \
  --output-dir experiments/phase3_synthetic
```

The run creates:

```text
experiments/phase3_synthetic/
├── run_manifest.json
├── shared/normalization_stats.json
├── E01_majority/results.json
├── E02_logistic/results.json
└── E03_random_forest/results.json
```

## Validation result

The baseline run completed with 140 training, 30 validation, and 30 test synthetic participant records. The majority baseline correctly demonstrates poor abnormal-event recall. Logistic regression and random forest achieve perfect synthetic separation; this is expected from the generator and must not be cited as health-monitor performance.

The test partition is recorded as not used for tuning. The next real-data gate replaces only the data source and retains the same preprocessing artifact, split discipline, manifests, and baseline interfaces.
