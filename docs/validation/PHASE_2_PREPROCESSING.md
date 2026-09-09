# Phase 2 Validation — Canonical ECG Preprocessing and SQI

Date: 2026-09-09
Preprocessing version: `1.1.0`
Status: `VALIDATED_ON_SYNTHETIC_FIXTURES`

## Frozen behavior

- Signals are validated, resampled to 250 Hz, band-pass filtered, notch filtered, normalized, and clipped through one top-level canonical pipeline.
- Training normalization statistics are fitted from signals after resampling and filtering, then serialized as a JSON artifact.
- Strict/reportable mode rejects a signal when that training-derived artifact is absent.
- Filter failure and invalid source sampling rate produce invalid preprocessing results rather than silently continuing.
- Every output records whether normalization came from a training artifact, per-signal exploratory normalization, or no normalization.
- SQI now handles empty signals as unusable and exposes its exact threshold configuration for artifact logging.

## Validation completed

Focused preprocessing/SQI tests passed using deterministic synthetic ECG fixtures:

- clean ECG;
- 500 Hz to 250 Hz resampling;
- NaN/Inf handling;
- clipping;
- deterministic repeated processing;
- empty and invalid-rate rejection;
- training-only normalization fit/save/load/use;
- strict-mode missing-artifact rejection;
- fixed window length, stride, provenance, and determinism;
- clean, noisy, baseline-drift, clipped, missing-data, flatline, and empty SQI cases.

The exact same tests must be extended with PTB-XL and captured AD8232 fixtures before this phase receives `VALIDATED_ON_REAL_DATA` status.

## Known boundary

This phase intentionally does not select a classifier, fit a decision threshold, or make clinical claims. Existing synthetic preprocessing supports software validation only.
