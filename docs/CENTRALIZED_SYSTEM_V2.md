# CENTRALIZED_SYSTEM_V2 — clinically honest centralized release

## Release decision

`MODEL_V1` remains the locked ECG classifier. `PPG_PULSE_ESTIMATOR_V3` is the
new non-circular PPG component. Synchronized SpO2 is treated as a measured
sensor value with validity and quality checks; it is not fabricated from the
single PPG channel in BIDMC. The released system combines component decisions
with a quality-gated conservative OR.

This is **clinical-validation ready research software**, not a clinically
eligible medical device. Clinical eligibility requires prospective hardware
data, an appropriate reference standard, subgroup analysis, clinical review,
and the applicable regulatory process.

## Released components

| Component | Version | Evidence | Release status |
| --- | --- | --- | --- |
| ECG pattern classifier | `MODEL_V1` | PTB-XL test F1 0.8389, AUROC 0.9010, AUPRC 0.9319 | locked research-monitoring model |
| PPG pulse estimator | `PPG_PULSE_ESTIMATOR_V3` | BIDMC patient-held-out test MAE 2.346 BPM; gated MAE 1.810 BPM at 77.4% coverage | software release; external wearable confirmation required |
| SpO2 | synchronized device measurement | range, missingness, timestamp, and quality validation | hardware calibration pending |
| Learned BIDMC fusion | historical V1 checkpoint | circular original endpoint; future endpoint AUROC 0.3768 | rejected for release/FL |
| System fusion | deterministic quality-gated OR | unit/API integration tests | software release; prospective validation pending |

The PPG test pulse-alert endpoint (`<60` or `>100` BPM) achieved F1 0.8659,
sensitivity 0.8712, and specificity 0.9542. This endpoint is narrower than the
old composite BIDMC label and must not be compared directly with its F1.

## Why the model structure changed

The old learned fusion used HR and SpO2 as inputs while those same values
defined its contemporaneous target. That makes high scores circular rather
than evidence of generalization. A non-circular 30–90 second future-alert
replacement failed on held-out patients. Consequently:

1. ECG performs ECG pattern classification.
2. PPG estimates pulse from waveform timing and reports uncertainty.
3. SpO2 comes from the future MAX30102 red/IR sensor path and is never inferred
   from BIDMC's single pleth waveform.
4. Fusion preserves independent outputs and refuses low-quality evidence.

## Hosted endpoints

- `POST /model/ecg/infer` — locked calibrated ECG inference.
- `POST /model/vitals/infer` — PPG pulse plus measured-SpO2 validation.
- `POST /model/system/infer` — integrated quality-gated system decision.
- `POST /model/infer` — retained legacy learned-fusion research endpoint; not
  release eligible.

## Known limitations and next legitimate gate

`PPG_PULSE_ESTIMATOR_V2` initially failed on one test subject by locking onto a
second harmonic. That report is preserved. V3 adds a validation-selected
spectral disagreement fallback and is therefore a post-test corrective
candidate, not pristine external confirmation. Actual MAX30102 sessions are
required to confirm V3 and calibrate SpO2. Until then `clinical_use_eligible`
remains `false` in every new endpoint.

