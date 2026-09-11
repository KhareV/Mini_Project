# Pre-hardware centralized readiness V1

## Release decision

The centralized ECG path is ready for later hardware-domain validation and FL.
The multimodal path is technically integrated and testable, but is not a
deployment-eligible clinical predictor because the supplied BIDMC data lacks a
non-circular diagnostic target that generalizes across held-out patients.

## Verified ECG evidence

- Canonical checkpoint SHA-256:
  `2646ca77828200011996fe17137829e168c19e4bc5182fc1d1b2778ae4b01475`.
- Duplicate-cleaned, validation-calibrated PTB-XL: F1 0.8389, AUROC 0.9010,
  AUPRC 0.9319, sensitivity 0.8685, specificity 0.7409, n=3,207.
- Five-seed frozen-recipe stability: F1 0.8225 ± 0.0033 and AUROC
  0.8973 ± 0.0049 at the uncalibrated 0.5 cutoff.
- Strongest classical baseline: Random Forest F1 0.8053/AUROC 0.8650.
- MIT-BIH external and NSTDB degradation are retained as known limitations.
- TP, TN, FP, and FN examples are exported in JSON and waveform PNG form.

## End-to-end software evidence

- 122 Python tests pass.
- `POST /model/ecg/infer` serves the calibrated deployment-eligible ECG model;
  its tested quality gate returns `UNRELIABLE_SIGNAL` below 0.50.
- Canonical checkpoint, calibration, inference, quality gate, stream event
  state, persistence, and API status are covered.
- A 100-window BIDMC software replay produced 98 normal and 2 abnormal
  research decisions, with one event open and one clear transition.
- Svelte checking and production build complete successfully; existing CSS and
  bundle-size warnings do not prevent the build.

## Deferred by project phase

- Actual AD8232/MAX30102 acquisition, synchronization, domain testing, and
  wearable metrics require the physical hardware and collected sessions.
- FL, privacy, client heterogeneity, communication cost, and edge deployment
  belong to Person 3 and begin from the frozen ECG contract.
- Multimodal FL must remain an experimental track until a suitable labelled
  multimodal dataset or collected hardware cohort supports a valid model.
