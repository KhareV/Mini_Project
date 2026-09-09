# NHM ML System Specification

Status: `FROZEN_FOR_PHASE_1`
Specification version: `1.0.0`
Owner: sole project maintainer
Last reviewed: 2026-09-09

This is the authoritative contract for the Non-Invasive Health Monitor (NHM). If code, configuration, UI copy, or another document conflicts with this specification, this specification wins until it is deliberately versioned.

## 1. Objective

Build a research prototype that continuously acquires wearable physiological signals, rejects unreliable measurements, identifies potentially abnormal physiological patterns, and studies centralized, multimodal, federated, privacy, and edge-computing trade-offs.

The system is not a medical device and does not diagnose, treat, or provide emergency guidance.

## 2. Primary ML task

The first locked task is binary ECG-window classification:

- `NORMAL` (`0`): normal monitored ECG pattern.
- `ABNORMAL` (`1`): potentially abnormal ECG pattern.

Signal usability is evaluated before classification:

- `UNRELIABLE`: input quality is inadequate; a normal/abnormal result must not be exposed to the application.

The application wording is “Normal monitored pattern”, “Potentially abnormal pattern detected”, or “Signal unreliable—remeasure”. Disease diagnosis is outside scope.

## 3. Secondary tasks

Secondary work is gated behind a frozen centralized ECG model:

1. PPG quality and pulse representation.
2. HR and SpO2 context.
3. ECG/PPG/HR/SpO2 late fusion.
4. Federated training under controlled client heterogeneity.
5. Secure aggregation and optional differential privacy.
6. A separately benchmarked edge model.

## 4. Dataset roles

| Dataset | Frozen role | Training allowed |
|---|---|---:|
| PTB-XL 1.0.3 | Primary ECG development; train/validation/internal test | Yes, training partition only |
| MIT-BIH Arrhythmia 1.0.0 | External ECG validation | No |
| MIT-BIH Noise Stress Test 1.0.0 | Controlled robustness at 24, 18, 12, 6, 0, -6 dB | No model selection |
| BIDMC 1.0.0 | Synchronized ECG/PPG/SpO2 development | Multimodal experiments only |
| AD8232/MAX30102 recordings | Hardware and domain-shift validation | No, until a later protocol version explicitly permits it |

Raw datasets are immutable and excluded from Git. Dataset manifests, checksums, split assignments, and provenance are versioned.

## 5. ECG lead policy

The primary PTB-XL experiment uses Lead I because it is the closest declared reference for the initial AD8232 wearable placement. Lead II is a predefined sensitivity experiment, not a silent substitute. MIT-BIH MLII remains external validation and its lead mismatch must be reported as domain shift.

Changing the wearable electrode placement requires a specification revision and a new model version.

## 6. Label policy

PTB-XL diagnostic superclasses map as follows:

- `NORM` -> `NORMAL`
- `MI`, `STTC`, `CD`, `HYP` -> `ABNORMAL`

The Phase 1 data report must quantify multi-label and conflicting records before the final record-resolution policy is frozen. Until then, records containing both normal and abnormal diagnostic evidence must be flagged rather than resolved by arbitrary dictionary order.

MIT-BIH beat labels are mapped using `docs/LABEL_SCHEMA.md`. Its window-label policy is evaluated as an external-validation design choice and must report windows containing minority abnormal beats separately.

## 7. Patient-level splitting

- Split before window generation.
- Participant is the grouping unit.
- Initial proportions are 70% train, 15% validation, 15% internal test.
- Seed is 42.
- Split assignments are persisted and content-hashed.
- No participant may occur in more than one partition.
- The test partition is not used for feature selection, preprocessing selection, architecture selection, calibration, or threshold selection.

## 8. Canonical signal and window contract

Canonical ECG processing rate: 250 Hz.
Window duration: 10 seconds.
Window samples: 2500.
Stride: 5 seconds (1250 samples).

Every sample/window carries:

- participant, session, record, device, and source identifiers;
- dataset and dataset version;
- start/end timestamps and source sample offsets;
- signal, units, source rate, canonical rate, and lead;
- label and label provenance;
- quality score, state, reasons, and usability;
- preprocessing and split versions;
- modality-presence mask for later multimodal work.

## 9. ECG preprocessing

The only canonical pipeline is the top-level `preprocessing` package:

1. Validate dimensionality, length, finiteness, units, and sampling rate.
2. Resample with a documented anti-aliasing method.
3. Apply zero-phase 0.5–40 Hz filtering.
4. Apply the configured 50 Hz notch where valid for the source rate.
5. Normalize using statistics fitted on the training partition only.
6. Apply documented clipping.
7. Window after participant splitting.
8. Assess window quality before inference.

Normalization statistics are serialized with the model. Per-record normalization is not permitted in reportable experiments unless introduced as a separately named preprocessing experiment.

Backend, replay, and hardware inference must import this pipeline; duplicate preprocessing implementations cannot be used for reportable results.

## 10. Signal quality

Quality is a first-class subsystem. Initial states are `EXCELLENT`, `GOOD`, `FAIR`, and `POOR`; usability is an independently recorded Boolean derived from a versioned threshold.

The application-level rule is absolute: unusable input produces `UNRELIABLE` and does not expose a physiological class. SQI must cover at least flatline, clipping, baseline drift, high-frequency noise, power-line contamination, missing data, and implausible amplitude.

## 11. Centralized baselines and candidate model

Required order:

1. Majority baseline.
2. Logistic regression over frozen named features.
3. Random forest over the same features.
4. Lightweight 1D CNN.

Candidate models are not called `MODEL_V1`. `MODEL_V1` is created only after validation-based selection, calibration, threshold locking, robustness analysis, and packaging.

## 12. Evaluation protocol

Required classification metrics: precision, recall/sensitivity, specificity, F1, AUROC, AUPRC, confusion matrix, and sample count. Accuracy is supplementary.

Required reliability metrics: Brier score, calibration curve, and expected calibration error. The decision threshold is selected on validation data and then locked.

Required streaming metrics: false alarms per hour, event sensitivity, and detection delay. Required reporting levels are window, participant, subgroup where metadata supports it, and dataset.

Report bootstrap confidence intervals for final results. Never infer clinical validity from a single aggregate score.

## 13. Multimodal model

The first multimodal architecture uses explicit late fusion:

`ECG encoder + PPG encoder + HR/SpO2 scalars + quality/presence mask -> fusion classifier`

Required ablations are ECG; PPG; ECG+PPG; ECG+PPG+HR; and ECG+PPG+HR+SpO2. Missing and degraded modality tests are mandatory. A multimodal model advances only when it demonstrates measurable benefit or useful fallback behavior.

## 14. Model interface

Every trainable model must expose deterministic construction, forward inference, loss creation, optimizer creation, parameter serialization/loading, probability prediction, and evaluation. A packaged model includes weights, architecture config, preprocessing config, normalization artifact, label schema, calibration object, threshold, data/split identifiers, metrics, model card, and checksums.

## 15. Federated learning

Federated learning begins only after the centralized model interface and `MODEL_V1` are frozen. A client represents a deterministic group of complete participants, never arbitrary windows.

Required order:

1. IID FedAvg equivalence baseline.
2. FedAvg under label, quantity, and domain/noise skew independently.
3. FedProx on identical partitions and budgets.
4. QAPFL on identical partitions and budgets with component ablations.
5. Optional personalization.

Track global, mean-client, and worst-client metrics; selected/successful clients; samples; local epochs; round duration; failures; and transmitted bytes.

## 16. Privacy

Raw physiological data remains within its client boundary during federated experiments. Secure aggregation is the first privacy mechanism. Differential privacy is optional and, if used, must report epsilon, delta, clipping norm, noise multiplier, sampling rate, and number of rounds. UI privacy claims must reflect mechanisms actually enabled in a recorded run.

## 17. Hardware contract

Hardware development proceeds through capture, deterministic replay, serial streaming, then BLE streaming. All transports emit the same `SensorFrame` containing device/firmware/session IDs, sequence number, device and host timestamps, ECG ADC values, red/IR PPG values, rates, ADC/VREF/gain/lead metadata, contact/battery state, and packet-validity status.

The first hardware gate is replay equivalence: a captured session must produce the same processed samples and model output in offline and streamed replay. Packet loss, clock drift, reconnects, and buffer overflow are observable—not silently hidden.

## 18. Research and edge models

The research model prioritizes validated performance. The edge model is a separately versioned compressed derivative. Edge feasibility requires measured model size, parameter count, peak memory, latency, accuracy change, and—where possible—energy on the intended hardware. Laptop execution is not evidence of ESP32 feasibility.

## 19. Reproducibility and versioning

Every reportable run records:

- experiment and run IDs;
- Git commit;
- dataset, manifest, and split versions/hashes;
- preprocessing, label-schema, and model versions;
- seed and complete configuration;
- environment and dependency versions;
- produced artifact checksums;
- metrics and execution duration.

Semantic versioning is used for contracts and model packages. Each completed phase is validated, committed independently, tagged `phase-N-vX.Y.Z`, and pushed. Generated raw data, secrets, environments, and large checkpoints remain outside Git.

## 20. Truthfulness boundary

All values exposed by the backend or UI are labeled as one of `SIMULATED`, `REPLAY`, `HARDWARE`, or `EXPERIMENT`. Planned capabilities are never presented as active. Synthetic results test software behavior and cannot support scientific performance claims.

## 21. Phase 0 acceptance criteria

- One authoritative system specification exists.
- Solo ownership replaces the obsolete three-person handoff model.
- Phase order and gates are documented.
- Dataset roles, hardware boundary, output states, and evaluation rules are explicit.
- A machine-readable base configuration matches the frozen decisions.
- Automated contract tests prevent silent drift in critical settings.
