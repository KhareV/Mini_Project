# NHM Iterative Implementation Roadmap

This roadmap is executed by one maintainer. Every phase follows the same loop:

1. Confirm the phase inputs and frozen contracts.
2. Implement the smallest end-to-end increment.
3. Add automated tests and one reproducible validation command.
4. Produce a human-readable validation report.
5. Review the diff and confirm no future capability is being claimed prematurely.
6. Commit only the phase scope, tag it, and push it.

## Status vocabulary

- `NOT_STARTED`: only an idea or placeholder exists.
- `SCAFFOLDED`: interfaces or simulations exist but no real evidence exists.
- `IMPLEMENTED`: code exists and focused automated tests pass.
- `VALIDATED`: acceptance criteria pass on the intended real dataset/device.
- `FROZEN`: versioned artifacts are locked for downstream consumers.

## Phase plan

| Phase | Outcome | Validation gate | Initial status |
|---:|---|---|---|
| 0 | System, evaluation, data, hardware, and version-control contracts | Contract tests and document review | IMPLEMENTED |
| 1 | Reproducible PTB-XL data foundation | Completeness report, hashes, zero participant leakage | VALIDATED (local 1.0.1 source; 1.0.3 remains target) |
| 2 | Frozen canonical ECG preprocessing and SQI | Synthetic/filter tests plus real-record audit | VALIDATED_ON_SYNTHETIC_FIXTURES |
| 3 | Centralized classical reference models | Reproducible PTB-XL validation runs | VALIDATED (local 1.0.1 source) |
| 4 | Calibrated centralized ECG CNN and MODEL_V1 | Locked threshold; internal and external reports | VALIDATED (internal PTB-XL 1.0.1; external pending) |
| 5 | Backend WFDB/captured-file replay using MODEL_V1 | Offline/stream prediction equivalence | IMPLEMENTED (offline replay boundary) |
| 6 | PPG, SpO2, BIDMC, and synchronization | Timestamp-alignment and quality tests | SCAFFOLDED |
| 7 | Centralized multimodal model | Required ablations and missing-modality tests | NOT_STARTED |
| 8 | Deterministic federated client partitions | IID/non-IID manifest audits | NOT_STARTED |
| 9 | Actual Flower FedAvg | Toy aggregation and IID convergence comparison | SCAFFOLDED |
| 10 | FedProx, QAPFL, and heterogeneity study | Controlled same-budget comparisons | SCAFFOLDED |
| 11 | Secure aggregation and optional DP | Threat-model and privacy/overhead reports | NOT_STARTED |
| 12 | Captured, serial, then BLE hardware integration | Replay equivalence and packet-loss tests | SCAFFOLDED |
| 13 | Separately versioned edge model | Device/gateway latency, memory, size and accuracy | NOT_STARTED |
| 14 | Evidence-backed UI and final integrated system | Every visible value has source provenance | SCAFFOLDED |

## Phase 1 exact scope

Phase 1 is deliberately limited to data integrity:

1. Keep 1.0.3 as the canonical target; accept the locally supplied 1.0.1 archive only as an explicitly versioned source until 1.0.3 is available.
2. Add a resumable acquisition/check script without committing raw records.
3. Validate required metadata and WFDB file pairs.
4. Correct manifest metadata such as sample counts and source paths.
5. audit multi-label records before freezing their resolution rule.
6. Generate versioned patient-level split manifests.
7. Report record, patient, label, lead, sampling-rate, and invalid-file counts.
8. Test determinism and leakage prevention.

Training does not begin in Phase 1.

## Commit convention

Use focused conventional commits:

- `docs(phase-0): freeze ML system and roadmap`
- `feat(phase-1): add reproducible PTB-XL data foundation`
- `feat(phase-2): freeze ECG preprocessing and quality pipeline`
- `feat(phase-3): validate centralized ECG baselines`

Tags identify validated phase releases, for example `phase-0-v1.0.0`. A phase is not tagged when its validation gate is failing.
