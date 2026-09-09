# NHM project status and reconciled execution plan

Last reconciled: 2026-09-09  
Owner: one maintainer  
Truth boundary: research prototype; not a medical device.

## What is actually complete

| Area | Status | Evidence |
|---|---|---|
| System/data/label contracts | Frozen for current phases | `docs/ML_SYSTEM_SPEC.md`, `configs/base.yaml` |
| PTB-XL foundation | Validated on supplied local 1.0.1 archive | `docs/validation/PHASE_1B_REAL_PTBXL.md` |
| ECG preprocessing/SQI | Validated on fixtures; strict train-only stats enforced | `docs/validation/PHASE_2_PREPROCESSING.md` |
| Centralized classical baselines | Validated on PTB-XL 1.0.1 | `docs/validation/PHASE_3B_REAL_PTBXL.md` |
| Centralized ECG CNN | Internally validated on PTB-XL 1.0.1; external validation pending | `docs/validation/PHASE_4_REAL_PTBXL.md` |
| Offline/stream/backend replay | Implemented and tested | `docs/validation/PHASE_5_OFFLINE_REPLAY.md` |
| Hardware | Protocol/decoder preparation only; no device integration | `docs/validation/HARDWARE_PROTOCOL.md` |

The supplied dataset is PTB-XL 1.0.1. PTB-XL 1.0.3 remains the canonical target in the frozen specification; results are always labeled 1.0.1 and are never silently called 1.0.3.

## What remains, in order

1. Phase 6: PPG/SpO2 contracts, BIDMC acquisition, timestamp synchronization, quality and missing-modality tests.
2. Phase 7: centralized late-fusion model and ablations.
3. Phase 8: deterministic participant-based IID/non-IID federated manifests.
4. Phase 9: actual Flower FedAvg against the centralized reference.
5. Phase 10: FedProx/QAPFL controlled comparisons.
6. Phase 11: secure aggregation and optional differential privacy.
7. External MIT-BIH/NSTDB validation and robustness/calibration reporting integrated at the appropriate gates.
8. Phase 12: hardware capture, serial, then BLE integration. This is deliberately last among system-building work.
9. Phase 13: separately versioned edge model and measured device constraints.
10. Phase 14: evidence-backed UI integration.

## Explicit non-claims

No live BLE/serial acquisition, firmware, edge deployment, real federated training, secure aggregation, clinical diagnosis, or production-grade PPG/SpO2 system is complete. Synthetic runs are software smoke tests only.

Every implementation iteration must add tests, produce a validation note, commit a focused change, and push a version tag when its gate passes.
