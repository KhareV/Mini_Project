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
| Centralized ECG CNN | Internally validated on PTB-XL 1.0.1; MIT-BIH external validation complete, robustness pending | `docs/validation/PHASE_4_REAL_PTBXL.md`, `docs/validation/MITBIH_MODEL_V1_EXTERNAL.md` |
| Offline/stream/backend replay | Implemented and tested | `docs/validation/PHASE_5_OFFLINE_REPLAY.md` |
| Hardware | Protocol/decoder preparation only; no device integration | `docs/validation/HARDWARE_PROTOCOL.md` |

The supplied dataset is PTB-XL 1.0.1. PTB-XL 1.0.3 remains the canonical target in the frozen specification; results are always labeled 1.0.1 and are never silently called 1.0.3.

## What remains, in order

1. MIT-BIH/NSTDB noise robustness using locked MODEL_V1; investigate the observed cross-dataset false-positive shift without test retuning.
2. Complete Phase 6: PPG/SpO2, BIDMC acquisition, synchronization, and quality validation.
3. Train/evaluate centralized multimodal fusion and required ablations.
4. Complete streaming/event metrics and centralized hosted software integration.
5. Freeze the centralized software system.
6. Phase 12: hardware capture, serial, then BLE integration. This is deliberately last among system-building work.
7. Phase 13: separately versioned edge model and measured device constraints.
8. Only after hardware validation: real FedAvg, non-IID, FedProx, secure aggregation, and optional DP.
9. Phase 14: evidence-backed UI integration and final system evidence.

## Explicit non-claims

No live BLE/serial acquisition, firmware, edge deployment, real federated training, secure aggregation, clinical diagnosis, or production-grade PPG/SpO2 system is complete. Synthetic runs are software smoke tests only.

Every implementation iteration must add tests, produce a validation note, commit a focused change, and push a version tag when its gate passes.
