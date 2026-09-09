# Validation and runbook

This is the current, honest way to verify the repository. Raw datasets and model checkpoints are local artifacts and are intentionally not committed.

## 1. Fast verification of all committed phases

From the repository root:

```bash
python3 scripts/verify_phases.py
```

This checks the PTB-XL inventory, real baseline provenance, calibrated CNN evidence, required phase documents, and then runs the full test suite.

Expected result at the current revision: **all checks pass and the full test suite passes** (the count increases as new tests are added).

Calibration primitives are available for evaluation code as `brier_score`, `expected_calibration_error`, and `reliability_bins` in `evaluation.calibration`. They require persisted validation/test probabilities; the existing JSON run artifacts do not retroactively contain per-example probabilities.

## 2. Focused phase checks

```bash
python3 -m pytest -q tests/test_ptbxl_data_foundation.py tests/test_synchronization.py
python3 -m pytest -q tests/test_multimodal_quality.py tests/test_bidmc.py tests/test_multimodal_fusion.py
python3 -m pytest -q tests/test_federated_partitions.py tests/test_fedavg.py tests/test_toy_fedavg.py
python3 -m pytest -q tests/test_hardware_protocol.py tests/test_hardware_framing.py tests/test_hardware_bridge.py
```

## 3. Re-run the real centralized reference stages

The supplied local archive is PTB-XL 1.0.1. It must be passed explicitly:

```bash
python3 scripts/prepare_ptbxl.py inventory \
  --data-dir ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 \
  --version 1.0.1 --sampling-rate 500

python3 training/train_classical.py --ptbxl \
  ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 \
  --dataset-version 1.0.1 --sampling-rate 500 --lead I
```

The real CNN run is CPU-intensive; use the same dataset/version/lead arguments and an explicit output directory. Do not call synthetic metrics research results.

## 4. Replay the trained model

```bash
python3 scripts/replay_ptbxl.py \
  --data-dir ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 \
  --checkpoint experiments/phase4_ptbxl_v1/E04_ecg_cnn/best_checkpoint.pt \
  --normalization experiments/phase4_ptbxl_v1/E04_ecg_cnn/normalization_stats.json \
  --packet-samples 125 --jitter-ms 4
```

## 5. Backend endpoint

Configure artifacts before starting the backend:

```bash
export NHM_ECG_MODEL_CHECKPOINT="$PWD/experiments/phase4_ptbxl_v1/E04_ecg_cnn/best_checkpoint.pt"
export NHM_ECG_NORMALIZATION_STATS="$PWD/experiments/phase4_ptbxl_v1/E04_ecg_cnn/normalization_stats.json"
uvicorn backend.main:app --reload --port 8000
```

Then POST a JSON body to `http://localhost:8000/replay/ecg` with `samples` and `sampling_rate`. Without both artifact variables, the endpoint intentionally returns `503` rather than making up a prediction.

## 6. What is not yet runnable as a completed system

BIDMC acquisition, external MIT-BIH/NSTDB evaluation, centralized multimodal training, Flower runtime federated learning, secure aggregation/DP, live BLE/serial acquisition, edge export, and production frontend provenance are not complete. Hardware protocol code is only a tested preparation boundary and is deliberately scheduled last.
