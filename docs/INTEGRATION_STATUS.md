# Integration Status — Pre-Federation Gate Complete

## Verified commands

Run from the repository root:

```bash
python3 -m pytest -q
```

Run the deterministic synthetic end-to-end multimodal smoke test:

```bash
cd continuous-health-monitor
python3 scripts/run_demo.py --participants 12 --epochs 1 --seed 42
```

Run the Svelte checks and production build:

```bash
cd frontend
npm run check
npm run build
```

## Contract decisions now enforced

- Test discovery includes Person 1/root and Person 2/nested tests.
- The demo partitions by `participant_id` before every baseline, fusion,
  ablation, and missing-modality experiment. The split manifest records seed,
  patients, and sample counts.
- Person 1's classifier exposes `extract_features()`; its binary logits are
  never passed to multimodal fusion as an embedding.
- `Person1ECGEncoderAdapter` resamples a 10-second 125 Hz Person 2 ECG window
  to Person 1's 250 Hz MODEL_V1 input length and freezes the backbone by
  default. Its projection layer is a new fusion parameter, so any model using
  it must be retrained and evaluated as a new version.
- Fusion checkpoints include their architecture metadata, and wearable
  validation reconstructs the exact checkpoint architecture instead of using
  hard-coded dimensions.

## Completed pre-federation artifacts

- PTB-XL: 21,430 valid labelled records; deterministic patient split persisted
  at `data/splits/ptbxl_splits.csv` (15,039 train / 3,181 validation / 3,210 test).
- Person 1 `MODEL_V1`: `models/MODEL_V1.pt`, SHA-256
  `2646ca77828200011996fe17137829e168c19e4bc5182fc1d1b2778ae4b01475`.
  Its held-out PTB-XL F1 is 0.8202 and AUROC is 0.9012; config and report are
  in `experiments/E04_ecg_cnn/`.
- BIDMC: 53 recordings with patient-disjoint 39 / 7 / 7 train/validation/test
  partitions and 3,705 / 665 / 665 windows.
- Multimodal lock: `multimodal_p1_integrated`, with frozen Person 1 adapter and
  canonical Person 1 ECG preprocessing applied to BIDMC before fusion. Its
  held-out BIDMC F1 is 0.7770 and AUROC is 0.8482. Checkpoint, run manifest,
  and report reside in `continuous-health-monitor/experiments/results/multimodal_p1_integrated/`.
- Backend bridge: `GET /model/status` and `POST /model/infer`; the request
  contract is 2,500 canonical ECG samples plus 1,250 preprocessed PPG samples.
  The frontend API client exposes these through `api.model`.

## Gate for Person 3 federation work

Do not start a research federated run until all items below are present:

All four pre-federation requirements are now met: locked ECG model, canonical
ECG preprocessing for BIDMC, retrained adapter-backed fusion checkpoint, and a
tested backend request/response contract. The current backend simulator remains
separate; `/model/infer` is the explicit research-model path.

When those four items are met, Person 3 can use the locked multimodal model's
`state_dict`, deterministic client manifests, and the common evaluation suite
for FedAvg first, then matched non-IID/FedProx experiments.

The prepared integration command is:

```bash
cd continuous-health-monitor
python3 scripts/train_multimodal.py \
  --ecg_checkpoint ../models/MODEL_V1.pt \
  --train_file data/processed/bidmc_train_windows.json \
  --val_file data/processed/bidmc_val_windows.json \
  --test_file data/processed/bidmc_test_windows.json
```
