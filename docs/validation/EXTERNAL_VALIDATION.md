# External validation contract

`scripts/evaluate_mitbih.py` evaluates the locked PTB-XL-trained `MODEL_V1` on MIT-BIH. It requires the threshold selected on PTB-XL validation, never fits or tunes on MIT-BIH, and reports the same classification metrics with `split=external`.

Example:

```bash
python3 scripts/evaluate_mitbih.py \
  --data-dir data/raw/mitbih \
  --checkpoint experiments/phase4_ptbxl_v1/E04_ecg_cnn/best_checkpoint.pt \
  --normalization experiments/phase4_ptbxl_v1/E04_ecg_cnn/normalization_stats.json \
  --threshold 0.554572 \
  --output data/reports/mitbih_model_v1_external.json
```

The report is not generated until the local MIT-BIH files and locked checkpoint are available. It is an external validation result, not a retraining step.
