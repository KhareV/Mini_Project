# CENTRALIZED_SYSTEM_V1 release record

> Historical release record. `CENTRALIZED_SYSTEM_V2.md` supersedes its learned
> multimodal handoff after the BIDMC target-leakage audit.

This is the pre-federation, pre-hardware centralized release. Person 3 federation and sensor hardware are not included.

## Authoritative artifacts

| Role | Single authoritative artifact | SHA-256 |
| --- | --- | --- |
| Standalone ECG reference | `models/MODEL_V1.pt` | `2646ca77828200011996fe17137829e168c19e4bc5182fc1d1b2778ae4b01475` |
| Final centralized multimodal classifier | `continuous-health-monitor/experiments/results/multimodal_p1_integrated/checkpoint.pt` | `b3023c323bcde90b305ba23053cea7fd5334844e571e1eb204aac7ebe60bbb42` |
| Fusion calibration | `continuous-health-monitor/experiments/results/multimodal_p1_integrated/calibration.json` | `13281944d1aed3cfa211175cf57ec4246a87641e69d5c1edd03b09e45910c2a9` |

`models/candidates/C1_validation_selected_nonrelease.pt` is a **non-release optimization candidate**. It must not be used by serving, fusion, Person 3, or any release claim. Its validation-only selection experiment is retained at `experiments/ecg_model_selection/` for auditability.

## Results and decision

- Historical ECG evaluation at the conventional 0.50 cutoff: F1 0.8202, AUROC 0.9012. See `reports/canonical_ecg_test.json`.
- Validated ECG operating point: validation-only Platt calibration and locked 0.37 threshold, F1 0.8389, AUROC 0.9010 on 3,207 duplicate-cleaned PTB-XL test records. See `reports/MODEL_V1_validation_calibration.json`, `reports/canonical_ecg_calibrated_test.json`, and `reports/PTBXL_CROSS_SPLIT_DUPLICATE_AUDIT.json`.
- Canonical ECG, MIT-BIH external evaluation: F1 0.4630, AUROC 0.7204. See `reports/canonical_ecg_mitbih_external.json`.
- Production quality-aware fusion, held-out BIDMC: F1 0.7770, AUROC 0.8482. The ablation-only quality-aware full-fusion result is F1 0.7981; it is evidence, not the serving checkpoint.

Real duplicate-cleaned PTB-XL baselines are now available: majority F1 0.7195
(specificity 0), Logistic Regression F1 0.7875/AUROC 0.8573, and Random Forest
F1 0.8053/AUROC 0.8650. Five fixed-seed repetitions of the CNN recipe yielded
F1 0.8225 ± 0.0033 and AUROC 0.8973 ± 0.0049 at the conventional 0.5 cutoff.
See `reports/PTBXL_REAL_BASELINES_V2.json` and
`reports/ECG_SEED_STABILITY_V2.json`.

The BIDMC scores above are retained as historical integration results only.
Their label was derived from contemporaneous HR/SpO2 values that were also
model inputs, so they are not eligible evidence for clinical multimodal
performance. The non-circular future-alert audit also failed to generalize and
was rejected; see the P2 `BIDMC_FUTURE_ALERT_V1.md` record.

`MODEL_V1.pt` is the standalone ECG reference; the quality-aware fusion checkpoint is the final centralized full-input classifier. Their reported scores are from different datasets and task distributions (PTB-XL versus BIDMC), so they must **not** be used to claim that either architecture outperforms the other.

This original two-track handoff is superseded by `CENTRALIZED_SYSTEM_V2.md`:
federate the ECG reference only. The learned BIDMC fusion checkpoint is audit
history, not a validated FL target. The current contract is in
`MODEL_CONTRACT.md`.

## Locked software contract

- ECG: P1 canonical preprocessing; one lead; `float32`; exactly 2,500 samples (10 seconds at 250 Hz).
- PPG: P2 preprocessing; `float32`; exactly 1,250 samples (10 seconds at 125 Hz).
- Additional fusion values: mean HR, mean SpO2; presence and quality vectors ordered ECG, PPG, HR, SpO2.
- Fusion calibration: validation-only Platt calibration with locked threshold 0.28.
- Quality rule: ECG unavailable or any available modality quality below 0.50 returns `UNRELIABLE_SIGNAL`; it never becomes a normal or abnormal pattern output.
- Event rule: two consecutive abnormal windows open an event; two normal windows clear it.

## Honest robustness interpretation

The NST/MIT-BIH stress report demonstrates substantial failure under degraded ECG. At +24 dB, specificity is 42.25%; at -6 dB, specificity is 0.18%. Quality gating prevents low-quality input from being presented as a valid prediction; it does **not** establish robust physiological classification under severe motion/noise. Latency reported by the API is CPU-server process latency, not wearable/edge latency.

## Model-validity gate

`reports/MODEL_VALIDITY_CHECK_V1.json` records passed patient-disjointness,
record uniqueness, tiny-set overfit, shuffled-label, and MIT-BIH label-mapping
checks. The full audit found eight exact waveform duplicate groups crossing
PTB-XL partitions. No duplicate evaluation record is used for validation
calibration or final test reporting; the audit preserves the complete list.

## Evidence and repeatable commands

```sh
python3 scripts/evaluate_selected_ecg.py --checkpoint models/MODEL_V1.pt --no-copy --output reports/canonical_ecg_test.json
python3 scripts/evaluate_mitbih_external.py --checkpoint models/MODEL_V1.pt --output reports/canonical_ecg_mitbih_external.json
python3 scripts/evaluate_noise_robustness.py --checkpoint models/MODEL_V1.pt --output reports/canonical_ecg_noise_robustness.json
python3 scripts/replay_bidmc_stream.py --max-windows 100
python3 -m pytest -q
```
