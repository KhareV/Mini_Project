# Phase 1B — Supplied PTB-XL archive validation

Date: 2026-09-09  
Source: local PTB-XL archive `1.0.1` (not committed)  
Sampling rate: 500 Hz  
Primary lead: I  
Seed: 42

## Gate result

**PASS.** The supplied archive is complete and usable for the centralized reference stage.

- 21,837 metadata records and 18,885 patients.
- 43,674 required WFDB files present (21,837 headers + 21,837 signal files).
- 20,985 records passed header and signal-level validation.
- 852 records excluded by the frozen label policy: 445 normal/abnormal conflicts and 407 unsupported labels.
- No missing file pairs, NaN/Inf signals, flatlines, or unsupported sampling rates were observed.
- Patient split: 14,649 train, 3,160 validation, 3,176 test records; no patient crosses splits.

Artifacts:

- `data/reports/ptbxl_1.0.1_inventory.json`
- `data/reports/ptbxl_1.0.1_labels.json`
- `data/reports/ptbxl_1.0.1_validation.json`
- `data/manifests/ptbxl_1.0.1_manifest.csv`
- `data/splits/ptbxl_1.0.1_splits_seed42.csv`

The archive is kept outside `data/raw` and ignored by Git. Its version is carried through every manifest and run record; these results must not be described as PTB-XL 1.0.3 results.
