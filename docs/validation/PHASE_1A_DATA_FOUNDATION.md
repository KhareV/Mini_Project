# Phase 1A Validation — PTB-XL Data Foundation

Date: 2026-09-09
Dataset: PTB-XL 1.0.3
Resolution selected for the primary experiment: 500 Hz
Primary lead: Lead I
Status: `VALIDATED_METADATA`; waveform completion remains pending

## Implemented

- Versioned PTB-XL 1.0.3 configuration and canonical path.
- Resumable, retrying acquisition utility with metadata-only support.
- Dataset inventory with metadata hashes and exact missing-file counts.
- Safe `scp_codes` parsing.
- Full diagnostic-superclass preservation.
- Explicit exclusion of normal-plus-abnormal conflicts.
- Actual WFDB header-derived sample counts and lead names.
- `.hea` and `.dat` pair validation.
- Deterministic participant-level splits with leakage assertions.
- Split SHA-256 sidecar metadata.
- Focused tests for label policy, resolution selection, completeness, deterministic splitting, and invalid split fractions.

## Real metadata audit

| Measure | Result |
|---|---:|
| Metadata records | 21,799 |
| Unique patients | 18,869 |
| Resolved normal records | 9,069 |
| Resolved abnormal records | 11,874 |
| Normal/abnormal conflicts excluded | 445 |
| Unsupported/no diagnostic superclass excluded | 411 |
| Records eligible before waveform validation | 20,943 |

The frozen Phase 1 label policy is:

- normal-only diagnostic evidence -> `NORMAL`;
- one or more abnormal superclasses without `NORM` -> `ABNORMAL`;
- simultaneous normal and abnormal superclasses -> excluded as `CONFLICT`;
- no supported diagnostic superclass -> excluded as `UNSUPPORTED`.

This policy avoids silently selecting one superclass from a multi-label record.

## Metadata provenance

| File | SHA-256 |
|---|---|
| `ptbxl_database.csv` | `7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b` |
| `scp_statements.csv` | `ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f` |
| `RECORDS` | `56d37274b0b02339e9c30bff8c9a9f2a6fb3cb2bb7bcd5d7c55355451d53896a` |
| `LICENSE.txt` | `9a78e7f22742dde9f66ae235ec793ba2212019dc4d0ced75c4da09ced0b35fb2` |

## Remaining Phase 1 gate

The waveform inventory currently requires 43,598 files: one `.hea` and one `.dat` file for each of 21,799 500 Hz records. No waveform file was present at the time of this validation.

Phase 1 must not be marked complete until:

1. the resumable waveform acquisition finishes;
2. inventory reports zero missing files;
3. every WFDB header is readable and matches the expected lead/rate contract;
4. signal validation and invalid-reason reports complete;
5. versioned manifest and split files are generated twice with identical hashes;
6. zero participant leakage is confirmed.

## Reproduction commands

```bash
python3 scripts/prepare_ptbxl.py download --metadata-only
python3 scripts/prepare_ptbxl.py labels
python3 scripts/prepare_ptbxl.py inventory
python3 -m pytest tests/test_ptbxl_data_foundation.py -q
```

After waveform acquisition:

```bash
python3 scripts/prepare_ptbxl.py download --workers 4
python3 scripts/prepare_ptbxl.py build
```
