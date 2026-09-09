# MIT-BIH external validation — locked `MODEL_V1`

Date: 2026-09-09  
Dataset: MIT-BIH Arrhythmia 1.0.0 (local archive)  
Model: PTB-XL 1.0.1 `MODEL_V1`  
Threshold: `0.554572` selected on PTB-XL validation only  
Windows: 16,559

## Result

| Metric | External result |
|---|---:|
| AUROC | 0.7325 |
| AUPRC | 0.5048 |
| Accuracy | 0.2722 |
| F1 | 0.4001 |
| Sensitivity | 1.0000 |
| Specificity | 0.0391 |

The model predicted every abnormal window correctly but generated many false positives under this cross-dataset shift. This is an external validation finding, not a deployment claim and not a reason to retune the threshold on MIT-BIH.

Two records did not expose the requested `MLII` channel and the loader used their first available lead (`V5`); this is recorded as a lead-compatibility limitation. Lead mapping must be made explicit in the next robustness report.
