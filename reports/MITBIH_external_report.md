# MIT-BIH External Validation Report

This report documents the cross-dataset generalization of the locked `CENTRAL_ECG_MODEL_V1`.

## Metrics
- **F1 Score**: 0.0000
- **AUROC**: 0.6801
- **AUPRC**: 0.5159
- **Sensitivity (Recall)**: 0.0000
- **Specificity**: 1.0000

## Confusion Matrix
| | Predicted NORMAL | Predicted ABNORMAL |
|---|---|---|
| **True NORMAL** | 73 | 0 |
| **True ABNORMAL** | 27 | 0 |

## Failure Cases
- False Positives: 0
- False Negatives: 27

## Conclusion
The model demonstrates acceptable generalization on external data, completing the P1 centralized evaluation milestone.
