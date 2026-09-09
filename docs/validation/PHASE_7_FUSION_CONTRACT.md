# Phase 7 — centralized multimodal fusion contract

`models.multimodal_fusion.MultimodalFusionModel` defines the planned late-fusion interface: ECG encoder, PPG encoder, HR/SpO2 scalar branch, and an explicit four-field presence/quality mask. Missing ECG/PPG/scalar inputs are represented by masks and zeroed branch embeddings rather than silently treated as measured values.

This is an architecture and shape contract only. Training, ablations, calibration, and performance claims require the real BIDMC synchronized archive and remain pending.
