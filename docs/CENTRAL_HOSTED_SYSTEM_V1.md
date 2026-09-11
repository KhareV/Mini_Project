# CENTRAL_HOSTED_SYSTEM_V1

The deployed, evidence-backed software baseline is the canonical ECG model.
The quality-aware multimodal endpoint remains an integration/research service,
not a clinically validated classifier: BIDMC's original target was circular,
and the corrected future-alert experiment did not generalize.

## API boundary

`POST /model/ecg/infer` is the deployment-eligible software path. It accepts a
canonical 2,500-sample ECG window plus a normalized quality value, applies the
locked quality gate, MODEL_V1 checkpoint, validation-only calibration, and
0.37 threshold, and returns a monitored-pattern decision.

`POST /model/infer` accepts one already canonical 10-second window:

- ECG: 2,500 `float32` samples at 250 Hz, using the P1 pipeline.
- PPG: 1,250 `float32` samples at 125 Hz, using the P2 pipeline.
- HR, SpO2, presence mask, and quality scores in their locked order.

The endpoint validates schema/lengths, applies the quality gate, extracts
features, runs the locked fusion checkpoint, applies its validation-only
calibration and threshold, and returns an auditable **research** decision.
The multimodal route is research-only. `POST /model/stream/{session_id}` adds ordered-window event handling and
persists decisions. `GET /model/status` exposes artifact readiness.

## Explicit limitation

This is a hosted *canonical-window* API. It deliberately does not accept raw
AD8232 or MAX30102 payloads yet: wearable sampling, calibration, and
validation remain hardware-phase work. Future hardware ingestion must convert
raw signals through the same canonical preprocessing contracts before calling
this API.

## Release eligibility

`MODEL_V1.pt` is the only current deployment-eligible predictive model. The
multimodal checkpoint may be used to verify data flow, API behaviour, quality
gating, and later hardware integration, but must not be described as a
validated abnormality detector until a non-circular labelled multimodal task
passes held-out and external validation.
