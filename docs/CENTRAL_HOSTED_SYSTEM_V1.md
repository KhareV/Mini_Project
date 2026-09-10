# CENTRAL_HOSTED_SYSTEM_V1

The production software baseline is one centrally hosted, quality-aware
multimodal classifier—not separate ECG, PPG, HR, and SpO2 diagnosis models.

## API boundary

`POST /model/infer` accepts one already canonical 10-second window:

- ECG: 2,500 `float32` samples at 250 Hz, using the P1 pipeline.
- PPG: 1,250 `float32` samples at 125 Hz, using the P2 pipeline.
- HR, SpO2, presence mask, and quality scores in their locked order.

The endpoint validates schema/lengths, applies the quality gate, extracts
features, runs the locked fusion checkpoint, applies its validation-only
calibration and threshold, and returns an auditable research decision.
`POST /model/stream/{session_id}` adds ordered-window event handling and
persists decisions. `GET /model/status` exposes artifact readiness.

## Explicit limitation

This is a hosted *canonical-window* API. It deliberately does not accept raw
AD8232 or MAX30102 payloads yet: wearable sampling, calibration, and
validation remain hardware-phase work. Future hardware ingestion must convert
raw signals through the same canonical preprocessing contracts before calling
this API.
