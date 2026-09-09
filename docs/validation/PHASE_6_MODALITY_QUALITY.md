# Phase 6 — PPG/SpO2 quality contracts

The first Phase 6 increment replaces fixed placeholder outputs with explicit software contracts:

- PPG quality records score, label, usability, and reasons for missing/non-finite, flatline, and implausible pulse intervals.
- SpO2 accepts only physiologically plausible values (70–100%), reports confidence and trend, and returns `UNRELIABLE` when no usable value exists.
- Missing or degraded modalities remain represented in the synchronization presence mask; they are not silently imputed.

These are modality-level contracts and synthetic/unit validation only. BIDMC ingestion, real PPG/SpO2 validation, and late fusion remain pending.
