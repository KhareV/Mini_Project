# Phase 5 — Offline `MODEL_V1` replay boundary

The `inference.ECGModelV1` wrapper is the first integration seam between the trained model and later backend/device adapters. It loads the model checkpoint and its matching training-derived normalization artifact, rejects preprocessing-version mismatches, applies the canonical 250 Hz pipeline, pads/truncates to the model window, and returns a structured prediction with model and preprocessing provenance.

The replay path is deterministic for the same checkpoint, normalization artifact, signal, and source sampling rate. Hardware and streaming adapters are intentionally not included yet.
