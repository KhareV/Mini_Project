# Phase 5 — Offline `MODEL_V1` replay boundary

The `inference.ECGModelV1` wrapper is the first integration seam between the trained model and later backend/device adapters. It loads the model checkpoint and its matching training-derived normalization artifact, rejects preprocessing-version mismatches, applies the canonical 250 Hz pipeline, pads/truncates to the model window, and returns a structured prediction with model and preprocessing provenance.

The replay path is deterministic for the same checkpoint, normalization artifact, signal, and source sampling rate. The `ECGStreamReplayer` adapter emits only complete 10-second windows at a 5-second stride, accepts arbitrary packet boundaries, rejects non-finite samples, and produces the same windows for chunked and contiguous ingestion. Hardware transport is intentionally not included yet.

`ECGPacketReplayer` adds sequence-aware transport replay. A missing or out-of-order packet increments the gap counters and resets the window buffer, so unknown samples are never silently bridged. Packet timing/jitter can therefore be tested independently from BLE or serial implementation.
