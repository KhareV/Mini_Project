# Phase 6 — BIDMC ingestion contract

`datasets.bidmc.BIDMCDataset` now provides deterministic WFDB record discovery, stable channel aliases (`ecg`, `ppg`, `resp`, `spo2`), and a loader that returns timestamp-free signal/rate pairs ready for the synchronization contract. The archive is not currently present in the repository, so this phase increment is a loader contract and unit validation—not a BIDMC result.

Once BIDMC 1.0.0 is acquired, the acceptance run must verify channel availability, source rates, timestamp alignment, missing-modality masks, and quality-state distributions before training any fusion model.
