# Phase 8 — deterministic federated client partitions

`federated.partitions.partition_manifest` assigns complete participants—not arbitrary windows—to deterministic clients. It supports an IID shuffle and a controlled dominant-label-skew mode, validates that every participant is assigned exactly once, and exposes a stable assignment digest.

This is a partition contract only. No federated model updates, Flower server, privacy mechanism, or convergence claim is included.
