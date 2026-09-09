# Phase 9 — controlled FedAvg baseline

`federated.fedavg.weighted_average_state_dicts` implements sample-count-weighted aggregation with strict state-key validation and explicit handling of non-floating metadata tensors. This is an in-process mathematical baseline only; it does not claim Flower client/server execution, privacy, convergence on ECG, or communication robustness.

`federated.toy.run_toy_fedavg` provides a deterministic scalar-client convergence check with identical local budgets and sample-weighted global target. It validates the round mechanics only; it is not an ECG experiment.
