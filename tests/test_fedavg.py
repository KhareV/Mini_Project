import torch

from federated.fedavg import weighted_average_state_dicts


def test_fedavg_is_sample_weighted():
    result = weighted_average_state_dicts([
        ({"weight": torch.tensor([0.0]), "step": torch.tensor([1])}, 1),
        ({"weight": torch.tensor([1.0]), "step": torch.tensor([2])}, 3),
    ])
    assert result["weight"].item() == 0.75
    assert result["step"].item() == 1  # non-floating metadata is preserved


def test_fedavg_rejects_mismatched_models():
    try:
        weighted_average_state_dicts([({"a": torch.ones(1)}, 1), ({"b": torch.ones(1)}, 1)])
    except ValueError as exc:
        assert "keys" in str(exc)
    else:
        raise AssertionError("mismatched state keys must be rejected")
