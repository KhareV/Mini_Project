from federated.toy import run_toy_fedavg


def test_toy_fedavg_converges_to_sample_weighted_target():
    result = run_toy_fedavg([1.0, 3.0], [1, 3], rounds=20, local_steps=5)
    assert abs(result.global_values[-1] - result.target) < 1e-3
    assert abs(result.global_values[-1] - result.target) < abs(result.global_values[0] - result.target)


def test_toy_fedavg_rejects_invalid_clients():
    try:
        run_toy_fedavg([], [])
    except ValueError:
        pass
    else:
        raise AssertionError("empty client set must be rejected")
