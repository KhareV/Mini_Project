import numpy as np

from preprocessing.synchronization import synchronize_modalities


def test_modalities_align_without_extrapolation():
    result = synchronize_modalities({
        "ecg": (np.arange(0, 2.01, 0.5), np.arange(5.0)),
        "ppg": (np.arange(0.5, 1.51, 0.5), np.ones(3)),
    }, target_fs=2)
    assert len(result.timestamps) == 5
    assert result.presence["ppg"].tolist() == [False, True, True, True, False]
    assert np.isnan(result.values["ppg"][0])


def test_non_monotonic_timestamps_are_rejected():
    try:
        synchronize_modalities({"ecg": (np.array([0.0, 0.0]), np.ones(2))}, 10)
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("non-monotonic timestamps must be rejected")
