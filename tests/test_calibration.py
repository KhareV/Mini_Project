import numpy as np

from evaluation.calibration import select_f1_threshold


def test_threshold_is_selected_from_validation_scores():
    threshold, f1 = select_f1_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])
    )
    assert 0.4 < threshold <= 0.6
    assert f1 == 1.0


def test_threshold_rejects_invalid_vectors():
    try:
        select_f1_threshold(np.array([0]), np.array([np.nan]))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid scores must be rejected")
