import numpy as np

from evaluation.calibration import brier_score, expected_calibration_error, reliability_bins


def test_perfect_probabilities_are_well_calibrated():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert brier_score(y, p) == 0.0
    assert expected_calibration_error(y, p) == 0.0
    assert sum(item["count"] for item in reliability_bins(y, p)) == 4


def test_calibration_metrics_reject_invalid_probability():
    try:
        brier_score(np.array([0, 1]), np.array([0.2, 1.2]))
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range probabilities must be rejected")
