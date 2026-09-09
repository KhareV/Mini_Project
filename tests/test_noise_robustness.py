import numpy as np

from evaluation.noise_robustness import add_awgn_at_snr, evaluate_snr_curve


def test_awgn_is_seeded_and_changes_with_snr():
    signal = np.ones(1000, dtype=np.float32)
    first = add_awgn_at_snr(signal, 6, seed=3)
    second = add_awgn_at_snr(signal, 6, seed=3)
    assert np.array_equal(first, second)
    assert np.std(first - signal) > np.std(add_awgn_at_snr(signal, 24, seed=3) - signal)


def test_snr_curve_does_not_fit_model():
    class FakeModel:
        def predict(self, signal, source_fs):
            from inference.ecg_model_v1 import ECGPrediction
            return ECGPrediction("NORMAL", float(np.mean(signal) > 0), 0.5, "TEST", "1.1.0")
    curve = evaluate_snr_curve(FakeModel(), [np.ones(20)], np.array([0]), snr_levels=(12, 0))
    assert [item["snr_db"] for item in curve] == [12.0, 0.0]
    assert all(item["n_samples"] == 1 for item in curve)
