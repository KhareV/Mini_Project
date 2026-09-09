import numpy as np

from preprocessing.multimodal_quality import assess_ppg, assess_spo2


def test_ppg_quality_rejects_flatline_and_accepts_pulsatile_signal():
    poor = assess_ppg(np.ones(100), np.ones(100))
    assert poor.usable is False
    good = assess_ppg(np.sin(np.linspace(0, 20, 100)), np.sin(np.linspace(0, 20, 100)),
                      np.full(4, 1.0))
    assert good.usable is True


def test_spo2_quality_handles_missing_and_invalid_values():
    assert assess_spo2(np.array([])).state == "UNRELIABLE"
    result = assess_spo2(np.array([98.0, 97.0, 99.0, 98.0, np.nan]))
    assert result.usable is True
    assert result.value == 98.0
    assert result.confidence == 0.8
