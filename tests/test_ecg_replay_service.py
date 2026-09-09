import numpy as np

from backend.ml.ecg_replay_service import ECGReplayService
from inference.ecg_model_v1 import ECGPrediction
from inference.packet_replay import ECGPacket


class FakeRunner:
    model_version = "MODEL_V1"

    class _Preprocessor:
        config_dict = {"preprocessing_version": "1.1.0"}

    preprocessor = _Preprocessor()

    def predict(self, samples, source_fs):
        return ECGPrediction("NORMAL", 0.2, 0.55, "MODEL_V1", "1.1.0")


def test_backend_service_preserves_prediction_provenance():
    service = ECGReplayService(FakeRunner(), source_fs=10)
    result = service.predict_window(np.ones(20), source_fs=10)
    assert result["model_version"] == "MODEL_V1"
    assert result["preprocessing_version"] == "1.1.0"
    assert result["threshold"] == 0.55


def test_backend_service_replays_packet_gaps():
    service = ECGReplayService(FakeRunner(), source_fs=10)
    result = service.replay_packets([
        ECGPacket(0, np.ones(5)), ECGPacket(2, np.ones(5))
    ])
    assert result["packets_dropped"] == 1
    assert result["discontinuities"] == 1
    assert result["preprocessing_version"] == "1.1.0"
