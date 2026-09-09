import numpy as np

from inference.ecg_model_v1 import ECGModelV1
from models.ecg_cnn import ECGCNN1D, ECGCNNConfig
from preprocessing.ecg import NormalizationStats


def test_model_v1_replay_is_deterministic(tmp_path):
    model = ECGCNN1D(ECGCNNConfig(input_length=2500, model_version="MODEL_V1",
                                  preprocessing_version="1.1.0"))
    checkpoint = tmp_path / "model.pt"
    stats_path = tmp_path / "stats.json"
    model.save_checkpoint(str(checkpoint), epoch=1, metrics={})
    NormalizationStats(mean=0.0, std=1.0, n_samples_used=10,
                       preprocessing_version="1.1.0").save(stats_path)

    runner = ECGModelV1.load(str(checkpoint), str(stats_path), threshold=0.55)
    signal = np.sin(np.linspace(0, 80, 5000)).astype(np.float32)
    first = runner.predict(signal, source_fs=500)
    second = runner.predict(signal, source_fs=500)
    assert first.to_dict() == second.to_dict()
    assert 0.0 <= first.probability_abnormal <= 1.0
    assert first.preprocessing_version == "1.1.0"
