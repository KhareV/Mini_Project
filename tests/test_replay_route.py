from fastapi import HTTPException

from backend.api.routes import replay
from backend.ml.ecg_replay_service import ECGReplayService
from inference.ecg_model_v1 import ECGPrediction


class FakeRunner:
    model_version = "MODEL_V1"

    class _Preprocessor:
        config_dict = {"preprocessing_version": "1.1.0"}

    preprocessor = _Preprocessor()

    def predict(self, samples, source_fs):
        return ECGPrediction("NORMAL", 0.2, 0.55, "MODEL_V1", "1.1.0")


def test_replay_route_returns_model_provenance(monkeypatch):
    monkeypatch.setattr(replay, "_service", lambda: ECGReplayService(FakeRunner(), 500))
    result = replay.replay_ecg(replay.ECGReplayRequest(samples=[0.0, 0.1]))
    assert result["model_version"] == "MODEL_V1"
    assert result["preprocessing_version"] == "1.1.0"


def test_replay_route_reports_unconfigured_service(monkeypatch):
    def unavailable():
        raise RuntimeError("not configured")
    monkeypatch.setattr(replay, "_service", unavailable)
    try:
        replay.replay_ecg(replay.ECGReplayRequest(samples=[0.0]))
    except HTTPException as exc:
        assert exc.status_code == 503
    else:
        raise AssertionError("unconfigured model must return 503")
