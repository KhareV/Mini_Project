from backend.services.streaming_inference import StreamingInferenceService


def test_event_requires_two_abnormal_windows_and_recovers_after_two_normal(monkeypatch):
    predictions = iter([
        {"prediction": "POTENTIALLY_ABNORMAL_PATTERN", "model_confidence": .9},
        {"prediction": "POTENTIALLY_ABNORMAL_PATTERN", "model_confidence": .9},
        {"prediction": "NORMAL_MONITORED_PATTERN", "model_confidence": .1},
        {"prediction": "NORMAL_MONITORED_PATTERN", "model_confidence": .1},
    ])
    monkeypatch.setattr("backend.services.streaming_inference.multimodal_inference_service.predict", lambda **_: next(predictions))
    service = StreamingInferenceService()
    output = service.ingest("patient-a", [{}, {}, {}, {}])
    assert [item["transition"] for item in output["decisions"]] == [None, "EVENT_OPENED", None, "EVENT_CLEARED"]
    assert output["event_state"] == "CLEAR"


def test_unreliable_is_visible_without_opening_event(monkeypatch):
    monkeypatch.setattr("backend.services.streaming_inference.multimodal_inference_service.predict", lambda **_: {"prediction": "UNRELIABLE_SIGNAL", "model_confidence": None})
    output = StreamingInferenceService().ingest("patient-b", [{}])
    assert output["decisions"][0]["event_state"] == "UNRELIABLE"
    assert output["decisions"][0]["transition"] is None
