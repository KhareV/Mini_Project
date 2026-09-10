"""Window buffering and deterministic event policy for centralized replay/live software.

This is intentionally a software-only boundary: chunks must already meet the
locked preprocessing contract. Hardware acquisition is outside this service.
"""
from collections import defaultdict, deque
from typing import Any, Dict, Iterable

from backend.services.multimodal_inference import multimodal_inference_service


class StreamingInferenceService:
    def __init__(self, window_count: int = 2):
        self.window_count = window_count
        self.states: Dict[str, Dict[str, Any]] = defaultdict(self._new_state)

    @staticmethod
    def _new_state() -> Dict[str, Any]:
        return {"abnormal_streak": 0, "normal_streak": 0, "event_state": "CLEAR", "history": deque(maxlen=100)}

    def reset(self, session_id: str) -> None:
        self.states[session_id] = self._new_state()

    def ingest(self, session_id: str, windows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        state = self.states[session_id]
        decisions = []
        for window in windows:
            result = multimodal_inference_service.predict(**window)
            prediction = result["prediction"]
            transition = None
            if prediction == "UNRELIABLE_SIGNAL":
                state["abnormal_streak"] = state["normal_streak"] = 0
                # Quality is surfaced immediately, without manufacturing an event.
                event_state = "UNRELIABLE"
            elif prediction == "POTENTIALLY_ABNORMAL_PATTERN":
                state["abnormal_streak"] += 1
                state["normal_streak"] = 0
                if state["abnormal_streak"] >= self.window_count and state["event_state"] != "ACTIVE":
                    state["event_state"] = "ACTIVE"
                    transition = "EVENT_OPENED"
                event_state = state["event_state"]
            else:
                state["normal_streak"] += 1
                state["abnormal_streak"] = 0
                if state["event_state"] == "ACTIVE" and state["normal_streak"] >= self.window_count:
                    state["event_state"] = "CLEAR"
                    transition = "EVENT_CLEARED"
                event_state = state["event_state"]
            decision = {**result, "event_state": event_state, "transition": transition}
            state["history"].append(decision)
            decisions.append(decision)
        return {"session_id": session_id, "decisions": decisions, "event_state": state["event_state"], "policy": "two consecutive abnormal windows open an event; two normal windows clear it"}


streaming_inference_service = StreamingInferenceService()
