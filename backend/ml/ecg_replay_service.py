"""Backend adapter for the versioned offline/stream ECG replay boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from inference.ecg_model_v1 import ECGModelV1
from inference.packet_replay import ECGPacket, ECGPacketReplayer


class ECGReplayService:
    """Small dependency-injection friendly service for future API routes."""

    def __init__(self, runner: ECGModelV1, source_fs: int = 500):
        self.runner = runner
        self.source_fs = int(source_fs)

    @classmethod
    def from_artifacts(cls, checkpoint: str | Path, normalization: str | Path,
                       threshold: float = 0.5, source_fs: int = 500,
                       device: str = "cpu") -> "ECGReplayService":
        runner = ECGModelV1.load(str(checkpoint), str(normalization), threshold, device)
        return cls(runner, source_fs=source_fs)

    def predict_window(self, samples: Iterable[float], source_fs: int | None = None) -> dict:
        values = np.asarray(list(samples), dtype=np.float32)
        prediction = self.runner.predict(values, source_fs or self.source_fs)
        return prediction.to_dict()

    def replay_packets(self, packets: Iterable[ECGPacket]) -> dict:
        replayer = ECGPacketReplayer(self.runner, self.source_fs)
        predictions = []
        for packet in packets:
            predictions.extend(replayer.ingest_packet(packet))
        return {
            "predictions": [prediction.to_dict() for prediction in predictions],
            "packets_dropped": replayer.dropped_packets,
            "discontinuities": replayer.discontinuities,
            "model_version": self.runner.model_version,
            "preprocessing_version": self.runner.preprocessor.config_dict["preprocessing_version"],
        }
