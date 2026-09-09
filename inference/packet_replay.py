"""Transport-neutral packet replay with explicit gap handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .ecg_model_v1 import ECGModelV1, ECGPrediction
from .stream_replay import ECGStreamReplayer


@dataclass(frozen=True)
class ECGPacket:
    sequence: int
    samples: np.ndarray


class ECGPacketReplayer:
    """Feed ordered packets while refusing to bridge missing samples."""

    def __init__(self, model: ECGModelV1, source_fs: int,
                 window_seconds: float = 10.0, stride_seconds: float = 5.0):
        self.stream = ECGStreamReplayer(model, source_fs, window_seconds, stride_seconds)
        self.expected_sequence = 0
        self.dropped_packets = 0
        self.discontinuities = 0

    def ingest_packet(self, packet: ECGPacket) -> List[ECGPrediction]:
        if packet.sequence != self.expected_sequence:
            self.dropped_packets += max(1, packet.sequence - self.expected_sequence)
            self.discontinuities += 1
            self.stream.reset()
            self.expected_sequence = packet.sequence
        predictions = self.stream.ingest(packet.samples)
        self.expected_sequence += 1
        return predictions
