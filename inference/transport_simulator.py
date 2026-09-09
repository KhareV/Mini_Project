"""Deterministic transport stress simulation for captured ECG signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .packet_replay import ECGPacket, ECGPacketReplayer


@dataclass(frozen=True)
class TimedPacket:
    packet: ECGPacket
    timestamp_ms: float


def packetize_signal(signal: np.ndarray, source_fs: int, packet_samples: int = 125,
                     jitter_ms: float = 0.0, drop_every: int = 0,
                     seed: int = 42) -> List[TimedPacket]:
    """Split a signal into packets and optionally drop every Nth packet."""
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if source_fs <= 0 or packet_samples <= 0 or jitter_ms < 0 or drop_every < 0:
        raise ValueError("invalid transport simulation parameters")
    rng = np.random.RandomState(seed)
    packets: List[TimedPacket] = []
    sequence = 0
    for start in range(0, len(values), packet_samples):
        packet_index = start // packet_samples
        chunk = values[start:start + packet_samples]
        if drop_every and (packet_index + 1) % drop_every == 0:
            sequence += 1
            continue
        jitter = float(rng.uniform(-jitter_ms, jitter_ms)) if jitter_ms else 0.0
        timestamp_ms = (start / source_fs) * 1000.0 + jitter
        packets.append(TimedPacket(ECGPacket(sequence, chunk), timestamp_ms))
        sequence += 1
    return packets


def replay_packets(model, packets: List[TimedPacket], source_fs: int) -> dict:
    """Replay a packet trace and return transport/prediction counters."""
    replayer = ECGPacketReplayer(model, source_fs)
    predictions = []
    for timed in packets:
        predictions.extend(replayer.ingest_packet(timed.packet))
    timestamps = [p.timestamp_ms for p in packets]
    return {
        "packets_received": len(packets),
        "predictions_emitted": len(predictions),
        "dropped_packets_detected": replayer.dropped_packets,
        "discontinuities": replayer.discontinuities,
        "timestamp_jitter_span_ms": (max(timestamps) - min(timestamps)) if timestamps else 0.0,
    }
