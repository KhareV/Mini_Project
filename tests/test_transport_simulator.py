import numpy as np

from inference.transport_simulator import packetize_signal, replay_packets


class FakeModel:
    def predict(self, signal, source_fs):
        from inference.ecg_model_v1 import ECGPrediction
        return ECGPrediction("NORMAL", 0.1, 0.5, "TEST", "1.1.0")


def test_packet_trace_is_seeded_and_gap_detectable():
    signal = np.ones(300, dtype=np.float32)
    first = packetize_signal(signal, 100, packet_samples=25, jitter_ms=4, drop_every=4, seed=7)
    second = packetize_signal(signal, 100, packet_samples=25, jitter_ms=4, drop_every=4, seed=7)
    assert [(p.packet.sequence, p.timestamp_ms) for p in first] == [
        (p.packet.sequence, p.timestamp_ms) for p in second
    ]
    report = replay_packets(FakeModel(), first, 100)
    assert report["packets_received"] == 9
    # The final dropped packet has no subsequent packet to reveal the gap.
    assert report["dropped_packets_detected"] == 2
    assert report["discontinuities"] == 2


def test_packetizer_rejects_invalid_settings():
    try:
        packetize_signal(np.ones(10), 100, packet_samples=0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid packet size must be rejected")
