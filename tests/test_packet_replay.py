import numpy as np

from inference.packet_replay import ECGPacket, ECGPacketReplayer


class FakeModel:
    def predict(self, signal, source_fs):
        from inference.ecg_model_v1 import ECGPrediction
        return ECGPrediction("NORMAL", 0.1, 0.5, "TEST", "1.1.0")


def test_ordered_packets_match_contiguous_stream():
    signal = np.arange(6000, dtype=np.float32)
    replay = ECGPacketReplayer(FakeModel(), 500)
    outputs = []
    for seq, start in enumerate(range(0, len(signal), 300)):
        outputs.extend(replay.ingest_packet(ECGPacket(seq, signal[start:start + 300])))
    assert len(outputs) == 1
    assert replay.dropped_packets == 0
    assert replay.discontinuities == 0


def test_gap_resets_buffer_and_is_reported():
    replay = ECGPacketReplayer(FakeModel(), 10, window_seconds=1, stride_seconds=0.5)
    replay.ingest_packet(ECGPacket(0, np.ones(8, dtype=np.float32)))
    assert replay.ingest_packet(ECGPacket(2, np.ones(8, dtype=np.float32))) == []
    assert replay.dropped_packets == 1
    assert replay.discontinuities == 1
    # The post-gap stream must fill a fresh window, not use the old 8 samples.
    assert replay.ingest_packet(ECGPacket(3, np.ones(2, dtype=np.float32))) == [
        replay.stream.model.predict(np.ones(10, dtype=np.float32), 10)
    ]
