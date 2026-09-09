import numpy as np

from hardware.bridge import HardwareECGBridge
from hardware.protocol import DeviceECGPacket, encode_packet
from inference.ecg_model_v1 import ECGPrediction


class FakeModel:
    def predict(self, signal, source_fs):
        return ECGPrediction("NORMAL", 0.1, 0.5, "TEST", "1.1.0")


def test_bridge_handles_fragmentation_and_forwards_valid_packets():
    bridge = HardwareECGBridge(FakeModel(), source_fs=10)
    frames = [encode_packet(DeviceECGPacket("edge", i, i * 1000, np.ones(5)))
              for i in range(2)]
    predictions = []
    payload = b"".join(frames)
    for offset in range(0, len(payload), 3):
        predictions.extend(bridge.feed(payload[offset:offset + 3]))
    assert bridge.device_packets == 2
    assert bridge.frames_rejected == 0
    assert bridge.discontinuities == 0
    assert predictions == []  # 10-second window not yet complete


def test_bridge_counts_corruption_and_sequence_gap():
    bridge = HardwareECGBridge(FakeModel(), source_fs=10)
    bad = bytearray(encode_packet(DeviceECGPacket("edge", 0, 0, np.ones(5))))
    bad[-1] ^= 1
    good = encode_packet(DeviceECGPacket("edge", 2, 0, np.ones(5)))
    bridge.feed(bytes(bad) + good)
    assert bridge.frames_rejected >= 1
    assert bridge.device_packets == 1
    assert bridge.discontinuities == 1
