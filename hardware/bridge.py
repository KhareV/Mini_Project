"""Bridge verified device frames into the deterministic ECG replay service."""

from __future__ import annotations

from .framing import ECGFrameDecoder
from .protocol import DeviceECGPacket
from inference.packet_replay import ECGPacket, ECGPacketReplayer


class HardwareECGBridge:
    """Decode transport bytes and forward only validated ECG packets."""

    def __init__(self, model, source_fs: int, max_samples: int = 4096):
        self.decoder = ECGFrameDecoder(max_samples=max_samples)
        self.replayer = ECGPacketReplayer(model, source_fs)
        self.device_packets = 0

    def feed(self, data: bytes) -> list:
        predictions = []
        for packet in self.decoder.feed(data):
            self.device_packets += 1
            predictions.extend(self.replayer.ingest_packet(
                ECGPacket(packet.sequence, packet.samples)
            ))
        return predictions

    @property
    def frames_rejected(self) -> int:
        return self.decoder.frames_rejected

    @property
    def discontinuities(self) -> int:
        return self.replayer.discontinuities
