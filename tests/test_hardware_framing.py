import numpy as np

from hardware.framing import ECGFrameDecoder
from hardware.protocol import DeviceECGPacket, encode_packet


def test_decoder_handles_fragmented_frames():
    frame = encode_packet(DeviceECGPacket("edge", 2, 99, np.arange(6, dtype=np.float32)))
    decoder = ECGFrameDecoder()
    packets = []
    for byte in frame:
        packets.extend(decoder.feed(bytes([byte])))
    assert len(packets) == 1
    assert packets[0].sequence == 2
    assert decoder.buffered_bytes == 0


def test_decoder_resynchronizes_after_corrupt_frame():
    first = bytearray(encode_packet(DeviceECGPacket("edge", 1, 10, np.ones(2))))
    first[-1] ^= 0xAA
    second = encode_packet(DeviceECGPacket("edge", 2, 20, np.zeros(2)))
    decoder = ECGFrameDecoder()
    packets = decoder.feed(b"noise" + bytes(first) + second)
    assert [packet.sequence for packet in packets] == [2]
    assert decoder.frames_rejected >= 1
