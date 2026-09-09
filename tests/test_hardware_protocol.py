import numpy as np

from hardware.protocol import DeviceECGPacket, decode_packet, encode_packet


def test_packet_round_trip_preserves_samples_and_metadata():
    packet = DeviceECGPacket("esp32-edge", 7, 123456, np.array([0.1, -0.2], dtype=np.float32))
    decoded = decode_packet(encode_packet(packet))
    assert decoded.device_id == packet.device_id
    assert decoded.sequence == 7
    assert decoded.timestamp_us == 123456
    np.testing.assert_allclose(decoded.samples, packet.samples)


def test_packet_crc_rejects_corruption():
    frame = bytearray(encode_packet(DeviceECGPacket("edge", 1, 2, np.ones(3))))
    frame[-5] ^= 0xFF
    try:
        decode_packet(bytes(frame))
    except ValueError as exc:
        assert "CRC" in str(exc)
    else:
        raise AssertionError("corrupt frame must be rejected")
