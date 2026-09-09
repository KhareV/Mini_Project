"""Versioned binary ECG packet contract for future serial/BLE adapters.

Frame: magic(4) version(1) device_id(16) sequence(u32) timestamp_us(u64)
sample_count(u16) samples(float32 LE * N) crc32(u32). CRC covers the header
and sample payload, allowing corrupted radio frames to be rejected early.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

MAGIC = b"NHM1"
VERSION = 1
DEVICE_BYTES = 16
HEADER = struct.Struct("<4sB16sIQH")
CRC = struct.Struct("<I")


@dataclass(frozen=True)
class DeviceECGPacket:
    device_id: str
    sequence: int
    timestamp_us: int
    samples: np.ndarray


def encode_packet(packet: DeviceECGPacket) -> bytes:
    device = packet.device_id.encode("ascii")
    samples = np.asarray(packet.samples, dtype="<f4").reshape(-1)
    if not 0 < len(device) <= DEVICE_BYTES or len(samples) > 65535:
        raise ValueError("device id or sample count exceeds protocol limits")
    if packet.sequence < 0 or packet.timestamp_us < 0 or not np.isfinite(samples).all():
        raise ValueError("invalid sequence, timestamp, or samples")
    padded_device = device.ljust(DEVICE_BYTES, b"\0")
    body = HEADER.pack(MAGIC, VERSION, padded_device, packet.sequence,
                       packet.timestamp_us, len(samples)) + samples.tobytes()
    return body + CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def decode_packet(frame: bytes) -> DeviceECGPacket:
    if len(frame) < HEADER.size + CRC.size:
        raise ValueError("truncated ECG packet")
    body, received_crc = frame[:-CRC.size], CRC.unpack(frame[-CRC.size:])[0]
    if zlib.crc32(body) & 0xFFFFFFFF != received_crc:
        raise ValueError("ECG packet CRC mismatch")
    magic, version, raw_device, sequence, timestamp_us, count = HEADER.unpack(body[:HEADER.size])
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported ECG packet version")
    payload = body[HEADER.size:]
    if len(payload) != count * 4:
        raise ValueError("ECG sample count does not match payload")
    samples = np.frombuffer(payload, dtype="<f4").copy()
    return DeviceECGPacket(raw_device.rstrip(b"\0").decode("ascii"), sequence,
                            timestamp_us, samples)
