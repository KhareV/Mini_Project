"""Incremental decoder for fragmented/corrupted hardware ECG byte streams."""

from __future__ import annotations

import struct

from .protocol import CRC, HEADER, MAGIC, decode_packet


class ECGFrameDecoder:
    """Recover valid packets from arbitrary transport byte chunks."""

    def __init__(self, max_samples: int = 4096):
        self.max_samples = max_samples
        self._buffer = bytearray()
        self.frames_rejected = 0

    def feed(self, data: bytes) -> list:
        self._buffer.extend(data)
        decoded = []
        while True:
            start = self._buffer.find(MAGIC)
            if start < 0:
                self._buffer = self._buffer[-(len(MAGIC) - 1):]
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < HEADER.size:
                break
            count = struct.unpack_from("<H", self._buffer, HEADER.size - 2)[0]
            if count > self.max_samples:
                del self._buffer[:len(MAGIC)]
                self.frames_rejected += 1
                continue
            frame_size = HEADER.size + count * 4 + CRC.size
            if len(self._buffer) < frame_size:
                break
            frame = bytes(self._buffer[:frame_size])
            try:
                decoded.append(decode_packet(frame))
                del self._buffer[:frame_size]
            except ValueError:
                del self._buffer[:1]
                self.frames_rejected += 1
        return decoded

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)
