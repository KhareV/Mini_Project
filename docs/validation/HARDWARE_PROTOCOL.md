# Hardware packet protocol (pre-driver contract)

The hardware boundary uses a small little-endian binary frame with a magic/version header, fixed-width device identifier, monotonic sequence, microsecond timestamp, float32 ECG samples, and CRC32. `hardware.protocol` is transport-independent and can be used by serial or BLE drivers later. It does not claim a radio implementation, clock accuracy, or production security; those are Phase 12 acceptance gates.
