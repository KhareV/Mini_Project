"""Replay one PTB-XL WFDB record through the backend ECG service."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ml.ecg_replay_service import ECGReplayService
from datasets.ptbxl import PTBXLDataset
from inference.packet_replay import ECGPacket
from inference.transport_simulator import packetize_signal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--record-id", default="records500/00000/00001_hr")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--dataset-version", default="1.0.1")
    parser.add_argument("--packet-samples", type=int, default=125)
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument("--drop-every", type=int, default=0)
    args = parser.parse_args()

    dataset = PTBXLDataset(args.data_dir, sampling_rate=500, target_lead="I",
                           dataset_version=args.dataset_version)
    records = {record.record_id: record for record in dataset.load_all_records()}
    if args.record_id not in records:
        raise SystemExit(f"Record not found: {args.record_id}")
    record = records[args.record_id]
    signal, source_fs = dataset.load_signal(record)
    service = ECGReplayService.from_artifacts(
        args.checkpoint, args.normalization, threshold=0.554572, source_fs=source_fs
    )
    trace = packetize_signal(signal, source_fs, packet_samples=args.packet_samples,
                             jitter_ms=args.jitter_ms, drop_every=args.drop_every)
    started = time.perf_counter()
    replay = service.replay_packets([timed.packet for timed in trace])
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result = {
        "record_id": record.record_id,
        "source_label": record.label_canonical,
        "source_fs": source_fs,
        "samples": int(len(signal)),
        "packet_samples": args.packet_samples,
        "jitter_ms": args.jitter_ms,
        "drop_every": args.drop_every,
        "replay_wall_time_ms": round(elapsed_ms, 3),
        **replay,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
