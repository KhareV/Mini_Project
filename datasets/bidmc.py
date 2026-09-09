"""BIDMC synchronized ECG/PPG/SpO2 loader contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BIDMCRecord:
    record_id: str
    file_path: str
    sampling_rate: float
    channels: tuple[str, ...]


class BIDMCDataset:
    """Discover and load PhysioNet BIDMC WFDB records.

    Channel names vary slightly across exports; the loader maps common ECG,
    plethysmogram, respiration, and SpO2 aliases into stable keys.
    """

    ALIASES = {
        "ecg": {"ecg", "ii", "mlii"},
        "ppg": {"pleth", "ppg", "photoplethysmogram"},
        "resp": {"resp", "respiration"},
        "spo2": {"spo2", "o2", "oximetry"},
    }

    def __init__(self, data_dir: str = "data/raw/bidmc", dataset_version: str = "1.0.0"):
        self.data_dir = Path(data_dir)
        self.dataset_version = dataset_version

    def discover(self) -> list[BIDMCRecord]:
        records = []
        for header in sorted(self.data_dir.glob("*.hea")):
            record_id = header.stem
            records.append(BIDMCRecord(record_id, str(self.data_dir / record_id), 0.0, ()))
        return records

    @classmethod
    def canonical_channels(cls, names: list[str] | tuple[str, ...]) -> dict[str, str]:
        mapping = {}
        for name in names:
            normalized = name.strip().lower()
            for canonical, aliases in cls.ALIASES.items():
                if normalized in aliases and canonical not in mapping:
                    mapping[canonical] = name
        return mapping

    def load_record(self, record_id: str) -> tuple[BIDMCRecord, dict[str, tuple[np.ndarray, float]]]:
        try:
            import wfdb
        except ImportError as exc:
            raise ImportError("wfdb is required for BIDMC loading") from exc
        path = self.data_dir / record_id
        wfdb_record = wfdb.rdrecord(str(path))
        mapping = self.canonical_channels(list(wfdb_record.sig_name))
        signals = {}
        for canonical, original in mapping.items():
            index = wfdb_record.sig_name.index(original)
            signals[canonical] = (wfdb_record.p_signal[:, index].astype(np.float32),
                                  float(wfdb_record.fs))
        metadata = BIDMCRecord(record_id, str(path), float(wfdb_record.fs),
                               tuple(wfdb_record.sig_name))
        return metadata, signals
