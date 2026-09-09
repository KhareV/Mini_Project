from pathlib import Path

from datasets.bidmc import BIDMCDataset


def test_bidmc_discovery_is_deterministic(tmp_path: Path):
    (tmp_path / "bidmc_02.hea").write_text("", encoding="utf-8")
    (tmp_path / "bidmc_01.hea").write_text("", encoding="utf-8")
    records = BIDMCDataset(str(tmp_path)).discover()
    assert [record.record_id for record in records] == ["bidmc_01", "bidmc_02"]


def test_bidmc_channel_aliases_are_canonicalized():
    mapping = BIDMCDataset.canonical_channels(["II", "PLETH", "RESP", "SpO2"])
    assert mapping == {"ecg": "II", "ppg": "PLETH", "resp": "RESP", "spo2": "SpO2"}
