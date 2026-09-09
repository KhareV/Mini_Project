"""Phase 1 tests for PTB-XL versioning, labels, inventory, and splits."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from datasets.ptbxl import PTBXLDataset
from scripts.prepare_ptbxl import inventory, label_audit, required_record_files


@pytest.fixture
def statements():
    return pd.DataFrame(
        {
            "diagnostic": [1, 1, 1, 0],
            "diagnostic_class": ["NORM", "MI", "STTC", None],
        },
        index=["NORM_CODE", "MI_CODE", "STTC_CODE", "NON_DIAGNOSTIC"],
    )


def test_label_resolution_preserves_multilabel_and_flags_conflict(statements):
    dataset = PTBXLDataset()
    assert dataset.resolve_label({"MI_CODE": 80, "STTC_CODE": 20}, statements) == (
        ["MI", "STTC"], "ABNORMAL", 1, "resolved"
    )
    assert dataset.resolve_label({"NORM_CODE": 100, "MI_CODE": 20}, statements) == (
        ["MI", "NORM"], "CONFLICT", -1, "conflict"
    )
    assert dataset.resolve_label({"NON_DIAGNOSTIC": 100}, statements) == (
        [], "UNKNOWN", -1, "unsupported"
    )


def test_required_files_follow_selected_resolution(tmp_path):
    metadata = tmp_path / "ptbxl_database.csv"
    pd.DataFrame({
        "filename_lr": ["records100/a_lr", "records100/b_lr"],
        "filename_hr": ["records500/a_hr", "records500/b_hr"],
    }).to_csv(metadata, index=False)
    assert required_record_files(metadata, 500) == [
        "records500/a_hr.hea", "records500/a_hr.dat",
        "records500/b_hr.hea", "records500/b_hr.dat",
    ]


def test_inventory_reports_missing_pairs_without_claiming_completeness(tmp_path):
    pd.DataFrame({
        "patient_id": [1, 2],
        "filename_lr": ["records100/a_lr", "records100/b_lr"],
        "filename_hr": ["records500/a_hr", "records500/b_hr"],
    }).to_csv(tmp_path / "ptbxl_database.csv", index=False)
    (tmp_path / "records500").mkdir()
    (tmp_path / "records500" / "a_hr.hea").write_text("fixture", encoding="utf-8")
    report = inventory(tmp_path, "1.0.3", 500)
    assert report["status"] == "incomplete"
    assert report["required_signal_files"] == 4
    assert report["present_signal_files"] == 1
    assert report["missing_signal_files"] == 3


def test_label_audit_counts_real_resolution_outcomes(tmp_path):
    pd.DataFrame({
        "ecg_id": [1, 2, 3],
        "patient_id": [10, 11, 12],
        "scp_codes": [
            "{'NORM_CODE': 100}",
            "{'MI_CODE': 80}",
            "{'NORM_CODE': 100, 'MI_CODE': 20}",
        ],
        "filename_lr": ["a", "b", "c"],
        "filename_hr": ["a", "b", "c"],
    }).to_csv(tmp_path / "ptbxl_database.csv", index=False)
    pd.DataFrame({
        "diagnostic": [1, 1],
        "diagnostic_class": ["NORM", "MI"],
    }, index=["NORM_CODE", "MI_CODE"]).to_csv(tmp_path / "scp_statements.csv")
    report = label_audit(tmp_path, "1.0.3")
    assert report["outcomes"] == {
        "ABNORMAL:resolved": 1,
        "CONFLICT:conflict": 1,
        "NORMAL:resolved": 1,
    }


def test_patient_splits_are_deterministic_disjoint_and_hashed(tmp_path):
    rows = []
    for patient in range(20):
        for record in range(2):
            rows.append({
                "participant_id": str(patient),
                "record_id": f"record-{patient}-{record}",
                "label_int": patient % 2,
            })
    manifest = pd.DataFrame(rows)
    dataset = PTBXLDataset(dataset_version="1.0.3")
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first = dataset.generate_splits(manifest, seed=42, output_path=str(first_path))
    second = dataset.generate_splits(manifest, seed=42, output_path=str(second_path))

    assert first["split"].tolist() == second["split"].tolist()
    assert first.groupby("participant_id")["split"].nunique().max() == 1
    metadata = json.loads(Path(f"{first_path}.metadata.json").read_text())
    assert metadata["dataset_version"] == "1.0.3"
    assert metadata["sha256"] == hashlib.sha256(first_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("val_fraction,test_fraction", [(0, 0.15), (0.9, 0.2)])
def test_invalid_split_fractions_are_rejected(tmp_path, val_fraction, test_fraction):
    dataset = PTBXLDataset()
    manifest = pd.DataFrame({"participant_id": ["1", "2"]})
    with pytest.raises(ValueError):
        dataset.generate_splits(
            manifest, val_fraction=val_fraction, test_fraction=test_fraction,
            output_path=str(tmp_path / "invalid.csv"),
        )
