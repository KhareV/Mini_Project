"""Guard the frozen Phase 0 decisions against accidental configuration drift."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_base_config():
    with (ROOT / "configs" / "base.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_primary_task_and_unreliable_state_are_frozen():
    config = load_base_config()
    assert config["outputs"]["physiological_classes"] == {
        "NORMAL": 0,
        "ABNORMAL": 1,
    }
    assert config["outputs"]["unreliable_state"] == "UNRELIABLE"


def test_patient_split_is_complete_and_exclusive():
    split = load_base_config()["split"]
    assert split["unit"] == "participant"
    assert sum(
        split[name]
        for name in ("train_fraction", "validation_fraction", "test_fraction")
    ) == 1.0


def test_dataset_roles_prevent_external_validation_leakage():
    datasets = load_base_config()["data"]
    assert datasets["ptbxl"]["role"] == "primary_development"
    assert datasets["mitbih"]["role"] == "external_validation_only"
    assert datasets["nstdb"]["role"] == "noise_robustness_only"


def test_signal_shape_contract():
    signal = load_base_config()["signal"]
    assert signal["canonical_ecg_rate_hz"] == 250
    assert signal["ecg_window_seconds"] == 10
    assert signal["ecg_stride_seconds"] == 5
    assert signal["canonical_ecg_rate_hz"] * signal["ecg_window_seconds"] == 2500


def test_test_set_cannot_be_used_for_model_development():
    evaluation = load_base_config()["evaluation"]
    assert evaluation["selection_split"] == "validation"
    assert evaluation["threshold_split"] == "validation"
    assert evaluation["calibration_split"] == "validation"
    assert evaluation["test_for_tuning"] is False


def test_federation_and_hardware_orders_are_dependency_safe():
    config = load_base_config()
    assert config["federated"]["prerequisite_model_state"] == "FROZEN"
    assert config["federated"]["algorithm_order"][0] == "fedavg_iid"
    assert config["hardware"]["integration_order"][:2] == ["capture", "replay"]

def test_authoritative_spec_contains_phase_zero_contract():
    spec = (ROOT / "docs" / "ML_SYSTEM_SPEC.md").read_text(encoding="utf-8")
    for required in (
        "FROZEN_FOR_PHASE_1",
        "UNRELIABLE",
        "PTB-XL 1.0.3",
        "Patient-level splitting",
        "Centralized baselines",
        "Federated learning",
        "Hardware contract",
        "Truthfulness boundary",
    ):
        assert required in spec
