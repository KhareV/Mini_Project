from pathlib import Path

from training.run_manifest import build_run_manifest, canonical_json_hash, write_manifest
from training.train_classical import predict_probabilities_checked


def test_hash_is_order_independent():
    assert canonical_json_hash({"a": 1, "b": 2}) == canonical_json_hash({"b": 2, "a": 1})


def test_synthetic_manifest_is_explicit_and_does_not_tune_test_set(tmp_path):
    manifest = build_run_manifest(
        experiment_group="test", dataset="SYNTHETIC", seed=42,
        split_counts={"train": 7, "val": 2, "test": 1},
        preprocessing={"preprocessing_version": "1.1.0"},
        normalization_stats={"mean": 0.0, "std": 1.0},
        synthetic=True, project_root=Path(__file__).parents[1],
        dataset_details={"participant_split": "patient_level"},
    )
    assert manifest["synthetic"] is True
    assert manifest["test_used_for_tuning"] is False
    assert "not clinical" in manifest["research_claim_boundary"]
    destination = tmp_path / "run_manifest.json"
    write_manifest(manifest, destination)
    assert destination.is_file()


def test_checked_probabilities_reject_invalid_classifier_output():
    class InvalidClassifier:
        def predict_proba(self, features):
            return [[float("nan"), 1.0]]

    try:
        predict_probabilities_checked(InvalidClassifier(), [[0.0]])
    except RuntimeError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("Expected invalid probabilities to be rejected")
