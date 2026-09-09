import pandas as pd

from federated.partitions import partition_digest, partition_manifest


def fixture_manifest():
    return pd.DataFrame({
        "participant_id": [f"p{i}" for i in range(12) for _ in range(2)],
        "label_int": [label for i in range(12) for label in (i % 2, i % 2)],
        "record_id": [f"r{i}_{j}" for i in range(12) for j in range(2)],
    })


def test_partitions_are_deterministic_and_leak_free():
    manifest = fixture_manifest()
    first = partition_manifest(manifest, 3, mode="iid", seed=42)
    second = partition_manifest(manifest, 3, mode="iid", seed=42)
    assert partition_digest(first) == partition_digest(second)
    assigned = pd.concat(first.values())
    assert assigned.groupby("participant_id").client_id.nunique().max() == 1
    assert len(assigned) == len(manifest)


def test_label_skew_is_supported():
    partitions = partition_manifest(fixture_manifest(), 4, mode="label_skew", seed=7)
    assert set(partitions) == {"client_000", "client_001", "client_002", "client_003"}
