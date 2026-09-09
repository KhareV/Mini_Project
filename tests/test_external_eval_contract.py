from pathlib import Path


def test_external_evaluation_runner_exists_and_requires_locked_threshold():
    source = Path("scripts/evaluate_mitbih.py").read_text(encoding="utf-8")
    assert "--threshold" in source
    assert "no MIT-BIH fitting or tuning" in source
