"""Verify committed phase artifacts and run the complete automated suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        raise RuntimeError(f"missing artifact: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_artifacts() -> list[str]:
    checks = []
    inventory = load_json("data/reports/ptbxl_1.0.1_inventory.json")
    if inventory.get("status") != "complete" or inventory.get("missing_signal_files") != 0:
        raise RuntimeError("PTB-XL 1.0.1 inventory is incomplete")
    checks.append("Phase 1: PTB-XL inventory complete")

    run = load_json("experiments/phase3_ptbxl_1.0.1/run_manifest.json")
    if run.get("synthetic") or run.get("dataset") != "PTBXL_1.0.1":
        raise RuntimeError("Phase 3 real run manifest is not correctly identified")
    if run.get("test_used_for_tuning"):
        raise RuntimeError("Phase 3 manifest says test data was used for tuning")
    checks.append("Phase 3: real baseline provenance valid")

    cnn = load_json("experiments/phase4_ptbxl_v1/E04_ecg_cnn/results.json")
    threshold = cnn.get("decision_threshold")
    if not 0.0 < float(threshold) < 1.0:
        raise RuntimeError("Phase 4 decision threshold is not locked in (0, 1)")
    if cnn.get("test", {}).get("split") != "test":
        raise RuntimeError("Phase 4 results are missing test split metadata")
    checks.append("Phase 4: calibrated MODEL_V1 evidence present")

    for path in ("docs/PROJECT_STATUS.md", "docs/validation/PHASE_5_OFFLINE_REPLAY.md",
                 "docs/validation/PHASE_6_BIDMC_CONTRACT.md",
                 "docs/validation/PHASE_8_PARTITIONS.md",
                 "docs/validation/PHASE_9_FEDAVG_BASELINE.md"):
        if not (ROOT / path).is_file():
            raise RuntimeError(f"missing phase documentation: {path}")
    checks.append("Phase 0–9 validation documentation present")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    try:
        for check in verify_artifacts():
            print(f"PASS  {check}")
    except RuntimeError as exc:
        print(f"FAIL  {exc}")
        return 1
    if not args.skip_tests:
        print("RUN   python3 -m pytest -q")
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
        if result.returncode:
            return result.returncode
    print("PASS  phase verification complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
