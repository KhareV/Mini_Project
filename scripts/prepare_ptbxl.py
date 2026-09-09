"""Acquire and audit the frozen PTB-XL 1.0.3 source reproducibly.

The downloader is resumable and downloads only the configured signal
resolution. Raw files remain under data/raw and are never committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.ptbxl import PTBXLDataset


LOGGER = logging.getLogger("prepare_ptbxl")
BASE_URL = "https://physionet.org/files/ptb-xl/{version}"
METADATA_FILES = (
    "ptbxl_database.csv",
    "scp_statements.csv",
    "RECORDS",
    "LICENSE.txt",
)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> Path:
    """Download with curl retry and resume support."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    command = [
        "curl", "--fail", "--location", "--silent", "--show-error", "--retry", "8",
        "--retry-all-errors", "--continue-at", "-",
        "--output", str(partial), url,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed ({result.returncode}): {url}")
    partial.replace(destination)
    return destination


def download_many(base_url: str, destination: Path, relative_files: Iterable[str], workers: int) -> None:
    files = tuple(dict.fromkeys(relative_files))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(download_file, f"{base_url}/{relative}", destination / relative): relative
            for relative in files
        }
        for index, future in enumerate(as_completed(pending), start=1):
            relative = pending[future]
            future.result()
            LOGGER.info("Downloaded %s (%d/%d)", relative, index, len(files))


def required_record_files(metadata_path: Path, sampling_rate: int) -> list[str]:
    frame = pd.read_csv(metadata_path, usecols=["filename_lr", "filename_hr"])
    column = "filename_hr" if sampling_rate == 500 else "filename_lr"
    bases = frame[column].dropna().astype(str)
    return [f"{base}{extension}" for base in bases for extension in (".hea", ".dat")]


def inventory(data_dir: Path, version: str, sampling_rate: int) -> dict:
    metadata_path = data_dir / "ptbxl_database.csv"
    if not metadata_path.is_file():
        return {
            "dataset": "ptbxl",
            "version": version,
            "status": "metadata_missing",
            "data_dir": str(data_dir),
        }

    frame = pd.read_csv(metadata_path)
    required = required_record_files(metadata_path, sampling_rate)
    missing = [relative for relative in required if not (data_dir / relative).is_file()]
    metadata_hashes = {
        name: sha256(data_dir / name)
        for name in METADATA_FILES
        if (data_dir / name).is_file()
    }
    return {
        "dataset": "ptbxl",
        "version": version,
        "status": "complete" if not missing else "incomplete",
        "data_dir": str(data_dir),
        "sampling_rate": sampling_rate,
        "metadata_records": int(len(frame)),
        "metadata_patients": int(frame["patient_id"].nunique()),
        "required_signal_files": len(required),
        "present_signal_files": len(required) - len(missing),
        "missing_signal_files": len(missing),
        "missing_examples": missing[:20],
        "metadata_sha256": metadata_hashes,
    }


def label_audit(data_dir: Path, version: str) -> dict:
    """Audit every metadata label without requiring waveform downloads."""
    dataset = PTBXLDataset(data_dir=str(data_dir), dataset_version=version)
    metadata = dataset.load_metadata()
    statements = dataset.load_scp_statements()
    outcomes = {}
    combinations = {}
    for codes in metadata["scp_codes"]:
        superclasses, canonical, _, status = dataset.resolve_label(codes, statements)
        outcome = f"{canonical}:{status}"
        combination = "|".join(superclasses) or "UNKNOWN"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        combinations[combination] = combinations.get(combination, 0) + 1
    return {
        "dataset": "ptbxl",
        "version": version,
        "metadata_records": len(metadata),
        "outcomes": dict(sorted(outcomes.items())),
        "superclass_combinations": dict(
            sorted(combinations.items(), key=lambda item: (-item[1], item[0]))
        ),
        "policy": {
            "normal_only": "NORMAL",
            "one_or_more_abnormal_without_normal": "ABNORMAL",
            "normal_and_abnormal": "EXCLUDE_AS_CONFLICT",
            "no_supported_diagnostic_superclass": "EXCLUDE_AS_UNSUPPORTED",
        },
    }
def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "inventory", "labels", "build"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--version", help="Override PTB-XL version for an existing local copy")
    parser.add_argument("--sampling-rate", type=int, choices=(100, 500),
                        help="Override source sampling rate")
    parser.add_argument("--lead", help="Override target lead")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    settings = config["data"]["ptbxl"]
    version = str(args.version or settings["version"])
    sampling_rate = int(args.sampling_rate or settings["source_rate_hz"])
    lead = str(args.lead or settings["primary_lead"])
    data_dir = args.data_dir or ROOT / "data" / "raw" / "ptbxl" / version
    base_url = BASE_URL.format(version=version)

    if args.command == "download":
        download_many(base_url, data_dir, METADATA_FILES, args.workers)
        if not args.metadata_only:
            files = required_record_files(data_dir / "ptbxl_database.csv", sampling_rate)
            download_many(base_url, data_dir, files, args.workers)

    if args.command == "labels":
        report = label_audit(data_dir, version)
        report_path = ROOT / "data" / "reports" / f"ptbxl_{version}_labels.json"
        write_report(report, report_path)
        return 0

    report = inventory(data_dir, version, sampling_rate)
    report_path = ROOT / "data" / "reports" / f"ptbxl_{version}_inventory.json"
    write_report(report, report_path)

    if args.command == "build":
        if report["status"] != "complete":
            LOGGER.error("Dataset is incomplete; manifest and split were not generated")
            return 2
        dataset = PTBXLDataset(
            data_dir=str(data_dir), sampling_rate=sampling_rate,
            target_lead=lead, dataset_version=version,
        )
        manifest_path = ROOT / "data" / "manifests" / f"ptbxl_{version}_manifest.csv"
        split_path = ROOT / "data" / "splits" / f"ptbxl_{version}_splits_seed42.csv"
        manifest = dataset.build_manifest(str(manifest_path), args.max_records)
        if manifest.empty:
            LOGGER.error("No valid resolved records were available")
            return 3
        dataset.generate_splits(manifest, seed=int(config["seed"]), output_path=str(split_path))
        LOGGER.info("Manifest: %s", manifest_path)
        LOGGER.info("Splits: %s", split_path)

    return 0 if report["status"] == "complete" or args.command == "download" else 2


if __name__ == "__main__":
    raise SystemExit(main())
