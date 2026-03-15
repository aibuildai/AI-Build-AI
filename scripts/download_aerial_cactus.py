#!/usr/bin/env python3
"""Download and prepare the Aerial Cactus Identification dataset from Kaggle.

Usage:
    python scripts/download_aerial_cactus.py \
        --kaggle-username <user> --kaggle-key <key> \
        --data-dir /path/to/data/aerial-cactus-identification

Prerequisites:
    pip install kaggle
    You must also accept the competition rules at:
      https://www.kaggle.com/competitions/aerial-cactus-identification/rules
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


COMPETITION = "aerial-cactus-identification"
EXPECTED_FILES = ["train", "train.csv", "test", "sample_submission.csv"]


def authenticate_kaggle_api(username: str, key: str):
    """Authenticate and return a KaggleApi instance."""
    # Must set env vars BEFORE importing kaggle, as it authenticates on import
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Error: 'kaggle' package not found. Install it with:")
        print("  pip install kaggle")
        sys.exit(1)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"Kaggle authentication failed: {e}")
        print()
        print("Please ensure your Kaggle username and API key are correct.")
        print("  Go to https://www.kaggle.com/settings -> API -> 'Generate New Token'")
        sys.exit(1)

    return api


def download_competition_data(data_dir: Path):
    """Download and extract competition data to data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {COMPETITION} dataset...")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(data_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Download failed:\n{result.stderr or result.stdout}")
        if "403" in (result.stderr or "") or "rule" in (result.stderr or "").lower():
            print()
            print("You must accept the competition rules before downloading:")
            print(f"  https://www.kaggle.com/competitions/{COMPETITION}/rules")
        sys.exit(1)

    zip_path = data_dir / f"{COMPETITION}.zip"
    if not zip_path.exists():
        zips = list(data_dir.glob("*.zip"))
        if zips:
            zip_path = zips[0]
        else:
            print(f"Error: No zip file found in {data_dir}")
            sys.exit(1)

    print(f"Extracting dataset to {data_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    zip_path.unlink()

    # Extract any nested zips (e.g. train.zip, test.zip)
    for nested_zip in list(data_dir.glob("*.zip")):
        print(f"Extracting {nested_zip.name}...")
        with zipfile.ZipFile(nested_zip, "r") as zf:
            zf.extractall(data_dir)
        nested_zip.unlink()


def verify_dataset(data_dir: Path) -> bool:
    """Check that expected files/dirs exist."""
    errors = 0
    for item in EXPECTED_FILES:
        if not (data_dir / item).exists():
            print(f"Warning: expected {item} not found in {data_dir}")
            errors += 1
    return errors == 0


def main():
    parser = argparse.ArgumentParser(
        description="Download the Aerial Cactus Identification dataset from Kaggle."
    )
    parser.add_argument(
        "--kaggle-username",
        type=str,
        required=True,
        help="Your Kaggle username",
    )
    parser.add_argument(
        "--kaggle-key",
        type=str,
        required=True,
        help="Your Kaggle API key",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory to store the dataset",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    authenticate_kaggle_api(args.kaggle_username, args.kaggle_key)
    download_competition_data(data_dir)

    if verify_dataset(data_dir):
        print()
        print(f"Dataset ready at: {data_dir}")
        print("  train/              — training images")
        print("  train.csv           — training labels")
        print("  test/               — test images")
        print("  sample_submission.csv")
    else:
        print()
        print("Dataset extracted but some expected files are missing.")
        print(f"Please check {data_dir} manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
