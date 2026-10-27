"""Download the BAF dataset from Kaggle."""

from __future__ import annotations

import argparse
from pathlib import Path

DATASET = "sgpjesus/bank-account-fraud-dataset-neurips-2022"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise SystemExit("Install download support first: pip install -e '.[download]'") from error
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as error:
        raise SystemExit(
            "Kaggle authentication failed. Create an API token in Kaggle settings, configure it "
            "as Kaggle documents, then rerun this command."
        ) from error
    api.dataset_download_files(DATASET, path=str(args.destination), unzip=True, quiet=False)
    expected = ["Base.csv", "Variant I.csv", "Variant II.csv"]
    missing = [name for name in expected if not (args.destination / name).exists()]
    if missing:
        raise SystemExit(f"Download finished but required files are missing: {missing}")
    print("Required BAF files are ready in", args.destination)
    print("Review the upstream dataset card/license before packaging or redistributing any CSV.")


if __name__ == "__main__":
    main()
