#!/usr/bin/env python
"""Download a benchmark dataset.

Usage:
    python scripts/download_dataset.py --name tape_ss3
    python scripts/download_dataset.py --name tape_ss3 --force
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a benchmark dataset")
    parser.add_argument("--name", type=str, required=True, help="Dataset name (e.g., tape_ss3)")
    parser.add_argument("--output", type=str, default="outputs/cache/datasets",
                        help="Output cache directory")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if cached")
    parser.add_argument("--verify", action="store_true",
                        help="Verify integrity after download")
    parser.add_argument("--preprocess", action="store_true",
                        help="Preprocess after download")
    args = parser.parse_args()

    from src.registry.dataset_registry import DatasetRegistry
    from src.interfaces.base_dataset import DatasetSplit

    # Register built-in datasets
    import src.datasets  # noqa: F401

    cache_dir = Path(args.output)
    data_dir = cache_dir / args.name

    ds = DatasetRegistry.create(
        args.name,
        data_dir=cache_dir,
        split=DatasetSplit.TRAIN,
    )

    # Download
    print(f"\nDownloading dataset '{args.name}' to {data_dir}")
    ds.download(force=args.force)
    print("Download complete.")

    # Verify
    if args.verify:
        print("\nVerifying dataset integrity...")
        result = ds.verify()
        if result["valid"]:
            print("Integrity check: PASSED")
        else:
            print(f"Integrity check: FAILED")
            for err in result.get("errors", []):
                print(f"  ERROR: {err}")
            for warn in result.get("warnings", []):
                print(f"  WARNING: {warn}")
            if not result["valid"]:
                sys.exit(1)

    # Preprocess
    if args.preprocess:
        print("\nPreprocessing dataset...")
        ds.preprocess()
        print("Preprocessing complete.")
        print(f"Processed data at: {ds.processed_dir}")

    print(f"\nDataset ready: {data_dir}")


if __name__ == "__main__":
    main()
