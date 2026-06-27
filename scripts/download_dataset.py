#!/usr/bin/env python
"""Download a benchmark dataset.

Usage:
    python scripts/download_dataset.py --name tape_ss3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a benchmark dataset")
    parser.add_argument("--name", type=str, required=True, help="Dataset name")
    parser.add_argument("--output", type=str, default="outputs/cache/datasets", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{args.name}' to {output_dir}")
    print("Dataset download logic will be implemented per dataset.")


if __name__ == "__main__":
    main()