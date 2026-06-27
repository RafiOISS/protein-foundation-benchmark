#!/usr/bin/env python
"""Evaluate a trained model on a dataset.

Usage:
    python scripts/evaluate.py --model esm2 --dataset tape_ss3 --checkpoint outputs/experiments/.../checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ProteinBenchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    args = parser.parse_args()

    benchmark = ProteinBenchmark()
    print(f"Evaluating {args.model} on {args.dataset} from {args.checkpoint}")
    print("Evaluation logic will be implemented per model.")


if __name__ == "__main__":
    main()