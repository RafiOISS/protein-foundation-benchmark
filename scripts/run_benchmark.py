#!/usr/bin/env python
"""CLI entry point for running a benchmark experiment.

Usage:
    python scripts/run_benchmark.py --config configs/experiments/my_exp.yaml
    python scripts/run_benchmark.py --name my_exp --dataset tape_ss3 --models esm2 protbert
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ProteinBenchmark
from src.framework.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a protein benchmark experiment")
    parser.add_argument("--config", type=str, help="Path to YAML experiment config")
    parser.add_argument("--name", type=str, default="benchmark", help="Experiment name")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name")
    parser.add_argument("--models", type=str, nargs="+", default=[], help="Model names")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig(
            name=args.name,
            dataset=args.dataset,
            models=args.models,
            device=args.device,
            epochs=args.epochs,
            seed=args.seed,
        )

    benchmark = ProteinBenchmark(seed=config.seed)
    results = benchmark.run(config)

    for r in results:
        print(f"  {r.model_name} | {r.task} | metrics={r.metrics} | {r.status}")


if __name__ == "__main__":
    main()