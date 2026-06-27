#!/usr/bin/env python
"""Export benchmark results to publication formats.

Usage:
    python scripts/export_results.py --experiment <id> --format latex
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ProteinBenchmark
from src.reporter.reporter import Reporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark results")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment ID")
    parser.add_argument("--format", type=str, choices=["csv", "latex", "md", "excel"], default="csv")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    benchmark = ProteinBenchmark()
    exp = benchmark.get_experiment(args.experiment)
    if not exp:
        print(f"Experiment {args.experiment} not found")
        return

    reporter = Reporter(output_dir=Path(args.output) if args.output else None)
    reporter.export(exp.results, fmt=args.format)
    print(f"Exported experiment {args.experiment} as {args.format}")


if __name__ == "__main__":
    main()