"""ProteinBenchmark — main entry point for running benchmarks.

Usage:
    from src import ProteinBenchmark

    benchmark = ProteinBenchmark()
    benchmark.register_model("esm2", ESM2)
    benchmark.register_dataset("tape_ss3", TAPESS3Dataset)
    results = benchmark.run(experiment="my_exp", dataset="tape_ss3", models=["esm2"])

CLI (with Hydra):
    python benchmark.py model=esm2 dataset=tape_ss3 experiment=baseline
"""

from .framework.benchmark import ProteinBenchmark

if __name__ == "__main__":
    pass