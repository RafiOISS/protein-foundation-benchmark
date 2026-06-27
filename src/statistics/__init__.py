"""
Statistics package for the Protein Foundation Model Benchmark Framework.

Contains statistical significance tests for model comparison.
"""

from .statistical_tests import (
    wilcoxon_test,
    paired_ttest,
    bootstrap_ci,
    friedman_test,
    nemenyi_test,
    multiple_comparison_correction,
    compare_models,
)

__all__ = [
    "wilcoxon_test",
    "paired_ttest",
    "bootstrap_ci",
    "friedman_test",
    "nemenyi_test",
    "multiple_comparison_correction",
    "compare_models",
]