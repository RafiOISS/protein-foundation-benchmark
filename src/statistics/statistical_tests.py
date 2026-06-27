"""Statistical tests for comparing model performance."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import wilcoxon, ttest_rel, friedmanchisquare, bootstrap

from ..utils.logging import get_logger


logger = get_logger(__name__)


def wilcoxon_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    mask = ~(np.isnan(scores1) | np.isnan(scores2))
    s1, s2 = scores1[mask], scores2[mask]
    if len(s1) < 2:
        return {"statistic": np.nan, "pvalue": np.nan, "significant": False}
    try:
        stat, p = wilcoxon(s1, s2, alternative=alternative)
        return {"statistic": float(stat), "pvalue": float(p), "significant": bool(p < 0.05)}
    except Exception as e:
        return {"statistic": np.nan, "pvalue": np.nan, "significant": False, "error": str(e)}


def paired_ttest(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    mask = ~(np.isnan(scores1) | np.isnan(scores2))
    s1, s2 = scores1[mask], scores2[mask]
    if len(s1) < 2:
        return {"statistic": np.nan, "pvalue": np.nan, "significant": False}
    try:
        stat, p = ttest_rel(s1, s2, alternative=alternative)
        return {"statistic": float(stat), "pvalue": float(p), "significant": bool(p < 0.05)}
    except Exception as e:
        return {"statistic": np.nan, "pvalue": np.nan, "significant": False, "error": str(e)}


def bootstrap_ci(
    scores1: np.ndarray,
    scores2: np.ndarray,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    mask = ~(np.isnan(scores1) | np.isnan(scores2))
    s1, s2 = scores1[mask], scores2[mask]
    if len(s1) < 2:
        return {"ci_low": np.nan, "ci_high": np.nan, "significant": False}

    def stat(x, y):
        return np.mean(x) - np.mean(y)

    try:
        result = bootstrap((s1, s2), stat, n_resamples=n_resamples, confidence_level=confidence_level, paired=True, random_state=42)
        ci_low, ci_high = result.confidence_interval
        return {
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "point_estimate": float(np.mean(s1) - np.mean(s2)),
            "significant": not (ci_low <= 0 <= ci_high),
        }
    except Exception as e:
        return {"ci_low": np.nan, "ci_high": np.nan, "significant": False, "error": str(e)}


def friedman_test(scores_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
    lengths = [len(v) for v in scores_dict.values()]
    if len(set(lengths)) > 1:
        raise ValueError("All score arrays must have same length")
    names = list(scores_dict.keys())
    data = np.array([scores_dict[n] for n in names]).T
    try:
        stat, p = friedmanchisquare(*data.T)
        return {"statistic": float(stat), "pvalue": float(p), "significant": bool(p < 0.05), "models": names}
    except Exception as e:
        return {"statistic": np.nan, "pvalue": np.nan, "significant": False, "error": str(e)}


def nemenyi_test(scores_dict: Dict[str, np.ndarray], alpha: float = 0.05) -> Dict[str, Any]:
    names = list(scores_dict.keys())
    pairwise = {}
    for i, m1 in enumerate(names):
        for j, m2 in enumerate(names):
            if i >= j:
                continue
            pairwise[f"{m1}_vs_{m2}"] = wilcoxon_test(scores_dict[m1], scores_dict[m2])
    return {"pairwise": pairwise, "alpha": alpha}


def multiple_comparison_correction(
    pvalues: List[float],
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    p = np.array(pvalues)
    n = len(p)
    if method == "bonferroni":
        corrected = np.minimum(p * n, 1.0)
    elif method == "benjamini_hochberg":
        sorted_idx = np.argsort(p)
        corrected = np.zeros_like(p)
        for i, idx in enumerate(sorted_idx):
            corrected[idx] = min(p[idx] * n / (i + 1), 1.0)
    elif method == "holm":
        sorted_idx = np.argsort(p)
        for i, idx in enumerate(sorted_idx):
            corrected[idx] = min(p[idx] * (n - i), 1.0)
    else:
        corrected = p
    return {"corrected_pvalues": corrected.tolist(), "significant": (corrected < alpha).tolist(), "method": method, "alpha": alpha}


def compare_models(
    scores_dict: Dict[str, np.ndarray],
    baseline: str,
    tests: Optional[List[str]] = None,
    correction: str = "bonferroni",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    if tests is None:
        tests = ["wilcoxon", "bootstrap"]
    if baseline not in scores_dict:
        raise ValueError(f"Baseline '{baseline}' not found")

    base = scores_dict[baseline]
    results = {}
    for name, scores in scores_dict.items():
        if name == baseline:
            continue
        r = {"model": name, "baseline": baseline, "diff_mean": float(np.mean(scores) - np.mean(base))}
        for t in tests:
            if t == "wilcoxon":
                r["wilcoxon"] = wilcoxon_test(scores, base)
            elif t == "bootstrap":
                r["bootstrap"] = bootstrap_ci(scores, base)
        results[name] = r
    return {"baseline": baseline, "comparisons": results, "correction": correction, "alpha": alpha}