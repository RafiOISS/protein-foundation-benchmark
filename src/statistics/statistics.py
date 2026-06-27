"""Dataset statistics computation for the Protein Foundation Model Benchmark Framework."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..utils.io import save_csv
from ..utils.logging import get_logger


logger = get_logger(__name__)


def compute_dataset_statistics(
    sequences: List[str],
    labels: Optional[List[Any]] = None,
    split_names: Optional[Dict[str, List[int]]] = None,
    label_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive statistics for a dataset.

    Args:
        sequences: List of protein sequence strings.
        labels: Optional list of label values.
        split_names: Dict of split_name -> list of indices in that split.
        label_names: Optional mapping from label index to human-readable name.

    Returns:
        Dictionary of statistics.
    """
    lengths = [len(s) for s in sequences]
    n = len(sequences)

    stats: Dict[str, Any] = {
        "num_sequences": n,
        "min_length": int(min(lengths)) if lengths else 0,
        "max_length": int(max(lengths)) if lengths else 0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_length": float(np.median(lengths)) if lengths else 0.0,
        "std_length": float(np.std(lengths)) if lengths else 0.0,
    }

    # Split sizes
    if split_names:
        split_sizes = {}
        for name, indices in split_names.items():
            split_sizes[name] = len(indices)
            split_sizes[f"{name}_pct"] = round(100 * len(indices) / n, 2) if n else 0.0
        stats["split_sizes"] = split_sizes

    # Label distribution
    if labels is not None and len(labels) > 0:
        label_array = np.array(labels)
        unique, counts = np.unique(label_array, return_counts=True)
        label_dist = {}
        for u, c in zip(unique, counts):
            key = str(label_names.get(int(u), u)) if label_names else str(u)
            label_dist[key] = int(c)
        stats["label_distribution"] = label_dist
        stats["num_classes"] = len(unique)

        # Missing labels
        missing = sum(1 for l in labels if l is None or (isinstance(l, float) and np.isnan(l)))
        stats["missing_labels"] = missing

    # Duplicate sequences
    seen = set()
    dup_count = 0
    for s in sequences:
        if s in seen:
            dup_count += 1
        else:
            seen.add(s)
    stats["duplicate_sequences"] = dup_count
    stats["unique_sequences"] = n - dup_count

    return stats


def save_statistics_csv(
    stats: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Save statistics as a CSV file with one row per statistic.

    Args:
        stats: Statistics dictionary.
        output_path: Output CSV path.

    Returns:
        Path to saved file.
    """
    rows = []

    # Top-level scalar stats
    for key in ("num_sequences", "min_length", "max_length", "mean_length",
                "median_length", "std_length", "num_classes",
                "missing_labels", "duplicate_sequences", "unique_sequences"):
        if key in stats:
            rows.append({"statistic": key, "value": stats[key]})

    # Split sizes
    if "split_sizes" in stats:
        for k, v in stats["split_sizes"].items():
            rows.append({"statistic": f"split_{k}", "value": v})

    # Label distribution
    if "label_distribution" in stats:
        for k, v in stats["label_distribution"].items():
            rows.append({"statistic": f"label_{k}", "value": v})

    df = pd.DataFrame(rows)
    return save_csv(df, output_path)


def print_statistics(stats: Dict[str, Any]) -> None:
    """Print statistics to console in a readable format."""
    print("\n=== Dataset Statistics ===")
    print(f"  Sequences:         {stats.get('num_sequences', '?')}")
    print(f"  Unique sequences:  {stats.get('unique_sequences', '?')}")
    print(f"  Duplicate seqs:    {stats.get('duplicate_sequences', '?')}")
    print(f"  Length range:      {stats.get('min_length', '?')} - {stats.get('max_length', '?')}")
    print(f"  Mean length:       {stats.get('mean_length', '?'):.1f}")
    print(f"  Median length:     {stats.get('median_length', '?'):.1f}")
    print(f"  Std length:        {stats.get('std_length', '?'):.2f}")

    if "num_classes" in stats and stats["num_classes"]:
        print(f"  Num classes:       {stats['num_classes']}")
    if "missing_labels" in stats and stats["missing_labels"]:
        print(f"  Missing labels:    {stats['missing_labels']}")

    if "split_sizes" in stats:
        print(f"  Split sizes:")
        for k, v in stats["split_sizes"].items():
            if not k.endswith("_pct"):
                pct_key = f"{k}_pct"
                pct = stats["split_sizes"].get(pct_key, "")
                pct_str = f" ({pct}%)" if pct != "" else ""
                print(f"    {k}: {v}{pct_str}")

    if "label_distribution" in stats:
        print(f"  Label distribution:")
        for k, v in sorted(stats["label_distribution"].items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
    print()
