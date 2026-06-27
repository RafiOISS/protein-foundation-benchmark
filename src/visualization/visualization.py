"""Dataset visualization — publication-quality figures for dataset analysis."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..utils.logging import get_logger


logger = get_logger(__name__)


# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")


def plot_length_histogram(
    sequences: List[str],
    output_path: Union[str, Path],
    split_name: str = "all",
    dpi: int = 300,
) -> Path:
    """Plot sequence length histogram.

    Args:
        sequences: List of protein sequences.
        output_path: Output path (without extension — saves both .png and .pdf).
        split_name: Name of the split for title.
        dpi: Image resolution.

    Returns:
        Path to saved PNG file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lengths = [len(s) for s in sequences]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Sequence Length Distribution ({split_name})", fontsize=14)

    # Add summary stats
    stats_text = (
        f"n={len(lengths):,}\n"
        f"mean={np.mean(lengths):.1f}\n"
        f"median={np.median(lengths):.0f}\n"
        f"min={min(lengths):,}  max={max(lengths):,}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved length histogram: {output_path.with_suffix('.png')}")
    return output_path.with_suffix(".png")


def plot_length_boxplot(
    split_lengths: Dict[str, List[int]],
    output_path: Union[str, Path],
    dpi: int = 300,
) -> Path:
    """Plot sequence length boxplot comparing splits.

    Args:
        split_lengths: Dict of split_name -> list of lengths.
        output_path: Output path (without extension).
        dpi: Image resolution.

    Returns:
        Path to saved PNG file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    for split_name, lengths in split_lengths.items():
        data.append(lengths)
        labels.append(f"{split_name}\n(n={len(lengths):,})")

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], sns.color_palette()):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Sequence Length", fontsize=12)
    ax.set_title("Sequence Length by Split", fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved length boxplot: {output_path.with_suffix('.png')}")
    return output_path.with_suffix(".png")


def plot_class_distribution(
    labels: List[Any],
    output_path: Union[str, Path],
    label_names: Optional[Dict[int, str]] = None,
    split_name: str = "all",
    dpi: int = 300,
) -> Path:
    """Plot class/label distribution bar chart.

    Args:
        labels: List of label values.
        output_path: Output path (without extension).
        label_names: Optional mapping label_value -> name.
        split_name: Split name for title.
        dpi: Image resolution.

    Returns:
        Path to saved PNG file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique, counts = np.unique(labels, return_counts=True)
    names = [str(label_names.get(int(u), u)) if label_names else str(u) for u in unique]

    fig, ax = plt.subplots(figsize=(max(6, len(unique) * 0.6), 5))
    bars = ax.bar(range(len(unique)), counts, alpha=0.7, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels(names, fontsize=10, rotation=45, ha="right")
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Class Distribution ({split_name})", fontsize=14)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(count), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved class distribution: {output_path.with_suffix('.png')}")
    return output_path.with_suffix(".png")


def plot_split_comparison(
    split_data: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
    label_names: Optional[Dict[int, str]] = None,
    dpi: int = 300,
) -> Path:
    """Plot side-by-side comparison of label distributions across splits.

    Args:
        split_data: Dict of split_name -> {"labels": [...], "sequences": [...]}.
        output_path: Output path (without extension).
        label_names: Optional label name mapping.
        dpi: Image resolution.

    Returns:
        Path to saved PNG file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all unique classes
    all_labels = set()
    for data in split_data.values():
        all_labels.update(data.get("labels", []))
    all_labels = sorted(all_labels)

    # Build percentage matrix
    fig, axes = plt.subplots(1, len(split_data), figsize=(5 * len(split_data), 5),
                             sharey=True)
    if len(split_data) == 1:
        axes = [axes]

    for ax, (split_name, data) in zip(axes, split_data.items()):
        labels_list = data.get("labels", [])
        if not labels_list:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        unique, counts = np.unique(labels_list, return_counts=True)
        total = len(labels_list)
        pcts = {int(u): 100 * c / total for u, c in zip(unique, counts)}

        names = []
        values = []
        for lbl in all_labels:
            names.append(str(label_names.get(lbl, lbl)) if label_names else str(lbl))
            values.append(pcts.get(lbl, 0))

        ax.bar(range(len(names)), values, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9, rotation=45, ha="right")
        ax.set_ylabel("Percentage (%)" if ax == axes[0] else "")
        ax.set_title(f"{split_name}\n(n={total:,})", fontsize=12)

    fig.suptitle("Label Distribution Across Splits", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved split comparison: {output_path.with_suffix('.png')}")
    return output_path.with_suffix(".png")


def generate_all_figures(
    sequences: List[str],
    labels: Optional[List[Any]] = None,
    split_sequences: Optional[Dict[str, List[str]]] = None,
    split_labels: Optional[Dict[str, List[Any]]] = None,
    label_names: Optional[Dict[int, str]] = None,
    output_dir: Union[str, Path] = "figures",
) -> Dict[str, Path]:
    """Generate all standard dataset figures.

    Args:
        sequences: All sequences.
        labels: All labels.
        split_sequences: Dict split_name -> sequences.
        split_labels: Dict split_name -> labels.
        label_names: Label value -> name mapping.
        output_dir: Output directory for figures.

    Returns:
        Dict of figure_name -> Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Overall length histogram
    figures["length_histogram"] = plot_length_histogram(
        sequences, output_dir / "length_histogram", split_name="all"
    )

    # Length boxplot by split
    if split_sequences:
        split_lengths = {name: [len(s) for s in seqs] for name, seqs in split_sequences.items()}
        figures["length_boxplot"] = plot_length_boxplot(
            split_lengths, output_dir / "length_boxplot"
        )

    # Class distribution
    if labels is not None and len(labels) > 0:
        figures["class_distribution"] = plot_class_distribution(
            labels, output_dir / "class_distribution",
            label_names=label_names, split_name="all"
        )

    # Split comparison
    if split_labels and len(split_labels) > 1:
        split_data = {}
        for name, lbls in split_labels.items():
            split_data[name] = {"labels": lbls}
        figures["split_comparison"] = plot_split_comparison(
            split_data, output_dir / "split_comparison",
            label_names=label_names
        )

    return figures
