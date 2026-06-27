"""ProteinBERT preprocessing pipeline — full preprocessing with statistics, figures, and reports.

Reuses existing framework components:
  - statistics/statistics.py for computation
  - visualization/visualization.py for figures
  - reporter/reporter.py for report generation
  - datasets/tape_ss3.py for raw data loading

Every stage records statistics. Nothing silently disappears.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ...utils.logging import get_logger
from ...utils.io import ensure_dir, save_csv, save_json
from ...datasets.tape_ss3 import TapeSS3Dataset, SS3_LABEL_NAMES
from ...interfaces.base_dataset import DatasetSplit
from ...statistics.statistics import save_statistics_csv, print_statistics
from .constants import (
    AA_ALPHABET,
    AA_TO_ID,
    DEFAULT_PREPROCESSING_CONFIG,
    LABEL_TO_ID,
    PAD_ID,
)
from .encoder import (
    validate_sequences,
    encode_sequence,
    encode_labels,
    pad_sequences,
    truncate_sequences,
    create_attention_mask,
    create_batches,
)


logger = get_logger(__name__)


# ------------------------------------------------------------------
# Preprocessing configuration
# ------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """Configuration for the ProteinBERT preprocessing pipeline."""
    padding: str = "right"
    truncation: bool = True
    mask_padding: bool = True
    max_length: int = 512
    unknown_token: str = "X"
    extended_alphabet: bool = False
    batch_size: int = 8
    shuffle: bool = False
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "padding": self.padding,
            "truncation": self.truncation,
            "mask_padding": self.mask_padding,
            "max_length": self.max_length,
            "unknown_token": self.unknown_token,
            "extended_alphabet": self.extended_alphabet,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
        }


# ------------------------------------------------------------------
# Statistics collectors
# ------------------------------------------------------------------


def compute_aa_statistics(sequences: List[str]) -> Dict[str, Any]:
    """Compute amino-acid frequency statistics across all sequences.

    Args:
        sequences: List of raw protein sequences.

    Returns:
        Dict with AA frequencies, percentages, unknown counts.
    """
    total_residues = sum(len(s) for s in sequences)
    aa_counts: Dict[str, int] = {aa: 0 for aa in AA_ALPHABET}
    unknown_count = 0
    invalid_count = 0

    for seq in sequences:
        for char in seq.upper():
            if char in aa_counts:
                aa_counts[char] += 1
            elif char in ("X", "B", "Z", "U", "O"):
                unknown_count += 1
            else:
                invalid_count += 1

    stats: Dict[str, Any] = {
        "total_residues": total_residues,
        "aa_counts": aa_counts,
        "aa_percentages": {
            aa: round(100 * count / total_residues, 4) if total_residues else 0.0
            for aa, count in aa_counts.items()
        },
        "unknown_residue_count": unknown_count,
        "invalid_residue_count": invalid_count,
    }

    return stats


def compute_padding_statistics(
    input_lengths: np.ndarray,
    max_length: int,
) -> Dict[str, Any]:
    """Compute padding statistics after encoding.

    Args:
        input_lengths: Array of original sequence lengths.
        max_length: Maximum length used for padding.

    Returns:
        Dict with padding metrics.
    """
    n = len(input_lengths)
    padding_lengths = np.maximum(0, max_length - input_lengths)
    total_padding = int(padding_lengths.sum())
    total_tokens = n * max_length
    padding_ratio = round(total_padding / total_tokens, 4) if total_tokens else 0.0

    stats: Dict[str, Any] = {
        "max_length": max_length,
        "total_padding_tokens": total_padding,
        "padding_ratio": padding_ratio,
        "average_padding_per_sequence": round(float(padding_lengths.mean()), 2) if n else 0.0,
        "median_padding_per_sequence": int(np.median(padding_lengths)) if n else 0,
        "min_padding": int(padding_lengths.min()) if n else 0,
        "max_padding": int(padding_lengths.max()) if n else 0,
        "sequences_with_padding": int((padding_lengths > 0).sum()),
        "pct_sequences_with_padding": round(100 * (padding_lengths > 0).sum() / n, 2) if n else 0.0,
    }

    return stats


def compute_truncation_statistics(
    original_lengths: np.ndarray,
    max_length: int,
) -> Dict[str, Any]:
    """Compute truncation statistics.

    Args:
        original_lengths: Array of original sequence lengths.
        max_length: Maximum allowed length.

    Returns:
        Dict with truncation metrics.
    """
    truncated = original_lengths > max_length
    n_truncated = int(truncated.sum())
    n_total = len(original_lengths)
    residues_removed = np.maximum(0, original_lengths - max_length)

    stats: Dict[str, Any] = {
        "truncation_enabled": True,
        "max_length": max_length,
        "num_truncated_sequences": n_truncated,
        "pct_truncated": round(100 * n_truncated / n_total, 2) if n_total else 0.0,
        "total_residues_removed": int(residues_removed.sum()),
        "average_truncation_length": round(float(residues_removed[truncated].mean()), 2) if n_truncated else 0.0,
        "max_truncation_length": int(residues_removed.max()) if n_total else 0,
        "min_truncation_length": int(residues_removed[truncated].min()) if n_truncated else 0,
    }

    return stats


def compute_validation_statistics(
    total_input: int,
    invalid_sequences: List[Tuple[int, str]],
    invalid_residues: int,
    unknown_labels: List[int],
) -> Dict[str, Any]:
    """Compute validation/filtering statistics.

    Args:
        total_input: Total number of input samples.
        invalid_sequences: List of (index, error) tuples.
        invalid_residues: Count of invalid residues found.
        unknown_labels: Indices of samples with unknown labels.

    Returns:
        Dict with validation metrics.
    """
    n_invalid = len(invalid_sequences)
    n_unknown_labels = len(unknown_labels)
    retained = total_input - n_invalid - n_unknown_labels

    stats: Dict[str, Any] = {
        "total_input_samples": total_input,
        "invalid_sequences": n_invalid,
        "invalid_residue_count": invalid_residues,
        "unknown_labels": n_unknown_labels,
        "filtered_samples": n_invalid + n_unknown_labels,
        "retained_samples": retained,
        "retention_rate": round(100 * retained / total_input, 2) if total_input else 0.0,
        "filter_rate": round(100 * (n_invalid + n_unknown_labels) / total_input, 2) if total_input else 0.0,
    }

    return stats


# ------------------------------------------------------------------
# Visualization helpers
# ------------------------------------------------------------------


def _plot_aa_distribution(
    aa_stats: Dict[str, Any],
    output_path: Union[str, Path],
    dpi: int = 300,
) -> Path:
    """Plot amino-acid frequency distribution bar chart.

    Args:
        aa_stats: AA statistics from compute_aa_statistics.
        output_path: Output path (without extension).
        dpi: Image resolution.

    Returns:
        Path to saved PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("colorblind")

    aas = list(aa_stats["aa_percentages"].keys())
    pcts = [aa_stats["aa_percentages"][aa] for aa in aas]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(aas)), pcts, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(aas)))
    ax.set_xticklabels(aas, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Amino-Acid Distribution", fontsize=14)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved AA distribution: {output_path.with_suffix('.png')}")
    return output_path.with_suffix(".png")


def _plot_padding_distribution(
    input_lengths: np.ndarray,
    max_length: int,
    output_path: Union[str, Path],
    dpi: int = 300,
) -> Path:
    """Plot padding length distribution.

    Args:
        input_lengths: Array of original sequence lengths.
        max_length: Maximum sequence length.
        output_path: Output path (without extension).
        dpi: Image resolution.

    Returns:
        Path to saved PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    padding_lengths = np.maximum(0, max_length - input_lengths)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(padding_lengths, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Padding Length", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Padding Distribution", fontsize=14)

    stats_text = (
        f"mean={padding_lengths.mean():.1f}\n"
        f"median={np.median(padding_lengths):.0f}\n"
        f"min={padding_lengths.min():,}  max={padding_lengths.max():,}"
    )
    axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes,
                 fontsize=9, verticalalignment="top", horizontalalignment="right",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Padding ratio per sequence
    ratios = padding_lengths / max_length
    axes[1].hist(ratios, bins=50, alpha=0.7, color="orange", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Padding Ratio", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Padding Ratio Distribution", fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved padding distribution: {output_path.with_suffix('.png')}")
    return output_path.with_suffix(".png")


def _plot_truncation_distribution(
    original_lengths: np.ndarray,
    max_length: int,
    output_path: Union[str, Path],
    dpi: int = 300,
) -> Path:
    """Plot truncation distribution if truncation occurs.

    Args:
        original_lengths: Array of original sequence lengths.
        max_length: Maximum sequence length.
        output_path: Output path (without extension).
        dpi: Image resolution.

    Returns:
        Path to saved PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    residues_removed = np.maximum(0, original_lengths - max_length)
    truncated = residues_removed > 0

    fig, ax = plt.subplots(figsize=(8, 5))

    if truncated.any():
        ax.hist(residues_removed[truncated], bins=50, alpha=0.7,
                color="red", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Residues Removed", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Truncation Distribution ({truncated.sum():,} sequences truncated)", fontsize=14)

        stats_text = (
            f"total removed={int(residues_removed.sum()):,}\n"
            f"mean={residues_removed[truncated].mean():.1f}\n"
            f"max={residues_removed.max():,}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "No sequences were truncated", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.set_title("Truncation Analysis", fontsize=14)
        ax.set_xlabel("Residues Removed")
        ax.set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved truncation distribution: {output_path.with_suffix('.png')}")
    return output_path.with_suffix(".png")


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------


def _generate_preprocessing_report(
    config: PreprocessingConfig,
    dataset_stats: Dict[str, Any],
    aa_stats: Dict[str, Any],
    ss3_stats: Dict[str, Any],
    padding_stats: Dict[str, Any],
    truncation_stats: Dict[str, Any],
    validation_stats: Dict[str, Any],
    output_dir: Union[str, Path],
    dataset_name: str = "tape_ss3",
) -> Path:
    """Generate a preprocessing report suitable for Methods section of a paper.

    Args:
        config: Preprocessing configuration used.
        dataset_stats: Dataset overview statistics.
        aa_stats: Amino-acid statistics.
        ss3_stats: SS3 label statistics.
        padding_stats: Padding statistics.
        truncation_stats: Truncation statistics.
        validation_stats: Validation/filtering statistics.
        output_dir: Output directory for the report.
        dataset_name: Name of the dataset.

    Returns:
        Path to generated report file.
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    lines = []
    lines.append(f"# Preprocessing Report: {dataset_name}")
    lines.append("")
    lines.append(f"- **Generated**: {datetime.now().isoformat()}")
    lines.append(f"- **Model**: ProteinBERT")
    lines.append("")

    # 1. Preprocessing steps
    lines.append("## Preprocessing Steps")
    lines.append("")
    lines.append("The following preprocessing pipeline was applied:")
    lines.append("")
    steps = [
        "1. **Sequence Validation** — Validate each sequence against the standard 20 amino-acid alphabet.",
        "2. **Label Validation** — Validate SS3 labels ('H', 'E', 'C') and encode to integer IDs (0, 1, 2).",
        "3. **Sequence Encoding** — Encode each residue to an integer ID using the canonical alphabet mapping.",
        "4. " + ("**Truncation**" if config.truncation else "**No Truncation**") +
        (" — Sequences exceeding max_length are truncated from the end." if config.truncation else " — All sequences are within max_length."),
        "5. **Padding** — Sequences are padded to a uniform length of "
        f"{config.max_length} using {config.padding}-side padding with pad_id={PAD_ID}.",
        "6. **Attention Mask** — A binary attention mask is created (1 for real tokens, 0 for padding).",
        "7. **Batching** — Encoded sequences are grouped into batches of size " + str(config.batch_size) + ".",
    ]
    for step in steps:
        lines.append(step)
    lines.append("")

    # 2. Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for key, value in config.to_dict().items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    # 3. Sequence validation results
    lines.append("## Sequence Validation")
    lines.append("")
    lines.append(f"- **Input sequences**: {validation_stats.get('total_input_samples', '?')}")
    lines.append(f"- **Invalid sequences**: {validation_stats.get('invalid_sequences', 0)}")
    lines.append(f"- **Invalid residues**: {validation_stats.get('invalid_residue_count', 0)}")
    lines.append(f"- **Unknown labels**: {validation_stats.get('unknown_labels', 0)}")
    lines.append(f"- **Filtered samples**: {validation_stats.get('filtered_samples', 0)}")
    lines.append(f"- **Retained samples**: {validation_stats.get('retained_samples', '?')}")
    lines.append(f"- **Retention rate**: {validation_stats.get('retention_rate', '?')}%")
    lines.append("")

    # 4. Dataset overview
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|-------|")
    overview_keys = [
        ("num_sequences", "Number of sequences"),
        ("total_residues", "Total residues"),
        ("min_length", "Minimum length"),
        ("max_length", "Maximum length"),
        ("mean_length", "Mean length"),
        ("median_length", "Median length"),
        ("std_length", "Standard deviation"),
    ]
    for key, label in overview_keys:
        val = dataset_stats.get(key, "?" if key != "total_residues" else aa_stats.get("total_residues", "?"))
        if isinstance(val, float):
            val = f"{val:.2f}"
        lines.append(f"| {label} | {val} |")
    lines.append("")

    # 5. Amino-acid statistics
    lines.append("## Amino-Acid Composition")
    lines.append("")
    lines.append(f"- **Standard residues**: {aa_stats.get('total_residues', '?')}")
    lines.append(f"- **Unknown residues (X, B, Z, U, O)**: {aa_stats.get('unknown_residue_count', 0)}")
    lines.append(f"- **Invalid residues**: {aa_stats.get('invalid_residue_count', 0)}")
    lines.append("")
    lines.append("| Residue | Count | Percentage |")
    lines.append("|---------|-------|------------|")
    for aa in AA_ALPHABET:
        count = aa_stats.get("aa_counts", {}).get(aa, 0)
        pct = aa_stats.get("aa_percentages", {}).get(aa, 0.0)
        lines.append(f"| {aa} | {count:,} | {pct:.2f}% |")
    lines.append("")

    # 6. SS3 label statistics
    lines.append("## Secondary Structure (SS3) Statistics")
    lines.append("")
    lines.append("| Label | Count | Percentage |")
    lines.append("|-------|-------|------------|")
    label_dist = ss3_stats.get("label_distribution", {})
    for lbl_id in range(3):
        name = SS3_LABEL_NAMES.get(lbl_id, str(lbl_id))
        count = label_dist.get(name, label_dist.get(str(lbl_id), 0))
        total = sum(label_dist.values()) if label_dist else 0
        pct = round(100 * count / total, 2) if total else 0.0
        lines.append(f"| {name} | {count:,} | {pct}% |")
    lines.append("")
    lines.append(f"- **Imbalance ratio (max/min)**: {ss3_stats.get('imbalance_ratio', 'N/A')}")
    lines.append("")

    # 7. Padding statistics
    lines.append("## Padding Analysis")
    lines.append("")
    if padding_stats:
        lines.append(f"- **Max sequence length**: {padding_stats.get('max_length', '?')}")
        lines.append(f"- **Total padding tokens**: {padding_stats.get('total_padding_tokens', 0):,}")
        lines.append(f"- **Padding ratio**: {padding_stats.get('padding_ratio', 0.0)}")
        lines.append(f"- **Average padding per sequence**: {padding_stats.get('average_padding_per_sequence', 0.0)}")
        lines.append(f"- **Sequences with padding**: {padding_stats.get('sequences_with_padding', 0):,} "
                     f"({padding_stats.get('pct_sequences_with_padding', 0.0)}%)")
    lines.append("")

    # 8. Truncation statistics
    lines.append("## Truncation Analysis")
    lines.append("")
    if truncation_stats.get("truncation_enabled", False):
        if truncation_stats.get("num_truncated_sequences", 0) > 0:
            lines.append(f"- **Truncation enabled**: Yes")
            lines.append(f"- **Truncated sequences**: {truncation_stats.get('num_truncated_sequences', 0):,} "
                         f"({truncation_stats.get('pct_truncated', 0.0)}%)")
            lines.append(f"- **Total residues removed**: {truncation_stats.get('total_residues_removed', 0):,}")
            lines.append(f"- **Average truncation**: {truncation_stats.get('average_truncation_length', 0.0)} residues")
            lines.append(f"- **Maximum truncation**: {truncation_stats.get('max_truncation_length', 0)} residues")
        else:
            lines.append("- **Truncation enabled**: Yes — no sequences exceeded max_length")
    else:
        lines.append("- **Truncation**: Disabled")
    lines.append("")

    # 9. Figures
    lines.append("## Generated Figures")
    lines.append("")
    figures_dir = output_dir.parent / "figures"
    figure_list = [
        "sequence_length_histogram",
        "amino_acid_distribution",
        "ss3_class_distribution",
        "padding_distribution",
        "truncation_distribution",
    ]
    for fig_name in figure_list:
        fig_path_png = figures_dir / f"{fig_name}.png"
        if fig_path_png.exists():
            try:
                rel = fig_path_png.relative_to(output_dir)
                rel_str = str(rel.as_posix())
            except ValueError:
                rel_str = str(fig_path_png)
            lines.append(f"![{fig_name}]({rel_str})")
            lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by ProteinBERT preprocessing pipeline*")

    content = "\n".join(lines)
    report_path = output_dir / "preprocessing_report.md"
    report_path.write_text(content, encoding="utf-8")
    logger.info(f"Preprocessing report saved to {report_path}")

    return report_path


# ------------------------------------------------------------------
# Main preprocessing pipeline
# ------------------------------------------------------------------


class PreprocessingPipeline:
    """Complete ProteinBERT preprocessing pipeline.

    Orchestrates: data loading → validation → encoding → padding/truncation →
    batching → statistics collection → figure generation → report generation.
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], PreprocessingConfig]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize preprocessing pipeline.

        Args:
            config: Preprocessing configuration (dict or PreprocessingConfig).
            output_dir: Output directory for artifacts. If None, uses
                        outputs/experiments/<id>/preprocessing/.
        """
        if isinstance(config, dict):
            self.config = PreprocessingConfig(**config)
        elif config is None:
            self.config = PreprocessingConfig()
        else:
            self.config = config

        self.output_dir = Path(output_dir) if output_dir else Path("outputs/preprocessing")
        self.statistics_dir = self.output_dir / "statistics"
        self.figures_dir = self.output_dir / "figures"
        self.reports_dir = self.output_dir / "reports"

        self._stats: Dict[str, Any] = {}
        self._figures: Dict[str, Path] = {}

    def run(
        self,
        dataset: TapeSS3Dataset,
        splits: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the full preprocessing pipeline on a TAPE SS3 dataset.

        Args:
            dataset: TapeSS3Dataset instance (must have data loaded).
            splits: Which splits to process (default: all available).

        Returns:
            Dict with preprocessing results (stats, figures, report path).
        """
        ensure_dir(self.statistics_dir)
        ensure_dir(self.figures_dir)
        ensure_dir(self.reports_dir)

        # Get all splits
        if splits is None:
            try:
                raw_data = dataset.load_processed_data()
                splits = sorted(raw_data.keys())
            except FileNotFoundError:
                splits = ["train"]
                raw_data = {}

        logger.info(f"Starting preprocessing for splits: {splits}")

        all_sequences: List[str] = []
        all_labels: List[str] = []
        split_info: Dict[str, Dict[str, Any]] = {}

        for split_name in splits:
            try:
                data = dataset.load_processed_data()
                if split_name in data:
                    sd = data[split_name]
                    seqs = sd.get("sequence", sd.get("sequences", []))
                    lbls = sd.get("ss3_label", sd.get("labels", sd.get("targets", [])))
                else:
                    logger.warning(f"Split '{split_name}' not found in processed data, skipping")
                    continue
            except FileNotFoundError:
                logger.warning(f"No processed data found, using dataset directly (split: {split_name})")
                seqs = dataset.get_sequences()
                lbls = dataset.get_targets()

            # Convert labels from int back to strings if needed
            str_labels: List[str] = []
            for lbl in lbls:
                if isinstance(lbl, int):
                    from .constants import ID_TO_LABEL
                    str_labels.append(ID_TO_LABEL.get(lbl, "C"))
                else:
                    str_labels.append(str(lbl))

            split_info[split_name] = {
                "sequences": seqs,
                "labels": str_labels,
                "lengths": [len(s) for s in seqs],
            }
            all_sequences.extend(seqs)
            all_labels.extend(str_labels)

        n_total = len(all_sequences)

        # ------------------------------------------------------------------
        # Stage 1: Validation
        # ------------------------------------------------------------------
        logger.info("Stage 1/5: Validating sequences and labels...")
        invalid_seqs = validate_sequences(
            all_sequences,
            extended_alphabet=self.config.extended_alphabet,
            raise_on_invalid=False,
        )
        invalid_residue_count = sum(
            sum(1 for c in s.upper() if c not in AA_ALPHABET and c not in ("X", "B", "Z", "U", "O"))
            for s in all_sequences
        )

        unknown_label_indices = []
        for i, lbl in enumerate(all_labels):
            if lbl.upper().strip() not in LABEL_TO_ID:
                unknown_label_indices.append(i)

        validation_stats = compute_validation_statistics(
            total_input=n_total,
            invalid_sequences=invalid_seqs,
            invalid_residues=invalid_residue_count,
            unknown_labels=unknown_label_indices,
        )
        self._stats["validation"] = validation_stats

        # Filter out invalid samples
        invalid_indices = {idx for idx, _ in invalid_seqs} | set(unknown_label_indices)
        valid_indices = [i for i in range(n_total) if i not in invalid_indices]

        valid_sequences = [all_sequences[i] for i in valid_indices]
        valid_labels = [all_labels[i] for i in valid_indices]
        n_valid = len(valid_sequences)

        logger.info(f"  Retained {n_valid}/{n_total} samples after validation")

        # ------------------------------------------------------------------
        # Stage 2: Dataset statistics
        # ------------------------------------------------------------------
        logger.info("Stage 2/5: Computing dataset statistics...")
        original_lengths = np.array([len(s) for s in valid_sequences], dtype=np.int32)

        dataset_stats: Dict[str, Any] = {
            "num_sequences": n_valid,
            "total_residues": int(original_lengths.sum()),
            "min_length": int(original_lengths.min()) if n_valid else 0,
            "max_length": int(original_lengths.max()) if n_valid else 0,
            "mean_length": round(float(original_lengths.mean()), 2) if n_valid else 0.0,
            "median_length": round(float(np.median(original_lengths)), 2) if n_valid else 0.0,
            "std_length": round(float(original_lengths.std()), 2) if n_valid else 0.0,
        }

        # Quartiles
        if n_valid > 0:
            q1, q2, q3 = np.percentile(original_lengths, [25, 50, 75])
            dataset_stats["q1_length"] = round(float(q1), 2)
            dataset_stats["q2_length"] = round(float(q2), 2)
            dataset_stats["q3_length"] = round(float(q3), 2)

        self._stats["dataset"] = dataset_stats
        save_statistics_csv(dataset_stats, self.statistics_dir / "dataset_summary.csv")

        # ------------------------------------------------------------------
        # Stage 3: Amino-acid statistics
        # ------------------------------------------------------------------
        logger.info("Stage 3/5: Computing amino-acid statistics...")
        aa_stats = compute_aa_statistics(valid_sequences)
        self._stats["amino_acid"] = aa_stats

        aa_rows = []
        for aa in AA_ALPHABET:
            aa_rows.append({
                "residue": aa,
                "count": aa_stats["aa_counts"][aa],
                "percentage": aa_stats["aa_percentages"][aa],
            })
        aa_rows.append({
            "residue": "UNKNOWN",
            "count": aa_stats["unknown_residue_count"],
            "percentage": round(100 * aa_stats["unknown_residue_count"] / aa_stats["total_residues"], 4) if aa_stats["total_residues"] else 0.0,
        })
        save_csv(pd.DataFrame(aa_rows), self.statistics_dir / "amino_acid_frequencies.csv")

        # ------------------------------------------------------------------
        # Stage 4: SS3 statistics
        # ------------------------------------------------------------------
        logger.info("Stage 4/5: Computing SS3 statistics...")

        encoded_labels = np.array([LABEL_TO_ID[lbl.upper().strip()] for lbl in valid_labels], dtype=np.int32)
        unique, counts = np.unique(encoded_labels, return_counts=True)
        ss3_label_dist = {}
        for u, c in zip(unique, counts):
            ss3_label_dist[SS3_LABEL_NAMES[int(u)]] = int(c)
        total_labels = len(valid_labels)

        imbalance_ratio = "N/A"
        if len(counts) > 1:
            imbalance_ratio = round(float(counts.max() / counts.min()), 2)

        ss3_stats = {
            "num_samples": total_labels,
            "label_distribution": ss3_label_dist,
            "imbalance_ratio": imbalance_ratio,
        }
        self._stats["ss3"] = ss3_stats

        ss3_rows = []
        for name, count in ss3_label_dist.items():
            ss3_rows.append({
                "label": name,
                "count": count,
                "percentage": round(100 * count / total_labels, 2),
            })
        save_csv(pd.DataFrame(ss3_rows), self.statistics_dir / "ss3_label_distribution.csv")

        # ------------------------------------------------------------------
        # Stage 5: Encoding, padding, truncation
        # ------------------------------------------------------------------
        logger.info("Stage 5/5: Encoding, padding, and batching...")

        encoded = [encode_sequence(s, self.config.extended_alphabet, self.config.unknown_token)
                   for s in valid_sequences]
        encoded_lengths = np.array([len(e) for e in encoded], dtype=np.int32)

        # Truncation statistics (before truncation)
        truncation_stats = compute_truncation_statistics(
            encoded_lengths, self.config.max_length
        )
        self._stats["truncation"] = truncation_stats

        # Truncate
        if self.config.truncation:
            encoded = truncate_sequences(encoded, self.config.max_length)

        # Padding statistics
        padded_lengths = np.array([len(e) for e in encoded], dtype=np.int32)
        padding_stats = compute_padding_statistics(padded_lengths, self.config.max_length)
        self._stats["padding"] = padding_stats

        # Pad
        input_ids = pad_sequences(encoded, self.config.max_length, self.config.padding)
        attention_mask = create_attention_mask(input_ids)

        # ------------------------------------------------------------------
        # Save artifacts
        # ------------------------------------------------------------------

        # Save encoded dataset summary
        encoded_summary = {
            "num_sequences": n_valid,
            "max_length": self.config.max_length,
            "shape": list(input_ids.shape),
            "dtype": str(input_ids.dtype),
        }
        save_json(encoded_summary, self.statistics_dir / "encoded_summary.json")

        # Save all stats as JSON
        all_stats = {
            "preprocessing_config": self.config.to_dict(),
            "validation": validation_stats,
            "dataset": dataset_stats,
            "amino_acid": aa_stats,
            "ss3": ss3_stats,
            "padding": padding_stats,
            "truncation": truncation_stats,
        }
        save_json(all_stats, self.statistics_dir / "preprocessing_stats.json")
        self._stats["all"] = all_stats

        # ------------------------------------------------------------------
        # Generate figures
        # ------------------------------------------------------------------
        logger.info("Generating figures...")

        if n_valid > 0:
            self._figures["sequence_length_histogram"] = (
                import_plot_length_histogram(valid_sequences, self.figures_dir / "sequence_length_histogram")
            )

            self._figures["amino_acid_distribution"] = _plot_aa_distribution(
                aa_stats, self.figures_dir / "amino_acid_distribution"
            )

            if len(encoded_labels) > 0:
                self._figures["ss3_class_distribution"] = (
                    import_plot_class_distribution(
                        [int(encoded_labels[i]) for i in range(len(encoded_labels))],
                        self.figures_dir / "ss3_class_distribution",
                        label_names={0: "Helix (H)", 1: "Strand (E)", 2: "Coil (C)"},
                    )
                )

            self._figures["padding_distribution"] = _plot_padding_distribution(
                padded_lengths, self.config.max_length, self.figures_dir / "padding_distribution"
            )

            self._figures["truncation_distribution"] = _plot_truncation_distribution(
                encoded_lengths, self.config.max_length, self.figures_dir / "truncation_distribution"
            )

        # ------------------------------------------------------------------
        # Generate report
        # ------------------------------------------------------------------
        logger.info("Generating preprocessing report...")

        report_path = _generate_preprocessing_report(
            config=self.config,
            dataset_stats=dataset_stats,
            aa_stats=aa_stats,
            ss3_stats=ss3_stats,
            padding_stats=padding_stats,
            truncation_stats=truncation_stats,
            validation_stats=validation_stats,
            output_dir=self.reports_dir,
        )

        result = {
            "stats": all_stats,
            "figures": {k: str(v) for k, v in self._figures.items()},
            "report": str(report_path),
            "statistics_dir": str(self.statistics_dir),
            "figures_dir": str(self.figures_dir),
            "encoded_shape": list(input_ids.shape),
            "num_valid_sequences": n_valid,
            "num_filtered": n_total - n_valid,
        }

        logger.info(f"Preprocessing complete: {n_valid} sequences, figures in {self.figures_dir}")
        return result

    def run_on_sequences(
        self,
        sequences: List[str],
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the full preprocessing pipeline on raw sequences directly.

        Args:
            sequences: List of raw protein sequence strings.
            labels: Optional list of SS3 labels ('H', 'E', 'C').

        Returns:
            Dict with preprocessing results (stats, figures, report path).
        """
        from ...datasets.tape_ss3 import SS3_LABEL_NAMES
        import pandas as pd

        ensure_dir(self.statistics_dir)
        ensure_dir(self.figures_dir)
        ensure_dir(self.reports_dir)

        all_sequences = list(sequences)
        all_labels = list(labels) if labels is not None else []
        n_total = len(all_sequences)

        # ------------------------------------------------------------------
        # Stage 1: Validation
        # ------------------------------------------------------------------
        logger.info("Stage 1/5: Validating sequences and labels...")
        invalid_seqs = validate_sequences(
            all_sequences,
            extended_alphabet=self.config.extended_alphabet,
            raise_on_invalid=False,
        )
        invalid_residue_count = sum(
            sum(1 for c in s.upper() if c not in AA_ALPHABET and c not in ("X", "B", "Z", "U", "O"))
            for s in all_sequences
        )

        unknown_label_indices = []
        for i, lbl in enumerate(all_labels):
            if lbl.upper().strip() not in LABEL_TO_ID:
                unknown_label_indices.append(i)

        validation_stats = compute_validation_statistics(
            total_input=n_total,
            invalid_sequences=invalid_seqs,
            invalid_residues=invalid_residue_count,
            unknown_labels=unknown_label_indices,
        )
        self._stats["validation"] = validation_stats

        # Filter out invalid samples
        invalid_indices = {idx for idx, _ in invalid_seqs} | set(unknown_label_indices)
        valid_indices = [i for i in range(n_total) if i not in invalid_indices]

        valid_sequences = [all_sequences[i] for i in valid_indices]
        valid_labels = [all_labels[i] for i in valid_indices] if all_labels else []
        n_valid = len(valid_sequences)

        logger.info(f"  Retained {n_valid}/{n_total} samples after validation")

        # ------------------------------------------------------------------
        # Stage 2: Dataset statistics
        # ------------------------------------------------------------------
        logger.info("Stage 2/5: Computing dataset statistics...")
        original_lengths = np.array([len(s) for s in valid_sequences], dtype=np.int32)

        dataset_stats: Dict[str, Any] = {
            "num_sequences": n_valid,
            "total_residues": int(original_lengths.sum()),
            "min_length": int(original_lengths.min()) if n_valid else 0,
            "max_length": int(original_lengths.max()) if n_valid else 0,
            "mean_length": round(float(original_lengths.mean()), 2) if n_valid else 0.0,
            "median_length": round(float(np.median(original_lengths)), 2) if n_valid else 0.0,
            "std_length": round(float(original_lengths.std()), 2) if n_valid else 0.0,
        }

        if n_valid > 0:
            q1, q2, q3 = np.percentile(original_lengths, [25, 50, 75])
            dataset_stats["q1_length"] = round(float(q1), 2)
            dataset_stats["q2_length"] = round(float(q2), 2)
            dataset_stats["q3_length"] = round(float(q3), 2)

        self._stats["dataset"] = dataset_stats
        save_statistics_csv(dataset_stats, self.statistics_dir / "dataset_summary.csv")

        # ------------------------------------------------------------------
        # Stage 3: Amino-acid statistics
        # ------------------------------------------------------------------
        logger.info("Stage 3/5: Computing amino-acid statistics...")
        aa_stats = compute_aa_statistics(valid_sequences)
        self._stats["amino_acid"] = aa_stats

        aa_rows = []
        for aa in AA_ALPHABET:
            aa_rows.append({
                "residue": aa,
                "count": aa_stats["aa_counts"][aa],
                "percentage": aa_stats["aa_percentages"][aa],
            })
        aa_rows.append({
            "residue": "UNKNOWN",
            "count": aa_stats["unknown_residue_count"],
            "percentage": round(100 * aa_stats["unknown_residue_count"] / aa_stats["total_residues"], 4) if aa_stats["total_residues"] else 0.0,
        })
        save_csv(pd.DataFrame(aa_rows), self.statistics_dir / "amino_acid_frequencies.csv")

        # ------------------------------------------------------------------
        # Stage 4: SS3 statistics
        # ------------------------------------------------------------------
        logger.info("Stage 4/5: Computing SS3 statistics...")

        encoded_labels = np.array([], dtype=np.int32)
        ss3_stats = {"num_samples": 0, "label_distribution": {}, "imbalance_ratio": "N/A"}
        if valid_labels:
            encoded_labels = np.array([LABEL_TO_ID[lbl.upper().strip()] for lbl in valid_labels], dtype=np.int32)
            unique, counts = np.unique(encoded_labels, return_counts=True)
            ss3_label_dist = {}
            for u, c in zip(unique, counts):
                ss3_label_dist[SS3_LABEL_NAMES[int(u)]] = int(c)
            total_labels = len(valid_labels)
            imbalance_ratio = "N/A"
            if len(counts) > 1:
                imbalance_ratio = round(float(counts.max() / counts.min()), 2)
            ss3_stats = {
                "num_samples": total_labels,
                "label_distribution": ss3_label_dist,
                "imbalance_ratio": imbalance_ratio,
            }

        self._stats["ss3"] = ss3_stats

        ss3_rows = []
        for name, count in ss3_stats.get("label_distribution", {}).items():
            ss3_rows.append({
                "label": name,
                "count": count,
                "percentage": round(100 * count / total_labels, 2) if "total_labels" in locals() and total_labels else 0.0,
            })
        save_csv(pd.DataFrame(ss3_rows), self.statistics_dir / "ss3_label_distribution.csv")

        # ------------------------------------------------------------------
        # Stage 5: Encoding, padding, truncation
        # ------------------------------------------------------------------
        logger.info("Stage 5/5: Encoding, padding, and batching...")

        encoded = [encode_sequence(s, self.config.extended_alphabet, self.config.unknown_token)
                   for s in valid_sequences]
        encoded_lengths = np.array([len(e) for e in encoded], dtype=np.int32)

        truncation_stats = compute_truncation_statistics(encoded_lengths, self.config.max_length)
        self._stats["truncation"] = truncation_stats

        if self.config.truncation:
            encoded = truncate_sequences(encoded, self.config.max_length)

        padded_lengths = np.array([len(e) for e in encoded], dtype=np.int32)
        padding_stats = compute_padding_statistics(padded_lengths, self.config.max_length)
        self._stats["padding"] = padding_stats

        input_ids = pad_sequences(encoded, self.config.max_length, self.config.padding)
        attention_mask = create_attention_mask(input_ids)

        # ------------------------------------------------------------------
        # Save artifacts
        # ------------------------------------------------------------------
        encoded_summary = {
            "num_sequences": n_valid,
            "max_length": self.config.max_length,
            "shape": list(input_ids.shape),
            "dtype": str(input_ids.dtype),
        }
        save_json(encoded_summary, self.statistics_dir / "encoded_summary.json")

        all_stats = {
            "preprocessing_config": self.config.to_dict(),
            "validation": validation_stats,
            "dataset": dataset_stats,
            "amino_acid": aa_stats,
            "ss3": ss3_stats,
            "padding": padding_stats,
            "truncation": truncation_stats,
        }
        save_json(all_stats, self.statistics_dir / "preprocessing_stats.json")
        self._stats["all"] = all_stats

        # ------------------------------------------------------------------
        # Generate figures
        # ------------------------------------------------------------------
        logger.info("Generating figures...")

        if n_valid > 0:
            self._figures["sequence_length_histogram"] = (
                import_plot_length_histogram(valid_sequences, self.figures_dir / "sequence_length_histogram")
            )

            self._figures["amino_acid_distribution"] = _plot_aa_distribution(
                aa_stats, self.figures_dir / "amino_acid_distribution"
            )

            if len(encoded_labels) > 0:
                self._figures["ss3_class_distribution"] = (
                    import_plot_class_distribution(
                        [int(encoded_labels[i]) for i in range(len(encoded_labels))],
                        self.figures_dir / "ss3_class_distribution",
                        label_names={0: "Helix (H)", 1: "Strand (E)", 2: "Coil (C)"},
                    )
                )

            self._figures["padding_distribution"] = _plot_padding_distribution(
                padded_lengths, self.config.max_length, self.figures_dir / "padding_distribution"
            )

            self._figures["truncation_distribution"] = _plot_truncation_distribution(
                encoded_lengths, self.config.max_length, self.figures_dir / "truncation_distribution"
            )

        # ------------------------------------------------------------------
        # Generate report
        # ------------------------------------------------------------------
        logger.info("Generating preprocessing report...")

        report_path = _generate_preprocessing_report(
            config=self.config,
            dataset_stats=dataset_stats,
            aa_stats=aa_stats,
            ss3_stats=ss3_stats,
            padding_stats=padding_stats,
            truncation_stats=truncation_stats,
            validation_stats=validation_stats,
            output_dir=self.reports_dir,
        )

        result = {
            "stats": all_stats,
            "figures": {k: str(v) for k, v in self._figures.items()},
            "report": str(report_path),
            "statistics_dir": str(self.statistics_dir),
            "figures_dir": str(self.figures_dir),
            "encoded_shape": list(input_ids.shape),
            "num_valid_sequences": n_valid,
            "num_filtered": n_total - n_valid,
        }

        logger.info(f"Preprocessing complete: {n_valid} sequences, figures in {self.figures_dir}")
        return result


def import_plot_length_histogram(
    sequences: List[str],
    output_path: Union[str, Path],
    split_name: str = "all",
    dpi: int = 300,
) -> Path:
    """Wrapper around visualization.plot_length_histogram to avoid import-time Agg."""
    from ...visualization.visualization import plot_length_histogram as _plh
    return _plh(sequences, output_path, split_name, dpi)


def import_plot_class_distribution(
    labels: List[Any],
    output_path: Union[str, Path],
    label_names=None,
    split_name: str = "all",
    dpi: int = 300,
) -> Path:
    """Wrapper around visualization.plot_class_distribution to avoid import-time Agg."""
    from ...visualization.visualization import plot_class_distribution as _pcd
    return _pcd(labels, output_path, label_names, split_name, dpi)
