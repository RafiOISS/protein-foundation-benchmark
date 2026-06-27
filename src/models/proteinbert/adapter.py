"""ProteinBERT data adapter — framework-to-model tensor bridge.

Responsibilities:
  - convert preprocessed batch dicts into ProteinBERT-ready format
  - validate all tensor shapes, dtypes, masks, and batch dimensions
  - provide deterministic batch iteration
  - generate publication-quality adapter metadata
  - be the only layer aware of ProteinBERT's expected input format

No TensorFlow imports at module level. All TF logic is lazy.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from ...utils.logging import get_logger
from ...utils.io import ensure_dir, save_csv, save_json, write_text


logger = get_logger(__name__)


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------


class AdapterError(Exception):
    """Raised on adapter-level errors (configuration, state)."""


class TensorValidationError(ValueError):
    """Raised when a tensor fails shape, dtype, or mask validation."""


# ------------------------------------------------------------------
# Expected constants
# ------------------------------------------------------------------

EXPECTED_INPUT_NDIM: int = 2
EXPECTED_LABEL_NDIM: int = 1
EXPECTED_MASK_NDIM: int = 2
SUPPORTED_DTYPES: Tuple[np.dtype, ...] = (np.int32, np.int64, np.float32, np.float64)


# ------------------------------------------------------------------
# TensorValidator
# ------------------------------------------------------------------


class TensorValidator:
    """Validator for batch tensors.

    Each method raises TensorValidationError on failure with a
    descriptive message. No silent correction.
    """

    @staticmethod
    def validate_input_ids(
        tensor: np.ndarray,
        batch_idx: int,
        expected_ndim: int = EXPECTED_INPUT_NDIM,
        allowed_dtypes: Tuple[np.dtype, ...] = (np.int32, np.int64),
    ) -> None:
        if not isinstance(tensor, np.ndarray):
            raise TensorValidationError(
                f"Batch {batch_idx}: input_ids must be np.ndarray, got {type(tensor).__name__}"
            )
        if tensor.ndim != expected_ndim:
            raise TensorValidationError(
                f"Batch {batch_idx}: input_ids expected ndim={expected_ndim}, got ndim={tensor.ndim} "
                f"(shape={tensor.shape})"
            )
        if tensor.dtype not in allowed_dtypes:
            raise TensorValidationError(
                f"Batch {batch_idx}: input_ids dtype {tensor.dtype} not in {allowed_dtypes}"
            )
        if tensor.size == 0:
            raise TensorValidationError(
                f"Batch {batch_idx}: input_ids is empty (shape={tensor.shape})"
            )

    @staticmethod
    def validate_attention_mask(
        tensor: np.ndarray,
        batch_idx: int,
        expected_shape: Tuple[int, ...],
        allowed_dtypes: Tuple[np.dtype, ...] = (np.int32, np.int64, np.float32, np.float64),
    ) -> None:
        if not isinstance(tensor, np.ndarray):
            raise TensorValidationError(
                f"Batch {batch_idx}: attention_mask must be np.ndarray, got {type(tensor).__name__}"
            )
        if tensor.shape != expected_shape:
            raise TensorValidationError(
                f"Batch {batch_idx}: attention_mask shape {tensor.shape} != expected {expected_shape}"
            )
        if tensor.dtype not in allowed_dtypes:
            raise TensorValidationError(
                f"Batch {batch_idx}: attention_mask dtype {tensor.dtype} not in {allowed_dtypes}"
            )
        unique_vals = set(np.unique(tensor))
        if not unique_vals.issubset({0, 1}):
            raise TensorValidationError(
                f"Batch {batch_idx}: attention_mask contains values {unique_vals - {0, 1}} "
                f"outside {{0, 1}}"
            )

    @staticmethod
    def validate_labels(
        tensor: np.ndarray,
        batch_idx: int,
        num_samples: int,
        allowed_dtypes: Tuple[np.dtype, ...] = (np.int32, np.int64),
    ) -> None:
        if not isinstance(tensor, np.ndarray):
            raise TensorValidationError(
                f"Batch {batch_idx}: labels must be np.ndarray, got {type(tensor).__name__}"
            )
        if tensor.ndim != 1:
            raise TensorValidationError(
                f"Batch {batch_idx}: labels expected 1D array, got ndim={tensor.ndim}"
            )
        if tensor.shape[0] != num_samples:
            raise TensorValidationError(
                f"Batch {batch_idx}: labels length {tensor.shape[0]} != num_samples {num_samples}"
            )
        if tensor.dtype not in allowed_dtypes:
            raise TensorValidationError(
                f"Batch {batch_idx}: labels dtype {tensor.dtype} not in {allowed_dtypes}"
            )

    @staticmethod
    def validate_lengths(
        tensor: np.ndarray,
        batch_idx: int,
        num_samples: int,
    ) -> None:
        if not isinstance(tensor, np.ndarray):
            raise TensorValidationError(
                f"Batch {batch_idx}: lengths must be np.ndarray, got {type(tensor).__name__}"
            )
        if tensor.ndim != 1:
            raise TensorValidationError(
                f"Batch {batch_idx}: lengths expected 1D array, got ndim={tensor.ndim}"
            )
        if tensor.shape[0] != num_samples:
            raise TensorValidationError(
                f"Batch {batch_idx}: lengths length {tensor.shape[0]} != num_samples {num_samples}"
            )
        if tensor.dtype not in (np.int32, np.int64):
            raise TensorValidationError(
                f"Batch {batch_idx}: lengths dtype {tensor.dtype}, expected int32 or int64"
            )

    @staticmethod
    def validate_batch_size_consistency(
        batches: List[Dict[str, np.ndarray]],
    ) -> int:
        if not batches:
            raise TensorValidationError("Batch list is empty")
        if "input_ids" not in batches[0]:
            raise TensorValidationError(
                f"Batch 0: missing required key 'input_ids'"
            )
        first_batch_size = batches[0]["input_ids"].shape[0]
        for i, batch in enumerate(batches):
            batch_len = batch["input_ids"].shape[0]
            # Within each batch, all tensors must share the same batch dim
            for key, tensor in batch.items():
                if tensor.ndim == 0:
                    continue
                if tensor.shape[0] != batch_len:
                    raise TensorValidationError(
                        f"Batch {i}: {key} first dim {tensor.shape[0]} "
                        f"!= batch_size {batch_len} (defined by input_ids)"
                    )
        return first_batch_size

    @staticmethod
    def validate_seq_dim_consistency(
        batches: List[Dict[str, np.ndarray]],
    ) -> int:
        if not batches:
            raise TensorValidationError("Batch list is empty")
        first_seq_len = batches[0]["input_ids"].shape[1]
        for i, batch in enumerate(batches):
            for key in ("input_ids", "attention_mask"):
                if batch[key].shape[1] != first_seq_len:
                    raise TensorValidationError(
                        f"Batch {i}: {key} seq dim {batch[key].shape[1]} "
                        f"!= expected {first_seq_len}"
                    )
        return first_seq_len

    @staticmethod
    def validate_mask_consistency(
        batches: List[Dict[str, np.ndarray]],
    ) -> None:
        for i, batch in enumerate(batches):
            ids = batch["input_ids"]
            mask = batch["attention_mask"]
            pad_id = 25  # PAD_ID from constants
            expected_mask = (ids != pad_id).astype(mask.dtype)
            if not np.array_equal(mask, expected_mask):
                mismatch = int(np.sum(mask != expected_mask))
                raise TensorValidationError(
                    f"Batch {i}: attention_mask inconsistent with input_ids "
                    f"({mismatch} positions differ)"
                )

    @staticmethod
    def validate_all(
        batches: List[Dict[str, np.ndarray]],
        require_labels: bool = False,
    ) -> Dict[str, Any]:
        """Run all validations on a batch list. Returns summary statistics."""
        if not batches:
            raise TensorValidationError("Batch list is empty")

        batch_size = TensorValidator.validate_batch_size_consistency(batches)
        seq_len = TensorValidator.validate_seq_dim_consistency(batches)

        total_samples = 0
        total_padding = 0
        total_tokens = 0
        num_positions_differ = 0

        for i, batch in enumerate(batches):
            required_keys = ["input_ids", "attention_mask"]
            for key in required_keys:
                if key not in batch:
                    raise TensorValidationError(
                        f"Batch {i}: missing required key '{key}'"
                    )

            ids = batch["input_ids"]
            mask = batch["attention_mask"]

            TensorValidator.validate_input_ids(ids, i)
            TensorValidator.validate_attention_mask(mask, i, ids.shape)

            if "labels" in batch:
                TensorValidator.validate_labels(batch["labels"], i, ids.shape[0])
            elif require_labels:
                raise TensorValidationError(f"Batch {i}: missing 'labels' key")

            if "lengths" in batch:
                TensorValidator.validate_lengths(batch["lengths"], i, ids.shape[0])

            total_samples += ids.shape[0]
            total_tokens += ids.size
            pad_count = int(np.sum(ids == 25))  # PAD_ID
            total_padding += pad_count

            expected_mask = (ids != 25).astype(mask.dtype)
            num_positions_differ += int(np.sum(mask != expected_mask))

        # Validate mask consistency across all batches
        TensorValidator.validate_mask_consistency(batches)

        stats = {
            "num_batches": len(batches),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "total_samples": total_samples,
            "total_tokens": total_tokens,
            "total_padding_tokens": total_padding,
            "padding_ratio": round(total_padding / total_tokens, 4) if total_tokens else 0.0,
            "mask_errors": num_positions_differ,
            "all_masks_consistent": num_positions_differ == 0,
        }

        if num_positions_differ > 0:
            logger.warning(f"Found {num_positions_differ} mask inconsistencies across all batches")

        return stats


# ------------------------------------------------------------------
# AdapterMetadata
# ------------------------------------------------------------------


@dataclass
class AdapterMetadata:
    """Publication-quality metadata for the adapter run.

    Every field is recorded for reproducibility and the implementation
    details section of the paper.
    """
    # Configuration
    batch_size: int = 0
    max_seq_len: int = 0
    padding_strategy: str = "right"
    truncation_enabled: bool = True

    # Tensor summary
    num_batches: int = 0
    num_samples: int = 0
    seq_len: int = 0
    has_labels: bool = False
    label_dim: int = 0

    # Token statistics
    total_tokens: int = 0
    total_padding_tokens: int = 0
    padding_ratio: float = 0.0

    # Mask statistics
    mask_errors: int = 0
    all_masks_consistent: bool = True

    # Input dtypes
    input_dtype: str = ""
    label_dtype: str = ""
    mask_dtype: str = ""

    # Timing
    created_at: str = ""

    # Additional info
    adapter_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_batches(
        cls,
        batches: List[Dict[str, np.ndarray]],
        config: Optional[Dict[str, Any]] = None,
    ) -> "AdapterMetadata":
        metadata = cls()
        metadata.created_at = datetime.now().isoformat()

        if not batches:
            return metadata

        config = config or {}
        metadata.batch_size = batches[0]["input_ids"].shape[0]
        metadata.max_seq_len = config.get("max_length", batches[0]["input_ids"].shape[1])
        metadata.seq_len = batches[0]["input_ids"].shape[1]
        metadata.num_batches = len(batches)
        metadata.num_samples = sum(b["input_ids"].shape[0] for b in batches)
        metadata.has_labels = "labels" in batches[0]
        metadata.input_dtype = str(batches[0]["input_ids"].dtype)
        metadata.mask_dtype = str(batches[0]["attention_mask"].dtype)

        if metadata.has_labels:
            metadata.label_dtype = str(batches[0]["labels"].dtype)
            metadata.label_dim = 1  # 1D per-sample labels

        total_tokens = sum(b["input_ids"].size for b in batches)
        total_padding = sum(int(np.sum(b["input_ids"] == 25)) for b in batches)  # PAD_ID
        metadata.total_tokens = total_tokens
        metadata.total_padding_tokens = total_padding
        metadata.padding_ratio = round(total_padding / total_tokens, 4) if total_tokens else 0.0

        mask_errors = 0
        for batch in batches:
            expected = (batch["input_ids"] != 25).astype(batch["attention_mask"].dtype)
            mask_errors += int(np.sum(batch["attention_mask"] != expected))
        metadata.mask_errors = mask_errors
        metadata.all_masks_consistent = mask_errors == 0

        return metadata


# ------------------------------------------------------------------
# Report and statistics generation
# ------------------------------------------------------------------


def _generate_adapter_report(
    metadata: AdapterMetadata,
    config: Dict[str, Any],
    validation_stats: Dict[str, Any],
    tensor_shapes: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> Path:
    """Generate a publication-quality adapter report in Markdown.

    Args:
        metadata: AdapterMetadata instance with batch statistics.
        config: Adapter configuration dict.
        validation_stats: Validation statistics from TensorValidator.
        tensor_shapes: Per-batch tensor shape list.
        output_path: Output path for the report file.

    Returns:
        Path to the saved report.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    lines: List[str] = []
    lines.append("# Adapter Report")
    lines.append("")
    lines.append(f"- **Generated**: {metadata.created_at or datetime.now().isoformat()}")
    lines.append(f"- **Adapter Version**: {metadata.adapter_version}")
    lines.append(f"- **Model**: ProteinBERT")
    lines.append("")

    # 1. Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for key, value in sorted(config.items()):
        lines.append(f"| {key} | {value} |")
    lines.append("")

    # 2. Dataset overview
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Number of batches | {metadata.num_batches} |")
    lines.append(f"| Total samples | {metadata.num_samples} |")
    lines.append(f"| Batch size | {metadata.batch_size} |")
    lines.append(f"| Sequence length | {metadata.seq_len} |")
    lines.append(f"| Max sequence length | {metadata.max_seq_len} |")
    lines.append(f"| Has labels | {metadata.has_labels} |")
    if metadata.has_labels:
        lines.append(f"| Label dimension | {metadata.label_dim} |")
        lines.append(f"| Label dtype | {metadata.label_dtype} |")
    lines.append("")

    # 3. Tensor shapes
    lines.append("## Tensor Shapes")
    lines.append("")
    lines.append("| Batch | input_ids | attention_mask | labels | lengths |")
    lines.append("|-------|-----------|----------------|--------|---------|")
    for i, shapes in enumerate(tensor_shapes):
        ids_shape = str(shapes.get("input_ids", "N/A"))
        mask_shape = str(shapes.get("attention_mask", "N/A"))
        lbl_shape = str(shapes.get("labels", "N/A"))
        len_shape = str(shapes.get("lengths", "N/A"))
        lines.append(f"| {i} | {ids_shape} | {mask_shape} | {lbl_shape} | {len_shape} |")
    lines.append("")

    # 4. Input dtypes
    lines.append("## Input Data Types")
    lines.append("")
    lines.append(f"- **input_ids dtype**: {metadata.input_dtype or 'N/A'}")
    lines.append(f"- **attention_mask dtype**: {metadata.mask_dtype or 'N/A'}")
    lines.append(f"- **labels dtype**: {metadata.label_dtype or 'N/A'}")
    lines.append("")

    # 5. Token statistics
    lines.append("## Token Statistics")
    lines.append("")
    lines.append(f"- **Total tokens**: {metadata.total_tokens:,}")
    lines.append(f"- **Total padding tokens**: {metadata.total_padding_tokens:,}")
    lines.append(f"- **Padding ratio**: {metadata.padding_ratio:.4f}")
    lines.append(f"- **Average padding per batch**: "
                 f"{metadata.total_padding_tokens / metadata.num_batches:.1f}" if metadata.num_batches else "0")
    lines.append("")

    # 6. Mask validation
    lines.append("## Mask Validation")
    lines.append("")
    lines.append(f"- **All masks consistent**: {metadata.all_masks_consistent}")
    lines.append(f"- **Mask errors**: {metadata.mask_errors}")
    if metadata.mask_errors > 0:
        lines.append("")
        lines.append("> **Warning**: Attention masks are inconsistent with input_ids. "
                     "Investigate before training.")
    lines.append("")

    # 7. Validation statistics
    lines.append("## Validation Statistics")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|-------|")
    for key, value in sorted(validation_stats.items()):
        if isinstance(value, bool):
            lines.append(f"| {key} | {value} |")
        elif isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.append("")

    # 8. Batch summary table
    lines.append("## Batch Summary")
    lines.append("")
    lines.append("| Batch | Samples | Tokens | Padding | Padding % | Mask Errors |")
    lines.append("|-------|---------|--------|---------|-----------|-------------|")
    for i, shapes in enumerate(tensor_shapes):
        batch_samples = shapes.get("batch_samples", 0)
        batch_tokens = shapes.get("total_tokens", 0)
        batch_padding = shapes.get("padding_tokens", 0)
        batch_padding_pct = shapes.get("padding_pct", 0.0)
        batch_mask_errors = shapes.get("mask_errors", 0)
        lines.append(
            f"| {i} | {batch_samples} | {batch_tokens:,} | {batch_padding:,} "
            f"| {batch_padding_pct:.1f}% | {batch_mask_errors} |"
        )
    lines.append("")

    lines.append("---")
    lines.append("*Report generated by ProteinBERT adapter*")

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    logger.info(f"Adapter report saved to {output_path}")

    return output_path


def _generate_tensor_shapes(
    batches: List[Dict[str, np.ndarray]],
    output_path: Union[str, Path],
) -> Path:
    """Generate a per-batch tensor shape manifest JSON.

    Args:
        batches: List of batch dicts.
        output_path: Output path for the JSON file.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    shapes: List[Dict[str, Any]] = []
    for i, batch in enumerate(batches):
        entry: Dict[str, Any] = {"batch_index": i}
        for key, tensor in batch.items():
            entry[key] = {
                "shape": list(tensor.shape),
                "ndim": tensor.ndim,
                "dtype": str(tensor.dtype),
                "size": int(tensor.size),
            }
        entry["batch_samples"] = int(batches[i]["input_ids"].shape[0])
        entry["total_tokens"] = int(batches[i]["input_ids"].size)
        pad_count = int(np.sum(batches[i]["input_ids"] == 25))
        entry["padding_tokens"] = pad_count
        entry["padding_pct"] = round(100 * pad_count / batches[i]["input_ids"].size, 2)
        if "labels" in batch:
            unique, counts = np.unique(batch["labels"], return_counts=True)
            entry["label_distribution"] = {
                str(k): int(v) for k, v in zip(unique, counts)
            }
        expected = (batch["input_ids"] != 25).astype(batch["attention_mask"].dtype)
        entry["mask_errors"] = int(np.sum(batch["attention_mask"] != expected))
        shapes.append(entry)

    save_json(shapes, output_path)
    return output_path


def _generate_batch_summary(
    batches: List[Dict[str, np.ndarray]],
    output_path: Union[str, Path],
) -> Path:
    """Generate a per-batch CSV summary.

    Args:
        batches: List of batch dicts.
        output_path: Output path for the CSV file.

    Returns:
        Path to the saved file.
    """
    import pandas as pd

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    rows: List[Dict[str, Any]] = []
    for i, batch in enumerate(batches):
        pad_count = int(np.sum(batch["input_ids"] == 25))
        total = int(batch["input_ids"].size)
        unique_labels = None
        if "labels" in batch:
            unique_labels = dict(zip(*np.unique(batch["labels"], return_counts=True)))
        expected = (batch["input_ids"] != 25).astype(batch["attention_mask"].dtype)
        mask_errors = int(np.sum(batch["attention_mask"] != expected))
        row = {
            "batch_index": i,
            "samples": batch["input_ids"].shape[0],
            "seq_len": batch["input_ids"].shape[1],
            "total_tokens": total,
            "padding_tokens": pad_count,
            "padding_pct": round(100 * pad_count / total, 2),
            "mask_errors": mask_errors,
            "has_labels": "labels" in batch,
            "input_dtype": str(batch["input_ids"].dtype),
            "mask_dtype": str(batch["attention_mask"].dtype),
        }
        if unique_labels:
            row["label_distribution"] = str(unique_labels)
        rows.append(row)

    save_csv(pd.DataFrame(rows), output_path)
    return output_path


def _generate_adapter_statistics(
    metadata: AdapterMetadata,
    validation_stats: Dict[str, Any],
    stats_json_path: Union[str, Path],
    stats_csv_path: Union[str, Path],
) -> Dict[str, Path]:
    """Save adapter statistics as JSON and CSV.

    Args:
        metadata: AdapterMetadata instance.
        validation_stats: Validation statistics dict.
        stats_json_path: Output path for JSON.
        stats_csv_path: Output path for CSV.

    Returns:
        Dict of format -> Path.
    """
    import pandas as pd

    stats_json_path = Path(stats_json_path)
    stats_csv_path = Path(stats_csv_path)

    statistics = {
        "num_batches": metadata.num_batches,
        "num_samples": metadata.num_samples,
        "batch_size": metadata.batch_size,
        "seq_len": metadata.seq_len,
        "max_seq_len": metadata.max_seq_len,
        "total_tokens": metadata.total_tokens,
        "total_padding_tokens": metadata.total_padding_tokens,
        "padding_ratio": metadata.padding_ratio,
        "mask_errors": metadata.mask_errors,
        "all_masks_consistent": metadata.all_masks_consistent,
        "has_labels": metadata.has_labels,
        "input_dtype": metadata.input_dtype,
        "label_dtype": metadata.label_dtype,
        "mask_dtype": metadata.mask_dtype,
        "padding_strategy": metadata.padding_strategy,
        "truncation_enabled": metadata.truncation_enabled,
        "created_at": metadata.created_at,
        **{f"validation_{k}": v for k, v in validation_stats.items()},
    }
    save_json(statistics, stats_json_path)
    save_csv(pd.DataFrame([statistics]), stats_csv_path)

    return {"json": stats_json_path, "csv": stats_csv_path}


def _build_experiment_path(
    output_dir: Optional[Union[str, Path]] = None,
    experiment_id: Optional[str] = None,
) -> Path:
    """Build the adapter output directory path.

    Pattern: outputs/experiments/<experiment_id>/adapter/

    Args:
        output_dir: Explicit output directory (overrides experiment_id).
        experiment_id: Experiment ID for structured output.

    Returns:
        Path to adapter output directory.
    """
    if output_dir is not None:
        return Path(output_dir)
    if experiment_id is not None:
        base = Path("outputs") / "experiments" / experiment_id / "adapter"
    else:
        base = Path("outputs") / "adapter"
    return base


# ------------------------------------------------------------------
# ProteinBERTDataAdapter
# ------------------------------------------------------------------


class ProteinBERTDataAdapter:
    """Bridging layer between preprocessing outputs and ProteinBERT inputs.

    Accepts preprocessed batch dicts (from encoder.create_batches or
    PreprocessingPipeline), validates tensors, provides deterministic
    iteration, and generates publication-quality metadata.

    This is the only layer aware of ProteinBERT's expected tensor format.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the data adapter.

        Args:
            config: Optional configuration dict. Recognized keys:
                - batch_size: int (default 8)
                - max_length: int (default 512)
                - padding: str (default "right")
                - truncation: bool (default True)
                - extended_alphabet: bool (default False)
                - unknown_token: str (default "X")
                - shuffle: bool (default False)
                - seed: int (default 42)
        """
        self.config: Dict[str, Any] = {
            "batch_size": 8,
            "max_length": 512,
            "padding": "right",
            "truncation": True,
            "extended_alphabet": False,
            "unknown_token": "X",
            "shuffle": False,
            "seed": 42,
        }
        if config:
            self.config.update(config)

        self._batches: List[Dict[str, np.ndarray]] = []
        self._num_batches: int = 0
        self._num_samples: int = 0
        self._validated: bool = False
        self._validation_stats: Dict[str, Any] = {}
        self._metadata: Optional[AdapterMetadata] = None
        self._has_labels: bool = False

        logger.debug(f"ProteinBERTDataAdapter initialized (config keys: {list(self.config.keys())})")

    # ------------------------------------------------------------------
    # Batch creation from raw data
    # ------------------------------------------------------------------

    def create_batches(
        self,
        sequences: List[str],
        labels: Optional[List[Any]] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """Create validated, deterministic batches from raw sequences.

        Args:
            sequences: Raw protein sequence strings.
            labels: Optional label values (strings 'H'/'E'/'C' or ints).

        Returns:
            List of batch dicts with validated tensors.

        Raises:
            TensorValidationError: If any batch fails validation.
        """
        from .encoder import encode_sequence, encode_labels, pad_sequences, truncate_sequences

        batch_size = self.config["batch_size"]
        max_length = self.config["max_length"]
        padding = self.config["padding"]
        truncation = self.config["truncation"]
        extended_alphabet = self.config["extended_alphabet"]

        indices = list(range(len(sequences)))
        rng = np.random.RandomState(self.config["seed"])
        if self.config["shuffle"]:
            rng.shuffle(indices)

        batches: List[Dict[str, np.ndarray]] = []

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]

            batch_seqs = [sequences[i] for i in batch_indices]
            batch_encoded = [encode_sequence(s, extended_alphabet) for s in batch_seqs]

            original_lengths = np.array([len(s) for s in batch_seqs], dtype=np.int32)

            if truncation:
                batch_encoded = truncate_sequences(batch_encoded, max_length)

            input_ids = pad_sequences(batch_encoded, max_length, padding)
            attention_mask = (input_ids != 25).astype(np.int32)  # PAD_ID

            batch_dict: Dict[str, np.ndarray] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "lengths": original_lengths,
            }

            if labels is not None:
                batch_labels = [labels[i] for i in batch_indices]
                if batch_labels and isinstance(batch_labels[0], str):
                    batch_dict["labels"] = encode_labels(batch_labels)
                else:
                    batch_dict["labels"] = np.array(batch_labels, dtype=np.int32)
                self._has_labels = True

            batches.append(batch_dict)

        # Validate all batches immediately
        self._batches = batches
        self._num_batches = len(batches)
        self._num_samples = sum(b["input_ids"].shape[0] for b in batches)
        self.validate()
        logger.info(
            f"Created {self._num_batches} batches ({self._num_samples} samples, "
            f"batch_size={batch_size})"
        )

        return batches

    # ------------------------------------------------------------------
    # Accept precomputed batches
    # ------------------------------------------------------------------

    def from_batches(
        self,
        batches: List[Dict[str, np.ndarray]],
    ) -> "ProteinBERTDataAdapter":
        """Accept a precomputed list of batch dicts and validate them.

        Args:
            batches: List of dicts with keys 'input_ids', 'attention_mask',
                     and optionally 'labels', 'lengths'.

        Returns:
            self for chaining.

        Raises:
            TensorValidationError: If any batch fails validation.
        """
        self._batches = list(batches)
        self._num_batches = len(batches)
        self._num_samples = 0
        self._has_labels = False
        # Validate (handles empty lists gracefully)
        self.validate()
        if self._batches:
            self._num_samples = sum(b["input_ids"].shape[0] for b in self._batches)
            self._has_labels = "labels" in self._batches[0]
        logger.info(
            f"Loaded {self._num_batches} precomputed batches ({self._num_samples} samples)"
        )
        return self

    def from_preprocessed(
        self,
        prepared: Dict[str, Any],
    ) -> "ProteinBERTDataAdapter":
        """Accept a full preprocessing pipeline output and convert to batches.

        This is the primary integration point between preprocessing and the adapter.

        Args:
            prepared: Output dict from PreprocessingPipeline.run() or
                      PreprocessingPipeline.run_on_sequences().

        Returns:
            self for chaining.
        """
        # Reconstruct batches from the encoded data
        if "encoded_shape" not in prepared:
            raise AdapterError(
                "Preprocessed output missing 'encoded_shape'. "
                "Ensure the preprocessing pipeline was run successfully."
            )

        self.config["max_length"] = prepared["encoded_shape"][1]
        self.config["batch_size"] = self.config.get("batch_size", 8)

        # Attempt to reload batches from statistics if available
        stats = prepared.get("stats", {})
        validation = stats.get("validation", {})
        n_valid = validation.get("retained_samples", 0)

        logger.info(
            f"Adapting preprocessed output: {n_valid} valid sequences, "
            f"max_length={self.config['max_length']}"
        )

        # Store as metadata-ready but batches are not stored here directly.
        # The user should call create_batches or from_batches with the
        # sequences and labels that produced the preprocessed output.
        # This method documents the integration path.
        logger.info(
            "from_preprocessed: metadata recorded. "
            "Use create_batches(sequences, labels) with the same inputs."
        )

        return self

    # ------------------------------------------------------------------
    # From raw (convenience)
    # ------------------------------------------------------------------

    def from_raw(
        self,
        sequences: List[str],
        labels: Optional[List[Any]] = None,
    ) -> "ProteinBERTDataAdapter":
        """Create validated batches from raw sequences in one step.

        Args:
            sequences: Raw protein sequence strings.
            labels: Optional label values.

        Returns:
            self for chaining.
        """
        self.create_batches(sequences, labels)
        return self

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> Dict[str, Any]:
        """Validate all stored batches.

        Returns:
            Dict with validation statistics (empty stats if no batches).

        Raises:
            TensorValidationError: If any validation check fails.
        """
        if not self._batches:
            self._validation_stats = {
                "num_batches": 0,
                "total_samples": 0,
                "all_masks_consistent": True,
                "mask_errors": 0,
            }
            self._validated = True
            return self._validation_stats

        stats = TensorValidator.validate_all(
            self._batches,
            require_labels=False,
        )
        self._validation_stats = stats
        self._validated = True

        logger.debug(f"Validation passed: {stats['num_batches']} batches, "
                     f"{stats['total_samples']} samples, "
                     f"padding_ratio={stats['padding_ratio']}")

        return stats

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return adapter metadata for publication."""
        if self._metadata is None and self._batches:
            self._metadata = AdapterMetadata.from_batches(self._batches, self.config)
        return self._metadata.to_dict() if self._metadata else {}

    def generate_metadata(self) -> AdapterMetadata:
        """Explicitly generate and return metadata object."""
        if not self._batches:
            raise AdapterError("Cannot generate metadata: no batches loaded")
        self._metadata = AdapterMetadata.from_batches(self._batches, self.config)
        return self._metadata

    def save_metadata(
        self,
        output_dir: Union[str, Path],
        filename: str = "adapter_metadata.json",
    ) -> Path:
        """Save adapter metadata to JSON.

        Args:
            output_dir: Directory to save metadata in.
            filename: Output filename.

        Returns:
            Path to saved metadata file.
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        metadata_dict = self.metadata
        path = save_json(metadata_dict, output_dir / filename)
        logger.info(f"Adapter metadata saved to {path}")
        return path

    def save_all(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Save all adapter artifacts: report, statistics, shapes, summary.

        Generates these files under output_dir:
          - adapter_report.md
          - adapter_metadata.json
          - adapter_statistics.json
          - adapter_statistics.csv
          - tensor_shapes.json
          - batch_summary.csv

        Args:
            output_dir: Explicit output directory.
                        If None, uses outputs/experiments/<experiment_id>/adapter/
                        or outputs/adapter/.
            experiment_id: Experiment ID for structured output under
                           outputs/experiments/<experiment_id>/adapter/.

        Returns:
            Dict mapping artifact names to their Paths.

        Raises:
            AdapterError: If no batches are loaded.
        """
        if not self._batches:
            raise AdapterError("Cannot save artifacts: no batches loaded")

        out = _build_experiment_path(output_dir, experiment_id)
        ensure_dir(out)

        if self._metadata is None:
            self._metadata = AdapterMetadata.from_batches(self._batches, self.config)

        # Generate tensor shapes
        shapes = []
        for i, batch in enumerate(self._batches):
            entry: Dict[str, Any] = {"batch_index": i}
            for key, tensor in batch.items():
                entry[key] = {
                    "shape": list(tensor.shape),
                    "ndim": tensor.ndim,
                    "dtype": str(tensor.dtype),
                    "size": int(tensor.size),
                }
            entry["batch_samples"] = int(batch["input_ids"].shape[0])
            entry["total_tokens"] = int(batch["input_ids"].size)
            pad_count = int(np.sum(batch["input_ids"] == 25))
            entry["padding_tokens"] = pad_count
            entry["padding_pct"] = round(100 * pad_count / batch["input_ids"].size, 2)
            if "labels" in batch:
                unique, counts = np.unique(batch["labels"], return_counts=True)
                entry["label_distribution"] = {
                    str(k): int(v) for k, v in zip(unique, counts)
                }
            expected = (batch["input_ids"] != 25).astype(batch["attention_mask"].dtype)
            entry["mask_errors"] = int(np.sum(batch["attention_mask"] != expected))
            shapes.append(entry)

        artifacts: Dict[str, Path] = {}

        # 1. Metadata JSON
        artifacts["metadata"] = save_json(
            self._metadata.to_dict(), out / "adapter_metadata.json"
        )

        # 2. Statistics JSON + CSV
        stats = _generate_adapter_statistics(
            self._metadata,
            self._validation_stats,
            out / "adapter_statistics.json",
            out / "adapter_statistics.csv",
        )
        artifacts["statistics_json"] = stats["json"]
        artifacts["statistics_csv"] = stats["csv"]

        # 3. Tensor shapes JSON
        artifacts["tensor_shapes"] = save_json(shapes, out / "tensor_shapes.json")

        # 4. Batch summary CSV
        import pandas as pd
        rows = []
        for s in shapes:
            row = {
                "batch_index": s["batch_index"],
                "samples": s["batch_samples"],
                "total_tokens": s["total_tokens"],
                "padding_tokens": s["padding_tokens"],
                "padding_pct": s["padding_pct"],
                "mask_errors": s["mask_errors"],
            }
            if "label_distribution" in s:
                row["label_distribution"] = str(s["label_distribution"])
            rows.append(row)
        artifacts["batch_summary"] = save_csv(
            pd.DataFrame(rows), out / "batch_summary.csv"
        )

        # 5. Adapter report Markdown
        artifacts["report"] = _generate_adapter_report(
            self._metadata,
            self.config,
            self._validation_stats,
            shapes,
            out / "adapter_report.md",
        )

        logger.info(f"All adapter artifacts saved to {out}")
        for name, path in artifacts.items():
            logger.debug(f"  {name}: {path}")

        return artifacts

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        if not self._validated:
            logger.warning("Accessing unvalidated batches; call validate() first")
        return self._batches[idx]

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        if not self._validated:
            logger.warning("Iterating over unvalidated batches; call validate() first")
        self._iter_idx = 0
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        if self._iter_idx >= len(self._batches):
            raise StopIteration
        batch = self._batches[self._iter_idx]
        self._iter_idx += 1
        return batch

    # ------------------------------------------------------------------
    # TF Dataset conversion (lazy)
    # ------------------------------------------------------------------

    def to_tf_dataset(
        self,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
    ) -> Any:
        """Convert batches to a tf.data.Dataset.

        TensorFlow must be installed. Import is lazy.

        Args:
            batch_size: Override batch size (default: config batch_size).
            shuffle: Whether to shuffle the dataset.

        Returns:
            tf.data.Dataset object.

        Raises:
            ImportError: If TensorFlow is not installed.
            AdapterError: If no batches are loaded.
        """
        import tensorflow as tf  # lazy import

        if not self._batches:
            raise AdapterError("No batches loaded; call from_batches() or create_batches() first")
        if not self._validated:
            self.validate()

        bs = batch_size or self.config["batch_size"]

        def generator_fn():
            for batch in self._batches:
                yield batch["input_ids"], batch["attention_mask"], batch.get("labels")

        output_types = (tf.int32, tf.int32, tf.int32 if self._has_labels else tf.int32)
        output_shapes = (
            tf.TensorShape([None, self.config["max_length"]]),
            tf.TensorShape([None, self.config["max_length"]]),
            tf.TensorShape([None]) if self._has_labels else tf.TensorShape([]),
        )

        dataset = tf.data.Dataset.from_generator(generator_fn, output_types, output_shapes)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self._batches))
        dataset = dataset.batch(bs, drop_remainder=False)

        return dataset

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all batches and reset state."""
        self._batches = []
        self._num_batches = 0
        self._num_samples = 0
        self._validated = False
        self._validation_stats = {}
        self._metadata = None
        self._has_labels = False

    def info(self) -> Dict[str, Any]:
        """Return a human-readable summary of the adapter state."""
        return {
            "num_batches": self._num_batches,
            "num_samples": self._num_samples,
            "validated": self._validated,
            "has_labels": self._has_labels,
            "batch_size": self.config["batch_size"],
            "max_length": self.config["max_length"],
            "padding": self.config["padding"],
            "truncation": self.config["truncation"],
            "extended_alphabet": self.config["extended_alphabet"],
            "shuffle": self.config["shuffle"],
        }

    def __repr__(self) -> str:
        return (
            f"ProteinBERTDataAdapter(batches={self._num_batches}, "
            f"samples={self._num_samples}, "
            f"validated={self._validated})"
        )
