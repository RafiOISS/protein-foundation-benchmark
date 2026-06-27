"""ProteinBERT loader — checkpoint/resource location and loading.

Responsibilities:
  - locate checkpoints in a given directory
  - load pretrained weights (download if missing when supported)
  - no model inference logic
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ...utils.logging import get_logger


logger = get_logger(__name__)


# Default HuggingFace model ID for ProteinBERT
_DEFAULT_PROTEINBERT_ID = "google/proteinbert"


def get_default_model_id() -> str:
    """Return the default ProteinBERT model identifier."""
    return _DEFAULT_PROTEINBERT_ID


def locate_checkpoints(
    directory: Union[str, Path],
    pattern: str = "*.h5",
) -> List[Path]:
    """Locate checkpoint files in a directory.

    Args:
        directory: Directory to search.
        pattern: Glob pattern for checkpoint files.

    Returns:
        Sorted list of checkpoint paths.
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Checkpoint directory not found: {directory}")
        return []
    checkpoints = sorted(directory.glob(pattern))
    logger.info(f"Found {len(checkpoints)} checkpoint(s) in {directory}")
    return checkpoints


def load_pretrained(
    model_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """Load pretrained ProteinBERT model and tokenizer.

    TensorFlow and proteinbert are imported lazily.
    Downloads the model if path is None and supported by the backend.

    Args:
        model_path: Optional path to pretrained weights.
        config: Optional model configuration dict.

    Returns:
        Tuple of (tf_model, tokenizer).

    Raises:
        ImportError: If TensorFlow or proteinbert is not installed.
        FileNotFoundError: If model_path does not exist.
    """
    import tensorflow as tf
    from proteinbert import load_pretrained_model
    from proteinbert.tokenization import ProteinBERTTokenizer

    if model_path is not None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ProteinBERT model path not found: {model_path}")

    try:
        if model_path is not None:
            tf_model = load_pretrained_model(str(model_path))
            logger.info(f"Loaded ProteinBERT from {model_path}")
        else:
            tf_model = load_pretrained_model()
            logger.info("Loaded pretrained ProteinBERT (default initialization)")

        tokenizer = ProteinBERTTokenizer()
        logger.info("ProteinBERT tokenizer loaded")

        return tf_model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load ProteinBERT: {e}")
        raise


def download_if_missing(
    model_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Ensure ProteinBERT resources are available, downloading if supported.

    Currently ProteinBERT does not have a standard HuggingFace hub download;
    this is a placeholder for future checkpoint management.

    Args:
        model_path: User-specified model path.
        cache_dir: Cache directory for automatic download.

    Returns:
        Resolved model path, or None if not found.
    """
    if model_path is not None:
        resolved = Path(model_path)
        if resolved.exists():
            return resolved
        logger.warning(f"Specified model path does not exist: {resolved}")

    if cache_dir is not None:
        cache_dir = Path(cache_dir) / "proteinbert"
        if cache_dir.exists():
            return cache_dir

    logger.info("No local ProteinBERT checkpoint found; will use default loading")
    return None
