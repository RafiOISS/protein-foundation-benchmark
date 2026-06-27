"""I/O Utilities for the Protein Foundation Model Benchmark Framework.

Provides functions for loading/saving various file formats.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from ..utils.logging import get_logger


logger = get_logger(__name__)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed dictionary.
    """
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Save dictionary to YAML file.

    Args:
        data: Dictionary to save.
        path: Output path.

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return path


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON data.
    """
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> Path:
    """Save data to JSON file.

    Args:
        data: Data to save.
        path: Output path.
        indent: JSON indentation.

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)
    return path


def load_pickle(path: Union[str, Path]) -> Any:
    """Load pickle file.

    Args:
        path: Path to pickle file.

    Returns:
        Loaded object.
    """
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Union[str, Path]) -> Path:
    """Save object to pickle file.

    Args:
        obj: Object to save.
        path: Output path.

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_npz(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load NPZ file.

    Args:
        path: Path to NPZ file.

    Returns:
        Dictionary of arrays.
    """
    path = Path(path)
    return dict(np.load(path))


def save_npz(data: Dict[str, np.ndarray], path: Union[str, Path], compressed: bool = True) -> Path:
    """Save arrays to NPZ file.

    Args:
        data: Dictionary of arrays.
        path: Output path.
        compressed: Whether to compress.

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **data)
    else:
        np.savez(path, **data)
    return path


def load_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load CSV file.

    Args:
        path: Path to CSV file.
        **kwargs: Additional arguments to pd.read_csv.

    Returns:
        DataFrame.
    """
    path = Path(path)
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> Path:
    """Save DataFrame to CSV.

    Args:
        df: DataFrame to save.
        path: Output path.
        **kwargs: Additional arguments to df.to_csv.

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    return path


def load_parquet(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load Parquet file.

    Args:
        path: Path to Parquet file.
        **kwargs: Additional arguments to pd.read_parquet.

    Returns:
        DataFrame.
    """
    path = Path(path)
    return pd.read_parquet(path, **kwargs)


def save_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> Path:
    """Save DataFrame to Parquet.

    Args:
        df: DataFrame to save.
        path: Output path.
        **kwargs: Additional arguments to df.to_parquet.

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)
    return path


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(path: Union[str, Path]) -> int:
    """Get file size in bytes.

    Args:
        path: File path.

    Returns:
        File size in bytes.
    """
    return Path(path).stat().st_size


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False,
) -> List[Path]:
    """List files matching pattern.

    Args:
        directory: Directory to search.
        pattern: Glob pattern.
        recursive: Whether to search recursively.

    Returns:
        List of file paths.
    """
    directory = Path(directory)
    if recursive:
        return list(directory.rglob(pattern))
    return list(directory.glob(pattern))


def read_text(path: Union[str, Path]) -> str:
    """Read text file.

    Args:
        path: File path.

    Returns:
        File contents.
    """
    path = Path(path)
    return path.read_text()


def write_text(content: str, path: Union[str, Path]) -> Path:
    """Write text to file.

    Args:
        content: Text content.
        path: Output path.

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """Copy file.

    Args:
        src: Source path.
        dst: Destination path.

    Returns:
        Destination path.
    """
    import shutil
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """Move file.

    Args:
        src: Source path.
        dst: Destination path.

    Returns:
        Destination path.
    """
    import shutil
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(src, dst)
    return dst


def delete_file(path: Union[str, Path]) -> bool:
    """Delete file.

    Args:
        path: File path.

    Returns:
        True if deleted, False if not found.
    """
    path = Path(path)
    if path.exists():
        path.unlink()
        return True
    return False