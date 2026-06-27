"""Downloader — HTTP download with resume, checksums, and progress."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Union
from urllib.parse import urlparse

import requests

from ..utils.logging import get_logger


logger = get_logger(__name__)


_CHUNK_SIZE = 8192


def compute_sha256(path: Union[str, Path]) -> str:
    """Compute SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _get_remote_file_size(url: str) -> Optional[int]:
    """Get file size from Content-Length header without downloading."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=30)
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        return int(length) if length else None
    except requests.RequestException:
        return None


def download_file(
    url: str,
    dest: Union[str, Path],
    expected_sha256: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    timeout: int = 300,
) -> Path:
    """Download a file with resume support and optional SHA-256 verification.

    Args:
        url: Source URL.
        dest: Destination file path.
        expected_sha256: Optional SHA-256 hash for verification.
        progress_callback: Called with (downloaded_bytes, total_bytes).
        timeout: Request timeout in seconds.

    Returns:
        Path to downloaded file.

    Raises:
        ValueError: If checksum verification fails.
        requests.RequestException: On download failure.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded and valid
    if dest.exists() and expected_sha256:
        actual = compute_sha256(dest)
        if actual == expected_sha256:
            logger.info(f"File already exists with valid checksum: {dest.name}")
            return dest
        logger.info(f"Checksum mismatch, re-downloading: {dest.name}")

    # Resume support: get remote size and local size
    remote_size = _get_remote_file_size(url)
    local_size = dest.stat().st_size if dest.exists() else 0
    resume_pos = 0

    headers = {}
    if dest.exists() and local_size > 0:
        headers["Range"] = f"bytes={local_size}-"
        resume_pos = local_size
        logger.info(f"Resuming download at byte {local_size}")

    # Stream download
    resp = requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()

    total = remote_size or int(resp.headers.get("Content-Length", 0))
    downloaded = resume_pos
    mode = "ab" if resume_pos else "wb"

    sha = hashlib.sha256()

    with open(dest, mode) as f:
        for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                sha.update(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(downloaded, total)
                elif total and total > 10 * 1024 * 1024:
                    # Log progress every 10% for files > 10MB
                    pct = int(100 * downloaded / total)
                    if pct % 10 == 0:
                        mb = downloaded / (1024 * 1024)
                        tmb = total / (1024 * 1024)
                        logger.debug(f"Downloaded {mb:.1f}/{tmb:.1f} MB ({pct}%)")

    # Verify checksum
    if expected_sha256:
        actual = sha.hexdigest()
        if actual != expected_sha256:
            dest.unlink()
            raise ValueError(
                f"SHA-256 mismatch for {dest.name}: "
                f"expected {expected_sha256}, got {actual}"
            )
        logger.info(f"SHA-256 verified: {dest.name}")

    mb = downloaded / (1024 * 1024)
    logger.info(f"Downloaded {dest.name} ({mb:.1f} MB)")
    return dest


def extract_archive(
    archive_path: Union[str, Path],
    dest_dir: Union[str, Path],
    format: Optional[str] = None,
) -> Path:
    """Extract a compressed archive.

    Supports: .tar.gz, .tar.bz2, .tar, .zip, .gz.

    Args:
        archive_path: Path to archive file.
        dest_dir: Destination directory.
        format: Archive format (auto-detected from extension if None).

    Returns:
        Path to extraction directory.
    """
    import tarfile
    import zipfile

    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    suffix = format or "".join(archive_path.suffixes)

    if ".tar" in suffix or ".tgz" in suffix:
        mode = "r:*"
        with tarfile.open(archive_path, mode) as tar:
            tar.extractall(path=dest_dir)
    elif ".zip" in suffix:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif suffix == ".gz":
        import gzip
        import shutil
        out_path = dest_dir / archive_path.stem.replace(".tar", "")
        with gzip.open(archive_path, "rb") as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        raise ValueError(f"Unsupported archive format: {suffix}")

    logger.info(f"Extracted {archive_path.name} to {dest_dir}")
    return dest_dir


def download_and_extract(
    url: str,
    dest_dir: Union[str, Path],
    expected_sha256: Optional[str] = None,
    remove_archive: bool = True,
) -> Path:
    """Download and extract an archive in one step.

    Args:
        url: Archive URL.
        dest_dir: Destination for extracted contents.
        expected_sha256: Optional SHA-256 of archive.
        remove_archive: Whether to delete archive after extraction.

    Returns:
        Path to extraction directory.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    archive_name = Path(urlparse(url).path).name or "download"
    archive_path = dest_dir / archive_name

    download_file(url, archive_path, expected_sha256)
    extract_archive(archive_path, dest_dir)

    if remove_archive:
        archive_path.unlink()
        logger.debug(f"Removed archive: {archive_path.name}")

    return dest_dir


def list_missing_files(directory: Union[str, Path], required_files: Dict[str, Optional[str]]) -> list:
    """Check which required files are missing or have wrong checksums.

    Args:
        directory: Directory to search.
        required_files: Dict of relative_path -> optional SHA-256.

    Returns:
        List of (path, reason) tuples for missing/invalid files.
    """
    directory = Path(directory)
    missing = []
    for rel_path, expected_sha in required_files.items():
        full_path = directory / rel_path
        if not full_path.exists():
            missing.append((rel_path, "not found"))
        elif expected_sha:
            actual = compute_sha256(full_path)
            if actual != expected_sha:
                missing.append((rel_path, f"SHA-256 mismatch"))
    return missing
