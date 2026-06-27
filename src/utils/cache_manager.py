"""CacheManager — centralized workspace cache management.

Responsibilities:
  - create and manage all workspace cache directories
  - return canonical cache paths for all models
  - validate directory writability
  - prevent duplicate downloads via manifest tracking
  - configure environment variables to redirect third-party caches
  - expose reusable APIs for all future models

No TensorFlow imports at module level.
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from ..utils.logging import get_logger


logger = get_logger(__name__)


# ------------------------------------------------------------------
# Default cache subdirectory structure
# ------------------------------------------------------------------

CACHE_SUBDIRS: Dict[str, str] = {
    "hub": "hub",                     # HuggingFace hub cache
    "checkpoints": "checkpoints",     # Model checkpoints
    "datasets": "datasets",           # Downloaded datasets
    "models": "models",               # Model weight files
    "misc": "misc",                   # Miscellaneous cache
    "tmp": "tmp",                     # Temporary files
}


# ------------------------------------------------------------------
# Environment variables to configure for third-party caches
# ------------------------------------------------------------------

CACHE_ENV_VARS: Dict[str, str] = {
    "HF_HOME": "hub",
    "HF_HUB_CACHE": "hub",
    "TRANSFORMERS_CACHE": "hub",
    "HUGGINGFACE_HUB_CACHE": "hub",
    "DATASETS_CACHE": "datasets",
    "TORCH_HOME": "hub",
    "TFHUB_CACHE_DIR": "hub",
    "KERAS_HOME": "hub",
    "XDG_CACHE_HOME": None,  # special: parent of cache root
}


# ------------------------------------------------------------------
# CacheManager
# ------------------------------------------------------------------


class CacheManager:
    """Centralized workspace cache management.

    All framework-controlled downloads, caches, checkpoints, and
    temporary artifacts are confined to the project workspace.

    Usage:
        cm = CacheManager(workspace_root)
        cm.configure_environment()  # sets env vars *before* library imports
        hub_dir = cm.get_cache_dir("hub")
        cm.ensure_all()
        cm.validate()
    """

    def __init__(
        self,
        workspace_root: Union[str, Path],
        subdirs: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize CacheManager.

        Args:
            workspace_root: Root directory for all caches.
                           Typically project_root / "outputs" / "cache".
            subdirs: Dict of cache_name -> subdirectory name.
                     Defaults to CACHE_SUBDIRS.
        """
        self._workspace_root = Path(workspace_root).resolve()
        self._subdirs: Dict[str, str] = dict(subdirs or CACHE_SUBDIRS)
        self._cache_dirs: Dict[str, Path] = {}
        self._env_configured: bool = False
        self._manifest: Dict[str, Any] = {"files": [], "directories": []}

        # Build cache directory paths
        for name, subdir in self._subdirs.items():
            self._cache_dirs[name] = self._workspace_root / subdir

        logger.debug(f"CacheManager initialized (root={self._workspace_root})")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def workspace_root(self) -> Path:
        return self._workspace_root

    @property
    def cache_dirs(self) -> Dict[str, Path]:
        return dict(self._cache_dirs)

    @property
    def env_configured(self) -> bool:
        return self._env_configured

    # ------------------------------------------------------------------
    # Cache directory access
    # ------------------------------------------------------------------

    def get_cache_dir(self, name: str) -> Path:
        """Get the canonical cache directory path for a named cache.

        Args:
            name: Cache directory name (e.g., 'hub', 'checkpoints', 'datasets').

        Returns:
            Path to the cache directory.

        Raises:
            KeyError: If the cache name is not recognized.
        """
        if name not in self._cache_dirs:
            raise KeyError(
                f"Unknown cache directory: '{name}'. "
                f"Available: {list(self._cache_dirs.keys())}"
            )
        return self._cache_dirs[name]

    def get_checkpoint_dir(self, experiment_id: str) -> Path:
        """Get the checkpoint directory for a specific experiment.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            Path to the experiment's checkpoint directory.
        """
        return self._cache_dirs["checkpoints"] / experiment_id

    def get_hub_dir(self) -> Path:
        """Get the HuggingFace / model hub cache directory."""
        return self._cache_dirs["hub"]

    def get_datasets_dir(self) -> Path:
        """Get the datasets cache directory."""
        return self._cache_dirs["datasets"]

    def get_models_dir(self) -> Path:
        """Get the model weights cache directory."""
        return self._cache_dirs["models"]

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def ensure_all(self) -> Dict[str, Path]:
        """Create all cache directories.

        Returns:
            Dict of cache name -> Path.
        """
        created: Dict[str, Path] = {}
        for name, path in self._cache_dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            created[name] = path
            logger.debug(f"Ensured cache directory: {path}")
        logger.info(f"All cache directories created under {self._workspace_root}")
        return created

    def ensure_dir(self, name: str) -> Path:
        """Ensure a specific cache directory exists.

        Args:
            name: Cache directory name.

        Returns:
            Path to the directory.
        """
        path = self.get_cache_dir(name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Environment variable configuration
    # ------------------------------------------------------------------

    def configure_environment(self) -> Dict[str, str]:
        """Configure all cache-related environment variables.

        Sets environment variables to redirect third-party library caches
        into the workspace. Must be called before importing any library
        that may cache files.

        Returns:
            Dict of env var -> value set.
        """
        configured: Dict[str, str] = {}

        for var_name, subdir_name in CACHE_ENV_VARS.items():
            if subdir_name is not None:
                target_dir = self._cache_dirs.get(subdir_name)
                if target_dir is not None:
                    value = str(target_dir)
                    os.environ[var_name] = value
                    configured[var_name] = value
            else:
                # Special case: XDG_CACHE_HOME points to workspace cache root
                os.environ[var_name] = str(self._workspace_root)
                configured[var_name] = str(self._workspace_root)

        self._env_configured = True
        logger.info(
            f"Configured {len(configured)} cache environment variables "
            f"-> {self._workspace_root}"
        )
        return configured

    def get_environment(self) -> Dict[str, Optional[str]]:
        """Get current values of all cache-related environment variables.

        Returns:
            Dict of env var name -> current value (or None if not set).
        """
        return {var: os.environ.get(var) for var in CACHE_ENV_VARS}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> Dict[str, Any]:
        """Validate all cache directories.

        Checks:
          - directories exist
          - directories are writable

        Returns:
            Dict with validation results.

        Raises:
            RuntimeError: If any directory fails validation.
        """
        results: Dict[str, Any] = {
            "workspace_root": str(self._workspace_root),
            "directories": {},
            "all_ok": True,
            "errors": [],
        }

        for name, path in self._cache_dirs.items():
            status: Dict[str, Any] = {"path": str(path), "exists": False, "writable": False}
            try:
                status["exists"] = path.exists()
                if not status["exists"]:
                    path.mkdir(parents=True, exist_ok=True)
                    status["exists"] = True

                # Check writability
                test_file = path / ".write_test"
                test_file.touch()
                test_file.unlink()
                status["writable"] = True

            except (OSError, PermissionError) as e:
                status["error"] = str(e)
                results["all_ok"] = False
                results["errors"].append(f"{name}: {e}")

            results["directories"][name] = status

        if not results["all_ok"]:
            raise RuntimeError(
                f"Cache validation failed:\n" + "\n".join(results["errors"])
            )

        logger.info("Cache validation passed")
        return results

    def available_disk_space(self, path: Optional[Union[str, Path]] = None) -> float:
        """Get available disk space in GB at the given path.

        Args:
            path: Path to check (default: workspace root).

        Returns:
            Available disk space in GB.
        """
        path = Path(path or self._workspace_root)
        try:
            import shutil
            usage = shutil.disk_usage(path)
            return usage.free / (1024 ** 3)
        except Exception:
            logger.warning(f"Could not check disk space at {path}")
            return -1.0

    # ------------------------------------------------------------------
    # Manifest tracking (prevent duplicate downloads)
    # ------------------------------------------------------------------

    def record_download(self, url: str, local_path: Union[str, Path], checksum: Optional[str] = None) -> None:
        """Record a downloaded file in the manifest.

        Args:
            url: Source URL.
            local_path: Local file path.
            checksum: Optional SHA-256 checksum.
        """
        self._manifest["files"].append({
            "url": url,
            "local_path": str(local_path),
            "checksum": checksum,
            "size": Path(local_path).stat().st_size if Path(local_path).exists() else None,
        })

    def is_downloaded(self, url: str, checksum: Optional[str] = None) -> bool:
        """Check if a URL has already been downloaded.

        Args:
            url: Source URL.
            checksum: Optional checksum to verify.

        Returns:
            True if already downloaded.
        """
        for entry in self._manifest["files"]:
            if entry["url"] == url:
                if checksum and entry.get("checksum") != checksum:
                    continue
                if entry.get("local_path") and Path(entry["local_path"]).exists():
                    return True
        return False

    def save_manifest(self, output_dir: Union[str, Path]) -> Path:
        """Save the download manifest to a JSON file.

        Args:
            output_dir: Directory to save the manifest.

        Returns:
            Path to saved manifest file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "cache_manifest.json"
        with open(path, "w") as f:
            json.dump(self._manifest, f, indent=2, default=str)
        logger.info(f"Cache manifest saved to {path}")
        return path

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear(self, name: Optional[str] = None) -> None:
        """Clear cache directories.

        Args:
            name: Specific cache to clear (clears all if None).
        """
        if name is not None:
            path = self.get_cache_dir(name)
            if path.exists():
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cleared cache: {name} ({path})")
        else:
            for cache_name, cache_path in self._cache_dirs.items():
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                    cache_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared all caches under {self._workspace_root}")

    def clear_manifest(self) -> None:
        """Reset the download manifest."""
        self._manifest = {"files": [], "directories": []}

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """Return a summary of the CacheManager state."""
        return {
            "workspace_root": str(self._workspace_root),
            "directories": {k: str(v) for k, v in self._cache_dirs.items()},
            "env_configured": self._env_configured,
            "num_manifest_entries": len(self._manifest["files"]),
        }

    def __repr__(self) -> str:
        return (
            f"CacheManager(root={self._workspace_root}, "
            f"dirs={len(self._cache_dirs)}, "
            f"env_configured={self._env_configured})"
        )
