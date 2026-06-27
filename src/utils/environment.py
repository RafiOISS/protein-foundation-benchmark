"""Environment — runtime environment validation and cache configuration.

Responsibilities:
  - configure all cache-related environment variables (HF, TF, Torch, etc.)
  - pre-flight validation before any caching library is imported
  - disk space, directory permissions, dependency checks
  - GPU detection and reporting
  - environment metadata for reproducibility

All cache env vars must be configured before any library that may
cache or download files is imported.
"""

import importlib
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ..utils.logging import get_logger


logger = get_logger(__name__)


# ------------------------------------------------------------------
# Third-party cache environment variables
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
    "XDG_CACHE_HOME": None,  # set to cache root
}

MINIMUM_DISK_SPACE_GB: float = 1.0

ESSENTIAL_PACKAGES: List[str] = [
    "torch",
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "yaml",
]

OPTIONAL_PACKAGES: List[str] = [
    "tensorflow",
    "transformers",
    "datasets",
    "seaborn",
    "rich",
    "psutil",
    "proteinbert",
]


# ------------------------------------------------------------------
# Cache environment configuration
# ------------------------------------------------------------------


def configure_cache_environment(cache_root: Union[str, Path]) -> Dict[str, str]:
    """Configure all cache-related environment variables.

    Must be called before importing HuggingFace, Transformers, TensorFlow,
    or any library that may cache files.

    Args:
        cache_root: Root directory for all caches.

    Returns:
        Dict of env var name -> value set.
    """
    cache_root = Path(cache_root).resolve()
    configured: Dict[str, str] = {}

    for var_name, subdir_name in CACHE_ENV_VARS.items():
        if subdir_name is not None:
            target = cache_root / subdir_name
            os.environ[var_name] = str(target)
            configured[var_name] = str(target)
        else:
            os.environ[var_name] = str(cache_root)
            configured[var_name] = str(cache_root)

    logger.info(f"Configured {len(configured)} cache env vars -> {cache_root}")
    return configured


def get_cache_environment() -> Dict[str, Optional[str]]:
    """Get current values of all cache-related environment variables.

    Returns:
        Dict of env var -> value (None if not set).
    """
    return {var: os.environ.get(var) for var in CACHE_ENV_VARS}


# ------------------------------------------------------------------
# Directory validation
# ------------------------------------------------------------------


def validate_directory(
    path: Union[str, Path],
    writable: bool = True,
    create: bool = True,
) -> Dict[str, Any]:
    """Validate a single directory.

    Args:
        path: Directory path.
        writable: Whether to check writability.
        create: Whether to create the directory if it doesn't exist.

    Returns:
        Dict with status information.

    Raises:
        RuntimeError: If validation fails.
    """
    path = Path(path)
    result: Dict[str, Any] = {
        "path": str(path),
        "exists": False,
        "writable": False,
        "error": None,
    }

    try:
        if not path.exists():
            if create:
                path.mkdir(parents=True, exist_ok=True)
                result["exists"] = True
            else:
                result["error"] = f"Directory does not exist: {path}"
                raise RuntimeError(result["error"])
        else:
            result["exists"] = True

        if writable:
            test_file = path / ".validation_test"
            test_file.touch()
            test_file.unlink()
            result["writable"] = True

    except (OSError, PermissionError) as e:
        result["error"] = str(e)
        raise RuntimeError(f"Directory validation failed: {path}: {e}") from e

    return result


def validate_directories(
    directories: Dict[str, Union[str, Path]],
    writable: bool = True,
    create: bool = True,
) -> Dict[str, Any]:
    """Validate multiple directories.

    Args:
        directories: Dict of name -> path.
        writable: Whether to check writability.
        create: Whether to create missing directories.

    Returns:
        Dict of name -> validation result.

    Raises:
        RuntimeError: If any directory fails validation.
    """
    results: Dict[str, Any] = {}
    all_ok = True
    errors: List[str] = []

    for name, path in directories.items():
        try:
            results[name] = validate_directory(path, writable=writable, create=create)
        except RuntimeError as e:
            results[name] = {"path": str(path), "error": str(e)}
            all_ok = False
            errors.append(str(e))

    if not all_ok:
        raise RuntimeError(
            f"Directory validation failed ({len(errors)} errors):\n" +
            "\n".join(errors)
        )

    return results


# ------------------------------------------------------------------
# Disk space validation
# ------------------------------------------------------------------


def check_disk_space(
    path: Union[str, Path],
    minimum_gb: float = MINIMUM_DISK_SPACE_GB,
) -> Dict[str, Any]:
    """Check available disk space.

    Args:
        path: Path to check.
        minimum_gb: Minimum required space in GB.

    Returns:
        Dict with disk space info.

    Raises:
        RuntimeError: If disk space is below minimum.
    """
    path = Path(path)
    try:
        import shutil
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)

        result = {
            "path": str(path),
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "minimum_required_gb": minimum_gb,
            "sufficient": free_gb >= minimum_gb,
        }

        if free_gb < minimum_gb:
            raise RuntimeError(
                f"Insufficient disk space at {path}: "
                f"{free_gb:.1f} GB free, need {minimum_gb:.1f} GB"
            )

        logger.info(f"Disk space OK: {free_gb:.1f} GB free at {path}")
        return result

    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Could not check disk space at {path}: {e}")
        return {
            "path": str(path),
            "free_gb": -1.0,
            "total_gb": -1.0,
            "used_gb": -1.0,
            "minimum_required_gb": minimum_gb,
            "sufficient": True,  # assume OK if we can't check
            "warning": str(e),
        }


# ------------------------------------------------------------------
# Dependency checking (without triggering downloads)
# ------------------------------------------------------------------


def check_package_available(name: str) -> bool:
    """Check if a Python package is available without importing it.

    Uses importlib.util.find_spec which does not trigger side effects
    like caching or downloading.

    Args:
        name: Package name.

    Returns:
        True if the package is installed.
    """
    try:
        spec = importlib.util.find_spec(name)
        return spec is not None
    except ModuleNotFoundError:
        return False


def check_dependencies(
    required: Optional[List[str]] = None,
    optional: Optional[List[str]] = None,
    silent: bool = False,
) -> Dict[str, Any]:
    """Check package dependencies.

    Uses importlib.util.find_spec to avoid triggering downloads.

    Args:
        required: List of required package names.
        optional: List of optional package names.
        silent: If True, don't log warnings.

    Returns:
        Dict with 'status' ('ok' | 'missing'), 'missing_required' list,
        'missing_optional' list, and 'available' dict.
    """
    required = required or ESSENTIAL_PACKAGES
    optional = optional or OPTIONAL_PACKAGES

    available: Dict[str, bool] = {}
    missing_required: List[str] = []
    missing_optional: List[str] = []

    for pkg in required:
        ok = check_package_available(pkg)
        available[pkg] = ok
        if not ok:
            missing_required.append(pkg)

    for pkg in optional:
        ok = check_package_available(pkg)
        available[pkg] = ok
        if not ok:
            missing_optional.append(pkg)

    status = "ok" if not missing_required else "missing"

    if not silent:
        if missing_required:
            logger.error(f"Missing required packages: {missing_required}")
        if missing_optional:
            logger.warning(f"Missing optional packages: {missing_optional}")

    if missing_required:
        raise RuntimeError(
            f"Missing required packages: {missing_required}. "
            f"Install with: pip install {' '.join(missing_required)}"
        )

    return {
        "status": status,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "available": available,
    }


# ------------------------------------------------------------------
# TensorFlow availability (without triggering full import)
# ------------------------------------------------------------------


def check_tensorflow_available() -> bool:
    """Check if TensorFlow is available without triggering downloads.

    Uses importlib.util.find_spec to avoid loading TF.

    Returns:
        True if TensorFlow is installed.
    """
    return check_package_available("tensorflow")


def check_proteinbert_available() -> bool:
    """Check if the proteinbert package is available.

    Returns:
        True if proteinbert is installed.
    """
    return check_package_available("proteinbert")


# ------------------------------------------------------------------
# GPU detection
# ------------------------------------------------------------------


def detect_gpus() -> List[Dict[str, Any]]:
    """Detect available GPUs.

    Returns:
        List of GPU info dicts.
    """
    if not torch.cuda.is_available():
        return []

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "index": i,
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "total_memory_gb": round(props.total_memory / 1e9, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        })

    return gpus


# ------------------------------------------------------------------
# Pre-flight validation
# ------------------------------------------------------------------


def validate_environment(
    workspace_root: Union[str, Path],
    cache_root: Union[str, Path],
    min_disk_gb: float = MINIMUM_DISK_SPACE_GB,
    require_gpu: bool = False,
) -> Dict[str, Any]:
    """Comprehensive pre-flight validation.

    Validates:
      - workspace exists
      - cache directories exist and are writable
      - sufficient disk space
      - required packages are installed
      - GPU availability (optional)

    This function does NOT import any library that may cache/download files.

    Args:
        workspace_root: Project workspace root.
        cache_root: Cache root directory.
        min_disk_gb: Minimum required disk space in GB.
        require_gpu: If True, requires at least one GPU.

    Returns:
        Dict with validation results.

    Raises:
        RuntimeError: If any validation check fails.
    """
    workspace_root = Path(workspace_root).resolve()
    cache_root = Path(cache_root).resolve()

    results: Dict[str, Any] = {
        "status": "ok",
        "workspace_root": str(workspace_root),
        "cache_root": str(cache_root),
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    # 1. Validate workspace directory
    try:
        results["checks"]["workspace"] = validate_directory(workspace_root)
    except RuntimeError as e:
        results["errors"].append(f"workspace: {e}")
        results["status"] = "failed"

    # 2. Validate cache directory
    try:
        results["checks"]["cache_root"] = validate_directory(cache_root)
    except RuntimeError as e:
        results["errors"].append(f"cache_root: {e}")
        results["status"] = "failed"

    # 3. Check disk space
    try:
        results["checks"]["disk_space"] = check_disk_space(cache_root, min_disk_gb)
    except RuntimeError as e:
        results["errors"].append(f"disk_space: {e}")
        results["status"] = "failed"

    # 4. Check dependencies (no imports)
    try:
        dep_results = check_dependencies(silent=True)
        results["checks"]["dependencies"] = dep_results
    except RuntimeError as e:
        results["errors"].append(f"dependencies: {e}")
        results["status"] = "failed"

    # 5. GPU detection (lightweight, no side effects)
    gpus = detect_gpus()
    results["checks"]["gpus"] = {
        "available": len(gpus) > 0,
        "count": len(gpus),
        "list": gpus,
    }
    if require_gpu and not gpus:
        results["errors"].append("gpu: No GPUs available but require_gpu=True")
        results["status"] = "failed"

    if results["status"] == "failed":
        error_msg = "Environment validation failed:\n" + "\n".join(results["errors"])
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"Environment validation passed ({workspace_root})")
    return results


# ------------------------------------------------------------------
# Software version info (safe, no download-triggering imports)
# ------------------------------------------------------------------


def get_software_versions() -> Dict[str, Optional[str]]:
    """Get versions of key software components.

    Only imports packages safely — uses __version__ attributes.

    Returns:
        Dict of component -> version string.
    """
    versions: Dict[str, Optional[str]] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    # PyTorch (already imported at module level)
    versions["torch"] = torch.__version__
    versions["cuda"] = torch.version.cuda

    # Safe version lookups (packages may not be installed)
    safe_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("transformers", "transformers"),
        ("tensorflow", "tensorflow"),
    ]

    for name, pkg in safe_packages:
        try:
            mod = importlib.import_module(pkg)
            versions[name] = getattr(mod, "__version__", "unknown")
        except (ImportError, ModuleNotFoundError):
            versions[name] = None

    return versions


# ------------------------------------------------------------------
# Compatibility wrappers (preserve src.utils public API)
# ------------------------------------------------------------------


def get_device() -> str:
    """Detect available compute device.

    Returns:
        'cuda' if GPU available, else 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_num_gpus() -> int:
    """Get the number of available GPUs.

    Returns:
        GPU count.
    """
    return torch.cuda.device_count()


def get_cuda_version() -> Optional[str]:
    """Get CUDA version string.

    Returns:
        CUDA version or None.
    """
    return torch.version.cuda


def log_environment_info(logger_override=None) -> Dict[str, Any]:
    """Log and return environment information for debugging.

    Args:
        logger_override: Optional logger instance.

    Returns:
        Dict of environment info.
    """
    info = get_software_versions()
    info["gpus"] = detect_gpus()
    info["gpu_count"] = len(info["gpus"])
    info["device"] = get_device()

    log = logger_override or logger
    log.info(f"Python {info.get('python', '?')} on {info.get('platform', '?')}")
    log.info(f"PyTorch {info.get('torch', '?')}, CUDA {info.get('cuda', 'N/A')}")
    log.info(f"Device: {info['device']}, GPUs: {info['gpu_count']}")

    return info
