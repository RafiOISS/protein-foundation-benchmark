"""Environment Utilities for the Protein Foundation Model Benchmark Framework.

Provides functions for querying and logging environment information.
"""

import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..utils.logging import get_logger


logger = get_logger(__name__)


def get_device(device: str = "auto") -> torch.device:
    """Get torch device.

    Args:
        device: Device string ('auto', 'cuda', 'cpu', 'mps', 'cuda:N').

    Returns:
        Torch device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def get_num_gpus() -> int:
    """Get number of available GPUs.

    Returns:
        Number of GPUs.
    """
    return torch.cuda.device_count()


def get_gpu_info() -> List[Dict[str, Any]]:
    """Get GPU information.

    Returns:
        List of GPU info dictionaries.
    """
    if not torch.cuda.is_available():
        return []

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "index": i,
            "name": props.name,
            "total_memory": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
        })
    return gpus


def get_cuda_version() -> Optional[str]:
    """Get CUDA version.

    Returns:
        CUDA version string or None.
    """
    return torch.version.cuda


def get_cudnn_version() -> Optional[int]:
    """Get cuDNN version.

    Returns:
        cuDNN version or None.
    """
    return torch.backends.cudnn.version()


def get_pytorch_version() -> str:
    """Get PyTorch version.

    Returns:
        PyTorch version string.
    """
    return torch.__version__


def get_python_version() -> str:
    """Get Python version.

    Returns:
        Python version string.
    """
    return sys.version


def get_platform_info() -> Dict[str, str]:
    """Get platform information.

    Returns:
        Dictionary with platform info.
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }


def log_environment_info(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Log comprehensive environment information.

    Args:
        logger: Logger instance (uses module logger if None).

    Returns:
        Dictionary with environment info.
    """
    log = logger or get_logger(__name__)

    info = {
        "pytorch_version": get_pytorch_version(),
        "python_version": get_python_version(),
        "cuda_version": get_cuda_version(),
        "cudnn_version": get_cudnn_version(),
        "num_gpus": get_num_gpus(),
        "gpu_info": get_gpu_info(),
        "platform": get_platform_info(),
    }

    log.info("Environment Information:")
    log.info(f"  PyTorch: {info['pytorch_version']}")
    log.info(f"  Python: {info['python_version'].split()[0]}")
    log.info(f"  CUDA: {info['cuda_version']}")
    log.info(f"  cuDNN: {info['cudnn_version']}")
    log.info(f"  GPUs: {info['num_gpus']}")
    for gpu in info["gpu_info"]:
        log.info(f"    GPU {gpu['index']}: {gpu['name']} ({gpu['total_memory'] / 1e9:.1f} GB)")
    log.info(f"  Platform: {info['platform']['system']} {info['platform']['release']}")

    return info


def check_dependencies(
    required: List[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, bool]:
    """Check if required dependencies are installed.

    Args:
        required: List of required package names.
        logger: Logger instance.

    Returns:
        Dictionary of package -> availability.
    """
    log = logger or get_logger(__name__)

    if required is None:
        required = [
            "torch",
            "transformers",
            "numpy",
            "pandas",
            "sklearn",
            "matplotlib",
            "seaborn",
            "yaml",
            "rich",
        ]

    results = {}
    for pkg in required:
        try:
            __import__(pkg)
            results[pkg] = True
            log.debug(f"Dependency {pkg}: OK")
        except ImportError:
            results[pkg] = False
            log.warning(f"Dependency {pkg}: MISSING")

    missing = [p for p, v in results.items() if not v]
    if missing:
        log.warning(f"Missing dependencies: {missing}")

    return results


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage.

    Returns:
        Dictionary with memory info in GB.
    """
    import psutil

    process = psutil.Process()
    mem_info = process.memory_info()

    result = {
        "rss_gb": mem_info.rss / 1e9,
        "vms_gb": mem_info.vms / 1e9,
    }

    if torch.cuda.is_available():
        result["cuda_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        result["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        for i in range(torch.cuda.device_count()):
            result[f"cuda_{i}_allocated_gb"] = torch.cuda.memory_allocated(i) / 1e9
            result[f"cuda_{i}_reserved_gb"] = torch.cuda.memory_reserved(i) / 1e9

    return result


def log_memory_usage(logger: Optional[logging.Logger] = None) -> None:
    """Log current memory usage.

    Args:
        logger: Logger instance.
    """
    log = logger or get_logger(__name__)
    mem = get_memory_usage()
    log.info("Memory Usage:")
    for key, value in mem.items():
        log.info(f"  {key}: {value:.2f} GB")


def get_git_info() -> Dict[str, Optional[str]]:
    """Get git repository information.

    Returns:
        Dictionary with git info.
    """
    import subprocess

    info = {
        "commit": None,
        "branch": None,
        "remote_url": None,
        "dirty": None,
    }

    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass

    try:
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass

    try:
        info["remote_url"] = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass

    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["dirty"] = len(dirty) > 0
    except Exception:
        pass

    return info