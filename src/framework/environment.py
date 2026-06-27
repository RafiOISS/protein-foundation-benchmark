"""Environment — capture runtime environment for reproducibility."""

import platform
import sys
from typing import Any, Dict, List, Optional

import torch

from ..utils.logging import get_logger


logger = get_logger(__name__)


class Environment:
    """Captures and reports runtime environment information."""

    @staticmethod
    def capture() -> Dict[str, Any]:
        """Capture full environment snapshot."""
        info = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "num_gpus": torch.cuda.device_count(),
            "gpus": [],
        }

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append({
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1e9, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })

        return info

    @staticmethod
    def log() -> None:
        info = Environment.capture()
        logger.info(f"Python {info['python_version']} | PyTorch {info['pytorch_version']}")
        logger.info(f"CUDA {info['cuda_version']} | GPUs: {info['num_gpus']}")
        for gpu in info["gpus"]:
            logger.info(f"  {gpu['name']} ({gpu['total_memory_gb']} GB)")

    @staticmethod
    def check_dependencies(required: List[str]) -> Dict[str, bool]:
        results = {}
        for pkg in required:
            try:
                __import__(pkg)
                results[pkg] = True
            except ImportError:
                results[pkg] = False
                logger.warning(f"Missing dependency: {pkg}")
        return results