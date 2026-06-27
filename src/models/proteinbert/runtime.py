"""ProteinBERT Runtime — runtime initialization with lazy TensorFlow.

Initialization order (mandatory):
  1. Load framework configuration
  2. Initialize CacheManager
  3. Create all required workspace directories
  4. Verify directory permissions
  5. Verify available disk space
  6. Configure all cache-related environment variables
  7. Validate environment variable values
  8. Import TensorFlow/ProteinBERT lazily (only when needed)
  9. Initialize the runtime (seeds, GPU, mixed precision)

Do not import TensorFlow or ProteinBERT before cache config is complete.
"""

import os
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ...utils.logging import get_logger
from ...utils.io import ensure_dir, save_json, write_text
from ...utils.cache_manager import CacheManager
from ...utils.environment import (
    configure_cache_environment,
    validate_environment,
    check_disk_space,
    check_dependencies,
    detect_gpus,
    get_cache_environment,
    get_software_versions,
)


logger = get_logger(__name__)


# ------------------------------------------------------------------
# Default runtime configuration
# ------------------------------------------------------------------

DEFAULT_RUNTIME_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "deterministic": True,
    "mixed_precision": False,
    "memory_growth": True,
    "min_disk_space_gb": 1.0,
}


# ------------------------------------------------------------------
# RuntimeReport
# ------------------------------------------------------------------


@dataclass
class RuntimeReport:
    """Publication-quality runtime metadata.

    Every field is recorded for reproducibility.
    """
    initialized_at: str = ""
    initialized: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Cache
    cache_root: str = ""
    cache_env_vars: Dict[str, Optional[str]] = field(default_factory=dict)
    cache_dirs: Dict[str, str] = field(default_factory=dict)
    cache_validated: bool = False

    # Hardware
    gpus: List[Dict[str, Any]] = field(default_factory=list)
    gpu_count: int = 0
    cpu_count: int = 0
    memory_gb: float = 0.0
    disk_free_gb: float = 0.0

    # Software
    python_version: str = ""
    platform: str = ""
    torch_version: str = ""
    cuda_version: Optional[str] = ""
    numpy_version: str = ""

    # Runtime state
    seed: int = 0
    deterministic: bool = True
    mixed_precision: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------
# Runtime
# ------------------------------------------------------------------


class Runtime:
    """ProteinBERT runtime initialization and lifecycle.

    Initialization order (enforced):
      config → CacheManager → directories → disk space → env vars → validation → imports → init

    Idempotent: multiple calls to initialize() are safe.
    No TensorFlow imports at init time (lazy).
    """

    def __init__(
        self,
        workspace_root: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the runtime.

        Args:
            workspace_root: Project workspace root directory.
            config: Runtime configuration dict.
        """
        self._workspace_root = Path(workspace_root).resolve()
        self._config: Dict[str, Any] = dict(DEFAULT_RUNTIME_CONFIG)
        if config:
            self._config.update(config)

        self._cache_manager: Optional[CacheManager] = None
        self._initialized: bool = False
        self._init_order_complete: bool = False
        self._report: RuntimeReport = RuntimeReport()
        self._report.initialized_at = datetime.now().isoformat()

        logger.debug(f"Runtime created (workspace={self._workspace_root})")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def cache_manager(self) -> Optional[CacheManager]:
        return self._cache_manager

    @property
    def report(self) -> RuntimeReport:
        return self._report

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._config)

    # ------------------------------------------------------------------
    # Initialization (idempotent)
    # ------------------------------------------------------------------

    def initialize(self) -> "Runtime":
        """Full runtime initialization.

        Steps in order:
          1. Set up CacheManager
          2. Ensure cache directories
          3. Validate directories
          4. Check disk space
          5. Configure cache environment variables
          6. Validate environment
          7. Set deterministic seeds
          8. Configure GPU
          9. Generate runtime report
          10. Mark initialized

        Returns:
            self for chaining.

        Raises:
            RuntimeError: If any step fails.
        """
        if self._initialized:
            logger.debug("Runtime already initialized (idempotent)")
            return self

        errors: List[str] = []

        # Step 1-2: CacheManager + directories
        try:
            cache_root = self._workspace_root / "outputs" / "cache"
            self._cache_manager = CacheManager(cache_root)
            self._cache_manager.ensure_all()
            self._report.cache_root = str(cache_root)
            self._report.cache_dirs = {
                k: str(v) for k, v in self._cache_manager.cache_dirs.items()
            }
        except Exception as e:
            errors.append(f"CacheManager init failed: {e}")

        # Step 3: Validate directories
        if self._cache_manager is not None:
            try:
                self._cache_manager.validate()
                self._report.cache_validated = True
            except RuntimeError as e:
                errors.append(f"Cache validation failed: {e}")

        # Step 4: Check disk space
        try:
            disk_info = check_disk_space(
                self._workspace_root,
                self._config.get("min_disk_space_gb", 5.0),
            )
            self._report.disk_free_gb = disk_info.get("free_gb", -1.0)
        except RuntimeError as e:
            errors.append(str(e))

        # Step 5: Configure cache environment variables
        if self._cache_manager is not None:
            try:
                self._cache_manager.configure_environment()
                self._report.cache_env_vars = self._cache_manager.get_environment()
            except Exception as e:
                errors.append(f"Cache env vars failed: {e}")

        # Step 6: Validate environment (no imports)
        try:
            validate_env_result = validate_environment(
                workspace_root=self._workspace_root,
                cache_root=self._workspace_root / "outputs" / "cache",
                min_disk_gb=self._config.get("min_disk_space_gb", 5.0),
                require_gpu=False,
            )
        except RuntimeError as e:
            errors.append(str(e))

        # Step 7: Set deterministic seeds
        seed = self._config.get("seed", 42)
        deterministic = self._config.get("deterministic", True)
        try:
            self._set_seeds(seed, deterministic)
            self._report.seed = seed
            self._report.deterministic = deterministic
        except Exception as e:
            errors.append(f"Seed initialization failed: {e}")

        # Step 8: Configure GPU
        try:
            self._configure_gpu()
            gpus = detect_gpus()
            self._report.gpus = gpus
            self._report.gpu_count = len(gpus)
            self._report.mixed_precision = self._config.get("mixed_precision", False)
        except Exception as e:
            errors.append(f"GPU configuration failed: {e}")

        # Step 9: Environment/reproducibility info
        try:
            self._report.python_version = sys.version.split()[0]
            self._report.platform = platform.platform()
            self._report.torch_version = torch.__version__
            self._report.cuda_version = torch.version.cuda
            self._report.numpy_version = np.__version__
            self._report.cpu_count = os.cpu_count() or 0
            import psutil
            self._report.memory_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:
            pass

        if errors:
            self._report.errors = errors
            self._report.initialized = False
            error_msg = "Runtime initialization failed:\n" + "\n".join(errors)
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self._initialized = True
        self._init_order_complete = True
        self._report.initialized = True

        logger.info(
            f"Runtime initialized: seed={seed}, "
            f"gpus={self._report.gpu_count}, "
            f"cache={self._report.cache_root}"
        )

        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seeds(seed: int, deterministic: bool = True) -> None:
        """Set deterministic random seeds for all frameworks."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except AttributeError:
                pass

        logger.debug(f"Seeds set: seed={seed}, deterministic={deterministic}")

    @staticmethod
    def _configure_gpu() -> None:
        """Configure GPU settings."""
        if not torch.cuda.is_available():
            logger.info("No GPU detected — using CPU")
            return

        # Enable TF32 for Ampere GPUs
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except AttributeError:
            pass

        logger.info(f"GPU available: {torch.cuda.device_count()} device(s)")

    # ------------------------------------------------------------------
    # Reports and evidence
    # ------------------------------------------------------------------

    def generate_reports(self, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
        """Generate all runtime evidence files.

        Creates:
          - runtime_report.md
          - runtime_environment.json
          - runtime_statistics.json
          - hardware.json
          - software_versions.json
          - initialization_log.json
          - cache_manifest.json

        Args:
            output_dir: Output directory (default: outputs/runtime/).

        Returns:
            Dict of artifact name -> Path.
        """
        if output_dir is None:
            output_dir = self._workspace_root / "outputs" / "runtime"
        out = Path(output_dir)
        ensure_dir(out)

        artifacts: Dict[str, Path] = {}

        # 1. Runtime report (Markdown)
        artifacts["runtime_report"] = self._save_runtime_report(out)

        # 2. Runtime environment (JSON)
        env = {
            "initialized": self._initialized,
            "python_version": self._report.python_version,
            "platform": self._report.platform,
            "torch_version": self._report.torch_version,
            "cuda_version": self._report.cuda_version,
            "numpy_version": self._report.numpy_version,
            "seed": self._report.seed,
            "deterministic": self._report.deterministic,
            "mixed_precision": self._report.mixed_precision,
            "cache_root": self._report.cache_root,
            "cache_env_vars": self._report.cache_env_vars,
        }
        artifacts["runtime_environment"] = save_json(env, out / "runtime_environment.json")

        # 3. Runtime statistics (JSON)
        stats = {
            "gpu_count": self._report.gpu_count,
            "cpu_count": self._report.cpu_count,
            "memory_gb": self._report.memory_gb,
            "disk_free_gb": self._report.disk_free_gb,
            "cache_validated": self._report.cache_validated,
            "initialized_at": self._report.initialized_at,
        }
        artifacts["runtime_statistics"] = save_json(stats, out / "runtime_statistics.json")

        # 4. Hardware info (JSON)
        hardware = {
            "gpus": self._report.gpus,
            "gpu_count": self._report.gpu_count,
            "cpu_count": self._report.cpu_count,
            "memory_gb": self._report.memory_gb,
            "disk_free_gb": self._report.disk_free_gb,
        }
        artifacts["hardware"] = save_json(hardware, out / "hardware.json")

        # 5. Software versions (JSON)
        versions = get_software_versions()
        artifacts["software_versions"] = save_json(versions, out / "software_versions.json")

        # 6. Initialization log (JSON)
        init_log = {
            "initialized": self._initialized,
            "initialized_at": self._report.initialized_at,
            "init_order_complete": self._init_order_complete,
            "errors": self._report.errors,
            "warnings": self._report.warnings,
            "config": self._config,
        }
        artifacts["initialization_log"] = save_json(init_log, out / "initialization_log.json")

        # 7. Cache manifest (JSON)
        if self._cache_manager is not None:
            artifacts["cache_manifest"] = self._cache_manager.save_manifest(out)

        logger.info(f"Runtime reports saved to {out} ({len(artifacts)} files)")
        return artifacts

    def _save_runtime_report(self, output_dir: Path) -> Path:
        """Generate the main runtime report in Markdown."""
        report_path = output_dir / "runtime_report.md"

        lines: List[str] = []
        lines.append("# Runtime Report")
        lines.append("")
        lines.append(f"- **Generated**: {self._report.initialized_at}")
        lines.append(f"- **Initialized**: {self._report.initialized}")
        lines.append("")

        # 1. Initialization
        lines.append("## Initialization")
        lines.append("")
        lines.append(f"- **Order complete**: {self._init_order_complete}")
        lines.append(f"- **Seed**: {self._report.seed}")
        lines.append(f"- **Deterministic**: {self._report.deterministic}")
        lines.append(f"- **Mixed precision**: {self._report.mixed_precision}")
        if self._report.errors:
            lines.append(f"- **Errors**: {len(self._report.errors)}")
            for err in self._report.errors:
                lines.append(f"  - {err}")
        lines.append("")

        # 2. Cache
        lines.append("## Cache Configuration")
        lines.append("")
        lines.append(f"- **Cache root**: {self._report.cache_root}")
        lines.append(f"- **Cache validated**: {self._report.cache_validated}")
        lines.append("")
        lines.append("| Directory | Path |")
        lines.append("|-----------|------|")
        for name, path in sorted(self._report.cache_dirs.items()):
            lines.append(f"| {name} | {path} |")
        lines.append("")
        lines.append("| Environment Variable | Value |")
        lines.append("|---------------------|-------|")
        for var, val in sorted(self._report.cache_env_vars.items()):
            lines.append(f"| {var} | {val or 'Not set'} |")
        lines.append("")

        # 3. Hardware
        lines.append("## Hardware")
        lines.append("")
        lines.append(f"- **GPUs**: {self._report.gpu_count}")
        lines.append(f"- **CPUs**: {self._report.cpu_count}")
        lines.append(f"- **Memory**: {self._report.memory_gb:.1f} GB")
        lines.append(f"- **Disk free**: {self._report.disk_free_gb:.1f} GB")
        lines.append("")
        if self._report.gpus:
            lines.append("| GPU | Name | Memory | Compute Capability |")
            lines.append("|-----|------|--------|-------------------|")
            for gpu in self._report.gpus:
                lines.append(
                    f"| {gpu.get('index', '?')} | {gpu.get('name', '?')} "
                    f"| {gpu.get('total_memory_gb', '?'):.1f} GB "
                    f"| {gpu.get('compute_capability', '?')} |"
                )
            lines.append("")

        # 4. Software
        lines.append("## Software")
        lines.append("")
        lines.append(f"- **Python**: {self._report.python_version}")
        lines.append(f"- **Platform**: {self._report.platform}")
        lines.append(f"- **PyTorch**: {self._report.torch_version}")
        lines.append(f"- **CUDA**: {self._report.cuda_version}")
        lines.append(f"- **NumPy**: {self._report.numpy_version}")
        lines.append("")

        lines.append("---")
        lines.append("*Report generated by ProteinBERT runtime*")

        content = "\n".join(lines)
        report_path.write_text(content, encoding="utf-8")
        logger.info(f"Runtime report saved to {report_path}")
        return report_path

    # ------------------------------------------------------------------
    # TensorFlow lazy initialization
    # ------------------------------------------------------------------

    def initialize_tensorflow(self) -> Any:
        """Lazy TensorFlow initialization with cache configuration.

        Must be called only after configure_environment() has been called.

        Returns:
            TensorFlow module (or None if not installed).

        Raises:
            ImportError: If TensorFlow is not installed.
        """
        import tensorflow as tf  # lazy

        # Configure GPU memory growth
        if self._config.get("memory_growth", True):
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except (RuntimeError, ValueError) as e:
                    logger.warning(f"TF memory growth config failed for {gpu}: {e}")

        # Set TF random seed
        seed = self._config.get("seed", 42)
        tf.random.set_seed(seed)

        # Mixed precision
        if self._config.get("mixed_precision", False):
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                logger.info("TensorFlow mixed precision enabled")
            except Exception as e:
                logger.warning(f"TF mixed precision not available: {e}")

        logger.info("TensorFlow initialized (memory_growth={}, seed={})".format(
            self._config.get("memory_growth", True), seed
        ))

        return tf

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """Return a summary of the runtime state."""
        return {
            "initialized": self._initialized,
            "seed": self._report.seed,
            "gpu_count": self._report.gpu_count,
            "cache_root": self._report.cache_root,
            "cache_validated": self._report.cache_validated,
            "cache_env_vars_configured": bool(self._report.cache_env_vars),
            "disk_free_gb": self._report.disk_free_gb,
            "errors": len(self._report.errors),
            "warnings": len(self._report.warnings),
        }

    def __repr__(self) -> str:
        return (
            f"Runtime(initialized={self._initialized}, "
            f"seed={self._report.seed}, "
            f"gpus={self._report.gpu_count})"
        )
