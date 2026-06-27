"""Tests for CacheManager, Environment, and Runtime modules.

Covers:
  - CacheManager: init, directory management, env vars, validation, manifest
  - Environment: cache env var config, directory validation, disk space,
    dependency checking (no download), GPU detection, pre-flight validation
  - Runtime: initialization lifecycle, idempotency, lazy TF, report generation,
    wrapper integration
  - no TF import at module level
"""

import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.cache_manager import CacheManager
from src.utils.environment import (
    configure_cache_environment,
    get_cache_environment,
    validate_directory,
    validate_directories,
    check_disk_space,
    check_package_available,
    check_dependencies,
    check_tensorflow_available,
    check_proteinbert_available,
    detect_gpus,
    validate_environment,
    get_software_versions,
)
from src.models.proteinbert.runtime import Runtime, RuntimeReport
from src.models.proteinbert import ProteinBERTModel


# ======================================================================
# CacheManager Tests
# ======================================================================


class TestCacheManagerInit:
    """CacheManager creation and default state."""

    def test_init_creates_dirs_dict(self, tmp_path):
        cm = CacheManager(tmp_path)
        assert isinstance(cm.cache_dirs, dict)
        assert "hub" in cm.cache_dirs
        assert "checkpoints" in cm.cache_dirs
        assert "datasets" in cm.cache_dirs

    def test_init_not_configured(self, tmp_path):
        cm = CacheManager(tmp_path)
        assert cm.env_configured is False

    def test_init_resolves_path(self, tmp_path):
        cm = CacheManager(str(tmp_path))
        assert cm.workspace_root == tmp_path.resolve()

    def test_get_unknown_dir_raises(self, tmp_path):
        cm = CacheManager(tmp_path)
        with pytest.raises(KeyError):
            cm.get_cache_dir("nonexistent")

    def test_repr(self, tmp_path):
        cm = CacheManager(tmp_path)
        r = repr(cm)
        assert "CacheManager" in r
        assert str(tmp_path) in r


class TestCacheManagerDirectories:
    """CacheManager directory creation and access."""

    def test_ensure_all_creates_all(self, tmp_path):
        cm = CacheManager(tmp_path)
        created = cm.ensure_all()
        for name, path in created.items():
            assert Path(path).exists()
        assert cm.workspace_root.exists()

    def test_ensure_single(self, tmp_path):
        cm = CacheManager(tmp_path)
        hub_dir = cm.ensure_dir("hub")
        assert hub_dir.exists()
        assert hub_dir == tmp_path / "hub"

    def test_get_checkpoint_dir(self, tmp_path):
        cm = CacheManager(tmp_path)
        ckpt = cm.get_checkpoint_dir("exp_001")
        assert ckpt == tmp_path / "checkpoints" / "exp_001"

    def test_get_hub_dir(self, tmp_path):
        cm = CacheManager(tmp_path)
        assert cm.get_hub_dir() == tmp_path / "hub"

    def test_get_datasets_dir(self, tmp_path):
        cm = CacheManager(tmp_path)
        assert cm.get_datasets_dir() == tmp_path / "datasets"

    def test_get_models_dir(self, tmp_path):
        cm = CacheManager(tmp_path)
        assert cm.get_models_dir() == tmp_path / "models"


class TestCacheManagerEnvVars:
    """CacheManager environment variable configuration."""

    def test_configure_environment_sets_vars(self, tmp_path):
        cm = CacheManager(tmp_path)
        cm.ensure_all()
        configured = cm.configure_environment()
        for var_name in ["HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME"]:
            assert var_name in configured
            assert configured[var_name].startswith(str(tmp_path))
        assert cm.env_configured is True

    def test_get_environment_returns_dict(self, tmp_path):
        cm = CacheManager(tmp_path)
        env = cm.get_environment()
        assert isinstance(env, dict)
        for var in ["HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME"]:
            assert var in env

    def test_configure_environment_idempotent(self, tmp_path):
        cm = CacheManager(tmp_path)
        cm.ensure_all()
        cm.configure_environment()
        cm.configure_environment()


class TestCacheManagerValidation:
    """CacheManager validation and disk space."""

    def test_validate_passes_when_dirs_exist(self, tmp_path):
        cm = CacheManager(tmp_path)
        cm.ensure_all()
        result = cm.validate()
        assert result["all_ok"] is True

    def test_available_disk_space_positive(self, tmp_path):
        cm = CacheManager(tmp_path)
        space = cm.available_disk_space()
        assert space > 0


class TestCacheManagerManifest:
    """CacheManager download manifest tracking."""

    def test_record_and_check_download(self, tmp_path):
        cm = CacheManager(tmp_path)
        f = tmp_path / "test_file.bin"
        f.write_bytes(b"data")
        cm.record_download("https://example.com/file.bin", f)
        assert cm.is_downloaded("https://example.com/file.bin") is True

    def test_not_downloaded(self, tmp_path):
        cm = CacheManager(tmp_path)
        assert cm.is_downloaded("https://example.com/missing.bin") is False

    def test_save_manifest(self, tmp_path):
        cm = CacheManager(tmp_path)
        f = tmp_path / "data.bin"
        f.write_bytes(b"test")
        cm.record_download("https://example.com/data.bin", f)
        out = tmp_path / "manifests"
        path = cm.save_manifest(out)
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["files"]) == 1


class TestCacheManagerClear:
    """CacheManager cache clearing."""

    def test_clear_single_cache(self, tmp_path):
        cm = CacheManager(tmp_path)
        cm.ensure_all()
        hub_dir = cm.get_cache_dir("hub")
        (hub_dir / "test.txt").write_text("hello")
        cm.clear("hub")
        assert hub_dir.exists()
        assert not (hub_dir / "test.txt").exists()

    def test_clear_all(self, tmp_path):
        cm = CacheManager(tmp_path)
        cm.ensure_all()
        for d in cm.cache_dirs.values():
            (Path(d) / "test.txt").write_text("data")
        cm.clear()
        for d in cm.cache_dirs.values():
            assert Path(d).exists()


class TestCacheManagerInfo:
    """CacheManager info and reporting."""

    def test_info_returns_summary(self, tmp_path):
        cm = CacheManager(tmp_path)
        info = cm.info()
        assert "workspace_root" in info
        assert "directories" in info
        assert info["env_configured"] is False


# ======================================================================
# Environment Tests
# ======================================================================


class TestEnvironmentCacheConfig:
    """Environment cache configuration functions."""

    def test_configure_cache_environment(self, tmp_path):
        configured = configure_cache_environment(tmp_path)
        assert "HF_HOME" in configured
        assert configured["HF_HOME"].startswith(str(tmp_path))

    def test_get_cache_environment(self, tmp_path):
        env = get_cache_environment()
        assert isinstance(env, dict)

    def test_cache_vars_include_all_expected(self, tmp_path):
        configure_cache_environment(tmp_path)
        env = get_cache_environment()
        for var in ["HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME"]:
            assert var in env


class TestEnvironmentDirectoryValidation:
    """Directory validation functions."""

    def test_validate_directory_creates(self, tmp_path):
        d = tmp_path / "new_dir"
        result = validate_directory(d, create=True)
        assert result["exists"] is True
        assert result["writable"] is True

    def test_validate_directory_not_found(self, tmp_path):
        d = tmp_path / "missing"
        with pytest.raises(RuntimeError, match="does not exist"):
            validate_directory(d, create=False)

    def test_validate_directories_all_pass(self, tmp_path):
        dirs = {"a": tmp_path / "a", "b": tmp_path / "b"}
        results = validate_directories(dirs)
        assert "a" in results
        assert results["a"]["exists"] is True

    def test_validate_directories_raises_on_failure(self, tmp_path):
        dirs = {"a": tmp_path / "nonexistent"}
        with pytest.raises(RuntimeError):
            validate_directories(dirs, create=False)


class TestEnvironmentDiskSpace:
    """Disk space checking."""

    def test_check_disk_space_returns_info(self, tmp_path):
        result = check_disk_space(tmp_path, minimum_gb=0.0)
        assert "free_gb" in result
        assert result["free_gb"] > 0
        assert result["sufficient"] is True

    def test_check_disk_space_insufficient(self, tmp_path):
        with pytest.raises(RuntimeError):
            check_disk_space(tmp_path, minimum_gb=1e9)


class TestEnvironmentDependencies:
    """Check package availability without triggering downloads."""

    def test_check_package_available_true(self):
        assert check_package_available("torch") is True

    def test_check_package_available_false(self):
        assert check_package_available("nonexistent_package_xyz") is False

    def test_check_dependencies_required_pass(self):
        result = check_dependencies(required=["torch", "numpy"], silent=True)
        assert result["status"] == "ok"

    def test_check_dependencies_missing_required_raises(self):
        with pytest.raises(RuntimeError):
            check_dependencies(
                required=["nonexistent_pkg_123"], silent=True
            )

    def test_check_dependencies_optional_not_found(self):
        result = check_dependencies(
            optional=["nonexistent_pkg_456"], silent=True
        )
        assert result["status"] == "ok"
        assert "nonexistent_pkg_456" in result["missing_optional"]

    def test_check_tensorflow_available(self):
        available = check_tensorflow_available()
        assert isinstance(available, bool)

    def test_check_proteinbert_available(self):
        available = check_proteinbert_available()
        assert isinstance(available, bool)


class TestEnvironmentGPU:
    """GPU detection."""

    def test_detect_gpus_returns_list(self):
        gpus = detect_gpus()
        assert isinstance(gpus, list)


class TestEnvironmentPreFlight:
    """Pre-flight validation."""

    def test_validate_environment_passes(self, tmp_path):
        ws = tmp_path / "workspace"
        cache = tmp_path / "cache"
        ws.mkdir(parents=True)
        result = validate_environment(ws, cache, min_disk_gb=0.0)
        assert result["status"] == "ok"

    def test_validate_environment_checks_gpu(self, tmp_path):
        ws = tmp_path / "ws"
        cache = tmp_path / "cache"
        ws.mkdir(parents=True)
        result = validate_environment(ws, cache, min_disk_gb=0.0)
        assert "gpus" in result["checks"]


class TestEnvironmentSoftwareVersions:
    """Software version reporting."""

    def test_get_software_versions_contains_python(self):
        versions = get_software_versions()
        assert "python" in versions
        assert "torch" in versions
        assert "platform" in versions

    def test_get_software_versions_torch_version(self):
        versions = get_software_versions()
        assert versions["torch"] == torch.__version__


# ======================================================================
# Runtime Tests
# ======================================================================


class TestRuntimeInit:
    """Runtime object creation and default state."""

    def test_create_not_initialized(self, tmp_path):
        rt = Runtime(tmp_path)
        assert rt.initialized is False
        assert rt.cache_manager is None

    def test_create_default_config(self, tmp_path):
        rt = Runtime(tmp_path)
        assert rt.config["seed"] == 42
        assert rt.config["deterministic"] is True

    def test_create_custom_config(self, tmp_path):
        rt = Runtime(tmp_path, config={"seed": 99, "min_disk_space_gb": 10.0})
        assert rt.config["seed"] == 99
        assert rt.config["min_disk_space_gb"] == 10.0

    def test_report_before_init(self, tmp_path):
        rt = Runtime(tmp_path)
        assert rt.report.initialized is False
        assert rt.report.initialized_at != ""

    def test_repr(self, tmp_path):
        rt = Runtime(tmp_path)
        r = repr(rt)
        assert "Runtime" in r
        assert "initialized=False" in r


class TestRuntimeInitialize:
    """Full runtime initialization lifecycle."""

    def test_initialize_sets_initialized(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        assert rt.initialized is True

    def test_initialize_creates_cache_manager(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        assert rt.cache_manager is not None

    def test_configured_cache_dirs_exist(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        for name, path in rt.cache_manager.cache_dirs.items():
            assert path.exists(), f"Cache dir '{name}' does not exist: {path}"

    def test_initialize_idempotent(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        rt.initialize()
        assert rt.initialized is True

    def test_initialize_report_populated(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        assert rt.report.seed == 42
        assert rt.report.deterministic is True
        assert rt.report.cache_validated is True
        assert rt.report.cache_root != ""
        assert rt.report.disk_free_gb > 0

    def test_initialize_cache_root_under_workspace(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        assert str(tmp_path) in rt.report.cache_root

    def test_info_after_init(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        info = rt.info()
        assert info["initialized"] is True
        assert info["seed"] == 42
        assert info["cache_validated"] is True


class TestRuntimeReportGeneration:
    """Runtime report generation."""

    def test_generate_reports_creates_files(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        out = tmp_path / "runtime_out"
        artifacts = rt.generate_reports(output_dir=out)
        assert len(artifacts) >= 6
        for name, path in artifacts.items():
            assert Path(path).exists(), f"Artifact missing: {name} -> {path}"

    def test_runtime_report_md_content(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        out = tmp_path / "runtime_out"
        artifacts = rt.generate_reports(output_dir=out)
        md_path = artifacts["runtime_report"]
        content = md_path.read_text()
        assert "# Runtime Report" in content
        assert "## Initialization" in content
        assert "## Cache Configuration" in content
        assert "## Hardware" in content
        assert "## Software" in content

    def test_runtime_environment_json(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        out = tmp_path / "runtime_out"
        artifacts = rt.generate_reports(output_dir=out)
        data = json.loads(Path(artifacts["runtime_environment"]).read_text())
        assert data["initialized"] is True
        assert data["python_version"] != ""

    def test_runtime_statistics_json(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        out = tmp_path / "runtime_out"
        artifacts = rt.generate_reports(output_dir=out)
        data = json.loads(Path(artifacts["runtime_statistics"]).read_text())
        assert "gpu_count" in data
        assert "memory_gb" in data
        assert "disk_free_gb" in data

    def test_hardware_json(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        out = tmp_path / "runtime_out"
        artifacts = rt.generate_reports(output_dir=out)
        data = json.loads(Path(artifacts["hardware"]).read_text())
        assert "gpus" in data
        assert "cpu_count" in data

    def test_software_versions_json(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        out = tmp_path / "runtime_out"
        artifacts = rt.generate_reports(output_dir=out)
        data = json.loads(Path(artifacts["software_versions"]).read_text())
        assert "python" in data
        assert "torch" in data

    def test_initialization_log_json(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        out = tmp_path / "runtime_out"
        artifacts = rt.generate_reports(output_dir=out)
        data = json.loads(Path(artifacts["initialization_log"]).read_text())
        assert data["initialized"] is True
        assert data["config"]["seed"] == 42

    def test_cache_manifest_json(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        out = tmp_path / "runtime_out"
        artifacts = rt.generate_reports(output_dir=out)
        assert "cache_manifest" in artifacts


class TestRuntimeDefaultOutputDir:
    """Runtime report generation with default output directory."""

    def test_default_output_dir_under_workspace(self, tmp_path):
        rt = Runtime(tmp_path)
        rt.initialize()
        artifacts = rt.generate_reports()
        for path in artifacts.values():
            assert str(tmp_path) in str(path)


class TestRuntimeTensorFlowLazy:
    """Runtime lazy TensorFlow initialization."""

    def test_initialize_tensorflow_returns_module(self, tmp_path):
        import importlib
        tf_available = importlib.util.find_spec("tensorflow") is not None
        if not tf_available:
            pytest.skip("TensorFlow not installed")
        rt = Runtime(tmp_path)
        rt.initialize()
        tf = rt.initialize_tensorflow()
        assert tf is not None
        assert hasattr(tf, "__version__")

    def test_initialize_tensorflow_before_env_config(self, tmp_path):
        import importlib
        tf_available = importlib.util.find_spec("tensorflow") is not None
        if not tf_available:
            pytest.skip("TensorFlow not installed")
        rt = Runtime(tmp_path)
        rt.initialize()
        tf = rt.initialize_tensorflow()
        import os
        assert os.environ.get("TFHUB_CACHE_DIR", "").startswith(str(tmp_path))


# ======================================================================
# Wrapper Integration Tests
# ======================================================================


class TestWrapperRuntimeIntegration:
    """ProteinBERTModel integration with CacheManager and Runtime."""

    def test_initialize_runtime_on_wrapper(self, tmp_path):
        model = ProteinBERTModel()
        rt = model.initialize_runtime(workspace_root=tmp_path)
        assert rt.initialized is True
        assert model.runtime is rt
        assert model.cache_manager is rt.cache_manager

    def test_initialize_runtime_idempotent(self, tmp_path):
        model = ProteinBERTModel()
        model.initialize_runtime(workspace_root=tmp_path)
        model.initialize_runtime(workspace_root=tmp_path)
        assert model.runtime.initialized is True

    def test_cache_manager_property(self, tmp_path):
        model = ProteinBERTModel()
        assert model.cache_manager is None
        model.initialize_runtime(workspace_root=tmp_path)
        assert model.cache_manager is not None

    def test_runtime_property_before_init(self):
        model = ProteinBERTModel()
        assert model.runtime is None

    def test_generate_runtime_reports(self, tmp_path):
        model = ProteinBERTModel()
        model.initialize_runtime(workspace_root=tmp_path)
        out = tmp_path / "reports"
        artifacts = model.generate_runtime_reports(output_dir=out)
        assert len(artifacts) >= 6

    def test_generate_runtime_reports_without_init_raises(self):
        model = ProteinBERTModel()
        with pytest.raises(RuntimeError, match="Runtime not initialized"):
            model.generate_runtime_reports()

    def test_validate_environment_on_wrapper(self, tmp_path):
        model = ProteinBERTModel()
        model.initialize_runtime(workspace_root=tmp_path)
        result = model.validate_environment()
        assert "status" in result


# ======================================================================
# RuntimeReport Tests
# ======================================================================


class TestRuntimeReport:
    """RuntimeReport dataclass."""

    def test_defaults(self):
        r = RuntimeReport()
        assert r.initialized is False
        assert r.seed == 0
        assert r.gpu_count == 0

    def test_to_dict(self):
        r = RuntimeReport()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["initialized"] is False

    def test_roundtrip_to_dict(self):
        r = RuntimeReport()
        r.initialized = True
        r.seed = 42
        r.gpu_count = 1
        d = r.to_dict()
        assert d["initialized"] is True
        assert d["seed"] == 42
        assert d["gpu_count"] == 1
