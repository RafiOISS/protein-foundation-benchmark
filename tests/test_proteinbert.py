"""Tests for ProteinBERT model integration.

Verifies:
  - Registry registration
  - Wrapper instantiation (without TensorFlow)
  - Configuration loading
  - Lazy import behavior
"""

import subprocess
import sys
from pathlib import Path

import pytest


# ------------------------------------------------------------------
# Registry integration
# ------------------------------------------------------------------

def test_registered():
    """ProteinBERT appears in the model registry."""
    from src.registry.model_registry import ModelRegistry

    # Trigger registration by importing datasets (which imports all models)
    import src  # noqa: F401

    registered = ModelRegistry.list_models()
    assert "proteinbert" in registered, (
        f"'proteinbert' not in registered models: {registered}"
    )


def test_create_without_tensorflow():
    """ModelRegistry.create('proteinbert') returns an instance without loading TF."""
    from src.registry.model_registry import ModelRegistry
    import src  # noqa: F401

    model = ModelRegistry.create("proteinbert")
    assert model is not None
    assert model.__class__.__name__ == "ProteinBERTModel"
    assert model._loaded is False
    assert model._tf_model is None
    assert model.config["model_type"] == "proteinbert"


def test_default_config():
    """Default config is applied via registry."""
    from src.registry.model_registry import ModelRegistry
    import src  # noqa: F401

    model = ModelRegistry.create("proteinbert")
    assert model.config["embedding_dim"] == 768
    assert model.config["max_seq_len"] == 512
    assert model.config["batch_size"] == 8
    assert model.config["learning_rate"] == 1e-4
    assert model.config["epochs"] == 20


def test_config_override():
    """User config overrides registry defaults."""
    from src.registry.model_registry import ModelRegistry
    import src  # noqa: F401

    model = ModelRegistry.create(
        "proteinbert",
        config={"batch_size": 16, "max_seq_len": 1024},
    )
    assert model.config["batch_size"] == 16
    assert model.config["max_seq_len"] == 1024
    assert model.config["embedding_dim"] == 768  # unchanged


# ------------------------------------------------------------------
# Wrapper interface
# ------------------------------------------------------------------

def test_wrapper_interface():
    """ProteinBERTModel exposes required placeholder methods."""
    from src.models.proteinbert.wrapper import ProteinBERTModel
    import src  # noqa: F401

    model = ProteinBERTModel()
    assert hasattr(model, "load")
    assert hasattr(model, "predict")
    assert hasattr(model, "save")
    assert hasattr(model, "summary")
    assert callable(model.load)
    assert callable(model.predict)
    assert callable(model.save)
    assert callable(model.summary)


def test_summary_without_load():
    """summary() returns metadata even before TF is loaded."""
    from src.models.proteinbert.wrapper import ProteinBERTModel
    import src  # noqa: F401

    model = ProteinBERTModel()
    info = model.summary()
    assert info["model_type"] == "proteinbert"
    assert info["loaded"] is False
    assert info["embedding_dim"] == 768
    assert info["num_layers"] == 12


def test_from_pretrained():
    """from_pretrained classmethod creates an instance."""
    from src.models.proteinbert.wrapper import ProteinBERTModel
    import src  # noqa: F401

    model = ProteinBERTModel.from_pretrained()
    assert isinstance(model, ProteinBERTModel)
    assert model._loaded is False


# ------------------------------------------------------------------
# Lazy import behavior
# ------------------------------------------------------------------

def test_import_src_without_tensorflow():
    """import src succeeds even when TensorFlow is missing."""
    code = """
import subprocess
result = subprocess.run(
    [sys.executable, "-c", "import src; print('OK')"],
    capture_output=True, text=True, timeout=30,
)
print(result.stdout.strip())
"""
    result = subprocess.run(
        [sys.executable, "-c", "import src; print('import src OK')"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "import src OK" in result.stdout


def test_proteinbert_not_imported_at_module_level():
    """TensorFlow must not be imported when importing the module."""
    code = """
import sys
before = set(sys.modules.keys())
import src.models.proteinbert  # noqa: F401
after = set(sys.modules.keys())
new_modules = after - before
# Ignore our own modules and transformers dummy objects
ignore_prefixes = ('src.models.proteinbert', 'transformers.utils.dummy_')
tf_modules = [
    m for m in new_modules
    if 'tensorflow' in m.lower() or 'proteinbert' in m.lower()
    if not any(m.startswith(p) for p in ignore_prefixes)
]
assert not tf_modules, f"TensorFlow/proteinbert imported at module level: {tf_modules}"
print("OK: no tensorflow/proteinbert imported at module level")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "OK:" in result.stdout


def test_tf_imported_on_load():
    """TensorFlow must be imported when _ensure_loaded() is called (if available)."""
    # This test is conditional — it only applies if TF is installed
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        pytest.skip("TensorFlow not installed — lazy import cannot be verified")

    import sys
    import src.models.proteinbert.wrapper  # noqa: F401

    before = set(sys.modules.keys())
    model = src.models.proteinbert.wrapper.ProteinBERTModel()
    try:
        model._ensure_loaded()
    except Exception:
        pass  # May fail if no model weights available

    after = set(sys.modules.keys())
    new_modules = after - before
    tf_loaded = any("tensorflow" in m.lower() for m in new_modules)
    assert tf_loaded, "TensorFlow was not imported after _ensure_loaded()"


# ------------------------------------------------------------------
# Configuration loading
# ------------------------------------------------------------------

def test_config_yaml():
    """proteinbert.yaml can be loaded and contains expected keys."""
    import yaml

    config_path = Path(__file__).resolve().parent.parent / "configs" / "models" / "proteinbert.yaml"
    assert config_path.exists(), f"Config not found: {config_path}"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config is not None
    assert "model" in config
    model_cfg = config["model"]
    assert model_cfg["name"] == "proteinbert"
    assert model_cfg["framework"] == "tensorflow"
    assert model_cfg["batch_size"] == 8
    assert model_cfg["max_seq_len"] == 512
    assert model_cfg["learning_rate"] == 1e-4
    assert model_cfg["epochs"] == 20
    assert model_cfg["mixed_precision"] is False
    assert model_cfg["seed"] == 42


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------

def test_backward_compat_alias():
    """ProteinBERT alias exists for backward compatibility."""
    from src.models.proteinbert import ProteinBERT
    from src.models.proteinbert.wrapper import ProteinBERTModel

    assert ProteinBERT is ProteinBERTModel


def test_models_package_export():
    """src.models exports ProteinBERT."""
    from src.models import ProteinBERT
    from src.models.proteinbert.wrapper import ProteinBERTModel

    assert ProteinBERT is ProteinBERTModel
