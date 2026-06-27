"""Tests for CheckpointManager and ProteinBERTTrainer.

Covers:
  - CheckpointManager: init, paths, save, load, resume, manifest, pretrained
  - ProteinBERTTrainer: init, config, build methods, history, evidence
  - TensorFlow-dependent tests (skip gracefully if unavailable)
  - no TF import at module level
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models.proteinbert.checkpoints import (
    CheckpointManager,
    CheckpointMetadata,
    CHECKPOINT_SUBDIR,
)
from src.models.proteinbert.trainer import (
    ProteinBERTTrainer,
    TrainingHistory,
    DEFAULT_TRAINING_CONFIG,
)
from src.models.proteinbert import ProteinBERTModel


# ======================================================================
# CheckpointMetadata Tests
# ======================================================================


class TestCheckpointMetadata:
    """CheckpointMetadata dataclass."""

    def test_defaults(self):
        m = CheckpointMetadata(epoch=1, path="/tmp/ckpt")
        assert m.epoch == 1
        assert m.timestamp != ""

    def test_to_dict(self):
        m = CheckpointMetadata(epoch=5, path="/tmp/ckpt", loss=0.1, is_best=True)
        d = m.to_dict()
        assert d["epoch"] == 5
        assert d["is_best"] is True
        assert d["loss"] == 0.1

    def test_is_best_flag(self):
        m = CheckpointMetadata(epoch=2, path="/tmp/ckpt", is_best=True)
        assert m.is_best is True


# ======================================================================
# CheckpointManager Tests
# ======================================================================


class TestCheckpointManagerInit:
    """CheckpointManager creation."""

    def test_creates_checkpoint_dir(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        assert cm.checkpoint_root.exists()
        assert cm.checkpoint_root.name == CHECKPOINT_SUBDIR
        assert cm.checkpoint_root.parent.name == "checkpoints"

    def test_checkpoint_root_under_workspace(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        assert str(tmp_path) in str(cm.checkpoint_root)

    def test_initial_manifest_empty(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        assert cm.manifest["checkpoints"] == []
        assert cm.manifest["latest_epoch"] == 0

    def test_info(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        info = cm.info()
        assert info["total_checkpoints"] == 0
        assert info["latest_epoch"] == 0
        assert info["has_best"] is False

    def test_repr(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        r = repr(cm)
        assert "CheckpointManager" in r
        assert CHECKPOINT_SUBDIR in r


class TestCheckpointManagerSaveLoad:
    """Checkpoint save and load."""

    def test_save_and_load_latest(self, tmp_path):
        tf = _require_tf()
        model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])
        model.compile(optimizer="adam", loss="mse")
        model.build((None, 5))

        cm = CheckpointManager(tmp_path)
        cm.save(model, epoch=1, loss=0.5, val_loss=0.6)

        latest = cm.load_latest()
        assert latest is not None
        assert latest["epoch"] == 1
        assert latest["loss"] == 0.5
        assert latest["val_loss"] == 0.6

    def test_save_multiple_epochs(self, tmp_path):
        tf = _require_tf()
        model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])
        model.build((None, 5))

        cm = CheckpointManager(tmp_path)
        cm.save(model, epoch=1, loss=0.5)
        cm.save(model, epoch=2, loss=0.3, val_loss=0.4)
        cm.save(model, epoch=3, loss=0.2, val_loss=0.25, is_best=True)

        latest = cm.load_latest()
        assert latest["epoch"] == 3

        best = cm.load_best()
        assert best is not None
        assert best["epoch"] == 3

        ckpts = cm.list_checkpoints()
        assert len(ckpts) == 3

    def test_load_latest_returns_none_when_empty(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        assert cm.load_latest() is None

    def test_load_best_returns_none_when_empty(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        assert cm.load_best() is None

    def test_get_latest_epoch(self, tmp_path):
        tf = _require_tf()
        model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])
        model.build((None, 5))

        cm = CheckpointManager(tmp_path)
        assert cm.get_latest_epoch() == 0
        cm.save(model, epoch=5)
        assert cm.get_latest_epoch() == 5


class TestCheckpointManagerRestore:
    """Model weight restoration."""

    def test_restore_model(self, tmp_path):
        tf = _require_tf()
        model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])
        model.build((None, 5))

        cm = CheckpointManager(tmp_path)
        cm.save(model, epoch=2, loss=0.3)

        new_model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])
        new_model.build((None, 5))
        restored_epoch = cm.restore_model(new_model)
        assert restored_epoch == 2

    def test_restore_specific_epoch(self, tmp_path):
        tf = _require_tf()
        model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])
        model.build((None, 5))

        cm = CheckpointManager(tmp_path)
        cm.save(model, epoch=1)
        cm.save(model, epoch=2)

        new_model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])
        new_model.build((None, 5))
        restored = cm.restore_model(new_model, epoch=1)
        assert restored == 1

    def test_restore_no_checkpoint(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        restored = cm.restore_model(None)
        assert restored == 0


class TestCheckpointManagerResume:
    """Training resume support."""

    def test_resume_state(self, tmp_path):
        tf = _require_tf()
        model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])
        model.build((None, 5))

        cm = CheckpointManager(tmp_path)
        cm.save(model, epoch=3, loss=0.2, is_best=True)
        state = cm.get_resume_state()
        assert state["epoch"] == 3
        assert state["best"] is not None
        assert len(state["checkpoints"]) == 1

    def test_resume_state_empty(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        state = cm.get_resume_state()
        assert state["epoch"] == 0
        assert state["best"] is None
        assert state["checkpoints"] == []


class TestCheckpointManagerManifest:
    """Manifest persistence."""

    def test_manifest_saved_to_disk(self, tmp_path):
        tf = _require_tf()
        model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])
        model.build((None, 5))

        cm = CheckpointManager(tmp_path)
        cm.save(model, epoch=1)
        assert (tmp_path / "checkpoints" / CHECKPOINT_SUBDIR / "checkpoint_manifest.json").exists()

    def test_manifest_reloads(self, tmp_path):
        tf = _require_tf()
        model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])
        model.build((None, 5))

        cm = CheckpointManager(tmp_path)
        cm.save(model, epoch=1, loss=0.1)

        cm2 = CheckpointManager(tmp_path)
        assert cm2.get_latest_epoch() == 1


class TestCheckpointManagerPretrained:
    """Pretrained download stubs."""

    def test_pretrained_not_downloaded_empty(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        pretrained = cm._find_pretrained_weights(cm.checkpoint_root / "pretrained")
        assert isinstance(pretrained, Path)


# ======================================================================
# TrainingHistory Tests
# ======================================================================


class TestTrainingHistory:
    """TrainingHistory dataclass."""

    def test_empty_history(self):
        h = TrainingHistory()
        assert h.epochs == []

    def test_add_epoch(self):
        h = TrainingHistory()
        h.add_epoch(1, 0.5, 0.6, 0.7, 0.8, 0.001, 10.0)
        assert len(h.epochs) == 1
        assert h.train_loss[0] == 0.5
        assert h.val_accuracy[0] == 0.8

    def test_to_dict(self):
        h = TrainingHistory()
        h.add_epoch(1, 0.5, 0.6, 0.7, 0.8, 0.001, 10.0)
        d = h.to_dict()
        assert "epochs" in d
        assert "train_loss" in d
        assert d["train_loss"] == [0.5]

    def test_to_csv_rows(self):
        h = TrainingHistory()
        h.add_epoch(1, 0.5, 0.6, 0.7, 0.8, 0.001, 10.0)
        rows = h.to_csv_rows()
        assert len(rows) == 1
        assert rows[0]["epoch"] == 1
        assert rows[0]["train_loss"] == 0.5

    def test_multiple_epochs(self):
        h = TrainingHistory()
        h.add_epoch(1, 0.5, 0.6, 0.7, 0.8, 0.001, 10.0)
        h.add_epoch(2, 0.3, 0.4, 0.9, 0.85, 0.0005, 9.5)
        assert len(h.epochs) == 2
        assert h.learning_rates == [0.001, 0.0005]


# ======================================================================
# ProteinBERTTrainer Tests
# ======================================================================


class TestTrainerInit:
    """ProteinBERTTrainer initialization."""

    def test_default_config(self):
        trainer = ProteinBERTTrainer()
        assert trainer.config["epochs"] == 20
        assert trainer.config["optimizer"] == "adam"
        assert trainer.config["early_stopping"] is True

    def test_custom_config(self):
        trainer = ProteinBERTTrainer(config={"epochs": 5, "patience": 3})
        assert trainer.config["epochs"] == 5
        assert trainer.config["patience"] == 3

    def test_not_trained(self):
        trainer = ProteinBERTTrainer()
        assert trainer.trained_epochs == 0
        assert trainer.stopped_early is False

    def test_repr(self):
        trainer = ProteinBERTTrainer()
        r = repr(trainer)
        assert "ProteinBERTTrainer" in r


class TestTrainerBuildMethods:
    """Internal builder methods."""

    def test_build_optimizer_default(self):
        tf = _require_tf()
        trainer = ProteinBERTTrainer()
        opt = trainer._build_optimizer()
        assert isinstance(opt, tf.keras.optimizers.Optimizer)

    def test_build_optimizer_sgd(self):
        tf = _require_tf()
        trainer = ProteinBERTTrainer(config={"optimizer": "sgd"})
        opt = trainer._build_optimizer()
        assert isinstance(opt, tf.keras.optimizers.SGD)

    def test_build_loss(self):
        tf = _require_tf()
        trainer = ProteinBERTTrainer()
        loss = trainer._build_loss()
        assert isinstance(loss, tf.keras.losses.Loss)


class TestTrainerHistory:
    """Training history through the trainer."""

    def test_history_after_construction(self):
        trainer = ProteinBERTTrainer()
        assert trainer.history.epochs == []

    def test_history_type(self):
        trainer = ProteinBERTTrainer()
        assert isinstance(trainer.history, TrainingHistory)


class TestTrainerEvidence:
    """Training evidence generation."""

    def test_save_history_csv(self, tmp_path):
        trainer = ProteinBERTTrainer()
        trainer._history.add_epoch(1, 0.5, 0.6, 0.7, 0.8, 0.001, 10.0)
        trainer._save_history_csv(tmp_path)
        assert (tmp_path / "history.csv").exists()

    def test_save_history_json(self, tmp_path):
        trainer = ProteinBERTTrainer()
        trainer._history.add_epoch(1, 0.5, 0.6, 0.7, 0.8, 0.001, 10.0)
        trainer._save_history_json(tmp_path)
        assert (tmp_path / "history.json").exists()

    def test_save_training_report(self, tmp_path):
        trainer = ProteinBERTTrainer()
        trainer._history.add_epoch(1, 0.5, 0.6, 0.7, 0.8, 0.001, 10.0)
        trainer._save_training_report(tmp_path)
        assert (tmp_path / "training_report.md").exists()
        content = (tmp_path / "training_report.md").read_text()
        assert "Training Report" in content

    def test_save_epoch_times_csv(self, tmp_path):
        trainer = ProteinBERTTrainer()
        trainer._history.add_epoch(1, 0.5, 0.6, 0.7, 0.8, 0.001, 10.0)
        trainer._save_epoch_times_csv(tmp_path)
        assert (tmp_path / "epoch_times.csv").exists()

    def test_save_lr_csv(self, tmp_path):
        trainer = ProteinBERTTrainer()
        trainer._history.add_epoch(1, 0.5, 0.6, 0.7, 0.8, 0.001, 10.0)
        trainer._save_lr_csv(tmp_path)
        assert (tmp_path / "learning_rate.csv").exists()

    def test_save_best_checkpoint_json(self, tmp_path):
        trainer = ProteinBERTTrainer()
        trainer._history.add_epoch(1, 0.5, 0.6, 0.7, 0.8, 0.001, 10.0)
        trainer._save_best_checkpoint_json(tmp_path)
        assert (tmp_path / "best_checkpoint.json").exists()

    def test_save_early_stopping_json_not_saved(self, tmp_path):
        trainer = ProteinBERTTrainer()
        trainer._history.add_epoch(1, 0.5, 0.6, 0.7, 0.8, 0.001, 10.0)
        trainer._save_early_stopping_json(tmp_path)
        assert (tmp_path / "early_stopping.json").exists()

    def test_generate_figures_no_history(self, tmp_path):
        trainer = ProteinBERTTrainer()
        figs = trainer._generate_figures(tmp_path)
        assert figs == {}

    def test_generate_figures_with_history(self, tmp_path):
        trainer = ProteinBERTTrainer()
        trainer._history.add_epoch(1, 0.5, 0.6, 0.7, 0.8, 0.001, 10.0)
        trainer._history.add_epoch(2, 0.3, 0.4, 0.85, 0.9, 0.0005, 9.0)
        figs = trainer._generate_figures(tmp_path)
        assert "loss" in figs
        assert "accuracy" in figs
        assert (tmp_path / "loss.png").exists()
        assert (tmp_path / "loss.pdf").exists()
        assert (tmp_path / "accuracy.png").exists()


class TestTrainerSystemUsage:
    """Runtime monitoring."""

    def test_get_system_usage_returns_floats(self):
        cpu, ram = ProteinBERTTrainer._get_system_usage()
        assert isinstance(cpu, float)
        assert isinstance(ram, float)
        assert cpu >= 0
        assert ram >= 0


# ======================================================================
# Wrapper Integration Tests
# ======================================================================


class TestWrapperCheckpointIntegration:
    """ProteinBERTModel integration with CheckpointManager."""

    def test_initialize_checkpoint_manager(self, tmp_path):
        model = ProteinBERTModel()
        model.initialize_runtime(workspace_root=tmp_path)
        cm = model.initialize_checkpoint_manager(workspace_root=tmp_path)
        assert cm is not None
        assert model.checkpoint_manager is cm
        assert cm.checkpoint_root.exists()

    def test_checkpoint_manager_property_before_init(self):
        model = ProteinBERTModel()
        assert model.checkpoint_manager is None

    def test_checkpoint_manager_under_checkpoints(self, tmp_path):
        model = ProteinBERTModel()
        model.initialize_runtime(workspace_root=tmp_path)
        cm = model.initialize_checkpoint_manager(workspace_root=tmp_path)
        assert "checkpoints" in str(cm.checkpoint_root)


# ======================================================================
# Module-Level Import Safety
# ======================================================================


def test_checkpoints_exported_from_package():
    from src.models.proteinbert import CheckpointManager
    assert CheckpointManager is not None


def test_trainer_exported_from_package():
    from src.models.proteinbert import ProteinBERTTrainer
    assert ProteinBERTTrainer is not None


def test_no_tf_at_module_level():
    import sys
    assert "tensorflow" not in sys.modules


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _require_tf():
    """Skip test if TensorFlow is not available."""
    import importlib
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("TensorFlow not installed")
    import tensorflow as tf
    return tf
