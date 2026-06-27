"""ProteinBERTTrainer — TensorFlow training engine for ProteinBERT.

Responsibilities:
  - deterministic initialization
  - configurable optimizer, scheduler, loss, batch size
  - gradient updates and epoch management
  - early stopping (configurable patience)
  - checkpoint saving during training
  - training history collection
  - training evidence generation (report, figures, manifests)
  - resume from checkpoint

No evaluation metrics beyond training/validation monitoring.
All TF imports are lazy.
"""

import csv
import json
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ...utils.logging import get_logger
from ...utils.io import ensure_dir, save_json, write_text
from .checkpoints import CheckpointManager


logger = get_logger(__name__)


# ------------------------------------------------------------------
# Default training configuration
# ------------------------------------------------------------------

DEFAULT_TRAINING_CONFIG: Dict[str, Any] = {
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "epochs": 20,
    "batch_size": 8,
    "early_stopping": True,
    "patience": 5,
    "resume": True,
    "save_best_only": True,
    "validation_split": 0.2,
}


# ------------------------------------------------------------------
# TrainingHistory
# ------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """Training history per epoch."""
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    gpu_memory: List[float] = field(default_factory=list)
    cpu_percent: List[float] = field(default_factory=list)
    ram_used_gb: List[float] = field(default_factory=list)

    def add_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_accuracy: float,
        val_accuracy: float,
        learning_rate: float,
        epoch_time: float,
        gpu_mem: float = 0.0,
        cpu_pct: float = 0.0,
        ram_gb: float = 0.0,
    ) -> None:
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_accuracy.append(train_accuracy)
        self.val_accuracy.append(val_accuracy)
        self.learning_rates.append(learning_rate)
        self.epoch_times.append(epoch_time)
        self.gpu_memory.append(gpu_mem)
        self.cpu_percent.append(cpu_pct)
        self.ram_used_gb.append(ram_gb)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_csv_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for i in range(len(self.epochs)):
            rows.append({
                "epoch": self.epochs[i] if i < len(self.epochs) else None,
                "train_loss": self.train_loss[i] if i < len(self.train_loss) else None,
                "val_loss": self.val_loss[i] if i < len(self.val_loss) else None,
                "train_accuracy": self.train_accuracy[i] if i < len(self.train_accuracy) else None,
                "val_accuracy": self.val_accuracy[i] if i < len(self.val_accuracy) else None,
                "learning_rate": self.learning_rates[i] if i < len(self.learning_rates) else None,
                "epoch_time": self.epoch_times[i] if i < len(self.epoch_times) else None,
                "gpu_memory_mb": self.gpu_memory[i] if i < len(self.gpu_memory) else None,
                "cpu_percent": self.cpu_percent[i] if i < len(self.cpu_percent) else None,
                "ram_used_gb": self.ram_used_gb[i] if i < len(self.ram_used_gb) else None,
            })
        return rows


# ------------------------------------------------------------------
# ProteinBERTTrainer
# ------------------------------------------------------------------


class ProteinBERTTrainer:
    """TensorFlow training engine for ProteinBERT.

    Usage:
        trainer = ProteinBERTTrainer(config={...}, checkpoint_manager=cm)
        history = trainer.train(model, train_data, val_data, output_dir=...)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        self._config: Dict[str, Any] = dict(DEFAULT_TRAINING_CONFIG)
        if config:
            self._config.update(config)
        self._checkpoint_manager = checkpoint_manager
        self._history = TrainingHistory()
        self._start_time: Optional[float] = None
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._stopped_early = False
        self._trained_epochs = 0
        logger.debug(f"ProteinBERTTrainer initialized (epochs={self._config.get('epochs')})")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._config)

    @property
    def history(self) -> TrainingHistory:
        return self._history

    @property
    def stopped_early(self) -> bool:
        return self._stopped_early

    @property
    def trained_epochs(self) -> int:
        return self._trained_epochs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        model: Any,
        train_data: Any,
        val_data: Optional[Any] = None,
        output_dir: Optional[Union[str, Path]] = None,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the training loop.

        Args:
            model: Loaded TF model (must have train_on_batch or fit).
            train_data: Training data (ProteinBERTDataAdapter or tuple of arrays).
            val_data: Validation data (ProteinBERTDataAdapter or tuple of arrays).
            output_dir: Output directory for training artifacts.
            experiment_id: Experiment ID for structured output path.

        Returns:
            Dict with training results including history, best metrics, etc.
        """
        import tensorflow as tf

        self._start_time = time.time()
        epochs = self._config.get("epochs", 20)
        start_epoch = 0

        # Resolve output directory
        if output_dir is None:
            if experiment_id:
                output_dir = Path.cwd() / "outputs" / "experiments" / experiment_id / "training"
            else:
                output_dir = Path.cwd() / "outputs" / "training"
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        # Build TF datasets from adapter
        train_ds, val_ds, steps_per_epoch, val_steps = self._build_tf_datasets(
            train_data, val_data, model
        )

        # Compile model
        optimizer = self._build_optimizer()
        loss_fn = self._build_loss()
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

        # Resume from checkpoint
        if self._config.get("resume", True) and self._checkpoint_manager is not None:
            resume_epoch = self._checkpoint_manager.restore_model(model)
            if resume_epoch > 0:
                start_epoch = resume_epoch
                logger.info(f"Resuming training from epoch {resume_epoch}")

        # Determine best checkpoint path
        best_ckpt_path = self._get_best_checkpoint_path(output_dir)

        # Training loop
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # Training
            train_loss = 0.0
            train_acc = 0.0
            train_batches = 0

            for batch_idx, (x_batch, y_batch) in enumerate(train_ds):
                with tf.GradientTape() as tape:
                    logits = model(x_batch, training=True)
                    loss_value = loss_fn(y_batch, logits)

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                train_loss += float(loss_value)
                train_acc_metric.update_state(y_batch, logits)
                train_batches += 1

                if batch_idx + 1 >= steps_per_epoch:
                    break

            train_loss /= max(train_batches, 1)
            train_acc = float(train_acc_metric.result())
            train_acc_metric.reset_state()

            # Validation
            val_loss = 0.0
            val_acc = 0.0
            val_batches = 0

            for batch_idx, (x_batch, y_batch) in enumerate(val_ds):
                logits = model(x_batch, training=False)
                loss_value = loss_fn(y_batch, logits)
                val_loss += float(loss_value)
                val_acc_metric.update_state(y_batch, logits)
                val_batches += 1

                if batch_idx + 1 >= val_steps:
                    break

            val_loss /= max(val_batches, 1)
            val_acc = float(val_acc_metric.result())
            val_acc_metric.reset_state()

            epoch_time = time.time() - epoch_start

            # Collect runtime metrics
            gpu_mem = self._get_gpu_memory_usage()
            cpu_pct, ram_gb = self._get_system_usage()

            # Record history
            current_lr = float(tf.keras.backend.get_value(optimizer.lr))
            self._history.add_epoch(
                epoch=epoch + 1,
                train_loss=round(train_loss, 6),
                val_loss=round(val_loss, 6),
                train_accuracy=round(train_acc, 6),
                val_accuracy=round(val_acc, 6),
                learning_rate=current_lr,
                epoch_time=round(epoch_time, 2),
                gpu_mem=gpu_mem,
                cpu_pct=cpu_pct,
                ram_gb=ram_gb,
            )

            logger.info(
                f"  train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"({epoch_time:.1f}s)"
            )

            # Checkpoint saving
            is_best = val_loss < self._best_val_loss
            if is_best:
                self._best_val_loss = val_loss
            if self._checkpoint_manager is not None:
                self._checkpoint_manager.save(
                    model=model,
                    epoch=epoch + 1,
                    loss=train_loss,
                    val_loss=val_loss,
                    accuracy=train_acc,
                    val_accuracy=val_acc,
                    optimizer=optimizer,
                    is_best=is_best,
                )

            # Save best weights locally
            if is_best:
                model.save_weights(str(best_ckpt_path))
                logger.debug(f"Best checkpoint updated (val_loss={val_loss:.4f})")

            # Early stopping
            if self._config.get("early_stopping", True):
                if val_loss < self._best_val_loss:
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1
                    patience = self._config.get("patience", 5)
                    if self._patience_counter >= patience:
                        logger.info(
                            f"Early stopping triggered (patience={patience})"
                        )
                        self._stopped_early = True
                        break

            self._trained_epochs += 1

        # Save training evidence
        self._save_evidence(output_dir, model, optimizer, experiment_id)

        total_time = time.time() - self._start_time

        result = {
            "trained_epochs": self._trained_epochs,
            "total_time_seconds": round(total_time, 2),
            "best_val_loss": round(self._best_val_loss, 6),
            "stopped_early": self._stopped_early,
            "output_dir": str(output_dir),
            "history": self._history.to_dict(),
        }
        logger.info(f"Training complete: {self._trained_epochs} epochs in {total_time:.1f}s")
        return result

    # ------------------------------------------------------------------
    # Internal build methods
    # ------------------------------------------------------------------

    def _build_tf_datasets(
        self,
        train_data: Any,
        val_data: Optional[Any],
        model: Any,
    ) -> Tuple[Any, Any, int, int]:
        """Convert adapter data to tf.data.Dataset.

        Returns (train_ds, val_ds, steps_per_epoch, val_steps).
        """
        import tensorflow as tf

        batch_size = self._config.get("batch_size", 8)

        # Extract numpy arrays from adapter or use directly
        if hasattr(train_data, "__iter__") and not isinstance(train_data, np.ndarray):
            train_inputs, train_labels = self._extract_from_adapter(train_data)
        else:
            train_inputs, train_labels = train_data

        if val_data is not None:
            if hasattr(val_data, "__iter__") and not isinstance(val_data, np.ndarray):
                val_inputs, val_labels = self._extract_from_adapter(val_data)
            else:
                val_inputs, val_labels = val_data
        else:
            split = self._config.get("validation_split", 0.2)
            split_idx = int(len(train_inputs) * (1 - split))
            val_inputs, val_labels = train_inputs[split_idx:], train_labels[split_idx:]
            train_inputs, train_labels = train_inputs[:split_idx], train_labels[:split_idx]

        # Build TF datasets
        train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
        train_ds = train_ds.shuffle(1024, seed=self._config.get("seed", 42))
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels))
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        steps_per_epoch = max(len(train_inputs) // batch_size, 1)
        val_steps = max(len(val_inputs) // batch_size, 1)

        return train_ds, val_ds, steps_per_epoch, val_steps

    def _extract_from_adapter(self, adapter: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Extract input_ids and labels from a ProteinBERTDataAdapter."""
        all_inputs = []
        all_labels = []
        for batch in adapter:
            all_inputs.append(batch["input_ids"])
            all_labels.append(batch["labels"])
        return np.concatenate(all_inputs), np.concatenate(all_labels)

    def _build_optimizer(self) -> Any:
        """Build TF optimizer from config."""
        import tensorflow as tf

        name = self._config.get("optimizer", "adam").lower()
        lr = self._config.get("learning_rate", 1e-4)

        if name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=lr)
        elif name == "adamw":
            return tf.keras.optimizers.AdamW(learning_rate=lr)
        elif name == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        else:
            logger.warning(f"Unknown optimizer '{name}', using Adam")
            return tf.keras.optimizers.Adam(learning_rate=lr)

    def _build_loss(self) -> Any:
        """Build loss function."""
        import tensorflow as tf
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # ------------------------------------------------------------------
    # Checkpoint path
    # ------------------------------------------------------------------

    def _get_best_checkpoint_path(self, output_dir: Path) -> Path:
        best_dir = output_dir / "checkpoints"
        best_dir.mkdir(parents=True, exist_ok=True)
        return best_dir / "best_weights.h5"

    # ------------------------------------------------------------------
    # Runtime monitoring
    # ------------------------------------------------------------------

    @staticmethod
    def _get_gpu_memory_usage() -> float:
        """Get GPU memory usage in MB. Returns 0.0 if no GPU."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return 0.0
            for gpu in gpus:
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    return float(details.get("device_name", "0"))
                except Exception:
                    pass
            return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _get_system_usage() -> Tuple[float, float]:
        """Get CPU usage percent and RAM used in GB."""
        cpu_pct = 0.0
        ram_gb = 0.0
        try:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=0.1)
            ram_gb = round(psutil.virtual_memory().used / (1024 ** 3), 2)
        except Exception:
            pass
        return cpu_pct, ram_gb

    # ------------------------------------------------------------------
    # Training evidence
    # ------------------------------------------------------------------

    def _save_evidence(
        self,
        output_dir: Path,
        model: Any,
        optimizer: Any,
        experiment_id: Optional[str] = None,
    ) -> None:
        """Generate all training evidence artifacts."""
        logger.info(f"Saving training evidence to {output_dir}")

        # 1. Training report (Markdown)
        self._save_training_report(output_dir)

        # 2. History CSV
        self._save_history_csv(output_dir)

        # 3. History JSON
        self._save_history_json(output_dir)

        # 4. Epoch times CSV
        self._save_epoch_times_csv(output_dir)

        # 5. Learning rate CSV
        self._save_lr_csv(output_dir)

        # 6. Checkpoint manifest
        if self._checkpoint_manager is not None:
            self._save_checkpoint_manifest(output_dir)

        # 7. Best checkpoint info
        self._save_best_checkpoint_json(output_dir)

        # 8. Early stopping info
        if self._stopped_early:
            self._save_early_stopping_json(output_dir)

        # 9. Figures
        self._generate_figures(output_dir)

        logger.info(f"Training evidence saved ({len(list(output_dir.iterdir()))} files)")

    def _save_training_report(self, output_dir: Path) -> Path:
        report_path = output_dir / "training_report.md"
        lines = []
        lines.append("# Training Report")
        lines.append("")
        lines.append(f"- **Date**: {datetime.now().isoformat()}")
        lines.append(f"- **Platform**: {platform.platform()}")
        lines.append(f"- **Python**: {sys.version.split()[0]}")
        lines.append("")
        lines.append("## Configuration")
        lines.append("")
        for k, v in sorted(self._config.items()):
            lines.append(f"- **{k}**: {v}")
        lines.append("")
        lines.append("## Results")
        lines.append("")
        lines.append(f"- **Trained epochs**: {self._trained_epochs}")
        lines.append(f"- **Best validation loss**: {self._best_val_loss:.6f}")
        lines.append(f"- **Stopped early**: {self._stopped_early}")
        lines.append(f"- **Total time**: {self._history.epoch_times[-1] if self._history.epoch_times else 0:.1f}s")
        lines.append("")
        if self._history.epochs:
            lines.append("## Per-Epoch History")
            lines.append("")
            lines.append("| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | LR | Time (s) |")
            lines.append("|-------|-----------|----------|-----------|---------|----|----------|")
            for i, ep in enumerate(self._history.epochs):
                lines.append(
                    f"| {ep} | {self._history.train_loss[i]:.4f} | "
                    f"{self._history.val_loss[i]:.4f} | "
                    f"{self._history.train_accuracy[i]:.4f} | "
                    f"{self._history.val_accuracy[i]:.4f} | "
                    f"{self._history.learning_rates[i]:.6f} | "
                    f"{self._history.epoch_times[i]:.1f} |"
                )
            lines.append("")
        lines.append("---")
        lines.append("*Report generated by ProteinBERT training engine*")
        content = "\n".join(lines)
        report_path.write_text(content, encoding="utf-8")
        logger.debug(f"Training report saved to {report_path}")
        return report_path

    def _save_history_csv(self, output_dir: Path) -> Path:
        path = output_dir / "history.csv"
        rows = self._history.to_csv_rows()
        if rows:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        logger.debug(f"History CSV saved to {path}")
        return path

    def _save_history_json(self, output_dir: Path) -> Path:
        path = output_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self._history.to_dict(), f, indent=2)
        logger.debug(f"History JSON saved to {path}")
        return path

    def _save_epoch_times_csv(self, output_dir: Path) -> Path:
        path = output_dir / "epoch_times.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "time_seconds"])
            for i, ep in enumerate(self._history.epochs):
                writer.writerow([ep, self._history.epoch_times[i]])
        logger.debug(f"Epoch times saved to {path}")
        return path

    def _save_lr_csv(self, output_dir: Path) -> Path:
        path = output_dir / "learning_rate.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "learning_rate"])
            for i, ep in enumerate(self._history.epochs):
                writer.writerow([ep, self._history.learning_rates[i]])
        logger.debug(f"Learning rate CSV saved to {path}")
        return path

    def _save_checkpoint_manifest(self, output_dir: Path) -> Path:
        if self._checkpoint_manager is None:
            return output_dir / "checkpoint_manifest.json"
        return self._checkpoint_manager._save_manifest()

    def _save_best_checkpoint_json(self, output_dir: Path) -> Path:
        path = output_dir / "best_checkpoint.json"
        data = {
            "best_val_loss": self._best_val_loss,
            "epoch": self._history.epochs[-1] if self._history.epochs else 0,
            "stopped_early": self._stopped_early,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def _save_early_stopping_json(self, output_dir: Path) -> Path:
        path = output_dir / "early_stopping.json"
        data = {
            "triggered": True,
            "patience": self._config.get("patience", 5),
            "epochs_trained": self._trained_epochs,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def _generate_figures(self, output_dir: Path) -> Dict[str, Path]:
        """Generate training figures as PNG (300dpi) and PDF."""
        if not self._history.epochs:
            logger.warning("No training history to plot")
            return {}

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available, skipping figures")
            return {}

        figs = {}
        epochs = self._history.epochs

        # 1. Loss curves
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, self._history.train_loss, "b-", label="Train Loss")
        ax.plot(epochs, self._history.val_loss, "r-", label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        figs["loss"] = self._save_figure(fig, output_dir, "loss")
        plt.close(fig)

        # 2. Accuracy curves
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, self._history.train_accuracy, "b-", label="Train Accuracy")
        ax.plot(epochs, self._history.val_accuracy, "r-", label="Val Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training and Validation Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        figs["accuracy"] = self._save_figure(fig, output_dir, "accuracy")
        plt.close(fig)

        # 3. Learning rate schedule
        if len(set(self._history.learning_rates)) > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, self._history.learning_rates, "g-", label="Learning Rate")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.legend()
            ax.grid(True, alpha=0.3)
            figs["learning_rate"] = self._save_figure(fig, output_dir, "learning_rate")
            plt.close(fig)

        logger.debug(f"Generated {len(figs)} training figures")
        return figs

    def _save_figure(
        self, fig: Any, output_dir: Path, name: str
    ) -> Path:
        png_path = output_dir / f"{name}.png"
        pdf_path = output_dir / f"{name}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        return png_path

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        return {
            "config": dict(self._config),
            "trained_epochs": self._trained_epochs,
            "stopped_early": self._stopped_early,
            "best_val_loss": self._best_val_loss,
            "has_checkpoint_manager": self._checkpoint_manager is not None,
        }

    def __repr__(self) -> str:
        return (
            f"ProteinBERTTrainer(epochs={self._config.get('epochs')}, "
            f"trained={self._trained_epochs})"
        )
