"""Metrics — computation functions for all benchmark metrics."""

from typing import Dict, List, Optional

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..utils.logging import get_logger


logger = get_logger(__name__)


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metrics: List[str],
    task_type: str = "regression",
    threshold: float = 0.5,
) -> Dict[str, float]:
    if task_type == "regression":
        return _regression_metrics(predictions, targets, metrics)
    elif task_type in ("binary_classification",):
        return _binary_classification_metrics(predictions, targets, metrics, threshold)
    elif task_type == "multiclass_classification":
        return _multiclass_metrics(predictions, targets, metrics)
    elif task_type == "multilabel_classification":
        return _multilabel_metrics(predictions, targets, metrics, threshold)
    elif task_type == "token_classification":
        return _token_classification_metrics(predictions, targets, metrics)
    return {}


def _regression_metrics(preds: torch.Tensor, targets: torch.Tensor, metrics: List[str]) -> Dict[str, float]:
    p = preds.squeeze().cpu().numpy()
    t = targets.squeeze().cpu().numpy()
    mask = ~(np.isnan(p) | np.isnan(t))
    p, t = p[mask], t[mask]

    results = {}
    if "mse" in metrics:
        results["mse"] = float(mean_squared_error(t, p))
    if "mae" in metrics:
        results["mae"] = float(mean_absolute_error(t, p))
    if "rmse" in metrics:
        results["rmse"] = float(np.sqrt(mean_squared_error(t, p)))
    if "spearman" in metrics and len(p) > 1:
        corr, _ = scipy_stats.spearmanr(p, t)
        results["spearman"] = float(corr) if not np.isnan(corr) else 0.0
    if "pearson" in metrics and len(p) > 1:
        corr, _ = scipy_stats.pearsonr(p, t)
        results["pearson"] = float(corr) if not np.isnan(corr) else 0.0
    if "r2" in metrics:
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        results["r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return results


def _binary_classification_metrics(preds: torch.Tensor, targets: torch.Tensor, metrics: List[str], threshold: float) -> Dict[str, float]:
    p = preds.cpu().numpy()
    t = targets.cpu().numpy().astype(int)
    probs = 1 / (1 + np.exp(-p.squeeze()))
    pred_labels = (probs > threshold).astype(int)

    results = {}
    if "accuracy" in metrics:
        results["accuracy"] = float(accuracy_score(t, pred_labels))
    if "f1" in metrics:
        results["f1"] = float(f1_score(t, pred_labels, zero_division=0))
    if "precision" in metrics:
        results["precision"] = float(precision_score(t, pred_labels, zero_division=0))
    if "recall" in metrics:
        results["recall"] = float(recall_score(t, pred_labels, zero_division=0))
    if "auroc" in metrics:
        try:
            results["auroc"] = float(roc_auc_score(t, probs))
        except ValueError:
            results["auroc"] = float("nan")
    if "auprc" in metrics:
        try:
            results["auprc"] = float(average_precision_score(t, probs))
        except ValueError:
            results["auprc"] = float("nan")
    if "mcc" in metrics:
        results["mcc"] = float(matthews_corrcoef(t, pred_labels))
    return results


def _multiclass_metrics(preds: torch.Tensor, targets: torch.Tensor, metrics: List[str]) -> Dict[str, float]:
    p = preds.cpu().numpy()
    t = targets.cpu().numpy().astype(int)
    from scipy.special import softmax
    pred_labels = np.argmax(softmax(p, axis=1), axis=1)

    results = {}
    if "accuracy" in metrics:
        results["accuracy"] = float(accuracy_score(t, pred_labels))
    if "f1_macro" in metrics:
        results["f1_macro"] = float(f1_score(t, pred_labels, average="macro", zero_division=0))
    if "f1_micro" in metrics:
        results["f1_micro"] = float(f1_score(t, pred_labels, average="micro", zero_division=0))
    if "f1_weighted" in metrics:
        results["f1_weighted"] = float(f1_score(t, pred_labels, average="weighted", zero_division=0))
    return results


def _multilabel_metrics(preds: torch.Tensor, targets: torch.Tensor, metrics: List[str], threshold: float) -> Dict[str, float]:
    p = preds.cpu().numpy()
    t = targets.cpu().numpy().astype(int)
    probs = 1 / (1 + np.exp(-p))
    pred_labels = (probs > threshold).astype(int)

    results = {}
    if "f1_macro" in metrics:
        results["f1_macro"] = float(f1_score(t, pred_labels, average="macro", zero_division=0))
    if "f1_micro" in metrics:
        results["f1_micro"] = float(f1_score(t, pred_labels, average="micro", zero_division=0))
    if "hamming_loss" in metrics:
        results["hamming_loss"] = float(hamming_loss(t, pred_labels))
    if "auroc_macro" in metrics:
        try:
            results["auroc_macro"] = float(roc_auc_score(t, probs, average="macro"))
        except ValueError:
            results["auroc_macro"] = float("nan")
    return results


def _token_classification_metrics(preds: torch.Tensor, targets: torch.Tensor, metrics: List[str], ignore_index: int = -100) -> Dict[str, float]:
    p = preds.cpu().numpy()
    t = targets.cpu().numpy()
    pred_labels = np.argmax(p, axis=-1).flatten()
    t_flat = t.flatten()
    mask = t_flat != ignore_index
    pred_labels = pred_labels[mask]
    t_flat = t_flat[mask]

    results = {}
    if "accuracy" in metrics:
        results["accuracy"] = float(accuracy_score(t_flat, pred_labels))
    if "f1_macro" in metrics:
        results["f1_macro"] = float(f1_score(t_flat, pred_labels, average="macro", zero_division=0))
    if "f1_micro" in metrics:
        results["f1_micro"] = float(f1_score(t_flat, pred_labels, average="micro", zero_division=0))
    return results


def get_default_metrics(task_type: str) -> List[str]:
    defaults = {
        "regression": ["mse", "mae", "spearman", "pearson", "rmse", "r2"],
        "binary_classification": ["accuracy", "f1", "precision", "recall", "auroc", "auprc", "mcc"],
        "multiclass_classification": ["accuracy", "f1_macro", "f1_micro", "f1_weighted"],
        "multilabel_classification": ["f1_macro", "f1_micro", "hamming_loss", "auroc_macro"],
        "token_classification": ["accuracy", "f1_macro", "f1_micro"],
    }
    return defaults.get(task_type, ["accuracy"])