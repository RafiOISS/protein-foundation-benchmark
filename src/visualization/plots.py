"""Publication-quality plotting functions for benchmark results."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_style(
    style: str = "seaborn-v0_8-whitegrid",
    context: str = "paper",
    font_size: int = 12,
    palette: str = "colorblind",
) -> None:
    plt.style.use(style)
    sns.set_context(context)
    sns.set_palette(palette)
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_figure(fig: plt.Figure, path: Union[str, Path], formats: Optional[List[str]] = None, dpi: int = 300) -> List[Path]:
    if formats is None:
        formats = ["png", "pdf"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        p = path.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi, format=fmt)
        saved.append(p)
    plt.close(fig)
    return saved


def plot_metric_comparison(
    results: pd.DataFrame,
    metric: str,
    x: str = "model_name",
    hue: str = "dataset_name",
    kind: str = "bar",
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if kind == "bar":
        sns.barplot(data=results, x=x, y=metric, hue=hue, ax=ax)
    elif kind == "box":
        sns.boxplot(data=results, x=x, y=metric, hue=hue, ax=ax)
    elif kind == "violin":
        sns.violinplot(data=results, x=x, y=metric, hue=hue, ax=ax)
    ax.set_title(title or f"{metric.replace('_', ' ').title()} by {x.replace('_', ' ').title()}")
    ax.set_ylabel(metric.replace("_", " ").title())
    plt.tight_layout()
    return fig


def plot_learning_curves(history: Dict[str, List[float]], figsize: Tuple[int, int] = (10, 4)) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    axes[0].plot(epochs, history.get("train_loss", []), label="Train", marker="o")
    axes[0].plot(epochs, history.get("val_loss", []), label="Val", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history.get("val_loss", []), label="Val Loss", marker="s", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None, figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_embedding_tsne(embeddings: np.ndarray, labels: Optional[np.ndarray] = None, perplexity: int = 30, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    from sklearn.manifold import TSNE
    emb = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=figsize)
    if labels is not None:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, alpha=0.6, cmap="tab10")
        plt.colorbar(sc, ax=ax)
    else:
        ax.scatter(emb[:, 0], emb[:, 1], alpha=0.6)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, ax=ax)
    plt.tight_layout()
    return fig


def plot_model_size_vs_performance(df: pd.DataFrame, size_col: str = "params", metric_col: str = "spearman", hue_col: str = "model_type", figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df, x=size_col, y=metric_col, hue=hue_col, s=100, alpha=0.7, ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel(size_col.replace("_", " ").title())
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig