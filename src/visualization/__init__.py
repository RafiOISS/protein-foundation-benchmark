"""
Visualization package for the Protein Foundation Model Benchmark Framework.

Contains publication-quality plotting functions.
"""

from .plots import (
    plot_metric_comparison,
    plot_learning_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_embedding_tsne,
    plot_correlation_heatmap,
    plot_model_size_vs_performance,
    save_figure,
    setup_style,
)

__all__ = [
    "plot_metric_comparison",
    "plot_learning_curves",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_embedding_tsne",
    "plot_correlation_heatmap",
    "plot_model_size_vs_performance",
    "save_figure",
    "setup_style",
]