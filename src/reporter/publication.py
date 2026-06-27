"""Publication — paper-ready figure and table generation for publications."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..visualization.plots import (
    plot_metric_comparison,
    plot_learning_curves,
    save_figure,
)
from ..utils.logging import get_logger


logger = get_logger(__name__)


class Publication:
    """Generates publication-ready figures and tables from benchmark results."""

    def __init__(self, output_dir: Union[str, Path] = "outputs/reports/publication") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_figures(
        self,
        results_df: "pd.DataFrame",
        metric: str = "spearman",
    ) -> List[Path]:
        """Generate all standard publication figures.

        Args:
            results_df: DataFrame with benchmark results.
            metric: Metric column to plot.

        Returns:
            List of saved figure paths.
        """
        import pandas as pd

        figures = []

        fig = plot_metric_comparison(results_df, metric=metric)
        paths = save_figure(fig, self.output_dir / f"metric_comparison_{metric}")
        figures.extend(paths)

        return figures

    def generate_tables(
        self,
        results: List[Dict[str, Any]],
        fmt: str = "latex",
        caption: str = "Benchmark Results",
        label: str = "tab:benchmark",
    ) -> str:
        """Generate a publication table.

        Args:
            results: List of result dictionaries.
            fmt: 'latex' or 'md'.
            caption: Table caption.
            label: LaTeX label.

        Returns:
            Table string.
        """
        if fmt == "latex":
            from .latex import to_latex_table
            return to_latex_table(results, caption=caption, label=label)
        elif fmt == "md":
            from .markdown import to_markdown_table
            return to_markdown_table(results)
        else:
            raise ValueError(f"Unsupported format: {fmt}")