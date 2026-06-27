"""Reporter — orchestrates benchmark result export across multiple formats."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.logging import get_logger


logger = get_logger(__name__)


class Reporter:
    """Exports benchmark results to CSV, LaTeX, Markdown, and publication tables.

    Usage:
        reporter = Reporter()
        reporter.export(results, fmt="csv")
        reporter.export(results, fmt="latex")
        reporter.export(results, fmt="md")
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        results: List[Dict[str, Any]],
        fmt: str = "csv",
        name: str = "benchmark_results",
        **kwargs,
    ) -> Path:
        """Export results in the specified format.

        Args:
            results: List of result dictionaries.
            fmt: 'csv', 'latex', 'md', 'excel', 'json'.
            name: Base output filename (without extension).
            **kwargs: Additional format-specific options.

        Returns:
            Path to the exported file.
        """
        fmt_map = {
            "csv": self._to_csv,
            "latex": self._to_latex,
            "md": self._to_markdown,
            "excel": self._to_excel,
            "json": self._to_json,
        }

        exporter = fmt_map.get(fmt)
        if exporter is None:
            raise ValueError(f"Unsupported format: {fmt}. Choose from {list(fmt_map.keys())}")

        path = exporter(results, name, **kwargs)
        logger.info(f"Exported {len(results)} results to {path}")
        return path

    def summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary dictionary from results."""
        if not results:
            return {}
        summary = {
            "num_runs": len(results),
            "models": list({r.get("model_name") for r in results}),
            "datasets": list({r.get("dataset_name") for r in results}),
        }
        return summary

    def _to_csv(self, results: List[Dict[str, Any]], name: str, **kwargs) -> Path:
        path = self.output_dir / f"{name}.csv"
        import pandas as pd
        pd.DataFrame(results).to_csv(path, index=False)
        return path

    def _to_latex(self, results: List[Dict[str, Any]], name: str, **kwargs) -> Path:
        path = self.output_dir / f"{name}.tex"
        from .latex import to_latex_table
        table = to_latex_table(results, **kwargs)
        path.write_text(table)
        return path

    def _to_markdown(self, results: List[Dict[str, Any]], name: str, **kwargs) -> Path:
        path = self.output_dir / f"{name}.md"
        from .markdown import to_markdown_table
        table = to_markdown_table(results, **kwargs)
        path.write_text(table)
        return path

    def _to_excel(self, results: List[Dict[str, Any]], name: str, **kwargs) -> Path:
        path = self.output_dir / f"{name}.xlsx"
        import pandas as pd
        pd.DataFrame(results).to_excel(path, index=False)
        return path

    def _to_json(self, results: List[Dict[str, Any]], name: str, **kwargs) -> Path:
        path = self.output_dir / f"{name}.json"
        from ..utils.io import save_json
        save_json(results, path)
        return path