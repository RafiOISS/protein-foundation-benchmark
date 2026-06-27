"""Reporter — orchestrates benchmark result export across multiple formats.

Also supports dataset-level reporting (dataset_report.md).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.io import ensure_dir
from ..utils.logging import get_logger


logger = get_logger(__name__)


class Reporter:
    """Exports benchmark results to CSV, LaTeX, Markdown, and publication tables.

    Usage:
        reporter = Reporter()
        reporter.export(results, fmt="csv")
        reporter.export(results, fmt="latex")
        reporter.dataset_report(dataset)
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

    def dataset_report(
        self,
        dataset: "BaseDataset",  # noqa: F821
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate a dataset report in Markdown.

        Includes: dataset summary, statistics table, generated figures,
        validation results, and manifest info.

        Args:
            dataset: BaseDataset instance (after download, preprocess, statistics, visualize).
            output_path: Output path for the report. Default: dataset.figures_dir / "dataset_report.md".

        Returns:
            Path to saved report.
        """
        from ..interfaces.base_dataset import BaseDataset

        if not isinstance(dataset, BaseDataset):
            raise TypeError(f"Expected BaseDataset, got {type(dataset)}")

        # Compute stats if not done
        stats = dataset.statistics()
        figures = dataset.visualize()
        manifest = dataset.generate_manifest()

        from ..statistics.statistics import print_statistics
        print_statistics(stats)

        output_path = Path(output_path) if output_path else dataset.figures_dir / "dataset_report.md"
        ensure_dir(output_path.parent)

        lines = []
        lines.append(f"# Dataset Report: {dataset.DATASET_NAME}")
        lines.append("")
        lines.append(f"- **Version**: {dataset.DATASET_VERSION}")
        lines.append(f"- **Framework**: {manifest.get('framework_version', '?')}")
        lines.append(f"- **Git Commit**: {manifest.get('git_commit', '?')}")
        lines.append(f"- **Generated**: {manifest.get('generated_at', '?')}")
        lines.append("")

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Statistic | Value |")
        lines.append("|-----------|-------|")

        summary_keys = [
            ("num_sequences", "Number of Sequences"),
            ("unique_sequences", "Unique Sequences"),
            ("duplicate_sequences", "Duplicate Sequences"),
            ("missing_labels", "Missing Labels"),
            ("min_length", "Min Length"),
            ("max_length", "Max Length"),
            ("mean_length", "Mean Length"),
            ("median_length", "Median Length"),
            ("std_length", "Std Length"),
        ]
        for key, label in summary_keys:
            val = stats.get(key, "?")
            if isinstance(val, float):
                val = f"{val:.2f}"
            lines.append(f"| {label} | {val} |")

        # Split sizes
        if "split_sizes" in stats:
            lines.append("")
            lines.append("## Split Sizes")
            lines.append("")
            lines.append("| Split | Count | Percentage |")
            lines.append("|-------|-------|------------|")
            for k, v in stats["split_sizes"].items():
                if k.endswith("_pct"):
                    continue
                pct = stats["split_sizes"].get(f"{k}_pct", "")
                pct_str = f"{pct}%" if pct != "" else "-"
                lines.append(f"| {k} | {v} | {pct_str} |")

        # Label distribution
        if "label_distribution" in stats:
            lines.append("")
            lines.append("## Label Distribution")
            lines.append("")
            lines.append("| Label | Count |")
            lines.append("|-------|-------|")
            for k, v in sorted(stats["label_distribution"].items(), key=lambda x: -x[1]):
                lines.append(f"| {k} | {v} |")

        # Figures
        if figures:
            lines.append("")
            lines.append("## Figures")
            lines.append("")

            # Relative paths from report location
            for fig_name, fig_path in figures.items():
                try:
                    rel = fig_path.relative_to(output_path.parent)
                    rel_str = str(rel.as_posix())
                except ValueError:
                    rel_str = str(fig_path)
                lines.append(f"![{fig_name}]({rel_str})")
                lines.append("")

        # Verification
        try:
            verify_result = dataset.verify()
            lines.append("")
            lines.append("## Integrity Verification")
            lines.append("")
            lines.append(f"- **Valid**: {verify_result.get('valid', '?')}")
            if verify_result.get("errors"):
                lines.append("- **Errors**:")
                for err in verify_result["errors"]:
                    lines.append(f"  - {err}")
            if verify_result.get("warnings"):
                lines.append("- **Warnings**:")
                for warn in verify_result["warnings"]:
                    lines.append(f"  - {warn}")
        except Exception:
            lines.append("\n## Integrity Verification\n- Verification: skipped\n")

        # Manifest info
        lines.append("")
        lines.append("## Manifest")
        lines.append("")
        lines.append("| Key | Value |")
        lines.append("|-----|-------|")
        for key in ("dataset_version", "preprocessing_version", "framework_version", "git_commit", "generated_at"):
            val = manifest.get(key, "?")
            lines.append(f"| {key} | {val} |")

        lines.append("")
        lines.append("---")
        lines.append(f"*Report generated by protein-foundation-benchmark v{manifest.get('framework_version', '?')}*")

        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")
        logger.info(f"Dataset report saved to {output_path}")

        return output_path

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