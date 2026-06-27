"""LaTeX table generation for publication-quality papers."""

from typing import Any, Dict, List, Optional


def to_latex_table(
    results: List[Dict[str, Any]],
    caption: str = "Benchmark Results",
    label: str = "tab:benchmark",
    metrics: Optional[List[str]] = None,
) -> str:
    """Generate a LaTeX table from benchmark results.

    Args:
        results: List of result dictionaries with 'model_name', 'dataset_name', and metric keys.
        caption: Table caption.
        label: LaTeX label.
        metrics: Ordered list of metric columns to include. Auto-detected if None.

    Returns:
        LaTeX table string.
    """
    if not results:
        return ""

    if metrics is None:
        first = results[0]
        metrics = [k for k in first if k not in ("model_name", "dataset_name", "experiment_id", "experiment_name", "status", "config")]

    # Build header
    cols = ["Model", "Dataset"] + [m.replace("_", " ").title() for m in metrics]
    n_cols = len(cols)

    lines = [
        f"\\begin{{table}}[htbp]",
        f"\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{|{'c' * n_cols}|}}",
        "\\hline",
        " & ".join(f"\\textbf{{{c}}}" for c in cols) + " \\\\",
        "\\hline",
    ]

    for r in results:
        row = [r.get("model_name", ""), r.get("dataset_name", "")]
        for m in metrics:
            val = r.get(m, "")
            row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append(" & ".join(row) + " \\\\")

    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ]

    return "\n".join(lines) + "\n"