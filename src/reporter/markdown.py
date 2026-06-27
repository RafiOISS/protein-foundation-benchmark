"""Markdown table generation for reports and README."""

from typing import Any, Dict, List, Optional


def to_markdown_table(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> str:
    """Generate a Markdown table from benchmark results.

    Args:
        results: List of result dictionaries.
        metrics: Ordered list of metric columns. Auto-detected if None.

    Returns:
        Markdown table string.
    """
    if not results:
        return ""

    if metrics is None:
        first = results[0]
        metrics = [k for k in first if k not in ("model_name", "dataset_name", "experiment_id", "experiment_name", "status", "config")]

    headers = ["Model", "Dataset"] + [m.replace("_", " ").title() for m in metrics]
    sep = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]

    for r in results:
        row = [r.get("model_name", ""), r.get("dataset_name", "")]
        for m in metrics:
            val = r.get(m, "")
            row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"