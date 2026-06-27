"""Excel export — multi-sheet benchmark reports."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def to_excel(
    results: List[Dict[str, Any]],
    path: Union[str, Path],
    sheets: Optional[Dict[str, List[Dict]]] = None,
) -> Path:
    """Export results to Excel with optional multiple sheets.

    Args:
        results: Main results list.
        path: Output path.
        sheets: Optional dict of sheet_name -> data for additional sheets.

    Returns:
        Path to the saved file.
    """
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(results).to_excel(writer, sheet_name="Results", index=False)

        if sheets:
            for sheet_name, data in sheets.items():
                pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name, index=False)

    return path