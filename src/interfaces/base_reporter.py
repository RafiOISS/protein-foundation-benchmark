"""Base reporter interface for the Protein Foundation Model Benchmark Framework.

All report/result generators must inherit from BaseReporter.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class BaseReporter(ABC):
    """Abstract base class for result reporters.

    Generates publication-quality reports and summaries from benchmark results.
    """

    @abstractmethod
    def generate_report(
        self,
        results: List[Dict[str, Any]],
        output_path: Union[str, Path],
        **kwargs,
    ) -> Path:
        """Generate a report from benchmark results.

        Args:
            results: List of result dictionaries.
            output_path: Path to save the report.
            **kwargs: Additional parameters.

        Returns:
            Path to the generated report.
        """
        pass

    @abstractmethod
    def generate_summary(
        self,
        results: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a summary dictionary from results.

        Args:
            results: List of result dictionaries.
            **kwargs: Additional parameters.

        Returns:
            Summary dictionary.
        """
        pass

    @abstractmethod
    def export(
        self,
        data: Any,
        output_path: Union[str, Path],
        format: str = "json",
        **kwargs,
    ) -> Path:
        """Export data to a file.

        Args:
            data: Data to export.
            output_path: Output path.
            format: Export format ('json', 'csv', 'parquet', 'html').
            **kwargs: Additional parameters.

        Returns:
            Path to the exported file.
        """
        pass