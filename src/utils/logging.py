"""Logging Utilities for the Protein Foundation Model Benchmark Framework.

Provides structured logging configuration and utilities.
Rich is optional — falls back to standard logging if not installed.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    from rich.logging import RichHandler
    from rich.console import Console
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    console: bool = True,
    rich_console: bool = True,
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        level: Logging level.
        log_file: Optional log file path.
        format_string: Custom format string.
        console: Whether to log to console.
        rich_console: Whether to use rich console handler.
        max_bytes: Max log file size before rotation.
        backup_count: Number of backup files.

    Returns:
        Configured root logger.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Convert level to int if string
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    # Console handler
    if console:
        if rich_console and _HAS_RICH:
            console_handler = RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
            )
        else:
            console_handler = logging.StreamHandler(sys.stdout)

        console_handler.setLevel(level)
        console_formatter = logging.Formatter(format_string)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get logger instance.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def log_config(config: Dict[str, Any], logger: logging.Logger, prefix: str = "Config") -> None:
    """Log configuration dictionary.

    Args:
        config: Configuration dictionary.
        logger: Logger instance.
        prefix: Prefix for log messages.
    """
    logger.info(f"{prefix}:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")


def log_metrics(metrics: Dict[str, float], logger: logging.Logger, prefix: str = "Metrics") -> None:
    """Log metrics dictionary.

    Args:
        metrics: Metrics dictionary.
        logger: Logger instance.
        prefix: Prefix for log messages.
    """
    logger.info(f"{prefix}:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.6f}")


def set_log_level(logger: logging.Logger, level: Union[int, str]) -> None:
    """Set log level for a logger.

    Args:
        logger: Logger instance.
        level: Log level.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)


class TqdmLoggingHandler(logging.Handler):
    """Logging handler compatible with tqdm progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)