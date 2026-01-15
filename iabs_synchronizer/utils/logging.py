"""
Logging utilities for capturing and managing synchronization process logs.

This module provides tools for capturing print statements during processing,
replacing the notebook's Capturing() utility with proper logging infrastructure.
"""

import sys
import logging
from io import StringIO
from contextlib import contextmanager
from typing import Generator


@contextmanager
def LogCapture() -> Generator[StringIO, None, None]:
    """
    Context manager to capture print statements to a StringIO buffer.

    This replaces the notebook's Capturing() utility. All print statements
    within the context are captured and can be retrieved via getvalue().

    Yields:
        StringIO: Buffer containing captured output

    Example:
        >>> with LogCapture() as log:
        ...     print("Loading data...")
        ...     print("Processing...")
        >>> messages = log.getvalue()
        >>> print(messages)
        Loading data...
        Processing...

        >>> # Split into list of lines
        >>> log_lines = log.getvalue().split('\\n')
        >>> print(log_lines)
        ['Loading data...', 'Processing...', '']
    """
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    try:
        yield captured
    finally:
        sys.stdout = old_stdout


def setup_logging(level: int = logging.INFO,
                 format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s') -> None:
    """
    Configure logging for the synchronizer package.

    Args:
        level: Logging level (default: logging.INFO)
        format_string: Format for log messages

    Example:
        >>> from iabs_synchronizer.utils.logging import setup_logging
        >>> import logging
        >>> setup_logging(level=logging.DEBUG)
        >>> logging.info("Synchronization started")
    """
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


class SynchronizerLogger:
    """
    Dedicated logger for synchronization operations.

    Provides structured logging with automatic prefixing for different
    synchronization phases (read, filter, align).

    Attributes:
        logger: Python logger instance
        logs: Dictionary storing logs by phase

    Example:
        >>> logger = SynchronizerLogger('experiment_001')
        >>> logger.log_read("Loading calcium data...")
        >>> logger.log_filter("Filtered 10 NaN indices")
        >>> logger.log_align("Aligned using cast_to_ca mode")
        >>> print(logger.get_logs('read'))
        ['Loading calcium data...']
    """

    def __init__(self, name: str = 'synchronizer'):
        """
        Initialize logger.

        Args:
            name: Logger name (typically experiment name)
        """
        self.logger = logging.getLogger(name)
        self.logs = {
            'read': [],
            'filter': [],
            'align': [],
            'postprocess': []
        }

    def log_read(self, message: str) -> None:
        """Log message from data reading phase."""
        self.logger.info(f"[READ] {message}")
        self.logs['read'].append(message)

    def log_filter(self, message: str) -> None:
        """Log message from filtering phase."""
        self.logger.info(f"[FILTER] {message}")
        self.logs['filter'].append(message)

    def log_align(self, message: str) -> None:
        """Log message from alignment phase."""
        self.logger.info(f"[ALIGN] {message}")
        self.logs['align'].append(message)

    def log_postprocess(self, message: str) -> None:
        """Log message from postprocessing phase."""
        self.logger.info(f"[POSTPROCESS] {message}")
        self.logs['postprocess'].append(message)

    def get_logs(self, phase: str) -> list:
        """
        Get logs for a specific phase.

        Args:
            phase: Phase name ('read', 'filter', 'align', 'postprocess')

        Returns:
            List of log messages for that phase
        """
        return self.logs.get(phase, [])

    def get_all_logs(self) -> dict:
        """
        Get all logs organized by phase.

        Returns:
            Dictionary mapping phase names to log message lists
        """
        return self.logs.copy()

    def clear_logs(self) -> None:
        """Clear all stored logs."""
        for phase in self.logs:
            self.logs[phase] = []
