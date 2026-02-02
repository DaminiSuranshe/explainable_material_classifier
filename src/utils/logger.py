"""
Centralized logging configuration.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Create a logger with console + file handlers.

    Args:
        name: Logger name
        log_dir: Directory to store logs
        level: Logging level

    Returns:
        Configured logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        Path(log_dir) / f"{name}.log"
    )
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
