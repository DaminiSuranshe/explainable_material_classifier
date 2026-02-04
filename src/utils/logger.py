"""
Centralized logging configuration.
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: str | None = None,
    json_logs: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # prevent duplicate handlers

    handler = logging.StreamHandler(sys.stdout)

    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
            )
        )

    logger.addHandler(handler)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
        file_handler.setFormatter(handler.formatter)
        logger.addHandler(file_handler)

    return logger
