import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache

# Try importing settings, but don't fail if circular import
try:
    from app.core.config import get_settings
    settings = get_settings()
    LOG_LEVEL = getattr(logging, settings.LOG_LEVEL.upper())
    LOG_DIR = Path(settings.DATA_DIR) / "logs"
except ImportError:
    LOG_LEVEL = logging.INFO
    LOG_DIR = Path("./data/logs")

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    """
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields from record
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_data)


@lru_cache()
def setup_logger(name: str = "bo_engine") -> logging.Logger:
    """
    Setup and configure logger.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    # File handler for structured logs (JSON)
    log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(JsonFormatter())

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log initialization
    logger.info(f"Logger initialized: {name}")

    return logger
