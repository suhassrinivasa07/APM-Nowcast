"""
Structured JSON logging configuration shared across all APM-Nowcast modules.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class _UTCJsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON with UTC timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any extra fields passed via the `extra` kwarg
        _skip = {
            "args", "asctime", "created", "exc_info", "exc_text",
            "filename", "funcName", "id", "levelname", "levelno",
            "lineno", "module", "msecs", "message", "msg", "name",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "thread", "threadName", "taskName",
        }
        for key, val in record.__dict__.items():
            if key not in _skip:
                try:
                    json.dumps(val)
                    payload[key] = val
                except (TypeError, ValueError):
                    payload[key] = str(val)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a module-level logger with structured JSON output.

    Usage::

        from common.logging_config import get_logger
        log = get_logger(__name__)
        log.info("File ingested", extra={"path": "/data/raw/foo.nc"})
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_UTCJsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger
