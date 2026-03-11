from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


LOGGER_NAME = "eva_robot"


def configure_logging(level: str = "INFO") -> None:
    normalized_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=normalized_level,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    return str(value)


class StructuredLogger:
    def __init__(self, name: str = LOGGER_NAME) -> None:
        self._logger = logging.getLogger(name)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, **fields)

    def _log(self, level: int, event: str, **fields: Any) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
        }
        payload.update({key: _serialize_value(value) for key, value in fields.items()})
        self._logger.log(level, json.dumps(payload, ensure_ascii=False))
