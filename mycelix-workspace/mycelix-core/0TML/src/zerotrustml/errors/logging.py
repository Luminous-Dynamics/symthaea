# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Error Logging Utilities

Provides structured error logging in JSON format for production
and human-readable format for development.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .base import MycelixError
from .categories import ErrorSeverity


def format_error_json(
    error: MycelixError,
    include_stack: bool = False
) -> str:
    """
    Format error as JSON for structured logging.

    Output format:
    {
        "timestamp": "2024-01-15T10:30:00.000Z",
        "error_type": "NetworkError",
        "code": "ERR_3000",
        "category": "network",
        "severity": "error",
        "message": "Connection failed",
        "recoverable": true,
        "retry_after": 5.0,
        "context": {...},
        "cause": {...}
    }
    """
    data = error.to_dict()
    data["timestamp"] = datetime.now(timezone.utc).isoformat()

    if include_stack and error.context and error.context.stack_trace:
        data["stack_trace"] = error.context.stack_trace

    return json.dumps(data, default=str, indent=None)


def format_error_human(
    error: MycelixError,
    include_context: bool = True,
    include_stack: bool = False
) -> str:
    """
    Format error for human-readable console output.

    Output format:
    [ERR_3000] NetworkError: Connection failed
      Category: network | Severity: error | Recoverable: yes (retry after 5.0s)
      Context: at src/holochain/bridge.py:42 during connect_conductor
      Caused by: ConnectionRefusedError: Connection refused
    """
    lines = [f"[{error.code.value}] {type(error).__name__}: {error.message}"]

    # Status line
    status_parts = [
        f"Category: {error.category.value}",
        f"Severity: {error.severity.value}",
    ]
    if error.recoverable:
        retry_info = f"yes (retry after {error.retry_after}s)" if error.retry_after else "yes"
        status_parts.append(f"Recoverable: {retry_info}")
    else:
        status_parts.append("Recoverable: no")
    lines.append(f"  {' | '.join(status_parts)}")

    # Context
    if include_context and error.context:
        lines.append(f"  Context: {error.context}")
        if error.context.metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in error.context.metadata.items())
            lines.append(f"  Metadata: {meta_str}")

    # Cause
    if error.cause:
        lines.append(f"  Caused by: {type(error.cause).__name__}: {error.cause}")

    # Stack trace
    if include_stack and error.context and error.context.stack_trace:
        lines.append("  Stack trace:")
        for line in error.context.stack_trace.strip().split("\n"):
            lines.append(f"    {line}")

    return "\n".join(lines)


class ErrorLogger:
    """
    Specialized logger for MycelixErrors.

    Automatically:
    - Uses appropriate log level based on severity
    - Formats as JSON in production
    - Formats as human-readable in development
    - Includes structured fields for log aggregation
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        json_format: bool = True,
        include_stack_on_error: bool = False
    ):
        self.logger = logger or logging.getLogger("mycelix.errors")
        self.json_format = json_format
        self.include_stack_on_error = include_stack_on_error

    def log(
        self,
        error: MycelixError,
        extra_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log error with appropriate level and format.

        Args:
            error: The error to log
            extra_context: Additional context to include
        """
        level = error.severity.log_level
        include_stack = self.include_stack_on_error and error.severity >= ErrorSeverity.ERROR

        if self.json_format:
            message = format_error_json(error, include_stack=include_stack)
        else:
            message = format_error_human(error, include_stack=include_stack)

        # Build extra fields for structured logging
        extra = {
            "error_code": error.code.value,
            "error_category": error.category.value,
            "error_severity": error.severity.value,
            "error_recoverable": error.recoverable,
        }

        if error.context:
            if error.context.correlation_id:
                extra["correlation_id"] = error.context.correlation_id
            if error.context.operation:
                extra["operation"] = error.context.operation
            if error.context.component:
                extra["component"] = error.context.component

        if extra_context:
            extra.update(extra_context)

        self.logger.log(level, message, extra=extra)

    def debug(self, error: MycelixError, **extra) -> None:
        """Log error at DEBUG level (overrides severity)."""
        self.logger.debug(
            format_error_human(error) if not self.json_format else format_error_json(error),
            extra=extra
        )

    def info(self, error: MycelixError, **extra) -> None:
        """Log error at INFO level (overrides severity)."""
        self.logger.info(
            format_error_human(error) if not self.json_format else format_error_json(error),
            extra=extra
        )

    def warning(self, error: MycelixError, **extra) -> None:
        """Log error at WARNING level (overrides severity)."""
        self.logger.warning(
            format_error_human(error) if not self.json_format else format_error_json(error),
            extra=extra
        )

    def error(self, error: MycelixError, **extra) -> None:
        """Log error at ERROR level (overrides severity)."""
        self.logger.error(
            format_error_human(error) if not self.json_format else format_error_json(error),
            extra=extra
        )

    def critical(self, error: MycelixError, **extra) -> None:
        """Log error at CRITICAL level (overrides severity)."""
        self.logger.critical(
            format_error_human(error) if not self.json_format else format_error_json(error),
            extra=extra
        )


# Global error logger instance
_error_logger: Optional[ErrorLogger] = None


def get_error_logger() -> ErrorLogger:
    """Get or create global error logger."""
    global _error_logger
    if _error_logger is None:
        _error_logger = ErrorLogger()
    return _error_logger


def configure_error_logger(
    json_format: bool = True,
    include_stack_on_error: bool = False,
    logger: Optional[logging.Logger] = None
) -> ErrorLogger:
    """Configure and return global error logger."""
    global _error_logger
    _error_logger = ErrorLogger(
        logger=logger,
        json_format=json_format,
        include_stack_on_error=include_stack_on_error,
    )
    return _error_logger


def log_error(
    error: MycelixError,
    extra_context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Convenience function to log an error.

    Uses global error logger.
    """
    get_error_logger().log(error, extra_context)
