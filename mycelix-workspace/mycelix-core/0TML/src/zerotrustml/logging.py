# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Unified Logging System for Mycelix/ZeroTrustML

Production-grade logging with:
- Structured JSON logging
- Correlation ID support for distributed tracing
- Performance timing decorators
- Context managers for operation tracking
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

Usage:
    from zerotrustml.logging import get_logger, configure_logging, correlation_context

    # Configure logging at application startup
    configure_logging(level="INFO", json_format=True)

    # Get a logger for your module
    logger = get_logger(__name__)

    # Use correlation IDs for distributed tracing
    with correlation_context(node_id="hospital-a"):
        logger.info("Processing gradient", extra={"round": 5})
"""

import logging
import json
import sys
import time
import uuid
import functools
import asyncio
from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from dataclasses import dataclass, field
import threading

# ============================================================
# Context Variables for Async-Safe Correlation
# ============================================================

# Primary correlation ID context variable (async-safe)
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')

# Additional context variables for distributed tracing
node_id: ContextVar[str] = ContextVar('node_id', default='')
round_id: ContextVar[int] = ContextVar('round_id', default=0)
operation_id: ContextVar[str] = ContextVar('operation_id', default='')

# Type variables for decorators
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])


# ============================================================
# Correlation ID Context
# ============================================================

class CorrelationContext:
    """Thread-local and async-aware correlation context for distributed tracing."""

    _local = threading.local()
    _async_contexts: Dict[int, Dict[str, Any]] = {}

    @classmethod
    def get(cls) -> Dict[str, Any]:
        """Get current correlation context."""
        # Try async context first
        try:
            task = asyncio.current_task()
            if task and id(task) in cls._async_contexts:
                return cls._async_contexts[id(task)]
        except RuntimeError:
            pass

        # Fall back to thread-local
        if not hasattr(cls._local, 'context'):
            cls._local.context = {}
        return cls._local.context

    @classmethod
    def set(cls, **kwargs) -> None:
        """Set correlation context values."""
        ctx = cls.get()
        ctx.update(kwargs)

    @classmethod
    def clear(cls) -> None:
        """Clear current context."""
        try:
            task = asyncio.current_task()
            if task and id(task) in cls._async_contexts:
                del cls._async_contexts[id(task)]
                return
        except RuntimeError:
            pass

        if hasattr(cls._local, 'context'):
            cls._local.context = {}

    @classmethod
    def get_correlation_id(cls) -> str:
        """Get or generate correlation ID."""
        ctx = cls.get()
        if 'correlation_id' not in ctx:
            ctx['correlation_id'] = str(uuid.uuid4())
        return ctx['correlation_id']

    @classmethod
    def set_async_context(cls, task_id: int, context: Dict[str, Any]) -> None:
        """Set async task context."""
        cls._async_contexts[task_id] = context

    @classmethod
    def remove_async_context(cls, task_id: int) -> None:
        """Remove async task context."""
        cls._async_contexts.pop(task_id, None)


@contextmanager
def correlation_context(
    correlation_id: Optional[str] = None,
    **extra_context
):
    """
    Context manager for setting correlation context.

    Args:
        correlation_id: Optional correlation ID (generated if not provided)
        **extra_context: Additional context values (node_id, round, etc.)

    Example:
        with correlation_context(node_id="hospital-a", round=5):
            logger.info("Processing gradient")
    """
    old_context = CorrelationContext.get().copy()

    new_context = {
        'correlation_id': correlation_id or str(uuid.uuid4()),
        **extra_context
    }
    CorrelationContext.set(**new_context)

    try:
        yield new_context
    finally:
        CorrelationContext.clear()
        CorrelationContext.set(**old_context)


@asynccontextmanager
async def async_correlation_context(
    correlation_id: Optional[str] = None,
    **extra_context
):
    """
    Async context manager for setting correlation context.

    Args:
        correlation_id: Optional correlation ID (generated if not provided)
        **extra_context: Additional context values

    Example:
        async with async_correlation_context(node_id="hospital-a"):
            await process_gradient()
    """
    task = asyncio.current_task()
    task_id = id(task) if task else 0

    old_context = CorrelationContext._async_contexts.get(task_id, {}).copy()

    new_context = {
        'correlation_id': correlation_id or str(uuid.uuid4()),
        **extra_context
    }
    CorrelationContext.set_async_context(task_id, new_context)

    try:
        yield new_context
    finally:
        if old_context:
            CorrelationContext.set_async_context(task_id, old_context)
        else:
            CorrelationContext.remove_async_context(task_id)


# ============================================================
# JSON Formatter
# ============================================================

class StructuredJSONFormatter(logging.Formatter):
    """
    Structured JSON log formatter for production observability.

    Output format:
    {
        "timestamp": "2024-01-15T10:30:00.000Z",
        "level": "INFO",
        "logger": "zerotrustml.core.coordinator",
        "message": "Gradient accepted",
        "correlation_id": "abc-123",
        "node_id": "hospital-a",
        "round": 5,
        "duration_ms": 42.5,
        "extra": {...}
    }
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_location: bool = False,
        extra_fields: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_location = include_location
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log entry
        log_entry = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Timestamp
        if self.include_timestamp:
            log_entry["timestamp"] = datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat()

        # Location info (file, line, function)
        if self.include_location:
            log_entry["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName
            }

        # Correlation context
        ctx = CorrelationContext.get()
        if ctx:
            log_entry.update(ctx)

        # Extra fields from record
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in (
                    'name', 'msg', 'args', 'created', 'filename', 'funcName',
                    'levelname', 'levelno', 'lineno', 'module', 'msecs',
                    'pathname', 'process', 'processName', 'relativeCreated',
                    'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                    'message', 'taskName'
                ):
                    log_entry[key] = value

        # Static extra fields (service name, environment, etc.)
        log_entry.update(self.extra_fields)

        # Exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Colored console formatter for development.

    Colors:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Bold Red
    """

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[1;31m', # Bold Red
    }
    RESET = '\033[0m'

    def __init__(
        self,
        fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M:%S",
        include_correlation: bool = True
    ):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.include_correlation = include_correlation

    def format(self, record: logging.LogRecord) -> str:
        # Add color
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"

        # Format base message
        formatted = super().format(record)

        # Add correlation context
        if self.include_correlation:
            ctx = CorrelationContext.get()
            if ctx:
                ctx_str = ' '.join(f'{k}={v}' for k, v in ctx.items())
                formatted = f"{formatted} [{ctx_str}]"

        return formatted


# ============================================================
# Custom Logger Class
# ============================================================

class MycelixLogger(logging.Logger):
    """
    Extended logger with Mycelix-specific features.

    Features:
    - Automatic correlation ID injection
    - Structured logging helpers
    - Performance timing methods
    """

    def _log_with_context(
        self,
        level: int,
        msg: str,
        args: tuple,
        exc_info: Any = None,
        extra: Optional[Dict[str, Any]] = None,
        stack_info: bool = False,
        **kwargs
    ) -> None:
        """Log with automatic context injection."""
        extra = extra or {}

        # Inject correlation context
        ctx = CorrelationContext.get()
        for key, value in ctx.items():
            if key not in extra:
                extra[key] = value

        # Add any kwargs as extra fields
        extra.update(kwargs)

        super()._log(level, msg, args, exc_info=exc_info, extra=extra, stack_info=stack_info)

    def info_with_context(self, msg: str, **kwargs) -> None:
        """Log info with structured context."""
        self._log_with_context(logging.INFO, msg, (), **kwargs)

    def error_with_context(self, msg: str, **kwargs) -> None:
        """Log error with structured context."""
        self._log_with_context(logging.ERROR, msg, (), **kwargs)

    def audit(self, msg: str, **kwargs) -> None:
        """Log audit event (always INFO level, always included)."""
        kwargs['audit'] = True
        self._log_with_context(logging.INFO, msg, (), **kwargs)


# Register custom logger class
logging.setLoggerClass(MycelixLogger)


# ============================================================
# Logger Factory
# ============================================================

def get_logger(name: str) -> MycelixLogger:
    """
    Get a Mycelix logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        MycelixLogger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Starting coordinator")
    """
    return logging.getLogger(name)


# ============================================================
# Configuration
# ============================================================

@dataclass
class LoggingConfig:
    """Logging configuration dataclass."""

    level: str = "INFO"
    json_format: bool = False
    include_timestamp: bool = True
    include_location: bool = False
    colored_output: bool = True

    # File logging
    log_file: Optional[str] = None
    log_file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_file_backup_count: int = 5

    # Extra fields for all log entries
    service_name: Optional[str] = None
    environment: Optional[str] = None
    version: Optional[str] = None

    # Module-specific levels
    module_levels: Dict[str, str] = field(default_factory=dict)


_configured = False


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    include_timestamp: bool = True,
    include_location: bool = False,
    colored_output: bool = True,
    log_file: Optional[str] = None,
    log_file_max_bytes: int = 10 * 1024 * 1024,
    log_file_backup_count: int = 5,
    service_name: Optional[str] = None,
    environment: Optional[str] = None,
    version: Optional[str] = None,
    module_levels: Optional[Dict[str, str]] = None,
    reset: bool = False
) -> None:
    """
    Configure Mycelix logging system.

    Args:
        level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use structured JSON format (for production)
        include_timestamp: Include timestamp in logs
        include_location: Include file/line info in logs
        colored_output: Use colored output for console (dev only)
        log_file: Path to log file (optional)
        log_file_max_bytes: Max size of log file before rotation
        log_file_backup_count: Number of backup log files to keep
        service_name: Service name for structured logs
        environment: Environment name (dev, staging, prod)
        version: Application version
        module_levels: Dict of module-specific log levels
        reset: Reset existing configuration

    Example:
        # Development
        configure_logging(level="DEBUG", colored_output=True)

        # Production
        configure_logging(
            level="INFO",
            json_format=True,
            service_name="mycelix-coordinator",
            environment="production",
            log_file="/var/log/mycelix/coordinator.log"
        )
    """
    global _configured

    if _configured and not reset:
        return

    # Build extra fields for structured logging
    extra_fields = {}
    if service_name:
        extra_fields['service'] = service_name
    if environment:
        extra_fields['environment'] = environment
    if version:
        extra_fields['version'] = version

    # Create root handler
    root_logger = logging.getLogger()

    # Clear existing handlers if resetting
    if reset:
        root_logger.handlers.clear()

    # Set root level
    root_logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        formatter = StructuredJSONFormatter(
            include_timestamp=include_timestamp,
            include_location=include_location,
            extra_fields=extra_fields
        )
    elif colored_output and sys.stdout.isatty():
        formatter = ColoredConsoleFormatter(include_correlation=True)
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=log_file_max_bytes,
            backupCount=log_file_backup_count
        )
        file_handler.setLevel(getattr(logging, level.upper()))

        # Always use JSON for file logs
        file_formatter = StructuredJSONFormatter(
            include_timestamp=True,
            include_location=True,
            extra_fields=extra_fields
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Module-specific levels
    if module_levels:
        for module, mod_level in module_levels.items():
            logging.getLogger(module).setLevel(getattr(logging, mod_level.upper()))

    _configured = True


def configure_from_dict(config: Dict[str, Any]) -> None:
    """Configure logging from a dictionary (e.g., from config file)."""
    configure_logging(**config)


# ============================================================
# Timing Decorators
# ============================================================

def timed(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    message: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator to log function execution time.

    Args:
        logger: Logger to use (defaults to function's module logger)
        level: Log level for timing messages
        message: Custom message template

    Example:
        @timed(level=logging.INFO)
        def process_gradient(gradient):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or get_logger(func.__module__)
            func_name = func.__qualname__

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                msg = message or f"{func_name} completed"
                _logger.log(level, msg, extra={
                    'function': func_name,
                    'duration_ms': round(elapsed_ms, 2),
                    'success': True
                })
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                _logger.log(logging.ERROR, f"{func_name} failed", extra={
                    'function': func_name,
                    'duration_ms': round(elapsed_ms, 2),
                    'success': False,
                    'error': str(e)
                })
                raise

        return wrapper
    return decorator


def async_timed(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    message: Optional[str] = None
) -> Callable[[AsyncF], AsyncF]:
    """
    Async decorator to log function execution time.

    Args:
        logger: Logger to use (defaults to function's module logger)
        level: Log level for timing messages
        message: Custom message template

    Example:
        @async_timed(level=logging.INFO)
        async def process_gradient(gradient):
            ...
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            _logger = logger or get_logger(func.__module__)
            func_name = func.__qualname__

            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                msg = message or f"{func_name} completed"
                _logger.log(level, msg, extra={
                    'function': func_name,
                    'duration_ms': round(elapsed_ms, 2),
                    'success': True
                })
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                _logger.log(logging.ERROR, f"{func_name} failed", extra={
                    'function': func_name,
                    'duration_ms': round(elapsed_ms, 2),
                    'success': False,
                    'error': str(e)
                })
                raise

        return wrapper
    return decorator


def with_correlation_id(
    correlation_id_param: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator to set correlation ID for a function invocation.

    The correlation ID is automatically propagated through the call stack
    and included in all log messages within the decorated function.

    Args:
        correlation_id_param: Optional correlation ID (generated if not provided)

    Example:
        @with_correlation_id()
        def process_request(request):
            logger.info("Processing request")  # Includes correlation_id
            process_items(request.items)

        @with_correlation_id("custom-correlation-123")
        async def handle_event(event):
            await process_event(event)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            cid = correlation_id_param or str(uuid.uuid4())
            token = correlation_id.set(cid)
            try:
                return func(*args, **kwargs)
            finally:
                correlation_id.reset(token)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cid = correlation_id_param or str(uuid.uuid4())
            token = correlation_id.set(cid)
            try:
                return await func(*args, **kwargs)
            finally:
                correlation_id.reset(token)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# ============================================================
# Context Managers for Operation Tracking
# ============================================================

@contextmanager
def log_operation(
    name: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    **context
):
    """
    Context manager for tracking operation timing and success.

    Args:
        name: Operation name
        logger: Logger to use
        level: Log level for messages
        **context: Additional context to include in logs

    Example:
        with log_operation("gradient_aggregation", round=5, nodes=10):
            aggregate_gradients(...)
    """
    _logger = logger or get_logger("zerotrustml")

    start_time = time.perf_counter()
    _logger.log(level, f"Starting: {name}", extra={'operation': name, **context})

    try:
        yield
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _logger.log(level, f"Completed: {name}", extra={
            'operation': name,
            'duration_ms': round(elapsed_ms, 2),
            'success': True,
            **context
        })
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _logger.log(logging.ERROR, f"Failed: {name}", extra={
            'operation': name,
            'duration_ms': round(elapsed_ms, 2),
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            **context
        })
        raise


@asynccontextmanager
async def async_log_operation(
    name: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    **context
):
    """
    Async context manager for tracking operation timing and success.

    Args:
        name: Operation name
        logger: Logger to use
        level: Log level for messages
        **context: Additional context to include in logs

    Example:
        async with async_log_operation("gradient_submission", node_id="hospital-a"):
            await submit_gradient(...)
    """
    _logger = logger or get_logger("zerotrustml")

    start_time = time.perf_counter()
    _logger.log(level, f"Starting: {name}", extra={'operation': name, **context})

    try:
        yield
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _logger.log(level, f"Completed: {name}", extra={
            'operation': name,
            'duration_ms': round(elapsed_ms, 2),
            'success': True,
            **context
        })
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _logger.log(logging.ERROR, f"Failed: {name}", extra={
            'operation': name,
            'duration_ms': round(elapsed_ms, 2),
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            **context
        })
        raise


# ============================================================
# Metrics Collection (for observability integration)
# ============================================================

class LogMetricsCollector:
    """
    Collects metrics from logs for observability integration.

    Can be used to export metrics to Prometheus, StatsD, etc.
    """

    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._histograms: Dict[str, list] = {}
        self._lock = threading.Lock()

    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter."""
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def record_timing(self, name: str, value_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing value."""
        key = self._make_key(name, tags)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value_ms)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self._lock:
            return {
                'counters': self._counters.copy(),
                'histograms': {
                    k: {
                        'count': len(v),
                        'min': min(v) if v else 0,
                        'max': max(v) if v else 0,
                        'avg': sum(v) / len(v) if v else 0
                    }
                    for k, v in self._histograms.items()
                }
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()

    def _make_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for a metric."""
        if not tags:
            return name
        tag_str = ','.join(f'{k}={v}' for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"


# Global metrics collector
metrics = LogMetricsCollector()


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Logger factory
    'get_logger',
    'MycelixLogger',

    # Configuration
    'configure_logging',
    'configure_from_dict',
    'LoggingConfig',

    # Context variables (async-safe)
    'correlation_id',
    'node_id',
    'round_id',
    'operation_id',

    # Correlation context
    'CorrelationContext',
    'correlation_context',
    'async_correlation_context',

    # Formatters
    'StructuredJSONFormatter',
    'ColoredConsoleFormatter',

    # Decorators
    'timed',
    'async_timed',
    'with_correlation_id',

    # Context managers
    'log_operation',
    'async_log_operation',

    # Metrics
    'LogMetricsCollector',
    'metrics',
]
