# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Structured Logging for ZeroTrustML.

Provides JSON-formatted logging suitable for log aggregation
systems like ELK, Datadog, or CloudWatch.
"""

import logging
import json
import time
import sys
import traceback
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from functools import wraps
from datetime import datetime
import threading


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, service_name: str = "zerotrustml", environment: str = "development"):
        super().__init__()
        self.service_name = service_name
        self.environment = environment
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "environment": self.environment,
        }
        
        # Add location info
        log_data["location"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message', 'taskName'
            }:
                log_data[key] = value
        
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Format extra fields
        extras = []
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message', 'taskName'
            }:
                if isinstance(value, (dict, list)):
                    extras.append(f"{key}={json.dumps(value)}")
                else:
                    extras.append(f"{key}={value}")
        
        extra_str = " " + " ".join(extras) if extras else ""
        
        base = f"{color}{timestamp} [{record.levelname:8}]{self.RESET} {record.name}: {record.getMessage()}{extra_str}"
        
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        
        return base


_configured = False
_loggers: Dict[str, logging.Logger] = {}


def configure_logging(
    service_name: str = "zerotrustml",
    environment: str = "development",
    level: int = logging.INFO,
    json_output: bool = None,
) -> None:
    """
    Configure logging for the application.
    
    Args:
        service_name: Name of the service for log tagging
        environment: Environment (development, staging, production)
        level: Logging level
        json_output: Force JSON output (auto-detected if None)
    """
    global _configured
    
    # Auto-detect JSON output (use JSON if not a TTY)
    if json_output is None:
        json_output = not sys.stderr.isatty()
    
    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)
    
    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    # Add appropriate handler
    handler = logging.StreamHandler(sys.stderr)
    
    if json_output:
        handler.setFormatter(JSONFormatter(service_name, environment))
    else:
        handler.setFormatter(ConsoleFormatter())
    
    root.addHandler(handler)
    
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger
    """
    global _configured
    
    if not _configured:
        configure_logging()
    
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    
    return _loggers[name]


@dataclass
class LogContext:
    """Context for structured log fields."""
    
    fields: Dict[str, Any] = field(default_factory=dict)
    _local: threading.local = field(default_factory=threading.local, repr=False)
    
    def __post_init__(self):
        if not hasattr(self._local, 'stack'):
            self._local.stack = []
    
    def add(self, **kwargs) -> 'LogContext':
        """Add fields to context."""
        self.fields.update(kwargs)
        return self
    
    @contextmanager
    def scope(self, **kwargs):
        """Create a nested context scope."""
        self._local.stack.append(self.fields.copy())
        self.fields.update(kwargs)
        try:
            yield self
        finally:
            self.fields = self._local.stack.pop()
    
    def get_extra(self) -> Dict[str, Any]:
        """Get fields for logging extra parameter."""
        return self.fields.copy()


def log_operation(
    logger: logging.Logger = None,
    operation: str = None,
    level: int = logging.INFO,
) -> Callable:
    """
    Decorator for logging function entry/exit with timing.
    
    Args:
        logger: Logger instance (defaults to function's module logger)
        operation: Operation name (defaults to function name)
        level: Logging level for success
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger, operation
        
        if logger is None:
            logger = get_logger(func.__module__)
        
        if operation is None:
            operation = func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            
            logger.log(level, f"{operation}_started", extra={
                "operation": operation,
                "phase": "start",
            })
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000
                
                logger.log(level, f"{operation}_completed", extra={
                    "operation": operation,
                    "phase": "complete",
                    "duration_ms": round(duration_ms, 2),
                })
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                
                logger.error(f"{operation}_failed", extra={
                    "operation": operation,
                    "phase": "error",
                    "duration_ms": round(duration_ms, 2),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }, exc_info=True)
                
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            
            logger.log(level, f"{operation}_started", extra={
                "operation": operation,
                "phase": "start",
            })
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000
                
                logger.log(level, f"{operation}_completed", extra={
                    "operation": operation,
                    "phase": "complete",
                    "duration_ms": round(duration_ms, 2),
                })
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                
                logger.error(f"{operation}_failed", extra={
                    "operation": operation,
                    "phase": "error",
                    "duration_ms": round(duration_ms, 2),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }, exc_info=True)
                
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


class FLLogger:
    """
    Specialized logger for Federated Learning operations.
    
    Provides convenience methods for common FL logging patterns.
    """
    
    def __init__(self, name: str, node_id: str = None):
        self.logger = get_logger(name)
        self.node_id = node_id
        self.context = LogContext()
        
        if node_id:
            self.context.add(node_id=node_id)
    
    def _extra(self, **kwargs) -> Dict[str, Any]:
        """Build extra dict with context."""
        extra = self.context.get_extra()
        extra.update(kwargs)
        return extra
    
    def round_started(self, round_id: int, participants: int):
        """Log FL round start."""
        self.logger.info("fl_round_started", extra=self._extra(
            round_id=round_id,
            participants=participants,
            event="round_start",
        ))
    
    def round_completed(self, round_id: int, duration_ms: float, accuracy: float = None):
        """Log FL round completion."""
        extra = self._extra(
            round_id=round_id,
            duration_ms=round(duration_ms, 2),
            event="round_complete",
        )
        if accuracy is not None:
            extra["accuracy"] = accuracy
        self.logger.info("fl_round_completed", extra=extra)
    
    def gradient_submitted(self, round_id: int, participant_id: str, dimension: int):
        """Log gradient submission."""
        self.logger.debug("gradient_submitted", extra=self._extra(
            round_id=round_id,
            participant_id=participant_id,
            dimension=dimension,
            event="gradient_submit",
        ))
    
    def gradient_rejected(self, round_id: int, participant_id: str, reason: str):
        """Log gradient rejection."""
        self.logger.warning("gradient_rejected", extra=self._extra(
            round_id=round_id,
            participant_id=participant_id,
            reason=reason,
            event="gradient_reject",
        ))
    
    def byzantine_detected(self, round_id: int, participant_ids: list, method: str):
        """Log Byzantine node detection."""
        self.logger.warning("byzantine_detected", extra=self._extra(
            round_id=round_id,
            byzantine_nodes=participant_ids,
            detection_method=method,
            byzantine_count=len(participant_ids),
            event="byzantine_detect",
        ))
    
    def aggregation_completed(self, round_id: int, method: str, participants: int, duration_ms: float):
        """Log aggregation completion."""
        self.logger.info("aggregation_completed", extra=self._extra(
            round_id=round_id,
            aggregation_method=method,
            participants=participants,
            duration_ms=round(duration_ms, 2),
            event="aggregation_complete",
        ))
    
    def reputation_updated(self, participant_id: str, old_score: float, new_score: float):
        """Log reputation update."""
        self.logger.info("reputation_updated", extra=self._extra(
            participant_id=participant_id,
            old_score=old_score,
            new_score=new_score,
            delta=round(new_score - old_score, 4),
            event="reputation_update",
        ))
    
    def contract_interaction(self, contract: str, function: str, success: bool, tx_hash: str = None):
        """Log smart contract interaction."""
        extra = self._extra(
            contract=contract,
            function=function,
            success=success,
            event="contract_call",
        )
        if tx_hash:
            extra["tx_hash"] = tx_hash
        
        level = logging.INFO if success else logging.ERROR
        self.logger.log(level, f"contract_{function}{'_succeeded' if success else '_failed'}", extra=extra)
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log an error with context."""
        extra = self._extra(**kwargs)
        if error:
            extra["error_type"] = type(error).__name__
            extra["error_message"] = str(error)
        self.logger.error(message, extra=extra, exc_info=error is not None)
