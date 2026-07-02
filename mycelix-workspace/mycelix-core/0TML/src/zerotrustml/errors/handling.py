# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Error Handling Utilities

Provides utilities for safe execution and error collection:
- safe_execute: Execute function with error wrapping
- collect_errors: Context manager for error collection
- ErrorCollector: Collect multiple errors in batch operations
"""

import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any, Callable, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union
)

from .base import MycelixError, wrap_exception
from .categories import ErrorCode, ErrorSeverity
from .context import ErrorContext
from .logging import log_error


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def safe_execute(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
    reraise: bool = False,
    wrap_as: Optional[Type[MycelixError]] = None,
    error_code: ErrorCode = ErrorCode.INTERNAL,
    log_errors: bool = True,
    **kwargs
) -> Tuple[Optional[T], Optional[MycelixError]]:
    """
    Execute a function with error handling.

    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default value on error
        on_error: Callback on error
        reraise: Re-raise after handling
        wrap_as: Wrap exceptions in this MycelixError type
        error_code: Error code for wrapped exceptions
        log_errors: Log errors automatically
        **kwargs: Keyword arguments

    Returns:
        Tuple of (result or default, error or None)

    Usage:
        result, error = safe_execute(
            connect_to_holochain,
            host="localhost",
            port=8888,
            default=None,
            log_errors=True
        )
        if error:
            handle_connection_failure(error)
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except MycelixError as e:
        if log_errors:
            log_error(e)
        if on_error:
            on_error(e)
        if reraise:
            raise
        return default, e
    except Exception as e:
        # Wrap in MycelixError
        if wrap_as:
            wrapped = wrap_as(
                message=str(e),
                code=error_code,
                cause=e,
            )
        else:
            wrapped = wrap_exception(e, code=error_code)

        if log_errors:
            log_error(wrapped)
        if on_error:
            on_error(wrapped)
        if reraise:
            raise wrapped from e
        return default, wrapped


def safe_execute_decorator(
    default: Optional[Any] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
    error_code: ErrorCode = ErrorCode.INTERNAL,
    log_errors: bool = True
) -> Callable[[F], F]:
    """
    Decorator version of safe_execute.

    Usage:
        @safe_execute_decorator(default=[], log_errors=True)
        def get_gradients():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result, error = safe_execute(
                func,
                *args,
                default=default,
                on_error=on_error,
                error_code=error_code,
                log_errors=log_errors,
                **kwargs
            )
            return result

        return wrapper
    return decorator


@dataclass
class CollectedError:
    """A single error in a collection."""
    error: MycelixError
    item_id: Optional[str] = None
    item_index: Optional[int] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorCollector:
    """
    Collect errors from batch operations.

    Allows processing to continue despite individual failures,
    then provides summary and allows controlled error handling.

    Args:
        fail_fast_on: Exception types that should stop immediately
        max_errors: Maximum errors to collect
        log_errors: Log errors as they occur

    Usage:
        collector = ErrorCollector(max_errors=100)

        for i, item in enumerate(items):
            with collector.capture(item_id=str(item.id), item_index=i):
                process(item)

        # Check results
        if collector.has_errors:
            print(f"Failed {collector.error_count}/{collector.total_count}")

        # Get all errors for reporting
        for err in collector.errors:
            print(f"  Item {err.item_id}: {err.error.message}")

        # Raise if threshold exceeded
        collector.raise_if_threshold_exceeded(0.1)  # 10% failure rate
    """

    def __init__(
        self,
        fail_fast_on: Tuple[Type[Exception], ...] = (),
        max_errors: int = 1000,
        log_errors: bool = True
    ):
        self.fail_fast_on = fail_fast_on
        self.max_errors = max_errors
        self.log_errors = log_errors

        self._errors: List[CollectedError] = []
        self._success_count = 0
        self._total_count = 0

    @property
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self._errors) > 0

    @property
    def error_count(self) -> int:
        """Number of errors collected."""
        return len(self._errors)

    @property
    def success_count(self) -> int:
        """Number of successful operations."""
        return self._success_count

    @property
    def total_count(self) -> int:
        """Total operations attempted."""
        return self._total_count

    @property
    def failure_rate(self) -> float:
        """Failure rate (0.0-1.0)."""
        if self._total_count == 0:
            return 0.0
        return len(self._errors) / self._total_count

    @property
    def errors(self) -> List[CollectedError]:
        """Get collected errors."""
        return self._errors.copy()

    @contextmanager
    def capture(
        self,
        item_id: Optional[str] = None,
        item_index: Optional[int] = None
    ) -> Generator[None, None, None]:
        """
        Context manager to capture errors.

        Args:
            item_id: Identifier for the item being processed
            item_index: Index of the item in the batch

        Usage:
            with collector.capture(item_id="gradient-123"):
                validate_gradient(gradient)
        """
        self._total_count += 1

        try:
            yield
            self._success_count += 1
        except self.fail_fast_on:
            # Re-raise immediately
            raise
        except MycelixError as e:
            self._handle_error(e, item_id, item_index)
        except Exception as e:
            # Wrap generic exceptions
            wrapped = wrap_exception(e)
            self._handle_error(wrapped, item_id, item_index)

    def _handle_error(
        self,
        error: MycelixError,
        item_id: Optional[str],
        item_index: Optional[int]
    ) -> None:
        """Handle a captured error."""
        if self.log_errors:
            log_error(error, extra_context={
                "item_id": item_id,
                "item_index": item_index,
            })

        if len(self._errors) < self.max_errors:
            self._errors.append(CollectedError(
                error=error,
                item_id=item_id,
                item_index=item_index,
            ))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collection results."""
        errors_by_code: Dict[str, int] = {}
        for err in self._errors:
            code = err.error.code.value
            errors_by_code[code] = errors_by_code.get(code, 0) + 1

        return {
            "total_count": self._total_count,
            "success_count": self._success_count,
            "error_count": len(self._errors),
            "failure_rate": self.failure_rate,
            "errors_by_code": errors_by_code,
            "sample_errors": [
                {
                    "item_id": e.item_id,
                    "code": e.error.code.value,
                    "message": e.error.message,
                }
                for e in self._errors[:5]
            ],
        }

    def raise_if_any_errors(self) -> None:
        """Raise if any errors were collected."""
        if self.has_errors:
            raise MycelixError(
                message=f"Batch operation had {len(self._errors)} error(s)",
                code=ErrorCode.INTERNAL,
                severity=ErrorSeverity.ERROR,
                context=ErrorContext.capture(
                    metadata=self.get_summary(),
                    skip_frames=2,
                ),
            )

    def raise_if_threshold_exceeded(self, threshold: float) -> None:
        """Raise if failure rate exceeds threshold."""
        if self.failure_rate > threshold:
            raise MycelixError(
                message=f"Batch failure rate ({self.failure_rate:.1%}) exceeded threshold ({threshold:.1%})",
                code=ErrorCode.INTERNAL,
                severity=ErrorSeverity.ERROR,
                context=ErrorContext.capture(
                    metadata=self.get_summary(),
                    skip_frames=2,
                ),
            )

    def clear(self) -> None:
        """Clear collected errors and counters."""
        self._errors.clear()
        self._success_count = 0
        self._total_count = 0


@contextmanager
def collect_errors(
    fail_fast_on: Tuple[Type[Exception], ...] = (),
    log_errors: bool = True
) -> Generator[ErrorCollector, None, None]:
    """
    Context manager that provides an ErrorCollector.

    Usage:
        with collect_errors() as collector:
            for item in items:
                with collector.capture(item_id=item.id):
                    process(item)

        print(f"Processed with {collector.error_count} errors")
    """
    collector = ErrorCollector(
        fail_fast_on=fail_fast_on,
        log_errors=log_errors,
    )
    yield collector
