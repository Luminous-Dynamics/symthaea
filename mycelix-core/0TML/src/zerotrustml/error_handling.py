# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Error Handling Utilities for Mycelix/ZeroTrustML

Production-grade error handling patterns:
- Retry decorators with exponential backoff
- Circuit breaker for external services
- Error aggregation for batch operations
- Graceful degradation helpers

Usage:
    from zerotrustml.error_handling import (
        retry_with_backoff,
        async_retry,
        CircuitBreaker,
        ErrorAggregator
    )

    # Retry with exponential backoff
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def call_external_service():
        ...

    # Circuit breaker for external services
    holochain_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

    @holochain_breaker
    async def call_holochain():
        ...

    # Error aggregation for batch operations
    aggregator = ErrorAggregator()
    for item in batch:
        with aggregator.capture(item_id=item.id):
            process(item)
    if aggregator.has_errors:
        aggregator.raise_if_critical()
"""

import asyncio
import functools
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union
)

from .exceptions import MycelixError, ErrorCode
from .logging import get_logger

logger = get_logger(__name__)

# Type variables
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])


# ============================================================
# Retry Configuration
# ============================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_max: float = 0.5

    # Exception handling
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()

    # Callbacks
    on_retry: Optional[Callable[[int, Exception, float], None]] = None

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        if self.jitter:
            import random
            jitter_amount = random.uniform(0, self.jitter_max * delay)
            delay += jitter_amount

        return delay


# ============================================================
# Retry Decorators
# ============================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
) -> Callable[[F], F]:
    """
    Decorator for retry with exponential backoff (synchronous functions).

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Add random jitter to delay
        retryable_exceptions: Exceptions that trigger retry
        non_retryable_exceptions: Exceptions that should not be retried
        on_retry: Callback called on each retry (attempt, exception, delay)

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def call_external_api():
            response = requests.get(...)
            return response.json()
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
        on_retry=on_retry
    )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.non_retryable_exceptions:
                    raise
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_retries:
                        logger.error(
                            f"All retries exhausted for {func.__qualname__}",
                            extra={
                                'function': func.__qualname__,
                                'attempts': attempt + 1,
                                'error': str(e)
                            }
                        )
                        raise

                    delay = config.calculate_delay(attempt)

                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for {func.__qualname__} "
                        f"after {delay:.2f}s: {e}",
                        extra={
                            'function': func.__qualname__,
                            'attempt': attempt + 1,
                            'max_retries': config.max_retries,
                            'delay_seconds': delay,
                            'error': str(e)
                        }
                    )

                    if config.on_retry:
                        config.on_retry(attempt, e, delay)

                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


def async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator for retry with exponential backoff (async functions).

    Args:
        Same as retry_with_backoff

    Example:
        @async_retry(max_retries=3)
        async def fetch_data():
            async with aiohttp.ClientSession() as session:
                async with session.get(...) as response:
                    return await response.json()
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
        on_retry=on_retry
    )

    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except config.non_retryable_exceptions:
                    raise
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_retries:
                        logger.error(
                            f"All retries exhausted for {func.__qualname__}",
                            extra={
                                'function': func.__qualname__,
                                'attempts': attempt + 1,
                                'error': str(e)
                            }
                        )
                        raise

                    delay = config.calculate_delay(attempt)

                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for {func.__qualname__} "
                        f"after {delay:.2f}s: {e}",
                        extra={
                            'function': func.__qualname__,
                            'attempt': attempt + 1,
                            'max_retries': config.max_retries,
                            'delay_seconds': delay,
                            'error': str(e)
                        }
                    )

                    if config.on_retry:
                        config.on_retry(attempt, e, delay)

                    await asyncio.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


# ============================================================
# Circuit Breaker
# ============================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_transitions: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None


class CircuitBreakerError(MycelixError):
    """Raised when circuit breaker is open."""

    def __init__(
        self,
        message: str,
        circuit_name: str,
        recovery_time: Optional[datetime] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'circuit_name': circuit_name,
            'recovery_time': recovery_time.isoformat() if recovery_time else None
        })
        super().__init__(
            message=message,
            code=ErrorCode.CONNECTION_REFUSED,
            context=context,
            recoverable=True,
            retry_after=30.0,
            **kwargs
        )


class CircuitBreaker:
    """
    Circuit breaker for external service calls.

    Prevents cascading failures by "opening" the circuit when failures
    exceed a threshold, allowing the external service time to recover.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Failure threshold exceeded, calls are rejected
    - HALF_OPEN: Testing if service recovered with limited calls

    Args:
        name: Name for logging and identification
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds to wait before testing recovery
        half_open_max_calls: Max calls allowed in half-open state
        success_threshold: Successes needed in half-open to close
        failure_exceptions: Exceptions that count as failures

    Example:
        holochain_breaker = CircuitBreaker(
            name="holochain",
            failure_threshold=5,
            recovery_timeout=60
        )

        @holochain_breaker
        async def call_holochain_zome(zome, fn, payload):
            return await client.call_zome(zome, fn, payload)
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
        failure_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        excluded_exceptions: Tuple[Type[Exception], ...] = ()
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.failure_exceptions = failure_exceptions
        self.excluded_exceptions = excluded_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.RLock()

        self.stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, transitioning from OPEN to HALF_OPEN if timeout passed."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        return self.state == CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        elapsed = datetime.now() - self._last_failure_time
        return elapsed >= timedelta(seconds=self.recovery_timeout)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self.stats.state_transitions += 1

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0

        logger.info(
            f"Circuit breaker '{self.name}' transitioned: {old_state.value} -> {new_state.value}",
            extra={
                'circuit_name': self.name,
                'old_state': old_state.value,
                'new_state': new_state.value,
                'failure_count': self._failure_count,
                'success_count': self._success_count
            }
        )

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self.stats.total_calls += 1
            self.stats.successful_calls += 1
            self.stats.last_success_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._failure_count = 0
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self.stats.total_calls += 1
            self.stats.failed_calls += 1
            self.stats.last_failure_time = datetime.now()

            self._failure_count += 1
            self._last_failure_time = datetime.now()

            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure: {exception}",
                extra={
                    'circuit_name': self.name,
                    'failure_count': self._failure_count,
                    'threshold': self.failure_threshold,
                    'error': str(exception)
                }
            )

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _check_state(self) -> None:
        """Check if calls are allowed, raising if circuit is open."""
        with self._lock:
            state = self.state  # Triggers potential OPEN -> HALF_OPEN

            if state == CircuitState.OPEN:
                self.stats.rejected_calls += 1
                recovery_time = None
                if self._last_failure_time:
                    recovery_time = self._last_failure_time + timedelta(seconds=self.recovery_timeout)

                raise CircuitBreakerError(
                    message=f"Circuit breaker '{self.name}' is OPEN",
                    circuit_name=self.name,
                    recovery_time=recovery_time
                )

            if state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self.stats.rejected_calls += 1
                    raise CircuitBreakerError(
                        message=f"Circuit breaker '{self.name}' is HALF_OPEN (max calls reached)",
                        circuit_name=self.name
                    )
                self._half_open_calls += 1

    def __call__(self, func: F) -> F:
        """Use circuit breaker as a decorator."""
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                self._check_state()
                try:
                    result = await func(*args, **kwargs)
                    self._record_success()
                    return result
                except self.excluded_exceptions:
                    self._record_success()  # Don't count as failure
                    raise
                except self.failure_exceptions as e:
                    self._record_failure(e)
                    raise

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                self._check_state()
                try:
                    result = func(*args, **kwargs)
                    self._record_success()
                    return result
                except self.excluded_exceptions:
                    self._record_success()
                    raise
                except self.failure_exceptions as e:
                    self._record_failure(e)
                    raise

            return sync_wrapper

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            self._transition_to(CircuitState.CLOSED)

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self._failure_count,
                'failure_threshold': self.failure_threshold,
                'total_calls': self.stats.total_calls,
                'successful_calls': self.stats.successful_calls,
                'failed_calls': self.stats.failed_calls,
                'rejected_calls': self.stats.rejected_calls,
                'state_transitions': self.stats.state_transitions,
                'last_failure': self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
                'last_success': self.stats.last_success_time.isoformat() if self.stats.last_success_time else None
            }


# ============================================================
# Error Aggregation
# ============================================================

@dataclass
class AggregatedError:
    """A single error in an aggregation."""
    exception: Exception
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    is_critical: bool = False


class BatchOperationError(MycelixError):
    """Raised when batch operation has errors."""

    def __init__(
        self,
        message: str,
        errors: List[AggregatedError],
        success_count: int,
        total_count: int,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'success_count': success_count,
            'error_count': len(errors),
            'total_count': total_count,
            'success_rate': success_count / total_count if total_count > 0 else 0
        })
        super().__init__(
            message=message,
            code=ErrorCode.INTERNAL_ERROR,
            context=context,
            **kwargs
        )
        self.errors = errors
        self.success_count = success_count
        self.total_count = total_count

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count > 0 else 0


class ErrorAggregator:
    """
    Aggregates errors during batch operations.

    Allows processing to continue despite individual failures,
    then provides summary and optionally raises if threshold exceeded.

    Args:
        critical_exception_types: Exception types that should stop immediately
        failure_threshold: Maximum failure rate before raising (0.0-1.0)
        max_errors: Maximum errors to collect (prevents memory issues)

    Example:
        aggregator = ErrorAggregator(failure_threshold=0.1)

        for item in items:
            with aggregator.capture(item_id=item.id):
                process(item)

        # Check results
        if aggregator.has_errors:
            logger.warning(f"Processed with {aggregator.error_count} errors")

        # Raise if too many failures
        aggregator.raise_if_threshold_exceeded()

        # Or raise on any critical error
        aggregator.raise_if_critical()
    """

    def __init__(
        self,
        critical_exception_types: Tuple[Type[Exception], ...] = (),
        failure_threshold: float = 1.0,
        max_errors: int = 1000
    ):
        self.critical_exception_types = critical_exception_types
        self.failure_threshold = failure_threshold
        self.max_errors = max_errors

        self._errors: List[AggregatedError] = []
        self._success_count = 0
        self._total_count = 0
        self._lock = threading.Lock()

    @property
    def has_errors(self) -> bool:
        """Check if any errors were recorded."""
        return len(self._errors) > 0

    @property
    def error_count(self) -> int:
        """Get number of errors."""
        return len(self._errors)

    @property
    def success_count(self) -> int:
        """Get number of successes."""
        return self._success_count

    @property
    def total_count(self) -> int:
        """Get total operation count."""
        return self._total_count

    @property
    def failure_rate(self) -> float:
        """Get failure rate (0.0-1.0)."""
        if self._total_count == 0:
            return 0.0
        return len(self._errors) / self._total_count

    @property
    def has_critical(self) -> bool:
        """Check if any critical errors were recorded."""
        return any(e.is_critical for e in self._errors)

    @contextmanager
    def capture(self, **context):
        """
        Context manager to capture errors.

        Args:
            **context: Context to attach to any captured error

        Example:
            with aggregator.capture(node_id="hospital-a", round=5):
                submit_gradient(gradient)
        """
        with self._lock:
            self._total_count += 1

        try:
            yield
            with self._lock:
                self._success_count += 1
        except Exception as e:
            is_critical = isinstance(e, self.critical_exception_types)

            error = AggregatedError(
                exception=e,
                context=context,
                is_critical=is_critical
            )

            with self._lock:
                if len(self._errors) < self.max_errors:
                    self._errors.append(error)

            # Re-raise critical exceptions immediately
            if is_critical:
                raise

    def get_errors(self) -> List[AggregatedError]:
        """Get list of aggregated errors."""
        return self._errors.copy()

    def get_errors_by_type(self) -> Dict[str, List[AggregatedError]]:
        """Group errors by exception type."""
        by_type: Dict[str, List[AggregatedError]] = {}
        for error in self._errors:
            type_name = type(error.exception).__name__
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(error)
        return by_type

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of aggregation results."""
        errors_by_type = self.get_errors_by_type()
        return {
            'total_count': self._total_count,
            'success_count': self._success_count,
            'error_count': len(self._errors),
            'failure_rate': self.failure_rate,
            'has_critical': self.has_critical,
            'error_types': {k: len(v) for k, v in errors_by_type.items()},
            'sample_errors': [
                {
                    'type': type(e.exception).__name__,
                    'message': str(e.exception),
                    'context': e.context
                }
                for e in self._errors[:5]  # First 5 errors as samples
            ]
        }

    def raise_if_threshold_exceeded(self) -> None:
        """Raise BatchOperationError if failure rate exceeds threshold."""
        if self.failure_rate > self.failure_threshold:
            raise BatchOperationError(
                message=f"Batch operation failure rate ({self.failure_rate:.1%}) "
                        f"exceeded threshold ({self.failure_threshold:.1%})",
                errors=self._errors,
                success_count=self._success_count,
                total_count=self._total_count
            )

    def raise_if_critical(self) -> None:
        """Raise BatchOperationError if any critical errors occurred."""
        if self.has_critical:
            critical_errors = [e for e in self._errors if e.is_critical]
            raise BatchOperationError(
                message=f"Batch operation had {len(critical_errors)} critical error(s)",
                errors=critical_errors,
                success_count=self._success_count,
                total_count=self._total_count
            )

    def raise_if_any_errors(self) -> None:
        """Raise BatchOperationError if any errors occurred."""
        if self.has_errors:
            raise BatchOperationError(
                message=f"Batch operation had {len(self._errors)} error(s)",
                errors=self._errors,
                success_count=self._success_count,
                total_count=self._total_count
            )

    def reset(self) -> None:
        """Reset aggregator state."""
        with self._lock:
            self._errors.clear()
            self._success_count = 0
            self._total_count = 0


# ============================================================
# Graceful Degradation
# ============================================================

class Fallback:
    """
    Decorator for graceful degradation with fallback values.

    Args:
        fallback_value: Value to return on failure
        fallback_func: Function to call on failure (receives exception)
        exceptions: Exceptions that trigger fallback
        log_level: Logging level for fallback events

    Example:
        @Fallback(fallback_value=0, exceptions=(ConnectionError,))
        async def get_credit_balance(node_id):
            return await holochain.get_balance(node_id)

        # Or with a fallback function:
        @Fallback(fallback_func=lambda e: cached_balance.get(node_id, 0))
        async def get_credit_balance(node_id):
            return await holochain.get_balance(node_id)
    """

    def __init__(
        self,
        fallback_value: Any = None,
        fallback_func: Optional[Callable[[Exception], Any]] = None,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        log_level: int = 20  # logging.WARNING
    ):
        self.fallback_value = fallback_value
        self.fallback_func = fallback_func
        self.exceptions = exceptions
        self.log_level = log_level

    def __call__(self, func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except self.exceptions as e:
                    logger.log(
                        self.log_level,
                        f"Fallback triggered for {func.__qualname__}: {e}",
                        extra={
                            'function': func.__qualname__,
                            'error': str(e),
                            'fallback_used': True
                        }
                    )
                    if self.fallback_func:
                        return self.fallback_func(e)
                    return self.fallback_value

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except self.exceptions as e:
                    logger.log(
                        self.log_level,
                        f"Fallback triggered for {func.__qualname__}: {e}",
                        extra={
                            'function': func.__qualname__,
                            'error': str(e),
                            'fallback_used': True
                        }
                    )
                    if self.fallback_func:
                        return self.fallback_func(e)
                    return self.fallback_value

            return sync_wrapper


# ============================================================
# Rate Limiter
# ============================================================

class RateLimiter:
    """
    Token bucket rate limiter.

    Args:
        rate: Tokens per second
        burst: Maximum burst size (bucket capacity)

    Example:
        limiter = RateLimiter(rate=10, burst=20)  # 10 req/s, burst of 20

        @limiter
        async def call_api():
            ...
    """

    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on time elapsed."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_update = now

    def acquire(self, tokens: int = 1, block: bool = True) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            block: If True, wait for tokens; if False, return immediately

        Returns:
            True if tokens acquired, False if not (only when block=False)
        """
        with self._lock:
            while True:
                self._refill()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                if not block:
                    return False

                # Calculate wait time
                wait_time = (tokens - self._tokens) / self.rate

        time.sleep(wait_time)
        return self.acquire(tokens, block=True)

    async def async_acquire(self, tokens: int = 1) -> bool:
        """Async version of acquire."""
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            wait_time = (tokens - self._tokens) / self.rate

        await asyncio.sleep(wait_time)
        return await self.async_acquire(tokens)

    def __call__(self, func: F) -> F:
        """Use rate limiter as a decorator."""
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                await self.async_acquire()
                return await func(*args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                self.acquire()
                return func(*args, **kwargs)

            return sync_wrapper


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Configuration
    'RetryConfig',

    # Retry decorators
    'retry_with_backoff',
    'async_retry',

    # Circuit breaker
    'CircuitState',
    'CircuitBreakerStats',
    'CircuitBreakerError',
    'CircuitBreaker',

    # Error aggregation
    'AggregatedError',
    'BatchOperationError',
    'ErrorAggregator',

    # Graceful degradation
    'Fallback',

    # Rate limiting
    'RateLimiter',
]
