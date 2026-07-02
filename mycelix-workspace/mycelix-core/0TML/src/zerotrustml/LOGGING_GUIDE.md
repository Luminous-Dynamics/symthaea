# Mycelix Logging and Error Handling Guide

This document provides comprehensive usage documentation for the unified logging and error handling system in Mycelix/ZeroTrustML.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Logging System](#logging-system)
   - [Configuration](#configuration)
   - [Getting a Logger](#getting-a-logger)
   - [Structured Logging](#structured-logging)
   - [Correlation IDs](#correlation-ids)
   - [Performance Timing](#performance-timing)
3. [Exception Hierarchy](#exception-hierarchy)
   - [Base Exception](#base-exception)
   - [Exception Categories](#exception-categories)
   - [Error Codes](#error-codes)
4. [Error Handling Patterns](#error-handling-patterns)
   - [Retry with Backoff](#retry-with-backoff)
   - [Circuit Breaker](#circuit-breaker)
   - [Error Aggregation](#error-aggregation)
   - [Graceful Degradation](#graceful-degradation)
5. [Best Practices](#best-practices)
6. [Production Configuration](#production-configuration)

---

## Quick Start

```python
from zerotrustml.logging import (
    configure_logging,
    get_logger,
    correlation_context,
    timed
)
from zerotrustml.exceptions import (
    MycelixError,
    ByzantineDetectionError,
    HolochainConnectionError
)
from zerotrustml.error_handling import (
    retry_with_backoff,
    CircuitBreaker,
    ErrorAggregator
)

# Configure logging at application startup
configure_logging(
    level="INFO",
    json_format=True,
    service_name="mycelix-coordinator"
)

# Get a logger for your module
logger = get_logger(__name__)

# Log with correlation context
with correlation_context(node_id="hospital-a"):
    logger.info("Processing gradient", extra={"round": 5})
```

---

## Logging System

### Configuration

Configure logging once at application startup:

```python
from zerotrustml.logging import configure_logging

# Development configuration
configure_logging(
    level="DEBUG",
    json_format=False,
    colored_output=True
)

# Production configuration
configure_logging(
    level="INFO",
    json_format=True,
    include_timestamp=True,
    include_location=True,
    service_name="mycelix-coordinator",
    environment="production",
    version="1.0.0",
    log_file="/var/log/mycelix/coordinator.log",
    log_file_max_bytes=10 * 1024 * 1024,  # 10MB
    log_file_backup_count=5,
    module_levels={
        "zerotrustml.holochain": "DEBUG",
        "zerotrustml.aggregation": "INFO"
    }
)
```

#### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | str | "INFO" | Default log level |
| `json_format` | bool | False | Use JSON structured output |
| `include_timestamp` | bool | True | Include timestamps |
| `include_location` | bool | False | Include file/line info |
| `colored_output` | bool | True | Colorize console output |
| `log_file` | str | None | Path to log file |
| `log_file_max_bytes` | int | 10MB | Max log file size |
| `log_file_backup_count` | int | 5 | Number of backup files |
| `service_name` | str | None | Service name for structured logs |
| `environment` | str | None | Environment (dev/staging/prod) |
| `version` | str | None | Application version |
| `module_levels` | dict | None | Module-specific log levels |

### Getting a Logger

```python
from zerotrustml.logging import get_logger

# Get a logger for your module (use __name__ for automatic module naming)
logger = get_logger(__name__)

# Basic logging
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")

# Logging with extra context
logger.info("Gradient received", extra={
    "node_id": "hospital-a",
    "round": 5,
    "gradient_hash": "abc123"
})
```

### Structured Logging

When `json_format=True`, logs are output as JSON for easy parsing:

```json
{
    "timestamp": "2024-01-15T10:30:00.000Z",
    "level": "INFO",
    "logger": "zerotrustml.core.coordinator",
    "message": "Gradient accepted",
    "correlation_id": "abc-123-def-456",
    "node_id": "hospital-a",
    "round": 5,
    "duration_ms": 42.5,
    "service": "mycelix-coordinator",
    "environment": "production"
}
```

### Correlation IDs

Correlation IDs enable distributed tracing across service boundaries.

#### Using Context Variables (Recommended for Async)

```python
from zerotrustml.logging import correlation_id, node_id, get_logger

logger = get_logger(__name__)

# Set correlation ID for current async context
token = correlation_id.set("request-12345")
try:
    logger.info("Processing request")  # Includes correlation_id in context
finally:
    correlation_id.reset(token)
```

#### Using the Decorator

```python
from zerotrustml.logging import with_correlation_id, get_logger

logger = get_logger(__name__)

@with_correlation_id()
def process_request(request):
    # All logs within this function include a correlation ID
    logger.info("Processing request")
    handle_items(request.items)

@with_correlation_id("custom-correlation-123")
async def handle_event(event):
    await process_event(event)
```

#### Using Context Managers

```python
from zerotrustml.logging import correlation_context, async_correlation_context

# Synchronous context
with correlation_context(node_id="hospital-a", round=5):
    logger.info("Processing gradient")  # Includes node_id and round

# Asynchronous context
async with async_correlation_context(node_id="hospital-a"):
    await process_gradient()
```

### Performance Timing

#### Timing Decorators

```python
from zerotrustml.logging import timed, async_timed
import logging

@timed(level=logging.INFO)
def process_gradient(gradient):
    # Function execution time is automatically logged
    return validate_and_process(gradient)

@async_timed(level=logging.INFO, message="Gradient aggregation complete")
async def aggregate_gradients(gradients):
    return await perform_aggregation(gradients)
```

Output:
```json
{
    "level": "INFO",
    "message": "process_gradient completed",
    "function": "process_gradient",
    "duration_ms": 42.5,
    "success": true
}
```

#### Operation Context Managers

```python
from zerotrustml.logging import log_operation, async_log_operation

# Synchronous operations
with log_operation("gradient_aggregation", round=5, nodes=10):
    result = aggregate_gradients(gradients)

# Asynchronous operations
async with async_log_operation("gradient_submission", node_id="hospital-a"):
    await submit_gradient(gradient)
```

---

## Exception Hierarchy

### Base Exception

All Mycelix exceptions inherit from `MycelixError`:

```python
from zerotrustml.exceptions import MycelixError, ErrorCode

try:
    process_gradient(gradient)
except MycelixError as e:
    print(f"Error code: {e.code.value}")
    print(f"Message: {e.message}")
    print(f"Context: {e.context}")
    print(f"Recoverable: {e.recoverable}")
    print(f"Retry after: {e.retry_after} seconds")

    # Serialize to dict for API responses
    error_dict = e.to_dict()
```

### Exception Categories

#### Byzantine Detection Errors

```python
from zerotrustml.exceptions import (
    ByzantineDetectionError,
    GradientPoisoningError,
    SybilAttackError
)

# Raise when Byzantine behavior detected
raise ByzantineDetectionError(
    message="Gradient statistical anomaly detected",
    node_id="hospital-c",
    detection_method="statistical_analysis",
    severity="high",
    evidence={"z_score": 4.5, "threshold": 3.0}
)

# Specific gradient poisoning
raise GradientPoisoningError(
    message="Gradient values outside valid range",
    node_id="hospital-c",
    gradient_hash="abc123",
    pogq_score=0.15
)
```

#### Connection Errors

```python
from zerotrustml.exceptions import (
    HolochainConnectionError,
    HolochainZomeError,
    PostgresConnectionError
)

# Holochain connection failure
raise HolochainConnectionError(
    message="Failed to connect to Holochain conductor",
    admin_url="ws://localhost:9000",
    app_url="ws://localhost:9001"
)

# Zome call failure
raise HolochainZomeError(
    message="Zome call timed out",
    zome_name="gradient_store",
    fn_name="submit_gradient"
)
```

#### Proof Verification Errors

```python
from zerotrustml.exceptions import (
    ProofVerificationError,
    ZKPoCVerificationError
)

raise ZKPoCVerificationError(
    message="ZK-PoC range proof failed verification",
    node_id="hospital-a",
    threshold=0.7
)
```

#### Configuration Errors

```python
from zerotrustml.exceptions import (
    ConfigurationError,
    ConfigMissingError
)

raise ConfigMissingError(config_key="holochain.admin_url")

raise ConfigurationError(
    message="Invalid aggregation algorithm",
    config_key="aggregation.algorithm",
    expected_type="str",
    actual_value=123
)
```

#### Identity Errors

```python
from zerotrustml.exceptions import (
    IdentityError,
    IdentityNotFoundError,
    IdentityVerificationError,
    DIDResolutionError
)

raise IdentityVerificationError(
    message="Insufficient assurance level for operation",
    participant_id="hospital-a",
    required_assurance="high",
    actual_assurance="medium"
)
```

#### Governance Errors

```python
from zerotrustml.exceptions import (
    GovernanceError,
    GovernanceUnauthorizedError,
    CapabilityDeniedError,
    GuardianAuthorizationError
)

raise CapabilityDeniedError(
    capability_id="submit_gradient",
    participant_id="hospital-a",
    reason="Reputation below minimum threshold"
)

raise GuardianAuthorizationError(
    action="model_deployment",
    participant_id="hospital-a",
    threshold=0.67,
    current_approvals=2
)
```

### Error Codes

All errors have standardized codes for programmatic handling:

```python
from zerotrustml.exceptions import ErrorCode

# Error code categories:
# ERR_1xxx - General errors
# ERR_2xxx - Configuration errors
# ERR_3xxx - Connection errors
# ERR_4xxx - Byzantine detection errors
# ERR_5xxx - Proof verification errors
# ERR_6xxx - Identity errors
# ERR_7xxx - Governance errors
# ERR_8xxx - Storage errors
# ERR_9xxx - Aggregation errors
# ERR_10xxx - Encryption errors

# Example: Check error code
if error.code == ErrorCode.BYZANTINE_GRADIENT_POISONING:
    quarantine_node(error.context['node_id'])
```

---

## Error Handling Patterns

### Retry with Backoff

```python
from zerotrustml.error_handling import retry_with_backoff, async_retry
from zerotrustml.exceptions import HolochainConnectionError

# Synchronous retry
@retry_with_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    retryable_exceptions=(ConnectionError, TimeoutError),
    non_retryable_exceptions=(ValueError,)
)
def call_external_api():
    return requests.get("https://api.example.com/data").json()

# Asynchronous retry
@async_retry(
    max_retries=5,
    base_delay=0.5,
    retryable_exceptions=(HolochainConnectionError,)
)
async def submit_to_holochain(data):
    return await holochain_client.call_zome("gradient_store", "submit", data)

# With callback on retry
def on_retry_callback(attempt, exception, delay):
    metrics.increment("retry_count", tags={"function": "submit_gradient"})

@retry_with_backoff(max_retries=3, on_retry=on_retry_callback)
def submit_gradient(gradient):
    ...
```

### Circuit Breaker

```python
from zerotrustml.error_handling import CircuitBreaker, CircuitBreakerError

# Create a circuit breaker for Holochain calls
holochain_breaker = CircuitBreaker(
    name="holochain",
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=60.0,    # Wait 60s before testing
    half_open_max_calls=3,    # Allow 3 test calls in half-open
    success_threshold=2,      # Need 2 successes to close
    failure_exceptions=(ConnectionError, TimeoutError),
    excluded_exceptions=(ValueError,)  # Don't count as failures
)

# Use as decorator
@holochain_breaker
async def call_holochain_zome(zome, fn, payload):
    return await client.call_zome(zome, fn, payload)

# Handle circuit open state
try:
    result = await call_holochain_zome("gradient_store", "submit", gradient)
except CircuitBreakerError as e:
    logger.warning(f"Circuit breaker open: {e.circuit_name}")
    # Use fallback behavior
    result = await fallback_submit(gradient)

# Monitor circuit breaker state
stats = holochain_breaker.get_stats()
print(f"State: {stats['state']}")
print(f"Failure count: {stats['failure_count']}/{stats['failure_threshold']}")
print(f"Total calls: {stats['total_calls']}")

# Manual reset if needed
holochain_breaker.reset()
```

### Error Aggregation

For batch operations where you want to continue despite individual failures:

```python
from zerotrustml.error_handling import ErrorAggregator
from zerotrustml.exceptions import ByzantineDetectionError

# Create aggregator with configuration
aggregator = ErrorAggregator(
    critical_exception_types=(ByzantineDetectionError,),
    failure_threshold=0.1,  # Max 10% failure rate
    max_errors=1000         # Prevent memory issues
)

# Process batch with error capture
for gradient in gradients:
    with aggregator.capture(node_id=gradient.node_id, round=round_num):
        validate_and_store(gradient)

# Check results
print(f"Processed: {aggregator.total_count}")
print(f"Succeeded: {aggregator.success_count}")
print(f"Failed: {aggregator.error_count}")
print(f"Failure rate: {aggregator.failure_rate:.1%}")

# Get detailed summary
summary = aggregator.get_summary()
print(f"Error types: {summary['error_types']}")

# Raise if too many failures
aggregator.raise_if_threshold_exceeded()

# Or raise on any critical error
aggregator.raise_if_critical()

# Group errors by type for analysis
errors_by_type = aggregator.get_errors_by_type()
for error_type, errors in errors_by_type.items():
    print(f"{error_type}: {len(errors)} occurrences")
```

### Graceful Degradation

```python
from zerotrustml.error_handling import Fallback

# With a fallback value
@Fallback(fallback_value=0, exceptions=(ConnectionError,))
async def get_credit_balance(node_id):
    return await holochain.get_balance(node_id)

# With a fallback function
cache = {}

@Fallback(
    fallback_func=lambda e: cache.get("last_balance", 0),
    exceptions=(ConnectionError, TimeoutError)
)
async def get_credit_balance(node_id):
    balance = await holochain.get_balance(node_id)
    cache["last_balance"] = balance
    return balance
```

### Rate Limiting

```python
from zerotrustml.error_handling import RateLimiter

# Create a rate limiter: 10 requests/second, burst of 20
limiter = RateLimiter(rate=10, burst=20)

@limiter
async def call_rate_limited_api():
    return await api.call()

# Manual token acquisition
if limiter.acquire(tokens=1, block=False):
    # Proceed with call
    pass
else:
    # Would exceed rate limit
    pass
```

---

## Best Practices

### 1. Always Use Structured Logging in Production

```python
configure_logging(
    level="INFO",
    json_format=True,
    service_name="mycelix-node"
)
```

### 2. Include Correlation IDs for Distributed Tracing

```python
@with_correlation_id()
async def handle_request(request):
    # All downstream logs include correlation_id
    await process(request)
```

### 3. Use Specific Exception Types

```python
# Good: Specific exception
raise GradientPoisoningError(
    message="Invalid gradient",
    node_id=node_id,
    pogq_score=score
)

# Avoid: Generic exception
raise Exception("Gradient validation failed")
```

### 4. Include Context in Exceptions

```python
raise ByzantineDetectionError(
    message="Statistical anomaly detected",
    node_id=node_id,
    detection_method="z_score",
    severity="high",
    evidence={"z_score": z, "threshold": threshold}
)
```

### 5. Handle Recoverable vs Non-Recoverable Errors

```python
try:
    await submit_gradient(gradient)
except MycelixError as e:
    if e.recoverable:
        await asyncio.sleep(e.retry_after)
        await submit_gradient(gradient)  # Retry
    else:
        logger.error(f"Non-recoverable error: {e}")
        raise
```

### 6. Use Circuit Breakers for External Services

```python
# Protect external service calls
db_breaker = CircuitBreaker(name="database", failure_threshold=3)
holochain_breaker = CircuitBreaker(name="holochain", failure_threshold=5)

@db_breaker
async def query_database(query):
    ...

@holochain_breaker
async def call_holochain(zome, fn, payload):
    ...
```

### 7. Aggregate Errors in Batch Operations

```python
aggregator = ErrorAggregator(failure_threshold=0.05)  # 5% max failure

for item in large_batch:
    with aggregator.capture(item_id=item.id):
        process(item)

# Log summary even on success
logger.info("Batch complete", extra=aggregator.get_summary())
aggregator.raise_if_threshold_exceeded()
```

---

## Production Configuration

### Environment Variables

```bash
# Logging
export MYCELIX_LOG_LEVEL=INFO
export MYCELIX_LOG_FORMAT=json
export MYCELIX_LOG_FILE=/var/log/mycelix/app.log

# Service identification
export MYCELIX_SERVICE_NAME=mycelix-coordinator
export MYCELIX_ENVIRONMENT=production
export MYCELIX_VERSION=1.0.0
```

### Configuration from Environment

```python
import os
from zerotrustml.logging import configure_logging

configure_logging(
    level=os.getenv("MYCELIX_LOG_LEVEL", "INFO"),
    json_format=os.getenv("MYCELIX_LOG_FORMAT", "text") == "json",
    log_file=os.getenv("MYCELIX_LOG_FILE"),
    service_name=os.getenv("MYCELIX_SERVICE_NAME"),
    environment=os.getenv("MYCELIX_ENVIRONMENT"),
    version=os.getenv("MYCELIX_VERSION")
)
```

### Log Rotation (systemd example)

```ini
# /etc/logrotate.d/mycelix
/var/log/mycelix/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 mycelix mycelix
    postrotate
        systemctl reload mycelix
    endscript
}
```

### Integration with Observability Stack

```python
# For Prometheus metrics
from zerotrustml.logging import metrics

# Record custom metrics
metrics.increment("gradients_processed", tags={"node": "hospital-a"})
metrics.record_timing("aggregation_duration", elapsed_ms, tags={"algorithm": "krum"})

# Export metrics
prometheus_metrics = metrics.get_metrics()
```

---

## API Reference

For complete API documentation, see the module docstrings:

- `zerotrustml.logging` - Logging configuration and utilities
- `zerotrustml.exceptions` - Exception hierarchy and error codes
- `zerotrustml.error_handling` - Retry, circuit breaker, and error aggregation
