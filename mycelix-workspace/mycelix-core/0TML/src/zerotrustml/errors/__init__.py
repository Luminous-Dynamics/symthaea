# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Unified Error Types for Mycelix Ecosystem

This module provides standardized error categories, severity levels,
and structured error formatting for the entire Mycelix ecosystem.

Error Categories:
- Network: Connection failures, timeouts, protocol errors
- Storage: Database, filesystem, DHT errors
- Validation: Input validation, schema validation
- Crypto: Encryption, signing, verification errors
- Zome: Holochain zome call failures
- Byzantine: Detected attacks, malicious behavior

Usage:
    from zerotrustml.errors import (
        MycelixError,
        NetworkError,
        StorageError,
        ValidationError,
        CryptoError,
        ZomeError,
        ErrorSeverity,
        ErrorContext,
        format_error_json,
    )

    # Raise with context
    raise NetworkError(
        message="Connection to conductor failed",
        context=ErrorContext(
            file=__file__,
            line=42,
            operation="connect_holochain",
            component="HolochainBridge",
        ),
        severity=ErrorSeverity.ERROR,
        recoverable=True,
        retry_after=5.0,
    )
"""

from .categories import (
    ErrorCategory,
    ErrorSeverity,
    ErrorCode,
)
from .context import ErrorContext
from .base import (
    MycelixError,
    NetworkError,
    StorageError,
    ValidationError,
    CryptoError,
    ZomeError,
    ByzantineError,
    GovernanceError,
    IdentityError,
    ConfigurationError,
    AggregationError,
)
from .logging import (
    format_error_json,
    format_error_human,
    log_error,
    ErrorLogger,
)
from .handling import (
    safe_execute,
    collect_errors,
    ErrorCollector,
)

__all__ = [
    # Categories and codes
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorCode",

    # Context
    "ErrorContext",

    # Base exceptions
    "MycelixError",
    "NetworkError",
    "StorageError",
    "ValidationError",
    "CryptoError",
    "ZomeError",
    "ByzantineError",
    "GovernanceError",
    "IdentityError",
    "ConfigurationError",
    "AggregationError",

    # Logging
    "format_error_json",
    "format_error_human",
    "log_error",
    "ErrorLogger",

    # Handling utilities
    "safe_execute",
    "collect_errors",
    "ErrorCollector",
]
