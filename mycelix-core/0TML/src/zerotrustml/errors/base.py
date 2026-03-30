# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Base Error Classes for Mycelix Ecosystem

Provides a hierarchy of exception classes with:
- Structured error codes
- Severity levels
- Context information
- Recoverability hints
- Serializable format
"""

from typing import Any, Dict, Optional
from .categories import ErrorCategory, ErrorSeverity, ErrorCode
from .context import ErrorContext


class MycelixError(Exception):
    """
    Base exception for all Mycelix errors.

    Features:
    - Standardized error codes
    - Severity levels for alerting
    - Structured context for debugging
    - Recoverability hints for retry logic
    - Serializable format for logging

    Attributes:
        message: Human-readable error message
        code: Standardized error code
        category: Error category (network, storage, etc.)
        severity: Error severity (debug to critical)
        context: Structured context information
        cause: Original exception (if wrapping)
        recoverable: Whether error is recoverable
        retry_after: Suggested retry delay in seconds

    Usage:
        raise MycelixError(
            message="Failed to connect to conductor",
            code=ErrorCode.HOLOCHAIN_CONNECTION_FAILED,
            severity=ErrorSeverity.ERROR,
            recoverable=True,
            retry_after=5.0,
        )
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        category: Optional[ErrorCategory] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = False,
        retry_after: Optional[float] = None,
    ):
        self.message = message
        self.code = code
        self.category = category or code.category
        self.severity = severity
        self.context = context
        self.cause = cause
        self.recoverable = recoverable
        self.retry_after = retry_after

        # Build full message
        full_message = f"[{code.value}] {message}"
        if cause:
            full_message += f" (caused by: {type(cause).__name__}: {cause})"

        super().__init__(full_message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "error_type": type(self).__name__,
            "code": self.code.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "recoverable": self.recoverable,
        }

        if self.retry_after is not None:
            result["retry_after"] = self.retry_after

        if self.context:
            result["context"] = self.context.to_dict()

        if self.cause:
            result["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
            }

        return result

    def with_context(
        self,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        **metadata
    ) -> "MycelixError":
        """Return new error with additional context."""
        new_context = ErrorContext.capture(
            operation=operation,
            component=component,
            metadata=metadata,
            skip_frames=2,
        )

        # Merge with existing context if present
        if self.context:
            new_context = ErrorContext(
                file=new_context.file or self.context.file,
                line=new_context.line or self.context.line,
                function=new_context.function or self.context.function,
                operation=operation or self.context.operation,
                component=component or self.context.component,
                timestamp=self.context.timestamp,
                metadata={**self.context.metadata, **metadata},
                correlation_id=self.context.correlation_id,
                stack_trace=self.context.stack_trace,
            )

        return type(self)(
            message=self.message,
            code=self.code,
            category=self.category,
            severity=self.severity,
            context=new_context,
            cause=self.cause,
            recoverable=self.recoverable,
            retry_after=self.retry_after,
        )


class NetworkError(MycelixError):
    """
    Network-related errors.

    Covers:
    - Connection failures
    - Timeouts
    - Protocol errors
    - DNS resolution failures
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.CONNECTION_FAILED,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs
    ):
        # Add host/port to context
        context = kwargs.pop("context", None)
        if context is None:
            context = ErrorContext.capture(
                metadata={"host": host, "port": port} if host else {},
                skip_frames=2,
            )
        elif host:
            context = context.with_metadata(host=host, port=port)

        # Network errors are usually recoverable
        kwargs.setdefault("recoverable", True)
        kwargs.setdefault("retry_after", 5.0)
        kwargs.setdefault("category", ErrorCategory.NETWORK)

        super().__init__(message=message, code=code, context=context, **kwargs)
        self.host = host
        self.port = port


class StorageError(MycelixError):
    """
    Storage-related errors.

    Covers:
    - Database errors
    - Filesystem errors
    - DHT errors
    - Data integrity failures
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.STORAGE_ERROR,
        backend: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", None)
        if context is None:
            context = ErrorContext.capture(
                operation=operation,
                metadata={"backend": backend} if backend else {},
                skip_frames=2,
            )

        kwargs.setdefault("category", ErrorCategory.STORAGE)

        super().__init__(message=message, code=code, context=context, **kwargs)
        self.backend = backend
        self.storage_operation = operation


class ValidationError(MycelixError):
    """
    Validation errors.

    Covers:
    - Input validation failures
    - Schema validation failures
    - Constraint violations
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.VALIDATION,
        field: Optional[str] = None,
        expected: Optional[str] = None,
        actual: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", None)
        if context is None:
            metadata = {}
            if field:
                metadata["field"] = field
            if expected:
                metadata["expected"] = expected
            if actual:
                metadata["actual"] = str(actual)
            context = ErrorContext.capture(metadata=metadata, skip_frames=2)

        kwargs.setdefault("category", ErrorCategory.VALIDATION)
        kwargs.setdefault("severity", ErrorSeverity.WARNING)

        super().__init__(message=message, code=code, context=context, **kwargs)
        self.field = field
        self.expected = expected
        self.actual = actual


class CryptoError(MycelixError):
    """
    Cryptographic errors.

    Covers:
    - Encryption/decryption failures
    - Signature verification failures
    - Hash mismatches
    - Key errors
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.ENCRYPTION_ERROR,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", None)
        if context is None:
            context = ErrorContext.capture(
                operation=operation,
                skip_frames=2,
            )

        kwargs.setdefault("category", ErrorCategory.CRYPTO)
        kwargs.setdefault("severity", ErrorSeverity.ERROR)

        super().__init__(message=message, code=code, context=context, **kwargs)


class ZomeError(MycelixError):
    """
    Holochain zome call errors.

    Covers:
    - Zome call failures
    - Cell not found
    - App not installed
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.HOLOCHAIN_ZOME_CALL_FAILED,
        zome_name: Optional[str] = None,
        fn_name: Optional[str] = None,
        cell_id: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", None)
        metadata = {}
        if zome_name:
            metadata["zome_name"] = zome_name
        if fn_name:
            metadata["fn_name"] = fn_name
        if cell_id:
            metadata["cell_id"] = cell_id

        if context is None:
            context = ErrorContext.capture(
                operation=f"{zome_name}.{fn_name}" if zome_name and fn_name else None,
                metadata=metadata,
                skip_frames=2,
            )

        kwargs.setdefault("category", ErrorCategory.ZOME)
        kwargs.setdefault("recoverable", True)
        kwargs.setdefault("retry_after", 2.0)

        super().__init__(message=message, code=code, context=context, **kwargs)
        self.zome_name = zome_name
        self.fn_name = fn_name
        self.cell_id = cell_id


class ByzantineError(MycelixError):
    """
    Byzantine behavior detection errors.

    Covers:
    - Gradient poisoning
    - Sybil attacks
    - Collusion
    - Free-riding
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.BYZANTINE_DETECTED,
        node_id: Optional[str] = None,
        detection_method: Optional[str] = None,
        severity_level: str = "medium",
        evidence: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.pop("context", None)
        metadata = {
            "node_id": node_id,
            "detection_method": detection_method,
            "severity_level": severity_level,
        }
        if evidence:
            metadata["evidence"] = evidence

        if context is None:
            context = ErrorContext.capture(metadata=metadata, skip_frames=2)

        kwargs.setdefault("category", ErrorCategory.BYZANTINE)
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        kwargs.setdefault("recoverable", False)

        super().__init__(message=message, code=code, context=context, **kwargs)
        self.node_id = node_id
        self.detection_method = detection_method
        self.attack_severity = severity_level
        self.evidence = evidence


class GovernanceError(MycelixError):
    """
    Governance and authorization errors.

    Covers:
    - Unauthorized actions
    - Capability denials
    - Voting failures
    - Guardian authorization
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.GOVERNANCE_ERROR,
        participant_id: Optional[str] = None,
        action: Optional[str] = None,
        required_capability: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", None)
        metadata = {}
        if participant_id:
            metadata["participant_id"] = participant_id
        if action:
            metadata["action"] = action
        if required_capability:
            metadata["required_capability"] = required_capability

        if context is None:
            context = ErrorContext.capture(
                operation=action,
                metadata=metadata,
                skip_frames=2,
            )

        kwargs.setdefault("category", ErrorCategory.GOVERNANCE)

        super().__init__(message=message, code=code, context=context, **kwargs)
        self.participant_id = participant_id
        self.action = action
        self.required_capability = required_capability


class IdentityError(MycelixError):
    """
    Identity and credential errors.

    Covers:
    - DID resolution failures
    - Credential validation failures
    - Assurance level issues
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.IDENTITY_ERROR,
        participant_id: Optional[str] = None,
        did: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", None)
        metadata = {}
        if participant_id:
            metadata["participant_id"] = participant_id
        if did:
            metadata["did"] = did

        if context is None:
            context = ErrorContext.capture(metadata=metadata, skip_frames=2)

        kwargs.setdefault("category", ErrorCategory.IDENTITY)

        super().__init__(message=message, code=code, context=context, **kwargs)
        self.participant_id = participant_id
        self.did = did


class ConfigurationError(MycelixError):
    """
    Configuration errors.

    Covers:
    - Missing configuration
    - Invalid values
    - Type mismatches
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.CONFIG_INVALID,
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.pop("context", None)
        metadata = {}
        if config_key:
            metadata["config_key"] = config_key
        if expected_type:
            metadata["expected_type"] = expected_type
        if actual_value is not None:
            metadata["actual_value"] = str(actual_value)

        if context is None:
            context = ErrorContext.capture(metadata=metadata, skip_frames=2)

        kwargs.setdefault("category", ErrorCategory.CONFIGURATION)
        kwargs.setdefault("recoverable", False)

        super().__init__(message=message, code=code, context=context, **kwargs)
        self.config_key = config_key
        self.expected_type = expected_type
        self.actual_value = actual_value


class AggregationError(MycelixError):
    """
    Federated learning aggregation errors.

    Covers:
    - Insufficient gradients
    - Algorithm failures
    - Quorum not met
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.AGGREGATION_ERROR,
        round_num: Optional[int] = None,
        algorithm: Optional[str] = None,
        available: Optional[int] = None,
        required: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.pop("context", None)
        metadata = {}
        if round_num is not None:
            metadata["round_num"] = round_num
        if algorithm:
            metadata["algorithm"] = algorithm
        if available is not None:
            metadata["available"] = available
        if required is not None:
            metadata["required"] = required

        if context is None:
            context = ErrorContext.capture(metadata=metadata, skip_frames=2)

        kwargs.setdefault("category", ErrorCategory.AGGREGATION)
        kwargs.setdefault("recoverable", True)
        kwargs.setdefault("retry_after", 30.0)

        super().__init__(message=message, code=code, context=context, **kwargs)
        self.round_num = round_num
        self.algorithm = algorithm
        self.available = available
        self.required = required


def wrap_exception(
    exc: Exception,
    code: ErrorCode = ErrorCode.INTERNAL,
    message: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    **context_metadata
) -> MycelixError:
    """
    Wrap a generic exception in a MycelixError.

    Usage:
        try:
            some_external_call()
        except SomeLibraryError as e:
            raise wrap_exception(
                e,
                code=ErrorCode.CONNECTION_FAILED,
                message="Failed to connect",
                host="localhost"
            )
    """
    context = ErrorContext.capture(
        metadata=context_metadata,
        skip_frames=2,
    )

    return MycelixError(
        message=message or str(exc),
        code=code,
        severity=severity,
        context=context,
        cause=exc,
    )
