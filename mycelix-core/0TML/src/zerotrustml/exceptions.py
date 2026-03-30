# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Custom Exception Hierarchy for Mycelix/ZeroTrustML

Provides structured, descriptive exceptions for all system components:
- Byzantine detection errors
- Holochain connection errors
- Proof verification errors
- Configuration errors
- Identity errors
- Governance errors

Each exception includes:
- Error code for programmatic handling
- Detailed message for debugging
- Optional context data
- Cause chaining support

Usage:
    from zerotrustml.exceptions import ByzantineDetectionError, MycelixError

    try:
        validate_gradient(gradient)
    except ByzantineDetectionError as e:
        logger.error(f"Byzantine attack detected: {e.code} - {e.message}")
        # Handle the specific error
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# Error Codes
# ============================================================

class ErrorCode(Enum):
    """
    Standardized error codes for Mycelix system.

    Format: CATEGORY_SPECIFIC_ERROR
    """
    # General errors (1xxx)
    UNKNOWN_ERROR = "ERR_1000"
    INTERNAL_ERROR = "ERR_1001"
    TIMEOUT_ERROR = "ERR_1002"
    VALIDATION_ERROR = "ERR_1003"

    # Configuration errors (2xxx)
    CONFIG_INVALID = "ERR_2000"
    CONFIG_MISSING = "ERR_2001"
    CONFIG_TYPE_ERROR = "ERR_2002"
    CONFIG_CONSTRAINT_VIOLATION = "ERR_2003"

    # Connection errors (3xxx)
    CONNECTION_FAILED = "ERR_3000"
    CONNECTION_TIMEOUT = "ERR_3001"
    CONNECTION_REFUSED = "ERR_3002"
    CONNECTION_LOST = "ERR_3003"
    HOLOCHAIN_CONNECTION_FAILED = "ERR_3100"
    HOLOCHAIN_CONDUCTOR_UNAVAILABLE = "ERR_3101"
    HOLOCHAIN_ZOME_CALL_FAILED = "ERR_3102"
    POSTGRES_CONNECTION_FAILED = "ERR_3200"
    WEBSOCKET_ERROR = "ERR_3300"

    # Byzantine detection errors (4xxx)
    BYZANTINE_DETECTED = "ERR_4000"
    BYZANTINE_GRADIENT_POISONING = "ERR_4001"
    BYZANTINE_MODEL_CORRUPTION = "ERR_4002"
    BYZANTINE_SYBIL_ATTACK = "ERR_4003"
    BYZANTINE_FREE_RIDER = "ERR_4004"
    BYZANTINE_COLLUSION = "ERR_4005"
    POGQ_BELOW_THRESHOLD = "ERR_4100"
    REPUTATION_TOO_LOW = "ERR_4101"

    # Proof verification errors (5xxx)
    PROOF_VERIFICATION_FAILED = "ERR_5000"
    PROOF_INVALID_FORMAT = "ERR_5001"
    PROOF_SIGNATURE_INVALID = "ERR_5002"
    PROOF_EXPIRED = "ERR_5003"
    ZKPOC_VERIFICATION_FAILED = "ERR_5100"
    ZKPOC_RANGE_PROOF_INVALID = "ERR_5101"
    ZKPOC_COMMITMENT_MISMATCH = "ERR_5102"

    # Identity errors (6xxx)
    IDENTITY_ERROR = "ERR_6000"
    IDENTITY_NOT_FOUND = "ERR_6001"
    IDENTITY_VERIFICATION_FAILED = "ERR_6002"
    IDENTITY_EXPIRED = "ERR_6003"
    DID_RESOLUTION_FAILED = "ERR_6100"
    DID_INVALID_FORMAT = "ERR_6101"
    CREDENTIAL_INVALID = "ERR_6200"
    CREDENTIAL_EXPIRED = "ERR_6201"
    CREDENTIAL_REVOKED = "ERR_6202"
    ASSURANCE_LEVEL_INSUFFICIENT = "ERR_6300"

    # Governance errors (7xxx)
    GOVERNANCE_ERROR = "ERR_7000"
    GOVERNANCE_UNAUTHORIZED = "ERR_7001"
    GOVERNANCE_PROPOSAL_NOT_FOUND = "ERR_7002"
    GOVERNANCE_VOTE_FAILED = "ERR_7003"
    GOVERNANCE_QUORUM_NOT_MET = "ERR_7004"
    GOVERNANCE_VOTING_CLOSED = "ERR_7005"
    CAPABILITY_DENIED = "ERR_7100"
    CAPABILITY_NOT_FOUND = "ERR_7101"
    CAPABILITY_RATE_LIMITED = "ERR_7102"
    GUARDIAN_AUTHORIZATION_REQUIRED = "ERR_7200"
    GUARDIAN_AUTHORIZATION_EXPIRED = "ERR_7201"

    # Storage errors (8xxx)
    STORAGE_ERROR = "ERR_8000"
    STORAGE_READ_FAILED = "ERR_8001"
    STORAGE_WRITE_FAILED = "ERR_8002"
    STORAGE_NOT_FOUND = "ERR_8003"
    STORAGE_INTEGRITY_ERROR = "ERR_8004"
    GRADIENT_NOT_FOUND = "ERR_8100"
    GRADIENT_CORRUPTED = "ERR_8101"

    # Aggregation errors (9xxx)
    AGGREGATION_ERROR = "ERR_9000"
    AGGREGATION_INSUFFICIENT_GRADIENTS = "ERR_9001"
    AGGREGATION_ALGORITHM_FAILED = "ERR_9002"
    AGGREGATION_QUORUM_NOT_MET = "ERR_9003"

    # Encryption errors (10xxx)
    ENCRYPTION_ERROR = "ERR_10000"
    ENCRYPTION_KEY_MISSING = "ERR_10001"
    ENCRYPTION_KEY_INVALID = "ERR_10002"
    DECRYPTION_FAILED = "ERR_10003"


# ============================================================
# Base Exception
# ============================================================

class MycelixError(Exception):
    """
    Base exception for all Mycelix/ZeroTrustML errors.

    Attributes:
        code: Standardized error code
        message: Human-readable error message
        context: Additional context data
        cause: Original exception (if any)
        recoverable: Whether the error is recoverable
        retry_after: Suggested retry delay in seconds (if applicable)

    Example:
        try:
            process_gradient(gradient)
        except MycelixError as e:
            logger.error(f"Error {e.code}: {e.message}", extra=e.context)
            if e.recoverable:
                await asyncio.sleep(e.retry_after)
                # Retry
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = False,
        retry_after: Optional[float] = None
    ):
        self.message = message
        self.code = code
        self.context = context or {}
        self.cause = cause
        self.recoverable = recoverable
        self.retry_after = retry_after

        # Build full message
        full_message = f"[{code.value}] {message}"
        if cause:
            full_message += f" (caused by: {type(cause).__name__}: {cause})"

        super().__init__(full_message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            'code': self.code.value,
            'error_type': type(self).__name__,
            'message': self.message,
            'context': self.context,
            'recoverable': self.recoverable,
            'retry_after': self.retry_after,
            'cause': str(self.cause) if self.cause else None
        }

    def with_context(self, **extra_context) -> 'MycelixError':
        """Return a new exception with additional context."""
        new_context = {**self.context, **extra_context}
        return type(self)(
            message=self.message,
            code=self.code,
            context=new_context,
            cause=self.cause,
            recoverable=self.recoverable,
            retry_after=self.retry_after
        )


# ============================================================
# Byzantine Detection Errors
# ============================================================

class ByzantineDetectionError(MycelixError):
    """
    Raised when Byzantine behavior is detected.

    Covers:
    - Gradient poisoning attacks
    - Model corruption attempts
    - Sybil attacks
    - Free-rider detection
    - Collusion detection

    Attributes:
        node_id: ID of the Byzantine node
        detection_method: How the attack was detected
        severity: Attack severity (low, medium, high, critical)
        evidence: Evidence of the attack
    """

    def __init__(
        self,
        message: str,
        node_id: Optional[str] = None,
        detection_method: Optional[str] = None,
        severity: str = "medium",
        evidence: Optional[Dict[str, Any]] = None,
        code: ErrorCode = ErrorCode.BYZANTINE_DETECTED,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'node_id': node_id,
            'detection_method': detection_method,
            'severity': severity,
            'evidence': evidence
        })
        super().__init__(
            message=message,
            code=code,
            context=context,
            recoverable=False,  # Byzantine attacks are not recoverable
            **kwargs
        )
        self.node_id = node_id
        self.detection_method = detection_method
        self.severity = severity
        self.evidence = evidence


class GradientPoisoningError(ByzantineDetectionError):
    """Raised when gradient poisoning attack is detected."""

    def __init__(
        self,
        message: str,
        node_id: Optional[str] = None,
        gradient_hash: Optional[str] = None,
        pogq_score: Optional[float] = None,
        **kwargs
    ):
        evidence = kwargs.pop('evidence', {})
        evidence.update({
            'gradient_hash': gradient_hash,
            'pogq_score': pogq_score
        })
        super().__init__(
            message=message,
            node_id=node_id,
            detection_method="gradient_validation",
            code=ErrorCode.BYZANTINE_GRADIENT_POISONING,
            evidence=evidence,
            **kwargs
        )


class SybilAttackError(ByzantineDetectionError):
    """Raised when a Sybil attack is detected."""

    def __init__(
        self,
        message: str,
        suspected_nodes: Optional[list] = None,
        **kwargs
    ):
        evidence = kwargs.pop('evidence', {})
        evidence['suspected_nodes'] = suspected_nodes
        super().__init__(
            message=message,
            detection_method="sybil_detection",
            code=ErrorCode.BYZANTINE_SYBIL_ATTACK,
            severity="critical",
            evidence=evidence,
            **kwargs
        )


# ============================================================
# Connection Errors
# ============================================================

class ConnectionError(MycelixError):
    """Base class for connection errors."""

    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        code: ErrorCode = ErrorCode.CONNECTION_FAILED,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'host': host,
            'port': port
        })
        # Connection errors are usually recoverable
        kwargs.setdefault('recoverable', True)
        kwargs.setdefault('retry_after', 5.0)
        super().__init__(message=message, code=code, context=context, **kwargs)
        self.host = host
        self.port = port


class HolochainConnectionError(ConnectionError):
    """
    Raised when Holochain conductor connection fails.

    Covers:
    - Conductor not running
    - WebSocket connection failures
    - Admin/App interface issues
    - Cell ID resolution failures
    """

    def __init__(
        self,
        message: str,
        admin_url: Optional[str] = None,
        app_url: Optional[str] = None,
        cell_id: Optional[str] = None,
        code: ErrorCode = ErrorCode.HOLOCHAIN_CONNECTION_FAILED,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'admin_url': admin_url,
            'app_url': app_url,
            'cell_id': cell_id
        })
        super().__init__(message=message, code=code, context=context, **kwargs)
        self.admin_url = admin_url
        self.app_url = app_url
        self.cell_id = cell_id


class HolochainZomeError(HolochainConnectionError):
    """Raised when a Holochain zome call fails."""

    def __init__(
        self,
        message: str,
        zome_name: Optional[str] = None,
        fn_name: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'zome_name': zome_name,
            'fn_name': fn_name
        })
        super().__init__(
            message=message,
            code=ErrorCode.HOLOCHAIN_ZOME_CALL_FAILED,
            context=context,
            **kwargs
        )
        self.zome_name = zome_name
        self.fn_name = fn_name


class PostgresConnectionError(ConnectionError):
    """Raised when PostgreSQL connection fails."""

    def __init__(
        self,
        message: str,
        database: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context['database'] = database
        super().__init__(
            message=message,
            code=ErrorCode.POSTGRES_CONNECTION_FAILED,
            context=context,
            **kwargs
        )


# ============================================================
# Proof Verification Errors
# ============================================================

class ProofVerificationError(MycelixError):
    """
    Raised when proof verification fails.

    Covers:
    - ZK-PoC range proof failures
    - Signature verification failures
    - Commitment mismatches
    - Expired proofs
    """

    def __init__(
        self,
        message: str,
        proof_type: Optional[str] = None,
        verifier: Optional[str] = None,
        code: ErrorCode = ErrorCode.PROOF_VERIFICATION_FAILED,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'proof_type': proof_type,
            'verifier': verifier
        })
        super().__init__(message=message, code=code, context=context, **kwargs)
        self.proof_type = proof_type
        self.verifier = verifier


class ZKPoCVerificationError(ProofVerificationError):
    """Raised when ZK-PoC verification fails."""

    def __init__(
        self,
        message: str,
        node_id: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'node_id': node_id,
            'threshold': threshold
        })
        super().__init__(
            message=message,
            proof_type="zkpoc",
            code=ErrorCode.ZKPOC_VERIFICATION_FAILED,
            context=context,
            **kwargs
        )


# ============================================================
# Configuration Errors
# ============================================================

class ConfigurationError(MycelixError):
    """
    Raised when configuration is invalid or missing.

    Covers:
    - Missing required configuration
    - Invalid configuration values
    - Type mismatches
    - Constraint violations
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
        code: ErrorCode = ErrorCode.CONFIG_INVALID,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'config_key': config_key,
            'expected_type': expected_type,
            'actual_value': str(actual_value) if actual_value is not None else None
        })
        super().__init__(
            message=message,
            code=code,
            context=context,
            recoverable=False,
            **kwargs
        )
        self.config_key = config_key
        self.expected_type = expected_type
        self.actual_value = actual_value


class ConfigMissingError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str, **kwargs):
        super().__init__(
            message=f"Required configuration missing: {config_key}",
            config_key=config_key,
            code=ErrorCode.CONFIG_MISSING,
            **kwargs
        )


# ============================================================
# Identity Errors
# ============================================================

class IdentityError(MycelixError):
    """
    Raised when identity operations fail.

    Covers:
    - Identity not found
    - Verification failures
    - DID resolution failures
    - Credential issues
    - Assurance level requirements
    """

    def __init__(
        self,
        message: str,
        participant_id: Optional[str] = None,
        did: Optional[str] = None,
        code: ErrorCode = ErrorCode.IDENTITY_ERROR,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'participant_id': participant_id,
            'did': did
        })
        super().__init__(message=message, code=code, context=context, **kwargs)
        self.participant_id = participant_id
        self.did = did


class IdentityNotFoundError(IdentityError):
    """Raised when an identity cannot be found."""

    def __init__(self, participant_id: str, **kwargs):
        super().__init__(
            message=f"Identity not found: {participant_id}",
            participant_id=participant_id,
            code=ErrorCode.IDENTITY_NOT_FOUND,
            **kwargs
        )


class IdentityVerificationError(IdentityError):
    """Raised when identity verification fails."""

    def __init__(
        self,
        message: str,
        participant_id: Optional[str] = None,
        required_assurance: Optional[str] = None,
        actual_assurance: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'required_assurance': required_assurance,
            'actual_assurance': actual_assurance
        })
        super().__init__(
            message=message,
            participant_id=participant_id,
            code=ErrorCode.IDENTITY_VERIFICATION_FAILED,
            context=context,
            **kwargs
        )


class DIDResolutionError(IdentityError):
    """Raised when DID resolution fails."""

    def __init__(self, did: str, **kwargs):
        super().__init__(
            message=f"Failed to resolve DID: {did}",
            did=did,
            code=ErrorCode.DID_RESOLUTION_FAILED,
            **kwargs
        )


# ============================================================
# Governance Errors
# ============================================================

class GovernanceError(MycelixError):
    """
    Raised when governance operations fail.

    Covers:
    - Unauthorized actions
    - Proposal issues
    - Voting failures
    - Capability denials
    - Guardian authorization
    """

    def __init__(
        self,
        message: str,
        participant_id: Optional[str] = None,
        action: Optional[str] = None,
        code: ErrorCode = ErrorCode.GOVERNANCE_ERROR,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'participant_id': participant_id,
            'action': action
        })
        super().__init__(message=message, code=code, context=context, **kwargs)
        self.participant_id = participant_id
        self.action = action


class GovernanceUnauthorizedError(GovernanceError):
    """Raised when a participant is not authorized for an action."""

    def __init__(
        self,
        action: str,
        participant_id: Optional[str] = None,
        required_capability: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context['required_capability'] = required_capability
        super().__init__(
            message=f"Unauthorized: {participant_id} cannot perform {action}",
            participant_id=participant_id,
            action=action,
            code=ErrorCode.GOVERNANCE_UNAUTHORIZED,
            context=context,
            **kwargs
        )


class CapabilityDeniedError(GovernanceError):
    """Raised when a capability check fails."""

    def __init__(
        self,
        capability_id: str,
        participant_id: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'capability_id': capability_id,
            'reason': reason
        })
        super().__init__(
            message=f"Capability denied: {capability_id} for {participant_id}",
            participant_id=participant_id,
            action=capability_id,
            code=ErrorCode.CAPABILITY_DENIED,
            context=context,
            **kwargs
        )


class GuardianAuthorizationError(GovernanceError):
    """Raised when guardian authorization is required but not obtained."""

    def __init__(
        self,
        action: str,
        participant_id: Optional[str] = None,
        threshold: Optional[float] = None,
        current_approvals: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'threshold': threshold,
            'current_approvals': current_approvals
        })
        super().__init__(
            message=f"Guardian authorization required for: {action}",
            participant_id=participant_id,
            action=action,
            code=ErrorCode.GUARDIAN_AUTHORIZATION_REQUIRED,
            context=context,
            recoverable=True,  # Can be recovered by getting approvals
            retry_after=60.0,  # Check again in 1 minute
            **kwargs
        )


# ============================================================
# Storage Errors
# ============================================================

class StorageError(MycelixError):
    """Raised when storage operations fail."""

    def __init__(
        self,
        message: str,
        backend: Optional[str] = None,
        operation: Optional[str] = None,
        code: ErrorCode = ErrorCode.STORAGE_ERROR,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'backend': backend,
            'operation': operation
        })
        super().__init__(message=message, code=code, context=context, **kwargs)
        self.backend = backend
        self.operation = operation


class StorageIntegrityError(StorageError):
    """Raised when data integrity verification fails."""

    def __init__(
        self,
        message: str,
        expected_hash: Optional[str] = None,
        actual_hash: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'expected_hash': expected_hash,
            'actual_hash': actual_hash
        })
        super().__init__(
            message=message,
            code=ErrorCode.STORAGE_INTEGRITY_ERROR,
            context=context,
            **kwargs
        )


class GradientNotFoundError(StorageError):
    """Raised when a gradient is not found."""

    def __init__(self, gradient_id: str, **kwargs):
        context = kwargs.pop('context', {})
        context['gradient_id'] = gradient_id
        super().__init__(
            message=f"Gradient not found: {gradient_id}",
            code=ErrorCode.GRADIENT_NOT_FOUND,
            context=context,
            **kwargs
        )


# ============================================================
# Aggregation Errors
# ============================================================

class AggregationError(MycelixError):
    """Raised when gradient aggregation fails."""

    def __init__(
        self,
        message: str,
        round_num: Optional[int] = None,
        algorithm: Optional[str] = None,
        code: ErrorCode = ErrorCode.AGGREGATION_ERROR,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'round_num': round_num,
            'algorithm': algorithm
        })
        super().__init__(message=message, code=code, context=context, **kwargs)
        self.round_num = round_num
        self.algorithm = algorithm


class InsufficientGradientsError(AggregationError):
    """Raised when there are not enough gradients to aggregate."""

    def __init__(
        self,
        round_num: int,
        available: int,
        required: int,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'available': available,
            'required': required
        })
        super().__init__(
            message=f"Insufficient gradients for round {round_num}: {available}/{required}",
            round_num=round_num,
            code=ErrorCode.AGGREGATION_INSUFFICIENT_GRADIENTS,
            context=context,
            recoverable=True,  # Can be recovered by waiting for more gradients
            retry_after=30.0,
            **kwargs
        )


# ============================================================
# Encryption Errors
# ============================================================

class EncryptionError(MycelixError):
    """Raised when encryption/decryption operations fail."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,  # 'encrypt' or 'decrypt'
        code: ErrorCode = ErrorCode.ENCRYPTION_ERROR,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context['operation'] = operation
        super().__init__(message=message, code=code, context=context, **kwargs)
        self.operation = operation


class EncryptionKeyError(EncryptionError):
    """Raised when encryption key is missing or invalid."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code=ErrorCode.ENCRYPTION_KEY_INVALID,
            **kwargs
        )


class DecryptionError(EncryptionError):
    """Raised when decryption fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            operation='decrypt',
            code=ErrorCode.DECRYPTION_FAILED,
            **kwargs
        )


# ============================================================
# Exception Handler Helper
# ============================================================

def wrap_exception(
    exc: Exception,
    code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    message: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> MycelixError:
    """
    Wrap a generic exception in a MycelixError.

    Args:
        exc: The original exception
        code: Error code to use
        message: Custom message (defaults to str(exc))
        context: Additional context

    Returns:
        MycelixError wrapping the original exception

    Example:
        try:
            some_external_call()
        except SomeLibraryError as e:
            raise wrap_exception(e, ErrorCode.CONNECTION_FAILED, context={'host': 'example.com'})
    """
    return MycelixError(
        message=message or str(exc),
        code=code,
        context=context,
        cause=exc
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Error codes
    'ErrorCode',

    # Base exception
    'MycelixError',

    # Byzantine errors
    'ByzantineDetectionError',
    'GradientPoisoningError',
    'SybilAttackError',

    # Connection errors
    'ConnectionError',
    'HolochainConnectionError',
    'HolochainZomeError',
    'PostgresConnectionError',

    # Proof errors
    'ProofVerificationError',
    'ZKPoCVerificationError',

    # Configuration errors
    'ConfigurationError',
    'ConfigMissingError',

    # Identity errors
    'IdentityError',
    'IdentityNotFoundError',
    'IdentityVerificationError',
    'DIDResolutionError',

    # Governance errors
    'GovernanceError',
    'GovernanceUnauthorizedError',
    'CapabilityDeniedError',
    'GuardianAuthorizationError',

    # Storage errors
    'StorageError',
    'StorageIntegrityError',
    'GradientNotFoundError',

    # Aggregation errors
    'AggregationError',
    'InsufficientGradientsError',

    # Encryption errors
    'EncryptionError',
    'EncryptionKeyError',
    'DecryptionError',

    # Helper
    'wrap_exception',
]
