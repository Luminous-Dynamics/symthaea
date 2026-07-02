# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZK Proof Verification Exceptions

Provides structured exceptions for ZK proof operations:
- Deserialization failures
- Verification timeouts
- Invalid proof detection
- Method ID mismatches
"""

from typing import Any, Dict, Optional
from zerotrustml.exceptions import MycelixError, ErrorCode


class ZKVerificationError(MycelixError):
    """
    Base exception for ZK proof verification errors.

    Attributes:
        proof_type: Type of ZK proof (e.g., "risc0_stark", "bulletproof")
        node_id: Node that submitted the proof (if known)
        round_number: Training round (if known)
    """

    def __init__(
        self,
        message: str,
        proof_type: str = "risc0_stark",
        node_id: Optional[str] = None,
        round_number: Optional[int] = None,
        code: ErrorCode = ErrorCode.PROOF_VERIFICATION_FAILED,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'proof_type': proof_type,
            'node_id': node_id,
            'round_number': round_number,
        })
        # Pop recoverable from kwargs if present to allow subclasses to override
        recoverable = kwargs.pop('recoverable', False)
        super().__init__(
            message=message,
            code=code,
            context=context,
            recoverable=recoverable,
            **kwargs
        )
        self.proof_type = proof_type
        self.node_id = node_id
        self.round_number = round_number


class ZKProofDeserializationError(ZKVerificationError):
    """
    Raised when proof bytes cannot be deserialized.

    This typically indicates:
    - Corrupted proof data
    - Incompatible proof version
    - Truncated transmission
    """

    def __init__(
        self,
        message: str,
        proof_size: Optional[int] = None,
        expected_format: str = "bincode",
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'proof_size': proof_size,
            'expected_format': expected_format,
        })
        super().__init__(
            message=message,
            code=ErrorCode.PROOF_INVALID_FORMAT,
            context=context,
            **kwargs
        )


class ZKProofTimeoutError(ZKVerificationError):
    """
    Raised when proof verification exceeds timeout.

    This can happen with:
    - Very large proofs
    - Resource-constrained environments
    - Network delays (for remote verification)
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        elapsed_seconds: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'timeout_seconds': timeout_seconds,
            'elapsed_seconds': elapsed_seconds,
        })
        super().__init__(
            message=message,
            code=ErrorCode.TIMEOUT_ERROR,
            context=context,
            recoverable=True,  # Timeouts can be retried
            retry_after=timeout_seconds * 2,  # Suggest longer timeout
            **kwargs
        )


class ZKProofInvalidError(ZKVerificationError):
    """
    Raised when proof verification fails cryptographically.

    This indicates:
    - Invalid proof (Byzantine gradient)
    - Method ID mismatch (wrong circuit)
    - Tampered proof data
    """

    def __init__(
        self,
        message: str,
        failure_reason: str = "unknown",
        gradient_hash: Optional[str] = None,
        model_hash: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'failure_reason': failure_reason,
            'gradient_hash': gradient_hash,
            'model_hash': model_hash,
        })
        super().__init__(
            message=message,
            code=ErrorCode.ZKPOC_VERIFICATION_FAILED,
            context=context,
            **kwargs
        )
        self.failure_reason = failure_reason


class ZKSecurityError(ZKVerificationError):
    """
    Raised when a security violation is detected in ZK proof operations.

    SEC-004: This exception is raised when:
    - RISC0_DEV_MODE is set in production
    - Simulation mode is detected in production
    - Verification is skipped when production_mode=True
    - Proof verification is suspiciously fast (potential dev mode)

    This is a CRITICAL security issue that should block deployment.
    """

    def __init__(
        self,
        message: str,
        violation_type: str = "unknown",
        environment_vars: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            'violation_type': violation_type,
            'environment_vars': environment_vars or {},
            'security_issue': 'SEC-004',
        })
        # Use ZKPOC_VERIFICATION_FAILED as closest code
        # Note: recoverable=False is inherited from ZKVerificationError
        super().__init__(
            message=message,
            code=ErrorCode.ZKPOC_VERIFICATION_FAILED,
            context=context,
            **kwargs
        )
        self.violation_type = violation_type
        self.environment_vars = environment_vars or {}
