# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
RISC Zero zkSTARK Verifier for Federated Learning

Provides ZK proof verification for gradient validity using RISC Zero:
- Proof deserialization from wire format
- Cryptographic verification against METHOD_ID
- Result extraction and validation
- Metrics tracking

The verifier integrates with gen7-zkstark bindings to verify proofs
that gradients are valid (no NaN/Inf, bounded norm, correct commitment).

Usage:
    from zerotrustml.zk.verifier import RISCZeroVerifier, ZKProofConfig

    # Initialize verifier
    config = ZKProofConfig(verification_timeout=30.0)
    verifier = RISCZeroVerifier(config)

    # Verify a proof
    result = verifier.verify(proof_bytes)
    if result.is_valid:
        print(f"Gradient from node {result.node_id.hex()[:8]} verified")
        print(f"Gradient norm: {result.norm}")
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from zerotrustml.zk.exceptions import (
    ZKVerificationError,
    ZKProofDeserializationError,
    ZKProofTimeoutError,
    ZKProofInvalidError,
    ZKSecurityError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ZKProofConfig:
    """
    Configuration for ZK proof verification.

    Attributes:
        zk_proofs_required: Reject gradients without valid ZK proofs
        zk_proofs_preferred: Weight verified gradients higher in aggregation
        verification_timeout: Maximum time for proof verification (seconds)
        max_proof_size: Maximum proof size in bytes (reject larger)
        enable_parallel_verification: Verify multiple proofs in parallel
        max_parallel_verifications: Maximum concurrent verifications
        cache_verification_results: Cache recent verification results
        cache_ttl_seconds: Time to live for cached results
        production_mode: Enable production security checks (SEC-004)
        min_verification_time_ms: Minimum expected verification time (dev mode detection)
        fail_on_dev_mode: Raise exception if dev mode detected in production
    """

    # ZK mode settings
    zk_proofs_required: bool = False  # Default off for backward compatibility
    zk_proofs_preferred: bool = True  # Weight verified gradients higher

    # Verification settings
    verification_timeout: float = 30.0  # seconds
    max_proof_size: int = 10 * 1024 * 1024  # 10 MB max

    # Performance settings
    enable_parallel_verification: bool = True
    max_parallel_verifications: int = 4

    # Caching (optional)
    cache_verification_results: bool = False
    cache_ttl_seconds: float = 300.0  # 5 minutes

    # SEC-004: Production mode enforcement
    # When production_mode=True:
    # - RISC0_DEV_MODE environment variable must NOT be set
    # - Verification time must meet minimum threshold (dev mode is fast)
    # - Simulation mode is blocked
    production_mode: bool = False  # Set True in production deployments

    # Minimum expected verification time in milliseconds
    # Real RISC Zero proofs take significant time (~10-60 seconds)
    # Dev mode completes in milliseconds
    min_verification_time_ms: float = 1000.0  # 1 second minimum

    # If True, raise ZKSecurityError when dev mode is detected
    # If False, log warning but allow verification to proceed
    fail_on_dev_mode: bool = True


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class ZKVerificationResult:
    """
    Result of ZK proof verification.

    Attributes:
        is_valid: Whether the proof verified successfully
        node_id: 32-byte node identifier from proof
        round_number: Training round from proof
        gradient_hash: SHA256 hash of the gradient
        model_hash: Hash of the model being updated
        gradient_len: Number of gradient elements
        norm: L2 norm of the gradient (float)
        norm_squared: L2 norm squared (fixed-point)
        mean: Mean gradient value (float)
        variance: Gradient variance (float)
        verification_time_ms: Time taken to verify (milliseconds)
        error_message: Error message if verification failed
    """

    is_valid: bool
    node_id: Optional[bytes] = None
    round_number: Optional[int] = None
    gradient_hash: Optional[bytes] = None
    model_hash: Optional[bytes] = None
    gradient_len: Optional[int] = None
    norm: Optional[float] = None
    norm_squared: Optional[int] = None
    mean: Optional[float] = None
    variance: Optional[float] = None
    verification_time_ms: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for logging/serialization."""
        return {
            'is_valid': self.is_valid,
            'node_id': self.node_id.hex() if self.node_id else None,
            'round_number': self.round_number,
            'gradient_hash': self.gradient_hash.hex() if self.gradient_hash else None,
            'model_hash': self.model_hash.hex() if self.model_hash else None,
            'gradient_len': self.gradient_len,
            'norm': self.norm,
            'mean': self.mean,
            'variance': self.variance,
            'verification_time_ms': self.verification_time_ms,
            'error_message': self.error_message,
        }


@dataclass
class ZKGradientProof:
    """
    Wrapper for a ZK gradient proof with metadata.

    Attributes:
        proof_bytes: Serialized RISC Zero receipt
        node_id: Expected node ID (for validation)
        round_number: Expected round number (for validation)
        model_hash: Expected model hash (for validation)
        submitted_at: Timestamp when proof was submitted
    """

    proof_bytes: bytes
    node_id: Optional[bytes] = None
    round_number: Optional[int] = None
    model_hash: Optional[bytes] = None
    submitted_at: Optional[float] = None

    def __post_init__(self):
        if self.submitted_at is None:
            self.submitted_at = time.time()


# =============================================================================
# Verifier Implementation
# =============================================================================

class RISCZeroVerifier:
    """
    RISC Zero zkSTARK proof verifier for gradient validity.

    Integrates with gen7-zkstark Python bindings to verify proofs
    generated by the RISC Zero zkVM.

    Features:
    - Proof deserialization and validation
    - Timeout handling for verification
    - Metrics tracking
    - Optional parallel verification
    - Result caching

    Thread Safety:
    - Verification is thread-safe (uses isolated state)
    - Metrics updates are not atomic (use for monitoring only)
    """

    def __init__(self, config: Optional[ZKProofConfig] = None):
        """
        Initialize the verifier.

        Args:
            config: Verification configuration (uses defaults if None)

        Raises:
            ZKSecurityError: If production_mode=True and RISC0_DEV_MODE is set
        """
        self.config = config or ZKProofConfig()
        self._gen7_zkstark = None
        self._available = False
        self._executor: Optional[ThreadPoolExecutor] = None
        self._dev_mode_detected = False

        # Metrics
        self._metrics = {
            'proofs_verified': 0,
            'proofs_valid': 0,
            'proofs_invalid': 0,
            'proofs_timeout': 0,
            'proofs_deserialization_error': 0,
            'proofs_dev_mode_warning': 0,
            'total_verification_time_ms': 0.0,
            'average_verification_time_ms': 0.0,
        }

        # Result cache (proof_hash -> (result, expiry_time))
        self._cache: Dict[str, Tuple[ZKVerificationResult, float]] = {}

        # SEC-004: Check for dev mode environment variables in production
        self._check_production_environment()

        # Initialize RISC Zero bindings
        self._initialize_bindings()

    def _check_production_environment(self):
        """
        Check environment for production security violations (SEC-004).

        Raises:
            ZKSecurityError: If production_mode=True and dev mode is detected
        """
        dev_mode_vars = {
            'RISC0_DEV_MODE': os.environ.get('RISC0_DEV_MODE'),
            'RISC0_SKIP_VERIFY': os.environ.get('RISC0_SKIP_VERIFY'),
            'ZK_SIMULATION_MODE': os.environ.get('ZK_SIMULATION_MODE'),
        }

        # Check for any dev mode environment variables
        detected_vars = {k: v for k, v in dev_mode_vars.items() if v}

        if detected_vars:
            self._dev_mode_detected = True

            if self.config.production_mode:
                # In production mode, this is a critical security error
                env_var_list = ', '.join(f"{k}={v}" for k, v in detected_vars.items())

                if self.config.fail_on_dev_mode:
                    raise ZKSecurityError(
                        message=(
                            f"SEC-004 CRITICAL: ZK proof dev/simulation mode detected in production. "
                            f"Environment variables set: {env_var_list}. "
                            f"This means proofs are NOT being cryptographically verified. "
                            f"Remove these environment variables before deploying to production."
                        ),
                        violation_type="dev_mode_in_production",
                        environment_vars=detected_vars,
                    )
                else:
                    logger.critical(
                        f"SEC-004 WARNING: ZK proof dev/simulation mode detected in production. "
                        f"Environment variables set: {env_var_list}. "
                        f"Proofs may NOT be cryptographically verified!"
                    )
            else:
                logger.warning(
                    f"ZK dev mode environment variables detected: {detected_vars}. "
                    f"This is acceptable for development but NOT for production."
                )

    def _initialize_bindings(self):
        """Initialize gen7-zkstark Python bindings."""
        try:
            import gen7_zkstark
            self._gen7_zkstark = gen7_zkstark
            self._available = True
            logger.info("RISC Zero zkSTARK verifier initialized (gen7_zkstark available)")
        except ImportError as e:
            self._available = False
            logger.warning(
                f"gen7_zkstark not available: {e}. "
                "ZK proof verification will be disabled. "
                "Install with: pip install gen7_zkstark"
            )

    @property
    def available(self) -> bool:
        """Check if ZK verification is available."""
        return self._available

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get verification metrics."""
        return self._metrics.copy()

    def verify(self, proof: ZKGradientProof) -> ZKVerificationResult:
        """
        Verify a ZK gradient proof synchronously.

        Args:
            proof: ZK proof wrapper with proof bytes and metadata

        Returns:
            ZKVerificationResult with verification status and extracted data

        Raises:
            ZKVerificationError: If verification fails critically
            ZKProofDeserializationError: If proof cannot be deserialized
            ZKProofTimeoutError: If verification exceeds timeout
        """
        start_time = time.monotonic()

        # Validate proof size first (before availability check)
        if len(proof.proof_bytes) > self.config.max_proof_size:
            self._metrics['proofs_deserialization_error'] += 1
            raise ZKProofDeserializationError(
                message=f"Proof size {len(proof.proof_bytes)} exceeds maximum {self.config.max_proof_size}",
                proof_size=len(proof.proof_bytes),
            )

        # Check availability
        if not self._available:
            logger.warning("ZK verification not available, returning invalid result")
            return ZKVerificationResult(
                is_valid=False,
                error_message="ZK verification not available (gen7_zkstark not installed)",
                verification_time_ms=0.0,
            )

        # Check cache
        if self.config.cache_verification_results:
            cached = self._check_cache(proof.proof_bytes)
            if cached is not None:
                logger.debug("Returning cached verification result")
                return cached

        try:
            # Run verification with timeout
            result = self._verify_with_timeout(proof.proof_bytes)

            # Validate against expected values if provided
            if result.is_valid:
                self._validate_proof_metadata(proof, result)

            # Update metrics
            elapsed_ms = (time.monotonic() - start_time) * 1000
            result.verification_time_ms = elapsed_ms
            self._update_metrics(result)

            # SEC-004: Check for suspiciously fast verification (dev mode detection)
            if result.is_valid:
                self._check_verification_timing(elapsed_ms)

            # Cache result
            if self.config.cache_verification_results and result.is_valid:
                self._cache_result(proof.proof_bytes, result)

            return result

        except FuturesTimeoutError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._metrics['proofs_timeout'] += 1
            raise ZKProofTimeoutError(
                message=f"Proof verification timed out after {self.config.verification_timeout}s",
                timeout_seconds=self.config.verification_timeout,
                elapsed_seconds=elapsed_ms / 1000,
            )

    async def verify_async(self, proof: ZKGradientProof) -> ZKVerificationResult:
        """
        Verify a ZK gradient proof asynchronously.

        Runs verification in a thread pool to avoid blocking the event loop.

        Args:
            proof: ZK proof wrapper with proof bytes and metadata

        Returns:
            ZKVerificationResult with verification status and extracted data
        """
        loop = asyncio.get_event_loop()

        # Initialize executor if needed
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_parallel_verifications,
                thread_name_prefix="zk_verifier_"
            )

        # Run verification in thread pool
        return await loop.run_in_executor(self._executor, self.verify, proof)

    async def verify_batch(
        self,
        proofs: List[ZKGradientProof]
    ) -> List[ZKVerificationResult]:
        """
        Verify multiple proofs in parallel.

        Args:
            proofs: List of ZK proofs to verify

        Returns:
            List of verification results (same order as input)
        """
        if not self.config.enable_parallel_verification:
            # Sequential verification
            return [await self.verify_async(p) for p in proofs]

        # Parallel verification
        tasks = [self.verify_async(p) for p in proofs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to invalid results
        processed_results = []
        for r in results:
            if isinstance(r, Exception):
                processed_results.append(ZKVerificationResult(
                    is_valid=False,
                    error_message=str(r),
                ))
            else:
                processed_results.append(r)

        return processed_results

    def _verify_with_timeout(self, proof_bytes: bytes) -> ZKVerificationResult:
        """
        Run verification with timeout protection.

        Args:
            proof_bytes: Serialized proof data

        Returns:
            ZKVerificationResult from verification

        Raises:
            FuturesTimeoutError: If verification exceeds timeout
        """
        # For synchronous verification, we use threading
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._do_verify, proof_bytes)
            return future.result(timeout=self.config.verification_timeout)

    def _do_verify(self, proof_bytes: bytes) -> ZKVerificationResult:
        """
        Perform the actual verification using gen7_zkstark.

        Args:
            proof_bytes: Serialized RISC Zero receipt

        Returns:
            ZKVerificationResult with extracted data
        """
        try:
            # Call the RISC Zero verification
            result_dict = self._gen7_zkstark.verify_gradient_proof(proof_bytes)

            # Extract and convert results
            return ZKVerificationResult(
                is_valid=result_dict.get('is_valid', False),
                node_id=result_dict.get('node_id'),
                round_number=result_dict.get('round_number'),
                gradient_hash=result_dict.get('gradient_hash'),
                model_hash=result_dict.get('model_hash'),
                gradient_len=result_dict.get('gradient_len'),
                norm=result_dict.get('norm'),
                norm_squared=result_dict.get('norm_squared'),
                mean=result_dict.get('mean'),
                variance=result_dict.get('variance'),
            )

        except Exception as e:
            error_str = str(e)

            # Categorize the error
            if "Deserialization" in error_str or "bincode" in error_str.lower():
                self._metrics['proofs_deserialization_error'] += 1
                raise ZKProofDeserializationError(
                    message=f"Failed to deserialize proof: {error_str}",
                    proof_size=len(proof_bytes),
                )

            if "Verification failed" in error_str or "METHOD_ID" in error_str:
                # Verification failure (invalid proof)
                return ZKVerificationResult(
                    is_valid=False,
                    error_message=f"Proof verification failed: {error_str}",
                )

            # Unknown error
            raise ZKVerificationError(
                message=f"Unexpected verification error: {error_str}",
                cause=e,
            )

    def _validate_proof_metadata(
        self,
        proof: ZKGradientProof,
        result: ZKVerificationResult
    ):
        """
        Validate that proof metadata matches expected values.

        Logs warnings if metadata doesn't match but doesn't fail verification
        (the cryptographic proof is still valid).

        Args:
            proof: Original proof with expected metadata
            result: Verification result with actual metadata
        """
        if proof.node_id is not None and result.node_id != proof.node_id:
            logger.warning(
                f"Node ID mismatch: expected {proof.node_id.hex()[:8]}, "
                f"got {result.node_id.hex()[:8] if result.node_id else 'None'}"
            )

        if proof.round_number is not None and result.round_number != proof.round_number:
            logger.warning(
                f"Round number mismatch: expected {proof.round_number}, "
                f"got {result.round_number}"
            )

        if proof.model_hash is not None and result.model_hash != proof.model_hash:
            logger.warning(
                f"Model hash mismatch: expected {proof.model_hash.hex()[:8]}, "
                f"got {result.model_hash.hex()[:8] if result.model_hash else 'None'}"
            )

    def _check_verification_timing(self, elapsed_ms: float):
        """
        Check if verification completed suspiciously fast (SEC-004).

        Real RISC Zero proof verification takes significant time (10-60+ seconds).
        If verification completes in milliseconds, it's likely running in dev/simulation mode.

        Args:
            elapsed_ms: Time taken for verification in milliseconds

        Raises:
            ZKSecurityError: If production_mode=True and verification is too fast
        """
        if elapsed_ms < self.config.min_verification_time_ms:
            self._metrics['proofs_dev_mode_warning'] += 1

            warning_msg = (
                f"SEC-004: Proof verification completed suspiciously fast ({elapsed_ms:.2f}ms). "
                f"Real zkSTARK verification typically takes {self.config.min_verification_time_ms}ms+. "
                f"This suggests RISC0_DEV_MODE or simulation mode may be active."
            )

            if self.config.production_mode:
                if self.config.fail_on_dev_mode:
                    raise ZKSecurityError(
                        message=(
                            f"{warning_msg} "
                            f"In production mode, verification must be cryptographically real. "
                            f"Ensure RISC0_DEV_MODE is not set and genuine proofs are being verified."
                        ),
                        violation_type="suspiciously_fast_verification",
                        environment_vars={
                            'verification_time_ms': elapsed_ms,
                            'min_expected_ms': self.config.min_verification_time_ms,
                        },
                    )
                else:
                    logger.critical(warning_msg)
            else:
                logger.warning(warning_msg)

    def _update_metrics(self, result: ZKVerificationResult):
        """Update verification metrics."""
        self._metrics['proofs_verified'] += 1

        if result.is_valid:
            self._metrics['proofs_valid'] += 1
        else:
            self._metrics['proofs_invalid'] += 1

        if result.verification_time_ms:
            self._metrics['total_verification_time_ms'] += result.verification_time_ms
            self._metrics['average_verification_time_ms'] = (
                self._metrics['total_verification_time_ms'] /
                self._metrics['proofs_verified']
            )

    def _check_cache(self, proof_bytes: bytes) -> Optional[ZKVerificationResult]:
        """Check cache for existing verification result."""
        import hashlib
        proof_hash = hashlib.sha256(proof_bytes).hexdigest()

        if proof_hash in self._cache:
            result, expiry = self._cache[proof_hash]
            if time.time() < expiry:
                return result
            else:
                del self._cache[proof_hash]

        return None

    def _cache_result(self, proof_bytes: bytes, result: ZKVerificationResult):
        """Cache a verification result."""
        import hashlib
        proof_hash = hashlib.sha256(proof_bytes).hexdigest()
        expiry = time.time() + self.config.cache_ttl_seconds
        self._cache[proof_hash] = (result, expiry)

    def shutdown(self):
        """Shutdown the verifier and release resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


# =============================================================================
# Utility Functions
# =============================================================================

def create_verifier(config: Optional[ZKProofConfig] = None) -> RISCZeroVerifier:
    """
    Factory function to create a configured verifier.

    Args:
        config: Verification configuration

    Returns:
        Configured RISCZeroVerifier instance
    """
    return RISCZeroVerifier(config)


def check_gen7_zkstark_available() -> Tuple[bool, str]:
    """
    Check if gen7_zkstark is available and return version info.

    Returns:
        Tuple of (is_available, message)
    """
    try:
        import gen7_zkstark
        version = getattr(gen7_zkstark, '__version__', 'unknown')
        return True, f"gen7_zkstark v{version} available"
    except ImportError as e:
        return False, f"gen7_zkstark not available: {e}"


def preflight_check_production_environment() -> Tuple[bool, List[str]]:
    """
    Pre-flight check for production ZK verification environment (SEC-004).

    This function should be called before deployment to ensure the environment
    is properly configured for production ZK proof verification.

    Returns:
        Tuple of (is_safe, list of issues)
        - is_safe: True if environment is safe for production
        - issues: List of security issues found (empty if safe)

    Usage:
        is_safe, issues = preflight_check_production_environment()
        if not is_safe:
            print("DEPLOYMENT BLOCKED - Security issues:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
    """
    issues = []

    # Check for dev mode environment variables
    dev_mode_vars = {
        'RISC0_DEV_MODE': 'Disables cryptographic verification of RISC Zero proofs',
        'RISC0_SKIP_VERIFY': 'Skips proof verification entirely',
        'ZK_SIMULATION_MODE': 'Runs ZK proofs in simulation mode',
    }

    for var, description in dev_mode_vars.items():
        value = os.environ.get(var)
        if value:
            issues.append(
                f"SEC-004: Environment variable {var}={value} is set. "
                f"{description}. This MUST be unset for production."
            )

    # Check if gen7_zkstark is available
    available, msg = check_gen7_zkstark_available()
    if not available:
        issues.append(
            f"SEC-004: {msg}. "
            "ZK proof verification will be disabled without this library."
        )

    is_safe = len(issues) == 0

    if is_safe:
        logger.info("SEC-004 pre-flight check PASSED: Environment is safe for production ZK verification")
    else:
        for issue in issues:
            logger.critical(issue)

    return is_safe, issues


def verify_proof_verification_is_real(
    verifier: 'RISCZeroVerifier',
    test_timeout_seconds: float = 5.0
) -> Tuple[bool, str]:
    """
    Verify that ZK proof verification is actually happening (SEC-004).

    This function creates a corrupt proof and verifies that it fails.
    If verification passes for a corrupt proof, it indicates dev mode is active.

    Args:
        verifier: The RISCZeroVerifier to test
        test_timeout_seconds: Maximum time to wait for test

    Returns:
        Tuple of (is_real, message)
        - is_real: True if proofs are being cryptographically verified
        - message: Description of result

    Usage:
        is_real, msg = verify_proof_verification_is_real(verifier)
        if not is_real:
            raise SecurityError(f"SEC-004: {msg}")
    """
    if not verifier.available:
        return False, "ZK verification not available (gen7_zkstark not installed)"

    # Create a corrupt/garbage proof
    import secrets
    corrupt_proof_bytes = secrets.token_bytes(1024)

    proof = ZKGradientProof(proof_bytes=corrupt_proof_bytes)

    try:
        start = time.monotonic()
        result = verifier.verify(proof)
        elapsed_ms = (time.monotonic() - start) * 1000

        if result.is_valid:
            # This is BAD - corrupt proof should NEVER pass
            return False, (
                f"CRITICAL: Corrupt proof was accepted as valid! "
                f"This indicates RISC0_DEV_MODE or simulation mode is active. "
                f"Verification time: {elapsed_ms:.2f}ms"
            )
        else:
            # Good - corrupt proof was rejected
            return True, (
                f"Proof verification is working correctly. "
                f"Corrupt proof rejected in {elapsed_ms:.2f}ms"
            )

    except ZKProofDeserializationError:
        # This is expected - corrupt bytes can't be deserialized
        return True, "Proof verification is working (corrupt proof rejected during deserialization)"

    except ZKSecurityError as e:
        # Security check caught an issue
        return False, f"Security check failed: {e}"

    except Exception as e:
        return False, f"Unexpected error during verification test: {e}"
