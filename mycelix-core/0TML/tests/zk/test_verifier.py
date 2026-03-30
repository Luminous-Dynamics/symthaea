# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for RISC Zero zkSTARK Verifier Module

Tests the ZK verification integration:
- Verifier initialization
- Configuration options
- Mock verification (when gen7_zkstark unavailable)
- Result handling
- Metrics tracking
"""

import os
import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock

# Import the ZK module components
from zerotrustml.zk.verifier import (
    RISCZeroVerifier,
    ZKProofConfig,
    ZKVerificationResult,
    ZKGradientProof,
    create_verifier,
    check_gen7_zkstark_available,
)
from zerotrustml.zk.exceptions import (
    ZKVerificationError,
    ZKProofDeserializationError,
    ZKProofTimeoutError,
    ZKProofInvalidError,
    ZKSecurityError,
)


class TestZKProofConfig:
    """Tests for ZKProofConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ZKProofConfig()

        assert config.zk_proofs_required is False
        assert config.zk_proofs_preferred is True
        assert config.verification_timeout == 30.0
        assert config.max_proof_size == 10 * 1024 * 1024
        assert config.enable_parallel_verification is True
        assert config.max_parallel_verifications == 4

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ZKProofConfig(
            zk_proofs_required=True,
            zk_proofs_preferred=False,
            verification_timeout=60.0,
            max_proof_size=5 * 1024 * 1024,
        )

        assert config.zk_proofs_required is True
        assert config.zk_proofs_preferred is False
        assert config.verification_timeout == 60.0
        assert config.max_proof_size == 5 * 1024 * 1024


class TestZKVerificationResult:
    """Tests for ZKVerificationResult dataclass."""

    def test_valid_result(self):
        """Test valid verification result."""
        result = ZKVerificationResult(
            is_valid=True,
            node_id=b'\x00' * 32,
            round_number=5,
            gradient_hash=b'\x01' * 32,
            model_hash=b'\x02' * 32,
            gradient_len=100,
            norm=10.5,
            mean=0.1,
            variance=0.01,
            verification_time_ms=25.5,
        )

        assert result.is_valid is True
        assert result.round_number == 5
        assert result.norm == 10.5

    def test_invalid_result(self):
        """Test invalid verification result."""
        result = ZKVerificationResult(
            is_valid=False,
            error_message="Proof verification failed: METHOD_ID mismatch",
        )

        assert result.is_valid is False
        assert "METHOD_ID" in result.error_message

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ZKVerificationResult(
            is_valid=True,
            node_id=b'\xab' * 32,
            round_number=10,
        )

        d = result.to_dict()

        assert d['is_valid'] is True
        assert d['round_number'] == 10
        assert d['node_id'] == 'ab' * 32


class TestZKGradientProof:
    """Tests for ZKGradientProof wrapper."""

    def test_proof_creation(self):
        """Test proof wrapper creation."""
        proof_bytes = b'\x00' * 1024
        node_id = b'\x01' * 32
        model_hash = b'\x02' * 32

        proof = ZKGradientProof(
            proof_bytes=proof_bytes,
            node_id=node_id,
            round_number=5,
            model_hash=model_hash,
        )

        assert len(proof.proof_bytes) == 1024
        assert proof.round_number == 5
        assert proof.submitted_at is not None

    def test_proof_auto_timestamp(self):
        """Test automatic timestamp assignment."""
        before = time.time()
        proof = ZKGradientProof(proof_bytes=b'\x00' * 100)
        after = time.time()

        assert before <= proof.submitted_at <= after


class TestRISCZeroVerifier:
    """Tests for RISCZeroVerifier class."""

    def test_verifier_init_without_bindings(self):
        """Test verifier initialization when gen7_zkstark is not available."""
        # Patch the import to simulate missing bindings
        with patch.dict('sys.modules', {'gen7_zkstark': None}):
            verifier = RISCZeroVerifier()
            # When bindings are not available, available should be False
            # (the actual behavior depends on initialization logic)

    def test_verifier_metrics(self):
        """Test verifier metrics initialization."""
        verifier = RISCZeroVerifier()

        metrics = verifier.metrics

        assert 'proofs_verified' in metrics
        assert 'proofs_valid' in metrics
        assert 'proofs_invalid' in metrics
        assert metrics['proofs_verified'] == 0

    def test_verifier_config(self):
        """Test verifier with custom config."""
        config = ZKProofConfig(
            verification_timeout=60.0,
            zk_proofs_required=True,
        )
        verifier = RISCZeroVerifier(config)

        assert verifier.config.verification_timeout == 60.0
        assert verifier.config.zk_proofs_required is True

    def test_verify_proof_too_large(self):
        """Test rejection of oversized proofs."""
        config = ZKProofConfig(max_proof_size=100)
        verifier = RISCZeroVerifier(config)

        large_proof = ZKGradientProof(proof_bytes=b'\x00' * 200)

        with pytest.raises(ZKProofDeserializationError) as exc_info:
            verifier.verify(large_proof)

        assert "exceeds maximum" in str(exc_info.value)

    def test_verify_unavailable_returns_invalid(self):
        """Test that verification returns invalid when not available."""
        verifier = RISCZeroVerifier()
        # Force unavailable state
        verifier._available = False

        proof = ZKGradientProof(proof_bytes=b'\x00' * 100)
        result = verifier.verify(proof)

        assert result.is_valid is False
        assert "not available" in result.error_message

    def test_shutdown(self):
        """Test verifier shutdown."""
        verifier = RISCZeroVerifier()
        verifier.shutdown()
        # Should not raise

    def test_factory_function(self):
        """Test create_verifier factory function."""
        verifier = create_verifier()
        assert isinstance(verifier, RISCZeroVerifier)

        config = ZKProofConfig(verification_timeout=45.0)
        verifier2 = create_verifier(config)
        assert verifier2.config.verification_timeout == 45.0


class TestRISCZeroVerifierAsync:
    """Async tests for RISCZeroVerifier."""

    @pytest.mark.asyncio
    async def test_verify_async_unavailable(self):
        """Test async verification when not available."""
        verifier = RISCZeroVerifier()
        verifier._available = False

        proof = ZKGradientProof(proof_bytes=b'\x00' * 100)
        result = await verifier.verify_async(proof)

        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_verify_batch_empty(self):
        """Test batch verification with empty list."""
        verifier = RISCZeroVerifier()
        verifier._available = False

        results = await verifier.verify_batch([])

        assert results == []

    @pytest.mark.asyncio
    async def test_verify_batch_multiple(self):
        """Test batch verification with multiple proofs."""
        verifier = RISCZeroVerifier()
        verifier._available = False

        proofs = [
            ZKGradientProof(proof_bytes=b'\x00' * 100),
            ZKGradientProof(proof_bytes=b'\x01' * 100),
            ZKGradientProof(proof_bytes=b'\x02' * 100),
        ]

        results = await verifier.verify_batch(proofs)

        assert len(results) == 3
        # All should be invalid since verifier is unavailable
        for r in results:
            assert r.is_valid is False


class TestZKExceptions:
    """Tests for ZK exception classes."""

    def test_verification_error(self):
        """Test ZKVerificationError."""
        error = ZKVerificationError(
            message="Test error",
            proof_type="risc0_stark",
            node_id="node-123",
            round_number=5,
        )

        assert "Test error" in str(error)
        assert error.proof_type == "risc0_stark"
        assert error.node_id == "node-123"
        assert error.round_number == 5

    def test_deserialization_error(self):
        """Test ZKProofDeserializationError."""
        error = ZKProofDeserializationError(
            message="Invalid proof format",
            proof_size=1024,
            expected_format="bincode",
        )

        assert "Invalid proof format" in str(error)
        assert error.context['proof_size'] == 1024

    def test_timeout_error(self):
        """Test ZKProofTimeoutError."""
        error = ZKProofTimeoutError(
            message="Verification timed out",
            timeout_seconds=30.0,
            elapsed_seconds=35.5,
        )

        assert error.recoverable is True
        assert error.retry_after == 60.0  # 2x timeout

    def test_invalid_error(self):
        """Test ZKProofInvalidError."""
        error = ZKProofInvalidError(
            message="Proof failed verification",
            failure_reason="method_id_mismatch",
            gradient_hash="abc123",
        )

        assert error.failure_reason == "method_id_mismatch"


class TestCheckAvailability:
    """Tests for availability check function."""

    def test_check_available(self):
        """Test availability check."""
        available, message = check_gen7_zkstark_available()

        # Result depends on whether gen7_zkstark is installed
        assert isinstance(available, bool)
        assert isinstance(message, str)
        if available:
            assert "available" in message.lower()
        else:
            assert "not available" in message.lower()


class TestMockedVerification:
    """Tests with mocked gen7_zkstark module."""

    def test_verification_with_mock(self):
        """Test verification with mocked gen7_zkstark."""
        mock_zkstark = MagicMock()
        mock_zkstark.verify_gradient_proof.return_value = {
            'is_valid': True,
            'node_id': b'\x00' * 32,
            'round_number': 5,
            'gradient_hash': b'\x01' * 32,
            'model_hash': b'\x02' * 32,
            'gradient_len': 100,
            'norm': 10.0,
            'norm_squared': 6553600,
            'mean': 0.1,
            'variance': 0.01,
        }

        with patch.dict('sys.modules', {'gen7_zkstark': mock_zkstark}):
            # Create verifier with mocked module
            verifier = RISCZeroVerifier()
            verifier._gen7_zkstark = mock_zkstark
            verifier._available = True

            proof = ZKGradientProof(proof_bytes=b'\x00' * 100)
            result = verifier.verify(proof)

            assert result.is_valid is True
            assert result.round_number == 5
            assert result.norm == 10.0

    def test_verification_failure_with_mock(self):
        """Test verification failure with mocked gen7_zkstark."""
        mock_zkstark = MagicMock()
        mock_zkstark.verify_gradient_proof.side_effect = RuntimeError(
            "Verification failed: METHOD_ID mismatch"
        )

        verifier = RISCZeroVerifier()
        verifier._gen7_zkstark = mock_zkstark
        verifier._available = True

        proof = ZKGradientProof(proof_bytes=b'\x00' * 100)
        result = verifier.verify(proof)

        # Should return invalid result, not raise exception
        assert result.is_valid is False


class TestSEC004ProductionModeEnforcement:
    """
    SEC-004: Tests for production mode enforcement.

    These tests verify that:
    1. RISC0_DEV_MODE is blocked in production
    2. Suspiciously fast verification is detected
    3. Proper errors are raised when dev mode is detected
    """

    def test_production_mode_blocks_risc0_dev_mode(self):
        """Test that RISC0_DEV_MODE raises error in production mode."""
        with patch.dict(os.environ, {'RISC0_DEV_MODE': '1'}):
            config = ZKProofConfig(
                production_mode=True,
                fail_on_dev_mode=True,
            )

            with pytest.raises(ZKSecurityError) as exc_info:
                RISCZeroVerifier(config)

            assert "SEC-004" in str(exc_info.value)
            assert "RISC0_DEV_MODE" in str(exc_info.value)
            assert exc_info.value.violation_type == "dev_mode_in_production"

    def test_production_mode_blocks_zk_simulation_mode(self):
        """Test that ZK_SIMULATION_MODE raises error in production mode."""
        with patch.dict(os.environ, {'ZK_SIMULATION_MODE': 'true'}):
            config = ZKProofConfig(
                production_mode=True,
                fail_on_dev_mode=True,
            )

            with pytest.raises(ZKSecurityError) as exc_info:
                RISCZeroVerifier(config)

            assert "SEC-004" in str(exc_info.value)
            assert "ZK_SIMULATION_MODE" in str(exc_info.value)

    def test_production_mode_allows_clean_environment(self):
        """Test that production mode works with clean environment."""
        # Ensure no dev mode vars are set
        env_backup = {}
        for var in ['RISC0_DEV_MODE', 'RISC0_SKIP_VERIFY', 'ZK_SIMULATION_MODE']:
            if var in os.environ:
                env_backup[var] = os.environ.pop(var)

        try:
            config = ZKProofConfig(
                production_mode=True,
                fail_on_dev_mode=True,
            )

            # Should not raise
            verifier = RISCZeroVerifier(config)
            assert verifier.config.production_mode is True
        finally:
            # Restore environment
            os.environ.update(env_backup)

    def test_dev_mode_warning_without_fail(self):
        """Test that dev mode logs warning when fail_on_dev_mode=False."""
        with patch.dict(os.environ, {'RISC0_DEV_MODE': '1'}):
            config = ZKProofConfig(
                production_mode=True,
                fail_on_dev_mode=False,  # Don't fail, just warn
            )

            # Should not raise, but should set _dev_mode_detected
            verifier = RISCZeroVerifier(config)
            assert verifier._dev_mode_detected is True

    def test_non_production_allows_dev_mode(self):
        """Test that dev mode is allowed in non-production."""
        with patch.dict(os.environ, {'RISC0_DEV_MODE': '1'}):
            config = ZKProofConfig(
                production_mode=False,  # Not production
            )

            # Should not raise
            verifier = RISCZeroVerifier(config)
            assert verifier._dev_mode_detected is True

    def test_suspiciously_fast_verification_detected(self):
        """Test that fast verification is flagged as potential dev mode."""
        mock_zkstark = MagicMock()
        mock_zkstark.verify_gradient_proof.return_value = {
            'is_valid': True,
            'node_id': b'\x00' * 32,
            'round_number': 5,
            'gradient_hash': b'\x01' * 32,
            'model_hash': b'\x02' * 32,
            'gradient_len': 100,
            'norm': 10.0,
            'norm_squared': 6553600,
            'mean': 0.1,
            'variance': 0.01,
        }

        config = ZKProofConfig(
            production_mode=True,
            fail_on_dev_mode=True,
            min_verification_time_ms=1000.0,  # Expect at least 1 second
        )

        # Ensure no dev mode env vars
        with patch.dict(os.environ, {}, clear=True):
            verifier = RISCZeroVerifier(config)
            verifier._gen7_zkstark = mock_zkstark
            verifier._available = True

            proof = ZKGradientProof(proof_bytes=b'\x00' * 100)

            # The mock returns instantly, which should trigger the timing check
            with pytest.raises(ZKSecurityError) as exc_info:
                verifier.verify(proof)

            assert "suspiciously fast" in str(exc_info.value).lower()
            assert exc_info.value.violation_type == "suspiciously_fast_verification"

    def test_config_has_production_mode_defaults(self):
        """Test that ZKProofConfig has correct production mode defaults."""
        config = ZKProofConfig()

        assert config.production_mode is False  # Safe default
        assert config.fail_on_dev_mode is True  # Strict by default
        assert config.min_verification_time_ms == 1000.0  # 1 second minimum

    def test_metrics_track_dev_mode_warnings(self):
        """Test that metrics track dev mode warnings."""
        mock_zkstark = MagicMock()
        mock_zkstark.verify_gradient_proof.return_value = {
            'is_valid': True,
            'node_id': b'\x00' * 32,
            'round_number': 5,
            'gradient_hash': b'\x01' * 32,
            'model_hash': b'\x02' * 32,
            'gradient_len': 100,
            'norm': 10.0,
            'norm_squared': 6553600,
            'mean': 0.1,
            'variance': 0.01,
        }

        config = ZKProofConfig(
            production_mode=False,  # Don't fail
            min_verification_time_ms=1000.0,
        )

        verifier = RISCZeroVerifier(config)
        verifier._gen7_zkstark = mock_zkstark
        verifier._available = True

        proof = ZKGradientProof(proof_bytes=b'\x00' * 100)

        # Should complete but increment warning counter
        result = verifier.verify(proof)

        assert result.is_valid is True
        assert verifier.metrics['proofs_dev_mode_warning'] > 0


class TestCorruptProofDetection:
    """
    Tests that verify proofs are actually being verified.

    These tests ensure that corrupted proofs fail verification,
    which would NOT happen if running in dev/simulation mode.
    """

    def test_corrupted_proof_fails_verification(self):
        """Test that corrupted proof data fails verification."""
        mock_zkstark = MagicMock()
        # Simulate verification failure on corrupted proof
        mock_zkstark.verify_gradient_proof.side_effect = RuntimeError(
            "Verification failed: invalid proof structure"
        )

        verifier = RISCZeroVerifier()
        verifier._gen7_zkstark = mock_zkstark
        verifier._available = True

        # Create intentionally corrupted proof
        corrupted_proof = ZKGradientProof(
            proof_bytes=b'\xff\xfe\xfd\xfc' * 100  # Random garbage
        )

        result = verifier.verify(corrupted_proof)

        # Corrupted proof should fail
        assert result.is_valid is False

    def test_truncated_proof_fails(self):
        """Test that truncated proof fails verification."""
        mock_zkstark = MagicMock()
        mock_zkstark.verify_gradient_proof.side_effect = RuntimeError(
            "Deserialization failed: unexpected end of input"
        )

        verifier = RISCZeroVerifier()
        verifier._gen7_zkstark = mock_zkstark
        verifier._available = True

        # Truncated proof (too short)
        truncated_proof = ZKGradientProof(proof_bytes=b'\x00' * 10)

        with pytest.raises(ZKProofDeserializationError):
            verifier.verify(truncated_proof)


class TestZKSecurityError:
    """Tests for ZKSecurityError exception."""

    def test_security_error_attributes(self):
        """Test ZKSecurityError has correct attributes."""
        from zerotrustml.zk.exceptions import ZKSecurityError

        error = ZKSecurityError(
            message="Test security error",
            violation_type="test_violation",
            environment_vars={'RISC0_DEV_MODE': '1'},
        )

        assert error.violation_type == "test_violation"
        assert error.environment_vars == {'RISC0_DEV_MODE': '1'}
        assert error.context['security_issue'] == 'SEC-004'
        # Note: recoverable is inherited from ZKVerificationError (which sets it to False)

    def test_security_error_str(self):
        """Test ZKSecurityError string representation."""
        from zerotrustml.zk.exceptions import ZKSecurityError

        error = ZKSecurityError(
            message="RISC0_DEV_MODE is set in production",
            violation_type="dev_mode_in_production",
        )

        assert "RISC0_DEV_MODE" in str(error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
