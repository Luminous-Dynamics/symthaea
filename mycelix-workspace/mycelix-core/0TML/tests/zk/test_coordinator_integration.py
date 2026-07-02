# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Integration Tests for RISC Zero ZK Proofs in FL Coordinator

Tests the ZK proof integration in Phase10Coordinator:
- Gradient submission with ZK proofs
- Aggregation with weighted ZK-verified gradients
- Configuration options (required vs preferred)
- Metrics tracking
"""

import pytest
import asyncio
import json
import hashlib
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# These imports assume the zerotrustml package is installed or in PYTHONPATH
try:
    from zerotrustml.core.phase10_coordinator import (
        Phase10Coordinator,
        Phase10Config,
    )
    COORDINATOR_AVAILABLE = True
except ImportError:
    COORDINATOR_AVAILABLE = False
    Phase10Coordinator = None
    Phase10Config = None


@pytest.fixture
def mock_backend():
    """Create a mock storage backend."""
    backend = AsyncMock()
    backend.backend_type = Mock()
    backend.backend_type.value = "mock"
    backend.connect = AsyncMock()
    backend.disconnect = AsyncMock()
    backend.store_gradient = AsyncMock(return_value="gradient-id-123")
    backend.issue_credit = AsyncMock()
    backend.get_gradients_by_round = AsyncMock(return_value=[])
    backend.get_reputation = AsyncMock(return_value={"score": 0.9})
    backend.log_byzantine_event = AsyncMock()
    backend.get_stats = AsyncMock(return_value={"gradients": 0})
    return backend


@pytest.fixture
def mock_zk_verifier():
    """Create a mock ZK verifier."""
    from zerotrustml.zk.verifier import ZKVerificationResult

    verifier = Mock()
    verifier.available = True
    verifier.metrics = {
        'proofs_verified': 0,
        'proofs_valid': 0,
        'proofs_invalid': 0,
    }
    verifier.verify_async = AsyncMock(return_value=ZKVerificationResult(
        is_valid=True,
        node_id=b'\x00' * 32,
        round_number=0,
        gradient_hash=b'\x01' * 32,
        model_hash=b'\x02' * 32,
        gradient_len=100,
        norm=10.0,
        verification_time_ms=25.0,
    ))
    verifier.shutdown = Mock()
    return verifier


@pytest.fixture
def coordinator_config():
    """Create a test coordinator configuration."""
    if not COORDINATOR_AVAILABLE:
        pytest.skip("Phase10Coordinator not available")

    return Phase10Config(
        postgres_enabled=False,
        holochain_enabled=False,
        localfile_enabled=False,  # We'll mock the backend
        storage_strategy="primary",
        zkpoc_enabled=False,  # Disable legacy ZK-PoC for these tests
        encryption_enabled=False,  # Simplify tests
        risc0_zk_proofs_required=False,
        risc0_zk_proofs_preferred=True,
        risc0_zk_verification_timeout=30.0,
        risc0_zk_weight_multiplier=1.5,
    )


class TestCoordinatorZKIntegration:
    """Tests for ZK proof integration in Phase10Coordinator."""

    @pytest.mark.skipif(not COORDINATOR_AVAILABLE, reason="Coordinator not available")
    def test_config_has_zk_options(self, coordinator_config):
        """Test that config includes ZK options."""
        assert hasattr(coordinator_config, 'risc0_zk_proofs_required')
        assert hasattr(coordinator_config, 'risc0_zk_proofs_preferred')
        assert hasattr(coordinator_config, 'risc0_zk_verification_timeout')
        assert hasattr(coordinator_config, 'risc0_zk_weight_multiplier')

    @pytest.mark.skipif(not COORDINATOR_AVAILABLE, reason="Coordinator not available")
    def test_coordinator_init_with_zk_config(self, coordinator_config):
        """Test coordinator initialization with ZK config."""
        coordinator = Phase10Coordinator(coordinator_config)

        assert coordinator.config.risc0_zk_proofs_required is False
        assert coordinator.config.risc0_zk_proofs_preferred is True
        assert coordinator.config.risc0_zk_weight_multiplier == 1.5

    @pytest.mark.skipif(not COORDINATOR_AVAILABLE, reason="Coordinator not available")
    def test_coordinator_has_zk_metrics(self, coordinator_config):
        """Test that coordinator tracks ZK metrics."""
        coordinator = Phase10Coordinator(coordinator_config)

        assert 'risc0_proofs_submitted' in coordinator.metrics
        assert 'risc0_proofs_verified' in coordinator.metrics
        assert 'risc0_proofs_invalid' in coordinator.metrics
        assert 'risc0_proofs_timeout' in coordinator.metrics
        assert 'risc0_avg_verification_ms' in coordinator.metrics


@pytest.mark.asyncio
class TestCoordinatorGradientSubmission:
    """Async tests for gradient submission with ZK proofs."""

    @pytest.mark.skipif(not COORDINATOR_AVAILABLE, reason="Coordinator not available")
    async def test_submit_gradient_with_zk_proof(
        self, coordinator_config, mock_backend, mock_zk_verifier
    ):
        """Test gradient submission with valid ZK proof."""
        coordinator = Phase10Coordinator(coordinator_config)
        coordinator.backends = [mock_backend]
        coordinator.risc0_verifier = mock_zk_verifier

        # Create test gradient
        gradient = json.dumps([0.1] * 100).encode('utf-8')
        proof = b'\x00' * 1024  # Mock proof bytes

        result = await coordinator.handle_gradient_submission(
            node_id="test-node-1",
            encrypted_gradient=gradient,
            pogq_score=0.9,  # Required when ZK-PoC disabled
            risc0_proof=proof,
            model_hash=b'\x02' * 32,
        )

        assert result['accepted'] is True
        assert result['risc0_verified'] is True
        assert coordinator.metrics['risc0_proofs_submitted'] == 1
        assert coordinator.metrics['risc0_proofs_verified'] == 1

    @pytest.mark.skipif(not COORDINATOR_AVAILABLE, reason="Coordinator not available")
    async def test_submit_gradient_with_invalid_zk_proof(
        self, coordinator_config, mock_backend, mock_zk_verifier
    ):
        """Test gradient submission with invalid ZK proof (proofs not required)."""
        from zerotrustml.zk.verifier import ZKVerificationResult

        # Make verifier return invalid result
        mock_zk_verifier.verify_async = AsyncMock(return_value=ZKVerificationResult(
            is_valid=False,
            error_message="Verification failed: invalid proof",
        ))

        coordinator = Phase10Coordinator(coordinator_config)
        coordinator.backends = [mock_backend]
        coordinator.risc0_verifier = mock_zk_verifier

        gradient = json.dumps([0.1] * 100).encode('utf-8')
        proof = b'\x00' * 1024

        result = await coordinator.handle_gradient_submission(
            node_id="test-node-1",
            encrypted_gradient=gradient,
            pogq_score=0.9,
            risc0_proof=proof,
        )

        # Should still be accepted (proofs not required)
        assert result['accepted'] is True
        assert result['risc0_verified'] is False
        assert coordinator.metrics['risc0_proofs_invalid'] == 1

    @pytest.mark.skipif(not COORDINATOR_AVAILABLE, reason="Coordinator not available")
    async def test_submit_gradient_invalid_proof_when_required(
        self, coordinator_config, mock_backend, mock_zk_verifier
    ):
        """Test gradient rejection when ZK proof required but invalid."""
        from zerotrustml.zk.verifier import ZKVerificationResult

        # Enable required proofs
        coordinator_config.risc0_zk_proofs_required = True

        # Make verifier return invalid result
        mock_zk_verifier.verify_async = AsyncMock(return_value=ZKVerificationResult(
            is_valid=False,
            error_message="Verification failed: invalid proof",
        ))

        coordinator = Phase10Coordinator(coordinator_config)
        coordinator.backends = [mock_backend]
        coordinator.risc0_verifier = mock_zk_verifier

        gradient = json.dumps([0.1] * 100).encode('utf-8')
        proof = b'\x00' * 1024

        result = await coordinator.handle_gradient_submission(
            node_id="test-node-1",
            encrypted_gradient=gradient,
            pogq_score=0.9,
            risc0_proof=proof,
        )

        # Should be rejected
        assert result['accepted'] is False
        assert "verification failed" in result['reason'].lower()

    @pytest.mark.skipif(not COORDINATOR_AVAILABLE, reason="Coordinator not available")
    async def test_submit_gradient_no_proof_when_required(
        self, coordinator_config, mock_backend, mock_zk_verifier
    ):
        """Test gradient rejection when ZK proof required but not provided."""
        # Enable required proofs
        coordinator_config.risc0_zk_proofs_required = True

        coordinator = Phase10Coordinator(coordinator_config)
        coordinator.backends = [mock_backend]
        coordinator.risc0_verifier = mock_zk_verifier

        gradient = json.dumps([0.1] * 100).encode('utf-8')

        result = await coordinator.handle_gradient_submission(
            node_id="test-node-1",
            encrypted_gradient=gradient,
            pogq_score=0.9,
            # No risc0_proof provided
        )

        # Should be rejected
        assert result['accepted'] is False
        assert "required" in result['reason'].lower()

    @pytest.mark.skipif(not COORDINATOR_AVAILABLE, reason="Coordinator not available")
    async def test_submit_gradient_without_proof_when_not_required(
        self, coordinator_config, mock_backend
    ):
        """Test gradient acceptance without proof when not required."""
        coordinator = Phase10Coordinator(coordinator_config)
        coordinator.backends = [mock_backend]
        coordinator.risc0_verifier = None  # No verifier

        gradient = json.dumps([0.1] * 100).encode('utf-8')

        result = await coordinator.handle_gradient_submission(
            node_id="test-node-1",
            encrypted_gradient=gradient,
            pogq_score=0.9,
        )

        # Should be accepted (no proof required)
        assert result['accepted'] is True
        assert result['risc0_verified'] is False


@pytest.mark.asyncio
class TestCoordinatorAggregation:
    """Tests for aggregation with ZK-verified gradients."""

    @pytest.mark.skipif(not COORDINATOR_AVAILABLE, reason="Coordinator not available")
    async def test_aggregation_weights_verified_gradients(
        self, coordinator_config, mock_backend
    ):
        """Test that ZK-verified gradients receive higher weight."""
        # Create mock gradients with mixed verification status
        mock_gradient_class = Mock()
        mock_gradient_class.node_id = "node-1"
        mock_gradient_class.gradient = [0.1] * 100
        mock_gradient_class.gradient_hash = "hash1"
        mock_gradient_class.pogq_score = 0.9
        mock_gradient_class.zkpoc_verified = False
        mock_gradient_class.risc0_verified = True

        mock_gradient_class2 = Mock()
        mock_gradient_class2.node_id = "node-2"
        mock_gradient_class2.gradient = [0.2] * 100
        mock_gradient_class2.gradient_hash = "hash2"
        mock_gradient_class2.pogq_score = 0.9
        mock_gradient_class2.zkpoc_verified = False
        mock_gradient_class2.risc0_verified = False

        mock_backend.get_gradients_by_round = AsyncMock(
            return_value=[mock_gradient_class, mock_gradient_class2]
        )

        coordinator = Phase10Coordinator(coordinator_config)
        coordinator.backends = [mock_backend]

        # The aggregation should use different weights
        # This is a simplified test - full test would verify actual weights used
        result = await coordinator.aggregate_round()

        # Should include ZK stats in result
        assert result['success'] is True
        assert 'risc0_verified_count' in result
        assert result['risc0_verified_count'] == 1
        assert result['non_verified_count'] == 1


@pytest.mark.asyncio
class TestCoordinatorStats:
    """Tests for coordinator statistics with ZK metrics."""

    @pytest.mark.skipif(not COORDINATOR_AVAILABLE, reason="Coordinator not available")
    async def test_stats_include_zk_metrics(
        self, coordinator_config, mock_backend
    ):
        """Test that get_stats includes ZK metrics."""
        coordinator = Phase10Coordinator(coordinator_config)
        coordinator.backends = [mock_backend]

        stats = await coordinator.get_stats()

        assert 'risc0_zkstark' in stats
        zk_stats = stats['risc0_zkstark']

        assert 'proofs_required' in zk_stats
        assert 'proofs_preferred' in zk_stats
        assert 'weight_multiplier' in zk_stats
        assert 'proofs_submitted' in zk_stats
        assert 'proofs_verified' in zk_stats

    @pytest.mark.skipif(not COORDINATOR_AVAILABLE, reason="Coordinator not available")
    async def test_stats_include_verifier_metrics(
        self, coordinator_config, mock_backend, mock_zk_verifier
    ):
        """Test that get_stats includes verifier-specific metrics."""
        coordinator = Phase10Coordinator(coordinator_config)
        coordinator.backends = [mock_backend]
        coordinator.risc0_verifier = mock_zk_verifier

        stats = await coordinator.get_stats()

        assert 'risc0_zkstark' in stats
        assert 'verifier_metrics' in stats['risc0_zkstark']


class TestCoordinatorKrumAggregation:
    """Tests for weighted Krum aggregation."""

    @pytest.mark.skipif(not COORDINATOR_AVAILABLE, reason="Coordinator not available")
    def test_krum_aggregation_basic(self, coordinator_config):
        """Test basic Krum aggregation."""
        coordinator = Phase10Coordinator(coordinator_config)

        gradients = [
            [0.1] * 10,
            [0.2] * 10,
            [0.15] * 10,
        ]

        aggregated, byzantine_count = coordinator._krum_aggregation(gradients)

        assert len(aggregated) == 10
        assert byzantine_count >= 0

    @pytest.mark.skipif(not COORDINATOR_AVAILABLE, reason="Coordinator not available")
    def test_krum_aggregation_weighted(self, coordinator_config):
        """Test weighted Krum aggregation."""
        coordinator = Phase10Coordinator(coordinator_config)

        gradients = [
            [0.1] * 10,
            [0.2] * 10,
            [0.15] * 10,
        ]
        weights = [1.5, 1.0, 1.0]  # First gradient has higher weight

        aggregated, byzantine_count = coordinator._krum_aggregation_weighted(
            gradients, weights
        )

        assert len(aggregated) == 10
        assert byzantine_count >= 0

    @pytest.mark.skipif(not COORDINATOR_AVAILABLE, reason="Coordinator not available")
    def test_krum_aggregation_empty(self, coordinator_config):
        """Test Krum aggregation with empty input."""
        coordinator = Phase10Coordinator(coordinator_config)

        aggregated, byzantine_count = coordinator._krum_aggregation([])

        # Should return zero gradient
        assert len(aggregated) == 100  # Default size
        assert byzantine_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
