#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test FL Coordinator Client

Tests the FLCoordinatorClient that connects to the Holochain
federated_learning_coordinator zome.

Run with:
    RUN_HOLOCHAIN_TESTS=1 pytest tests/test_fl_coordinator_client.py -v
"""

import os
import sys
import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add source path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zerotrustml.holochain import (
    FLCoordinatorClient,
    create_fl_coordinator,
    GradientSubmission,
    ModelGradient,
    NodeReputation,
    ByzantineDetectionResult,
    ByzantineType,
)


class TestFLCoordinatorClientUnit:
    """Unit tests (no Holochain required)"""

    def test_client_creation(self):
        """Test client can be created"""
        client = FLCoordinatorClient(
            admin_url="ws://localhost:8888",
            app_url="ws://localhost:8889",
            app_id="mycelix-fl",
        )
        assert client.admin_url == "ws://localhost:8888"
        assert client.app_url == "ws://localhost:8889"
        assert client.app_id == "mycelix-fl"
        assert client.zome_name == "federated_learning_coordinator"

    def test_factory_function(self):
        """Test create_fl_coordinator factory"""
        client = create_fl_coordinator(
            admin_url="ws://localhost:8888",
            app_url="ws://localhost:8889",
        )
        assert isinstance(client, FLCoordinatorClient)

    def test_dataclass_creation(self):
        """Test dataclass types work correctly"""
        # GradientSubmission
        submission = GradientSubmission(
            action_hash="abc123",
            node_id="node-1",
            round_num=1,
            timestamp=1234567890,
            pogq_score=0.8,
            reputation_delta=0.05,
        )
        assert submission.action_hash == "abc123"
        assert submission.pogq_score == 0.8

        # ModelGradient
        gradient = ModelGradient(
            node_id="node-1",
            round_num=1,
            gradient_data=b"\x00\x01\x02",
            gradient_shape=[100, 10],
            gradient_dtype="float32",
        )
        assert gradient.node_id == "node-1"
        assert gradient.gradient_shape == [100, 10]

        # NodeReputation
        rep = NodeReputation(
            node_id="node-1",
            score=0.85,
            total_rounds=10,
            successful_rounds=9,
            byzantine_flags=0,
            last_update=1234567890,
        )
        assert rep.score == 0.85
        assert rep.successful_rounds == 9

        # ByzantineDetectionResult
        detection = ByzantineDetectionResult(
            detected=True,
            byzantine_nodes=["node-5"],
            detection_method="hierarchical",
            confidence=0.95,
        )
        assert detection.detected
        assert "node-5" in detection.byzantine_nodes

    def test_byzantine_type_enum(self):
        """Test ByzantineType enum values"""
        assert ByzantineType.DATA_POISONING.value == "DataPoisoning"
        assert ByzantineType.MODEL_POISONING.value == "ModelPoisoning"
        assert ByzantineType.FREE_RIDING.value == "FreeRiding"
        assert ByzantineType.CARTEL_COLLUSION.value == "CartelCollusion"


class TestFLCoordinatorClientMocked:
    """Tests with mocked WebSocket connections"""

    @pytest.fixture
    def mock_client(self):
        """Create client with mocked connections"""
        client = FLCoordinatorClient()
        client._connected = True
        client.cell_id = "test-cell-id"
        return client

    @pytest.mark.asyncio
    async def test_submit_gradient(self, mock_client):
        """Test gradient submission with mock"""
        mock_client._call_zome = AsyncMock(return_value="action-hash-123")

        result = await mock_client.submit_gradient(
            node_id="node-1",
            round_num=1,
            gradient_data=b"\x00\x01\x02\x03",
            gradient_shape=[100, 10],
        )

        assert isinstance(result, GradientSubmission)
        assert result.action_hash == "action-hash-123"
        assert result.node_id == "node-1"
        assert result.round_num == 1

        # Verify zome was called
        mock_client._call_zome.assert_called_once()
        call_args = mock_client._call_zome.call_args
        assert call_args[0][0] == "submit_gradient"

    @pytest.mark.asyncio
    async def test_get_round_gradients(self, mock_client):
        """Test getting round gradients"""
        mock_client._call_zome = AsyncMock(return_value=[
            (
                "action-1",
                {
                    "node_id": "node-1",
                    "round_num": 1,
                    "gradient_data": "AAEC",  # base64 encoded
                    "gradient_shape": [100],
                    "gradient_dtype": "float32",
                    "timestamp": 1234567890,
                },
            ),
            (
                "action-2",
                {
                    "node_id": "node-2",
                    "round_num": 1,
                    "gradient_data": "AwQF",
                    "gradient_shape": [100],
                    "gradient_dtype": "float32",
                    "timestamp": 1234567891,
                },
            ),
        ])

        gradients = await mock_client.get_round_gradients(round_num=1)

        assert len(gradients) == 2
        assert all(isinstance(g, ModelGradient) for g in gradients)
        assert gradients[0].node_id == "node-1"
        assert gradients[1].node_id == "node-2"

    @pytest.mark.asyncio
    async def test_get_reputation(self, mock_client):
        """Test getting node reputation"""
        mock_client._call_zome = AsyncMock(return_value={
            "score": 0.85,
            "total_rounds": 10,
            "successful_rounds": 9,
            "byzantine_flags": 0,
            "last_update": 1234567890,
            "trend": "improving",
        })

        rep = await mock_client.get_reputation("node-1")

        assert isinstance(rep, NodeReputation)
        assert rep.score == 0.85
        assert rep.trend == "improving"

    @pytest.mark.asyncio
    async def test_detect_byzantine_hierarchical(self, mock_client):
        """Test Byzantine detection"""
        mock_client._call_zome = AsyncMock(return_value={
            "detected": True,
            "byzantine_nodes": ["node-5", "node-7"],
            "confidence": 0.92,
            "details": {"method": "hierarchical", "threshold": 0.3},
        })

        result = await mock_client.detect_byzantine_hierarchical(round_num=1)

        assert isinstance(result, ByzantineDetectionResult)
        assert result.detected
        assert "node-5" in result.byzantine_nodes
        assert result.confidence == 0.92

    @pytest.mark.asyncio
    async def test_record_byzantine(self, mock_client):
        """Test recording Byzantine event"""
        mock_client._call_zome = AsyncMock(return_value="action-hash-byzantine")

        result = await mock_client.record_byzantine(
            node_id="node-5",
            round_num=1,
            byzantine_type=ByzantineType.DATA_POISONING,
            evidence_hash="evidence-abc123",
            detected_by="node-1",
            confidence=0.95,
        )

        assert result == "action-hash-byzantine"

        # Verify payload
        call_args = mock_client._call_zome.call_args
        payload = call_args[0][1]
        assert payload["byzantine_type"] == "DataPoisoning"
        assert payload["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_health_check_connected(self, mock_client):
        """Test health check when connected"""
        mock_client._call_admin = AsyncMock(return_value={"apps": []})

        result = await mock_client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self):
        """Test health check when disconnected"""
        client = FLCoordinatorClient()
        client._connected = False

        result = await client.health_check()
        assert result is False


# Integration tests (require running Holochain)
if os.environ.get("RUN_HOLOCHAIN_TESTS"):

    class TestFLCoordinatorClientIntegration:
        """Integration tests with real Holochain conductor"""

        @pytest.fixture
        async def client(self):
            """Create and connect client"""
            client = create_fl_coordinator()
            await client.connect()
            yield client
            await client.disconnect()

        @pytest.mark.asyncio
        async def test_connection(self, client):
            """Test connection to conductor"""
            assert client.is_connected
            assert client.cell_id is not None

        @pytest.mark.asyncio
        async def test_health_check(self, client):
            """Test health check with real conductor"""
            result = await client.health_check()
            assert result is True

        @pytest.mark.asyncio
        async def test_submit_and_get_gradient(self, client):
            """Test gradient submission and retrieval"""
            import random
            import struct

            # Create test gradient
            gradient_values = [random.random() for _ in range(100)]
            gradient_bytes = struct.pack(f"{len(gradient_values)}f", *gradient_values)

            # Submit
            result = await client.submit_gradient(
                node_id="test-node-1",
                round_num=999,  # Use high round to avoid conflicts
                gradient_data=gradient_bytes,
                gradient_shape=[100],
            )

            assert result.action_hash
            assert result.node_id == "test-node-1"

            # Retrieve
            gradients = await client.get_round_gradients(round_num=999)
            assert any(g.node_id == "test-node-1" for g in gradients)

        @pytest.mark.asyncio
        async def test_reputation_flow(self, client):
            """Test reputation get/update"""
            # Get initial reputation (may not exist)
            rep = await client.get_reputation("test-node-1")

            # Update reputation
            await client.update_reputation(
                node_id="test-node-1",
                delta=0.1,
                reason="test update",
                round_num=999,
            )

            # Get updated reputation
            new_rep = await client.get_reputation("test-node-1")
            assert new_rep is not None


else:
    # Skip integration tests if not enabled
    class TestFLCoordinatorClientIntegration:
        @pytest.fixture
        def skip_reason(self):
            pytest.skip(
                "Integration tests require running Holochain; "
                "set RUN_HOLOCHAIN_TESTS=1 to enable"
            )

        def test_placeholder(self, skip_reason):
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
