#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test FL Pipeline Holochain Bridge

Tests the FLPipelineClient that wraps FLCoordinatorClient with
on-chain aggregation pipeline operations.

Run with:
    pytest tests/test_holochain_bridge_fl_pipeline.py -v
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add source path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zerotrustml.holochain_bridge.fl_pipeline import (
    FLPipelineClient,
    create_fl_pipeline_client,
    DetectionSummary,
    ValidatorPipelineResult,
    ReputationInfo,
    HV16_BYTES,
    HV16_BITS,
)


class TestPipelineDataclasses:
    """Unit tests for pipeline dataclass types."""

    def test_detection_summary_from_dict(self):
        data = {
            "flagged_nodes": {"node-3": 0.95, "node-7": 0.88},
            "detection_layers_used": ["cosine", "magnitude", "cartel"],
            "total_checked": 10,
            "total_flagged": 2,
        }
        summary = DetectionSummary.from_dict(data)
        assert len(summary.flagged_nodes) == 2
        assert summary.flagged_nodes["node-3"] == 0.95
        assert "cosine" in summary.detection_layers_used
        assert summary.total_flagged == 2

    def test_detection_summary_from_empty_dict(self):
        summary = DetectionSummary.from_dict({})
        assert summary.flagged_nodes == {}
        assert summary.total_checked == 0

    def test_validator_pipeline_result_from_dict(self):
        hv_bytes = bytes([0xFF] * HV16_BYTES)
        data = {
            "commitment_hash": "sha256:abc123",
            "aggregated_hv": list(hv_bytes),
            "method": "trimmed_mean",
            "gradient_count": 8,
            "excluded_count": 1,
            "excluded_participants": ["node-3"],
            "detection_summary": {
                "flagged_nodes": {"node-3": 0.92},
                "detection_layers_used": ["cosine"],
                "total_checked": 8,
                "total_flagged": 1,
            },
        }
        result = ValidatorPipelineResult.from_dict(data)
        assert result.commitment_hash == "sha256:abc123"
        assert result.method == "trimmed_mean"
        assert result.gradient_count == 8
        assert result.excluded_count == 1
        assert "node-3" in result.excluded_participants
        assert len(result.aggregated_hv) == HV16_BYTES
        assert result.detection_summary.total_flagged == 1

    def test_validator_pipeline_result_base64_hv(self):
        import base64

        hv_bytes = bytes([0xAA] * 32)  # small for test
        data = {
            "commitment_hash": "sha256:xyz",
            "aggregated_hv": base64.b64encode(hv_bytes).decode(),
            "method": "median",
            "gradient_count": 5,
            "excluded_count": 0,
            "excluded_participants": [],
            "detection_summary": {},
        }
        result = ValidatorPipelineResult.from_dict(data)
        assert result.aggregated_hv == hv_bytes

    def test_reputation_info_from_dict(self):
        data = {
            "node_id": "node-1",
            "reputation_score": 0.82,
            "successful_rounds": 15,
            "failed_rounds": 2,
            "last_updated": 1708000000,
        }
        rep = ReputationInfo.from_dict(data)
        assert rep.node_id == "node-1"
        assert rep.reputation_score == 0.82
        assert rep.successful_rounds == 15

    def test_reputation_info_defaults(self):
        rep = ReputationInfo.from_dict({})
        assert rep.reputation_score == 0.5  # neutral default
        assert rep.successful_rounds == 0

    def test_hv16_constants(self):
        assert HV16_BITS == 16_384
        assert HV16_BYTES == 2048


class TestFLPipelineClientUnit:
    """Unit tests for FLPipelineClient (no Holochain required)."""

    def test_client_creation(self):
        mock_fl = MagicMock()
        client = FLPipelineClient(mock_fl)
        assert client._fl is mock_fl

    def test_factory_function(self):
        mock_fl = MagicMock()
        client = create_fl_pipeline_client(mock_fl)
        assert isinstance(client, FLPipelineClient)


class TestFLPipelineClientMocked:
    """Tests with mocked _call_zome."""

    @pytest.fixture
    def fl_client(self):
        client = MagicMock()
        client._connected = True
        client.cell_id = "test-cell-id"
        return client

    @pytest.fixture
    def pipeline_client(self, fl_client):
        return FLPipelineClient(fl_client)

    @pytest.mark.asyncio
    async def test_run_validator_pipeline(self, pipeline_client):
        pipeline_client._fl._call_zome = AsyncMock(
            return_value={
                "commitment_hash": "sha256:pipeline_result",
                "aggregated_hv": list(bytes(HV16_BYTES)),
                "method": "trimmed_mean",
                "gradient_count": 10,
                "excluded_count": 2,
                "excluded_participants": ["node-3", "node-7"],
                "detection_summary": {
                    "flagged_nodes": {
                        "node-3": 0.95,
                        "node-7": 0.88,
                    },
                    "detection_layers_used": ["cosine", "cartel"],
                    "total_checked": 10,
                    "total_flagged": 2,
                },
            }
        )

        result = await pipeline_client.run_validator_pipeline(round=5)
        assert isinstance(result, ValidatorPipelineResult)
        assert result.gradient_count == 10
        assert result.excluded_count == 2
        assert len(result.aggregated_hv) == HV16_BYTES

        # Verify the zome function was called correctly
        pipeline_client._fl._call_zome.assert_called_once_with(
            "run_validator_pipeline", 5
        )

    @pytest.mark.asyncio
    async def test_run_and_commit(self, pipeline_client):
        pipeline_client._fl._call_zome = AsyncMock(
            return_value={"action_hash": "uhCkk_commit_123"}
        )

        action_hash = await pipeline_client.run_and_commit(round=5)
        assert action_hash == "uhCkk_commit_123"

    @pytest.mark.asyncio
    async def test_run_and_reveal(self, pipeline_client):
        pipeline_client._fl._call_zome = AsyncMock(
            return_value={"action_hash": "uhCkk_reveal_456"}
        )

        action_hash = await pipeline_client.run_and_reveal(round=5)
        assert action_hash == "uhCkk_reveal_456"

    @pytest.mark.asyncio
    async def test_compute_hypervector_similarity(self, pipeline_client):
        pipeline_client._fl._call_zome = AsyncMock(return_value=0.75)

        hv1 = bytes([0xFF] * HV16_BYTES)
        hv2 = bytes([0xFF] * HV16_BYTES)
        sim = await pipeline_client.compute_hypervector_similarity(hv1, hv2)
        assert sim == 0.75

    @pytest.mark.asyncio
    async def test_compute_similarity_wrong_size(self, pipeline_client):
        with pytest.raises(ValueError, match="2048 bytes"):
            await pipeline_client.compute_hypervector_similarity(
                b"\x00" * 100, b"\x00" * 100
            )

    @pytest.mark.asyncio
    async def test_get_reputation(self, pipeline_client):
        pipeline_client._fl._call_zome = AsyncMock(
            return_value={
                "node_id": "node-1",
                "reputation_score": 0.85,
                "successful_rounds": 20,
                "failed_rounds": 1,
                "last_updated": 1708000000,
            }
        )

        rep = await pipeline_client.get_reputation("node-1")
        assert isinstance(rep, ReputationInfo)
        assert rep.reputation_score == 0.85
        assert rep.successful_rounds == 20
