#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test MATL Holochain Bridge

Tests the MATLClient that wraps FLCoordinatorClient with PoGQ v4.1
and MATL composite scoring functionality.

Run with:
    pytest tests/test_holochain_bridge_matl.py -v
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add source path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zerotrustml.holochain_bridge.matl import (
    MATLClient,
    create_matl_client,
    PoGQv41Config,
    PoGQv41EvaluationData,
    PoGQv41Stats,
    PoGQv41Result,
    RoundMATLStats,
    CompositeScore,
    POGQ_WEIGHT,
    TCDM_WEIGHT,
    ENTROPY_WEIGHT,
    MAX_BYZANTINE_FRACTION,
)


class TestMATLDataclasses:
    """Unit tests for MATL dataclass types."""

    def test_pogq_v41_config_defaults(self):
        config = PoGQv41Config()
        assert config.ema_beta == 0.85
        assert config.warmup_rounds == 3
        assert config.hysteresis_k == 2
        assert config.hysteresis_m == 3
        assert config.egregious_threshold == 0.15
        assert config.byzantine_threshold == 0.5

    def test_pogq_v41_config_to_dict(self):
        config = PoGQv41Config(ema_beta=0.9, warmup_rounds=5)
        d = config.to_dict()
        assert d["ema_beta"] == 0.9
        assert d["warmup_rounds"] == 5
        assert len(d) == 6

    def test_pogq_v41_evaluation_from_dict(self):
        data = {
            "client_id": "node-1",
            "round": 5,
            "raw_score": 0.85,
            "ema_score": 0.82,
            "final_score": 0.83,
            "is_byzantine": False,
            "is_quarantined": False,
            "in_warmup": True,
            "rejection_reason": None,
            "confidence": 0.95,
        }
        eval_data = PoGQv41EvaluationData.from_dict(data)
        assert eval_data.client_id == "node-1"
        assert eval_data.round == 5
        assert eval_data.raw_score == 0.85
        assert eval_data.in_warmup is True
        assert eval_data.rejection_reason is None

    def test_pogq_v41_evaluation_from_empty_dict(self):
        eval_data = PoGQv41EvaluationData.from_dict({})
        assert eval_data.client_id == ""
        assert eval_data.round == 0
        assert eval_data.is_byzantine is False

    def test_pogq_v41_stats_from_dict(self):
        data = {
            "total_clients": 10,
            "quarantined_clients": 1,
            "clients_in_warmup": 3,
            "average_score": 0.78,
            "current_round": 7,
        }
        stats = PoGQv41Stats.from_dict(data)
        assert stats.total_clients == 10
        assert stats.quarantined_clients == 1
        assert stats.average_score == 0.78

    def test_pogq_v41_result_from_dict(self):
        data = {
            "action_hash": "uhCkk123",
            "evaluation": {
                "client_id": "node-2",
                "round": 3,
                "raw_score": 0.9,
                "ema_score": 0.88,
                "final_score": 0.89,
                "is_byzantine": False,
                "is_quarantined": False,
                "in_warmup": False,
                "rejection_reason": None,
                "confidence": 0.99,
            },
            "statistics": {
                "total_clients": 5,
                "quarantined_clients": 0,
                "clients_in_warmup": 0,
                "average_score": 0.87,
                "current_round": 3,
            },
        }
        result = PoGQv41Result.from_dict(data)
        assert result.action_hash == "uhCkk123"
        assert result.evaluation.client_id == "node-2"
        assert result.statistics.total_clients == 5

    def test_round_matl_stats_from_dict(self):
        data = {
            "round": 10,
            "participant_count": 20,
            "average_quality": 0.88,
            "average_consistency": 0.91,
            "average_composite": 0.86,
            "byzantine_count": 2,
            "byzantine_fraction": 0.10,
            "within_tolerance": True,
        }
        stats = RoundMATLStats.from_dict(data)
        assert stats.round == 10
        assert stats.participant_count == 20
        assert stats.byzantine_fraction == 0.10
        assert stats.within_tolerance is True

    def test_composite_score_creation(self):
        score = CompositeScore(
            pogq=0.9, tcdm=0.8, entropy=0.7, composite=0.82
        )
        assert score.pogq == 0.9
        assert score.composite == 0.82


class TestMATLClientUnit:
    """Unit tests for MATLClient (no Holochain required)."""

    def test_client_creation(self):
        mock_fl = MagicMock()
        client = MATLClient(mock_fl)
        assert client._fl is mock_fl

    def test_factory_function(self):
        mock_fl = MagicMock()
        client = create_matl_client(mock_fl)
        assert isinstance(client, MATLClient)

    def test_compute_composite_score(self):
        client = MATLClient(MagicMock())
        score = client.compute_composite_score(
            pogq=1.0, tcdm=1.0, entropy=1.0
        )
        assert score.composite == pytest.approx(1.0)

    def test_compute_composite_score_weighted(self):
        client = MATLClient(MagicMock())
        score = client.compute_composite_score(
            pogq=0.8, tcdm=0.6, entropy=0.4
        )
        expected = 0.8 * 0.4 + 0.6 * 0.3 + 0.4 * 0.3
        assert score.composite == pytest.approx(expected)

    def test_compute_composite_score_clamped(self):
        client = MATLClient(MagicMock())
        score = client.compute_composite_score(
            pogq=0.0, tcdm=0.0, entropy=0.0
        )
        assert score.composite == 0.0

    def test_byzantine_tolerance_within(self):
        client = MATLClient(MagicMock())
        assert client.is_within_byzantine_tolerance(0.30) is True
        assert client.is_within_byzantine_tolerance(0.34) is True

    def test_byzantine_tolerance_exceeded(self):
        client = MATLClient(MagicMock())
        assert client.is_within_byzantine_tolerance(0.35) is False
        assert client.is_within_byzantine_tolerance(0.45) is False

    def test_weights_sum_to_one(self):
        assert POGQ_WEIGHT + TCDM_WEIGHT + ENTROPY_WEIGHT == pytest.approx(
            1.0
        )

    def test_max_byzantine_fraction(self):
        assert MAX_BYZANTINE_FRACTION == 0.34


class TestMATLClientMocked:
    """Tests with mocked _call_zome."""

    @pytest.fixture
    def fl_client(self):
        client = MagicMock()
        client._connected = True
        client.cell_id = "test-cell-id"
        return client

    @pytest.fixture
    def matl_client(self, fl_client):
        return MATLClient(fl_client)

    @pytest.mark.asyncio
    async def test_submit_gradient_with_pogq_v41(self, matl_client):
        matl_client._fl._call_zome = AsyncMock(
            return_value={
                "action_hash": "uhCkk456",
                "evaluation": {
                    "client_id": "node-1",
                    "round": 1,
                    "raw_score": 0.85,
                    "ema_score": 0.85,
                    "final_score": 0.85,
                    "is_byzantine": False,
                    "is_quarantined": False,
                    "in_warmup": True,
                    "rejection_reason": None,
                    "confidence": 0.7,
                },
                "statistics": {
                    "total_clients": 1,
                    "quarantined_clients": 0,
                    "clients_in_warmup": 1,
                    "average_score": 0.85,
                    "current_round": 1,
                },
            }
        )

        result = await matl_client.submit_gradient_with_pogq_v41(
            node_id="node-1",
            round=1,
            gradient_hash="abc123",
            quality=0.85,
            consistency=0.90,
        )
        assert isinstance(result, PoGQv41Result)
        assert result.action_hash == "uhCkk456"
        assert result.evaluation.in_warmup is True
        assert result.statistics.clients_in_warmup == 1

    @pytest.mark.asyncio
    async def test_submit_gradient_with_config(self, matl_client):
        matl_client._fl._call_zome = AsyncMock(
            return_value={
                "action_hash": "uhCkk789",
                "evaluation": {
                    "client_id": "node-1",
                    "round": 1,
                    "raw_score": 0.9,
                    "ema_score": 0.9,
                    "final_score": 0.9,
                    "is_byzantine": False,
                    "is_quarantined": False,
                    "in_warmup": False,
                    "rejection_reason": None,
                    "confidence": 0.95,
                },
                "statistics": {
                    "total_clients": 5,
                    "quarantined_clients": 0,
                    "clients_in_warmup": 0,
                    "average_score": 0.88,
                    "current_round": 10,
                },
            }
        )

        config = PoGQv41Config(ema_beta=0.9, warmup_rounds=5)
        result = await matl_client.submit_gradient_with_pogq_v41(
            node_id="node-1",
            round=10,
            gradient_hash="def456",
            quality=0.9,
            consistency=0.95,
            config=config,
        )
        # Verify config was passed in the payload
        call_args = matl_client._fl._call_zome.call_args
        payload = call_args[0][1]
        assert payload["config"]["ema_beta"] == 0.9
        assert payload["config"]["warmup_rounds"] == 5

    @pytest.mark.asyncio
    async def test_get_round_matl_stats(self, matl_client):
        matl_client._fl._call_zome = AsyncMock(
            return_value={
                "round": 5,
                "participant_count": 15,
                "average_quality": 0.82,
                "average_consistency": 0.88,
                "average_composite": 0.84,
                "byzantine_count": 1,
                "byzantine_fraction": 0.067,
                "within_tolerance": True,
            }
        )

        stats = await matl_client.get_round_matl_stats(round=5)
        assert isinstance(stats, RoundMATLStats)
        assert stats.participant_count == 15
        assert stats.within_tolerance is True
