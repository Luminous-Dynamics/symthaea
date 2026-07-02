# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Unit Tests for FL Governance Integration
Week 7-8 Phase 5: Testing & Validation

Tests for FLGovernanceIntegration and FLGovernanceConfig
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from zerotrustml.governance.fl_integration import (
    FLGovernanceIntegration,
    FLGovernanceConfig,
)
from zerotrustml.governance.models import ProposalType, ProposalStatus


@pytest.fixture
def fl_gov_config():
    """Create test FL governance configuration"""
    return FLGovernanceConfig(
        require_capability_for_submit=True,
        require_capability_for_request=True,
        emergency_stop_enabled=True,
        emergency_stop_requires_guardian=True,
        ban_requires_guardian=True,
        ban_requires_proposal=True,
        parameter_change_requires_proposal=True,
        critical_parameters=["min_reputation", "byzantine_threshold"],
        reputation_weighted_rewards=True,
        governance_participation_bonus=1.2,
    )


@pytest.fixture
def mock_governance_coordinator():
    """Create mock governance coordinator"""
    mock = MagicMock()

    # Mock authorize_fl_action
    mock.authorize_fl_action = AsyncMock(return_value=(True, "Authorized"))

    # Mock create_governance_proposal
    mock.create_governance_proposal = AsyncMock(
        return_value=(True, "Proposal created", "prop_test123")
    )

    # Mock guardian_auth_mgr
    mock.guardian_auth_mgr = MagicMock()
    mock.guardian_auth_mgr.request_authorization = AsyncMock(
        return_value=(True, "Request created", "auth_test123")
    )

    # Mock proposal_mgr
    mock.proposal_mgr = MagicMock()
    mock.proposal_mgr.get_proposal = AsyncMock(return_value=None)
    mock.proposal_mgr.update_proposal_status = AsyncMock(return_value=True)

    # Mock gov_extensions
    mock.gov_extensions = MagicMock()
    mock.gov_extensions.calculate_vote_weight = AsyncMock(return_value=3.0)
    mock.gov_extensions.calculate_vote_budget = MagicMock(return_value=100)
    mock.gov_extensions.calculate_effective_votes = MagicMock(return_value=10.0)
    mock.gov_extensions.get_governance_stats = AsyncMock(return_value={})

    return mock


@pytest.fixture
def fl_gov_integration(mock_governance_coordinator, fl_gov_config):
    """Create FLGovernanceIntegration instance"""
    return FLGovernanceIntegration(
        governance_coordinator=mock_governance_coordinator,
        config=fl_gov_config,
    )


class TestFLGovernanceConfig:
    """Tests for FLGovernanceConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = FLGovernanceConfig()

        assert config.require_capability_for_submit is True
        assert config.require_capability_for_request is True
        assert config.emergency_stop_enabled is True
        assert config.reputation_weighted_rewards is True
        assert "min_reputation" in config.critical_parameters

    def test_custom_config(self, fl_gov_config):
        """Test custom configuration"""
        assert fl_gov_config.emergency_stop_requires_guardian is True
        assert fl_gov_config.ban_requires_proposal is True
        assert fl_gov_config.governance_participation_bonus == 1.2


class TestPreRoundCapabilityChecks:
    """Tests for pre-round capability verification"""

    @pytest.mark.asyncio
    async def test_verify_participant_authorized(self, fl_gov_integration, mock_governance_coordinator):
        """Test participant verification when authorized"""
        authorized, reason = await fl_gov_integration.verify_participant_for_round(
            participant_id="alice_id",
            round_number=42,
        )

        assert authorized is True
        assert reason == "Authorized"
        mock_governance_coordinator.authorize_fl_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_participant_emergency_stopped(self, fl_gov_integration):
        """Test participant verification during emergency stop"""
        fl_gov_integration.emergency_stopped = True

        authorized, reason = await fl_gov_integration.verify_participant_for_round(
            participant_id="alice_id",
            round_number=42,
        )

        assert authorized is False
        assert reason == "Emergency stop active"

    @pytest.mark.asyncio
    async def test_verify_participant_banned_permanent(self, fl_gov_integration):
        """Test participant verification when banned (permanent)"""
        fl_gov_integration.banned_participants["alice_id"] = {
            "permanent": True,
            "reason": "Repeated violations",
        }

        authorized, reason = await fl_gov_integration.verify_participant_for_round(
            participant_id="alice_id",
            round_number=42,
        )

        assert authorized is False
        assert "permanently banned" in reason

    @pytest.mark.asyncio
    async def test_verify_participant_banned_temporary_active(self, fl_gov_integration):
        """Test participant verification with active temporary ban"""
        ban_until = int(time.time()) + 86400  # 1 day from now
        fl_gov_integration.banned_participants["alice_id"] = {
            "permanent": False,
            "until": ban_until,
            "reason": "Temporary suspension",
        }

        authorized, reason = await fl_gov_integration.verify_participant_for_round(
            participant_id="alice_id",
            round_number=42,
        )

        assert authorized is False
        assert "banned until" in reason

    @pytest.mark.asyncio
    async def test_verify_participant_banned_temporary_expired(self, fl_gov_integration, mock_governance_coordinator):
        """Test participant verification with expired temporary ban"""
        ban_until = int(time.time()) - 86400  # 1 day ago (expired)
        fl_gov_integration.banned_participants["alice_id"] = {
            "permanent": False,
            "until": ban_until,
            "reason": "Temporary suspension",
        }

        authorized, reason = await fl_gov_integration.verify_participant_for_round(
            participant_id="alice_id",
            round_number=42,
        )

        # Ban expired, should be removed and participant authorized
        assert authorized is True
        assert "alice_id" not in fl_gov_integration.banned_participants

    @pytest.mark.asyncio
    async def test_verify_participant_capability_failed(self, fl_gov_integration, mock_governance_coordinator):
        """Test participant verification when capability check fails"""
        mock_governance_coordinator.authorize_fl_action.return_value = (
            False,
            "Insufficient reputation: 0.4"
        )

        authorized, reason = await fl_gov_integration.verify_participant_for_round(
            participant_id="alice_id",
            round_number=42,
        )

        assert authorized is False
        assert "Capability check failed" in reason

    @pytest.mark.asyncio
    async def test_verify_model_request_authorized(self, fl_gov_integration, mock_governance_coordinator):
        """Test model request verification when authorized"""
        authorized, reason = await fl_gov_integration.verify_model_request(
            participant_id="alice_id"
        )

        assert authorized is True
        assert reason == "Authorized"

    @pytest.mark.asyncio
    async def test_verify_model_request_emergency_stopped(self, fl_gov_integration):
        """Test model request during emergency stop"""
        fl_gov_integration.emergency_stopped = True

        authorized, reason = await fl_gov_integration.verify_model_request(
            participant_id="alice_id"
        )

        assert authorized is False
        assert reason == "Emergency stop active"


class TestEmergencyActions:
    """Tests for emergency stop and resume"""

    @pytest.mark.asyncio
    async def test_request_emergency_stop_with_guardian(self, fl_gov_integration, mock_governance_coordinator):
        """Test emergency stop request requiring guardian approval"""
        mock_governance_coordinator.authorize_fl_action.return_value = (
            False,
            "Guardian approval required"
        )

        success, message, request_id = await fl_gov_integration.request_emergency_stop(
            requester_participant_id="alice_id",
            reason="Byzantine attack detected",
            evidence={"malicious_nodes": ["eve_id"]},
        )

        assert success is True
        assert "pending guardian approval" in message
        assert request_id == "auth_test123"

    @pytest.mark.asyncio
    async def test_request_emergency_stop_immediate(self, fl_gov_integration, mock_governance_coordinator):
        """Test emergency stop with immediate authorization"""
        mock_governance_coordinator.authorize_fl_action.return_value = (True, "Authorized")

        success, message, action_id = await fl_gov_integration.request_emergency_stop(
            requester_participant_id="guardian_alice",
            reason="Critical security breach",
        )

        assert success is True
        assert fl_gov_integration.emergency_stopped is True
        assert action_id is not None

    @pytest.mark.asyncio
    async def test_execute_emergency_stop(self, fl_gov_integration):
        """Test emergency stop execution"""
        assert fl_gov_integration.emergency_stopped is False

        success, message, action_id = await fl_gov_integration.execute_emergency_stop(
            reason="Emergency confirmed",
            executed_by="guardian_alice",
        )

        assert success is True
        assert fl_gov_integration.emergency_stopped is True
        assert action_id is not None

    @pytest.mark.asyncio
    async def test_execute_emergency_stop_already_active(self, fl_gov_integration):
        """Test emergency stop when already active"""
        fl_gov_integration.emergency_stopped = True

        success, message, action_id = await fl_gov_integration.execute_emergency_stop(
            reason="Another emergency",
            executed_by="guardian_bob",
        )

        assert success is False
        assert "already active" in message

    @pytest.mark.asyncio
    async def test_resume_fl_training(self, fl_gov_integration, mock_governance_coordinator):
        """Test FL training resumption"""
        fl_gov_integration.emergency_stopped = True

        success, message = await fl_gov_integration.resume_fl_training(
            requester_participant_id="alice_id",
            reason="Emergency resolved",
        )

        assert success is True
        assert "Proposal created" in message
        mock_governance_coordinator.create_governance_proposal.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_fl_training_not_stopped(self, fl_gov_integration):
        """Test resumption when not stopped"""
        fl_gov_integration.emergency_stopped = False

        success, message = await fl_gov_integration.resume_fl_training(
            requester_participant_id="alice_id",
            reason="Resume anyway",
        )

        assert success is False
        assert "not stopped" in message


class TestParticipantManagement:
    """Tests for participant banning and unbanning"""

    @pytest.mark.asyncio
    async def test_request_ban_with_proposal(self, fl_gov_integration, mock_governance_coordinator):
        """Test ban request requiring governance proposal"""
        mock_governance_coordinator.authorize_fl_action.return_value = (True, "Authorized")

        success, message, proposal_id = await fl_gov_integration.request_participant_ban(
            requester_participant_id="alice_id",
            target_participant_id="eve_id",
            reason="Repeated Byzantine attacks",
            permanent=False,
            ban_duration_seconds=86400 * 7,
        )

        assert success is True
        assert "Ban proposal created" in message
        assert proposal_id == "prop_test123"

    @pytest.mark.asyncio
    async def test_execute_participant_ban_permanent(self, fl_gov_integration):
        """Test permanent participant ban execution"""
        success, message, action_id = await fl_gov_integration.execute_participant_ban(
            target_participant_id="eve_id",
            reason="Malicious behavior",
            permanent=True,
            ban_duration_seconds=0,
            executed_by="governance",
        )

        assert success is True
        assert "eve_id" in fl_gov_integration.banned_participants
        assert fl_gov_integration.banned_participants["eve_id"]["permanent"] is True

    @pytest.mark.asyncio
    async def test_execute_participant_ban_temporary(self, fl_gov_integration):
        """Test temporary participant ban execution"""
        ban_duration = 86400 * 30  # 30 days

        success, message, action_id = await fl_gov_integration.execute_participant_ban(
            target_participant_id="mallory_id",
            reason="Suspicious activity",
            permanent=False,
            ban_duration_seconds=ban_duration,
            executed_by="guardian_alice",
        )

        assert success is True
        ban_info = fl_gov_integration.banned_participants["mallory_id"]
        assert ban_info["permanent"] is False
        assert ban_info["duration_seconds"] == ban_duration
        assert ban_info["until"] > int(time.time())

    @pytest.mark.asyncio
    async def test_execute_participant_ban_already_banned(self, fl_gov_integration):
        """Test banning already banned participant"""
        fl_gov_integration.banned_participants["eve_id"] = {"permanent": True}

        success, message, action_id = await fl_gov_integration.execute_participant_ban(
            target_participant_id="eve_id",
            reason="Another violation",
            permanent=False,
            ban_duration_seconds=86400,
            executed_by="alice_id",
        )

        assert success is False
        assert "already banned" in message

    @pytest.mark.asyncio
    async def test_execute_participant_unban(self, fl_gov_integration):
        """Test participant unban execution"""
        fl_gov_integration.banned_participants["eve_id"] = {
            "permanent": False,
            "until": int(time.time()) + 86400,
        }

        success, message = await fl_gov_integration.execute_participant_unban(
            target_participant_id="eve_id",
            reason="Appeal approved",
            executed_by="governance",
        )

        assert success is True
        assert "eve_id" not in fl_gov_integration.banned_participants

    @pytest.mark.asyncio
    async def test_is_participant_banned_permanent(self, fl_gov_integration):
        """Test checking permanently banned participant"""
        fl_gov_integration.banned_participants["eve_id"] = {"permanent": True}

        assert fl_gov_integration.is_participant_banned("eve_id") is True

    @pytest.mark.asyncio
    async def test_is_participant_banned_temporary_active(self, fl_gov_integration):
        """Test checking temporarily banned participant (active)"""
        ban_until = int(time.time()) + 86400
        fl_gov_integration.banned_participants["eve_id"] = {
            "permanent": False,
            "until": ban_until,
        }

        assert fl_gov_integration.is_participant_banned("eve_id") is True

    @pytest.mark.asyncio
    async def test_is_participant_banned_temporary_expired(self, fl_gov_integration):
        """Test checking temporarily banned participant (expired)"""
        ban_until = int(time.time()) - 86400  # Expired
        fl_gov_integration.banned_participants["eve_id"] = {
            "permanent": False,
            "until": ban_until,
        }

        assert fl_gov_integration.is_participant_banned("eve_id") is False
        assert "eve_id" not in fl_gov_integration.banned_participants


class TestParameterManagement:
    """Tests for FL parameter management"""

    @pytest.mark.asyncio
    async def test_propose_parameter_change_critical(self, fl_gov_integration, mock_governance_coordinator):
        """Test proposing critical parameter change"""
        success, message, proposal_id = await fl_gov_integration.propose_parameter_change(
            proposer_participant_id="alice_id",
            parameter_name="min_reputation",
            current_value=0.5,
            new_value=0.7,
            rationale="Increase security",
        )

        assert success is True
        assert proposal_id == "prop_test123"
        mock_governance_coordinator.create_governance_proposal.assert_called_once()

    @pytest.mark.asyncio
    async def test_propose_parameter_change_non_critical(self, fl_gov_integration, mock_governance_coordinator, fl_gov_config):
        """Test proposing non-critical parameter change"""
        fl_gov_config.parameter_change_requires_proposal = True

        success, message, proposal_id = await fl_gov_integration.propose_parameter_change(
            proposer_participant_id="alice_id",
            parameter_name="batch_size",  # Not in critical_parameters
            current_value=32,
            new_value=64,
            rationale="Improve performance",
        )

        # Still requires proposal because parameter_change_requires_proposal=True
        assert success is True
        assert proposal_id == "prop_test123"

    @pytest.mark.asyncio
    async def test_execute_parameter_change(self, fl_gov_integration):
        """Test parameter change execution"""
        success, message = await fl_gov_integration.execute_parameter_change(
            parameter_name="min_reputation",
            new_value=0.7,
            proposal_id="prop_test123",
        )

        assert success is True
        assert fl_gov_integration.fl_parameters["min_reputation"] == 0.7

    @pytest.mark.asyncio
    async def test_get_parameter(self, fl_gov_integration):
        """Test retrieving FL parameter"""
        fl_gov_integration.fl_parameters["min_reputation"] = 0.7

        value = fl_gov_integration.get_parameter("min_reputation")
        assert value == 0.7

        value = fl_gov_integration.get_parameter("non_existent")
        assert value is None


class TestReputationWeightedRewards:
    """Tests for reputation-weighted reward calculation"""

    @pytest.mark.asyncio
    async def test_calculate_reward_high_reputation(self, fl_gov_integration, mock_governance_coordinator):
        """Test reward calculation for high-reputation participant"""
        mock_governance_coordinator.gov_extensions.calculate_vote_weight.return_value = 5.0

        adjusted = await fl_gov_integration.calculate_reputation_weighted_reward(
            participant_id="alice_id",
            base_reward=100.0,
        )

        # vote_weight=5.0 → multiplier=0.8+(5.0/20.0)=1.05
        assert adjusted == pytest.approx(105.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_calculate_reward_low_reputation(self, fl_gov_integration, mock_governance_coordinator):
        """Test reward calculation for low-reputation participant"""
        mock_governance_coordinator.gov_extensions.calculate_vote_weight.return_value = 0.66

        adjusted = await fl_gov_integration.calculate_reputation_weighted_reward(
            participant_id="dave_id",
            base_reward=100.0,
        )

        # vote_weight=0.66 → multiplier=0.8+(0.66/20.0)=0.833
        assert adjusted == pytest.approx(83.3, rel=0.01)

    @pytest.mark.asyncio
    async def test_calculate_reward_disabled(self, fl_gov_integration, fl_gov_config):
        """Test reward calculation when reputation weighting disabled"""
        fl_gov_config.reputation_weighted_rewards = False

        adjusted = await fl_gov_integration.calculate_reputation_weighted_reward(
            participant_id="alice_id",
            base_reward=100.0,
        )

        assert adjusted == 100.0  # No adjustment


class TestStatusQueries:
    """Tests for status query methods"""

    def test_get_fl_governance_status(self, fl_gov_integration):
        """Test FL governance status query"""
        fl_gov_integration.emergency_stopped = True
        fl_gov_integration.banned_participants = {"eve_id": {}, "mallory_id": {}}
        fl_gov_integration.fl_parameters = {"min_reputation": 0.7, "batch_size": 64}

        status = fl_gov_integration.get_fl_governance_status()

        assert status["emergency_stopped"] is True
        assert status["banned_participants"] == 2
        assert status["active_parameters"] == 2
        assert status["config"]["emergency_stop_enabled"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
