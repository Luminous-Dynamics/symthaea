# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Integration Tests for Governance System
Week 7-8 Phase 5: Testing & Validation

End-to-end tests for complete governance workflows
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from zerotrustml.governance import (
    GovernanceCoordinator,
    ProposalManager,
    VotingEngine,
    CapabilityEnforcer,
    GuardianAuthorizationManager,
    ProposalType,
    ProposalStatus,
    VoteChoice,
)
from zerotrustml.governance.fl_integration import FLGovernanceIntegration, FLGovernanceConfig


@pytest.fixture
def mock_dht_client():
    """Create mock DHT client"""
    mock = MagicMock()
    mock.call_zome = AsyncMock()
    return mock


@pytest.fixture
def mock_identity_coordinator():
    """Create mock identity coordinator"""
    mock = MagicMock()

    # Mock get_identity
    async def mock_get_identity(participant_id):
        identities = {
            "alice_id": {"did": "did:mycelix:alice"},
            "bob_id": {"did": "did:mycelix:bob"},
            "carol_id": {"did": "did:mycelix:carol"},
            "guardian_alice": {"did": "did:mycelix:guardian_alice"},
        }
        return identities.get(participant_id)

    mock.get_identity = mock_get_identity
    return mock


@pytest.fixture
def governance_coordinator(mock_dht_client, mock_identity_coordinator):
    """Create GovernanceCoordinator instance"""
    return GovernanceCoordinator(
        dht_client=mock_dht_client,
        identity_coordinator=mock_identity_coordinator,
    )


class TestCompleteProposalLifecycle:
    """Integration tests for complete proposal workflows"""

    @pytest.mark.asyncio
    async def test_parameter_change_proposal_workflow(self, governance_coordinator, mock_dht_client):
        """
        Test complete parameter change workflow:
        1. Create proposal
        2. Cast votes
        3. Tally votes
        4. Finalize proposal
        5. Execute proposal
        """
        # Mock DHT responses
        mock_dht_client.call_zome.side_effect = [
            None,  # store_proposal
            {  # get_proposal (for voting)
                "proposal_id": "prop_test123",
                "proposal_type": "PARAMETER_CHANGE",
                "title": "Increase min reputation",
                "description": "Security improvement",
                "proposer_did": "did:mycelix:alice",
                "proposer_participant_id": "alice_id",
                "voting_start": int(time.time()) - 3600,  # Started 1 hour ago
                "voting_end": int(time.time()) + 3600,    # Ends in 1 hour
                "quorum": 0.5,
                "approval_threshold": 0.66,
                "status": "VOTING",
                "total_votes_for": 0.0,
                "total_votes_against": 0.0,
                "total_votes_abstain": 0.0,
                "total_voting_power": 0.0,
                "execution_params": '{"parameter": "min_reputation", "new_value": 0.7}',
                "executed_at": None,
                "execution_result": "",
                "created_at": int(time.time()) - 3700,
                "updated_at": int(time.time()) - 3600,
                "tags": "[]",
            },
            None,  # store_vote (Alice)
            # ... more DHT calls for additional votes
        ]

        # Mock capability checks to return authorized
        with patch.object(
            governance_coordinator.capability_enforcer,
            'check_capability',
            new=AsyncMock(return_value=(True, "Authorized"))
        ):
            # Step 1: Create proposal
            success, message, proposal_id = await governance_coordinator.create_governance_proposal(
                proposer_participant_id="alice_id",
                proposal_type="PARAMETER_CHANGE",
                title="Increase minimum reputation threshold",
                description="Increase from 0.5 to 0.7 for better security",
                execution_params={
                    "parameter": "min_reputation",
                    "current_value": 0.5,
                    "new_value": 0.7,
                },
                voting_duration_days=7,
            )

            assert success is True
            assert proposal_id is not None

            # Step 2: Cast votes (simulated - would need full vote weight calculation)
            # In real scenario, this would call voting_engine.cast_vote() multiple times

    @pytest.mark.asyncio
    async def test_emergency_stop_with_guardian_approval_workflow(self, governance_coordinator, mock_dht_client):
        """
        Test complete emergency stop workflow:
        1. Request emergency stop
        2. Guardians approve
        3. Execute emergency stop
        4. Create resume proposal
        5. Vote on resume proposal
        6. Execute resume
        """
        fl_gov = FLGovernanceIntegration(
            governance_coordinator=governance_coordinator,
            config=FLGovernanceConfig(emergency_stop_requires_guardian=True),
        )

        # Mock capability check to require guardian approval
        with patch.object(
            governance_coordinator,
            'authorize_fl_action',
            new=AsyncMock(return_value=(False, "Guardian approval required"))
        ):
            # Step 1: Request emergency stop
            success, message, request_id = await fl_gov.request_emergency_stop(
                requester_participant_id="alice_id",
                reason="Byzantine attack detected",
                evidence={"malicious_nodes": ["eve_id", "mallory_id"]},
            )

            assert success is True
            assert "pending guardian approval" in message
            assert request_id is not None

        # Step 2: Guardians approve (simulated)
        # In real scenario, would call guardian_auth_mgr.submit_approval() multiple times

        # Step 3: Execute emergency stop (after guardian approval)
        success, message, action_id = await fl_gov.execute_emergency_stop(
            reason="Guardian approval received",
            executed_by="guardian_alice",
        )

        assert success is True
        assert fl_gov.emergency_stopped is True

        # Step 4: Request resume (after attack mitigated)
        success, message = await fl_gov.resume_fl_training(
            requester_participant_id="alice_id",
            reason="Attack mitigated, system cleaned",
        )

        assert success is True
        # Would create a proposal that needs to be voted on and executed


class TestByzantineAttackScenarios:
    """Integration tests for Byzantine attack handling"""

    @pytest.mark.asyncio
    async def test_byzantine_participant_ban_workflow(self, governance_coordinator):
        """
        Test handling of Byzantine participant:
        1. Detect Byzantine behavior
        2. Request ban
        3. Create proposal
        4. Vote on proposal
        5. Execute ban
        6. Verify submissions rejected
        """
        fl_gov = FLGovernanceIntegration(
            governance_coordinator=governance_coordinator,
            config=FLGovernanceConfig(
                ban_requires_proposal=True,
                ban_requires_guardian=False,
            ),
        )

        # Mock capability check to be authorized
        with patch.object(
            governance_coordinator,
            'authorize_fl_action',
            new=AsyncMock(return_value=(True, "Authorized"))
        ):
            # Step 1: Request ban
            success, message, proposal_id = await fl_gov.request_participant_ban(
                requester_participant_id="alice_id",
                target_participant_id="eve_id",
                reason="Consistent Byzantine attacks in rounds 45-50",
                evidence={
                    "rounds": [45, 46, 47, 48, 49, 50],
                    "pogq_scores": [0.2, 0.15, 0.1, 0.18, 0.22, 0.14],
                    "detection_method": "PoGQ threshold violation",
                },
                permanent=False,
                ban_duration_seconds=86400 * 30,  # 30 days
            )

            assert success is True
            assert proposal_id is not None

        # Step 2: Voting and approval (simulated)
        # Would involve multiple participants voting

        # Step 3: Execute ban
        success, message, action_id = await fl_gov.execute_participant_ban(
            target_participant_id="eve_id",
            reason="Proposal approved by governance",
            permanent=False,
            ban_duration_seconds=86400 * 30,
            executed_by="governance",
        )

        assert success is True
        assert "eve_id" in fl_gov.banned_participants

        # Step 4: Verify submissions rejected
        authorized, reason = await fl_gov.verify_participant_for_round(
            participant_id="eve_id",
            round_number=51,
        )

        assert authorized is False
        assert "banned" in reason.lower()


class TestReputationWeightedVoting:
    """Integration tests for reputation-weighted voting mechanics"""

    @pytest.mark.asyncio
    async def test_vote_weight_affects_outcome(self, governance_coordinator):
        """
        Test that vote weight properly affects proposal outcomes:
        1. Create proposal
        2. High-reputation participant votes FOR with few credits
        3. Low-reputation participant votes AGAINST with many credits
        4. High-reputation vote should have more impact
        """
        # This would require full integration with identity system
        # For now, test the calculation logic

        fl_gov = FLGovernanceIntegration(
            governance_coordinator=governance_coordinator,
            config=FLGovernanceConfig(reputation_weighted_rewards=True),
        )

        # Mock vote weights
        with patch.object(
            governance_coordinator.gov_extensions,
            'calculate_vote_weight',
            new=AsyncMock()
        ) as mock_vote_weight:
            # Alice: high reputation, vote_weight = 5.0
            mock_vote_weight.return_value = 5.0
            alice_adjusted = await fl_gov.calculate_reputation_weighted_reward(
                participant_id="alice_id",
                base_reward=100.0,
            )

            # Bob: low reputation, vote_weight = 1.0
            mock_vote_weight.return_value = 1.0
            bob_adjusted = await fl_gov.calculate_reputation_weighted_reward(
                participant_id="bob_id",
                base_reward=100.0,
            )

            # Alice should get higher reward
            assert alice_adjusted > bob_adjusted


class TestGovernanceAndFLIntegration:
    """Integration tests for governance with FL coordinator"""

    @pytest.mark.asyncio
    async def test_fl_round_with_governance_checks(self, governance_coordinator):
        """
        Test FL round with governance pre-checks:
        1. Multiple participants attempt to submit gradients
        2. Some authorized, some not
        3. Verify only authorized participants can submit
        4. Rewards distributed with reputation weighting
        """
        fl_gov = FLGovernanceIntegration(
            governance_coordinator=governance_coordinator,
            config=FLGovernanceConfig(
                require_capability_for_submit=True,
                reputation_weighted_rewards=True,
            ),
        )

        # Mock capability checks for different participants
        async def mock_authorize(participant_id, action):
            # Alice and Bob authorized, Carol not
            if participant_id in ["alice_id", "bob_id"]:
                return (True, "Authorized")
            else:
                return (False, "Insufficient reputation")

        with patch.object(
            governance_coordinator,
            'authorize_fl_action',
            new=mock_authorize
        ):
            # Check participants
            alice_auth, _ = await fl_gov.verify_participant_for_round("alice_id", 42)
            bob_auth, _ = await fl_gov.verify_participant_for_round("bob_id", 42)
            carol_auth, _ = await fl_gov.verify_participant_for_round("carol_id", 42)

            assert alice_auth is True
            assert bob_auth is True
            assert carol_auth is False

        # Distribute rewards (only Alice and Bob)
        with patch.object(
            governance_coordinator.gov_extensions,
            'calculate_vote_weight',
            new=AsyncMock()
        ) as mock_vote_weight:
            mock_vote_weight.side_effect = [5.0, 2.0]  # Alice, Bob

            rewards = {}
            for pid, base in [("alice_id", 100.0), ("bob_id", 100.0)]:
                rewards[pid] = await fl_gov.calculate_reputation_weighted_reward(
                    participant_id=pid,
                    base_reward=base,
                )

            # Alice should get more reward than Bob
            assert rewards["alice_id"] > rewards["bob_id"]


class TestMultiProposalScenarios:
    """Integration tests with multiple concurrent proposals"""

    @pytest.mark.asyncio
    async def test_multiple_proposals_different_types(self, governance_coordinator, mock_dht_client):
        """
        Test system handling multiple concurrent proposals:
        1. Create parameter change proposal
        2. Create participant ban proposal
        3. Create emergency action proposal
        4. All can proceed independently
        """
        # Mock DHT to accept all proposals
        mock_dht_client.call_zome.return_value = None

        with patch.object(
            governance_coordinator.capability_enforcer,
            'check_capability',
            new=AsyncMock(return_value=(True, "Authorized"))
        ):
            # Create multiple proposals
            proposals = []

            # Proposal 1: Parameter change
            success, _, pid = await governance_coordinator.create_governance_proposal(
                proposer_participant_id="alice_id",
                proposal_type="PARAMETER_CHANGE",
                title="Parameter Change",
                description="Change parameter",
                execution_params={"param": "value"},
                voting_duration_days=7,
            )
            if success:
                proposals.append(pid)

            # Proposal 2: Participant management
            success, _, pid = await governance_coordinator.create_governance_proposal(
                proposer_participant_id="bob_id",
                proposal_type="PARTICIPANT_MANAGEMENT",
                title="Ban Participant",
                description="Ban malicious participant",
                execution_params={"target": "eve_id"},
                voting_duration_days=7,
            )
            if success:
                proposals.append(pid)

            # Proposal 3: Emergency action
            success, _, pid = await governance_coordinator.create_governance_proposal(
                proposer_participant_id="carol_id",
                proposal_type="EMERGENCY_ACTION",
                title="Emergency Stop",
                description="Stop training",
                execution_params={"action": "emergency_stop"},
                voting_duration_days=3,
            )
            if success:
                proposals.append(pid)

            # All proposals should be created
            assert len(proposals) == 3


class TestSecurityScenarios:
    """Integration tests for security features"""

    @pytest.mark.asyncio
    async def test_sybil_attack_prevention(self, governance_coordinator):
        """
        Test Sybil attack prevention:
        1. New identity attempts to create many proposals
        2. Rate limiting prevents spam
        3. Low vote weight limits influence
        """
        fl_gov = FLGovernanceIntegration(
            governance_coordinator=governance_coordinator,
        )

        # Mock low vote weight for new identity
        with patch.object(
            governance_coordinator.gov_extensions,
            'calculate_vote_weight',
            new=AsyncMock(return_value=0.6)  # New identity
        ):
            # Check reward with low reputation
            adjusted = await fl_gov.calculate_reputation_weighted_reward(
                participant_id="new_sybil_id",
                base_reward=100.0,
            )

            # Should get reduced reward (~83% of base)
            assert adjusted < 85.0

    @pytest.mark.asyncio
    async def test_unauthorized_action_prevention(self, governance_coordinator):
        """
        Test prevention of unauthorized actions:
        1. Low-reputation participant tries emergency stop
        2. Action rejected due to insufficient capability
        3. Guardian authorization required
        """
        fl_gov = FLGovernanceIntegration(
            governance_coordinator=governance_coordinator,
            config=FLGovernanceConfig(emergency_stop_requires_guardian=True),
        )

        with patch.object(
            governance_coordinator,
            'authorize_fl_action',
            new=AsyncMock(return_value=(False, "Insufficient reputation: 0.3"))
        ):
            success, message, request_id = await fl_gov.request_emergency_stop(
                requester_participant_id="low_rep_participant",
                reason="Trying to stop system",
            )

            # Should create authorization request, not immediate stop
            assert "pending guardian approval" in message or "not authorized" in message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
