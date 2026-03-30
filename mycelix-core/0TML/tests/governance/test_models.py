# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Unit Tests for Governance Data Models
Week 7-8 Phase 5: Testing & Validation

Tests for ProposalData, VoteData, AuthorizationRequestData, GuardianApprovalData
"""

import pytest
import time
import json
from typing import Dict, Any

from zerotrustml.governance.models import (
    ProposalType,
    ProposalStatus,
    VoteChoice,
    AuthorizationStatus,
    ProposalData,
    VoteData,
    AuthorizationRequestData,
    GuardianApprovalData,
)


class TestProposalData:
    """Tests for ProposalData model"""

    def test_create_proposal_data(self):
        """Test creating a proposal"""
        proposal = ProposalData(
            proposal_id="prop_test123",
            proposal_type=ProposalType.PARAMETER_CHANGE,
            title="Test Proposal",
            description="A test proposal for unit testing",
            proposer_did="did:mycelix:alice",
            proposer_participant_id="alice_id",
            voting_start=int(time.time()) + 3600,
            voting_end=int(time.time()) + 7200,
            quorum=0.5,
            approval_threshold=0.66,
        )

        assert proposal.proposal_id == "prop_test123"
        assert proposal.proposal_type == ProposalType.PARAMETER_CHANGE
        assert proposal.status == ProposalStatus.DRAFT
        assert proposal.total_votes_for == 0.0
        assert proposal.total_votes_against == 0.0
        assert proposal.quorum == 0.5
        assert proposal.approval_threshold == 0.66

    def test_proposal_to_dict(self):
        """Test serialization to dictionary"""
        proposal = ProposalData(
            proposal_id="prop_test123",
            proposal_type=ProposalType.EMERGENCY_ACTION,
            title="Emergency Stop",
            description="Emergency stop request",
            proposer_did="did:mycelix:alice",
            proposer_participant_id="alice_id",
            voting_start=1699747200,
            voting_end=1699833600,
            quorum=0.4,
            approval_threshold=0.7,
            execution_params={"action": "emergency_stop"},
            tags=["emergency", "security"],
        )

        data = proposal.to_dict()

        assert data["proposal_id"] == "prop_test123"
        assert data["proposal_type"] == "EMERGENCY_ACTION"
        assert data["title"] == "Emergency Stop"
        assert data["quorum"] == 0.4
        assert data["approval_threshold"] == 0.7
        assert json.loads(data["execution_params"]) == {"action": "emergency_stop"}
        assert json.loads(data["tags"]) == ["emergency", "security"]

    def test_proposal_from_dict(self):
        """Test deserialization from dictionary"""
        data = {
            "proposal_id": "prop_test456",
            "proposal_type": "PARTICIPANT_MANAGEMENT",
            "title": "Ban Participant",
            "description": "Ban malicious participant",
            "proposer_did": "did:mycelix:bob",
            "proposer_participant_id": "bob_id",
            "voting_start": 1699747200,
            "voting_end": 1699833600,
            "quorum": 0.5,
            "approval_threshold": 0.66,
            "status": "VOTING",
            "total_votes_for": 150.5,
            "total_votes_against": 48.2,
            "total_votes_abstain": 10.0,
            "total_voting_power": 208.7,
            "execution_params": json.dumps({"target": "eve_id"}),
            "executed_at": None,
            "execution_result": "",
            "created_at": 1699747100,
            "updated_at": 1699747150,
            "tags": json.dumps(["security"]),
        }

        proposal = ProposalData.from_dict(data)

        assert proposal.proposal_id == "prop_test456"
        assert proposal.proposal_type == ProposalType.PARTICIPANT_MANAGEMENT
        assert proposal.status == ProposalStatus.VOTING
        assert proposal.total_votes_for == 150.5
        assert proposal.total_votes_against == 48.2
        assert proposal.execution_params == {"target": "eve_id"}
        assert proposal.tags == ["security"]

    def test_proposal_round_trip(self):
        """Test serialization round-trip"""
        original = ProposalData(
            proposal_id="prop_roundtrip",
            proposal_type=ProposalType.PROTOCOL_UPGRADE,
            title="Protocol Upgrade v2.0",
            description="Upgrade to new protocol version",
            proposer_did="did:mycelix:system",
            proposer_participant_id="system_id",
            voting_start=1699747200,
            voting_end=1699919999,
            quorum=0.6,
            approval_threshold=0.75,
            execution_params={"version": "2.0.0", "breaking": True},
            tags=["protocol", "upgrade"],
        )

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = ProposalData.from_dict(data)

        # Compare
        assert restored.proposal_id == original.proposal_id
        assert restored.proposal_type == original.proposal_type
        assert restored.title == original.title
        assert restored.description == original.description
        assert restored.execution_params == original.execution_params
        assert restored.tags == original.tags


class TestVoteData:
    """Tests for VoteData model"""

    def test_create_vote_data(self):
        """Test creating a vote"""
        vote = VoteData(
            vote_id="vote_test123",
            proposal_id="prop_test123",
            voter_did="did:mycelix:alice",
            voter_participant_id="alice_id",
            choice=VoteChoice.FOR,
            credits_spent=100,
            vote_weight=3.5,
            effective_votes=35.0,
            signature="sig_abc123",
        )

        assert vote.vote_id == "vote_test123"
        assert vote.proposal_id == "prop_test123"
        assert vote.choice == VoteChoice.FOR
        assert vote.credits_spent == 100
        assert vote.vote_weight == 3.5
        assert vote.effective_votes == 35.0

    def test_vote_to_dict(self):
        """Test vote serialization"""
        vote = VoteData(
            vote_id="vote_test456",
            proposal_id="prop_test456",
            voter_did="did:mycelix:bob",
            voter_participant_id="bob_id",
            choice=VoteChoice.AGAINST,
            credits_spent=50,
            vote_weight=2.0,
            effective_votes=14.14,
            timestamp=1699747200,
            signature="sig_def456",
        )

        data = vote.to_dict()

        assert data["vote_id"] == "vote_test456"
        assert data["choice"] == "AGAINST"
        assert data["credits_spent"] == 50
        assert data["vote_weight"] == 2.0
        assert data["effective_votes"] == 14.14
        assert data["timestamp"] == 1699747200

    def test_vote_from_dict(self):
        """Test vote deserialization"""
        data = {
            "vote_id": "vote_test789",
            "proposal_id": "prop_test789",
            "voter_did": "did:mycelix:carol",
            "voter_participant_id": "carol_id",
            "choice": "ABSTAIN",
            "credits_spent": 25,
            "vote_weight": 1.5,
            "effective_votes": 7.5,
            "timestamp": 1699747300,
            "signature": "sig_ghi789",
        }

        vote = VoteData.from_dict(data)

        assert vote.vote_id == "vote_test789"
        assert vote.choice == VoteChoice.ABSTAIN
        assert vote.credits_spent == 25
        assert vote.vote_weight == 1.5

    def test_vote_choice_enum(self):
        """Test VoteChoice enum values"""
        assert VoteChoice.FOR.value == "FOR"
        assert VoteChoice.AGAINST.value == "AGAINST"
        assert VoteChoice.ABSTAIN.value == "ABSTAIN"


class TestAuthorizationRequestData:
    """Tests for AuthorizationRequestData model"""

    def test_create_authorization_request(self):
        """Test creating authorization request"""
        request = AuthorizationRequestData(
            request_id="auth_test123",
            subject_participant_id="alice_id",
            action="emergency_stop",
            action_params={"reason": "Byzantine attack"},
            required_threshold=0.7,
            expires_at=int(time.time()) + 3600,
        )

        assert request.request_id == "auth_test123"
        assert request.subject_participant_id == "alice_id"
        assert request.action == "emergency_stop"
        assert request.required_threshold == 0.7
        assert request.status == AuthorizationStatus.PENDING

    def test_authorization_request_to_dict(self):
        """Test authorization request serialization"""
        request = AuthorizationRequestData(
            request_id="auth_test456",
            subject_participant_id="bob_id",
            action="ban_participant",
            action_params={"target": "eve_id", "permanent": False},
            required_threshold=0.8,
            expires_at=1699833600,
            status=AuthorizationStatus.APPROVED,
            created_at=1699747200,
            updated_at=1699750800,
        )

        data = request.to_dict()

        assert data["request_id"] == "auth_test456"
        assert data["action"] == "ban_participant"
        assert json.loads(data["action_params"]) == {"target": "eve_id", "permanent": False}
        assert data["required_threshold"] == 0.8
        assert data["status"] == "APPROVED"

    def test_authorization_request_from_dict(self):
        """Test authorization request deserialization"""
        data = {
            "request_id": "auth_test789",
            "subject_participant_id": "carol_id",
            "action": "change_parameters",
            "action_params": json.dumps({"param": "min_reputation", "value": 0.7}),
            "required_threshold": 0.75,
            "expires_at": 1699833600,
            "status": "REJECTED",
            "created_at": 1699747200,
            "updated_at": 1699751000,
        }

        request = AuthorizationRequestData.from_dict(data)

        assert request.request_id == "auth_test789"
        assert request.status == AuthorizationStatus.REJECTED
        assert request.action_params == {"param": "min_reputation", "value": 0.7}


class TestGuardianApprovalData:
    """Tests for GuardianApprovalData model"""

    def test_create_guardian_approval(self):
        """Test creating guardian approval"""
        approval = GuardianApprovalData(
            approval_id="approval_test123",
            request_id="auth_test123",
            guardian_did="did:mycelix:guardian_alice",
            guardian_participant_id="guardian_alice_id",
            approved=True,
            reasoning="Evidence is conclusive",
            signature="sig_xyz123",
        )

        assert approval.approval_id == "approval_test123"
        assert approval.request_id == "auth_test123"
        assert approval.approved is True
        assert approval.reasoning == "Evidence is conclusive"

    def test_guardian_approval_to_dict(self):
        """Test guardian approval serialization"""
        approval = GuardianApprovalData(
            approval_id="approval_test456",
            request_id="auth_test456",
            guardian_did="did:mycelix:guardian_bob",
            guardian_participant_id="guardian_bob_id",
            approved=False,
            reasoning="Need more investigation",
            timestamp=1699747200,
            signature="sig_uvw456",
        )

        data = approval.to_dict()

        assert data["approval_id"] == "approval_test456"
        assert data["approved"] is False
        assert data["reasoning"] == "Need more investigation"
        assert data["timestamp"] == 1699747200

    def test_guardian_approval_from_dict(self):
        """Test guardian approval deserialization"""
        data = {
            "approval_id": "approval_test789",
            "request_id": "auth_test789",
            "guardian_did": "did:mycelix:guardian_carol",
            "guardian_participant_id": "guardian_carol_id",
            "approved": True,
            "reasoning": "Approved with reservations",
            "timestamp": 1699747300,
            "signature": "sig_rst789",
        }

        approval = GuardianApprovalData.from_dict(data)

        assert approval.approval_id == "approval_test789"
        assert approval.approved is True
        assert approval.reasoning == "Approved with reservations"


class TestEnums:
    """Tests for governance enums"""

    def test_proposal_type_enum(self):
        """Test ProposalType enum"""
        assert ProposalType.PARAMETER_CHANGE.value == "PARAMETER_CHANGE"
        assert ProposalType.PARTICIPANT_MANAGEMENT.value == "PARTICIPANT_MANAGEMENT"
        assert ProposalType.EMERGENCY_ACTION.value == "EMERGENCY_ACTION"
        assert ProposalType.PROTOCOL_UPGRADE.value == "PROTOCOL_UPGRADE"
        assert ProposalType.CUSTOM.value == "CUSTOM"

    def test_proposal_status_enum(self):
        """Test ProposalStatus enum"""
        assert ProposalStatus.DRAFT.value == "DRAFT"
        assert ProposalStatus.SUBMITTED.value == "SUBMITTED"
        assert ProposalStatus.VOTING.value == "VOTING"
        assert ProposalStatus.APPROVED.value == "APPROVED"
        assert ProposalStatus.REJECTED.value == "REJECTED"
        assert ProposalStatus.EXECUTED.value == "EXECUTED"
        assert ProposalStatus.FAILED.value == "FAILED"

    def test_authorization_status_enum(self):
        """Test AuthorizationStatus enum"""
        assert AuthorizationStatus.PENDING.value == "PENDING"
        assert AuthorizationStatus.APPROVED.value == "APPROVED"
        assert AuthorizationStatus.REJECTED.value == "REJECTED"
        assert AuthorizationStatus.EXPIRED.value == "EXPIRED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
