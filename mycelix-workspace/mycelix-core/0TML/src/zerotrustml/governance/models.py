# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Governance Data Models
Week 7-8 Phase 3: Governance Coordinator

Data structures for proposals, votes, and authorization requests.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
import time


class ProposalType(Enum):
    """Types of governance proposals"""
    PARAMETER_CHANGE = "PARAMETER_CHANGE"
    PARTICIPANT_MANAGEMENT = "PARTICIPANT_MANAGEMENT"
    EMERGENCY_ACTION = "EMERGENCY_ACTION"
    PROTOCOL_UPGRADE = "PROTOCOL_UPGRADE"
    CUSTOM = "CUSTOM"


class ProposalStatus(Enum):
    """Proposal lifecycle status"""
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    VOTING = "VOTING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


class VoteChoice(Enum):
    """Vote options"""
    FOR = "FOR"
    AGAINST = "AGAINST"
    ABSTAIN = "ABSTAIN"


class AuthorizationStatus(Enum):
    """Guardian authorization request status"""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class ProposalData:
    """
    Governance proposal data structure

    Represents a proposal to change the system, add/remove participants,
    or execute emergency actions.
    """
    proposal_id: str
    proposal_type: ProposalType
    title: str
    description: str
    proposer_did: str
    proposer_participant_id: str

    # Voting parameters
    voting_start: int  # Unix timestamp
    voting_end: int    # Unix timestamp
    quorum: float      # 0.0-1.0
    approval_threshold: float  # 0.5-1.0

    # Current status
    status: ProposalStatus = ProposalStatus.DRAFT
    total_votes_for: float = 0.0
    total_votes_against: float = 0.0
    total_votes_abstain: float = 0.0
    total_voting_power: float = 0.0

    # Execution
    execution_params: Dict[str, Any] = field(default_factory=dict)
    executed_at: Optional[int] = None
    execution_result: str = ""

    # Metadata
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DHT storage"""
        import json
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type.value,
            "title": self.title,
            "description": self.description,
            "proposer_did": self.proposer_did,
            "proposer_participant_id": self.proposer_participant_id,
            "voting_start": self.voting_start,
            "voting_end": self.voting_end,
            "quorum": self.quorum,
            "approval_threshold": self.approval_threshold,
            "status": self.status.value,
            "total_votes_for": self.total_votes_for,
            "total_votes_against": self.total_votes_against,
            "total_votes_abstain": self.total_votes_abstain,
            "total_voting_power": self.total_voting_power,
            "execution_params": json.dumps(self.execution_params),
            "executed_at": self.executed_at,
            "execution_result": self.execution_result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": json.dumps(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProposalData":
        """Create from dictionary (DHT retrieval)"""
        import json
        return cls(
            proposal_id=data["proposal_id"],
            proposal_type=ProposalType(data["proposal_type"]),
            title=data["title"],
            description=data["description"],
            proposer_did=data["proposer_did"],
            proposer_participant_id=data["proposer_participant_id"],
            voting_start=data["voting_start"],
            voting_end=data["voting_end"],
            quorum=data["quorum"],
            approval_threshold=data["approval_threshold"],
            status=ProposalStatus(data.get("status", "DRAFT")),
            total_votes_for=data.get("total_votes_for", 0.0),
            total_votes_against=data.get("total_votes_against", 0.0),
            total_votes_abstain=data.get("total_votes_abstain", 0.0),
            total_voting_power=data.get("total_voting_power", 0.0),
            execution_params=json.loads(data.get("execution_params", "{}")),
            executed_at=data.get("executed_at"),
            execution_result=data.get("execution_result", ""),
            created_at=data.get("created_at", int(time.time())),
            updated_at=data.get("updated_at", int(time.time())),
            tags=json.loads(data.get("tags", "[]")),
        )


@dataclass
class VoteData:
    """
    Vote record data structure

    Represents a single vote on a proposal.
    """
    vote_id: str
    proposal_id: str
    voter_did: str
    voter_participant_id: str
    choice: VoteChoice
    credits_spent: int
    vote_weight: float
    effective_votes: float
    timestamp: int = field(default_factory=lambda: int(time.time()))
    signature: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DHT storage"""
        return {
            "vote_id": self.vote_id,
            "proposal_id": self.proposal_id,
            "voter_did": self.voter_did,
            "voter_participant_id": self.voter_participant_id,
            "choice": self.choice.value,
            "credits_spent": self.credits_spent,
            "vote_weight": self.vote_weight,
            "effective_votes": self.effective_votes,
            "timestamp": self.timestamp,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoteData":
        """Create from dictionary (DHT retrieval)"""
        return cls(
            vote_id=data["vote_id"],
            proposal_id=data["proposal_id"],
            voter_did=data["voter_did"],
            voter_participant_id=data["voter_participant_id"],
            choice=VoteChoice(data["choice"]),
            credits_spent=data["credits_spent"],
            vote_weight=data["vote_weight"],
            effective_votes=data["effective_votes"],
            timestamp=data.get("timestamp", int(time.time())),
            signature=data.get("signature", ""),
        )


@dataclass
class AuthorizationRequestData:
    """
    Guardian authorization request data structure

    Represents a request for guardian approval of an emergency action.
    """
    request_id: str
    subject_participant_id: str
    action: str
    action_params: Dict[str, Any]
    required_threshold: float
    expires_at: int
    status: AuthorizationStatus = AuthorizationStatus.PENDING
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DHT storage"""
        import json
        return {
            "request_id": self.request_id,
            "subject_participant_id": self.subject_participant_id,
            "action": self.action,
            "action_params": json.dumps(self.action_params),
            "required_threshold": self.required_threshold,
            "expires_at": self.expires_at,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthorizationRequestData":
        """Create from dictionary (DHT retrieval)"""
        import json
        return cls(
            request_id=data["request_id"],
            subject_participant_id=data["subject_participant_id"],
            action=data["action"],
            action_params=json.loads(data.get("action_params", "{}")),
            required_threshold=data["required_threshold"],
            expires_at=data["expires_at"],
            status=AuthorizationStatus(data.get("status", "PENDING")),
            created_at=data.get("created_at", int(time.time())),
            updated_at=data.get("updated_at", int(time.time())),
        )


@dataclass
class GuardianApprovalData:
    """
    Guardian approval record data structure

    Represents a guardian's approval or rejection of an authorization request.
    """
    approval_id: str
    request_id: str
    guardian_did: str
    guardian_participant_id: str
    approved: bool
    reasoning: str
    timestamp: int = field(default_factory=lambda: int(time.time()))
    signature: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DHT storage"""
        return {
            "approval_id": self.approval_id,
            "request_id": self.request_id,
            "guardian_did": self.guardian_did,
            "guardian_participant_id": self.guardian_participant_id,
            "approved": self.approved,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GuardianApprovalData":
        """Create from dictionary (DHT retrieval)"""
        return cls(
            approval_id=data["approval_id"],
            request_id=data["request_id"],
            guardian_did=data["guardian_did"],
            guardian_participant_id=data["guardian_participant_id"],
            approved=data["approved"],
            reasoning=data.get("reasoning", ""),
            timestamp=data.get("timestamp", int(time.time())),
            signature=data.get("signature", ""),
        )


@dataclass
class ProposalSignature:
    """
    Ed25519 cryptographic signature for a governance proposal.

    Ensures proposal authenticity and non-repudiation.
    """
    proposal_id: str
    proposer_did: str
    proposer_public_key: str  # Hex-encoded Ed25519 public key (32 bytes)
    signature: str  # Hex-encoded Ed25519 signature (64 bytes)
    signed_at: int = field(default_factory=lambda: int(time.time()))

    # Optional fields for signature metadata
    algorithm: str = "Ed25519"
    key_id: str = ""  # Optional key identifier for key rotation support

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DHT storage"""
        return {
            "proposal_id": self.proposal_id,
            "proposer_did": self.proposer_did,
            "proposer_public_key": self.proposer_public_key,
            "signature": self.signature,
            "signed_at": self.signed_at,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProposalSignature":
        """Create from dictionary (DHT retrieval)"""
        return cls(
            proposal_id=data["proposal_id"],
            proposer_did=data["proposer_did"],
            proposer_public_key=data["proposer_public_key"],
            signature=data["signature"],
            signed_at=data.get("signed_at", int(time.time())),
            algorithm=data.get("algorithm", "Ed25519"),
            key_id=data.get("key_id", ""),
        )

    def get_signable_data(self) -> Dict[str, Any]:
        """
        Get the canonical data that was signed.

        Returns a deterministic representation suitable for verification.
        """
        return {
            "proposal_id": self.proposal_id,
            "proposer_did": self.proposer_did,
            "signed_at": self.signed_at,
        }


@dataclass
class GuardianRelationship:
    """
    Represents a guardian relationship for voting endorsement.

    Tracks guardian relationships with weights for vote tallying.
    """
    subject_participant_id: str
    guardian_participant_id: str
    relationship_type: str  # "RECOVERY", "ENDORSEMENT", "DELEGATION"
    weight: float  # Guardian influence (0.0-1.0)
    assurance_level: str  # Guardian's assurance level at time of relationship
    created_at: int = field(default_factory=lambda: int(time.time()))
    expires_at: Optional[int] = None
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DHT storage"""
        return {
            "subject_participant_id": self.subject_participant_id,
            "guardian_participant_id": self.guardian_participant_id,
            "relationship_type": self.relationship_type,
            "weight": self.weight,
            "assurance_level": self.assurance_level,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GuardianRelationship":
        """Create from dictionary (DHT retrieval)"""
        return cls(
            subject_participant_id=data["subject_participant_id"],
            guardian_participant_id=data["guardian_participant_id"],
            relationship_type=data["relationship_type"],
            weight=data["weight"],
            assurance_level=data["assurance_level"],
            created_at=data.get("created_at", int(time.time())),
            expires_at=data.get("expires_at"),
            is_active=data.get("is_active", True),
        )

    def is_valid(self) -> bool:
        """Check if guardian relationship is currently valid"""
        if not self.is_active:
            return False
        if self.expires_at is not None and self.expires_at < int(time.time()):
            return False
        return True


@dataclass
class VotingPowerCalculation:
    """
    Detailed breakdown of a participant's voting power.

    Captures all factors contributing to voting weight for transparency.
    """
    participant_id: str
    base_weight: float = 1.0
    assurance_multiplier: float = 1.0
    reputation_multiplier: float = 1.0
    sybil_bonus: float = 0.0
    guardian_endorsements: float = 0.0
    time_weight: float = 1.0  # Activity recency weighting
    total_voting_power: float = 0.0
    calculated_at: int = field(default_factory=lambda: int(time.time()))

    # Breakdown details
    assurance_level: str = "E0"
    reputation_score: float = 0.0
    sybil_resistance: float = 0.0
    endorsing_guardians: List[str] = field(default_factory=list)
    last_activity_timestamp: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "participant_id": self.participant_id,
            "base_weight": self.base_weight,
            "assurance_multiplier": self.assurance_multiplier,
            "reputation_multiplier": self.reputation_multiplier,
            "sybil_bonus": self.sybil_bonus,
            "guardian_endorsements": self.guardian_endorsements,
            "time_weight": self.time_weight,
            "total_voting_power": self.total_voting_power,
            "calculated_at": self.calculated_at,
            "assurance_level": self.assurance_level,
            "reputation_score": self.reputation_score,
            "sybil_resistance": self.sybil_resistance,
            "endorsing_guardians": self.endorsing_guardians,
            "last_activity_timestamp": self.last_activity_timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VotingPowerCalculation":
        """Create from dictionary"""
        return cls(
            participant_id=data["participant_id"],
            base_weight=data.get("base_weight", 1.0),
            assurance_multiplier=data.get("assurance_multiplier", 1.0),
            reputation_multiplier=data.get("reputation_multiplier", 1.0),
            sybil_bonus=data.get("sybil_bonus", 0.0),
            guardian_endorsements=data.get("guardian_endorsements", 0.0),
            time_weight=data.get("time_weight", 1.0),
            total_voting_power=data.get("total_voting_power", 0.0),
            calculated_at=data.get("calculated_at", int(time.time())),
            assurance_level=data.get("assurance_level", "E0"),
            reputation_score=data.get("reputation_score", 0.0),
            sybil_resistance=data.get("sybil_resistance", 0.0),
            endorsing_guardians=data.get("endorsing_guardians", []),
            last_activity_timestamp=data.get("last_activity_timestamp", 0),
        )


@dataclass
class VoteTally:
    """
    Complete vote tally for a proposal with all validation details.
    """
    proposal_id: str
    total_votes_for: float = 0.0
    total_votes_against: float = 0.0
    total_votes_abstain: float = 0.0
    total_voting_power_cast: float = 0.0
    total_eligible_voting_power: float = 0.0
    unique_voters: int = 0

    # Quorum and threshold tracking
    quorum_required: float = 0.5
    quorum_achieved: float = 0.0
    quorum_met: bool = False
    approval_threshold: float = 0.6
    approval_ratio: float = 0.0
    approved: bool = False

    # Timing
    voting_start: int = 0
    voting_end: int = 0
    tallied_at: int = field(default_factory=lambda: int(time.time()))
    is_finalized: bool = False

    # Verification
    verified_signatures: int = 0
    invalid_signatures: int = 0
    guardian_verified_votes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "proposal_id": self.proposal_id,
            "total_votes_for": self.total_votes_for,
            "total_votes_against": self.total_votes_against,
            "total_votes_abstain": self.total_votes_abstain,
            "total_voting_power_cast": self.total_voting_power_cast,
            "total_eligible_voting_power": self.total_eligible_voting_power,
            "unique_voters": self.unique_voters,
            "quorum_required": self.quorum_required,
            "quorum_achieved": self.quorum_achieved,
            "quorum_met": self.quorum_met,
            "approval_threshold": self.approval_threshold,
            "approval_ratio": self.approval_ratio,
            "approved": self.approved,
            "voting_start": self.voting_start,
            "voting_end": self.voting_end,
            "tallied_at": self.tallied_at,
            "is_finalized": self.is_finalized,
            "verified_signatures": self.verified_signatures,
            "invalid_signatures": self.invalid_signatures,
            "guardian_verified_votes": self.guardian_verified_votes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoteTally":
        """Create from dictionary"""
        return cls(
            proposal_id=data["proposal_id"],
            total_votes_for=data.get("total_votes_for", 0.0),
            total_votes_against=data.get("total_votes_against", 0.0),
            total_votes_abstain=data.get("total_votes_abstain", 0.0),
            total_voting_power_cast=data.get("total_voting_power_cast", 0.0),
            total_eligible_voting_power=data.get("total_eligible_voting_power", 0.0),
            unique_voters=data.get("unique_voters", 0),
            quorum_required=data.get("quorum_required", 0.5),
            quorum_achieved=data.get("quorum_achieved", 0.0),
            quorum_met=data.get("quorum_met", False),
            approval_threshold=data.get("approval_threshold", 0.6),
            approval_ratio=data.get("approval_ratio", 0.0),
            approved=data.get("approved", False),
            voting_start=data.get("voting_start", 0),
            voting_end=data.get("voting_end", 0),
            tallied_at=data.get("tallied_at", int(time.time())),
            is_finalized=data.get("is_finalized", False),
            verified_signatures=data.get("verified_signatures", 0),
            invalid_signatures=data.get("invalid_signatures", 0),
            guardian_verified_votes=data.get("guardian_verified_votes", 0),
        )
