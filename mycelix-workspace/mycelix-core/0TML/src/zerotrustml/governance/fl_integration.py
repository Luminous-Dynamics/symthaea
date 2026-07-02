# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
FL Integration for Governance System
Week 7-8 Phase 4: FL Integration

Integrates identity-gated governance with the FL coordinator:
- Pre-round capability checks
- Parameter change proposal execution
- Emergency stop integration
- Participant ban/unban via governance
- Reputation-based rewards
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from zerotrustml.core.phase10_coordinator import Phase10Coordinator
from dataclasses import dataclass, field

from .coordinator import GovernanceCoordinator
from .models import ProposalType, ProposalStatus

# Import DHT client for storing governance records
try:
    from zerotrustml.holochain.identity_dht_client import IdentityDHTClient
    DHT_AVAILABLE = True
except ImportError:
    IdentityDHTClient = None
    DHT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FLGovernanceConfig:
    """Configuration for FL governance integration"""

    # Capability requirements for FL actions
    require_capability_for_submit: bool = True
    require_capability_for_request: bool = True

    # Emergency stop configuration
    emergency_stop_enabled: bool = True
    emergency_stop_requires_guardian: bool = True

    # Participant management
    ban_requires_guardian: bool = True
    ban_requires_proposal: bool = True

    # Parameter changes
    parameter_change_requires_proposal: bool = True
    critical_parameters: List[str] = field(default_factory=lambda: [
        "min_reputation",
        "byzantine_threshold",
        "aggregation_strategy",
        "quorum_size",
    ])

    # Reward distribution
    reputation_weighted_rewards: bool = True
    governance_participation_bonus: float = 1.2  # 20% bonus for governance participation


class FLGovernanceIntegration:
    """
    FL Governance Integration Layer

    Responsibilities:
    - Pre-round capability verification
    - Emergency action coordination
    - Proposal execution
    - Reputation-weighted rewards
    """

    def __init__(
        self,
        governance_coordinator: GovernanceCoordinator,
        config: Optional[FLGovernanceConfig] = None,
        fl_coordinator: Optional[Any] = None,
        dht_client: Optional[Any] = None,
    ):
        """
        Args:
            governance_coordinator: Governance coordinator instance
            config: FL governance configuration
            fl_coordinator: Optional reference to Phase10Coordinator for parameter updates
            dht_client: Optional Holochain DHT client for storing governance records
        """
        self.gov_coord = governance_coordinator
        self.config = config or FLGovernanceConfig()

        # Reference to FL coordinator for parameter application
        self._fl_coordinator = fl_coordinator

        # DHT client for storing governance records (emergency stops, bans, etc.)
        self._dht_client = dht_client

        # FL state tracking
        self.emergency_stopped = False
        self.banned_participants: Dict[str, Dict[str, Any]] = {}
        self.fl_parameters: Dict[str, Any] = {}

        # Governance participation tracking
        self._participation_records: Dict[str, Dict[str, Any]] = {}

        logger.info("FLGovernanceIntegration initialized")

    def set_fl_coordinator(self, fl_coordinator: Any) -> None:
        """
        Set the FL coordinator reference for parameter updates.

        Args:
            fl_coordinator: Phase10Coordinator instance
        """
        self._fl_coordinator = fl_coordinator
        logger.info("FL coordinator reference set for governance integration")

    def set_dht_client(self, dht_client: Any) -> None:
        """
        Set the DHT client for storing governance records.

        Args:
            dht_client: Holochain DHT client instance
        """
        self._dht_client = dht_client
        logger.info("DHT client set for governance record storage")

    # Pre-Round Capability Checks

    async def verify_participant_for_round(
        self,
        participant_id: str,
        round_number: int,
    ) -> Tuple[bool, str]:
        """
        Verify participant can participate in FL round

        Checks:
        1. Not banned
        2. Has submit_update capability
        3. Emergency stop not active
        4. Identity meets minimum requirements

        Args:
            participant_id: Participant to verify
            round_number: Current round number

        Returns:
            Tuple of (authorized, reason)
        """
        # Check 1: Emergency stop
        if self.emergency_stopped:
            return False, "Emergency stop active"

        # Check 2: Participant banned
        if participant_id in self.banned_participants:
            ban_info = self.banned_participants[participant_id]
            if ban_info.get("permanent", False):
                return False, "Participant permanently banned"

            # Check temporary ban expiration
            ban_until = ban_info.get("until", 0)
            if time.time() < ban_until:
                return False, f"Participant banned until {ban_until}"
            else:
                # Ban expired, remove from list
                del self.banned_participants[participant_id]

        # Check 3: Capability requirement
        if self.config.require_capability_for_submit:
            authorized, reason = await self.gov_coord.authorize_fl_action(
                participant_id=participant_id,
                action="submit_update",
            )

            if not authorized:
                return False, f"Capability check failed: {reason}"

        # Check 4: Governance participation (optional reputation boost)
        if self.config.reputation_weighted_rewards:
            # Participant is authorized and will get reputation-weighted rewards
            pass

        return True, "Authorized"

    async def verify_model_request(
        self,
        participant_id: str,
    ) -> Tuple[bool, str]:
        """
        Verify participant can request global model

        Args:
            participant_id: Participant requesting model

        Returns:
            Tuple of (authorized, reason)
        """
        # Check emergency stop
        if self.emergency_stopped:
            return False, "Emergency stop active"

        # Check banned
        if participant_id in self.banned_participants:
            return False, "Participant banned"

        # Check capability
        if self.config.require_capability_for_request:
            return await self.gov_coord.authorize_fl_action(
                participant_id=participant_id,
                action="request_model",
            )

        return True, "Authorized"

    # Emergency Actions

    async def request_emergency_stop(
        self,
        requester_participant_id: str,
        reason: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Request emergency stop of FL training

        If guardian authorization required, creates authorization request.
        Otherwise, executes immediately (if participant has capability).

        Args:
            requester_participant_id: Participant requesting stop
            reason: Reason for emergency stop
            evidence: Optional evidence supporting request

        Returns:
            Tuple of (success, message, request_id or None)
        """
        # Check capability
        authorized, auth_reason = await self.gov_coord.authorize_fl_action(
            participant_id=requester_participant_id,
            action="emergency_stop",
        )

        if not authorized:
            # Check if guardian approval required
            if self.config.emergency_stop_requires_guardian:
                # Create authorization request
                success, message, request_id = await self.gov_coord.guardian_auth_mgr.request_authorization(
                    participant_id=requester_participant_id,
                    capability_id="emergency_stop",
                    action_params={
                        "reason": reason,
                        "evidence": evidence or {},
                        "timestamp": int(time.time()),
                    },
                    timeout_seconds=3600,
                )

                if success:
                    logger.info(
                        f"Emergency stop request created: {request_id} "
                        f"by {requester_participant_id}"
                    )
                    return True, "Emergency stop request pending guardian approval", request_id
                else:
                    return False, f"Failed to create authorization request: {message}", None
            else:
                return False, f"Not authorized for emergency stop: {auth_reason}", None

        # Participant has capability, execute immediately
        return await self.execute_emergency_stop(
            reason=reason,
            executed_by=requester_participant_id,
        )

    async def execute_emergency_stop(
        self,
        reason: str,
        executed_by: str,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Execute emergency stop (after authorization)

        Args:
            reason: Reason for stop
            executed_by: Participant executing stop

        Returns:
            Tuple of (success, message, action_id)
        """
        if self.emergency_stopped:
            return False, "Emergency stop already active", None

        self.emergency_stopped = True

        # Generate unique action ID
        action_id = f"emergency_stop_{int(time.time())}"
        timestamp = int(time.time())

        # Store emergency stop record on DHT for immutable audit trail
        emergency_record = {
            "action_id": action_id,
            "action_type": "EMERGENCY_STOP",
            "executed_by": executed_by,
            "reason": reason,
            "timestamp": timestamp,
            "fl_round": getattr(self._fl_coordinator, 'current_round', 0) if self._fl_coordinator else 0,
            "status": "ACTIVE",
        }

        # Attempt to store on DHT (non-blocking - emergency stop proceeds even if DHT fails)
        if self._dht_client is not None:
            try:
                await self._dht_client._call_zome(
                    zome_name="governance_record",
                    fn_name="store_emergency_action",
                    payload=emergency_record,
                )
                logger.info(f"Emergency stop record stored on DHT: {action_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to store emergency stop on DHT (proceeding anyway): {e}"
                )
        else:
            logger.warning(
                "DHT client not configured - emergency stop record stored locally only"
            )

        # Also store via governance coordinator's DHT client if available
        if hasattr(self.gov_coord, 'dht') and self.gov_coord.dht is not None:
            try:
                await self.gov_coord.dht.call_zome(
                    zome_name="governance_record",
                    fn_name="store_emergency_action",
                    payload=emergency_record,
                )
                logger.info(f"Emergency stop record stored via governance DHT: {action_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to store emergency stop via governance DHT: {e}"
                )

        # Track participation for the executor
        self._record_governance_participation(
            participant_id=executed_by,
            action_type="emergency_stop",
            action_id=action_id,
        )

        logger.critical(
            f"EMERGENCY STOP ACTIVATED by {executed_by}: {reason}"
        )

        return True, "Emergency stop activated", action_id

    def _record_governance_participation(
        self,
        participant_id: str,
        action_type: str,
        action_id: str,
    ) -> None:
        """
        Record governance participation for a participant.

        Args:
            participant_id: Participant who performed the action
            action_type: Type of governance action (vote, proposal, emergency_stop, etc.)
            action_id: Unique identifier for the action
        """
        if participant_id not in self._participation_records:
            self._participation_records[participant_id] = {
                "total_actions": 0,
                "actions_by_type": {},
                "recent_actions": [],
                "first_action_at": int(time.time()),
                "last_action_at": 0,
            }

        record = self._participation_records[participant_id]
        record["total_actions"] += 1
        record["last_action_at"] = int(time.time())

        if action_type not in record["actions_by_type"]:
            record["actions_by_type"][action_type] = 0
        record["actions_by_type"][action_type] += 1

        # Keep last 100 recent actions
        record["recent_actions"].append({
            "action_type": action_type,
            "action_id": action_id,
            "timestamp": int(time.time()),
        })
        if len(record["recent_actions"]) > 100:
            record["recent_actions"] = record["recent_actions"][-100:]

    def _get_participant_governance_stats(
        self,
        participant_id: str,
    ) -> Dict[str, Any]:
        """
        Get governance participation statistics for a participant.

        Args:
            participant_id: Participant ID to get stats for

        Returns:
            Dictionary with participation statistics
        """
        if participant_id not in self._participation_records:
            return {
                "total_actions": 0,
                "actions_by_type": {},
                "recent_actions": [],
                "first_action_at": 0,
                "last_action_at": 0,
                "participation_score": 0.0,
            }

        record = self._participation_records[participant_id]

        # Calculate participation score (0.0 to 1.0)
        # Based on: total actions, recency, action diversity
        now = int(time.time())
        days_since_last = (now - record["last_action_at"]) / 86400 if record["last_action_at"] > 0 else 365

        # Recency factor: full score if active in last 7 days, decays over 30 days
        recency_factor = max(0.0, min(1.0, 1.0 - (days_since_last - 7) / 23))

        # Activity factor: based on total actions (capped at 100)
        activity_factor = min(1.0, record["total_actions"] / 100)

        # Diversity factor: bonus for multiple action types
        num_action_types = len(record["actions_by_type"])
        diversity_factor = min(1.0, num_action_types / 5)  # Full score at 5+ action types

        # Combined participation score
        participation_score = (
            recency_factor * 0.4 +
            activity_factor * 0.4 +
            diversity_factor * 0.2
        )

        return {
            **record,
            "participation_score": participation_score,
            "recency_factor": recency_factor,
            "activity_factor": activity_factor,
            "diversity_factor": diversity_factor,
        }

    def _calculate_governance_participation_bonus(
        self,
        participant_id: str,
        local_participation: Dict[str, Any],
        extension_stats: Dict[str, Any],
    ) -> float:
        """
        Calculate governance participation bonus factor (0.0 to 1.0).

        Combines local tracking with governance extension stats.

        Args:
            participant_id: Participant ID
            local_participation: Stats from local tracking
            extension_stats: Stats from governance extensions

        Returns:
            Bonus factor between 0.0 and 1.0
        """
        # Local participation score
        local_score = local_participation.get("participation_score", 0.0)

        # Extension stats score
        proposals = extension_stats.get("proposals_submitted", 0)
        votes = extension_stats.get("votes_cast", 0)
        capabilities = sum(extension_stats.get("capabilities_invoked", {}).values())

        # Normalize extension stats (proposals worth more than votes)
        ext_proposal_score = min(1.0, proposals / 10)  # Full score at 10 proposals
        ext_vote_score = min(1.0, votes / 50)  # Full score at 50 votes
        ext_capability_score = min(1.0, capabilities / 100)  # Full score at 100 invocations

        extension_score = (
            ext_proposal_score * 0.4 +
            ext_vote_score * 0.4 +
            ext_capability_score * 0.2
        )

        # Combined bonus (average of local and extension scores)
        combined_bonus = (local_score + extension_score) / 2.0

        return min(1.0, max(0.0, combined_bonus))

    def get_governance_participation_summary(
        self,
        participant_id: str,
    ) -> Dict[str, Any]:
        """
        Get a comprehensive summary of governance participation for a participant.

        Args:
            participant_id: Participant ID

        Returns:
            Dictionary with comprehensive participation summary
        """
        local_stats = self._get_participant_governance_stats(participant_id)
        ext_stats = self.gov_coord.gov_extensions.get_governance_stats(participant_id)
        bonus = self._calculate_governance_participation_bonus(
            participant_id, local_stats, ext_stats
        )

        return {
            "participant_id": participant_id,
            "local_tracking": local_stats,
            "extension_stats": ext_stats,
            "governance_bonus": bonus,
            "reward_multiplier": 1.0 + (bonus * (self.config.governance_participation_bonus - 1.0)),
        }

    async def resume_fl_training(
        self,
        requester_participant_id: str,
        reason: str,
    ) -> Tuple[bool, str]:
        """
        Resume FL training after emergency stop

        Requires governance proposal approval.

        Args:
            requester_participant_id: Participant requesting resume
            reason: Reason for resumption

        Returns:
            Tuple of (success, message)
        """
        if not self.emergency_stopped:
            return False, "FL training not stopped"

        # Create governance proposal for resumption
        success, message, proposal_id = await self.gov_coord.create_governance_proposal(
            proposer_participant_id=requester_participant_id,
            proposal_type=ProposalType.EMERGENCY_ACTION.value,
            title="Resume FL Training After Emergency Stop",
            description=f"Proposal to resume FL training. Reason: {reason}",
            execution_params={
                "action": "resume_training",
                "reason": reason,
                "stopped_by": "emergency_stop",
            },
            voting_duration_days=3,  # Shorter voting period for emergency
        )

        if success:
            logger.info(
                f"Resume training proposal created: {proposal_id} "
                f"by {requester_participant_id}"
            )
            return True, f"Proposal created: {proposal_id}"
        else:
            return False, f"Failed to create proposal: {message}"

    # Participant Management

    async def request_participant_ban(
        self,
        requester_participant_id: str,
        target_participant_id: str,
        reason: str,
        evidence: Optional[Dict[str, Any]] = None,
        permanent: bool = False,
        ban_duration_seconds: int = 86400 * 7,  # 7 days default
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Request to ban participant from FL

        If guardian authorization required, creates authorization request.
        If proposal required, creates governance proposal.

        Args:
            requester_participant_id: Participant requesting ban
            target_participant_id: Participant to ban
            reason: Reason for ban
            evidence: Optional evidence
            permanent: Whether ban is permanent
            ban_duration_seconds: Duration for temporary ban

        Returns:
            Tuple of (success, message, request_id or proposal_id)
        """
        # Check if requester has capability
        authorized, auth_reason = await self.gov_coord.authorize_fl_action(
            participant_id=requester_participant_id,
            action="ban_participant",
        )

        if not authorized:
            # Check if guardian approval required
            if self.config.ban_requires_guardian:
                # Create authorization request
                success, message, request_id = await self.gov_coord.guardian_auth_mgr.request_authorization(
                    participant_id=requester_participant_id,
                    capability_id="ban_participant",
                    action_params={
                        "target": target_participant_id,
                        "reason": reason,
                        "evidence": evidence or {},
                        "permanent": permanent,
                        "duration_seconds": ban_duration_seconds,
                    },
                    timeout_seconds=3600,
                )

                if success:
                    return True, "Ban request pending guardian approval", request_id
                else:
                    return False, f"Failed to create authorization request: {message}", None
            else:
                return False, f"Not authorized to ban participants: {auth_reason}", None

        # Check if proposal required
        if self.config.ban_requires_proposal:
            # Create governance proposal
            success, message, proposal_id = await self.gov_coord.create_governance_proposal(
                proposer_participant_id=requester_participant_id,
                proposal_type=ProposalType.PARTICIPANT_MANAGEMENT.value,
                title=f"Ban Participant {target_participant_id}",
                description=f"Proposal to ban {target_participant_id}. Reason: {reason}",
                execution_params={
                    "action": "ban_participant",
                    "target": target_participant_id,
                    "reason": reason,
                    "evidence": evidence or {},
                    "permanent": permanent,
                    "duration_seconds": ban_duration_seconds,
                },
                voting_duration_days=7,
            )

            if success:
                return True, f"Ban proposal created: {proposal_id}", proposal_id
            else:
                return False, f"Failed to create proposal: {message}", None

        # Requester authorized, execute immediately
        return await self.execute_participant_ban(
            target_participant_id=target_participant_id,
            reason=reason,
            permanent=permanent,
            ban_duration_seconds=ban_duration_seconds,
            executed_by=requester_participant_id,
        )

    async def execute_participant_ban(
        self,
        target_participant_id: str,
        reason: str,
        permanent: bool,
        ban_duration_seconds: int,
        executed_by: str,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Execute participant ban (after authorization)

        Args:
            target_participant_id: Participant to ban
            reason: Reason for ban
            permanent: Whether ban is permanent
            ban_duration_seconds: Duration for temporary ban
            executed_by: Participant executing ban

        Returns:
            Tuple of (success, message, action_id)
        """
        if target_participant_id in self.banned_participants:
            return False, "Participant already banned", None

        ban_info = {
            "participant_id": target_participant_id,
            "reason": reason,
            "permanent": permanent,
            "banned_at": int(time.time()),
            "executed_by": executed_by,
        }

        if not permanent:
            ban_info["until"] = int(time.time()) + ban_duration_seconds
            ban_info["duration_seconds"] = ban_duration_seconds

        self.banned_participants[target_participant_id] = ban_info

        action_id = f"ban_{target_participant_id}_{int(time.time())}"

        logger.warning(
            f"Participant banned: {target_participant_id} by {executed_by}. "
            f"Reason: {reason}. Permanent: {permanent}"
        )

        return True, "Participant banned successfully", action_id

    async def execute_participant_unban(
        self,
        target_participant_id: str,
        reason: str,
        executed_by: str,
    ) -> Tuple[bool, str]:
        """
        Unban participant (after proposal approval)

        Args:
            target_participant_id: Participant to unban
            reason: Reason for unban
            executed_by: Participant executing unban

        Returns:
            Tuple of (success, message)
        """
        if target_participant_id not in self.banned_participants:
            return False, "Participant not banned"

        del self.banned_participants[target_participant_id]

        logger.info(
            f"Participant unbanned: {target_participant_id} by {executed_by}. "
            f"Reason: {reason}"
        )

        return True, "Participant unbanned successfully"

    # Parameter Management

    async def propose_parameter_change(
        self,
        proposer_participant_id: str,
        parameter_name: str,
        current_value: Any,
        new_value: Any,
        rationale: str,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Propose FL parameter change via governance

        Args:
            proposer_participant_id: Participant proposing change
            parameter_name: Name of parameter to change
            current_value: Current parameter value
            new_value: Proposed new value
            rationale: Explanation for change

        Returns:
            Tuple of (success, message, proposal_id)
        """
        # Check if parameter change requires proposal
        is_critical = parameter_name in self.config.critical_parameters

        if is_critical and not self.config.parameter_change_requires_proposal:
            # Critical parameter but proposals not required (unusual)
            pass
        elif is_critical or self.config.parameter_change_requires_proposal:
            # Create governance proposal
            success, message, proposal_id = await self.gov_coord.create_governance_proposal(
                proposer_participant_id=proposer_participant_id,
                proposal_type=ProposalType.PARAMETER_CHANGE.value,
                title=f"Change {parameter_name}: {current_value} → {new_value}",
                description=f"Proposal to change FL parameter. Rationale: {rationale}",
                execution_params={
                    "action": "change_parameter",
                    "parameter_name": parameter_name,
                    "current_value": current_value,
                    "new_value": new_value,
                    "rationale": rationale,
                },
                voting_duration_days=7,
            )

            return success, message, proposal_id

        # Non-critical parameter, no proposal required
        return False, "Parameter change does not require proposal", None

    async def execute_parameter_change(
        self,
        parameter_name: str,
        new_value: Any,
        proposal_id: str,
    ) -> Tuple[bool, str]:
        """
        Execute approved parameter change

        Args:
            parameter_name: Parameter to change
            new_value: New value
            proposal_id: Approved proposal ID

        Returns:
            Tuple of (success, message)
        """
        # Store in local parameter cache
        old_value = self.fl_parameters.get(parameter_name, None)
        self.fl_parameters[parameter_name] = new_value

        logger.info(
            f"FL parameter changed: {parameter_name} = {new_value} "
            f"(was {old_value}), proposal {proposal_id}"
        )

        # Apply to actual FL coordinator if available
        applied_to_coordinator = False
        if self._fl_coordinator is not None:
            applied_to_coordinator = self._apply_parameter_to_fl_coordinator(
                parameter_name=parameter_name,
                new_value=new_value,
            )

        # Store parameter change record on DHT for audit trail
        if hasattr(self.gov_coord, 'dht') and self.gov_coord.dht is not None:
            try:
                await self.gov_coord.dht.call_zome(
                    zome_name="governance_record",
                    fn_name="store_parameter_change",
                    payload={
                        "proposal_id": proposal_id,
                        "parameter_name": parameter_name,
                        "old_value": str(old_value) if old_value is not None else None,
                        "new_value": str(new_value),
                        "timestamp": int(time.time()),
                        "applied_to_coordinator": applied_to_coordinator,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to store parameter change on DHT: {e}")

        status_msg = f"Parameter {parameter_name} updated to {new_value}"
        if applied_to_coordinator:
            status_msg += " (applied to FL coordinator)"
        else:
            status_msg += " (pending FL coordinator restart)"

        return True, status_msg

    def _apply_parameter_to_fl_coordinator(
        self,
        parameter_name: str,
        new_value: Any,
    ) -> bool:
        """
        Apply a parameter change to the FL coordinator.

        Maps governance parameter names to Phase10Config attributes.

        Args:
            parameter_name: Name of the parameter to change
            new_value: New value for the parameter

        Returns:
            True if successfully applied, False otherwise
        """
        if self._fl_coordinator is None:
            return False

        # Map governance parameter names to Phase10Config attributes
        parameter_mapping = {
            "min_reputation": ("config", "zkpoc_pogq_threshold"),
            "byzantine_threshold": ("config", "zkpoc_pogq_threshold"),
            "aggregation_strategy": ("config", "aggregation_algorithm"),
            "quorum_size": None,  # Not directly configurable at runtime
            "zkpoc_enabled": ("config", "zkpoc_enabled"),
            "zkpoc_pogq_threshold": ("config", "zkpoc_pogq_threshold"),
            "risc0_zk_proofs_required": ("config", "risc0_zk_proofs_required"),
            "risc0_zk_proofs_preferred": ("config", "risc0_zk_proofs_preferred"),
            "risc0_zk_weight_multiplier": ("config", "risc0_zk_weight_multiplier"),
            "storage_strategy": ("storage_strategy", None),
        }

        mapping = parameter_mapping.get(parameter_name)

        if mapping is None:
            logger.warning(
                f"Parameter '{parameter_name}' is not dynamically configurable"
            )
            return False

        try:
            if mapping[1] is None:
                # Direct attribute on coordinator
                attr_name = mapping[0]
                if hasattr(self._fl_coordinator, attr_name):
                    setattr(self._fl_coordinator, attr_name, new_value)
                    logger.info(
                        f"Applied parameter {parameter_name} to FL coordinator: "
                        f"{attr_name} = {new_value}"
                    )
                    return True
            else:
                # Nested attribute (e.g., config.zkpoc_pogq_threshold)
                parent_attr, child_attr = mapping
                parent = getattr(self._fl_coordinator, parent_attr, None)
                if parent is not None and hasattr(parent, child_attr):
                    setattr(parent, child_attr, new_value)
                    logger.info(
                        f"Applied parameter {parameter_name} to FL coordinator: "
                        f"{parent_attr}.{child_attr} = {new_value}"
                    )
                    return True

            logger.warning(
                f"Could not find attribute for parameter '{parameter_name}' "
                f"in FL coordinator"
            )
            return False

        except Exception as e:
            logger.error(
                f"Failed to apply parameter '{parameter_name}' to FL coordinator: {e}"
            )
            return False

    # Proposal Execution

    async def execute_approved_proposal(
        self,
        proposal_id: str,
    ) -> Tuple[bool, str]:
        """
        Execute an approved governance proposal

        Handles different proposal types:
        - PARAMETER_CHANGE: Update FL parameters
        - PARTICIPANT_MANAGEMENT: Ban/unban participants
        - EMERGENCY_ACTION: Resume training, etc.

        Args:
            proposal_id: Approved proposal to execute

        Returns:
            Tuple of (success, message)
        """
        # Retrieve proposal
        proposal = await self.gov_coord.proposal_mgr.get_proposal(proposal_id)

        if not proposal:
            return False, "Proposal not found"

        if proposal.status != ProposalStatus.APPROVED:
            return False, f"Proposal not approved: {proposal.status.value}"

        # Execute based on type
        try:
            if proposal.proposal_type == ProposalType.PARAMETER_CHANGE:
                # Extract parameters
                params = proposal.execution_params
                parameter_name = params.get("parameter_name")
                new_value = params.get("new_value")

                if not parameter_name:
                    return False, "Missing parameter_name in execution_params"

                success, message = await self.execute_parameter_change(
                    parameter_name=parameter_name,
                    new_value=new_value,
                    proposal_id=proposal_id,
                )

                if success:
                    # Update proposal status
                    await self.gov_coord.proposal_mgr.update_proposal_status(
                        proposal_id=proposal_id,
                        new_status=ProposalStatus.EXECUTED,
                        execution_result=message,
                    )

                return success, message

            elif proposal.proposal_type == ProposalType.PARTICIPANT_MANAGEMENT:
                # Extract action
                params = proposal.execution_params
                action = params.get("action")

                if action == "ban_participant":
                    target = params.get("target")
                    reason = params.get("reason", "Governance proposal approved")
                    permanent = params.get("permanent", False)
                    duration = params.get("duration_seconds", 86400 * 7)

                    success, message, _ = await self.execute_participant_ban(
                        target_participant_id=target,
                        reason=reason,
                        permanent=permanent,
                        ban_duration_seconds=duration,
                        executed_by="governance",
                    )

                elif action == "unban_participant":
                    target = params.get("target")
                    reason = params.get("reason", "Governance proposal approved")

                    success, message = await self.execute_participant_unban(
                        target_participant_id=target,
                        reason=reason,
                        executed_by="governance",
                    )

                else:
                    return False, f"Unknown participant management action: {action}"

                if success:
                    await self.gov_coord.proposal_mgr.update_proposal_status(
                        proposal_id=proposal_id,
                        new_status=ProposalStatus.EXECUTED,
                        execution_result=message,
                    )

                return success, message

            elif proposal.proposal_type == ProposalType.EMERGENCY_ACTION:
                # Extract action
                params = proposal.execution_params
                action = params.get("action")

                if action == "resume_training":
                    if not self.emergency_stopped:
                        return False, "FL training not stopped"

                    self.emergency_stopped = False
                    message = "FL training resumed via governance proposal"

                    logger.info(message)

                    await self.gov_coord.proposal_mgr.update_proposal_status(
                        proposal_id=proposal_id,
                        new_status=ProposalStatus.EXECUTED,
                        execution_result=message,
                    )

                    return True, message

                else:
                    return False, f"Unknown emergency action: {action}"

            else:
                return False, f"Unsupported proposal type for execution: {proposal.proposal_type.value}"

        except Exception as e:
            error_message = f"Execution failed: {str(e)}"
            logger.error(f"Failed to execute proposal {proposal_id}: {e}")

            # Mark as failed
            await self.gov_coord.proposal_mgr.update_proposal_status(
                proposal_id=proposal_id,
                new_status=ProposalStatus.FAILED,
                execution_result=error_message,
            )

            return False, error_message

    # Rewards and Reputation

    async def calculate_reputation_weighted_reward(
        self,
        participant_id: str,
        base_reward: float,
    ) -> float:
        """
        Calculate reputation-weighted reward for participant

        Participants with higher governance participation get bonus rewards.

        Args:
            participant_id: Participant receiving reward
            base_reward: Base reward amount

        Returns:
            Adjusted reward amount
        """
        if not self.config.reputation_weighted_rewards:
            return base_reward

        # Get governance participation stats from local tracking
        participation = self._get_participant_governance_stats(participant_id)

        # Also get stats from governance extensions
        ext_stats = self.gov_coord.gov_extensions.get_governance_stats(participant_id)

        # Calculate governance participation bonus based on activity
        governance_bonus = self._calculate_governance_participation_bonus(
            participant_id=participant_id,
            local_participation=participation,
            extension_stats=ext_stats,
        )

        # Apply reputation-weighted bonus
        vote_weight = await self.gov_coord.gov_extensions.calculate_vote_weight(
            participant_id=participant_id
        )

        # Scale reward by vote weight
        # vote_weight range: 0.5 - 10.0, we want reward multiplier: 0.8 - 1.5
        base_multiplier = 0.8 + (vote_weight / 20.0)  # 0.5->0.825, 10.0->1.3
        base_multiplier = min(1.5, max(0.8, base_multiplier))

        # Apply governance participation bonus (up to 20% additional)
        governance_multiplier = 1.0 + (governance_bonus * (self.config.governance_participation_bonus - 1.0))

        # Combined reward multiplier
        reward_multiplier = base_multiplier * governance_multiplier

        adjusted_reward = base_reward * reward_multiplier

        logger.debug(
            f"Reward adjusted for {participant_id}: "
            f"{base_reward} × {reward_multiplier:.2f} = {adjusted_reward:.2f}"
        )

        return adjusted_reward

    # Status Queries

    def get_fl_governance_status(self) -> Dict[str, Any]:
        """
        Get current FL governance status

        Returns:
            Dictionary with status information
        """
        return {
            "emergency_stopped": self.emergency_stopped,
            "banned_participants": len(self.banned_participants),
            "active_parameters": len(self.fl_parameters),
            "config": {
                "require_capability_for_submit": self.config.require_capability_for_submit,
                "require_capability_for_request": self.config.require_capability_for_request,
                "emergency_stop_enabled": self.config.emergency_stop_enabled,
                "ban_requires_proposal": self.config.ban_requires_proposal,
                "parameter_change_requires_proposal": self.config.parameter_change_requires_proposal,
                "reputation_weighted_rewards": self.config.reputation_weighted_rewards,
            },
        }

    def is_participant_banned(self, participant_id: str) -> bool:
        """Check if participant is currently banned"""
        if participant_id not in self.banned_participants:
            return False

        ban_info = self.banned_participants[participant_id]

        # Check if temporary ban expired
        if not ban_info.get("permanent", False):
            ban_until = ban_info.get("until", 0)
            if time.time() >= ban_until:
                # Ban expired
                del self.banned_participants[participant_id]
                return False

        return True

    def get_parameter(self, parameter_name: str) -> Optional[Any]:
        """Get current value of FL parameter"""
        return self.fl_parameters.get(parameter_name)
