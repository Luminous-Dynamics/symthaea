# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Governance-Enabled FL Coordinator
Week 7-8 Phase 4: FL Integration

Extends Phase10Coordinator with identity-gated governance capabilities.
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np

from zerotrustml.core.phase10_coordinator import Phase10Coordinator, Phase10Config
from .fl_integration import FLGovernanceIntegration, FLGovernanceConfig
from .coordinator import GovernanceCoordinator

logger = logging.getLogger(__name__)


class GovernedFLCoordinator(Phase10Coordinator):
    """
    FL Coordinator with Governance Integration

    Extends Phase10Coordinator with:
    - Pre-round capability checks
    - Emergency stop via guardian authorization
    - Participant banning via governance proposals
    - Parameter changes via governance proposals
    - Reputation-weighted rewards
    """

    def __init__(
        self,
        phase10_config: Phase10Config,
        governance_coordinator: GovernanceCoordinator,
        fl_governance_config: Optional[FLGovernanceConfig] = None,
    ):
        """
        Args:
            phase10_config: Phase 10 FL configuration
            governance_coordinator: Governance coordinator instance
            fl_governance_config: FL governance configuration
        """
        super().__init__(phase10_config)

        self.gov_coord = governance_coordinator
        self.fl_gov = FLGovernanceIntegration(
            governance_coordinator=governance_coordinator,
            config=fl_governance_config,
            fl_coordinator=self,  # Pass self reference for parameter updates
        )

        logger.info("GovernedFLCoordinator initialized with governance integration")

    async def initialize(self):
        """Initialize FL coordinator with governance"""
        await super().initialize()

        # Apply any governance-set parameters
        applied_count = 0
        for param_name, param_value in self.fl_gov.fl_parameters.items():
            logger.info(f"Applying governance parameter: {param_name} = {param_value}")
            if self._apply_governance_parameter(param_name, param_value):
                applied_count += 1

        if applied_count > 0:
            logger.info(f"Applied {applied_count} governance parameters to FL config")

    def _apply_governance_parameter(
        self,
        param_name: str,
        param_value: Any,
    ) -> bool:
        """
        Apply a governance parameter to the FL config.

        Args:
            param_name: Parameter name
            param_value: Parameter value

        Returns:
            True if applied successfully, False otherwise
        """
        # Map parameter names to config attributes
        config_mapping = {
            "min_reputation": "zkpoc_pogq_threshold",
            "byzantine_threshold": "zkpoc_pogq_threshold",
            "aggregation_strategy": "aggregation_algorithm",
            "zkpoc_enabled": "zkpoc_enabled",
            "zkpoc_pogq_threshold": "zkpoc_pogq_threshold",
            "risc0_zk_proofs_required": "risc0_zk_proofs_required",
            "risc0_zk_proofs_preferred": "risc0_zk_proofs_preferred",
            "risc0_zk_weight_multiplier": "risc0_zk_weight_multiplier",
            "encryption_enabled": "encryption_enabled",
        }

        # Special handling for storage_strategy (direct attribute on coordinator)
        if param_name == "storage_strategy":
            if hasattr(self, "storage_strategy"):
                self.storage_strategy = param_value
                logger.info(f"Applied {param_name} = {param_value} to coordinator")
                return True
            return False

        # Map to config attribute
        config_attr = config_mapping.get(param_name)
        if config_attr is None:
            logger.warning(f"Unknown governance parameter: {param_name}")
            return False

        # Apply to config
        if hasattr(self.config, config_attr):
            old_value = getattr(self.config, config_attr)
            setattr(self.config, config_attr, param_value)
            logger.info(
                f"Applied governance parameter to config: "
                f"{config_attr} = {param_value} (was {old_value})"
            )
            return True
        else:
            logger.warning(f"Config has no attribute: {config_attr}")
            return False

    # Override FL methods with governance checks

    async def handle_gradient_submission(
        self,
        node_id: str,
        encrypted_gradient: bytes,
        zkpoc_proof: Optional[Any] = None,
        pogq_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Handle gradient submission with governance pre-check

        Args:
            node_id: Node submitting gradient
            encrypted_gradient: Encrypted gradient data
            zkpoc_proof: Optional ZK proof
            pogq_score: Optional PoGQ score

        Returns:
            Result dict with status
        """
        # Pre-round governance check
        authorized, reason = await self.fl_gov.verify_participant_for_round(
            participant_id=node_id,
            round_number=self.current_round,
        )

        if not authorized:
            logger.warning(
                f"Gradient submission rejected for {node_id}: {reason}"
            )
            return {
                "accepted": False,
                "reason": f"Governance check failed: {reason}",
                "gradient_id": None,
            }

        # Proceed with normal FL submission
        result = await super().handle_gradient_submission(
            node_id=node_id,
            encrypted_gradient=encrypted_gradient,
            zkpoc_proof=zkpoc_proof,
            pogq_score=pogq_score,
        )

        return result

    async def request_global_model(
        self,
        node_id: str,
    ) -> Dict[str, Any]:
        """
        Request global model with governance check

        Args:
            node_id: Node requesting model

        Returns:
            Result dict with model or error
        """
        # Governance check
        authorized, reason = await self.fl_gov.verify_model_request(
            participant_id=node_id,
        )

        if not authorized:
            logger.warning(
                f"Model request rejected for {node_id}: {reason}"
            )
            return {
                "success": False,
                "reason": f"Governance check failed: {reason}",
                "model": None,
            }

        # Return global model
        return {
            "success": True,
            "model": self.global_model,
            "round": self.current_round,
        }

    # Emergency Actions

    async def emergency_stop(
        self,
        requester_participant_id: str,
        reason: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Request emergency stop via governance

        Args:
            requester_participant_id: Participant requesting stop
            reason: Reason for stop
            evidence: Optional evidence

        Returns:
            Result dict
        """
        success, message, request_id = await self.fl_gov.request_emergency_stop(
            requester_participant_id=requester_participant_id,
            reason=reason,
            evidence=evidence,
        )

        return {
            "success": success,
            "message": message,
            "request_id": request_id,
            "emergency_stopped": self.fl_gov.emergency_stopped,
        }

    async def resume_training(
        self,
        requester_participant_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Request to resume training after emergency stop

        Args:
            requester_participant_id: Participant requesting resume
            reason: Reason for resumption

        Returns:
            Result dict
        """
        success, message = await self.fl_gov.resume_fl_training(
            requester_participant_id=requester_participant_id,
            reason=reason,
        )

        return {
            "success": success,
            "message": message,
        }

    # Participant Management

    async def ban_participant(
        self,
        requester_participant_id: str,
        target_participant_id: str,
        reason: str,
        evidence: Optional[Dict[str, Any]] = None,
        permanent: bool = False,
        ban_duration_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Request to ban participant via governance

        Args:
            requester_participant_id: Participant requesting ban
            target_participant_id: Participant to ban
            reason: Reason for ban
            evidence: Optional evidence
            permanent: Whether ban is permanent
            ban_duration_days: Duration for temporary ban

        Returns:
            Result dict
        """
        success, message, request_or_proposal_id = await self.fl_gov.request_participant_ban(
            requester_participant_id=requester_participant_id,
            target_participant_id=target_participant_id,
            reason=reason,
            evidence=evidence,
            permanent=permanent,
            ban_duration_seconds=ban_duration_days * 86400,
        )

        return {
            "success": success,
            "message": message,
            "request_id": request_or_proposal_id,
        }

    async def unban_participant(
        self,
        requester_participant_id: str,
        target_participant_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Request to unban participant (requires proposal)

        Args:
            requester_participant_id: Participant requesting unban
            target_participant_id: Participant to unban
            reason: Reason for unban

        Returns:
            Result dict
        """
        # Create unban proposal
        success, message, proposal_id = await self.gov_coord.create_governance_proposal(
            proposer_participant_id=requester_participant_id,
            proposal_type="PARTICIPANT_MANAGEMENT",
            title=f"Unban Participant {target_participant_id}",
            description=f"Proposal to unban {target_participant_id}. Reason: {reason}",
            execution_params={
                "action": "unban_participant",
                "target": target_participant_id,
                "reason": reason,
            },
            voting_duration_days=7,
        )

        return {
            "success": success,
            "message": message,
            "proposal_id": proposal_id,
        }

    # Parameter Management

    async def propose_parameter_change(
        self,
        proposer_participant_id: str,
        parameter_name: str,
        current_value: Any,
        new_value: Any,
        rationale: str,
    ) -> Dict[str, Any]:
        """
        Propose FL parameter change via governance

        Args:
            proposer_participant_id: Participant proposing change
            parameter_name: Parameter to change
            current_value: Current value
            new_value: Proposed new value
            rationale: Explanation

        Returns:
            Result dict
        """
        success, message, proposal_id = await self.fl_gov.propose_parameter_change(
            proposer_participant_id=proposer_participant_id,
            parameter_name=parameter_name,
            current_value=current_value,
            new_value=new_value,
            rationale=rationale,
        )

        return {
            "success": success,
            "message": message,
            "proposal_id": proposal_id,
        }

    # Reward Distribution

    async def distribute_rewards(
        self,
        round_number: int,
        participants: List[str],
        base_rewards: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Distribute reputation-weighted rewards

        Args:
            round_number: FL round number
            participants: List of participant IDs
            base_rewards: Base reward for each participant

        Returns:
            Dictionary of adjusted rewards
        """
        adjusted_rewards = {}

        for participant_id in participants:
            base_reward = base_rewards.get(participant_id, 0.0)

            # Apply reputation weighting
            adjusted_reward = await self.fl_gov.calculate_reputation_weighted_reward(
                participant_id=participant_id,
                base_reward=base_reward,
            )

            adjusted_rewards[participant_id] = adjusted_reward

        logger.info(
            f"Round {round_number} rewards distributed with reputation weighting: "
            f"total base={sum(base_rewards.values()):.2f}, "
            f"total adjusted={sum(adjusted_rewards.values()):.2f}"
        )

        return adjusted_rewards

    # Proposal Execution Hook

    async def execute_governance_proposal(
        self,
        proposal_id: str,
    ) -> Dict[str, Any]:
        """
        Execute approved governance proposal

        This is called by governance system or admin after proposal approval.

        Args:
            proposal_id: Approved proposal ID

        Returns:
            Result dict
        """
        success, message = await self.fl_gov.execute_approved_proposal(
            proposal_id=proposal_id,
        )

        return {
            "success": success,
            "message": message,
            "proposal_id": proposal_id,
        }

    # Status Queries

    def get_governance_status(self) -> Dict[str, Any]:
        """Get FL governance status"""
        fl_gov_status = self.fl_gov.get_fl_governance_status()

        return {
            "fl_governance": fl_gov_status,
            "current_round": self.current_round,
            "total_participants": len(self.pending_gradients),
        }

    def is_participant_authorized(self, participant_id: str) -> bool:
        """Check if participant is authorized (not banned, not emergency stopped)"""
        if self.fl_gov.emergency_stopped:
            return False

        if self.fl_gov.is_participant_banned(participant_id):
            return False

        return True
