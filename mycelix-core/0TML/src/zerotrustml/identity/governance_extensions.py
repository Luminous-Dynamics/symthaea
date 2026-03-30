# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Identity Governance Extensions

Extends the identity coordinator with governance-specific functionality:
- Vote weight calculation based on identity verification
- Capability-based access control
- Guardian authorization for emergency actions
- Governance participation tracking

Week 7-8 Phase 2: Identity Governance Extensions
"""

import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Import identity modules
try:
    from .did_manager import DIDManager
    from .assurance import AssuranceLevel
except ImportError:
    # For testing without full identity system
    DIDManager = None
    AssuranceLevel = None


# ========================================
# Capability Definitions
# ========================================

@dataclass
class Capability:
    """Governance capability definition"""
    capability_id: str
    name: str
    description: str
    required_assurance: str  # E0-E4
    required_reputation: float  # 0.0-1.0
    required_sybil_resistance: float  # 0.0-1.0
    required_guardian_approval: bool
    guardian_threshold: float  # If guardian approval required
    rate_limit: Optional[int] = None  # Max invocations per time period
    time_period_seconds: Optional[int] = None


# Standard capability registry
CAPABILITY_REGISTRY: Dict[str, Capability] = {
    "submit_mip": Capability(
        capability_id="submit_mip",
        name="Submit MIP",
        description="Submit a Mycelix Improvement Proposal",
        required_assurance="E2",
        required_reputation=0.6,
        required_sybil_resistance=0.5,
        required_guardian_approval=False,
        guardian_threshold=0.0,
        rate_limit=10,
        time_period_seconds=86400  # 10 per day
    ),
    "vote_on_mip": Capability(
        capability_id="vote_on_mip",
        name="Vote on MIP",
        description="Vote on existing Mycelix Improvement Proposal",
        required_assurance="E1",
        required_reputation=0.4,
        required_sybil_resistance=0.3,
        required_guardian_approval=False,
        guardian_threshold=0.0,
    ),
    "emergency_stop": Capability(
        capability_id="emergency_stop",
        name="Emergency Stop",
        description="Emergency stop FL training",
        required_assurance="E3",
        required_reputation=0.8,
        required_sybil_resistance=0.7,
        required_guardian_approval=True,
        guardian_threshold=0.7,
        rate_limit=5,
        time_period_seconds=86400  # 5 per day
    ),
    "update_parameters": Capability(
        capability_id="update_parameters",
        name="Update Parameters",
        description="Modify FL hyperparameters",
        required_assurance="E2",
        required_reputation=0.7,
        required_sybil_resistance=0.6,
        required_guardian_approval=True,
        guardian_threshold=0.6,
    ),
    "ban_participant": Capability(
        capability_id="ban_participant",
        name="Ban Participant",
        description="Remove participant from network",
        required_assurance="E3",
        required_reputation=0.8,
        required_sybil_resistance=0.7,
        required_guardian_approval=True,
        guardian_threshold=0.8,
        rate_limit=20,
        time_period_seconds=86400  # 20 per day
    ),
    "treasury_withdrawal": Capability(
        capability_id="treasury_withdrawal",
        name="Treasury Withdrawal",
        description="Withdraw from treasury",
        required_assurance="E4",
        required_reputation=0.9,
        required_sybil_resistance=0.9,
        required_guardian_approval=True,
        guardian_threshold=0.9,
        rate_limit=1,
        time_period_seconds=86400  # 1 per day
    ),
}


# ========================================
# Vote Weight Configuration
# ========================================

# Assurance level multipliers
ASSURANCE_FACTORS = {
    "E0": 1.0,
    "E1": 1.2,
    "E2": 1.5,
    "E3": 2.0,
    "E4": 3.0,
}

# Base budget for quadratic voting
BASE_BUDGET = 100  # credits


# ========================================
# Identity Governance Extensions
# ========================================

class IdentityGovernanceExtensions:
    """
    Extensions for governance integration with identity system

    Provides methods for:
    - Identity verification for governance participation
    - Vote weight calculation
    - Capability-based access control
    - Guardian authorization management
    """

    def __init__(self, identity_coordinator):
        """
        Initialize governance extensions

        Args:
            identity_coordinator: DHT_IdentityCoordinator instance
        """
        self.coordinator = identity_coordinator

        # Track capability invocations for rate limiting
        self.capability_invocations: Dict[str, List[int]] = {}

        # Track pending guardian authorization requests
        self.pending_authorizations: Dict[str, Dict[str, Any]] = {}

    # ========================================
    # Identity Verification for Governance
    # ========================================

    async def verify_identity_for_governance(
        self,
        participant_id: str,
        required_assurance: str = "E1"
    ) -> Dict[str, Any]:
        """
        Verify identity meets governance participation requirements

        Args:
            participant_id: Participant identifier
            required_assurance: Minimum assurance level (E0-E4)

        Returns:
            Dict with verification result:
            {
                "verified": bool,
                "participant_id": str,
                "did": str,
                "assurance_level": str,
                "sybil_resistance": float,
                "reputation": float,
                "risk_level": str,
                "reasons": List[str]
            }
        """
        # Use FL verification as base (same logic applies)
        verification = await self.coordinator.verify_identity_for_fl(
            participant_id=participant_id,
            required_assurance=required_assurance
        )

        # Add governance-specific checks
        if verification["verified"]:
            # Check if identity has sufficient history for governance
            identity = await self.coordinator.get_identity(participant_id=participant_id)

            if identity:
                # Identity should exist for at least some time
                did_document = identity.get("did_document", {})
                created_at = did_document.get("created", 0)
                age_seconds = time.time() - (created_at / 1_000_000 if created_at > 0 else 0)

                # Minimum age: 1 hour for basic participation
                if age_seconds < 3600:
                    verification["verified"] = False
                    verification["reasons"].append(
                        f"Identity too new for governance: {age_seconds/3600:.1f} hours old (need 1+ hours)"
                    )

        return verification

    # ========================================
    # Vote Weight Calculation
    # ========================================

    async def calculate_vote_weight(
        self,
        participant_id: str
    ) -> float:
        """
        Calculate reputation-weighted vote power

        Formula:
            vote_weight = base_weight × (1 + sybil_resistance) × assurance_factor × reputation_factor

        Where:
            base_weight = 1.0
            sybil_resistance = 0.0-1.0 (from identity signals)
            assurance_factor = {E0: 1.0, E1: 1.2, E2: 1.5, E3: 2.0, E4: 3.0}
            reputation_factor = 0.5 + (reputation × 0.5)  # Maps 0.0-1.0 rep to 0.5-1.0

        Args:
            participant_id: Participant identifier

        Returns:
            Vote weight (typically 0.5-10.0, with most in 0.6-4.0 range)
        """
        # Verify identity and get signals
        verification = await self.verify_identity_for_governance(participant_id, "E0")

        if not verification["verified"]:
            # Minimal vote weight for unverified identities
            return 0.5

        # Extract components
        assurance_level = verification.get("assurance_level", "E0")
        sybil_resistance = verification.get("sybil_resistance", 0.0)
        reputation = verification.get("reputation", 0.0)

        # Calculate factors
        base_weight = 1.0
        sybil_bonus = 1.0 + sybil_resistance  # 1.0-2.0
        assurance_factor = ASSURANCE_FACTORS.get(assurance_level, 1.0)
        reputation_factor = 0.5 + (reputation * 0.5)  # 0.5-1.0

        # Composite weight
        vote_weight = base_weight * sybil_bonus * assurance_factor * reputation_factor

        return vote_weight

    async def calculate_vote_budget(
        self,
        participant_id: str
    ) -> int:
        """
        Calculate quadratic voting budget (in credits)

        Budget = BASE_BUDGET × vote_weight

        Args:
            participant_id: Participant identifier

        Returns:
            Vote budget in credits (typically 50-1000)
        """
        vote_weight = await self.calculate_vote_weight(participant_id)
        budget = int(BASE_BUDGET * vote_weight)
        return budget

    def calculate_effective_votes(
        self,
        credits_spent: int
    ) -> float:
        """
        Calculate effective votes from credits spent (quadratic formula)

        effective_votes = sqrt(credits_spent)

        Args:
            credits_spent: Number of credits spent on vote

        Returns:
            Effective votes (square root of credits)
        """
        return math.sqrt(credits_spent)

    # ========================================
    # Capability-Based Access Control
    # ========================================

    async def authorize_capability(
        self,
        participant_id: str,
        capability_id: str
    ) -> Tuple[bool, str]:
        """
        Verify participant has permission for capability

        Checks:
        1. Identity assurance level
        2. Reputation threshold
        3. Sybil resistance
        4. Guardian approval (if required)
        5. Rate limiting

        Args:
            participant_id: Participant identifier
            capability_id: Capability to check

        Returns:
            Tuple of (authorized: bool, reason: str)
        """
        # Get capability definition
        capability = CAPABILITY_REGISTRY.get(capability_id)

        if not capability:
            return False, f"Unknown capability: {capability_id}"

        # Step 1: Verify identity meets assurance requirement
        verification = await self.verify_identity_for_governance(
            participant_id=participant_id,
            required_assurance=capability.required_assurance
        )

        if not verification["verified"]:
            reasons = ", ".join(verification.get("reasons", ["Unknown"]))
            return False, f"Insufficient identity assurance: {reasons}"

        # Step 2: Check reputation threshold
        reputation = verification.get("reputation", 0.0)
        if reputation < capability.required_reputation:
            return False, (
                f"Insufficient reputation: {reputation:.2f} < "
                f"{capability.required_reputation:.2f}"
            )

        # Step 3: Check Sybil resistance
        sybil_resistance = verification.get("sybil_resistance", 0.0)
        if sybil_resistance < capability.required_sybil_resistance:
            return False, (
                f"Insufficient Sybil resistance: {sybil_resistance:.2f} < "
                f"{capability.required_sybil_resistance:.2f}"
            )

        # Step 4: Check guardian approval (if required)
        if capability.required_guardian_approval:
            # Note: Guardian approval is async and must be requested separately
            # This check verifies a pending authorization exists
            if not self._has_pending_authorization(participant_id, capability_id):
                return False, (
                    f"Guardian approval required (threshold: "
                    f"{capability.guardian_threshold:.0%}). "
                    f"Call request_guardian_authorization() first."
                )

        # Step 5: Check rate limiting
        if capability.rate_limit:
            recent_count = self._count_recent_invocations(
                participant_id=participant_id,
                capability_id=capability_id,
                time_period_seconds=capability.time_period_seconds
            )

            if recent_count >= capability.rate_limit:
                return False, (
                    f"Rate limit exceeded: {recent_count}/{capability.rate_limit} "
                    f"in last {capability.time_period_seconds}s"
                )

        return True, "Authorized"

    def _has_pending_authorization(
        self,
        participant_id: str,
        capability_id: str
    ) -> bool:
        """Check if participant has pending authorization for capability"""
        key = f"{participant_id}:{capability_id}"
        auth = self.pending_authorizations.get(key)

        if not auth:
            return False

        # Check if authorization is still valid (not expired)
        if auth.get("expires_at", 0) < time.time():
            # Expired, remove
            del self.pending_authorizations[key]
            return False

        # Check if authorized
        return auth.get("status") == "APPROVED"

    def _count_recent_invocations(
        self,
        participant_id: str,
        capability_id: str,
        time_period_seconds: int
    ) -> int:
        """Count recent invocations for rate limiting"""
        key = f"{participant_id}:{capability_id}"
        invocations = self.capability_invocations.get(key, [])

        # Filter to recent invocations
        cutoff = time.time() - time_period_seconds
        recent = [ts for ts in invocations if ts >= cutoff]

        # Update storage
        self.capability_invocations[key] = recent

        return len(recent)

    def record_capability_invocation(
        self,
        participant_id: str,
        capability_id: str
    ):
        """Record capability invocation for rate limiting"""
        key = f"{participant_id}:{capability_id}"

        if key not in self.capability_invocations:
            self.capability_invocations[key] = []

        self.capability_invocations[key].append(time.time())

    # ========================================
    # Guardian Authorization Management
    # ========================================

    async def request_guardian_authorization(
        self,
        participant_id: str,
        capability_id: str,
        action_params: Dict[str, Any],
        timeout_seconds: int = 3600
    ) -> str:
        """
        Request guardian authorization for emergency action

        Args:
            participant_id: Participant requesting authorization
            capability_id: Capability requiring guardian approval
            action_params: Action-specific parameters
            timeout_seconds: How long request is valid (default 1 hour)

        Returns:
            request_id: Unique identifier for authorization request
        """
        # Get capability
        capability = CAPABILITY_REGISTRY.get(capability_id)

        if not capability:
            raise ValueError(f"Unknown capability: {capability_id}")

        if not capability.required_guardian_approval:
            raise ValueError(f"Capability {capability_id} does not require guardian approval")

        # Generate request ID
        request_id = f"auth_{participant_id}_{capability_id}_{int(time.time())}"

        # Store pending authorization
        key = f"{participant_id}:{capability_id}"
        self.pending_authorizations[key] = {
            "request_id": request_id,
            "participant_id": participant_id,
            "capability_id": capability_id,
            "action_params": action_params,
            "required_threshold": capability.guardian_threshold,
            "status": "PENDING",
            "created_at": time.time(),
            "expires_at": time.time() + timeout_seconds,
            "approving_guardians": []
        }

        # Notify guardians via the external notification system
        await self._notify_guardians(
            participant_id=participant_id,
            request_id=request_id,
            capability_id=capability_id,
            capability_name=capability.name,
            action_params=action_params,
            required_threshold=capability.guardian_threshold,
            timeout_seconds=timeout_seconds
        )

        return request_id

    async def _notify_guardians(
        self,
        participant_id: str,
        request_id: str,
        capability_id: str,
        capability_name: str,
        action_params: Dict[str, Any],
        required_threshold: float,
        timeout_seconds: int
    ):
        """
        Notify guardians about a pending authorization request.

        Sends notifications via multiple channels:
        1. DHT signal broadcast to guardian network
        2. Webhook notifications to registered endpoints
        3. In-app notification storage for guardian retrieval

        Args:
            participant_id: Participant requesting authorization
            request_id: Unique authorization request ID
            capability_id: Capability being requested
            capability_name: Human-readable capability name
            action_params: Parameters for the action
            required_threshold: Required guardian approval threshold
            timeout_seconds: Time until request expires
        """
        # Get the identity to find associated guardians
        identity = await self.coordinator.get_identity(participant_id=participant_id)
        if not identity:
            return

        # Get guardian list from identity document
        did_document = identity.get("did_document", {})
        guardian_dids = did_document.get("guardian", [])

        if not guardian_dids:
            # No guardians configured - this is okay, request will timeout
            return

        # Build notification payload
        notification_payload = {
            "type": "guardian_authorization_request",
            "request_id": request_id,
            "subject_participant_id": participant_id,
            "subject_did": did_document.get("id", ""),
            "capability_id": capability_id,
            "capability_name": capability_name,
            "action_params": action_params,
            "required_threshold": required_threshold,
            "expires_at": time.time() + timeout_seconds,
            "created_at": time.time(),
            "message": f"Authorization requested for '{capability_name}' by {participant_id}"
        }

        # Channel 1: DHT signal broadcast
        # This notifies any guardians connected to the DHT network
        try:
            if hasattr(self.coordinator, 'dht') and self.coordinator.dht:
                await self.coordinator.dht.call_zome(
                    zome_name="governance_record",
                    fn_name="broadcast_guardian_notification",
                    payload={
                        "guardian_dids": guardian_dids,
                        "notification": notification_payload
                    }
                )
        except Exception as e:
            # DHT notification is best-effort, don't fail the request
            pass

        # Channel 2: Store in-app notification for guardian retrieval
        # Guardians can poll for pending notifications
        try:
            if hasattr(self.coordinator, 'dht') and self.coordinator.dht:
                for guardian_did in guardian_dids:
                    await self.coordinator.dht.call_zome(
                        zome_name="governance_record",
                        fn_name="store_guardian_notification",
                        payload={
                            "guardian_did": guardian_did,
                            "notification": notification_payload,
                            "status": "unread"
                        }
                    )
        except Exception as e:
            # Notification storage is best-effort
            pass

        # Channel 3: Webhook notifications (if configured)
        # Look up webhook endpoints for each guardian
        try:
            if hasattr(self.coordinator, 'dht') and self.coordinator.dht:
                for guardian_did in guardian_dids:
                    # Get guardian's notification preferences
                    guardian_prefs = await self.coordinator.dht.call_zome(
                        zome_name="identity",
                        fn_name="get_notification_preferences",
                        payload={"did": guardian_did}
                    )

                    if guardian_prefs and guardian_prefs.get("webhook_url"):
                        webhook_url = guardian_prefs["webhook_url"]
                        # Send webhook notification asynchronously
                        import aiohttp
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.post(
                                    webhook_url,
                                    json=notification_payload,
                                    timeout=aiohttp.ClientTimeout(total=10)
                                ) as response:
                                    if response.status == 200:
                                        pass  # Successfully notified
                        except Exception:
                            # Webhook delivery is best-effort
                            pass
        except Exception as e:
            # Webhook notifications are optional
            pass

    async def check_authorization_status(
        self,
        request_id: str
    ) -> Dict[str, Any]:
        """
        Check status of guardian authorization request

        Args:
            request_id: Authorization request identifier

        Returns:
            Dict with authorization status:
            {
                "request_id": str,
                "status": str,  # "PENDING", "APPROVED", "REJECTED", "EXPIRED"
                "approval_weight": float,
                "required_threshold": float,
                "approving_guardians": List[str],
                "created_at": float,
                "expires_at": float
            }
        """
        # Find authorization by request_id
        for key, auth in self.pending_authorizations.items():
            if auth["request_id"] == request_id:
                # Check if expired
                if auth["expires_at"] < time.time():
                    auth["status"] = "EXPIRED"
                    return {
                        "request_id": request_id,
                        "status": "EXPIRED",
                        "approval_weight": 0.0,
                        "required_threshold": auth["required_threshold"],
                        "approving_guardians": [],
                        "created_at": auth["created_at"],
                        "expires_at": auth["expires_at"]
                    }

                # Calculate current approval weight
                participant_id = auth["participant_id"]
                approving_guardians = auth["approving_guardians"]

                # Query guardian weights from identity coordinator
                approval_result = await self.coordinator.authorize_recovery(
                    subject_participant_id=participant_id,
                    approving_guardian_ids=approving_guardians,
                    required_threshold=auth["required_threshold"]
                )

                # Update status if threshold met
                if approval_result["authorized"]:
                    auth["status"] = "APPROVED"

                return {
                    "request_id": request_id,
                    "status": auth["status"],
                    "approval_weight": approval_result["approval_weight"],
                    "required_threshold": auth["required_threshold"],
                    "approving_guardians": approving_guardians,
                    "created_at": auth["created_at"],
                    "expires_at": auth["expires_at"]
                }

        return {
            "request_id": request_id,
            "status": "NOT_FOUND",
            "approval_weight": 0.0,
            "required_threshold": 0.0,
            "approving_guardians": [],
            "created_at": 0.0,
            "expires_at": 0.0
        }

    async def approve_authorization(
        self,
        request_id: str,
        guardian_id: str,
        approve: bool = True
    ) -> Dict[str, Any]:
        """
        Guardian approves or rejects authorization request

        Args:
            request_id: Authorization request identifier
            guardian_id: Guardian participant ID
            approve: True to approve, False to reject

        Returns:
            Updated authorization status
        """
        # Find authorization
        for key, auth in self.pending_authorizations.items():
            if auth["request_id"] == request_id:
                # Verify guardian relationship
                is_guardian = await self.coordinator.authorize_recovery(
                    subject_participant_id=auth["participant_id"],
                    approving_guardian_ids=[guardian_id],
                    required_threshold=0.0  # Just checking if guardian exists
                )

                if is_guardian["approval_weight"] == 0.0:
                    raise PermissionError(
                        f"{guardian_id} is not a guardian of {auth['participant_id']}"
                    )

                # Record approval/rejection
                if approve:
                    if guardian_id not in auth["approving_guardians"]:
                        auth["approving_guardians"].append(guardian_id)
                else:
                    # Remove from approving list if previously approved
                    if guardian_id in auth["approving_guardians"]:
                        auth["approving_guardians"].remove(guardian_id)

                # Return updated status
                return await self.check_authorization_status(request_id)

        raise ValueError(f"Authorization request not found: {request_id}")

    # ========================================
    # Governance Participation Tracking
    # ========================================

    async def get_governance_stats(
        self,
        participant_id: str
    ) -> Dict[str, Any]:
        """
        Get governance participation statistics for participant.

        Queries governance DHT records to aggregate comprehensive statistics
        about a participant's governance activity.

        Args:
            participant_id: Participant identifier

        Returns:
            Dict with governance statistics:
            {
                "proposals_submitted": int,
                "votes_cast": int,
                "capabilities_invoked": Dict[str, int],
                "authorizations_requested": int,
                "authorizations_approved": int,
                "guardian_approvals_given": int,
                "reputation_score": float,
                "last_activity": float (timestamp)
            }
        """
        # Initialize stats with local tracking data
        capability_counts = {}
        for key, invocations in self.capability_invocations.items():
            if key.startswith(f"{participant_id}:"):
                capability_id = key.split(":", 1)[1]
                capability_counts[capability_id] = len(invocations)

        # Count local authorization requests
        local_auth_count = sum(
            1 for auth in self.pending_authorizations.values()
            if auth["participant_id"] == participant_id
        )

        local_approved_count = sum(
            1 for auth in self.pending_authorizations.values()
            if auth["participant_id"] == participant_id and auth.get("status") == "APPROVED"
        )

        # Initialize result with local data
        stats = {
            "proposals_submitted": capability_counts.get("submit_mip", 0),
            "votes_cast": capability_counts.get("vote_on_mip", 0),
            "capabilities_invoked": capability_counts,
            "authorizations_requested": local_auth_count,
            "authorizations_approved": local_approved_count,
            "guardian_approvals_given": 0,
            "reputation_score": 0.5,
            "last_activity": 0.0,
        }

        # Query DHT for comprehensive statistics
        if hasattr(self.coordinator, 'dht') and self.coordinator.dht:
            try:
                # Query proposals submitted by this participant
                proposals_result = await self.coordinator.dht.call_zome(
                    zome_name="governance_record",
                    fn_name="count_proposals_by_participant",
                    payload={"participant_id": participant_id}
                )
                if proposals_result:
                    stats["proposals_submitted"] = max(
                        stats["proposals_submitted"],
                        proposals_result.get("count", 0)
                    )

                # Query votes cast by this participant
                votes_result = await self.coordinator.dht.call_zome(
                    zome_name="governance_record",
                    fn_name="count_votes_by_participant",
                    payload={"participant_id": participant_id}
                )
                if votes_result:
                    stats["votes_cast"] = max(
                        stats["votes_cast"],
                        votes_result.get("count", 0)
                    )

                # Query authorization requests
                auth_result = await self.coordinator.dht.call_zome(
                    zome_name="governance_record",
                    fn_name="get_authorization_history",
                    payload={"participant_id": participant_id}
                )
                if auth_result:
                    auth_list = auth_result.get("authorizations", [])
                    stats["authorizations_requested"] = max(
                        stats["authorizations_requested"],
                        len(auth_list)
                    )
                    stats["authorizations_approved"] = sum(
                        1 for a in auth_list if a.get("status") == "APPROVED"
                    )

                # Query guardian approvals given by this participant
                guardian_result = await self.coordinator.dht.call_zome(
                    zome_name="governance_record",
                    fn_name="count_guardian_approvals",
                    payload={"guardian_participant_id": participant_id}
                )
                if guardian_result:
                    stats["guardian_approvals_given"] = guardian_result.get("count", 0)

                # Get reputation score from identity system
                identity = await self.coordinator.get_identity(participant_id=participant_id)
                if identity:
                    signals = identity.get("identity_signals", {})
                    stats["reputation_score"] = signals.get("reputation", 0.5)

                # Query last governance activity timestamp
                activity_result = await self.coordinator.dht.call_zome(
                    zome_name="governance_record",
                    fn_name="get_last_activity",
                    payload={"participant_id": participant_id}
                )
                if activity_result:
                    stats["last_activity"] = activity_result.get("timestamp", 0.0)

                # Query detailed capability invocation history from DHT
                capability_history = await self.coordinator.dht.call_zome(
                    zome_name="governance_record",
                    fn_name="get_capability_history",
                    payload={"participant_id": participant_id}
                )
                if capability_history:
                    dht_capability_counts = capability_history.get("capabilities", {})
                    # Merge with local counts, taking max
                    for cap_id, count in dht_capability_counts.items():
                        current = stats["capabilities_invoked"].get(cap_id, 0)
                        stats["capabilities_invoked"][cap_id] = max(current, count)

            except Exception as e:
                # DHT query failure - fall back to local data
                # This is non-critical, so we don't raise
                pass

        return stats

    def get_governance_stats_sync(
        self,
        participant_id: str
    ) -> Dict[str, Any]:
        """
        Synchronous version of get_governance_stats for backward compatibility.

        Returns local tracking data only (no DHT queries).

        Args:
            participant_id: Participant identifier

        Returns:
            Dict with local governance statistics
        """
        # Count capability invocations from local tracking
        capability_counts = {}
        for key, invocations in self.capability_invocations.items():
            if key.startswith(f"{participant_id}:"):
                capability_id = key.split(":", 1)[1]
                capability_counts[capability_id] = len(invocations)

        # Count authorization requests
        auth_count = sum(
            1 for auth in self.pending_authorizations.values()
            if auth["participant_id"] == participant_id
        )

        approved_count = sum(
            1 for auth in self.pending_authorizations.values()
            if auth["participant_id"] == participant_id and auth.get("status") == "APPROVED"
        )

        return {
            "proposals_submitted": capability_counts.get("submit_mip", 0),
            "votes_cast": capability_counts.get("vote_on_mip", 0),
            "capabilities_invoked": capability_counts,
            "authorizations_requested": auth_count,
            "authorizations_approved": approved_count,
            "guardian_approvals_given": 0,
            "reputation_score": 0.5,
            "last_activity": 0.0,
        }
