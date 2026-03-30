# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Governance Coordinator Implementation
Week 7-8 Phase 3: Governance Coordinator

Complete governance system with proposal management, voting, capability enforcement,
and guardian authorization for emergency actions.
"""

import json
import os
import time
import hashlib
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import asdict, dataclass, field
from collections import deque

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import (
    Encoding, PrivateFormat, PublicFormat, NoEncryption
)

# Unified logging and error handling
from zerotrustml.logging import get_logger, log_operation, async_log_operation
from zerotrustml.exceptions import (
    GovernanceError,
    GovernanceUnauthorizedError,
    CapabilityDeniedError,
    GuardianAuthorizationError,
    IdentityNotFoundError,
    ConfigurationError,
    ErrorCode,
)
from zerotrustml.error_handling import async_retry, Fallback

from .models import (
    ProposalData,
    ProposalType,
    ProposalStatus,
    VoteData,
    VoteChoice,
    AuthorizationRequestData,
    GuardianApprovalData,
    AuthorizationStatus,
    ProposalSignature,
    GuardianRelationship,
    VotingPowerCalculation,
    VoteTally,
)

logger = get_logger(__name__)


class GovernanceSigner:
    """
    Ed25519 cryptographic signer for governance operations.

    Manages key generation, loading, signing, and verification for
    governance messages (votes, guardian approvals, etc.).

    Keys are stored in PEM format in a configurable directory.
    """

    def __init__(self, key_dir: Optional[Path] = None, participant_id: Optional[str] = None):
        """
        Initialize the governance signer.

        Args:
            key_dir: Directory to store/load keys. Defaults to ~/.mycelix/governance_keys/
            participant_id: Participant ID for key isolation. If provided, keys are stored
                           in a subdirectory named after the participant.
        """
        if key_dir is None:
            key_dir = Path.home() / ".mycelix" / "governance_keys"

        if participant_id:
            # Sanitize participant_id for filesystem safety
            safe_id = hashlib.sha256(participant_id.encode()).hexdigest()[:16]
            key_dir = key_dir / safe_id

        self.key_dir = key_dir
        self.key_dir.mkdir(parents=True, exist_ok=True)

        self._private_key: Optional[ed25519.Ed25519PrivateKey] = None
        self._public_key: Optional[ed25519.Ed25519PublicKey] = None

        # Initialize keys
        self._initialize_keys()

    def _initialize_keys(self) -> None:
        """Load existing keys or generate new ones."""
        private_key_file = self.key_dir / "governance_signing_key.pem"
        public_key_file = self.key_dir / "governance_verify_key.pem"

        if private_key_file.exists():
            # Load existing keys
            try:
                with open(private_key_file, "rb") as f:
                    self._private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None
                    )
                with open(public_key_file, "rb") as f:
                    self._public_key = serialization.load_pem_public_key(f.read())

                logger.info(f"Loaded governance signing keys from {self.key_dir}")
            except Exception as e:
                logger.error(f"Failed to load governance keys: {e}")
                raise RuntimeError(f"Failed to load governance keys: {e}")
        else:
            # Generate new keys
            self._private_key = ed25519.Ed25519PrivateKey.generate()
            self._public_key = self._private_key.public_key()

            # Save keys with restrictive permissions
            try:
                # Save private key
                private_key_file.write_bytes(
                    self._private_key.private_bytes(
                        encoding=Encoding.PEM,
                        format=PrivateFormat.PKCS8,
                        encryption_algorithm=NoEncryption()
                    )
                )
                # Set restrictive permissions (owner read/write only)
                os.chmod(private_key_file, 0o600)

                # Save public key
                public_key_file.write_bytes(
                    self._public_key.public_bytes(
                        encoding=Encoding.PEM,
                        format=PublicFormat.SubjectPublicKeyInfo
                    )
                )

                logger.info(f"Generated new governance signing keys in {self.key_dir}")
            except Exception as e:
                logger.error(f"Failed to save governance keys: {e}")
                raise RuntimeError(f"Failed to save governance keys: {e}")

    def sign(self, message: bytes) -> bytes:
        """
        Sign a message using the Ed25519 private key.

        Args:
            message: The message bytes to sign

        Returns:
            The signature bytes (64 bytes for Ed25519)

        Raises:
            RuntimeError: If signing key is not available
        """
        if not self._private_key:
            raise RuntimeError("Signing key not initialized")

        return self._private_key.sign(message)

    def sign_governance_data(self, data: Dict[str, Any]) -> str:
        """
        Sign governance data (vote, approval, etc.) and return hex-encoded signature.

        The data is serialized to JSON with sorted keys for deterministic signing.

        Args:
            data: Dictionary containing the governance data to sign

        Returns:
            Hex-encoded signature string
        """
        # Create deterministic JSON representation
        message = json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')
        signature = self.sign(message)
        return signature.hex()

    def verify(self, message: bytes, signature: bytes, public_key: ed25519.Ed25519PublicKey) -> bool:
        """
        Verify a message signature.

        Args:
            message: The original message bytes
            signature: The signature to verify
            public_key: The public key to verify against

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            public_key.verify(signature, message)
            return True
        except Exception:
            return False

    def verify_governance_signature(
        self,
        data: Dict[str, Any],
        signature_hex: str,
        public_key_bytes: bytes
    ) -> bool:
        """
        Verify a governance data signature.

        Args:
            data: The governance data that was signed
            signature_hex: Hex-encoded signature string
            public_key_bytes: Raw public key bytes (32 bytes for Ed25519)

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            message = json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')
            signature = bytes.fromhex(signature_hex)
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
            return self.verify(message, signature, public_key)
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False

    def get_public_key_bytes(self) -> bytes:
        """
        Get the raw public key bytes for sharing.

        Returns:
            32-byte public key in raw format
        """
        if not self._public_key:
            raise RuntimeError("Public key not initialized")

        return self._public_key.public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw
        )

    def get_public_key_hex(self) -> str:
        """
        Get the public key as a hex string for storage/sharing.

        Returns:
            Hex-encoded public key string
        """
        return self.get_public_key_bytes().hex()

    def get_public_key_pem(self) -> bytes:
        """
        Get the public key in PEM format.

        Returns:
            PEM-encoded public key bytes
        """
        if not self._public_key:
            raise RuntimeError("Public key not initialized")

        return self._public_key.public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo
        )


class ProposalManager:
    """
    Manages proposal lifecycle from creation to execution

    Responsibilities:
    - Create and validate proposals
    - Sign proposals with Ed25519 for cryptographic authenticity
    - Transition proposal status through lifecycle
    - Store proposals on DHT
    - Query proposals by status/type
    """

    def __init__(
        self,
        dht_client,
        identity_governance_extensions,
        governance_signer: Optional[GovernanceSigner] = None,
    ):
        """
        Args:
            dht_client: Holochain DHT client for governance_record zome
            identity_governance_extensions: Identity governance extensions instance
            governance_signer: Optional Ed25519 signer for proposal authentication
        """
        self.dht = dht_client
        self.gov_extensions = identity_governance_extensions
        self.signer = governance_signer

        # Cache for proposal signatures
        self._proposal_signatures: Dict[str, ProposalSignature] = {}

    async def create_proposal(
        self,
        proposer_participant_id: str,
        proposal_type: ProposalType,
        title: str,
        description: str,
        voting_duration_seconds: int,
        execution_params: Dict[str, Any],
        quorum: float = 0.5,
        approval_threshold: float = 0.6,
        tags: Optional[List[str]] = None,
    ) -> Tuple[bool, str, Optional[ProposalData]]:
        """
        Create a new governance proposal

        Args:
            proposer_participant_id: Participant ID of proposer
            proposal_type: Type of proposal (PARAMETER_CHANGE, etc.)
            title: Proposal title
            description: Detailed description
            voting_duration_seconds: How long voting period lasts
            execution_params: Parameters for execution if approved
            quorum: Minimum voting power participation (0.0-1.0)
            approval_threshold: Minimum approval ratio (0.5-1.0)
            tags: Optional tags for categorization

        Returns:
            Tuple of (success, message, proposal_data)
        """
        # Step 1: Verify proposer has capability
        can_propose, reason = await self.gov_extensions.authorize_capability(
            participant_id=proposer_participant_id,
            capability_id="submit_mip"
        )

        if not can_propose:
            return False, f"Proposer not authorized: {reason}", None

        # Step 2: Get proposer's DID
        identity = await self.gov_extensions.coordinator.get_identity(
            participant_id=proposer_participant_id
        )
        if not identity:
            return False, "Proposer identity not found", None

        proposer_did = identity.get("did", "")

        # Step 3: Generate proposal ID
        proposal_id = self._generate_proposal_id(proposer_participant_id, title)

        # Step 4: Set voting period
        now = int(time.time())
        voting_start = now + 3600  # Start 1 hour from now (review period)
        voting_end = voting_start + voting_duration_seconds

        # Step 5: Create proposal data
        proposal = ProposalData(
            proposal_id=proposal_id,
            proposal_type=proposal_type,
            title=title,
            description=description,
            proposer_did=proposer_did,
            proposer_participant_id=proposer_participant_id,
            voting_start=voting_start,
            voting_end=voting_end,
            quorum=quorum,
            approval_threshold=approval_threshold,
            status=ProposalStatus.SUBMITTED,
            execution_params=execution_params,
            tags=tags or [],
            created_at=now,
            updated_at=now,
        )

        # Step 6: Sign proposal with Ed25519
        proposal_signature = None
        if self.signer:
            try:
                # Create signable data for the proposal
                signable_data = {
                    "proposal_id": proposal_id,
                    "proposer_did": proposer_did,
                    "signed_at": now,
                    "title": title,
                    "proposal_type": proposal_type.value,
                }
                signature_hex = self.signer.sign_governance_data(signable_data)

                proposal_signature = ProposalSignature(
                    proposal_id=proposal_id,
                    proposer_did=proposer_did,
                    proposer_public_key=self.signer.get_public_key_hex(),
                    signature=signature_hex,
                    signed_at=now,
                )
                self._proposal_signatures[proposal_id] = proposal_signature
                logger.debug(f"Proposal signed with Ed25519: {proposal_id}")
            except Exception as e:
                logger.error(f"Failed to sign proposal: {e}")
                return False, f"Proposal signing failed: {str(e)}", None
        else:
            logger.warning("No governance signer configured - proposal will be unsigned")

        # Step 7: Store on DHT
        try:
            proposal_payload = proposal.to_dict()
            if proposal_signature:
                proposal_payload["signature"] = proposal_signature.to_dict()

            await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="store_proposal",
                payload=proposal_payload,
            )
        except Exception as e:
            logger.error(f"Failed to store proposal on DHT: {e}")
            return False, f"DHT storage failed: {str(e)}", None

        # Step 8: Record capability invocation
        self.gov_extensions._record_invocation(proposer_participant_id, "submit_mip")

        logger.info(
            f"Proposal created: {proposal_id} by {proposer_participant_id}, "
            f"voting {voting_start} to {voting_end}, signed={proposal_signature is not None}"
        )

        return True, "Proposal created successfully", proposal

    async def verify_proposal_signature(
        self,
        proposal_id: str,
        public_key_bytes: Optional[bytes] = None,
    ) -> Tuple[bool, str]:
        """
        Verify the Ed25519 signature on a proposal.

        Args:
            proposal_id: The proposal to verify
            public_key_bytes: Optional proposer's public key. If not provided,
                            attempts to retrieve from cached signature.

        Returns:
            Tuple of (valid, message)
        """
        if not self.signer:
            return False, "No governance signer configured for verification"

        # Get cached signature
        proposal_signature = self._proposal_signatures.get(proposal_id)
        if not proposal_signature:
            return False, "No signature found for proposal"

        # Get public key from signature if not provided
        if public_key_bytes is None:
            try:
                public_key_bytes = bytes.fromhex(proposal_signature.proposer_public_key)
            except Exception as e:
                return False, f"Invalid public key in signature: {e}"

        # Get proposal to reconstruct signable data
        proposal = await self.get_proposal(proposal_id)
        if not proposal:
            return False, "Proposal not found"

        # Reconstruct signable data
        signable_data = {
            "proposal_id": proposal_id,
            "proposer_did": proposal.proposer_did,
            "signed_at": proposal_signature.signed_at,
            "title": proposal.title,
            "proposal_type": proposal.proposal_type.value,
        }

        # Verify signature
        try:
            is_valid = self.signer.verify_governance_signature(
                data=signable_data,
                signature_hex=proposal_signature.signature,
                public_key_bytes=public_key_bytes,
            )
            if is_valid:
                return True, "Proposal signature verified"
            else:
                return False, "Proposal signature verification failed"
        except Exception as e:
            logger.error(f"Proposal signature verification error: {e}")
            return False, f"Verification error: {str(e)}"

    def get_proposal_signature(self, proposal_id: str) -> Optional[ProposalSignature]:
        """Get the signature for a proposal if available."""
        return self._proposal_signatures.get(proposal_id)

    async def get_proposal(self, proposal_id: str) -> Optional[ProposalData]:
        """
        Retrieve proposal by ID

        Args:
            proposal_id: Unique proposal identifier

        Returns:
            ProposalData if found, None otherwise
        """
        try:
            result = await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="get_proposal",
                payload={"proposal_id": proposal_id},
            )

            if result:
                return ProposalData.from_dict(result)
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve proposal {proposal_id}: {e}")
            return None

    async def list_proposals_by_status(
        self,
        status: ProposalStatus,
        limit: int = 100
    ) -> List[ProposalData]:
        """
        List proposals by status

        Args:
            status: Proposal status to filter by
            limit: Maximum number to return

        Returns:
            List of matching proposals
        """
        try:
            results = await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="list_proposals_by_status",
                payload={"status": status.value, "limit": limit},
            )

            return [ProposalData.from_dict(p) for p in results]

        except Exception as e:
            logger.error(f"Failed to list proposals by status {status}: {e}")
            return []

    async def update_proposal_status(
        self,
        proposal_id: str,
        new_status: ProposalStatus,
        vote_totals: Optional[Dict[str, float]] = None,
        execution_result: Optional[str] = None,
    ) -> bool:
        """
        Update proposal status after vote tallying or execution

        Args:
            proposal_id: Proposal to update
            new_status: New status
            vote_totals: Optional vote totals (for/against/abstain/total)
            execution_result: Optional execution result

        Returns:
            True if updated successfully
        """
        try:
            payload = {
                "proposal_id": proposal_id,
                "status": new_status.value,
                "updated_at": int(time.time()),
            }

            if vote_totals:
                payload.update({
                    "total_votes_for": vote_totals.get("for", 0.0),
                    "total_votes_against": vote_totals.get("against", 0.0),
                    "total_votes_abstain": vote_totals.get("abstain", 0.0),
                    "total_voting_power": vote_totals.get("total", 0.0),
                })

            if execution_result:
                payload["execution_result"] = execution_result
                payload["executed_at"] = int(time.time())

            await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="update_proposal_status",
                payload=payload,
            )

            logger.info(f"Proposal {proposal_id} status updated to {new_status.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to update proposal {proposal_id} status: {e}")
            return False

    def _generate_proposal_id(self, proposer_id: str, title: str) -> str:
        """Generate unique proposal ID"""
        data = f"{proposer_id}:{title}:{int(time.time())}"
        return f"prop_{hashlib.sha256(data.encode()).hexdigest()[:16]}"


# ========================================
# Time weight decay constants
# ========================================
TIME_WEIGHT_HALF_LIFE_DAYS = 30  # Activity weight halves every 30 days
TIME_WEIGHT_MIN = 0.5  # Minimum time weight for inactive participants


class VotingPowerCalculator:
    """
    Calculates voting power based on MATL reputation, guardian endorsements,
    and time-weighted activity.

    Formula:
        voting_power = base_weight
                      * assurance_multiplier
                      * reputation_multiplier
                      * (1 + sybil_bonus)
                      * (1 + guardian_endorsement_bonus)
                      * time_weight

    Where:
        - base_weight = 1.0
        - assurance_multiplier = {E0: 1.0, E1: 1.2, E2: 1.5, E3: 2.0, E4: 3.0}
        - reputation_multiplier = 0.5 + (reputation * 0.5)  # 0.5-1.0
        - sybil_bonus = sybil_resistance * 0.5  # 0.0-0.5
        - guardian_endorsement_bonus = sum(guardian_weights) * 0.1  # Up to 0.5
        - time_weight = max(0.5, 2^(-days_since_activity / 30))  # 0.5-1.0
    """

    # Assurance level multipliers
    ASSURANCE_MULTIPLIERS = {
        "E0": 1.0,
        "E1": 1.2,
        "E2": 1.5,
        "E3": 2.0,
        "E4": 3.0,
    }

    def __init__(self, identity_governance_extensions, dht_client=None):
        """
        Args:
            identity_governance_extensions: Identity governance extensions instance
            dht_client: Optional DHT client for guardian queries
        """
        self.gov_extensions = identity_governance_extensions
        self.dht = dht_client

        # Cache for guardian relationships (subject_id -> list of GuardianRelationship)
        self._guardian_cache: Dict[str, List[GuardianRelationship]] = {}

        # Cache for calculated voting power
        self._voting_power_cache: Dict[str, VotingPowerCalculation] = {}
        self._cache_ttl = 300  # 5 minute cache

    async def calculate_voting_power(
        self,
        participant_id: str,
        include_guardian_endorsements: bool = True,
    ) -> VotingPowerCalculation:
        """
        Calculate total voting power for a participant.

        Args:
            participant_id: Participant to calculate power for
            include_guardian_endorsements: Whether to include guardian bonuses

        Returns:
            VotingPowerCalculation with full breakdown
        """
        now = int(time.time())

        # Check cache
        cached = self._voting_power_cache.get(participant_id)
        if cached and (now - cached.calculated_at) < self._cache_ttl:
            return cached

        # Get identity verification
        verification = await self.gov_extensions.verify_identity_for_governance(
            participant_id=participant_id,
            required_assurance="E0"
        )

        # Initialize calculation
        calc = VotingPowerCalculation(participant_id=participant_id)
        calc.base_weight = 1.0

        if not verification.get("verified", False):
            # Minimal voting power for unverified
            calc.total_voting_power = 0.5
            calc.calculated_at = now
            self._voting_power_cache[participant_id] = calc
            return calc

        # Extract components from verification
        calc.assurance_level = verification.get("assurance_level", "E0")
        calc.reputation_score = verification.get("reputation", 0.0)
        calc.sybil_resistance = verification.get("sybil_resistance", 0.0)

        # Calculate assurance multiplier
        calc.assurance_multiplier = self.ASSURANCE_MULTIPLIERS.get(
            calc.assurance_level, 1.0
        )

        # Calculate reputation multiplier (0.5 to 1.0)
        calc.reputation_multiplier = 0.5 + (calc.reputation_score * 0.5)

        # Calculate sybil bonus (0.0 to 0.5)
        calc.sybil_bonus = calc.sybil_resistance * 0.5

        # Calculate guardian endorsement bonus
        if include_guardian_endorsements:
            endorsements = await self._get_guardian_endorsements(participant_id)
            calc.endorsing_guardians = [e.guardian_participant_id for e in endorsements]
            total_weight = sum(e.weight for e in endorsements if e.is_valid())
            calc.guardian_endorsements = min(total_weight * 0.1, 0.5)  # Cap at 0.5

        # Calculate time weight based on recent activity
        calc.last_activity_timestamp = await self._get_last_activity(participant_id)
        if calc.last_activity_timestamp > 0:
            days_since_activity = (now - calc.last_activity_timestamp) / 86400
            calc.time_weight = max(
                TIME_WEIGHT_MIN,
                math.pow(2, -days_since_activity / TIME_WEIGHT_HALF_LIFE_DAYS)
            )
        else:
            calc.time_weight = TIME_WEIGHT_MIN

        # Calculate total voting power
        calc.total_voting_power = (
            calc.base_weight
            * calc.assurance_multiplier
            * calc.reputation_multiplier
            * (1 + calc.sybil_bonus)
            * (1 + calc.guardian_endorsements)
            * calc.time_weight
        )

        calc.calculated_at = now
        self._voting_power_cache[participant_id] = calc

        logger.debug(
            f"Voting power for {participant_id}: {calc.total_voting_power:.2f} "
            f"(assurance={calc.assurance_level}, rep={calc.reputation_score:.2f}, "
            f"sybil={calc.sybil_resistance:.2f}, guardians={len(calc.endorsing_guardians)})"
        )

        return calc

    async def calculate_total_eligible_voting_power(
        self,
        participant_ids: List[str],
        batch_size: int = 50,
    ) -> float:
        """
        Calculate total eligible voting power for a set of participants.

        Uses parallel execution for improved performance on large participant sets.
        Includes caching and batching for efficiency.

        Args:
            participant_ids: List of participant IDs to include
            batch_size: Number of participants to process in parallel per batch

        Returns:
            Total voting power across all participants
        """
        if not participant_ids:
            return 0.0

        total = 0.0
        failed_count = 0

        # Process in batches for memory efficiency
        for i in range(0, len(participant_ids), batch_size):
            batch = participant_ids[i:i + batch_size]

            # Create tasks for parallel execution
            tasks = [
                self.calculate_voting_power(pid)
                for pid in batch
            ]

            # Execute batch in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate results
            for pid, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to calculate voting power for {pid}: {result}")
                    failed_count += 1
                elif result is not None:
                    total += result.total_voting_power

        # Log summary if there were failures
        if failed_count > 0:
            logger.info(
                f"Total eligible voting power calculation: {total:.4f} "
                f"({len(participant_ids) - failed_count}/{len(participant_ids)} participants succeeded)"
            )

        return total

    async def get_voting_power_distribution(
        self,
        participant_ids: List[str],
    ) -> Dict[str, VotingPowerCalculation]:
        """
        Get detailed voting power breakdown for multiple participants.

        Useful for governance dashboards and transparency reports.

        Args:
            participant_ids: List of participant IDs

        Returns:
            Dict mapping participant_id to their VotingPowerCalculation
        """
        distribution = {}

        tasks = [
            self.calculate_voting_power(pid)
            for pid in participant_ids
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for pid, result in zip(participant_ids, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to get voting power for {pid}: {result}")
            elif result is not None:
                distribution[pid] = result

        return distribution

    async def _get_guardian_endorsements(
        self,
        participant_id: str,
    ) -> List[GuardianRelationship]:
        """Get guardian endorsements for a participant."""
        # Check cache
        if participant_id in self._guardian_cache:
            return self._guardian_cache[participant_id]

        if not self.dht:
            return []

        try:
            # Query DHT for guardian relationships
            result = await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="get_guardian_relationships",
                payload={
                    "subject_participant_id": participant_id,
                    "relationship_type": "ENDORSEMENT",
                },
            )

            guardians = [GuardianRelationship.from_dict(g) for g in (result or [])]
            self._guardian_cache[participant_id] = guardians
            return guardians

        except Exception as e:
            logger.warning(f"Failed to fetch guardian endorsements for {participant_id}: {e}")
            return []

    async def _get_last_activity(self, participant_id: str) -> int:
        """Get timestamp of last governance activity for participant."""
        # Check governance stats (try async first, fall back to sync)
        try:
            # Use async version if available
            if hasattr(self.gov_extensions, 'get_governance_stats'):
                stats = await self.gov_extensions.get_governance_stats(participant_id)
            else:
                stats = self.gov_extensions.get_governance_stats_sync(participant_id)
        except TypeError:
            # Fall back to sync version if async call fails
            stats = self.gov_extensions.get_governance_stats_sync(participant_id)

        # Check last_activity timestamp from DHT query
        last_activity = stats.get("last_activity", 0)
        if last_activity > 0:
            return int(last_activity)

        # Return current time if they have recent invocations
        invocations = stats.get("capabilities_invoked", {})
        if invocations:
            return int(time.time())

        # Otherwise return 0 for inactive
        return 0

    def clear_cache(self, participant_id: Optional[str] = None):
        """Clear voting power cache."""
        if participant_id:
            self._voting_power_cache.pop(participant_id, None)
            self._guardian_cache.pop(participant_id, None)
        else:
            self._voting_power_cache.clear()
            self._guardian_cache.clear()


class GuardianVerifier:
    """
    Verifies guardian relationships and detects circular guardianships.

    Ensures that:
    1. Guardian relationships are valid and not expired
    2. Guardians have sufficient assurance levels
    3. No circular guardian chains exist (A guards B guards C guards A)
    """

    # Maximum depth for guardian chain verification
    MAX_GUARDIAN_DEPTH = 10

    def __init__(self, identity_governance_extensions, dht_client=None):
        """
        Args:
            identity_governance_extensions: Identity governance extensions instance
            dht_client: Optional DHT client for guardian queries
        """
        self.gov_extensions = identity_governance_extensions
        self.dht = dht_client

        # Cache for guardian relationships
        self._guardian_graph: Dict[str, Set[str]] = {}

    async def verify_guardian(
        self,
        guardian_participant_id: str,
        subject_participant_id: str,
        required_assurance: str = "E2",
    ) -> Tuple[bool, str]:
        """
        Verify that a guardian relationship is valid.

        Args:
            guardian_participant_id: The guardian
            subject_participant_id: The subject being guarded
            required_assurance: Minimum assurance level for guardian

        Returns:
            Tuple of (valid, reason)
        """
        # Step 1: Verify guardian exists and has sufficient assurance
        verification = await self.gov_extensions.verify_identity_for_governance(
            participant_id=guardian_participant_id,
            required_assurance=required_assurance,
        )

        if not verification.get("verified", False):
            reasons = ", ".join(verification.get("reasons", ["Unknown"]))
            return False, f"Guardian not verified: {reasons}"

        # Step 2: Check for circular guardianship
        has_cycle, cycle_path = await self.detect_circular_guardianship(
            guardian_participant_id,
            subject_participant_id,
        )

        if has_cycle:
            return False, f"Circular guardianship detected: {' -> '.join(cycle_path)}"

        # Step 3: Verify guardian is not also guarded by subject (direct cycle)
        subject_guardians = await self._get_guardians_for_subject(guardian_participant_id)
        if subject_participant_id in subject_guardians:
            return False, "Direct circular guardianship: subject is also guardian's guardian"

        return True, "Guardian verified"

    async def detect_circular_guardianship(
        self,
        guardian_id: str,
        subject_id: str,
    ) -> Tuple[bool, List[str]]:
        """
        Detect if adding a guardian relationship would create a cycle.

        Uses BFS to detect cycles in the guardian graph.

        Args:
            guardian_id: Proposed guardian
            subject_id: Proposed subject

        Returns:
            Tuple of (has_cycle, cycle_path)
        """
        # Build the guardian graph if not cached
        await self._build_guardian_graph()

        # Check if subject can reach guardian through existing relationships
        # If so, adding guardian -> subject would create a cycle

        visited: Set[str] = set()
        queue: deque = deque()
        parent: Dict[str, Optional[str]] = {subject_id: None}

        queue.append(subject_id)
        visited.add(subject_id)

        while queue:
            current = queue.popleft()

            # Get guardians of current node
            guardians = self._guardian_graph.get(current, set())

            for g in guardians:
                if g == guardian_id:
                    # Found a path from subject to guardian
                    # This means adding guardian -> subject creates a cycle
                    cycle_path = self._reconstruct_path(parent, current, guardian_id)
                    return True, cycle_path

                if g not in visited:
                    visited.add(g)
                    parent[g] = current
                    queue.append(g)

                    # Limit search depth
                    if len(visited) > self.MAX_GUARDIAN_DEPTH * 10:
                        break

        return False, []

    def _reconstruct_path(
        self,
        parent: Dict[str, Optional[str]],
        end: str,
        start: str,
    ) -> List[str]:
        """Reconstruct the cycle path."""
        path = [start, end]
        current = end
        while parent.get(current) is not None:
            current = parent[current]
            path.append(current)
        path.reverse()
        return path

    async def _build_guardian_graph(self):
        """Build/refresh the guardian relationship graph."""
        if self._guardian_graph:
            return  # Already built

        if not self.dht:
            return

        try:
            # Query all guardian relationships
            result = await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="list_all_guardian_relationships",
                payload={},
            )

            for rel in (result or []):
                subject = rel.get("subject_participant_id")
                guardian = rel.get("guardian_participant_id")
                if subject and guardian:
                    if subject not in self._guardian_graph:
                        self._guardian_graph[subject] = set()
                    self._guardian_graph[subject].add(guardian)

        except Exception as e:
            logger.warning(f"Failed to build guardian graph: {e}")

    async def _get_guardians_for_subject(self, subject_id: str) -> Set[str]:
        """Get all guardians for a subject."""
        await self._build_guardian_graph()
        return self._guardian_graph.get(subject_id, set())

    async def get_guardian_weight(
        self,
        guardian_participant_id: str,
        subject_participant_id: str,
    ) -> float:
        """
        Get the weight of a guardian relationship.

        Args:
            guardian_participant_id: The guardian
            subject_participant_id: The subject

        Returns:
            Weight (0.0-1.0) or 0.0 if relationship doesn't exist
        """
        if not self.dht:
            return 0.0

        try:
            result = await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="get_guardian_relationship",
                payload={
                    "subject_participant_id": subject_participant_id,
                    "guardian_participant_id": guardian_participant_id,
                },
            )

            if result:
                rel = GuardianRelationship.from_dict(result)
                if rel.is_valid():
                    return rel.weight
            return 0.0

        except Exception as e:
            logger.warning(f"Failed to get guardian weight: {e}")
            return 0.0

    def clear_cache(self):
        """Clear the guardian graph cache."""
        self._guardian_graph.clear()


class VotingEngine:
    """
    Handles vote collection and tallying

    Responsibilities:
    - Cast votes with quadratic voting
    - Tally votes by proposal with proper quorum calculation
    - Determine proposal outcome with supermajority thresholds
    - Update proposal status based on results
    - Sign votes with Ed25519 for cryptographic authenticity
    - Verify signatures before accepting votes
    - Enforce time-locked voting periods
    """

    def __init__(
        self,
        dht_client,
        identity_governance_extensions,
        proposal_manager,
        governance_signer: Optional[GovernanceSigner] = None,
        voting_power_calculator: Optional[VotingPowerCalculator] = None,
        guardian_verifier: Optional[GuardianVerifier] = None,
    ):
        """
        Args:
            dht_client: Holochain DHT client
            identity_governance_extensions: Identity governance extensions
            proposal_manager: Proposal manager instance
            governance_signer: Optional Ed25519 signer for vote authentication
            voting_power_calculator: Optional calculator for voting power
            guardian_verifier: Optional verifier for guardian relationships
        """
        self.dht = dht_client
        self.gov_extensions = identity_governance_extensions
        self.proposal_mgr = proposal_manager
        self.signer = governance_signer

        # Initialize voting power calculator and guardian verifier
        self.voting_power_calc = voting_power_calculator or VotingPowerCalculator(
            identity_governance_extensions=identity_governance_extensions,
            dht_client=dht_client,
        )
        self.guardian_verifier = guardian_verifier or GuardianVerifier(
            identity_governance_extensions=identity_governance_extensions,
            dht_client=dht_client,
        )

        # Cache for verified voter public keys (participant_id -> public_key_bytes)
        self._voter_public_keys: Dict[str, bytes] = {}

        # Track all eligible voters for quorum calculation
        self._eligible_voters: Set[str] = set()

    async def cast_vote(
        self,
        voter_participant_id: str,
        proposal_id: str,
        choice: VoteChoice,
        credits_spent: int,
    ) -> Tuple[bool, str]:
        """
        Cast a vote on a proposal

        Args:
            voter_participant_id: Participant casting vote
            proposal_id: Proposal to vote on
            choice: Vote choice (FOR, AGAINST, ABSTAIN)
            credits_spent: Credits to spend (quadratic voting)

        Returns:
            Tuple of (success, message)
        """
        # Step 1: Validate proposal exists and is in voting period
        proposal = await self.proposal_mgr.get_proposal(proposal_id)
        if not proposal:
            return False, "Proposal not found"

        now = int(time.time())
        if now < proposal.voting_start:
            return False, f"Voting hasn't started yet (starts at {proposal.voting_start})"

        if now > proposal.voting_end:
            return False, f"Voting has ended (ended at {proposal.voting_end})"

        if proposal.status != ProposalStatus.VOTING:
            return False, f"Proposal not in voting status: {proposal.status.value}"

        # Step 2: Verify voter has governance rights
        verification = await self.gov_extensions.verify_identity_for_governance(
            participant_id=voter_participant_id,
            required_assurance="E0",  # Minimum assurance for voting
        )

        if not verification["verified"]:
            reasons = ", ".join(verification.get("reasons", ["Unknown"]))
            return False, f"Voter not verified: {reasons}"

        # Step 3: Calculate vote weight
        vote_weight = await self.gov_extensions.calculate_vote_weight(
            participant_id=voter_participant_id
        )

        # Step 4: Check vote budget
        budget = self.gov_extensions.calculate_vote_budget(vote_weight)
        if credits_spent > budget:
            return False, f"Insufficient credits: {credits_spent} > {budget}"

        # Step 5: Calculate effective votes (quadratic)
        effective_votes = self.gov_extensions.calculate_effective_votes(credits_spent)
        effective_votes *= vote_weight  # Scale by vote weight

        # Step 6: Get voter's DID
        identity = await self.gov_extensions.coordinator.get_identity(
            participant_id=voter_participant_id
        )
        voter_did = identity.get("did", "")

        # Step 7: Generate vote ID
        vote_id = self._generate_vote_id(voter_participant_id, proposal_id)

        # Step 8: Create signable vote data (without signature field)
        signable_data = {
            "vote_id": vote_id,
            "proposal_id": proposal_id,
            "voter_did": voter_did,
            "voter_participant_id": voter_participant_id,
            "choice": choice.value,
            "credits_spent": credits_spent,
            "vote_weight": vote_weight,
            "effective_votes": effective_votes,
            "timestamp": now,
        }

        # Step 9: Sign vote with Ed25519
        signature = ""
        if self.signer:
            try:
                signature = self.signer.sign_governance_data(signable_data)
                logger.debug(f"Vote signed with Ed25519 for {voter_participant_id}")
            except Exception as e:
                logger.error(f"Failed to sign vote: {e}")
                return False, f"Vote signing failed: {str(e)}"
        else:
            logger.warning("No governance signer configured - vote will be unsigned")

        # Step 10: Create vote record with signature
        vote = VoteData(
            vote_id=vote_id,
            proposal_id=proposal_id,
            voter_did=voter_did,
            voter_participant_id=voter_participant_id,
            choice=choice,
            credits_spent=credits_spent,
            vote_weight=vote_weight,
            effective_votes=effective_votes,
            timestamp=now,
            signature=signature,
        )

        # Step 11: Store vote on DHT
        try:
            await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="store_vote",
                payload=vote.to_dict(),
            )
        except Exception as e:
            logger.error(f"Failed to store vote on DHT: {e}")
            return False, f"DHT storage failed: {str(e)}"

        logger.info(
            f"Vote cast: {voter_participant_id} -> {proposal_id} "
            f"({choice.value}, {effective_votes:.2f} effective votes)"
        )

        return True, f"Vote cast successfully ({effective_votes:.2f} effective votes)"

    async def tally_votes(
        self,
        proposal_id: str,
        verify_signatures: bool = True,
    ) -> Dict[str, Any]:
        """
        Tally all votes for a proposal with full quorum and signature verification.

        Args:
            proposal_id: Proposal to tally
            verify_signatures: Whether to verify vote signatures

        Returns:
            Dictionary with vote totals and outcome
        """
        # Step 1: Retrieve all votes
        try:
            votes_data = await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="get_votes",
                payload={"proposal_id": proposal_id},
            )
            votes = [VoteData.from_dict(v) for v in votes_data]
        except Exception as e:
            logger.error(f"Failed to retrieve votes for {proposal_id}: {e}")
            return {
                "success": False,
                "error": str(e),
            }

        # Step 2: Get proposal details
        proposal = await self.proposal_mgr.get_proposal(proposal_id)
        if not proposal:
            return {"success": False, "error": "Proposal not found"}

        # Step 3: Verify signatures and sum vote totals
        total_for = 0.0
        total_against = 0.0
        total_abstain = 0.0
        verified_signatures = 0
        invalid_signatures = 0
        guardian_verified_votes = 0
        unique_voter_ids: Set[str] = set()

        for vote in votes:
            # Track unique voters
            unique_voter_ids.add(vote.voter_participant_id)

            # Optionally verify signature
            if verify_signatures and vote.signature:
                is_valid, _ = await self.verify_vote_signature(vote)
                if is_valid:
                    verified_signatures += 1
                else:
                    invalid_signatures += 1
                    logger.warning(
                        f"Invalid signature for vote {vote.vote_id} "
                        f"from {vote.voter_participant_id}"
                    )
                    # Skip votes with invalid signatures
                    continue

            # Sum effective votes by choice
            if vote.choice == VoteChoice.FOR:
                total_for += vote.effective_votes
            elif vote.choice == VoteChoice.AGAINST:
                total_against += vote.effective_votes
            elif vote.choice == VoteChoice.ABSTAIN:
                total_abstain += vote.effective_votes

        total_voting_power_cast = total_for + total_against + total_abstain

        # Step 4: Calculate total eligible voting power
        # Get all registered eligible voters from the system
        eligible_participant_ids = list(self._eligible_voters)
        if not eligible_participant_ids:
            # Fall back to querying DHT for all participants
            try:
                all_participants = await self.dht.call_zome(
                    zome_name="governance_record",
                    fn_name="list_eligible_voters",
                    payload={},
                )
                eligible_participant_ids = [p.get("participant_id") for p in (all_participants or [])]
            except Exception as e:
                logger.warning(f"Could not fetch eligible voters: {e}")
                # Fall back to using voters who cast votes as minimum
                eligible_participant_ids = list(unique_voter_ids)

        # Calculate total eligible voting power
        total_eligible_power = await self.voting_power_calc.calculate_total_eligible_voting_power(
            eligible_participant_ids
        )

        # If total_eligible_power is 0, use cast votes as denominator
        if total_eligible_power <= 0:
            total_eligible_power = total_voting_power_cast

        # Step 5: Check quorum
        if total_eligible_power > 0:
            quorum_achieved = total_voting_power_cast / total_eligible_power
        else:
            quorum_achieved = 0.0

        quorum_met = quorum_achieved >= proposal.quorum

        # Step 6: Check approval threshold (supermajority)
        # Approval ratio is FOR / (FOR + AGAINST), abstains don't count
        votes_counted = total_for + total_against
        if votes_counted > 0:
            approval_ratio = total_for / votes_counted
        else:
            approval_ratio = 0.0

        # Proposal approved if quorum met AND approval threshold met
        approved = quorum_met and approval_ratio >= proposal.approval_threshold

        # Create comprehensive tally
        tally = VoteTally(
            proposal_id=proposal_id,
            total_votes_for=total_for,
            total_votes_against=total_against,
            total_votes_abstain=total_abstain,
            total_voting_power_cast=total_voting_power_cast,
            total_eligible_voting_power=total_eligible_power,
            unique_voters=len(unique_voter_ids),
            quorum_required=proposal.quorum,
            quorum_achieved=quorum_achieved,
            quorum_met=quorum_met,
            approval_threshold=proposal.approval_threshold,
            approval_ratio=approval_ratio,
            approved=approved,
            voting_start=proposal.voting_start,
            voting_end=proposal.voting_end,
            verified_signatures=verified_signatures,
            invalid_signatures=invalid_signatures,
            guardian_verified_votes=guardian_verified_votes,
        )

        result = {
            "success": True,
            "proposal_id": proposal_id,
            "total_votes": len(votes),
            "total_for": total_for,
            "total_against": total_against,
            "total_abstain": total_abstain,
            "total_voting_power": total_voting_power_cast,
            "total_eligible_voting_power": total_eligible_power,
            "unique_voters": len(unique_voter_ids),
            "approval_ratio": approval_ratio,
            "quorum_required": proposal.quorum,
            "quorum_achieved": quorum_achieved,
            "quorum_met": quorum_met,
            "approved": approved,
            "required_threshold": proposal.approval_threshold,
            "verified_signatures": verified_signatures,
            "invalid_signatures": invalid_signatures,
            "tally": tally.to_dict(),
        }

        logger.info(
            f"Votes tallied for {proposal_id}: "
            f"{total_for:.2f} FOR, {total_against:.2f} AGAINST, "
            f"approval {approval_ratio:.2%} (threshold {proposal.approval_threshold:.2%}), "
            f"quorum {quorum_achieved:.2%}/{proposal.quorum:.2%}"
        )

        return result

    def register_eligible_voter(self, participant_id: str) -> None:
        """
        Register a participant as an eligible voter.

        Args:
            participant_id: The participant to register
        """
        self._eligible_voters.add(participant_id)

    def unregister_eligible_voter(self, participant_id: str) -> None:
        """
        Remove a participant from the eligible voter list.

        Args:
            participant_id: The participant to remove
        """
        self._eligible_voters.discard(participant_id)

    def get_eligible_voters(self) -> Set[str]:
        """Get all registered eligible voters."""
        return self._eligible_voters.copy()

    async def finalize_proposal(self, proposal_id: str) -> Tuple[bool, str]:
        """
        Finalize proposal after voting period ends

        Args:
            proposal_id: Proposal to finalize

        Returns:
            Tuple of (success, message)
        """
        # Step 1: Verify voting period has ended
        proposal = await self.proposal_mgr.get_proposal(proposal_id)
        if not proposal:
            return False, "Proposal not found"

        now = int(time.time())
        if now < proposal.voting_end:
            return False, f"Voting period not ended yet (ends at {proposal.voting_end})"

        # Step 2: Tally votes
        tally = await self.tally_votes(proposal_id)
        if not tally["success"]:
            return False, f"Vote tallying failed: {tally.get('error', 'Unknown')}"

        # Step 3: Determine outcome
        new_status = ProposalStatus.APPROVED if tally["approved"] else ProposalStatus.REJECTED

        # Step 4: Update proposal status
        vote_totals = {
            "for": tally["total_for"],
            "against": tally["total_against"],
            "abstain": tally["total_abstain"],
            "total": tally["total_voting_power"],
        }

        success = await self.proposal_mgr.update_proposal_status(
            proposal_id=proposal_id,
            new_status=new_status,
            vote_totals=vote_totals,
        )

        if not success:
            return False, "Failed to update proposal status"

        logger.info(
            f"Proposal {proposal_id} finalized: {new_status.value} "
            f"({tally['approval_ratio']:.2%} approval)"
        )

        return True, f"Proposal {new_status.value.lower()}"

    async def verify_vote_signature(
        self,
        vote: VoteData,
        public_key_bytes: Optional[bytes] = None,
    ) -> Tuple[bool, str]:
        """
        Verify the Ed25519 signature on a vote.

        Args:
            vote: The vote data to verify
            public_key_bytes: The voter's public key bytes. If not provided,
                            attempts to retrieve from identity system.

        Returns:
            Tuple of (valid, message)
        """
        if not vote.signature:
            return False, "Vote has no signature"

        if not self.signer:
            return False, "No governance signer configured for verification"

        # Get public key for voter if not provided
        if public_key_bytes is None:
            # Try to get from cache
            if vote.voter_participant_id in self._voter_public_keys:
                public_key_bytes = self._voter_public_keys[vote.voter_participant_id]
            else:
                # Try to retrieve from identity system
                try:
                    identity = await self.gov_extensions.coordinator.get_identity(
                        participant_id=vote.voter_participant_id
                    )
                    if identity and "public_key" in identity:
                        public_key_hex = identity.get("public_key", "")
                        if public_key_hex:
                            public_key_bytes = bytes.fromhex(public_key_hex)
                            self._voter_public_keys[vote.voter_participant_id] = public_key_bytes
                except Exception as e:
                    logger.warning(f"Failed to retrieve voter public key: {e}")

        if public_key_bytes is None:
            return False, "Could not retrieve voter's public key"

        # Reconstruct signable data
        signable_data = {
            "vote_id": vote.vote_id,
            "proposal_id": vote.proposal_id,
            "voter_did": vote.voter_did,
            "voter_participant_id": vote.voter_participant_id,
            "choice": vote.choice.value,
            "credits_spent": vote.credits_spent,
            "vote_weight": vote.vote_weight,
            "effective_votes": vote.effective_votes,
            "timestamp": vote.timestamp,
        }

        # Verify signature
        try:
            is_valid = self.signer.verify_governance_signature(
                data=signable_data,
                signature_hex=vote.signature,
                public_key_bytes=public_key_bytes,
            )
            if is_valid:
                return True, "Vote signature verified"
            else:
                return False, "Vote signature verification failed"
        except Exception as e:
            logger.error(f"Vote signature verification error: {e}")
            return False, f"Verification error: {str(e)}"

    def register_voter_public_key(self, participant_id: str, public_key_bytes: bytes) -> None:
        """
        Register a voter's public key for signature verification.

        Args:
            participant_id: The voter's participant ID
            public_key_bytes: The voter's Ed25519 public key (32 bytes)
        """
        self._voter_public_keys[participant_id] = public_key_bytes
        logger.debug(f"Registered public key for voter {participant_id}")

    def _generate_vote_id(self, voter_id: str, proposal_id: str) -> str:
        """Generate unique vote ID"""
        data = f"{voter_id}:{proposal_id}:{int(time.time())}"
        return f"vote_{hashlib.sha256(data.encode()).hexdigest()[:16]}"


class CapabilityEnforcer:
    """
    Enforces capability-based access control

    Responsibilities:
    - Verify participant capabilities
    - Check identity verification requirements
    - Enforce rate limiting
    - Coordinate with guardian authorization
    """

    def __init__(self, identity_governance_extensions):
        """
        Args:
            identity_governance_extensions: Identity governance extensions
        """
        self.gov_extensions = identity_governance_extensions

    async def check_capability(
        self,
        participant_id: str,
        capability_id: str,
    ) -> Tuple[bool, str]:
        """
        Check if participant has capability

        Args:
            participant_id: Participant to check
            capability_id: Capability to verify

        Returns:
            Tuple of (authorized, reason)
        """
        return await self.gov_extensions.authorize_capability(
            participant_id=participant_id,
            capability_id=capability_id,
        )

    async def enforce_capability(
        self,
        participant_id: str,
        capability_id: str,
        action_name: str,
    ) -> None:
        """
        Enforce capability requirement (raises exception if not authorized)

        Args:
            participant_id: Participant performing action
            capability_id: Required capability
            action_name: Name of action for error message

        Raises:
            PermissionError: If participant not authorized
        """
        authorized, reason = await self.check_capability(
            participant_id=participant_id,
            capability_id=capability_id,
        )

        if not authorized:
            raise PermissionError(
                f"Participant {participant_id} not authorized for {action_name}: {reason}"
            )

    def get_capability_requirements(self, capability_id: str) -> Optional[Dict[str, Any]]:
        """
        Get capability requirements

        Args:
            capability_id: Capability to query

        Returns:
            Dictionary of requirements or None if not found
        """
        from zerotrustml.identity import CAPABILITY_REGISTRY

        capability = CAPABILITY_REGISTRY.get(capability_id)
        if not capability:
            return None

        return {
            "capability_id": capability.capability_id,
            "name": capability.name,
            "description": capability.description,
            "required_assurance": capability.required_assurance,
            "required_reputation": capability.required_reputation,
            "required_sybil_resistance": capability.required_sybil_resistance,
            "required_guardian_approval": capability.required_guardian_approval,
            "guardian_threshold": capability.guardian_threshold,
            "rate_limit": capability.rate_limit,
            "time_period_seconds": capability.time_period_seconds,
        }


class GuardianAuthorizationManager:
    """
    Manages guardian authorization for emergency actions

    Responsibilities:
    - Create authorization requests
    - Collect guardian approvals with verification
    - Check approval threshold
    - Update authorization status
    - Sign approvals with Ed25519 for cryptographic authenticity
    - Verify guardian relationships before counting endorsements
    - Check for circular guardianships
    """

    def __init__(
        self,
        dht_client,
        identity_governance_extensions,
        governance_signer: Optional[GovernanceSigner] = None,
        guardian_verifier: Optional[GuardianVerifier] = None,
        voting_power_calculator: Optional[VotingPowerCalculator] = None,
    ):
        """
        Args:
            dht_client: Holochain DHT client
            identity_governance_extensions: Identity governance extensions
            governance_signer: Optional Ed25519 signer for approval authentication
            guardian_verifier: Optional verifier for guardian relationships
            voting_power_calculator: Optional calculator for reputation-weighted voting power
        """
        self.dht = dht_client
        self.gov_extensions = identity_governance_extensions
        self.signer = governance_signer

        # Initialize guardian verifier
        self.guardian_verifier = guardian_verifier or GuardianVerifier(
            identity_governance_extensions=identity_governance_extensions,
            dht_client=dht_client,
        )

        # Initialize voting power calculator for reputation-weighted approvals
        self.voting_power_calc = voting_power_calculator or VotingPowerCalculator(
            identity_governance_extensions=identity_governance_extensions,
            dht_client=dht_client,
        )

        # Cache for verified guardian public keys (participant_id -> public_key_bytes)
        self._guardian_public_keys: Dict[str, bytes] = {}

    async def request_authorization(
        self,
        participant_id: str,
        capability_id: str,
        action_params: Dict[str, Any],
        timeout_seconds: int = 3600,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Request guardian authorization for emergency action

        Args:
            participant_id: Participant requesting authorization
            capability_id: Capability requiring authorization
            action_params: Parameters for the action
            timeout_seconds: How long request is valid

        Returns:
            Tuple of (success, message, request_id)
        """
        # Step 1: Create authorization request
        try:
            request_id = await self.gov_extensions.request_guardian_authorization(
                participant_id=participant_id,
                capability_id=capability_id,
                action_params=action_params,
                timeout_seconds=timeout_seconds,
            )
        except Exception as e:
            return False, str(e), None

        # Step 2: Store on DHT
        from zerotrustml.identity import CAPABILITY_REGISTRY

        capability = CAPABILITY_REGISTRY.get(capability_id)
        if not capability:
            return False, f"Unknown capability: {capability_id}", None

        request_data = AuthorizationRequestData(
            request_id=request_id,
            subject_participant_id=participant_id,
            action=capability_id,
            action_params=action_params,
            required_threshold=capability.guardian_threshold,
            expires_at=int(time.time()) + timeout_seconds,
            status=AuthorizationStatus.PENDING,
        )

        try:
            await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="store_authorization_request",
                payload=request_data.to_dict(),
            )
        except Exception as e:
            logger.error(f"Failed to store authorization request: {e}")
            return False, f"DHT storage failed: {str(e)}", None

        logger.info(
            f"Authorization request created: {request_id} for {participant_id}"
        )

        return True, "Authorization request created", request_id

    async def submit_approval(
        self,
        guardian_participant_id: str,
        request_id: str,
        approved: bool,
        reasoning: str = "",
    ) -> Tuple[bool, str]:
        """
        Submit guardian approval for authorization request

        Args:
            guardian_participant_id: Guardian submitting approval
            request_id: Authorization request ID
            approved: True for approval, False for rejection
            reasoning: Optional reasoning for decision

        Returns:
            Tuple of (success, message)
        """
        # Step 1: Get the authorization request to find subject
        try:
            request_data = await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="get_authorization_request",
                payload={"request_id": request_id},
            )
            if not request_data:
                return False, f"Authorization request not found: {request_id}"

            subject_participant_id = request_data.get("subject_participant_id")
        except Exception as e:
            return False, f"Failed to retrieve authorization request: {e}"

        # Step 2: Verify guardian status using GuardianVerifier
        is_valid, verification_reason = await self.guardian_verifier.verify_guardian(
            guardian_participant_id=guardian_participant_id,
            subject_participant_id=subject_participant_id,
            required_assurance="E2",  # Guardians need E2 assurance minimum
        )

        if not is_valid:
            return False, f"Guardian verification failed: {verification_reason}"

        # Step 3: Get guardian's DID
        identity = await self.gov_extensions.coordinator.get_identity(
            participant_id=guardian_participant_id
        )
        if not identity:
            return False, "Guardian identity not found"

        guardian_did = identity.get("did", "")

        # Step 3: Generate approval ID
        approval_id = self._generate_approval_id(guardian_participant_id, request_id)
        now = int(time.time())

        # Step 4: Create signable approval data (without signature field)
        signable_data = {
            "approval_id": approval_id,
            "request_id": request_id,
            "guardian_did": guardian_did,
            "guardian_participant_id": guardian_participant_id,
            "approved": approved,
            "reasoning": reasoning,
            "timestamp": now,
        }

        # Step 5: Sign approval with Ed25519
        signature = ""
        if self.signer:
            try:
                signature = self.signer.sign_governance_data(signable_data)
                logger.debug(f"Approval signed with Ed25519 for {guardian_participant_id}")
            except Exception as e:
                logger.error(f"Failed to sign approval: {e}")
                return False, f"Approval signing failed: {str(e)}"
        else:
            logger.warning("No governance signer configured - approval will be unsigned")

        # Step 6: Create approval record with signature
        approval = GuardianApprovalData(
            approval_id=approval_id,
            request_id=request_id,
            guardian_did=guardian_did,
            guardian_participant_id=guardian_participant_id,
            approved=approved,
            reasoning=reasoning,
            timestamp=now,
            signature=signature,
        )

        # Step 7: Store on DHT
        try:
            await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="store_guardian_approval",
                payload=approval.to_dict(),
            )
        except Exception as e:
            logger.error(f"Failed to store guardian approval: {e}")
            return False, f"DHT storage failed: {str(e)}"

        # Step 8: Check if threshold reached
        await self._check_authorization_threshold(request_id)

        logger.info(
            f"Guardian approval submitted: {guardian_participant_id} "
            f"{'approved' if approved else 'rejected'} request {request_id}"
        )

        return True, "Approval submitted successfully"

    async def verify_approval_signature(
        self,
        approval: GuardianApprovalData,
        public_key_bytes: Optional[bytes] = None,
    ) -> Tuple[bool, str]:
        """
        Verify the Ed25519 signature on a guardian approval.

        Args:
            approval: The approval data to verify
            public_key_bytes: The guardian's public key bytes. If not provided,
                            attempts to retrieve from identity system.

        Returns:
            Tuple of (valid, message)
        """
        if not approval.signature:
            return False, "Approval has no signature"

        if not self.signer:
            return False, "No governance signer configured for verification"

        # Get public key for guardian if not provided
        if public_key_bytes is None:
            # Try to get from cache
            if approval.guardian_participant_id in self._guardian_public_keys:
                public_key_bytes = self._guardian_public_keys[approval.guardian_participant_id]
            else:
                # Try to retrieve from identity system
                try:
                    identity = await self.gov_extensions.coordinator.get_identity(
                        participant_id=approval.guardian_participant_id
                    )
                    if identity and "public_key" in identity:
                        public_key_hex = identity.get("public_key", "")
                        if public_key_hex:
                            public_key_bytes = bytes.fromhex(public_key_hex)
                            self._guardian_public_keys[approval.guardian_participant_id] = public_key_bytes
                except Exception as e:
                    logger.warning(f"Failed to retrieve guardian public key: {e}")

        if public_key_bytes is None:
            return False, "Could not retrieve guardian's public key"

        # Reconstruct signable data
        signable_data = {
            "approval_id": approval.approval_id,
            "request_id": approval.request_id,
            "guardian_did": approval.guardian_did,
            "guardian_participant_id": approval.guardian_participant_id,
            "approved": approval.approved,
            "reasoning": approval.reasoning,
            "timestamp": approval.timestamp,
        }

        # Verify signature
        try:
            is_valid = self.signer.verify_governance_signature(
                data=signable_data,
                signature_hex=approval.signature,
                public_key_bytes=public_key_bytes,
            )
            if is_valid:
                return True, "Approval signature verified"
            else:
                return False, "Approval signature verification failed"
        except Exception as e:
            logger.error(f"Approval signature verification error: {e}")
            return False, f"Verification error: {str(e)}"

    def register_guardian_public_key(self, participant_id: str, public_key_bytes: bytes) -> None:
        """
        Register a guardian's public key for signature verification.

        Args:
            participant_id: The guardian's participant ID
            public_key_bytes: The guardian's Ed25519 public key (32 bytes)
        """
        self._guardian_public_keys[participant_id] = public_key_bytes
        logger.debug(f"Registered public key for guardian {participant_id}")

    async def check_authorization_status(
        self,
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Check current status of authorization request

        Args:
            request_id: Authorization request ID

        Returns:
            Dictionary with status and approval details
        """
        # Step 1: Retrieve request
        try:
            request_data = await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="get_authorization_request",
                payload={"request_id": request_id},
            )

            if not request_data:
                return {"success": False, "error": "Request not found"}

            request = AuthorizationRequestData.from_dict(request_data)

        except Exception as e:
            logger.error(f"Failed to retrieve authorization request: {e}")
            return {"success": False, "error": str(e)}

        # Step 2: Retrieve approvals
        try:
            approvals_data = await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="get_guardian_approvals",
                payload={"request_id": request_id},
            )

            approvals = [GuardianApprovalData.from_dict(a) for a in approvals_data]

        except Exception as e:
            logger.error(f"Failed to retrieve guardian approvals: {e}")
            return {"success": False, "error": str(e)}

        # Step 3: Calculate approval weight weighted by guardian reputation
        # Each guardian's vote is weighted by their voting power (based on reputation,
        # assurance level, sybil resistance, and time-based activity)
        weighted_approvals = 0.0
        weighted_rejections = 0.0
        total_guardian_weight = 0.0
        guardian_weights: Dict[str, float] = {}

        for approval in approvals:
            try:
                # Get guardian's voting power (includes reputation weighting)
                voting_power = await self.voting_power_calc.calculate_voting_power(
                    participant_id=approval.guardian_participant_id,
                    include_guardian_endorsements=False,  # Avoid circular lookups
                )
                weight = voting_power.total_voting_power
                guardian_weights[approval.guardian_participant_id] = weight
                total_guardian_weight += weight

                if approval.approved:
                    weighted_approvals += weight
                else:
                    weighted_rejections += weight
            except Exception as e:
                logger.warning(
                    f"Failed to calculate voting power for guardian "
                    f"{approval.guardian_participant_id}: {e}. Using default weight of 1.0"
                )
                # Fall back to equal weight if voting power calculation fails
                weight = 1.0
                guardian_weights[approval.guardian_participant_id] = weight
                total_guardian_weight += weight
                if approval.approved:
                    weighted_approvals += weight
                else:
                    weighted_rejections += weight

        # Calculate weighted approval ratio
        total_approvals = sum(1 for a in approvals if a.approved)
        total_rejections = sum(1 for a in approvals if not a.approved)

        approval_weight = (
            weighted_approvals / total_guardian_weight if total_guardian_weight > 0 else 0.0
        )

        # Step 4: Check threshold
        threshold_met = approval_weight >= request.required_threshold

        # Step 5: Check expiration
        now = int(time.time())
        expired = now > request.expires_at

        total_guardians = total_approvals + total_rejections

        result = {
            "success": True,
            "request_id": request_id,
            "status": request.status.value,
            "total_approvals": total_approvals,
            "total_rejections": total_rejections,
            "total_guardians": total_guardians,
            "weighted_approvals": weighted_approvals,
            "weighted_rejections": weighted_rejections,
            "total_guardian_weight": total_guardian_weight,
            "guardian_weights": guardian_weights,
            "approval_weight": approval_weight,
            "required_threshold": request.required_threshold,
            "threshold_met": threshold_met,
            "expired": expired,
            "expires_at": request.expires_at,
        }

        return result

    async def _check_authorization_threshold(self, request_id: str) -> None:
        """
        Check if authorization threshold is met and update status

        Args:
            request_id: Authorization request to check
        """
        status = await self.check_authorization_status(request_id)

        if not status["success"]:
            return

        # Update request status if threshold met or expired
        new_status = None

        if status["expired"]:
            new_status = AuthorizationStatus.EXPIRED
        elif status["threshold_met"]:
            new_status = AuthorizationStatus.APPROVED
        elif status["weighted_rejections"] > 0:
            # Check if weighted rejection threshold exceeded
            # Use reputation-weighted rejection ratio for fairness
            total_weight = status["total_guardian_weight"]
            if total_weight > 0:
                weighted_rejection_ratio = status["weighted_rejections"] / total_weight
                if weighted_rejection_ratio >= (1.0 - status["required_threshold"]):
                    new_status = AuthorizationStatus.REJECTED

        if new_status:
            try:
                await self.dht.call_zome(
                    zome_name="governance_record",
                    fn_name="update_authorization_status",
                    payload={
                        "request_id": request_id,
                        "status": new_status.value,
                        "updated_at": int(time.time()),
                    },
                )

                logger.info(f"Authorization request {request_id} status updated to {new_status.value}")

            except Exception as e:
                logger.error(f"Failed to update authorization status: {e}")

    def _generate_approval_id(self, guardian_id: str, request_id: str) -> str:
        """Generate unique approval ID"""
        data = f"{guardian_id}:{request_id}:{int(time.time())}"
        return f"approval_{hashlib.sha256(data.encode()).hexdigest()[:16]}"

    async def calculate_weighted_guardian_approvals(
        self,
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Calculate weighted guardian approvals using comprehensive reputation metrics.

        Weights each guardian's approval by their reputation score, incorporating:
        1. MATL assurance level (verified identity strength)
        2. Historical governance participation
        3. Guardian relationship quality (endorsement history)
        4. Time-weighted activity decay
        5. Sybil resistance bonus

        Args:
            request_id: Authorization request ID

        Returns:
            Dict with detailed weighting breakdown:
            {
                "guardian_weights": Dict[str, GuardianWeightBreakdown],
                "total_weighted_approvals": float,
                "total_weighted_rejections": float,
                "effective_approval_ratio": float,
                "weight_distribution": Dict with stats
            }
        """
        # Retrieve approvals
        try:
            approvals_data = await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="get_guardian_approvals",
                payload={"request_id": request_id},
            )
            approvals = [GuardianApprovalData.from_dict(a) for a in (approvals_data or [])]
        except Exception as e:
            logger.error(f"Failed to retrieve guardian approvals: {e}")
            return {"success": False, "error": str(e)}

        if not approvals:
            return {
                "success": True,
                "guardian_weights": {},
                "total_weighted_approvals": 0.0,
                "total_weighted_rejections": 0.0,
                "effective_approval_ratio": 0.0,
                "weight_distribution": {"min": 0.0, "max": 0.0, "avg": 0.0, "std": 0.0}
            }

        # Calculate weights for each guardian in parallel
        tasks = []
        for approval in approvals:
            tasks.append(self._calculate_guardian_weight(approval))

        weight_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        guardian_weights = {}
        total_weighted_approvals = 0.0
        total_weighted_rejections = 0.0
        all_weights = []

        for approval, result in zip(approvals, weight_results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Failed to calculate weight for guardian {approval.guardian_participant_id}: {result}"
                )
                # Use fallback weight
                weight = 1.0
                breakdown = {
                    "total_weight": weight,
                    "base_weight": 1.0,
                    "assurance_multiplier": 1.0,
                    "reputation_multiplier": 1.0,
                    "time_weight": 1.0,
                    "sybil_bonus": 0.0,
                    "guardian_quality_bonus": 0.0,
                    "error": str(result)
                }
            else:
                weight = result["total_weight"]
                breakdown = result

            guardian_weights[approval.guardian_participant_id] = {
                "approved": approval.approved,
                "weight": weight,
                "breakdown": breakdown
            }

            all_weights.append(weight)

            if approval.approved:
                total_weighted_approvals += weight
            else:
                total_weighted_rejections += weight

        # Calculate weight distribution statistics
        import statistics
        weight_stats = {
            "min": min(all_weights) if all_weights else 0.0,
            "max": max(all_weights) if all_weights else 0.0,
            "avg": statistics.mean(all_weights) if all_weights else 0.0,
            "std": statistics.stdev(all_weights) if len(all_weights) > 1 else 0.0,
        }

        total_weight = total_weighted_approvals + total_weighted_rejections
        effective_ratio = (
            total_weighted_approvals / total_weight if total_weight > 0 else 0.0
        )

        return {
            "success": True,
            "guardian_weights": guardian_weights,
            "total_weighted_approvals": total_weighted_approvals,
            "total_weighted_rejections": total_weighted_rejections,
            "effective_approval_ratio": effective_ratio,
            "weight_distribution": weight_stats,
            "total_guardians": len(approvals),
            "approving_guardians": sum(1 for a in approvals if a.approved),
            "rejecting_guardians": sum(1 for a in approvals if not a.approved),
        }

    async def _calculate_guardian_weight(
        self,
        approval: "GuardianApprovalData",
    ) -> Dict[str, float]:
        """
        Calculate comprehensive weight for a single guardian's approval.

        Args:
            approval: Guardian approval data

        Returns:
            Dict with weight breakdown
        """
        guardian_id = approval.guardian_participant_id

        # Get base voting power from calculator
        voting_power = await self.voting_power_calc.calculate_voting_power(
            participant_id=guardian_id,
            include_guardian_endorsements=False,  # Avoid circular lookups
        )

        # Base components from voting power calculation
        base_weight = voting_power.base_weight
        assurance_multiplier = voting_power.assurance_multiplier
        reputation_multiplier = voting_power.reputation_multiplier
        time_weight = voting_power.time_weight
        sybil_bonus = voting_power.sybil_bonus

        # Additional: Guardian quality bonus
        # Guardians who have successfully participated in previous authorizations
        # get a small quality bonus
        guardian_quality_bonus = 0.0
        try:
            guardian_history = await self.dht.call_zome(
                zome_name="governance_record",
                fn_name="get_guardian_approval_history",
                payload={
                    "guardian_participant_id": guardian_id,
                    "limit": 100,
                },
            )

            if guardian_history:
                successful_approvals = sum(
                    1 for h in guardian_history
                    if h.get("outcome") == "approved" and h.get("was_approver", False)
                )
                # Up to 10% bonus for consistent guardian participation
                guardian_quality_bonus = min(0.1, successful_approvals * 0.01)

        except Exception as e:
            # Guardian history is optional - continue without bonus
            pass

        # Calculate total weight
        total_weight = (
            base_weight
            * assurance_multiplier
            * reputation_multiplier
            * (1 + sybil_bonus + guardian_quality_bonus)
            * time_weight
        )

        return {
            "total_weight": total_weight,
            "base_weight": base_weight,
            "assurance_multiplier": assurance_multiplier,
            "reputation_multiplier": reputation_multiplier,
            "time_weight": time_weight,
            "sybil_bonus": sybil_bonus,
            "guardian_quality_bonus": guardian_quality_bonus,
        }


class GovernanceCoordinator:
    """
    Main governance coordinator

    Integrates all governance components:
    - Proposal management
    - Voting engine
    - Capability enforcement
    - Guardian authorization
    - Ed25519 cryptographic signing for governance messages

    Provides high-level governance operations for FL coordinator integration.
    """

    def __init__(
        self,
        dht_client,
        identity_coordinator,
        key_dir: Optional[Path] = None,
        participant_id: Optional[str] = None,
    ):
        """
        Args:
            dht_client: Holochain DHT client
            identity_coordinator: Identity coordinator instance
            key_dir: Optional directory for storing Ed25519 signing keys.
                    Defaults to ~/.mycelix/governance_keys/
            participant_id: Optional participant ID for key isolation.
                           Each participant gets their own signing key.
        """
        self.dht = dht_client
        self.identity_coord = identity_coordinator

        # Import governance extensions
        from zerotrustml.identity import IdentityGovernanceExtensions

        self.gov_extensions = IdentityGovernanceExtensions(identity_coordinator)

        # Initialize Ed25519 governance signer
        self.signer = GovernanceSigner(key_dir=key_dir, participant_id=participant_id)
        logger.info(f"Governance signer initialized with public key: {self.signer.get_public_key_hex()[:16]}...")

        # Initialize shared components
        self.voting_power_calc = VotingPowerCalculator(
            identity_governance_extensions=self.gov_extensions,
            dht_client=dht_client,
        )

        self.guardian_verifier = GuardianVerifier(
            identity_governance_extensions=self.gov_extensions,
            dht_client=dht_client,
        )

        # Initialize component managers
        self.proposal_mgr = ProposalManager(
            dht_client=dht_client,
            identity_governance_extensions=self.gov_extensions,
            governance_signer=self.signer,
        )

        self.voting_engine = VotingEngine(
            dht_client=dht_client,
            identity_governance_extensions=self.gov_extensions,
            proposal_manager=self.proposal_mgr,
            governance_signer=self.signer,
            voting_power_calculator=self.voting_power_calc,
            guardian_verifier=self.guardian_verifier,
        )

        self.capability_enforcer = CapabilityEnforcer(
            identity_governance_extensions=self.gov_extensions,
        )

        self.guardian_auth_mgr = GuardianAuthorizationManager(
            dht_client=dht_client,
            identity_governance_extensions=self.gov_extensions,
            governance_signer=self.signer,
            guardian_verifier=self.guardian_verifier,
            voting_power_calculator=self.voting_power_calc,
        )

        logger.info("GovernanceCoordinator initialized with Ed25519 cryptographic signing")

    # High-level operations for FL integration

    async def authorize_fl_action(
        self,
        participant_id: str,
        action: str,
        action_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Authorize FL action (for FL coordinator integration)

        Args:
            participant_id: Participant performing action
            action: Action name (submit_update, request_model, etc.)
            action_params: Optional action parameters

        Returns:
            Tuple of (authorized, reason)
        """
        # Map FL actions to capabilities
        action_capability_map = {
            "submit_update": "submit_update",
            "request_model": "request_model",
            "submit_mip": "submit_mip",
            "emergency_stop": "emergency_stop",
            "ban_participant": "ban_participant",
            "change_parameters": "change_parameters",
        }

        capability_id = action_capability_map.get(action)
        if not capability_id:
            return True, "Action does not require governance"

        # Check capability
        return await self.capability_enforcer.check_capability(
            participant_id=participant_id,
            capability_id=capability_id,
        )

    async def create_governance_proposal(
        self,
        proposer_participant_id: str,
        proposal_type: str,
        title: str,
        description: str,
        execution_params: Dict[str, Any],
        voting_duration_days: int = 7,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Create governance proposal (simplified interface)

        Args:
            proposer_participant_id: Participant proposing
            proposal_type: Type string (PARAMETER_CHANGE, etc.)
            title: Proposal title
            description: Detailed description
            execution_params: Parameters for execution
            voting_duration_days: Voting period in days

        Returns:
            Tuple of (success, message, proposal_id)
        """
        try:
            prop_type = ProposalType(proposal_type)
        except ValueError:
            return False, f"Invalid proposal type: {proposal_type}", None

        voting_duration_seconds = voting_duration_days * 86400

        success, message, proposal = await self.proposal_mgr.create_proposal(
            proposer_participant_id=proposer_participant_id,
            proposal_type=prop_type,
            title=title,
            description=description,
            voting_duration_seconds=voting_duration_seconds,
            execution_params=execution_params,
        )

        proposal_id = proposal.proposal_id if proposal else None

        return success, message, proposal_id

    async def cast_governance_vote(
        self,
        voter_participant_id: str,
        proposal_id: str,
        choice: str,
        credits: int = 100,
    ) -> Tuple[bool, str]:
        """
        Cast vote on proposal (simplified interface)

        Args:
            voter_participant_id: Participant voting
            proposal_id: Proposal ID
            choice: Vote choice string (FOR, AGAINST, ABSTAIN)
            credits: Credits to spend

        Returns:
            Tuple of (success, message)
        """
        try:
            vote_choice = VoteChoice(choice)
        except ValueError:
            return False, f"Invalid vote choice: {choice}"

        return await self.voting_engine.cast_vote(
            voter_participant_id=voter_participant_id,
            proposal_id=proposal_id,
            choice=vote_choice,
            credits_spent=credits,
        )

    async def get_governance_stats(self) -> Dict[str, Any]:
        """
        Get governance system statistics

        Returns:
            Dictionary with system stats
        """
        # Get proposal counts by status
        status_counts = {}
        for status in ProposalStatus:
            proposals = await self.proposal_mgr.list_proposals_by_status(status)
            status_counts[status.value] = len(proposals)

        # Get participant stats
        participant_stats = await self.gov_extensions.get_governance_stats()

        return {
            "proposal_counts": status_counts,
            "participant_stats": participant_stats,
            "system_version": "0.1.0-alpha",
            "signing_enabled": self.signer is not None,
            "public_key": self.signer.get_public_key_hex() if self.signer else None,
        }

    def get_signing_public_key(self) -> str:
        """
        Get the governance coordinator's Ed25519 public key.

        This key should be shared with other participants for signature verification.

        Returns:
            Hex-encoded public key string
        """
        return self.signer.get_public_key_hex()

    def get_signing_public_key_bytes(self) -> bytes:
        """
        Get the governance coordinator's Ed25519 public key as bytes.

        Returns:
            32-byte public key
        """
        return self.signer.get_public_key_bytes()

    async def verify_vote(
        self,
        vote: VoteData,
        public_key_bytes: Optional[bytes] = None,
    ) -> Tuple[bool, str]:
        """
        Verify an Ed25519 signature on a vote.

        Args:
            vote: The vote data to verify
            public_key_bytes: Optional voter's public key. If not provided,
                            will attempt to look up from identity system.

        Returns:
            Tuple of (valid, message)
        """
        return await self.voting_engine.verify_vote_signature(vote, public_key_bytes)

    async def verify_guardian_approval(
        self,
        approval: GuardianApprovalData,
        public_key_bytes: Optional[bytes] = None,
    ) -> Tuple[bool, str]:
        """
        Verify an Ed25519 signature on a guardian approval.

        Args:
            approval: The approval data to verify
            public_key_bytes: Optional guardian's public key. If not provided,
                            will attempt to look up from identity system.

        Returns:
            Tuple of (valid, message)
        """
        return await self.guardian_auth_mgr.verify_approval_signature(approval, public_key_bytes)

    def register_participant_public_key(
        self,
        participant_id: str,
        public_key_bytes: bytes,
        is_guardian: bool = False,
    ) -> None:
        """
        Register a participant's public key for signature verification.

        Args:
            participant_id: The participant's ID
            public_key_bytes: The participant's Ed25519 public key (32 bytes)
            is_guardian: If True, also register as a guardian key
        """
        self.voting_engine.register_voter_public_key(participant_id, public_key_bytes)
        if is_guardian:
            self.guardian_auth_mgr.register_guardian_public_key(participant_id, public_key_bytes)
