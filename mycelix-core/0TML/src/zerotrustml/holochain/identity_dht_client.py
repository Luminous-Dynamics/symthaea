#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Identity DHT Client for Zero-TrustML

Extends HolochainClient with identity DNA operations:
- DID Registry: DID document management
- Identity Store: Multi-factor identity storage and trust signals
- Reputation Sync: Cross-network reputation aggregation
- Guardian Graph: Social recovery and cartel detection

Week 5-6 Phase 5: Python DHT Client Integration
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio

from .client import HolochainClient

logger = logging.getLogger(__name__)


@dataclass
class DIDDocument:
    """W3C DID Document"""
    id: str
    controller: str  # AgentPubKey as hex string
    verification_methods: List[Dict[str, Any]]
    authentication: List[str]
    created: int
    updated: int
    proof: Optional[Dict[str, Any]] = None


@dataclass
class IdentityFactor:
    """Identity factor (CryptoKey, GitcoinPassport, etc.)"""
    factor_id: str
    factor_type: str
    category: str
    status: str
    metadata: str  # JSON string
    added: int
    last_verified: int


@dataclass
class IdentitySignals:
    """Computed identity trust signals"""
    did: str
    assurance_level: str  # E0-E4
    sybil_resistance: float
    risk_level: str
    guardian_graph_diversity: float
    verified_human: bool
    credential_count: int
    computed_at: int


@dataclass
class ReputationEntry:
    """Reputation from a specific network"""
    did: str
    network_id: str
    reputation_score: float
    raw_score: float
    score_type: str
    metadata: str
    issued_at: int
    expires_at: Optional[int]
    issuer: str


@dataclass
class AggregatedReputation:
    """Aggregated reputation across networks"""
    did: str
    global_score: float
    network_count: int
    network_scores: str  # JSON map
    trust_score: float
    contribution_score: float
    verification_score: float
    last_updated: int
    version: int


@dataclass
class GuardianRelationship:
    """Guardian relationship between DIDs"""
    subject_did: str
    guardian_did: str
    relationship_type: str  # RECOVERY, ENDORSEMENT, DELEGATION
    weight: float
    status: str  # ACTIVE, PENDING, REVOKED
    metadata: str
    established_at: int
    expires_at: Optional[int]
    mutual: bool


@dataclass
class GuardianGraphMetrics:
    """Guardian network metrics"""
    did: str
    guardian_count: int
    guarded_count: int
    diversity_score: float
    cartel_risk_score: float
    cluster_id: Optional[str]
    average_guardian_reputation: float
    network_degree: int
    computed_at: int
    version: int


class IdentityDHTClient(HolochainClient):
    """
    Identity DHT Client for Zero-TrustML

    Extends HolochainClient with identity DNA operations across 4 zomes:
    1. DID Registry: DID document management
    2. Identity Store: Multi-factor identity and trust signals
    3. Reputation Sync: Cross-network reputation
    4. Guardian Graph: Social recovery
    """

    def __init__(
        self,
        admin_url: str = "ws://localhost:8888",
        app_url: str = "ws://localhost:8889",
        app_id: str = "zerotrustml_identity",  # Identity DNA app ID
        cell_id: Optional[str] = None
    ):
        super().__init__(admin_url, app_url, app_id, cell_id)
        logger.info(f"✨ Identity DHT Client initialized (app: {app_id})")

    # ============================================================
    # DID Registry Operations (Phase 1)
    # ============================================================

    async def create_did(
        self,
        did: str,
        controller: str,
        verification_methods: List[Dict[str, Any]],
        authentication: List[str],
        proof: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create DID document on DHT

        Args:
            did: DID string (e.g., "did:mycelix:abc123")
            controller: AgentPubKey as hex string
            verification_methods: List of verification method dicts
            authentication: List of key IDs for authentication
            proof: Optional self-signature proof

        Returns:
            ActionHash as hex string
        """
        import time

        payload = {
            "document": {
                "id": did,
                "controller": controller,
                "verification_methods": verification_methods,
                "authentication": authentication,
                "created": int(time.time() * 1_000_000),  # Microseconds
                "updated": int(time.time() * 1_000_000),
                "proof": proof
            }
        }

        result = await self._call_zome("did_registry", "create_did", payload)
        logger.info(f"✅ DID created: {did}")
        return result

    async def resolve_did(self, did: str) -> Optional[DIDDocument]:
        """
        Resolve DID to DID document

        Args:
            did: DID string to resolve

        Returns:
            DIDDocument or None if not found
        """
        result = await self._call_zome("did_registry", "resolve_did", did)

        if result:
            return DIDDocument(**result)
        return None

    async def update_did(
        self,
        did: str,
        original_action_hash: str,
        new_document: Dict[str, Any]
    ) -> str:
        """
        Update existing DID document

        Args:
            did: DID string
            original_action_hash: Hash of current version
            new_document: New DID document dict

        Returns:
            ActionHash of updated document
        """
        payload = {
            "did": did,
            "original_action_hash": original_action_hash,
            "new_document": new_document
        }

        result = await self._call_zome("did_registry", "update_did", payload)
        logger.info(f"✅ DID updated: {did}")
        return result

    async def deactivate_did(self, did: str) -> None:
        """
        Deactivate DID (make unresolvable)

        Args:
            did: DID string to deactivate
        """
        await self._call_zome("did_registry", "deactivate_did", did)
        logger.info(f"❌ DID deactivated: {did}")

    # ============================================================
    # Identity Store Operations (Phase 2)
    # ============================================================

    async def store_identity_factors(
        self,
        did: str,
        factors: List[Dict[str, Any]]
    ) -> str:
        """
        Store identity factors for a DID

        Args:
            did: DID string
            factors: List of identity factor dicts

        Returns:
            ActionHash
        """
        payload = {
            "did": did,
            "factors": factors
        }

        result = await self._call_zome("identity_store", "store_identity_factors", payload)
        logger.info(f"✅ Identity factors stored for {did}: {len(factors)} factors")
        return result

    async def get_identity_factors(self, did: str) -> Optional[Dict[str, Any]]:
        """
        Get identity factors for a DID

        Args:
            did: DID string

        Returns:
            IdentityFactors dict or None
        """
        result = await self._call_zome("identity_store", "get_identity_factors", did)
        return result

    async def update_identity_factors(
        self,
        did: str,
        factors_to_add: List[Dict[str, Any]],
        factors_to_update: List[Dict[str, Any]],
        factor_ids_to_remove: List[str]
    ) -> str:
        """
        Update identity factors (add/update/remove)

        Args:
            did: DID string
            factors_to_add: New factors to add
            factors_to_update: Existing factors to update
            factor_ids_to_remove: Factor IDs to remove

        Returns:
            ActionHash
        """
        payload = {
            "did": did,
            "factors_to_add": factors_to_add,
            "factors_to_update": factors_to_update,
            "factor_ids_to_remove": factor_ids_to_remove
        }

        result = await self._call_zome("identity_store", "update_identity_factors", payload)
        logger.info(f"✅ Identity factors updated for {did}")
        return result

    async def store_verifiable_credential(
        self,
        credential_id: str,
        issuer_did: str,
        subject_did: str,
        vc_type: str,
        claims: str,  # JSON string
        proof: bytes,
        issued_at: int,
        expires_at: Optional[int] = None
    ) -> str:
        """
        Store verifiable credential

        Args:
            credential_id: Unique credential ID
            issuer_did: DID of issuer
            subject_did: DID of subject
            vc_type: Credential type (e.g., "VerifiedHuman")
            claims: JSON-encoded claims
            proof: Cryptographic proof bytes
            issued_at: Issuance timestamp (microseconds)
            expires_at: Optional expiration timestamp

        Returns:
            ActionHash
        """
        payload = {
            "credential": {
                "id": credential_id,
                "issuer_did": issuer_did,
                "subject_did": subject_did,
                "vc_type": vc_type,
                "claims": claims,
                "proof": list(proof),  # Convert bytes to list for serialization
                "issued_at": issued_at,
                "expires_at": expires_at
            }
        }

        result = await self._call_zome("identity_store", "store_verifiable_credential", payload)
        logger.info(f"✅ Verifiable credential stored: {vc_type} for {subject_did}")
        return result

    async def get_credentials(self, did: str) -> List[Dict[str, Any]]:
        """
        Get all credentials for a DID

        Args:
            did: DID string

        Returns:
            List of credential dicts
        """
        result = await self._call_zome("identity_store", "get_credentials", did)
        return result or []

    async def get_credentials_by_type(self, did: str, vc_type: str) -> List[Dict[str, Any]]:
        """
        Get credentials of specific type for a DID

        Args:
            did: DID string
            vc_type: Credential type to query

        Returns:
            List of credential dicts
        """
        payload = {
            "did": did,
            "vc_type": vc_type
        }

        result = await self._call_zome("identity_store", "get_credentials_by_type", payload)
        return result or []

    async def compute_identity_signals(self, did: str) -> IdentitySignals:
        """
        Compute identity trust signals (E0-E4, Sybil resistance, etc.)

        Args:
            did: DID string

        Returns:
            IdentitySignals
        """
        result = await self._call_zome("identity_store", "compute_identity_signals", did)
        logger.info(f"✅ Identity signals computed for {did}: {result['assurance_level']}")
        return IdentitySignals(**result)

    async def get_identity_signals(self, did: str) -> Optional[IdentitySignals]:
        """
        Get computed identity signals

        Args:
            did: DID string

        Returns:
            IdentitySignals or None
        """
        result = await self._call_zome("identity_store", "get_identity_signals", did)

        if result:
            return IdentitySignals(**result)
        return None

    # ============================================================
    # Reputation Sync Operations (Phase 3)
    # ============================================================

    async def store_reputation_entry(
        self,
        did: str,
        network_id: str,
        reputation_score: float,
        raw_score: float,
        score_type: str,
        metadata: str,
        issued_at: int,
        issuer: str,
        expires_at: Optional[int] = None
    ) -> str:
        """
        Store reputation entry from a network

        Args:
            did: Subject DID
            network_id: Network identifier (e.g., "zero_trustml", "gitcoin_passport")
            reputation_score: Normalized score (0.0-1.0)
            raw_score: Original network score
            score_type: Type ("trust", "contribution", "verification")
            metadata: JSON-encoded network-specific data
            issued_at: Timestamp (microseconds)
            issuer: Network authority DID
            expires_at: Optional expiration timestamp

        Returns:
            ActionHash
        """
        payload = {
            "entry": {
                "did": did,
                "network_id": network_id,
                "reputation_score": reputation_score,
                "raw_score": raw_score,
                "score_type": score_type,
                "metadata": metadata,
                "issued_at": issued_at,
                "expires_at": expires_at,
                "issuer": issuer
            }
        }

        result = await self._call_zome("reputation_sync", "store_reputation_entry", payload)
        logger.info(f"✅ Reputation entry stored: {network_id} for {did} ({reputation_score:.2f})")
        return result

    async def get_reputation_for_network(self, did: str, network_id: str) -> Optional[ReputationEntry]:
        """
        Get reputation entry for specific network

        Args:
            did: DID string
            network_id: Network identifier

        Returns:
            ReputationEntry or None
        """
        result = await self._call_zome("reputation_sync", "get_reputation_for_network", {
            "did": did,
            "network_id": network_id
        })

        if result:
            return ReputationEntry(**result)
        return None

    async def get_all_reputation_entries(self, did: str) -> List[ReputationEntry]:
        """
        Get all reputation entries for a DID

        Args:
            did: DID string

        Returns:
            List of ReputationEntry
        """
        result = await self._call_zome("reputation_sync", "get_all_reputation_entries", did)
        return [ReputationEntry(**entry) for entry in result]

    async def get_aggregated_reputation(self, did: str) -> Optional[AggregatedReputation]:
        """
        Get aggregated reputation across all networks

        Args:
            did: DID string

        Returns:
            AggregatedReputation or None
        """
        result = await self._call_zome("reputation_sync", "get_aggregated_reputation", did)

        if result:
            return AggregatedReputation(**result)
        return None

    async def update_aggregated_reputation(self, did: str) -> str:
        """
        Force update of aggregated reputation

        Args:
            did: DID string

        Returns:
            ActionHash
        """
        result = await self._call_zome("reputation_sync", "update_aggregated_reputation", did)
        logger.info(f"✅ Aggregated reputation updated for {did}")
        return result

    # ============================================================
    # Guardian Graph Operations (Phase 4)
    # ============================================================

    async def add_guardian(
        self,
        subject_did: str,
        guardian_did: str,
        relationship_type: str,
        weight: float,
        status: str = "ACTIVE",
        metadata: str = "{}",
        mutual: bool = False,
        expires_at: Optional[int] = None
    ) -> str:
        """
        Add guardian relationship

        Args:
            subject_did: DID being guarded
            guardian_did: DID acting as guardian
            relationship_type: Type ("RECOVERY", "ENDORSEMENT", "DELEGATION")
            weight: Guardian influence (0.0-1.0)
            status: Status ("ACTIVE", "PENDING", "REVOKED")
            metadata: JSON-encoded relationship data
            mutual: Whether relationship is bidirectional
            expires_at: Optional expiration timestamp

        Returns:
            ActionHash
        """
        import time

        payload = {
            "relationship": {
                "subject_did": subject_did,
                "guardian_did": guardian_did,
                "relationship_type": relationship_type,
                "weight": weight,
                "status": status,
                "metadata": metadata,
                "established_at": int(time.time() * 1_000_000),
                "expires_at": expires_at,
                "mutual": mutual
            }
        }

        result = await self._call_zome("guardian_graph", "add_guardian", payload)
        logger.info(f"✅ Guardian added: {guardian_did} guards {subject_did} ({relationship_type}, weight={weight})")
        return result

    async def remove_guardian(
        self,
        subject_did: str,
        guardian_did: str,
        original_action_hash: str
    ) -> None:
        """
        Remove (revoke) guardian relationship

        Args:
            subject_did: DID being guarded
            guardian_did: Guardian DID
            original_action_hash: Hash of relationship to revoke
        """
        payload = {
            "subject_did": subject_did,
            "guardian_did": guardian_did,
            "original_action_hash": original_action_hash
        }

        await self._call_zome("guardian_graph", "remove_guardian", payload)
        logger.info(f"❌ Guardian removed: {guardian_did} no longer guards {subject_did}")

    async def get_guardians(self, did: str) -> List[GuardianRelationship]:
        """
        Get all guardians for a DID

        Args:
            did: DID string

        Returns:
            List of GuardianRelationship
        """
        result = await self._call_zome("guardian_graph", "get_guardians", did)
        return [GuardianRelationship(**rel) for rel in result]

    async def get_guarded_by(self, guardian_did: str) -> List[GuardianRelationship]:
        """
        Get all DIDs guarded by a guardian

        Args:
            guardian_did: Guardian DID

        Returns:
            List of GuardianRelationship
        """
        result = await self._call_zome("guardian_graph", "get_guarded_by", guardian_did)
        return [GuardianRelationship(**rel) for rel in result]

    async def compute_guardian_metrics(self, did: str) -> str:
        """
        Compute guardian graph metrics

        Args:
            did: DID string

        Returns:
            ActionHash
        """
        result = await self._call_zome("guardian_graph", "compute_guardian_metrics", did)
        logger.info(f"✅ Guardian metrics computed for {did}")
        return result

    async def get_guardian_metrics(self, did: str) -> Optional[GuardianGraphMetrics]:
        """
        Get guardian graph metrics

        Args:
            did: DID string

        Returns:
            GuardianGraphMetrics or None
        """
        result = await self._call_zome("guardian_graph", "get_guardian_metrics", did)

        if result:
            return GuardianGraphMetrics(**result)
        return None

    async def authorize_recovery(
        self,
        subject_did: str,
        approving_guardians: List[str],
        required_threshold: float
    ) -> Dict[str, Any]:
        """
        Check if recovery is authorized by guardian consensus

        Args:
            subject_did: DID to recover
            approving_guardians: List of guardian DIDs approving recovery
            required_threshold: Minimum total weight needed (e.g., 0.6 for 60%)

        Returns:
            RecoveryAuthorization dict with:
                - authorized: bool
                - reason: str
                - threshold_met: bool
                - approval_weight: float
                - required_weight: float
        """
        payload = {
            "subject_did": subject_did,
            "approving_guardians": approving_guardians,
            "required_threshold": required_threshold
        }

        result = await self._call_zome("guardian_graph", "authorize_recovery", payload)

        if result["authorized"]:
            logger.info(f"✅ Recovery authorized for {subject_did}: {result['approval_weight']}/{required_threshold}")
        else:
            logger.warning(f"❌ Recovery NOT authorized for {subject_did}: {result['reason']}")

        return result

    # ============================================================
    # High-Level Identity Operations
    # ============================================================

    async def get_complete_identity(self, did: str) -> Dict[str, Any]:
        """
        Get complete identity profile (DID + factors + signals + reputation + guardians)

        Args:
            did: DID string

        Returns:
            Complete identity dict
        """
        logger.info(f"🔍 Fetching complete identity for {did}...")

        # Fetch all components in parallel
        results = await asyncio.gather(
            self.resolve_did(did),
            self.get_identity_factors(did),
            self.get_identity_signals(did),
            self.get_aggregated_reputation(did),
            self.get_guardians(did),
            self.get_guardian_metrics(did),
            return_exceptions=True
        )

        did_doc, factors, signals, reputation, guardians, guardian_metrics = results

        # Handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Error fetching component {i}: {result}")

        identity = {
            "did": did,
            "did_document": did_doc,
            "identity_factors": factors,
            "identity_signals": signals,
            "aggregated_reputation": reputation,
            "guardians": guardians,
            "guardian_metrics": guardian_metrics,
            "complete": all(not isinstance(r, Exception) for r in results)
        }

        logger.info(f"✅ Complete identity fetched for {did}")
        return identity

    async def verify_identity_for_fl(self, did: str, required_assurance: str = "E2") -> Dict[str, Any]:
        """
        Verify identity meets requirements for federated learning participation

        Args:
            did: DID string
            required_assurance: Minimum assurance level (default E2)

        Returns:
            Verification result dict with:
                - verified: bool
                - assurance_level: str
                - sybil_resistance: float
                - cartel_risk: float
                - reasons: List[str]
        """
        logger.info(f"🔒 Verifying identity for FL: {did} (min: {required_assurance})")

        # Get identity signals and guardian metrics
        signals = await self.get_identity_signals(did)
        guardian_metrics = await self.get_guardian_metrics(did)

        if not signals:
            return {
                "verified": False,
                "reasons": ["No identity signals found - identity not initialized"]
            }

        reasons = []
        verified = True

        # Check assurance level
        assurance_levels = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 4}
        if assurance_levels.get(signals.assurance_level, 0) < assurance_levels.get(required_assurance, 2):
            verified = False
            reasons.append(f"Assurance level {signals.assurance_level} below required {required_assurance}")

        # Check Sybil resistance
        if signals.sybil_resistance < 0.5:
            verified = False
            reasons.append(f"Sybil resistance too low: {signals.sybil_resistance:.2f} < 0.5")

        # Check cartel risk
        if guardian_metrics and guardian_metrics.cartel_risk_score > 0.7:
            verified = False
            reasons.append(f"High cartel risk detected: {guardian_metrics.cartel_risk_score:.2f}")

        # Check risk level
        if signals.risk_level == "HIGH":
            verified = False
            reasons.append(f"Identity flagged as high risk")

        if verified:
            reasons.append("Identity verification passed")
            logger.info(f"✅ Identity verified for FL: {did} ({signals.assurance_level})")
        else:
            logger.warning(f"❌ Identity verification failed for FL: {did} - {reasons}")

        return {
            "verified": verified,
            "assurance_level": signals.assurance_level,
            "sybil_resistance": signals.sybil_resistance,
            "cartel_risk": guardian_metrics.cartel_risk_score if guardian_metrics else 0.0,
            "reasons": reasons
        }


# ============================================================
# Convenience Functions
# ============================================================

async def create_identity_client(
    admin_url: str = "ws://localhost:8888",
    app_url: str = "ws://localhost:8889",
    app_id: str = "zerotrustml_identity"
) -> IdentityDHTClient:
    """
    Create and connect to identity DHT client

    Args:
        admin_url: Holochain admin WebSocket URL
        app_url: Holochain app WebSocket URL
        app_id: Identity DNA app ID

    Returns:
        Connected IdentityDHTClient
    """
    client = IdentityDHTClient(admin_url, app_url, app_id)
    await client.connect()
    return client
