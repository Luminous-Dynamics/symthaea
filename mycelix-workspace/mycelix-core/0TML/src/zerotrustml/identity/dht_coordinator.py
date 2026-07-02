#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
DHT-Aware Identity Coordinator for Zero-TrustML

Bridges existing Python identity infrastructure with Holochain DHT:
- Bidirectional DID synchronization
- Identity factor DHT storage
- Reputation sync from FL to DHT
- Guardian network management

Week 5-6 Phase 6: Integration with Identity Coordinator
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from .did_manager import DIDManager, MycelixDID, AgentType
from .factors import (
    IdentityFactor,
    CryptoKeyFactor,
    GitcoinPassportFactor,
    SocialRecoveryFactor,
    HardwareKeyFactor,
    FactorStatus,
    FactorCategory
)
from .assurance import AssuranceLevel, calculate_assurance_level
from .recovery import RecoveryManager, RecoveryScenario
from .verifiable_credentials import VCManager, VerifiableCredential
from .matl_integration import IdentityMATLBridge, IdentityRiskLevel

# Import DHT client (will work if holochain module available)
try:
    from ..holochain import (
        IdentityDHTClient,
        create_identity_client,
        IdentitySignals
    )
    DHT_AVAILABLE = True
except ImportError:
    DHT_AVAILABLE = False
    logging.warning("Holochain DHT client not available - operating in local-only mode")

logger = logging.getLogger(__name__)


@dataclass
class IdentityCoordinatorConfig:
    """Configuration for DHT-aware identity coordinator"""
    # DHT connection
    dht_enabled: bool = True
    dht_admin_url: str = "ws://localhost:8888"
    dht_app_url: str = "ws://localhost:8889"
    dht_app_id: str = "zerotrustml_identity"

    # Synchronization
    auto_sync_to_dht: bool = True
    sync_interval: int = 300  # 5 minutes
    sync_on_update: bool = True

    # Caching
    cache_dht_queries: bool = True
    cache_ttl: int = 600  # 10 minutes

    # Local storage fallback
    enable_local_fallback: bool = True


class DHT_IdentityCoordinator:
    """
    DHT-Aware Identity Coordinator

    Integrates existing Python identity infrastructure with Holochain DHT:
    1. Creates/manages local identities (DIDManager, factors, credentials)
    2. Syncs identities to DHT (bidirectional)
    3. Queries DHT for identity verification
    4. Manages guardians on DHT
    5. Syncs reputation from FL coordinator
    """

    def __init__(
        self,
        config: Optional[IdentityCoordinatorConfig] = None,
        did_manager: Optional[DIDManager] = None,
        recovery_manager: Optional[RecoveryManager] = None,
        vc_manager: Optional[VCManager] = None,
        matl_bridge: Optional[IdentityMATLBridge] = None
    ):
        """
        Initialize DHT-aware identity coordinator

        Args:
            config: Coordinator configuration
            did_manager: Existing DID manager (or create new)
            recovery_manager: Existing recovery manager
            vc_manager: Existing verifiable credentials manager
            matl_bridge: Existing MATL bridge
        """
        self.config = config or IdentityCoordinatorConfig()

        # Initialize local managers
        self.did_manager = did_manager or DIDManager()
        self.recovery_manager = recovery_manager or RecoveryManager()
        self.vc_manager = vc_manager or VCManager()
        self.matl_bridge = matl_bridge or IdentityMATLBridge()

        # DHT client (async)
        self.dht_client: Optional[IdentityDHTClient] = None
        self.dht_connected = False

        # Local caches
        self.identity_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, float] = {}

        # Identity registry (participant_id -> DID mapping)
        self.participant_dids: Dict[str, str] = {}

        logger.info(f"✨ DHT-Aware Identity Coordinator initialized (DHT: {self.config.dht_enabled})")

    async def initialize_dht(self) -> bool:
        """
        Initialize connection to Holochain DHT

        Returns:
            True if connection successful, False otherwise
        """
        if not self.config.dht_enabled:
            logger.info("DHT disabled in config - operating in local-only mode")
            return False

        if not DHT_AVAILABLE:
            logger.warning("DHT client not available - install websockets and msgpack")
            return False

        try:
            self.dht_client = await create_identity_client(
                admin_url=self.config.dht_admin_url,
                app_url=self.config.dht_app_url,
                app_id=self.config.dht_app_id
            )
            self.dht_connected = True
            logger.info("✅ DHT connection established")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to connect to DHT: {e}")
            if self.config.enable_local_fallback:
                logger.info("Falling back to local-only mode")
            return False

    async def close_dht(self):
        """Close DHT connection"""
        if self.dht_client:
            await self.dht_client.disconnect()
            self.dht_connected = False
            logger.info("DHT connection closed")

    # ============================================================
    # Identity Creation and Registration
    # ============================================================

    async def create_identity(
        self,
        participant_id: str,
        agent_type: AgentType = AgentType.HUMAN_MEMBER,
        initial_factors: Optional[List[IdentityFactor]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create new identity (local + DHT)

        Args:
            participant_id: Local participant identifier
            agent_type: Type of agent (HUMAN_MEMBER, INSTRUMENTAL_ACTOR, etc.)
            initial_factors: Initial identity factors
            metadata: Additional metadata

        Returns:
            DID string (e.g., "did:mycelix:z6Mk...")
        """
        logger.info(f"Creating identity for participant: {participant_id}")

        # 1. Create local DID
        mycelix_did = self.did_manager.create_did(agent_type=agent_type, metadata=metadata)
        did_string = mycelix_did.to_string()

        # 2. Store participant -> DID mapping
        self.participant_dids[participant_id] = did_string

        # 3. Add initial factors if provided
        if initial_factors:
            for factor in initial_factors:
                self.did_manager.storage[did_string]["factors"] = (
                    self.did_manager.storage.get(did_string, {}).get("factors", []) + [factor]
                )

        # 4. Sync to DHT if enabled
        if self.dht_connected and self.config.auto_sync_to_dht:
            try:
                await self._sync_to_dht(mycelix_did, initial_factors or [])
                logger.info(f"✅ Identity synced to DHT: {did_string}")
            except Exception as e:
                logger.error(f"Failed to sync to DHT: {e}")
                if not self.config.enable_local_fallback:
                    raise

        logger.info(f"✅ Identity created: {did_string} (participant: {participant_id})")
        return did_string

    async def _sync_to_dht(self, mycelix_did: MycelixDID, factors: List[IdentityFactor]):
        """Sync local identity to DHT"""
        if not self.dht_client:
            return

        did_string = mycelix_did.to_string()
        did_doc = mycelix_did.to_did_document()

        # 1. Create DID on DHT
        controller_hex = mycelix_did.public_key.hex() if mycelix_did.public_key else "0" * 64

        await self.dht_client.create_did(
            did=did_string,
            controller=controller_hex,
            verification_methods=did_doc.get("verificationMethod", []),
            authentication=did_doc.get("authentication", [])
        )

        # 2. Store identity factors on DHT
        if factors:
            dht_factors = self._convert_factors_to_dht(factors)
            await self.dht_client.store_identity_factors(did_string, dht_factors)

        # 3. Compute initial identity signals
        await self.dht_client.compute_identity_signals(did_string)

    def _convert_factors_to_dht(self, factors: List[IdentityFactor]) -> List[Dict[str, Any]]:
        """Convert local factors to DHT format"""
        dht_factors = []

        for factor in factors:
            # Map factor types
            if isinstance(factor, CryptoKeyFactor):
                factor_type = "CryptoKey"
                category = "PRIMARY"
            elif isinstance(factor, GitcoinPassportFactor):
                factor_type = "GitcoinPassport"
                category = "REPUTATION"
            elif isinstance(factor, SocialRecoveryFactor):
                factor_type = "SocialRecovery"
                category = "SOCIAL"
            elif isinstance(factor, HardwareKeyFactor):
                factor_type = "HardwareKey"
                category = "BACKUP"
            else:
                factor_type = "Unknown"
                category = "PRIMARY"

            dht_factor = {
                "factor_id": getattr(factor, 'factor_id', f"factor_{len(dht_factors)}"),
                "factor_type": factor_type,
                "category": category,
                "status": "ACTIVE" if getattr(factor, 'status', None) == FactorStatus.ACTIVE else "INACTIVE",
                "metadata": json.dumps(getattr(factor, 'metadata', {})),
                "added": int(time.time() * 1_000_000),
                "last_verified": int(time.time() * 1_000_000)
            }
            dht_factors.append(dht_factor)

        return dht_factors

    # ============================================================
    # Identity Query and Verification
    # ============================================================

    async def get_identity(
        self,
        participant_id: Optional[str] = None,
        did: Optional[str] = None,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get complete identity (local + DHT)

        Args:
            participant_id: Local participant ID
            did: DID string
            use_cache: Use cached data if available

        Returns:
            Complete identity dict or None
        """
        # Resolve DID from participant_id if needed
        if participant_id and not did:
            did = self.participant_dids.get(participant_id)

        if not did:
            logger.warning(f"No DID found for participant: {participant_id}")
            return None

        # Check cache
        if use_cache and self.config.cache_dht_queries:
            cached = self._get_cached_identity(did)
            if cached:
                return cached

        # Query DHT if available
        if self.dht_connected:
            try:
                identity = await self.dht_client.get_complete_identity(did)
                self._cache_identity(did, identity)
                return identity
            except Exception as e:
                logger.error(f"DHT query failed for {did}: {e}")

        # Fallback to local data
        local_did = self.did_manager.resolve_did(did)
        if local_did:
            identity = {
                "did": did,
                "did_document": local_did.to_did_document(),
                "complete": False,
                "source": "local"
            }
            return identity

        return None

    async def verify_identity_for_fl(
        self,
        participant_id: str,
        required_assurance: str = "E2"
    ) -> Dict[str, Any]:
        """
        Verify identity meets FL participation requirements

        Args:
            participant_id: Local participant ID
            required_assurance: Minimum assurance level (E0-E4)

        Returns:
            Verification result dict
        """
        did = self.participant_dids.get(participant_id)

        if not did:
            return {
                "verified": False,
                "reasons": [f"No DID registered for participant: {participant_id}"]
            }

        # Query DHT if available
        if self.dht_connected:
            try:
                verification = await self.dht_client.verify_identity_for_fl(did, required_assurance)
                logger.info(f"Identity verification for {participant_id}: {'✅ PASS' if verification['verified'] else '❌ FAIL'}")
                return verification
            except Exception as e:
                logger.error(f"DHT verification failed for {participant_id}: {e}")

        # Fallback to local verification
        return self._verify_identity_local(participant_id, required_assurance)

    def _verify_identity_local(self, participant_id: str, required_assurance: str) -> Dict[str, Any]:
        """Local fallback for identity verification"""
        did = self.participant_dids.get(participant_id)

        if not did:
            return {"verified": False, "reasons": ["No DID found"]}

        # Get local factors
        local_data = self.did_manager.storage.get(did, {})
        factors = local_data.get("factors", [])

        # Calculate local assurance level
        local_assurance = calculate_assurance_level(factors)

        assurance_levels = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 4}
        verified = assurance_levels.get(local_assurance.value, 0) >= assurance_levels.get(required_assurance, 2)

        return {
            "verified": verified,
            "assurance_level": local_assurance.value,
            "sybil_resistance": 0.5,  # Default
            "cartel_risk": 0.0,
            "reasons": ["Verified locally"] if verified else [f"Assurance level {local_assurance.value} < {required_assurance}"],
            "source": "local"
        }

    # ============================================================
    # Reputation Synchronization
    # ============================================================

    async def sync_reputation_to_dht(
        self,
        participant_id: str,
        reputation_score: float,
        score_type: str = "trust",
        metadata: Optional[Dict] = None
    ):
        """
        Sync participant reputation from FL coordinator to DHT

        Args:
            participant_id: Local participant ID
            reputation_score: Reputation score (0.0-1.0)
            score_type: Score type ("trust", "contribution", "verification")
            metadata: Additional metadata
        """
        did = self.participant_dids.get(participant_id)

        if not did or not self.dht_connected:
            return

        try:
            await self.dht_client.store_reputation_entry(
                did=did,
                network_id="zero_trustml",
                reputation_score=reputation_score,
                raw_score=reputation_score * 1000,  # Scale for raw score
                score_type=score_type,
                metadata=json.dumps(metadata or {}),
                issued_at=int(time.time() * 1_000_000),
                issuer="did:mycelix:zero_trustml_coordinator"
            )

            logger.info(f"✅ Reputation synced to DHT: {participant_id} ({reputation_score:.2f})")

        except Exception as e:
            logger.error(f"Failed to sync reputation for {participant_id}: {e}")

    async def sync_all_reputations(self, reputation_scores: Dict[str, float]):
        """
        Batch sync all participant reputations to DHT

        Args:
            reputation_scores: Dict of participant_id -> reputation_score
        """
        if not self.dht_connected:
            return

        for participant_id, score in reputation_scores.items():
            await self.sync_reputation_to_dht(participant_id, score)

    # ============================================================
    # Guardian Management
    # ============================================================

    async def add_guardian(
        self,
        subject_participant_id: str,
        guardian_participant_id: str,
        relationship_type: str = "RECOVERY",
        weight: float = 1.0
    ) -> bool:
        """
        Add guardian relationship on DHT

        Args:
            subject_participant_id: Subject (being guarded)
            guardian_participant_id: Guardian
            relationship_type: Type ("RECOVERY", "ENDORSEMENT", "DELEGATION")
            weight: Guardian influence (0.0-1.0)

        Returns:
            True if successful
        """
        subject_did = self.participant_dids.get(subject_participant_id)
        guardian_did = self.participant_dids.get(guardian_participant_id)

        if not subject_did or not guardian_did:
            logger.error(f"Missing DID for subject or guardian")
            return False

        if not self.dht_connected:
            logger.warning("DHT not connected - cannot add guardian")
            return False

        try:
            await self.dht_client.add_guardian(
                subject_did=subject_did,
                guardian_did=guardian_did,
                relationship_type=relationship_type,
                weight=weight
            )

            logger.info(f"✅ Guardian added: {guardian_participant_id} guards {subject_participant_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add guardian: {e}")
            return False

    async def authorize_recovery(
        self,
        subject_participant_id: str,
        approving_guardian_ids: List[str],
        required_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Check if recovery is authorized by guardians

        Args:
            subject_participant_id: Subject to recover
            approving_guardian_ids: List of approving guardian participant IDs
            required_threshold: Minimum weight needed (e.g., 0.6 = 60%)

        Returns:
            Authorization result dict
        """
        subject_did = self.participant_dids.get(subject_participant_id)

        if not subject_did:
            return {"authorized": False, "reason": "No DID found for subject"}

        # Convert participant IDs to DIDs
        approving_dids = [
            self.participant_dids.get(pid)
            for pid in approving_guardian_ids
            if self.participant_dids.get(pid)
        ]

        if not self.dht_connected:
            return {"authorized": False, "reason": "DHT not connected"}

        try:
            result = await self.dht_client.authorize_recovery(
                subject_did=subject_did,
                approving_guardians=approving_dids,
                required_threshold=required_threshold
            )

            return result

        except Exception as e:
            logger.error(f"Recovery authorization failed: {e}")
            return {"authorized": False, "reason": str(e)}

    # ============================================================
    # Caching
    # ============================================================

    def _get_cached_identity(self, did: str) -> Optional[Dict[str, Any]]:
        """Get cached identity if not expired"""
        if did not in self.identity_cache:
            return None

        cache_time = self.cache_timestamps.get(did, 0)
        if time.time() - cache_time > self.config.cache_ttl:
            # Cache expired
            del self.identity_cache[did]
            del self.cache_timestamps[did]
            return None

        return self.identity_cache[did]

    def _cache_identity(self, did: str, identity: Dict[str, Any]):
        """Cache identity data"""
        self.identity_cache[did] = identity
        self.cache_timestamps[did] = time.time()

    def clear_cache(self, did: Optional[str] = None):
        """Clear cache for specific DID or all"""
        if did:
            self.identity_cache.pop(did, None)
            self.cache_timestamps.pop(did, None)
        else:
            self.identity_cache.clear()
            self.cache_timestamps.clear()

    # ============================================================
    # Utility Methods
    # ============================================================

    def get_participant_did(self, participant_id: str) -> Optional[str]:
        """Get DID for participant ID"""
        return self.participant_dids.get(participant_id)

    def get_participant_id(self, did: str) -> Optional[str]:
        """Get participant ID for DID (reverse lookup)"""
        for participant_id, participant_did in self.participant_dids.items():
            if participant_did == did:
                return participant_id
        return None

    def register_participant_did(self, participant_id: str, did: str):
        """Manually register participant -> DID mapping"""
        self.participant_dids[participant_id] = did
        logger.info(f"Registered: {participant_id} -> {did}")

    async def get_identity_statistics(self) -> Dict[str, Any]:
        """Get identity system statistics"""
        stats = {
            "total_participants": len(self.participant_dids),
            "dht_connected": self.dht_connected,
            "cache_size": len(self.identity_cache),
            "participants": []
        }

        # Query DHT for each participant (if connected)
        if self.dht_connected:
            for participant_id, did in self.participant_dids.items():
                try:
                    identity = await self.dht_client.get_complete_identity(did)
                    signals = identity.get("identity_signals")

                    if signals:
                        stats["participants"].append({
                            "participant_id": participant_id,
                            "did": did,
                            "assurance_level": signals.assurance_level,
                            "sybil_resistance": signals.sybil_resistance,
                            "verified_human": signals.verified_human
                        })
                except Exception as e:
                    logger.debug(f"Could not fetch stats for {participant_id}: {e}")

        return stats
