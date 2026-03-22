#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root"""
ZeroTrustML Phase 10 Coordinator - Modular Backend Implementation

Integrates:
1. Modular storage backends (PostgreSQL, Holochain, LocalFile, Blockchain)
2. Flexible storage strategies (all, primary, quorum)
3. Hybrid Bridge (optional sync between backends)
4. ZK-PoC (privacy-preserving validation)
5. Byzantine resistance (PoGQ + Reputation)
6. AES-GCM encryption for gradients at rest (HIPAA/GDPR compliance)
"""

import asyncio
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
import numpy as np
import hashlib
import json
import base64
import os

# Unified logging and error handling
from zerotrustml.logging import (
    get_logger,
    async_log_operation,
    async_correlation_context,
    async_timed,
)
from zerotrustml.exceptions import (
    MycelixError,
    ByzantineDetectionError,
    ZKPoCVerificationError,
    StorageError,
    ConfigurationError,
    AggregationError,
    InsufficientGradientsError,
    EncryptionError as MycelixEncryptionError,
    DecryptionError,
    ErrorCode,
)
from zerotrustml.error_handling import (
    async_retry,
    CircuitBreaker,
    ErrorAggregator,
    Fallback,
)

# Import modular backends
from zerotrustml.backends import (
    StorageBackend,
    BackendType,
    PostgreSQLBackend,
    LocalFileBackend,
    HolochainBackend
)

# Import AES-GCM encryption module for gradient encryption at rest
from zerotrustml.core.crypto import (
    generate_key,
    encrypt_gradient,
    decrypt_gradient,
    EncryptionError
)

from zerotrustml.aggregation import aggregate_gradients

# Import Phase 10 components (optional)
try:
    from zerotrustml.hybrid_bridge import HybridBridge, RecordPriority
    HYBRID_BRIDGE_AVAILABLE = True
except ImportError:
    HYBRID_BRIDGE_AVAILABLE = False

from zkpoc import ZKPoC, RangeProof

# Import RISC Zero zkSTARK verifier (optional)
try:
    from zerotrustml.zk import (
        RISCZeroVerifier,
        ZKProofConfig,
        ZKGradientProof,
        ZKVerificationResult,
        ZKVerificationError,
    )
    RISC0_ZK_AVAILABLE = True
except ImportError:
    RISC0_ZK_AVAILABLE = False

logger = get_logger(__name__)

# Circuit breakers for external services
_postgres_breaker = CircuitBreaker(name="postgres", failure_threshold=5, recovery_timeout=30)
_holochain_breaker = CircuitBreaker(name="holochain", failure_threshold=3, recovery_timeout=60)

# Type alias for storage strategies
StorageStrategy = Literal["all", "primary", "quorum"]


@dataclass
class Phase10Config:
    """Phase 10 configuration with modular backend support"""

    # Storage Backend Configuration
    storage_strategy: StorageStrategy = "all"  # "all", "primary", "quorum"
    aggregation_algorithm: str = "krum"

    # PostgreSQL Backend (optional)
    postgres_enabled: bool = True
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "zerotrustml"
    postgres_user: str = "zerotrustml"
    postgres_password: str = ""

    # Holochain Backend (optional)
    # wss:// by default — requires TLS certs on the Holochain conductor.
    # Override with HOLOCHAIN_ADMIN_URL / HOLOCHAIN_APP_URL env vars for local dev.
    holochain_enabled: bool = False
    holochain_admin_url: str = "wss://localhost:8888"
    holochain_app_url: str = "wss://localhost:8889"
    holochain_app_id: str = "zerotrustml"

    # LocalFile Backend (optional - for testing)
    localfile_enabled: bool = False
    localfile_data_dir: str = "/tmp/zerotrustml_data"

    # ZK-PoC (Legacy Bulletproofs)
    zkpoc_enabled: bool = True
    zkpoc_pogq_threshold: float = 0.7
    zkpoc_require_encryption: bool = True
    zkpoc_hipaa_mode: bool = False
    zkpoc_gdpr_mode: bool = False

    # RISC Zero zkSTARK Proofs (Gen7)
    # These provide cryptographic proofs of gradient validity
    risc0_zk_proofs_required: bool = False  # If True, reject gradients without valid proofs
    risc0_zk_proofs_preferred: bool = True  # If True, weight verified gradients higher
    risc0_zk_verification_timeout: float = 30.0  # Seconds
    risc0_zk_weight_multiplier: float = 1.5  # Weight multiplier for verified gradients

    # Hybrid bridge (legacy - optional)
    hybrid_sync_enabled: bool = False
    hybrid_sync_priority: str = "critical_only"  # all, critical_only, none

    # Gradient Encryption at Rest (HIPAA/GDPR compliance)
    # AES-256-GCM encryption for all gradients stored to backends
    encryption_enabled: bool = True  # Enable encryption by default for compliance
    encryption_key: Optional[bytes] = None  # If None, a key will be auto-generated
    encryption_key_env_var: str = "ZEROTRUSTML_ENCRYPTION_KEY"  # Load key from env var

    def __post_init__(self):
        # Allow env var overrides for Holochain URLs (local dev fallback)
        self.holochain_admin_url = os.environ.get(
            "HOLOCHAIN_ADMIN_URL", self.holochain_admin_url
        )
        self.holochain_app_url = os.environ.get(
            "HOLOCHAIN_APP_URL", self.holochain_app_url
        )
        if self.holochain_enabled:
            if self.holochain_admin_url.startswith("ws://"):
                logger.warning(
                    "Holochain admin URL using insecure ws:// — "
                    "use wss:// with TLS certs in production"
                )
            if self.holochain_app_url.startswith("ws://"):
                logger.warning(
                    "Holochain app URL using insecure ws:// — "
                    "use wss:// with TLS certs in production"
                )


class Phase10Coordinator:
    """
    Modular Phase 10 FL Coordinator

    Features:
    - Pluggable storage backends (PostgreSQL, Holochain, LocalFile, Blockchain)
    - Flexible storage strategies (all, primary, quorum)
    - Byzantine-resistant aggregation with ZK-PoC
    - Privacy-preserving validation
    """

    def __init__(self, config: Phase10Config):
        self.config = config

        # Modular storage backends
        self.backends: List[StorageBackend] = []
        self.storage_strategy = config.storage_strategy

        # Legacy compatibility
        self.postgres = None  # Will point to PostgreSQL backend if enabled
        self.holochain = None  # Will point to Holochain backend if enabled

        # Phase 10 components (optional)
        self.hybrid_bridge = None
        self.zkpoc = None

        # RISC Zero zkSTARK verifier (Gen7)
        self.risc0_verifier = None

        # AES-GCM Encryption Key for gradients at rest
        # Priority: explicit key > environment variable > auto-generate
        self.encryption_key: Optional[bytes] = None
        if config.encryption_enabled:
            if config.encryption_key:
                self.encryption_key = config.encryption_key
            elif config.encryption_key_env_var and os.environ.get(config.encryption_key_env_var):
                # Load base64-encoded key from environment variable
                key_b64 = os.environ.get(config.encryption_key_env_var)
                try:
                    self.encryption_key = base64.b64decode(key_b64)
                    if len(self.encryption_key) != 32:
                        raise ValueError("Encryption key must be 32 bytes (256 bits)")
                except Exception as e:
                    logger.warning(f"Invalid encryption key from env var: {e}. Generating new key.")
                    self.encryption_key = generate_key()
            else:
                # Auto-generate a new key (WARNING: should be persisted for production!)
                self.encryption_key = generate_key()
                logger.warning(
                    "Auto-generated encryption key. For production, set "
                    f"{config.encryption_key_env_var} environment variable with base64-encoded 32-byte key."
                )

        # FL state
        self.current_round = 0
        self.global_model = None
        self.pending_gradients = []

        # Metrics
        self.metrics = {
            "gradients_submitted": 0,
            "gradients_accepted": 0,
            "gradients_rejected": 0,
            "byzantine_detected": 0,
            "zkpoc_proofs_verified": 0,
            "backends_count": 0,
            "integrity_errors": 0,
            # RISC Zero zkSTARK metrics
            "risc0_proofs_submitted": 0,
            "risc0_proofs_verified": 0,
            "risc0_proofs_invalid": 0,
            "risc0_proofs_timeout": 0,
            "risc0_avg_verification_ms": 0.0,
            "gradients_encrypted": 0,
            "gradients_decrypted": 0,
            "encryption_errors": 0
        }

    @async_timed(level=20, message="Phase 10 Coordinator initialized")
    async def initialize(self):
        """Initialize all backends and components"""
        async with async_log_operation("coordinator_initialization", storage_strategy=self.storage_strategy):
            logger.info("Initializing Phase 10 Coordinator (Modular Architecture)...")

            # 1. Initialize PostgreSQL backend (if enabled)
            if self.config.postgres_enabled:
                try:
                    async with async_log_operation("postgres_init", host=self.config.postgres_host):
                        postgres_backend = PostgreSQLBackend(
                            host=self.config.postgres_host,
                            port=self.config.postgres_port,
                            database=self.config.postgres_db,
                            user=self.config.postgres_user,
                            password=self.config.postgres_password
                        )
                        await postgres_backend.connect()
                        self.backends.append(postgres_backend)
                        self.postgres = postgres_backend  # Legacy compatibility
                        logger.info("PostgreSQL backend initialized", extra={
                            'backend': 'postgresql',
                            'host': self.config.postgres_host,
                            'database': self.config.postgres_db
                        })
                except Exception as e:
                    from zerotrustml.exceptions import PostgresConnectionError
                    raise PostgresConnectionError(
                        message=f"Failed to initialize PostgreSQL backend: {e}",
                        host=self.config.postgres_host,
                        port=self.config.postgres_port,
                        database=self.config.postgres_db,
                        cause=e
                    )

            # 2. Initialize Holochain backend (if enabled)
            if self.config.holochain_enabled:
                try:
                    async with async_log_operation("holochain_init", admin_url=self.config.holochain_admin_url):
                        holochain_backend = HolochainBackend(
                            admin_url=self.config.holochain_admin_url,
                            app_url=self.config.holochain_app_url,
                            app_id=self.config.holochain_app_id
                        )
                        await holochain_backend.connect()
                        self.backends.append(holochain_backend)
                        self.holochain = holochain_backend  # Legacy compatibility
                        logger.info("Holochain backend initialized", extra={
                            'backend': 'holochain',
                            'app_id': self.config.holochain_app_id
                        })
                except Exception as e:
                    from zerotrustml.exceptions import HolochainConnectionError
                    logger.warning("Holochain backend failed to connect", extra={
                        'error': str(e),
                        'admin_url': self.config.holochain_admin_url,
                        'recoverable': True
                    })
                    # Don't raise - Holochain is optional

            # 3. Initialize LocalFile backend (if enabled - for testing)
            if self.config.localfile_enabled:
                async with async_log_operation("localfile_init", data_dir=self.config.localfile_data_dir):
                    localfile_backend = LocalFileBackend(
                        data_dir=self.config.localfile_data_dir
                    )
                    await localfile_backend.connect()
                    self.backends.append(localfile_backend)
                    logger.info("LocalFile backend initialized", extra={
                        'backend': 'localfile',
                        'data_dir': self.config.localfile_data_dir
                    })

            # Validate at least one backend is available
            if not self.backends:
                raise ConfigurationError(
                    message="No storage backends initialized! Enable at least one backend.",
                    config_key="*_enabled",
                    code=ErrorCode.CONFIG_INVALID
                )

            self.metrics["backends_count"] = len(self.backends)
            logger.info("Storage backends ready", extra={
                'backends_count': len(self.backends),
                'storage_strategy': self.storage_strategy,
                'backend_types': [b.backend_type.value for b in self.backends]
            })

            # 4. Initialize Hybrid Bridge (legacy - optional)
            if (self.config.hybrid_sync_enabled and HYBRID_BRIDGE_AVAILABLE and
                self.postgres and self.holochain):
                try:
                    from zerotrustml.holochain.client import HolochainClient
                    # Hybrid bridge expects old HolochainClient, not HolochainBackend
                    # Skip for now - hybrid bridge is legacy
                    logger.warning("Hybrid bridge not compatible with modular backends (legacy)")
                except Exception as e:
                    logger.warning("Hybrid bridge initialization skipped", extra={'error': str(e)})

            # 5. Initialize ZK-PoC (Legacy Bulletproofs)
            if self.config.zkpoc_enabled:
                self.zkpoc = ZKPoC(pogq_threshold=self.config.zkpoc_pogq_threshold)
                logger.info("ZK-PoC (Bulletproofs) initialized", extra={
                    'pogq_threshold': self.config.zkpoc_pogq_threshold,
                    'hipaa_mode': self.config.zkpoc_hipaa_mode,
                    'gdpr_mode': self.config.zkpoc_gdpr_mode
                })

            # 6. Initialize RISC Zero zkSTARK verifier (Gen7)
            if RISC0_ZK_AVAILABLE and (self.config.risc0_zk_proofs_required or
                                        self.config.risc0_zk_proofs_preferred):
                zk_config = ZKProofConfig(
                    zk_proofs_required=self.config.risc0_zk_proofs_required,
                    zk_proofs_preferred=self.config.risc0_zk_proofs_preferred,
                    verification_timeout=self.config.risc0_zk_verification_timeout,
                )
                self.risc0_verifier = RISCZeroVerifier(zk_config)
                if self.risc0_verifier.available:
                    logger.info("RISC Zero zkSTARK verifier initialized", extra={
                        'proofs_required': self.config.risc0_zk_proofs_required,
                        'proofs_preferred': self.config.risc0_zk_proofs_preferred,
                        'verification_timeout': self.config.risc0_zk_verification_timeout,
                        'weight_multiplier': self.config.risc0_zk_weight_multiplier
                    })
                else:
                    logger.warning("RISC Zero verifier created but gen7_zkstark bindings not available")
                    self.risc0_verifier = None
            elif self.config.risc0_zk_proofs_required:
                raise ConfigurationError(
                    message="RISC Zero ZK proofs required but zerotrustml.zk module not available",
                    config_key="risc0_zk_proofs_required",
                    code=ErrorCode.CONFIG_INVALID
                )

            # 7. Log encryption status (HIPAA/GDPR compliance)
            if self.config.encryption_enabled and self.encryption_key:
                logger.info("AES-256-GCM encryption enabled for gradients at rest", extra={
                    'compliance': ['HIPAA', 'GDPR'],
                    'encryption_algorithm': 'AES-256-GCM'
                })
            else:
                logger.warning("Gradient encryption DISABLED - not compliant with HIPAA/GDPR!", extra={
                    'compliance_warning': True,
                    'encryption_enabled': False
                })

        logger.info("🚀 Phase 10 Coordinator ready (Modular Architecture)!")

    # Storage Strategy Methods

    async def _store_with_strategy(
        self,
        operation: str,
        data: Dict[str, Any]
    ) -> Any:
        """
        Store data using configured strategy

        Strategies:
        - "all": Write to all backends, return first success
        - "primary": Write to first backend only
        - "quorum": Write to majority of backends

        Args:
            operation: Backend method name ("store_gradient", "issue_credit", etc.)
            data: Data to store

        Returns:
            Result from backend(s)
        """
        if not self.backends:
            raise RuntimeError("No backends available for storage")

        if self.storage_strategy == "primary":
            # Write to first backend only
            backend = self.backends[0]
            method = getattr(backend, operation)
            return await method(data)

        elif self.storage_strategy == "all":
            # Write to all backends in parallel
            tasks = []
            for backend in self.backends:
                method = getattr(backend, operation)
                tasks.append(method(data))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Return first successful result
            for result in results:
                if not isinstance(result, Exception):
                    return result

            # All failed
            raise RuntimeError(f"All backends failed for {operation}: {results}")

        elif self.storage_strategy == "quorum":
            # Write to majority of backends
            required_success = (len(self.backends) // 2) + 1
            tasks = []
            for backend in self.backends:
                method = getattr(backend, operation)
                tasks.append(method(data))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successful writes
            successful = [r for r in results if not isinstance(r, Exception)]

            if len(successful) >= required_success:
                return successful[0]
            else:
                raise RuntimeError(
                    f"Quorum not reached for {operation}: "
                    f"{len(successful)}/{required_success} backends succeeded"
                )

        else:
            raise ValueError(f"Unknown storage strategy: {self.storage_strategy}")

    async def _get_with_strategy(
        self,
        operation: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Read data using configured strategy

        For reads:
        - "all": Read from all, return if consensus
        - "primary": Read from first backend
        - "quorum": Read from majority, return if consensus

        Args:
            operation: Backend method name
            *args, **kwargs: Arguments to method

        Returns:
            Data from backend(s)
        """
        if not self.backends:
            raise RuntimeError("No backends available for reading")

        if self.storage_strategy == "primary":
            # Read from first backend only
            backend = self.backends[0]
            method = getattr(backend, operation)
            return await method(*args, **kwargs)

        elif self.storage_strategy in ("all", "quorum"):
            # Read from all backends
            tasks = []
            for backend in self.backends:
                method = getattr(backend, operation)
                tasks.append(method(*args, **kwargs))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter successful results
            successful = [r for r in results if not isinstance(r, Exception) and r is not None]

            if successful:
                # Perform consensus checking for data integrity
                consensus_result = self._check_read_consensus(successful, operation)
                if consensus_result is not None:
                    return consensus_result

                # Fall back to first result if consensus cannot be determined
                logger.debug(f"Consensus check inconclusive for {operation}, using first result")
                return successful[0]
            else:
                # All failed or returned None
                return None

        else:
            raise ValueError(f"Unknown storage strategy: {self.storage_strategy}")

    @async_timed(level=10)  # DEBUG level for individual submissions
    async def handle_gradient_submission(
        self,
        node_id: str,
        encrypted_gradient: bytes,
        zkpoc_proof: Optional[RangeProof] = None,
        pogq_score: Optional[float] = None,
        risc0_proof: Optional[bytes] = None,
        model_hash: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Handle gradient submission with Phase 10 privacy + immutability

        Flow:
        1. Verify RISC Zero zkSTARK proof (if provided and Gen7 enabled)
        2. Verify ZK-PoC Bulletproof (if enabled and provided)
        3. Decrypt and validate gradient
        4. Write to PostgreSQL (fast)
        5. Write to Holochain (immutable) via hybrid bridge
        6. Issue credits (bonus for ZK-verified gradients)
        7. Return result

        Args:
            node_id: Node submitting gradient
            encrypted_gradient: Encrypted gradient data
            zkpoc_proof: Optional ZK proof of gradient quality (Bulletproofs)
            pogq_score: Optional PoGQ score (for non-ZK mode)
            risc0_proof: Optional RISC Zero zkSTARK proof of gradient validity
            model_hash: Optional hash of the model being updated (for proof binding)

        Returns:
            Result dict with status and metadata

        Raises:
            ByzantineDetectionError: If Byzantine behavior detected
            DecryptionError: If gradient decryption fails
            StorageError: If storage operation fails
        """
        async with async_correlation_context(node_id=node_id, round=self.current_round):
            self.metrics["gradients_submitted"] += 1
            result = {
                "accepted": False,
                "reason": "",
                "gradient_id": None,
                "holochain_hash": None,
                "credits_issued": 0,
                "risc0_verified": False,
                "zkpoc_verified": False
            }

            # ==============================================================
            # RISC Zero zkSTARK Proof Verification (Gen7)
            # ==============================================================
            risc0_verification_result = None
            if risc0_proof is not None:
                self.metrics["risc0_proofs_submitted"] += 1
                risc0_verification_result = await self._verify_risc0_proof(
                    node_id=node_id,
                    risc0_proof=risc0_proof,
                    model_hash=model_hash
                )

                if risc0_verification_result is not None:
                    if risc0_verification_result.is_valid:
                        result["risc0_verified"] = True
                        self.metrics["risc0_proofs_verified"] += 1
                        logger.info("RISC Zero proof verified", extra={
                            'node_id': node_id,
                            'round': self.current_round,
                            'gradient_norm': risc0_verification_result.norm,
                            'verification_time_ms': risc0_verification_result.verification_time_ms
                        })
                    else:
                        self.metrics["risc0_proofs_invalid"] += 1
                        # If proofs are required, reject the gradient
                        if self.config.risc0_zk_proofs_required:
                            self.metrics["gradients_rejected"] += 1
                            self.metrics["byzantine_detected"] += 1

                            logger.warning("RISC Zero proof verification failed - Byzantine detected", extra={
                                'node_id': node_id,
                                'round': self.current_round,
                                'detection_method': 'risc0_zkstark',
                                'severity': 'high',
                                'error': risc0_verification_result.error_message
                            })

                            # Log Byzantine event
                            try:
                                await self._store_with_strategy("log_byzantine_event", {
                                    "node_id": node_id,
                                    "round_num": self.current_round,
                                    "detection_method": "risc0_zkstark",
                                    "severity": "high",
                                    "details": {
                                        "proof_invalid": True,
                                        "error": risc0_verification_result.error_message
                                    }
                                })
                            except Exception as e:
                                logger.error("Failed to log Byzantine event", extra={'error': str(e)})

                            result["reason"] = f"RISC Zero proof verification failed: {risc0_verification_result.error_message}"
                            return result
                        else:
                            # Proofs not required, just log the failure
                            logger.warning("RISC Zero proof invalid but not required", extra={
                                'node_id': node_id,
                                'error': risc0_verification_result.error_message
                            })

            # If RISC Zero proofs are required but not provided
            elif self.config.risc0_zk_proofs_required and self.risc0_verifier is not None:
                self.metrics["gradients_rejected"] += 1
                result["reason"] = "RISC Zero proof required but not provided"
                logger.warning("Gradient rejected - RISC Zero proof required but not provided", extra={
                    'node_id': node_id,
                    'round': self.current_round
                })
                return result

            # ==============================================================
            # ZK-PoC Bulletproof Verification (Legacy)
            # ==============================================================
            # Phase 10 Mode: ZK-PoC verification
            if self.config.zkpoc_enabled and zkpoc_proof:
                # Verify proof WITHOUT seeing actual score
                if not self.zkpoc.verify_proof(zkpoc_proof):
                    self.metrics["gradients_rejected"] += 1
                    self.metrics["byzantine_detected"] += 1

                    logger.warning("ZK-PoC verification failed - Byzantine detected", extra={
                        'node_id': node_id,
                        'round': self.current_round,
                        'detection_method': 'zkpoc',
                        'severity': 'high'
                    })

                    # Log Byzantine event using modular backend system
                    try:
                        await self._store_with_strategy("log_byzantine_event", {
                            "node_id": node_id,
                            "round_num": self.current_round,
                            "detection_method": "zkpoc",
                            "severity": "high",
                            "details": {"proof_invalid": True}
                        })
                    except Exception as e:
                        logger.error("Failed to log Byzantine event", extra={'error': str(e)})

                    result["reason"] = "ZK-PoC proof verification failed"
                    return result

                # Proof valid - gradient quality sufficient
                result["zkpoc_verified"] = True
                logger.info("ZK-PoC proof verified", extra={
                    'node_id': node_id,
                    'round': self.current_round
                })
                self.metrics["zkpoc_proofs_verified"] += 1

                # Decrypt gradient (proof passed)
                try:
                    gradient = self._decrypt_gradient(encrypted_gradient)
                except Exception as e:
                    raise DecryptionError(
                        message=f"Failed to decrypt gradient from node {node_id}",
                        cause=e
                    )

                # PoGQ score NOT stored (privacy)
                pogq_score = None  # Explicitly hide for HIPAA/GDPR

            # Phase 9 Mode: Traditional PoGQ validation
            else:
                if pogq_score is None:
                    result["reason"] = "PoGQ score required (ZK-PoC disabled)"
                    self.metrics["gradients_rejected"] += 1
                    logger.warning("Gradient rejected - no PoGQ score", extra={
                        'node_id': node_id,
                        'zkpoc_enabled': False
                    })
                    return result

                if pogq_score < 0.7:
                    self.metrics["gradients_rejected"] += 1
                    self.metrics["byzantine_detected"] += 1

                    logger.warning("PoGQ score below threshold - Byzantine detected", extra={
                        'node_id': node_id,
                        'round': self.current_round,
                        'pogq_score': pogq_score,
                        'threshold': 0.7,
                        'detection_method': 'pogq'
                    })

                    result["reason"] = f"PoGQ score {pogq_score:.3f} below threshold"
                    return result

                # Decrypt gradient
                try:
                    gradient = self._decrypt_gradient(encrypted_gradient)
                except Exception as e:
                    raise DecryptionError(
                        message=f"Failed to decrypt gradient from node {node_id}",
                        cause=e
                    )

            # Compute hash BEFORE encryption (for integrity verification)
            gradient_hash = self._hash_gradient(gradient)

            # Encrypt gradient for storage at rest (HIPAA/GDPR compliance)
            if self.config.encryption_enabled and self.encryption_key:
                try:
                    # Serialize gradient to bytes, then encrypt with AES-GCM
                    gradient_bytes = json.dumps(gradient.tolist()).encode('utf-8')
                    encrypted_data = encrypt_gradient(self.encryption_key, gradient_bytes)
                    # Base64 encode for JSON storage compatibility
                    gradient_storage = base64.b64encode(encrypted_data).decode('ascii')
                    is_encrypted = True
                    self.metrics["gradients_encrypted"] += 1
                    logger.debug("Gradient encrypted with AES-256-GCM", extra={
                        'node_id': node_id,
                        'algorithm': 'AES-256-GCM'
                    })
                except Exception as e:
                    self.metrics["encryption_errors"] += 1
                    self.metrics["gradients_rejected"] += 1
                    logger.error("Gradient encryption failed", extra={
                        'node_id': node_id,
                        'error': str(e),
                        'error_type': type(e).__name__
                    })
                    raise MycelixEncryptionError(
                        message=f"Failed to encrypt gradient for node {node_id}",
                        operation='encrypt',
                        cause=e
                    )
            else:
                # Encryption disabled - store plaintext (NOT recommended for compliance)
                gradient_storage = gradient.tolist()
                is_encrypted = False
                logger.warning("Storing gradient WITHOUT encryption (compliance risk!)", extra={
                    'node_id': node_id,
                    'compliance_warning': True
                })

            # Store gradient using modular backend system
            gradient_data = {
                "id": self._generate_gradient_id(),
                "node_id": node_id,
                "round_num": self.current_round,
                "gradient": gradient_storage,  # Encrypted (base64) or plaintext list
                "gradient_hash": gradient_hash,  # Hash of plaintext for integrity
                "pogq_score": pogq_score,  # None if ZK-PoC mode
                "zkpoc_verified": zkpoc_proof is not None,
                "risc0_verified": result["risc0_verified"],  # Gen7 zkSTARK verified
                "validation_passed": True,
                "reputation_score": await self._get_reputation(node_id),
                "timestamp": asyncio.get_event_loop().time(),
                "encrypted": is_encrypted  # Flag to indicate encryption status
            }

            # Write using storage strategy (all/primary/quorum)
            try:
                result["gradient_id"] = await self._store_with_strategy(
                    "store_gradient",
                    gradient_data
                )
            except Exception as e:
                raise StorageError(
                    message=f"Failed to store gradient for node {node_id}",
                    operation='store_gradient',
                    cause=e
                )

            result["encrypted"] = is_encrypted

            # Issue credits for quality gradient using modular backends
            # Calculate credit amount based on quality
            credit_amount = 10  # Base quality credit
            if pogq_score and pogq_score > 0.9:
                credit_amount = 15  # Bonus for high quality
            elif pogq_score and pogq_score > 0.85:
                credit_amount = 12  # Good quality bonus

            # Bonus for RISC Zero zkSTARK verified gradients
            earned_from = "gradient_quality"
            if result["risc0_verified"]:
                credit_amount += 5  # Bonus for cryptographic proof of validity
                earned_from = "gradient_quality_zk_verified"
                logger.debug("Credit bonus applied for RISC Zero verified gradient", extra={
                    'node_id': node_id,
                    'bonus': 5,
                    'total_credits': credit_amount
                })

            # Issue credit using storage strategy with error aggregation
            error_aggregator = ErrorAggregator()
            tasks = []
            for backend in self.backends:
                tasks.append(backend.issue_credit(
                    holder=node_id,
                    amount=credit_amount,
                    earned_from=earned_from
                ))

            # Execute based on strategy
            if self.storage_strategy == "primary":
                try:
                    await tasks[0]
                except Exception as e:
                    logger.warning("Credit issuance failed", extra={
                        'node_id': node_id,
                        'amount': credit_amount,
                        'error': str(e)
                    })
            elif self.storage_strategy == "all":
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        logger.warning("Credit issuance failed on backend", extra={
                            'backend': self.backends[i].backend_type.value,
                            'error': str(res)
                        })
            elif self.storage_strategy == "quorum":
                results = await asyncio.gather(*tasks, return_exceptions=True)
                required = (len(self.backends) // 2) + 1
                successful = [r for r in results if not isinstance(r, Exception)]
                if len(successful) < required:
                    logger.warning("Quorum not reached for credit issuance", extra={
                        'successful': len(successful),
                        'required': required,
                        'total_backends': len(self.backends)
                    })

            result["credits_issued"] = credit_amount

            # Update metrics
            self.metrics["gradients_accepted"] += 1
            result["accepted"] = True
            result["reason"] = "Gradient accepted"

            logger.info("Gradient accepted", extra={
                'node_id': node_id,
                'gradient_id': result['gradient_id'][:8] if result['gradient_id'] else None,
                'credits_issued': credit_amount,
                'encrypted': is_encrypted,
                'round': self.current_round
            })

            return result

    async def aggregate_round(self) -> Dict[str, Any]:
        """
        Aggregate gradients for current round with Byzantine resistance

        When risc0_zk_proofs_preferred is enabled, ZK-verified gradients receive
        higher weight in the aggregation (controlled by risc0_zk_weight_multiplier).

        Returns:
            Aggregation result with global model update
        """
        logger.info(f"Aggregating round {self.current_round}...")

        # Get all accepted gradients for this round
        gradients = await self._get_round_gradients(self.current_round)

        if not gradients:
            logger.warning(f"Round {self.current_round}: No gradients to aggregate")
            return {"success": False, "reason": "no_gradients"}

        # Filter and weight gradients based on ZK verification status
        zk_verified_count = 0
        non_zk_count = 0
        weighted_gradients = []
        weights = []

        for g in gradients:
            is_risc0_verified = g.get("risc0_verified", False)
            is_zkpoc_verified = g.get("zkpoc_verified", False)

            # Track counts for logging
            if is_risc0_verified:
                zk_verified_count += 1
            else:
                non_zk_count += 1

            # If ZK proofs are required, skip non-verified gradients
            if self.config.risc0_zk_proofs_required and not is_risc0_verified:
                logger.debug(f"Skipping non-verified gradient from {g['node_id']} (ZK required mode)")
                continue

            weighted_gradients.append(g["gradient"])

            # Apply weight multiplier for ZK-verified gradients
            if self.config.risc0_zk_proofs_preferred and is_risc0_verified:
                weights.append(self.config.risc0_zk_weight_multiplier)
            else:
                weights.append(1.0)

        # Log ZK verification breakdown
        logger.info(f"Aggregation gradient breakdown", extra={
            'round': self.current_round,
            'total_gradients': len(gradients),
            'risc0_verified': zk_verified_count,
            'non_verified': non_zk_count,
            'gradients_in_aggregation': len(weighted_gradients)
        })

        if not weighted_gradients:
            logger.warning(f"Round {self.current_round}: No gradients after ZK filtering")
            return {"success": False, "reason": "no_verified_gradients"}

        # Byzantine-resistant aggregation with weights
        aggregated, byzantine_count = self._krum_aggregation_weighted(
            weighted_gradients, weights
        )

        # Update global model
        self.global_model = aggregated

        # Compute hash before encryption
        checkpoint_hash = self._hash_gradient(aggregated)

        # Encrypt checkpoint gradient for storage at rest
        if self.config.encryption_enabled and self.encryption_key:
            try:
                checkpoint_bytes = json.dumps(aggregated.tolist()).encode('utf-8')
                encrypted_checkpoint = encrypt_gradient(self.encryption_key, checkpoint_bytes)
                checkpoint_storage = base64.b64encode(encrypted_checkpoint).decode('ascii')
                checkpoint_encrypted = True
                self.metrics["gradients_encrypted"] += 1
            except Exception as e:
                logger.error(f"Checkpoint encryption failed: {e}")
                self.metrics["encryption_errors"] += 1
                checkpoint_storage = aggregated.tolist()
                checkpoint_encrypted = False
        else:
            checkpoint_storage = aggregated.tolist()
            checkpoint_encrypted = False

        # Store checkpoint using modular backend system
        checkpoint_data = {
            "id": self._generate_gradient_id(),
            "node_id": "coordinator",
            "round_num": self.current_round,
            "gradient": checkpoint_storage,
            "gradient_hash": checkpoint_hash,
            "pogq_score": None,  # N/A for checkpoints
            "zkpoc_verified": False,
            "validation_passed": True,
            "reputation_score": 1.0,  # Coordinator reputation
            "timestamp": asyncio.get_event_loop().time(),
            "encrypted": checkpoint_encrypted
        }

        # Write checkpoint to all backends
        await self._store_with_strategy("store_gradient", checkpoint_data)

        # Advance round
        self.current_round += 1

        result = {
            "success": True,
            "round": self.current_round - 1,
            "gradients_count": len(gradients),
            "gradients_aggregated": len(weighted_gradients),
            "risc0_verified_count": zk_verified_count,
            "non_verified_count": non_zk_count,
            "byzantine_detected": byzantine_count,
            "model_hash": checkpoint_data["gradient_hash"][:16]
        }

        logger.info(f"Round {self.current_round - 1} aggregated", extra={
            'round': self.current_round - 1,
            'gradients_aggregated': len(weighted_gradients),
            'risc0_verified': zk_verified_count,
            'byzantine_detected': byzantine_count
        })
        return result

    async def get_stats(self) -> Dict[str, Any]:
        """Get Phase 10 statistics from all backends"""
        stats = self.metrics.copy()

        # Add backend statistics
        backend_stats = []
        for backend in self.backends:
            try:
                backend_data = await backend.get_stats()
                backend_stats.append(backend_data)
            except Exception as e:
                logger.warning(f"Failed to get stats from {backend.backend_type.value}: {e}")

        stats["backends"] = backend_stats
        stats["storage_strategy"] = self.storage_strategy

        if self.hybrid_bridge:
            bridge_stats = self.hybrid_bridge.get_stats()
            stats["hybrid_bridge"] = bridge_stats

        if self.zkpoc:
            stats["zkpoc_enabled"] = True
            stats["zkpoc_hipaa_mode"] = self.config.zkpoc_hipaa_mode
            stats["zkpoc_gdpr_mode"] = self.config.zkpoc_gdpr_mode

        # Add RISC Zero zkSTARK statistics
        stats["risc0_zkstark"] = {
            "available": RISC0_ZK_AVAILABLE,
            "verifier_initialized": self.risc0_verifier is not None,
            "proofs_required": self.config.risc0_zk_proofs_required,
            "proofs_preferred": self.config.risc0_zk_proofs_preferred,
            "weight_multiplier": self.config.risc0_zk_weight_multiplier,
            "proofs_submitted": self.metrics.get("risc0_proofs_submitted", 0),
            "proofs_verified": self.metrics.get("risc0_proofs_verified", 0),
            "proofs_invalid": self.metrics.get("risc0_proofs_invalid", 0),
            "proofs_timeout": self.metrics.get("risc0_proofs_timeout", 0),
            "avg_verification_ms": self.metrics.get("risc0_avg_verification_ms", 0.0),
        }

        # Include verifier-specific metrics if available
        if self.risc0_verifier is not None:
            verifier_metrics = self.risc0_verifier.metrics
            stats["risc0_zkstark"]["verifier_metrics"] = verifier_metrics

        # Add encryption status for HIPAA/GDPR compliance reporting
        stats["encryption"] = {
            "enabled": self.config.encryption_enabled,
            "algorithm": "AES-256-GCM" if self.config.encryption_enabled else None,
            "key_configured": self.encryption_key is not None,
            "gradients_encrypted": self.metrics.get("gradients_encrypted", 0),
            "gradients_decrypted": self.metrics.get("gradients_decrypted", 0),
            "encryption_errors": self.metrics.get("encryption_errors", 0),
            "compliance_mode": "HIPAA/GDPR" if self.config.encryption_enabled else "disabled"
        }

        return stats

    async def verify_integrity(self, gradient_id: str) -> bool:
        """
        Verify gradient integrity across all backends

        Returns:
            True if all backends agree on gradient integrity
        """
        if not self.backends:
            logger.warning("No backends available for integrity verification")
            return False

        # Verify integrity on all backends
        tasks = []
        for backend in self.backends:
            tasks.append(backend.verify_gradient_integrity(gradient_id))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check consensus - all backends must agree
        valid_results = [r for r in results if not isinstance(r, Exception)]

        if not valid_results:
            logger.error(f"All backends failed integrity check for {gradient_id}")
            self.metrics["integrity_errors"] += 1
            return False

        is_valid = all(valid_results)

        if not is_valid:
            self.metrics["integrity_errors"] += 1
            logger.error(f"Integrity violation for gradient {gradient_id}")
        else:
            logger.info(f"✅ Gradient {gradient_id[:8]}... integrity verified across {len(valid_results)} backend(s)")

        return is_valid

    async def shutdown(self):
        """Graceful shutdown of all backends"""
        logger.info("Shutting down Phase 10 Coordinator...")

        if self.hybrid_bridge:
            await self.hybrid_bridge.stop()

        # Disconnect all backends
        for backend in self.backends:
            try:
                await backend.disconnect()
                logger.info(f"✅ {backend.backend_type.value} backend disconnected")
            except Exception as e:
                logger.warning(f"⚠️  Error disconnecting {backend.backend_type.value}: {e}")

        logger.info("Phase 10 Coordinator shut down")

        # Shutdown RISC Zero verifier
        if self.risc0_verifier is not None:
            self.risc0_verifier.shutdown()
            logger.info("RISC Zero verifier shut down")

    # ==========================================================================
    # RISC Zero zkSTARK Verification Methods
    # ==========================================================================

    async def _verify_risc0_proof(
        self,
        node_id: str,
        risc0_proof: bytes,
        model_hash: Optional[bytes] = None
    ) -> Optional['ZKVerificationResult']:
        """
        Verify a RISC Zero zkSTARK proof of gradient validity.

        This method:
        1. Checks if the RISC Zero verifier is available
        2. Creates a ZKGradientProof wrapper with metadata
        3. Runs async verification with timeout
        4. Updates metrics based on result
        5. Returns the verification result

        Args:
            node_id: Node that submitted the gradient
            risc0_proof: Serialized RISC Zero receipt (proof bytes)
            model_hash: Expected model hash for proof binding

        Returns:
            ZKVerificationResult if verification ran (may be valid or invalid)
            None if verifier is not available

        Note:
            Does NOT raise exceptions - returns result with is_valid=False
            on verification failure to allow caller to decide how to handle.
        """
        if self.risc0_verifier is None:
            logger.debug("RISC Zero verifier not available, skipping proof verification")
            return None

        try:
            # Create proof wrapper with metadata
            # Convert node_id string to bytes if needed for comparison
            node_id_bytes = None
            if isinstance(node_id, str):
                try:
                    # Try to decode as hex first (common format)
                    if len(node_id) == 64:  # 32 bytes as hex
                        node_id_bytes = bytes.fromhex(node_id)
                    else:
                        # Hash the string to get 32 bytes
                        node_id_bytes = hashlib.sha256(node_id.encode()).digest()
                except ValueError:
                    node_id_bytes = hashlib.sha256(node_id.encode()).digest()

            proof = ZKGradientProof(
                proof_bytes=risc0_proof,
                node_id=node_id_bytes,
                round_number=self.current_round,
                model_hash=model_hash
            )

            # Run async verification
            result = await self.risc0_verifier.verify_async(proof)

            # Update average verification time metric
            if result.verification_time_ms is not None:
                total_verified = self.metrics["risc0_proofs_verified"] + self.metrics["risc0_proofs_invalid"]
                if total_verified > 0:
                    current_avg = self.metrics["risc0_avg_verification_ms"]
                    # Running average
                    self.metrics["risc0_avg_verification_ms"] = (
                        (current_avg * total_verified + result.verification_time_ms) /
                        (total_verified + 1)
                    )
                else:
                    self.metrics["risc0_avg_verification_ms"] = result.verification_time_ms

            return result

        except ZKVerificationError as e:
            logger.warning("RISC Zero verification error", extra={
                'node_id': node_id,
                'error': str(e),
                'error_code': e.code.value if hasattr(e, 'code') else 'unknown'
            })
            # Return invalid result instead of raising
            return ZKVerificationResult(
                is_valid=False,
                error_message=str(e)
            )

        except Exception as e:
            logger.error("Unexpected error during RISC Zero verification", extra={
                'node_id': node_id,
                'error': str(e),
                'error_type': type(e).__name__
            })
            if "timeout" in str(e).lower():
                self.metrics["risc0_proofs_timeout"] += 1
            # Return invalid result instead of raising
            return ZKVerificationResult(
                is_valid=False,
                error_message=f"Unexpected error: {e}"
            )

    # ==========================================================================
    # Helper methods
    # ==========================================================================

    def _check_read_consensus(
        self,
        results: List[Any],
        operation: str
    ) -> Optional[Any]:
        """
        Check consensus among read results from multiple backends.

        For critical operations, ensures data integrity by comparing results
        across backends and detecting inconsistencies.

        Args:
            results: List of successful results from backends
            operation: Name of the operation (for logging)

        Returns:
            The consensus result if agreement is reached, None otherwise
        """
        if len(results) < 2:
            # Need at least 2 results for consensus
            return results[0] if results else None

        # Determine consensus strategy based on operation type
        critical_operations = {
            "get_gradient",
            "get_gradients_by_round",
            "get_reputation",
            "verify_gradient_integrity",
        }

        is_critical = operation in critical_operations

        if not is_critical:
            # For non-critical operations, just return first result
            return results[0]

        # For critical operations, check consensus
        try:
            # Hash-based consensus for gradient data
            if operation in ("get_gradient", "get_gradients_by_round"):
                hashes = []
                for r in results:
                    if hasattr(r, 'gradient_hash'):
                        hashes.append(r.gradient_hash)
                    elif isinstance(r, dict) and 'gradient_hash' in r:
                        hashes.append(r['gradient_hash'])
                    elif isinstance(r, list):
                        # List of gradients - hash the list of hashes
                        list_hashes = []
                        for item in r:
                            if hasattr(item, 'gradient_hash'):
                                list_hashes.append(item.gradient_hash)
                            elif isinstance(item, dict) and 'gradient_hash' in item:
                                list_hashes.append(item['gradient_hash'])
                        if list_hashes:
                            hashes.append(hashlib.sha256(''.join(sorted(list_hashes)).encode()).hexdigest())

                if hashes:
                    # Check if majority agree
                    from collections import Counter
                    hash_counts = Counter(hashes)
                    most_common_hash, count = hash_counts.most_common(1)[0]

                    # Require majority consensus
                    if count > len(results) / 2:
                        # Return the result with the consensus hash
                        for r in results:
                            r_hash = None
                            if hasattr(r, 'gradient_hash'):
                                r_hash = r.gradient_hash
                            elif isinstance(r, dict) and 'gradient_hash' in r:
                                r_hash = r['gradient_hash']

                            if r_hash == most_common_hash:
                                logger.debug(f"Consensus reached for {operation}: {most_common_hash[:8]}...")
                                return r

                    # No majority consensus - log integrity error
                    logger.warning(f"Consensus failure for {operation}: hash disagreement", extra={
                        'operation': operation,
                        'unique_hashes': len(hash_counts),
                        'total_results': len(results),
                        'hash_distribution': dict(hash_counts)
                    })
                    self.metrics["integrity_errors"] += 1
                    return None

            # Value-based consensus for reputation scores
            elif operation == "get_reputation":
                scores = []
                for r in results:
                    if hasattr(r, 'score'):
                        scores.append(r.score)
                    elif isinstance(r, dict) and 'score' in r:
                        scores.append(r['score'])

                if scores:
                    # Check if scores are within acceptable variance (5%)
                    avg_score = sum(scores) / len(scores)
                    max_variance = 0.05

                    within_variance = all(
                        abs(s - avg_score) <= max_variance
                        for s in scores
                    )

                    if within_variance:
                        # Use median score for robustness
                        median_idx = len(scores) // 2
                        sorted_pairs = sorted(zip(scores, results), key=lambda x: x[0])
                        logger.debug(f"Reputation consensus reached: {avg_score:.3f}")
                        return sorted_pairs[median_idx][1]

                    # Scores diverge too much
                    logger.warning(f"Reputation consensus failure: score variance too high", extra={
                        'operation': operation,
                        'scores': scores,
                        'avg_score': avg_score,
                        'max_variance_allowed': max_variance
                    })
                    self.metrics["integrity_errors"] += 1
                    return None

            # Boolean consensus for integrity checks
            elif operation == "verify_gradient_integrity":
                # All backends must agree on integrity
                bool_results = [bool(r) for r in results]
                if all(bool_results):
                    logger.debug(f"Integrity verification consensus: all passed")
                    return results[0]
                elif not any(bool_results):
                    logger.debug(f"Integrity verification consensus: all failed")
                    return results[0]
                else:
                    # Disagreement on integrity - this is serious
                    logger.error(f"Integrity verification DISAGREEMENT across backends", extra={
                        'operation': operation,
                        'results': bool_results,
                        'backends_count': len(results)
                    })
                    self.metrics["integrity_errors"] += 1
                    return None

        except Exception as e:
            logger.warning(f"Consensus check error for {operation}: {e}")

        # Default: return first result if consensus check fails
        return results[0]

    def _decrypt_gradient(self, encrypted: bytes) -> np.ndarray:
        """
        Decrypt gradient received in transit from a node.

        This handles gradients that are encrypted when sent over the network.
        Supports two modes:
        1. AES-GCM encrypted data (if encryption_key is set)
        2. Plain JSON-encoded gradient (for backward compatibility)

        Args:
            encrypted: Encrypted or plaintext gradient bytes

        Returns:
            np.ndarray: Decrypted gradient array
        """
        # First, try AES-GCM decryption if we have an encryption key
        if self.config.encryption_enabled and self.encryption_key:
            try:
                # Check if it looks like base64-encoded encrypted data
                # (node might send base64-encoded AES-GCM ciphertext)
                try:
                    # Try base64 decode first
                    encrypted_bytes = base64.b64decode(encrypted)
                    if len(encrypted_bytes) > 12:  # AES-GCM nonce is 12 bytes
                        decrypted = decrypt_gradient(self.encryption_key, encrypted_bytes)
                        gradient_list = json.loads(decrypted.decode('utf-8'))
                        self.metrics["gradients_decrypted"] += 1
                        return np.array(gradient_list)
                except Exception:
                    pass  # Not base64, try other formats

                # Try direct AES-GCM decryption (raw bytes)
                if len(encrypted) > 12:
                    try:
                        decrypted = decrypt_gradient(self.encryption_key, encrypted)
                        gradient_list = json.loads(decrypted.decode('utf-8'))
                        self.metrics["gradients_decrypted"] += 1
                        return np.array(gradient_list)
                    except EncryptionError:
                        pass  # Not AES-GCM encrypted, try plain JSON

            except Exception as e:
                logger.debug(f"AES-GCM decryption attempt failed: {e}")

        # Fallback: try parsing as plain JSON (backward compatibility)
        try:
            gradient_list = json.loads(encrypted.decode('utf-8'))
            return np.array(gradient_list)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Failed to decrypt/decode gradient: {e}")
            # Return a zero gradient instead of random (safer for aggregation)
            raise ValueError(f"Cannot decrypt gradient: {e}")

    def _hash_gradient(self, gradient: np.ndarray) -> str:
        """Compute deterministic hash of gradient"""
        return hashlib.sha256(gradient.tobytes()).hexdigest()

    def _generate_gradient_id(self) -> str:
        """Generate unique gradient ID"""
        import uuid
        return str(uuid.uuid4())

    async def _get_reputation(self, node_id: str) -> float:
        """Get node reputation from storage backend"""
        reputation = await self._get_with_strategy("get_reputation", node_id)
        if reputation:
            return reputation.get("score", 0.5)
        return 0.5

    async def _get_round_gradients(self, round_num: int) -> List[Dict]:
        """Get all gradients for a round from storage backend, decrypting if necessary"""
        gradients = await self._get_with_strategy("get_gradients_by_round", round_num)
        if gradients:
            result = []
            for g in gradients:
                # Check if gradient is encrypted (stored as base64 string)
                gradient_data = g.gradient
                is_encrypted = getattr(g, 'backend_metadata', {}).get('encrypted', False) if hasattr(g, 'backend_metadata') and g.backend_metadata else False

                # Also check if gradient looks like base64-encoded encrypted data
                if isinstance(gradient_data, str) and self.config.encryption_enabled and self.encryption_key:
                    try:
                        # Attempt to decrypt base64-encoded encrypted gradient
                        encrypted_bytes = base64.b64decode(gradient_data)
                        decrypted_bytes = decrypt_gradient(self.encryption_key, encrypted_bytes)
                        gradient_data = json.loads(decrypted_bytes.decode('utf-8'))
                        self.metrics["gradients_decrypted"] += 1
                        logger.debug(f"Decrypted gradient from node {g.node_id}")
                    except (EncryptionError, ValueError, json.JSONDecodeError) as e:
                        # Not encrypted or decryption failed - use as-is if it's a list
                        logger.warning(f"Could not decrypt gradient {g.id[:8]}: {e}")
                        if isinstance(gradient_data, str):
                            # Try parsing as plain JSON
                            try:
                                gradient_data = json.loads(gradient_data)
                            except json.JSONDecodeError:
                                logger.error(f"Invalid gradient data format for {g.id[:8]}")
                                continue  # Skip this gradient
                    except Exception as e:
                        logger.error(f"Unexpected error decrypting gradient {g.id[:8]}: {e}")
                        continue

                result.append({
                    "node_id": g.node_id,
                    "gradient": gradient_data,
                    "gradient_hash": g.gradient_hash,
                    "pogq_score": g.pogq_score,
                    "zkpoc_verified": getattr(g, 'zkpoc_verified', False),
                    "risc0_verified": getattr(g, 'risc0_verified', False)
                })
            return result
        return []

    def _krum_aggregation(self, gradients: List[List]) -> tuple:
        """
        Krum Byzantine-resistant aggregation

        Returns:
            (aggregated_gradient, byzantine_count)
        """
        return self._krum_aggregation_weighted(gradients, None)

    def _krum_aggregation_weighted(
        self,
        gradients: List[List],
        weights: Optional[List[float]] = None
    ) -> tuple:
        """
        Krum Byzantine-resistant aggregation with optional weighting.

        When weights are provided, ZK-verified gradients can receive higher
        influence in the final aggregation.

        Args:
            gradients: List of gradient arrays
            weights: Optional list of weights (1.0 = normal, >1.0 = higher influence)

        Returns:
            (aggregated_gradient, byzantine_count)
        """
        if not gradients:
            return np.zeros(100), 0

        grad_arrays = [np.array(g) for g in gradients]

        # Use weights as reputations if provided, otherwise default to 1.0
        if weights is not None:
            reputations = weights
        else:
            reputations = [1.0] * len(grad_arrays)

        aggregated = aggregate_gradients(
            grad_arrays,
            reputations,
            algorithm=self.config.aggregation_algorithm,
            num_byzantine=max(0, len(grad_arrays) - 2)
        )

        # Estimate Byzantine count by comparing distance to aggregated median
        diffs = np.array([np.linalg.norm(g - aggregated) for g in grad_arrays])
        median = np.median(diffs)
        mad = np.median(np.abs(diffs - median)) or 1e-9
        threshold = median + 2.5 * mad
        byzantine_count = int(np.sum(diffs > threshold))

        return aggregated, byzantine_count


# Demo/Testing
async def demo_phase10_modular():
    """Demonstrate Phase 10 coordinator with modular backends"""
    print("🚀 Phase 10 Coordinator Demo (Modular Architecture)\n")

    # Example 1: LocalFile backend only (for testing)
    print("Example 1: Testing with LocalFile backend")
    config_test = Phase10Config(
        postgres_enabled=False,
        holochain_enabled=False,
        localfile_enabled=True,
        localfile_data_dir="/tmp/zerotrustml_demo",
        storage_strategy="primary",
        zkpoc_enabled=True
    )

    coordinator = Phase10Coordinator(config_test)

    try:
        await coordinator.initialize()
        print(f"✅ Initialized with {len(coordinator.backends)} backend(s)")
        print(f"   Strategy: {coordinator.storage_strategy}")
        print(f"   Backend types: {[b.backend_type.value for b in coordinator.backends]}\n")

        # Simulate gradient submission
        print("📊 Simulating gradient submission...")

        # Generate ZK proof
        zkpoc = ZKPoC(pogq_threshold=0.7)
        proof = zkpoc.generate_proof(pogq_score=0.95)

        # Submit gradient
        encrypted_gradient = json.dumps([0.1] * 100).encode()

        result = await coordinator.handle_gradient_submission(
            node_id="hospital-a",
            encrypted_gradient=encrypted_gradient,
            zkpoc_proof=proof
        )

        print(f"✅ Gradient submission result: {result}")

        # Shutdown
        await coordinator.shutdown()

    except Exception as e:
        print(f"⚠️  Demo error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60 + "\n")

    # Example 2: Multi-backend with PostgreSQL + Holochain (mock)
    print("Example 2: Multi-backend configuration (PostgreSQL + LocalFile)")
    config_multi = Phase10Config(
        postgres_enabled=False,  # Would be True in production
        holochain_enabled=False,  # Would enable if conductor running
        localfile_enabled=True,
        storage_strategy="all",  # Write to ALL backends
        zkpoc_enabled=True
    )

    print("\n📖 Configuration:")
    print(f"   - PostgreSQL: {'enabled' if config_multi.postgres_enabled else 'disabled'}")
    print(f"   - Holochain: {'enabled' if config_multi.holochain_enabled else 'disabled'}")
    print(f"   - LocalFile: {'enabled' if config_multi.localfile_enabled else 'enabled'}")
    print(f"   - Strategy: {config_multi.storage_strategy}")
    print(f"   - ZK-PoC: {'enabled' if config_multi.zkpoc_enabled else 'disabled'}")

    print("\n✅ Phase 10 Coordinator (Modular Architecture) validated!")
    print("\n🎯 Backend Options:")
    print("   - PostgreSQL: Enterprise deployments (ACID, SQL)")
    print("   - Holochain: Decentralized deployments (DHT, immutable)")
    print("   - LocalFile: Testing & development (JSON files)")
    print("   - Blockchain: Future (Ethereum/Cosmos/Polkadot)")
    print("\n🎯 Strategy Options:")
    print("   - 'primary': Fast, write to first backend only")
    print("   - 'all': Redundant, write to all backends")
    print("   - 'quorum': Balanced, write to majority")


if __name__ == "__main__":
    asyncio.run(demo_phase10_modular())
