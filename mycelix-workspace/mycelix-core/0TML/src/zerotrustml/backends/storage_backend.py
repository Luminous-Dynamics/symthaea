#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Abstract Storage Backend Interface

Defines the contract all storage backends must implement.
Enables modular architecture supporting PostgreSQL, Holochain, Blockchain, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class BackendType(Enum):
    """Types of storage backends"""
    POSTGRESQL = "postgresql"
    HOLOCHAIN = "holochain"
    BLOCKCHAIN = "blockchain"
    LOCALFILE = "localfile"
    IPFS = "ipfs"


@dataclass
class GradientRecord:
    """Gradient record returned by backends"""
    id: str
    node_id: str
    round_num: int
    gradient: List[float]
    gradient_hash: str
    pogq_score: Optional[float]
    zkpoc_verified: bool
    reputation_score: float
    timestamp: float
    backend_metadata: Optional[Dict[str, Any]] = None  # Backend-specific data
    edge_proof: Optional[Dict[str, Any]] = None
    committee_votes: Optional[List[Dict[str, Any]]] = None


@dataclass
class CreditRecord:
    """Credit issuance record"""
    transaction_id: str
    holder: str
    amount: int
    earned_from: str
    timestamp: float
    backend_metadata: Optional[Dict[str, Any]] = None


@dataclass
class ByzantineEvent:
    """Byzantine detection event"""
    event_id: str
    node_id: str
    round_num: int
    detection_method: str
    severity: str
    details: Dict[str, Any]
    timestamp: float


class StorageBackend(ABC):
    """
    Abstract base class for storage backends

    All backends (PostgreSQL, Holochain, Blockchain, LocalFile) must implement
    these methods to provide consistent interface to Phase10Coordinator.

    Design Principles:
    - Backend-agnostic: Coordinator doesn't know which backend is used
    - Async-first: All operations are async for performance
    - Type-safe: Strongly typed return values
    - Verifiable: All operations return IDs for later verification
    """

    def __init__(self, backend_type: BackendType):
        self.backend_type = backend_type
        self.connected = False

    # Connection Management

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to backend storage system

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnect from backend"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check backend health and return status

        Returns:
            {
                "healthy": bool,
                "latency_ms": float,
                "storage_available": bool,
                "metadata": dict  # Backend-specific health info
            }
        """
        pass

    # Gradient Operations

    @abstractmethod
    async def store_gradient(self, gradient_data: Dict[str, Any]) -> str:
        """
        Store gradient submission

        Args:
            gradient_data: {
                "id": str,
                "node_id": str,
                "round_num": int,
                "gradient": List[float],
                "gradient_hash": str,
                "pogq_score": Optional[float],
                "zkpoc_verified": bool,
                "validation_passed": bool,
                "reputation_score": float,
                "timestamp": float
            }

        Returns:
            gradient_id: Unique identifier for retrieval
        """
        pass

    @abstractmethod
    async def get_gradient(self, gradient_id: str) -> Optional[GradientRecord]:
        """
        Retrieve gradient by ID

        Returns:
            GradientRecord or None if not found
        """
        pass

    @abstractmethod
    async def get_gradients_by_round(self, round_num: int) -> List[GradientRecord]:
        """
        Get all gradients for a specific round

        Returns:
            List of GradientRecords
        """
        pass

    @abstractmethod
    async def verify_gradient_integrity(self, gradient_id: str) -> bool:
        """
        Verify gradient hasn't been tampered with

        For immutable backends (Holochain, Blockchain): Always true
        For mutable backends (PostgreSQL): Check hash against stored value

        Returns:
            True if gradient is intact
        """
        pass

    # Credit Operations

    @abstractmethod
    async def issue_credit(
        self,
        holder: str,
        amount: int,
        earned_from: str
    ) -> str:
        """
        Issue credits to a node

        Args:
            holder: Node ID receiving credits
            amount: Credit amount
            earned_from: Reason for credit (e.g., "gradient_quality")

        Returns:
            transaction_id: Unique identifier for this credit issuance
        """
        pass

    @abstractmethod
    async def get_credit_balance(self, node_id: str) -> int:
        """
        Get total credit balance for a node

        Returns:
            Total credits earned
        """
        pass

    @abstractmethod
    async def get_credit_history(self, node_id: str) -> List[CreditRecord]:
        """
        Get credit transaction history for a node

        Returns:
            List of CreditRecords
        """
        pass

    # Reputation Operations

    @abstractmethod
    async def get_reputation(self, node_id: str) -> Dict[str, Any]:
        """
        Get reputation data for a node

        Returns:
            {
                "node_id": str,
                "score": float,  # 0.0-1.0
                "gradients_submitted": int,
                "gradients_accepted": int,
                "byzantine_events": int,
                "total_credits": int
            }
        """
        pass

    @abstractmethod
    async def update_reputation(
        self,
        node_id: str,
        score_delta: float,
        reason: str
    ) -> None:
        """
        Update node reputation

        Args:
            node_id: Node to update
            score_delta: Change in reputation (-1.0 to 1.0)
            reason: Why reputation changed
        """
        pass

    # Byzantine Event Logging

    @abstractmethod
    async def log_byzantine_event(self, event: Dict[str, Any]) -> str:
        """
        Log a Byzantine detection event

        Args:
            event: {
                "node_id": str,
                "round_num": int,
                "detection_method": str,  # "zkpoc", "pogq", "krum"
                "severity": str,  # "low", "medium", "high"
                "details": dict
            }

        Returns:
            event_id: Unique identifier for this event
        """
        pass

    @abstractmethod
    async def get_byzantine_events(
        self,
        node_id: Optional[str] = None,
        round_num: Optional[int] = None
    ) -> List[ByzantineEvent]:
        """
        Get Byzantine events (optionally filtered)

        Args:
            node_id: Filter by node (optional)
            round_num: Filter by round (optional)

        Returns:
            List of ByzantineEvents
        """
        pass

    # Statistics & Metrics

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get backend statistics

        Returns:
            {
                "backend_type": str,
                "total_gradients": int,
                "total_credits_issued": int,
                "total_byzantine_events": int,
                "storage_size_bytes": int,
                "uptime_seconds": float,
                "metadata": dict  # Backend-specific stats
            }
        """
        pass


class StorageBackendError(Exception):
    """Base exception for storage backend errors"""
    pass


class ConnectionError(StorageBackendError):
    """Backend connection failed"""
    pass


class IntegrityError(StorageBackendError):
    """Data integrity check failed"""
    pass


class NotFoundError(StorageBackendError):
    """Requested resource not found"""
    pass
