#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
LocalFile Backend Implementation

Simple JSON-based file storage for testing and development.
Perfect for quick prototyping without database infrastructure.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import json
import os
import time
from pathlib import Path
import hashlib
import uuid

from .storage_backend import (
    StorageBackend,
    BackendType,
    GradientRecord,
    CreditRecord,
    ByzantineEvent,
    ConnectionError as BackendConnectionError,
    IntegrityError,
    NotFoundError
)

logger = logging.getLogger(__name__)


class LocalFileBackend(StorageBackend):
    """
    LocalFile backend for testing and development

    Features:
    - No database required
    - Human-readable JSON files
    - Fast iteration for testing
    - Easy inspection and debugging
    - Git-friendly (can commit test data)

    Use cases:
    - Unit testing
    - Development
    - Quick prototypes
    - Demo mode
    """

    def __init__(self, data_dir: str = "/tmp/zerotrustml_data"):
        super().__init__(BackendType.LOCALFILE)
        self.data_dir = Path(data_dir)
        self.gradients_dir = self.data_dir / "gradients"
        self.credits_dir = self.data_dir / "credits"
        self.byzantine_dir = self.data_dir / "byzantine_events"
        self.reputation_file = self.data_dir / "reputation.json"
        self.connection_time = None

    async def connect(self) -> bool:
        """Create directory structure"""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.gradients_dir.mkdir(exist_ok=True)
            self.credits_dir.mkdir(exist_ok=True)
            self.byzantine_dir.mkdir(exist_ok=True)

            # Initialize reputation file if doesn't exist
            if not self.reputation_file.exists():
                self.reputation_file.write_text(json.dumps({}))

            self.connected = True
            self.connection_time = time.time()
            logger.info(f"✅ LocalFile backend ready: {self.data_dir}")
            return True

        except Exception as e:
            logger.error(f"LocalFile backend failed to initialize: {e}")
            raise BackendConnectionError(f"Failed to create directories: {e}")

    async def disconnect(self) -> None:
        """No cleanup needed for file-based storage"""
        self.connected = False
        logger.info("LocalFile backend closed")

    async def health_check(self) -> Dict[str, Any]:
        """Check directory access and permissions"""
        try:
            # Test write access
            test_file = self.data_dir / ".health_check"
            test_file.write_text("test")
            test_file.unlink()

            # Count files
            gradient_count = len(list(self.gradients_dir.glob("*.json")))
            credit_count = len(list(self.credits_dir.glob("*.json")))

            return {
                "healthy": True,
                "latency_ms": 0.1,  # File I/O is fast
                "storage_available": True,
                "metadata": {
                    "data_dir": str(self.data_dir),
                    "gradient_files": gradient_count,
                    "credit_files": credit_count,
                    "disk_writable": True
                }
            }

        except Exception as e:
            return {
                "healthy": False,
                "latency_ms": None,
                "storage_available": False,
                "metadata": {"error": str(e)}
            }

    async def store_gradient(self, gradient_data: Dict[str, Any]) -> str:
        """Store gradient as JSON file (supports encrypted gradients)"""
        gradient_id = gradient_data.get("id", str(uuid.uuid4()))

        # Prepare gradient record
        # Note: gradient field may be encrypted (base64 string) or plaintext (list)
        record = {
            "id": gradient_id,
            "node_id": gradient_data["node_id"],
            "round_num": gradient_data["round_num"],
            "gradient": gradient_data["gradient"],  # May be encrypted base64 string
            "gradient_hash": gradient_data["gradient_hash"],
            "pogq_score": gradient_data.get("pogq_score"),
            "zkpoc_verified": gradient_data.get("zkpoc_verified", False),
            "validation_passed": gradient_data.get("validation_passed", True),
            "reputation_score": gradient_data.get("reputation_score", 0.5),
            "timestamp": gradient_data.get("timestamp", time.time()),
            "encrypted": gradient_data.get("encrypted", False)  # Track encryption status
        }

        # Write to file
        file_path = self.gradients_dir / f"{gradient_id}.json"
        file_path.write_text(json.dumps(record, indent=2))

        encryption_status = "encrypted" if record["encrypted"] else "plaintext"
        logger.debug(f"LocalFile: Gradient stored {gradient_id[:8]}... ({encryption_status})")
        return gradient_id

    async def get_gradient(self, gradient_id: str) -> Optional[GradientRecord]:
        """Retrieve gradient from JSON file"""
        file_path = self.gradients_dir / f"{gradient_id}.json"

        if not file_path.exists():
            return None

        data = json.loads(file_path.read_text())

        return GradientRecord(
            id=data["id"],
            node_id=data["node_id"],
            round_num=data["round_num"],
            gradient=data["gradient"],  # May be encrypted base64 string
            gradient_hash=data["gradient_hash"],
            pogq_score=data.get("pogq_score"),
            zkpoc_verified=data.get("zkpoc_verified", False),
            reputation_score=data.get("reputation_score", 0.5),
            timestamp=data["timestamp"],
            backend_metadata={
                "file_path": str(file_path),
                "encrypted": data.get("encrypted", False)
            }
        )

    async def get_gradients_by_round(self, round_num: int) -> List[GradientRecord]:
        """Get all gradients for a round"""
        gradients = []

        for file_path in self.gradients_dir.glob("*.json"):
            data = json.loads(file_path.read_text())
            if data["round_num"] == round_num and data.get("validation_passed", True):
                gradients.append(GradientRecord(
                    id=data["id"],
                    node_id=data["node_id"],
                    round_num=data["round_num"],
                    gradient=data["gradient"],  # May be encrypted base64 string
                    gradient_hash=data["gradient_hash"],
                    pogq_score=data.get("pogq_score"),
                    zkpoc_verified=data.get("zkpoc_verified", False),
                    reputation_score=data.get("reputation_score", 0.5),
                    timestamp=data["timestamp"],
                    backend_metadata={
                        "file_path": str(file_path),
                        "encrypted": data.get("encrypted", False)
                    }
                ))

        # Sort by timestamp
        gradients.sort(key=lambda g: g.timestamp)
        return gradients

    async def verify_gradient_integrity(self, gradient_id: str) -> bool:
        """Verify gradient hash"""
        gradient = await self.get_gradient(gradient_id)
        if not gradient:
            raise NotFoundError(f"Gradient {gradient_id} not found")

        # Recompute hash
        import numpy as np
        gradient_array = np.array(gradient.gradient)
        computed_hash = hashlib.sha256(gradient_array.tobytes()).hexdigest()

        return computed_hash == gradient.gradient_hash

    async def issue_credit(
        self,
        holder: str,
        amount: int,
        earned_from: str
    ) -> str:
        """Issue credit and save to file"""
        credit_id = str(uuid.uuid4())

        record = {
            "transaction_id": credit_id,
            "holder": holder,
            "amount": amount,
            "earned_from": earned_from,
            "timestamp": time.time()
        }

        # Write to file
        file_path = self.credits_dir / f"{credit_id}.json"
        file_path.write_text(json.dumps(record, indent=2))

        logger.debug(f"LocalFile: Credit issued {credit_id[:8]}... ({amount} to {holder})")
        return credit_id

    async def get_credit_balance(self, node_id: str) -> int:
        """Get total credit balance"""
        total = 0

        for file_path in self.credits_dir.glob("*.json"):
            data = json.loads(file_path.read_text())
            if data["holder"] == node_id:
                total += data["amount"]

        return total

    async def get_credit_history(self, node_id: str) -> List[CreditRecord]:
        """Get credit history for node"""
        credits = []

        for file_path in self.credits_dir.glob("*.json"):
            data = json.loads(file_path.read_text())
            if data["holder"] == node_id:
                credits.append(CreditRecord(
                    transaction_id=data["transaction_id"],
                    holder=data["holder"],
                    amount=data["amount"],
                    earned_from=data["earned_from"],
                    timestamp=data["timestamp"]
                ))

        # Sort by timestamp descending
        credits.sort(key=lambda c: c.timestamp, reverse=True)
        return credits

    async def get_reputation(self, node_id: str) -> Dict[str, Any]:
        """Get node reputation"""
        # Count gradients
        total_submitted = 0
        accepted = 0

        for file_path in self.gradients_dir.glob("*.json"):
            data = json.loads(file_path.read_text())
            if data["node_id"] == node_id:
                total_submitted += 1
                if data.get("validation_passed", True):
                    accepted += 1

        # Count Byzantine events
        byzantine_count = 0
        for file_path in self.byzantine_dir.glob("*.json"):
            data = json.loads(file_path.read_text())
            if data["node_id"] == node_id:
                byzantine_count += 1

        # Calculate score
        base_score = accepted / max(1, total_submitted)
        byzantine_penalty = min(0.5, byzantine_count * 0.1)
        score = max(0.0, base_score - byzantine_penalty)

        # Get total credits
        total_credits = await self.get_credit_balance(node_id)

        return {
            "node_id": node_id,
            "score": score,
            "gradients_submitted": total_submitted,
            "gradients_accepted": accepted,
            "byzantine_events": byzantine_count,
            "total_credits": total_credits
        }

    async def update_reputation(
        self,
        node_id: str,
        score_delta: float,
        reason: str
    ) -> None:
        """Update reputation (tracked via events in LocalFile backend)"""
        logger.debug(f"Reputation update for {node_id}: {score_delta:+.3f} ({reason})")

    async def log_byzantine_event(self, event: Dict[str, Any]) -> str:
        """Log Byzantine event"""
        event_id = str(uuid.uuid4())

        record = {
            "event_id": event_id,
            "node_id": event["node_id"],
            "round_num": event["round_num"],
            "detection_method": event["detection_method"],
            "severity": event["severity"],
            "details": event.get("details", {}),
            "timestamp": time.time()
        }

        file_path = self.byzantine_dir / f"{event_id}.json"
        file_path.write_text(json.dumps(record, indent=2))

        logger.info(f"LocalFile: Byzantine event logged {event_id[:8]}...")
        return event_id

    async def get_byzantine_events(
        self,
        node_id: Optional[str] = None,
        round_num: Optional[int] = None
    ) -> List[ByzantineEvent]:
        """Get Byzantine events (filtered)"""
        events = []

        for file_path in self.byzantine_dir.glob("*.json"):
            data = json.loads(file_path.read_text())

            # Apply filters
            if node_id and data["node_id"] != node_id:
                continue
            if round_num is not None and data["round_num"] != round_num:
                continue

            events.append(ByzantineEvent(
                event_id=data["event_id"],
                node_id=data["node_id"],
                round_num=data["round_num"],
                detection_method=data["detection_method"],
                severity=data["severity"],
                details=data["details"],
                timestamp=data["timestamp"]
            ))

        # Sort by timestamp descending
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events

    async def get_stats(self) -> Dict[str, Any]:
        """Get LocalFile backend statistics"""
        gradient_count = len(list(self.gradients_dir.glob("*.json")))
        credit_count = len(list(self.credits_dir.glob("*.json")))
        byzantine_count = len(list(self.byzantine_dir.glob("*.json")))

        # Calculate total credits
        total_credits = 0
        for file_path in self.credits_dir.glob("*.json"):
            data = json.loads(file_path.read_text())
            total_credits += data["amount"]

        # Estimate storage size
        storage_size = sum(
            f.stat().st_size
            for f in self.data_dir.rglob("*.json")
        )

        uptime = time.time() - self.connection_time if self.connection_time else 0

        return {
            "backend_type": "localfile",
            "total_gradients": gradient_count,
            "total_credits_issued": total_credits,
            "total_byzantine_events": byzantine_count,
            "storage_size_bytes": storage_size,
            "uptime_seconds": uptime,
            "metadata": {
                "data_dir": str(self.data_dir),
                "gradient_files": gradient_count,
                "credit_files": credit_count,
                "byzantine_files": byzantine_count
            }
        }
