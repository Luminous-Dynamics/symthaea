# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Identity Recovery System with Shamir Secret Sharing
Enables account recovery without centralized authority

Implements 4 recovery scenarios:
1. Single factor loss (simple reauth)
2. Multi-factor loss (social recovery)
3. Catastrophic loss (Knowledge Council intervention)
4. Planned migration (new device/key rotation)
"""

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import json


class RecoveryStatus(Enum):
    """Status of a recovery request"""
    PENDING = "Pending"
    GUARDIAN_APPROVAL = "GuardianApproval"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    COMPLETED = "Completed"
    EXPIRED = "Expired"


class RecoveryScenario(Enum):
    """Type of recovery scenario"""
    SINGLE_FACTOR = "SingleFactor"  # Lost one factor, have others
    MULTI_FACTOR = "MultiFactor"  # Lost multiple factors, need social recovery
    CATASTROPHIC = "Catastrophic"  # Lost everything, need Council
    PLANNED_MIGRATION = "PlannedMigration"  # Device upgrade, key rotation


@dataclass
class RecoveryRequest:
    """
    A request to recover access to an identity

    Tracks the recovery process through guardian approval
    """
    request_id: str
    did: str
    scenario: RecoveryScenario
    status: RecoveryStatus = RecoveryStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=7)
    )

    # Guardian approval tracking
    required_approvals: int = 0
    received_approvals: List[str] = field(default_factory=list)  # Guardian DIDs
    approval_signatures: Dict[str, bytes] = field(default_factory=dict)

    # Evidence for recovery
    evidence: Dict[str, any] = field(default_factory=dict)

    # New keys for migration
    new_public_key: Optional[bytes] = None

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def add_approval(self, guardian_did: str, signature: bytes):
        """Add a guardian approval"""
        if guardian_did not in self.received_approvals:
            self.received_approvals.append(guardian_did)
            self.approval_signatures[guardian_did] = signature

            if len(self.received_approvals) >= self.required_approvals:
                self.status = RecoveryStatus.APPROVED

    def is_expired(self) -> bool:
        """Check if request has expired"""
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict:
        """Serialize recovery request"""
        return {
            "request_id": self.request_id,
            "did": self.did,
            "scenario": self.scenario.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "required_approvals": self.required_approvals,
            "received_approvals": len(self.received_approvals),
            "evidence": self.evidence,
            "metadata": self.metadata
        }


class ShamirSecretSharing:
    """
    Shamir Secret Sharing implementation

    Splits a secret into N shares where any K shares can reconstruct it
    """

    def __init__(self, threshold: int, total_shares: int):
        """
        Initialize Shamir Secret Sharing

        Args:
            threshold: Minimum shares needed to reconstruct (K)
            total_shares: Total number of shares to create (N)
        """
        if threshold > total_shares:
            raise ValueError("Threshold cannot exceed total shares")
        if threshold < 2:
            raise ValueError("Threshold must be at least 2")

        self.threshold = threshold
        self.total_shares = total_shares
        self.prime = self._generate_prime()

    @staticmethod
    def _generate_prime() -> int:
        """Generate a large prime number for the field"""
        # Use a well-known prime for cryptographic operations
        # Mersenne prime 2^521 - 1
        return 2**521 - 1

    def split_secret(self, secret: bytes) -> List[Tuple[int, bytes]]:
        """
        Split a secret into shares

        Args:
            secret: Secret to split (e.g., private key)

        Returns:
            List of (share_id, share_data) tuples
        """
        # Convert secret to integer
        secret_int = int.from_bytes(secret, byteorder='big')

        # Generate random coefficients for polynomial
        coefficients = [secret_int]  # a0 = secret
        for _ in range(self.threshold - 1):
            coefficients.append(secrets.randbelow(self.prime))

        # Evaluate polynomial at different points to create shares
        shares = []
        for x in range(1, self.total_shares + 1):
            y = self._evaluate_polynomial(coefficients, x)
            share_data = y.to_bytes((y.bit_length() + 7) // 8, byteorder='big')
            shares.append((x, share_data))

        return shares

    def reconstruct_secret(self, shares: List[Tuple[int, bytes]]) -> bytes:
        """
        Reconstruct secret from shares

        Args:
            shares: List of (share_id, share_data) tuples

        Returns:
            Reconstructed secret
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares, got {len(shares)}")

        # Convert shares to integers
        points = []
        for x, share_data in shares[:self.threshold]:
            y = int.from_bytes(share_data, byteorder='big')
            points.append((x, y))

        # Use Lagrange interpolation to find a0 (the secret)
        secret_int = self._lagrange_interpolation(points, 0)

        # Convert back to bytes
        secret = secret_int.to_bytes((secret_int.bit_length() + 7) // 8, byteorder='big')
        return secret

    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at x using Horner's method"""
        result = 0
        for coef in reversed(coefficients):
            result = (result * x + coef) % self.prime
        return result

    def _lagrange_interpolation(self, points: List[Tuple[int, int]], x: int) -> int:
        """
        Lagrange interpolation to find polynomial value at x

        Args:
            points: List of (x, y) coordinate tuples
            x: Point to evaluate at

        Returns:
            Polynomial value at x
        """
        result = 0

        for i, (xi, yi) in enumerate(points):
            # Calculate Lagrange basis polynomial
            numerator = 1
            denominator = 1

            for j, (xj, _) in enumerate(points):
                if i != j:
                    numerator = (numerator * (x - xj)) % self.prime
                    denominator = (denominator * (xi - xj)) % self.prime

            # Modular multiplicative inverse
            denominator_inv = pow(denominator, self.prime - 2, self.prime)

            # Add to result
            term = (yi * numerator * denominator_inv) % self.prime
            result = (result + term) % self.prime

        return result


class RecoveryManager:
    """
    Manages identity recovery processes

    Coordinates social recovery through guardian networks
    """

    def __init__(self, storage_backend=None):
        """
        Initialize Recovery Manager

        Args:
            storage_backend: Backend for persistent storage
        """
        self.storage = storage_backend or {}  # In-memory for Phase 1
        self.active_requests: Dict[str, RecoveryRequest] = {}

    def initiate_recovery(
        self,
        did: str,
        scenario: RecoveryScenario,
        guardian_dids: List[str],
        threshold: int,
        evidence: Optional[Dict] = None,
        new_public_key: Optional[bytes] = None
    ) -> RecoveryRequest:
        """
        Initiate a recovery request

        Args:
            did: DID to recover
            scenario: Type of recovery scenario
            guardian_dids: List of guardian DIDs
            threshold: Number of guardian approvals needed
            evidence: Optional evidence supporting recovery
            new_public_key: Optional new public key for migration

        Returns:
            RecoveryRequest object
        """
        # Generate unique request ID
        request_id = self._generate_request_id(did)

        # Create recovery request
        request = RecoveryRequest(
            request_id=request_id,
            did=did,
            scenario=scenario,
            required_approvals=threshold,
            evidence=evidence or {},
            new_public_key=new_public_key
        )

        # Store request
        self.active_requests[request_id] = request
        self.storage[request_id] = request.to_dict()

        return request

    def approve_recovery(
        self,
        request_id: str,
        guardian_did: str,
        signature: bytes
    ) -> bool:
        """
        Guardian approval of recovery request

        Args:
            request_id: Recovery request ID
            guardian_did: DID of approving guardian
            signature: Guardian's signature

        Returns:
            True if approval recorded successfully
        """
        request = self.active_requests.get(request_id)
        if not request:
            return False

        if request.is_expired():
            request.status = RecoveryStatus.EXPIRED
            return False

        # Add approval
        request.add_approval(guardian_did, signature)

        # Update storage
        self.storage[request_id] = request.to_dict()

        return True

    def complete_recovery(self, request_id: str) -> bool:
        """
        Complete the recovery process

        Args:
            request_id: Recovery request ID

        Returns:
            True if recovery completed successfully
        """
        request = self.active_requests.get(request_id)
        if not request:
            return False

        if request.status != RecoveryStatus.APPROVED:
            return False

        # Mark as completed
        request.status = RecoveryStatus.COMPLETED

        # Update storage
        self.storage[request_id] = request.to_dict()

        return True

    def reject_recovery(self, request_id: str, reason: str) -> bool:
        """
        Reject a recovery request

        Args:
            request_id: Recovery request ID
            reason: Reason for rejection

        Returns:
            True if rejection recorded
        """
        request = self.active_requests.get(request_id)
        if not request:
            return False

        request.status = RecoveryStatus.REJECTED
        request.metadata['rejection_reason'] = reason

        self.storage[request_id] = request.to_dict()

        return True

    def get_recovery_status(self, request_id: str) -> Optional[RecoveryRequest]:
        """Get the status of a recovery request"""
        return self.active_requests.get(request_id)

    def list_pending_recoveries(self, guardian_did: str) -> List[RecoveryRequest]:
        """
        List recovery requests pending guardian approval

        Args:
            guardian_did: DID of guardian

        Returns:
            List of pending recovery requests for this guardian
        """
        pending = []
        for request in self.active_requests.values():
            if (request.status == RecoveryStatus.PENDING and
                not request.is_expired() and
                guardian_did not in request.received_approvals):
                pending.append(request)

        return pending

    @staticmethod
    def _generate_request_id(did: str) -> str:
        """Generate a unique recovery request ID"""
        timestamp = datetime.now(timezone.utc).isoformat()
        data = f"{did}:{timestamp}".encode('utf-8')
        return hashlib.sha256(data).hexdigest()[:16]

    def split_key_for_guardians(
        self,
        private_key: bytes,
        guardian_dids: List[str],
        threshold: int
    ) -> Dict[str, bytes]:
        """
        Split a private key into shares for guardians

        Args:
            private_key: Private key to split
            guardian_dids: List of guardian DIDs
            threshold: Number of shares needed to reconstruct

        Returns:
            Dictionary mapping guardian DID to their share
        """
        sss = ShamirSecretSharing(threshold, len(guardian_dids))
        shares = sss.split_secret(private_key)

        # Map shares to guardians
        guardian_shares = {}
        for guardian_did, (share_id, share_data) in zip(guardian_dids, shares):
            guardian_shares[guardian_did] = share_data

        return guardian_shares

    def reconstruct_key_from_guardians(
        self,
        guardian_shares: Dict[str, Tuple[int, bytes]],
        threshold: int
    ) -> bytes:
        """
        Reconstruct a private key from guardian shares

        Args:
            guardian_shares: Dictionary mapping guardian DID to (share_id, share_data)
            threshold: Number of shares needed

        Returns:
            Reconstructed private key
        """
        if len(guardian_shares) < threshold:
            raise ValueError(f"Need at least {threshold} shares")

        # Get share tuples
        shares = list(guardian_shares.values())

        # Reconstruct
        sss = ShamirSecretSharing(threshold, len(shares))
        return sss.reconstruct_secret(shares)


# Example usage
if __name__ == "__main__":
    print("=== Shamir Secret Sharing Example ===")

    # Create a secret (simulating a private key)
    secret = b"This is a very secret private key!"
    print(f"Original secret: {secret}")

    # Split into 5 shares, requiring 3 to reconstruct
    sss = ShamirSecretSharing(threshold=3, total_shares=5)
    shares = sss.split_secret(secret)

    print(f"\nCreated {len(shares)} shares:")
    for share_id, share_data in shares:
        print(f"  Share {share_id}: {share_data.hex()[:32]}...")

    # Reconstruct with 3 shares
    print("\nReconstructing with shares 1, 3, 5...")
    selected_shares = [shares[0], shares[2], shares[4]]
    reconstructed = sss.reconstruct_secret(selected_shares)

    print(f"Reconstructed secret: {reconstructed}")
    print(f"Match: {secret == reconstructed}")

    # Recovery workflow example
    print("\n=== Recovery Workflow Example ===")

    manager = RecoveryManager()

    # Guardian DIDs
    guardians = [
        "did:mycelix:guardian1",
        "did:mycelix:guardian2",
        "did:mycelix:guardian3",
        "did:mycelix:guardian4",
        "did:mycelix:guardian5"
    ]

    # Initiate recovery
    request = manager.initiate_recovery(
        did="did:mycelix:alice",
        scenario=RecoveryScenario.MULTI_FACTOR,
        guardian_dids=guardians,
        threshold=3,
        evidence={"reason": "Lost phone and backup codes"}
    )

    print(f"Recovery request created: {request.request_id}")
    print(f"Status: {request.status.value}")
    print(f"Required approvals: {request.required_approvals}")

    # Guardians approve
    for i, guardian in enumerate(guardians[:3]):
        signature = f"signature_from_{guardian}".encode('utf-8')
        success = manager.approve_recovery(request.request_id, guardian, signature)
        print(f"\nGuardian {i+1} approval: {'✓' if success else '✗'}")
        print(f"Approvals received: {len(request.received_approvals)}/{request.required_approvals}")

    print(f"\nFinal status: {request.status.value}")

    # Complete recovery
    if manager.complete_recovery(request.request_id):
        print("Recovery completed successfully!")
