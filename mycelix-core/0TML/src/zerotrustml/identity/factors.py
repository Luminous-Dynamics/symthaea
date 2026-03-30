# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Identity Factors for Multi-Factor Decentralized Identity (MFDI)
Phase 1 Implementation

Implements the 9 identity factor types across 5 categories
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from cryptography.hazmat.primitives.asymmetric import ed25519
import requests


class FactorCategory(Enum):
    """Categories of identity factors"""
    PRIMARY = "Primary"
    BACKUP = "Backup"
    SOCIAL = "Social"
    REPUTATION = "Reputation"
    BIOMETRIC = "Biometric"


class FactorStatus(Enum):
    """Status of an identity factor"""
    ACTIVE = "Active"
    INACTIVE = "Inactive"
    REVOKED = "Revoked"
    PENDING = "Pending"


@dataclass
class IdentityFactor(ABC):
    """
    Base class for all identity factors

    Each factor provides a verification method and contributes
    to the overall assurance level of the identity
    """
    factor_id: str
    factor_type: str
    category: FactorCategory
    status: FactorStatus = FactorStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_verified: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def verify(self, challenge: Any) -> bool:
        """
        Verify this factor with a challenge

        Args:
            challenge: Factor-specific verification challenge

        Returns:
            True if verification succeeds
        """
        pass

    @abstractmethod
    def get_contribution(self) -> float:
        """
        Get this factor's contribution to assurance level (0.0-1.0)

        Returns:
            Contribution weight
        """
        pass

    def to_dict(self) -> Dict:
        """Serialize factor to dictionary"""
        return {
            "factor_id": self.factor_id,
            "factor_type": self.factor_type,
            "category": self.category.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_verified": self.last_verified.isoformat() if self.last_verified else None,
            "metadata": self.metadata
        }


@dataclass
class CryptoKeyFactor(IdentityFactor):
    """
    Primary cryptographic key pair (Ed25519)

    This is the foundational factor - the DID itself is derived from this key
    """
    public_key: Optional[bytes] = None
    private_key: Optional[bytes] = None  # Never serialized or transmitted!

    def __post_init__(self):
        """Initialize with proper values"""
        if not self.factor_type:
            self.factor_type = "CryptoKey"
        if not self.category:
            self.category = FactorCategory.PRIMARY

    def verify(self, challenge: Dict[str, Any]) -> bool:
        """
        Verify ownership of private key by signing a challenge

        Args:
            challenge: Dict with 'message' (bytes) and 'signature' (bytes)

        Returns:
            True if signature is valid
        """
        if not self.public_key:
            return False

        try:
            message = challenge['message']
            signature = challenge['signature']

            public_key_obj = ed25519.Ed25519PublicKey.from_public_bytes(self.public_key)
            public_key_obj.verify(signature, message)

            self.last_verified = datetime.now(timezone.utc)
            self.status = FactorStatus.ACTIVE
            return True
        except Exception:
            return False

    def sign(self, message: bytes) -> bytes:
        """Sign a message with the private key"""
        if not self.private_key:
            raise ValueError("Private key not available")

        private_key_obj = ed25519.Ed25519PrivateKey.from_private_bytes(self.private_key)
        return private_key_obj.sign(message)

    def get_contribution(self) -> float:
        """Primary key contributes 50% to assurance level"""
        if self.status == FactorStatus.ACTIVE:
            return 0.5
        return 0.0

    def to_dict(self) -> Dict:
        """Serialize, excluding private key"""
        base = super().to_dict()
        # NEVER include private key in serialization
        base["has_private_key"] = self.private_key is not None
        return base


@dataclass
class GitcoinPassportFactor(IdentityFactor):
    """
    Gitcoin Passport for Sybil resistance

    Phase 1 Requirement: Score ≥ 20 for governance participation
    """
    passport_address: Optional[str] = None
    score: float = 0.0
    stamps: List[str] = field(default_factory=list)
    expiry: Optional[datetime] = None

    def __post_init__(self):
        if not self.factor_type:
            self.factor_type = "GitcoinPassport"
        if not self.category:
            self.category = FactorCategory.REPUTATION

    def verify(self, challenge: Dict[str, Any]) -> bool:
        """
        Verify Gitcoin Passport by checking score and stamps

        Args:
            challenge: Dict with 'api_key' and 'scorer_id' for Gitcoin API

        Returns:
            True if passport is valid and score meets threshold
        """
        if not self.passport_address:
            return False

        try:
            # Fetch passport data from Gitcoin API
            api_key = challenge.get('api_key')
            scorer_id = challenge.get('scorer_id')

            headers = {
                'X-API-KEY': api_key,
                'Content-Type': 'application/json'
            }

            url = f"https://api.scorer.gitcoin.co/registry/score/{scorer_id}/{self.passport_address}"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                self.score = float(data.get('score', 0))
                self.stamps = [stamp['credential']['credentialSubject']['provider']
                              for stamp in data.get('stamps', [])]

                # Phase 1 requirement: Score ≥ 20
                if self.score >= 20.0:
                    self.last_verified = datetime.now(timezone.utc)
                    self.status = FactorStatus.ACTIVE
                    return True

            return False
        except Exception as e:
            self.metadata['last_error'] = str(e)
            return False

    def get_contribution(self) -> float:
        """
        Contribution based on Gitcoin score

        Score ≥ 20: 0.3 contribution
        Score ≥ 50: 0.4 contribution
        """
        if self.status != FactorStatus.ACTIVE:
            return 0.0

        if self.score >= 50.0:
            return 0.4
        elif self.score >= 20.0:
            return 0.3
        return 0.0

    def to_dict(self) -> Dict:
        """Serialize passport data"""
        base = super().to_dict()
        base.update({
            "passport_address": self.passport_address,
            "score": self.score,
            "stamps": self.stamps,
            "expiry": self.expiry.isoformat() if self.expiry else None
        })
        return base


@dataclass
class SocialRecoveryFactor(IdentityFactor):
    """
    Social recovery guardians using Shamir Secret Sharing

    Enables recovery if primary key is lost
    """
    guardian_dids: List[str] = field(default_factory=list)
    threshold: int = 0  # Minimum guardians needed for recovery
    total_shares: int = 0
    recovery_instructions: Optional[str] = None

    def __post_init__(self):
        if not self.factor_type:
            self.factor_type = "SocialRecovery"
        if not self.category:
            self.category = FactorCategory.SOCIAL

    def verify(self, challenge: Dict[str, Any]) -> bool:
        """
        Verify guardian configuration

        Args:
            challenge: Dict with 'guardian_signatures' - signatures from guardians

        Returns:
            True if threshold number of guardians sign
        """
        signatures = challenge.get('guardian_signatures', [])

        if len(signatures) >= self.threshold:
            self.last_verified = datetime.now(timezone.utc)
            self.status = FactorStatus.ACTIVE
            return True

        return False

    def get_contribution(self) -> float:
        """
        Contribution based on guardian configuration

        Well-configured social recovery (5+ guardians, threshold ≥ 3): 0.3
        Basic configuration (3+ guardians, threshold ≥ 2): 0.2
        """
        if self.status != FactorStatus.ACTIVE:
            return 0.0

        if len(self.guardian_dids) >= 5 and self.threshold >= 3:
            return 0.3
        elif len(self.guardian_dids) >= 3 and self.threshold >= 2:
            return 0.2
        return 0.1

    def add_guardian(self, guardian_did: str):
        """Add a guardian to the recovery set"""
        if guardian_did not in self.guardian_dids:
            self.guardian_dids.append(guardian_did)
            self.total_shares = len(self.guardian_dids)

    def remove_guardian(self, guardian_did: str):
        """Remove a guardian from the recovery set"""
        if guardian_did in self.guardian_dids:
            self.guardian_dids.remove(guardian_did)
            self.total_shares = len(self.guardian_dids)

    def to_dict(self) -> Dict:
        """Serialize guardian configuration"""
        base = super().to_dict()
        base.update({
            "guardian_dids": self.guardian_dids,
            "threshold": self.threshold,
            "total_shares": self.total_shares,
            "recovery_instructions": self.recovery_instructions
        })
        return base


@dataclass
class BiometricFactor(IdentityFactor):
    """
    Biometric identity factor (hash only, never raw biometric data)

    Phase 1: Basic implementation (placeholder)
    Phase 2+: Full privacy-preserving biometric verification
    """
    biometric_type: str = ""  # "face", "fingerprint", "iris", "voice"
    template_hash: Optional[str] = None

    def __post_init__(self):
        if not self.factor_type:
            self.factor_type = "Biometric"
        if not self.category:
            self.category = FactorCategory.BIOMETRIC

    def verify(self, challenge: Dict[str, Any]) -> bool:
        """
        Verify biometric by comparing hashes

        Args:
            challenge: Dict with 'template_hash' to compare

        Returns:
            True if hashes match (Phase 1 simple implementation)
        """
        if not self.template_hash:
            return False

        challenge_hash = challenge.get('template_hash')

        if challenge_hash == self.template_hash:
            self.last_verified = datetime.now(timezone.utc)
            self.status = FactorStatus.ACTIVE
            return True

        return False

    def get_contribution(self) -> float:
        """
        Biometric contributes 0.2 to assurance level

        Note: Phase 1 contribution is limited due to simple implementation
        Phase 2+ with privacy-preserving verification: 0.3-0.4
        """
        if self.status == FactorStatus.ACTIVE:
            return 0.2
        return 0.0

    def to_dict(self) -> Dict:
        """Serialize biometric metadata (no raw data)"""
        base = super().to_dict()
        base.update({
            "biometric_type": self.biometric_type,
            "has_template": self.template_hash is not None
        })
        return base


@dataclass
class HardwareKeyFactor(IdentityFactor):
    """
    Hardware security key (YubiKey, Ledger, etc.)

    Phase 1: Basic support
    Phase 2+: Full WebAuthn/FIDO2 integration
    """
    device_type: str = ""  # "yubikey", "ledger", "trezor", etc.
    device_id: Optional[str] = None
    public_key: Optional[bytes] = None

    def __post_init__(self):
        if not self.factor_type:
            self.factor_type = "HardwareKey"
        if not self.category:
            self.category = FactorCategory.BACKUP

    def verify(self, challenge: Dict[str, Any]) -> bool:
        """
        Verify hardware key by signature

        Args:
            challenge: Dict with 'message' and 'signature' from device

        Returns:
            True if signature is valid
        """
        if not self.public_key:
            return False

        try:
            message = challenge['message']
            signature = challenge['signature']

            # Phase 1: Simple Ed25519 verification
            # Phase 2+: Full WebAuthn/FIDO2 attestation
            public_key_obj = ed25519.Ed25519PublicKey.from_public_bytes(self.public_key)
            public_key_obj.verify(signature, message)

            self.last_verified = datetime.now(timezone.utc)
            self.status = FactorStatus.ACTIVE
            return True
        except Exception:
            return False

    def get_contribution(self) -> float:
        """
        Hardware key contributes 0.3 to assurance level

        High contribution due to physical device requirement
        """
        if self.status == FactorStatus.ACTIVE:
            return 0.3
        return 0.0

    def to_dict(self) -> Dict:
        """Serialize hardware key metadata"""
        base = super().to_dict()
        base.update({
            "device_type": self.device_type,
            "device_id": self.device_id,
            "has_public_key": self.public_key is not None
        })
        return base


# Factory function for creating factors
def create_factor(factor_type: str, **kwargs) -> IdentityFactor:
    """
    Factory function to create identity factors

    Args:
        factor_type: Type of factor to create
        **kwargs: Factor-specific parameters

    Returns:
        IdentityFactor instance
    """
    factor_map = {
        "CryptoKey": CryptoKeyFactor,
        "GitcoinPassport": GitcoinPassportFactor,
        "SocialRecovery": SocialRecoveryFactor,
        "Biometric": BiometricFactor,
        "HardwareKey": HardwareKeyFactor,
    }

    factor_class = factor_map.get(factor_type)
    if not factor_class:
        raise ValueError(f"Unknown factor type: {factor_type}")

    return factor_class(**kwargs)


# Example usage
if __name__ == "__main__":
    # Create a primary crypto key factor
    from cryptography.hazmat.primitives.asymmetric import ed25519

    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    crypto_factor = CryptoKeyFactor(
        factor_id="crypto-001",
        factor_type="CryptoKey",
        category=FactorCategory.PRIMARY,
        public_key=public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        ),
        private_key=private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )
    )

    # Test signing and verification
    message = b"Test message"
    signature = crypto_factor.sign(message)

    verified = crypto_factor.verify({
        'message': message,
        'signature': signature
    })

    print(f"Crypto factor verified: {verified}")
    print(f"Contribution: {crypto_factor.get_contribution()}")

    # Create a Gitcoin Passport factor
    passport_factor = GitcoinPassportFactor(
        factor_id="gitcoin-001",
        factor_type="GitcoinPassport",
        category=FactorCategory.REPUTATION,
        passport_address="0x1234567890123456789012345678901234567890"
    )

    print(f"\nPassport factor created: {passport_factor.to_dict()}")

    # Create social recovery
    recovery_factor = SocialRecoveryFactor(
        factor_id="recovery-001",
        factor_type="SocialRecovery",
        category=FactorCategory.SOCIAL,
        threshold=3
    )

    recovery_factor.add_guardian("did:mycelix:guardian1")
    recovery_factor.add_guardian("did:mycelix:guardian2")
    recovery_factor.add_guardian("did:mycelix:guardian3")
    recovery_factor.add_guardian("did:mycelix:guardian4")
    recovery_factor.add_guardian("did:mycelix:guardian5")

    print(f"\nSocial recovery configured: {len(recovery_factor.guardian_dids)} guardians")
    print(f"Contribution: {recovery_factor.get_contribution()}")
