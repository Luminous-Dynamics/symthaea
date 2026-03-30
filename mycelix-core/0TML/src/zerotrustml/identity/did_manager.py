# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
W3C DID Manager for Mycelix

Implements did:mycelix: method specification
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization


class AgentType(Enum):
    """Types of agents in Mycelix"""
    HUMAN_MEMBER = "HumanMember"
    INSTRUMENTAL_ACTOR = "InstrumentalActor"  # AI agents
    DAO_COLLECTIVE = "DAOCollective"


@dataclass
class MycelixDID:
    """
    W3C Decentralized Identifier for Mycelix

    Format: did:mycelix:{public_key_hash}

    Example: did:mycelix:z6MkpTHR8VNsBxYAAWHut2Geadd9jSwuBV8xRoAnwWsdvktH
    """
    method: str = "mycelix"
    identifier: str = ""  # Public key hash (base58btc encoded)
    agent_type: AgentType = AgentType.HUMAN_MEMBER
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Cryptographic keys
    public_key: Optional[bytes] = None
    private_key: Optional[bytes] = None  # Never leave device!

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Generate identifier from public key if not provided"""
        if not self.identifier and self.public_key:
            self.identifier = self._generate_identifier(self.public_key)

    @staticmethod
    def _generate_identifier(public_key: bytes) -> str:
        """
        Generate identifier from public key using multibase encoding

        Uses base58btc with 'z' prefix (multibase standard)
        """
        import base58

        # Hash the public key
        hash_bytes = hashlib.sha256(public_key).digest()

        # Encode with base58btc (multibase 'z' prefix)
        return "z" + base58.b58encode(hash_bytes).decode('utf-8')

    def to_string(self) -> str:
        """Return full DID string"""
        return f"did:{self.method}:{self.identifier}"

    def to_did_document(self) -> Dict:
        """
        Generate W3C DID Document

        https://www.w3.org/TR/did-core/
        """
        doc = {
            "@context": [
                "https://www.w3.org/ns/did/v1",
                "https://w3id.org/security/suites/ed25519-2020/v1"
            ],
            "id": self.to_string(),
            "created": self.created_at.isoformat(),
            "updated": self.created_at.isoformat(),
            "verificationMethod": [
                {
                    "id": f"{self.to_string()}#keys-1",
                    "type": "Ed25519VerificationKey2020",
                    "controller": self.to_string(),
                    "publicKeyMultibase": self.identifier
                }
            ],
            "authentication": [f"{self.to_string()}#keys-1"],
            "assertionMethod": [f"{self.to_string()}#keys-1"],
            "capabilityInvocation": [f"{self.to_string()}#keys-1"],
            "capabilityDelegation": [f"{self.to_string()}#keys-1"],

            # Mycelix-specific extensions
            "mycelix": {
                "agentType": self.agent_type.value,
                "metadata": self.metadata
            }
        }

        return doc

    def sign(self, message: bytes) -> bytes:
        """Sign a message with private key"""
        if not self.private_key:
            raise ValueError("Private key not available")

        private_key_obj = ed25519.Ed25519PrivateKey.from_private_bytes(self.private_key)
        return private_key_obj.sign(message)

    def verify(self, message: bytes, signature: bytes) -> bool:
        """Verify a signature with public key"""
        if not self.public_key:
            raise ValueError("Public key not available")

        public_key_obj = ed25519.Ed25519PublicKey.from_public_bytes(self.public_key)

        try:
            public_key_obj.verify(signature, message)
            return True
        except Exception:
            return False


class DIDManager:
    """
    Manages Mycelix DIDs

    Handles creation, resolution, and storage of DIDs
    """

    def __init__(self, storage_backend=None):
        """
        Initialize DID Manager

        Args:
            storage_backend: Backend for persistent storage (DHT, database, etc.)
        """
        self.storage = storage_backend or {}  # In-memory for Phase 1

    def create_did(
        self,
        agent_type: AgentType = AgentType.HUMAN_MEMBER,
        metadata: Optional[Dict] = None
    ) -> MycelixDID:
        """
        Create a new Mycelix DID

        Generates Ed25519 key pair and creates DID

        Args:
            agent_type: Type of agent (Human, AI, DAO)
            metadata: Optional metadata to include

        Returns:
            MycelixDID object with generated keys
        """
        # Generate Ed25519 key pair
        private_key_obj = ed25519.Ed25519PrivateKey.generate()
        public_key_obj = private_key_obj.public_key()

        # Serialize keys
        private_key_bytes = private_key_obj.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_key_bytes = public_key_obj.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

        # Create DID
        did = MycelixDID(
            public_key=public_key_bytes,
            private_key=private_key_bytes,
            agent_type=agent_type,
            metadata=metadata or {}
        )

        # Store DID document (public parts only)
        self.store_did(did)

        return did

    def store_did(self, did: MycelixDID):
        """
        Store DID document in storage backend

        Only stores public information, never private keys
        """
        did_string = did.to_string()
        did_document = did.to_did_document()

        # Remove private key before storage
        storage_doc = {
            "did": did_string,
            "document": did_document,
            "created_at": did.created_at.isoformat(),
            "agent_type": did.agent_type.value,
        }

        self.storage[did_string] = storage_doc

    def resolve_did(self, did_string: str) -> Optional[Dict]:
        """
        Resolve a DID to its DID Document

        Args:
            did_string: Full DID (e.g., did:mycelix:z6Mk...)

        Returns:
            DID Document or None if not found
        """
        return self.storage.get(did_string)

    def update_metadata(self, did_string: str, metadata: Dict):
        """Update DID metadata"""
        if did_string in self.storage:
            self.storage[did_string]["document"]["mycelix"]["metadata"] = metadata
            self.storage[did_string]["document"]["updated"] = datetime.now(timezone.utc).isoformat()

    def list_dids(self) -> List[str]:
        """List all DIDs in storage"""
        return list(self.storage.keys())

    @staticmethod
    def parse_did(did_string: str) -> Dict[str, str]:
        """
        Parse a DID string into components

        Args:
            did_string: Full DID string

        Returns:
            Dict with method and identifier
        """
        if not did_string.startswith("did:"):
            raise ValueError(f"Invalid DID format: {did_string}")

        parts = did_string.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid DID format: {did_string}")

        return {
            "scheme": parts[0],
            "method": parts[1],
            "identifier": parts[2]
        }

    @staticmethod
    def validate_did(did_string: str) -> bool:
        """Validate DID format"""
        try:
            parsed = DIDManager.parse_did(did_string)
            return parsed["method"] == "mycelix" and len(parsed["identifier"]) > 0
        except Exception:
            return False


# Example usage
if __name__ == "__main__":
    # Create DID manager
    manager = DIDManager()

    # Create a human member DID
    human_did = manager.create_did(
        agent_type=AgentType.HUMAN_MEMBER,
        metadata={"nickname": "Alice"}
    )

    print("Created DID:", human_did.to_string())
    print("\nDID Document:")
    print(json.dumps(human_did.to_did_document(), indent=2))

    # Sign and verify a message
    message = b"Hello, Mycelix!"
    signature = human_did.sign(message)
    verified = human_did.verify(message, signature)
    print(f"\nSignature verified: {verified}")

    # Resolve DID
    resolved = manager.resolve_did(human_did.to_string())
    print(f"\nResolved DID: {resolved is not None}")
