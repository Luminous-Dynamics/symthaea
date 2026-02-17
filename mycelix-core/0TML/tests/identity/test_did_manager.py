"""
Unit tests for DID Manager
Tests W3C DID creation, resolution, and cryptographic operations
"""

import pytest
from datetime import datetime, timezone

from zerotrustml.identity import DIDManager, MycelixDID
from zerotrustml.identity.did_manager import AgentType


class TestMycelixDID:
    """Test MycelixDID dataclass"""

    def test_did_string_format(self):
        """Test DID string format is correct"""
        manager = DIDManager()
        did = manager.create_did(agent_type=AgentType.HUMAN_MEMBER)

        did_string = did.to_string()

        assert did_string.startswith("did:mycelix:z")
        assert len(did_string) > 20  # Reasonable length check

    def test_did_identifier_generation(self):
        """Test identifier is properly generated from public key"""
        manager = DIDManager()
        did = manager.create_did()

        # Identifier should be generated
        assert did.identifier
        assert did.identifier.startswith("z")  # Base58btc prefix
        assert len(did.identifier) > 32  # SHA256 hash encoded

    def test_agent_type_assignment(self):
        """Test different agent types can be assigned"""
        manager = DIDManager()

        # Human member
        human_did = manager.create_did(agent_type=AgentType.HUMAN_MEMBER)
        assert human_did.agent_type == AgentType.HUMAN_MEMBER

        # AI agent
        ai_did = manager.create_did(agent_type=AgentType.INSTRUMENTAL_ACTOR)
        assert ai_did.agent_type == AgentType.INSTRUMENTAL_ACTOR

        # DAO
        dao_did = manager.create_did(agent_type=AgentType.DAO_COLLECTIVE)
        assert dao_did.agent_type == AgentType.DAO_COLLECTIVE

    def test_metadata_storage(self):
        """Test metadata can be stored and retrieved"""
        manager = DIDManager()
        metadata = {"nickname": "Alice", "location": "Earth"}

        did = manager.create_did(metadata=metadata)

        assert did.metadata == metadata
        assert did.metadata["nickname"] == "Alice"

    def test_created_timestamp(self):
        """Test creation timestamp is set"""
        manager = DIDManager()
        before = datetime.now(timezone.utc)

        did = manager.create_did()

        after = datetime.now(timezone.utc)
        assert before <= did.created_at <= after

    def test_keys_are_generated(self):
        """Test public and private keys are generated"""
        manager = DIDManager()
        did = manager.create_did()

        assert did.public_key is not None
        assert did.private_key is not None
        assert len(did.public_key) == 32  # Ed25519 public key size
        assert len(did.private_key) == 32  # Ed25519 private key size

    def test_sign_and_verify(self):
        """Test signing and verification"""
        manager = DIDManager()
        did = manager.create_did()

        message = b"Test message"
        signature = did.sign(message)

        assert signature is not None
        assert len(signature) == 64  # Ed25519 signature size
        assert did.verify(message, signature)

    def test_verify_wrong_signature(self):
        """Test verification fails with wrong signature"""
        manager = DIDManager()
        did = manager.create_did()

        message = b"Test message"
        signature = did.sign(message)

        # Modify signature
        bad_signature = signature[:-1] + b'\x00'

        assert not did.verify(message, bad_signature)

    def test_did_document_structure(self):
        """Test DID document conforms to W3C spec"""
        manager = DIDManager()
        did = manager.create_did()

        doc = did.to_did_document()

        # Check W3C required fields
        assert "@context" in doc
        assert "id" in doc
        assert "verificationMethod" in doc
        assert "authentication" in doc

        # Check Mycelix extensions
        assert "mycelix" in doc
        assert doc["mycelix"]["agentType"] in [t.value for t in AgentType]

    def test_different_dids_are_unique(self):
        """Test each DID creation produces unique identifier"""
        manager = DIDManager()

        did1 = manager.create_did()
        did2 = manager.create_did()

        assert did1.to_string() != did2.to_string()
        assert did1.identifier != did2.identifier


class TestDIDManager:
    """Test DIDManager functionality"""

    def test_create_did(self):
        """Test basic DID creation"""
        manager = DIDManager()

        did = manager.create_did()

        assert isinstance(did, MycelixDID)
        assert did.to_string().startswith("did:mycelix:")

    def test_store_and_resolve_did(self):
        """Test DID storage and resolution"""
        manager = DIDManager()

        did = manager.create_did(metadata={"test": "data"})
        did_string = did.to_string()

        # Resolve DID
        resolved = manager.resolve_did(did_string)

        assert resolved is not None
        assert resolved["did"] == did_string
        assert resolved["document"]["mycelix"]["metadata"]["test"] == "data"

    def test_resolve_nonexistent_did(self):
        """Test resolving a DID that doesn't exist"""
        manager = DIDManager()

        resolved = manager.resolve_did("did:mycelix:nonexistent")

        assert resolved is None

    def test_update_metadata(self):
        """Test updating DID metadata"""
        manager = DIDManager()

        did = manager.create_did(metadata={"version": 1})
        did_string = did.to_string()

        # Update metadata
        new_metadata = {"version": 2, "updated": True}
        manager.update_metadata(did_string, new_metadata)

        # Verify update
        resolved = manager.resolve_did(did_string)
        assert resolved["document"]["mycelix"]["metadata"] == new_metadata

    def test_list_dids(self):
        """Test listing all DIDs"""
        manager = DIDManager()

        did1 = manager.create_did()
        did2 = manager.create_did()

        dids = manager.list_dids()

        assert len(dids) >= 2
        assert did1.to_string() in dids
        assert did2.to_string() in dids

    def test_parse_did_valid(self):
        """Test parsing a valid DID string"""
        did_string = "did:mycelix:z6MkpTHR8VNsBxYAAWHut2Geadd9jSwuBV8xRoAnwWsdvktH"

        parsed = DIDManager.parse_did(did_string)

        assert parsed["scheme"] == "did"
        assert parsed["method"] == "mycelix"
        assert parsed["identifier"] == "z6MkpTHR8VNsBxYAAWHut2Geadd9jSwuBV8xRoAnwWsdvktH"

    def test_parse_did_invalid(self):
        """Test parsing invalid DID strings"""
        with pytest.raises(ValueError):
            DIDManager.parse_did("not-a-did")

        with pytest.raises(ValueError):
            DIDManager.parse_did("did:only_two_parts")

    def test_validate_did_valid(self):
        """Test DID validation for valid DIDs"""
        manager = DIDManager()
        did = manager.create_did()

        assert DIDManager.validate_did(did.to_string())

    def test_validate_did_invalid(self):
        """Test DID validation for invalid DIDs"""
        assert not DIDManager.validate_did("not-a-did")
        assert not DIDManager.validate_did("did:other:identifier")
        assert not DIDManager.validate_did("did:mycelix:")  # Empty identifier

    def test_private_key_not_stored(self):
        """Test private keys are not stored in storage"""
        manager = DIDManager()

        did = manager.create_did()
        resolved = manager.resolve_did(did.to_string())

        # Private key should NOT be in stored document
        assert "private_key" not in resolved
        assert "private_key" not in resolved["document"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
