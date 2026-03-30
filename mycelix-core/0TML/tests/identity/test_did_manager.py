# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
        assert doc["mycelix"]["agentType"] in [t.value for t in AgentTyp