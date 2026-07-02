# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Phase 2.5: Authenticated Gradient Proof Integration

Combines zkSTARK integrity proofs (Phase 2) with Dilithium authentication (Phase 2.5)
to create zk-DASTARK hybrid system for federated learning.

Usage:
    # Client-side: Generate authenticated proof
    client = AuthenticatedGradientClient(client_keypair)
    auth_proof = client.generate_proof(gradient, model_params, data, labels, round_number)

    # Coordinator-side: Verify authenticated proof
    coordinator = AuthenticatedGradientCoordinator()
    coordinator.register_client(client_id, public_key)
    is_valid, error = coordinator.verify_proof(auth_proof, round_number)
"""

import os
import time
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend

# Import Phase 2 zkSTARK functionality
try:
    import gen7_zkstark
    DILITHIUM_AVAILABLE = True
except ImportError:
    DILITHIUM_AVAILABLE = False
    print("Warning: gen7_zkstark not available. Install via maturin build in gen7-zkstark/bindings/")


@dataclass
class AuthenticatedProof:
    """
    Authenticated gradient proof combining zkSTARK + Dilithium signature

    Attributes:
        stark_proof: zkSTARK proof bytes (61KB)
        signature: Dilithium5 signature bytes (4595 bytes exact)
        client_id: SHA-256 hash of client's public key (32 bytes)
        round_number: Training round number
        timestamp: Unix timestamp when proof was generated
        nonce: 32-byte random value for replay protection
        model_hash: SHA-256 hash of initial model parameters (32 bytes)
        gradient_hash: SHA-256 hash of computed gradient (32 bytes)

    Total size: ~65.8KB (4.7KB overhead from Phase 2)
    """
    stark_proof: bytes
    signature: bytes
    client_id: bytes
    round_number: int
    timestamp: int
    nonce: bytes
    model_hash: bytes
    gradient_hash: bytes

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for network transmission"""
        return {
            'stark_proof': self.stark_proof.hex(),
            'signature': self.signature.hex(),
            'client_id': self.client_id.hex(),
            'round_number': self.round_number,
            'timestamp': self.timestamp,
            'nonce': self.nonce.hex(),
            'model_hash': self.model_hash.hex(),
            'gradient_hash': self.gradient_hash.hex(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuthenticatedProof':
        """Deserialize from dictionary"""
        return cls(
            stark_proof=bytes.fromhex(data['stark_proof']),
            signature=bytes.fromhex(data['signature']),
            client_id=bytes.fromhex(data['client_id']),
            round_number=data['round_number'],
            timestamp=data['timestamp'],
            nonce=bytes.fromhex(data['nonce']),
            model_hash=bytes.fromhex(data['model_hash']),
            gradient_hash=bytes.fromhex(data['gradient_hash']),
        )

    def size_bytes(self) -> int:
        """Get total proof size in bytes"""
        return (
            len(self.stark_proof) +
            len(self.signature) +
            len(self.client_id) +
            8 +  # round_number
            8 +  # timestamp
            len(self.nonce) +
            len(self.model_hash) +
            len(self.gradient_hash)
        )


class AuthenticatedGradientClient:
    """
    Client-side interface for generating authenticated gradient proofs

    Handles:
        1. zkSTARK proof generation (gradient computation integrity)
        2. Dilithium signature generation (client authentication)
        3. Proof packaging and serialization
    """

    def __init__(self, keypair=None):
        """
        Initialize client with Dilithium keypair

        Args:
            keypair: DilithiumKeypair instance (creates new if None)
        """
        if not DILITHIUM_AVAILABLE:
            raise RuntimeError("gen7_zkstark module not available")

        if keypair is None:
            self.keypair = gen7_zkstark.DilithiumKeypair()
        else:
            self.keypair = keypair

        self.client_id = bytes(self.keypair.get_client_id())

    def generate_proof(
        self,
        gradient: list,
        model_params: list,
        local_data: list,
        local_labels: list,
        round_number: int,
        num_samples: int = None,
        input_dim: int = None,
        num_classes: int = None,
        epochs: int = 1,
        learning_rate: float = 0.01,
    ) -> AuthenticatedProof:
        """
        Generate authenticated gradient proof

        Args:
            gradient: Computed gradient (flat list of floats)
            model_params: Initial model parameters
            local_data: Private training data (flattened)
            local_labels: Private labels
            round_number: Current training round
            num_samples: Number of samples (auto-detect if None)
            input_dim: Input dimension (auto-detect if None)
            num_classes: Number of classes (auto-detect if None)
            epochs: Number of local training epochs
            learning_rate: Learning rate used

        Returns:
            AuthenticatedProof with zkSTARK + signature
        """
        # Auto-detect dimensions if not provided
        if num_samples is None:
            num_samples = len(local_labels)
        if input_dim is None:
            input_dim = len(local_data) // num_samples
        if num_classes is None:
            num_classes = max(local_labels) + 1

        # Step 1: Generate zkSTARK proof (Phase 2)
        stark_proof_bytes = gen7_zkstark.prove_gradient_zkstark(
            model_params=model_params,
            gradient=gradient,
            local_data=local_data,
            local_labels=local_labels,
            num_samples=num_samples,
            input_dim=input_dim,
            num_classes=num_classes,
            epochs=epochs,
            learning_rate=learning_rate,
        )

        # Step 2: Compute model and gradient hashes (Phase 2.5 security enhancement)
        model_hash = bytes(gen7_zkstark.hash_model_params_py(model_params))
        gradient_hash = bytes(gen7_zkstark.hash_gradient_py(gradient))

        # Step 3: Generate nonce and timestamp
        nonce = bytes(gen7_zkstark.generate_nonce())
        timestamp = gen7_zkstark.current_timestamp()

        # Step 4: Construct message to sign (with domain separation and hash binding)
        # Message = SHA-256(domain_tag || protocol_version || client_id || round_number ||
        #                   timestamp || nonce || model_hash || gradient_hash || stark_proof)
        message = gen7_zkstark.AuthenticatedGradientProof.construct_message(
            stark_proof=stark_proof_bytes,
            client_id=self.client_id,
            round_number=round_number,
            timestamp=timestamp,
            nonce=nonce,
            model_hash=model_hash,
            gradient_hash=gradient_hash,
        )

        # Step 5: Sign with Dilithium
        signature = bytes(self.keypair.sign(message))

        # Step 6: Package into authenticated proof
        return AuthenticatedProof(
            stark_proof=stark_proof_bytes,
            signature=signature,
            client_id=self.client_id,
            round_number=round_number,
            timestamp=timestamp,
            nonce=nonce,
            model_hash=model_hash,
            gradient_hash=gradient_hash,
        )

    def get_public_key(self) -> bytes:
        """Get Dilithium public key for registration"""
        return bytes(self.keypair.get_public_key())

    def get_client_id(self) -> bytes:
        """Get client ID (SHA-256 of public key)"""
        return self.client_id

    def export_keypair(self, password: str = None) -> Dict[str, bytes]:
        """
        Export keypair for storage

        Args:
            password: Optional password for AES-GCM encryption of the secret key

        Returns:
            Dict with 'public_key' and either:
                - 'secret_key' (unencrypted bytes) if no password
                - 'encrypted_secret_key', 'salt', 'nonce' if password provided
        """
        public_key = self.get_public_key()
        secret_key = bytes(self.keypair.get_secret_key())

        if password is None:
            return {
                'public_key': public_key,
                'secret_key': secret_key,
            }

        # Derive encryption key from password using Scrypt
        salt = os.urandom(16)
        kdf = Scrypt(
            salt=salt,
            length=32,
            n=2**14,
            r=8,
            p=1,
            backend=default_backend()
        )
        encryption_key = kdf.derive(password.encode('utf-8'))

        # Encrypt secret key with AES-256-GCM
        nonce = os.urandom(12)  # 96-bit nonce for AES-GCM
        aesgcm = AESGCM(encryption_key)
        encrypted_secret_key = aesgcm.encrypt(nonce, secret_key, associated_data=public_key)

        return {
            'public_key': public_key,
            'encrypted_secret_key': encrypted_secret_key,
            'salt': salt,
            'nonce': nonce,
        }

    @classmethod
    def from_keypair_bytes(cls, public_key: bytes, secret_key: bytes) -> 'AuthenticatedGradientClient':
        """Load client from exported keypair (unencrypted)"""
        if not DILITHIUM_AVAILABLE:
            raise RuntimeError("gen7_zkstark module not available")

        keypair = gen7_zkstark.DilithiumKeypair.from_bytes(
            public_key=list(public_key),
            secret_key=list(secret_key),
        )
        return cls(keypair=keypair)

    @classmethod
    def from_encrypted_keypair(
        cls,
        public_key: bytes,
        encrypted_secret_key: bytes,
        salt: bytes,
        nonce: bytes,
        password: str,
    ) -> 'AuthenticatedGradientClient':
        """
        Load client from password-encrypted keypair

        Args:
            public_key: Dilithium5 public key bytes
            encrypted_secret_key: AES-GCM encrypted secret key
            salt: Scrypt salt used for key derivation
            nonce: AES-GCM nonce
            password: Password to decrypt the secret key

        Returns:
            AuthenticatedGradientClient instance

        Raises:
            ValueError: If decryption fails (wrong password or corrupted data)
        """
        if not DILITHIUM_AVAILABLE:
            raise RuntimeError("gen7_zkstark module not available")

        # Derive decryption key from password using same Scrypt parameters
        kdf = Scrypt(
            salt=salt,
            length=32,
            n=2**14,
            r=8,
            p=1,
            backend=default_backend()
        )
        decryption_key = kdf.derive(password.encode('utf-8'))

        # Decrypt secret key with AES-256-GCM
        aesgcm = AESGCM(decryption_key)
        try:
            secret_key = aesgcm.decrypt(nonce, encrypted_secret_key, associated_data=public_key)
        except Exception as e:
            raise ValueError(f"Decryption failed (wrong password or corrupted data): {e}")

        return cls.from_keypair_bytes(public_key, secret_key)


class AuthenticatedGradientCoordinator:
    """
    Coordinator-side interface for verifying authenticated gradient proofs

    Handles:
        1. Client registration (storing public keys)
        2. Nonce tracking (replay attack prevention)
        3. zkSTARK + Dilithium signature verification
        4. Proof validation and acceptance

    Database Backend:
        Uses ClientRegistry for persistent storage of client data and nonces.
        Supports SQLite (dev) and PostgreSQL (prod).
    """

    def __init__(
        self,
        max_timestamp_delta: int = 300,
        database_url: str = "sqlite:///ztml_clients.db",
        use_database: bool = True,
    ):
        """
        Initialize coordinator

        Args:
            max_timestamp_delta: Maximum allowed timestamp difference in seconds (default: 5 minutes)
            database_url: Database connection URL (default: SQLite file)
            use_database: Use database-backed registry (recommended). If False, uses in-memory dict (legacy)
        """
        if not DILITHIUM_AVAILABLE:
            raise RuntimeError("gen7_zkstark module not available")

        self.max_timestamp_delta = max_timestamp_delta
        self.current_round: Optional[int] = None

        # Initialize registry (database-backed or legacy in-memory)
        if use_database:
            from .client_registry import ClientRegistry
            self.registry = ClientRegistry(
                database_url=database_url,
                auto_cleanup=True,
                nonce_max_age=3600,  # 1 hour
            )
            self.use_database = True
        else:
            # Legacy in-memory mode (for backward compatibility)
            self.client_registry: Dict[bytes, bytes] = {}  # client_id -> public_key
            self.used_nonces: set = set()  # Track used nonces
            self.use_database = False
            self.registry = None

    def register_client(self, client_id: bytes, public_key: bytes):
        """
        Register a client's public key

        Args:
            client_id: SHA-256 hash of public key (32 bytes)
            public_key: Dilithium5 public key bytes (2592 bytes)
        """
        if self.use_database:
            # Database mode: Use ClientRegistry
            try:
                self.registry.register_client(public_key)
            except ValueError:
                # Already registered - this is acceptable
                pass
        else:
            # Legacy in-memory mode
            self.client_registry[client_id] = public_key

    def set_round(self, round_number: int):
        """
        Set current training round

        Args:
            round_number: Current round number
        """
        if self.current_round is not None and round_number > self.current_round:
            # New round: clear used nonces from previous round
            self.used_nonces.clear()

        self.current_round = round_number

    def verify_proof(
        self,
        auth_proof: AuthenticatedProof,
        gradient_commitment: bytes = None,
    ) -> Tuple[bool, str]:
        """
        Verify authenticated gradient proof

        Checks:
            1. Client is registered
            2. Round number matches current round
            3. Timestamp is fresh (within ±max_timestamp_delta)
            4. Nonce hasn't been used (replay protection)
            5. Dilithium signature is valid
            6. zkSTARK proof is valid

        Args:
            auth_proof: Authenticated proof to verify
            gradient_commitment: Optional expected gradient hash (verified by zkSTARK)

        Returns:
            (is_valid: bool, error_message: str)
        """
        # Check 1: Client registered?
        if self.use_database:
            # Database mode: Use registry
            client = self.registry.get_client(auth_proof.client_id)
            if client is None:
                return False, f"Unknown client: {auth_proof.client_id.hex()[:16]}..."
            client_public_key = client.public_key
        else:
            # Legacy in-memory mode
            if auth_proof.client_id not in self.client_registry:
                return False, f"Unknown client: {auth_proof.client_id.hex()[:16]}..."
            client_public_key = self.client_registry[auth_proof.client_id]

        # Check 2: Nonce not already used (replay attack prevention)
        if self.use_database:
            # Database mode: Check via registry
            if self.registry.is_nonce_used(auth_proof.client_id, auth_proof.nonce):
                return False, "Nonce already used (replay attack detected)"
        else:
            # Legacy in-memory mode
            nonce_key = auth_proof.nonce
            if nonce_key in self.used_nonces:
                return False, "Nonce already used (replay attack detected)"

        # Check 3-5: Round, timestamp, signature (via Rust)
        rust_proof = gen7_zkstark.AuthenticatedGradientProof(
            stark_proof=auth_proof.stark_proof,
            signature=auth_proof.signature,
            client_id=list(auth_proof.client_id),
            round_number=auth_proof.round_number,
            timestamp=auth_proof.timestamp,
            nonce=list(auth_proof.nonce),
            model_hash=list(auth_proof.model_hash),
            gradient_hash=list(auth_proof.gradient_hash),
        )

        is_valid, error = rust_proof.verify(
            current_round=self.current_round,
            client_public_key=list(client_public_key),
            max_timestamp_delta=self.max_timestamp_delta,
        )

        if not is_valid:
            return False, error

        # Check 6: zkSTARK proof validity
        try:
            result = gen7_zkstark.verify_gradient_zkstark(auth_proof.stark_proof)

            if gradient_commitment is not None:
                # Verify gradient hash matches expected
                if result['gradient_hash'] != gradient_commitment:
                    return False, "Gradient commitment mismatch"

            # All checks passed! Mark nonce as used
            if self.use_database:
                # Database mode: Persist nonce
                self.registry.mark_nonce_used(
                    auth_proof.client_id,
                    auth_proof.nonce,
                    auth_proof.round_number,
                    auth_proof.timestamp,
                )
                # Update participation statistics
                self.registry.update_participation(auth_proof.client_id)
            else:
                # Legacy in-memory mode
                self.used_nonces.add(nonce_key)

            return True, "Proof verified successfully"

        except Exception as e:
            return False, f"zkSTARK verification failed: {str(e)}"

    def get_client_count(self) -> int:
        """Get number of registered clients"""
        if self.use_database:
            return self.registry.get_client_count()
        else:
            return len(self.client_registry)

    def clear_nonces(self):
        """Clear used nonces (e.g., at round boundary)"""
        if self.use_database:
            self.registry.clear_all_nonces()
        else:
            self.used_nonces.clear()

    def cleanup_old_nonces(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up expired nonces (database mode only)

        Args:
            max_age_seconds: Maximum nonce age in seconds (default: 1 hour)

        Returns:
            Number of nonces deleted
        """
        if self.use_database:
            return self.registry.cleanup_old_nonces(max_age_seconds)
        else:
            # Legacy mode: no cleanup needed (in-memory clears at round boundary)
            return 0


# Convenience functions for quick usage

def generate_authenticated_proof(
    client_keypair,
    gradient: list,
    model_params: list,
    local_data: list,
    local_labels: list,
    round_number: int,
    **kwargs
) -> AuthenticatedProof:
    """
    Convenience function: Generate authenticated proof in one call

    See AuthenticatedGradientClient.generate_proof() for full argument list
    """
    client = AuthenticatedGradientClient(keypair=client_keypair)
    return client.generate_proof(
        gradient=gradient,
        model_params=model_params,
        local_data=local_data,
        local_labels=local_labels,
        round_number=round_number,
        **kwargs
    )


def verify_authenticated_proof(
    coordinator: AuthenticatedGradientCoordinator,
    auth_proof: AuthenticatedProof,
) -> Tuple[bool, str]:
    """
    Convenience function: Verify authenticated proof in one call
    """
    return coordinator.verify_proof(auth_proof)


# Example usage (for documentation)
if __name__ == "__main__":
    print("Phase 2.5: Authenticated Gradient Proof Example")
    print("=" * 60)

    if not DILITHIUM_AVAILABLE:
        print("ERROR: gen7_zkstark module not available")
        print("Build with: cd gen7-zkstark/bindings && maturin build --release")
        exit(1)

    # Initialize coordinator
    coordinator = AuthenticatedGradientCoordinator(max_timestamp_delta=300)
    coordinator.set_round(round_number=1)

    # Create client with new keypair
    client = AuthenticatedGradientClient()

    # Register client with coordinator
    coordinator.register_client(
        client_id=client.get_client_id(),
        public_key=client.get_public_key(),
    )

    print(f"✅ Client registered: {client.get_client_id().hex()[:16]}...")
    print(f"✅ Coordinator has {coordinator.get_client_count()} clients")
    print()

    # Generate dummy training data
    num_samples = 10
    input_dim = 5
    num_classes = 3

    model_params = [0.1] * (input_dim * num_classes)
    gradient = [0.01] * (input_dim * num_classes)
    local_data = [0.5] * (num_samples * input_dim)
    local_labels = [i % num_classes for i in range(num_samples)]

    print("📊 Generating authenticated proof...")
    start_time = time.time()

    auth_proof = client.generate_proof(
        gradient=gradient,
        model_params=model_params,
        local_data=local_data,
        local_labels=local_labels,
        round_number=1,
        num_samples=num_samples,
        input_dim=input_dim,
        num_classes=num_classes,
        epochs=1,
        learning_rate=0.01,
    )

    proof_time_ms = (time.time() - start_time) * 1000
    proof_size_kb = auth_proof.size_bytes() / 1024

    print(f"✅ Proof generated in {proof_time_ms:.1f}ms")
    print(f"✅ Proof size: {proof_size_kb:.1f}KB")
    print()

    # Verify proof
    print("🔍 Verifying authenticated proof...")
    start_time = time.time()

    is_valid, error = coordinator.verify_proof(auth_proof)

    verify_time_ms = (time.time() - start_time) * 1000

    if is_valid:
        print(f"✅ Proof verified in {verify_time_ms:.1f}ms")
        print(f"✅ Message: {error}")
    else:
        print(f"❌ Verification failed: {error}")

    print()
    print("=" * 60)
    print("Phase 2.5 integration complete!")
    print(f"Total overhead: +{(proof_size_kb - 61):.1f}KB from Phase 2")
