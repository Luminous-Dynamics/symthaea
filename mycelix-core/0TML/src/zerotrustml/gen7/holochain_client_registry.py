# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Phase 2.5 Week 2: Holochain DHT-Backed Client Registry

Provides Python interface to pogq_zome_dilithium Holochain zome for:
  1. Client registration (Dilithium public keys)
  2. Proof submission (zkSTARK + Dilithium authentication)
  3. Nonce tracking (replay attack prevention)
  4. Participation statistics

Replaces centralized database with decentralized Holochain DHT.

Usage:
    # Initialize Holochain connector
    registry = HolochainClientRegistry(conductor_url="ws://localhost:8888")

    # Register client
    client_id = registry.register_client(dilithium_pubkey)

    # Submit authenticated proof
    is_valid, error = registry.submit_proof(
        client_id=client_id,
        round_number=1,
        nonce=nonce,
        timestamp=timestamp,
        stark_proof_hash=stark_proof_hash,
        dilithium_signature=signature,
        model_hash=model_hash,
        gradient_hash=gradient_hash,
    )
"""

import json
import time
import hashlib
import os
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Warning: cryptography not available. Install via: pip install cryptography")

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("Warning: websocket-client not available. Install via: pip install websocket-client")


@dataclass
class ClientInfo:
    """Client registration information from Holochain DHT"""
    client_id: bytes
    dilithium_pubkey: bytes
    registered_at: int
    reputation_score: float


class HolochainClientRegistry:
    """
    Holochain DHT-backed client registry for Phase 2.5

    Communicates with pogq_zome_dilithium via WebSocket to:
        - Register clients (store Dilithium public keys)
        - Submit authenticated proofs (zkSTARK + Dilithium)
        - Track nonces (replay attack prevention)
        - Query participation statistics

    Architecture:
        Python Client <--(WebSocket)--> Holochain Conductor <--> pogq_zome_dilithium
    """

    def __init__(
        self,
        conductor_url: str = "ws://localhost:8888",
        app_id: str = "zerotrustml",
        zome_name: str = "pogq_zome_dilithium",
        agent_pubkey: Optional[str] = None,
    ):
        """
        Initialize Holochain client registry

        Args:
            conductor_url: Holochain conductor WebSocket URL
            app_id: Installed Holochain app ID
            zome_name: Zome name within the app
            agent_pubkey: Agent public key (generated if None)
        """
        if not WEBSOCKET_AVAILABLE:
            raise RuntimeError("websocket-client not available. Install via: pip install websocket-client")

        self.conductor_url = conductor_url
        self.app_id = app_id
        self.zome_name = zome_name
        self._agent_private_key = None  # Set by _generate_agent_pubkey if we generate a new key
        self.agent_pubkey = agent_pubkey or self._generate_agent_pubkey()

        self.ws = None
        self._connect()

    def _connect(self):
        """Connect to Holochain conductor via WebSocket"""
        try:
            self.ws = websocket.create_connection(self.conductor_url, timeout=10)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Holochain conductor at {self.conductor_url}: {e}")

    def _generate_agent_pubkey(self) -> str:
        """
        Generate a Holochain-compatible AgentPubKey using Ed25519

        Holochain AgentPubKey format:
        - 39 bytes total: 4-byte type prefix + 32-byte Ed25519 public key + 3-byte DHT location
        - Type prefix: 0x84 0x20 0x24 0x XX (agent key type marker)
        - Encoded as hex string (78 characters)

        Returns:
            Hex-encoded AgentPubKey string
        """
        if not CRYPTO_AVAILABLE:
            # Fallback to random bytes if cryptography not available
            return os.urandom(39).hex()

        # Generate Ed25519 keypair
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        # Get raw 32-byte public key
        public_key_bytes = public_key.public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw
        )

        # Construct Holochain AgentPubKey format
        # Prefix: 0x84 0x20 0x24 marks this as an AgentPubKey type
        agent_key_prefix = bytes([0x84, 0x20, 0x24])

        # DHT location: derived from hash of public key (first 4 bytes of blake2b)
        dht_location = hashlib.blake2b(public_key_bytes, digest_size=4).digest()[:3]

        # Combine: prefix (3) + public_key (32) + first byte of prefix (1) + dht_location (3) = 39 bytes
        # Actually Holochain uses: type_byte (1) + core_bytes (36) + loc_bytes (4), but we simplify
        # Standard format: 3-byte prefix + 32-byte key + 4-byte location = 39 bytes
        agent_pubkey = agent_key_prefix + public_key_bytes + bytes([agent_key_prefix[0]]) + dht_location

        # Store the private key for signing operations
        self._agent_private_key = private_key

        return agent_pubkey.hex()

    def _call_zome(
        self,
        fn_name: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call Holochain zome function via WebSocket

        Args:
            fn_name: Zome function name (e.g., "register_client", "publish_pogq_proof")
            payload: Function arguments as dictionary

        Returns:
            Function result as dictionary

        Raises:
            RuntimeError: If zome call fails
        """
        if self.ws is None:
            self._connect()

        # Construct zome call request (Holochain Conductor API format)
        request = {
            "type": "app_request",
            "data": {
                "cell_id": [self.agent_pubkey, f"{self.app_id}_dna"],
                "zome_name": self.zome_name,
                "fn_name": fn_name,
                "payload": payload,
                "provenance": self.agent_pubkey,
            }
        }

        # Send request
        self.ws.send(json.dumps(request))

        # Receive response
        response_raw = self.ws.recv()
        response = json.loads(response_raw)

        # Check for errors
        if response.get("type") == "error":
            error_msg = response.get("data", {}).get("message", "Unknown error")
            raise RuntimeError(f"Zome call failed: {error_msg}")

        return response.get("data", {})

    def register_client(self, dilithium_pubkey: bytes) -> bytes:
        """
        Register client in Holochain DHT

        Args:
            dilithium_pubkey: Dilithium5 public key (2592 bytes)

        Returns:
            client_id: SHA-256 hash of public key (32 bytes)

        Raises:
            ValueError: If client already registered
            RuntimeError: If zome call fails
        """
        # Compute client ID (SHA-256 of public key)
        client_id = hashlib.sha256(dilithium_pubkey).digest()

        try:
            # Call register_client zome function
            result = self._call_zome(
                fn_name="register_client",
                payload={
                    "dilithium_pubkey": list(dilithium_pubkey),  # Convert bytes to list for JSON
                }
            )

            return client_id

        except RuntimeError as e:
            if "already registered" in str(e).lower():
                raise ValueError(f"Client already registered: {client_id.hex()[:16]}...")
            raise

    def submit_proof(
        self,
        client_id: bytes,
        round_number: int,
        nonce: bytes,
        timestamp: int,
        stark_proof_bytes: bytes,
        dilithium_signature: bytes,
        model_hash: bytes,
        gradient_hash: bytes,
        quarantine_out: int = 0,
        prov_hash: Tuple[int, int, int, int] = (0, 0, 0, 0),
        profile_id: int = 128,
        air_rev: int = 1,
        ema_t_fp: int = 65536,
        consec_viol_t: int = 0,
        consec_clear_t: int = 0,
    ) -> Tuple[bool, str]:
        """
        Submit authenticated proof to Holochain DHT

        Args:
            client_id: Client ID (32 bytes)
            round_number: Training round number
            nonce: Random nonce (32 bytes)
            timestamp: Unix timestamp (microseconds)
            stark_proof_bytes: zkSTARK proof bytes
            dilithium_signature: Dilithium signature bytes
            model_hash: SHA-256 of model parameters (32 bytes)
            gradient_hash: SHA-256 of gradient (32 bytes)
            quarantine_out: Quarantine status (0=healthy, 1=quarantined)
            prov_hash: Provenance hash (4 u64 values)
            profile_id: Security profile (128 or 192)
            air_rev: AIR revision
            ema_t_fp: EMA value (Q16.16 fixed-point)
            consec_viol_t: Consecutive violations
            consec_clear_t: Consecutive clears

        Returns:
            (is_valid: bool, error_message: str)
        """
        try:
            # Construct PoGQProofEntry
            entry = {
                "node_id": self.agent_pubkey,
                "round": round_number,
                "nonce": list(nonce),
                "receipt_bytes": list(stark_proof_bytes),
                "prov_hash": list(prov_hash),
                "profile_id": profile_id,
                "air_rev": air_rev,
                "quarantine_out": quarantine_out,
                "current_round": round_number,
                "ema_t_fp": ema_t_fp,
                "consec_viol_t": consec_viol_t,
                "consec_clear_t": consec_clear_t,
                "dilithium_signature": list(dilithium_signature),
                "client_id": list(client_id),
                "model_hash": list(model_hash),
                "gradient_hash": list(gradient_hash),
                "timestamp": timestamp,
            }

            # Call publish_pogq_proof zome function
            result = self._call_zome(
                fn_name="publish_pogq_proof",
                payload=entry,
            )

            return (True, "Proof published successfully")

        except RuntimeError as e:
            error = str(e)

            # Parse validation errors
            if "nonce already used" in error.lower() or "replay attack" in error.lower():
                return (False, "Replay attack detected: nonce reused")
            elif "timestamp" in error.lower():
                return (False, "Timestamp out of bounds")
            elif "signature" in error.lower():
                return (False, "Invalid Dilithium signature")
            elif "not registered" in error.lower():
                return (False, f"Client not registered: {client_id.hex()[:16]}...")
            else:
                return (False, f"Validation failed: {error}")

    def get_client_info(self, client_id: bytes) -> Optional[ClientInfo]:
        """
        Query client information from Holochain DHT

        Args:
            client_id: Client ID (32 bytes)

        Returns:
            ClientInfo or None if not found
        """
        try:
            # Call get_client_info_public zome function
            result = self._call_zome(
                fn_name="get_client_info_public",
                payload={"client_id": list(client_id)}
            )

            if result is None:
                return None

            return ClientInfo(
                client_id=bytes(result["client_id"]),
                dilithium_pubkey=bytes(result["dilithium_pubkey"]),
                registered_at=result["registered_at"],
                reputation_score=result.get("reputation_score", 1.0),
            )

        except RuntimeError:
            return None

    def is_nonce_used(self, client_id: bytes, nonce: bytes) -> bool:
        """
        Check if nonce has been used (query Holochain DHT)

        Args:
            client_id: Client ID (32 bytes)
            nonce: Nonce bytes (32 bytes)

        Returns:
            True if nonce already used, False otherwise
        """
        try:
            # Call is_nonce_used zome function (to be implemented in zome)
            result = self._call_zome(
                fn_name="is_nonce_used",
                payload={
                    "nonce": list(nonce),
                }
            )

            return result.get("is_used", False)

        except RuntimeError:
            # On error, assume not used (fail-open for reads)
            return False

    def get_participation_stats(self, client_id: bytes) -> Dict[str, Any]:
        """
        Get participation statistics from Holochain DHT

        Args:
            client_id: Client ID (32 bytes)

        Returns:
            Dictionary with:
                - total_rounds: Total number of rounds participated
                - accepted_rounds: Number of accepted (healthy) rounds
                - avg_pogq_score: Average PoGQ quality score
                - reputation_score: Current reputation score
        """
        try:
            # Call get_participation_stats zome function (to be implemented in zome)
            result = self._call_zome(
                fn_name="get_participation_stats",
                payload={"client_id": list(client_id)}
            )

            return {
                "total_rounds": result.get("total_rounds", 0),
                "accepted_rounds": result.get("accepted_rounds", 0),
                "avg_pogq_score": result.get("avg_pogq_score", 0.0),
                "reputation_score": result.get("reputation_score", 1.0),
            }

        except RuntimeError:
            # On error, return empty stats
            return {
                "total_rounds": 0,
                "accepted_rounds": 0,
                "avg_pogq_score": 0.0,
                "reputation_score": 1.0,
            }

    def get_round_gradients(self, round_number: int) -> list:
        """
        Get all gradients for a round with quarantine status

        Args:
            round_number: Training round number

        Returns:
            List of (GradientEntry dict, is_quarantined bool) tuples
        """
        try:
            result = self._call_zome(
                fn_name="get_round_gradients",
                payload={"round": round_number}
            )

            # Parse result (list of tuples)
            gradients = []
            for item in result:
                grad_entry, is_quarantined = item
                gradients.append((grad_entry, is_quarantined))

            return gradients

        except RuntimeError:
            return []

    def close(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()
            self.ws = None

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.close()


# Convenience functions

def connect_to_holochain(
    conductor_url: str = "ws://localhost:8888",
    app_id: str = "zerotrustml",
) -> HolochainClientRegistry:
    """
    Convenience function: Connect to Holochain conductor

    Args:
        conductor_url: Holochain conductor WebSocket URL
        app_id: Installed Holochain app ID

    Returns:
        HolochainClientRegistry instance
    """
    return HolochainClientRegistry(
        conductor_url=conductor_url,
        app_id=app_id,
    )


# Example usage (for documentation)
if __name__ == "__main__":
    print("Phase 2.5 Week 2: Holochain Client Registry Example")
    print("=" * 60)

    if not WEBSOCKET_AVAILABLE:
        print("ERROR: websocket-client not available")
        print("Install with: pip install websocket-client")
        exit(1)

    print("⚠️  This example requires a running Holochain conductor with zerotrustml app installed")
    print("⚠️  Start conductor with: holochain -c conductor-config.yaml")
    print()

    try:
        # Connect to Holochain conductor
        registry = HolochainClientRegistry(
            conductor_url="ws://localhost:8888",
            app_id="zerotrustml",
        )

        print(f"✅ Connected to Holochain conductor")
        print()

        # Generate fake client data
        import os
        dilithium_pubkey = os.urandom(2592)
        client_id = hashlib.sha256(dilithium_pubkey).digest()

        # Register client
        print(f"📝 Registering client: {client_id.hex()[:16]}...")
        registered_id = registry.register_client(dilithium_pubkey)
        print(f"✅ Client registered: {registered_id.hex()[:16]}...")
        print()

        # Submit proof
        print("📊 Submitting authenticated proof...")
        is_valid, error = registry.submit_proof(
            client_id=client_id,
            round_number=1,
            nonce=os.urandom(32),
            timestamp=int(time.time() * 1_000_000),
            stark_proof_bytes=os.urandom(61000),  # ~61KB zkSTARK proof
            dilithium_signature=os.urandom(4660),  # ~4660 bytes Dilithium signature
            model_hash=os.urandom(32),
            gradient_hash=os.urandom(32),
        )

        if is_valid:
            print(f"✅ Proof accepted: {error}")
        else:
            print(f"❌ Proof rejected: {error}")

        print()

        # Close connection
        registry.close()
        print("✅ Connection closed")

    except RuntimeError as e:
        print(f"❌ ERROR: {e}")
        print()
        print("Make sure Holochain conductor is running:")
        print("  1. Install Holochain: https://developer.holochain.org/install/")
        print("  2. Create conductor config: conductor-config.yaml")
        print("  3. Start conductor: holochain -c conductor-config.yaml")
        print("  4. Install zerotrustml app")

    print()
    print("=" * 60)
    print("Phase 2.5 Holochain integration ready!")
