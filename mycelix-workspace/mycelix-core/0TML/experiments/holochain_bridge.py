# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Holochain Bridge for 0TML Zero-Knowledge Federated Learning

This module provides integration between 0TML experiments and Holochain DHT,
enabling decentralized storage and verification of VSV-STARK zero-knowledge proofs.

Architecture:
- HolochainBridge: Main class for DHT interaction
- ProofSubmitter: Handles proof packaging and submission
- ProofVerifier: Retrieves and validates peer proofs

Created: November 9, 2025
Status: Ready for testing (blocked by zome build)
"""

import json
import hashlib
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProofMetadata:
    """Metadata for a VSV-STARK proof stored in Holochain DHT"""
    node_id: str
    round_number: int
    proof_hash: str
    public_hash: str
    guest_image_id: str
    quarantine_decision: int
    timestamp: int
    action_hash: Optional[str] = None  # Set after DHT submission


class HolochainBridge:
    """
    Bridge between 0TML experiments and Holochain DHT.

    Enables:
    1. Submitting VSV-STARK proofs to DHT after PoGQ decisions
    2. Retrieving peer proofs for verification
    3. Querying proof history and consensus state

    Example:
        bridge = HolochainBridge("http://localhost:9888")

        # After generating proof with VSV-STARK
        action_hash = bridge.submit_proof(
            node_id="node_0",
            round_number=8,
            proof_path=Path("proofs/proof.bin"),
            public_path=Path("proofs/public_echo.json"),
            journal_path=Path("proofs/journal.bin")
        )

        # Verify peer's proof
        result = bridge.verify_peer_proof("node_1", 8)
        if result["valid"]:
            print(f"Peer decision: {result['quarantine_decision']}")
    """

    def __init__(self, conductor_url: str = "http://localhost:9888"):
        """
        Initialize Holochain bridge.

        Args:
            conductor_url: URL of Holochain conductor admin interface
                          Default: http://localhost:9888 (first conductor)
        """
        self.conductor_url = conductor_url
        self.app_id = "pogq_zkfl"  # App ID for the installed hApp
        self.zome_name = "pogq_proof_validation"

        logger.info(f"Initialized HolochainBridge for {conductor_url}")

    def _call_zome(
        self,
        function_name: str,
        payload: Dict
    ) -> Dict:
        """
        Call a zome function via Holochain conductor API.

        Args:
            function_name: Name of the zome function to call
            payload: Input parameters for the function

        Returns:
            Response from the zome function

        Raises:
            requests.RequestException: If HTTP request fails
            ValueError: If zome returns an error
        """
        url = f"{self.conductor_url}/call"

        request_body = {
            "app_id": self.app_id,
            "zome_name": self.zome_name,
            "fn_name": function_name,
            "payload": payload
        }

        logger.debug(f"Calling zome function: {function_name}")

        try:
            response = requests.post(url, json=request_body, timeout=30)
            response.raise_for_status()

            result = response.json()

            # Check for zome-level errors
            if "error" in result:
                raise ValueError(f"Zome error: {result['error']}")

            return result

        except requests.RequestException as e:
            logger.error(f"Failed to call zome function {function_name}: {e}")
            raise

    def _compute_hash(self, data: bytes) -> str:
        """
        Compute SHA-256 hash of data.

        Args:
            data: Binary data to hash

        Returns:
            Hex string of SHA-256 hash (64 characters)
        """
        return hashlib.sha256(data).hexdigest()

    def submit_proof(
        self,
        node_id: str,
        round_number: int,
        proof_path: Path,
        public_path: Path,
        journal_path: Path,
        guest_image_id: str = "47353ec9f0a9f4b8d72e4f84c2a8e3f5d1b2c3a4e5f6d7e8f9a0b1c2d3e4f5a6"
    ) -> str:
        """
        Submit a VSV-STARK proof to the Holochain DHT.

        Args:
            node_id: Identifier for this node (e.g., "node_0")
            round_number: FL round number (e.g., 8)
            proof_path: Path to proof.bin file (~221 KB)
            public_path: Path to public_echo.json file
            journal_path: Path to journal.bin file (~22 bytes)
            guest_image_id: METHOD_ID of the zkVM guest program

        Returns:
            ActionHash of the stored proof entry

        Raises:
            FileNotFoundError: If any proof file doesn't exist
            ValueError: If proof validation fails
        """
        # Read proof files
        if not proof_path.exists():
            raise FileNotFoundError(f"Proof file not found: {proof_path}")
        if not public_path.exists():
            raise FileNotFoundError(f"Public inputs file not found: {public_path}")
        if not journal_path.exists():
            raise FileNotFoundError(f"Journal file not found: {journal_path}")

        proof_binary = proof_path.read_bytes()
        journal_binary = journal_path.read_bytes()
        public_inputs = public_path.read_text()

        # Compute hashes
        proof_hash = self._compute_hash(proof_binary)
        public_hash = self._compute_hash(public_inputs.encode())

        # Extract quarantine decision from journal
        # Journal format: MessagePack with single u8 value (0 or 1)
        if len(journal_binary) < 1:
            raise ValueError("Journal binary is empty")

        # Simple extraction: last byte should be the decision (0 or 1)
        # TODO: Proper MessagePack deserialization if format becomes complex
        quarantine_decision = journal_binary[-1]
        if quarantine_decision not in (0, 1):
            raise ValueError(f"Invalid quarantine decision: {quarantine_decision}")

        # Get current timestamp (microseconds since epoch)
        import time
        timestamp = int(time.time() * 1_000_000)

        # Create proof entry
        proof_entry = {
            "node_id": node_id,
            "round_number": round_number,
            "proof_binary": list(proof_binary),  # Convert bytes to list of ints
            "journal_binary": list(journal_binary),
            "public_inputs": public_inputs,
            "proof_hash": proof_hash,
            "public_hash": public_hash,
            "guest_image_id": guest_image_id,
            "quarantine_decision": quarantine_decision,
            "timestamp": timestamp
        }

        logger.info(
            f"Submitting proof: node={node_id}, round={round_number}, "
            f"decision={quarantine_decision}, proof_size={len(proof_binary)}"
        )

        # Call store_proof zome function
        result = self._call_zome("store_proof", proof_entry)

        action_hash = result.get("action_hash")
        if not action_hash:
            raise ValueError("No action hash returned from store_proof")

        logger.info(f"Proof stored successfully: {action_hash}")

        return action_hash

    def get_proof(self, action_hash: str) -> Optional[Dict]:
        """
        Retrieve a proof entry by its ActionHash.

        Args:
            action_hash: Hash returned from submit_proof()

        Returns:
            Proof entry dict if found, None otherwise
        """
        logger.debug(f"Retrieving proof: {action_hash}")

        result = self._call_zome("get_proof", {"hash": action_hash})

        if result.get("record") is None:
            logger.warning(f"Proof not found: {action_hash}")
            return None

        return result["record"]

    def find_proof(
        self,
        node_id: str,
        round_number: int
    ) -> Optional[str]:
        """
        Find a proof by node ID and round number.

        Args:
            node_id: Node identifier (e.g., "node_0")
            round_number: FL round number

        Returns:
            ActionHash if found, None otherwise

        Note:
            Current implementation may be slow - requires link indexing
            in the zome for efficient lookup.
        """
        logger.debug(f"Finding proof: node={node_id}, round={round_number}")

        result = self._call_zome(
            "find_proof",
            {"node_id": node_id, "round_number": round_number}
        )

        action_hash = result.get("action_hash")

        if action_hash is None:
            logger.warning(f"Proof not found: node={node_id}, round={round_number}")

        return action_hash

    def verify_peer_proof(
        self,
        peer_id: str,
        round_number: int
    ) -> Dict:
        """
        Retrieve and verify a peer's proof.

        Args:
            peer_id: Peer node identifier
            round_number: FL round number

        Returns:
            Dict with keys:
                - valid: bool (whether proof was found and valid)
                - quarantine_decision: int (0 or 1, if valid)
                - proof_hash: str (if valid)
                - error: str (if invalid)
        """
        try:
            # Find the proof
            action_hash = self.find_proof(peer_id, round_number)

            if action_hash is None:
                return {
                    "valid": False,
                    "error": f"No proof found for {peer_id} at round {round_number}"
                }

            # Retrieve the proof entry
            record = self.get_proof(action_hash)

            if record is None:
                return {
                    "valid": False,
                    "error": f"Failed to retrieve proof {action_hash}"
                }

            # Extract proof entry from record
            entry = record.get("entry", {})

            return {
                "valid": True,
                "quarantine_decision": entry.get("quarantine_decision"),
                "proof_hash": entry.get("proof_hash"),
                "guest_image_id": entry.get("guest_image_id"),
                "node_id": entry.get("node_id"),
                "round_number": entry.get("round_number")
            }

        except Exception as e:
            logger.error(f"Error verifying peer proof: {e}")
            return {
                "valid": False,
                "error": str(e)
            }

    def get_consensus_state(
        self,
        round_number: int,
        node_ids: List[str]
    ) -> Dict:
        """
        Get consensus state for a round across multiple nodes.

        Args:
            round_number: FL round number
            node_ids: List of node identifiers to query

        Returns:
            Dict with keys:
                - round: int
                - total_nodes: int
                - nodes_with_proofs: int
                - quarantine_votes: int (number voting to quarantine)
                - no_quarantine_votes: int
                - consensus_reached: bool
                - consensus_decision: int (0 or 1, if consensus reached)
                - node_decisions: Dict[str, int] (node_id -> decision)
        """
        node_decisions = {}
        quarantine_count = 0
        no_quarantine_count = 0

        for node_id in node_ids:
            result = self.verify_peer_proof(node_id, round_number)

            if result["valid"]:
                decision = result["quarantine_decision"]
                node_decisions[node_id] = decision

                if decision == 1:
                    quarantine_count += 1
                else:
                    no_quarantine_count += 1

        total_votes = quarantine_count + no_quarantine_count

        # Simple majority consensus
        consensus_reached = total_votes > 0
        consensus_decision = 1 if quarantine_count > no_quarantine_count else 0

        return {
            "round": round_number,
            "total_nodes": len(node_ids),
            "nodes_with_proofs": total_votes,
            "quarantine_votes": quarantine_count,
            "no_quarantine_votes": no_quarantine_count,
            "consensus_reached": consensus_reached,
            "consensus_decision": consensus_decision,
            "node_decisions": node_decisions
        }


def integrate_with_experiment(
    experiment_config: Dict,
    conductor_url: str = "http://localhost:9888"
) -> HolochainBridge:
    """
    Integrate Holochain bridge with a 0TML experiment.

    Args:
        experiment_config: Experiment configuration dict
        conductor_url: Holochain conductor URL

    Returns:
        Configured HolochainBridge instance

    Example:
        # In experiment runner
        if config.get("use_holochain"):
            bridge = integrate_with_experiment(config)

            # After PoGQ decision and proof generation
            action_hash = bridge.submit_proof(
                node_id=f"node_{client_id}",
                round_number=round_num,
                proof_path=proof_path,
                public_path=public_path,
                journal_path=journal_path
            )
    """
    bridge = HolochainBridge(conductor_url)

    logger.info(
        f"Integrated Holochain bridge for experiment: "
        f"{experiment_config.get('experiment_id', 'unknown')}"
    )

    return bridge


if __name__ == "__main__":
    """
    Test the Holochain bridge (requires running conductor).

    Usage:
        python experiments/holochain_bridge.py
    """
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Holochain Bridge Test")
    print("=" * 60)

    # Initialize bridge
    bridge = HolochainBridge("http://localhost:9888")

    # Test proof submission (requires actual proof files)
    test_proof_dir = Path("proofs/test/")

    if test_proof_dir.exists():
        try:
            action_hash = bridge.submit_proof(
                node_id="test_node",
                round_number=1,
                proof_path=test_proof_dir / "proof.bin",
                public_path=test_proof_dir / "public_echo.json",
                journal_path=test_proof_dir / "journal.bin"
            )

            print(f"✅ Proof submitted: {action_hash}")

            # Retrieve it back
            proof = bridge.get_proof(action_hash)
            if proof:
                print(f"✅ Proof retrieved successfully")
            else:
                print(f"❌ Failed to retrieve proof")

        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"⚠️  Test proof directory not found: {test_proof_dir}")
        print("   Create test proofs with VSV-STARK prover first")

    print("=" * 60)
    print("Test complete")
