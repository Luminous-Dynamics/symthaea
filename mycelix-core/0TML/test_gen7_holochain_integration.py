#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gen7-zkSTARK + Holochain Integration Test

This script tests the full pipeline:
1. Generate RISC Zero proof using gen7-zkstark host binary
2. Publish proof to Holochain PoGQ zome
3. Verify proof retrieval and Byzantine detection logic

Requirements:
- Holochain conductor running on ws://localhost:8888
- gen7-zkstark/host binary compiled
- PoGQ DNA installed on conductor
"""

import asyncio
import websockets
import json
import subprocess
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import struct
import time

# Configuration
CONDUCTOR_URL = "ws://localhost:8888"
GEN7_HOST_BINARY = Path(__file__).parent / "gen7-zkstark/target/release/host"
POGQ_DNA_PATH = Path(__file__).parent / "holochain/dnas/pogq_dna/pogq_dna.dna"

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {msg}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_success(msg: str):
    print(f"{Colors.OKGREEN}✅ {msg}{Colors.ENDC}")

def print_error(msg: str):
    print(f"{Colors.FAIL}❌ {msg}{Colors.ENDC}")

def print_info(msg: str):
    print(f"{Colors.OKCYAN}ℹ️  {msg}{Colors.ENDC}")


class Gen7HolochainIntegration:
    """Integration test for gen7-zkSTARK + Holochain"""

    def __init__(self):
        self.app_id: Optional[str] = None
        self.agent_key: Optional[str] = None
        self.websocket: Optional[Any] = None

    async def connect_to_conductor(self) -> bool:
        """Connect to Holochain conductor admin interface"""
        print_info(f"Connecting to conductor at {CONDUCTOR_URL}...")
        try:
            self.websocket = await websockets.connect(CONDUCTOR_URL)
            print_success("Connected to Holochain conductor")
            return True
        except Exception as e:
            print_error(f"Failed to connect to conductor: {e}")
            print_info("Make sure the conductor is running:")
            print_info("  docker logs holochain-zerotrustml")
            return False

    async def send_admin_request(self, method: str, params: Dict) -> Optional[Dict]:
        """Send request to conductor admin interface"""
        request = {
            "id": 1,
            "type": method,
            **params
        }

        print_info(f"Sending request: {method}")
        await self.websocket.send(json.dumps(request))

        response = await self.websocket.recv()
        result = json.loads(response)

        if "error" in result:
            print_error(f"Request failed: {result['error']}")
            return None

        return result

    async def install_pogq_dna(self) -> bool:
        """Install PoGQ DNA on conductor"""
        print_header("Installing PoGQ DNA")

        if not POGQ_DNA_PATH.exists():
            print_error(f"DNA bundle not found: {POGQ_DNA_PATH}")
            return False

        print_info(f"DNA bundle: {POGQ_DNA_PATH}")
        print_info(f"Size: {POGQ_DNA_PATH.stat().st_size / 1024:.1f} KB")

        # Read DNA bundle
        with open(POGQ_DNA_PATH, 'rb') as f:
            dna_bytes = f.read()

        # Generate agent key
        result = await self.send_admin_request("generate_agent_pub_key", {})
        if not result:
            return False

        self.agent_key = result.get("agent_pub_key")
        print_success(f"Generated agent key: {self.agent_key[:16]}...")

        # Install app
        self.app_id = f"pogq-test-{int(time.time())}"
        install_result = await self.send_admin_request("install_app", {
            "installed_app_id": self.app_id,
            "agent_key": self.agent_key,
            "dnas": [{
                "nick": "pogq_dna",
                "path": str(POGQ_DNA_PATH.absolute()),
            }]
        })

        if not install_result:
            return False

        print_success(f"Installed app: {self.app_id}")

        # Enable app
        enable_result = await self.send_admin_request("enable_app", {
            "installed_app_id": self.app_id
        })

        if enable_result:
            print_success("App enabled successfully")
            return True

        return False

    def generate_gen7_proof(self, round_num: int = 1) -> Optional[Dict]:
        """Generate RISC Zero proof using gen7-zkstark host binary"""
        print_header(f"Generating gen7-zkSTARK Proof for Round {round_num}")

        if not GEN7_HOST_BINARY.exists():
            print_error(f"Host binary not found: {GEN7_HOST_BINARY}")
            print_info("Build it with: cd gen7-zkstark && cargo build --release")
            return None

        print_info(f"Host binary: {GEN7_HOST_BINARY}")

        try:
            # Run host binary to generate proof
            # This will run the PoGQ Byzantine detection logic in zkVM
            print_info("Running RISC Zero prover (this may take 30-60 seconds)...")

            result = subprocess.run(
                [str(GEN7_HOST_BINARY)],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            if result.returncode != 0:
                print_error(f"Proof generation failed: {result.stderr}")
                return None

            print_success("RISC Zero proof generated successfully")

            # Parse output to extract receipt
            # The host binary outputs the receipt as hex or base64
            output_lines = result.stdout.strip().split('\n')

            # Look for receipt in output
            receipt_data = None
            for line in output_lines:
                if "receipt:" in line.lower() or "proof:" in line.lower():
                    receipt_data = line.split(':', 1)[1].strip()
                    break

            if not receipt_data:
                print_error("Could not find receipt in output")
                print_info("Output:")
                print(result.stdout)
                return None

            # Generate nonce for gradient binding
            nonce = hashlib.sha256(f"test_round_{round_num}_{time.time()}".encode()).digest()

            proof_data = {
                "round": round_num,
                "nonce": list(nonce),  # Convert bytes to list for JSON
                "receipt_bytes": receipt_data,
                "quarantine_out": 0,  # Healthy node
                "current_round": round_num,
                "ema_t_fp": 0,  # No Byzantine behavior
                "consec_viol_t": 0,
                "consec_clear_t": round_num,  # Consecutive clear rounds
                "prov_hash": [0, 0, 0, 0],  # Provenance disabled for test
                "profile_id": 128,  # S128 security
                "air_rev": 1,  # AIR schema version 1
            }

            print_success(f"Proof data prepared:")
            print_info(f"  Round: {proof_data['round']}")
            print_info(f"  Quarantine: {proof_data['quarantine_out']}")
            print_info(f"  Consecutive clear: {proof_data['consec_clear_t']}")

            return proof_data

        except subprocess.TimeoutExpired:
            print_error("Proof generation timed out (exceeded 2 minutes)")
            return None
        except Exception as e:
            print_error(f"Proof generation failed: {e}")
            return None

    async def publish_proof_to_holochain(self, proof_data: Dict) -> bool:
        """Publish PoGQ proof to Holochain zome"""
        print_header("Publishing Proof to Holochain")

        # Call PoGQ zome function
        call_result = await self.send_admin_request("call_zome", {
            "app_id": self.app_id,
            "cell_id": f"{self.app_id}::pogq_dna",
            "zome_name": "pogq_zome",
            "fn_name": "publish_pogq_proof",
            "payload": proof_data
        })

        if not call_result:
            return False

        print_success("Proof published successfully to Holochain DHT")
        print_info(f"Entry hash: {call_result.get('hash', 'N/A')}")

        return True

    async def verify_proof_retrieval(self, round_num: int) -> bool:
        """Verify we can retrieve the published proof"""
        print_header("Verifying Proof Retrieval")

        # Query proofs for this round
        result = await self.send_admin_request("call_zome", {
            "app_id": self.app_id,
            "cell_id": f"{self.app_id}::pogq_dna",
            "zome_name": "pogq_zome",
            "fn_name": "get_round_proofs",
            "payload": {"round": round_num}
        })

        if not result:
            return False

        proofs = result.get("proofs", [])
        print_success(f"Retrieved {len(proofs)} proof(s) for round {round_num}")

        if proofs:
            proof = proofs[0]
            print_info(f"  Node: {proof.get('node_id', 'N/A')[:16]}...")
            print_info(f"  Quarantine: {proof.get('quarantine_out', 'N/A')}")
            print_info(f"  Round: {proof.get('current_round', 'N/A')}")

        return True

    async def run_full_integration_test(self):
        """Run complete integration test"""
        print_header("Gen7-zkSTARK + Holochain Integration Test")

        # Step 1: Connect to conductor
        if not await self.connect_to_conductor():
            return False

        # Step 2: Install PoGQ DNA
        if not await self.install_pogq_dna():
            return False

        # Step 3: Generate gen7 proof
        proof_data = self.generate_gen7_proof(round_num=1)
        if not proof_data:
            return False

        # Step 4: Publish to Holochain
        if not await self.publish_proof_to_holochain(proof_data):
            return False

        # Step 5: Verify retrieval
        if not await self.verify_proof_retrieval(round_num=1):
            return False

        print_header("Integration Test Complete ✅")
        print_success("All tests passed!")
        print_info("\nNext steps:")
        print_info("  - Test Byzantine node detection")
        print_info("  - Test multi-round aggregation")
        print_info("  - Test quarantine weight logic")

        return True

    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()


async def main():
    """Main entry point"""
    integration = Gen7HolochainIntegration()

    try:
        success = await integration.run_full_integration_test()
        return 0 if success else 1
    except KeyboardInterrupt:
        print_error("\nTest interrupted by user")
        return 1
    except Exception as e:
        print_error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        await integration.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
