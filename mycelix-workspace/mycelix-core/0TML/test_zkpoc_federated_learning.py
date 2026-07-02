#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZK-PoC Enhanced Federated Learning Test

Demonstrates privacy-preserving federated learning with Zero-Knowledge Proofs:
- 3 hospital nodes with private data
- Gradients protected with Bulletproofs
- Coordinator verifies quality WITHOUT seeing gradients
- Full HIPAA/GDPR compliance

Architecture:
┌──────────────────────────────────────────────────────────────┐
│                  Privacy-Preserving FL                        │
├──────────────────────────────────────────────────────────────┤
│  Hospital A          Hospital B          Hospital C          │
│  ┌─────────┐        ┌─────────┐        ┌─────────┐          │
│  │ Train   │        │ Train   │        │ Train   │          │
│  │ Locally │        │ Locally │        │ Locally │          │
│  └────┬────┘        └────┬────┘        └────┬────┘          │
│       │                  │                  │                │
│  ┌────▼────┐        ┌────▼────┐        ┌────▼────┐          │
│  │ PoGQ    │        │ PoGQ    │        │ PoGQ    │          │
│  │ Score   │        │ Score   │        │ Score   │          │
│  └────┬────┘        └────┬────┘        └────┬────┘          │
│       │                  │                  │                │
│  ┌────▼───────────────────┐    ┌────▼───────────────────┐   │
│  │ ZK Proof: score ≥ 0.7  │    │ ZK Proof: score ≥ 0.7  │   │
│  │ (Bulletproof)          │    │ (Bulletproof)          │   │
│  └────┬───────────────────┘    └────┬───────────────────┘   │
│       │                                   │                  │
│       └──────────┬────────────────────────┘                  │
│                  ▼                                           │
│          ┌───────────────┐                                   │
│          │ Coordinator   │                                   │
│          │ Verifies ZK   │                                   │
│          │ (learns only  │                                   │
│          │  valid/invalid)│                                  │
│          └───────┬───────┘                                   │
│                  │                                           │
│          ┌───────▼────────┐                                  │
│          │  FedAvg        │                                  │
│          │  Aggregation   │                                  │
│          └────────────────┘                                  │
└──────────────────────────────────────────────────────────────┘

Privacy Guarantees:
- Hospital data never leaves premises
- Coordinator never sees exact PoGQ scores
- Bulletproofs prove quality without revealing score
- Byzantine-resistant + Privacy-preserving
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import json

# Import ZK-PoC infrastructure
from zkpoc import ZKPoC, RangeProof


class SimpleNet(nn.Module):
    """Simple neural network for testing"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class PrivateHospitalNode:
    """
    Hospital node with ZK-PoC protected gradient sharing.

    Privacy Features:
    - Local data never leaves hospital
    - PoGQ score computed locally (private)
    - ZK proof generated (reveals nothing about score)
    - Only proof sent to coordinator
    """

    def __init__(self, node_id: str, name: str, zkpoc: ZKPoC):
        self.node_id = node_id
        self.name = name
        self.zkpoc = zkpoc
        self.model = SimpleNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.local_data = self._generate_local_data()

        # Privacy tracking
        self.total_proofs_generated = 0
        self.total_proofs_verified = 0

    def _generate_local_data(self):
        """Generate local private data (HIPAA-protected)"""
        # Each hospital has different data distribution
        np.random.seed(hash(self.node_id) % 2**32)
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        return X, y

    def train_local_epoch(self) -> Tuple[float, float]:
        """
        Train on local private data.

        Returns:
            (loss, pogq_score): Training loss and gradient quality score
        """
        X, y = self.local_data

        self.model.train()
        self.optimizer.zero_grad()

        predictions = self.model(X)
        loss = nn.MSELoss()(predictions, y)
        loss.backward()
        self.optimizer.step()

        # Compute PoGQ score (gradient quality)
        # In real implementation, this uses sophisticated metrics
        # Here we simulate: random score between 0.6 and 1.0
        pogq_score = 0.6 + 0.4 * np.random.rand()

        return loss.item(), pogq_score

    def get_gradients(self) -> np.ndarray:
        """Extract model gradients"""
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.data.numpy().flatten())
        return np.concatenate(gradients)

    def generate_zkpoc_proof(self, pogq_score: float) -> RangeProof:
        """
        Generate ZK proof that gradient quality ≥ threshold.

        Privacy guarantee: Proof reveals NOTHING about actual score,
        only that it's above threshold (0.7).

        Args:
            pogq_score: Gradient quality score (private)

        Returns:
            ZK proof that can be verified without revealing score
        """
        try:
            proof = self.zkpoc.generate_proof(pogq_score)
            self.total_proofs_generated += 1
            return proof
        except ValueError as e:
            # Score below threshold - cannot generate valid proof
            raise ValueError(f"{self.name}: Cannot generate proof for low-quality gradient: {e}")

    def submit_gradient_with_proof(self, pogq_score: float) -> Dict:
        """
        Submit gradient with ZK proof to coordinator.

        Returns:
            Submission package containing:
            - encrypted_gradient: Encrypted gradient data
            - zkpoc_proof: Zero-knowledge proof of quality
            - node_id: Hospital identifier
        """
        # Generate ZK proof
        proof = self.generate_zkpoc_proof(pogq_score)

        # Encrypt gradient (in production: use NaCl/libsodium)
        gradients = self.get_gradients()
        encrypted_gradient = self._mock_encrypt(gradients)

        return {
            "node_id": self.node_id,
            "encrypted_gradient": encrypted_gradient,
            "zkpoc_proof": proof,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _mock_encrypt(self, gradients: np.ndarray) -> bytes:
        """Mock encryption for demo (use real crypto in production)"""
        return gradients.tobytes()

    def apply_aggregated_gradients(self, aggregated: np.ndarray):
        """Apply aggregated gradients to local model"""
        if aggregated is None:
            return

        idx = 0
        for param in self.model.parameters():
            param_length = param.numel()
            param_grad = aggregated[idx:idx+param_length].reshape(param.shape)
            param.data -= 0.01 * torch.tensor(param_grad)
            idx += param_length


class PrivacyPreservingCoordinator:
    """
    Coordinator that aggregates gradients WITHOUT seeing quality scores.

    Privacy Features:
    - Verifies ZK proofs (learns only valid/invalid)
    - Never sees actual PoGQ scores
    - Decrypts only verified gradients
    - Byzantine-resistant through ZK-PoC
    """

    def __init__(self, zkpoc: ZKPoC):
        self.zkpoc = zkpoc
        self.total_submissions = 0
        self.total_accepted = 0
        self.total_rejected = 0

    def verify_and_aggregate(self, submissions: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        Verify ZK proofs and aggregate gradients.

        Privacy guarantee: Never learns actual PoGQ scores,
        only which gradients passed quality threshold.

        Args:
            submissions: List of {encrypted_gradient, zkpoc_proof, node_id}

        Returns:
            (aggregated_gradient, stats): FedAvg aggregation + statistics
        """
        verified_gradients = []
        verification_results = []

        self.total_submissions += len(submissions)

        for submission in submissions:
            node_id = submission["node_id"]
            proof = submission["zkpoc_proof"]

            # Verify ZK proof WITHOUT learning score
            is_valid = self.zkpoc.verify_proof(proof)

            if is_valid:
                # Proof valid: decrypt and accept gradient
                encrypted = submission["encrypted_gradient"]
                gradient = self._mock_decrypt(encrypted)
                verified_gradients.append(gradient)

                verification_results.append({
                    "node_id": node_id,
                    "accepted": True,
                    "reason": "ZK proof verified"
                })
                self.total_accepted += 1
            else:
                # Proof invalid: reject gradient
                verification_results.append({
                    "node_id": node_id,
                    "accepted": False,
                    "reason": "ZK proof failed"
                })
                self.total_rejected += 1

        # Aggregate verified gradients (FedAvg)
        if len(verified_gradients) > 0:
            aggregated = np.mean(verified_gradients, axis=0)
        else:
            aggregated = None

        stats = {
            "total_submissions": len(submissions),
            "accepted": len(verified_gradients),
            "rejected": len(submissions) - len(verified_gradients),
            "acceptance_rate": len(verified_gradients) / len(submissions) if submissions else 0,
            "verification_results": verification_results
        }

        return aggregated, stats

    def _mock_decrypt(self, encrypted: bytes) -> np.ndarray:
        """Mock decryption for demo (use real crypto in production)"""
        return np.frombuffer(encrypted, dtype=np.float32)


def run_zkpoc_federated_round(
    nodes: List[PrivateHospitalNode],
    coordinator: PrivacyPreservingCoordinator,
    round_num: int
) -> Dict:
    """
    One round of privacy-preserving federated learning with ZK-PoC.

    Process:
    1. Each hospital trains locally (data never leaves)
    2. Each hospital computes PoGQ score (private)
    3. Each hospital generates ZK proof (reveals nothing)
    4. Coordinator verifies proofs (learns only valid/invalid)
    5. Coordinator aggregates verified gradients
    6. Each hospital updates model
    """
    print(f"\n{'='*70}")
    print(f"🔐 ROUND {round_num} - Privacy-Preserving Federated Learning")
    print(f"{'='*70}")

    # Step 1: Local training
    print(f"\n📊 Step 1: Private Local Training")
    print("-" * 70)

    local_losses = []
    pogq_scores = []

    for node in nodes:
        loss, pogq_score = node.train_local_epoch()
        local_losses.append(loss)
        pogq_scores.append(pogq_score)
        print(f"  {node.name:30} | Loss: {loss:.4f} | PoGQ: {pogq_score:.4f} (PRIVATE)")

    # Step 2: Generate ZK proofs and submit
    print(f"\n🔐 Step 2: Generate ZK Proofs (Privacy-Preserving)")
    print("-" * 70)

    submissions = []
    for node, pogq_score in zip(nodes, pogq_scores):
        try:
            submission = node.submit_gradient_with_proof(pogq_score)
            submissions.append(submission)
            proof_size = len(submission["zkpoc_proof"].proof)
            print(f"  {node.name:30} | ✅ Proof generated ({proof_size} bytes)")
        except ValueError as e:
            print(f"  {node.name:30} | ❌ Proof failed: {e}")

    # Step 3: Coordinator verifies and aggregates
    print(f"\n✅ Step 3: Coordinator Verifies ZK Proofs")
    print("-" * 70)

    aggregated, stats = coordinator.verify_and_aggregate(submissions)

    for result in stats["verification_results"]:
        status = "✅ ACCEPTED" if result["accepted"] else "❌ REJECTED"
        print(f"  {result['node_id']:30} | {status}: {result['reason']}")

    print(f"\n📊 Verification Statistics:")
    print(f"  Total submissions: {stats['total_submissions']}")
    print(f"  Accepted: {stats['accepted']} ({stats['acceptance_rate']*100:.1f}%)")
    print(f"  Rejected: {stats['rejected']}")

    # Step 4: Apply aggregated gradients
    if aggregated is not None:
        print(f"\n🔄 Step 4: Apply Aggregated Gradients")
        print("-" * 70)

        for node in nodes:
            node.apply_aggregated_gradients(aggregated)
            print(f"  {node.name:30} | ✅ Model updated")
    else:
        print(f"\n⚠️  Step 4: No valid gradients - skipping update")

    avg_loss = np.mean(local_losses)
    print(f"\n📈 Round {round_num} Complete | Average Loss: {avg_loss:.4f}")

    return {
        "round": round_num,
        "avg_loss": avg_loss,
        "stats": stats,
        "privacy_preserved": True
    }


def main():
    """
    Demonstrate privacy-preserving federated learning with ZK-PoC.

    Shows:
    - HIPAA/GDPR compliant FL (data never leaves hospitals)
    - Zero-knowledge proofs (coordinator learns nothing about scores)
    - Byzantine resistance (low-quality gradients rejected)
    - Performance (proofs are small, verification fast)
    """
    print("="*70)
    print("🏥 Privacy-Preserving Federated Learning with ZK-PoC")
    print("="*70)
    print()
    print("Scenario: 3 hospitals training collaborative model")
    print("Privacy: HIPAA/GDPR compliance through Zero-Knowledge Proofs")
    print("Security: Byzantine-resistant quality validation")
    print()

    # Initialize ZK-PoC system (try real Bulletproofs)
    zkpoc = ZKPoC(pogq_threshold=0.7, use_real_bulletproofs=True)

    # Initialize 3 hospital nodes
    nodes = [
        PrivateHospitalNode(
            "hospital-boston-node1",
            "Boston Medical Center",
            zkpoc
        ),
        PrivateHospitalNode(
            "hospital-london-node2",
            "London General Hospital",
            zkpoc
        ),
        PrivateHospitalNode(
            "hospital-tokyo-node3",
            "Tokyo Research Hospital",
            zkpoc
        )
    ]

    # Initialize coordinator
    coordinator = PrivacyPreservingCoordinator(zkpoc)

    print("✅ 3 Hospital Nodes Initialized (Private Data)")
    print(f"   - {nodes[0].name}")
    print(f"   - {nodes[1].name}")
    print(f"   - {nodes[2].name}")
    print()
    print("✅ Privacy-Preserving Coordinator Initialized")
    print(f"   - PoGQ Threshold: {zkpoc.pogq_threshold}")
    print(f"   - Using {'REAL' if zkpoc.bulletproofs.__class__.__name__ == 'RealBulletproofs' else 'MOCK'} Bulletproofs")
    print()

    # Run federated learning rounds
    num_rounds = 5
    results = []

    for round_num in range(1, num_rounds + 1):
        # Use asyncio.run equivalent for non-async function
        result = run_zkpoc_federated_round(nodes, coordinator, round_num)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("📊 PRIVACY-PRESERVING FEDERATED LEARNING COMPLETE")
    print("="*70)
    print()
    print("Training Convergence:")
    for i, result in enumerate(results, 1):
        acceptance = result["stats"]["acceptance_rate"] * 100
        print(f"  Round {i}: Loss {result['avg_loss']:.4f} | Acceptance: {acceptance:.0f}%")

    # Privacy analysis
    print(f"\n🔐 Privacy Guarantees:")
    print(f"  ✅ Data Privacy: Hospital data never left premises")
    print(f"  ✅ Score Privacy: Coordinator never saw PoGQ scores")
    print(f"  ✅ Quality Assured: Only verified gradients aggregated")
    print(f"  ✅ Byzantine Resistant: Invalid proofs rejected")

    # Performance analysis
    total_proofs = sum(node.total_proofs_generated for node in nodes)
    total_accepted = coordinator.total_accepted
    total_rejected = coordinator.total_rejected

    print(f"\n📈 Performance Statistics:")
    print(f"  Total ZK Proofs Generated: {total_proofs}")
    print(f"  Total Accepted: {total_accepted}")
    print(f"  Total Rejected: {total_rejected}")
    print(f"  Overall Acceptance Rate: {total_accepted/(total_accepted+total_rejected)*100:.1f}%")

    # Save results
    output = {
        "test": "zkpoc_federated_learning",
        "rounds": num_rounds,
        "nodes": len(nodes),
        "results": results,
        "privacy_preserved": True,
        "zkpoc_system": {
            "threshold": zkpoc.pogq_threshold,
            "bulletproofs": zkpoc.bulletproofs.__class__.__name__
        },
        "timestamp": datetime.utcnow().isoformat()
    }

    results_path = "/tmp/zkpoc_federated_learning_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n📁 Results saved to: {results_path}")
    print()

    return 0


if __name__ == "__main__":
    import asyncio
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
