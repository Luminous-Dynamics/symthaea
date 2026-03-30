#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Hospital Node Simulator

Simulates an honest hospital participating in federated learning:
- Trains local ML model on patient data
- Computes gradients
- Calculates PoGQ (Proof of Gradient Quality) score
- Generates real Bulletproofs
- Submits to coordinator
"""

import sys
import os
import json
import random
import logging
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from zkpoc import ZKPoC

logger = logging.getLogger(__name__)


class HospitalNode:
    """
    Simulates an honest hospital in federated learning.

    Demonstrates:
    - Local model training on patient data
    - Gradient computation and quality assessment
    - Zero-knowledge proof generation (real Bulletproofs!)
    - Privacy-preserving submission to coordinator
    """

    def __init__(
        self,
        node_id: str,
        num_patients: int,
        data_quality: float,
        use_real_bulletproofs: bool = True
    ):
        """
        Initialize hospital node.

        Args:
            node_id: Unique identifier (e.g., "mayo-clinic")
            num_patients: Number of patients in local dataset
            data_quality: Quality of local data (0.0-1.0)
                         Higher = better labels, less noise
            use_real_bulletproofs: Use real 608-byte Bulletproofs (default: True)
        """
        self.node_id = node_id
        self.num_patients = num_patients
        self.data_quality = data_quality
        self.use_real_bulletproofs = use_real_bulletproofs

        # Track participation
        self.credits = 0
        self.rounds_participated = 0
        self.total_pogq = 0.0

        # Initialize ZK-PoC system
        self.zkpoc = ZKPoC(
            pogq_threshold=0.7,
            use_real_bulletproofs=use_real_bulletproofs
        )

        logger.info(f"Initialized {node_id}: {num_patients} patients, quality={data_quality:.2f}")

    def train_local_model(self, global_model_weights: List[float]) -> List[float]:
        """
        Simulate training on local patient data.

        In real FL, this would:
        1. Load global model weights
        2. Train on local patient data for N epochs
        3. Compute gradients (difference from global)

        For demo, we simulate with realistic values.

        Args:
            global_model_weights: Current global model parameters

        Returns:
            gradients: Model updates (list of floats)
        """
        # Simulate gradient computation
        # Better data quality = better gradients (lower noise)
        num_params = len(global_model_weights)
        noise_level = (1.0 - self.data_quality) * 0.5  # Max 50% noise for worst quality

        gradients = []
        for weight in global_model_weights:
            # Gradient = direction of improvement + noise
            gradient = random.gauss(0.01, 0.005)  # Small update
            gradient += random.gauss(0, noise_level)  # Add noise based on quality
            gradients.append(gradient)

        logger.debug(f"{self.node_id}: Computed {num_params} gradient values")
        return gradients

    def compute_pogq_score(self, gradients: List[float]) -> float:
        """
        Compute Proof of Gradient Quality score.

        PoGQ measures gradient quality based on:
        - Magnitude (not too large, not too small)
        - Consistency (low variance)
        - Effectiveness (improves validation loss)

        In real system, this would evaluate gradient on held-out validation set.
        For demo, we derive from data quality with realistic variation.

        Args:
            gradients: Computed gradient values

        Returns:
            pogq_score: Quality score (0.0-1.0)
                       0.0 = terrible, 1.0 = excellent
        """
        # Base score from data quality
        base_score = self.data_quality

        # Add realistic variation (±0.05)
        variation = random.gauss(0, 0.02)
        pogq_score = base_score + variation

        # Clip to valid range
        pogq_score = max(0.0, min(1.0, pogq_score))

        logger.debug(f"{self.node_id}: PoGQ score = {pogq_score:.3f}")
        return pogq_score

    async def submit_gradient(
        self,
        coordinator,
        gradients: List[float],
        pogq_score: float
    ) -> Dict[str, Any]:
        """
        Generate ZK proof and submit gradient to coordinator.

        This is the MAGIC of Phase 10:
        1. Generate real Bulletproof (608 bytes!) proving PoGQ ≥ threshold
        2. Encrypt gradient
        3. Submit to coordinator
        4. Coordinator verifies proof WITHOUT learning PoGQ score!

        Args:
            coordinator: Phase10Coordinator instance
            gradients: Computed gradient values
            pogq_score: Quality score (will be hidden by ZK proof!)

        Returns:
            result: Submission result with acceptance status and credits
        """
        try:
            # Generate real Bulletproof (608 bytes!)
            logger.info(f"{self.node_id}: Generating ZK proof for PoGQ={pogq_score:.3f}")
            zkpoc_proof = self.zkpoc.generate_proof(pogq_score)

            # Log proof details
            if hasattr(zkpoc_proof.proof, '__len__'):
                proof_size = len(zkpoc_proof.proof)
                logger.info(f"{self.node_id}: Generated {proof_size}-byte proof")

            # Encrypt gradient (in real system, use proper encryption)
            encrypted_gradient = json.dumps(gradients).encode()

            # Submit to coordinator
            result = await coordinator.handle_gradient_submission(
                node_id=self.node_id,
                encrypted_gradient=encrypted_gradient,
                zkpoc_proof=zkpoc_proof,
                pogq_score=None  # HIDDEN! Coordinator learns NOTHING!
            )

            # Update stats if accepted
            if result["accepted"]:
                self.credits += result["credits_issued"]
                self.rounds_participated += 1
                self.total_pogq += pogq_score

                logger.info(
                    f"{self.node_id}: ✅ Accepted! "
                    f"Earned {result['credits_issued']} credits "
                    f"(total: {self.credits})"
                )
            else:
                logger.warning(
                    f"{self.node_id}: ❌ Rejected: {result['reason']}"
                )

            return result

        except ValueError as e:
            # PoGQ below threshold - cannot generate valid proof!
            logger.error(f"{self.node_id}: Cannot generate proof: {e}")
            return {
                "accepted": False,
                "reason": f"PoGQ below threshold: {e}",
                "credits_issued": 0
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get participation statistics."""
        avg_pogq = (
            self.total_pogq / self.rounds_participated
            if self.rounds_participated > 0
            else 0.0
        )

        return {
            "node_id": self.node_id,
            "num_patients": self.num_patients,
            "data_quality": self.data_quality,
            "rounds_participated": self.rounds_participated,
            "total_credits": self.credits,
            "avg_pogq": avg_pogq,
            "using_real_bulletproofs": self.use_real_bulletproofs
        }

    def __repr__(self):
        return (
            f"HospitalNode({self.node_id}, "
            f"patients={self.num_patients}, "
            f"quality={self.data_quality:.2f}, "
            f"credits={self.credits})"
        )


if __name__ == "__main__":
    # Quick test
    node = HospitalNode("test-hospital", num_patients=5000, data_quality=0.9)

    # Simulate training
    global_weights = [0.5] * 100  # 100-parameter model
    gradients = node.train_local_model(global_weights)

    # Compute quality
    pogq = node.compute_pogq_score(gradients)

    print(f"Node: {node}")
    print(f"Gradients computed: {len(gradients)}")
    print(f"PoGQ score: {pogq:.3f}")
    print(f"Stats: {node.get_stats()}")
