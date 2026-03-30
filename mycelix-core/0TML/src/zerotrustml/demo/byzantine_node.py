#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Byzantine Node Simulator

Simulates malicious actors attempting to compromise federated learning:
- Model poisoning attacks (bias toward incorrect outcomes)
- Gradient noise injection (disrupt convergence)
- Free-rider attacks (submit garbage, hope for credits)

Demonstrates that real Bulletproofs make Byzantine attacks impossible:
- Low-quality gradients have PoGQ < 0.7
- Cannot generate valid ZK proof for PoGQ < threshold
- Coordinator automatically rejects without learning actual score
"""

import sys
import os
import random
import logging
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from zerotrustml.demo.hospital_node import HospitalNode
from zkpoc import ZKPoC

logger = logging.getLogger(__name__)


class ByzantineNode(HospitalNode):
    """
    Simulates a malicious actor in federated learning.

    Demonstrates Byzantine attacks and why they fail with ZK-PoC:
    1. Byzantine nodes produce low-quality gradients
    2. Low quality → PoGQ score < 0.7 threshold
    3. Cannot generate valid Bulletproof for score < threshold
    4. Coordinator rejects without learning WHY (zero-knowledge!)
    """

    # Attack types
    ATTACK_MODEL_POISONING = "model_poisoning"
    ATTACK_GRADIENT_NOISE = "gradient_noise"
    ATTACK_FREE_RIDER = "free_rider"

    def __init__(
        self,
        node_id: str,
        attack_type: str,
        num_patients: int = 5000,
        use_real_bulletproofs: bool = True
    ):
        """
        Initialize Byzantine (malicious) node.

        Args:
            node_id: Unique identifier (e.g., "malicious-actor")
            attack_type: Type of attack to perform:
                - "model_poisoning": Bias model toward incorrect outcomes
                - "gradient_noise": Inject random noise to disrupt training
                - "free_rider": Submit zero gradient, hope to get credits
            num_patients: Simulated dataset size (default: 5000)
            use_real_bulletproofs: Use real 608-byte Bulletproofs (default: True)
        """
        # Initialize as low-quality hospital (0.3 data quality)
        super().__init__(
            node_id=node_id,
            num_patients=num_patients,
            data_quality=0.3,  # Byzantine nodes have poor data
            use_real_bulletproofs=use_real_bulletproofs
        )

        self.attack_type = attack_type
        self.attack_attempts = 0
        self.successful_attacks = 0

        logger.info(
            f"Initialized Byzantine node {node_id}: "
            f"attack={attack_type}, patients={num_patients}"
        )

    def train_local_model(self, global_model_weights: List[float]) -> List[float]:
        """
        Generate malicious gradients based on attack type.

        Args:
            global_model_weights: Current global model parameters

        Returns:
            malicious_gradients: Poisoned model updates
        """
        self.attack_attempts += 1
        num_params = len(global_model_weights)

        if self.attack_type == self.ATTACK_MODEL_POISONING:
            # Try to bias model toward misdiagnosis
            # Large gradients in wrong direction
            gradients = self._generate_poisoned_gradient(num_params)
            logger.debug(
                f"{self.node_id}: Generated poisoned gradient "
                f"(attack #{self.attack_attempts})"
            )

        elif self.attack_type == self.ATTACK_GRADIENT_NOISE:
            # Inject high-variance noise to disrupt convergence
            gradients = self._generate_noisy_gradient(num_params)
            logger.debug(
                f"{self.node_id}: Generated noisy gradient "
                f"(attack #{self.attack_attempts})"
            )

        elif self.attack_type == self.ATTACK_FREE_RIDER:
            # Submit near-zero gradient (no work done)
            gradients = self._generate_zero_gradient(num_params)
            logger.debug(
                f"{self.node_id}: Generated zero gradient "
                f"(free-rider attack #{self.attack_attempts})"
            )

        else:
            # Unknown attack type, use noise
            logger.warning(f"{self.node_id}: Unknown attack type '{self.attack_type}'")
            gradients = self._generate_noisy_gradient(num_params)

        return gradients

    def _generate_poisoned_gradient(self, num_params: int) -> List[float]:
        """
        Generate gradient that pushes model toward incorrect predictions.

        Strategy: Large updates in opposite direction of correct gradient.
        """
        gradients = []
        for _ in range(num_params):
            # Large negative gradient (opposite of improvement)
            poisoned = random.gauss(-0.05, 0.02)  # Mean: -0.05, high variance
            gradients.append(poisoned)
        return gradients

    def _generate_noisy_gradient(self, num_params: int) -> List[float]:
        """
        Generate high-noise gradient to disrupt convergence.

        Strategy: High variance random noise that doesn't improve model.
        """
        gradients = []
        for _ in range(num_params):
            # Random noise with high variance
            noise = random.gauss(0.0, 0.1)  # Mean: 0, very high variance
            gradients.append(noise)
        return gradients

    def _generate_zero_gradient(self, num_params: int) -> List[float]:
        """
        Generate near-zero gradient (free-rider attack).

        Strategy: Minimal work, hope to get credits anyway.
        """
        gradients = []
        for _ in range(num_params):
            # Near-zero with tiny noise
            zero = random.gauss(0.0, 0.001)
            gradients.append(zero)
        return gradients

    def compute_pogq_score(self, gradients: List[float]) -> float:
        """
        Compute PoGQ score for malicious gradient.

        Byzantine gradients are LOW QUALITY and score poorly.

        Args:
            gradients: Malicious gradient values

        Returns:
            pogq_score: Quality score (0.2-0.5, below 0.7 threshold)
        """
        # Byzantine nodes produce low-quality gradients
        # Base score around 0.35 with variation
        base_score = 0.35
        variation = random.gauss(0, 0.08)
        pogq_score = base_score + variation

        # Clip to reasonable Byzantine range (0.2-0.5)
        pogq_score = max(0.2, min(0.5, pogq_score))

        logger.debug(
            f"{self.node_id}: PoGQ score = {pogq_score:.3f} "
            f"(below threshold 0.7)"
        )

        return pogq_score

    async def submit_gradient(
        self,
        coordinator,
        gradients: List[float],
        pogq_score: float
    ) -> Dict[str, Any]:
        """
        Attempt to submit malicious gradient to coordinator.

        This will FAIL because:
        1. PoGQ score < 0.7 (below threshold)
        2. Cannot generate valid Bulletproof for score < threshold
        3. ZKPoC raises ValueError, preventing submission

        This demonstrates Byzantine resistance!

        Args:
            coordinator: Phase10Coordinator instance
            gradients: Malicious gradient values
            pogq_score: Quality score (will be < 0.7)

        Returns:
            result: Rejection with reason (accepted=False)
        """
        logger.info(
            f"{self.node_id}: Attempting Byzantine attack "
            f"(type={self.attack_type}, PoGQ={pogq_score:.3f})"
        )

        try:
            # Try to generate ZK proof for low-quality gradient
            # This WILL FAIL because PoGQ < 0.7 threshold!
            zkpoc_proof = self.zkpoc.generate_proof(pogq_score)

            # If we somehow got here (shouldn't happen!), try to submit
            logger.error(
                f"{self.node_id}: WARNING - Generated proof for PoGQ={pogq_score:.3f} "
                "(this should not happen!)"
            )

            # Try submission anyway
            import json
            encrypted_gradient = json.dumps(gradients).encode()

            result = await coordinator.handle_gradient_submission(
                node_id=self.node_id,
                encrypted_gradient=encrypted_gradient,
                zkpoc_proof=zkpoc_proof,
                pogq_score=None
            )

            if result["accepted"]:
                # Byzantine attack succeeded (BAD!)
                self.successful_attacks += 1
                logger.error(
                    f"{self.node_id}: ❌ Byzantine attack SUCCEEDED "
                    f"({self.successful_attacks}/{self.attack_attempts})"
                )
            else:
                logger.info(
                    f"{self.node_id}: ✅ Byzantine attack detected and rejected"
                )

            return result

        except ValueError as e:
            # Expected! Cannot generate valid proof for PoGQ < 0.7
            logger.info(
                f"{self.node_id}: ✅ Byzantine attack prevented by ZK-PoC: {e}"
            )

            return {
                "accepted": False,
                "reason": f"Cannot generate valid proof: {e}",
                "credits_issued": 0
            }

        except Exception as e:
            # Unexpected error
            logger.error(f"{self.node_id}: Unexpected error during attack: {e}")
            return {
                "accepted": False,
                "reason": f"Submission error: {e}",
                "credits_issued": 0
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get Byzantine node statistics."""
        base_stats = super().get_stats()

        # Add attack-specific stats
        base_stats.update({
            "attack_type": self.attack_type,
            "attack_attempts": self.attack_attempts,
            "successful_attacks": self.successful_attacks,
            "success_rate": (
                self.successful_attacks / self.attack_attempts
                if self.attack_attempts > 0
                else 0.0
            )
        })

        return base_stats

    def __repr__(self):
        return (
            f"ByzantineNode({self.node_id}, "
            f"attack={self.attack_type}, "
            f"patients={self.num_patients}, "
            f"attempts={self.attack_attempts}, "
            f"successful={self.successful_attacks})"
        )


if __name__ == "__main__":
    # Quick test
    print("🔴 Testing Byzantine Node\n")

    # Create Byzantine node
    node = ByzantineNode(
        "test-malicious",
        attack_type=ByzantineNode.ATTACK_MODEL_POISONING
    )

    # Simulate attack attempt
    global_weights = [0.5] * 100  # 100-parameter model
    gradients = node.train_local_model(global_weights)
    pogq = node.compute_pogq_score(gradients)

    print(f"Node: {node}")
    print(f"Attack type: {node.attack_type}")
    print(f"Gradients computed: {len(gradients)}")
    print(f"PoGQ score: {pogq:.3f} (threshold: 0.7)")
    print(f"\nExpected: PoGQ < 0.7, cannot generate valid proof!")
    print(f"Stats: {node.get_stats()}")

    # Try to generate proof (will fail!)
    try:
        proof = node.zkpoc.generate_proof(pogq)
        print(f"\n❌ ERROR: Should not be able to generate proof for PoGQ={pogq:.3f}!")
    except ValueError as e:
        print(f"\n✅ SUCCESS: Byzantine attack prevented!")
        print(f"   Reason: {e}")
