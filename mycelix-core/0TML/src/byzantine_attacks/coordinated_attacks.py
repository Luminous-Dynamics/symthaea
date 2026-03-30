# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Coordinated Byzantine Attacks - High Sophistication

These attacks involve multiple Byzantine nodes working together:
- Coordinated collusion
- Sybil attacks (single adversary, multiple identities)

Detectability: Medium (depends on number of attackers)
Sophistication: High
Real-World: Organized adversary, nation-state attacks
"""

import numpy as np
from typing import List, Optional


class CoordinatedCollusionAttack:
    """
    Coordinated Collusion - Multiple Byzantine Nodes Collaborate

    Behavior:
    - All Byzantine nodes send similar malicious gradients
    - Coordinate poison direction
    - Try to create false "majority"

    Detection:
    - Moderate difficulty (depends on % of Byzantine nodes)
    - At <35% BFT: Ensemble detector should catch
    - At >35% BFT: Peer-comparison fails (no honest majority)
    - PoGQ: Still detects (judges each independently)

    Real-World:
    - Organized Byzantine coalition
    - Multiple compromised nodes
    - Coordinated attack campaign
    """

    def __init__(
        self,
        honest_weight: float = 0.3,
        poison_weight: float = 0.7,
        coordination_noise: float = 0.05
    ):
        """
        Args:
            honest_weight: Weight for honest component (default: 0.3)
            poison_weight: Weight for poison component (default: 0.7)
            coordination_noise: Small noise for each node (default: 0.05)
        """
        self.honest_weight = honest_weight
        self.poison_weight = poison_weight
        self.coordination_noise = coordination_noise

    def generate(
        self,
        honest_gradient: np.ndarray,
        round_num: int,
        node_id: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate coordinated attack gradient

        All Byzantine nodes submit similar gradients (with small variations)
        """
        # Coordinated poison direction (all Byzantine nodes use same direction)
        poison_direction = -honest_gradient / np.linalg.norm(honest_gradient)

        # Base attack (same for all Byzantine nodes)
        attack_base = (
            self.honest_weight * honest_gradient +
            self.poison_weight * poison_direction * np.linalg.norm(honest_gradient)
        )

        # Add small per-node noise (makes coordination less obvious)
        if node_id is not None:
            np.random.seed(node_id)  # Deterministic per node
        coordination_noise_vec = np.random.normal(
            0, self.coordination_noise, honest_gradient.shape
        )

        return attack_base + coordination_noise_vec

    def reset(self) -> None:
        """No state to reset"""
        pass

    @property
    def name(self) -> str:
        return f"Coordinated Collusion ({self.honest_weight}/{self.poison_weight})"

    @property
    def sophistication(self) -> str:
        return "high"


class SybilAttack:
    """
    Sybil Attack - Single Adversary Controls Multiple Identities

    Behavior:
    - One powerful adversary controls N Byzantine nodes
    - All N nodes submit identical or near-identical gradients
    - Try to overwhelm honest nodes with volume

    Detection:
    - Moderate difficulty
    - Similarity between Byzantine nodes: Very high (identical)
    - At <35% BFT: Detectable if Byzantine cluster is obvious
    - At >35% BFT: Can create false majority
    - Identity verification: TEE/hardware attestation can prevent

    Real-World:
    - Single powerful adversary (nation-state, large org)
    - Cloud provider compromise (controls multiple VMs)
    - Botnet-style attack
    """

    def __init__(
        self,
        attack_type: str = "sign_flip",
        sybil_variation: float = 0.01
    ):
        """
        Args:
            attack_type: Type of attack all Sybil nodes perform
                - "sign_flip": All nodes flip sign
                - "scaling": All nodes scale gradient
                - "noise_masked": All nodes submit noise-masked poison
            sybil_variation: Small variation between Sybil identities (default: 0.01)
        """
        self.attack_type = attack_type
        self.sybil_variation = sybil_variation

    def generate(
        self,
        honest_gradient: np.ndarray,
        round_num: int,
        sybil_id: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate Sybil attack gradient

        All Sybil identities submit nearly identical gradients
        """
        # Base attack (same for all Sybil nodes)
        if self.attack_type == "sign_flip":
            attack_base = -honest_gradient

        elif self.attack_type == "scaling":
            attack_base = honest_gradient * 100.0

        elif self.attack_type == "noise_masked":
            poison = -honest_gradient
            noise = np.random.normal(0, 0.4, honest_gradient.shape)
            attack_base = 0.5 * honest_gradient + 0.3 * poison + 0.2 * noise

        else:
            raise ValueError(f"Unknown Sybil attack type: {self.attack_type}")

        # Add tiny per-identity variation (makes them seem like different nodes)
        if sybil_id is not None:
            np.random.seed(sybil_id)  # Deterministic per Sybil ID
        variation = np.random.normal(0, self.sybil_variation, honest_gradient.shape)

        return attack_base + variation

    def reset(self) -> None:
        """No state to reset"""
        pass

    @property
    def name(self) -> str:
        return f"Sybil ({self.attack_type}, var={self.sybil_variation})"

    @property
    def sophistication(self) -> str:
        return "high"


# Example usage
if __name__ == "__main__":
    print("Testing Coordinated Byzantine Attacks...")

    # Create honest gradient
    honest_grad = np.random.randn(100)

    print("\n=== Coordinated Collusion ===")
    collusion = CoordinatedCollusionAttack()

    # Simulate 3 Byzantine nodes colluding
    byzantine_grads = []
    for node_id in range(3):
        attack_grad = collusion.generate(honest_grad, round_num=1, node_id=node_id)
        byzantine_grads.append(attack_grad)

    # Check similarity between Byzantine nodes
    g1, g2, g3 = byzantine_grads
    sim_12 = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))
    sim_13 = np.dot(g1, g3) / (np.linalg.norm(g1) * np.linalg.norm(g3))
    sim_23 = np.dot(g2, g3) / (np.linalg.norm(g2) * np.linalg.norm(g3))

    print(f"Similarity between Byzantine nodes:")
    print(f"  Node 1 vs Node 2: {sim_12:.3f}")
    print(f"  Node 1 vs Node 3: {sim_13:.3f}")
    print(f"  Node 2 vs Node 3: {sim_23:.3f}")
    print(f"  Average: {(sim_12 + sim_13 + sim_23) / 3:.3f}")

    print("\n=== Sybil Attack ===")
    sybil = SybilAttack(attack_type="sign_flip", sybil_variation=0.01)

    # Simulate 3 Sybil identities controlled by same adversary
    sybil_grads = []
    for sybil_id in range(3):
        attack_grad = sybil.generate(honest_grad, round_num=1, sybil_id=sybil_id)
        sybil_grads.append(attack_grad)

    # Check similarity between Sybil identities
    s1, s2, s3 = sybil_grads
    sim_12 = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2))
    sim_13 = np.dot(s1, s3) / (np.linalg.norm(s1) * np.linalg.norm(s3))
    sim_23 = np.dot(s2, s3) / (np.linalg.norm(s2) * np.linalg.norm(s3))

    print(f"Similarity between Sybil identities:")
    print(f"  Identity 1 vs Identity 2: {sim_12:.3f}")
    print(f"  Identity 1 vs Identity 3: {sim_13:.3f}")
    print(f"  Identity 2 vs Identity 3: {sim_23:.3f}")
    print(f"  Average: {(sim_12 + sim_13 + sim_23) / 3:.3f}")

    print("\n✅ Coordinated attacks test complete!")
    print("   Sybil identities should have very high similarity (>0.99)")
    print("   Collusion nodes should have high similarity (>0.95)")
