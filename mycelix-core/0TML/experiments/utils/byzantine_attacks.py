# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Byzantine Attack Types for Federated Learning Research

Implements 7 types of Byzantine attacks to test robustness of aggregation methods.

References:
- Blanchard et al., "Machine Learning with Adversaries", NeurIPS 2017
- Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust FL", USENIX 2020
- Baruch et al., "A Little Is Enough: Circumventing Defenses For Distributed Learning", NeurIPS 2019
"""

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from enum import Enum


class ByzantineAttackType(Enum):
    """Types of Byzantine attacks."""
    GAUSSIAN_NOISE = "gaussian_noise"      # Random noise (current implementation)
    SIGN_FLIP = "sign_flip"                # Negate gradients
    LABEL_FLIP = "label_flip"              # Train on corrupted labels
    TARGETED_POISON = "targeted_poison"    # Backdoor-style attack
    MODEL_REPLACEMENT = "model_replacement"  # Send completely different model
    ADAPTIVE = "adaptive"                  # Adapts to defense mechanism
    SYBIL = "sybil"                        # Coordinates multiple clients


class ByzantineAttacker:
    """
    Implements various Byzantine attack strategies.

    Usage:
        attacker = ByzantineAttacker(attack_type=ByzantineAttackType.SIGN_FLIP)
        malicious_weights = attacker.attack(honest_weights, context)
    """

    def __init__(
        self,
        attack_type: ByzantineAttackType,
        severity: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize Byzantine attacker.

        Args:
            attack_type: Type of attack to perform
            severity: Attack severity multiplier (0.0 to inf)
            seed: Random seed for reproducibility
        """
        self.attack_type = attack_type
        self.severity = severity

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def attack(
        self,
        honest_weights: List[np.ndarray],
        context: Optional[Dict] = None
    ) -> List[np.ndarray]:
        """
        Generate malicious update based on attack type.

        Args:
            honest_weights: Model weights from honest training
            context: Additional context (peer weights, round number, etc.)

        Returns:
            Malicious weights to send to server
        """
        if self.attack_type == ByzantineAttackType.GAUSSIAN_NOISE:
            return self._gaussian_noise_attack(honest_weights)

        elif self.attack_type == ByzantineAttackType.SIGN_FLIP:
            return self._sign_flip_attack(honest_weights)

        elif self.attack_type == ByzantineAttackType.LABEL_FLIP:
            # This requires retraining - handled at client level
            return honest_weights  # Placeholder

        elif self.attack_type == ByzantineAttackType.TARGETED_POISON:
            return self._targeted_poison_attack(honest_weights, context)

        elif self.attack_type == ByzantineAttackType.MODEL_REPLACEMENT:
            return self._model_replacement_attack(honest_weights)

        elif self.attack_type == ByzantineAttackType.ADAPTIVE:
            return self._adaptive_attack(honest_weights, context)

        elif self.attack_type == ByzantineAttackType.SYBIL:
            return self._sybil_attack(honest_weights, context)

        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")

    # =========================================================================
    # Attack Implementations
    # =========================================================================

    def _gaussian_noise_attack(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Attack Type 1: Gaussian Noise

        Adds random Gaussian noise to all parameters.
        Simple but easily detected by robust aggregators.

        Severity: σ = 10.0 * severity
        """
        sigma = 10.0 * self.severity
        return [w + np.random.normal(0, sigma, w.shape) for w in weights]

    def _sign_flip_attack(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Attack Type 2: Sign Flip

        Negates all gradient directions, pushing model away from convergence.
        More sophisticated than noise, harder to detect.

        Paper: Fang et al., "Local Model Poisoning Attacks", USENIX 2020
        """
        return [-self.severity * w for w in weights]

    def _label_flip_attack(
        self,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        config: any,
        device: str = 'cpu'
    ) -> List[np.ndarray]:
        """
        Attack Type 3: Label Flip

        Trains on corrupted labels (e.g., 0→1, 1→0, etc.)
        Creates subtle but persistent model degradation.

        Note: This requires actual retraining with flipped labels.
        Returns weights trained on corrupted data.
        """
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(config.local_epochs):
            for data, target in train_data:
                data, target = data.to(device), target.to(device)

                # Flip labels with severity probability
                if np.random.rand() < self.severity:
                    num_classes = config.num_classes if hasattr(config, 'num_classes') else 10
                    target = (target + np.random.randint(1, num_classes)) % num_classes

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return [param.cpu().detach().numpy().copy() for param in model.parameters()]

    def _targeted_poison_attack(
        self,
        weights: List[np.ndarray],
        context: Optional[Dict] = None
    ) -> List[np.ndarray]:
        """
        Attack Type 4: Targeted Poison (Backdoor)

        Subtly alters weights to create backdoor behavior.
        Model performs well on normal data but fails on trigger patterns.

        Paper: Bagdasaryan et al., "How To Backdoor FL", AISTATS 2020

        Strategy: Amplify gradient updates by severity factor to dominate aggregation.
        """
        # Amplify gradient magnitude to dominate aggregation
        amplification = 10.0 * self.severity
        return [amplification * w for w in weights]

    def _model_replacement_attack(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Attack Type 5: Model Replacement

        Sends completely random or adversarially crafted weights.
        Most extreme attack - tries to directly control global model.

        Paper: Bhagoji et al., "Analyzing FL against Backdoors", ICML 2019
        """
        # Return completely random weights with same shape
        return [np.random.randn(*w.shape) * self.severity for w in weights]

    def _adaptive_attack(
        self,
        weights: List[np.ndarray],
        context: Optional[Dict] = None
    ) -> List[np.ndarray]:
        """
        Attack Type 6: Adaptive Attack

        Adapts to defense mechanism by analyzing peer gradients.
        Most sophisticated - tries to evade Byzantine-robust aggregation.

        Paper: Baruch et al., "A Little Is Enough", NeurIPS 2019

        Strategy: If peer weights available, position malicious update
        near cluster of honest weights to evade distance-based detection.
        """
        if context is None or 'peer_weights' not in context:
            # Fall back to sign flip if no peer info
            return self._sign_flip_attack(weights)

        peer_weights_list = context['peer_weights']

        # Calculate mean of peer weights (honest cluster approximation)
        mean_weights = []
        for i in range(len(weights)):
            peer_params = [pw[i] for pw in peer_weights_list]
            mean_weights.append(np.mean(peer_params, axis=0))

        # Position malicious update near honest cluster but in opposite direction
        # This evades distance-based aggregators like Krum
        attack_weights = []
        for w_honest, w_mean in zip(weights, mean_weights):
            # Compute attack direction (away from honest)
            attack_direction = w_mean - w_honest
            # Scale by severity and add to mean (stays near cluster)
            attack_weights.append(w_mean + self.severity * attack_direction)

        return attack_weights

    def _sybil_attack(
        self,
        weights: List[np.ndarray],
        context: Optional[Dict] = None
    ) -> List[np.ndarray]:
        """
        Attack Type 7: Sybil Attack

        Coordinates multiple Byzantine clients to amplify effect.
        Each client in Sybil coalition sends similar malicious updates.

        Strategy: All Sybil clients send identical strong sign-flip updates
        to maximize impact on aggregation.

        Note: Requires multiple Byzantine clients coordinating.
        Context should contain 'sybil_id' to synchronize updates.
        """
        # All Sybil clients send identical sign-flipped updates
        # This amplifies the attack when multiple clients coordinate
        attack_strength = 5.0 * self.severity
        return [-attack_strength * w for w in weights]


def create_byzantine_client(
    client_id: str,
    model: nn.Module,
    train_data: torch.utils.data.DataLoader,
    config: any,
    attack_type: ByzantineAttackType = ByzantineAttackType.GAUSSIAN_NOISE,
    severity: float = 1.0,
    device: str = 'cpu'
):
    """
    Factory function to create a Byzantine client with specified attack.

    Args:
        client_id: Unique identifier
        model: PyTorch model
        train_data: DataLoader with local data
        config: Training configuration
        attack_type: Type of Byzantine attack
        severity: Attack severity (0.0 to inf)
        device: 'cpu' or 'cuda'

    Returns:
        Byzantine client wrapper with attack strategy
    """
    class ByzantineClient:
        def __init__(self):
            self.client_id = client_id
            self.model = model.to(device)
            self.train_data = train_data
            self.config = config
            self.device = device
            self.attacker = ByzantineAttacker(attack_type, severity)
            self.num_samples = len(train_data.dataset)

            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate
            )
            self.criterion = nn.CrossEntropyLoss()

        def get_model_weights(self) -> List[np.ndarray]:
            return [p.cpu().detach().numpy().copy() for p in self.model.parameters()]

        def set_model_weights(self, weights: List[np.ndarray]):
            with torch.no_grad():
                for param, new_weight in zip(self.model.parameters(), weights):
                    param.copy_(torch.tensor(new_weight, device=self.device))

        def train(self, context: Optional[Dict] = None) -> Dict:
            """Train and return malicious update."""

            # Special handling for label flip attack
            if self.attacker.attack_type == ByzantineAttackType.LABEL_FLIP:
                malicious_weights = self.attacker._label_flip_attack(
                    self.model, self.train_data, self.config, self.device
                )
            else:
                # Train normally first
                self.model.train()
                for epoch in range(self.config.local_epochs):
                    for data, target in self.train_data:
                        data, target = data.to(self.device), target.to(self.device)
                        self.optimizer.zero_grad()
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        loss.backward()
                        self.optimizer.step()

                # Then apply Byzantine attack to weights
                honest_weights = self.get_model_weights()
                malicious_weights = self.attacker.attack(honest_weights, context)

            return {
                'weights': malicious_weights,
                'num_samples': self.num_samples,
                'loss': 0.0,  # Fake metrics
                'accuracy': 0.0,  # Fake metrics
                'client_id': self.client_id,
                'is_byzantine': True,
                'attack_type': self.attacker.attack_type.value
            }

    return ByzantineClient()


# =========================================================================
# Utility Functions
# =========================================================================

def evaluate_attack_effectiveness(
    honest_accuracy: float,
    attacked_accuracy: float,
    baseline_name: str,
    attack_type: ByzantineAttackType
) -> Dict:
    """
    Evaluate how effective an attack was against a defense.

    Returns:
        Dictionary with effectiveness metrics
    """
    accuracy_drop = honest_accuracy - attacked_accuracy
    relative_drop = accuracy_drop / honest_accuracy if honest_accuracy > 0 else 0

    # Categorize attack success
    if relative_drop > 0.5:
        success = "Critical"
    elif relative_drop > 0.2:
        success = "Significant"
    elif relative_drop > 0.05:
        success = "Moderate"
    else:
        success = "Negligible"

    return {
        'baseline': baseline_name,
        'attack_type': attack_type.value,
        'honest_accuracy': honest_accuracy,
        'attacked_accuracy': attacked_accuracy,
        'accuracy_drop': accuracy_drop,
        'relative_drop': relative_drop,
        'attack_success': success
    }


def generate_attack_comparison_table(results: List[Dict]) -> str:
    """
    Generate markdown table comparing attack effectiveness.

    Args:
        results: List of evaluation results

    Returns:
        Markdown formatted table string
    """
    table = "| Baseline | Attack Type | Honest Acc | Attacked Acc | Drop | Success |\n"
    table += "|----------|-------------|------------|--------------|------|--------|\n"

    for r in results:
        table += f"| {r['baseline']:<12} | "
        table += f"{r['attack_type']:<15} | "
        table += f"{r['honest_accuracy']:.2%} | "
        table += f"{r['attacked_accuracy']:.2%} | "
        table += f"{r['accuracy_drop']:.2%} | "
        table += f"{r['attack_success']} |\n"

    return table
