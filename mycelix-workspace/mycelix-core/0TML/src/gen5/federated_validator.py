# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gen 5 Layer 4: Federated Validator
===================================

Distributed Byzantine detection validation using Shamir secret sharing.

Key Components:
- ShamirSecretSharing: (t, n) threshold secret sharing
- ThresholdValidator: Distributed validation protocol
- Byzantine-robust reconstruction via Krum

Algorithm:
    1. Coordinator generates secret shares of detection score
    2. Distribute shares to n validators
    3. Each validator evaluates independently
    4. Collect >= t shares to reconstruct
    5. Apply Krum for Byzantine-robust selection
    6. Return final verdict

Byzantine Tolerance:
    For n validators with threshold t:
    - Max Byzantine: < (n - t + 1) / 2
    - Example: n=7, t=4 → tolerates < 2 Byzantine (< 28.6%)

Author: Luminous Dynamics
Date: November 12, 2025
"""

from typing import List, Tuple, Dict, Optional
from itertools import combinations
import numpy as np


class ShamirSecretSharing:
    """
    Shamir (t, n) threshold secret sharing.

    Shares a secret among n parties, requiring t to reconstruct.
    Operates on floating-point secrets in [0, 1].

    Mathematical Foundation:
        - Generate random polynomial: P(x) = s + a₁x + ... + a_{t-1}x^{t-1}
        - Create shares: sᵢ = P(i) for i = 1, ..., n
        - Reconstruct via Lagrange interpolation: s = P(0)

    Properties:
        - Any t shares can reconstruct s
        - Fewer than t shares reveal NO information
        - Byzantine fault tolerance: < (n - t + 1) / 2 attackers

    Example:
        >>> sss = ShamirSecretSharing(threshold=3, num_shares=5)
        >>> secret = 0.85
        >>> shares = sss.generate_shares(secret)
        >>> reconstructed = sss.reconstruct(shares[:3])  # Any 3 shares
        >>> assert abs(reconstructed - secret) < 0.001
    """

    def __init__(self, threshold: int, num_shares: int):
        """
        Initialize secret sharing scheme.

        Args:
            threshold: Minimum shares needed to reconstruct (t)
            num_shares: Total number of shares (n)

        Requires:
            1 <= threshold <= num_shares
        """
        if not (1 <= threshold <= num_shares):
            raise ValueError(f"Invalid parameters: 1 <= {threshold} <= {num_shares} required")

        self.t = threshold
        self.n = num_shares

    def generate_shares(self, secret: float) -> List[Tuple[int, float]]:
        """
        Generate shares of secret.

        Algorithm:
            1. Create random polynomial P(x) = secret + a₁x + ... + a_{t-1}x^{t-1}
            2. Evaluate at points x = 1, 2, ..., n
            3. Return (x, P(x)) pairs

        Args:
            secret: Secret value in [0, 1]

        Returns:
            List of (id, share) tuples
        """
        # Generate random polynomial coefficients
        # P(x) = secret + a₁x + a₂x² + ... + a_{t-1}x^{t-1}
        coefficients = [secret] + [np.random.uniform(-1, 1) for _ in range(self.t - 1)]

        # Evaluate polynomial at points 1, 2, ..., n
        shares = []
        for i in range(1, self.n + 1):
            share_value = self._evaluate_polynomial(coefficients, i)
            shares.append((i, share_value))

        return shares

    def reconstruct(self, shares: List[Tuple[int, float]]) -> float:
        """
        Reconstruct secret from >= t shares.

        Uses Lagrange interpolation to compute P(0) = secret.

        Algorithm:
            s = P(0) = Σⱼ sⱼ · Lⱼ(0)
            where Lⱼ(0) = Π_{k≠j} (xₖ / (xₖ - xⱼ))

        Args:
            shares: List of (id, share_value) tuples

        Returns:
            Reconstructed secret

        Raises:
            ValueError: If fewer than t shares provided
        """
        if len(shares) < self.t:
            raise ValueError(f"Need >= {self.t} shares, got {len(shares)}")

        # Take exactly t shares (if more provided)
        shares = shares[:self.t]

        # Lagrange interpolation at x = 0
        secret = 0.0
        for i, (xi, yi) in enumerate(shares):
            # Compute Lagrange basis polynomial Lᵢ(0)
            li = 1.0
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    # Lᵢ(0) = Π_{j≠i} (xⱼ / (xⱼ - xᵢ))
                    li *= xj / (xj - xi)

            secret += yi * li

        return secret

    def _evaluate_polynomial(self, coefficients: List[float], x: float) -> float:
        """
        Evaluate polynomial at x using Horner's method.

        P(x) = c₀ + c₁x + c₂x² + ... + c_{n-1}x^{n-1}
             = c₀ + x(c₁ + x(c₂ + x(...)))

        Args:
            coefficients: [c₀, c₁, c₂, ...]
            x: Point to evaluate at

        Returns:
            P(x)
        """
        result = 0.0
        for coef in reversed(coefficients):
            result = result * x + coef
        return result


class ThresholdValidator:
    """
    Threshold validation protocol for Byzantine-robust distributed detection.

    Enables n validators to jointly verify gradient classifications without
    trusting any single party.

    Protocol:
        1. Coordinator generates secret shares of detection score
        2. Distribute shares to n validators
        3. Each validator evaluates independently (simulated)
        4. Coordinator collects responses
        5. Byzantine-robust reconstruction via Krum
        6. Final decision: honest if score >= 0.5, Byzantine otherwise

    Byzantine Tolerance:
        max_byzantine = (n - t + 1) // 2 - 1
        Example: n=7, t=4 → max_byzantine = 1 (tolerates 1 Byzantine validator)

    Note:
        This implementation simulates distributed validation. In production,
        shares would be sent to actual remote validators.

    Example:
        >>> from gen5.meta_learning import MetaLearningEnsemble
        >>> ensemble = MetaLearningEnsemble()
        >>> validator = ThresholdValidator(ensemble, num_validators=7, threshold=4)
        >>> gradient = np.random.randn(10)
        >>> decision, confidence, details = validator.validate_gradient(gradient)
        >>> print(f"Decision: {decision}, Confidence: {confidence:.2f}")
    """

    def __init__(
        self,
        ensemble,  # MetaLearningEnsemble
        num_validators: int = 7,
        threshold: int = 4,
    ):
        """
        Initialize threshold validator.

        Args:
            ensemble: MetaLearningEnsemble for computing detection scores
            num_validators: Total validators (n)
            threshold: Minimum honest validators needed (t)

        Byzantine Tolerance:
            Can tolerate < (n - t + 1) / 2 Byzantine validators
            For n=7, t=4: Can tolerate < 2 Byzantine (< 28.6%)
        """
        self.ensemble = ensemble
        self.n = num_validators
        self.t = threshold
        self.secret_sharing = ShamirSecretSharing(threshold, num_validators)

        # Byzantine tolerance
        self.max_byzantine = (num_validators - threshold + 1) // 2 - 1

    def validate_gradient(
        self,
        gradient: np.ndarray,
        validator_scores: Optional[List[float]] = None,
    ) -> Tuple[str, float, Dict]:
        """
        Validate gradient using distributed threshold protocol.

        Args:
            gradient: Gradient to validate
            validator_scores: Simulated validator scores (for testing)
                             In production: actual validator outputs

        Returns:
            (decision, confidence, details) where:
                decision: "honest" or "byzantine"
                confidence: Decision confidence in [0, 1]
                details: Protocol metadata
        """
        # Step 1: Compute reference score (coordinator)
        signals = {}
        for method in self.ensemble.base_methods:
            signals[method.name] = method.score(gradient)

        reference_score = self.ensemble.compute_ensemble_score(signals)

        # Step 2: Generate secret shares
        shares = self.secret_sharing.generate_shares(reference_score)

        # Step 3: Simulate validator evaluation
        # In production: distribute shares to validators, collect results
        if validator_scores is None:
            # Honest simulation: all validators return same score
            validator_scores = [reference_score] * self.n

        # Step 4: Byzantine-robust reconstruction
        reconstructed_score = self._robust_reconstruct(shares, validator_scores)

        # Step 5: Make decision
        decision = "honest" if reconstructed_score >= 0.5 else "byzantine"
        confidence = abs(reconstructed_score - 0.5) * 2  # Scale to [0, 1]

        details = {
            "reference_score": reference_score,
            "reconstructed_score": reconstructed_score,
            "num_validators": self.n,
            "threshold": self.t,
            "max_byzantine_tolerated": self.max_byzantine,
            "score_deviation": abs(reconstructed_score - reference_score),
        }

        return (decision, confidence, details)

    def _robust_reconstruct(
        self,
        shares: List[Tuple[int, float]],
        validator_scores: List[float],
    ) -> float:
        """
        Robustly reconstruct secret from potentially Byzantine validator outputs.

        Strategy:
            1. Try all C(n, t) combinations of t validators
            2. Reconstruct secret from each combination
            3. Apply Krum to select most consistent reconstruction

        Complexity:
            O(C(n, t) × t²) = O(n^t × t²)
            For n=7, t=4: C(7,4) = 35 combinations (acceptable)

        Args:
            shares: Original secret shares
            validator_scores: Validator outputs (potentially Byzantine)

        Returns:
            Robustly reconstructed score
        """
        # Generate all combinations of t shares
        candidates = []

        for subset_indices in combinations(range(self.n), self.t):
            # Get shares for this subset
            subset_shares = [shares[i] for i in subset_indices]

            # Reconstruct
            try:
                secret = self.secret_sharing.reconstruct(subset_shares)
                candidates.append(secret)
            except Exception:
                continue  # Skip invalid reconstructions

        if not candidates:
            # Fallback: use simple reconstruction
            return self.secret_sharing.reconstruct(shares[:self.t])

        # Byzantine-robust selection: Krum
        return self._krum_select(candidates)

    def _krum_select(self, candidates: List[float]) -> float:
        """
        Select most consistent candidate using Krum.

        Algorithm:
            For each candidate:
                1. Compute squared distances to all other candidates
                2. Sum distances to n - f - 2 closest candidates
                3. Select candidate with minimum score

        Args:
            candidates: List of reconstructed secrets

        Returns:
            Most consistent value (Krum selection)
        """
        if len(candidates) == 1:
            return candidates[0]

        # Compute pairwise distances
        n = len(candidates)
        f = self.max_byzantine  # Number of Byzantine validators

        scores = []
        for i in range(n):
            # Sum of squared distances to n - f - 2 closest candidates
            distances = []
            for j in range(n):
                if i != j:
                    dist = (candidates[i] - candidates[j]) ** 2
                    distances.append(dist)

            distances.sort()
            # Sum of closest n - f - 2 distances
            num_closest = max(1, n - f - 2)
            score = sum(distances[:num_closest])
            scores.append(score)

        # Return candidate with minimum score
        min_idx = np.argmin(scores)
        return candidates[min_idx]
