#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Byzantine Defense Integration Tests
====================================

Tests the complete FL training round with Byzantine detection,
simulating the Holochain zome interactions:

1. gradient_storage: Gradient commitments
2. cbd_zome: Causal Byzantine Detection
3. reputation_tracker_v2: Multi-round reputation with EMA
4. defense_coordinator: Three-layer defense orchestration

These tests validate:
- Fixed-point arithmetic matches Python
- Defense layers work together correctly
- Byzantine nodes are detected and excluded
- Honest nodes' gradients are aggregated properly

Author: Luminous Dynamics Research Team
Date: December 2025
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple
import hashlib
import time

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# =============================================================================
# Q16.16 Fixed-Point Simulation (matches Rust implementation)
# =============================================================================

FP_SCALE = 65536  # 2^16


def fp_from_f64(x: float) -> int:
    """Convert float to Q16.16 fixed-point."""
    return int(x * FP_SCALE)


def fp_to_f64(x: int) -> float:
    """Convert Q16.16 fixed-point to float."""
    return x / FP_SCALE


def fp_mul(a: int, b: int) -> int:
    """Multiply two Q16.16 fixed-point numbers."""
    return (a * b) // FP_SCALE


def fp_div(a: int, b: int) -> int:
    """Divide two Q16.16 fixed-point numbers."""
    if b == 0:
        return 2**31 - 1 if a >= 0 else -(2**31)
    return (a * FP_SCALE) // b


def fp_sqrt(x: int) -> int:
    """Integer square root for Q16.16 fixed-point."""
    if x <= 0:
        return 0
    guess = x // 2
    if guess == 0:
        guess = 1
    for _ in range(5):
        div = fp_div(x, guess)
        guess = (guess + div) // 2
    return guess


def fp_abs(x: int) -> int:
    """Absolute value."""
    return -x if x < 0 else x


# =============================================================================
# Data Structures (match Rust zome entries)
# =============================================================================

class ReputationState(Enum):
    TRUSTED = "Trusted"
    SUSPICIOUS = "Suspicious"
    UNTRUSTED = "Untrusted"


class AggregationMethod(Enum):
    MEAN = "Mean"
    COORDINATE_MEDIAN = "CoordinateMedian"
    KRUM = "Krum"
    MULTI_KRUM = "MultiKrum"
    TRIMMED_MEAN = "TrimmedMean"


@dataclass
class GradientCommitment:
    """Matches gradient_storage zome entry."""
    node_id: str
    round: int
    commitment_hash: bytes
    gradient_fp: List[int]  # Q16.16 fixed-point
    timestamp: int


@dataclass
class ReputationEntry:
    """Matches reputation_tracker_v2 zome entry."""
    node_id: str
    reputation: int  # Q16.16
    state: ReputationState
    z_score_ema: int
    z_score_variance: int
    times_flagged: int
    rounds_participated: int


@dataclass
class CausalAnalysis:
    """Matches cbd_zome analysis entry."""
    node_id: str
    round: int
    z_score: int  # Q16.16
    causal_effect: int  # Q16.16
    is_byzantine: bool
    confidence: int  # Q16.16


@dataclass
class DefenseResult:
    """Matches defense_coordinator result entry."""
    round: int
    layer1_flagged: List[str]
    layer2_flagged: List[str]
    detected_byzantine: List[str]
    aggregated_gradient: List[int]  # Q16.16
    aggregation_method: AggregationMethod
    honest_count: int


# =============================================================================
# Simulated Zome Functions
# =============================================================================

class GradientStorageZome:
    """Simulates gradient_storage zome."""

    def __init__(self):
        self.commitments: Dict[str, GradientCommitment] = {}

    def submit_gradient(self, node_id: str, gradient: np.ndarray, round_num: int) -> GradientCommitment:
        """Submit a gradient commitment."""
        # Convert to fixed-point
        gradient_fp = [fp_from_f64(g) for g in gradient]

        # Create commitment hash
        gradient_bytes = np.array(gradient).tobytes()
        commitment_hash = hashlib.sha256(gradient_bytes).digest()

        commitment = GradientCommitment(
            node_id=node_id,
            round=round_num,
            commitment_hash=commitment_hash,
            gradient_fp=gradient_fp,
            timestamp=int(time.time()),
        )

        self.commitments[f"{round_num}:{node_id}"] = commitment
        return commitment

    def get_round_gradients(self, round_num: int) -> List[GradientCommitment]:
        """Get all gradients for a round."""
        return [c for c in self.commitments.values() if c.round == round_num]


class CBDZome:
    """Simulates cbd_zome (Causal Byzantine Detection)."""

    def __init__(self, z_threshold: float = 3.0, causal_threshold: float = 0.2):
        self.z_threshold = fp_from_f64(z_threshold)
        self.causal_threshold = fp_from_f64(causal_threshold)

    def analyze_node(self, node_gradient: List[int], all_gradients: List[List[int]],
                     node_id: str, round_num: int) -> CausalAnalysis:
        """Perform causal analysis on a single node."""
        # Compute mean of other nodes
        other_gradients = [g for i, g in enumerate(all_gradients) if g != node_gradient]
        if not other_gradients:
            return CausalAnalysis(
                node_id=node_id, round=round_num, z_score=0,
                causal_effect=0, is_byzantine=False, confidence=FP_SCALE
            )

        n = len(other_gradients)
        dim = len(node_gradient)
        mean_other = [sum(g[d] for g in other_gradients) // n for d in range(dim)]

        # Compute norms
        node_norm_sq = sum(fp_mul(g, g) for g in node_gradient)
        other_norm_sq = sum(fp_mul(g, g) for g in mean_other)

        node_norm = fp_sqrt(node_norm_sq)
        other_norm = fp_sqrt(other_norm_sq)

        # Compute cosine similarity
        dot = sum(fp_mul(node_gradient[d], mean_other[d]) for d in range(dim))
        if node_norm > 0 and other_norm > 0:
            cosine = fp_div(dot, fp_mul(node_norm, other_norm))
        else:
            cosine = FP_SCALE

        # Compute magnitude ratio
        if other_norm > 0:
            magnitude_ratio = fp_div(node_norm, other_norm)
        else:
            magnitude_ratio = FP_SCALE

        # Estimate causal effect
        if cosine < 0:
            # Adversarial direction
            log_approx = FP_SCALE + fp_div(magnitude_ratio, 2 * FP_SCALE)
            causal_effect = fp_mul(fp_abs(cosine), log_approx)
        else:
            # Aligned direction
            if magnitude_ratio > 3 * FP_SCALE:
                causal_effect = fp_mul(magnitude_ratio - 3 * FP_SCALE, fp_from_f64(0.05))
            else:
                causal_effect = -fp_mul(cosine, fp_from_f64(0.01))

        # Compute z-score using all norms
        all_norms = [fp_sqrt(sum(fp_mul(g[d], g[d]) for d in range(dim))) for g in all_gradients]
        sorted_norms = sorted(all_norms)
        n_norms = len(sorted_norms)
        median = sorted_norms[n_norms // 2]

        deviations = sorted([fp_abs(n - median) for n in all_norms])
        mad = deviations[n_norms // 2]
        mad = max(mad, 1)

        k = fp_from_f64(0.6745)
        z_score = fp_mul(k, fp_div(node_norm - median, mad))

        # Determine if Byzantine
        is_byzantine = (
            fp_abs(z_score) > self.z_threshold or
            causal_effect > self.causal_threshold
        )

        # Compute confidence (inversely proportional to uncertainty)
        confidence = max(0, FP_SCALE - fp_abs(z_score - self.z_threshold) // 10)

        return CausalAnalysis(
            node_id=node_id,
            round=round_num,
            z_score=z_score,
            causal_effect=causal_effect,
            is_byzantine=is_byzantine,
            confidence=confidence,
        )


class ReputationTrackerV2Zome:
    """Simulates reputation_tracker_v2 zome."""

    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = fp_from_f64(ema_alpha)
        self.one_minus_alpha = FP_SCALE - self.ema_alpha
        self.reputations: Dict[str, ReputationEntry] = {}

        # Thresholds
        self.suspicious_threshold = fp_from_f64(0.7)
        self.untrusted_threshold = fp_from_f64(0.3)

    def get_or_create(self, node_id: str) -> ReputationEntry:
        """Get or create reputation entry for a node."""
        if node_id not in self.reputations:
            self.reputations[node_id] = ReputationEntry(
                node_id=node_id,
                reputation=FP_SCALE,  # Start at 1.0
                state=ReputationState.TRUSTED,
                z_score_ema=0,
                z_score_variance=0,
                times_flagged=0,
                rounds_participated=0,
            )
        return self.reputations[node_id]

    def update_reputation(self, node_id: str, z_score: int, is_byzantine: bool) -> ReputationEntry:
        """Update reputation based on behavior."""
        entry = self.get_or_create(node_id)

        # Update z-score EMA
        old_ema = entry.z_score_ema
        entry.z_score_ema = fp_mul(self.ema_alpha, z_score) + fp_mul(self.one_minus_alpha, old_ema)

        # Update variance for sudden change detection
        diff = fp_abs(z_score - old_ema)
        entry.z_score_variance = fp_mul(self.ema_alpha, diff) + fp_mul(self.one_minus_alpha, entry.z_score_variance)

        # Update reputation
        if is_byzantine:
            entry.times_flagged += 1
            # Decay reputation by 20%
            penalty = fp_from_f64(0.2)
            entry.reputation = fp_mul(entry.reputation, FP_SCALE - penalty)
        else:
            # Slow recovery
            recovery = fp_from_f64(0.02)
            new_rep = entry.reputation + fp_mul(FP_SCALE - entry.reputation, recovery)
            entry.reputation = min(FP_SCALE, new_rep)

        # Update state
        if entry.reputation >= self.suspicious_threshold:
            entry.state = ReputationState.TRUSTED
        elif entry.reputation >= self.untrusted_threshold:
            entry.state = ReputationState.SUSPICIOUS
        else:
            entry.state = ReputationState.UNTRUSTED

        entry.rounds_participated += 1
        self.reputations[node_id] = entry

        return entry


class DefenseCoordinatorZome:
    """Simulates defense_coordinator zome."""

    def __init__(self, cbd: CBDZome, reputation: ReputationTrackerV2Zome):
        self.cbd = cbd
        self.reputation = reputation

    def run_defense(self, gradients: List[Tuple[str, List[int]]], round_num: int) -> DefenseResult:
        """Run three-layer defense on submitted gradients."""
        node_ids = [g[0] for g in gradients]
        gradient_list = [g[1] for g in gradients]

        # Layer 1: Statistical detection (CBD)
        layer1_flagged = []
        analyses = []
        for i, (node_id, gradient) in enumerate(gradients):
            analysis = self.cbd.analyze_node(gradient, gradient_list, node_id, round_num)
            analyses.append(analysis)
            if analysis.is_byzantine:
                layer1_flagged.append(node_id)

        # Layer 2: Reputation filtering
        layer2_flagged = []
        for node_id in node_ids:
            rep = self.reputation.get_or_create(node_id)
            if rep.state == ReputationState.UNTRUSTED:
                layer2_flagged.append(node_id)

        # Combine detections
        detected_byzantine = list(set(layer1_flagged + layer2_flagged))

        # Update reputations
        for analysis in analyses:
            self.reputation.update_reputation(
                analysis.node_id,
                analysis.z_score,
                analysis.is_byzantine
            )

        # Filter honest gradients
        honest_gradients = [
            g for node_id, g in gradients
            if node_id not in detected_byzantine
        ]

        # Layer 3: Robust aggregation (Multi-Krum)
        if len(honest_gradients) < 2:
            # Fallback to mean of all if not enough honest nodes
            aggregated = self._aggregate_mean(gradient_list)
            method = AggregationMethod.MEAN
        else:
            aggregated = self._aggregate_multi_krum(honest_gradients, k=max(1, len(honest_gradients) // 2))
            method = AggregationMethod.MULTI_KRUM

        return DefenseResult(
            round=round_num,
            layer1_flagged=layer1_flagged,
            layer2_flagged=layer2_flagged,
            detected_byzantine=detected_byzantine,
            aggregated_gradient=aggregated,
            aggregation_method=method,
            honest_count=len(honest_gradients),
        )

    def _aggregate_mean(self, gradients: List[List[int]]) -> List[int]:
        """Simple mean aggregation."""
        n = len(gradients)
        dim = len(gradients[0])
        return [sum(g[d] for g in gradients) // n for d in range(dim)]

    def _aggregate_multi_krum(self, gradients: List[List[int]], k: int) -> List[int]:
        """Multi-Krum aggregation."""
        n = len(gradients)
        dim = len(gradients[0])
        f = n // 4  # Assume up to 25% adversarial

        # Compute pairwise distances
        distances = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = sum(
                    fp_mul(gradients[i][d] - gradients[j][d], gradients[i][d] - gradients[j][d])
                    for d in range(dim)
                )
                dist = fp_sqrt(dist_sq)
                distances[i][j] = dist
                distances[j][i] = dist

        # Compute Krum scores
        scores = []
        for i in range(n):
            sorted_dists = sorted(distances[i])
            # Sum of smallest n-f-1 distances (excluding self)
            score = sum(sorted_dists[1:max(2, n - f - 1)])
            scores.append((i, score))

        # Select k best
        scores.sort(key=lambda x: x[1])
        selected = [scores[i][0] for i in range(k)]

        # Average selected gradients
        result = []
        for d in range(dim):
            avg = sum(gradients[s][d] for s in selected) // k
            result.append(avg)

        return result


# =============================================================================
# Test Cases
# =============================================================================

class TestByzantineDetectionIntegration:
    """Integration tests for Byzantine detection system."""

    @pytest.fixture
    def setup_zomes(self):
        """Create simulated zome instances."""
        gradient_storage = GradientStorageZome()
        cbd = CBDZome()
        reputation = ReputationTrackerV2Zome()
        defense = DefenseCoordinatorZome(cbd, reputation)
        return gradient_storage, cbd, reputation, defense

    def test_honest_nodes_pass_detection(self, setup_zomes):
        """Test that honest nodes are not flagged."""
        gradient_storage, cbd, reputation, defense = setup_zomes

        # Create honest gradients (similar directions, normal magnitudes)
        np.random.seed(42)
        base_gradient = np.random.randn(10)

        gradients = []
        for i in range(10):
            # Small random perturbation
            noise = np.random.randn(10) * 0.1
            gradient = base_gradient + noise
            gradient_fp = [fp_from_f64(g) for g in gradient]
            gradients.append((f"node_{i}", gradient_fp))

        # Run defense
        result = defense.run_defense(gradients, round_num=1)

        # All nodes should be honest
        assert len(result.detected_byzantine) == 0, f"Honest nodes incorrectly flagged: {result.detected_byzantine}"
        assert result.honest_count == 10

    def test_adversarial_gradient_attack_detected(self, setup_zomes):
        """Test that sign-flipping attacks are detected."""
        gradient_storage, cbd, reputation, defense = setup_zomes

        np.random.seed(42)
        base_gradient = np.random.randn(10) + 1.0  # Bias positive

        gradients = []
        # 7 honest nodes
        for i in range(7):
            noise = np.random.randn(10) * 0.1
            gradient = base_gradient + noise
            gradient_fp = [fp_from_f64(g) for g in gradient]
            gradients.append((f"honest_{i}", gradient_fp))

        # 3 Byzantine nodes (sign-flipping attack)
        for i in range(3):
            # Flip the gradient direction
            gradient = -base_gradient * 2  # Opposite direction, larger magnitude
            gradient_fp = [fp_from_f64(g) for g in gradient]
            gradients.append((f"byzantine_{i}", gradient_fp))

        # Run defense
        result = defense.run_defense(gradients, round_num=1)

        # At least some Byzantine nodes should be detected
        byzantine_detected = [n for n in result.detected_byzantine if n.startswith("byzantine")]
        assert len(byzantine_detected) >= 2, f"Too few Byzantine nodes detected: {byzantine_detected}"

    def test_scaling_attack_detected(self, setup_zomes):
        """Test that scaling attacks are detected."""
        gradient_storage, cbd, reputation, defense = setup_zomes

        np.random.seed(42)
        base_gradient = np.random.randn(10)

        gradients = []
        # 7 honest nodes
        for i in range(7):
            noise = np.random.randn(10) * 0.1
            gradient = base_gradient + noise
            gradient_fp = [fp_from_f64(g) for g in gradient]
            gradients.append((f"honest_{i}", gradient_fp))

        # 3 Byzantine nodes (scaling attack)
        for i in range(3):
            # Same direction but 10x magnitude
            gradient = base_gradient * 10
            gradient_fp = [fp_from_f64(g) for g in gradient]
            gradients.append((f"byzantine_{i}", gradient_fp))

        # Run defense
        result = defense.run_defense(gradients, round_num=1)

        # Byzantine nodes with extreme magnitudes should be detected
        byzantine_detected = [n for n in result.detected_byzantine if n.startswith("byzantine")]
        assert len(byzantine_detected) >= 1, "Scaling attack not detected"

    def test_reputation_decay_over_rounds(self, setup_zomes):
        """Test reputation decay for repeatedly bad actors."""
        gradient_storage, cbd, reputation, defense = setup_zomes

        np.random.seed(42)
        base_gradient = np.random.randn(10)

        # Run multiple rounds with a persistent attacker
        for round_num in range(5):
            gradients = []

            # 5 honest nodes
            for i in range(5):
                noise = np.random.randn(10) * 0.1
                gradient = base_gradient + noise
                gradient_fp = [fp_from_f64(g) for g in gradient]
                gradients.append((f"honest_{i}", gradient_fp))

            # 1 persistent Byzantine node
            gradient = -base_gradient * 3
            gradient_fp = [fp_from_f64(g) for g in gradient]
            gradients.append(("byzantine_0", gradient_fp))

            defense.run_defense(gradients, round_num=round_num)

        # Check reputation of Byzantine node
        rep = reputation.reputations.get("byzantine_0")
        assert rep is not None, "Byzantine node reputation not tracked"
        assert rep.state in [ReputationState.SUSPICIOUS, ReputationState.UNTRUSTED], \
            f"Byzantine node should have low reputation, got {rep.state}"
        assert rep.times_flagged >= 3, f"Byzantine node should be flagged multiple times, got {rep.times_flagged}"

    def test_sleeper_agent_detection(self, setup_zomes):
        """Test detection of sleeper agents (sudden behavior change)."""
        gradient_storage, cbd, reputation, defense = setup_zomes

        np.random.seed(42)
        base_gradient = np.random.randn(10)

        # Sleeper agent behaves honestly for 5 rounds
        for round_num in range(5):
            gradients = []
            for i in range(5):
                noise = np.random.randn(10) * 0.1
                gradient = base_gradient + noise
                gradient_fp = [fp_from_f64(g) for g in gradient]
                gradients.append((f"node_{i}", gradient_fp))

            defense.run_defense(gradients, round_num=round_num)

        # Check that sleeper's reputation is good
        rep_before = reputation.reputations.get("node_0")
        assert rep_before.state == ReputationState.TRUSTED

        # Now sleeper attacks in round 5
        gradients = []
        # Sleeper flips gradient
        gradient = -base_gradient * 5
        gradient_fp = [fp_from_f64(g) for g in gradient]
        gradients.append(("node_0", gradient_fp))

        # Other honest nodes
        for i in range(1, 5):
            noise = np.random.randn(10) * 0.1
            gradient = base_gradient + noise
            gradient_fp = [fp_from_f64(g) for g in gradient]
            gradients.append((f"node_{i}", gradient_fp))

        result = defense.run_defense(gradients, round_num=5)

        # Sleeper should be detected
        assert "node_0" in result.detected_byzantine or "node_0" in result.layer1_flagged, \
            "Sleeper agent attack not detected"

    def test_aggregation_excludes_byzantine(self, setup_zomes):
        """Test that Multi-Krum robustly aggregates despite Byzantine presence."""
        gradient_storage, cbd, reputation, defense = setup_zomes

        np.random.seed(42)
        # Create varied honest gradients (more realistic scenario)
        base = np.random.randn(10)

        gradients = []
        for i in range(7):
            # Honest nodes have gradients with reasonable variation
            noise = np.random.randn(10) * 0.3
            gradient = base + noise
            gradient_fp = [fp_from_f64(g) for g in gradient]
            gradients.append((f"honest_{i}", gradient_fp))

        # 3 Byzantine gradients with opposite direction (not extreme)
        for i in range(3):
            # Byzantine gradients are opposite but similar magnitude
            byzantine_gradient = -base * 1.5 + np.random.randn(10) * 0.2
            gradient_fp = [fp_from_f64(g) for g in byzantine_gradient]
            gradients.append((f"byzantine_{i}", gradient_fp))

        result = defense.run_defense(gradients, round_num=1)

        # Some Byzantine nodes should be detected via CBD's causal effect
        # (negative cosine with main cluster)
        layer1_byzantine = [n for n in result.layer1_flagged if n.startswith("byzantine")]

        # The key property: Multi-Krum should select gradients similar to honest cluster
        # even without perfect detection, because honest nodes cluster together
        aggregated_float = np.array([fp_to_f64(g) for g in result.aggregated_gradient])
        honest_mean = np.mean([
            [fp_to_f64(g) for g in grads[1]]
            for grads in gradients if grads[0].startswith("honest")
        ], axis=0)

        # Aggregated should be closer to honest mean than Byzantine mean
        byzantine_mean = np.mean([
            [fp_to_f64(g) for g in grads[1]]
            for grads in gradients if grads[0].startswith("byzantine")
        ], axis=0)

        honest_dist = np.linalg.norm(aggregated_float - honest_mean)
        byzantine_dist = np.linalg.norm(aggregated_float - byzantine_mean)

        # Robust aggregation should produce result closer to honest consensus
        assert honest_dist < byzantine_dist, \
            f"Aggregation closer to Byzantine! honest_dist={honest_dist:.3f}, byzantine_dist={byzantine_dist:.3f}"

    def test_multi_round_convergence(self, setup_zomes):
        """Test that defense improves over multiple rounds."""
        gradient_storage, cbd, reputation, defense = setup_zomes

        np.random.seed(42)
        base_gradient = np.random.randn(10)

        detection_rates = []

        for round_num in range(10):
            gradients = []

            # 7 honest
            for i in range(7):
                noise = np.random.randn(10) * 0.1
                gradient = base_gradient + noise
                gradient_fp = [fp_from_f64(g) for g in gradient]
                gradients.append((f"honest_{i}", gradient_fp))

            # 3 Byzantine (varying attacks)
            for i in range(3):
                if round_num % 2 == 0:
                    gradient = -base_gradient * 2
                else:
                    gradient = base_gradient * 5
                gradient_fp = [fp_from_f64(g) for g in gradient]
                gradients.append((f"byzantine_{i}", gradient_fp))

            result = defense.run_defense(gradients, round_num=round_num)

            byzantine_detected = len([n for n in result.detected_byzantine if n.startswith("byzantine")])
            detection_rates.append(byzantine_detected / 3)

        # Detection rate should improve or stay high
        late_detection = np.mean(detection_rates[-5:])
        assert late_detection >= 0.5, f"Detection rate too low: {late_detection}"


class TestFixedPointAccuracy:
    """Test that fixed-point matches float computations."""

    def test_cosine_similarity_accuracy(self):
        """Test cosine similarity within tolerance."""
        np.random.seed(42)

        for _ in range(100):
            a = np.random.randn(10)
            b = np.random.randn(10)

            # Float computation
            cos_float = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            # Fixed-point computation
            a_fp = [fp_from_f64(x) for x in a]
            b_fp = [fp_from_f64(x) for x in b]

            dot_fp = sum(fp_mul(x, y) for x, y in zip(a_fp, b_fp))
            norm_a_sq = sum(fp_mul(x, x) for x in a_fp)
            norm_b_sq = sum(fp_mul(x, x) for x in b_fp)
            norm_a_fp = fp_sqrt(norm_a_sq)
            norm_b_fp = fp_sqrt(norm_b_sq)

            if norm_a_fp > 0 and norm_b_fp > 0:
                cos_fp = fp_div(dot_fp, fp_mul(norm_a_fp, norm_b_fp))
                cos_fp_float = fp_to_f64(cos_fp)
            else:
                cos_fp_float = 0.0

            error = abs(cos_float - cos_fp_float)
            assert error < 0.05, f"Cosine error too large: {error}"

    def test_ema_update_accuracy(self):
        """Test EMA update within tolerance."""
        alpha = 0.3
        alpha_fp = fp_from_f64(alpha)
        one_minus_alpha = FP_SCALE - alpha_fp

        ema_float = 0.0
        ema_fp = 0

        np.random.seed(42)
        values = np.random.randn(50) * 2

        for v in values:
            # Float EMA
            ema_float = alpha * v + (1 - alpha) * ema_float

            # Fixed-point EMA
            v_fp = fp_from_f64(v)
            ema_fp = fp_mul(alpha_fp, v_fp) + fp_mul(one_minus_alpha, ema_fp)

        error = abs(ema_float - fp_to_f64(ema_fp))
        assert error < 0.01, f"EMA error too large: {error}"


class TestCrossZomeCommunication:
    """Test simulated cross-zome communication patterns."""

    def test_gradient_to_cbd_flow(self, setup_zomes):
        """Test gradient storage -> CBD flow."""
        gradient_storage, cbd, reputation, defense = setup_zomes

        np.random.seed(42)

        # Submit gradients
        for i in range(5):
            gradient = np.random.randn(10)
            gradient_storage.submit_gradient(f"node_{i}", gradient, round_num=1)

        # Get round gradients
        commitments = gradient_storage.get_round_gradients(round_num=1)
        assert len(commitments) == 5

        # Convert to CBD input format
        all_gradients = [c.gradient_fp for c in commitments]

        # Run CBD analysis
        for i, commitment in enumerate(commitments):
            analysis = cbd.analyze_node(
                commitment.gradient_fp,
                all_gradients,
                commitment.node_id,
                round_num=1
            )
            assert analysis.node_id == commitment.node_id

    def test_cbd_to_reputation_flow(self, setup_zomes):
        """Test CBD -> reputation tracker flow."""
        gradient_storage, cbd, reputation, defense = setup_zomes

        np.random.seed(42)
        base_gradient = np.random.randn(10)

        # Create analyses
        gradients = []
        for i in range(5):
            gradient = base_gradient + np.random.randn(10) * 0.1
            gradient_fp = [fp_from_f64(g) for g in gradient]
            gradients.append(gradient_fp)

        # One Byzantine
        byzantine_gradient = -base_gradient * 3
        gradients.append([fp_from_f64(g) for g in byzantine_gradient])

        # Run CBD
        for i, g in enumerate(gradients):
            node_id = f"node_{i}"
            analysis = cbd.analyze_node(g, gradients, node_id, round_num=1)

            # Update reputation
            reputation.update_reputation(node_id, analysis.z_score, analysis.is_byzantine)

        # Check Byzantine node has lower reputation
        rep_byzantine = reputation.reputations["node_5"]
        rep_honest = reputation.reputations["node_0"]

        assert rep_byzantine.reputation < rep_honest.reputation, \
            "Byzantine node should have lower reputation"

    @pytest.fixture
    def setup_zomes(self):
        """Create simulated zome instances."""
        gradient_storage = GradientStorageZome()
        cbd = CBDZome()
        reputation = ReputationTrackerV2Zome()
        defense = DefenseCoordinatorZome(cbd, reputation)
        return gradient_storage, cbd, reputation, defense


class TestPerformanceSimulation:
    """Test performance characteristics."""

    def test_defense_round_performance(self):
        """Test defense round completes in reasonable time."""
        import time

        cbd = CBDZome()
        reputation = ReputationTrackerV2Zome()
        defense = DefenseCoordinatorZome(cbd, reputation)

        np.random.seed(42)
        base_gradient = np.random.randn(100)  # 100-dim gradients

        # Simulate 20 nodes
        gradients = []
        for i in range(20):
            noise = np.random.randn(100) * 0.1
            gradient = base_gradient + noise
            gradient_fp = [fp_from_f64(g) for g in gradient]
            gradients.append((f"node_{i}", gradient_fp))

        # Time defense round
        start = time.time()
        for round_num in range(10):
            defense.run_defense(gradients, round_num=round_num)
        elapsed = time.time() - start

        # Should complete 10 rounds in < 2 seconds
        assert elapsed < 2.0, f"Defense round too slow: {elapsed}s for 10 rounds"

    def test_large_gradient_handling(self):
        """Test handling of large gradients (1000-dim)."""
        cbd = CBDZome()
        reputation = ReputationTrackerV2Zome()
        defense = DefenseCoordinatorZome(cbd, reputation)

        np.random.seed(42)
        base_gradient = np.random.randn(1000)  # 1000-dim gradients

        gradients = []
        for i in range(10):
            noise = np.random.randn(1000) * 0.1
            gradient = base_gradient + noise
            gradient_fp = [fp_from_f64(g) for g in gradient]
            gradients.append((f"node_{i}", gradient_fp))

        # Should complete without error
        result = defense.run_defense(gradients, round_num=1)
        assert len(result.aggregated_gradient) == 1000


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
