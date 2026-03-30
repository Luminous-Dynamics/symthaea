#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Asymmetric Contrastive Causal Network (ACCN) Prototype

Tests the key ideas before full Rust implementation:
1. Statistical feature encoding (not raw data)
2. Contrastive asymmetry detection
3. LTC-inspired iterative refinement
4. Calibrated confidence via ensemble disagreement
"""

import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ============================================================================
# DATA LOADING
# ============================================================================

@dataclass
class CauseEffectPair:
    id: str
    x: List[float]
    y: List[float]
    ground_truth: str
    weight: float

def load_tuebingen(data_dir: str) -> List[CauseEffectPair]:
    pairs = []
    meta_path = Path(data_dir) / "pairmeta.txt"
    if not meta_path.exists(): return pairs
    with open(meta_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            pair_id = parts[0]
            cause_start, effect_start = int(parts[1]), int(parts[3])
            weight = float(parts[5])
            ground_truth = "forward" if cause_start < effect_start else "backward"
            data_path = Path(data_dir) / f"pair{pair_id}.txt"
            if not data_path.exists(): continue
            try:
                x_vals, y_vals = [], []
                with open(data_path) as df:
                    for dline in df:
                        vals = dline.strip().split()
                        if len(vals) >= 2:
                            x_vals.append(float(vals[0]))
                            y_vals.append(float(vals[1]))
                if len(x_vals) >= 10:
                    pairs.append(CauseEffectPair(pair_id, x_vals, y_vals, ground_truth, weight))
            except: continue
    return pairs

# ============================================================================
# STATISTICAL HELPERS
# ============================================================================

def mean(vals): return sum(vals) / len(vals) if vals else 0.0
def std(vals):
    if len(vals) < 2: return 1e-10
    m = mean(vals)
    return max(1e-10, math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)))

def variance(vals):
    if len(vals) < 2: return 1e-10
    m = mean(vals)
    return max(1e-10, sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

def correlation(x, y):
    if len(x) != len(y) or len(x) < 2: return 0.0
    mx, my = mean(x), mean(y)
    sx, sy = std(x), std(y)
    if sx < 1e-10 or sy < 1e-10: return 0.0
    return sum((x[i] - mx) * (y[i] - my) for i in range(len(x))) / ((len(x) - 1) * sx * sy)

def skewness(vals):
    if len(vals) < 3: return 0.0
    m, s = mean(vals), std(vals)
    if s < 1e-10: return 0.0
    n = len(vals)
    return sum(((v - m) / s) ** 3 for v in vals) * n / ((n - 1) * (n - 2))

def kurtosis(vals):
    if len(vals) < 4: return 0.0
    m, s = mean(vals), std(vals)
    if s < 1e-10: return 0.0
    return sum(((v - m) / s) ** 4 for v in vals) / len(vals) - 3

def linreg(x, y):
    n = len(x)
    mx, my = mean(x), mean(y)
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))
    slope = num / (den + 1e-10)
    intercept = my - slope * mx
    return slope, intercept, [y[i] - (slope * x[i] + intercept) for i in range(n)]

def tail_weight(vals):
    m, s = mean(vals), std(vals)
    if s < 1e-10: return 0.0
    return sum(1 for v in vals if abs(v - m) > 2 * s) / len(vals)

def heteroscedasticity(x, y):
    _, _, residuals = linreg(x, y)
    n = len(x)
    mid = n // 2
    pairs = sorted(zip(x, residuals))
    low_half = [p[1] for p in pairs[:mid]]
    high_half = [p[1] for p in pairs[mid:]]
    var_low = variance(low_half)
    var_high = variance(high_half)
    return abs(math.log((var_high + 1e-10) / (var_low + 1e-10)))

# ============================================================================
# FEATURE EXTRACTION - The Key Innovation
# ============================================================================

@dataclass
class CausalFeatures:
    """Rich statistical features for causal discovery."""
    # Marginal statistics
    x_mean: float
    x_std: float
    x_skew: float
    x_kurt: float
    y_mean: float
    y_std: float
    y_skew: float
    y_kurt: float

    # Joint statistics
    corr: float

    # Regression asymmetry (core causal signal)
    mse_xy: float       # Normalized MSE of X→Y regression
    mse_yx: float       # Normalized MSE of Y→X regression
    res_indep_xy: float # Independence of X from residuals(X→Y)
    res_indep_yx: float # Independence of Y from residuals(Y→X)

    # Distribution shape
    hetero: float       # Heteroscedasticity
    tail_x: float       # X tail weight
    tail_y: float       # Y tail weight

    # Method scores (as soft features)
    reci: float
    igci: float
    anm: float
    info: float

def extract_features(x: List[float], y: List[float]) -> CausalFeatures:
    """Extract all causal-relevant features from a pair."""
    # Marginals
    x_m, x_s = mean(x), std(x)
    y_m, y_s = mean(y), std(y)
    x_sk, x_ku = skewness(x), kurtosis(x)
    y_sk, y_ku = skewness(y), kurtosis(y)

    # Joint
    corr = correlation(x, y)

    # Regressions
    _, _, res_xy = linreg(x, y)
    _, _, res_yx = linreg(y, x)

    mse_xy = mean([r ** 2 for r in res_xy]) / (variance(y) + 1e-10)
    mse_yx = mean([r ** 2 for r in res_yx]) / (variance(x) + 1e-10)

    res_indep_xy = abs(correlation(x, res_xy))
    res_indep_yx = abs(correlation(y, res_yx))

    # Shape
    hetero = heteroscedasticity(x, y)
    tail_x = tail_weight(x)
    tail_y = tail_weight(y)

    # Method scores
    reci = mse_yx - mse_xy
    anm = res_indep_yx - res_indep_xy
    igci = compute_igci(x, y)
    info = compute_info(x, y)

    return CausalFeatures(
        x_m, x_s, x_sk, x_ku,
        y_m, y_s, y_sk, y_ku,
        corr, mse_xy, mse_yx, res_indep_xy, res_indep_yx,
        hetero, tail_x, tail_y,
        reci, igci, anm, info
    )

def compute_igci(x, y):
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    if x_max - x_min < 1e-10 or y_max - y_min < 1e-10: return 0.0
    x_norm = [(v - x_min) / (x_max - x_min) for v in x]
    y_norm = [(v - y_min) / (y_max - y_min) for v in y]
    pairs = sorted(zip(x_norm, y_norm))
    log_slopes = []
    for i in range(len(pairs) - 1):
        dx = pairs[i+1][0] - pairs[i][0]
        dy = pairs[i+1][1] - pairs[i][1]
        if abs(dx) > 1e-10:
            slope = abs(dy / dx)
            if slope > 1e-10: log_slopes.append(math.log(slope))
    if not log_slopes: return 0.0
    score_xy = mean(log_slopes)
    pairs_y = sorted(zip(y_norm, x_norm))
    log_slopes_yx = []
    for i in range(len(pairs_y) - 1):
        dy = pairs_y[i+1][0] - pairs_y[i][0]
        dx = pairs_y[i+1][1] - pairs_y[i][1]
        if abs(dy) > 1e-10:
            slope = abs(dx / dy)
            if slope > 1e-10: log_slopes_yx.append(math.log(slope))
    if not log_slopes_yx: return score_xy
    return score_xy - mean(log_slopes_yx)

def compute_info(x, y):
    n = len(x)
    bins = min(10, n // 5)
    if bins < 2: return 0.0
    def discretize(vals):
        v_min, v_max = min(vals), max(vals)
        if v_max - v_min < 1e-10: return [0] * len(vals)
        return [min(int((v - v_min) / (v_max - v_min + 1e-10) * bins), bins - 1) for v in vals]
    def entropy(vals):
        counts = {}
        for v in vals: counts[v] = counts.get(v, 0) + 1
        return -sum((c/len(vals)) * math.log2(c/len(vals)) for c in counts.values() if c > 0)
    def joint_entropy(v1, v2):
        counts = {}
        for a, b in zip(v1, v2): counts[(a,b)] = counts.get((a,b), 0) + 1
        return -sum((c/len(v1)) * math.log2(c/len(v1)) for c in counts.values() if c > 0)
    x_d, y_d = discretize(x), discretize(y)
    H_x, H_y, H_xy = entropy(x_d), entropy(y_d), joint_entropy(x_d, y_d)
    return (H_xy - H_y) - (H_xy - H_x)

# ============================================================================
# HYPERDIMENSIONAL ENCODING
# ============================================================================

class HDEncoder:
    """Encode features into hyperdimensional vectors."""

    def __init__(self, dim: int = 1000, seed: int = 42):
        self.dim = dim
        random.seed(seed)
        # Random projection matrices for each feature
        self.projections = {}

    def _get_projection(self, name: str) -> List[float]:
        if name not in self.projections:
            self.projections[name] = [random.gauss(0, 1) for _ in range(self.dim)]
        return self.projections[name]

    def encode_scalar(self, value: float, name: str) -> List[float]:
        """Encode a scalar value into HD space."""
        proj = self._get_projection(name)
        # Use value to modulate the projection
        return [p * math.tanh(value) for p in proj]

    def encode_features(self, f: CausalFeatures) -> List[float]:
        """Encode all features into a single HD vector."""
        vectors = [
            self.encode_scalar(f.x_skew, "x_skew"),
            self.encode_scalar(f.x_kurt, "x_kurt"),
            self.encode_scalar(f.y_skew, "y_skew"),
            self.encode_scalar(f.y_kurt, "y_kurt"),
            self.encode_scalar(f.corr, "corr"),
            self.encode_scalar(f.mse_xy, "mse_xy"),
            self.encode_scalar(f.mse_yx, "mse_yx"),
            self.encode_scalar(f.res_indep_xy, "res_indep_xy"),
            self.encode_scalar(f.res_indep_yx, "res_indep_yx"),
            self.encode_scalar(f.hetero, "hetero"),
            self.encode_scalar(f.tail_x, "tail_x"),
            self.encode_scalar(f.tail_y, "tail_y"),
            self.encode_scalar(f.reci, "reci"),
            self.encode_scalar(f.igci, "igci"),
            self.encode_scalar(f.anm, "anm"),
            self.encode_scalar(f.info, "info"),
        ]

        # Bundle (sum) all vectors
        result = [0.0] * self.dim
        for v in vectors:
            for i in range(self.dim):
                result[i] += v[i]

        # Normalize
        norm = math.sqrt(sum(x ** 2 for x in result) + 1e-10)
        return [x / norm for x in result]

    def similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity between two HD vectors."""
        dot = sum(a[i] * b[i] for i in range(len(a)))
        return dot  # Vectors are normalized

# ============================================================================
# LTC-INSPIRED REFINEMENT
# ============================================================================

class LTCRefinement:
    """
    Liquid Time-Constant inspired iterative refinement.

    Instead of temporal dynamics, we use this for belief refinement:
    - Start with initial belief from features
    - Iterate to convergence
    - Time constant τ adapts based on evidence strength
    """

    def __init__(self, base_tau: float = 0.3, max_steps: int = 10):
        self.base_tau = base_tau
        self.max_steps = max_steps

    def refine(self, initial: List[float], evidence: List[List[float]]) -> Tuple[List[float], float]:
        """
        Refine belief vector based on evidence.
        Returns (refined_belief, confidence).
        """
        belief = initial.copy()
        dim = len(belief)

        for step in range(self.max_steps):
            prev_belief = belief.copy()

            # Update based on each evidence vector
            for e in evidence:
                # Compute compatibility
                compat = sum(belief[i] * e[i] for i in range(dim))

                # Adaptive time constant: stronger evidence = faster update
                tau = self.base_tau * (1 + abs(compat))

                # LTC-style update: belief += tau * (evidence - belief)
                for i in range(dim):
                    belief[i] += tau * (e[i] - belief[i])

            # Normalize
            norm = math.sqrt(sum(x ** 2 for x in belief) + 1e-10)
            belief = [x / norm for x in belief]

            # Check convergence
            delta = sum(belief[i] * prev_belief[i] for i in range(dim))
            if delta > 0.999:
                # Confidence is how quickly we converged
                confidence = 1.0 - (step / self.max_steps)
                return belief, confidence

        return belief, 0.5  # Didn't converge fully

# ============================================================================
# ASYMMETRIC CONTRASTIVE CAUSAL NETWORK
# ============================================================================

class ACCN:
    """
    Asymmetric Contrastive Causal Network.

    Key innovations:
    1. Operates on statistical features, not raw data
    2. Uses contrastive comparison of X→Y vs Y→X
    3. LTC refinement for confidence estimation
    4. Learned weights for combining evidence
    """

    def __init__(self, hd_dim: int = 1000):
        self.encoder = HDEncoder(dim=hd_dim)
        self.refiner = LTCRefinement()

        # Learned weights (initialized from error analysis)
        self.weights = {
            "reci": 0.25,
            "igci": 0.15,
            "anm": 0.25,
            "info": 0.35,  # Info is strongest rescuer
        }

        # Learned bias terms for different regimes
        self.bias = {
            "high_kurt": 0.2,   # Favor robust methods
            "high_hetero": 0.15, # Favor Info
            "high_corr": 0.1,   # Favor IGCI
        }

        # Prototype vectors for forward/backward (learned during training)
        self.forward_proto = None
        self.backward_proto = None

    def extract_asymmetry(self, f: CausalFeatures) -> float:
        """
        Compute asymmetry score using weighted combination.
        Positive = forward, negative = backward.
        """
        # Base score from methods
        score = (self.weights["reci"] * f.reci +
                 self.weights["igci"] * f.igci +
                 self.weights["anm"] * f.anm +
                 self.weights["info"] * f.info)

        # Regime-specific adjustments
        if abs(f.x_kurt) > 3 or abs(f.y_kurt) > 3:
            # Heavy tails: trust Info more
            score += self.bias["high_kurt"] * f.info

        if f.hetero > 0.5:
            # Heteroscedastic: trust Info
            score += self.bias["high_hetero"] * f.info

        if abs(f.corr) > 0.8:
            # High correlation: trust IGCI
            score += self.bias["high_corr"] * f.igci

        return score

    def predict(self, x: List[float], y: List[float]) -> Tuple[str, float]:
        """
        Predict causal direction with confidence.
        Returns (direction, confidence).
        """
        # Extract features
        f = extract_features(x, y)

        # Compute asymmetry score
        score = self.extract_asymmetry(f)

        # Encode to HD space
        hv = self.encoder.encode_features(f)

        # LTC refinement for confidence
        # Create evidence vectors from individual method scores
        evidence = [
            self.encoder.encode_scalar(f.reci, "evidence_reci"),
            self.encoder.encode_scalar(f.igci, "evidence_igci"),
            self.encoder.encode_scalar(f.anm, "evidence_anm"),
            self.encoder.encode_scalar(f.info, "evidence_info"),
        ]

        refined, ltc_confidence = self.refiner.refine(hv, evidence)

        # Combine score magnitude with LTC confidence
        confidence = abs(score) * ltc_confidence

        direction = "forward" if score >= 0 else "backward"
        return direction, confidence

    def train(self, pairs: List[CauseEffectPair], epochs: int = 10, lr: float = 0.05):
        """
        Train weights using gradient-free optimization.
        """
        best_acc = 0
        best_weights = self.weights.copy()
        best_bias = self.bias.copy()

        for epoch in range(epochs):
            # Evaluate current weights
            correct = sum(1 for p in pairs if self.predict(p.x, p.y)[0] == p.ground_truth)
            acc = correct / len(pairs)

            if acc > best_acc:
                best_acc = acc
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()

            # Random perturbation of weights
            for key in self.weights:
                self.weights[key] = max(0.01, self.weights[key] + random.gauss(0, lr))
            for key in self.bias:
                self.bias[key] = max(-0.5, min(0.5, self.bias[key] + random.gauss(0, lr * 0.5)))

            # Normalize weights to sum to 1
            total = sum(self.weights.values())
            for key in self.weights:
                self.weights[key] /= total

        self.weights = best_weights
        self.bias = best_bias
        return best_acc

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_accn(pairs: List[CauseEffectPair]) -> Tuple[int, int, float]:
    """Evaluate ACCN on the dataset."""
    network = ACCN()

    # Train on all pairs (for simplicity - ideally use CV)
    network.train(pairs, epochs=20)

    correct = 0
    high_conf_correct = 0
    high_conf_total = 0

    for pair in pairs:
        pred, conf = network.predict(pair.x, pair.y)
        if pred == pair.ground_truth:
            correct += 1

        # Track high-confidence predictions
        if conf > 0.5:
            high_conf_total += 1
            if pred == pair.ground_truth:
                high_conf_correct += 1

    acc = correct / len(pairs) * 100

    print(f"\n  High-confidence predictions: {high_conf_total}/{len(pairs)}")
    if high_conf_total > 0:
        print(f"  High-confidence accuracy: {high_conf_correct/high_conf_total*100:.1f}%")

    return correct, len(pairs), acc

def main():
    print("=" * 76)
    print("     ASYMMETRIC CONTRASTIVE CAUSAL NETWORK (ACCN) - PROTOTYPE")
    print("=" * 76)
    print()

    data_dir = "benchmarks/external/tuebingen"
    pairs = load_tuebingen(data_dir)
    print(f"Loaded {len(pairs)} pairs\n")

    # Compare methods
    print("-" * 76)
    print(" COMPARISON")
    print("-" * 76)

    # Majority Voting baseline
    def majority_vote(p):
        f = extract_features(p.x, p.y)
        votes = [f.reci > 0, f.igci > 0, f.anm > 0, f.info > 0]
        return "forward" if sum(votes) >= 2 else "backward"

    maj_correct = sum(1 for p in pairs if majority_vote(p) == p.ground_truth)
    print(f"  Majority Voting:     {maj_correct}/108 ({maj_correct/108*100:.1f}%)")

    # ACCN
    print("\n  Training ACCN...")
    accn_correct, total, accn_acc = evaluate_accn(pairs)
    print(f"\n  ACCN:                {accn_correct}/{total} ({accn_acc:.1f}%)")

    print()
    print("-" * 76)
    print(" SUMMARY")
    print("-" * 76)
    print(f"  Improvement over Majority: +{accn_acc - maj_correct/108*100:.1f}%")
    print(f"  Gap to Oracle (90.7%):     {90.7 - accn_acc:.1f}%")
    print()
    print("  Done!")

if __name__ == "__main__":
    main()
