#!/usr/bin/env python3
"""
Fixed-Point vs Floating-Point Accuracy Benchmark

Validates that Q16.16 fixed-point arithmetic used in Rust zomes
maintains sufficient accuracy for Byzantine detection.

Tests:
1. Cosine similarity accuracy
2. Z-score computation accuracy
3. EMA update accuracy
4. Aggregation accuracy (Mean, Median, Krum)

Author: Luminous Dynamics Research Team
Date: December 2025
"""

import numpy as np
from typing import List, Tuple
import time


# Q16.16 Fixed-Point Simulation
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


class FixedPointBenchmark:
    """Benchmark fixed-point vs floating-point accuracy."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def test_cosine_similarity_accuracy(self, n_tests: int = 1000) -> dict:
        """Test cosine similarity computation accuracy."""
        errors = []

        for _ in range(n_tests):
            # Generate random vectors
            a = np.random.randn(100)
            b = np.random.randn(100)

            # Floating-point computation
            dot_fp = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            cos_float = dot_fp / (norm_a * norm_b + 1e-10)

            # Fixed-point computation
            a_fp = [fp_from_f64(x) for x in a]
            b_fp = [fp_from_f64(x) for x in b]

            dot_fixed = sum(fp_mul(x, y) for x, y in zip(a_fp, b_fp))
            norm_a_sq = sum(fp_mul(x, x) for x in a_fp)
            norm_b_sq = sum(fp_mul(x, x) for x in b_fp)
            norm_a_fixed = fp_sqrt(norm_a_sq)
            norm_b_fixed = fp_sqrt(norm_b_sq)

            if norm_a_fixed > 0 and norm_b_fixed > 0:
                cos_fixed = fp_div(dot_fixed, fp_mul(norm_a_fixed, norm_b_fixed))
                cos_fixed_f64 = fp_to_f64(cos_fixed)
            else:
                cos_fixed_f64 = 0.0

            error = abs(cos_float - cos_fixed_f64)
            errors.append(error)

        return {
            "mean_error": np.mean(errors),
            "max_error": np.max(errors),
            "std_error": np.std(errors),
            "within_1pct": np.mean(np.array(errors) < 0.01) * 100,
            "within_5pct": np.mean(np.array(errors) < 0.05) * 100,
        }

    def test_z_score_accuracy(self, n_tests: int = 1000) -> dict:
        """Test MAD-based z-score computation accuracy."""
        errors = []

        for _ in range(n_tests):
            # Generate random norms (simulating gradient norms)
            norms = np.abs(np.random.randn(20)) + 0.1

            # Floating-point computation
            median_float = np.median(norms)
            mad_float = np.median(np.abs(norms - median_float))
            mad_float = max(mad_float, 1e-8)
            z_float = 0.6745 * (norms[0] - median_float) / mad_float

            # Fixed-point computation
            norms_fp = [fp_from_f64(x) for x in norms]
            sorted_norms = sorted(norms_fp)
            n = len(sorted_norms)
            median_fixed = (sorted_norms[n//2 - 1] + sorted_norms[n//2]) // 2 if n % 2 == 0 else sorted_norms[n//2]

            deviations = sorted([fp_abs(x - median_fixed) for x in norms_fp])
            mad_fixed = (deviations[n//2 - 1] + deviations[n//2]) // 2 if n % 2 == 0 else deviations[n//2]
            mad_fixed = max(mad_fixed, 1)

            k = fp_from_f64(0.6745)
            z_fixed = fp_mul(k, fp_div(norms_fp[0] - median_fixed, mad_fixed))
            z_fixed_f64 = fp_to_f64(z_fixed)

            error = abs(z_float - z_fixed_f64)
            errors.append(error)

        return {
            "mean_error": np.mean(errors),
            "max_error": np.max(errors),
            "std_error": np.std(errors),
            "within_0.5": np.mean(np.array(errors) < 0.5) * 100,
            "within_1.0": np.mean(np.array(errors) < 1.0) * 100,
        }

    def test_ema_accuracy(self, n_rounds: int = 100) -> dict:
        """Test EMA (Exponential Moving Average) accuracy."""
        alpha = 0.3

        # Generate random z-scores
        z_scores = np.random.randn(n_rounds) * 2

        # Floating-point EMA
        ema_float = 0.0
        ema_history_float = []
        for z in z_scores:
            ema_float = alpha * z + (1 - alpha) * ema_float
            ema_history_float.append(ema_float)

        # Fixed-point EMA
        alpha_fp = fp_from_f64(alpha)
        one_minus_alpha = FP_SCALE - alpha_fp
        ema_fixed = 0
        ema_history_fixed = []
        for z in z_scores:
            z_fp = fp_from_f64(z)
            ema_fixed = fp_mul(alpha_fp, z_fp) + fp_mul(one_minus_alpha, ema_fixed)
            ema_history_fixed.append(fp_to_f64(ema_fixed))

        errors = np.abs(np.array(ema_history_float) - np.array(ema_history_fixed))

        return {
            "mean_error": np.mean(errors),
            "max_error": np.max(errors),
            "final_error": errors[-1],
            "accumulated_drift": errors[-1] - errors[0],
        }

    def test_krum_accuracy(self, n_clients: int = 20, dim: int = 100) -> dict:
        """Test Krum aggregation accuracy."""
        # Generate random gradients
        gradients = [np.random.randn(dim) for _ in range(n_clients)]

        # Floating-point Krum
        def krum_float(grads, k=1):
            n = len(grads)
            f = n // 4

            # Compute pairwise distances
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(grads[i] - grads[j])
                    distances[i, j] = d
                    distances[j, i] = d

            # Compute scores
            scores = []
            for i in range(n):
                sorted_dists = np.sort(distances[i])
                score = np.sum(sorted_dists[1:n-f-1])
                scores.append((i, score))

            scores.sort(key=lambda x: x[1])
            selected = [scores[i][0] for i in range(k)]
            return np.mean([grads[i] for i in selected], axis=0)

        # Fixed-point Krum (simplified)
        def krum_fixed(grads, k=1):
            grads_fp = [[fp_from_f64(x) for x in g] for g in grads]
            n = len(grads_fp)
            f = n // 4

            # Compute pairwise distances
            distances = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    dist_sq = sum(fp_mul(grads_fp[i][d] - grads_fp[j][d],
                                        grads_fp[i][d] - grads_fp[j][d])
                                 for d in range(len(grads_fp[i])))
                    d = fp_sqrt(dist_sq)
                    distances[i][j] = d
                    distances[j][i] = d

            # Compute scores
            scores = []
            for i in range(n):
                sorted_dists = sorted(distances[i])
                score = sum(sorted_dists[1:n-f-1])
                scores.append((i, score))

            scores.sort(key=lambda x: x[1])
            selected = [scores[i][0] for i in range(k)]

            # Average selected
            result = []
            for d in range(len(grads_fp[0])):
                avg = sum(grads_fp[s][d] for s in selected) // len(selected)
                result.append(fp_to_f64(avg))
            return np.array(result)

        result_float = krum_float(gradients, k=3)
        result_fixed = krum_fixed(gradients, k=3)

        # Compute error
        error = np.linalg.norm(result_float - result_fixed) / np.linalg.norm(result_float)

        return {
            "relative_error": error,
            "within_1pct": error < 0.01,
            "within_5pct": error < 0.05,
        }

    def run_all_benchmarks(self) -> dict:
        """Run all benchmarks and return results."""
        results = {}

        print("=" * 60)
        print("Fixed-Point vs Floating-Point Accuracy Benchmark")
        print("=" * 60)

        # Cosine similarity
        print("\n[1/4] Testing Cosine Similarity Accuracy...")
        start = time.time()
        results["cosine_similarity"] = self.test_cosine_similarity_accuracy()
        print(f"  Completed in {time.time() - start:.2f}s")
        print(f"  Mean error: {results['cosine_similarity']['mean_error']:.6f}")
        print(f"  Max error: {results['cosine_similarity']['max_error']:.6f}")
        print(f"  Within 1%: {results['cosine_similarity']['within_1pct']:.1f}%")

        # Z-score
        print("\n[2/4] Testing Z-Score Accuracy...")
        start = time.time()
        results["z_score"] = self.test_z_score_accuracy()
        print(f"  Completed in {time.time() - start:.2f}s")
        print(f"  Mean error: {results['z_score']['mean_error']:.6f}")
        print(f"  Max error: {results['z_score']['max_error']:.6f}")
        print(f"  Within 0.5: {results['z_score']['within_0.5']:.1f}%")

        # EMA
        print("\n[3/4] Testing EMA Accuracy...")
        start = time.time()
        results["ema"] = self.test_ema_accuracy()
        print(f"  Completed in {time.time() - start:.2f}s")
        print(f"  Mean error: {results['ema']['mean_error']:.6f}")
        print(f"  Final error: {results['ema']['final_error']:.6f}")
        print(f"  Accumulated drift: {results['ema']['accumulated_drift']:.6f}")

        # Krum
        print("\n[4/4] Testing Krum Aggregation Accuracy...")
        start = time.time()
        results["krum"] = self.test_krum_accuracy()
        print(f"  Completed in {time.time() - start:.2f}s")
        print(f"  Relative error: {results['krum']['relative_error']:.6f}")
        print(f"  Within 1%: {results['krum']['within_1pct']}")
        print(f"  Within 5%: {results['krum']['within_5pct']}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        all_pass = (
            results["cosine_similarity"]["within_5pct"] > 95 and
            results["z_score"]["within_1.0"] > 95 and
            results["ema"]["final_error"] < 0.1 and
            results["krum"]["within_5pct"]
        )

        if all_pass:
            print("✅ All accuracy tests PASSED!")
            print("   Q16.16 fixed-point is suitable for Byzantine detection.")
        else:
            print("⚠️ Some accuracy tests may need attention.")

        print(f"\nQ16.16 Range: -32768.0 to +32767.99998")
        print(f"Q16.16 Precision: ~0.000015 (1/{FP_SCALE})")

        return results


if __name__ == "__main__":
    benchmark = FixedPointBenchmark(seed=42)
    results = benchmark.run_all_benchmarks()
