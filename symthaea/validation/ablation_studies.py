#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Ablation Studies for Hyperdimensional Active Inference

Studies:
1. HDC Dimension: 1024, 4096, 16384, 65536
2. Precision Initialization: 0.1, 0.5, 1.0, 2.0, 5.0
3. EFE Weight Ratios: pragmatic vs epistemic vs novelty
"""

import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result from a single ablation run."""
    condition: str
    value: float
    inference_time_ms: float
    action_time_ms: float
    free_energy: float
    success_rate: float
    convergence_iterations: int


class HDCActiveInference:
    """Simplified HAI implementation for ablation studies."""

    def __init__(self, dim: int = 16384, pi_s: float = 1.0, pi_p: float = 1.0,
                 w_pragmatic: float = 1.0, w_epistemic: float = 0.5, w_novelty: float = 0.1):
        self.dim = dim
        self.pi_s = pi_s
        self.pi_p = pi_p
        self.w_pragmatic = w_pragmatic
        self.w_epistemic = w_epistemic
        self.w_novelty = w_novelty

        # Initialize hypervectors
        self.belief = self._random_hv()
        self.prior = self._random_hv()

    def _random_hv(self) -> np.ndarray:
        """Generate random hypervector."""
        return np.random.randn(self.dim).astype(np.float32)

    def _normalize(self, hv: np.ndarray) -> np.ndarray:
        """Normalize hypervector to unit length."""
        norm = np.linalg.norm(hv)
        return hv / norm if norm > 0 else hv

    def _cosine_similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(hv1, hv2) / (np.linalg.norm(hv1) * np.linalg.norm(hv2) + 1e-10))

    def compute_free_energy(self, observation: np.ndarray) -> float:
        """Compute variational free energy."""
        # Accuracy: similarity to observation
        accuracy = -0.5 * self.pi_s * (1 - self._cosine_similarity(self.belief, observation))**2
        # Complexity: divergence from prior
        complexity = 0.5 * self.pi_p * (1 - self._cosine_similarity(self.belief, self.prior))
        return complexity - accuracy

    def update_belief(self, observation: np.ndarray, learning_rate: float = 0.1) -> float:
        """Update belief via gradient descent on free energy."""
        cos_obs = self._cosine_similarity(self.belief, observation)
        cos_prior = self._cosine_similarity(self.belief, self.prior)

        # Gradient
        grad = (self.pi_s * (observation - self.belief * cos_obs) +
                self.pi_p * (self.prior - self.belief * cos_prior))

        # Update
        self.belief = self._normalize(self.belief + learning_rate * grad)

        return self.compute_free_energy(observation)

    def compute_efe(self, action_idx: int, goal: np.ndarray) -> float:
        """Compute expected free energy for action."""
        # Simulate next state (simplified)
        predicted_next = self._normalize(self.belief + 0.1 * np.random.randn(self.dim))

        # Pragmatic value: distance to goal
        pragmatic = self.w_pragmatic * (1 - self._cosine_similarity(predicted_next, goal))

        # Epistemic value: expected information gain (approximated)
        epistemic = self.w_epistemic * np.abs(self._cosine_similarity(predicted_next, self.belief) - 0.5)

        # Novelty bonus
        novelty = self.w_novelty * np.random.random()

        return pragmatic + epistemic - novelty

    def select_action(self, goal: np.ndarray, num_actions: int = 8, temperature: float = 1.0) -> Tuple[int, float]:
        """Select action minimizing EFE."""
        efes = [self.compute_efe(a, goal) for a in range(num_actions)]

        # Softmax
        efes = np.array(efes)
        probs = np.exp(-efes / temperature)
        probs /= probs.sum()

        action = int(np.argmax(probs))
        return action, float(efes[action])


def run_dimension_ablation(trials: int = 50) -> List[AblationResult]:
    """Ablation study on HDC dimension."""
    logger.info("Running HDC dimension ablation...")
    results = []

    dimensions = [1024, 4096, 16384, 65536]

    for dim in dimensions:
        logger.info(f"  Testing dimension: {dim}")

        inference_times = []
        action_times = []
        free_energies = []
        successes = 0
        convergence_iters = []

        for trial in range(trials):
            agent = HDCActiveInference(dim=dim)
            goal = agent._random_hv()
            observation = agent._random_hv()

            # Inference timing
            start = time.perf_counter()
            for i in range(20):
                fe = agent.update_belief(observation)
                if fe < 0.5:
                    convergence_iters.append(i)
                    break
            else:
                convergence_iters.append(20)
            inference_times.append((time.perf_counter() - start) * 1000)

            # Action selection timing
            start = time.perf_counter()
            action, efe = agent.select_action(goal)
            action_times.append((time.perf_counter() - start) * 1000)

            free_energies.append(fe)

            # Success: free energy below threshold
            if fe < 0.5:
                successes += 1

        results.append(AblationResult(
            condition="dimension",
            value=float(dim),
            inference_time_ms=float(np.mean(inference_times)),
            action_time_ms=float(np.mean(action_times)),
            free_energy=float(np.mean(free_energies)),
            success_rate=successes / trials,
            convergence_iterations=int(np.mean(convergence_iters))
        ))

    return results


def run_precision_ablation(trials: int = 50) -> List[AblationResult]:
    """Ablation study on precision initialization."""
    logger.info("Running precision initialization ablation...")
    results = []

    precisions = [0.1, 0.5, 1.0, 2.0, 5.0]

    for pi in precisions:
        logger.info(f"  Testing precision: {pi}")

        inference_times = []
        action_times = []
        free_energies = []
        successes = 0
        convergence_iters = []

        for trial in range(trials):
            agent = HDCActiveInference(pi_s=pi, pi_p=pi)
            goal = agent._random_hv()
            observation = agent._random_hv()

            # Inference timing
            start = time.perf_counter()
            for i in range(20):
                fe = agent.update_belief(observation)
                if fe < 0.5:
                    convergence_iters.append(i)
                    break
            else:
                convergence_iters.append(20)
            inference_times.append((time.perf_counter() - start) * 1000)

            # Action selection timing
            start = time.perf_counter()
            action, efe = agent.select_action(goal)
            action_times.append((time.perf_counter() - start) * 1000)

            free_energies.append(fe)

            if fe < 0.5:
                successes += 1

        results.append(AblationResult(
            condition="precision",
            value=float(pi),
            inference_time_ms=float(np.mean(inference_times)),
            action_time_ms=float(np.mean(action_times)),
            free_energy=float(np.mean(free_energies)),
            success_rate=successes / trials,
            convergence_iterations=int(np.mean(convergence_iters))
        ))

    return results


def run_efe_weights_ablation(trials: int = 50) -> List[AblationResult]:
    """Ablation study on EFE weight ratios."""
    logger.info("Running EFE weights ablation...")
    results = []

    # (w_pragmatic, w_epistemic, w_novelty)
    weight_configs = [
        (1.0, 0.0, 0.0),   # Pure exploitation
        (0.0, 1.0, 0.0),   # Pure exploration
        (1.0, 0.5, 0.0),   # Balanced (no novelty)
        (1.0, 0.5, 0.1),   # Balanced (default)
        (1.0, 1.0, 0.5),   # High exploration + novelty
    ]

    weight_names = [
        "pure_exploit",
        "pure_explore",
        "balanced_no_novelty",
        "balanced_default",
        "high_explore_novelty"
    ]

    for (w_p, w_e, w_n), name in zip(weight_configs, weight_names):
        logger.info(f"  Testing weights: {name} (p={w_p}, e={w_e}, n={w_n})")

        inference_times = []
        action_times = []
        free_energies = []
        successes = 0
        convergence_iters = []

        for trial in range(trials):
            agent = HDCActiveInference(w_pragmatic=w_p, w_epistemic=w_e, w_novelty=w_n)
            goal = agent._random_hv()
            observation = agent._random_hv()

            # Inference timing
            start = time.perf_counter()
            for i in range(20):
                fe = agent.update_belief(observation)
                if fe < 0.5:
                    convergence_iters.append(i)
                    break
            else:
                convergence_iters.append(20)
            inference_times.append((time.perf_counter() - start) * 1000)

            # Action selection timing
            start = time.perf_counter()
            action, efe = agent.select_action(goal)
            action_times.append((time.perf_counter() - start) * 1000)

            free_energies.append(fe)

            if fe < 0.5:
                successes += 1

        # Encode weights as single value for result (ratio)
        weight_ratio = w_e / (w_p + 0.01)  # Exploration/exploitation ratio

        results.append(AblationResult(
            condition=f"efe_weights_{name}",
            value=weight_ratio,
            inference_time_ms=float(np.mean(inference_times)),
            action_time_ms=float(np.mean(action_times)),
            free_energy=float(np.mean(free_energies)),
            success_rate=successes / trials,
            convergence_iterations=int(np.mean(convergence_iters))
        ))

    return results


def generate_ablation_report(results: Dict[str, List[AblationResult]], output_path: Path):
    """Generate markdown report from ablation results."""

    with open(output_path, 'w') as f:
        f.write("# HAI Ablation Studies Results\n\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Dimension ablation
        f.write("## 1. HDC Dimension Ablation\n\n")
        f.write("| Dimension | Inference (ms) | Action (ms) | Free Energy | Success Rate | Convergence |\n")
        f.write("|-----------|----------------|-------------|-------------|--------------|-------------|\n")
        for r in results['dimension']:
            f.write(f"| {int(r.value):,} | {r.inference_time_ms:.3f} | {r.action_time_ms:.3f} | "
                   f"{r.free_energy:.4f} | {r.success_rate*100:.1f}% | {r.convergence_iterations} |\n")
        f.write("\n")

        # Precision ablation
        f.write("## 2. Precision Initialization Ablation\n\n")
        f.write("| Precision (π) | Inference (ms) | Action (ms) | Free Energy | Success Rate | Convergence |\n")
        f.write("|---------------|----------------|-------------|-------------|--------------|-------------|\n")
        for r in results['precision']:
            f.write(f"| {r.value:.1f} | {r.inference_time_ms:.3f} | {r.action_time_ms:.3f} | "
                   f"{r.free_energy:.4f} | {r.success_rate*100:.1f}% | {r.convergence_iterations} |\n")
        f.write("\n")

        # EFE weights ablation
        f.write("## 3. EFE Weight Ratios Ablation\n\n")
        f.write("| Configuration | Explore/Exploit | Inference (ms) | Action (ms) | Free Energy | Success Rate |\n")
        f.write("|---------------|-----------------|----------------|-------------|-------------|---------------|\n")
        for r in results['efe_weights']:
            config_name = r.condition.replace('efe_weights_', '')
            f.write(f"| {config_name} | {r.value:.2f} | {r.inference_time_ms:.3f} | {r.action_time_ms:.3f} | "
                   f"{r.free_energy:.4f} | {r.success_rate*100:.1f}% |\n")
        f.write("\n")

        # Key findings
        f.write("## Key Findings\n\n")

        # Best dimension
        dim_results = results['dimension']
        best_dim = max(dim_results, key=lambda x: x.success_rate)
        f.write(f"1. **Optimal Dimension**: {int(best_dim.value):,} (success rate: {best_dim.success_rate*100:.1f}%)\n")

        # Best precision
        prec_results = results['precision']
        best_prec = max(prec_results, key=lambda x: x.success_rate)
        f.write(f"2. **Optimal Precision**: π = {best_prec.value:.1f} (success rate: {best_prec.success_rate*100:.1f}%)\n")

        # Best EFE config
        efe_results = results['efe_weights']
        best_efe = max(efe_results, key=lambda x: x.success_rate)
        f.write(f"3. **Optimal EFE Weights**: {best_efe.condition.replace('efe_weights_', '')} "
               f"(success rate: {best_efe.success_rate*100:.1f}%)\n")

        f.write("\n---\n")
        f.write(f"*50 trials per configuration, 20 inference iterations per trial*\n")


def main():
    """Run all ablation studies."""
    print("=" * 60)
    print("HAI Ablation Studies")
    print("=" * 60)

    np.random.seed(42)  # Reproducibility

    results = {
        'dimension': run_dimension_ablation(trials=50),
        'precision': run_precision_ablation(trials=50),
        'efe_weights': run_efe_weights_ablation(trials=50),
    }

    # Save raw results as JSON
    output_dir = Path(__file__).parent.parent / "docs"

    json_results = {k: [asdict(r) for r in v] for k, v in results.items()}
    with open(output_dir / "ABLATION_RESULTS.json", 'w') as f:
        json.dump(json_results, f, indent=2)

    # Generate markdown report
    generate_ablation_report(results, output_dir / "ABLATION_STUDIES_REPORT.md")

    print("\n" + "=" * 60)
    print("Results saved to:")
    print(f"  - {output_dir / 'ABLATION_STUDIES_REPORT.md'}")
    print(f"  - {output_dir / 'ABLATION_RESULTS.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
