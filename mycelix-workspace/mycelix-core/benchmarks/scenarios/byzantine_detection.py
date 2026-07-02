# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Byzantine Detection Benchmark Scenario

Tests Byzantine detection rates at various adversarial ratios:
- 10%, 20%, 30%, 40%, 45% Byzantine nodes
- Multiple attack types (gradient scaling, sign flip, noise injection)
- Measures precision, recall, F1 score, and detection accuracy

Expected results based on Mycelix capabilities:
- 30% Byzantine: 99.5%+ detection rate
- 45% Byzantine: 99%+ detection rate

Author: Luminous Dynamics
Date: January 2026
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "0TML" / "src"))

logger = logging.getLogger(__name__)


@dataclass
class ByzantineDetectionConfig:
    """Configuration for Byzantine detection benchmark."""
    byzantine_ratios: List[float] = field(
        default_factory=lambda: [0.10, 0.20, 0.30, 0.40, 0.45]
    )
    num_nodes: int = 20
    gradient_size: int = 10000
    num_rounds: int = 20
    warmup_rounds: int = 5
    attack_types: List[str] = field(
        default_factory=lambda: ["gradient_scaling", "sign_flip", "gaussian_noise"]
    )
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "byzantine_ratios": self.byzantine_ratios,
            "num_nodes": self.num_nodes,
            "gradient_size": self.gradient_size,
            "num_rounds": self.num_rounds,
            "warmup_rounds": self.warmup_rounds,
            "attack_types": self.attack_types,
            "seed": self.seed,
        }


@dataclass
class DetectionRateResult:
    """Result for a single Byzantine ratio test."""
    byzantine_ratio: float
    num_nodes: int
    num_byzantine: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    detection_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    attack_types_used: List[str] = field(default_factory=list)
    layer_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "byzantine_ratio": self.byzantine_ratio,
            "num_nodes": self.num_nodes,
            "num_byzantine": self.num_byzantine,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "detection_rate": self.detection_rate,
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "attack_types_used": self.attack_types_used,
            "layer_results": self.layer_results,
        }


class ByzantineDetectionScenario:
    """
    Byzantine detection benchmark scenario.

    Tests detection rates at various adversarial ratios using the
    multi-layer Byzantine detection stack.

    Example:
        >>> scenario = ByzantineDetectionScenario()
        >>> results = scenario.run()
        >>> for r in results:
        ...     print(f"{r.byzantine_ratio:.0%}: P={r.precision:.3f}, R={r.recall:.3f}")
    """

    def __init__(self, config: Optional[ByzantineDetectionConfig] = None):
        """Initialize scenario with configuration."""
        self.config = config or ByzantineDetectionConfig()
        self.rng = np.random.RandomState(self.config.seed)

        # Try importing mycelix_fl
        try:
            from mycelix_fl import (
                MycelixFL, FLConfig,
                MultiLayerByzantineDetector,
                AttackOrchestrator, create_attack, AttackType,
            )
            self.MycelixFL = MycelixFL
            self.FLConfig = FLConfig
            self.MultiLayerByzantineDetector = MultiLayerByzantineDetector
            self.AttackOrchestrator = AttackOrchestrator
            self.create_attack = create_attack
            self.AttackType = AttackType
            self.available = True
        except ImportError as e:
            logger.warning(f"mycelix_fl not available: {e}")
            self.available = False

    def generate_gradients(self, num_nodes: int) -> Dict[str, np.ndarray]:
        """Generate honest gradients."""
        base_gradient = self.rng.randn(self.config.gradient_size).astype(np.float32)
        gradients = {}

        for i in range(num_nodes):
            noise = self.rng.randn(self.config.gradient_size).astype(np.float32) * 0.1
            gradients[f"node_{i}"] = base_gradient + noise

        return gradients

    def inject_byzantine(
        self,
        gradients: Dict[str, np.ndarray],
        byzantine_ratio: float,
    ) -> tuple[Dict[str, np.ndarray], Set[str], List[str]]:
        """Inject Byzantine behavior and return modified gradients."""
        node_ids = list(gradients.keys())
        num_byzantine = int(len(node_ids) * byzantine_ratio)

        if num_byzantine == 0:
            return gradients, set(), []

        byzantine_ids = set(self.rng.choice(node_ids, num_byzantine, replace=False))

        if not self.available:
            # Fallback
            modified = dict(gradients)
            attacks_used = []
            for node_id in byzantine_ids:
                grad = gradients[node_id]
                attack = self.rng.choice(["scale", "flip", "noise"])
                if attack == "scale":
                    modified[node_id] = grad * self.rng.uniform(5, 100)
                    attacks_used.append("gradient_scaling")
                elif attack == "flip":
                    modified[node_id] = -grad
                    attacks_used.append("sign_flip")
                else:
                    modified[node_id] = grad + self.rng.randn(*grad.shape) * 10
                    attacks_used.append("gaussian_noise")
            return modified, byzantine_ids, attacks_used

        # Use mycelix_fl attacks
        orchestrator = self.AttackOrchestrator(random_seed=self.config.seed)
        attacks_used = []

        for node_id in byzantine_ids:
            attack_type_str = self.rng.choice(self.config.attack_types)
            attacks_used.append(attack_type_str)
            try:
                attack = self.create_attack(self.AttackType(attack_type_str))
                orchestrator.register_attacker(node_id, attack)
            except Exception:
                # Fallback for unknown attack types
                pass

        modified, _ = orchestrator.apply_attacks(gradients)
        return modified, byzantine_ids, list(set(attacks_used))

    def run_single_ratio(self, byzantine_ratio: float) -> DetectionRateResult:
        """Run benchmark for a single Byzantine ratio."""
        logger.info(f"Testing Byzantine ratio: {byzantine_ratio:.0%}")

        # Reset RNG for reproducibility
        self.rng = np.random.RandomState(self.config.seed + int(byzantine_ratio * 100))

        # Generate gradients
        gradients = self.generate_gradients(self.config.num_nodes)

        # Inject Byzantine
        modified, byzantine_ids, attacks_used = self.inject_byzantine(
            gradients, byzantine_ratio
        )

        num_byzantine = len(byzantine_ids)
        num_honest = self.config.num_nodes - num_byzantine

        if not self.available:
            # Fallback result
            return DetectionRateResult(
                byzantine_ratio=byzantine_ratio,
                num_nodes=self.config.num_nodes,
                num_byzantine=num_byzantine,
                true_positives=0,
                false_positives=0,
                false_negatives=num_byzantine,
                true_negatives=num_honest,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                accuracy=num_honest / self.config.num_nodes,
                detection_rate=0.0,
                mean_latency_ms=0.0,
                p95_latency_ms=0.0,
                attack_types_used=attacks_used,
            )

        # Set up FL system
        fl_config = self.FLConfig(
            byzantine_threshold=0.49,
            use_detection=True,
            use_healing=True,
        )
        fl = self.MycelixFL(config=fl_config)

        # Warmup
        for i in range(self.config.warmup_rounds):
            fl.execute_round(modified, round_num=i + 1)

        # Benchmark
        latencies = []
        all_detected: Set[str] = set()
        layer_results = {}

        import time

        for i in range(self.config.num_rounds):
            start = time.perf_counter()
            result = fl.execute_round(modified, round_num=self.config.warmup_rounds + i + 1)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

            all_detected.update(result.byzantine_nodes)

            # Collect layer-level results from last round
            if i == self.config.num_rounds - 1 and hasattr(result, 'detection_result'):
                for lr in result.detection_result.layer_results:
                    layer_results[lr.layer.name] = {
                        "byzantine_count": len(lr.byzantine_nodes),
                        "healed_count": len(lr.healed_nodes),
                        "confidence": lr.confidence,
                        "latency_ms": lr.latency_ms,
                    }

        # Calculate metrics
        detected = all_detected
        tp = len(detected & byzantine_ids)
        fp = len(detected - byzantine_ids)
        fn = len(byzantine_ids - detected)
        tn = self.config.num_nodes - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / self.config.num_nodes

        # Detection rate = recall (what fraction of Byzantine were caught)
        detection_rate = recall

        return DetectionRateResult(
            byzantine_ratio=byzantine_ratio,
            num_nodes=self.config.num_nodes,
            num_byzantine=num_byzantine,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            detection_rate=detection_rate,
            mean_latency_ms=float(np.mean(latencies)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            attack_types_used=attacks_used,
            layer_results=layer_results,
        )

    def run(self) -> List[DetectionRateResult]:
        """Run benchmark for all configured Byzantine ratios."""
        results = []

        for ratio in self.config.byzantine_ratios:
            result = self.run_single_ratio(ratio)
            results.append(result)

            logger.info(
                f"  {ratio:.0%} Byzantine: "
                f"P={result.precision:.3f} "
                f"R={result.recall:.3f} "
                f"F1={result.f1_score:.3f} "
                f"({result.mean_latency_ms:.1f}ms)"
            )

        return results

    def to_benchmark_output(
        self,
        results: List[DetectionRateResult],
    ) -> Dict[str, Any]:
        """Convert results to standard benchmark output format."""
        import socket
        import platform

        return {
            "benchmark": "byzantine_detection",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "environment": {
                "hostname": socket.gethostname(),
                "platform": platform.system(),
                "python_version": platform.python_version(),
            },
            "parameters": self.config.to_dict(),
            "results": {
                "by_ratio": {
                    f"{int(r.byzantine_ratio * 100)}%": r.to_dict()
                    for r in results
                },
                "summary": {
                    "mean_precision": float(np.mean([r.precision for r in results])),
                    "mean_recall": float(np.mean([r.recall for r in results])),
                    "mean_f1": float(np.mean([r.f1_score for r in results])),
                    "max_detection_rate": max(r.detection_rate for r in results),
                    "min_detection_rate": min(r.detection_rate for r in results),
                },
            },
            "metadata": {
                "mycelix_available": self.available,
                "num_ratios_tested": len(results),
            },
        }


def run_byzantine_benchmark(
    output_dir: Optional[Path] = None,
    config: Optional[ByzantineDetectionConfig] = None,
) -> Dict[str, Any]:
    """
    Run complete Byzantine detection benchmark.

    Args:
        output_dir: Directory for output files
        config: Custom configuration

    Returns:
        Benchmark results dictionary
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Byzantine Detection Benchmark")
    logger.info("=" * 60)

    scenario = ByzantineDetectionScenario(config=config)
    results = scenario.run()
    output = scenario.to_benchmark_output(results)

    if output_dir:
        output_path = Path(output_dir) / "byzantine_detection.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved results to {output_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Byzantine Detection Benchmark")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--num-nodes", type=int, default=20, help="Number of nodes")
    parser.add_argument("--num-rounds", type=int, default=20, help="Rounds per ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = ByzantineDetectionConfig(
        num_nodes=args.num_nodes,
        num_rounds=args.num_rounds,
        seed=args.seed,
    )

    output = run_byzantine_benchmark(
        output_dir=args.output_dir,
        config=config,
    )

    print("\n" + "=" * 60)
    print("Summary:")
    for ratio, data in output["results"]["by_ratio"].items():
        print(f"  {ratio}: P={data['precision']:.3f} R={data['recall']:.3f} F1={data['f1_score']:.3f}")
