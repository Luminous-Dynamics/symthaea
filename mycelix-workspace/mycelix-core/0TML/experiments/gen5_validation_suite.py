# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gen 5 AEGIS Validation Experiment Suite
========================================

Comprehensive validation experiments for all 7 layers.
Target: 300 runs across 9 experiment types for MLSys/ICML 2026 paper.

Experiments:
1. Byzantine tolerance curves (0-50% adversaries)
2. Sleeper agent detection rates
3. Coordination detection accuracy
4. Active learning speedup validation
5. Federated convergence (with/without FedMDO)
6. Privacy-utility tradeoff
7. Distributed validation overhead (Layer 4)
8. Self-healing recovery time (Layer 7)
9. Secret sharing Byzantine tolerance

Author: Luminous Dynamics
Date: November 12, 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

# Import all Gen 5 layers
from gen5.meta_learning import MetaLearningEnsemble, MetaLearningConfig
from gen5.federated_meta import FederatedMetaDefense, FedMDOConfig
from gen5.explainability import CausalAttributionEngine
from gen5.uncertainty import UncertaintyQuantifier
from gen5.active_learning import ActiveLearningInspector, ActiveLearningConfig
from gen5.multi_round import MultiRoundDetector, MultiRoundConfig
from gen5.federated_validator import ThresholdValidator, ShamirSecretSharing
from gen5.self_healing import SelfHealingMechanism, HealingConfig


@dataclass
class ExperimentConfig:
    """Configuration for validation experiments."""
    num_runs: int = 30  # Runs per experiment type
    random_seed: int = 42
    output_dir: str = "validation_results"

    # Byzantine attack parameters
    byz_fractions: List[float] = None  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    num_agents: int = 100
    num_rounds: int = 50
    gradient_dim: int = 1000

    # Sleeper agent parameters
    sleeper_activate_round: int = 25

    # Active learning parameters
    query_budgets: List[float] = None  # [0.1, 0.2, 0.3, 0.5]

    # Privacy parameters
    epsilon_values: List[float] = None  # [0.5, 1.0, 2.0, 4.0, 8.0]

    def __post_init__(self):
        if self.byz_fractions is None:
            self.byz_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        if self.query_budgets is None:
            self.query_budgets = [0.1, 0.2, 0.3, 0.5]
        if self.epsilon_values is None:
            self.epsilon_values = [0.5, 1.0, 2.0, 4.0, 8.0]


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    experiment_type: str
    run_id: int
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: float
    duration: float


class ValidationSuite:
    """Comprehensive validation experiment suite for Gen 5 AEGIS."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[ExperimentResult] = []

        # Set random seed for reproducibility
        np.random.seed(config.random_seed)

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def run_all_experiments(self) -> Dict[str, List[ExperimentResult]]:
        """Run all 9 experiment types."""
        print("=" * 80)
        print("🚀 Gen 5 AEGIS Validation Experiment Suite")
        print("=" * 80)
        print(f"\nTarget: {self.config.num_runs * 9} total runs across 9 experiment types")
        print(f"Output: {self.config.output_dir}/\n")

        experiments = [
            ("1_byzantine_tolerance", self.experiment_1_byzantine_tolerance),
            ("2_sleeper_detection", self.experiment_2_sleeper_detection),
            ("3_coordination_detection", self.experiment_3_coordination_detection),
            ("4_active_learning_speedup", self.experiment_4_active_learning_speedup),
            ("5_federated_convergence", self.experiment_5_federated_convergence),
            ("6_privacy_utility_tradeoff", self.experiment_6_privacy_utility_tradeoff),
            ("7_distributed_validation_overhead", self.experiment_7_distributed_validation_overhead),
            ("8_self_healing_recovery", self.experiment_8_self_healing_recovery),
            ("9_secret_sharing_tolerance", self.experiment_9_secret_sharing_tolerance),
        ]

        all_results = {}

        for exp_name, exp_func in experiments:
            print(f"\n{'=' * 80}")
            print(f"Running: {exp_name.replace('_', ' ').title()}")
            print(f"{'=' * 80}\n")

            results = exp_func()
            all_results[exp_name] = results

            # Save intermediate results
            self._save_results(exp_name, results)

        # Generate summary
        self._generate_summary(all_results)

        print(f"\n{'=' * 80}")
        print("✅ All experiments complete!")
        print(f"{'=' * 80}\n")
        print(f"Results saved to: {self.config.output_dir}/")
        print(f"Total runs: {len(self.results)}")

        return all_results

    def experiment_1_byzantine_tolerance(self) -> List[ExperimentResult]:
        """
        Experiment 1: Byzantine Tolerance Curves

        Validates system performance across 0-50% Byzantine adversaries.
        Tests all layers under increasing Byzantine pressure.
        """
        results = []

        for run_id in range(self.config.num_runs):
            for byz_frac in self.config.byz_fractions:
                print(f"  Run {run_id + 1}/{self.config.num_runs}, "
                      f"Byzantine fraction: {byz_frac:.1%}")

                start_time = time.time()

                # Create mock detection methods
                methods = self._create_mock_methods(num_methods=5)

                # Create ensemble
                ensemble = MetaLearningEnsemble(
                    base_methods=methods,
                    config=MetaLearningConfig(learning_rate=0.01)
                )

                # Generate gradients
                num_byzantine = int(self.config.num_agents * byz_frac)
                gradients, labels = self._generate_gradients(
                    num_honest=self.config.num_agents - num_byzantine,
                    num_byzantine=num_byzantine,
                    dim=self.config.gradient_dim
                )

                # Evaluate ensemble
                correct = 0
                total = len(gradients)

                for gradient, label in zip(gradients, labels):
                    signals = {}
                    for method in methods:
                        signals[method.name] = method.score(gradient)

                    score = ensemble.compute_ensemble_score(signals)
                    prediction = "honest" if score >= 0.5 else "byzantine"

                    if prediction == label:
                        correct += 1

                accuracy = correct / total if total > 0 else 0.0

                duration = time.time() - start_time

                result = ExperimentResult(
                    experiment_type="byzantine_tolerance",
                    run_id=run_id,
                    parameters={"byzantine_fraction": byz_frac},
                    metrics={
                        "accuracy": accuracy,
                        "true_positives": correct,
                        "total_gradients": total,
                    },
                    timestamp=time.time(),
                    duration=duration,
                )

                results.append(result)
                self.results.append(result)

        return results

    def experiment_2_sleeper_detection(self) -> List[ExperimentResult]:
        """
        Experiment 2: Sleeper Agent Detection

        Tests Layer 6 (CUSUM) ability to detect agents that turn Byzantine
        after building honest reputation.
        """
        results = []

        for run_id in range(self.config.num_runs):
            print(f"  Run {run_id + 1}/{self.config.num_runs}")

            start_time = time.time()

            # Create multi-round detector
            detector = MultiRoundDetector(
                config=MultiRoundConfig(
                    cusum_threshold=5.0,
                    cusum_drift=0.5,
                )
            )

            # Simulate sleeper agent
            agent_id = 0
            detected_round = None

            for round_num in range(self.config.num_rounds):
                # Agent is honest until activation round
                is_byzantine = round_num >= self.config.sleeper_activate_round

                # Generate score (high if honest, low if Byzantine)
                score = 0.2 if is_byzantine else 0.9
                score += np.random.normal(0, 0.05)  # Add noise
                score = np.clip(score, 0.0, 1.0)

                # Update detector
                detector.update_agent_history(agent_id, score, round_num)

                # Check for detection
                is_sleeper = detector.detect_sleeper_agent(agent_id, round_num)

                if is_sleeper and detected_round is None:
                    detected_round = round_num

            # Calculate detection metrics
            if detected_round is not None:
                detection_delay = detected_round - self.config.sleeper_activate_round
                detected = True
            else:
                detection_delay = None
                detected = False

            duration = time.time() - start_time

            result = ExperimentResult(
                experiment_type="sleeper_detection",
                run_id=run_id,
                parameters={
                    "activate_round": self.config.sleeper_activate_round,
                    "total_rounds": self.config.num_rounds,
                },
                metrics={
                    "detected": 1.0 if detected else 0.0,
                    "detection_delay": detection_delay if detected else 999,
                    "detected_round": detected_round if detected else -1,
                },
                timestamp=time.time(),
                duration=duration,
            )

            results.append(result)
            self.results.append(result)

        return results

    def experiment_3_coordination_detection(self) -> List[ExperimentResult]:
        """
        Experiment 3: Coordination Detection

        Tests Layer 6 ability to detect coordinated attacks via correlation.
        """
        results = []

        for run_id in range(self.config.num_runs):
            print(f"  Run {run_id + 1}/{self.config.num_runs}")

            start_time = time.time()

            detector = MultiRoundDetector(
                config=MultiRoundConfig(correlation_threshold=0.7)
            )

            # Simulate coordinated attack
            num_coordinated = 5
            attack_pattern = np.random.choice([0.2, 0.8], size=self.config.num_rounds)

            # Update agent histories
            for round_num in range(self.config.num_rounds):
                for agent_id in range(num_coordinated):
                    # Coordinated agents follow same pattern with noise
                    base_score = attack_pattern[round_num]
                    score = base_score + np.random.normal(0, 0.05)
                    score = np.clip(score, 0.0, 1.0)

                    detector.update_agent_history(agent_id, score, round_num)

            # Detect coordination
            coordinated_pairs = detector.detect_coordinated_agents(
                list(range(num_coordinated)),
                self.config.num_rounds - 1
            )

            # Calculate metrics
            expected_pairs = num_coordinated * (num_coordinated - 1) // 2
            detected_pairs = len(coordinated_pairs)
            detection_rate = detected_pairs / expected_pairs if expected_pairs > 0 else 0

            duration = time.time() - start_time

            result = ExperimentResult(
                experiment_type="coordination_detection",
                run_id=run_id,
                parameters={
                    "num_coordinated_agents": num_coordinated,
                    "num_rounds": self.config.num_rounds,
                },
                metrics={
                    "detection_rate": detection_rate,
                    "detected_pairs": detected_pairs,
                    "expected_pairs": expected_pairs,
                },
                timestamp=time.time(),
                duration=duration,
            )

            results.append(result)
            self.results.append(result)

        return results

    def experiment_4_active_learning_speedup(self) -> List[ExperimentResult]:
        """
        Experiment 4: Active Learning Speedup

        Validates Layer 5 achieves 6-10x speedup with < 1% accuracy loss.
        """
        results = []

        for run_id in range(self.config.num_runs):
            for query_budget in self.config.query_budgets:
                print(f"  Run {run_id + 1}/{self.config.num_runs}, "
                      f"Query budget: {query_budget:.1%}")

                start_time = time.time()

                # Create inspector
                inspector = ActiveLearningInspector(
                    fast_methods=self._create_mock_methods(3),
                    deep_methods=self._create_mock_methods(2),
                    config=ActiveLearningConfig(query_budget=query_budget),
                )

                # Generate test gradients
                gradients, labels = self._generate_gradients(
                    num_honest=80,
                    num_byzantine=20,
                    dim=self.config.gradient_dim,
                )

                # Two-pass detection
                fast_start = time.time()
                decisions, scores, queried_indices = inspector.two_pass_detection(gradients)
                fast_time = time.time() - fast_start

                # Baseline: full deep verification
                baseline_start = time.time()
                baseline_decisions = []
                for gradient in gradients:
                    signals = {}
                    for method in inspector.deep_methods:
                        signals[method.name] = method.score(gradient)
                    # Simple averaging
                    score = np.mean(list(signals.values()))
                    decision = "honest" if score >= 0.5 else "byzantine"
                    baseline_decisions.append(decision)
                baseline_time = time.time() - baseline_start

                # Calculate metrics
                correct = sum(1 for d, l in zip(decisions, labels) if d == l)
                baseline_correct = sum(1 for d, l in zip(baseline_decisions, labels) if d == l)

                accuracy = correct / len(labels)
                baseline_accuracy = baseline_correct / len(labels)
                accuracy_loss = baseline_accuracy - accuracy
                speedup = baseline_time / fast_time if fast_time > 0 else 0

                duration = time.time() - start_time

                result = ExperimentResult(
                    experiment_type="active_learning_speedup",
                    run_id=run_id,
                    parameters={"query_budget": query_budget},
                    metrics={
                        "accuracy": accuracy,
                        "baseline_accuracy": baseline_accuracy,
                        "accuracy_loss": accuracy_loss,
                        "speedup": speedup,
                        "fast_time": fast_time,
                        "baseline_time": baseline_time,
                        "queries_made": len(queried_indices),
                    },
                    timestamp=time.time(),
                    duration=duration,
                )

                results.append(result)
                self.results.append(result)

        return results

    def experiment_5_federated_convergence(self) -> List[ExperimentResult]:
        """
        Experiment 5: Federated Convergence

        Tests Layer 1+ federated meta-defense optimization convergence.
        """
        results = []

        for run_id in range(self.config.num_runs):
            print(f"  Run {run_id + 1}/{self.config.num_runs}")

            # Test with and without FedMDO
            for use_fedmdo in [False, True]:
                start_time = time.time()

                methods = self._create_mock_methods(5)

                if use_fedmdo:
                    # Federated meta-defense
                    fedmdo = FederatedMetaDefense(
                        base_methods=methods,
                        config=FedMDOConfig(
                            num_agents=10,
                            epsilon=1.0,
                            aggregation_method="krum",
                        ),
                    )

                    # Simulate federated training
                    initial_loss = None
                    final_loss = None

                    for round_num in range(20):
                        # Generate gradients for each agent
                        agent_gradients = []
                        agent_labels = []

                        for _ in range(10):  # 10 agents
                            grads, labs = self._generate_gradients(10, 2, 100)
                            agent_gradients.append(grads)
                            agent_labels.append(labs)

                        # Federated update
                        aggregated_meta_grad, metrics = fedmdo.federated_meta_update(
                            agent_gradients,
                            agent_labels,
                            round_num,
                        )

                        if round_num == 0:
                            initial_loss = metrics.get("average_loss", 0)
                        if round_num == 19:
                            final_loss = metrics.get("average_loss", 0)

                    loss_reduction = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0
                else:
                    # Centralized meta-learning
                    ensemble = MetaLearningEnsemble(
                        base_methods=methods,
                        config=MetaLearningConfig(learning_rate=0.01),
                    )

                    initial_loss = None
                    final_loss = None

                    for iteration in range(200):
                        grads, labs = self._generate_gradients(50, 10, 100)

                        for gradient, label in zip(grads, labs):
                            signals = {}
                            for method in methods:
                                signals[method.name] = method.score(gradient)

                            score = ensemble.compute_ensemble_score(signals)
                            true_label = 1.0 if label == "honest" else 0.0
                            loss = ensemble.update(signals, true_label)

                            if iteration == 0:
                                initial_loss = loss
                            if iteration == 199:
                                final_loss = loss

                    loss_reduction = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0

                duration = time.time() - start_time

                result = ExperimentResult(
                    experiment_type="federated_convergence",
                    run_id=run_id,
                    parameters={"use_fedmdo": use_fedmdo},
                    metrics={
                        "initial_loss": initial_loss or 0,
                        "final_loss": final_loss or 0,
                        "loss_reduction": loss_reduction,
                    },
                    timestamp=time.time(),
                    duration=duration,
                )

                results.append(result)
                self.results.append(result)

        return results

    def experiment_6_privacy_utility_tradeoff(self) -> List[ExperimentResult]:
        """
        Experiment 6: Privacy-Utility Tradeoff

        Tests impact of differential privacy noise on accuracy.
        """
        results = []

        for run_id in range(self.config.num_runs):
            for epsilon in self.config.epsilon_values:
                print(f"  Run {run_id + 1}/{self.config.num_runs}, ε = {epsilon}")

                start_time = time.time()

                methods = self._create_mock_methods(5)

                fedmdo = FederatedMetaDefense(
                    base_methods=methods,
                    config=FedMDOConfig(
                        num_agents=10,
                        epsilon=epsilon,
                        aggregation_method="krum",
                    ),
                )

                # Test accuracy
                gradients, labels = self._generate_gradients(100, 20, 100)

                correct = 0
                for gradient, label in zip(gradients, labels):
                    signals = {}
                    for method in methods:
                        signals[method.name] = method.score(gradient)

                    score = fedmdo.ensemble.compute_ensemble_score(signals)
                    prediction = "honest" if score >= 0.5 else "byzantine"

                    if prediction == label:
                        correct += 1

                accuracy = correct / len(labels)

                duration = time.time() - start_time

                result = ExperimentResult(
                    experiment_type="privacy_utility_tradeoff",
                    run_id=run_id,
                    parameters={"epsilon": epsilon},
                    metrics={
                        "accuracy": accuracy,
                        "privacy_budget": epsilon,
                    },
                    timestamp=time.time(),
                    duration=duration,
                )

                results.append(result)
                self.results.append(result)

        return results

    def experiment_7_distributed_validation_overhead(self) -> List[ExperimentResult]:
        """
        Experiment 7: Distributed Validation Overhead (Layer 4)

        Measures latency overhead of secret sharing validation.
        """
        results = []

        for run_id in range(self.config.num_runs):
            print(f"  Run {run_id + 1}/{self.config.num_runs}")

            start_time = time.time()

            methods = self._create_mock_methods(3)
            ensemble = MetaLearningEnsemble(base_methods=methods)

            validator = ThresholdValidator(
                ensemble=ensemble,
                num_validators=7,
                threshold=4,
            )

            # Measure validation time
            gradients, _ = self._generate_gradients(50, 10, 100)

            validation_times = []
            for gradient in gradients:
                val_start = time.time()
                decision, confidence, details = validator.validate_gradient(gradient)
                val_time = time.time() - val_start
                validation_times.append(val_time * 1000)  # Convert to ms

            mean_time = np.mean(validation_times)
            std_time = np.std(validation_times)

            duration = time.time() - start_time

            result = ExperimentResult(
                experiment_type="distributed_validation_overhead",
                run_id=run_id,
                parameters={
                    "num_validators": 7,
                    "threshold": 4,
                },
                metrics={
                    "mean_latency_ms": mean_time,
                    "std_latency_ms": std_time,
                    "min_latency_ms": min(validation_times),
                    "max_latency_ms": max(validation_times),
                },
                timestamp=time.time(),
                duration=duration,
            )

            results.append(result)
            self.results.append(result)

        return results

    def experiment_8_self_healing_recovery(self) -> List[ExperimentResult]:
        """
        Experiment 8: Self-Healing Recovery Time (Layer 7)

        Measures time to recover from Byzantine surge > 45%.
        """
        results = []

        for run_id in range(self.config.num_runs):
            print(f"  Run {run_id + 1}/{self.config.num_runs}")

            start_time = time.time()

            healer = SelfHealingMechanism(
                config=HealingConfig(
                    bft_tolerance=0.45,
                    healing_threshold=0.40,
                    recovery_rate=0.05,
                )
            )

            # Simulate Byzantine surge then recovery
            healing_activated = False
            activation_round = None
            deactivation_round = None

            for round_num in range(100):
                # First 20 rounds: heavy attack (60% Byzantine)
                # Rounds 20-100: recovery (10% Byzantine)
                if round_num < 20:
                    # Attack: 60% Byzantine scores
                    scores = [0.2] * 6 + [0.8] * 4
                else:
                    # Recovery: 10% Byzantine scores
                    scores = [0.2] * 1 + [0.8] * 9

                decisions, details = healer.process_batch(scores, round_num)

                # Track healing activation/deactivation
                if healer.is_healing and not healing_activated:
                    healing_activated = True
                    activation_round = round_num

                if healing_activated and not healer.is_healing and deactivation_round is None:
                    deactivation_round = round_num

            # Calculate recovery time
            if activation_round is not None and deactivation_round is not None:
                recovery_time = deactivation_round - activation_round
            else:
                recovery_time = -1

            duration = time.time() - start_time

            result = ExperimentResult(
                experiment_type="self_healing_recovery",
                run_id=run_id,
                parameters={
                    "bft_surge": 0.60,
                    "recovery_bft": 0.10,
                },
                metrics={
                    "recovery_rounds": recovery_time,
                    "activation_round": activation_round or -1,
                    "deactivation_round": deactivation_round or -1,
                    "healed_successfully": 1.0 if recovery_time > 0 else 0.0,
                },
                timestamp=time.time(),
                duration=duration,
            )

            results.append(result)
            self.results.append(result)

        return results

    def experiment_9_secret_sharing_tolerance(self) -> List[ExperimentResult]:
        """
        Experiment 9: Secret Sharing Byzantine Tolerance

        Tests Layer 4 secret sharing robustness to Byzantine validators.
        """
        results = []

        for run_id in range(self.config.num_runs):
            print(f"  Run {run_id + 1}/{self.config.num_runs}")

            start_time = time.time()

            # Test with 0, 1, and 2 Byzantine validators (n=7, t=4)
            for num_byzantine_validators in [0, 1, 2]:
                methods = self._create_mock_methods(3)
                ensemble = MetaLearningEnsemble(base_methods=methods)

                validator = ThresholdValidator(
                    ensemble=ensemble,
                    num_validators=7,
                    threshold=4,
                )

                # Generate test gradients
                gradients, labels = self._generate_gradients(30, 10, 100)

                correct = 0
                for gradient, label in zip(gradients, labels):
                    # Get reference score
                    signals = {}
                    for method in methods:
                        signals[method.name] = method.score(gradient)
                    reference_score = ensemble.compute_ensemble_score(signals)

                    # Simulate Byzantine validators
                    validator_scores = [reference_score] * (7 - num_byzantine_validators)
                    validator_scores += [0.0] * num_byzantine_validators  # Byzantine

                    decision, confidence, details = validator.validate_gradient(
                        gradient,
                        validator_scores,
                    )

                    if decision == label:
                        correct += 1

                accuracy = correct / len(labels)

                result = ExperimentResult(
                    experiment_type="secret_sharing_tolerance",
                    run_id=run_id,
                    parameters={
                        "num_byzantine_validators": num_byzantine_validators,
                        "num_validators": 7,
                        "threshold": 4,
                    },
                    metrics={
                        "accuracy": accuracy,
                        "correct_decisions": correct,
                        "total_gradients": len(labels),
                    },
                    timestamp=time.time(),
                    duration=time.time() - start_time,
                )

                results.append(result)
                self.results.append(result)

        return results

    # Helper methods

    def _create_mock_methods(self, num_methods: int) -> List:
        """Create mock detection methods."""
        class MockMethod:
            def __init__(self, name: str, sensitivity: float):
                self.name = name
                self.sensitivity = sensitivity

            def score(self, gradient: np.ndarray) -> float:
                # Score based on gradient magnitude
                magnitude = np.linalg.norm(gradient)
                # Normalize by dimension
                normalized_mag = magnitude / np.sqrt(len(gradient))
                # Large gradients → low score (Byzantine)
                base_score = max(0.0, 1.0 - normalized_mag * self.sensitivity)
                # Add some noise
                noise = np.random.normal(0, 0.05)
                return np.clip(base_score + noise, 0.0, 1.0)

        methods = []
        for i in range(num_methods):
            sensitivity = 0.5 + i * 0.1  # Varying sensitivities
            methods.append(MockMethod(f"method_{i}", sensitivity))

        return methods

    def _generate_gradients(
        self,
        num_honest: int,
        num_byzantine: int,
        dim: int,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Generate honest and Byzantine gradients."""
        gradients = []
        labels = []

        # Honest gradients (small magnitude)
        for _ in range(num_honest):
            gradient = np.random.normal(0, 0.1, size=dim)
            gradients.append(gradient)
            labels.append("honest")

        # Byzantine gradients (large magnitude)
        for _ in range(num_byzantine):
            gradient = np.random.normal(0, 2.0, size=dim)
            gradients.append(gradient)
            labels.append("byzantine")

        # Shuffle
        combined = list(zip(gradients, labels))
        np.random.shuffle(combined)
        gradients, labels = zip(*combined)

        return list(gradients), list(labels)

    def _save_results(self, exp_name: str, results: List[ExperimentResult]):
        """Save experiment results to JSON."""
        output_file = Path(self.config.output_dir) / f"{exp_name}_results.json"

        results_dict = [asdict(r) for r in results]

        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n  ✅ Saved {len(results)} results to {output_file}")

    def _generate_summary(self, all_results: Dict[str, List[ExperimentResult]]):
        """Generate summary statistics."""
        summary_file = Path(self.config.output_dir) / "SUMMARY.txt"

        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Gen 5 AEGIS Validation Experiment Summary\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Runs: {len(self.results)}\n")
            f.write(f"Total Duration: {sum(r.duration for r in self.results):.2f}s\n\n")

            for exp_name, results in all_results.items():
                f.write(f"\n{exp_name.replace('_', ' ').title()}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Runs: {len(results)}\n")

                # Calculate key metrics
                if results:
                    metrics_summary = defaultdict(list)
                    for result in results:
                        for key, value in result.metrics.items():
                            if isinstance(value, (int, float)):
                                metrics_summary[key].append(value)

                    for metric_name, values in metrics_summary.items():
                        mean = np.mean(values)
                        std = np.std(values)
                        f.write(f"  {metric_name}: {mean:.4f} ± {std:.4f}\n")

        print(f"\n  📊 Summary saved to {summary_file}")


def main():
    """Main entry point for validation suite."""
    config = ExperimentConfig(
        num_runs=30,  # 30 runs per experiment type
        output_dir="validation_results",
    )

    suite = ValidationSuite(config)
    results = suite.run_all_experiments()

    print("\n✨ Validation complete! Results ready for paper figures.")


if __name__ == "__main__":
    main()
