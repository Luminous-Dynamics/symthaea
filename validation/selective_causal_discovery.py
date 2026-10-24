#!/usr/bin/env python3
"""Exact selective causal-discovery evaluator under model misspecification.

Standard-library only. Compares always-commit, posterior-confidence gating, and
predictive-surprise abstention on in-class and out-of-class three-node SCMs.
This is a transparent falsification harness, not an external leaderboard adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SCHEMA = "symthaea-selective-causal-discovery-v1"
ACQUISITION_POLICIES = ("fixed_order", "greedy_eig", "assumed_model_oracle")
DECISION_RULES = (
    "always_commit",
    "posterior_gate_0.8",
    "surprise_gate_2_bits",
    "surprise_gate_3_bits",
    "surprise_gate_4_bits",
    "surprise_3_bits_and_posterior_0.8",
    "calibrated_surprise_alpha_0.05",
)


@dataclass(frozen=True)
class SCM:
    name: str
    parents: tuple[tuple[int, ...], ...]
    truth_tables: tuple[int, ...]

    def graph_key(self) -> tuple[tuple[int, ...], ...]:
        return self.parents


@dataclass(frozen=True)
class Action:
    identifier: int
    target: int
    value: int


@dataclass(frozen=True)
class Terminal:
    probability: float
    posterior: tuple[float, ...]
    probes_used: int
    max_surprisal_bits: float
    cumulative_log_loss_bits: float
    stopped_for_surprise: bool


def entropy(weights: Iterable[float]) -> float:
    values = [float(w) for w in weights if w > 0.0]
    total = sum(values)
    if total <= 0.0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in values)


def mechanism_output(model: SCM, node: int, assignment: tuple[int, ...]) -> int:
    parent_assignment = 0
    for bit, parent in enumerate(model.parents[node]):
        parent_assignment |= (assignment[parent] & 1) << bit
    return (model.truth_tables[node] >> parent_assignment) & 1


def likelihood(model: SCM, action: Action, outcome: tuple[int, ...], noise: float) -> float:
    if outcome[action.target] != action.value:
        return 0.0
    result = 1.0
    for node in range(len(outcome)):
        if node == action.target:
            continue
        expected = mechanism_output(model, node, outcome)
        result *= (1.0 - noise) if outcome[node] == expected else noise
    return result


def outcome_distribution(model: SCM, action: Action, noise: float) -> dict[tuple[int, ...], float]:
    dist: dict[tuple[int, ...], float] = {}
    for outcome in itertools.product((0, 1), repeat=len(model.parents)):
        p = likelihood(model, action, outcome, noise)
        if p > 0.0:
            dist[outcome] = p
    total = sum(dist.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"true outcome distribution does not normalize: {total}")
    return dist


def predictive_outcomes(models: tuple[SCM, ...], posterior: tuple[float, ...],
                        action: Action, assumed_noise: float) -> dict[tuple[int, ...], float]:
    dist: dict[tuple[int, ...], float] = {}
    for model, weight in zip(models, posterior):
        for outcome, p in outcome_distribution(model, action, assumed_noise).items():
            dist[outcome] = dist.get(outcome, 0.0) + weight * p
    total = sum(dist.values())
    if total <= 0.0:
        raise ValueError("learner predictive distribution has zero mass")
    dist = {outcome: p / total for outcome, p in dist.items()}
    if not math.isclose(sum(dist.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("predictive distribution does not normalize")
    return dist


def update_posterior(models: tuple[SCM, ...], posterior: tuple[float, ...], action: Action,
                     outcome: tuple[int, ...], assumed_noise: float) -> tuple[float, ...]:
    unnormalized = tuple(weight * likelihood(model, action, outcome, assumed_noise)
                         for model, weight in zip(models, posterior))
    total = sum(unnormalized)
    if total <= 0.0:
        raise ValueError("observed outcome outside learner support; fail closed")
    updated = tuple(value / total for value in unnormalized)
    if not math.isclose(sum(updated), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("posterior does not normalize")
    return updated


def graph_posterior(models: tuple[SCM, ...], posterior: tuple[float, ...]) -> dict[tuple[tuple[int, ...], ...], float]:
    result: dict[tuple[tuple[int, ...], ...], float] = {}
    for model, weight in zip(models, posterior):
        graph = model.graph_key()
        result[graph] = result.get(graph, 0.0) + weight
    return result


def expected_optimal_entropy(models: tuple[SCM, ...], posterior: tuple[float, ...],
                             actions: tuple[Action, ...], depth: int, noise: float) -> float:
    if depth <= 0 or not actions:
        return entropy(graph_posterior(models, posterior).values())
    alternatives = []
    for action in actions:
        predictive = predictive_outcomes(models, posterior, action, noise)
        remaining = tuple(a for a in actions if a.identifier != action.identifier)
        expected = 0.0
        for outcome, probability in predictive.items():
            updated = update_posterior(models, posterior, action, outcome, noise)
            expected += probability * expected_optimal_entropy(models, updated, remaining, depth - 1, noise)
        alternatives.append(expected)
    return min(alternatives)


def choose_action(policy: str, models: tuple[SCM, ...], posterior: tuple[float, ...],
                  actions: tuple[Action, ...], depth_left: int, assumed_noise: float) -> Action:
    if not actions:
        raise ValueError("no intervention actions remain")
    if policy == "fixed_order":
        return min(actions, key=lambda a: a.identifier)
    if policy == "greedy_eig":
        def value(action: Action) -> tuple[float, int]:
            predictive = predictive_outcomes(models, posterior, action, assumed_noise)
            remaining = tuple(a for a in actions if a.identifier != action.identifier)
            expected_h = sum(probability * entropy(graph_posterior(
                models, update_posterior(models, posterior, action, outcome, assumed_noise)).values())
                for outcome, probability in predictive.items())
            return expected_h, action.identifier
        return min(actions, key=value)
    if policy == "assumed_model_oracle":
        def value(action: Action) -> tuple[float, int]:
            predictive = predictive_outcomes(models, posterior, action, assumed_noise)
            remaining = tuple(a for a in actions if a.identifier != action.identifier)
            expected_h = sum(probability * expected_optimal_entropy(
                models, update_posterior(models, posterior, action, outcome, assumed_noise),
                remaining, depth_left - 1, assumed_noise)
                for outcome, probability in predictive.items())
            return expected_h, action.identifier
        return min(actions, key=value)
    raise ValueError(f"unknown acquisition policy: {policy}")


def fixtures() -> tuple[tuple[SCM, ...], dict[str, SCM], tuple[Action, ...]]:
    # Boolean truth-table bit k is output for parent assignment k.
    chain = SCM("candidate_chain", ((), (0,), (1,)), (0, 0b10, 0b10))
    fork = SCM("candidate_fork", ((), (0,), (0,)), (0, 0b10, 0b10))
    true_ood = SCM("true_xor_collider", ((), (), (0, 1)), (0, 0, 0b0110))
    actions = tuple(Action(2 * node + value, node, value)
                    for node in range(3) for value in (0, 1))
    return (chain, fork), {"in_class": chain, "out_of_class": true_ood}, actions


def decision_parameters(rule: str) -> tuple[float | None, float | None]:
    """Return (surprise threshold in bits, minimum MAP confidence)."""
    if rule == "always_commit":
        return None, None
    if rule == "posterior_gate_0.8":
        return None, 0.8
    if rule == "surprise_gate_2_bits":
        return 2.0, None
    if rule == "surprise_gate_3_bits":
        return 3.0, None
    if rule == "surprise_gate_4_bits":
        return 4.0, None
    if rule == "surprise_3_bits_and_posterior_0.8":
        return 3.0, 0.8
    if rule == "calibrated_surprise_alpha_0.05":
        return None, None
    raise ValueError(f"unknown decision rule: {rule}")


def enumerate_terminals(models: tuple[SCM, ...], true_world: SCM, actions: tuple[Action, ...],
                        acquisition_policy: str, assumed_noise: float, actual_noise: float,
                        decision_rule: str, budget: int = 2,
                        surprise_threshold_override: float | None = None) -> list[Terminal]:
    initial = tuple(1.0 / len(models) for _ in models)
    surprise_threshold, _confidence_threshold = decision_parameters(decision_rule)
    if surprise_threshold_override is not None:
        surprise_threshold = surprise_threshold_override
    terminals: list[Terminal] = []

    def recurse(posterior: tuple[float, ...], remaining: tuple[Action, ...], depth_left: int,
                path_probability: float, probes_used: int, max_surprise: float,
                cumulative_loss: float) -> None:
        if depth_left <= 0 or not remaining:
            terminals.append(Terminal(path_probability, posterior, probes_used, max_surprise,
                                      cumulative_loss, False))
            return
        action = choose_action(acquisition_policy, models, posterior, remaining,
                               depth_left, assumed_noise)
        learner_predictive = predictive_outcomes(models, posterior, action, assumed_noise)
        actual = outcome_distribution(true_world, action, actual_noise)
        next_actions = tuple(a for a in remaining if a.identifier != action.identifier)
        for outcome, true_probability in actual.items():
            predicted_probability = learner_predictive.get(outcome, 0.0)
            if predicted_probability <= 0.0:
                surprisal = math.inf
            else:
                surprisal = -math.log2(predicted_probability)
            updated = update_posterior(models, posterior, action, outcome, assumed_noise)
            p_path = path_probability * true_probability
            seen_max = max(max_surprise, surprisal)
            seen_loss = cumulative_loss + surprisal
            probes = probes_used + 1
            if surprise_threshold is not None and surprisal > surprise_threshold:
                terminals.append(Terminal(p_path, updated, probes, seen_max, seen_loss, True))
            else:
                recurse(updated, next_actions, depth_left - 1, p_path, probes, seen_max, seen_loss)

    recurse(initial, actions, budget, 1.0, 0, 0.0, 0.0)
    total_mass = sum(row.probability for row in terminals)
    if not math.isclose(total_mass, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise AssertionError(f"terminal trajectory mass does not normalize: {total_mass}")
    return terminals


def evaluate(models: tuple[SCM, ...], true_world: SCM, actions: tuple[Action, ...],
             acquisition_policy: str, assumed_noise: float, actual_noise: float,
             decision_rule: str, budget: int = 2,
             surprise_threshold_override: float | None = None,
             calibration_false_rejection_rate: float | None = None) -> dict:
    _rule_threshold, confidence_threshold = decision_parameters(decision_rule)
    terminals = enumerate_terminals(models, true_world, actions, acquisition_policy,
                                    assumed_noise, actual_noise, decision_rule, budget,
                                    surprise_threshold_override)
    support = any(model.graph_key() == true_world.graph_key() for model in models)
    mass_commit = 0.0
    mass_error = 0.0
    mass_correct = 0.0
    mass_surprise_abstain = 0.0
    mass_confidence_abstain = 0.0
    map_prob_weighted = 0.0
    entropy_weighted = 0.0
    probes_weighted = 0.0
    log_loss_weighted = 0.0
    for terminal in terminals:
        gp = graph_posterior(models, terminal.posterior)
        map_graph, map_probability = max(gp.items(), key=lambda item: (item[1], item[0]))
        entropy_weighted += terminal.probability * entropy(gp.values())
        map_prob_weighted += terminal.probability * map_probability
        probes_weighted += terminal.probability * terminal.probes_used
        log_loss_weighted += terminal.probability * terminal.cumulative_log_loss_bits
        if terminal.stopped_for_surprise:
            mass_surprise_abstain += terminal.probability
            continue
        confident = confidence_threshold is None or map_probability >= confidence_threshold
        if not confident:
            mass_confidence_abstain += terminal.probability
            continue
        mass_commit += terminal.probability
        correct = support and map_graph == true_world.graph_key()
        if correct:
            mass_correct += terminal.probability
        else:
            mass_error += terminal.probability
    return {
        "acquisition_policy": acquisition_policy,
        "decision_rule": decision_rule,
        "actual_noise": actual_noise,
        "assumed_noise": assumed_noise,
        "true_world": true_world.name,
        "true_graph_in_candidate_support": support,
        "coverage": mass_commit,
        "abstention_rate": 1.0 - mass_commit,
        "early_surprise_abstention_rate": mass_surprise_abstain,
        "low_confidence_abstention_rate": mass_confidence_abstain,
        "false_claim_mass": mass_error,
        "correct_claim_mass": mass_correct,
        "selective_error_rate": (mass_error / mass_commit) if mass_commit > 1e-15 else None,
        "mean_final_map_graph_probability": map_prob_weighted,
        "mean_final_graph_entropy_bits": entropy_weighted,
        "expected_interventions_used": probes_weighted,
        "expected_prequential_log_loss_bits": log_loss_weighted,
        "effective_surprise_threshold_bits": surprise_threshold_override if surprise_threshold_override is not None else _rule_threshold,
        "calibration_false_rejection_rate": calibration_false_rejection_rate,
    }


def calibrate_surprise_threshold(models: tuple[SCM, ...], calibration_world: SCM,
                                 actions: tuple[Action, ...], policy: str,
                                 assumed_noise: float, alpha: float = 0.05,
                                 budget: int = 2) -> tuple[float, float]:
    """Calibrate a fixed max-surprisal cutoff on an in-class reference world.

    Choose the lowest observed score cutoff whose exact in-class upper-tail mass
    is at most alpha. This is benchmark-oracle calibration for a toy known-truth
    fixture, not a claim that such ground-truth calibration is available in deployment.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    terminals = enumerate_terminals(models, calibration_world, actions, policy,
                                    assumed_noise, assumed_noise, "always_commit", budget)
    scores = sorted({terminal.max_surprisal_bits for terminal in terminals})
    if not scores:
        raise ValueError("calibration generated no trajectories")
    threshold = scores[-1]
    false_rejection = 0.0
    for candidate in scores:
        tail = sum(t.probability for t in terminals if t.max_surprisal_bits > candidate)
        if tail <= alpha + 1e-12:
            threshold = candidate
            false_rejection = tail
            break
    return threshold, false_rejection


def run_benchmark(source_git_blob: str = "UNBOUND") -> dict:
    models, worlds, actions = fixtures()
    calibration = {}
    for policy in ACQUISITION_POLICIES:
        threshold, false_rejection = calibrate_surprise_threshold(
            models, worlds["in_class"], actions, policy, assumed_noise=0.1, alpha=0.05, budget=2)
        calibration[policy] = {"threshold_bits": threshold,
                               "in_class_false_rejection_rate": false_rejection,
                               "calibration_world": "candidate_chain",
                               "calibration_actual_noise": 0.1,
                               "assumed_noise": 0.1,
                               "target_alpha": 0.05}
    rows = []
    for world_name, world in worlds.items():
        for actual_noise in (0.1, 0.25):
            for policy in ACQUISITION_POLICIES:
                for rule in DECISION_RULES:
                    threshold_override = None
                    calibration_fpr = None
                    if rule == "calibrated_surprise_alpha_0.05":
                        threshold_override = calibration[policy]["threshold_bits"]
                        calibration_fpr = calibration[policy]["in_class_false_rejection_rate"]
                    rows.append(evaluate(models, world, actions, policy, 0.1,
                                         actual_noise, rule, budget=2,
                                         surprise_threshold_override=threshold_override,
                                         calibration_false_rejection_rate=calibration_fpr))
    return {
        "schema": SCHEMA,
        "status": "LOCAL_EXACT_TOY_BENCHMARK_NOT_EXTERNAL_QUALIFICATION",
        "source_git_blob": source_git_blob,
        "candidate_models": [model.name for model in models],
        "worlds": {
            name: {"name": world.name, "parents": world.parents, "truth_tables": world.truth_tables,
                   "is_in_candidate_support": any(m.graph_key() == world.graph_key() for m in models)}
            for name, world in worlds.items()
        },
        "surprise_threshold_calibration": {
            "method": "choose smallest max-surprisal cutoff with exact in-class calibration-world rejection mass <= 0.05",
            "scope_warning": "single known-truth chain fixture; optimistic oracle calibration, not a deployment procedure",
            "by_acquisition_policy": calibration,
        },
        "contract": {
            "nodes": 3,
            "budget": 2,
            "actions": "hard do(X_i=v), each action run as an independent experiment; no repeated action",
            "actual_noise_grid": [0.1, 0.25],
            "assumed_noise": 0.1,
            "acquisition_policies": list(ACQUISITION_POLICIES),
            "decision_rules": list(DECISION_RULES),
            "enumeration": "exact over all binary outcomes at each adaptive intervention; path masses are exact up to floating arithmetic",
            "surprise": "per-observation surprisal -log2 predictive_probability, measured under the learner's current candidate model mixture",
            "commit_rule": "always, posterior MAP graph probability >= 0.8, surprise cutoff, or surprise cutoff plus posterior gate as named",
        },
        "metrics": {
            "coverage": "true-world probability mass on which a graph claim is emitted",
            "selective_error_rate": "wrong graph claims divided by all graph claims; every claim is wrong when truth graph is outside candidate support",
            "false_claim_mass": "true-world probability mass producing a wrong graph claim",
            "correct_claim_mass": "true-world probability mass producing the exact true graph claim",
            "expected_interventions_used": "expected number of interventions before commitment/abstention, bounded by budget",
            "expected_prequential_log_loss_bits": "expected cumulative negative log2 predictive probability under the true data generator",
        },
        "rows": rows,
    }


def self_test() -> None:
    models, worlds, actions = fixtures()
    chain, fork = models
    assert chain.graph_key() != fork.graph_key()
    assert worlds["in_class"].graph_key() == chain.graph_key()
    assert not any(m.graph_key() == worlds["out_of_class"].graph_key() for m in models)
    for noise in (0.1, 0.25):
        for world in worlds.values():
            for action in actions:
                dist = outcome_distribution(world, action, noise)
                assert math.isclose(sum(dist.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
                predicted = predictive_outcomes(models, (0.5, 0.5), action, 0.1)
                assert math.isclose(sum(predicted.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
                for outcome in predicted:
                    post = update_posterior(models, (0.5, 0.5), action, outcome, 0.1)
                    assert math.isclose(sum(post), 1.0, rel_tol=0.0, abs_tol=1e-12)
    for rule in DECISION_RULES:
        terms = enumerate_terminals(models, worlds["out_of_class"], actions, "greedy_eig", 0.1, 0.1, rule)
        assert math.isclose(sum(t.probability for t in terms), 1.0, rel_tol=0.0, abs_tol=1e-10)
    for policy in ACQUISITION_POLICIES:
        threshold, fpr = calibrate_surprise_threshold(models, worlds["in_class"], actions,
                                                      policy, 0.1, alpha=0.05)
        self_terms = enumerate_terminals(models, worlds["in_class"], actions, policy,
                                         0.1, 0.1, "calibrated_surprise_alpha_0.05",
                                         surprise_threshold_override=threshold)
        measured_fpr = sum(t.probability for t in self_terms if t.stopped_for_surprise)
        assert measured_fpr <= 0.05 + 1e-12
        assert abs(measured_fpr - fpr) <= 1e-10
    rows = run_benchmark("self-test")
    assert len(rows["rows"]) == 2 * 2 * len(ACQUISITION_POLICIES) * len(DECISION_RULES)
    # Lower surprise cutoffs must never increase coverage when the other policy is held fixed.
    for world in worlds.values():
        loose = evaluate(models, world, actions, "greedy_eig", 0.1, 0.1, "surprise_gate_4_bits")
        strict = evaluate(models, world, actions, "greedy_eig", 0.1, 0.1, "surprise_gate_2_bits")
        assert strict["coverage"] <= loose["coverage"] + 1e-12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-git-blob", default="UNBOUND")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            print("SELECTIVE_CAUSAL_DISCOVERY_SELF_TESTS_PASSED")
            return 0
        result = run_benchmark(args.source_git_blob)
        encoded = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(encoded)
            print(f"wrote {args.output} bytes={len(encoded)} sha256={hashlib.sha256(encoded).hexdigest()}")
        else:
            print(encoded.decode(), end="")
        return 0
    except (OSError, ValueError, AssertionError, RuntimeError) as error:
        print(f"NOT QUALIFIED: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
