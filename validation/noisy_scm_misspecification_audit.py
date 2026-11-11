#!/usr/bin/env python3
"""Exact finite SCM audit separating the true world from the learner's model class.

Standard-library only. This is a falsification fixture, not a general causal-discovery
algorithm or an ACDB/CausalArena reproduction. Outcomes are enumerated exactly; no
Monte Carlo sampling is used. The true graph is deliberately absent from the learner's
candidate set, so falling posterior entropy is not evidence of graph correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCHEMA = "noisy-scm-model-misspecification-audit-v1"
POLICIES = ("fixed_order", "random", "greedy_eig", "assumed_model_oracle")


@dataclass(frozen=True)
class SCM:
    name: str
    parents: tuple[tuple[int, ...], ...]
    tables: tuple[int, ...]

    def graph_key(self) -> tuple[tuple[int, ...], ...]:
        return self.parents


@dataclass(frozen=True)
class Action:
    identifier: int
    target: int
    value: int


@dataclass(frozen=True)
class Metrics:
    residual_graph_entropy_bits: float
    final_map_graph_probability: float
    prequential_log_loss_bits: float
    false_high_confidence: float


def entropy(weights: Iterable[float]) -> float:
    positive = [float(p) for p in weights if p > 0.0]
    total = sum(positive)
    if total <= 0.0:
        return 0.0
    return -sum((p / total) * math.log2(p / total) for p in positive)


def mechanism_output(model: SCM, node: int, assignment: tuple[int, ...]) -> int:
    parent_assignment = 0
    for bit, parent in enumerate(model.parents[node]):
        parent_assignment |= (assignment[parent] & 1) << bit
    return (model.tables[node] >> parent_assignment) & 1


def likelihood(model: SCM, action: Action, outcome: tuple[int, ...], noise: float) -> float:
    """P(outcome | model, hard do(X_target=value), symmetric mechanism noise)."""
    if outcome[action.target] != action.value:
        return 0.0
    prob = 1.0
    for node in range(len(outcome)):
        if node == action.target:
            continue
        expected = mechanism_output(model, node, outcome)
        prob *= (1.0 - noise) if outcome[node] == expected else noise
    return prob


def outcome_distribution(model: SCM, action: Action, noise: float) -> dict[tuple[int, ...], float]:
    result: dict[tuple[int, ...], float] = {}
    for outcome in itertools.product((0, 1), repeat=len(model.parents)):
        p = likelihood(model, action, outcome, noise)
        if p > 0.0:
            result[outcome] = p
    return result


def predictive_outcomes(models: tuple[SCM, ...], posterior: tuple[float, ...],
                        action: Action, assumed_noise: float) -> dict[tuple[int, ...], float]:
    result: dict[tuple[int, ...], float] = {}
    for model, prior_mass in zip(models, posterior):
        for outcome, p in outcome_distribution(model, action, assumed_noise).items():
            result[outcome] = result.get(outcome, 0.0) + prior_mass * p
    total = sum(result.values())
    if total <= 0.0:
        raise ValueError("learner predictive distribution has zero mass")
    return {outcome: p / total for outcome, p in result.items()}


def update_posterior(models: tuple[SCM, ...], posterior: tuple[float, ...], action: Action,
                     outcome: tuple[int, ...], assumed_noise: float) -> tuple[float, ...]:
    values = tuple(p * likelihood(m, action, outcome, assumed_noise)
                   for m, p in zip(models, posterior))
    total = sum(values)
    if total <= 0.0:
        raise ValueError("observation lies outside learner support; fail closed")
    updated = tuple(v / total for v in values)
    if not math.isclose(sum(updated), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("posterior normalization failed")
    return updated


def graph_posterior(models: tuple[SCM, ...], posterior: tuple[float, ...]) -> dict[tuple[tuple[int, ...], ...], float]:
    result: dict[tuple[tuple[int, ...], ...], float] = {}
    for model, weight in zip(models, posterior):
        key = model.graph_key()
        result[key] = result.get(key, 0.0) + weight
    return result


def graph_entropy(models: tuple[SCM, ...], posterior: tuple[float, ...]) -> float:
    return entropy(graph_posterior(models, posterior).values())


def expected_residual_entropy(models: tuple[SCM, ...], posterior: tuple[float, ...],
                              actions: tuple[Action, ...], depth: int,
                              assumed_noise: float) -> float:
    if depth <= 0 or not actions:
        return graph_entropy(models, posterior)
    values = []
    for action in actions:
        predictive = predictive_outcomes(models, posterior, action, assumed_noise)
        after = sum(p * expected_residual_entropy(
            models, update_posterior(models, posterior, action, outcome, assumed_noise),
            tuple(a for a in actions if a.identifier != action.identifier), depth - 1,
            assumed_noise) for outcome, p in predictive.items())
        values.append(after)
    return min(values)


def choose_action(policy: str, models: tuple[SCM, ...], posterior: tuple[float, ...],
                  actions: tuple[Action, ...], depth_left: int, assumed_noise: float,
                  random_rank: dict[int, int]) -> Action:
    if not actions:
        raise ValueError("no remaining interventions")
    if policy == "fixed_order":
        return min(actions, key=lambda a: a.identifier)
    if policy == "random":
        return min(actions, key=lambda a: (random_rank[a.identifier], a.identifier))
    if policy == "greedy_eig":
        def value(action: Action) -> tuple[float, int]:
            predictive = predictive_outcomes(models, posterior, action, assumed_noise)
            after_h = sum(p * graph_entropy(models, update_posterior(
                models, posterior, action, outcome, assumed_noise))
                for outcome, p in predictive.items())
            return (after_h, action.identifier)
        return min(actions, key=value)
    if policy == "assumed_model_oracle":
        # Exact dynamic programming under the learner's predictive distribution.
        def value(action: Action) -> tuple[float, int]:
            predictive = predictive_outcomes(models, posterior, action, assumed_noise)
            remaining = tuple(a for a in actions if a.identifier != action.identifier)
            after = sum(p * expected_residual_entropy(
                models, update_posterior(models, posterior, action, outcome, assumed_noise),
                remaining, depth_left - 1, assumed_noise)
                for outcome, p in predictive.items())
            return (after, action.identifier)
        return min(actions, key=value)
    raise ValueError(f"unknown policy: {policy}")


def evaluate(policy: str, models: tuple[SCM, ...], true_world: SCM, actions: tuple[Action, ...],
             assumed_noise: float, actual_noise: float, random_seed: int) -> Metrics:
    rng = random.Random(random_seed)
    ids = [a.identifier for a in actions]
    rng.shuffle(ids)
    rank = {action_id: position for position, action_id in enumerate(ids)}
    uniform = tuple(1.0 / len(models) for _ in models)

    def walk(posterior: tuple[float, ...], remaining: tuple[Action, ...], depth: int,
             accumulated_log_loss: float) -> tuple[float, float, float, float]:
        if depth <= 0 or not remaining:
            probs = graph_posterior(models, posterior)
            best = max(probs.values(), default=0.0)
            # True graph is outside support: MAP posterior >= .9 is false confidence.
            return (graph_entropy(models, posterior), best, accumulated_log_loss,
                    1.0 if best >= 0.9 else 0.0)
        action = choose_action(policy, models, posterior, remaining, depth,
                               assumed_noise, rank)
        learner_predictive = predictive_outcomes(models, posterior, action, assumed_noise)
        actual = outcome_distribution(true_world, action, actual_noise)
        next_actions = tuple(a for a in remaining if a.identifier != action.identifier)
        branches = []
        for outcome, true_p in actual.items():
            predicted_p = learner_predictive.get(outcome, 0.0)
            log_loss = float("inf") if predicted_p <= 0.0 else -math.log2(predicted_p)
            updated = update_posterior(models, posterior, action, outcome, assumed_noise)
            branches.append((true_p, walk(updated, next_actions, depth - 1,
                                           accumulated_log_loss + log_loss)))
        return tuple(sum(p * values[i] for p, values in branches) for i in range(4))  # type: ignore[return-value]

    # Two interventions; expectations are enumerated exactly under the true world.
    return Metrics(*walk(uniform, actions, 2, 0.0))


def fixture() -> tuple[tuple[SCM, ...], SCM, tuple[Action, ...]]:
    # Learner: chain and fork. Both are plausible, but neither is the truth.
    chain = SCM("candidate_chain", ((), (0,), (1,)), (0, 0b10, 0b10))
    fork = SCM("candidate_fork", ((), (0,), (0,)), (0, 0b10, 0b10))
    # Truth: XOR collider X0 -> X2 <- X1, excluded from the candidate class.
    collider = SCM("true_xor_collider", ((), (), (0, 1)), (0, 0, 0b0110))
    actions = tuple(Action(2 * node + value, node, value)
                    for node in range(3) for value in (0, 1))
    return (chain, fork), collider, actions


def run_audit(source_sha: str = "UNBOUND") -> dict:
    models, true_world, actions = fixture()
    if any(m.graph_key() == true_world.graph_key() for m in models):
        raise AssertionError("falsification fixture accidentally includes true graph")
    scenarios = []
    for actual_noise in (0.05, 0.2, 0.35):
        for assumed_noise in (0.1, 0.25):
            for policy in POLICIES:
                seeds = range(8) if policy == "random" else range(1)
                runs = [evaluate(policy, models, true_world, actions, assumed_noise,
                                 actual_noise, seed) for seed in seeds]
                n = len(runs)
                scenarios.append({
                    "actual_noise": actual_noise,
                    "assumed_noise": assumed_noise,
                    "policy": policy,
                    "random_replicates": n,
                    "expected_residual_graph_entropy_bits": sum(r.residual_graph_entropy_bits for r in runs) / n,
                    "expected_final_map_graph_probability": sum(r.final_map_graph_probability for r in runs) / n,
                    "expected_prequential_log_loss_bits": sum(r.prequential_log_loss_bits for r in runs) / n,
                    "false_high_confidence_rate": sum(r.false_high_confidence for r in runs) / n,
                    "true_graph_in_candidate_support": False,
                })
    return {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "status": "EXPLORATORY_EXACT_ENUMERATION_NOT_EXTERNAL_BENCHMARK",
        "true_world": {"name": true_world.name, "parents": true_world.parents, "tables": true_world.tables},
        "candidate_models": [{"name": m.name, "parents": m.parents, "tables": m.tables} for m in models],
        "intervention_semantics": "hard do(X_i=v); target mechanism replaced; other equations independently flip with configured probability",
        "assumed_noise_grid": [0.1, 0.25],
        "actual_noise_grid": [0.05, 0.2, 0.35],
        "metrics": {
            "residual graph entropy": "learner posterior uncertainty, not correctness",
            "prequential log loss": "bits assigned by learner's current predictive distribution to true-world outcomes",
            "false high confidence": "fraction of exact outcome branches ending with MAP candidate-graph posterior >= 0.9 while true graph is outside support"
        },
        "true_graph_in_candidate_support": False,
        "results": scenarios,
    }


def self_test() -> None:
    models, true_world, actions = fixture()
    for action in actions:
        actual = outcome_distribution(true_world, action, 0.2)
        assert math.isclose(sum(actual.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
        predicted = predictive_outcomes(models, (0.5, 0.5), action, 0.1)
        assert math.isclose(sum(predicted.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
        for outcome in predicted:
            updated = update_posterior(models, (0.5, 0.5), action, outcome, 0.1)
            assert math.isclose(sum(updated), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert not any(m.graph_key() == true_world.graph_key() for m in models)
    for policy in POLICIES:
        assert evaluate(policy, models, true_world, actions, 0.1, 0.2, 41) == evaluate(
            policy, models, true_world, actions, 0.1, 0.2, 41)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-sha", default="UNBOUND")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            print("NOISY_SCM_MISSPECIFICATION_SELF_TESTS_PASSED")
            return 0
        payload = run_audit(args.source_sha)
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
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
