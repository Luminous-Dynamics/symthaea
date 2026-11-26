#!/usr/bin/env python3
"""HIST-CAUSAL-001 v0.6 exact out-of-family challenge.

The hidden regime is redrawn per independent probe trial, creating a
population-level mixture. Research reference only; no physical claim.
"""
from __future__ import annotations
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import sys

V05 = Path(__file__).resolve().parent.parent / "v0.5"
sys.path.insert(0, str(V05))
import generator as base

OOD_MODEL = "out_of_family_sensor_actuator_mixture"
MIXTURE_WEIGHT = F(1, 2)
MIX_COMPONENTS = ("primary_sensor_artefact", "actuator_noncompliance")


def ood_distribution(probe: str, params: dict, weight: F = MIXTURE_WEIGHT):
    if not F(0) < weight < F(1):
        raise ValueError("mixture weight must be strictly between zero and one")
    left = base._dist(MIX_COMPONENTS[0], probe, params)
    right = base._dist(MIX_COMPONENTS[1], probe, params)
    result = {}
    for outcome in set(left) | set(right):
        value = weight * left.get(outcome, F(0)) + (1 - weight) * right.get(outcome, F(0))
        if value:
            result[outcome] = value
    if sum(result.values(), F(0)) != 1:
        raise ArithmeticError(f"out-of-family distribution not normalized: {probe}")
    return result


def total_variation(left: dict, right: dict) -> F:
    return sum((abs(left.get(k, F(0)) - right.get(k, F(0)))
                for k in set(left) | set(right)), F(0)) / 2


def signature_distance(model: str, params: dict, weight: F = MIXTURE_WEIGHT) -> F:
    return sum((total_variation(ood_distribution(p, params, weight),
                                 base._dist(model, p, params))
                for p in base.PROBES), F(0)) / len(base.PROBES)


def validate_ood(seed: int, weight: F = MIXTURE_WEIGHT) -> dict:
    task = base.generate_recipe(seed)
    params = task["_params"]
    distances = {m: signature_distance(m, params, weight) for m in base.MODELS}
    minimum = min(distances.values())
    if minimum <= 0:
        raise AssertionError(f"OOD signature aliases an in-family model at seed {seed}")
    return {
        "seed": seed,
        "task_id": task["agent_view"]["task_id"],
        "minimum_average_total_variation": str(minimum),
        "nearest_in_family_models": [m for m, d in distances.items() if d == minimum],
        "all_candidate_distances": {m: str(d) for m, d in distances.items()},
        "all_probe_distributions_normalized": all(
            sum(ood_distribution(p, params, weight).values(), F(0)) == 1
            for p in base.PROBES),
    }


def agent_task(seed: int) -> dict:
    task = base.generate_recipe(seed)
    view = dict(task["agent_view"])
    view["allowed_outputs"] = ["claim_model", "equivalence_class", "model_misspecified", "abstain"]
    view["open_world_warning"] = (
        "The true data-generating process may be outside the six named candidates. "
        "Do not force a singleton; return model_misspecified or abstain when all candidates fail."
    )
    view["task_id"] = sha256(f"HIST-CAUSAL-001|0.6|{seed}".encode()).hexdigest()[:20]
    return view


def evaluator_key(seed: int, weight: F = MIXTURE_WEIGHT) -> dict:
    task = base.generate_recipe(seed)
    return {
        "seed": seed,
        "target_model": OOD_MODEL,
        "mixture_components": list(MIX_COMPONENTS),
        "mixture_weight": str(weight),
        "recipe_parameters": {k: str(v) for k, v in task["_params"].items()},
        "claim_ceiling": "out-of-family mixture of two declared finite mechanisms; not a physical mechanism",
    }


def sample_ood_trial_evaluator(seed: int, probe: str, replicate: int, params: dict,
                               weight: F = MIXTURE_WEIGHT):
    # The hidden component is redrawn per independent trial/experimental unit.
    salt = sum((i + 1) * b for i, b in enumerate((probe + "|ood-component").encode()))
    _, raw = base._splitmix64((seed ^ salt ^ ((replicate + 1) * 0xD6E8FEB86659FD93)) & base.MASK)
    choose_left = F(raw, 1 << 64) < weight
    component = MIX_COMPONENTS[0] if choose_left else MIX_COMPONENTS[1]
    outcome = base.sample_probe(seed ^ 0xA5A5A5A5A5A5A5A5, probe, replicate,
                                component, params)
    return outcome, component


def sample_ood_probe(seed: int, probe: str, replicate: int, params: dict,
                     weight: F = MIXTURE_WEIGHT):
    """Return only the public observation; never return latent component to solver."""
    outcome, _private_component = sample_ood_trial_evaluator(seed, probe, replicate, params, weight)
    return outcome


def run(seed_count: int = 100) -> dict:
    rows = [validate_ood(seed) for seed in range(seed_count)]
    min_sep = min(F(r["minimum_average_total_variation"]) for r in rows)
    mean_sep = sum((F(r["minimum_average_total_variation"]) for r in rows), F(0)) / len(rows)
    return {
        "benchmark": "HIST-CAUSAL-001",
        "version": "0.6",
        "out_of_family_model": OOD_MODEL,
        "mixture_components": list(MIX_COMPONENTS),
        "mixture_weight": str(MIXTURE_WEIGHT),
        "development_seeds_validated": seed_count,
        "all_ood_signatures_differ_from_all_six_candidates": True,
        "minimum_nearest_model_average_tv": str(min_sep),
        "mean_nearest_model_average_tv": str(mean_sep),
        "claim_ceiling": "finite mixture model is outside the six declared signatures over this probe menu; finite-sample detection not yet established",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
