#!/usr/bin/env python3
"""Exact open-world model-adequacy control; standard library only."""
import argparse
import copy
import json
import sys
from fractions import Fraction as F
from pathlib import Path


def vec(x):
    if not isinstance(x, list) or not x:
        raise ValueError("expected non-empty vector")
    return [F(v) for v in x]


def mat(x):
    a = [vec(row) for row in x]
    if not a or any(len(r) != len(a[0]) for r in a):
        raise ValueError("invalid matrix")
    return a


def inv(a):
    n = len(a)
    if n == 0 or any(len(r) != n for r in a):
        raise ValueError("inverse requires square matrix")
    z = [r[:] + [F(i == j) for j in range(n)] for i, r in enumerate(a)]
    for c in range(n):
        p = next((r for r in range(c, n) if z[r][c]), None)
        if p is None:
            raise ValueError("singular covariance")
        z[c], z[p] = z[p], z[c]
        q = z[c][c]
        z[c] = [v/q for v in z[c]]
        for r in range(n):
            if r != c:
                q = z[r][c]
                z[r] = [v-q*w for v, w in zip(z[r], z[c])]
    return [r[n:] for r in z]


def qform(delta, precision):
    if len(delta) != len(precision) or any(len(r) != len(delta) for r in precision):
        raise ValueError("quadratic form dimensions differ")
    transformed = [sum((a*b for a, b in zip(row, delta)), F(0)) for row in precision]
    return sum((a*b for a, b in zip(delta, transformed)), F(0))


def same(a, b, label):
    if a != b:
        raise AssertionError(f"{label}: expected {b}, got {a}")


def evaluate(p):
    obs, cov = vec(p["held_out_observation"]), mat(p["held_out_covariance"])
    if len(cov) != len(obs) or any(len(r) != len(obs) for r in cov):
        raise ValueError("held-out covariance dimensions differ")
    if any(cov[i][j] != cov[j][i] for i in range(len(cov)) for j in range(len(cov))):
        raise ValueError("held-out covariance must be symmetric")
    precision = inv(cov)
    gate = F(p["maximum_admissible_squared_residual"])
    expected = p["expected"]
    if obs != vec(expected["frozen_held_out_observation"]):
        raise AssertionError("held-out observation differs from frozen fixture")
    if gate != F(expected["frozen_maximum_admissible_squared_residual"]):
        raise AssertionError("adequacy threshold differs from predeclared gate")
    candidates = p["initial_candidate_family"]
    if len(candidates) < 2:
        raise ValueError("at least two rival candidates are required")
    means = {name: vec(model["predicted_mean"]) for name, model in candidates.items()}
    if any(len(mean) != len(obs) for mean in means.values()):
        raise ValueError("candidate and held-out observation dimensions differ")
    residuals = {name: qform([o-m for o, m in zip(obs, mean)], precision) for name, mean in means.items()}
    ids = list(means)
    separation = qform([a-b for a, b in zip(means[ids[0]], means[ids[1]])], precision)
    admissible = [name for name, residual in residuals.items() if residual <= gate]
    disposition = "CANDIDATE_FAMILY_HAS_ADMISSIBLE_MODEL" if admissible else "ABSTAIN_AND_EXPAND_MODEL_CLASS"
    same(separation, F(expected["pairwise_candidate_separation_squared"]), "pairwise separation")
    same(residuals, {"H0": F(expected["H0_squared_held_out_residual"]), "H1": F(expected["H1_squared_held_out_residual"])}, "held-out residuals")
    same(disposition, expected["initial_family_disposition"], "open-world disposition")
    if admissible:
        raise AssertionError("the initial candidate family unexpectedly passed adequacy")

    witness = p["outside_family_witness"]
    witness_mean = vec(witness["predicted_mean"])
    witness_residual = qform([o-m for o, m in zip(obs, witness_mean)], precision)
    same(witness_residual, F(witness["expected_squared_held_out_residual"]), "outside-family residual")
    if witness_residual > gate:
        raise AssertionError("outside-family witness should pass the synthetic adequacy gate")

    rejected = []
    def must_reject(candidate, label):
        try:
            evaluate(candidate)
        except (ValueError, AssertionError, KeyError, TypeError):
            rejected.append(label)
            return
        raise AssertionError(f"hostile mutation was accepted: {label}")

    bad = copy.deepcopy(p)
    bad["maximum_admissible_squared_residual"] = "3"
    must_reject(bad, "post_hoc_gate_relaxation")
    bad = copy.deepcopy(p)
    bad["held_out_observation"] = ["1", "0"]
    must_reject(bad, "held_out_observation_mutation")
    bad = copy.deepcopy(p)
    bad["initial_candidate_family"]["H0"]["predicted_mean"] = ["0", "0"]
    must_reject(bad, "candidate_prediction_mutation")
    bad = copy.deepcopy(p)
    bad["outside_family_witness"]["predicted_mean"] = ["1", "0"]
    must_reject(bad, "outside_family_witness_mutation")

    return {
        "status": "PASS_EXACT_SYNTHETIC_OPEN_WORLD_CONTROL_ONLY",
        "pairwise_candidate_separation_squared": str(separation),
        "held_out_residual_squared": {k: str(v) for k, v in residuals.items()},
        "maximum_admissible_squared_residual": str(gate),
        "initial_family_disposition": disposition,
        "outside_family_witness_residual_squared": str(witness_residual),
        "outside_family_witness_disposition": witness["expected_expanded_family_disposition"],
        "all_initial_candidates_rejected": True,
        "negative_controls_passed": len(rejected),
        "negative_controls": rejected,
        "claim_ceiling": p["claim_ceiling"]
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture", nargs="?", type=Path, default=Path(__file__).with_name("BH_EGB_OPEN_WORLD_MODEL_ADEQUACY_CONTROL_V0_1.json"))
    args = ap.parse_args()
    try:
        p = json.loads(args.fixture.read_text(encoding="utf-8"))
        report = evaluate(p)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
