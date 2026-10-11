#!/usr/bin/env python3
"""RH-006 effect/risk-radius and branch-conditioning diagnostic.

This is a research-only geometry mirror. It reports how much induced
held-out outcome signal is required to reach the finite-sample equal-risk
surface and how sensitive each root is to coefficient/nuisance perturbation.
No formal inference is performed.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from rh006_null_surface_eigenstructure_splitmix64 import (
    SM, feat, mu0, sig0, quadratic, discriminant_matrix,
    eigenstructure_directions, roots, N, ORIGINS, STEP, TRAIN, GAP, TEST,
)

RADII = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
WEAK = np.array([0.0, 1.0, -2.0], dtype=float)
WEAK /= np.linalg.norm(WEAK)

def effect_metric(F, sig):
    Z = F[:, 3:6]
    G = np.zeros((3, 3))
    for o in range(ORIGINS):
        s = o * STEP
        te = s + TRAIN + GAP
        tt = slice(te, te + TEST)
        G += (Z[tt].T @ Z[tt]) / TEST
    G /= ORIGINS
    sigma_ref = float(np.sqrt(np.mean(np.square(sig))))
    return G, max(sigma_ref, 1e-12)

def point_metrics(Q, b, c, v, t, G, sigma_ref):
    A = float(v @ Q @ v)
    B = float(b @ v)
    D = B * B - 4.0 * A * c
    gamma = t * v
    effect = float(np.sqrt(max(gamma @ G @ gamma, 0.0)) / sigma_ref)
    radial = abs(2.0 * A * t + B)
    radial_scale = abs(B) + 2.0 * abs(A) * abs(t)
    radial_condition = radial / max(radial_scale, 1e-30)
    return {
        "A": A,
        "B": B,
        "D": D,
        "root": float(t),
        "effect_radius": effect,
        "radial_derivative_abs": radial,
        "radial_condition": radial_condition,
        "critical": radial_condition < 1e-4,
        "curvature_ratio": abs(A) / max(abs(np.linalg.eigvalsh((Q + Q.T) / 2)).max(), 1e-30),
    }

def run(args):
    rng = SM(args.seed)
    surface = defaultdict(lambda: {
        "attempts": 0, "roots": 0, "effect_values": [], "radial_values": [],
        "disc_values": [], "curvature_values": [], "radius_counts": defaultdict(int),
        "critical": 0,
    })
    weak = {"attempts": 0, "roots": 0, "effect_values": [], "radial_values": [],
            "disc_values": [], "radius_counts": defaultdict(int), "critical": 0}
    branches = defaultdict(int)

    for _ in range(args.paths):
        F = feat(rng, args.epsilon)
        m = mu0(F)
        sig = sig0(F)
        Q, b, c, _ = quadratic(F, args.ridge, m, 0.5, sig)
        R = discriminant_matrix(Q, b, c)
        labels, directions = eigenstructure_directions(Q, R)
        G, sigma_ref = effect_metric(F, sig)

        for label, v in zip(labels, directions):
            bucket = surface[label]
            bucket["attempts"] += 1
            rr = roots(Q, b, c, v)
            if not rr:
                continue

            bucket["roots"] += len(rr)
            for j, t in enumerate(rr):
                branch = "near" if j == 0 else "far"
                branches[branch] += 1
                pm = point_metrics(Q, b, c, v, t, G, sigma_ref)
                bucket["effect_values"].append(pm["effect_radius"])
                bucket["radial_values"].append(pm["radial_condition"])
                bucket["disc_values"].append(pm["D"])
                bucket["curvature_values"].append(pm["curvature_ratio"])
                bucket["critical"] += int(pm["critical"])
                for r in RADII:
                    bucket["radius_counts"][str(r)] += int(pm["effect_radius"] <= r)

                if np.allclose(v, WEAK, rtol=1e-10, atol=1e-10):
                    weak["roots"] += 1
                    weak["effect_values"].append(pm["effect_radius"])
                    weak["radial_values"].append(pm["radial_condition"])
                    weak["disc_values"].append(pm["D"])
                    weak["critical"] += int(pm["critical"])
                    for r in RADII:
                        weak["radius_counts"][str(r)] += int(pm["effect_radius"] <= r)
        weak["attempts"] += 1

    def summarize(d):
        n = len(d["effect_values"])
        return {
            "attempts": d["attempts"],
            "roots": d["roots"],
            "root_rate_per_attempt": d["roots"] / max(d["attempts"], 1),
            "effect_radius_quantiles": np.quantile(d["effect_values"], [0, .1, .5, .9, .99, 1]).tolist() if n else [],
            "radial_condition_quantiles": np.quantile(d["radial_values"], [0, .1, .5, .9, .99, 1]).tolist() if n else [],
            "discriminant_quantiles": np.quantile(d["disc_values"], [0, .1, .5, .9, .99, 1]).tolist() if n else [],
            "critical_fraction": d["critical"] / max(n, 1),
            "supported_within_effect_radius": {
                r: d["radius_counts"].get(str(r), 0) / max(n, 1) for r in RADII
            },
        }

    direction_summary = {}
    for label, d in sorted(surface.items()):
        direction_summary[label] = summarize(d)

    return {
        "schema": "rh006-null-surface-effect-radius-executed/v1",
        "status": "research-diagnostic-only",
        "formal_inference": False,
        "applicability": "not-approved-for-execution",
        "selection": "stop-assumption-failure",
        "mechanics_scope": "independent Python estimator-mechanics mirror; not the Symthaea Rust implementation",
        "geometry": {
            "train": TRAIN, "gap": GAP, "test": TEST, "origins": ORIGINS, "step": STEP,
            "epsilon": args.epsilon, "ridge": args.ridge, "paths": args.paths, "seed": args.seed,
            "effect_metric": "sqrt(gamma'Ggamma)/sigma_ref, where G is mean held-out Z'Z/test and sigma_ref is mean synthetic innovation SD",
            "radius_ladder": list(RADII),
        },
        "branch_conditioning": {
            "identity": "At a quadratic root, abs(2At+B)=sqrt(D)",
            "critical_rule": "normalized radial condition < 1e-4",
            "root_branches": "both positive roots retained; near/far are ordering labels only",
        },
        "results": {
            "weak_direction": summarize(weak),
            "directions": direction_summary,
            "branch_counts": dict(branches),
        },
        "gates": {
            "surface_support": "measured",
            "effect_locality_support": "measured",
            "branch_conditioning": "measured",
            "uniform_calibration": "not-established",
            "least_favorable_distribution": "not-established",
        },
        "nonclaims": [
            "No formal p-value or confidence interval.",
            "No uniform size guarantee.",
            "No least-favorable critical value.",
            "No empirical validation.",
            "No claim that the chosen radius ladder is a formal scientific locality policy.",
        ],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--epsilon", type=float, default=1e-4)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()
    out = run(args)
    payload = json.dumps(out, sort_keys=True, separators=(",", ":"))
    out["payload_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    print(json.dumps(out, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
