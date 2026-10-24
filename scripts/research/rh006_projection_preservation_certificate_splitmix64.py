#!/usr/bin/env python3
"""RH-006 origin-local projection hypothesis-preservation certificate.

Research diagnostic only. Tests whether an origin-local SVD reduction preserves
predeclared coefficient-space hypotheses. A projection may define a new
projected estimand, but it cannot represent the original directional hypothesis
unless the declared effect direction is preserved by the projection at every
required origin.
"""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

TRAIN, GAP, TEST, ORIGINS, STEP = 48, 4, 16, 24, 16
N = TRAIN + GAP + TEST + (ORIGINS - 1) * STEP
EPSILONS = (1e-2, 1e-3, 1e-4, 0.0)
TAUS = (1e-2, 1e-3, 1e-4)
PRESERVE_TOL = 1e-10
DIRECTIONS = {
    "strong": np.array([1.0, 1.0, 0.5], float),
    "weak": np.array([0.0, 1.0, -2.0], float),
}


class SplitMix64:
    def __init__(self, seed):
        self.state = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)

    def rademacher(self, size):
        size = int(size)
        idx = self.state + np.arange(size, dtype=np.uint64)
        z = idx + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        self.state = np.uint64((int(self.state) + size) & 0xFFFFFFFFFFFFFFFF)
        return np.where((z & np.uint64(1)) == 0, -1.0, 1.0)


def ar_series(n, rng, rho):
    e = rng.rademacher(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def features(rng, epsilon):
    raw = np.column_stack([ar_series(N, rng, 0.35) for _ in range(7)])

    def n01(x):
        return np.clip(0.5 + 0.18 * x, 0.0, 1.0)

    a = n01(raw[:, 0])
    b = n01(raw[:, 1])
    common = n01(raw[:, 2])
    alignment = n01(0.65 * raw[:, 3] + 0.35 * raw[:, 4])
    a_to_b = n01(raw[:, 5])
    b_to_a = n01(raw[:, 6])
    turn = n01(0.45 * raw[:, 4] + 0.55 * raw[:, 5])
    F = np.column_stack([a, b, alignment, a_to_b, b_to_a, turn, common])
    if epsilon < 1.0:
        z = F.copy()
        z[:, 4] = np.clip(F[:, 3] + epsilon * 0.18 * ar_series(N, rng, 0.25), 0.0, 1.0)
        z[:, 5] = np.clip(0.25 + 0.5 * F[:, 3] + epsilon * 0.18 * ar_series(N, rng, 0.25), 0.0, 1.0)
        F = z
    return F


def origin_projection(F, start, tau):
    R = F[start:start + TRAIN, 3:6]
    scale = np.where(R.std(axis=0) <= 1e-12, 1.0, R.std(axis=0))
    Rstd = (R - R.mean(axis=0)) / scale
    _, S, Vt = np.linalg.svd(Rstd, full_matrices=False)
    if S[0] <= 0:
        k = 0
        P = np.zeros((3, 3))
    else:
        k = int(np.sum(S / S[0] >= tau))
        V = Vt[:k].T
        P = V @ V.T
    return P, S, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()

    agg = {(eps, tau): {name: [] for name in DIRECTIONS} for eps in EPSILONS for tau in TAUS}
    path_exact = {(eps, tau): {name: [] for name in DIRECTIONS} for eps in EPSILONS for tau in TAUS}
    path_worst = {(eps, tau): {name: [] for name in DIRECTIONS} for eps in EPSILONS for tau in TAUS}
    full = {(eps, tau): [] for eps in EPSILONS for tau in TAUS}

    rng = SplitMix64(args.seed)
    for _ in range(args.mc):
        for eps in EPSILONS:
            F = features(rng, eps)
            for tau in TAUS:
                path_full_ok = True
                path_dir_ok = {name: True for name in DIRECTIONS}
                path_dir_worst = {name: 1.0 for name in DIRECTIONS}
                for o in range(ORIGINS):
                    P, S, scale = origin_projection(F, o * STEP, tau)
                    k = len(S) if S.size == 0 else int(np.sum(S / S[0] >= tau))
                    if k < 3:
                        path_full_ok = False
                    for name, v in DIRECTIONS.items():
                        vstd = scale * v
                        denom = float(np.linalg.norm(vstd))
                        retention = float(np.linalg.norm(P @ vstd) / denom) if denom > 0 else 0.0
                        agg[(eps, tau)][name].append((retention, 1.0 - retention * retention))
                        if retention < 1.0 - PRESERVE_TOL:
                            path_dir_ok[name] = False
                        path_dir_worst[name] = min(path_dir_worst[name], retention)
                full[(eps, tau)].append(path_full_ok)
                for name in DIRECTIONS:
                    path_exact[(eps, tau)][name].append(path_dir_ok[name])
                    path_worst[(eps, tau)][name].append(path_dir_worst[name])

    rows = []
    for eps in EPSILONS:
        for tau in TAUS:
            row = {
                "epsilon": eps,
                "tau": tau,
                "full_three_channel_preserved_rate": float(np.mean(full[(eps, tau)])),
                "directions": {}
            }
            for name in DIRECTIONS:
                vals = np.asarray(agg[(eps, tau)][name], float)
                ret = vals[:, 0]
                loss = vals[:, 1]
                row["directions"][name] = {
                    "origin_retention_median": float(np.median(ret)),
                    "origin_retention_min_global": float(np.min(ret)),
                    "origin_retention_q05": float(np.quantile(ret, 0.05)),
                    "origin_effect_energy_loss_median": float(np.median(loss)),
                    "origin_effect_energy_loss_max_global": float(np.max(loss)),
                    "origin_exact_preservation_rate": float(np.mean(ret >= 1.0 - PRESERVE_TOL)),
                    "path_exact_preservation_rate": float(np.mean(path_exact[(eps, tau)][name])),
                    "path_worst_retention_median": float(np.median(path_worst[(eps, tau)][name])),
                    "path_worst_retention_min_global": float(np.min(path_worst[(eps, tau)][name])),
                    "admissible_origin_cells": int(len(ret)),
                }
            rows.append(row)

    out = {
        "schema": "rh006-projection-hypothesis-preservation-certificate-executed/v1",
        "status": "research-diagnostic-only",
        "faithfulness_scope": "independent estimator-mechanics reimplementation; not the Symthaea Rust implementation",
        "rng": "SplitMix64 v1 + Rademacher innovations",
        "geometry": {
            "train": TRAIN, "gap": GAP, "test": TEST, "origins": ORIGINS, "step": STEP,
            "epsilons": list(EPSILONS), "taus": list(TAUS), "mc_reps": args.mc,
            "seed": args.seed, "preserve_tolerance": PRESERVE_TOL,
        },
        "hypothesis_rule": {
            "full_three_channel": "preserve the original RH-006 coefficient space only when all three singular directions are retained at every required origin",
            "directional": "a predeclared directional hypothesis is preserved only when its standardized coefficient vector lies in the retained projection at every required origin within numerical tolerance",
            "projected_estimand": "otherwise the projection is treated as a distinct projected estimand and must not be reported as inference for the original hypothesis",
        },
        "directions": {
            "strong": "raw coefficient direction [1,1,0.5]; converted to the origin's training-standardized coefficient coordinates before projection retention is computed",
            "weak": "raw coefficient direction [0,1,-2]; converted to the origin's training-standardized coefficient coordinates before projection retention is computed",
        },
        "results": rows,
        "interpretation": {
            "formal_inference": False,
            "applicability": "not-approved-for-execution",
            "selection": "stop-assumption-failure",
            "finding": "Rank improvement is not sufficient for hypothesis preservation. Any origin-local projection that removes a declared effect direction changes the inferential target; the original full-space RH-006 hypothesis requires an origin-local preservation certificate.",
        },
        "references": [
            "Chernozhukov, Hansen & Spindler (2015), Valid Post-Selection and Post-Regularization Inference",
            "Kuchibhotla, Kolassa & Kuffner (2022), Post-Selection Inference",
        ],
    }
    raw = json.dumps(out, sort_keys=True, separators=(",", ":"))
    out["payload_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    out["execution"] = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": f"python scripts/research/rh006_projection_preservation_certificate_splitmix64.py --mc {args.mc} --seed {args.seed}",
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
