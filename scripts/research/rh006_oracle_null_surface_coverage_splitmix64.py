#!/usr/bin/env python3
"""
RH-006 null-surface coverage pilot.

Independent estimator-mechanics reimplementation. For each fixed feature path,
construct the exact dependent/heteroskedastic finite-sample risk quadratic,
choose several predeclared directions on its equal-risk surface, and compare
oracle risk-null bootstrap rejection rates across those nuisance points.

Research diagnostic only. Not the Symthaea Rust implementation and not a
formal inference procedure.
"""
import argparse
import hashlib
import json
import math
import numpy as np

TRAIN, TEST, GAP, ORIGINS, STEP = 48, 16, 4, 24, 16
RIDGE, SIGMA, LAG, ALPHA = 1e-8, 0.08, 3, 0.05
N = TRAIN + GAP + TEST + (ORIGINS - 1) * STEP

DIRECTIONS = [
    ("a2b", np.array([1.0, 0.0, 0.0])),
    ("b2a", np.array([0.0, 1.0, 0.0])),
    ("turn_taking", np.array([0.0, 0.0, 1.0])),
    ("plus_plus_plus", np.array([1.0, 1.0, 1.0])),
    ("plus_plus_minus", np.array([1.0, 1.0, -1.0])),
    ("plus_minus_plus", np.array([1.0, -1.0, 1.0])),
    ("minus_plus_plus", np.array([-1.0, 1.0, 1.0])),
    ("plus_minus_minus", np.array([1.0, -1.0, -1.0])),
]
DIRECTIONS = [(name, v / np.linalg.norm(v)) for name, v in DIRECTIONS]


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
    eps = rng.rademacher(n)
    x = np.empty(n)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    return x


def features(n, rng, rho=0.35):
    raw = np.column_stack([ar_series(n, rng, rho) for _ in range(7)])

    def n01(x):
        return np.clip(0.5 + 0.18 * x, 0.0, 1.0)

    a = n01(raw[:, 0])
    b = n01(raw[:, 1])
    common = n01(raw[:, 2])
    alignment = n01(0.65 * raw[:, 3] + 0.35 * raw[:, 4])
    a2b = n01(raw[:, 5])
    b2a = n01(raw[:, 6])
    turn = n01(0.45 * raw[:, 4] + 0.55 * raw[:, 5])
    return np.column_stack([a, b, alignment, a2b, b2a, turn, common])


def covariance(n, rho, hetero, common):
    sigma = SIGMA * (1.0 + 0.7 * (common - 0.5)) if hetero else np.full(n, SIGMA)
    L = np.zeros((n, n))
    for t in range(n):
        L[t, t] = sigma[t]
        if t:
            L[t, :t] = (rho ** np.arange(t, 0, -1)) * sigma[:t]
    return L @ L.T, sigma


def operator(Xtr, Xte):
    mu = Xtr.mean(0)
    sc = np.where(Xtr.std(0) <= 1e-12, 1.0, Xtr.std(0))
    Atr = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sc])
    Ate = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sc])
    M = Atr.T @ Atr
    M[1:, 1:] += RIDGE * np.eye(Xtr.shape[1])
    return Ate @ np.linalg.solve(M, Atr.T)


def build_ops(F):
    nr = [0, 1, 2, 6]
    rel = [0, 1, 2, 3, 4, 5, 6]
    out = []
    for o in range(ORIGINS):
        s = o * STEP
        te = s + TRAIN + GAP
        out.append((
            operator(F[s:s + TRAIN, nr], F[te:te + TEST, nr]),
            operator(F[s:s + TRAIN, rel], F[te:te + TEST, rel]),
        ))
    return out


def base_mean(F):
    a, b, alignment, _a2b, _b2a, _turn, common = F.T
    mu0 = 0.15 + 0.45 * a + 0.25 * b + 0.20 * common + 0.30 * alignment
    return mu0, F[:, 3:6]


def quadratic(F, ops, rho, hetero):
    mu0, Z = base_mean(F)
    Sigma, _ = covariance(len(F), rho, hetero, F[:, -1])
    Q = np.zeros((3, 3))
    b = np.zeros(3)
    c = 0.0

    for o, (Wc, Wr) in enumerate(ops):
        s = o * STEP
        te = s + TRAIN + GAP
        mtr = mu0[s:s + TRAIN]
        mte = mu0[te:te + TEST]
        Ztr = Z[s:s + TRAIN]
        Zte = Z[te:te + TEST]
        qc = Wc @ mtr - mte
        qr = Wr @ mtr - mte
        Hc = Wc @ Ztr - Zte
        Hr = Wr @ Ztr - Zte
        S_tt = Sigma[s:s + TRAIN, s:s + TRAIN]
        S_ty = Sigma[s:s + TRAIN, te:te + TEST]

        def stochastic(W):
            return np.trace(W @ S_tt @ W.T) - 2.0 * np.trace(W @ S_ty)

        Q += (Hc.T @ Hc - Hr.T @ Hr) / TEST
        b += 2.0 * (Hc.T @ qc - Hr.T @ qr) / TEST
        c += (qc @ qc - qr @ qr + stochastic(Wc) - stochastic(Wr)) / TEST

    return Q / ORIGINS, b / ORIGINS, c / ORIGINS


def root(Q, b, c, v):
    A = float(v @ Q @ v)
    B = float(b @ v)
    rr = np.roots([A, B, c]) if abs(A) > 1e-14 else np.roots([B, c])
    positive = [float(x.real) for x in rr if abs(x.imag) < 1e-9 and x.real > 0]
    return min(positive) if positive else None


def studentized(D):
    m = D.mean()
    lrv = np.mean((D - m) ** 2)
    for k in range(1, LAG + 1):
        lrv += 2.0 * (1.0 - k / (LAG + 1)) * np.mean((D[k:] - m) * (D[:-k] - m))
    return 0.0 if lrv <= 1e-14 else math.sqrt(len(D)) * m / math.sqrt(lrv)


def bootstrap_p(F, y, mu, ops, rng, rho, hetero, B):
    Dobs = []
    for o, (Wc, Wr) in enumerate(ops):
        s = o * STEP
        te = s + TRAIN + GAP
        pn = Wc @ y[s:s + TRAIN]
        pr = Wr @ y[s:s + TRAIN]
        yt = y[te:te + TEST]
        Dobs.append(np.mean((pn - yt) ** 2 - (pr - yt) ** 2))
    Tobs = studentized(np.asarray(Dobs))

    _S, sigma = covariance(len(y), rho, hetero, F[:, -1])
    signs = rng.rademacher(B * len(y)).reshape(B, len(y))
    E = np.empty_like(signs)
    E[:, 0] = sigma[0] * signs[:, 0]
    for t in range(1, len(y)):
        E[:, t] = rho * E[:, t - 1] + sigma[t] * signs[:, t]
    Y = mu[None, :] + E

    DT = np.empty((B, ORIGINS))
    for o, (Wc, Wr) in enumerate(ops):
        s = o * STEP
        te = s + TRAIN + GAP
        pn = Y[:, s:s + TRAIN] @ Wc.T
        pr = Y[:, s:s + TRAIN] @ Wr.T
        yt = Y[:, te:te + TEST]
        DT[:, o] = np.mean((pn - yt) ** 2 - (pr - yt) ** 2, axis=1)
    Tb = np.apply_along_axis(studentized, 1, DT)
    p = (1 + int(np.count_nonzero(Tb >= Tobs))) / (B + 1)
    return p, Tobs, float(np.mean(Dobs))


def run_scenario(rho, hetero, args, seed):
    rng = SplitMix64(seed)
    rows = []
    for pidx in range(args.paths):
        F = features(N, rng)
        ops = build_ops(F)
        Q, b, c = quadratic(F, ops, rho, hetero)
        mu0, Z = base_mean(F)

        for name, v in DIRECTIONS:
            k = root(Q, b, c, v)
            if k is None:
                rows.append({"path": pidx, "direction": name, "root": None})
                continue
            gamma = k * v
            mu = mu0 + Z @ gamma
            ps = []
            for _ in range(args.outcomes_per_point):
                _S, sigma = covariance(N, rho, hetero, F[:, -1])
                signs = rng.rademacher(N)
                e = np.empty(N)
                e[0] = sigma[0] * signs[0]
                for t in range(1, N):
                    e[t] = rho * e[t - 1] + sigma[t] * signs[t]
                y = mu + e
                p, _t, _d = bootstrap_p(F, y, mu, ops, rng, rho, hetero, args.bootstrap)
                ps.append(p)
            rows.append({
                "path": pidx,
                "direction": name,
                "root": k,
                "reject_rate": float(np.mean(np.asarray(ps) < ALPHA)),
                "mean_p": float(np.mean(ps)),
                "outcomes": len(ps),
            })

    by_direction = {}
    for name, _ in DIRECTIONS:
        x = [r for r in rows if r["direction"] == name and r.get("root") is not None]
        by_direction[name] = {
            "points": len(x),
            "total_outcomes": int(sum(r["outcomes"] for r in x)),
            "aggregate_rejection_rate": float(np.average(
                [r["reject_rate"] for r in x], weights=[r["outcomes"] for r in x]
            )),
            "mean_p": float(np.mean([r["mean_p"] for r in x])),
            "root_mean": float(np.mean([r["root"] for r in x])),
            "root_min": float(np.min([r["root"] for r in x])),
            "root_max": float(np.max([r["root"] for r in x])),
        }

    overall = [r for r in rows if r.get("root") is not None]
    return {
        "rho": rho,
        "heteroskedastic": hetero,
        "paths": args.paths,
        "directions": [name for name, _ in DIRECTIONS],
        "by_direction": by_direction,
        "overall": {
            "total_null_points": len(overall),
            "total_outcomes": int(sum(r["outcomes"] for r in overall)),
            "rejection_rate": float(np.average(
                [r["reject_rate"] for r in overall], weights=[r["outcomes"] for r in overall]
            )),
            "direction_rejection_min": float(min(x["aggregate_rejection_rate"] for x in by_direction.values())),
            "direction_rejection_max": float(max(x["aggregate_rejection_rate"] for x in by_direction.values())),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=20)
    ap.add_argument("--outcomes-per-point", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=199)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()

    results = [
        run_scenario(0.0, False, args, args.seed),
        run_scenario(0.5, False, args, args.seed + 1000),
        run_scenario(0.8, False, args, args.seed + 2000),
        run_scenario(0.5, True, args, args.seed + 3000),
    ]
    out = {
        "schema": "rh006-risk-null-surface-coverage-executed/v1",
        "status": "research-diagnostic-only",
        "geometry": {
            "train": TRAIN, "test": TEST, "gap": GAP, "origins": ORIGINS,
            "step": STEP, "ridge": RIDGE, "bartlett_lag": LAG, "alpha": ALPHA
        },
        "design": {
            "paths": args.paths,
            "outcomes_per_point": args.outcomes_per_point,
            "bootstrap_reps": args.bootstrap,
            "fixed_directions": [name for name, _ in DIRECTIONS]
        },
        "interpretation": {
            "purpose": "Compare oracle risk-null bootstrap behavior across multiple predeclared nuisance points on the same finite-sample equal-risk surface.",
            "formal_inference": False,
            "applicability": "not-approved-for-execution",
            "selection": "stop-assumption-failure",
            "nuisance_point_policy": "unresolved",
            "endogeneity_gate": True
        },
        "scenarios": results,
    }
    raw = json.dumps(out, sort_keys=True, separators=(",", ":"))
    out["payload_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
