#!/usr/bin/env python3
"""RH-006 joint feature/outcome endogeneity stress.

Research-only mechanics mirror. The marginal outcome process remains AR(0.5)
with the declared heteroskedastic scale. Endogeneity is introduced by sharing
an innovation shock between one relational feature channel and the outcome
innovation. The nominal finite-sample equal-risk root is still constructed
under the independence covariance formula, allowing the null-location error
to be measured.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

TRAIN, GAP, TEST, ORIGINS, STEP = 48, 4, 16, 24, 16
N = TRAIN + GAP + TEST + (ORIGINS - 1) * STEP
RIDGE = 1e-4
ALPHA = 0.05
DIRECTIONS = {
    "weak": np.array([0., 1., -2.]) / np.sqrt(5.),
    "strong": np.array([1., 1., .5]) / np.sqrt(2.25),
}

class SM:
    def __init__(self, x):
        self.x = np.uint64(int(x) & 0xffffffffffffffff)
    def r(self, n):
        n = int(n)
        i = self.x + np.arange(n, dtype=np.uint64)
        z = i + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
        self.x = np.uint64((int(self.x) + n) & 0xffffffffffffffff)
        return np.where((z & np.uint64(1)) == 0, -1., 1.)

def ar_from_e(e, rho):
    x = np.empty(N)
    x[0] = e[0]
    for i in range(1, N):
        x[i] = rho * x[i - 1] + e[i]
    return x

def feature_path(g, common_channel=5):
    common = g.r(N)
    raw = np.column_stack([
        ar_from_e(common if j == common_channel else g.r(N), .35)
        for j in range(7)
    ])
    f = lambda x: np.clip(.5 + .18 * x, 0, 1)
    a, b, c = f(raw[:, 0]), f(raw[:, 1]), f(raw[:, 2])
    al = f(.65 * raw[:, 3] + .35 * raw[:, 4])
    a2b, b2a = f(raw[:, 5]), f(raw[:, 6])
    turn = f(.45 * raw[:, 4] + .55 * raw[:, 5])
    F = np.column_stack([a, b, al, a2b, b2a, turn, c])
    return F, common

def mu0(F):
    return .15 + .45 * F[:, 0] + .25 * F[:, 1] + .20 * F[:, 6] + .30 * F[:, 2]

def sig0(F):
    return .08 * (1 + .7 * (F[:, -1] - .5))

def op(X, Y, ridge):
    m = X.mean(0)
    sc = np.where(X.std(0) <= 1e-12, 1., X.std(0))
    A = np.c_[np.ones(len(X)), (X - m) / sc]
    B = np.c_[np.ones(len(Y)), (Y - m) / sc]
    G = A.T @ A
    G[1:, 1:] += ridge * np.eye(X.shape[1])
    return B @ np.linalg.solve(G, A.T)

def build_ops(F):
    nr, rel, out = [0,1,2,6], list(range(7)), []
    for o in range(ORIGINS):
        s, te = o * STEP, o * STEP + TRAIN + GAP
        out.append((
            op(F[s:s+TRAIN, nr], F[te:te+TEST, nr], RIDGE),
            op(F[s:s+TRAIN, rel], F[te:te+TEST, rel], RIDGE),
        ))
    return out

def covariance(F):
    sig = sig0(F)
    L = np.zeros((N, N))
    for t in range(N):
        L[t, t] = sig[t]
        if t:
            L[t, :t] = (.5 ** np.arange(t, 0, -1)) * sig[:t]
    return L @ L.T

def quadratic_independence_null(F, ops):
    S = covariance(F)
    m = mu0(F)
    Z = F[:, 3:6]
    Q = np.zeros((3, 3)); b = np.zeros(3); c = 0.
    for o, (Wc, Wr) in enumerate(ops):
        s, te = o * STEP, o * STEP + TRAIN + GAP
        tr, tt = slice(s, s+TRAIN), slice(te, te+TEST)
        qc, qr = Wc @ m[tr] - m[tt], Wr @ m[tr] - m[tt]
        Hc, Hr = Wc @ Z[tr] - Z[tt], Wr @ Z[tr] - Z[tt]
        St, Sty = S[s:s+TRAIN, s:s+TRAIN], S[s:s+TRAIN, te:te+TEST]
        st = lambda W: np.trace(W @ St @ W.T) - 2 * np.trace(W @ Sty)
        Q += (Hc.T @ Hc - Hr.T @ Hr) / TEST
        b += 2 * (Hc.T @ qc - Hr.T @ qr) / TEST
        c += (qc @ qc - qr @ qr + st(Wc) - st(Wr)) / TEST
    return Q / ORIGINS, b / ORIGINS, c / ORIGINS

def positive_root(Q, b, c, v):
    A, B = float(v @ Q @ v), float(b @ v)
    if abs(A) <= 1e-14:
        t = -c / B if abs(B) > 1e-14 else np.nan
        return float(t) if np.isfinite(t) and t > 0 else None
    D = B * B - 4 * A * c
    if D < 0:
        return None
    rr = [(-B - np.sqrt(max(D, 0))) / (2 * A), (-B + np.sqrt(max(D, 0))) / (2 * A)]
    rr = [x for x in rr if np.isfinite(x) and x > 0]
    return min(rr) if rr else None

def outcome(g, F, common, mu, omega):
    z = g.r(N)
    eta = np.sqrt(max(0., 1. - omega * omega)) * z + omega * common
    sig = sig0(F)
    e = np.empty(N)
    e[0] = sig[0] * eta[0]
    for t in range(1, N):
        e[t] = .5 * e[t-1] + sig[t] * eta[t]
    return mu + e

def loss_diff(y, F, ops):
    vals = []
    for o, (Wc, Wr) in enumerate(ops):
        s, te = o * STEP, o * STEP + TRAIN + GAP
        pc = Wc @ y[s:s+TRAIN]
        pr = Wr @ y[s:s+TRAIN]
        yt = y[te:te+TEST]
        vals.append(np.mean((pc-yt)**2 - (pr-yt)**2))
    return float(np.mean(vals))

def run(args):
    result = {}
    for dname, v in DIRECTIONS.items():
        for omega in args.omegas:
            rng = SM(args.seed + int(round(omega*1000)) + (100000 if dname=="strong" else 0))
            zero, nominal = [], []
            roots = 0
            for _ in range(args.paths):
                F, common = feature_path(rng)
                ops = build_ops(F)
                m = mu0(F)
                Q, b, c = quadratic_independence_null(F, ops)
                t = positive_root(Q, b, c, v)
                if t is None:
                    continue
                roots += 1
                mu_zero = m
                mu_null = m + t * (F[:, 3:6] @ v)
                zero_vals = [
                    loss_diff(outcome(rng, F, common, mu_zero, omega), F, ops)
                    for _ in range(args.outcomes)
                ]
                null_vals = [
                    loss_diff(outcome(rng, F, common, mu_null, omega), F, ops)
                    for _ in range(args.outcomes)
                ]
                zero.append(float(np.mean(zero_vals)))
                nominal.append(float(np.mean(null_vals)))
            result[f"{dname}_{omega:g}"] = {
                "omega": omega,
                "paths": args.paths,
                "outcomes_per_path": args.outcomes,
                "nominal_root_support_rate": roots / max(args.paths, 1),
                "mean_loss_difference_at_gamma0": float(np.mean(zero)) if zero else None,
                "median_loss_difference_at_gamma0": float(np.median(zero)) if zero else None,
                "mean_loss_difference_at_nominal_independence_root": float(np.mean(nominal)) if nominal else None,
                "median_loss_difference_at_nominal_independence_root": float(np.median(nominal)) if nominal else None,
            }
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=40)
    ap.add_argument("--outcomes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261080)
    ap.add_argument("--omegas", type=float, nargs="+", default=[0., .25, .5, .75, .9])
    a = ap.parse_args()
    out = {
        "schema": "rh006-joint-feature-outcome-endogeneity-executed/v1",
        "status": "research-diagnostic-only",
        "formal_inference": False,
        "applicability": "not-approved-for-execution",
        "selection": "stop-assumption-failure",
        "mechanics_scope": "independent Python estimator-mechanics mirror; not the Symthaea Rust implementation",
        "marginal_outcome_control": "AR(0.5) with the declared heteroskedastic scale is preserved; only feature/outcome innovation correlation is varied",
        "endogeneity_design": "the innovation shock driving relational feature channel raw[5] is shared with the outcome innovation with correlation parameter omega",
        "null_construction": "finite-sample equal-risk quadratic uses the existing independence covariance formula, so joint-DGP displacement measures the uncovered feature/outcome-dependence assumption",
        "results": run(a),
        "nonclaims": [
            "No formal p-value or confidence interval.",
            "No uniform size guarantee.",
            "No empirical validation.",
            "No claim that joint endogeneity invalidates all RH-006 procedures.",
            "No exact-Rust implementation qualification."
        ],
    }
    raw = json.dumps(out, sort_keys=True, separators=(",", ":"))
    out["payload_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    print(json.dumps(out, indent=2, sort_keys=True))
if __name__ == "__main__":
    main()
