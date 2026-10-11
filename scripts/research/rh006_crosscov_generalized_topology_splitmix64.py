#!/usr/bin/env python3
"""RH-006 cross-covariance generalization + exact quadratic-topology checker.

Research-only diagnostic. The executed family uses latent feature-driver
processes and outcome innovations of the form

    eta = sqrt(1-omega^2) z + omega q,

where q is a normalized linear combination of feature-driver channels and
lags. This is a finite-rank family of feature/outcome cross-covariance
operators. No formal inference is implemented.
"""
import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path
import numpy as np

TRAIN, GAP, TEST, ORIGINS, STEP = 48, 4, 16, 24, 16
N = TRAIN + GAP + TEST + (ORIGINS - 1) * STEP
RIDGE = 1e-4
OMEGAS = (0.0, 0.25, 0.5, 0.75, 0.9)
DIRECTIONS = {
    "weak": np.array([0.0, 1.0, -2.0]) / np.sqrt(5.0),
    "strong": np.array([1.0, 1.0, 0.5]) / np.sqrt(2.25),
}
KINDS = ("single_rel", "multi_rel", "mixed_common_rel", "anti_rel", "lag_rel")

class SplitMix64:
    def __init__(self, seed):
        self.state = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    def r(self, n):
        n = int(n)
        idx = self.state + np.arange(n, dtype=np.uint64)
        z = idx + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
        self.state = np.uint64((int(self.state) + n) & 0xFFFFFFFFFFFFFFFF)
        return np.where((z & np.uint64(1)) == 0, -1.0, 1.0)

def ar_from_e(e, rho):
    x = np.empty(N)
    x[0] = e[0]
    for i in range(1, N):
        x[i] = rho * x[i - 1] + e[i]
    return x

def feature_path(g):
    common = g.r(N)
    raw = np.column_stack([
        ar_from_e(common if j == 5 else g.r(N), 0.35)
        for j in range(7)
    ])
    f = lambda x: np.clip(0.5 + 0.18 * x, 0.0, 1.0)
    a, b, c = f(raw[:, 0]), f(raw[:, 1]), f(raw[:, 2])
    alignment = f(0.65 * raw[:, 3] + 0.35 * raw[:, 4])
    a2b, b2a = f(raw[:, 5]), f(raw[:, 6])
    turn = f(0.45 * raw[:, 4] + 0.55 * raw[:, 5])
    F = np.column_stack([a, b, alignment, a2b, b2a, turn, c])
    return F, raw

def normalize(q):
    q = np.asarray(q, dtype=float)
    q = q - np.mean(q)
    sd = float(np.std(q))
    if sd <= 1e-14:
        raise ValueError("degenerate cross-covariance driver")
    return q / sd

def cross_driver(raw, kind):
    if kind == "single_rel":
        q = raw[:, 5]
    elif kind == "multi_rel":
        q = (raw[:, 5] + raw[:, 6] + raw[:, 4]) / np.sqrt(3.0)
    elif kind == "mixed_common_rel":
        q = (raw[:, 2] + raw[:, 5]) / np.sqrt(2.0)
    elif kind == "anti_rel":
        q = (raw[:, 5] - raw[:, 6]) / np.sqrt(2.0)
    elif kind == "lag_rel":
        q = np.zeros(N)
        q[1:] = raw[:-1, 5]
    else:
        raise ValueError(kind)
    return normalize(q)

def mu0(F):
    return 0.15 + 0.45 * F[:, 0] + 0.25 * F[:, 1] + 0.20 * F[:, 6] + 0.30 * F[:, 2]

def sig0(F):
    return 0.08 * (1.0 + 0.7 * (F[:, -1] - 0.5))

def op(Xtr, Xte):
    m = Xtr.mean(0)
    sc = np.where(Xtr.std(0) <= 1e-12, 1.0, Xtr.std(0))
    A = np.c_[np.ones(len(Xtr)), (Xtr - m) / sc]
    B = np.c_[np.ones(len(Xte)), (Xte - m) / sc]
    G = A.T @ A
    G[1:, 1:] += RIDGE * np.eye(Xtr.shape[1])
    return B @ np.linalg.solve(G, A.T)

def build_ops(F):
    nr, rel = [0, 1, 2, 6], list(range(7))
    out = []
    for o in range(ORIGINS):
        s, te = o * STEP, o * STEP + TRAIN + GAP
        out.append((
            op(F[s:s+TRAIN, nr], F[te:te+TEST, nr]),
            op(F[s:s+TRAIN, rel], F[te:te+TEST, rel]),
        ))
    return out

def loading(sig):
    L = np.zeros((N, N))
    for t in range(N):
        L[t, t] = sig[t]
        if t:
            L[t, :t] = (0.5 ** np.arange(t, 0, -1)) * sig[:t]
    return L

def surface(F, q):
    m, sig = mu0(F), sig0(F)
    L = loading(sig)
    ell = L @ q
    S = L @ L.T
    Z = F[:, 3:6]
    ops = build_ops(F)
    Q = np.zeros((3, 3))
    b0 = np.zeros(3)
    b1 = np.zeros(3)
    c0 = c1 = c2 = 0.0
    for o, (Wc, Wr) in enumerate(ops):
        s, te = o * STEP, o * STEP + TRAIN + GAP
        tr = np.arange(s, s + TRAIN)
        tt = np.arange(te, te + TEST)
        Hc, Hr = Wc @ Z[tr] - Z[tt], Wr @ Z[tr] - Z[tt]
        qc, qr = Wc @ m[tr] - m[tt], Wr @ m[tr] - m[tt]
        ac, ar = Wc @ ell[tr] - ell[tt], Wr @ ell[tr] - ell[tt]
        Q += (Hc.T @ Hc - Hr.T @ Hr) / TEST
        b0 += 2.0 * (Hc.T @ qc - Hr.T @ qr) / TEST
        b1 += 2.0 * (Hc.T @ ac - Hr.T @ ar) / TEST
        c0 += (qc @ qc - qr @ qr) / TEST
        c1 += 2.0 * (qc @ ac - qr @ ar) / TEST
        c2 += (ac @ ac - ar @ ar) / TEST
        St = S[np.ix_(tr, tr)]
        Sty = S[np.ix_(tr, tt)]
        vd = (
            np.trace(Wc @ St @ Wc.T) - 2.0 * np.trace(Wc @ Sty)
            - np.trace(Wr @ St @ Wr.T) + 2.0 * np.trace(Wr @ Sty)
        ) / TEST
        c0 += vd
        c2 -= vd
    z = 1.0 / ORIGINS
    return Q * z, b0 * z, b1 * z, c0 * z, c1 * z, c2 * z, ops

def coeff(poly, omega):
    Q, b0, b1, c0, c1, c2, _ = poly
    return Q, b0 + omega * b1, c0 + omega * c1 + omega * omega * c2

def delta(poly, omega, gamma):
    Q, b, c = coeff(poly, omega)
    return float(gamma @ Q @ gamma + b @ gamma + c)

def roots(Q, b, c, u):
    A, B = float(u @ Q @ u), float(b @ u)
    D = B * B - 4.0 * A * c
    if D < 0.0:
        return []
    if abs(A) <= 1e-14:
        t = -c / B if abs(B) > 1e-14 else np.nan
        return [float(t)] if np.isfinite(t) and t > 0.0 else []
    s = math.sqrt(max(D, 0.0))
    rr = [(-B - s) / (2.0 * A), (-B + s) / (2.0 * A)]
    return sorted(float(t) for t in rr if np.isfinite(t) and t > 0.0)

def matrix_signature(R, tol=1e-12):
    ev = np.linalg.eigvalsh((R + R.T) / 2.0)
    return tuple(int(1 if x > tol else -1 if x < -tol else 0) for x in ev), ev

def theorem_case(Q, b, c):
    Qs = (Q + Q.T) / 2.0
    ew = np.linalg.eigvalsh(Qs)
    if np.min(ew) <= 1e-10:
        return "non-PD-Q", ew, None
    if c < -1e-10:
        return "all-directions-one-positive-root", ew, None
    qinv = float(b @ np.linalg.solve(Qs, b))
    if abs(c) <= 1e-10:
        return "c-zero-boundary", ew, qinv - 4.0 * c
    margin = qinv - 4.0 * c
    return ("cone-support" if margin > 1e-10 else "no-discriminant-direction"), ew, margin

def outcome(g, F, q, mu, omega):
    z = g.r(N)
    eta = math.sqrt(max(0.0, 1.0 - omega * omega)) * z + omega * q
    sig = sig0(F)
    e = np.empty(N)
    e[0] = sig[0] * eta[0]
    for t in range(1, N):
        e[t] = 0.5 * e[t-1] + sig[t] * eta[t]
    return mu + e

def loss_diff(y, F, ops):
    vals = []
    for o, (Wc, Wr) in enumerate(ops):
        s, te = o * STEP, o * STEP + TRAIN + GAP
        pc, pr = Wc @ y[s:s+TRAIN], Wr @ y[s:s+TRAIN]
        yt = y[te:te+TEST]
        vals.append(np.mean((pc - yt)**2 - (pr - yt)**2))
    return float(np.mean(vals))

def run(args):
    rng = SplitMix64(args.seed)
    summary = {}
    identity_checks = []
    for kind in KINDS:
        topo = {str(w): Counter() for w in OMEGAS}
        ray_support = {str(w): {d: 0 for d in DIRECTIONS} for w in OMEGAS}
        margins = {str(w): [] for w in OMEGAS}
        cvals = {str(w): [] for w in OMEGAS}
        qmins = {str(w): [] for w in OMEGAS}
        spectrum_match = {str(w): 0 for w in OMEGAS}
        topology_checks = {str(w): 0 for w in OMEGAS}
        for path in range(args.paths):
            F, raw = feature_path(rng)
            q = cross_driver(raw, kind)
            poly = surface(F, q)
            Q0, b00, b10, c00, c10, c20, ops = poly
            for omega in OMEGAS:
                Q, b, c = coeff(poly, omega)
                case, ew, margin = theorem_case(Q, b, c)
                R = np.outer(b, b) - 4.0 * c * ((Q + Q.T) / 2.0)
                actual_sig, _ = matrix_signature(R)
                expected_sig = (
                    (1, 1, 1) if case == "all-directions-one-positive-root"
                    else (-1, -1, 1) if case == "cone-support"
                    else (0, 0, 1) if case == "c-zero-boundary"
                    else (-1, -1, -1) if case == "no-discriminant-direction"
                    else None
                )
                topology_checks[str(omega)] += 1
                spectrum_match[str(omega)] += int(expected_sig is None or actual_sig == expected_sig)
                qmins[str(omega)].append(float(np.min(ew)))
                cvals[str(omega)].append(float(c))
                if margin is not None:
                    margins[str(omega)].append(float(margin))
                if case == "non-PD-Q":
                    sig = "non-PD"
                elif case == "all-directions-one-positive-root":
                    sig = "all+"
                elif case == "c-zero-boundary":
                    sig = "boundary"
                elif case == "cone-support":
                    sig = "--+"
                else:
                    sig = "---"
                topo[str(omega)][sig] += 1
                for d, u in DIRECTIONS.items():
                    if roots(Q, b, c, u):
                        ray_support[str(omega)][d] += 1
            if path < args.mc_paths:
                for d, u in DIRECTIONS.items():
                    r0 = roots(Q0, b00, c00, u)
                    if not r0:
                        continue
                    mu = mu0(F) + r0[0] * (F[:, 3:6] @ u)
                    for omega in (0.0, 0.5, 0.9):
                        analytic = delta(poly, omega, r0[0] * u)
                        gr = SplitMix64(
                            args.seed + 900000 + path * 1000
                            + int(round(1000 * omega))
                            + (1 if d == "strong" else 0)
                        )
                        sims = [
                            loss_diff(outcome(gr, F, q, mu, omega), F, ops)
                            for _ in range(args.mc_outcomes)
                        ]
                        emp = float(np.mean(sims))
                        identity_checks.append(abs(emp - analytic))
        summary[kind] = {
            "topology_signature_counts": {w: dict(c) for w, c in topo.items()},
            "theorem_vs_actual_spectrum_match_rate": {
                w: spectrum_match[w] / max(topology_checks[w], 1)
                for w in topology_checks
            },
            "weak_ray_support_rate": {
                w: ray_support[w]["weak"] / args.paths for w in ray_support
            },
            "strong_ray_support_rate": {
                w: ray_support[w]["strong"] / args.paths for w in ray_support
            },
            "q_min_eigenvalue_quantiles": {
                w: np.quantile(v, [0, 0.5, 1]).tolist()
                for w, v in qmins.items()
            },
            "c_quantiles": {
                w: np.quantile(v, [0, 0.5, 1]).tolist()
                for w, v in cvals.items()
            },
            "qinv_minus_4c_quantiles": {
                w: np.quantile(v, [0, 0.5, 1]).tolist() if v else []
                for w, v in margins.items()
            },
        }
    return {
        "schema": "rh006-crosscov-generalized-topology/v1",
        "status": "research-diagnostic-only",
        "formal_inference": False,
        "applicability": "not-approved-for-execution",
        "selection": "stop-assumption-failure",
        "execution": {
            "paths": args.paths,
            "seed": args.seed,
            "omegas": list(OMEGAS),
            "cross_covariance_families": list(KINDS),
            "geometry": {"train": TRAIN, "gap": GAP, "test": TEST, "origins": ORIGINS, "step": STEP},
            "ridge": RIDGE,
            "mc_identity_checks": len(identity_checks),
        },
        "general_identity": {
            "conditional_mean": "E[e|q]=omega Lq for the executed rank-one cross-covariance family",
            "conditional_covariance": "Var(e|q)=(1-omega^2)LL'",
            "surface": "Delta(gamma;omega)=gamma'Qgamma+(b0+omega*b1)'gamma+c0+omega*c1+omega^2*c2",
            "cross_covariance_scope": "finite-rank linear feature-driver/lags represented by normalized q"
        },
        "exact_topology_theorem": {
            "assumption": "Q is positive definite",
            "c_lt_0": "D(u)>0 for every nonzero u; each ray has exactly one positive root",
            "c_eq_0": "D(u)=(b'u)^2; the nonzero positive root exists exactly on the b'u<0 hemisphere",
            "c_gt_0": "R=bb'-4cQ has at most one positive eigenvalue; global real-root support exists iff b'Q^{-1}b>4c, and supported positive roots lie in the discriminant cone with b'u<0",
            "weak_Q_case": "the theorem is not applied; retain eigenstructure/stratified analysis"
        },
        "results": summary,
        "identity_check": {
            "max_abs_error": max(identity_checks) if identity_checks else 0.0,
            "mean_abs_error": float(np.mean(identity_checks)) if identity_checks else 0.0,
            "checks": len(identity_checks),
        },
        "nonclaims": [
            "No formal p-value or confidence interval.",
            "No uniform or least-favorable size guarantee.",
            "No empirical validation.",
            "No claim this finite-rank family spans every possible endogenous DGP.",
            "No claim the topology transition is universal when Q is not positive definite.",
            "No exact-Rust implementation qualification."
        ],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261080)
    ap.add_argument("--mc-paths", type=int, default=3)
    ap.add_argument("--mc-outcomes", type=int, default=250)
    args = ap.parse_args()
    out = run(args)
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    raw = json.dumps(out, sort_keys=True, separators=(",", ":"))
    out["payload_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    print(json.dumps(out, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
