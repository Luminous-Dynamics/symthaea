#!/usr/bin/env python3
"""RH-006 conditioning × ridge × estimand-policy × size/power diagnostic.

Independent estimator-mechanics reimplementation of the current RH-006
rolling estimator. Research diagnostic only; not the Symthaea Rust
implementation and not a formal inference procedure.
"""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

TRAIN, TEST, GAP, ORIGINS, STEP = 48, 16, 4, 24, 16
N = TRAIN + GAP + TEST + (ORIGINS - 1) * STEP
SIGMA = 0.08
RHO_ERR = 0.5
HETERO = True
BARTLETT_LAG = 3
ALPHA_CUTOFF = 1.645
LOCAL_SIGNAL_RMS = 0.25 * SIGMA
CONDITION_THRESHOLD = 1e-3
RIDGE_POLICIES = (1e-8, 1e-4)
SUBSPACE_THRESHOLD = 1e-3
EPSILONS = (1e-2, 1e-3, 1e-4, 0.0)
DIRECTIONS = {
    "strong": np.array([1.0, 1.0, 0.5]) / np.linalg.norm([1.0, 1.0, 0.5]),
    "weak": np.array([0.0, 1.0, -2.0]) / np.linalg.norm([0.0, 1.0, -2.0]),
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


def generate_features(rng, epsilon):
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


def outcome_sigma(F):
    common = F[:, -1]
    return SIGMA * (1.0 + 0.7 * (common - 0.5)) if HETERO else np.full(N, SIGMA)


def mean_components(F):
    a, b, alignment = F[:, 0], F[:, 1], F[:, 2]
    common = F[:, 6]
    return 0.15 + 0.45 * a + 0.25 * b + 0.20 * common + 0.30 * alignment


def standardized_operator(Xtr, Xte, ridge):
    mu = Xtr.mean(axis=0)
    scale = np.where(Xtr.std(axis=0) <= 1e-12, 1.0, Xtr.std(axis=0))
    Atr = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / scale])
    Ate = np.column_stack([np.ones(len(Xte)), (Xte - mu) / scale])
    G = Atr.T @ Atr
    if ridge:
        G[1:, 1:] += ridge * np.eye(Xtr.shape[1])
    return Ate @ np.linalg.solve(G, Atr.T)


def build_policy(F, kind, parameter):
    nr = [0, 1, 2, 6]
    origin = []
    mu0 = mean_components(F)
    Z = F[:, 3:6]
    sig = outcome_sigma(F)
    var = np.empty(N)
    var[0] = sig[0] ** 2
    for i in range(1, N):
        var[i] = RHO_ERR * RHO_ERR * var[i - 1] + sig[i] * sig[i]

    for o in range(ORIGINS):
        s = o * STEP
        te = s + TRAIN + GAP
        Ctr, Cte = F[s:s + TRAIN, nr], F[te:te + TEST, nr]
        Rtr, Rte = Z[s:s + TRAIN], Z[te:te + TEST]
        m = Rtr.mean(axis=0)
        sc = np.where(Rtr.std(axis=0) <= 1e-12, 1.0, Rtr.std(axis=0))
        Rstd = (Rtr - m) / sc
        _, S, Vt = np.linalg.svd(Rstd, full_matrices=False)
        ratio = float(S[-1] / S[0]) if S[0] > 0 else 0.0

        if kind == "hard":
            if ratio < parameter:
                return None
            Rfit_tr, Rfit_te, ridge = Rtr, Rte, 0.0
        elif kind == "ridge":
            Rfit_tr, Rfit_te, ridge = Rtr, Rte, parameter
        elif kind == "subspace":
            keep = max(1, int(np.sum(S / S[0] >= parameter))) if S[0] > 0 else 1
            V = Vt[:keep].T
            Rfit_tr = Rstd @ V
            Rfit_te = ((Rte - m) / sc) @ V
            ridge = 1e-8
        else:
            raise ValueError(kind)

        Xtr = np.column_stack([Ctr, Rfit_tr])
        Xte = np.column_stack([Cte, Rfit_te])
        W = standardized_operator(Xtr, Xte, ridge)
        Wc = standardized_operator(Ctr, Cte, ridge if kind != "hard" else 0.0)

        tr = np.arange(s, s + TRAIN)
        tt = np.arange(te, te + TEST)
        lo = np.minimum(tr[:, None], tr[None, :])
        hi = np.maximum(tr[:, None], tr[None, :])
        St = (RHO_ERR ** (hi - lo)) * var[lo]
        lo2 = np.minimum(tr[:, None], tt[None, :])
        hi2 = np.maximum(tr[:, None], tt[None, :])
        Sty = (RHO_ERR ** (hi2 - lo2)) * var[lo2]
        var_diff = (
            np.trace(Wc @ St @ Wc.T) - 2.0 * np.trace(Wc @ Sty)
            - np.trace(W @ St @ W.T) + 2.0 * np.trace(W @ Sty)
        )
        qc = Wc @ mu0[tr] - mu0[tt]
        qr = W @ mu0[tr] - mu0[tt]
        Hc = Wc @ Z[tr] - Z[tt]
        Hr = W @ Z[tr] - Z[tt]
        origin.append((Wc, W, ratio, qc, qr, Hc, Hr, var_diff))

    return origin


def root_for_direction(origins, v):
    q = b = 0.0
    c = 0.0
    for _, _, _, qc, qr, Hc, Hr, var_diff in origins:
        q += float(v @ (Hc.T @ Hc - Hr.T @ Hr) @ v) / TEST
        b += 2.0 * float(v @ (Hc.T @ qc - Hr.T @ qr)) / TEST
        c += float(qc @ qc - qr @ qr + var_diff) / TEST
    q /= ORIGINS
    b /= ORIGINS
    c /= ORIGINS
    if not all(np.isfinite([q, b, c])):
        return None
    roots = np.roots([q, b, c] if abs(q) > 1e-14 else [b, c])
    positive = [float(r.real) for r in roots if abs(r.imag) < 1e-8 and r.real > 0 and np.isfinite(r.real)]
    return min(positive) if positive else None


def generate_outcome(F, rng, kappa, v):
    mu = mean_components(F) + kappa * (F[:, 3:6] @ v)
    sig = outcome_sigma(F)
    eps = rng.rademacher(N)
    e = np.empty(N)
    e[0] = sig[0] * eps[0]
    for i in range(1, N):
        e[i] = RHO_ERR * e[i - 1] + sig[i] * eps[i]
    return mu + e


def studentized(T):
    T = np.asarray(T, dtype=float)
    m = T.mean()
    lrv = np.mean((T - m) ** 2)
    for k in range(1, BARTLETT_LAG + 1):
        lrv += 2.0 * (1.0 - k / (BARTLETT_LAG + 1)) * np.mean((T[k:] - m) * (T[:-k] - m))
    return math.sqrt(len(T)) * m / math.sqrt(lrv) if lrv > 1e-14 else 0.0


def statistic(F, y, origins):
    D = []
    for o, (Wc, W, *_rest) in enumerate(origins):
        s = o * STEP
        te = s + TRAIN + GAP
        D.append(np.mean((Wc @ y[s:s + TRAIN] - y[te:te + TEST]) ** 2)
                 - np.mean((W @ y[s:s + TRAIN] - y[te:te + TEST]) ** 2))
    return studentized(np.asarray(D))


def signal_sd(F, v):
    z = F[:, 3:6] @ v
    z = z - z.mean()
    return float(np.sqrt(np.mean(z * z)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--cutoff", type=float, default=ALPHA_CUTOFF)
    args = ap.parse_args()

    policy_specs = {
        "full-space-hard-fail": ("hard", CONDITION_THRESHOLD),
        "full-space-ridge-1e-8": ("ridge", 1e-8),
        "full-space-ridge-1e-4": ("ridge", 1e-4),
        "identifiable-subspace-tau-1e-3": ("subspace", SUBSPACE_THRESHOLD),
    }
    accum = {}

    for eps in EPSILONS:
        for pname in policy_specs:
            for dname in DIRECTIONS:
                key = (eps, pname, dname)
                accum[key] = {
                    "fail": 0, "no_root": 0, "null": [], "alt": [],
                    "roots": [], "deltas": [], "ratios": [], "signal_sds": []
                }

    rng = SplitMix64(args.seed)
    for _rep in range(args.mc):
        for eps in EPSILONS:
            F = generate_features(rng, eps)
            for pname, spec in policy_specs.items():
                origins = build_policy(F, *spec)
                if origins is None:
                    for dname in DIRECTIONS:
                        accum[(eps, pname, dname)]["fail"] += 1
                    continue
                minratio = min(x[2] for x in origins)
                roots = {dname: root_for_direction(origins, v) for dname, v in DIRECTIONS.items()}
                for dname, v in DIRECTIONS.items():
                    a = accum[(eps, pname, dname)]
                    a["ratios"].append(float(minratio))
                    r = roots[dname]
                    zsd = signal_sd(F, v)
                    a["signal_sds"].append(zsd)
                    if r is None or zsd <= 1e-12:
                        a["no_root"] += 1
                        continue
                    delta = LOCAL_SIGNAL_RMS / zsd
                    a["roots"].append(r)
                    a["deltas"].append(delta)
                    for scenario, kappa in (("null", r), ("alt", r + delta)):
                        y = generate_outcome(F, rng, kappa, v)
                        a[scenario].append(statistic(F, y, origins))

    results = []
    for (eps, pname, dname), a in accum.items():
        null = np.asarray(a["null"], dtype=float)
        alt = np.asarray(a["alt"], dtype=float)
        fail_rate = a["fail"] / args.mc
        results.append({
            "epsilon": eps,
            "policy": pname,
            "direction": dname,
            "conditioning": {
                "threshold": CONDITION_THRESHOLD,
                "min_ratio_median": float(np.median(a["ratios"])) if a["ratios"] else None,
                "fail_rate": fail_rate,
                "fail_count": a["fail"],
            },
            "equal_risk_null": {
                "root_available_rate": len(a["null"]) / args.mc,
                "no_root_rate": a["no_root"] / args.mc,
                "root_median": float(np.median(a["roots"])) if a["roots"] else None,
                "null_T_mean": float(null.mean()) if len(null) else None,
                "null_T_reject_rate_at_fixed_cutoff": float(np.mean(null > args.cutoff)) if len(null) else None,
                "null_n": len(null),
                "definition": (
                    "Full-space equal-risk surface for full-space policies; policy-specific projected-space "
                    "equal-risk surface for the identifiable-subspace policy. The latter is not the original "
                    "three-coefficient RH-006 estimand."
                ),
            },
            "local_alternative": {
                "signal_rms": LOCAL_SIGNAL_RMS,
                "delta_median": float(np.median(a["deltas"])) if a["deltas"] else None,
                "alt_T_mean": float(alt.mean()) if len(alt) else None,
                "rejection_rate_at_fixed_cutoff": float(np.mean(alt > args.cutoff)) if len(alt) else None,
                "alt_n": len(alt),
                "definition": "Outcome-scale local shift equal to 0.25 × nominal outcome innovation scale, not coefficient-scale closeness.",
            },
        })

    out = {
        "schema": "rh006-conditioning-ridge-estimand-policy-power-executed/v1",
        "status": "research-diagnostic-only",
        "faithfulness_scope": "independent estimator-mechanics reimplementation; not the Symthaea Rust implementation",
        "rng": "SplitMix64 v1 + Rademacher innovations",
        "execution": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "command": f"python scripts/research/rh006_conditioning_ridge_estimand_policy_power.py --mc {args.mc} --seed {args.seed}",
        },
        "geometry": {
            "train": TRAIN, "test": TEST, "gap": GAP, "origins": ORIGINS, "step": STEP,
            "horizon": 1.0, "outcome_rho": RHO_ERR, "heteroskedastic": HETERO,
            "bartlett_lag": BARTLETT_LAG, "fixed_T_cutoff": args.cutoff,
            "mc_reps": args.mc, "seed": args.seed,
            "epsilons": list(EPSILONS), "condition_threshold": CONDITION_THRESHOLD,
            "subspace_threshold": SUBSPACE_THRESHOLD, "ridge_values": list(RIDGE_POLICIES),
            "local_signal_rms": LOCAL_SIGNAL_RMS,
        },
        "policies": {
            "full-space-hard-fail": "full 3-channel relational space; OLS solve; reject the dataset diagnostically if any origin-local singular ratio < 1e-3",
            "full-space-ridge-1e-8": "full 3-channel relational space; fixed ridge 1e-8; no conditioning stop, but diagnostics retain origin-local conditioning",
            "full-space-ridge-1e-4": "full 3-channel relational space; fixed ridge 1e-4; no conditioning stop, but diagnostics retain origin-local conditioning",
            "identifiable-subspace-tau-1e-3": "per-origin SVD projection of the 3 relational channels, retaining singular directions with ratio >= 1e-3; fixed ridge 1e-8; separately declared projected estimand",
        },
        "directions": {
            "strong": "[1,1,0.5] normalized: aligned with the approximately dominant near-collinear feature direction",
            "weak": "[0,1,-2] normalized: approximately orthogonal to the dominant near-collinear direction, probing a weakly identified relational contrast",
        },
        "design_note": "The equal-risk root is solved against the actual AR(0.5)+heteroskedastic error covariance of each synthetic feature path, so the policy comparison does not silently substitute the structural beta=0 null for the finite-sample equal-risk null.",
        "results": results,
        "interpretation": {
            "primary": "Origin-local hard-fail closes the last-origins loophole but converts near-singularity into explicit inapplicability; ridge preserves executability but does not preserve coefficient identifiability; adaptive SVD projection can materially change the tested direction and may eliminate the full-space equal-risk root in weak directions.",
            "size_power": "Rejection rates use a fixed N(0,1) screening cutoff of 1.645 only as a cross-condition stress diagnostic. They are not formal size or power guarantees, and the cutoff is not a repository-approved inference rule.",
            "projection_boundary": "Any apparent improvement from projection must be evaluated as inference for the projected estimand, not as inference for the original three-channel RH-006 hypothesis. This is consistent with the broader post-selection literature's warning that the inferential target after data-dependent reduction needs its own specification.",
            "formal_inference": False,
            "applicability": "not-approved-for-execution",
            "selection": "stop-assumption-failure",
        },
        "references": [
            "Giacomini & White (2006), DOI 10.1111/j.1468-0262.2006.00718.x",
            "Clark & McCracken (2015), DOI 10.1016/j.jeconom.2014.06.016",
            "Doko Tchatoka & Haque (2023), DOI 10.1002/for.2987",
            "Zhu & Timmermann (2020), arXiv:2006.03238",
            "Harvey, Leybourne & Zu (2025), DOI 10.1080/07350015.2024.2418835",
            "Tibshirani et al. (2016), post-selection inference / arXiv:1506.06266",
        ],
    }
    raw = json.dumps(out, sort_keys=True, separators=(",", ":"))
    out["payload_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    print(json.dumps(out, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
