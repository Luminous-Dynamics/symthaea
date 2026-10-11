#!/usr/bin/env python3
"""
RH-006 dependent/heteroskedastic finite-sample forecast-risk-null oracle.

Independent estimator-mechanics reimplementation of the current RH-006
rolling geometry. It derives the exact conditional finite-sample forecast-risk
difference for a fixed feature path and a known error covariance, then
calibrates an oracle bootstrap at a predeclared equal-risk point on a
predeclared relational direction.

Research diagnostic only. Not the Symthaea Rust implementation and not a
formal inference procedure.
"""
import argparse
import hashlib
import json
import math
import numpy as np

TRAIN, TEST, GAP, ORIGINS, STEP = 48, 16, 4, 24, 16
RIDGE_DEFAULT, SIGMA_DEFAULT, BARTLETT_LAG, ALPHA = 1e-8, 0.08, 3, 0.05
N = TRAIN + GAP + TEST + (ORIGINS - 1) * STEP
DIRECTION = np.ones(3) / math.sqrt(3)


class SplitMix64:
    def __init__(self, seed: int):
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


def generate_features(n, rng, rho=0.35):
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


def outcome_covariance(n, rho, heteroskedastic, common):
    sigma = SIGMA_DEFAULT * (1.0 + 0.7 * (common - 0.5)) if heteroskedastic else np.full(n, SIGMA_DEFAULT)
    # e_t = rho e_{t-1} + sigma_t u_t, Var(u_t)=1, with e_0=sigma_0 u_0.
    L = np.zeros((n, n))
    for t in range(n):
        L[t, t] = sigma[t]
        if t:
            L[t, :t] = (rho ** np.arange(t, 0, -1)) * sigma[:t]
    return L @ L.T, sigma


def operator(Xtr, Xte, ridge):
    mu = Xtr.mean(axis=0)
    sc = np.where(Xtr.std(axis=0) <= 1e-12, 1.0, Xtr.std(axis=0))
    Atr = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sc])
    Ate = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sc])
    M = Atr.T @ Atr
    if ridge:
        M[1:, 1:] += ridge * np.eye(Xtr.shape[1])
    return Ate @ np.linalg.solve(M, Atr.T)


def build_ops(F, ridge):
    nr = [0, 1, 2, 6]
    rel = [0, 1, 2, 3, 4, 5, 6]
    ops = []
    for o in range(ORIGINS):
        s = o * STEP
        te = s + TRAIN + GAP
        ops.append((
            operator(F[s:s + TRAIN, nr], F[te:te + TEST, nr], ridge),
            operator(F[s:s + TRAIN, rel], F[te:te + TEST, rel], ridge),
        ))
    return ops


def mean_components(F):
    a, b, alignment, _a2b, _b2a, _turn, common = F.T
    mu0 = 0.15 + 0.45 * a + 0.25 * b + 0.20 * common + 0.30 * alignment
    z = F[:, 3:6] @ DIRECTION
    return mu0, z


def quadratic(F, rho, heteroskedastic, ridge):
    ops = build_ops(F, ridge)
    mu0, z = mean_components(F)
    Sigma, _sigma = outcome_covariance(len(F), rho, heteroskedastic, F[:, -1])

    Q = np.zeros((3, 3))
    b = np.zeros(3)
    c = 0.0
    for o, (Wc, Wr) in enumerate(ops):
        s = o * STEP
        te = s + TRAIN + GAP
        mtr = mu0[s:s + TRAIN]
        mte = mu0[te:te + TEST]
        Ztr = F[s:s + TRAIN, 3:6]
        Zte = F[te:te + TEST, 3:6]
        qc = Wc @ mtr - mte
        qr = Wr @ mtr - mte
        Hc = Wc @ Ztr - Zte
        Hr = Wr @ Ztr - Zte
        S_tt = Sigma[s:s + TRAIN, s:s + TRAIN]
        S_ty = Sigma[s:s + TRAIN, te:te + TEST]

        # E||Wy_tr-y_te||^2 = ||W mu_tr-mu_te||^2
        #   + tr(W S_tt W') + tr(S_yy) - 2 tr(W S_ty).
        # The S_yy term is common to both nested forecasts and cancels in Δ.
        def stochastic_term(W):
            return np.trace(W @ S_tt @ W.T) - 2.0 * np.trace(W @ S_ty)

        Q += (Hc.T @ Hc - Hr.T @ Hr) / TEST
        b += 2.0 * (Hc.T @ qc - Hr.T @ qr) / TEST
        c += (
            qc @ qc
            - qr @ qr
            + stochastic_term(Wc)
            - stochastic_term(Wr)
        ) / TEST

    return Q / ORIGINS, b / ORIGINS, c / ORIGINS


def positive_root(A, B, C):
    coeff = [A, B, C] if abs(A) > 1e-14 else [B, C]
    roots = np.roots(coeff)
    positive = [
        float(r.real)
        for r in roots
        if abs(r.imag) < 1e-9 and r.real > 0
    ]
    return min(positive) if positive else None


def directional_root(Q, b, c, v):
    return positive_root(float(v @ Q @ v), float(b @ v), float(c))


def studentized(D):
    D = np.asarray(D, float)
    m = D.mean()
    lrv = np.mean((D - m) ** 2)
    for k in range(1, BARTLETT_LAG + 1):
        lrv += 2.0 * (1.0 - k / (BARTLETT_LAG + 1)) * np.mean(
            (D[k:] - m) * (D[:-k] - m)
        )
    return 0.0 if lrv <= 1e-14 else math.sqrt(len(D)) * m / math.sqrt(lrv)


def studentized_batch(DT):
    m = DT.mean(axis=1)
    lrv = np.mean((DT - m[:, None]) ** 2, axis=1)
    for k in range(1, BARTLETT_LAG + 1):
        lrv += 2.0 * (1.0 - k / (BARTLETT_LAG + 1)) * np.mean(
            (DT[:, k:] - m[:, None]) * (DT[:, :-k] - m[:, None]),
            axis=1,
        )
    return np.where(lrv > 1e-14, np.sqrt(ORIGINS) * m / np.sqrt(lrv), 0.0)


def bootstrap_pvalue(F, y, mu, ops, rng, rho, heteroskedastic, B):
    Dobs = []
    for o, (Wc, Wr) in enumerate(ops):
        s = o * STEP
        te = s + TRAIN + GAP
        pn = Wc @ y[s:s + TRAIN]
        pr = Wr @ y[s:s + TRAIN]
        yte = y[te:te + TEST]
        Dobs.append(np.mean((pn - yte) ** 2 - (pr - yte) ** 2))
    Dobs = np.asarray(Dobs)
    Tobs = studentized(Dobs)

    _Sigma, sigma = outcome_covariance(len(y), rho, heteroskedastic, F[:, -1])
    signs = rng.rademacher(B * len(y)).reshape(B, len(y))
    E = np.empty_like(signs)
    E[:, 0] = sigma[0] * signs[:, 0]
    for t in range(1, len(y)):
        E[:, t] = rho * E[:, t - 1] + sigma[t] * signs[:, t]
    Yb = mu[None, :] + E

    DT = np.empty((B, ORIGINS))
    for o, (Wc, Wr) in enumerate(ops):
        s = o * STEP
        te = s + TRAIN + GAP
        pn = Yb[:, s:s + TRAIN] @ Wc.T
        pr = Yb[:, s:s + TRAIN] @ Wr.T
        yte = Yb[:, te:te + TEST]
        DT[:, o] = np.mean((pn - yte) ** 2 - (pr - yte) ** 2, axis=1)
    Tb = studentized_batch(DT)
    p = (1 + int(np.count_nonzero(Tb >= Tobs))) / (B + 1)
    return p, Tobs, float(Dobs.mean())


def monte_carlo_check(F, mu, ops, rng, rho, heteroskedastic, root, reps):
    _Sigma, sigma = outcome_covariance(len(F), rho, heteroskedastic, F[:, -1])
    signs = rng.rademacher(reps * len(F)).reshape(reps, len(F))
    E = np.empty_like(signs)
    E[:, 0] = sigma[0] * signs[:, 0]
    for t in range(1, len(F)):
        E[:, t] = rho * E[:, t - 1] + sigma[t] * signs[:, t]
    Y = mu[None, :] + E
    DT = np.empty((reps, ORIGINS))
    for o, (Wc, Wr) in enumerate(ops):
        s = o * STEP
        te = s + TRAIN + GAP
        pn = Y[:, s:s + TRAIN] @ Wc.T
        pr = Y[:, s:s + TRAIN] @ Wr.T
        yte = Y[:, te:te + TEST]
        DT[:, o] = np.mean((pn - yte) ** 2 - (pr - yte) ** 2, axis=1)
    estimates = DT.mean(axis=1)
    return {
        "root": float(root),
        "analytic_delta_at_root": 0.0,
        "mc_delta_at_root": float(estimates.mean()),
        "mc_se": float(estimates.std(ddof=1) / math.sqrt(reps)),
        "mc_reps": reps,
    }


def fibonacci_directions(n):
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * math.pi * (i / golden)
        yield np.array([r * math.cos(theta), r * math.sin(theta), z])


def run_scenario(rho, heteroskedastic, args, seed):
    rng = SplitMix64(seed)
    roots = []
    eigmins = []
    eigmaxs = []
    root_ranges = []
    calibration_ps = []
    calibration_ds = []
    calibration_ts = []
    mc_checks = []

    for _path in range(args.paths):
        F = generate_features(N, rng)
        Q, b, c = quadratic(F, rho, heteroskedastic, args.ridge)
        root = directional_root(Q, b, c, DIRECTION)
        if root is None:
            continue
        roots.append(root)
        eig = np.linalg.eigvalsh(Q)
        eigmins.append(float(eig[0]))
        eigmaxs.append(float(eig[-1]))

        directional_roots = [
            r for v in fibonacci_directions(args.directions)
            if (r := directional_root(Q, b, c, v)) is not None
        ]
        root_ranges.append((min(directional_roots), max(directional_roots)))

        mu0, z = mean_components(F)
        mu = mu0 + root * z
        ops = build_ops(F, args.ridge)

        mc_checks.append(
            monte_carlo_check(
                F, mu, ops, rng, rho, heteroskedastic, root, args.mc_verify
            )
        )

        for _ in range(args.calibration_outcomes_per_path):
            signs = rng.rademacher(N)
            _Sigma, sigma = outcome_covariance(N, rho, heteroskedastic, F[:, -1])
            e = np.empty(N)
            e[0] = sigma[0] * signs[0]
            for t in range(1, N):
                e[t] = rho * e[t - 1] + sigma[t] * signs[t]
            y = mu + e
            p, tstat, dbar = bootstrap_pvalue(
                F, y, mu, ops, rng, rho, heteroskedastic, args.bootstrap
            )
            calibration_ps.append(p)
            calibration_ts.append(tstat)
            calibration_ds.append(dbar)

    ps = np.asarray(calibration_ps)
    rejections = ps < ALPHA
    rr = np.asarray(root_ranges)
    mc = np.asarray([x["mc_delta_at_root"] for x in mc_checks])
    return {
        "rho": rho,
        "heteroskedastic": heteroskedastic,
        "paths": len(roots),
        "positive_root_count": len(roots),
        "positive_root_mean": float(np.mean(roots)),
        "positive_root_median": float(np.median(roots)),
        "positive_root_min": float(np.min(roots)),
        "positive_root_max": float(np.max(roots)),
        "Q_eigen_min_mean": float(np.mean(eigmins)),
        "Q_eigen_max_mean": float(np.mean(eigmaxs)),
        "directional_positive_root_fraction": float(len(root_ranges) / len(roots)),
        "directional_root_min_mean": float(np.mean(rr[:, 0])),
        "directional_root_max_mean": float(np.mean(rr[:, 1])),
        "directional_root_global_min": float(np.min(rr[:, 0])),
        "directional_root_global_max": float(np.max(rr[:, 1])),
        "risk_null_mc_delta_mean": float(mc.mean()),
        "risk_null_mc_delta_se_mean": float(mc.std(ddof=1) / math.sqrt(len(mc))),
        "risk_null_bootstrap": {
            "replicates": len(ps),
            "outcomes_per_feature_path": args.calibration_outcomes_per_path,
            "reject_count": int(rejections.sum()),
            "empirical_size": float(rejections.mean()),
            "monte_carlo_se": float(math.sqrt(rejections.mean() * (1.0 - rejections.mean()) / len(ps))),
            "mean_p": float(ps.mean()),
            "mean_T": float(np.mean(calibration_ts)),
            "mean_D": float(np.mean(calibration_ds)),
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=50)
    ap.add_argument("--directions", type=int, default=128)
    ap.add_argument("--calibration-outcomes-per-path", type=int, default=4)
    ap.add_argument("--mc-verify", type=int, default=500)
    ap.add_argument("--bootstrap", type=int, default=199)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--ridge", type=float, default=RIDGE_DEFAULT)
    args = ap.parse_args()

    results = [
        run_scenario(0.0, False, args, args.seed + 0),
        run_scenario(0.5, False, args, args.seed + 1000),
        run_scenario(0.8, False, args, args.seed + 2000),
        run_scenario(0.5, True, args, args.seed + 3000),
    ]
    out = {
        "schema": "rh006-dependent-heteroskedastic-risk-null-oracle-executed/v1",
        "status": "research-diagnostic-only",
        "faithfulness_scope": "independent estimator-mechanics reimplementation; not the Symthaea Rust implementation",
        "geometry": {
            "train": TRAIN,
            "test": TEST,
            "gap": GAP,
            "origins": ORIGINS,
            "step": STEP,
            "ridge": args.ridge,
            "bartlett_lag": BARTLETT_LAG,
            "alpha": ALPHA,
            "paths": args.paths,
            "directions": args.directions,
            "calibration_outcomes_per_path": args.calibration_outcomes_per_path,
            "mc_verify": args.mc_verify,
            "bootstrap_reps": args.bootstrap,
        },
        "risk_equation": {
            "vector_form": "Delta(gamma) = gamma' Q gamma + b' gamma + c",
            "directional_form": "Delta(kappa; v) = (v'Qv) kappa^2 + (b'v) kappa + c",
            "covariance_terms": ["train/train", "test/test", "train/test"],
            "test_test_cancellation": "trace(Sigma_test,test) is common to both forecasts and cancels in the paired loss difference",
        },
        "null_construction": {
            "structural_null": "beta_relational = 0",
            "forecast_accuracy_null": "E[D_o] = 0",
            "direction": "equal-amplitude [a_to_b, b_to_a, turn_taking]",
            "root_policy": "smallest positive root on the predeclared direction; root uses feature path and oracle-known covariance only",
        },
        "scenarios": results,
        "interpretation": {
            "primary_finding": "Serial correlation materially changes the finite-sample equal-risk boundary even when the estimator and feature path are held fixed. The test/test covariance is part of the complete risk identity but cancels from the paired MSE differential.",
            "secondary_finding": "The equal-risk null is generally a quadratic surface in the full relational coefficient vector. A scalar direction such as equal-amplitude is therefore a slice through the null and is not, by itself, a direction-free characterization of equal predictive accuracy.",
            "bootstrap_finding": "Oracle risk-null refit bootstrap behavior was simulated at a predeclared equal-risk root. This validates the mechanics under the stated synthetic DGP only; it does not establish validity for RH-006 data or estimated nuisance parameters.",
            "feature_endogeneity_gate": "The covariance calculation conditions on the feature path. If features are endogenous with outcomes, the bootstrap DGP must generate features and outcomes jointly.",
            "formal_inference": False,
            "applicability": "not-approved-for-execution",
            "selection": "stop-assumption-failure",
        },
        "references": [
            "Clark and McCracken (2015), DOI 10.1016/j.jeconom.2014.06.016",
            "Doko Tchatoka and Haque (2023), DOI 10.1002/for.2987",
            "Zhu and Timmermann (2020), arXiv:2006.03238 / subsequent working-paper versions",
            "Harvey, Leybourne and Zu (2025), DOI 10.1080/07350015.2024.2418835",
        ],
    }
    raw = json.dumps(out, sort_keys=True, separators=(",", ":"))
    out["payload_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
