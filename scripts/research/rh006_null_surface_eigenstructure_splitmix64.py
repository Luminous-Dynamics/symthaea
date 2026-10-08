#!/usr/bin/env python3
"""RH-006 quadratic null-surface eigenstructure/curvature diagnostic.

Research-only mechanics mirror. It does not execute formal inference.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

TRAIN, GAP, TEST, ORIGINS, STEP = 48, 4, 16, 24, 16
N = TRAIN + GAP + TEST + (ORIGINS - 1) * STEP
ALPHA = 0.05
WEAK_DIRECTION = np.array([0.0, 1.0, -2.0], dtype=float)
WEAK_DIRECTION /= np.linalg.norm(WEAK_DIRECTION)

class SM:
    def __init__(self, x):
        self.x = np.uint64(int(x) & 0xFFFFFFFFFFFFFFFF)
    def r(self, n):
        n = int(n)
        i = self.x + np.arange(n, dtype=np.uint64)
        z = i + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
        self.x = np.uint64((int(self.x) + n) & 0xFFFFFFFFFFFFFFFF)
        return np.where((z & np.uint64(1)) == 0, -1.0, 1.0)

def ar(g, n, rho):
    e = g.r(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x

def feat(g, epsilon):
    raw = np.column_stack([ar(g, N, 0.35) for _ in range(7)])
    f = lambda x: np.clip(0.5 + 0.18 * x, 0, 1)
    a, b, c = f(raw[:, 0]), f(raw[:, 1]), f(raw[:, 2])
    al = f(0.65 * raw[:, 3] + 0.35 * raw[:, 4])
    a2b, b2a = f(raw[:, 5]), f(raw[:, 6])
    turn = f(0.45 * raw[:, 4] + 0.55 * raw[:, 5])
    F = np.column_stack([a, b, al, a2b, b2a, turn, c])
    z = F.copy()
    z[:, 4] = np.clip(F[:, 3] + epsilon * 0.18 * ar(g, N, 0.25), 0, 1)
    z[:, 5] = np.clip(0.25 + 0.5 * F[:, 3] + epsilon * 0.18 * ar(g, N, 0.25), 0, 1)
    return z

def mu0(F):
    return 0.15 + 0.45 * F[:, 0] + 0.25 * F[:, 1] + 0.20 * F[:, 6] + 0.30 * F[:, 2]

def sig0(F):
    return 0.08 * (1 + 0.7 * (F[:, -1] - 0.5))

def op(X, Y, ridge):
    m = X.mean(0)
    sc = np.where(X.std(0) <= 1e-12, 1.0, X.std(0))
    A = np.c_[np.ones(len(X)), (X - m) / sc]
    B = np.c_[np.ones(len(Y)), (Y - m) / sc]
    G = A.T @ A
    G[1:, 1:] += ridge * np.eye(X.shape[1])
    return B @ np.linalg.solve(G, A.T)

def build_ops(F, ridge):
    nr = [0, 1, 2, 6]
    rel = list(range(7))
    out = []
    for o in range(ORIGINS):
        s = o * STEP
        te = s + TRAIN + GAP
        out.append((
            op(F[s:s + TRAIN, nr], F[te:te + TEST, nr], ridge),
            op(F[s:s + TRAIN, rel], F[te:te + TEST, rel], ridge),
        ))
    return out

def covariance(rho, sig):
    L = np.zeros((N, N))
    for t in range(N):
        L[t, t] = sig[t]
        if t:
            L[t, :t] = (rho ** np.arange(t, 0, -1)) * sig[:t]
    return L @ L.T

def quadratic(F, ridge, m0, rho, sig):
    ops = build_ops(F, ridge)
    S = covariance(rho, sig)
    Z = F[:, 3:6]
    Q = np.zeros((3, 3))
    b = np.zeros(3)
    c = 0.0
    for o, (Wc, Wr) in enumerate(ops):
        s = o * STEP
        te = s + TRAIN + GAP
        tr = np.arange(s, s + TRAIN)
        tt = np.arange(te, te + TEST)
        qc = Wc @ m0[tr] - m0[tt]
        qr = Wr @ m0[tr] - m0[tt]
        Hc = Wc @ Z[tr] - Z[tt]
        Hr = Wr @ Z[tr] - Z[tt]
        St = S[np.ix_(tr, tr)]
        Sty = S[np.ix_(tr, tt)]
        st = lambda W: np.trace(W @ St @ W.T) - 2 * np.trace(W @ Sty)
        Q += (Hc.T @ Hc - Hr.T @ Hr) / TEST
        b += 2 * (Hc.T @ qc - Hr.T @ qr) / TEST
        c += (qc @ qc - qr @ qr + st(Wc) - st(Wr)) / TEST
    return Q / ORIGINS, b / ORIGINS, c / ORIGINS, ops

def roots(Q, b, c, v):
    A = float(v @ Q @ v)
    B = float(b @ v)
    C = float(c)
    if abs(A) <= 1e-14:
        if abs(B) <= 1e-14:
            return []
        t = -C / B
        return [float(t)] if np.isfinite(t) and t > 0 else []
    D = B * B - 4 * A * C
    if D < -1e-12 * (B * B + abs(4 * A * C) + 1e-30):
        return []
    D = max(D, 0.0)
    s = math.sqrt(D)
    if s == 0:
        rr = [-B / (2 * A)]
    else:
        q = -0.5 * (B + math.copysign(s, B))
        rr = [q / A, C / q] if q != 0 else [(-B - s) / (2 * A), (-B + s) / (2 * A)]
    return sorted(float(x) for x in rr if np.isfinite(x) and x > 0)

def signature(vals, rel_tol=1e-10):
    s = max(float(np.max(np.abs(vals))), 1e-30)
    return "".join("+" if x > rel_tol * s else "-" if x < -rel_tol * s else "0" for x in vals)

def discriminant_matrix(Q, b, c):
    Qs = (Q + Q.T) / 2
    return np.outer(b, b) - 4 * c * Qs

def stable_eigensystem(M):
    Ms = (M + M.T) / 2
    return np.linalg.eigh(Ms)

def eigenstructure_directions(Q, Dmat):
    _, U = stable_eigensystem(Q)
    _, V = stable_eigensystem(Dmat)
    dirs = []
    labels = []
    for i in range(3):
        for sg in (-1.0, 1.0):
            dirs.append(sg * U[:, i])
            labels.append(f"Qe{i}{'+' if sg>0 else '-'}")
    angles = [5, 15, 30, 45, 60, 75, 85, 89]
    for i in range(3):
        for j in range(i + 1, 3):
            for deg in angles:
                th = math.radians(deg)
                for sg in (-1.0, 1.0):
                    vv = math.cos(th) * U[:, i] + sg * math.sin(th) * U[:, j]
                    dirs.append(vv / np.linalg.norm(vv))
                    labels.append(f"Qmix{i}{j}_{deg:02d}{'+' if sg>0 else '-'}")
    for i in range(3):
        for sg in (-1.0, 1.0):
            dirs.append(sg * V[:, i])
            labels.append(f"De{i}{'+' if sg>0 else '-'}")
    for sg in (-1.0, 1.0):
        dirs.append(sg * WEAK_DIRECTION)
        labels.append(f"weak{'+' if sg>0 else '-'}")
    return labels, np.asarray(dirs)

def root_case(Q, b, c, v):
    A = float(v @ Q @ v)
    B = float(b @ v)
    D = B * B - 4 * A * c
    rr = roots(Q, b, c, v)
    if rr:
        return "two_positive" if len(rr) >= 2 else "one_positive", A, B, D, rr
    scale = B * B + abs(4 * A * c) + 1e-30
    if D < -1e-12 * scale:
        return "discriminant_negative", A, B, D, []
    if abs(A) <= 1e-14:
        return "linear_nonpositive", A, B, D, []
    return "no_positive_real", A, B, D, []

def principal_curvatures(Q, b, x):
    g = 2 * Q @ x + b
    gn = float(np.linalg.norm(g))
    if not np.isfinite(gn) or gn <= 1e-14:
        return None
    P = np.eye(3) - np.outer(g, g) / (gn * gn)
    vals = 2.0 * np.linalg.eigvalsh(P @ Q @ P) / gn
    vals = vals[np.argsort(np.abs(vals))]
    return float(vals[1]), float(vals[2]), gn

def summarize(args):
    rng = SM(args.seed)
    signatures = Counter()
    discrim_signatures = Counter()
    failure_cases = Counter()
    all_abs_curv = []
    all_abs_grad = []
    support_grid = []
    weak_supported = 0
    weak_total = 0
    two_positive_grid = 0
    eigenratio = []
    discrimratio = []
    weak_curvature_ratio = []
    weak_discriminant_ratio = []
    discrim_outer_ratio = []
    discrim_q_alignment = []
    strata = Counter()
    roots_per_grid = []
    root_magnitudes = []
    gradient_criticality = []
    weak_root_magnitudes = []
    for _ in range(args.paths):
        F = feat(rng, args.epsilon)
        m = mu0(F)
        sig = sig0(F)
        Q, b, c, _ = quadratic(F, args.ridge, m, 0.5, sig)
        ew, U = stable_eigensystem(Q)
        R = discriminant_matrix(Q, b, c)
        er, _ = stable_eigensystem(R)
        base_R = -4.0 * c * ((Q + Q.T) / 2)
        discrim_outer_ratio.append(float(np.linalg.norm(np.outer(b, b)) / max(np.linalg.norm(base_R), 1e-30)))
        denom_align = max(np.linalg.norm(R) * np.linalg.norm(base_R), 1e-30)
        discrim_q_alignment.append(float(np.sum(R * base_R) / denom_align))
        signatures[signature(ew)] += 1
        discrim_signatures[signature(er)] += 1
        nz = np.max(np.abs(ew))
        eigenratio.append(float(np.sort(np.abs(ew))[-2] / max(nz, 1e-30)))
        discrimratio.append(float(np.sort(np.abs(er))[-2] / max(np.max(np.abs(er)), 1e-30)))
        Aweak = float(WEAK_DIRECTION @ Q @ WEAK_DIRECTION)
        Rweak = float(WEAK_DIRECTION @ R @ WEAK_DIRECTION)
        weak_curvature_ratio.append(float(Aweak / max(nz, 1e-30)))
        weak_discriminant_ratio.append(float(Rweak / max(np.max(np.abs(er)), 1e-30)))
        labels, dirs = eigenstructure_directions(Q, R)
        supported = 0
        local_roots = 0
        for label, vv in zip(labels, dirs):
            typ, A, B, D, rr = root_case(Q, b, c, vv)
            if rr:
                supported += 1
                local_roots += len(rr)
                for t in sorted(set(rr)):
                    root_magnitudes.append(abs(t))
                    k = principal_curvatures(Q, b, t * vv)
                    if k:
                        all_abs_curv.extend([abs(k[0]), abs(k[1])])
                        all_abs_grad.append(k[2])
                        gradient_criticality.append(k[2] / max(np.linalg.norm(b) + 2.0 * np.linalg.norm(Q, 2) * abs(t), 1e-30))
                    if label.startswith("weak+"):
                        weak_root_magnitudes.append(abs(t))
            if typ == "two_positive":
                two_positive_grid += 1
            if label.startswith("weak+"):
                weak_total += 1
                if rr:
                    weak_supported += 1
                else:
                    failure_cases[root_case(Q, b, c, WEAK_DIRECTION)[0]] += 1
            Arel = abs(A) / max(nz, 1e-30)
            if Arel > 1e-2:
                strata["high_curvature"] += 1
            elif Arel > 1e-6:
                strata["low_curvature"] += 1
            elif A >= 0:
                strata["near_flat"] += 1
            else:
                strata["negative_weak_curvature"] += 1
        support_grid.append(supported / len(dirs))
        roots_per_grid.append(local_roots / len(dirs))
    return {
        "schema": "rh006-null-surface-eigenstructure-diagnostic/v1",
        "status": "research-diagnostic-only",
        "formal_inference": False,
        "applicability": "not-approved-for-execution",
        "selection": "stop-assumption-failure",
        "execution": {
            "mc_paths": args.paths,
            "seed": args.seed,
            "epsilon": args.epsilon,
            "ridge": args.ridge,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "frozen_geometry": {
            "train": TRAIN, "gap": GAP, "test": TEST, "origins": ORIGINS, "step": STEP,
            "weak_direction": WEAK_DIRECTION.tolist(),
            "direction_design": "Q principal axes ±; fixed angular sweeps in every Q eigenplane; discriminant-matrix axes ±; weak-direction ±",
            "angles_degrees": [5,15,30,45,60,75,85,89],
        },
        "quadratic_form": "Delta(gamma)=gamma'Q gamma+b'gamma+c",
        "ray_parameterization": {
            "definition": "gamma=t u, ||u||=1, t>0",
            "A": "u'Q u",
            "B": "b'u",
            "C": "c",
            "discriminant": "D(u)=B^2-4AC = u'(bb'-4cQ)u",
            "positive_root_condition": "solved exactly from A,B,C after classifying D; no root is silently imputed",
            "branch_rule": "both positive roots are retained when present; no near/far branch is declared scientifically preferred",
        },
        "results": {
            "q_signature_counts": dict(signatures),
            "discriminant_signature_counts": dict(discrim_signatures),
            "q_second_abs_eigen_ratio_quantiles": np.quantile(eigenratio,[0,.1,.5,.9,1]).tolist(),
            "discriminant_second_abs_eigen_ratio_quantiles": np.quantile(discrimratio,[0,.1,.5,.9,1]).tolist(),
            "discriminant_outer_to_q_term_norm_ratio_quantiles": np.quantile(discrim_outer_ratio,[0,.1,.5,.9,1]).tolist(),
            "discriminant_to_q_term_frobenius_alignment_quantiles": np.quantile(discrim_q_alignment,[0,.1,.5,.9,1]).tolist(),
            "weak_direction_root_support_rate": weak_supported / max(weak_total,1),
            "weak_direction_failure_cases": dict(failure_cases),
            "eigenstructure_grid_support_rate_quantiles": np.quantile(support_grid,[0,.1,.5,.9,1]).tolist(),
            "grid_positive_root_multiplicity_rate_quantiles": np.quantile(roots_per_grid,[0,.1,.5,.9,1]).tolist(),
            "root_magnitude_quantiles": np.quantile(root_magnitudes,[0,.01,.1,.5,.9,.99,1]).tolist() if root_magnitudes else [],
            "root_fraction_above_radius": {str(r): float(np.mean(np.asarray(root_magnitudes) > r)) for r in [0.1,1,10,100,1000]} if root_magnitudes else {},
            "surface_gradient_normalized_criticality_quantiles": np.quantile(gradient_criticality,[0,.01,.1,.5,.9,.99,1]).tolist() if gradient_criticality else [],
            "weak_direction_root_magnitude_quantiles": np.quantile(weak_root_magnitudes,[0,.1,.5,.9,.99,1]).tolist() if weak_root_magnitudes else [],
            "weak_direction_roots_above_100": int(np.count_nonzero(np.asarray(weak_root_magnitudes) > 100)) if weak_root_magnitudes else 0,
            "total_two_positive_grid_points": two_positive_grid,
            "surface_principal_curvature_abs_quantiles": np.quantile(all_abs_curv,[0,.1,.5,.9,1]).tolist() if all_abs_curv else [],
            "surface_gradient_norm_quantiles": np.quantile(all_abs_grad,[0,.1,.5,.9,1]).tolist() if all_abs_grad else [],
            "weak_ray_curvature_ratio_quantiles": np.quantile(weak_curvature_ratio,[0,.1,.5,.9,1]).tolist(),
            "weak_ray_discriminant_ratio_quantiles": np.quantile(weak_discriminant_ratio,[0,.1,.5,.9,1]).tolist(),
        },
        "interpretation": {
            "geometry": "The null surface is better represented as a ray-parameterized quadratic surface with two coupled spectral geometries: Q controls curvature, while R=bb'-4cQ controls discriminant/root-existence geometry.",
            "weak_identification": "A weakly curved Q direction can sit inside a narrow no-root cone even when a generic spherical grid reports near-complete support. Eigen-aligned sweeps therefore target the boundary rather than relying on coarse isotropic coverage.",
            "branch_semantics": "The evaluator retains both positive branches if they occur. If only one branch occurs in the frozen design, that is an observed DGP property, not a universal theorem.",
            "least_favorable_boundary": "This geometry diagnostic does not construct a least-favorable distribution or critical value.",
        },
        "gates": {
            "null_surface_support": "required",
            "root_existence_boundary": "required",
            "curvature_stratification": "required",
            "estimated_nuisance_by_stratum": "not_yet_executed_in_this_artifact",
            "uniform_size_control": "not_established",
        },
        "nonclaims": [
            "No formal p-value.",
            "No formal confidence interval.",
            "No uniform size guarantee.",
            "No empirical validation.",
            "No least-favorable critical value.",
            "No claim that the mechanics mirror is the exact Rust implementation.",
        ],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--epsilon", type=float, default=1e-4)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()
    out = summarize(args)
    payload = json.dumps(out, sort_keys=True, separators=(",", ":"))
    out["payload_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    print(json.dumps(out, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
