#!/usr/bin/env python3
"""RH-006 eigenstructure-stratified plug-in nuisance bootstrap diagnostic."""
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

from rh006_null_surface_eigenstructure_splitmix64 import (
    SM, feat, mu0, sig0, quadratic, eigenstructure_directions,
    roots, discriminant_matrix, N, ORIGINS, STEP, TRAIN, GAP, TEST,
)

ALPHA = 0.05
K = 3
NR = [0, 1, 2, 6]
REL = list(range(7))
Q_AXIS = 6
MIX = 48
D_START = Q_AXIS + MIX
W_START = D_START + 6
DIRECTION_INDICES = list(range(6)) + list(range(D_START, W_START)) + [W_START, W_START + 1]

def est_nuisance(F, y):
    X = F[:, REL]
    m, sc = X.mean(0), np.where(X.std(0) <= 1e-12, 1., X.std(0))
    A = np.c_[np.ones(N), (X - m) / sc]
    G = A.T @ A
    G[1:, 1:] += 1e-8 * np.eye(len(REL))
    bf = np.linalg.solve(G, A.T @ y)
    res = y - A @ bf
    C = F[:, NR]
    cm, cs = C.mean(0), np.where(C.std(0) <= 1e-12, 1., C.std(0))
    Ac = np.c_[np.ones(N), (C - cm) / cs]
    Gc = Ac.T @ Ac
    Gc[1:, 1:] += 1e-8 * np.eye(len(NR))
    bc = np.linalg.solve(Gc, Ac.T @ y)
    m0 = Ac @ bc
    den = np.dot(res[:-1], res[:-1])
    rho = float(np.clip(np.dot(res[1:], res[:-1]) / den if den > 1e-14 else 0., -.98, .98))
    inn = res[1:] - rho * res[:-1]
    x = F[1:, -1] - .5
    ab = np.linalg.lstsq(np.c_[np.ones(len(x)), x], np.abs(inn), rcond=None)[0]
    a = max(float(ab[0]), 1e-5)
    h = float(np.clip(ab[1] / a, -.95, .95))
    sig = np.maximum(a * (1 + h * (F[:, -1] - .5)), 1e-5)
    return m0, rho, sig

def stat(Y, ops):
    D = np.empty((Y.shape[0], ORIGINS))
    for o, (Wc, Wr) in enumerate(ops):
        s, te = o * STEP, o * STEP + TRAIN + GAP
        pn = Y[:, s:s + TRAIN] @ Wc.T
        pr = Y[:, s:s + TRAIN] @ Wr.T
        yt = Y[:, te:te + TEST]
        D[:, o] = np.mean((pn - yt) ** 2 - (pr - yt) ** 2, axis=1)
    m = D.mean(1)
    l = np.mean((D - m[:, None]) ** 2, 1)
    for k in range(1, K + 1):
        l += 2 * (1 - k / (K + 1)) * np.mean((D[:, k:] - m[:, None]) * (D[:, :-k] - m[:, None]), 1)
    return np.where(l > 1e-14, np.sqrt(ORIGINS) * m / np.sqrt(l), 0.)

def generate(F, rng, mu, rho, sig, B):
    z = rng.r(B * N).reshape(B, N)
    E = np.empty((B, N)); E[:, 0] = sig[0] * z[:, 0]
    for t in range(1, N):
        E[:, t] = rho * E[:, t - 1] + sig[t] * z[:, t]
    return mu[None, :] + E

def bands(Q, b, t, v):
    lam = max(abs(np.linalg.eigvalsh((Q + Q.T) / 2)))
    A = float(v @ Q @ v)
    cr = abs(A) / max(lam, 1e-30)
    cb = "high" if cr >= 1e-3 else "intermediate" if cr >= 1e-6 else "near_flat"
    rb = "local" if abs(t) <= 1 else "moderate" if abs(t) <= 100 else "remote"
    g = 2 * Q @ (t * v) + b
    crit = np.linalg.norm(g) / max(np.linalg.norm(b) + 2 * np.linalg.norm(Q, 2) * abs(t), 1e-30)
    kb = "critical" if crit < 1e-4 else "regular"
    sg = "positive" if A >= 0 else "negative"
    return f"{cb}|{rb}|{kb}|A_{sg}", cr, crit

def scenario(args, eps):
    rng = SM(args.seed + int(round(eps * 1e6)))
    strata = defaultdict(lambda: {"points":0,"est":0,"rr":[],"por":[],"pes":[],"or_rej":0,"es_rej":0,"crit":[]})
    support = defaultdict(lambda: [0,0]); branches = defaultdict(int)
    for _ in range(args.paths):
        F = feat(rng, eps); m, sig = mu0(F), sig0(F)
        Q,b,c,ops = quadratic(F, args.ridge, m, .5, sig)
        labels, dirs = eigenstructure_directions(Q, discriminant_matrix(Q,b,c))
        for idx in DIRECTION_INDICES:
            name, v = labels[idx], dirs[idx]
            support[name][1] += 1
            rr = roots(Q,b,c,v)
            if not rr: continue
            support[name][0] += 1
            for j,t in enumerate(rr):
                branches["near" if j == 0 else "far"] += 1
                key, _, crit = bands(Q,b,t,v)
                s = strata[key]; s["points"] += 1; s["crit"].append(crit)
                mu = m + t * (F[:,3:6] @ v)
                y = generate(F,rng,mu,.5,sig,1)[0]
                Tobs = stat(y[None,:],ops)[0]
                mh,rh,sh = est_nuisance(F,y)
                Qh,bh,ch,_ = quadratic(F,args.ridge,mh,rh,sh)
                er = roots(Qh,bh,ch,v)
                if not er: continue
                et = er[0]; s["est"] += 1; s["rr"].append(et/t)
                muhat = mh + et * (F[:,3:6] @ v)
                pe = (1 + np.count_nonzero(stat(generate(F,rng,muhat,rh,sh,args.bootstrap),ops) >= Tobs)) / (args.bootstrap+1)
                po = (1 + np.count_nonzero(stat(generate(F,rng,mu,.5,sig,args.bootstrap),ops) >= Tobs)) / (args.bootstrap+1)
                s["pes"].append(pe); s["por"].append(po)
                s["es_rej"] += int(pe < ALPHA); s["or_rej"] += int(po < ALPHA)
    out = {}
    for k,s in sorted(strata.items()):
        n = len(s["pes"])
        out[k] = {
            "points": s["points"],
            "estimated_root_support": s["est"]/s["points"] if s["points"] else None,
            "median_root_ratio": float(np.median(s["rr"])) if s["rr"] else None,
            "oracle_reject_rate": s["or_rej"]/n if n else None,
            "estimated_nuisance_reject_rate": s["es_rej"]/n if n else None,
            "mean_oracle_p": float(np.mean(s["por"])) if s["por"] else None,
            "mean_estimated_p": float(np.mean(s["pes"])) if s["pes"] else None,
            "median_criticality": float(np.median(s["crit"])) if s["crit"] else None,
        }
    return {
        "epsilon": eps, "paths": args.paths, "bootstrap": args.bootstrap,
        "directions": len(DIRECTION_INDICES), "branch_counts": dict(branches),
        "direction_support": {k: {"supported":v[0],"attempted":v[1],"support_rate":v[0]/v[1]} for k,v in support.items()},
        "strata": out,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=30)
    ap.add_argument("--bootstrap", type=int, default=39)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261060)
    ap.add_argument("--epsilons", type=float, nargs="+", default=[1e-3,1e-4])
    a = ap.parse_args()
    out = {
        "schema":"rh006-null-surface-eigenstructure-nuisance-diagnostic/v1",
        "status":"research-diagnostic-only","applicability":"not-approved-for-execution",
        "selection":"stop-assumption-failure","formal_inference":False,
        "mechanics_scope":"independent Python estimator-mechanics mirror; not the Symthaea Rust implementation",
        "frozen_direction_design":"Q principal axes ±, discriminant-matrix principal axes ±, predeclared weak direction ±",
        "stratification":{
            "curvature":"|u'Q u|/lambda_max(|Q|): high >=1e-3; intermediate [1e-6,1e-3); near_flat <1e-6",
            "root_radius":"|t| <=1 local; 1<|t|<=100 moderate; >100 remote",
            "surface_criticality":"critical when ||2Qx+b||/(||b||+2||Q||_2|t|) < 1e-4",
            "branches":"both positive roots retained; near/far are ordering labels only",
        },
        "results":[scenario(a,e) for e in a.epsilons],
        "gates":{"null_surface_support":"required","curvature_stratification":"executed-diagnostic-only",
                 "estimated_nuisance_by_stratum":"executed-diagnostic-only","local_null_radius_policy":"not-yet-predeclared",
                 "uniform_size_control":"not-established","rust_implementation_validation":"not-established"},
        "nonclaims":["No formal p-value or confidence interval.","No least-favorable critical value.","No empirical validation.","No uniform size guarantee.","No claim that remote null roots are scientifically interchangeable with local roots."],
    }
    payload=json.dumps(out,sort_keys=True,separators=(",",":"))
    out["payload_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
