#!/usr/bin/env python3
"""
RH-006 oracle restricted-DGP nested rolling bootstrap benchmark.

Independent estimator-mechanics reimplementation of the RH-006 rolling estimator:
fixed disjoint held-out blocks, training-only standardization, fixed ridge
with an unpenalized intercept, nested NonRelationalContext vs
RelationalAugmented fits, origin-level squared-loss differences, and a
fixed Bartlett studentizer.

This is a research diagnostic. It is not the Symthaea Rust implementation
and does not authorize formal inference.
"""
import argparse, hashlib, json, math
import numpy as np

TRAIN, TEST, GAP, ORIGINS, STEP = 48, 16, 4, 24, 16
RIDGE, K, ALPHA = 1e-8, 3, 0.05
N = TRAIN + GAP + TEST + (ORIGINS - 1) * STEP

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
    x = np.empty(n)
    eps = rng.rademacher(n)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = rho * x[i-1] + eps[i]
    return x

def generate_features(n, rng, rho=0.35):
    raw = np.column_stack([ar_series(n, rng, rho) for _ in range(7)])
    def n01(x):
        return np.clip(0.5 + 0.18*x, 0.0, 1.0)
    a = n01(raw[:,0]); b = n01(raw[:,1]); common = n01(raw[:,2])
    alignment = n01(0.65*raw[:,3] + 0.35*raw[:,4])
    a2b = n01(raw[:,5]); b2a = n01(raw[:,6])
    turn = n01(0.45*raw[:,4] + 0.55*raw[:,5])
    return np.column_stack([a,b,alignment,a2b,b2a,turn,common])

def generate_outcome(F, rng, rho_err, heteroskedastic):
    a,b,alignment,a2b,b2a,turn,common = F.T
    mu = 0.15 + 0.45*a + 0.25*b + 0.20*common + 0.30*alignment
    sigma = 0.08 * (1.0 + 0.7*(common-0.5)) if heteroskedastic else np.full(len(F), 0.08)
    eps = rng.rademacher(len(F))
    e = np.empty(len(F))
    e[0] = sigma[0] * eps[0]
    for i in range(1, len(F)):
        e[i] = rho_err*e[i-1] + sigma[i]*eps[i]
    return mu + e, mu

def op_matrix(Xtr, Xte):
    mu = Xtr.mean(axis=0)
    sc = Xtr.std(axis=0)
    sc = np.where(sc <= 1e-12, 1.0, sc)
    Atr = np.column_stack([np.ones(len(Xtr)), (Xtr-mu)/sc])
    Ate = np.column_stack([np.ones(len(Xte)), (Xte-mu)/sc])
    M = Atr.T @ Atr
    M[1:,1:] += RIDGE*np.eye(Xtr.shape[1])
    return Ate @ np.linalg.solve(M, Atr.T)

def build_ops(F):
    nr = [0,1,2,6]
    rel = [0,1,2,3,4,5,6]
    ops = []
    for o in range(ORIGINS):
        s=o*STEP; te=s+TRAIN+GAP
        ops.append((
            op_matrix(F[s:s+TRAIN,nr], F[te:te+TEST,nr]),
            op_matrix(F[s:s+TRAIN,rel], F[te:te+TEST,rel])
        ))
    return ops

def studentized(D):
    D = np.asarray(D, float)
    m = D.mean()
    lrv = np.mean((D-m)**2)
    for k in range(1, K+1):
        g = np.mean((D[k:]-m)*(D[:-k]-m))
        lrv += 2*(1-k/(K+1))*g
    return 0.0 if lrv <= 1e-14 else math.sqrt(len(D))*m/math.sqrt(lrv)

def observed_stat(F, y, ops):
    D = []
    for o,(Wnr,Wrel) in enumerate(ops):
        s=o*STEP; te=s+TRAIN+GAP
        ytr=y[s:s+TRAIN]; yte=y[te:te+TEST]
        pn=Wnr@ytr; pr=Wrel@ytr
        D.append(np.mean((pn-yte)**2-(pr-yte)**2))
    D = np.asarray(D)
    return studentized(D), float(D.mean())

def oracle_pvalue(F, y, mu, ops, rng, rho_err, heteroskedastic, B):
    Tobs, Dbar = observed_stat(F, y, ops)
    common = F[:,-1]
    signs = rng.rademacher(B*len(y)).reshape(B,len(y))
    sigma = 0.08*(1.0+0.7*(common-0.5)) if heteroskedastic else np.full(len(y), 0.08)
    e = np.empty_like(signs)
    e[:,0] = sigma[0] * signs[:,0]
    for i in range(1, len(y)):
        e[:,i] = rho_err*e[:,i-1] + sigma[i]*signs[:,i]
    Yb = mu[None,:] + e
    DT = np.empty((B, ORIGINS))
    for o,(Wnr,Wrel) in enumerate(ops):
        s=o*STEP; te=s+TRAIN+GAP
        pn = Yb[:,s:s+TRAIN] @ Wnr.T
        pr = Yb[:,s:s+TRAIN] @ Wrel.T
        yte = Yb[:,te:te+TEST]
        DT[:,o] = np.mean((pn-yte)**2-(pr-yte)**2, axis=1)
    m = DT.mean(axis=1)
    lrv = np.mean((DT-m[:,None])**2, axis=1)
    for k in range(1,K+1):
        lrv += 2*(1-k/(K+1))*np.mean(
            (DT[:,k:]-m[:,None])*(DT[:,:-k]-m[:,None]), axis=1
        )
    Tb = np.where(lrv>1e-14, np.sqrt(ORIGINS)*m/np.sqrt(lrv), 0.0)
    p = (1 + int(np.count_nonzero(Tb >= Tobs))) / (B+1)
    return p,Tobs,Dbar

def run_scenario(rho_err, hetero, mc, B, seed):
    rng = SplitMix64(seed)
    ps=[]; ts=[]; ds=[]
    for _ in range(mc):
        F = generate_features(N, rng)
        y, mu = generate_outcome(F, rng, rho_err, hetero)
        ops = build_ops(F)
        p,t,d = oracle_pvalue(F, y, mu, ops, rng, rho_err, hetero, B)
        ps.append(p); ts.append(t); ds.append(d)
    ps=np.asarray(ps)
    re=ps<ALPHA
    size=float(re.mean())
    return {
        "rho": rho_err,
        "heteroskedastic": hetero,
        "reject_count": int(re.sum()),
        "empirical_size": size,
        "monte_carlo_se": math.sqrt(size*(1-size)/mc),
        "mean_p": float(ps.mean()),
        "mean_T": float(np.mean(ts)),
        "mean_D": float(np.mean(ds))
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mc",type=int,default=400)
    ap.add_argument("--bootstrap",type=int,default=199)
    ap.add_argument("--seed",type=int,default=20261008)
    args=ap.parse_args()
    results=[]
    for idx,(rho,het) in enumerate(((0.0,False),(0.5,False),(0.8,False),(0.5,True))):
        results.append(run_scenario(rho,het,args.mc,args.bootstrap,args.seed+idx*1000))
    out={
        "schema":"rh006-oracle-restricted-dgp-refit-bootstrap-executed/v2",
        "status":"research-diagnostic-only",
        "faithfulness_scope":"independent-estimator-mechanics-reimplementation; not the Symthaea Rust implementation",
        "rng":"SplitMix64 v1 + Rademacher innovations",
        "geometry":{"train":TRAIN,"test":TEST,"gap":GAP,"origins":ORIGINS,"step":STEP,"horizon":1.0,"ridge":RIDGE,"bartlett_lag":K,"alpha":ALPHA,"monte_carlo_reps":args.mc,"bootstrap_reps":args.bootstrap},
        "null":{"restriction":"beta_relational = 0","feature_conditioning":"fixed feature path in bootstrap","dgp_parameters":"oracle-known","full_refit_each_bootstrap":True},
        "scenarios":results,
        "interpretation":{
            "finding":"Oracle conditional-on-features bootstrap with full nested rolling refits was executed independently. The observed pilot size behavior is a diagnostic only.",
            "not_a_validity_proof":True,
            "formal_inference":False,
            "applicability":"not-approved-for-execution",
            "selection":"stop-assumption-failure"
        }
    }
    raw=json.dumps(out,sort_keys=True,separators=(",",":"))
    out["payload_sha256"]=hashlib.sha256(raw.encode()).hexdigest()
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
