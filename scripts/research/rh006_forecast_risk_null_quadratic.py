#!/usr/bin/env python3
"""
Finite-sample RH-006 forecast-risk-null quadratic calibration.

For a fixed feature path and fixed i.i.d. error variance, the expected
forecast-MSE difference between NonRelationalContext and RelationalAugmented
is quadratic in a scalar amplitude multiplying a predeclared relational
feature direction.

Research diagnostic only; not the Symthaea Rust implementation and not a
formal inference procedure.
"""
import argparse, hashlib, json, math
import numpy as np

TRAIN, TEST, GAP, ORIGINS, STEP = 48, 16, 4, 24, 16
RIDGE, SIGMA2 = 1e-8, 0.08**2
N = TRAIN + GAP + TEST + (ORIGINS - 1) * STEP

class SplitMix64:
    def __init__(self, seed: int):
        self.state = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    def rademacher(self, size: int):
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
        x[i] = rho*x[i-1] + eps[i]
    return x

def generate_features(n, rng, rho=0.35):
    raw = np.column_stack([ar_series(n, rng, rho) for _ in range(7)])
    def n01(x):
        return np.clip(0.5 + 0.18*x, 0.0, 1.0)
    a=n01(raw[:,0]); b=n01(raw[:,1]); common=n01(raw[:,2])
    alignment=n01(0.65*raw[:,3]+0.35*raw[:,4])
    a2b=n01(raw[:,5]); b2a=n01(raw[:,6])
    turn=n01(0.45*raw[:,4]+0.55*raw[:,5])
    return np.column_stack([a,b,alignment,a2b,b2a,turn,common])

def op_matrix(Xtr, Xte):
    mu=Xtr.mean(0)
    sc=np.where(Xtr.std(0)<=1e-12,1.0,Xtr.std(0))
    Atr=np.column_stack([np.ones(len(Xtr)),(Xtr-mu)/sc])
    Ate=np.column_stack([np.ones(len(Xte)),(Xte-mu)/sc])
    M=Atr.T@Atr
    M[1:,1:]+=RIDGE*np.eye(Xtr.shape[1])
    return Ate@np.linalg.solve(M,Atr.T)

def quadratic_for_path(F):
    a,b,alignment,a2b,b2a,turn,common=F.T
    mu0=0.15+0.45*a+0.25*b+0.20*common+0.30*alignment
    Z=F[:,[3,4,5]]
    direction=np.ones(3)/math.sqrt(3)
    ops=[]
    for o in range(ORIGINS):
        s=o*STEP; te=s+TRAIN+GAP
        ops.append((op_matrix(F[s:s+TRAIN,[0,1,2,6]],F[te:te+TEST,[0,1,2,6]]),
                    op_matrix(F[s:s+TRAIN,[0,1,2,3,4,5,6]],F[te:te+TEST,[0,1,2,3,4,5,6]])))

    def expected_delta(kappa):
        mu=mu0+Z@(kappa*direction)
        total=0.0
        for o,(Wnr,Wrel) in enumerate(ops):
            s=o*STEP; te=s+TRAIN+GAP
            mtr=mu[s:s+TRAIN]; mte=mu[te:te+TEST]
            bnr=Wnr@mtr-mte; br=Wrel@mtr-mte
            mse_nr=np.mean(bnr*bnr)+SIGMA2*(np.mean(np.sum(Wnr*Wnr,axis=1))+1.0)
            mse_rel=np.mean(br*br)+SIGMA2*(np.mean(np.sum(Wrel*Wrel,axis=1))+1.0)
            total += mse_nr-mse_rel
        return total/ORIGINS

    c=expected_delta(0.0)
    dp=expected_delta(0.25); dm=expected_delta(-0.25)
    A=(dp+dm-2*c)/(2*0.25**2)
    B=(dp-dm)/0.5
    roots=np.roots([A,B,c]) if abs(A)>1e-14 else np.roots([B,c])
    positive=[float(r.real) for r in roots if abs(r.imag)<1e-10 and r.real>0]
    return A,B,c,positive

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--paths",type=int,default=50)
    ap.add_argument("--seed",type=int,default=20261008)
    args=ap.parse_args()
    rows=[]
    for i in range(args.paths):
        rng=SplitMix64(args.seed+i)
        F=generate_features(N,rng)
        A,B,C,roots=quadratic_for_path(F)
        rows.append({"seed":args.seed+i,"A":A,"B":B,"C":C,"positive_roots":roots})
    roots=[r for row in rows for r in row["positive_roots"]]
    Cs=[row["C"] for row in rows]
    out={
        "schema":"rh006-forecast-risk-null-quadratic-v1",
        "status":"research-diagnostic-only",
        "faithfulness_scope":"independent-estimator-mechanics reimplementation; not the Symthaea Rust implementation",
        "geometry":{"train":TRAIN,"test":TEST,"gap":GAP,"origins":ORIGINS,"step":STEP,"ridge":RIDGE,"sigma2":SIGMA2,"paths":args.paths},
        "null_direction":"equal-amplitude direction over [a_to_b, b_to_a, turn_taking]",
        "method":"conditional fixed-design expected MSE difference under i.i.d. error variance",
        "rows":rows,
        "summary":{
            "positive_root_count":len(roots),
            "positive_root_mean":float(np.mean(roots)),
            "positive_root_median":float(np.median(roots)),
            "positive_root_min":float(np.min(roots)),
            "positive_root_max":float(np.max(roots)),
            "delta_at_zero_mean":float(np.mean(Cs)),
            "delta_at_zero_min":float(np.min(Cs)),
            "delta_at_zero_max":float(np.max(Cs))
        },
        "interpretation":{
            "key_finding":"The finite-sample equal-risk null can occur at a nonzero incremental relational signal amplitude because nested estimator variance and omitted-signal bias trade off.",
            "not_a_formal_inference_method":True,
            "next_requirement":"For dependent or heteroskedastic errors replace the i.i.d. covariance term with a declared covariance model and derive the corresponding quadratic risk expression."
        }
    }
    raw=json.dumps(out,sort_keys=True,separators=(",",":"))
    out["payload_sha256"]=hashlib.sha256(raw.encode()).hexdigest()
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
