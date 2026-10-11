#!/usr/bin/env python3
"""RH-006 exact conditional joint-DGP risk surface + topology diagnostic.

Uses the existing RH-006 joint feature/outcome stress mechanics and derives
the conditional finite-sample forecast-risk surface under the shared-shock
DGP. Research diagnostic only; no inference is implemented.
"""
import argparse, hashlib, importlib.util, json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "rh006_joint_feature_outcome_endogeneity_splitmix64.py"
spec = importlib.util.spec_from_file_location("rh006_endog", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load RH-006 endogeneity mechanics")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

OMEGAS = (0., .25, .5, .75, .9)
DIRECTIONS = base.DIRECTIONS

def loading(sig, rho=.5):
    L = np.zeros((base.N, base.N))
    for t in range(base.N):
        L[t, t] = sig[t]
        if t:
            L[t, :t] = (rho ** np.arange(t, 0, -1)) * sig[:t]
    return L

def surface(F, q):
    m, sig = base.mu0(F), base.sig0(F)
    L = loading(sig); S = L @ L.T; ell = L @ q
    Z = F[:, 3:6]
    Q = np.zeros((3,3)); b0 = np.zeros(3); b1 = np.zeros(3)
    c0 = c1 = c2 = 0.
    ops = base.build_ops(F)
    for o, (Wc, Wr) in enumerate(ops):
        s, te = o*base.STEP, o*base.STEP+base.TRAIN+base.GAP
        tr, tt = slice(s,s+base.TRAIN), slice(te,te+base.TEST)
        Hc, Hr = Wc@Z[tr]-Z[tt], Wr@Z[tr]-Z[tt]
        qc, qr = Wc@m[tr]-m[tt], Wr@m[tr]-m[tt]
        ac, ar = Wc@ell[tr]-ell[tt], Wr@ell[tr]-ell[tt]
        Q += (Hc.T@Hc-Hr.T@Hr)/base.TEST
        b0 += 2*(Hc.T@qc-Hr.T@qr)/base.TEST
        b1 += 2*(Hc.T@ac-Hr.T@ar)/base.TEST
        c0 += (qc@qc-qr@qr)/base.TEST
        c1 += 2*(qc@ac-qr@ar)/base.TEST
        c2 += (ac@ac-ar@ar)/base.TEST
        tri, tti = np.arange(s,s+base.TRAIN), np.arange(te,te+base.TEST)
        St, Sty = S[np.ix_(tri,tri)], S[np.ix_(tri,tti)]
        vd = (np.trace(Wc@St@Wc.T)-2*np.trace(Wc@Sty)
              -np.trace(Wr@St@Wr.T)+2*np.trace(Wr@Sty))/base.TEST
        c0 += vd; c2 -= vd
    z = 1/base.ORIGINS
    return Q*z,b0*z,b1*z,c0*z,c1*z,c2*z,ops

def coeff(poly,w):
    Q,b0,b1,c0,c1,c2,_ = poly
    return Q,b0+w*b1,c0+w*c1+w*w*c2

def delta(poly,w,gamma):
    Q,b,c = coeff(poly,w)
    return float(gamma@Q@gamma+b@gamma+c)

def roots(Q,b,c,u):
    A=float(u@Q@u); B=float(b@u); D=B*B-4*A*c
    if D < 0: return []
    if abs(A) <= 1e-14:
        t=-c/B if abs(B)>1e-14 else float("nan")
        return [float(t)] if np.isfinite(t) and t>0 else []
    s=math.sqrt(max(D,0)); rr=[(-B-s)/(2*A),(-B+s)/(2*A)]
    return sorted(float(t) for t in rr if np.isfinite(t) and t>0)

def run(a):
    rng=base.SM(a.seed)
    paths=[base.feature_path(rng) for _ in range(a.paths)]
    rows=[]; topo={str(w):[] for w in OMEGAS}; czeros=[]; mc=[]; maxerr=0.
    for i,(F,qinnov) in enumerate(paths):
        poly=surface(F,qinnov); Q,b0,b1,c0,c1,c2,ops=poly
        rr=np.roots([c2,c1,c0]) if abs(c2)>1e-20 else np.roots([c1,c0])
        pos=sorted(float(x.real) for x in rr if abs(x.imag)<1e-10 and 0<=x.real<=.9)
        if pos: czeros.append(pos[0])
        for w in OMEGAS:
            Qw,bw,cw=coeff(poly,w); Qs=(Qw+Qw.T)/2
            R=np.outer(bw,bw)-4*cw*Qs
            ev=np.linalg.eigvalsh((R+R.T)/2)
            sig=''.join('+' if x>0 else '-' if x<0 else '0' for x in ev)
            topo[str(w)].append((ev,sig))
            for d,u in DIRECTIONS.items():
                r0=roots(Q,b0,c0,u)
                ind=r0[0] if r0 else None
                rw=roots(Qw,bw,cw,u)
                rows.append({
                    "path":i,"direction":d,"omega":w,
                    "discriminant":float((u@bw)**2-4*(u@Qw@u)*cw),
                    "positive_roots":rw,"independence_root":ind,
                    "delta_at_independence_root":None if ind is None else delta(poly,w,ind*u)
                })
        if i<a.mc_paths:
            for d,u in DIRECTIONS.items():
                r0=roots(Q,b0,c0,u)
                if not r0: continue
                mu=base.mu0(F)+r0[0]*(F[:,3:6]@u)
                for w in (0.,.5,.9):
                    analytic=delta(poly,w,r0[0]*u)
                    gr=base.SM(a.seed+1000000+i*1000+int(100*w)+(1 if d=="strong" else 0))
                    sims=[base.loss_diff(base.outcome(gr,F,qinnov,mu,w),F,ops) for _ in range(a.mc_outcomes)]
                    emp=float(np.mean(sims)); err=abs(emp-analytic); maxerr=max(maxerr,err)
                    mc.append({"path":i,"direction":d,"omega":w,"analytic":analytic,"mc_mean":emp,"abs_error":err})
    spectrum={}
    for w,vals in topo.items():
        arr=np.asarray([v[0] for v in vals]); counts={}
        for _,s in vals: counts[s]=counts.get(s,0)+1
        spectrum[w]={"signature_counts":counts,
                     "eigenvalue_quantiles":[np.quantile(arr[:,j],[0,.1,.5,.9,1]).tolist() for j in range(3)]}
    return {
        "schema":"rh006-joint-dgp-risk-surface-cross-moment/v2",
        "status":"research-diagnostic-only","formal_inference":False,
        "applicability":"not-approved-for-execution","selection":"stop-assumption-failure",
        "source_identity":{"base_script":BASE.name,"base_script_sha256":hashlib.sha256(BASE.read_bytes()).hexdigest()},
        "execution":{"paths":a.paths,"seed":a.seed,"omega_cells":list(OMEGAS),
                     "mc_paths":a.mc_paths,"mc_outcomes":a.mc_outcomes,
                     "geometry":{"train":base.TRAIN,"gap":base.GAP,"test":base.TEST,"origins":base.ORIGINS,"step":base.STEP},
                     "ridge":base.RIDGE},
        "surface":{"form":"Delta(gamma;omega)=gamma'Qgamma+(b0+omega*b1)'gamma+c0+omega*c1+omega^2*c2",
                   "conditional_mean":"E[e|q]=omega Lq",
                   "conditional_covariance":"Var(e|q)=(1-omega^2)LL'",
                   "discriminant_matrix":"R(omega)=b(omega)b(omega)'-4c(omega)Q"},
        "topology":{"spectrum":spectrum,
                    "c_zero_omega_quantiles":np.quantile(czeros,[0,.1,.5,.9,1]).tolist(),
                    "interpretation":"R is positive definite before c=0 and has signature (+,-,-) after c=0 in this construction."},
        "directional_summary":rows,
        "mc_validation":{"checks":mc,"max_absolute_error":maxerr},
        "nonclaims":["No formal p-value.","No confidence interval.",
                     "No uniform or least-favorable size guarantee.","No empirical validation.",
                     "No exact-Rust implementation qualification.",
                     "No claim topology or polynomial omega structure is universal across arbitrary endogenous DGPs."]
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--paths",type=int,default=40)
    ap.add_argument("--seed",type=int,default=20261080)
    ap.add_argument("--mc-paths",type=int,default=4)
    ap.add_argument("--mc-outcomes",type=int,default=1000)
    a=ap.parse_args(); out=run(a)
    out["script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    raw=json.dumps(out,sort_keys=True,separators=(",",":"))
    out["payload_sha256"]=hashlib.sha256(raw.encode()).hexdigest()
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
