#!/usr/bin/env python3
"""Capacity-aware randomized decisions over the full closed-eBH certified menu.

Offline fixed-family FDR only. The cutting-plane LP uses an exact subset
separation oracle; suffix constraints are warm starts, not a substitute for
checking all intersections.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

SCHEMA="symthaea-utility-recertification-v1"


def validate_e(e):
    x=np.asarray(e,dtype=np.float64)
    if x.ndim!=1 or x.size<1 or np.any(~np.isfinite(x)) or np.any(x<0):
        raise ValueError("e-values must be a nonempty finite nonnegative vector")
    return x


def validate_alpha(a):
    if not math.isfinite(a) or not 0<a<1:
        raise ValueError("alpha must lie in (0,1)")


def gcm(y):
    y=np.asarray(y,dtype=float)
    if y.ndim!=1 or y.size<2 or np.any(~np.isfinite(y)):
        raise ValueError("GCM input must be a finite vector of length >=2")
    hull=[]
    for x,v in enumerate(y,1):
        hull.append((x,float(v)))
        while len(hull)>=3:
            x0,y0=hull[-3];x1,y1=hull[-2];x2,y2=hull[-1]
            if (y1-y0)/(x1-x0)>(y2-y1)/(x2-x1)+1e-14:
                hull.pop(-2)
            else: break
    out=np.empty_like(y)
    for (x0,y0),(x1,y1) in zip(hull[:-1],hull[1:]):
        slope=(y1-y0)/(x1-x0)
        for x in range(x0,x1+1): out[x-1]=y0+slope*(x-x0)
    return out


def rc_tails(e,alpha):
    s=np.sort(validate_e(e))[::-1];validate_alpha(alpha);k=s.size
    tail=np.r_[np.cumsum(s[::-1])[::-1],0.]
    beta=alpha*tail[:-1]/np.arange(k,0,-1,dtype=float)
    beta[0]=min(1.,beta[0])
    S=gcm(np.r_[beta,0.])
    r=np.arange(1,k+1,dtype=float)
    q=r*S[:-1]-(r-1)*S[1:]
    if np.any(q < -1e-9) or np.any(q>1+1e-9) or np.any(np.diff(q)>1e-8):
        raise ArithmeticError("invalid RC-eBH tails")
    return q


def capacity_utilities(e,B):
    x=validate_e(e)
    if not isinstance(B,(int,np.integer)) or B<0: raise ValueError("B must be nonnegative integer")
    u=np.zeros(x.size);u[np.argsort(-x,kind="stable")[:min(B,x.size)]]=1.
    return u


def _coefficients(u):
    k=len(u);a=np.empty(k);a[0]=u[0]
    for j in range(1,k):
        r=j+1;a[j]=r*u[j]-(r-2)*u[j-1]
    return a


def distribution(q):
    q=np.asarray(q,dtype=float)
    if q.ndim!=1 or q.size<1 or np.any(~np.isfinite(q)):raise ValueError("invalid tails")
    p=np.r_[1-q[0],q[:-1]-q[1:],q[-1]]
    if np.any(p < -1e-8) or abs(float(p.sum())-1)>1e-8: raise ArithmeticError("invalid rejection mass")
    return p


def charges(q):
    p=distribution(q);k=len(q)
    return np.array([sum(p[n]/n for n in range(r,k+1)) for r in range(1,k+1)])


def solve_utility_recertification(evalues,utilities,alpha,tol=1e-10,max_iterations=1000):
    """Maximize expected nonnegative rank utility over all closed-eBH menus.

    Variables S_r=sum_{j>=r} c_j, S_{K+1}=0, c_r=S_r-S_{r+1},
    q_r=r*S_r-(r-1)*S_{r+1}. For every T, sum_{r in T}c_r <=
    alpha*mean(e_T). A separation oracle finds the most violated subset
    at each cardinality, adding exact cuts until none remain.
    """
    e=validate_e(evalues);u0=np.asarray(utilities,dtype=float);validate_alpha(alpha)
    if u0.shape!=e.shape or np.any(~np.isfinite(u0)) or np.any(u0<0):
        raise ValueError("utilities must match e-values and be finite nonnegative")
    if tol<=0 or not math.isfinite(tol) or max_iterations<1:raise ValueError("invalid solver parameters")
    k=len(e);order=np.argsort(-e,kind="stable");s=e[order];u=u0[order]
    objective=_coefficients(u)
    sums=np.r_[np.cumsum(s[::-1])[::-1],0.]
    beta=alpha*sums[:-1]/np.arange(k,0,-1,dtype=float);beta[0]=min(1.,beta[0])
    A=[];b=[]
    # Monotone S: S[j+1]-S[j] <= 0.
    for j in range(k-1):
        row=np.zeros(k);row[j+1]=1.;row[j]=-1.;A.append(row);b.append(0.)
    # Convex S, with S[K+1]=0.
    for j in range(k):
        row=np.zeros(k);row[j]=-1.
        if j+1<k:row[j+1]=2.
        if j+2<k:row[j+2]=-1.
        A.append(row);b.append(0.)
    # Initial suffix cuts.
    for j in range(k):
        row=np.zeros(k);row[j]=1.;A.append(row);b.append(float(beta[j]))
    cuts=set()
    def add_cut(idx):
        key=tuple(sorted(int(i) for i in idx))
        if key in cuts:return False
        cuts.add(key);row=np.zeros(k)
        for r in key:
            row[r]+=1.
            if r+1<k:row[r+1]-=1.
        A.append(row);b.append(float(alpha*np.mean(s[list(key)])))
        return True
    for t in range(1,k+1):add_cut(range(k-t,k))
    for iteration in range(1,max_iterations+1):
        res=linprog(-objective,A_ub=np.asarray(A),b_ub=np.asarray(b),
                    bounds=[(0.,None)]*k,method="highs",
                    options={"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10})
        if not res.success or res.x is None:raise RuntimeError(res.message)
        S=np.r_[np.asarray(res.x,dtype=float),0.]
        c=S[:-1]-S[1:];new=0;maxv=0.
        for t in range(1,k+1):
            score=c-alpha*s/t
            idx=np.argsort(-score,kind="stable")[:t]
            v=float(score[idx].sum());maxv=max(maxv,v)
            if v>tol:new+=int(add_cut(idx))
        if new==0:
            if maxv>10*tol:raise ArithmeticError(f"duplicate violated subset cut: {maxv}")
            r=np.arange(1,k+1,dtype=float)
            q=r*S[:-1]-(r-1)*S[1:]
            if np.any(q < -1e-8) or np.any(q>1+1e-8) or np.any(np.diff(q)>1e-8):
                raise ArithmeticError("LP returned invalid rejection tails")
            p=distribution(q);c=charges(q)
            return {"tails":q,"probabilities":p,"order":order,"sorted_evalues":s,
                    "sorted_utilities":u,"cumulative_charges":S,"per_rank_charges":c,
                    "objective_value":float(np.dot(u,q)),"expected_rejections":float(q.sum()),
                    "iterations":iteration,"number_of_cuts":len(cuts),
                    "max_subset_violation":maxv,"solver_objective":float(-res.fun)}
    raise RuntimeError("cutting plane did not converge")


def verify_all_subsets(e,alpha,q,max_k=16):
    from itertools import combinations
    x=validate_e(e);validate_alpha(alpha)
    if len(x)>max_k:raise ValueError("K exceeds exhaustive verifier limit")
    order=np.argsort(-x,kind="stable");s=x[order];c=charges(q)
    worst=(-math.inf,())
    for t in range(1,len(x)+1):
        for subset in combinations(range(len(x)),t):
            idx=np.asarray(subset,dtype=int)
            v=float(c[idx].sum()-alpha*s[idx].mean())
            if v>worst[0]:worst=(v,subset)
    return {"valid":worst[0]<=1e-8,"max_violation":worst[0],"worst_subset":worst[1]}


def sample_count(q,u):
    q=np.asarray(q,dtype=float)
    if not 0<=u<=1 or np.any(np.diff(q)>1e-8):raise ValueError("invalid tails/uniform")
    hit=np.flatnonzero(u<=q)
    return int(hit[-1]+1) if hit.size else 0


def simulate_cell(repetitions,K,qtrue,mu,rho,alpha,B,seed):
    if repetitions<1 or K<2 or not 0<=qtrue<=K or not 0<=rho<1 or not 0<alpha<1:
        raise ValueError("invalid simulation settings")
    rng=np.random.default_rng(seed);common=rng.standard_normal((repetitions,1))
    noise=rng.standard_normal((repetitions,K))
    z=math.sqrt(rho)*common+math.sqrt(1-rho)*noise
    null=np.ones(K,dtype=bool)
    if qtrue:null[-qtrue:]=False;z[:,-qtrue:]+=mu
    E=np.exp(2*z-2);U=rng.random(repetitions)
    raw={m:{x:[] for x in ("fdp","power","k","action","true_action","expected_action")} for m in ("RC","LP")}
    times=[];cuts=[]
    for i in range(repetitions):
        e=E[i];order=np.argsort(-e,kind="stable");nullsort=null[order]
        q0=rc_tails(e,alpha);u=capacity_utilities(e,B)
        t0=time.perf_counter();sol=solve_utility_recertification(e,u,alpha)
        times.append(time.perf_counter()-t0);cuts.append(sol["number_of_cuts"])
        for name,qtail in (("RC",q0),("LP",sol["tails"])):
            kappa=sample_count(qtail,float(U[i]));f=int(nullsort[:kappa].sum())
            action=min(kappa,B);raw[name]["fdp"].append(f/max(kappa,1))
            raw[name]["power"].append((kappa-f)/max(qtrue,1) if qtrue else 0.)
            raw[name]["k"].append(kappa);raw[name]["action"].append(action)
            raw[name]["true_action"].append(int(np.sum(~nullsort[:action])))
            raw[name]["expected_action"].append(float(qtail[:B].sum()))
    def stat(a):
        x=np.asarray(a,dtype=float);m=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(repetitions))
        return {"mean":m,"ci95":[m-1.96*se,m+1.96*se]}
    metrics={}
    for name in raw:
        metrics[name]={"FDR":stat(raw[name]["fdp"]),"power":stat(raw[name]["power"]),
            "mean_rejections":float(np.mean(raw[name]["k"])),
            "actionable":stat(raw[name]["action"]),"true_actionable":stat(raw[name]["true_action"]),
            "conditional_expected_actionable":stat(raw[name]["expected_action"])}
    delta=np.asarray(raw["LP"]["action"])-np.asarray(raw["RC"]["action"])
    dtrue=np.asarray(raw["LP"]["true_action"])-np.asarray(raw["RC"]["true_action"])
    return {"protocol":{"repetitions":repetitions,"K":K,"q":qtrue,"mu":mu,"rho":rho,
            "alpha":alpha,"B":B,"seed":seed},
        "methods":metrics,"paired":{"actionable_gain":stat(delta),"true_actionable_gain":stat(dtrue),
        "mean_rejection_difference":float(np.mean(raw["LP"]["k"])-np.mean(raw["RC"]["k"]))},
        "solver":{"mean_ms":1000*float(np.mean(times)),"p95_ms":1000*float(np.quantile(times,.95)),
                  "mean_cuts":float(np.mean(cuts)),"max_cuts":int(np.max(cuts))}}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    cases=[(200,10,2.5,0.,10,20261032),(200,50,2.5,0.,10,20261033),
           (200,90,2.,0.,10,20261034),(200,50,2.5,.5,10,20261030),
           (200,50,2.5,.5,20,20261035),(500,0,2.5,0.,10,20261036)]
    result={"schema":SCHEMA,"status":"EXPLORATORY_LOCAL_PILOT_NOT_A_SOTA_CLAIM",
            "objective":"maximize E[min(kappa,B)] over the full closed-eBH menu",
            "cells":[simulate_cell(n,100,q,mu,rho,.1,B,seed) for n,q,mu,rho,B,seed in cases],
            "limitations":["small repetitions: 200 alternative, 500 all-null",
                "Gaussian equicorrelation only","capacity is slots, not truth","all-null interval crosses alpha",
                "different objective from expected total rejections","offline FDR only; no SupFDR claim"]}
    raw=(json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    Path(args.output).write_bytes(raw)
    print(json.dumps({"output":args.output,"bytes":len(raw),
        "sha256":hashlib.sha256(raw).hexdigest(),"cells":len(result["cells"])},indent=2))
