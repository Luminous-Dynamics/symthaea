#!/usr/bin/env python3
"""Independent full-constraint LP oracle for RC-eBH optimality (K<=5 by default).

This checks the expected-rejection optimum claim in Bai & Ramdas (2026),
arXiv:2610.35821, Proposition 4.2(iii), by solving the full closed-eBH
certificate polytope with every nonempty subset inequality explicit.

Offline fixed-family FDR only. This is a finite computational check, not a
proof of the general theorem and not a SupFDR procedure.
"""
from __future__ import annotations
from itertools import combinations
import math
import numpy as np
from scipy.optimize import linprog


def validate_evalues(evalues):
    e=np.asarray(evalues,dtype=np.float64)
    if e.ndim!=1 or e.size<1 or np.any(~np.isfinite(e)) or np.any(e<0):
        raise ValueError("e-values must be a nonempty finite nonnegative vector")
    return e


def greatest_convex_minorant(values):
    y=np.asarray(values,dtype=np.float64)
    if y.ndim!=1 or y.size<2 or np.any(~np.isfinite(y)):
        raise ValueError("GCM needs a finite vector of length >=2")
    hull=[]
    for x,v in enumerate(y,1):
        hull.append((x,float(v)))
        while len(hull)>=3:
            x0,y0=hull[-3];x1,y1=hull[-2];x2,y2=hull[-1]
            sl=(y1-y0)/(x1-x0);sr=(y2-y1)/(x2-x1)
            if sl>sr+1e-14:hull.pop(-2)
            else:break
    out=np.empty_like(y)
    for (x0,y0),(x1,y1) in zip(hull[:-1],hull[1:]):
        slope=(y1-y0)/(x1-x0)
        for x in range(x0,x1+1):out[x-1]=y0+slope*(x-x0)
    return out


def rc_ebh_tails(evalues,alpha):
    e=np.sort(validate_evalues(evalues))[::-1]
    if not math.isfinite(alpha) or not 0<alpha<1:
        raise ValueError("alpha must lie in (0,1)")
    k=e.size
    suffix=np.r_[np.cumsum(e[::-1])[::-1],0.]
    beta=alpha*suffix[:-1]/np.arange(k,0,-1,dtype=float)
    beta[0]=min(1.,beta[0])
    S=greatest_convex_minorant(np.r_[beta,0.])
    r=np.arange(1,k+1,dtype=float)
    q=r*S[:-1]-(r-1)*S[1:]
    if np.any(q < -1e-9) or np.any(q>1+1e-9) or np.any(np.diff(q)>1e-8):
        raise ArithmeticError("RC-eBH returned invalid tails")
    return q


def charges_from_tails(tails):
    q=np.asarray(tails,dtype=float);k=q.size
    if q.ndim!=1 or k<1 or np.any(q < -1e-9) or np.any(q>1+1e-9) or np.any(np.diff(q)>1e-8):
        raise ValueError("invalid rejection-count tails")
    p=np.r_[1-q[0],q[:-1]-q[1:],q[-1]]
    if np.any(p < -1e-8):
        raise ArithmeticError("negative rejection probability")
    return np.array([sum(p[n]/n for n in range(r,k+1)) for r in range(1,k+1)])


def verify_all_subset_constraints(evalues,alpha,tails,tolerance=1e-9):
    e=np.sort(validate_evalues(evalues))[::-1]
    c=charges_from_tails(tails);k=len(e)
    worst=(-math.inf,None)
    for t in range(1,k+1):
        for subset in combinations(range(k),t):
            idx=np.asarray(subset,dtype=int)
            violation=float(c[idx].sum()-alpha*e[idx].mean())
            if violation>worst[0]:worst=(violation,subset)
    return {"valid":worst[0]<=tolerance,"max_violation":worst[0],"worst_subset":worst[1]}


def full_all_subsets_lp(evalues,alpha):
    """Optimize expected rejection count over the complete closed-eBH menu.

    Variables S_r=sum_{j>=r} c_j, S_{K+1}=0; c_r=S_r-S_{r+1};
    q_r=r*S_r-(r-1)*S_{r+1}. Structural constraints give q a valid
    nonincreasing tail sequence. Every nonempty subset T contributes
    sum_{r in T}c_r <= alpha*mean(e_[r] : r in T).
    """
    e=np.sort(validate_evalues(evalues))[::-1];k=e.size
    if not math.isfinite(alpha) or not 0<alpha<1:
        raise ValueError("alpha must lie in (0,1)")
    A=[];b=[]
    # Probability that kappa>0 cannot exceed one.
    row=np.zeros(k);row[0]=1.;A.append(row);b.append(1.)
    # Nonincreasing cumulative charges S.
    for j in range(k-1):
        row=np.zeros(k);row[j]=-1.;row[j+1]=1.
        A.append(row);b.append(0.)
    # Convex S (equivalent to nonnegative rejection-count masses).
    for j in range(k):
        row=np.zeros(k);row[j]=-1.
        if j+1<k:row[j+1]=2.
        if j+2<k:row[j+2]=-1.
        A.append(row);b.append(0.)
    # All 2^K-1 intersection constraints, not just suffix subsets.
    for t in range(1,k+1):
        for subset in combinations(range(k),t):
            row=np.zeros(k)
            for i in subset:
                row[i]+=1.
                if i+1<k:row[i+1]-=1.
            A.append(row);b.append(float(alpha*np.mean(e[list(subset)])))
    # E[kappa]=sum_r q_r=S_1+2*sum_{r>=2} S_r.
    objective=np.full(k,-2.,dtype=float);objective[0]=-1.
    sol=linprog(objective,A_ub=np.asarray(A),b_ub=np.asarray(b),
                bounds=[(0.,None)]*k,method="highs",
                options={"primal_feasibility_tolerance":1e-10,
                         "dual_feasibility_tolerance":1e-10})
    if not sol.success or sol.x is None:
        raise RuntimeError(sol.message)
    S=np.r_[sol.x,0.];r=np.arange(1,k+1,dtype=float)
    q=r*S[:-1]-(r-1)*S[1:]
    p=np.r_[1-q[0],q[:-1]-q[1:],q[-1]]
    if np.any(q < -1e-8) or np.any(q>1+1e-8) or np.any(np.diff(q)>1e-8) or np.any(p < -1e-8):
        raise ArithmeticError("full LP returned an invalid randomized rule")
    return {"tails":q,"probabilities":p,"cumulative_charges":S[:-1],
            "objective":float(q.sum()),"solver_objective":float(-sol.fun),
            "constraint_count":len(b)}


def compare_rc_to_full_lp(evalues,alpha,tolerance=1e-8):
    qrc=rc_ebh_tails(evalues,alpha)
    lp=full_all_subsets_lp(evalues,alpha)
    feasible=verify_all_subset_constraints(evalues,alpha,qrc)
    gap=float(lp["objective"]-qrc.sum())
    return {"rc_feasible":feasible["valid"],"max_rc_subset_violation":feasible["max_violation"],
            "rc_objective":float(qrc.sum()),"lp_objective":lp["objective"],
            "objective_gap":gap,"objective_match":abs(gap)<=tolerance,
            "rc_tails":qrc.tolist(),"lp_tails":lp["tails"].tolist(),
            "tail_max_abs_difference":float(np.max(np.abs(qrc-lp["tails"]))),
            "constraint_count":lp["constraint_count"]}
