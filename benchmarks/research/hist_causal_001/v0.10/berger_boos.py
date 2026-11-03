#!/usr/bin/env python3
"""HIST-CAUSAL-001 v0.10 exact finite-grid Berger–Boos nuisance reference.

An independent pilot sample constructs a confidence set C for nuisance theta.
The evaluation sample yields a simple-null p-value p_theta. The adjusted
p-value min(1, sup(theta in C) p_theta + beta) is finite-sample valid when
the confidence set has coverage >= 1-beta. Exact enumeration verifies a
three-point binomial nuisance grid only; this is not a continuous-parameter
implementation or a general six-mechanism solver.
"""
from __future__ import annotations
from fractions import Fraction as F
from math import comb
from typing import Iterable

THETA_GRID = (F(1,4), F(1,2), F(3,4))


def pmf(n: int, k: int, theta: F) -> F:
    if not isinstance(n, int) or n < 0 or not 0 <= k <= n:
        return F(0)
    if not 0 <= theta <= 1:
        raise ValueError("theta must be in [0,1]")
    return F(comb(n,k)) * theta**k * (1-theta)**(n-k)


def support_probs(n: int, theta: F) -> tuple[F,...]:
    probs=tuple(pmf(n,k,theta) for k in range(n+1))
    if sum(probs,F(0)) != 1:
        raise ArithmeticError("binomial kernel failed exact normalization")
    return probs


def exact_two_sided_pvalue(n_pilot: int, observed: int, theta: F) -> F:
    """Probability-ordering two-sided binomial p-value; ties included."""
    probs=support_probs(n_pilot,theta)
    threshold=probs[observed]
    return sum((p for p in probs if p <= threshold),F(0))


def confidence_set(n_pilot: int, x: int, grid: Iterable[F], beta: F) -> tuple[F,...]:
    """Invert exact two-sided tests to obtain a conservative confidence set."""
    if not 0 < beta < 1:
        raise ValueError("beta must be in (0,1)")
    if not 0 <= x <= n_pilot:
        raise ValueError("pilot count outside support")
    grid=tuple(grid)
    if not grid or any(not 0 < t < 1 for t in grid):
        raise ValueError("grid must contain interior probabilities")
    return tuple(theta for theta in grid
                 if exact_two_sided_pvalue(n_pilot,x,theta) > beta)


def lower_tail_pvalue(n_eval: int, y: int, theta: F) -> F:
    """Exact lower-tail p-value for a small-response alternative."""
    if not 0 <= y <= n_eval:
        raise ValueError("evaluation count outside support")
    return sum(support_probs(n_eval,theta)[:y+1],F(0))


def berger_boos_pvalue(n_pilot: int, x: int, n_eval: int, y: int,
                       grid: Iterable[F]=THETA_GRID,
                       beta: F=F(1,100)) -> F:
    """Conservative nuisance-adjusted p-value, with independent pilot data."""
    if n_pilot < 0 or n_eval < 0:
        raise ValueError("sample sizes must be non-negative")
    cset=confidence_set(n_pilot,x,grid,beta)
    # Empty C is possible only on a low-probability pilot event; do not reject.
    if not cset:
        return F(1)
    return min(F(1),max(lower_tail_pvalue(n_eval,y,t) for t in cset)+beta)


def exact_coverage(n_pilot: int, theta_true: F,
                   grid: Iterable[F]=THETA_GRID,
                   beta: F=F(1,100)) -> F:
    return sum((pmf(n_pilot,x,theta_true)
                for x in range(n_pilot+1)
                if theta_true in confidence_set(n_pilot,x,grid,beta)),F(0))


def exact_rejection_probability(n_pilot: int, n_eval: int, theta_true: F,
                                alpha: F=F(1,20), beta: F=F(1,100),
                                grid: Iterable[F]=THETA_GRID) -> F:
    """Exhaustive P(p_BB <= alpha) under a grid null value."""
    total=F(0)
    for x in range(n_pilot+1):
        px=pmf(n_pilot,x,theta_true)
        if not px:
            continue
        for y in range(n_eval+1):
            py=pmf(n_eval,y,theta_true)
            if py and berger_boos_pvalue(n_pilot,x,n_eval,y,grid,beta) <= alpha:
                total += px*py
    return total


def exact_power(n_pilot: int, n_eval: int, theta_alt: F,
                alpha: F=F(1,20), beta: F=F(1,100),
                grid: Iterable[F]=THETA_GRID) -> F:
    """Power under a declared alternative probability outside the null grid."""
    total=F(0)
    for x in range(n_pilot+1):
        px=pmf(n_pilot,x,theta_alt)
        if not px:
            continue
        for y in range(n_eval+1):
            py=pmf(n_eval,y,theta_alt)
            if py and berger_boos_pvalue(n_pilot,x,n_eval,y,grid,beta) <= alpha:
                total += px*py
    return total


def verify(n_pilot: int=20,n_eval: int=20,
           grid: Iterable[F]=THETA_GRID,
           alpha: F=F(1,20),beta: F=F(1,100),
           theta_alt: F=F(1,8)) -> dict:
    grid=tuple(grid)
    rows=[]
    for theta in grid:
        cov=exact_coverage(n_pilot,theta,grid,beta)
        size=exact_rejection_probability(n_pilot,n_eval,theta,alpha,beta,grid)
        rows.append({"theta":str(theta),"confidence_coverage":str(cov),
                     "false_rejection_probability":str(size),
                     "coverage_at_least_1_minus_beta":cov >= 1-beta,
                     "size_at_most_alpha":size <= alpha})
    power=exact_power(n_pilot,n_eval,theta_alt,alpha,beta,grid)
    return {"method":"independent-pilot Berger-Boos adjustment; exact finite nuisance grid",
            "n_pilot":n_pilot,"n_evaluation":n_eval,
            "theta_grid":[str(t) for t in grid],"alpha":str(alpha),"beta":str(beta),
            "alternative_theta":str(theta_alt),"null_checks":rows,
            "minimum_null_coverage":str(min(F(r["confidence_coverage"]) for r in rows)),
            "maximum_null_rejection":str(max(F(r["false_rejection_probability"]) for r in rows)),
            "alternative_power":str(power),
            "claim_ceiling":"exact validity on declared finite theta grid only; not continuous nuisance coverage"}


if __name__ == "__main__":
    import argparse,json
    parser=argparse.ArgumentParser()
    parser.add_argument("--pilot",type=int,default=20)
    parser.add_argument("--evaluation",type=int,default=20)
    args=parser.parse_args()
    print(json.dumps(verify(args.pilot,args.evaluation),indent=2))
