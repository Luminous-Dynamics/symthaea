#!/usr/bin/env python3
"""Global utility optimization over the exact finite-exchangeability CRC polytope.

For linear losses L_r(z,a)=g[r,z]*(1-a), this solves one LP over every
calibration-multiset action a_D and all augmented-multiset risk constraints.
The weights only define the utility objective; hard constraints retain
validity under every finite exchangeable law.
"""
from __future__ import annotations
import math
from typing import Iterable
import numpy as np
from scipy.optimize import linprog


def count_vectors(total: int, types: int) -> list[tuple[int, ...]]:
    if total < 0 or types < 1:
        raise ValueError("total must be nonnegative and types positive")
    if types == 1:
        return [(total,)]
    out=[]
    for first in range(total+1):
        for tail in count_vectors(total-first,types-1):
            out.append((first,)+tail)
    return out


def multinomial_state_weights(n: int, probabilities: Iterable[float]) -> tuple[list[tuple[int,...]],np.ndarray]:
    p=np.asarray(list(probabilities),dtype=np.float64)
    if p.ndim!=1 or p.size<2 or np.any(~np.isfinite(p)) or np.any(p<0) or not np.isclose(p.sum(),1.0):
        raise ValueError("probabilities must be a finite simplex vector")
    states=count_vectors(n,p.size)
    weights=[]
    for d in states:
        coeff=math.factorial(n)
        for count in d:
            coeff/=math.factorial(count)
        weight=coeff
        for count,prob in zip(d,p):
            weight*=prob**count
        weights.append(weight)
    weights=np.asarray(weights,dtype=np.float64)
    if not np.isclose(weights.sum(),1.0,atol=1e-12):
        raise ArithmeticError("multinomial state weights do not sum to one")
    return states,weights


class GlobalLinearCRCOptimizer:
    """Deterministic continuous-action LP for finite-type linear CRC losses."""

    def __init__(self, n_calibration: int,
                 risk_coefficients: Iterable[Iterable[float]] | Iterable[float],
                 alpha: float | Iterable[float]):
        g=np.asarray(risk_coefficients,dtype=np.float64)
        if g.ndim==1:g=g[None,:]
        if g.ndim!=2 or g.shape[0]<1 or g.shape[1]<2 or np.any(~np.isfinite(g)) or np.any(g<0):
            raise ValueError("risk_coefficients must be a finite nonnegative risk-by-type matrix")
        a=np.asarray([alpha] if np.ndim(alpha)==0 else alpha,dtype=np.float64)
        if a.shape!=(g.shape[0],) or np.any(~np.isfinite(a)) or np.any((a<=0)|(a>=1)):
            raise ValueError("one alpha in (0,1) is required for every risk")
        if n_calibration<1:raise ValueError("n_calibration must be positive")
        self.n=int(n_calibration)
        self.g=g
        self.alpha=a
        self.R,self.K=g.shape
        self.budgets=(self.n+1)*a
        self.states=count_vectors(self.n,self.K)
        self.augmented=count_vectors(self.n+1,self.K)
        self.index={d:i for i,d in enumerate(self.states)}

    def verify(self, actions: Iterable[float], tolerance: float=1e-9) -> dict:
        a=np.asarray(list(actions),dtype=np.float64)
        if a.shape!=(len(self.states),) or np.any(~np.isfinite(a)) or np.any((a<0)|(a>1)):
            raise ValueError("one finite action in [0,1] is required for every calibration state")
        worst=[(-math.inf,None) for _ in range(self.R)]
        for S in self.augmented:
            total=np.zeros(self.R)
            for z,multiplicity in enumerate(S):
                if multiplicity:
                    D=list(S);D[z]-=1
                    total+=multiplicity*self.g[:,z]*(1.0-a[self.index[tuple(D)]])
            for r in range(self.R):
                excess=float(total[r]-self.budgets[r])
                if excess>worst[r][0]:
                    worst[r]=(excess,S)
        return {
            "valid":all(excess<=tolerance for excess,_ in worst),
            "max_excess_by_risk":[float(x[0]) for x in worst],
            "worst_augmented_by_risk":[x[1] for x in worst],
            "augmented_states_checked":len(self.augmented),
            "calibration_states":len(self.states),
        }

    def solve_reference_weighted(self, weights: Iterable[float],
                                 utility_slope: Iterable[float] | None=None) -> dict:
        """Maximize E_w[1-s_D*a_D] over a universal CRC-feasible controller.

        Weights are a frozen reference distribution over calibration count
        states. They change only the objective, never certificate constraints.
        """
        w=np.asarray(list(weights),dtype=np.float64)
        if w.shape!=(len(self.states),) or np.any(~np.isfinite(w)) or np.any(w<0) or not np.isclose(w.sum(),1.0):
            raise ValueError("weights must be a probability vector aligned with calibration states")
        slope=np.ones(len(self.states)) if utility_slope is None else np.asarray(list(utility_slope),dtype=np.float64)
        if slope.shape!=(len(self.states),) or np.any(~np.isfinite(slope)) or np.any(slope<0):
            raise ValueError("utility_slope must be a nonnegative value per state")
        c=w*slope
        A=[];b=[]
        for S in self.augmented:
            for r in range(self.R):
                row=np.zeros(len(self.states))
                total=float(sum(S[z]*self.g[r,z] for z in range(self.K)))
                for z,multiplicity in enumerate(S):
                    if multiplicity:
                        D=list(S);D[z]-=1
                        row[self.index[tuple(D)]]-=multiplicity*self.g[r,z]
                A.append(row)
                b.append(float(self.budgets[r]-total))
        sol=linprog(c,A_ub=np.asarray(A),b_ub=np.asarray(b),
                    bounds=[(0.0,1.0)]*len(self.states),method="highs",
                    options={"primal_feasibility_tolerance":1e-10,
                             "dual_feasibility_tolerance":1e-10})
        if not sol.success or sol.x is None:
            raise RuntimeError(f"global CRC LP failed: {sol.message}")
        actions=np.asarray(sol.x,dtype=np.float64)
        check=self.verify(actions)
        if not check["valid"]:raise ArithmeticError(f"LP output failed exact CRC check: {check}")
        utility=float(np.sum(w*(1.0-slope*actions)))
        return {
            "actions":actions.tolist(),
            "expected_utility":utility,
            "expected_action":float(np.dot(w,actions)),
            "reference_weights":w.tolist(),
            "utility_slope":slope.tolist(),
            "verification":check,
            "lp_status":sol.message,
            "objective_constraints":len(b),
        }

    def solve_distribution_free_maximin(self, utility_slope: Iterable[float] | None=None) -> dict:
        """Maximize worst-state utility over every calibration multiset.

        The objective is max_a min_D (1-s_D*a_D). With arbitrary exchangeable
        calibration laws, a worst-case law may put all mass on any one state.
        """
        slope=np.ones(len(self.states)) if utility_slope is None else np.asarray(list(utility_slope),dtype=np.float64)
        if slope.shape!=(len(self.states),) or np.any(~np.isfinite(slope)) or np.any(slope<0):
            raise ValueError("utility_slope must be a nonnegative value per state")
        nvar=len(self.states)+1
        A=[];b=[]
        for S in self.augmented:
            for r in range(self.R):
                row=np.zeros(nvar)
                total=float(sum(S[z]*self.g[r,z] for z in range(self.K)))
                for z,multiplicity in enumerate(S):
                    if multiplicity:
                        D=list(S);D[z]-=1
                        row[self.index[tuple(D)]]-=multiplicity*self.g[r,z]
                A.append(row)
                b.append(float(self.budgets[r]-total))
        for i,s in enumerate(slope):
            row=np.zeros(nvar)
            row[i]=s
            row[-1]=-1.0
            A.append(row);b.append(0.0)
        c=np.zeros(nvar);c[-1]=1.0
        sol=linprog(c,A_ub=np.asarray(A),b_ub=np.asarray(b),
                    bounds=[(0.0,1.0)]*len(self.states)+[(0.0,None)],
                    method="highs")
        if not sol.success or sol.x is None:
            raise RuntimeError(f"maximin CRC LP failed: {sol.message}")
        actions=np.asarray(sol.x[:-1])
        check=self.verify(actions)
        if not check["valid"]:raise ArithmeticError("maximin output failed exact CRC check")
        max_penalty=float(sol.x[-1])
        return {
            "actions":actions.tolist(),
            "worst_state_utility":1.0-max_penalty,
            "max_utility_penalty":max_penalty,
            "verification":check,
            "lp_status":sol.message,
        }


def equal_share_two_type_reference() -> dict[tuple[int,int],float]:
    return {(0,2):0.125,(1,1):0.375,(2,0):0.5}


def run_reference_demo() -> dict:
    engine=GlobalLinearCRCOptimizer(2,[1.0,0.5],0.5)
    states=engine.states
    equal=equal_share_two_type_reference()
    equal_actions=np.asarray([equal[D] for D in states])
    rows=[]
    for p_first in (0.1,0.2,0.5,0.8,0.9):
        _,weights=multinomial_state_weights(2,[p_first,1-p_first])
        result=engine.solve_reference_weighted(weights)
        equal_utility=float(np.dot(weights,1.0-equal_actions))
        rows.append({
            "p_type_0":p_first,
            "equal_share_utility":equal_utility,
            "global_lp_utility":result["expected_utility"],
            "utility_gain":result["expected_utility"]-equal_utility,
            "global_actions_by_state":{str(D):result["actions"][i] for i,D in enumerate(states)},
            "lp_verified":result["verification"]["valid"],
            "all_augmented_constraints":len(engine.augmented),
        })
    robust=engine.solve_distribution_free_maximin()
    return {
        "n":2,"risk_coefficients":[1.0,0.5],"alpha":0.5,
        "calibration_state_order":[list(D) for D in states],
        "equal_share_actions":equal_actions.tolist(),
        "reference_law_results":rows,
        "distribution_free_maximin":{
            "actions_by_state":{str(D):robust["actions"][i] for i,D in enumerate(states)},
            "worst_state_utility":robust["worst_state_utility"],
            "valid":robust["verification"]["valid"],
        },
        "interpretation":[
            "Every LP controller satisfies the same exact augmented-multiset constraints, hence finite-exchangeability validity is retained.",
            "Reference weights change only the optimization objective; they do not weaken the validity constraints.",
            "The utility-optimal controller may trade informativeness between calibration states and need not pointwise dominate equal-share recertification.",
            "The distribution-free maximin objective does not obtain the reference-law gains; it optimizes worst-state utility instead."
        ],
    }


if __name__=="__main__":
    import json
    print(json.dumps(run_reference_demo(),indent=2))
