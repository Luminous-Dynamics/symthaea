#!/usr/bin/env python3
"""Global randomized CRC LP for common-depth hierarchical classification.

Uses the exact finite-population hypergeometric distribution for calibration
count states and the residual-pool test-type distribution conditional on each
calibration state. Action support is the three piecewise-linear breakpoints
a in {0,1,2}. Randomizing over these actions convexifies the piecewise-linear
risk/utility trade-off.

The LP enforces every augmented-multiset risk constraint. The population
counts used for the utility objective are an oracle benchmark: they must be
known independently/frozen before calibration for a deployable procedure.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Iterable
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


def canonical_sha256(payload: object) -> str:
    encoded=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def count_vectors(total: int, types: int=3) -> list[tuple[int,...]]:
    if total<0 or types<1:raise ValueError("invalid count-vector dimensions")
    if types==1:return [(total,)]
    out=[]
    for first in range(total+1):
        for tail in count_vectors(total-first,types-1):
            out.append((first,)+tail)
    return out


def log_combination(n: int, k: int) -> float:
    if k<0 or k>n:return -math.inf
    return math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)


def finite_population_state_weights(
    n_calibration: int, pool_counts: Iterable[int]
) -> tuple[list[tuple[int,...]],np.ndarray]:
    pool=np.asarray(list(pool_counts),dtype=np.int64)
    if pool.ndim!=1 or pool.size<2 or np.any(pool<0):
        raise ValueError("pool_counts must be a nonnegative vector with at least two types")
    if int(pool.sum())<n_calibration or n_calibration<1:
        raise ValueError("calibration sample exceeds the pool or is empty")
    states=count_vectors(n_calibration,len(pool))
    log_den=log_combination(int(pool.sum()),n_calibration)
    log_weights=[]
    for d in states:
        if any(d[i]>pool[i] for i in range(len(pool))):
            log_weights.append(-math.inf)
        else:
            log_weights.append(sum(log_combination(int(pool[i]),int(d[i]))
                                   for i in range(len(pool)))-log_den)
    w=np.exp(np.asarray(log_weights,dtype=np.float64))
    total=float(w.sum())
    if not math.isfinite(total) or total<=0:raise ArithmeticError("invalid hypergeometric weights")
    w/=total
    return states,w


def depth_probabilities(a: float) -> np.ndarray:
    if not math.isfinite(a) or not 0<=a<=2:raise ValueError("depth must be in [0,2]")
    return np.asarray([max(1-a,0.0),1-abs(a-1.0),max(a-1.0,0.0)])


def type_loss(error_type: int, a: float) -> float:
    if error_type==0:return 0.0
    if error_type==1:return .5*max(a-1.0,0.0)
    if error_type==2:return .5*a
    raise ValueError("error_type must be 0,1,2")


def per_type_metrics(a: float) -> dict[str,np.ndarray]:
    pi=depth_probabilities(a)
    superclass_information=math.log(20.0)/math.log(100.0)
    specificity=np.asarray([
        pi[1]*superclass_information+pi[2],
        pi[1]*superclass_information,
        0.0
    ])
    node_size=np.full(3,100.0*pi[0]+5.0*pi[1]+pi[2])
    risks=np.asarray([type_loss(z,a) for z in range(3)])
    return {"risk":risks,"specificity":specificity,"node_size":node_size}


def residual_test_probabilities(pool_counts: Iterable[int], calibration_counts: tuple[int,...]) -> np.ndarray:
    pool=np.asarray(list(pool_counts),dtype=np.int64)
    d=np.asarray(calibration_counts,dtype=np.int64)
    if pool.shape!=d.shape or np.any(d>pool):
        raise ValueError("calibration counts incompatible with population")
    remaining=pool-d
    denom=int(remaining.sum())
    if denom<=0:raise ValueError("no test population remains after calibration")
    return remaining.astype(np.float64)/denom


class RandomizedCommonDepthCRC:
    """LP optimizer over randomized common-depth maps on the {0,1,2} action support."""

    def __init__(self, n_calibration: int, alpha: float,
                 pool_counts: Iterable[int]=(7300,969,1731),
                 action_grid: Iterable[float]=(0.0,1.0,2.0)):
        pool=np.asarray(list(pool_counts),dtype=np.int64)
        actions=np.asarray(list(action_grid),dtype=np.float64)
        if pool.shape!=(3,) or np.any(pool<0) or pool.sum()<=n_calibration:
            raise ValueError("common-depth CIFAR pool must contain three types and exceed n")
        if n_calibration<1 or not math.isfinite(alpha) or not 0<alpha<1:
            raise ValueError("invalid n or alpha")
        if actions.ndim!=1 or actions.size<2 or np.any(~np.isfinite(actions)) or np.any((actions<0)|(actions>2)):
            raise ValueError("action_grid must be a finite vector in [0,2]")
        if len(set(actions.tolist()))!=actions.size or not np.all(np.diff(actions)>0):
            raise ValueError("action_grid must be strictly increasing and unique")
        if not all(any(abs(a-x)<1e-12 for x in actions) for a in (0.,1.,2.)):
            raise ValueError("piecewise-linear exact support must include breakpoints 0,1,2")
        self.n=int(n_calibration)
        self.alpha=float(alpha)
        self.pool=pool
        self.actions=actions
        self.budget=(self.n+1)*self.alpha
        self.states,self.weights=finite_population_state_weights(self.n,self.pool)
        self.augmented=count_vectors(self.n+1,3)
        self.index={D:i for i,D in enumerate(self.states)}
        self.conditional_test_probs=np.asarray([
            residual_test_probabilities(self.pool,D) for D in self.states
        ])
        self.losses=np.asarray([
            [type_loss(z,a) for a in self.actions] for z in range(3)
        ])
        self.metrics=[per_type_metrics(float(a)) for a in self.actions]
        self.utility=np.asarray([
            [np.dot(self.conditional_test_probs[i],self.metrics[j]["specificity"])
             for j in range(len(self.actions))]
            for i in range(len(self.states))
        ])

    def build_certificate(self, probabilities: np.ndarray,
                          verification: dict,
                          dual_certificate: dict,
                          dual_witness: dict) -> dict:
        policy_payload={
            "calibration_states":[list(D) for D in self.states],
            "action_grid":self.actions.tolist(),
            "probabilities":np.asarray(probabilities,dtype=np.float64).tolist(),
        }
        constraint_payload={
            "risk_family":"common_depth_hierarchical_crc",
            "loss_by_type_and_action":self.losses.tolist(),
            "alpha":self.alpha,
            "budget":self.budget,
            "calibration_states":[list(D) for D in self.states],
            "augmented_states":[list(S) for S in self.augmented],
        }
        objective_payload={
            "reference_pool_counts":self.pool.tolist(),
            "hypergeometric_calibration_weights":self.weights.tolist(),
            "conditional_test_type_probabilities":self.conditional_test_probs.tolist(),
            "specificity_utility_by_state_and_action":self.utility.tolist(),
        }
        return {
            "schema":"symthaea-risk-certificate-v1",
            "certificate_type":"RandomizedFiniteExchangeabilityCRC",
            "proof_lane":"ExactFiniteExchangeabilityPlusLPOptimality",
            "risk_target":{
                "metric":"expected_hierarchical_loss",
                "level":self.alpha,
                "guarantee":"expected_risk",
            },
            "assumption_contract":{
                "finite_exchangeability":True,
                "finite_observation_alphabet":3,
                "frozen_loss_and_action_domain":True,
                "randomization_included_in_expectation":True,
                "distribution_shift_guarantee":False,
                "objective_weights_must_be_independently_frozen":True,
            },
            "scope":{
                "n_calibration":self.n,
                "calibration_state_count":len(self.states),
                "augmented_state_count":len(self.augmented),
                "type_count":3,
                "action_grid":self.actions.tolist(),
            },
            "evidence_digests":{
                "decision_output_sha256":canonical_sha256(policy_payload),
                "constraint_domain_sha256":canonical_sha256(constraint_payload),
                "objective_reference_sha256":canonical_sha256(objective_payload),
            },
            "verifier_receipt":{
                "status":"pass" if verification["valid"] else "fail",
                "method":"exhaustive_augmented_multiset_expected_loss",
                "states_checked":verification["augmented_states_checked"],
                "max_budget_excess":verification["max_budget_excess"],
                "tolerance":1e-9,
                "worst_augmented_count":list(verification["worst_augmented_count"]),
                "worst_augmented_total":verification["worst_augmented_total"],
            },
            "optimization_receipt":{
                "solver":"HiGHS",
                "kkt_summary":dual_certificate,
                "dual_witness":dual_witness,
                "status":"pass" if dual_certificate["duality_gap_abs"]<=1e-7 else "fail",
            },
        }

    def verify_optimality_witness(self, dual_witness: dict,
                                  probabilities: np.ndarray,
                                  tolerance: float=1e-7) -> dict:
        n_states=len(self.states)
        n_actions=len(self.actions)
        n_variables=n_states*n_actions
        A_ub=lil_matrix((len(self.augmented),n_variables),dtype=np.float64)
        for row,S in enumerate(self.augmented):
            for z,multiplicity in enumerate(S):
                if multiplicity:
                    D=list(S);D[z]-=1
                    i=self.index[tuple(D)]
                    start=i*n_actions
                    A_ub[row,start:start+n_actions]+=multiplicity*self.losses[z]
        A_eq=lil_matrix((n_states,n_variables),dtype=np.float64)
        for i in range(n_states):
            A_eq[i,i*n_actions:(i+1)*n_actions]=1.0
        A_ub=A_ub.tocsr()
        A_eq=A_eq.tocsr()
        b_ub=np.full(len(self.augmented),self.budget)
        b_eq=np.ones(n_states)
        lower=np.zeros(n_variables)
        upper=np.ones(n_variables)
        P=np.asarray(probabilities,dtype=np.float64)
        x=P.reshape(-1)
        c=-(self.weights[:,None]*self.utility).reshape(-1)
        try:
            y_ub=np.asarray(dual_witness.get("inequality_marginals",[]),dtype=np.float64)
            y_eq=np.asarray(dual_witness.get("equality_marginals",[]),dtype=np.float64)
            y_lower=np.asarray(dual_witness.get("lower_marginals",[]),dtype=np.float64)
            y_upper=np.asarray(dual_witness.get("upper_marginals",[]),dtype=np.float64)
        except (TypeError,ValueError,OverflowError):
            return {"valid":False,"reason":"dual witness contains nonnumeric values"}
        if (y_ub.shape!=(len(self.augmented),) or y_eq.shape!=(n_states,)
                or y_lower.shape!=(n_variables,) or y_upper.shape!=(n_variables,)):
            return {"valid":False,"reason":"dual witness shape mismatch"}
        if any(np.any(~np.isfinite(v)) for v in (y_ub,y_eq,y_lower,y_upper)):
            return {"valid":False,"reason":"dual witness contains nonfinite values"}
        inequality_slack=b_ub-A_ub@x
        equality_residual=A_eq@x-b_eq
        stationarity=c-A_ub.T@y_ub-A_eq.T@y_eq-y_lower-y_upper
        dual_objective=float(np.dot(b_ub,y_ub)+np.dot(b_eq,y_eq)
                             +np.dot(lower,y_lower)+np.dot(upper,y_upper))
        complementarity=max(
            float(np.max(np.abs(y_ub*inequality_slack))),
            float(np.max(np.abs(y_lower*(x-lower)))),
            float(np.max(np.abs(y_upper*(upper-x)))),
        )
        receipt={
            "primal_max_inequality_violation":float(np.max(np.maximum(-inequality_slack,0.0))),
            "primal_max_equality_residual":float(np.max(np.abs(equality_residual))),
            "primal_max_bound_violation":float(max(np.max(np.maximum(lower-x,0.0)),
                                                   np.max(np.maximum(x-upper,0.0)))),
            "dual_inequality_sign_violation":float(np.max(np.maximum(y_ub,0.0))),
            "dual_lower_sign_violation":float(np.max(np.maximum(-y_lower,0.0))),
            "dual_upper_sign_violation":float(np.max(np.maximum(y_upper,0.0))),
            "stationarity_max_abs":float(np.max(np.abs(stationarity))),
            "complementarity_max_abs":complementarity,
            "primal_objective_min":float(np.dot(c,x)),
            "dual_objective_min":dual_objective,
            "duality_gap_abs":float(abs(np.dot(c,x)-dual_objective)),
        }
        worst=max(receipt[k] for k in receipt
                  if k.endswith("violation") or k.endswith("residual")
                  or k.endswith("max_abs") or k=="duality_gap_abs")
        receipt["valid"]=bool(worst<=tolerance)
        return receipt

    def verify_certificate(self, certificate: dict,
                           probabilities: np.ndarray,
                           tolerance: float=1e-7) -> dict:
        """Fail-closed verifier for a serialized policy, validity and LP-optimality receipt."""
        if not isinstance(certificate,dict):
            return {"valid":False,"reason":"certificate must be a mapping"}
        if certificate.get("schema")!="symthaea-risk-certificate-v1":
            return {"valid":False,"reason":"unsupported certificate schema"}
        if certificate.get("certificate_type")!="RandomizedFiniteExchangeabilityCRC":
            return {"valid":False,"reason":"certificate type mismatch"}
        if certificate.get("proof_lane")!="ExactFiniteExchangeabilityPlusLPOptimality":
            return {"valid":False,"reason":"proof lane mismatch"}
        optimization=certificate.get("optimization_receipt")
        if not isinstance(optimization,dict):
            return {"valid":False,"reason":"optimization receipt is malformed"}
        kkt_summary=optimization.get("kkt_summary")
        dual_witness=optimization.get("dual_witness")
        if not isinstance(kkt_summary,dict) or not isinstance(dual_witness,dict):
            return {"valid":False,"reason":"LP optimality evidence is malformed"}
        scope=certificate.get("scope")
        risk_target=certificate.get("risk_target")
        evidence_digests=certificate.get("evidence_digests")
        if not isinstance(scope,dict) or not isinstance(risk_target,dict):
            return {"valid":False,"reason":"scope or risk target metadata is malformed"}
        if not isinstance(evidence_digests,dict):
            return {"valid":False,"reason":"evidence digest metadata is malformed"}
        try:
            if scope.get("n_calibration")!=self.n:
                return {"valid":False,"reason":"calibration scope mismatch"}
            if risk_target.get("level")!=self.alpha:
                return {"valid":False,"reason":"risk target mismatch"}
            verification=self.verify(probabilities)
        except (TypeError,ValueError,KeyError,IndexError,AttributeError) as exc:
            return {"valid":False,"reason":f"invalid certificate policy or scope: {exc}"}
        if not verification["valid"]:
            return {"valid":False,"reason":"exact augmented-state verification failed",
                    "verification":verification}
        # Build expected identity fields without trusting the supplied KKT
        # summary to be complete or well-formed.
        expected=self.build_certificate(
            probabilities,verification,{"duality_gap_abs":0.0},dual_witness
        )
        for name,value in expected["evidence_digests"].items():
            if evidence_digests.get(name)!=value:
                return {"valid":False,"reason":f"evidence digest mismatch: {name}"}
        for field in ("proof_lane","risk_target","assumption_contract","scope","verifier_receipt"):
            if certificate.get(field)!=expected.get(field):
                return {"valid":False,"reason":f"certificate field mismatch: {field}"}
        if optimization.get("solver")!="HiGHS":
            return {"valid":False,"reason":"unsupported optimization solver receipt"}
        if optimization.get("status")!="pass":
            return {"valid":False,"reason":"optimization receipt is not marked pass"}
        dual=self.verify_optimality_witness(dual_witness,probabilities,tolerance)
        if not dual.get("valid",False):
            return {"valid":False,"reason":"LP dual witness failed KKT checks",
                    "optimality":dual}
        for name,value in dual.items():
            if name=="valid":
                continue
            if name not in kkt_summary:
                return {"valid":False,"reason":f"KKT summary missing field: {name}"}
            try:
                reported_value=float(kkt_summary[name])
            except (TypeError,ValueError):
                return {"valid":False,"reason":f"KKT summary malformed field: {name}"}
            if not math.isfinite(reported_value) or not np.isclose(
                reported_value,float(value),atol=tolerance,rtol=0.0
            ):
                return {"valid":False,"reason":f"KKT summary mismatch: {name}"}
        if kkt_summary.get("valid") is not True:
            return {"valid":False,"reason":"reported KKT summary is not marked valid"}
        return {
            "valid":True,
            "reason":None,
            "exact_validity":verification,
            "optimality":dual,
        }

    def solve(self) -> dict:
        n_states=len(self.states)
        n_actions=len(self.actions)
        n_variables=n_states*n_actions
        A_ub=lil_matrix((len(self.augmented),n_variables),dtype=np.float64)
        for row,S in enumerate(self.augmented):
            for z,multiplicity in enumerate(S):
                if multiplicity:
                    D=list(S);D[z]-=1
                    state_index=self.index[tuple(D)]
                    start=state_index*n_actions
                    A_ub[row,start:start+n_actions]+=multiplicity*self.losses[z]
        A_eq=lil_matrix((n_states,n_variables),dtype=np.float64)
        for i in range(n_states):
            A_eq[i,i*n_actions:(i+1)*n_actions]=1.0
        objective=-(self.weights[:,None]*self.utility).reshape(-1)
        started=time.perf_counter()
        A_ub_csr=A_ub.tocsr()
        A_eq_csr=A_eq.tocsr()
        b_ub=np.full(len(self.augmented),self.budget)
        b_eq=np.ones(n_states)
        lower_bounds=np.zeros(n_variables)
        upper_bounds=np.ones(n_variables)
        result=linprog(
            objective,
            A_ub=A_ub_csr,
            b_ub=b_ub,
            A_eq=A_eq_csr,
            b_eq=b_eq,
            bounds=list(zip(lower_bounds,upper_bounds)),
            method="highs",
            options={"primal_feasibility_tolerance":1e-9,
                     "dual_feasibility_tolerance":1e-9},
        )
        runtime=time.perf_counter()-started
        if not result.success or result.x is None:
            raise RuntimeError(f"randomized CRC LP failed: {result.message}")
        probabilities=result.x.reshape(n_states,n_actions)

        # Independently audit the HiGHS primal/dual solution. SciPy's
        # marginals use the sign convention in which KKT stationarity is
        # c - A_ub.T*y_ub - A_eq.T*y_eq - lower - upper = 0.
        y_ub=np.asarray(result.ineqlin.marginals,dtype=np.float64)
        y_eq=np.asarray(result.eqlin.marginals,dtype=np.float64)
        y_lower=np.asarray(result.lower.marginals,dtype=np.float64)
        y_upper=np.asarray(result.upper.marginals,dtype=np.float64)
        stationarity=(
            objective
            - A_ub_csr.T@y_ub
            - A_eq_csr.T@y_eq
            - y_lower
            - y_upper
        )
        dual_objective=float(
            np.dot(b_ub,y_ub)
            + np.dot(b_eq,y_eq)
            + np.dot(lower_bounds,y_lower)
            + np.dot(upper_bounds,y_upper)
        )
        inequality_slack=b_ub-A_ub_csr@result.x
        equality_residual=A_eq_csr@result.x-b_eq
        complementarity=max(
            float(np.max(np.abs(y_ub*inequality_slack))),
            float(np.max(np.abs(y_lower*(result.x-lower_bounds)))),
            float(np.max(np.abs(y_upper*(upper_bounds-result.x)))),
        )
        dual_certificate={
            "primal_max_inequality_violation":float(np.max(np.maximum(-inequality_slack,0.0))),
            "primal_max_equality_residual":float(np.max(np.abs(equality_residual))),
            "primal_max_bound_violation":float(max(
                np.max(np.maximum(lower_bounds-result.x,0.0)),
                np.max(np.maximum(result.x-upper_bounds,0.0)))),
            "dual_inequality_sign_violation":float(np.max(np.maximum(y_ub,0.0))),
            "dual_lower_sign_violation":float(np.max(np.maximum(-y_lower,0.0))),
            "dual_upper_sign_violation":float(np.max(np.maximum(y_upper,0.0))),
            "stationarity_max_abs":float(np.max(np.abs(stationarity))),
            "complementarity_max_abs":complementarity,
            "primal_objective_min":float(result.fun),
            "dual_objective_min":dual_objective,
            "duality_gap_abs":float(abs(result.fun-dual_objective)),
            "convention":"SciPy/HiGHS LP marginals; stationarity c-A_ub.T*y_ub-A_eq.T*y_eq-lower-upper=0"
        }
        worst_kkt=max(v for k,v in dual_certificate.items()
                      if k.endswith("violation") or k.endswith("residual")
                      or k.endswith("max_abs") or k=="duality_gap_abs")
        if worst_kkt>1e-7:
            raise ArithmeticError(f"LP primal/dual optimality receipt failed: {dual_certificate}")
        report=self.verify(probabilities)
        if not report["valid"]:
            raise ArithmeticError(f"LP solution failed exact augmented-state verification: {report}")
        metrics=self.evaluate(probabilities)
        supports=(probabilities>1e-8).sum(axis=1)
        dual_witness={
            "inequality_marginals":y_ub.tolist(),
            "equality_marginals":y_eq.tolist(),
            "lower_marginals":y_lower.tolist(),
            "upper_marginals":y_upper.tolist(),
        }
        certificate=self.build_certificate(
            probabilities,report,dual_certificate,dual_witness
        )
        return {
            "action_grid":self.actions.tolist(),
            "probabilities":probabilities.tolist(),
            "expected_metrics":metrics,
            "verification":report,
            "certificate":certificate,
            "solver":{"method":"HiGHS","success":bool(result.success),
                      "status":str(result.message),"runtime_seconds":runtime,
                      "objective_value":float(-result.fun),
                      "n_variables":n_variables,
                      "n_equality_constraints":n_states,
                      "n_inequality_constraints":len(self.augmented),
                      "dual_certificate":dual_certificate},
            "randomization":{
                "states_with_multiple_actions":int(np.sum(supports>1)),
                "fraction_states_randomized":float(np.mean(supports>1)),
                "support_histogram":{str(k):int(np.sum(supports==k)) for k in np.unique(supports)},
            },
        }

    def verify(self, probabilities: np.ndarray, tolerance: float=1e-9) -> dict:
        P=np.asarray(probabilities,dtype=np.float64)
        if P.shape!=(len(self.states),len(self.actions)) or np.any(~np.isfinite(P)) or np.any(P<-tolerance) or np.any(P>1.0+tolerance):
            raise ValueError("invalid randomized action table")
        row_sums=P.sum(axis=1)
        if np.max(np.abs(row_sums-1.0))>tolerance:
            raise ValueError("every calibration state must have a normalized action distribution")
        worst=(-math.inf,None)
        totals={}
        for S in self.augmented:
            total=0.0
            for z,multiplicity in enumerate(S):
                if multiplicity:
                    D=list(S);D[z]-=1
                    i=self.index[tuple(D)]
                    total+=multiplicity*float(np.dot(P[i],self.losses[z]))
            totals[",".join(map(str,S))]=total
            if total>worst[0]:worst=(total,S)
        excess=float(worst[0]-self.budget)
        return {
            "valid":bool(excess<=tolerance),
            "max_budget_excess":excess,
            "worst_augmented_count":worst[1],
            "worst_augmented_total":float(worst[0]),
            "budget":self.budget,
            "augmented_states_checked":len(self.augmented),
            "calibration_states":len(self.states),
            "all_augmented_totals":totals,
        }

    def evaluate(self, probabilities: np.ndarray) -> dict:
        P=np.asarray(probabilities,dtype=np.float64)
        totals={"risk":0.0,"correct_specificity":0.0,"mean_node_size":0.0}
        for i in range(len(self.states)):
            w=self.weights[i]
            q=self.conditional_test_probs[i]
            for j in range(len(self.actions)):
                mass=w*P[i,j]
                if mass==0:continue
                totals["risk"]+=mass*float(np.dot(q,self.metrics[j]["risk"]))
                totals["correct_specificity"]+=mass*float(self.utility[i,j])
                totals["mean_node_size"]+=mass*float(np.dot(q,self.metrics[j]["node_size"]))
        return totals


def evaluate_deterministic_controller(engine: RandomizedCommonDepthCRC,
                                      controller: dict[tuple[int,...],float]) -> dict:
    metrics={"risk":0.0,"correct_specificity":0.0,"mean_node_size":0.0}
    for i,D in enumerate(engine.states):
        q=engine.conditional_test_probs[i]
        w=engine.weights[i]
        action=float(controller[D])
        point=per_type_metrics(action)
        metrics["risk"]+=w*float(np.dot(q,point["risk"]))
        metrics["correct_specificity"]+=w*float(np.dot(q,point["specificity"]))
        metrics["mean_node_size"]+=w*float(np.dot(q,point["node_size"]))
    return metrics


def run_suite(n: int=50, pool_counts: tuple[int,int,int]=(7300,969,1731),
              alphas: tuple[float,...]=(.02,.05,.10,.15), recertification_rounds: int=500) -> dict:
    from common_depth_cifar_crc import CommonDepthCIFARCRC
    results=[]
    for alpha in alphas:
        engine=RandomizedCommonDepthCRC(n,alpha,pool_counts)
        lp=engine.solve()
        baseline_engine=CommonDepthCIFARCRC(n,alpha)
        crc=baseline_engine.baseline("crc")
        envelope=baseline_engine.baseline("envelope")
        recertified,history=baseline_engine.recertify(envelope,rounds=recertification_rounds)
        results.append({
            "alpha":alpha,
            "n_states":len(engine.states),
            "n_augmented_states":len(engine.augmented),
            "standard_crc":evaluate_deterministic_controller(engine,crc),
            "envelope_crc":evaluate_deterministic_controller(engine,envelope),
            "equal_share_recertified":evaluate_deterministic_controller(engine,recertified),
            "equal_share_rounds_executed":len(history)-1,
            "randomized_global_lp":lp["expected_metrics"],
            "global_gain_vs_equal_share":{
                key:lp["expected_metrics"][key]-evaluate_deterministic_controller(engine,recertified)[key]
                for key in lp["expected_metrics"]},
            "solver":lp["solver"],
            "certificate":lp["certificate"],
            "certificate_verification":engine.verify_certificate(
                lp["certificate"],np.asarray(lp["probabilities"],dtype=np.float64)
            ),
            "verification":{
                "valid":lp["verification"]["valid"],
                "max_budget_excess":lp["verification"]["max_budget_excess"],
                "worst_augmented_count":lp["verification"]["worst_augmented_count"],
                "augmented_states_checked":lp["verification"]["augmented_states_checked"]},
            "randomization":lp["randomization"],
            "probabilities":lp["probabilities"],
        })
    return {
        "schema":"symthaea-randomized-global-common-depth-crc-v1",
        "status":"ORACLE_FINITE_POPULATION_UTILITY_BENCHMARK_NOT_DEPLOYMENT_RESULT",
        "reference":"Bai & Ramdas (2026), arXiv:2610.35821, common-depth CIFAR-100 family",
        "protocol":{
            "pool_counts":list(pool_counts),"n_calibration":n,"alphas":list(alphas),
            "reference_calibration_weights":"exact multivariate hypergeometric from the finite pool",
            "conditional_test_type_weights":"residual pool counts after each calibration multiset",
            "action_support":[0.0,1.0,2.0],
            "decision":"for each calibration state D, sample one depth from the solved categorical distribution, then apply the paper's prediction-node randomization",
            "objective":"expected correct specificity on the residual finite population",
            "validity":"all augmented multiset expected-loss constraints checked exactly",
            "objective_provenance":"oracle benchmark using the published full-pool type counts; a deployable system must freeze weights from independent data"
        },
        "results":results,
        "limitations":[
            "uses aggregate full-pool type counts to define the utility objective; this is an oracle benchmark and not a deployable calibration rule",
            "risk validity is certified by every augmented multiset constraint, not inferred from empirical test risk",
            "common-depth family only; confidence-adaptive image-specific controller is separate",
            "the global LP optimizes the chosen finite action support and hierarchy-specific expected specificity, not all possible utilities",
            "finite exchangeability only; no arbitrary distribution-shift guarantee"
        ],
        "environment":{"python":platform.python_version(),"numpy":np.__version__}
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    payload=run_suite()
    Path(args.output).write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":args.output,"results":[
        {"alpha":x["alpha"],"global":x["randomized_global_lp"],
         "equal_share":x["equal_share_recertified"],
         "gain":x["global_gain_vs_equal_share"],
         "max_violation":x["verification"]["max_budget_excess"],
         "runtime_seconds":x["solver"]["runtime_seconds"]}
        for x in payload["results"]]},indent=2))
