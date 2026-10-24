#!/usr/bin/env python3
"""Convexify fixed-prior minimax policy frontiers with randomized policy-tree mixtures.

Each deterministic observation-contingent policy tree has one expected terminal
cross-entropy for each candidate true prior. Randomizing over whole trees makes the
risk vector a convex combination. A linear program minimizes worst-case risk over
that convex hull. This is an exact solution over the generated finite frontier,
not over arbitrary models or missing candidate priors.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from objective_matched_oracle_benchmark import OBJECTIVES
from prior_shift_oracle_benchmark import PRIOR_NAMES, build_prior
from fixed_prior_minimax_policy_benchmark import make_frontier_solver, evaluate_tree_from_posteriors

SCHEMA = 'symthaea-randomized-minimax-policy-mixture-v1'
POLICIES = ('randomized_minimax_mixture','deterministic_minimax_frontier','robust_minimax_one_step','greedy_joint','round_robin','random_uniform')

def solve_minimax_mixture(risk_vectors):
    """Solve min_w max_s sum_j w_j R[j,s], weights form a simplex."""
    vectors=np.asarray(risk_vectors,dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError('risk_vectors must be a nonempty 2D matrix')
    if not np.isfinite(vectors).all():
        raise ValueError('risk vectors must be finite')
    n_policies, n_scenarios=vectors.shape
    c=np.zeros(n_policies+1); c[-1]=1.0
    a_ub=np.hstack([vectors.T, -np.ones((n_scenarios,1))])
    b_ub=np.zeros(n_scenarios)
    a_eq=np.zeros((1,n_policies+1)); a_eq[0,:n_policies]=1.0
    b_eq=np.ones(1)
    bounds=[(0.0,None)]*n_policies+[(0.0,None)]
    res=linprog(c,A_ub=a_ub,b_ub=b_ub,A_eq=a_eq,b_eq=b_eq,bounds=bounds,method='highs')
    if not res.success: raise RuntimeError(f'minimax LP failed: {res.message}')
    weights=np.maximum(res.x[:n_policies],0.0)
    weights=weights/weights.sum()
    actual=vectors.T@weights
    return {'weights':weights.tolist(),'mixture_risk_vector':actual.tolist(),
            'worst_case_bits':float(actual.max()),'mean_scenario_bits':float(actual.mean()),
            'support_size':int(np.count_nonzero(weights>1e-8)),
            'lp_reported_objective':float(res.fun),'lp_status':res.message}

def run_benchmark(candidate_prior_names=PRIOR_NAMES, learner_prior_names=PRIOR_NAMES,
                  objectives=OBJECTIVES, horizons=(1,2)):
    for names in (candidate_prior_names,learner_prior_names,objectives,horizons):
        if not names: raise ValueError('input lists must be nonempty')
    if any(n not in PRIOR_NAMES for n in candidate_prior_names+learner_prior_names): raise ValueError('unknown prior')
    if any(o not in OBJECTIVES for o in objectives): raise ValueError('unknown objective')
    if any(h not in (1,2,3) for h in horizons): raise ValueError('horizons must be 1,2,3')
    rows=[]; complexity={}
    for obj in objectives:
        initial,frontier,best,stats=make_frontier_solver(obj,candidate_prior_names)
        for learner_name in learner_prior_names:
            p=tuple(build_prior(learner_name))
            for h in horizons:
                t0=time.perf_counter(); options=frontier(p,initial,h); exact_time=time.perf_counter()-t0
                risks=np.asarray([v for v,_ in options],dtype=float)
                t1=time.perf_counter(); lp=solve_minimax_mixture(risks); lp_time=time.perf_counter()-t1
                deterministic_vector,deterministic_tree=best(p,h)
                deterministic_worst=max(deterministic_vector)
                # Independently rescore each selected deterministic tree against all scenarios.
                chosen=[]
                for i,prior_name in enumerate(candidate_prior_names):
                    score=evaluate_tree_from_posteriors(deterministic_tree,tuple(build_prior(prior_name)),p,obj,h)
                    if abs(score-deterministic_vector[i])>1e-8: raise AssertionError('deterministic tree risk vector mismatch')
                    chosen.append(score)
                if lp['worst_case_bits'] > deterministic_worst+1e-8:
                    raise AssertionError('mixture LP should never be worse than its best deterministic candidate')
                row={'objective':obj,'learner_prior':learner_name,'candidate_prior_names':list(candidate_prior_names),
                     'horizon':h,'frontier_policy_count':len(options),'frontier_peak':stats['max'],'frontier_states':stats['states'],
                     'deterministic_minimax_risk_vector':list(deterministic_vector),
                     'deterministic_worst_case_bits':deterministic_worst,
                     'randomized_minimax':lp,'worst_case_improvement_bits':deterministic_worst-lp['worst_case_bits'],
                     'well_specified_candidates':[n==learner_name for n in candidate_prior_names],
                     'exact_frontier_seconds':exact_time,'mixture_lp_seconds':lp_time}
                rows.append(row)
                complexity[f'{obj}/{learner_name}/h{h}']={'frontier_count':len(options),'max_frontier_seen':stats['max'],
                                                          'states':stats['states'],'frontier_seconds':exact_time,'lp_seconds':lp_time,
                                                          'mixture_support_size':lp['support_size']}
    return {'schema':SCHEMA,'status':'EXACT_FINITE_MODEL_RANDOMIZED_POLICY_MIXTURE_PILOT',
            'protocol':{'candidate_true_prior_identity_fixed_per_episode':True,'policy_mixture_drawn_once_at_episode_start':True,
                        'true_prior_not_given_to_policy':True,'score':'expected terminal categorical cross-entropy (bits)',
                        'optimization':'linear program over convex combinations of exact deterministic policy-tree risk vectors',
                        'limitations':['16 joint hypotheses; chain/fork graphs only','six finite, uncalibrated candidate priors',
                                       'exact realized target-subset telemetry','no action cost, telemetry noise/missingness, or task disturbance',
                                       'no external benchmark run']},
            'source_reference':'fixed_prior_minimax_policy_benchmark.py','complexity':complexity,'rows':rows}

def self_test():
    r=solve_minimax_mixture([[0,1],[1,0]])
    assert abs(r['worst_case_bits']-0.5)<1e-8 and r['support_size']==2
    d=solve_minimax_mixture([[0,2],[1,3],[2,0]])
    assert d['weights'] and abs(sum(d['weights'])-1)<1e-10
    result=run_benchmark(('uniform','high_spill_biased'),('uniform','chain_biased'),('joint',),(1,2))
    assert len(result['rows'])==4
    for row in result['rows']:
        assert row['randomized_minimax']['worst_case_bits'] <= row['deterministic_worst_case_bits']+1e-8
        assert abs(sum(row['randomized_minimax']['weights'])-1.0)<1e-8
    return True

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--candidate-priors',default=','.join(PRIOR_NAMES))
    ap.add_argument('--learner-priors',default=','.join(PRIOR_NAMES))
    ap.add_argument('--objectives',default=','.join(OBJECTIVES))
    ap.add_argument('--horizons',default='1,2')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--source-git-blob',default='UNBOUND')
    args=ap.parse_args()
    try:
        if args.self_test: self_test(); print('RANDOMIZED_POLICY_MIXTURE_TESTS_PASSED'); return 0
        out=run_benchmark(tuple(args.candidate_priors.split(',')),tuple(args.learner_priors.split(',')),
                          tuple(args.objectives.split(',')),tuple(int(x) for x in args.horizons.split(',')))
        out['source_git_blob']=args.source_git_blob
        raw=(json.dumps(out,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        digest=hashlib.sha256(raw).hexdigest()
        if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_bytes(raw); print(f'wrote {args.output} bytes={len(raw)} sha256={digest}')
        else: print(raw.decode(),end='')
        return 0
    except (ValueError,RuntimeError,AssertionError,OSError,RecursionError) as exc: print(f'NOT QUALIFIED: {exc}'); return 2
if __name__=='__main__': raise SystemExit(main())
