# Risk-bounded stochastic adaptive connectome intervention policies V1

**Tracking:** #10831 — SCI-CONNECTOMICS-096
**Composes:** #10670, #10823, #10825, #10827, #10603, #10656, #9255
**Status:** exact finite synthetic stochastic-policy contract; no biological calibration or intervention result is claimed.

## 1. What changes relative to deterministic V1

#10823 models a complete deterministic potential-outcome vector for each biological-world × nuisance state. It can certify zero-error query consensus when all compatible worlds agree. Biological measurements are stochastic: useful observations can have overlapping likelihoods even when they sharply reduce expected scientific decision error.

V1 therefore separates:
- **Support guarantee:** all compatible worlds agree with the query (the deterministic #10823 mode, epsilon = 0).
- **Expected-risk guarantee:** expected query loss under each frozen generative model is at most epsilon.
- **Tail-risk guarantee:** under each model, the probability that the robust conditional posterior risk at the terminal history exceeds tau is at most delta.
- **Worst-history posterior risk:** a diagnostic maximum across reachable terminal histories. It is reported but is not conflated with expected loss or tail probability.

Requiring every possible noisy terminal history to have posterior risk <= epsilon can make a finite policy impossible even when expected error is low: a near-balanced but low-probability history can remain uncertain after many informative measurements. Do not impose that stronger condition without naming it as a separate application requirement.

## 2. A complete generative model is the unit of robustness

Each frozen generative model g contains:
- a prior distribution over biological-world × fixed-nuisance latent states;
- for each latent state, a probability distribution over complete joint potential-outcome vectors across the entire finite action universe;
- one immutable observation-model/source digest.

One complete joint outcome vector assigns an outcome to every intervention. The planner observes a projection of that vector as actions are taken. This preserves cross-action dependence. Do not multiply single-action marginals unless a factorization has been justified, declared and encoded in the joint law.

The nuisance model is part of the latent state and remains fixed. The complete generative model is fixed throughout a trajectory. A robust policy is one common mapping from observed histories to next action or terminal decision across all g; it cannot select a different model after seeing a favorable observation.

V1 uses exact rational probability masses `(numerator, denominator)` for synthetic known answers. Each model's latent-state prior sums to one; for every latent state, the joint potential-outcome atom masses sum to one. Equal atoms may be aggregated but must not be silently duplicated.

## 3. Policy and risk semantics

A policy maps an observed history to either a feasible unused intervention or a terminal decision `TRUE` / `FALSE`.

For model g and history h, Bayes' rule over the frozen latent-state prior and full-vector likelihood induces a posterior over biological query decisions. For terminal decision d, define conditional query loss:

`r_g(h,d) = P_g(d != q(W) | h)`

and robust conditional risk:

`r*(h,d) = max_g r_g(h,d)`.

For a policy pi, compute independently for every g:
- expected intervention effort `C_g(pi)`;
- expected query loss `R_g(pi) = E_g[1{d(H) != q(W)}]`;
- tail mass `T_g(pi) = P_g(r*(H,d(H)) > tau)`;
- maximum robust conditional risk over reachable terminal histories.

Promotion constraints:
- `max_g R_g(pi) <= epsilon`;
- `max_g T_g(pi) <= delta`.

Primary optimization: among policies satisfying both constraints, minimize `max_g C_g(pi)`. Report the entire per-model metric vector. This is a constrained robust objective, not a single score that hides calibration, query risk or cost.

Different models can have different outcome probabilities, so local minimax recursion can be time-inconsistent: the worst model at one branch may not be worst at the root. Exact bounded search must retain the Pareto frontier of per-model `(expected_cost, expected_loss, tail_mass)` vectors or prove an equivalent valid recursion. Do not allow rectangular model switching between action steps unless separately declared.

Fixed-plan comparator chooses an intervention subset in advance and applies the same stopping decision and risk constraints after observing its joint outcome tuple. Adaptive and fixed policies must use identical priors, full joint laws, action costs and feasibility.

## 4. Synthetic known-answer case: noisy but useful adaptive screen

Two biological worlds have opposite query decisions. Two complete generative models encode prior sensitivity and slightly different screen likelihoods. Both models use the same joint outcome mechanism for `confirm`.

Actions:

| Action | Cost | Outcome |
|---|---:|---|
| `screen` | 1 | `signal_true` or `signal_weak` |
| `confirm` | 5 | `confirm_true` or `confirm_false` |

Model `balanced` prior is 0.5 / 0.5 over TRUE / FALSE. Model `prior-shift` prior is 0.4 / 0.6.

Conditional screen likelihoods:

| Model | P(signal_true | TRUE) | P(signal_true | FALSE) |
|---|---:|---:|
| balanced | 0.80 | 0.05 |
| prior-shift | 0.70 | 0.05 |

For both models, confirm has 0.99 probability of the outcome matching the true query state. The fixture explicitly enumerates the **joint distribution** over `(screen, confirm)` outcome pairs for every model × world; the oracle consumes those four joint atoms per latent state. It does not infer independence from marginals.

Known adaptive policy:
- run `screen`;
- if `signal_true`, decide TRUE;
- if `signal_weak`, run `confirm` and decide by its result.

Model-specific exact results:

| Metric | balanced | prior-shift |
|---|---:|---:|
| P(signal_weak) | 0.575 | 0.690 |
| adaptive expected cost | 3.875 | 4.450 |
| adaptive expected query loss | 0.03075 | 0.03690 |
| max terminal robust posterior risk | shared across models; computed by oracle | shared across models; computed by oracle |

The fixed plan `confirm` alone costs 5, with expected query loss 0.01 in each model. The adaptive policy has worst-model expected cost 4.45 and worst-model expected query loss 0.0369, both within epsilon = 0.04, giving 0.55 effort units / 11% expected-cost saving versus the fixed plan. The screen-only policy fails the expected-risk constraint.

Risk thresholds in the fixture: epsilon = 0.04; tau = 0.05; delta = 0.45. The oracle must compute tail probabilities from terminal history masses rather than infer them from the maximum posterior risk alone.

This known answer is a finite synthetic calculation, not evidence that a biological screen has these likelihoods or costs.

## 5. Required adversarial cases

1. Correlated repeat measurements where product-of-marginals would double-count evidence and falsely pass epsilon.
2. Two priors yield conflicting posterior decisions at a history; one common terminal decision must be used.
3. A low-probability high-risk history: expected loss may pass while the tail bound fails.
4. A policy whose terminal expected risk passes but whose worst-history risk is high; report both without conflating them.
5. A nuisance model's identity is switched between steps; reject the run.
6. A generative model is chosen opportunistically after outcomes; reject the run.
7. A zero-probability history is supplied as a reachable policy branch; reject it.
8. An action outcome distribution does not sum exactly to one; reject it.
9. Joint atom vectors are duplicated or omit an action; reject them.
10. Search budget exhaustion returns INCOMPLETE, never an exact risk/cost optimum.
11. Deterministic point-mass and epsilon=0 special case reproduces #10823.
12. Disallowed or repeated interventions never enter a policy.

## 6. Comparators

- deterministic support-only exact policy (#10823);
- fixed-prior Bayesian adaptive policy;
- robust fixed non-adaptive subset;
- expected information gain / mutual information greedy;
- single-step decision value of information (#10670);
- expert-authored sequential decision tree;
- random feasible policy;
- Monte Carlo policy search compared with exact rational small-instance Pareto-frontier oracles.

All receive identical complete joint laws, frozen prior/model set, loss, risk thresholds, costs, feasibility and compute budgets.

## 7. Benchmarks and promotion targets

Report per model and worst case:
- expected query loss;
- tail probability above tau;
- worst terminal-history posterior risk;
- expected intervention cost;
- fixed-plan optimum and adaptive saving;
- calibration, proper scores, interval coverage and sharpness;
- uncertainty set / prior sensitivity and policy abstention;
- acquisition, reviewer, inference and rework costs.

Aspirational targets (not results):
- 100% exact rational expected-risk and policy-optimum agreement on bounded fixtures;
- 0 promotions when any model exceeds epsilon or delta;
- 0 false independence promotions from product-of-marginals shortcuts;
- exact recovery of #10823 for point-mass / epsilon-zero inputs;
- 0 model/nuisance switching within a path;
- nominal 90% predictive intervals with coverage within a preregistered tolerance on held-out animals/targets/sessions before biological promotion;
- at least 20% lower worst-case expected effort than exact fixed plans on preregistered tasks where adaptivity is useful, without violating either risk constraint.

## 8. Claim ceiling

Exactness applies only to the finite declared worlds, complete generative models, priors, full joint outcome laws, loss/risk thresholds and intervention universe. It does not establish that model uncertainty is complete, that priors are objectively true, that mechanisms are causal, or that wet-lab outcomes follow the supplied laws.