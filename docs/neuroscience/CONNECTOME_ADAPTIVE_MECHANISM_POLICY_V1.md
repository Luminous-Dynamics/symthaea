# Exact nuisance-robust adaptive mechanism experiment policies V1

**Tracking:** #10743 — SCI-CONNECTOMICS-095
**Composes:** #10603, #10670, #10737, #9587, #9522, #9575, #9578, #9255
**Status:** finite synthetic exact-oracle contract; no biological result is claimed.

## 1. Objective

Select a sequence of experiments adaptively, based on observed outcomes, until both the scientific query and the required mechanism class are identified under every still-compatible declared nuisance profile. Minimize worst-case-over-nuisance expected total experiment cost.

This extends single-action mechanism identifiability (#10737) and query-directed intervention selection (#10670). It does not replace #10603's reconstruction-world semantics, #9587's blind intervention benchmark, or biological adjudication.

## 2. State and posterior semantics

Mechanisms `m` have a frozen prior `p(m)`, binary scientific query `q(m)`, and mechanism class `k(m)`. A nuisance profile `eta` supplies the categorical observable distribution `P(y | m, a, eta)` for each feasible action `a`.

For observed action/outcome history `h`, the nuisance-specific posterior is:

`p_eta(m | h) ∝ p(m) × Π_(a,y in h) P(y | m,a,eta)`.

A nuisance profile is eliminated only when the observed history has zero marginal likelihood under the entire profile. If every nuisance profile assigns zero probability to an observation, return `OUT_OF_MODEL_OBSERVATION`; do not fabricate a posterior or label the event a negative biological response.

The policy is one common mapping from observable history to the next action. It may not depend on the unknown true nuisance profile. This is the key anti-leakage invariant.

## 3. Terminal identification

At each history, for every still-compatible nuisance profile:
- query Bayes 0–1 risk is `min(P(q=true | h,eta), P(q=false | h,eta))`;
- mechanism-class Bayes 0–1 risk is `1 - max_c P(k=c | h,eta)`.

A node can STOP only when:
1. query risk is at or below `alpha_query` for every surviving nuisance profile;
2. mechanism-class risk is at or below `alpha_mechanism` for every surviving nuisance profile;
3. the Bayes-optimal query label is identical across all surviving nuisance profiles;
4. the Bayes-optimal mechanism class is identical across all surviving nuisance profiles.

If every mechanism has the same query decision, query identification is already satisfied (vacuously); likewise, if every mechanism has the same mechanism class, mechanism identification is already satisfied. The policy must not demand an unnecessary opposing pair.

A reachable outcome branch that does not satisfy all terminal requirements by the frozen maximum depth invalidates the entire policy certificate. Expected success on average is not enough.

## 4. Exact adaptive objective

Actions have strictly positive costs and may be used at most once. The maximum policy depth is frozen. At each node the policy chooses STOP only if terminal conditions pass, otherwise one feasible remaining action.

For each candidate policy, compute a vector of expected future costs, one component per nuisance profile. Under nuisance profile `eta`, predictive outcome probability at history `h` is:

`P_eta(y | h,a) = Σ_m p_eta(m | h) P(y | m,a,eta)`.

The expected cost component is the immediate action cost plus the outcome-probability-weighted expected child cost. A policy is shared across nuisance profiles; children are selected jointly across all outcome branches.

Among policies that terminate correctly on every reachable branch, choose the one minimizing `max_eta E_eta[total_cost]`. Tie-break by sum of expected costs over the nuisance profiles, then a stable canonical policy digest.

### Why a Pareto frontier is necessary

A tempting but invalid strategy is to independently select the cheapest child policy for each nuisance profile. The deployed policy cannot observe which nuisance profile is true. Child policies must therefore be the same function of observed outcomes across profiles.

V1 maintains the non-dominated vectors of per-profile expected costs for each state. For each action, it combines child frontiers over every possible outcome, probability-weights each vector under each nuisance profile, adds the current action cost, and prunes dominated vectors. It chooses the final vector by the robust minimax objective.

Search is exact only within its declared state/frontier budget. If the Cartesian policy frontier exceeds the cap, return `SEARCH_INCOMPLETE`; never return the best-so-far vector as an exact optimum.

## 5. Bounded V1 limits

The starter oracle is deliberately small:
- at most 6 mechanism hypotheses;
- at most 3 nuisance profiles;
- at most 5 actions;
- at most 3 categorical outcomes per action;
- maximum depth 4;
- maximum 100,000 retained Pareto vectors across the run (including intermediate frontiers).

Inputs that exceed a limit or produce an incomplete frontier return a typed incomplete status, not an approximation. Probabilities must be finite, non-negative and sum to one. Priors must be finite, non-negative and sum to one.

## 6. Starter synthetic known-answer

Mechanisms:

| Mechanism | Prior | Query | Mechanism class |
|---|---:|---:|---|
| `DIRECT` | 0.40 | true | DIRECT_SYNAPTIC |
| `EXTRA` | 0.40 | true | EXTRASYNAPTIC_LATENT |
| `SHARED` | 0.20 | false | SHARED_INPUT |

Two nuisance profiles slightly change the probability of the first-stage supporting outcome.

- Action `A` costs 1. It returns `SUPPORTING` for DIRECT and EXTRA, and `REFUTING` for SHARED, with probabilities pinned separately for each nuisance profile.
- If `REFUTING` is observed, SHARED is the only compatible mechanism and the policy stops.
- If `SUPPORTING` is observed, DIRECT and EXTRA remain compatible; action `B`, cost 2, robustly separates them under both nuisance profiles.

The exact adaptive policy therefore runs A first and runs B only after SUPPORTING. It must have strictly lower worst-case expected cost than the fixed A+B bundle, while all reachable branches still meet both terminal risk thresholds.

A second fixture collapses B's distributions under one nuisance profile; no robust policy exists within the horizon, even if B remains useful under the nominal profile.

The fixture includes exact expected-cost vectors and an independently computed policy digest. It tests adaptivity and common-policy semantics, not biological experiment performance.

## 7. Required baselines

All methods receive the same worlds, priors, nuisance profiles, observation models, feasibility constraints, horizon and computational budget:

1. Exact non-adaptive fixed-bundle optimum.
2. Greedy one-step expected information gain.
3. Greedy one-step reduction in query/mechanism Bayes risk.
4. Expert-ranked sequence.
5. Random feasible sequence.
6. Exact Pareto-frontier adaptive oracle.
7. GO-CBED (Zhang et al., TMLR 2026): non-myopic, goal-oriented sequential Bayesian design that maximizes EIG on specified causal quantities with an amortized policy and variational estimator. Reproduce its own estimator/objective; compare only on matched synthetic tasks.
8. GoBOED (Go et al., 2026 preprint): goal-driven Bayesian design under model uncertainty, optimizing decision loss rather than parameter entropy. Treat as a preprint comparator and do not equate its approximate posterior/decision objective with this finite nuisance-worst-case oracle.

Sources: [GO-CBED](https://tmlr.infinite-conf.org/paper_pages/6dVCZT7pO2.html); [GoBOED](https://arxiv.org/abs/2605.26093).

Do not collapse EIG, Bayes risk, worst-case expected cost, finite-sample error and multiplicity-adjusted biological inference into one score.

## 8. Adversarial controls

- One nuisance profile assigns zero likelihood to an observed outcome: remove only that profile.
- All nuisance profiles assign zero likelihood: `OUT_OF_MODEL_OBSERVATION`.
- One reachable leaf remains unresolved: reject the full policy.
- The exact state or Pareto-frontier budget is exceeded: `SEARCH_INCOMPLETE`.
- A profile-specific policy would be cheaper than every common policy: reject the profile-specific result as invalid.
- A greedy EIG result differs from exact minimax: preserve exact oracle optimum and report regret.
- Reusing an action, selecting a disallowed action, malformed prior/probability, missing Cartesian prediction, or stale model digest: fail closed.
- Unanimous query or mechanism classes are terminal for that dimension without a spurious opposing-pair requirement.
- Lockbox data influence priors, nuisance profiles, thresholds or policy: reject the run.

## 9. Metrics and promotion targets

Report separately:
- worst-case expected total experiment cost;
- expected cost by nuisance profile;
- fixed-bundle and adaptive-optimum regret;
- policy depth, branch count, state count, Pareto-frontier peak and optimization compute;
- terminal query and mechanism risks by nuisance profile;
- fraction of reachable leaves resolved;
- explicit no-policy, incomplete, and out-of-model dispositions;
- acquisition, inference, biological replication, review and analysis cost.

Aspirational targets, not current results:
- 100% exact policy and cost agreement on bounded known-answer fixtures;
- 0 false terminal certificates on any reachable branch;
- 0 nuisance-profile-specific policy leakage;
- 100% fail-closed behavior on incomplete search and out-of-model observations;
- at least 20% lower worst-case expected experiment cost than the strongest frozen fixed/greedy baseline on preregistered held-out cases, with paired uncertainty supporting improvement;
- no reduction in query/mechanism identification thresholds to obtain cost improvements.

## 10. Biological staging and claim ceiling

Start with synthetic exact fixtures. Later evaluate with animal-level independent units on the C. elegans signal-propagation atlas (23,433 stimulated/recorded neuron pairs across 113 animals) and Drosophila CPG competing mechanisms under exact source custody. The C. elegans atlas measures effective signal propagation over all paths, including extrasynaptic signalling; it is not a direct monosynaptic ground-truth table. Use explicit abstention and mechanism-class boundaries.

Reference: [Randi et al., Nature 2023](https://www.nature.com/articles/s41586-023-06683-4).

A passing finite oracle establishes exact optimization only for its bounded declared model and nuisance universe. It cannot prove biological mechanism, nuisance completeness, or in-vivo intervention efficacy. Biological promotion requires source-bound data, blinded animal/session lockboxes, independent mechanism adjudication, adequate replication and prospective validation.