# Decision-Focused Value-of-Computation for NAS v0.1

**Status:** Rust source and test declarations added to a dependent research branch. Compilation, execution, calibration qualification and benchmark performance are unverified.

**Tracking:** [#9267 — adaptive value-of-computation scheduler](https://github.com/Luminous-Dynamics/symthaea/issues/9267).

**Implementation:** `crates/domains/symthaea-architecture-engineering/src/decision_focused_voc.rs`.

## 1. Research gap

The current multi-fidelity NAS series includes:
- metered proxy acquisition, a shared target-query oracle and final-fidelity confirmation in #9172;
- split-conformal learning-curve prediction and static guard slots in #9172;
- selection-conditioned finite-population false-pruning audits and a budget allocator in #9104;
- a bounded run-wide confidence ledger in #9329;
- a risk-bounded adaptive scheduler specification in #9267.

A next-action rule based only on interval-width reduction per cost is not decision-focused. Narrowing an interval that cannot change the eventual architecture recommendation may have no practical value. A measurement that changes the preferred top-K set may be valuable even when total interval width barely changes.

## 2. Paradigm shift: expected reduction in recommendation regret

Represent uncertainty about final architecture quality using a joint discrete posterior over final-fidelity candidate utilities. A posterior particle (j) has probability (p_j) and final utility (u_{j,i}) for every candidate (i).

For a fixed top-K recommendation, define Bayes top-K simple regret:

`R_K(p) = sum_j p_j * sum_{i in TopK(u_j)} u_{j,i}
           - sum_{i in TopK(E_p[u_i])} E_p[u_i]`

The first term is expected oracle top-K utility under the posterior. The second is the utility of the best posterior-mean top-K recommendation. For K=1 this reduces to Bayes simple regret for best-arm identification.

An action (a) has discrete possible observation outcomes (o) and likelihoods (P(o | j,a)). The planner applies Bayes' rule and calculates:

`VOI_K(a) = R_K(p) - sum_o P(o | a) * R_K(posterior after o)`

This is decision-focused expected value of sample information under the supplied posterior and likelihood model. It is not a frequentist confidence guarantee and it is not automatically robust to model misspecification.

The net decision value is:

`net(a) = VOI_K(a) - lambda * incremental_resource_cost(a)`

where lambda is a preregistered regret-equivalent cost penalty. The planner selects the action with the largest positive net value among actions that satisfy total budget and protected-reserve constraints. If none qualifies, it returns STOP with an explicit reason.

This is a NAS-oriented extension inspired by cost-aware best-arm identification, including DBCARE (Harding & Kandasamy, NeurIPS 2025). It is not a reproduction of DBCARE or a claim of its theorem.

## 3. Supported action classes

- **FidelityQuery:** purchase one next-fidelity observation for an active candidate.
- **DormantReactivation:** spend resource to revisit a previously dormant candidate.
- **SentinelAudit:** acquire an aggregate pruning-audit disposition. Only aggregate outcomes are allowed; candidate IDs and raw sentinel scores must stay in the evaluator.

The model is one-step and replanning-based: execute at most the selected action, update the posterior from the actual observation, reconcile actual cost, then call the planner again. It does not assume multiple future actions can be executed using a stale posterior.

## 4. Provenance and calibration contract

The planner requires:
- a pinned benchmark, metric, direction and evaluation task;
- SHA-256 digests for source artifact, posterior model and observation-likelihood model;
- calibration task IDs disjoint from the evaluation task;
- an upstream task-calibration qualification flag for the exact model;
- action-level bindings to the exact posterior, source and likelihood-model digests;
- finite posterior utilities and normalized posterior probabilities;
- a normalized observation-outcome distribution for every posterior particle.

The planner computes and checks canonical posterior and action-likelihood digests. Candidate/action/particle order and action/outcome order do not affect the canonical content digest when semantic content is unchanged.

The planner verifies supplied qualification metadata; it does not independently reproduce the cross-task calibration experiment. A false qualification flag, invalid digest, missing action outcome, inconsistent likelihood model, leakage of target task into calibration tasks, or mismatched provenance must fail closed.

For a sentinel-audit action, the observation categories are restricted to aggregate dispositions:
- `continue_pruning`
- `disable_pruning_insufficient_audit`
- `disable_pruning_observed_risk`

The raw sentinel identities and scores are not part of the policy input. The actual audit still must be produced by #9104 and recorded through the run-wide ledger in #9329. This planner cannot override a failed audit certificate.

## 5. Resource and reserve policy

v0.1 uses one declared resource unit per planning invocation (for example, GPU-seconds or epoch-equivalent units) and one explicit regret-equivalent cost penalty. It does not silently combine CPU time, accelerator time, energy, memory and currency.

When there is no defensible common unit or price, compare separate Pareto frontiers and do not force a scalar utility.

The caller supplies:
- total resource budget and already-spent units;
- spend already attributed to each action class;
- minimum reserves for fidelity queries, dormant reactivation and sentinel audit.

An action may spend its own reserve but cannot consume resources protected for other action classes. The returned plan records each action's total cost, expected regret reduction, regret reduction per unit, cost penalty, net value and feasibility disposition.

## 6. Authored test coverage

Thirteen Rust test declarations cover:
- the discrete-posterior value-of-information example (current regret 0.5, expected post-action regret 0.1, VOI 0.4);
- no-information actions and explicit stopping;
- top-K objective and lower-is-better metric semantics;
- protected dormant-reactivation reserves;
- hard budget exhaustion;
- calibration-task leakage and unqualified model rejection;
- provenance mismatch and non-normalized likelihood rejection;
- aggregate-only sentinel outcomes;
- deterministic action tie-breaking;
- canonical posterior/action-model digests;
- order-independent plan digest;
- malformed likelihood dimensions fail without panic.

These are authored tests, not passing tests. Rust compilation and execution are outstanding.

## 7. Qualification experiments

Compare at equal measured compute and separately at equal full-fidelity confirmation count:

1. Random full-fidelity and random early stopping.
2. Raw-score Successive Halving / Hyperband.
3. Calibrated lower-confidence-bound promotion.
4. Calibrated promotion with static optimistic guard slots.
5. Adaptive VoC using interval-width reduction per cost (baseline heuristic).
6. Decision-focused discrete-posterior VoC.
7. Decision-focused VoC with the run-wide pruning-risk ledger.
8. IISE and Optimal Multi-Fidelity Best-Arm Identification for K=1 under independently qualified fidelity-bias bounds.
9. DBCARE-inspired cost-risk stopping baseline; do not claim the original DBCARE theorem applies to heterogeneous multi-fidelity NAS.
10. Pinned ZAPS, CoRA-NAS, R-MF-NAS and HiP-BO variants where code and artifact availability allow faithful reproduction.

Use real, SHA-256-pinned NAS-Bench-201 CIFAR-10/100/ImageNet16-120 trajectories first. Add NAS-Bench-101 and at least one held-out family to assess task/search-space transfer. The raw artifact, imported architecture mapping, split manifest, posterior model, observation-likelihood model and run receipts must all be pinned.

Stress the model with rank crossings, late learners, multimodal final performance, heteroscedastic and heavy-tailed noise, misspecified likelihoods, task shift, sentinel-audit outcomes, dormant candidate recovery, and model calibration degradation.

## 8. Metrics and targets

Report:
- final-confirmed P@1/P@10/P@100, normalized regret and CVaR-10%;
- Bayes top-K regret predicted before action versus realized final regret;
- empirical calibration of posterior particles and discrete observation likelihoods;
- VOI prediction error against replayed actual outcome-conditioned regret reduction;
- net regret reduction per resource unit;
- cost-risk objective versus DBCARE-inspired and fixed-budget baselines;
- stopping quality, under-spend, over-spend and probability of premature stopping;
- dormant reactivation recall and false reactivation cost;
- audit reserve use, failed-audit fallback rate and run-wide audit-bound coverage;
- worst-task/worst-seed outcomes and quality/resource Pareto frontiers.

Initial engineering target, not a result: at equal total measured cost, improve median final normalized regret by at least 10% relative to the strongest pinned baseline without worsening P@1/P@10 materially or regressing any preregistered critical task by more than 5%.

A 100+ paired-seed confirmatory study or preregistered power analysis is required. No recalibration on held-out task labels is permitted.

## 9. Claim ceiling

This module computes exact one-step expected Bayes top-K regret reduction under a supplied finite posterior and discrete observation likelihood model. It is not a proof of posterior calibration, a guarantee of final top-K survival, a reproduction of DBCARE/HiP-BO/Optimal MF-BAI, or evidence of SOTA performance. Statistical pruning certificates remain in #9104/#9329 and are separate from the model-based acquisition value.

## References

- Harding & Kandasamy, [Balancing Performance and Costs in Best Arm Identification, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/b0dfe8236981ba84b4d98a7ed0426f0b-Abstract-Conference.html).
- Poiani et al., [Optimal Multi-Fidelity Best-Arm Identification, NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/hash/dc9e095f668044e7a0909a4ea3926beb-Abstract-Conference.html).
- Yang, Tan & Jin, [Best Arm Identification with Minimal Regret, JMLR 2026](https://www.jmlr.org/beta/papers/v27/24-1612.html).
- Cai, Xian & Huang, [HiP-BO, online 16 October 2026](https://www.sciencedirect.com/science/article/pii/S092523122602802X).
- [NASLib benchmark and learning-curve artifacts](https://automl.github.io/NASLib/).
