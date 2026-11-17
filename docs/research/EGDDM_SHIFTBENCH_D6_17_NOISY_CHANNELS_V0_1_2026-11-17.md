# EGDDM ShiftBench D6.17 — Noisy evidence channels and dependence-aware posterior risk v0.1

**Date:** 2026-11-17  
**Status:** evaluator prototype and synthetic known-answer controls; no Rust execution or real-agent performance claim  
**Related:** [Issue #12529](https://github.com/Luminous-Dynamics/symthaea/issues/12529), D6.15 risk-bounded acquisition, D6.16 bounded-prior credal risk.

## Research question

Can an engineering agent make defensible decisions when both the distribution over possible worlds and the channels that produce its observations are uncertain?

D6.16 qualifies prior ambiguity but treats observations as deterministic filters. D6.17 introduces evaluator-qualified likelihood intervals for noisy evidence channels, calibration lifecycle checks, and explicit dependence modeling. A measurement is not a fact: it is an output emitted by a source under a particular calibration, configuration, environment, authority and time window.

## Paradigm shift

**Do not trust a source label, a sensor's confidence, or a source's self-reported independence. Qualify the observation-generating process.**

The evaluator separates:
1. bounded-simplex prior ambiguity over admissible hypotheses;
2. per-hypothesis likelihood intervals for every modeled channel outcome;
3. source identity and dependence groups;
4. independent calibration authority, scope, validity and revocation state;
5. exact-set conditional-independence factorization certificates;
6. hidden actual hypothesis and action-safety labels.

Candidate-visible likelihood bounds, source group labels, action policies, priors and option availability are checked against evaluator-authoritative copies. The candidate cannot edit a calibration certificate, promote a source to an independent group, widen its own safety budget, or change the decision cutoff.

## Mathematical contract

Let p_i be prior mass for hypothesis i, l_i^p <= p_i <= u_i^p, and sum_i p_i = 10,000. For evidence outcome y from channel factor c, the independently qualified likelihood interval is:

    lower_q[i,c,y] <= P(y | h_i,c) <= upper_q[i,c,y].

For an accepted factorization into conditionally independent joint factors, compose likelihood bounds with outward rounding. Lower bounds round down; upper bounds round up. For action a, let H_a be unsafe hypotheses and G_a be safe hypotheses. Robust posterior risk is:

    sup over admissible p and q:
      sum(i in H_a) p_i q_i
      -------------------------
      sum(i in H_a union G_a) p_i q_i

For binary action-safety labels, the likelihood endpoint maximizing risk is upper_q_i for unsafe hypotheses and lower_q_i for safe hypotheses. For a candidate risk threshold r in basis points, maximize:

    sum(i in H_a) p_i (10,000-r) upper_q_i
      - sum(i in G_a) p_i r lower_q_i.

The bounded-simplex maximizer is obtained by assigning all lower prior masses first, then allocating the remaining mass to hypotheses in descending coefficient order up to their upper bounds. Binary search over integer r in [0, 10,000] gives a conservative 1-basis-point upper risk bound without floating-point optimization or an external LP solver.

The evaluator also computes a lower bound on observed-evidence support by minimizing sum_i p_i lower_q_i over the bounded simplex. Zero support is **unknown evidence**, never zero risk. A zero/unknown support result blocks action.

Every factor's outcome intervals must themselves define a feasible likelihood distribution for each hypothesis: outcome lower bounds sum to at most 10,000, upper bounds sum to at least 10,000, and nominal outcome probabilities sum exactly to 10,000.

## Dependence is a first-class trust boundary

Different source names are not evidence of independence. Factors sharing an optical path, power supply, clock, firmware, upstream provider, test fixture, operator or calibration regime are modeled as one joint factor when acquired together. Separate factors are multiplied only when the evaluator holds a certificate for the **exact set** of dependence groups used, with conditional mutual-independence scope, authority, action scope, and validity at both observation and decision time.

Pairwise independence is deliberately insufficient for three or more factors: pairwise-independent variables need not be mutually independent. A certificate must cover the exact factorization set, not a collection of pairwise name comparisons. If no exact-set qualification exists, the evaluator refuses the factorization and the decision cannot be justified.

v0.1 uses a conservative fail-closed rule for unqualified dependence: evidence from the unqualified set is not multiplied or used to narrow the prior. A later version may add a joint-channel copula / partial-dependence ambiguity set; it must not silently default to independence.

## Calibration lifecycle semantics

A factor is usable only when its evaluator-held calibration certificate:
- is issued by a non-empty independent authority;
- explicitly covers that factor;
- was valid at the evidence observation time;
- remains valid at the decision cutoff; and
- was not revoked at or before either time.

This intentionally makes the decision-time validity condition explicit. A certificate valid when a measurement was taken but expired or revoked before the decision does not silently remain current. Production deployments may adopt a different renewal policy, but it must be pre-registered and tested.

The evaluator does not verify a real calibration authority, cryptographic signature, measurement chain or physical instrument. It consumes supplied evaluator-side qualification records only.

## Required controls

1. Point-likelihood optimism: nominal posterior risk is below a reversible envelope while the robust likelihood/prior bound exceeds it.
2. Joint correlated sensor: two optical sources share a path; the joint likelihood model is used without multiplying their individual likelihoods.
3. False independence: two different source names in one dependence group are submitted as separate factors; the evaluator refuses the product.
4. Qualified factorization: two distinct groups combine only under an exact-set mutual conditional-independence certificate in scope.
5. Three-way factorization: pairwise certificates alone do not authorize a three-group product.
6. Calibration expired after observation but before decision.
7. Calibration revoked after observation but before decision.
8. False-positive / false-negative channel outcomes with held-out likelihood intervals.
9. Adversarial outcome impossible under the evaluator's actual world.
10. Zero or unknown evidence-support lower bound.
11. Source drift, degradation and contamination widen likelihood intervals rather than preserving obsolete certainty.
12. Future, backdated or not-yet-available evidence is rejected.
13. Candidate changes to prior, policy, likelihood bounds, dependence group, calibration reference, availability or cutoff are rejected.
14. Always-act and always-abstain fail paired positive/negative controls.

## Baselines and evaluation

Compare against D6.16 prior-only credal posterior; nominal Bayesian point-likelihood update; fixed-prior / fixed-likelihood Bayesian parity control; robust prior with point likelihoods; point prior with robust likelihoods; full robust prior + likelihood interval evaluator; naive independent sensor multiplication; exact joint-channel model; exact-set qualified factorization; source ablation / single-source baselines; always-act, always-abstain and expert/operator adjudication.

Compare the process dimensions with [RigorBench](https://arxiv.org/abs/2606.22678), the trajectory and multi-source failure attribution dimensions with [TraceTrial-Bench](https://arxiv.org/abs/2610.32285), and trajectory-level calibration with [Uncertainty Quantification for LLM Agents](https://arxiv.org/abs/2609.07395). These benchmarks are complementary; they do not share one scalar outcome and should not be collapsed into a leaderboard score.

Metrological guidance is also relevant: traceability alone does not ensure measurement uncertainty is adequate for a purpose, and the traceability chain includes uncertainty, procedure, competence and calibration intervals. See [JCGM 200:2012](https://www.oiml.org/en/files/pdf_v/v002-200-e12.pdf).

## Metrics — never trade critical safety for cost

Report separately by consequence and source family: nominal and worst-case posterior risk; prior-optimism and likelihood-optimism gaps; robust evidence-support lower bound; likelihood interval coverage on held-out outcomes; Brier score / log score and reliability curves by source and trajectory horizon; false-safe and false-alarm rates; source drift detection delay; calibration validity / revocation errors; false independence and unqualified-factorization attempts; common-mode survival rate; useful action coverage and unnecessary deferral; evidence acquisition count and D6.15 cold / warm / retry cost distributions; p50 / p95 / p99 total decision cost and risk-cost Pareto frontier.

No weighted score may trade a false-safe critical action, a hard-gate violation, or correlation double-counting against lower cost. Report hard safety failures as separate gating outcomes.

## Proposed scale and targets

- Development: 12 deterministic known-answer controls spanning at least four source types.
- Validity pilot: 8 channel-fault families × 6 hypothesis compositions × 3 seeds = 144 decisions.
- Research suite: 10 consequence classes × 8 channel/dependence families × 5 seeds = 400 episodes.
- At least three independently adjudicated calibration strata and a held-out temporal drift split.
- Zero false-safe actions in registered high-consequence controls; zero false independence claims; 100% known-answer source/temporal/correlation controls; at least 90% held-out likelihood-interval coverage; at least 95% justified action coverage on qualified positive controls. These remain targets pending calibration corpus adjudication and power analysis.

For zero observed failures in 400 independent episodes, the approximate one-sided 95% binomial upper bound is still about 0.75% (rule of three). Therefore, 400 episodes is a research pilot, not evidence of zero real-world risk. Episodes also need clustering-aware confidence intervals when seeds share scenarios, sources or models.

## D6.15 / D6.16 integration and cost frontier

Keep risk and cost as a constrained multi-objective evaluation. First reject any policy violating the registered risk envelope or hard gate. Among admissible policies, compare measured acquisition cost, p95 tail cost, evidence support and useful action coverage. Do not average risk and cost into one reward.

For each agent and matched model/harness, publish point-prior + point-likelihood baseline; D6.15 acquisition with point probabilities; D6.16 robust prior with deterministic observations; D6.17 robust prior + noisy channels + dependence controls; full revalidation; cheapest-first; and human/operator evidence plan.

Use paired scenario seeds and identical runner, model version, token/tool limits and time budgets. Record cold, warm and retry cost separately; disclose timeouts, skipped checks and censored runs.

## Implementation boundary and claim ceiling

D6.17 v0.1 is an evaluator over supplied bounded priors, per-outcome likelihood intervals, calibration metadata and exact-set factorization certificates. It is not a sensor-calibration method, proof of source integrity, cryptographic attestation verifier, general solution to arbitrary dependence uncertainty, real-world probability guarantee or production autonomy mechanism.

Rust format, compilation, tests, Clippy and CI have not been executed in the current environment. Synthetic fixtures and independent small-case mathematical checks do not establish runtime correctness or empirical validity. Keep the PR draft until exact-head execution and independent adjudication complete.

## Next research increments

- D6.17.1: joint likelihood tables for larger common-cause groups and partial-dependence ambiguity (Fréchet / copula bounds).
- D6.17.2: online drift detection with delayed labels and calibration-set coverage monitoring.
- D6.17.3: adaptive acquisition policy over noisy outcomes, with expected value of sample information and worst-case cost.
- D6.18: evidence-carrying action certificates linking world, observation channel, calibration, policy version, cutoff, risk calculation and execution receipt.
- D6.19: blind evaluation with separate runner/evaluator identities, read-only mounts, canary oracle files, access logs and adversarial leakage tests.
