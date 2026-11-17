# EGDDM ShiftBench D6.17.2.2.1-alpha — Stochastic adaptive policy scoring over evaluator-owned joint kernels

**Date:** 2026-11-17  
**Status:** fixed-policy scorer prototype; deterministic parity and common-cause synthetic known-answer fixtures; Rust execution pending  
**Related:** D6.17.2.2 parent issue https://github.com/Luminous-Dynamics/symthaea/issues/12630; stochastic scorer issue https://github.com/Luminous-Dynamics/symthaea/issues/12653; joint-kernel validator PR https://github.com/Luminous-Dynamics/symthaea/pull/12655; deterministic adaptive oracle PR https://github.com/Luminous-Dynamics/symthaea/pull/12649; run-level cost receipts PR https://github.com/Luminous-Dynamics/symthaea/pull/12658.

## 1. Research positioning: what is already SOTA

The field already has mature work on sequential, non-myopic and robust experimental design. We should not claim that this branch invents adaptive acquisition, optimal stopping, robust Bayesian design, or stochastic dynamic programming.

Relevant comparators available by 17 November 2026:

- Huan, Jagalur and Marzouk, Optimal experimental design: Formulations and computations, Acta Numerica (published online 4 September 2024). The review covers Bayesian/decision-theoretic OED, information criteria, combinatorial observation selection and sequential non-myopic design. https://doi.org/10.1017/S0962492924000023
- Cheng and Huan, Optimal stopping for sequential Bayesian experimental design, arXiv:2509.21734 (2025; revised June 2026). This treats design and stopping as coupled finite-horizon decisions, and evaluates learned value-based policies on numerical experiments. https://arxiv.org/abs/2509.21734
- Zhang, Dong, Liu and Huan, Goal-Oriented Sequential Bayesian Experimental Design for Causal Learning, Transactions on Machine Learning Research (September 2026). GO-CBED optimizes non-myopic intervention sequences for a specified causal quantity of interest using an amortized policy and variational objective. https://www.jmlr.org/tmlr/papers/
- Wen et al., Robust Sequential Experimental Design for A/B Testing, ICML 2026 / PMLR 306. It addresses sequential design under model misspecification for contextual bandit and dynamic settings, with theoretical MSE bounds and empirical studies. https://proceedings.mlr.press/v306/wen26i.html
- Iyengar, Robust Dynamic Programming, Mathematics of Operations Research (2005). This establishes robust dynamic-programming results under rectangular sets of conditional transition measures; the role of rectangularity is central to what EGDDM must make explicit. https://doi.org/10.1287/moor.1040.0129
- NIST's ongoing 2026 Building Evaluation Probes into Agentic AI programme integrates adversarial verifiers into workflows and produces machine-readable audit trails for claim grounding, including faithfulness, completeness and sufficiency. https://www.nist.gov/programs-projects/building-evaluation-probes-agentic-ai
- The 2026 review From benchmarks to deployment argues that task-success scores often omit deployment-critical safety, security, cost and trajectory reliability. Its numerical conclusions refer to the 15 benchmarks it sampled, not every benchmark or industrial evaluation programme. https://doi.org/10.1007/s10462-026-11571-0

### Competitive interpretation

GO-CBED is an important comparator, not a straw man. It is more sophisticated than greedy information gain: it plans whole intervention sequences around a causal question. Optimal stopping work already treats continuation and termination jointly. Robust sequential A/B design already addresses misspecification.

EGDDM's defensible direction is complementary:

1. Decision-specific assurance: acquisition value is tied to whether a consequential action can be robustly justified, not only to information gained.
2. Dependence-preserving uncertainty: observations are represented by evaluator-owned joint outcome kernels; marginals are not multiplied to manufacture conditional independence.
3. Executable policy qualification: every terminal action is replayed against hard gates, prior uncertainty, branch likelihood intervals and read-only deferral semantics.
4. Unknown-world admission: D6.18's open-world reserve must qualify every history; the registered hypothesis set is not silently treated as complete.
5. Execution economics: complete run receipts measure queueing, retries, unknown outcomes and total trajectory cost; marginal p95 sums are not mislabelled as full-run tail guarantees.
6. Independent evidence: candidate policies cannot authoritatively set their own kernels, costs, action permissions or model-validity records.

The field-changing claim, if evidence eventually supports it, is not "we choose better experiments than all OED." It is "we make adaptive decision policies independently auditable under joint outcome dependence, model uncertainty, hard action constraints and measured execution cost."

## 2. Stochastic policy model

For hypothesis/world h, an evaluator-owned joint kernel has finite rows j in J_h. Each row describes one complete outcome vector over all registered acquisition options:

\[
\underline q_{hj}\le q_{hj}\le\overline q_{hj},\qquad
q^0_{hj}\in[\underline q_{hj},\overline q_{hj}],\qquad
\sum_j q_{hj}=1,\qquad \sum_jq^0_{hj}=1.
\]

The bounds must form a non-empty bounded simplex:

\[
\sum_j\underline q_{hj}\le1\le\sum_j\overline q_{hj}.
\]

For any event A contained in J_h, the exact probability interval is:

\[
\underline q_h(A)=\max\left(\sum_{j\in A}\underline q_{hj},
1-\sum_{j\notin A}\overline q_{hj}\right),
\]

\[
\overline q_h(A)=\min\left(\sum_{j\in A}\overline q_{hj},
1-\sum_{j\notin A}\underline q_{hj}\right),
\quad
q^0_h(A)=\sum_{j\in A}q^0_{hj}.
\]

The submitted policy is a finite tree. An acquisition node chooses one registered option; the observed outcome selects a branch. A stop node either invokes a consequential action or performs the unique evaluator-approved globally safe read-only deferral. The scorer replays every complete joint outcome vector through the entire submitted tree. It does not assume the channels are independent.

For each terminal leaf l, the row membership set A_hl defines the event that hypothesis h's joint outcome vector reaches that leaf. Its event probability interval is obtained by the exact formula above. Those intervals are passed to the D6.17.1 robust posterior-risk calculation with the evaluator-owned bounded prior.

A consequential action is justified only when:
- event support has a strictly positive lower bound;
- worst-case posterior risk satisfies the registered action threshold;
- a hard-gate action has zero robust residual risk;
- the entire path stays within the evaluator-owned acquisition budget.

Read-only deferral is permitted independently of evidence support, but only for the unique evaluator-owned read-only action that is safe under every registered hypothesis. Deferral is not represented as proof that a consequential action is justified.

## 3. Exact expected-cost calculation

For a fixed policy, each complete joint row determines a path cost c_hj, the sum of evaluator-owned fixed option costs along that path.

Conditional nominal expected cost:

\[
C^0_h(\pi)=\sum_{j\in J_h}q^0_{hj}c_{hj}.
\]

Conditional worst-case expected cost:

\[
C^{\max}_h(\pi)=
\max_{q_h\in\mathcal Q_h}\sum_{j\in J_h}q_{hj}c_{hj}.
\]

This is a linear programme over a bounded simplex. It is solved exactly by starting with each row's lower mass and allocating remaining mass to rows in descending path-cost order until their upper bounds are reached.

For prior credal set P, under the explicit product-set assumption that each hypothesis's kernel ambiguity can vary independently:

\[
C^{\max}(\pi)=\max_{p\in\mathcal P}\sum_h p_h C^{\max}_h(\pi).
\]

The outer optimization is another bounded-simplex linear programme. The scorer also evaluates the nominal prior/nominal kernel expectation and sums all option costs as the full-revalidation comparator.

**Rectangularity limit:** these formulas are exact for the registered product set. If one latent physical cause couples kernel uncertainty across different hypotheses, the product set is not the correct model. Treating it as rectangular may be conservative and can distort policy ranking. D6.17.2.2.3 is required to represent and compare globally coupled/shared-latent ambiguity sets. No result from this alpha should be generalized beyond its stated product-set assumption.

The cost model in this scorer is deterministic per option. Real cold/warm/retry/queue/provider cost distributions are handled by D6.17.2.3 run receipts and will be joined in a later integration. They are not inferred from this synthetic scorer.

## 4. Safety–utility envelopes, not just one coverage number

A nominal coverage score alone can conceal large prior and joint-kernel ambiguity. For every hypothesis, the scorer unions all terminal joint rows whose leaf performs a justified, within-budget act. It computes the exact event lower/nominal/upper probability for that union, then aggregates across the prior credal set:

\[
\underline{\mathrm{Coverage}}=
\min_{p\in\mathcal P}\sum_h p_h\underline q_h(A_h),
\]

\[
\mathrm{Coverage}^{0}=\sum_h p^0_h q^0_h(A_h),
\qquad
\overline{\mathrm{Coverage}}=
\max_{p\in\mathcal P}\sum_h p_h\overline q_h(A_h).
\]

The scorer reports the same lower/nominal/upper envelope for unsafe-action mass. It separately reports the actual-hypothesis unsafe probability interval. The nominal point is not presented as the robust guarantee.

This gives a safety–utility envelope: how much justified action coverage survives the uncertainty set, and how much unsafe action mass the set still permits.

## 5. Reference fixtures

### A. Deterministic limit

The deterministic fixture contains four hypotheses and one joint outcome vector per hypothesis. It must preserve the D6.17.2.1 adaptive result:

- nominal justified action coverage: 6,000 bps (60%);
- prior-robust justified coverage lower/upper bounds: 5,000 / 7,000 bps;
- nominal expected acquisition cost: 1,300 micro-USD;
- worst-case expected acquisition cost: 1,420 micro-USD;
- full revalidation cost: 1,500 micro-USD;
- worst-case path cost: 1,500 micro-USD;
- no robust-risk or budget violations.

The test reads the original D6.17.2.1 expected report and asserts parity for the deterministic-policy metrics. It is a deterministic-limit equivalence control, not a new benchmark win.

### B. Common-cause stochastic limit

The common-cause fixture has two hypotheses:
- safe-world prior: lower 8,000, nominal 9,000, upper 9,500 bps;
- common-cause hazard prior: lower 500, nominal 1,000, upper 2,000 bps.

Under safe-world, the two sensors each have a 50% nominal failure marginal, but the evaluator's joint kernel places 45% nominal mass on pass/pass and 45% on fail/fail. The independence shortcut would assign only 25% to each joint pair.

The policy releases only after pass/pass:
- conditional on safe-world, pass/pass has 4,500 bps nominal probability;
- unconditional nominal justified release coverage is 4,050 bps (40.5%) after multiplying by the 90% nominal safe-world prior;
- prior/kernel robust justified-coverage envelope is [3,200, 4,750] bps;
- nominal expected cost is 1,060 micro-USD;
- worst-case expected cost under the declared product set is 1,156 micro-USD;
- full revalidation cost is 1,500 micro-USD;
- no unsafe action mass in the unmutated fixture.

These are exact calculations on synthetic evaluator-owned inputs, not measured sensor performance or production cost savings.

A negative control marks release unsafe in the actual safe-world oracle label. The scorer must then report 4,500 bps nominal / 5,000 bps upper actual-world unsafe probability, 3,200 / 4,050 / 4,750 bps prior/kernel unsafe-action mass bounds, and one robust-risk-violating action leaf.

## 6. Implemented alpha slice

The Rust binary stochastic-adaptive-policy-scorer.rs:
- imports and invokes the D6.17.2.2 joint-kernel validator before scoring;
- requires exact candidate/evaluator copies of actions, priors, options and channel registry;
- validates current calibration scope and validity at the decision cutoff;
- requires evaluator-owned joint kernels, costs, safety labels and one actual-hypothesis label;
- validates complete option outcome branches, acyclic tree structure, reachability, no repeated acquisition on a path, availability and time ordering;
- requires a unique safe read-only deferral action;
- replays all complete joint outcome vectors through the candidate policy;
- derives leaf-wise likelihood intervals from joint row membership, then calculates nominal and robust posterior risk;
- calculates exact conditional nominal and worst-case kernel costs and outer prior-robust cost under the declared product-set assumption;
- reports robust lower/nominal/upper justified-action coverage and unsafe-action mass, actual-hypothesis unsafe probability, path/budget violations, leaf-level claim understatement and kernel provenance.

Candidate-supplied world_kernels, cost_options, hypothesis_safety, or oracle calibration records are rejected as unknown candidate fields. The evaluator-owned data are structurally separate from the policy proposal.

## 7. Benchmark design and industry comparison

The scorer is a fixed-policy evaluator, not yet a policy optimizer. Compare policy generators and stopping rules using the same evaluator and held-out environment.

**Baselines**
1. Full revalidation.
2. D6.15 minimum-sufficient evidence set.
3. Nominal greedy expected value of sample information.
4. Robust one-step EVSI.
5. Non-myopic GO-CBED-style goal-oriented sequential design.
6. Bayesian learned design-and-stopping policy (Cheng & Huan).
7. Robust sequential A/B design under misspecification (Wen et al.).
8. Fréchet-only robust joint-event fallback.
9. Conditional-independence multiplication as a deliberately unsafe negative control.
10. Exact finite stochastic policy enumeration on small instances.
11. Receding-horizon or learned policy with certified lower/upper gaps when exact search no longer completes.
12. Human/operator plan, always-act and always-defer controls.

Use identical task distributions, candidate action definitions, model/harness versions, permissions, runner classes, concurrency, acquisition menus and paired seeds. Distinguish policy-generation runtime from measurement/execution cost.

**Model and dependence benchmarks**
- Held-out marginal and joint-event interval coverage by hypothesis, outcome, device, site, environment and cause family.
- Nominal calibration, interval sharpness and Brier/log scores where the probabilistic assumptions justify them.
- Pairwise and higher-order common-cause miss rate.
- False-independence underestimation.
- Leave-one-common-cause-family-out transfer.
- Dependence drift, stale-kernel rejection and revocation latency.
- Open-world false-safe rate under D6.18 reserve and unknown outcome injections.

**Policy benchmarks**
- Robust lower / nominal / upper justified-action coverage.
- Robust upper unsafe-action mass and actual-world unsafe probability.
- Hard-gate / risk / budget violation counts.
- Exact policy regret when exhaustive enumeration completes.
- Cost regret to exact oracle on small trees, then certified optimality gaps on larger problems.
- Premature stopping, unnecessary abstention and value-of-information misranking.
- Sensitivity to prior widening, likelihood contamination and new unseen hypotheses.

**Execution economics**
- End-to-end full-run cost p50/p95/p99 and wall-time p50/p95/p99 from complete D6.17.2.3 receipts.
- Queue delay, cold/warm/retry costs, attempts per decision and unknown/censored outcome rate.
- Cost per robustly justified action at matched risk and coverage.
- Receipt integrity, provider reconciliation and independent replay failures.

### Proposed targets — experiments, not achievements

- Zero hard-gate, robust-risk and budget violations on registered critical controls.
- Zero false-safe actions on the fully adjudicated challenge suite; report confidence intervals rather than claiming zero real-world risk.
- At least 95% justified-action coverage on positive controls where the evaluator confirms sufficient information exists.
- At least 95% held-out joint-event interval coverage in each pre-registered critical stratum, while separately reporting interval width/sharpness.
- Zero accepted stale/revoked calibration or joint-kernel records.
- Exact zero policy regret on small complete finite search cases.
- Adaptive heuristic cost regret at most 1% against the exact oracle on a registered development slice, at matched robust coverage.
- At least 20% lower full-run median cost and 10% lower full-run p95 cost than full revalidation, at the same robust-risk envelope, after independent receipt replay.
- Begin with a 144-decision adjudicated validity pilot, then 400 development episodes. Expand to thousands of independent or cluster-aware episodes before rare-event or tail claims.

Targets must be frozen before held-out evaluation. If an approach fails, preserve and report the failure rather than weakening the threshold or changing the evaluation set.

## 8. Next research increments

1. D6.17.2.2.3 — Shared-latent, non-rectangular ambiguity: represent uncertainty sets that couple kernels across hypotheses; compare exact product-rectangular and coupled optimization on identical fixtures.
2. D6.17.2.2.4 — Exact policy search: enumerate finite stochastic trees and provide exact regret on small cases; for larger spaces, provide certified lower/upper policy-value gaps.
3. D6.17.2.2.5 — Goal-oriented stopping: use target-specific causal/value-of-information objectives to rank safety-feasible acquisitions, with a robust stop/defer decision and D6.18 unknown-world reserve.
4. Join D6.17.2.3 full-run cost receipts to policy nodes and observed joint outcomes.
5. Validate kernels with independently collected repeated-measurement data and family-aware holdout splits.

## 9. Claim ceiling

The alpha exactly evaluates a submitted finite policy over the registered finite joint-outcome credal kernels, bounded prior and fixed evaluator costs. Robust posterior risk, coverage and unsafe-mass envelopes assume independent per-hypothesis kernel ambiguity sets (a product/rectangular model). It does not optimize over the full policy space, establish empirical calibration, validate cross-hypothesis shared-latent ambiguity, model stochastic execution cost, prove hypothesis completeness, authenticate oracle ownership, or authorize real-world actions.

Rust compilation, tests, rustfmt, Clippy and CI have not been executed here. Keep this branch draft until exact-head toolchain validation and independent fixture replay.
