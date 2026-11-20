# Causal Memory Identification & Transfer Challenge (CMIT) v0.1

**Owner:** Symthaea / Broca MACE (#9516)  
**Status:** Research protocol. No real-world causal discovery, benchmark rank or SOTA advantage is claimed.

## 1. Research problem

Agent memory systems increasingly preserve facts, episodes, preferences, skills and trajectories. But a trajectory does not automatically identify why an outcome occurred.

The next capability should be **causal-design-aware memory**: preserve the method by which an action-effect claim was identified, the estimand it refers to, the assumptions that permit its use, and the evidence that those assumptions still hold in a later environment.

```text
action then outcome             != identified causal effect
randomized assignment           != treatment received
observational association       != causal identification
valid effect estimate           != effect that transports
deterministic error bound       != statistical confidence interval
receipt digest                  != proof that the design is sound
```

## 2. Method-specific tracks

| Method | Required qualification evidence | Adversarial controls |
|---|---|---|
| Randomized controlled design | Preregistered estimand, allocation schedule, eligible population, assignment probabilities, attrition, balance, interference, ITT/outcome reconciliation | Post-hoc registration, assignment override, poor overlap/balance, selective attrition, spillovers, outcome switching |
| Difference-in-differences | Treatment timing, comparison groups, pre-trend/event-study diagnostics, anticipation, estimator/aggregation assumptions | Divergent pre-trends, anticipation, composition change |
| Instrumental variables | Instrument definition, first-stage strength, exclusion/independence/monotonicity assumptions, weak-instrument-robust inference | Weak/invalid instrument, direct instrument effect, heterogeneous compliance |
| Regression discontinuity | Frozen cutoff, running variable, bandwidth, continuity and density/manipulation diagnostics | Sorting at cutoff, bandwidth cherry-picking, discontinuous covariates |
| Synthetic control | Donor pool, pre-intervention fit, predictor balance, in-space/in-time placebo tests | Contaminated donors, poor pre-fit, donor leakage |
| Observational adjustment | Estimand, DAG/adjustment assumptions, overlap, negative controls, sensitivity to unmeasured confounding | Hidden confounder, collider adjustment, positivity failure, common-cause source laundering |

MACE v0.12 intentionally accepts observations from all methods but only promotes evidence with a receipt-verified randomized design. It does not claim that RCTs are the only valid causal method. Quasi-experimental designs need dedicated assumption and diagnostic contracts; they must not be admitted by loosening a generic boolean.

## 3. Episode protocol

1. **Session A — preregistration:** record prediction, estimand, treatment/control definitions, eligible population, design and analysis-plan digest before acting.
2. **Evidence acquisition:** the environment independently records assignment, observations and outcomes. Raw event, authority, lineage, calibration, causal-design and outcome receipts are generated independently of the agent.
3. **Consolidation:** preserve method, design, estimand, effect estimate, uncertainty, assumptions, validity domain and complete receipt lineage.
4. **Session B — transfer:** change environment, objective, population, intervention or measurement process. The agent must reuse, adapt, probe or abstain.
5. **Session C — correction:** introduce delayed outcome, source correction, revoked authority or failed design assumption; stale claims must be withdrawn without corrupting historical as-known queries.
6. **Held-out task:** use new source roots, task families, mechanisms and environment seeds.

The evaluator owns hidden structural equations, randomized assignment schedule, potential outcomes and mechanism changes. Candidate policy receives only the observations and receipts permitted by the protocol.

## 4. Mandatory adversarial families

**Identification failures:** high-reward observational association with hidden confounding; post-hoc estimand or analysis plan; assignment propensity outside the frozen overlap range; covariate imbalance; selective attrition; treatment/control interference; assignment without treatment received; outcome switching; invalid quasi-experimental diagnostics.

**Memory/provenance failures:** two publishers share one upstream root; duplicated events under multiple IDs; source retraction/supersession; agent attempts to narrow its own error bound; post-hoc causal receipt; effect applied to the wrong population/objective/estimand; common-mode baseline shift mistaken for treatment-effect shift; late correction leaking into an earlier (valid_at, known_at) replay.

## 5. Metrics and targets

| Metric | Initial qualification target |
|---|---:|
| False causal promotion | ≤0.1% on ≥10,000 held-out adversarial negatives, with one-sided confidence bound |
| Causal-claim provenance completeness | 100% of promoted claims bind method, estimand, design, source lineage and outcome |
| Method classification | Per-method precision/recall and confusion; no aggregate that hides unsupported quasi-experimental acceptance |
| Estimand error | ATE/ATT/CATE error against the evaluator's exact estimand |
| Statistical interval coverage | For nominal 95% intervals, 90–98% empirical held-out coverage plus interval width and subgroup coverage |
| Transport false acceptance | ≤0.1% on mechanism/assumption-break controls |
| Stale reuse after known correction | Zero on deterministic as-known controls |
| Decision quality | Paired utility lift, regret, selective risk/coverage and worst-family outcomes |
| Lifecycle cost | Tokens, storage, tool calls, CPU/GPU/energy and p50/p95/p99 latency |

A deterministic absolute measurement-error bound is not a statistical confidence interval and must not be scored as one.

## 6. Baselines

Compare under the same backbone, context, tool, storage and compute budgets: no-memory; frequency/recency; vector retrieval; scope-aware retrieval; method-aware causal memory; invariant transport; active probing; a trusted method-specific causal estimator supplied the same observations; and blinded human adjudication on a stratified subset.

## 7. Statistical protocol

- Freeze candidate, evaluator, data, model and analysis-plan digests before held-out execution.
- Split by environment cluster, task family, upstream source root, mechanism and seed—not random rows.
- Use at least five development seeds and a sealed final set.
- Bootstrap at environment/task cluster level; publish paired effect sizes, intervals, worst-family outcomes and failures.
- Keep model tuning, threshold selection and calibration outside held-out outcomes.
- Require evaluator-owned label custody; suppress comparative claims if oracle provenance is insufficient (#11816).

## 8. External benchmark alignment

CMIT complements, not replaces, MemoryArena, AMA-Bench, TimlMemBench, Agent Memory Challenge and LongMemEval. Their scores are not interchangeable with MACE/CMIT metrics. AML Add/Search submissions must follow the official dataset, API, model and evaluator contract.

## 9. Implementation roadmap and limitations

1. Preserve typed causal identification metadata from ingress into canonical consolidation receipts — implemented in the MACE branch, with field-by-field receipt digest mutation coverage.
2. Add observational and unsupported quasi-experimental negative controls — authored in fixed and generated evaluators.
3. Implement method-specific verifiers for DiD, IV, RD and synthetic controls rather than a generic `causal=true`.
4. Bind assignment schedule, eligible population, estimand, balance diagnostics and analysis plan to independently verified receipts.
5. Add calibrated statistical intervals and method-specific sensitivity analysis.
6. Connect corrections to bitemporal source-event semantics (#9075), and evaluator custody to #11816.

Current verifier implementations are deterministic fixtures. The prototype propensity range [0.05, 0.95] and SMD ≤0.10 are not universal methodological standards and are not independently recomputed by fixture verifiers. No Rust test or CI pass, official benchmark submission, statistical causal identification or SOTA result is claimed here.
