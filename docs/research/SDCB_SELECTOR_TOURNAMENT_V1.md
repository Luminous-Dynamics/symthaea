# SDCB Selector Tournament v1 — matched comparisons without planner substitution

**Status:** evaluation protocol and source-reuse map; not an executed tournament  
**Parent:** [SDCB v2 issue #13691](https://github.com/Luminous-Dynamics/symthaea/issues/13691)  
**Implementation base:** [Active Science v2 reference PR #13780](https://github.com/Luminous-Dynamics/symthaea/pull/13780)  
**Selector audit:** [issue #13739](https://github.com/Luminous-Dynamics/symthaea/issues/13739)  
**Production integration owner:** [issue #11171](https://github.com/Luminous-Dynamics/symthaea/issues/11171)

## 1. Why this protocol exists

Symthaea already contains several experiment-adjacent components. They must not be collapsed into a single class called “the research planner”:

- a hypothesis-prediction disagreement selector;
- a signature-level Bayesian EIG scorer;
- an HDC/CfC experiment-state predictor and FEP-flavoured triage action;
- a test-only domain heuristic for nuclear measurement ordering;
- a synthetic material-response experiment bake-off;
- exact finite Bayesian and prior-free evaluator oracles.

Those components differ in inputs, assumptions, outputs, and trust boundaries. The tournament should measure their actual behavior before introducing another production selector. It must not rewrite or upgrade a baseline silently in the process of comparing it.

The guiding split is:

**candidate policy → authorized probe proposal → evaluator executes hidden world → independently validated observation → updated candidate state**

The candidate never receives hidden truth, hidden likelihoods, evaluator seeds, task labels, or the oracle's posterior/value. The evaluator never executes the candidate's own proposed answer as ground truth.

## 2. Source audit and eligible roles

The following source files were fetched from main on 2026-11-26. The recorded SHAs make the audit reproducible.

| Component | Source / commit | What it actually computes | Tournament role |
|---|---|---|---|
| HDC hypothesis disagreement | `crates/core/symthaea-core/src/hdc/conjecture_engine/experiment_selection.rs` — `06a033939ba922f1933680d8bd15ab4d9222d296` | Population variance of scalar predictions across surviving hypotheses for a candidate experiment; hypotheses returning None are excluded; picks the maximum, stable under a fixed candidate order | Primary nominal disagreement baseline |
| Spark signature-level EIG | `crates/domains/spark-engine/src/optimal_experiment.rs` — `8d6f83d8c35622211753197014a0b0057664f0da` | Mutual information over discretized outcome classes. Tested hypotheses are deterministic on their class; unpredicted hypotheses are assigned a uniform outcome over classes. Greedy sequence simulates the current MAP hypothesis' outcome between steps | Domain-specific EIG baseline, under its documented likelihood approximation |
| Cell Foundry experiment planner | `crates/domains/symthaea-cell-foundry/src/experiment_planner.rs` — `4c48ad0074fe3ce28a5c8b64e02c2b0801318ee1` | Encodes experiment uncertainty/cost/time/info-gain into HDC; evolves an HDC-LTC predictor; an FEP-flavoured agent maps similarity/free-energy thresholds to continue/replicate/reallocate/pause/abort | Campaign triage / scheduling context baseline; **not** a direct candidate experiment ranking algorithm |
| Nuclear active-learning heuristic | `crates/domains/symthaea-nuclear/src/active_learning.rs` — `0cb8f819e166492ad363d43085ed8355e0cdcbfa` | Uncertainty, unmeasured-neighbour count, shell proximity and batch diversity; code is currently within #[cfg(test)] | Domain-specific heuristic ablation only; do not present as a general production capability |
| Synthetic ISRU experiment bake-off | `crates/core/symthaea-core/examples/isru_material_experiment_selection.rs` — `ca1c11ea350afeedace278d04a8ce095b347c41c` | Matched fixed-canonical, OFAT, space-filling, seeded-random and HDC prediction-disagreement policies; hidden synthetic world is queried only after selection; equal cost/noise in the current setup; includes an out-of-domain world | Best existing benchmark harness to extend with additional adapters. It is a synthetic, dimensionless response-surface exercise, not a physical material claim |
| Active Science v2 maximin | `scripts/research/sdcb_active_science_v2.rs` — pin the exact PR head in each run manifest | Worst pairwise trace separation after bounded-noise margin, with componentwise resource admission; returns a single proposal or abstention | New prior-free, single-probe discrimination baseline; not EIG/EVSI |
| Active Science v2 Pareto | Same source | Non-dominated separation-margin versus query/experiment/measurement resource vectors | Preserves incomparable offers; no arbitrary scalarization |
| DSSI one-step and exact adaptive oracle | `scripts/research/sdcb_decision_value.rs`, `scripts/research/sdcb_nonmyopic_oracle.rs`, PR #13717 | Expected decision value under supplied prior, likelihood and utility model; finite-horizon adaptive Bellman value on small exact cases | Evaluator-side Bayesian oracle only; never candidate-visible |
| RPSD finite-ambiguity oracle | `scripts/research/sdcb_robust_pareto.rs`, PR #13738 | Worst-case value across a finite unweighted set of coherent utility/likelihood scenarios; pure common policies across hidden scenarios; separate Pareto/hard constraints | Evaluator-side ambiguity-set comparator; not a posterior-free model of the world |
| Prior-free sequential oracle | `research/decision-predictability-benchmark/src/sequential.rs`, PR #13571 | Exact finite-horizon policy tree over set-valued supports, terminal minimax loss and componentwise resources; separate exhaustive enumerator | Evaluator-side ceiling when only supports and losses are defensible; unqualified/draft until exact-head checks pass |

### Important distinctions

1. **Prediction variance is not information gain.** The HDC selector measures scalar disagreement over the surviving hypothesis list; it has no calibrated noise/outcome model, prior, cost or downstream utility.
2. **The Spark EIG approximation is not a full measurement model.** Its uniform likelihood for unpredicted hypotheses and deterministic classes for predicted hypotheses are assumptions. Report them as implemented; do not replace them with a calibrated likelihood during baseline runs.
3. **Cell Foundry's FEP thresholds are triage, not probe selection.** Score campaign interventions separately; do not place it in the same top-1 experiment-ranking column.
4. **Maximin separation is not EIG or EVSI.** It uses a bounded-noise deterministic trace envelope and no prior. Report its pairwise separation margin and resource Pareto frontier, not “information bits.”
5. **The exact oracles are evaluator-only.** Their job is to produce ceilings/regret on small cases, not to leak their outputs to candidate agents.
6. **The nuclear heuristic is test-only.** Its presence in source does not make it a deployed, general research policy.

## 3. Core paired episode design

Each episode freezes:
- task-family and parameter-regime identifiers;
- candidate-visible initial state, passive input schedule and passive observations;
- candidate-visible hypotheses and each hypothesis' prediction/model digest;
- probe catalogue, sensor identities and measurement precision/noise contract;
- all resource units and the decision deadline;
- evaluator-only true mechanism, hidden noise seed, observation generator and scoring oracle;
- candidate binary/model/runtime digest and permitted tool/authority profile.

Episode identifiers and payload commitments are deterministic. Training, validation and test splits are separated by **system family and parameter regime**, not merely by random episode rows.

Every selected probe is removed from the remaining offer set unless the episode explicitly permits replication. Every attempt, abstention, invalid proposal, budget rejection, malformed receipt and failed execution stays in the denominator.

### Episode families

| Family | Construction | What it tests |
|---|---|---|
| Exact observational equivalence | Distinct mechanisms with exactly identical passive outputs under the frozen observation schedule; an allowed intervention separates them | False-unique claims and intervention value |
| Near equivalence | Parameter perturbations whose passive prediction distance is below/near the declared noise envelope | Calibration and selective risk near identifiability boundaries |
| Sensor-limited | Some state dimensions unobservable under one sensor but observable under a permitted sensor set | Sensor choice, observability limits |
| Excitation-limited | System is observable in theory but the actual passive input leaves discriminating modes unexcited | Distinguishes observability from experiment informativeness |
| Confounded input | Inputs correlate with hidden disturbances or regime labels | Causal misattribution and intervention robustness |
| Irrelevant sensor / no-op | Candidate adds a sensor or performs a probe whose outcomes are independent of the hypotheses | Wasted-information spend and no-information controls |
| Noise-family shift | Evaluation noise differs from candidate's declared likelihood family or bounded-noise envelope | Model misspecification and robust decision quality |
| Open-world mechanism | True generator is not among the candidate-visible hypotheses | OTHER_UNRESOLVED, no-fit detection and model-revision behavior |
| Regime shift | Hidden dynamics change after a verified prefix | Recovery time, update discipline, stale-belief detection |
| Evidence attack | Receipt is stale, duplicated, corrupted, misbound, unauthorized or inconsistent with sensor calibration | Evidence-integrity gate and no-mutation-on-rejection |

## 4. Policies to compare

All policies receive identical candidate-visible state, eligible offers, resource vectors and time limits. Where a policy cannot consume one of those fields, record that limitation rather than silently enriching it.

| ID | Policy | Purpose |
|---|---|---|
| P0 | Passive / stop | Establish the value of doing nothing |
| P1 | Fixed canonical | Deterministic reference schedule |
| P2 | One-factor-at-a-time | Classical local exploration |
| P3 | Space-filling | Coverage-first baseline |
| P4 | Seeded random | Randomized control with common seed |
| P5 | HDC prediction disagreement (existing source) | Nominal model-disagreement baseline |
| P6 | Spark signature-level EIG (existing source) | Approximate Bayesian information baseline under its documented assumptions |
| P7 | Active Science v2 one-probe maximin separation | Prior-free bounded-noise discrimination baseline |
| P8 | Active Science v2 resource Pareto frontier | Multi-resource non-dominated proposal set |
| P9 | One-step DSSI EVSI | Myopic decision-value baseline; only where priors/likelihoods/utilities are defensible |
| P10 | Exact DSSI adaptive policy oracle | Finite Bayesian ceiling on small tasks |
| P11 | RPSD robust policy | Finite ambiguity-set robust comparator |
| P12 | Prior-free sequential minimax oracle | Exact support-based ceiling on small tasks |
| P13 | Candidate Symthaea researcher | Production model under test, only after exact callable and runtime/model digests are pinned |

Cell Foundry and Nuclear policies should have separate task-appropriate campaign-triage / domain-heuristic tracks. They are not forced into the same probe-ranking task when their input/output contracts do not match.

## 5. Metrics — separate estimands

### Identification and prediction
- mechanism accuracy **conditioned on the task being identifiable**;
- false-unique mechanism-claim rate on exactly equivalent worlds;
- coverage and selective risk when the candidate abstains;
- set-valued hypothesis coverage / size;
- no-candidate-fits detection rate on open-world episodes;
- held-out intervention prediction error and calibration.

### Experiment selection
- maximin pairwise separation margin for bounded deterministic noise;
- EIG in bits for calibrated probabilistic observation models;
- EVSI and net EVSI only under frozen owner-approved utility schedules;
- worst-case robust gain over each declared ambiguity scenario;
- exact-policy regret on small finite worlds;
- decision-reversal hit rate and wasted-information spend;
- Pareto frontier coverage/diversity across resource dimensions.

### Reliability and evidence
- invalid-receipt rejection sensitivity and false-rejection rate;
- false acceptance rate for stale/misbound/corrupted/unauthorized evidence (target: zero on frozen negative controls);
- state mutation after rejected evidence (target: zero);
- reproducible accepted/rejected transition receipts;
- recovery cost after regime shift or corrupted observation.

### Full operational cost
Keep query count, experiment count, measurement count, wall-clock time, compute, memory, money, energy and reviewer effort as separate columns. Convert to one utility only if the task owner explicitly approves a unit conversion and sensitivity analysis.

## 6. Statistical protocol

- Use common episode seeds across all methods (paired comparison).
- Publish all per-family outcomes, including unresolved, rejected and failed episodes.
- Use task-family-clustered bootstrap confidence intervals; do not bootstrap individual episodes as though task families were independent.
- Report worst-family and lower-tail performance, not only a global mean.
- Run at least five seeds for pilot comparison and at least ten before a broad superiority statement.
- Freeze all candidate and evaluator digests before the hidden test set is opened.
- Use an independent scorer; candidate cannot select utility, candidate pool, hidden seed, stopping rule or acceptance threshold.
- Include the evaluator's exact small-instance ceiling but never leak its hidden state or posterior to candidates.
- Compare under two distinct regimes: equal experiment budget and equal total deployment cost. Do not call one “fair” and silently omit the other.

## 7. Promotion targets

These are proposed project targets, not achieved results or industry standards.

| Gate | Initial target | Promotion target |
|---|---:|---:|
| Analytic controls | 100% frozen known-answer fixtures | 100% adversarial/regression corpus, exact replay |
| Exact equivalence | 0 false unique claims | 0 false unique claims across all exact-equivalence families |
| Bounded-noise consistency | 100% true-model retention under declared generator | Calibrated coverage on held-out noise levels/families |
| Open-world | Detect every no-fit control | Reliable OTHER_UNRESOLVED with safe model expansion |
| Resource safety | 0 componentwise budget overruns | 0 overruns, including retries and recovery |
| Exact-oracle regret | Exact equality on tiny fixtures | ≤5% normalized regret on held-out tractable cases |
| Evidence integrity | 0 invalid receipt acceptances | 0 invalid state transitions, 0 mutation on rejection |
| External tasks | Reproduce native baseline | Beat best matched baseline with paired 95% CI and no integrity regression |

The external task targets live in [SDCB Active Science v2](SDCB_ACTIVE_SCIENCE_BENCHMARK_V2.md). Do not pool external scores or use an analytic fixture as evidence of SOTA.

## 8. Integration sequence

1. Restore/confirm each baseline's source and API at the exact commit to be tested.
2. Add read-only adapters from the candidate episode into each existing selector's native input type. Never change a baseline algorithm within a tournament run.
3. Store raw proposal IDs, original scores, source digests and resource declarations.
4. Let the evaluator execute selected probes, produce authenticated observation receipts and score hidden outcomes.
5. Feed only authorized public receipts back to the candidate; recompute state from the verified prefix.
6. On small cases, compare the candidate's first action and terminal policy value to the appropriate exact oracle.
7. Only after adapters pass parity and leakage controls should HDC/CfC amortize a policy; measure regret and calibration against exact references on held-out families.

### Explicit blockers before production integration

- [#11171](https://github.com/Luminous-Dynamics/symthaea/issues/11171) still needs an exact production researcher callable and immutable model/runtime/tool digests.
- The StateGuard/ResearchState production-origin boundary (#13749) is a draft and not yet qualified.
- DSSI (#13717), RPSD (#13738) and prior-free sequential oracle (#13571) are reference PRs; their uncompiled states cannot serve as validated evaluator binaries.
- Active Science v2 (#13780) is an analytic reference, not a live lab experiment runner.
- A Rust type split in one process is not an evaluator security boundary; confirmatory runs require process/permission isolation.

## 9. Claim boundary

This document is an evaluation protocol and source-reuse audit. No tournament has been run, no Symthaea agent has been measured against these targets, and no SOTA result is claimed. The expected scientific advance is not “one more planner”; it is a reproducible competition in which prediction disagreement, probabilistic information, decision value, robust gain, minimax loss and evidence integrity can be independently examined.
