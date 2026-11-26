# SDCB acquisition comparison bake-off v1

**Status:** preregistration-style evaluation protocol; no comparison run has been executed.
**Existing hosts:** `crates/core/symthaea-core/examples/isru_material_experiment_selection.rs` and `docs/research/DECISION_RELEVANT_PREDICTABILITY_CHAOS_BENCHMARK_V0_2026-11-17.md`.
**Reference stack:** DSSI EVSI/non-myopic, RPSD robust/Pareto, R-IDeA, EGBT.

## 1. Purpose

Compare existing Symthaea experiment selectors with exact scientific-decision references before changing production planners. Avoid a new leaderboard that duplicates the existing code. The target is a controlled adapter bake-off where each method sees the same candidate pool, hypotheses, allowed observations, utility schedule, cost/budget and stop rule.

The main question is not which selector gets the largest internal score. It is which method reaches better held-out decisions and generalizes under model misspecification without violating evidence or safety constraints.

## 2. Existing methods and exact reference comparators

| ID | Method | Source / implementation | Required interpretation |
|---|---|---|---|
| B0 | Fixed canonical | Existing ISRU example | Fixed sequence, no adaptive evidence use |
| B1 | OFAT | Existing ISRU example | One-factor-at-a-time sequence |
| B2 | Space-filling | Existing ISRU example | Geometric coverage baseline |
| B3 | Seeded random | Existing ISRU example | Deterministic seeded coverage baseline |
| B4 | Prediction disagreement | `symthaea-core` `select_most_informative_experiment` | Population variance of candidate point predictions; not Bayesian EIG |
| B5 | Spark EIG | `spark-engine::optimal_experiment` | Signature-level discrete outcome model; preserve documented uniform treatment of unpredicted hypotheses and MAP-simulated greedy update |
| B6 | R-I | `scripts/research/sdcb_ridea.rs` | EIG × MMD representativeness ratio |
| B7 | R-IDeA | `scripts/research/sdcb_ridea.rs` | R-I × proxy de-amplification sigmoid; proxy training is a separate component |
| B8 | One-step EVSI | `scripts/research/sdcb_decision_value.rs` | Expected decision improvement, with explicit task utility and cost |
| B9 | Exact non-myopic policy | `scripts/research/sdcb_nonmyopic_oracle.rs` | Small finite-horizon dynamic-programming oracle; non-deployable upper bound |
| B10 | Finite-scenario robust EVSI | `scripts/research/sdcb_robust_pareto.rs` | Shared action policy across hidden plausible scenarios; maximin, not an average |
| B11 | Pareto/hard-constraint policy | `scripts/research/sdcb_robust_pareto.rs` | Non-dominated objectives plus per-hypothesis hard constraints |
| B12 | qLogNEHVI / qParEGO / novelty-aware evolutionary MOBO | Upstream implementations, where task assumptions fit | Established multi-objective baselines; pin release and optimizer budget |\n| B13 | Multi-fidelity BO / qMFMES | Upstream implementation with fidelity-dependent likelihood/cost model | Compare low/medium/high fidelity selection and calibration under equal resource envelopes |
| B14 | Robust multi-fidelity decision reference | `scripts/research/sdcb_multifidelity.rs` | Finite maximin decision gain across fidelity scenarios plus componentwise resource caps/Pareto frontier |

Do not compare internal scores directly across methods. A variance score, EIG in bits, EIG in nats, EVSI in loss units, R-IDeA score and worst-case utility are different quantities.

## 3. Evaluation stages

### Stage A — Known-answer analytic controls

Use the frozen EIG-vs-EVSI fixture, two-step non-myopic fixture, robust ambiguity-set fixture, Pareto/hard-safety fixture and R-IDeA MMD fixture. Every formula should match its independent expected file before any model comparison.

Acceptance gates:

- EIG and EVSI reproduce their hand-calculated values.
- The EVSI selector rejects a high-information probe with zero decision value.
- The exact non-myopic oracle matches the two-step adaptive policy value.
- Robust policies share one outcome-to-action map across hidden scenarios.
- Pareto controls never trade away a hard safety floor.
- R-IDeA reproduces MMD, R-I and DeA formula values.
- Every invalid EGBT receipt causes zero belief/decision-state mutation.

### Stage B — Existing synthetic ISRU experiment selector

Reuse `isru_material_experiment_selection.rs` unchanged as the first synthetic integration host. Its authors explicitly mark its response curves as synthetic and dimensionless, not physical material models. Keep its four hidden worlds, eligible 20-point candidate catalogue, equal declared cost/noise, 6-experiment limit, and existing fixed/OFAT/space-filling/seeded-random/disagreement policies.

Extend it through a separate adapter rather than altering the existing selector's score. Add Spark EIG, R-I, R-IDeA, one-step EVSI, non-myopic oracle, robust EVSI and Pareto policy outputs. The hidden world is queried only after experiment commitment. For every new method, preserve the same candidate availability and acquisition cost.

Important: the source uses a fixed synthetic seed and deterministic observation perturbation. This is a software fixture, not a statistical benchmark. Add a versioned seed grid and held-out response-curve families before making comparative claims.

### Stage C — Paired-world decision-predictability benchmark

Use the existing paired-world constitution. Include worlds with identical passive observations but different correct actions, and both:

- high information gain with zero decision value;
- lower information gain with positive decision value.

The evaluator computes information-set Bayes risk and policy regret. Candidate-visible observations must not expose hidden mechanism or action-loss truth. Use the exact EVSI and finite-horizon oracles only on tractable finite fixtures; larger worlds evaluate approximate policy regret.

### Stage D — Model misspecification and covariate shift

Vary:

- omitted variables and missing interaction terms;
- wrong likelihood/noise family, heteroskedasticity and autocorrelation;
- parameter drift and regime changes;
- target/test design distribution shift;
- proxy-model error and proxy/current-model disagreement;
- unpredicted or out-of-domain hypotheses.

Compare Random, EIG-only, R-I, R-IDeA, EVSI, robust maximin and combinations only as separately preregistered arms. Do not create an unvalidated weighted scalar that multiplies EIG, EVSI, R-IDeA and robust gain.

Use the same target distribution for MMD in each paired task. Freeze lambda, tau, kappa and kernel bandwidth on validation; never tune them on hidden test families. Report the zero-baseline-MMD case as undefined for the R-I ratio; do not silently replace it with raw EIG.

### Stage E — Multi-objective and hard safety

Objectives may include scientific utility, sample efficiency, cost, wall time, compute, measured energy, environmental impact and safety margin. Keep them separate. Compare qLogNEHVI, qParEGO and novelty-aware evolutionary-Bayesian hybrids where the design space fits their assumptions. Freeze objective scaling and the hypervolume reference point before evaluation.

Hard safety constraints are non-compensatory and checked for every plausible state/scenario required by the task. Do not use expected averages to conceal a violation. Include a zero-cost no-experiment lottery baseline so an uninformative random outcome cannot be credited as scientific information.

## 4. Shared experimental contract

For each task/run, freeze and record:

- task generator, source revision, task-family split and hidden-seed commitment;
- prior/hypothesis-set digest and candidate experiment catalogue digest;
- permitted observations, modalities, time horizon and intervention authority;
- action set and owner-approved utility/loss schedule;
- experiment costs and resource-budget units;
- model/checkpoint, proxy model, prompt/tool runtime and random seeds;
- scoring, calibration, statistical correction and stop rules;
- hardware, wall-clock and resource measurement boundaries.

All methods receive identical lawful information and candidate actions. If one algorithm relies on a stronger likelihood/model assumption than another, disclose the assumption and report a separate information-access tier rather than force an unfair ranking.

## 5. Metrics — no composite winner score

### Scientific and decision quality

- mechanism recovery conditioned on identifiability;
- false mechanism claim rate and calibrated abstention;
- observable prediction error and proper predictive scores;
- final Bayes decision value, task-loss regret and risk-adjusted regret;
- counterfactual/intervention fidelity on held-out regimes;
- discovery quality and unmeasured-candidate learning as separate outcomes.

### Acquisition and sample efficiency

- EIG, R-I, DeA and R-IDeA reported as their own units;
- one-step EVSI and exact-oracle non-myopic policy regret;
- robust scenario-value vector and worst-case regret;
- simple/cumulative regret and target utility reached per measurement;
- experiments and money/time/energy spent to first useful decision;
- R-IDeA MMD before/after, target-distribution coverage and sensitivity to kernel/threshold.

### Robustness, multi-objective and safety

- predictive generalization error on declared target distribution;
- performance under each misspecification/shift family and worst regime;
- Pareto hypervolume (frozen reference point), IGD, coverage and diversity;
- hard-constraint violation rate (must be zero on frozen negative controls);
- sensitivity to adding/removing ambiguity scenarios and perturbing utility schedules.

### Integrity and deployment

- stale/duplicate/misbound/unauthorized observation detection;
- independent replay of accepted belief and decision transitions;
- failed-run cost, recovery time, latency, peak memory and measured energy;
- invalid/unsafe experiment attempts and budget/authority violations.

No one metric or weighted aggregate may hide a regression on another safety-critical axis.

## 6. Statistical protocol

- Use paired task/seed comparisons and at least 20 seeds per family where the generator supports it; report confidence intervals.
- Cluster bootstrap by task/system family, not by correlated observations/windows.
- Hold out complete mechanism and parameter families, not only randomly selected episode rows.
- Freeze all acquisition hyperparameters on validation only.
- Report every attempt, failure, aborted run and consumed cost.
- Use Holm or another preregistered family-wise correction when testing many tasks/claims.
- Publish per-family tables and Pareto frontiers; do not rank systems by partially completed benchmarks.
- External reproduction by a second implementation or team is required before a SOTA claim.

## 7. Competitive targets (research targets, not results)

1. Exact agreement on all known-answer analytic controls.
2. At least 10% lower decision regret than EIG-only on one preregistered decision task, with paired uncertainty intervals excluding zero and no material predictive or safety regression.
3. At least 10% lower target-distribution generalization error than EIG-only under a preregistered misspecification family, while remaining non-inferior in the well-specified regime.
4. Less than 5% normalized regret against the exact adaptive policy on small unseen finite tasks.
5. Zero hard safety violations and zero unauthorized belief transitions on frozen controls, with valid-receipt false rejection reported.
6. Better hypervolume/IGD/coverage than qLogNEHVI or qParEGO on at least one preregistered many-objective task under matched evaluation budget.
7. At least five seeds and one held-out generator family reproduced by an independent runner.

These are deliberately modest first milestones. They become claims only after actual paired experiments and reproducibility.

## 8. Integration sequence

1. Reconcile the existing decision-predictability benchmark README/Cargo-manifest discrepancy recorded in issue #13739.
2. Add read-only adapters for prediction disagreement and Spark EIG; do not change their existing output semantics.
3. Add R-I/R-IDeA as separate arms with a validated proxy training pipeline.
4. Connect DSSI and RPSD exact reference evaluators to the paired-world harness.
5. Add the seed grid and held-out synthetic ISRU response families.
6. Run candidate methods in identical information/resource envelopes.
7. Only then consider HDC/CfC amortizing a policy; compare its regret to the exact oracles on held-out tasks.

## 9. Claim boundary

This document is an evaluation protocol. None of the method comparisons above has been executed by Symthaea. Existing selectors and R-IDeA reference code are not SOTA results. No actual physical ISRU, cell-fabrication, nuclear, or chaotic-systems experiment is claimed.