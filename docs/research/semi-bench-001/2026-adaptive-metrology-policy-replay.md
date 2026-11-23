# Adaptive metrology and evidence-qualified semiconductor intelligence

**Review date:** 2026-11-23  
**Tracking:** [SEMI-BENCH-001 #13058](https://github.com/Luminous-Dynamics/symthaea/issues/13058)  
**New policy replay issue / draft PR:** [#13328](https://github.com/Luminous-Dynamics/symthaea/pull/13328)  
**Claim class:** research design and synthetic prototype; no fab deployment or physical-yield result claimed.

## Executive direction

The next semiconductor intelligence increment should not be another wafer-failure classifier. It should qualify a manufacturing decision from the evidence available at the time that decision was made.

The system must distinguish four different things:

1. **Process state:** what the equipment and wafer process were doing.
2. **Observation process:** which sensors and measurements were available, missing, delayed, corrupted, recalibrated or never installed.
3. **Acquisition and selection policy:** why this wafer was measured, skipped, escalated or sampled at random.
4. **Decision and evidence state:** what the system predicted, how uncertain it was, what it asked to measure, what matured later, and what was independently verified.

This distinction is essential because adaptive inspection changes the data distribution. A system that measures only the wafers it already considers risky can become confident about a selected subset while becoming blind to the population it is supposed to protect.

The design goal is therefore:

> Minimize physical metrology cost subject to an independently qualified upper bound on escaped-defect risk, valid population inference, calibrated uncertainty and explicit refusal under unsupported transfer.

A reduction in physical measurements is not a success if it is achieved by losing visibility into low-risk or previously unseen strata.

## Why this is no longer just a virtual-metrology problem

The latest literature means that combining a VM model, uncertainty and a dynamic inspection policy is not by itself a defensible novelty claim.

### Strong comparators

**NIST / IEEE TSM — online Gaussian-process dynamic sampling (2025).** Han et al. study Online Gaussian Process virtual metrology with uncertainty-driven dynamic sampling on a public CMP dataset. It is a useful reproducibility anchor for uncertainty-based measurement reduction and model adaptation.

- DOI: [10.1109/TSM.2025.3531920](https://doi.org/10.1109/TSM.2025.3531920)
- NIST publication record: [A Comparative Study of Semiconductor Virtual Metrology Methods and Novel Algorithmic Framework for Dynamic Sampling](https://www.nist.gov/publications/comparative-study-semiconductor-virtual-metrology-methods-and-novel-algorithmic)

**IEEE TSM — uncertainty-aware VM and physical-metrology policy (2026).** Bertelson et al. describe a unified linear-Gaussian framework for virtual-metrology uncertainty, physical-metrology selection and model adaptation, evaluated on 114,774 wafers from an industrial 300 mm plasma-etch fab.

- DOI: [10.1109/TSM.2026.3709162](https://doi.org/10.1109/TSM.2026.3709162)

This is a serious industrial comparator. Its proprietary industrial population is not available to this project, so its reported result must not be described as reproduced or directly beaten on public SECOM data.

**IEEE OJIES — GFA-VM+ cross-machine / cross-process / cross-fab few-shot VM (2026).** This work is an important transfer comparator because it reports target-label-budget performance across different machines, processes and fabs, rather than relying only on random within-domain splits.

- DOI: [10.1109/OJIES.2026.3748865](https://doi.org/10.1109/OJIES.2026.3748865)

The paper reports strong low-label results, including 10-label cases, while also showing that severe sensor dropout degrades transfer. The lesson is not that a representation learner alone solves transfer: feature identity, sensor alignment, target evidence and missingness are part of the problem.

**IEEE T-ASE — feasibility-aware inverse VM / Top-K correction recommendation (2026).** This is a useful but distinct comparator. It evaluates suggested process-correction options, not the decision to acquire another physical measurement.

- DOI: [10.1109/TASE.2026.3715101](https://doi.org/10.1109/TASE.2026.3715101)

Do not merge correction recommendation, inspection allocation and process actuation into one leaderboard. They have different action spaces, loss functions and authority requirements.

## Where Luminous should differentiate

The opportunity is an **evidence and decision assurance layer** around VM and manufacturing analytics:

- **Semantic correctness:** labels are derived from validated numeric measurements, units, thresholds and authority-bound measurand definitions. A caller-supplied positive Boolean is not sufficient evidence.
- **As-of-time validity:** a prediction only consumes observations and labels available at the decision cutoff. Late labels cannot flow backwards into earlier model state.
- **Observation-process awareness:** missingness, structural absence, instrument outage, calibration drift, schema changes and sampling policy are explicit state, not a generic null or zero.
- **Selection accountability:** every candidate has a logged first-order inclusion probability; positivity violations and unsupported strata block population-wide claims.
- **Population-level evaluation:** compare raw selected-sample metrics to Horvitz–Thompson and Hájek estimates, design variance and interval coverage against known finite-population truth.
- **Transfer qualification:** report performance by held-out tool, chamber, recipe, sensor schema and fab. The system must detect unsupported transfer and abstain, not silently convert uncertainty into a high-confidence accept.
- **Intervention qualification:** a recommendation carries expected benefit, feasibility, uncertainty, evidence dependencies, rollback/reconciliation conditions and explicit non-authority. A replay result is not permission to edit a recipe or control loop.

This is a claim about the quality and auditability of decisions—not a claim that the model has discovered a new semiconductor process law.

## New executable benchmark: SEMI policy replay

Issue / draft PR: [#13328](https://github.com/Luminous-Dynamics/symthaea/pull/13328).

The first increment adds a deterministic, synthetic finite-population generator and a policy-facing observation view separated from private evaluation truth. Current generator dimensions include:

- stable, gradual-drift, abrupt-shift, calibration-drift, correlated-outage and prevalence-shift regimes;
- MCAR, MAR and MNAR sensor dropout mechanisms;
- observed, communication-loss and structurally-not-installed sensor states;
- wafer, lot, tool, chamber and process-step identity;
- delayed label maturation retained outside the policy view;
- synthetic-unit measurements that cannot be mistaken for physical metrology.

The evaluator is intended to accept a completed propensity-bearing audit selection report. It checks that the selection digest matches its payload; row identities, strata and label semantics match; exploit/audit/not-selected methods are coherent; inclusion probabilities equal declared stratum quota divided by residual population; report counts and costs reconcile; and budget limits are respected.

For stratified simple random sampling without replacement, the evaluator also computes finite-population design variance for prevalence and Brier loss. A variance estimate is marked unqualified when a residual stratum has fewer than two sampled units and is not a census. It does not silently manufacture a confidence interval where the design cannot support one.

**Important current limitations:** this is a prototype generator and evaluator, not yet a complete policy tournament. It does not reproduce a physical fab, instrument response, real metrology sensitivity/specificity, causal process-control effect, or the 2026 industrial paper. Multi-seed interval-coverage qualification, the full matched-policy runner, a public CMP reproduction, cross-fab evidence and prospective fab validation remain future gates.

## Required policy tournament

Every policy receives the same scenario seed, observation packet, initial calibration data, candidate universe and metrology budget. The evaluator—not the policy—retains hidden truth.

Minimum comparison set:

1. fixed-rate / periodic sampling;
2. simple random sampling;
3. deterministic top-risk sampling;
4. uncertainty-only sampling;
5. expected-value-of-sample-information (EVSI) sampling;
6. OGP-style uncertainty-driven dynamic sampling;
7. uncertainty-aware dynamic VM with a linear-Gaussian state-space comparator;
8. deterministic exploit plus stratified randomized audit;
9. conformal abstention plus exploit and randomized audit;
10. oracle-information upper bound, evaluation-only and clearly marked unattainable.

Report two views: equal measurement count/cost and equal decision loss/escape risk. Do not compare a cheap model at one budget to an expensive model at another and call the score a win.

## Frozen scenario matrix

At minimum, preregister paired seeds for:

| Family | What it tests | Required failure signal |
|---|---|---|
| Stable process / stable sensors | No-shift control | False drift or needless abstention |
| Gradual and abrupt process drift | Model and calibration transfer | Detection delay, risk under drift |
| Calibration reset / sensor drift | Observation vs process change | False process attribution |
| MCAR / MAR / MNAR | Selection and missingness bias | Population-estimate bias |
| Correlated sensor-group outage | Common-cause measurement failure | Unsupported-view detection |
| Structural absence / schema change / unit mismatch | Feature identity and physical semantics | Positional alignment or fabricated zero |
| Delayed / provisional / revised / censored / invalid label | Outcome lifecycle | Future leakage or duplicate label counting |
| Prevalence and subgroup-mix shift | Calibration and threshold transport | Underestimated rare subgroup risk |
| Unseen tool / chamber / recipe / fab | Domain transfer | Negative transfer and false acceptance |

A stable no-shift negative control is required in every benchmark release.

## Metric contract

### Virtual metrology and prediction

- MAE, RMSE and residual bias;
- uncertainty interval coverage and width;
- per-tool / chamber / recipe / fab error;
- Brier score, log loss and calibration slope/intercept;
- AUROC and especially average precision for rare binary defects;
- recall at a predeclared false-alarm rate and false alarms per 1,000 wafers;
- support counts and paired uncertainty intervals.

A probability threshold of 0.5 is not a default operating point. Select operating thresholds on validation data against a declared cost or false-alarm constraint, then apply once to the held-out evaluation set.

### Metrology acquisition and population inference

- absolute number and cost of physical measurements;
- measurements per 1,000 wafers and label-maturation delay;
- raw selected-sample prevalence and loss;
- Horvitz–Thompson and Hájek estimates;
- bias, RMSE, finite-population variance and confidence-interval coverage;
- effective sample size, design effect, stratum coverage and positivity violations;
- decision loss at matched measurement budgets.

An estimator cannot claim population-wide validity if any residual subgroup that matters has zero inclusion probability.

### Decision safety and selective prediction

- escaped-defect rate and confidence interval;
- false-hold rate and operator-review burden;
- excursion detection probability and delay;
- risk-versus-coverage and coverage-versus-measurement curves;
- abstention when calibration, sensor identity or target-domain support is invalid;
- recovery after maintenance or a newly matured target label.

### Cross-tool / cross-fab transfer

- leave-one-tool-out, leave-one-chamber-out, leave-one-recipe-out and leave-one-fab-out evaluation where metadata permits;
- target-label budgets 0, 1, 3, 5, 10, 20 and full target labels;
- label efficiency to a declared MAE / risk target;
- worst-domain degradation and negative-transfer rate;
- time to detect that transfer is unsupported;
- sensor schema, unit and timebase alignment failures.

### Runtime and operations

- p50/p95/p99 prediction and acquisition latency;
- update latency after label maturation;
- memory and compute cost;
- exact replay determinism and artifact/digest verification;
- operator review and reconciliation workload.

## Target hypotheses, not achievements

The first economic target is **20–30% fewer physical measurements than a fixed-rate baseline**, conditional on a predeclared, fab-owner-approved non-inferiority margin for escaped-defect risk. It is not an achieved result.

Additional target hypotheses:

- design-based population estimates remain unbiased within preregistered tolerances over paired known-truth scenarios;
- nominal 95% intervals achieve empirically measured coverage near their declared level across enough independent seeds and scenario strata;
- the audit design maintains positive inclusion probability for every population stratum for which it claims inference;
- an evidence-qualified policy beats top-risk-only and uncertainty-only baselines at matched cost without increasing escape risk beyond the approved margin;
- unsupported transfer is detected before high-confidence false acceptance;
- any decision depending on immature, invalid, semantically mismatched or unavailable evidence abstains or escalates.

A single seed is not a result. A single synthetic generator is not a fab model. A leaderboard winner is not a manufacturing qualification.

## Execution gates

1. **Synthetic known-truth replay:** validate generator, policy isolation, propensity integrity, variance calculation and paired multi-seed reports.
2. **Public CMP reproduction:** match the NIST/IEEE OGP protocol and baselines where data and method details permit; label any non-faithful implementation OGP-style.
3. **SECOM chronological diagnostic:** preserve B0–B6 negative results, rare-class counts, temporal splits and selection controls. SECOM is not cross-fab evidence.
4. **Cross-domain transfer track:** matched target-label budgets and exact sensor/unit alignment, compared with GFA-VM+ where reproducible materials exist.
5. **Read-only fab shadow mode:** only with a data partner, process-owner-approved measurands/losses, privacy/data-rights review, independent metrology validation and a preregistered non-inferiority margin.
6. **Physical impact evaluation:** only after shadow-mode qualification. Any later actuation must have a separate authorization, safety case, rollback and equipment-owner approval.

No benchmark in this workstream grants wafer hold/release, physical measurement scheduling, recipe editing, APC/R2R writes or equipment-control authority.

## Claim ceiling

A passing synthetic replay demonstrates only that the declared policy/evaluator behaves as expected on the declared synthetic scenarios. Public-dataset reproduction demonstrates only the reproduced data/protocol. Neither proves real-fab transfer, yield gain, scrap reduction, lower operating cost, metrology traceability or safe process control.
