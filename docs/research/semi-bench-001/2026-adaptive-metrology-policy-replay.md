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

The Rust prototype now includes a paired multi-seed tournament runner. It applies every policy template to the same scenario seeds, requires one common total measurement budget, derives policy selection seeds deterministically from scenario and template seeds, and emits per-seed receipts plus per-policy means/sample standard deviations and support counts. The current test fixture covers stable, gradual-drift and abrupt-shift regimes with randomized audit, risk-prioritized exploit/audit and uncertainty-prioritized exploit/audit. These are executable synthetic comparisons, not evidence that one policy wins in an industrial fab.

## Standards and architecture alignment: NIST TIDAL

NIST's ongoing 2026 **Super Intelligence for Manufacturing** programme organizes research around TIDAL: Teamed, Integrated, Digital-twin-assured and Layered Intelligence. Its threads include trustworthy-manufacturing fundamentals, human-machine teaming, contextualized manufacturing data, digital-twin VVUQ, and an integration/test-bench programme spanning industrial information architectures and benchmark studies. This is a useful direction-of-travel signal, not a claim that our project has been endorsed or validated by NIST.

Luminous should align through concrete interoperability rather than terminology:

- **Asset / site / enterprise context:** preserve equipment, tool, chamber, wafer, lot, recipe, process-step, site and ownership identifiers with explicit scope and provenance.
- **MOM / OT integration:** build read-only adapters and replay fixtures first; do not add write paths to APC, R2R or equipment controllers.
- **Human-machine teaming:** each recommendation names the decision owner, reviewer, uncertainty, evidence dependencies, expiry and reconciliation condition.
- **Digital-twin VVUQ:** attach model/version, calibration, source, uncertainty, intended-use and applicability evidence; a simulated state is never represented as a physical observation.
- **Standards mapping:** map evidence envelopes and context links to applicable SEMI/ISA-95/OPC UA/I3X concepts only after the relevant standard text and conformance requirements are independently checked.

The potential paradigm shift is not an “agent that controls a fab.” It is a portable evidence contract that lets multiple models and vendors exchange contextualized observations, decisions and verification receipts without granting them authority they have not earned.

A practical standards map already exists in the semiconductor scorecard issue [#13273](https://github.com/Luminous-Dynamics/symthaea/issues/13273) and the broader manufacturing-process semantics issue [#5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686). For this programme, map fields and adapter behavior—not a claim of formal conformance:

| Concern | Relevant standards family | Evidence-layer use |
|---|---|---|
| Wafer/substrate identity and movement | SEMI E87 / E90 / E94 | Carrier, substrate, control-job, process-step and location lineage |
| Equipment model and self-description | SEMI E120 / E125 / E164 | Versioned equipment, module, sensor and metadata identities |
| Data collection and client trust | SEMI E132 / E134 | Authenticated read-only collection plans and provenance of acquisition |
| Measurement / data-quality state | SEMI E151 / E160 | Explicit quality status, reason, origin and quality communication |
| Equipment and process quality context | SEMI E116 / E126 / E157 | Equipment performance, quality parameters and module/process tracking |
| Process-control boundary | SEMI E133 | Map advisory evidence to APC interfaces without granting APC write authority |
| Operations/enterprise context | ISA-95 / IEC 62264 | Resource, equipment, material, job and production-operation context |

SEMI describes EDA / Interface A as a high-volume data-acquisition channel distinct from SECS/GEM equipment control; preserve that boundary in adapters. The initial integration target should be authenticated, read-only collection and replay, not a command path.

### GFA-VM+ cross-domain benchmark: concrete 10-label reference

The IEEE Early Access abstract for GFA-VM+ (published 29 October 2026; DOI [10.1109/OJIES.2026.3748865](https://doi.org/10.1109/OJIES.2026.3748865)) reports, at ten target labels:
- EPI thickness: MAE 1.01 nm;
- EPI doping: MAE (3.49\times10^{15}\) cm⁻³;
- EPI-to-ETCH etch depth: MAE 4.51 nm;
- early-ramp cross-fab critical dimension: MAE 3.19 nm.

It reports diminishing-return regions near nine target labels for cross-machine and eleven for cross-process transfer. Its architecture combines mask-aware sensor alignment, variational representation learning, MAML-style adaptation and optional FedAvg. It also reports reduced performance under severe sensor dropout and empirical model-update inversion; raw-data non-centralization is not a privacy proof. <Cite refs={["turn293282search1"]}/>

These are four different physical targets and domains, not one pooled score and not comparable to SECOM AP or CMP MRR RMSE. The correct challenge is to reproduce each exact task/protocol with matched label draws, then compare the *additional assurance* we can provide: calibrated uncertainty, sensor/units mapping validation, negative-transfer detection, abstention and update-inversion resistance under a declared attacker.

## Industry comparison: what would count as better

The current baseline landscape is already more advanced than an ordinary SECOM classifier:

- **Industrial VM/inspection/adaptation:** Bertelson et al. report a linear-Gaussian dynamic VM and physical-metrology selection/adaptation framework evaluated on 114,774 wafers from a 300 mm plasma-etch fab (IEEE TSM 2026, DOI [10.1109/TSM.2026.3709162](https://doi.org/10.1109/TSM.2026.3709162)). We should not pretend that a public SECOM split beats this industrial evidence.
- **Cross-domain few-shot transfer:** GFA-VM+ reports 10-target-label MAEs of 1.01 nm EPI thickness, 3.49 × 10^15 cm⁻³ EPI doping, 4.51 nm EPI-to-ETCH depth and 3.19 nm early-ramp cross-fab CD, with diminishing returns around 9–11 labels. Those are separate tasks and cannot be pooled or compared to SECOM AP. <Cite refs={["turn293282search1"]}/>
- **CMP temporal prediction and constrained screening:** a 2026 PHM 2016 CMP study reports R² 0.9543, RMSE 5.9743, MAE 4.7199 and 95.96% empirical coverage for guarded 95% intervals on 396 time-out observations. In its 20 model-in-the-loop cases, the proposed and constrained baselines both attained 95% target-window hit rate and 100% feasibility; the authors attribute the feasibility improvement to the shared constraints rather than a unique optimizer gain. This is a useful public-data comparator, not a physical recipe-intervention result. <Cite refs={["turn293282search0"]}/>
- **Manufacturing systems benchmark:** NIST TIDAL explicitly expects context ablations, grounding, latency/cost measurement, digital-twin VVUQ and human-machine teaming. Context and measurement metadata should be benchmark dimensions, not narrative decoration. <Cite refs={["turn324740view2"]}/>

The relevant question is therefore not “can our model beat one headline MAE?” It is “can we deliver better-qualified decisions at the same or lower measurement cost, while accounting for assay error, selection bias, calibration drift, cross-domain failure and privacy leakage?”

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
- synthetic-unit measurements that cannot be mistaken for physical metrology;
- explicit synthetic assay sensitivity/specificity, with deterministic per-wafer measurement-error replay.

The evaluator now separates defects never selected for inspection from assay false negatives and false positives. It reports selected-defect capture, actual detected-defect capture, total undetected defects and an assay-adjusted Horvitz–Thompson prevalence estimate using the declared sensitivity/specificity. The adjusted estimator is marked unqualified when the assay is uninformative (Youden's J is zero) or design variance cannot be estimated from the randomized audit sample. The policy-facing observation view omits the assay characteristics and hidden outcomes.

This is still a synthetic error model with constant, known assay characteristics. It does not model continuous metrology error distributions, instrument drift during a run, cross-wafer correlated measurement error, reference-standard traceability, gauge R&R, or operator/instrument-specific sensitivity. Those must be added before using this to represent a real metrology system.

The evaluator accepts a propensity-bearing audit selection report. It checks that the selection digest matches its payload; row identities, strata and label semantics match; row order is canonical; deterministic exploit ordering replays from recorded priority scores and costs; the exact per-stratum SRSWOR draw replays from the declared seed; inclusion probabilities equal declared stratum quota divided by residual population; report counts and costs reconcile; and budget limits are respected. A negative fixture swaps one selected and unselected audit unit while preserving the quota and digest; the replay verifier must reject it.

The remaining receipt limitation is tracked in [#13336](https://github.com/Luminous-Dynamics/symthaea/issues/13336): the v1 selection report does not persist all upstream risk/uncertainty/EVSI components and scoring weights, so its verifier cannot yet independently reconstruct every priority score from the original candidate packet and frozen policy config. The unsigned report hash is not an authenticated producer signature.

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

## Measurement-system characterization and multi-fidelity metrology

The complementary workstream [#13356](https://github.com/Luminous-Dynamics/symthaea/issues/13356) expands the current fixed synthetic sensitivity/specificity model into an actual measurement-system characterization benchmark: repeatability, reproducibility, trueness/bias, linearity, hysteresis, resolution, calibration stability, correlated/common-cause errors and reference-standard traceability. It also defines a multi-fidelity policy path—cheap screening followed by higher-accuracy confirmation—where the optimized object is the value of the complete evidence package rather than the first measurement alone.

This matters because the inspection measurement is not perfect ground truth. An instrument result can be biased, correlated with other measurements, outside its calibration validity or unable to distinguish the decision threshold. The current assay simulator is only a first correction to the perfect-label assumption; it does not yet model continuous gauge error, calibration drift, reference artifacts or gauge R&R.

## Related CMP workstream: data integrity before recipe optimization

The complementary issue [#13334](https://github.com/Luminous-Dynamics/symthaea/issues/13334) addresses temporal probabilistic soft sensing and constrained Bayesian recipe screening on the PHM Society 2016 CMP Challenge dataset. It is intentionally separate from #13328: one evaluates metrology acquisition and population inference; the other evaluates whether data and process constraints are trustworthy enough to support an offline recipe proposal.

A late-2026 article (DOI [10.1080/21642583.2026.2753948](https://doi.org/10.1080/21642583.2026.2753948)) reports 672,744 sensor records, 1,981 labelled wafer-stage combinations and four unusually large MRR labels. The authors identify those values as unverified anomalies that could be consistent with export, unit-conversion or metrology faults, but the public data do not establish a cause. The correct response is not to delete or “correct” these observations by intuition. The benchmark should preserve the raw files, carry disputed-label state, compare robust and sensitivity analyses side-by-side, and require source evidence before authoritative reinterpretation.

Primary source discovery should begin with the [PHM Society 2016 CMP Data Challenge](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2016/phm-data-challenge-4/) and the [PHM Society wafer CMP dataset repository](https://data.phmsociety.org/nasa/). [Industrial Atlas](https://www.indatlas.com/datasets/phm-2016-cmp/) and [public GitHub mirrors](https://github.com/akangel0307/PHM-Data-Challenge) are useful for cross-checking file structure, but should not replace source admission, licence review and per-file digest verification.

The deeper paradigm shift is **feasibility before optimization**: a Bayesian optimizer must not maximize predicted utility through a questionable target scale or a constraint whose unit/authority is unresolved. Every proposed recipe remains an offline, independently re-scoreable candidate; this work creates no production-control authority.

## Claim ceiling

A passing synthetic replay demonstrates only that the declared policy/evaluator behaves as expected on the declared synthetic scenarios. Public-dataset reproduction demonstrates only the reproduced data/protocol. Neither proves real-fab transfer, yield gain, scrap reduction, lower operating cost, metrology traceability or safe process control.
