# CMAQ-006: Sequential Causal-Mechanism Change and Adaptation

**Status:** protocol RFC; evaluator and candidate adapters are not implemented. No Symthaea results or qualification claims.  
**Track:** CMAQ / causal world-model qualification  
**Research question:** when a causal mechanism changes during deployment, can an agent detect the change, determine what remains invariant, revise its causal predictions, and recover safe decision quality without repeatedly relearning everything?

## Why this is a separate track

CMAQ-005 asks whether intervention predictions transfer to a mechanism family withheld during development. CMAQ-006 asks a different question: after a system has learned and used a mechanism, what happens when that mechanism changes *within a live sequence*?

A model can pass a static held-out transfer test and still fail in deployment by retaining a stale causal belief, detecting a shift too late, overreacting to harmless observation changes, or updating the wrong part of its model. CMAQ-006 measures this sequential failure-and-recovery process.

The intended causal loop is:

**forecast -> act or probe -> receive independently recorded outcome -> detect possible change -> localize changed mechanism -> update only justified beliefs -> act under the revised model**

Do not treat a distribution shift as proof of a causal-mechanism shift. Do not treat a causal-mechanism shift as proof that every previously learned relationship is invalid.

## Threat model and candidate/evaluator boundary

The evaluator generates ordered sequences with an evaluator-only mechanism schedule. Candidate-visible observations may include ordinary state, context, prior permitted actions, action receipts, outcome timing and the declared action/probe budget. They must not include the hidden mechanism identifier, regime identifier, future change point, unobserved counterfactual outcomes or evaluator safety labels.

For each decision step, a candidate trace should record:
- opaque case/step identity and sequence position;
- predicted outcome distribution for each available action, or an explicit abstention where the effect is not identifiable;
- predicted safety probability for each action;
- selected action or abstention;
- whether a mechanism change is suspected, and the first step at which the alarm is raised;
- optional causal diagnosis: changed mechanism component, affected variables, confidence and evidence references;
- experiments/probes requested, with their costs and authorization.

The independent environment recorder—not the candidate—supplies observed outcome, actual safety, action identity and postcondition evidence. Candidate self-reports must never be accepted as environment truth.

The first simulator lane may contain potential outcomes for all actions because its structural equations are known to the evaluator. A real-world adapter must score only outcomes actually observed under performed interventions, with logged propensities and explicit identification assumptions. It must not fabricate unchosen potential outcomes.

## Shift taxonomy

The generator should isolate these shift classes before combining them:

1. **Parameter shift:** a structural equation keeps its form but one or more coefficients change.
2. **Functional-form shift:** linear, nonlinear, threshold, saturating or interaction terms change while the variable set may remain fixed.
3. **Graph/mechanism shift:** a parent edge appears, disappears or reverses, or a variable's causal role changes.
4. **Noise/uncertainty shift:** exogenous-noise variance, tail behavior or dependence changes while the structural function is held fixed.
5. **Observation/measurement shift:** sensors, missingness, units, encodings or observation channels change while the underlying physical mechanism remains fixed. This is a *nuisance-shift control* unless it also changes the causal system.
6. **Population/context shift:** effect modifiers or population composition change; the source effect may or may not transport.
7. **Actuator/intervention shift:** the same nominal action produces a different physical intervention, dose, latency or failure probability.
8. **No-shift controls:** marginals, irrelevant features or visual appearance change while the target causal effect remains invariant.

Include both announced and unannounced shifts, abrupt and gradual transitions, repeated shifts, reversals, temporary regimes and recovery to a previous regime. Start with one controlled change point per sequence; add multiple changes only after the single-change scorer is trustworthy.

Some cases must be observationally indistinguishable before intervention while having different intervention effects. In those cases a candidate should receive credit for recognizing non-identifiability, abstaining or purchasing a safe informative probe—not for guessing the hidden answer.

## Protocol

1. Freeze model weights, configuration, memory policy, change detector, adaptation policy, candidate adapter and resource budgets before qualification.
2. Train and tune only on development mechanism families and development shift schedules.
3. Stream observations in causal time order. No future rows, hidden regime IDs or answer-key lookahead.
4. At each step, record predictions and chosen action before the evaluator reveals the next outcome.
5. Permit only predeclared, budgeted probes. The environment validates permissions, costs and safety independently.
6. Keep a separate no-shift partition to measure false alarms and unnecessary resets.
7. Hold out entire mechanism families and shift templates for qualification. A mechanism family must never cross development and qualification.
8. Report per-shift-class, per-family and worst-family results, with paired confidence intervals clustered by generated mechanism/sequence—not by timestep.
9. Retain immutable run manifests, generator version, candidate digest, adapter digest, seeds, budget policy, trace digest and evaluator version.

Public fixtures may test schema behavior and known answers only. They are not a sealed qualification set and must never be used as blind performance evidence.

## Metrics: measure the whole failure-and-recovery curve

### Primary decision metrics
- Cumulative constrained regret before and after each change point.
- Post-change regret area under the curve over a fixed adaptation horizon.
- Safe task completion and unsafe executed-action count (non-compensatory hard gate).
- Intervention/probe cost and utility gained per unit of probe cost.
- Worst-family and worst-shift-class regret, not just a pooled mean.

### Detection metrics
- Detection delay from the evaluator's change point to the first alarm.
- Missed-change rate.
- False alarms per 1,000 no-shift steps.
- Precision/recall and alarm lead/lag distribution.
- False alarms caused by nuisance-only shifts.

### Diagnosis and adaptation metrics
- Changed-component localization precision/recall where ground truth makes localization identifiable.
- Unchanged-mechanism retention: degradation on components not affected by the shift.
- Recovery time, defined before qualification as the first point at which the rolling constrained-regret rate stays within a preregistered tolerance of the strongest eligible post-shift baseline for a fixed consecutive window.
- Relearning cost: new observations, probes, updates and compute required to recover.
- Stale-belief persistence: unsafe or high-regret actions attributable to a superseded mechanism.

### Predictive and systems metrics
- Post-shift intervention outcome MAE and causal-effect MAE/sign accuracy.
- Safety Brier score, calibration error, and false-safe predictions.
- Forecast coverage and interval calibration if predictive intervals are emitted.
- Wall-clock latency, peak memory, CPU/GPU time, energy where measurable, and number of model/memory updates.

Never combine these into one compensatory score. Safety, authorization and evidence integrity remain hard gates.

## Required baselines

All methods receive the same observation history, action menu, probe budget, allowed context and compute budget.

1. Frozen pre-shift model with no detector or adaptation.
2. Sliding-window retraining / recency-only model.
3. Exponentially weighted residual detector with a fixed adaptation rule.
4. Page-Hinkley and CUSUM change detectors.
5. ADWIN or another declared adaptive-window detector.
6. Bayesian online change-point detection, with the prior and hazard rate preregistered.
7. Online linear/system-identification baseline where its assumptions hold.
8. Invariant prediction / invariant risk minimization or effect-invariant baseline where multiple environments and assumptions support them.
9. HDC-only, recurrent-only (including matched CfC/LTC), memory-only, world-model-only and planner-only ablations.
10. Full Symthaea system, with each subsystem's update/reset policy frozen.
11. Oracle change-time and oracle post-change model, reported only as a non-deployable upper bound.

Strong statistical baselines must be tuned on the development split. Do not compare the hybrid only with a weak fixed prior or a deliberately uncalibrated detector.

## Negative and adversarial controls

- Shuffle action/outcome links while preserving marginal outcome frequencies.
- Shift only irrelevant observations; the candidate should avoid unnecessary causal-model resets.
- Change the intervention actuator while leaving its nominal action label unchanged; the candidate should detect that the intervention itself changed.
- Hide the change point and vary the number of post-change observations before evidence becomes available.
- Reverse the causal effect sign after a regime transition.
- Introduce a temporary regime and then restore the original mechanism.
- Replay delayed, duplicated, stale or mismatched outcome receipts.
- Poison the memory with repeated but unverified post-shift claims.
- Provide two observationally equivalent regimes with different interventional effects; require abstention or informative safe experimentation until identified.
- Give a detector a no-shift sequence with high noise to expose alarm-rate inflation.

A benchmark that continues to pass when its causal evidence is shuffled or its nuisance shifts are mislabeled is not qualified.

## Proposed qualification targets

Targets are hypotheses to preregister after a development-only power and variance study, not achieved results:

- At least 10% lower post-change constrained regret than the strongest eligible matched-budget statistical/adaptive baseline, with paired 95% confidence intervals supporting improvement.
- No unsafe executed actions and no unauthorized probes on the qualification panel.
- No more than one false alarm per 1,000 no-shift steps, with the denominator and confidence interval reported.
- Report median and p95 detection delay, missed-change rate, recovery time and worst-family regret.
- Demonstrate that the model preserves unaffected mechanisms better than full reset/retrain and adapts faster than the frozen model.
- Demonstrate an explicit abstention/probing advantage on non-identifiable cases rather than rewarding unsupported point predictions.

These goals must be revised only before qualification is opened. A failed target is a result, not permission to weaken the gate after seeing results.

## Relationship to external benchmarks

CMAQ-006 is a focused sequential adaptation lane, not a replacement for external benchmarks.

- **WorldTest / AutumnBench (ICML 2026):** environment-level and counterfactual queries beyond a single observed trajectory. Run the published protocol separately: https://proceedings.mlr.press/v306/warrier26a.html
- **World-In-World (ICLR 2026):** closed-loop world-model utility and action-conditioned task success. Match inference-time compute when comparing: https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html
- **Active Causal Discovery Benchmark (2026):** graph recovery and budgeted interventions; useful for discovery capability, not a substitute for sequential change adaptation: https://arxiv.org/abs/2609.31675
- **CausalGame (ICML 2026):** interactive causal discovery under a finite experimental budget: https://mbzuai-clear.github.io/projects/causalgame/
- **EconCausal (2026):** empirical cross-context causal-effect transfer and source-sign over-transfer; useful for a real-evidence transfer challenge, distinct from synthetic sequential drift: https://econai.kaist.ac.kr/econcausal/
- **CSuite:** observational and interventional SCM data for causal discovery and ATE/CATE estimation: https://github.com/SCAI-Lab/csuite-causal-benchmark
- **IncomeSCM (NeurIPS 2024):** sequential SCM construction from real tabular distributions, including mechanism compositions: https://papers.nips.cc/paper_files/paper/2024/hash/f5c7fb98367440a5eb01d70294c180ca-Abstract-Datasets_and_Benchmarks_Track.html
- **CauseMe:** nonlinear and time-series causal discovery challenges including autocorrelation, nonlinearity, chaotic dynamics and measurement error: https://causeme.uv.es/

Report every external score under its published split, metric and protocol. Never place CMAQ and external benchmark scores in a shared leaderboard as if they were commensurate.

## Integration with CMAQ-001 to CMAQ-005

- CMAQ-001: does causal memory improve action selection?
- CMAQ-002: does model-guided planning survive closed-loop execution?
- CMAQ-003: does uncertainty-aware selection prevent exploitable decisions?
- CMAQ-004: are active experiments worth their cost?
- CMAQ-005: do intervention predictions transfer to held-out mechanisms?
- **CMAQ-006: can the system detect, diagnose and safely adapt when mechanisms change during deployment?**

The eventual shared protocol should freeze candidate artifacts and resource budgets, use independent outcome receipts, report per-family/worst-family metrics, and preserve qualification custody. This RFC deliberately does not claim that one shared binary scorer already exists.

## Implementation sequence

1. Finalize a generator contract and one-change-per-sequence trace schema.
2. Implement a dependency-free Rust evaluator with strict parsing, split isolation, sequence ordering, complete coverage, safe-action gates and deterministic known-answer fixtures.
3. Add no-shift and nuisance-shift negative controls, then repeated/gradual-shift cases.
4. Implement baseline adapters and a matched-budget experiment runner.
5. Freeze a separate qualification panel and candidate artifacts before scoring.

**Current limitation:** this PR starts the research protocol only. It does not implement the evaluator, train Symthaea, run a qualification experiment or establish superiority over any baseline.
