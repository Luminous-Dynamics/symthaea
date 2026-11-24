# SEMI causal drift diagnosis V0 — 24 November 2026

**Issue:** #13494  
**Parent benchmark:** CEB causal-debugging / ledger integration #10366 / #10387  
**Status:** domain mapping + synthetic SCM development oracle; not an industrial diagnosis capability.

## Decision: reuse the CEB causal ledger

The generic causal-investigation ledger already exists as a standalone research tool in:

- `tools/ceb-causal-ledger/src/lib.rs`
- `docs/research/CEB_CAUSAL_INTEGRATION_ARCHITECTURE_V0_2026-11-04.md`
- branch `research/ceb-causal-debugging-2026-11-04`

It defines task-bound evidence, competing hypothesis revisions, provenance authority, evidence ancestry, prospective forecasts, typed actions, outcomes with explicit missingness, matched ablations and proceed/abstain/escalate decisions.

This semiconductor programme does not introduce a second ledger, graph, action vocabulary, receipt or EVSI scheduler. It contributes:
1. semiconductor mechanism taxonomy and graph mapping;
2. synthetic process/sensor/metrology SCM scenarios with known interventions;
3. diagnostic and experiment-selection evaluation;
4. later, a thin domain adapter from semiconductor evidence/events into CEB records.

The upstream CEB tool remains standalone and unqualified as a production dependency. The currently described prototype uses caller-provided opaque digests and performs structural validation only; this domain layer does not upgrade those limits.

## Paradigm shift

A residual is a symptom, not a cause. A feature attribution is a model explanation, not an intervention result. A VM label is a measurement, not necessarily physical truth.

The diagnostic system should maintain a *set of competing mechanisms* and ask which measurement or safe experiment would distinguish them. It must be permitted to conclude NOT_IDENTIFIABLE.

## Domain causal variables

| Variable family | Examples | Causal role |
|---|---|---|
| Product/route | wafer family, layer, route, target specification | selection/confounding context |
| Recipe/actuator | RF power, pressure, gas flow, time, temperature | controllable process inputs |
| Tool/chamber | chamber identity, wall condition, hardware age, maintenance | process response context |
| Latent process | plasma state, etch rate, film growth, removal response | physical mechanism |
| Sensor | raw channels, calibration, mapping, sample rate, clock | imperfect process observation |
| Metrology | instrument, measurand, calibration, reference standard, uncertainty | measured quality evidence |
| Observation policy | inspection decision, sample selection, queue delay | label availability/selection mechanism |
| Common cause | utilities, shared gas, shared reference, common maintenance | cross-tool dependence |

A candidate causal graph includes:

`product/route -> recipe assignment -> latent process state -> physical quality`

`tool/chamber/maintenance -> latent process state`

`latent process state -> sensor -> recorded telemetry`

`sensor calibration/mapping/clock -> recorded telemetry`

`physical quality -> metrology instrument/reference -> reported label`

`inspection policy -> which reported labels become observed`

`shared utility/reference -> multiple tools/chambers`

The graph is a model to test, not an assumed ground-truth manufacturing law. Every edge used for an identification claim needs stated assumptions and evidence authority.

## Hypothesis taxonomy

The diagnostic should preserve simultaneous and competing explanations:

1. Physical response shift: recipe-to-quality relation changed.
2. Sensor fault/drift: the recorded telemetry changed while physical quality did not.
3. Metrology/calibration/reference drift: measured labels changed while physical quality did not.
4. Chamber aging/contamination/hardware change.
5. Recipe/software/feature-map change.
6. Product/route mix shift.
7. Common-cause utility/reference/maintenance shift.
8. Observability loss: sensor dropout, remapping, sampling-rate or time-alignment error.
9. Inspection/label selection bias.
10. Unknown mixture or NOT_IDENTIFIABLE.

The generic CEB ledger's `HypothesisRecord` carries the task binding, applicability domain, competing hypothesis IDs, assumptions, supporting/contradicting evidence and epistemic status. These semiconductor mechanism IDs and evidence references are populated through that existing contract; they are not a new generic hypothesis type.

## Synthetic structural causal model

The initial reference simulator is intentionally small and transparent:

[
q = b_q + eta_r r + eta_p p + u_{tool} + u_{common}
]

[
s = a_s q + b_s + epsilon_s
]

[
y = q + b_m + epsilon_m
]

where:
- (q): latent physical quality;
- (r): recipe/actuator input;
- (p): product/route context;
- (u_{tool}): tool/chamber-specific process contribution;
- (u_{common}): shared utility/process contribution;
- (s): sensor observation;
- (a_s,b_s): sensor gain/bias;
- (y): reported metrology label;
- (b_m): metrology/reference bias;
- (epsilon_s,epsilon_m): independently generated noise.

The evaluator knows the hidden parameters and intervention assignments. The diagnostic receives only permitted observations. Scenarios can switch measurement channels on/off, delay labels, corrupt clocks, and make inspection depend on predictions or product family.

A critical negative control: with only a single reported label (y=q+b_m), a process shift and equal/opposite metrology bias can be observationally indistinguishable. The correct outcome is NOT_IDENTIFIABLE until an independent reference assay, redundant sensor, matched control or authorized intervention becomes available.

## Initial public development suite

The integration test file `crates/domains/symthaea-semiconductor/tests/causal_drift_scm_reference.rs` is a transparent development oracle. It tests:

- process shift moves physical-reference measurement and sensor response together;
- sensor-only drift moves telemetry while physical reference remains stable;
- metrology bias moves reported labels but not the independent physical reference;
- common-cause shifts affect multiple tools in a shared pattern;
- chamber-specific aging remains localized;
- process shift and metrology bias can be observationally equivalent without an independent measurement;
- a reference assay breaks that equivalence;
- product mix can move the pooled mean without changing within-family response;
- absent sensor channels remain missing, not zero;
- selection-biased metrology cannot be equated with population prevalence.

These are known-answer simulator checks. They do not evaluate an AI model, demonstrate causal discovery, represent production data, or constitute a sealed benchmark.

## Measurement / experiment selection

Candidate discriminating observations should reuse the CEB forecast/action/outcome record semantics:
- independent reference-standard check;
- repeat metrology on a retained sample;
- redundant sensor or independent channel;
- matched control chamber with same recipe and product;
- maintenance/consumable record verification;
- independent wafer-site measurement;
- synchronized telemetry/event-clock audit;
- randomized audit sample from the unmeasured population.

For candidate request (a), evaluate expected reduction in mechanism uncertainty (or decision loss) per cost:

[
EVSI(a)=mathbb{E}_{osim p(omid a,mathcal{H})}
  [V(mathcal{H})-V(mathcal{H}mid o,a)]-C(a)
]

This equation is a design target; implementation must handle calibrated likelihoods, unknown models, capacity, measurement uncertainty and action risk. If the likelihood model is not defensible, preserve the request as a heuristic ranking rather than a numeric causal probability.

Forecasts must be immutable and precede the observation. Missing counterfactuals remain missing, not zero.

## Baselines and competitive comparison

- CUSUM/EWMA and static/adaptive control charts.
- PCA/PLS residual monitoring and Hotelling T²/Q.
- M2D2-style multimodal short-/long-term drift scores.
- Isolation Forest / one-class SVM.
- Supervised FDC where labels exist.
- SHAP/LIME/attention explanations attached to drift alarms.
- Bayesian dynamic VM with explicit observability-error modelling.
- Causal discovery/invariance/change-point baselines where their assumptions hold.
- CEB ledger plus a semiconductor EVSI measurement selector.

M2D2 is a 2025 published comparator for multi-machine/multimodal drift detection and adaptation. Its feature explanations should be compared as *diagnostic evidence*, not presumed physical cause. The 2022 Bayesian VM paper is a comparator for separating concept drift from observability errors.

## Metrics

**Detection:** event precision/recall, AUROC/AUPRC, false alarms per tool-week and per 1,000 wafers, detection delay and missed-event duration.

**Diagnosis:** top-1/top-3 mechanism accuracy where identifiable; true mechanism coverage in the reported hypothesis set; false causal assertion rate; NOT_IDENTIFIABLE accuracy on deliberately indistinguishable cases; common-cause localisation and confounder sensitivity.

**Experiment selection:** true-mechanism discrimination per metrology dollar/sample; EVSI regret versus oracle selection; measurement budget consumed; number of hypotheses eliminated; randomized-audit correction.

**Robustness:** missing-not-at-random inspection, sensor remapping, clock skew, calibration revocation, delayed labels, common-cause events, product mix, unknown chambers and simultaneous faults.

## Proposed research gates

- 100% NOT_IDENTIFIABLE on constructed observation-equivalent mechanisms until discriminating evidence arrives.
- >=95% top-3 mechanism recall on identifiable synthetic scenarios at a declared false-alarm budget.
- <=1 false causal assertion per 10,000 synthetic wafers in the high-consequence suite.
- detect a common reference shift without falsely labelling every chamber as independent.
- improve mechanism discrimination per fixed metrology budget over random/fixed selectors with paired confidence intervals.
- an intervention-supported status requires intervention evidence in the reused CEB ledger; correlation or feature attribution cannot promote it.

These are hypotheses, not achieved results or industry standards.

## Claim ceiling

This is a transparent synthetic causal fixture and domain mapping. It is not a real-fab diagnosis system. Correlation and feature attribution do not establish physical causality. Real causal identification requires trustworthy event alignment, calibrated metrology, interventions or justified quasi-experiments, confounder control, process expertise and prospective validation.
