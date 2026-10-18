# Manufacturing Assumption Falsification and Experiment Design

**Status:** research/design proposal; not a validated manufacturing control system  
**Owner subsystem:** Symthaea engineering reasoning  
**Intended integrations:** Symtropy schedule/simulation, Sol-Atlas capacity evidence, Mycelix provenance  
**Safety posture:** advisory and evidence-gated; never directly actuates machinery or replaces qualified engineering approval

## 1. Thesis

Most engineering workflows ask whether a design or plan appears to work. This workstream asks a harder question:

> What is the cheapest, safest, most discriminating test that could show our current engineering belief is wrong?

Symthaea should maintain explicit assumptions, identify which conclusions depend on them, generate counterexamples, and propose experiments or simulations that distinguish competing explanations. It should not optimize confidence in its current answer. It should actively search for evidence that would overturn it.

This is a complement to—not a replacement for—verification, validation and uncertainty quantification (VVUQ). NIST identifies quantified uncertainty, interoperability and trustworthy validation as persistent digital-twin challenges; active learning and optimal experimental design provide established foundations for selecting informative observations. This proposal combines those foundations with explicit falsification, traceable evidence and fail-closed decision gates.

## 2. Core objects

The first implementation should use small, typed, serializable records. Keep domain-specific units explicit and avoid a generic untyped "confidence" number.

### Assumption

An assumption record must include:

- stable ID and revision;
- precise claim and scope, including operating envelope;
- kind: physical law/model form, material property, boundary condition, process capability, duration, sensor calibration, geometry, supplier, or human/operational procedure;
- value or proposition, units where applicable, and declared validity range;
- source evidence and provenance;
- uncertainty representation and its semantics (interval, distribution, epistemic ignorance, or unknown);
- dependencies: the model outputs, decisions and other assumptions that rely on it;
- a falsifier: a condition or observation that would contradict the claim;
- severity if wrong, reversibility, exposure, and accountable reviewer;
- status: proposed, supported, challenged, falsified, superseded, or not-applicable, with reason and timestamp.

A missing value must remain missing. An interval must not be silently treated as a probability distribution. A source hash establishes identity of bytes, not truth.

### Hypothesis set

Where evidence permits multiple explanations, preserve alternatives instead of prematurely collapsing them into one model. Each hypothesis names its assumptions, predicted observations, discriminating observations, and the domain in which it is expected to hold. Priors, if used, must be explicit and revisable; uncalibrated Symthaea scores are not probabilities.

### Experiment candidate

Each candidate experiment or simulation contains:

- intervention and controlled variables;
- required equipment, material, operator qualifications and safety review;
- predicted outcomes under each competing hypothesis;
- measurement method, calibration status, resolution and uncertainty;
- cost, duration, resource demand, waste and reversibility;
- expected information gain or expected decision-value improvement, with method and assumptions;
- risk controls, stop conditions and approval requirements;
- data schema, acceptance criteria and a reproducible execution capsule.

Simulation, public-data analysis, coupon testing and full-scale testing must remain separate evidence classes. A simulation result cannot be relabelled as a physical measurement.

### Observation and decision receipt

Store raw observation references separately from transformed data and model interpretation. Record instrument identity/calibration, units, environment, operator/procedure, timestamps, transformations, exclusions, software versions and hashes. A decision receipt binds the exact assumptions, source evidence, experiment, result, model version and reviewer decision.

## 3. Decision algorithm

The initial system should be a transparent pipeline rather than an opaque agent:

1. **Scope the claim.** Define the exact engineering decision and operating envelope; refuse a broad claim when evidence only covers a narrow case.
2. **Build dependency closure.** Trace which assumptions support the decision and which downstream results depend on it.
3. **Rank critical unknowns.** Combine consequence if wrong, exposure, uncertainty, reversibility and evidence quality. Do not let an arbitrary weighted score override hard safety gates.
4. **Generate falsifiers.** For each critical assumption, state observations that would contradict it and negative controls that should remain negative.
5. **Enumerate candidate tests.** Prefer existing evidence, analytic checks and cheap reversible simulations before physical experiments, unless the cheaper test cannot discriminate the hypotheses.
6. **Predict outcomes before observing.** Freeze predictions and thresholds before execution to reduce hindsight bias and selective reporting.
7. **Select the next test.** Among approved candidates, optimize expected reduction in decision-relevant uncertainty per cost/time, while preserving coverage of important failure modes. Include diversity so a greedy information-gain strategy does not repeatedly sample one region.
8. **Check safety and authority.** Reject candidates lacking qualified personnel, valid equipment, safe bounds or required approvals. Symthaea recommends; an authorized person or validated control system approves and executes.
9. **Compare observation with predictions.** Account for measurement uncertainty, missingness and model discrepancy. Classify outcomes as support, contradiction, inconclusive, or invalid measurement—not simply pass/fail.
10. **Update without erasing history.** Revise the hypothesis set and assumptions, retain prior versions, and record whether the result falsified a claim, exposed a measurement problem, or revealed an out-of-scope regime.
11. **Re-run dependent decisions.** Any changed critical assumption invalidates downstream receipts until their dependency closure is re-evaluated.
12. **Stop responsibly.** Stop when decision risk is below an explicitly approved threshold, the experiment budget is exhausted, or remaining uncertainty cannot be resolved safely. Report unresolved uncertainty; never convert budget exhaustion into qualification.

### Candidate-selection objective

A candidate can be scored using a documented decision-theoretic objective such as:

`utility(test) = expected reduction in decision loss - test cost - risk penalty`

Information gain can be one input, but is not the objective by itself. Learning a parameter precisely is not useful if it cannot change the engineering decision. Hard safety and authorization constraints are filters, not terms that can be traded away for utility.

For early deterministic implementations, use exact enumerated hypotheses and analytic outcomes. Introduce Bayesian expected information gain only when likelihoods, priors and calibration are documented. Preserve an explicit unknown/insufficient-evidence state.

## 4. Manufacturing vertical slice

Start with a low-risk, measurable product: a non-safety-critical bracket, fixture, sensor mount or enclosure. Avoid starting with pressure boundaries, lifting hardware, flight-critical components, structural ship welds or other high-consequence parts.

For the first vertical slice:

1. Freeze a versioned CAD/requirements baseline and a small set of dimensional or process claims.
2. Identify assumptions about material, stock condition, tool access, fixturing, machine capability, measurement uncertainty and process time.
3. Generate a tolerance stack and a schedule using Symtropy, keeping nominal feasibility separate from robust feasibility under declared duration bounds.
4. Have Symthaea identify the assumptions whose failure could invalidate the result.
5. Generate a ranked list of safe tests: analytic checks, gauge/fixture verification, toolpath simulation, inspection-system checks, then physical coupon or part measurement as appropriate.
6. Pre-register predicted outcomes, uncertainty, acceptance criteria and negative controls.
7. Capture actual measurements and discrepancies. Do not tune the model against the same data and then report that data as independent validation.
8. Publish a reproducible report with source hashes, units, calibration records, excluded observations, model versions and exact unresolved issues.
9. Promote only the narrowly scoped claim that the evidence supports.

The initial product is not an autonomous factory. It is a reproducible engineering investigation that makes weak assumptions visible before they become expensive physical failures.

## 5. Benchmark suite

All targets below are proposed gates. They must be measured against pinned corpora and independent expected results; none is claimed achieved by this design document.

### A. Assumption integrity

- 100% of decision-critical assumptions have a scope, provenance, falsifier and explicit status before the decision is eligible for review.
- 100% of downstream receipts are invalidated or re-evaluated when a depended-on critical assumption changes.
- 100% of unknown or malformed critical values remain unknown/invalid; no coercion to zero or a positive confidence score.
- Exact reproducibility of normalized reports for identical inputs, code and configuration.

### B. Falsification quality

- Zero missed contradictions in a pinned synthetic adversarial corpus where the falsifier is logically entailed by the input.
- Zero false claims of falsification when the observation is within the declared measurement uncertainty.
- Negative controls are run and reported; no test can silently omit a failed or inconvenient observation.
- Track the fraction of seeded false assumptions detected, time/cost to detection, false alarms, inconclusive tests and severity-weighted missed detections.
- Evaluate on held-out scenarios and mutated assumptions, not only examples used to author the rules.

### C. Experiment selection

Compare against random selection, a fixed expert-authored sequence, uncertainty-only sampling and a simple cost-normalized information-gain baseline.

- On a synthetic oracle with known ground truth, reach a predeclared decision threshold with at least 30% fewer experiments than random selection, across a pinned suite of problem classes. This is a research target, not a promised result.
- Report performance over at least 30 independent seeds per stochastic benchmark, with confidence intervals and all failures.
- Include adversarial cases where the most uncertain variable is not decision-relevant; ensure selection is driven by expected decision value rather than uncertainty alone.
- Measure wall-clock time, compute cost, physical experiment cost (when known), decision regret, and coverage of distinct failure modes.
- Compare the selected experiment with an exact optimum on small enumerable problems; report optimality gap or regret on larger problems.

### D. Uncertainty and calibration

- Separate aleatoric variation, epistemic uncertainty, measurement uncertainty, model discrepancy and missing evidence in the schema and report.
- For probabilistic predictions, report held-out interval coverage, calibration by process/material/operating regime, sharpness and sample count. Do not claim P90/P95 reliability from min/max bounds or synthetic data alone.
- Test distribution shift, sensor bias, correlated disruptions and out-of-domain observations.
- A model outside its validated envelope must return abstain/escalate, not an unqualified recommendation.

### E. Evidence chain

- 100% of decision-eligible claims resolve to exact source revisions, units, transformation versions and applicable calibration evidence.
- Mutation tests alter a source artifact, unit, calibration state, threshold, assumption or model version; every relevant change must either change the result/receipt or cause explicit rejection.
- Replaying the same execution capsule produces the same normalized deterministic receipt.
- A digest mismatch, missing authority or stale critical dependency causes fail-closed rejection.

## 6. Architecture and boundaries

- **Symthaea:** assumption graph, contradiction reasoning, hypothesis comparison, experiment ranking and uncertainty-aware explanation.
- **Symtropy:** geometric/process simulation, schedules, resource calendars and scenario execution. The schedule validator remains independent of experiment ranking.
- **Sol-Atlas:** location- and facility-specific capability evidence, capacity estimates and provenance-aware availability. Estimates must not masquerade as confirmed machine slots.
- **Mycelix:** versioned evidence exchange, authorship/authority records and integrity linkage. Cryptographic integrity does not prove engineering truth.
- **Independent verifier:** checks input schemas, hard constraints, evidence lineage, frozen predictions and receipt determinism without reusing the same high-level reasoning path as the proposer.

Avoid coupling the first prototype to a large agent framework. Start with Rust data types, deterministic rules and a small oracle-backed experiment selector. Add probabilistic models only behind explicit interfaces and calibration gates.

## 7. Threat model and failure modes

The design must explicitly test for:

- confirmation bias and post-hoc changing of thresholds;
- selection bias, survivorship bias and unreported failed experiments;
- leakage between training, calibration and evaluation sets;
- sensor drift, unit mismatch, timestamp mismatch and stale calibration;
- confounding variables and unmodelled common-cause events;
- false precision, uncalibrated probabilities and model extrapolation;
- tampered or superseded evidence;
- unsafe test proposals, unqualified operators or invalid equipment;
- a model declaring its own output valid without an independent check;
- incomplete dependency closure that leaves downstream approvals apparently current.

For physical experiments, the system must not issue machine commands. Human authorization, interlocks, risk assessment, material traceability and applicable codes remain mandatory.

## 8. Research basis and relationship to existing practice

This proposal extends established practice rather than claiming to invent VVUQ or active learning:

- NIST, **Digital Twins for Advanced Manufacturing**: https://www.nist.gov/programs-projects/digital-twins-advanced-manufacturing — emphasizes trustworthy digital twins, quantified uncertainty, interoperability and VVUQ.
- NIST, **Digital Twins Workshops Summary Report**, NISTIR 8620 (published 2026-07-21): https://www.nist.gov/publications/digital-twins-workshops-summary-report — highlights interoperability, VVUQ, cybersecurity and workforce readiness.
- Yang, Chen and Wu, **Active learning of model discrepancy with Bayesian experimental design**, Computer Methods in Applied Mechanics and Engineering (2025): https://doi.org/10.1016/j.cma.2025.118198 — studies informative sampling for learning discrepancies between digital-twin models and observed behaviour.
- Huan, Jagalur and Marzouk, **Optimal experimental design: Formulations and computations** (2024): https://arxiv.org/abs/2407.16212 — surveys information-based, Bayesian and decision-theoretic experiment design.

The proposed contribution is the integration contract: explicit falsifiers, dependency invalidation, safe candidate selection, pre-registered predictions, negative controls and evidence receipts tied to manufacturing decisions.

## 9. Milestones

1. **M0 — Schema and deterministic evaluator:** assumption/hypothesis/experiment/observation/receipt types; malformed-input tests; no ML required.
2. **M1 — Synthetic oracle:** small finite hypothesis sets, known outcomes, falsification and dependency-invalidation tests.
3. **M2 — Selector baseline:** random, fixed-sequence and cost-normalized policies; deterministic seeds; regret and cost metrics.
4. **M3 — Manufacturing fixture slice:** a low-risk bracket or fixture with requirements, tolerance model, schedule, measurement plan and explicitly unqualified physical status until measured.
5. **M4 — Public dataset adapter:** pinned, rights-cleared artifact and parser; separate training/calibration/evaluation partitions; no claim of physical qualification from dataset ingestion alone.
6. **M5 — Controlled physical study:** only after risk review, suitable equipment, calibrated instruments, qualified procedures and human approval.
7. **M6 — Deployment assessment:** cybersecurity, operator workflow, failure recovery, auditability and independent review.

Each milestone has its own acceptance receipt. Passing a synthetic test does not advance the physical-qualification state. No milestone is passed merely because CI is queued, skipped or green on a different commit.

## 10. Exit criteria for the first prototype

The first prototype is useful when it can take a small, versioned engineering decision; enumerate critical assumptions and dependencies; generate falsifiers and safe candidate tests; compare selection policies against an independent oracle; preserve failed/inconclusive outcomes; and produce a reproducible receipt that changes when critical evidence changes.

The prototype should make it easier for a human engineer to discover that they are wrong—early, cheaply and with a traceable reason.
