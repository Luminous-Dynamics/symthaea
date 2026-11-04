# SAM Decision-Sensitive Measurement v0.1

**Status:** research protocol / analytic fixtures  
**Issue:** [SAM-SENSE-001 #10320](https://github.com/Luminous-Dynamics/symthaea/issues/10320)  
**Composition:** #10256, #10252, #10257, #10237, #10266, #10318, #10012, #10083; Symtropy #3471 and #3387  
**Claim ceiling:** query-selection behavior on named analytic/benchmark tasks; no physical safety or vehicle-control authority.

## 1. Objective

Select the next observation or computation by its potential to change a declared decision—not by prediction uncertainty or information gain alone.

A large uncertainty set may be irrelevant when every admitted state/reference supports the same action. A narrow interval can be decisive when it crosses a hard feasibility boundary or reverses a pairwise action preference.

The experiment is:

`frozen decision context
  -> admitted state/model/reference envelope
  -> action-feasibility and preference classification
  -> decision-invariance test
  -> candidate measurement/query evaluation
  -> observe / compute / abstain
  -> recompute the decision envelope`

This is not a new sensor stack or planner. It consumes prediction sets, reference qualifications, finite action sets and replay receipts owned by existing work.

## 2. Terms and outcome states

A **measurement action** may be a physical sensor request, a model query, an independent reference execution, or a derived computation. These are different evidence methods and MUST keep separate provenance, applicability, uncertainty and cost.

The current decision envelope returns one of:

- `DecisionInvariant`: one action remains preferred and required hard-constraint dispositions do not change across every admitted cell.
- `FeasibilitySensitive`: at least one admitted cell changes an action's hard-feasibility disposition.
- `PreferenceSensitive`: at least two admitted cells support different strict action preferences.
- `Unresolved`: the admitted evidence does not determine invariance or sensitivity.
- `Incomplete`: expected action/reference/measurement cells or required identities are absent, failed, censored, stale, timed out or unsupported.
- `OutOfScope`: the required measurand or reference is not qualified/applicable to the current context.

These classifications are conditional on the declared envelope. `DecisionInvariant` is not safety certification, global optimality, permission to execute, or evidence that the envelope spans reality.

## 3. Frozen decision context

Each campaign MUST bind:

- protocol/schema version and immutable campaign identity;
- scenario, initial state, environment and decision-time observation history;
- exact finite action set, command payloads and completeness declaration;
- each objective and hard constraint, including units, frames, direction, margins and threshold semantics;
- prediction-set/model profiles and source artifact digests;
- reference qualification, applicability and common-mode dependency manifests;
- actor-policy and permitted-observation profile;
- clock/tick frequency, decision time, forecast horizon and hard deadline;
- available measurement actions, permissions, resource budget and cost model;
- exact run/output/evidence receipts.

Planner selection is committed before query outcomes or evaluator-only reference outcomes are exposed. A new observation creates a new evidence state linked to—but not overwriting—the previous one.

## 4. V0 decision-invariance test

For each action (a), objective (j), constraint (k), admitted state/reference cell (s), compute:

- feasibility disposition under each hard constraint;
- objective interval/value under its declared uncertainty semantics;
- pairwise action preference using the frozen per-objective rule.

The decision is invariant only when the selected preferred-action set and every required hard-constraint disposition are identical across all admitted cells, with no incomplete or out-of-scope required cell.

A reference count is not a probability. Do not create a probability distribution by counting references, models, synthetic seeds or scenario variants. Common-mode dependencies remain disclosed via #10266.

When action sets are incomplete, reference applicability is unknown, or a required result is missing, return `Incomplete`/ `OutOfScope`; do not claim invariance.

## 5. Candidate measurement contract

A candidate measurement/query MUST declare:

| Field | Required meaning |
|---|---|
| Query identity | Stable ID, protocol version, artifact/config digest |
| Evidence method | Physical sensor, model query, reference execution, or derived computation |
| Target | Measurand, unit, frame, spatial/temporal scope |
| Access | Authorized observer, allowed input fields and delivery channel |
| Freshness | Maximum age, observation tick, delivery tick, freshness rule |
| Qualification | Source identity, calibration/reference qualification and context of use |
| Latency | Evidence-backed bound or calibrated latency distribution |
| Cost | Compute, energy, money, communication and/or opportunity cost |
| Outcomes | Observed, stale, missing, occluded, censored, timed out, failed, unsupported |
| Causal effect | Whether the query changes the world or observability |
| Deadline | Latest useful delivery tick and clock policy |

A second endpoint or new model run ID is not evidence independence. Reuse common-mode manifests and evidence ownership. An unavailable or stale result never narrows the admitted envelope.

## 6. Value hierarchy

The protocol deliberately separates three levels.

### V0 — robust decision-set reduction

No probabilities are needed. Enumerate the possible outcomes of each query under the admitted query model, partition the current envelope, and report the worst-case and per-outcome reduction in:
- unresolved pairwise action preferences;
- feasibility ambiguity;
- size of the non-dominated candidate set, where the multi-objective relation is fully specified.

Do not treat the cardinality of a reference list as an uncertainty probability. Query outcome branches must be justified by the query contract; a fabricated outcome partition is not information.

### V1 — robust regret-envelope reduction

Allowed only when a single frozen scalar utility, complete finite action set and qualified reference set justify regret. For each possible query outcome, compute the upper finite-action regret envelope before/after observation. Report worst-case regret reduction, per-reference values and measurement cost/latency separately.

No global action optimum is implied. The result is conditional on the enumerated candidates and reference portfolio.

### V2 — expected value of sample information

EVSI is allowed only when query-outcome probabilities are calibrated for the exact query/context and the selection/sampling process is qualified. Bind the predictive distribution, calibration data, evaluation split, dependence, source and uncertainty. Show sensitivity to plausible probability misspecification.

Never infer query probabilities from the number of references, a small synthetic fixture or the mere existence of model ensembles.

## 7. Query selection and stop rules

Keep decision consequence, feasibility resolution, deadline risk, compute, sensing cost, energy and communication as separate Pareto dimensions unless a frozen, justified utility explicitly requires scalarization.

The query selector MUST NOT choose a query solely because it has high entropy reduction or high prediction-error reduction. A measurement with no effect on action preference/feasibility can still be useful for a separately declared scientific purpose, but that is a distinct objective.

After each valid observation, recompute the envelope. Stop when:
- the decision is invariant under the declared envelope and all required checks are complete;
- no admissible query can return before the hard deadline;
- the sensing/compute budget is exhausted; or
- evidence integrity/applicability is insufficient.

Return distinct outcomes such as `NoQueryDecisionInvariant`, `QuerySelected`, `Abstain`, `DeadlineExceeded`, `BudgetExhausted`, `EvidenceIncomplete`, and `OutOfScope`. Control authority, safe fallback and actuation remain with #10012 and other declared owners. This evaluator MUST NOT command vehicle actuators.

## 8. Causal and temporal rules

- A query result cannot influence a decision before its declared delivery tick.
- Actor policies receive only the observation allow-list permitted by Symtropy #3471.
- Hidden simulator state, future ego commands, holdout IDs and evaluator-only outcomes are forbidden query/policy inputs.
- Query selection is committed before query results are exposed.
- If a query or intervention changes occlusion, actor behavior, environment or sensing conditions, downstream states must be recomputed; do not force matched branches to retain conditions that are no longer exogenous.
- Exact same-runtime replay and portable semantic replay remain separate, using #3387 receipts.
- Duplicate observations, events and receipts cannot increase the apparent number of independent measurements.

## 9. Falsification fixtures

The machine-readable fixture `fixtures/SAM_DECISION_SENSITIVE_MEASUREMENT_V0_1.json` defines the following known-answer cases:

1. High uncertainty with a decision-invariant action: do not query solely to reduce entropy.
2. Narrow uncertainty straddling a hard feasibility boundary: classify as feasibility-sensitive and select a timely query.
3. Stale or missing query outcome: do not reduce uncertainty.
4. A query that arrives after the hard deadline: do not select it as useful for the current decision.
5. Large entropy reduction but zero decision change: entropy-only baseline must not dominate decision-sensitive selection.
6. A query whose result reverses the action preference: recompute and retain the decision-reversal evidence.
7. Query result probabilities unavailable: V0 robust analysis allowed; V2 EVSI prohibited.
8. Shared model/data/calibration roots: no false independent-reference count.
9. Duplicate event/receipt: reject duplicate evidence.
10. Future observation or outcome leakage: reject the causal ordering.

Each case is synthetic. Fixture parsing does not execute a policy or establish sensor validity.

## 10. Benchmark ladder and comparison arms

| Stage | Evaluation | Main question |
|---|---|---|
| Analytic | Exact thresholds and finite action sets | Does the selector query exactly when decision boundaries require it? |
| Reference portfolio | #10318 fixtures | Does reference disagreement alter query value without false consensus? |
| Interactive synthetic | Symtropy #3471 and #3387 | Are actor-visible cue timing and fresh-process replay respected? |
| Motion forecasting | Argoverse 2 / Waymo Open Motion | Do coverage/uncertainty signals identify decision-sensitive episodes? |
| OOD closed loop | Fail2Drive | Does targeted information reduce decision regret/degradation under shift? |
| Closed loop | CARLA 2.1 / Bench2Drive v0.0.4 | Does sensing/compute allocation improve task outcomes under matched constraints? |

Compare fixed-rate sensing, uncertainty-threshold sensing, entropy/information-gain sensing, no-additional-sensing, decision-sensitive robust selection and an oracle query selector used only as an evaluator upper bound.

Match model-visible inputs, sensor privileges, inference/query latency, compute, budgets, scenario exposure, action sets and score definitions. Forecast error, query efficiency and closed-loop task outcomes remain distinct measures.

## 11. Acceptance and claim ceiling

A future implementation is acceptable only when:
- analytic golden cases are independently recomputed;
- all decision classifications and query outcomes are typed;
- high uncertainty with invariant action does not force sensing;
- small decision-critical uncertainty triggers query or abstention;
- stale/missing/late/unsupported outputs never reduce uncertainty;
- outcome blinding, permissions, causal delivery and deadlines are enforced;
- query budgets and costs are explicit;
- EVSI is unavailable unless its probabilities and selection assumptions are qualified;
- #10266 common-mode references are not counted as independent corroboration;
- exact replay binds every query, observation, action and output;
- compile, format, unit and exact-head CI evidence is recorded before any runtime claim.

Passing establishes only that the named query-selection policy improves the declared decision-information objective on the specified analytic/benchmark tasks. It does not prove physical safety, real-world sensing validity, low crash probability, human behavioral fidelity, roadworthiness, certification, deployment permission or SAE Level 4/5 capability.

## 12. Prospective targets (not results)

- At matched sensing/compute budget, target at least 20% reduction in unresolved action pairs or upper finite-set regret versus fixed-rate and entropy-only baselines.
- At matched decision coverage, reduce sensing/compute cost versus fixed-rate sensing.
- No increase in infeasible selected actions, deadline misses or evidence-incomplete false passes.
- 100% of observations retain source, freshness, applicability and typed-missingness state.
- 0 late, missing, stale or unsupported results counted as new information.
- 0 actuator authority granted by the query evaluator.
