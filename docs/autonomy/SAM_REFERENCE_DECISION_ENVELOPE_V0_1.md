# SAM Reference-Set Decision Envelope v0.1

**Status:** research protocol / analytic fixtures only  
**Issue:** [SAM-REF-DECISION-001 #10318](https://github.com/Luminous-Dynamics/symthaea/issues/10318)  
**Owners composed:** Symthaea #10266, #10237, #10252, #10257; Symtropy #1173, #3387  
**Claim ceiling:** decision-ordering evidence conditional on the named reference portfolio, objective definitions, action set and scenario context.

## 1. Why this exists

SAM already has three related but distinct capabilities planned:

- reference qualification and common-mode dependency analysis (#10266);
- action-set counterfactual decision quality and regret (#10237);
- planner ranking stability across world-model profiles (#10252).

These do not answer the same question as **reference-set disagreement**. A qualified analytic oracle, independent implementation and context-qualified simulator can disagree. Their domains, measurands, uncertainty descriptions and common-mode roots may differ.

This protocol consumes reference qualifications and execution receipts. It does not select a universal truth model, rank references globally, create a simulator, or replace the existing decision evaluator.

The target output is an explicit decision envelope: which pairwise preferences are invariant across the admitted references, which reverse, which remain unresolved, and which cannot be evaluated.

## 2. Normative vocabulary

The terms **MUST**, **MUST NOT**, **SHOULD** and **MAY** describe this protocol's requirements.

A *reference source* is an implementation, analytic derivation, measurement process or simulator that produces an outcome for a declared measurand.

A *qualified reference* has a separate context-of-use / qualification record. A unique name, source repository, model ID or digest is not qualification.

A *reference family* is a declared common-mode dependency cluster. Family membership is not an independence proof.

A *measurand* is the quantity intended to be measured, including unit, frame, direction and operational definition.

A *decision envelope* is a set of per-measurand, per-reference comparison classifications. It is not a probability distribution over reality.

## 3. Input matrix and identity contract

The atomic input is one result cell for one:

`scenario × action × reference × measurand × horizon`.

Every cell MUST bind:

- protocol and schema version;
- scenario and scenario-family IDs;
- complete finite action-set ID and exact action ID;
- exact control payload / action-history digest;
- initial-state, environment, actor-policy and exogenous-stream identities;
- decision tick, forecast horizon and clock profile;
- reference ID and immutable source/build artifact identity;
- reference qualification record and context-of-use scope;
- common-mode root / reference-family manifest;
- measurand ID, unit, coordinate frame, favorable direction and comparison margin;
- value or source-defined uncertainty interval and uncertainty semantics;
- feasibility disposition and typed execution outcome;
- exact execution receipt and output-artifact identity.

The action-set generator, planner selection and reference evaluator remain separate. The planner MUST commit its selection before evaluator-only outcomes become available. The candidate set MUST be complete for any complete-best-action or complete-regret statement.

### 3.1 Applicability is per measurand

Each reference/measurand/scenario combination has an explicit applicability disposition:

- `Applicable`
- `Inapplicable(reason, scope)`
- `ApplicabilityUnknown(reason)`

An inapplicable reference is not a failed reference. Its scope and reason remain visible and it is excluded only from the measurand/context it does not cover. Unknown applicability blocks a complete portfolio claim. A reference applicable to braking distance may be inapplicable to pedestrian intent; applicability must not be inherited across measurands.

### 3.2 Outcomes are never silently collapsed

Execution outcomes preserve at least:

- `Observed`
- `Missing`
- `Censored`
- `TimedOut`
- `Unsupported`
- `ExecutionFailed`
- `ReferenceInapplicable`

Only `Observed` values enter a numeric comparison. A missing or failed applicable reference cell makes the relevant comparison `Incomplete`; it is never a zero residual, a clean event, or agreement.

## 4. Interval-aware pairwise ordering

For a measurand with a favorable direction, normalize it so that **larger is better**:

`z = value` for higher-is-better and `z = -value` for lower-is-better.

Do not normalize away units or compare different measurands. The normalized values remain tied to the original metric identity.

For action A and B under applicable reference r, let the source-supported normalized intervals be:

`A_r = [L_A, U_A]` and `B_r = [L_B, U_B]`.

The interval for the difference A − B is:

`D_r(A,B) = [L_A - U_B, U_A - L_B]`.

For a preregistered practical-equivalence margin `δ >= 0`:

- `A PortfolioDominates B` only when `lower(D_r) > δ` for **every** applicable admitted reference r.
- `B PortfolioDominates A` only when `upper(D_r) < -δ` for every applicable admitted reference r.
- `PortfolioEquivalent` only when every applicable `D_r` lies wholly inside `[-δ, +δ]`.
- `ReferenceSensitive` when at least one applicable reference supports `lower(D_r) > δ` and a different applicable reference supports `upper(D_r) < -δ`.
- `Unresolved` when uncertainty intervals overlap the decision margin or the portfolio does not establish one of the above relations.
- `Incomplete` when an expected applicable cell, applicability declaration, action or execution receipt is missing or invalid.

The comparison MUST be evaluated symmetrically: `D_r(B,A) = -D_r(A,B)`. Input order MUST NOT affect the result.

**Important:** these are portfolio-conditional classifications. They do not imply that all references are independent, that the portfolio spans reality, or that an action is safe. A singleton reference portfolio can support only a result conditional on that one reference.

### 4.1 Point estimates are not zero-width uncertainty intervals

A source may publish a point estimate without a defensible uncertainty interval. That is not equivalent to known exactness.

Each numeric result must carry an uncertainty state:

- `SourceBoundedInterval` — interval semantics and derivation are supplied;
- `PointEstimateOnly` — only a point estimate is supplied;
- `UncertaintyUnavailable` — the source cannot quantify uncertainty.

When any required applicable reference is point-only, the protocol MAY report a separate `PointwisePreference` or `PointwiseReferenceSensitive` diagnostic from the reported point values. It MUST NOT label that result interval-robust or imply zero uncertainty. An interval-robust portfolio classification requires a source-supported interval for every reference cell used in that classification.

Intervals must retain their type (e.g. hard bound, measurement interval, model uncertainty interval, statistical interval), method, coverage/confidence interpretation when applicable, and source. Intervals of unlike semantics MUST NOT be pooled or averaged. A statistical confidence interval is not a hard physical bound.

## 5. Multi-objective comparison without hidden scalarization

Hard feasibility constraints and preference measures are separate.

For each declared objective, report its own pairwise matrix, unit, direction, margin, reference applicability, uncertainty state and raw denominator. Examples include a scenario-defined conflict margin, progress, rule compliance, comfort, energy and deadline performance. The exact operational definition must come from the scenario profile; no generic proxy silently becomes a safety outcome.

A robust Pareto-dominance relation MAY be reported only when:
1. all required objectives are present and comparable;
2. no required objective is incomplete or inapplicable without an explicit protocol disposition;
3. the action is no worse on every objective under the declared relation;
4. it is strictly better on at least one objective;
5. hard constraints are satisfied under the separately declared feasibility rule.

Do not assign weights, rank objectives, or create a composite score unless a separate preregistered utility specification explicitly requires it. Such a scalar result is a different estimand and must be reported alongside, not instead of, the per-objective envelope.

A non-dominated action is not necessarily optimal, feasible, safe, or suitable for execution.

## 6. Scalar regret when a scalar objective is actually justified

Scalar regret is allowed only when all references use a common predeclared utility definition, common units/direction, common feasibility semantics and matched scenario context. Otherwise, report the per-objective partial order and leave scalar regret undefined.

For one reference r, normalized higher-is-better utility intervals `U_r(a) = [L_a, H_a]`, and a complete reference-feasible action set:

`regret_r(a) = max(0, max_b L_b - H_a) .. max(0, max_b H_b - L_a)`.

This is an interval envelope for the finite candidate set, not a global regret bound over unenumerated actions. Report:
- the action-set identity and completeness assertion;
- per-reference regret interval;
- per-reference reference-specific best action(s);
- maximum upper regret across the named applicable reference set;
- reference-family/common-mode disclosure;
- uncertainty and incomplete denominators.

The maximum over named references is a sensitivity bound for that finite portfolio, not a population probability, confidence bound, or assurance that the true system lies inside the set. Do not average regret across references or treat correlated references as independent samples.

## 7. Common-mode dependence and source counts

Reuse the dependency graph from #10266.

Retain dependencies across:
- equations and governing assumptions;
- source code, shared libraries and numerical methods;
- parameters and calibration;
- datasets, annotations and label generation;
- sensors, clocks and coordinate transforms;
- preprocessing and postprocessing;
- evaluation code and human-review lineage.

Do not infer independence from different model names, teams, repositories, vendors or digests. A shared dependency cluster is not an independent vote. Conversely, correlated sources are not discarded: each can still expose sensitivity, but the dependence is disclosed and no corroboration count is inflated.

Report source count and distinct declared dependency-cluster count separately. Neither is a sample size drawn from reality.

Leave-one-reference and leave-one-family sensitivity diagnostics MAY be computed. They are descriptive influence analyses and do not prove independence or truth.

## 8. Pairing, replay and blinding

For a matched counterfactual, hold fixed the decision-time initial state, scenario revision, observation availability, actor policy, action-history prefix, exogenous disturbance stream, clock/timestep and requested horizon where each reference supports the same semantics.

The exact action command payload must be identical across references. Any unsupported mapping, approximation or altered semantics produces an explicit typed result, not a silent match.

A changed exogenous stream, observation window, actor policy, initial state or horizon creates a different comparison cell. Do not call it paired.

The reference evaluator must be outcome-blinded until the planner commits its selected action. The selection receipt, action-set digest and evaluator reveal boundary are retained.

Same-runtime bitwise replay and cross-runtime semantic replay remain separate claims, as defined by Symtropy #3387.

## 9. Falsification controls

The companion JSON fixture contains analytic golden cases for:

1. unanimous interval-supported preference;
2. opposite strict reference preferences;
3. overlapping intervals and an unresolved comparison;
4. metric-specific inapplicability;
5. point-estimate-only sources;
6. an expected reference cell that times out;
7. shared common-mode roots;
8. a reference interval widened without changing its point estimate;
9. incomplete action sets;
10. order-invariant canonical input processing.

Runtime qualification must additionally test duplicate source/receipt IDs, changed action payloads, context mismatch, unit/frame/direction mismatch, protocol mutation, digest tampering, missing/censored/unsupported outputs and objective-weight changes after evaluation.

The evaluator's own tests MUST include positive and negative controls. A valid JSON fixture is not a runtime pass.

## 10. Benchmark and research ladder

| Stage | Dataset / system | What it can test | What it cannot establish alone |
|---|---|---|---|
| Analytic | Closed-form crossing/stopping fixtures | Correct ordering, intervals, regret arithmetic | Vehicle physics or human behavior |
| Independent reference | Separately implemented solver / admitted reference | Common-mode disclosure, context-specific disagreement | That references span reality |
| Interactive execution | Symtropy #3387 + #3471 actor policies | Causal observation mediation and replay | Human behavior validity |
| OOD closed loop | Fail2Drive paired routes | Shift-sensitive decision degradation | Real-world crash probability |
| Closed-loop route | CARLA Leaderboard, Bench2Drive | Route completion and scenario/infraction behavior | Comparable performance without matched versions/protocols |
| Actor forecasting | Argoverse 2 Motion, Waymo Open Motion | Forecast error and set coverage | Closed-loop decision quality |
| Black-box decision evaluation | NIST four-tier measurement programme | Surrogate measures through counterfactual choice and recurring diagnosis | NIST endorsement or conformance from conceptual alignment |

Freeze each dataset release, license, split, scoring implementation, sensor privilege, inference compute, training-data exposure and simulator version. Report separate tracks for matched-compute and best-attainable-compute comparisons. Never compare headline metrics across different benchmark tasks as if they shared a scale.

ASAM OpenSCENARIO XML 1.4.0 and DSL 2.2.0 are the current 2026 planning targets for scenario interchange. The adapter must maintain a versioned conformance corpus and classify unsupported or approximated semantics explicitly. Version compatibility is not lossless semantic equivalence.

## 11. Acceptance / claim ceiling

A future implementation is acceptable only when:
- golden arithmetic fixtures match an independently recomputed implementation;
- each outcome distinguishes portfolio dominance, equivalence, reference sensitivity, unresolved, pointwise-only and incomplete;
- uncertainty semantics and applicability are preserved per cell;
- common-mode clusters do not become independent votes;
- multi-objective results are not silently scalarized;
- scalar regret is restricted to a common declared utility and complete finite action set;
- every classification is recomputable from canonical retained inputs;
- adversarial controls fail closed;
- exact-head compilation, tests, format checks and workflow evidence are recorded separately.

Passing establishes only a comparison result conditional on the named and qualified references, contexts, interval semantics, metric definitions, candidate set and scenarios. It does not establish physical safety, real-world generalization, collision-risk reduction, roadworthiness, certification, deployment permission, SAE Level 4/5 capability, or global optimality.

## 12. Prospective targets (not results)

- 100% of expected action/reference/measurand cells have explicit applicability and outcome dispositions.
- 0 missing, failed, censored, timed-out or unsupported applicable cells silently count as agreement.
- 0 point estimates are represented as zero-width uncertainty bounds.
- 0 common-mode source families are treated as independent votes.
- 0 hidden objective scalarizations.
- 100% of emitted pairwise classifications independently recomputable from retained cells.
- 100% of incomplete action/reference matrices suppress complete-best and complete-regret claims.
