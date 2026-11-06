# Query-conditioned connectome proofreading stopping benchmark V1

**Tracking:** #10697 — SCI-CONNECTOMICS-094
**Composes:** #10656/#10657/#10660, #10670/#10675, #10661/#10682, #10676, #9255
**Status:** protocol and synthetic known-answer cases; no biological release decision or empirical stopping advantage is claimed.

## 1. Objective

Separate two stopping questions:

- **Global reconstruction stabilization:** are selected global graph invariants approaching stable values under a declared edit-history and model-fit protocol?
- **Query-conditioned scientific stability:** do all still-admissible reconstruction worlds and observation models within a frozen scope support the same answer to one particular scientific question?

These are not interchangeable. Martinez et al. (bioRxiv v1, 4 July 2026) fit monotone asymptotic curves to global graph invariants from progressive proofreading snapshots. The zebrafish spinal-cord case uses 15 invariants at 11 checkpoints and forecasts six invariants converging around 71,000 edits, with a conservative upper estimate of 89,000. The authors explicitly state that global invariant proxies cannot determine progress toward a specific scientific question.

## 2. Paradigm shift: release per question, not one global 'done'

For each frozen scientific query:

1. Bind query specification, decision rule/loss, scope, required paths/cells, source revision and allowable observation models.
2. Bind an ordered sequence of immutable reconstruction snapshots and the exact edit-history intervals between them.
3. Fit/score global graph-invariant stabilization separately, preserving the exact metrics, checkpoints, model family, fit status, uncertainty and edit-strategy regime.
4. Compute query decision sets over every admissible reconstruction world in the declared scope.
5. Account for unadjudicated alternatives, all-list-missed uncertainty, region boundaries, measurement-model identifiability and late additions.
6. Emit a query state: `QUERY_STABLE_WITHIN_DECLARED_SCOPE`, `QUERY_UNSTABLE`, `QUERY_NOT_IDENTIFIABLE`, or `INSUFFICIENT_COVERAGE`.
7. Permit a query-specific stop only when the declared decision-loss/risk bound is satisfied and no admitted opposing world, unresolved adjudication or boundary/coverage gap invalidates it.
8. Keep global stop estimates and query-safe checkpoints as separate outputs. A later admissible query-changing alternative invalidates or supersedes the previous query certificate; it never silently rewrites history.

A query can be safe to answer while unrelated graph metrics still drift. A graph can be globally stable while a rare pathway or low-frequency alternative still changes one important scientific conclusion. The benchmark must test both directions.

## 3. Evidence semantics and fail-closed states

Keep separate:
- exact structural graph snapshot and source provenance;
- global graph-invariant trajectory and its asymptotic forecast;
- candidate/admissible reconstruction worlds and adjudication state;
- query decision per world and robust decision agreement;
- measurement/observation-model identifiability;
- population-frame or all-list-missed uncertainty;
- region-boundary completeness and margin;
- query-specific certificate version and invalidation history.

Do not infer that a global metric's convergence means synaptic completeness. Do not infer that a non-significant perturbation is a functional zero. Do not treat a sampled frame as exhaustive if all-list-missed sensitivity remains nonzero. Do not promote an exact reconstruction identification claim where only the query decision is stable.

Allowed terminal states:

| State | Meaning | Query stop allowed?
|---|---|---:|
| QUERY_STABLE_WITHIN_DECLARED_SCOPE | All admitted worlds agree on the decision and declared risk/coverage/identifiability gates pass | Yes, for this query and scope only
| QUERY_UNSTABLE | An admitted, unresolved or plausible world changes the decision, or risk bound is exceeded | No
| QUERY_NOT_IDENTIFIABLE | Distinct query decisions remain observationally equivalent under the declared measurement model | No
| INSUFFICIENT_COVERAGE | Required snapshots, boundary margins, source identities or adjudication evidence are missing | No

Global status is separate: `GLOBAL_CONVERGED`, `GLOBAL_NOT_CONVERGED`, `GLOBAL_REGIME_SHIFT`, or `GLOBAL_NOT_ESTIMABLE`.

## 4. Synthetic known-answer corpus

### Track A — rare-path false-safe-stop adversary

One query depends on a low-frequency path whose omission barely moves global graph invariants:

| Checkpoint | Cumulative edits | Global status | Admissible query decisions | Query status | Safe to stop?
|---|---:|---|---|---|---:|
| A0 | 10,000 | GLOBAL_NOT_CONVERGED | {true, false} | QUERY_UNSTABLE | No
| A1 | 50,000 | GLOBAL_CONVERGED | {true, false} | QUERY_UNSTABLE | No
| A2 | 60,000 | GLOBAL_CONVERGED | {true} | QUERY_STABLE_WITHIN_DECLARED_SCOPE | Yes

A global-only stopping baseline would stop at A1 and be wrong for this query. The query-conditioned policy must reject A1 and only certify A2. This is the primary false-safe-stop known answer.

### Track B — query-specific early-stop case

Another query is confined to a well-covered, already adjudicated circuit. Unrelated regions continue changing:

| Checkpoint | Cumulative edits | Global status | Admissible query decisions | Query status | Safe to stop?
|---|---:|---|---|---|---:|
| B0 | 5,000 | GLOBAL_NOT_CONVERGED | {false} | QUERY_STABLE_WITHIN_DECLARED_SCOPE | Yes
| B1 | 20,000 | GLOBAL_NOT_CONVERGED | {false} | QUERY_STABLE_WITHIN_DECLARED_SCOPE | Yes
| B2 | 50,000 | GLOBAL_CONVERGED | {false} | QUERY_STABLE_WITHIN_DECLARED_SCOPE | Yes

The query-conditioned policy may stop at B0 for this query only; it does not claim the whole reconstruction is globally complete.

### Required additional adversarial tracks

1. **Strategy regime shift:** proofreading switches from merge-focused to split-focused or automated edits; invalidate or refit global forecasts and do not extrapolate across the regime without evidence.
2. **Boundary miss:** cropped region omits incoming/outgoing edges; no query stop until declared margins and boundary-edge accounting are closed.
3. **Late alternative admission:** add a query-flipping world after a stop; the old certificate must be invalidated or superseded with explicit ancestry.
4. **Sampling miss:** sampled review finds no critical alternative but all-list-missed sensitivity is positive; keep the query certificate bounded to the sampled frame or refuse the population claim.
5. **Query-invariant ambiguity:** multiple graph worlds remain unresolved, but all admitted worlds agree on the query. The correct status is query stable, not exact graph identity.
6. **Observation confounding:** latent states differ but the measurement model makes their responses identical; return QUERY_NOT_IDENTIFIABLE.
7. **Insufficient power:** a response is non-significant but equivalence bounds are not met; do not treat it as no effect.

## 5. Metrics and baseline matrix

Report each independently:

- global invariant forecast error, calibration and interval coverage;
- remaining-edit / person-hour / GPU-hour forecast error and cost;
- query-specific false-safe-stop rate (primary safety metric);
- unnecessary-work / false-not-stable cost;
- earliest query-safe checkpoint and savings against global-only, fixed-budget and expert baselines;
- robust query decision agreement, expected/robust query regret and risk-bound coverage;
- recall of independently adjudicated query-critical alternatives;
- abstention, not-identifiable and insufficient-coverage rates;
- region-boundary leakage, regime-shift detection delay, and stale-certificate invalidation correctness.

Baselines under matched edit histories and evidence access:

1. Published global graph-invariant convergence (15 invariants / monotone asymptotic families / BIC model selection where exact reproduction supports it).
2. Fixed edit budget and expert stop date.
3. Local structural QA and task-/region-specific graph metrics.
4. Query-world robust decision without adaptive actions.
5. Query-conditioned adaptive decision-value-of-information policy (#10670/#10675).

Never compress false-safe-stop, cost, calibration and recall into one score. Report a Pareto frontier and confidence intervals clustered by independent query/dataset/animal as appropriate.

## 6. Aspirational promotion targets

- 0 false-safe-stop certificates on the held-out adversarial corpus;
- 100% refusal to certify while an admitted query-changing world remains compatible;
- ≥95% recall of independently adjudicated query-critical alternatives at the frozen reviewer-time budget;
- ≥20% lower total review/acquisition cost than the strongest frozen safe baseline on held-out queries, without increased false-safe-stop risk;
- calibrated global effort and query-risk intervals with subgroup and edit-regime reporting;
- exact invalidation/supersession when a late admitted alternative changes the query decision.

These are targets, not results. Synthetic correctness does not establish biological reliability or SOTA superiority.

## 7. Reproduction and custody requirements

Before numerical comparison, pin the exact Martinez et al. preprint/version, code and data repositories, progressive snapshot identities, edit-history cut points, graph extraction versions, invariant definitions, model families and fitting procedure. The published 71,000 / 89,000 edit estimates are not universal thresholds and must not be copied onto other datasets.

Every query certificate binds the exact snapshot, query spec, admitted-world set, alternative evidence/adjudication, all-list-missed bound, region-margin receipt, observation-model identities and evaluation code. Certificates are append-only; new evidence creates a new generation and preserves previous decisions.

## 8. Claim ceiling

The protocol can qualify a query decision only over its declared snapshots, admissible worlds, observation models, coverage and boundary assumptions. It cannot prove biological completeness or a globally complete hypothesis space. A query-specific release is not a whole-connectome completeness claim.