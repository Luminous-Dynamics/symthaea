# Paired Intervention Contrast Evaluator v0.1

**Status:** Rust evaluator and 20 deterministic tests authored; exact-head Rust 1.100.0 qualification pending.  
**Research owner:** [#13720 — CMAQ-019](https://github.com/Luminous-Dynamics/symthaea/issues/13720).  
**Transfer context:** [#13719 — frozen mechanism transfer](https://github.com/Luminous-Dynamics/symthaea/pull/13719).  
**Statistical inference:** #8578.  
**Isolation and evidence custody:** #9437.

## Research question

When exactly one causal variable is deliberately changed, does the model predict the correct change in the world—and preserve the variables that should remain invariant?

The evaluator uses matched control/treatment branches with:
- one common, digest-bound pre-intervention source state;
- one common, digest-bound exogenous-forcing commitment;
- one frozen source model checkpoint;
- one evaluator-assigned target variable and a changed control/treatment value;
- separate control and treatment branch manifests;
- evaluator-only truth and candidate predictions.

The central metric is the *paired causal contrast*, not two unrelated prediction scores.

## Paired contrast

For outcome (Y), the evaluator calculates:

\[
\Delta_Y^{true}=Y_{treatment}^{true}-Y_{control}^{true}
\]

and:

\[
\Delta_Y^{model}=Y_{treatment}^{model}-Y_{control}^{model}.
\]

The absolute contrast error is:

\[
E_{\Delta}=\left|\Delta_Y^{model}-\Delta_Y^{true}\right|.
\]

Direction accuracy is measured only when the absolute true effect exceeds a preregistered resolution floor. Null/small effects are excluded from sign accuracy, not incorrectly counted as directional successes.

A paired predicted-effect interval is evaluated for coverage of the true contrast. Marginal branch predictions and the paired interval remain separate artifacts.

## Intervention families

The frozen scorecard requires eight distinct families:

1. Directional intervention.
2. Parameter change.
3. Structural change.
4. Temporal intervention.
5. Interaction intervention.
6. Null/invariance intervention.
7. Confounding-control intervention.
8. Regime-shift intervention.

These are separate family results, never pooled into one causal-understanding scalar.

## Invariance contract

The context declares the exact evaluator-assigned set of invariant variables. Every completed pair must supply both control/treatment truth and control/treatment predictions for exactly that set.

- If evaluator truth violates a declared invariant, the evaluator rejects the pair contract; the generator/profile is inconsistent.
- If true values are invariant but predicted values change beyond the preregistered tolerance, the pair records a false-change event.
- A missing, duplicate or unregistered invariant variable rejects the completed pair.
- Failed, timed-out, invalid and abstained pairs stay in the original pair denominator with explicit disposition and assigned penalty. They are not dropped or silently zero-filled.

The intervention target cannot itself be listed as invariant, and its control/treatment values must differ.

## Statistical unit and aggregation

The independent unit is the environment-generation cluster, not the branch, pair row, action, frame or query.

For each intervention family:
1. compute pair-level contrast errors;
2. average repeated pairs inside each independent cluster;
3. calculate the primary family contrast error as the equal-weighted mean across completed cluster means.

A task-pair-weighted error mean is retained as a diagnostic only. Family summaries preserve pair counts, completed/failed/timed-out/invalid/abstained counts, completed-cluster counts, sign accuracy, interval coverage, invariant observations and false changes.

A minimum number of independent completed clusters per family is preregistered and enforced. The Rust evaluator preserves the cluster deltas needed by #8578; it does not calculate confidence intervals or infer superiority.

## Resource/provenance contract

The certificate binds candidate/model/source-history/generator/intervention-design/evaluator/seed/split digests, pair/run/branch IDs, source-state and exogenous-forcing commitments, branch manifests, source checkpoints, intervention family profiles and both branch resource records.

Branch interactions, model queries, elapsed time and optional energy are summed across both control and treatment branches. An unexecuted branch must have no output or nonzero cost. If an energy ceiling is declared, every branch—including a not-executed branch—must have an explicit measurement.

The candidate process must never receive evaluator truth, counterfactual branch outcomes, hidden mechanism parameters or signing keys. Those controls are enforced by the trusted launcher and #9437, not by a Rust struct alone.

## Qualification fixtures

The 19 deterministic Rust tests cover:
- all eight intervention families;
- paired contrast error where marginal predictions could look plausible;
- direction accuracy and the effect-resolution floor;
- null interventions;
- invariant-variable false changes and evaluator-truth violations;
- missing and duplicate pair IDs;
- source-checkpoint mutation;
- branch/pair disposition consistency;
- failed and abstained pair retention and penalties;
- cluster/environment-family identity;
- paired interval ordering and coverage;
- frozen invariant-variable membership;
- repeated pairs within one cluster not increasing the independent-cluster count;
- both-branch energy measurement;
- unique branch IDs, actual target-value changes and target/invariant disjointness;
- total resource-budget enforcement.

## External benchmark positioning

- **What-If World (arXiv:2605.27589):** paired physical-change prompts on nuScenes/DROID; a later adapter feasibility study, subject to license and scorer review. Its vision-language score is not pooled with synthetic fixed-point causal-contrast units.
- **Causal-Discovery-Bench (arXiv:2611.09934):** sequential experiment design and resource-aware intervention comparator; use process metrics and matched budgets.
- **Active Causal Discovery Benchmark (arXiv:2609.31675):** skeleton/DAG recovery and hard-intervention efficiency as a distinct SCM track. Treat its early v0 baseline caveats explicitly.
- **CausalWorld:** later robotic parameter-intervention track. Simulated and physical trials remain separate.
- **WorldArena 2.0:** later visuo-tactile paired intervention / robotic execution track.

## Claim ceiling

A passing Rust evaluator establishes only the arithmetic, pair completeness, provenance binding and declared score semantics. It does not establish a Symthaea causal-understanding advantage, benchmark reproduction, SOTA, physical safety or general real-world transfer. Confirmatory claims require held-out intervention families, matched baselines, paired cluster-level inference and independent evidence custody. Blind runs remain blocked until #9437's OS/process isolation and signed receipt controls pass.
