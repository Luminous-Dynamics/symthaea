# MACE paired ITT estimator — protocol notes

This protocol estimates a randomized agent-system intervention (such as skill availability) without conditioning the primary estimate on post-treatment retrieval, selection, loading, invocation or execution.

## Assignment contract

Freeze one assignment row for every task × seed pair before execution. Include pair ID, task ID/revision, domain, model/harness/prompt/policy/environment/evaluator stratum, matched seed/block ID and randomized arm order. Each estimator invocation accepts one fixed configuration stratum. Isolate treatment and control arms from shared mutable memory, caches, files, credentials and external state.

The seed is a matched-block label; it does not prove that the two arms consumed identical internal random streams. Publish the canonical assignment digest before execution, while recognizing that a digest alone does not prove preregistration timing or source authenticity.

## Outcome and estimand

Record each arm's task verdict (pass/fail/indeterminate), control gate (pass/not_passed/unknown), unauthorized-effect status (present/absent/unknown) and optional stage diagnostic. Missing rows or outcomes remain unknown in the assigned denominator.

Primary estimand: task-paired intention-to-treat difference, treatment minus control, across every randomized assignment. For each endpoint, first average paired seed effects within task, then equally weight task clusters. Do not filter by whether a skill was retrieved or invoked.

Report four endpoints separately: task success, control-gate passage, clean success (task pass AND controls pass), and unauthorized-effect incidence. Never combine them into one weighted score.

## Missingness and uncertainty

If an assigned arm outcome is unknown, contribute [0,1] to that arm's binary endpoint, producing a [-1,+1] pair effect bound. Withhold a point estimate when any pair is unknown for the endpoint; do not replace it with a complete-case estimate. A known task failure or known control non-pass makes clean success zero even if the other axis is unknown.

Bootstrap task clusters, preserving all repeated seeds within each selected task. With incomplete endpoints, resample the lower and upper identification endpoints separately and report a conservative confidence region, not a conventional point-estimate confidence interval. Do not emit a task-cluster interval with fewer than two task clusters.

## Diagnostics

Retrieval, selection, resolution, loading, invocation and execution are post-treatment mechanism diagnostics. Publish their counts, but do not condition the ITT estimate on them. Stage fields supplied by a harness are not independent proof of hook completeness.

## Claim boundaries

The estimator does not randomize assignments, validate task oracles, authenticate traces, prove runtime-hook coverage, establish external effect truth or guarantee causal identification. Causal interpretation requires valid randomization, no cross-arm interference, matched frozen controls and independently adjudicated outcomes. Canonical hashes bind content, not authenticity.

See issue #11763 for the implementation and acceptance tests, #11760 for preregistered repeated-run assurance evaluation, and #11772 for judge/oracle validation.