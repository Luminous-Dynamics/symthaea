# MACE paired ITT estimator — executable contract v0.1

## Assignment record

Every row represents one matched task × seed pair and contains:
- unique pair_id, experiment_id, task_id, task_revision, domain, and seed;
- one 64-character SHA-256 stratum_digest that binds model/harness/prompt/tool-policy/permissions/environment/evaluator configuration;
- arm_order (with_skill_first or without_skill_first);
- distinct run IDs for both arms;
- task verdict, mandatory-control verdict, unauthorized-effect verdict and treatment-arm lifecycle diagnostics for both runs.

All records in one invocation must use the same experiment, configuration stratum, task revision per task and task domain per task. The estimator rejects duplicate pair IDs, run IDs and task/seed pairs. Stage flags must respect retrieved → selected → loaded → invoked → executed. The no-skill arm cannot claim the treatment skill's lifecycle.

## Endpoints

| Endpoint | Positive state | Negative state | Indeterminate state |
|---|---|---|---|
| Task success | pass | fail | indeterminate |
| Control passage | pass | not_passed | indeterminate |
| Clean success | task pass AND controls pass | task fail OR controls not_passed | all other combinations not resolved by a known failure |
| Unauthorized-effect incidence | present | absent | indeterminate |

For clean success, a known failure on either input fixes the outcome to zero. This avoids treating an indeterminate task as a clean success when mandatory controls are known not to have passed.

## Pair and task aggregation

For each binary endpoint, an arm outcome is represented by lower/upper/known values. For a paired effect (treated minus control):
- lower = treated lower − control upper;
- upper = treated upper − control lower.

Repeated seeds are averaged within each task. Task clusters are then equally weighted. The four endpoints are computed and reported independently.

## Missingness

Unknown outcomes are [0,1], not silently omitted. Lower and upper partial-identification bounds are reported. The assignment-roster option checks that the pair IDs exactly match a supplied roster; it does not authenticate that roster or certify randomization.

## Uncertainty

The percentile bootstrap resamples task clusters, preserving their within-task repeated seeds and aggregated endpoint bounds. It resamples the lower and upper identification endpoints separately. With one task cluster, confidence intervals are not emitted. The number of bootstrap replicates and seed are included in the report.

## Interpretation boundaries

- A higher unauthorized_effect_incidence effect means the treatment arm caused more unauthorized effects and is therefore worse.
- A positive control-passage effect means controls passed more often with the treatment.
- The estimator consumes adjudicated outcomes; it does not provide the oracle, runtime observer, policy enforcement or effect receipt.
- assignment_roster_matched=true means pair IDs matched a supplied list. It does not prove manifest authenticity, timing or randomized allocation.
- A stratum_digest is a content identifier, not a provenance signature.
- Results are causal only to the extent that the upstream randomization, matching, isolation and adjudication assumptions hold.
