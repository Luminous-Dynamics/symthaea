# Spark Planning Snapshot v1 — design / qualification contract

Status: staged architecture only. No product implementation. No scientific or action authority.

Tracks: #906

Depends conceptually on: #904, #885, #868, #857, #890, #892, SCI-002/003/004/005/009.

## Purpose

Make one experiment recommendation replayable and auditable without confusing a planning record with experiment authorization.

Current `SequenceStep` retains only the winner name, cost, cumulative cost, EIG-at-selection, and synthetic post-step entropy. That is not enough to reconstruct the decision.

## Non-equivalences

```text
recommendation text
    != planning state
    != planner execution
    != planning decision receipt
    != scientific disposition
    != experiment authorization
```

## Input snapshot

A future Spark-local `PlanningSnapshotV1` should bind, by value or stable reference, the exact state used for one planning decision:

```text
PlanningSnapshotV1 {
    schema_profile,
    belief_state,
    candidate_inventory,
    structural_validation,
    declared_target_scope,
    prediction_coverage,
    eligibility_frontier,
    observation_model_profile,
    likelihood_model_profile,
    objective_profile,
    budget_state,
    algorithm_profile,
    preference_and_tie_break_profile,
    minimum_information_thresholds,
    evidence_or_observation_history_ref,
    limitations,
}
```

Every field that can change a selection belongs either in the snapshot or behind an exact bound identity.

## Decision receipt

A separate `PlanningDecisionReceiptV1` should record the output derived from exactly one snapshot:

```text
PlanningDecisionReceiptV1 {
    planning_snapshot_ref,
    status,
    selected_candidate?,
    selected_coordinates?,
    alternatives,
    rollout_semantics,
    assumed_scenario?,
    limitations,
}
```

Possible status vocabulary includes:

- `ProposedNextExperiment`
- `NoEligibleExperiment`
- `PlanningInputInvalid`
- `PrerequisiteBlocked`
- `BudgetBlocked`
- `InsufficientPredictiveCoverage`
- `NoUsefulDiscrimination`
- `PlannerUnavailable`
- `ScenarioRolloutOnly`

An empty vector is not a sufficient failure ontology.

## Alternatives are evidence

A reproducible choice must retain relevant non-winners and why they were not selected. Example reason vocabulary:

- `InvalidPlanningInput`
- `IneligiblePrerequisite`
- `Unaffordable`
- `IncompleteFullScopeEIG`
- `BelowMinimumInformationThreshold`
- `DominatedUnderObjectiveProfile`
- `LowerPreferenceCoordinate`
- `LostDeterministicTieBreak`
- `CharacterizationOnlyForThisPlanner`

Do not retain only the winner.

## MAP scenario boundary

For #892-compatible scenario rollouts, bind each synthetic step explicitly:

```text
ScenarioAssumptionV1 {
    kind: CurrentMapHypothesis,
    assumed_hypothesis,
    synthetic_observation,
    pre_step_belief,
    post_step_synthetic_belief,
}
```

Synthetic observation lineage must never be serialized in a form indistinguishable from a real observation receipt.

## Receding-horizon preferred pilot

The first implementation should be one-step and non-authorizing:

```text
real current evidence
 -> immutable PlanningSnapshotV1
 -> one-step assessment of eligible candidates
 -> PlanningDecisionReceiptV1
 -> human/separate authorization boundary
```

After a real observation, create a new snapshot. Never mutate the old snapshot into the new plan.

## Identity boundary

Ordinary serde output is diagnostic serialization, not scientific content identity.

Future authoritative references should derive from SCI-002 canonical artifact identity. If replay depends on executable environment/toolchain, bind SCI-003 execution-capsule identity too.

Do not use `DefaultHasher`, Debug strings, map iteration order, or incidental JSON field order as authority-bearing identity.

## Required qualification cases

1. Same selected experiment under two different beliefs => different snapshots.
2. Same belief/candidates under two budgets => different snapshots/receipts.
3. Same winner under two observation-model profiles => distinguishable provenance.
4. Blocked high-EIG candidate remains present in alternatives with a prerequisite reason.
5. Invalid prediction candidate remains invalid, not silently transformed into EIG zero.
6. Exact ties retain the explicit preference/tie-break reason.
7. MAP rollout binds the assumed hypothesis and synthetic observation.
8. Replanning from real data creates a new snapshot with ancestry to the previous decision/observation.
9. Equal beliefs supplied in different constructor orders converge after #904.
10. Snapshot/receipt construction alone grants no experiment execution authority.

## Replay theorem

A qualified replay test should eventually establish only:

```text
same qualified planning snapshot
+ same planner algorithm/execution profile
=> same planning decision receipt
```

It does not establish that the recommendation is scientifically optimal or physically safe.
