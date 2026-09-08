# Spark Scenario Rollout Semantics v1 — staged qualification design

Status: **queue-neutral design artifact only**

Base: `main@2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`

Tracking issue: #892

This file is not an implementation patch and carries no planner qualification or experiment authority.

## Existing computation

Spark's current `greedy_sequence()` repeatedly:

```text
1. ranks remaining affordable designs by current model-relative EIG/$
2. selects the best design
3. chooses the current MAP hypothesis
4. synthesizes the observation that MAP hypothesis would produce
5. updates the belief on that synthetic observation
6. repeats
```

The implementation documents this approximation and says to re-derive after real observations.

That honesty should be preserved.

## Core semantic distinction

```text
ScenarioConditionalRollout
    !=
ObservationContingentPolicy
    !=
ExpectedOptimalPolicy
    !=
RobustScientificProgram
```

The current object is closest to:

```text
ScenarioConditionalRollout {
    assumed_world = CurrentMapHypothesis,
    synthetic_observation_history,
    selected_steps,
}
```

## Why a type split matters

A one-dimensional sequence cannot represent a branch such as:

```text
Experiment A
  -> outcome O1 -> next best B
  -> outcome O2 -> next best C
```

A current-MAP rollout chooses one of those paths before the observation exists.

That is useful scenario planning, but the unchosen branch remains scientifically possible.

## Compatibility-safe v1 direction

Preserve the existing algorithm numerically and make semantics explicit first.

Conceptual additive record:

```text
ScenarioKindV1 {
    CurrentMapHypothesis,
    ExplicitHypothesis(...),
    ExplicitSyntheticOutcomeTrace(...),
}

ScenarioConditionalRolloutV1 {
    planning_snapshot_ref,
    scenario_kind,
    assumed_outcome_refs,
    steps,
    budget,
    model_identity,
    limitations,
}
```

The first tranche should **not** claim to implement an adaptive policy tree.

## Preferred near-term operational planner

For real science, a strong default can remain simple receding-horizon planning:

```text
current evidence snapshot
  -> compute eligible frontier
  -> choose/propose one next experiment
  -> execute only after separate authorization
  -> admit real observation
  -> freeze new planning snapshot
  -> recompute
```

This avoids overcommitting to simulated future observations.

The MAP rollout can remain useful for:

```text
budget scenarios
capacity planning
what-if analysis
sensitivity to assumed world
```

## Future adaptive-policy object

Only a later implementation that explicitly retains observation branches should use semantics like:

```text
AdaptiveExperimentPolicyV1 {
    planning_snapshot_ref,
    observation_model_ref,
    policy_tree_or_policy_ref,
    branch_probabilities_or_bounds,
    objective_profile,
    resource_constraints,
    horizon,
}
```

Where branch probabilities are unavailable or uncalibrated, a robust/minimax/set-valued policy may be more honest than invented probabilities.

## Report-language rule

Current reports should eventually distinguish:

```text
MAP-scenario rollout
```

from:

```text
recommended next eligible experiment
```

and from:

```text
adaptive policy
```

A scenario rollout may be shown as a planning aid but should not be presented as if later steps are unconditional recommendations.

## Required future negative controls

### A. Branch-dependent second action

Create two hypotheses and three experiments such that:

```text
first choice = A under current belief
if A -> O1, B is best
if A -> O2, C is best
```

Current MAP rollout must demonstrate it preserves only one synthetic branch.

Post-semantics repair: the rollout record must say which branch/world was assumed.

### B. MAP switch sensitivity

Use nearly tied priors such that a small prior change switches MAP H1 -> H2 while one-step EIG for A remains similar.

The resulting multi-step rollout may change substantially.

The record must make this scenario sensitivity visible.

### C. Real observation invalidates synthetic continuation

After generating a rollout under O1, provide real O2.

A receding-horizon planner must create a new planning snapshot and recompute; it may not continue the stale synthetic O1 branch as if observed.

### D. No policy laundering

A serialization/display round-trip of `ScenarioConditionalRolloutV1` must not deserialize/convert into a type named or authorized as `AdaptiveExperimentPolicyV1` without an independently qualified constructor.

## Relationship to other Spark hardening

```text
planning input validation                 #885 staged contract
  -> target / prediction scope            #885
  -> current eligibility frontier         #890
  -> observation-profile discrimination   #857
  -> predictive coverage / likelihood     #868
  -> one-step SCI-009 planning coordinates
  -> scenario rollout OR later policy
```

Do not solve multi-step planning before the observation model is coherent.

## Planner meta-science

Later frozen benchmarks can compare:

```text
one-step receding-horizon
current-MAP rollout
explicit-hypothesis scenario rollouts
posterior-sampled rollouts
expected policy tree
robust/minimax policy
```

Measures may include realized scientific information, falsifier resolution, cost, duration, invalid/blocked selections, regret and robustness.

No benchmark result transfers universal planner superiority.

## Deliberate non-claims

This contract does not claim:

- current MAP rollouts are invalid;
- policy trees are always superior;
- Spark priors/likelihoods are calibrated;
- expected information gain is globally optimal;
- any planned physical experiment is safe or authorized.

It defines only the semantic boundary required to keep a simulated future from masquerading as an observed or universally recommended future.