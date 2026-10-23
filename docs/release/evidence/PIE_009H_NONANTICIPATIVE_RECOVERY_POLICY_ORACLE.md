# PIE-009H non-anticipative recovery-policy oracle evidence

Date: 2026-09-12

## Scope

This evidence note records the independent synthetic reference semantics in `scripts/pie-009h-nonanticipative-recovery-policy-oracle.py`.

The purpose is to prevent a recovery planner from using information about future failures that would not yet be observable in a real campaign.

## Independent execution

The final candidate self-test was executed locally with Python 3 on 2026-09-12 and returned:

`ok`

The oracle imports no Symthaea module.

## Core non-anticipation rule

For deterministic policies, if two campaign histories are observationally indistinguishable through step `t`, their policy actions must be identical through step `t`.

Equivalently, the policy must satisfy:

`action_t = policy(observation_history_through_t)`

and must not depend on the scenario ID, future shock schedule, or unrevealed failure state.

The simulator enforces this structurally by passing only an `Observation` object into `RecoveryPolicy.decide(...)`; the policy never receives a `ShockScenario`.

## Reference semantics

- failures occur in physical state independently of observation;
- each failure may have an explicit nonnegative detection delay;
- a failure cannot be repaired before it becomes visible to the policy;
- repairs consume real spare inventory;
- a repair started in step `N` restores equipment only at the next step boundary;
- service priority is explicit and must contain each service exactly once;
- reserve use is explicit and finite;
- policy comparison remains a vector rather than a hidden weighted score;
- malformed scenario references and malformed policy actions fail closed.

## Executed synthetic fixtures

The self-test demonstrates:

1. two scenarios with identical observations through step 2 but different future failures produce identical actions through step 2;
2. renaming a scenario does not change observations or actions;
3. a one-step detection delay prevents same-step repair of a newly failed generator;
4. a repair begins only after observation and becomes effective at the following step boundary;
5. distinct deterministic policies may choose different actions from the same observation, but each remains future-blind;
6. increasing reserve energy cannot reduce the essential-service floor in the reference fixture;
7. adding a later unseen failure cannot alter earlier actions;
8. incomplete service-priority actions fail closed;
9. unknown equipment references fail closed;
10. scenario-set policy comparison exposes worst-case service/deficit/reserve/repair metrics without scalar weighting.

## Important limitations

This is not an optimal-control solver, probabilistic risk assessment, partially observable Markov model, reliability prediction, economics model, or autonomous hardware controller. Capacities, demands, repair times, detection delays, spare counts, and shock events are synthetic fixtures.

The oracle models delayed observation but not stochastic sensor error, communication topology, operator authority, command latency, multi-agent coordination, or model-predictive optimization. Those belong in later tranches after the information boundary is frozen.

## Promotion boundary

The intended path is:

`independent non-anticipation oracle -> production policy interface -> cross-check with PIE-009E shock campaigns -> robust policy comparison -> evidence-bounded Moon/Mars recovery strategies`

A production recovery optimizer should be rejected if it can produce different actions before two futures become observationally distinguishable.
