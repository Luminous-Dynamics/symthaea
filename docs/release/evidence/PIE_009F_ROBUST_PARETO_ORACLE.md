# PIE-009F robust architecture-selection oracle evidence

Date: 2026-09-12

## Scope

This evidence note records the independent synthetic reference semantics in `scripts/pie-009f-robust-pareto-oracle.py`.

The oracle consumes already-declared scenario outcomes. It does not generate failure probabilities or reliability predictions.

## Independent execution

The final candidate self-test was executed locally with Python 3 on 2026-09-12 and returned:

`ok`

No Symthaea module is imported by the oracle.

## Decision discipline

The reference uses a two-stage rule:

1. hard survival constraints first;
2. Pareto comparison only among architectures that survive every mandatory scenario.

An architecture that misses an essential-service floor or mandatory recovery requirement is `NotRobust`; extra production, lower mass, or another objective cannot compensate for that failure.

## Robust vector

Among surviving architectures the oracle preserves these objectives separately:

- worst essential-service floor — maximize;
- worst cumulative service deficit — minimize;
- worst recovery time — minimize;
- seed mass — minimize;
- reserve mass — minimize;
- imported blocker mass — minimize;
- productive closure — maximize.

No weighted scalar score is introduced.

## Executed synthetic fixtures

The self-test includes:

- a low-reserve bulk-output architecture that fails mandatory power/common-mode scenarios and is `NotRobust`;
- a reserve-heavy architecture that survives and remains Pareto-nondominated;
- a diversity-heavy architecture that survives common-mode loss and remains Pareto-nondominated;
- a heavier/worse architecture that survives but is `RobustDominated`;
- a monotonicity check that adding mandatory scenarios cannot turn a previously non-robust architecture robust;
- a monotonicity check that stricter survival thresholds cannot enlarge the robust feasible set;
- fail-closed validation when any architecture/scenario result is missing.

## Classification

The reference emits exactly three structural classes:

- `NotRobust`;
- `RobustDominated`;
- `RobustPareto`.

These are not probabilities or mission recommendations.

## Important limitations

Scenario results are synthetic evidence inputs. The oracle does not decide which shocks should be mandatory, assign event likelihoods, estimate actual lunar/Mars failure rates, model economics, or replace the dynamic campaign/resilience simulators. A later production layer should consume evidence-bounded scenario results from PIE-009E and campaign models.

## Promotion boundary

The intended promotion path is:

`shock campaign evidence -> mandatory survival gates -> robust Pareto frontier -> decision sensitivity / experiment priority -> evidence-bounded Moon/Mars architecture comparison`

Tracks #1793, #1788, #1752, #1710, #1713, #1647 and master #1604.
