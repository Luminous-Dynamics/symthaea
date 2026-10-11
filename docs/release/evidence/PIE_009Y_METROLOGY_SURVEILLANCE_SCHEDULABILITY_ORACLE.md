# PIE-009Y — Independent Metrology Surveillance Schedulability Oracle

## Purpose

This reference asks whether continuous critical metrology surveillance is schedulable at all under finite comparison capacity, and what the minimum constant comparison capacity is for a bounded synthetic campaign.

It complements PIE-009X, which evaluates an explicit schedule.

## Core invariant

For each opening step, every active critical reference must already have a fresh, failure-domain-independent three-reference surveillance witness.

Comparison work performed during step `N` becomes visible only at opening step `N+1`.

The exact finite search enumerates candidate comparison subsets allowed by the capacity profile and accepts a campaign only if critical coverage is preserved at every opening boundary.

## Minimum constant capacity

The oracle tests constant capacities from zero upward and returns the first capacity with a feasible schedule.

This is an exact finite-horizon result for the synthetic fixture, not a claim about large production networks.

No hidden weighted score is used.

## Planned outages

Capacity is represented as an explicit per-step profile.

This allows PIE to distinguish:

- an outage that can be absorbed by pre/post comparison capacity;
- an outage that causes unavoidable evidence staleness;
- a topology that is unschedulable regardless of raw comparison capacity.

## Independent execution fixture

The final candidate self-test was executed locally with Python 3 on 2026-09-13 and returned `ok`.

The fixture demonstrates:

1. three critical references with maximum evidence age 2 over seven steps require exactly one constant comparison slot per step;
2. zero constant comparison capacity is infeasible;
3. with maximum evidence age 0, all three pairwise comparisons must be refreshed every step, so minimum constant capacity is exactly 3;
4. capacity profile `[2,1,0,2,1,0,2]` is feasible and uses pre-outage batching;
5. profile `[1,1,0,1,1,1,1]` is infeasible;
6. shared comparison common mode makes the topology `Unschedulable` even with maximal nominal slots;
7. adding an unmonitored noncritical reference does not increase the critical minimum;
8. malformed capacity profiles fail closed.

## Interpretation

The returned minimum is a structural service-capacity requirement for the declared synthetic topology, freshness bound, initial evidence state, and campaign horizon.

It is useful for exposing statements such as:

> Three independent primary standards are not enough; this architecture also needs at least one sustained comparison slot per campaign step.

or:

> This comparison topology cannot sustain the required surveillance even if every comparison channel is run every step, because the channels share an unacceptable common-mode failure domain.

## Non-claims

This reference is not a production-scale operations scheduler, stochastic reliability optimizer, workforce/economic model, physical comparison model, cyber model, or hardware-control system.
