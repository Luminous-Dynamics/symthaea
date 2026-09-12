# PIE-009E resilience / shock oracle evidence

Date: 2026-09-12

## Scope

This evidence note records the independent synthetic reference semantics in `scripts/pie-009e-resilience-shock-oracle.py`.

The oracle is intentionally deterministic and non-probabilistic. It does not claim lunar or Martian failure rates, reliability, mission safety, economics, or plant qualification.

## Independent execution

The final candidate self-test was executed locally with Python 3 on 2026-09-12 and returned:

`ok`

No Symthaea module is imported by the oracle.

## Reference semantics

The oracle freezes the following rules:

- essential and nonessential services remain separate;
- service priority under shared power scarcity is explicit and unique;
- reserve energy is finite and single-spend;
- common-mode groups can fail multiple nominally redundant machines at once;
- site/resource disruption scales only the declared dependent service;
- scheduled spare imports arrive only while transport is available;
- a repair consumes a real opening spare and cannot borrow a future or missed import;
- repair completion occurs at the step boundary and cannot retroactively restore service in the failure step;
- recovery is judged by delivered service, not merely hardware state;
- a scenario-set summary reports worst-case structural outcomes without assigning probabilities or a hidden weighted score.

## Executed synthetic fixtures

The self-test demonstrates:

1. a nominal campaign supplies all declared services;
2. a finite reserve can preserve essential life-support/water minima while nonessential industry sheds;
3. reducing reserve cannot improve the conservative essential-service floor;
4. a common-mode water-system failure defeats two otherwise redundant units;
5. repair consumes a spare and only increases usable capacity in the following step;
6. transport loss blocks a scheduled spare import, so the repair plan fails instead of borrowing future inventory;
7. a site interruption can suppress industry while essential services remain available;
8. changing the explicit service-priority order visibly changes scarcity allocation;
9. duplicate service priorities fail closed;
10. multi-scenario summary exposes the worst service floor and earliest critical shortfall without probability weighting.

## Metrics

The reference output exposes at least:

- essential-service floor ratio;
- first critical-shortfall step;
- cumulative nominal-service deficit;
- cumulative nonessential-service deficit;
- reserve energy used;
- missed import units;
- failed/repaired machine state;
- first later step that restores nominal service after a critical shortfall;
- worst-case scenario-set summary.

## Important limitations

This is not a reliability model. Failure events are inputs, not stochastic predictions. Repair times and capacities are synthetic fixtures. Power is represented as one shared scalar service budget. Resource interruption is represented as a service-capacity factor rather than a full material-flow model. A later production campaign should compose the already-separate PIE mass, utility, throughput, lifecycle, dependency, and dynamic-bootstrap semantics rather than enlarging this oracle into a second production simulator.

## Promotion boundary

The intended promotion path is:

`independent resilience oracle -> exact production semantics -> cross-check against PIE-009D dynamic campaign -> synthetic compound-shock campaign -> evidence-bounded Moon/Mars scenarios`

Real planetary shock campaigns must retain frozen evidence roots and must not infer failure probabilities from this oracle.

Tracks #1788, #1752, #1647 and master #1604.
