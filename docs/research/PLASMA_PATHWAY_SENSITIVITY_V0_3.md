# Plasma pathway Monte Carlo stability benchmark v0.3

**Status:** Synthetic-only reproducibility and convergence diagnostic.  
**Parent:** [PLASMA-PATHWAY-DECISION-V02 #12542](https://github.com/Luminous-Dynamics/symthaea/issues/12542).  
**Implementation:** `xtask/src/plasma_pathway_decision.rs`, function `evaluate_decision_sensitivity_json`.  
**Fixture:** `xtask/plasma-audit-harness/fixtures/pathway-sensitivity-v0.json`.

## Research question

Does the v0.2 decision output remain stable when the same declared input distributions are sampled with different seeds and larger Monte Carlo budgets?

This tests numerical sampling sensitivity only. It does not validate the input distributions, establish empirical uncertainty calibration, or say anything about physical plasma performance.

## Method

The frozen synthetic decision fixture is evaluated over:
- three unique seeds: 1, 42 and 2026;
- two increasing draw budgets: 10,000 and 20,000;
- six total repeated runs.

For every candidate route, the report records:
- minimum, maximum and range of feasibility probability;
- number of runs in which the route passes the frozen robust-feasibility threshold;
- whether that hard classification is stable across the grid;
- range of Pareto-membership probability where the route is eligible for Pareto analysis;
- top-rank frequency for each diagnostic baseline.

For each group, the output preserves disposition counts over the sensitivity grid. An abstention remains an abstention. A baseline ranking cannot turn an ineligible route into a feasible route.

## Proposed gates

These are algorithm-engineering targets, not empirical calibration standards:
- all seed IDs unique and all draw budgets strictly increasing;
- at least three seeds and two draw budgets;
- minimum 10,000 draws per run;
- probability values finite and within [0, 1];
- zero silent route/classification loss across runs;
- disclose feasibility-probability and Pareto-membership ranges;
- report any robust-feasibility classification flip;
- never infer empirical or physical validity from low Monte Carlo variation.

The fixture declares probability-range tolerances to make sensitivity measurable. Exceeding a tolerance is reported, not hidden or repaired by changing the input band.

## Baselines and SOTA comparison

The same v0.2 report retains its four diagnostic baselines:
- best qualified non-plasma incumbent by mean cost;
- fixed equal-weight normalized sum;
- TOPSIS;
- central-mean non-dominated sorting (not NSGA-II).

This v0.3 increment measures how often their top route changes across seeds/budgets. These methods remain diagnostic-only. The authoritative decision remains hard constraints + robust Pareto + abstention.

## What this does not solve

- source-derived empirical distributions or dependence/copulas;
- calibration coverage on sealed physical/historical holdouts;
- model-discrepancy estimation;
- false-feasible confidence bounds;
- full NSGA-II head-to-head under equal computational budgets;
- value-of-information / next-experiment selection.

Those remain explicit v0.2/v0.4 research requirements. The source candidate groups remain NOT_COMPARABLE until service contracts and process boundaries are actually matched.
