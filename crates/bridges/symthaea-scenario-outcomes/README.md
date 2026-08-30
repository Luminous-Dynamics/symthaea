# symthaea-scenario-outcomes

Plural, distribution-aware consequence vectors for evidence-bearing scenarios.

## Core rule

There is no built-in civilization score, utility scalar, fitness value, or stakeholder-independent ranking.

A scenario may expose multiple incommensurate consequence dimensions such as:

- water reliability;
- habitat area;
- household disruption;
- energy demand;
- capital cost;
- resilience;
- health or safety indicators.

Each dimension carries units, optional uncertainty bounds, optional time horizon, provenance/source references, and optional distributional slices.

## Distributional visibility

System-wide averages can hide who benefits and who bears costs. `DistributionSlice` makes heterogeneous outcomes explicit across geographic regions, population groups, facilities, ecosystems, or stakeholder groups.

## Preference boundary

This crate does not decide whether higher or lower values are morally preferable, and it does not assign hidden weights across dimensions.

If a user, community, institution, or governance process wants to rank scenarios, that preference model must be explicit, auditable, and outside this crate.

## Scenario binding

`ScenarioOutcomeVector::for_scenario` binds consequence results directly to a `CounterfactualScenarioEnvelope` so callers do not manually retype scenario identifiers.

## Non-goals

This crate does not:

- grant execution authority;
- estimate causal effects;
- convert simulation results into truth;
- choose stakeholder weights;
- collapse disagreement into a single score.
