# SX-035: Evidence-Aware Peak Resource Usage Measurements v0.1

**Status:** experimental measurement/forecast comparison; synthetic declarations only  
**Implementation:** suit_peak_usage_measurements  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/7577  
**Predecessor:** SX-034 time-windowed peak resource reservations, PR #7575

## Paradigm shift: estimates must eventually face observations

SX-033 checks cumulative budgets; SX-034 checks peak concurrent reservations. Both necessarily use declared capacity/usage envelopes. SX-035 introduces a distinct observation ledger so instrumented/simulated usage can be compared against the exact forecast that was evaluated. The ledger does not silently rewrite forecasts from observations, promote evidence maturity, or certify measurement apparatus.

Each observation pins:
- exact SX-031 configuration digest, SX-032 task-plan digest, SX-034 reservation-plan digest, SX-034 report digest and reservation receipt digest;
- resource pool/kind and exact provider component/hardware revision;
- mission-relative half-open measurement window;
- observed peak interval [lower, nominal, upper], explicitly peak-over-window rather than mean, instantaneous value or cumulative total;
- sample count, declared maximum sampling interval, measurement method, evidence level and artifact/reference.

A measurement plan has a caller-selected minimum sample count, maximum sampling interval and evidence floor. Those thresholds are policy declarations—not guarantees that a sensor is accurate or that the trace is complete.

## Forecast comparison

The evaluator re-runs SX-034 from the exact configuration, task plan and reservation plan, and requires the supplied report and receipt to match. It rejects stale plan/report/receipt pins, unknown pools, duplicate observation IDs, invalid time windows, bad intervals and resource-kind mismatch.

For each observation, SX-035 finds the largest declared concurrent load in all SX-034 report slices that intersect the observation window. It then separately compares:
1. the observed peak interval to the declared forecast peak upper bound;
2. the observed peak interval to the declared **usable** capacity, which is the conservative capacity lower bound after SX-034 reserve.

This distinguishes:
- forecast certainly below observation, when observed lower bound is above forecast upper bound;
- possible forecast drift, when only the observation upper bound exceeds the forecast;
- possible usable-budget exceedance, when the observed interval straddles the usable capacity;
- definite usable-budget exceedance, when observed lower bound exceeds usable capacity;
- positive observed use when no declared forecast slice overlaps the observation.

A definite exceedance of declared usable capacity blocks the ledger result; possible exceedance or forecast drift degrades it and calls for review. These are software budget classifications, not assertions about the hardware's true physical limit.

## Evidence and limitations

Measurement-method and evidence-level enums are caller declarations. SX-035 does not authenticate sensors, calibration certificates, operator identity, timestamps, trace completeness or data provenance. A hash proves deterministic change binding, not that an observation is genuine. The report never promotes an observation to BenchTested, HumanInLoop or Qualified; maturity remains explicit evidence supplied by the caller.

The ledger is report-only. It does not schedule tasks, actuate hardware, update SX-034 plans automatically, learn safety margins online, or authorize EVA. Peak values are not averaged/cumulative metrics. Time-window alignment, sensor calibration, sampling theorem adequacy, measurement uncertainty and correlated sensor errors remain external engineering responsibilities.

## Current synthetic tests

The unit tests include:
- observed peak below the declared forecast and usable capacity;
- a definite forecast underprediction that remains below usable capacity;
- a definite usable-budget exceedance that blocks;
- possible usable-budget exceedance that degrades rather than claims certainty;
- minimum sample count and sampling interval violations;
- provider revision mismatch;
- positive observed usage with no overlapping declared forecast;
- stale SX-034 receipt rejection;
- resource-kind mismatch and invalid measurement window;
- input order invariance and receipt mutation rejection.

The integration replay independently reimplements the canonical measurement-plan, report and receipt digest encoding and checks baseline conformance, forecast drift and definite budget exceedance. All mission IDs, providers, revisions, usage values and evidence references are synthetic. The fixtures are not instrumentation data or physical validation.

## Proposed benchmark targets

These targets are not measured results. Before enabling timing gates, freeze the reference runner CPU/OS, Rust toolchain, profile, deterministic data seed and corpus manifest.

| Benchmark | Initial target |
|---|---|
| Stale SX-034 plan/report/receipt | 100% rejected |
| Resource-kind/pool identity mismatch | 100% rejected |
| Definite budget overrun (observed lower > usable capacity) | 100% blocked |
| Possible budget overrun (observed upper > usable capacity >= observed lower) | 100% degraded, not labelled definite |
| Forecast upper-bound mutation / measurement mutation | 100% receipt rejection |
| Observation permutation | Identical report/receipt across 1,000 permutations |
| Generated usage/forecast corpus | 2,000 seeded interval cases, zero unexpected panics |
| Scale target | 10,000 measurements / 100 pools p95 under 250 ms on a documented runner |

For every 100% claim, record numerator, denominator, seed and the corpus manifest. Preserve definite/possible boundary examples and positive controls.

## Research basis

NASA's Software Engineering Handbook guidance recommends quantitative margins for constrained resources, robust responses to oversubscription and timely visibility into resource usage so margins can be checked during testing and operations. SX-035 adopts that measurement/estimate-maturation principle, not a claim of compliance with NASA software or flight requirements.

- NASA SWEHB 9.12, Resource Margins: https://swehb.nasa.gov/spaces/SWEHBVC/pages/85426336/9.12%2BResource%2BMargins
- NASA SWEHB 9.13, Resource Oversubscription: https://swehb.nasa.gov/spaces/SWEHBVD/pages/102695804/9.13%2BResource%2BOversubscription
- NASA SWEHB 9.14, Resource Usage Measurement: https://swehb.nasa.gov/spaces/SWEHBVC/pages/85426340/9.14%2BResource%2BUsage%2BMeasurement
- NASA Systems Engineering Handbook appendix: https://www.nasa.gov/reference/system-engineering-handbook-appendix/
