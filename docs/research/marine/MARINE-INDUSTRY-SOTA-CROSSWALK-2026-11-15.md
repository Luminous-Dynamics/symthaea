# Marine engineering industry and SOTA benchmark crosswalk
## Positioning the Luminous evidence-execution architecture

**Status:** dated research crosswalk v0.1.0  
**Date:** 2026-11-15  
**Tracking:** [MARINE-PROPULSOR-FAMILY-REPRO-001 #12392](https://github.com/Luminous-Dynamics/symthaea/issues/12392)  
**Protocol:** [MARINE-PROPULSOR-FAMILY-REPRO-001.md](./MARINE-PROPULSOR-FAMILY-REPRO-001.md)

> Vendor feature descriptions and case-study outcomes are attributed claims, not independently reproduced benchmark results. This crosswalk identifies what we must compare; it does not claim Luminous currently exceeds any listed tool.

## 1. Industry capability baseline

| System | Publicly described capability | What it means for our benchmark | What we must not claim |
|---|---|---|---|
| DLR Odyssa | Whole-ship aero/hydrodynamic assessment, statistically relevant load/power/motion profiles from transport tasks and historical weather, energy/propulsion optimization, techno-economic analysis; DLR describes current expansion into stability, safety, manoeuvrability and new energy systems. | Benchmark mission/system integration, weather-conditioned performance, energy architecture and lifecycle decision support. | We cannot claim novelty from merely connecting hull, wind, propulsion, weather and economics. |
| SHIPFLOW | Specialized multi-fidelity ship-hydrodynamics workflow spanning potential flow, RANS resistance/self-propulsion/free-surface flow and time-dependent seakeeping/added resistance/manoeuvring, with CAESES integration. | Use as a specialist hydrodynamic baseline and interoperability target. Compare accuracy, runtime and operating-envelope coverage on the same benchmark case. | We cannot claim our reduced-order hydrostatics or a mesh contract replaces RANS, seakeeping or manoeuvring solvers. |
| CAESES | Parametric hull/appendage/propulsor geometry, geometry morphing, constraints, CFD-ready exports, automated DoE/optimization and analytics. Its public maritime page advertises broad design-variant exploration and customer metrics. | Compare geometry robustness, failed-variant rate, automation, design-space coverage and design-to-simulation integration. | Vendor-advertised variant counts or savings are not comparable measured performance unless scenario, boundary and validation method match. |
| Siemens Simcenter STAR-CCM+ | Marine free-surface CFD, rotating propeller domains, cavitation, and six-degree-of-freedom fluid-body interaction capabilities are described in Siemens materials. | Treat as a high-fidelity specialist solver comparator; run identical geometries/conditions where licensed access and solver setup permit. | Do not claim our current geometry/hydrostatic implementation beats commercial multiphase CFD. |
| MARIN / ITTC test practice | Experimental hydrodynamic facilities and published procedures govern open-water, self-propulsion, wake, manoeuvring and other model-test/CFD comparison practices. | Treat ITTC measurement and validation protocols as part of the benchmark's ground truth and reproducibility contract. | A passing software fixture is not equivalent to a towing-tank test or a validated CFD campaign. |

### Public evidence examples

- DLR Odyssa: https://www.dlr.de/en/ms/research-transfer/topics/design-and-optimization-of-future-oriented-ships
- SHIPFLOW: https://shipflow.se/
- CAESES maritime: https://www.caeses.com/applications/maritime/
- Siemens MRA GRC propulsion case: https://resources.sw.siemens.com/en-GB/case-study-maritime-research-associates/
- Siemens cavitating-flow paper: https://static.sw.cdn.siemens.com/siemens-disw-assets/public/64axBt2TbiTVer3SzH3l68/en-US/Siemens%20SW%20Using%20simulation%20to%20predict%20cavitating%20flow%20White%20Paper.pdf
- ITTC Open Water Test 7.5-02-03-02.1, Revision 04 (effective 2021): https://www.ittc.info/media/9621/75-02-03-021.pdf
- ITTC search index includes Practical Guidelines for Ship Self-Propulsion CFD 7.5-03-03-01 (2024 Rev.01): https://www.ittc.info/search/?q=Self-Propulsion

Siemens' MRA case reports a wake-adapted strut/propeller/rudder design for a 43 m patrol boat, 10 propeller variants and five strut/rudder iterations, with reported model- and full-scale physical-test validation. Siemens reports annual fuel savings, speed increase, cavitation elimination on selected appendages and lower radiated pressure-pulse amplitude. These are a valuable case-study reproduction target, but remain vendor-published project claims until independently reproduced under our own frozen boundary.

## 2. Academic SOTA is not one algorithm

Recent hull-form optimization work spans:
- parameterized CFD and evolutionary optimization;
- offline data-driven evolutionary algorithms operating in latent geometry spaces;
- incremental approximation/surrogate models that reuse historical expensive CFD samples;
- adjoint shape optimization with learned propulsion surrogates;
- multi-fidelity constrained Bayesian optimization and acquisition methods that account for downstream validation cost.

These methods optimize different geometry classes, objectives, constraints, solver budgets and data regimes. A single published percentage cannot establish an overall leaderboard.

Relevant dated research anchors:
- Zheng et al., *The generative marine hull form optimization based on offline data-driven evolutionary algorithm*, Ocean Engineering, 15 June 2026, DOI 10.1016/j.oceaneng.2026.125935: latent-space, data-driven evolutionary search under an expensive-CFD budget.
- Liu et al., *Ship hydrodynamic optimization for minimum resistance through an improved grey wolf algorithm and ship model experimental verification*, Ocean Engineering, 1 May 2026, DOI 10.1016/j.oceaneng.2026.125012: parametric/CFD optimization with ship-model experiment.
- Chang et al., *Research on the Application of Incremental Approximation Models in Hull Form Optimization Design*, Machines, 26 August 2026, DOI 10.3390/machines14090968: incremental approximation-model construction intended to reuse historical samples as the design space grows.

The relevant comparison is therefore not "our algorithm versus their algorithm" in isolation. Freeze a common design parameterization, solver, mesh policy, evaluation budget, holdout conditions and hard constraints. Then compare search strategies and evidence cost.

## 3. Where Luminous should compete

The credible differentiation is an **evidence-execution architecture**, not a replacement physics solver:

1. **Evidence-bearing geometry:** source requirements, parameterization, revisions, coordinate frames, units, geometry digests and independent admission remain linked.
2. **Solver interoperability:** consume specialist solvers and their native artifacts; preserve solver build, configuration, mesh, convergence, uncertainty and result digests.
3. **Causal sensitivity and experiment selection:** determine which design variables control which observables, identify model discrepancy, and choose the next experiment by expected reduction in decision uncertainty—not only expected objective improvement.
4. **Regime-conditional trust:** calibrate each observable by vessel family, speed/loading, Reynolds/cavitation regime, hull state and measurement facility; do not hide local failure behind pooled accuracy.
5. **Robust decision quality:** use uncertainty intervals and non-compensable feasibility gates; allow abstention when rankings are unresolved.
6. **Lifecycle digital thread:** retain design, construction, as-built inspection, retrofit, degradation and in-service trial evidence across revisions.
7. **Fleet reconstruction and change detection:** reconstruct historic design intent, then compare it with surveys and observed performance without rewriting history or treating drawings as as-built truth.

This makes existing commercial tools potential solver providers and benchmark comparators rather than systems we need to replace.

## 4. Head-to-head benchmark protocol

### A. Matched hydrodynamic case

Use a published hull/propeller case with accessible geometry, test data and rights. Freeze geometry, scale, loading, trim, speed, water properties, turbulence/model settings, mesh-refinement plan and boundary conditions.

Report resistance, sinkage/trim, thrust, torque, wake, pressure pulse, cavitation observables, convergence and uncertainty. Compare at the same fidelity and report compute cost. If solver configurations differ, report the difference; do not normalize them away.

### B. Matched optimization case

Use the same baseline geometry, admissible design variables, parameter bounds, objective/constraints, solver build and evaluation budget. Compare:
- fixed-grid or space-filling design of experiments;
- random sampling;
- conventional evolutionary optimization;
- standard Bayesian optimization;
- incremental/online surrogate update;
- current multifidelity acquisition;
- sensitivity/causal-information-guided acquisition.

Seal the final holdout before tuning. Keep all low-/high-fidelity and experimental groups disjoint by vessel lineage and operating condition.

### C. Independent validation

For each candidate accepted by the optimizer, perform the same prescribed higher-fidelity or experimental validation. Report:
- false-feasible rate;
- conditional calibration/coverage for each observable and operating regime;
- ranking stability and top-k selection regret;
- validated Pareto hypervolume, with identical normalization and holdout basis;
- number of expensive validations and total core-hours/wall time;
- cost to first independently confirmed improvement;
- cost to independently confirmed Pareto frontier;
- fraction of candidate designs that remain unresolved or are correctly rejected.

Do not let the optimizer select the holdout cases or change acceptance thresholds after seeing results.

### D. Lifecycle confirmation

Only after physical/solver evidence exists, compare annual delivered cargo work, energy, availability, lifecycle cost, repair burden and environmental outcomes under matched route/service conditions. Use the existing WAPS, energy, service-robustness and acoustic owners.

## 5. Proposed programme targets

These are research targets, not current achievements:
- **Propulsor-specific milestone:** at least 5% independently validated reduction in delivered propulsor/shaft power under matched vessel service; 10% stretch target.
- **Whole-mission parent target:** at least 10% validated whole-mission energy improvement; 15% stretch target.
- at least 10% validated Pareto-hypervolume improvement under the frozen common objective normalization;
- at least 20% reduction in compute/evaluation cost to the independently confirmed Pareto frontier versus preregistered random/grid baselines;
- no increase in false-feasible rate relative to the matched incumbent search process;
- condition-specific calibration reported for every primary observable; unresolved conditions remain unresolved;
- zero mandatory safety/environmental gate violations in any admitted candidate;
- 100% provenance closure for geometry, solver configuration, raw/derived evidence and test reports used in accepted claims.

A propulsor-level power result is not a whole-mission energy result. The parent targets are assessed only after route, payload, auxiliary loads, schedule and lifecycle effects are included.

These are only meaningful when candidate service and constraints match. Failure to achieve them is a publishable result.

## 6. Execution sequence

1. Reproduce a public experimental baseline before introducing new propeller geometries.
2. Reproduce one specialist-solver/self-propulsion case with held-out comparison data.
3. Compare search/acquisition policies under identical solver and compute budgets.
4. Measure how often screening predicts a false winner or falsely feasible candidate.
5. Integrate the confirmed results with mission/lifecycle assessment and the existing evidence owners.
6. Only after the above, claim a competitive advantage in validated decision quality or cost-to-validation.

## 7. Claim ceiling

This crosswalk is a research plan based on public vendor descriptions, published studies and ITTC procedures. Vendor case studies and marketing metrics are not independently verified Luminous results. No solver accuracy, hull improvement, fuel saving, cavitation benefit, class approval or physical qualification is claimed by this document.
