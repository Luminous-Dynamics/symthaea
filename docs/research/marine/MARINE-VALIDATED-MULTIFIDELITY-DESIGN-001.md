# MARINE-VALIDATED-MULTIFIDELITY-DESIGN-001
## Optimize for independently validated improvement, not predicted improvement

**Status:** research specification v0.1.0  
**Created:** 10 November 2026  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)

> Research specification only. No multi-fidelity optimizer, model, vessel design or validated performance improvement is claimed.

## 1. Paradigm shift

Multi-fidelity optimization is not itself novel in naval architecture. MARIN has documented practical afterbody optimization using many low-cost potential-flow/coarse-grid RANS evaluations and fewer high-fidelity free-surface RANS evaluations. Recent work extends multi-fidelity Bayesian optimization to aircraft constraints, hydrofoil CFD and validation-aware acquisition functions.

The important distinction is that a low-fidelity optimum is not a result until its feasibility and objective value are checked at the required higher fidelity. That final verification is a deferred cost. A design process that optimizes predicted low-fidelity performance while omitting mandatory validation cost can systematically over-sample cheap but uninformative evaluations and select designs that fail later.

Our target should be **cost to independently validated Pareto improvement**, not model RMSE or surrogate optimum alone.

## 2. SOTA and industry comparison

- **MARIN practical ship afterbody optimization (Raven & Klinkenberg, 2023):** combines many lower-cost potential-flow or coarse-grid RANS calculations with few high-fidelity free-surface RANS simulations for hull afterbody optimization.
- **Multi-fidelity constrained Bayesian optimization for aircraft design (Cordelier et al., G-2026-17):** uses objective and constraint information for fidelity selection; reports more constraint-compliant solutions under a limited budget on its aircraft design application.
- **Consequential improvement acquisition function (Aydogdu et al., 27 March 2026):** explicitly accounts for the hidden downstream cost of validating a low-fidelity selected optimum, rather than reward every cheap low-fidelity objective improvement as if it were final.
- **Adaptive gated multi-fidelity network for hydrofoil design (Paladaechanan et al., Ocean Engineering, 15 January 2026):** demonstrates an adaptive surrogate/optimization application to hydrofoil CFD.

Luminous should not claim to invent multi-fidelity optimization. The differentiator is a maritime, evidence-qualified benchmark where the value of a query includes hard constraints, model discrepancy, cargo-service outcomes and the required independent validation of final designs.

## 3. Fidelity stack

| Fidelity | Typical method | Decision role |
|---|---|---|
| F0 | Analytical / empirical screening | Reject obviously poor design regions |
| F1 | Low-order aero/hydro, 1-DOF propulsion | Broad design and route screening |
| F2 | 3-/4-DOF response, leeway/rudder and propeller-hull coupling | WAPS design/route refinement |
| F3 | Refined CFD/FEA and transient aero models | Local physics resolution |
| F4 | Controlled model tests, towing tank and cyber-physical tests | Physical-model validation |
| F5 | Instrumented full-scale / sea-trial evidence | In-service physical qualification |

Do not assume fidelities are interchangeable or perfectly correlated. Each model needs its own domain, numerical and model discrepancy, and output-specific validation state. F3 CFD is not automatically a substitute for F4/EFD or F5 evidence where these are required.

## 4. Objective and acquisition policy

Maintain a Pareto vector of delivered cargo-service cost, primary energy, WtW GHG, reliability, payload/service and risk. Keep non-compensable hard gates outside weighted objective trade-offs.

For each candidate query, track:
- evaluation cost at each fidelity;
- expected objective information gain;
- expected feasibility/constraint information gain;
- probability of rank reversal at the decision fidelity;
- expected reduction in final decision regret;
- required cost of eventual validation;
- model applicability and evidence eligibility.

Compare cost-aware expected improvement and knowledge gradient against constraint-aware and consequential-improvement acquisition. A low-fidelity query has no realized decision benefit merely because it improves a cheap surrogate. Its value must be assessed through the resulting final high-fidelity/physical validation outcome.

## 5. Benchmark design

### Baseline algorithms
1. Single-fidelity high-fidelity search under a fixed budget.
2. Space-filling low-to-high fidelity escalation.
3. Cost-aware expected improvement / knowledge gradient.
4. Constraint-aware multi-fidelity Bayesian optimization.
5. Consequential-improvement / validation-aware acquisition.
6. Adaptive fidelity based on Pareto-rank uncertainty, constraint uncertainty and model discrepancy.

### Test families
- ITTC Series 60 resistance/flow uncertainty example.
- Marine hull-afterbody multi-fidelity optimization benchmark.
- SOBC-1 wind-assist cyber-physical data and #11510 WAPS force/application-point studies.
- Kamsarmax suction-sail 1-/3-DOF route benchmark.
- Single-screw leeway/rudder/propeller self-propulsion case.

Keep cases separate by hull, ship class, observable and service. No pooling into one score unless the statistical and engineering compatibility is justified.

### Measurement
Report:
- total cost to independently validated Pareto front;
- high-fidelity and physical-test query counts;
- hypervolume gap at the decision fidelity;
- p50/p95 design-rank inversion;
- false-feasible and false-infeasible rates;
- constraint-margin calibration and out-of-domain abstention;
- final validation regret and objective/metric discrepancy;
- cost and performance by fidelity, not only total wall-clock.

## 6. Proposed research targets

- Zero false-feasible selected designs in the declared negative-control corpus.
- Zero final candidates labelled validated solely by F0–F3 evidence.
- At least 30% lower total evaluation cost to reach a validated Pareto frontier than a fixed high-fidelity baseline, while preserving the same validation quality.
- 50% lower high-fidelity query count as a stretch target only if validation regret and hypervolume gap are not worsened.
- Complete provenance for every model observation, version, solver input, calibration and validation record.
- Final superiority must exceed combined numerical/experimental uncertainty and persist on blind operating-condition slices.

Targets are aspirational; they are not achieved results.

## 7. Integration with Luminous

- Symthaea: decision constraints, evidence eligibility, model discrepancy and Pareto/validation-regret analysis.
- Symtropy: deterministic execution, solver replay and cost/energy accounting.
- Sol-Atlas: geometry, mesh, force-application locations and route/spatial conditions.
- Mycelix: immutable model, calibration, test, supplier and physical evidence provenance.

The architecture should orchestrate and qualify specialist tools; it should not attempt to reproduce a commercial CFD solver or physical testing laboratory.

## 8. Sources

- Raven & Klinkenberg, *Practical ship afterbody optimisation by multifidelity techniques* (2023): https://www.marin.nl/en/publications/practical-ship-afterbody-optimisation-by-multifidelity-techniques
- Cordelier et al., *Multi-fidelity approaches for general constrained Bayesian optimization with application to aircraft design* (2026): https://www.gerad.ca/en/papers/G-2026-17
- Aydogdu et al., *Consequential improvement acquisition function for efficient multi-fidelity Bayesian optimization* (27 March 2026): https://doi.org/10.1007/s11081-026-10078-5
- Paladaechanan et al., *Ensemble adaptive gated multi-fidelity neural network for Bayesian optimization: Application to hydrofoil design* (15 January 2026): https://doi.org/10.1016/j.oceaneng.2025.123314

## 9. Claim ceiling

A surrogate improvement is not a validated improvement. A multi-fidelity sample-efficiency gain does not demonstrate better ship performance. Final candidates require independently qualified higher-fidelity or physical validation, matched cargo service and declared uncertainty.


## 10. Environmental-condition extensions: acoustic noise and hull biofouling

The validated design frontier now includes two measured environmental dimensions that are coupled to propulsor and afterbody choices:

### 10.1 Underwater radiated noise

The acoustic objective is a frequency-resolved spectrum with standard part, water-depth regime, hydrophone calibration, ambient reference and expanded uncertainty attached. Do not collapse one-third-octave and decidecade results, deep-water and shallow-water methods, or directly measured radiated-noise level and equivalent monopole source level into a single score without a reviewed conversion.

Use the separate measurement and evidence contract in MARINE-ACOUSTIC-ECOLOGY-001 (#11603). F0/F1 acoustic proxies are screening only. F3 acoustic CFD/cavitation prediction remains model evidence until benchmarked against test data; F4/F5 physical acoustic trials qualify the measurement domain independently from hydrodynamic resistance.

An acoustic candidate is not better where an adverse spectral band increases beyond combined measurement/model uncertainty, even if overall energy or integrated dB improves. Report all claimed bands and conditions.

### 10.2 Hull biofouling, cleaning and biosecurity

Hull roughness/fouling state, coating-system revision, inspection coverage, niche-area access, cleaning method, capture/filtration and waste disposition can change resistance, shaft power, cavitation/noise, off-hire time and invasive-species risk. A clean-hull assumption is not free.

Use the hull inspection/cleaning evidence contract in MARINE-BIOFOULING-ENERGY-ECOLOGY-001 (#11608). Inspection, species evidence and cleaning authorization are separate evidence dimensions. Unapproved cleaning or uncontrolled release is a hard block, not a weighted penalty. Partial inspection coverage remains an explicit unknown surface fraction.

### 10.3 Cross-domain join

MARINE-HULL-STATE-EVIDENCE-JOIN-001 (#11612) links the acoustic trace, inspection, cleaning and pre/post propulsion observations through vessel revision, service fingerprint, trace digests, time and condition identity. It provides contextual linkage only; it does not prove causal fuel/noise reduction or biological effect.

### 10.4 Extended Pareto vector and hard gates

For the same service contract, preserve a Pareto vector including:
- lifecycle cost per delivered cargo work;
- primary energy and well-to-wake GHG per delivered work;
- schedule reliability, throughput and availability;
- frequency-resolved underwater radiated-noise metrics and uncertainty;
- hull-condition/fouling state, cleaning interval, off-hire and coating damage;
- captured/uncontrolled cleaning waste and biosecurity evidence;
- model discrepancy and evidence/inspection completeness.

Non-compensable gates include vessel/propulsor operating limits, trial/data rights, calibration and measurement validity, cleaning authorization, required waste containment, and out-of-domain solver evidence. No lower energy or lifecycle cost may compensate for a blocked environmental or operational hard gate.

### 10.5 Multi-fidelity acquisition and value of measurement

An acoustic test, ROV inspection, coating survey or cleaning-waste sample is an evaluation with cost and decision value. Its acquisition value depends on whether it changes the validated Pareto ranking, reduces uncertainty around a hard gate, or resolves an out-of-domain decision—not just information gain.

The benchmark should compare:
- optimization that treats acoustic/fouling data as post-hoc annotations;
- optimization that includes them as explicit objectives and hard constraints;
- measurement selection by uncertainty reduction;
- decision-value-aware measurement selection that accounts for candidate rank/feasibility changes and the future validation cost.

Report cost to independently validated environmental-service Pareto improvement, not count of cheap surrogate queries.

## 11. Extended falsification suite

- A low-fidelity acoustic score cannot qualify a candidate for measured-URN superiority.
- A modelled or uninspected hull cannot be assigned zero fouling or zero invasive-species risk.
- A synthetic or structurally joined acoustic/hull/performance record cannot be promoted to measured causal evidence.
- An unauthorized or uncontrolled in-water cleaning event blocks the affected candidate.
- A candidate with improved energy but unresolved or materially worse declared noise bands cannot be ranked as unconditionally superior.
- A design may claim improvement only on the held-out fidelity/condition slices for which both the physical performance and environmental evidence are independently eligible.

These additions do not alter the earlier acceptance targets; they extend the benchmark's objective and constraint contract.

