# Candidate-Specific Mission Model Contract

**Date:** 2026-10-17  
**Status:** engineering contract draft; not an aircraft performance model, design approval, or flight authorization.  
**Tracking issue:** [AIRROAD-003 #7729](https://github.com/Luminous-Dynamics/symthaea/issues/7729)  
**Related implementation:** [aircraft-energy screening PR #7647](https://github.com/Luminous-Dynamics/symthaea/pull/7647)  
**Related benchmark:** [Roadster / air-road system benchmark #7692](https://github.com/Luminous-Dynamics/symthaea/pull/7692)

## 1. Purpose and failure this contract prevents

The aircraft-energy crate can screen a mission profile against a supplied energy/power envelope. It can also compare candidate-specific mission profiles when the caller has already generated those profiles. It does **not** currently determine flight power demand from geometry, mass, atmosphere, and propulsion characteristics.

Holding the power trace fixed while changing installed-system mass is not a coupled vehicle-sizing calculation. It can make a heavier or lighter system appear better without accounting for the effect on lift, drag, climb, hover, transition, or propulsion load. The six-axis battery sweep is therefore a bounded *energy-system sensitivity screen* only.

This contract defines the evidence that must travel with a candidate-specific mission profile, and the interface a later reduced-order vehicle model must satisfy. It deliberately does not invent drag coefficients, propulsion efficiencies, battery transients, or transition performance.

## 2. Model boundary and terminology

Keep these distinct:

1. **Candidate configuration:** a versioned, hash-identified proposed physical configuration.
2. **Mass-properties record:** component boundaries, total mass, payload, centre-of-gravity reference and relevant limits.
3. **Mission family:** the intended use case shared by candidates, including the frozen route/payload/environment/reserve assumptions. It is a comparability declaration, not proof of physical equivalence.
4. **Candidate-specific mission run:** a traceable model run using one specific candidate configuration and cited aerodynamic/propulsion/thermal data.
5. **Energy screen result:** the result of evaluating that candidate's bus-demand profile against a declared energy-system envelope.

Do not label a result "feasible aircraft" solely because the arithmetic screen returns FeasibleWithinScreen.

## 3. Required input manifest

Every model run must bind the following sections to one immutable configuration revision.

### 3.1 Identity and provenance

Required fields:
- Stable, non-empty candidate ID and mission-family ID.
- SHA-256 digest of the canonical candidate configuration manifest.
- Model run ID, model version, source revision, and digest for the model artifact.
- Separate, versioned references for aerodynamic data, propulsion/rotor maps, energy-system assumptions and thermal/derating data, where used.
- Each artifact reference includes a locator, revision, digest, retrieval date, units, reference conditions, applicability domain, uncertainty description, and limitations. If a field does not apply to source code or a synthetic fixture, record an explicit not-applicable reason; do not leave it blank.
- Candidate configuration, energy-system envelope, route, mission family, mass properties, environment, and phase-demand output each have declared digests. Each must match its corresponding source artifact digest and the model run's declared input/output digest. The phase-demand digest must cover an explicitly ordered canonical representation of every phase ID, duration, nominal power, duty class, uncertainty interval, and uncertainty method.
- The manifest producer must canonicalize and calculate these digests; the Rust validator checks syntax and declared equality bindings only. It does not hash the Rust structs or prove that a referenced artifact resolves to the bytes claimed.
- Solver settings, convergence/error metadata, and assumptions needed to reproduce the output.

A digest establishes identity, not truth. A hash that cannot be resolved to a source artifact is insufficient. Missing provenance is indeterminate; never fill it with an undocumented default.

### 3.2 Mass and configuration closure

Use mutually exclusive boundaries to avoid double-counting.

| Quantity | Contract definition |
|---|---|
| Fixed airframe/vehicle mass | Structure and fixed systems not included in the variable energy or propulsion masses |
| Installed energy-system mass | Battery pack plus declared conversion, cooling, protection, mounting and distribution hardware—the same boundary used by the energy crate |
| Installed propulsion mass | Motors/engines, propulsors and associated hardware, excluding anything already counted in the energy-system boundary |
| Payload mass | Occupants, baggage, cargo and mission equipment in the declared mission configuration |
| Mission takeoff mass | Sum of the four non-overlapping terms, with consumables added exactly once |
| Maximum takeoff mass limit | Explicit candidate-specific limit with source and applicability; absence is not unlimited capacity |
| Centre-of-gravity reference | Coordinate frame, datum and candidate-specific CG record with loading assumptions and uncertainty |

Reject negative, non-finite, missing, duplicated, or internally inconsistent mass values. If the total exceeds the declared limit, reject that configuration. A valid mass sum does not establish structural strength, stability, controllability, or airworthiness.

### 3.3 Environment and use case

The frozen use-case manifest must state:
- Route distance and endpoints or a versioned route reference.
- Payload/occupant/cargo configuration.
- Ambient temperature, pressure and air density, with source and validity domain.
- Wind/environment envelope or scenario distribution.
- Takeoff/landing mode and site assumptions.
- Mission phases, reserve policy, diversion/abort policy, and relevant charging/turnaround assumptions.
- Configuration-specific limits, such as seating, module installed, or road-only / hover / transition / cruise capability.

A 100 km route with one payload and reserve policy is not comparable to an easier route just because both documents reuse the same family label.

### 3.4 Candidate-specific phase demand

For each phase, preserve:
- Stable phase ID and semantic phase category.
- Duration in seconds.
- Required bus power in watts and energy in watt-hours.
- Continuous/peak classification and cumulative peak-duration accounting.
- Environmental conditions, candidate-configuration digest and source-data/model versions.
- Lower/upper power uncertainty bounds that contain the nominal power value, together with an explicit uncertainty method; do not call a bound a confidence interval without a justified method.
- Sensitivity, solver status, validity-domain checks and limitations.

The validated output must preserve the original provenance-bearing manifest, phase uncertainty intervals, candidate-specific MissionProfile and recomputable nominal energy total. Do not strip the evidence manifest before producing a comparison report.

Energy must be recomputable from recorded phase values with unit-consistent arithmetic. Do not silently combine bus power with shaft power or gross thermal output. Conversion/propulsive losses and thermal derating must be located in an explicit boundary so they are counted exactly once.

If the upstream model cannot establish a phase demand inside its supported validity domain, return an indeterminate finding. Do not extrapolate beyond data limits without an explicitly qualified method and uncertainty.

## 4. Candidate-mass coupling requirement

If energy-system mass or propulsion layout changes, recompute dependent phase demands rather than reusing a stale profile.

1. Freeze a versioned candidate and payload.
2. Close the mutually exclusive mass breakdown and CG/loading assumptions.
3. Resolve geometry, aerodynamic and propulsion maps at a versioned model and applicable domain.
4. Calculate candidate-specific phase demand, retaining environmental conditions and uncertainty.
5. Verify that the mission-run input configuration digest matches the candidate configuration digest.
6. Convert the verified phase record into a Symthaea MissionProfile.
7. Run the energy screen and candidate comparison, preserving invalid, infeasible and indeterminate candidates.
8. Compare only candidates declared against the same mission-family contract and publish the complete assumptions manifest.

No stage may infer unprovided geometry, propulsive efficiency, lift/drag, transition capability, or reserve policy.

## 5. Evidence semantics

- **Configuration valid:** required identifiers, hashes, mass boundaries and input values pass contract validation.
- **Mission model complete:** required phase outputs and data references are present and in-domain.
- **Indeterminate:** evidence is missing, invalid, stale, mismatched, or outside the qualified model domain.
- **Infeasible within model:** an explicitly modeled limit is exceeded.
- **Feasible within screen:** declared arithmetic constraints pass within the supplied model.
- **Externally validated model:** independent comparison against applicable reference data reports error and uncertainty.
- **Approved/airworthy:** never inferred by this crate; requires the relevant authority and compliance evidence.

A model can be internally complete and still be inaccurate. Agreement with one reference point does not validate every operating regime. Preserve negative controls and publish applicability limits.

## 6. Independent validation plan

The first reference-data card is [Cessna Skyhawk manufacturer-published baseline](REFERENCE_BASELINE_SKYHAWK_2026-10-17.md). It captures basic dimensions, weight definitions, performance headline values and the limits of using a summary specification. This is **M1 scaffolding, not an M1 pass**: the public manufacturer page does not by itself provide the controlled performance tables, engine/propeller maps, configuration-specific loading, or independent measurements required to validate a candidate-specific mission model.



Use references for verification, not as borrowed authority:
- NASA OpenVSP/VSPAERO documentation for conceptual geometry and aerodynamic analysis. VSPAERO uses a potential-flow approach; the chosen method must suit the configuration and operating regime.
- OpenVSP published verification/validation studies: https://github.com/OpenVSP/OpenVSP/blob/main/examples/scripts/python_scripts/report.md, each with its own geometry, assumptions and validity scope.
- One documented conventional aircraft with traceable published mass/performance data as an early reference fixture.
- Hand-calculated limiting cases for energy, continuous/peak power, mass sums, reserve and unit conversions.
- Sensitivity and negative-control cases where increasing mass or payload under otherwise fixed assumptions cannot accidentally reduce the demand due to a hidden default or stale cache.

Do not tune on one reference and report the same reference as independent validation. Report absolute and relative error, uncertainty bounds, domain mismatches and unresolved discrepancies. A conceptual model is not certification evidence.

Sources:
- NASA, [OpenVSP / VSPAERO basics](https://www.nasa.gov/reference/openvsp-vspaero-basics/).
- NASA, [OpenVSP ground school](https://www.nasa.gov/software/openvsp-ground-school/).
- FAA, [How It Works: Aircraft Certification (2026-04-23)](https://www.faa.gov/newsroom/how-it-works-aircraft-certification).

## 7. Acceptance gates

### M0 — schema and evidence contract
Versioned types for candidate, mass breakdown, mission family, environment, phase-demand records, model/data references and uncertainty. Test missing/stale/mismatched digests, invalid mass closure, out-of-domain data and invalid unit assumptions.

### M1 — reproducible reference fixture
One named conventional reference with sourced data and a frozen configuration manifest. Hand-calculated mass/energy results must match executable output. Manifest digests and report ordering must be deterministic.

### M2 — coupled reduced-order model
Candidate-dependent profiles demonstrably change when mass/configuration changes. Equations, calibration ranges, and data sources are explicit; unknown coefficients never silently default. Sensitivity sweeps and negative controls are required.

### M3 — independent comparison
Compare against suitable published analytical or experimental reference cases not used for tuning. Report error and uncertainty rather than only pass/fail. Out-of-domain cases are indeterminate.

### M4 — physical evidence boundary
No human-carrying or public-flight inference from software outputs. Hardware testing, specialist review, test permits and authority approvals remain separate gates.

## 8. Immediate next implementation

Start with Rust contract types and validation errors under the existing aircraft-energy crate, plus pure unit tests for mass closure, candidate/configuration hash identity, source provenance, use-case comparability and phase-energy arithmetic. Keep aerodynamic and propulsion computation behind a provider/interface boundary until a model and reference dataset are selected and independently reviewed. Do not add defaults merely to produce a number.

## Nonclaims

This contract does not specify or validate an aircraft, prove a flying-car architecture feasible, provide flight-control logic, authorize vehicle operation, establish certification, or replace qualified aerospace engineering. Numerical targets remain hypotheses until backed by versioned inputs, measurements and independent review.
