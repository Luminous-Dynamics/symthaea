# Mass-Coupled Battery Sizing: Reduced-Order Coordinator and Evidence Gates

**Date:** 2026-10-17  
**Status:** research implementation; not a validated aircraft-performance solver  
**Tracking:** [AIRROAD-003 #7729](https://github.com/Luminous-Dynamics/symthaea/issues/7729)  
**Related:** [aircraft energy screening PR #7647](https://github.com/Luminous-Dynamics/symthaea/pull/7647), [S4 requirements PR #7761](https://github.com/Luminous-Dynamics/symthaea/pull/7761), [serviceability benchmark PR #7773](https://github.com/Luminous-Dynamics/symthaea/pull/7773)

## 1. Why this matters

A battery-electric aircraft cannot be fairly compared using one frozen bus-power trace if candidate masses differ. Installed energy-system mass affects takeoff mass; changed mass affects the forces and power needed across the mission; the newly calculated power and energy demands then change the battery required. That new battery changes mass again.

The fixed-profile design-space sweep in `design_space.rs` explicitly describes itself as first-pass sensitivity analysis and does not close this feedback loop. This module adds the missing **iteration/orchestration layer**, but it deliberately does not create aerodynamic coefficients, propulsion maps, geometry, or a flight-performance solver from assumptions.

## 2. Scope of implementation

`crates/domains/symthaea-aircraft-energy/src/mass_coupled_sizing.rs` adds:
- A `MassSensitiveMissionModel` adapter that must provide a newly calculated phase profile for the exact candidate/configuration and the supplied gross mass.
- Explicit mutually exclusive non-energy mass terms, a maximum takeoff-mass limit, and pack-level battery energy/power properties.
- Battery-pack sizing by the maximum of energy-limited, continuous-power-limited, and peak-power-limited mass.
- A bounded, relaxed fixed-point iteration with explicit mass residual, absolute/relative tolerances, and a maximum of 10,000 iterations.
- A complete iteration trace with the gross mass used by the demand model, upper-bound mission energy, power-related mass floors, required pack and installed-system masses, target takeoff mass, and closure residual.
- Fail-closed states for invalid inputs/demand, missing model outputs, peak-duration overruns, takeoff-mass-limit exceedance, and non-convergence.
- 12 synthetic Rust test declarations, including mass-sensitive demand, peak-power sizing, uncertainty-bound rejection, peak-duration and takeoff-mass hard stops, model failure, and bounded non-convergence settings.

This provides the coupling *plumbing*. No actual aerodynamic or propulsion model is shipped by this module, and the test fixtures are not aircraft performance data.

## 3. Reduced-order equations

For phase (i), the upstream model supplies duration (t_i), nominal bus power (P_i), duty class, and an explicit uncertainty interval ([P_{i,L},P_{i,U}]). The current sizing coordinator uses upper-bound power for a deterministic bound-based screening case:

[
E_U = \sum_i \frac{P_{i,U} t_i}{3600}
]

Let pack-level properties be specific energy (e_p), usable fraction (f_u), temperature capacity fraction (f_T), end-of-life health (f_{EOL}), discharge efficiency (\eta), and reserve fraction (r). The energy-limited pack mass is:

[
m_E = \frac{E_U}{e_p f_u f_T f_{EOL}\eta(1-r)}
]

Given pack-level continuous and peak specific power (p_c,p_p), with power derating (f_P), derive separate lower bounds:

[
m_c = \frac{P_{c,U}}{p_c f_P}, \qquad
m_p = \frac{P_{p,U}}{p_p f_P}
]

The required pack mass is the maximum—not an average—of these constraints:

[
m_{pack}=\max(m_E,m_c,m_p)
]

The input's declared installed-system multiplier (k_s\ge1) must represent the complete system boundary relative to pack mass, including the parts included by that boundary, such as cooling, protection, mounting, conversion, and distribution. Then:

[
m_{installed}=k_s m_{pack}
]

[
m_{TO,target}=m_{airframe}+m_{propulsion}
+m_{consumables}+m_{payload}+m_{installed}
]

The coordinator calls the upstream mission model again at the updated gross mass, sizes the battery again, and repeats until the mass residual meets declared absolute/relative tolerances. Relaxation is exposed because poorly conditioned fixed-point problems may oscillate or fail to converge.

These equations are intentionally explicit and simple. They do **not** replace battery thermal/transient modeling, voltage/current limits, motor/inverter efficiency maps, generator/fuel models, aerodynamics, CG calculation, structural margins, takeoff/climb/stall calculations, or any required certification analysis.

## 4. Why upper-bound sizing and hard constraints are separate

The energy/power calculations use the declared upper power bounds; the bounds are only as trustworthy as the supplied method and supporting evidence. They are not automatically probability distributions, calibrated confidence intervals, or safety margins. Their derivation, domain of validity, and calibration must live in the source artifact manifest.

Two conditions stop sizing rather than being averaged into a score:
1. Total peak-phase duration exceeds the pack's declared mission budget. Adding pack mass does not solve a duration-budget mismatch under this model.
2. The resulting required takeoff mass exceeds the declared limit.

Missing/invalid phases, nonfinite arithmetic, failed model execution, and failure to converge are never successes. A nonconverged result still contains the iteration trace for diagnosis, but has no accepted final energy envelope.

## 5. Interface responsibilities and provenance

The upstream implementation of `MassSensitiveMissionModel` is responsible for:
- matching the exact candidate geometry/configuration, mass state, mission family, route and atmospheric/wind state;
- using traceable aerodynamic and propulsion data/models with units, reference conditions, applicability domain, uncertainty, and limitations;
- recalculating phase loads at each queried gross mass, not replaying a trace derived for a different candidate or mass;
- retaining model version and run provenance; and
- validating its model against appropriate analytical/experimental reference data before performance predictions are used in design decisions.

The existing `mission_model_contract.rs` validates a candidate manifest's structure and digest-equality bindings. It does not independently fetch the underlying artifacts or recompute their hashes/physics. The new coordinator does not add that authentication. A solver's output must be attached to the existing provenance contract by its caller; a locally valid contract is not proof that the cited aerodynamic model is correct.

## 6. Candidate benchmark and verification plan

### M0 — arithmetic and interface invariants

- A known synthetic mass-sensitive model changes power monotonically with the mass passed to it; the trace must show repeated evaluation and a residual within the pre-registered tolerance.
- Candidate identity/configuration identity is retained in the result.
- Pack sizing satisfies both worst-case mission energy including reserve and continuous/peak power bounds, using the *same* pack mass.
- Upper power bound greater than nominal changes the energy/power sizing as expected.
- Empty phases, duplicate phase identities, inverted/non-containing uncertainty bounds, nonfinite demand, and model errors do not produce convergence.
- Peak-duration and MTOW violations produce explicit hard-stop dispositions.
- Iteration count is bounded; a deliberately insufficient iteration budget returns non-convergence.

### M1 — reference aircraft data

Select a published, configuration-specific conventional reference such as the Cessna Skyhawk, with the reference data card preserving source conditions and known limitations. Use the data only for requirements and comparison design at first; a manufacturer brochure does not itself supply a validated electrical phase-demand trace.

### M2 — candidate-specific physics adapter

Connect an independently reviewed geometry/performance model. NASA OpenVSP/VSPAERO offers geometry and aerodynamic analysis, along with published verification/validation comparisons to theoretical or experimental cases. This is a possible component in a toolchain, not a standalone validation of a complete airplane. Select relevant cases, document solver version/settings and applicability, and report error/uncertainty rather than tune the reference until it passes.

### M3 — independent validation

Compare computed quantities against withheld/reference analytical and experimental values with pre-registered error bands, unit/reference-condition checks, mass/configuration reconciliation, and negative controls. Separate validation of individual solvers from validation of their integration.

### M4 — physical/authority path

Ground tests, battery/thermal testing, structural tests, ground/flight tests and regulatory compliance are separate qualified programs. Simulation convergence does not authorize construction, maintenance, flight, or certification.

## 7. Initial numeric targets

The numbers that can be justified at the coordinator level are invariants, not aircraft-performance targets:
- mass residual at convergence no greater than the declared `absolute_tolerance + relative_tolerance × max(|current mass|, |target mass|)`;
- all declared non-energy mass terms included exactly once;
- no acceptance if required takeoff mass exceeds its limit;
- all sizing equations use upper declared power bounds and the same sizing pack mass;
- zero silent fallback values for missing mass, battery, route, environment, or phase demand;
- 100% trace visibility for every iteration and stop condition;
- max 10,000 iterations, with non-convergence explicitly reported.

Do not assign “acceptable range”, “safe climb”, takeoff-distance, payload, battery lifespan, or cost advantages from this module. Those require a validated candidate-specific performance model plus mission, atmosphere, reserves, payload manifest, and lifecycle cost data.

## 8. Research references

- NASA, [OpenVSP/VSPAERO basics](https://www.nasa.gov/reference/openvsp-vspaero-basics/), including reference-value handling and model/result inspection (page updated 2025-09-30).
- OpenVSP, [VSPAERO verification and validation studies](https://github.com/OpenVSP/OpenVSP/blob/main/examples/scripts/python_scripts/report.md), with example geometry and comparisons against published theoretical/experimental cases.
- FAA, [How it Works: Aircraft Certification](https://www.faa.gov/newsroom/how-it-works-aircraft-certification) (2026-04-23): certification progresses through standards, definition, detail, build, test, certification, and production phases.
- FAA, [MOSAIC](https://www.faa.gov/aircraft/MOSAIC), for the current US light-sport certification rule context; exact route must be selected for the intended configuration and intended use.

## 9. Nonclaims

This is a mass-coupling coordinator and arithmetic sizing screen. It does not establish that any aircraft, battery, route, or mission is physically feasible, safe, economical, certifiable, or ready to fly. It contains no new aerodynamic coefficients, propulsion maps, geometry, or flight-control law. A result of `ConvergedWithinModel` means only that the declared input equations and caller-supplied model reached the stated numerical mass tolerance.
