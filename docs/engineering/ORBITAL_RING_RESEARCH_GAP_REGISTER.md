# Orbital-ring research gap register

**Status:** research triage and decision gates; not a feasibility verdict.  
**Prepared:** 2026-10-10.  
**Rule:** literature existence is not validation of this architecture. Keep primary-source figures, secondary summaries, analogy methods, and calculated outputs separate.

## Evidence-state vocabulary

- **ResearchLeadOnly:** a source has raised a question; its claim has not been reconstructed independently.
- **NotRun:** the planned calculation or test has not been executed.
- **Inconclusive:** evidence exists but cannot decide the question within declared uncertainty.
- **PassedAtSpecifiedFidelity:** the bounded claim passed its stated checks; no higher-fidelity claim follows automatically.
- **Blocked:** required architecture, input, frame, or boundary data are missing.

## Literature triage

### Primary orbital-ring source — Paul Birch, 1982

Source: https://nss.org/wp-content/uploads/Orbital-Rings.pdf

Birch's paper is the primary source for the frequently repeated orbital-ring design point, but it is a historical theoretical study, not a modern validated design. It describes a continuous-skyhook configuration with an orbital-ring mass around 5.6 × 10^8 kg and explicitly depends on assumptions about geometry, superconductors, insulation and heat-rejection temperature.

**Important source-reconstruction issue:** the discussion before Table 8 gives cryogenic heat losses of about 100 MW for the whole system and says cryostat electrical power is about 10 GW for heat rejection to a 300 K sink. Table 8 for a typical continuous-skyhook ORS lists total cryogenic heat loss of 5 × 10^8 W and power to cryostats of 5 × 10^10 W, with a note that the estimate assumes a 300 K sink and multilayer vacuum insulation. The paired figures in each passage imply the same rough electrical-input-to-cold-load ratio of 100:1, but the total scale differs by about a factor of five. That may reflect different configurations or assumptions; it should not be called a proven contradiction without reconstructing the design point. Resolve geometry, total length, radiating/cold surface area, temperature assumptions, cryocooler coefficient of performance, and unit conversions before accepting a baseline.

A September 2026 Canadian Institute for Exotic Science brief repeats the 50 GW figure and labels itself “Not independently reviewed”: https://www.instituteforexoticscience.ca/content/frontier/fr-orbital-rings.html. Treat it as an issue-discovery lead, not independent confirmation. The brief's wording also needs to distinguish cold-side heat load from electrical power supplied to the cryostat.

**Conditional stored-energy scale, not a failure prediction:** using Birch's illustrative total ORS mass of 5.6 × 10^8 kg and the 8 × 10^3 m/s speed assumption used in the paper's drag calculation, assigning that inertial speed magnitude to all of the stated mass gives 0.5 M v² = 1.792 × 10^16 J (17.92 PJ). Counter-rotation may change net vector momentum, but does not cancel the individual rings' positive kinetic energies. This arithmetic does not say how much energy would be deposited in a support, a fragment field, or the environment during any failure; the transient energy destinations remain an open physics problem.

### Adjacent mechanics literature — not orbital-ring validation

- Arena and Lacarbonara, “On the stability of magnetically levitated rotating rings,” International Journal of Mechanical Sciences 131–132 (2017), 286–295. DOI: https://doi.org/10.1016/j.ijmecsci.2017.07.007. The paper develops a three-dimensional geometrically exact elastic-ring model with magnetic levitation/recentering forces, rotating-frame formulation, modal analysis and critical-speed stability results. Transferable methods include equilibrium continuation, gyroscopic-mode analysis, and sensitivity to stiffness/cross-section; its geometry and operating regime are not the planetary orbital-ring architecture.
- Tuan and Viet, “Stability of a spinning ring of magnetic dipoles,” Journal of Magnetism and Magnetic Materials 630 (2025), article 173396. DOI: https://doi.org/10.1016/j.jmmm.2025.173396. It analyses linear stability and critical loading/rotation limits for a ring of dipoles. Use it as a candidate method/reference for magnetic-ring submodels only, not as evidence that an orbital ring works.
- NASA Orbital Debris Program Office FAQ: https://orbitaldebris.jsc.nasa.gov/faq/. NASA states that debris in LEO travels about 7–8 km/s, average object-object impact speed is around 10 km/s and can reach 15 km/s, with catalogued debris concentrated unevenly by altitude. A continuous ring's long exposure and cross-section need an explicit flux, shielding, repair and failed-segment analysis.

## Prioritized open questions

| ID | Decision-critical question | Current state | Minimum closure evidence | Claim blocked until closure |
|---|---|---|---|---|
| RG-01 | Can the chosen moving stream and support structure satisfy a vector equilibrium in one fully declared frame? | Blocked | Frozen architecture, geometry, control volume, inertial/relative velocity transform, line density or discrete masses, gravity and every support-force term; independently derived momentum balance and residual tests | “The ring is in equilibrium” |
| RG-02 | What are the defensible cryogenic heat and wall-plug power requirements? | ResearchLeadOnly; primary-source values conflict across passages/configuration descriptions | Reconstruct Birch's tables/equations; normalize total length and geometry; publish separate watts of thermal load and electrical input; evaluate sink temperature, cryocooler COP, insulation, radiation and uncertainty | Any claimed operational power, operating cost, or economic viability |
| RG-03 | What happens when momentum drive, magnetic confinement, superconductivity, or a stream segment fails? | NotRun; high-consequence unknown | Distinct fault trees and transients; close energy and vector-momentum ledgers; verify the inertial energy of each ring; model force interruption, quench, current decay, actuator saturation, fragment/stream dynamics, containment and repair windows | “Power loss is safe,” “the ring comes down slowly,” or “the stored energy is released catastrophically” |
| RG-04 | Can the architecture be constructed, closed and spun up from a non-ring state? | NotRun | Stage-by-stage mass, energy, momentum, trajectory and schedule accounting; demonstrate whether partial arcs have any support state; fail-safe states during bootstrap | Any bootstrap schedule or minimum viable cost |
| RG-05 | What debris/micrometeoroid environment and repair burden does a continuous structure face? | NotRun | Altitude/epoch/orientation, catalogue plus untracked population, material cross-section, relative-speed distribution, shielding, segmentation/isolation, repair logistics and uncertainty | Lifetime, availability, shield mass or maintenance-cost claims |
| RG-06 | Are coupled elastic, magnetic and gyroscopic modes stable over a bounded envelope? | NotRun; analogous literature exists | Independently checked force laws; geometry/material/cross-section; equilibrium first; Jacobian/eigenmodes, nonlinear disturbance cases and convergence evidence; lab correlation before stronger claims | Any “stable” claim based only on an optimizer residual or nominal simulation |
| RG-07 | Does the system deliver net useful payload service after support power and losses? | NotRun | Consistent boundary for payload energy/momentum, launch/braking, cryogenics, drag, power conversion, heat rejection, throughput, maintenance, downtime and lifecycle cost | Cheap-launch, throughput or net-energy claims |

## Execution order

1. **Keep the existing T0 checker.** Its current 47 checks cover circular-orbit arithmetic and one fixed-circle kinematic limit only.
2. **Next: RG-02 source audit in parallel with RG-01 geometry specification.** Reproduce primary-source power/thermal numbers before sizing energy infrastructure or choosing performance targets. Preserve the 100 MW / 10 GW versus 500 MW / 50 GW discrepancy as an open question; do not average the estimates.
3. **Then RG-01 / RG-03.** Freeze one architecture and complete the momentum/energy boundary, including stored stream energy and fault destinations, before a high-energy coupled simulation is treated as meaningful.
4. **Then RG-06 and RG-05.** Use adjacent ring-stability methods cautiously and model debris as a structural/operations constraint, not as a final checklist item.
5. **Only then consider a contained demonstrator.** Its scientific question must be useful even if the megastructure concept fails; stored energy, containment, shutdown and operator procedure require independent safety review.

## Integration boundaries for the toolchain

- **Symthaea:** candidate numerical models and bounded scenario execution; the cognitive planner must not be the safety authority.
- **Mycelix:** provenance and signed records for frozen scenario manifests, model/code versions, input digests, raw outputs, verifier reports and reviewer dispositions. Signatures establish origin/integrity, not physical correctness.
- **Sol Atlas:** communicate architecture assumptions, scenario differences, uncertainties and evidence state. Visualization is not a solver result and cannot upgrade evidence state.
- **Independent verification:** use separately implemented reference calculations and appropriately scoped structural/orbital solvers; a shared code path is not an independent check.

This register should change as evidence is acquired. Do not upgrade a row's state merely because a document, model, simulation, CI job, or signed event exists.
