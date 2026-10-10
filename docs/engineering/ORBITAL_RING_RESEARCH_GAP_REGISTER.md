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

**Critical source-audit distinction: three different power numbers must not be conflated.**

1. **Conducting-sheet eddy-current drag, pp. 482–483, §3.4.** The paper first estimates about 0.2 GW for one illustrative aluminium-sheet case. It then gives a total precessing force of roughly 5 × 10^11 N and says the power loss for the preceding conducting-sheet model would be around 500 GW. Using the paper's stated relation \(P_E=F_L\,2/(\mu_0\sigma\delta)\), with aluminium conductivity \(3.54\times10^7\,\mathrm{S/m}\), sheet thickness 0.05 m, and the rounded force estimate gives an equivalent coefficient of about 0.899 m/s and power of about 449.6 GW. This reproduces the cited ~500 GW scale to within roughly 10%. It is a **model-specific electromagnetic drag estimate**, not cryostat wall-plug power and not a universal ORS operating-power requirement. The same section says linear induction motors might cancel much of the eddy-current loss, so any design must compare the actual coupling, control and power-recovery architecture rather than blindly carry 500 GW forward.

2. **Proposed superconducting surface, p. 483.** Birch then describes dissipation reduced to cryogenic heat losses of about 100 MW for the whole ORS and approximately 10 GW power to cryostats for rejection to a 300 K sink. This describes a different proposed mitigation/configuration from the conducting-sheet drag estimate.

3. **Table 8, p. 485, typical continuous-skyhook ORS.** The table reports total cryogenic heat loss of 5 × 10^8 W and cryostat power of 5 × 10^10 W, with a note that the estimate is conservative, assumes a 300 K sink, and assumes multilayer vacuum insulation. Both p. 483 and Table 8 imply a rough 100:1 wall-plug-power-to-cold-load ratio; the absolute scales differ by five. A new internal-consistency check divides the table's per-ring superconductor masses by their mass-per-length values: 5.3 × 10⁷ kg / 1.2 kg m⁻¹ ≈ 44.17 Mm and 8.8 × 10⁷ kg / 2.0 kg m⁻¹ = 44.00 Mm. Their mean length combined with 10 W/m implies 440.8 MW, within about 12% of the reported 500 MW. This supports internal consistency at the table's rough precision, but does not explain the earlier 100 MW paragraph. See [the detailed power-budget audit](ORBITAL_RING_POWER_BUDGET_AUDIT.md); source geometry, scope, and the paragraph/table difference remain open.

The audit script scripts/engineering/orbital_ring_power_source_audit.py encodes these source numbers, reproduces the smaller ~0.17984 GW worked example and the larger ~449.59 GW estimate, checks the paired cryogenic ratios and the 5× scale difference, reconstructs the Table 8 implied length/heat-load scale, and records conditional stored-energy arithmetic. Its **16/16** checks passed locally and repeated JSON output was byte-identical. Source Git blob: `5b6e23922027c64b604158b2557fc68673a4ecc9`; source SHA-256: `adab9dcdd4bd8fc6f8ffa99d5fb42035849054d68852ddaca631da63c9ad7514`; output SHA-256: `b353bdf568f2f1358cfff49cb5d2bf334e37b137d41876bbd79a193e7aa8df5d`. See [the detailed power-budget audit](ORBITAL_RING_POWER_BUDGET_AUDIT.md) for source page references, derivation, caveats, and the next reconstruction tasks. This is source-number arithmetic, not a complete reconstruction of Birch's design or independent engineering validation.

A September 2026 Canadian Institute for Exotic Science brief repeats the 50 GW figure and labels itself “Not independently reviewed”: https://www.instituteforexoticscience.ca/content/frontier/fr-orbital-rings.html. Treat it as an issue-discovery lead, not independent confirmation.

**Conditional stored-energy scale, not a failure prediction:** using Birch's illustrative total ORS mass of 5.6 × 10^8 kg and the 8 × 10^3 m/s speed assumption used in the paper's drag calculation, assigning that inertial speed magnitude to all of the stated mass gives \(0.5 M v^2 = 1.792\times10^{16}\,\mathrm{J}\) (17.92 PJ). Counter-rotation may change net vector momentum, but does not cancel the individual rings' positive kinetic energies. This arithmetic does not say how much energy would be deposited in a support, a fragment field, or the environment during any failure; the transient energy destinations remain an open physics problem.

### Adjacent mechanics literature — not orbital-ring validation

- Arena and Lacarbonara, “On the stability of magnetically levitated rotating rings,” International Journal of Mechanical Sciences 131–132 (2017), 286–295. DOI: https://doi.org/10.1016/j.ijmecsci.2017.07.007. The paper develops a three-dimensional geometrically exact elastic-ring model with magnetic levitation/recentering forces, rotating-frame formulation, modal analysis and critical-speed stability results. Transferable methods include equilibrium continuation, gyroscopic-mode analysis, and sensitivity to stiffness/cross-section; its geometry and operating regime are not the planetary orbital-ring architecture.
- Tuan and Viet, “Stability of a spinning ring of magnetic dipoles,” Journal of Magnetism and Magnetic Materials 630 (2025), article 173396. DOI: https://doi.org/10.1016/j.jmmm.2025.173396. It analyses linear stability and critical loading/rotation limits for a ring of dipoles. Use it as a candidate method/reference for magnetic-ring submodels only, not as evidence that an orbital ring works.
- NASA Orbital Debris Program Office FAQ: https://orbitaldebris.jsc.nasa.gov/faq/. NASA states that debris in LEO travels about 7–8 km/s, average object-object impact speed is around 10 km/s and can reach 15 km/s, with catalogued debris concentrated unevenly by altitude. A continuous ring's long exposure and cross-section need an explicit flux, shielding, repair and failed-segment analysis.

## Prioritized open questions

| ID | Decision-critical question | Current state | Minimum closure evidence | Claim blocked until closure |
|---|---|---|---|---|
| RG-01 | Can the chosen moving stream and support structure satisfy a vector equilibrium in one fully declared frame? | Blocked | Frozen architecture, geometry, control volume, inertial/relative velocity transform, line density or discrete masses, gravity and every support-force term; independently derived momentum balance and residual tests | “The ring is in equilibrium” |
| RG-02 | What are the defensible electromagnetic-drag, cold-side heat, and cryostat wall-plug power requirements? | Arithmetic source audit passed; design-point/column normalization unresolved | Reconstruct Birch's conducting-sheet force model and the Table 8 geometry; separate drag power from cryogenic heat load and cryostat electrical input; normalize total length/surface area; evaluate sink temperature, cryocooler COP, insulation, motor recovery and uncertainty | Any claimed operational power, operating cost, or economic viability |
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
