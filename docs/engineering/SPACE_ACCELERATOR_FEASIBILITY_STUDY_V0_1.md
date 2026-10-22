# Space-Based Particle-Accelerator Feasibility Study (v0.1)

**Status:** Preliminary engineering trade study. This is not a design approval, cost estimate, orbital-ring qualification, or hardware safety case.  
**Snapshot date:** 2026-10-14  
**Code and engineering home:** Symthaea; the four projects retain distinct responsibilities.

## Executive decision

Do not choose LEO, the Moon, or Mars until the scientific mission is fixed. For a high-energy particle collider, **Earth is the mandatory reference case and remains the strongest first-pass option** because of existing power infrastructure, specialist labour, precision manufacturing, maintenance and scientific supply chains. No off-world location currently has a demonstrated scientific or lifecycle-cost advantage large enough to overcome its infrastructure deficit.

If the study requires an off-world surface location, carry the **Moon** as the leading candidate and **Mars** as a longer-term scenario. Carry **LEO** when operating in orbit itself is required by the experiment, or when a smaller orbital demonstrator would answer a specific question. A space vacuum does not remove the need for a controlled accelerator beam vacuum.

Keep any accelerator mechanically independent of the dynamic orbital-ring architecture until both systems have independently validated models and a tested interface. The open [Symthaea orbital-ring T0 feasibility PR #7353](https://github.com/Luminous-Dynamics/symthaea/pull/7353) explicitly rejects a ring-mounted accelerator as unsupported in its present model. Its bounded arithmetic and manifest checks are useful progress, but they do not establish ring equilibrium, structural stability or physical feasibility.

## 1. First decide what type of machine we mean

The term “particle accelerator” covers several different missions whose requirements should not be mixed.

1. **High-energy collider:** Counter-propagating beams, high-quality magnets, radio-frequency systems, beam vacuum, detectors, collimation, shielding, cooling and machine protection.
2. **Fixed-target or compact research accelerator:** A potentially much smaller facility for materials, isotope, detector or radiation research. Its location may be driven by the experiment more than by collision energy.
3. **Space-environment accelerator experiment:** Justified only where microgravity, orbital exposure or the space environment is a required part of the experiment.
4. **Mass driver or propulsion system:** A different engineering and safety problem, not a particle-physics collider.

Before making a site or cost recommendation, specify the science case, particle species, energy per beam or target energy, luminosity or beam-current requirement, and whether a collider is necessary. Until then, this study can compare architectures and scale envelopes, but cannot responsibly claim a selected design.

## 2. Reference physics and scale checks

### Illustrative 100 TeV proton-collider envelope

For an ultrarelativistic proton beam, a first-order magnetic bending-radius relation is:

\[
R\,[\mathrm{m}] \simeq \frac{p\,[\mathrm{GeV}/c]}{0.2998 B\,[\mathrm{T}]}.
\]

For a 100 TeV centre-of-mass proton-proton collider, each beam has approximately 50 TeV momentum. At an illustrative 16 T dipole field:

- bending radius: about **10.4 km**;
- circumference consisting entirely of bending arc: about **65.5 km**;
- a machine roughly 91–100 km in circumference would leave the remaining length for straight sections, interaction regions, injection, collimation and other systems.

At 8 T, the corresponding bending-only circumference would be roughly 131 km before adding straight sections. These are idealized magnetic-geometry calculations only. They do not include magnet feasibility, beam optics, detector layout, civil works, beam losses, utilities or a machine-protection design.

The Future Circular Collider (FCC) is a roughly 91 km underground proposal. In May 2026, CERN's Council updated the European Strategy for Particle Physics with FCC-ee as the preferred next flagship option; a 100 TeV hadron collider is a potential later stage, not the approved near-term machine. This is why the study must compare against a real Earth-based scientific mission rather than assume that a 100 TeV collider is already the reference design. See [CERN's 2026 strategy update](https://home.cern/the-cern-council-decided-to-update-the-european-strategy-for-particle-physics/) and [FCC programme overview](https://home.cern/science/accelerators/future-circular-collider/).

### Power is a first-order site constraint

Published FCC-ee design material gives illustrative peak electrical power of about **222–357 MW** and annual consumption of about **1.17–1.85 TWh**, depending on centre-of-mass energy. These are comparator values for a mature collider concept, not a universal accelerator estimate and not a final approved site specification. See [FCC-ee update material presented in 2025](https://indico.cern.ch/event/1439855/contributions/6461636/attachments/3076197/5443758/233-Update-FCC_ee_10_page-26052025-V2.pdf).

NASA's fission-surface-power programme has used **40 kWe** as a target class for an initial lunar surface power system. At the order-of-magnitude level, 222–357 MW of peak load is 5,550–8,925 times 40 kW. Annual energy of 1.17–1.85 TWh implies an average load of roughly 134–211 MW, or 3,340–5,280 times 40 kW continuously. This is a scale comparison, not a proposal to build thousands of identical small reactors. It shows that a collider-class lunar facility would require an industrial-scale dedicated power system, not just a first lunar-base power unit. See [NASA's fission surface power programme](https://www.nasa.gov/exploration-systems-development-mission-directorate/fission-surface-power/).

### Waste heat: ideal lower-bound sanity check

For a simplified one-sided radiator emitting to deep space, use the Stefan–Boltzmann lower-bound relation:

\[
A=\frac{P}{\epsilon \sigma T^4}.
\]

If, purely illustratively, 300 MW had to be radiated with emissivity 0.9, the ideal effective emitting area would be around 0.73 km² at 300 K, 0.23 km² at 400 K, or 0.094 km² at 500 K. These are thermodynamic lower bounds—not radiator design areas. A real system needs a heat-flow breakdown, temperature limits, view factors, radiator mass, redundant loops, pump power, thermal cycling, micrometeoroid protection, dust allowances and occlusion analysis. Electrical input and accelerator heat deposited in a particular subsystem must not be conflated.

### LEO gravity-gradient scale check

A 91–100 km circumference structure has a nominal radius of about 14.5–15.9 km. At an altitude of 400 km, the orbital mean motion is approximately \(1.13\times10^{-3}\ \mathrm{s}^{-1}\). In a linearized rotating-orbit frame, differential-acceleration terms are on the scale of \(n^2L\) to \(3n^2L\); across a 15.9 km radial scale this is roughly 0.02–0.06 m/s², depending on direction and geometry.

This is a **differential-acceleration scale, not a structural-load or stability result**. A real LEO facility requires coupled orbit, attitude, thermal and flexible-structure models with a declared orientation and support topology. Orbital debris is also a first-order design input. ESA estimates more than a million objects larger than 1 cm in orbit, many too small to be individually tracked, and warns that impacts at this size can cause severe damage. See [ESA's Zero Debris technologies overview](https://www.esa.int/Space_Safety/Clean_Space/Zero_Debris_Technologies).

## 3. Site trade study

| Candidate | Plausible value | Dominant blockers | First-pass disposition |
|---|---|---|---|
| **Earth surface / underground** | Existing grid, accelerator workforce, manufacturing, diagnostic capability and maintenance. Geological siting can provide shielding. | Civil works, environmental and land impact, permitting, power and water demands, cost, schedule. | **Reference baseline for every high-energy case.** An off-world option must demonstrate a credible advantage. |
| **LEO** | Appropriate if the experiment genuinely needs orbital exposure or microgravity; useful for selected assembly or small-system demonstrations. | Large-structure orbital dynamics, alignment and vibration control, attitude/orbit maintenance, radiation, debris, assembly and service logistics, heat rejection. The beam still needs engineered vacuum. | **Specialized alternative or demonstrator.** No established reason yet to prefer LEO for a 100 km-class collider. |
| **Moon** | Candidate off-world surface site if a local industrial ecosystem is developed; possible excavation and local shielding; closer logistics than Mars. | Hundreds-of-megawatts-scale power for collider-class comparators, waste heat, dust, cryogenics, excavation and geotechnical uncertainty, radiation protection, imported precision hardware, maintenance. | **Leading off-world surface candidate for study**, not yet an economically justified collider location. |
| **Mars** | Could eventually benefit from a mature local industrial base and surface/subsurface infrastructure. | Longer transport and communications, power, dust, spare parts, precision maintenance and build-up of industry. Its thin atmosphere is still not accelerator beam vacuum. | **Long-term scenario.** Revisit only if industrial maturity materially changes the inputs. |

The critical distinction is between a location that makes a particular experiment possible and a location that makes the accelerator cheaper or easier to build. These are not interchangeable claims. No surface location eliminates the need for precision alignment, radio-frequency power, high-quality magnets, engineered beam vacuum, cooling, detectors or machine protection.

## 4. Orbital-ring coupling: keep the interface blocked

An orbital ring is transportation and support infrastructure; an accelerator is a precision beam machine. Mounting the accelerator on a dynamic ring would couple their failure modes and should be treated as a separate, higher-risk architecture.

Before a ring-mounted accelerator can be considered beyond a blocked scenario, an evidence-backed model needs at minimum:

- complete ring, rotor, support, rider and tether geometry, with declared reference frame and epoch;
- mass distribution, force laws, momentum and energy boundaries, and support-load transfer;
- structural modes, disturbance spectra, alignment tolerances and vibration isolation;
- magnetic compatibility, quench and power-loss cases, cryogenic heat paths and failure-energy disposition;
- debris exposure, inspectability, repair and safe-shutdown or evacuation behaviour;
- independent numerical comparisons for dynamics, structural response, thermal behaviour and energy/momentum closure.

A steady circular-orbit reference value is not a dynamic equilibrium proof. A passing manifest validator is not a physical-solver result. An energy ledger is only as complete as its control-volume boundary and force/energy inventory.

The right sequence is to qualify ring dynamics separately, qualify accelerator beam physics separately, and then validate the coupled interface as a third system. Until then, keep the accelerator outside the ring's load path and ensure that any visualization cannot imply that the integrated architecture is feasible.

## 5. How the four projects should divide responsibility

### Symthaea — observability and advisory prediction

Current prototype: [accelerator model source](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-physics/src/accelerator.rs).

The prototype encodes five scalar readings into a hypervector, advances an HDC/LTC state, and scores similarity between a predicted state and the current input. Its horizon outputs are **model-state similarities, not yet calibrated forecasts of physical beam observables**. Current self-reference and timing tests are useful software checks, but they do not demonstrate predictive accuracy on real accelerator data or safety of recommended actions.

Priority improvements:

- Validate units, range, sensor quality, timestamp, missingness and staleness before encoding. Invalid input must not quietly become an ordinary healthy-looking reading.
- Distinguish anomaly score, physical forecast, recommendation and authoritative machine-protection action in types and output reports.
- Preserve simple baselines (engineering thresholds, EWMA/CUSUM or robust statistical detectors) and compare them on held-out, time-ordered data.
- Measure false negatives, false positives, detection latency, calibration and abstention on labelled normal and fault traces, including negative controls.
- Keep Symthaea advisory-only. Independent, qualified machine-protection hardware and controls must own beam-dump authority.

### Mycelix Space — orbital mechanics and space-domain inputs

Relevant code: [Mycelix Space](https://github.com/Luminous-Dynamics/mycelix/tree/main/mycelix-workspace/mycelix-space) and [shared orbital-mechanics crate](https://github.com/Luminous-Dynamics/mycelix/tree/main/mycelix-workspace/crates/orbital-mechanics).

Use this work for orbit propagation, ephemerides, frames, conjunction/debris environments, trajectory constraints, covariance and uncertainty around candidate orbital infrastructure. Do not turn it into an accelerator beam-tracking engine. Existing draft work is adding method-labelled probability-of-collision reference checks and correlated-covariance tests. Reuse that numerical verification discipline instead of duplicating the same kernels. See [PR #4999](https://github.com/Luminous-Dynamics/mycelix/pull/4999) and [PR #5006](https://github.com/Luminous-Dynamics/mycelix/pull/5006).

### Symtropy — geometry, kinematics and physical construction models

The proposed [astrodynamics state/frame contract PR #1573](https://github.com/Luminous-Dynamics/symtropy/pull/1573) and [epoch-bound kinematic-transform PR #1575](https://github.com/Luminous-Dynamics/symtropy/pull/1575) are relevant prerequisites for exchanging physically meaningful states. They are draft interfaces, not independent validation of orbital or structural dynamics. Keep the distinction between representation validity, kinematic correctness, dynamic equilibrium, structural stability and failure survivability explicit. Validate frame and transformation conventions against independent, mature references before using them to support a physical engineering claim.

### Sol Atlas — read-only exploration and evidence presentation

Sol Atlas should present scenarios, uncertainty bands, inputs and verification state; it must not become the authority computing or certifying physics. The draft [dataset-load-health PR #87](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/87) distinguishes loaded, malformed, stale and unavailable data, while [terminal/replay PR #16](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/16) separates a replay address from authoritative evidence and actual execution.

Display statuses such as **Loaded, Stale, Rejected, Blocked, NotRun, and Passed at specified fidelity** explicitly. Never replace unresolved inputs with inferred values. A rendered scenario is a view of a model, not a validation artifact.

### Mycelix DeSci — typed evidence envelopes

The draft [space-engineering evidence envelope PR #4964](https://github.com/Luminous-Dynamics/mycelix/pull/4964) is the natural place to carry source facts, assumptions, model results and independent-check identities. Bind source revision/path, Git object ID, SHA-256, scenario/input manifest, tool/build fingerprint and immutable output artifact. Keep primary run and independent verification as separate records.

This envelope is evidence bookkeeping, not proof that an artifact was authenticated, a solver was executed or a physical claim is true.

**Cross-project rule:** do not collapse these outputs into one opaque “truth score”. Exchange typed records with explicit units, frame/epoch, uncertainty, exact source/build identity, evidence state and bounded claim scope.

## 6. Stage gates and evidence standard

### T0 — mission and comparison contract

Specify the mission type, particle species, energy target, luminosity or beam-current requirement, and Earth baseline. Define bounded LEO, Moon and Mars cases with source-linked assumptions.

**Fail closed:** if the science case or target energy is absent, site selection remains Blocked. Do not let missing inputs default to a preferred architecture.

### T1 — checked analytical envelope

Implement Rust-first, checked analytical functions for:

- bending radius and circumference;
- continuous and peak power scale with explicit duty cycle;
- ideal radiator lower bound with temperature/emissivity assumptions;
- a clearly bounded LEO differential-gravity reference case.

Use SI internally; reject non-finite, physically invalid and overflowed values; pin named fixtures and test monotonicity/property invariants. Keep the simplified reference models separate from full facility solvers.

### T2 — independent numerical verification

- Accelerator tracking, optics and beam-loss claims need established accelerator-physics tools, independent analytic fixtures, or measured data; they must not be “validated” only by Symthaea's own representations.
- Orbital propagation and frame transformations need independent comparisons using the same frame, epoch, units and gravity assumptions.
- Structural and thermal claims need separately validated solvers and boundary conditions.
- Store exact source commit/blob, SHA-256, input manifest, build/tool fingerprint, output artifact and independent-check identity.

A deterministic repeat of the same code demonstrates reproducibility, not independent physical correctness. A test fixture cannot be promoted to a real-world result.

### T3 — comparable site models

For Earth, LEO, Moon and Mars, estimate ranges for:

- facility geometry, subsystem mass and maintainability;
- peak and average power, storage and conditioning, heat-rejection area/mass;
- shielding, radiation, thermal environment, dust/debris and geotechnical assumptions;
- launch/transport/assembly, spare parts, repair time and availability;
- failure consequences, end-of-life disposition and major exclusions;
- lifecycle cost with source dates, sensitivity analysis and uncertainty bounds.

Do not issue a single headline cost until major unknowns and denominators are closed.

### T4 — prototype before megastructure

Validate the smallest experiment that addresses the key scientific uncertainty. Qualify orbital-ring dynamics and load transfer independently; qualify accelerator beam physics, diagnostics, power and thermal systems independently; then test their interface as a later gate. A ring-mounted accelerator must never be the first demonstrator.

## 7. First local-agent work package

1. Inspect the exact heads and current checks for Symthaea #7353, Symtropy #1573/#1575, Mycelix #4964/#4999/#5006, and Sol Atlas #16/#87 before modifying them. They are separate draft PRs; do not assume they are merged or have passed CI.
2. Add a Rust-first typed site-scenario schema with explicit units, frame/epoch where applicable, uncertainty, provenance, and Blocked/NotRun states. Unknown fields, incompatible units and non-finite values must fail closed.
3. Implement the four small analytical kernels from T1 with source-pinned fixtures and a standalone independent checker.
4. Add negative controls for malformed/non-finite values, unit/frame mismatches, inadequate power, impossible magnet/geometry combinations, thermal infeasibility and missing provenance.
5. Compare forecast and anomaly outputs against time-ordered reference traces and strong simple baselines before giving Symthaea model outputs a physical-forecast label.
6. Once numerical and evidence contracts are independently checked, add a read-only scenario view in Sol Atlas with visible status, uncertainty ranges and direct links to exact records.

## References

- CERN, [2026 European Strategy for Particle Physics update](https://home.cern/the-cern-council-decided-to-update-the-european-strategy-for-particle-physics/).
- CERN, [Future Circular Collider programme](https://home.cern/science/accelerators/future-circular-collider/).
- CERN, [FCC-ee power and design update material (2025)](https://indico.cern.ch/event/1439855/contributions/6461636/attachments/3076197/5443758/233-Update-FCC_ee_10_page-26052025-V2.pdf).
- NASA, [Fission Surface Power programme](https://www.nasa.gov/exploration-systems-development-mission-directorate/fission-surface-power/).
- NASA, [Lunar surface technologies and infrastructure call (September 2026)](https://www.nasa.gov/news-release/nasa-calls-for-proposals-to-accelerate-lunar-surface-technologies/).
- ESA, [Zero Debris technologies and orbital environment](https://www.esa.int/Space_Safety/Clean_Space/Zero_Debris_Technologies).
