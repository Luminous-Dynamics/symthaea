# Orbital-Ring Feasibility Workstream — T0 Contract

**Status:** research contract; not a flight design, validated simulation, or construction recommendation  
**Prepared:** 2026-10-10  
**Initial owner:** Symthaea engineering / orbital research  
**Decision rule:** every feasibility claim must name its architecture, reference frame, assumptions, model fidelity, independent check, and evidence state.

## 1. Purpose and scope

Establish a falsifiable, reproducible T0/T1 study for dynamic orbital-ring concepts. The first milestone is not to demonstrate that a planetary ring is buildable. It is to determine which combinations of geometry, rotor configuration, support forces, materials, power, and fault response are physically self-consistent, and to expose assumptions that make a candidate infeasible.

This work should reuse existing repository assets rather than build another general-purpose physics engine:

- `crates/domains/symthaea-orbital/`: spacecraft servicing, simplified orbital propagation, rendezvous, station-keeping, momentum limits, and failure scenarios.
- `crates/domains/symthaea-engineering/` and `symthaea-sim-bridge/`: requirements and simulation orchestration boundaries.
- `symthaea-formal-safety/`: proof-obligation/evidence vocabulary (currently primitives; attaching a reference does not independently validate the evidence).
- `symthaea-materials/` and `symthaea-fabrication-kernel/`: material properties, aging/manufacturability and geometry workflows.
- Mycelix's `orbital-mechanics` library and space-automation plan: validated-to-scope astrodynamics, catalog propagation, conjunction analysis, and an integration roadmap.
- Sol Atlas: visualization and scenario comparison; visualization must not be treated as proof.

The detailed scope is deliberately narrow. T0 is analytical screening and unit-consistent data models. T1 adds independently checked numerical models. Physical experiments and flight operations are later, separately gated phases.

## 2. Architecture must be explicit before any equation is evaluated

The phrase “orbital ring” is used for different mechanical arrangements. Never put results for one arrangement under another label.

Define a versioned `RingArchitectureMode` with at least:

1. **Orbiting ring / orbital structure:** the supporting structure follows an orbital trajectory. It does not remain over one ground point merely because it is ring-shaped.
2. **Stationary supported ring:** the intended structure is held in a defined non-Keplerian frame by dynamic support. Its inertial and Earth-fixed behavior must be specified independently.
3. **Partial launch loop:** a localized elevated track supported by a high-speed moving mass stream, following the Lofstrom family of concepts.
4. **Momentum-exchange tether / skyhook:** a separate tether architecture whose payload exchanges angular momentum and orbital energy with its host.
5. **Ring-mounted particle accelerator:** a separate beam-optics and accelerator-physics system, never implied by a transportation-ring model.

For each scenario, specify: Earth reference ellipsoid and gravity model; inertial frame (e.g. GCRF/ECI); Earth-fixed frame (e.g. ITRF/ECEF); local orbital/body frame if relevant; Earth rotation; ring-plane orientation and precession; the shell's inertial state; rotor's relative motion; tether endpoints and whether endpoints are ground-fixed, moving, or orbital; and all external forces.

**Blocking rule:** reject an input that leaves the architecture mode, reference frame, or tether endpoint semantics unspecified. A visualization may show an illustrative scene, but must label it as such and must not backfill missing physics silently.

## 3. T0 baseline calculations

Use SI internally: metres, kilograms, seconds, newtons, joules, pascals, radians. Use typed newtypes or an equivalent checked boundary; do not allow bare mixed-unit arrays into the authoritative model.

For a freely orbiting object in a circular two-body Earth orbit, with geocentric radius (r = R_E + h), gravitational parameter μ:

- Circular orbital speed: (v_c = \sqrt{\mu/r}\)
- Circular orbital period: (T = 2\pi\sqrt{r^3/\mu}\)
- Circumference of the chosen geometric circle: (C = 2\pi r\)
- Specific mechanical energy for a circular Kepler orbit: ε = −μ/(2r)

Use these equations only for the stated circular-orbit model. A stationary supported ring is not a circular Kepler orbit, so the equations do not establish its equilibrium or its required rotor speed.

For a curved moving mass stream, the local normal-force requirement is related to mass flow, curvature, and **inertial** velocity. A first-principles model must derive this from the actual control volume and velocity field. Do not substitute rotor speed relative to the sheath directly into (a_n=v^2/r) unless the reference frame and transformation justify it.

If a local continuous-stream approximation is used, record line density λ, curvature κ, relative velocity, support-frame velocity and acceleration, and control-volume assumptions. Then derive the vector acceleration/force balance in the chosen frame. In a rotating body frame this must account for transport terms (including Coriolis, centrifugal and angular-acceleration terms as applicable). Validate the resulting expression against a small, independently hand-checkable limiting case.

Energy and momentum are bookkeeping constraints, not optional secondary metrics:

- Rotor kinetic energy: (E_k = \frac12 m u^2) for a defined mass (m) and speed (u) in the same stated frame.
- Every payload launch/braking case must show where energy and momentum go: rotor, supporting structure, Earth, electrical store, propulsion, or another explicitly modelled body.
- A replenishment term must be external energy input, not a free “momentum reset.”
- Power loss, loss of magnetic support, loss of confinement, and rotor rupture are distinct fault classes. Loss of drive power does not instantaneously erase stored kinetic energy; nevertheless, failure of support/confinement can cause a severe mechanical transient.

T0 outputs: an equation register with units and source; reference-frame diagram; architecture assumptions; independent numerical checks; and a parameter envelope, not a single preferred design.

## 4. Baseline scenario set

Start with deliberately small scenarios to make the failure behavior legible:

- **S0 — Geometry sanity:** verify radius, circumference, altitude, and frame conversions. Include negative tests for km/m and inertial/Earth-fixed mix-ups.
- **S1 — Circular-orbit baseline:** verify Kepler speed/period against a separately implemented formula and existing astrodynamics tools. Converge step size before using the propagator to score control.
- **S2 — Moving-stream kinematics:** use a toy circular guide and an independently derived force balance. Test zero relative speed, same-direction/opposite-direction relative speed, and a stationary-support limiting case.
- **S3 — Equilibrium residual:** report the vector sum of modeled gravitational, support, tension/contact, electromagnetic, and external forces. Do not label a candidate “stable” because an optimizer found a small residual.
- **S4 — Linearized local stability:** only after equilibrium is independently established, evaluate the Jacobian/eigenmodes and state which omitted physics could change those modes.
- **S5 — Power and thermal budget:** account for mechanical losses, electrical losses, conduction/cooling assumptions, redundancy, and rejection of waste heat. Unknown values are intervals, not zeros.
- **S6 — Payload energy/momentum ledger:** run launch and braking symmetrically and report energy/momentum residuals.
- **S7 — Fault injection:** support actuator stuck on/off, sensor bias/dropout, control latency, power-bus failure, local support loss, rotor speed/position disturbance, impact damage, and communications loss. Report transient peak loads and whether isolation/recovery remains physically possible.
- **S8 — Debris exposure:** state altitude, inclination, epoch, catalog source, untracked-debris model, cross-section, shielding assumptions, and uncertainty. A point-in-time catalogue is not a long-term collision-rate estimate.
- **S9 — Architecture comparison:** apply the same throughput, safety, availability, power, mass, maintenance, and cost-accounting conventions to an orbital ring, launch loop, conventional launch, orbital tethers and modular in-space assembly.

For Monte Carlo studies, freeze the seed, input distribution, sample count, model version, and failed-run policy. Failed and numerically divergent runs must remain in the denominator and be reported explicitly.

## 5. Fidelity ladder

| Level | Permitted claim | Minimum evidence |
|---|---|---|
| T0 — analytical screening | “This equation/assumption is internally consistent under stated conditions.” | Dimensional checks, derivation, edge cases, independent arithmetic |
| T1 — numerical subsystem | “This specified subsystem model passes these verification cases.” | Convergence, benchmark comparison, independent implementation or trusted solver comparison |
| T2 — coupled model | “These subsystems interact as modelled over this bounded envelope.” | Coupling tests, conservation/error budget, uncertainty/sensitivity, independent review |
| T3 — laboratory article | “This article matched these predictions in these tests.” | Calibration, raw data, test configuration, repeatability, instrument uncertainty, failure records |
| T4 — representative environment / HIL | “This system demonstrated bounded behavior on this hardware in these environments.” | Hardware-in-the-loop evidence, fault injection, timing/thermal/EMC evidence as applicable |
| T5 — flight/infrastructure qualification | No claim until a specific mission's independent qualification is complete. | Approved requirements, validated models, complete safety case, independent authorities, operational approvals |

Passing one level never automatically passes the next. A simulation pass is not a materials qualification, test campaign, flight approval, or construction feasibility result.

## 6. Evidence contract and fail-closed gates

Each run artifact must bind to:

- immutable scenario identifier and schema version;
- exact source commit, model and solver versions, dependency lock state, and solver configuration;
- canonical SI input file and SHA-256 digest;
- input provenance, units, uncertainty distributions, and explicit assumptions;
- command line / entry point, environment, start/stop status, exit code, logs, and raw outputs;
- numerical residuals, convergence record, error bounds, warnings, and all failed samples;
- independent verification result and verifier version;
- claim-to-evidence map and reviewer disposition.

Use separate states such as `NotRun`, `Running`, `PassedAtSpecifiedFidelity`, `Failed`, `Inconclusive`, and `ReviewRequired`. Never infer `Passed` from an artifact existing, a file name, a non-empty log, an optimizer's objective score, a CI job being queued, or a solver process exiting successfully.

Block advancement when units/frames are unknown; the scenario differs from its frozen manifest; a required solver is a dry-run/mock adapter; an external solver output fails parsing; convergence is not demonstrated; the verification model is not independent for the claim being tested; or a blocking safety obligation remains open. If an assumption is knowingly unmodelled, downgrade the allowed claim rather than silently assuming it is safe.

For critical control, keep Symthaea's cognitive planner outside the final safety authority. Use a small bounded runtime-assurance layer, physically justified safe states, actuator-level protection where software commands cannot stop a failed actuator, and independently tested fallback modes. “Set all commands to zero” is not a universal safe response for a dynamically supported structure.

## 7. Implementation sequence

### Increment A — documentation and data contract (this PR)

- Record the architecture distinctions and block mode/frame ambiguity.
- Define the minimal scenario/evidence contract and permitted claim ladder.
- Pin references and known repository limitations.

### Increment B — pure Rust T0 kernel

Create a small feature-light crate or module for units, architecture-mode types, parameter validation, baseline orbital geometry, kinetic-energy ledger, and residual reporting. Keep it deterministic and independent of Symthaea cognitive state. Do not add a free-form rotor equilibrium formula until the derivation and reference frame are reviewed.

Required tests: unit conversions; positive/finite parameter validation; circular-orbit reference cases; frame/mode mismatch rejection; conservation-ledger residual tests; boundary and property tests; deterministic serialized fixtures.

### Increment C — independent model check

Recompute the orbital baseline using an independently implemented analytical reference and GMAT/Orekit/Basilisk where their model scope is appropriate. For rotor/support, validate a deliberately simplified test geometry with a separate implementation before coupling it to orbit dynamics. A shared dependency is not an independent check.

### Increment D — structural/electromagnetic/thermal interfaces

Only after Increment B/C passes, specify solver input/output schemas for structural dynamics, electromagnetic support, rotor dynamics, power electronics and thermal rejection. Treat adapters as unavailable until each invokes a real solver, checks the exit status, parses a known fixture, records versions, and passes validation tests. A dry-run adapter may test orchestration, but its output must not satisfy a physics evidence obligation.

### Increment E — fault and uncertainty campaign

Add bounded disturbances, component faults, sensitivity analysis, and Monte Carlo experiments with preregistered metrics. Include common-mode faults and correlated uncertainty; report tail outcomes and failed model runs. Do not optimize only for nominal stability.

### Increment F — hardware demonstrator decision

Propose a lab rig only after the model identifies a low-energy, contained, safely testable question whose outcome is useful even if the megastructure concept fails. Separate test approval, instrumentation calibration, emergency shutdown, containment, and operator procedure from software checks.

## 8. Initial acceptance criteria

The first executable prototype is accepted only if all are true:

1. Architecture mode, coordinate frames, gravity model, units, epoch, and tether-endpoint semantics are required fields or explicitly marked not applicable.
2. Invalid/non-finite values and incompatible frames fail closed.
3. Circular-orbit formulas agree with an independent reference within a declared tolerance.
4. Each computed quantity carries its units and source assumptions in the output.
5. Energy and momentum ledgers expose unmodelled source/sink terms; unexplained residuals fail the relevant check.
6. Evidence status is distinct from solver process status; a dry-run result cannot pass a physics check.
7. The scenario manifest and each artifact digest are stable across repeat runs with the same pinned inputs.
8. The report plainly distinguishes verified calculations, modeled assumptions, unknowns, and speculation.

This increment must not claim that an orbital ring is stable, cheap, safe, buildable, able to provide “infinite” launch, or suitable for a particle collider.

## 9. Research references and rationale

- Paul Birch, “Orbital Ring Systems and Jacob’s Ladders,” *Journal of the British Interplanetary Society* 35 (1982), pp. 475–497. The concept explicitly spans multiple ring orientations and configurations; do not reduce that family to one ambiguous geometry. Copy of paper: https://nss.org/wp-content/uploads/2017/07/Orbital-Rings.pdf
- K. H. Lofstrom, “The Launch Loop — A Low Cost Earth-to-High-Orbit Launch System,” AIAA Paper 85-1368 (1985). Preliminary design of a different architecture: https://ntrs.nasa.gov/api/citations/19860018633/downloads/19860018633.pdf
- NASA Orbital Debris Program Office, FAQ. Reports that catalogued debris is not uniformly distributed and that typical LEO object-object impact speeds are around 10 km/s, with impacts up to about 15 km/s: https://orbitaldebris.jsc.nasa.gov/faq/
- NASA OSAM-1 status page. Illustrates why robotic on-orbit servicing needs hard cost/schedule/technical and partner gates; it records the decision to cancel OSAM-1 and proceed with shutdown, not a proof that all ISAM is infeasible: https://www.nasa.gov/mission/on-orbit-servicing-assembly-and-manufacturing-1/
- CERN Future Circular Collider programme. A concrete reminder that accelerator feasibility depends on magnets, beam energy, luminosity, synchrotron radiation, heat removal, and machine protection—not circumference alone: https://fcc.cern/accelerators
- Symthaea's existing engineering roadmap: https://github.com/Luminous-Dynamics/symthaea/blob/main/docs/engineering/SYMTHAEA_ENGINEERING_ROADMAP.md
- Symthaea orbital model scope: https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-orbital/README.md
- Monorepo space-automation plan and candid maturity assessment: https://github.com/Luminous-Dynamics/luminous-dynamics/blob/55757dfac7791d6d5d4733f7398f4ca17af215d0/symthaea/SPACE_AUTOMATION_PLAN_2026-07-06.md

## 10. Decision requested from reviewers

Review the architecture-mode taxonomy, frame contract, T0 equations, conservation ledger and fidelity gates before implementation begins. The key review question is: **are the modelled geometry and reference frames sufficient to uniquely define the physical system whose equilibrium is being calculated?** If not, revise the contract; do not patch the ambiguity in code.

