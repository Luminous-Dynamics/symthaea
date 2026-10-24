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

For a freely orbiting object in a circular two-body Earth orbit, with geocentric radius \(r = R_E + h\) and gravitational parameter \(\mu\):

- Circular orbital speed: \(v_c = \sqrt{\mu/r}\)
- Circular orbital period: \(T = 2\pi\sqrt{r^3/\mu}\)
- Circumference of the chosen geometric circle: \(C = 2\pi r\)
- Specific mechanical energy for a circular Kepler orbit: \(\epsilon = -\mu/(2r)\)

Use these equations only for the stated circular-orbit model. A stationary supported ring is not a circular Kepler orbit, so the equations do not establish its equilibrium or its required rotor speed.

Using mu = 3.986004418e14 m^3/s^2 and an equatorial reference radius of 6,378.137 km gives this independently evaluated *two-body circular-orbit* sanity table:

| Altitude h (km) | Radius r (km) | Circumference (km) | Circular speed (km/s) | Period (min) | Kinetic energy per kg (MJ/kg) | Specific mechanical energy per kg (MJ/kg) |
|---:|---:|---:|---:|---:|---:|
| 200 | 6,578.137 | 41,331.65 | 7.78426 | 88.494 | 30.297 | -30.297 |
| 300 | 6,678.137 | 41,959.97 | 7.72576 | 90.520 | 29.844 | -29.844 |
| 400 | 6,778.137 | 42,588.29 | 7.66856 | 92.560 | 29.403 | -29.403 |
| 500 | 6,878.137 | 43,216.61 | 7.61261 | 94.616 | 28.976 | -28.976 |
| 600 | 6,978.137 | 43,844.93 | 7.55787 | 96.687 | 28.561 | -28.561 |

These numbers are reference fixtures for formula/unit tests only. They omit atmosphere, oblateness, lunar/solar perturbations, inclination, attitude, and all ring-support physics; they are **not** a claim that any listed altitude is suitable for a long-lived infrastructure ring. The table's kinetic-energy column is the ideal orbital specific kinetic energy, not the rotor's energy and not the energy required to build, deploy, or support a ring.

For a curved moving mass stream, the local normal-force requirement is related to mass flow, curvature, and **inertial** velocity. A first-principles model must derive this from the actual control volume and velocity field. Do not substitute rotor speed relative to the sheath directly into \(a_n = v^2/r\) unless the reference frame and transformation justify it.

If a local continuous-stream approximation is used, record line density λ, curvature κ, relative velocity, support-frame velocity and acceleration, and control-volume assumptions. Then derive the vector acceleration/force balance in the chosen frame. In a rotating body frame this must account for transport terms (including Coriolis, centrifugal and angular-acceleration terms as applicable). Validate the resulting expression against a small, independently hand-checkable limiting case.

Energy and momentum are bookkeeping constraints, not optional secondary metrics:

- Rotor kinetic energy: for absolute conservation accounting, calculate \(E_k = \frac{1}{2}m\lVert\mathbf{v}_{\mathrm{inertial}}\rVert^2\) using the rotor mass's velocity in one stated inertial bookkeeping frame (for example, GCRF). Relative-to-sheath kinetic energy is a useful subsystem diagnostic, but it is frame-dependent; do not add it to inertial-frame energy terms without transforming velocities and accounting for support-frame motion.
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

**C1 implemented: independent T0 arithmetic and limiting-case checker (not full Increment C).** The standard-library Python program `scripts/engineering/orbital_ring_reference_check.py` independently evaluates the four published circular-orbit fixtures, checks each table entry within half a unit of the last displayed decimal place, compares the 400 km case to the existing Rust unit-test numeric anchors, and tests three algebraic consistency relationships. It also checks a distinct, hand-checkable moving-stream limit:

\[
q_n = \lambda v_{\mathrm{inertial}}^2 / R
\]

For a steady mass stream on a fixed circular path in an inertial frame, using line density `1 kg/m`, inertial speed `3 m/s`, and path radius `10 m`, the net normal resultant is `0.9 N/m`. Tests cover zero speed, quadratic speed scaling, inverse-radius scaling, line-density scaling, and invalid/non-finite inputs. This is the **net normal resultant per unit path length**; it does not assign the load to a magnetic actuator, include a moving support frame or gravity, or establish orbital-ring equilibrium.

Command from the repository root:

```sh
python3 scripts/engineering/orbital_ring_reference_check.py
```

The prior v1 reference-checker source (Git blob `7adbaab223517494332711b480bdf37e0b216811`, SHA-256 `3ab051fc31ba1b4673c07578cfd047a0ce43339a4f8bf37de86992e8588953dc`) ran **47/47 checks passed** with byte-identical repeat output. Current v3 adds the source-bound 600 km row plus total specific mechanical-energy output and the invariant \(\epsilon=e_k-\mu/r=-\mu/(2r)\): Git blob `a7c39906e9f5cf34711630e1813e4b4f67527eef`, SHA-256 `c4eabd9a2e02f246d7d337926b9a8f0e422f1eced879d1ed57c2c26b1ffc98c2`, **66 checks expected; not yet runtime-verified**. It remains a T0 arithmetic/limiting-case check, not a solver comparison, independent peer review of the physical derivation, or T1 validation. See [the source-bounded historical geometry reference](ORBITAL_RING_HISTORICAL_GEOMETRY_REFERENCE.md). The symbolic derivation and fields required before coupling the model are documented in [the control-volume note](ORBITAL_RING_CONTROL_VOLUME_NOTE.md). The source-audit, failure, structural-stability and debris questions are tracked separately in the [research gap register](ORBITAL_RING_RESEARCH_GAP_REGISTER.md).

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

- Paul Birch, “Orbital Ring Systems and Jacob’s Ladders,” *Journal of the British Interplanetary Society* 35 (1982), pp. 475–497. The concept spans multiple ring orientations and configurations. Birch discusses stability to small perturbations for one discrete-skyhook arrangement, but also says the exact behaviour and stability limits of a particular system need further work. This is theoretical analysis, not experimental validation. Do not reduce this family to one ambiguous geometry. Copy of paper: https://nss.org/wp-content/uploads/2017/07/Orbital-Rings.pdf
- K. H. Lofstrom, “The Launch Loop — A Low Cost Earth-to-High-Orbit Launch System,” AIAA Paper 85-1368 (1985). The NASA-indexed paper describes a preliminary design of a different architecture, not an operationally demonstrated system: https://ntrs.nasa.gov/api/citations/19860018633/downloads/19860018633.pdf
- NASA Orbital Debris Program Office, FAQ. Catalogued debris is not uniformly distributed; the FAQ reports typical LEO object-object impact speeds around 10 km/s, potentially reaching 15 km/s. This makes debris flux, impact response, and maintenance core design inputs: https://orbitaldebris.jsc.nasa.gov/faq/
- NASA OSAM-1 status page. NASA's published history says the project was discontinued after technical, cost and schedule challenges, a change in the servicing market, and no committed transition partner; the mission page records an orderly shutdown. This is a reason to stage demonstrations and use explicit investment gates, not proof that all in-space servicing or assembly is infeasible: https://www.nasa.gov/mission/on-orbit-servicing-assembly-and-manufacturing-1/
- CERN Future Circular Collider programme. The FCC remains a proposed research infrastructure and feasibility programme; its published design aims at 100 TeV proton-proton collisions and identifies substantial technology development needs. This is a reminder that accelerator feasibility depends on beam energy, luminosity, magnets, radio-frequency systems, radiation, heat removal and machine protection—not circumference alone: https://fcc.cern/accelerators
- Symthaea's existing engineering roadmap: https://github.com/Luminous-Dynamics/symthaea/blob/main/docs/engineering/SYMTHAEA_ENGINEERING_ROADMAP.md
- Symthaea orbital model scope: https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-orbital/README.md
- Monorepo space-automation plan and candid maturity assessment: https://github.com/Luminous-Dynamics/luminous-dynamics/blob/55757dfac7791d6d5d4733f7398f4ca17af215d0/symthaea/SPACE_AUTOMATION_PLAN_2026-07-06.md
- [Orbital-ring research gap register](ORBITAL_RING_RESEARCH_GAP_REGISTER.md): source-audit priorities, failure-energy questions, debris exposure, and explicit evidence states.
- [Failure-mode evidence note](ORBITAL_RING_FAILURE_MODE_NOTE.md): Part II source findings on skyhook load steps, lift-coil loss, cryogenic quench, inter-ring riders, and distinct ladder-collapse hazards.
- [Load-step input/evidence ledger](ORBITAL_RING_LOAD_STEP_INPUT_LEDGER.md): definitions, units, evidence-class requirements and closure criteria for all 23 scenario inputs.
- [F-SKYHOOK-LOAD-STEP manifest](scenarios/F_SKYHOOK_LOAD_STEP.json): 23 explicit required fields; correctly blocked until geometry, initial equilibrium, rider force law and boundary ledgers are specified.
- [Fail-closed manifest validator](../../scripts/engineering/validate_orbital_ring_failure_manifest.py) and [its 33-check report](evidence/orbital_ring_failure_manifest_validator_selftest.json): completeness/type gates only, no physics execution.
- [Birch power source audit script](../../scripts/engineering/orbital_ring_power_source_audit.py): repeatable arithmetic check distinguishing eddy-current drag from cryostat electrical power.
- [Power-budget audit note](ORBITAL_RING_POWER_BUDGET_AUDIT.md): source page references, derivation, results, and open reconstruction tasks.
- [Captured power audit JSON](evidence/orbital_ring_power_source_audit.json): deterministic output from the 20-check run.

## 10. Decision requested from reviewers

Review the architecture-mode taxonomy, frame contract, T0 equations, conservation ledger and fidelity gates before implementation begins. The key review question is: **are the modelled geometry and reference frames sufficient to uniquely define the physical system whose equilibrium is being calculated?** If not, revise the contract; do not patch the ambiguity in code.



## 11. Implementation ledger (2026-10-10)

### Increment A — contract and fidelity gates

**Implemented on the research branch.** The contract distinguishes the relevant architecture families, defines SI and frame boundaries, documents the ideal circular-orbit fixtures, and establishes evidence/fidelity gates.

### Increment B — pure Rust T0 kernel

**Source added; runtime verification pending.** The new workspace crate `crates/domains/symthaea-orbital-ring` currently provides:

- checked `Meters`, `Kilograms`, `Seconds`, and `Joules` boundary types;
- scenario metadata for architecture, frame, gravity model, drag state, and altitude;
- closed-form circular-orbit values for a freely orbiting structure under the Earth point-mass model only;
- an energy ledger and momentum ledger with explicit frame requirements, residuals, tolerances, and fail-closed rejection of non-finite residuals;
- 11 unit tests covering reference fixtures, monotonic trends, invalid assumptions, unsupported modes, balance/residuals, overflow, and determinism;
- workspace engineering-tier registration and a crate README with intended local verification commands.

The kernel intentionally does not implement rotor/support equilibrium, electromagnetic force, structural dynamics, a digital twin, or a validated solver adapter.

### Increment C1 — executable T0 cross-check

**Implemented and executed locally; runtime verification applies to this Python checker only.** The committed script checks the published four-altitude circular-orbit table, the four 400 km Rust numeric anchors, algebraic identities, altitude-grid trends, and the inertial circular-stream limiting case with scaling and input guards.

- Result: **47 passed, 0 failed**.
- Repeated-run determinism: output was byte-identical.
- Source SHA-256: `f978fef9b474f9eb304c7a55130eca4314928f3483e273bdb3a7d6f96203681d`.
- Deterministic JSON report SHA-256: `2bdf854edc398f087ef487a4a0608c434331d3642813e5ad3332781a32676080`.
- The locally executed source's Git blob SHA matches the GitHub file blob SHA, confirming that the committed source is the one checked locally.

### Increment RG-02a — Birch power source audit

**Arithmetic source audit implemented and executed locally; complete physical/source reconstruction remains open.** The dependency-free script `scripts/engineering/orbital_ring_power_source_audit.py` re-evaluates the p. 482 conducting-sheet example and larger precessing-force case using Birch's stated aluminium conductivity and a 5 cm sheet. It computes **0.17984 GW** for the rounded 2 × 10⁸ N example force and **449.59 GW** for the rounded 5 × 10¹¹ N force; the latter is within approximately 10% of the paper's “around 500 GW” statement.

The script separately records the p. 483 pair (about 100 MW cold-side load / 10 GW cryostat power) and the Table 8 pair on p. 485 (5 × 10⁸ W / 5 × 10¹⁰ W). Both imply 100:1, but the passages describe different superconducting configurations and should not automatically be combined. Table 8 leaves the orbital-ring cryogenic-loss cell blank and places 10 W/m under the geosynchronous/support column; this matches p. 484's statement that the orbital rings need no active cryogenic equipment because they sit inside the fixed ring's vacuum flask. The mass-per-length rows imply **44.17 Mm** for the precession/orbital-ring entry and **44.00 Mm** for the support entry. Only the support length should be multiplied by 10 W/m, yielding **440 MW**, within about 12% of the rounded 500 MW table value. Its circular-equivalent altitude is about **625 km**, with circumference 0.35% longer than a nominal 600 km circular path. This is a table-consistency check, not proof of exact geometry; the distinct p. 483 100 MW / 10 GW alternative remains to be reconciled. The script also records the conditional 17.92 PJ mass/speed arithmetic.

- Result: **20 passed, 0 failed**.
- `py_compile` succeeded; a repeat run produced byte-identical JSON.
- Source Git blob SHA: `b6406e391e49d53cd077cf08b3d8691c40472611`; source SHA-256: `80c716c0f374de4acb706a7aac5ca467e00e285dcb99e6f97b403f90eb9c6a0b`.
- Exact output SHA-256: `d04091d63db550b2ce771a6d06fbbfe61998ce2c3806488dea53bd12e39e8165`; report Git blob: `0dd0e85ce020e874163d7dcc63fa0e209d4c3558`.

This is an arithmetic cross-check of transcribed source figures, not evidence that Birch's model is physically accurate, a complete reconstruction of the ring geometry, or a present-day feasibility result. Conducting-sheet eddy-current drag and superconducting cryostat power are different categories and must not be added or substituted without a consistent scenario definition.

### Increment RG-03a — support-load scenario gate

**A fault-scenario contract now exists; the physics case remains blocked.** The template `docs/engineering/scenarios/F_SKYHOOK_LOAD_STEP.json` records the Part II source basis and names 23 required inputs. The validator `scripts/engineering/validate_orbital_ring_failure_manifest.py` checks missing/unknown inputs, field-level provenance, rejection of test-fixture/rejected evidence, typed vectors, support-frame from/to frames and epoch, support-frame quaternion, unit ring-plane normal, geometric/load consistency, step/ramp consistency, rider parameters, and explicit energy/momentum term inventories. The template intentionally has 23 unresolved inputs, `declared_state = Blocked`, `analysis_state = NotRun`, and no result artifacts.

- Local syntax check: `python3 -m py_compile scripts/engineering/validate_orbital_ring_failure_manifest.py` succeeded.
- Last executed validator self-tests: **73 passed, 0 failed** against source blob `2d07b86480dea71f3272fbee2a9fde4152727abb`. The current source adds twelve fail-closed regressions for malformed top-level objects, malformed/unknown keys, malformed field-list types, wraparound duplicate angles, duplicate JSON object keys, and non-standard `NaN` tokens; they are not yet runtime-verified. Do not claim the expanded suite passed.
- Template assessment: contract validation `passed_contract_checks`; expected/declared scenario state remains **Blocked**, with 23 missing fields, zero type errors in the null-only template, and no physics claim.
- Both emitted JSON reports were byte-identical on repeat and are preserved under `docs/engineering/evidence/`.
- Current validator Git blob SHA: `734fae84c6d5b1fb7b895b56b021a75892546206`; source SHA-256: `532de2bb5707db127fa9a031e233e25100673491e658d0b2173845bfeebf85f4`.
- Template Git blob SHA: `a65250ba7284b503e1effb804a3a30668445017e`; SHA-256: `60c747bf226f436c28f08ca047e77faae2368e200bf72f600e3b84f7dd5234b1`; validation report blob: `8d0f200eb2db33c4c4ef6b8dc07ef98d21e184d6` (SHA-256 `0aa07355674fb104d0e9bb38ad3bafad3cd768b3557e4bfe012126b25f8c253c`); self-test report blob: `87612c32dfe45c3c0fb8b8588716b35c0457aa67` (SHA-256 `dc7718642fc57f35ca813fb3be857194c9ec74d9bc9413db9810d8a1432367c5`).

A green manifest-contract result does **not** mean model inputs are physically valid, that a transient ran, or that the ORS is safe. The next gate is independent source-backed specification of those inputs, not automatic transition to ready.

### Checks actually performed

- Read back the exact branch versions of all eighteen changed files.
- Confirmed the crate has no external dependencies and its location matches the root workspace's `crates/domains/*` member glob.
- Confirmed the new crate appears in the workspace engineering metadata and the engineering roadmap links to this contract.
- Checked balanced delimiter counts in the Rust source and enumerated the 11 Rust test functions.
- The prior reference checker independently recomputed four documented circular-orbit rows; v2 adds the fifth (600 km) row based on Birch's primary-source representative height, with current checks pending execution.
- Ran the column-aware power-source audit (20/20 checks) and repeated it with byte-identical JSON output; local copy's Git blob SHA matched the branch source.
- The last executed failure-manifest self-test (73/73) applies only to the prior validator blob. The current source fingerprint and not-run status are recorded in [validator verification freshness status](evidence/orbital_ring_validator_verification_status.json).

### Checks not performed

The present execution environment does not have `cargo` installed. The Rust tests, rustfmt, Clippy, and a workspace Cargo resolution/check have **not** been executed in this environment. No CI pass, solver pass, structural pass, or feasibility claim is made. The README lists the local commands required to perform those checks.

### Next engineering increment

Continue analytical work in parallel with the outstanding Rust toolchain gate. The scenario contract now defines **what must be known**, but the load-step simulation is still blocked by design: do not fill the 23 inputs with placeholders or convert the validator's contract pass into a physics pass. The next process increment is independent specification/review of the initial equilibrium, support and rider topology, force law, frame transforms, disturbance profile, and energy/momentum boundaries. Once those are fully specified, derive a limiting case independently and review the acceptance metrics before implementing any transient solver. Keep the p. 483 superconducting-surface estimate distinct from Table 8's continuous-skyhook case. No stronger fidelity claim is allowed without a suitable independent solver or reviewer.

The load-step contract is now schema/version v2 and declares `initial_path_constraint=circular_reference_path`; a populated initial state must bound its radial velocity by `acceptance_limits.max_initial_radial_velocity_m_s`. This adds a dimensional, field-level consistency gate without treating tangency as proof of force balance or stability.

Strict JSON parsing rejects duplicate object member names and `NaN`/`Infinity` constants; these forms are unsuitable for canonical, cross-implementation evidence manifests (RFC 8259 §4 and §6: https://www.rfc-editor.org/rfc/rfc8259.html). These checks protect manifest interpretation, not physical correctness.

A dedicated draft-safe Python workflow now lives at `.github/workflows/orbital-ring-manifest-contract.yml`, separate from the broad `.github/workflows/ci.yml` Rust matrix. Before testing, it compares the live validator Git blob and SHA-256 with `evidence/orbital_ring_validator_verification_status.json` and checks the expected 84-test count. It then requires `py_compile`, two byte-identical self-test outputs, 84 unique passing check IDs, and exact equality between generated validation JSON and the committed blocked-template report. The workflow archives raw outputs and source fingerprints. The isolated workflow was discovered and queued as run #3; it is not yet a pass. The twelve regression cases added after the historical 73-check run remain explicitly unverified until it executes.

Structural/schema errors now also force the expected gate state to `Blocked`, rather than allowing a malformed complete-looking manifest to report `ReadyForIndependentReview`. The expanded contract has 85 expected tests; the current source remains unexecuted pending the draft-safe workflow.

The 300 km and 600 km heights are explicitly listed by Birch as representative ORS heights on printed p. 476. The new 600 km two-body fixture is a reference comparator, not a claim that the real ring path is circular or that either altitude is a feasible design. The v3 checker reports kinetic and total specific orbital mechanical energy and independently checks \(\epsilon=e_k-\mu/r=-\mu/(2r)\). It uses NGA WGS 84 constants and the isolated workflow is configured to require 66 unique checks; until that job passes on the exact source blob, the count remains expected, not evidence of execution.

A further fail-closed guard handles declared-state mismatch: if the manifest claims `Blocked` while the complete synthetic fixture would otherwise be review-ready, the validator reports a structural error and forces `expected_gate_state=Blocked`. The 85-check suite is still expected, not a claimed pass.
