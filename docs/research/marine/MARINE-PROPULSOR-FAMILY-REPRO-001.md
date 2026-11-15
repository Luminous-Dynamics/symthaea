# MARINE-PROPULSOR-FAMILY-REPRO-001
## Matched-condition architecture and toroidal design benchmark

**Status:** research protocol v0.1.0  
**Created:** 2026-11-15  
**Owner:** Symthaea research coordination; physical solvers and experimental qualification remain with their existing owners  
**Tracking issue:** [#12392](https://github.com/Luminous-Dynamics/symthaea/issues/12392)  
**Parent programmes:** [HYDRO-CO-DESIGN-001 #8082](https://github.com/Luminous-Dynamics/symthaea/issues/8082), [MARINE-WAPS-CO-DESIGN-001 #11406](https://github.com/Luminous-Dynamics/symthaea/issues/11406)  
**Reference ledger:** [marine-propulsor-reference-ledger-2026-11-15.json](./marine-propulsor-reference-ledger-2026-11-15.json)  
**Ledger schema:** [marine-propulsor-reference-ledger.schema.json](./marine-propulsor-reference-ledger.schema.json)

> This is a research and benchmark contract. It is not a validated propulsor design, a CFD result, an experimentally demonstrated Luminous performance improvement, or a vessel approval.

## 1. Objective

Compare marine propulsor architectures under a common vessel, hull, mission, operating envelope and evidence boundary. The optimization target is **independently supported delivered service at minimum whole-system energy and lifecycle burden**, not a single peak open-water efficiency value.

Question:

> For the same vessel and delivered service, which applicable hull–wake–propulsor–drive configuration offers the strongest robust Pareto frontier across power, noise, cavitation, availability, maintenance, payload and lifecycle cost?

This programme explicitly treats toroidal propellers as an important candidate family, not a presumed winner.

## 2. Research evidence and comparator integrity

The dated machine-readable ledger is intentionally conservative. Each claim is bound to its named comparator, operating condition and evidence class. It distinguishes:
- direct experimental observations;
- CFD methods compared with experimental data;
- CFD-only self-propulsion predictions;
- surrogate/ML screening results;
- thesis/research prototypes.

A result with a different comparator is a separate claim. In particular, Liu et al.'s 7.26% power reduction is reported against its prototype at matched thrust, while its 1.74 dB noise reduction is reported relative to the best sampled design. These must not be combined into a single "7.26% more efficient and 1.74 dB quieter than a conventional propeller" statement.

The January 2026 DTC/container-ship toroidal study is a necessary counterexample: it reports greatly reduced tip-vortex cavitation while also reporting slightly lower thrust and efficiency than the conventional reference. A benchmark that records only favorable outcomes is invalid.

Raw datasets, code, geometry, and redistribution rights are not assumed to be available merely because a paper is public. The ledger marks rights and artifact availability for review. A metric reported in a paper is not the same thing as an independently reproduced measurement.

## 3. Architecture taxonomy

The architecture menu is class- and mission-dependent:

| Family | Required variants / design dimensions | Main applicability questions |
|---|---|---|
| Conventional open propeller | Fixed pitch, controllable pitch, blade count, diameter, pitch distribution | Incumbent baseline, broad operating envelope |
| Ducted/nozzle propeller | Fixed/steerable duct, duct section, clearance, fixed/controllable pitch | Bollard pull, towing, low-speed thrust versus high-speed duct drag |
| Contra-rotating/coaxial | Rotor spacing, rotation ratio, blade count, gearing/motor arrangement | Wake recovery, torque cancellation, added machinery and interaction losses |
| Toroidal/closed-loop | Loop topology, fore/aft blade interaction, blade count, pitch/vertical angle, section and thickness | Vortex/cavitation/noise behavior, torque, manufacturability and hull-wake sensitivity |
| Waterjet | Inlet, pump, impeller, stator, nozzle, steering and reverse system | High-speed/shallow-draft applicability and complete installed-system efficiency |
| Cycloidal / Voith-Schneider | Blade path, pitch law, disk loading, control and drive | Low-speed thrust vectoring and manoeuvring; not an open-water screw substitute |
| Rim-driven / distributed electric | Motor topology, stator/rotor integration, cooling, cables, redundancy | Installation, electrical losses, maintainability and redundancy |
| Flexible / compliant / composite blades | Material, stiffness distribution, passive deformation, structural damping | Orthogonal blade design dimension affecting pitch, fatigue, force and noise |

Flexibility is an orthogonal design variable, not a universal architecture class. Likewise, a waterjet or cycloidal propulsor should not be forced into a deep-sea container-ship comparison where its installation is not applicable. Exclusion requires an explicit mission/applicability rationale.

## 4. Qualification ladder

### L0 — source, rights and data reproducibility

Freeze paper revision, DOI, publisher/artifact links, rights status, geometry origin, test rig, scale, conditions, sensors, calibration, uncertainty, preprocessing, code revision and exclusions. Record unavailable data as unavailable. Do not digitize plots and silently promote the reconstructed points to raw measurements.

### L1 — open-water characterization

Preserve full curves and uncertainty across the declared operating envelope. Minimum observables:
- thrust coefficient (K_T), torque coefficient (K_Q), and advance coefficient (J);
- shaft power and open-water efficiency η_0;
- Reynolds number, rotation rate, diameter and reference velocity;
- blade/shaft unsteady forces;
- cavitation inception/type/extent under declared cavitation number;
- frequency-resolved acoustic observations where measured.

A peak efficiency at one (J) does not establish superiority over another architecture across a mission.

### L2 — non-uniform inflow and wake

Use the same frozen propeller-plane coordinate frame and wake field. Preserve axial, tangential and radial components, angular origin, blockage, spatial quadrature and operating condition. Evaluate wake harmonics, thrust/torque fluctuation, pressure pulses, cavitation intermittency and acoustic response. Consume the existing [MARINE-WAKE-FIDELITY-001 #11729](https://github.com/Luminous-Dynamics/symthaea/issues/11729) contract; do not implement another wake evaluator.

### L3 — hull/propulsor self-propulsion

Hold hull/appendage identity, geometry revision, roughness/fouling state, displacement, trim, draft, speed, water properties and environmental condition fixed. Report:
- thrust deduction and wake fraction;
- relative rotative efficiency and delivered shaft/electrical power;
- propeller rpm, torque, thrust and operating point;
- solver, discretization, model-scale and full-scale transfer uncertainty.

A propulsor change must not silently change the hull, rudder, motor, reduction gear, drive boundary, control law or loading condition.

### L4 — mission and lifecycle

Evaluate delivered cargo work and service quality over the route or annual duty cycle. Include whole-vessel energy, fuel/GHG boundary, payload displacement, availability, degradation, maintenance, spares, inspection, repair, underwater noise and lifecycle cost. Consume existing mission-energy, robustness, WAPS and environmental owners.

### L5 — independent physical confirmation

Use preregistered A/B trials where practical, repeated independent runs, synchronized raw traces, calibrated sensors, exact configuration records and retained failed/aborted trials. The trial plan must be blinded where feasible and must specify the analysis before candidate results are unsealed. Keep model-scale, full-scale, numerical, synthetic and derived evidence in separate categories.

## 5. Fair comparison contract

A candidate pair is comparable only when all required identities match:
1. vessel, hull, appendage and configuration revision;
2. mission/service fingerprint and delivered cargo-work requirement;
3. propulsor installation geometry, units, scale and coordinate frames;
4. speed, loading, draft, trim, water density/temperature, depth and metocean condition;
5. propeller-plane inflow/wake and reference velocity;
6. measurement boundary (shaft, motor terminals, DC bus or fuel input) and included auxiliaries;
7. test method, instrumentation, calibration epoch, sample window, uncertainty method and scale-transfer basis.

Explicitly refuse these invalid substitutions:
- peak open-water efficiency for installed delivered power;
- model scale for full scale without an admitted transfer method;
- static bollard pull for transit efficiency;
- unmatched advance coefficients or loading conditions;
- shaft power for electrical input power;
- one source's comparator for another source's result;
- a predicted or surrogate output for an observed measurement.

## 6. Metrics and non-compensable gates

**Primary outcomes**
- delivered shaft/electrical power at matched required thrust and service;
- mission energy per delivered cargo-work unit;
- robust nondominated set across energy, lifecycle cost, availability, payload and environmental burden.

**Supporting observables**
- full (K_T), (K_Q), (J), Reynolds and η_0 curves;
- thrust deduction, wake fraction and relative rotative efficiency;
- cavitation inception, type, extent and intermittency;
- frequency-resolved underwater radiated noise and pressure-pulse amplitude;
- unsteady loads and fatigue-relevant structural response;
- mass, diameter, draft/clearance, installation losses, maintenance and repairability.

**Hard gates** are structural strength/fatigue, rotor containment, cavitation/vibration limits for the declared service, manoeuvring and degraded-mode requirements, environmental/acoustic requirements, manufacturing/inspection/repair requirements, and applicable statutory/class/flag assessment by competent authorities.

Unknown or uncertainty-overlapping hard gates HOLD or remain INCOMPLETE. They are not converted into passes or compensated by energy savings.

## 7. Falsifiable hypotheses

- **H1 — conditional toroidal benefit:** toroidal designs may improve selected vortex/noise observables in particular operating regimes, but do not universally dominate conventional designs in thrust, efficiency or installed power.
- **H2 — wake interaction:** open-water rankings can reverse under non-uniform hull wake and self-propulsion.
- **H3 — compliant blades:** reduced unsteady forces/noise may trade against propulsive efficiency, fatigue and durability; the balance is regime-dependent.
- **H4 — experiment value:** architecture-aware multi-fidelity acquisition can reach a validated Pareto frontier with fewer expensive evaluations than fixed-grid, random and single-fidelity baselines without increasing false-feasible selection.
- **H5 — mission relevance:** component-level improvements matter only when whole-vessel service, energy and lifecycle gains survive uncertainty, installation and degradation.

## 8. Benchmark targets (research targets, not achieved results)

1. Reproduce one published toroidal experimental/CFD case and one conventional comparator before designing a new candidate.
2. Report held-out calibration by observable and operating regime, including false-feasible rate, conditional coverage, ranking stability and selection regret.
3. For this propulsor-specific benchmark, target at least **5% independently validated reduction in delivered propulsor/shaft power** under matched vessel service; stretch target **10%**. The full service and every hard gate must remain satisfied.
4. Target at least **20% lower cost to reach the independently confirmed Pareto frontier** versus preregistered random/grid baselines, using the existing #11513 evaluation-cost owner.
5. Admit **zero mandatory safety/environmental gate violations** to the validated frontier.
6. Publish negative, neutral and positive results. Do not issue a universal architecture winner ranking.

These are subsystem-stage research hypotheses, not expected outcomes or current achievements. They do not replace the parent MARINE-WAPS programme targets: at least **10% validated whole-mission energy improvement**, **15% stretch**, at least **10% validated Pareto-hypervolume improvement**, and at least **20% lower cost to reach the validated frontier**. Those targets are assessed only at the whole-vessel service boundary; a propulsor-level improvement does not automatically imply a mission-level gain.

## 9. Work sequence

- **P0:** source/rights/condition ledger and claim/comparator normalization (this branch).
- **P1:** architecture and matched-operating-point schema/fixtures; reproduce published conventional and toroidal reference cases.
- **P2:** wake and hull integration using existing solver V&V and wake-fidelity contracts.
- **P3:** experiment-value replay against fixed-grid, random and current multifidelity baselines.
- **P4:** independent tunnel/self-propulsion validation when rights, facilities, funding and physical authority exist.
- **P5:** extend to mission and lifecycle decisions, manufacturing and maintenance.

No stage is promoted because a runner is queued or a synthetic fixture passes. Each claim requires the exact upstream evidence it consumes.

## 10. Existing owners and non-duplication

- Symthaea #8082 owns hull/propulsor co-design and physical design reasoning.
- Symthaea #11406/#11410 owns whole-vessel WAPS/energy/service comparison.
- Symthaea #11442 owns hydrodynamic solver V&V and external solver interoperation.
- Symthaea #11513 owns multifidelity acquisition and evaluation-cost accounting.
- Symthaea #11729/#11740 owns propeller-plane wake fidelity.
- Symthaea #11603 and child acoustic audits own acoustic/environmental observation and interpretation.
- Symthaea #11765/#11789 owns interval-robust Pareto decisions.
- Symtropy shipbuild owners own DFM, manufacturing, assembly, repair and physical process qualification.
- Mycelix owns evidence lineage, rights/provenance and authority hand-off.
- Sol-Atlas owns spatial/metocean/test-context provenance.

## 11. Claim ceiling

A passing source ledger or schema check establishes only that the declared records are structurally valid and claim boundaries are preserved. It does not validate a CFD solver, authenticate publishers or raw data, prove a propulsor advantage, establish cavitation/noise/structural safety, approve a design, or authorize manufacture or operation. Physical conclusions require independent evidence for the exact vessel, propulsor, condition envelope and measurement boundary.
