# Plasma model credibility census — first-pass source review

**Status:** Initial static-source census only. No tests were run for this document. This file does not qualify a plasma model or certify a physical design.  
**Audit work item:** [PLASMA-CRED-001 #7600](https://github.com/Luminous-Dynamics/symthaea/issues/7600)  
**Architecture:** [PLASMA-ENG-000 #5669](https://github.com/Luminous-Dynamics/symthaea/issues/5669)  
**External solver benchmark:** [PLASMA-VV-001 #7588](https://github.com/Luminous-Dynamics/symthaea/issues/7588)

## 1. Why this census exists

The repository contains useful analytical plasma calculations, HDC encoders, data-facing fusion digital-twin vocabulary, and prototype control/optimization examples. These are potential ingredients for a design-and-discovery workflow. Their presence is not evidence of a validated plasma-discharge solver.

The engineering objective is to make every plasma-related claim carry its model identity, units, validity envelope, evidence class, independent reference, uncertainty and permitted-use boundary. We should be able to reuse exploratory code without accidentally promoting it to physical authority.

## 2. First-pass source inventory

The source SHA values below are blob identities returned during source inspection. Re-resolve them against the exact commit being audited before implementing or accepting a code change.

| Source | Inspected blob SHA | What it appears to provide | Initial evidence class | Initial boundary |
|---|---|---|---|---|
| [plasma_mhd.rs](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-continuum-physics/src/plasma_mhd.rs) | 7092984243e2342e30c49b287506367423672d3f | Closed-form quantities including Debye length, electron plasma frequency, cyclotron frequency, Larmor radius, Alfvén speed, sound speed, plasma beta, magnetic pressure, magnetic Reynolds number, Spitzer resistivity approximation, Coulomb logarithm, Z-pinch field, Bennett current and tokamak safety factor | Analytical calculator with a small set of formula-level tests | Not a time-dependent plasma solver; units/domain correctness and physical applicability are not comprehensively established |
| [plasma_physics.rs](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/core/symthaea-core/src/physics/plasma_physics.rs) | 3e3c451ba77dd9076d9f94cfc7d61f9fed19b1f0 | HDC representation of plasma concepts; module header explicitly labels it exploratory | Representation / exploratory | Encoded similarity cannot be treated as a plasma observable or validated physical calculation |
| [plasma_hdc_encoder.rs](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-physics/src/plasma_hdc_encoder.rs) | 25a04afc5ad22fc0111779d087a903a32734db2a | Converts named sensor-like quantities into HDC representations | Representation / sensor-data encoding | Encoding and downstream prediction must be independently evaluated; encoded inputs are not themselves diagnostic truth |
| [plasma_control.rs](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-physics/src/plasma_control.rs) | a1c71e104e279d3a091a8212d80e9fbc987cf9c6 | Prototype control concepts, configurable Phi thresholds, gas-species table and mapping into cognitive motor-command names | Heuristic / prototype; complete tests/status still need review | Not a qualified plasma controller, disruption predictor, safe-landing procedure or actuator interface |
| [plasma_disruption_benchmark.rs](https://github.com/Luminous-Dynamics/symthaea/blob/main/examples/plasma_disruption_benchmark.rs) | Re-resolve before audit | A disruption benchmark example with synthetic-data-generation utilities in its imports | Synthetic/example until verified otherwise | Synthetic benchmark results must not be reported as real-device prediction performance |
| [plasma_threshold_optimizer.rs](https://github.com/Luminous-Dynamics/symthaea/blob/main/examples/plasma_threshold_optimizer.rs) | Re-resolve before audit | Threshold selection/optimization example | Research example until corpus and oracle are audited | Optimization success does not establish a qualified operating threshold |

This is a prioritized sample, not an exhaustive repository-wide census. Files, examples, tests, exported APIs, downstream consumers and crate feature gates still need to be enumerated.

## 3. First-pass technical red flags to resolve

### 3.1 Spitzer resistivity formula and unit convention

The current spitzer_resistivity(temperature, z_eff, coulomb_log) expression uses SI-labelled constants (elementary charge, electron mass and Boltzmann constant) and a temperature term formed with K_B * temperature, but the implemented expression does not visibly include the vacuum-permittivity-dependent factor present in standard SI formulations. The function signature does not label input or output units.

PlasmaPy documents a Spitzer-resistivity API with explicit temperature and density units and describes the collision-frequency basis for the resistivity: https://docs.plasmapy.org/en/stable/api_static/plasmapy.formulary.collisions.misc.Spitzer_resistivity.html. A standard SI expression likewise includes the permittivity of free space; see the CERN plasma-physics formulary: https://cas.web.cern.ch/sites/default/files/CAS_Gibbon_Lecture1.pdf.

**Disposition:** treat this API as unqualified for quantitative engineering until an independent derivation, dimensional check, coefficient/convention comparison, and reference-value test establish the correct formulation and assumptions. Do not patch from a superficial coefficient comparison alone; specify whether parallel/perpendicular resistivity, charge state, collision model and Coulomb-log convention apply.

### 3.2 Mixed input units

The Coulomb-logarithm helper explicitly documents electron temperature in eV and density in cm^-3, while nearby plasma routines use SI-looking arguments without unit-bearing types. This mixture is manageable only if interfaces make conventions difficult to confuse.

**Disposition:** document every parameter unit at the API boundary. Prefer validated unit-aware inputs or strongly named wrapper types. SI-only conventions must not silently consume eV or cm^-3.

### 3.3 Formula test strength and singularities

Examples in the inspected MHD module check broad output ranges, such as an order-of-magnitude Debye length or qualitative plasma-beta regime. Those are useful smoke tests, but not independent oracles for the exact formula. Several public functions divide by field strength, mass, density, radius or major radius, or take square roots of expressions whose domains must be positive.

**Disposition:** add independent reference values, dimensional tests and invalid-input behavior. Test limits and monotonicity only where equation assumptions justify them. Decide which low-level functions return structured errors and which require validated domain types; never turn NaN/Inf or invalid conditions into plausible engineering outputs.

### 3.4 Safety factor is not a universal pass/fail stability gate

The current safety_factor helper computes a local algebraic quantity. An adjacent test comment describes q > 2 as a stable tokamak case. That is acceptable as one named example, but must not become a universal stability rule.

An experiment reported controlled DIII-D divertor operation at edge safety factor q95 = 1.9 for 150 instability growth times with magnetic feedback control: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.045003. This does not imply arbitrary q < 2 operation is safe; it demonstrates that profile, mode, configuration and feedback context matter.

**Disposition:** make geometry/profile/definition explicit and keep a bare q calculation separate from MHD stability assessment.

### 3.5 Phi thresholds, command labels and fixed gas factors

The inspected plasma_control.rs module maps configurable Phi thresholds to warning/critical/emergency narratives and maps cognitive MotorCommandType variants to actions such as power reduction, magnetic-field adjustment, gas injection and soft landing. It also assigns fixed per-species cooling_factor and z_effective values.

No empirical support for these particular numerical thresholds or species factors was established by this static inspection. A command-name mapping also does not establish actuator semantics, plant dynamics or a safe plasma-control policy. Effective charge and cooling/radiation consequences depend on plasma state, composition, transport and operating regime; a fixed per-element number is not a universal physical model.

**Disposition:** treat these values and transitions as illustrative/unvalidated heuristics unless the exact mapping has a reference, regime definition, evidence dataset, uncertainty and independent benchmark. Do not wire these into real-time control or represent them as a validated disruption-avoidance strategy. Phi/HDC may be investigated as advisory features only under separate empirical evaluation against established baselines.

## 4. Evidence classes for the full census

Every exported formula, model, encoder, threshold and example should carry one of these distinct labels:

1. **Analytical implementation** — equation encoded, not yet independently verified.
2. **Unit / domain checked** — dimensional and valid-domain behavior inspected.
3. **Reference checked** — compared to independent trusted values over a declared test envelope.
4. **Numerically verified** — solver implementation/discretization checked against appropriate analytical/manufactured results and convergence tests.
5. **Code benchmarked** — compared with an independent code using matched equations and boundary conditions.
6. **Experimentally validated for profile** — compared with actual measurements, instrument/calibration provenance, uncertainty and a specific qualified operating envelope.
7. **Heuristic / synthetic-only / unresolved** — remains non-authoritative for physical claims.

These classes are not interchangeable. Reference-checking an analytical formula does not make it experimentally validated. A result may require several independent evidence items rather than one global status.

## 5. Independent qualification corpus

The audit should produce fixtures for:
- dimensional and unit consistency with an independent oracle;
- reference values from primary texts or qualified external libraries;
- monotonicity and limiting behavior with valid assumptions recorded;
- negative, zero, non-finite and singular inputs;
- distinct charge/species/temperature conventions;
- fluid-model outputs used outside their documented validity envelope;
- synthetic data mislabeled as measured observations;
- HDC similarity or Phi score incorrectly projected as a calibrated sensor or stability probability;
- an optimizer selecting a constraint-violating or out-of-domain candidate;
- output from a changed source SHA incorrectly reusing a previous receipt.

External libraries are references, not infallible truth. Record package version and API semantics; do not silently assume that two implementations share conventions.

## 6. Relationship to the plasma discovery program

1. Complete this source census and qualify analytical helpers that will actually be used.
2. Select a pinned, supported cold-plasma reference case for [PLASMA-VV-001 #7588](https://github.com/Luminous-Dynamics/symthaea/issues/7588).
3. Integrate its outputs through Symtropy's solver-neutral boundary, [PLASMA-TWIN-001 #1685](https://github.com/Luminous-Dynamics/symtropy/issues/1685).
4. Only then enable multi-objective design exploration with equal-budget simple and Bayesian/multi-fidelity baselines.
5. Qualify each new regime (cold atmospheric, thermal arc/jet, kinetic, fusion) independently.

The community Zapdos verification/validation study is a useful initial low-temperature RF fluid-model reference, but it reports discrepancies at lower pressures and discusses limits of the fluid assumption or boundary conditions: https://doi.org/10.1016/j.cpc.2023.108837.

PlasmaPy is a valuable open community toolkit for plasma formulary, analysis, particles, diagnostics and related research workflows. Use it for suitable utilities and cross-checks, not as a universally capable device-discharge solver: https://docs.plasmapy.org/en/latest/index.html.

## 7. Acceptance gates

- Full source census of public plasma formulas, examples, encoders and control/optimization concepts.
- Each entry has a source identity, unit contract, applicability boundary, evidence class and claim ceiling.
- No formula is marked reference-checked without an independent oracle.
- No Phi/HDC-only output can be called a plasma measurement, validated stability estimate or actuator authority.
- Invalid, singular and non-finite input cases are explicitly handled.
- Synthetic results are clearly labelled and not used as physical validation.
- Exact-revision independent evidence is required for every PASS; queued CI does not qualify code.
- All unresolved discrepancies stay open and visible.

## 8. Safety and non-goals

This is offline source review, analytical verification and simulation/evidence architecture. It does not authorize plasma ignition, high-voltage/RF/laser operation, hazardous gas use, real-time control, device fabrication, safe-landing procedures or actuation. No control policy should be attached to hardware without separate qualified safety case, domain-expert review and controlled experimental validation.
