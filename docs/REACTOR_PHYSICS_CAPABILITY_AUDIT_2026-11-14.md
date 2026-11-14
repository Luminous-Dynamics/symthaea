# Reactor Physics Capability Audit — 2026-11-14

**Status:** Repository/source audit; no physical validation claimed  
**Owner:** Symthaea research intelligence, composing with Spark Engine and Symtropy

## Executive finding

Symthaea has useful reactor-adjacent research code, but the codebase currently mixes:
- active physics primitives;
- exploratory HDC encoders and analytic formula helpers;
- dormant source files not declared by the physics module;
- model-generated prototype outputs with hard-coded engineering assumptions.

The next step is evidence qualification and correct claim semantics, not a larger reactor design generator.

## Source inventory

| Source | Current status from main | Safe interpretation |
|---|---|---|
| physics/spark_engine.rs | Declared in physics/mod.rs; module header calls the v1.0 architecture hypothetical and explicitly says no net energy gain is established | Active conceptual LCF model; not a validated device |
| physics/spark_prototype_spec.rs | Declared in physics/mod.rs; generates dimensions/BOM/cost/safety/manufacturing sections from a coupled-model result | Model-derived concept output. Its result.feasible flag must not produce a physical DESIGN VALIDATED claim |
| physics/radiation_damage.rs | Declared and re-exported | Exploratory radiation-damage calculation requiring model-by-model VVUQ and source-qualified inputs |
| physics/trigger_systems.rs | Declared; trigger taxonomy and TRL/cost categories | Taxonomy only. Cost/TRL values require dated evidence and applicability boundaries |
| physics/reactor_architectures.rs | Source file exists but no module declaration appears in the reviewed physics/mod.rs | Dormant/uncompiled through the core physics module |
| physics/scaling_study.rs | Source file exists; not declared by reviewed physics/mod.rs; it imports the dormant architecture module | Dormant/uncompiled through the core physics module; old report metrics are not an active capability claim |
| physics/plasma_physics.rs | Public exploratory HDC encoding module with formula-level unit tests | Plasma concept encoding and scalar relations, not a plasma-field solver |
| symthaea-continuum-physics/src/plasma_mhd.rs | Formula helpers for Debye length, plasma frequency, beta, Alfvén speed, resistivity and simple MHD relations | Analytic utilities, not a validated MHD PDE/equilibrium solver |
| symthaea-digital-twin | Generic asset, telemetry, uncertainty and prediction-residual state | Useful observation/residual substrate, not a physics solver |
| symthaea-engineering::MultiPhysicsRequest | Example constructs ordered stages and named consumed/produced fields | Request/schema orchestration example; does not prove a coupled solver actually executed |

The Symthaea vendored crates/domains/spark-engine README says that crate is excluded from the workspace, has no application consumers and contains placeholder bridge hooks. The standalone public repository remains the owner of the Spark crate.

## Claim-integrity issue

The active prototype-spec generator chooses the summary phrase DESIGN VALIDATED when CoupledSimulationResult.feasible is true. That flag is a model result. It cannot establish:
- experimental replication;
- measured energy balance;
- neutron/radiation safety;
- qualified materials lifetime;
- pressure-boundary or mechanical integrity;
- manufacturing process qualification;
- regulatory compliance;
- net electricity export.

The generator also embeds illustrative cost, conversion-efficiency, lifetime, material and manufacturing assumptions. These must be represented as explicit, source-qualified inputs with uncertainty—or remain clearly labelled as illustrative placeholders. The prototype-spec output must not be used as a build package, safety case or manufacturing release.

## Radiation-damage qualification requirements

Before any DPA/lifetime result is used for architecture comparison:
1. Identify the damage model and validity domain (including nuclear data, recoil spectrum, displacement-threshold assumptions and damage metric).
2. Keep primary knock-on energy, displacement estimate, DPA, defect evolution and component lifetime as distinct quantities.
3. Separate model predictions from measured irradiation/characterization outcomes.
4. Compare against an independent reference implementation and published benchmark data.
5. Include material, temperature, spectrum, dose-rate and microstructure scope in every result.
6. Treat defect recombination or "self-healing" as a material-specific hypothesis requiring evidence, not a generic credit.
7. Do not convert a DPA estimate directly into a qualified service life without a validated damage-to-property model and application-specific evidence.

This is a model-qualification research path, not a physical irradiation recipe.

## Plasma and materials work: preserve domain distinctions

The public Symthaea work on atmospheric-plasma surface activation and durable bonding (including #11780, #11795 and #11907) is valuable adjacent manufacturing research. It addresses treatment uniformity, bondability lifetime, process-to-outcome evidence and experiment selection. It is not the same physical regime as fusion plasma or lattice-mediated nuclear reactions.

Likewise, plasma HDC encodings and analytic MHD helpers are useful knowledge and feature representations, but they must not be represented as field solvers. Symtropy's existing #1685 solver-neutral adapter should own scenario/replay contracts for specialist external solvers.

## Immediate work packages

### P0 — Remove claim promotion

- Replace model-only "design validated" wording with an explicit model-screening result and a permanent statement that physical validation is not established.
- Rename/label generated prototype output as conceptual screening, not a fabrication specification.
- Keep safety, regulatory, lifetime, manufacturing and power-conversion fields as assumptions unless independently qualified.
- Unknown inputs remain unknown; no silently populated cost, TRL, conversion or lifetime values.

### P1 — Resolve dormant modules

- Confirm whether reactor_architectures.rs and scaling_study.rs are intentionally archived or intended to be active.
- If activation is intended, first add provenance, units, model validity, uncertainty and negative controls; then add explicit module declarations and exact-head build/test evidence.
- Do not use historical report claims as evidence that dormant files were compiled or tested.

### P2 — Qualify radiation/material models

- Add a model registry with source, version, equation, assumptions, validity envelope, unit convention and calibration state.
- Establish independent, published reference cases before ranking materials or lifetimes.
- Keep lattice materials science (diffusion, phonons, defects, strain, surfaces/interfaces) connected through sample/process lineage.

### P3 — Use Symthaea for research intelligence

- Compose the existing robust model-adequacy and experiment-design work rather than creating another point-EIG engine.
- Require rival hypotheses, model-misspecification controls, held-out material/energy conditions and preregistered observable definitions.
- Emit candidate hypotheses and experiment proposals; never authorize physical actuation.

## Cross-repository ownership

- **Spark Engine:** reaction-rate calculations, fusion channels, LCF evidence corpus and fusion-specific model comparison.
- **Symthaea:** hypothesis generation, physics knowledge representation, causal/model-adequacy reasoning and source-qualified research synthesis.
- **Symtropy:** scenario identity, external solver adapters, deterministic replay and independent evidence receipts.
- **Specialist solvers/labs:** authoritative numerical results for their qualified equations and physical measurements for their instrument/facility scope.

See Spark's docs/CROSS_REPOSITORY_REACTOR_WORKSTREAM_2026.md and Symtropy's reactor twin validation note for the interop contract.

## References

- [Spark Engine public repository](https://github.com/Luminous-Dynamics/spark-engine)
- [Symtropy PLASMA-TWIN-001 #1685](https://github.com/Luminous-Dynamics/symtropy/issues/1685)
- [Symthaea PLASMA-SURFACE-001 #11795](https://github.com/Luminous-Dynamics/symthaea/issues/11795)
- [Symthaea PLASMA-OED-001 #11907](https://github.com/Luminous-Dynamics/symthaea/issues/11907)
- [Symtropy SHIPYARD-WELD-PHYSICS-VVUQ-001 #4614](https://github.com/Luminous-Dynamics/symtropy/issues/4614)

**Claim boundary:** This audit is based on source inspection. It does not qualify a reactor model, a physical component, a neutron source, a plasma process, a manufacturing procedure or a safety case.
