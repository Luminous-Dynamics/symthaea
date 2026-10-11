# symthaea-orbital-ring

Deterministic T0 calculation kernel for early orbital-ring research.

**Status:** research scaffolding. Not an orbital-ring simulator, structural design, stability analysis, solver adapter, or flight-qualified component.

## What exists at this increment

- Checked SI boundary types for length, mass, duration, and energy.
- Explicit architecture, reference-frame, gravity-model, and atmospheric-drag assumptions.
- Earth point-mass circular-orbit reference values for the freely orbiting architecture only.
- Energy and momentum ledger residuals in an explicitly declared Earth-centred inertial frame.
- Unit tests for the documented reference values, unsupported assumptions, invalid values, conservation residuals, and determinism.

The circular-orbit helper intentionally rejects a dynamically supported stationary ring, launch loop, momentum-exchange tether, or ring-mounted accelerator. The model does not contain rotor/support physics, so it cannot establish their equilibrium.

## Run locally

From the Symthaea workspace:

```sh
cargo check -p symthaea-orbital-ring
cargo test -p symthaea-orbital-ring
cargo fmt --check -p symthaea-orbital-ring
cargo clippy -p symthaea-orbital-ring --all-targets -- -D warnings
```

These commands are a verification target; this README does not claim they have run in the current environment.

## Independent T0 reference check

From the repository root:

```sh
python3 scripts/engineering/orbital_ring_reference_check.py
```

This standard-library Python implementation checks the documented orbit fixtures and Rust numeric anchors, algebraic consistency relations, and a hand-checkable inertial circular-stream limit. It emits deterministic JSON and exits nonzero on failure. The current recorded local run passed 47/47 checks with byte-identical output on repeat. This is arithmetic/limiting-case evidence only; it does not compile or test this Rust crate, compare against an external astrodynamics solver, validate moving-support physics, or prove ring equilibrium. See the [control-volume derivation note](../../../docs/engineering/ORBITAL_RING_CONTROL_VOLUME_NOTE.md) for the equation boundary and the inputs that must be frozen before a coupled model is implemented. The [Birch power source audit](../../../scripts/engineering/orbital_ring_power_source_audit.py) independently checks quoted eddy-current drag and cryostat figures; it does not validate the historical model. Its derivation and source-context caveats are documented in [the power-budget audit note](../../../docs/engineering/ORBITAL_RING_POWER_BUDGET_AUDIT.md). The audit applies Table 8's 10 W/m entry to the geosynchronous/support column only, matching the scanned table and its preceding explanation of the vacuum-flask thermal boundary.

## Next steps

1. Keep the Python arithmetic check as a reproducible T0 gate; obtain actual Rust-toolchain results as a separate code-quality gate.
2. Specify and independently derive one rotor/support control volume, including all reference-frame transforms and mass/energy/momentum boundary flows.
3. Compare the bounded orbital subsystem to an appropriately scoped external solver or independent review before making stronger fidelity claims.
4. Add solver-backed structural, electromagnetic, thermal, and dynamics interfaces only after the T0 assumptions are fixed.

See [the workstream contract](../../../docs/engineering/ORBITAL_RING_FEASIBILITY_WORKSTREAM.md) for the fidelity ladder, evidence requirements, and staged plan.
