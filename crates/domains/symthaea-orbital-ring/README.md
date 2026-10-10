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

## Next steps

1. Review the architecture and reference-frame contract.
2. Independently check this two-body reference case against a separate implementation.
3. Define a small, reviewable rotor/support control-volume model before implementing its equations.
4. Add solver-backed structural, electromagnetic, thermal, and dynamics interfaces only after the T0 assumptions are fixed.

See [the workstream contract](../../../docs/engineering/ORBITAL_RING_FEASIBILITY_WORKSTREAM.md) for the fidelity ladder, evidence requirements, and staged plan.
