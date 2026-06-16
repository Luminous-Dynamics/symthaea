# Wiring symthaea-quantum-chemistry into CognitiveLoopService

## The Integration Point

In `symthaea/src/cognitive_loop/cycle_subsystems.rs` (line ~249) and
`symthaea/src/cognitive_loop/cycle_phase_feedback.rs` (line ~541),
replace the hardcoded `substrate_feasibility: 1.0` with physics-derived values.

## Step 1: Add dependency (symthaea/Cargo.toml)

```toml
[dependencies]
symthaea-quantum-chemistry = { path = "crates/symthaea-quantum-chemistry", optional = true }

[features]
quantum-consciousness = ["dep:symthaea-quantum-chemistry"]
```

## Step 2: Wire in (src/cognitive_loop/cycle_subsystems.rs)

```rust
// At the top:
#[cfg(feature = "quantum-consciousness")]
use symthaea_quantum_chemistry::cognitive_loop_bridge::substrate_feasibility_from_physics;

// Replace line ~249:
// OLD: substrate_feasibility: 1.0,
// NEW:
substrate_feasibility: {
    #[cfg(feature = "quantum-consciousness")]
    {
        // Use water molecule as the reference substrate
        // (can be configured via CognitiveLoopConfig.substrate_molecule)
        let mol = symthaea_quantum_chemistry::Molecule::water();
        substrate_feasibility_from_physics(&mol, 310.0) // Body temperature
    }
    #[cfg(not(feature = "quantum-consciousness"))]
    { 1.0 }
},
```

## Step 3: Full theory grounding (optional, deeper integration)

In `consciousness_equation_v2.rs`, ground all 7 CoreComponent values:

```rust
#[cfg(feature = "quantum-consciousness")]
{
    use symthaea_quantum_chemistry::cognitive_loop_bridge::physics_to_consciousness_state;
    let physics = physics_to_consciousness_state(&substrate_molecule, 310.0);

    state.core_values.insert(CoreComponent::Integration, physics.integration);
    state.core_values.insert(CoreComponent::Binding, physics.binding);
    state.core_values.insert(CoreComponent::Workspace, physics.workspace);
    state.core_values.insert(CoreComponent::Attention, physics.attention);
    state.core_values.insert(CoreComponent::Recursion, physics.recursion);
    state.core_values.insert(CoreComponent::Efficacy, physics.efficacy);
    state.core_values.insert(CoreComponent::Knowledge, physics.knowledge);
}
```

## Step 4: Test

```bash
cargo test -p symthaea --features quantum-consciousness --lib -- consciousness
```

## What This Changes

Before: ConsciousnessEquationV2 uses heuristic parameters (0.0-1.0) for each
theory component. The substrate_feasibility is hardcoded to 1.0.

After: Each theory component is grounded in a specific, computable molecular
quantity derived from the Schrödinger equation. The substrate_feasibility
reflects the multi-theory composite consciousness score for the physical
substrate the system runs on.

The cognitive loop's 31Hz consciousness measurement becomes a function of
actual molecular physics rather than tuned hyperparameters.
