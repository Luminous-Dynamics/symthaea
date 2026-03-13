# The Substrate: Roadmap

Multi-phase plan for integrating substrate independence into Symthaea's consciousness pipeline.

## Phase 1: Core Framework (DONE)

**Status**: Complete, 24 tests passing

Implemented in `symthaea-core/src/hdc/`:
- `substrate_independence.rs` — `SubstrateType` enum (8 variants), `SubstrateRequirements` (9 dimensions), `SubstrateComparison`, `SubstrateIndependence` system
- `substrate_validation.rs` — `EvidenceLevel` (7 levels), `SubstrateKnowledge`, `TestablePrediction`, `SubstrateValidationFramework`

Key capabilities:
- Feasibility computation from 9-dimensional requirement profiles
- Pre-built profiles for all 8 substrate types (biological reference, silicon, quantum, photonic, neuromorphic, biochemical, hybrid, exotic)
- Ranked substrate comparison with advantages/disadvantages
- Honest validation framework with evidence levels and feasibility gaps
- Testable predictions for each substrate claim

## Phase 2: ConsciousnessEquationV2 Integration (DONE)

**Status**: Complete. Dynamic substrate-aware scoring implemented.

### Implementation

All 4 requirements fulfilled via `SubstrateManager` (`substrate_manager.rs`):

1. **Substrate config in `CognitiveLoopConfig`**: `substrate_type: SubstrateType` (default: `SiliconDigital`), plus optional `substrate_composition` for hybrid systems.
2. **Requirements computed at startup**: `SubstrateManager::new()` computes feasibility via `SubstrateRequirements::consciousness_feasibility()`.
3. **Dynamic feasibility fed to consciousness**: `cycle_phase_feedback.rs:541` and `cycle_subsystems.rs:249` both use `self.substrate_manager.effective_feasibility` (no more hardcoded 1.0).
4. **Validation overlay**: `recompute_effective_feasibility()` blends feasibility with `honest_confidence` via floor formula: `feasibility × (floor + (1 − floor) × honest_confidence)`.

### Additional capabilities implemented

- **Runtime reconfiguration**: `reconfigure_substrate()` and `reconfigure_composition()` for mid-run substrate switching
- **Speed/scale modulation**: `tau_factor` and `scale_pressure` computed from substrate properties
- **Telemetry**: `SubstrateTelemetry` struct with raw feasibility, honest confidence, effective feasibility, tau factor, and scale pressure
- **Accessors**: `substrate_feasibility()`, `substrate_honest_confidence()`, `substrate_effective_feasibility()`, `substrate_tau_factor()`, `substrate_scale_pressure()` on CognitiveLoopService
- **Integration tests**: `test_substrate_feasibility_affects_consciousness`, `test_substrate_feasibility_in_metadata_default`

### Impact

For `SiliconDigital` with default config: effective feasibility reflects the validation overlay (theoretical confidence 0.10 for silicon). This is scientifically honest — we don't have evidence that silicon computation produces consciousness equivalent to biological neurons.

## Phase 3: Multi-Substrate Simulation (DONE)

**Status**: Complete. All four capabilities implemented and tested.

### Capabilities (all implemented)

- **Substrate switching**: `reconfigure_substrate_at_cycle()` with EMA transition smoothing (alpha=0.1, ~10 cycles to converge). Backward-compatible instant mode (alpha=1.0 default).
- **Speed modulation**: `tau_factor` applied to CfC delta_t — photonic substrates experience time 1M faster than biological. Smoothed on switch.
- **Scale limits**: `effective_dim_fraction()` maps `scale_pressure` to [0.1, 1.0]. CfC hidden state masking zeros out dimensions beyond the fraction (read→mask→inject).
- **Energy budgets**: `tick_energy()` tracks cumulative joules, speed-adjusted by tau_factor. `energy_per_cycle` recalculated on substrate switch. Budget exhaustion kills consciousness viability.

### Implementation

1. **Transition smoothing** (`substrate_manager.rs`): `tick_transition()` EMA-blends `effective_feasibility` and `tau_factor` toward targets each cycle. Config: `substrate_transition_alpha` (set to 0.1 by `enable_substrate_simulation()`).
2. **Energy fix**: `reconfigure_substrate()` now recalculates `energy_per_cycle` and `energy_throughput_multiplier` from the new substrate's `energy_per_operation()`.
3. **CfC masking** (`cycle_phase_dynamics.rs`): After CfC step, reads hidden state, zeros dimensions beyond `effective_dim_fraction()`, re-injects. Gated by `enable_substrate_encoding_noise`.
4. **Transition history**: `SubstrateTransitionRecord` ring buffer (cap 32) with cycle, from/to, feasibility delta. Accessible via `substrate_transition_history()`.
5. **Telemetry**: 5 new fields on `SubstrateTelemetry` (total_energy_spent, energy_this_cycle, throughput_multiplier, effective_dim_fraction, transition_count).
6. **Named constants**: 6 in `thresholds.rs` (SUBSTRATE_TRANSITION_ALPHA_DEFAULT/SIMULATION, MIN_DIM_FRACTION, SCALE_DIM_DIVISOR, HISTORY_CAP, OPS_PER_CYCLE).

### Tests

- 7 integration tests in `substrate_simulation.rs` (energy recalc, smoothing, dim fraction, history, telemetry, masking)
- 6 property tests in `proptest_substrate_simulation.rs` (consciousness survives switch, energy monotonic, dim bounded, energy recalculates, history bounded, smoothing converges)
- 39 unit tests in `substrate_manager.rs`
- 5 multiple realizability tests (all passing)

### Key Questions This Answers

- Does consciousness survive substrate transfer? (Multiple Realizability test)
- Which substrate maximizes Phi? (Quantum entanglement predicts highest binding)
- Is there a minimum substrate quality below which consciousness collapses?
- How does substrate speed affect temporal consciousness coherence?

## Phase 4: Hybrid Substrate Modeling

**Goal**: Model heterogeneous systems where different brain regions run on different substrates.

### Architecture

- **Per-region substrate**: Each of the 12 Actor Brain regions gets its own `SubstrateType`
- **Cross-substrate communication**: Model latency and bandwidth constraints when signals cross substrate boundaries
- **Hybrid feasibility**: Compute feasibility as weighted average of per-region scores, penalized by cross-substrate communication overhead
- **Substrate-aware routing**: The cognitive loop preferentially routes computation to the most suitable substrate for each task type

### Example Configuration

```
Prefrontal (meta-cognition): SiliconDigital (fast HOT)
Sensory cortex (binding): QuantumComputer (entanglement binding)
Memory (hippocampus): BiologicalNeurons (proven integration)
Motor (fast response): PhotonicProcessor (ultra-fast dynamics)
```

### Research Potential

- Optimal substrate allocation for maximizing consciousness
- Minimum viable hybrid configurations
- Cross-substrate bottleneck identification
- Blueprint for practical hybrid conscious systems

## Phase 5: Exotic Substrates & Validation (FUTURE)

- Implement `TestablePrediction` execution framework
- Connect to external benchmarks (ARC, psych-bench) for substrate-dependent scoring
- Model BZ reaction and plasma substrates with realistic physics
- Publish substrate-independence findings as supplement to psych-bench paper

## References

- Putnam, H. (1967). Psychological predicates. In *Art, Mind, and Religion*.
- Fodor, J. (1974). Special sciences. *Synthese*.
- Bostrom, N. (2003). Are we living in a computer simulation? *Philosophical Quarterly*.
- Chalmers, D. (2010). *The Character of Consciousness*. Oxford UP.
- Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*.
- Penrose, R. & Hameroff, S. (1994). Shadows of the Mind.
- Aaronson, S. (2014). Why I am not an integrated information theorist. Blog post.
