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

## Phase 4: Hybrid Substrate Modeling (DONE)

**Status**: Complete. Per-region substrate assignment, consciousness coupling, and cross-substrate penalties implemented.

### Capabilities (all implemented)

- **Per-region substrate**: Each of the 12 `CorticalRegion` variants gets its own `SubstrateType` via `per_region_substrates: HashMap<CorticalRegion, SubstrateType>`
- **Cross-substrate communication penalty**: `recompute_aggregate_from_regions()` applies 0.95× per distinct substrate pair to aggregate feasibility
- **Region-aware consciousness**: 4 capability dimensions (binding, workspace, attention, HOT) resolve per-region substrate requirements into the consciousness equation
- **Runtime reconfiguration**: `reconfigure_region(region, substrate)` updates individual regions mid-run with automatic aggregate recomputation

### Implementation

1. **CorticalRegion enum** (`substrate_independence.rs`): 12 variants — Prefrontal, Motor, Sensory, Visual, Auditory, Language, Memory, Emotional, Social, Creative, Executive, Integration. `CorticalRegion::ALL` for iteration.
2. **Per-region storage** (`substrate_manager.rs`): `per_region_substrates: Option<HashMap<CorticalRegion, SubstrateType>>`, `per_region_feasibility: HashMap<CorticalRegion, f32>`. Initialized from `CognitiveLoopConfig.per_region_substrates`.
3. **Capability methods**: `binding_capability(config)` uses Sensory region, `workspace_capability(config)` and `attention_capability(config)` use Prefrontal, `hot_capability(config)` uses Prefrontal. All fall back to global substrate when per-region not set.
4. **Consciousness coupling** (`consciousness_engine/measure.rs`): `binding_capability` scales IIT coherence, `workspace_capability` scales GWT workspace, `attention_capability` scales phi attention weight.
5. **HOT depth** (`cycle_phase_feedback.rs`): `hot_depth = (meta_cognition.depth / 3.0) × substrate.hot_capability` — substrate constrains meta-cognitive recursion.
6. **Telemetry**: `per_region_feasibility: Vec<(String, f32)>` in `SubstrateTelemetry`, populated from HashMap for serialization.
7. **Accessors**: `region_feasibility(region)` and `reconfigure_region(region, substrate)` on `CognitiveLoopService`.

### Example Configuration

```
Prefrontal (meta-cognition): SiliconDigital (fast HOT)
Sensory cortex (binding): QuantumComputer (entanglement binding)
Memory (hippocampus): BiologicalNeurons (proven integration)
Motor (fast response): PhotonicProcessor (ultra-fast dynamics)
```

### Tests

- 3 integration tests in `substrate_simulation.rs` (per_region_substrate_assignment, per_region_runtime_reconfiguration, cross_substrate_penalty)
- 3 integration tests in `foveal_bridge_integration.rs` (per_region configuration, feasibility variance, consciousness level impact)
- 1 unit test in `substrate_manager.rs` (per_region_feasibility_fallback)

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
