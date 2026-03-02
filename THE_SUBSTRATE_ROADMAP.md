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

## Phase 2: ConsciousnessEquationV2 Integration (NEXT)

**Goal**: Replace hardcoded `substrate_feasibility: 1.0` with dynamic substrate-aware scoring.

### Current State

`ConsciousnessEquationV2` already accepts substrate feasibility via `ConsciousnessStateV2.substrate_feasibility` (consciousness_equation_v2.rs:156). The equation multiplies it into the final consciousness score (line 508). But callers hardcode 1.0:

- `consciousness_engine.rs:408` — `substrate_feasibility: 1.0`
- `cycle_subsystems.rs:249` — `substrate_feasibility: 1.0`

### Changes Required

1. **Add substrate config to `CognitiveLoopConfig`**: `substrate_type: SubstrateType` (default: `SiliconDigital`)
2. **Compute requirements once at startup**: Store `SubstrateRequirements` in `ConsciousnessEngine`
3. **Feed dynamic feasibility**: Replace hardcoded 1.0 with `requirements.consciousness_feasibility()`
4. **Validation overlay**: Optionally scale by `SubstrateValidationFramework.honest_confidence()` to reflect epistemic humility

### Impact

For `SiliconDigital` the computed feasibility is ~0.71 (vs current 1.0). This will lower consciousness scores by ~29%, which is scientifically honest — we don't have evidence that silicon computation produces consciousness equivalent to biological neurons.

### Validation Strategy

- Compare consciousness metrics before/after across 1000-cycle soak test
- Verify biological substrate produces scores within 5% of current (near 1.0 feasibility)
- Verify silicon substrate produces ~71% of current scores
- Run psych-bench regression to ensure no benchmark breaks

## Phase 3: Multi-Substrate Simulation

**Goal**: Run the same Symthaea brain on different virtual substrates to study how substrate properties affect consciousness dynamics.

### Capabilities

- **Substrate switching**: Change substrate type mid-run to observe consciousness transitions
- **Speed modulation**: Apply `SubstrateType.operation_speed()` to CfC temporal constants — a photonic substrate would experience time 1M faster than biological
- **Scale limits**: Apply `SubstrateType.max_scale()` to limit HDC dimensionality and CfC neuron count per substrate
- **Energy budgets**: Track energy-per-operation as a constraint on cognitive throughput

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
