# The Substrate: Quick Reference

Substrate Independence is Symthaea's framework for analyzing consciousness across different physical substrates. It implements the Multiple Realizability thesis (Putnam 1967): consciousness depends on computational organization, not physical medium.

## Location

- **Core framework**: `symthaea-core/src/hdc/substrate_independence.rs` (~840 LOC, 13 tests)
- **Validation framework**: `symthaea-core/src/hdc/substrate_validation.rs` (~580 LOC, 11 tests)
- **Consumer**: `ConsciousnessEquationV2` via `ConsciousnessStateV2.substrate_feasibility`

## Key Types

### `SubstrateType` (enum, 8 variants + 3 aliases)

| Variant | Medium | Speed | Energy/Op |
|---------|--------|-------|-----------|
| BiologicalNeurons | Carbon, wet | ~1 ms | ~10 fJ |
| SiliconDigital | Electronic, dry | ~1 ns | ~1 fJ |
| QuantumComputer | Qubits | ~1 us | ~0.1 aJ |
| PhotonicProcessor | Light-based | ~1 ps | ~10 aJ |
| NeuromorphicChip | Analog, spike-based | ~1 us | ~1 fJ |
| BiochemicalComputer | DNA/molecular | ~1 s | ~1 pJ |
| HybridSystem | Multiple substrates | varies | varies |
| ExoticSubstrate | Plasma, BZ reactions | ~10 ms | varies |

Aliases (`Biological`, `Silicon`, `Quantum`, `Hybrid`) map to canonical variants via `.canonical()`.

### `SubstrateRequirements` (struct, 9 dimensions)

Each dimension scored 0.0-1.0:
- **causality** — causal interactions (rules out lookup tables)
- **integration_capacity** — information integration across units
- **temporal_dynamics** — rich temporal evolution (not static)
- **recurrence** — feedback loops (not feedforward only)
- **binding_capability** — synchronous feature binding
- **attention_capability** — selective amplification
- **workspace_capability** — global broadcasting (GWT)
- **hot_capability** — meta-representation (Higher-Order Thought)
- **quantum_support** — quantum phenomena support

**Feasibility formula**: `critical_min * workspace * (0.5 + 0.5 * enhancement_avg)`
- Critical = min(causality, integration, dynamics, recurrence)
- Enhancement = avg(binding, attention, HOT)

### `SubstrateComparison` (struct)

Full profile for a substrate: type, requirements, computed feasibility, advantages/disadvantages/best-for lists.

### `SubstrateIndependence` (main system)

HashMap of all substrate comparisons. Methods:
- `rank_by_feasibility()` — sorted substrate rankings
- `can_be_conscious(substrate)` — threshold check (feasibility > 0.3)
- `generate_report()` — formatted multi-substrate analysis

### Validation: `SubstrateValidationFramework`

The *honest counterpart* to feasibility scores. Explicitly acknowledges uncertainty:

| Evidence Level | Confidence | Example |
|----------------|-----------|---------|
| Validated | 0.95 | Biological neurons |
| Experimental | 0.80 | — |
| Observational | 0.60 | — |
| Theoretical | 0.10 | Silicon, Quantum |
| None | 0.00 | Hybrid systems |

**Key insight**: Hypothetical feasibility (from substrate_independence.rs) and honest confidence (from substrate_validation.rs) can diverge significantly. The `feasibility_gap()` method measures this.

## Current Integration

`ConsciousnessEquationV2` receives `substrate_feasibility` in `ConsciousnessStateV2`.
**Production sites are wired to `SubstrateManager::effective_feasibility`** (Roadmap Phase 2, DONE):
- `src/cognitive_loop/cycle_subsystems.rs:254` — cycle metadata assembly
- `src/cognitive_loop/cycle_phase_feedback.rs:899` — consciousness feedback
- `src/cognitive_loop/substrate_manager.rs:502` — telemetry

Hardcoded `1.0` now only remains in:
- `src/cognitive_loop/consciousness_engine/tests.rs` (9 sites — test fixtures with controlled inputs; legitimate)
- `src/consciousness/measurement/consciousness_equation_v2.rs:194` (default-constructor helper)
- `examples/evolve_consciousness_equation.rs:155` (example script)

Optional enrichment, **already wired** (this note previously said "integration deferred" —
corrected 2026-07-02, verified against `src/cognitive_loop/substrate_manager.rs:265-278`):
the `quantum-consciousness` feature (`crates/domains/symthaea-quantum-chemistry/`,
`WIRING_INSTRUCTIONS.md`) blends an ab initio multi-theory physics score
(`cognitive_loop_bridge::substrate_feasibility_from_physics`, water at 310K as the
reference substrate) 50/50 into `effective_feasibility` inside
`SubstrateManager::recompute_effective_feasibility`. It's off by default (not in the
default feature set) but is real, feature-gated production code, not a stub.

See `THE_SUBSTRATE_ROADMAP.md` for the full status across Phases 1–5.

## Running Tests

```bash
cargo test -p symthaea-core --lib substrate_independence
cargo test -p symthaea-core --lib substrate_validation
```
