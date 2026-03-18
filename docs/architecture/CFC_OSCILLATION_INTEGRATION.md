# CfC Oscillation Integration Guide

## Status: READY FOR WIRING (2026-03-16)

All four oscillation components are built, tested, and compile-clean.
This document specifies exactly how to wire them into the live cognitive loop.

## Prerequisites

All components pass tests independently:
- Phase 1: Fourier fused-path fix — 11 tests (symthaea-core)
- Phase 2: CPG Manager — 20 tests (symthaea, feature `cpg`)
- Phase 3: Complex CfC Neuron — 14 tests (symthaea-core, feature `complex_cfc`)
- Phase 4: Spectral Manager — 10 tests (symthaea, feature `spectral_state`)
- Total: 99 oscillation tests, all passing

---

## Priority 2: Wire Into Cognitive Loop

### 2a. CPG Manager Integration

**Files to modify:**

1. `src/cognitive_loop/config.rs` — Add fields:
   ```rust
   #[cfg(feature = "cpg")]
   pub enable_cpg: bool,        // default: false
   #[cfg(feature = "cpg")]
   pub cpg_config: CpgConfig,   // default: CpgConfig::default()
   ```

2. `src/cognitive_loop/mod.rs` (CLS struct) — Add field:
   ```rust
   #[cfg(feature = "cpg")]
   pub(crate) cpg_manager: CpgManager,
   ```

3. `src/cognitive_loop/constructor.rs` — Initialize:
   ```rust
   #[cfg(feature = "cpg")]
   cpg_manager: CpgManager::new(config.cpg_config.clone()),
   ```

4. `src/cognitive_loop/cycle_phase_dynamics.rs` — Phase B, after existing managers:
   ```rust
   #[cfg(feature = "cpg")]
   if self.config.enable_cpg && self.cpg_manager.should_run(cycle, urgency) {
       let cpg_output = self.cpg_manager.process(&snapshot);
       collector.record("cpg", cpg_output);
   }
   ```

   Note: `should_run` is not on the trait — use interval check:
   ```rust
   if self.config.enable_cpg && (cycle % self.cpg_manager.interval() as u64 == 0) {
       ...
   }
   ```

5. `src/cognitive_loop/types/telemetry.rs` (CycleMetadata) — Add field:
   ```rust
   #[cfg(feature = "cpg")]
   pub cpg: Option<CpgTelemetry>,
   ```

   Populate after Phase B:
   ```rust
   #[cfg(feature = "cpg")]
   { metadata.cpg = Some(self.cpg_manager.telemetry().clone()); }
   ```

### 2b. Spectral Manager Integration

Same pattern as CPG:

1. `config.rs`:
   ```rust
   #[cfg(feature = "spectral_state")]
   pub enable_spectral: bool,
   #[cfg(feature = "spectral_state")]
   pub spectral_config: SpectralManagerConfig,
   ```

2. `mod.rs` (CLS): field `spectral_manager: SpectralManager`

3. `constructor.rs`: `SpectralManager::new(config.spectral_config.clone())`

4. `cycle_phase_dynamics.rs` Phase B:
   ```rust
   #[cfg(feature = "spectral_state")]
   if self.config.enable_spectral {
       // Spectral manager records state EVERY cycle (for history continuity)
       // but only runs full analysis at its interval
       self.spectral_manager.record_state(&snapshot.compressed_state);
       if cycle % self.spectral_manager.interval() as u64 == 0 {
           let spectral_output = self.spectral_manager.process(&snapshot);
           collector.record("spectral_twin", spectral_output);
       }
   }
   ```

   **Important**: The spectral manager needs to record state every cycle
   (not just at interval ticks) to maintain a continuous history. The
   `process()` method already handles recording + analysis, but if we
   only call it at interval 67, we get gaps in the time series. Solution:
   call `record_state()` every cycle, `process()` at interval.

   Actually, looking at the implementation, `process()` calls `record_state()`
   internally. So we can just call `process()` every cycle and let the
   cognitive loop's interval logic gate the output integration. Or we
   split: `record_state()` every cycle, `process()` at interval.

5. `types/telemetry.rs`:
   ```rust
   #[cfg(feature = "spectral_state")]
   pub spectral: Option<SpectralTelemetry>,
   ```

### 2c. Complex CfC Neuron Integration

This is different — it replaces the temporal backend, not a manager.

1. `src/cognitive_loop/config.rs` — Add variant:
   ```rust
   pub enum TemporalBackend {
       CfC,
       HdcLtc,
       HierarchicalCfC,
       #[cfg(feature = "complex_cfc")]
       ComplexCfC,
   }
   ```

2. `src/cognitive_loop/temporal_network.rs` — Add variant:
   ```rust
   #[cfg(feature = "complex_cfc")]
   ComplexCfC(ComplexCfcNeuron),
   ```

   Implement `read_state()`, `inject_state()`, `evolve()` for the new variant.
   `read_state()` calls `neuron.to_continuous_hv()` and packs into Vec<f32>.

3. `src/cognitive_loop/constructor.rs` — Match on `TemporalBackend::ComplexCfC`:
   ```rust
   #[cfg(feature = "complex_cfc")]
   TemporalBackend::ComplexCfC => {
       TemporalNetwork::ComplexCfC(ComplexCfcNeuron::new_default(genesis_seed))
   }
   ```

### 2d. Fourier Motor Preset Activation

No integration code needed — just config. Users enable it via:
```rust
let config = CognitiveLoopConfig {
    cfc_config: UnifiedConfig::with_motor_rhythm(),
    ..default
};
```

Or at runtime:
```rust
service.temporal_network.update_fourier_frequencies(&[8.0, 13.0, 30.0]);
```

---

## Priority 3: Cross-Coupling

### 3a. Spectral → CPG (gamma detection boosts motor confidence)

In `cycle_phase_dynamics.rs` after both managers have run:
```rust
#[cfg(all(feature = "cpg", feature = "spectral_state"))]
if self.config.enable_cpg && self.config.enable_spectral {
    let rel_gamma = self.spectral_manager.telemetry().band_power.relative_gamma();
    if rel_gamma > 0.3 {
        // High gamma = active binding = motor system should be confident
        // (no extra output needed — spectral already boosts confidence)
    }
}
```

### 3b. CPG → Spectral (sync index as motor rhythm signal)

The CPG sync index is already in `CpgTelemetry` and accessible via
`self.cpg_manager.telemetry().sync_index`. The spectral manager reads
`compressed_state` which already contains the CfC hidden state that
CPG modulates (via SubsystemOutput → Phase C integration → next cycle).
The coupling is implicit through the shared state.

### 3c. Spectral → Complex CfC (adapt eigenvalue frequencies)

When spectral analysis shows dominant frequencies that don't match the
eigenvalue distribution, adjust eigenvalues toward the useful frequencies:

```rust
#[cfg(all(feature = "complex_cfc", feature = "spectral_state"))]
if let TemporalNetwork::ComplexCfC(ref mut neuron) = self.temporal_network {
    let dominant = &self.spectral_manager.telemetry().dominant_band;
    // Future: adapt eigenvalue imaginary parts toward dominant band
    // This closes the loop: neuron oscillation frequencies adapt
    // based on what the spectral twin detects as useful
}
```

This is the most speculative coupling and should be implemented after
the basic wiring (2a-2d) is validated.

### 3d. Substrate → CPG + Fourier (tau_factor scaling)

```rust
#[cfg(feature = "cpg")]
self.cpg_manager.set_tau_factor(self.substrate_manager.tau_factor() as f64);
```

Already in the CPG API, just needs to be called when substrate changes.

---

## Priority 4: Validation

### 4a. Integration Test: Oscillation Pipeline

Create `tests/oscillation_pipeline.rs`:

```rust
#[test]
fn test_fourier_injection_changes_consciousness() {
    // Build CLS with motor rhythm enabled
    // Run 200 cycles
    // Compare consciousness_level with and without Fourier
    // Expect measurable difference
}

#[test]
#[cfg(feature = "cpg")]
fn test_cpg_produces_rhythmic_output() {
    // Build CLS with CPG enabled
    // Set motor_active = true
    // Run 100 cycles
    // Verify CpgTelemetry shows oscillatory output
    // Verify sync_index is reasonable for walk gait
}

#[test]
#[cfg(feature = "spectral_state")]
fn test_spectral_twin_detects_injected_frequency() {
    // Build CLS with spectral_state enabled
    // Inject a known frequency into the CfC state
    // Run enough cycles for history to fill
    // Verify spectral telemetry shows the correct dominant band
}

#[test]
#[cfg(feature = "complex_cfc")]
fn test_complex_cfc_oscillates_in_pipeline() {
    // Build CLS with ComplexCfC backend
    // Run 200 cycles
    // Verify state norm is non-monotonic (oscillation)
    // Verify no NaN/Inf in consciousness metrics
}
```

### 4b. Psych-Bench Comparison

Run psych-bench with and without complex_cfc:
```bash
cargo test -p symthaea-psych-bench --features complex_cfc -- --nocapture
```

Compare z-scores, especially for:
- Mathematics benchmarks (the claimed bottleneck area)
- Attention benchmarks (oscillation should help temporal attention)

### 4c. Consciousness Metric Impact

After wiring, compare:
- Phi (IIT) with vs without Fourier/complex CfC
- Spectral entropy of CfC state
- Theta-gamma PAC values during different task types

---

## Wiring Checklist

- [ ] CPG Manager: config fields
- [ ] CPG Manager: CLS field
- [ ] CPG Manager: constructor init
- [ ] CPG Manager: Phase B processing
- [ ] CPG Manager: telemetry in CycleMetadata
- [ ] Spectral Manager: config fields
- [ ] Spectral Manager: CLS field
- [ ] Spectral Manager: constructor init
- [ ] Spectral Manager: Phase B processing (record every cycle, analyze at interval)
- [ ] Spectral Manager: telemetry in CycleMetadata
- [ ] Complex CfC: TemporalBackend variant
- [ ] Complex CfC: TemporalNetwork variant
- [ ] Complex CfC: constructor match arm
- [ ] Substrate → CPG tau_factor wiring
- [ ] Integration test: oscillation_pipeline.rs
- [ ] Psych-bench comparison run
