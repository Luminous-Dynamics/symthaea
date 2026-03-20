# CfC Oscillation Integration Guide

Technical reference for wiring CPG (Central Pattern Generator) and Spectral Twin oscillation
subsystems into the CfC dynamics pipeline. This document covers the current state, gap analysis,
specific integration points, implementation plan, test plan, and risks.

---

## 1. Current State

### 1.1 CPG Manager (Kuramoto Coupled Oscillators)

- **File**: `src/cognitive_loop/managers/cpg_manager.rs` (~1100 LOC, 20 tests)
- **Feature gate**: `#[cfg(feature = "cpg")]`
- **Interval**: 59 (co-prime with all other manager intervals)
- **Model**: 8 Kuramoto oscillators in a quad-locomotion layout (4 limbs x 2 joints)
- **Gait presets**: Walk (2 Hz), Trot (4 Hz), Gallop (6 Hz) with distinct coupling matrices
- **Key outputs**:
  - `sync_index` (Kuramoto order parameter r, 0.0-1.0)
  - Per-oscillator phases and output signals
  - Desynchronization alerts (safety-relevant during active motor)
- **Current couplings**:
  - **Substrate -> CPG**: `tau_factor` scales oscillator frequency (`cycle_phase_dynamics.rs:504-505`)
  - **Arousal -> CPG**: `snapshot.arousal` modulates frequency via Yerkes-Dodson scaling (`cpg_manager.rs:540`)
  - **CPG -> SubsystemOutput**: Desync during idle produces `exploration_delta`; critical desync produces `arousal_delta` and `ANOMALY_DETECTED` flag (`cpg_manager.rs:567-579`)
- **Telemetry**: 4 fields wired to `CycleMetadata` (`cycle_phase_output.rs:861-867`): `cpg_sync_index`, `cpg_mean_freq`, `cpg_motor_active`, `cpg_desync_alert`

### 1.2 Spectral Twin Manager (Frequency-Domain CfC Analysis)

- **File**: `src/cognitive_loop/managers/spectral_manager.rs` (~650 LOC, 11 tests)
- **Feature gate**: `#[cfg(feature = "spectral_state")]`
- **Interval**: 67 (co-prime with all other manager intervals)
- **Model**: Rolling 128-cycle ring buffer of `compressed_state` snapshots; FFT band power analysis (delta/theta/alpha/beta/gamma), spectral entropy, theta-gamma phase-amplitude coupling (PAC)
- **Key outputs**:
  - Band power distribution (5 bands)
  - Spectral entropy (broadband richness)
  - Theta-gamma PAC strength (information integration correlate)
  - Dominant frequency band
- **Current couplings**:
  - **State recording**: `compressed_state` recorded every tick for history continuity (`cycle_phase_dynamics.rs:518-519`)
  - **Gamma -> Consciousness**: High relative gamma power produces `confidence_delta` boost (`spectral_manager.rs:366-368`)
  - **Delta -> Rest**: Delta dominance requests rest via `REQUEST_REST` flag and negative `arousal_delta` (`spectral_manager.rs:372-375`)
  - **Entropy -> Exploration**: High spectral entropy produces `exploration_delta` (`spectral_manager.rs:380-383`)
  - **PAC -> Confidence**: Theta-gamma coupling above threshold produces `confidence_delta` (`spectral_manager.rs:387-389`)
- **Telemetry**: 8 fields wired to `CycleMetadata` (via `SpectrumInfo` Pulse pane)

### 1.3 Bifurcation Detection (Error Oscillation)

- **File**: `src/cognitive_loop/helpers/cycle_phases_urgency.rs:370-386`
- **Model**: `oscillation_ratio` computed from prediction error sign-change rate in the error history buffer. High oscillation (> `ERROR_OSCILLATION_BIFURCATION` threshold) triggers bifurcation response: LR freeze, exploration boost, NE impulse
- **Threshold**: `ERROR_OSCILLATION_BIFURCATION` in `thresholds.rs:2417`
- **Output**: `oscillation_ratio` flows through `PerceptionPhaseResult` -> `CycleMetadata.oscillation_ratio` and `error_bifurcation_response`

### 1.4 SubsystemOutput Integration Path

All manager outputs flow through `SubsystemCollector`:
1. Managers call `self.subsystem_collector.record(name, output)` during Phase B (`cycle_phase_dynamics.rs:342-725`)
2. Collector integrates in Phase 4 (`cycle_phase_output.rs:1533`): sums deltas for confidence, exploration, arousal, valence; multiplies LR modulations
3. Integrated result applied to CLS state (`cycle_phase_output.rs:1534-1556`)

---

## 2. Gap Analysis

### 2.1 CPG Phase Output is Isolated from CfC Dynamics

The CPG produces rich rhythmic phase information (8 oscillator phases, sync index, mean frequency),
but this data **never modulates the CfC temporal step**. The delta_t computation chain
(`cycle_phase_dynamics.rs:3234-3248`) multiplies 12 factors:

1. `config.cfc_config.delta_t` (base)
2. `resonance_tau_factor` (resonator familiarity)
3. `arousal_tau_factor` (body arousal)
4. `codebook_tau_factor` (codebook novelty)
5. `arousal_recovery_tau_factor` (trap recovery)
6. `fep_tau_factor` (FEP surprise)
7. `coherence_velocity_tau_factor` (coherence rate)
8. `prediction_horizon_tau` (prediction error)
9. `interoceptive_signals.tau_slowdown_factor` (somatic)
10. `substrate_manager.tau_factor` (substrate speed)
11. `thermal_tau_factor` (platform temperature)
12. `neuroevo_tau_factor` (evolved tau ratio)

**Missing**: No CPG-derived factor (e.g., oscillator sync or phase) modulates delta_t.

### 2.2 Spectral Band Power Does Not Modulate CfC Dynamics

The Spectral Twin detects neural oscillation bands in the CfC hidden state but only produces
indirect modulations through `SubsystemOutput` (confidence, exploration, arousal deltas).
It does **not** feed back into the CfC step itself. There is no spectral-derived tau factor.

### 2.3 No CPG -> Consciousness Coupling

CPG sync_index is a direct analog of neural synchronization, which is a core component of
Integrated Information Theory (binding requires synchronized activity). Currently, `sync_index`
is only used for telemetry and desync alerts -- it does not influence the consciousness equation
(`ConsciousnessEquationV2`) or Phi computation.

### 2.4 No Spectral -> CfC Hidden State Feedback

The Spectral Twin reads CfC state but never writes back. In biological brains, oscillatory
dynamics are bidirectional: neural rhythms constrain and are constrained by spike timing.
The current architecture is read-only.

### 2.5 No Cross-Coupling Between CPG and Spectral Twin

CPG operates at the motor level (8 oscillators, gait rhythms) and the Spectral Twin operates
at the cognitive level (CfC hidden state frequency analysis). In biological systems, motor
rhythms and cortical rhythms are coupled (sensorimotor mu rhythm, beta rebound). Currently
these two subsystems are completely independent.

### 2.6 Bifurcation Detection is Input-Side Only

`oscillation_ratio` is computed from **prediction error** history (input-derived). It does not
incorporate CPG sync state or spectral dynamics. A more complete bifurcation detector would
fuse error oscillation with CPG desync and spectral entropy spikes.

---

## 3. Integration Points

### 3.1 CPG Sync -> delta_t Modulation

**Location**: `src/cognitive_loop/cycle_phase_dynamics.rs:3234-3248` (the delta_t product chain)

**Rationale**: High CPG synchronization (r -> 1.0) indicates stable motor rhythm. The CfC can
afford to take larger temporal steps (faster integration) when motor output is coherent.
Desynchronization should slow dynamics for careful state correction.

**Proposed factor**: `cpg_sync_tau_factor`
```rust
// After line 3232 (neuroevo_tau_factor), before the delta_t product:
#[cfg(feature = "cpg")]
let cpg_sync_tau_factor = {
    let r = self.cpg_manager.sync_index();
    if self.cpg_manager.config().motor_active {
        // High sync → speed up (1.0 + 0.1), low sync → slow down (1.0 - 0.1)
        // Grillner (2006): stable CPG = safe to integrate faster
        1.0 + (r as f32 - 0.5) * CPG_SYNC_TAU_SCALE  // new constant, suggest 0.2
    } else {
        1.0 // No modulation when motor is idle
    }
};
#[cfg(not(feature = "cpg"))]
let cpg_sync_tau_factor: f32 = 1.0;
```

Then add `* cpg_sync_tau_factor` to the delta_t product at line 3248.

### 3.2 Spectral Dominant Band -> delta_t Modulation

**Location**: Same delta_t product chain (`cycle_phase_dynamics.rs:3234-3248`)

**Rationale**: Different oscillatory regimes imply different optimal integration timescales.
Theta dominance (memory encoding) benefits from slower, deeper integration. Beta/gamma
dominance (active processing) benefits from faster temporal steps.

**Proposed factor**: `spectral_band_tau_factor`
```rust
// After cpg_sync_tau_factor:
#[cfg(feature = "spectral_state")]
let spectral_band_tau_factor = {
    let telem = self.spectral_manager.telemetry();
    if telem.history_len >= SPECTRAL_MIN_HISTORY as usize {
        match telem.dominant_band.as_str() {
            "theta" => SPECTRAL_THETA_TAU_FACTOR,   // suggest 0.9 (slower, consolidate)
            "gamma" => SPECTRAL_GAMMA_TAU_FACTOR,   // suggest 1.1 (faster, bind)
            "delta" => SPECTRAL_DELTA_TAU_FACTOR,    // suggest 0.85 (much slower, rest)
            _ => 1.0
        }
    } else {
        1.0
    }
};
#[cfg(not(feature = "spectral_state"))]
let spectral_band_tau_factor: f32 = 1.0;
```

### 3.3 CPG Sync -> Consciousness Coupling

**Location**: `src/cognitive_loop/cycle_phase_feedback.rs`

The `consciousness_engine` computes Phi from the CfC connectivity matrix. CPG sync_index is a
direct measure of neural binding (Kuramoto order parameter = global synchronization), which maps
directly to IIT's integration concept.

**Proposed coupling**: Multiply the binding term in the consciousness equation by a CPG-derived
factor, similar to how `binding_capability` from the substrate already modulates IIT coherence
(see `consciousness_engine/measure.rs`).

**Implementation**: In `cycle_phase_feedback.rs`, after consciousness computation:
```rust
#[cfg(feature = "cpg")]
{
    let sync = self.cpg_manager.sync_index() as f64;
    // CPG sync modulates consciousness +-3%
    // Varela (2001): neural synchrony is a correlate of conscious binding
    let cpg_consciousness_mod = 1.0 + (sync - 0.5) * CPG_CONSCIOUSNESS_COUPLING_SCALE;
    consciousness_level *= cpg_consciousness_mod.clamp(0.97, 1.03);
}
```

### 3.4 Spectral PAC -> Consciousness Coupling

**Location**: Same feedback phase

**Rationale**: Theta-gamma PAC is one of the strongest neural correlates of conscious information
integration (Canolty & Knight 2010). Currently PAC only boosts confidence via SubsystemOutput.
It should directly modulate the consciousness score.

**Proposed coupling**: Similar to 3.3 but using `theta_gamma_pac`:
```rust
#[cfg(feature = "spectral_state")]
{
    let pac = self.spectral_manager.telemetry().theta_gamma_pac;
    if pac > SPECTRAL_PAC_THRESHOLD as f64 {
        let pac_consciousness_boost = (pac - SPECTRAL_PAC_THRESHOLD as f64)
            * SPECTRAL_PAC_CONSCIOUSNESS_SCALE;  // new constant, suggest 0.05
        consciousness_level += pac_consciousness_boost.min(0.03);
    }
}
```

### 3.5 Cross-Coupling: Spectral Band Power -> CPG Frequency

**Location**: `src/cognitive_loop/cycle_phase_dynamics.rs:500-511` (CPG section in Phase B)

**Rationale**: Cortical beta rhythm (13-30 Hz) is associated with motor execution (Pfurtscheller
1999). When the Spectral Twin detects beta dominance, the CPG should upregulate frequency.
Alpha dominance (motor-ready but idle) should maintain baseline.

**Implementation**: Before CPG processing:
```rust
#[cfg(all(feature = "cpg", feature = "spectral_state"))]
{
    let telem = self.spectral_manager.telemetry();
    let beta_power = telem.band_power.beta;
    let total = telem.band_power.total();
    if total > 1e-6 {
        let beta_ratio = beta_power / total;
        // Beta dominance → CPG frequency boost (motor execution)
        if beta_ratio > SPECTRAL_BETA_CPG_THRESHOLD {
            let freq_scale = 1.0 + (beta_ratio - SPECTRAL_BETA_CPG_THRESHOLD) * 0.5;
            // Apply via existing modulate_frequency or adjust natural_freq directly
        }
    }
}
```

### 3.6 Enhanced Bifurcation Detection

**Location**: `src/cognitive_loop/helpers/cycle_phases_urgency.rs:370-386`

**Rationale**: Fuse the existing `oscillation_ratio` (error-based) with CPG desync and spectral
entropy to create a multi-signal bifurcation detector. Kelso (1995) shows that critical
transitions exhibit signatures across multiple observation channels simultaneously.

**Proposed enhancement**: In the bifurcation check block:
```rust
let mut bifurcation_evidence = 0u32;
if oscillation_ratio > ERROR_OSCILLATION_BIFURCATION {
    bifurcation_evidence += 1;
}
#[cfg(feature = "cpg")]
if cpg_sync_index < CPG_CRITICAL_DESYNC {
    bifurcation_evidence += 1;
}
#[cfg(feature = "spectral_state")]
if spectral_entropy > SPECTRAL_BIFURCATION_ENTROPY_THRESHOLD {
    bifurcation_evidence += 1;
}
// Require 2+ signals for bifurcation (reduces false positives)
if bifurcation_evidence >= 2 { /* trigger bifurcation response */ }
```

This requires threading CPG/spectral state into the urgency computation, which currently only
sees `CycleSnapshot` and error history.

---

## 4. Implementation Plan

### Phase A: New Constants (thresholds.rs)

Add to `src/cognitive_loop/thresholds.rs` after the existing CPG/Spectral constant blocks
(after line 4379):

| Constant | Value | Description |
|----------|-------|-------------|
| `CPG_SYNC_TAU_SCALE` | 0.2 | Sync index -> delta_t scaling magnitude |
| `CPG_CONSCIOUSNESS_COUPLING_SCALE` | 0.06 | Sync -> consciousness modulation |
| `SPECTRAL_THETA_TAU_FACTOR` | 0.92 | Theta dominance -> slower CfC dynamics |
| `SPECTRAL_GAMMA_TAU_FACTOR` | 1.08 | Gamma dominance -> faster CfC dynamics |
| `SPECTRAL_DELTA_TAU_FACTOR` | 0.85 | Delta dominance -> much slower dynamics |
| `SPECTRAL_PAC_CONSCIOUSNESS_SCALE` | 0.05 | PAC -> consciousness boost scale |
| `SPECTRAL_BETA_CPG_THRESHOLD` | 0.35 | Beta relative power to trigger CPG coupling |

Add ordering tests in the existing `#[cfg(test)] mod tests` block.

### Phase B: delta_t Integration (Integration Points 3.1 + 3.2)

1. In `phase_dynamics_cfc_planning()` (`cycle_phase_dynamics.rs`), after `neuroevo_tau_factor`
   (line ~3232), add CPG sync tau factor and spectral band tau factor computations.
2. Add both factors to the delta_t product chain at line 3248.
3. Add NaN guards (`.clamp(0.5, 2.0)` on each new factor).
4. Update the `dynamics_delta_t_finite_across_many_cycles` test (line 4269) to document the
   new factor count (14 factors total, up from 12).

### Phase C: Consciousness Coupling (Integration Points 3.3 + 3.4)

1. In `cycle_phase_feedback.rs`, after consciousness level computation, add CPG sync and
   spectral PAC modulations.
2. Clamp total modulation to prevent runaway: each coupling should be bounded to +/-3%.
3. Add the new coupling values to `CycleMetadata` telemetry for observability.

### Phase D: Cross-Coupling (Integration Point 3.5)

1. In `cycle_phase_dynamics.rs` Phase B, before CPG processing, inject spectral beta power
   as a CPG frequency modulator.
2. Gate behind both feature flags: `#[cfg(all(feature = "cpg", feature = "spectral_state"))]`

### Phase E: Enhanced Bifurcation (Integration Point 3.6)

1. Extend `CycleSnapshot` with optional `cpg_sync_index: Option<f64>` and
   `spectral_entropy: Option<f64>` fields.
2. Populate them in Phase A (`cycle_phase_dynamics.rs:317-339`) from manager state.
3. Update urgency computation to fuse multiple bifurcation signals.

### Phase F: Integration Testing

Write tests (see Section 5 below).

---

## 5. Test Plan

### 5.1 Unit Tests (in respective manager files)

**CPG sync tau factor** (`cpg_manager.rs`):
- `test_sync_tau_factor_high_sync_speeds_up`: r=0.9 should produce factor > 1.0
- `test_sync_tau_factor_low_sync_slows_down`: r=0.2 should produce factor < 1.0
- `test_sync_tau_factor_idle_neutral`: motor_active=false should produce factor = 1.0

**Spectral band tau factor** (`spectral_manager.rs`):
- `test_theta_dominant_slows_dynamics`: Feed 6 Hz sine, verify tau factor < 1.0
- `test_gamma_dominant_speeds_dynamics`: Feed 40 Hz sine, verify tau factor > 1.0
- `test_warmup_neutral`: Insufficient history should produce factor = 1.0

### 5.2 Integration Tests (new file: `tests/oscillation_integration.rs`)

These tests should verify the full wiring through the cognitive loop:

1. **`test_cpg_sync_modulates_delta_t`**: Enable `cpg` feature, set motor_active=true, run
   cycles with high coupling (fast sync convergence). Compare delta_t with and without CPG
   feature. Assert delta_t differs by at least `CPG_SYNC_TAU_SCALE * 0.1`.

2. **`test_spectral_band_modulates_delta_t`**: Enable `spectral_state` feature, feed
   oscillatory input that produces theta dominance. After warmup (32+ cycles), verify delta_t
   is reduced relative to baseline.

3. **`test_cpg_sync_modulates_consciousness`**: Enable `cpg` feature, run cycles to steady
   state. Compare consciousness_level between high-sync (coupled oscillators) and low-sync
   (scrambled phases, zero coupling). Assert consciousness difference within expected range.

4. **`test_spectral_pac_boosts_consciousness`**: Enable `spectral_state`, feed mixed
   theta+gamma signal. Verify consciousness_level is higher than with flat signal.

5. **`test_cross_coupling_beta_boosts_cpg_freq`**: Enable both features, feed beta-dominant
   signal. Verify CPG mean_freq increases relative to baseline.

6. **`test_multi_signal_bifurcation_detection`**: Enable both features, drive system into
   high error oscillation + CPG desync + spectral entropy spike simultaneously. Verify
   bifurcation response fires. Then verify single-signal case does NOT fire (specificity).

7. **`test_oscillation_integration_nan_guard`**: Feed extreme inputs, verify all new factors
   remain finite and within clamp bounds.

### 5.3 Property Tests (new file: `tests/proptest_oscillation.rs`)

1. **`prop_cpg_tau_factor_bounded`**: For any sync_index in [0,1], factor is in [0.5, 2.0].
2. **`prop_spectral_tau_factor_bounded`**: For any band power distribution, factor is in [0.5, 2.0].
3. **`prop_consciousness_with_oscillation_bounded`**: With both features enabled, consciousness_level stays in [0, 1] after all couplings.
4. **`prop_delta_t_product_positive`**: With all 14 factors, delta_t remains strictly positive.

---

## 6. Risks

### 6.1 Feedback Instability (HIGH)

The most dangerous risk. Adding CPG sync and spectral dynamics to the delta_t chain creates
new feedback loops:

- CfC state -> Spectral analysis -> spectral_band_tau_factor -> delta_t -> CfC step -> CfC state

This is a closed loop. If spectral analysis detects gamma dominance and speeds up dynamics,
the faster dynamics may produce even more gamma, creating a runaway acceleration. Similarly,
delta dominance could create a runaway deceleration (CfC stops evolving -> more delta -> slower).

**Mitigation**:
- Clamp all new tau factors to conservative ranges (e.g., [0.85, 1.15]).
- Use EMA smoothing on the spectral tau factor (not raw per-tick values).
- The existing `delta_t` product already has 12 factors, so each new factor's marginal
  contribution is diluted. But monitor total delta_t range in tests.

### 6.2 Feature Interaction Complexity (MEDIUM)

With `cpg` and `spectral_state` as independent feature flags, there are 4 combinations:
neither, cpg-only, spectral-only, both. The cross-coupling (Section 3.5) requires both.
Testing all 4 combinations in CI adds matrix complexity.

**Mitigation**: The existing CI feature matrix (`symthaea-ci.yml`) already tests 49 feature
combinations. Add `cpg` + `spectral_state` to the matrix. Gate cross-coupling behind
`#[cfg(all(feature = "cpg", feature = "spectral_state"))]`.

### 6.3 Consciousness Score Inflation (MEDIUM)

Adding both CPG sync coupling (+3%) and spectral PAC coupling (+3%) means consciousness_level
could be boosted by up to +6% when both are active and favorable. Combined with existing
couplings (coherence field +/-5%, substrate feasibility, substrate binding/workspace/attention),
the total modulation range grows.

**Mitigation**: After all oscillation couplings, apply a single combined clamp. Consider
reducing individual coupling strengths if the combined effect is too large. The total
oscillation contribution should not exceed +/-5%.

### 6.4 Performance Impact (LOW)

The new computations are lightweight:
- CPG sync_index is already computed every interval-59 tick
- Spectral telemetry is already computed every interval-67 tick
- Reading their state for delta_t is O(1) field access
- No additional FFT or matrix operations

**Mitigation**: Profile with `module_timings` to verify zero measurable overhead.

### 6.5 CPG Warmup Interaction (LOW)

CPG starts with preset phases (gait-specific initialization), so `sync_index` may be
artificially high on the first few ticks before the Kuramoto dynamics settle. This could
produce a spurious delta_t boost during warmup.

**Mitigation**: Gate CPG tau factor behind `cycle_count > DYNAMICS_STARTUP_WARMUP_CYCLES`
(the same warmup guard used by other dynamics subsystems, currently set to 3).

### 6.6 Spectral Warmup (LOW)

The Spectral Twin requires `SPECTRAL_MIN_HISTORY` (32) cycles before producing meaningful
analysis. The spectral tau factor should return 1.0 during warmup. The proposed implementation
already includes this guard (`history_len >= SPECTRAL_MIN_HISTORY`).

---

## References

- Brown, T.G. (1911). The intrinsic factors in the act of progression in the mammal.
- Buzsaki, G. (2006). Rhythms of the Brain. Oxford University Press.
- Canolty, R.T. & Knight, R.T. (2010). The functional role of cross-frequency coupling. Trends in Cognitive Sciences.
- Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience.
- Grillner, S. (2006). Biological pattern generation: the cellular and computational logic of networks in motion. Neuron.
- Kelso, J.A.S. (1995). Dynamic Patterns: The Self-Organization of Brain and Behavior. MIT Press.
- Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators. Lecture Notes in Physics.
- Pfurtscheller, G. & Lopes da Silva, F. (1999). Event-related EEG/MEG synchronization and desynchronization. Clinical Neurophysiology.
- Varela, F., Lachaux, J.-P., Rodriguez, E., & Martinerie, J. (2001). The brainweb: phase synchronization and large-scale integration. Nature Reviews Neuroscience.
