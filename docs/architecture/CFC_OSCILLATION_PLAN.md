# CfC Oscillation & Rhythm Architecture Plan

## Status: ALL PHASES DONE (2026-03-15/16)

A 4-phase plan to give Symthaea's HDC-LTC/CfC unified neuron native rhythm
and sustained oscillation capabilities.

---

## The Problem

The CfC closed-form solution is a damped interpolation toward equilibrium:

```
x(t+dt) = x_inf + (x(t) - x_inf) * exp(-dt/tau)
```

The exponential decay term `exp(-dt/tau)` monotonically drives the state toward
`x_inf`, which is itself a fixed point of the activation function. For constant
input, the system has a unique fixed point that the exponential decay converges
to. **No sustained oscillation can emerge from this math alone** -- it is a
contraction mapping. The state-dependent tau modulates *speed* of convergence
but does not change the attractor topology.

This limits Symthaea's ability to:
- Generate motor rhythms (walking, grasping, breathing: 1-50 Hz)
- Maintain neural oscillations (alpha, beta, gamma bands)
- Bridge from simulation to embodied robotics

### What Already Exists

| Mechanism | Location | Status | Band |
|-----------|----------|--------|------|
| Fourier basis injection | `hdc_ltc_unified.rs:488-528` | Disabled by default; fused path **skips it entirely** (bug) | Configurable |
| Liquid-Mamba fusion | `crates/symthaea-broca/src/liquid_mamba.rs` | Working (S4/Mamba complex eigenvalues) | Sequence-level |
| Chronobiology | `src/chronobiology.rs` | Working (circadian/ultradian) | ~0.00001 Hz |
| Complex number support | `symthaea-core/src/hdc/complex.rs` | Full arithmetic incl. `exp()` | Not used for oscillation |
| Geometric ops (Riemannian) | `src/hdc/geometric_ops.rs` | SLERP, Frechet mean, PGA on S^{d-1} | Concept-space only |

### The Gap

The 1-50 Hz motor band is uncovered. Chronobiology is too slow (hours).
Mamba operates at the token/sequence level. No dedicated motor rhythm generator exists.

---

## Architectural Clarifications

### CfC State Space is Euclidean, Not Spherical

The neuron state lives in the **open ball B^d(5.0)** -- a soft-bounded region
of Euclidean space -- not on the unit hypersphere S^{d-1}.

```rust
// hdc_ltc_unified.rs:952
fn apply_state_bounds(&mut self) {
    let norm = self.state.norm();
    if norm > 5.0 {          // norm cap is 5.0, NOT 1.0
        let scale = 5.0 / norm;
        self.state.scale_in_place(scale);
    }
}
```

Weight vectors are clipped to norm <= 2.0. Tanh activation bounds components to
[-1, 1] but does not constrain the overall norm. The state is never normalized
to unit length during evolution.

The `geometric_ops.rs` module (SLERP, Frechet mean, etc.) operates on
**normalized concept vectors** (encoded percepts, memory embeddings) which are
projected onto S^{d-1} before comparison. The neuron's dynamical state is not
one of these objects. Riemannian geometry is relevant for concept-space
operations and potentially for eigenvalue parameter learning (Phase 3), but not
for CfC state stability.

### Blast Radius of Complex CfC is Contained

The complex representation is **internal to the neuron**. Everything downstream
-- consciousness engine, HAI free energy, ethics engine, prediction, training,
all 7,315+ tests -- consumes `ContinuousHV` output via a trivial interleave:

```
ComplexHdcLtcNeuron internals:
  state_re[0..8191], state_im[0..8191]
  |
  | evolve with complex eigenvalues
  | to_continuous_hv()    <-- THE BOUNDARY
  v
ContinuousHV[0..16383]   <-- everything downstream touches THIS
```

No `Hypervector` trait abstraction needed. `ContinuousHV` already IS the
abstraction boundary. Compile-time feature selection (`complex_cfc`) swaps the
neuron implementation; no runtime dispatch, no trait generics leaking upward.

### Cognitive Loop Budget

Measured rate: ~31 Hz. Budget: 20 Hz. This provides meaningful headroom for
additional computation in the neuron evolution step.

---

## Phase 1: Fix & Activate Fourier Basis Injection

**Effort**: Low (1-2 sessions)
**Impact**: High -- unblocks motor-relevant configs immediately
**Feature flag**: None (bug fix + config presets)

### Bug: Fused Path Skips Fourier

The performance-critical `evolve_closed_form_fused` (line 714) computes
`x_inf = tanh(W*x + U*u)` inline per dimension using AVX2+FMA SIMD. It does
NOT call `compute_equilibrium()` and therefore **ignores the Fourier basis
injection entirely**. Any config with `fourier_frequencies` non-empty that uses
the fused path gets silently ignored Fourier components.

### Fix

Pre-compute `compute_fourier_basis()` once before the fused loop. Index into
the cached Fourier HV as a third additive term per dimension:

```
x_inf[i] = tanh((W[i]*x[i] + U[i]*u[i] + F[i]) * scale)
```

One allocation per evolve call (the Fourier HV), not per dimension. Falls back
to existing 2-component fused path when `fourier_frequencies` is empty (zero
cost when disabled).

### Config Presets

```rust
impl UnifiedConfig {
    /// Motor rhythm preset: alpha, beta, gamma bands
    pub fn with_motor_rhythm() -> Self {
        Self {
            fourier_frequencies: vec![
                FOURIER_MOTOR_ALPHA_HZ,  // 8.0 Hz
                FOURIER_MOTOR_BETA_HZ,   // 13.0 Hz
                FOURIER_MOTOR_GAMMA_HZ,  // 30.0 Hz
            ],
            fourier_amplitude: FOURIER_MOTOR_AMPLITUDE, // 0.15
            ..Self::default()
        }
    }

    /// Substrate-scaled rhythm (frequencies shift with tau_factor)
    pub fn with_substrate_scaled_rhythm(tau_factor: f32) -> Self {
        let base = Self::with_motor_rhythm();
        Self {
            fourier_frequencies: base.fourier_frequencies
                .iter()
                .map(|f| f * tau_factor)
                .collect(),
            ..base
        }
    }
}
```

### Named Constants (`thresholds.rs`)

| Constant | Value | Citation |
|----------|-------|----------|
| `FOURIER_MOTOR_ALPHA_HZ` | 8.0 | Pfurtscheller (1999) -- alpha band motor planning |
| `FOURIER_MOTOR_BETA_HZ` | 13.0 | Pfurtscheller & Lopes da Silva (1999) -- beta motor execution |
| `FOURIER_MOTOR_GAMMA_HZ` | 30.0 | Crone et al. (1998) -- gamma fine motor control |
| `FOURIER_MOTOR_AMPLITUDE` | 0.15 | Strong enough to influence equilibrium, not dominate |
| `FOURIER_AMPLITUDE_MAX` | 0.5 | Safety cap -- beyond this, oscillation overwhelms content |

### Files Modified

- `symthaea-core/src/hdc/hdc_ltc_unified.rs` -- fix fused path, add presets
- `symthaea/src/cognitive_loop/thresholds.rs` -- named constants

### Tests

- `test_fourier_fused_matches_nonfused` -- parity between fused and non-fused paths when Fourier is active
- `test_fourier_motor_preset_oscillates` -- with motor config, repeated evolution produces non-monotonic state norm
- `test_fourier_substrate_scaling` -- frequency shift with tau_factor
- `test_fourier_disabled_zero_cost` -- empty frequencies = identical to current behavior

---

## Phase 2: Central Pattern Generator Module

**Effort**: Medium (2-3 sessions)
**Impact**: High -- biologically grounded motor rhythm generation
**Feature flag**: `cpg`
**Parallelizable** with Phase 1 (different files, different concepts)

### Scientific Basis

Biological CPGs in the spinal cord generate locomotion rhythms without cortical
input (Brown 1911, Grillner 2006). The Kuramoto model (Kuramoto 1975) is the
standard for coupled oscillator synchronization. CPGs receive descending
frequency/amplitude commands from motor cortex but handle the rhythm internally.

### Architecture

`CpgManager` implementing `CognitiveSubsystem` at interval **59** (co-prime
with existing intervals: 7, 11, 13, 19, 29, 37, 41, 53).

```
Motor cortex (CfC state)
  |
  | frequency/amplitude commands
  v
CpgManager [8 Kuramoto oscillators]
  | dtheta_i/dt = omega_i + sum_j(K_ij * sin(theta_j - theta_i))
  |
  | phase-locked timing signals
  v
MotorOutputBridge
```

### Core Components

**`CpgOscillator`** (per-oscillator state):
- `phase: f64` -- current phase [0, 2pi)
- `natural_freq: f64` -- intrinsic frequency (Hz)
- `amplitude: f64` -- output scaling
- `output: f64` -- sin(phase) * amplitude

**`CpgManager`** (subsystem):
- `oscillators: Vec<CpgOscillator>` -- default 8 (quad locomotion)
- `coupling: Vec<Vec<f64>>` -- Kuramoto coupling matrix K_ij
- `gait: GaitPreset` -- phase relationships for gaits
- `sync_index: f64` -- Kuramoto order parameter r = |1/N * sum(exp(i*theta_j))|

**`GaitPreset`** enum:
- `Walk` -- alternating limbs (180-degree phase offset)
- `Trot` -- diagonal pairs in phase
- `Gallop` -- front pair leads, back pair follows
- `Custom(CouplingMatrix)` -- user-defined

### Integration Points

| System | Coupling |
|--------|----------|
| CycleSnapshot.arousal | Modulates oscillator frequency (higher arousal = faster gait) |
| compressed_state motor channels | Descending frequency/amplitude commands |
| SubstrateManager | `CorticalRegion::Motor` substrate determines CPG temporal resolution |
| MotorOutputBridge | CPG output as rhythmic timing commands (`CpgMotorSignal`) |
| SwarmManager | Synchronized locomotion via shared CPG phase (`SwarmEvent`) |

### Named Constants (`thresholds.rs`)

| Constant | Value | Citation |
|----------|-------|----------|
| `CPG_DEFAULT_COUPLING_K` | 2.0 | Kuramoto (1975) -- critical coupling for synchronization |
| `CPG_WALK_FREQ_HZ` | 2.0 | Grillner (2006) -- human walking ~2 Hz |
| `CPG_TROT_FREQ_HZ` | 4.0 | Quadruped trot frequency |
| `CPG_AROUSAL_FREQ_SCALE` | 0.5 | Arousal-to-frequency modulation gain |
| `CPG_SYNC_ORDER_THRESHOLD` | 0.8 | Order parameter threshold for "synchronized" |
| `CPG_DESYNC_EXPLORATION_BOOST` | 0.02 | Desynchronization nudges exploration |
| `CPG_INTERVAL` | 59 | Co-prime tick interval |

### Note on Riemannian Geometry

Kuramoto oscillator phases live on S^1 (the circle). The model handles this
natively with `sin(theta_j - theta_i)`, which is intrinsically periodic. The
order parameter `r = |1/N * sum(exp(i*theta_j))|` is already circular. Full
Riemannian machinery (SLERP, log/exp maps) from `geometric_ops.rs` is not
needed here -- it solves a different problem at a different scale.

### Files

- **Create**: `src/cognitive_loop/managers/cpg_manager.rs` (~600-800 LOC)
- **Modify**: `src/cognitive_loop/managers/mod.rs` -- register module
- **Modify**: `src/cognitive_loop/config.rs` -- `CpgConfig`, `enable_cpg`
- **Modify**: `src/cognitive_loop/cycle_phase_dynamics.rs` -- Phase B processing
- **Modify**: `src/cognitive_loop/motor_output_bridge.rs` -- `CpgMotorSignal`
- **Modify**: `src/cognitive_loop/thresholds.rs` -- named constants
- **Modify**: `Cargo.toml` -- `cpg = []` feature

### Safety Integration: Motor Desynchronization Alerting

The Kuramoto order parameter `r` (already computed by CpgManager) feeds into
SafetyAgent as a motor coherence signal. If oscillators desynchronize during
active motor commands, this indicates the motor subsystem is "stumbling" --
a safety-relevant condition that should be flagged before physical consequences.

**Mechanism**: CpgManager includes `sync_index` in its `SubsystemOutput`. The
SafetyAgent reads this alongside the current gait's expected minimum coherence:

| Gait | Expected min `r` | Rationale |
|------|-------------------|-----------|
| Walk | 0.7 | High coherence (alternating 180-degree phase) |
| Trot | 0.6 | Diagonal pairs, moderate coherence |
| Gallop | 0.4 | Asymmetric phase, inherently lower coherence |

Alert levels:
- `r < gait.expected_min_sync` during active motor command: **SafetyLevel::Orange** (motor desync warning)
- `r < 0.2` during any motor state: **SafetyLevel::Red** (total motor incoherence)
- `r < CPG_DESYNC_EXPLORATION_BOOST` threshold when idle: no alert (desync during rest is normal)

Named constants:

| Constant | Value | Citation |
|----------|-------|----------|
| `CPG_WALK_MIN_SYNC` | 0.7 | Expected coherence for alternating gait |
| `CPG_TROT_MIN_SYNC` | 0.6 | Expected coherence for diagonal gait |
| `CPG_GALLOP_MIN_SYNC` | 0.4 | Expected coherence for asymmetric gait |
| `CPG_CRITICAL_DESYNC` | 0.2 | Total incoherence threshold (Red alert) |

### Tests (~25)

- Kuramoto sync: random initial phases converge to r > 0.8
- Gait preset phase relationships (walk = alternating, trot = diagonal)
- Arousal modulation: higher arousal = higher frequency
- Phase reset recovery after perturbation
- Integration: CPG output modulates motor bridge timing
- Safety: desync during walk triggers Orange alert
- Safety: critical desync triggers Red alert
- Safety: desync during idle does NOT trigger alert
- Safety: gallop tolerates lower r than walk
- Proptest: oscillator count invariants, frequency bounds, coupling symmetry

---

## Phase 3: Complex-Valued CfC Neuron

**Effort**: High (3-4 sessions)
**Impact**: High -- native oscillation in the neuron dynamics
**Feature flag**: `complex_cfc`
**Depends on**: Phase 1 validation (understanding Fourier behavior)

### Scientific Basis

Complex-valued neural networks naturally represent oscillations via Euler's
formula (Hirose 2012). The real part controls decay/growth, the imaginary part
is the oscillation frequency. This is how S4/Mamba achieves stable long-range
dependencies (Gu et al. 2022). The Broca crate already uses this approach in
its Mamba backend.

### Core Innovation

Each of 8,192 complex dimensions has a learned eigenvalue lambda_k = a_k + i*b_k:

```
x_k(t+dt) = x_inf_k + (x_k(t) - x_inf_k) * exp(lambda_k * dt)
```

When a_k < 0 (stable) and b_k != 0:

```
exp((a + bi)t) = e^{at} * (cos(bt) + i*sin(bt))
                  ^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^
                  decay    oscillation at b/(2*pi) Hz
```

The pendulum swings instead of settling. The exponential decay (a < 0) is the
stability mechanism -- no manifold constraint or Riemannian projection needed.
The oscillation frequency is a learnable parameter per dimension.

### Implementation

**`ComplexHdcLtcNeuron`** struct:
```
state_re: [f32; 8192]           // Real part
state_im: [f32; 8192]           // Imaginary part
weight_re: [f32; 8192]          // Complex weight (real)
weight_im: [f32; 8192]          // Complex weight (imaginary)
eigenvalues: Vec<Complex>       // Per-dimension eigenvalue (a_k + b_k*i)
// ... input masks, tau modulator similarly complex
```

**Eigenvalue constraints** (stability by construction):
- Real part: `a_k in [-1.0, -0.01]` -- must be negative for bounded dynamics
- Imaginary part: `b_k in [-50*pi, 50*pi]` -- covers 0-25 Hz motor band
- Learning: eigenvalues are gradient-updated with projection back to constraints

**API boundary** (`to_continuous_hv()` / `from_continuous_hv()`):
- Interleave real/imaginary into 16,384D `ContinuousHV`
- All downstream consumers (consciousness engine, HAI, ethics, prediction) work unchanged
- Round-trip is lossless (trivial interleave/deinterleave)

**Complex binding**: `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`
- 4 muls + 2 adds per complex element (vs 1 mul for real)
- SIMD kernel needed: AVX2 `vfmadd`/`vfmsub` pairs for efficient cross-terms
- ~2x throughput reduction vs real binding (4 FLOPs vs 1, but pipelined)

### SIMD Considerations

The existing `fused_tanh_avx2` kernel processes 8 f32s per AVX2 register. A
complex variant operates on 4 complex pairs (8 f32s interleaved). The key
operations:

```
// Complex multiply: (a+bi)(c+di)
real = vfmsub(a, c, vmul(b, d))   // ac - bd
imag = vfmadd(a, d, vmul(b, c))   // ad + bc
```

This is well-suited to FMA instructions. Expected throughput: ~50-60% of real
fused path, which is well within the 31 Hz -> 20 Hz budget headroom.

### Where Riemannian Geometry Applies

The `riemannian_gradient()` function from `geometric_ops.rs` is useful for
**eigenvalue learning**, not state evolution. If eigenvalues are constrained to
a stability manifold (real < 0, imaginary bounded), unconstrained SGD produces
gradients that push them off the constraint surface. Riemannian gradient
projection keeps updates tangent to the constraint manifold.

### Named Constants (`thresholds.rs`)

| Constant | Value | Citation |
|----------|-------|----------|
| `COMPLEX_CFC_EIGENVALUE_REAL_MIN` | -1.0 | Stability bound (Gu et al. 2022) |
| `COMPLEX_CFC_EIGENVALUE_REAL_MAX` | -0.01 | Must be negative for stability |
| `COMPLEX_CFC_MOTOR_FREQ_MIN_HZ` | 1.0 | Lowest motor frequency (Brown 1911) |
| `COMPLEX_CFC_MOTOR_FREQ_MAX_HZ` | 50.0 | Highest motor frequency (gamma band) |
| `COMPLEX_CFC_EIGENVALUE_LR` | 0.001 | Eigenvalue learning rate (conservative) |

### Files

- **Create**: `symthaea-core/src/hdc/complex_cfc_neuron.rs` (~500-700 LOC)
- **Modify**: `symthaea-core/src/hdc/mod.rs` -- register module
- **Modify**: `symthaea-core/Cargo.toml` -- `complex_cfc = []` feature
- **Modify**: `symthaea/Cargo.toml` -- `complex_cfc = ["symthaea-core/complex_cfc"]`
- **Modify**: `src/cognitive_loop/config.rs` -- `TemporalBackend::ComplexCfC`
- **Modify**: `src/cognitive_loop/thresholds.rs` -- named constants

### Tests (~15)

- Complex evolution produces oscillation (state norm non-monotonic over multiple steps)
- Negative real eigenvalue ensures bounded state (no NaN/Inf)
- `to_continuous_hv` / `from_continuous_hv` round-trip is lossless
- Complex binding compatible with real binding on interleaved form
- Eigenvalue learning converges to match target frequency
- Integration: complex CfC produces valid predictions through full pipeline
- Proptest: eigenvalue constraints maintained under learning, state bounded

---

## Phase 4: Spectral State Representation

**Effort**: Very high (research-grade)
**Impact**: Research -- not needed for motor rhythm
**Feature flag**: `spectral_state`
**Depends on**: Phase 3 validation

### Concept

Maintain a parallel frequency-domain twin of the CfC state via FFT. The CfC
dynamics operate in time-domain for transient responses; the spectral twin
maintains stable frequency content. Neural oscillations are most naturally
analyzed in the frequency domain (Buzsaki 2006).

### Components

**`SpectralState`** struct:
- `frequency_bins: Vec<Complex>` -- FFT of CfC state (8,192 complex bins)
- `power_spectrum: Vec<f32>` -- |F[k]|^2 per bin
- `dominant_freqs: Vec<(f32, f32)>` -- top-K (frequency_hz, power) peaks

**`SpectralManager`** (`CognitiveSubsystem` at interval 67):
- Each tick: FFT current CfC state, compute band powers, detect dominant freqs
- Output: band power ratios, cross-frequency coupling metrics
- Alpha/beta ratio predicts motor readiness
- Theta/gamma coupling as consciousness correlate (Canolty & Knight 2010)

### Dependencies

- `rustfft = { version = "6.2", optional = true }` gated behind `spectral_state`

### Integration Points

- Band powers feed into consciousness equation
- Beta power gates motor output timing
- Delta power detects slow-wave sleep analogues for dream engine
- Cross-frequency coupling for IIT phi modulation

### Intentionally Light

This phase is research-stage. Detailed design should follow Phase 3 validation.

---

## Implementation Sequencing

```
Phase 1 (Fourier fix) -----> Phase 3 (Complex CfC) -----> Phase 4 (Spectral)
     1-2 sessions               3-4 sessions                  future
     bug fix + presets           structural fix                research
     |
     |  (parallelizable)
     v
Phase 2 (CPG module)
     2-3 sessions
     validates oscillation concept
     independent manager, no neuron changes
```

Phase 1 and Phase 2 are independent -- different files, different concepts.
Phase 3 depends on Phase 1 (understanding Fourier behavior in the fused path).
Phase 4 depends on Phase 3 (spectral analysis of complex CfC state).

Recommended order: Phase 1 first (bug fix), Phase 2 next (validates motor
rhythm end-to-end with zero risk to neuron code), Phase 3 after (builds on
proven motor pipeline).

---

## Key Design Decisions

### Why NOT a `Hypervector` trait

A trait abstracting over `RealHV` and `ComplexHV` would leak complexity upward
-- every consumer becomes generic over `T: Hypervector`, which is exactly the
blast radius concern it claims to avoid. `ContinuousHV` already IS the
abstraction boundary. The complex neuron is a leaf node in the dependency graph;
swapping a leaf does not require generics.

### Why NOT Riemannian state evolution

The CfC state lives in B^d(5.0) (soft-bounded Euclidean ball), not on S^{d-1}
(the unit hypersphere). Stability comes from eigenvalue constraints (real < 0)
and norm clipping, both Euclidean operations. Riemannian tools from
`geometric_ops.rs` belong in the concept-space layer and potentially in
eigenvalue parameter optimization, not in state dynamics.

### Why eigenvalue constraint, not gradient clipping

Gradient clipping (the "band-aid") fights the symptom: signals fading or
exploding. Eigenvalue constraint fixes the cause: the dynamics themselves are
stable by construction when `real(lambda) < 0`. Clipping remains as a defense-
in-depth safety net (`apply_state_bounds`), not as the primary stability
mechanism.

### Why CPG before Complex CfC

The CPG validates the oscillation concept end-to-end (motor commands -> rhythmic
output) with zero risk to existing neuron code. It also establishes the motor
output pipeline that Phase 3's complex CfC will feed into. Build the consumer
before the producer.

---

## References

- Brown, T.G. (1911). "The intrinsic factors in the act of progression in the mammal." Proc. Royal Soc. B.
- Buzsaki, G. (2006). "Rhythms of the Brain." Oxford UP.
- Canolty, R.T. & Knight, R.T. (2010). "The functional role of cross-frequency coupling." Trends in Cognitive Sciences.
- Crone, N.E. et al. (1998). "Functional mapping of human sensorimotor cortex with electrocorticographic spectral analysis." Brain.
- Grillner, S. (2006). "Biological pattern generation: the cellular and computational logic of networks in motion." Neuron.
- Gu, A. et al. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR.
- Hirose, A. (2012). "Complex-Valued Neural Networks." Springer.
- Kuramoto, Y. (1975). "Self-entrainment of a population of coupled non-linear oscillators." Lecture Notes in Physics.
- Pfurtscheller, G. & Lopes da Silva, F.H. (1999). "Event-related EEG/MEG synchronization and desynchronization." Clinical Neurophysiology.
