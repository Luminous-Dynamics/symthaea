// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified HDC-LTC Neuron Architecture
//!
//! A revolutionary architecture where the neuron STATE is a hypervector that evolves
//! through Liquid Time-Constant (LTC) dynamics with closed-form solutions.
//!
//! ## Core Innovation
//!
//! Traditional LTC neurons use scalar states with weight matrices:
//! ```text
//! dx/dt = (-x + f(Wx + Uu)) / τ
//! ```
//!
//! Our unified architecture makes the state itself a hypervector and replaces
//! matrix operations with HDC algebraic operations:
//! ```text
//! dx/dt = (-x ⊕ f(W⊗x ⊕ U⊗u)) / τ(||x||)
//! ```
//!
//! Where:
//! - `x` is the neuron state (ContinuousHV, 16,384D)
//! - `W`, `U` are weight hypervectors (not matrices!)
//! - `⊗` is HDC binding (element-wise multiplication)
//! - `⊕` is HDC bundling (normalized sum)
//! - `τ(||x||)` is state-dependent time constant
//!
//! ## Closed-Form Solution
//!
//! Like CfC networks, we derive a closed-form solution enabling O(1) temporal jumps:
//! ```text
//! x(t+Δt) = x_∞ + (x(t) - x_∞) * exp(-Δt/τ)
//! ```
//!
//! Where `x_∞ = f(W⊗x + U⊗u)` is the equilibrium state.
//!
//! ## Key Benefits
//!
//! 1. **O(1) Temporal Jumps**: No ODE integration needed for large time steps
//! 2. **HDC Algebraic Operations**: Binding/bundling instead of matrix multiply
//! 3. **State-Dependent Dynamics**: Time constant adapts to state complexity
//! 4. **Memory as Computation**: HV state IS the memory (holographic)
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
//!
//! let config = UnifiedConfig::default();
//! let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);
//!
//! // O(1) jump to arbitrary time
//! let input = ContinuousHV::random_default(123);
//! neuron.evolve_closed_form(1.0, &input); // Jump 1 second
//! neuron.evolve_closed_form(100.0, &input); // Jump 100 seconds (same cost!)
//! ```

use crate::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for HdcLtcUnifiedNeuron
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedConfig {
    /// Base time constant τ₀ (seconds)
    /// Controls baseline response speed
    pub tau_base: f32,

    /// State-dependent time constant scaling factor
    /// τ(x) = τ₀ × (1 + backbone × ||x||)
    pub backbone_tau: f32,

    /// Dimension of hypervectors (default: 16,384)
    pub dimension: usize,

    /// Activation function type
    pub activation: UnifiedActivation,

    /// Learning rate for online adaptation
    pub learning_rate: f32,

    /// Momentum for gradient updates
    pub momentum: f32,

    /// L2 regularization strength
    pub weight_decay: f32,

    /// Gating sigmoid steepness for closed-form solution
    pub gating_steepness: f32,

    /// Interpolation bias (controls equilibrium influence)
    pub interp_bias: f32,

    /// Fourier basis injection frequencies (Hz). Empty = disabled.
    /// When non-empty, sin/cos signals at these frequencies are bundled
    /// into the equilibrium computation as a time-varying perturbation.
    #[serde(default)]
    pub fourier_frequencies: Vec<f32>,

    /// Amplitude of the Fourier basis injection (default: 0.1).
    /// Kept small so it acts as a perturbation, not a driver.
    #[serde(default = "default_fourier_amplitude")]
    pub fourier_amplitude: f32,
}

fn default_fourier_amplitude() -> f32 {
    0.1
}

impl Default for UnifiedConfig {
    fn default() -> Self {
        Self {
            tau_base: 0.1,            // 100ms base time constant
            backbone_tau: 0.5,        // Moderate state dependency
            dimension: HDC_DIMENSION, // 16,384
            activation: UnifiedActivation::Tanh,
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0001,
            gating_steepness: 1.0,           // Standard sigmoid
            interp_bias: 0.0,                // Neutral interpolation
            fourier_frequencies: Vec::new(), // Disabled by default
            fourier_amplitude: 0.1,
        }
    }
}

/// Activation function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum UnifiedActivation {
    /// Hyperbolic tangent: tanh(x)
    Tanh,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// SiLU (Swish): x * sigmoid(x)
    SiLU,
    /// Identity (linear)
    Identity,
    /// Bounded tanh with scaling
    BoundedTanh { scale: f32 },
}

impl UnifiedActivation {
    /// Apply activation element-wise to a hypervector
    #[inline]
    pub fn apply(&self, hv: &ContinuousHV) -> ContinuousHV {
        let values: Vec<f32> = match self {
            UnifiedActivation::Tanh => hv.values.iter().map(|x| x.tanh()).collect(),
            UnifiedActivation::Sigmoid => {
                hv.values.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
            }
            UnifiedActivation::SiLU => hv
                .values
                .iter()
                .map(|x| x * (1.0 / (1.0 + (-x).exp())))
                .collect(),
            UnifiedActivation::Identity => hv.values.clone(),
            UnifiedActivation::BoundedTanh { scale } => {
                hv.values.iter().map(|x| (x * scale).tanh()).collect()
            }
        };
        ContinuousHV::from_values(values)
    }

    /// Compute derivative for backpropagation
    #[inline]
    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            UnifiedActivation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            UnifiedActivation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            UnifiedActivation::SiLU => {
                let s = 1.0 / (1.0 + (-x).exp());
                s + x * s * (1.0 - s)
            }
            UnifiedActivation::Identity => 1.0,
            UnifiedActivation::BoundedTanh { scale } => {
                let t = (x * scale).tanh();
                scale * (1.0 - t * t)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIFIED HDC-LTC NEURON
// ═══════════════════════════════════════════════════════════════════════════════

/// Unified HDC-LTC Neuron with Closed-Form Solution
///
/// The core innovation: neuron STATE is a hypervector that evolves through
/// LTC dynamics using HDC algebraic operations instead of matrix multiplication.
///
/// ## Dynamics
///
/// Standard form:
/// ```text
/// dx/dt = (-x + f(W⊗x ⊕ U⊗u)) / τ(||x||)
/// ```
///
/// Closed-form solution for arbitrary time jumps:
/// ```text
/// x(t+Δt) = x_∞ + (x(t) - x_∞) * exp(-Δt/τ)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcLtcUnifiedNeuron {
    /// Current state (hypervector)
    state: ContinuousHV,

    /// Weight hypervector for state transformation (W)
    /// Replaces weight MATRIX with single HV - uses binding
    weight_hv: ContinuousHV,

    /// Input mask hypervector (U)
    /// Transforms input via binding
    input_mask: ContinuousHV,

    /// Time constant modulator HV
    /// Enables input-dependent τ adjustment
    tau_modulator: ContinuousHV,

    /// Gating function weight HV (for closed-form)
    /// Computes interpolation factor σ
    gate_weight: ContinuousHV,

    /// Gating bias HV
    gate_bias: ContinuousHV,

    /// Configuration
    config: UnifiedConfig,

    /// Momentum for weight updates
    weight_momentum: ContinuousHV,

    /// Momentum for input mask updates
    input_momentum: ContinuousHV,

    /// Running statistics
    running_mean: f32,
    running_var: f32,

    /// Total time evolved
    total_time: f64,

    /// Number of updates
    update_count: u64,
}

/// Fast rational approximation of `tanh(x)` that auto-vectorizes on AVX2.
///
/// Uses `tanh(x) ≈ x * (27 + x²) / (27 + 9x²)` for |x| < 4.97.
/// Max error ~0.004 (0.4%) within the approximation range.
/// For |x| >= 4.97, returns ±1.0 (exact within f32 precision).
///
/// Unlike `f32::tanh()` (libm `tanhf`), this is pure arithmetic and LLVM
/// can vectorize it — processing 8 elements per cycle on AVX2.
#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    if x.abs() > 4.97 {
        x.signum()
    } else {
        let x2 = x * x;
        x * (27.0 + x2) / (27.0 + 9.0 * x2)
    }
}

// =============================================================================
// SIMD FUSED EVOLUTION KERNELS
// Hand-written AVX2+FMA intrinsics for the fused equilibrium inner loops.
// Processes 8 f32 elements per cycle (256-bit registers).
// =============================================================================

/// Fused tanh evolution kernel using AVX2+FMA intrinsics.
///
/// Computes `state[i] = (1-σ)*state[i] + σ*fast_tanh((W[i]*state[i] + M[i]*input[i])*scale)`
/// for all elements in a single pass with zero intermediate allocations.
///
/// `pre_scale` is 0.5 for Tanh, 0.5*scale for BoundedTanh.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fused_tanh_avx2(
    state: &mut [f32],
    weight: &[f32],
    input_mask: &[f32],
    input: &[f32],
    sigma: f32,
    one_minus_sigma: f32,
    pre_scale: f32,
) {
    unsafe {
        use std::arch::x86_64::*;

        let dim = state.len();
        let chunks = dim / 8;

        let sigma_v = _mm256_set1_ps(sigma);
        let oms_v = _mm256_set1_ps(one_minus_sigma);
        let scale_v = _mm256_set1_ps(pre_scale);
        let c27 = _mm256_set1_ps(27.0);
        let c9 = _mm256_set1_ps(9.0);
        let thresh = _mm256_set1_ps(4.97);
        let neg_thresh = _mm256_set1_ps(-4.97);
        let one = _mm256_set1_ps(1.0);
        let neg_one = _mm256_set1_ps(-1.0);

        let s_ptr = state.as_mut_ptr();
        let w_ptr = weight.as_ptr();
        let m_ptr = input_mask.as_ptr();
        let i_ptr = input.as_ptr();

        for c in 0..chunks {
            let off = c * 8;

            // Load 8 elements from each array
            let s = _mm256_loadu_ps(s_ptr.add(off));
            let w = _mm256_loadu_ps(w_ptr.add(off));
            let m = _mm256_loadu_ps(m_ptr.add(off));
            let inp = _mm256_loadu_ps(i_ptr.add(off));

            // pre_act = (w*s + m*inp) * pre_scale
            // FMA: w*s is mul, then fmadd for m*inp + w*s, then mul by scale
            let ws = _mm256_mul_ps(w, s);
            let pre_act = _mm256_mul_ps(_mm256_fmadd_ps(m, inp, ws), scale_v);

            // fast_tanh SIMD: x*(27+x²) / (27+9x²)
            let x2 = _mm256_mul_ps(pre_act, pre_act);
            let num = _mm256_mul_ps(pre_act, _mm256_add_ps(c27, x2));
            let denom = _mm256_fmadd_ps(c9, x2, c27);
            let tanh_v = _mm256_div_ps(num, denom);

            // Clip: |x| > 4.97 → signum(x)
            let clip_hi = _mm256_cmp_ps(pre_act, thresh, _CMP_GT_OQ);
            let clip_lo = _mm256_cmp_ps(pre_act, neg_thresh, _CMP_LT_OQ);
            let x_inf = _mm256_blendv_ps(tanh_v, one, clip_hi);
            let x_inf = _mm256_blendv_ps(x_inf, neg_one, clip_lo);

            // Lerp: new_state = oms*s + sigma*x_inf
            let new_s = _mm256_fmadd_ps(sigma_v, x_inf, _mm256_mul_ps(oms_v, s));

            _mm256_storeu_ps(s_ptr.add(off), new_s);
        }

        // Scalar remainder (dim % 8 elements)
        for i in (chunks * 8)..dim {
            let si = *state.get_unchecked(i);
            let x = (*weight.get_unchecked(i) * si
                + *input_mask.get_unchecked(i) * *input.get_unchecked(i))
                * pre_scale;
            let x_inf = fast_tanh(x);
            *state.get_unchecked_mut(i) = one_minus_sigma * si + sigma * x_inf;
        }
    }
}

/// Fused identity (linear) evolution kernel using AVX2+FMA intrinsics.
///
/// Computes `state[i] = (1-σ)*state[i] + σ*(W[i]*state[i] + M[i]*input[i])*0.5`
/// — no activation function, just linear interpolation.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fused_identity_avx2(
    state: &mut [f32],
    weight: &[f32],
    input_mask: &[f32],
    input: &[f32],
    sigma: f32,
    one_minus_sigma: f32,
) {
    unsafe {
        use std::arch::x86_64::*;

        let dim = state.len();
        let chunks = dim / 8;

        let sigma_v = _mm256_set1_ps(sigma);
        let oms_v = _mm256_set1_ps(one_minus_sigma);
        let half = _mm256_set1_ps(0.5);

        let s_ptr = state.as_mut_ptr();
        let w_ptr = weight.as_ptr();
        let m_ptr = input_mask.as_ptr();
        let i_ptr = input.as_ptr();

        for c in 0..chunks {
            let off = c * 8;

            let s = _mm256_loadu_ps(s_ptr.add(off));
            let w = _mm256_loadu_ps(w_ptr.add(off));
            let m = _mm256_loadu_ps(m_ptr.add(off));
            let inp = _mm256_loadu_ps(i_ptr.add(off));

            // x_inf = (w*s + m*inp) * 0.5
            let x_inf = _mm256_mul_ps(_mm256_fmadd_ps(m, inp, _mm256_mul_ps(w, s)), half);

            // Lerp: new_state = oms*s + sigma*x_inf
            let new_s = _mm256_fmadd_ps(sigma_v, x_inf, _mm256_mul_ps(oms_v, s));

            _mm256_storeu_ps(s_ptr.add(off), new_s);
        }

        // Scalar remainder
        for i in (chunks * 8)..dim {
            let si = *state.get_unchecked(i);
            let x_inf = (*weight.get_unchecked(i) * si
                + *input_mask.get_unchecked(i) * *input.get_unchecked(i))
                * 0.5;
            *state.get_unchecked_mut(i) = one_minus_sigma * si + sigma * x_inf;
        }
    }
}

/// Fused tanh evolution kernel using NEON intrinsics (AArch64).
/// Same algorithm as `fused_tanh_avx2` but with 4-wide f32 lanes.
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
unsafe fn fused_tanh_neon(
    state: &mut [f32],
    weight: &[f32],
    input_mask: &[f32],
    input: &[f32],
    sigma: f32,
    one_minus_sigma: f32,
    pre_scale: f32,
) {
    use std::arch::aarch64::*;

    let dim = state.len();
    let chunks = dim / 4;

    let sigma_v = vdupq_n_f32(sigma);
    let oms_v = vdupq_n_f32(one_minus_sigma);
    let scale_v = vdupq_n_f32(pre_scale);
    let c27 = vdupq_n_f32(27.0);
    let c9 = vdupq_n_f32(9.0);
    let thresh = vdupq_n_f32(4.97);
    let neg_thresh = vdupq_n_f32(-4.97);
    let one = vdupq_n_f32(1.0);
    let neg_one = vdupq_n_f32(-1.0);

    let s_ptr = state.as_mut_ptr();
    let w_ptr = weight.as_ptr();
    let m_ptr = input_mask.as_ptr();
    let i_ptr = input.as_ptr();

    for c in 0..chunks {
        let off = c * 4;

        let s = vld1q_f32(s_ptr.add(off));
        let w = vld1q_f32(w_ptr.add(off));
        let m = vld1q_f32(m_ptr.add(off));
        let inp = vld1q_f32(i_ptr.add(off));

        // pre_act = (w*s + m*inp) * pre_scale
        let ws = vmulq_f32(w, s);
        let pre_act = vmulq_f32(vfmaq_f32(ws, m, inp), scale_v);

        // fast_tanh: x*(27+x^2) / (27+9*x^2)
        let x2 = vmulq_f32(pre_act, pre_act);
        let num = vmulq_f32(pre_act, vaddq_f32(c27, x2));
        let denom = vfmaq_f32(c27, c9, x2);
        let tanh_v = vdivq_f32(num, denom);

        // Clip: |x| > 4.97 -> signum(x)
        let clip_hi = vcgtq_f32(pre_act, thresh);
        let clip_lo = vcltq_f32(pre_act, neg_thresh);
        let x_inf = vbslq_f32(clip_hi, one, tanh_v);
        let x_inf = vbslq_f32(clip_lo, neg_one, x_inf);

        // Lerp: new_state = oms*s + sigma*x_inf
        let new_s = vfmaq_f32(vmulq_f32(oms_v, s), sigma_v, x_inf);

        vst1q_f32(s_ptr.add(off), new_s);
    }

    // Scalar remainder
    for i in (chunks * 4)..dim {
        let si = *state.get_unchecked(i);
        let x = (*weight.get_unchecked(i) * si
            + *input_mask.get_unchecked(i) * *input.get_unchecked(i))
            * pre_scale;
        let x_inf = fast_tanh(x);
        *state.get_unchecked_mut(i) = one_minus_sigma * si + sigma * x_inf;
    }
}

/// Fused identity (linear) evolution kernel using NEON intrinsics (AArch64).
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
unsafe fn fused_identity_neon(
    state: &mut [f32],
    weight: &[f32],
    input_mask: &[f32],
    input: &[f32],
    sigma: f32,
    one_minus_sigma: f32,
) {
    use std::arch::aarch64::*;

    let dim = state.len();
    let chunks = dim / 4;

    let sigma_v = vdupq_n_f32(sigma);
    let oms_v = vdupq_n_f32(one_minus_sigma);
    let half = vdupq_n_f32(0.5);

    let s_ptr = state.as_mut_ptr();
    let w_ptr = weight.as_ptr();
    let m_ptr = input_mask.as_ptr();
    let i_ptr = input.as_ptr();

    for c in 0..chunks {
        let off = c * 4;

        let s = vld1q_f32(s_ptr.add(off));
        let w = vld1q_f32(w_ptr.add(off));
        let m = vld1q_f32(m_ptr.add(off));
        let inp = vld1q_f32(i_ptr.add(off));

        let x_inf = vmulq_f32(vfmaq_f32(vmulq_f32(w, s), m, inp), half);
        let new_s = vfmaq_f32(vmulq_f32(oms_v, s), sigma_v, x_inf);

        vst1q_f32(s_ptr.add(off), new_s);
    }

    for i in (chunks * 4)..dim {
        let si = *state.get_unchecked(i);
        let x_inf = (*weight.get_unchecked(i) * si
            + *input_mask.get_unchecked(i) * *input.get_unchecked(i))
            * 0.5;
        *state.get_unchecked_mut(i) = one_minus_sigma * si + sigma * x_inf;
    }
}

impl HdcLtcUnifiedNeuron {
    /// Create a new unified neuron with given configuration and seed.
    ///
    /// Uses Gram-Schmidt orthogonalization for the 5 internal HVs to ensure
    /// minimal interference at initialization (pairwise similarity < 0.01).
    pub fn new(config: UnifiedConfig, seed: u64) -> Self {
        let dim = config.dimension;

        // Generate 5 orthogonal unit vectors via modified Gram-Schmidt
        let ortho = ContinuousHV::orthogonal_set(dim, 5, seed);

        Self {
            state: ContinuousHV::zero(dim),
            weight_hv: ortho[0].clone(),
            input_mask: ortho[1].clone(),
            tau_modulator: ortho[2].clone(),
            gate_weight: ortho[3].clone(),
            gate_bias: ortho[4].scale(0.1), // Scale down for bias
            weight_momentum: ContinuousHV::zero(dim),
            input_momentum: ContinuousHV::zero(dim),
            running_mean: 0.0,
            running_var: 1.0,
            total_time: 0.0,
            update_count: 0,
            config,
        }
    }

    /// Create a neuron with all internal HVs deterministically derived from a genesis seed.
    ///
    /// Domain labels are derived as `"{label}::weight_hv"`, `"{label}::input_mask"`, etc.
    pub fn from_genesis(
        config: UnifiedConfig,
        genesis: &crate::genesis::GenesisSeed,
        label: &str,
    ) -> Self {
        let dim = config.dimension;
        Self {
            state: ContinuousHV::zero(dim),
            weight_hv: genesis.hv(&format!("{label}::weight_hv"), dim),
            input_mask: genesis.hv(&format!("{label}::input_mask"), dim),
            tau_modulator: genesis.hv(&format!("{label}::tau_modulator"), dim),
            gate_weight: genesis.hv(&format!("{label}::gate_weight"), dim),
            gate_bias: genesis.hv(&format!("{label}::gate_bias"), dim).scale(0.1),
            weight_momentum: ContinuousHV::zero(dim),
            input_momentum: ContinuousHV::zero(dim),
            running_mean: 0.0,
            running_var: 1.0,
            total_time: 0.0,
            update_count: 0,
            config,
        }
    }

    /// Create with default configuration
    pub fn new_default(seed: u64) -> Self {
        Self::new(UnifiedConfig::default(), seed)
    }

    /// Compute a Fourier basis HV from configured frequencies and current total_time.
    ///
    /// For each frequency f, sin(2πf·t) and cos(2πf·t) are distributed across
    /// the dimension via a strided pattern. Scaled by `fourier_amplitude`.
    /// Returns None when `fourier_frequencies` is empty.
    fn compute_fourier_basis(&self) -> Option<ContinuousHV> {
        let freqs = &self.config.fourier_frequencies;
        if freqs.is_empty() {
            return None;
        }

        let dim = self.config.dimension;
        let amp = self.config.fourier_amplitude;
        let t = self.total_time;
        let mut values = vec![0.0f32; dim];
        let two_pi = 2.0 * std::f32::consts::PI;

        // Distribute sin/cos pairs across dimension with striding
        // Each frequency gets 2 channels (sin + cos), striped across dim
        let total_channels = freqs.len() * 2;

        for (fi, &freq) in freqs.iter().enumerate() {
            let sin_val = (two_pi * freq * t as f32).sin() * amp;
            let cos_val = (two_pi * freq * t as f32).cos() * amp;

            // Stride: sin channel at offset 2*fi, cos at 2*fi+1
            let sin_channel = 2 * fi;
            let cos_channel = 2 * fi + 1;

            for idx in (sin_channel..dim).step_by(total_channels.max(1)) {
                values[idx] = sin_val;
            }
            if cos_channel < dim {
                for idx in (cos_channel..dim).step_by(total_channels.max(1)) {
                    values[idx] = cos_val;
                }
            }
        }

        Some(ContinuousHV::from_vec(values))
    }

    /// Compute the equilibrium state x_∞ for given input
    ///
    /// x_∞ = f(W⊗x + U⊗u) where f is the activation function.
    /// When Fourier basis is configured, it is bundled in as a third component.
    #[inline]
    fn compute_equilibrium(&self, input: &ContinuousHV) -> ContinuousHV {
        // HDC binding: W⊗x (state transformation via binding, not matrix mul)
        let transformed_state = self.weight_hv.bind(&self.state);

        // HDC binding: U⊗u (input transformation)
        let masked_input = self.input_mask.bind(input);

        // HDC bundling: combine state and input contributions
        // If Fourier basis exists, bundle as third component
        let combined = if let Some(ref fourier) = self.compute_fourier_basis() {
            ContinuousHV::bundle(&[&transformed_state, &masked_input, fourier])
        } else {
            ContinuousHV::bundle(&[&transformed_state, &masked_input])
        };

        // Apply activation function
        self.config.activation.apply(&combined)
    }

    /// Compute effective time constant τ(||x||, u)
    ///
    /// Time constant adapts to both state complexity and input
    #[inline]
    fn compute_tau(&self, input: &ContinuousHV) -> f32 {
        let state_norm = self.state.norm();

        // Input-dependent adjustment via similarity with tau_modulator
        let input_adjustment = input.similarity(&self.tau_modulator);

        // τ = τ₀ × (1 + backbone × ||x||) × (1 + 0.2 × input_adjustment)
        let tau = self.config.tau_base
            * (1.0 + self.config.backbone_tau * state_norm)
            * (1.0 + 0.2 * input_adjustment);

        // Clamp to reasonable range
        tau.clamp(0.01, 10.0)
    }

    /// Compute gating/interpolation factor σ for closed-form solution
    ///
    /// σ determines how much to interpolate between current and equilibrium state
    #[inline]
    fn compute_gating(&self, input: &ContinuousHV, dt: f32) -> f32 {
        let tau = self.compute_tau(input);

        // Fused bundle + similarity: avoid allocating a 16384-element bundle vector.
        // bundle([state, input]) = (state + input) / 2
        // similarity(bundle, gate_weight) = dot(bundle, gw) / (||bundle|| * ||gw||)
        // We compute dot((state+input)/2, gw) and norms inline.
        let dim = self.config.dimension;
        let inv_dim = 1.0 / dim as f32;
        let mut dot = 0.0f32;
        let mut bundle_norm_sq = 0.0f32;
        let mut gw_norm_sq = 0.0f32;
        let mut bias_sum = 0.0f32;

        for i in 0..dim {
            let b = (self.state.values[i] + input.values[i]) * 0.5;
            let g = self.gate_weight.values[i];
            dot += b * g;
            bundle_norm_sq += b * b;
            gw_norm_sq += g * g;
            bias_sum += self.gate_bias.values[i];
        }

        let denom = (bundle_norm_sq * gw_norm_sq).sqrt();
        let sim = if denom < 1e-10 { 0.0 } else { dot / denom };
        let gate_activation = sim + bias_sum * inv_dim;

        // Sigmoid gating with steepness control
        let sigma_base = 1.0 / (1.0 + (-gate_activation * self.config.gating_steepness).exp());

        // Time-scaled gating: larger dt → more interpolation toward equilibrium
        let decay = (-dt / tau).exp();
        let sigma = 1.0 - decay * (1.0 - sigma_base);

        sigma.clamp(0.0, 1.0)
    }

    /// Evolve neuron using Euler integration
    ///
    /// dx/dt = (-x + x_∞) / τ
    /// x(t+dt) = x(t) + dt × dx/dt
    pub fn evolve(&mut self, dt: f32, input: &ContinuousHV) {
        let x_inf = self.compute_equilibrium(input);
        let tau = self.compute_tau(input);

        // Compute derivative: dx/dt = (x_∞ - x) / τ
        let derivative = x_inf.subtract(&self.state).scale(1.0 / tau);

        // Euler step
        let delta = derivative.scale(dt);
        self.state = self.state.add(&delta);

        // Soft bounding to prevent explosion
        self.apply_state_bounds();

        // Update statistics
        self.update_stats(dt);
    }

    /// Evolve neuron using RK4 integration (more accurate)
    pub fn evolve_rk4(&mut self, dt: f32, input: &ContinuousHV) {
        let h = dt;

        // k1
        let k1 = self.compute_derivative(input, &self.state);

        // k2
        let state_k2 = self.state.add(&k1.scale(h / 2.0));
        let k2 = self.compute_derivative(input, &state_k2);

        // k3
        let state_k3 = self.state.add(&k2.scale(h / 2.0));
        let k3 = self.compute_derivative(input, &state_k3);

        // k4
        let state_k4 = self.state.add(&k3.scale(h));
        let k4 = self.compute_derivative(input, &state_k4);

        // Combine: x += (k1 + 2k2 + 2k3 + k4) * h/6
        let sum = k1.add(&k2.scale(2.0)).add(&k3.scale(2.0)).add(&k4);
        self.state = self.state.add(&sum.scale(h / 6.0));

        self.apply_state_bounds();
        self.update_stats(dt);
    }

    /// **CLOSED-FORM EVOLUTION** - O(1) temporal jump to any time horizon
    ///
    /// This is the key innovation enabling efficient temporal reasoning.
    ///
    /// ## Mathematical Basis
    ///
    /// For the ODE: dx/dt = (x_∞ - x) / τ
    ///
    /// The exact solution is:
    /// ```text
    /// x(t+Δt) = x_∞ + (x(t) - x_∞) × exp(-Δt/τ)
    /// ```
    ///
    /// With adaptive gating (CfC-style):
    /// ```text
    /// x(t+Δt) = σ × x_∞ + (1-σ) × x(t)
    /// ```
    /// where σ = σ(Δt, x, u) is the interpolation factor
    ///
    /// ## Complexity
    ///
    /// O(D) where D is hypervector dimension - independent of Δt!
    #[inline]
    pub fn evolve_closed_form(&mut self, dt: f32, input: &ContinuousHV) {
        // Compute equilibrium state
        let x_inf = self.compute_equilibrium(input);

        // Compute adaptive gating factor
        let sigma = self.compute_gating(input, dt);

        // Closed-form interpolation: x' = (1-σ) × x + σ × x_∞ (in-place, zero allocs)
        self.state.lerp_in_place(&x_inf, 1.0 - sigma, sigma);

        self.apply_state_bounds();
        self.update_stats(dt);
    }

    /// **FUSED closed-form evolution** — zero intermediate allocations.
    ///
    /// Combines `compute_equilibrium` (2 binds + 1 bundle + 1 activation) and
    /// `lerp_in_place` into a single pass through the dimension.
    ///
    /// Eliminates 4 × D × sizeof(f32) bytes of intermediate allocations per call.
    /// For D=16384, that is **256 KB saved per invocation**.
    ///
    /// On x86_64 with AVX2+FMA, dispatches to hand-written SIMD intrinsics that
    /// process 8 f32 elements per cycle. Uses `fast_tanh` rational approximation
    /// (max error ~0.004) which maps directly to SIMD arithmetic.
    ///
    /// Mathematically equivalent to `evolve_closed_form` within ~0.4% tolerance.
    #[inline]
    pub fn evolve_closed_form_fused(&mut self, dt: f32, input: &ContinuousHV) {
        // Compute gating FIRST (reads self.state immutably, no mutation)
        let sigma = self.compute_gating(input, dt);
        let one_minus_sigma = 1.0 - sigma;
        let dim = self.config.dimension;

        // Single fused pass: bind(W⊗x) + bind(U⊗u) + bundle + activate + lerp
        // Replaces: compute_equilibrium (4 allocs) + lerp_in_place
        match self.config.activation {
            UnifiedActivation::Tanh => {
                // AVX2+FMA fast path: hand-written SIMD intrinsics
                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if crate::hdc::simd_detect::has_avx2() && crate::hdc::simd_detect::has_fma() {
                        // SAFETY: AVX2+FMA availability verified by runtime feature detection above.
                        // All slice arguments are the same length (ContinuousHV dimension).
                        unsafe {
                            fused_tanh_avx2(
                                &mut self.state.values,
                                &self.weight_hv.values,
                                &self.input_mask.values,
                                &input.values,
                                sigma,
                                one_minus_sigma,
                                0.5,
                            );
                        }
                        self.apply_state_bounds();
                        self.update_stats(dt);
                        return;
                    }
                }
                // NEON fast path (AArch64)
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    // SAFETY: NEON availability verified by compile-time target_arch check.
                    // All slice arguments are the same length (ContinuousHV dimension).
                    unsafe {
                        fused_tanh_neon(
                            &mut self.state.values,
                            &self.weight_hv.values,
                            &self.input_mask.values,
                            &input.values,
                            sigma,
                            one_minus_sigma,
                            0.5,
                        );
                    }
                    self.apply_state_bounds();
                    self.update_stats(dt);
                    return;
                }
                // Scalar fallback (auto-vectorized by LLVM where possible)
                for i in 0..dim {
                    let state_i = self.state.values[i];
                    let x_inf = fast_tanh(
                        (self.weight_hv.values[i] * state_i
                            + self.input_mask.values[i] * input.values[i])
                            * 0.5,
                    );
                    self.state.values[i] = one_minus_sigma * state_i + sigma * x_inf;
                }
            }
            UnifiedActivation::Sigmoid => {
                for i in 0..dim {
                    let state_i = self.state.values[i];
                    let combined = (self.weight_hv.values[i] * state_i
                        + self.input_mask.values[i] * input.values[i])
                        * 0.5;
                    let x_inf = 1.0 / (1.0 + (-combined).exp());
                    self.state.values[i] = one_minus_sigma * state_i + sigma * x_inf;
                }
            }
            UnifiedActivation::SiLU => {
                for i in 0..dim {
                    let state_i = self.state.values[i];
                    let combined = (self.weight_hv.values[i] * state_i
                        + self.input_mask.values[i] * input.values[i])
                        * 0.5;
                    let x_inf = combined / (1.0 + (-combined).exp());
                    self.state.values[i] = one_minus_sigma * state_i + sigma * x_inf;
                }
            }
            UnifiedActivation::Identity => {
                // AVX2+FMA fast path
                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if crate::hdc::simd_detect::has_avx2() && crate::hdc::simd_detect::has_fma() {
                        // SAFETY: AVX2+FMA availability verified by runtime feature detection.
                        // All slice arguments are the same length (ContinuousHV dimension).
                        unsafe {
                            fused_identity_avx2(
                                &mut self.state.values,
                                &self.weight_hv.values,
                                &self.input_mask.values,
                                &input.values,
                                sigma,
                                one_minus_sigma,
                            );
                        }
                        self.apply_state_bounds();
                        self.update_stats(dt);
                        return;
                    }
                }
                // NEON fast path (AArch64)
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    // SAFETY: NEON availability verified by compile-time target_arch check.
                    // All slice arguments are the same length (ContinuousHV dimension).
                    unsafe {
                        fused_identity_neon(
                            &mut self.state.values,
                            &self.weight_hv.values,
                            &self.input_mask.values,
                            &input.values,
                            sigma,
                            one_minus_sigma,
                        );
                    }
                    self.apply_state_bounds();
                    self.update_stats(dt);
                    return;
                }
                // Scalar fallback
                for i in 0..dim {
                    let state_i = self.state.values[i];
                    let x_inf = (self.weight_hv.values[i] * state_i
                        + self.input_mask.values[i] * input.values[i])
                        * 0.5;
                    self.state.values[i] = one_minus_sigma * state_i + sigma * x_inf;
                }
            }
            UnifiedActivation::BoundedTanh { scale } => {
                // AVX2+FMA fast path (reuses tanh kernel with custom pre_scale)
                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if crate::hdc::simd_detect::has_avx2() && crate::hdc::simd_detect::has_fma() {
                        // SAFETY: AVX2+FMA availability verified by runtime feature detection above.
                        // All slice arguments are the same length (ContinuousHV dimension).
                        unsafe {
                            fused_tanh_avx2(
                                &mut self.state.values,
                                &self.weight_hv.values,
                                &self.input_mask.values,
                                &input.values,
                                sigma,
                                one_minus_sigma,
                                0.5 * scale,
                            );
                        }
                        self.apply_state_bounds();
                        self.update_stats(dt);
                        return;
                    }
                }
                // NEON fast path (AArch64)
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    // SAFETY: NEON availability verified by compile-time target_arch check.
                    // All slice arguments are the same length (ContinuousHV dimension).
                    unsafe {
                        fused_tanh_neon(
                            &mut self.state.values,
                            &self.weight_hv.values,
                            &self.input_mask.values,
                            &input.values,
                            sigma,
                            one_minus_sigma,
                            0.5 * scale,
                        );
                    }
                    self.apply_state_bounds();
                    self.update_stats(dt);
                    return;
                }
                // Scalar fallback
                for i in 0..dim {
                    let state_i = self.state.values[i];
                    let x_inf = fast_tanh(
                        (self.weight_hv.values[i] * state_i
                            + self.input_mask.values[i] * input.values[i])
                            * 0.5
                            * scale,
                    );
                    self.state.values[i] = one_minus_sigma * state_i + sigma * x_inf;
                }
            }
        }

        self.apply_state_bounds();
        self.update_stats(dt);
    }

    /// **PURE ANALYTICAL CLOSED-FORM EVOLUTION** - Single-step exponential decay
    ///
    /// This method implements the exponential decay solution WITHOUT learned gating.
    ///
    /// ## Mathematical Basis
    ///
    /// For the ODE: dx/dt = (x_∞ - x) / τ with CONSTANT x_∞
    ///
    /// The exact analytical solution is:
    /// ```text
    /// x(t+Δt) = x_∞ + (x(t) - x_∞) × exp(-Δt/τ)
    /// ```
    ///
    /// ## Important Note
    ///
    /// In the HDC-LTC architecture, x_∞ = f(W⊗x + U⊗u) depends on the current state x,
    /// making this a NONLINEAR ODE. The closed-form solution assumes x_∞ is constant
    /// during the time step, which introduces error for large dt.
    ///
    /// For accurate results with nonlinear dynamics, use `evolve_closed_form_iterative`
    /// which sub-steps to handle the changing equilibrium.
    pub fn evolve_closed_form_exact(&mut self, dt: f32, input: &ContinuousHV) {
        // Compute equilibrium state (NOTE: this depends on current state!)
        let x_inf = self.compute_equilibrium(input);

        // Compute effective tau
        let tau = self.compute_tau(input);

        // Pure exponential decay: σ = 1 - exp(-dt/τ)
        // Clamp exponent to prevent f32 underflow (ln(f32::MIN_POSITIVE) ≈ -87.3)
        let decay = (-dt / tau).max(-87.0).exp();
        let sigma = 1.0 - decay;

        // Analytical solution (exact only if x_∞ were constant)
        let weighted_equilibrium = x_inf.scale(sigma);
        let weighted_current = self.state.scale(1.0 - sigma);
        self.state = weighted_equilibrium.add(&weighted_current);

        self.apply_state_bounds();
        self.update_stats(dt);
    }

    /// **ITERATIVE CLOSED-FORM EVOLUTION** - Handles nonlinear equilibrium
    ///
    /// For accurate closed-form evolution of the nonlinear ODE where x_∞ depends on x,
    /// this method uses sub-stepping to recompute equilibrium periodically.
    ///
    /// ## Strategy
    ///
    /// 1. Split large dt into smaller sub-steps (at most tau/10 per step)
    /// 2. Apply closed-form exponential decay for each sub-step
    /// 3. Recompute equilibrium after each sub-step
    ///
    /// This achieves O(dt/tau) complexity instead of O(1) but provides accuracy
    /// comparable to RK4 integration.
    ///
    /// ## Parameters
    ///
    /// - `dt`: Total time to evolve
    /// - `input`: Input hypervector (assumed constant during evolution)
    pub fn evolve_closed_form_iterative(&mut self, dt: f32, input: &ContinuousHV) {
        // Determine sub-step size based on tau
        // Using tau/10 gives high accuracy for the nonlinear dynamics (<1% error)
        let base_tau = self.config.tau_base;
        let max_substep = base_tau / 10.0;

        // Calculate number of sub-steps needed
        let n_steps = ((dt / max_substep).ceil() as usize).max(1);
        let sub_dt = dt / n_steps as f32;

        // Apply closed-form evolution iteratively
        for _ in 0..n_steps {
            // Recompute equilibrium at current state
            let x_inf = self.compute_equilibrium(input);
            let tau = self.compute_tau(input);

            // Apply exponential decay (clamped to prevent f32 underflow)
            let decay = (-sub_dt / tau).max(-87.0).exp();
            let weighted_equilibrium = x_inf.scale(1.0 - decay);
            let weighted_current = self.state.scale(decay);
            self.state = weighted_equilibrium.add(&weighted_current);
        }

        self.apply_state_bounds();
        self.update_stats(dt);
    }

    /// Compute derivative dx/dt for given state (used by RK4)
    fn compute_derivative(&self, input: &ContinuousHV, state: &ContinuousHV) -> ContinuousHV {
        // Temporarily use the provided state for equilibrium computation
        let transformed_state = self.weight_hv.bind(state);
        let masked_input = self.input_mask.bind(input);
        let combined = ContinuousHV::bundle(&[&transformed_state, &masked_input]);
        let x_inf = self.config.activation.apply(&combined);

        // Compute tau with temporary state
        let state_norm = state.norm();
        let input_adjustment = input.similarity(&self.tau_modulator);
        let tau = self.config.tau_base
            * (1.0 + self.config.backbone_tau * state_norm)
            * (1.0 + 0.2 * input_adjustment);
        let tau = tau.clamp(0.01, 10.0);

        // dx/dt = (x_∞ - x) / τ
        x_inf.subtract(state).scale(1.0 / tau)
    }

    /// Apply soft state bounds to prevent numerical explosion
    #[inline]
    fn apply_state_bounds(&mut self) {
        let norm = self.state.norm();
        if norm > 5.0 {
            let scale = 5.0 / norm;
            self.state.scale_in_place(scale);
        }
    }

    /// Update running statistics
    #[inline]
    fn update_stats(&mut self, dt: f32) {
        self.total_time += dt as f64;
        self.update_count += 1;

        let alpha = 0.01;
        let new_norm = self.state.norm();
        let old_mean = self.running_mean;
        self.running_mean = (1.0 - alpha) * self.running_mean + alpha * new_norm;
        let diff_sq = (new_norm - old_mean).powi(2).min(1e6);
        self.running_var = (1.0 - alpha) * self.running_var + alpha * diff_sq;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get current state
    pub fn state(&self) -> &ContinuousHV {
        &self.state
    }

    /// Get mutable state reference
    pub fn state_mut(&mut self) -> &mut ContinuousHV {
        &mut self.state
    }

    /// Set state directly
    pub fn set_state(&mut self, state: ContinuousHV) {
        self.state = state;
    }

    /// Update the Fourier basis frequencies used in equilibrium modulation.
    pub fn set_fourier_frequencies(&mut self, freqs: &[f32]) {
        self.config.fourier_frequencies = freqs.to_vec();
    }

    /// Reset state to zero
    pub fn reset(&mut self) {
        self.state = ContinuousHV::zero(self.config.dimension);
        self.total_time = 0.0;
        self.update_count = 0;
    }

    /// Get effective tau at current state
    pub fn effective_tau(&self, input: &ContinuousHV) -> f32 {
        self.compute_tau(input)
    }

    /// Get configuration
    pub fn config(&self) -> &UnifiedConfig {
        &self.config
    }

    /// Set the base time constant τ₀ directly.
    ///
    /// Clamped to [0.01, 1.0] to prevent degenerate dynamics.
    pub fn set_tau_base(&mut self, tau: f32) {
        self.config.tau_base = tau.clamp(0.01, 1.0);
    }

    /// Get total time evolved
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Get update count
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // LEARNING
    // ═══════════════════════════════════════════════════════════════════════════

    /// Hebbian-like weight update based on state correlation
    pub fn hebbian_update(&mut self, input: &ContinuousHV, learning_rate: Option<f32>) {
        let lr = learning_rate.unwrap_or(self.config.learning_rate);

        // Correlation between input and state (Hebbian: what fires together wires together)
        let correlation = input.bind(&self.state);

        // Update with momentum
        let m = self.config.momentum;
        self.weight_momentum = self.weight_momentum.scale(m).add(&correlation.scale(lr));

        // Apply weight decay and momentum
        let decay = self.config.weight_decay;
        self.weight_hv = self.weight_hv.scale(1.0 - decay).add(&self.weight_momentum);

        // Normalize to prevent explosion
        if self.weight_hv.norm() > 2.0 {
            self.weight_hv = self.weight_hv.normalize().scale(2.0);
        }
    }

    /// Contrastive learning update (for prediction tasks)
    pub fn contrastive_update(
        &mut self,
        positive: &ContinuousHV,
        negative: &ContinuousHV,
        lr: f32,
    ) {
        // Pull toward positive examples
        let pos_delta = positive.subtract(&self.state);
        let pos_gradient = self.weight_hv.bind(&pos_delta);

        // Push away from negative examples
        let neg_delta = self.state.subtract(negative);
        let neg_gradient = self.weight_hv.bind(&neg_delta);

        // Combined update
        let gradient = pos_gradient.add(&neg_gradient.scale(0.5));

        self.weight_hv = self.weight_hv.add(&gradient.scale(lr));

        // Normalize
        if self.weight_hv.norm() > 2.0 {
            self.weight_hv = self.weight_hv.normalize().scale(2.0);
        }
    }

    /// STDP-like update: Spike-Timing Dependent Plasticity
    ///
    /// Updates weights based on the relative timing of pre and post signals.
    /// If pre fires before post (positive dt), strengthen connection.
    /// If post fires before pre (negative dt), weaken connection.
    ///
    /// # Arguments
    /// * `pre` - Pre-synaptic signal (input)
    /// * `post` - Post-synaptic signal (target/output)
    /// * `dt` - Time difference (pre - post timing). Positive = pre before post
    /// * `lr` - Learning rate
    ///
    /// # STDP Window
    /// Uses exponential windows with tau_plus=20ms (LTP) and tau_minus=20ms (LTD)
    pub fn stdp_update(&mut self, pre: &ContinuousHV, post: &ContinuousHV, dt: f32, lr: f32) {
        // STDP time constants (in same units as dt)
        const TAU_PLUS: f32 = 0.02; // 20ms for LTP
        const TAU_MINUS: f32 = 0.02; // 20ms for LTD
        const A_PLUS: f32 = 1.0; // LTP amplitude
        const A_MINUS: f32 = 0.5; // LTD amplitude (asymmetric)

        // Compute STDP weight change
        let delta_w = if dt > 0.0 {
            // Pre before post: LTP (strengthen)
            A_PLUS * (-dt / TAU_PLUS).exp()
        } else {
            // Post before pre: LTD (weaken)
            -A_MINUS * (dt / TAU_MINUS).exp()
        };

        // Compute correlation-based update direction
        let correlation = pre.bind(post);

        // Apply STDP-modulated update with momentum
        let m = self.config.momentum;
        self.weight_momentum = self
            .weight_momentum
            .scale(m)
            .add(&correlation.scale(lr * delta_w));

        // Apply weight decay and momentum
        let decay = self.config.weight_decay;
        self.weight_hv = self.weight_hv.scale(1.0 - decay).add(&self.weight_momentum);

        // Normalize
        if self.weight_hv.norm() > 2.0 {
            self.weight_hv = self.weight_hv.normalize().scale(2.0);
        }
    }

    /// Adaptive learning rate update based on gradient history (Adam-like)
    ///
    /// Tracks first and second moment estimates to adapt learning rate per-dimension.
    /// This provides faster convergence and better handling of sparse gradients.
    ///
    /// # Arguments
    /// * `gradient` - The gradient hypervector to apply
    /// * `base_lr` - Base learning rate
    /// * `beta1` - Exponential decay rate for first moment (default: 0.9)
    /// * `beta2` - Exponential decay rate for second moment (default: 0.999)
    pub fn adaptive_update(
        &mut self,
        gradient: &ContinuousHV,
        base_lr: f32,
        beta1: f32,
        beta2: f32,
    ) {
        const EPSILON: f32 = 1e-8;

        // Update biased first moment estimate (momentum)
        self.weight_momentum = self
            .weight_momentum
            .scale(beta1)
            .add(&gradient.scale(1.0 - beta1));

        // For simplicity, we use the momentum directly without full Adam
        // (full Adam would need separate second moment tracking)

        // Bias correction for early steps
        let t = self.update_count.max(1) as f32;
        let bc1 = 1.0 - beta1.powf(t);

        // Corrected estimate
        let corrected = self.weight_momentum.scale(1.0 / (bc1 + EPSILON));

        // Apply weight decay
        let decay = self.config.weight_decay;
        self.weight_hv = self
            .weight_hv
            .scale(1.0 - decay)
            .add(&corrected.scale(base_lr));

        // Normalize
        if self.weight_hv.norm() > 2.0 {
            self.weight_hv = self.weight_hv.normalize().scale(2.0);
        }
    }

    /// Regularized Hebbian update with L2 penalty and homeostatic plasticity
    ///
    /// Combines Hebbian learning with:
    /// 1. L2 weight regularization (prevent large weights)
    /// 2. Homeostatic scaling (maintain target activity level)
    ///
    /// # Arguments
    /// * `input` - Input pattern
    /// * `lr` - Learning rate
    /// * `target_activity` - Target state norm (homeostatic setpoint)
    pub fn regularized_hebbian_update(
        &mut self,
        input: &ContinuousHV,
        lr: f32,
        target_activity: f32,
    ) {
        // Standard Hebbian correlation
        let correlation = input.bind(&self.state);

        // Homeostatic scaling: if activity is too high, reduce learning; if too low, increase
        let current_activity = self.state.norm();
        let homeostatic_factor = if current_activity > 0.001 {
            (target_activity / current_activity).clamp(0.5, 2.0)
        } else {
            1.0
        };

        // L2 regularization term (pushes weights toward zero)
        let l2_penalty = self.weight_hv.scale(self.config.weight_decay * 2.0);

        // Combined update with momentum
        let m = self.config.momentum;
        let update = correlation
            .scale(lr * homeostatic_factor)
            .subtract(&l2_penalty);

        self.weight_momentum = self.weight_momentum.scale(m).add(&update);

        // Apply momentum
        self.weight_hv = self.weight_hv.add(&self.weight_momentum);

        // Soft weight clipping
        let norm = self.weight_hv.norm();
        if norm > 2.0 {
            self.weight_hv = self.weight_hv.normalize().scale(2.0);
        }
    }

    /// Triplet loss-style update for metric learning
    ///
    /// Learns to make anchor-positive distance smaller than anchor-negative distance.
    ///
    /// # Arguments
    /// * `anchor` - Reference pattern
    /// * `positive` - Pattern that should be similar to anchor
    /// * `negative` - Pattern that should be dissimilar to anchor
    /// * `margin` - Minimum margin between positive and negative distances
    /// * `lr` - Learning rate
    pub fn triplet_update(
        &mut self,
        anchor: &ContinuousHV,
        positive: &ContinuousHV,
        negative: &ContinuousHV,
        margin: f32,
        lr: f32,
    ) {
        // Compute distances (using 1 - similarity as distance proxy)
        let dist_pos = 1.0 - self.state.similarity(positive);
        let dist_neg = 1.0 - self.state.similarity(negative);

        // Triplet loss: max(0, dist_pos - dist_neg + margin)
        let loss = (dist_pos - dist_neg + margin).max(0.0);

        if loss > 0.0 {
            // Gradient: push toward positive, away from negative
            let pos_gradient = positive.subtract(&self.state);
            let neg_gradient = self.state.subtract(negative);

            // Scale by loss magnitude
            let combined = pos_gradient.add(&neg_gradient).scale(loss);
            let weight_gradient = self.weight_hv.bind(&combined);

            // Update with momentum
            let m = self.config.momentum;
            self.weight_momentum = self
                .weight_momentum
                .scale(m)
                .add(&weight_gradient.scale(lr));

            self.weight_hv = self
                .weight_hv
                .scale(1.0 - self.config.weight_decay)
                .add(&self.weight_momentum);

            // Normalize
            if self.weight_hv.norm() > 2.0 {
                self.weight_hv = self.weight_hv.normalize().scale(2.0);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // BACKPROPAGATION THROUGH TIME (BPTT)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Compute analytical BPTT gradients for a single closed-form step.
    ///
    /// Re-runs the forward pass internally, then backpropagates through:
    /// ```text
    /// x' = σ × x_∞ + (1-σ) × x       // interpolation
    /// x_∞ = f(W⊗x + U⊗u)             // equilibrium
    /// σ = 1 - exp(-dt/τ)              // gating from tau
    /// τ = τ₀ × (1 + β×||x||) × (1 + 0.2×sim(u, τ_mod))
    /// ```
    ///
    /// Since HDC binding is element-wise multiply, ∂(A⊗B)/∂A = B element-wise.
    pub fn backward(
        &self,
        input: &ContinuousHV,
        target: &ContinuousHV,
        dt: f32,
    ) -> HdcLtcGradients {
        let dim = self.config.dimension;

        // --- Forward recomputation ---
        // Pre-activation: z = W⊗x + U⊗u (bundled)
        let wx = self.weight_hv.bind(&self.state);
        let uu = self.input_mask.bind(input);
        let z = ContinuousHV::bundle(&[&wx, &uu]);

        // Equilibrium: x_∞ = f(z)
        let x_inf = self.config.activation.apply(&z);

        // Tau and sigma
        let tau = self.compute_tau(input);
        let decay = (-dt / tau).exp();
        let sigma = 1.0 - decay;

        // New state: x' = σ × x_∞ + (1-σ) × x
        let new_state = x_inf.scale(sigma).add(&self.state.scale(1.0 - sigma));

        // --- Backward pass ---
        // dL/dx' = 2(x' - target) / dim   (MSE gradient)
        let dh = new_state.subtract(target).scale(2.0 / dim as f32);

        // Through interpolation: dL/dx_∞ = dL/dx' × σ  (element-wise scale)
        let dx_inf = dh.scale(sigma);

        // Through activation: dL/dz = dL/dx_∞ ⊙ f'(z)
        // We need element-wise: dx_inf[i] * activation.derivative(z[i])
        let activation = self.config.activation;
        let dz_values: Vec<f32> = dx_inf
            .values
            .iter()
            .zip(z.values.iter())
            .map(|(&di, &zi)| di * activation.derivative(zi))
            .collect();
        let dz = ContinuousHV::from_values(dz_values);

        // Bundle = (wx + uu) / n, so ∂bundle/∂wx = 1/n element-wise.
        // We absorb the 1/n into the learning rate (standard practice) and pass
        // the full gradient through for stronger signal.

        // Weight gradients via binding chain rule:
        // W⊗x is element-wise multiply, so ∂(W⊗x)/∂W = x
        let dw = dz.bind(&self.state);

        // dL/dU = dz ⊙ u
        let du = dz.bind(input);

        // Tau gradient (scalar):
        // dσ/dτ = -dt/τ² × exp(-dt/τ)  (since σ = 1 - exp(-dt/τ))
        // dL/dτ = Σ_i dh_i × (x_∞_i - x_i) × dσ/dτ
        //       = Σ_i dh_i × (x_inf_i - state_i) × (-dt/τ²) × exp(-dt/τ)
        let diff = x_inf.subtract(&self.state);
        let dtau_scalar: f32 = dh
            .values
            .iter()
            .zip(diff.values.iter())
            .map(|(&dhi, &di)| dhi * di)
            .sum::<f32>()
            * (-dt / (tau * tau))
            * decay;

        // Gradient w.r.t. the actual input vector (for inter-layer BPTT):
        // U⊗u is element-wise multiply, so ∂(U⊗u)/∂u = U
        let d_input = dz.bind(&self.input_mask);

        HdcLtcGradients {
            dw,
            du,
            dtau_scalar,
            d_input,
        }
    }

    /// Apply BPTT gradients with SGD + momentum, weight decay, and norm clipping.
    pub fn apply_gradients(&mut self, grads: &HdcLtcGradients, lr: f32) {
        self.apply_gradients_inner(grads, lr, true)
    }

    /// Apply gradients without per-step weight decay.
    ///
    /// Use this for BPTT training where thousands of gradient steps occur per
    /// epoch. The standard `apply_gradients` applies weight_decay (default
    /// 0.0001) per step, which over 67K steps/epoch decays weights to ~0.1%
    /// of original (`(1-0.0001)^67646 ≈ 0.001`), destroying the network.
    pub fn apply_gradients_no_decay(&mut self, grads: &HdcLtcGradients, lr: f32) {
        self.apply_gradients_inner(grads, lr, false)
    }

    fn apply_gradients_inner(&mut self, grads: &HdcLtcGradients, lr: f32, apply_decay: bool) {
        let m = self.config.momentum;
        let decay_factor = if apply_decay {
            1.0 - self.config.weight_decay
        } else {
            1.0
        };

        // Weight HV update with momentum
        self.weight_momentum = self.weight_momentum.scale(m).add(&grads.dw.scale(-lr));
        self.weight_hv = self
            .weight_hv
            .scale(decay_factor)
            .add(&self.weight_momentum);

        // Input mask update with momentum
        self.input_momentum = self.input_momentum.scale(m).add(&grads.du.scale(-lr));
        self.input_mask = self
            .input_mask
            .scale(decay_factor)
            .add(&self.input_momentum);

        // Tau modulator update (project scalar gradient onto tau_modulator direction)
        if grads.dtau_scalar.abs() > 1e-10 {
            self.tau_modulator = self.tau_modulator.add(
                &self
                    .tau_modulator
                    .normalize()
                    .scale(-lr * grads.dtau_scalar),
            );
        }

        // Norm clip to 2.0
        if self.weight_hv.norm() > 2.0 {
            self.weight_hv = self.weight_hv.normalize().scale(2.0);
        }
        if self.input_mask.norm() > 2.0 {
            self.input_mask = self.input_mask.normalize().scale(2.0);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ADDITIONAL ACCESSORS (for testing/benchmarking)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get reference to weight hypervector (for analysis)
    pub fn weight_hv_ref(&self) -> &ContinuousHV {
        &self.weight_hv
    }

    /// Get mutable reference to weight hypervector
    pub fn weight_hv_mut(&mut self) -> &mut ContinuousHV {
        &mut self.weight_hv
    }

    /// Get reference to input mask hypervector
    pub fn input_mask_ref(&self) -> &ContinuousHV {
        &self.input_mask
    }

    /// Get reference to weight momentum
    pub fn weight_momentum_ref(&self) -> &ContinuousHV {
        &self.weight_momentum
    }

    /// Get mutable reference to weight momentum
    pub fn weight_momentum_mut(&mut self) -> &mut ContinuousHV {
        &mut self.weight_momentum
    }

    /// Get reference to input mask
    pub fn input_mask_mut(&mut self) -> &mut ContinuousHV {
        &mut self.input_mask
    }

    /// Get reference to input momentum
    pub fn input_momentum_ref(&self) -> &ContinuousHV {
        &self.input_momentum
    }

    /// Get mutable reference to input momentum
    pub fn input_momentum_mut(&mut self) -> &mut ContinuousHV {
        &mut self.input_momentum
    }

    /// Get reference to tau modulator
    pub fn tau_modulator_ref(&self) -> &ContinuousHV {
        &self.tau_modulator
    }

    /// Get mutable reference to tau modulator
    pub fn tau_modulator_mut(&mut self) -> &mut ContinuousHV {
        &mut self.tau_modulator
    }

    /// Get reference to gate weight
    pub fn gate_weight_ref(&self) -> &ContinuousHV {
        &self.gate_weight
    }

    /// Get reference to gate bias
    pub fn gate_bias_ref(&self) -> &ContinuousHV {
        &self.gate_bias
    }

    /// Reset momentum accumulators (useful for fine-tuning)
    pub fn reset_momentum(&mut self) {
        self.weight_momentum = ContinuousHV::zero(self.config.dimension);
        self.input_momentum = ContinuousHV::zero(self.config.dimension);
    }

    /// Get statistics
    pub fn stats(&self) -> UnifiedNeuronStats {
        UnifiedNeuronStats {
            state_norm: self.state.norm(),
            running_mean: self.running_mean,
            running_std: self.running_var.sqrt(),
            total_time: self.total_time,
            update_count: self.update_count,
            weight_norm: self.weight_hv.norm(),
        }
    }
}

/// Gradients computed by BPTT `backward()` for a single closed-form step.
#[derive(Debug, Clone)]
pub struct HdcLtcGradients {
    /// Gradient for `weight_hv`
    pub dw: ContinuousHV,
    /// Gradient for `input_mask`
    pub du: ContinuousHV,
    /// Scalar gradient for tau (applied to `tau_modulator`)
    pub dtau_scalar: f32,
    /// Gradient w.r.t. the input vector (for inter-layer backpropagation).
    /// Represents ∂L/∂u, allowing proper gradient flow to the previous layer.
    pub d_input: ContinuousHV,
}

/// Statistics for unified neuron
#[derive(Debug, Clone)]
pub struct UnifiedNeuronStats {
    /// Current state norm
    pub state_norm: f32,
    /// Running mean of state norm
    pub running_mean: f32,
    /// Running std of state norm
    pub running_std: f32,
    /// Total time evolved
    pub total_time: f64,
    /// Number of updates
    pub update_count: u64,
    /// Weight hypervector norm
    pub weight_norm: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIFIED NETWORK
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for unified network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedNetworkConfig {
    /// Number of neurons per layer
    pub layer_sizes: Vec<usize>,

    /// Neuron configuration
    pub neuron_config: UnifiedConfig,

    /// Use layer-wise binding
    pub use_layer_binding: bool,

    /// Use skip connections
    pub skip_connections: bool,
}

impl Default for UnifiedNetworkConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![4, 8, 4],
            neuron_config: UnifiedConfig::default(),
            use_layer_binding: true,
            skip_connections: false,
        }
    }
}

/// Network of unified HDC-LTC neurons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcLtcUnifiedNetwork {
    /// Layers of neurons
    layers: Vec<Vec<HdcLtcUnifiedNeuron>>,

    /// Inter-layer binding vectors
    layer_bindings: Vec<ContinuousHV>,

    /// Configuration
    config: UnifiedNetworkConfig,

    /// Cached layer outputs for skip connections
    layer_outputs: Vec<ContinuousHV>,
}

impl HdcLtcUnifiedNetwork {
    /// Create a network with all neurons and layer bindings deterministically
    /// derived from a genesis seed.
    ///
    /// Domain labels follow the pattern:
    /// - Neurons: `"layer_{l}::neuron_{n}"`
    /// - Layer bindings: `"layer_binding_{l}"`
    pub fn from_genesis(
        config: UnifiedNetworkConfig,
        genesis: &crate::genesis::GenesisSeed,
    ) -> Self {
        let mut layers = Vec::new();

        for (l, &layer_size) in config.layer_sizes.iter().enumerate() {
            let layer: Vec<HdcLtcUnifiedNeuron> = (0..layer_size)
                .map(|n| {
                    let label = format!("layer_{l}::neuron_{n}");
                    HdcLtcUnifiedNeuron::from_genesis(config.neuron_config.clone(), genesis, &label)
                })
                .collect();
            layers.push(layer);
        }

        let dim = config.neuron_config.dimension;
        let layer_bindings: Vec<ContinuousHV> = (0..config.layer_sizes.len())
            .map(|l| genesis.hv(&format!("layer_binding_{l}"), dim))
            .collect();

        let layer_outputs = config
            .layer_sizes
            .iter()
            .map(|_| ContinuousHV::zero(dim))
            .collect();

        Self {
            layers,
            layer_bindings,
            layer_outputs,
            config,
        }
    }

    /// Create new network
    pub fn new(config: UnifiedNetworkConfig, seed: u64) -> Self {
        let mut layers = Vec::new();
        let mut current_seed = seed;

        for &layer_size in &config.layer_sizes {
            let layer: Vec<HdcLtcUnifiedNeuron> = (0..layer_size)
                .map(|_| {
                    current_seed += 1;
                    HdcLtcUnifiedNeuron::new(config.neuron_config.clone(), current_seed)
                })
                .collect();
            layers.push(layer);
        }

        let dim = config.neuron_config.dimension;
        let layer_bindings: Vec<ContinuousHV> = (0..config.layer_sizes.len())
            .map(|i| ContinuousHV::random(dim, seed + 10000 + i as u64))
            .collect();

        let layer_outputs = config
            .layer_sizes
            .iter()
            .map(|_| ContinuousHV::zero(dim))
            .collect();

        Self {
            layers,
            layer_bindings,
            layer_outputs,
            config,
        }
    }

    /// Evolve network with standard integration
    pub fn evolve(&mut self, dt: f32, input: &ContinuousHV) {
        // Layer 0: Direct input
        for neuron in &mut self.layers[0] {
            neuron.evolve(dt, input);
        }
        self.cache_layer_output(0);

        // Subsequent layers
        for layer_idx in 1..self.layers.len() {
            let layer_input = self.compute_layer_input(layer_idx, input);

            for neuron in &mut self.layers[layer_idx] {
                neuron.evolve(dt, &layer_input);
            }
            self.cache_layer_output(layer_idx);
        }
    }

    /// Evolve network with closed-form solution (O(1) temporal jump)
    pub fn evolve_closed_form(&mut self, dt: f32, input: &ContinuousHV) {
        // Layer 0: Direct input
        for neuron in &mut self.layers[0] {
            neuron.evolve_closed_form(dt, input);
        }
        self.cache_layer_output(0);

        // Subsequent layers
        for layer_idx in 1..self.layers.len() {
            let layer_input = self.compute_layer_input(layer_idx, input);

            for neuron in &mut self.layers[layer_idx] {
                neuron.evolve_closed_form(dt, &layer_input);
            }
            self.cache_layer_output(layer_idx);
        }
    }

    /// Cache layer output
    #[inline]
    fn cache_layer_output(&mut self, layer_idx: usize) {
        let neurons = &self.layers[layer_idx];
        let output = &mut self.layer_outputs[layer_idx];
        for v in output.values.iter_mut() {
            *v = 0.0;
        }
        if neurons.is_empty() {
            return;
        }
        let inv_n = 1.0 / neurons.len() as f32;
        for neuron in neurons {
            for (o, &s) in output.values.iter_mut().zip(neuron.state().values.iter()) {
                *o += s;
            }
        }
        for v in output.values.iter_mut() {
            *v *= inv_n;
        }
    }

    /// Compute input for a layer (layer_idx must be ≥ 1; layer 0 uses original_input directly).
    fn compute_layer_input(&self, layer_idx: usize, original_input: &ContinuousHV) -> ContinuousHV {
        debug_assert!(
            layer_idx > 0,
            "compute_layer_input called with layer_idx=0; use original_input directly"
        );
        let prev_output = &self.layer_outputs[layer_idx.saturating_sub(1)];

        // Apply layer binding if configured
        let bound_input = if self.config.use_layer_binding {
            self.layer_bindings[layer_idx].bind(prev_output)
        } else {
            prev_output.clone()
        };

        // Add skip connection from input if configured
        if self.config.skip_connections && layer_idx > 0 {
            ContinuousHV::bundle(&[&bound_input, original_input])
        } else {
            bound_input
        }
    }

    /// Get the effective input for a given layer (after layer binding + skip connections).
    ///
    /// Layer 0 receives `original_input` directly.
    /// Deeper layers receive the bundled previous-layer output, optionally bound
    /// with a layer-binding vector and optionally skip-connected with the original input.
    pub fn layer_input(&self, layer_idx: usize, original_input: &ContinuousHV) -> ContinuousHV {
        if layer_idx == 0 {
            original_input.clone()
        } else {
            self.compute_layer_input(layer_idx, original_input)
        }
    }

    /// Get network output (bundled final layer states)
    pub fn output(&self) -> ContinuousHV {
        self.layer_outputs
            .last()
            .cloned()
            .unwrap_or_else(|| ContinuousHV::zero(self.config.neuron_config.dimension))
    }

    /// Update Fourier basis frequencies for all neurons in the network.
    pub fn update_fourier_frequencies(&mut self, freqs: &[f32]) {
        for layer in &mut self.layers {
            for neuron in layer {
                neuron.set_fourier_frequencies(freqs);
            }
        }
        self.config.neuron_config.fourier_frequencies = freqs.to_vec();
    }

    /// Reset momentum on all neurons (call between BPTT epochs to prevent
    /// accumulated directional bias from 67K+ steps).
    pub fn reset_momentum(&mut self) {
        for layer in &mut self.layers {
            for neuron in layer {
                neuron.reset_momentum();
            }
        }
    }

    /// Reset all neurons
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            for neuron in layer {
                neuron.reset();
            }
        }
        for output in &mut self.layer_outputs {
            *output = ContinuousHV::zero(self.config.neuron_config.dimension);
        }
    }

    /// Get number of layers
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get layer by index
    pub fn layer(&self, idx: usize) -> Option<&Vec<HdcLtcUnifiedNeuron>> {
        self.layers.get(idx)
    }

    /// Get mutable layer by index
    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut Vec<HdcLtcUnifiedNeuron>> {
        self.layers.get_mut(idx)
    }

    /// Get network configuration
    pub fn config(&self) -> &UnifiedNetworkConfig {
        &self.config
    }

    /// Get layer binding vector by index
    pub fn layer_binding(&self, idx: usize) -> &ContinuousHV {
        &self.layer_bindings[idx]
    }

    /// Get the bundled output of a specific layer.
    pub fn output_at_layer(&self, idx: usize) -> Option<&ContinuousHV> {
        self.layer_outputs.get(idx)
    }

    /// Get network statistics
    pub fn stats(&self) -> UnifiedNetworkStats {
        let all_stats: Vec<UnifiedNeuronStats> = self
            .layers
            .iter()
            .flat_map(|layer| layer.iter().map(|n| n.stats()))
            .collect();

        let n = all_stats.len();
        if n == 0 {
            return UnifiedNetworkStats {
                n_neurons: 0,
                n_layers: self.layers.len(),
                avg_state_norm: 0.0,
                avg_weight_norm: 0.0,
                total_updates: 0,
            };
        }
        let avg_norm = all_stats.iter().map(|s| s.state_norm).sum::<f32>() / n as f32;
        let avg_weight_norm = all_stats.iter().map(|s| s.weight_norm).sum::<f32>() / n as f32;

        UnifiedNetworkStats {
            n_neurons: n,
            n_layers: self.layers.len(),
            avg_state_norm: avg_norm,
            avg_weight_norm,
            total_updates: all_stats.iter().map(|s| s.update_count).sum(),
        }
    }
}

/// Network statistics
#[derive(Debug, Clone)]
pub struct UnifiedNetworkStats {
    /// Total neurons
    pub n_neurons: usize,
    /// Number of layers
    pub n_layers: usize,
    /// Average state norm
    pub avg_state_norm: f32,
    /// Average weight norm
    pub avg_weight_norm: f32,
    /// Total updates
    pub total_updates: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_neuron_creation() {
        let config = UnifiedConfig::default();
        let neuron = HdcLtcUnifiedNeuron::new(config, 42);

        assert_eq!(neuron.state().dim(), HDC_DIMENSION);
        assert_eq!(neuron.update_count(), 0);
        assert_eq!(neuron.total_time(), 0.0);
    }

    #[test]
    fn test_unified_neuron_euler_evolution() {
        let mut neuron = HdcLtcUnifiedNeuron::new_default(42);
        let input = ContinuousHV::random_default(123);

        let initial_norm = neuron.state().norm();
        assert!(initial_norm < 0.01, "State should start near zero");

        // Evolve with Euler integration
        for _ in 0..100 {
            neuron.evolve(0.01, &input);
        }

        let final_norm = neuron.state().norm();
        assert!(
            final_norm > initial_norm,
            "State should evolve away from zero"
        );
        assert!(final_norm < 10.0, "State should remain bounded");
        assert_eq!(neuron.update_count(), 100);
    }

    #[test]
    fn test_unified_neuron_closed_form_evolution() {
        let mut neuron = HdcLtcUnifiedNeuron::new_default(42);
        let input = ContinuousHV::random_default(123);

        // Single large time jump
        neuron.evolve_closed_form(1.0, &input);

        let norm = neuron.state().norm();
        assert!(norm > 0.0, "State should have evolved");
        assert!(norm < 10.0, "State should remain bounded");
    }

    #[test]
    fn test_closed_form_vs_euler_convergence() {
        let input = ContinuousHV::random_default(123);

        // Euler with many small steps
        let mut neuron_euler = HdcLtcUnifiedNeuron::new_default(42);
        for _ in 0..1000 {
            neuron_euler.evolve(0.001, &input);
        }

        // Closed-form with single large step
        let mut neuron_cf = HdcLtcUnifiedNeuron::new_default(42);
        neuron_cf.evolve_closed_form(1.0, &input);

        // Both should reach similar equilibrium (not identical due to gating differences)
        let similarity = neuron_euler.state().similarity(neuron_cf.state());
        assert!(
            similarity > 0.5,
            "Euler and closed-form should produce similar results: {}",
            similarity
        );
    }

    #[test]
    fn test_closed_form_o1_property() {
        let input = ContinuousHV::random_default(123);

        // Small time step
        let mut neuron_small = HdcLtcUnifiedNeuron::new_default(42);
        neuron_small.evolve_closed_form(0.1, &input);

        // Large time step (should be same computational cost)
        let mut neuron_large = HdcLtcUnifiedNeuron::new_default(42);
        neuron_large.evolve_closed_form(100.0, &input);

        // Both should produce valid states (the point is computational cost is the same)
        assert!(neuron_small.state().norm() > 0.0);
        assert!(neuron_large.state().norm() > 0.0);
        assert!(
            neuron_large.state().norm() < 10.0,
            "Large jump should still be bounded"
        );
    }

    #[test]
    fn test_hdc_binding_for_weights() {
        let neuron = HdcLtcUnifiedNeuron::new_default(42);
        let input = ContinuousHV::random_default(123);

        // The key innovation: weights use BINDING not matrix multiplication
        let transformed = neuron.weight_hv.bind(&input);

        // Binding should produce dissimilar output
        let sim_to_weight = transformed.similarity(&neuron.weight_hv);
        let sim_to_input = transformed.similarity(&input);

        assert!(
            sim_to_weight.abs() < 0.3,
            "Binding should be dissimilar to weight"
        );
        assert!(
            sim_to_input.abs() < 0.3,
            "Binding should be dissimilar to input"
        );
    }

    #[test]
    fn test_state_dependent_tau() {
        let mut neuron = HdcLtcUnifiedNeuron::new_default(42);
        let input = ContinuousHV::random_default(123);

        let tau_initial = neuron.effective_tau(&input);

        // Evolve to change state
        for _ in 0..50 {
            neuron.evolve(0.01, &input);
        }

        let tau_final = neuron.effective_tau(&input);

        // Tau should change as state changes
        assert!(
            (tau_final - tau_initial).abs() > 0.001,
            "Tau should be state-dependent: initial={}, final={}",
            tau_initial,
            tau_final
        );
    }

    #[test]
    fn test_unified_network_creation() {
        let config = UnifiedNetworkConfig::default();
        let network = HdcLtcUnifiedNetwork::new(config.clone(), 42);

        assert_eq!(network.n_layers(), config.layer_sizes.len());
        assert_eq!(network.layer(0).unwrap().len(), config.layer_sizes[0]);
    }

    #[test]
    fn test_unified_network_evolution() {
        let config = UnifiedNetworkConfig {
            layer_sizes: vec![2, 3, 2],
            ..Default::default()
        };
        let mut network = HdcLtcUnifiedNetwork::new(config, 42);
        let input = ContinuousHV::random_default(123);

        // Evolve with standard integration
        for _ in 0..50 {
            network.evolve(0.01, &input);
        }

        let output = network.output();
        assert_eq!(output.dim(), HDC_DIMENSION);

        let stats = network.stats();
        assert_eq!(stats.n_layers, 3);
        assert!(stats.total_updates > 0);
    }

    #[test]
    fn test_unified_network_closed_form() {
        let config = UnifiedNetworkConfig {
            layer_sizes: vec![2, 2],
            ..Default::default()
        };
        let mut network = HdcLtcUnifiedNetwork::new(config, 42);
        let input = ContinuousHV::random_default(123);

        // Single large time jump
        network.evolve_closed_form(1.0, &input);

        let output = network.output();
        assert!(output.norm() > 0.0, "Network should produce output");
    }

    #[test]
    fn test_hebbian_update() {
        let mut neuron = HdcLtcUnifiedNeuron::new_default(42);
        let input = ContinuousHV::random_default(123);

        // Evolve first
        for _ in 0..20 {
            neuron.evolve(0.01, &input);
        }

        let weight_before = neuron.weight_hv.clone();

        // Hebbian update
        neuron.hebbian_update(&input, Some(0.1));

        // Weights should change
        let similarity = weight_before.similarity(&neuron.weight_hv);
        assert!(similarity < 1.0, "Weights should have changed");
        assert!(neuron.weight_hv.norm() <= 2.1, "Weights should be bounded");
    }

    #[test]
    fn test_activation_functions() {
        let hv = ContinuousHV::random(100, 42);

        for activation in [
            UnifiedActivation::Tanh,
            UnifiedActivation::Sigmoid,
            UnifiedActivation::SiLU,
            UnifiedActivation::Identity,
            UnifiedActivation::BoundedTanh { scale: 0.5 },
        ] {
            let result = activation.apply(&hv);
            assert_eq!(result.dim(), 100);

            // Check bounds for bounded activations
            match activation {
                UnifiedActivation::Tanh | UnifiedActivation::BoundedTanh { .. } => {
                    assert!(result.values.iter().all(|&x| x >= -1.0 && x <= 1.0));
                }
                UnifiedActivation::Sigmoid => {
                    assert!(result.values.iter().all(|&x| x >= 0.0 && x <= 1.0));
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_equilibrium_computation() {
        let neuron = HdcLtcUnifiedNeuron::new_default(42);
        let input = ContinuousHV::random_default(123);

        let equilibrium = neuron.compute_equilibrium(&input);

        // Equilibrium should be bounded (due to tanh activation)
        assert!(equilibrium.values.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_gating_factor() {
        let neuron = HdcLtcUnifiedNeuron::new_default(42);
        let input = ContinuousHV::random_default(123);

        // Small dt should give smaller sigma
        let sigma_small = neuron.compute_gating(&input, 0.01);

        // Large dt should give larger sigma (more interpolation)
        let sigma_large = neuron.compute_gating(&input, 10.0);

        assert!(
            sigma_small >= 0.0 && sigma_small <= 1.0,
            "Sigma should be in [0,1]"
        );
        assert!(
            sigma_large >= 0.0 && sigma_large <= 1.0,
            "Sigma should be in [0,1]"
        );
        assert!(
            sigma_large >= sigma_small,
            "Larger dt should give larger sigma"
        );
    }

    #[test]
    fn test_contrastive_update() {
        let mut neuron = HdcLtcUnifiedNeuron::new_default(42);
        let input = ContinuousHV::random_default(123);

        // Evolve first
        for _ in 0..20 {
            neuron.evolve(0.01, &input);
        }

        let positive = ContinuousHV::random_default(456);
        let negative = ContinuousHV::random_default(789);

        let state_before = neuron.state().clone();

        neuron.contrastive_update(&positive, &negative, 0.1);

        // Weights should have changed (we didn't update state, but weights affect future dynamics)
        assert!(
            neuron.weight_hv.norm() <= 2.1,
            "Weights should remain bounded"
        );
    }

    #[test]
    fn test_neuron_from_genesis() {
        use crate::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test constitution");
        let config = UnifiedConfig::default();

        let n1 = HdcLtcUnifiedNeuron::from_genesis(config.clone(), &genesis, "neuron_0");
        let n2 = HdcLtcUnifiedNeuron::from_genesis(config, &genesis, "neuron_0");

        // Weight HVs must be bit-identical across two constructions
        assert_eq!(
            n1.weight_hv.values, n2.weight_hv.values,
            "from_genesis must produce identical weight_hv"
        );
        assert_eq!(
            n1.input_mask.values, n2.input_mask.values,
            "from_genesis must produce identical input_mask"
        );
    }

    #[test]
    fn test_network_from_genesis() {
        use crate::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("network test");
        let config = UnifiedNetworkConfig {
            layer_sizes: vec![2, 3, 2],
            ..Default::default()
        };

        let net1 = HdcLtcUnifiedNetwork::from_genesis(config.clone(), &genesis);
        let net2 = HdcLtcUnifiedNetwork::from_genesis(config, &genesis);

        // All neurons must be bit-identical
        for l in 0..net1.n_layers() {
            for (n, (a, b)) in net1
                .layer(l)
                .unwrap()
                .iter()
                .zip(net2.layer(l).unwrap().iter())
                .enumerate()
            {
                assert_eq!(
                    a.weight_hv_ref().values,
                    b.weight_hv_ref().values,
                    "layer {} neuron {} weight_hv mismatch",
                    l,
                    n
                );
            }
        }

        // Layer bindings must be identical
        assert_eq!(
            net1.layer_bindings
                .iter()
                .map(|hv| &hv.values)
                .collect::<Vec<_>>(),
            net2.layer_bindings
                .iter()
                .map(|hv| &hv.values)
                .collect::<Vec<_>>(),
            "layer_bindings mismatch"
        );

        // Different neurons in different layers should be near-orthogonal
        let n00 = net1.layer(0).unwrap()[0].weight_hv_ref();
        let n11 = net1.layer(1).unwrap()[1].weight_hv_ref();
        let sim = n00.similarity(n11);
        assert!(
            sim.abs() < 0.05,
            "Different neurons should be near-orthogonal, got {}",
            sim
        );
    }

    #[test]
    fn test_network_from_genesis_evolves() {
        use crate::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("evolve test");
        let config = UnifiedNetworkConfig {
            layer_sizes: vec![2, 2],
            ..Default::default()
        };
        let mut net = HdcLtcUnifiedNetwork::from_genesis(config, &genesis);
        let input = ContinuousHV::random_default(123);

        net.evolve_closed_form(1.0, &input);

        let output = net.output();
        assert!(
            output.norm() > 0.0,
            "Network should produce output after evolution"
        );
    }

    #[test]
    fn test_extreme_dt_no_underflow() {
        // dt=100.0, tau=0.01 → exponent = -10_000 → would underflow without clamp
        let mut neuron = HdcLtcUnifiedNeuron::new_default(42);
        let input = ContinuousHV::random_default(123);

        // First evolve to a non-zero state
        for _ in 0..10 {
            neuron.evolve_closed_form(0.1, &input);
        }

        // Now apply extreme dt — should not produce NaN/Inf
        neuron.evolve_closed_form(100.0, &input);
        let state = neuron.state();
        assert!(
            state.norm().is_finite(),
            "State norm must be finite after extreme dt"
        );

        // Also test the iterative path
        neuron.evolve_closed_form_iterative(100.0, &input);
        let state = neuron.state();
        assert!(
            state.norm().is_finite(),
            "State norm must be finite after extreme dt (iterative)"
        );

        // Verify decay > 0 invariant: with tau=0.01 and sub_dt as small as possible,
        // the clamped exponent (-87.0) gives exp(-87) ≈ 1.6e-38 > 0
        let decay = (-100.0_f32 / 0.01).max(-87.0).exp();
        assert!(
            decay > 0.0,
            "Decay must be strictly positive, got {}",
            decay
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 2: Orthogonal HV init tests
    // ═══════════════════════════════════════════════════════════════════

    // ═══════════════════════════════════════════════════════════════════
    // Phase 3: Fourier basis injection tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_fourier_basis_empty_config() {
        let neuron = HdcLtcUnifiedNeuron::new(UnifiedConfig::default(), 42);
        assert!(
            neuron.compute_fourier_basis().is_none(),
            "Empty fourier_frequencies should return None"
        );
    }

    #[test]
    fn test_fourier_basis_single_freq() {
        let config = UnifiedConfig {
            fourier_frequencies: vec![1.0],
            fourier_amplitude: 0.5,
            ..UnifiedConfig::default()
        };
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);
        neuron.total_time = 0.25; // t=0.25s → sin(2π·1·0.25) = sin(π/2) = 1.0
        let basis = neuron.compute_fourier_basis().unwrap();
        // Sin channel should have amplitude ≈ 0.5
        assert!(
            (basis.values[0] - 0.5).abs() < 0.01,
            "Sin(2π·1·0.25) × 0.5 should be ≈ 0.5, got {}",
            basis.values[0]
        );
    }

    #[test]
    fn test_fourier_basis_multiple_freqs() {
        let config = UnifiedConfig {
            fourier_frequencies: vec![1.0, 2.0],
            fourier_amplitude: 0.1,
            ..UnifiedConfig::default()
        };
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);
        neuron.total_time = 0.5;
        let basis = neuron.compute_fourier_basis().unwrap();
        // With 2 freqs, total_channels=4, stride=4
        // Channel 0 (sin 1Hz), 1 (cos 1Hz), 2 (sin 2Hz), 3 (cos 2Hz)
        assert!(basis.values[0].abs() <= 0.1 + 0.01); // sin(2π·1·0.5) = sin(π) ≈ 0
        assert!((basis.values[1] - (-0.1)).abs() < 0.01); // cos(2π·1·0.5) = cos(π) = -1 × 0.1
    }

    #[test]
    fn test_fourier_evolve_differs_from_no_fourier() {
        let input = ContinuousHV::random_default(123);

        // Without Fourier
        let mut neuron_no = HdcLtcUnifiedNeuron::new(UnifiedConfig::default(), 42);
        for _ in 0..50 {
            neuron_no.evolve_closed_form(0.02, &input);
        }

        // With Fourier
        let config_fourier = UnifiedConfig {
            fourier_frequencies: vec![1.0, 5.0],
            fourier_amplitude: 0.1,
            ..UnifiedConfig::default()
        };
        let mut neuron_f = HdcLtcUnifiedNeuron::new(config_fourier, 42);
        for _ in 0..50 {
            neuron_f.evolve_closed_form(0.02, &input);
        }

        let sim = neuron_no.state().similarity(neuron_f.state());
        assert!(
            sim < 0.99,
            "Fourier injection should produce different state, got similarity {}",
            sim
        );
    }

    #[test]
    fn test_fourier_time_varying() {
        let config = UnifiedConfig {
            fourier_frequencies: vec![1.0],
            fourier_amplitude: 0.1,
            ..UnifiedConfig::default()
        };

        let mut neuron1 = HdcLtcUnifiedNeuron::new(config.clone(), 42);
        neuron1.total_time = 0.0;
        let basis1 = neuron1.compute_fourier_basis().unwrap();

        let mut neuron2 = HdcLtcUnifiedNeuron::new(config, 42);
        neuron2.total_time = 0.25;
        let basis2 = neuron2.compute_fourier_basis().unwrap();

        let sim = basis1.similarity(&basis2);
        assert!(
            sim < 0.99,
            "Fourier basis at different times should differ, got similarity {}",
            sim
        );
    }

    #[test]
    fn test_fourier_amplitude_scaling() {
        let config_small = UnifiedConfig {
            fourier_frequencies: vec![1.0],
            fourier_amplitude: 0.01,
            ..UnifiedConfig::default()
        };
        let config_large = UnifiedConfig {
            fourier_frequencies: vec![1.0],
            fourier_amplitude: 1.0,
            ..UnifiedConfig::default()
        };

        let mut neuron_s = HdcLtcUnifiedNeuron::new(config_small, 42);
        neuron_s.total_time = 0.25;
        let basis_s = neuron_s.compute_fourier_basis().unwrap();

        let mut neuron_l = HdcLtcUnifiedNeuron::new(config_large, 42);
        neuron_l.total_time = 0.25;
        let basis_l = neuron_l.compute_fourier_basis().unwrap();

        let norm_s = basis_s.norm();
        let norm_l = basis_l.norm();
        assert!(
            norm_l > norm_s * 10.0,
            "Large amplitude should produce larger norm: small={}, large={}",
            norm_s,
            norm_l
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 2: Orthogonal HV init tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_neuron_orthogonal_init_separates_hvs() {
        let neuron = HdcLtcUnifiedNeuron::new(UnifiedConfig::default(), 42);
        let hvs = [
            &neuron.weight_hv,
            &neuron.input_mask,
            &neuron.tau_modulator,
            &neuron.gate_weight,
        ];
        for i in 0..hvs.len() {
            for j in (i + 1)..hvs.len() {
                let sim = hvs[i].similarity(hvs[j]);
                assert!(
                    sim.abs() < 0.02,
                    "HVs {} and {} should be near-orthogonal, got similarity {}",
                    i,
                    j,
                    sim
                );
            }
        }
    }

    #[test]
    fn test_neuron_orthogonal_init_evolves_correctly() {
        let mut neuron = HdcLtcUnifiedNeuron::new(UnifiedConfig::default(), 42);
        let input = ContinuousHV::random_default(123);

        // Should evolve without issues
        for _ in 0..50 {
            neuron.evolve_closed_form(0.02, &input);
        }
        let norm = neuron.state().norm();
        assert!(
            norm > 0.0 && norm.is_finite(),
            "State should evolve, got norm={}",
            norm
        );
    }

    #[test]
    fn test_neuron_backward_compatible() {
        // After 100 steps, state should be bounded
        let mut neuron = HdcLtcUnifiedNeuron::new(UnifiedConfig::default(), 42);
        let input = ContinuousHV::random_default(100);
        for _ in 0..100 {
            neuron.evolve_closed_form(0.01, &input);
        }
        let norm = neuron.state().norm();
        assert!(
            norm < 200.0,
            "State should remain bounded after 100 steps, got {}",
            norm
        );
    }

    #[test]
    fn test_set_fourier_frequencies_neuron() {
        let config = UnifiedConfig {
            fourier_frequencies: vec![1.0, 2.0],
            ..UnifiedConfig::default()
        };
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);
        assert_eq!(neuron.config().fourier_frequencies, vec![1.0, 2.0]);

        neuron.set_fourier_frequencies(&[5.0, 10.0, 15.0]);
        assert_eq!(neuron.config().fourier_frequencies, vec![5.0, 10.0, 15.0]);
    }
}
