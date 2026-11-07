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

    /// State and Fourier phase immediately before the last standard adaptive
    /// closed-form step. Used by backward() when the caller trains after forward.
    /// This is ephemeral execution state and is intentionally not serialized.
    #[serde(skip)]
    pre_evolution_state: Option<ContinuousHV>,
    #[serde(skip)]
    pre_evolution_time: f64,
    #[serde(skip)]
    pre_evolution_dt: f32,

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

#[inline]
fn sanitize_hdc_ltc_dt(dt: f32) -> f32 {
    if dt.is_finite() { dt.max(0.0) } else { 0.0 }
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

/// Stable typed identity for a learnable hypervector in the unified HDC-LTC neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HdcLtcParameterGroupId {
    WeightHypervector,
    InputMask,
    TauModulator,
    GateWeight,
    GateBias,
}

/// Failure returned when a proposed HDC-LTC parameter-group edit is invalid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HdcLtcParameterEditError {
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteValue {
        index: usize,
    },
    NormExceeded {
        group: HdcLtcParameterGroupId,
        norm_micros: u64,
    },
}

impl std::fmt::Display for HdcLtcParameterEditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "HDC-LTC parameter dimension mismatch: expected {expected}, received {actual}"
                )
            }
            Self::NonFiniteValue { index } => {
                write!(f, "HDC-LTC parameter value at index {index} is not finite")
            }
            Self::NormExceeded { group, norm_micros } => write!(
                f,
                "HDC-LTC parameter group {group:?} exceeds the norm budget: {:.6}",
                *norm_micros as f64 / 1_000_000.0
            ),
        }
    }
}

impl std::error::Error for HdcLtcParameterEditError {}

/// Architecture-aware summary of one HDC-LTC parameter hypervector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcLtcParameterGroupProfile {
    pub id: HdcLtcParameterGroupId,
    pub name: String,
    pub dimension: usize,
    pub norm: f32,
    pub mean: f32,
    pub variance: f32,
    pub min_value: f32,
    pub max_value: f32,
    pub non_finite_count: usize,
    pub saturated_count: usize,
}

/// Read-only, versioned introspection result for one HDC-LTC neuron.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcLtcNeuronParameterProfile {
    pub format_version: u32,
    pub dimension: usize,
    pub tau_base: f32,
    pub backbone_tau: f32,
    pub gating_steepness: f32,
    pub fourier_frequencies: Vec<f32>,
    pub groups: Vec<HdcLtcParameterGroupProfile>,
}

fn profile_hypervector(
    id: HdcLtcParameterGroupId,
    name: &str,
    hv: &ContinuousHV,
) -> HdcLtcParameterGroupProfile {
    let dimension = hv.values.len();
    let non_finite_count = hv.values.iter().filter(|value| !value.is_finite()).count();
    let finite_values: Vec<f32> = hv
        .values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let mean = if finite_values.is_empty() {
        0.0
    } else {
        finite_values.iter().sum::<f32>() / finite_values.len() as f32
    };
    let variance = if finite_values.is_empty() {
        0.0
    } else {
        finite_values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / finite_values.len() as f32
    };
    let min_value = finite_values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_value = finite_values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    HdcLtcParameterGroupProfile {
        id,
        name: name.to_string(),
        dimension,
        norm: hv.norm(),
        mean,
        variance,
        min_value: if finite_values.is_empty() {
            0.0
        } else {
            min_value
        },
        max_value: if finite_values.is_empty() {
            0.0
        } else {
            max_value
        },
        non_finite_count,
        saturated_count: finite_values
            .iter()
            .filter(|value| value.abs() >= 0.99)
            .count(),
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
            pre_evolution_state: None,
            pre_evolution_time: 0.0,
            pre_evolution_dt: 0.0,
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

    /// Create a neuron with domain-separated, deterministic GenesisSeed parameters.
    ///
    /// Each generated vector is orthogonalized and normalized to match the
    /// unit-norm, low-interference initialization contract of [`Self::new`].
    /// Domain labels are derived as `"{label}::weight_hv"`,
    /// `"{label}::input_mask"`, and so on.
    pub fn from_genesis(
        config: UnifiedConfig,
        genesis: &crate::genesis::GenesisSeed,
        label: &str,
    ) -> Self {
        let dim = config.dimension;

        // Match new()'s initialization contract while retaining GenesisSeed's
        // domain-separated determinism. The raw genesis HV components are
        // uniform in [-1, 1] (norm ~= sqrt(D/3)); feeding those directly into
        // Hadamard state transforms creates a different dynamical regime from
        // new()'s unit-norm orthogonal vectors. Modified Gram-Schmidt makes
        // both constructors use unit-norm, low-interference parameter vectors.
        let mut parameters = vec![
            genesis.hv(&format!("{label}::weight_hv"), dim),
            genesis.hv(&format!("{label}::input_mask"), dim),
            genesis.hv(&format!("{label}::tau_modulator"), dim),
            genesis.hv(&format!("{label}::gate_weight"), dim),
            genesis.hv(&format!("{label}::gate_bias"), dim),
        ];

        for index in 0..parameters.len() {
            for previous in 0..index {
                let projection = parameters[index].dot(&parameters[previous]);
                parameters[index] =
                    parameters[index].subtract(&parameters[previous].scale(projection));
            }

            let norm = parameters[index].norm();
            if norm > 1e-10 {
                parameters[index] = parameters[index].scale(1.0 / norm);
            }
        }

        Self {
            state: ContinuousHV::zero(dim),
            pre_evolution_state: None,
            pre_evolution_time: 0.0,
            pre_evolution_dt: 0.0,
            weight_hv: parameters[0].clone(),
            input_mask: parameters[1].clone(),
            tau_modulator: parameters[2].clone(),
            gate_weight: parameters[3].clone(),
            gate_bias: parameters[4].scale(0.1),
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

    /// Retain the state and Fourier phase before the next adaptive closed-form step.
    ///
    /// Reuses the same allocation after the first capture so steady-state evolution
    /// does not allocate an additional hypervector on every frame.
    fn capture_pre_evolution_state(&mut self, dt: f32) {
        self.pre_evolution_time = self.total_time;
        self.pre_evolution_dt = dt;
        match self.pre_evolution_state.as_mut() {
            Some(previous) => previous.values.copy_from_slice(&self.state.values),
            None => self.pre_evolution_state = Some(self.state.clone()),
        }
    }

    /// Clear the local-gradient snapshot when a different integrator or an
    /// external state mutation makes the adaptive one-step derivative inapplicable.
    fn clear_pre_evolution_state(&mut self) {
        self.pre_evolution_state = None;
    }

    /// Compute a Fourier basis HV from configured frequencies at an explicit
    /// evolution time. The explicit time is required when differentiating a
    /// forward step after the neuron's live clock has advanced.
    fn compute_fourier_basis_at(&self, time: f64) -> Option<ContinuousHV> {
        let freqs = &self.config.fourier_frequencies;
        if freqs.is_empty() {
            return None;
        }

        let dim = self.config.dimension;
        let amp = self.config.fourier_amplitude;
        let mut values = vec![0.0f32; dim];
        let two_pi = 2.0 * std::f32::consts::PI;
        let total_channels = freqs.len() * 2;

        for (fi, &freq) in freqs.iter().enumerate() {
            let sin_val = (two_pi * freq * time as f32).sin() * amp;
            let cos_val = (two_pi * freq * time as f32).cos() * amp;
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

    #[inline]
    fn compute_fourier_basis(&self) -> Option<ContinuousHV> {
        self.compute_fourier_basis_at(self.total_time)
    }

    /// Pre-activation shared by forward evolution and local gradients.
    #[inline]
    fn compute_pre_activation_at(
        &self,
        state: &ContinuousHV,
        input: &ContinuousHV,
        time: f64,
    ) -> ContinuousHV {
        let transformed_state = self.weight_hv.bind(state);
        let masked_input = self.input_mask.bind(input);

        if let Some(ref fourier) = self.compute_fourier_basis_at(time) {
            ContinuousHV::bundle(&[&transformed_state, &masked_input, fourier])
        } else {
            ContinuousHV::bundle(&[&transformed_state, &masked_input])
        }
    }

    #[inline]
    fn compute_pre_activation(&self, input: &ContinuousHV) -> ContinuousHV {
        self.compute_pre_activation_at(&self.state, input, self.total_time)
    }

    /// Compute the equilibrium state x_∞ for the current state and input.
    #[inline]
    fn compute_equilibrium(&self, input: &ContinuousHV) -> ContinuousHV {
        self.config
            .activation
            .apply(&self.compute_pre_activation(input))
    }

    /// Compute effective time constant τ(||x||, u)
    ///
    /// Time constant adapts to both state complexity and input
    #[inline]
    fn compute_tau_unclamped_at(&self, state: &ContinuousHV, input: &ContinuousHV) -> f32 {
        let state_norm = state.norm();
        let input_adjustment = cosine_similarity_full(input, &self.tau_modulator);

        self.config.tau_base
            * (1.0 + self.config.backbone_tau * state_norm)
            * (1.0 + 0.2 * input_adjustment)
    }

    #[inline]
    fn compute_tau_unclamped(&self, input: &ContinuousHV) -> f32 {
        self.compute_tau_unclamped_at(&self.state, input)
    }

    #[inline]
    fn compute_tau(&self, input: &ContinuousHV) -> f32 {
        self.compute_tau_unclamped(input).clamp(0.01, 10.0)
    }

    /// State/input similarity plus mean gate bias before the sigmoid.
    ///
    /// This is factored out so forward evolution and local gradients use exactly
    /// the same gate definition.
    #[inline]
    fn compute_gate_activation_at(&self, state: &ContinuousHV, input: &ContinuousHV) -> f32 {
        let dim = self.config.dimension;
        let mut dot = 0.0f32;
        let mut bundle_norm_sq = 0.0f32;
        let mut gate_norm_sq = 0.0f32;
        let mut bias_sum = 0.0f32;

        for i in 0..dim {
            let bundled = (state.values[i] + input.values[i]) * 0.5;
            let gate = self.gate_weight.values[i];
            dot += bundled * gate;
            bundle_norm_sq += bundled * bundled;
            gate_norm_sq += gate * gate;
            bias_sum += self.gate_bias.values[i];
        }

        let denominator = (bundle_norm_sq * gate_norm_sq).sqrt();
        let similarity = if denominator < 1e-10 {
            0.0
        } else {
            dot / denominator
        };
        similarity + bias_sum / dim as f32
    }

    #[inline]
    fn compute_gate_activation(&self, input: &ContinuousHV) -> f32 {
        self.compute_gate_activation_at(&self.state, input)
    }

    /// Return the base sigmoid gate before the time-scaled interpolation.
    #[inline]
    fn compute_gate_base(&self, input: &ContinuousHV) -> f32 {
        let activation = self.compute_gate_activation(input);
        1.0 / (1.0 + (-activation * self.config.gating_steepness).exp())
    }

    /// Compute the continuous-time interpolation factor.
    ///
    /// The learned gate scales the inverse time constant:
    /// sigma = 1 - exp(-dt * gate(input, state) / tau).
    /// Unlike an additive interpolation floor, this satisfies sigma(0) = 0.
    #[inline]
    fn compute_gating(&self, input: &ContinuousHV, dt: f32) -> f32 {
        let dt = sanitize_hdc_ltc_dt(dt);
        let tau = self.compute_tau(input);
        let gate_rate = self.compute_gate_base(input);
        let exponent = -dt * gate_rate / tau;
        (-exponent.exp_m1()).clamp(0.0, 1.0)
    }

    /// Evolve the same gated liquid ODE with forward Euler integration.
    ///
    /// dx/dt = gate_rate(x,u) * (x_inf(x,u,t) - x) / tau(x,u)
    pub fn evolve(&mut self, dt: f32, input: &ContinuousHV) {
        self.clear_pre_evolution_state();
        let dt = sanitize_hdc_ltc_dt(dt);
        let x_inf = self.compute_equilibrium(input);
        let tau = self.compute_tau(input);
        let gate_rate = self.compute_gate_base(input);

        // dx/dt = gate_rate(x,u) * (x_inf(x,u) - x) / tau(x,u)
        let derivative = x_inf.subtract(&self.state).scale(gate_rate / tau);

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
        self.clear_pre_evolution_state();
        let h = sanitize_hdc_ltc_dt(dt);
        let start_time = self.total_time;

        // Each RK stage evaluates the same state/input/time-dependent vector field.
        let k1 = self.compute_derivative(input, &self.state, start_time);

        let state_k2 = self.state.add(&k1.scale(h / 2.0));
        let k2 = self.compute_derivative(input, &state_k2, start_time + (h / 2.0) as f64);

        let state_k3 = self.state.add(&k2.scale(h / 2.0));
        let k3 = self.compute_derivative(input, &state_k3, start_time + (h / 2.0) as f64);

        let state_k4 = self.state.add(&k3.scale(h));
        let k4 = self.compute_derivative(input, &state_k4, start_time + h as f64);

        // Combine: x += (k1 + 2k2 + 2k3 + k4) * h/6
        let sum = k1.add(&k2.scale(2.0)).add(&k3.scale(2.0)).add(&k4);
        self.state = self.state.add(&sum.scale(h / 6.0));

        self.apply_state_bounds();
        self.update_stats(h);
    }

    /// **Frozen-coefficient liquid update** - O(D) one-step temporal update
    ///
    /// This is the key innovation enabling efficient temporal reasoning.
    ///
    /// ## Mathematical Basis
    ///
    /// For the gated ODE:
    /// dx/dt = gate_rate(x,u) * (x_∞(x,u,t) - x) / τ(x,u).
    ///
    /// Freezing equilibrium, tau, and gate_rate over one interval gives the
    /// local exponential update:
    /// ```text
    /// σ = 1 - exp(-Δt * gate_rate / τ)
    /// x(t+Δt) = σ * x_∞ + (1-σ) * x(t)
    /// ```
    ///
    /// With a learned positive rate multiplier g(x,u):
    /// ```text
    /// σ = 1 - exp(-Δt * g(x,u) / τ(x,u))
    /// x(t+Δt) = σ × x_∞ + (1-σ) × x(t)
    /// ```
    /// so σ(0)=0 and the zero-duration map is the identity. The frozen
    /// equilibrium is still a one-step approximation to the nonlinear ODE.
    ///
    /// ## Complexity
    ///
    /// O(D) where D is hypervector dimension - independent of Δt!
    #[inline]
    pub fn evolve_closed_form(&mut self, dt: f32, input: &ContinuousHV) {
        let dt = sanitize_hdc_ltc_dt(dt);
        self.capture_pre_evolution_state(dt);
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
        let dt = sanitize_hdc_ltc_dt(dt);
        // The fused kernels use an approximate activation in their SIMD paths
        // and do not share the exact scalar forward map used by backward().
        self.clear_pre_evolution_state();
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
    /// This method freezes the learned gate rate and state-dependent equilibrium
    /// for one exponential step; it is not a nonlinear exact solution.
    ///
    /// ## Mathematical Basis
    ///
    /// For the frozen-coefficient ODE:
    /// dx/dt = gate_rate * (x_∞ - x) / τ with CONSTANT x_∞, tau and gate_rate
    ///
    /// The exact analytical solution for those frozen coefficients is:
    /// ```text
    /// x(t+Δt) = x_∞ + (x(t) - x_∞) × exp(-Δt * gate_rate / τ)
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
        self.clear_pre_evolution_state();
        let dt = sanitize_hdc_ltc_dt(dt);

        // Freeze the state-dependent equilibrium, tau, and gate at the start
        // of the interval; integrate that local linear system analytically.
        let x_inf = self.compute_equilibrium(input);
        let tau = self.compute_tau(input);
        let gate_rate = self.compute_gate_base(input);
        let exponent = -dt * gate_rate / tau;
        let sigma = (-exponent.exp_m1()).clamp(0.0, 1.0);

        self.state.lerp_in_place(&x_inf, 1.0 - sigma, sigma);
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
    /// 1. Split large dt into smaller sub-steps (at most nominal tau/10 per step)
    /// 2. Recompute time-dependent Fourier basis, equilibrium, tau and gate
    /// 3. Apply the matching frozen-coefficient exponential update per sub-step
    ///
    /// This achieves O(dt/tau) complexity instead of O(1) but provides accuracy
    /// comparable to RK4 integration.
    ///
    /// ## Parameters
    ///
    /// - `dt`: Total time to evolve
    /// - `input`: Input hypervector (assumed constant during evolution)
    pub fn evolve_closed_form_iterative(&mut self, dt: f32, input: &ContinuousHV) {
        self.clear_pre_evolution_state();
        let dt = sanitize_hdc_ltc_dt(dt);
        let max_substep = (self.config.tau_base / 10.0).max(1e-5);
        let n_steps = ((dt / max_substep).ceil() as usize).max(1);
        let sub_dt = dt / n_steps as f32;
        let start_time = self.total_time;

        for step in 0..n_steps {
            let step_time = start_time + (step as f64 * sub_dt as f64);
            let equilibrium = self
                .config
                .activation
                .apply(&self.compute_pre_activation_at(&self.state, input, step_time));
            let tau = self
                .compute_tau_unclamped_at(&self.state, input)
                .clamp(0.01, 10.0);
            let gate_activation = self.compute_gate_activation_at(&self.state, input);
            let gate_rate = 1.0 / (1.0 + (-gate_activation * self.config.gating_steepness).exp());
            let exponent = -sub_dt * gate_rate / tau;
            let sigma = (-exponent.exp_m1()).clamp(0.0, 1.0);
            self.state.lerp_in_place(&equilibrium, 1.0 - sigma, sigma);
        }

        self.apply_state_bounds();
        self.update_stats(dt);
    }

    /// Evaluate the same continuous-time vector field at an explicit state/time.
    /// This is the numerical reference used by RK4.
    fn compute_derivative(
        &self,
        input: &ContinuousHV,
        state: &ContinuousHV,
        time: f64,
    ) -> ContinuousHV {
        let equilibrium = self
            .config
            .activation
            .apply(&self.compute_pre_activation_at(state, input, time));
        let tau = self
            .compute_tau_unclamped_at(state, input)
            .clamp(0.01, 10.0);
        let gate_activation = self.compute_gate_activation_at(state, input);
        let gate_rate = 1.0 / (1.0 + (-gate_activation * self.config.gating_steepness).exp());

        // dx/dt = gate_rate(state,input) * (equilibrium(state,input,time) - state) / tau(state,input)
        equilibrium.subtract(state).scale(gate_rate / tau)
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

    /// Return a versioned, read-only summary of all learnable HDC-LTC vectors.
    pub fn parameter_profile(&self) -> HdcLtcNeuronParameterProfile {
        HdcLtcNeuronParameterProfile {
            format_version: 1,
            dimension: self.config.dimension,
            tau_base: self.config.tau_base,
            backbone_tau: self.config.backbone_tau,
            gating_steepness: self.config.gating_steepness,
            fourier_frequencies: self.config.fourier_frequencies.clone(),
            groups: vec![
                profile_hypervector(
                    HdcLtcParameterGroupId::WeightHypervector,
                    "weight_hv",
                    &self.weight_hv,
                ),
                profile_hypervector(
                    HdcLtcParameterGroupId::InputMask,
                    "input_mask",
                    &self.input_mask,
                ),
                profile_hypervector(
                    HdcLtcParameterGroupId::TauModulator,
                    "tau_modulator",
                    &self.tau_modulator,
                ),
                profile_hypervector(
                    HdcLtcParameterGroupId::GateWeight,
                    "gate_weight",
                    &self.gate_weight,
                ),
                profile_hypervector(
                    HdcLtcParameterGroupId::GateBias,
                    "gate_bias",
                    &self.gate_bias,
                ),
            ],
        }
    }

    /// Return an immutable reference to one typed learnable parameter vector.
    pub fn parameter_group(&self, id: HdcLtcParameterGroupId) -> &ContinuousHV {
        match id {
            HdcLtcParameterGroupId::WeightHypervector => &self.weight_hv,
            HdcLtcParameterGroupId::InputMask => &self.input_mask,
            HdcLtcParameterGroupId::TauModulator => &self.tau_modulator,
            HdcLtcParameterGroupId::GateWeight => &self.gate_weight,
            HdcLtcParameterGroupId::GateBias => &self.gate_bias,
        }
    }

    /// Read a bounded immutable window from a named parameter vector.
    ///
    /// Returns None for an out-of-range window. This is the preferred inspection
    /// boundary for tools that should not materialize a complete HDC vector.
    pub fn parameter_group_window(
        &self,
        id: HdcLtcParameterGroupId,
        offset: usize,
        length: usize,
    ) -> Option<&[f32]> {
        let end = offset.checked_add(length)?;
        self.parameter_group(id).values.get(offset..end)
    }

    /// Replace one typed learnable hypervector after validating the entire payload.
    ///
    /// The candidate is prepared and validated before the selected live vector is
    /// replaced. The 2.0 L2 budget matches the existing post-update norm ceiling.
    pub fn try_replace_parameter_group(
        &mut self,
        id: HdcLtcParameterGroupId,
        values: &[f32],
    ) -> Result<(), HdcLtcParameterEditError> {
        if values.len() != self.config.dimension {
            return Err(HdcLtcParameterEditError::DimensionMismatch {
                expected: self.config.dimension,
                actual: values.len(),
            });
        }

        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(HdcLtcParameterEditError::NonFiniteValue { index });
        }

        let candidate = ContinuousHV::from_values(values.to_vec());
        let norm = candidate.norm();
        if norm > 2.0 {
            return Err(HdcLtcParameterEditError::NormExceeded {
                group: id,
                norm_micros: (norm as f64 * 1_000_000.0).round() as u64,
            });
        }

        self.clear_pre_evolution_state();
        match id {
            HdcLtcParameterGroupId::WeightHypervector => {
                self.weight_hv = candidate;
                self.weight_momentum = ContinuousHV::zero(self.config.dimension);
            }
            HdcLtcParameterGroupId::InputMask => {
                self.input_mask = candidate;
                self.input_momentum = ContinuousHV::zero(self.config.dimension);
            }
            HdcLtcParameterGroupId::TauModulator => self.tau_modulator = candidate,
            HdcLtcParameterGroupId::GateWeight => self.gate_weight = candidate,
            HdcLtcParameterGroupId::GateBias => self.gate_bias = candidate,
        }

        Ok(())
    }

    /// Get current state
    pub fn state(&self) -> &ContinuousHV {
        &self.state
    }

    /// Get mutable state reference
    pub fn state_mut(&mut self) -> &mut ContinuousHV {
        self.clear_pre_evolution_state();
        &mut self.state
    }

    /// Set state directly
    pub fn set_state(&mut self, state: ContinuousHV) {
        self.clear_pre_evolution_state();
        self.state = state;
    }

    /// Update the Fourier basis frequencies used in equilibrium modulation.
    pub fn set_fourier_frequencies(&mut self, freqs: &[f32]) {
        self.clear_pre_evolution_state();
        self.config.fourier_frequencies = freqs.to_vec();
    }

    /// Reset state to zero
    pub fn reset(&mut self) {
        self.clear_pre_evolution_state();
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
        self.clear_pre_evolution_state();
        self.config.tau_base = tau.clamp(0.01, 1.0);
    }

    /// Get total time evolved
    /// Evolution clock: (total_time, update_count). total_time is the
    /// Fourier equilibrium phase — evolution state, snapshot-relevant.
    pub fn evolution_clock(&self) -> (f64, u64) {
        (self.total_time, self.update_count)
    }

    /// Restore the evolution clock (see [`Self::evolution_clock`]).
    pub fn set_evolution_clock(&mut self, total_time: f64, update_count: u64) {
        self.clear_pre_evolution_state();
        self.total_time = total_time;
        self.update_count = update_count;
    }

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
        self.clear_pre_evolution_state();
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
        self.clear_pre_evolution_state();
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
        self.clear_pre_evolution_state();
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
        self.clear_pre_evolution_state();
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
        self.clear_pre_evolution_state();
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
        self.clear_pre_evolution_state();
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
    // ONE-STEP LOCAL GRADIENTS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Compute analytical one-step local gradients for the adaptive closed-form map.
    ///
    /// After `evolve_closed_form()`, this uses the retained pre-evolution state
    /// and Fourier phase, while `target` is expressed in post-evolution output
    /// space. Without a compatible retained step, it falls back to current state.
    ///
    /// This is local credit assignment for one step, not BPTT through time or
    /// a full chain-rule gradient through the network's layer composition.
    /// Since HDC binding is Hadamard product, ∂(A⊙B)/∂A = B element-wise.
    pub fn backward(
        &self,
        input: &ContinuousHV,
        target: &ContinuousHV,
        dt: f32,
    ) -> HdcLtcGradients {
        match self.pre_evolution_state.as_ref() {
            Some(previous) => self.backward_from_state(
                previous,
                self.pre_evolution_time,
                input,
                target,
                self.pre_evolution_dt,
            ),
            None => self.backward_at_current(input, target, dt),
        }
    }

    /// Compute the same one-step local gradient at the neuron's current state.
    ///
    /// Use this when the learning objective is explicitly evaluated at the
    /// current state rather than against a completed forward step.
    pub fn backward_at_current(
        &self,
        input: &ContinuousHV,
        target: &ContinuousHV,
        dt: f32,
    ) -> HdcLtcGradients {
        self.backward_from_state(&self.state, self.total_time, input, target, dt)
    }

    fn backward_from_state(
        &self,
        state: &ContinuousHV,
        state_time: f64,
        input: &ContinuousHV,
        target: &ContinuousHV,
        dt: f32,
    ) -> HdcLtcGradients {
        let dim = self.config.dimension;

        // The local derivative is evaluated at the exact state/time used by
        // the forward prediction. backward() supplies the retained context.
        let z = self.compute_pre_activation_at(state, input, state_time);
        let equilibrium = self.config.activation.apply(&z);
        let tau_unclamped = self.compute_tau_unclamped_at(state, input);
        let tau = tau_unclamped.clamp(0.01, 10.0);
        let tau_gradient_active = tau_unclamped > 0.01 && tau_unclamped < 10.0;
        let gate_activation = self.compute_gate_activation_at(state, input);
        let gate_base = 1.0 / (1.0 + (-gate_activation * self.config.gating_steepness).exp());
        let dt = sanitize_hdc_ltc_dt(dt);
        let exponent = -dt * gate_base / tau;
        let decay = exponent.exp();
        let sigma = (-exponent.exp_m1()).clamp(0.0, 1.0);
        let new_state = equilibrium.scale(sigma).add(&state.scale(1.0 - sigma));

        // Mean-squared-error derivative at the one-step prediction.
        let dh = new_state.subtract(target).scale(2.0 / dim as f32);
        let dloss_dsigma: f32 = dh
            .values
            .iter()
            .zip(equilibrium.values.iter().zip(state.values.iter()))
            .map(|(&error, (&eq, &state))| error * (eq - state))
            .sum();

        // The bundle is the arithmetic mean of two or three components.
        let bundle_terms = if self.config.fourier_frequencies.is_empty() {
            2.0
        } else {
            3.0
        };
        let activation = self.config.activation;
        let dz = ContinuousHV::from_values(
            dh.values
                .iter()
                .zip(z.values.iter())
                .map(|(&error, &pre)| error * sigma * activation.derivative(pre) / bundle_terms)
                .collect(),
        );

        // Direct equilibrium-path derivatives (binding is Hadamard product).
        let dw = dz.bind(&state);
        let du = dz.bind(input);

        // Exact derivative of sigma with respect to the effective scalar tau.
        let dsigma_dtau = if tau_gradient_active {
            -decay * gate_base * dt / (tau * tau)
        } else {
            0.0
        };
        let dloss_dtau = dloss_dsigma * dsigma_dtau;

        // tau = tau_base * (1 + backbone_tau * ||state||)
        //                 * (1 + 0.2 * cosine(input, tau_modulator)).
        let state_norm = state.norm();
        let input_tau_similarity = cosine_similarity_full(input, &self.tau_modulator);
        let tau_mod_coefficient =
            self.config.tau_base * (1.0 + self.config.backbone_tau * state_norm) * 0.2;
        let tau_similarity_gradient = dloss_dtau * tau_mod_coefficient;
        let d_tau_modulator =
            cosine_gradient_wrt_right(input, &self.tau_modulator).scale(tau_similarity_gradient);
        let d_input_tau =
            cosine_gradient_wrt_left(input, &self.tau_modulator).scale(tau_similarity_gradient);
        let tau_state_coefficient =
            self.config.tau_base * self.config.backbone_tau * (1.0 + 0.2 * input_tau_similarity);
        let d_state_tau = if state_norm > 1e-10 {
            state
                .scale(tau_state_coefficient / state_norm)
                .scale(dloss_dtau)
        } else {
            ContinuousHV::zero(dim)
        };

        // The adaptive gate is based on cosine(bundle(state, input), gate_weight)
        // plus the mean gate bias. Differentiate this path as well.
        let bundled_state_input = state.add(input).scale(0.5);
        let dloss_dgate_activation = dloss_dsigma
            * decay
            * (dt / tau)
            * self.config.gating_steepness
            * gate_base
            * (1.0 - gate_base);
        let d_gate_weight = cosine_gradient_wrt_right(&bundled_state_input, &self.gate_weight)
            .scale(dloss_dgate_activation);
        let d_gate_bias = ContinuousHV::from_values(vec![dloss_dgate_activation / dim as f32; dim]);
        let gate_bundle_gradient =
            cosine_gradient_wrt_left(&bundled_state_input, &self.gate_weight)
                .scale(0.5 * dloss_dgate_activation);

        // d_input and d_state include the direct equilibrium path, the gate path,
        // the tau path, and (for state) the residual interpolation branch.
        let d_input = dz
            .bind(&self.input_mask)
            .add(&gate_bundle_gradient)
            .add(&d_input_tau);
        let d_state = dz
            .bind(&self.weight_hv)
            .add(&dh.scale(1.0 - sigma))
            .add(&gate_bundle_gradient)
            .add(&d_state_tau);

        HdcLtcGradients {
            dw,
            du,
            dtau_scalar: dloss_dtau,
            d_tau_modulator,
            d_gate_weight,
            d_gate_bias,
            d_input,
            d_state,
        }
    }

    /// Apply one-step local gradients with SGD + momentum, weight decay, and norm clipping.
    pub fn apply_gradients(&mut self, grads: &HdcLtcGradients, lr: f32) {
        self.apply_gradients_inner(grads, lr, true)
    }

    /// Apply gradients without per-step weight decay.
    ///
    /// Use this for sequence trainers where thousands of local gradient steps occur per
    /// epoch. The standard `apply_gradients` applies weight_decay (default
    /// 0.0001) per step, which over 67K steps/epoch decays weights to ~0.1%
    /// of original (`(1-0.0001)^67646 ≈ 0.001`), destroying the network.
    pub fn apply_gradients_no_decay(&mut self, grads: &HdcLtcGradients, lr: f32) {
        self.apply_gradients_inner(grads, lr, false)
    }

    fn apply_gradients_inner(&mut self, grads: &HdcLtcGradients, lr: f32, apply_decay: bool) {
        self.clear_pre_evolution_state();
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

        // Update cosine-based tau and gate directions. Their exact gradients are
        // orthogonal to the current unit vector; norm clipping prevents drift.
        self.tau_modulator = self
            .tau_modulator
            .subtract(&grads.d_tau_modulator.scale(lr));
        self.gate_weight = self.gate_weight.subtract(&grads.d_gate_weight.scale(lr));
        self.gate_bias = self.gate_bias.subtract(&grads.d_gate_bias.scale(lr));

        // Norm clip to 2.0 for all vector-valued parameter groups.
        if self.weight_hv.norm() > 2.0 {
            self.weight_hv = self.weight_hv.normalize().scale(2.0);
        }
        if self.input_mask.norm() > 2.0 {
            self.input_mask = self.input_mask.normalize().scale(2.0);
        }
        if self.tau_modulator.norm() > 2.0 {
            self.tau_modulator = self.tau_modulator.normalize().scale(2.0);
        }
        if self.gate_weight.norm() > 2.0 {
            self.gate_weight = self.gate_weight.normalize().scale(2.0);
        }
        if self.gate_bias.norm() > 2.0 {
            self.gate_bias = self.gate_bias.normalize().scale(2.0);
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
        self.clear_pre_evolution_state();
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
        self.clear_pre_evolution_state();
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
        self.clear_pre_evolution_state();
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

    /// Reset all transient state tied to an earlier parameterization.
    ///
    /// Use on an isolated candidate after an external weight-space edit:
    /// hidden state, temporal clock, update counter, momentum, and running
    /// dynamical statistics must not be carried across a changed substrate.
    pub fn reset_after_parameter_edit(&mut self) {
        self.reset();
        self.reset_momentum();
        self.running_mean = 0.0;
        self.running_var = 1.0;
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

/// Exact full-dimensional cosine similarity, independent of the global similarity stride.
#[inline]
fn cosine_similarity_full(left: &ContinuousHV, right: &ContinuousHV) -> f32 {
    let left_norm = left.norm();
    let right_norm = right.norm();
    let denominator = left_norm * right_norm;
    if denominator < 1e-10 {
        0.0
    } else {
        left.dot(right) / denominator
    }
}

/// Gradient of full-dimensional cosine similarity with respect to the left vector.
#[inline]
fn cosine_gradient_wrt_left(left: &ContinuousHV, right: &ContinuousHV) -> ContinuousHV {
    let left_norm = left.norm();
    let right_norm = right.norm();
    if left_norm < 1e-10 || right_norm < 1e-10 {
        return ContinuousHV::zero(left.dim());
    }
    let similarity = cosine_similarity_full(left, right);
    right
        .scale(1.0 / (left_norm * right_norm))
        .subtract(&left.scale(similarity / (left_norm * left_norm)))
}

/// Gradient of full-dimensional cosine similarity with respect to the right vector.
#[inline]
fn cosine_gradient_wrt_right(left: &ContinuousHV, right: &ContinuousHV) -> ContinuousHV {
    let left_norm = left.norm();
    let right_norm = right.norm();
    if left_norm < 1e-10 || right_norm < 1e-10 {
        return ContinuousHV::zero(right.dim());
    }
    let similarity = cosine_similarity_full(left, right);
    left.scale(1.0 / (left_norm * right_norm))
        .subtract(&right.scale(similarity / (right_norm * right_norm)))
}

/// One-step local gradients for the actual adaptive HDC-LTC forward map.
///
/// These gradients are local in time: they do not constitute BPTT through a
/// sequence. They include the current step's equilibrium, adaptive gate, and
/// state/input-dependent tau paths.
#[derive(Debug, Clone)]
pub struct HdcLtcGradients {
    /// Partial gradient for the state-binding hypervector `weight_hv`.
    pub dw: ContinuousHV,
    /// Partial gradient for the input-binding hypervector `input_mask`.
    pub du: ContinuousHV,
    /// Gradient with respect to the effective scalar time constant (diagnostic).
    pub dtau_scalar: f32,
    /// Gradient for the tau-modulator hypervector, including cosine geometry.
    pub d_tau_modulator: ContinuousHV,
    /// Gradient for the adaptive gate-weight hypervector.
    pub d_gate_weight: ContinuousHV,
    /// Gradient for the per-coordinate gate-bias storage (the forward path uses its mean).
    pub d_gate_bias: ContinuousHV,
    /// Local derivative with respect to the input hypervector, including gate and tau paths.
    pub d_input: ContinuousHV,
    /// Local derivative with respect to the pre-evolution state hypervector.
    pub d_state: ContinuousHV,
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

/// Reusable buffer holding a network's mutable evolution state
/// (see [`HdcLtcUnifiedNetwork::snapshot_state_into`]).
#[derive(Debug, Clone, Default)]
pub struct NetworkStateSnapshot {
    neuron_states: Vec<ContinuousHV>,
    layer_outputs: Vec<ContinuousHV>,
    /// Per-neuron (total_time, update_count). total_time is NOT telemetry:
    /// it is the phase clock of the Fourier equilibrium basis
    /// (compute_fourier_basis), i.e. genuine evolution state. Excluding it
    /// made snapshot/restore subtly impure — invisible while the bind-chain
    /// annihilation kept outputs at ~1e-10, exposed by the 2026-07-18 scale
    /// restoration (the purity twin test caught it at honest magnitudes).
    neuron_clocks: Vec<(f64, u64)>,
}

impl NetworkStateSnapshot {
    /// Dimension of the captured neuron-state HVs (None if never filled).
    /// Lets callers detect stale snapshots after an adaptive network resize.
    pub fn dimension(&self) -> Option<usize> {
        self.neuron_states.first().map(|hv| hv.values.len())
    }

    /// Element-wise approximate equality of two snapshots (states, layer
    /// outputs, and evolution clocks). The formal test surface for
    /// "this operation does not perturb evolution state".
    pub fn approx_eq(&self, other: &Self, eps: f32) -> bool {
        if self.neuron_states.len() != other.neuron_states.len()
            || self.layer_outputs.len() != other.layer_outputs.len()
            || self.neuron_clocks.len() != other.neuron_clocks.len()
        {
            return false;
        }
        let hv_eq = |a: &ContinuousHV, b: &ContinuousHV| {
            a.values.len() == b.values.len()
                && a.values
                    .iter()
                    .zip(b.values.iter())
                    .all(|(x, y)| (x - y).abs() <= eps)
        };
        self.neuron_states
            .iter()
            .zip(other.neuron_states.iter())
            .all(|(a, b)| hv_eq(a, b))
            && self
                .layer_outputs
                .iter()
                .zip(other.layer_outputs.iter())
                .all(|(a, b)| hv_eq(a, b))
            && self
                .neuron_clocks
                .iter()
                .zip(other.neuron_clocks.iter())
                .all(|(&(t1, n1), &(t2, n2))| (t1 - t2).abs() <= eps as f64 && n1 == n2)
    }
}

/// Copy one HV's values into another, reallocating only on dimension change.
#[inline]
fn copy_hv_into(src: &ContinuousHV, dst: &mut ContinuousHV) {
    if dst.values.len() == src.values.len() {
        dst.values.copy_from_slice(&src.values);
    } else {
        *dst = src.clone();
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

    /// Evolve each layer, calculate its local target gradient against the
    /// exact pre-evolution state of that prediction, then update its parameters.
    ///
    /// This is greedy/local supervised learning, not temporal BPTT or a full
    /// chain-rule gradient through the network. It is useful for controlled
    /// neuron-learning experiments where every layer receives the same target
    /// representation. The layer's prediction is produced before its update,
    /// and the post-evolution output is supplied to the next layer.
    pub fn train_local_target_step(
        &mut self,
        dt: f32,
        input: &ContinuousHV,
        target: &ContinuousHV,
        learning_rate: f32,
    ) -> ContinuousHV {
        for layer_idx in 0..self.layers.len() {
            let layer_input = if layer_idx == 0 {
                input.clone()
            } else {
                self.compute_layer_input(layer_idx, input)
            };

            for neuron in &mut self.layers[layer_idx] {
                neuron.evolve_closed_form(dt, &layer_input);
                let gradients = neuron.backward(&layer_input, target, dt);
                neuron.apply_gradients_no_decay(&gradients, learning_rate);
            }

            self.cache_layer_output(layer_idx);
        }

        self.output()
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

    /// Snapshot the network's mutable evolution state (neuron state HVs +
    /// cached layer outputs) into a reusable buffer, resizing it on first use.
    ///
    /// This exists to make prediction pure: snapshot → evolve at a horizon →
    /// read output → restore. It deliberately excludes learning parameters
    /// (weights, momentum) — those are only mutated by explicit training
    /// calls. Per-neuron evolution clocks (total_time/update_count) ARE
    /// captured: total_time is the Fourier equilibrium phase, i.e. genuine
    /// evolution state (a 2026-07-18 correction — the earlier "harmless
    /// telemetry drift" claim was false and broke purity at honest signal
    /// magnitudes).
    ///
    /// After the buffer warms up (same network shape), snapshot/restore are
    /// allocation-free element-wise copies.
    pub fn snapshot_state_into(&self, snap: &mut NetworkStateSnapshot) {
        let n_neurons: usize = self.layers.iter().map(|l| l.len()).sum();
        snap.neuron_states
            .resize_with(n_neurons, || ContinuousHV::zero(0));
        snap.layer_outputs
            .resize_with(self.layer_outputs.len(), || ContinuousHV::zero(0));

        snap.neuron_clocks.resize(n_neurons, (0.0, 0));
        let mut i = 0;
        for layer in &self.layers {
            for neuron in layer {
                copy_hv_into(neuron.state(), &mut snap.neuron_states[i]);
                snap.neuron_clocks[i] = neuron.evolution_clock();
                i += 1;
            }
        }
        for (src, dst) in self.layer_outputs.iter().zip(snap.layer_outputs.iter_mut()) {
            copy_hv_into(src, dst);
        }
    }

    /// Restore evolution state previously captured by [`snapshot_state_into`].
    ///
    /// Panics in debug builds if the snapshot shape does not match the network
    /// (wrong neuron count); in release the mismatched tail is left untouched.
    pub fn restore_state_from(&mut self, snap: &NetworkStateSnapshot) {
        debug_assert_eq!(
            snap.neuron_states.len(),
            self.layers.iter().map(|l| l.len()).sum::<usize>(),
            "NetworkStateSnapshot shape does not match network"
        );
        let mut i = 0;
        for layer in &mut self.layers {
            for neuron in layer {
                if let Some(saved) = snap.neuron_states.get(i) {
                    // Dimension guard (2026-07-17): after an adaptive resize a
                    // stale snapshot holds old-dimension HVs; cloning them into
                    // neuron state would poison it and panic the next bind().
                    // Skip mismatches — rolling snapshot queues self-heal
                    // within a couple of cycles.
                    if saved.values.len() == neuron.state().values.len() {
                        copy_hv_into(saved, neuron.state_mut());
                        if let Some(&(t, n)) = snap.neuron_clocks.get(i) {
                            neuron.set_evolution_clock(t, n);
                        }
                    }
                }
                i += 1;
            }
        }
        for (dst, src) in self.layer_outputs.iter_mut().zip(snap.layer_outputs.iter()) {
            if src.values.len() == dst.values.len() {
                copy_hv_into(src, dst);
            }
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
    fn test_zero_elapsed_time_is_identity_for_every_integrator() {
        let config = UnifiedConfig {
            dimension: 64,
            fourier_frequencies: vec![2.0, 5.0],
            fourier_amplitude: 0.03,
            ..UnifiedConfig::default()
        };
        let input = ContinuousHV::random(64, 601);
        let mut initial = HdcLtcUnifiedNeuron::new(config, 602);
        initial.set_state(ContinuousHV::random(64, 603).scale(0.1));
        let before = initial.state().clone();

        assert_eq!(initial.compute_gating(&input, 0.0), 0.0);

        let mut candidates = vec![
            initial.clone(),
            initial.clone(),
            initial.clone(),
            initial.clone(),
            initial.clone(),
            initial.clone(),
        ];
        candidates[0].evolve_closed_form(0.0, &input);
        candidates[1].evolve_closed_form_fused(0.0, &input);
        candidates[2].evolve_closed_form_exact(0.0, &input);
        candidates[3].evolve_closed_form_iterative(0.0, &input);
        candidates[4].evolve_rk4(0.0, &input);
        candidates[5].evolve(0.0, &input);

        for (index, candidate) in candidates.iter().enumerate() {
            let max_delta = candidate
                .state()
                .values
                .iter()
                .zip(before.values.iter())
                .map(|(after, before)| (after - before).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_delta < 1e-7,
                "integrator {index} changed the state at dt=0 by {max_delta}"
            );
        }
    }

    #[test]
    fn test_non_finite_and_negative_dt_are_sanitized_to_identity() {
        let mut neuron = HdcLtcUnifiedNeuron::new(
            UnifiedConfig {
                dimension: 64,
                ..UnifiedConfig::default()
            },
            604,
        );
        let input = ContinuousHV::random(64, 605);
        neuron.set_state(ContinuousHV::random(64, 606).scale(0.1));
        let before = neuron.state().clone();

        for dt in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1] {
            neuron.evolve_closed_form(dt, &input);
            assert!(
                neuron
                    .state()
                    .values
                    .iter()
                    .zip(before.values.iter())
                    .all(|(after, before)| (after - before).abs() < 1e-7),
                "invalid dt={dt} must not mutate state"
            );
        }
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
    fn test_parameter_profile_exposes_all_learnable_hypervectors() {
        let neuron = HdcLtcUnifiedNeuron::new(
            UnifiedConfig {
                dimension: 64,
                ..UnifiedConfig::default()
            },
            52,
        );
        let profile = neuron.parameter_profile();

        assert_eq!(profile.format_version, 1);
        assert_eq!(profile.dimension, 64);
        assert_eq!(profile.groups.len(), 5);
        assert_eq!(
            profile
                .groups
                .iter()
                .map(|group| group.id)
                .collect::<Vec<_>>(),
            vec![
                HdcLtcParameterGroupId::WeightHypervector,
                HdcLtcParameterGroupId::InputMask,
                HdcLtcParameterGroupId::TauModulator,
                HdcLtcParameterGroupId::GateWeight,
                HdcLtcParameterGroupId::GateBias,
            ]
        );
        assert!(
            profile
                .groups
                .iter()
                .all(|group| group.non_finite_count == 0)
        );
    }

    #[test]
    fn test_parameter_group_replacement_resets_associated_momentum() {
        let mut neuron = HdcLtcUnifiedNeuron::new(
            UnifiedConfig {
                dimension: 64,
                ..UnifiedConfig::default()
            },
            55,
        );
        neuron.weight_momentum = ContinuousHV::random(64, 551);
        neuron.input_momentum = ContinuousHV::random(64, 552);

        let weight = neuron.weight_hv_ref().values.clone();
        neuron
            .try_replace_parameter_group(HdcLtcParameterGroupId::WeightHypervector, &weight)
            .unwrap();
        assert_eq!(neuron.weight_momentum.norm(), 0.0);
        assert!(neuron.input_momentum.norm() > 0.0);

        let input = neuron.input_mask_ref().values.clone();
        neuron
            .try_replace_parameter_group(HdcLtcParameterGroupId::InputMask, &input)
            .unwrap();
        assert_eq!(neuron.input_momentum.norm(), 0.0);
    }

    #[test]
    fn test_parameter_group_window_is_bounded_and_read_only() {
        let neuron = HdcLtcUnifiedNeuron::new(
            UnifiedConfig {
                dimension: 64,
                ..UnifiedConfig::default()
            },
            54,
        );
        let id = HdcLtcParameterGroupId::WeightHypervector;
        let expected = &neuron.parameter_group(id).values[8..16];

        assert_eq!(neuron.parameter_group_window(id, 8, 8), Some(expected));
        assert!(neuron.parameter_group_window(id, 60, 8).is_none());
        assert!(neuron.parameter_group_window(id, usize::MAX, 8).is_none());
    }

    #[test]
    fn test_parameter_group_replacement_is_validated_before_mutation() {
        let mut neuron = HdcLtcUnifiedNeuron::new(
            UnifiedConfig {
                dimension: 64,
                ..UnifiedConfig::default()
            },
            53,
        );
        let before = neuron.weight_hv_ref().values.clone();

        let mut invalid = vec![0.0; 64];
        invalid[7] = f32::NAN;
        assert!(matches!(
            neuron
                .try_replace_parameter_group(HdcLtcParameterGroupId::WeightHypervector, &invalid,),
            Err(HdcLtcParameterEditError::NonFiniteValue { index: 7 })
        ));
        assert_eq!(neuron.weight_hv_ref().values, before);

        assert!(matches!(
            neuron.try_replace_parameter_group(
                HdcLtcParameterGroupId::WeightHypervector,
                &[0.0; 63],
            ),
            Err(HdcLtcParameterEditError::DimensionMismatch {
                expected: 64,
                actual: 63
            })
        ));
        assert_eq!(neuron.weight_hv_ref().values, before);

        let valid = vec![0.01; 64];
        assert!(
            neuron
                .try_replace_parameter_group(HdcLtcParameterGroupId::WeightHypervector, &valid,)
                .is_ok()
        );
        assert_eq!(neuron.weight_hv_ref().values, valid);
    }

    #[test]
    fn test_genesis_initialization_matches_unit_orthogonal_scale() {
        let genesis = crate::genesis::GenesisSeed::from_phrase("hdc-ltc-init-contract");
        let config = UnifiedConfig {
            dimension: 256,
            ..UnifiedConfig::default()
        };
        let neuron = HdcLtcUnifiedNeuron::from_genesis(config, &genesis, "test-neuron");
        let parameters = [
            neuron.weight_hv_ref(),
            neuron.input_mask_ref(),
            &neuron.tau_modulator,
            &neuron.gate_weight,
        ];

        for (index, parameter) in parameters.iter().enumerate() {
            let norm = parameter.norm();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "genesis parameter {index} should have unit norm, got {norm}"
            );
            for other in parameters.iter().skip(index + 1) {
                let similarity = parameter.similarity(other);
                assert!(
                    similarity.abs() < 0.02,
                    "genesis parameter vectors should be near-orthogonal, got {similarity}"
                );
            }
        }

        assert!(
            (neuron.gate_bias.norm() - 0.1).abs() < 1e-3,
            "gate bias should retain the 0.1 initialization scale"
        );
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
    fn test_backward_after_evolution_uses_retained_pre_state_and_fourier_phase() {
        let config = UnifiedConfig {
            dimension: 64,
            fourier_frequencies: vec![0.7, 2.3],
            fourier_amplitude: 0.05,
            ..UnifiedConfig::default()
        };
        let mut initial = HdcLtcUnifiedNeuron::new(config, 781);
        initial.set_state(ContinuousHV::random(64, 782).scale(0.08));
        initial.set_evolution_clock(0.137, 4);

        let input = ContinuousHV::random(64, 783).scale(0.15);
        let target = ContinuousHV::random(64, 784).scale(0.15);
        let dt = 0.013;

        // Reference gradient from the exact pre-forward state and Fourier phase.
        let expected = initial.backward_at_current(&input, &target, dt);

        // Production ordering: evolve, then evaluate the local output-space target.
        let mut after_forward = initial.clone();
        after_forward.evolve_closed_form(dt, &input);
        assert!((after_forward.pre_evolution_time - 0.137).abs() < 1e-9);
        assert_eq!(
            after_forward.pre_evolution_state.as_ref().unwrap().values,
            initial.state().values
        );

        let actual = after_forward.backward(&input, &target, dt);
        let max_diff = |left: &ContinuousHV, right: &ContinuousHV| {
            left.values
                .iter()
                .zip(right.values.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };

        assert!(max_diff(&actual.dw, &expected.dw) < 1e-6);
        assert!(max_diff(&actual.du, &expected.du) < 1e-6);
        assert!(max_diff(&actual.d_tau_modulator, &expected.d_tau_modulator) < 1e-6);
        assert!(max_diff(&actual.d_gate_weight, &expected.d_gate_weight) < 1e-6);
        assert!(max_diff(&actual.d_gate_bias, &expected.d_gate_bias) < 1e-6);
        assert!(max_diff(&actual.d_input, &expected.d_input) < 1e-6);
        assert!(max_diff(&actual.d_state, &expected.d_state) < 1e-6);
        assert!((actual.dtau_scalar - expected.dtau_scalar).abs() < 1e-6);
    }

    #[test]
    fn test_external_state_mutation_invalidates_retained_pre_state() {
        let config = UnifiedConfig {
            dimension: 64,
            ..UnifiedConfig::default()
        };
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 791);
        let input = ContinuousHV::random(64, 792);
        let target = ContinuousHV::random(64, 793);
        neuron.evolve_closed_form(0.01, &input);
        assert!(neuron.pre_evolution_state.is_some());

        neuron.state_mut().values[3] += 0.1;
        assert!(neuron.pre_evolution_state.is_none());

        let explicit = neuron.backward_at_current(&input, &target, 0.01);
        let fallback = neuron.backward(&input, &target, 0.01);
        assert!(
            explicit
                .dw
                .values
                .iter()
                .zip(fallback.dw.values.iter())
                .all(|(a, b)| (a - b).abs() < 1e-7)
        );
    }

    #[test]
    fn test_backward_weight_gradient_matches_forward_finite_difference() {
        let config = UnifiedConfig {
            dimension: 64,
            fourier_frequencies: Vec::new(),
            ..UnifiedConfig::default()
        };
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);
        let input = ContinuousHV::random(64, 100);
        let target = ContinuousHV::random(64, 200);
        neuron.set_state(ContinuousHV::random(64, 300).scale(0.1));

        let dt = 0.013;
        let gradients = neuron.backward(&input, &target, dt);
        let index = 7;
        let epsilon = 1e-3;

        let loss_at = |mut candidate: HdcLtcUnifiedNeuron| {
            candidate.evolve_closed_form(dt, &input);
            candidate
                .state()
                .values
                .iter()
                .zip(target.values.iter())
                .map(|(&predicted, &expected)| (predicted - expected).powi(2))
                .sum::<f32>()
                / 64.0
        };

        let mut plus = neuron.clone();
        plus.weight_hv.values[index] += epsilon;
        let mut minus = neuron.clone();
        minus.weight_hv.values[index] -= epsilon;
        let finite_difference = (loss_at(plus) - loss_at(minus)) / (2.0 * epsilon);

        assert!(
            (gradients.dw.values[index] - finite_difference).abs() < 2e-3,
            "weight gradient mismatch: analytic={}, finite_difference={}",
            gradients.dw.values[index],
            finite_difference
        );
    }

    #[test]
    fn test_backward_weight_gradient_matches_forward_with_fourier_basis() {
        let config = UnifiedConfig {
            dimension: 64,
            fourier_frequencies: vec![1.0, 3.0],
            fourier_amplitude: 0.07,
            ..UnifiedConfig::default()
        };
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 73);
        let input = ContinuousHV::random(64, 101).scale(0.15);
        let target = ContinuousHV::random(64, 202).scale(0.15);
        neuron.set_state(ContinuousHV::random(64, 303).scale(0.08));

        let dt = 0.011;
        let epsilon = 1e-3;
        let index = 13;
        let original = neuron.clone();
        let mut forward = original.clone();
        forward.evolve_closed_form(dt, &input);
        let gradients = forward.backward(&input, &target, dt);

        let loss_at = |mut candidate: HdcLtcUnifiedNeuron| {
            candidate.evolve_closed_form(dt, &input);
            candidate
                .state()
                .values
                .iter()
                .zip(target.values.iter())
                .map(|(&predicted, &expected)| (predicted - expected).powi(2))
                .sum::<f32>()
                / 64.0
        };

        let mut plus = original.clone();
        plus.weight_hv.values[index] += epsilon;
        let mut minus = original.clone();
        minus.weight_hv.values[index] -= epsilon;
        let finite_difference = (loss_at(plus) - loss_at(minus)) / (2.0 * epsilon);

        assert!(
            (gradients.dw.values[index] - finite_difference).abs() < 2e-3,
            "Fourier-bundle weight gradient mismatch: analytic={}, finite_difference={}",
            gradients.dw.values[index],
            finite_difference
        );
    }

    #[test]
    fn test_adaptive_gate_and_tau_gradients_match_finite_difference() {
        let config = UnifiedConfig {
            dimension: 64,
            fourier_frequencies: Vec::new(),
            ..UnifiedConfig::default()
        };
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 91);
        let input = ContinuousHV::random(64, 111).scale(0.2);
        let target = ContinuousHV::random(64, 222).scale(0.2);
        neuron.set_state(ContinuousHV::random(64, 333).scale(0.1));

        let dt = 0.017;
        let epsilon = 5e-3;
        let index = 9;
        let mut forward = neuron.clone();
        forward.evolve_closed_form(dt, &input);
        let gradients = forward.backward(&input, &target, dt);

        fn loss(
            original: &HdcLtcUnifiedNeuron,
            input: &ContinuousHV,
            target: &ContinuousHV,
            dt: f32,
        ) -> f32 {
            let mut candidate = original.clone();
            candidate.evolve_closed_form(dt, input);
            candidate
                .state()
                .values
                .iter()
                .zip(target.values.iter())
                .map(|(&predicted, &expected)| (predicted - expected).powi(2))
                .sum::<f32>()
                / target.dim() as f32
        }

        let finite_difference = |plus: HdcLtcUnifiedNeuron, minus: HdcLtcUnifiedNeuron| {
            (loss(&plus, &input, &target, dt) - loss(&minus, &input, &target, dt)) / (2.0 * epsilon)
        };

        let mut plus = neuron.clone();
        plus.tau_modulator.values[index] += epsilon;
        let mut minus = neuron.clone();
        minus.tau_modulator.values[index] -= epsilon;
        let tau_gradient_fd = finite_difference(plus, minus);
        assert!(
            (gradients.d_tau_modulator.values[index] - tau_gradient_fd).abs() < 2e-3,
            "tau-modulator gradient mismatch: analytic={}, finite_difference={}",
            gradients.d_tau_modulator.values[index],
            tau_gradient_fd
        );

        let mut plus = neuron.clone();
        plus.gate_weight.values[index] += epsilon;
        let mut minus = neuron.clone();
        minus.gate_weight.values[index] -= epsilon;
        let gate_gradient_fd = finite_difference(plus, minus);
        assert!(
            (gradients.d_gate_weight.values[index] - gate_gradient_fd).abs() < 2e-3,
            "gate-weight gradient mismatch: analytic={}, finite_difference={}",
            gradients.d_gate_weight.values[index],
            gate_gradient_fd
        );

        let mut plus = neuron.clone();
        plus.gate_bias.values[index] += epsilon;
        let mut minus = neuron.clone();
        minus.gate_bias.values[index] -= epsilon;
        let gate_bias_gradient_fd = finite_difference(plus, minus);
        assert!(
            (gradients.d_gate_bias.values[index] - gate_bias_gradient_fd).abs() < 2e-3,
            "gate-bias gradient mismatch: analytic={}, finite_difference={}",
            gradients.d_gate_bias.values[index],
            gate_bias_gradient_fd
        );

        let mut input_plus = input.clone();
        input_plus.values[index] += epsilon;
        let mut input_minus = input.clone();
        input_minus.values[index] -= epsilon;
        let input_gradient_fd = (loss(&neuron, &input_plus, &target, dt)
            - loss(&neuron, &input_minus, &target, dt))
            / (2.0 * epsilon);
        assert!(
            (gradients.d_input.values[index] - input_gradient_fd).abs() < 2e-3,
            "input gradient mismatch: analytic={}, finite_difference={}",
            gradients.d_input.values[index],
            input_gradient_fd
        );

        let mut state_plus = neuron.clone();
        let mut state_plus_hv = neuron.state().clone();
        state_plus_hv.values[index] += epsilon;
        state_plus.set_state(state_plus_hv);
        let mut state_minus = neuron.clone();
        let mut state_minus_hv = neuron.state().clone();
        state_minus_hv.values[index] -= epsilon;
        state_minus.set_state(state_minus_hv);
        let state_gradient_fd = finite_difference(state_plus, state_minus);
        assert!(
            (gradients.d_state.values[index] - state_gradient_fd).abs() < 2e-3,
            "state gradient mismatch: analytic={}, finite_difference={}",
            gradients.d_state.values[index],
            state_gradient_fd
        );
    }

    #[test]
    fn test_backward_after_forward_uses_retained_pre_evolution_state() {
        let config = UnifiedConfig {
            dimension: 64,
            fourier_frequencies: vec![1.0, 3.0],
            fourier_amplitude: 0.07,
            ..UnifiedConfig::default()
        };
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 190);
        let input = ContinuousHV::random(64, 191).scale(0.12);
        let target = ContinuousHV::random(64, 192).scale(0.12);
        neuron.set_state(ContinuousHV::random(64, 193).scale(0.09));

        let dt = 0.013;
        let epsilon = 1e-3;
        let index = 17;
        let pre_forward = neuron.clone();
        let pre_state = neuron.state().clone();
        let pre_time = neuron.total_time();

        neuron.evolve_closed_form(dt, &input);
        let post_state = neuron.state().clone();
        let gradients = neuron.backward(&input, &target, dt);

        let loss = |mut candidate: HdcLtcUnifiedNeuron| {
            candidate.evolve_closed_form(dt, &input);
            candidate
                .state()
                .values
                .iter()
                .zip(target.values.iter())
                .map(|(&prediction, &expected)| (prediction - expected).powi(2))
                .sum::<f32>()
                / 64.0
        };

        let mut plus = pre_forward.clone();
        plus.weight_hv.values[index] += epsilon;
        let mut minus = pre_forward.clone();
        minus.weight_hv.values[index] -= epsilon;
        let finite_difference = (loss(plus) - loss(minus)) / (2.0 * epsilon);

        // The retained pre-step state is required; using the live post-state
        // would produce a different derivative, especially with Fourier input.
        assert!(pre_state.similarity(&post_state) < 0.999_999);
        assert!(pre_time < neuron.total_time());
        assert!(
            (gradients.dw.values[index] - finite_difference).abs() < 2e-3,
            "post-forward weight gradient mismatch: analytic={}, finite_difference={}",
            gradients.dw.values[index],
            finite_difference
        );
    }

    #[test]
    fn test_backward_snapshot_is_invalidated_by_parameter_or_integrator_change() {
        let mut neuron = HdcLtcUnifiedNeuron::new(
            UnifiedConfig {
                dimension: 64,
                ..UnifiedConfig::default()
            },
            194,
        );
        let input = ContinuousHV::random(64, 195);
        let target = ContinuousHV::random(64, 196);

        neuron.evolve_closed_form(0.01, &input);
        assert!(neuron.pre_evolution_state.is_some());

        neuron.set_tau_base(0.04);
        assert!(neuron.pre_evolution_state.is_none());

        neuron.evolve_closed_form(0.01, &input);
        assert!(neuron.pre_evolution_state.is_some());

        neuron.evolve_rk4(0.01, &input);
        assert!(neuron.pre_evolution_state.is_none());

        let gradients = neuron.backward(&input, &target, 0.01);
        assert!(gradients.dw.values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_local_target_step_updates_parameters_and_evolves() {
        let config = UnifiedConfig {
            dimension: 64,
            fourier_frequencies: Vec::new(),
            ..UnifiedConfig::default()
        };
        let network_config = UnifiedNetworkConfig {
            layer_sizes: vec![2, 2],
            neuron_config: config,
            use_layer_binding: true,
            skip_connections: false,
        };
        let mut network = HdcLtcUnifiedNetwork::new(network_config, 42);
        let input = ContinuousHV::random(64, 100);
        let target = ContinuousHV::random(64, 200);
        let before = network.layers[0][0].weight_hv.clone();

        let output = network.train_local_target_step(0.01, &input, &target, 0.05);

        assert!(output.values.iter().all(|value| value.is_finite()));
        assert_ne!(network.layers[0][0].weight_hv.values, before.values);
        assert_eq!(network.layers[0][0].update_count, 1);
        assert_eq!(network.layers[1][0].update_count, 1);
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
