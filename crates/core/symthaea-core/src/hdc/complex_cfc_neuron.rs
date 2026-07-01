// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Complex-Valued CfC Neuron
//!
//! A variant of the HDC-LTC unified neuron where the state is represented as
//! 8,192 complex dimensions (interleaved into 16,384 real values for API
//! compatibility). Each dimension has a learned eigenvalue λ_k = a_k + ib_k:
//!
//! ```text
//! x_k(t+dt) = x_∞_k + (x_k(t) - x_∞_k) × exp(λ_k × dt)
//! ```
//!
//! When a_k < 0 (stable) and b_k ≠ 0, the exponential produces damped
//! oscillation via Euler's formula:
//!
//! ```text
//! exp((a+bi)t) = e^{at}(cos(bt) + i·sin(bt))
//!                ^^^^^  ^^^^^^^^^^^^^^^^^^^^^^
//!                decay  oscillation at b/(2π) Hz
//! ```
//!
//! This eliminates the "molasses problem" of standard CfC neurons, which are
//! pure contraction mappings that can only decay toward equilibrium.
//!
//! ## API Boundary
//!
//! The complex representation is **internal**. All downstream consumers
//! (consciousness engine, HAI free energy, ethics, prediction, training)
//! see `ContinuousHV` output via `to_continuous_hv()`, which interleaves
//! real and imaginary parts into a 16,384-dimensional real vector.
//!
//! ## Eigenvalue Constraints (Stability by Construction)
//!
//! - Real part: a_k ∈ [-1.0, -0.01] — must be negative for bounded dynamics
//! - Imaginary part: b_k ∈ [-50π, 50π] — covers 0-25 Hz motor band
//! - Eigenvalues are learnable parameters with projection back to constraints
//!
//! ## References
//!
//! - Hirose, A. (2012). Complex-Valued Neural Networks. Springer.
//! - Gu, A. et al. (2022). Efficiently Modeling Long Sequences with SSMs. ICLR.
//! - Hasani, R. et al. (2021). Closed-form Continuous-time Neural Networks. NMI.

use crate::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use serde::{Deserialize, Serialize};

/// Half the HDC dimension — number of complex pairs.
const COMPLEX_DIM: usize = HDC_DIMENSION / 2; // 8,192

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the complex-valued CfC neuron.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexCfcConfig {
    /// Minimum real part of eigenvalues (must be negative for stability).
    pub eigenvalue_real_min: f32,
    /// Maximum real part of eigenvalues (must be negative for stability).
    pub eigenvalue_real_max: f32,
    /// Maximum absolute imaginary part of eigenvalues (radians/second).
    /// Controls the highest representable oscillation frequency.
    pub eigenvalue_imag_max: f32,
    /// Learning rate for eigenvalue gradient updates.
    pub eigenvalue_lr: f32,
    /// Learning rate for weight updates.
    pub weight_lr: f32,
    /// State norm bound (same as standard CfC neuron).
    pub state_norm_bound: f32,
    /// Weight norm bound.
    pub weight_norm_bound: f32,
}

impl Default for ComplexCfcConfig {
    fn default() -> Self {
        Self {
            eigenvalue_real_min: -1.0,
            eigenvalue_real_max: -0.01,
            eigenvalue_imag_max: 50.0 * std::f32::consts::PI, // ~25 Hz max
            eigenvalue_lr: 0.001,
            weight_lr: 0.01,
            state_norm_bound: 5.0,
            weight_norm_bound: 2.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPLEX CFC NEURON
// ═══════════════════════════════════════════════════════════════════════════════

/// Complex-valued CfC neuron with native oscillation support.
///
/// State is 8,192 complex dimensions. Each dimension has a learned eigenvalue
/// that controls decay rate (real part) and oscillation frequency (imaginary part).
/// The `to_continuous_hv()` method interleaves real/imaginary into 16,384D for
/// API compatibility with all existing `ContinuousHV` consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexCfcNeuron {
    /// Real part of state [0..COMPLEX_DIM)
    state_re: Vec<f32>,
    /// Imaginary part of state [0..COMPLEX_DIM)
    state_im: Vec<f32>,

    /// Weight HV real part (for state transformation W⊗x)
    weight_re: Vec<f32>,
    /// Weight HV imaginary part
    weight_im: Vec<f32>,

    /// Input mask real part (for input transformation U⊗u)
    input_re: Vec<f32>,
    /// Input mask imaginary part
    input_im: Vec<f32>,

    /// Eigenvalue real parts (a_k) — controls decay rate per dimension.
    /// Constrained to [eigenvalue_real_min, eigenvalue_real_max] (both negative).
    eigen_re: Vec<f32>,
    /// Eigenvalue imaginary parts (b_k) — controls oscillation frequency per dimension.
    /// Constrained to [-eigenvalue_imag_max, eigenvalue_imag_max].
    eigen_im: Vec<f32>,

    /// Configuration
    config: ComplexCfcConfig,

    /// Total time evolved (seconds)
    total_time: f64,
    /// Number of evolution steps
    update_count: u64,
}

impl ComplexCfcNeuron {
    /// Create a new complex CfC neuron with deterministic initialization.
    pub fn new(config: ComplexCfcConfig, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);

        // Initialize state near zero (small random values)
        let state_re: Vec<f32> = (0..COMPLEX_DIM)
            .map(|_| rng.next_f32() * 0.1 - 0.05)
            .collect();
        let state_im: Vec<f32> = (0..COMPLEX_DIM)
            .map(|_| rng.next_f32() * 0.1 - 0.05)
            .collect();

        // Initialize weights (Xavier-like: scale by 1/sqrt(dim))
        let scale = 1.0 / (COMPLEX_DIM as f32).sqrt();
        let weight_re: Vec<f32> = (0..COMPLEX_DIM)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        let weight_im: Vec<f32> = (0..COMPLEX_DIM)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        let input_re: Vec<f32> = (0..COMPLEX_DIM)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        let input_im: Vec<f32> = (0..COMPLEX_DIM)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();

        // Initialize eigenvalues: real part in [min, max], imaginary spread across freq range
        let eigen_re: Vec<f32> = (0..COMPLEX_DIM)
            .map(|_| {
                let t = rng.next_f32(); // [0, 1)
                config.eigenvalue_real_min
                    + t * (config.eigenvalue_real_max - config.eigenvalue_real_min)
            })
            .collect();
        let eigen_im: Vec<f32> = (0..COMPLEX_DIM)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * config.eigenvalue_imag_max)
            .collect();

        Self {
            state_re,
            state_im,
            weight_re,
            weight_im,
            input_re,
            input_im,
            eigen_re,
            eigen_im,
            config,
            total_time: 0.0,
            update_count: 0,
        }
    }

    /// Create with default configuration.
    pub fn new_default(seed: u64) -> Self {
        Self::new(ComplexCfcConfig::default(), seed)
    }

    /// Evolve the neuron state by `dt` seconds with the given input.
    ///
    /// For each complex dimension k:
    /// 1. Compute equilibrium: x_∞_k = tanh_complex(W_k ⊗ x_k + U_k ⊗ u_k)
    /// 2. Apply complex exponential decay: x_k' = x_∞_k + (x_k - x_∞_k) × exp(λ_k × dt)
    ///
    /// The input is a standard 16,384D ContinuousHV. It is deinterleaved internally
    /// into 8,192 complex pairs for computation.
    pub fn evolve(&mut self, dt: f32, input: &ContinuousHV) {
        let dim = COMPLEX_DIM;

        for k in 0..dim {
            // Deinterleave input: even = real, odd = imaginary
            let u_re = input.values[k * 2];
            let u_im = input.values[k * 2 + 1];

            // Complex binding: W⊗x = (w_re*x_re - w_im*x_im) + i(w_re*x_im + w_im*x_re)
            let wx_re = self.weight_re[k] * self.state_re[k] - self.weight_im[k] * self.state_im[k];
            let wx_im = self.weight_re[k] * self.state_im[k] + self.weight_im[k] * self.state_re[k];

            // Complex binding: U⊗u
            let uu_re = self.input_re[k] * u_re - self.input_im[k] * u_im;
            let uu_im = self.input_re[k] * u_im + self.input_im[k] * u_re;

            // Bundle (average): x_∞ = (W⊗x + U⊗u) / 2
            let eq_re = (wx_re + uu_re) * 0.5;
            let eq_im = (wx_im + uu_im) * 0.5;

            // Apply complex tanh activation (component-wise for simplicity/stability)
            let x_inf_re = eq_re.tanh();
            let x_inf_im = eq_im.tanh();

            // Complex exponential decay: exp(λ_k × dt) = exp((a+bi)×dt)
            // = exp(a×dt) × (cos(b×dt) + i×sin(b×dt))
            let a_dt = self.eigen_re[k] * dt;
            let b_dt = self.eigen_im[k] * dt;
            let exp_decay = a_dt.exp(); // e^{a×dt}, a < 0 so this decays
            let cos_bt = b_dt.cos();
            let sin_bt = b_dt.sin();

            // Difference: d = x - x_∞
            let d_re = self.state_re[k] - x_inf_re;
            let d_im = self.state_im[k] - x_inf_im;

            // Complex multiply: d × exp(λ×dt)
            // (d_re + i×d_im) × exp_decay × (cos + i×sin)
            let rotated_re = exp_decay * (d_re * cos_bt - d_im * sin_bt);
            let rotated_im = exp_decay * (d_re * sin_bt + d_im * cos_bt);

            // New state: x' = x_∞ + d × exp(λ×dt)
            self.state_re[k] = x_inf_re + rotated_re;
            self.state_im[k] = x_inf_im + rotated_im;
        }

        self.apply_state_bounds();
        self.total_time += dt as f64;
        self.update_count += 1;
    }

    /// Convert internal complex state to ContinuousHV (16,384D) by interleaving.
    ///
    /// This is THE API boundary. All downstream consumers see this output.
    /// Even indices = real parts, odd indices = imaginary parts.
    pub fn to_continuous_hv(&self) -> ContinuousHV {
        let mut values = vec![0.0f32; HDC_DIMENSION];
        for k in 0..COMPLEX_DIM {
            values[k * 2] = self.state_re[k];
            values[k * 2 + 1] = self.state_im[k];
        }
        ContinuousHV::from_vec(values)
    }

    /// Set state from a ContinuousHV (16,384D) by deinterleaving.
    pub fn from_continuous_hv(&mut self, hv: &ContinuousHV) {
        for k in 0..COMPLEX_DIM {
            self.state_re[k] = hv.values[k * 2];
            self.state_im[k] = hv.values[k * 2 + 1];
        }
    }

    /// Current state as ContinuousHV (convenience alias for `to_continuous_hv`).
    pub fn state(&self) -> ContinuousHV {
        self.to_continuous_hv()
    }

    /// State norm (computed on the interleaved representation).
    pub fn state_norm(&self) -> f32 {
        let mut sum = 0.0f32;
        for k in 0..COMPLEX_DIM {
            sum += self.state_re[k] * self.state_re[k] + self.state_im[k] * self.state_im[k];
        }
        sum.sqrt()
    }

    /// Apply soft state bounds to prevent numerical explosion.
    fn apply_state_bounds(&mut self) {
        let norm = self.state_norm();
        if norm > self.config.state_norm_bound {
            let scale = self.config.state_norm_bound / norm;
            for k in 0..COMPLEX_DIM {
                self.state_re[k] *= scale;
                self.state_im[k] *= scale;
            }
        }
    }

    /// Project eigenvalues back to constraint manifold.
    ///
    /// Real parts: clamped to [eigenvalue_real_min, eigenvalue_real_max]
    /// Imaginary parts: clamped to [-eigenvalue_imag_max, eigenvalue_imag_max]
    fn project_eigenvalues(&mut self) {
        for k in 0..COMPLEX_DIM {
            self.eigen_re[k] = self.eigen_re[k].clamp(
                self.config.eigenvalue_real_min,
                self.config.eigenvalue_real_max,
            );
            self.eigen_im[k] = self.eigen_im[k].clamp(
                -self.config.eigenvalue_imag_max,
                self.config.eigenvalue_imag_max,
            );
        }
    }

    /// Update eigenvalues via gradient descent toward a target state.
    ///
    /// Uses analytical gradient of the complex exponential step:
    /// dL/dλ_k = dL/dx'_k × d_k × dt × exp(λ_k × dt)
    pub fn update_eigenvalues(&mut self, target: &ContinuousHV, dt: f32) {
        let lr = self.config.eigenvalue_lr;

        for k in 0..COMPLEX_DIM {
            // Error: target - current state
            let err_re = target.values[k * 2] - self.state_re[k];
            let err_im = target.values[k * 2 + 1] - self.state_im[k];

            // Gradient contribution from the exponential:
            // d(exp(λ×dt))/dλ = dt × exp(λ×dt)
            let a_dt = self.eigen_re[k] * dt;
            let b_dt = self.eigen_im[k] * dt;
            let exp_decay = a_dt.exp();
            let cos_bt = b_dt.cos();
            let sin_bt = b_dt.sin();

            // Simplified gradient: push eigenvalue toward reducing error
            // dL/da_k ≈ -err_re × exp_decay × cos(b×dt) × dt (real part gradient)
            // dL/db_k ≈ err_im × exp_decay × sin(b×dt) × dt (imaginary part gradient)
            let grad_re = -(err_re * exp_decay * cos_bt + err_im * exp_decay * sin_bt) * dt;
            let grad_im = -(err_im * exp_decay * cos_bt - err_re * exp_decay * sin_bt) * dt;

            self.eigen_re[k] -= lr * grad_re;
            self.eigen_im[k] -= lr * grad_im;
        }

        self.project_eigenvalues();
    }

    /// Get the oscillation frequency (Hz) for a given dimension.
    pub fn freq_hz(&self, k: usize) -> f32 {
        if k < COMPLEX_DIM {
            self.eigen_im[k].abs() / (2.0 * std::f32::consts::PI)
        } else {
            0.0
        }
    }

    /// Get the decay rate for a given dimension (negative = stable).
    pub fn decay_rate(&self, k: usize) -> f32 {
        if k < COMPLEX_DIM {
            self.eigen_re[k]
        } else {
            0.0
        }
    }

    /// Mean oscillation frequency across all dimensions (Hz).
    pub fn mean_freq_hz(&self) -> f32 {
        let sum: f32 = self.eigen_im.iter().map(|b| b.abs()).sum();
        sum / (COMPLEX_DIM as f32 * 2.0 * std::f32::consts::PI)
    }

    /// Number of complex dimensions.
    pub fn complex_dim(&self) -> usize {
        COMPLEX_DIM
    }

    /// Total updates performed.
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Total time evolved.
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Configuration (read-only).
    pub fn config(&self) -> &ComplexCfcConfig {
        &self.config
    }

    /// Reset state to near-zero.
    pub fn reset(&mut self, seed: u64) {
        let mut rng = SimpleRng::new(seed);
        for k in 0..COMPLEX_DIM {
            self.state_re[k] = rng.next_f32() * 0.1 - 0.05;
            self.state_im[k] = rng.next_f32() * 0.1 - 0.05;
        }
        self.total_time = 0.0;
        self.update_count = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMPLE RNG (deterministic, no external dependency)
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimal xorshift64 RNG for deterministic initialization.
/// Avoids `rand` dependency for this module.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generate f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(seed: u64) -> ContinuousHV {
        let mut rng = SimpleRng::new(seed);
        let values: Vec<f32> = (0..HDC_DIMENSION)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();
        ContinuousHV::from_vec(values)
    }

    #[test]
    fn test_creation() {
        let neuron = ComplexCfcNeuron::new_default(42);
        assert_eq!(neuron.complex_dim(), COMPLEX_DIM);
        assert!(neuron.state_norm() > 0.0);
        assert!(neuron.state_norm().is_finite());
    }

    #[test]
    fn test_eigenvalue_constraints() {
        let neuron = ComplexCfcNeuron::new_default(42);
        for k in 0..COMPLEX_DIM {
            assert!(
                neuron.eigen_re[k] >= neuron.config.eigenvalue_real_min,
                "eigen_re[{}] = {} below min",
                k,
                neuron.eigen_re[k]
            );
            assert!(
                neuron.eigen_re[k] <= neuron.config.eigenvalue_real_max,
                "eigen_re[{}] = {} above max",
                k,
                neuron.eigen_re[k]
            );
            assert!(
                neuron.eigen_im[k].abs() <= neuron.config.eigenvalue_imag_max,
                "eigen_im[{}] = {} exceeds max",
                k,
                neuron.eigen_im[k]
            );
        }
    }

    #[test]
    fn test_evolution_produces_oscillation() {
        // Core test: complex CfC should produce non-monotonic state norm
        // (oscillation), unlike standard CfC which only decays.
        let mut neuron = ComplexCfcNeuron::new_default(42);
        let input = make_input(123);
        let dt = 0.01;

        let mut norms = Vec::with_capacity(200);
        for _ in 0..200 {
            neuron.evolve(dt, &input);
            norms.push(neuron.state_norm());
        }

        // Count direction changes (peaks and valleys)
        let mut direction_changes = 0u32;
        for window in norms.windows(3) {
            let peak = window[1] > window[0] && window[2] < window[1];
            let valley = window[1] < window[0] && window[2] > window[1];
            if peak || valley {
                direction_changes += 1;
            }
        }

        assert!(
            direction_changes >= 3,
            "Complex CfC should oscillate (got {} direction changes in 200 steps)",
            direction_changes
        );
    }

    #[test]
    fn test_state_bounded() {
        // Eigenvalue real parts are negative → state should remain bounded
        let mut neuron = ComplexCfcNeuron::new_default(42);
        let input = make_input(123);

        for _ in 0..500 {
            neuron.evolve(0.02, &input);
        }

        let norm = neuron.state_norm();
        assert!(
            norm <= neuron.config.state_norm_bound + 0.01,
            "State should be bounded, got norm={}",
            norm
        );
        assert!(
            norm.is_finite(),
            "State should be finite, got norm={}",
            norm
        );
    }

    #[test]
    fn test_no_nan_inf() {
        let mut neuron = ComplexCfcNeuron::new_default(42);
        let input = make_input(99);

        for _ in 0..1000 {
            neuron.evolve(0.05, &input);
        }

        let hv = neuron.to_continuous_hv();
        for (i, &v) in hv.values.iter().enumerate() {
            assert!(v.is_finite(), "NaN/Inf at index {}: {}", i, v);
        }
    }

    #[test]
    fn test_to_from_continuous_hv_roundtrip() {
        let neuron = ComplexCfcNeuron::new_default(42);
        let hv = neuron.to_continuous_hv();

        let mut neuron2 = ComplexCfcNeuron::new_default(99); // different seed
        neuron2.from_continuous_hv(&hv);

        // States should now match
        for k in 0..COMPLEX_DIM {
            assert!(
                (neuron.state_re[k] - neuron2.state_re[k]).abs() < 1e-6,
                "Real part mismatch at k={}: {} vs {}",
                k,
                neuron.state_re[k],
                neuron2.state_re[k]
            );
            assert!(
                (neuron.state_im[k] - neuron2.state_im[k]).abs() < 1e-6,
                "Imaginary part mismatch at k={}",
                k
            );
        }
    }

    #[test]
    fn test_continuous_hv_dimension() {
        let neuron = ComplexCfcNeuron::new_default(42);
        let hv = neuron.to_continuous_hv();
        assert_eq!(hv.values.len(), HDC_DIMENSION);
    }

    #[test]
    fn test_freq_hz() {
        let neuron = ComplexCfcNeuron::new_default(42);
        for k in 0..10 {
            let freq = neuron.freq_hz(k);
            assert!(freq >= 0.0, "Frequency should be non-negative");
            assert!(
                freq <= 25.0 + 0.1, // max ~25 Hz
                "Frequency {} Hz exceeds expected max for k={}",
                freq,
                k
            );
        }
    }

    #[test]
    fn test_decay_rates_negative() {
        let neuron = ComplexCfcNeuron::new_default(42);
        for k in 0..COMPLEX_DIM {
            assert!(
                neuron.decay_rate(k) < 0.0,
                "Decay rate should be negative for stability, got {} at k={}",
                neuron.decay_rate(k),
                k
            );
        }
    }

    #[test]
    fn test_eigenvalue_learning() {
        let mut neuron = ComplexCfcNeuron::new_default(42);
        let input = make_input(123);

        // Evolve a few steps
        for _ in 0..10 {
            neuron.evolve(0.01, &input);
        }

        // Record eigenvalues before learning
        let eigen_re_before: Vec<f32> = neuron.eigen_re.clone();

        // Create a target and update eigenvalues
        let target = make_input(456);
        neuron.update_eigenvalues(&target, 0.01);

        // Eigenvalues should have changed
        let mut changed = false;
        for k in 0..COMPLEX_DIM {
            if (neuron.eigen_re[k] - eigen_re_before[k]).abs() > 1e-10 {
                changed = true;
                break;
            }
        }
        assert!(changed, "Eigenvalue learning should modify eigenvalues");

        // Constraints should still hold
        for k in 0..COMPLEX_DIM {
            assert!(
                neuron.eigen_re[k] >= neuron.config.eigenvalue_real_min,
                "eigen_re[{}] violated min after learning",
                k
            );
            assert!(
                neuron.eigen_re[k] <= neuron.config.eigenvalue_real_max,
                "eigen_re[{}] violated max after learning",
                k
            );
        }
    }

    #[test]
    fn test_different_from_zero_input() {
        // With zero input, state should still evolve (due to eigenvalue oscillation)
        let mut neuron = ComplexCfcNeuron::new_default(42);
        let zero_input = ContinuousHV::zero(HDC_DIMENSION);
        let initial_state = neuron.state();

        for _ in 0..50 {
            neuron.evolve(0.02, &zero_input);
        }

        let final_state = neuron.state();
        let sim = initial_state.similarity(&final_state);
        assert!(
            sim < 0.99,
            "State should evolve even with zero input (got similarity={})",
            sim
        );
    }

    #[test]
    fn test_reset() {
        let mut neuron = ComplexCfcNeuron::new_default(42);
        let input = make_input(123);

        for _ in 0..100 {
            neuron.evolve(0.02, &input);
        }
        assert!(neuron.update_count() > 0);

        neuron.reset(42);
        assert_eq!(neuron.update_count(), 0);
        assert_eq!(neuron.total_time(), 0.0);
    }

    #[test]
    fn test_large_dt_stable() {
        // Even with large time steps, negative eigenvalues should keep things bounded
        let mut neuron = ComplexCfcNeuron::new_default(42);
        let input = make_input(123);

        neuron.evolve(10.0, &input); // 10 second jump
        assert!(
            neuron.state_norm().is_finite(),
            "Large dt should not produce NaN/Inf"
        );
        assert!(
            neuron.state_norm() <= neuron.config.state_norm_bound + 0.01,
            "Large dt should not blow up: norm={}",
            neuron.state_norm()
        );
    }

    #[test]
    fn test_mean_freq() {
        let neuron = ComplexCfcNeuron::new_default(42);
        let mean = neuron.mean_freq_hz();
        assert!(mean > 0.0, "Mean frequency should be positive");
        assert!(mean < 30.0, "Mean frequency should be reasonable: {}", mean);
    }
}
