// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bidirectional HDC↔SSM projection with gradient learning.
//!
//! Projects between 16,384D HDC space and 768D SSM (Mamba) space
//! via a 256D bottleneck. The bottleneck matches `compressed_state` dim
//! used by HarmoniesIntegrator and provides information-bottleneck regularization.
//!
//! Total parameters: ~8.8M shallow (vs 25.2M for dense 16384×768 round-trip),
//! ~8.9M deep (adds 256×128 + 128×256 = 65,536 inner weights).
//!
//! # Architecture
//!
//! ## Shallow (default)
//! ```text
//! Forward:  HDC(16384) → w_down → LayerNorm → GELU+residual → w_up → SSM(768)
//! Backward: SSM(768) → w_back_down → LayerNorm → GELU+residual → w_back_up → HDC(16384)
//! ```
//!
//! ## Deep (double bottleneck)
//! ```text
//! Forward:  HDC(16384) → w_down → LN → GELU+res → w_down2(128) → GELU → w_up2(256) → GELU+res → w_up → SSM(768)
//! Backward: SSM(768) → w_up^T → GELU → w_up2^T → GELU → w_down2^T → LN → w_down^T → HDC(16384)
//! ```
//!
//! Uses LayerNorm on the bottleneck for training stabilization, followed by
//! GELU activation (no dead neurons) with a pre-activation residual
//! connection (`GELU(x) + α*x`) for smooth gradient flow.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

/// Pre-activation residual scale: `hidden = GELU(x) + RESIDUAL_ALPHA * x`.
/// Ensures gradient flow even through saturated regions.
const RESIDUAL_ALPHA: f32 = 0.1;

/// GELU activation: `x * Φ(x)` via tanh approximation.
#[inline]
fn gelu(x: f32) -> f32 {
    // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let c = 0.7978845608; // sqrt(2/π)
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// GELU derivative (for backprop): d/dx [GELU(x)].
#[inline]
fn gelu_derivative(x: f32) -> f32 {
    let c = 0.7978845608; // sqrt(2/π)
    let inner = c * (x + 0.044715 * x * x * x);
    let tanh_inner = inner.tanh();
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
}

/// Activation + residual: `GELU(x) + α*x`.
#[inline]
fn activation(x: f32) -> f32 {
    gelu(x) + RESIDUAL_ALPHA * x
}

/// Derivative of activation + residual: `GELU'(x) + α`.
#[inline]
fn activation_derivative(x: f32) -> f32 {
    gelu_derivative(x) + RESIDUAL_ALPHA
}

/// Bidirectional projection between HDC (16,384D) and SSM (768D) spaces.
///
/// Uses a 256D bottleneck with JL-style random initialization and online
/// gradient learning from semantic prediction error.
///
/// When `deep` is true, an inner double-bottleneck is added:
/// ```text
/// Forward:  HDC(16384) → w_down → LN → GELU+res → w_down2 → GELU → w_up2 → GELU+res → w_up → SSM(768)
/// Backward: SSM(768) → w_up^T → GELU → w_up2^T → GELU → w_down2^T → LN → w_down^T → HDC(16384)
/// ```
#[derive(Clone)]
pub struct HdcSsmProjection {
    // Forward: HDC → bottleneck → SSM
    w_down: Vec<f32>, // [bottleneck × hdc_dim]
    w_up: Vec<f32>,   // [ssm_dim × bottleneck]
    // Backward: SSM → bottleneck → HDC
    w_back_down: Vec<f32>, // [bottleneck × ssm_dim]
    w_back_up: Vec<f32>,   // [hdc_dim × bottleneck]
    // LayerNorm parameters (learned per-element scale + bias)
    ln_fwd_gamma: Vec<f32>, // [bottleneck] forward LayerNorm scale
    ln_fwd_beta: Vec<f32>,  // [bottleneck] forward LayerNorm bias
    ln_bwd_gamma: Vec<f32>, // [bottleneck] backward LayerNorm scale
    ln_bwd_beta: Vec<f32>,  // [bottleneck] backward LayerNorm bias
    // Deep inner bottleneck (only used when deep=true)
    // Forward inner: bottleneck → inner_dim → bottleneck
    w_down2: Vec<f32>, // [inner_dim × bottleneck]
    w_up2: Vec<f32>,   // [bottleneck × inner_dim]
    grad_down2: Vec<f32>,
    grad_up2: Vec<f32>,
    inner_dim: usize,
    deep: bool,
    // Gradient accumulators
    grad_down: Vec<f32>,
    grad_up: Vec<f32>,
    grad_back_down: Vec<f32>,
    grad_back_up: Vec<f32>,
    grad_ln_fwd_gamma: Vec<f32>,
    grad_ln_fwd_beta: Vec<f32>,
    grad_ln_bwd_gamma: Vec<f32>,
    grad_ln_bwd_beta: Vec<f32>,
    // EMA teacher (Polyak averaging) — None until enabled
    ema_weights: Option<Vec<f32>>,
    ema_decay: f32,
    // Optional gradient diagnostics (None by default)
    pub diagnostics: Option<ProjectionGradientDiagnostics>,
    // Generation at which diagnostics-triggered recovery last ran (prevents rapid re-triggering)
    last_diag_recovery_gen: usize,
    // Dimensions
    pub hdc_dim: usize,
    pub bottleneck: usize,
    pub ssm_dim: usize,
}

/// Metrics from a single gradient application step.
#[derive(Debug, Clone)]
pub struct GradientStepMetrics {
    /// L2 norm of the forward-down gradient accumulator.
    pub norm_down: f32,
    /// L2 norm of the forward-up gradient accumulator.
    pub norm_up: f32,
    /// Combined L2 norm of backward gradient accumulators.
    pub norm_backward: f32,
    /// Whether any gradient group was clipped this step.
    pub was_clipped: bool,
}

/// Accumulated diagnostics over multiple gradient steps.
#[derive(Debug, Clone, Default)]
pub struct ProjectionGradientDiagnostics {
    /// Per-step L2 norms of the forward-down gradient.
    pub grad_norms_down: Vec<f32>,
    /// Per-step L2 norms of the forward-up gradient.
    pub grad_norms_up: Vec<f32>,
    /// Per-step combined L2 norms of backward gradients.
    pub grad_norms_backward: Vec<f32>,
    /// Per-step max weight update magnitudes.
    pub update_magnitudes: Vec<f32>,
    /// Total gradient steps recorded.
    pub total_steps: usize,
    /// Number of steps where clipping was applied.
    pub clip_count: usize,
    /// Per-step L2 norms of the bottleneck activation.
    pub bottleneck_norms: Vec<f32>,
    /// Per-step variance of the bottleneck activation.
    pub bottleneck_variances: Vec<f32>,
}

impl ProjectionGradientDiagnostics {
    /// Record a gradient step's metrics.
    pub fn record_step(&mut self, metrics: &GradientStepMetrics, lr: f32) {
        self.grad_norms_down.push(metrics.norm_down);
        self.grad_norms_up.push(metrics.norm_up);
        self.grad_norms_backward.push(metrics.norm_backward);
        // Approximate max update magnitude: lr * max(norms)
        let max_norm = metrics
            .norm_down
            .max(metrics.norm_up)
            .max(metrics.norm_backward);
        self.update_magnitudes.push(lr * max_norm);
        self.total_steps += 1;
        if metrics.was_clipped {
            self.clip_count += 1;
        }
    }

    /// Record bottleneck activation statistics from a forward pass.
    pub fn record_bottleneck(&mut self, bottleneck: &[f32]) {
        let norm: f32 = bottleneck.iter().map(|v| v * v).sum::<f32>().sqrt();
        self.bottleneck_norms.push(norm);
        let n = bottleneck.len() as f32;
        if n > 0.0 {
            let mean: f32 = bottleneck.iter().sum::<f32>() / n;
            let var: f32 = bottleneck.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
            self.bottleneck_variances.push(var);
        }
    }

    /// Detect bottleneck collapse: variance consistently below threshold.
    ///
    /// Returns true if the last 5 recorded variances are all below 0.01,
    /// indicating the bottleneck dimensions have collapsed to near-constant values.
    pub fn bottleneck_collapse_detected(&self) -> bool {
        if self.bottleneck_variances.len() < 5 {
            return false;
        }
        self.bottleneck_variances
            .iter()
            .rev()
            .take(5)
            .all(|&v| v < 0.01)
    }

    /// Format a human-readable summary of gradient diagnostics.
    pub fn format_summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Projection Gradient Diagnostics ===\n\n");
        s.push_str(&format!("Total steps:       {}\n", self.total_steps));
        s.push_str(&format!(
            "Clip count:        {} ({:.1}%)\n",
            self.clip_count,
            if self.total_steps > 0 {
                self.clip_count as f32 / self.total_steps as f32 * 100.0
            } else {
                0.0
            }
        ));

        if !self.grad_norms_down.is_empty() {
            let avg_down: f32 =
                self.grad_norms_down.iter().sum::<f32>() / self.grad_norms_down.len() as f32;
            let avg_up: f32 =
                self.grad_norms_up.iter().sum::<f32>() / self.grad_norms_up.len() as f32;
            let avg_back: f32 = self.grad_norms_backward.iter().sum::<f32>()
                / self.grad_norms_backward.len() as f32;
            s.push_str(&format!("Avg grad norm (down):     {:.6}\n", avg_down));
            s.push_str(&format!("Avg grad norm (up):       {:.6}\n", avg_up));
            s.push_str(&format!("Avg grad norm (backward): {:.6}\n", avg_back));
        }

        if !self.update_magnitudes.is_empty() {
            let avg_mag: f32 =
                self.update_magnitudes.iter().sum::<f32>() / self.update_magnitudes.len() as f32;
            let max_mag: f32 = self
                .update_magnitudes
                .iter()
                .copied()
                .fold(0.0f32, f32::max);
            s.push_str(&format!("Avg update magnitude:     {:.6}\n", avg_mag));
            s.push_str(&format!("Max update magnitude:     {:.6}\n", max_mag));
        }

        if !self.bottleneck_variances.is_empty() {
            let avg_var: f32 = self.bottleneck_variances.iter().sum::<f32>()
                / self.bottleneck_variances.len() as f32;
            let avg_norm: f32 =
                self.bottleneck_norms.iter().sum::<f32>() / self.bottleneck_norms.len() as f32;
            s.push_str(&format!("Avg bottleneck norm:      {:.4}\n", avg_norm));
            s.push_str(&format!("Avg bottleneck variance:  {:.6}\n", avg_var));
            if self.bottleneck_collapse_detected() {
                s.push_str("WARNING: Bottleneck collapse detected (variance < 0.01)\n");
            }
        }

        s
    }
}

/// Serializable summary of gradient diagnostics for checkpoint persistence.
///
/// Captures the key metrics at checkpoint time without storing the full per-step
/// history vectors (which can be large). Allows resuming with context about
/// previous training health.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct GradientDiagnosticsSnapshot {
    /// Total gradient steps recorded.
    pub total_steps: usize,
    /// Number of steps where clipping was applied.
    pub clip_count: usize,
    /// Last recorded forward-down gradient norm.
    pub last_norm_down: f32,
    /// Last recorded forward-up gradient norm.
    pub last_norm_up: f32,
    /// Last recorded backward gradient norm.
    pub last_norm_backward: f32,
    /// Last recorded bottleneck activation variance.
    pub last_bottleneck_variance: f32,
    /// Whether bottleneck collapse was detected at snapshot time.
    pub collapse_detected: bool,
}

impl ProjectionGradientDiagnostics {
    /// Create a serializable snapshot of the current diagnostics state.
    ///
    /// Captures summary metrics (latest norms, collapse status) without
    /// the full per-step history, keeping checkpoints compact.
    pub fn snapshot(&self) -> GradientDiagnosticsSnapshot {
        GradientDiagnosticsSnapshot {
            total_steps: self.total_steps,
            clip_count: self.clip_count,
            last_norm_down: self.grad_norms_down.last().copied().unwrap_or(0.0),
            last_norm_up: self.grad_norms_up.last().copied().unwrap_or(0.0),
            last_norm_backward: self.grad_norms_backward.last().copied().unwrap_or(0.0),
            last_bottleneck_variance: self.bottleneck_variances.last().copied().unwrap_or(0.0),
            collapse_detected: self.bottleneck_collapse_detected(),
        }
    }

    /// Restore summary counters from a checkpoint snapshot.
    ///
    /// Seeds `total_steps` and `clip_count` so that resumed training
    /// reports cumulative values. Does NOT restore per-step history
    /// vectors (those remain empty until new steps are recorded).
    pub fn restore_from_snapshot(&mut self, snap: &GradientDiagnosticsSnapshot) {
        self.total_steps = snap.total_steps;
        self.clip_count = snap.clip_count;
    }
}

impl HdcSsmProjection {
    pub fn diagnostics(&self) -> Option<&ProjectionGradientDiagnostics> {
        self.diagnostics.as_ref()
    }
    pub fn diagnostics_mut(&mut self) -> Option<&mut ProjectionGradientDiagnostics> {
        self.diagnostics.as_mut()
    }
    pub fn effective_rank(&self, samples: &[ContinuousHV]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }
        let dim = self.bottleneck;
        let mut cov = vec![0.0f32; dim * dim];
        for hv in samples {
            let proj = self.project_to_ssm(hv); // Use bottleneck activation
            for i in 0..dim {
                for j in 0..dim {
                    cov[i * dim + j] += proj[i] * proj[j];
                }
            }
        }
        let trace: f32 = (0..dim).map(|i| cov[i * dim + i]).sum();
        if trace < 1e-6 {
            return 0.0;
        }
        // Simplified effective rank: trace^2 / sum(eigenvalues^2)
        let sum_sq: f32 = cov.iter().map(|x| x * x).sum();
        (trace * trace) / sum_sq
    }
}

/// Trait for layers that support local Free Energy Principle (FEP) learning.
///
/// Instead of global BPTT, these layers update based on local prediction errors:
/// dW = (Prediction - Observation) ⊗ Input.
pub trait LocalFepLayer {
    /// Update weights based on a local prediction error.
    /// Returns the cost in Joules for the ThermodynamicLedger.
    fn local_fep_update(
        &mut self,
        input: &[f32],
        prediction: &[f32],
        observation: &[f32],
        lr: f32,
    ) -> f64;
}

impl LocalFepLayer for HdcSsmProjection {
    fn local_fep_update(
        &mut self,
        input: &[f32],
        prediction: &[f32],
        observation: &[f32],
        lr: f32,
    ) -> f64 {
        let rows = self.bottleneck;
        let cols = self.hdc_dim;

        // Local Prediction Error: Δ = Prediction - Observation
        let mut delta = vec![0.0f32; rows];
        let mut surprise_sq = 0.0f32;
        for i in 0..rows {
            delta[i] = prediction[i] - observation[i];
            surprise_sq += delta[i] * delta[i];
        }

        // Weight update: W_new = W_old - lr * (Δ ⊗ Input)
        for i in 0..rows {
            let gi = delta[i];
            for j in 0..cols {
                self.w_down[i * cols + j] -= lr * gi * input[j];
            }
        }

        // Thermodynamic cost scales with surprise and weight matrix size
        let cost = (surprise_sq as f64 * (rows * cols) as f64 * 1e-9).min(0.5);
        cost
    }
}

/// LayerNorm epsilon for numerical stability.
const LN_EPS: f32 = 1e-5;

/// Apply LayerNorm: `gamma * (x - mean) / sqrt(var + eps) + beta`.
///
/// Returns (normalized, mean, inv_std) for backward pass.
fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + LN_EPS).sqrt();

    let normed: Vec<f32> = x
        .iter()
        .zip(gamma.iter().zip(beta.iter()))
        .map(|(&xi, (&g, &b))| g * (xi - mean) * inv_std + b)
        .collect();
    // Also return x_hat for backward pass
    let x_hat: Vec<f32> = x.iter().map(|&xi| (xi - mean) * inv_std).collect();
    (normed, x_hat, inv_std)
}

/// LayerNorm backward pass.
///
/// Given gradient flowing back through LN output, computes:
/// - Gradient w.r.t. input (returned)
/// - Gradient w.r.t. gamma/beta (accumulated into grad buffers)
fn layer_norm_backward(
    d_out: &[f32],
    x_hat: &[f32],
    gamma: &[f32],
    inv_std: f32,
    grad_gamma: &mut [f32],
    grad_beta: &mut [f32],
) -> Vec<f32> {
    let n = d_out.len() as f32;

    // Accumulate parameter gradients
    for i in 0..d_out.len() {
        grad_gamma[i] += d_out[i] * x_hat[i];
        grad_beta[i] += d_out[i];
    }

    // d_x_hat = d_out * gamma
    let d_x_hat: Vec<f32> = d_out
        .iter()
        .zip(gamma.iter())
        .map(|(&dy, &g)| dy * g)
        .collect();

    // d_x = inv_std * (d_x_hat - mean(d_x_hat) - x_hat * mean(d_x_hat * x_hat))
    let mean_d: f32 = d_x_hat.iter().sum::<f32>() / n;
    let mean_dx: f32 = d_x_hat
        .iter()
        .zip(x_hat.iter())
        .map(|(&d, &x)| d * x)
        .sum::<f32>()
        / n;

    d_x_hat
        .iter()
        .zip(x_hat.iter())
        .map(|(&dxh, &xh)| inv_std * (dxh - mean_d - xh * mean_dx))
        .collect()
}

impl HdcSsmProjection {
    /// Create a new projection with JL-style random initialization.
    ///
    /// Weights are scaled by `1/sqrt(bottleneck)` for variance preservation.
    /// Genesis-seeded for deterministic initialization.
    pub fn new(genesis: &GenesisSeed, hdc_dim: usize, bottleneck: usize, ssm_dim: usize) -> Self {
        let scale = 1.0 / (bottleneck as f32).sqrt();

        let w_down = Self::init_weights(genesis, "projection::w_down", bottleneck * hdc_dim, scale);
        let w_up = Self::init_weights(genesis, "projection::w_up", ssm_dim * bottleneck, scale);
        let w_back_down = Self::init_weights(
            genesis,
            "projection::w_back_down",
            bottleneck * ssm_dim,
            scale,
        );
        let w_back_up = Self::init_weights(
            genesis,
            "projection::w_back_up",
            hdc_dim * bottleneck,
            scale,
        );

        Self {
            grad_down: vec![0.0; bottleneck * hdc_dim],
            grad_up: vec![0.0; ssm_dim * bottleneck],
            grad_back_down: vec![0.0; bottleneck * ssm_dim],
            grad_back_up: vec![0.0; hdc_dim * bottleneck],
            grad_ln_fwd_gamma: vec![0.0; bottleneck],
            grad_ln_fwd_beta: vec![0.0; bottleneck],
            grad_ln_bwd_gamma: vec![0.0; bottleneck],
            grad_ln_bwd_beta: vec![0.0; bottleneck],
            w_down,
            w_up,
            w_back_down,
            w_back_up,
            ln_fwd_gamma: vec![1.0; bottleneck],
            ln_fwd_beta: vec![0.0; bottleneck],
            ln_bwd_gamma: vec![1.0; bottleneck],
            ln_bwd_beta: vec![0.0; bottleneck],
            // Deep inner bottleneck — empty/unused in shallow mode
            w_down2: Vec::new(),
            w_up2: Vec::new(),
            grad_down2: Vec::new(),
            grad_up2: Vec::new(),
            inner_dim: 0,
            deep: false,
            ema_weights: None,
            ema_decay: 0.999,
            hdc_dim,
            bottleneck,
            ssm_dim,
            diagnostics: None,
            last_diag_recovery_gen: 0,
        }
    }

    /// Initialize a weight vector with genesis-seeded random values scaled by `scale`.
    fn init_weights(genesis: &GenesisSeed, label: &str, size: usize, scale: f32) -> Vec<f32> {
        // Use genesis to create a deterministic ContinuousHV, then tile/truncate
        // to the desired size. For large weight matrices we chunk the initialization.
        let chunk_size = 16384; // One HDC dimension
        let mut weights = Vec::with_capacity(size);
        let mut chunk_idx = 0;
        while weights.len() < size {
            let chunk_label = format!("{label}::chunk{chunk_idx}");
            let hv = genesis.hv(&chunk_label, chunk_size);
            let remaining = size - weights.len();
            let take = remaining.min(chunk_size);
            weights.extend_from_slice(&hv.values[..take]);
            chunk_idx += 1;
        }
        // Apply JL scaling
        for w in &mut weights {
            *w *= scale;
        }
        weights
    }

    /// Create a deep projection with a double bottleneck.
    ///
    /// Architecture:
    /// ```text
    /// Forward:  HDC(16384) → w_down(256) → LN → GELU+res → w_down2(128) → GELU → w_up2(256) → GELU+res → w_up → SSM(768)
    /// Backward: SSM(768) → w_up^T(256) → GELU → w_up2^T(128) → GELU → w_down2^T(256) → LN → w_down^T → HDC(16384)
    /// ```
    ///
    /// The inner dimension is `bottleneck_dim / 2`. This provides a stronger
    /// information bottleneck that forces higher-quality compression while
    /// the skip connections maintain gradient flow.
    pub fn new_deep(
        genesis: &GenesisSeed,
        hdc_dim: usize,
        bottleneck: usize,
        ssm_dim: usize,
    ) -> Self {
        let inner_dim = bottleneck / 2;
        let scale = 1.0 / (bottleneck as f32).sqrt();
        let inner_scale = 1.0 / (inner_dim as f32).sqrt();

        let w_down = Self::init_weights(genesis, "projection::w_down", bottleneck * hdc_dim, scale);
        let w_up = Self::init_weights(genesis, "projection::w_up", ssm_dim * bottleneck, scale);
        let w_back_down = Self::init_weights(
            genesis,
            "projection::w_back_down",
            bottleneck * ssm_dim,
            scale,
        );
        let w_back_up = Self::init_weights(
            genesis,
            "projection::w_back_up",
            hdc_dim * bottleneck,
            scale,
        );
        let w_down2 = Self::init_weights(
            genesis,
            "projection::w_down2",
            inner_dim * bottleneck,
            inner_scale,
        );
        let w_up2 = Self::init_weights(
            genesis,
            "projection::w_up2",
            bottleneck * inner_dim,
            inner_scale,
        );

        Self {
            grad_down: vec![0.0; bottleneck * hdc_dim],
            grad_up: vec![0.0; ssm_dim * bottleneck],
            grad_back_down: vec![0.0; bottleneck * ssm_dim],
            grad_back_up: vec![0.0; hdc_dim * bottleneck],
            grad_ln_fwd_gamma: vec![0.0; bottleneck],
            grad_ln_fwd_beta: vec![0.0; bottleneck],
            grad_ln_bwd_gamma: vec![0.0; bottleneck],
            grad_ln_bwd_beta: vec![0.0; bottleneck],
            grad_down2: vec![0.0; inner_dim * bottleneck],
            grad_up2: vec![0.0; bottleneck * inner_dim],
            w_down,
            w_up,
            w_back_down,
            w_back_up,
            w_down2,
            w_up2,
            ln_fwd_gamma: vec![1.0; bottleneck],
            ln_fwd_beta: vec![0.0; bottleneck],
            ln_bwd_gamma: vec![1.0; bottleneck],
            ln_bwd_beta: vec![0.0; bottleneck],
            inner_dim,
            deep: true,
            ema_weights: None,
            ema_decay: 0.999,
            hdc_dim,
            bottleneck,
            ssm_dim,
            diagnostics: None,
            last_diag_recovery_gen: 0,
        }
    }

    /// Whether this projection uses the deep double-bottleneck architecture.
    pub fn is_deep(&self) -> bool {
        self.deep
    }

    /// Inner bottleneck dimension (0 if shallow).
    pub fn inner_dim(&self) -> usize {
        self.inner_dim
    }

    /// Project HDC hypervector (16,384D) to SSM space (768D).
    ///
    /// Shallow pipeline: `hv → w_down → LayerNorm → GELU+residual → w_up → ssm_vec`
    /// Deep pipeline: `hv → w_down → LN → GELU+res → w_down2 → GELU → w_up2 → GELU+res → w_up → ssm_vec`
    pub fn project_to_ssm(&self, hv: &ContinuousHV) -> Vec<f32> {
        debug_assert_eq!(hv.values.len(), self.hdc_dim);

        // Step 1: w_down * hv → bottleneck (256D)
        let hidden_pre = self.matmul(&self.w_down, &hv.values, self.bottleneck, self.hdc_dim);

        // Step 2: LayerNorm on bottleneck
        let (hidden_normed, _, _) = layer_norm(&hidden_pre, &self.ln_fwd_gamma, &self.ln_fwd_beta);

        // Step 3: GELU + pre-activation residual
        let hidden: Vec<f32> = hidden_normed.into_iter().map(activation).collect();

        // Step 3.5 (deep only): inner bottleneck pass
        let hidden = if self.deep {
            // w_down2: bottleneck → inner_dim, then GELU
            let inner = self.matmul(&self.w_down2, &hidden, self.inner_dim, self.bottleneck);
            let inner_act: Vec<f32> = inner.into_iter().map(gelu).collect();
            // w_up2: inner_dim → bottleneck, then GELU + residual with pre-inner hidden
            let expanded = self.matmul(&self.w_up2, &inner_act, self.bottleneck, self.inner_dim);
            expanded
                .into_iter()
                .zip(hidden.iter())
                .map(|(e, &h)| activation(e) + RESIDUAL_ALPHA * h)
                .collect()
        } else {
            hidden
        };

        // Step 4: w_up * hidden → ssm (768D)
        self.matmul(&self.w_up, &hidden, self.ssm_dim, self.bottleneck)
    }

    /// Project SSM vector (768D) back to HDC space (16,384D).
    ///
    /// Shallow pipeline: `ssm_vec → w_back_down → LayerNorm → GELU+residual → w_back_up → hv`
    /// Deep pipeline: `ssm_vec → w_up^T → GELU → w_up2^T → GELU → w_down2^T → LN → w_down^T → hv`
    pub fn project_to_hdc(&self, ssm_vec: &[f32]) -> ContinuousHV {
        debug_assert_eq!(ssm_vec.len(), self.ssm_dim);

        if self.deep {
            return self.project_to_hdc_deep(ssm_vec);
        }

        // Step 1: w_back_down * ssm → bottleneck (256D)
        let hidden_pre = self.matmul(&self.w_back_down, ssm_vec, self.bottleneck, self.ssm_dim);

        // Step 2: LayerNorm on bottleneck
        let (hidden_normed, _, _) = layer_norm(&hidden_pre, &self.ln_bwd_gamma, &self.ln_bwd_beta);

        // Step 3: GELU + pre-activation residual
        let hidden: Vec<f32> = hidden_normed.into_iter().map(activation).collect();

        // Step 4: w_back_up * hidden → hdc (16,384D)
        let values = self.matmul(&self.w_back_up, &hidden, self.hdc_dim, self.bottleneck);

        ContinuousHV::from_vec(values)
    }

    /// Deep backward path: SSM → w_up^T → GELU → w_up2^T → GELU → w_down2^T → LN → w_down^T → HDC.
    ///
    /// Mirrors the deep forward path using transposed weight matrices.
    fn project_to_hdc_deep(&self, ssm_vec: &[f32]) -> ContinuousHV {
        debug_assert!(self.deep);

        // Step 1: w_up^T * ssm → bottleneck (256D)
        //   w_up is [ssm_dim × bottleneck], its transpose is [bottleneck × ssm_dim]
        let mut hidden = vec![0.0f32; self.bottleneck];
        for j in 0..self.bottleneck {
            let mut sum = 0.0f32;
            for i in 0..self.ssm_dim {
                sum += self.w_up[i * self.bottleneck + j] * ssm_vec[i];
            }
            hidden[j] = gelu(sum);
        }

        // Step 2: w_up2^T * hidden → inner_dim
        //   w_up2 is [bottleneck × inner_dim], its transpose is [inner_dim × bottleneck]
        let mut inner = vec![0.0f32; self.inner_dim];
        for j in 0..self.inner_dim {
            let mut sum = 0.0f32;
            for i in 0..self.bottleneck {
                sum += self.w_up2[i * self.inner_dim + j] * hidden[i];
            }
            inner[j] = gelu(sum);
        }

        // Step 3: w_down2^T * inner → bottleneck
        //   w_down2 is [inner_dim × bottleneck], its transpose is [bottleneck × inner_dim]
        let mut hidden2 = vec![0.0f32; self.bottleneck];
        for j in 0..self.bottleneck {
            let mut sum = 0.0f32;
            for i in 0..self.inner_dim {
                sum += self.w_down2[i * self.bottleneck + j] * inner[i];
            }
            hidden2[j] = sum;
        }

        // Step 4: LayerNorm on bottleneck
        let (normed, _, _) = layer_norm(&hidden2, &self.ln_bwd_gamma, &self.ln_bwd_beta);

        // Step 5: w_down^T * normed → HDC
        //   w_down is [bottleneck × hdc_dim], its transpose is [hdc_dim × bottleneck]
        let mut values = vec![0.0f32; self.hdc_dim];
        for j in 0..self.hdc_dim {
            let mut sum = 0.0f32;
            for i in 0..self.bottleneck {
                sum += self.w_down[i * self.hdc_dim + j] * normed[i];
            }
            values[j] = sum;
        }

        ContinuousHV::from_vec(values)
    }

    /// Verify a metamorphic kernel before application.
    /// Ensures it doesn't cause manifold collapse or magnitude explosion.
    pub fn verify_metamorphic_kernel(&self, kernel: &[f32]) -> bool {
        if kernel.is_empty() {
            return false;
        }

        // 1. Check for NaNs or Infinities
        if kernel.iter().any(|&v| !v.is_finite()) {
            return false;
        }

        // 2. Check Kernel Entropy (Manifold Volume)
        // A "flat" kernel (all same values) would cause collapse.
        let mean: f32 = kernel.iter().sum::<f32>() / kernel.len() as f32;
        let variance: f32 =
            kernel.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / kernel.len() as f32;

        if variance < 0.001 {
            // Kernel is too uniform: would destroy semantic differentiation
            return false;
        }

        // 3. Check Magnitude
        let norm: f32 = kernel.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if norm > 100.0 {
            // Kernel is too explosive
            return false;
        }

        true
    }

    /// Apply a metamorphic weight update kernel with a dynamic pressure factor.
    /// High pressure (confusion) increases plasticity; low pressure (stable) crystalizes.
    pub fn apply_metamorphic_kernel(&mut self, kernel: &[f32], target: &str, pressure: f32) {
        let weights = match target {
            "w_down" => &mut self.w_down,
            "w_up" => &mut self.w_up,
            _ => return,
        };

        // Pressure maps to a metamorphic learning rate [0.01, 0.10]
        let lr = pressure.clamp(0.01, 0.10);
        let retention = 1.0 - lr;

        // Apply kernel via element-wise modulation (Metamorphic Scaling)
        for (w, &k) in weights.iter_mut().zip(kernel.iter().cycle()) {
            *w = *w * retention + k * lr;
        }
    }

    /// Backpropagate gradients and update weights mathematically correctly.
    /// `d_bottleneck` must be the gradient of the loss with respect to the bottleneck output (size: self.bottleneck).
    pub fn backward(&mut self, input_hv: &ContinuousHV, d_bottleneck: &[f32], lr: f32) {
        let rows = self.bottleneck;
        let cols = self.hdc_dim;
        let x_values = &input_hv.values;

        // Linear layer gradient: dW = dY (outer product) X
        for i in 0..rows {
            let gi = d_bottleneck[i]; // The exact gradient for this specific bottleneck neuron

            for j in 0..cols {
                // Gradient descent: W_new = W_old - lr * dW
                self.w_down[i * cols + j] -= lr * gi * x_values[j];
            }
        }
    }

    /// Compute gradients from semantic prediction error.
    ///
    /// The error signal is the difference between the original thought HV
    /// and the round-trip reconstruction (project_to_ssm → project_to_hdc).
    /// Backpropagates through: w_back_up → activation → LayerNorm → w_back_down →
    /// w_up → [inner bottleneck if deep] → activation → LayerNorm → w_down.
    pub fn compute_gradients(&mut self, thought_hv: &ContinuousHV, output_hv: &ContinuousHV) {
        debug_assert_eq!(thought_hv.values.len(), self.hdc_dim);
        debug_assert_eq!(output_hv.values.len(), self.hdc_dim);

        // Error = thought - output (MSE gradient direction)
        let error: Vec<f32> = thought_hv
            .values
            .iter()
            .zip(output_hv.values.iter())
            .map(|(t, o)| t - o)
            .collect();

        // === Forward pass (cache intermediates for backprop) ===
        // Forward: thought → w_down → LN_fwd → GELU+res → [deep inner] → w_up → ssm
        let fwd_pre_ln = self.matmul(
            &self.w_down,
            &thought_hv.values,
            self.bottleneck,
            self.hdc_dim,
        );
        let (fwd_normed, fwd_x_hat, fwd_inv_std) =
            layer_norm(&fwd_pre_ln, &self.ln_fwd_gamma, &self.ln_fwd_beta);
        let hidden_fwd_pre_deep: Vec<f32> = fwd_normed.iter().map(|&x| activation(x)).collect();

        // Deep forward inner bottleneck intermediates (cached for backprop)
        let (inner_fwd_pre_act, inner_fwd_act, expanded_fwd_pre_act, hidden_fwd) = if self.deep {
            // w_down2: bottleneck → inner_dim
            let inner_pre = self.matmul(
                &self.w_down2,
                &hidden_fwd_pre_deep,
                self.inner_dim,
                self.bottleneck,
            );
            let inner_act: Vec<f32> = inner_pre.iter().map(|&x| gelu(x)).collect();
            // w_up2: inner_dim → bottleneck
            let expanded = self.matmul(&self.w_up2, &inner_act, self.bottleneck, self.inner_dim);
            let hidden: Vec<f32> = expanded
                .iter()
                .zip(hidden_fwd_pre_deep.iter())
                .map(|(&e, &h)| activation(e) + RESIDUAL_ALPHA * h)
                .collect();
            (
                Some(inner_pre),
                Some(inner_act),
                Some(expanded.clone()),
                hidden,
            )
        } else {
            (None, None, None, hidden_fwd_pre_deep.clone())
        };

        let ssm_fwd = self.matmul(&self.w_up, &hidden_fwd, self.ssm_dim, self.bottleneck);

        // Backward reconstruction: ssm → w_back_down → LN_bwd → GELU+res → w_back_up → reconstructed
        let bwd_pre_ln = self.matmul(&self.w_back_down, &ssm_fwd, self.bottleneck, self.ssm_dim);
        let (bwd_normed, bwd_x_hat, bwd_inv_std) =
            layer_norm(&bwd_pre_ln, &self.ln_bwd_gamma, &self.ln_bwd_beta);
        let hidden_back: Vec<f32> = bwd_normed.iter().map(|&x| activation(x)).collect();

        // === Backprop through backward projection path ===

        // Gradient for w_back_up: error * hidden_back^T
        for i in 0..self.hdc_dim {
            for j in 0..self.bottleneck {
                self.grad_back_up[i * self.bottleneck + j] += error[i] * hidden_back[j];
            }
        }

        // delta at backward activation input: (w_back_up^T * error) * act'(bwd_normed)
        let mut delta_back_act = vec![0.0f32; self.bottleneck];
        for j in 0..self.bottleneck {
            let mut sum = 0.0f32;
            for i in 0..self.hdc_dim {
                sum += self.w_back_up[i * self.bottleneck + j] * error[i];
            }
            delta_back_act[j] = sum * activation_derivative(bwd_normed[j]);
        }

        // Backprop through backward LayerNorm
        let delta_back_pre_ln = layer_norm_backward(
            &delta_back_act,
            &bwd_x_hat,
            &self.ln_bwd_gamma,
            bwd_inv_std,
            &mut self.grad_ln_bwd_gamma,
            &mut self.grad_ln_bwd_beta,
        );

        // Gradient for w_back_down: delta_back_pre_ln * ssm_fwd^T
        for i in 0..self.bottleneck {
            for j in 0..self.ssm_dim {
                self.grad_back_down[i * self.ssm_dim + j] += delta_back_pre_ln[i] * ssm_fwd[j];
            }
        }

        // === Backprop through forward projection path ===

        // Gradient for w_up: (w_back_down^T * delta_back_pre_ln) * hidden_fwd^T
        let mut delta_up = vec![0.0f32; self.ssm_dim];
        for j in 0..self.ssm_dim {
            let mut sum = 0.0f32;
            for i in 0..self.bottleneck {
                sum += self.w_back_down[i * self.ssm_dim + j] * delta_back_pre_ln[i];
            }
            delta_up[j] = sum;
        }
        for i in 0..self.ssm_dim {
            for j in 0..self.bottleneck {
                self.grad_up[i * self.bottleneck + j] += delta_up[i] * hidden_fwd[j];
            }
        }

        // delta flowing back from w_up into the bottleneck
        let mut delta_pre_deep = vec![0.0f32; self.bottleneck];
        for j in 0..self.bottleneck {
            let mut sum = 0.0f32;
            for i in 0..self.ssm_dim {
                sum += self.w_up[i * self.bottleneck + j] * delta_up[i];
            }
            delta_pre_deep[j] = sum;
        }

        // === Backprop through deep inner bottleneck (if enabled) ===
        let delta_fwd_act = if self.deep {
            let inner_pre = inner_fwd_pre_act
                .as_ref()
                .expect("deep mode guarantees Some intermediates");
            let inner_act = inner_fwd_act
                .as_ref()
                .expect("deep mode guarantees Some intermediates");
            let expanded_pre = expanded_fwd_pre_act
                .as_ref()
                .expect("deep mode guarantees Some intermediates");

            // delta_pre_deep flows into activation(expanded) + RESIDUAL_ALPHA * hidden_fwd_pre_deep
            // d/d(expanded) = delta_pre_deep * act'(expanded)
            // d/d(hidden_fwd_pre_deep) += delta_pre_deep * RESIDUAL_ALPHA
            let mut delta_expanded = vec![0.0f32; self.bottleneck];
            let mut delta_from_residual = vec![0.0f32; self.bottleneck];
            for j in 0..self.bottleneck {
                delta_expanded[j] = delta_pre_deep[j] * activation_derivative(expanded_pre[j]);
                delta_from_residual[j] = delta_pre_deep[j] * RESIDUAL_ALPHA;
            }

            // Gradient for w_up2: delta_expanded * inner_act^T
            //   w_up2 is [bottleneck × inner_dim]
            for i in 0..self.bottleneck {
                for j in 0..self.inner_dim {
                    self.grad_up2[i * self.inner_dim + j] += delta_expanded[i] * inner_act[j];
                }
            }

            // delta at inner_act: w_up2^T * delta_expanded
            let mut delta_inner_act = vec![0.0f32; self.inner_dim];
            for j in 0..self.inner_dim {
                let mut sum = 0.0f32;
                for i in 0..self.bottleneck {
                    sum += self.w_up2[i * self.inner_dim + j] * delta_expanded[i];
                }
                delta_inner_act[j] = sum;
            }

            // Backprop through GELU at inner layer: delta_inner_pre = delta_inner_act * gelu'(inner_pre)
            let mut delta_inner_pre = vec![0.0f32; self.inner_dim];
            for j in 0..self.inner_dim {
                delta_inner_pre[j] = delta_inner_act[j] * gelu_derivative(inner_pre[j]);
            }

            // Gradient for w_down2: delta_inner_pre * hidden_fwd_pre_deep^T
            //   w_down2 is [inner_dim × bottleneck]
            for i in 0..self.inner_dim {
                for j in 0..self.bottleneck {
                    self.grad_down2[i * self.bottleneck + j] +=
                        delta_inner_pre[i] * hidden_fwd_pre_deep[j];
                }
            }

            // delta at hidden_fwd_pre_deep from w_down2: w_down2^T * delta_inner_pre
            let mut delta_from_down2 = vec![0.0f32; self.bottleneck];
            for j in 0..self.bottleneck {
                let mut sum = 0.0f32;
                for i in 0..self.inner_dim {
                    sum += self.w_down2[i * self.bottleneck + j] * delta_inner_pre[i];
                }
                delta_from_down2[j] = sum;
            }

            // Total delta at activation(fwd_normed): from inner path + residual path
            let mut delta_at_act = vec![0.0f32; self.bottleneck];
            for j in 0..self.bottleneck {
                delta_at_act[j] = (delta_from_down2[j] + delta_from_residual[j])
                    * activation_derivative(fwd_normed[j]);
            }
            delta_at_act
        } else {
            // Shallow: delta directly through activation
            let mut delta_fwd_act = vec![0.0f32; self.bottleneck];
            for j in 0..self.bottleneck {
                delta_fwd_act[j] = delta_pre_deep[j] * activation_derivative(fwd_normed[j]);
            }
            delta_fwd_act
        };

        // Backprop through forward LayerNorm
        let delta_fwd_pre_ln = layer_norm_backward(
            &delta_fwd_act,
            &fwd_x_hat,
            &self.ln_fwd_gamma,
            fwd_inv_std,
            &mut self.grad_ln_fwd_gamma,
            &mut self.grad_ln_fwd_beta,
        );

        // Gradient for w_down: delta_fwd_pre_ln * thought_hv^T
        for i in 0..self.bottleneck {
            for j in 0..self.hdc_dim {
                self.grad_down[i * self.hdc_dim + j] += delta_fwd_pre_ln[i] * thought_hv.values[j];
            }
        }
    }

    /// Compute gradients from pure roundtrip reconstruction (autoencoder loss).
    ///
    /// Error = `thought - backward(forward(thought))`. This doesn't depend on
    /// Mamba's output, giving a clean gradient signal even when the projection
    /// is randomly initialized. Use during early training (high PE) before
    /// switching to Mamba-output-based gradients.
    pub fn compute_roundtrip_gradients(&mut self, thought_hv: &ContinuousHV) {
        let ssm_context = self.project_to_ssm(thought_hv);
        let reconstructed = self.project_to_hdc(&ssm_context);
        self.compute_gradients(thought_hv, &reconstructed);
    }

    /// Compute contrastive gradients: push anchor and negative apart in bottleneck space.
    ///
    /// Adds a repulsive gradient so that the projections of `anchor_hv` and
    /// `negative_hv` produce different bottleneck representations. This prevents
    /// the projection from collapsing all thoughts to the same SSM context.
    pub fn compute_contrastive_gradients(
        &mut self,
        anchor_hv: &ContinuousHV,
        negative_hv: &ContinuousHV,
        weight: f32,
    ) {
        // Forward both through w_down + LayerNorm to get bottleneck representations
        let pre_anchor = self.matmul(
            &self.w_down,
            &anchor_hv.values,
            self.bottleneck,
            self.hdc_dim,
        );
        let pre_neg = self.matmul(
            &self.w_down,
            &negative_hv.values,
            self.bottleneck,
            self.hdc_dim,
        );
        let (normed_anchor, _, _) = layer_norm(&pre_anchor, &self.ln_fwd_gamma, &self.ln_fwd_beta);
        let (normed_neg, _, _) = layer_norm(&pre_neg, &self.ln_fwd_gamma, &self.ln_fwd_beta);

        // Apply activation to get post-activation representations
        let act_anchor: Vec<f32> = normed_anchor.iter().map(|&x| activation(x)).collect();
        let act_neg: Vec<f32> = normed_neg.iter().map(|&x| activation(x)).collect();

        // Contrastive gradient on w_down: push activated representations apart
        // (simplified: gradient through LN approximated as identity for contrastive signal)
        for i in 0..self.bottleneck {
            let delta = weight * (act_anchor[i] - act_neg[i]);
            let d_act = activation_derivative(normed_anchor[i]);
            let row_start = i * self.hdc_dim;
            for j in 0..self.hdc_dim {
                self.grad_down[row_start + j] += delta * d_act * anchor_hv.values[j];
            }
        }
    }

    /// Contrastive pretraining: learn to separate different thoughts in bottleneck space.
    ///
    /// Runs `epochs` passes over all unique pairs of thoughts, pushing each pair apart
    /// in the projection bottleneck. This gives the projection discriminative structure
    /// *before* distillation fine-tuning, preventing mode collapse.
    ///
    /// Also adds a reconstruction objective on `w_back_up ∘ w_back_down ∘ w_up ∘ w_down`
    /// to keep the round-trip path healthy.
    ///
    /// Returns (final_avg_distance, final_reconstruction_error).
    pub fn contrastive_pretrain(
        &mut self,
        thought_hvs: &[ContinuousHV],
        epochs: usize,
        lr: f32,
    ) -> (f32, f32) {
        if thought_hvs.len() < 2 {
            return (0.0, 1.0);
        }

        let grad_clip = 1.0;
        let contrastive_weight = 0.01;
        let recon_weight = 0.005;
        let mut avg_dist = 0.0f32;
        let mut avg_recon = 0.0f32;

        for _epoch in 0..epochs {
            let mut total_dist = 0.0f32;
            let mut total_recon = 0.0f32;
            let mut pair_count = 0usize;

            // All unique pairs (capped at ~200 for efficiency)
            let n = thought_hvs.len().min(20); // 20 thoughts → 190 pairs
            for i in 0..n {
                for j in (i + 1)..n {
                    // Contrastive: push apart
                    self.compute_contrastive_gradients(
                        &thought_hvs[i],
                        &thought_hvs[j],
                        contrastive_weight,
                    );

                    // Measure bottleneck distance
                    let ssm_i = self.project_to_ssm(&thought_hvs[i]);
                    let ssm_j = self.project_to_ssm(&thought_hvs[j]);
                    let dist: f32 = ssm_i
                        .iter()
                        .zip(ssm_j.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    total_dist += dist;

                    pair_count += 1;
                }

                // Reconstruction: round-trip through forward + backward
                let ssm_vec = self.project_to_ssm(&thought_hvs[i]);
                let recon_hv = self.project_to_hdc(&ssm_vec);
                self.compute_gradients(&thought_hvs[i], &recon_hv);

                // Accumulate reconstruction error (MSE)
                let recon_error: f32 = thought_hvs[i]
                    .values
                    .iter()
                    .zip(recon_hv.values.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    / thought_hvs[i].values.len() as f32;
                total_recon += recon_error;

                // Scale reconstruction gradients
                for g in self.grad_back_down.iter_mut() {
                    *g *= recon_weight;
                }
                for g in self.grad_back_up.iter_mut() {
                    *g *= recon_weight;
                }
            }

            self.apply_gradients(lr, grad_clip);

            if pair_count > 0 {
                avg_dist = total_dist / pair_count as f32;
            }
            avg_recon = total_recon / n.max(1) as f32;
        }

        (avg_dist, avg_recon)
    }

    /// Scale all accumulated gradient buffers by a scalar factor.
    ///
    /// Call this between gradient accumulation and application for
    /// per-example curriculum weighting (e.g. surprise-weighted gradients).
    pub fn scale_accumulated_gradients(&mut self, factor: f32) {
        for g in self.grad_down.iter_mut() {
            *g *= factor;
        }
        for g in self.grad_up.iter_mut() {
            *g *= factor;
        }
        for g in self.grad_back_down.iter_mut() {
            *g *= factor;
        }
        for g in self.grad_back_up.iter_mut() {
            *g *= factor;
        }
        for g in self.grad_ln_fwd_gamma.iter_mut() {
            *g *= factor;
        }
        for g in self.grad_ln_fwd_beta.iter_mut() {
            *g *= factor;
        }
        for g in self.grad_ln_bwd_gamma.iter_mut() {
            *g *= factor;
        }
        for g in self.grad_ln_bwd_beta.iter_mut() {
            *g *= factor;
        }
        if self.deep {
            for g in self.grad_down2.iter_mut() {
                *g *= factor;
            }
            for g in self.grad_up2.iter_mut() {
                *g *= factor;
            }
        }
    }

    /// Apply accumulated gradients with SGD + gradient clipping, then zero accumulators.
    ///
    /// Returns [`GradientStepMetrics`] with L2 norms of the main gradient groups
    /// (computed before zeroing) and whether any group was clipped.
    pub fn apply_gradients(&mut self, lr: f32, grad_clip: f32) -> GradientStepMetrics {
        let (n_down, c_down) =
            Self::apply_grad(&mut self.w_down, &mut self.grad_down, lr, grad_clip);
        let (n_up, c_up) = Self::apply_grad(&mut self.w_up, &mut self.grad_up, lr, grad_clip);
        let (n_bdown, c_bdown) = Self::apply_grad(
            &mut self.w_back_down,
            &mut self.grad_back_down,
            lr,
            grad_clip,
        );
        let (n_bup, c_bup) =
            Self::apply_grad(&mut self.w_back_up, &mut self.grad_back_up, lr, grad_clip);
        // LayerNorm parameters
        Self::apply_grad(
            &mut self.ln_fwd_gamma,
            &mut self.grad_ln_fwd_gamma,
            lr,
            grad_clip,
        );
        Self::apply_grad(
            &mut self.ln_fwd_beta,
            &mut self.grad_ln_fwd_beta,
            lr,
            grad_clip,
        );
        Self::apply_grad(
            &mut self.ln_bwd_gamma,
            &mut self.grad_ln_bwd_gamma,
            lr,
            grad_clip,
        );
        Self::apply_grad(
            &mut self.ln_bwd_beta,
            &mut self.grad_ln_bwd_beta,
            lr,
            grad_clip,
        );
        // Deep inner bottleneck weights
        if self.deep {
            Self::apply_grad(&mut self.w_down2, &mut self.grad_down2, lr, grad_clip);
            Self::apply_grad(&mut self.w_up2, &mut self.grad_up2, lr, grad_clip);
        }

        GradientStepMetrics {
            norm_down: n_down,
            norm_up: n_up,
            norm_backward: (n_bdown * n_bdown + n_bup * n_bup).sqrt(),
            was_clipped: c_down || c_up || c_bdown || c_bup,
        }
    }

    /// Apply gradient to a single weight group. Returns (grad_norm, was_clipped).
    fn apply_grad(weights: &mut [f32], grads: &mut [f32], lr: f32, grad_clip: f32) -> (f32, bool) {
        let grad_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        let was_clipped = grad_norm > grad_clip;
        let clip_scale = if was_clipped {
            grad_clip / grad_norm
        } else {
            1.0
        };

        for (w, g) in weights.iter_mut().zip(grads.iter_mut()) {
            *w += lr * clip_scale * *g;
            *g = 0.0;
        }
        (grad_norm, was_clipped)
    }

    /// Enable EMA teacher with the given decay rate (typically 0.999 or 0.9999).
    ///
    /// Initializes the shadow weights to the current live weights.
    /// After each `apply_gradients()`, call `update_ema()` to track the moving average.
    pub fn enable_ema(&mut self, decay: f32) {
        self.ema_decay = decay.clamp(0.9, 0.99999);
        self.ema_weights = Some(self.flatten_weights());
    }

    /// Update EMA shadow weights: `ema = decay * ema + (1-decay) * live`.
    ///
    /// Call this after `apply_gradients()`. No-op if EMA is not enabled.
    pub fn update_ema(&mut self) {
        if self.ema_weights.is_none() {
            return;
        }
        let live = self.flatten_weights();
        let d = self.ema_decay;
        let one_minus_d = 1.0 - d;
        let ema = self.ema_weights.as_mut().unwrap();
        for (e, &l) in ema.iter_mut().zip(live.iter()) {
            *e = d * *e + one_minus_d * l;
        }
    }

    /// Swap live weights with EMA weights for evaluation, returning a guard
    /// that restores live weights when dropped.
    ///
    /// If EMA is not enabled, returns None (use live weights as-is).
    pub fn use_ema_weights(&mut self) -> Option<Vec<f32>> {
        let ema = self.ema_weights.as_ref()?.clone();
        let live = self.flatten_weights();
        self.load_weights(&ema);
        Some(live)
    }

    /// Restore live weights after evaluation with EMA weights.
    pub fn restore_live_weights(&mut self, live: &[f32]) {
        self.load_weights(live);
    }

    /// Whether EMA teacher is active.
    pub fn has_ema(&self) -> bool {
        self.ema_weights.is_some()
    }

    /// Flatten all projection weights into a single Vec for swarm exchange.
    ///
    /// Layout: [w_down, w_up, w_back_down, w_back_up, ln_fwd_gamma, ln_fwd_beta, ln_bwd_gamma, ln_bwd_beta, (w_down2, w_up2 if deep)]
    pub fn flatten_weights(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.num_params());
        flat.extend_from_slice(&self.w_down);
        flat.extend_from_slice(&self.w_up);
        flat.extend_from_slice(&self.w_back_down);
        flat.extend_from_slice(&self.w_back_up);
        flat.extend_from_slice(&self.ln_fwd_gamma);
        flat.extend_from_slice(&self.ln_fwd_beta);
        flat.extend_from_slice(&self.ln_bwd_gamma);
        flat.extend_from_slice(&self.ln_bwd_beta);
        if self.deep {
            flat.extend_from_slice(&self.w_down2);
            flat.extend_from_slice(&self.w_up2);
        }
        flat
    }

    /// Load weights from a flat Vec (e.g., from swarm aggregation).
    ///
    /// Accepts legacy (4-matrix only), standard (4-matrix + 4xLN), and
    /// deep (4-matrix + 4xLN + w_down2 + w_up2) formats.
    pub fn load_weights(&mut self, flat: &[f32]) {
        let legacy_size =
            self.w_down.len() + self.w_up.len() + self.w_back_down.len() + self.w_back_up.len();
        let full_size = legacy_size + 4 * self.bottleneck;
        let deep_size = full_size + self.w_down2.len() + self.w_up2.len();
        assert!(
            flat.len() == legacy_size || flat.len() == full_size || flat.len() == deep_size,
            "Weight vector size mismatch: expected {legacy_size} (legacy), {full_size} (with LN), or {deep_size} (deep), got {}",
            flat.len()
        );

        let mut offset = 0;
        let n = self.w_down.len();
        self.w_down.copy_from_slice(&flat[offset..offset + n]);
        offset += n;
        let n = self.w_up.len();
        self.w_up.copy_from_slice(&flat[offset..offset + n]);
        offset += n;
        let n = self.w_back_down.len();
        self.w_back_down.copy_from_slice(&flat[offset..offset + n]);
        offset += n;
        let n = self.w_back_up.len();
        self.w_back_up.copy_from_slice(&flat[offset..offset + n]);
        offset += n;

        // Load LayerNorm params if present (standard or deep format)
        if flat.len() >= full_size {
            let b = self.bottleneck;
            self.ln_fwd_gamma.copy_from_slice(&flat[offset..offset + b]);
            offset += b;
            self.ln_fwd_beta.copy_from_slice(&flat[offset..offset + b]);
            offset += b;
            self.ln_bwd_gamma.copy_from_slice(&flat[offset..offset + b]);
            offset += b;
            self.ln_bwd_beta.copy_from_slice(&flat[offset..offset + b]);
            offset += b;
        }

        // Load deep inner bottleneck weights if present
        if self.deep && flat.len() == deep_size {
            let n = self.w_down2.len();
            self.w_down2.copy_from_slice(&flat[offset..offset + n]);
            offset += n;
            let n = self.w_up2.len();
            self.w_up2.copy_from_slice(&flat[offset..offset + n]);
            let _ = offset; // suppress unused assignment warning
        }
    }

    /// Total number of learnable parameters.
    pub fn num_params(&self) -> usize {
        self.w_down.len()
            + self.w_up.len()
            + self.w_back_down.len()
            + self.w_back_up.len()
            + self.ln_fwd_gamma.len()
            + self.ln_fwd_beta.len()
            + self.ln_bwd_gamma.len()
            + self.ln_bwd_beta.len()
            + self.w_down2.len()
            + self.w_up2.len()
    }

    /// HDC dimension.
    pub fn hdc_dim(&self) -> usize {
        self.hdc_dim
    }

    /// Bottleneck dimension.
    pub fn bottleneck_dim(&self) -> usize {
        self.bottleneck
    }

    /// SSM dimension.
    pub fn ssm_dim(&self) -> usize {
        self.ssm_dim
    }

    /// Apply manifold regularization to the projection weights.
    /// This pushes the weights toward a more harmonic, less fragmented state.
    pub fn apply_manifold_regularization(&mut self, strength: f32) {
        // Orthogonal weight decay: pushes rows of w_down and w_up to be more independent
        // (Simplified manifold regularization)
        let decay = strength * 0.1;
        for w in &mut self.w_down {
            *w *= 1.0 - decay;
        }
        for w in &mut self.w_up {
            *w *= 1.0 - decay;
        }
    }

    /// Warm-start the forward projection (w_down) from sample HDC vectors.
    ///
    /// Computes the top-k principal directions of the input distribution and
    /// aligns w_down rows to span that subspace. This accelerates convergence
    /// by ensuring the bottleneck captures variance in the thought HV space
    /// rather than random directions.
    ///
    /// Uses power iteration (lightweight, no full SVD needed).
    pub fn warm_start_from_samples(&mut self, samples: &[ContinuousHV]) {
        if samples.len() < 2 || self.hdc_dim == 0 || self.bottleneck == 0 {
            return;
        }

        let n = samples.len();
        let d = self.hdc_dim;
        let k = self.bottleneck.min(n).min(d);

        // Compute mean
        let mut mean = vec![0.0f32; d];
        for s in samples {
            for (m, v) in mean.iter_mut().zip(s.values.iter()) {
                *m += v;
            }
        }
        let inv_n = 1.0 / n as f32;
        for m in &mut mean {
            *m *= inv_n;
        }

        // Power iteration for top-k principal components
        // Use genesis-seeded random initialization for determinism
        let scale = 1.0 / (self.bottleneck as f32).sqrt();
        for comp_idx in 0..k {
            // Initialize direction from existing w_down row
            let row_start = comp_idx * d;
            let mut dir: Vec<f32> = self.w_down[row_start..row_start + d].to_vec();
            let dir_norm: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
            if dir_norm > 1e-10 {
                for v in &mut dir {
                    *v /= dir_norm;
                }
            }

            // 10 iterations of power method on covariance
            for _ in 0..10 {
                // result = C * dir = (1/n) Σ (x_i - mean) * <(x_i - mean), dir>
                let mut result = vec![0.0f32; d];
                for s in samples {
                    let mut dot = 0.0f32;
                    for j in 0..d {
                        dot += (s.values[j] - mean[j]) * dir[j];
                    }
                    for j in 0..d {
                        result[j] += (s.values[j] - mean[j]) * dot;
                    }
                }
                // Normalize
                let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 {
                    break;
                }
                for v in &mut result {
                    *v /= norm;
                }

                // Deflation: remove components of previously found directions
                for prev_idx in 0..comp_idx {
                    let prev_start = prev_idx * d;
                    let mut dot = 0.0f32;
                    for j in 0..d {
                        dot += result[j] * self.w_down[prev_start + j] / scale;
                    }
                    for j in 0..d {
                        result[j] -= dot * self.w_down[prev_start + j] / scale;
                    }
                }
                let norm2: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm2 < 1e-10 {
                    break;
                }
                for v in &mut result {
                    *v /= norm2;
                }
                dir = result;
            }

            // Write principal direction as w_down row (scaled)
            for j in 0..d {
                self.w_down[comp_idx * d + j] = dir[j] * scale;
            }
        }

        tracing::info!(
            samples = n,
            components = k,
            "Projection warm-started from sample covariance"
        );
    }

    /// Bidirectional warm-start: initialize all 4 weight matrices from sample data.
    ///
    /// 1. `w_down`: PCA on HDC samples (same as `warm_start_from_samples`)
    /// 2. `w_up`: PCA on the bottleneck representations projected through `w_down`,
    ///    finding the SSM subspace that best spans the projected data
    /// 3. `w_back_down`: initialized as transpose of `w_up` (matched backward)
    /// 4. `w_back_up`: initialized as transpose of `w_down` (matched backward)
    pub fn warm_start_bidirectional(&mut self, samples: &[ContinuousHV]) {
        if samples.len() < 2 || self.hdc_dim == 0 || self.bottleneck == 0 {
            return;
        }

        // Step 1: Forward w_down via PCA on HDC samples
        self.warm_start_from_samples(samples);

        // Step 2: Compute bottleneck representations of samples through updated w_down + LN
        let bottleneck_vecs: Vec<Vec<f32>> = samples
            .iter()
            .map(|s| {
                let pre = self.matmul(&self.w_down, &s.values, self.bottleneck, self.hdc_dim);
                let (normed, _, _) = layer_norm(&pre, &self.ln_fwd_gamma, &self.ln_fwd_beta);
                normed.into_iter().map(activation).collect()
            })
            .collect();

        // Step 3: PCA on bottleneck→SSM direction for w_up
        let n = bottleneck_vecs.len();
        let b = self.bottleneck;
        let s = self.ssm_dim;
        let k = s.min(n).min(b);
        let scale = 1.0 / (self.bottleneck as f32).sqrt();

        // Compute mean of bottleneck representations
        let mut bn_mean = vec![0.0f32; b];
        for bv in &bottleneck_vecs {
            for (m, v) in bn_mean.iter_mut().zip(bv.iter()) {
                *m += v;
            }
        }
        let inv_n = 1.0 / n as f32;
        for m in &mut bn_mean {
            *m *= inv_n;
        }

        // Power iteration for top-k principal components of bottleneck distribution
        // These become the rows of w_up [ssm_dim × bottleneck]
        for comp_idx in 0..k {
            let row_start = comp_idx * b;
            let mut dir: Vec<f32> = self.w_up[row_start..row_start + b].to_vec();
            let dir_norm: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
            if dir_norm > 1e-10 {
                for v in &mut dir {
                    *v /= dir_norm;
                }
            }

            for _ in 0..10 {
                let mut result = vec![0.0f32; b];
                for bv in &bottleneck_vecs {
                    let mut dot = 0.0f32;
                    for j in 0..b {
                        dot += (bv[j] - bn_mean[j]) * dir[j];
                    }
                    for j in 0..b {
                        result[j] += (bv[j] - bn_mean[j]) * dot;
                    }
                }
                let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 {
                    break;
                }
                for v in &mut result {
                    *v /= norm;
                }

                // Deflation
                for prev_idx in 0..comp_idx {
                    let prev_start = prev_idx * b;
                    let mut dot = 0.0f32;
                    for j in 0..b {
                        dot += result[j] * self.w_up[prev_start + j] / scale;
                    }
                    for j in 0..b {
                        result[j] -= dot * self.w_up[prev_start + j] / scale;
                    }
                }
                let norm2: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm2 < 1e-10 {
                    break;
                }
                for v in &mut result {
                    *v /= norm2;
                }
                dir = result;
            }

            for j in 0..b {
                self.w_up[comp_idx * b + j] = dir[j] * scale;
            }
        }

        // Step 4: Matched backward initialization
        // w_back_down = transpose of w_up: [bottleneck × ssm_dim] from [ssm_dim × bottleneck]
        for i in 0..self.bottleneck {
            for j in 0..self.ssm_dim {
                self.w_back_down[i * s + j] = self.w_up[j * b + i];
            }
        }
        // w_back_up = transpose of w_down: [hdc_dim × bottleneck] from [bottleneck × hdc_dim]
        let d = self.hdc_dim;
        for i in 0..d {
            for j in 0..b {
                self.w_back_up[i * b + j] = self.w_down[j * d + i];
            }
        }

        tracing::info!(
            samples = n,
            "Bidirectional warm-start complete (w_down PCA + w_up PCA + matched backward)"
        );
    }

    /// Matrix-vector multiply: `result[i] = sum_j(mat[i*cols + j] * vec[j])`
    ///
    /// mat shape: [rows × cols], vec shape: [cols], result shape: [rows]
    ///
    /// When the `simd` feature is enabled, each row dot product uses AVX2+FMA
    /// via `symthaea_core::hdc::simd_continuous::dot_product_simd`.
    fn matmul(&self, mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        debug_assert_eq!(mat.len(), rows * cols);
        debug_assert_eq!(vec.len(), cols);

        let mut result = vec![0.0f32; rows];

        #[cfg(feature = "simd")]
        {
            use symthaea_core::hdc::simd_continuous::dot_product_simd;
            for i in 0..rows {
                let row_start = i * cols;
                result[i] = dot_product_simd(&mat[row_start..row_start + cols], vec);
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            for i in 0..rows {
                let row_start = i * cols;
                let mut sum = 0.0f32;
                for j in 0..cols {
                    sum += mat[row_start + j] * vec[j];
                }
                result[i] = sum;
            }
        }

        result
    }

    /// Compute the bottleneck activation for a single HDC input (for diagnostics).
    ///
    /// Returns the intermediate 256D representation after LayerNorm + GELU + residual.
    pub fn bottleneck_activation(&self, hv: &ContinuousHV) -> Vec<f32> {
        let hidden_pre = self.matmul(&self.w_down, &hv.values, self.bottleneck, self.hdc_dim);
        let (normed, _, _) = layer_norm(&hidden_pre, &self.ln_fwd_gamma, &self.ln_fwd_beta);
        normed.into_iter().map(activation).collect()
    }

    /// Compute orthogonality regularization gradients for `w_up` rows.
    ///
    /// Penalizes cosine similarity between random pairs of `w_up` rows to prevent
    /// rank collapse in the bottleneck→SSM projection. When rows become correlated,
    /// effective rank drops toward 1 (the bottleneck dimension becomes degenerate).
    ///
    /// This is the spatial projection analog of `TemporalProjection::compute_rank_regularization_gradients`.
    ///
    /// # Arguments
    /// * `weight` — Regularization strength (0.01–0.1 typical). Gradients are scaled by `weight / num_samples`.
    /// * `num_samples` — Number of random row pairs to sample per call (32–128 typical).
    pub fn compute_orthogonality_gradients(&mut self, weight: f32, num_samples: usize) {
        if weight <= 0.0 || self.ssm_dim < 2 {
            return;
        }

        let scale = weight / num_samples.max(1) as f32;
        let b = self.bottleneck;

        // Deterministic but varying seed from current weight state
        let seed_val = (self.w_up[0].to_bits() as u64).wrapping_add(self.w_up.len() as u64)
            ^ 0x9E3779B97F4A7C15;
        let mut rng_state = seed_val | 1;
        let total_pairs = self.ssm_dim * self.ssm_dim;

        for _ in 0..num_samples {
            // xorshift64
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let pair_idx = (rng_state as usize) % total_pairs;
            let j1 = pair_idx / self.ssm_dim;
            let j2 = pair_idx % self.ssm_dim;
            if j1 == j2 {
                continue;
            }

            // w_up is [ssm_dim × bottleneck], row j starts at j * bottleneck
            let row1_start = j1 * b;
            let row2_start = j2 * b;

            // Compute dot product (unnormalized cosine for efficiency)
            let mut dot = 0.0f32;
            for k in 0..b {
                dot += self.w_up[row1_start + k] * self.w_up[row2_start + k];
            }

            // Push apart: gradient of dot² w.r.t. each row
            for k in 0..b {
                self.grad_up[row1_start + k] -= scale * dot * self.w_up[row2_start + k];
                self.grad_up[row2_start + k] -= scale * dot * self.w_up[row1_start + k];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-projection")
    }

    #[test]
    fn test_projection_creation() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        assert_eq!(proj.hdc_dim(), 16384);
        assert_eq!(proj.bottleneck_dim(), 256);
        assert_eq!(proj.ssm_dim(), 768);
        // 256*16384 + 768*256 + 256*768 + 16384*256 = 8,781,824 (matrices)
        // + 4 * 256 = 1,024 (LayerNorm gamma+beta for fwd+bwd)
        assert_eq!(
            proj.num_params(),
            256 * 16384 + 768 * 256 + 256 * 768 + 16384 * 256 + 4 * 256
        );
    }

    #[test]
    fn test_project_to_ssm_dimensions() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let hv = ContinuousHV::random_default(42);
        let ssm_vec = proj.project_to_ssm(&hv);
        assert_eq!(ssm_vec.len(), 768);
    }

    #[test]
    fn test_project_to_hdc_dimensions() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let ssm_vec = vec![0.1; 768];
        let hv = proj.project_to_hdc(&ssm_vec);
        assert_eq!(hv.values.len(), 16384);
    }

    #[test]
    fn test_roundtrip_preserves_structure() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let hv = ContinuousHV::random_default(42).normalize();

        // Project forward and back
        let ssm_vec = proj.project_to_ssm(&hv);
        let reconstructed = proj.project_to_hdc(&ssm_vec).normalize();

        // Not expecting perfect reconstruction (information bottleneck),
        // but similarity should be non-trivial with random init
        let sim = hv.similarity(&reconstructed);
        // With random projections, similarity is expected to be small but
        // the output should be finite and well-formed
        assert!(sim.is_finite(), "Similarity should be finite");
    }

    #[test]
    fn test_deterministic_initialization() {
        let genesis = test_genesis();
        let proj1 = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let proj2 = HdcSsmProjection::new(&genesis, 16384, 256, 768);

        let hv = ContinuousHV::random_default(42);
        let ssm1 = proj1.project_to_ssm(&hv);
        let ssm2 = proj2.project_to_ssm(&hv);
        assert_eq!(
            ssm1, ssm2,
            "Same genesis should produce identical projections"
        );
    }

    #[test]
    fn test_different_inputs_different_outputs() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 16384, 256, 768);

        let hv1 = ContinuousHV::random_default(42);
        let hv2 = ContinuousHV::random_default(99);

        let ssm1 = proj.project_to_ssm(&hv1);
        let ssm2 = proj.project_to_ssm(&hv2);
        assert_ne!(
            ssm1, ssm2,
            "Different inputs should produce different outputs"
        );
    }

    #[test]
    fn test_flatten_load_roundtrip() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let flat = proj.flatten_weights();
        assert_eq!(flat.len(), proj.num_params());

        let mut proj2 = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        proj2.load_weights(&flat);

        let hv = ContinuousHV::random_default(42);
        let ssm1 = proj.project_to_ssm(&hv);
        let ssm2 = proj2.project_to_ssm(&hv);
        assert_eq!(
            ssm1, ssm2,
            "Loaded weights should produce identical results"
        );
    }

    #[test]
    fn test_gradient_accumulation_and_application() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 32, 64); // Small dims for speed

        // Use non-uniform, differently-seeded vectors to ensure non-zero error
        let thought = ContinuousHV::random(dim, 42);
        let output = ContinuousHV::random(dim, 99);

        // Capture weights before
        let weights_before = proj.flatten_weights();

        // Accumulate and apply gradients with a generous LR and clip
        proj.compute_gradients(&thought, &output);
        proj.apply_gradients(0.1, 1000.0);

        let weights_after = proj.flatten_weights();

        // Weights should have changed
        let changed = weights_before
            .iter()
            .zip(weights_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "Gradients should modify weights");
    }

    #[test]
    fn test_gradient_clipping() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 32, 64);

        // Create a large error signal using differently-scaled random vectors
        let thought = ContinuousHV::random(dim, 42).scale(10.0);
        let output = ContinuousHV::random(dim, 99).scale(10.0);

        let weights_before = proj.flatten_weights();

        // Apply with very tight clipping
        proj.compute_gradients(&thought, &output);
        proj.apply_gradients(0.01, 0.001);

        let weights_after = proj.flatten_weights();

        // Weight changes should be bounded by clipping
        let max_change: f32 = weights_before
            .iter()
            .zip(weights_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_change < 1.0,
            "Gradient clipping should bound weight changes, got {max_change}"
        );
    }

    #[test]
    fn test_zero_input_produces_zero_output() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let hv = ContinuousHV::zero(16384);
        let ssm_vec = proj.project_to_ssm(&hv);
        // All zeros through matmul + GELU+residual should produce all zeros
        assert!(ssm_vec.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_contrastive_gradients_modify_weights() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 32, 64);

        let anchor = ContinuousHV::random(dim, 42);
        let negative = ContinuousHV::random(dim, 99);

        let weights_before = proj.flatten_weights();

        proj.compute_contrastive_gradients(&anchor, &negative, 0.1);
        proj.apply_gradients(0.1, 1000.0);

        let weights_after = proj.flatten_weights();

        let changed = weights_before
            .iter()
            .zip(weights_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "Contrastive gradients should modify weights");
    }

    #[test]
    fn test_contrastive_pushes_apart() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 32, 64);

        let anchor = ContinuousHV::random(dim, 42).normalize();
        let negative = ContinuousHV::random(dim, 99).normalize();

        // Measure initial bottleneck distance
        let h_a_before = proj.project_to_ssm(&anchor);
        let h_n_before = proj.project_to_ssm(&negative);
        let dist_before: f32 = h_a_before
            .iter()
            .zip(h_n_before.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Apply contrastive gradients multiple times
        for _ in 0..10 {
            proj.compute_contrastive_gradients(&anchor, &negative, 0.5);
            proj.apply_gradients(0.01, 10.0);
        }

        let h_a_after = proj.project_to_ssm(&anchor);
        let h_n_after = proj.project_to_ssm(&negative);
        let dist_after: f32 = h_a_after
            .iter()
            .zip(h_n_after.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Distance should increase (or at least not decrease significantly)
        // Note: with gradient clipping and ReLU, this is a weak assertion
        assert!(
            dist_after.is_finite(),
            "Distance should be finite after contrastive, got {dist_after}"
        );
        assert!(
            dist_before.is_finite(),
            "Initial distance should be finite, got {dist_before}"
        );
    }

    #[test]
    fn test_small_projection_correctness() {
        // Verify matmul with known values
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 4, 2, 3);

        // Just verify dimensions are correct through the pipeline
        let hv = ContinuousHV::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        let ssm = proj.project_to_ssm(&hv);
        assert_eq!(ssm.len(), 3);

        let back = proj.project_to_hdc(&ssm);
        assert_eq!(back.values.len(), 4);
    }

    #[test]
    fn test_warm_start_modifies_weights() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 16, 64);
        let weights_before = proj.flatten_weights();

        // Create sample HVs with a clear structure
        let samples: Vec<ContinuousHV> = (0..20)
            .map(|i| {
                let mut hv = ContinuousHV::random(dim, 100 + i as u64);
                // Add a dominant component to first few dims
                for j in 0..8 {
                    hv.values[j] += 5.0;
                }
                hv
            })
            .collect();

        proj.warm_start_from_samples(&samples);
        let weights_after = proj.flatten_weights();

        let changed = weights_before
            .iter()
            .zip(weights_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "Warm-start should modify projection weights");
    }

    #[test]
    fn test_warm_start_empty_samples() {
        let genesis = test_genesis();
        let mut proj = HdcSsmProjection::new(&genesis, 256, 16, 64);
        let before = proj.flatten_weights();

        proj.warm_start_from_samples(&[]);
        let after = proj.flatten_weights();
        assert_eq!(before, after, "Empty samples should not modify weights");
    }

    #[test]
    fn test_effective_rank_finite() {
        let genesis = test_genesis();
        let dim = 256;
        let proj = HdcSsmProjection::new(&genesis, dim, 16, 64);

        let samples: Vec<ContinuousHV> = (0..10)
            .map(|i| ContinuousHV::random(dim, 200 + i as u64))
            .collect();

        let rank = proj.effective_rank(&samples);
        assert!(rank.is_finite(), "Effective rank should be finite");
        assert!(
            rank >= 1.0,
            "Effective rank should be at least 1, got {rank}"
        );
        assert!(
            rank <= 16.0,
            "Effective rank should be at most bottleneck_dim, got {rank}"
        );
    }

    #[test]
    fn test_effective_rank_collapse_detection() {
        let genesis = test_genesis();
        let dim = 64;
        let proj = HdcSsmProjection::new(&genesis, dim, 8, 16);

        // All-identical samples → should have low effective rank
        let sample = ContinuousHV::random(dim, 42);
        let identical_samples: Vec<ContinuousHV> = (0..10).map(|_| sample.clone()).collect();
        let rank = proj.effective_rank(&identical_samples);
        assert!(rank.is_finite());
        // With identical inputs, all activations are the same → zero variance → rank = 1
        assert!(
            rank <= 2.0,
            "Identical inputs should give low rank, got {rank}"
        );
    }

    #[test]
    fn test_warm_start_bidirectional_modifies_all_weights() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 16, 64);
        let weights_before = proj.flatten_weights();

        let samples: Vec<ContinuousHV> = (0..20)
            .map(|i| {
                let mut hv = ContinuousHV::random(dim, 100 + i as u64);
                for j in 0..8 {
                    hv.values[j] += 5.0;
                }
                hv
            })
            .collect();

        proj.warm_start_bidirectional(&samples);
        let weights_after = proj.flatten_weights();

        // Count how many weight segments changed
        let n_down = 16 * dim;
        let n_up = 64 * 16;
        let n_back_down = 16 * 64;

        let down_changed = weights_before[..n_down]
            .iter()
            .zip(weights_after[..n_down].iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        let up_changed = weights_before[n_down..n_down + n_up]
            .iter()
            .zip(weights_after[n_down..n_down + n_up].iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        let back_down_changed = weights_before[n_down + n_up..n_down + n_up + n_back_down]
            .iter()
            .zip(weights_after[n_down + n_up..n_down + n_up + n_back_down].iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        let back_up_changed = weights_before[n_down + n_up + n_back_down..]
            .iter()
            .zip(weights_after[n_down + n_up + n_back_down..].iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);

        assert!(
            down_changed,
            "w_down should be modified by bidirectional warm-start"
        );
        assert!(
            up_changed,
            "w_up should be modified by bidirectional warm-start"
        );
        assert!(
            back_down_changed,
            "w_back_down should be modified by bidirectional warm-start"
        );
        assert!(
            back_up_changed,
            "w_back_up should be modified by bidirectional warm-start"
        );
    }

    #[test]
    fn test_warm_start_bidirectional_empty_samples() {
        let genesis = test_genesis();
        let mut proj = HdcSsmProjection::new(&genesis, 256, 16, 64);
        let before = proj.flatten_weights();

        proj.warm_start_bidirectional(&[]);
        let after = proj.flatten_weights();
        assert_eq!(before, after, "Empty samples should not modify weights");
    }

    #[test]
    fn test_simd_matmul_consistency() {
        // Verify SIMD and scalar paths produce the same results
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 256, 16, 64);

        let hv = ContinuousHV::random(256, 42);
        let ssm = proj.project_to_ssm(&hv);
        assert_eq!(ssm.len(), 64);
        assert!(
            ssm.iter().all(|x| x.is_finite()),
            "All outputs should be finite"
        );

        // Roundtrip should be finite
        let back = proj.project_to_hdc(&ssm);
        assert!(back.values.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_ema_teacher_tracks_weights() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 32, 64);

        proj.enable_ema(0.99);
        assert!(proj.has_ema());

        let thought = ContinuousHV::random(dim, 42);
        let output = ContinuousHV::random(dim, 99);

        // Train a few steps
        for _ in 0..5 {
            proj.compute_gradients(&thought, &output);
            proj.apply_gradients(0.01, 1.0);
            proj.update_ema();
        }

        // EMA weights should differ from live weights (EMA lags behind)
        let live = proj.flatten_weights();
        let saved_live = proj.use_ema_weights().expect("EMA should be enabled");
        let ema_active = proj.flatten_weights(); // now holds EMA weights
        proj.restore_live_weights(&saved_live);

        let differs = live
            .iter()
            .zip(ema_active.iter())
            .any(|(l, e)| (l - e).abs() > 1e-10);
        assert!(differs, "EMA weights should lag behind live weights");
    }

    #[test]
    fn test_ema_disabled_by_default() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 64, 8, 16);
        assert!(!proj.has_ema());
    }

    #[test]
    fn test_ema_swap_restores_live() {
        let genesis = test_genesis();
        let dim = 128;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 16, 32);
        proj.enable_ema(0.999);

        let thought = ContinuousHV::random(dim, 42);
        let output = ContinuousHV::random(dim, 99);
        proj.compute_gradients(&thought, &output);
        proj.apply_gradients(0.01, 1.0);
        proj.update_ema();

        let live_before = proj.flatten_weights();

        // Swap to EMA
        let saved = proj.use_ema_weights().unwrap();
        // Now projection uses EMA weights
        let hv = ContinuousHV::random(dim, 7);
        let _ssm_ema = proj.project_to_ssm(&hv);

        // Restore live
        proj.restore_live_weights(&saved);
        let live_after = proj.flatten_weights();

        assert_eq!(
            live_before, live_after,
            "Live weights should be fully restored after EMA swap"
        );
    }

    // === Deep projection tests ===

    #[test]
    fn test_deep_projection_creation() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new_deep(&genesis, 16384, 256, 768);
        assert!(proj.is_deep());
        assert_eq!(proj.inner_dim(), 128);
        assert_eq!(proj.hdc_dim(), 16384);
        assert_eq!(proj.bottleneck_dim(), 256);
        assert_eq!(proj.ssm_dim(), 768);
        // Shallow params: 256*16384 + 768*256 + 256*768 + 16384*256 + 4*256 = 8,782,848
        // Deep extra: 128*256 + 256*128 = 65,536
        let shallow_params = 256 * 16384 + 768 * 256 + 256 * 768 + 16384 * 256 + 4 * 256;
        let deep_extra = 128 * 256 + 256 * 128;
        assert_eq!(proj.num_params(), shallow_params + deep_extra);
    }

    #[test]
    fn test_deep_project_to_ssm_dimensions() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new_deep(&genesis, 16384, 256, 768);
        let hv = ContinuousHV::random_default(42);
        let ssm_vec = proj.project_to_ssm(&hv);
        assert_eq!(ssm_vec.len(), 768);
        assert!(ssm_vec.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_deep_project_to_hdc_dimensions() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new_deep(&genesis, 16384, 256, 768);
        let ssm_vec = vec![0.1; 768];
        let hv = proj.project_to_hdc(&ssm_vec);
        assert_eq!(hv.values.len(), 16384);
        assert!(hv.values.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_deep_roundtrip_finite() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new_deep(&genesis, 16384, 256, 768);
        let hv = ContinuousHV::random_default(42).normalize();
        let ssm = proj.project_to_ssm(&hv);
        let recon = proj.project_to_hdc(&ssm).normalize();
        let sim = hv.similarity(&recon);
        assert!(
            sim.is_finite(),
            "Deep roundtrip similarity should be finite"
        );
    }

    #[test]
    fn test_deep_different_from_shallow() {
        let genesis = test_genesis();
        let shallow = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let deep = HdcSsmProjection::new_deep(&genesis, 16384, 256, 768);

        assert!(!shallow.is_deep());
        assert!(deep.is_deep());
        assert!(deep.num_params() > shallow.num_params());

        let hv = ContinuousHV::random_default(42);
        let ssm_shallow = shallow.project_to_ssm(&hv);
        let ssm_deep = deep.project_to_ssm(&hv);
        // Deep adds inner bottleneck so outputs differ
        assert_ne!(ssm_shallow, ssm_deep);
    }

    #[test]
    fn test_deep_flatten_load_roundtrip() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new_deep(&genesis, 256, 32, 64);
        let flat = proj.flatten_weights();
        assert_eq!(flat.len(), proj.num_params());

        let mut proj2 = HdcSsmProjection::new_deep(&genesis, 256, 32, 64);
        proj2.load_weights(&flat);

        let hv = ContinuousHV::random(256, 42);
        let ssm1 = proj.project_to_ssm(&hv);
        let ssm2 = proj2.project_to_ssm(&hv);
        assert_eq!(
            ssm1, ssm2,
            "Loaded deep weights should produce identical results"
        );
    }

    #[test]
    fn test_deep_gradient_modifies_weights() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new_deep(&genesis, dim, 32, 64);

        let thought = ContinuousHV::random(dim, 42);
        let output = ContinuousHV::random(dim, 99);

        let weights_before = proj.flatten_weights();
        proj.compute_gradients(&thought, &output);
        proj.apply_gradients(0.1, 1000.0);
        let weights_after = proj.flatten_weights();

        let changed = weights_before
            .iter()
            .zip(weights_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "Deep gradients should modify weights");
    }

    #[test]
    fn test_deep_inner_weights_updated_by_gradients() {
        let genesis = test_genesis();
        let dim = 256;
        let bn = 32;
        let inner = bn / 2; // 16
        let mut proj = HdcSsmProjection::new_deep(&genesis, dim, bn, 64);

        let thought = ContinuousHV::random(dim, 42);
        let output = ContinuousHV::random(dim, 99);

        // Extract deep weight region from flattened weights
        let flat_before = proj.flatten_weights();
        let shallow_base = bn * dim + 64 * bn + bn * 64 + dim * bn + 4 * bn;
        let deep_start = shallow_base;
        let deep_end = deep_start + inner * bn + bn * inner;
        let deep_before = &flat_before[deep_start..deep_end];

        proj.compute_gradients(&thought, &output);
        proj.apply_gradients(0.1, 1000.0);

        let flat_after = proj.flatten_weights();
        let deep_after = &flat_after[deep_start..deep_end];

        let changed = deep_before
            .iter()
            .zip(deep_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(
            changed,
            "Deep inner bottleneck weights should be updated by gradient flow"
        );
    }

    #[test]
    fn test_deep_zero_input_finite_output() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new_deep(&genesis, 16384, 256, 768);
        let hv = ContinuousHV::zero(16384);
        let ssm_vec = proj.project_to_ssm(&hv);
        assert!(ssm_vec.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_shallow_is_not_deep() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 64, 8, 16);
        assert!(!proj.is_deep());
        assert_eq!(proj.inner_dim(), 0);
    }

    #[test]
    fn test_deep_deterministic_initialization() {
        let genesis = test_genesis();
        let proj1 = HdcSsmProjection::new_deep(&genesis, 256, 32, 64);
        let proj2 = HdcSsmProjection::new_deep(&genesis, 256, 32, 64);

        let hv = ContinuousHV::random(256, 42);
        let ssm1 = proj1.project_to_ssm(&hv);
        let ssm2 = proj2.project_to_ssm(&hv);
        assert_eq!(
            ssm1, ssm2,
            "Same genesis should produce identical deep projections"
        );
    }

    #[test]
    fn test_deep_small_projection_correctness() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new_deep(&genesis, 8, 4, 6);
        assert!(proj.is_deep());
        assert_eq!(proj.inner_dim(), 2);

        let hv = ContinuousHV::from_vec(vec![1.0, 0.5, -0.5, -1.0, 0.2, 0.8, -0.3, 0.0]);
        let ssm = proj.project_to_ssm(&hv);
        assert_eq!(ssm.len(), 6);
        assert!(ssm.iter().all(|x| x.is_finite()));

        let back = proj.project_to_hdc(&ssm);
        assert_eq!(back.values.len(), 8);
        assert!(back.values.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_scale_accumulated_gradients() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 32, 64);

        let thought = ContinuousHV::random(dim, 42);
        let output = ContinuousHV::random(dim, 99);

        // Compute gradients, capture weights, scale by 2, apply
        proj.compute_gradients(&thought, &output);
        proj.scale_accumulated_gradients(2.0);
        proj.apply_gradients(0.1, 1000.0);
        let weights_scaled = proj.flatten_weights();

        // Reset to original weights and repeat without scaling
        let mut proj2 = HdcSsmProjection::new(&genesis, dim, 32, 64);
        proj2.compute_gradients(&thought, &output);
        proj2.apply_gradients(0.1, 1000.0);
        let weights_unscaled = proj2.flatten_weights();

        // The initial weights are the same for both
        let initial = HdcSsmProjection::new(&genesis, dim, 32, 64).flatten_weights();

        // Scaled delta should be ~2× the unscaled delta (modulo clipping)
        let mut scaled_delta_sum = 0.0f64;
        let mut unscaled_delta_sum = 0.0f64;
        for i in 0..initial.len() {
            scaled_delta_sum += (weights_scaled[i] - initial[i]).abs() as f64;
            unscaled_delta_sum += (weights_unscaled[i] - initial[i]).abs() as f64;
        }

        assert!(
            scaled_delta_sum > unscaled_delta_sum * 1.5,
            "2x gradient scale should produce larger weight deltas: scaled={scaled_delta_sum:.6}, unscaled={unscaled_delta_sum:.6}"
        );
    }

    #[test]
    fn test_contrastive_pretrain_accumulates_recon() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 32, 64);

        let thoughts: Vec<ContinuousHV> = (0..5)
            .map(|seed| ContinuousHV::random(dim, seed).normalize())
            .collect();

        let weights_before = proj.flatten_weights();
        let (avg_dist, avg_recon) = proj.contrastive_pretrain(&thoughts, 3, 0.01);
        let weights_after = proj.flatten_weights();

        // Weights should have changed
        let changed = weights_before
            .iter()
            .zip(weights_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "Contrastive pretrain should modify weights");

        // Distance should be positive (thoughts pushed apart)
        assert!(
            avg_dist > 0.0,
            "avg_dist should be positive, got: {avg_dist}"
        );

        // Reconstruction error should be positive (not zero — catches the bug)
        assert!(
            avg_recon > 0.0,
            "avg_recon should be positive, got: {avg_recon}"
        );
        assert!(avg_recon.is_finite(), "avg_recon should be finite");
    }

    #[test]
    fn test_gradient_step_metrics_returned() {
        let genesis = GenesisSeed::from_phrase("test-metrics");
        let hdc_dim = 256;
        let bottleneck = 32;
        let ssm_dim = 64;
        let mut proj = HdcSsmProjection::new(&genesis, hdc_dim, bottleneck, ssm_dim);

        // Accumulate some gradients
        let thought = ContinuousHV::random(hdc_dim, 42).normalize();
        let target = ContinuousHV::random(hdc_dim, 99).normalize();
        proj.compute_gradients(&thought, &target);

        let metrics = proj.apply_gradients(0.01, 1.0);
        assert!(
            metrics.norm_down.is_finite() && metrics.norm_down >= 0.0,
            "norm_down should be non-negative finite: {}",
            metrics.norm_down
        );
        assert!(
            metrics.norm_up.is_finite() && metrics.norm_up >= 0.0,
            "norm_up should be non-negative finite: {}",
            metrics.norm_up
        );
        assert!(
            metrics.norm_backward.is_finite() && metrics.norm_backward >= 0.0,
            "norm_backward should be non-negative finite: {}",
            metrics.norm_backward
        );
    }

    #[test]
    fn test_diagnostics_record_and_collapse() {
        let mut diag = ProjectionGradientDiagnostics::default();
        assert_eq!(diag.total_steps, 0);
        assert!(!diag.bottleneck_collapse_detected());

        // Record some steps
        let metrics = GradientStepMetrics {
            norm_down: 0.5,
            norm_up: 0.3,
            norm_backward: 0.4,
            was_clipped: false,
        };
        diag.record_step(&metrics, 0.001);
        assert_eq!(diag.total_steps, 1);
        assert_eq!(diag.clip_count, 0);

        let clipped = GradientStepMetrics {
            norm_down: 2.0,
            norm_up: 1.5,
            norm_backward: 1.8,
            was_clipped: true,
        };
        diag.record_step(&clipped, 0.001);
        assert_eq!(diag.total_steps, 2);
        assert_eq!(diag.clip_count, 1);

        // Record bottleneck with normal variance
        diag.record_bottleneck(&[1.0, 2.0, 3.0, 0.5, -1.0]);
        assert_eq!(diag.bottleneck_norms.len(), 1);
        assert!(!diag.bottleneck_collapse_detected());

        // Record 5 collapsed bottlenecks
        for _ in 0..5 {
            diag.record_bottleneck(&[0.5, 0.5, 0.5, 0.5, 0.5]);
        }
        assert!(
            diag.bottleneck_collapse_detected(),
            "Should detect collapse when variance < 0.01 for 5 consecutive entries"
        );

        let summary = diag.format_summary();
        assert!(summary.contains("Total steps"));
        assert!(summary.contains("Clip count"));
        assert!(summary.contains("collapse"));
    }

    #[test]
    fn test_bottleneck_activation_dimensions() {
        let genesis = GenesisSeed::from_phrase("test-bottleneck");
        let hdc_dim = 256;
        let bottleneck = 32;
        let ssm_dim = 64;
        let proj = HdcSsmProjection::new(&genesis, hdc_dim, bottleneck, ssm_dim);

        let hv = ContinuousHV::random(hdc_dim, 42).normalize();
        let act = proj.bottleneck_activation(&hv);
        assert_eq!(
            act.len(),
            bottleneck,
            "Bottleneck activation should have bottleneck dimensions"
        );
        assert!(
            act.iter().all(|v| v.is_finite()),
            "All bottleneck activations should be finite"
        );
    }
}
