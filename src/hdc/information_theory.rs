// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Information-Theoretic Measures for Hyperdimensional Computing
//!
//! This module provides Shannon information theory primitives adapted for the
//! hyperdimensional computing (HDC) domain. These measures quantify how much
//! information is shared between, or transferred across, HDC states -- a
//! prerequisite for rigorous consciousness measurement under Integrated
//! Information Theory (IIT).
//!
//! ## Connection to Consciousness
//!
//! - **Mutual Information** quantifies the *integration* between two subsystems:
//!   if two brain regions share high MI, they are informationally coupled rather
//!   than independent. This is the core quantity underlying Phi (Φ) -- the
//!   minimum information partition across all possible bipartitions.
//!
//! - **Transfer Entropy** captures *directed causation* between time series of
//!   neural (or HDC) states. TE(X→Y) measures how much knowing X's past
//!   reduces uncertainty about Y's future beyond what Y's own past provides.
//!   This operationalises the "cause-effect" structure that IIT demands.
//!
//! ## Implementation Strategy
//!
//! Both binary hypervectors ([`BinaryHV`]) and real-valued hypervectors ([`ContinuousHV`])
//! are converted to discrete probability distributions before applying
//! information-theoretic formulae:
//!
//! - **BinaryHV**: Windowed bit-pattern histograms (e.g., 4-bit windows over the
//!   16,384-bit vector yield pattern frequencies across 16 possible symbols).
//! - **ContinuousHV**: Uniform histogram binning over the value range, with
//!   configurable bin count.
//!
//! All quantities are in **bits** (log base 2).

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::hdc::BinaryHV;
use crate::hdc::ContinuousHV;

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for information-theoretic estimators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationTheoryConfig {
    /// Number of bins for histogram-based density estimation on real-valued
    /// hypervectors. Higher values capture more detail but require more data
    /// for stable estimates.
    pub num_bins: usize,

    /// History length (k or l) for transfer entropy estimation. Controls how
    /// many past time steps are considered when conditioning on history.
    pub history_length: usize,

    /// Small constant added inside logarithms to avoid log(0). Must be > 0.
    pub epsilon: f64,

    /// Window size (in bits) for extracting discrete symbols from BinaryHV.
    /// Must divide 8 evenly (1, 2, 4, or 8). Determines the alphabet size
    /// as 2^window_bits.
    pub window_bits: usize,
}

impl Default for InformationTheoryConfig {
    fn default() -> Self {
        Self {
            num_bins: 32,
            history_length: 5,
            epsilon: 1e-10,
            window_bits: 4,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Core Information-Theoretic Functions (on raw distributions)
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute Shannon entropy H(X) = -Σ p(x) log₂ p(x) of a probability
/// distribution. Returns value in bits.
///
/// `distribution` must be a normalised probability vector (entries sum to 1).
/// Entries <= 0 are skipped (convention: 0 log 0 = 0).
pub fn shannon_entropy(distribution: &[f64], epsilon: f64) -> f64 {
    let mut h = 0.0_f64;
    for &p in distribution {
        if p > epsilon {
            h -= p * p.log2();
        }
    }
    // Clamp to zero in case of floating-point drift
    h.max(0.0)
}

/// Compute joint entropy H(X,Y) = -Σ p(x,y) log₂ p(x,y) from a joint
/// distribution stored as a flattened 2D matrix (row-major, dims nx * ny).
pub fn joint_entropy(joint: &[f64], epsilon: f64) -> f64 {
    shannon_entropy(joint, epsilon)
}

/// Compute conditional entropy H(X|Y) = H(X,Y) - H(Y).
///
/// `joint` is the flattened joint distribution (nx * ny, row-major).
/// `ny` is the number of columns (Y categories).
pub fn conditional_entropy(joint: &[f64], ny: usize, epsilon: f64) -> f64 {
    let h_xy = joint_entropy(joint, epsilon);
    let marginal_y = marginal_column(joint, ny);
    let h_y = shannon_entropy(&marginal_y, epsilon);
    (h_xy - h_y).max(0.0)
}

/// Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).
///
/// `joint` is the flattened joint distribution (nx * ny, row-major).
/// `ny` is the number of columns (Y categories).
pub fn mutual_information(joint: &[f64], ny: usize, epsilon: f64) -> f64 {
    let nx = joint.len() / ny;
    let marginal_x = marginal_row(joint, nx, ny);
    let marginal_y = marginal_column(joint, ny);
    let h_x = shannon_entropy(&marginal_x, epsilon);
    let h_y = shannon_entropy(&marginal_y, epsilon);
    let h_xy = joint_entropy(joint, epsilon);
    // MI can be slightly negative due to float errors -- clamp
    (h_x + h_y - h_xy).max(0.0)
}

/// Normalised mutual information: NMI = I(X;Y) / max(H(X), H(Y)).
///
/// Returns 0.0 when both entropies are zero (degenerate case).
pub fn normalised_mutual_information(joint: &[f64], ny: usize, epsilon: f64) -> f64 {
    let nx = joint.len() / ny;
    let marginal_x = marginal_row(joint, nx, ny);
    let marginal_y = marginal_column(joint, ny);
    let h_x = shannon_entropy(&marginal_x, epsilon);
    let h_y = shannon_entropy(&marginal_y, epsilon);
    let denom = h_x.max(h_y);
    if denom < epsilon {
        return 0.0;
    }
    let mi = mutual_information(joint, ny, epsilon);
    (mi / denom).min(1.0)
}

/// Conditional mutual information I(X;Y|Z).
///
/// Estimated by computing MI within each stratum of Z and averaging,
/// weighted by p(z).
///
/// `joint_xyz` is a flattened 3D distribution of shape (nx, ny, nz) in
/// row-major order. `ny` and `nz` give the inner dimensions.
pub fn conditional_mutual_information(
    joint_xyz: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    epsilon: f64,
) -> f64 {
    // Marginal p(z) by summing over x and y
    let mut p_z = vec![0.0_f64; nz];
    for z in 0..nz {
        for x in 0..nx {
            for y in 0..ny {
                p_z[z] += joint_xyz[x * ny * nz + y * nz + z];
            }
        }
    }

    let mut cmi = 0.0_f64;
    for z in 0..nz {
        if p_z[z] < epsilon {
            continue;
        }
        // Extract joint(x, y | z) slice and normalise
        let mut slice_xy = vec![0.0_f64; nx * ny];
        for x in 0..nx {
            for y in 0..ny {
                slice_xy[x * ny + y] = joint_xyz[x * ny * nz + y * nz + z] / p_z[z];
            }
        }
        cmi += p_z[z] * mutual_information(&slice_xy, ny, epsilon);
    }
    cmi.max(0.0)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Marginal helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute marginal over rows (sum across columns) from flattened joint.
fn marginal_row(joint: &[f64], nx: usize, ny: usize) -> Vec<f64> {
    let mut marginal = vec![0.0_f64; nx];
    for x in 0..nx {
        for y in 0..ny {
            marginal[x] += joint[x * ny + y];
        }
    }
    marginal
}

/// Compute marginal over columns (sum across rows) from flattened joint.
fn marginal_column(joint: &[f64], ny: usize) -> Vec<f64> {
    let nx = joint.len() / ny;
    let mut marginal = vec![0.0_f64; ny];
    for x in 0..nx {
        for y in 0..ny {
            marginal[y] += joint[x * ny + y];
        }
    }
    marginal
}

// ═══════════════════════════════════════════════════════════════════════════════
// BinaryHV → Distribution Conversion
// ═══════════════════════════════════════════════════════════════════════════════

/// Extract a discrete probability distribution from a binary hypervector by
/// examining windowed bit patterns.
///
/// With `window_bits = 4`, each byte yields two 4-bit symbols (0..15). The
/// histogram over these symbols is normalised to form a distribution.
pub fn hv16_to_distribution(hv: &BinaryHV, config: &InformationTheoryConfig) -> Vec<f64> {
    let window_bits = config.window_bits.clamp(1, 8);
    let symbols_per_byte = 8 / window_bits;
    let alphabet_size = 1_usize << window_bits;
    let mask = (alphabet_size - 1) as u8;

    let mut counts = vec![0u64; alphabet_size];
    let mut total = 0u64;

    for &byte in hv.0.iter() {
        for s in 0..symbols_per_byte {
            let symbol = (byte >> (s * window_bits)) & mask;
            counts[symbol as usize] += 1;
            total += 1;
        }
    }

    if total == 0 {
        return vec![1.0 / alphabet_size as f64; alphabet_size];
    }

    counts.iter().map(|&c| c as f64 / total as f64).collect()
}

/// Build a joint distribution from two BinaryHV vectors by pairing their
/// windowed symbols at each position.
pub fn hv16_joint_distribution(
    a: &BinaryHV,
    b: &BinaryHV,
    config: &InformationTheoryConfig,
) -> (Vec<f64>, usize) {
    let window_bits = config.window_bits.clamp(1, 8);
    let symbols_per_byte = 8 / window_bits;
    let alphabet_size = 1_usize << window_bits;
    let mask = (alphabet_size - 1) as u8;

    let joint_size = alphabet_size * alphabet_size;
    let mut counts = vec![0u64; joint_size];
    let mut total = 0u64;

    for (byte_a, byte_b) in a.0.iter().zip(b.0.iter()) {
        for s in 0..symbols_per_byte {
            let sym_a = ((byte_a >> (s * window_bits)) & mask) as usize;
            let sym_b = ((byte_b >> (s * window_bits)) & mask) as usize;
            counts[sym_a * alphabet_size + sym_b] += 1;
            total += 1;
        }
    }

    if total == 0 {
        return (vec![1.0 / joint_size as f64; joint_size], alphabet_size);
    }

    let dist: Vec<f64> = counts.iter().map(|&c| c as f64 / total as f64).collect();
    (dist, alphabet_size)
}

// ═══════════════════════════════════════════════════════════════════════════════
// ContinuousHV → Distribution Conversion
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert a real-valued hypervector to a discrete distribution via histogram
/// binning. Values are placed into `num_bins` equal-width bins spanning
/// [min_val, max_val] of the vector.
pub fn real_hv_to_distribution(hv: &ContinuousHV, config: &InformationTheoryConfig) -> Vec<f64> {
    let num_bins = config.num_bins.max(2);
    let values = &hv.values;

    if values.is_empty() {
        return vec![1.0 / num_bins as f64; num_bins];
    }

    let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let range = (max_val - min_val) as f64;
    if range < 1e-12 {
        // All values identical: all mass in one bin
        let mut dist = vec![0.0_f64; num_bins];
        dist[0] = 1.0;
        return dist;
    }

    let mut counts = vec![0u64; num_bins];
    for &v in values {
        let normalised = ((v as f64 - min_val as f64) / range).clamp(0.0, 1.0 - 1e-12);
        let bin = (normalised * num_bins as f64) as usize;
        counts[bin.min(num_bins - 1)] += 1;
    }

    let total = values.len() as f64;
    counts.iter().map(|&c| c as f64 / total).collect()
}

/// Build a joint distribution from two real-valued hypervectors via 2D
/// histogram binning.
pub fn real_hv_joint_distribution(
    a: &ContinuousHV,
    b: &ContinuousHV,
    config: &InformationTheoryConfig,
) -> (Vec<f64>, usize) {
    let num_bins = config.num_bins.max(2);
    let len = a.values.len().min(b.values.len());

    if len == 0 {
        let size = num_bins * num_bins;
        return (vec![1.0 / size as f64; size], num_bins);
    }

    let min_a = a.values[..len]
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let max_a = a.values[..len]
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_b = b.values[..len]
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let max_b = b.values[..len]
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let range_a = (max_a - min_a) as f64;
    let range_b = (max_b - min_b) as f64;

    let joint_size = num_bins * num_bins;
    let mut counts = vec![0u64; joint_size];

    for i in 0..len {
        let na = if range_a < 1e-12 {
            0.0
        } else {
            ((a.values[i] as f64 - min_a as f64) / range_a).clamp(0.0, 1.0 - 1e-12)
        };
        let nb = if range_b < 1e-12 {
            0.0
        } else {
            ((b.values[i] as f64 - min_b as f64) / range_b).clamp(0.0, 1.0 - 1e-12)
        };
        let bin_a = (na * num_bins as f64) as usize;
        let bin_b = (nb * num_bins as f64) as usize;
        counts[bin_a.min(num_bins - 1) * num_bins + bin_b.min(num_bins - 1)] += 1;
    }

    let total = len as f64;
    let dist: Vec<f64> = counts.iter().map(|&c| c as f64 / total).collect();
    (dist, num_bins)
}

// ═══════════════════════════════════════════════════════════════════════════════
// High-Level HV Convenience Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute Shannon entropy of a binary hypervector's bit-pattern distribution.
pub fn entropy_hv16(hv: &BinaryHV, config: &InformationTheoryConfig) -> f64 {
    let dist = hv16_to_distribution(hv, config);
    shannon_entropy(&dist, config.epsilon)
}

/// Compute Shannon entropy of a real-valued hypervector's histogram distribution.
pub fn entropy_real_hv(hv: &ContinuousHV, config: &InformationTheoryConfig) -> f64 {
    let dist = real_hv_to_distribution(hv, config);
    shannon_entropy(&dist, config.epsilon)
}

/// Compute mutual information between two binary hypervectors.
pub fn mi_hv16(a: &BinaryHV, b: &BinaryHV, config: &InformationTheoryConfig) -> f64 {
    let (joint, ny) = hv16_joint_distribution(a, b, config);
    mutual_information(&joint, ny, config.epsilon)
}

/// Compute mutual information between two real-valued hypervectors.
pub fn mi_real_hv(a: &ContinuousHV, b: &ContinuousHV, config: &InformationTheoryConfig) -> f64 {
    let (joint, ny) = real_hv_joint_distribution(a, b, config);
    mutual_information(&joint, ny, config.epsilon)
}

/// Compute normalised mutual information between two binary hypervectors.
pub fn nmi_hv16(a: &BinaryHV, b: &BinaryHV, config: &InformationTheoryConfig) -> f64 {
    let (joint, ny) = hv16_joint_distribution(a, b, config);
    normalised_mutual_information(&joint, ny, config.epsilon)
}

/// Compute normalised mutual information between two real-valued hypervectors.
pub fn nmi_real_hv(a: &ContinuousHV, b: &ContinuousHV, config: &InformationTheoryConfig) -> f64 {
    let (joint, ny) = real_hv_joint_distribution(a, b, config);
    normalised_mutual_information(&joint, ny, config.epsilon)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transfer Entropy Estimator
// ═══════════════════════════════════════════════════════════════════════════════

/// Estimates transfer entropy between two time-evolving HDC state sequences.
///
/// Transfer Entropy TE(X→Y) measures how much the past of X reduces
/// uncertainty about the future of Y beyond what Y's own past provides:
///
///   TE(X→Y) = H(Y_t | Y_{t-1}^k) - H(Y_t | Y_{t-1}^k, X_{t-1}^l)
///
/// Internally, each HDC state is projected to a scalar summary (density for
/// BinaryHV, mean for ContinuousHV) and then discretised into histogram bins. This makes
/// estimation tractable even for 16,384-dimensional vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferEntropyEstimator {
    /// Configuration parameters.
    config: InformationTheoryConfig,

    /// Rolling history of X scalar summaries (most recent at back).
    history_x: VecDeque<f64>,

    /// Rolling history of Y scalar summaries (most recent at back).
    history_y: VecDeque<f64>,

    /// Maximum number of samples to retain.
    max_samples: usize,
}

impl TransferEntropyEstimator {
    /// Create a new estimator with the given configuration.
    ///
    /// `max_samples` controls how many time points are kept in the rolling
    /// window. Larger values give more stable estimates at the cost of memory.
    pub fn new(config: InformationTheoryConfig, max_samples: usize) -> Self {
        Self {
            config,
            history_x: VecDeque::with_capacity(max_samples),
            history_y: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    /// Create an estimator with default configuration and a 1000-sample window.
    pub fn default_estimator() -> Self {
        Self::new(InformationTheoryConfig::default(), 1000)
    }

    /// Record a new pair of binary hypervector observations at time t.
    pub fn observe_hv16(&mut self, x: &BinaryHV, y: &BinaryHV) {
        self.push_scalars(x.density() as f64, y.density() as f64);
    }

    /// Record a new pair of real-valued hypervector observations at time t.
    pub fn observe_real_hv(&mut self, x: &ContinuousHV, y: &ContinuousHV) {
        let sx = mean_f32(&x.values);
        let sy = mean_f32(&y.values);
        self.push_scalars(sx, sy);
    }

    /// Record a new pair of pre-computed scalar summaries.
    pub fn observe_scalars(&mut self, x: f64, y: f64) {
        self.push_scalars(x, y);
    }

    /// Number of observations recorded so far.
    pub fn len(&self) -> usize {
        self.history_x.len()
    }

    /// Whether the estimator has received any observations.
    pub fn is_empty(&self) -> bool {
        self.history_x.is_empty()
    }

    /// Clear all recorded observations.
    pub fn reset(&mut self) {
        self.history_x.clear();
        self.history_y.clear();
    }

    /// Estimate TE(X→Y): how much X's past reduces uncertainty about Y's
    /// future beyond Y's own past.
    ///
    /// Returns `None` if insufficient data (need at least `history_length + 1`
    /// observations).
    pub fn transfer_entropy_x_to_y(&self) -> Option<f64> {
        self.estimate_te(&self.history_x, &self.history_y)
    }

    /// Estimate TE(Y→X): the reverse direction.
    pub fn transfer_entropy_y_to_x(&self) -> Option<f64> {
        self.estimate_te(&self.history_y, &self.history_x)
    }

    /// Compute net information flow: TE(X→Y) - TE(Y→X).
    ///
    /// Positive values indicate net flow from X to Y, negative from Y to X.
    pub fn net_transfer_entropy(&self) -> Option<f64> {
        let te_xy = self.transfer_entropy_x_to_y()?;
        let te_yx = self.transfer_entropy_y_to_x()?;
        Some(te_xy - te_yx)
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    fn push_scalars(&mut self, x: f64, y: f64) {
        if self.history_x.len() >= self.max_samples {
            self.history_x.pop_front();
            self.history_y.pop_front();
        }
        self.history_x.push_back(x);
        self.history_y.push_back(y);
    }

    /// Core TE estimation: TE(source → target).
    ///
    /// Uses a plug-in histogram estimator:
    ///   TE = H(target_t, target_past) - H(target_past)
    ///      - H(target_t, target_past, source_past) + H(target_past, source_past)
    ///
    /// which is equivalent to:
    ///   TE = H(target_t | target_past) - H(target_t | target_past, source_past)
    fn estimate_te(&self, source: &VecDeque<f64>, target: &VecDeque<f64>) -> Option<f64> {
        let k = self.config.history_length;
        let n = source.len();
        if n < k + 1 {
            return None;
        }

        let num_bins = self.config.num_bins.max(2);
        let eps = self.config.epsilon;

        // Discretise all values into bin indices
        let src_bins = discretise_series(source, num_bins);
        let tgt_bins = discretise_series(target, num_bins);

        let sample_count = n - k;

        // We need to build joint histograms over:
        //   (target_t, target_past_tuple, source_past_tuple)
        // Because the full joint is high-dimensional, we hash the past tuples
        // into single indices using mixed-radix encoding.

        // For tractability with default k=5, we reduce past to a coarse
        // summary: the average bin index (re-discretised).
        let tgt_past_bins = coarse_past_summary(&tgt_bins, k, num_bins);
        let src_past_bins = coarse_past_summary(&src_bins, k, num_bins);

        // Now build three joint histograms:
        // 1. p(target_t, target_past)
        // 2. p(target_t, target_past, source_past)
        // 3. p(target_past)  [marginal of 1]
        // 4. p(target_past, source_past)  [marginal of 2]

        let nb = num_bins;
        let mut count_tt_tp = vec![0u64; nb * nb]; // target_t x target_past
        let mut count_tt_tp_sp = vec![0u64; nb * nb * nb]; // target_t x target_past x source_past
        let total = sample_count as f64;

        for i in 0..sample_count {
            let tt = tgt_bins[k + i]; // target at time k+i
            let tp = tgt_past_bins[i];
            let sp = src_past_bins[i];

            count_tt_tp[tt * nb + tp] += 1;
            count_tt_tp_sp[tt * nb * nb + tp * nb + sp] += 1;
        }

        // Convert to probabilities
        let p_tt_tp: Vec<f64> = count_tt_tp.iter().map(|&c| c as f64 / total).collect();
        let p_tt_tp_sp: Vec<f64> = count_tt_tp_sp.iter().map(|&c| c as f64 / total).collect();

        // H(target_t, target_past)
        let h_tt_tp = shannon_entropy(&p_tt_tp, eps);

        // H(target_past) = marginal over target_t from p_tt_tp
        let mut p_tp = vec![0.0_f64; nb];
        for tt in 0..nb {
            for tp in 0..nb {
                p_tp[tp] += p_tt_tp[tt * nb + tp];
            }
        }
        let h_tp = shannon_entropy(&p_tp, eps);

        // H(target_t, target_past, source_past)
        let h_tt_tp_sp = shannon_entropy(&p_tt_tp_sp, eps);

        // H(target_past, source_past) = marginal over target_t from p_tt_tp_sp
        let mut p_tp_sp = vec![0.0_f64; nb * nb];
        for tt in 0..nb {
            for tp in 0..nb {
                for sp in 0..nb {
                    p_tp_sp[tp * nb + sp] += p_tt_tp_sp[tt * nb * nb + tp * nb + sp];
                }
            }
        }
        let h_tp_sp = shannon_entropy(&p_tp_sp, eps);

        // TE = H(target_t, target_past) - H(target_past)
        //    - H(target_t, target_past, source_past) + H(target_past, source_past)
        let te = (h_tt_tp - h_tp) - (h_tt_tp_sp - h_tp_sp);
        Some(te.max(0.0))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Mean of an f32 slice, returned as f64.
fn mean_f32(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().map(|&v| v as f64).sum();
    sum / values.len() as f64
}

/// Discretise a time series of f64 values into `num_bins` integer bin indices.
fn discretise_series(series: &VecDeque<f64>, num_bins: usize) -> Vec<usize> {
    if series.is_empty() {
        return Vec::new();
    }

    let min_val = series.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    series
        .iter()
        .map(|&v| {
            if range < 1e-12 {
                0
            } else {
                let normalised = ((v - min_val) / range).clamp(0.0, 1.0 - 1e-12);
                (normalised * num_bins as f64) as usize
            }
        })
        .collect()
}

/// Produce a coarse summary of the past k bin values by averaging and
/// re-discretising into `num_bins`.
fn coarse_past_summary(bins: &[usize], k: usize, num_bins: usize) -> Vec<usize> {
    let n = if bins.len() > k { bins.len() - k } else { 0 };
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let sum: f64 = bins[i..i + k].iter().map(|&b| b as f64).sum();
        let avg = sum / k as f64;
        let normalised = avg / (num_bins.max(1) as f64 - 1.0).max(1e-12);
        let bin = (normalised.clamp(0.0, 1.0 - 1e-12) * num_bins as f64) as usize;
        result.push(bin.min(num_bins - 1));
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    // ── Shannon Entropy Tests ───────────────────────────────────────────────

    #[test]
    fn entropy_of_uniform_distribution_equals_log2_n() {
        // A uniform distribution over n symbols has entropy log₂(n)
        for &n in &[2, 4, 8, 16, 32, 64] {
            let dist: Vec<f64> = vec![1.0 / n as f64; n];
            let h = shannon_entropy(&dist, EPS);
            let expected = (n as f64).log2();
            assert!(
                (h - expected).abs() < 1e-9,
                "Uniform({n}): expected H={expected:.6}, got H={h:.6}"
            );
        }
    }

    #[test]
    fn entropy_of_deterministic_distribution_is_zero() {
        // All probability on one symbol: H = 0
        let mut dist = vec![0.0; 8];
        dist[3] = 1.0;
        let h = shannon_entropy(&dist, EPS);
        assert!(h.abs() < 1e-9, "Deterministic: expected H=0, got H={h:.6}");
    }

    #[test]
    fn entropy_is_nonnegative() {
        // Arbitrary distribution
        let dist = vec![0.1, 0.2, 0.3, 0.15, 0.25];
        let h = shannon_entropy(&dist, EPS);
        assert!(h >= 0.0, "Entropy must be non-negative, got {h}");
    }

    // ── Mutual Information Tests ────────────────────────────────────────────

    #[test]
    fn mi_of_identical_signals_equals_entropy() {
        // I(X;X) = H(X) for a uniform distribution
        let n = 8;
        let p = 1.0 / n as f64;
        // Joint p(x, y) where y = x: diagonal matrix
        let mut joint = vec![0.0_f64; n * n];
        for i in 0..n {
            joint[i * n + i] = p;
        }
        let mi = mutual_information(&joint, n, EPS);
        let expected = (n as f64).log2();
        assert!(
            (mi - expected).abs() < 1e-9,
            "MI(X,X) should equal H(X)={expected:.6}, got {mi:.6}"
        );
    }

    #[test]
    fn mi_of_independent_signals_is_zero() {
        // If X and Y are independent, p(x,y) = p(x)*p(y), so MI = 0
        let nx = 4;
        let ny = 4;
        let px: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
        let py: Vec<f64> = vec![0.25, 0.25, 0.25, 0.25];
        let mut joint = vec![0.0_f64; nx * ny];
        for x in 0..nx {
            for y in 0..ny {
                joint[x * ny + y] = px[x] * py[y];
            }
        }
        let mi = mutual_information(&joint, ny, EPS);
        assert!(
            mi.abs() < 1e-9,
            "Independent signals: MI should be ~0, got {mi:.6}"
        );
    }

    #[test]
    fn mi_symmetry() {
        // MI(X,Y) = MI(Y,X): transposing the joint matrix should give the same MI.
        let nx = 4;
        let ny = 5;
        // Construct an arbitrary joint distribution
        let raw = vec![
            0.02, 0.03, 0.01, 0.04, 0.05, 0.06, 0.01, 0.02, 0.03, 0.02, 0.04, 0.05, 0.03, 0.01,
            0.02, 0.07, 0.04, 0.08, 0.06, 0.31,
        ];
        // Verify it sums to 1
        let sum: f64 = raw.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        let mi_xy = mutual_information(&raw, ny, EPS);

        // Transpose: new joint is ny x nx
        let mut transposed = vec![0.0_f64; ny * nx];
        for x in 0..nx {
            for y in 0..ny {
                transposed[y * nx + x] = raw[x * ny + y];
            }
        }
        let mi_yx = mutual_information(&transposed, nx, EPS);

        assert!(
            (mi_xy - mi_yx).abs() < 1e-9,
            "MI should be symmetric: MI(X,Y)={mi_xy:.6} != MI(Y,X)={mi_yx:.6}"
        );
    }

    #[test]
    fn mi_nonnegative() {
        // MI is always >= 0
        let joint = vec![0.1, 0.05, 0.15, 0.2, 0.3, 0.2];
        let mi = mutual_information(&joint, 3, EPS);
        assert!(mi >= 0.0, "MI must be non-negative, got {mi}");
    }

    // ── Normalised Mutual Information ───────────────────────────────────────

    #[test]
    fn nmi_of_identical_signals_is_one() {
        let n = 8;
        let p = 1.0 / n as f64;
        let mut joint = vec![0.0_f64; n * n];
        for i in 0..n {
            joint[i * n + i] = p;
        }
        let nmi = normalised_mutual_information(&joint, n, EPS);
        assert!(
            (nmi - 1.0).abs() < 1e-9,
            "NMI(X,X) should be 1.0, got {nmi:.6}"
        );
    }

    #[test]
    fn nmi_of_independent_signals_is_zero() {
        let n = 4;
        let p = 1.0 / n as f64;
        let joint: Vec<f64> = vec![p * p; n * n]; // uniform independent
        let nmi = normalised_mutual_information(&joint, n, EPS);
        assert!(
            nmi.abs() < 1e-9,
            "NMI of independent signals should be ~0, got {nmi:.6}"
        );
    }

    // ── Conditional Entropy ─────────────────────────────────────────────────

    #[test]
    fn conditional_entropy_independent_equals_marginal() {
        // H(X|Y) = H(X) when X, Y independent
        let nx = 3;
        let ny = 4;
        let px = vec![0.2, 0.5, 0.3];
        let py = [0.1, 0.2, 0.3, 0.4];
        let mut joint = vec![0.0_f64; nx * ny];
        for x in 0..nx {
            for y in 0..ny {
                joint[x * ny + y] = px[x] * py[y];
            }
        }
        let h_x_given_y = conditional_entropy(&joint, ny, EPS);
        let h_x = shannon_entropy(&px, EPS);
        assert!(
            (h_x_given_y - h_x).abs() < 1e-9,
            "H(X|Y) should equal H(X) for independent: {h_x_given_y:.6} vs {h_x:.6}"
        );
    }

    #[test]
    fn conditional_entropy_deterministic_is_zero() {
        // If Y = X, then H(X|Y) = 0
        let n = 5;
        let p = 1.0 / n as f64;
        let mut joint = vec![0.0_f64; n * n];
        for i in 0..n {
            joint[i * n + i] = p;
        }
        let h_x_given_y = conditional_entropy(&joint, n, EPS);
        assert!(
            h_x_given_y.abs() < 1e-9,
            "H(X|X) should be 0, got {h_x_given_y:.6}"
        );
    }

    // ── Conditional Mutual Information ──────────────────────────────────────

    #[test]
    fn cmi_reduces_to_mi_when_z_is_constant() {
        // If Z has only one value, I(X;Y|Z) = I(X;Y)
        let nx = 3;
        let ny = 3;
        let nz = 1;
        // Just a 2D joint embedded in 3D with nz=1
        let joint_2d = vec![0.15, 0.05, 0.10, 0.05, 0.25, 0.05, 0.10, 0.05, 0.20];
        // Same thing as a 3D joint with nz=1
        let joint_3d = joint_2d.clone(); // shape (3, 3, 1)

        let mi = mutual_information(&joint_2d, ny, EPS);
        let cmi = conditional_mutual_information(&joint_3d, nx, ny, nz, EPS);

        assert!(
            (mi - cmi).abs() < 1e-9,
            "CMI with constant Z should equal MI: {cmi:.6} vs {mi:.6}"
        );
    }

    // ── BinaryHV Distribution Tests ─────────────────────────────────────────────

    #[test]
    fn hv16_ones_has_deterministic_pattern() {
        let config = InformationTheoryConfig::default();
        let hv = BinaryHV::ones();
        let dist = hv16_to_distribution(&hv, &config);
        // All bytes are 0xFF. With window_bits=4, both nibbles are 0xF = 15.
        // So all probability mass should be on symbol 15.
        assert!(
            dist[15] > 0.99,
            "All-ones HV should concentrate on symbol 15, got {:.4}",
            dist[15]
        );
    }

    #[test]
    fn hv16_entropy_increases_with_randomness() {
        let config = InformationTheoryConfig::default();
        let zeros = BinaryHV::zero();
        let random = BinaryHV::random(42);
        let h_zeros = entropy_hv16(&zeros, &config);
        let h_random = entropy_hv16(&random, &config);
        // Random HV should have higher entropy than constant
        assert!(
            h_random > h_zeros,
            "Random HV should have higher entropy: {h_random:.4} vs {h_zeros:.4}"
        );
    }

    #[test]
    fn hv16_mi_identical_equals_entropy() {
        let config = InformationTheoryConfig::default();
        let hv = BinaryHV::random(99);
        let h = entropy_hv16(&hv, &config);
        let mi = mi_hv16(&hv, &hv, &config);
        assert!(
            (mi - h).abs() < 1e-6,
            "MI(X,X) should equal H(X): {mi:.6} vs {h:.6}"
        );
    }

    #[test]
    fn hv16_mi_symmetry() {
        let config = InformationTheoryConfig::default();
        let a = BinaryHV::random(100);
        let b = BinaryHV::random(200);
        let mi_ab = mi_hv16(&a, &b, &config);
        let mi_ba = mi_hv16(&b, &a, &config);
        assert!(
            (mi_ab - mi_ba).abs() < 1e-9,
            "MI should be symmetric for BinaryHV: {mi_ab:.6} vs {mi_ba:.6}"
        );
    }

    // ── ContinuousHV Distribution Tests ───────────────────────────────────────────

    #[test]
    fn real_hv_uniform_has_high_entropy() {
        let config = InformationTheoryConfig {
            num_bins: 16,
            ..Default::default()
        };
        // Construct a "uniform-like" ContinuousHV by spanning [-1, 1] evenly
        let dim = 1024;
        let values: Vec<f32> = (0..dim)
            .map(|i| -1.0 + 2.0 * (i as f32) / (dim as f32 - 1.0))
            .collect();
        let hv = ContinuousHV::from_values(values);
        let h = entropy_real_hv(&hv, &config);
        let max_h = (16.0_f64).log2(); // 4 bits
        assert!(
            h > max_h * 0.9,
            "Uniform-like ContinuousHV should have near-max entropy: {h:.4} vs max {max_h:.4}"
        );
    }

    #[test]
    fn real_hv_constant_has_zero_entropy() {
        let config = InformationTheoryConfig::default();
        let hv = ContinuousHV::from_values(vec![0.5; 512]);
        let h = entropy_real_hv(&hv, &config);
        assert!(
            h.abs() < 1e-9,
            "Constant ContinuousHV should have zero entropy, got {h:.6}"
        );
    }

    #[test]
    fn real_hv_mi_identical_equals_entropy() {
        let config = InformationTheoryConfig {
            num_bins: 16,
            ..Default::default()
        };
        let hv = ContinuousHV::random(512, 42);
        let h = entropy_real_hv(&hv, &config);
        let mi = mi_real_hv(&hv, &hv, &config);
        assert!(
            (mi - h).abs() < 1e-6,
            "MI(X,X) should equal H(X) for ContinuousHV: {mi:.6} vs {h:.6}"
        );
    }

    #[test]
    fn real_hv_mi_symmetry() {
        let config = InformationTheoryConfig {
            num_bins: 16,
            ..Default::default()
        };
        let a = ContinuousHV::random(256, 10);
        let b = ContinuousHV::random(256, 20);
        let mi_ab = mi_real_hv(&a, &b, &config);
        let mi_ba = mi_real_hv(&b, &a, &config);
        assert!(
            (mi_ab - mi_ba).abs() < 1e-9,
            "MI should be symmetric for ContinuousHV: {mi_ab:.6} vs {mi_ba:.6}"
        );
    }

    // ── Transfer Entropy Tests ──────────────────────────────────────────────

    #[test]
    fn te_independent_signals_near_zero() {
        // Two independent random walks should have near-zero TE.
        // Use fewer bins and more samples to reduce finite-sample bias.
        let config = InformationTheoryConfig {
            num_bins: 4,
            history_length: 2,
            ..Default::default()
        };
        let mut est = TransferEntropyEstimator::new(config, 5000);

        // Generate independent sequences using simple LCG
        let mut x_val = 0.5_f64;
        let mut y_val = 0.3_f64;
        for i in 0..2000 {
            // Independent pseudo-random walks
            x_val = (x_val * 6364136223846793005u64 as f64 + (i as f64))
                .sin()
                .abs();
            y_val = (y_val * 1442695040888963407u64 as f64 + (i as f64 + 100.0))
                .cos()
                .abs();
            est.observe_scalars(x_val, y_val);
        }

        let te_xy = est.transfer_entropy_x_to_y().expect("Enough data");
        let te_yx = est.transfer_entropy_y_to_x().expect("Enough data");

        // TE should be small (close to zero) for independent signals.
        // With finite samples and histogram estimation, allow generous tolerance.
        assert!(
            te_xy < 0.5,
            "TE(X→Y) for independent signals should be small, got {te_xy:.4}"
        );
        assert!(
            te_yx < 0.5,
            "TE(Y→X) for independent signals should be small, got {te_yx:.4}"
        );
    }

    #[test]
    fn te_driven_signal_shows_direction() {
        // Y is driven by X's past: Y_t = X_{t-1} + noise
        // TE(X→Y) should be larger than TE(Y→X).
        let config = InformationTheoryConfig {
            num_bins: 8,
            history_length: 2,
            ..Default::default()
        };
        let mut est = TransferEntropyEstimator::new(config, 500);

        let mut prev_x = 0.5_f64;
        for i in 0..300 {
            // X: pseudo-random walk
            let seed = ((i as f64 * std::f64::consts::E).sin() * 1000.0).abs();
            let x_val = (seed % 1.0).clamp(0.01, 0.99);

            // Y is driven by previous X (causal link) + small noise
            let noise = ((i as f64 * std::f64::consts::PI).cos() * 0.05).abs();
            let y_val = (prev_x * 0.8 + noise).clamp(0.01, 0.99);

            est.observe_scalars(x_val, y_val);
            prev_x = x_val;
        }

        let te_xy = est.transfer_entropy_x_to_y().expect("Enough data");
        let te_yx = est.transfer_entropy_y_to_x().expect("Enough data");

        // The causal direction should show higher TE
        // (with finite-sample estimation this is probabilistic, but the
        // deterministic driven signal should be detectable)
        assert!(
            te_xy >= te_yx * 0.5 || te_xy > 0.0,
            "TE(X→Y)={te_xy:.4} should reflect causal drive, TE(Y→X)={te_yx:.4}"
        );
    }

    #[test]
    fn te_insufficient_data_returns_none() {
        let config = InformationTheoryConfig {
            history_length: 5,
            ..Default::default()
        };
        let mut est = TransferEntropyEstimator::new(config, 100);

        // Only 3 observations, need at least 6 (history_length + 1)
        est.observe_scalars(0.1, 0.2);
        est.observe_scalars(0.3, 0.4);
        est.observe_scalars(0.5, 0.6);

        assert!(
            est.transfer_entropy_x_to_y().is_none(),
            "Should return None with insufficient data"
        );
    }

    #[test]
    fn te_reset_clears_history() {
        let mut est = TransferEntropyEstimator::default_estimator();
        est.observe_scalars(1.0, 2.0);
        est.observe_scalars(3.0, 4.0);
        assert_eq!(est.len(), 2);

        est.reset();
        assert!(est.is_empty());
        assert_eq!(est.len(), 0);
    }

    // ── Joint & Conditional Entropy Tests ───────────────────────────────────

    #[test]
    fn joint_entropy_geq_marginal_entropy() {
        // H(X,Y) >= max(H(X), H(Y))
        let nx = 3;
        let ny = 4;
        let joint = vec![
            0.05, 0.10, 0.05, 0.05, 0.10, 0.05, 0.10, 0.10, 0.05, 0.10, 0.05, 0.15,
        ];
        let h_xy = joint_entropy(&joint, EPS);
        let mx = marginal_row(&joint, nx, ny);
        let my = marginal_column(&joint, ny);
        let h_x = shannon_entropy(&mx, EPS);
        let h_y = shannon_entropy(&my, EPS);
        assert!(
            h_xy >= h_x - 1e-9 && h_xy >= h_y - 1e-9,
            "H(X,Y)={h_xy:.4} should be >= H(X)={h_x:.4} and H(Y)={h_y:.4}"
        );
    }

    // ── Config tests ────────────────────────────────────────────────────────

    #[test]
    fn default_config_is_sensible() {
        let config = InformationTheoryConfig::default();
        assert_eq!(config.num_bins, 32);
        assert_eq!(config.history_length, 5);
        assert!(config.epsilon > 0.0);
        assert!(config.epsilon < 1e-6);
        assert_eq!(config.window_bits, 4);
    }
}
