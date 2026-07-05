// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Spectral MIP Finder — O(n³) via Fiedler Ordering + Bordered Cholesky
//!
//! Scales MIP search from O(2^n) exhaustive to **O(n³)** by exploiting the spectral
//! structure of the mutual information graph.
//!
//! ## Algorithm (5 steps)
//!
//! 1. **Covariance estimation**: Sliding window of T HDC snapshots → n×n Σ̂ + εI
//! 2. **MI Laplacian**: Pairwise Gaussian MI → graph Laplacian L = D - W
//! 3. **Fiedler ordering**: 2nd eigenvector of L via shifted inverse iteration with
//!    deflation (O(n³/6) Cholesky + 30 × O(n²) solves)
//! 4. **Bordered Cholesky sweep**: n-1 contiguous cuts with O(k²) Cholesky updates,
//!    left/right sweeps in parallel via `rayon::join` → O(n³/6) wall time
//! 5. **MIP selection**: `Φ = MI_total - min_k MI_cut(k)`
//!
//! ## Key Insight
//!
//! The Fiedler vector of the MI Laplacian transforms MIP search from NP-hard
//! graph bisection into finding the weakest link in a 1D chain. Contiguous cuts
//! are nested (each adds one row/col), so bordered Cholesky update is O(k²) per
//! step → O(n³) total.
//!
//! ## Performance
//!
//! | Operation | n=128 | n=256 |
//! |-----------|-------|-------|
//! | push      | 442ns | 772ns |
//! | pipeline  | 5.5ms | —     |
//!
//! Well within the 20ms (50Hz) cognitive loop budget.
//!
//! ## References
//!
//! - Fiedler (1973), "Algebraic connectivity of graphs"
//! - Kitazono et al. (2018), "Efficient Algorithms for Searching the MIP in IIT"
//! - Oizumi et al. (2016), "Measuring Integrated Information from the Decoding Perspective"
//!
//! Full documentation: `docs/architecture/SPECTRAL_MIP_ALGORITHM.md`

use std::collections::VecDeque;

use crate::hdc::unified_hv::ContinuousHV;

use super::TruePartition;

/// Configuration for spectral MIP computation.
#[derive(Debug, Clone)]
pub struct SpectralMIPConfig {
    /// Number of HDC components to track (subsample from full dimension).
    /// Default: 128
    pub num_components: usize,
    /// Sliding window size (number of state snapshots). Default: 50
    pub window_size: usize,
    /// Minimum window fill before computing. Default: 10
    pub min_samples: usize,
    /// Regularization added to covariance diagonal. Default: 1e-6
    pub regularization: f64,
    /// Normalize covariance to correlation before computing Phi.
    /// When true, each dimension is z-scored (variance → 1.0), so Phi
    /// measures pattern integration rather than signal amplitude.
    /// Removes confound where stable (low-variance) signals produce inflated Phi.
    /// Science: Zalesky et al. (2012) — standard practice in fMRI connectivity.
    /// Default: true
    pub normalize_variance: bool,
    /// Apply first-difference filter before covariance estimation.
    /// When true, covariance is built from Δx[t] = x[t] - x[t-1] instead
    /// of raw snapshots, removing temporal autocorrelation. This measures
    /// *novelty of integration patterns* — how dimensions co-vary in their
    /// *changes* — rather than persistence of a static correlation structure.
    /// Removes confound where frozen dynamics (no learning, no embodiment)
    /// produce inflated Phi via high temporal autocorrelation.
    /// Science: Hamilton (1994) — first-differencing for stationarity;
    /// Friston et al. (2003) — dynamic causal modeling uses rate-of-change.
    /// Default: true
    pub temporal_decorrelation: bool,
}

impl Default for SpectralMIPConfig {
    fn default() -> Self {
        Self {
            num_components: 128,
            window_size: 50,
            min_samples: 10,
            regularization: 1e-6,
            normalize_variance: true,
            temporal_decorrelation: true,
        }
    }
}

impl SpectralMIPConfig {
    /// Validate spectral MIP configuration parameters.
    ///
    /// Checks that:
    /// - `num_components` >= 2 (minimum for a bipartition; >= 4 recommended for meaningful MIP)
    /// - `window_size` >= `min_samples` (window must fit minimum required samples)
    /// - `min_samples` >= 2 (need at least 2 samples for covariance)
    /// - `regularization` is positive and finite
    pub fn validate(&self) -> Result<(), String> {
        if self.num_components < 2 {
            return Err(format!(
                "SpectralMIPConfig: num_components must be >= 2, got {}",
                self.num_components
            ));
        }
        if self.min_samples < 2 {
            return Err(format!(
                "SpectralMIPConfig: min_samples must be >= 2, got {}",
                self.min_samples
            ));
        }
        if self.window_size < self.min_samples {
            return Err(format!(
                "SpectralMIPConfig: window_size ({}) must be >= min_samples ({})",
                self.window_size, self.min_samples
            ));
        }
        if self.regularization <= 0.0 || !self.regularization.is_finite() {
            return Err(format!(
                "SpectralMIPConfig: regularization must be positive and finite, got {}",
                self.regularization
            ));
        }
        Ok(())
    }
}

/// Result of a spectral MIP computation.
#[derive(Debug, Clone)]
pub struct SpectralMIPResult {
    /// Phi via spectral MIP (non-negative)
    pub phi: f64,
    /// Total Gaussian MI of the system
    pub total_mi: f64,
    /// MI at the minimum information partition cut
    pub mip_mi: f64,
    /// The partition found (original indices)
    pub mip: TruePartition,
    /// Fiedler-sorted dimension indices
    pub spectral_order: Vec<usize>,
    /// Cut position in spectral order (left block = 0..=mip_bond)
    pub mip_bond: usize,
    /// Where the Fiedler vector changes sign
    pub fiedler_zero_crossing: usize,
    /// MI at each of n-1 contiguous cuts (diagnostic)
    pub cut_mis: Vec<f64>,
    /// Number of snapshots used
    pub window_used: usize,
    /// Number of dimensions tracked
    pub num_components: usize,
}

/// Result of hierarchical (multi-scale) spectral MIP computation.
///
/// Contains results at each scale level, from coarsest to finest.
/// The final level's phi is the most precise estimate.
#[derive(Debug, Clone)]
pub struct HierarchicalMIPResult {
    /// Final (finest-scale) phi
    pub phi: f64,
    /// Scales used (number of components at each level)
    pub scales: Vec<usize>,
    /// Results at each scale, from coarsest to finest (indices mapped to full-n space)
    pub levels: Vec<SpectralMIPResult>,
}

/// Spectral MIP Finder — O(n³) MIP search via Fiedler ordering + bordered Cholesky.
///
/// Supports **adaptive dimension selection**: after each compute, `adapt()` uses the
/// Fiedler ordering to concentrate tracked dimensions around the MIP boundary where
/// they are most informative, while maintaining exploration via evenly-spaced and
/// random dimensions. Adaptation flushes the window (new dims = new column semantics).
#[derive(Debug, Clone)]
pub struct SpectralMIPFinder {
    window: VecDeque<Vec<f64>>,
    config: SpectralMIPConfig,
    /// Indices into the full HDC dimension space that we're tracking.
    /// `None` = evenly-spaced (default behavior).
    active_dims: Option<Vec<usize>>,
    /// Full dimension of the HDC state (set on first push).
    full_dim: usize,
    /// Running sum of component values (for online mean computation).
    sum: Vec<f64>,
    /// Running sum of outer products, upper triangular packed as n×n flat array.
    /// `cross_sum[i*n+j]` = Σ_t x_t[i]*x_t[j] for j >= i.
    cross_sum: Vec<f64>,
}

impl SpectralMIPFinder {
    /// Create a new spectral MIP finder with the given configuration.
    ///
    /// # Panics
    ///
    /// Panics if the configuration is invalid (see [`SpectralMIPConfig::validate`]).
    /// Use `SpectralMIPConfig::validate()` beforehand for graceful error handling.
    pub fn new(config: SpectralMIPConfig) -> Self {
        if let Err(e) = config.validate() {
            panic!("Invalid SpectralMIPConfig: {e}");
        }
        let n = config.num_components;
        Self {
            window: VecDeque::with_capacity(config.window_size),
            sum: vec![0.0; n],
            cross_sum: vec![0.0; n * n],
            config,
            active_dims: None,
            full_dim: 0,
        }
    }

    /// Create with default configuration (128 components, 50-sample window).
    pub fn with_defaults() -> Self {
        Self::new(SpectralMIPConfig::default())
    }

    /// Feed a new HDC state snapshot into the sliding window.
    ///
    /// Maintains online running sums for O(1) covariance matrix construction.
    /// If `active_dims` is set (after adaptation), reads those specific dimensions.
    /// Otherwise uses evenly-spaced subsampling.
    pub fn push(&mut self, state: &ContinuousHV) {
        let dim = state.dim();
        self.full_dim = dim;
        let n = self.config.num_components.min(dim);

        // If window full, subtract outgoing snapshot from running sums.
        // Reuse its allocation for the incoming snapshot to avoid a heap alloc.
        let mut snapshot: Vec<f64> = if self.window.len() >= self.config.window_size {
            if let Some(old) = self.window.pop_front() {
                for i in 0..n.min(old.len()) {
                    self.sum[i] -= old[i];
                    for j in i..n.min(old.len()) {
                        self.cross_sum[i * n + j] -= old[i] * old[j];
                    }
                }
                old // reuse allocation
            } else {
                Vec::with_capacity(n)
            }
        } else {
            Vec::with_capacity(n)
        };

        // Fill snapshot, reusing the buffer
        snapshot.clear();
        if let Some(ref indices) = self.active_dims {
            snapshot.extend(
                indices
                    .iter()
                    .take(n)
                    .map(|&i| state.values[i.min(dim - 1)] as f64),
            );
        } else {
            let stride = if n > 1 { dim / n } else { 1 };
            snapshot.extend((0..n).map(|i| state.values[i * stride] as f64));
        }

        // Add incoming snapshot to running sums
        for i in 0..n.min(snapshot.len()) {
            self.sum[i] += snapshot[i];
            for j in i..n.min(snapshot.len()) {
                self.cross_sum[i * n + j] += snapshot[i] * snapshot[j];
            }
        }

        self.window.push_back(snapshot);
    }

    /// Whether we have enough samples to compute.
    pub fn ready(&self) -> bool {
        self.window.len() >= self.config.min_samples
    }

    /// Compute spectral MIP from the current window.
    pub fn compute(&self) -> Option<SpectralMIPResult> {
        if !self.ready() {
            return None;
        }

        let n = self.config.num_components;
        let t = self.window.len();
        let cov = self.build_effective_covariance(n, t);
        self.compute_from_covariance(&cov, n, t)
    }

    /// Build covariance matrix with all configured preprocessing applied.
    /// Centralizes temporal decorrelation + variance normalization so that
    /// `compute()`, `compute_hierarchical()`, and `compute_structural_hierarchy()`
    /// all operate on the same preprocessed covariance.
    fn build_effective_covariance(&self, n: usize, t: usize) -> Vec<f64> {
        let mut cov = if self.config.temporal_decorrelation && t >= 3 {
            self.build_differenced_covariance(n)
        } else {
            self.build_covariance_matrix(n, t)
        };
        if self.config.normalize_variance {
            covariance_to_correlation(&mut cov, n);
        }
        cov
    }

    /// Compute spectral MIP from a pre-built covariance matrix (flat row-major, n×n).
    pub fn compute_from_covariance(
        &self,
        cov: &[f64],
        n: usize,
        window_used: usize,
    ) -> Option<SpectralMIPResult> {
        if n < 2 || cov.len() != n * n {
            return None;
        }

        // Step 1: Build MI Laplacian
        let (mi_weights, laplacian) = build_mi_laplacian(cov, n);

        // Step 2: Fiedler ordering
        let (spectral_order, fiedler_zero_crossing) =
            compute_fiedler_order(&laplacian, &mi_weights, n);

        // Step 3: Reorder covariance by spectral order
        let reordered = reorder_matrix(cov, n, &spectral_order);

        // Step 4: Sweep contiguous cuts with bordered Cholesky
        let cut_mis = sweep_contiguous_cuts(&reordered, n);

        // Step 5: Find MIP (minimum cut MI)
        let (mip_bond, mip_mi) = cut_mis
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, &v)| (i, v))
            .unwrap_or((0, 0.0));

        // Step 6: Total MI
        let total_mi = gaussian_mi_total(cov, n);

        // Step 7: Build partition in original indices
        let part_a: Vec<usize> = spectral_order[..=mip_bond].to_vec();
        let part_b: Vec<usize> = spectral_order[mip_bond + 1..].to_vec();

        let phi = (total_mi - mip_mi).max(0.0);

        Some(SpectralMIPResult {
            phi,
            total_mi,
            mip_mi,
            mip: TruePartition { part_a, part_b },
            spectral_order,
            mip_bond,
            fiedler_zero_crossing,
            cut_mis,
            window_used,
            num_components: n,
        })
    }

    /// Reset the sliding window and running statistics.
    pub fn reset(&mut self) {
        self.window.clear();
        self.sum.fill(0.0);
        self.cross_sum.fill(0.0);
    }

    /// Number of snapshots currently in the window.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Whether adaptive dimension selection is active.
    pub fn is_adapted(&self) -> bool {
        self.active_dims.is_some()
    }

    /// Current active dimension indices (into the full HDC space), if adapted.
    pub fn active_dim_indices(&self) -> Option<&[usize]> {
        self.active_dims.as_deref()
    }

    /// Adapt dimension selection based on a previous MIP result.
    ///
    /// Concentrates 60% of tracked dimensions near the MIP boundary (where
    /// partition decisions are most sensitive), 20% evenly spaced (coverage),
    /// and 20% random (exploration). Flushes the window since column semantics
    /// change.
    ///
    /// Call this every 2nd compute (e.g. every 100 cycles) to balance
    /// adaptation frequency with window stability.
    pub fn adapt(&mut self, result: &SpectralMIPResult) {
        let n = self.config.num_components;
        let dim = self.full_dim;
        if dim == 0 || n < 4 || result.spectral_order.len() != n {
            return;
        }

        // Map current spectral order back to full-space dimension indices
        let current_dims: Vec<usize> = self.active_dims.clone().unwrap_or_else(|| {
            let stride = if n > 1 { dim / n } else { 1 };
            (0..n).map(|i| (i * stride).min(dim - 1)).collect()
        });

        // Fiedler-ordered full-space indices
        let spectral_full: Vec<usize> = result
            .spectral_order
            .iter()
            .map(|&i| current_dims[i.min(current_dims.len() - 1)])
            .collect();

        // Budget: 60% boundary, 20% coverage, 20% exploration
        let n_boundary = (n * 6 / 10).max(4);
        let n_coverage = (n * 2 / 10).max(2);

        let mut new_dims: Vec<usize> = Vec::with_capacity(n);
        let mut used = vec![false; dim]; // track used full-space indices

        // 1. Boundary: dims centered on mip_bond in Fiedler order
        let mip_bond = result.mip_bond.min(spectral_full.len() - 1);
        let half = n_boundary / 2;
        let start = mip_bond.saturating_sub(half);
        let end = (start + n_boundary).min(spectral_full.len());
        let start = end.saturating_sub(n_boundary);
        for &idx in &spectral_full[start..end] {
            if idx < dim && !used[idx] {
                new_dims.push(idx);
                used[idx] = true;
            }
        }

        // 2. Coverage: evenly spaced across full dimension space
        let cov_stride = dim / n_coverage.max(1);
        for i in 0..n_coverage {
            let idx = (i * cov_stride).min(dim - 1);
            if !used[idx] {
                new_dims.push(idx);
                used[idx] = true;
            }
        }

        // 3. Exploration: deterministic pseudo-random dims not already tracked
        let mut rng_state =
            result.phi.to_bits() ^ (result.mip_bond as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let mut attempts = 0u32;
        while new_dims.len() < n && attempts < n as u32 * 10 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let idx = (rng_state as usize) % dim;
            if !used[idx] {
                new_dims.push(idx);
                used[idx] = true;
            }
            attempts += 1;
        }

        // Sort for cache-friendly access during push()
        new_dims.sort_unstable();
        new_dims.truncate(n);

        self.active_dims = Some(new_dims);
        self.window.clear(); // column semantics changed — invalidate window
        self.sum.fill(0.0);
        self.cross_sum.fill(0.0);
    }

    /// Compute hierarchical (multi-scale) spectral MIP.
    ///
    /// Runs spectral MIP at increasing resolutions (e.g., 32 → 64 → n),
    /// using each level's Fiedler ordering to focus the next level's component
    /// selection on the MIP boundary region. Returns results at every scale.
    ///
    /// The covariance is built once from the window; each level extracts a
    /// submatrix at different granularity. Final-level phi is the most precise.
    pub fn compute_hierarchical(&self) -> Option<HierarchicalMIPResult> {
        if !self.ready() {
            return None;
        }

        let n = self.config.num_components;
        let t = self.window.len();
        let cov = self.build_effective_covariance(n, t);

        // Build scale ladder: 32, 64, 128, ... up to n (deduplicated)
        let mut scales: Vec<usize> = Vec::new();
        let mut s = 32_usize.min(n);
        while s < n {
            if s >= 4 {
                scales.push(s);
            }
            s *= 2;
        }
        scales.push(n);
        scales.dedup();

        // If n is small enough for a single pass, just return that
        if scales.len() <= 1 {
            let result = self.compute_from_covariance(&cov, n, t)?;
            return Some(HierarchicalMIPResult {
                phi: result.phi,
                scales: vec![n],
                levels: vec![result],
            });
        }

        let mut levels = Vec::with_capacity(scales.len());
        let mut focused_indices: Option<Vec<usize>> = None;

        for &scale in &scales {
            // Select which of the n components to use at this scale
            let indices: Vec<usize> = if scale == n {
                (0..n).collect()
            } else if let Some(ref prev) = focused_indices {
                // Focus around previous MIP boundary + coverage + exploration
                prev.iter().copied().take(scale).collect()
            } else {
                // Evenly spaced for first (coarsest) level
                let stride = (n / scale).max(1);
                (0..scale).map(|i| (i * stride).min(n - 1)).collect()
            };

            let actual_scale = indices.len();
            if actual_scale < 4 {
                continue;
            }

            // Extract sub-covariance matrix
            let sub_cov = extract_submatrix(&cov, n, &indices);

            // Run spectral MIP at this scale
            if let Some(mut result) = self.compute_from_covariance(&sub_cov, actual_scale, t) {
                // Map spectral_order and partition back to full-n index space
                result.spectral_order = result
                    .spectral_order
                    .iter()
                    .map(|&i| indices[i.min(indices.len() - 1)])
                    .collect();
                result.mip.part_a = result
                    .mip
                    .part_a
                    .iter()
                    .map(|&i| indices[i.min(indices.len() - 1)])
                    .collect();
                result.mip.part_b = result
                    .mip
                    .part_b
                    .iter()
                    .map(|&i| indices[i.min(indices.len() - 1)])
                    .collect();

                // Build focused indices for next scale: 60% boundary, 40% coverage
                if scale < n {
                    let next_scale = scales.iter().find(|&&s| s > scale).copied().unwrap_or(n);
                    focused_indices = Some(focus_dims_for_next_level(n, next_scale, &result));
                }

                levels.push(result);
            }
        }

        let phi = levels.last().map(|l| l.phi).unwrap_or(0.0);

        Some(HierarchicalMIPResult {
            phi,
            scales,
            levels,
        })
    }

    // ─── Internal helpers ──────────────────────────────────────────────

    /// Build n×n covariance matrix from first-differenced window.
    ///
    /// Computes Δx[t] = x[t] - x[t-1] for consecutive snapshots, then
    /// builds covariance of the Δx series. O(n²·T) but T ≤ 50, n ≤ 128.
    /// Removes temporal autocorrelation so Phi measures co-variation in
    /// *changes* rather than persistence of a static pattern.
    fn build_differenced_covariance(&self, n: usize) -> Vec<f64> {
        let t_diff = self.window.len() - 1; // number of differences
        let inv_t = 1.0 / t_diff as f64;

        // Compute means of differences
        let mut mean = vec![0.0f64; n];
        let slices: Vec<&[f64]> = self.window.iter().map(|v| v.as_slice()).collect();
        for t in 0..t_diff {
            for i in 0..n.min(slices[t].len()).min(slices[t + 1].len()) {
                mean[i] += slices[t + 1][i] - slices[t][i];
            }
        }
        for m in mean.iter_mut() {
            *m *= inv_t;
        }

        // Covariance of differences
        let mut cov = vec![0.0f64; n * n];
        for t in 0..t_diff {
            let len = n.min(slices[t].len()).min(slices[t + 1].len());
            for i in 0..len {
                let di = (slices[t + 1][i] - slices[t][i]) - mean[i];
                for j in i..len {
                    let dj = (slices[t + 1][j] - slices[t][j]) - mean[j];
                    cov[i * n + j] += di * dj;
                }
            }
        }
        // Symmetrize and add regularization
        for i in 0..n {
            for j in (i + 1)..n {
                cov[i * n + j] *= inv_t;
                cov[j * n + i] = cov[i * n + j];
            }
            cov[i * n + i] = cov[i * n + i] * inv_t + self.config.regularization;
        }
        cov
    }

    /// Build n×n covariance matrix from online running sums.
    ///
    /// Uses `cov[i,j] = cross_sum[i,j]/T - mean[i]*mean[j]` — O(n²) from
    /// pre-computed sums, vs O(n²·T) for full window iteration.
    fn build_covariance_matrix(&self, n: usize, t: usize) -> Vec<f64> {
        let inv_t = 1.0 / t as f64;
        let mut cov = vec![0.0f64; n * n];

        for i in 0..n {
            let mean_i = self.sum[i] * inv_t;
            for j in i..n {
                let mean_j = self.sum[j] * inv_t;
                let c = self.cross_sum[i * n + j] * inv_t - mean_i * mean_j;
                cov[i * n + j] = c;
                if j > i {
                    cov[j * n + i] = c;
                }
            }
            cov[i * n + i] += self.config.regularization;
        }
        cov
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FREE FUNCTIONS (used by both SpectralMIPFinder and tests)
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert a covariance matrix to a correlation matrix in-place.
///
/// `corr[i,j] = cov[i,j] / (σ_i · σ_j)` where `σ_i = sqrt(cov[i,i])`.
/// After normalization, all diagonal entries are 1.0 and Phi measures
/// pattern integration independent of per-dimension signal amplitude.
/// Dimensions with zero variance are left untouched (regularization prevents this).
fn covariance_to_correlation(cov: &mut [f64], n: usize) {
    let inv_std: Vec<f64> = (0..n)
        .map(|i| {
            let var = cov[i * n + i];
            if var > 0.0 { 1.0 / var.sqrt() } else { 1.0 }
        })
        .collect();

    for i in 0..n {
        for j in 0..n {
            cov[i * n + j] *= inv_std[i] * inv_std[j];
        }
    }
}

/// Build the MI weight matrix and graph Laplacian from a covariance matrix.
///
/// Pairwise Gaussian MI: `MI(i,j) = -0.5 * ln(1 - r_ij²)` where r_ij is correlation.
/// Laplacian: `L = D - W` where D is diagonal degree matrix.
fn build_mi_laplacian(cov: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut weights = vec![0.0f64; n * n];

    for i in 0..n {
        for j in (i + 1)..n {
            let var_i = cov[i * n + i];
            let var_j = cov[j * n + j];
            let cov_ij = cov[i * n + j];
            let denom = var_i * var_j;
            if denom <= 0.0 {
                continue;
            }
            let r_sq = (cov_ij * cov_ij / denom).min(1.0 - 1e-15);
            // MI(i,j) = -0.5 * ln(1 - r²)
            let mi = -0.5 * (1.0 - r_sq).ln();
            weights[i * n + j] = mi;
            weights[j * n + i] = mi;
        }
    }

    // Laplacian: L = D - W
    let mut laplacian = vec![0.0f64; n * n];
    for i in 0..n {
        let degree: f64 = (0..n).map(|j| weights[i * n + j]).sum();
        for j in 0..n {
            if i == j {
                laplacian[i * n + j] = degree;
            } else {
                laplacian[i * n + j] = -weights[i * n + j];
            }
        }
    }

    (weights, laplacian)
}

/// Compute Fiedler ordering: sort indices by the second eigenvector of the Laplacian.
///
/// Uses shifted inverse iteration with deflation — O(n³/6 + k*n²) vs O(n³) full QR.
/// Only extracts the Fiedler vector (2nd eigenvector), not all eigenvalues.
///
/// Returns (spectral_order, fiedler_zero_crossing).
fn compute_fiedler_order(laplacian: &[f64], _mi_weights: &[f64], n: usize) -> (Vec<usize>, usize) {
    use nalgebra::{DMatrix, DVector};

    // Shift Laplacian to make it positive definite: M = L + σI
    // (L has eigenvalue 0 with eigenvector = constant; shift makes it invertible)
    let sigma = 1e-8;
    let mut shifted = DMatrix::from_row_slice(n, n, laplacian);
    for i in 0..n {
        shifted[(i, i)] += sigma;
    }

    // One-time Cholesky factorization: O(n³/6)
    let cholesky = match shifted.cholesky() {
        Some(c) => c,
        None => {
            // Fallback to full eigendecomposition if Cholesky fails
            return compute_fiedler_order_full(laplacian, n);
        }
    };

    // Deterministic start vector orthogonal to constant eigenvector:
    // v[i] = i - (n-1)/2, then deflate and normalize
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    let mut v = DVector::from_fn(n, |i, _| i as f64 - (n as f64 - 1.0) / 2.0);
    // Deflate: remove constant-vector component
    let dot_const: f64 = v.iter().sum::<f64>() * inv_sqrt_n;
    for vi in v.iter_mut() {
        *vi -= dot_const * inv_sqrt_n;
    }
    let norm = v.norm();
    if norm < 1e-15 {
        return compute_fiedler_order_full(laplacian, n);
    }
    v /= norm;

    // Inverse iteration with deflation: ~20-30 iterations of O(n²) each
    for _ in 0..30 {
        // Solve (L + σI) * w = v via Cholesky: O(n²)
        let w = cholesky.solve(&v);
        // Deflate: remove constant-vector component
        let dot_const: f64 = w.iter().sum::<f64>() * inv_sqrt_n;
        let mut w_deflated = w;
        for wi in w_deflated.iter_mut() {
            *wi -= dot_const * inv_sqrt_n;
        }
        let norm = w_deflated.norm();
        if norm < 1e-15 {
            break;
        }
        v = w_deflated / norm;
    }

    // v is now the Fiedler vector
    let fiedler_vec: Vec<f64> = v.iter().copied().collect();

    // Sort indices by Fiedler vector value
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| fiedler_vec[a].total_cmp(&fiedler_vec[b]));

    // Find zero-crossing in sorted Fiedler values
    let sorted_fiedler: Vec<f64> = order.iter().map(|&i| fiedler_vec[i]).collect();
    let zero_crossing = sorted_fiedler
        .windows(2)
        .position(|w| w[0] <= 0.0 && w[1] > 0.0)
        .unwrap_or(n / 2);

    (order, zero_crossing)
}

/// Fallback: full eigendecomposition via nalgebra::SymmetricEigen.
fn compute_fiedler_order_full(laplacian: &[f64], n: usize) -> (Vec<usize>, usize) {
    use nalgebra::DMatrix;

    let lap_matrix = DMatrix::from_row_slice(n, n, laplacian);
    let eigen = nalgebra::SymmetricEigen::new(lap_matrix);

    let mut eigen_indices: Vec<usize> = (0..n).collect();
    let eigenvalues = &eigen.eigenvalues;
    eigen_indices.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));

    let fiedler_idx = eigen_indices[1];
    let fiedler_vec: Vec<f64> = (0..n)
        .map(|i| eigen.eigenvectors[(i, fiedler_idx)])
        .collect();

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| fiedler_vec[a].total_cmp(&fiedler_vec[b]));

    let sorted_fiedler: Vec<f64> = order.iter().map(|&i| fiedler_vec[i]).collect();
    let zero_crossing = sorted_fiedler
        .windows(2)
        .position(|w| w[0] <= 0.0 && w[1] > 0.0)
        .unwrap_or(n / 2);

    (order, zero_crossing)
}

/// Reorder a matrix by permuting both rows and columns according to `order`.
fn reorder_matrix(matrix: &[f64], n: usize, order: &[usize]) -> Vec<f64> {
    let mut reordered = vec![0.0f64; n * n];
    for (new_i, &old_i) in order.iter().enumerate() {
        for (new_j, &old_j) in order.iter().enumerate() {
            reordered[new_i * n + new_j] = matrix[old_i * n + old_j];
        }
    }
    reordered
}

/// Sweep all n-1 contiguous bipartitions using bordered Cholesky updates.
///
/// For each cut k (left = 0..=k, right = k+1..n-1):
///   cut_mi[k] = mi_left[k] + mi_right[k]
///
/// Uses O(k²) bordered Cholesky per step → O(n³) total.
/// Left and right sweeps run in parallel via `rayon::join`.
fn sweep_contiguous_cuts(cov: &[f64], n: usize) -> Vec<f64> {
    if n < 2 {
        return vec![];
    }

    // Left and right sweeps are independent — run in parallel when available.
    #[cfg(feature = "parallel")]
    let ((left_ln_det, left_sum_ln_var), (right_ln_det, right_sum_ln_var)) =
        rayon::join(|| sweep_left(cov, n), || sweep_right(cov, n));
    #[cfg(not(feature = "parallel"))]
    let ((left_ln_det, left_sum_ln_var), (right_ln_det, right_sum_ln_var)) =
        (sweep_left(cov, n), sweep_right(cov, n));

    // ─── Combine: for cut at k, left = [0..=k], right = [k+1..n-1] ───
    let mut cut_mis = Vec::with_capacity(n - 1);
    for k in 0..(n - 1) {
        let mi_left = if k == 0 {
            0.0 // single-element partition has zero MI
        } else {
            0.5 * (left_sum_ln_var[k] - left_ln_det[k])
        };
        let mi_right = if k == n - 2 {
            0.0 // single-element partition
        } else {
            0.5 * (right_sum_ln_var[k + 1] - right_ln_det[k + 1])
        };
        cut_mis.push(mi_left + mi_right);
    }

    cut_mis
}

/// Left sweep: grow Cholesky factor from index 0 → n-1.
fn sweep_left(cov: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut ln_det = vec![0.0f64; n];
    let mut sum_ln_var = vec![0.0f64; n];
    let mut factor: Vec<f64> = Vec::with_capacity(n * (n + 1) / 2);

    let d0 = cov[0].max(1e-30);
    factor.push(d0.sqrt());
    ln_det[0] = d0.sqrt().ln() * 2.0;
    sum_ln_var[0] = d0.ln();

    for k in 1..n {
        let col: Vec<f64> = (0..k).map(|i| cov[i * n + k]).collect();
        let diag = cov[k * n + k];

        let (x, l_new) = bordered_cholesky_step(&factor, k, &col, diag);

        factor.extend_from_slice(&x);
        factor.push(l_new);

        ln_det[k] = ln_det[k - 1] + 2.0 * l_new.max(1e-300).ln();
        sum_ln_var[k] = sum_ln_var[k - 1] + cov[k * n + k].max(1e-30).ln();
    }

    (ln_det, sum_ln_var)
}

/// Right sweep: grow Cholesky factor from index n-1 → 0.
fn sweep_right(cov: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut ln_det = vec![0.0f64; n];
    let mut sum_ln_var = vec![0.0f64; n];
    let mut factor: Vec<f64> = Vec::with_capacity(n * (n + 1) / 2);

    let dn = cov[(n - 1) * n + (n - 1)].max(1e-30);
    factor.push(dn.sqrt());
    ln_det[n - 1] = dn.sqrt().ln() * 2.0;
    sum_ln_var[n - 1] = dn.ln();

    for s in 1..n {
        let p = n - 1 - s;
        // Right block elements in order added: [n-1, n-2, ..., p+1]
        // Element at right-index r → original index (n-1-r)
        let col: Vec<f64> = (0..s)
            .map(|r| {
                let orig_r = n - 1 - r;
                cov[orig_r * n + p]
            })
            .collect();
        let diag = cov[p * n + p];

        let (x, l_new) = bordered_cholesky_step(&factor, s, &col, diag);

        factor.extend_from_slice(&x);
        factor.push(l_new);

        ln_det[p] = ln_det[p + 1] + 2.0 * l_new.max(1e-300).ln();
        sum_ln_var[p] = sum_ln_var[p + 1] + cov[p * n + p].max(1e-30).ln();
    }

    (ln_det, sum_ln_var)
}

/// Bordered Cholesky update: extend a k×k Cholesky factor by one row/column.
///
/// Given L (k×k lower-triangular, packed row-major: row i has i+1 elements),
/// the new column `col` = C[0..k, k], and diagonal `diag` = C[k, k],
/// compute the new row x and diagonal l_new such that:
///   L' = [[L, 0], [x^T, l_new]]
///
/// Forward substitution: L * x = col → O(k²)
pub(crate) fn bordered_cholesky_step(
    factor: &[f64],
    k: usize,
    col: &[f64],
    diag: f64,
) -> (Vec<f64>, f64) {
    let mut x = vec![0.0f64; k];

    // Forward substitution on packed lower-triangular factor
    // Row i of the factor starts at offset i*(i+1)/2
    for i in 0..k {
        let row_start = i * (i + 1) / 2;
        let mut sum = 0.0;
        for j in 0..i {
            sum += factor[row_start + j] * x[j];
        }
        let l_ii = factor[row_start + i];
        x[i] = if l_ii.abs() > 1e-30 {
            (col[i] - sum) / l_ii
        } else {
            0.0
        };
    }

    let x_sq: f64 = x.iter().map(|v| v * v).sum();
    let l_new = (diag - x_sq).max(1e-30).sqrt();

    (x, l_new)
}

/// Total Gaussian MI of a system: MI = 0.5 * (Σ ln(σ²_i) - ln(det(Σ)))
fn gaussian_mi_total(cov: &[f64], n: usize) -> f64 {
    let sum_ln_var: f64 = (0..n).map(|i| cov[i * n + i].max(1e-30).ln()).sum();
    let ln_det = ln_determinant_cholesky(cov, n);
    0.5 * (sum_ln_var - ln_det)
}

/// ln(det(M)) via Cholesky decomposition. Falls back to diagonal if not PD.
pub(crate) fn ln_determinant_cholesky(matrix: &[f64], n: usize) -> f64 {
    let mut l = vec![0.0f64; n * n];
    let mut ok = true;
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for p in 0..j {
                sum += l[i * n + p] * l[j * n + p];
            }
            if i == j {
                let diag = matrix[i * n + i] - sum;
                if diag <= 0.0 {
                    ok = false;
                    break;
                }
                l[i * n + j] = diag.sqrt();
            } else {
                l[i * n + j] = (matrix[i * n + j] - sum) / l[j * n + j];
            }
        }
        if !ok {
            break;
        }
    }
    if ok {
        let mut result = 0.0;
        for i in 0..n {
            result += l[i * n + i].ln();
        }
        2.0 * result
    } else {
        (0..n).map(|i| matrix[i * n + i].max(1e-30).ln()).sum()
    }
}

/// Extract a submatrix from an n×n matrix given a list of row/column indices.
fn extract_submatrix(matrix: &[f64], n: usize, indices: &[usize]) -> Vec<f64> {
    let m = indices.len();
    let mut sub = vec![0.0f64; m * m];
    for (new_i, &old_i) in indices.iter().enumerate() {
        for (new_j, &old_j) in indices.iter().enumerate() {
            sub[new_i * m + new_j] = matrix[old_i * n + old_j];
        }
    }
    sub
}

/// Build focused dimension indices for the next hierarchical level.
///
/// 60% from the MIP boundary region in Fiedler order, 40% evenly spaced coverage.
fn focus_dims_for_next_level(
    full_n: usize,
    target_count: usize,
    result: &SpectralMIPResult,
) -> Vec<usize> {
    let target = target_count.min(full_n);
    let n_boundary = (target * 6 / 10).max(4);
    let n_coverage = target.saturating_sub(n_boundary);

    let mut dims: Vec<usize> = Vec::with_capacity(target);
    let mut used = vec![false; full_n];

    // 1. Boundary: dims centered on mip_bond in spectral order (already mapped to full-n)
    let spectral = &result.spectral_order;
    if !spectral.is_empty() {
        let mip_bond = result.mip_bond.min(spectral.len() - 1);
        let half = n_boundary / 2;
        let start = mip_bond.saturating_sub(half);
        let end = (start + n_boundary).min(spectral.len());
        let start = end.saturating_sub(n_boundary);
        for &idx in &spectral[start..end] {
            if idx < full_n && !used[idx] {
                dims.push(idx);
                used[idx] = true;
            }
        }
    }

    // 2. Coverage: evenly spaced across full range
    if n_coverage > 0 {
        let stride = full_n / n_coverage.max(1);
        for i in 0..n_coverage {
            let idx = (i * stride).min(full_n - 1);
            if !used[idx] {
                dims.push(idx);
                used[idx] = true;
            }
        }
    }

    // Fill remaining if needed
    let mut next = 0;
    while dims.len() < target && next < full_n {
        if !used[next] {
            dims.push(next);
            used[next] = true;
        }
        next += 1;
    }

    dims.sort_unstable();
    dims.truncate(target);
    dims
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRUCTURAL HIERARCHICAL PHI — cluster-level decomposition
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of structural (cluster-based) hierarchical Phi decomposition.
///
/// Uses the Fiedler ordering from [`SpectralMIPResult`] to identify natural
/// clusters via MI-valley detection, then decomposes Phi into three scales:
///
/// - **Micro**: average within-cluster integration (local binding)
/// - **Meso**: average pairwise MI between clusters (regional coordination)
/// - **Macro**: global spectral MIP Phi (unified consciousness)
///
/// ## Emergence Ratio
///
/// `emergence_ratio = macro_phi / (mean(cluster_phis) × num_clusters)`
///
/// - `> 1.0`: emergent integration — the whole exceeds the sum of parts
/// - `< 1.0`: weak global binding — local regions integrate but don't unify
/// - `≈ 1.0`: additive integration — no emergent or deficient binding
///
/// ## Scientific Basis
///
/// - Tononi (2004) — Integrated Information Theory (Phi decomposition)
/// - Kitazono et al. (2018) — Hierarchical MIP and spectral methods
/// - Mediano et al. (2022) — Multi-scale integrated information
#[derive(Debug, Clone)]
pub struct StructuralPhiResult {
    /// Micro-scale Φ: average Gaussian MI Phi within clusters
    pub micro_phi: f64,
    /// Meso-scale Φ: average pairwise Gaussian MI between clusters
    pub meso_phi: f64,
    /// Macro-scale Φ: global spectral MIP (same as SpectralMIPResult.phi)
    pub macro_phi: f64,
    /// Number of natural clusters detected
    pub num_clusters: usize,
    /// Size of each cluster (in spectral order)
    pub cluster_sizes: Vec<usize>,
    /// Per-cluster Phi values
    pub cluster_phis: Vec<f64>,
    /// Integration bottleneck: |macro - meso| gap (lower = better integration)
    pub bottleneck_score: f64,
    /// Emergence ratio: macro / (mean_cluster_phi × num_clusters)
    pub emergence_ratio: f64,
}

impl SpectralMIPFinder {
    /// Compute structural hierarchical Phi decomposition.
    ///
    /// Uses the Fiedler ordering from a previous [`compute()`] result to identify
    /// natural clusters (via MI-valley detection in the cut sweep), then computes
    /// Phi at three scales: micro (within-cluster), meso (between-cluster), and
    /// macro (global).
    ///
    /// For each cluster, the within-cluster Phi is computed using the same
    /// bordered Cholesky sweep algorithm as the global MIP, applied to the
    /// cluster's covariance submatrix.
    ///
    /// # Returns
    ///
    /// `None` if the window isn't ready or the MIP result is invalid.
    ///
    /// # Performance
    ///
    /// O(n²) for covariance rebuild + O(Σ m_k³) for per-cluster sweeps,
    /// where m_k is the size of cluster k. Typically 2-3ms for n=128 with
    /// 3-5 clusters.
    pub fn compute_structural_hierarchy(
        &self,
        mip_result: &SpectralMIPResult,
    ) -> Option<StructuralPhiResult> {
        if !self.ready() || mip_result.spectral_order.len() < 4 {
            return None;
        }

        let n = self.config.num_components;
        let t = self.window.len();
        let cov = self.build_effective_covariance(n, t);

        self.compute_structural_hierarchy_from_cov(mip_result, &cov, n)
    }

    /// Compute structural hierarchical Phi from a pre-built covariance matrix.
    ///
    /// Same algorithm as [`compute_structural_hierarchy`] but accepts an external
    /// covariance matrix instead of building one from the window. Useful for
    /// testing with known covariance structures.
    pub fn compute_structural_hierarchy_from_cov(
        &self,
        mip_result: &SpectralMIPResult,
        cov: &[f64],
        n: usize,
    ) -> Option<StructuralPhiResult> {
        if mip_result.spectral_order.len() < 4 || n < 4 || cov.len() != n * n {
            return None;
        }

        // Reorder covariance by spectral order (Fiedler ordering)
        let reordered = reorder_matrix(cov, n, &mip_result.spectral_order);

        // Step 1: Identify clusters from cut_mis valleys
        let clusters = find_spectral_clusters(&mip_result.cut_mis, n);
        let num_clusters = clusters.len();

        if num_clusters < 2 {
            return Some(StructuralPhiResult {
                micro_phi: mip_result.phi,
                meso_phi: 0.0,
                macro_phi: mip_result.phi,
                num_clusters: 1,
                cluster_sizes: vec![n],
                cluster_phis: vec![mip_result.phi],
                bottleneck_score: 0.0,
                emergence_ratio: 1.0,
            });
        }

        // Step 2: Compute per-cluster Phi (micro)
        let mut cluster_phis = Vec::with_capacity(num_clusters);
        let mut cluster_sizes = Vec::with_capacity(num_clusters);

        for cluster_indices in &clusters {
            let m = cluster_indices.len();
            cluster_sizes.push(m);

            if m < 2 {
                cluster_phis.push(0.0);
                continue;
            }

            // Extract submatrix for this cluster (in spectral order)
            let sub_cov = extract_submatrix(&reordered, n, cluster_indices);

            // Compute Phi: MI_total - MI_at_MIP
            let total_mi = gaussian_mi_total(&sub_cov, m);
            if m == 2 {
                // Only one possible cut for n=2
                cluster_phis.push(total_mi.max(0.0));
            } else {
                let cut_mis = sweep_contiguous_cuts(&sub_cov, m);
                let min_cut = cut_mis.iter().copied().fold(f64::MAX, f64::min);
                cluster_phis.push((total_mi - min_cut).max(0.0));
            }
        }

        let micro_phi = if !cluster_phis.is_empty() {
            cluster_phis.iter().sum::<f64>() / cluster_phis.len() as f64
        } else {
            0.0
        };

        // Step 3: Compute meso-Phi (between-cluster MI)
        let meso_phi = compute_meso_mi(&reordered, n, &clusters);

        // Step 4: Macro is the original result
        let macro_phi = mip_result.phi;

        // Step 5: Emergence and bottleneck
        let mean_cluster_phi = micro_phi;
        let expected = mean_cluster_phi * num_clusters as f64;
        let emergence_ratio = if expected > 1e-6 {
            macro_phi / expected
        } else {
            1.0
        };
        let bottleneck_score = (macro_phi - meso_phi).abs();

        Some(StructuralPhiResult {
            micro_phi,
            meso_phi,
            macro_phi,
            num_clusters,
            cluster_sizes,
            cluster_phis,
            bottleneck_score,
            emergence_ratio,
        })
    }
}

/// Identify natural cluster boundaries from the MI cut sweep.
///
/// Clusters are separated by valleys in `cut_mis` — positions where mutual
/// information between the left and right halves is significantly below the
/// median. These correspond to weak integration points in the Fiedler ordering.
///
/// Returns lists of spectral-order position indices for each cluster.
fn find_spectral_clusters(cut_mis: &[f64], n: usize) -> Vec<Vec<usize>> {
    if cut_mis.is_empty() || n < 4 {
        return vec![(0..n).collect()];
    }

    // Find the 25th percentile of cut_mis as threshold
    let mut sorted = cut_mis.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let p25_idx = sorted.len() / 4;
    let threshold = sorted[p25_idx];

    // Find local minima below threshold
    let mut boundaries: Vec<(usize, f64)> = Vec::new();
    for i in 0..cut_mis.len() {
        let is_local_min = (i == 0 || cut_mis[i] <= cut_mis[i - 1])
            && (i == cut_mis.len() - 1 || cut_mis[i] <= cut_mis[i + 1]);
        if is_local_min && cut_mis[i] < threshold {
            boundaries.push((i, cut_mis[i]));
        }
    }

    // Sort by MI value (weakest = best cluster boundary)
    boundaries.sort_by(|a, b| a.1.total_cmp(&b.1));

    // Limit cluster count: max 7 clusters (min cluster size ≈ n/7)
    let max_boundaries = ((n / 4).max(2) - 1).min(6);
    boundaries.truncate(max_boundaries);

    // Sort boundaries by position for sequential cluster creation
    let mut cut_positions: Vec<usize> = boundaries.iter().map(|(pos, _)| pos + 1).collect();
    cut_positions.sort_unstable();
    cut_positions.dedup();

    // Build clusters from boundaries
    let mut clusters = Vec::new();
    let mut start = 0;
    for &boundary in &cut_positions {
        let boundary = boundary.min(n);
        if boundary > start && boundary - start >= 2 {
            clusters.push((start..boundary).collect());
            start = boundary;
        }
    }
    // Add final cluster
    if n > start && n - start >= 2 {
        clusters.push((start..n).collect());
    }

    if clusters.is_empty() {
        vec![(0..n).collect()]
    } else {
        clusters
    }
}

/// Compute average pairwise Gaussian MI between clusters.
///
/// For each pair of clusters (A, B), computes:
///   `MI(A;B) = 0.5 × (ln|Σ_A| + ln|Σ_B| - ln|Σ_{A∪B}|)`
///
/// Returns the average pairwise inter-cluster MI.
fn compute_meso_mi(cov: &[f64], n: usize, clusters: &[Vec<usize>]) -> f64 {
    if clusters.len() < 2 {
        return 0.0;
    }

    // Cache per-cluster log determinants
    let ln_dets: Vec<f64> = clusters
        .iter()
        .map(|c| {
            let sub = extract_submatrix(cov, n, c);
            ln_determinant_cholesky(&sub, c.len())
        })
        .collect();

    let mut total_mi = 0.0;
    let mut pair_count = 0;

    for i in 0..clusters.len() {
        for j in (i + 1)..clusters.len() {
            let mut combined: Vec<usize> =
                Vec::with_capacity(clusters[i].len() + clusters[j].len());
            combined.extend_from_slice(&clusters[i]);
            combined.extend_from_slice(&clusters[j]);
            combined.sort_unstable();

            let sub = extract_submatrix(cov, n, &combined);
            let ln_det_ab = ln_determinant_cholesky(&sub, combined.len());

            // MI(A;B) = 0.5 × (ln|Σ_A| + ln|Σ_B| - ln|Σ_{A∪B}|)
            let mi = 0.5 * (ln_dets[i] + ln_dets[j] - ln_det_ab);
            total_mi += mi.max(0.0);
            pair_count += 1;
        }
    }

    if pair_count > 0 {
        total_mi / pair_count as f64
    } else {
        0.0
    }
}

// Tests in consciousness_metrics/tests/spectral_mip_tests.rs
