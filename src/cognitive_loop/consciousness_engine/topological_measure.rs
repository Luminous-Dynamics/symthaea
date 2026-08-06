// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Layer 5: Topological consciousness measurement via Hodge decomposition.
//!
//! Complements the spectral Phi (Layer 1) with a topological characterization
//! of consciousness. While spectral Phi measures integrated information via
//! eigenvalue decomposition, topological Phi measures irreducible harmonic
//! modes — information flow that is topologically protected and cannot be
//! decomposed into feedforward (gradient) or feedback (curl) components.
//!
//! # Theory
//!
//! Harmonic modes in the Hodge decomposition correspond to topologically
//! irreducible information integration (Petri et al. 2014). A system with
//! many independent loops (high beta_1) and voids (high beta_2) has richer
//! topological structure — analogous to how high Phi indicates integrated
//! information that exists only in the whole.
//!
//! # Pipeline
//!
//! 1. Take CfC hidden state dimensions (subsampled to keep O(n^3) tractable)
//! 2. Build correlation matrix from pairwise dimension correlations
//! 3. Threshold into a `ConsciousnessComplex` (simplicial complex)
//! 4. Run `HodgeConsciousnessDecomposition::decompose()` on edge signals
//! 5. Compute `TopologicalPhi::from_decomposition()`
//! 6. Optionally estimate manifold geometry from a state window
//!
//! # References
//!
//! - Hodge (1941): Harmonic integrals
//! - Petri et al. (2014): Topological strata of weighted complex networks
//! - Tononi (2004): IIT — complementary spectral characterization
//! - Eckmann (1944): Simplicial Hodge theory

use symthaea_hodge::consciousness_topology::{
    ConsciousnessComplex, ConsciousnessManifold, HodgeConsciousnessDecomposition, TopologicalPhi,
};

/// Topological consciousness measurement result.
///
/// Contains the full Hodge decomposition analysis of the current
/// consciousness state, providing a complementary view to spectral Phi.
#[derive(Debug, Clone)]
pub struct TopologicalConsciousnessResult {
    /// Phi from harmonic energy ratio — fraction of total signal energy
    /// that resides in topologically-protected harmonic modes [0, 1].
    pub phi_harmonic: f64,
    /// Phi from Betti number structure — weighted combination of
    /// independent loops (beta_1) and higher-order voids (beta_2) [0, 1].
    pub phi_betti: f64,
    /// Combined topological consciousness score — geometric mean of
    /// harmonic and Betti measures [0, 1].
    pub phi_topological: f64,
    /// Betti numbers: [beta_0 (components), beta_1 (loops), beta_2 (voids)].
    pub betti_numbers: Vec<usize>,
    /// Euler characteristic: chi = sum(-1)^k * beta_k.
    pub euler_characteristic: i64,
    /// Ratio of harmonic energy to total energy [0, 1].
    pub harmonic_ratio: f64,
    /// Intrinsic dimensionality of the consciousness manifold.
    /// Only populated when a state window is provided.
    pub manifold_dimension: usize,
    /// Ricci scalar curvature at the current state.
    /// Only populated when a state window is provided.
    pub manifold_curvature: f64,
}

impl Default for TopologicalConsciousnessResult {
    fn default() -> Self {
        Self {
            phi_harmonic: 0.0,
            phi_betti: 0.0,
            phi_topological: 0.0,
            betti_numbers: vec![0, 0, 0],
            euler_characteristic: 0,
            harmonic_ratio: 0.0,
            manifold_dimension: 0,
            manifold_curvature: 0.0,
        }
    }
}

/// Maximum number of dimensions to subsample for the correlation matrix.
/// Keeps the O(n^3) Hodge Laplacian tractable within the 20Hz budget.
const MAX_TOPO_DIMENSIONS: usize = 32;

/// Correlation threshold for building the simplicial complex.
/// Edges with |correlation| below this are excluded.
const CORRELATION_THRESHOLD: f64 = 0.3;

/// Compute topological consciousness from a CfC hidden state snapshot.
///
/// Subsamples the hidden state to `MAX_TOPO_DIMENSIONS` evenly-spaced
/// dimensions, builds a correlation matrix from those dimensions,
/// constructs a simplicial complex, and runs the full Hodge pipeline.
///
/// # Arguments
///
/// * `hidden_state` - CfC hidden state as a slice of f64 values
/// * `state_window` - Optional window of recent states for manifold estimation
///
/// # Returns
///
/// A `TopologicalConsciousnessResult` with all topological consciousness metrics.
pub fn compute_topological_consciousness(
    hidden_state: &[f64],
    state_window: Option<&[Vec<f64>]>,
) -> TopologicalConsciousnessResult {
    if hidden_state.len() < 3 {
        return TopologicalConsciousnessResult::default();
    }

    // Subsample to keep computation tractable
    let dim = hidden_state.len();
    let n = dim.min(MAX_TOPO_DIMENSIONS);
    let step = if n >= dim { 1 } else { dim / n };

    let sampled: Vec<f64> = (0..n).map(|i| hidden_state[i * step]).collect();

    // Build the correlation matrix. Prefer a real temporal correlation across the state window;
    // fall back to the single-snapshot path only when no usable window exists.
    //
    // WHY THIS MATTERS (fixed 2026-07-30): the single-snapshot path is
    // `(a*b) / (|a|*|b|)`, which is algebraically `sign(a) * sign(b)` — exactly ±1 for every
    // non-zero pair. Against CORRELATION_THRESHOLD every pair therefore became an edge and every
    // triple a triangle, making the complex the complete 2-skeleton and the Betti numbers a
    // CONSTANT independent of cognitive state (β = [1, 0, C(n-1, 3)]). The art path consumed
    // that constant while ~10^9 f64 ops per refresh computed it.
    let correlation_matrix = match state_window {
        Some(window) if window.len() >= MIN_TEMPORAL_SAMPLES => {
            build_temporal_correlation_matrix(window, n, step)
        }
        _ => build_correlation_matrix(&sampled),
    };

    // Construct simplicial complex
    let complex =
        ConsciousnessComplex::from_correlation_matrix(&correlation_matrix, CORRELATION_THRESHOLD);

    if complex.edges.is_empty() {
        // No edges above threshold — disconnected state
        let betti = vec![n, 0, 0];
        return TopologicalConsciousnessResult {
            phi_harmonic: 0.0,
            phi_betti: 0.0,
            phi_topological: 0.0,
            betti_numbers: betti,
            euler_characteristic: n as i64,
            harmonic_ratio: 0.0,
            manifold_dimension: 0,
            manifold_curvature: 0.0,
        };
    }

    // Build edge signal from the sampled state: for each edge (i,j),
    // the signal is the difference in activation (gradient proxy).
    let edge_signal: Vec<f64> = complex
        .edges
        .iter()
        .map(|&(i, j, _)| sampled[i] - sampled[j])
        .collect();

    // Run Hodge decomposition
    let decomposition = HodgeConsciousnessDecomposition::decompose(&complex, &edge_signal);

    // Compute topological Phi
    let topo_phi = TopologicalPhi::from_decomposition(&decomposition);

    // Manifold geometry (optional, from state window)
    let (manifold_dimension, manifold_curvature) = if let Some(window) = state_window {
        if window.len() >= 3 {
            // Subsample each state in the window to match our dimensionality
            let subsampled_window: Vec<Vec<f64>> = window
                .iter()
                .map(|state| {
                    let d = state.len();
                    let s = if n >= d { 1 } else { d / n };
                    (0..n.min(d)).map(|i| state[i * s]).collect()
                })
                .collect();
            let manifold = ConsciousnessManifold::from_state_window(&subsampled_window);
            (manifold.dimension, manifold.curvature_scalar)
        } else {
            (0, 0.0)
        }
    } else {
        (0, 0.0)
    };

    TopologicalConsciousnessResult {
        phi_harmonic: topo_phi.phi_harmonic,
        phi_betti: topo_phi.phi_betti,
        phi_topological: topo_phi.phi_topological,
        betti_numbers: decomposition.betti_numbers,
        euler_characteristic: decomposition.euler_characteristic,
        harmonic_ratio: decomposition.harmonic_ratio,
        manifold_dimension,
        manifold_curvature,
    }
}

/// Build a correlation matrix from a single state snapshot.
///
/// Uses normalized outer products: corr(i,j) = x_i * x_j / (|x_i| * |x_j| + eps).
/// This gives a symmetric matrix with values in [-1, 1] suitable for
/// thresholding into a simplicial complex.
/// Minimum number of time samples before a temporal correlation is meaningful.
/// Pearson correlation over fewer than 3 samples is degenerate (2 points are always perfectly
/// correlated or anti-correlated, which would reproduce the ±1 pathology this replaced).
const MIN_TEMPORAL_SAMPLES: usize = 3;

/// Pearson correlation between each pair of sampled dimensions, computed ACROSS TIME.
///
/// This is what a correlation matrix requires and what the single-snapshot path cannot provide:
/// correlation is a statement about co-variation over samples, and one sample has no variation.
///
/// `window` is oldest-to-newest; each element is a full state vector. `n`/`step` mirror the
/// caller's subsampling so the dimensions here are index-identical to `sampled`.
fn build_temporal_correlation_matrix(window: &[Vec<f64>], n: usize, step: usize) -> Vec<Vec<f64>> {
    let t = window.len();

    // Gather each sampled dimension's trajectory over the window.
    let mut series: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let idx = i * step;
        series.push(
            window
                .iter()
                .map(|st| st.get(idx).copied().unwrap_or(0.0))
                .collect(),
        );
    }

    let means: Vec<f64> = series
        .iter()
        .map(|v| v.iter().sum::<f64>() / t as f64)
        .collect();
    // Population standard deviation; only its ratio matters here.
    let sds: Vec<f64> = series
        .iter()
        .zip(&means)
        .map(|(v, &m)| (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / t as f64).sqrt())
        .collect();

    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n {
            // A dimension that never varies has no correlation with anything — reporting 0 is
            // correct and, importantly, keeps it OUT of the complex rather than fabricating an
            // edge. The old code gave a constant dimension ±1 against every other dimension.
            let corr = if sds[i] <= f64::EPSILON || sds[j] <= f64::EPSILON {
                0.0
            } else {
                let cov = series[i]
                    .iter()
                    .zip(&series[j])
                    .map(|(a, b)| (a - means[i]) * (b - means[j]))
                    .sum::<f64>()
                    / t as f64;
                (cov / (sds[i] * sds[j])).clamp(-1.0, 1.0)
            };
            matrix[i][j] = corr;
            matrix[j][i] = corr;
        }
    }
    matrix
}

/// Degenerate single-snapshot fallback, used only when no temporal window is available.
///
/// **This cannot compute a correlation.** A single sample has no variance, so there is nothing to
/// correlate. It returns the identity matrix, which yields a fully disconnected complex
/// (β = [n, 0, 0]) — an honest "no topological structure observed" rather than the previous
/// behaviour, which returned `sign(a)*sign(b)` (always ±1) and therefore fabricated a COMPLETE
/// graph on every call, producing a state-independent constant.
///
/// If you find yourself relying on this path, supply a real window instead.
fn build_correlation_matrix(state: &[f64]) -> Vec<Vec<f64>> {
    let n = state.len();
    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        matrix[i][i] = 1.0;
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highly_connected_higher_phi_than_sparse() {
        // Highly connected: all dimensions have similar strong positive activations
        // This should produce many edges and triangles -> higher topological Phi.
        let connected_state: Vec<f64> = (0..16).map(|i| 0.8 + 0.1 * (i as f64 / 16.0)).collect();

        // Sparse: alternating positive/negative with varying magnitudes
        // This should produce fewer high-correlation edges -> lower topological Phi.
        let sparse_state: Vec<f64> = (0..16)
            .map(|i| if i % 3 == 0 { 0.9 } else { -0.1 * (i as f64) })
            .collect();

        let connected_result = compute_topological_consciousness(&connected_state, None);
        let sparse_result = compute_topological_consciousness(&sparse_state, None);

        assert!(
            connected_result.phi_topological >= sparse_result.phi_topological,
            "Highly connected state (phi_topo={}) should have >= topological Phi than sparse state (phi_topo={})",
            connected_result.phi_topological,
            sparse_result.phi_topological,
        );
    }

    #[test]
    fn test_disconnected_state_has_near_zero_phi() {
        // Very low activations — no correlations above threshold
        let disconnected: Vec<f64> = vec![0.01, -0.02, 0.01, -0.01, 0.02, -0.01, 0.01, -0.02];

        let result = compute_topological_consciousness(&disconnected, None);

        assert!(
            result.phi_topological < 0.01,
            "Disconnected state should have near-zero phi_topological, got {}",
            result.phi_topological,
        );
        assert!(
            result.phi_harmonic < 0.01,
            "Disconnected state should have near-zero phi_harmonic, got {}",
            result.phi_harmonic,
        );
    }

    #[test]
    fn test_betti_numbers_for_known_connectivity() {
        // Build a state that produces a cycle graph (4 nodes in a ring).
        // Nodes 0,1 correlated; 1,2 correlated; 2,3 correlated; 3,0 correlated;
        // but 0,2 and 1,3 not correlated -> cycle -> beta_1 = 1.
        //
        // We construct this by using the correlation matrix directly.
        let matrix = vec![
            vec![1.0, 0.8, 0.0, 0.8],
            vec![0.8, 1.0, 0.8, 0.0],
            vec![0.0, 0.8, 1.0, 0.8],
            vec![0.8, 0.0, 0.8, 1.0],
        ];

        let complex = ConsciousnessComplex::from_correlation_matrix(&matrix, CORRELATION_THRESHOLD);
        let betti = complex.betti_numbers();

        assert_eq!(betti[0], 1, "Should have 1 connected component");
        assert_eq!(
            betti[1], 1,
            "Cycle should have beta_1 = 1 (one independent loop)"
        );
        assert_eq!(betti[2], 0, "No triangles -> beta_2 = 0");
    }

    #[test]
    fn test_bridge_produces_valid_bounded_outputs() {
        // Random-ish state
        let state: Vec<f64> = (0..32).map(|i| ((i as f64) * 0.37).sin()).collect();

        let result = compute_topological_consciousness(&state, None);

        // All Phi values should be non-negative and bounded
        assert!(
            result.phi_harmonic >= 0.0 && result.phi_harmonic <= 1.0,
            "phi_harmonic out of bounds: {}",
            result.phi_harmonic,
        );
        assert!(
            result.phi_betti >= 0.0 && result.phi_betti <= 1.0,
            "phi_betti out of bounds: {}",
            result.phi_betti,
        );
        assert!(
            result.phi_topological >= 0.0 && result.phi_topological <= 1.0,
            "phi_topological out of bounds: {}",
            result.phi_topological,
        );
        assert!(
            result.harmonic_ratio >= 0.0 && result.harmonic_ratio <= 1.0,
            "harmonic_ratio out of bounds: {}",
            result.harmonic_ratio,
        );

        // Betti numbers should have at least 3 entries
        assert!(
            result.betti_numbers.len() >= 3,
            "Expected at least 3 Betti numbers, got {}",
            result.betti_numbers.len(),
        );

        // Euler characteristic should be consistent with Betti numbers
        let expected_euler: i64 = result
            .betti_numbers
            .iter()
            .enumerate()
            .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
            .sum();
        assert_eq!(
            result.euler_characteristic, expected_euler,
            "Euler characteristic inconsistent with Betti numbers",
        );
    }

    #[test]
    fn test_manifold_geometry_from_state_window() {
        // Build a window of states tracing a curve in state space
        let window: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let t = i as f64 * 0.1;
                (0..8).map(|j| (t + j as f64 * 0.5).sin()).collect()
            })
            .collect();

        let state = &window[window.len() - 1];
        let result = compute_topological_consciousness(state, Some(&window));

        assert!(
            result.manifold_dimension > 0,
            "Manifold dimension should be > 0 for a non-degenerate trajectory",
        );
        assert!(
            result.manifold_curvature >= 0.0,
            "Manifold curvature should be non-negative for a smooth curve",
        );
    }

    /// PAIRED CONTROL for the 2026-07-30 sign-collapse fix.
    ///
    /// The old `build_correlation_matrix` computed `(a*b)/(|a|*|b|)`, which is algebraically
    /// `sign(a)*sign(b)` — exactly ±1 for every non-zero pair. Every pair therefore cleared
    /// CORRELATION_THRESHOLD, the complex became the complete 2-skeleton, and the Betti numbers
    /// were a CONSTANT regardless of input.
    ///
    /// This test would FAIL against that code: it asserts the Betti numbers actually VARY with
    /// the state. A test that merely checked "returns numbers" would have passed on the bug.
    #[test]
    fn betti_numbers_vary_with_state_given_a_real_window() {
        // Window A: two groups of dimensions moving in opposite phase -> real structure.
        let mut window_a: Vec<Vec<f64>> = Vec::new();
        for t in 0..12 {
            let phase = t as f64 * 0.5;
            window_a.push(
                (0..64)
                    .map(|d| {
                        if d % 2 == 0 {
                            phase.sin()
                        } else {
                            (phase + std::f64::consts::PI).sin()
                        }
                    })
                    .collect(),
            );
        }
        // Window B: every dimension follows an independent, unrelated ramp -> different structure.
        let mut window_b: Vec<Vec<f64>> = Vec::new();
        for t in 0..12 {
            window_b.push(
                (0..64)
                    .map(|d| ((t * 7 + d * 13) % 23) as f64 / 23.0)
                    .collect(),
            );
        }

        let state = vec![0.5_f64; 64];
        let a = compute_topological_consciousness(&state, Some(&window_a));
        let b = compute_topological_consciousness(&state, Some(&window_b));

        assert_ne!(
            a.betti_numbers, b.betti_numbers,
            "Betti numbers must depend on the state window. Identical values across structurally \
             different windows is the signature of the sign-collapse bug (fixed 2026-07-30), \
             which made them a constant."
        );
    }

    /// The degenerate single-snapshot path must report NO structure rather than fabricate a
    /// complete graph. Pre-fix this returned beta = [1, 0, C(n-1,3)]; now it returns a fully
    /// disconnected complex, which is the honest answer when there is nothing to correlate.
    #[test]
    fn single_snapshot_path_reports_no_structure_not_a_complete_graph() {
        let state: Vec<f64> = (0..64).map(|i| (i as f64 * 0.37).sin()).collect();
        let r = compute_topological_consciousness(&state, None);
        let n = 64usize.min(MAX_TOPO_DIMENSIONS);
        assert_eq!(
            r.betti_numbers[0], n,
            "identity correlation must give one component per dimension (fully disconnected)"
        );
        assert_eq!(r.betti_numbers[1], 0, "no loops without edges");
        assert_eq!(
            r.betti_numbers[2], 0,
            "no voids without triangles — pre-fix this was C(n-1,3) = 4495 for n=32"
        );
    }

    /// A constant dimension has no variance and therefore no correlation with anything. Pre-fix it
    /// received ±1 against every other dimension, which is how a flat signal fabricated edges.
    #[test]
    fn constant_dimensions_produce_no_edges() {
        let window: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0_f64; 64]).collect();
        let state = vec![1.0_f64; 64];
        let r = compute_topological_consciousness(&state, Some(&window));
        let n = 64usize.min(MAX_TOPO_DIMENSIONS);
        assert_eq!(
            r.betti_numbers[0], n,
            "a wholly constant window must yield no edges, hence n components"
        );
    }
}
