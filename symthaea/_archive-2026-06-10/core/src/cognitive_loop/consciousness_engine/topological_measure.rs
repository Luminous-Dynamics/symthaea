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

    // Build correlation matrix from the sampled dimensions.
    // Since we have a single snapshot, we use pairwise products as a
    // proxy for connectivity strength (normalized by signal magnitude).
    let correlation_matrix = build_correlation_matrix(&sampled);

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
fn build_correlation_matrix(state: &[f64]) -> Vec<Vec<f64>> {
    let n = state.len();
    let eps = 1e-12;

    // Compute magnitudes for normalization
    let magnitudes: Vec<f64> = state.iter().map(|x| x.abs().max(eps)).collect();

    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n {
            let corr = (state[i] * state[j]) / (magnitudes[i] * magnitudes[j]);
            matrix[i][j] = corr;
            matrix[j][i] = corr;
        }
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
}
