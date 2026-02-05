//! Algebraic Connectivity (Spectral Graph Metric) for RealHV
//!
//! # Purpose
//!
//! This module measures **algebraic connectivity** (λ₂) - a spectral graph property
//! that quantifies how well-connected a network is for information flow.
//!
//! # Important: NOT IIT Φ
//!
//! This module computes **spectral graph properties**, NOT Integrated Information
//! Theory's Φ (integrated information). See `METRIC_CLARIFICATION.md` for details.
//!
//! For IIT-aligned consciousness research, use:
//! - `tiered_phi/core.rs` (Exact tier) - True MIP-based IIT calculation
//! - `integrated_information.rs` - IIT formula implementations
//!
//! # Algorithm
//!
//! 1. Compute pairwise cosine similarities between all RealHV components
//! 2. Build weighted similarity matrix (adjacency matrix)
//! 3. Construct normalized graph Laplacian: L = I - D^(-1/2) A D^(-1/2)
//! 4. Compute eigenvalues using `nalgebra::SymmetricEigen` (QR algorithm)
//! 5. Return λ₂ / 2 as the connectivity value in [0, 1]
//!
//! # What λ₂ Measures
//!
//! | Property | λ₂ (Algebraic Connectivity) |
//! |----------|----------------------------|
//! | Measures | Graph mixing time / spectral gap |
//! | Basis | Spectral graph theory |
//! | Complexity | O(n³) |
//! | Favors | Uniform k-regular graphs |
//!
//! # Valid Use Cases
//!
//! ✅ Network connectivity analysis
//! ✅ Synchronization potential estimation
//! ✅ Graph mixing properties
//! ✅ Topology comparison (for connectivity, NOT consciousness)
//!
//! # Invalid Use Cases
//!
//! ❌ Consciousness measurement claims
//! ❌ IIT Φ approximation (r = -0.62 correlation - nearly opposite!)
//! ❌ Integrated information estimation
//!
//! # Example
//!
//! ```rust,ignore
//! use symthaea::hdc::{
//!     real_hv::RealHV,
//!     spectral_connectivity::ConnectivityCalculator,
//! };
//!
//! let components = vec![
//!     RealHV::random(RealHV::DEFAULT_DIM, 1),
//!     RealHV::random(RealHV::DEFAULT_DIM, 2),
//!     RealHV::random(RealHV::DEFAULT_DIM, 3),
//! ];
//!
//! let calculator = ConnectivityCalculator::new();
//! let lambda2 = calculator.algebraic_connectivity(&components);
//! println!("λ₂ = {:.4}", lambda2);
//! ```

use crate::hdc::real_hv::RealHV;
use nalgebra::DMatrix;

/// Spectral connectivity calculator using algebraic connectivity (λ₂)
///
/// Computes the Fiedler value (second smallest eigenvalue of the normalized
/// graph Laplacian) on continuous hypervector similarity graphs. This measures
/// the **spectral gap** - how well-connected the graph is for information flow.
///
/// # What This Measures
///
/// - **Mixing time**: How fast information spreads across the network
/// - **Partitioning robustness**: How resistant the graph is to cuts
/// - **Synchronization potential**: How easily the network can synchronize
///
/// # What This Does NOT Measure
///
/// - IIT Φ (integrated information) - correlation r = -0.62 (nearly opposite!)
/// - Consciousness or awareness
/// - Information integration in the IIT sense
///
/// # Eigenvalue Computation
///
/// Uses `nalgebra::SymmetricEigen` (QR algorithm) for validated, numerically
/// stable eigenvalue calculation. The normalized Laplacian ensures eigenvalues
/// are bounded in [0, 2] for consistent scaling.
#[derive(Clone, Debug)]
pub struct ConnectivityCalculator {
    /// Minimum algebraic connectivity (for normalization)
    min_connectivity: f64,
    /// Maximum algebraic connectivity (theoretical upper bound, used as fallback)
    max_connectivity: f64,
}

impl ConnectivityCalculator {
    /// Create new connectivity calculator
    pub fn new() -> Self {
        Self {
            min_connectivity: 0.0,
            max_connectivity: 2.0, // Theoretical max for normalized Laplacian
        }
    }

    /// Compute algebraic connectivity (λ₂) for a set of RealHV components
    ///
    /// # Arguments
    /// * `components` - Vector of RealHV representations (nodes in graph topology)
    ///
    /// # Returns
    /// λ₂ value normalized to range [0, 1], where:
    /// - 0 = Disconnected components (no information flow)
    /// - 1 = Maximum connectivity (fastest mixing time)
    ///
    /// # Note
    /// This measures **spectral connectivity**, NOT IIT integrated information.
    pub fn algebraic_connectivity(&self, components: &[RealHV]) -> f64 {
        let n = components.len();

        if n < 2 {
            return 0.0; // No connectivity with single component
        }

        // Step 1: Build cosine similarity matrix
        let similarity_matrix = self.build_similarity_matrix(components);

        // Step 2: Compute algebraic connectivity
        let lambda2 = self.compute_lambda2(&similarity_matrix);

        // Step 3: Normalize to [0, 1]
        // With normalized Laplacian, eigenvalues are bounded [0, 2]
        let normalized = (lambda2 / self.max_connectivity).clamp(0.0, 1.0);

        normalized
    }

    /// Deprecated: Use `algebraic_connectivity()` instead
    ///
    /// This method is kept for backward compatibility but will be removed
    /// in a future version. The name "compute" was misleading as it implied
    /// IIT Φ calculation, which this does NOT perform.
    #[deprecated(since = "0.5.0", note = "Use algebraic_connectivity() instead - this measures λ₂, NOT IIT Φ")]
    pub fn compute(&self, components: &[RealHV]) -> f64 {
        self.algebraic_connectivity(components)
    }

    /// Build pairwise cosine similarity matrix (optimized: exploits symmetry)
    ///
    /// Returns n×n matrix where entry (i,j) is cosine similarity between
    /// components[i] and components[j], normalized to [0, 1].
    ///
    /// Optimization: Since similarity(i,j) = similarity(j,i), we only compute
    /// n(n-1)/2 similarities instead of n², reducing computation by ~50%.
    pub fn build_similarity_matrix(&self, components: &[RealHV]) -> Vec<Vec<f64>> {
        let n = components.len();
        let mut matrix = vec![vec![0.0_f64; n]; n];

        // Set diagonal to 1.0 (perfect self-similarity)
        for i in 0..n {
            matrix[i][i] = 1.0;
        }

        // Only compute upper triangle, then copy to lower (symmetry optimization)
        for i in 0..n {
            for j in (i + 1)..n {
                // Compute cosine similarity
                let similarity = components[i].similarity(&components[j]);

                // Normalize from [-1, 1] to [0, 1]
                let normalized_similarity = (similarity as f64 + 1.0) / 2.0;

                // Set both (i,j) and (j,i) - symmetry!
                matrix[i][j] = normalized_similarity;
                matrix[j][i] = normalized_similarity;
            }
        }

        matrix
    }

    /// Compute λ₂ (second smallest eigenvalue of normalized Laplacian)
    ///
    /// Uses the **normalized Laplacian** for proper scaling across graph sizes:
    /// L_norm = I - D^(-1/2) * A * D^(-1/2)
    ///
    /// Key properties:
    /// - Eigenvalues are ALWAYS in [0, 2] regardless of graph size
    /// - λ₁ = 0 for connected graphs
    /// - λ₂ = algebraic connectivity (Fiedler value)
    pub fn compute_lambda2(&self, similarity_matrix: &[Vec<f64>]) -> f64 {
        let n = similarity_matrix.len();

        if n < 2 {
            return 0.0;
        }

        // Step 1: Compute degrees (D[i] = sum of edge weights from node i)
        let mut degrees: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    degrees[i] += similarity_matrix[i][j];
                }
            }
        }

        // Step 2: Compute D^(-1/2) for normalization
        let inv_sqrt_degrees: Vec<f64> = degrees.iter()
            .map(|&d| if d > 1e-10 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        // Step 3: Build normalized Laplacian: L_norm = I - D^(-1/2) * A * D^(-1/2)
        let mut laplacian_data = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                if i == j {
                    // Diagonal: L_norm[i,i] = 1 (if degree > 0)
                    laplacian_data[idx] = if degrees[i] > 1e-10 { 1.0 } else { 0.0 };
                } else {
                    // Off-diagonal: L_norm[i,j] = -A[i,j] / sqrt(D[i] * D[j])
                    let normalization = inv_sqrt_degrees[i] * inv_sqrt_degrees[j];
                    laplacian_data[idx] = -similarity_matrix[i][j] * normalization;
                }
            }
        }

        // Step 4: Convert to nalgebra DMatrix and compute eigenvalues
        let laplacian = DMatrix::from_row_slice(n, n, &laplacian_data);

        // Use SymmetricEigen for validated eigenvalue computation
        let eigen = nalgebra::SymmetricEigen::new(laplacian);

        // Get eigenvalues and sort them in ascending order
        let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().cloned().collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // λ₂ is the 2nd smallest eigenvalue
        if eigenvalues.len() >= 2 {
            eigenvalues[1].max(0.0)
        } else {
            0.0
        }
    }

}

impl Default for ConnectivityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// BACKWARD COMPATIBILITY ALIASES
// ============================================================================

/// Deprecated type alias for backward compatibility
///
/// Use `ConnectivityCalculator` instead. This alias will be removed in v1.0.0.
#[deprecated(since = "0.5.0", note = "Use ConnectivityCalculator instead - this measures λ₂, NOT IIT Φ")]
pub type RealPhiCalculator = ConnectivityCalculator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_creation() {
        let calc = ConnectivityCalculator::new();
        assert_eq!(calc.min_connectivity, 0.0);
        assert_eq!(calc.max_connectivity, 2.0);
    }

    #[test]
    fn test_single_component() {
        let calc = ConnectivityCalculator::new();
        let components = vec![RealHV::random(256, 42)];
        let lambda2 = calc.algebraic_connectivity(&components);
        assert_eq!(lambda2, 0.0, "Single component should have λ₂ = 0");
    }

    #[test]
    fn test_two_identical_components() {
        let calc = ConnectivityCalculator::new();
        let hv = RealHV::random(256, 42);
        let components = vec![hv.clone(), hv.clone()];

        let lambda2 = calc.algebraic_connectivity(&components);

        assert!(lambda2 >= 0.0 && lambda2 <= 1.0, "λ₂ should be in valid range: {}", lambda2);
    }

    #[test]
    fn test_orthogonal_components() {
        let calc = ConnectivityCalculator::new();

        let components = vec![
            RealHV::basis(0, 256),
            RealHV::basis(1, 256),
            RealHV::basis(2, 256),
        ];

        let lambda2 = calc.algebraic_connectivity(&components);

        assert!(lambda2 > 0.0, "Orthogonal components should have λ₂ > 0");
        assert!(lambda2 < 1.0, "Finite λ₂ should be < 1.0");
    }

    #[test]
    fn test_similarity_matrix() {
        let calc = ConnectivityCalculator::new();

        let components = vec![
            RealHV::random(128, 1),
            RealHV::random(128, 2),
        ];

        let matrix = calc.build_similarity_matrix(&components);

        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
        assert!((matrix[0][0] - 1.0).abs() < 1e-10);
        assert!((matrix[1][1] - 1.0).abs() < 1e-10);
        assert!((matrix[0][1] - matrix[1][0]).abs() < 1e-10);
        assert!(matrix[0][1] >= 0.0 && matrix[0][1] <= 1.0);
    }

    #[test]
    fn test_lambda2_three_nodes() {
        let calc = ConnectivityCalculator::new();

        let matrix_high = vec![
            vec![1.0, 0.9, 0.8],
            vec![0.9, 1.0, 0.85],
            vec![0.8, 0.85, 1.0],
        ];
        let conn_high = calc.compute_lambda2(&matrix_high);

        let matrix_low = vec![
            vec![1.0, 0.1, 0.05],
            vec![0.1, 1.0, 0.15],
            vec![0.05, 0.15, 1.0],
        ];
        let conn_low = calc.compute_lambda2(&matrix_low);

        assert!(conn_high > 0.0, "λ₂ should be positive: {}", conn_high);
        assert!(conn_low > 0.0, "λ₂ should be positive: {}", conn_low);
    }

    #[test]
    fn test_complete_graph_k3() {
        let calc = ConnectivityCalculator::new();

        let matrix_k3 = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        let lambda2_k3 = calc.compute_lambda2(&matrix_k3);

        assert!((lambda2_k3 - 1.5).abs() < 0.01, "K3 λ₂ should be ~1.5, got {}", lambda2_k3);
    }

    #[test]
    #[allow(deprecated)]
    fn test_backward_compatibility() {
        let calc: RealPhiCalculator = RealPhiCalculator::new();
        let components = vec![
            RealHV::random(128, 1),
            RealHV::random(128, 2),
        ];

        let result = calc.compute(&components);
        assert!(result >= 0.0 && result <= 1.0);

        let new_result = calc.algebraic_connectivity(&components);
        assert!((result - new_result).abs() < 1e-10);
    }
}
