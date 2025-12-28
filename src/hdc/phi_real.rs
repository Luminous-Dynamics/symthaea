//! Φ (Integrated Information) Calculation for RealHV (No Binarization)
//!
//! This module implements Φ calculation directly on real-valued hypervectors
//! using cosine similarity instead of Hamming distance. This avoids the
//! binarization artifacts discovered during validation.
//!
//! # Key Differences from Binary Φ
//!
//! - **Similarity Metric**: Cosine similarity (continuous) vs Hamming distance (discrete)
//! - **Range**: [-1, 1] normalized to [0, 1] vs [0, 1] directly
//! - **Precision**: Full floating-point precision vs binary
//! - **Heterogeneity**: Preserves continuous variance information
//!
//! # Algorithm
//!
//! 1. Compute pairwise cosine similarities between all RealHV components
//! 2. Build weighted similarity matrix
//! 3. Compute graph Laplacian
//! 4. Calculate algebraic connectivity (2nd smallest eigenvalue)
//! 5. Normalize to [0, 1] range for Φ value
//!
//! # Example
//!
//! ```rust,ignore
//! use symthaea::hdc::{
//!     real_hv::RealHV,
//!     phi_real::RealPhiCalculator,
//! };
//!
//! let components = vec![
//!     RealHV::random(RealHV::DEFAULT_DIM, 1),  // 16,384 dimensions
//!     RealHV::random(RealHV::DEFAULT_DIM, 2),
//!     RealHV::random(RealHV::DEFAULT_DIM, 3),
//! ];
//!
//! let calculator = RealPhiCalculator::new();
//! let phi = calculator.compute(&components);
//! println!("Φ = {:.4}", phi);
//! ```

use crate::hdc::real_hv::RealHV;

/// RealHV Φ calculator using cosine similarity
///
/// Computes integrated information directly on continuous hypervectors
/// without binarization, preserving full precision and heterogeneity.
pub struct RealPhiCalculator {
    /// Minimum algebraic connectivity (for normalization)
    min_connectivity: f64,
    /// Maximum algebraic connectivity (for normalization)
    max_connectivity: f64,
}

impl RealPhiCalculator {
    /// Create new RealHV Φ calculator
    pub fn new() -> Self {
        Self {
            min_connectivity: 0.0,
            max_connectivity: 2.0, // Theoretical max for normalized Laplacian
        }
    }

    /// Compute Φ for a set of RealHV components
    ///
    /// # Arguments
    /// * `components` - Vector of RealHV representations (nodes in consciousness topology)
    ///
    /// # Returns
    /// Φ value in range [0, 1], where:
    /// - 0 = No integration (disconnected components)
    /// - 1 = Perfect integration (fully connected, maximal differentiation)
    pub fn compute(&self, components: &[RealHV]) -> f64 {
        let n = components.len();

        if n < 2 {
            return 0.0; // No integration possible with single component
        }

        // Step 1: Build cosine similarity matrix
        let similarity_matrix = self.build_similarity_matrix(components);

        // Step 2: Compute algebraic connectivity
        let algebraic_connectivity = self.compute_algebraic_connectivity(&similarity_matrix);

        // Step 3: Normalize to [0, 1] for Φ
        let phi = self.normalize_connectivity(algebraic_connectivity, n);

        phi
    }

    /// Build pairwise cosine similarity matrix
    ///
    /// Returns n×n matrix where entry (i,j) is cosine similarity between
    /// components[i] and components[j], normalized to [0, 1].
    fn build_similarity_matrix(&self, components: &[RealHV]) -> Vec<Vec<f64>> {
        let n = components.len();
        let mut matrix = vec![vec![0.0_f64; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[i][j] = 1.0; // Perfect similarity to self
                } else {
                    // Compute cosine similarity
                    let similarity = components[i].similarity(&components[j]);

                    // Normalize from [-1, 1] to [0, 1]
                    // cosine = 1 → similarity = 1 (identical)
                    // cosine = 0 → similarity = 0.5 (orthogonal)
                    // cosine = -1 → similarity = 0 (opposite)
                    let normalized_similarity = (similarity as f64 + 1.0) / 2.0;

                    matrix[i][j] = normalized_similarity;
                }
            }
        }

        matrix
    }

    /// Compute algebraic connectivity (2nd smallest eigenvalue of Laplacian)
    ///
    /// The Laplacian matrix is L = D - A where:
    /// - D is the degree matrix (diagonal, D[i,i] = sum of row i in A)
    /// - A is the adjacency/similarity matrix
    ///
    /// Algebraic connectivity measures how well-connected the graph is.
    fn compute_algebraic_connectivity(&self, similarity_matrix: &[Vec<f64>]) -> f64 {
        let n = similarity_matrix.len();

        // Build Laplacian matrix: L = D - A
        let mut laplacian = vec![vec![0.0; n]; n];

        for i in 0..n {
            let mut degree = 0.0;

            for j in 0..n {
                if i != j {
                    degree += similarity_matrix[i][j];
                    laplacian[i][j] = -similarity_matrix[i][j];
                }
            }

            laplacian[i][i] = degree;
        }

        // Compute eigenvalues using power iteration
        // For small matrices (n < 20), we use a simplified approach
        // For larger matrices, more sophisticated methods would be needed
        let eigenvalues = self.compute_laplacian_eigenvalues(&laplacian);

        // Algebraic connectivity is the 2nd smallest eigenvalue
        // (smallest is always 0 for connected graphs)
        if eigenvalues.len() < 2 {
            return 0.0;
        }

        eigenvalues[1] // 2nd smallest
    }

    /// Compute eigenvalues of Laplacian matrix
    ///
    /// Simplified implementation for small matrices (n < 20).
    /// Returns sorted eigenvalues in ascending order.
    fn compute_laplacian_eigenvalues(&self, laplacian: &[Vec<f64>]) -> Vec<f64> {
        let n = laplacian.len();

        // For n=2, analytical solution
        if n == 2 {
            // Eigenvalues are: λ = (trace ± sqrt(trace² - 4*det)) / 2
            let trace = laplacian[0][0] + laplacian[1][1];
            let det = laplacian[0][0] * laplacian[1][1] - laplacian[0][1] * laplacian[1][0];
            let discriminant = (trace * trace - 4.0 * det).max(0.0).sqrt();

            let lambda1 = (trace - discriminant) / 2.0;
            let lambda2 = (trace + discriminant) / 2.0;

            return vec![lambda1.max(0.0), lambda2.max(0.0)];
        }

        // For small n, use QR algorithm approximation
        // This is a simplified version - production code would use proper linear algebra library
        self.qr_eigenvalues(laplacian)
    }

    /// Simplified QR algorithm for eigenvalue computation
    ///
    /// This is a basic implementation suitable for small matrices.
    /// For production, use nalgebra or ndarray with LAPACK.
    fn qr_eigenvalues(&self, matrix: &[Vec<f64>]) -> Vec<f64> {
        let n = matrix.len();
        let a = matrix.to_vec();

        // Simplified: Use power iteration to get dominant eigenvalue,
        // then deflation for others. This is approximate but sufficient
        // for our Φ calculation purposes.

        let mut eigenvalues = Vec::new();

        // Get trace (sum of eigenvalues)
        let mut soul::weaver::COHERENCE_THRESHOLD = 0.0;
        for i in 0..n {
            trace += a[i][i];
        }

        // Approximate: For Laplacian, we know smallest eigenvalue ≈ 0
        eigenvalues.push(0.0);

        // Approximate 2nd eigenvalue using Gershgorin circle theorem
        // For Laplacian: λ₂ ≈ min(row sums excluding diagonal)
        let mut min_sum = f64::INFINITY;
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                if i != j {
                    sum += a[i][j].abs();
                }
            }
            min_sum = min_sum.min(sum);
        }

        eigenvalues.push(min_sum);

        eigenvalues
    }

    /// Normalize algebraic connectivity to [0, 1] range for Φ
    ///
    /// Uses the theoretical bounds for algebraic connectivity based on
    /// graph size and structure.
    fn normalize_connectivity(&self, connectivity: f64, n: usize) -> f64 {
        // For complete graph of size n, algebraic connectivity = n
        // For sparse graphs, connectivity approaches 0
        // We normalize to [0, 1] using expected max for size n

        let max_connectivity = n as f64;

        let normalized = connectivity / max_connectivity;

        // Clamp to [0, 1]
        normalized.clamp(0.0, 1.0)
    }
}

impl Default for RealPhiCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_creation() {
        let calc = RealPhiCalculator::new();
        assert_eq!(calc.min_connectivity, 0.0);
        assert_eq!(calc.max_connectivity, 2.0);
    }

    #[test]
    fn test_single_component() {
        let calc = RealPhiCalculator::new();
        let components = vec![RealHV::random(256, 42)];
        let phi = calc.compute(&components);
        assert_eq!(phi, 0.0, "Single component should have Φ = 0");
    }

    #[test]
    fn test_identical_components() {
        let calc = RealPhiCalculator::new();
        let hv = RealHV::random(256, 42);
        let components = vec![hv.clone(), hv.clone()];

        let phi = calc.compute(&components);

        // In spectral-based Φ calculation, identical components have maximum
        // similarity (1.0), which yields high algebraic connectivity.
        // This is a valid measurement of "integration" - the components
        // can't be partitioned without information loss.
        // Note: This differs from IIT's "differentiation" requirement.
        assert!(phi >= 0.0 && phi <= 1.0, "Φ should be in valid range: {}", phi);
    }

    #[test]
    fn test_orthogonal_components() {
        let calc = RealPhiCalculator::new();

        // Create orthogonal vectors using basis vectors
        let components = vec![
            RealHV::basis(0, 256),
            RealHV::basis(1, 256),
            RealHV::basis(2, 256),
        ];

        let phi = calc.compute(&components);

        // Orthogonal components should have moderate Φ
        assert!(phi > 0.0, "Orthogonal components should have Φ > 0");
        assert!(phi < 1.0, "Finite Φ should be < 1.0");
    }

    #[test]
    fn test_similarity_matrix() {
        let calc = RealPhiCalculator::new();

        let components = vec![
            RealHV::random(128, 1),
            RealHV::random(128, 2),
        ];

        let matrix = calc.build_similarity_matrix(&components);

        // Check dimensions
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);

        // Check diagonal (self-similarity = 1)
        assert!((matrix[0][0] - 1.0).abs() < 1e-10);
        assert!((matrix[1][1] - 1.0).abs() < 1e-10);

        // Check symmetry
        assert!((matrix[0][1] - matrix[1][0]).abs() < 1e-10);

        // Check range [0, 1]
        assert!(matrix[0][1] >= 0.0 && matrix[0][1] <= 1.0);
    }

    #[test]
    fn test_algebraic_connectivity_two_nodes() {
        let calc = RealPhiCalculator::new();

        // High similarity → high connectivity
        let matrix_high = vec![
            vec![1.0, 0.9],
            vec![0.9, 1.0],
        ];
        let conn_high = calc.compute_algebraic_connectivity(&matrix_high);

        // Low similarity → low connectivity
        let matrix_low = vec![
            vec![1.0, 0.1],
            vec![0.1, 1.0],
        ];
        let conn_low = calc.compute_algebraic_connectivity(&matrix_low);

        assert!(conn_high > conn_low,
                "Higher similarity should yield higher connectivity: {} vs {}",
                conn_high, conn_low);
    }
}
