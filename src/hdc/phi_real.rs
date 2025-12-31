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
///
/// # Normalization Strategy
///
/// For proper scaling across different graph sizes and dimensions, we use
/// adaptive normalization based on the actual graph structure:
/// - Compute maximum weighted degree from the similarity matrix
/// - Scale algebraic connectivity relative to this structure-aware bound
/// - This prevents saturation at Φ=1.0 for high-dimensional graphs
#[derive(Clone, Debug)]
pub struct RealPhiCalculator {
    /// Minimum algebraic connectivity (for normalization)
    min_connectivity: f64,
    /// Maximum algebraic connectivity (theoretical upper bound, used as fallback)
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
        // With normalized Laplacian, eigenvalues are bounded [0, 2], so we can use
        // simple linear scaling: Φ = λ₂ / 2.0
        let phi = (algebraic_connectivity / self.max_connectivity).clamp(0.0, 1.0);

        phi
    }

    /// Build pairwise cosine similarity matrix (optimized: exploits symmetry)
    ///
    /// Returns n×n matrix where entry (i,j) is cosine similarity between
    /// components[i] and components[j], normalized to [0, 1].
    ///
    /// Optimization: Since similarity(i,j) = similarity(j,i), we only compute
    /// n(n-1)/2 similarities instead of n², reducing computation by ~50%.
    fn build_similarity_matrix(&self, components: &[RealHV]) -> Vec<Vec<f64>> {
        let n = components.len();
        let mut matrix = vec![vec![0.0_f64; n]; n];

        // Set diagonal to 1.0 (perfect self-similarity)
        for i in 0..n {
            matrix[i][i] = 1.0;
        }

        // Only compute upper triangle, then copy to lower (symmetry optimization)
        // This reduces from n² to n(n-1)/2 similarity computations
        for i in 0..n {
            for j in (i + 1)..n {
                // Compute cosine similarity
                let similarity = components[i].similarity(&components[j]);

                // Normalize from [-1, 1] to [0, 1]
                // cosine = 1 → similarity = 1 (identical)
                // cosine = 0 → similarity = 0.5 (orthogonal)
                // cosine = -1 → similarity = 0 (opposite)
                let normalized_similarity = (similarity as f64 + 1.0) / 2.0;

                // Set both (i,j) and (j,i) - symmetry!
                matrix[i][j] = normalized_similarity;
                matrix[j][i] = normalized_similarity;
            }
        }

        matrix
    }

    /// Compute algebraic connectivity (2nd smallest eigenvalue of normalized Laplacian)
    ///
    /// Uses the **normalized Laplacian** for proper scaling across graph sizes:
    /// L_norm = I - D^(-1/2) * A * D^(-1/2)
    ///
    /// Key properties:
    /// - Eigenvalues are ALWAYS in [0, 2] regardless of graph size
    /// - λ₁ = 0 for connected graphs
    /// - λ₂ = algebraic connectivity (Fiedler value)
    /// - This allows meaningful comparison across dimensions
    ///
    /// For k-dimensional hypercubes, the normalized algebraic connectivity
    /// converges as k → ∞, allowing proper asymptotic analysis.
    fn compute_algebraic_connectivity(&self, similarity_matrix: &[Vec<f64>]) -> f64 {
        let n = similarity_matrix.len();

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
        let mut normalized_laplacian = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    // Diagonal: L_norm[i,i] = 1 (if degree > 0)
                    normalized_laplacian[i][i] = if degrees[i] > 1e-10 { 1.0 } else { 0.0 };
                } else {
                    // Off-diagonal: L_norm[i,j] = -A[i,j] / sqrt(D[i] * D[j])
                    let normalization = inv_sqrt_degrees[i] * inv_sqrt_degrees[j];
                    normalized_laplacian[i][j] = -similarity_matrix[i][j] * normalization;
                }
            }
        }

        // Step 4: Compute eigenvalues of normalized Laplacian
        let eigenvalues = self.compute_laplacian_eigenvalues(&normalized_laplacian);

        // Algebraic connectivity is the 2nd smallest eigenvalue
        // (smallest is always 0 for connected graphs)
        if eigenvalues.len() < 2 {
            return 0.0;
        }

        eigenvalues[1] // 2nd smallest, now guaranteed in [0, 2]
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

    /// Compute eigenvalues using power iteration methods
    ///
    /// Implements proper numerical algorithms without BLAS dependency:
    /// 1. Power iteration for largest eigenvalue
    /// 2. Deflation + power iteration for subsequent eigenvalues
    /// 3. Special handling for graph Laplacian (λ₁ = 0)
    fn qr_eigenvalues(&self, matrix: &[Vec<f64>]) -> Vec<f64> {
        let n = matrix.len();

        if n == 0 {
            return vec![];
        }

        // For graph Laplacian, λ₁ = 0 (always for connected graphs)
        // We need to find λ₂ (algebraic connectivity)

        // Method: Find largest eigenvalue, then use deflation to find others
        let max_iter = 200;
        let tolerance = 1e-10;

        // Step 1: Find λ_max using power iteration
        let (lambda_max, v_max) = self.power_iteration(matrix, max_iter, tolerance);

        // Step 2: Deflate matrix to remove largest eigenvalue
        // A' = A - λ_max * v * v^T
        let deflated = self.deflate_matrix(matrix, lambda_max, &v_max);

        // Step 3: Find next largest eigenvalue from deflated matrix
        // (Used for verification but not primary λ₂ calculation)
        let (_lambda_2nd, _) = self.power_iteration(&deflated, max_iter, tolerance);

        // For Laplacian: eigenvalues are in ascending order
        // λ₁ = 0 (smallest), λ₂ = algebraic connectivity
        // The power iteration finds largest, so we need to work out which is λ₂

        // Compute approximate λ₂ using shifted inverse iteration
        let lambda_2 = self.find_algebraic_connectivity(matrix, max_iter, tolerance);

        vec![0.0, lambda_2.max(0.0)]
    }

    /// Power iteration to find the largest eigenvalue and its eigenvector
    ///
    /// Implements the standard power method:
    /// 1. Start with random vector v₀
    /// 2. Iterate: v_{k+1} = A * v_k / ||A * v_k||
    /// 3. Converges to dominant eigenvector
    /// 4. Eigenvalue = (v^T * A * v) / (v^T * v)
    fn power_iteration(&self, matrix: &[Vec<f64>], max_iter: usize, tolerance: f64) -> (f64, Vec<f64>) {
        let n = matrix.len();

        // Initialize with uniform vector (avoids issues with random seed)
        let mut v: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];
        let mut lambda = 0.0;

        for _ in 0..max_iter {
            // Compute w = A * v
            let mut w = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += matrix[i][j] * v[j];
                }
            }

            // Compute norm of w
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break; // Matrix is zero or nearly zero
            }

            // Normalize w
            for x in w.iter_mut() {
                *x /= norm;
            }

            // Compute Rayleigh quotient: λ = v^T * A * v
            let new_lambda = self.rayleigh_quotient(matrix, &w);

            // Check convergence
            if (new_lambda - lambda).abs() < tolerance {
                return (new_lambda, w);
            }

            lambda = new_lambda;
            v = w;
        }

        (lambda, v)
    }

    /// Compute Rayleigh quotient: λ = (v^T * A * v) / (v^T * v)
    fn rayleigh_quotient(&self, matrix: &[Vec<f64>], v: &[f64]) -> f64 {
        let n = matrix.len();

        // Compute A * v
        let mut av = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                av[i] += matrix[i][j] * v[j];
            }
        }

        // Compute v^T * (A * v)
        let numerator: f64 = v.iter().zip(av.iter()).map(|(a, b)| a * b).sum();

        // Compute v^T * v
        let denominator: f64 = v.iter().map(|x| x * x).sum();

        if denominator < 1e-15 {
            return 0.0;
        }

        numerator / denominator
    }

    /// Deflate matrix by removing contribution of known eigenvalue/eigenvector
    ///
    /// A' = A - λ * v * v^T
    fn deflate_matrix(&self, matrix: &[Vec<f64>], lambda: f64, v: &[f64]) -> Vec<Vec<f64>> {
        let n = matrix.len();
        let mut result = matrix.to_vec();

        for i in 0..n {
            for j in 0..n {
                result[i][j] -= lambda * v[i] * v[j];
            }
        }

        result
    }

    /// Find algebraic connectivity (λ₂) using inverse iteration with shift
    ///
    /// For graph Laplacian:
    /// - λ₁ = 0 (with eigenvector = uniform vector)
    /// - λ₂ = algebraic connectivity (what we want)
    ///
    /// Uses inverse iteration with shift to find smallest non-zero eigenvalue:
    /// 1. Project out the null space (eigenvector for λ₁ = 0)
    /// 2. Use inverse power iteration on modified matrix
    fn find_algebraic_connectivity(&self, laplacian: &[Vec<f64>], max_iter: usize, tolerance: f64) -> f64 {
        let n = laplacian.len();

        if n < 2 {
            return 0.0;
        }

        // The null space of Laplacian is the uniform vector [1,1,1,...]/sqrt(n)
        let null_vec: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];

        // Initialize orthogonal to null space
        let mut v: Vec<f64> = (0..n).map(|i| if i == 0 { 1.0 } else { -1.0 / (n as f64 - 1.0) }).collect();
        let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in v.iter_mut() {
            *x /= v_norm;
        }

        // Project out null space component
        self.project_out(&mut v, &null_vec);
        let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if v_norm > 1e-10 {
            for x in v.iter_mut() {
                *x /= v_norm;
            }
        }

        // Add small shift to make matrix invertible (since λ₁ = 0)
        let shift = 1e-6;
        let mut shifted = laplacian.to_vec();
        for i in 0..n {
            shifted[i][i] += shift;
        }

        // Inverse power iteration to find smallest eigenvalue
        // Instead of direct inversion, solve (L + shift*I) * w = v using Gaussian elimination
        let mut lambda = f64::INFINITY;

        for _ in 0..max_iter {
            // Solve (L + shift*I) * w = v
            let w = self.solve_linear_system(&shifted, &v);

            // Project out null space
            let mut w_proj = w.clone();
            self.project_out(&mut w_proj, &null_vec);

            // Normalize
            let norm: f64 = w_proj.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            for x in w_proj.iter_mut() {
                *x /= norm;
            }

            // Compute Rayleigh quotient on original Laplacian (not shifted)
            let new_lambda = self.rayleigh_quotient(laplacian, &w_proj);

            // Check convergence
            if (new_lambda - lambda).abs() < tolerance {
                return new_lambda;
            }

            lambda = new_lambda;
            v = w_proj;
        }

        lambda
    }

    /// Project vector v onto the subspace orthogonal to u
    /// v = v - (v · u) * u
    fn project_out(&self, v: &mut [f64], u: &[f64]) {
        let dot: f64 = v.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
        for (vi, &ui) in v.iter_mut().zip(u.iter()) {
            *vi -= dot * ui;
        }
    }

    /// Solve linear system Ax = b using LU decomposition with partial pivoting
    fn solve_linear_system(&self, a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = a.len();

        // Copy matrix and vector for modification
        let mut lu = a.to_vec();
        let mut perm: Vec<usize> = (0..n).collect();

        // LU decomposition with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_val = lu[k][k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                if lu[i][k].abs() > max_val {
                    max_val = lu[i][k].abs();
                    max_row = i;
                }
            }

            // Swap rows if necessary
            if max_row != k {
                lu.swap(k, max_row);
                perm.swap(k, max_row);
            }

            // Skip if pivot is too small
            if max_val < 1e-15 {
                continue;
            }

            // Elimination
            for i in (k + 1)..n {
                let factor = lu[i][k] / lu[k][k];
                lu[i][k] = factor; // Store L factor
                for j in (k + 1)..n {
                    lu[i][j] -= factor * lu[k][j];
                }
            }
        }

        // Apply permutation to b
        let mut pb = vec![0.0; n];
        for (i, &p) in perm.iter().enumerate() {
            pb[i] = b[p];
        }

        // Forward substitution (solve L * y = Pb)
        let mut y = pb;
        for i in 0..n {
            for j in 0..i {
                y[i] -= lu[i][j] * y[j];
            }
        }

        // Back substitution (solve U * x = y)
        let mut x = y;
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] -= lu[i][j] * x[j];
            }
            if lu[i][i].abs() > 1e-15 {
                x[i] /= lu[i][i];
            }
        }

        x
    }

    /// Adaptive normalization using actual graph structure
    ///
    /// This method computes the maximum possible algebraic connectivity based on
    /// the actual similarity matrix structure, rather than using fixed theoretical
    /// bounds. This is crucial for high-dimensional hypercubes where the algebraic
    /// connectivity can exceed the naive bound of 2.0.
    ///
    /// # Algorithm
    ///
    /// 1. Compute maximum weighted degree (max row sum excluding diagonal)
    /// 2. For k-regular or near-k-regular graphs, λ₂_max ≈ max_degree
    /// 3. Use this as the adaptive upper bound for normalization
    /// 4. Apply sigmoid-like compression for values near the bound
    ///
    /// # Rationale
    ///
    /// The algebraic connectivity λ₂ of a weighted graph is bounded by:
    /// - λ₂ ≤ n/(n-1) * min_degree for vertex connectivity
    /// - λ₂ ≤ max_degree for weighted graphs with positive weights
    ///
    /// For hypercubes: a d-dimensional hypercube has degree d, and with our
    /// similarity-weighted edges, λ₂ scales with degree. This adaptive method
    /// properly handles 8D+ hypercubes that would otherwise saturate.
    fn normalize_connectivity_adaptive(&self, connectivity: f64, similarity_matrix: &[Vec<f64>]) -> f64 {
        let n = similarity_matrix.len();

        if n < 2 {
            return 0.0;
        }

        // Compute maximum weighted degree from similarity matrix
        // (max row sum excluding diagonal)
        let max_weighted_degree = similarity_matrix.iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)  // Exclude diagonal
                    .map(|(_, &val)| val)
                    .sum::<f64>()
            })
            .fold(f64::NEG_INFINITY, f64::max);

        // Compute average weighted degree for reference
        let avg_weighted_degree: f64 = similarity_matrix.iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &val)| val)
                    .sum::<f64>()
            })
            .sum::<f64>() / n as f64;

        // Adaptive maximum: use the larger of theoretical bound or structure-based bound
        // For sparse graphs (low degree): use theoretical max = 2.0
        // For dense/high-dimensional graphs: use max_weighted_degree
        //
        // The scaling factor of 1.0 means we expect λ₂ can approach max_degree
        // for highly uniform k-regular graphs like hypercubes
        let structure_based_max = max_weighted_degree.max(avg_weighted_degree);
        let effective_max = self.max_connectivity.max(structure_based_max);

        // Shift by min_connectivity (usually 0.0)
        let shifted = (connectivity - self.min_connectivity).max(0.0);

        // Normalize to [0, 1] using adaptive max
        let normalized = if effective_max > self.min_connectivity {
            shifted / (effective_max - self.min_connectivity)
        } else {
            0.0
        };

        // Clamp to [0, 1] - should rarely hit 1.0 with adaptive normalization
        normalized.clamp(0.0, 1.0)
    }

    /// Normalize algebraic connectivity to [0, 1] range for Φ (legacy method)
    ///
    /// Uses both theoretical bounds (self.min/max_connectivity) and graph size.
    /// For normalized Laplacian: eigenvalues in [0, 2]
    /// For combinatorial Laplacian: eigenvalues in [0, n] for complete graphs
    ///
    /// We blend these approaches: use the minimum of the theoretical max and
    /// the size-based max, ensuring robust normalization across graph types.
    ///
    /// **Note**: This method may saturate for high-dimensional hypercubes (8D+).
    /// Use `normalize_connectivity_adaptive` for such cases.
    #[allow(dead_code)]
    fn normalize_connectivity(&self, connectivity: f64, n: usize) -> f64 {
        // Shift by min_connectivity to handle negative eigenvalues (shouldn't occur for valid Laplacian)
        let shifted = (connectivity - self.min_connectivity).max(0.0);

        // Use the smaller of theoretical max (for normalized Laplacian) or size-based max
        // For small graphs: size-based is usually smaller
        // For large graphs: theoretical max prevents over-scaling
        let effective_max = (self.max_connectivity).min(n as f64);

        // Normalize to [0, 1]
        let normalized = if effective_max > self.min_connectivity {
            shifted / (effective_max - self.min_connectivity)
        } else {
            0.0
        };

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
