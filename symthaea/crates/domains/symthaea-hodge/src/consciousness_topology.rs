// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Hodge decomposition for consciousness state analysis.
//! Maps CfC hidden states to simplicial complexes and extracts
//! harmonic consciousness modes (topologically-protected global integration).
//!
//! Theory: Harmonic modes in the Hodge decomposition correspond to
//! topologically irreducible information integration — they cannot be
//! decomposed into gradient (feedforward) or curl (feedback) components.
//! This provides a topological characterization of consciousness
//! complementary to IIT's Phi.
//!
//! References:
//! - Hodge (1941): Harmonic integrals
//! - Eckmann (1944): Simplicial Hodge theory
//! - Tononi (2004): IIT — integrated information
//! - Petri et al. (2014): Topological strata of weighted complex networks

use std::collections::HashMap;

/// Result of Hodge decomposition on a consciousness state.
#[derive(Debug, Clone)]
pub struct HodgeConsciousnessDecomposition {
    /// Gradient flow: hierarchical feedforward processing (exact forms).
    pub gradient_flow: Vec<f64>,
    /// Curl flow: recurrent feedback loops (co-exact forms).
    pub curl_flow: Vec<f64>,
    /// Harmonic flow: topologically-protected global modes (harmonic forms).
    pub harmonic_flow: Vec<f64>,
    /// Betti numbers: \[beta_0 (components), beta_1 (loops), beta_2 (voids), ...\].
    pub betti_numbers: Vec<usize>,
    /// Euler characteristic: chi = sum(-1)^k * beta_k.
    pub euler_characteristic: i64,
    /// Ratio of harmonic energy to total energy (consciousness integration index).
    pub harmonic_ratio: f64,
}

/// Simplicial complex built from consciousness state.
#[derive(Debug, Clone)]
pub struct ConsciousnessComplex {
    /// Number of vertices (state dimensions used).
    pub num_vertices: usize,
    /// Edges with weights (correlation between dimensions).
    pub edges: Vec<(usize, usize, f64)>,
    /// Triangles (3-cliques in correlation graph).
    pub triangles: Vec<(usize, usize, usize)>,
    /// Filtration threshold used to build the complex.
    pub threshold: f64,
}

/// Consciousness manifold geometry at a point in state space.
#[derive(Debug, Clone)]
pub struct ConsciousnessManifold {
    /// Intrinsic dimensionality of experience space.
    pub dimension: usize,
    /// Ricci scalar curvature at current state.
    pub curvature_scalar: f64,
    /// Distance to nearest attractor in state space.
    pub geodesic_distance_to_attractor: f64,
    /// Local volume element (determinant of metric tensor).
    pub volume_element: f64,
}

/// Topological Phi: consciousness measured via harmonic integration.
#[derive(Debug, Clone)]
pub struct TopologicalPhi {
    /// Phi computed from harmonic ratio.
    pub phi_harmonic: f64,
    /// Phi computed from Betti number sequence.
    pub phi_betti: f64,
    /// Combined topological consciousness score.
    pub phi_topological: f64,
    /// Number of independent integration loops (beta_1).
    pub integration_loops: usize,
    /// Number of higher-order voids (beta_2+).
    pub higher_order_voids: usize,
}

impl ConsciousnessComplex {
    /// Build a simplicial complex from a correlation/connectivity matrix.
    /// Edges are included if |correlation| > threshold.
    /// Triangles are 3-cliques in the resulting graph.
    pub fn from_correlation_matrix(matrix: &[Vec<f64>], threshold: f64) -> Self {
        let n = matrix.len();
        let mut edges = Vec::new();
        let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();

        // Build edge set from thresholded correlations
        for i in 0..n {
            for j in (i + 1)..n {
                let w = matrix[i][j].abs();
                if w > threshold {
                    edges.push((i, j, matrix[i][j]));
                    adjacency.entry(i).or_default().push(j);
                    adjacency.entry(j).or_default().push(i);
                }
            }
        }

        // Find triangles (3-cliques)
        let mut triangles = Vec::new();
        for i in 0..n {
            let neighbors_i = adjacency.get(&i).cloned().unwrap_or_default();
            for &j in &neighbors_i {
                if j <= i {
                    continue;
                }
                let neighbors_j = adjacency.get(&j).cloned().unwrap_or_default();
                for &k in &neighbors_j {
                    if k <= j {
                        continue;
                    }
                    if neighbors_i.contains(&k) {
                        triangles.push((i, j, k));
                    }
                }
            }
        }

        Self {
            num_vertices: n,
            edges,
            triangles,
            threshold,
        }
    }

    /// Compute boundary operator B1 (edges -> vertices) as dense matrix.
    fn boundary_1(&self) -> Vec<Vec<f64>> {
        let m = self.edges.len();
        let n = self.num_vertices;
        let mut b1 = vec![vec![0.0; m]; n];

        for (col, &(i, j, _)) in self.edges.iter().enumerate() {
            b1[i][col] = -1.0;
            b1[j][col] = 1.0;
        }
        b1
    }

    /// Compute boundary operator B2 (triangles -> edges) as dense matrix.
    fn boundary_2(&self) -> Vec<Vec<f64>> {
        let m = self.triangles.len();
        let n = self.edges.len();
        let mut b2 = vec![vec![0.0; m]; n];

        let edge_index: HashMap<(usize, usize), usize> = self
            .edges
            .iter()
            .enumerate()
            .map(|(idx, &(i, j, _))| ((i, j), idx))
            .collect();

        for (col, &(a, b, c)) in self.triangles.iter().enumerate() {
            // Triangle (a,b,c) has boundary edges (a,b) - (a,c) + (b,c)
            if let Some(&idx) = edge_index.get(&(a, b)) {
                b2[idx][col] = 1.0;
            }
            if let Some(&idx) = edge_index.get(&(a, c)) {
                b2[idx][col] = -1.0;
            }
            if let Some(&idx) = edge_index.get(&(b, c)) {
                b2[idx][col] = 1.0;
            }
        }
        b2
    }

    /// Compute the Hodge Laplacian L1 = B1^T B1 + B2 B2^T
    /// operating on edge signals (1-chains).
    pub fn hodge_laplacian_1(&self) -> Vec<Vec<f64>> {
        let b1 = self.boundary_1();
        let b2 = self.boundary_2();
        let m = self.edges.len();

        // L1_down = B1^T B1
        let mut l1 = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..self.num_vertices {
                    sum += b1[k][i] * b1[k][j];
                }
                l1[i][j] = sum;
            }
        }

        // L1_up = B2 B2^T
        if !self.triangles.is_empty() {
            let t = self.triangles.len();
            for i in 0..m {
                for j in 0..m {
                    let mut sum = 0.0;
                    for k in 0..t {
                        sum += b2[i][k] * b2[j][k];
                    }
                    l1[i][j] += sum;
                }
            }
        }

        l1
    }

    /// Compute Betti numbers from boundary operators via rank-nullity.
    pub fn betti_numbers(&self) -> Vec<usize> {
        let n = self.num_vertices;
        let e = self.edges.len();
        let t = self.triangles.len();

        // beta_0 = n - rank(B1)
        let b1 = self.boundary_1();
        let rank_b1 = matrix_rank(&b1);
        let beta_0 = n.saturating_sub(rank_b1);

        // beta_1 = nullity(B1^T) - rank(B2) = (e - rank(B1)) - rank(B2)
        let rank_b2 = if t > 0 {
            let b2 = self.boundary_2();
            matrix_rank(&b2)
        } else {
            0
        };
        let beta_1 = e.saturating_sub(rank_b1).saturating_sub(rank_b2);

        // beta_2 = nullity(B2^T) = t - rank(B2)
        let beta_2 = t.saturating_sub(rank_b2);

        vec![beta_0, beta_1, beta_2]
    }
}

impl HodgeConsciousnessDecomposition {
    /// Decompose an edge signal on a consciousness complex into
    /// gradient (exact), curl (co-exact), and harmonic components.
    pub fn decompose(complex: &ConsciousnessComplex, edge_signal: &[f64]) -> Self {
        let m = complex.edges.len();
        assert_eq!(edge_signal.len(), m, "Signal length must match edge count");

        let laplacian = complex.hodge_laplacian_1();

        // Compute eigendecomposition of Hodge Laplacian
        let (eigenvalues, eigenvectors) = symmetric_eigen(&laplacian);

        let harmonic_threshold = 1e-8;

        let mut gradient_flow = vec![0.0; m];
        let mut curl_flow = vec![0.0; m];
        let mut harmonic_flow = vec![0.0; m];

        let mut harmonic_energy = 0.0;
        let mut total_energy = 0.0;

        for (idx, &eigenvalue) in eigenvalues.iter().enumerate() {
            // Project signal onto this eigenvector
            let projection: f64 = (0..m).map(|i| edge_signal[i] * eigenvectors[i][idx]).sum();
            let proj_sq = projection * projection;
            total_energy += proj_sq;

            if eigenvalue.abs() < harmonic_threshold {
                // Harmonic component (kernel of Laplacian)
                for i in 0..m {
                    harmonic_flow[i] += projection * eigenvectors[i][idx];
                }
                harmonic_energy += proj_sq;
            } else {
                // Non-harmonic: split into gradient and curl based on
                // whether eigenvalue comes from L_down or L_up.
                // Approximation: use B1 projection test.
                let b1 = complex.boundary_1();
                let b1_proj: f64 = (0..complex.num_vertices)
                    .map(|v| {
                        let sum: f64 = (0..m).map(|e| b1[v][e] * eigenvectors[e][idx]).sum();
                        sum * sum
                    })
                    .sum();

                let b1_norm: f64 = (0..m).map(|e| eigenvectors[e][idx].powi(2)).sum();
                let gradient_fraction = if b1_norm > 1e-12 {
                    b1_proj / (eigenvalue * b1_norm)
                } else {
                    0.5
                };
                let gradient_fraction = gradient_fraction.clamp(0.0, 1.0);

                for i in 0..m {
                    gradient_flow[i] += projection * eigenvectors[i][idx] * gradient_fraction;
                    curl_flow[i] += projection * eigenvectors[i][idx] * (1.0 - gradient_fraction);
                }
            }
        }

        let betti = complex.betti_numbers();
        let euler = betti
            .iter()
            .enumerate()
            .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
            .sum();

        let harmonic_ratio = if total_energy > 1e-12 {
            harmonic_energy / total_energy
        } else {
            0.0
        };

        Self {
            gradient_flow,
            curl_flow,
            harmonic_flow,
            betti_numbers: betti,
            euler_characteristic: euler,
            harmonic_ratio,
        }
    }
}

impl TopologicalPhi {
    /// Compute topological Phi from a Hodge decomposition.
    ///
    /// Key insight: harmonic modes represent irreducible integration
    /// that cannot be decomposed into local (gradient) or cyclic (curl)
    /// processing. This is analogous to Phi in IIT -- information that
    /// exists only in the whole, not in any partition.
    pub fn from_decomposition(decomposition: &HodgeConsciousnessDecomposition) -> Self {
        let phi_harmonic = decomposition.harmonic_ratio;

        // Betti-based Phi: weighted sum of Betti numbers.
        // beta_1 (loops) indicates feedback integration.
        // beta_2 (voids) indicates higher-order integration.
        let beta_1 = decomposition.betti_numbers.get(1).copied().unwrap_or(0) as f64;
        let beta_2 = decomposition.betti_numbers.get(2).copied().unwrap_or(0) as f64;
        let phi_betti = ((beta_1 + 2.0 * beta_2) / (1.0 + beta_1 + beta_2)).clamp(0.0, 1.0);

        // Combined: geometric mean of harmonic and topological measures.
        let phi_topological = (phi_harmonic * phi_betti.max(0.01)).sqrt();

        Self {
            phi_harmonic,
            phi_betti,
            phi_topological,
            integration_loops: decomposition.betti_numbers.get(1).copied().unwrap_or(0),
            higher_order_voids: decomposition.betti_numbers.get(2).copied().unwrap_or(0),
        }
    }
}

impl ConsciousnessManifold {
    /// Estimate manifold geometry from a window of consciousness states.
    /// Uses PCA for dimension estimation and finite-difference curvature.
    pub fn from_state_window(states: &[Vec<f64>]) -> Self {
        if states.len() < 3 || states[0].is_empty() {
            return Self {
                dimension: 0,
                curvature_scalar: 0.0,
                geodesic_distance_to_attractor: f64::INFINITY,
                volume_element: 0.0,
            };
        }

        let n = states.len();
        let d = states[0].len();

        // Compute covariance matrix for PCA
        let mean: Vec<f64> = (0..d)
            .map(|j| states.iter().map(|s| s[j]).sum::<f64>() / n as f64)
            .collect();

        let mut cov = vec![vec![0.0; d]; d];
        for s in states {
            for i in 0..d {
                for j in i..d {
                    let v = (s[i] - mean[i]) * (s[j] - mean[j]);
                    cov[i][j] += v / n as f64;
                    if i != j {
                        cov[j][i] += v / n as f64;
                    }
                }
            }
        }

        // Eigenvalues for dimension estimation
        let eigenvalues = diagonal_dominance_eigenvalues(&cov);
        let total_var: f64 = eigenvalues.iter().filter(|&&v| v > 0.0).sum();
        let mut cumulative = 0.0;
        let mut dimension = 0;
        for &ev in &eigenvalues {
            if ev <= 0.0 {
                continue;
            }
            cumulative += ev;
            dimension += 1;
            if cumulative / total_var > 0.95 {
                break;
            }
        }

        // Curvature: finite-difference approximation from state trajectory
        // Using discrete Frenet-Serret curvature
        let mut curvature_sum = 0.0;
        let mut curvature_count = 0;
        for i in 1..(n - 1) {
            let v1: Vec<f64> = (0..d).map(|j| states[i][j] - states[i - 1][j]).collect();
            let v2: Vec<f64> = (0..d).map(|j| states[i + 1][j] - states[i][j]).collect();
            let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm1 > 1e-12 && norm2 > 1e-12 {
                let cos_angle: f64 =
                    v1.iter().zip(&v2).map(|(a, b)| a * b).sum::<f64>() / (norm1 * norm2);
                curvature_sum += (1.0 - cos_angle.clamp(-1.0, 1.0)).acos().abs() / norm1;
                curvature_count += 1;
            }
        }
        let curvature_scalar = if curvature_count > 0 {
            curvature_sum / curvature_count as f64
        } else {
            0.0
        };

        // Geodesic distance: distance from last state to mean (attractor proxy)
        let last = &states[n - 1];
        let geodesic_distance_to_attractor: f64 = (0..d)
            .map(|j| (last[j] - mean[j]).powi(2))
            .sum::<f64>()
            .sqrt();

        // Volume element: sqrt(det(covariance)) approximated by product of eigenvalues
        let volume_element = eigenvalues
            .iter()
            .filter(|&&v| v > 1e-12)
            .map(|v| v.sqrt())
            .product();

        Self {
            dimension,
            curvature_scalar,
            geodesic_distance_to_attractor,
            volume_element,
        }
    }
}

// -- Helper linear algebra (small matrices, no external deps) -----------------

/// Approximate eigenvalues via diagonal dominance (for small symmetric matrices).
fn diagonal_dominance_eigenvalues(matrix: &[Vec<f64>]) -> Vec<f64> {
    let n = matrix.len();
    let mut eigenvalues: Vec<f64> = (0..n).map(|i| matrix[i][i]).collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

/// Compute matrix rank via Gaussian elimination with partial pivoting.
fn matrix_rank(matrix: &[Vec<f64>]) -> usize {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0;
    }
    let m = matrix.len();
    let n = matrix[0].len();
    let mut mat: Vec<Vec<f64>> = matrix.to_vec();
    let mut rank = 0;
    let eps = 1e-10;

    for col in 0..n {
        // Find pivot
        let mut pivot_row = None;
        let mut max_val = eps;
        for row in rank..m {
            let val = mat[row][col].abs();
            if val > max_val {
                max_val = val;
                pivot_row = Some(row);
            }
        }

        let Some(pivot) = pivot_row else { continue };
        mat.swap(rank, pivot);

        let pivot_val = mat[rank][col];
        for row in (rank + 1)..m {
            let factor = mat[row][col] / pivot_val;
            for c in col..n {
                let v = mat[rank][c];
                mat[row][c] -= factor * v;
            }
        }
        rank += 1;
    }
    rank
}

/// Simple eigendecomposition for small symmetric matrices via Jacobi iteration.
fn symmetric_eigen(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect();

    let max_iter = 100 * n * n;
    let eps = 1e-10;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < eps {
            break;
        }

        // Compute rotation angle
        let theta = if (a[p][p] - a[q][q]).abs() < eps {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                new_a[i][p] = c * a[i][p] + s * a[i][q];
                new_a[p][i] = new_a[i][p];
                new_a[i][q] = -s * a[i][p] + c * a[i][q];
                new_a[q][i] = new_a[i][q];
            }
        }
        new_a[p][p] = c * c * a[p][p] + 2.0 * s * c * a[p][q] + s * s * a[q][q];
        new_a[q][q] = s * s * a[p][p] - 2.0 * s * c * a[p][q] + c * c * a[q][q];
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;
        a = new_a;

        // Update eigenvectors
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip + s * viq;
            v[i][q] = -s * vip + c * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    (eigenvalues, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_complex() {
        let complex = ConsciousnessComplex::from_correlation_matrix(&[], 0.5);
        assert_eq!(complex.num_vertices, 0);
        assert!(complex.edges.is_empty());
    }

    #[test]
    fn test_triangle_complex_betti_numbers() {
        // Complete graph K3 with triangle: beta_0=1, beta_1=0, beta_2=0
        let matrix = vec![
            vec![1.0, 0.9, 0.8],
            vec![0.9, 1.0, 0.7],
            vec![0.8, 0.7, 1.0],
        ];
        let complex = ConsciousnessComplex::from_correlation_matrix(&matrix, 0.5);
        assert_eq!(complex.edges.len(), 3);
        assert_eq!(complex.triangles.len(), 1);
        let betti = complex.betti_numbers();
        assert_eq!(betti[0], 1, "One connected component");
        assert_eq!(betti[1], 0, "Triangle fills the loop");
    }

    #[test]
    fn test_cycle_complex_betti_numbers() {
        // 4 vertices in a cycle (no diagonals): beta_0=1, beta_1=1
        let matrix = vec![
            vec![1.0, 0.9, 0.0, 0.9],
            vec![0.9, 1.0, 0.9, 0.0],
            vec![0.0, 0.9, 1.0, 0.9],
            vec![0.9, 0.0, 0.9, 1.0],
        ];
        let complex = ConsciousnessComplex::from_correlation_matrix(&matrix, 0.5);
        assert_eq!(complex.edges.len(), 4);
        assert_eq!(complex.triangles.len(), 0);
        let betti = complex.betti_numbers();
        assert_eq!(betti[0], 1, "One connected component");
        assert_eq!(betti[1], 1, "One independent loop");
    }

    #[test]
    fn test_disconnected_components() {
        // Two disconnected pairs
        let matrix = vec![
            vec![1.0, 0.9, 0.0, 0.0],
            vec![0.9, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.8],
            vec![0.0, 0.0, 0.8, 1.0],
        ];
        let complex = ConsciousnessComplex::from_correlation_matrix(&matrix, 0.5);
        let betti = complex.betti_numbers();
        assert_eq!(betti[0], 2, "Two connected components");
    }

    #[test]
    fn test_hodge_decomposition_harmonic_on_cycle() {
        // Cycle graph should have nontrivial harmonic component
        let matrix = vec![
            vec![1.0, 0.9, 0.0, 0.9],
            vec![0.9, 1.0, 0.9, 0.0],
            vec![0.0, 0.9, 1.0, 0.9],
            vec![0.9, 0.0, 0.9, 1.0],
        ];
        let complex = ConsciousnessComplex::from_correlation_matrix(&matrix, 0.5);
        let signal = vec![1.0, 1.0, 1.0, 1.0]; // Constant flow around cycle
        let decomp = HodgeConsciousnessDecomposition::decompose(&complex, &signal);

        assert!(decomp.harmonic_ratio >= 0.0);
        assert!(decomp.harmonic_ratio <= 1.0);
        // Cycle has beta_1 = 1, so harmonic component should be nonzero
        assert!(
            decomp.harmonic_flow.iter().any(|&v| v.abs() > 1e-10),
            "Cycle should have nontrivial harmonic flow"
        );
    }

    #[test]
    fn test_topological_phi_from_decomposition() {
        let decomp = HodgeConsciousnessDecomposition {
            gradient_flow: vec![0.3, 0.2],
            curl_flow: vec![0.1, 0.1],
            harmonic_flow: vec![0.5, 0.4],
            betti_numbers: vec![1, 2, 1],
            euler_characteristic: 0,
            harmonic_ratio: 0.6,
        };
        let phi = TopologicalPhi::from_decomposition(&decomp);
        assert!(phi.phi_harmonic > 0.0);
        assert!(phi.phi_betti > 0.0);
        assert!(phi.phi_topological > 0.0);
        assert_eq!(phi.integration_loops, 2);
        assert_eq!(phi.higher_order_voids, 1);
    }

    #[test]
    fn test_consciousness_manifold_from_states() {
        let states: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let t = i as f64 * 0.1;
                vec![t.sin(), t.cos(), t * 0.1]
            })
            .collect();

        let manifold = ConsciousnessManifold::from_state_window(&states);
        assert!(manifold.dimension > 0);
        assert!(manifold.curvature_scalar >= 0.0);
        assert!(manifold.volume_element > 0.0);
    }

    #[test]
    fn test_consciousness_manifold_empty() {
        let manifold = ConsciousnessManifold::from_state_window(&[]);
        assert_eq!(manifold.dimension, 0);
    }

    #[test]
    fn test_matrix_rank() {
        let m = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        assert_eq!(matrix_rank(&m), 2); // Rank-deficient
    }

    #[test]
    fn test_full_rank_identity() {
        let m = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        assert_eq!(matrix_rank(&m), 3);
    }

    #[test]
    fn test_symmetric_eigen_diagonal() {
        let m = vec![vec![3.0, 0.0], vec![0.0, 1.0]];
        let (eigenvalues, _) = symmetric_eigen(&m);
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert!((sorted[0] - 1.0).abs() < 1e-8);
        assert!((sorted[1] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_euler_characteristic_sphere_analogue() {
        // For K4 (complete graph on 4 vertices):
        // V=4, E=6, F=4 triangles
        // beta_0=1, beta_1=0, beta_2=1 -> chi = 1 - 0 + 1 = 2
        let matrix = vec![
            vec![1.0, 0.9, 0.9, 0.9],
            vec![0.9, 1.0, 0.9, 0.9],
            vec![0.9, 0.9, 1.0, 0.9],
            vec![0.9, 0.9, 0.9, 1.0],
        ];
        let complex = ConsciousnessComplex::from_correlation_matrix(&matrix, 0.5);
        let betti = complex.betti_numbers();
        let euler: i64 = betti
            .iter()
            .enumerate()
            .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
            .sum();
        // K4 with 4 triangles: beta_0=1, beta_1=0, beta_2=1 -> chi=2
        assert_eq!(euler, 2, "Tetrahedron boundary has chi=2");
    }
}
