#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![allow(clippy::needless_range_loop, clippy::new_without_default)]

//! Hodge Laplacian for Simplicial Complexes
//!
//! # Mathematical Foundation
//!
//! The Hodge Laplacian generalizes the graph Laplacian to higher-dimensional
//! simplicial structures, enabling spectral analysis of signals defined on
//! edges, triangles, and higher simplices -- not just vertices.
//!
//! ## Why This Matters for Consciousness
//!
//! Pairwise connectivity (edges) captures only dyadic neural interactions.
//! But conscious experience involves **higher-order interactions** among neural
//! populations: three regions co-activating simultaneously is qualitatively
//! different from three pairwise activations. Simplicial topology captures
//! exactly these higher-order relationships.
//!
//! The Hodge decomposition reveals three distinct types of information flow:
//! - **Gradient flow** (exact component): hierarchical, feed-forward processing
//! - **Curl flow** (coexact component): recurrent, feedback processing
//! - **Harmonic flow** (kernel component): global, topologically-protected modes
//!
//! The harmonic component is especially interesting for consciousness research:
//! these are signals that cannot be reduced to local interactions and correspond
//! to the topological "holes" (Betti numbers) in the neural interaction structure.
//!
//! ## Key Equations
//!
//! **Boundary operator** (partial_k):
//! ```text
//! partial_k([v_0, ..., v_k]) = sum_{i=0}^{k} (-1)^i [v_0, ..., v_hat_i, ..., v_k]
//! ```
//!
//! **Hodge k-Laplacian**:
//! ```text
//! L_k = B_k^T B_k + B_{k+1} B_{k+1}^T
//! ```
//!
//! **Hodge decomposition** of any k-signal omega:
//! ```text
//! omega = d alpha + d* beta + h
//! ```
//! where h is harmonic (in ker L_k), and dim(ker L_k) = beta_k (the k-th Betti number).
//!
//! ## References
//!
//! - Lim, L.-H. (2020). "Hodge Laplacians on graphs"
//! - Barbarossa & Sardellitti (2020). "Topological Signal Processing over Simplicial Complexes"
//! - Schaub et al. (2021). "Signal processing on higher-order networks"
//! - Reimann et al. (2017). "Cliques of Neurons Bound into Cavities Provide a
//!   Missing Link between Structure and Function" (Blue Brain Project)

use std::collections::HashMap;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// A simplicial complex: a collection of simplices closed under taking faces.
///
/// Simplices are organized by dimension:
/// - `simplices[0]` = 0-simplices (vertices)
/// - `simplices[1]` = 1-simplices (edges)
/// - `simplices[2]` = 2-simplices (triangles)
/// - `simplices[k]` = k-simplices
///
/// Each simplex is a sorted list of vertex indices.
#[derive(Debug, Clone)]
pub struct SimplicialComplex {
    /// Set of vertex indices present in the complex
    pub vertices: Vec<usize>,
    /// Simplices organized by dimension: `simplices[k]` contains all k-simplices.
    /// Each k-simplex is represented as a sorted `Vec<usize>` of (k+1) vertex indices.
    pub simplices: Vec<Vec<Vec<usize>>>,
    /// Maximum simplex dimension in the complex
    pub max_dim: usize,
}

impl SimplicialComplex {
    /// Create an empty simplicial complex.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            simplices: vec![Vec::new()], // At least dimension 0
            max_dim: 0,
        }
    }

    /// Add a simplex (given as a list of vertex indices) and all its faces.
    ///
    /// The closure property is enforced automatically: every face of the added
    /// simplex is also inserted into the complex.
    pub fn add_simplex(&mut self, mut verts: Vec<usize>) {
        verts.sort();
        verts.dedup();
        if verts.is_empty() {
            return;
        }

        let dim = verts.len() - 1;

        // Ensure we have enough levels
        while self.simplices.len() <= dim {
            self.simplices.push(Vec::new());
        }
        if dim > self.max_dim {
            self.max_dim = dim;
        }

        // Add all vertices
        for &v in &verts {
            if !self.vertices.contains(&v) {
                self.vertices.push(v);
                self.vertices.sort();
            }
            let singleton = vec![v];
            if !self.simplices[0].contains(&singleton) {
                self.simplices[0].push(singleton);
                self.simplices[0].sort();
            }
        }

        // Add the simplex itself (if not already present)
        if !self.simplices[dim].contains(&verts) {
            self.simplices[dim].push(verts.clone());
            self.simplices[dim].sort();
        }

        // Add all proper faces (recursively ensures closure)
        if verts.len() > 1 {
            for i in 0..verts.len() {
                let mut face = verts.clone();
                face.remove(i);
                self.add_simplex(face);
            }
        }
    }

    /// Build a simplicial complex from a graph adjacency matrix.
    ///
    /// Creates vertices and edges from the adjacency matrix, then adds a
    /// 2-simplex (triangle) for every 3-clique found in the graph.
    pub fn from_graph(adjacency: &[Vec<bool>]) -> Self {
        let n = adjacency.len();
        let mut complex = Self::new();

        // Add vertices
        for i in 0..n {
            complex.add_simplex(vec![i]);
        }

        // Add edges
        for i in 0..n {
            for j in (i + 1)..n {
                if adjacency[i][j] {
                    complex.add_simplex(vec![i, j]);
                }
            }
        }

        // Add triangles for 3-cliques
        for i in 0..n {
            for j in (i + 1)..n {
                if !adjacency[i][j] {
                    continue;
                }
                for k in (j + 1)..n {
                    if adjacency[i][k] && adjacency[j][k] {
                        complex.add_simplex(vec![i, j, k]);
                    }
                }
            }
        }

        complex
    }

    /// Number of k-simplices in the complex.
    pub fn count(&self, k: usize) -> usize {
        if k < self.simplices.len() {
            self.simplices[k].len()
        } else {
            0
        }
    }
}

// ============================================================================
// HODGE LAPLACIAN
// ============================================================================

/// The Hodge Laplacian computed from a simplicial complex.
///
/// Stores the boundary matrices B_k and the Hodge Laplacians L_k for each
/// dimension. The Laplacians enable spectral analysis of simplicial signals
/// and rigorous computation of Betti numbers via kernel dimension counting.
#[derive(Debug, Clone)]
pub struct HodgeLaplacian {
    /// The underlying simplicial complex
    pub complex: SimplicialComplex,
    /// Boundary matrices: `boundary_matrices[k]` = B_k (maps k-simplices to (k-1)-simplices).
    /// B_k has rows indexed by (k-1)-simplices and columns indexed by k-simplices.
    /// `boundary_matrices[0]` is empty (no boundary below vertices).
    pub boundary_matrices: Vec<Vec<Vec<f64>>>,
    /// Hodge Laplacians: `laplacians[k]` = L_k = B_k^T B_k + B_{k+1} B_{k+1}^T.
    pub laplacians: Vec<Vec<Vec<f64>>>,
}

impl HodgeLaplacian {
    /// Construct the Hodge Laplacian from a simplicial complex.
    ///
    /// Computes all boundary matrices B_k and all Hodge Laplacians L_k
    /// for k = 0, 1, ..., max_dim.
    pub fn new(complex: SimplicialComplex) -> Self {
        // Build index maps for each dimension (simplex -> column/row index)
        let mut index_maps: Vec<HashMap<Vec<usize>, usize>> = Vec::new();
        for k in 0..=complex.max_dim {
            let mut map = HashMap::new();
            if k < complex.simplices.len() {
                for (idx, simplex) in complex.simplices[k].iter().enumerate() {
                    map.insert(simplex.clone(), idx);
                }
            }
            index_maps.push(map);
        }

        // Compute boundary matrices B_1, B_2, ..., B_{max_dim}
        // boundary_matrices[0] is a dummy empty matrix (no boundary below vertices)
        let mut boundary_matrices: Vec<Vec<Vec<f64>>> = Vec::new();
        boundary_matrices.push(Vec::new()); // B_0 = empty

        for k in 1..=complex.max_dim {
            let n_rows = complex.count(k - 1); // (k-1)-simplices
            let n_cols = complex.count(k); // k-simplices

            let mut bk = vec![vec![0.0; n_cols]; n_rows];

            if k < complex.simplices.len() {
                for (col, simplex) in complex.simplices[k].iter().enumerate() {
                    // boundary of [v0, ..., vk] = sum_i (-1)^i [v0, ..., hat(vi), ..., vk]
                    for i in 0..simplex.len() {
                        let mut face = simplex.clone();
                        face.remove(i);
                        if let Some(&row) = index_maps[k - 1].get(&face) {
                            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                            bk[row][col] = sign;
                        }
                    }
                }
            }

            boundary_matrices.push(bk);
        }

        // Compute Hodge Laplacians L_k = B_k^T B_k + B_{k+1} B_{k+1}^T
        let mut laplacians: Vec<Vec<Vec<f64>>> = Vec::new();

        for k in 0..=complex.max_dim {
            let n = complex.count(k);
            let mut lk = vec![vec![0.0; n]; n];

            // Lower part: B_k^T B_k (how k-simplices share (k-1)-boundaries)
            if k >= 1 && k < boundary_matrices.len() {
                let bk = &boundary_matrices[k];
                let bkt_bk = mat_transpose_times_mat(bk);
                mat_add_inplace(&mut lk, &bkt_bk);
            }

            // Upper part: B_{k+1} B_{k+1}^T (how k-simplices co-bound (k+1)-simplices)
            if k + 1 < boundary_matrices.len() {
                let bk1 = &boundary_matrices[k + 1];
                let bk1_bk1t = mat_times_transpose(bk1);
                mat_add_inplace(&mut lk, &bk1_bk1t);
            }

            laplacians.push(lk);
        }

        Self {
            complex,
            boundary_matrices,
            laplacians,
        }
    }

    /// Get the boundary matrix B_k.
    ///
    /// Returns None if k is out of range or k == 0 (no boundary below vertices).
    pub fn boundary_matrix(&self, k: usize) -> Option<&Vec<Vec<f64>>> {
        if k == 0 || k >= self.boundary_matrices.len() {
            None
        } else {
            Some(&self.boundary_matrices[k])
        }
    }

    /// Get the Hodge k-Laplacian L_k.
    ///
    /// Returns None if k is out of range.
    pub fn laplacian(&self, k: usize) -> Option<&Vec<Vec<f64>>> {
        if k >= self.laplacians.len() {
            None
        } else {
            Some(&self.laplacians[k])
        }
    }

    /// Compute the Betti numbers of the simplicial complex.
    ///
    /// Uses the Hodge theorem: beta_k = dim(ker(L_k)), i.e., the number of
    /// zero eigenvalues of the k-th Hodge Laplacian equals the k-th Betti number.
    ///
    /// This is mathematically rigorous, unlike Euler-characteristic approximations.
    pub fn betti_numbers(&self) -> BettiNumbers {
        let mut numbers = Vec::new();
        for k in 0..=self.complex.max_dim {
            let lk = &self.laplacians[k];
            let kernel_dim = count_kernel_dimension(lk);
            numbers.push(kernel_dim);
        }
        BettiNumbers { numbers }
    }

    /// Perform the Hodge decomposition of a signal on k-simplices.
    ///
    /// Decomposes a k-chain omega into three mutually orthogonal components:
    /// - **exact**: in image(B_{k+1}^T) -- "gradient" flow from higher simplices
    /// - **coexact**: in image(B_k) -- "curl" flow from lower simplices
    /// - **harmonic**: in ker(L_k) -- topologically-protected global modes
    ///
    /// The signal length must equal the number of k-simplices.
    pub fn hodge_decompose(&self, k: usize, signal: &[f64]) -> Option<HodgeDecomposition> {
        if k > self.complex.max_dim {
            return None;
        }
        let n = self.complex.count(k);
        if signal.len() != n || n == 0 {
            return None;
        }

        // Compute the harmonic component by projecting onto ker(L_k)
        let lk = &self.laplacians[k];
        let harmonic = project_onto_kernel(lk, signal);

        // Compute the exact component: projection onto image(B_{k+1}^T)
        // image(B_{k+1}^T) = column space of B_{k+1}^T = row space of B_{k+1}
        let exact = if k + 1 < self.boundary_matrices.len() {
            let bk1 = &self.boundary_matrices[k + 1];
            // B_{k+1}^T has shape (n_k x n_{k+1}), its columns span the exact subspace
            let bk1_t = mat_transpose(bk1);
            project_onto_column_space(&bk1_t, signal)
        } else {
            vec![0.0; n]
        };

        // Coexact = signal - exact - harmonic (by orthogonality of the decomposition)
        let coexact: Vec<f64> = (0..n).map(|i| signal[i] - exact[i] - harmonic[i]).collect();

        Some(HodgeDecomposition {
            exact,
            coexact,
            harmonic,
        })
    }

    /// Compute the eigenvalues of the k-th Hodge Laplacian.
    ///
    /// Uses the Jacobi eigenvalue algorithm for symmetric matrices.
    /// Returns eigenvalues sorted in ascending order.
    pub fn spectrum(&self, k: usize) -> Option<Vec<f64>> {
        if k >= self.laplacians.len() {
            return None;
        }
        let lk = &self.laplacians[k];
        if lk.is_empty() {
            return Some(Vec::new());
        }
        Some(symmetric_eigenvalues(lk))
    }

    /// Compute the full Hodge spectrum for all dimensions.
    ///
    /// Returns eigenvalues, spectral gaps, and Betti numbers for each dimension.
    pub fn full_spectrum(&self) -> HodgeSpectrum {
        let betti = self.betti_numbers();
        let mut eigenvalues = Vec::new();
        let mut spectral_gaps = Vec::new();

        for k in 0..=self.complex.max_dim {
            let eigs = self.spectrum(k).unwrap_or_default();
            // Spectral gap = smallest nonzero eigenvalue
            let gap = eigs
                .iter()
                .copied()
                .find(|&e| e > ZERO_THRESHOLD)
                .unwrap_or(0.0);
            spectral_gaps.push(gap);
            eigenvalues.push(eigs);
        }

        HodgeSpectrum {
            eigenvalues,
            spectral_gaps,
            betti_numbers: betti,
        }
    }
}

// ============================================================================
// RESULT TYPES
// ============================================================================

/// The Hodge decomposition of a simplicial signal.
///
/// Any k-chain omega decomposes uniquely into three mutually orthogonal components:
/// omega = exact + coexact + harmonic
#[derive(Debug, Clone)]
pub struct HodgeDecomposition {
    /// Exact component (in image of d = B_{k+1}^T): gradient-like flow
    pub exact: Vec<f64>,
    /// Coexact component (in image of d* = B_k): curl-like flow
    pub coexact: Vec<f64>,
    /// Harmonic component (in ker L_k): topologically-protected global modes
    pub harmonic: Vec<f64>,
}

impl HodgeDecomposition {
    fn norm_sq(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum()
    }

    /// Total signal energy (||exact||² + ||coexact||² + ||harmonic||²).
    pub fn total_energy(&self) -> f64 {
        Self::norm_sq(&self.exact) + Self::norm_sq(&self.coexact) + Self::norm_sq(&self.harmonic)
    }

    /// Fraction of signal energy in the harmonic component (0.0–1.0).
    /// Topologically-protected global resonance modes.
    pub fn harmonic_fraction(&self) -> f64 {
        let total = self.total_energy();
        if total < 1e-15 {
            return 0.0;
        }
        Self::norm_sq(&self.harmonic) / total
    }

    /// Fraction in the gradient (exact) component (0.0–1.0).
    /// Hierarchical, feed-forward information transfer.
    pub fn gradient_fraction(&self) -> f64 {
        let total = self.total_energy();
        if total < 1e-15 {
            return 0.0;
        }
        Self::norm_sq(&self.exact) / total
    }

    /// Fraction in the curl (coexact) component (0.0–1.0).
    /// Recurrent, rotational information cycling.
    pub fn curl_fraction(&self) -> f64 {
        let total = self.total_energy();
        if total < 1e-15 {
            return 0.0;
        }
        Self::norm_sq(&self.coexact) / total
    }

    /// All three fractions: (gradient, curl, harmonic). Sum ≈ 1.0.
    pub fn fractions(&self) -> (f64, f64, f64) {
        (
            self.gradient_fraction(),
            self.curl_fraction(),
            self.harmonic_fraction(),
        )
    }
}

/// Betti numbers computed from the Hodge Laplacian.
///
/// beta_k = dim(ker(L_k)) counts the number of k-dimensional "holes":
/// - beta_0 = connected components
/// - beta_1 = independent loops / tunnels
/// - beta_2 = enclosed voids / cavities
#[derive(Debug, Clone)]
pub struct BettiNumbers {
    /// Betti numbers indexed by dimension: numbers[k] = beta_k
    pub numbers: Vec<usize>,
}

impl BettiNumbers {
    /// Get beta_k, returning 0 if k is out of range.
    pub fn get(&self, k: usize) -> usize {
        self.numbers.get(k).copied().unwrap_or(0)
    }

    /// Euler characteristic: chi = sum_k (-1)^k beta_k
    pub fn euler_characteristic(&self) -> i64 {
        self.numbers
            .iter()
            .enumerate()
            .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
            .sum()
    }
}

/// Full spectral analysis of the Hodge Laplacian across all dimensions.
#[derive(Debug, Clone)]
pub struct HodgeSpectrum {
    /// Eigenvalues of L_k for each dimension k
    pub eigenvalues: Vec<Vec<f64>>,
    /// Spectral gap (smallest nonzero eigenvalue) for each dimension k
    pub spectral_gaps: Vec<f64>,
    /// Betti numbers derived from kernel dimensions
    pub betti_numbers: BettiNumbers,
}

// ============================================================================
// LINEAR ALGEBRA HELPERS (no external dependencies)
// ============================================================================

/// Threshold for considering an eigenvalue as zero.
const ZERO_THRESHOLD: f64 = 1e-10;

/// Maximum iterations for eigenvalue algorithms.
const MAX_JACOBI_ITERATIONS: usize = 200;

/// Compute A^T * A for a matrix A stored as rows.
fn mat_transpose_times_mat(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let m = a.len(); // rows
    let n = a[0].len(); // cols

    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for row in a.iter().take(m) {
                sum += row[i] * row[j];
            }
            result[i][j] = sum;
        }
    }
    result
}

/// Compute A * A^T for a matrix A stored as rows.
fn mat_times_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let m = a.len(); // rows
    let n = a[0].len(); // cols

    let mut result = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i][k] * a[j][k];
            }
            result[i][j] = sum;
        }
    }
    result
}

/// Transpose a matrix.
fn mat_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let m = a.len();
    let n = a[0].len();
    let mut result = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            result[j][i] = a[i][j];
        }
    }
    result
}

/// Add matrix B into matrix A in-place: A += B.
fn mat_add_inplace(a: &mut [Vec<f64>], b: &[Vec<f64>]) {
    for i in 0..a.len().min(b.len()) {
        for j in 0..a[i].len().min(b[i].len()) {
            a[i][j] += b[i][j];
        }
    }
}

/// Matrix-vector multiply: result = A * v.
#[cfg(test)]
fn mat_vec_mul(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Dot product of two vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a vector.
fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Compute eigenvalues of a real symmetric matrix using the Jacobi method.
///
/// The Jacobi eigenvalue algorithm iteratively applies Givens rotations to
/// diagonalize a symmetric matrix. It is unconditionally stable and converges
/// for all symmetric matrices.
///
/// Returns eigenvalues sorted in ascending order.
fn symmetric_eigenvalues(mat: &[Vec<f64>]) -> Vec<f64> {
    let n = mat.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![mat[0][0]];
    }

    // Work on a mutable copy
    let mut a: Vec<Vec<f64>> = mat.to_vec();

    for _ in 0..MAX_JACOBI_ITERATIONS {
        // Find the largest off-diagonal element
        let mut max_val = 0.0_f64;
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

        // Convergence check
        if max_val < 1e-14 {
            break;
        }

        // Compute Jacobi rotation parameters
        let theta = if (a[p][p] - a[q][q]).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation: A' = G^T A G
        let mut new_a = a.clone();

        // Update rows/cols p and q
        for i in 0..n {
            if i == p || i == q {
                continue;
            }
            new_a[i][p] = c * a[i][p] + s * a[i][q];
            new_a[p][i] = new_a[i][p];
            new_a[i][q] = -s * a[i][p] + c * a[i][q];
            new_a[q][i] = new_a[i][q];
        }

        new_a[p][p] = c * c * a[p][p] + 2.0 * s * c * a[p][q] + s * s * a[q][q];
        new_a[q][q] = s * s * a[p][p] - 2.0 * s * c * a[p][q] + c * c * a[q][q];
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;

        a = new_a;
    }

    // Diagonal entries are the eigenvalues
    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();

    // Clamp tiny negatives to zero (numerical noise on positive semi-definite matrices)
    for e in &mut eigenvalues {
        if *e < 0.0 && *e > -ZERO_THRESHOLD {
            *e = 0.0;
        }
    }

    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

/// Count the dimension of the kernel (null space) of a symmetric matrix.
///
/// Uses eigenvalue computation: kernel dimension = number of zero eigenvalues.
fn count_kernel_dimension(mat: &[Vec<f64>]) -> usize {
    if mat.is_empty() {
        return 0;
    }
    let eigenvalues = symmetric_eigenvalues(mat);
    eigenvalues
        .iter()
        .filter(|&&e| e.abs() < ZERO_THRESHOLD)
        .count()
}

/// Project a vector onto the kernel (null space) of a symmetric matrix.
///
/// Uses the eigendecomposition: the kernel is spanned by eigenvectors
/// corresponding to zero eigenvalues. We compute these via inverse iteration
/// with shift (shifted inverse power method).
fn project_onto_kernel(mat: &[Vec<f64>], signal: &[f64]) -> Vec<f64> {
    let n = mat.len();
    if n == 0 || signal.len() != n {
        return vec![0.0; signal.len()];
    }

    // Find eigenvectors for zero eigenvalues using the Jacobi method
    // (we need the full eigendecomposition here)
    let eigenvecs = symmetric_eigenvectors(mat);
    let eigenvals = symmetric_eigenvalues(mat);

    let mut projection = vec![0.0; n];
    for (idx, &eval) in eigenvals.iter().enumerate() {
        if eval.abs() < ZERO_THRESHOLD && idx < eigenvecs.len() {
            let evec = &eigenvecs[idx];
            let coeff = dot(signal, evec);
            for i in 0..n {
                projection[i] += coeff * evec[i];
            }
        }
    }

    projection
}

/// Project a vector onto the column space of a matrix.
///
/// Uses the normal equation approach: projection = A * (A^T A)^{-1} * A^T * v,
/// but computed more stably via QR-like orthogonalization of columns.
fn project_onto_column_space(a: &[Vec<f64>], signal: &[f64]) -> Vec<f64> {
    if a.is_empty() || signal.is_empty() {
        return vec![0.0; signal.len()];
    }
    let n = a.len(); // rows
    let m = a[0].len(); // cols

    if n != signal.len() {
        return vec![0.0; signal.len()];
    }

    // Extract columns and orthogonalize via modified Gram-Schmidt
    let mut columns: Vec<Vec<f64>> = Vec::new();
    for j in 0..m {
        let col: Vec<f64> = (0..n).map(|i| a[i][j]).collect();
        columns.push(col);
    }

    let ortho_basis = gram_schmidt(&columns);

    // Project signal onto the orthonormal basis
    let mut projection = vec![0.0; n];
    for basis_vec in &ortho_basis {
        let coeff = dot(signal, basis_vec);
        for i in 0..n {
            projection[i] += coeff * basis_vec[i];
        }
    }

    projection
}

/// Modified Gram-Schmidt orthonormalization.
///
/// Returns an orthonormal basis for the column space, discarding
/// near-zero vectors (linearly dependent columns).
fn gram_schmidt(vectors: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut basis: Vec<Vec<f64>> = Vec::new();

    for v in vectors {
        let mut u = v.clone();

        // Subtract projections onto existing basis vectors
        for b in &basis {
            let coeff = dot(&u, b);
            for i in 0..u.len() {
                u[i] -= coeff * b[i];
            }
        }

        let n = norm(&u);
        if n > 1e-12 {
            for x in &mut u {
                *x /= n;
            }
            basis.push(u);
        }
    }

    basis
}

/// Compute eigenvectors of a symmetric matrix using the Jacobi method.
///
/// Returns eigenvectors sorted by eigenvalue (ascending).
fn symmetric_eigenvectors(mat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = mat.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![vec![1.0]];
    }

    let mut a: Vec<Vec<f64>> = mat.to_vec();

    // Accumulated rotation matrix (starts as identity)
    let mut v: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    for _ in 0..MAX_JACOBI_ITERATIONS {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
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

        if max_val < 1e-14 {
            break;
        }

        let theta = if (a[p][p] - a[q][q]).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to A
        let mut new_a = a.clone();
        for i in 0..n {
            if i == p || i == q {
                continue;
            }
            new_a[i][p] = c * a[i][p] + s * a[i][q];
            new_a[p][i] = new_a[i][p];
            new_a[i][q] = -s * a[i][p] + c * a[i][q];
            new_a[q][i] = new_a[i][q];
        }
        new_a[p][p] = c * c * a[p][p] + 2.0 * s * c * a[p][q] + s * s * a[q][q];
        new_a[q][q] = s * s * a[p][p] - 2.0 * s * c * a[p][q] + c * c * a[q][q];
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;
        a = new_a;

        // Accumulate rotation into V
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip + s * viq;
            v[i][q] = -s * vip + c * viq;
        }
    }

    // Extract eigenvalues and pair with eigenvectors
    let mut eigen_pairs: Vec<(f64, Vec<f64>)> = Vec::new();
    for j in 0..n {
        let eigenval = a[j][j];
        let eigenvec: Vec<f64> = (0..n).map(|i| v[i][j]).collect();
        eigen_pairs.push((eigenval, eigenvec));
    }

    // Sort by eigenvalue ascending
    eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    eigen_pairs.into_iter().map(|(_, vec)| vec).collect()
}

pub mod consciousness_topology;

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: check that two floats are approximately equal.
    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ---- Betti number tests ----

    #[test]
    fn test_filled_triangle_betti_numbers() {
        // A filled triangle: 3 vertices, 3 edges, 1 face
        // beta_0 = 1 (connected), beta_1 = 0 (no loops -- face fills the hole), beta_2 = 0
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2]); // Adds the triangle and all faces

        assert_eq!(complex.count(0), 3, "should have 3 vertices");
        assert_eq!(complex.count(1), 3, "should have 3 edges");
        assert_eq!(complex.count(2), 1, "should have 1 triangle");

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();

        assert_eq!(betti.get(0), 1, "beta_0 should be 1 (connected)");
        assert_eq!(
            betti.get(1),
            0,
            "beta_1 should be 0 (no loop -- face fills it)"
        );
        assert_eq!(betti.get(2), 0, "beta_2 should be 0 (no void)");
    }

    #[test]
    fn test_triangle_boundary_only_betti_numbers() {
        // Triangle boundary (no face): 3 vertices, 3 edges, 0 faces
        // beta_0 = 1 (connected), beta_1 = 1 (one loop!)
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![1, 2]);
        complex.add_simplex(vec![0, 2]);

        assert_eq!(complex.count(0), 3, "should have 3 vertices");
        assert_eq!(complex.count(1), 3, "should have 3 edges");
        assert_eq!(complex.count(2), 0, "should have 0 triangles");

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();

        assert_eq!(betti.get(0), 1, "beta_0 should be 1 (connected)");
        assert_eq!(betti.get(1), 1, "beta_1 should be 1 (one loop)");
    }

    #[test]
    fn test_two_disconnected_vertices() {
        // Two isolated vertices: beta_0 = 2
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0]);
        complex.add_simplex(vec![1]);

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();

        assert_eq!(betti.get(0), 2, "beta_0 should be 2 (disconnected)");
    }

    #[test]
    fn test_single_edge() {
        // One edge {0, 1}: beta_0 = 1, beta_1 = 0
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();

        assert_eq!(betti.get(0), 1, "beta_0 should be 1 (connected)");
        assert_eq!(betti.get(1), 0, "beta_1 should be 0 (no loop)");
    }

    // ---- Boundary of boundary = 0 ----

    #[test]
    fn test_boundary_of_boundary_is_zero() {
        // For a triangle [0,1,2], B_1 * B_2 should be the zero matrix.
        // This is the fundamental property: partial^2 = 0.
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2]);

        let hodge = HodgeLaplacian::new(complex);

        let b1 = hodge.boundary_matrix(1).expect("B_1 should exist");
        let b2 = hodge.boundary_matrix(2).expect("B_2 should exist");

        // Compute B_1 * B_2
        let n_rows = b1.len();
        let n_inner = b1[0].len();
        let n_cols = b2[0].len();

        for i in 0..n_rows {
            for j in 0..n_cols {
                let mut val = 0.0;
                for k in 0..n_inner {
                    val += b1[i][k] * b2[k][j];
                }
                assert!(
                    val.abs() < 1e-12,
                    "B_1 * B_2 should be zero, but got [{i}][{j}] = {val}"
                );
            }
        }
    }

    #[test]
    fn test_boundary_squared_tetrahedron() {
        // For a tetrahedron [0,1,2,3], B_2 * B_3 should be zero.
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2, 3]);

        let hodge = HodgeLaplacian::new(complex);

        let b2 = hodge.boundary_matrix(2).expect("B_2 should exist");
        let b3 = hodge.boundary_matrix(3).expect("B_3 should exist");

        let n_rows = b2.len();
        let n_inner = b2[0].len();
        let n_cols = b3[0].len();

        for i in 0..n_rows {
            for j in 0..n_cols {
                let mut val = 0.0;
                for k in 0..n_inner {
                    val += b2[i][k] * b3[k][j];
                }
                assert!(
                    val.abs() < 1e-12,
                    "B_2 * B_3 should be zero, but got [{i}][{j}] = {val}"
                );
            }
        }
    }

    // ---- Hodge decomposition tests ----

    #[test]
    fn test_harmonic_has_zero_laplacian() {
        // For a triangle boundary (has a loop), the harmonic component
        // should satisfy L_1 * h = 0.
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![1, 2]);
        complex.add_simplex(vec![0, 2]);

        let hodge = HodgeLaplacian::new(complex);

        // An arbitrary edge signal
        let signal = vec![1.0, -0.5, 0.3];
        let decomp = hodge
            .hodge_decompose(1, &signal)
            .expect("decomposition should work");

        // L_1 * harmonic should be approximately zero
        let l1 = hodge.laplacian(1).unwrap();
        let l1_h = mat_vec_mul(l1, &decomp.harmonic);

        for (i, val) in l1_h.iter().enumerate() {
            assert!(
                val.abs() < 1e-8,
                "L_1 * harmonic should be zero, but component {i} = {val}"
            );
        }
    }

    #[test]
    fn test_hodge_decomposition_orthogonality() {
        // The three components of the Hodge decomposition should be mutually orthogonal.
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![1, 2]);
        complex.add_simplex(vec![0, 2]);

        let hodge = HodgeLaplacian::new(complex);

        let signal = vec![2.0, -1.0, 0.5];
        let decomp = hodge
            .hodge_decompose(1, &signal)
            .expect("decomposition should work");

        // Check pairwise orthogonality
        let dot_eh = dot(&decomp.exact, &decomp.harmonic);
        let dot_ec = dot(&decomp.exact, &decomp.coexact);
        let dot_ch = dot(&decomp.coexact, &decomp.harmonic);

        assert!(
            dot_eh.abs() < 1e-8,
            "exact and harmonic should be orthogonal, dot = {dot_eh}"
        );
        assert!(
            dot_ec.abs() < 1e-8,
            "exact and coexact should be orthogonal, dot = {dot_ec}"
        );
        assert!(
            dot_ch.abs() < 1e-8,
            "coexact and harmonic should be orthogonal, dot = {dot_ch}"
        );

        // Check reconstruction: exact + coexact + harmonic = signal
        for i in 0..signal.len() {
            let reconstructed = decomp.exact[i] + decomp.coexact[i] + decomp.harmonic[i];
            assert!(
                approx_eq(reconstructed, signal[i], 1e-8),
                "reconstruction failed at {i}: {reconstructed} != {}",
                signal[i]
            );
        }
    }

    // ---- Spectrum tests ----

    #[test]
    fn test_spectrum_zero_eigenvalues_match_betti() {
        // The number of zero eigenvalues of L_k should equal beta_k.
        let mut complex = SimplicialComplex::new();
        // Triangle boundary: beta_0 = 1, beta_1 = 1
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![1, 2]);
        complex.add_simplex(vec![0, 2]);

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();

        // L_0 spectrum
        let spec0 = hodge.spectrum(0).unwrap();
        let zeros_0 = spec0.iter().filter(|&&e| e.abs() < ZERO_THRESHOLD).count();
        assert_eq!(
            zeros_0,
            betti.get(0),
            "zero eigenvalues of L_0 should match beta_0"
        );

        // L_1 spectrum
        let spec1 = hodge.spectrum(1).unwrap();
        let zeros_1 = spec1.iter().filter(|&&e| e.abs() < ZERO_THRESHOLD).count();
        assert_eq!(
            zeros_1,
            betti.get(1),
            "zero eigenvalues of L_1 should match beta_1"
        );
    }

    #[test]
    fn test_laplacian_is_positive_semidefinite() {
        // All eigenvalues of L_k should be >= 0 (Hodge Laplacian is PSD).
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2]);
        complex.add_simplex(vec![2, 3]);

        let hodge = HodgeLaplacian::new(complex);

        for k in 0..=hodge.complex.max_dim {
            let spec = hodge.spectrum(k).unwrap();
            for (i, &e) in spec.iter().enumerate() {
                assert!(
                    e >= -ZERO_THRESHOLD,
                    "L_{k} eigenvalue [{i}] = {e} is negative (should be >= 0)"
                );
            }
        }
    }

    // ---- Euler characteristic ----

    #[test]
    fn test_euler_characteristic() {
        // Filled triangle: chi = 1 - 0 + 0 = 1
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2]);

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();

        assert_eq!(
            betti.euler_characteristic(),
            1,
            "chi of filled triangle should be 1"
        );
    }

    #[test]
    fn test_euler_characteristic_boundary() {
        // Triangle boundary: chi = 1 - 1 = 0
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![1, 2]);
        complex.add_simplex(vec![0, 2]);

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();

        assert_eq!(
            betti.euler_characteristic(),
            0,
            "chi of triangle boundary should be 0"
        );
    }

    // ---- from_graph test ----

    #[test]
    fn test_from_graph_triangle() {
        // A complete graph on 3 vertices should produce a filled triangle
        let adjacency = vec![
            vec![false, true, true],
            vec![true, false, true],
            vec![true, true, false],
        ];
        let complex = SimplicialComplex::from_graph(&adjacency);

        assert_eq!(complex.count(0), 3);
        assert_eq!(complex.count(1), 3);
        assert_eq!(complex.count(2), 1, "3-clique should produce a triangle");

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();
        assert_eq!(betti.get(0), 1);
        assert_eq!(betti.get(1), 0, "filled triangle has no loop");
    }

    #[test]
    fn test_from_graph_path() {
        // Path graph: 0-1-2 (no triangle)
        let adjacency = vec![
            vec![false, true, false],
            vec![true, false, true],
            vec![false, true, false],
        ];
        let complex = SimplicialComplex::from_graph(&adjacency);

        assert_eq!(complex.count(0), 3);
        assert_eq!(complex.count(1), 2);
        assert_eq!(complex.count(2), 0);

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();
        assert_eq!(betti.get(0), 1, "path graph is connected");
        assert_eq!(betti.get(1), 0, "no loop in a path");
    }

    // ---- Full spectrum test ----

    #[test]
    fn test_full_spectrum() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![1, 2]);
        complex.add_simplex(vec![0, 2]);

        let hodge = HodgeLaplacian::new(complex);
        let spectrum = hodge.full_spectrum();

        // beta_0 = 1 => spectral gap of L_0 should be > 0 (since graph is connected)
        assert!(
            spectrum.spectral_gaps[0] > 0.0,
            "connected graph should have nonzero spectral gap for L_0"
        );

        // beta_1 = 1 => L_1 should have one zero eigenvalue
        assert_eq!(spectrum.betti_numbers.get(1), 1);
    }

    // ---- Tetrahedron test ----

    #[test]
    fn test_filled_tetrahedron_betti() {
        // Filled tetrahedron: 4 vertices, 6 edges, 4 triangles, 1 tetrahedron
        // beta_0 = 1, beta_1 = 0, beta_2 = 0, beta_3 = 0
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2, 3]);

        assert_eq!(complex.count(0), 4);
        assert_eq!(complex.count(1), 6);
        assert_eq!(complex.count(2), 4);
        assert_eq!(complex.count(3), 1);

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();

        assert_eq!(betti.get(0), 1, "tetrahedron: beta_0 = 1");
        assert_eq!(betti.get(1), 0, "tetrahedron: beta_1 = 0");
        assert_eq!(betti.get(2), 0, "tetrahedron: beta_2 = 0");
        assert_eq!(betti.get(3), 0, "tetrahedron: beta_3 = 0");
    }

    #[test]
    fn test_tetrahedron_boundary_betti() {
        // Boundary of tetrahedron (hollow): 4 vertices, 6 edges, 4 triangles, 0 tetrahedra
        // This is topologically a sphere S^2: beta_0 = 1, beta_1 = 0, beta_2 = 1
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2]);
        complex.add_simplex(vec![0, 1, 3]);
        complex.add_simplex(vec![0, 2, 3]);
        complex.add_simplex(vec![1, 2, 3]);

        assert_eq!(complex.count(0), 4);
        assert_eq!(complex.count(1), 6);
        assert_eq!(complex.count(2), 4);
        assert_eq!(complex.count(3), 0);

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();

        assert_eq!(betti.get(0), 1, "sphere: beta_0 = 1");
        assert_eq!(betti.get(1), 0, "sphere: beta_1 = 0");
        assert_eq!(betti.get(2), 1, "sphere: beta_2 = 1 (one void)");
    }

    // ════════════════════════════════════════════════════════════════════════
    // NEW TESTS: Construction, Edge Cases, Invariants, Round-trips
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_empty_complex() {
        let complex = SimplicialComplex::new();
        assert_eq!(complex.count(0), 0);
        assert_eq!(complex.max_dim, 0);
        assert!(complex.vertices.is_empty());
    }

    #[test]
    fn test_single_vertex() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0]);
        assert_eq!(complex.count(0), 1);
        assert_eq!(complex.max_dim, 0);

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();
        assert_eq!(betti.get(0), 1, "Single vertex: beta_0 = 1");
    }

    #[test]
    fn test_add_duplicate_simplex() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![0, 1]); // Duplicate
        complex.add_simplex(vec![1, 0]); // Same edge reversed
        assert_eq!(complex.count(1), 1, "Duplicate edges should not be counted");
        assert_eq!(complex.count(0), 2);
    }

    #[test]
    fn test_add_simplex_empty_is_noop() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![]);
        assert_eq!(complex.count(0), 0, "Empty simplex should be ignored");
    }

    #[test]
    fn test_add_simplex_with_duplicates_deduped() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 0, 1]); // Duplicate vertex
        // After dedup: [0, 1] which is an edge
        assert_eq!(complex.count(1), 1);
        assert_eq!(complex.count(0), 2);
    }

    #[test]
    fn test_closure_property() {
        // Adding a triangle should also add its 3 edges and 3 vertices
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2]);
        assert_eq!(complex.count(0), 3, "Triangle should have 3 vertices");
        assert_eq!(complex.count(1), 3, "Triangle should have 3 edges");
        assert_eq!(complex.count(2), 1, "Triangle should have 1 face");
    }

    #[test]
    fn test_count_out_of_range() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        assert_eq!(
            complex.count(5),
            0,
            "Out of range dimension should return 0"
        );
    }

    #[test]
    fn test_betti_out_of_range() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();
        assert_eq!(
            betti.get(10),
            0,
            "Out of range betti number should return 0"
        );
    }

    #[test]
    fn test_boundary_matrix_k0_is_none() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        let hodge = HodgeLaplacian::new(complex);
        assert!(hodge.boundary_matrix(0).is_none(), "B_0 should not exist");
    }

    #[test]
    fn test_boundary_matrix_out_of_range_is_none() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        let hodge = HodgeLaplacian::new(complex);
        assert!(hodge.boundary_matrix(5).is_none());
    }

    #[test]
    fn test_laplacian_out_of_range_is_none() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        let hodge = HodgeLaplacian::new(complex);
        assert!(hodge.laplacian(5).is_none());
    }

    #[test]
    fn test_laplacian_symmetry() {
        // Hodge Laplacians should always be symmetric
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2]);
        complex.add_simplex(vec![2, 3]);
        let hodge = HodgeLaplacian::new(complex);

        for k in 0..=hodge.complex.max_dim {
            let lk = hodge.laplacian(k).unwrap();
            let n = lk.len();
            for i in 0..n {
                for j in 0..n {
                    assert!(
                        approx_eq(lk[i][j], lk[j][i], 1e-12),
                        "L_{k} should be symmetric: L[{i}][{j}]={} != L[{j}][{i}]={}",
                        lk[i][j],
                        lk[j][i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_hodge_decompose_invalid_dimension() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        let hodge = HodgeLaplacian::new(complex);
        assert!(hodge.hodge_decompose(5, &[1.0]).is_none());
    }

    #[test]
    fn test_hodge_decompose_wrong_signal_length() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![1, 2]);
        let hodge = HodgeLaplacian::new(complex);
        // We have 2 edges, but pass signal of length 3
        assert!(hodge.hodge_decompose(1, &[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn test_hodge_decompose_reconstruction() {
        // For any signal: exact + coexact + harmonic = signal
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2]); // Filled triangle
        complex.add_simplex(vec![2, 3]);

        let hodge = HodgeLaplacian::new(complex);

        // Edge signal (5 edges)
        let n_edges = hodge.complex.count(1);
        let signal: Vec<f64> = (0..n_edges).map(|i| (i as f64 + 1.0) * 0.3).collect();

        let decomp = hodge
            .hodge_decompose(1, &signal)
            .expect("decomposition should work");

        for i in 0..signal.len() {
            let reconstructed = decomp.exact[i] + decomp.coexact[i] + decomp.harmonic[i];
            assert!(
                approx_eq(reconstructed, signal[i], 1e-6),
                "Reconstruction failed at index {i}: {reconstructed} != {}",
                signal[i]
            );
        }
    }

    #[test]
    fn test_hodge_decompose_vertex_signal() {
        // Vertex signals (k=0) should decompose too
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2]);
        let hodge = HodgeLaplacian::new(complex);

        let signal = vec![1.0, 0.0, -1.0];
        let decomp = hodge
            .hodge_decompose(0, &signal)
            .expect("vertex decomposition should work");

        // Reconstruction check
        for i in 0..signal.len() {
            let reconstructed = decomp.exact[i] + decomp.coexact[i] + decomp.harmonic[i];
            assert!(
                approx_eq(reconstructed, signal[i], 1e-6),
                "Vertex signal reconstruction failed at {i}"
            );
        }
    }

    #[test]
    fn test_spectrum_out_of_range() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        let hodge = HodgeLaplacian::new(complex);
        assert!(hodge.spectrum(5).is_none());
    }

    #[test]
    fn test_spectrum_single_vertex() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0]);
        let hodge = HodgeLaplacian::new(complex);
        let spec = hodge.spectrum(0).unwrap();
        assert_eq!(spec.len(), 1);
        assert!(
            approx_eq(spec[0], 0.0, 1e-10),
            "Single isolated vertex has L_0 = [0]"
        );
    }

    #[test]
    fn test_spectrum_sorted_ascending() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![1, 2]);
        complex.add_simplex(vec![0, 2]);
        let hodge = HodgeLaplacian::new(complex);

        for k in 0..=hodge.complex.max_dim {
            let spec = hodge.spectrum(k).unwrap();
            for i in 1..spec.len() {
                assert!(
                    spec[i] >= spec[i - 1] - 1e-10,
                    "Spectrum should be sorted ascending at dim {k}: {} > {}",
                    spec[i - 1],
                    spec[i]
                );
            }
        }
    }

    #[test]
    fn test_two_disconnected_edges() {
        // Two disconnected edges: {0,1} and {2,3}
        // beta_0 = 2 (two components), beta_1 = 0
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![2, 3]);

        assert_eq!(complex.count(0), 4);
        assert_eq!(complex.count(1), 2);

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();

        assert_eq!(betti.get(0), 2, "Two disconnected edges: beta_0 = 2");
        assert_eq!(betti.get(1), 0, "No loops");
    }

    #[test]
    fn test_square_boundary_has_loop() {
        // Square boundary (4 vertices, 4 edges, no faces)
        // 0-1, 1-2, 2-3, 3-0 forms a cycle
        // beta_0 = 1, beta_1 = 1
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![1, 2]);
        complex.add_simplex(vec![2, 3]);
        complex.add_simplex(vec![0, 3]);

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();

        assert_eq!(betti.get(0), 1, "Square is connected");
        assert_eq!(betti.get(1), 1, "Square boundary has one loop");
    }

    #[test]
    fn test_euler_characteristic_two_triangles_sharing_edge() {
        // Two filled triangles sharing edge {1,2}:
        // [0,1,2] and [1,2,3]
        // 4 vertices, 5 edges, 2 triangles
        // chi = V - E + F = 4 - 5 + 2 = 1
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2]);
        complex.add_simplex(vec![1, 2, 3]);

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();

        assert_eq!(betti.euler_characteristic(), 1);
        assert_eq!(betti.get(0), 1, "Connected");
        assert_eq!(betti.get(1), 0, "No loops (faces fill them)");
    }

    #[test]
    fn test_from_graph_disconnected() {
        // 4 nodes, no edges
        let adjacency = vec![
            vec![false, false, false, false],
            vec![false, false, false, false],
            vec![false, false, false, false],
            vec![false, false, false, false],
        ];
        let complex = SimplicialComplex::from_graph(&adjacency);
        assert_eq!(complex.count(0), 4);
        assert_eq!(complex.count(1), 0);

        let hodge = HodgeLaplacian::new(complex);
        let betti = hodge.betti_numbers();
        assert_eq!(betti.get(0), 4, "4 isolated vertices: beta_0 = 4");
    }

    #[test]
    fn test_full_spectrum_spectral_gap() {
        // Connected graph should have positive spectral gap for L_0
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2]);
        let hodge = HodgeLaplacian::new(complex);
        let spec = hodge.full_spectrum();

        assert!(
            spec.spectral_gaps[0] > 0.0,
            "Connected: L_0 spectral gap > 0"
        );
    }

    #[test]
    fn test_laplacian_diagonal_nonnegative() {
        // Diagonal entries of Hodge Laplacian should be non-negative (degrees)
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2]);
        complex.add_simplex(vec![2, 3]);
        let hodge = HodgeLaplacian::new(complex);

        for k in 0..=hodge.complex.max_dim {
            let lk = hodge.laplacian(k).unwrap();
            for i in 0..lk.len() {
                assert!(
                    lk[i][i] >= -1e-10,
                    "L_{k}[{i}][{i}] = {} should be non-negative",
                    lk[i][i]
                );
            }
        }
    }
}
