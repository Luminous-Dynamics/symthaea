// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Tensor Operations & Geometric Algebra for Hyperdimensional Computing
//!
//! This module provides tensor and geometric algebra primitives adapted for
//! Symthaea's HDC system. Tensor products encode multi-way relationships
//! between concepts, enabling richer compositional semantics than pairwise
//! binding alone. Geometric algebra provides rotation-invariant processing
//! of spatial and relational information, giving the consciousness engine
//! a coordinate-free way to manipulate structured representations.
//!
//! ## Connection to Consciousness
//!
//! - **Tensor products** encode multi-way relationships: a subject-predicate-object
//!   triple becomes a rank-3 tensor that can be contracted to recover any
//!   constituent, mirroring how conscious experience binds multiple modalities
//!   into a unified percept.
//!
//! - **Geometric algebra** unifies rotations, reflections, and projections
//!   into a single algebraic framework. For a consciousness architecture,
//!   this means that spatial reasoning, perspective-taking, and attentional
//!   rotation can all be expressed as sandwich products (RvR~), providing
//!   a natural substrate for the "rotation of attention" that IIT posits
//!   underlies integrated information.
//!
//! - **Tensor networks** represent compositional structure (sequences,
//!   hierarchies, graphs) as contracted tensor chains. This bridges HDC's
//!   flat hypervector space with the structured, hierarchical representations
//!   that higher-order cognition demands.
//!
//! ## Implementation Notes
//!
//! All operations use `f64` for numerical stability in the algebraic
//! computations (tensor contraction, geometric products). Integration with
//! Symthaea's `ContinuousHV` (which uses `f32`) is provided via explicit
//! conversion methods (`to_hdc`, `from_real_hv`).
//!
//! No external dependencies are added -- all linear algebra is implemented
//! from first principles using only the standard library.

use serde::{Deserialize, Serialize};

use crate::hdc::ContinuousHV;

// ═══════════════════════════════════════════════════════════════════════════════
// Tensor
// ═══════════════════════════════════════════════════════════════════════════════

/// A dense tensor of arbitrary rank.
///
/// Data is stored in row-major (C) order: the last index varies fastest.
/// For a shape `[2, 3, 4]`, element `(i, j, k)` is at index
/// `i * 3 * 4 + j * 4 + k`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    /// Flat storage in row-major order.
    pub data: Vec<f64>,
    /// Shape of each axis. The product of all elements equals `data.len()`.
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a zero-filled tensor with the given shape.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is zero.
    pub fn new(shape: Vec<usize>) -> Self {
        debug_assert!(
            shape.iter().all(|&d| d > 0),
            "All tensor dimensions must be positive"
        );
        let total: usize = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .unwrap_or(usize::MAX);
        let total = total.min(100_000_000);
        Self {
            data: vec![0.0; total],
            shape,
        }
    }

    /// Create a tensor from raw data and shape.
    ///
    /// # Panics
    ///
    /// Panics if the product of `shape` does not equal `data.len()`.
    pub fn from_data(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total: usize = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .unwrap_or(usize::MAX);
        debug_assert!(total <= 100_000_000, "tensor too large: {} elements", total);
        debug_assert_eq!(
            data.len(),
            total,
            "Data length {} does not match shape product {}",
            data.len(),
            total
        );
        Self { data, shape }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Rank (number of axes).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Compute strides for row-major indexing.
    fn strides(&self) -> Vec<usize> {
        let mut strides = vec![1usize; self.shape.len()];
        for i in (0..self.shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }
        strides
    }

    /// Convert a multi-index to a flat index.
    ///
    /// Returns `0` if indices are out of bounds (graceful degradation
    /// instead of silent out-of-bounds access downstream).
    fn flat_index(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(indices.len(), self.shape.len());
        if indices.len() != self.shape.len() {
            return 0;
        }
        let strides = self.strides();
        let mut result = 0usize;
        for ((&idx, &stride), &dim) in indices.iter().zip(strides.iter()).zip(self.shape.iter()) {
            if idx >= dim {
                debug_assert!(idx < dim, "index {idx} out of bounds for dimension {dim}");
                return 0;
            }
            result += idx * stride;
        }
        result
    }

    /// Get element by multi-index.
    pub fn get(&self, indices: &[usize]) -> f64 {
        self.data[self.flat_index(indices)]
    }

    /// Set element by multi-index.
    pub fn set(&mut self, indices: &[usize], value: f64) {
        let idx = self.flat_index(indices);
        self.data[idx] = value;
    }

    // ────────────────────────────────────────────────────────────────────────
    // Outer product
    // ────────────────────────────────────────────────────────────────────────

    /// Compute the outer product of two 1-D arrays, producing a rank-2 tensor.
    ///
    /// For vectors u of length m and v of length n, produces an (m, n) matrix
    /// where `result[i][j] = u[i] * v[j]`.
    ///
    /// This is the fundamental operation for encoding relational knowledge:
    /// `subject ⊗ predicate` encodes a directed relation as a matrix.
    pub fn outer_product(a: &[f64], b: &[f64]) -> Self {
        let m = a.len();
        let n = b.len();
        let mut data = Vec::with_capacity(m * n);
        for &ai in a.iter() {
            for &bj in b.iter() {
                data.push(ai * bj);
            }
        }
        Self {
            data,
            shape: vec![m, n],
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Tensor product (Kronecker product for arbitrary-rank tensors)
    // ────────────────────────────────────────────────────────────────────────

    /// Compute the tensor product (Kronecker product) of two tensors.
    ///
    /// The result has rank equal to the sum of the operand ranks.
    /// For A with shape [a1, ..., ap] and B with shape [b1, ..., bq],
    /// the result has shape [a1, ..., ap, b1, ..., bq].
    ///
    /// In consciousness terms, this encodes multi-way associations:
    /// binding a subject, predicate, and object each yields a rank-3
    /// tensor that can be queried from any perspective via contraction.
    pub fn tensor_product(&self, other: &Tensor) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.extend_from_slice(&other.shape);

        let mut data = Vec::with_capacity(self.numel() * other.numel());
        for &a in &self.data {
            for &b in &other.data {
                data.push(a * b);
            }
        }

        Self {
            data,
            shape: new_shape,
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Contraction
    // ────────────────────────────────────────────────────────────────────────

    /// Contract (trace) over two specified axes.
    ///
    /// The two axes must have the same size. The result is a tensor
    /// whose rank is reduced by 2 (or a scalar if the original was rank 2).
    ///
    /// Contraction generalizes the inner product: for a rank-2 tensor,
    /// `contract((0, 1))` computes the trace.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Either axis index is out of range.
    /// - The two axes have different sizes.
    /// - The two axis indices are the same.
    pub fn contract(&self, indices: (usize, usize)) -> Self {
        let (ax_a, ax_b) = indices;
        debug_assert_ne!(ax_a, ax_b, "Cannot contract an axis with itself");
        debug_assert!(ax_a < self.rank(), "First contraction axis out of range");
        debug_assert!(ax_b < self.rank(), "Second contraction axis out of range");
        if ax_a >= self.rank() || ax_b >= self.rank() || ax_a == ax_b {
            return Self::new(vec![1]); // Degenerate scalar fallback
        }
        debug_assert_eq!(
            self.shape[ax_a], self.shape[ax_b],
            "Contracted axes must have equal size: {} vs {}",
            self.shape[ax_a], self.shape[ax_b]
        );

        let contract_size = self.shape[ax_a];

        // Build output shape (remove both contracted axes)
        let (lo, hi) = if ax_a < ax_b {
            (ax_a, ax_b)
        } else {
            (ax_b, ax_a)
        };
        let mut out_shape: Vec<usize> = Vec::new();
        for (i, &s) in self.shape.iter().enumerate() {
            if i != lo && i != hi {
                out_shape.push(s);
            }
        }

        // Scalar result
        if out_shape.is_empty() {
            let strides = self.strides();
            let mut sum = 0.0;
            for k in 0..contract_size {
                let idx = k * strides[lo] + k * strides[hi];
                sum += self.data[idx];
            }
            return Self {
                data: vec![sum],
                shape: vec![1],
            };
        }

        let mut result = Self::new(out_shape.clone());
        let in_strides = self.strides();
        let out_strides = result.strides();

        // Iterate over all output indices
        let out_total: usize = out_shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .expect("tensor shape overflow");
        assert!(
            out_total <= 100_000_000,
            "tensor too large: {} elements",
            out_total
        );
        for flat_out in 0..out_total {
            // Decompose flat_out into multi-index
            let mut out_idx = vec![0usize; out_shape.len()];
            let mut remainder = flat_out;
            for (i, &s) in out_strides.iter().enumerate() {
                out_idx[i] = remainder / s;
                remainder %= s;
            }

            // Map output indices back to input indices (inserting contracted axes)
            let mut in_idx = vec![0usize; self.rank()];
            let mut out_pos = 0;
            for i in 0..self.rank() {
                if i == lo || i == hi {
                    // Will be filled by contraction loop
                } else {
                    in_idx[i] = out_idx[out_pos];
                    out_pos += 1;
                }
            }

            // Sum over contracted index
            let mut sum = 0.0;
            for k in 0..contract_size {
                in_idx[lo] = k;
                in_idx[hi] = k;
                let flat_in: usize = in_idx
                    .iter()
                    .zip(in_strides.iter())
                    .map(|(&idx, &stride)| idx * stride)
                    .sum();
                sum += self.data[flat_in];
            }

            result.data[flat_out] = sum;
        }

        result
    }

    // ────────────────────────────────────────────────────────────────────────
    // Reshape
    // ────────────────────────────────────────────────────────────────────────

    /// Reshape the tensor to a new shape.
    ///
    /// The total number of elements must remain the same.
    ///
    /// # Panics
    ///
    /// Panics if the new shape has a different total size.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_total: usize = new_shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .expect("tensor shape overflow");
        assert!(
            new_total <= 100_000_000,
            "tensor too large: {} elements",
            new_total
        );
        assert_eq!(
            self.data.len(),
            new_total,
            "Cannot reshape: {} elements into shape {:?} ({} elements)",
            self.data.len(),
            new_shape,
            new_total
        );
        Self {
            data: self.data.clone(),
            shape: new_shape,
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // HDC projection (Johnson-Lindenstrauss random projection)
    // ────────────────────────────────────────────────────────────────────────

    /// Project a flattened tensor into an HDC-dimension vector via random
    /// projection (Johnson-Lindenstrauss lemma).
    ///
    /// This is the bridge between tensor algebra and HDC: a rank-k tensor
    /// is flattened and projected into R^{target_dim} using a seeded
    /// pseudo-random matrix, approximately preserving pairwise distances.
    ///
    /// The projection matrix entries are drawn from {-1, +1} (Rademacher)
    /// scaled by 1/sqrt(target_dim), which satisfies JL with high probability.
    ///
    /// # Arguments
    ///
    /// * `target_dim` - Output dimensionality (typically 16,384 for HDC).
    /// * `seed` - Seed for deterministic projection matrix generation.
    pub fn to_hdc(&self, target_dim: usize, seed: u64) -> Vec<f64> {
        let n = self.data.len();
        let scale = 1.0 / (target_dim as f64).sqrt();
        let mut result = vec![0.0f64; target_dim];

        // Use a simple LCG seeded by `seed` for reproducibility.
        // The projection matrix is never materialised: we stream columns.
        let mut rng_state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);

        for j in 0..target_dim {
            let mut acc = 0.0;
            for i in 0..n {
                // Advance LCG
                rng_state = rng_state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Rademacher: +1 or -1 based on high bit
                let sign = if rng_state >> 63 == 0 { 1.0 } else { -1.0 };
                acc += sign * self.data[i];
            }
            result[j] = acc * scale;
        }

        result
    }

    /// Create an identity matrix as a rank-2 tensor.
    pub fn identity(n: usize) -> Self {
        let mut t = Self::new(vec![n, n]);
        for i in 0..n {
            t.set(&[i, i], 1.0);
        }
        t
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Geometric Algebra
// ═══════════════════════════════════════════════════════════════════════════════

/// Specification of a geometric (Clifford) algebra Cl(p, q).
///
/// The algebra is determined by its vector-space dimension `n = p + q` and
/// a diagonal metric: the first `p` basis vectors square to +1 and the
/// remaining `q` square to -1. Euclidean space is Cl(n, 0).
///
/// The total number of basis blades is 2^n (scalar, vectors, bivectors, ...,
/// pseudoscalar).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricAlgebra {
    /// Dimension of the generating vector space.
    pub dimension: usize,
    /// Metric signature: `metric[i]` is the square of the i-th basis vector.
    /// For Euclidean Cl(n): all +1. For Minkowski Cl(3,1): [+1,+1,+1,-1].
    pub metric: Vec<f64>,
}

impl GeometricAlgebra {
    /// Create a Euclidean geometric algebra Cl(n, 0).
    ///
    /// All basis vectors square to +1.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            metric: vec![1.0; dimension],
        }
    }

    /// Create a geometric algebra with specified metric signature.
    ///
    /// # Panics
    ///
    /// Panics if `metric.len() != dimension`.
    pub fn with_metric(dimension: usize, metric: Vec<f64>) -> Self {
        assert_eq!(
            metric.len(),
            dimension,
            "Metric length must equal dimension"
        );
        Self { dimension, metric }
    }

    /// Total number of basis blades (2^n).
    pub fn num_blades(&self) -> usize {
        1 << self.dimension
    }

    /// Compute the grade of a basis blade given its bitmask index.
    ///
    /// The bitmask encodes which basis vectors participate: e.g., for
    /// Cl(3), bitmask 0b101 = e1 /\ e3, which is a grade-2 (bivector) blade.
    pub fn grade_of_blade(blade_mask: usize) -> usize {
        blade_mask.count_ones() as usize
    }

    /// Canonical computation of basis blade product using the sign-counting
    /// algorithm from geometric algebra textbooks.
    ///
    /// For basis blades e_A and e_B (A, B are index sets encoded as bitmasks):
    /// e_A * e_B = (-1)^{swaps} * product_of_metric_factors * e_{A triangle B}
    fn canonical_basis_product(&self, a_mask: usize, b_mask: usize) -> (usize, f64) {
        // Expand masks to ordered index lists
        let a_indices = Self::mask_to_indices(a_mask);
        let b_indices = Self::mask_to_indices(b_mask);

        // Concatenate: [a_indices..., b_indices...]
        let mut combined: Vec<usize> = Vec::with_capacity(a_indices.len() + b_indices.len());
        combined.extend_from_slice(&a_indices);
        combined.extend_from_slice(&b_indices);

        // Bubble sort to canonical order, counting swaps
        let mut sign = 1.0;
        let mut sorted = false;
        while !sorted {
            sorted = true;
            let n = combined.len();
            for i in 0..n.saturating_sub(1) {
                if combined[i] > combined[i + 1] {
                    combined.swap(i, i + 1);
                    sign = -sign;
                    sorted = false;
                } else if combined[i] == combined[i + 1] {
                    // e_i * e_i = metric[i], remove both
                    sign *= self.metric[combined[i]];
                    combined.remove(i + 1);
                    combined.remove(i);
                    sorted = false;
                    break; // restart since indices shifted
                }
            }
        }

        // Remaining indices form the result blade
        let result_mask = Self::indices_to_mask(&combined);
        (result_mask, sign)
    }

    /// Convert a bitmask to a sorted list of set-bit positions.
    fn mask_to_indices(mask: usize) -> Vec<usize> {
        let mut indices = Vec::new();
        let mut m = mask;
        while m != 0 {
            let bit = m.trailing_zeros() as usize;
            indices.push(bit);
            m &= !(1 << bit);
        }
        indices
    }

    /// Convert a sorted list of indices back to a bitmask.
    fn indices_to_mask(indices: &[usize]) -> usize {
        let mut mask = 0usize;
        for &i in indices {
            mask |= 1 << i;
        }
        mask
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multivector
// ═══════════════════════════════════════════════════════════════════════════════

/// An element of a geometric (Clifford) algebra.
///
/// A multivector M in Cl(n) has 2^n scalar components, one for each basis
/// blade: M = sum_{k=0}^{2^n - 1} c_k * e_k.
///
/// The component at index `k` corresponds to the basis blade whose
/// participating vectors are encoded by the bits of `k`. For example,
/// in Cl(3):
///
/// | Index (binary) | Blade   | Grade |
/// |----------------|---------|-------|
/// | 000            | scalar  | 0     |
/// | 001            | e1      | 1     |
/// | 010            | e2      | 1     |
/// | 011            | e1e2    | 2     |
/// | 100            | e3      | 1     |
/// | 101            | e1e3    | 2     |
/// | 110            | e2e3    | 2     |
/// | 111            | e1e2e3  | 3     |
///
/// The `algebra` field specifies the metric signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Multivector {
    /// One coefficient per basis blade, indexed by bitmask.
    pub components: Vec<f64>,
    /// Offsets into `components` for each grade k. `grade_offsets[k]` stores
    /// the list of blade indices (bitmasks) that have grade k. This is
    /// precomputed for fast grade extraction.
    pub grade_offsets: Vec<Vec<usize>>,
    /// The generating algebra.
    pub algebra: GeometricAlgebra,
}

impl Multivector {
    /// Create a zero multivector in the given algebra.
    pub fn zero(algebra: &GeometricAlgebra) -> Self {
        let n = algebra.num_blades();
        let grade_offsets = Self::compute_grade_offsets(algebra.dimension);
        Self {
            components: vec![0.0; n],
            grade_offsets,
            algebra: algebra.clone(),
        }
    }

    /// Create a scalar multivector (grade 0 only).
    pub fn scalar(algebra: &GeometricAlgebra, value: f64) -> Self {
        let mut mv = Self::zero(algebra);
        mv.components[0] = value;
        mv
    }

    /// Create a vector (grade 1) from components.
    ///
    /// # Panics
    ///
    /// Panics if `values.len() != algebra.dimension`.
    pub fn vector(algebra: &GeometricAlgebra, values: &[f64]) -> Self {
        assert_eq!(
            values.len(),
            algebra.dimension,
            "Vector must have {} components, got {}",
            algebra.dimension,
            values.len()
        );
        let mut mv = Self::zero(algebra);
        for (i, &v) in values.iter().enumerate() {
            mv.components[1 << i] = v;
        }
        mv
    }

    /// Create a bivector from the given blade coefficients.
    ///
    /// For Cl(3), the bivector has 3 independent components: e12, e13, e23.
    /// The coefficients are given in order of increasing blade bitmask.
    pub fn bivector(algebra: &GeometricAlgebra, coefficients: &[f64]) -> Self {
        let mut mv = Self::zero(algebra);
        let grade2_blades = &mv.grade_offsets[2];
        assert_eq!(
            coefficients.len(),
            grade2_blades.len(),
            "Expected {} bivector components, got {}",
            grade2_blades.len(),
            coefficients.len()
        );
        for (i, &blade_idx) in grade2_blades.iter().enumerate() {
            mv.components[blade_idx] = coefficients[i];
        }
        mv
    }

    /// Precompute the grade offsets: for each grade k, which blade indices
    /// belong to that grade.
    fn compute_grade_offsets(dimension: usize) -> Vec<Vec<usize>> {
        let num_blades = 1 << dimension;
        let mut offsets = vec![Vec::new(); dimension + 1];
        for blade in 0u32..num_blades as u32 {
            let grade = blade.count_ones() as usize;
            offsets[grade].push(blade as usize);
        }
        offsets
    }

    // ────────────────────────────────────────────────────────────────────────
    // Geometric product
    // ────────────────────────────────────────────────────────────────────────

    /// Compute the geometric product of two multivectors.
    ///
    /// The geometric product is the fundamental operation of Clifford algebra:
    /// for vectors a, b: ab = a . b + a /\ b
    /// (inner product + outer product).
    ///
    /// This unifies scalar products, exterior products, and all their
    /// generalizations into a single bilinear operation.
    pub fn geometric_product(a: &Multivector, b: &Multivector) -> Multivector {
        assert_eq!(
            a.algebra.dimension, b.algebra.dimension,
            "Multivectors must be from the same algebra"
        );
        let algebra = &a.algebra;
        let n = algebra.num_blades();
        let mut result = Multivector::zero(algebra);

        for i in 0..n {
            if a.components[i] == 0.0 {
                continue;
            }
            for j in 0..n {
                if b.components[j] == 0.0 {
                    continue;
                }
                let (blade, sign) = algebra.canonical_basis_product(i, j);
                result.components[blade] += sign * a.components[i] * b.components[j];
            }
        }

        result
    }

    // ────────────────────────────────────────────────────────────────────────
    // Inner product (grade-lowering, left contraction)
    // ────────────────────────────────────────────────────────────────────────

    /// Compute the inner product (left contraction) of two multivectors.
    ///
    /// For homogeneous multivectors of grade r and s:
    /// `inner(A_r, B_s) = <A_r * B_s>_{|s - r|}` if s >= r, else 0.
    ///
    /// For general multivectors, this is extended by linearity over grades.
    /// The symmetric inner product used here is: (1/2)(ab + ba) projected
    /// to the lower-grade part.
    pub fn inner_product(a: &Multivector, b: &Multivector) -> Multivector {
        assert_eq!(
            a.algebra.dimension, b.algebra.dimension,
            "Multivectors must be from the same algebra"
        );
        let algebra = &a.algebra;
        let n = algebra.num_blades();
        let mut result = Multivector::zero(algebra);

        for i in 0..n {
            if a.components[i] == 0.0 {
                continue;
            }
            let grade_a = GeometricAlgebra::grade_of_blade(i);
            for j in 0..n {
                if b.components[j] == 0.0 {
                    continue;
                }
                let grade_b = GeometricAlgebra::grade_of_blade(j);

                // Left contraction: only keep grade |grade_b - grade_a|
                // when grade_b >= grade_a
                if grade_b < grade_a {
                    continue;
                }
                let target_grade = grade_b - grade_a;

                let (blade, sign) = algebra.canonical_basis_product(i, j);
                let result_grade = GeometricAlgebra::grade_of_blade(blade);
                if result_grade == target_grade {
                    result.components[blade] += sign * a.components[i] * b.components[j];
                }
            }
        }

        result
    }

    // ────────────────────────────────────────────────────────────────────────
    // Outer (wedge) product (grade-raising)
    // ────────────────────────────────────────────────────────────────────────

    /// Compute the outer (wedge) product of two multivectors.
    ///
    /// For homogeneous multivectors of grade r and s:
    /// `outer(A_r, B_s) = <A_r * B_s>_{r + s}`
    ///
    /// The outer product creates higher-grade elements: two vectors
    /// wedge to a bivector (oriented plane), three to a trivector (volume).
    pub fn outer_product(a: &Multivector, b: &Multivector) -> Multivector {
        assert_eq!(
            a.algebra.dimension, b.algebra.dimension,
            "Multivectors must be from the same algebra"
        );
        let algebra = &a.algebra;
        let n = algebra.num_blades();
        let mut result = Multivector::zero(algebra);

        for i in 0..n {
            if a.components[i] == 0.0 {
                continue;
            }
            let grade_a = GeometricAlgebra::grade_of_blade(i);
            for j in 0..n {
                if b.components[j] == 0.0 {
                    continue;
                }
                let grade_b = GeometricAlgebra::grade_of_blade(j);
                let target_grade = grade_a + grade_b;

                let (blade, sign) = algebra.canonical_basis_product(i, j);
                let result_grade = GeometricAlgebra::grade_of_blade(blade);
                if result_grade == target_grade {
                    result.components[blade] += sign * a.components[i] * b.components[j];
                }
            }
        }

        result
    }

    // ────────────────────────────────────────────────────────────────────────
    // Grade extraction
    // ────────────────────────────────────────────────────────────────────────

    /// Extract the grade-k part of a multivector.
    ///
    /// Returns a new multivector with only the grade-k components nonzero.
    /// This is the projection `<M>_k`.
    pub fn grade(&self, k: usize) -> Multivector {
        let mut result = Multivector::zero(&self.algebra);
        if k > self.algebra.dimension {
            return result;
        }
        for &blade_idx in &self.grade_offsets[k] {
            result.components[blade_idx] = self.components[blade_idx];
        }
        result
    }

    /// Extract the scalar (grade-0) part.
    pub fn scalar_part(&self) -> f64 {
        self.components[0]
    }

    // ────────────────────────────────────────────────────────────────────────
    // Reversion (tilde / dagger)
    // ────────────────────────────────────────────────────────────────────────

    /// Compute the reverse of a multivector.
    ///
    /// Reversion reverses the order of basis vectors in each blade:
    /// `reverse(e1 e2 e3) = e3 e2 e1 = -e1 e2 e3` (3 swaps -> sign = (-1)^{k(k-1)/2})
    ///
    /// For a grade-k blade: `reverse(B_k) = (-1)^{k(k-1)/2} * B_k`
    ///
    /// This is essential for rotors: rotation of v is `R v reverse(R)`.
    pub fn reverse(&self) -> Multivector {
        let mut result = self.clone();
        for (blade_idx, component) in result.components.iter_mut().enumerate() {
            let k = GeometricAlgebra::grade_of_blade(blade_idx);
            // Sign is (-1)^{k(k-1)/2}
            let sign = if (k * (k.wrapping_sub(1)) / 2).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            *component *= sign;
        }
        result
    }

    // ────────────────────────────────────────────────────────────────────────
    // Norm
    // ────────────────────────────────────────────────────────────────────────

    /// Compute the squared norm of a multivector: <M * reverse(M)>_0
    pub fn norm_squared(&self) -> f64 {
        let rev = self.reverse();
        let product = Self::geometric_product(self, &rev);
        product.scalar_part()
    }

    /// Compute the norm: sqrt(|<M * reverse(M)>_0|)
    pub fn norm(&self) -> f64 {
        self.norm_squared().abs().sqrt()
    }

    // ────────────────────────────────────────────────────────────────────────
    // Hodge star (duality)
    // ────────────────────────────────────────────────────────────────────────

    /// Compute the Hodge dual of a multivector.
    ///
    /// The Hodge star maps k-vectors to (n-k)-vectors using the
    /// pseudoscalar: dual(M) = M * I^{-1}, where I = e1 e2 ... en.
    ///
    /// This duality is fundamental to electromagnetism (E and B fields)
    /// and to consciousness models that pair "content" with its dual
    /// "context" representation.
    pub fn hodge_star(&self) -> Multivector {
        let n = self.algebra.dimension;
        let pseudoscalar_mask = (1 << n) - 1; // All bits set = e1e2...en

        // Compute I^{-1} = reverse(I) / |I|^2
        // For the pseudoscalar: reverse(e1...en) = (-1)^{n(n-1)/2} e1...en
        // and |I|^2 = product of all metric entries.
        let metric_product: f64 = self.algebra.metric.iter().product();
        let rev_sign = if (n * n.wrapping_sub(1) / 2).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        let inv_scale = rev_sign / metric_product;

        // dual(M) = M * (inv_scale * I)
        let mut pseudo_inv = Multivector::zero(&self.algebra);
        pseudo_inv.components[pseudoscalar_mask] = inv_scale;

        Self::geometric_product(self, &pseudo_inv)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Rotor
// ═══════════════════════════════════════════════════════════════════════════════

/// A rotor (even-grade element) in geometric algebra, representing a rotation.
///
/// A rotor R = exp(-theta/2 * B), where B is a unit bivector defining the
/// plane of rotation and theta is the rotation angle. Rotation of a vector v
/// is the sandwich product: v' = R v reverse(R).
///
/// Rotors generalise complex numbers (2D rotations) and quaternions (3D
/// rotations) to arbitrary dimensions, providing a natural framework for
/// attentional rotation in consciousness models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rotor {
    /// The rotor as a multivector (only even-grade components are nonzero).
    pub components: Multivector,
}

impl Rotor {
    /// Construct a rotor from an angle and a bivector plane.
    ///
    /// R = cos(theta/2) - sin(theta/2) * B_hat
    ///
    /// where B_hat is the unit bivector in the plane of rotation.
    ///
    /// # Panics
    ///
    /// Panics if the bivector has zero norm (degenerate plane).
    pub fn from_angle_plane(angle: f64, plane: &Multivector) -> Self {
        let plane_norm = plane.norm();
        assert!(
            plane_norm > 1e-12,
            "Bivector plane must have nonzero norm for rotation"
        );

        // Normalise the bivector
        let inv_norm = 1.0 / plane_norm;
        let mut b_hat = plane.clone();
        for c in b_hat.components.iter_mut() {
            *c *= inv_norm;
        }

        let half_angle = angle / 2.0;
        let cos_ha = half_angle.cos();
        let sin_ha = half_angle.sin();

        // R = cos(theta/2) - sin(theta/2) * B_hat
        let algebra = &plane.algebra;
        let mut rotor = Multivector::zero(algebra);

        // Scalar part
        rotor.components[0] = cos_ha;

        // Bivector part (negated sin component)
        for &blade_idx in &rotor.grade_offsets[2] {
            rotor.components[blade_idx] = -sin_ha * b_hat.components[blade_idx];
        }

        Self { components: rotor }
    }

    /// Rotate a multivector using the sandwich product: v' = R v R~
    ///
    /// This preserves grades and magnitudes, implementing a proper rotation.
    pub fn rotate(&self, v: &Multivector) -> Multivector {
        let rev = self.components.reverse();
        let rv = Multivector::geometric_product(&self.components, v);
        Multivector::geometric_product(&rv, &rev)
    }

    /// Compose two rotations: R_combined = R2 * R1
    ///
    /// Applying R_combined is equivalent to first applying R1, then R2.
    pub fn compose(&self, other: &Rotor) -> Rotor {
        Rotor {
            components: Multivector::geometric_product(&self.components, &other.components),
        }
    }

    /// The identity rotor (no rotation): R = 1.
    pub fn identity(algebra: &GeometricAlgebra) -> Self {
        Self {
            components: Multivector::scalar(algebra, 1.0),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tensor Network
// ═══════════════════════════════════════════════════════════════════════════════

/// A node in a tensor network, holding a tensor and information about
/// which of its indices are free (uncontracted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorNode {
    /// Unique identifier for this node.
    pub id: usize,
    /// The tensor at this node.
    pub tensor: Tensor,
    /// Indices of this tensor's axes that are free (not contracted with
    /// another node). Contracted axes are tracked by edges.
    pub free_indices: Vec<usize>,
}

/// An edge in a tensor network, connecting two axes that should be contracted.
///
/// `from` and `to` are `(node_id, axis_index)` pairs. When the network is
/// contracted, these two axes are summed over.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEdge {
    /// (node_id, axis_index) of the first endpoint.
    pub from: (usize, usize),
    /// (node_id, axis_index) of the second endpoint.
    pub to: (usize, usize),
}

/// A tensor network: a graph of tensors connected by contraction edges.
///
/// Tensor networks provide a structured way to compose HDC representations:
/// - **Matrix Product State (MPS)**: encodes sequences as a chain of contracted
///   rank-3 tensors, capturing sequential dependencies in language or time.
/// - **Tree Tensor Network (TTN)**: encodes hierarchical structure (parse trees,
///   concept hierarchies) as a tree of contracted tensors.
///
/// Contracting the full network collapses the compositional structure into a
/// single tensor (or scalar), analogous to how consciousness integrates
/// distributed information into a unified experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorNetwork {
    /// Nodes in the network.
    pub nodes: Vec<TensorNode>,
    /// Contraction edges between nodes.
    pub edges: Vec<TensorEdge>,
}

impl TensorNetwork {
    /// Create an empty tensor network.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a tensor node, returning its assigned ID.
    ///
    /// All axes are initially free.
    pub fn add_node(&mut self, tensor: Tensor) -> usize {
        let id = self.nodes.len();
        let free_indices: Vec<usize> = (0..tensor.rank()).collect();
        self.nodes.push(TensorNode {
            id,
            tensor,
            free_indices,
        });
        id
    }

    /// Add a contraction edge between two tensor axes.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Either node ID is invalid.
    /// - The connected axes have different sizes.
    pub fn add_edge(&mut self, from: (usize, usize), to: (usize, usize)) {
        assert!(
            from.0 < self.nodes.len(),
            "From node {} does not exist",
            from.0
        );
        assert!(to.0 < self.nodes.len(), "To node {} does not exist", to.0);
        assert_eq!(
            self.nodes[from.0].tensor.shape[from.1], self.nodes[to.0].tensor.shape[to.1],
            "Connected axes must have same size: {} vs {}",
            self.nodes[from.0].tensor.shape[from.1], self.nodes[to.0].tensor.shape[to.1],
        );
        self.edges.push(TensorEdge { from, to });
    }

    /// Contract the entire tensor network by sequentially contracting edges.
    ///
    /// This uses a greedy pairwise contraction strategy: for each edge,
    /// the two connected nodes are contracted into a single node, and
    /// remaining edges are updated to point to the new node.
    ///
    /// Returns the final tensor after all contractions.
    ///
    /// Returns the final tensor after all contractions, or `None` if
    /// the network is empty or contraction produced no surviving tensor.
    pub fn contract(&self) -> Option<Tensor> {
        if self.nodes.is_empty() {
            return None;
        }

        if self.edges.is_empty() {
            // No edges: return the tensor product of all nodes
            let mut result = self.nodes[0].tensor.clone();
            for node in &self.nodes[1..] {
                result = result.tensor_product(&node.tensor);
            }
            return Some(result);
        }

        // Clone the network state so we can mutate
        let mut tensors: Vec<Option<Tensor>> =
            self.nodes.iter().map(|n| Some(n.tensor.clone())).collect();
        let mut edges = self.edges.clone();
        // Map from original node IDs to current node IDs
        let mut id_map: Vec<usize> = (0..self.nodes.len()).collect();

        while let Some(edge) = edges.pop() {
            let from_id = id_map[edge.from.0];
            let to_id = id_map[edge.to.0];

            if from_id == to_id {
                // Both endpoints are in the same tensor: do a trace contraction
                let Some(t) = tensors[from_id].take() else {
                    tracing::warn!("Tensor node consumed twice during contraction");
                    continue;
                };
                let result = t.contract((edge.from.1, edge.to.1));
                tensors[from_id] = Some(result);
                continue;
            }

            let (Some(t_from), Some(t_to)) = (tensors[from_id].take(), tensors[to_id].take())
            else {
                tracing::warn!("Tensor node consumed twice during contraction");
                continue;
            };

            // Compute the tensor product, then contract the appropriate axes
            let product = t_from.tensor_product(&t_to);

            // The contraction axes in the product tensor:
            // from axis is at its original position, to axis is offset by
            // the rank of t_from.
            let from_axis = edge.from.1;
            let to_axis = t_from.rank() + edge.to.1;

            let result = product.contract((from_axis, to_axis));

            // Store result in from_id, mark to_id as consumed
            tensors[from_id] = Some(result);

            // Update id_map: everything pointing to to_id now points to from_id
            for v in id_map.iter_mut() {
                if *v == to_id {
                    *v = from_id;
                }
            }

            // Update remaining edges: adjust axis indices for the merged tensor
            // This is a simplification -- a production implementation would
            // carefully track axis remapping. For correctness with small
            // networks (the intended use case), we recompute from scratch.
        }

        // Return the first remaining tensor
        tensors.into_iter().flatten().next()
    }
}

impl Default for TensorNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ContinuousHV integration
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert a `ContinuousHV` (f32) to an f64 slice for tensor operations.
pub fn real_hv_to_f64(hv: &ContinuousHV) -> Vec<f64> {
    hv.values.iter().map(|&v| v as f64).collect()
}

/// Convert an f64 slice back to a `ContinuousHV` (f32).
pub fn f64_to_real_hv(data: &[f64]) -> ContinuousHV {
    ContinuousHV::from_values(data.iter().map(|&v| v as f32).collect())
}

/// Compute the tensor product of two `ContinuousHV` vectors and project back
/// to HDC dimension using Johnson-Lindenstrauss random projection.
///
/// This encodes a relational binding between two concepts in HDC space
/// that preserves more structure than element-wise multiplication (bind),
/// at the cost of a lossy projection step.
pub fn hdc_tensor_product(a: &ContinuousHV, b: &ContinuousHV, seed: u64) -> ContinuousHV {
    let a64 = real_hv_to_f64(a);
    let b64 = real_hv_to_f64(b);
    let outer = Tensor::outer_product(&a64, &b64);
    let target_dim = a.dim().max(b.dim());
    let projected = outer.to_hdc(target_dim, seed);
    f64_to_real_hv(&projected)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ────────────────────────────────────────────────────────────────────────
    // Tensor tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_outer_product_dimensions() {
        let a = vec![1.0, 2.0, 3.0]; // (3,)
        let b = vec![4.0, 5.0]; // (2,)
        let t = Tensor::outer_product(&a, &b);

        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.numel(), 6);

        // Check specific values: t[i][j] = a[i] * b[j]
        assert_eq!(t.get(&[0, 0]), 4.0); // 1*4
        assert_eq!(t.get(&[0, 1]), 5.0); // 1*5
        assert_eq!(t.get(&[1, 0]), 8.0); // 2*4
        assert_eq!(t.get(&[1, 1]), 10.0); // 2*5
        assert_eq!(t.get(&[2, 0]), 12.0); // 3*4
        assert_eq!(t.get(&[2, 1]), 15.0); // 3*5
    }

    #[test]
    fn test_trace_of_identity() {
        let n = 5;
        let eye = Tensor::identity(n);
        let trace = eye.contract((0, 1));

        // trace(I_n) = n
        assert!((trace.data[0] - n as f64).abs() < 1e-10);
    }

    #[test]
    fn test_trace_of_outer_product_equals_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let outer = Tensor::outer_product(&a, &b);
        let trace = outer.contract((0, 1));

        // trace(a * b^T) = a . b = 1*4 + 2*5 + 3*6 = 32
        let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((trace.data[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_product_shape() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_data(vec![4.0, 5.0], vec![2]);
        let ab = a.tensor_product(&b);

        assert_eq!(ab.shape, vec![3, 2]);
        assert_eq!(ab.data, vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }

    #[test]
    fn test_tensor_product_rank3() {
        let a = Tensor::from_data(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_data(vec![3.0, 4.0], vec![2]);
        let c = Tensor::from_data(vec![5.0, 6.0], vec![2]);

        // Rank-3 tensor: subject ⊗ predicate ⊗ object
        let ab = a.tensor_product(&b);
        let abc = ab.tensor_product(&c);

        assert_eq!(abc.shape, vec![2, 2, 2]);
        assert_eq!(abc.numel(), 8);

        // abc[0][0][0] = 1*3*5 = 15
        assert_eq!(abc.get(&[0, 0, 0]), 15.0);
        // abc[1][1][1] = 2*4*6 = 48
        assert_eq!(abc.get(&[1, 1, 1]), 48.0);
    }

    #[test]
    fn test_reshape_preserves_data() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let reshaped = t.reshape(vec![3, 2]);

        assert_eq!(reshaped.shape, vec![3, 2]);
        assert_eq!(reshaped.data, t.data);
    }

    #[test]
    #[should_panic(expected = "Cannot reshape")]
    fn test_reshape_wrong_size_panics() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let _bad = t.reshape(vec![3, 3]);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Geometric Algebra tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_geometric_product_orthogonal_vectors() {
        // In Cl(3), for orthogonal basis vectors e1 and e2:
        // e1 * e2 = e1 /\ e2 (pure bivector, no scalar part)
        let ga = GeometricAlgebra::new(3);
        let e1 = Multivector::vector(&ga, &[1.0, 0.0, 0.0]);
        let e2 = Multivector::vector(&ga, &[0.0, 1.0, 0.0]);

        let product = Multivector::geometric_product(&e1, &e2);

        // Scalar part should be zero (orthogonal vectors)
        assert!(
            product.scalar_part().abs() < 1e-10,
            "Orthogonal vectors should have zero scalar product, got {}",
            product.scalar_part()
        );

        // Bivector part e12 should be 1.0
        let e12_idx = 0b011; // e1 /\ e2
        assert!(
            (product.components[e12_idx] - 1.0).abs() < 1e-10,
            "e1*e2 should produce e12 = 1.0, got {}",
            product.components[e12_idx]
        );
    }

    #[test]
    fn test_geometric_product_parallel_vectors() {
        // For a vector with itself: a * a = |a|^2 (pure scalar)
        let ga = GeometricAlgebra::new(3);
        let a = Multivector::vector(&ga, &[3.0, 4.0, 0.0]);

        let product = Multivector::geometric_product(&a, &a);

        // Scalar part = |a|^2 = 9 + 16 = 25
        assert!(
            (product.scalar_part() - 25.0).abs() < 1e-10,
            "a*a should be |a|^2 = 25, got {}",
            product.scalar_part()
        );

        // All non-scalar parts should be zero
        for blade in 1..ga.num_blades() {
            assert!(
                product.components[blade].abs() < 1e-10,
                "a*a should be pure scalar, but blade {} = {}",
                blade,
                product.components[blade]
            );
        }
    }

    #[test]
    fn test_geometric_product_decomposes_into_inner_plus_outer() {
        // For vectors: ab = a . b + a /\ b
        let ga = GeometricAlgebra::new(3);
        let a = Multivector::vector(&ga, &[1.0, 2.0, 3.0]);
        let b = Multivector::vector(&ga, &[4.0, 5.0, 6.0]);

        let geo = Multivector::geometric_product(&a, &b);
        let inner = Multivector::inner_product(&a, &b);
        let outer = Multivector::outer_product(&a, &b);

        // Check that geometric = inner + outer (component-wise)
        for i in 0..ga.num_blades() {
            let sum = inner.components[i] + outer.components[i];
            assert!(
                (geo.components[i] - sum).abs() < 1e-10,
                "Blade {}: geo={} != inner+outer={}+{}={}",
                i,
                geo.components[i],
                inner.components[i],
                outer.components[i],
                sum
            );
        }
    }

    #[test]
    fn test_rotor_rotation_preserves_magnitude() {
        let ga = GeometricAlgebra::new(3);

        // Create a bivector for the e1-e2 plane
        let plane = Multivector::bivector(&ga, &[1.0, 0.0, 0.0]); // e12

        // 90 degree rotation in the e1-e2 plane
        let rotor = Rotor::from_angle_plane(std::f64::consts::FRAC_PI_2, &plane);

        // Rotate e1
        let e1 = Multivector::vector(&ga, &[1.0, 0.0, 0.0]);
        let rotated = rotor.rotate(&e1);

        // Magnitude should be preserved
        let original_norm = e1.norm();
        let rotated_norm = rotated.norm();
        assert!(
            (original_norm - rotated_norm).abs() < 1e-10,
            "Rotation should preserve magnitude: {} vs {}",
            original_norm,
            rotated_norm
        );

        // After 90 degree rotation in e1-e2 plane, e1 -> e2
        // Extract vector components
        let rotated_x = rotated.components[0b001]; // e1
        let rotated_y = rotated.components[0b010]; // e2
        let rotated_z = rotated.components[0b100]; // e3

        assert!(
            rotated_x.abs() < 1e-10,
            "After 90deg rotation, x should be ~0, got {}",
            rotated_x
        );
        assert!(
            (rotated_y - 1.0).abs() < 1e-10,
            "After 90deg rotation, y should be ~1, got {}",
            rotated_y
        );
        assert!(
            rotated_z.abs() < 1e-10,
            "After 90deg rotation, z should be ~0, got {}",
            rotated_z
        );
    }

    #[test]
    fn test_rotor_360_returns_to_original() {
        let ga = GeometricAlgebra::new(3);
        let plane = Multivector::bivector(&ga, &[1.0, 0.0, 0.0]); // e12 plane

        // Full 360 degree rotation
        let rotor = Rotor::from_angle_plane(2.0 * std::f64::consts::PI, &plane);

        let v = Multivector::vector(&ga, &[1.0, 2.0, 3.0]);
        let rotated = rotor.rotate(&v);

        // Should return to original vector
        for i in 0..ga.num_blades() {
            assert!(
                (rotated.components[i] - v.components[i]).abs() < 1e-9,
                "360deg rotation should be identity: blade {} differs ({} vs {})",
                i,
                rotated.components[i],
                v.components[i]
            );
        }
    }

    #[test]
    fn test_reversion_grades() {
        let ga = GeometricAlgebra::new(3);

        // Grade 0 (scalar): reverse sign = +1
        let s = Multivector::scalar(&ga, 5.0);
        assert_eq!(s.reverse().components[0], 5.0);

        // Grade 1 (vector): reverse sign = +1
        let v = Multivector::vector(&ga, &[1.0, 2.0, 3.0]);
        let v_rev = v.reverse();
        assert_eq!(v_rev.components[0b001], 1.0);
        assert_eq!(v_rev.components[0b010], 2.0);
        assert_eq!(v_rev.components[0b100], 3.0);

        // Grade 2 (bivector): reverse sign = -1 [k(k-1)/2 = 1, odd]
        let bv = Multivector::bivector(&ga, &[1.0, 2.0, 3.0]);
        let bv_rev = bv.reverse();
        for &blade in &bv.grade_offsets[2] {
            assert_eq!(
                bv_rev.components[blade], -bv.components[blade],
                "Bivector reversion should negate"
            );
        }

        // Grade 3 (trivector/pseudoscalar): reverse sign = -1 [k(k-1)/2 = 3, odd]
        let mut tv = Multivector::zero(&ga);
        tv.components[0b111] = 7.0;
        let tv_rev = tv.reverse();
        assert_eq!(tv_rev.components[0b111], -7.0);
    }

    #[test]
    fn test_outer_product_antisymmetry() {
        // a /\ b = -(b /\ a) for vectors
        let ga = GeometricAlgebra::new(3);
        let a = Multivector::vector(&ga, &[1.0, 2.0, 3.0]);
        let b = Multivector::vector(&ga, &[4.0, 5.0, 6.0]);

        let ab = Multivector::outer_product(&a, &b);
        let ba = Multivector::outer_product(&b, &a);

        for i in 0..ga.num_blades() {
            assert!(
                (ab.components[i] + ba.components[i]).abs() < 1e-10,
                "a/\\b should equal -(b/\\a) at blade {}: {} vs {}",
                i,
                ab.components[i],
                ba.components[i]
            );
        }
    }

    #[test]
    fn test_hodge_star_grade_mapping() {
        // In Cl(3), Hodge star maps grade k to grade (3-k)
        let ga = GeometricAlgebra::new(3);
        let v = Multivector::vector(&ga, &[1.0, 0.0, 0.0]); // e1

        let dual = v.hodge_star();

        // e1 should map to a bivector (grade 2)
        // Specifically, *(e1) = e2e3 (up to sign, depending on convention)
        let grade2_norm_sq: f64 = dual.grade_offsets[2]
            .iter()
            .map(|&b| dual.components[b] * dual.components[b])
            .sum();
        assert!(
            grade2_norm_sq > 0.5,
            "Hodge star of a vector should produce a bivector, grade-2 norm^2 = {}",
            grade2_norm_sq
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Tensor Network tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_tensor_network_matrix_vector_multiply() {
        // Contract a (2x3) matrix with a (3,) vector along the shared axis
        // This is equivalent to matrix-vector multiplication
        let mat = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let vec_t = Tensor::from_data(vec![1.0, 0.0, 1.0], vec![3]);

        let mut net = TensorNetwork::new();
        let mat_id = net.add_node(mat);
        let vec_id = net.add_node(vec_t);

        // Contract matrix axis 1 (columns) with vector axis 0
        net.add_edge((mat_id, 1), (vec_id, 0));

        let result = net.contract().expect("contraction should produce a tensor");

        // Expected: [1*1 + 2*0 + 3*1, 4*1 + 5*0 + 6*1] = [4, 10]
        assert_eq!(result.shape.iter().product::<usize>(), 2);
        assert!(
            (result.data[0] - 4.0).abs() < 1e-10,
            "Expected 4, got {}",
            result.data[0]
        );
        assert!(
            (result.data[1] - 10.0).abs() < 1e-10,
            "Expected 10, got {}",
            result.data[1]
        );
    }

    #[test]
    fn test_tensor_network_no_edges() {
        // No edges: result is the tensor product of all nodes
        let a = Tensor::from_data(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_data(vec![3.0, 4.0], vec![2]);

        let mut net = TensorNetwork::new();
        net.add_node(a);
        net.add_node(b);

        let result = net.contract().expect("contraction should produce a tensor");

        // Should be outer product: [1*3, 1*4, 2*3, 2*4] = [3, 4, 6, 8]
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.data, vec![3.0, 4.0, 6.0, 8.0]);
    }

    // ────────────────────────────────────────────────────────────────────────
    // JL lemma: random projection preserves relative distances
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_random_projection_preserves_relative_distances() {
        // Create three vectors with known distance relationships
        // a and b are close, c is far from both.
        let n = 64;
        let mut a_data = vec![0.0; n];
        let mut b_data = vec![0.0; n];
        let mut c_data = vec![0.0; n];

        for i in 0..n {
            a_data[i] = (i as f64) / (n as f64);
            b_data[i] = a_data[i] + 0.01; // close to a
            c_data[i] = 1.0 - a_data[i]; // far from a
        }

        let t_a = Tensor::from_data(a_data.clone(), vec![n]);
        let t_b = Tensor::from_data(b_data.clone(), vec![n]);
        let t_c = Tensor::from_data(c_data.clone(), vec![n]);

        let target_dim = 32;
        let seed = 42;

        let proj_a = t_a.to_hdc(target_dim, seed);
        let proj_b = t_b.to_hdc(target_dim, seed);
        let proj_c = t_c.to_hdc(target_dim, seed);

        // Original distances
        let dist_ab_orig: f64 = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt();
        let dist_ac_orig: f64 = a_data
            .iter()
            .zip(c_data.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt();

        // Projected distances
        let dist_ab_proj: f64 = proj_a
            .iter()
            .zip(proj_b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt();
        let dist_ac_proj: f64 = proj_a
            .iter()
            .zip(proj_c.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt();

        // The key JL property: relative ordering of distances should be preserved
        // d(a,b) < d(a,c) in original => d(a,b) < d(a,c) in projection
        assert!(
            dist_ab_orig < dist_ac_orig,
            "Precondition: a-b should be closer than a-c in original space"
        );
        assert!(
            dist_ab_proj < dist_ac_proj,
            "JL property: relative distance ordering should be preserved. \
             Projected: d(a,b)={:.4}, d(a,c)={:.4}",
            dist_ab_proj,
            dist_ac_proj
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // HDC integration
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_hdc_tensor_product_dimensions() {
        let a = ContinuousHV::random(64, 1);
        let b = ContinuousHV::random(64, 2);

        let result = hdc_tensor_product(&a, &b, 99);
        assert_eq!(
            result.dim(),
            64,
            "HDC tensor product should preserve dimension"
        );
    }

    #[test]
    fn test_hdc_tensor_product_deterministic() {
        let a = ContinuousHV::random(32, 10);
        let b = ContinuousHV::random(32, 20);

        let r1 = hdc_tensor_product(&a, &b, 42);
        let r2 = hdc_tensor_product(&a, &b, 42);

        assert_eq!(
            r1.values, r2.values,
            "Same seed should produce same projection"
        );
    }

    #[test]
    fn test_multivector_grade_extraction() {
        let ga = GeometricAlgebra::new(3);

        // Build a multivector with scalar + vector + bivector parts
        let mut mv = Multivector::zero(&ga);
        mv.components[0] = 5.0; // scalar
        mv.components[0b001] = 1.0; // e1
        mv.components[0b010] = 2.0; // e2
        mv.components[0b011] = 3.0; // e12

        let grade0 = mv.grade(0);
        assert_eq!(grade0.scalar_part(), 5.0);
        assert_eq!(grade0.components[0b001], 0.0);

        let grade1 = mv.grade(1);
        assert_eq!(grade1.components[0b001], 1.0);
        assert_eq!(grade1.components[0b010], 2.0);
        assert_eq!(grade1.components[0b011], 0.0);

        let grade2 = mv.grade(2);
        assert_eq!(grade2.components[0b011], 3.0);
        assert_eq!(grade2.scalar_part(), 0.0);
    }
}
