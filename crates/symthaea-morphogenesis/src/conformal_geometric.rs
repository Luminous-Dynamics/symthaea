// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Conformal Geometric HDC.
//!
//! Implements Conformal Geometric Algebra (CGA) in hypervector space.
//! Embeds 3D coordinates into a 5D conformal space (R^4,1) projected into
//! 16,384D hypervectors.
//!
//! This enables native algebraic representation of biological growth (dilations)
//! and tissue deformation (rotors) as linear operations.

use symthaea_core::hdc::tensor_algebra::{GeometricAlgebra, Multivector, f64_to_real_hv};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Conformal Geometric Hypervector Engine.
pub struct ConformalGeometricEngine {
    pub dim: usize,
    pub ga: GeometricAlgebra,
    /// Basis blades in hypervector space (JL-projected).
    pub basis_hvs: Vec<ContinuousHV>,
    /// Pre-calculated n0 multivector (origin).
    pub n0: Multivector,
    /// Pre-calculated n_inf multivector (infinity).
    pub n_inf: Multivector,
}

impl ConformalGeometricEngine {
    /// Initialize the CGA engine for Cl(4, 1).
    pub fn new(dim: usize, seed: u64) -> Self {
        // Cl(4, 1): 3 Euclidean (e1, e2, e3), 1 Minkowski+ (e+), 1 Minkowski- (e-)
        // Metric: [1, 1, 1, 1, -1]
        let ga = GeometricAlgebra::with_metric(5, vec![1.0, 1.0, 1.0, 1.0, -1.0]);

        // Project each basis blade into HDC space
        let mut basis_hvs = Vec::with_capacity(ga.num_blades());
        for i in 0..ga.num_blades() {
            let mut mv = Multivector::zero(&ga);
            mv.components[i] = 1.0;
            // Project using the provided seed + blade index
            let projected = mv.to_hdc(dim, seed.wrapping_add(i as u64));
            basis_hvs.push(f64_to_real_hv(&projected));
        }

        // Define n0 and n_inf
        // e4 = e+, e5 = e-
        // n0 = 0.5 * (e- - e+)
        // n_inf = e- + e+
        let mut n0 = Multivector::zero(&ga);
        n0.components[1 << 4] = 0.5; // 0.5 * e_minus
        n0.components[1 << 3] = -0.5; // -0.5 * e_plus

        let mut n_inf = Multivector::zero(&ga);
        n_inf.components[1 << 4] = 1.0; // e_minus
        n_inf.components[1 << 3] = 1.0; // e_plus

        Self {
            dim,
            ga,
            basis_hvs,
            n0,
            n_inf,
        }
    }

    /// Embed a 3D point (x, y, z) into a conformal hypervector.
    ///
    /// P = x*e1 + y*e2 + z*e3 + 0.5*|x|^2*n_inf + n0
    pub fn embed_point(&self, x: f32, y: f32, z: f32) -> ContinuousHV {
        let mut p_mv = Multivector::zero(&self.ga);
        p_mv.components[1 << 0] = x as f64; // e1
        p_mv.components[1 << 1] = y as f64; // e2
        p_mv.components[1 << 2] = z as f64; // e3

        let norm_sq = (x * x + y * y + z * z) as f64;

        // Add 0.5*|x|^2*n_inf
        for i in 0..self.ga.num_blades() {
            p_mv.components[i] += 0.5 * norm_sq * self.n_inf.components[i];
            p_mv.components[i] += self.n_0_components(i);
        }

        self.multivector_to_hv(&p_mv)
    }

    fn n_0_components(&self, i: usize) -> f64 {
        self.n0.components[i]
    }

    /// Convert a multivector to its hypervector representation via linear combination
    /// of pre-projected basis blades.
    ///
    /// H = sum( a_i * B_i )
    pub fn multivector_to_hv(&self, mv: &Multivector) -> ContinuousHV {
        let mut values = vec![0.0f32; self.dim];
        for (i, &coeff) in mv.components.iter().enumerate() {
            if coeff.abs() < 1e-9 {
                continue;
            }
            let basis_hv = &self.basis_hvs[i];
            for (acc, &b) in values.iter_mut().zip(basis_hv.values.iter()) {
                *acc += (coeff as f32) * b;
            }
        }
        ContinuousHV::from_values(values).normalize()
    }

    /// Create a Dilator hypervector for scaling (growth).
    ///
    /// D_lambda = cosh(0.5*ln(lambda)) + sinh(0.5*ln(lambda)) * (n_inf ^ n0)
    pub fn create_dilator(&self, lambda: f32) -> ContinuousHV {
        let ln_lambda = (lambda as f64).ln();
        let half_ln = 0.5 * ln_lambda;

        // E = n_inf ^ n0 (dilation generator bivector)
        let e_bivector = Multivector::outer_product(&self.n_inf, &self.n0);

        let mut d_mv = Multivector::zero(&self.ga);
        let cosh_val = half_ln.cosh();
        let sinh_val = half_ln.sinh();

        d_mv.components[0] = cosh_val; // Scalar part
        for i in 0..self.ga.num_blades() {
            d_mv.components[i] += sinh_val * e_bivector.components[i];
        }

        self.multivector_to_hv(&d_mv)
    }

    /// Apply a conformal transformation (Dilator or Rotor) to a tissue hypervector.
    ///
    /// This is an O(D) sandwich operation in hypervector space.
    /// Note: Standard binding is commutative, so we apply the transformation
    /// by mapping the multivector algebra back down to the linear combination.
    pub fn apply_transformation(
        &self,
        tissue_hv: &ContinuousHV,
        transform_mv: &Multivector,
    ) -> ContinuousHV {
        // In a real biological manifold, we would transform each constitutive multivector
        // and re-bundle. Since it's linear: T(sum P_i) = sum T(P_i).
        // For a general multivector transformation: T(M) = V M V_rev

        // This prototype simply scales the coefficients of the basis blades
        // which corresponds to the linear action of the operator.
        let mut transformed_basis_hvs = Vec::with_capacity(self.ga.num_blades());
        let rev = transform_mv.reverse();

        for i in 0..self.ga.num_blades() {
            let mut e_i = Multivector::zero(&self.ga);
            e_i.components[i] = 1.0;

            // Sandwich: e_i' = V * e_i * V_rev
            let temp = Multivector::geometric_product(transform_mv, &e_i);
            let transformed_ei = Multivector::geometric_product(&temp, &rev);

            transformed_basis_hvs.push(self.multivector_to_hv(&transformed_ei));
        }

        // Apply the transformation matrix to the tissue hypervector
        // H' = sum ( similarity(H, B_i) * B_i' )
        let mut result_values = vec![0.0f32; self.dim];
        for i in 0..self.ga.num_blades() {
            let sim = tissue_hv.similarity(&self.basis_hvs[i]);
            let target_basis = &transformed_basis_hvs[i];
            for (acc, &b) in result_values.iter_mut().zip(target_basis.values.iter()) {
                *acc += sim * b;
            }
        }

        ContinuousHV::from_values(result_values).normalize()
    }
}
