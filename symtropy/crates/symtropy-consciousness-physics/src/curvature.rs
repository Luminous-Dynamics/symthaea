// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness-sourced space-time curvature.
//!
//! Implements conformal Riemannian geometry where consciousness (harmony field
//! energy) curves gamespace — the analog of mass curving spacetime in GR.
//!
//! # Conformal Metric
//!
//! ```text
//! g_ij(x) = e^{2σ(x)} × δ_ij
//! ```
//!
//! where `σ(x) = curvature_scale × harmony_field_energy(x)`.
//!
//! # Geodesic Equation
//!
//! For a conformal metric, the Christoffel connection gives:
//!
//! ```text
//! Γ^i_jk = δ^i_j ∂_k σ + δ^i_k ∂_j σ - δ_jk g^{il} ∂_l σ
//! ```
//!
//! The geodesic correction acceleration for a body with velocity v:
//!
//! ```text
//! a_correction = -2(v · ∇σ)v + |v|² ∇σ
//! ```
//!
//! This is clamped to `MAX_GEODESIC_ACCEL` to prevent numerical explosion
//! under semi-implicit Euler integration (the |v|² term would otherwise
//! create positive feedback loops for fast bodies).
//!
//! # Physical Meaning
//!
//! - Moving toward high-consciousness regions: velocity deflects toward source (attraction)
//! - Moving perpendicular to gradient: velocity bends inward (geodesic focusing)
//! - Stationary bodies: no correction (geodesics are trivial at v=0)
//! - High harmony → stronger curvature → stronger geodesic effects
//!
//! # References
//! - Carroll, S. (2004). *Spacetime and Geometry*. Section 3.4: Conformal transformations.
//! - Wald, R. (1984). *General Relativity*. Appendix D: Conformal transformations.

use nalgebra::SVector;

/// Maximum geodesic correction acceleration (units/s²).
///
/// Prevents numerical explosion when fast bodies enter steep gradients.
/// The raw geodesic correction scales with |v|², which under semi-implicit
/// Euler creates a positive feedback loop. This clamp preserves direction
/// while bounding magnitude.
pub const MAX_GEODESIC_ACCEL: f64 = 50.0;

/// Default curvature coupling constant.
/// Controls how strongly consciousness curves gamespace.
/// At 0.01, a harmony field energy of 10.0 produces σ ≈ 0.1 (subtle bending).
pub const DEFAULT_CURVATURE_SCALE: f64 = 0.01;

/// Conformal metric: g_ij = e^{2σ} × δ_ij where σ ∝ harmony field energy.
pub struct ConformalMetric<const D: usize> {
    /// Coupling constant: how strongly harmony energy curves space.
    pub curvature_scale: f64,
}

impl<const D: usize> ConformalMetric<D> {
    /// Create with default scale.
    pub fn new() -> Self {
        Self {
            curvature_scale: DEFAULT_CURVATURE_SCALE,
        }
    }

    /// Create with custom scale.
    pub fn with_scale(scale: f64) -> Self {
        Self {
            curvature_scale: scale,
        }
    }

    /// Conformal parameter σ at a given field energy.
    #[inline]
    pub fn sigma(&self, field_energy: f64) -> f64 {
        self.curvature_scale * field_energy
    }

    /// Conformal factor e^{2σ} at a given field energy.
    #[inline]
    pub fn conformal_factor(&self, field_energy: f64) -> f64 {
        (2.0 * self.sigma(field_energy)).exp()
    }

    /// Geodesic correction acceleration for a body with given velocity
    /// in a field with gradient ∇σ.
    ///
    /// Raw formula: a = -2(v · ∇σ)v + |v|² ∇σ
    ///
    /// Clamped to `MAX_GEODESIC_ACCEL` for numerical stability.
    pub fn geodesic_correction(
        &self,
        velocity: &SVector<f64, D>,
        sigma_gradient: &SVector<f64, D>,
    ) -> SVector<f64, D> {
        let v_dot_grad = velocity.dot(sigma_gradient);
        let v_sq = velocity.norm_squared();

        // Raw geodesic correction from conformal Christoffel symbols.
        let raw = sigma_gradient * v_sq - velocity * (2.0 * v_dot_grad);

        // CRITICAL: Clamp magnitude to prevent |v|² explosion.
        let mag = raw.norm();
        if mag > MAX_GEODESIC_ACCEL {
            raw * (MAX_GEODESIC_ACCEL / mag)
        } else {
            raw
        }
    }
}

    /// Ricci scalar curvature for validation (Fix 8).
    ///
    /// For conformal metric g = e^{2σ}δ in D dimensions:
    /// R = -2(D-1)(∇²σ + (D-1)|∇σ|²)   (small σ approximation)
    ///
    /// Negative R = positive curvature (space bends toward source).
    /// Zero R = flat space. Used for experiment validation, not dynamics.
    pub fn ricci_scalar(
        &self,
        sigma_gradient: &SVector<f64, D>,
        sigma_laplacian: f64,
    ) -> f64 {
        let d = D as f64;
        -2.0 * (d - 1.0) * (sigma_laplacian + (d - 1.0) * sigma_gradient.norm_squared())
    }
}

impl<const D: usize> Default for ConformalMetric<D> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_space_no_correction() {
        let metric = ConformalMetric::<2>::new();
        let v = SVector::from([10.0, 5.0]);
        let zero_grad = SVector::from([0.0, 0.0]);
        let correction = metric.geodesic_correction(&v, &zero_grad);
        assert!(correction.norm() < 1e-15, "Zero gradient = zero correction");
    }

    #[test]
    fn stationary_body_no_correction() {
        let metric = ConformalMetric::<2>::new();
        let zero_v = SVector::from([0.0, 0.0]);
        let grad = SVector::from([1.0, 0.5]);
        let correction = metric.geodesic_correction(&zero_v, &grad);
        assert!(correction.norm() < 1e-15, "Zero velocity = zero correction");
    }

    #[test]
    fn moving_body_deflected() {
        let metric = ConformalMetric::<2>::new();
        let v = SVector::from([10.0, 0.0]);
        let grad = SVector::from([0.0, 1.0]); // gradient perpendicular to motion
        let correction = metric.geodesic_correction(&v, &grad);
        // |v|² = 100, v·∇σ = 0 → a = |v|²∇σ = (0, 100) → clamped to (0, 50)
        assert!(correction[1] > 10.0, "Should deflect in gradient direction: {:?}", correction);
        assert!(correction.norm() <= MAX_GEODESIC_ACCEL + 1e-10, "Should be clamped");
    }

    #[test]
    fn conformal_factor_monotonic() {
        let metric = ConformalMetric::<3>::new();
        let f1 = metric.conformal_factor(1.0);
        let f2 = metric.conformal_factor(5.0);
        let f3 = metric.conformal_factor(10.0);
        assert!(f1 < f2 && f2 < f3, "Higher energy = larger conformal factor");
        assert!(f1 > 1.0, "Nonzero energy should produce factor > 1");
    }

    #[test]
    fn conformal_factor_identity_at_zero() {
        let metric = ConformalMetric::<2>::new();
        let f = metric.conformal_factor(0.0);
        assert!((f - 1.0).abs() < 1e-12, "Zero energy = flat space (factor=1)");
    }

    #[test]
    fn velocity_clamp_prevents_explosion() {
        let metric = ConformalMetric::<2>::new();
        // Very fast body + steep gradient → raw correction would be enormous.
        let v = SVector::from([1000.0, 0.0]); // |v|² = 1_000_000
        let grad = SVector::from([0.0, 10.0]);
        let correction = metric.geodesic_correction(&v, &grad);
        let mag = correction.norm();
        assert!(
            mag <= MAX_GEODESIC_ACCEL + 1e-10,
            "Correction magnitude {mag} should be clamped to {MAX_GEODESIC_ACCEL}"
        );
        // Direction should be preserved.
        assert!(correction[1] > 0.0, "Direction should point in gradient direction");
    }

    #[test]
    fn sigma_scales_with_energy() {
        let metric = ConformalMetric::<2>::with_scale(0.05);
        assert!((metric.sigma(10.0) - 0.5).abs() < 1e-12);
        assert!((metric.sigma(0.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn correction_3d() {
        let metric = ConformalMetric::<3>::new();
        let v = SVector::from([5.0, 0.0, 0.0]);
        let grad = SVector::from([0.0, 0.0, 1.0]);
        let correction = metric.geodesic_correction(&v, &grad);
        // |v|² = 25, v·∇σ = 0 → a = 25 * (0,0,1) = (0,0,25)
        assert!((correction[2] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn correction_4d() {
        let metric = ConformalMetric::<4>::new();
        let v = SVector::from([3.0, 0.0, 0.0, 0.0]);
        let grad = SVector::from([0.0, 1.0, 0.0, 0.0]);
        let correction = metric.geodesic_correction(&v, &grad);
        // |v|² = 9, v·∇σ = 0 → a = 9 * (0,1,0,0) = (0,9,0,0)
        assert!((correction[1] - 9.0).abs() < 1e-10);
    }
}
