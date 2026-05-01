// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Harmony-energy-parameterized conformal curvature.
//!
//! Implements conformal Riemannian geometry where the harmony field energy
//! parameterizes the metric tensor — analogous to mass-energy curving
//! spacetime in GR, but with harmony energy as the source term.
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
/// Controls how strongly harmony field energy parameterizes conformal curvature.
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
    /// Ricci scalar curvature for validation (Fix 8).
    ///
    /// For conformal metric g = e^{2σ}δ in D dimensions:
    /// R = -2(D-1)(∇²σ + (D-1)|∇σ|²)   (small σ approximation)
    ///
    /// Negative R = positive curvature (space bends toward source).
    /// Zero R = flat space. Used for experiment validation, not dynamics.
    pub fn ricci_scalar(&self, sigma_gradient: &SVector<f64, D>, sigma_laplacian: f64) -> f64 {
        let d = D as f64;
        -2.0 * (d - 1.0) * (sigma_laplacian + (d - 1.0) * sigma_gradient.norm_squared())
    }

    /// Integrate geodesic velocity correction using 4th-order Runge-Kutta.
    ///
    /// Solves the geodesic ODE for the velocity correction over timestep `dt`:
    ///
    /// ```text
    /// dv/dt = a(v) = -2(v · ∇σ)v + |v|² ∇σ
    /// ```
    ///
    /// The gradient `sigma_gradient` is held constant over the step (valid for
    /// small dt where the harmony field changes slowly relative to the physics
    /// tick). This gives O(dt⁴) local truncation error vs O(dt²) for Euler,
    /// making fast-moving bodies in steep harmony gradients numerically stable.
    ///
    /// The integrated correction `Δv = v_new − v` is clamped to
    /// `MAX_GEODESIC_ACCEL × dt` to prevent runaway under very large gradients.
    ///
    /// # Returns
    /// Updated velocity after one RK4 geodesic step.
    pub fn geodesic_rk4_step(
        &self,
        velocity: &SVector<f64, D>,
        sigma_gradient: &SVector<f64, D>,
        dt: f64,
    ) -> SVector<f64, D> {
        // Geodesic acceleration: a(v) = |v|²∇σ - 2(v·∇σ)v
        // (same formula as geodesic_correction but without clamping — we clamp the final delta)
        let accel = |v: &SVector<f64, D>| -> SVector<f64, D> {
            let v_dot_grad = v.dot(sigma_gradient);
            sigma_gradient * v.norm_squared() - v * (2.0 * v_dot_grad)
        };

        // Classical RK4 for dv/dt = accel(v), treating ∇σ as constant over dt.
        let k1 = accel(velocity);
        let k2 = accel(&(velocity + k1 * (dt * 0.5)));
        let k3 = accel(&(velocity + k2 * (dt * 0.5)));
        let k4 = accel(&(velocity + k3 * dt));

        let delta_v = (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);

        // Clamp |Δv| ≤ MAX_GEODESIC_ACCEL × dt to match the Euler safety guarantee.
        let max_delta = MAX_GEODESIC_ACCEL * dt.abs();
        let delta_mag = delta_v.norm();
        let clamped_delta = if delta_mag > max_delta && delta_mag > 1e-30 {
            delta_v * (max_delta / delta_mag)
        } else {
            delta_v
        };

        velocity + clamped_delta
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
        assert!(
            correction[1] > 10.0,
            "Should deflect in gradient direction: {:?}",
            correction
        );
        assert!(
            correction.norm() <= MAX_GEODESIC_ACCEL + 1e-10,
            "Should be clamped"
        );
    }

    #[test]
    fn conformal_factor_monotonic() {
        let metric = ConformalMetric::<3>::new();
        let f1 = metric.conformal_factor(1.0);
        let f2 = metric.conformal_factor(5.0);
        let f3 = metric.conformal_factor(10.0);
        assert!(
            f1 < f2 && f2 < f3,
            "Higher energy = larger conformal factor"
        );
        assert!(f1 > 1.0, "Nonzero energy should produce factor > 1");
    }

    #[test]
    fn conformal_factor_identity_at_zero() {
        let metric = ConformalMetric::<2>::new();
        let f = metric.conformal_factor(0.0);
        assert!(
            (f - 1.0).abs() < 1e-12,
            "Zero energy = flat space (factor=1)"
        );
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
        assert!(
            correction[1] > 0.0,
            "Direction should point in gradient direction"
        );
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

    // ── geodesic_rk4_step ─────────────────────────────────────────────────────

    #[test]
    fn rk4_zero_gradient_returns_unchanged_velocity() {
        let metric = ConformalMetric::<2>::new();
        let v = SVector::from([10.0, 5.0]);
        let zero_grad = SVector::from([0.0, 0.0]);
        let v_new = metric.geodesic_rk4_step(&v, &zero_grad, 0.016);
        assert!(
            (v_new - v).norm() < 1e-12,
            "zero gradient = no velocity change"
        );
    }

    #[test]
    fn rk4_zero_velocity_returns_unchanged_velocity() {
        let metric = ConformalMetric::<2>::new();
        let v = SVector::from([0.0, 0.0]);
        let grad = SVector::from([1.0, 0.5]);
        let v_new = metric.geodesic_rk4_step(&v, &grad, 0.016);
        // a(v=0) = |0|²∇σ - 2*(0·∇σ)*0 = 0 → no change
        assert!(
            (v_new - v).norm() < 1e-12,
            "zero velocity = no geodesic change"
        );
    }

    #[test]
    fn rk4_converges_to_euler_at_small_dt() {
        // Use a weak gradient so the raw correction stays well below MAX_GEODESIC_ACCEL
        // and the clamp never fires — both methods should then agree to O(dt²).
        // v = [3, 0], grad = [0, 0.5] → raw accel = 9 * [0, 0.5] = [0, 4.5] < 50.
        let metric = ConformalMetric::<2>::new();
        let v = SVector::from([3.0, 0.0]);
        let grad = SVector::from([0.0, 0.5]);

        let dt = 0.001; // small enough that O(dt²) ≈ 1e-6

        let euler = v + metric.geodesic_correction(&v, &grad) * dt;
        let rk4 = metric.geodesic_rk4_step(&v, &grad, dt);
        let diff = (euler - rk4).norm();
        assert!(
            diff < 1e-4,
            "small dt without clamping: euler and rk4 should agree to O(dt²), diff={diff}"
        );
    }

    #[test]
    fn rk4_correction_is_bounded_for_large_dt() {
        // For a typical physics tick (dt=0.016), the integrated correction must stay
        // within MAX_GEODESIC_ACCEL * dt regardless of gradient steepness.
        let metric = ConformalMetric::<2>::new();
        let v = SVector::from([10.0, 0.0]);
        let grad = SVector::from([0.0, 1.0]);
        let dt_large = 0.016;

        let rk4_large = metric.geodesic_rk4_step(&v, &grad, dt_large);
        let delta = (rk4_large - v).norm();
        assert!(
            delta > 1e-6,
            "rk4 should produce a nonzero correction for nonzero v and grad"
        );
        assert!(
            delta <= MAX_GEODESIC_ACCEL * dt_large + 1e-10,
            "rk4 delta should be clamped"
        );
    }

    #[test]
    fn rk4_clamps_large_correction() {
        let metric = ConformalMetric::<2>::new();
        // Extremely fast body in a very steep gradient.
        let v = SVector::from([1000.0, 0.0]);
        let grad = SVector::from([0.0, 100.0]);
        let dt = 0.016;
        let v_new = metric.geodesic_rk4_step(&v, &grad, dt);
        let delta = (v_new - v).norm();
        assert!(
            delta <= MAX_GEODESIC_ACCEL * dt + 1e-10,
            "delta {delta} exceeds MAX_GEODESIC_ACCEL * dt = {}",
            MAX_GEODESIC_ACCEL * dt
        );
    }

    #[test]
    fn rk4_direction_deflects_toward_gradient() {
        // Body moving in +x, gradient in +y → should acquire +y velocity component.
        let metric = ConformalMetric::<2>::new();
        let v = SVector::from([10.0, 0.0]);
        let grad = SVector::from([0.0, 1.0]);
        let v_new = metric.geodesic_rk4_step(&v, &grad, 0.016);
        assert!(
            v_new[1] > 0.0,
            "body perpendicular to gradient should acquire velocity in gradient direction"
        );
        assert!(
            v_new[0] > 0.0,
            "forward velocity should still be positive after small dt"
        );
    }

    #[test]
    fn rk4_3d_consistency() {
        let metric = ConformalMetric::<3>::new();
        let v = SVector::from([5.0, 0.0, 0.0]);
        let grad = SVector::from([0.0, 0.0, 1.0]);
        let v_new = metric.geodesic_rk4_step(&v, &grad, 0.016);
        // Should acquire +z component (gradient direction)
        assert!(
            v_new[2] > 0.0,
            "3D: should acquire velocity in gradient direction"
        );
        // x component should be approximately preserved for small dt
        assert!(
            (v_new[0] - 5.0).abs() < 1.0,
            "x velocity should be approximately preserved"
        );
    }

    #[test]
    fn rk4_4d_no_panic() {
        let metric = ConformalMetric::<4>::new();
        let v = SVector::from([3.0, 1.0, 0.0, 0.0]);
        let grad = SVector::from([0.0, 0.0, 1.0, 0.5]);
        let v_new = metric.geodesic_rk4_step(&v, &grad, 0.016);
        assert!(
            v_new.norm().is_finite(),
            "4D rk4 step should produce finite velocity"
        );
    }
}
