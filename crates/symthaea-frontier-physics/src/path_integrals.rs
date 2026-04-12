// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Feynman Path Integrals — sum over histories.
//!
//! K(b,a) = ∫ D[x(t)] exp(iS[x]/ℏ)
//!
//! The path integral unifies quantum mechanics and statistical mechanics:
//! QM propagator (real time) ↔ partition function (imaginary time).
//!
//! References:
//! - Feynman & Hibbs (1965). *Quantum Mechanics and Path Integrals*.
//! - Kleinert, H. (2009). *Path Integrals in Quantum Mechanics*. World Scientific.

use std::f64::consts::PI;

/// Discretized path integral propagator for a 1D potential.
///
/// K(x_f, x_i; T) ≈ (m/2πiℏΔt)^{N/2} × ∫ Π dx_k exp(iS/ℏ)
///
/// Uses Monte Carlo sampling of paths with N time slices.
pub struct PathIntegral1D {
    /// Number of time slices
    pub n_slices: usize,
    /// Total propagation time
    pub total_time: f64,
    /// Particle mass
    pub mass: f64,
    /// Potential function V(x)
    potential: Box<dyn Fn(f64) -> f64>,
}

impl PathIntegral1D {
    pub fn new(n_slices: usize, total_time: f64, mass: f64, potential: Box<dyn Fn(f64) -> f64>) -> Self {
        Self { n_slices, total_time, mass, potential }
    }

    /// Compute the Euclidean action S_E[path] = Σ [½m(Δx/Δt)² + V(x)] Δt.
    /// Euclidean (imaginary time) → statistical mechanics.
    pub fn euclidean_action(&self, path: &[f64]) -> f64 {
        let dt = self.total_time / self.n_slices as f64;
        let mut action = 0.0;
        for i in 0..path.len() - 1 {
            let dx = path[i + 1] - path[i];
            let kinetic = 0.5 * self.mass * dx * dx / (dt * dt);
            let potential = 0.5 * ((self.potential)(path[i]) + (self.potential)(path[i + 1]));
            action += (kinetic + potential) * dt;
        }
        action
    }

    /// Evaluate the partition function Z(β) via path integral Monte Carlo.
    /// Z = ∫ D[x] exp(-S_E[x]) with periodic boundary conditions.
    ///
    /// Uses Metropolis sampling. `beta` = 1/kT = imaginary time extent.
    pub fn partition_function_mc(&self, beta: f64, n_samples: usize, seed: u64) -> f64 {
        let n = self.n_slices;
        let dt = beta / n as f64;

        // Initialize path at x=0
        let mut path = vec![0.0; n + 1];
        path[n] = path[0]; // Periodic BC

        let mut rng_state = seed;
        let mut pseudo_random = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng_state >> 33) as f64 / (1u64 << 31) as f64
        };

        let mut sum_weight = 0.0;
        let mut s_current = self.euclidean_action_beta(&path, dt);

        for _ in 0..n_samples {
            // Propose move: perturb a random time slice
            let k = ((pseudo_random() * n as f64) as usize).min(n - 1) + 1;
            let old_x = path[k];
            let dx = (pseudo_random() - 0.5) * 2.0;
            path[k] = old_x + dx;

            let s_new = self.euclidean_action_beta(&path, dt);
            let ds = s_new - s_current;

            // Metropolis accept/reject
            if ds < 0.0 || pseudo_random() < (-ds).exp() {
                s_current = s_new;
            } else {
                path[k] = old_x;
            }

            sum_weight += (-s_current).exp();
        }

        sum_weight / n_samples as f64
    }

    fn euclidean_action_beta(&self, path: &[f64], dt: f64) -> f64 {
        let mut action = 0.0;
        for i in 0..path.len() - 1 {
            let dx = path[i + 1] - path[i];
            let kinetic = 0.5 * self.mass * dx * dx / dt;
            let potential = 0.5 * ((self.potential)(path[i]) + (self.potential)(path[i + 1]));
            action += kinetic + potential * dt;
        }
        action
    }
}

/// Free particle propagator (exact): K(x_f, x_i; t) = √(m/2πiℏt) exp(im(x_f-x_i)²/2ℏt)
/// Returns |K|² (probability density).
pub fn free_particle_propagator_sq(x_f: f64, x_i: f64, mass: f64, time: f64) -> f64 {
    if time <= 0.0 { return 0.0; }
    let prefactor = mass / (2.0 * PI * time);
    let dx = x_f - x_i;
    prefactor * (-mass * dx * dx / (2.0 * time)).exp()
    // Note: in natural units ℏ=1, the Gaussian spread gives this
    // The actual |K|² for imaginary time (Euclidean) propagator
}

/// Harmonic oscillator propagator (exact, Euclidean/thermal).
/// K_E(x_f, x_i; β) = √(mω/2πsinh(ωβ)) × exp(-mω/(2sinh(ωβ))[(x_f²+x_i²)cosh(ωβ) - 2x_f x_i])
pub fn harmonic_oscillator_propagator(
    x_f: f64, x_i: f64, mass: f64, omega: f64, beta: f64,
) -> f64 {
    let sinh_ob = (omega * beta).sinh();
    let cosh_ob = (omega * beta).cosh();

    if sinh_ob.abs() < 1e-30 { return 0.0; }

    let prefactor = (mass * omega / (2.0 * PI * sinh_ob)).sqrt();
    let exponent = -mass * omega / (2.0 * sinh_ob)
        * ((x_f * x_f + x_i * x_i) * cosh_ob - 2.0 * x_f * x_i);

    prefactor * exponent.exp()
}

/// Quantum tunneling rate via instanton (WKB path integral).
/// Γ ≈ A × exp(-S_bounce/ℏ) where S_bounce is the Euclidean action of the bounce solution.
///
/// For a symmetric double-well V(x) = λ(x²-a²)², the bounce action is:
/// S_bounce = (4/3)√(2m) a³ √λ
pub fn tunneling_rate_instanton(mass: f64, barrier_height: f64, barrier_width: f64) -> f64 {
    let s_bounce = (16.0 / 3.0) * (2.0 * mass).sqrt() * barrier_height * barrier_width;
    (-s_bounce).exp() // Prefactor omitted (A ~ ω)
}

/// Wick rotation: connect real-time QM ↔ imaginary-time StatMech.
/// t → -iτ (β = iT), action S → iS_E.
/// The QM propagator becomes the thermal density matrix:
/// ⟨x|e^{-βH}|x'⟩ = K_E(x, x'; β)
pub fn wick_rotate_energy(energy_real_time: f64) -> f64 {
    // In Euclidean time, energy eigenvalue just stays the same
    // but the propagator becomes exp(-E*tau) instead of exp(-iEt)
    energy_real_time
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_propagator_positive() {
        // Euclidean propagator should be positive
        let p = free_particle_propagator_sq(1.0, 0.0, 1.0, 1.0);
        assert!(p > 0.0, "Propagator should be positive: {:.6}", p);
        // At origin (x_f = x_i), propagator increases with time (Euclidean localization)
        let p_origin_short = free_particle_propagator_sq(0.0, 0.0, 1.0, 0.1);
        let p_origin_long = free_particle_propagator_sq(0.0, 0.0, 1.0, 1.0);
        assert!(p_origin_short > p_origin_long, "Origin propagator decreases: {:.6} > {:.6}", p_origin_short, p_origin_long);
    }

    #[test]
    fn test_harmonic_propagator_ground_state() {
        // At β→∞ (T→0), propagator projects onto ground state
        let k = harmonic_oscillator_propagator(0.0, 0.0, 1.0, 1.0, 100.0);
        assert!(k > 0.0, "Ground state propagator should be positive: {:.6}", k);
    }

    #[test]
    fn test_tunneling_barrier_dependence() {
        // Higher barrier → lower tunneling rate
        let r_low = tunneling_rate_instanton(1.0, 1.0, 1.0);
        let r_high = tunneling_rate_instanton(1.0, 2.0, 1.0);
        assert!(r_low > r_high, "Higher barrier = less tunneling");
    }

    #[test]
    fn test_euclidean_action_free_particle() {
        let pi = PathIntegral1D::new(10, 1.0, 1.0, Box::new(|_| 0.0));
        // Straight path from 0 to 1: S = ½m(Δx)²/T = 0.5
        let path: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
        let s = pi.euclidean_action(&path);
        assert!((s - 0.5).abs() < 0.1, "Free particle action = {:.4}, expected ~0.5", s);
    }

    #[test]
    fn test_path_integral_mc_runs() {
        let pi = PathIntegral1D::new(20, 1.0, 1.0, Box::new(|x| 0.5 * x * x));
        let z = pi.partition_function_mc(1.0, 1000, 42);
        assert!(z > 0.0 && z.is_finite(), "Z should be positive and finite: {:.6}", z);
    }
}
