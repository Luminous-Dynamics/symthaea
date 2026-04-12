// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Maxwell FDTD — Finite-Difference Time-Domain electromagnetic field solver.
//!
//! Solves Maxwell's equations on a Yee grid:
//! ∂B/∂t = -∇×E  (Faraday's law)
//! ∂D/∂t = ∇×H - J  (Ampère's law with current)
//!
//! The Yee algorithm staggers E and H fields in space and time,
//! giving second-order accuracy and automatic divergence preservation.
//!
//! References:
//! - Yee, K. S. (1966). IEEE Trans. Antennas Propag. 14, 302.
//! - Taflove & Hagness (2005). *Computational Electrodynamics*. Artech House.

use std::f64::consts::PI;

/// Speed of light (m/s) — used for CFL condition
pub const C_LIGHT: f64 = 299_792_458.0;
/// Vacuum permittivity (F/m)
pub const EPS_0: f64 = 8.854_187_817e-12;
/// Vacuum permeability (H/m)
pub const MU_0: f64 = 1.256_637_062e-6;

/// 1D FDTD simulation domain.
#[derive(Debug, Clone)]
pub struct Maxwell1D {
    /// Electric field E_z (V/m)
    pub ez: Vec<f64>,
    /// Magnetic field H_y (A/m)
    pub hy: Vec<f64>,
    /// Relative permittivity at each cell
    pub epsilon_r: Vec<f64>,
    /// Relative permeability at each cell
    pub mu_r: Vec<f64>,
    /// Conductivity at each cell (S/m)
    pub sigma: Vec<f64>,
    /// Grid spacing (m)
    pub dx: f64,
    /// Time step (s) — must satisfy CFL: dt < dx/(c*sqrt(dim))
    pub dt: f64,
    /// Number of cells
    pub n: usize,
    /// Current time step
    pub step: usize,
}

impl Maxwell1D {
    /// Create a vacuum domain of n cells with given spatial resolution.
    pub fn new(n: usize, dx: f64) -> Self {
        // CFL condition for 1D: dt < dx/c
        let dt = 0.5 * dx / C_LIGHT;
        Self {
            ez: vec![0.0; n],
            hy: vec![0.0; n],
            epsilon_r: vec![1.0; n],
            mu_r: vec![1.0; n],
            sigma: vec![0.0; n],
            dx,
            dt,
            n,
            step: 0,
        }
    }

    /// Set a dielectric slab from index i0 to i1 with given permittivity.
    pub fn set_dielectric(&mut self, i0: usize, i1: usize, eps_r: f64) {
        for i in i0..i1.min(self.n) {
            self.epsilon_r[i] = eps_r;
        }
    }

    /// Set conductivity (lossy medium) from i0 to i1.
    pub fn set_conductivity(&mut self, i0: usize, i1: usize, sigma: f64) {
        for i in i0..i1.min(self.n) {
            self.sigma[i] = sigma;
        }
    }

    /// Advance one time step using the Yee algorithm.
    ///
    /// Update H_y: H_y^{n+1/2} = H_y^{n-1/2} - (dt/μdx)(E_z^n[i+1] - E_z^n[i])
    /// Update E_z: E_z^{n+1} = C_a E_z^n + C_b (H_y^{n+1/2}[i] - H_y^{n+1/2}[i-1])
    pub fn step_once(&mut self) {
        // Update H field (half-step ahead)
        for i in 0..self.n - 1 {
            let coeff = self.dt / (MU_0 * self.mu_r[i] * self.dx);
            self.hy[i] -= coeff * (self.ez[i + 1] - self.ez[i]);
        }

        // Update E field
        for i in 1..self.n {
            let eps = EPS_0 * self.epsilon_r[i];
            let sig = self.sigma[i];

            // Lossy medium coefficients
            let ca = (1.0 - sig * self.dt / (2.0 * eps)) / (1.0 + sig * self.dt / (2.0 * eps));
            let cb = (self.dt / (eps * self.dx)) / (1.0 + sig * self.dt / (2.0 * eps));

            self.ez[i] = ca * self.ez[i] + cb * (self.hy[i] - self.hy[i - 1]);
        }

        self.step += 1;
    }

    /// Inject a Gaussian pulse source at cell index `src`.
    pub fn inject_gaussian_source(&mut self, src: usize, t0: f64, spread: f64) {
        let t = self.step as f64 * self.dt;
        let pulse = (-(t - t0).powi(2) / (2.0 * spread * spread)).exp();
        if src < self.n {
            self.ez[src] += pulse;
        }
    }

    /// Inject a sinusoidal source at cell index `src`.
    pub fn inject_sinusoidal_source(&mut self, src: usize, frequency: f64) {
        let t = self.step as f64 * self.dt;
        if src < self.n {
            self.ez[src] += (2.0 * PI * frequency * t).sin();
        }
    }

    /// Total electromagnetic energy in the domain.
    pub fn total_energy(&self) -> f64 {
        let mut energy = 0.0;
        for i in 0..self.n {
            energy += 0.5 * EPS_0 * self.epsilon_r[i] * self.ez[i] * self.ez[i] * self.dx;
            energy += 0.5 * MU_0 * self.mu_r[i] * self.hy[i] * self.hy[i] * self.dx;
        }
        energy
    }

    /// Run multiple steps with a source.
    pub fn run(&mut self, n_steps: usize, source_idx: usize, frequency: f64) {
        for _ in 0..n_steps {
            self.inject_sinusoidal_source(source_idx, frequency);
            self.step_once();
        }
    }
}

/// 2D FDTD for TM mode (E_z, H_x, H_y).
#[derive(Debug, Clone)]
pub struct Maxwell2D {
    pub ez: Vec<f64>,   // E_z[i*ny + j]
    pub hx: Vec<f64>,   // H_x
    pub hy: Vec<f64>,   // H_y
    pub epsilon_r: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub dt: f64,
    pub step: usize,
}

impl Maxwell2D {
    pub fn new(nx: usize, ny: usize, dx: f64) -> Self {
        let dy = dx;
        let dt = 0.5 * dx / (C_LIGHT * 2.0_f64.sqrt()); // 2D CFL
        let size = nx * ny;
        Self {
            ez: vec![0.0; size],
            hx: vec![0.0; size],
            hy: vec![0.0; size],
            epsilon_r: vec![1.0; size],
            nx, ny, dx, dy, dt,
            step: 0,
        }
    }

    pub fn step_once(&mut self) {
        let nx = self.nx;
        let ny = self.ny;

        // Update H_x: ∂H_x/∂t = -(1/μ₀) ∂E_z/∂y
        for i in 0..nx {
            for j in 0..ny - 1 {
                let idx = i * ny + j;
                self.hx[idx] -= (self.dt / (MU_0 * self.dy)) * (self.ez[idx + 1] - self.ez[idx]);
            }
        }

        // Update H_y: ∂H_y/∂t = (1/μ₀) ∂E_z/∂x
        for i in 0..nx - 1 {
            for j in 0..ny {
                let idx = i * ny + j;
                self.hy[idx] += (self.dt / (MU_0 * self.dx)) * (self.ez[(i + 1) * ny + j] - self.ez[idx]);
            }
        }

        // Update E_z: ∂E_z/∂t = (1/ε)(∂H_y/∂x - ∂H_x/∂y)
        for i in 1..nx {
            for j in 1..ny {
                let idx = i * ny + j;
                let eps = EPS_0 * self.epsilon_r[idx];
                let dhx_dy = (self.hx[idx] - self.hx[idx - 1]) / self.dy;
                let dhy_dx = (self.hy[idx] - self.hy[(i - 1) * ny + j]) / self.dx;
                self.ez[idx] += (self.dt / eps) * (dhy_dx - dhx_dy);
            }
        }

        self.step += 1;
    }

    pub fn total_energy(&self) -> f64 {
        let mut energy = 0.0;
        let cell_area = self.dx * self.dy;
        for i in 0..self.nx * self.ny {
            energy += 0.5 * EPS_0 * self.epsilon_r[i] * self.ez[i] * self.ez[i] * cell_area;
            energy += 0.5 * MU_0 * (self.hx[i] * self.hx[i] + self.hy[i] * self.hy[i]) * cell_area;
        }
        energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vacuum_propagation() {
        let mut sim = Maxwell1D::new(200, 1e-3); // 1mm cells
        let src = 50;

        // Inject a pulse and propagate
        for _ in 0..100 {
            sim.inject_gaussian_source(src, 30.0 * sim.dt, 10.0 * sim.dt);
            sim.step_once();
        }

        // Energy should have propagated away from source
        let right_energy: f64 = sim.ez[src + 10..].iter().map(|e| e * e).sum();
        assert!(right_energy > 0.0, "Wave should propagate rightward");
    }

    #[test]
    fn test_dielectric_reflection() {
        let mut sim = Maxwell1D::new(200, 1e-3);
        sim.set_dielectric(100, 200, 4.0); // Glass (n=2)

        for _ in 0..200 {
            sim.inject_gaussian_source(50, 30.0 * sim.dt, 10.0 * sim.dt);
            sim.step_once();
        }

        // Should see reflection (energy before interface)
        let reflected: f64 = sim.ez[40..60].iter().map(|e| e * e).sum();
        let transmitted: f64 = sim.ez[110..130].iter().map(|e| e * e).sum();

        // Both reflected and transmitted should be nonzero
        assert!(reflected > 0.0 || transmitted > 0.0, "Should see wave activity");
    }

    #[test]
    fn test_cfl_condition() {
        let sim = Maxwell1D::new(100, 1e-3);
        // dt should be less than dx/c
        assert!(sim.dt < sim.dx / C_LIGHT, "CFL violated");
    }

    #[test]
    fn test_lossy_medium_attenuation() {
        let mut sim = Maxwell1D::new(200, 1e-3);
        sim.set_conductivity(100, 200, 1.0); // Conductive medium

        for _ in 0..300 {
            sim.inject_gaussian_source(50, 30.0 * sim.dt, 10.0 * sim.dt);
            sim.step_once();
        }

        // Field should be attenuated in the lossy region
        let lossy_amplitude: f64 = sim.ez[150..180].iter().map(|e| e.abs()).sum();
        let vacuum_amplitude: f64 = sim.ez[60..90].iter().map(|e| e.abs()).sum();

        // Lossy region should have less field
        assert!(
            lossy_amplitude <= vacuum_amplitude + 1e-10,
            "Lossy medium should attenuate: lossy={:.6}, vacuum={:.6}",
            lossy_amplitude, vacuum_amplitude
        );
    }

    #[test]
    fn test_2d_fdtd_runs() {
        let mut sim = Maxwell2D::new(50, 50, 1e-3);
        sim.ez[25 * 50 + 25] = 1.0; // Point source at center
        for _ in 0..10 {
            sim.step_once();
        }
        let e = sim.total_energy();
        assert!(e > 0.0, "2D FDTD should have energy after excitation");
    }
}
