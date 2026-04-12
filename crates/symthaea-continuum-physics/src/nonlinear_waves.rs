// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Nonlinear wave equations — solitons, neural oscillations, KdV.
//!
//! Neural dynamics are fundamentally nonlinear wave phenomena.
//! This module solves the key equations governing oscillatory systems
//! relevant to consciousness.
//!
//! References:
//! - Hodgkin & Huxley (1952). J. Physiol. 117, 500.
//! - Korteweg & de Vries (1895). Phil. Mag. 39, 422 (KdV).
//! - FitzHugh, R. (1961). Biophys. J. 1, 445.

use std::f64::consts::PI;

/// KdV (Korteweg-de Vries) equation: u_t + 6u·u_x + u_xxx = 0
///
/// The KdV equation supports soliton solutions — localized waves that
/// maintain shape. It models shallow water waves and ion-acoustic waves.
#[derive(Debug, Clone)]
pub struct KdvSolver {
    pub u: Vec<f64>,
    pub dx: f64,
    pub dt: f64,
    pub n: usize,
}

impl KdvSolver {
    /// Create solver with periodic boundary conditions.
    pub fn new(n: usize, domain_length: f64) -> Self {
        let dx = domain_length / n as f64;
        let dt = 0.01 * dx * dx * dx; // CFL for third-order term
        Self {
            u: vec![0.0; n],
            dx, dt, n,
        }
    }

    /// Initialize with a soliton: u(x) = (c/2) sech²(√c/2 × (x - x0))
    pub fn init_soliton(&mut self, amplitude: f64, x0: f64) {
        let c = 2.0 * amplitude;
        let width = (c / 4.0).sqrt();
        let total_length = self.n as f64 * self.dx;
        for i in 0..self.n {
            let x = i as f64 * self.dx;
            let arg = width * (x - x0);
            let sech = 1.0 / arg.cosh();
            self.u[i] = amplitude * sech * sech;
        }
    }

    /// Advance one step using 4th-order Runge-Kutta with periodic BCs.
    pub fn step_once(&mut self) {
        let rhs = |u: &[f64]| -> Vec<f64> {
            let n = u.len();
            let dx = self.dx;
            let mut du = vec![0.0; n];
            for i in 0..n {
                let ip1 = (i + 1) % n;
                let im1 = (i + n - 1) % n;
                let ip2 = (i + 2) % n;
                let im2 = (i + n - 2) % n;

                // -6u·u_x (central difference for advection)
                let ux = (u[ip1] - u[im1]) / (2.0 * dx);
                // -u_xxx (central difference for dispersion)
                let uxxx = (u[ip2] - 2.0 * u[ip1] + 2.0 * u[im1] - u[im2]) / (2.0 * dx * dx * dx);

                du[i] = -6.0 * u[i] * ux - uxxx;
            }
            du
        };

        let dt = self.dt;
        let u = &self.u;
        let k1 = rhs(u);
        let u2: Vec<f64> = u.iter().zip(&k1).map(|(a, b)| a + 0.5 * dt * b).collect();
        let k2 = rhs(&u2);
        let u3: Vec<f64> = u.iter().zip(&k2).map(|(a, b)| a + 0.5 * dt * b).collect();
        let k3 = rhs(&u3);
        let u4: Vec<f64> = u.iter().zip(&k3).map(|(a, b)| a + dt * b).collect();
        let k4 = rhs(&u4);

        for i in 0..self.n {
            self.u[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }

    /// Total "mass" (integral of u): should be conserved by KdV.
    pub fn total_mass(&self) -> f64 {
        self.u.iter().sum::<f64>() * self.dx
    }
}

/// FitzHugh-Nagumo model: simplified Hodgkin-Huxley neural oscillator.
///
/// dv/dt = v - v³/3 - w + I_ext
/// dw/dt = ε(v + a - bw)
///
/// v = membrane voltage, w = recovery variable, I_ext = external current
#[derive(Debug, Clone)]
pub struct FitzHughNagumo {
    pub v: f64, // Membrane voltage
    pub w: f64, // Recovery variable
    pub a: f64, // Offset parameter (typ. 0.7)
    pub b: f64, // Recovery coupling (typ. 0.8)
    pub epsilon: f64, // Time scale separation (typ. 0.08)
    pub i_ext: f64, // External stimulus current
}

impl FitzHughNagumo {
    pub fn new() -> Self {
        Self {
            v: -1.0, w: -0.5,
            a: 0.7, b: 0.8, epsilon: 0.08,
            i_ext: 0.0,
        }
    }

    /// Advance by dt using RK4.
    pub fn step(&mut self, dt: f64) {
        let f = |v: f64, w: f64| -> (f64, f64) {
            let dv = v - v * v * v / 3.0 - w + self.i_ext;
            let dw = self.epsilon * (v + self.a - self.b * w);
            (dv, dw)
        };

        let (k1v, k1w) = f(self.v, self.w);
        let (k2v, k2w) = f(self.v + 0.5 * dt * k1v, self.w + 0.5 * dt * k1w);
        let (k3v, k3w) = f(self.v + 0.5 * dt * k2v, self.w + 0.5 * dt * k2w);
        let (k4v, k4w) = f(self.v + dt * k3v, self.w + dt * k3w);

        self.v += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        self.w += dt / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
    }

    /// Run for n steps, return voltage trace.
    pub fn run(&mut self, n_steps: usize, dt: f64) -> Vec<f64> {
        let mut trace = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            self.step(dt);
            trace.push(self.v);
        }
        trace
    }

    /// Check if the neuron is in the oscillatory regime (I_ext > threshold).
    /// The Hopf bifurcation occurs at I_ext ≈ (a - b*(a-1)/3) for standard params.
    pub fn is_oscillatory(&self) -> bool {
        self.i_ext > 0.33 // Approximate Hopf bifurcation for default params
    }
}

/// Coupled oscillator network — models neural populations.
#[derive(Debug, Clone)]
pub struct OscillatorNetwork {
    pub oscillators: Vec<FitzHughNagumo>,
    /// Coupling matrix: coupling[i*n + j] = strength from j to i
    pub coupling: Vec<f64>,
    pub n: usize,
}

impl OscillatorNetwork {
    /// Create n coupled oscillators with uniform coupling strength.
    pub fn new(n: usize, coupling_strength: f64) -> Self {
        let mut oscillators: Vec<FitzHughNagumo> = (0..n).map(|_| FitzHughNagumo::new()).collect();
        let mut coupling = vec![0.0; n * n];

        // All-to-all coupling (except self)
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    coupling[i * n + j] = coupling_strength / (n - 1) as f64;
                }
            }
            // Slightly different initial conditions for symmetry breaking
            oscillators[i].v = -1.0 + 0.1 * (i as f64 / n as f64);
        }

        Self { oscillators, coupling, n }
    }

    /// Advance one step with coupling: dv_i/dt += Σ_j g_ij (v_j - v_i)
    pub fn step(&mut self, dt: f64) {
        // Compute coupling currents
        let voltages: Vec<f64> = self.oscillators.iter().map(|o| o.v).collect();
        let mut coupling_currents = vec![0.0; self.n];

        for i in 0..self.n {
            for j in 0..self.n {
                coupling_currents[i] += self.coupling[i * self.n + j] * (voltages[j] - voltages[i]);
            }
        }

        // Step each oscillator with coupling as external current
        for i in 0..self.n {
            let original_i = self.oscillators[i].i_ext;
            self.oscillators[i].i_ext += coupling_currents[i];
            self.oscillators[i].step(dt);
            self.oscillators[i].i_ext = original_i;
        }
    }

    /// Synchronization order parameter: r = |1/N Σ exp(iθ_j)| ∈ [0, 1]
    /// Uses voltage phase estimated from zero-crossings.
    pub fn synchronization_index(&self) -> f64 {
        let n = self.n as f64;
        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;

        for osc in &self.oscillators {
            // Approximate phase from voltage (simple mapping)
            let phase = PI * (osc.v + 1.0) / 2.0;
            sum_cos += phase.cos();
            sum_sin += phase.sin();
        }

        ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kdv_soliton_mass_conservation() {
        let mut kdv = KdvSolver::new(256, 20.0);
        kdv.init_soliton(1.0, 10.0);

        let mass_0 = kdv.total_mass();
        for _ in 0..100 {
            kdv.step_once();
        }
        let mass_1 = kdv.total_mass();

        assert!(
            (mass_0 - mass_1).abs() / mass_0.abs().max(1e-10) < 0.01,
            "KdV mass: initial={:.6}, final={:.6}", mass_0, mass_1
        );
    }

    #[test]
    fn test_fhn_subthreshold_no_spike() {
        let mut fhn = FitzHughNagumo::new();
        fhn.i_ext = 0.1; // Below threshold
        let trace = fhn.run(1000, 0.1);

        let max_v = trace.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_v < 1.5, "Subthreshold: max v = {:.3} (no spike)", max_v);
    }

    #[test]
    fn test_fhn_suprathreshold_oscillation() {
        let mut fhn = FitzHughNagumo::new();
        fhn.i_ext = 0.5; // Above Hopf bifurcation
        let trace = fhn.run(5000, 0.1);

        // Should show oscillation: check that v crosses zero multiple times
        let mut crossings = 0;
        for i in 1..trace.len() {
            if trace[i - 1] < 0.0 && trace[i] >= 0.0 {
                crossings += 1;
            }
        }
        assert!(crossings > 2, "Should oscillate: {} zero-crossings", crossings);
    }

    #[test]
    fn test_coupled_oscillators_synchronize() {
        let mut net = OscillatorNetwork::new(5, 0.5);
        for osc in &mut net.oscillators {
            osc.i_ext = 0.5; // All in oscillatory regime
        }

        // Run until synchronization develops
        for _ in 0..5000 {
            net.step(0.1);
        }

        let r = net.synchronization_index();
        assert!(r > 0.3, "Coupled oscillators should tend to synchronize: r={:.3}", r);
    }
}
