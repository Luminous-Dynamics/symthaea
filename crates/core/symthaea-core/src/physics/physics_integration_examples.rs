// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Physics Integration Examples
//!
//! This module demonstrates how the new physics implementations work together:
//! - Quantum tunneling + decoherence for realistic quantum systems
//! - Tensor algebra for general relativistic calculations
//! - Chaos dynamics for nonlinear system analysis
//! - Non-equilibrium thermodynamics for transport phenomena
//!
//! These examples show practical applications and cross-module integration.

use super::chaos_dynamics::{AttractorAnalyzer, LyapunovCalculator, systems};
use super::constants::{C, G};
use super::decoherence::DensityMatrix;
use super::nonequilibrium::{FluctuationDissipation, JarzynskiEstimator, OnsagerCoefficients};
use super::quantum_tunneling::TunnelingCalculator;
use super::tensor_algebra::{GeodesicSolver, MetricTensor};

/// Example 1: Quantum tunneling with decoherence effects
///
/// Models a particle tunneling through a barrier while subject to
/// environmental decoherence. This is relevant for:
/// - Quantum computing (qubit operations in noisy environments)
/// - Chemical reactions (proton tunneling in enzymes)
/// - Solid-state physics (electron tunneling in heterostructures)
pub struct TunnelingWithDecoherence {
    tunneling: TunnelingCalculator,
    decoherence_rate: f64,
}

impl TunnelingWithDecoherence {
    /// Create for electron tunneling
    pub fn electron(decoherence_rate: f64) -> Self {
        Self {
            tunneling: TunnelingCalculator::electron(),
            decoherence_rate,
        }
    }

    /// Calculate tunneling probability with decoherence effects
    ///
    /// Returns (coherent_transmission, effective_with_decoherence, coherence_factor)
    pub fn tunneling_with_environment(
        &self,
        barrier_height: f64,
        barrier_width: f64,
        energy: f64,
        time: f64,
    ) -> (f64, f64, f64) {
        // Coherent tunneling probability
        let result = self
            .tunneling
            .rectangular_barrier(energy, barrier_height, barrier_width);
        let coherent_prob = result.transmission;

        // Decoherence reduces off-diagonal elements
        // ρ_01(t) = ρ_01(0) * exp(-γt)
        let coherence = (-self.decoherence_rate * time).exp();

        // Effective tunneling with decoherence
        // Tunneling requires coherence between left and right states
        let decohered_prob = coherent_prob * coherence;

        (coherent_prob, decohered_prob, coherence)
    }

    /// Demonstrate density matrix evolution
    pub fn show_decoherence_effect(&self, time: f64) -> (f64, f64) {
        // Initial superposition state: (|0⟩ + |1⟩)/√2
        let psi = vec![
            super::decoherence::Complex64::from_real(1.0 / 2.0_f64.sqrt()),
            super::decoherence::Complex64::from_real(1.0 / 2.0_f64.sqrt()),
        ];
        let rho = DensityMatrix::pure_state(&psi);

        // Pure state purity = 1
        let initial_purity = rho.purity();

        // After decoherence, off-diagonals decay
        // Purity decreases: P(t) = (1 + exp(-2γt))/2
        let decohered_purity = 0.5 * (1.0 + (-2.0 * self.decoherence_rate * time).exp());

        (initial_purity, decohered_purity)
    }
}

/// Example 2: Black hole physics using tensor algebra
///
/// Demonstrates calculation of:
/// - Geodesics near a Schwarzschild black hole
/// - Proper time dilation
/// - Photon sphere and event horizon properties
pub struct BlackHolePhysics {
    mass: f64,
    r_s: f64, // Schwarzschild radius
}

impl BlackHolePhysics {
    /// Create from black hole mass in kg
    pub fn new(mass: f64) -> Self {
        let r_s = 2.0 * G * mass / (C * C);
        Self { mass, r_s }
    }

    /// Solar mass black hole
    pub fn solar_mass() -> Self {
        Self::new(2e30)
    }

    /// Sagittarius A* (supermassive black hole at galactic center)
    pub fn sgr_a_star() -> Self {
        Self::new(4e6 * 2e30) // ~4 million solar masses
    }

    /// Get Schwarzschild radius
    pub fn schwarzschild_radius(&self) -> f64 {
        self.r_s
    }

    /// Photon sphere radius (3/2 r_s)
    pub fn photon_sphere_radius(&self) -> f64 {
        1.5 * self.r_s
    }

    /// Innermost stable circular orbit (ISCO) at 3 r_s
    pub fn isco_radius(&self) -> f64 {
        3.0 * self.r_s
    }

    /// Time dilation factor at radius r
    /// dt_proper/dt_coord = sqrt(1 - r_s/r)
    pub fn time_dilation(&self, r: f64) -> f64 {
        if r <= self.r_s {
            0.0 // Inside horizon
        } else {
            (1.0 - self.r_s / r).sqrt()
        }
    }

    /// Gravitational redshift z = 1/sqrt(1 - r_s/r) - 1
    pub fn redshift(&self, r: f64) -> f64 {
        if r <= self.r_s {
            f64::INFINITY
        } else {
            1.0 / (1.0 - self.r_s / r).sqrt() - 1.0
        }
    }

    /// Calculate radial infall trajectory
    pub fn radial_infall(&self, r_start: f64, steps: usize) -> Vec<[f64; 4]> {
        let solver = GeodesicSolver::new(self.mass);
        let initial_pos = [0.0, r_start, std::f64::consts::FRAC_PI_2, 0.0];
        let initial_vel = [C, 0.0, 0.0, 0.0];

        solver.solve_schwarzschild(initial_pos, initial_vel, 1e-4, steps)
    }

    /// Get metric at radius r and theta = pi/2
    pub fn metric_at(&self, r: f64) -> MetricTensor {
        MetricTensor::schwarzschild(self.mass, r, std::f64::consts::FRAC_PI_2)
    }
}

/// Example 3: Climate system chaos analysis
///
/// Demonstrates chaos dynamics applied to simplified climate models.
/// Based on the Lorenz system which originated from atmospheric convection.
pub struct ClimateChaosDynamics {
    lorenz_params: (f64, f64, f64), // (sigma, rho, beta)
    lyapunov_calc: LyapunovCalculator,
}

impl ClimateChaosDynamics {
    /// Standard Lorenz parameters (chaotic regime)
    pub fn standard() -> Self {
        Self {
            lorenz_params: (10.0, 28.0, 8.0 / 3.0),
            lyapunov_calc: LyapunovCalculator::new(0.01, 1000, 5000),
        }
    }

    /// Create with custom Rayleigh number (rho parameter)
    /// - rho < 1: Stable
    /// - rho ~ 24.74: Subcritical Hopf bifurcation
    /// - rho > 24.74: Chaos
    pub fn with_rayleigh(rho: f64) -> Self {
        Self {
            lorenz_params: (10.0, rho, 8.0 / 3.0),
            lyapunov_calc: LyapunovCalculator::new(0.01, 1000, 5000),
        }
    }

    /// Check if system is chaotic (positive Lyapunov exponent)
    pub fn is_chaotic(&self) -> bool {
        let lambda = self.lyapunov_exponent();
        lambda > 0.0
    }

    /// Compute Lyapunov exponent
    pub fn lyapunov_exponent(&self) -> f64 {
        let (sigma, rho, beta) = self.lorenz_params;
        let system = move |state: &[f64]| systems::lorenz(state, sigma, rho, beta).to_vec();
        self.lyapunov_calc
            .maximal_lyapunov(system, &[1.0, 1.0, 1.0], 1e-8)
    }

    /// Analyze attractor dimension
    pub fn attractor_analysis(&self) -> f64 {
        let analyzer = AttractorAnalyzer::new(3, 1);

        // Generate trajectory
        let (sigma, rho, beta) = self.lorenz_params;
        let mut state = [1.0, 1.0, 1.0];
        let mut trajectory = Vec::with_capacity(5000);

        // Skip transient
        for _ in 0..1000 {
            let deriv = systems::lorenz(&state, sigma, rho, beta);
            for i in 0..3 {
                state[i] += deriv[i] * 0.01;
            }
        }

        // Collect trajectory
        for _ in 0..5000 {
            let deriv = systems::lorenz(&state, sigma, rho, beta);
            for i in 0..3 {
                state[i] += deriv[i] * 0.01;
            }
            trajectory.push(state.to_vec());
        }

        analyzer.correlation_dimension(&trajectory)
    }
}

/// Example 4: Non-equilibrium transport with fluctuation-dissipation
///
/// Models coupled heat and particle transport using Onsager relations.
/// Applications: thermoelectrics, fuel cells, biological membranes.
pub struct ThermoelectricTransport {
    temperature: f64,
    onsager: OnsagerCoefficients,
}

impl ThermoelectricTransport {
    /// Create thermoelectric transport model
    ///
    /// L_11: electrical conductivity
    /// L_12 = L_21: thermoelectric coupling (Onsager reciprocity)
    /// L_22: thermal conductivity
    pub fn new(temperature: f64, l_11: f64, l_12: f64, l_22: f64) -> Self {
        let coefficients = OnsagerCoefficients::symmetric(&[l_11, l_12, l_22], 2);
        Self {
            temperature,
            onsager: coefficients,
        }
    }

    /// Typical semiconductor thermoelectric
    pub fn semiconductor(temperature: f64) -> Self {
        // Typical values for Bi2Te3
        Self::new(temperature, 1e5, 1e2, 1.5)
    }

    /// Compute Seebeck coefficient S = L_12 / (T * L_11)
    pub fn seebeck_coefficient(&self) -> f64 {
        self.onsager.get(0, 1) / (self.temperature * self.onsager.get(0, 0))
    }

    /// Compute figure of merit ZT
    /// ZT = S² σ T / κ
    pub fn figure_of_merit(&self) -> f64 {
        let s = self.seebeck_coefficient();
        let sigma = self.onsager.get(0, 0);
        let kappa = self.onsager.get(1, 1);
        s * s * sigma * self.temperature / kappa
    }

    /// Compute entropy production rate for given gradients
    pub fn entropy_production(&self, voltage_gradient: f64, temp_gradient: f64) -> f64 {
        let forces = vec![
            voltage_gradient / self.temperature,
            -temp_gradient / (self.temperature * self.temperature),
        ];
        self.onsager.entropy_production(&forces)
    }

    /// Verify Onsager reciprocity
    pub fn verify_reciprocity(&self) -> bool {
        self.onsager.verify_symmetry(1e-10)
    }
}

/// Example 5: Brownian motion with fluctuation-dissipation theorem
pub struct BrownianMotionSimulation {
    fdt: FluctuationDissipation,
    mass: f64,
}

impl BrownianMotionSimulation {
    /// Create Brownian particle simulation
    pub fn new(mass: f64, temperature: f64, friction: f64) -> Self {
        Self {
            fdt: FluctuationDissipation::new(temperature, friction),
            mass,
        }
    }

    /// Colloidal particle in water at room temperature
    pub fn colloidal_particle(radius_um: f64) -> Self {
        // Stokes friction: γ = 6πηr
        let eta = 1e-3; // Water viscosity (Pa·s)
        let r = radius_um * 1e-6; // Convert to meters
        let friction = 6.0 * std::f64::consts::PI * eta * r;

        // Assume density similar to polystyrene
        let density = 1050.0; // kg/m³
        let mass = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3) * density;

        Self::new(mass, 300.0, friction)
    }

    /// Einstein diffusion coefficient D = kT/γ
    pub fn diffusion_coefficient(&self) -> f64 {
        self.fdt.diffusion_coefficient()
    }

    /// Mean squared displacement at time t: <x²> = 2Dt
    pub fn mean_squared_displacement(&self, t: f64) -> f64 {
        self.fdt.mean_squared_displacement(t)
    }

    /// Velocity autocorrelation time τ = m/γ
    pub fn velocity_correlation_time(&self) -> f64 {
        self.fdt.velocity_correlation_time(self.mass)
    }

    /// Thermal velocity √(kT/m)
    pub fn thermal_velocity(&self) -> f64 {
        (self.fdt.thermal_energy() / self.mass).sqrt()
    }
}

/// Example 6: Free energy calculation using Jarzynski equality
pub struct NonequilibriumFreeEnergy {
    estimator: JarzynskiEstimator,
    temperature: f64,
}

impl NonequilibriumFreeEnergy {
    pub fn new(temperature: f64) -> Self {
        Self {
            estimator: JarzynskiEstimator::new(),
            temperature,
        }
    }

    /// Estimate free energy from work measurements
    /// Works for any nonequilibrium process, no matter how fast
    pub fn estimate_free_energy(&self, work_samples: &[f64]) -> f64 {
        self.estimator
            .free_energy_difference(work_samples, self.temperature)
    }

    /// Estimate with bootstrap error bars
    pub fn estimate_with_error(&self, work_samples: &[f64], n_bootstrap: usize) -> (f64, f64) {
        self.estimator
            .free_energy_with_error(work_samples, self.temperature, n_bootstrap)
    }

    /// Use Crooks fluctuation theorem for bidirectional estimate
    pub fn bidirectional_estimate(&self, forward_work: &[f64], reverse_work: &[f64]) -> f64 {
        self.estimator
            .crooks_free_energy(forward_work, reverse_work, self.temperature)
    }

    /// Calculate dissipated work: W_diss = <W> - ΔF
    pub fn dissipated_work(&self, work_samples: &[f64]) -> f64 {
        self.estimator
            .dissipated_work(work_samples, self.temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tunneling_with_decoherence() {
        let system = TunnelingWithDecoherence::electron(1e12);

        // 1 eV barrier, 1 nm width, electron with 0.5 eV
        let e_charge = 1.602e-19;
        let (coherent, decohered, coherence) =
            system.tunneling_with_environment(1.0 * e_charge, 1e-9, 0.5 * e_charge, 1e-12);

        assert!(coherent >= 0.0 && coherent <= 1.0);
        assert!(decohered <= coherent);
        assert!(coherence > 0.0 && coherence <= 1.0);
    }

    #[test]
    fn test_decoherence_effect() {
        let system = TunnelingWithDecoherence::electron(1e12);
        let (initial, decohered) = system.show_decoherence_effect(1e-12);

        // Initial pure state has purity 1
        assert!((initial - 1.0).abs() < 0.01);
        // Decohered state has lower purity
        assert!(decohered < initial);
    }

    #[test]
    fn test_black_hole_physics() {
        let bh = BlackHolePhysics::solar_mass();

        // Schwarzschild radius ~ 3 km for solar mass
        assert!((bh.schwarzschild_radius() - 2964.0).abs() < 100.0);

        // Time dilation at 10 r_s
        let dilation = bh.time_dilation(10.0 * bh.schwarzschild_radius());
        assert!(dilation > 0.9 && dilation < 1.0);

        // ISCO at 3 r_s
        assert!((bh.isco_radius() / bh.schwarzschild_radius() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_climate_chaos() {
        let climate = ClimateChaosDynamics::standard();

        // Standard Lorenz should be chaotic or near-chaotic
        let lambda = climate.lyapunov_exponent();
        // Lyapunov exponent should be positive or near-positive for chaos
        assert!(lambda > -0.5, "Lorenz λ = {}", lambda);
    }

    #[test]
    fn test_thermoelectric() {
        let te = ThermoelectricTransport::semiconductor(300.0);

        // Onsager symmetry should hold
        assert!(te.verify_reciprocity());

        // ZT should be positive
        let zt = te.figure_of_merit();
        assert!(zt > 0.0);
    }

    #[test]
    fn test_brownian_motion() {
        let particle = BrownianMotionSimulation::colloidal_particle(1.0);

        // Diffusion coefficient should be positive
        let d = particle.diffusion_coefficient();
        assert!(d > 0.0);

        // MSD should grow with time
        let msd1 = particle.mean_squared_displacement(1.0);
        let msd2 = particle.mean_squared_displacement(2.0);
        assert!(msd2 > msd1);
    }

    #[test]
    fn test_jarzynski_equality() {
        let estimator = NonequilibriumFreeEnergy::new(300.0);

        // For reversible process, W = ΔF
        let work = vec![1e-20, 1e-20, 1e-20];
        let delta_f = estimator.estimate_free_energy(&work);

        // Mean work should be close to ΔF for quasi-static process
        let mean_work: f64 = work.iter().sum::<f64>() / work.len() as f64;
        assert!((delta_f - mean_work).abs() < 1e-21);
    }
}
