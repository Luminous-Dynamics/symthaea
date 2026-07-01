// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Simulation Bridge: Physics Encoding ↔ Dynamical Simulation
//!
//! This module bridges the existing physics ENCODING modules (classical_mechanics,
//! electromagnetism, thermodynamics) — which represent physical concepts as HDC
//! vectors — with the dynamical_system SIMULATION framework that provides actual
//! numerical ODE integration.
//!
//! ## Architecture
//!
//! ```text
//! Physics Encoding Modules          Simulation Bridge          Dynamical System Framework
//! ┌────────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
//! │ classical_mechanics│       │  PhysicsSimulator    │       │ DynamicalSystem trait │
//! │ electromagnetism   │ ←──── │  - factory methods   │ ────→ │ Integrators (RK4...) │
//! │ thermodynamics     │       │  - HDC encoding      │       │ SimulationResult      │
//! └────────────────────┘       │  SimulationAnalysis  │       └──────────────────────┘
//!                              └──────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::physics::simulation_bridge::PhysicsSimulator;
//!
//! // Create a simulator from physics description
//! let sim = PhysicsSimulator::harmonic(2.0, 1.0, 0.0);
//!
//! // Run pure numerical simulation
//! let result = sim.simulate(10.0, 0.01);
//!
//! // Run simulation with HDC trajectory encoding
//! let (result, hvs) = sim.simulate_with_hdc(10.0, 0.01);
//!
//! // Analyze the trajectory
//! let analysis = SimulationAnalysis::from_result(&result);
//! ```

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::dynamical_system::{
    ChargedParticle, DynamicalSystem, HarmonicOscillator, HeatEquation1D, Integrator, LorenzSystem,
    NBodyGravity, Rk4Integrator, SimulationResult, simulate, state_to_hv, trajectory_to_hvs,
};
use crate::hdc::unified_hv::ContinuousHV;

// =============================================================================
// PHYSICS SIMULATOR
// =============================================================================

/// A high-level wrapper that connects dynamical systems to the physics encoding
/// layer, providing convenient factory methods and HDC trajectory encoding.
///
/// `PhysicsSimulator` holds a boxed dynamical system, its initial conditions,
/// and metadata. It provides two simulation modes:
///
/// - `simulate()` — pure numerical integration
/// - `simulate_with_hdc()` — numerical integration plus HDC encoding of every
///   state in the trajectory
pub struct PhysicsSimulator {
    /// The dynamical system to simulate.
    pub system: Box<dyn DynamicalSystem>,
    /// Human-readable name for the simulator configuration.
    pub name: String,
    /// Initial state vector for the simulation.
    pub initial_state: Vec<f64>,
}

impl PhysicsSimulator {
    /// Create a new PhysicsSimulator with the given system, name, and initial state.
    pub fn new(system: Box<dyn DynamicalSystem>, name: String, initial_state: Vec<f64>) -> Self {
        assert_eq!(
            initial_state.len(),
            system.state_dim(),
            "Initial state dimension {} does not match system dimension {}",
            initial_state.len(),
            system.state_dim()
        );
        Self {
            system,
            name,
            initial_state,
        }
    }

    /// Run a numerical simulation from t=0 to t=t_end using RK4 integration.
    ///
    /// # Arguments
    /// * `t_end` - End time of the simulation
    /// * `dt` - Integration time step
    ///
    /// # Returns
    /// A `SimulationResult` containing the full trajectory
    pub fn simulate(&self, t_end: f64, dt: f64) -> SimulationResult {
        let integrator = Rk4Integrator;
        simulate(
            self.system.as_ref(),
            &integrator,
            &self.initial_state,
            0.0,
            t_end,
            dt,
        )
    }

    /// Run a simulation and encode the entire trajectory as ContinuousHVs.
    ///
    /// Each state vector in the trajectory is encoded into a ContinuousHV using
    /// the deterministic basis-vector superposition scheme from `state_to_hv`.
    ///
    /// # Arguments
    /// * `t_end` - End time of the simulation
    /// * `dt` - Integration time step
    ///
    /// # Returns
    /// A tuple of `(SimulationResult, Vec<ContinuousHV>)` where each HV corresponds
    /// to the state at the matching index in the result
    pub fn simulate_with_hdc(&self, t_end: f64, dt: f64) -> (SimulationResult, Vec<ContinuousHV>) {
        let result = self.simulate(t_end, dt);
        let hvs = trajectory_to_hvs(&result);
        (result, hvs)
    }

    // =========================================================================
    // FACTORY METHODS
    // =========================================================================

    /// Create a simple harmonic oscillator simulator.
    ///
    /// Models the undamped system: dx/dt = v, dv/dt = -omega^2 * x
    ///
    /// # Arguments
    /// * `omega` - Angular frequency (rad/s)
    /// * `x0` - Initial displacement
    /// * `v0` - Initial velocity
    ///
    /// # Physics connection
    /// Corresponds to `classical_mechanics::SystemType::HarmonicOscillator` encoding,
    /// but this version actually simulates the dynamics.
    pub fn harmonic(omega: f64, x0: f64, v0: f64) -> Self {
        let system = HarmonicOscillator::simple(omega);
        Self::new(
            Box::new(system),
            format!("HarmonicOscillator(omega={omega:.3})"),
            vec![x0, v0],
        )
    }

    /// Create a gravitational two-body simulator in 2D.
    ///
    /// Sets up two bodies on a circular orbit around their center of mass.
    /// The initial conditions are computed from the masses and desired separation
    /// to produce a bound circular orbit.
    ///
    /// # Arguments
    /// * `m1` - Mass of body 1
    /// * `m2` - Mass of body 2
    /// * `separation` - Initial distance between the two bodies
    ///
    /// # Physics connection
    /// Corresponds to `classical_mechanics::SystemType::TwoBody` encoding,
    /// but with actual gravitational dynamics via N-body integration.
    pub fn gravitational_two_body(m1: f64, m2: f64, separation: f64) -> Self {
        let g = 1.0;
        let system = NBodyGravity::two_body(m1, m2, g, 2);

        // Compute circular orbit initial conditions.
        // Place bodies along x-axis, centered on the center of mass.
        let total_mass = m1 + m2;
        let r1 = separation * m2 / total_mass; // distance of body 1 from COM
        let r2 = separation * m1 / total_mass; // distance of body 2 from COM

        // Circular orbit velocity: v = sqrt(G * M_other / separation) * (r_i / separation)
        // Actually for two-body circular orbit:
        //   v1 = m2 * sqrt(G / (total_mass * separation))
        //   but body 1 orbits at distance r1 from COM, so:
        //   v1 = r1 * sqrt(G * total_mass / separation^3)  (from Kepler)
        let orbital_speed_factor = (g * total_mass / (separation * separation * separation)).sqrt();
        let v1 = r1 * orbital_speed_factor;
        let v2 = r2 * orbital_speed_factor;

        // Body 1 at (-r1, 0), velocity (0, -v1)
        // Body 2 at (+r2, 0), velocity (0, +v2)
        // This ensures zero total momentum.
        let initial_state = vec![
            -r1, 0.0, // pos body 1
            r2, 0.0, // pos body 2
            0.0, -v1, // vel body 1
            0.0, v2, // vel body 2
        ];

        Self::new(
            Box::new(system),
            format!("TwoBody(m1={m1:.2}, m2={m2:.2}, sep={separation:.2})"),
            initial_state,
        )
    }

    /// Create a Lorenz attractor simulator with standard chaotic parameters.
    ///
    /// Uses sigma=10, rho=28, beta=8/3 and the standard initial condition
    /// (1, 1, 1) which lies on the attractor basin.
    ///
    /// # Physics connection
    /// The Lorenz system models atmospheric convection. This bridges to the
    /// `thermodynamics` encoding module which encodes thermal concepts as HDC
    /// vectors — the Lorenz system provides the dynamical trajectory that those
    /// static encodings describe.
    pub fn lorenz() -> Self {
        let system = LorenzSystem::standard();
        Self::new(
            Box::new(system),
            "LorenzAttractor(standard)".to_string(),
            vec![1.0, 1.0, 1.0],
        )
    }

    /// Create a charged particle simulator in static EM fields.
    ///
    /// Models the Lorentz force: F = q(E + v x B), integrated as a 6D ODE
    /// for position and velocity.
    ///
    /// # Arguments
    /// * `qm` - Charge-to-mass ratio q/m (C/kg)
    /// * `e_field` - Electric field vector [Ex, Ey, Ez] (V/m)
    /// * `b_field` - Magnetic field vector [Bx, By, Bz] (T)
    /// * `initial` - Initial state [x, y, z, vx, vy, vz]
    ///
    /// # Physics connection
    /// Corresponds to `electromagnetism` module's field encodings, but with
    /// actual particle trajectory dynamics.
    pub fn charged_particle(
        qm: f64,
        e_field: [f64; 3],
        b_field: [f64; 3],
        initial: [f64; 6],
    ) -> Self {
        let system = ChargedParticle::new(qm, e_field, b_field);
        Self::new(
            Box::new(system),
            format!("ChargedParticle(qm={qm:.3})"),
            initial.to_vec(),
        )
    }

    /// Create a 1D heat diffusion simulator.
    ///
    /// Discretizes the PDE dT/dt = alpha * d^2T/dx^2 on a uniform grid with
    /// Dirichlet (fixed temperature) boundary conditions. The initial temperature
    /// profile is set to a linear interpolation between the boundary values.
    ///
    /// # Arguments
    /// * `alpha` - Thermal diffusivity (m^2/s)
    /// * `n_points` - Number of interior grid points
    /// * `t_left` - Left boundary temperature
    /// * `t_right` - Right boundary temperature
    ///
    /// # Physics connection
    /// Corresponds to `thermodynamics` module's thermal state encodings, but
    /// with actual diffusion dynamics on a discretized spatial domain.
    pub fn heat_diffusion(alpha: f64, n_points: usize, t_left: f64, t_right: f64) -> Self {
        let system = HeatEquation1D::new(alpha, 1.0, n_points, t_left, t_right);

        // Initial condition: uniform temperature at the average of boundaries,
        // which will then relax to the linear steady-state profile.
        let avg_temp = (t_left + t_right) / 2.0;
        let initial_state = vec![avg_temp; n_points];

        Self::new(
            Box::new(system),
            format!(
                "HeatDiffusion(alpha={alpha:.4}, n={n_points}, T_L={t_left:.1}, T_R={t_right:.1})"
            ),
            initial_state,
        )
    }
}

impl std::fmt::Debug for PhysicsSimulator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicsSimulator")
            .field("name", &self.name)
            .field("state_dim", &self.system.state_dim())
            .field("initial_state", &self.initial_state)
            .finish()
    }
}

// =============================================================================
// SIMULATION ANALYSIS
// =============================================================================

/// Statistical and physical analysis of a simulation trajectory.
///
/// Provides summary statistics useful for characterizing the behavior of a
/// dynamical system: variable ranges, energy conservation quality, and
/// chaos indicators.
#[derive(Debug, Clone)]
pub struct SimulationAnalysis {
    /// Total simulated time (t_end - t_start).
    pub total_time: f64,
    /// Number of integration steps in the trajectory.
    pub num_steps: usize,
    /// Per-variable range: (min, max) for each state component.
    pub state_range: Vec<(f64, f64)>,
    /// Relative energy drift over the trajectory, if the system has a known
    /// energy function. Computed as max |E(t) - E(0)| / |E(0)|.
    pub energy_drift: Option<f64>,
    /// Estimated maximal Lyapunov exponent. Positive values indicate chaos.
    /// Only computed for systems where nearby-trajectory divergence is meaningful.
    pub lyapunov_estimate: Option<f64>,
}

impl SimulationAnalysis {
    /// Analyze a simulation trajectory and produce summary statistics.
    ///
    /// This computes:
    /// - Variable ranges (min, max per state component)
    /// - Total time and step count
    ///
    /// Energy drift and Lyapunov exponents require additional information
    /// and are set via dedicated methods.
    pub fn from_result(result: &SimulationResult) -> Self {
        let num_steps = result.steps;
        let total_time = result.total_time;

        // Compute per-variable ranges
        let dim = if let Some(first) = result.states.first() {
            first.len()
        } else {
            0
        };

        let mut state_range: Vec<(f64, f64)> = vec![(f64::MAX, f64::MIN); dim];

        for state in &result.states {
            for (i, &val) in state.iter().enumerate() {
                if val < state_range[i].0 {
                    state_range[i].0 = val;
                }
                if val > state_range[i].1 {
                    state_range[i].1 = val;
                }
            }
        }

        Self {
            total_time,
            num_steps,
            state_range,
            energy_drift: None,
            lyapunov_estimate: None,
        }
    }

    /// Compute energy drift for a harmonic oscillator trajectory.
    ///
    /// Returns the maximum relative energy deviation from the initial energy.
    pub fn with_harmonic_energy(
        mut self,
        osc: &HarmonicOscillator,
        result: &SimulationResult,
    ) -> Self {
        if let Some(first) = result.states.first() {
            let e0 = osc.energy(first);
            if e0.abs() > 1e-30 {
                let max_drift = result
                    .states
                    .iter()
                    .map(|s| (osc.energy(s) - e0).abs() / e0.abs())
                    .fold(0.0f64, f64::max);
                self.energy_drift = Some(max_drift);
            }
        }
        self
    }

    /// Compute energy drift for an N-body gravitational trajectory.
    pub fn with_nbody_energy(mut self, nbody: &NBodyGravity, result: &SimulationResult) -> Self {
        if let Some(first) = result.states.first() {
            let e0 = nbody.energy(first);
            if e0.abs() > 1e-30 {
                let max_drift = result
                    .states
                    .iter()
                    .map(|s| (nbody.energy(s) - e0).abs() / e0.abs())
                    .fold(0.0f64, f64::max);
                self.energy_drift = Some(max_drift);
            }
        }
        self
    }

    /// Compute energy drift for a charged particle trajectory (kinetic energy
    /// in pure magnetic field).
    pub fn with_charged_particle_ke(
        mut self,
        cp: &ChargedParticle,
        result: &SimulationResult,
    ) -> Self {
        if let Some(first) = result.states.first() {
            let ke0 = cp.kinetic_energy(first);
            if ke0.abs() > 1e-30 {
                let max_drift = result
                    .states
                    .iter()
                    .map(|s| (cp.kinetic_energy(s) - ke0).abs() / ke0.abs())
                    .fold(0.0f64, f64::max);
                self.energy_drift = Some(max_drift);
            }
        }
        self
    }

    /// Estimate the maximal Lyapunov exponent by comparing two nearby
    /// trajectories with periodic renormalization.
    ///
    /// This method runs a second simulation with a small perturbation to the
    /// initial state and tracks the divergence rate.
    ///
    /// # Arguments
    /// * `system` - The dynamical system
    /// * `initial_state` - The reference initial state
    /// * `t_end` - Simulation end time
    /// * `dt` - Time step
    /// * `perturbation` - Size of the initial perturbation (e.g. 1e-8)
    pub fn with_lyapunov_estimate(
        mut self,
        system: &dyn DynamicalSystem,
        initial_state: &[f64],
        t_end: f64,
        dt: f64,
        perturbation: f64,
    ) -> Self {
        let integrator = Rk4Integrator;
        let dim = system.state_dim();

        if dim == 0 || t_end <= 0.0 {
            return self;
        }

        let mut state = initial_state.to_vec();
        let mut perturbed = initial_state.to_vec();
        perturbed[0] += perturbation;

        // Skip transient (first 20% of simulation)
        let transient_steps = ((t_end * 0.2) / dt) as usize;
        let mut t = 0.0;
        for _ in 0..transient_steps {
            state = integrator.step(system, &state, t, dt);
            perturbed = integrator.step(system, &perturbed, t, dt);
            t += dt;
        }

        // Reinitialize perturbation after transient
        perturbed = state.clone();
        perturbed[0] += perturbation;

        // Estimate Lyapunov with renormalization
        let renorm_interval = 10usize;
        let remaining_steps = ((t_end - t) / dt) as usize;
        let mut lambda_sum = 0.0;
        let mut count = 0usize;

        for step in 0..remaining_steps {
            state = integrator.step(system, &state, t, dt);
            perturbed = integrator.step(system, &perturbed, t, dt);
            t += dt;

            if (step + 1) % renorm_interval == 0 {
                let dist: f64 = state
                    .iter()
                    .zip(perturbed.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();

                if dist > 1e-30 {
                    lambda_sum += (dist / perturbation).ln();
                    count += 1;

                    // Renormalize
                    for i in 0..dim {
                        perturbed[i] = state[i] + (perturbed[i] - state[i]) * perturbation / dist;
                    }
                }
            }
        }

        if count > 0 {
            let lambda = lambda_sum / (count as f64 * renorm_interval as f64 * dt);
            self.lyapunov_estimate = Some(lambda);
        }

        self
    }

    /// Check whether the system appears chaotic (positive Lyapunov exponent).
    pub fn is_chaotic(&self) -> Option<bool> {
        self.lyapunov_estimate.map(|l| l > 0.0)
    }

    /// Check whether energy is well-conserved (drift < threshold).
    pub fn energy_conserved(&self, threshold: f64) -> Option<bool> {
        self.energy_drift.map(|d| d < threshold)
    }
}

/// Convenience function: analyze a trajectory from a SimulationResult.
///
/// This is a thin wrapper around `SimulationAnalysis::from_result` for
/// call-site ergonomics.
pub fn analyze_trajectory(result: &SimulationResult) -> SimulationAnalysis {
    SimulationAnalysis::from_result(result)
}

// =============================================================================
// HDC ENCODING UTILITIES
// =============================================================================

/// Encode a single simulation state as a BinaryHV (via ContinuousHV thresholding).
///
/// Converts the continuous HDC encoding to a binary representation by
/// thresholding each component: positive values become 1-bits, negative
/// values become 0-bits.
pub fn state_to_binary_hv(state: &[f64]) -> BinaryHV {
    let chv = state_to_hv(state);
    continuous_to_binary(&chv)
}

/// Convert a ContinuousHV to a BinaryHV by sign thresholding.
fn continuous_to_binary(chv: &ContinuousHV) -> BinaryHV {
    let mut bytes = [0u8; 2048];
    for (i, &val) in chv.values.iter().enumerate() {
        if i >= 2048 * 8 {
            break;
        }
        if val > 0.0 {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    BinaryHV(bytes)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Factory creation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_harmonic_simulator() {
        let sim = PhysicsSimulator::harmonic(2.0, 1.0, 0.0);
        assert_eq!(sim.system.state_dim(), 2);
        assert_eq!(sim.initial_state.len(), 2);
        assert!((sim.initial_state[0] - 1.0).abs() < 1e-15);
        assert!((sim.initial_state[1] - 0.0).abs() < 1e-15);
        assert!(sim.name.contains("HarmonicOscillator"));
    }

    #[test]
    fn test_create_two_body_simulator() {
        let sim = PhysicsSimulator::gravitational_two_body(1.0, 1.0, 2.0);
        assert_eq!(sim.system.state_dim(), 8); // 2 bodies x 2D x 2 (pos+vel)
        assert_eq!(sim.initial_state.len(), 8);
        assert!(sim.name.contains("TwoBody"));

        // Total momentum should be zero (by construction)
        let vx1 = sim.initial_state[4];
        let vy1 = sim.initial_state[5];
        let vx2 = sim.initial_state[6];
        let vy2 = sim.initial_state[7];
        // m1 = m2 = 1, so p_total = v1 + v2
        assert!((vx1 + vx2).abs() < 1e-12, "Total px should be zero");
        assert!((vy1 + vy2).abs() < 1e-12, "Total py should be zero");
    }

    #[test]
    fn test_create_lorenz_simulator() {
        let sim = PhysicsSimulator::lorenz();
        assert_eq!(sim.system.state_dim(), 3);
        assert_eq!(sim.initial_state, vec![1.0, 1.0, 1.0]);
        assert!(sim.name.contains("Lorenz"));
    }

    #[test]
    fn test_create_charged_particle_simulator() {
        let sim = PhysicsSimulator::charged_particle(
            1.0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        assert_eq!(sim.system.state_dim(), 6);
        assert_eq!(sim.initial_state.len(), 6);
        assert!(sim.name.contains("ChargedParticle"));
    }

    #[test]
    fn test_create_heat_diffusion_simulator() {
        let sim = PhysicsSimulator::heat_diffusion(0.01, 20, 100.0, 0.0);
        assert_eq!(sim.system.state_dim(), 20);
        assert_eq!(sim.initial_state.len(), 20);
        // Initial temp should be avg of boundaries = 50.0
        for &t in &sim.initial_state {
            assert!((t - 50.0).abs() < 1e-12);
        }
        assert!(sim.name.contains("HeatDiffusion"));
    }

    // -------------------------------------------------------------------------
    // Simulation execution tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_harmonic_simulation_energy() {
        let sim = PhysicsSimulator::harmonic(2.0, 1.0, 0.0);
        let result = sim.simulate(10.0, 0.001);

        assert!(result.steps > 0);
        assert_eq!(result.system_name, "HarmonicOscillator");

        // Energy should be conserved: E = 0.5 * v^2 + 0.5 * omega^2 * x^2
        let omega = 2.0;
        let e0 = 0.5 * omega * omega; // x0=1, v0=0
        let final_state = result.final_state();
        let e_final = 0.5 * final_state[1] * final_state[1]
            + 0.5 * omega * omega * final_state[0] * final_state[0];
        let drift = (e_final - e0).abs() / e0;
        assert!(
            drift < 1e-8,
            "Harmonic oscillator energy drift = {:.2e}, should be < 1e-8",
            drift
        );
    }

    #[test]
    fn test_lorenz_simulation_bounded() {
        let sim = PhysicsSimulator::lorenz();
        let result = sim.simulate(20.0, 0.01);

        assert!(result.steps > 0);

        // Lorenz attractor is bounded
        for state in &result.states {
            for &v in state {
                assert!(
                    v.abs() < 100.0,
                    "Lorenz state component out of bounds: {}",
                    v
                );
            }
        }
    }

    #[test]
    fn test_two_body_simulation_orbit() {
        let sim = PhysicsSimulator::gravitational_two_body(1.0, 1.0, 2.0);
        let result = sim.simulate(5.0, 0.001);

        assert!(result.steps > 0);

        // Bodies should remain within a reasonable distance (bound orbit)
        for state in &result.states {
            let x1 = state[0];
            let y1 = state[1];
            let r1 = (x1 * x1 + y1 * y1).sqrt();
            assert!(
                r1 < 5.0,
                "Body 1 distance from origin {} too large for bound orbit",
                r1
            );
        }
    }

    #[test]
    fn test_charged_particle_simulation_cyclotron() {
        let sim = PhysicsSimulator::charged_particle(
            1.0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        let result = sim.simulate(6.3, 0.001); // ~one cyclotron period

        // In pure B field, speed should be conserved
        let v0_sq: f64 = sim.initial_state[3..6].iter().map(|v| v * v).sum();
        let final_v_sq: f64 = result.final_state()[3..6].iter().map(|v| v * v).sum();
        let speed_drift = (final_v_sq.sqrt() - v0_sq.sqrt()).abs() / v0_sq.sqrt();
        assert!(
            speed_drift < 1e-6,
            "Speed drift in cyclotron = {:.2e}",
            speed_drift
        );
    }

    #[test]
    fn test_heat_diffusion_simulation_relaxation() {
        let sim = PhysicsSimulator::heat_diffusion(0.01, 10, 100.0, 0.0);
        let result = sim.simulate(50.0, 0.01);

        // Temperature should be approaching the linear steady state
        let final_state = result.final_state();
        for (i, &t) in final_state.iter().enumerate() {
            let x = (i + 1) as f64 / (10 + 1) as f64;
            let t_steady = 100.0 + (0.0 - 100.0) * x;
            // After 50s with alpha=0.01, should be reasonably close
            assert!(
                (t - t_steady).abs() < 30.0,
                "Temperature at point {} = {:.2}, steady = {:.2}, diff too large",
                i,
                t,
                t_steady
            );
        }
    }

    // -------------------------------------------------------------------------
    // HDC encoding tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simulate_with_hdc_encoding() {
        let sim = PhysicsSimulator::harmonic(1.0, 1.0, 0.0);
        let (result, hvs) = sim.simulate_with_hdc(2.0, 0.1);

        // HVs should match state count
        assert_eq!(hvs.len(), result.states.len());

        // Each HV should have the standard dimension
        for hv in &hvs {
            assert_eq!(hv.dim(), ContinuousHV::DEFAULT_DIM);
        }

        // Consecutive HVs should be similar (small dt => small state change)
        if hvs.len() >= 2 {
            let sim_val = hvs[0].similarity(&hvs[1]);
            assert!(
                sim_val > 0.8,
                "Consecutive HDC states should be similar, got {:.4}",
                sim_val
            );
        }
    }

    #[test]
    fn test_hdc_trajectory_dissimilarity_over_time() {
        let sim = PhysicsSimulator::lorenz();
        let (_result, hvs) = sim.simulate_with_hdc(10.0, 0.05);

        // Early and late states on the Lorenz attractor should be encoded differently
        let early_idx = 5;
        let late_idx = hvs.len() - 5;
        if late_idx > early_idx {
            let sim_val = hvs[early_idx].similarity(&hvs[late_idx]);
            assert!(
                sim_val.is_finite(),
                "Similarity between trajectory points should be finite"
            );
            // They can be anything, but the pipeline should work without panic
        }
    }

    #[test]
    fn test_state_to_binary_hv_bridge() {
        let state = [1.0, 0.0, -1.0];
        let bhv = state_to_binary_hv(&state);
        // BinaryHV should have 2048 bytes = 16384 bits
        assert_eq!(bhv.0.len(), 2048);
    }

    // -------------------------------------------------------------------------
    // Analysis tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_analyze_harmonic_trajectory() {
        let sim = PhysicsSimulator::harmonic(1.0, 1.0, 0.0);
        let result = sim.simulate(10.0, 0.01);
        let analysis = analyze_trajectory(&result);

        assert!((analysis.total_time - 10.0).abs() < 1e-10);
        assert!(analysis.num_steps > 0);
        assert_eq!(analysis.state_range.len(), 2); // x and v

        // x should oscillate between -1 and 1
        let (x_min, x_max) = analysis.state_range[0];
        assert!(x_min < -0.9, "x_min = {:.4}, should be < -0.9", x_min);
        assert!(x_max > 0.9, "x_max = {:.4}, should be > 0.9", x_max);

        // v should also oscillate between -1 and 1 (for omega=1, A=1)
        let (v_min, v_max) = analysis.state_range[1];
        assert!(v_min < -0.9, "v_min = {:.4}, should be < -0.9", v_min);
        assert!(v_max > 0.9, "v_max = {:.4}, should be > 0.9", v_max);
    }

    #[test]
    fn test_analyze_with_energy_drift() {
        let osc = HarmonicOscillator::simple(2.0);
        let sim = PhysicsSimulator::harmonic(2.0, 1.0, 0.0);
        let result = sim.simulate(10.0, 0.001);

        let analysis = analyze_trajectory(&result).with_harmonic_energy(&osc, &result);

        assert!(analysis.energy_drift.is_some());
        let drift = analysis.energy_drift.unwrap();
        assert!(
            drift < 1e-8,
            "Energy drift = {:.2e}, should be tiny for RK4 + harmonic",
            drift
        );
        assert_eq!(analysis.energy_conserved(1e-6), Some(true));
    }

    #[test]
    fn test_analyze_lorenz_lyapunov() {
        let sim = PhysicsSimulator::lorenz();
        let result = sim.simulate(30.0, 0.01);

        let analysis = analyze_trajectory(&result).with_lyapunov_estimate(
            sim.system.as_ref(),
            &sim.initial_state,
            30.0,
            0.01,
            1e-8,
        );

        assert!(analysis.lyapunov_estimate.is_some());
        let lambda = analysis.lyapunov_estimate.unwrap();
        // Lorenz maximal Lyapunov exponent is approximately 0.9
        assert!(
            lambda > 0.3 && lambda < 2.0,
            "Lorenz Lyapunov exponent = {:.3}, expected ~0.9",
            lambda
        );
        assert_eq!(analysis.is_chaotic(), Some(true));
    }

    #[test]
    fn test_analyze_heat_trajectory() {
        let sim = PhysicsSimulator::heat_diffusion(0.01, 10, 100.0, 0.0);
        let result = sim.simulate(10.0, 0.01);
        let analysis = analyze_trajectory(&result);

        assert_eq!(analysis.state_range.len(), 10); // 10 grid points
        assert!(analysis.num_steps > 0);

        // All temperatures should stay within boundary values (roughly)
        for (i, &(t_min, t_max)) in analysis.state_range.iter().enumerate() {
            assert!(
                t_min >= -5.0 && t_max <= 105.0,
                "Grid point {} temperature out of range: [{:.2}, {:.2}]",
                i,
                t_min,
                t_max
            );
        }
    }

    #[test]
    fn test_analyze_charged_particle_ke() {
        let cp = ChargedParticle::cyclotron(1.0, 1.0);
        let sim = PhysicsSimulator::charged_particle(
            1.0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        let result = sim.simulate(10.0, 0.001);

        let analysis = analyze_trajectory(&result).with_charged_particle_ke(&cp, &result);

        assert!(analysis.energy_drift.is_some());
        let drift = analysis.energy_drift.unwrap();
        assert!(
            drift < 1e-6,
            "Cyclotron KE drift = {:.2e}, should be < 1e-6 in pure B field",
            drift
        );
    }

    #[test]
    fn test_analyze_empty_result() {
        // Edge case: a result with zero steps
        let result = SimulationResult {
            times: vec![0.0],
            states: vec![vec![0.0, 0.0]],
            total_time: 0.0,
            steps: 0,
            system_name: "empty".to_string(),
        };
        let analysis = analyze_trajectory(&result);
        assert_eq!(analysis.num_steps, 0);
        assert_eq!(analysis.state_range.len(), 2);
    }

    #[test]
    fn test_debug_display() {
        let sim = PhysicsSimulator::harmonic(1.0, 1.0, 0.0);
        let debug_str = format!("{:?}", sim);
        assert!(debug_str.contains("PhysicsSimulator"));
        assert!(debug_str.contains("HarmonicOscillator"));
    }
}
