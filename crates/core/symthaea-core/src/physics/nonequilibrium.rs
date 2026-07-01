// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Non-equilibrium thermodynamics module
//!
//! This module implements non-equilibrium thermodynamics including:
//! - Onsager transport coefficients and reciprocal relations
//! - Fluctuation-dissipation theorem
//! - Jarzynski equality for free energy estimation
//! - Irreversible process modeling
//!
//! These tools are essential for understanding systems far from equilibrium,
//! including biological systems, chemical reactions, and transport phenomena.

use serde::{Deserialize, Serialize};

use super::constants::K_BOLTZMANN;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::ContinuousHV;

/// Onsager transport coefficients matrix
///
/// Relates thermodynamic fluxes J_i to thermodynamic forces X_j:
/// J_i = Σ L_ij X_j
///
/// The Onsager reciprocal relations state that L_ij = L_ji for systems
/// near equilibrium with time-reversal symmetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnsagerCoefficients {
    /// L_ij coefficient matrix (should be symmetric)
    coefficients: Vec<Vec<f64>>,
    /// Dimension of the system
    dim: usize,
}

impl OnsagerCoefficients {
    /// Create new Onsager coefficients matrix
    ///
    /// # Arguments
    /// * `coefficients` - Square matrix of transport coefficients
    ///
    /// # Panics
    /// Panics if the matrix is not square
    pub fn new(coefficients: Vec<Vec<f64>>) -> Self {
        let dim = coefficients.len();
        for row in &coefficients {
            assert_eq!(row.len(), dim, "Onsager matrix must be square");
        }
        Self { coefficients, dim }
    }

    /// Create diagonal Onsager matrix (uncoupled transport)
    pub fn diagonal(diagonal: &[f64]) -> Self {
        let dim = diagonal.len();
        let mut coefficients = vec![vec![0.0; dim]; dim];
        for (i, &val) in diagonal.iter().enumerate() {
            coefficients[i][i] = val;
        }
        Self { coefficients, dim }
    }

    /// Create symmetric Onsager matrix from upper triangular values
    pub fn symmetric(upper_triangular: &[f64], dim: usize) -> Self {
        assert_eq!(
            upper_triangular.len(),
            dim * (dim + 1) / 2,
            "Invalid number of elements for symmetric matrix"
        );

        let mut coefficients = vec![vec![0.0; dim]; dim];
        let mut idx = 0;
        for i in 0..dim {
            for j in i..dim {
                coefficients[i][j] = upper_triangular[idx];
                coefficients[j][i] = upper_triangular[idx];
                idx += 1;
            }
        }
        Self { coefficients, dim }
    }

    /// Compute fluxes from thermodynamic forces
    ///
    /// J_i = Σ L_ij X_j
    pub fn compute_fluxes(&self, forces: &[f64]) -> Vec<f64> {
        assert_eq!(forces.len(), self.dim, "Force vector dimension mismatch");

        let mut fluxes = vec![0.0; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                fluxes[i] += self.coefficients[i][j] * forces[j];
            }
        }
        fluxes
    }

    /// Compute entropy production rate
    ///
    /// σ = Σ J_i X_i = Σ_ij L_ij X_i X_j ≥ 0
    ///
    /// This is always non-negative for a thermodynamically consistent system
    /// (positive semi-definite Onsager matrix)
    pub fn entropy_production(&self, forces: &[f64]) -> f64 {
        let fluxes = self.compute_fluxes(forces);
        forces.iter().zip(fluxes.iter()).map(|(x, j)| x * j).sum()
    }

    /// Verify Onsager reciprocal relations: L_ij = L_ji
    ///
    /// Returns true if the matrix is symmetric within the given tolerance
    pub fn verify_symmetry(&self, tolerance: f64) -> bool {
        for i in 0..self.dim {
            for j in (i + 1)..self.dim {
                if (self.coefficients[i][j] - self.coefficients[j][i]).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Compute asymmetry measure
    ///
    /// Returns the maximum |L_ij - L_ji| across all pairs
    pub fn asymmetry_measure(&self) -> f64 {
        let mut max_asymmetry = 0.0;
        for i in 0..self.dim {
            for j in (i + 1)..self.dim {
                let asymmetry = (self.coefficients[i][j] - self.coefficients[j][i]).abs();
                if asymmetry > max_asymmetry {
                    max_asymmetry = asymmetry;
                }
            }
        }
        max_asymmetry
    }

    /// Check if the Onsager matrix is positive semi-definite
    ///
    /// Required for thermodynamic consistency (σ ≥ 0)
    pub fn is_positive_semidefinite(&self) -> bool {
        // Check via Sylvester's criterion (all principal minors ≥ 0)
        // For simplicity, we check eigenvalues are non-negative
        // For small matrices, use direct computation
        match self.dim {
            1 => self.coefficients[0][0] >= 0.0,
            2 => {
                // For 2x2: det ≥ 0 and trace ≥ 0
                let trace = self.coefficients[0][0] + self.coefficients[1][1];
                let det = self.coefficients[0][0] * self.coefficients[1][1]
                    - self.coefficients[0][1] * self.coefficients[1][0];
                trace >= 0.0 && det >= 0.0
            }
            _ => {
                // For larger matrices, check if all diagonal elements are positive
                // and use Gershgorin circle theorem as approximation
                for i in 0..self.dim {
                    let mut row_sum = 0.0;
                    for j in 0..self.dim {
                        if i != j {
                            row_sum += self.coefficients[i][j].abs();
                        }
                    }
                    if self.coefficients[i][i] < -row_sum {
                        return false;
                    }
                }
                true
            }
        }
    }

    /// Get coefficient L_ij
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.coefficients[i][j]
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Fluctuation-dissipation theorem implementation
///
/// Relates the response of a system to fluctuations in equilibrium.
/// Key result: D = k_B T / γ (Einstein relation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluctuationDissipation {
    /// Temperature (K)
    pub temperature: f64,
    /// Friction coefficient (kg/s)
    pub friction: f64,
}

impl FluctuationDissipation {
    /// Create new fluctuation-dissipation calculator
    pub fn new(temperature: f64, friction: f64) -> Self {
        Self {
            temperature,
            friction,
        }
    }

    /// Einstein relation: diffusion coefficient D = k_B T / γ
    ///
    /// Returns the diffusion coefficient in m²/s
    pub fn diffusion_coefficient(&self) -> f64 {
        K_BOLTZMANN * self.temperature / self.friction
    }

    /// Thermal energy k_B T
    pub fn thermal_energy(&self) -> f64 {
        K_BOLTZMANN * self.temperature
    }

    /// Noise amplitude for Langevin dynamics
    ///
    /// For overdamped Langevin: √(2 k_B T γ / dt)
    pub fn noise_amplitude(&self, dt: f64) -> f64 {
        (2.0 * K_BOLTZMANN * self.temperature * self.friction / dt).sqrt()
    }

    /// Noise amplitude for underdamped Langevin (per unit mass)
    ///
    /// √(2 γ k_B T / m / dt)
    pub fn noise_amplitude_underdamped(&self, mass: f64, dt: f64) -> f64 {
        (2.0 * self.friction * K_BOLTZMANN * self.temperature / mass / dt).sqrt()
    }

    /// Velocity autocorrelation decay time
    ///
    /// τ = m / γ
    pub fn velocity_correlation_time(&self, mass: f64) -> f64 {
        mass / self.friction
    }

    /// Mean squared displacement at time t (overdamped limit)
    ///
    /// <x²(t)> = 2 D t = 2 k_B T t / γ
    pub fn mean_squared_displacement(&self, t: f64) -> f64 {
        2.0 * self.diffusion_coefficient() * t
    }

    /// Generalized susceptibility from fluctuations (Kubo formula)
    ///
    /// χ(ω) = (1/k_B T) ∫ dt e^{iωt} <A(t)A(0)>
    pub fn susceptibility_from_correlation(
        &self,
        correlation_times: &[f64],
        correlations: &[f64],
        omega: f64,
    ) -> (f64, f64) {
        // Numerical Fourier transform
        let mut real_part = 0.0;
        let mut imag_part = 0.0;

        if correlation_times.len() < 2 {
            return (0.0, 0.0);
        }

        for i in 0..(correlation_times.len() - 1) {
            let t = correlation_times[i];
            let dt = correlation_times[i + 1] - t;
            let c = correlations[i];

            real_part += c * (omega * t).cos() * dt;
            imag_part += c * (omega * t).sin() * dt;
        }

        let prefactor = 1.0 / (K_BOLTZMANN * self.temperature);
        (prefactor * real_part, prefactor * imag_part)
    }
}

/// Jarzynski equality estimator
///
/// Relates nonequilibrium work measurements to free energy differences:
/// ΔF = -k_B T ln⟨exp(-W / k_B T)⟩
///
/// This remarkable equality holds even arbitrarily far from equilibrium.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JarzynskiEstimator;

impl JarzynskiEstimator {
    /// Create new Jarzynski estimator
    pub fn new() -> Self {
        Self
    }

    /// Estimate free energy difference using Jarzynski equality
    ///
    /// ΔF = -k_B T ln⟨exp(-W / k_B T)⟩
    ///
    /// # Arguments
    /// * `work_samples` - Work values from nonequilibrium trajectories (J)
    /// * `temperature` - Temperature (K)
    ///
    /// # Returns
    /// Free energy difference (J)
    pub fn free_energy_difference(&self, work_samples: &[f64], temperature: f64) -> f64 {
        if work_samples.is_empty() {
            return 0.0;
        }

        let beta = 1.0 / (K_BOLTZMANN * temperature);

        // Use log-sum-exp trick for numerical stability
        let min_work = work_samples.iter().cloned().fold(f64::INFINITY, f64::min);

        let sum_exp: f64 = work_samples
            .iter()
            .map(|&w| (-(w - min_work) * beta).exp())
            .sum();

        let log_avg = -min_work * beta + (sum_exp / work_samples.len() as f64).ln();

        -log_avg / beta
    }

    /// Estimate free energy with bootstrap error estimation
    pub fn free_energy_with_error(
        &self,
        work_samples: &[f64],
        temperature: f64,
        n_bootstrap: usize,
    ) -> (f64, f64) {
        let estimate = self.free_energy_difference(work_samples, temperature);

        if work_samples.len() < 2 || n_bootstrap == 0 {
            return (estimate, 0.0);
        }

        // Simple bootstrap with deterministic sampling
        let n = work_samples.len();
        let mut bootstrap_estimates = Vec::with_capacity(n_bootstrap);

        for b in 0..n_bootstrap {
            let mut bootstrap_sample = Vec::with_capacity(n);
            for i in 0..n {
                // Deterministic pseudo-random selection
                let idx = (b * 31 + i * 17) % n;
                bootstrap_sample.push(work_samples[idx]);
            }
            bootstrap_estimates.push(self.free_energy_difference(&bootstrap_sample, temperature));
        }

        let mean: f64 = bootstrap_estimates.iter().sum::<f64>() / n_bootstrap as f64;
        let variance: f64 = bootstrap_estimates
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (n_bootstrap - 1) as f64;

        (estimate, variance.sqrt())
    }

    /// Crooks fluctuation theorem ratio
    ///
    /// P_F(W) / P_R(-W) = exp((W - ΔF) / k_B T)
    ///
    /// Returns the estimated ΔF from comparing forward and reverse work distributions
    pub fn crooks_free_energy(
        &self,
        forward_work: &[f64],
        reverse_work: &[f64],
        temperature: f64,
    ) -> f64 {
        // Bennett acceptance ratio method
        let beta = 1.0 / (K_BOLTZMANN * temperature);

        // Initial guess: average of Jarzynski estimates
        let delta_f_fwd = self.free_energy_difference(forward_work, temperature);
        let neg_reverse: Vec<f64> = reverse_work.iter().map(|&w| -w).collect();
        let delta_f_rev = -self.free_energy_difference(&neg_reverse, temperature);

        let mut delta_f = (delta_f_fwd + delta_f_rev) / 2.0;

        // Iterate to find self-consistent solution
        for _ in 0..100 {
            let sum_fwd: f64 = forward_work
                .iter()
                .map(|&w| 1.0 / (1.0 + ((w - delta_f) * beta).exp()))
                .sum();

            let sum_rev: f64 = reverse_work
                .iter()
                .map(|&w| 1.0 / (1.0 + ((-w - delta_f) * beta).exp()))
                .sum();

            let n_f = forward_work.len() as f64;
            let n_r = reverse_work.len() as f64;

            let new_delta_f = delta_f + (sum_fwd / n_f - sum_rev / n_r) / beta;

            if (new_delta_f - delta_f).abs() < 1e-10 {
                break;
            }
            delta_f = new_delta_f;
        }

        delta_f
    }

    /// Dissipated work
    ///
    /// W_diss = ⟨W⟩ - ΔF
    pub fn dissipated_work(&self, work_samples: &[f64], temperature: f64) -> f64 {
        let mean_work: f64 = work_samples.iter().sum::<f64>() / work_samples.len() as f64;
        let delta_f = self.free_energy_difference(work_samples, temperature);
        mean_work - delta_f
    }
}

/// Irreversible process representation
///
/// Models coupled irreversible processes with their fluxes and forces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrreversibleProcess {
    /// Heat flux (W/m²)
    pub heat_flux: f64,
    /// Particle flux (mol/m²/s)
    pub particle_flux: f64,
    /// Chemical affinity (J/mol)
    pub chemical_affinity: f64,
    /// Temperature gradient (K/m)
    pub temperature_gradient: f64,
    /// Chemical potential gradient (J/mol/m)
    pub chemical_potential_gradient: f64,
}

impl IrreversibleProcess {
    /// Create new irreversible process
    pub fn new(
        heat_flux: f64,
        particle_flux: f64,
        chemical_affinity: f64,
        temperature_gradient: f64,
        chemical_potential_gradient: f64,
    ) -> Self {
        Self {
            heat_flux,
            particle_flux,
            chemical_affinity,
            temperature_gradient,
            chemical_potential_gradient,
        }
    }

    /// Total entropy production rate per unit volume (W/m³/K)
    ///
    /// σ = J_q · ∇(1/T) + J_n · (-∇μ/T) + r · A/T
    pub fn entropy_production(&self, temperature: f64) -> f64 {
        let inv_t = 1.0 / temperature;
        let inv_t2 = inv_t * inv_t;

        // Heat conduction contribution
        let sigma_heat = self.heat_flux * self.temperature_gradient * inv_t2;

        // Diffusion contribution
        let sigma_diff = self.particle_flux * self.chemical_potential_gradient * inv_t;

        // Chemical reaction contribution (assuming unit reaction rate)
        let sigma_chem = self.chemical_affinity * inv_t;

        sigma_heat + sigma_diff + sigma_chem
    }

    /// Thermal conductivity from heat flux and temperature gradient
    ///
    /// κ = -J_q / ∇T (Fourier's law)
    pub fn thermal_conductivity(&self) -> f64 {
        if self.temperature_gradient.abs() < 1e-20 {
            return 0.0;
        }
        -self.heat_flux / self.temperature_gradient
    }

    /// Diffusion coefficient from particle flux and gradient
    ///
    /// D = -J_n / (c · ∇μ/k_B T) (Fick's law with chemical potential)
    pub fn diffusion_coefficient(&self, concentration: f64, temperature: f64) -> f64 {
        if self.chemical_potential_gradient.abs() < 1e-20 || concentration < 1e-20 {
            return 0.0;
        }
        -self.particle_flux * K_BOLTZMANN * temperature
            / (concentration * self.chemical_potential_gradient)
    }
}

/// Prigogine's minimum entropy production principle
///
/// For systems near equilibrium with fixed boundary conditions,
/// the steady state minimizes entropy production.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimumEntropyProduction {
    /// Onsager coefficients
    onsager: OnsagerCoefficients,
    /// Fixed forces (boundary conditions) - None means free
    fixed_forces: Vec<Option<f64>>,
}

impl MinimumEntropyProduction {
    /// Create new minimum entropy production calculator
    pub fn new(onsager: OnsagerCoefficients, fixed_forces: Vec<Option<f64>>) -> Self {
        assert_eq!(
            fixed_forces.len(),
            onsager.dim(),
            "Fixed forces dimension mismatch"
        );
        Self {
            onsager,
            fixed_forces,
        }
    }

    /// Find steady-state forces that minimize entropy production
    ///
    /// Free forces adjust to minimize σ = Σ L_ij X_i X_j
    /// subject to fixed force constraints
    pub fn steady_state_forces(&self) -> Vec<f64> {
        let dim = self.onsager.dim();
        let mut forces = vec![0.0; dim];

        // Set fixed forces
        for (i, &fixed) in self.fixed_forces.iter().enumerate() {
            if let Some(value) = fixed {
                forces[i] = value;
            }
        }

        // For minimum entropy production with some fixed forces,
        // the free forces are determined by ∂σ/∂X_i = 0
        // This gives: Σ_j (L_ij + L_ji) X_j = 0 for free X_i

        // Collect indices of free forces
        let free_indices: Vec<usize> = self
            .fixed_forces
            .iter()
            .enumerate()
            .filter_map(|(i, &f)| if f.is_none() { Some(i) } else { None })
            .collect();

        if free_indices.is_empty() {
            return forces;
        }

        // Solve linear system for free forces
        // This is a simplified version for small systems
        if free_indices.len() == 1 {
            let i = free_indices[0];
            let mut rhs = 0.0;
            let mut diag = 0.0;

            for j in 0..dim {
                let l_sym = self.onsager.get(i, j) + self.onsager.get(j, i);
                if j == i {
                    diag = l_sym;
                } else if self.fixed_forces[j].is_some() {
                    rhs -= l_sym * forces[j];
                }
            }

            if diag.abs() > 1e-20 {
                forces[i] = rhs / diag;
            }
        }

        forces
    }

    /// Compute entropy production at steady state
    pub fn steady_state_entropy_production(&self) -> f64 {
        let forces = self.steady_state_forces();
        self.onsager.entropy_production(&forces)
    }
}

/// HDC encoder for non-equilibrium thermodynamics concepts
#[derive(Debug, Clone)]
pub struct NonequilibriumEncoder {
    _genesis: GenesisSeed,
    /// Basis vector for equilibrium states
    pub equilibrium: ContinuousHV,
    /// Basis vector for nonequilibrium states
    pub nonequilibrium: ContinuousHV,
    /// Basis vector for irreversibility
    pub irreversible: ContinuousHV,
    /// Basis vector for entropy production
    pub entropy_production: ContinuousHV,
    /// Basis vector for fluctuations
    pub fluctuation: ContinuousHV,
    /// Basis vector for dissipation
    pub dissipation: ContinuousHV,
}

impl NonequilibriumEncoder {
    /// Create new encoder with genesis seed
    pub fn new(genesis: GenesisSeed) -> Self {
        Self {
            equilibrium: genesis.hv("noneq_equilibrium", PHYSICS_DIM),
            nonequilibrium: genesis.hv("noneq_nonequilibrium", PHYSICS_DIM),
            irreversible: genesis.hv("noneq_irreversible", PHYSICS_DIM),
            entropy_production: genesis.hv("noneq_entropy_production", PHYSICS_DIM),
            fluctuation: genesis.hv("noneq_fluctuation", PHYSICS_DIM),
            dissipation: genesis.hv("noneq_dissipation", PHYSICS_DIM),
            _genesis: genesis,
        }
    }

    /// Encode the distance from equilibrium
    pub fn encode_departure(&self, entropy_production_rate: f64) -> ContinuousHV {
        // Blend between equilibrium (σ = 0) and nonequilibrium (σ > 0)
        let alpha = (-entropy_production_rate.abs()).exp();
        let scaled_eq = self.equilibrium.scale(alpha as f32);
        let scaled_neq = self.nonequilibrium.scale((1.0 - alpha) as f32);
        scaled_eq.add(&scaled_neq)
    }

    /// Encode fluctuation-dissipation relationship
    pub fn encode_fdt(&self, diffusion: f64, friction: f64, temperature: f64) -> ContinuousHV {
        // Check if FDT holds: D γ = k_B T
        let fdt_ratio = diffusion * friction / (K_BOLTZMANN * temperature);
        let deviation = (fdt_ratio - 1.0).abs();

        let scaled_fluct = self.fluctuation.scale((1.0 - deviation.min(1.0)) as f32);
        let scaled_diss = self.dissipation.scale(deviation.min(1.0) as f32);
        scaled_fluct.add(&scaled_diss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onsager_symmetry() {
        let coeffs = OnsagerCoefficients::symmetric(&[1.0, 0.5, 2.0], 2);
        assert!(coeffs.verify_symmetry(1e-10));
        assert_eq!(coeffs.get(0, 1), coeffs.get(1, 0));
    }

    #[test]
    fn test_onsager_fluxes() {
        let coeffs = OnsagerCoefficients::diagonal(&[2.0, 3.0]);
        let forces = vec![1.0, 1.0];
        let fluxes = coeffs.compute_fluxes(&forces);
        assert!((fluxes[0] - 2.0).abs() < 1e-10);
        assert!((fluxes[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_production_positive() {
        let coeffs = OnsagerCoefficients::symmetric(&[1.0, 0.2, 1.0], 2);
        let forces = vec![1.0, -1.0];
        let sigma = coeffs.entropy_production(&forces);
        assert!(sigma >= 0.0, "Entropy production must be non-negative");
    }

    #[test]
    fn test_einstein_relation() {
        let fdt = FluctuationDissipation::new(300.0, 1e-8);
        let d = fdt.diffusion_coefficient();

        // D = k_B T / γ
        let expected = K_BOLTZMANN * 300.0 / 1e-8;
        assert!((d - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_jarzynski_reversible() {
        // For reversible process, W = ΔF for all trajectories
        let estimator = JarzynskiEstimator::new();
        let work_samples = vec![1e-20, 1e-20, 1e-20, 1e-20]; // Nearly zero work
        let delta_f = estimator.free_energy_difference(&work_samples, 300.0);

        // Should give approximately 0
        assert!(delta_f.abs() < 1e-19);
    }

    #[test]
    fn test_jarzynski_inequality() {
        // Jarzynski inequality: ⟨W⟩ ≥ ΔF
        let estimator = JarzynskiEstimator::new();
        let work_samples = vec![1e-20, 2e-20, 1.5e-20, 3e-20];
        let mean_work: f64 = work_samples.iter().sum::<f64>() / work_samples.len() as f64;
        let delta_f = estimator.free_energy_difference(&work_samples, 300.0);

        assert!(
            mean_work >= delta_f - 1e-25,
            "Mean work must be >= free energy"
        );
    }

    #[test]
    fn test_minimum_entropy_production() {
        let onsager = OnsagerCoefficients::symmetric(&[1.0, 0.0, 1.0], 2);
        let fixed = vec![Some(1.0), None];
        let mep = MinimumEntropyProduction::new(onsager, fixed);

        let forces = mep.steady_state_forces();
        assert!((forces[0] - 1.0).abs() < 1e-10); // Fixed
        assert!((forces[1] - 0.0).abs() < 1e-10); // Free, minimizes σ
    }

    #[test]
    fn test_irreversible_process() {
        let process = IrreversibleProcess::new(
            100.0,  // heat flux W/m²
            0.01,   // particle flux mol/m²/s
            1000.0, // chemical affinity J/mol
            10.0,   // temp gradient K/m
            100.0,  // chem pot gradient J/mol/m
        );

        let sigma = process.entropy_production(300.0);
        assert!(sigma > 0.0, "Entropy production must be positive");
    }

    #[test]
    fn test_positive_semidefinite() {
        // Positive definite matrix
        let coeffs = OnsagerCoefficients::symmetric(&[2.0, 0.5, 2.0], 2);
        assert!(coeffs.is_positive_semidefinite());

        // Negative definite matrix
        let coeffs_neg = OnsagerCoefficients::symmetric(&[-1.0, 0.0, -1.0], 2);
        assert!(!coeffs_neg.is_positive_semidefinite());
    }

    #[test]
    fn test_encoder() {
        let genesis = GenesisSeed::from_phrase("nonequilibrium_test");
        let encoder = NonequilibriumEncoder::new(genesis);

        // At equilibrium (σ = 0), should be close to equilibrium vector
        let eq_state = encoder.encode_departure(0.0);
        let eq_similarity = eq_state.similarity(&encoder.equilibrium);

        // Far from equilibrium
        let neq_state = encoder.encode_departure(100.0);
        let neq_eq_similarity = neq_state.similarity(&encoder.equilibrium);

        // The key test: as entropy production increases, similarity to equilibrium decreases
        assert!(
            eq_similarity > neq_eq_similarity,
            "Higher entropy production should decrease similarity to equilibrium: eq_sim={} neq_eq_sim={}",
            eq_similarity,
            neq_eq_similarity
        );

        // Also verify the encoded vectors have correct dimension
        assert_eq!(eq_state.dim(), PHYSICS_DIM);
        assert_eq!(neq_state.dim(), PHYSICS_DIM);
    }
}
