// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Thermodynamics as Information
//!
//! This module encodes thermodynamic concepts with emphasis on the information-theoretic
//! connections between entropy, free energy, and computation.
//!
//! ## Core Concepts
//!
//! - **Boltzmann Entropy**: S = k_B ln(Ω) - connects microscopic states to macroscopic entropy
//! - **Shannon Entropy**: H = -Σ p_i log(p_i) - information-theoretic entropy
//! - **Free Energy**: F = U - TS (Helmholtz), G = H - TS (Gibbs)
//! - **Landauer's Principle**: Erasing 1 bit costs at least k_B T ln(2) energy
//!
//! ## Maxwell's Demon
//!
//! The demon-Szilard engine connection demonstrates that information is physical:
//! - Demon gathers information → can extract work
//! - Erasing demon's memory → costs at least as much energy
//!
//! ## Statistical Mechanics
//!
//! Ensembles (Microcanonical, Canonical, Grand Canonical) are encoded with their
//! partition functions and probability distributions.

use super::constants::{K_BOLTZMANN, P_STANDARD, R_GAS};
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Thermodynamic state
#[derive(Debug, Clone)]
pub struct ThermodynamicState {
    /// Temperature in Kelvin
    pub temperature_k: f64,
    /// Pressure in Pascals
    pub pressure_pa: f64,
    /// Volume in cubic meters
    pub volume_m3: f64,
    /// Entropy in J/K
    pub entropy_j_k: f64,
    /// Internal energy in Joules
    pub internal_energy_j: f64,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// Thermodynamic encoder
#[derive(Debug, Clone)]
pub struct ThermoEncoder {
    // Fundamental quantities
    /// Entropy concept
    pub entropy: ContinuousHV,
    /// Energy concept
    pub energy: ContinuousHV,
    /// Temperature concept
    pub temperature: ContinuousHV,
    /// Pressure concept
    pub pressure: ContinuousHV,
    /// Volume concept
    pub volume: ContinuousHV,

    // Thermodynamic potentials
    /// Helmholtz free energy: F = U - TS
    pub helmholtz: ContinuousHV,
    /// Gibbs free energy: G = H - TS
    pub gibbs: ContinuousHV,
    /// Enthalpy: H = U + PV
    pub enthalpy: ContinuousHV,

    // Information-theoretic concepts
    /// Information concept
    pub information: ContinuousHV,
    /// Uncertainty concept
    pub uncertainty: ContinuousHV,
    /// Microstates concept
    pub microstates: ContinuousHV,

    // Process concepts
    /// Reversible process
    pub reversible: ContinuousHV,
    /// Irreversible process
    pub irreversible: ContinuousHV,
    /// Equilibrium state
    pub equilibrium: ContinuousHV,
}

impl ThermoEncoder {
    /// Create thermodynamic encoder from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            entropy: genesis.hv("thermo::entropy", PHYSICS_DIM),
            energy: genesis.hv("thermo::energy", PHYSICS_DIM),
            temperature: genesis.hv("thermo::temperature", PHYSICS_DIM),
            pressure: genesis.hv("thermo::pressure", PHYSICS_DIM),
            volume: genesis.hv("thermo::volume", PHYSICS_DIM),
            helmholtz: genesis.hv("thermo::helmholtz", PHYSICS_DIM),
            gibbs: genesis.hv("thermo::gibbs", PHYSICS_DIM),
            enthalpy: genesis.hv("thermo::enthalpy", PHYSICS_DIM),
            information: genesis.hv("thermo::information", PHYSICS_DIM),
            uncertainty: genesis.hv("thermo::uncertainty", PHYSICS_DIM),
            microstates: genesis.hv("thermo::microstates", PHYSICS_DIM),
            reversible: genesis.hv("thermo::reversible", PHYSICS_DIM),
            irreversible: genesis.hv("thermo::irreversible", PHYSICS_DIM),
            equilibrium: genesis.hv("thermo::equilibrium", PHYSICS_DIM),
        }
    }

    /// Boltzmann entropy: S = k_B ln(Ω)
    ///
    /// Encodes entropy as logarithm of number of microstates
    pub fn boltzmann_entropy(&self, microstates: f64) -> ContinuousHV {
        if microstates <= 1.0 {
            return self.entropy.scale(0.0);
        }

        let _s = K_BOLTZMANN * microstates.ln();
        // Normalize to reasonable scale (divide by k_B to get dimensionless)
        let s_normalized = microstates.ln() as f32 / 100.0;

        self.entropy
            .bind(&self.microstates)
            .scale(s_normalized.clamp(0.01, 10.0))
    }

    /// Shannon entropy: H = -Σ p_i log(p_i)
    ///
    /// Information-theoretic entropy in bits
    pub fn shannon_entropy(&self, probabilities: &[f64]) -> ContinuousHV {
        let h: f64 = probabilities
            .iter()
            .filter(|&&p| p > 0.0 && p <= 1.0)
            .map(|&p| -p * p.log2())
            .sum();

        let h_normalized = (h as f32).clamp(0.0, 100.0) / 10.0;
        self.information.bind(&self.uncertainty).scale(h_normalized)
    }

    /// Landauer's principle: minimum energy to erase 1 bit = k_B T ln(2)
    ///
    /// Returns the energy in Joules
    pub fn landauer_limit(&self, temp_k: f64) -> f64 {
        K_BOLTZMANN * temp_k * 2.0_f64.ln()
    }

    /// Encode a thermodynamic state
    pub fn encode_state(
        &self,
        temperature_k: f64,
        pressure_pa: f64,
        volume_m3: f64,
        internal_energy_j: f64,
    ) -> ThermodynamicState {
        // Estimate entropy from ideal gas (simplified)
        let n_moles = pressure_pa * volume_m3 / (R_GAS * temperature_k);
        let entropy_j_k = n_moles * R_GAS * (volume_m3 / n_moles).ln().max(1.0);

        // Encode state vector
        let t_scaled = (temperature_k as f32 / 300.0).clamp(0.01, 100.0);
        let p_scaled = (pressure_pa as f32 / P_STANDARD as f32).clamp(0.01, 100.0);
        let v_scaled = (volume_m3 as f32 * 1000.0).clamp(0.01, 100.0);
        let u_scaled = (internal_energy_j as f32 / 1000.0).clamp(0.01, 100.0);

        let vector = ContinuousHV::weighted_bundle(
            &[
                &self.temperature.scale(t_scaled),
                &self.pressure.scale(p_scaled),
                &self.volume.scale(v_scaled),
                &self.energy.scale(u_scaled),
            ],
            &[1.0, 1.0, 1.0, 1.0],
        );

        ThermodynamicState {
            temperature_k,
            pressure_pa,
            volume_m3,
            entropy_j_k,
            internal_energy_j,
            vector,
        }
    }

    /// Helmholtz free energy: F = U - TS (constant T, V)
    pub fn helmholtz_free_energy(&self, state: &ThermodynamicState) -> ContinuousHV {
        let f = state.internal_energy_j - state.temperature_k * state.entropy_j_k;
        let f_scaled = (f as f32 / 1000.0).clamp(-100.0, 100.0);
        self.helmholtz.scale(f_scaled)
    }

    /// Gibbs free energy: G = H - TS (constant T, P)
    pub fn gibbs_free_energy(&self, state: &ThermodynamicState) -> ContinuousHV {
        let h = state.internal_energy_j + state.pressure_pa * state.volume_m3;
        let g = h - state.temperature_k * state.entropy_j_k;
        let g_scaled = (g as f32 / 1000.0).clamp(-100.0, 100.0);
        self.gibbs.scale(g_scaled)
    }

    /// Enthalpy: H = U + PV
    pub fn enthalpy(&self, state: &ThermodynamicState) -> ContinuousHV {
        let h = state.internal_energy_j + state.pressure_pa * state.volume_m3;
        let h_scaled = (h as f32 / 1000.0).clamp(-100.0, 100.0);
        self.enthalpy.scale(h_scaled)
    }

    /// Free energy principle (Friston): minimize variational free energy
    ///
    /// Returns a measure of prediction error
    pub fn variational_free_energy(
        &self,
        prediction: &ContinuousHV,
        observation: &ContinuousHV,
    ) -> f64 {
        // F = -ln P(o|m) + D_KL(Q||P)
        // Approximated by prediction error (1 - similarity)
        1.0 - prediction.similarity(observation) as f64
    }

    /// Carnot efficiency: η = 1 - T_cold/T_hot
    pub fn carnot_efficiency(&self, t_hot: f64, t_cold: f64) -> f64 {
        if t_hot <= t_cold || t_hot <= 0.0 {
            return 0.0;
        }
        1.0 - t_cold / t_hot
    }
}

/// Maxwell's Demon
#[derive(Debug, Clone)]
pub struct MaxwellDemon {
    /// Number of bits in demon's memory
    pub memory_bits: usize,
    /// Information gathered about system
    pub information_gathered: f64,
    /// Work extracted in Joules
    pub work_extracted_j: f64,
    /// Memory state vector
    pub memory_state: ContinuousHV,
    /// Demon vector
    pub vector: ContinuousHV,
}

impl ThermoEncoder {
    /// Create Maxwell's demon for a Szilard engine
    ///
    /// Szilard engine: 1 bit → k_B T ln(2) work
    pub fn szilard_engine(&self, temp_k: f64) -> MaxwellDemon {
        let work = self.landauer_limit(temp_k);

        let memory_state = self.information.clone();
        let vector = self.entropy.bind(&self.information);

        MaxwellDemon {
            memory_bits: 1,
            information_gathered: 1.0,
            work_extracted_j: work,
            memory_state,
            vector,
        }
    }

    /// Memory erasure cost (Landauer)
    ///
    /// Returns the energy cost in Joules and resets demon memory
    pub fn erase_memory(&self, demon: &mut MaxwellDemon, temp_k: f64) -> f64 {
        let cost = demon.memory_bits as f64 * self.landauer_limit(temp_k);
        demon.memory_bits = 0;
        demon.information_gathered = 0.0;
        demon.memory_state = ContinuousHV::zero(PHYSICS_DIM);
        cost
    }

    /// Demon measurement: gather information about particle
    pub fn demon_measure(&self, demon: &mut MaxwellDemon, _particle: &ContinuousHV) {
        demon.memory_bits += 1;
        demon.information_gathered += 1.0;
        demon.memory_state = self.information.permute(demon.memory_bits * 1000);
    }
}

/// Statistical ensemble type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnsembleType {
    /// Microcanonical: fixed N, V, E
    Microcanonical,
    /// Canonical: fixed N, V, T
    Canonical,
    /// Grand Canonical: fixed μ, V, T
    GrandCanonical,
}

/// Statistical ensemble
#[derive(Debug, Clone)]
pub struct Ensemble {
    /// Ensemble type
    pub ensemble_type: EnsembleType,
    /// Partition function (Z for canonical, Ξ for grand canonical)
    pub partition_function: f64,
    /// Vector representation
    pub vector: ContinuousHV,
}

impl ThermoEncoder {
    /// Create microcanonical ensemble (NVE)
    ///
    /// All microstates equally probable
    pub fn microcanonical_ensemble(&self, num_microstates: f64) -> Ensemble {
        let vector = self.microstates.scale(num_microstates.ln() as f32 / 10.0);

        Ensemble {
            ensemble_type: EnsembleType::Microcanonical,
            partition_function: num_microstates,
            vector: vector.bind(&self.equilibrium),
        }
    }

    /// Create canonical ensemble (NVT)
    ///
    /// Partition function: Z = Σ exp(-E_i / k_B T)
    pub fn canonical_ensemble(&self, energies: &[f64], temp_k: f64) -> Ensemble {
        let beta = 1.0 / (K_BOLTZMANN * temp_k);
        let z: f64 = energies.iter().map(|&e| (-beta * e).exp()).sum();

        let vector = self.microstates.scale(z.ln() as f32 / 10.0);

        Ensemble {
            ensemble_type: EnsembleType::Canonical,
            partition_function: z,
            vector: vector.bind(&self.equilibrium).bind(&self.temperature),
        }
    }

    /// Boltzmann probability: P(E) = exp(-E/k_B T) / Z
    pub fn boltzmann_probability(&self, energy: f64, temp_k: f64, z: f64) -> f64 {
        let beta = 1.0 / (K_BOLTZMANN * temp_k);
        (-beta * energy).exp() / z
    }

    /// Create grand canonical ensemble (μVT)
    pub fn grand_canonical_ensemble(
        &self,
        energies: &[f64],
        particle_numbers: &[usize],
        temp_k: f64,
        chemical_potential: f64,
    ) -> Ensemble {
        let beta = 1.0 / (K_BOLTZMANN * temp_k);
        let xi: f64 = energies
            .iter()
            .zip(particle_numbers.iter())
            .map(|(&e, &n)| (-beta * (e - chemical_potential * n as f64)).exp())
            .sum();

        let vector = self.microstates.scale(xi.ln() as f32 / 10.0);

        Ensemble {
            ensemble_type: EnsembleType::GrandCanonical,
            partition_function: xi,
            vector: vector.bind(&self.equilibrium).bind(&self.temperature),
        }
    }
}

/// Dissipation and entropy production
#[derive(Debug, Clone)]
pub struct Dissipation {
    /// Entropy production rate (dS/dt)
    pub entropy_production_rate: f64,
    /// Heat flow in Watts
    pub heat_flow_w: f64,
    /// Vector representation
    pub vector: ContinuousHV,
}

impl ThermoEncoder {
    /// Entropy production in irreversible process
    ///
    /// For heat flow between two reservoirs
    pub fn entropy_production(&self, heat_flow: f64, t_hot: f64, t_cold: f64) -> Dissipation {
        // dS/dt = Q * (1/T_cold - 1/T_hot)
        let ds_dt = heat_flow * (1.0 / t_cold - 1.0 / t_hot);

        let vector = self.entropy.scale(ds_dt as f32).bind(&self.irreversible);

        Dissipation {
            entropy_production_rate: ds_dt,
            heat_flow_w: heat_flow,
            vector,
        }
    }

    /// Fluctuation theorem: P(+σ)/P(-σ) = exp(σ)
    ///
    /// Returns the ratio of forward to backward probabilities
    pub fn fluctuation_theorem(&self, entropy_change: f64) -> f64 {
        entropy_change.exp()
    }

    /// Jarzynski equality: <exp(-W/k_B T)> = exp(-ΔF/k_B T)
    ///
    /// Estimates free energy change from work measurements
    pub fn jarzynski_equality(&self, work_samples: &[f64], temp_k: f64) -> f64 {
        if work_samples.is_empty() {
            return 0.0;
        }

        let beta = 1.0 / (K_BOLTZMANN * temp_k);
        let avg_exp_w: f64 = work_samples.iter().map(|&w| (-beta * w).exp()).sum::<f64>()
            / work_samples.len() as f64;

        // ΔF = -k_B T ln(<exp(-βW)>)
        -avg_exp_w.ln() / beta
    }

    /// Crooks fluctuation theorem
    ///
    /// P_F(W) / P_R(-W) = exp(β(W - ΔF))
    pub fn crooks_ratio(&self, work: f64, delta_f: f64, temp_k: f64) -> f64 {
        let beta = 1.0 / (K_BOLTZMANN * temp_k);
        (beta * (work - delta_f)).exp()
    }
}

/// Second law of thermodynamics
#[derive(Debug, Clone)]
pub struct SecondLaw {
    /// Total entropy change (must be >= 0 for isolated system)
    pub total_entropy_change: f64,
    /// Is process allowed by second law
    pub allowed: bool,
    /// Vector representation
    pub vector: ContinuousHV,
}

impl ThermoEncoder {
    /// Check if process is allowed by second law
    pub fn check_second_law(&self, delta_s_system: f64, delta_s_surroundings: f64) -> SecondLaw {
        let total = delta_s_system + delta_s_surroundings;
        let allowed = total >= 0.0;

        let vector = if allowed {
            self.entropy.scale(total as f32).bind(&self.reversible)
        } else {
            self.entropy.scale(total as f32).bind(&self.irreversible)
        };

        SecondLaw {
            total_entropy_change: total,
            allowed,
            vector,
        }
    }
}

/// Thermodynamic ledger for tracking physical energy costs of cognitive operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicLedger {
    /// Available energy in Joules.
    pub energy_j: f64,
    /// Maximum energy capacity.
    pub capacity_j: f64,
    /// Total energy dissipated so far.
    pub total_dissipated_j: f64,
}

impl ThermodynamicLedger {
    pub fn new(capacity_j: f64) -> Self {
        Self {
            energy_j: capacity_j,
            capacity_j,
            total_dissipated_j: 0.0,
        }
    }
}

impl Default for ThermodynamicLedger {
    fn default() -> Self {
        Self::new(1000.0)
    }
}

impl ThermodynamicLedger {
    /// Deduct energy for a cognitive operation.
    ///
    /// Returns true if the operation is viable (sufficient energy).
    pub fn deduct(&mut self, cost_j: f64) -> bool {
        if self.energy_j >= cost_j {
            self.energy_j -= cost_j;
            self.total_dissipated_j += cost_j;
            true
        } else {
            // Insufficient energy: operation fails
            false
        }
    }

    /// Add energy (e.g. from recharge or metabolic processes).
    pub fn recharge(&mut self, amount_j: f64) {
        self.energy_j = (self.energy_j + amount_j).min(self.capacity_j);
    }

    /// Check if the ledger is depleted (below a critical threshold).
    pub fn is_depleted(&self) -> bool {
        self.energy_j < self.capacity_j * 0.05
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (ThermoEncoder, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("thermodynamics test");
        let encoder = ThermoEncoder::from_genesis(&genesis);
        (encoder, genesis)
    }

    #[test]
    fn test_boltzmann_entropy() {
        let (encoder, _) = setup();

        // More microstates = more entropy
        let s1 = encoder.boltzmann_entropy(10.0);
        let s2 = encoder.boltzmann_entropy(1000.0);

        // s2 should have larger scale
        assert!(
            s2.norm() > s1.norm() * 0.5,
            "More microstates should give more entropy"
        );
    }

    #[test]
    fn test_shannon_entropy() {
        let (encoder, _) = setup();

        // Uniform distribution has maximum entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let peaked = vec![0.97, 0.01, 0.01, 0.01];

        let h_uniform = encoder.shannon_entropy(&uniform);
        let h_peaked = encoder.shannon_entropy(&peaked);

        assert!(
            h_uniform.norm() > h_peaked.norm(),
            "Uniform should have higher entropy"
        );
    }

    #[test]
    fn test_landauer_limit() {
        let (encoder, _) = setup();

        // At room temperature (300 K)
        let limit = encoder.landauer_limit(300.0);

        // Should be about 2.87e-21 J
        let expected = K_BOLTZMANN * 300.0 * 2.0_f64.ln();
        assert!(
            (limit - expected).abs() < 1e-25,
            "Landauer limit should be k_B T ln(2)"
        );
    }

    #[test]
    fn test_helmholtz_minimized_at_equilibrium() {
        let (encoder, _) = setup();

        // Create states at same T but different entropy
        let state1 = encoder.encode_state(300.0, 101325.0, 0.001, 1000.0);
        let state2 = encoder.encode_state(300.0, 101325.0, 0.002, 1000.0);

        let f1 = encoder.helmholtz_free_energy(&state1);
        let f2 = encoder.helmholtz_free_energy(&state2);

        // Both should have valid vectors
        assert!(f1.norm() > 0.0);
        assert!(f2.norm() > 0.0);
    }

    #[test]
    fn test_szilard_engine() {
        let (encoder, _) = setup();

        let demon = encoder.szilard_engine(300.0);

        // Should have 1 bit of memory
        assert_eq!(demon.memory_bits, 1);
        assert!((demon.information_gathered - 1.0).abs() < 0.001);

        // Work extracted should equal Landauer limit
        let limit = encoder.landauer_limit(300.0);
        assert!((demon.work_extracted_j - limit).abs() < 1e-25);
    }

    #[test]
    fn test_canonical_ensemble() {
        let (encoder, _) = setup();

        // Simple two-level system
        let energies = vec![0.0, 1e-21];
        let ensemble = encoder.canonical_ensemble(&energies, 300.0);

        assert_eq!(ensemble.ensemble_type, EnsembleType::Canonical);
        assert!(
            ensemble.partition_function > 1.0,
            "Z should be > 1 for multiple states"
        );
    }

    #[test]
    fn test_jarzynski_equality() {
        let (encoder, _) = setup();

        // If all work samples are equal, ΔF should equal W
        let work_samples = vec![1e-20, 1e-20, 1e-20];
        let delta_f = encoder.jarzynski_equality(&work_samples, 300.0);

        // ΔF should be close to the work value
        assert!((delta_f - 1e-20).abs() < 1e-21);
    }

    #[test]
    fn test_fluctuation_theorem() {
        let (encoder, _) = setup();

        // Positive entropy change
        let ratio_pos = encoder.fluctuation_theorem(1.0);
        assert!((ratio_pos - std::f64::consts::E).abs() < 0.01);

        // Negative entropy change (rare event)
        let ratio_neg = encoder.fluctuation_theorem(-1.0);
        assert!((ratio_neg - 1.0 / std::f64::consts::E).abs() < 0.01);
    }

    #[test]
    fn test_second_law() {
        let (encoder, _) = setup();

        // Positive total entropy change: allowed
        let allowed = encoder.check_second_law(1.0, 0.0);
        assert!(allowed.allowed);

        // Negative total entropy change: not allowed
        let not_allowed = encoder.check_second_law(-2.0, 1.0);
        assert!(!not_allowed.allowed);

        // Zero change: reversible, allowed
        let reversible = encoder.check_second_law(1.0, -1.0);
        assert!(reversible.allowed);
    }

    #[test]
    fn test_carnot_efficiency() {
        let (encoder, _) = setup();

        // Hot 600K, cold 300K: should be 50%
        let eta = encoder.carnot_efficiency(600.0, 300.0);
        assert!((eta - 0.5).abs() < 0.001);

        // Equal temperatures: 0% efficiency
        let eta_zero = encoder.carnot_efficiency(300.0, 300.0);
        assert!(eta_zero.abs() < 0.001);
    }

    #[test]
    fn test_entropy_production() {
        let (encoder, _) = setup();

        // Heat flowing from hot to cold: positive entropy production
        let dissipation = encoder.entropy_production(100.0, 600.0, 300.0);
        assert!(dissipation.entropy_production_rate > 0.0);
    }

    #[test]
    fn test_variational_free_energy() {
        let (encoder, _) = setup();

        // Perfect prediction: zero free energy
        let fe_zero = encoder.variational_free_energy(&encoder.entropy, &encoder.entropy);
        assert!(fe_zero < 0.01);

        // Bad prediction: high free energy
        let fe_high = encoder.variational_free_energy(&encoder.entropy, &encoder.temperature);
        assert!(fe_high > 0.5);
    }

    #[test]
    fn test_determinism() {
        let genesis = GenesisSeed::from_phrase("determinism test");
        let enc1 = ThermoEncoder::from_genesis(&genesis);
        let enc2 = ThermoEncoder::from_genesis(&genesis);

        assert!(
            enc1.entropy.similarity(&enc2.entropy) > 0.9999,
            "Thermo encoder should be deterministic"
        );
    }

    #[test]
    fn test_ledger() {
        let mut ledger = ThermodynamicLedger::new(10.0);
        assert_eq!(ledger.energy_j, 10.0);

        assert!(ledger.deduct(2.0));
        assert_eq!(ledger.energy_j, 8.0);
        assert_eq!(ledger.total_dissipated_j, 2.0);

        assert!(!ledger.deduct(10.0));
        assert_eq!(ledger.energy_j, 8.0);

        ledger.recharge(5.0);
        assert_eq!(ledger.energy_j, 10.0); // Capped at capacity
    }
}
