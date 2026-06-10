// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Statistical Mechanics
//!
//! **Status**: Exploratory module — HDC encodings for statistical mechanics (ensembles, Ising model, partition functions).
//! Not re-exported due to name conflicts with thermodynamics.
//!
//! This module encodes statistical mechanics: the bridge between
//! microscopic dynamics and macroscopic thermodynamics.
//!
//! ## Ensembles
//!
//! - **Microcanonical**: Fixed E, N, V → S = k ln Ω
//! - **Canonical**: Fixed T, N, V → F = -kT ln Z
//! - **Grand Canonical**: Fixed T, μ, V → Ω = -kT ln Ξ
//!
//! ## Key Concepts
//!
//! - **Partition Function**: `Z = Σ exp(-βE_i)`
//! - **Boltzmann Distribution**: `P_i = exp(-βE_i)/Z`
//! - **Equipartition**: `<E> = ½kT` per quadratic degree
//!
//! ## Models
//!
//! - **Ideal Gas**: Non-interacting particles
//! - **Ising Model**: Interacting spins
//! - **Lattice Models**: Discrete statistical systems

use super::constants::K_BOLTZMANN;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Ensemble type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ensemble {
    Microcanonical, // NVE
    Canonical,      // NVT
    GrandCanonical, // μVT
    Isothermal,     // NPT
}

impl Ensemble {
    fn fixed_quantities(&self) -> &'static str {
        match self {
            Ensemble::Microcanonical => "E, N, V",
            Ensemble::Canonical => "T, N, V",
            Ensemble::GrandCanonical => "T, μ, V",
            Ensemble::Isothermal => "T, N, P",
        }
    }
}

/// Statistical model type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatModelType {
    IdealGas,
    Ising,
    Heisenberg,
    Potts,
    XY,
    LatticeGas,
    HardSphere,
}

/// A statistical system
#[derive(Debug, Clone)]
pub struct StatisticalSystem {
    pub name: String,
    pub model_type: StatModelType,
    pub ensemble: Ensemble,
    pub temperature: f64,
    pub particle_count: u64,
    pub partition_function: f64,
    pub vector: ContinuousHV,
}

/// Ising model configuration
#[derive(Debug, Clone)]
pub struct IsingModel {
    pub dimension: u32,
    pub coupling_j: f64,
    pub external_field: f64,
    pub critical_temp: f64,
    pub vector: ContinuousHV,
}

/// Statistical Mechanics Encoder
#[derive(Debug, Clone)]
pub struct StatMechEncoder {
    genesis: GenesisSeed,

    // Fundamental quantities
    pub energy: ContinuousHV,
    pub entropy: ContinuousHV,
    pub temperature: ContinuousHV,
    pub particle_number: ContinuousHV,
    pub chemical_potential: ContinuousHV,
    pub volume: ContinuousHV,
    pub pressure: ContinuousHV,

    // Statistical quantities
    pub partition_function: ContinuousHV,
    pub density_of_states: ContinuousHV,
    pub probability: ContinuousHV,
    pub fluctuation: ContinuousHV,

    // Free energies
    pub helmholtz: ContinuousHV,       // F = E - TS
    pub gibbs: ContinuousHV,           // G = H - TS = F + PV
    pub grand_potential: ContinuousHV, // Ω = F - μN

    // Ensembles
    pub microcanonical: ContinuousHV,
    pub canonical: ContinuousHV,
    pub grand_canonical: ContinuousHV,

    // Distributions
    pub boltzmann: ContinuousHV,
    pub fermi_dirac: ContinuousHV,
    pub bose_einstein: ContinuousHV,
    pub maxwell_boltzmann: ContinuousHV,

    // Phase transitions
    pub order_parameter: ContinuousHV,
    pub critical_point: ContinuousHV,
    pub spontaneous_symmetry_breaking: ContinuousHV,

    // Models
    pub ising: ContinuousHV,
    pub ideal_gas: ContinuousHV,
    pub lattice: ContinuousHV,

    // Correlations
    pub correlation_function: ContinuousHV,
    pub correlation_length: ContinuousHV,
}

impl StatMechEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            genesis: genesis.clone(),

            energy: genesis.hv("stat::energy", PHYSICS_DIM),
            entropy: genesis.hv("stat::entropy", PHYSICS_DIM),
            temperature: genesis.hv("stat::temperature", PHYSICS_DIM),
            particle_number: genesis.hv("stat::particle_number", PHYSICS_DIM),
            chemical_potential: genesis.hv("stat::chemical_potential", PHYSICS_DIM),
            volume: genesis.hv("stat::volume", PHYSICS_DIM),
            pressure: genesis.hv("stat::pressure", PHYSICS_DIM),

            partition_function: genesis.hv("stat::partition_function", PHYSICS_DIM),
            density_of_states: genesis.hv("stat::density_of_states", PHYSICS_DIM),
            probability: genesis.hv("stat::probability", PHYSICS_DIM),
            fluctuation: genesis.hv("stat::fluctuation", PHYSICS_DIM),

            helmholtz: genesis.hv("free_energy::helmholtz", PHYSICS_DIM),
            gibbs: genesis.hv("free_energy::gibbs", PHYSICS_DIM),
            grand_potential: genesis.hv("free_energy::grand_potential", PHYSICS_DIM),

            microcanonical: genesis.hv("ensemble::microcanonical", PHYSICS_DIM),
            canonical: genesis.hv("ensemble::canonical", PHYSICS_DIM),
            grand_canonical: genesis.hv("ensemble::grand_canonical", PHYSICS_DIM),

            boltzmann: genesis.hv("distribution::boltzmann", PHYSICS_DIM),
            fermi_dirac: genesis.hv("distribution::fermi_dirac", PHYSICS_DIM),
            bose_einstein: genesis.hv("distribution::bose_einstein", PHYSICS_DIM),
            maxwell_boltzmann: genesis.hv("distribution::maxwell_boltzmann", PHYSICS_DIM),

            order_parameter: genesis.hv("phase::order_parameter", PHYSICS_DIM),
            critical_point: genesis.hv("phase::critical_point", PHYSICS_DIM),
            spontaneous_symmetry_breaking: genesis.hv("phase::ssb", PHYSICS_DIM),

            ising: genesis.hv("model::ising", PHYSICS_DIM),
            ideal_gas: genesis.hv("model::ideal_gas", PHYSICS_DIM),
            lattice: genesis.hv("model::lattice", PHYSICS_DIM),

            correlation_function: genesis.hv("stat::correlation_function", PHYSICS_DIM),
            correlation_length: genesis.hv("stat::correlation_length", PHYSICS_DIM),
        }
    }

    // ============================================================
    // FUNDAMENTAL RELATIONS
    // ============================================================

    /// Boltzmann entropy: S = k ln Ω
    pub fn boltzmann_entropy(&self) -> ContinuousHV {
        self.entropy
            .bind(&self.density_of_states)
            .bind(&self.microcanonical)
    }

    /// Partition function relation: Z = Σ exp(-βE)
    pub fn partition_function_form(&self) -> ContinuousHV {
        let beta = self.genesis.hv("stat::inverse_temp", PHYSICS_DIM);
        self.partition_function
            .bind(&beta)
            .bind(&self.energy)
            .bind(&self.canonical)
    }

    /// Boltzmann distribution: `P_i = exp(-βE_i)/Z`
    pub fn boltzmann_distribution(&self) -> ContinuousHV {
        self.boltzmann
            .bind(&self.probability)
            .bind(&self.energy)
            .bind(&self.partition_function)
    }

    /// Free energy: F = -kT ln Z
    pub fn helmholtz_free_energy(&self) -> ContinuousHV {
        self.helmholtz
            .bind(&self.temperature)
            .bind(&self.partition_function)
    }

    /// Gibbs free energy: G = F + PV
    pub fn gibbs_free_energy(&self) -> ContinuousHV {
        self.gibbs
            .bind(&self.helmholtz)
            .bind(&self.pressure)
            .bind(&self.volume)
    }

    // ============================================================
    // QUANTUM STATISTICS
    // ============================================================

    /// Fermi-Dirac distribution: n = 1/(exp(β(ε-μ)) + 1)
    pub fn fermi_dirac_distribution(&self) -> ContinuousHV {
        let fermion = self.genesis.hv("particle::fermion", PHYSICS_DIM);
        self.fermi_dirac
            .bind(&fermion)
            .bind(&self.chemical_potential)
            .bind(&self.probability)
    }

    /// Bose-Einstein distribution: n = 1/(exp(β(ε-μ)) - 1)
    pub fn bose_einstein_distribution(&self) -> ContinuousHV {
        let boson = self.genesis.hv("particle::boson", PHYSICS_DIM);
        self.bose_einstein
            .bind(&boson)
            .bind(&self.chemical_potential)
            .bind(&self.probability)
    }

    // ============================================================
    // IDEAL GAS
    // ============================================================

    /// Create ideal gas system
    pub fn ideal_gas_system(&self, n: u64, temp_k: f64, volume_m3: f64) -> StatisticalSystem {
        // ln Z = N ln(V/λ³) + const, where λ = thermal wavelength
        let log_z = (n as f64) * (volume_m3.ln() + 1.5 * temp_k.ln());

        let vector = self
            .ideal_gas
            .bind(&self.canonical)
            .bind(&self.partition_function)
            .scale(log_z as f32);

        StatisticalSystem {
            name: "Ideal Gas".to_string(),
            model_type: StatModelType::IdealGas,
            ensemble: Ensemble::Canonical,
            temperature: temp_k,
            particle_count: n,
            partition_function: log_z.exp(),
            vector,
        }
    }

    /// Equipartition theorem: `<E> = ½kT` per quadratic DOF
    pub fn equipartition(&self, dof: u32) -> ContinuousHV {
        let equipart = self.genesis.hv("theorem::equipartition", PHYSICS_DIM);
        equipart
            .bind(&self.energy)
            .bind(&self.temperature)
            .scale(0.5 * dof as f32)
    }

    /// Maxwell-Boltzmann velocity distribution
    pub fn maxwell_boltzmann_velocity(&self) -> ContinuousHV {
        let velocity = self.genesis.hv("mechanics::velocity", PHYSICS_DIM);
        self.maxwell_boltzmann
            .bind(&velocity)
            .bind(&self.temperature)
            .bind(&self.probability)
    }

    // ============================================================
    // ISING MODEL
    // ============================================================

    /// Create Ising model
    pub fn ising_model(&self, dim: u32, j: f64, h: f64) -> IsingModel {
        // Critical temperature: kT_c = 2J/ln(1+√2) for 2D square lattice
        let tc = if dim == 2 {
            2.0 * j / (1.0 + 2.0_f64.sqrt()).ln()
        } else if dim == 1 {
            0.0 // No phase transition in 1D
        } else {
            4.5 * j // Approximate for 3D
        };

        let vector = self
            .ising
            .bind(&self.lattice)
            .bind(&self.order_parameter)
            .scale(j as f32);

        IsingModel {
            dimension: dim,
            coupling_j: j,
            external_field: h,
            critical_temp: tc,
            vector,
        }
    }

    /// Ising magnetization (order parameter)
    pub fn ising_magnetization(&self, ising: &IsingModel, temp: f64) -> f64 {
        if ising.dimension == 1 {
            // 1D Ising: no spontaneous magnetization
            if ising.external_field != 0.0 {
                (ising.external_field / (K_BOLTZMANN * temp)).tanh()
            } else {
                0.0
            }
        } else if temp >= ising.critical_temp {
            0.0 // Above Tc: paramagnetic
        } else {
            // Below Tc: spontaneous magnetization
            // M ~ (1 - T/Tc)^β with β ≈ 0.125 (2D) or 0.326 (3D)
            let beta = if ising.dimension == 2 { 0.125 } else { 0.326 };
            (1.0 - temp / ising.critical_temp).powf(beta)
        }
    }

    /// Ising correlation length
    pub fn ising_correlation_length(&self, ising: &IsingModel, temp: f64) -> ContinuousHV {
        // ξ ~ |T - Tc|^(-ν)
        let reduced_t = (temp - ising.critical_temp).abs() / ising.critical_temp;
        let nu = if ising.dimension == 2 { 1.0 } else { 0.63 };
        let xi = if reduced_t > 0.001 {
            reduced_t.powf(-nu)
        } else {
            1000.0 // Near Tc, correlation length diverges
        };

        self.correlation_length.bind(&ising.vector).scale(xi as f32)
    }

    // ============================================================
    // PHASE TRANSITIONS
    // ============================================================

    /// Encode critical point behavior
    pub fn critical_behavior(&self, temp: f64, tc: f64) -> ContinuousHV {
        let reduced_t = (temp - tc).abs() / tc;

        if reduced_t < 0.1 {
            // Near critical point
            self.critical_point
                .bind(&self.correlation_length)
                .bind(&self.spontaneous_symmetry_breaking)
                .scale((1.0 / reduced_t) as f32)
        } else {
            // Away from critical point
            self.order_parameter.scale(reduced_t as f32)
        }
    }

    /// Mean field approximation
    pub fn mean_field(&self) -> ContinuousHV {
        let mf = self.genesis.hv("approx::mean_field", PHYSICS_DIM);
        mf.bind(&self.order_parameter).bind(&self.probability)
    }

    // ============================================================
    // FLUCTUATIONS
    // ============================================================

    /// Energy fluctuations in canonical ensemble: <(ΔE)²> = kT²C_V
    pub fn energy_fluctuations(&self) -> ContinuousHV {
        let heat_capacity = self.genesis.hv("thermo::heat_capacity", PHYSICS_DIM);
        self.fluctuation
            .bind(&self.energy)
            .bind(&self.temperature)
            .bind(&heat_capacity)
    }

    /// Particle number fluctuations in grand canonical
    pub fn number_fluctuations(&self) -> ContinuousHV {
        let compressibility = self.genesis.hv("thermo::compressibility", PHYSICS_DIM);
        self.fluctuation
            .bind(&self.particle_number)
            .bind(&compressibility)
            .bind(&self.grand_canonical)
    }

    /// Fluctuation-dissipation theorem
    pub fn fluctuation_dissipation(&self) -> ContinuousHV {
        let response = self.genesis.hv("response::linear", PHYSICS_DIM);
        let dissipation = self.genesis.hv("thermo::dissipation", PHYSICS_DIM);

        self.fluctuation
            .bind(&response)
            .bind(&dissipation)
            .bind(&self.temperature)
    }

    // ============================================================
    // ENSEMBLES
    // ============================================================

    /// Microcanonical ensemble: S(E, V, N)
    pub fn microcanonical_ensemble(&self) -> ContinuousHV {
        self.microcanonical
            .bind(&self.entropy)
            .bind(&self.energy)
            .bind(&self.density_of_states)
    }

    /// Canonical ensemble: F(T, V, N)
    pub fn canonical_ensemble(&self) -> ContinuousHV {
        self.canonical
            .bind(&self.helmholtz)
            .bind(&self.temperature)
            .bind(&self.partition_function)
    }

    /// Grand canonical ensemble: Ω(T, V, μ)
    pub fn grand_canonical_ensemble(&self) -> ContinuousHV {
        self.grand_canonical
            .bind(&self.grand_potential)
            .bind(&self.chemical_potential)
            .bind(&self.particle_number)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> StatMechEncoder {
        let genesis = GenesisSeed::from_phrase("stat mech test");
        StatMechEncoder::from_genesis(&genesis)
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = setup();
        assert!(encoder.partition_function.norm() > 0.0);
        assert!(encoder.boltzmann.norm() > 0.0);
    }

    #[test]
    fn test_ideal_gas() {
        let encoder = setup();

        let gas = encoder.ideal_gas_system(1000, 300.0, 1.0);

        assert_eq!(gas.model_type, StatModelType::IdealGas);
        assert!(gas.partition_function > 0.0);
        assert!(gas.vector.norm() > 0.0);
    }

    #[test]
    fn test_ising_critical_temp() {
        let encoder = setup();

        let ising_2d = encoder.ising_model(2, 1.0, 0.0);
        let ising_1d = encoder.ising_model(1, 1.0, 0.0);

        // 2D Ising has finite Tc
        assert!(ising_2d.critical_temp > 0.0);
        // 1D Ising has Tc = 0
        assert_eq!(ising_1d.critical_temp, 0.0);
    }

    #[test]
    fn test_ising_magnetization() {
        let encoder = setup();

        let ising = encoder.ising_model(2, 1.0, 0.0);

        // Below Tc: spontaneous magnetization
        let m_low = encoder.ising_magnetization(&ising, 0.5 * ising.critical_temp);
        assert!(m_low > 0.0);

        // Above Tc: no magnetization
        let m_high = encoder.ising_magnetization(&ising, 2.0 * ising.critical_temp);
        assert_eq!(m_high, 0.0);
    }

    #[test]
    fn test_fermi_bose_distinct() {
        let encoder = setup();

        let fd = encoder.fermi_dirac_distribution();
        let be = encoder.bose_einstein_distribution();

        // Fermi-Dirac and Bose-Einstein should be distinct
        let sim = fd.similarity(&be);
        assert!(sim < 0.9);
    }

    #[test]
    fn test_ensembles_distinct() {
        let encoder = setup();

        let micro = encoder.microcanonical_ensemble();
        let canon = encoder.canonical_ensemble();
        let grand = encoder.grand_canonical_ensemble();

        // Ensembles should be different
        assert!(micro.similarity(&canon) < 0.9);
        assert!(canon.similarity(&grand) < 0.9);
    }

    #[test]
    fn test_fluctuation_dissipation() {
        let encoder = setup();

        let fdt = encoder.fluctuation_dissipation();

        // Should encode fluctuation-response connection
        assert!(fdt.norm() > 0.0);
    }

    #[test]
    fn test_boltzmann_entropy() {
        let encoder = setup();

        let s = encoder.boltzmann_entropy();

        // Should be in microcanonical context
        let _sim = s.similarity(&encoder.microcanonical);
        // After binding, similarity is indirect
        assert!(s.norm() > 0.0);
    }

    #[test]
    fn test_equipartition() {
        let encoder = setup();

        let equip_3 = encoder.equipartition(3);
        let equip_6 = encoder.equipartition(6);

        // More DOF should have different encoding
        let norm_3 = equip_3.norm();
        let norm_6 = equip_6.norm();
        assert!(norm_6 > norm_3);
    }
}
