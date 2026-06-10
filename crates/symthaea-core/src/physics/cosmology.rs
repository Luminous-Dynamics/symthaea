// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cosmology
//!
//! This module encodes cosmological concepts from the Big Bang to large-scale structure.
//!
//! ## ΛCDM Model
//!
//! - **Λ**: Cosmological constant (dark energy)
//! - **CDM**: Cold dark matter
//! - Baryonic matter: ~5%
//! - Dark matter: ~27%
//! - Dark energy: ~68%
//!
//! ## Key Epochs
//!
//! - Inflation: Exponential expansion (t ~ 10⁻³⁶ s)
//! - Recombination: Atoms form (t ~ 380,000 yr)
//! - Reionization: First stars (t ~ 400 Myr)
//! - Present: 13.8 Gyr
//!
//! ## Cosmic Microwave Background
//!
//! - Temperature: 2.725 K
//! - Fluctuations: δT/T ~ 10⁻⁵

use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Hubble constant (km/s/Mpc)
pub const H0_KM_S_MPC: f64 = 67.4;

/// Age of the universe (Gyr)
pub const AGE_GYR: f64 = 13.8;

/// CMB temperature (K)
pub const T_CMB: f64 = 2.725;

/// Cosmological era
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CosmologicalEra {
    Planck,           // t < 10⁻⁴³ s
    GrandUnification, // 10⁻⁴³ to 10⁻³⁶ s
    Inflation,        // 10⁻³⁶ to 10⁻³² s
    Electroweak,      // 10⁻³² to 10⁻¹² s
    QuarkHadron,      // 10⁻¹² to 10⁻⁶ s
    Nucleosynthesis,  // 1 s to 3 min
    Recombination,    // 380,000 yr
    DarkAges,         // 380,000 to 400 Myr
    Reionization,     // 400 Myr to 1 Gyr
    GalaxyFormation,  // 1 to 10 Gyr
    Present,          // 13.8 Gyr
}

impl CosmologicalEra {
    fn domain(&self) -> &'static str {
        match self {
            CosmologicalEra::Planck => "cosmo::planck",
            CosmologicalEra::GrandUnification => "cosmo::gut",
            CosmologicalEra::Inflation => "cosmo::inflation",
            CosmologicalEra::Electroweak => "cosmo::electroweak",
            CosmologicalEra::QuarkHadron => "cosmo::qcd",
            CosmologicalEra::Nucleosynthesis => "cosmo::bbn",
            CosmologicalEra::Recombination => "cosmo::recombination",
            CosmologicalEra::DarkAges => "cosmo::dark_ages",
            CosmologicalEra::Reionization => "cosmo::reionization",
            CosmologicalEra::GalaxyFormation => "cosmo::galaxy_formation",
            CosmologicalEra::Present => "cosmo::present",
        }
    }
}

/// Dark matter candidate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DarkMatterCandidate {
    WIMP,            // Weakly interacting massive particle
    Axion,           // Light pseudoscalar
    SterileNeutrino, // Sterile neutrino
    PrimordialBlackHole,
    MACHO, // Massive compact halo object
}

impl DarkMatterCandidate {
    fn domain(&self) -> &'static str {
        match self {
            DarkMatterCandidate::WIMP => "dm::wimp",
            DarkMatterCandidate::Axion => "dm::axion",
            DarkMatterCandidate::SterileNeutrino => "dm::sterile_neutrino",
            DarkMatterCandidate::PrimordialBlackHole => "dm::pbh",
            DarkMatterCandidate::MACHO => "dm::macho",
        }
    }
}

/// Dark energy model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DarkEnergyModel {
    CosmologicalConstant, // Λ, w = -1
    Quintessence,         // Scalar field, w > -1
    PhantomEnergy,        // w < -1
}

impl DarkEnergyModel {
    fn domain(&self) -> &'static str {
        match self {
            DarkEnergyModel::CosmologicalConstant => "de::lambda",
            DarkEnergyModel::Quintessence => "de::quintessence",
            DarkEnergyModel::PhantomEnergy => "de::phantom",
        }
    }

    /// Equation of state parameter w
    pub fn w(&self) -> f64 {
        match self {
            DarkEnergyModel::CosmologicalConstant => -1.0,
            DarkEnergyModel::Quintessence => -0.9,
            DarkEnergyModel::PhantomEnergy => -1.1,
        }
    }
}

/// Structure type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CosmicStructure {
    Galaxy,
    GalaxyCluster,
    Supercluster,
    Filament,
    Void,
    Wall,
}

impl CosmicStructure {
    fn domain(&self) -> &'static str {
        match self {
            CosmicStructure::Galaxy => "structure::galaxy",
            CosmicStructure::GalaxyCluster => "structure::cluster",
            CosmicStructure::Supercluster => "structure::supercluster",
            CosmicStructure::Filament => "structure::filament",
            CosmicStructure::Void => "structure::void",
            CosmicStructure::Wall => "structure::wall",
        }
    }
}

/// The cosmic microwave background
#[derive(Debug, Clone)]
pub struct CMB {
    /// Temperature in Kelvin
    pub temperature_k: f64,
    /// Angular power spectrum peak
    pub first_peak_l: u32,
    /// Fractional fluctuation amplitude
    pub delta_t_over_t: f64,
    pub vector: ContinuousHV,
}

/// Cosmological parameters
#[derive(Debug, Clone)]
pub struct CosmologicalParameters {
    /// Hubble constant (km/s/Mpc)
    pub h0: f64,
    /// Total matter density
    pub omega_m: f64,
    /// Dark energy density
    pub omega_lambda: f64,
    /// Baryon density
    pub omega_b: f64,
    /// Curvature density
    pub omega_k: f64,
    /// Scalar spectral index
    pub n_s: f64,
    /// Amplitude of fluctuations
    pub sigma_8: f64,
    pub vector: ContinuousHV,
}

/// Cosmology encoder
#[derive(Debug, Clone)]
pub struct CosmologyEncoder {
    // Universe concepts
    pub universe: ContinuousHV,
    pub big_bang: ContinuousHV,
    pub expansion: ContinuousHV,
    pub hubble: ContinuousHV,

    // Eras
    pub inflation: ContinuousHV,
    pub recombination: ContinuousHV,
    pub nucleosynthesis: ContinuousHV,

    // Matter/energy content
    pub dark_matter: ContinuousHV,
    pub dark_energy: ContinuousHV,
    pub baryonic: ContinuousHV,
    pub radiation: ContinuousHV,

    // CMB
    pub cmb: ContinuousHV,
    pub anisotropy: ContinuousHV,
    pub acoustic_peak: ContinuousHV,

    // Structure
    pub density_perturbation: ContinuousHV,
    pub gravitational_collapse: ContinuousHV,
    pub galaxy: ContinuousHV,
    pub cluster: ContinuousHV,
    pub cosmic_web: ContinuousHV,

    // Geometry
    pub flat: ContinuousHV,
    pub open: ContinuousHV,
    pub closed: ContinuousHV,

    // Horizons
    pub horizon: ContinuousHV,
    pub particle_horizon: ContinuousHV,
    pub event_horizon: ContinuousHV,
}

impl CosmologyEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            universe: genesis.hv("cosmo::universe", PHYSICS_DIM),
            big_bang: genesis.hv("cosmo::big_bang", PHYSICS_DIM),
            expansion: genesis.hv("cosmo::expansion", PHYSICS_DIM),
            hubble: genesis.hv("cosmo::hubble", PHYSICS_DIM),

            inflation: genesis.hv(CosmologicalEra::Inflation.domain(), PHYSICS_DIM),
            recombination: genesis.hv(CosmologicalEra::Recombination.domain(), PHYSICS_DIM),
            nucleosynthesis: genesis.hv(CosmologicalEra::Nucleosynthesis.domain(), PHYSICS_DIM),

            dark_matter: genesis.hv("cosmo::dark_matter", PHYSICS_DIM),
            dark_energy: genesis.hv("cosmo::dark_energy", PHYSICS_DIM),
            baryonic: genesis.hv("cosmo::baryonic", PHYSICS_DIM),
            radiation: genesis.hv("cosmo::radiation", PHYSICS_DIM),

            cmb: genesis.hv("cosmo::cmb", PHYSICS_DIM),
            anisotropy: genesis.hv("cosmo::anisotropy", PHYSICS_DIM),
            acoustic_peak: genesis.hv("cosmo::acoustic_peak", PHYSICS_DIM),

            density_perturbation: genesis.hv("cosmo::density_perturbation", PHYSICS_DIM),
            gravitational_collapse: genesis.hv("cosmo::gravitational_collapse", PHYSICS_DIM),
            galaxy: genesis.hv(CosmicStructure::Galaxy.domain(), PHYSICS_DIM),
            cluster: genesis.hv(CosmicStructure::GalaxyCluster.domain(), PHYSICS_DIM),
            cosmic_web: genesis.hv("cosmo::cosmic_web", PHYSICS_DIM),

            flat: genesis.hv("geometry::flat", PHYSICS_DIM),
            open: genesis.hv("geometry::open", PHYSICS_DIM),
            closed: genesis.hv("geometry::closed", PHYSICS_DIM),

            horizon: genesis.hv("cosmo::horizon", PHYSICS_DIM),
            particle_horizon: genesis.hv("cosmo::particle_horizon", PHYSICS_DIM),
            event_horizon: genesis.hv("cosmo::event_horizon", PHYSICS_DIM),
        }
    }

    /// Hubble parameter at redshift z: H(z) = H₀ √(Ω_m(1+z)³ + Ω_Λ)
    pub fn hubble_parameter(&self, z: f64, omega_m: f64, omega_lambda: f64) -> f64 {
        H0_KM_S_MPC * (omega_m * (1.0 + z).powi(3) + omega_lambda).sqrt()
    }

    /// Comoving distance to redshift z (simplified flat universe)
    pub fn comoving_distance_mpc(&self, z: f64) -> f64 {
        // Simplified: d_c ≈ c z / H₀ for z << 1
        let c_km_s = 299792.458;
        c_km_s * z / H0_KM_S_MPC
    }

    /// Scale factor from redshift: a = 1/(1+z)
    pub fn scale_factor(&self, z: f64) -> f64 {
        1.0 / (1.0 + z)
    }

    /// Create CMB
    pub fn create_cmb(&self) -> CMB {
        let vector = self.cmb.bind(&self.recombination).bind(&self.anisotropy);

        CMB {
            temperature_k: T_CMB,
            first_peak_l: 220,
            delta_t_over_t: 1e-5,
            vector,
        }
    }

    /// Create standard ΛCDM parameters (Planck 2018)
    pub fn lcdm_parameters(&self) -> CosmologicalParameters {
        let vector = self
            .universe
            .bind(&self.dark_energy)
            .bind(&self.dark_matter)
            .bind(&self.flat);

        CosmologicalParameters {
            h0: 67.4,
            omega_m: 0.315,
            omega_lambda: 0.685,
            omega_b: 0.049,
            omega_k: 0.0, // Flat
            n_s: 0.965,
            sigma_8: 0.811,
            vector,
        }
    }

    /// Check if universe is flat: |Ω_k| < 0.01
    pub fn is_flat(&self, omega_k: f64) -> bool {
        omega_k.abs() < 0.01
    }

    /// Age of universe for flat ΛCDM (approximate)
    pub fn age_gyr(&self, h0: f64, omega_m: f64, omega_lambda: f64) -> f64 {
        // Approximate formula for flat ΛCDM
        let h = h0 / 100.0;
        (2.0 / 3.0) * (1.0 / omega_lambda.sqrt()) * ((omega_lambda / omega_m).sqrt()).asinh()
            / (h * 100.0)
            * 977.8
    }

    /// Encode an era
    pub fn encode_era(&self, era: CosmologicalEra, genesis: &GenesisSeed) -> ContinuousHV {
        genesis
            .hv(era.domain(), PHYSICS_DIM)
            .bind(&self.universe)
            .bind(&self.big_bang)
    }

    /// Inflation: exponential expansion
    pub fn inflation_expansion(&self) -> ContinuousHV {
        self.inflation.bind(&self.expansion).scale(60.0) // ~60 e-folds
    }

    /// Dark matter vector
    pub fn dark_matter_candidate(
        &self,
        candidate: DarkMatterCandidate,
        genesis: &GenesisSeed,
    ) -> ContinuousHV {
        let candidate_vec = genesis.hv(candidate.domain(), PHYSICS_DIM);
        self.dark_matter.bind(&candidate_vec)
    }

    /// Dark energy with equation of state
    pub fn dark_energy_model(&self, model: DarkEnergyModel, genesis: &GenesisSeed) -> ContinuousHV {
        let model_vec = genesis.hv(model.domain(), PHYSICS_DIM);
        self.dark_energy.bind(&model_vec).scale(model.w() as f32)
    }

    /// Cosmic structure at given scale
    pub fn cosmic_structure(
        &self,
        structure: CosmicStructure,
        size_mpc: f64,
        genesis: &GenesisSeed,
    ) -> ContinuousHV {
        let structure_vec = genesis.hv(structure.domain(), PHYSICS_DIM);
        self.cosmic_web
            .bind(&structure_vec)
            .scale(size_mpc.ln() as f32)
    }

    /// Friedmann equations (first): (ȧ/a)² = 8πGρ/3 - k/a² + Λ/3
    pub fn friedmann_first(&self) -> ContinuousHV {
        ContinuousHV::bundle(&[
            &self.expansion,
            &self.hubble,
            &ContinuousHV::bundle(&[&self.baryonic, &self.dark_matter, &self.radiation]),
            &self.dark_energy,
        ])
    }

    /// Jeans mass: M_J ~ T^(3/2) / ρ^(1/2)
    pub fn jeans_mass_solar(&self, temp_k: f64, density_kg_m3: f64) -> f64 {
        // Simplified estimate
        1e5 * (temp_k / 1000.0).powf(1.5) / (density_kg_m3 / 1e-20).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (CosmologyEncoder, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("cosmology test");
        let encoder = CosmologyEncoder::from_genesis(&genesis);
        (encoder, genesis)
    }

    #[test]
    fn test_cosmology_encoder_creation() {
        let (encoder, _) = setup();
        assert!(encoder.universe.norm() > 0.0);
        assert!(encoder.dark_matter.norm() > 0.0);
    }

    #[test]
    fn test_hubble_parameter() {
        let (encoder, _) = setup();

        // At z=0, H(0) = H₀
        let h0 = encoder.hubble_parameter(0.0, 0.3, 0.7);
        assert!((h0 - H0_KM_S_MPC).abs() < 1.0);

        // At higher z, H(z) > H₀
        let hz = encoder.hubble_parameter(1.0, 0.3, 0.7);
        assert!(hz > h0);
    }

    #[test]
    fn test_scale_factor() {
        let (encoder, _) = setup();

        // At z=0, a=1
        assert!((encoder.scale_factor(0.0) - 1.0).abs() < 0.001);

        // At z=1, a=0.5
        assert!((encoder.scale_factor(1.0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_cmb() {
        let (encoder, _) = setup();
        let cmb = encoder.create_cmb();

        assert!((cmb.temperature_k - T_CMB).abs() < 0.001);
        assert_eq!(cmb.first_peak_l, 220);
    }

    #[test]
    fn test_lcdm_parameters() {
        let (encoder, _) = setup();
        let params = encoder.lcdm_parameters();

        // Ω_m + Ω_Λ ≈ 1 (flat)
        assert!((params.omega_m + params.omega_lambda - 1.0).abs() < 0.01);
        assert!(encoder.is_flat(params.omega_k));
    }

    #[test]
    fn test_comoving_distance() {
        let (encoder, _) = setup();

        // z=0.1 should give ~400 Mpc
        let d = encoder.comoving_distance_mpc(0.1);
        assert!(d > 300.0 && d < 500.0);
    }

    #[test]
    fn test_inflation() {
        let (encoder, _) = setup();
        let inflation = encoder.inflation_expansion();

        assert!(inflation.norm() > 0.0);
    }

    #[test]
    fn test_dark_matter_candidates() {
        let (encoder, genesis) = setup();

        let wimp = encoder.dark_matter_candidate(DarkMatterCandidate::WIMP, &genesis);
        let axion = encoder.dark_matter_candidate(DarkMatterCandidate::Axion, &genesis);

        // Different candidates should give different vectors
        let sim = wimp.similarity(&axion);
        assert!(sim < 0.95, "Different DM candidates should differ");
    }

    #[test]
    fn test_dark_energy_models() {
        let (encoder, genesis) = setup();

        let lambda = encoder.dark_energy_model(DarkEnergyModel::CosmologicalConstant, &genesis);
        let quint = encoder.dark_energy_model(DarkEnergyModel::Quintessence, &genesis);

        // Λ has w = -1
        assert!((DarkEnergyModel::CosmologicalConstant.w() - (-1.0)).abs() < 0.01);

        let sim = lambda.similarity(&quint);
        assert!(sim < 0.95);
    }

    #[test]
    fn test_cosmic_structure() {
        let (encoder, genesis) = setup();

        let galaxy = encoder.cosmic_structure(CosmicStructure::Galaxy, 0.03, &genesis);
        let cluster = encoder.cosmic_structure(CosmicStructure::GalaxyCluster, 3.0, &genesis);

        assert!(galaxy.norm() > 0.0);
        assert!(cluster.norm() > 0.0);
    }

    #[test]
    fn test_friedmann() {
        let (encoder, _) = setup();
        let friedmann = encoder.friedmann_first();

        assert!(friedmann.norm() > 0.0);
    }

    #[test]
    fn test_eras() {
        let (encoder, genesis) = setup();

        let inflation = encoder.encode_era(CosmologicalEra::Inflation, &genesis);
        let present = encoder.encode_era(CosmologicalEra::Present, &genesis);

        // Different eras should have different vectors
        let sim = inflation.similarity(&present);
        assert!(sim < 0.9, "Different eras should differ");
    }
}
