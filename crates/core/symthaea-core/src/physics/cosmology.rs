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

/// Common BAO distance observables at a single redshift.
///
/// Distances are reported in units of the supplied sound horizon r_d. This
/// structure is a background-geometry calculation only: it does not compute
/// the sound horizon from recombination physics or evaluate a survey likelihood.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaoDistanceObservables {
    /// Redshift at which the observables are evaluated.
    pub redshift: f64,
    /// Sound horizon at the drag epoch, supplied by the caller [Mpc].
    pub sound_horizon_mpc: f64,
    /// Transverse comoving distance divided by r_d: D_M(z) / r_d.
    pub dm_over_rd: f64,
    /// Hubble distance divided by r_d: c / (H(z) r_d).
    pub dh_over_rd: f64,
    /// Volume-averaged BAO distance divided by r_d:
    /// D_V(z) / r_d = [z D_M(z)^2 D_H(z)]^(1/3) / r_d.
    pub dv_over_rd: f64,
}

/// Parameters for the homogeneous expansion history of an FLRW cosmology.
///
/// Density parameters use their conventional present-day normalization:
/// Ω_r + Ω_m + Ω_k + Ω_de = 1. The w0 parameter assumes constant dark-energy
/// equation of state, p_de = w0 rho_de. This is a background-distance model,
/// not a Boltzmann solver: it does not predict CMB power spectra or nonlinear
/// structure.
///
/// The default below is intentionally a radiation-free flat ΛCDM approximation
/// for comparison with the pre-existing simplified encoder, not a precision
/// Planck likelihood configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpansionHistory {
    /// Present-day Hubble constant [km s⁻¹ Mpc⁻¹].
    pub h0_km_s_mpc: f64,
    /// Present-day radiation density parameter Ω_r.
    pub omega_r: f64,
    /// Present-day nonrelativistic matter density parameter Ω_m.
    pub omega_m: f64,
    /// Curvature density parameter Ω_k; positive is open, negative is closed.
    pub omega_k: f64,
    /// Present-day dark-energy density parameter Ω_de.
    pub omega_de: f64,
    /// Constant dark-energy equation-of-state parameter w0.
    pub w0: f64,
}

/// Input or numerical-domain errors for a background cosmology calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpansionHistoryError {
    NonFiniteParameter,
    NonPositiveHubbleConstant,
    NegativeDensityParameter,
    ClosureViolation,
    InvalidRedshift,
    InvalidIntegrationIntervals,
    InvalidIntegrationTolerance,
    InvalidIntegrationDepth,
    IntegrationDidNotConverge,
    InvalidSoundHorizon,
    NonPhysicalExpansionRate,
    NumericalOverflow,
}

impl std::fmt::Display for ExpansionHistoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::NonFiniteParameter => "cosmological parameters must be finite",
            Self::NonPositiveHubbleConstant => "H0 must be positive",
            Self::NegativeDensityParameter => {
                "matter, radiation, and dark-energy densities must be non-negative"
            }
            Self::ClosureViolation => "density parameters must satisfy Ωr + Ωm + Ωk + Ωde = 1",
            Self::InvalidRedshift => "redshift must be finite and >= 0",
            Self::InvalidIntegrationIntervals => {
                "Simpson integration requires an even interval count in 2..=1_000_000"
            }
            Self::InvalidIntegrationTolerance => {
                "absolute tolerance must be positive; relative tolerance must be finite and in (0, 1)"
            }
            Self::InvalidIntegrationDepth => "adaptive integration depth must be in 1..=20",
            Self::IntegrationDidNotConverge => {
                "adaptive Simpson integration did not meet its tolerance before the depth limit"
            }
            Self::InvalidSoundHorizon => "sound horizon r_d must be finite and positive",
            Self::NonPhysicalExpansionRate => {
                "the model produced a non-positive or non-finite H(z)^2"
            }
            Self::NumericalOverflow => "the requested cosmology calculation overflowed",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ExpansionHistoryError {}

impl ExpansionHistory {
    /// A convenient flat ΛCDM background approximating the parameters already
    /// used by CosmologyEncoder. Radiation is omitted deliberately; use an
    /// explicitly configured ExpansionHistory for early-universe work.
    pub const fn flat_lcdm_approx() -> Self {
        Self {
            h0_km_s_mpc: H0_KM_S_MPC,
            omega_r: 0.0,
            omega_m: 0.315,
            omega_k: 0.0,
            omega_de: 0.685,
            w0: -1.0,
        }
    }

    /// Validate finite inputs, non-negative physical density terms, and the
    /// FLRW density-parameter closure relation to an absolute tolerance of 1e-6.
    pub fn validate(&self) -> Result<(), ExpansionHistoryError> {
        let values = [
            self.h0_km_s_mpc,
            self.omega_r,
            self.omega_m,
            self.omega_k,
            self.omega_de,
            self.w0,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ExpansionHistoryError::NonFiniteParameter);
        }
        if self.h0_km_s_mpc <= 0.0 {
            return Err(ExpansionHistoryError::NonPositiveHubbleConstant);
        }
        if self.omega_r < 0.0 || self.omega_m < 0.0 || self.omega_de < 0.0 {
            return Err(ExpansionHistoryError::NegativeDensityParameter);
        }
        let closure = self.omega_r + self.omega_m + self.omega_k + self.omega_de;
        if !closure.is_finite() {
            return Err(ExpansionHistoryError::NumericalOverflow);
        }
        if (closure - 1.0).abs() > 1e-6 {
            return Err(ExpansionHistoryError::ClosureViolation);
        }
        Ok(())
    }

    fn validate_redshift(z: f64) -> Result<(), ExpansionHistoryError> {
        if z.is_finite() && z >= 0.0 {
            Ok(())
        } else {
            Err(ExpansionHistoryError::InvalidRedshift)
        }
    }

    /// Dimensionless expansion rate E(z)^2 = H(z)^2 / H0^2 for constant-w
    /// dark energy in an FLRW background, including radiation and curvature.
    fn e_squared_unchecked(&self, z: f64) -> f64 {
        let one_plus_z = 1.0 + z;
        let radiation = if self.omega_r == 0.0 {
            0.0
        } else {
            self.omega_r * one_plus_z.powi(4)
        };
        let matter = if self.omega_m == 0.0 {
            0.0
        } else {
            self.omega_m * one_plus_z.powi(3)
        };
        let curvature = if self.omega_k == 0.0 {
            0.0
        } else {
            self.omega_k * one_plus_z.powi(2)
        };
        let dark_energy = if self.omega_de == 0.0 {
            0.0
        } else {
            self.omega_de * one_plus_z.powf(3.0 * (1.0 + self.w0))
        };
        radiation + matter + curvature + dark_energy
    }

    /// Return E(z)^2 = H(z)^2 / H0^2.
    pub fn e_squared(&self, z: f64) -> Result<f64, ExpansionHistoryError> {
        self.validate()?;
        Self::validate_redshift(z)?;
        let value = self.e_squared_unchecked(z);
        if value.is_finite() && value > 0.0 {
            Ok(value)
        } else {
            Err(ExpansionHistoryError::NonPhysicalExpansionRate)
        }
    }

    /// Hubble parameter H(z) [km s⁻¹ Mpc⁻¹].
    pub fn hubble_km_s_mpc(&self, z: f64) -> Result<f64, ExpansionHistoryError> {
        let value = self.h0_km_s_mpc * self.e_squared(z)?.sqrt();
        if value.is_finite() && value > 0.0 {
            Ok(value)
        } else {
            Err(ExpansionHistoryError::NumericalOverflow)
        }
    }

    /// Line-of-sight comoving distance χ(z) [Mpc], computed by composite
    /// Simpson integration of c/H(z). The interval count must be even.
    ///
    /// This is the radial comoving distance; curvature is accounted for only
    /// when converting it to transverse distance in transverse_comoving_distance_mpc.
    pub fn line_of_sight_comoving_distance_mpc(
        &self,
        z: f64,
        intervals: usize,
    ) -> Result<f64, ExpansionHistoryError> {
        self.validate()?;
        Self::validate_redshift(z)?;
        if intervals < 2 || intervals > 1_000_000 || intervals % 2 != 0 {
            return Err(ExpansionHistoryError::InvalidIntegrationIntervals);
        }
        if z == 0.0 {
            return Ok(0.0);
        }

        const C_KM_S: f64 = 299_792.458;
        let step = z / intervals as f64;
        let integrand = |redshift: f64| -> Result<f64, ExpansionHistoryError> {
            let e2 = self.e_squared_unchecked(redshift);
            if !e2.is_finite() || e2 <= 0.0 {
                return Err(ExpansionHistoryError::NonPhysicalExpansionRate);
            }
            let h = self.h0_km_s_mpc * e2.sqrt();
            let value = C_KM_S / h;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(ExpansionHistoryError::NumericalOverflow)
            }
        };

        let mut sum = integrand(0.0)? + integrand(z)?;
        for i in 1..intervals {
            let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += weight * integrand(step * i as f64)?;
        }
        let distance = step * sum / 3.0;
        if distance.is_finite() && distance >= 0.0 {
            Ok(distance)
        } else {
            Err(ExpansionHistoryError::NumericalOverflow)
        }
    }

    /// Adaptive line-of-sight comoving distance χ(z) [Mpc].
    ///
    /// The integrator recursively refines a Simpson estimate and reports a
    /// non-convergence error instead of silently returning an under-resolved
    /// value. Absolute tolerance is in Mpc; relative tolerance is dimensionless.
    pub fn line_of_sight_comoving_distance_mpc_adaptive(
        &self,
        z: f64,
        absolute_tolerance_mpc: f64,
        relative_tolerance: f64,
        max_depth: u32,
    ) -> Result<f64, ExpansionHistoryError> {
        self.validate()?;
        Self::validate_redshift(z)?;
        validate_adaptive_options(absolute_tolerance_mpc, relative_tolerance, max_depth)?;
        if z == 0.0 {
            return Ok(0.0);
        }

        const C_KM_S: f64 = 299_792.458;
        adaptive_simpson(
            |redshift| {
                let e2 = self.e_squared_unchecked(redshift);
                if !e2.is_finite() || e2 <= 0.0 {
                    return Err(ExpansionHistoryError::NonPhysicalExpansionRate);
                }
                let value = C_KM_S / (self.h0_km_s_mpc * e2.sqrt());
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(ExpansionHistoryError::NumericalOverflow)
                }
            },
            0.0,
            z,
            absolute_tolerance_mpc,
            relative_tolerance,
            max_depth,
        )
    }

    /// Transverse comoving distance D_M(z) [Mpc], applying the FLRW curvature
    /// mapping to the line-of-sight distance.
    pub fn transverse_comoving_distance_mpc(
        &self,
        z: f64,
        intervals: usize,
    ) -> Result<f64, ExpansionHistoryError> {
        let radial = self.line_of_sight_comoving_distance_mpc(z, intervals)?;
        if self.omega_k.abs() < 1e-12 {
            return Ok(radial);
        }
        const C_KM_S: f64 = 299_792.458;
        let hubble_distance = C_KM_S / self.h0_km_s_mpc;
        let curvature_scale = self.omega_k.abs().sqrt();
        let x = curvature_scale * radial / hubble_distance;
        let distance = if self.omega_k > 0.0 {
            // Ω_k > 0 is open: D_M = D_H sinh(sqrt(Ω_k) χ / D_H)/sqrt(Ω_k).
            hubble_distance * x.sinh() / curvature_scale
        } else {
            // Ω_k < 0 is closed: D_M = D_H sin(sqrt(|Ω_k|) χ / D_H)/sqrt(|Ω_k|).
            hubble_distance * x.sin() / curvature_scale
        };
        if distance.is_finite() && distance >= 0.0 {
            Ok(distance)
        } else {
            Err(ExpansionHistoryError::NumericalOverflow)
        }
    }

    /// Luminosity distance D_L(z) [Mpc] = (1 + z) D_M(z).
    pub fn luminosity_distance_mpc(
        &self,
        z: f64,
        intervals: usize,
    ) -> Result<f64, ExpansionHistoryError> {
        Self::validate_redshift(z)?;
        let distance = (1.0 + z) * self.transverse_comoving_distance_mpc(z, intervals)?;
        if distance.is_finite() {
            Ok(distance)
        } else {
            Err(ExpansionHistoryError::NumericalOverflow)
        }
    }

    /// Angular-diameter distance D_A(z) [Mpc] = D_M(z) / (1 + z).
    pub fn angular_diameter_distance_mpc(
        &self,
        z: f64,
        intervals: usize,
    ) -> Result<f64, ExpansionHistoryError> {
        Self::validate_redshift(z)?;
        Ok(self.transverse_comoving_distance_mpc(z, intervals)? / (1.0 + z))
    }

    /// Standard BAO distance observables at redshift z.
    ///
    /// The sound horizon r_d is supplied in Mpc. It is not inferred or
    /// computed here: a precision DESI likelihood must model or calibrate r_d
    /// consistently with its early-universe assumptions.
    pub fn bao_distance_observables(
        &self,
        z: f64,
        sound_horizon_mpc: f64,
        intervals: usize,
    ) -> Result<BaoDistanceObservables, ExpansionHistoryError> {
        Self::validate_redshift(z)?;
        if !sound_horizon_mpc.is_finite() || sound_horizon_mpc <= 0.0 {
            return Err(ExpansionHistoryError::InvalidSoundHorizon);
        }

        const C_KM_S: f64 = 299_792.458;
        let transverse_distance = self.transverse_comoving_distance_mpc(z, intervals)?;
        let hubble_rate = self.hubble_km_s_mpc(z)?;
        let hubble_distance = C_KM_S / hubble_rate;
        let volume_distance =
            (z * transverse_distance * transverse_distance * hubble_distance).cbrt();

        let dm_over_rd = transverse_distance / sound_horizon_mpc;
        let dh_over_rd = hubble_distance / sound_horizon_mpc;
        let dv_over_rd = volume_distance / sound_horizon_mpc;
        if [dm_over_rd, dh_over_rd, dv_over_rd]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        {
            Ok(BaoDistanceObservables {
                redshift: z,
                sound_horizon_mpc,
                dm_over_rd,
                dh_over_rd,
                dv_over_rd,
            })
        } else {
            Err(ExpansionHistoryError::NumericalOverflow)
        }
    }

    /// Adaptive cosmological lookback time from redshift z to the present [Gyr].
    ///
    /// Absolute tolerance is in Gyr; relative tolerance is dimensionless.
    /// This computes lookback time, not the total age of the universe.
    pub fn lookback_time_gyr_adaptive(
        &self,
        z: f64,
        absolute_tolerance_gyr: f64,
        relative_tolerance: f64,
        max_depth: u32,
    ) -> Result<f64, ExpansionHistoryError> {
        self.validate()?;
        Self::validate_redshift(z)?;
        validate_adaptive_options(absolute_tolerance_gyr, relative_tolerance, max_depth)?;
        if z == 0.0 {
            return Ok(0.0);
        }

        const HUBBLE_TIME_CONVERSION_GYR: f64 = 977.792_221_680_789_2;
        let conversion = HUBBLE_TIME_CONVERSION_GYR / self.h0_km_s_mpc;
        adaptive_simpson(
            |redshift| {
                let e2 = self.e_squared_unchecked(redshift);
                if !e2.is_finite() || e2 <= 0.0 {
                    return Err(ExpansionHistoryError::NonPhysicalExpansionRate);
                }
                let value = conversion / ((1.0 + redshift) * e2.sqrt());
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(ExpansionHistoryError::NumericalOverflow)
                }
            },
            0.0,
            z,
            absolute_tolerance_gyr,
            relative_tolerance,
            max_depth,
        )
    }

    /// Cosmological lookback time from redshift z to the present [Gyr].
    ///
    /// Integrates dt = dz / ((1 + z) H(z)) using composite Simpson
    /// integration. H0 is in km s^-1 Mpc^-1, so the dimensionless integral is
    /// multiplied by the conversion 977.7922216807892 / H0 to obtain Gyr.
    /// This calculates lookback time, not the total age of the universe; the
    /// radiation-free default is not intended for precision early-universe work.
    pub fn lookback_time_gyr(
        &self,
        z: f64,
        intervals: usize,
    ) -> Result<f64, ExpansionHistoryError> {
        self.validate()?;
        Self::validate_redshift(z)?;
        if intervals < 2 || intervals > 1_000_000 || intervals % 2 != 0 {
            return Err(ExpansionHistoryError::InvalidIntegrationIntervals);
        }
        if z == 0.0 {
            return Ok(0.0);
        }

        const HUBBLE_TIME_CONVERSION_GYR: f64 = 977.792_221_680_789_2;
        let step = z / intervals as f64;
        let integrand = |redshift: f64| -> Result<f64, ExpansionHistoryError> {
            let e2 = self.e_squared_unchecked(redshift);
            if !e2.is_finite() || e2 <= 0.0 {
                return Err(ExpansionHistoryError::NonPhysicalExpansionRate);
            }
            let value = 1.0 / ((1.0 + redshift) * e2.sqrt());
            if value.is_finite() {
                Ok(value)
            } else {
                Err(ExpansionHistoryError::NumericalOverflow)
            }
        };

        let mut sum = integrand(0.0)? + integrand(z)?;
        for i in 1..intervals {
            let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += weight * integrand(step * i as f64)?;
        }
        let lookback_gyr = (HUBBLE_TIME_CONVERSION_GYR / self.h0_km_s_mpc) * step * sum / 3.0;
        if lookback_gyr.is_finite() && lookback_gyr >= 0.0 {
            Ok(lookback_gyr)
        } else {
            Err(ExpansionHistoryError::NumericalOverflow)
        }
    }
}

#[cfg(test)]
mod expansion_history_tests {
    use super::*;

    const C_KM_S: f64 = 299_792.458;

    fn einstein_de_sitter() -> ExpansionHistory {
        ExpansionHistory {
            h0_km_s_mpc: 70.0,
            omega_r: 0.0,
            omega_m: 1.0,
            omega_k: 0.0,
            omega_de: 0.0,
            w0: -1.0,
        }
    }

    #[test]
    fn hubble_rate_matches_einstein_de_sitter_analytic_solution() {
        // Ω_m = 1 gives E(z)^2 = (1 + z)^3.
        let model = einstein_de_sitter();
        let expected = 70.0 * 8.0_f64.sqrt();
        let actual = model.hubble_km_s_mpc(1.0).unwrap();
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn comoving_distance_matches_einstein_de_sitter_analytic_solution() {
        // χ(z) = 2c/H0 * (1 - 1/sqrt(1+z)) for a matter-only flat universe.
        let model = einstein_de_sitter();
        let z: f64 = 1.0;
        let expected = 2.0 * C_KM_S / 70.0 * (1.0 - 1.0 / (1.0 + z).sqrt());
        let actual = model.line_of_sight_comoving_distance_mpc(z, 4096).unwrap();
        assert!(
            (actual - expected).abs() / expected < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn de_sitter_distance_is_exact_for_constant_hubble_rate() {
        let model = ExpansionHistory {
            h0_km_s_mpc: 70.0,
            omega_r: 0.0,
            omega_m: 0.0,
            omega_k: 0.0,
            omega_de: 1.0,
            w0: -1.0,
        };
        let z = 2.0;
        let expected = C_KM_S / 70.0 * z;
        let actual = model.line_of_sight_comoving_distance_mpc(z, 128).unwrap();
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn lookback_time_matches_einstein_de_sitter_analytic_solution() {
        let model = einstein_de_sitter();
        let z: f64 = 1.0;
        let hubble_time_gyr = 977.792_221_680_789_2 / 70.0;
        let expected = (2.0 / 3.0) * hubble_time_gyr * (1.0 - (1.0 + z).powf(-1.5));
        let actual = model.lookback_time_gyr(z, 4096).unwrap();
        assert!(
            (actual - expected).abs() / expected < 1e-9,
            "expected {expected}, got {actual}"
        );
        assert_eq!(model.lookback_time_gyr(0.0, 128).unwrap(), 0.0);
    }

    #[test]
    fn de_sitter_lookback_time_matches_logarithmic_solution() {
        let model = ExpansionHistory {
            h0_km_s_mpc: 70.0,
            omega_r: 0.0,
            omega_m: 0.0,
            omega_k: 0.0,
            omega_de: 1.0,
            w0: -1.0,
        };
        let z: f64 = 2.0;
        let expected = (977.792_221_680_789_2 / 70.0) * (1.0 + z).ln();
        let actual = model.lookback_time_gyr(z, 1024).unwrap();
        assert!((actual - expected).abs() < 1e-9);
    }

    #[test]
    fn radiation_dominated_distance_and_lookback_match_analytic_solutions() {
        let model = ExpansionHistory {
            h0_km_s_mpc: 70.0,
            omega_r: 1.0,
            omega_m: 0.0,
            omega_k: 0.0,
            omega_de: 0.0,
            w0: -1.0,
        };
        let z: f64 = 3.0;
        let e2_expected = (1.0 + z).powi(4);
        assert!((model.e_squared(z).unwrap() - e2_expected).abs() < 1e-12);

        let distance_expected = C_KM_S / 70.0 * z / (1.0 + z);
        let distance = model.line_of_sight_comoving_distance_mpc(z, 4096).unwrap();
        assert!(
            (distance - distance_expected).abs() / distance_expected < 1e-9,
            "expected {distance_expected}, got {distance}"
        );

        let hubble_time_gyr = 977.792_221_680_789_2 / 70.0;
        let lookback_expected = 0.5 * hubble_time_gyr * (1.0 - (1.0 + z).powi(-2));
        let lookback = model.lookback_time_gyr(z, 4096).unwrap();
        assert!(
            (lookback - lookback_expected).abs() / lookback_expected < 1e-9,
            "expected {lookback_expected}, got {lookback}"
        );
    }

    #[test]
    fn distance_duality_relation_holds_by_construction() {
        let model = ExpansionHistory::flat_lcdm_approx();
        let z = 0.8;
        let dl = model.luminosity_distance_mpc(z, 2048).unwrap();
        let da = model.angular_diameter_distance_mpc(z, 2048).unwrap();
        assert!((dl - (1.0 + z).powi(2) * da).abs() < 1e-9);
    }

    #[test]
    fn invalid_closure_and_redshift_fail_closed() {
        let mut bad = ExpansionHistory::flat_lcdm_approx();
        bad.omega_de = 0.5;
        assert_eq!(bad.validate(), Err(ExpansionHistoryError::ClosureViolation));
        assert_eq!(
            ExpansionHistory::flat_lcdm_approx().e_squared(-0.1),
            Err(ExpansionHistoryError::InvalidRedshift)
        );
        assert_eq!(
            ExpansionHistory::flat_lcdm_approx().line_of_sight_comoving_distance_mpc(1.0, 3),
            Err(ExpansionHistoryError::InvalidIntegrationIntervals)
        );

        let mut negative_density = ExpansionHistory::flat_lcdm_approx();
        negative_density.omega_m = -0.1;
        negative_density.omega_de = 1.1;
        assert_eq!(
            negative_density.validate(),
            Err(ExpansionHistoryError::NegativeDensityParameter)
        );

        let mut non_finite = ExpansionHistory::flat_lcdm_approx();
        non_finite.h0_km_s_mpc = f64::NAN;
        assert_eq!(
            non_finite.validate(),
            Err(ExpansionHistoryError::NonFiniteParameter)
        );

        assert_eq!(
            ExpansionHistory::flat_lcdm_approx().e_squared(f64::MAX),
            Err(ExpansionHistoryError::NonPhysicalExpansionRate)
        );
    }

    #[test]
    fn bao_distance_observables_match_flat_de_sitter_analytic_limit() {
        let model = ExpansionHistory {
            h0_km_s_mpc: 70.0,
            omega_r: 0.0,
            omega_m: 0.0,
            omega_k: 0.0,
            omega_de: 1.0,
            w0: -1.0,
        };
        let z: f64 = 1.0;
        let rd = 150.0;
        let hubble_distance = C_KM_S / 70.0;
        let transverse_distance = hubble_distance * z;
        let expected_dv = (z * transverse_distance.powi(2) * hubble_distance).cbrt();
        let actual = model.bao_distance_observables(z, rd, 128).unwrap();
        assert!((actual.dm_over_rd - transverse_distance / rd).abs() < 1e-12);
        assert!((actual.dh_over_rd - hubble_distance / rd).abs() < 1e-12);
        assert!((actual.dv_over_rd - expected_dv / rd).abs() < 1e-12);
        assert_eq!(actual.sound_horizon_mpc, rd);
    }

    #[test]
    fn bao_observables_reject_invalid_sound_horizon_and_bad_inputs() {
        let model = ExpansionHistory::flat_lcdm_approx();
        assert_eq!(
            model.bao_distance_observables(1.0, 0.0, 128),
            Err(ExpansionHistoryError::InvalidSoundHorizon)
        );
        assert_eq!(
            model.bao_distance_observables(1.0, f64::NAN, 128),
            Err(ExpansionHistoryError::InvalidSoundHorizon)
        );
        assert_eq!(
            model.bao_distance_observables(-0.1, 150.0, 128),
            Err(ExpansionHistoryError::InvalidRedshift)
        );
    }

    #[test]
    fn adaptive_integrals_match_analytic_matter_and_radiation_limits() {
        let eds = einstein_de_sitter();
        let z: f64 = 1.0;
        let expected_distance = 2.0 * C_KM_S / 70.0 * (1.0 - 1.0 / (1.0 + z).sqrt());
        let expected_time =
            (977.792_221_680_789_2 / 70.0) * (2.0 / 3.0) * (1.0 - (1.0 + z).powf(-1.5));
        let adaptive_distance = eds
            .line_of_sight_comoving_distance_mpc_adaptive(z, 1e-8, 1e-10, 20)
            .unwrap();
        let adaptive_time = eds.lookback_time_gyr_adaptive(z, 1e-10, 1e-10, 20).unwrap();
        assert!((adaptive_distance - expected_distance).abs() / expected_distance < 1e-9);
        assert!((adaptive_time - expected_time).abs() / expected_time < 1e-9);

        let radiation = ExpansionHistory {
            h0_km_s_mpc: 70.0,
            omega_r: 1.0,
            omega_m: 0.0,
            omega_k: 0.0,
            omega_de: 0.0,
            w0: -1.0,
        };
        let z: f64 = 3.0;
        let expected_distance = C_KM_S / 70.0 * z / (1.0 + z);
        let expected_time = (977.792_221_680_789_2 / 70.0) * 0.5 * (1.0 - (1.0 + z).powi(-2));
        let adaptive_distance = radiation
            .line_of_sight_comoving_distance_mpc_adaptive(z, 1e-8, 1e-10, 20)
            .unwrap();
        let adaptive_time = radiation
            .lookback_time_gyr_adaptive(z, 1e-10, 1e-10, 20)
            .unwrap();
        assert!((adaptive_distance - expected_distance).abs() / expected_distance < 1e-9);
        assert!((adaptive_time - expected_time).abs() / expected_time < 1e-9);
    }

    #[test]
    fn adaptive_and_fixed_grid_distances_agree_over_frozen_redshift_grid() {
        let model = ExpansionHistory::flat_lcdm_approx();
        for z in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0] {
            let fixed = model
                .line_of_sight_comoving_distance_mpc(z, 16_384)
                .unwrap();
            let adaptive = model
                .line_of_sight_comoving_distance_mpc_adaptive(z, 1e-7, 1e-10, 20)
                .unwrap();
            let relative_error = (fixed - adaptive).abs() / fixed.abs().max(1.0);
            assert!(
                relative_error < 1e-8,
                "z={z}: fixed={fixed}, adaptive={adaptive}, relerr={relative_error}"
            );
        }
    }

    #[test]
    fn adaptive_integrator_rejects_bad_tolerances_and_depth() {
        let model = ExpansionHistory::flat_lcdm_approx();
        assert_eq!(
            model.line_of_sight_comoving_distance_mpc_adaptive(1.0, 0.0, 1e-8, 10),
            Err(ExpansionHistoryError::InvalidIntegrationTolerance)
        );
        assert_eq!(
            model.lookback_time_gyr_adaptive(1.0, 1e-8, f64::NAN, 10),
            Err(ExpansionHistoryError::InvalidIntegrationTolerance)
        );
        assert_eq!(
            model.line_of_sight_comoving_distance_mpc_adaptive(1.0, 1e-8, 1e-8, 0),
            Err(ExpansionHistoryError::InvalidIntegrationDepth)
        );
    }

    #[test]
    fn adaptive_integrator_fails_closed_at_insufficient_depth() {
        let model = ExpansionHistory::flat_lcdm_approx();
        assert_eq!(
            model.line_of_sight_comoving_distance_mpc_adaptive(3.0, 1e-30, 1e-30, 1),
            Err(ExpansionHistoryError::IntegrationDidNotConverge)
        );
    }

    #[test]
    fn curvature_changes_transverse_distance_in_expected_direction() {
        let open = ExpansionHistory {
            h0_km_s_mpc: 70.0,
            omega_r: 0.0,
            omega_m: 0.3,
            omega_k: 0.1,
            omega_de: 0.6,
            w0: -1.0,
        };
        let closed = ExpansionHistory {
            h0_km_s_mpc: 70.0,
            omega_r: 0.0,
            omega_m: 0.3,
            omega_k: -0.1,
            omega_de: 0.8,
            w0: -1.0,
        };
        let z = 0.5;
        let open_radial = open.line_of_sight_comoving_distance_mpc(z, 2048).unwrap();
        let open_transverse = open.transverse_comoving_distance_mpc(z, 2048).unwrap();
        let closed_radial = closed.line_of_sight_comoving_distance_mpc(z, 2048).unwrap();
        let closed_transverse = closed.transverse_comoving_distance_mpc(z, 2048).unwrap();
        assert!(open_transverse > open_radial);
        assert!(closed_transverse < closed_radial);
    }
}

fn validate_adaptive_options(
    absolute_tolerance: f64,
    relative_tolerance: f64,
    max_depth: u32,
) -> Result<(), ExpansionHistoryError> {
    if !absolute_tolerance.is_finite()
        || absolute_tolerance <= 0.0
        || !relative_tolerance.is_finite()
        || relative_tolerance <= 0.0
        || relative_tolerance >= 1.0
    {
        return Err(ExpansionHistoryError::InvalidIntegrationTolerance);
    }
    if !(1..=20).contains(&max_depth) {
        return Err(ExpansionHistoryError::InvalidIntegrationDepth);
    }
    Ok(())
}

/// Adaptive Simpson quadrature with a Richardson error estimate.
///
/// The returned value is corrected by (S_refined - S_whole) / 15. If the
/// tolerance is not reached before max_depth, the function fails closed.
fn adaptive_simpson<F>(
    f: F,
    a: f64,
    b: f64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    max_depth: u32,
) -> Result<f64, ExpansionHistoryError>
where
    F: Fn(f64) -> Result<f64, ExpansionHistoryError>,
{
    let midpoint = a + (b - a) * 0.5;
    if !midpoint.is_finite() || midpoint <= a || midpoint >= b {
        return Err(ExpansionHistoryError::IntegrationDidNotConverge);
    }
    let fa = f(a)?;
    let fm = f(midpoint)?;
    let fb = f(b)?;
    let whole = simpson_estimate(a, b, fa, fm, fb);
    if !whole.is_finite() {
        return Err(ExpansionHistoryError::NumericalOverflow);
    }
    adaptive_simpson_recursive(
        &f,
        a,
        b,
        fa,
        fm,
        fb,
        whole,
        absolute_tolerance,
        relative_tolerance,
        max_depth,
    )
}

#[allow(clippy::too_many_arguments)]
fn adaptive_simpson_recursive<F>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fm: f64,
    fb: f64,
    whole: f64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    depth_remaining: u32,
) -> Result<f64, ExpansionHistoryError>
where
    F: Fn(f64) -> Result<f64, ExpansionHistoryError>,
{
    let midpoint = a + (b - a) * 0.5;
    let left_midpoint = a + (midpoint - a) * 0.5;
    let right_midpoint = midpoint + (b - midpoint) * 0.5;
    if left_midpoint <= a
        || left_midpoint >= midpoint
        || right_midpoint <= midpoint
        || right_midpoint >= b
    {
        return Err(ExpansionHistoryError::IntegrationDidNotConverge);
    }

    let flm = f(left_midpoint)?;
    let frm = f(right_midpoint)?;
    let left = simpson_estimate(a, midpoint, fa, flm, fm);
    let right = simpson_estimate(midpoint, b, fm, frm, fb);
    let refined = left + right;
    if !refined.is_finite() {
        return Err(ExpansionHistoryError::NumericalOverflow);
    }
    let error_estimate = (refined - whole).abs() / 15.0;
    let tolerance = absolute_tolerance + relative_tolerance * refined.abs();
    if !error_estimate.is_finite() || !tolerance.is_finite() {
        return Err(ExpansionHistoryError::NumericalOverflow);
    }
    if error_estimate <= tolerance {
        let corrected = refined + (refined - whole) / 15.0;
        if corrected.is_finite() {
            return Ok(corrected);
        }
        return Err(ExpansionHistoryError::NumericalOverflow);
    }
    if depth_remaining == 0 {
        return Err(ExpansionHistoryError::IntegrationDidNotConverge);
    }

    let left_result = adaptive_simpson_recursive(
        f,
        a,
        midpoint,
        fa,
        flm,
        fm,
        left,
        absolute_tolerance * 0.5,
        relative_tolerance,
        depth_remaining - 1,
    )?;
    let right_result = adaptive_simpson_recursive(
        f,
        midpoint,
        b,
        fm,
        frm,
        fb,
        right,
        absolute_tolerance * 0.5,
        relative_tolerance,
        depth_remaining - 1,
    )?;
    let total = left_result + right_result;
    if total.is_finite() {
        Ok(total)
    } else {
        Err(ExpansionHistoryError::NumericalOverflow)
    }
}

fn simpson_estimate(a: f64, b: f64, fa: f64, fm: f64, fb: f64) -> f64 {
    (b - a) * (fa + 4.0 * fm + fb) / 6.0
}
