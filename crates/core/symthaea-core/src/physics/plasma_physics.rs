// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Plasma Physics
//!
//! **Status**: Exploratory module — HDC encodings for plasma physics (Debye shielding, MHD, fusion confinement).
//! Used by physics_numerical_validation tests.
//!
//! This module encodes plasma physics concepts from Debye shielding to MHD.
//!
//! ## Plasma Parameters
//!
//! - **Debye length**: λ_D = √(ε₀ k_B T / n e²)
//! - **Plasma frequency**: ω_p = √(n e² / ε₀ m_e)
//! - **Cyclotron frequency**: ω_c = eB/m
//!
//! ## Magnetohydrodynamics (MHD)
//!
//! - Ideal MHD: perfectly conducting fluid
//! - Resistive MHD: finite conductivity
//! - Hall MHD: includes Hall term
//!
//! ## Fusion Plasmas
//!
//! - Tokamak confinement
//! - Stellarator confinement
//! - Inertial confinement

use super::constants::{E_CHARGE, EPSILON_0, K_BOLTZMANN, M_ELECTRON};
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Plasma confinement scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfinementScheme {
    Tokamak,
    Stellarator,
    InertialFusion,
    MagneticMirror,
    ZPinch,
}

impl ConfinementScheme {
    fn domain(&self) -> &'static str {
        match self {
            ConfinementScheme::Tokamak => "plasma::tokamak",
            ConfinementScheme::Stellarator => "plasma::stellarator",
            ConfinementScheme::InertialFusion => "plasma::icf",
            ConfinementScheme::MagneticMirror => "plasma::mirror",
            ConfinementScheme::ZPinch => "plasma::z_pinch",
        }
    }
}

/// Plasma instability type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Instability {
    KinkMode,
    SausageMode,
    RayleighTaylor,
    KelvinHelmholtz,
    TwoStreamInstability,
    MirrorInstability,
    Firehose,
}

impl Instability {
    fn domain(&self) -> &'static str {
        match self {
            Instability::KinkMode => "instability::kink",
            Instability::SausageMode => "instability::sausage",
            Instability::RayleighTaylor => "instability::rt",
            Instability::KelvinHelmholtz => "instability::kh",
            Instability::TwoStreamInstability => "instability::two_stream",
            Instability::MirrorInstability => "instability::mirror",
            Instability::Firehose => "instability::firehose",
        }
    }
}

/// Plasma wave type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlasmaWave {
    Langmuir,     // Electron plasma wave
    IonAcoustic,  // Ion sound wave
    Alfven,       // MHD wave along B
    Magnetosonic, // MHD compression wave
    Whistler,     // Electron cyclotron wave
}

impl PlasmaWave {
    fn domain(&self) -> &'static str {
        match self {
            PlasmaWave::Langmuir => "wave::langmuir",
            PlasmaWave::IonAcoustic => "wave::ion_acoustic",
            PlasmaWave::Alfven => "wave::alfven",
            PlasmaWave::Magnetosonic => "wave::magnetosonic",
            PlasmaWave::Whistler => "wave::whistler",
        }
    }
}

/// A plasma
#[derive(Debug, Clone)]
pub struct Plasma {
    pub name: String,
    /// Electron density (m⁻³)
    pub density_m3: f64,
    /// Temperature (eV)
    pub temperature_ev: f64,
    /// Magnetic field (Tesla)
    pub magnetic_field_t: f64,
    /// Debye length (m)
    pub debye_length_m: f64,
    /// Plasma frequency (rad/s)
    pub plasma_frequency_rad_s: f64,
    pub vector: ContinuousHV,
}

/// Fusion reactor
#[derive(Debug, Clone)]
pub struct FusionReactor {
    pub name: String,
    pub scheme: ConfinementScheme,
    /// Ion temperature (keV)
    pub ion_temp_kev: f64,
    /// Plasma density (m⁻³)
    pub density_m3: f64,
    /// Confinement time (s)
    pub confinement_time_s: f64,
    /// Triple product (m⁻³ keV s)
    pub triple_product: f64,
    pub vector: ContinuousHV,
}

/// Plasma physics encoder
#[derive(Debug, Clone)]
pub struct PlasmaEncoder {
    // Plasma concepts
    pub plasma: ContinuousHV,
    pub ionized: ContinuousHV,
    pub quasi_neutral: ContinuousHV,

    // Characteristic lengths
    pub debye_length: ContinuousHV,
    pub gyro_radius: ContinuousHV,
    pub mean_free_path: ContinuousHV,

    // Frequencies
    pub plasma_frequency: ContinuousHV,
    pub cyclotron_frequency: ContinuousHV,
    pub collision_frequency: ContinuousHV,

    // Fields
    pub electric_field: ContinuousHV,
    pub magnetic_field: ContinuousHV,
    pub current: ContinuousHV,

    // MHD concepts
    pub mhd: ContinuousHV,
    pub frozen_in: ContinuousHV,
    pub magnetic_pressure: ContinuousHV,
    pub magnetic_tension: ContinuousHV,
    pub beta: ContinuousHV, // Plasma beta = p_thermal / p_magnetic

    // Confinement
    pub confinement: ContinuousHV,
    pub tokamak: ContinuousHV,
    pub stellarator: ContinuousHV,

    // Waves
    pub alfven_wave: ContinuousHV,
    pub langmuir_wave: ContinuousHV,

    // Instabilities
    pub instability: ContinuousHV,
    pub turbulence: ContinuousHV,

    // Fusion
    pub fusion: ContinuousHV,
    pub ignition: ContinuousHV,
    pub lawson: ContinuousHV,
}

impl PlasmaEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            plasma: genesis.hv("plasma::plasma", PHYSICS_DIM),
            ionized: genesis.hv("plasma::ionized", PHYSICS_DIM),
            quasi_neutral: genesis.hv("plasma::quasi_neutral", PHYSICS_DIM),

            debye_length: genesis.hv("plasma::debye_length", PHYSICS_DIM),
            gyro_radius: genesis.hv("plasma::gyro_radius", PHYSICS_DIM),
            mean_free_path: genesis.hv("plasma::mean_free_path", PHYSICS_DIM),

            plasma_frequency: genesis.hv("plasma::plasma_frequency", PHYSICS_DIM),
            cyclotron_frequency: genesis.hv("plasma::cyclotron_frequency", PHYSICS_DIM),
            collision_frequency: genesis.hv("plasma::collision_frequency", PHYSICS_DIM),

            electric_field: genesis.hv("plasma::electric_field", PHYSICS_DIM),
            magnetic_field: genesis.hv("plasma::magnetic_field", PHYSICS_DIM),
            current: genesis.hv("plasma::current", PHYSICS_DIM),

            mhd: genesis.hv("plasma::mhd", PHYSICS_DIM),
            frozen_in: genesis.hv("plasma::frozen_in", PHYSICS_DIM),
            magnetic_pressure: genesis.hv("plasma::magnetic_pressure", PHYSICS_DIM),
            magnetic_tension: genesis.hv("plasma::magnetic_tension", PHYSICS_DIM),
            beta: genesis.hv("plasma::beta", PHYSICS_DIM),

            confinement: genesis.hv("plasma::confinement", PHYSICS_DIM),
            tokamak: genesis.hv(ConfinementScheme::Tokamak.domain(), PHYSICS_DIM),
            stellarator: genesis.hv(ConfinementScheme::Stellarator.domain(), PHYSICS_DIM),

            alfven_wave: genesis.hv(PlasmaWave::Alfven.domain(), PHYSICS_DIM),
            langmuir_wave: genesis.hv(PlasmaWave::Langmuir.domain(), PHYSICS_DIM),

            instability: genesis.hv("plasma::instability", PHYSICS_DIM),
            turbulence: genesis.hv("plasma::turbulence", PHYSICS_DIM),

            fusion: genesis.hv("plasma::fusion", PHYSICS_DIM),
            ignition: genesis.hv("plasma::ignition", PHYSICS_DIM),
            lawson: genesis.hv("plasma::lawson", PHYSICS_DIM),
        }
    }

    /// Calculate Debye length: λ_D = √(ε₀ k_B T / n e²)
    pub fn debye_length_m(&self, temp_ev: f64, density_m3: f64) -> f64 {
        let temp_k = temp_ev * 11604.0; // eV to Kelvin
        (EPSILON_0 * K_BOLTZMANN * temp_k / (density_m3 * E_CHARGE * E_CHARGE)).sqrt()
    }

    /// Calculate plasma frequency: ω_p = √(n e² / ε₀ m_e)
    pub fn plasma_frequency_rad_s(&self, density_m3: f64) -> f64 {
        (density_m3 * E_CHARGE * E_CHARGE / (EPSILON_0 * M_ELECTRON)).sqrt()
    }

    /// Calculate electron cyclotron frequency: ω_c = eB/m_e
    pub fn cyclotron_frequency_rad_s(&self, magnetic_field_t: f64) -> f64 {
        E_CHARGE * magnetic_field_t / M_ELECTRON
    }

    /// Calculate Alfvén velocity: v_A = B / √(μ₀ ρ)
    pub fn alfven_velocity(&self, magnetic_field_t: f64, mass_density_kg_m3: f64) -> f64 {
        let mu_0 = 4.0 * std::f64::consts::PI * 1e-7;
        magnetic_field_t / (mu_0 * mass_density_kg_m3).sqrt()
    }

    /// Calculate plasma beta: β = 2μ₀ n k_B T / B²
    pub fn plasma_beta(&self, density_m3: f64, temp_ev: f64, magnetic_field_t: f64) -> f64 {
        let mu_0 = 4.0 * std::f64::consts::PI * 1e-7;
        let temp_k = temp_ev * 11604.0;
        2.0 * mu_0 * density_m3 * K_BOLTZMANN * temp_k / (magnetic_field_t * magnetic_field_t)
    }

    /// Create a plasma
    pub fn create_plasma(
        &self,
        name: &str,
        density_m3: f64,
        temp_ev: f64,
        b_field_t: f64,
        _genesis: &GenesisSeed,
    ) -> Plasma {
        let debye = self.debye_length_m(temp_ev, density_m3);
        let omega_p = self.plasma_frequency_rad_s(density_m3);

        let vector = self
            .plasma
            .bind(&self.ionized)
            .bind(&self.quasi_neutral)
            .scale((density_m3.log10() - 15.0) as f32);

        Plasma {
            name: name.to_string(),
            density_m3,
            temperature_ev: temp_ev,
            magnetic_field_t: b_field_t,
            debye_length_m: debye,
            plasma_frequency_rad_s: omega_p,
            vector,
        }
    }

    /// Solar corona plasma
    pub fn solar_corona(&self, genesis: &GenesisSeed) -> Plasma {
        self.create_plasma("Solar Corona", 1e15, 100.0, 0.001, genesis)
    }

    /// Fusion plasma (tokamak core)
    pub fn fusion_core(&self, genesis: &GenesisSeed) -> Plasma {
        self.create_plasma("Fusion Core", 1e20, 10000.0, 5.0, genesis)
    }

    /// Lawson criterion: n τ_E > 1.5 × 10²⁰ m⁻³ s for D-T at 10 keV
    pub fn lawson_criterion_met(&self, density_m3: f64, confinement_time_s: f64) -> bool {
        density_m3 * confinement_time_s > 1.5e20
    }

    /// Triple product for fusion: n T τ
    pub fn triple_product(&self, density_m3: f64, temp_kev: f64, confinement_time_s: f64) -> f64 {
        density_m3 * temp_kev * confinement_time_s
    }

    /// Create a fusion reactor
    pub fn create_reactor(
        &self,
        name: &str,
        scheme: ConfinementScheme,
        temp_kev: f64,
        density_m3: f64,
        tau_s: f64,
        genesis: &GenesisSeed,
    ) -> FusionReactor {
        let triple = self.triple_product(density_m3, temp_kev, tau_s);
        let scheme_vec = genesis.hv(scheme.domain(), PHYSICS_DIM);

        let vector = self
            .fusion
            .bind(&self.confinement)
            .bind(&scheme_vec)
            .scale(triple.log10() as f32);

        FusionReactor {
            name: name.to_string(),
            scheme,
            ion_temp_kev: temp_kev,
            density_m3,
            confinement_time_s: tau_s,
            triple_product: triple,
            vector,
        }
    }

    /// ITER-like tokamak
    pub fn iter(&self, genesis: &GenesisSeed) -> FusionReactor {
        self.create_reactor("ITER", ConfinementScheme::Tokamak, 10.0, 1e20, 3.0, genesis)
    }

    /// Ideal MHD equations
    pub fn ideal_mhd(&self) -> ContinuousHV {
        ContinuousHV::bundle(&[
            &self.mhd,
            &self.frozen_in,
            &self.magnetic_pressure,
            &self.magnetic_tension,
        ])
    }

    /// Alfvén wave in plasma
    pub fn alfven_wave(&self, plasma: &Plasma, _genesis: &GenesisSeed) -> ContinuousHV {
        let mass_density = plasma.density_m3 * 1.67e-27; // Assume protons
        let v_a = self.alfven_velocity(plasma.magnetic_field_t, mass_density);

        self.alfven_wave
            .bind(&plasma.vector)
            .scale(v_a.log10() as f32)
    }

    /// Check for MHD instability (beta > 1 suggests pressure-driven instabilities)
    pub fn is_unstable(&self, beta: f64) -> bool {
        beta > 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (PlasmaEncoder, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("plasma physics test");
        let encoder = PlasmaEncoder::from_genesis(&genesis);
        (encoder, genesis)
    }

    #[test]
    fn test_plasma_encoder_creation() {
        let (encoder, _) = setup();
        assert!(encoder.plasma.norm() > 0.0);
        assert!(encoder.fusion.norm() > 0.0);
    }

    #[test]
    fn test_debye_length() {
        let (encoder, _) = setup();

        // Fusion plasma: T ~ 10 keV, n ~ 10²⁰ m⁻³
        let debye = encoder.debye_length_m(10000.0, 1e20);
        // Should be ~ 10⁻⁵ m
        assert!(debye > 1e-6 && debye < 1e-4);
    }

    #[test]
    fn test_plasma_frequency() {
        let (encoder, _) = setup();

        // n = 10²⁰ m⁻³
        let omega_p = encoder.plasma_frequency_rad_s(1e20);
        // Should be ~ 10¹² rad/s
        assert!(omega_p > 1e11 && omega_p < 1e13);
    }

    #[test]
    fn test_cyclotron_frequency() {
        let (encoder, _) = setup();

        // B = 5 T
        let omega_c = encoder.cyclotron_frequency_rad_s(5.0);
        // Should be ~ 10¹¹ rad/s
        assert!(omega_c > 1e10 && omega_c < 1e12);
    }

    #[test]
    fn test_plasma_beta() {
        let (encoder, _) = setup();

        // High beta plasma (pressure-dominated)
        let beta_high = encoder.plasma_beta(1e20, 10000.0, 1.0);
        // Low beta plasma (magnetic-dominated)
        let beta_low = encoder.plasma_beta(1e18, 10.0, 5.0);

        assert!(beta_high > beta_low);
    }

    #[test]
    fn test_plasma_creation() {
        let (encoder, genesis) = setup();

        let corona = encoder.solar_corona(&genesis);
        let core = encoder.fusion_core(&genesis);

        // Fusion core is hotter and denser
        assert!(core.temperature_ev > corona.temperature_ev);
        assert!(core.density_m3 > corona.density_m3);
    }

    #[test]
    fn test_lawson_criterion() {
        let (encoder, _) = setup();

        // ITER-like parameters should meet Lawson
        assert!(encoder.lawson_criterion_met(1e20, 3.0));

        // Low density shouldn't
        assert!(!encoder.lawson_criterion_met(1e18, 0.1));
    }

    #[test]
    fn test_triple_product() {
        let (encoder, _) = setup();

        let tp = encoder.triple_product(1e20, 10.0, 3.0);
        // Should be ~ 3 × 10²¹ m⁻³ keV s
        assert!(tp > 1e21 && tp < 1e22);
    }

    #[test]
    fn test_fusion_reactor() {
        let (encoder, genesis) = setup();

        let iter = encoder.iter(&genesis);
        assert_eq!(iter.scheme, ConfinementScheme::Tokamak);
        assert!(iter.triple_product > 1e21);
    }

    #[test]
    fn test_ideal_mhd() {
        let (encoder, _) = setup();
        let mhd = encoder.ideal_mhd();

        assert!(mhd.norm() > 0.0);
    }

    #[test]
    fn test_instability() {
        let (encoder, _) = setup();

        assert!(encoder.is_unstable(1.5));
        assert!(!encoder.is_unstable(0.1));
    }
}
