// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Electromagnetism
//!
//! **Status**: Exploratory module — HDC encodings for electromagnetism (Maxwell equations, photonics, optical phenomena).
//!
//! This module encodes electromagnetic concepts from Maxwell's equations to photonics.
//!
//! ## Maxwell's Equations
//!
//! - **Gauss's Law**: ∇·E = ρ/ε₀
//! - **Gauss's Law (B)**: ∇·B = 0
//! - **Faraday's Law**: ∇×E = -∂B/∂t
//! - **Ampère-Maxwell**: ∇×B = μ₀J + μ₀ε₀∂E/∂t
//!
//! ## Electromagnetic Waves
//!
//! - Light as transverse EM waves
//! - Polarization (linear, circular, elliptical)
//! - Dispersion and refraction
//!
//! ## Optics
//!
//! - Geometric optics (rays, lenses, mirrors)
//! - Wave optics (interference, diffraction)
//! - Photonics (lasers, fiber optics, metamaterials)

use super::constants::{C, H};
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Polarization state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Polarization {
    Linear,
    CircularRight,
    CircularLeft,
    Elliptical,
    Unpolarized,
}

impl Polarization {
    fn domain(&self) -> &'static str {
        match self {
            Polarization::Linear => "em::linear_pol",
            Polarization::CircularRight => "em::circular_r",
            Polarization::CircularLeft => "em::circular_l",
            Polarization::Elliptical => "em::elliptical",
            Polarization::Unpolarized => "em::unpolarized",
        }
    }
}

/// EM spectrum region
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpectrumRegion {
    Radio,
    Microwave,
    Infrared,
    Visible,
    Ultraviolet,
    XRay,
    GammaRay,
}

impl SpectrumRegion {
    fn domain(&self) -> &'static str {
        match self {
            SpectrumRegion::Radio => "em::radio",
            SpectrumRegion::Microwave => "em::microwave",
            SpectrumRegion::Infrared => "em::infrared",
            SpectrumRegion::Visible => "em::visible",
            SpectrumRegion::Ultraviolet => "em::uv",
            SpectrumRegion::XRay => "em::xray",
            SpectrumRegion::GammaRay => "em::gamma",
        }
    }

    /// Approximate frequency range (Hz)
    pub fn frequency_range(&self) -> (f64, f64) {
        match self {
            SpectrumRegion::Radio => (3e3, 3e9),
            SpectrumRegion::Microwave => (3e9, 3e11),
            SpectrumRegion::Infrared => (3e11, 4e14),
            SpectrumRegion::Visible => (4e14, 8e14),
            SpectrumRegion::Ultraviolet => (8e14, 3e16),
            SpectrumRegion::XRay => (3e16, 3e19),
            SpectrumRegion::GammaRay => (3e19, f64::INFINITY),
        }
    }
}

/// Optical phenomenon
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpticalPhenomenon {
    Reflection,
    Refraction,
    Diffraction,
    Interference,
    Dispersion,
    Scattering,
    Absorption,
    Emission,
}

impl OpticalPhenomenon {
    fn domain(&self) -> &'static str {
        match self {
            OpticalPhenomenon::Reflection => "optics::reflection",
            OpticalPhenomenon::Refraction => "optics::refraction",
            OpticalPhenomenon::Diffraction => "optics::diffraction",
            OpticalPhenomenon::Interference => "optics::interference",
            OpticalPhenomenon::Dispersion => "optics::dispersion",
            OpticalPhenomenon::Scattering => "optics::scattering",
            OpticalPhenomenon::Absorption => "optics::absorption",
            OpticalPhenomenon::Emission => "optics::emission",
        }
    }
}

/// An electromagnetic wave
#[derive(Debug, Clone)]
pub struct EMWave {
    /// Frequency in Hz
    pub frequency_hz: f64,
    /// Wavelength in meters
    pub wavelength_m: f64,
    /// Amplitude (arbitrary units)
    pub amplitude: f64,
    /// Polarization state
    pub polarization: Polarization,
    /// Spectrum region
    pub region: SpectrumRegion,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// A photon
#[derive(Debug, Clone)]
pub struct Photon {
    /// Energy in eV
    pub energy_ev: f64,
    /// Frequency in Hz
    pub frequency_hz: f64,
    /// Polarization
    pub polarization: Polarization,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// An optical element
#[derive(Debug, Clone)]
pub struct OpticalElement {
    pub name: String,
    pub element_type: OpticalElementType,
    /// Refractive index (if applicable)
    pub refractive_index: Option<f64>,
    /// Focal length in meters (if applicable)
    pub focal_length_m: Option<f64>,
    pub vector: ContinuousHV,
}

/// Type of optical element
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpticalElementType {
    Mirror,
    Lens,
    Prism,
    Grating,
    Polarizer,
    WavePlate,
    BeamSplitter,
    Fiber,
}

/// Electromagnetism encoder
#[derive(Debug, Clone)]
pub struct EMEncoder {
    // Fields
    pub electric_field: ContinuousHV,
    pub magnetic_field: ContinuousHV,
    pub em_field: ContinuousHV,

    // Maxwell's equations components
    pub divergence: ContinuousHV,
    pub curl: ContinuousHV,
    pub charge_density: ContinuousHV,
    pub current_density: ContinuousHV,

    // Wave properties
    pub wave: ContinuousHV,
    pub transverse: ContinuousHV,
    pub phase: ContinuousHV,
    pub amplitude: ContinuousHV,

    // Polarization
    pub linear_pol: ContinuousHV,
    pub circular_pol: ContinuousHV,

    // Spectrum regions
    pub radio: ContinuousHV,
    pub visible: ContinuousHV,
    pub xray: ContinuousHV,

    // Optical phenomena
    pub reflection: ContinuousHV,
    pub refraction: ContinuousHV,
    pub diffraction: ContinuousHV,
    pub interference: ContinuousHV,

    // Photon
    pub photon: ContinuousHV,
    pub quantized: ContinuousHV,

    // Materials response
    pub permittivity: ContinuousHV,
    pub permeability: ContinuousHV,
    pub conductivity: ContinuousHV,
}

impl EMEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            electric_field: genesis.hv("em::electric_field", PHYSICS_DIM),
            magnetic_field: genesis.hv("em::magnetic_field", PHYSICS_DIM),
            em_field: genesis.hv("em::em_field", PHYSICS_DIM),

            divergence: genesis.hv("em::divergence", PHYSICS_DIM),
            curl: genesis.hv("em::curl", PHYSICS_DIM),
            charge_density: genesis.hv("em::charge_density", PHYSICS_DIM),
            current_density: genesis.hv("em::current_density", PHYSICS_DIM),

            wave: genesis.hv("em::wave", PHYSICS_DIM),
            transverse: genesis.hv("em::transverse", PHYSICS_DIM),
            phase: genesis.hv("em::phase", PHYSICS_DIM),
            amplitude: genesis.hv("em::amplitude", PHYSICS_DIM),

            linear_pol: genesis.hv(Polarization::Linear.domain(), PHYSICS_DIM),
            circular_pol: genesis.hv(Polarization::CircularRight.domain(), PHYSICS_DIM),

            radio: genesis.hv(SpectrumRegion::Radio.domain(), PHYSICS_DIM),
            visible: genesis.hv(SpectrumRegion::Visible.domain(), PHYSICS_DIM),
            xray: genesis.hv(SpectrumRegion::XRay.domain(), PHYSICS_DIM),

            reflection: genesis.hv(OpticalPhenomenon::Reflection.domain(), PHYSICS_DIM),
            refraction: genesis.hv(OpticalPhenomenon::Refraction.domain(), PHYSICS_DIM),
            diffraction: genesis.hv(OpticalPhenomenon::Diffraction.domain(), PHYSICS_DIM),
            interference: genesis.hv(OpticalPhenomenon::Interference.domain(), PHYSICS_DIM),

            photon: genesis.hv("em::photon", PHYSICS_DIM),
            quantized: genesis.hv("em::quantized", PHYSICS_DIM),

            permittivity: genesis.hv("em::permittivity", PHYSICS_DIM),
            permeability: genesis.hv("em::permeability", PHYSICS_DIM),
            conductivity: genesis.hv("em::conductivity", PHYSICS_DIM),
        }
    }

    /// Create an EM wave
    pub fn create_wave(
        &self,
        frequency_hz: f64,
        polarization: Polarization,
        genesis: &GenesisSeed,
    ) -> EMWave {
        let wavelength_m = C / frequency_hz;
        let region = self.classify_region(frequency_hz);

        let region_vec = genesis.hv(region.domain(), PHYSICS_DIM);
        let pol_vec = genesis.hv(polarization.domain(), PHYSICS_DIM);

        let vector = self
            .wave
            .bind(&self.em_field)
            .bind(&region_vec)
            .bind(&pol_vec)
            .scale((frequency_hz.log10() - 10.0) as f32);

        EMWave {
            frequency_hz,
            wavelength_m,
            amplitude: 1.0,
            polarization,
            region,
            vector,
        }
    }

    fn classify_region(&self, frequency_hz: f64) -> SpectrumRegion {
        if frequency_hz < 3e9 {
            SpectrumRegion::Radio
        } else if frequency_hz < 3e11 {
            SpectrumRegion::Microwave
        } else if frequency_hz < 4e14 {
            SpectrumRegion::Infrared
        } else if frequency_hz < 8e14 {
            SpectrumRegion::Visible
        } else if frequency_hz < 3e16 {
            SpectrumRegion::Ultraviolet
        } else if frequency_hz < 3e19 {
            SpectrumRegion::XRay
        } else {
            SpectrumRegion::GammaRay
        }
    }

    /// Create visible light
    pub fn visible_light(&self, wavelength_nm: f64, genesis: &GenesisSeed) -> EMWave {
        let frequency_hz = C / (wavelength_nm * 1e-9);
        self.create_wave(frequency_hz, Polarization::Unpolarized, genesis)
    }

    /// Create a photon
    pub fn create_photon(
        &self,
        energy_ev: f64,
        polarization: Polarization,
        genesis: &GenesisSeed,
    ) -> Photon {
        let frequency_hz = energy_ev * 1.602e-19 / H;
        let pol_vec = genesis.hv(polarization.domain(), PHYSICS_DIM);

        let vector = self
            .photon
            .bind(&self.quantized)
            .bind(&pol_vec)
            .scale(energy_ev.ln() as f32);

        Photon {
            energy_ev,
            frequency_hz,
            polarization,
            vector,
        }
    }

    /// Gauss's law for electric field: ∇·E = ρ/ε₀
    pub fn gauss_law_e(&self) -> ContinuousHV {
        self.divergence
            .bind(&self.electric_field)
            .bind(&self.charge_density)
    }

    /// Gauss's law for magnetic field: ∇·B = 0
    pub fn gauss_law_b(&self) -> ContinuousHV {
        self.divergence.bind(&self.magnetic_field)
    }

    /// Faraday's law: ∇×E = -∂B/∂t
    pub fn faraday_law(&self) -> ContinuousHV {
        self.curl
            .bind(&self.electric_field)
            .bind(&self.magnetic_field)
    }

    /// Ampère-Maxwell law: ∇×B = μ₀J + μ₀ε₀∂E/∂t
    pub fn ampere_maxwell_law(&self) -> ContinuousHV {
        self.curl
            .bind(&self.magnetic_field)
            .bind(&ContinuousHV::bundle(&[
                &self.current_density,
                &self.electric_field,
            ]))
    }

    /// Full Maxwell equations (combined)
    pub fn maxwell_equations(&self) -> ContinuousHV {
        ContinuousHV::bundle(&[
            &self.gauss_law_e(),
            &self.gauss_law_b(),
            &self.faraday_law(),
            &self.ampere_maxwell_law(),
        ])
    }

    /// Snell's law: n₁ sin θ₁ = n₂ sin θ₂
    pub fn snells_law(&self, n1: f64, n2: f64, theta1_rad: f64) -> Option<f64> {
        let sin_theta2 = n1 * theta1_rad.sin() / n2;
        if sin_theta2.abs() <= 1.0 {
            Some(sin_theta2.asin())
        } else {
            None // Total internal reflection
        }
    }

    /// Calculate refractive index from permittivity and permeability
    pub fn refractive_index(&self, epsilon_r: f64, mu_r: f64) -> f64 {
        (epsilon_r * mu_r).sqrt()
    }

    /// Photon energy from frequency: E = hf
    pub fn photon_energy_ev(&self, frequency_hz: f64) -> f64 {
        H * frequency_hz / 1.602e-19
    }

    /// Create a lens
    pub fn lens(
        &self,
        focal_length_m: f64,
        refractive_index: f64,
        genesis: &GenesisSeed,
    ) -> OpticalElement {
        let vector = genesis
            .hv("optics::lens", PHYSICS_DIM)
            .bind(&self.refraction)
            .scale((1.0 / focal_length_m).ln() as f32);

        OpticalElement {
            name: "Lens".to_string(),
            element_type: OpticalElementType::Lens,
            refractive_index: Some(refractive_index),
            focal_length_m: Some(focal_length_m),
            vector,
        }
    }

    /// Create a diffraction grating
    pub fn diffraction_grating(&self, lines_per_mm: f64, genesis: &GenesisSeed) -> OpticalElement {
        let vector = genesis
            .hv("optics::grating", PHYSICS_DIM)
            .bind(&self.diffraction)
            .scale(lines_per_mm.ln() as f32);

        OpticalElement {
            name: "Diffraction Grating".to_string(),
            element_type: OpticalElementType::Grating,
            refractive_index: None,
            focal_length_m: None,
            vector,
        }
    }

    /// Two-slit interference pattern (Young's experiment)
    pub fn double_slit_interference(
        &self,
        wavelength_m: f64,
        slit_separation_m: f64,
        screen_distance_m: f64,
    ) -> ContinuousHV {
        let fringe_spacing = wavelength_m * screen_distance_m / slit_separation_m;
        self.interference
            .bind(&self.wave)
            .scale(fringe_spacing.log10() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (EMEncoder, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("electromagnetism test");
        let encoder = EMEncoder::from_genesis(&genesis);
        (encoder, genesis)
    }

    #[test]
    fn test_em_encoder_creation() {
        let (encoder, _) = setup();
        assert!(encoder.electric_field.norm() > 0.0);
        assert!(encoder.photon.norm() > 0.0);
    }

    #[test]
    fn test_visible_light_creation() {
        let (encoder, genesis) = setup();

        // Green light at 550 nm
        let green = encoder.visible_light(550.0, &genesis);

        assert_eq!(green.region, SpectrumRegion::Visible);
        assert!(green.vector.norm() > 0.0);
    }

    #[test]
    fn test_spectrum_classification() {
        let (encoder, genesis) = setup();

        let radio = encoder.create_wave(1e6, Polarization::Linear, &genesis);
        let xray = encoder.create_wave(1e18, Polarization::Unpolarized, &genesis);

        assert_eq!(radio.region, SpectrumRegion::Radio);
        assert_eq!(xray.region, SpectrumRegion::XRay);
    }

    #[test]
    fn test_photon_energy() {
        let (encoder, _) = setup();

        // Visible light photon: ~2 eV
        let energy = encoder.photon_energy_ev(5e14);
        assert!(energy > 1.0 && energy < 4.0);
    }

    #[test]
    fn test_maxwell_equations() {
        let (encoder, _) = setup();
        let maxwell = encoder.maxwell_equations();

        assert!(maxwell.norm() > 0.0);
    }

    #[test]
    fn test_snells_law() {
        let (encoder, _) = setup();

        // Air (n=1) to glass (n=1.5) at 45°
        let theta2 = encoder.snells_law(1.0, 1.5, std::f64::consts::FRAC_PI_4);
        assert!(theta2.is_some());

        let angle = theta2.unwrap();
        assert!(angle < std::f64::consts::FRAC_PI_4); // Refracted towards normal

        // Total internal reflection: glass to air at steep angle
        let tir = encoder.snells_law(1.5, 1.0, std::f64::consts::FRAC_PI_3);
        assert!(tir.is_none()); // TIR occurs
    }

    #[test]
    fn test_refractive_index() {
        let (encoder, _) = setup();

        // Vacuum: n = 1
        let n_vacuum = encoder.refractive_index(1.0, 1.0);
        assert!((n_vacuum - 1.0).abs() < 0.001);

        // Glass: n ~ 1.5 (ε_r ~ 2.25)
        let n_glass = encoder.refractive_index(2.25, 1.0);
        assert!((n_glass - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_optical_elements() {
        let (encoder, genesis) = setup();

        let lens = encoder.lens(0.1, 1.5, &genesis);
        assert_eq!(lens.element_type, OpticalElementType::Lens);
        assert!(lens.vector.norm() > 0.0);

        let grating = encoder.diffraction_grating(1000.0, &genesis);
        assert_eq!(grating.element_type, OpticalElementType::Grating);
    }

    #[test]
    fn test_polarization_types() {
        let (encoder, genesis) = setup();

        let linear = encoder.create_wave(5e14, Polarization::Linear, &genesis);
        let circular = encoder.create_wave(5e14, Polarization::CircularRight, &genesis);

        // Different polarizations should have different vectors
        let sim = linear.vector.similarity(&circular.vector);
        assert!(sim < 0.95, "Different polarizations should differ");
    }

    #[test]
    fn test_interference() {
        let (encoder, _) = setup();

        let interference = encoder.double_slit_interference(
            550e-9, // 550 nm green light
            0.5e-3, // 0.5 mm slit separation
            1.0,    // 1 m screen distance
        );

        assert!(interference.norm() > 0.0);
    }
}
