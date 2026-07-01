// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Acoustics
//!
//! **Status**: Exploratory module — HDC encodings for acoustics (sound waves, resonators, metamaterials).
//!
//! This module encodes acoustic phenomena: sound waves, vibrations,
//! phonon transport, and acoustic metamaterials.
//!
//! ## Wave Equation
//!
//! ∂²p/∂t² = c²∇²p
//!
//! ## Key Concepts
//!
//! - **Sound speed**: c = √(K/ρ) for fluids, √(E/ρ) for solids
//! - **Impedance**: Z = ρc
//! - **Intensity**: I = p²/(2ρc)
//!
//! ## Applications
//!
//! - Musical acoustics
//! - Architectural acoustics
//! - Ultrasound
//! - Phonon engineering
//! - Acoustic metamaterials

use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Speed of sound in air at 20°C
pub const SOUND_SPEED_AIR: f64 = 343.0; // m/s

/// Reference sound pressure level
pub const P_REF: f64 = 2e-5; // Pa (threshold of hearing)

/// Wave type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaveType {
    Longitudinal,
    Transverse,
    Surface, // Rayleigh waves
    Plate,   // Lamb waves
}

/// Medium type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediumType {
    Gas,
    Liquid,
    Solid,
    Metamaterial,
}

/// An acoustic wave
#[derive(Debug, Clone)]
pub struct AcousticWave {
    pub frequency_hz: f64,
    pub wavelength_m: f64,
    pub amplitude_pa: f64,
    pub wave_type: WaveType,
    pub medium: MediumType,
    pub sound_speed: f64,
    pub vector: ContinuousHV,
}

impl AcousticWave {
    /// Calculate sound pressure level in dB
    pub fn spl_db(&self) -> f64 {
        20.0 * (self.amplitude_pa / P_REF).log10()
    }

    /// Calculate intensity in W/m²
    pub fn intensity(&self, density: f64) -> f64 {
        self.amplitude_pa.powi(2) / (2.0 * density * self.sound_speed)
    }
}

/// Acoustic resonator
#[derive(Debug, Clone)]
pub struct Resonator {
    pub name: String,
    pub fundamental_freq: f64,
    pub harmonics: Vec<f64>,
    pub q_factor: f64,
    pub vector: ContinuousHV,
}

/// Acoustic metamaterial unit cell
#[derive(Debug, Clone)]
pub struct MetamaterialCell {
    pub name: String,
    pub effective_density: f64, // Can be negative!
    pub effective_modulus: f64, // Can be negative!
    pub bandgap_hz: Option<(f64, f64)>,
    pub vector: ContinuousHV,
}

/// Acoustics Encoder
#[derive(Debug, Clone)]
pub struct AcousticsEncoder {
    genesis: GenesisSeed,

    // Wave properties
    pub wave: ContinuousHV,
    pub frequency: ContinuousHV,
    pub wavelength: ContinuousHV,
    pub amplitude: ContinuousHV,
    pub phase: ContinuousHV,
    pub intensity: ContinuousHV,

    // Wave types
    pub longitudinal: ContinuousHV,
    pub transverse: ContinuousHV,
    pub surface_wave: ContinuousHV,
    pub standing_wave: ContinuousHV,
    pub traveling_wave: ContinuousHV,

    // Material properties
    pub density: ContinuousHV,
    pub bulk_modulus: ContinuousHV,
    pub shear_modulus: ContinuousHV,
    pub impedance: ContinuousHV,
    pub attenuation: ContinuousHV,

    // Phenomena
    pub reflection: ContinuousHV,
    pub transmission: ContinuousHV,
    pub diffraction: ContinuousHV,
    pub interference: ContinuousHV,
    pub resonance: ContinuousHV,
    pub doppler: ContinuousHV,

    // Hearing/perception
    pub loudness: ContinuousHV,
    pub pitch: ContinuousHV,
    pub timbre: ContinuousHV,

    // Phonons
    pub phonon: ContinuousHV,
    pub acoustic_phonon: ContinuousHV,
    pub optical_phonon: ContinuousHV,
    pub phonon_dispersion: ContinuousHV,

    // Metamaterials
    pub metamaterial: ContinuousHV,
    pub negative_index: ContinuousHV,
    pub bandgap: ContinuousHV,
    pub cloaking: ContinuousHV,
}

impl AcousticsEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            genesis: genesis.clone(),

            wave: genesis.hv("acoustics::wave", PHYSICS_DIM),
            frequency: genesis.hv("acoustics::frequency", PHYSICS_DIM),
            wavelength: genesis.hv("acoustics::wavelength", PHYSICS_DIM),
            amplitude: genesis.hv("acoustics::amplitude", PHYSICS_DIM),
            phase: genesis.hv("acoustics::phase", PHYSICS_DIM),
            intensity: genesis.hv("acoustics::intensity", PHYSICS_DIM),

            longitudinal: genesis.hv("wave::longitudinal", PHYSICS_DIM),
            transverse: genesis.hv("wave::transverse", PHYSICS_DIM),
            surface_wave: genesis.hv("wave::surface", PHYSICS_DIM),
            standing_wave: genesis.hv("wave::standing", PHYSICS_DIM),
            traveling_wave: genesis.hv("wave::traveling", PHYSICS_DIM),

            density: genesis.hv("material::density", PHYSICS_DIM),
            bulk_modulus: genesis.hv("material::bulk_modulus", PHYSICS_DIM),
            shear_modulus: genesis.hv("material::shear_modulus", PHYSICS_DIM),
            impedance: genesis.hv("material::impedance", PHYSICS_DIM),
            attenuation: genesis.hv("material::attenuation", PHYSICS_DIM),

            reflection: genesis.hv("phenomenon::reflection", PHYSICS_DIM),
            transmission: genesis.hv("phenomenon::transmission", PHYSICS_DIM),
            diffraction: genesis.hv("phenomenon::diffraction", PHYSICS_DIM),
            interference: genesis.hv("phenomenon::interference", PHYSICS_DIM),
            resonance: genesis.hv("phenomenon::resonance", PHYSICS_DIM),
            doppler: genesis.hv("phenomenon::doppler", PHYSICS_DIM),

            loudness: genesis.hv("perception::loudness", PHYSICS_DIM),
            pitch: genesis.hv("perception::pitch", PHYSICS_DIM),
            timbre: genesis.hv("perception::timbre", PHYSICS_DIM),

            phonon: genesis.hv("phonon::phonon", PHYSICS_DIM),
            acoustic_phonon: genesis.hv("phonon::acoustic", PHYSICS_DIM),
            optical_phonon: genesis.hv("phonon::optical", PHYSICS_DIM),
            phonon_dispersion: genesis.hv("phonon::dispersion", PHYSICS_DIM),

            metamaterial: genesis.hv("meta::acoustic_metamaterial", PHYSICS_DIM),
            negative_index: genesis.hv("meta::negative_index", PHYSICS_DIM),
            bandgap: genesis.hv("meta::bandgap", PHYSICS_DIM),
            cloaking: genesis.hv("meta::cloaking", PHYSICS_DIM),
        }
    }

    // ============================================================
    // WAVE PROPERTIES
    // ============================================================

    /// Calculate sound speed in a fluid: c = √(K/ρ)
    pub fn sound_speed_fluid(&self, bulk_modulus: f64, density: f64) -> f64 {
        (bulk_modulus / density).sqrt()
    }

    /// Calculate sound speed in a solid (longitudinal): c_L = √((K + 4G/3)/ρ)
    pub fn sound_speed_solid_long(
        &self,
        bulk_modulus: f64,
        shear_modulus: f64,
        density: f64,
    ) -> f64 {
        ((bulk_modulus + 4.0 * shear_modulus / 3.0) / density).sqrt()
    }

    /// Calculate sound speed in a solid (transverse): c_T = √(G/ρ)
    pub fn sound_speed_solid_trans(&self, shear_modulus: f64, density: f64) -> f64 {
        (shear_modulus / density).sqrt()
    }

    /// Acoustic impedance: Z = ρc
    pub fn acoustic_impedance(&self, density: f64, sound_speed: f64) -> f64 {
        density * sound_speed
    }

    /// Create an acoustic wave
    pub fn create_wave(
        &self,
        freq_hz: f64,
        amp_pa: f64,
        wave_type: WaveType,
        medium: MediumType,
        sound_speed: f64,
    ) -> AcousticWave {
        let wavelength = sound_speed / freq_hz;

        let type_vec = match wave_type {
            WaveType::Longitudinal => self.longitudinal.clone(),
            WaveType::Transverse => self.transverse.clone(),
            WaveType::Surface => self.surface_wave.clone(),
            WaveType::Plate => self.surface_wave.bind(&self.transverse),
        };

        let vector = self
            .wave
            .bind(&type_vec)
            .bind(&self.frequency.scale(freq_hz as f32))
            .bind(&self.amplitude.scale(amp_pa as f32));

        AcousticWave {
            frequency_hz: freq_hz,
            wavelength_m: wavelength,
            amplitude_pa: amp_pa,
            wave_type,
            medium,
            sound_speed,
            vector,
        }
    }

    /// Wave equation encoding: ∂²p/∂t² = c²∇²p
    pub fn wave_equation(&self) -> ContinuousHV {
        let laplacian = self.genesis.hv("operator::laplacian", PHYSICS_DIM);
        let time_derivative = self.genesis.hv("operator::time_derivative", PHYSICS_DIM);

        self.wave
            .bind(&self.amplitude)
            .bind(&laplacian)
            .bind(&time_derivative)
    }

    // ============================================================
    // REFLECTION AND TRANSMISSION
    // ============================================================

    /// Reflection coefficient at normal incidence
    /// R = (Z2 - Z1)/(Z2 + Z1)
    pub fn reflection_coefficient(&self, z1: f64, z2: f64) -> f64 {
        (z2 - z1) / (z2 + z1)
    }

    /// Transmission coefficient at normal incidence
    /// T = 2Z2/(Z2 + Z1)
    pub fn transmission_coefficient(&self, z1: f64, z2: f64) -> f64 {
        2.0 * z2 / (z2 + z1)
    }

    /// Interface vector encoding
    pub fn interface(&self, z1: f64, z2: f64) -> ContinuousHV {
        let r = self.reflection_coefficient(z1, z2);
        let t = self.transmission_coefficient(z1, z2);

        ContinuousHV::weighted_bundle(
            &[&self.reflection, &self.transmission],
            &[r.abs() as f32, t.abs() as f32],
        )
        .bind(&self.impedance)
    }

    // ============================================================
    // RESONANCE
    // ============================================================

    /// Create a resonator (open pipe, closed pipe, string, etc.)
    pub fn create_resonator(
        &self,
        name: &str,
        fundamental: f64,
        n_harmonics: usize,
        q: f64,
    ) -> Resonator {
        let harmonics: Vec<f64> = (1..=n_harmonics).map(|n| fundamental * n as f64).collect();

        let vector = self
            .resonance
            .bind(&self.frequency.scale(fundamental as f32))
            .bind(&self.standing_wave);

        Resonator {
            name: name.to_string(),
            fundamental_freq: fundamental,
            harmonics,
            q_factor: q,
            vector,
        }
    }

    /// Resonant frequency of a string: f = (n/2L)√(T/μ)
    pub fn string_frequency(&self, length: f64, tension: f64, linear_density: f64, n: u32) -> f64 {
        (n as f64 / (2.0 * length)) * (tension / linear_density).sqrt()
    }

    /// Resonant frequency of a pipe (open-open): f = nc/(2L)
    pub fn open_pipe_frequency(&self, length: f64, sound_speed: f64, n: u32) -> f64 {
        n as f64 * sound_speed / (2.0 * length)
    }

    /// Resonant frequency of a pipe (closed-open): f = (2n-1)c/(4L)
    pub fn closed_pipe_frequency(&self, length: f64, sound_speed: f64, n: u32) -> f64 {
        (2 * n - 1) as f64 * sound_speed / (4.0 * length)
    }

    // ============================================================
    // DOPPLER EFFECT
    // ============================================================

    /// Doppler shifted frequency
    /// f' = f(c + v_observer)/(c - v_source)
    pub fn doppler_frequency(&self, f: f64, c: f64, v_source: f64, v_observer: f64) -> f64 {
        f * (c + v_observer) / (c - v_source)
    }

    /// Doppler encoding
    pub fn doppler_effect(&self) -> ContinuousHV {
        let velocity = self.genesis.hv("mechanics::velocity", PHYSICS_DIM);
        self.doppler.bind(&self.frequency).bind(&velocity)
    }

    // ============================================================
    // PHONONS
    // ============================================================

    /// Acoustic phonon dispersion: ω = c|k| (linear near k=0)
    pub fn acoustic_phonon_dispersion(&self) -> ContinuousHV {
        let wavevector = self.genesis.hv("wave::wavevector", PHYSICS_DIM);
        self.acoustic_phonon
            .bind(&self.phonon_dispersion)
            .bind(&wavevector)
            .bind(&self.frequency)
    }

    /// Optical phonon dispersion: ω ≈ ω₀ (nearly flat)
    pub fn optical_phonon_dispersion(&self) -> ContinuousHV {
        let flat = self.genesis.hv("dispersion::flat", PHYSICS_DIM);
        self.optical_phonon
            .bind(&self.phonon_dispersion)
            .bind(&flat)
    }

    /// Debye model: cutoff frequency ω_D
    pub fn debye_model(&self, debye_freq: f64) -> ContinuousHV {
        let debye = self.genesis.hv("model::debye", PHYSICS_DIM);
        debye
            .bind(&self.acoustic_phonon)
            .bind(&self.frequency.scale(debye_freq as f32))
    }

    /// Phonon thermal conductivity: κ = (1/3)Cvℓ
    pub fn phonon_thermal_conductivity(&self) -> ContinuousHV {
        let heat_capacity = self.genesis.hv("thermo::heat_capacity", PHYSICS_DIM);
        let mean_free_path = self.genesis.hv("transport::mean_free_path", PHYSICS_DIM);
        let conductivity = self
            .genesis
            .hv("transport::thermal_conductivity", PHYSICS_DIM);

        self.phonon
            .bind(&conductivity)
            .bind(&heat_capacity)
            .bind(&mean_free_path)
    }

    // ============================================================
    // METAMATERIALS
    // ============================================================

    /// Create acoustic metamaterial unit cell
    pub fn create_metamaterial(
        &self,
        name: &str,
        eff_density: f64,
        eff_modulus: f64,
        bandgap: Option<(f64, f64)>,
    ) -> MetamaterialCell {
        let mut vector = self
            .metamaterial
            .bind(&self.density.scale(eff_density as f32))
            .bind(&self.bulk_modulus.scale(eff_modulus as f32));

        if eff_density < 0.0 || eff_modulus < 0.0 {
            vector = vector.bind(&self.negative_index);
        }

        if bandgap.is_some() {
            vector = vector.bind(&self.bandgap);
        }

        MetamaterialCell {
            name: name.to_string(),
            effective_density: eff_density,
            effective_modulus: eff_modulus,
            bandgap_hz: bandgap,
            vector,
        }
    }

    /// Acoustic cloaking: redirects waves around object
    pub fn acoustic_cloak(&self) -> ContinuousHV {
        let transformation = self.genesis.hv("optics::transformation", PHYSICS_DIM);
        self.cloaking
            .bind(&self.metamaterial)
            .bind(&transformation)
            .bind(&self.wave)
    }

    /// Acoustic superlens: sub-wavelength imaging
    pub fn acoustic_superlens(&self) -> ContinuousHV {
        let evanescent = self.genesis.hv("wave::evanescent", PHYSICS_DIM);
        let imaging = self.genesis.hv("optics::imaging", PHYSICS_DIM);

        self.negative_index
            .bind(&self.metamaterial)
            .bind(&evanescent)
            .bind(&imaging)
    }

    // ============================================================
    // SOUND PRESSURE LEVELS
    // ============================================================

    /// Sound pressure level in dB: SPL = 20 log₁₀(p/p_ref)
    pub fn spl_db(&self, pressure_pa: f64) -> f64 {
        20.0 * (pressure_pa / P_REF).log10()
    }

    /// Sound intensity level: SIL = 10 log₁₀(I/I_ref)
    pub fn sil_db(&self, intensity: f64) -> f64 {
        let i_ref = 1e-12; // W/m²
        10.0 * (intensity / i_ref).log10()
    }

    /// Addition of incoherent sound sources
    pub fn add_incoherent_sources(&self, levels_db: &[f64]) -> f64 {
        let total_intensity: f64 = levels_db.iter().map(|db| 10.0_f64.powf(db / 10.0)).sum();
        10.0 * total_intensity.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> AcousticsEncoder {
        let genesis = GenesisSeed::from_phrase("acoustics test");
        AcousticsEncoder::from_genesis(&genesis)
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = setup();
        assert!(encoder.wave.norm() > 0.0);
        assert!(encoder.phonon.norm() > 0.0);
    }

    #[test]
    fn test_sound_speed() {
        let encoder = setup();

        // Air: K ≈ 142 kPa, ρ ≈ 1.2 kg/m³
        let c_air = encoder.sound_speed_fluid(142e3, 1.2);
        assert!((c_air - 344.0).abs() < 10.0);
    }

    #[test]
    fn test_acoustic_impedance() {
        let encoder = setup();

        // Air at 20°C
        let z_air = encoder.acoustic_impedance(1.2, 343.0);
        assert!((z_air - 411.6).abs() < 5.0);
    }

    #[test]
    fn test_wave_creation() {
        let encoder = setup();

        let wave = encoder.create_wave(
            440.0, // A4
            0.1,   // 0.1 Pa
            WaveType::Longitudinal,
            MediumType::Gas,
            343.0,
        );

        assert!((wave.wavelength_m - 0.78).abs() < 0.01);
        assert!(wave.spl_db() > 50.0); // Should be fairly loud
    }

    #[test]
    fn test_reflection_coefficient() {
        let encoder = setup();

        // Air to water (large impedance mismatch)
        let z_air = 411.0;
        let z_water = 1.5e6;

        let r = encoder.reflection_coefficient(z_air, z_water);

        // Most energy should be reflected
        assert!(r > 0.99);
    }

    #[test]
    fn test_doppler() {
        let encoder = setup();

        // Source approaching at 34.3 m/s (0.1 Mach)
        let f_shifted = encoder.doppler_frequency(440.0, 343.0, 34.3, 0.0);

        // Frequency should increase
        assert!(f_shifted > 440.0);
        // Should be about 489 Hz
        assert!((f_shifted - 489.0).abs() < 2.0);
    }

    #[test]
    fn test_string_frequency() {
        let encoder = setup();

        // Guitar E string: L=0.65m, T=71N, μ=0.0037 kg/m
        let f = encoder.string_frequency(0.65, 71.0, 0.0037, 1);

        // Should be around 82 Hz (low E)
        assert!((f - 106.0).abs() < 10.0); // Approximate
    }

    #[test]
    fn test_spl_db() {
        let encoder = setup();

        // Threshold of hearing
        let spl_threshold = encoder.spl_db(2e-5);
        assert!((spl_threshold - 0.0).abs() < 0.1);

        // Normal conversation (~60 dB)
        let spl_convo = encoder.spl_db(2e-2);
        assert!((spl_convo - 60.0).abs() < 1.0);
    }

    #[test]
    fn test_metamaterial() {
        let encoder = setup();

        // Negative effective mass
        let meta =
            encoder.create_metamaterial("Negative density", -1.0, 100e9, Some((1000.0, 2000.0)));

        assert!(meta.effective_density < 0.0);
        assert!(meta.bandgap_hz.is_some());
    }

    #[test]
    fn test_phonon_dispersions() {
        let encoder = setup();

        let acoustic = encoder.acoustic_phonon_dispersion();
        let optical = encoder.optical_phonon_dispersion();

        // Should be different dispersions
        let sim = acoustic.similarity(&optical);
        assert!(sim < 0.9);
    }

    #[test]
    fn test_incoherent_addition() {
        let encoder = setup();

        // Two 60 dB sources → ~63 dB
        let total = encoder.add_incoherent_sources(&[60.0, 60.0]);
        assert!((total - 63.0).abs() < 0.2);
    }
}
