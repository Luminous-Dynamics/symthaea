// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Optics
//!
//! **Status**: Exploratory module — HDC encodings for optics (nonlinear, quantum, fiber optics, holography).
//!
//! This module encodes optical phenomena beyond basic electromagnetism:
//! nonlinear optics, quantum optics, fiber optics, and holography.
//!
//! ## Key Concepts
//!
//! - **Geometric optics**: Rays, lenses, mirrors
//! - **Wave optics**: Diffraction, interference, coherence
//! - **Nonlinear optics**: χ² and χ³ effects, harmonics
//! - **Quantum optics**: Photon statistics, squeezed states
//! - **Fiber optics**: Modes, dispersion, nonlinearity
//!
//! ## Applications
//!
//! - Lasers and amplifiers
//! - Holography
//! - Optical communications
//! - Quantum information with photons

use super::constants::{C, H};
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Optical regime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpticalRegime {
    Geometric, // Ray optics, λ << feature size
    Wave,      // Diffraction important
    Quantum,   // Single photon effects
}

/// Nonlinear process
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NonlinearProcess {
    SecondHarmonic,      // ω + ω → 2ω
    ThirdHarmonic,       // 3ω
    SumFrequency,        // ω₁ + ω₂ → ω₃
    DifferenceFrequency, // ω₁ - ω₂ → ω₃
    FourWaveMixing,      // χ³ process
    Kerr,                // Intensity-dependent refractive index
    Raman,               // Inelastic scattering
    Brillouin,           // Acoustic phonon scattering
}

/// Photon statistics type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhotonStatistics {
    Coherent,  // Poissonian (laser)
    Thermal,   // Super-Poissonian (chaotic)
    Squeezed,  // Sub-Poissonian
    Fock,      // Number state
    Entangled, // Two-photon correlations
}

/// An optical beam
#[derive(Debug, Clone)]
pub struct OpticalBeam {
    pub wavelength_nm: f64,
    pub power_w: f64,
    pub beam_waist_um: f64,
    pub coherence_length_m: f64,
    pub statistics: PhotonStatistics,
    pub vector: ContinuousHV,
}

impl OpticalBeam {
    /// Photon energy in eV
    pub fn photon_energy_ev(&self) -> f64 {
        1239.8 / self.wavelength_nm
    }

    /// Photon flux (photons/s)
    pub fn photon_flux(&self) -> f64 {
        self.power_w / (H * C / (self.wavelength_nm * 1e-9))
    }

    /// Rayleigh range
    pub fn rayleigh_range_mm(&self) -> f64 {
        std::f64::consts::PI * (self.beam_waist_um * 1e-3).powi(2) / (self.wavelength_nm * 1e-6)
    }
}

/// Optical fiber
#[derive(Debug, Clone)]
pub struct OpticalFiber {
    pub name: String,
    pub core_diameter_um: f64,
    pub numerical_aperture: f64,
    pub attenuation_db_km: f64,
    pub dispersion_ps_nm_km: f64,
    pub nonlinear_coeff: f64, // W⁻¹km⁻¹
    pub vector: ContinuousHV,
}

/// Optics Encoder
#[derive(Debug, Clone)]
pub struct OpticsEncoder {
    genesis: GenesisSeed,

    // Fundamental concepts
    pub light: ContinuousHV,
    pub photon: ContinuousHV,
    pub wavelength: ContinuousHV,
    pub frequency: ContinuousHV,
    pub phase: ContinuousHV,
    pub polarization: ContinuousHV,
    pub intensity: ContinuousHV,

    // Wave optics
    pub diffraction: ContinuousHV,
    pub interference: ContinuousHV,
    pub coherence: ContinuousHV,
    pub superposition: ContinuousHV,

    // Geometric optics
    pub ray: ContinuousHV,
    pub refraction: ContinuousHV,
    pub reflection: ContinuousHV,
    pub lens: ContinuousHV,
    pub mirror: ContinuousHV,
    pub focus: ContinuousHV,

    // Nonlinear optics
    pub chi2: ContinuousHV,
    pub chi3: ContinuousHV,
    pub harmonic: ContinuousHV,
    pub parametric: ContinuousHV,
    pub kerr: ContinuousHV,
    pub self_phase_mod: ContinuousHV,

    // Quantum optics
    pub fock_state: ContinuousHV,
    pub coherent_state: ContinuousHV,
    pub squeezed_state: ContinuousHV,
    pub entangled_photons: ContinuousHV,
    pub photon_antibunching: ContinuousHV,

    // Lasers
    pub stimulated_emission: ContinuousHV,
    pub population_inversion: ContinuousHV,
    pub gain_medium: ContinuousHV,
    pub cavity: ContinuousHV,
    pub mode_locking: ContinuousHV,

    // Fiber optics
    pub fiber: ContinuousHV,
    pub guided_mode: ContinuousHV,
    pub dispersion: ContinuousHV,
    pub soliton: ContinuousHV,

    // Holography
    pub hologram: ContinuousHV,
    pub reference_beam: ContinuousHV,
    pub reconstruction: ContinuousHV,
}

impl OpticsEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            genesis: genesis.clone(),

            light: genesis.hv("optics::light", PHYSICS_DIM),
            photon: genesis.hv("optics::photon", PHYSICS_DIM),
            wavelength: genesis.hv("optics::wavelength", PHYSICS_DIM),
            frequency: genesis.hv("optics::frequency", PHYSICS_DIM),
            phase: genesis.hv("optics::phase", PHYSICS_DIM),
            polarization: genesis.hv("optics::polarization", PHYSICS_DIM),
            intensity: genesis.hv("optics::intensity", PHYSICS_DIM),

            diffraction: genesis.hv("wave::diffraction", PHYSICS_DIM),
            interference: genesis.hv("wave::interference", PHYSICS_DIM),
            coherence: genesis.hv("wave::coherence", PHYSICS_DIM),
            superposition: genesis.hv("wave::superposition", PHYSICS_DIM),

            ray: genesis.hv("geometric::ray", PHYSICS_DIM),
            refraction: genesis.hv("geometric::refraction", PHYSICS_DIM),
            reflection: genesis.hv("geometric::reflection", PHYSICS_DIM),
            lens: genesis.hv("geometric::lens", PHYSICS_DIM),
            mirror: genesis.hv("geometric::mirror", PHYSICS_DIM),
            focus: genesis.hv("geometric::focus", PHYSICS_DIM),

            chi2: genesis.hv("nonlinear::chi2", PHYSICS_DIM),
            chi3: genesis.hv("nonlinear::chi3", PHYSICS_DIM),
            harmonic: genesis.hv("nonlinear::harmonic", PHYSICS_DIM),
            parametric: genesis.hv("nonlinear::parametric", PHYSICS_DIM),
            kerr: genesis.hv("nonlinear::kerr", PHYSICS_DIM),
            self_phase_mod: genesis.hv("nonlinear::spm", PHYSICS_DIM),

            fock_state: genesis.hv("quantum_optics::fock", PHYSICS_DIM),
            coherent_state: genesis.hv("quantum_optics::coherent", PHYSICS_DIM),
            squeezed_state: genesis.hv("quantum_optics::squeezed", PHYSICS_DIM),
            entangled_photons: genesis.hv("quantum_optics::entangled", PHYSICS_DIM),
            photon_antibunching: genesis.hv("quantum_optics::antibunching", PHYSICS_DIM),

            stimulated_emission: genesis.hv("laser::stimulated_emission", PHYSICS_DIM),
            population_inversion: genesis.hv("laser::population_inversion", PHYSICS_DIM),
            gain_medium: genesis.hv("laser::gain_medium", PHYSICS_DIM),
            cavity: genesis.hv("laser::cavity", PHYSICS_DIM),
            mode_locking: genesis.hv("laser::mode_locking", PHYSICS_DIM),

            fiber: genesis.hv("fiber::fiber", PHYSICS_DIM),
            guided_mode: genesis.hv("fiber::guided_mode", PHYSICS_DIM),
            dispersion: genesis.hv("fiber::dispersion", PHYSICS_DIM),
            soliton: genesis.hv("fiber::soliton", PHYSICS_DIM),

            hologram: genesis.hv("holography::hologram", PHYSICS_DIM),
            reference_beam: genesis.hv("holography::reference", PHYSICS_DIM),
            reconstruction: genesis.hv("holography::reconstruction", PHYSICS_DIM),
        }
    }

    // ============================================================
    // BEAM CREATION
    // ============================================================

    /// Create an optical beam
    pub fn create_beam(
        &self,
        wavelength_nm: f64,
        power_w: f64,
        waist_um: f64,
        statistics: PhotonStatistics,
    ) -> OpticalBeam {
        // Coherence length depends on source bandwidth
        let coherence_length = match statistics {
            PhotonStatistics::Coherent => wavelength_nm * 1e-9 * 1e6, // ~1m for laser
            PhotonStatistics::Thermal => wavelength_nm * 1e-9 * 10.0, // ~μm for thermal
            PhotonStatistics::Squeezed => wavelength_nm * 1e-9 * 1e6,
            PhotonStatistics::Fock => wavelength_nm * 1e-9 * 1e3,
            PhotonStatistics::Entangled => wavelength_nm * 1e-9 * 100.0,
        };

        let stat_vec = match statistics {
            PhotonStatistics::Coherent => &self.coherent_state,
            PhotonStatistics::Thermal => &self.light,
            PhotonStatistics::Squeezed => &self.squeezed_state,
            PhotonStatistics::Fock => &self.fock_state,
            PhotonStatistics::Entangled => &self.entangled_photons,
        };

        // Use log10(power + 1e-10) to avoid zero scaling at 1W
        let power_scale = (1.0 + power_w.abs()).log10() as f32;
        let vector = self
            .light
            .bind(&self.wavelength.scale((1000.0 / wavelength_nm) as f32))
            .bind(stat_vec)
            .bind(&self.intensity.scale(power_scale));

        OpticalBeam {
            wavelength_nm,
            power_w,
            beam_waist_um: waist_um,
            coherence_length_m: coherence_length,
            statistics,
            vector,
        }
    }

    // ============================================================
    // NONLINEAR OPTICS
    // ============================================================

    /// Second harmonic generation: ω → 2ω
    pub fn second_harmonic(&self, input: &OpticalBeam) -> ContinuousHV {
        let output_wavelength = input.wavelength_nm / 2.0;
        self.chi2
            .bind(&self.harmonic)
            .bind(&input.vector)
            .bind(&self.wavelength.scale((1000.0 / output_wavelength) as f32))
    }

    /// Kerr effect: n = n₀ + n₂I
    pub fn kerr_effect(&self, intensity: f64, n2: f64) -> ContinuousHV {
        let delta_n = n2 * intensity;
        self.kerr
            .bind(&self.chi3)
            .bind(&self.intensity.scale(intensity as f32))
            .scale(delta_n as f32)
    }

    /// Self-phase modulation in fiber
    pub fn self_phase_modulation(&self, power: f64, length_km: f64, gamma: f64) -> ContinuousHV {
        // Nonlinear phase: φ_NL = γ P L
        let phi_nl = gamma * power * length_km;
        self.self_phase_mod
            .bind(&self.kerr)
            .bind(&self.fiber)
            .scale(phi_nl as f32)
    }

    /// Parametric down-conversion: ω_p → ω_s + ω_i
    pub fn parametric_down_conversion(&self) -> ContinuousHV {
        self.chi2
            .bind(&self.parametric)
            .bind(&self.entangled_photons)
    }

    // ============================================================
    // QUANTUM OPTICS
    // ============================================================

    /// Create Fock state |n⟩
    pub fn fock_state_vec(&self, n: u32) -> ContinuousHV {
        self.fock_state.bind(&self.photon).scale(n as f32)
    }

    /// Coherent state |α⟩
    pub fn coherent_state_vec(&self, alpha: f64) -> ContinuousHV {
        // Mean photon number = |α|²
        let n_mean = alpha * alpha;
        self.coherent_state.bind(&self.photon).scale(n_mean as f32)
    }

    /// Squeezed state
    pub fn squeezed_state_vec(&self, squeeze_db: f64) -> ContinuousHV {
        self.squeezed_state
            .bind(&self.photon)
            .scale(squeeze_db as f32)
    }

    /// Hong-Ou-Mandel effect (two-photon interference)
    pub fn hong_ou_mandel(&self) -> ContinuousHV {
        self.interference
            .bind(&self.fock_state)
            .bind(&self.photon_antibunching)
    }

    /// Second-order correlation g²(0)
    pub fn g2_zero(&self, statistics: PhotonStatistics) -> f64 {
        match statistics {
            PhotonStatistics::Coherent => 1.0,  // Poissonian
            PhotonStatistics::Thermal => 2.0,   // Bunched
            PhotonStatistics::Squeezed => 0.5,  // Antibunched (approximate)
            PhotonStatistics::Fock => 0.0,      // Perfect antibunching for n=1
            PhotonStatistics::Entangled => 0.0, // Depends on state
        }
    }

    // ============================================================
    // LASERS
    // ============================================================

    /// Laser threshold condition
    pub fn laser_threshold(&self) -> ContinuousHV {
        self.stimulated_emission
            .bind(&self.population_inversion)
            .bind(&self.gain_medium)
            .bind(&self.cavity)
    }

    /// Mode-locked laser (ultrashort pulses)
    pub fn mode_locked_laser(&self, pulse_duration_fs: f64) -> ContinuousHV {
        self.mode_locking
            .bind(&self.cavity)
            .bind(&self.coherence)
            .scale((1000.0 / pulse_duration_fs) as f32) // Shorter pulse → higher score
    }

    /// Gain bandwidth product
    pub fn gain_bandwidth(&self, gain: f64, bandwidth_thz: f64) -> ContinuousHV {
        let gbp = gain * bandwidth_thz;
        self.gain_medium.bind(&self.frequency).scale(gbp as f32)
    }

    // ============================================================
    // FIBER OPTICS
    // ============================================================

    /// Create optical fiber
    pub fn create_fiber(
        &self,
        name: &str,
        core_um: f64,
        na: f64,
        attenuation: f64,
        dispersion: f64,
        nonlinear: f64,
    ) -> OpticalFiber {
        let vector = self
            .fiber
            .bind(&self.guided_mode)
            .bind(&self.dispersion.scale(dispersion.abs() as f32));

        OpticalFiber {
            name: name.to_string(),
            core_diameter_um: core_um,
            numerical_aperture: na,
            attenuation_db_km: attenuation,
            dispersion_ps_nm_km: dispersion,
            nonlinear_coeff: nonlinear,
            vector,
        }
    }

    /// Optical soliton (balance of dispersion and nonlinearity)
    pub fn optical_soliton(&self, fiber: &OpticalFiber, power: f64) -> ContinuousHV {
        // Soliton requires anomalous dispersion (D > 0 or β₂ < 0)
        if fiber.dispersion_ps_nm_km > 0.0 {
            self.soliton
                .bind(&fiber.vector)
                .bind(&self.kerr)
                .scale(power.sqrt() as f32)
        } else {
            ContinuousHV::zero(PHYSICS_DIM)
        }
    }

    /// V-number (normalized frequency)
    pub fn v_number(&self, fiber: &OpticalFiber, wavelength_nm: f64) -> f64 {
        let a = fiber.core_diameter_um / 2.0 * 1e-6;
        let lambda = wavelength_nm * 1e-9;
        2.0 * std::f64::consts::PI * a * fiber.numerical_aperture / lambda
    }

    /// Number of guided modes
    pub fn num_modes(&self, fiber: &OpticalFiber, wavelength_nm: f64) -> u32 {
        let v = self.v_number(fiber, wavelength_nm);
        if v < 2.405 {
            1 // Single mode
        } else {
            (v * v / 2.0) as u32 // Approximate for step-index
        }
    }

    // ============================================================
    // HOLOGRAPHY
    // ============================================================

    /// Record hologram
    pub fn record_hologram(&self) -> ContinuousHV {
        self.hologram
            .bind(&self.interference)
            .bind(&self.reference_beam)
            .bind(&self.coherence)
    }

    /// Reconstruct hologram
    pub fn reconstruct_hologram(&self) -> ContinuousHV {
        self.reconstruction
            .bind(&self.hologram)
            .bind(&self.diffraction)
    }

    // ============================================================
    // INTERFERENCE
    // ============================================================

    /// Two-beam interference: I = I₁ + I₂ + 2√(I₁I₂)cos(Δφ)
    pub fn two_beam_interference(&self, visibility: f64) -> ContinuousHV {
        self.interference
            .bind(&self.coherence)
            .bind(&self.superposition)
            .scale(visibility as f32)
    }

    /// Fabry-Perot interferometer
    pub fn fabry_perot(&self, finesse: f64) -> ContinuousHV {
        self.cavity
            .bind(&self.interference)
            .bind(&self.reflection)
            .scale(finesse as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> OpticsEncoder {
        let genesis = GenesisSeed::from_phrase("optics test");
        OpticsEncoder::from_genesis(&genesis)
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = setup();
        assert!(encoder.photon.norm() > 0.0);
        assert!(encoder.laser_threshold().norm() > 0.0);
    }

    #[test]
    fn test_beam_creation() {
        let encoder = setup();

        let laser = encoder.create_beam(532.0, 1.0, 100.0, PhotonStatistics::Coherent);

        // Green laser: photon energy ~2.33 eV
        let energy = laser.photon_energy_ev();
        assert!((energy - 2.33).abs() < 0.1);
    }

    #[test]
    fn test_second_harmonic() {
        let encoder = setup();

        let ir_laser = encoder.create_beam(1064.0, 1.0, 100.0, PhotonStatistics::Coherent);
        let shg = encoder.second_harmonic(&ir_laser);

        // SHG should produce non-zero vector
        assert!(shg.norm() > 0.0);
    }

    #[test]
    fn test_g2_values() {
        let encoder = setup();

        assert_eq!(encoder.g2_zero(PhotonStatistics::Coherent), 1.0);
        assert_eq!(encoder.g2_zero(PhotonStatistics::Thermal), 2.0);
        assert!(encoder.g2_zero(PhotonStatistics::Fock) < 1.0);
    }

    #[test]
    fn test_fiber_modes() {
        let encoder = setup();

        let smf = encoder.create_fiber("SMF-28", 8.2, 0.14, 0.2, 17.0, 1.3);
        let mmf = encoder.create_fiber("MMF", 62.5, 0.275, 0.5, -100.0, 1.0);

        // SMF at 1550nm should be single mode
        let n_smf = encoder.num_modes(&smf, 1550.0);
        assert_eq!(n_smf, 1);

        // MMF should have many modes
        let n_mmf = encoder.num_modes(&mmf, 850.0);
        assert!(n_mmf > 100);
    }

    #[test]
    fn test_soliton() {
        let encoder = setup();

        // Anomalous dispersion fiber
        let anom = encoder.create_fiber("Anomalous", 8.0, 0.12, 0.2, 17.0, 2.0);
        // Normal dispersion fiber
        let norm = encoder.create_fiber("Normal", 8.0, 0.12, 0.2, -20.0, 2.0);

        let sol_anom = encoder.optical_soliton(&anom, 1.0);
        let sol_norm = encoder.optical_soliton(&norm, 1.0);

        // Soliton only in anomalous dispersion
        assert!(sol_anom.norm() > 0.0);
        assert!(sol_norm.norm() < 0.01);
    }

    #[test]
    fn test_quantum_states() {
        let encoder = setup();

        let fock = encoder.fock_state_vec(5);
        let coherent = encoder.coherent_state_vec(2.0);
        let squeezed = encoder.squeezed_state_vec(10.0);

        // All should be valid
        assert!(fock.norm() > 0.0);
        assert!(coherent.norm() > 0.0);
        assert!(squeezed.norm() > 0.0);
    }

    #[test]
    fn test_holography() {
        let encoder = setup();

        let record = encoder.record_hologram();
        let reconstruct = encoder.reconstruct_hologram();

        assert!(record.norm() > 0.0);
        assert!(reconstruct.norm() > 0.0);
    }
}
