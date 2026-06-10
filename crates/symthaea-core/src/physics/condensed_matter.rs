// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Condensed Matter Physics
//!
//! **Status**: Exploratory module — HDC encodings for condensed matter physics (band theory, phonons, correlated electrons).
//!
//! This module encodes condensed matter concepts from band theory to correlated electrons.
//!
//! ## Band Theory
//!
//! - Electronic band structure from periodic potential
//! - Metals: partially filled bands
//! - Insulators: full valence band, empty conduction band
//! - Semiconductors: small band gap
//!
//! ## Phonons
//!
//! - Quantized lattice vibrations
//! - Acoustic and optical branches
//! - Electron-phonon coupling
//!
//! ## Correlated Electrons
//!
//! - Mott insulators
//! - Heavy fermions
//! - Quantum criticality

use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Material type based on band structure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialType {
    Metal,
    Semiconductor,
    Insulator,
    Semimetal,
    Superconductor,
}

impl MaterialType {
    fn domain(&self) -> &'static str {
        match self {
            MaterialType::Metal => "cm::metal",
            MaterialType::Semiconductor => "cm::semiconductor",
            MaterialType::Insulator => "cm::insulator",
            MaterialType::Semimetal => "cm::semimetal",
            MaterialType::Superconductor => "cm::superconductor",
        }
    }
}

/// Phonon branch type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhononBranch {
    AcousticLongitudinal,
    AcousticTransverse,
    OpticalLongitudinal,
    OpticalTransverse,
}

impl PhononBranch {
    fn domain(&self) -> &'static str {
        match self {
            PhononBranch::AcousticLongitudinal => "phonon::la",
            PhononBranch::AcousticTransverse => "phonon::ta",
            PhononBranch::OpticalLongitudinal => "phonon::lo",
            PhononBranch::OpticalTransverse => "phonon::to",
        }
    }
}

/// Crystal lattice type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatticeType {
    SimpleCubic,
    BodyCenteredCubic,
    FaceCenteredCubic,
    Hexagonal,
    Diamond,
    Graphene,
}

impl LatticeType {
    fn domain(&self) -> &'static str {
        match self {
            LatticeType::SimpleCubic => "lattice::sc",
            LatticeType::BodyCenteredCubic => "lattice::bcc",
            LatticeType::FaceCenteredCubic => "lattice::fcc",
            LatticeType::Hexagonal => "lattice::hcp",
            LatticeType::Diamond => "lattice::diamond",
            LatticeType::Graphene => "lattice::graphene",
        }
    }
}

/// A band structure
#[derive(Debug, Clone)]
pub struct BandStructure {
    pub material_type: MaterialType,
    /// Band gap in eV (0 for metals)
    pub band_gap_ev: f64,
    /// Effective electron mass (in units of m_e)
    pub effective_mass: f64,
    /// Fermi energy in eV
    pub fermi_energy_ev: f64,
    pub vector: ContinuousHV,
}

/// A phonon mode
#[derive(Debug, Clone)]
pub struct PhononMode {
    pub branch: PhononBranch,
    /// Frequency in THz
    pub frequency_thz: f64,
    /// Wave vector (arbitrary units)
    pub wave_vector: f64,
    pub vector: ContinuousHV,
}

/// Fermi surface topology
#[derive(Debug, Clone)]
pub struct FermiSurface {
    pub name: String,
    /// Density of states at Fermi level
    pub dos_at_fermi: f64,
    /// Fermi velocity in m/s
    pub fermi_velocity: f64,
    pub vector: ContinuousHV,
}

/// Condensed matter encoder
#[derive(Debug, Clone)]
pub struct CMEncoder {
    // Band structure concepts
    pub valence_band: ContinuousHV,
    pub conduction_band: ContinuousHV,
    pub band_gap: ContinuousHV,
    pub fermi_level: ContinuousHV,
    pub fermi_surface: ContinuousHV,

    // Material types
    pub metal: ContinuousHV,
    pub semiconductor: ContinuousHV,
    pub insulator: ContinuousHV,

    // Electron concepts
    pub electron: ContinuousHV,
    pub hole: ContinuousHV,
    pub effective_mass: ContinuousHV,
    pub mobility: ContinuousHV,

    // Phonon concepts
    pub phonon: ContinuousHV,
    pub acoustic: ContinuousHV,
    pub optical: ContinuousHV,
    pub debye: ContinuousHV,

    // Lattice
    pub lattice: ContinuousHV,
    pub brillouin_zone: ContinuousHV,
    pub reciprocal: ContinuousHV,

    // Interactions
    pub electron_phonon: ContinuousHV,
    pub coulomb: ContinuousHV,
    pub exchange: ContinuousHV,

    // Correlated phenomena
    pub mott: ContinuousHV,
    pub heavy_fermion: ContinuousHV,
    pub quantum_critical: ContinuousHV,

    // Magnetism
    pub ferromagnetic: ContinuousHV,
    pub antiferromagnetic: ContinuousHV,
    pub paramagnetic: ContinuousHV,
}

impl CMEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            valence_band: genesis.hv("cm::valence_band", PHYSICS_DIM),
            conduction_band: genesis.hv("cm::conduction_band", PHYSICS_DIM),
            band_gap: genesis.hv("cm::band_gap", PHYSICS_DIM),
            fermi_level: genesis.hv("cm::fermi_level", PHYSICS_DIM),
            fermi_surface: genesis.hv("cm::fermi_surface", PHYSICS_DIM),

            metal: genesis.hv(MaterialType::Metal.domain(), PHYSICS_DIM),
            semiconductor: genesis.hv(MaterialType::Semiconductor.domain(), PHYSICS_DIM),
            insulator: genesis.hv(MaterialType::Insulator.domain(), PHYSICS_DIM),

            electron: genesis.hv("cm::electron", PHYSICS_DIM),
            hole: genesis.hv("cm::hole", PHYSICS_DIM),
            effective_mass: genesis.hv("cm::effective_mass", PHYSICS_DIM),
            mobility: genesis.hv("cm::mobility", PHYSICS_DIM),

            phonon: genesis.hv("cm::phonon", PHYSICS_DIM),
            acoustic: genesis.hv("phonon::acoustic", PHYSICS_DIM),
            optical: genesis.hv("phonon::optical", PHYSICS_DIM),
            debye: genesis.hv("cm::debye", PHYSICS_DIM),

            lattice: genesis.hv("cm::lattice", PHYSICS_DIM),
            brillouin_zone: genesis.hv("cm::brillouin_zone", PHYSICS_DIM),
            reciprocal: genesis.hv("cm::reciprocal_space", PHYSICS_DIM),

            electron_phonon: genesis.hv("cm::electron_phonon", PHYSICS_DIM),
            coulomb: genesis.hv("cm::coulomb", PHYSICS_DIM),
            exchange: genesis.hv("cm::exchange", PHYSICS_DIM),

            mott: genesis.hv("cm::mott_insulator", PHYSICS_DIM),
            heavy_fermion: genesis.hv("cm::heavy_fermion", PHYSICS_DIM),
            quantum_critical: genesis.hv("cm::quantum_critical", PHYSICS_DIM),

            ferromagnetic: genesis.hv("cm::ferromagnetic", PHYSICS_DIM),
            antiferromagnetic: genesis.hv("cm::antiferromagnetic", PHYSICS_DIM),
            paramagnetic: genesis.hv("cm::paramagnetic", PHYSICS_DIM),
        }
    }

    /// Create band structure for a metal
    pub fn create_metal(&self, fermi_energy_ev: f64, _genesis: &GenesisSeed) -> BandStructure {
        let vector = self
            .metal
            .bind(&self.fermi_surface)
            .scale(fermi_energy_ev as f32);

        BandStructure {
            material_type: MaterialType::Metal,
            band_gap_ev: 0.0,
            effective_mass: 1.0,
            fermi_energy_ev,
            vector,
        }
    }

    /// Create band structure for a semiconductor
    pub fn create_semiconductor(
        &self,
        band_gap_ev: f64,
        effective_mass: f64,
        _genesis: &GenesisSeed,
    ) -> BandStructure {
        let vector = self
            .semiconductor
            .bind(&self.band_gap)
            .scale(band_gap_ev as f32)
            .bind(&self.effective_mass.scale(effective_mass as f32));

        BandStructure {
            material_type: MaterialType::Semiconductor,
            band_gap_ev,
            effective_mass,
            fermi_energy_ev: band_gap_ev / 2.0, // Intrinsic: Fermi level at mid-gap
            vector,
        }
    }

    /// Create band structure for an insulator
    pub fn create_insulator(&self, band_gap_ev: f64, _genesis: &GenesisSeed) -> BandStructure {
        let vector = self
            .insulator
            .bind(&self.band_gap)
            .scale(band_gap_ev as f32);

        BandStructure {
            material_type: MaterialType::Insulator,
            band_gap_ev,
            effective_mass: 1.0,
            fermi_energy_ev: band_gap_ev / 2.0,
            vector,
        }
    }

    /// Silicon (Si)
    pub fn silicon(&self, genesis: &GenesisSeed) -> BandStructure {
        self.create_semiconductor(1.12, 0.26, genesis)
    }

    /// Germanium (Ge)
    pub fn germanium(&self, genesis: &GenesisSeed) -> BandStructure {
        self.create_semiconductor(0.67, 0.12, genesis)
    }

    /// Gallium arsenide (GaAs)
    pub fn gallium_arsenide(&self, genesis: &GenesisSeed) -> BandStructure {
        self.create_semiconductor(1.42, 0.067, genesis)
    }

    /// Create a phonon mode
    pub fn create_phonon(
        &self,
        branch: PhononBranch,
        frequency_thz: f64,
        wave_vector: f64,
        genesis: &GenesisSeed,
    ) -> PhononMode {
        let branch_vec = genesis.hv(branch.domain(), PHYSICS_DIM);
        let vector = self.phonon.bind(&branch_vec).scale(frequency_thz as f32);

        PhononMode {
            branch,
            frequency_thz,
            wave_vector,
            vector,
        }
    }

    /// Acoustic phonon at long wavelength
    pub fn acoustic_phonon(&self, frequency_thz: f64, genesis: &GenesisSeed) -> PhononMode {
        self.create_phonon(
            PhononBranch::AcousticLongitudinal,
            frequency_thz,
            0.1,
            genesis,
        )
    }

    /// Optical phonon
    pub fn optical_phonon(&self, frequency_thz: f64, genesis: &GenesisSeed) -> PhononMode {
        self.create_phonon(
            PhononBranch::OpticalLongitudinal,
            frequency_thz,
            0.0,
            genesis,
        )
    }

    /// Debye model: phonon density of states ~ ω²
    pub fn debye_density_of_states(&self, omega: f64, omega_debye: f64) -> f64 {
        if omega <= omega_debye {
            9.0 * omega * omega / omega_debye.powi(3)
        } else {
            0.0
        }
    }

    /// Electron-phonon coupling strength (dimensionless λ)
    pub fn electron_phonon_coupling(
        &self,
        electron: &ContinuousHV,
        phonon: &PhononMode,
    ) -> ContinuousHV {
        electron.bind(&phonon.vector).bind(&self.electron_phonon)
    }

    /// Fermi-Dirac distribution
    pub fn fermi_dirac(&self, energy_ev: f64, fermi_ev: f64, temperature_k: f64) -> f64 {
        if temperature_k <= 0.0 {
            return if energy_ev <= fermi_ev { 1.0 } else { 0.0 };
        }
        let k_b = 8.617e-5; // eV/K
        let x = (energy_ev - fermi_ev) / (k_b * temperature_k);
        1.0 / (x.exp() + 1.0)
    }

    /// Mott insulator: correlation-driven insulating state
    pub fn mott_insulator(&self, u_ev: f64, t_ev: f64) -> ContinuousHV {
        // U = Coulomb repulsion, t = hopping
        // Mott transition when U/t ~ 1
        let u_over_t = u_ev / t_ev;
        self.mott
            .bind(&self.coulomb.scale(u_ev as f32))
            .scale(u_over_t as f32)
    }

    /// Heavy fermion system
    pub fn heavy_fermion_system(&self, effective_mass_enhancement: f64) -> ContinuousHV {
        self.heavy_fermion
            .bind(&self.effective_mass)
            .scale(effective_mass_enhancement.ln() as f32)
    }

    /// Bloch's theorem: ψ_k(r) = e^{ik·r} u_k(r)
    pub fn bloch_wave(&self, k_vector: &ContinuousHV, genesis: &GenesisSeed) -> ContinuousHV {
        let periodic = genesis.hv("cm::periodic", PHYSICS_DIM);
        k_vector.bind(&periodic).bind(&self.lattice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (CMEncoder, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("condensed matter test");
        let encoder = CMEncoder::from_genesis(&genesis);
        (encoder, genesis)
    }

    #[test]
    fn test_cm_encoder_creation() {
        let (encoder, _) = setup();
        assert!(encoder.valence_band.norm() > 0.0);
        assert!(encoder.phonon.norm() > 0.0);
    }

    #[test]
    fn test_material_types() {
        let (encoder, genesis) = setup();

        let metal = encoder.create_metal(5.0, &genesis);
        let semi = encoder.silicon(&genesis);
        let insulator = encoder.create_insulator(9.0, &genesis);

        assert_eq!(metal.material_type, MaterialType::Metal);
        assert_eq!(semi.material_type, MaterialType::Semiconductor);
        assert_eq!(insulator.material_type, MaterialType::Insulator);

        // Metal has no band gap
        assert!((metal.band_gap_ev - 0.0).abs() < 0.01);

        // Silicon band gap ~ 1.12 eV
        assert!((semi.band_gap_ev - 1.12).abs() < 0.01);
    }

    #[test]
    fn test_semiconductors() {
        let (encoder, genesis) = setup();

        let si = encoder.silicon(&genesis);
        let ge = encoder.germanium(&genesis);
        let gaas = encoder.gallium_arsenide(&genesis);

        // Band gaps: Si > Ge
        assert!(si.band_gap_ev > ge.band_gap_ev);

        // GaAs is direct gap
        assert!(gaas.band_gap_ev > 1.0);
    }

    #[test]
    fn test_phonon_modes() {
        let (encoder, genesis) = setup();

        let acoustic = encoder.acoustic_phonon(2.0, &genesis);
        let optical = encoder.optical_phonon(10.0, &genesis);

        assert_eq!(acoustic.branch, PhononBranch::AcousticLongitudinal);
        assert_eq!(optical.branch, PhononBranch::OpticalLongitudinal);

        // Optical phonons have higher frequency
        assert!(optical.frequency_thz > acoustic.frequency_thz);
    }

    #[test]
    fn test_fermi_dirac() {
        let (encoder, _) = setup();

        // At T=0, f(E) = 1 for E < E_F
        // At finite T, f(E_F) = 0.5
        let f_at_fermi = encoder.fermi_dirac(5.0, 5.0, 300.0);
        assert!((f_at_fermi - 0.5).abs() < 0.01);

        // Well below Fermi level: f ≈ 1
        let f_below = encoder.fermi_dirac(4.0, 5.0, 300.0);
        assert!(f_below > 0.99);

        // Well above Fermi level: f ≈ 0
        let f_above = encoder.fermi_dirac(6.0, 5.0, 300.0);
        assert!(f_above < 0.01);
    }

    #[test]
    fn test_mott_insulator() {
        let (encoder, _) = setup();

        // Large U/t → Mott insulator
        let mott_strong = encoder.mott_insulator(10.0, 1.0); // U/t = 10
        let mott_weak = encoder.mott_insulator(1.0, 10.0); // U/t = 0.1

        // Strong correlation (large U/t) should have larger norm
        // (scaling encodes magnitude)
        let strong_norm = mott_strong.norm();
        let weak_norm = mott_weak.norm();
        assert!(
            strong_norm > weak_norm,
            "Large U/t should give larger norm: {} vs {}",
            strong_norm,
            weak_norm
        );
    }

    #[test]
    fn test_heavy_fermion() {
        let (encoder, _) = setup();

        // Heavy fermion: m* ~ 100-1000 m_e
        let hf = encoder.heavy_fermion_system(100.0);
        assert!(hf.norm() > 0.0);
    }

    #[test]
    fn test_debye_dos() {
        let (encoder, _) = setup();

        // DOS should be zero above Debye frequency
        let dos_below = encoder.debye_density_of_states(0.5, 1.0);
        let dos_above = encoder.debye_density_of_states(1.5, 1.0);

        assert!(dos_below > 0.0);
        assert!((dos_above - 0.0).abs() < 0.001);
    }
}
