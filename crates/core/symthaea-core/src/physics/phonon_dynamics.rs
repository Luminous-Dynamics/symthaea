// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Phonon Dynamics: Lattice Vibrations for Nuclear Coupling
//!
//! This module models lattice vibrations (phonons) that can couple with nuclear transitions.
//!
//! ## The "Whispering Gallery" Effect
//!
//! Instead of brute-forcing a nucleus with X-rays, we look for materials where:
//! 1. The crystal lattice has a natural vibration frequency
//! 2. That frequency matches a nuclear transition
//! 3. Energy flows from sound → nucleus (or vice versa)
//!
//! ## Key Concepts
//!
//! - **Phonon**: A quantum of lattice vibration (like a photon but for sound)
//! - **Debye Frequency**: Maximum phonon frequency in a crystal
//! - **Phonon-Nuclear Coupling**: When lattice vibrations can trigger nuclear transitions
//!
//! ## Applications
//!
//! 1. **Lattice Confinement Fusion (LCF)**: NASA demonstrated deuterium fusion in Erbium lattices
//! 2. **Nuclear Battery Triggers**: Find lattices that naturally excite isomers
//! 3. **Mössbauer Effect**: Recoil-free nuclear transitions in crystals

use super::periodic_table::PeriodicTable;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Crystal structure types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CrystalStructure {
    /// Face-centered cubic (Cu, Al, Au, Ni)
    FCC,
    /// Body-centered cubic (Fe, W, Cr, Mo)
    BCC,
    /// Hexagonal close-packed (Ti, Zn, Mg)
    HCP,
    /// Diamond cubic (C, Si, Ge)
    Diamond,
    /// Fluorite (CaF2, ThO2)
    Fluorite,
    /// Rocksalt (NaCl)
    Rocksalt,
}

impl CrystalStructure {
    fn domain(&self) -> &'static str {
        match self {
            CrystalStructure::FCC => "crystal::fcc",
            CrystalStructure::BCC => "crystal::bcc",
            CrystalStructure::HCP => "crystal::hcp",
            CrystalStructure::Diamond => "crystal::diamond",
            CrystalStructure::Fluorite => "crystal::fluorite",
            CrystalStructure::Rocksalt => "crystal::rocksalt",
        }
    }

    /// Typical coordination number
    pub fn coordination(&self) -> u8 {
        match self {
            CrystalStructure::FCC => 12,
            CrystalStructure::BCC => 8,
            CrystalStructure::HCP => 12,
            CrystalStructure::Diamond => 4,
            CrystalStructure::Fluorite => 8,
            CrystalStructure::Rocksalt => 6,
        }
    }
}

/// A material with phonon properties
#[derive(Debug, Clone)]
pub struct LatticeMaterial {
    /// Material name
    pub name: String,
    /// Primary element atomic number
    pub primary_z: u8,
    /// Crystal structure
    pub structure: CrystalStructure,
    /// Debye temperature in Kelvin
    pub debye_temp_k: f64,
    /// Debye frequency in THz
    pub debye_freq_thz: f64,
    /// Lattice constant in Angstroms
    pub lattice_const_a: f64,
    /// Vector representation
    pub vector: ContinuousHV,
}

impl LatticeMaterial {
    /// Debye frequency in eV (ℏω)
    pub fn debye_energy_ev(&self) -> f64 {
        // ℏ = 6.582e-16 eV·s, 1 THz = 1e12 Hz
        // E = ℏω = 6.582e-16 * 1e12 * freq_thz = 6.582e-4 * freq_thz eV
        6.582e-4 * self.debye_freq_thz * 1000.0 // Convert to meV for clarity, then back
    }

    /// Maximum phonon energy in eV
    pub fn max_phonon_ev(&self) -> f64 {
        // E_phonon_max ≈ kB * Debye_temp
        // kB = 8.617e-5 eV/K
        8.617e-5 * self.debye_temp_k
    }
}

/// Phonon mode characteristics
#[derive(Debug, Clone)]
pub struct PhononMode {
    /// Mode frequency in THz
    pub frequency_thz: f64,
    /// Mode energy in eV
    pub energy_ev: f64,
    /// Mode type (acoustic/optical)
    pub mode_type: PhononType,
    /// Polarization (longitudinal/transverse)
    pub polarization: Polarization,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// Phonon mode types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhononType {
    /// Acoustic phonon (in-phase vibration)
    Acoustic,
    /// Optical phonon (out-of-phase vibration)
    Optical,
}

/// Phonon polarization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Polarization {
    Longitudinal,
    Transverse,
}

/// Phonon-nuclear coupling result
#[derive(Debug, Clone)]
pub struct PhononNuclearCoupling {
    /// The lattice material
    pub material: LatticeMaterial,
    /// The coupled phonon mode
    pub phonon: PhononMode,
    /// Target nuclear transition energy in eV
    pub nuclear_energy_ev: f64,
    /// Coupling strength (0-1)
    pub coupling_strength: f32,
    /// Detuning from resonance in eV
    pub detuning_ev: f64,
    /// Combined vector
    pub vector: ContinuousHV,
}

/// Phonon Dynamics System
#[derive(Debug, Clone)]
pub struct PhononDynamics {
    // Crystal structure vectors
    pub fcc: ContinuousHV,
    pub bcc: ContinuousHV,
    pub hcp: ContinuousHV,
    pub diamond: ContinuousHV,
    pub fluorite: ContinuousHV,
    pub rocksalt: ContinuousHV,

    // Phonon type vectors
    pub acoustic: ContinuousHV,
    pub optical: ContinuousHV,
    pub longitudinal: ContinuousHV,
    pub transverse: ContinuousHV,

    // Coupling concept vectors
    pub coherent: ContinuousHV,
    pub incoherent: ContinuousHV,
    pub resonant: ContinuousHV,
    pub anharmonic: ContinuousHV,

    // Material database
    pub materials: Vec<LatticeMaterial>,
}

impl PhononDynamics {
    /// Create phonon system from genesis
    pub fn from_genesis(genesis: &GenesisSeed, table: &PeriodicTable) -> Self {
        // Crystal structure vectors
        let fcc = genesis.hv(CrystalStructure::FCC.domain(), PHYSICS_DIM);
        let bcc = genesis.hv(CrystalStructure::BCC.domain(), PHYSICS_DIM);
        let hcp = genesis.hv(CrystalStructure::HCP.domain(), PHYSICS_DIM);
        let diamond = genesis.hv(CrystalStructure::Diamond.domain(), PHYSICS_DIM);
        let fluorite = genesis.hv(CrystalStructure::Fluorite.domain(), PHYSICS_DIM);
        let rocksalt = genesis.hv(CrystalStructure::Rocksalt.domain(), PHYSICS_DIM);

        // Phonon type vectors
        let acoustic = genesis.hv("phonon::acoustic", PHYSICS_DIM);
        let optical = genesis.hv("phonon::optical", PHYSICS_DIM);
        let longitudinal = genesis.hv("phonon::longitudinal", PHYSICS_DIM);
        let transverse = genesis.hv("phonon::transverse", PHYSICS_DIM);

        // Coupling concepts
        let coherent = genesis.hv("phonon::coherent", PHYSICS_DIM);
        let incoherent = genesis.hv("phonon::incoherent", PHYSICS_DIM);
        let resonant = genesis.hv("phonon::resonant", PHYSICS_DIM);
        let anharmonic = genesis.hv("phonon::anharmonic", PHYSICS_DIM);

        let mut system = Self {
            fcc,
            bcc,
            hcp,
            diamond,
            fluorite,
            rocksalt,
            acoustic,
            optical,
            longitudinal,
            transverse,
            coherent,
            incoherent,
            resonant,
            anharmonic,
            materials: Vec::new(),
        };

        system.init_materials(genesis, table);
        system
    }

    /// Initialize material database
    fn init_materials(&mut self, genesis: &GenesisSeed, table: &PeriodicTable) {
        // Common materials with known phonon properties
        let material_data = vec![
            // (name, Z, structure, Debye_K, Debye_THz, lattice_A)

            // Transition metals (good for LCF)
            ("Palladium", 46, CrystalStructure::FCC, 274.0, 5.7, 3.89),
            ("Titanium", 22, CrystalStructure::HCP, 420.0, 8.7, 2.95),
            ("Nickel", 28, CrystalStructure::FCC, 450.0, 9.4, 3.52),
            ("Erbium", 68, CrystalStructure::HCP, 168.0, 3.5, 3.56), // NASA LCF host
            ("Iron", 26, CrystalStructure::BCC, 470.0, 9.8, 2.87),   // Mössbauer
            // Actinide hosts
            ("Thorium", 90, CrystalStructure::FCC, 163.0, 3.4, 5.08),
            ("Uranium", 92, CrystalStructure::FCC, 207.0, 4.3, 3.47),
            // Fluorite structures (good for Thorium-229 clock)
            ("CaF2", 20, CrystalStructure::Fluorite, 510.0, 10.6, 5.46),
            ("ThO2", 90, CrystalStructure::Fluorite, 259.0, 5.4, 5.60),
            // Hydrogen storage / fusion hosts
            ("Deuterium-Pd", 46, CrystalStructure::FCC, 275.0, 5.7, 3.89),
            ("TiD2", 22, CrystalStructure::FCC, 400.0, 8.3, 4.45),
        ];

        for (name, z, structure, debye_k, debye_thz, lattice_a) in material_data {
            let struct_vec = self.structure_vector(structure);
            let element_vec = table
                .element(z)
                .map(|e| e.vector.clone())
                .unwrap_or_else(|| genesis.hv(&format!("element::{z}"), PHYSICS_DIM));

            // Material = Element bound with structure
            let mat_vec = element_vec.bind(struct_vec);

            self.materials.push(LatticeMaterial {
                name: name.to_string(),
                primary_z: z,
                structure,
                debye_temp_k: debye_k,
                debye_freq_thz: debye_thz,
                lattice_const_a: lattice_a,
                vector: mat_vec,
            });
        }
    }

    /// Get structure vector
    fn structure_vector(&self, structure: CrystalStructure) -> &ContinuousHV {
        match structure {
            CrystalStructure::FCC => &self.fcc,
            CrystalStructure::BCC => &self.bcc,
            CrystalStructure::HCP => &self.hcp,
            CrystalStructure::Diamond => &self.diamond,
            CrystalStructure::Fluorite => &self.fluorite,
            CrystalStructure::Rocksalt => &self.rocksalt,
        }
    }

    /// Create a phonon mode for analysis
    pub fn create_phonon_mode(
        &self,
        frequency_thz: f64,
        mode_type: PhononType,
        polarization: Polarization,
    ) -> PhononMode {
        let type_vec = match mode_type {
            PhononType::Acoustic => &self.acoustic,
            PhononType::Optical => &self.optical,
        };

        let pol_vec = match polarization {
            Polarization::Longitudinal => &self.longitudinal,
            Polarization::Transverse => &self.transverse,
        };

        // Energy: E = ℏω = h * f
        // h = 4.136e-15 eV·s
        let energy_ev = 4.136e-15 * frequency_thz * 1e12;

        let vector = type_vec.bind(pol_vec).scale((frequency_thz / 10.0) as f32);

        PhononMode {
            frequency_thz,
            energy_ev,
            mode_type,
            polarization,
            vector,
        }
    }

    /// Find materials with phonon modes matching a target energy
    pub fn find_resonant_materials(
        &self,
        target_ev: f64,
        tolerance: f64,
    ) -> Vec<(&LatticeMaterial, f64)> {
        let mut matches = Vec::new();

        for material in &self.materials {
            // Check if any phonon mode could match
            // Phonon energies range from 0 to max_phonon
            let max_phonon = material.max_phonon_ev();

            if target_ev <= max_phonon * 10.0 {
                // Within accessible range (allow multi-phonon processes)
                let n_phonons = (target_ev / max_phonon).ceil();
                let detuning = (target_ev - n_phonons * max_phonon).abs();
                let relative_detuning = detuning / target_ev.max(1e-6);

                if relative_detuning < tolerance {
                    matches.push((material, relative_detuning));
                }
            }
        }

        // Sort by best match
        matches.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }

    /// Calculate phonon-nuclear coupling strength
    ///
    /// This is a simplified model. Real calculations require
    /// detailed nuclear structure and solid-state physics.
    pub fn calculate_coupling(
        &self,
        material: &LatticeMaterial,
        nuclear_energy_ev: f64,
    ) -> PhononNuclearCoupling {
        // Create representative phonon mode
        let phonon = self.create_phonon_mode(
            material.debye_freq_thz * 0.7, // Use ~70% of Debye freq
            PhononType::Acoustic,
            Polarization::Longitudinal,
        );

        // Calculate how many phonons needed
        let n_phonons = (nuclear_energy_ev / phonon.energy_ev).round();
        let effective_energy = n_phonons * phonon.energy_ev;
        let detuning_ev = (nuclear_energy_ev - effective_energy).abs();

        // Coupling decreases with number of phonons required
        // (multi-phonon processes are less likely)
        let n_phonon_factor = (0.1_f64).powf(n_phonons.max(1.0) - 1.0);

        // Coupling decreases with detuning
        let resonance_factor = (-detuning_ev / nuclear_energy_ev.max(1.0)).exp();

        // Coherence factor from lattice
        let coherence_factor = match material.structure {
            CrystalStructure::FCC | CrystalStructure::BCC => 0.8,
            CrystalStructure::HCP => 0.7,
            CrystalStructure::Diamond | CrystalStructure::Fluorite => 0.9,
            CrystalStructure::Rocksalt => 0.6,
        };

        let coupling_strength = (n_phonon_factor * resonance_factor * coherence_factor) as f32;

        // Combined vector
        let vector = ContinuousHV::weighted_bundle(
            &[&material.vector, &phonon.vector, &self.resonant],
            &[1.0, 1.0, coupling_strength],
        );

        PhononNuclearCoupling {
            material: material.clone(),
            phonon,
            nuclear_energy_ev,
            coupling_strength,
            detuning_ev,
            vector,
        }
    }

    /// Find best lattice for hosting a nuclear isomer
    pub fn find_best_host(
        &self,
        _isomer_z: u8,
        nuclear_energy_ev: f64,
    ) -> Vec<PhononNuclearCoupling> {
        let mut couplings: Vec<PhononNuclearCoupling> = self
            .materials
            .iter()
            .map(|m| self.calculate_coupling(m, nuclear_energy_ev))
            .collect();

        // Sort by coupling strength
        couplings.sort_by(|a, b| {
            b.coupling_strength
                .partial_cmp(&a.coupling_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        couplings
    }

    /// Analyze Lattice Confinement Fusion potential
    ///
    /// For LCF, we need a lattice that:
    /// 1. Can host hydrogen/deuterium
    /// 2. Has high phonon density of states
    /// 3. Can confine and screen nuclei
    pub fn analyze_lcf_potential(&self, material: &LatticeMaterial) -> LCFAnalysis {
        // Hydrogen solubility indicator (high for Pd, Ti, Er)
        let h_solubility = match material.name.as_str() {
            "Palladium" | "Deuterium-Pd" => 1.0,
            "Titanium" | "TiD2" => 0.9,
            "Erbium" => 0.95, // NASA choice
            "Nickel" => 0.7,
            _ => 0.3,
        };

        // Electron screening factor (higher = more screening of Coulomb barrier)
        // Related to electron density at nucleus
        let coordination = material.structure.coordination() as f64;
        let screening_factor = coordination / 12.0; // Normalized to FCC

        // Phonon confinement factor for LCF
        // For LCF, moderate Debye temperatures are optimal:
        // - Too high = lattice too rigid, poor deuterium accommodation
        // - Too low = insufficient screening
        // Optimal range ~150-350K; use a broad Gaussian centered at 250K
        let debye_optimal = 250.0;
        let debye_sigma = 200.0;
        let phonon_factor =
            (-((material.debye_temp_k - debye_optimal) / debye_sigma).powi(2) / 2.0).exp();

        // Overall LCF potential: weighted combination favoring hydrogen solubility
        let lcf_potential =
            (0.5 * h_solubility + 0.25 * screening_factor + 0.25 * phonon_factor).min(1.0);

        LCFAnalysis {
            material_name: material.name.clone(),
            h_solubility,
            screening_factor,
            phonon_factor,
            lcf_potential: lcf_potential as f32,
        }
    }
}

/// Lattice Confinement Fusion analysis result
#[derive(Debug, Clone)]
pub struct LCFAnalysis {
    pub material_name: String,
    pub h_solubility: f64,
    pub screening_factor: f64,
    pub phonon_factor: f64,
    pub lcf_potential: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (GenesisSeed, PeriodicTable, PhononDynamics) {
        let genesis = GenesisSeed::from_phrase("phonon test");
        let model = super::super::StandardModel::from_genesis(&genesis);
        let hadrons = super::super::Hadrons::from_model(&model, &genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
        let phonons = PhononDynamics::from_genesis(&genesis, &table);
        (genesis, table, phonons)
    }

    #[test]
    fn test_phonon_creation() {
        let (_, _, phonons) = setup();
        assert!(!phonons.materials.is_empty());
    }

    #[test]
    fn test_material_energies() {
        let (_, _, phonons) = setup();

        for material in &phonons.materials {
            let max_phonon = material.max_phonon_ev();
            assert!(
                max_phonon > 0.0,
                "{} should have positive phonon energy",
                material.name
            );
            assert!(
                max_phonon < 0.1,
                "{} phonon energy {} too high",
                material.name,
                max_phonon
            );
        }
    }

    #[test]
    fn test_lcf_analysis() {
        let (_, _, phonons) = setup();

        // Find Palladium and Erbium
        let pd = phonons
            .materials
            .iter()
            .find(|m| m.name == "Palladium")
            .expect("Palladium should exist");

        let er = phonons
            .materials
            .iter()
            .find(|m| m.name == "Erbium")
            .expect("Erbium should exist");

        let pd_lcf = phonons.analyze_lcf_potential(pd);
        let er_lcf = phonons.analyze_lcf_potential(er);

        // Both should have high LCF potential
        assert!(
            pd_lcf.lcf_potential > 0.5,
            "Palladium should have good LCF potential"
        );
        assert!(
            er_lcf.lcf_potential > 0.5,
            "Erbium should have good LCF potential"
        );
    }

    #[test]
    fn test_phonon_mode_creation() {
        let (_, _, phonons) = setup();

        let mode = phonons.create_phonon_mode(
            5.0, // 5 THz
            PhononType::Acoustic,
            Polarization::Longitudinal,
        );

        assert!(mode.energy_ev > 0.0);
        assert_eq!(mode.frequency_thz, 5.0);
    }

    #[test]
    fn test_find_resonant_materials() {
        let (_, _, phonons) = setup();

        // Look for materials matching ~20 meV (typical phonon energy)
        let matches = phonons.find_resonant_materials(0.020, 0.5);

        assert!(
            !matches.is_empty(),
            "Should find materials with ~20 meV phonon modes"
        );
    }
}
