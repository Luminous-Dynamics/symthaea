// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Electron-Nuclear Coupling: The NEEC Bridge
//!
//! This module bridges the gap between Chemistry (electrons) and Nuclear Physics (nucleons).
//!
//! ## Key Insight
//!
//! The nucleus is not isolated - it's surrounded by an electron cloud. When electron
//! shell transition frequencies match nuclear transition frequencies, energy can
//! transfer between them through Nuclear Excitation by Electron Capture (NEEC).
//!
//! ## The Coupling Mechanism
//!
//! ```text
//! Standard View:          Coupled View:
//!
//!   Nucleus                  Nucleus ←──┐
//!     ↓                        ↓        │ Resonance
//!   [gap]                   Electron ───┘
//!     ↓                     Shell
//!   Electrons
//! ```
//!
//! ## Why This Matters
//!
//! Instead of needing a massive X-ray to trigger a nuclear transition (brute force),
//! you can "tickle" the electrons with a laser, and they trigger the nucleus.
//! This is the Rube Goldberg machine at atomic scale.
//!
//! ## Known Examples
//!
//! - **Thorium-229m**: 8.28 eV nuclear transition matches UV photon energy
//! - **Mo-93m**: 2.4 keV transition accessible via electron capture
//! - **Various isomers**: NEEC cross-sections being actively researched

use super::constants::RYDBERG_EV;
use super::hadrons::Hadrons;
use super::periodic_table::PeriodicTable;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Electron shell transition data
#[derive(Debug, Clone)]
pub struct ElectronTransition {
    /// Shell designation (K, L, M, N, O)
    pub shell: char,
    /// Sub-shell (s, p, d, f)
    pub subshell: char,
    /// Transition energy in eV
    pub energy_ev: f64,
    /// Transition width (lifetime uncertainty) in eV
    pub width_ev: f64,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// Nuclear transition data
#[derive(Debug, Clone)]
pub struct NuclearTransition {
    /// Atomic number
    pub z: u8,
    /// Mass number
    pub a: u16,
    /// Ground state to isomer transition energy in eV
    pub energy_ev: f64,
    /// Transition multipolarity (E1, M1, E2, M2, etc.)
    pub multipolarity: Multipolarity,
    /// Half-life in seconds
    pub half_life_s: f64,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// Electromagnetic transition multipolarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Multipolarity {
    E1, // Electric dipole
    M1, // Magnetic dipole
    E2, // Electric quadrupole
    M2, // Magnetic quadrupole
    E3, // Electric octupole
    M3, // Magnetic octupole
}

impl Multipolarity {
    fn domain(&self) -> &'static str {
        match self {
            Multipolarity::E1 => "transition::E1",
            Multipolarity::M1 => "transition::M1",
            Multipolarity::E2 => "transition::E2",
            Multipolarity::M2 => "transition::M2",
            Multipolarity::E3 => "transition::E3",
            Multipolarity::M3 => "transition::M3",
        }
    }

    /// Selection rule strength (higher = more likely transition)
    pub fn strength(&self) -> f32 {
        match self {
            Multipolarity::E1 => 1.0,
            Multipolarity::M1 => 0.8,
            Multipolarity::E2 => 0.3,
            Multipolarity::M2 => 0.2,
            Multipolarity::E3 => 0.05,
            Multipolarity::M3 => 0.03,
        }
    }
}

/// NEEC coupling result
#[derive(Debug, Clone)]
pub struct NEECCoupling {
    /// The electron transition involved
    pub electron: ElectronTransition,
    /// The nuclear transition involved
    pub nuclear: NuclearTransition,
    /// Energy match quality (0 = perfect match, higher = detuning)
    pub detuning_ev: f64,
    /// Coupling strength (0-1, based on overlap and selection rules)
    pub coupling_strength: f32,
    /// Combined vector representation
    pub vector: ContinuousHV,
}

/// Electron-Nuclear Coupling System
#[derive(Debug, Clone)]
pub struct ElectronNuclearCoupling {
    // Transition type vectors
    pub electric_dipole: ContinuousHV,
    pub magnetic_dipole: ContinuousHV,
    pub electric_quadrupole: ContinuousHV,
    pub magnetic_quadrupole: ContinuousHV,

    // Shell vectors
    pub k_shell: ContinuousHV,
    pub l_shell: ContinuousHV,
    pub m_shell: ContinuousHV,
    pub n_shell: ContinuousHV,

    // Coupling concept vectors
    pub resonance: ContinuousHV,
    pub detuning: ContinuousHV,
    pub coupling: ContinuousHV,
    pub forbidden: ContinuousHV,

    // Energy scale vectors (for interpolation)
    pub ev_scale: ContinuousHV,
    pub kev_scale: ContinuousHV,
    pub mev_scale: ContinuousHV,

    // Known promising candidates
    pub candidates: Vec<NEECCoupling>,
}

impl ElectronNuclearCoupling {
    /// Create NEEC system from genesis
    pub fn from_genesis(genesis: &GenesisSeed, table: &PeriodicTable, hadrons: &Hadrons) -> Self {
        // Transition multipolarity vectors
        let electric_dipole = genesis.hv(Multipolarity::E1.domain(), PHYSICS_DIM);
        let magnetic_dipole = genesis.hv(Multipolarity::M1.domain(), PHYSICS_DIM);
        let electric_quadrupole = genesis.hv(Multipolarity::E2.domain(), PHYSICS_DIM);
        let magnetic_quadrupole = genesis.hv(Multipolarity::M2.domain(), PHYSICS_DIM);

        // Electron shell vectors
        let k_shell = genesis.hv("shell::K", PHYSICS_DIM);
        let l_shell = genesis.hv("shell::L", PHYSICS_DIM);
        let m_shell = genesis.hv("shell::M", PHYSICS_DIM);
        let n_shell = genesis.hv("shell::N", PHYSICS_DIM);

        // Coupling concept vectors
        let resonance = genesis.hv("neec::resonance", PHYSICS_DIM);
        let detuning = genesis.hv("neec::detuning", PHYSICS_DIM);
        let coupling = genesis.hv("neec::coupling", PHYSICS_DIM);
        let forbidden = genesis.hv("neec::forbidden", PHYSICS_DIM);

        // Energy scale vectors
        let ev_scale = genesis.hv("energy::ev", PHYSICS_DIM);
        let kev_scale = genesis.hv("energy::kev", PHYSICS_DIM);
        let mev_scale = genesis.hv("energy::mev", PHYSICS_DIM);

        let mut system = Self {
            electric_dipole,
            magnetic_dipole,
            electric_quadrupole,
            magnetic_quadrupole,
            k_shell,
            l_shell,
            m_shell,
            n_shell,
            resonance,
            detuning,
            coupling,
            forbidden,
            ev_scale,
            kev_scale,
            mev_scale,
            candidates: Vec::new(),
        };

        // Initialize known NEEC candidates
        system.init_candidates(genesis, table, hadrons);

        system
    }

    /// Initialize known NEEC-promising candidates
    fn init_candidates(&mut self, genesis: &GenesisSeed, table: &PeriodicTable, hadrons: &Hadrons) {
        // Thorium-229m: The "nuclear clock" isomer
        // Extraordinarily low nuclear transition energy (~8.28 eV)
        // This is in the VUV range - accessible with lasers!
        let th229m = self.create_candidate(
            genesis,
            table,
            hadrons,
            90,
            229,  // Thorium-229
            8.28, // Nuclear transition energy in eV (VUV!)
            Multipolarity::M1,
            7200.0, // ~2 hour half-life for isomer
            "Th-229m: Nuclear clock candidate",
        );
        self.candidates.push(th229m);

        // Uranium-235m: Low-lying isomer
        let u235m = self.create_candidate(
            genesis,
            table,
            hadrons,
            92,
            235,
            76.8, // 76.8 eV - still very low for nuclear
            Multipolarity::M1,
            1560.0, // 26 minute half-life
            "U-235m: Low-lying actinide isomer",
        );
        self.candidates.push(u235m);

        // Molybdenum-93m: keV range, well-studied
        let mo93m = self.create_candidate(
            genesis,
            table,
            hadrons,
            42,
            93,
            2425.0, // 2.4 keV
            Multipolarity::M1,
            6.85 * 3600.0, // 6.85 hour half-life
            "Mo-93m: keV range isomer",
        );
        self.candidates.push(mo93m);

        // Iron-57m: Mössbauer famous
        let fe57m = self.create_candidate(
            genesis,
            table,
            hadrons,
            26,
            57,
            14400.0, // 14.4 keV - Mössbauer transition
            Multipolarity::M1,
            98e-9, // 98 ns half-life
            "Fe-57m: Mössbauer spectroscopy standard",
        );
        self.candidates.push(fe57m);

        // Technetium-99m: Medical imaging workhorse
        let tc99m = self.create_candidate(
            genesis,
            table,
            hadrons,
            43,
            99,
            140500.0, // 140.5 keV
            Multipolarity::M1,
            6.0 * 3600.0, // 6 hour half-life
            "Tc-99m: Medical imaging standard",
        );
        self.candidates.push(tc99m);
    }

    /// Create a NEEC candidate with full coupling analysis
    #[allow(clippy::too_many_arguments)]
    fn create_candidate(
        &self,
        _genesis: &GenesisSeed,
        _table: &PeriodicTable,
        hadrons: &Hadrons,
        z: u8,
        a: u16,
        nuclear_energy_ev: f64,
        multipolarity: Multipolarity,
        half_life_s: f64,
        _description: &str,
    ) -> NEECCoupling {
        // Create nuclear transition vector
        let nuclear_base = ContinuousHV::weighted_bundle(
            &[&hadrons.proton, &hadrons.neutron],
            &[z as f32, (a - z as u16) as f32],
        );

        let multi_vec = match multipolarity {
            Multipolarity::E1 => &self.electric_dipole,
            Multipolarity::M1 => &self.magnetic_dipole,
            Multipolarity::E2 => &self.electric_quadrupole,
            Multipolarity::M2 => &self.magnetic_quadrupole,
            _ => &self.electric_dipole,
        };

        // Encode energy scale
        let energy_vec = self.encode_energy(nuclear_energy_ev);

        let nuclear_vec = nuclear_base.bind(multi_vec).bind(&energy_vec);

        let nuclear = NuclearTransition {
            z,
            a,
            energy_ev: nuclear_energy_ev,
            multipolarity,
            half_life_s,
            vector: nuclear_vec.clone(),
        };

        // Find best matching electron transition
        // Use binding energy approximation: E_bind ≈ 13.6 * Z² / n² eV
        let (shell, shell_vec, electron_energy) = self.find_matching_shell(z, nuclear_energy_ev);

        let electron_vec = shell_vec.bind(&self.encode_energy(electron_energy));

        let electron = ElectronTransition {
            shell,
            subshell: 's',
            energy_ev: electron_energy,
            width_ev: electron_energy * 0.001, // Rough estimate
            vector: electron_vec.clone(),
        };

        // Calculate coupling strength
        let detuning_ev = (nuclear_energy_ev - electron_energy).abs();
        let relative_detuning = detuning_ev / nuclear_energy_ev.max(1.0);

        // Coupling strength decreases with detuning
        let resonance_factor = (-relative_detuning * 10.0).exp() as f32;
        let selection_factor = multipolarity.strength();
        let coupling_strength = resonance_factor * selection_factor;

        // Combined vector: bind electron and nuclear with resonance weighting
        let combined = if coupling_strength > 0.1 {
            ContinuousHV::weighted_bundle(
                &[&electron_vec, &nuclear_vec, &self.resonance],
                &[1.0, 1.0, coupling_strength],
            )
        } else {
            ContinuousHV::weighted_bundle(
                &[&electron_vec, &nuclear_vec, &self.detuning],
                &[1.0, 1.0, 1.0 - coupling_strength],
            )
        };

        NEECCoupling {
            electron,
            nuclear,
            detuning_ev,
            coupling_strength,
            vector: combined,
        }
    }

    /// Find the electron shell that best matches a given energy
    fn find_matching_shell(&self, z: u8, target_ev: f64) -> (char, &ContinuousHV, f64) {
        // NIST X-ray absorption edge energies (eV) for 19 elements
        // Format: (K, L, M, N) shell binding energies
        let nist: Option<(f64, f64, f64, f64)> = match z {
            22 => Some((4_966.0, 564.0, 61.0, 6.0)),             // Ti
            24 => Some((5_989.0, 696.0, 74.0, 7.0)),             // Cr
            25 => Some((6_539.0, 769.0, 83.0, 7.0)),             // Mn
            26 => Some((7_112.0, 846.0, 100.0, 8.0)),            // Fe
            28 => Some((8_333.0, 1_009.0, 112.0, 8.0)),          // Ni
            29 => Some((8_979.0, 1_097.0, 120.0, 8.0)),          // Cu
            30 => Some((9_659.0, 1_194.0, 137.0, 10.0)),         // Zn
            40 => Some((17_998.0, 2_532.0, 430.0, 51.0)),        // Zr
            42 => Some((20_000.0, 2_866.0, 505.0, 63.0)),        // Mo
            43 => Some((21_044.0, 3_043.0, 544.0, 68.0)),        // Tc
            47 => Some((25_514.0, 3_806.0, 719.0, 97.0)),        // Ag
            50 => Some((29_200.0, 4_465.0, 884.0, 137.0)),       // Sn
            56 => Some((37_441.0, 5_989.0, 1_293.0, 253.0)),     // Ba
            74 => Some((69_525.0, 12_100.0, 2_820.0, 595.0)),    // W
            78 => Some((78_395.0, 13_880.0, 3_296.0, 725.0)),    // Pt
            79 => Some((80_725.0, 14_353.0, 3_425.0, 762.0)),    // Au
            82 => Some((88_005.0, 15_861.0, 3_851.0, 894.0)),    // Pb
            90 => Some((109_651.0, 20_472.0, 5_182.0, 1_330.0)), // Th
            92 => Some((115_606.0, 21_757.0, 5_548.0, 1_441.0)), // U
            _ => None,
        };

        let (k_energy, l_energy, m_energy, n_energy) = match nist {
            Some((k, l, m, n)) => (k, l, m, n),
            None => {
                // Slater screening fallback for elements not in NIST table
                let z_eff = z as f64;
                let k = RYDBERG_EV * (z_eff - 0.3).max(1.0).powi(2);
                let l = RYDBERG_EV * (z_eff - 4.15).max(1.0).powi(2) / 4.0;
                let m = RYDBERG_EV * (z_eff - 11.25).max(1.0).powi(2) / 9.0;
                let n = RYDBERG_EV * (z_eff - 21.15).max(1.0).powi(2) / 16.0;
                (k, l, m, n)
            }
        };

        let shells = [
            ('K', &self.k_shell, k_energy),
            ('L', &self.l_shell, l_energy),
            ('M', &self.m_shell, m_energy),
            ('N', &self.n_shell, n_energy),
        ];

        // Find closest match
        shells
            .into_iter()
            .min_by(|a, b| {
                let diff_a = (a.2 - target_ev).abs();
                let diff_b = (b.2 - target_ev).abs();
                diff_a
                    .partial_cmp(&diff_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    }

    /// Encode an energy value as a vector
    fn encode_energy(&self, energy_ev: f64) -> ContinuousHV {
        // Determine which scale and create weighted interpolation
        if energy_ev < 1000.0 {
            // eV scale
            let weight = (energy_ev.ln() / 10.0).clamp(0.0, 1.0) as f32;
            self.ev_scale.scale(weight)
        } else if energy_ev < 1_000_000.0 {
            // keV scale
            let weight = ((energy_ev / 1000.0).ln() / 10.0).clamp(0.0, 1.0) as f32;
            ContinuousHV::weighted_bundle(
                &[&self.ev_scale, &self.kev_scale],
                &[1.0 - weight, weight],
            )
        } else {
            // MeV scale
            let weight = ((energy_ev / 1_000_000.0).ln() / 10.0).clamp(0.0, 1.0) as f32;
            ContinuousHV::weighted_bundle(
                &[&self.kev_scale, &self.mev_scale],
                &[1.0 - weight, weight],
            )
        }
    }

    /// Get best NEEC candidates sorted by coupling strength
    pub fn best_candidates(&self, n: usize) -> Vec<&NEECCoupling> {
        let mut sorted: Vec<&NEECCoupling> = self.candidates.iter().collect();
        sorted.sort_by(|a, b| {
            b.coupling_strength
                .partial_cmp(&a.coupling_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }

    /// Calculate NEEC cross-section estimate (relative units)
    ///
    /// Real NEEC cross-sections require detailed nuclear structure calculations,
    /// but we can estimate relative strengths using our coupling model.
    pub fn estimate_cross_section(&self, coupling: &NEECCoupling) -> f64 {
        // NEEC cross-section scales primarily with inverse transition energy squared
        // σ_NEEC ∝ (λ/2π)² ∝ 1/E² for the nuclear transition
        // Lower energy transitions have dramatically larger cross-sections
        let energy_ev = coupling.nuclear.energy_ev.max(1.0);
        let energy_factor = (1000.0 / energy_ev).powi(2);

        // Selection rule factor from multipolarity
        let selection_factor = coupling.nuclear.multipolarity.strength() as f64;

        // Half-life factor: longer-lived isomers have narrower linewidths
        // which can enhance resonant cross-section
        let halflife_factor = (coupling.nuclear.half_life_s.ln().max(0.0) + 1.0) / 10.0;

        energy_factor * selection_factor * (1.0 + halflife_factor)
    }

    /// Find resonances between two atoms (for lattice coupling)
    ///
    /// This looks for cases where atom A's electron transition
    /// can excite atom B's nuclear transition.
    pub fn find_cross_coupling(&self, donor_z: u8, acceptor: &NEECCoupling) -> f32 {
        // Get donor's characteristic electron energies
        let donor_k = 13.6 * (donor_z as f64 - 2.0).max(1.0).powi(2);
        let donor_l = 13.6 * (donor_z as f64 - 7.0).max(1.0).powi(2) / 4.0;

        // Check for resonance with acceptor's nuclear transition
        let target = acceptor.nuclear.energy_ev;

        let k_match = (-((donor_k - target).abs() / target.max(1.0)).powi(2)).exp();
        let l_match = (-((donor_l - target).abs() / target.max(1.0)).powi(2)).exp();

        (k_match.max(l_match) * acceptor.nuclear.multipolarity.strength() as f64) as f32
    }

    /// Compute resonance quality between two vectors
    pub fn resonance_quality(&self, a: &ContinuousHV, b: &ContinuousHV) -> f32 {
        // Resonance quality = how much both vectors align with resonance concept
        // plus how similar they are to each other (resonance = matching frequencies)
        let a_res = a.similarity(&self.resonance);
        let b_res = b.similarity(&self.resonance);
        let mutual = a.similarity(b);

        // Combined: both should be resonance-like and similar to each other
        (a_res + b_res) / 2.0 + mutual * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (GenesisSeed, PeriodicTable, Hadrons, ElectronNuclearCoupling) {
        let genesis = GenesisSeed::from_phrase("NEEC test");
        let model = super::super::StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
        let neec = ElectronNuclearCoupling::from_genesis(&genesis, &table, &hadrons);
        (genesis, table, hadrons, neec)
    }

    #[test]
    fn test_neec_creation() {
        let (_, _, _, neec) = setup();
        assert!(!neec.candidates.is_empty());
        assert_eq!(neec.resonance.dim(), PHYSICS_DIM);
    }

    #[test]
    fn test_thorium_229m_is_special() {
        let (_, _, _, neec) = setup();

        // Thorium-229m should have the lowest nuclear transition energy
        let th229m = neec
            .candidates
            .iter()
            .find(|c| c.nuclear.z == 90 && c.nuclear.a == 229)
            .expect("Th-229m should exist");

        assert!(
            th229m.nuclear.energy_ev < 100.0,
            "Th-229m should have very low transition energy: {} eV",
            th229m.nuclear.energy_ev
        );

        // It should have high coupling potential due to low energy
        let cross_section = neec.estimate_cross_section(th229m);

        let tc99m = neec
            .candidates
            .iter()
            .find(|c| c.nuclear.z == 43)
            .expect("Tc-99m should exist");
        let tc_cross = neec.estimate_cross_section(tc99m);

        assert!(
            cross_section > tc_cross,
            "Th-229m should have higher cross-section than Tc-99m"
        );
    }

    #[test]
    fn test_best_candidates() {
        let (_, _, _, neec) = setup();

        let best = neec.best_candidates(3);
        assert_eq!(best.len(), 3);

        // Should be sorted by coupling strength
        assert!(best[0].coupling_strength >= best[1].coupling_strength);
        assert!(best[1].coupling_strength >= best[2].coupling_strength);
    }

    #[test]
    fn test_energy_encoding() {
        let (_, _, _, neec) = setup();

        let ev_vec = neec.encode_energy(10.0);
        let kev_vec = neec.encode_energy(10_000.0);
        let mev_vec = neec.encode_energy(10_000_000.0);

        // Different energy scales should produce different vectors
        assert!(
            ev_vec.similarity(&mev_vec) < 0.5,
            "eV and MeV vectors should be distinct"
        );

        // But adjacent scales should have some similarity
        assert!(
            ev_vec.similarity(&kev_vec) > kev_vec.similarity(&mev_vec),
            "Adjacent scales should be more similar"
        );
    }

    #[test]
    fn test_resonance_quality() {
        let (_, _, _, neec) = setup();

        // Resonance with itself should be high
        let self_resonance = neec.resonance_quality(&neec.resonance, &neec.resonance);

        // Resonance with detuning should be lower
        let anti_resonance = neec.resonance_quality(&neec.resonance, &neec.detuning);

        assert!(
            self_resonance > anti_resonance,
            "Resonance-resonance should beat resonance-detuning"
        );
    }
}
