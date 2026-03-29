// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Nuclear Physics: Energy, Binding, and Isomers
//!
//! This module extends the physics hierarchy with nuclear energy concepts:
//! - Binding energy curves (why iron is most stable)
//! - Nuclear isomers (metastable excited states)
//! - Fusion and fission pathways
//! - Energy density comparison (eV vs keV vs MeV)
//!
//! ## The Energy Hierarchy
//!
//! ```text
//! Level        Energy Scale    Storage Mechanism
//! ─────────────────────────────────────────────────
//! Chemical     ~1 eV          Electron orbitals
//! Nuclear      ~1 MeV         Nuclear binding
//! Isomer       ~100 keV       Excited nuclear states
//! Antimatter   ~1 GeV         Mass-energy conversion
//! ```
//!
//! ## Key Insight: Compositional Energy
//!
//! Energy is not a scalar—it's a vector that can be:
//! - Bound to particles (binding energy)
//! - Released in transitions (decay energy)
//! - Stored in metastable states (isomer energy)
//!
//! This enables reasoning about energy transformations as
//! vector operations.

use super::hadrons::Hadrons;
use super::periodic_table::PeriodicTable;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Energy scales for comparison
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EnergyScale {
    /// Chemical bonds: ~1 eV
    Chemical,
    /// Nuclear gamma transitions: ~100 keV
    Isomeric,
    /// Nuclear binding: ~1 MeV per nucleon
    Nuclear,
    /// Mass-energy: ~1 GeV per nucleon
    Relativistic,
}

impl EnergyScale {
    /// Typical energy in eV
    pub fn typical_ev(&self) -> f64 {
        match self {
            EnergyScale::Chemical => 1.0,
            EnergyScale::Isomeric => 100_000.0,
            EnergyScale::Nuclear => 1_000_000.0,
            EnergyScale::Relativistic => 1_000_000_000.0,
        }
    }

    /// Energy density ratio compared to chemical
    pub fn density_ratio(&self) -> f64 {
        self.typical_ev() / EnergyScale::Chemical.typical_ev()
    }
}

/// Nuclear isomer (metastable excited state)
#[derive(Debug, Clone)]
pub struct NuclearIsomer {
    /// Base element (atomic number)
    pub atomic_number: u8,
    /// Mass number (protons + neutrons)
    pub mass_number: u16,
    /// Excitation energy in keV
    pub excitation_kev: f32,
    /// Half-life in seconds (None = stable ground state)
    pub half_life_s: Option<f64>,
    /// Isomer state name (e.g., "m1", "m2")
    pub state: String,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// Nuclear reaction types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NuclearReaction {
    /// Light nuclei combine: D + T → He + n
    Fusion,
    /// Heavy nucleus splits: U-235 → fragments
    Fission,
    /// Alpha particle emission
    AlphaDecay,
    /// Beta minus: neutron → proton + electron + antineutrino
    BetaMinusDecay,
    /// Beta plus: proton → neutron + positron + neutrino
    BetaPlusDecay,
    /// Gamma emission (isomeric transition)
    GammaDecay,
    /// Neutron capture
    NeutronCapture,
}

/// Binding energy data point
#[derive(Debug, Clone)]
pub struct BindingEnergyPoint {
    pub mass_number: u16,
    pub binding_per_nucleon_mev: f32,
    pub element_symbol: &'static str,
}

/// Nuclear physics system
#[derive(Debug, Clone)]
pub struct NuclearPhysics {
    // Energy scale vectors
    pub chemical_energy: ContinuousHV,
    pub isomeric_energy: ContinuousHV,
    pub nuclear_energy: ContinuousHV,
    pub relativistic_energy: ContinuousHV,

    // Reaction type vectors
    pub fusion: ContinuousHV,
    pub fission: ContinuousHV,
    pub alpha_decay: ContinuousHV,
    pub beta_decay: ContinuousHV,
    pub gamma_decay: ContinuousHV,

    // Stability concepts
    pub stable: ContinuousHV,
    pub radioactive: ContinuousHV,
    pub metastable: ContinuousHV,

    // Binding energy curve reference points
    binding_curve: Vec<BindingEnergyPoint>,

    // Notable isomers
    pub isomers: Vec<NuclearIsomer>,
}

impl NuclearPhysics {
    /// Create nuclear physics system from genesis
    pub fn from_genesis(genesis: &GenesisSeed, hadrons: &Hadrons, table: &PeriodicTable) -> Self {
        // Energy scale vectors
        let chemical_energy = genesis.hv("nuclear::energy::chemical", PHYSICS_DIM);
        let isomeric_energy = genesis.hv("nuclear::energy::isomeric", PHYSICS_DIM);
        let nuclear_energy = genesis.hv("nuclear::energy::nuclear", PHYSICS_DIM);
        let relativistic_energy = genesis.hv("nuclear::energy::relativistic", PHYSICS_DIM);

        // Reaction vectors
        let fusion = genesis.hv("nuclear::reaction::fusion", PHYSICS_DIM);
        let fission = genesis.hv("nuclear::reaction::fission", PHYSICS_DIM);
        let alpha_decay = genesis.hv("nuclear::reaction::alpha", PHYSICS_DIM);
        let beta_decay = genesis.hv("nuclear::reaction::beta", PHYSICS_DIM);
        let gamma_decay = genesis.hv("nuclear::reaction::gamma", PHYSICS_DIM);

        // Stability vectors
        let stable = genesis.hv("nuclear::stability::stable", PHYSICS_DIM);
        let radioactive = genesis.hv("nuclear::stability::radioactive", PHYSICS_DIM);
        let metastable = genesis.hv("nuclear::stability::metastable", PHYSICS_DIM);

        // Binding energy curve (empirical data)
        let binding_curve = Self::init_binding_curve();

        // Notable nuclear isomers
        let isomers = Self::init_isomers(genesis, table, hadrons);

        Self {
            chemical_energy,
            isomeric_energy,
            nuclear_energy,
            relativistic_energy,
            fusion,
            fission,
            alpha_decay,
            beta_decay,
            gamma_decay,
            stable,
            radioactive,
            metastable,
            binding_curve,
            isomers,
        }
    }

    /// Initialize binding energy curve
    fn init_binding_curve() -> Vec<BindingEnergyPoint> {
        vec![
            BindingEnergyPoint {
                mass_number: 2,
                binding_per_nucleon_mev: 1.11,
                element_symbol: "H",
            },
            BindingEnergyPoint {
                mass_number: 4,
                binding_per_nucleon_mev: 7.07,
                element_symbol: "He",
            },
            BindingEnergyPoint {
                mass_number: 12,
                binding_per_nucleon_mev: 7.68,
                element_symbol: "C",
            },
            BindingEnergyPoint {
                mass_number: 16,
                binding_per_nucleon_mev: 7.98,
                element_symbol: "O",
            },
            BindingEnergyPoint {
                mass_number: 56,
                binding_per_nucleon_mev: 8.79,
                element_symbol: "Fe",
            }, // PEAK
            BindingEnergyPoint {
                mass_number: 62,
                binding_per_nucleon_mev: 8.79,
                element_symbol: "Ni",
            },
            BindingEnergyPoint {
                mass_number: 238,
                binding_per_nucleon_mev: 7.57,
                element_symbol: "U",
            },
        ]
    }

    /// Initialize notable nuclear isomers
    ///
    /// Creates isomer vectors from their base nucleus composition, not from the
    /// periodic table, so isomers can exist even for elements not in the table.
    fn init_isomers(
        genesis: &GenesisSeed,
        table: &PeriodicTable,
        hadrons: &Hadrons,
    ) -> Vec<NuclearIsomer> {
        let mut isomers = Vec::new();

        // Helper to create isomer base vector from atomic composition
        let create_base = |z: u8, n: u8| -> ContinuousHV {
            // Create nucleus from protons and neutrons
            ContinuousHV::weighted_bundle(
                &[&hadrons.proton, &hadrons.neutron],
                &[z as f32, n as f32],
            )
        };

        // Tantalum-180m: The only naturally occurring isomer
        // Half-life > 10^15 years, 77 keV excitation
        // Ta: Z=73, A=180, N=107
        let ta_base = create_base(73, 107);
        let ta_isomer_label = genesis.hv("isomer::Ta180m", PHYSICS_DIM);
        let ta_vector = ta_base.bind(&ta_isomer_label);
        isomers.push(NuclearIsomer {
            atomic_number: 73,
            mass_number: 180,
            excitation_kev: 77.0,
            half_life_s: Some(1.0e23), // >10^15 years
            state: "m".to_string(),
            vector: ta_vector,
        });

        // Hafnium-178m2: High-energy isomer (2.4 MeV), studied for energy storage
        // Hf: Z=72, A=178, N=106
        let hf_base = create_base(72, 106);
        let hf_isomer_label = genesis.hv("isomer::Hf178m2", PHYSICS_DIM);
        let hf_vector = hf_base.bind(&hf_isomer_label);
        isomers.push(NuclearIsomer {
            atomic_number: 72,
            mass_number: 178,
            excitation_kev: 2446.0,
            half_life_s: Some(31.0 * 365.25 * 24.0 * 3600.0), // 31 years
            state: "m2".to_string(),
            vector: hf_vector,
        });

        // Technetium-99m: Medical imaging, 6-hour half-life
        // 140 keV gamma emission
        // Tc: Z=43, A=99, N=56
        let tc_base = table.isotope(43, 56, hadrons);
        let tc_isomer_label = genesis.hv("isomer::Tc99m", PHYSICS_DIM);
        let tc_vector = tc_base.bind(&tc_isomer_label);
        isomers.push(NuclearIsomer {
            atomic_number: 43,
            mass_number: 99,
            excitation_kev: 140.5,
            half_life_s: Some(6.0 * 3600.0), // 6 hours
            state: "m".to_string(),
            vector: tc_vector,
        });

        isomers
    }

    /// Get binding energy per nucleon for a given mass number
    pub fn binding_energy_per_nucleon(&self, mass_number: u16) -> f32 {
        // Interpolate from binding curve
        if mass_number < 2 {
            return 0.0;
        }

        // Find bracketing points
        let mut lower = &self.binding_curve[0];
        let mut upper = &self.binding_curve[self.binding_curve.len() - 1];

        for point in &self.binding_curve {
            if point.mass_number <= mass_number {
                lower = point;
            }
            if point.mass_number >= mass_number && point.mass_number < upper.mass_number {
                upper = point;
            }
        }

        if lower.mass_number == upper.mass_number {
            return lower.binding_per_nucleon_mev;
        }

        // Linear interpolation
        let t = (mass_number - lower.mass_number) as f32
            / (upper.mass_number - lower.mass_number) as f32;
        lower.binding_per_nucleon_mev
            + t * (upper.binding_per_nucleon_mev - lower.binding_per_nucleon_mev)
    }

    /// Calculate Q-value for fusion reaction
    ///
    /// Q = (m_reactants - m_products) * c²
    /// Positive Q = exothermic (releases energy)
    pub fn fusion_q_value(&self, a1: u16, a2: u16, product_a: u16) -> f32 {
        // Approximate using binding energy difference
        let be1 = self.binding_energy_per_nucleon(a1) * a1 as f32;
        let be2 = self.binding_energy_per_nucleon(a2) * a2 as f32;
        let be_product = self.binding_energy_per_nucleon(product_a) * product_a as f32;

        // Q ≈ BE_product - (BE1 + BE2)
        be_product - (be1 + be2)
    }

    /// Calculate Q-value for fission reaction
    pub fn fission_q_value(&self, parent_a: u16, fragment1_a: u16, fragment2_a: u16) -> f32 {
        let be_parent = self.binding_energy_per_nucleon(parent_a) * parent_a as f32;
        let be_f1 = self.binding_energy_per_nucleon(fragment1_a) * fragment1_a as f32;
        let be_f2 = self.binding_energy_per_nucleon(fragment2_a) * fragment2_a as f32;

        // Q ≈ (BE_f1 + BE_f2) - BE_parent
        (be_f1 + be_f2) - be_parent
    }

    /// Encode energy at a given scale
    pub fn encode_energy(&self, scale: EnergyScale, amount_ev: f64) -> ContinuousHV {
        let base_vector = match scale {
            EnergyScale::Chemical => &self.chemical_energy,
            EnergyScale::Isomeric => &self.isomeric_energy,
            EnergyScale::Nuclear => &self.nuclear_energy,
            EnergyScale::Relativistic => &self.relativistic_energy,
        };

        // Log-scale normalization
        let log_amount = (amount_ev.max(1.0)).ln();
        let normalized = (log_amount / 25.0) as f32; // ~e^25 ≈ 10^10 eV

        base_vector.scale(normalized.clamp(0.01, 1.0))
    }

    /// Compare energy storage: chemical vs nuclear vs isomeric
    pub fn energy_density_comparison(&self) -> ContinuousHV {
        // Bundle all energy scales with weights proportional to log(density)
        let weights = [
            1.0, // Chemical: 1 eV baseline
            5.0, // Isomeric: ~100 keV
            6.0, // Nuclear: ~1 MeV
            9.0, // Relativistic: ~1 GeV
        ];

        ContinuousHV::weighted_bundle(
            &[
                &self.chemical_energy,
                &self.isomeric_energy,
                &self.nuclear_energy,
                &self.relativistic_energy,
            ],
            &weights,
        )
    }

    /// Get isomer by element and mass number
    pub fn get_isomer(&self, atomic_number: u8, mass_number: u16) -> Option<&NuclearIsomer> {
        self.isomers
            .iter()
            .find(|i| i.atomic_number == atomic_number && i.mass_number == mass_number)
    }

    /// Encode a nuclear reaction
    pub fn encode_reaction(
        &self,
        reaction_type: NuclearReaction,
        reactants: &[&ContinuousHV],
        products: &[&ContinuousHV],
    ) -> ContinuousHV {
        let reaction_vector = match reaction_type {
            NuclearReaction::Fusion => &self.fusion,
            NuclearReaction::Fission => &self.fission,
            NuclearReaction::AlphaDecay => &self.alpha_decay,
            NuclearReaction::BetaMinusDecay | NuclearReaction::BetaPlusDecay => &self.beta_decay,
            NuclearReaction::GammaDecay => &self.gamma_decay,
            NuclearReaction::NeutronCapture => &self.fusion, // Similar to fusion
        };

        let reactant_bundle = ContinuousHV::bundle(reactants);
        let product_bundle = ContinuousHV::bundle(products);

        // Reaction = Bind(reactants, products, reaction_type)
        reactant_bundle.bind(&product_bundle).bind(reaction_vector)
    }

    /// Is this nucleus stable? (Based on binding energy curve position)
    pub fn stability_vector(&self, mass_number: u16) -> ContinuousHV {
        let be = self.binding_energy_per_nucleon(mass_number);

        // Peak stability around Fe-56 (8.79 MeV/nucleon)
        let stability_score = be / 8.79;

        ContinuousHV::weighted_bundle(
            &[&self.stable, &self.radioactive],
            &[stability_score, 1.0 - stability_score],
        )
    }

    /// Get the peak of the binding curve (iron group)
    pub fn most_stable_mass_number(&self) -> u16 {
        self.binding_curve
            .iter()
            .max_by(|a, b| {
                a.binding_per_nucleon_mev
                    .partial_cmp(&b.binding_per_nucleon_mev)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.mass_number)
            .unwrap_or(56)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{Hadrons, PeriodicTable, StandardModel};

    fn setup() -> (PeriodicTable, Hadrons, NuclearPhysics, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("nuclear test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
        let nuclear = NuclearPhysics::from_genesis(&genesis, &hadrons, &table);
        (table, hadrons, nuclear, genesis)
    }

    #[test]
    fn test_nuclear_creation() {
        let (_, _, nuclear, _) = setup();

        assert_eq!(nuclear.fusion.dim(), PHYSICS_DIM);
        assert!(!nuclear.isomers.is_empty());
    }

    #[test]
    fn test_binding_energy_curve() {
        let (_, _, nuclear, _) = setup();

        // Iron should have maximum binding energy per nucleon
        let fe_56 = nuclear.binding_energy_per_nucleon(56);
        let h_2 = nuclear.binding_energy_per_nucleon(2);
        let u_238 = nuclear.binding_energy_per_nucleon(238);

        assert!(fe_56 > h_2, "Fe-56 should be more tightly bound than H-2");
        assert!(
            fe_56 > u_238,
            "Fe-56 should be more tightly bound than U-238"
        );
        assert!(fe_56 > 8.5, "Fe-56 should have ~8.79 MeV/nucleon");
    }

    #[test]
    fn test_fusion_q_value() {
        let (_, _, nuclear, _) = setup();

        // D-T fusion: 2 + 3 → 4 (He) + neutron
        // Should release energy (positive Q)
        let q = nuclear.fusion_q_value(2, 3, 4);
        assert!(q > 0.0, "D-T fusion should be exothermic: Q={}", q);
    }

    #[test]
    fn test_fission_q_value() {
        let (_, _, nuclear, _) = setup();

        // U-235 fission: 235 → ~140 + ~95
        // Should release energy (positive Q)
        let q = nuclear.fission_q_value(235, 140, 95);
        assert!(q > 0.0, "U-235 fission should be exothermic: Q={}", q);
    }

    #[test]
    fn test_energy_scale_ratios() {
        assert!(EnergyScale::Nuclear.density_ratio() > 1e5);
        assert!(EnergyScale::Isomeric.density_ratio() > 1e4);
        assert_eq!(EnergyScale::Chemical.density_ratio(), 1.0);
    }

    #[test]
    fn test_isomer_lookup() {
        let (_, _, nuclear, _) = setup();

        // Hafnium-178m2 should exist
        let hf_isomer = nuclear.get_isomer(72, 178);
        assert!(hf_isomer.is_some(), "Hf-178m2 should exist");

        let isomer = hf_isomer.unwrap();
        assert!(
            isomer.excitation_kev > 2000.0,
            "Hf-178m2 has ~2.4 MeV excitation"
        );
    }

    #[test]
    fn test_most_stable() {
        let (_, _, nuclear, _) = setup();

        let most_stable = nuclear.most_stable_mass_number();
        assert!(
            most_stable >= 56 && most_stable <= 62,
            "Most stable should be Fe/Ni region"
        );
    }

    #[test]
    fn test_deterministic() {
        let genesis = GenesisSeed::from_phrase("determinism");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, &genesis);

        let n1 = NuclearPhysics::from_genesis(&genesis, &hadrons, &table);
        let n2 = NuclearPhysics::from_genesis(&genesis, &hadrons, &table);

        assert!(
            n1.fusion.similarity(&n2.fusion) > 0.9999,
            "Nuclear physics should be deterministic"
        );
    }
}
