// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Radiation Damage: PKA Cascades and Self-Healing
//!
//! This module models radiation damage from fusion neutrons and
//! the self-healing mechanisms that can mitigate it.
//!
//! ## The "Wolverine" Problem
//!
//! Lattice Confinement Fusion produces 14.1 MeV neutrons (D-T) or 2.45 MeV (D-D).
//! Each neutron creates a Primary Knock-on Atom (PKA) that cascades through
//! the lattice, displacing ~1000 atoms per event.
//!
//! ## Key Metrics
//!
//! - **DPA**: Displacements Per Atom - total damage accumulated
//! - **PKA Energy**: Energy transferred to first struck atom
//! - **Cascade Volume**: Region affected by one PKA event
//! - **Healing Rate**: How fast defects recombine
//!
//! ## Self-Healing Mechanisms
//!
//! 1. **Vacancy-Interstitial Recombination**: Defects annihilate each other
//! 2. **Defect Sinks**: Grain boundaries absorb defects
//! 3. **Sluggish Diffusion**: HEAs trap defects in local minima
//! 4. **Dynamic Recrystallization**: Lattice rebuilds under stress

use super::phonon_dynamics::{CrystalStructure, LatticeMaterial};
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Radiation damage characteristics
#[derive(Debug, Clone)]
pub struct RadiationDamage {
    /// Neutron energy in MeV
    pub neutron_energy_mev: f64,
    /// Average PKA energy in keV
    pub pka_energy_kev: f64,
    /// Displacements per PKA cascade
    pub displacements_per_cascade: f64,
    /// DPA rate per second at given flux
    pub dpa_rate_per_second: f64,
    /// Time to critical embrittlement (seconds)
    pub time_to_failure_s: f64,
}

/// Self-healing mechanisms and their effectiveness
#[derive(Debug, Clone)]
pub struct SelfHealingMechanism {
    /// Mechanism name
    pub name: String,
    /// Healing rate (defects healed per second per DPA)
    pub healing_rate: f64,
    /// Temperature dependence (activation energy in eV)
    pub activation_energy_ev: f64,
    /// Effectiveness at 300K (0-1)
    pub effectiveness_300k: f32,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// Material self-healing analysis result
#[derive(Debug, Clone)]
pub struct HealingAnalysis {
    /// Material name
    pub material_name: String,
    /// Net healing rate (healing - damage)
    pub net_healing_rate: f64,
    /// Sustainable DPA level
    pub sustainable_dpa: f64,
    /// Lifetime extension factor vs pure metal
    pub lifetime_factor: f64,
    /// Primary healing mechanism
    pub primary_mechanism: String,
    /// Overall self-healing score (0-1)
    pub healing_score: f32,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// Radiation damage and healing system
#[derive(Debug, Clone)]
pub struct RadiationDamageSystem {
    /// Damage concept vectors
    pub pka_cascade: ContinuousHV,
    pub vacancy: ContinuousHV,
    pub interstitial: ContinuousHV,
    pub dislocation: ContinuousHV,
    pub void_formation: ContinuousHV,
    pub embrittlement: ContinuousHV,

    /// Healing concept vectors
    pub recombination: ContinuousHV,
    pub defect_sink: ContinuousHV,
    pub recrystallization: ContinuousHV,
    pub sluggish_diffusion: ContinuousHV,
    pub annealing: ContinuousHV,

    /// Material property vectors
    pub radiation_resistant: ContinuousHV,
    pub self_healing: ContinuousHV,
    pub high_entropy: ContinuousHV,

    /// Healing mechanisms database
    pub mechanisms: Vec<SelfHealingMechanism>,
}

impl RadiationDamageSystem {
    /// Create from genesis seed
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        // Damage vectors
        let pka_cascade = genesis.hv("damage::pka_cascade", PHYSICS_DIM);
        let vacancy = genesis.hv("damage::vacancy", PHYSICS_DIM);
        let interstitial = genesis.hv("damage::interstitial", PHYSICS_DIM);
        let dislocation = genesis.hv("damage::dislocation", PHYSICS_DIM);
        let void_formation = genesis.hv("damage::void", PHYSICS_DIM);
        let embrittlement = genesis.hv("damage::embrittlement", PHYSICS_DIM);

        // Healing vectors
        let recombination = genesis.hv("healing::recombination", PHYSICS_DIM);
        let defect_sink = genesis.hv("healing::defect_sink", PHYSICS_DIM);
        let recrystallization = genesis.hv("healing::recrystallization", PHYSICS_DIM);
        let sluggish_diffusion = genesis.hv("healing::sluggish_diffusion", PHYSICS_DIM);
        let annealing = genesis.hv("healing::annealing", PHYSICS_DIM);

        // Property vectors
        let radiation_resistant = genesis.hv("property::radiation_resistant", PHYSICS_DIM);
        let self_healing = genesis.hv("property::self_healing", PHYSICS_DIM);
        let high_entropy = genesis.hv("property::high_entropy", PHYSICS_DIM);

        let mut system = Self {
            pka_cascade,
            vacancy,
            interstitial,
            dislocation,
            void_formation,
            embrittlement,
            recombination,
            defect_sink,
            recrystallization,
            sluggish_diffusion,
            annealing,
            radiation_resistant,
            self_healing,
            high_entropy,
            mechanisms: Vec::new(),
        };

        system.init_mechanisms(genesis);
        system
    }

    /// Initialize healing mechanisms
    fn init_mechanisms(&mut self, genesis: &GenesisSeed) {
        let mechanism_data = vec![
            // (name, healing_rate, activation_energy_eV, effectiveness_300K)

            // Vacancy-Interstitial recombination
            ("V-I Recombination", 1e12, 0.1, 0.9),
            // Grain boundary sinks (nano-crystalline)
            ("GB Defect Sinks", 1e10, 0.3, 0.7),
            // HEA sluggish diffusion (traps defects)
            ("Sluggish Diffusion", 1e8, 0.5, 0.85),
            // Dynamic recrystallization
            ("Dynamic Recrystallization", 1e6, 0.8, 0.4),
            // Thermal annealing (requires high T)
            ("Thermal Annealing", 1e14, 1.5, 0.1),
            // Interstitial clustering (can be positive)
            ("Interstitial Clustering", 1e9, 0.2, 0.5),
            // Liquid metal self-healing (no lattice to break)
            ("Liquid Phase", 1e15, 0.0, 1.0),
        ];

        for (name, rate, activation, eff_300k) in mechanism_data {
            let vec = match name {
                "V-I Recombination" => self.recombination.clone(),
                "GB Defect Sinks" => self.defect_sink.clone(),
                "Sluggish Diffusion" => self.sluggish_diffusion.clone(),
                "Dynamic Recrystallization" => self.recrystallization.clone(),
                "Thermal Annealing" => self.annealing.clone(),
                _ => genesis.hv(&format!("healing::{name}"), PHYSICS_DIM),
            };

            self.mechanisms.push(SelfHealingMechanism {
                name: name.to_string(),
                healing_rate: rate,
                activation_energy_ev: activation,
                effectiveness_300k: eff_300k,
                vector: vec,
            });
        }
    }

    /// Calculate radiation damage for a given neutron flux
    pub fn calculate_damage(
        &self,
        neutron_energy_mev: f64,
        flux_n_cm2_s: f64,
        material_z: u8,
    ) -> RadiationDamage {
        // Lindhard model for PKA energy
        // T_avg ≈ 2 * E_n * A_n * A_t / (A_n + A_t)^2
        // For neutron (A_n = 1) on material (A_t ≈ 2*Z for stability)
        let a_target = (material_z as f64) * 2.0;
        let pka_energy_mev = 2.0 * neutron_energy_mev * a_target / (1.0 + a_target).powi(2);
        let pka_energy_kev = pka_energy_mev * 1000.0;

        // Kinchin-Pease model for displacements
        // N_d = 0.8 * T_dam / (2 * E_d)
        // E_d ≈ 25 eV for most metals
        let e_d_ev = 25.0;
        let t_dam_ev = pka_energy_kev * 1000.0 * 0.8; // ~80% goes to damage
        let displacements_per_cascade = (0.8 * t_dam_ev / (2.0 * e_d_ev)).max(1.0);

        // DPA rate
        // DPA/s = flux * cross_section * displacements / N_atoms
        // Simplified: assume effective cross-section ~1 barn = 1e-24 cm^2
        let cross_section = 1e-24;
        let dpa_rate = flux_n_cm2_s * cross_section * displacements_per_cascade;

        // Time to failure (critical DPA ~10-100 for most metals)
        let critical_dpa = match material_z {
            22 => 50.0,  // Titanium
            26 => 100.0, // Iron (ferritic steels better)
            28 => 30.0,  // Nickel
            46 => 20.0,  // Palladium (poor)
            68 => 25.0,  // Erbium
            _ => 50.0,   // Default
        };

        let time_to_failure = critical_dpa / dpa_rate.max(1e-30);

        RadiationDamage {
            neutron_energy_mev,
            pka_energy_kev,
            displacements_per_cascade,
            dpa_rate_per_second: dpa_rate,
            time_to_failure_s: time_to_failure,
        }
    }

    /// Analyze self-healing capability of a material
    pub fn analyze_healing(
        &self,
        material: &LatticeMaterial,
        damage: &RadiationDamage,
        is_hea: bool,
        is_liquid: bool,
        grain_size_nm: f64,
    ) -> HealingAnalysis {
        let mut total_healing_factor = 0.0;
        let mut primary_mechanism = "None".to_string();
        let mut max_contribution = 0.0;

        // Base material healing capability (0-1 scale)
        let base_healing = self.quick_healing_estimate(material.structure, material.primary_z);

        for mechanism in &self.mechanisms {
            // Use relative contributions, not absolute rates
            let mut contribution = mechanism.effectiveness_300k as f64;

            // Adjust for material type
            match mechanism.name.as_str() {
                "Sluggish Diffusion" => {
                    if is_hea {
                        contribution *= 3.0; // HEAs excel here
                    } else {
                        contribution *= 0.1; // Pure metals barely have this
                    }
                }
                "GB Defect Sinks" => {
                    // Smaller grains = more sinks
                    contribution *= (100.0 / grain_size_nm.max(1.0)).min(5.0) / 5.0;
                }
                "Liquid Phase" => {
                    if is_liquid {
                        contribution *= 10.0; // Dominant for liquids
                    } else {
                        contribution = 0.0;
                    }
                }
                "V-I Recombination" => {
                    // Universal mechanism - scales with base healing
                    contribution *= base_healing as f64;
                }
                "Thermal Annealing" => {
                    // Only effective at high temperature, limited at 300K
                    if is_hea {
                        contribution *= 2.0; // HEAs anneal better
                    }
                }
                _ => {
                    contribution *= base_healing as f64;
                }
            }

            if contribution > max_contribution {
                max_contribution = contribution;
                primary_mechanism = mechanism.name.clone();
            }

            total_healing_factor += contribution;
        }

        // Normalize healing factor (typical range 0-10 without liquid/HEA)
        let normalized_healing = (total_healing_factor / 10.0).min(1.5);

        // Damage severity factor based on neutron energy
        let damage_severity = (damage.neutron_energy_mev / 20.0).min(1.0);

        // Net healing capability
        let net_factor = normalized_healing - damage_severity * 0.5;

        // Sustainable DPA based on healing vs damage balance
        let sustainable_dpa = if is_liquid || (is_hea && net_factor > 0.5) {
            1000.0
        } else if is_hea {
            100.0 * net_factor.max(0.1)
        } else {
            10.0 * net_factor.max(0.05)
        };

        // Lifetime extension
        let lifetime_factor = if is_liquid {
            1e6 // Liquid metals don't embrittle
        } else if is_hea {
            10.0 + sustainable_dpa / 10.0
        } else {
            1.0 + sustainable_dpa / 50.0
        };

        // Overall healing score - key differentiator
        let healing_score = if is_liquid {
            1.0
        } else if is_hea && normalized_healing > 0.8 {
            0.95 // Excellent HEA
        } else {
            (sustainable_dpa / 100.0).min(0.9) as f32
        };

        // Build vector
        let healing_vec = if is_liquid {
            ContinuousHV::weighted_bundle(
                &[&self.self_healing, &self.mechanisms[6].vector],
                &[1.0, 2.0],
            )
        } else if is_hea {
            ContinuousHV::weighted_bundle(
                &[
                    &self.self_healing,
                    &self.high_entropy,
                    &self.sluggish_diffusion,
                ],
                &[1.0, 1.5, 1.0],
            )
        } else {
            ContinuousHV::weighted_bundle(
                &[&material.vector, &self.radiation_resistant],
                &[1.0, healing_score],
            )
        };

        HealingAnalysis {
            material_name: material.name.clone(),
            net_healing_rate: net_factor,
            sustainable_dpa,
            lifetime_factor,
            primary_mechanism,
            healing_score,
            vector: healing_vec,
        }
    }

    /// Quick estimate of healing potential for a material
    pub fn quick_healing_estimate(&self, structure: CrystalStructure, z: u8) -> f32 {
        // Base score from structure
        let structure_score: f32 = match structure {
            CrystalStructure::BCC => 0.7,     // Good radiation resistance
            CrystalStructure::FCC => 0.5,     // Moderate
            CrystalStructure::HCP => 0.4,     // Poor
            CrystalStructure::Diamond => 0.3, // Very poor
            CrystalStructure::Fluorite => 0.6,
            CrystalStructure::Rocksalt => 0.5,
        };

        // Adjust for element
        let element_score: f32 = match z {
            22 => 0.7,  // Titanium - good
            23 => 0.8,  // Vanadium - excellent
            24 => 0.75, // Chromium - good
            26 => 0.7,  // Iron - good (especially ferritic)
            27 => 0.65, // Cobalt
            28 => 0.5,  // Nickel - moderate
            40 => 0.75, // Zirconium - good
            41 => 0.8,  // Niobium - excellent
            42 => 0.7,  // Molybdenum - good
            46 => 0.3,  // Palladium - poor
            68 => 0.35, // Erbium - poor
            73 => 0.85, // Tantalum - excellent
            74 => 0.8,  // Tungsten - excellent
            _ => 0.5,
        };

        (structure_score * element_score).min(1.0_f32)
    }
}

/// Neutron energy from fusion reactions
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FusionReaction {
    /// D + D → He-3 + n (2.45 MeV neutron)
    DD,
    /// D + T → He-4 + n (14.1 MeV neutron)
    DT,
    /// D + D → T + p (no neutron)
    DdProton,
    /// D + He-3 → He-4 + p (no neutron, aneutronic)
    DHe3,
}

impl FusionReaction {
    /// Neutron energy in MeV (None for aneutronic)
    pub fn neutron_energy_mev(&self) -> Option<f64> {
        match self {
            FusionReaction::DD => Some(2.45),
            FusionReaction::DT => Some(14.1),
            FusionReaction::DdProton => None,
            FusionReaction::DHe3 => None,
        }
    }

    /// Total energy release in MeV
    pub fn total_energy_mev(&self) -> f64 {
        match self {
            FusionReaction::DD => 3.27, // avg of both branches
            FusionReaction::DT => 17.6,
            FusionReaction::DdProton => 4.03,
            FusionReaction::DHe3 => 18.3,
        }
    }

    /// Cross-section peak temperature in keV
    pub fn optimal_temp_kev(&self) -> f64 {
        match self {
            FusionReaction::DD => 15.0,
            FusionReaction::DT => 60.0, // Peak cross-section temperature
            FusionReaction::DdProton => 15.0,
            FusionReaction::DHe3 => 58.0, // Highest - hardest to achieve
        }
    }

    /// Fraction of reactions that produce neutrons (branching ratio)
    /// D-D has two branches: ~50% He-3+n, ~50% T+p
    /// D-T is 100% neutron-producing
    pub fn neutron_yield_fraction(&self) -> f64 {
        match self {
            FusionReaction::DD => 0.5,       // Only He-3+n branch produces neutrons
            FusionReaction::DT => 1.0,       // 100% neutron yield
            FusionReaction::DdProton => 0.0, // Proton branch - no neutrons
            FusionReaction::DHe3 => 0.0,     // Aneutronic
        }
    }

    /// Self-shielding factor: fraction of neutrons that escape the core
    /// Accounts for moderation and absorption within the fusion zone
    /// Based on typical compact fusion device geometry
    pub fn self_shielding_factor(&self, core_radius_m: f64) -> f64 {
        // Mean free path in fusion plasma/liquid metal core
        // Galinstan has high density (~6400 kg/m³) providing significant moderation
        let mfp = match self {
            FusionReaction::DD => 0.15, // 2.45 MeV neutrons, shorter MFP
            FusionReaction::DT => 0.25, // 14.1 MeV neutrons, longer MFP
            _ => 1.0,                   // No neutrons
        };

        // Escape probability: exp(-R/λ) for spherical geometry
        // This is simplified - real transport is more complex
        let escape_prob = (-core_radius_m / mfp).exp();

        // Also account for isotropic emission (only ~50% directed outward)
        // and solid angle effects
        let geometric_factor = 0.5;

        (escape_prob * geometric_factor).clamp(0.001, 1.0)
    }

    /// Total fuel mass in g/mol (sum of reactant atomic masses)
    #[allow(dead_code)]
    pub fn fuel_mass_g_mol(&self) -> f64 {
        let d = 2.01410; // Deuterium
        let t = 3.01605; // Tritium
        let he3 = 3.01603; // Helium-3
        match self {
            FusionReaction::DD => 2.0 * d,
            FusionReaction::DT => d + t,
            FusionReaction::DdProton => 2.0 * d,
            FusionReaction::DHe3 => d + he3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{Hadrons, PeriodicTable, PhononDynamics, StandardModel};

    fn setup() -> (GenesisSeed, RadiationDamageSystem, PhononDynamics) {
        let genesis = GenesisSeed::from_phrase("wolverine test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
        let phonons = PhononDynamics::from_genesis(&genesis, &table);
        let radiation = RadiationDamageSystem::from_genesis(&genesis);
        (genesis, radiation, phonons)
    }

    #[test]
    fn test_damage_calculation() {
        let (_, radiation, _) = setup();

        // D-T fusion neutron on Titanium
        let damage = radiation.calculate_damage(14.1, 1e14, 22);

        assert!(
            damage.pka_energy_kev > 100.0,
            "PKA energy should be > 100 keV"
        );
        assert!(
            damage.displacements_per_cascade > 100.0,
            "Should create >100 displacements"
        );
        assert!(
            damage.time_to_failure_s > 0.0,
            "Time to failure should be positive"
        );
    }

    #[test]
    fn test_palladium_embrittlement() {
        let (_, radiation, phonons) = setup();

        let pd = phonons
            .materials
            .iter()
            .find(|m| m.name == "Palladium")
            .expect("Palladium should exist");

        let damage = radiation.calculate_damage(14.1, 1e14, 46);
        let healing = radiation.analyze_healing(pd, &damage, false, false, 100.0);

        // Palladium should have poor healing
        assert!(
            healing.healing_score < 0.5,
            "Palladium should have poor self-healing: {}",
            healing.healing_score
        );
    }

    #[test]
    fn test_hea_improvement() {
        let (_, radiation, phonons) = setup();

        let ti = phonons
            .materials
            .iter()
            .find(|m| m.name == "Titanium")
            .expect("Titanium should exist");

        let damage = radiation.calculate_damage(14.1, 1e14, 22);

        // Without HEA
        let healing_pure = radiation.analyze_healing(ti, &damage, false, false, 100.0);

        // With HEA
        let healing_hea = radiation.analyze_healing(ti, &damage, true, false, 100.0);

        assert!(
            healing_hea.lifetime_factor > healing_pure.lifetime_factor,
            "HEA should improve lifetime"
        );
        assert!(
            healing_hea.healing_score > healing_pure.healing_score,
            "HEA should improve healing score"
        );
    }

    #[test]
    fn test_liquid_metal_healing() {
        let (_, radiation, phonons) = setup();

        let pd = phonons
            .materials
            .iter()
            .find(|m| m.name == "Palladium")
            .expect("Palladium should exist");

        let damage = radiation.calculate_damage(14.1, 1e14, 46);
        let healing = radiation.analyze_healing(pd, &damage, false, true, 100.0);

        // Liquid metal should have perfect healing
        assert_eq!(
            healing.healing_score, 1.0,
            "Liquid metal should have perfect healing"
        );
        assert_eq!(healing.primary_mechanism, "Liquid Phase");
    }

    #[test]
    fn test_fusion_reactions() {
        assert_eq!(FusionReaction::DT.neutron_energy_mev(), Some(14.1));
        assert_eq!(FusionReaction::DD.neutron_energy_mev(), Some(2.45));
        assert_eq!(FusionReaction::DHe3.neutron_energy_mev(), None);
    }
}
