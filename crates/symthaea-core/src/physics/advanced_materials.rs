// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Advanced Materials: MAX Phases and Nano-Laminates
//!
//! Beyond High-Entropy Alloys, these material classes offer alternative
//! self-healing mechanisms for extreme radiation environments.
//!
//! ## MAX Phases (Mn+1AXn)
//!
//! Layered ternary carbides/nitrides with unique properties:
//! - **M**: Early transition metal (Ti, V, Cr, Nb, Ta, Zr, Hf)
//! - **A**: A-group element (Al, Si, Ge, Ga, S, Sn)
//! - **X**: Carbon or Nitrogen
//!
//! Self-healing via:
//! - Kink band formation absorbs deformation
//! - Delamination along weak M-A bonds dissipates energy
//! - Re-bonding during thermal cycling
//!
//! ## Nano-Laminates
//!
//! Alternating layers of immiscible metals:
//! - Cu/Nb, Cu/V, Fe/Cu, Ag/V
//! - Interfaces act as defect sinks
//! - Point defects recombine at boundaries
//! - Layer spacing controls healing rate
//!
//! ## Application to LCF
//!
//! These materials can serve as:
//! 1. Outer containment (MAX phase shell)
//! 2. Interface layers between HEA and liquid core
//! 3. Neutron moderating structures

use super::radiation_damage::RadiationDamage;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;

/// MAX Phase compound (Mn+1AXn)
#[derive(Debug, Clone)]
pub struct MAXPhase {
    /// Compound name (e.g., "Ti3SiC2")
    pub name: String,
    /// M element (transition metal)
    pub m_element: u8,
    /// A element (A-group)
    pub a_element: u8,
    /// X element (C or N)
    pub x_element: u8,
    /// n value (1, 2, or 3 for 211, 312, 413 phases)
    pub n_value: u8,
    /// Kink band formation energy (eV)
    pub kink_energy_ev: f64,
    /// Delamination energy (J/m²)
    pub delam_energy_j_m2: f64,
    /// Maximum operating temperature (K)
    pub max_temp_k: f64,
    /// Thermal conductivity (W/m·K)
    pub thermal_conductivity: f64,
    /// Self-healing score (0-1)
    pub healing_score: f32,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// Nano-laminate composite
#[derive(Debug, Clone)]
pub struct NanoLaminate {
    /// Composite name (e.g., "Cu/Nb")
    pub name: String,
    /// Layer A element
    pub layer_a: u8,
    /// Layer B element
    pub layer_b: u8,
    /// Layer spacing (nm)
    pub spacing_nm: f64,
    /// Interface energy (J/m²)
    pub interface_energy: f64,
    /// Defect sink efficiency (0-1)
    pub sink_efficiency: f64,
    /// Radiation tolerance (DPA threshold)
    pub dpa_tolerance: f64,
    /// Self-healing score (0-1)
    pub healing_score: f32,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// Advanced Materials Database
#[derive(Debug, Clone)]
pub struct AdvancedMaterials {
    /// Concept vectors
    pub max_phase: ContinuousHV,
    pub kink_band: ContinuousHV,
    pub delamination: ContinuousHV,
    pub nano_laminate: ContinuousHV,
    pub interface_sink: ContinuousHV,
    pub thermal_healing: ContinuousHV,

    /// MAX phases database
    pub max_phases: Vec<MAXPhase>,
    /// Nano-laminates database
    pub nano_laminates: Vec<NanoLaminate>,
}

impl AdvancedMaterials {
    /// Create from genesis seed
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        let max_phase = genesis.hv("material::max_phase", PHYSICS_DIM);
        let kink_band = genesis.hv("material::kink_band", PHYSICS_DIM);
        let delamination = genesis.hv("material::delamination", PHYSICS_DIM);
        let nano_laminate = genesis.hv("material::nano_laminate", PHYSICS_DIM);
        let interface_sink = genesis.hv("material::interface_sink", PHYSICS_DIM);
        let thermal_healing = genesis.hv("material::thermal_healing", PHYSICS_DIM);

        let mut system = Self {
            max_phase,
            kink_band,
            delamination,
            nano_laminate,
            interface_sink,
            thermal_healing,
            max_phases: Vec::new(),
            nano_laminates: Vec::new(),
        };

        system.init_max_phases(genesis);
        system.init_nano_laminates(genesis);
        system
    }

    /// Initialize MAX phases database
    fn init_max_phases(&mut self, _genesis: &GenesisSeed) {
        let max_data = vec![
            // (name, M, A, X, n, kink_eV, delam_J/m2, max_T_K, thermal_cond)

            // Ti-based (most studied)
            ("Ti3SiC2", 22, 14, 6, 2, 0.15, 8.5, 1700.0, 40.0), // Best characterized
            ("Ti3AlC2", 22, 13, 6, 2, 0.12, 7.2, 1600.0, 35.0), // Good oxidation resistance
            ("Ti2AlC", 22, 13, 6, 1, 0.18, 9.0, 1500.0, 30.0),  // 211 phase
            ("Ti2AlN", 22, 13, 7, 1, 0.20, 10.0, 1400.0, 28.0), // Nitride variant
            // Cr-based (oxidation resistant)
            ("Cr2AlC", 24, 13, 6, 1, 0.14, 7.8, 1200.0, 25.0), // Excellent oxidation
            ("Cr2GeC", 24, 32, 6, 1, 0.16, 8.2, 1100.0, 22.0),
            // V-based (radiation resistant)
            ("V2AlC", 23, 13, 6, 1, 0.13, 7.0, 1300.0, 28.0),
            // Nb-based (high temperature)
            ("Nb2AlC", 41, 13, 6, 1, 0.19, 9.5, 1800.0, 32.0),
            // Ta-based (refractory)
            ("Ta2AlC", 73, 13, 6, 1, 0.22, 11.0, 2000.0, 35.0),
            // Zr-based (neutron transparent)
            ("Zr2AlC", 40, 13, 6, 1, 0.17, 8.8, 1600.0, 30.0),
            // Hf-based
            ("Hf2AlC", 72, 13, 6, 1, 0.21, 10.5, 1900.0, 33.0),
        ];

        for (name, m, a, x, n, kink, delam, max_t, thermal) in max_data {
            // Healing score based on kink band and delamination properties
            let kink_factor = (0.25_f64 - kink).max(0.0) / 0.25; // Lower kink energy = better
            let delam_factor = delam / 12.0; // Higher delam energy = more energy absorbed
            let thermal_factor = (thermal / 50.0_f64).min(1.0); // Thermal healing

            let healing_score =
                (kink_factor * 0.4 + delam_factor * 0.4 + thermal_factor * 0.2) as f32;

            let vector = ContinuousHV::weighted_bundle(
                &[
                    &self.max_phase,
                    &self.kink_band,
                    &self.delamination,
                    &self.thermal_healing,
                ],
                &[
                    1.0,
                    kink_factor as f32,
                    delam_factor as f32,
                    thermal_factor as f32,
                ],
            );

            self.max_phases.push(MAXPhase {
                name: name.to_string(),
                m_element: m,
                a_element: a,
                x_element: x,
                n_value: n,
                kink_energy_ev: kink,
                delam_energy_j_m2: delam,
                max_temp_k: max_t,
                thermal_conductivity: thermal,
                healing_score,
                vector,
            });
        }
    }

    /// Initialize nano-laminates database
    fn init_nano_laminates(&mut self, _genesis: &GenesisSeed) {
        let laminate_data = vec![
            // (name, layer_a, layer_b, spacing_nm, interface_J/m2, sink_eff, dpa_tol)

            // Cu-based (FCC/BCC interfaces)
            ("Cu/Nb", 29, 41, 2.5, 0.85, 0.92, 150.0), // Most studied, excellent
            ("Cu/V", 29, 23, 3.0, 0.78, 0.88, 120.0),
            ("Cu/Mo", 29, 42, 2.8, 0.80, 0.85, 100.0),
            ("Cu/W", 29, 74, 2.2, 0.90, 0.90, 180.0), // High radiation resistance
            // Fe-based
            ("Fe/Cu", 26, 29, 4.0, 0.65, 0.75, 80.0),
            ("Fe/V", 26, 23, 3.5, 0.70, 0.80, 90.0),
            // Ag-based
            ("Ag/V", 47, 23, 3.2, 0.72, 0.82, 100.0),
            ("Ag/Ni", 47, 28, 3.8, 0.68, 0.78, 85.0),
            // Refractory
            ("Zr/Nb", 40, 41, 2.0, 0.95, 0.95, 200.0), // Excellent for nuclear
            ("Ti/Nb", 22, 41, 2.3, 0.88, 0.90, 160.0),
            // For H-storage (Pd interface)
            ("Pd/Nb", 46, 41, 2.5, 0.82, 0.85, 110.0), // H-permeable + rad resistant
        ];

        for (name, a, b, spacing, interface, sink, dpa) in laminate_data {
            // Healing score based on sink efficiency and DPA tolerance
            let sink_factor = sink;
            let dpa_factor = (dpa / 200.0_f64).min(1.0);
            let spacing_factor = (5.0_f64 - spacing).max(0.0) / 5.0; // Smaller spacing = better

            let healing_score =
                (sink_factor * 0.5 + dpa_factor * 0.3 + spacing_factor * 0.2) as f32;

            let vector = ContinuousHV::weighted_bundle(
                &[&self.nano_laminate, &self.interface_sink],
                &[1.0, healing_score],
            );

            self.nano_laminates.push(NanoLaminate {
                name: name.to_string(),
                layer_a: a,
                layer_b: b,
                spacing_nm: spacing,
                interface_energy: interface,
                sink_efficiency: sink,
                dpa_tolerance: dpa,
                healing_score,
                vector,
            });
        }
    }

    /// Analyze MAX phase for LCF application
    pub fn analyze_max_phase(&self, phase: &MAXPhase, damage: &RadiationDamage) -> MAXAnalysis {
        // Kink band absorption capacity
        let kink_absorption = damage.pka_energy_kev * 1000.0 / (phase.kink_energy_ev * 1e6);
        let normalized_absorption = (kink_absorption / 100.0).min(1.0);

        // Thermal stability under neutron flux
        let thermal_stability = if damage.neutron_energy_mev > 10.0 {
            (phase.max_temp_k / 2000.0).min(0.9) // High energy needs high temp tolerance
        } else {
            (phase.max_temp_k / 1500.0).min(1.0)
        };

        // Overall LCF suitability
        let lcf_score = (phase.healing_score as f64 * 0.4
            + normalized_absorption * 0.3
            + thermal_stability * 0.3) as f32;

        MAXAnalysis {
            phase_name: phase.name.clone(),
            kink_absorption: normalized_absorption as f32,
            thermal_stability: thermal_stability as f32,
            lcf_score,
            recommended_role: if lcf_score > 0.7 {
                "Primary containment shell".to_string()
            } else if lcf_score > 0.5 {
                "Interface layer".to_string()
            } else {
                "Structural support".to_string()
            },
        }
    }

    /// Analyze nano-laminate for LCF application
    pub fn analyze_nano_laminate(
        &self,
        laminate: &NanoLaminate,
        damage: &RadiationDamage,
    ) -> LaminateAnalysis {
        // Can it handle the DPA rate?
        let sustainable = laminate.dpa_tolerance / (damage.dpa_rate_per_second * 3.15e7); // years
        let sustainability_score = (sustainable / 20.0).min(1.0); // 20 years = perfect

        // Defect absorption rate
        let absorption_rate = laminate.sink_efficiency * (5.0 / laminate.spacing_nm);
        let normalized_absorption = (absorption_rate / 2.0).min(1.0);

        // H-compatibility (for Pd-containing laminates)
        let h_compatible = laminate.layer_a == 46
            || laminate.layer_b == 46
            || laminate.layer_a == 22
            || laminate.layer_b == 22
            || laminate.layer_a == 40
            || laminate.layer_b == 40;

        let lcf_score = (laminate.healing_score as f64 * 0.35
            + sustainability_score * 0.35
            + normalized_absorption * 0.2
            + if h_compatible { 0.1 } else { 0.0 }) as f32;

        LaminateAnalysis {
            laminate_name: laminate.name.clone(),
            sustainable_years: sustainable as f32,
            absorption_rate: normalized_absorption as f32,
            h_compatible,
            lcf_score,
            recommended_role: if h_compatible && lcf_score > 0.7 {
                "HEA/liquid interface".to_string()
            } else if lcf_score > 0.6 {
                "Neutron shield layer".to_string()
            } else {
                "Structural reinforcement".to_string()
            },
        }
    }

    /// Find best MAX phase for given conditions
    pub fn best_max_phase(&self, damage: &RadiationDamage) -> Option<(&MAXPhase, MAXAnalysis)> {
        self.max_phases
            .iter()
            .map(|p| (p, self.analyze_max_phase(p, damage)))
            .max_by(|a, b| {
                a.1.lcf_score
                    .partial_cmp(&b.1.lcf_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Find best nano-laminate for given conditions
    pub fn best_nano_laminate(
        &self,
        damage: &RadiationDamage,
    ) -> Option<(&NanoLaminate, LaminateAnalysis)> {
        self.nano_laminates
            .iter()
            .map(|l| (l, self.analyze_nano_laminate(l, damage)))
            .max_by(|a, b| {
                a.1.lcf_score
                    .partial_cmp(&b.1.lcf_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Get all materials sorted by LCF score
    pub fn rank_all_materials(&self, damage: &RadiationDamage) -> Vec<MaterialRanking> {
        let mut rankings = Vec::new();

        for phase in &self.max_phases {
            let analysis = self.analyze_max_phase(phase, damage);
            rankings.push(MaterialRanking {
                name: phase.name.clone(),
                material_type: MaterialType::MAXPhase,
                healing_score: phase.healing_score,
                lcf_score: analysis.lcf_score,
                recommended_role: analysis.recommended_role,
            });
        }

        for laminate in &self.nano_laminates {
            let analysis = self.analyze_nano_laminate(laminate, damage);
            rankings.push(MaterialRanking {
                name: laminate.name.clone(),
                material_type: MaterialType::NanoLaminate,
                healing_score: laminate.healing_score,
                lcf_score: analysis.lcf_score,
                recommended_role: analysis.recommended_role,
            });
        }

        rankings.sort_by(|a, b| {
            b.lcf_score
                .partial_cmp(&a.lcf_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rankings
    }
}

/// MAX phase analysis result
#[derive(Debug, Clone)]
pub struct MAXAnalysis {
    pub phase_name: String,
    pub kink_absorption: f32,
    pub thermal_stability: f32,
    pub lcf_score: f32,
    pub recommended_role: String,
}

/// Nano-laminate analysis result
#[derive(Debug, Clone)]
pub struct LaminateAnalysis {
    pub laminate_name: String,
    pub sustainable_years: f32,
    pub absorption_rate: f32,
    pub h_compatible: bool,
    pub lcf_score: f32,
    pub recommended_role: String,
}

/// Material type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialType {
    MAXPhase,
    NanoLaminate,
}

/// Unified material ranking
#[derive(Debug, Clone)]
pub struct MaterialRanking {
    pub name: String,
    pub material_type: MaterialType,
    pub healing_score: f32,
    pub lcf_score: f32,
    pub recommended_role: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::RadiationDamageSystem;

    fn setup() -> (GenesisSeed, AdvancedMaterials, RadiationDamageSystem) {
        let genesis = GenesisSeed::from_phrase("advanced materials test");
        let materials = AdvancedMaterials::from_genesis(&genesis);
        let radiation = RadiationDamageSystem::from_genesis(&genesis);
        (genesis, materials, radiation)
    }

    #[test]
    fn test_max_phases_created() {
        let (_, materials, _) = setup();
        assert!(
            materials.max_phases.len() >= 10,
            "Should have at least 10 MAX phases"
        );
    }

    #[test]
    fn test_nano_laminates_created() {
        let (_, materials, _) = setup();
        assert!(
            materials.nano_laminates.len() >= 10,
            "Should have at least 10 nano-laminates"
        );
    }

    #[test]
    fn test_ti3sic2_properties() {
        let (_, materials, _) = setup();

        let ti3sic2 = materials
            .max_phases
            .iter()
            .find(|p| p.name == "Ti3SiC2")
            .expect("Ti3SiC2 should exist");

        assert_eq!(ti3sic2.m_element, 22); // Ti
        assert_eq!(ti3sic2.a_element, 14); // Si
        assert_eq!(ti3sic2.x_element, 6); // C
        assert!(
            ti3sic2.healing_score > 0.5,
            "Ti3SiC2 should have good healing"
        );
    }

    #[test]
    fn test_cu_nb_properties() {
        let (_, materials, _) = setup();

        let cu_nb = materials
            .nano_laminates
            .iter()
            .find(|l| l.name == "Cu/Nb")
            .expect("Cu/Nb should exist");

        assert!(
            cu_nb.dpa_tolerance > 100.0,
            "Cu/Nb should have high DPA tolerance"
        );
        assert!(
            cu_nb.healing_score > 0.7,
            "Cu/Nb should have excellent healing"
        );
    }

    #[test]
    fn test_max_phase_analysis() {
        let (_, materials, radiation) = setup();

        let damage = radiation.calculate_damage(14.1, 1e14, 22);
        let (_best, analysis) = materials
            .best_max_phase(&damage)
            .expect("Should find best MAX phase");

        assert!(
            analysis.lcf_score > 0.4,
            "Best MAX phase should have decent LCF score"
        );
    }

    #[test]
    fn test_nano_laminate_analysis() {
        let (_, materials, radiation) = setup();

        let damage = radiation.calculate_damage(14.1, 1e14, 22);
        let (_best, analysis) = materials
            .best_nano_laminate(&damage)
            .expect("Should find best nano-laminate");

        assert!(
            analysis.lcf_score > 0.5,
            "Best nano-laminate should have good LCF score"
        );
    }

    #[test]
    fn test_material_ranking() {
        let (_, materials, radiation) = setup();

        let damage = radiation.calculate_damage(14.1, 1e14, 22);
        let rankings = materials.rank_all_materials(&damage);

        assert!(!rankings.is_empty(), "Should have material rankings");

        // Top material should have good score
        assert!(
            rankings[0].lcf_score > 0.5,
            "Top material should have good score"
        );
    }

    #[test]
    fn test_h_compatible_laminates() {
        let (_, materials, radiation) = setup();

        let damage = radiation.calculate_damage(14.1, 1e14, 22);

        let pd_nb = materials
            .nano_laminates
            .iter()
            .find(|l| l.name == "Pd/Nb")
            .expect("Pd/Nb should exist");

        let analysis = materials.analyze_nano_laminate(pd_nb, &damage);
        assert!(analysis.h_compatible, "Pd/Nb should be H-compatible");
    }
}
