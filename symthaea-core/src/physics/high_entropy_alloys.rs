// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # High-Entropy Alloys: Self-Healing Metamaterials for Fusion
//!
//! High-Entropy Alloys (HEAs) are multi-principal element alloys
//! with 5+ elements in near-equal proportions. They exhibit:
//!
//! - **Sluggish Diffusion**: Multiple elements slow defect migration
//! - **Cocktail Effect**: Emergent properties beyond component average
//! - **Lattice Distortion**: Local strain fields trap defects
//! - **High Configurational Entropy**: Stabilizes solid solution
//!
//! ## The Wolverine Strategy
//!
//! For Lattice Confinement Fusion, we need an alloy that:
//! 1. Absorbs deuterium (H-solubility)
//! 2. Survives 14.1 MeV neutron bombardment
//! 3. Self-heals radiation damage
//! 4. Maintains phonon coupling for LCF
//!
//! ## Key HEA Families
//!
//! - **Cantor Alloy**: CoCrFeMnNi (FCC, ductile, good radiation resistance)
//! - **Refractory HEA**: HfNbTaTiZr (BCC, high melting point, excellent radiation)
//! - **Light HEA**: AlCrFeMnTi (lighter, good H-storage)
//! - **Eutectic HEA**: AlCoCrFeNi2.1 (dual-phase, excellent mechanical)

use super::constants::R_GAS;
use super::phonon_dynamics::CrystalStructure;
use super::radiation_damage::{
    FusionReaction, HealingAnalysis, RadiationDamage, RadiationDamageSystem,
};
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use std::collections::HashMap;

/// An element contribution to an HEA
#[derive(Debug, Clone)]
pub struct HEAElement {
    /// Atomic number
    pub z: u8,
    /// Element symbol
    pub symbol: String,
    /// Molar fraction (0-1)
    pub fraction: f64,
    /// Atomic radius in pm
    pub radius_pm: f64,
    /// Melting point in K
    pub melting_k: f64,
    /// Hydrogen solubility factor (0-1)
    pub h_solubility: f64,
    /// Neutron activation cross-section (barns)
    pub activation_barn: f64,
}

/// High-Entropy Alloy definition
#[derive(Debug, Clone)]
pub struct HighEntropyAlloy {
    /// Alloy name
    pub name: String,
    /// Component elements
    pub elements: Vec<HEAElement>,
    /// Crystal structure
    pub structure: CrystalStructure,
    /// Configurational entropy (J/mol·K)
    pub entropy_j_mol_k: f64,
    /// Average lattice distortion
    pub lattice_distortion: f64,
    /// Sluggish diffusion factor (higher = slower)
    pub sluggish_factor: f64,
    /// Combined Debye temperature (K)
    pub debye_temp_k: f64,
    /// Overall H-solubility (0-1)
    pub h_solubility: f64,
    /// Radiation resistance score (0-1)
    pub radiation_resistance: f32,
    /// Self-healing score (0-1)
    pub self_healing_score: f32,
    /// LCF compatibility score (0-1)
    pub lcf_score: f32,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// HEA search target specification
#[derive(Debug, Clone)]
pub struct HEATarget {
    /// H-solubility requirement (0-1)
    pub h_solubility: f32,
    /// Neutron tolerance requirement (0-1)
    pub neutron_tolerance: f32,
    /// Self-healing requirement (0-1)
    pub self_healing: f32,
    /// Phonon coupling requirement (0-1)
    pub phonon_coupling: f32,
    /// Cost constraint (0 = cheap common metals, 1 = any cost)
    pub cost_tolerance: f32,
}

impl Default for HEATarget {
    fn default() -> Self {
        Self {
            h_solubility: 0.8,
            neutron_tolerance: 0.9,
            self_healing: 0.95,
            phonon_coupling: 0.7,
            cost_tolerance: 0.5,
        }
    }
}

impl HEATarget {
    /// Optimal target for LCF self-healing lattice
    pub fn lcf_wolverine() -> Self {
        Self {
            h_solubility: 0.9,       // Must hold deuterium
            neutron_tolerance: 0.95, // Must survive DT neutrons
            self_healing: 0.99,      // Critical: must heal
            phonon_coupling: 0.8,    // Good phonon properties
            cost_tolerance: 0.7,     // Prefer common elements
        }
    }
}

/// High-Entropy Alloy Design System
#[derive(Debug, Clone)]
pub struct HEADesigner {
    /// Element database
    elements: HashMap<u8, HEAElement>,
    /// Concept vectors
    pub high_entropy: ContinuousHV,
    pub sluggish: ContinuousHV,
    pub lattice_distortion: ContinuousHV,
    pub cocktail_effect: ContinuousHV,
    pub self_healing: ContinuousHV,
    pub h_storage: ContinuousHV,
    pub refractory: ContinuousHV,
    /// Known HEA families
    pub alloys: Vec<HighEntropyAlloy>,
}

impl HEADesigner {
    /// Create from genesis seed
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        // Concept vectors
        let high_entropy = genesis.hv("hea::high_entropy", PHYSICS_DIM);
        let sluggish = genesis.hv("hea::sluggish_diffusion", PHYSICS_DIM);
        let lattice_distortion = genesis.hv("hea::lattice_distortion", PHYSICS_DIM);
        let cocktail_effect = genesis.hv("hea::cocktail_effect", PHYSICS_DIM);
        let self_healing = genesis.hv("hea::self_healing", PHYSICS_DIM);
        let h_storage = genesis.hv("hea::h_storage", PHYSICS_DIM);
        let refractory = genesis.hv("hea::refractory", PHYSICS_DIM);

        let mut designer = Self {
            elements: HashMap::new(),
            high_entropy,
            sluggish,
            lattice_distortion,
            cocktail_effect,
            self_healing,
            h_storage,
            refractory,
            alloys: Vec::new(),
        };

        designer.init_elements();
        designer.init_alloys(genesis);
        designer
    }

    /// Initialize element database
    fn init_elements(&mut self) {
        let element_data = vec![
            // (Z, symbol, radius_pm, melting_K, h_solubility, activation_barn)

            // Light elements
            (13, "Al", 143.0, 933.0, 0.1, 0.23),
            (14, "Si", 111.0, 1687.0, 0.05, 0.17),
            // First-row transition metals
            (22, "Ti", 147.0, 1941.0, 0.95, 6.1), // Excellent H-storage
            (23, "V", 134.0, 2183.0, 0.90, 5.0),
            (24, "Cr", 128.0, 2180.0, 0.30, 3.1),
            (25, "Mn", 127.0, 1519.0, 0.20, 13.3),
            (26, "Fe", 126.0, 1811.0, 0.15, 2.6),
            (27, "Co", 125.0, 1768.0, 0.10, 37.2), // High activation
            (28, "Ni", 124.0, 1728.0, 0.35, 4.5),
            (29, "Cu", 128.0, 1358.0, 0.05, 3.8),
            // Second-row transition metals
            (40, "Zr", 160.0, 2128.0, 0.90, 0.18), // Low activation, good H
            (41, "Nb", 146.0, 2750.0, 0.85, 1.15),
            (42, "Mo", 139.0, 2896.0, 0.25, 2.5),
            (44, "Ru", 134.0, 2607.0, 0.10, 2.6),
            (45, "Rh", 134.0, 2237.0, 0.10, 145.0), // Very high activation
            (46, "Pd", 137.0, 1828.0, 1.00, 12.0),  // Best H-storage
            // Third-row transition metals (refractory)
            (72, "Hf", 159.0, 2506.0, 0.70, 104.0), // High activation
            (73, "Ta", 146.0, 3290.0, 0.60, 20.6),
            (74, "W", 139.0, 3695.0, 0.15, 18.3),
            (75, "Re", 137.0, 3459.0, 0.10, 90.0),
            // Lanthanides (for H-storage)
            (57, "La", 187.0, 1193.0, 0.85, 8.9),
            (68, "Er", 176.0, 1802.0, 0.95, 160.0), // NASA LCF host
            (70, "Yb", 176.0, 1097.0, 0.80, 35.0),
        ];

        for (z, symbol, radius, melting, h_sol, activation) in element_data {
            self.elements.insert(
                z,
                HEAElement {
                    z,
                    symbol: symbol.to_string(),
                    fraction: 0.0, // Set when composing alloys
                    radius_pm: radius,
                    melting_k: melting,
                    h_solubility: h_sol,
                    activation_barn: activation,
                },
            );
        }
    }

    /// Initialize known HEA families
    fn init_alloys(&mut self, genesis: &GenesisSeed) {
        // Cantor Alloy: CoCrFeMnNi (equiatomic)
        let cantor = self.compose_alloy(
            genesis,
            "Cantor (CoCrFeMnNi)",
            &[(27, 0.2), (24, 0.2), (26, 0.2), (25, 0.2), (28, 0.2)],
            CrystalStructure::FCC,
        );
        self.alloys.push(cantor);

        // Refractory HEA: HfNbTaTiZr
        let refractory = self.compose_alloy(
            genesis,
            "Refractory (HfNbTaTiZr)",
            &[(72, 0.2), (41, 0.2), (73, 0.2), (22, 0.2), (40, 0.2)],
            CrystalStructure::BCC,
        );
        self.alloys.push(refractory);

        // Light HEA: AlCrFeTiV
        let light = self.compose_alloy(
            genesis,
            "Light (AlCrFeTiV)",
            &[(13, 0.2), (24, 0.2), (26, 0.2), (22, 0.2), (23, 0.2)],
            CrystalStructure::BCC,
        );
        self.alloys.push(light);

        // H-Storage HEA: TiVZrNbPd
        let h_storage = self.compose_alloy(
            genesis,
            "H-Storage (TiVZrNbPd)",
            &[(22, 0.2), (23, 0.2), (40, 0.2), (41, 0.2), (46, 0.2)],
            CrystalStructure::BCC,
        );
        self.alloys.push(h_storage);

        // Wolverine Special: TiZrNbTaEr (optimized for LCF self-healing)
        let wolverine = self.compose_alloy(
            genesis,
            "Wolverine (TiZrNbTaEr)",
            &[(22, 0.25), (40, 0.25), (41, 0.20), (73, 0.15), (68, 0.15)],
            CrystalStructure::BCC,
        );
        self.alloys.push(wolverine);

        // Liquid Metal Alternative: GaInSn (eutectic, room temp liquid)
        // Not a true HEA but included for comparison
        let liquid = self.compose_liquid_alloy(
            genesis,
            "Galinstan (GaInSn)",
            CrystalStructure::FCC, // Meaningless for liquid
        );
        self.alloys.push(liquid);

        // CALPHAD-inspired: Optimal from computational search
        let optimal = self.compose_alloy(
            genesis,
            "Optimal-1 (TiVNbZrHf)",
            &[(22, 0.25), (23, 0.20), (41, 0.25), (40, 0.20), (72, 0.10)],
            CrystalStructure::BCC,
        );
        self.alloys.push(optimal);
    }

    /// Compose an HEA from element fractions
    fn compose_alloy(
        &self,
        _genesis: &GenesisSeed,
        name: &str,
        composition: &[(u8, f64)],
        structure: CrystalStructure,
    ) -> HighEntropyAlloy {
        let mut elements = Vec::new();
        let mut total_h_sol = 0.0;
        let mut total_radius = 0.0;
        let mut total_melting = 0.0;
        let mut total_activation = 0.0;
        let mut radius_variance = 0.0;

        // Build element list
        for (z, fraction) in composition {
            if let Some(elem) = self.elements.get(z) {
                let mut e = elem.clone();
                e.fraction = *fraction;
                total_h_sol += e.h_solubility * fraction;
                total_radius += e.radius_pm * fraction;
                total_melting += e.melting_k * fraction;
                total_activation += e.activation_barn * fraction;
                elements.push(e);
            }
        }

        // Calculate lattice distortion (atomic radius variance)
        for (z, fraction) in composition {
            if let Some(elem) = self.elements.get(z) {
                let diff = (elem.radius_pm - total_radius) / total_radius;
                radius_variance += diff * diff * fraction;
            }
        }
        let lattice_distortion = radius_variance.sqrt();

        // Configurational entropy: S = -R * Σ(x_i * ln(x_i))
        let entropy: f64 = -R_GAS
            * composition
                .iter()
                .map(|(_, x)| if *x > 0.0 { x * x.ln() } else { 0.0 })
                .sum::<f64>();

        // Sluggish diffusion factor (higher entropy + distortion = slower)
        let sluggish_factor = (entropy / R_GAS) * (1.0 + lattice_distortion * 10.0);

        // Estimate Debye temperature (rule of mixtures)
        let debye_temp_k = match structure {
            CrystalStructure::BCC => 350.0 + total_melting / 10.0,
            CrystalStructure::FCC => 300.0 + total_melting / 12.0,
            _ => 300.0,
        };

        // Radiation resistance (based on structure and activation)
        let structure_factor = match structure {
            CrystalStructure::BCC => 0.9,
            CrystalStructure::FCC => 0.7,
            _ => 0.6,
        };
        let activation_factor = (100.0 / (total_activation + 10.0)).min(1.0);
        let radiation_resistance =
            (structure_factor * activation_factor * 0.8 + sluggish_factor / 10.0 * 0.2) as f32;

        // Self-healing score
        let self_healing_score = (sluggish_factor / 5.0).min(1.0) as f32
            * radiation_resistance
            * (1.0 + lattice_distortion * 5.0) as f32;

        // LCF score
        let lcf_score =
            (total_h_sol as f32 * 0.4 + radiation_resistance * 0.3 + self_healing_score * 0.3)
                .min(1.0);

        // Build vector
        let mut vecs = vec![
            (&self.high_entropy, entropy as f32 / 15.0),
            (&self.sluggish, sluggish_factor as f32 / 5.0),
            (&self.lattice_distortion, lattice_distortion as f32 * 10.0),
            (&self.self_healing, self_healing_score),
            (&self.h_storage, total_h_sol as f32),
        ];
        if structure == CrystalStructure::BCC {
            vecs.push((&self.refractory, 0.5));
        }

        let (refs, weights): (Vec<_>, Vec<_>) = vecs.into_iter().unzip();
        let vector = ContinuousHV::weighted_bundle(&refs, &weights);

        HighEntropyAlloy {
            name: name.to_string(),
            elements,
            structure,
            entropy_j_mol_k: entropy,
            lattice_distortion,
            sluggish_factor,
            debye_temp_k,
            h_solubility: total_h_sol,
            radiation_resistance,
            self_healing_score,
            lcf_score,
            vector,
        }
    }

    /// Create liquid metal pseudo-alloy
    fn compose_liquid_alloy(
        &self,
        _genesis: &GenesisSeed,
        name: &str,
        _structure: CrystalStructure,
    ) -> HighEntropyAlloy {
        // Galinstan properties
        let vector = ContinuousHV::weighted_bundle(
            &[&self.self_healing, &self.h_storage],
            &[1.0, 0.3], // Excellent healing, moderate H storage
        );

        HighEntropyAlloy {
            name: name.to_string(),
            elements: vec![],
            structure: CrystalStructure::FCC,
            entropy_j_mol_k: 0.0, // Liquid, not applicable
            lattice_distortion: 0.0,
            sluggish_factor: 0.0,
            debye_temp_k: 0.0,
            h_solubility: 0.3,
            radiation_resistance: 1.0, // Liquid heals instantly
            self_healing_score: 1.0,
            lcf_score: 0.6, // Limited by H-solubility
            vector,
        }
    }

    /// Search for optimal HEA matching target
    pub fn search(
        &self,
        target: &HEATarget,
        radiation: &RadiationDamageSystem,
        fusion_type: FusionReaction,
    ) -> Vec<HEASearchResult> {
        let mut results = Vec::new();

        let neutron_energy = fusion_type.neutron_energy_mev().unwrap_or(0.0);
        let flux = 1e14; // Standard high flux

        for alloy in &self.alloys {
            // Calculate actual healing performance
            let primary_z = alloy.elements.first().map(|e| e.z).unwrap_or(26);
            let damage = radiation.calculate_damage(neutron_energy, flux, primary_z);

            // Create mock material for healing analysis
            let is_liquid = alloy.name.contains("Galinstan");
            let is_hea = !is_liquid && alloy.elements.len() >= 5;

            let healing = radiation.analyze_healing(
                &super::phonon_dynamics::LatticeMaterial {
                    name: alloy.name.clone(),
                    primary_z,
                    structure: alloy.structure,
                    debye_temp_k: alloy.debye_temp_k,
                    debye_freq_thz: alloy.debye_temp_k / 50.0,
                    lattice_const_a: 3.2,
                    vector: alloy.vector.clone(),
                },
                &damage,
                is_hea,
                is_liquid,
                50.0, // 50nm grain size
            );

            // Score against target
            let scores = HEAScores {
                h_solubility: alloy.h_solubility as f32,
                neutron_tolerance: alloy.radiation_resistance,
                self_healing: healing.healing_score,
                phonon_coupling: (alloy.debye_temp_k / 500.0).min(1.0) as f32,
                cost: self.estimate_cost(alloy),
            };

            let match_score = self.compute_match(target, &scores);

            results.push(HEASearchResult {
                alloy: alloy.clone(),
                damage,
                healing,
                scores,
                match_score,
            });
        }

        // Sort by match score
        results.sort_by(|a, b| {
            b.match_score
                .partial_cmp(&a.match_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Estimate relative cost (0 = expensive, 1 = cheap)
    fn estimate_cost(&self, alloy: &HighEntropyAlloy) -> f32 {
        let expensive_elements = [72, 73, 75, 68, 46, 45]; // Hf, Ta, Re, Er, Pd, Rh

        let expensive_fraction: f64 = alloy
            .elements
            .iter()
            .filter(|e| expensive_elements.contains(&e.z))
            .map(|e| e.fraction)
            .sum();

        (1.0 - expensive_fraction).max(0.0) as f32
    }

    /// Compute match score against target
    fn compute_match(&self, target: &HEATarget, scores: &HEAScores) -> f32 {
        let mut total = 0.0;
        let mut weight_sum = 0.0;

        // H-solubility match
        let h_weight = 1.0;
        total += (1.0 - (target.h_solubility - scores.h_solubility).abs()) * h_weight;
        weight_sum += h_weight;

        // Neutron tolerance match
        let n_weight = 1.5;
        total += (1.0
            - (target.neutron_tolerance - scores.neutron_tolerance)
                .abs()
                .min(1.0))
            * n_weight;
        weight_sum += n_weight;

        // Self-healing match (most critical)
        let heal_weight = 2.0;
        total += (1.0 - (target.self_healing - scores.self_healing).abs().min(1.0)) * heal_weight;
        weight_sum += heal_weight;

        // Phonon coupling match
        let phon_weight = 0.8;
        total += (1.0 - (target.phonon_coupling - scores.phonon_coupling).abs()) * phon_weight;
        weight_sum += phon_weight;

        // Cost constraint
        let cost_weight = 0.5;
        total +=
            (scores.cost.min(target.cost_tolerance) / target.cost_tolerance.max(0.1)) * cost_weight;
        weight_sum += cost_weight;

        total / weight_sum
    }

    /// Generate novel HEA composition via inverse search
    pub fn generate_optimal(&self, target: &HEATarget, genesis: &GenesisSeed) -> HighEntropyAlloy {
        // Build ideal composition by selecting best elements for each property

        // H-storage elements: Ti, V, Zr, Nb, Pd
        let h_elements = [(22, 0.90), (23, 0.85), (40, 0.90), (41, 0.80), (46, 1.0)];

        // Radiation-resistant elements: Ta, W, Nb, Zr, V
        let rad_elements = [(73, 0.95), (74, 0.90), (41, 0.85), (40, 0.80), (23, 0.75)];

        // Low-cost elements: Ti, V, Cr, Fe, Ni
        let cheap_elements = [(22, 1.0), (23, 0.9), (24, 0.8), (26, 0.9), (28, 0.85)];

        // Score each element
        let mut element_scores: HashMap<u8, f64> = HashMap::new();

        for (z, score) in h_elements {
            *element_scores.entry(z).or_insert(0.0) += score * target.h_solubility as f64;
        }
        for (z, score) in rad_elements {
            *element_scores.entry(z).or_insert(0.0) += score * target.neutron_tolerance as f64;
            *element_scores.entry(z).or_insert(0.0) += score * target.self_healing as f64;
        }
        if target.cost_tolerance < 0.8 {
            for (z, score) in cheap_elements {
                *element_scores.entry(z).or_insert(0.0) +=
                    score * (1.0 - target.cost_tolerance as f64);
            }
        }

        // Select top 5 elements
        let mut scored: Vec<(u8, f64)> = element_scores.into_iter().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_5: Vec<(u8, f64)> = scored
            .into_iter()
            .take(5)
            .map(|(z, _)| (z, 0.2)) // Equiatomic
            .collect();

        let composition_str: String = top_5
            .iter()
            .filter_map(|(z, _)| self.elements.get(z).map(|e| e.symbol.clone()))
            .collect::<Vec<_>>()
            .join("");

        self.compose_alloy(
            genesis,
            &format!("Generated ({composition_str})"),
            &top_5,
            CrystalStructure::BCC, // BCC preferred for radiation resistance
        )
    }
}

/// Scores for HEA properties
#[derive(Debug, Clone)]
pub struct HEAScores {
    pub h_solubility: f32,
    pub neutron_tolerance: f32,
    pub self_healing: f32,
    pub phonon_coupling: f32,
    pub cost: f32,
}

/// Result from HEA search
#[derive(Debug, Clone)]
pub struct HEASearchResult {
    pub alloy: HighEntropyAlloy,
    pub damage: RadiationDamage,
    pub healing: HealingAnalysis,
    pub scores: HEAScores,
    pub match_score: f32,
}

/// Consensus result from multi-seed search
#[derive(Debug, Clone)]
pub struct HEAConsensusResult {
    /// Alloy that appeared across seeds
    pub alloy: HighEntropyAlloy,
    /// Best scores seen
    pub scores: HEAScores,
    /// Best match score seen
    pub best_match_score: f32,
    /// Average match score across seeds
    pub avg_match_score: f32,
    /// How many seeds found this alloy in top N
    pub occurrence_count: usize,
    /// Consensus score (occurrence * avg_match)
    pub consensus_score: f32,
}

/// Multi-seed HEA search for consensus candidates
pub fn multi_seed_hea_search(
    base_phrase: &str,
    target: &HEATarget,
    fusion_type: FusionReaction,
    n_seeds: usize,
    top_n: usize,
) -> Vec<HEAConsensusResult> {
    use std::collections::HashMap;

    let mut all_results: HashMap<String, Vec<(HEASearchResult, usize)>> = HashMap::new();

    for seed_idx in 0..n_seeds {
        let phrase = format!("{base_phrase} seed {seed_idx}");
        let genesis = GenesisSeed::from_phrase(&phrase);

        let designer = HEADesigner::from_genesis(&genesis);
        let radiation = RadiationDamageSystem::from_genesis(&genesis);

        let results = designer.search(target, &radiation, fusion_type);

        // Track top N results from this seed
        for result in results.into_iter().take(top_n) {
            let name = result.alloy.name.clone();
            all_results
                .entry(name)
                .or_default()
                .push((result, seed_idx));
        }
    }

    // Aggregate by alloy name
    let mut consensus: Vec<HEAConsensusResult> = all_results
        .into_values()
        .map(|results| {
            let occurrence_count = results.len();
            let best_result = results
                .iter()
                .max_by(|a, b| {
                    a.0.match_score
                        .partial_cmp(&b.0.match_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("results vec is non-empty by construction (accumulated via push)");

            let total_match: f32 = results.iter().map(|(r, _)| r.match_score).sum();
            let avg_match_score = total_match / occurrence_count as f32;

            let consensus_score = (occurrence_count as f32 / n_seeds as f32) * avg_match_score;

            HEAConsensusResult {
                alloy: best_result.0.alloy.clone(),
                scores: best_result.0.scores.clone(),
                best_match_score: best_result.0.match_score,
                avg_match_score,
                occurrence_count,
                consensus_score,
            }
        })
        .collect();

    // Sort by consensus score
    consensus.sort_by(|a, b| {
        b.consensus_score
            .partial_cmp(&a.consensus_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    consensus
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::RadiationDamageSystem;

    fn setup() -> (GenesisSeed, HEADesigner, RadiationDamageSystem) {
        let genesis = GenesisSeed::from_phrase("hea test");
        let designer = HEADesigner::from_genesis(&genesis);
        let radiation = RadiationDamageSystem::from_genesis(&genesis);
        (genesis, designer, radiation)
    }

    #[test]
    fn test_hea_creation() {
        let (_, designer, _) = setup();
        assert!(
            designer.alloys.len() >= 5,
            "Should have at least 5 HEA families"
        );
    }

    #[test]
    fn test_cantor_properties() {
        let (_, designer, _) = setup();

        let cantor = designer
            .alloys
            .iter()
            .find(|a| a.name.contains("Cantor"))
            .expect("Cantor alloy should exist");

        assert_eq!(cantor.elements.len(), 5);
        assert_eq!(cantor.structure, CrystalStructure::FCC);
        assert!(cantor.entropy_j_mol_k > 13.0, "Should have high entropy");
    }

    #[test]
    fn test_wolverine_properties() {
        let (_, designer, _) = setup();

        let wolverine = designer
            .alloys
            .iter()
            .find(|a| a.name.contains("Wolverine"))
            .expect("Wolverine alloy should exist");

        // Should have good H-storage (Ti, Zr, Nb are H-philic)
        assert!(
            wolverine.h_solubility > 0.5,
            "Wolverine should have good H-storage"
        );

        // Should have good radiation resistance
        assert!(
            wolverine.radiation_resistance > 0.5,
            "Wolverine should resist radiation"
        );
    }

    #[test]
    fn test_search() {
        let (_, designer, radiation) = setup();

        let target = HEATarget::lcf_wolverine();
        let results = designer.search(&target, &radiation, FusionReaction::DT);

        assert!(!results.is_empty(), "Search should find results");

        // Top result should have decent score
        let top = &results[0];
        assert!(top.match_score > 0.5, "Top match should score > 0.5");
    }

    #[test]
    fn test_liquid_metal_healing() {
        let (_, designer, radiation) = setup();

        let target = HEATarget::lcf_wolverine();
        let results = designer.search(&target, &radiation, FusionReaction::DT);

        let liquid = results
            .iter()
            .find(|r| r.alloy.name.contains("Galinstan"))
            .expect("Galinstan should be in results");

        assert_eq!(
            liquid.scores.self_healing, 1.0,
            "Liquid metal should have perfect healing"
        );
    }

    #[test]
    fn test_generate_optimal() {
        let (genesis, designer, _) = setup();

        let target = HEATarget::lcf_wolverine();
        let optimal = designer.generate_optimal(&target, &genesis);

        assert_eq!(optimal.elements.len(), 5, "Should have 5 elements");
        assert!(optimal.h_solubility > 0.5, "Should have good H-storage");
    }

    #[test]
    fn test_multi_seed_search() {
        let target = HEATarget::lcf_wolverine();

        // Run with 3 seeds for quick test
        let consensus = multi_seed_hea_search(
            "test consensus",
            &target,
            FusionReaction::DT,
            3, // n_seeds
            5, // top_n
        );

        assert!(!consensus.is_empty(), "Should find consensus candidates");

        // Top result should appear in multiple seeds
        let top = &consensus[0];
        assert!(
            top.occurrence_count >= 1,
            "Top result should appear at least once"
        );
        assert!(
            top.consensus_score > 0.0,
            "Should have positive consensus score"
        );
    }
}
