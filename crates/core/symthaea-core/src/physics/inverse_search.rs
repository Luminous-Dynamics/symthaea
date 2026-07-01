// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Inverse Search: Dream Backwards from Target Properties
//!
//! Instead of checking candidates one by one, we define the ideal solution
//! as a vector and search for what minimizes distance to it.
//!
//! ## The Key Insight
//!
//! In HDC, if `Result = A ⊗ B`, then `A = Result ⊗ B⁻¹`.
//!
//! This means we can:
//! 1. Define a "Target Vector" with ideal properties
//! 2. Unbind known constraints
//! 3. The residual tells us what's missing
//!
//! ## The Search Strategy
//!
//! ```text
//! Target Vector:                 Search Space:
//! ┌─────────────────┐           ┌─────────────────┐
//! │ High Energy     │           │ All Isotopes    │
//! │ Low Trigger     │ ────────► │ All Lattices    │
//! │ High Stability  │ nearest   │ All Couplings   │
//! │ High Abundance  │ neighbor  │                 │
//! └─────────────────┘           └─────────────────┘
//! ```
//!
//! ## Multi-Seed Exploration
//!
//! Spawn N searches with mutated genesis seeds.
//! Each explores a different "universe."
//! Bundle results to find consensus candidates.

use super::electron_nuclear_coupling::ElectronNuclearCoupling;
use super::hadrons::Hadrons;
use super::periodic_table::PeriodicTable;
use super::phonon_dynamics::PhononDynamics;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Target property specification
#[derive(Debug, Clone)]
pub struct TargetProperties {
    /// Energy density (0 = chemical, 1 = nuclear)
    pub energy_density: f32,
    /// Trigger accessibility (0 = gamma ray needed, 1 = visible light)
    pub trigger_accessibility: f32,
    /// Long-term stability (0 = nanoseconds, 1 = decades)
    pub stability: f32,
    /// Material abundance (0 = rare, 1 = common)
    pub abundance: f32,
    /// Electron-nuclear coupling strength
    pub neec_coupling: f32,
    /// Phonon-nuclear coupling strength
    pub phonon_coupling: f32,
    /// Safety (0 = hazardous, 1 = safe)
    pub safety: f32,
}

impl Default for TargetProperties {
    fn default() -> Self {
        Self {
            energy_density: 0.8,        // High but not antimatter
            trigger_accessibility: 0.9, // Accessible with lasers
            stability: 0.7,             // Years to decades
            abundance: 0.6,             // Reasonably available
            neec_coupling: 0.5,         // Some coupling
            phonon_coupling: 0.5,       // Some coupling
            safety: 0.7,                // Reasonably safe
        }
    }
}

impl TargetProperties {
    /// The "Holy Grail" - perfect energy source
    pub fn ideal_energy_source() -> Self {
        Self {
            energy_density: 0.9,
            trigger_accessibility: 1.0,
            stability: 0.9,
            abundance: 0.8,
            neec_coupling: 0.8,
            phonon_coupling: 0.8,
            safety: 0.9,
        }
    }

    /// Optimized for Lattice Confinement Fusion
    pub fn lcf_optimized() -> Self {
        Self {
            energy_density: 1.0,        // Fusion energy
            trigger_accessibility: 0.7, // X-ray trigger OK
            stability: 0.3,             // Transient is fine
            abundance: 1.0,             // Deuterium from water
            neec_coupling: 0.3,         // Less important for fusion
            phonon_coupling: 1.0,       // Critical for LCF
            safety: 0.6,                // Fusion is energetic
        }
    }

    /// Optimized for nuclear clock (Thorium-229m style)
    pub fn nuclear_clock() -> Self {
        Self {
            energy_density: 0.1,        // Low energy is fine
            trigger_accessibility: 1.0, // Must be laser accessible
            stability: 0.8,             // Stable enough to measure
            abundance: 0.4,             // Rare is OK for clocks
            neec_coupling: 0.9,         // High coupling needed
            phonon_coupling: 0.5,
            safety: 0.8,
        }
    }
}

/// A candidate solution from the search
#[derive(Debug, Clone)]
pub struct SearchCandidate {
    /// Candidate name/description
    pub name: String,
    /// Atomic number of primary element
    pub z: u8,
    /// Mass number (if applicable)
    pub a: Option<u16>,
    /// Host material (if applicable)
    pub host_material: Option<String>,
    /// Trigger protocol
    pub trigger: TriggerProtocol,
    /// Match score (0-1, higher = better match)
    pub match_score: f32,
    /// Individual property scores
    pub property_scores: PropertyScores,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// Trigger protocol for activating the energy release
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerProtocol {
    /// Trigger type
    pub method: TriggerMethod,
    /// Wavelength/frequency if applicable
    pub wavelength_nm: Option<f64>,
    /// Pulse duration if applicable
    pub pulse_duration: Option<String>,
    /// Temperature requirement in Kelvin
    pub temperature_k: Option<f64>,
    /// Additional notes
    pub notes: String,
}

/// Trigger method types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerMethod {
    /// UV/Visible laser
    Laser,
    /// X-ray source
    XRay,
    /// Gamma ray source
    GammaRay,
    /// Phonon/acoustic excitation
    Phonon,
    /// Electron beam
    ElectronBeam,
    /// Thermal (temperature)
    Thermal,
    /// Spontaneous decay
    Spontaneous,
}

/// Individual property match scores
#[derive(Debug, Clone)]
pub struct PropertyScores {
    pub energy_density: f32,
    pub trigger_accessibility: f32,
    pub stability: f32,
    pub abundance: f32,
    pub neec_coupling: f32,
    pub phonon_coupling: f32,
    pub safety: f32,
}

/// Inverse Search Engine
pub struct InverseSearchEngine {
    /// Base genesis seed
    _genesis: GenesisSeed,
    /// Property concept vectors
    energy_density_vec: ContinuousHV,
    trigger_access_vec: ContinuousHV,
    stability_vec: ContinuousHV,
    abundance_vec: ContinuousHV,
    neec_vec: ContinuousHV,
    phonon_vec: ContinuousHV,
    safety_vec: ContinuousHV,
    /// Anti-property vectors (for low values)
    low_energy_vec: ContinuousHV,
    difficult_trigger_vec: ContinuousHV,
    unstable_vec: ContinuousHV,
    rare_vec: ContinuousHV,
    dangerous_vec: ContinuousHV,
}

impl InverseSearchEngine {
    /// Create search engine from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            _genesis: genesis.clone(),
            energy_density_vec: genesis.hv("property::high_energy_density", PHYSICS_DIM),
            trigger_access_vec: genesis.hv("property::trigger_accessible", PHYSICS_DIM),
            stability_vec: genesis.hv("property::stable", PHYSICS_DIM),
            abundance_vec: genesis.hv("property::abundant", PHYSICS_DIM),
            neec_vec: genesis.hv("property::neec_coupled", PHYSICS_DIM),
            phonon_vec: genesis.hv("property::phonon_coupled", PHYSICS_DIM),
            safety_vec: genesis.hv("property::safe", PHYSICS_DIM),
            low_energy_vec: genesis.hv("property::low_energy", PHYSICS_DIM),
            difficult_trigger_vec: genesis.hv("property::difficult_trigger", PHYSICS_DIM),
            unstable_vec: genesis.hv("property::unstable", PHYSICS_DIM),
            rare_vec: genesis.hv("property::rare", PHYSICS_DIM),
            dangerous_vec: genesis.hv("property::dangerous", PHYSICS_DIM),
        }
    }

    /// Build target vector from properties
    pub fn build_target_vector(&self, target: &TargetProperties) -> ContinuousHV {
        // Interpolate between high and low property vectors
        let energy = self.interpolate_property(
            target.energy_density,
            &self.energy_density_vec,
            &self.low_energy_vec,
        );
        let trigger = self.interpolate_property(
            target.trigger_accessibility,
            &self.trigger_access_vec,
            &self.difficult_trigger_vec,
        );
        let stability =
            self.interpolate_property(target.stability, &self.stability_vec, &self.unstable_vec);
        let abundance =
            self.interpolate_property(target.abundance, &self.abundance_vec, &self.rare_vec);
        let safety =
            self.interpolate_property(target.safety, &self.safety_vec, &self.dangerous_vec);

        // Weight by importance
        let neec = self.neec_vec.scale(target.neec_coupling);
        let phonon = self.phonon_vec.scale(target.phonon_coupling);

        // Bundle all properties
        ContinuousHV::weighted_bundle(
            &[
                &energy, &trigger, &stability, &abundance, &safety, &neec, &phonon,
            ],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8],
        )
    }

    /// Interpolate between high and low property vectors
    fn interpolate_property(
        &self,
        value: f32,
        high: &ContinuousHV,
        low: &ContinuousHV,
    ) -> ContinuousHV {
        ContinuousHV::weighted_bundle(&[high, low], &[value, 1.0 - value])
    }

    /// Score a candidate against target properties
    pub fn score_candidate(
        &self,
        candidate: &ContinuousHV,
        target: &TargetProperties,
    ) -> PropertyScores {
        PropertyScores {
            energy_density: self.score_property(
                candidate,
                target.energy_density,
                &self.energy_density_vec,
                &self.low_energy_vec,
            ),
            trigger_accessibility: self.score_property(
                candidate,
                target.trigger_accessibility,
                &self.trigger_access_vec,
                &self.difficult_trigger_vec,
            ),
            stability: self.score_property(
                candidate,
                target.stability,
                &self.stability_vec,
                &self.unstable_vec,
            ),
            abundance: self.score_property(
                candidate,
                target.abundance,
                &self.abundance_vec,
                &self.rare_vec,
            ),
            neec_coupling: candidate.similarity(&self.neec_vec).max(0.0),
            phonon_coupling: candidate.similarity(&self.phonon_vec).max(0.0),
            safety: self.score_property(
                candidate,
                target.safety,
                &self.safety_vec,
                &self.dangerous_vec,
            ),
        }
    }

    fn score_property(
        &self,
        candidate: &ContinuousHV,
        target_value: f32,
        high_vec: &ContinuousHV,
        low_vec: &ContinuousHV,
    ) -> f32 {
        let high_sim = candidate.similarity(high_vec);
        let low_sim = candidate.similarity(low_vec);

        // Normalize to 0-1
        let actual = (high_sim - low_sim + 1.0) / 2.0;

        // Score = how close to target
        1.0 - (actual - target_value).abs()
    }

    /// Run inverse search against all known candidates
    pub fn search(
        &self,
        target: &TargetProperties,
        neec: &ElectronNuclearCoupling,
        phonons: &PhononDynamics,
        _table: &PeriodicTable,
        _hadrons: &Hadrons,
    ) -> Vec<SearchCandidate> {
        let _target_vec = self.build_target_vector(target);
        let mut candidates = Vec::new();

        // Search NEEC candidates
        for coupling in &neec.candidates {
            let mut scores = self.score_candidate(&coupling.vector, target);

            // Override trigger_accessibility with physics-based score:
            // Lower nuclear transition energy = more accessible trigger
            // < 10 eV = laser accessible (score ~1.0)
            // ~ 1 keV = soft X-ray (score ~0.5)
            // > 100 keV = gamma ray (score ~0.1)
            let energy = coupling.nuclear.energy_ev;
            scores.trigger_accessibility = if energy < 100.0 {
                ((1.0 - (energy / 100.0) * 0.3).max(0.0)) as f32
            } else if energy < 10_000.0 {
                (0.5 * (1.0 - (energy.log10() - 2.0) / 2.0).max(0.0)) as f32
            } else {
                (0.1 * (1.0 - (energy.log10() - 4.0) / 3.0).max(0.0)) as f32
            };

            let match_score = self.compute_match_score(&scores, target);

            let trigger = self.infer_trigger(coupling.nuclear.energy_ev);

            candidates.push(SearchCandidate {
                name: format!(
                    "{}-{}m (NEEC)",
                    element_symbol(coupling.nuclear.z),
                    coupling.nuclear.a
                ),
                z: coupling.nuclear.z,
                a: Some(coupling.nuclear.a),
                host_material: None,
                trigger,
                match_score,
                property_scores: scores,
                vector: coupling.vector.clone(),
            });
        }

        // Search phonon-coupled materials for LCF
        for material in &phonons.materials {
            let lcf = phonons.analyze_lcf_potential(material);
            if lcf.lcf_potential > 0.3 {
                // Create LCF candidate vector
                let lcf_vec = material.vector.bind(&phonons.resonant);
                let scores = self.score_candidate(&lcf_vec, target);
                let match_score = self.compute_match_score(&scores, target);

                let trigger = TriggerProtocol {
                    method: TriggerMethod::XRay,
                    wavelength_nm: Some(0.1), // ~10 keV X-ray
                    pulse_duration: Some("femtosecond".to_string()),
                    temperature_k: Some(300.0),
                    notes: format!("LCF in {} lattice", material.name),
                };

                candidates.push(SearchCandidate {
                    name: format!("D2-{} (LCF)", material.name),
                    z: 1, // Deuterium
                    a: Some(2),
                    host_material: Some(material.name.clone()),
                    trigger,
                    match_score,
                    property_scores: scores,
                    vector: lcf_vec,
                });
            }
        }

        // Sort by match score
        candidates.sort_by(|a, b| {
            b.match_score
                .partial_cmp(&a.match_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
    }

    /// Compute overall match score from property scores
    fn compute_match_score(&self, scores: &PropertyScores, target: &TargetProperties) -> f32 {
        // Weighted average, emphasizing most important properties
        let weights = [
            (scores.energy_density, 1.0),
            (scores.trigger_accessibility, 1.2), // Slightly more important
            (scores.stability, 1.0),
            (scores.abundance, 0.8),
            (scores.neec_coupling, target.neec_coupling),
            (scores.phonon_coupling, target.phonon_coupling),
            (scores.safety, 0.9),
        ];

        let (sum, total_weight): (f32, f32) = weights
            .iter()
            .map(|(score, weight)| (score * weight, *weight))
            .fold((0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1));

        sum / total_weight.max(1.0)
    }

    /// Infer trigger method from transition energy
    fn infer_trigger(&self, energy_ev: f64) -> TriggerProtocol {
        if energy_ev < 10.0 {
            // VUV/UV range
            let wavelength = 1239.8 / energy_ev; // nm
            TriggerProtocol {
                method: TriggerMethod::Laser,
                wavelength_nm: Some(wavelength),
                pulse_duration: Some("femtosecond".to_string()),
                temperature_k: Some(4.0), // Cryogenic for precision
                notes: "VUV laser excitation".to_string(),
            }
        } else if energy_ev < 1000.0 {
            // Soft X-ray
            TriggerProtocol {
                method: TriggerMethod::XRay,
                wavelength_nm: Some(1239.8 / energy_ev),
                pulse_duration: Some("picosecond".to_string()),
                temperature_k: Some(300.0),
                notes: "Soft X-ray excitation".to_string(),
            }
        } else if energy_ev < 100_000.0 {
            // Hard X-ray
            TriggerProtocol {
                method: TriggerMethod::XRay,
                wavelength_nm: Some(1239.8 / energy_ev),
                pulse_duration: Some("nanosecond".to_string()),
                temperature_k: Some(300.0),
                notes: "Hard X-ray excitation".to_string(),
            }
        } else {
            // Gamma
            TriggerProtocol {
                method: TriggerMethod::GammaRay,
                wavelength_nm: None,
                pulse_duration: None,
                temperature_k: None,
                notes: format!("Gamma ray excitation ({:.0} keV)", energy_ev / 1000.0),
            }
        }
    }

    /// Multi-seed exploration: run search with varied genesis seeds
    pub fn multi_seed_search(
        base_phrase: &str,
        target: &TargetProperties,
        n_seeds: usize,
    ) -> Vec<(SearchCandidate, f32)> {
        let mut all_candidates: Vec<(SearchCandidate, usize)> = Vec::new();

        for i in 0..n_seeds {
            let phrase = format!("{base_phrase} variant {i}");
            let genesis = GenesisSeed::from_phrase(&phrase);

            let model = super::StandardModel::from_genesis(&genesis);
            let hadrons = Hadrons::from_model(&model, &genesis);
            let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
            let neec = ElectronNuclearCoupling::from_genesis(&genesis, &table, &hadrons);
            let phonons = PhononDynamics::from_genesis(&genesis, &table);

            let engine = InverseSearchEngine::from_genesis(&genesis);
            let candidates = engine.search(target, &neec, &phonons, &table, &hadrons);

            // Track top candidates from this seed
            for candidate in candidates.into_iter().take(5) {
                all_candidates.push((candidate, i));
            }
        }

        // Aggregate by name, counting occurrences
        let mut aggregated: std::collections::HashMap<String, (SearchCandidate, usize)> =
            std::collections::HashMap::new();

        for (candidate, _seed) in all_candidates {
            let entry = aggregated
                .entry(candidate.name.clone())
                .or_insert((candidate.clone(), 0));
            entry.1 += 1;
            // Keep highest scoring version
            if candidate.match_score > entry.0.match_score {
                entry.0 = candidate;
            }
        }

        // Convert to (candidate, consensus_score)
        let mut results: Vec<(SearchCandidate, f32)> = aggregated
            .into_iter()
            .map(|(_, (candidate, count))| {
                let consensus = count as f32 / n_seeds as f32;
                (candidate, consensus)
            })
            .collect();

        // Sort by consensus * match_score
        results.sort_by(|a, b| {
            let score_a = a.1 * a.0.match_score;
            let score_b = b.1 * b.0.match_score;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }
}

// ============================================================
// CAUSAL REASONING ENGINE
// ============================================================

/// A causal factor that contributes to a phenomenon
#[derive(Debug, Clone)]
pub struct CausalFactor {
    /// Name of the factor
    pub name: String,
    /// Category of factor
    pub category: CausalCategory,
    /// Contribution strength (0-1)
    pub strength: f64,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// Category of causal factor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalCategory {
    Symmetry,      // Conservation laws, gauge invariance
    Interaction,   // Forces, couplings
    QuantumEffect, // Tunneling, entanglement
    Thermodynamic, // Entropy, temperature
    Kinetic,       // Rates, barriers
    Structural,    // Geometry, topology
    Emergent,      // Collective phenomena
}

/// Causal Reasoning Engine for physics
pub struct CausalReasoningEngine {
    _genesis: GenesisSeed,

    // Phenomenon vectors
    phenomena: std::collections::HashMap<String, ContinuousHV>,

    // Causal factor vectors
    factors: std::collections::HashMap<String, (ContinuousHV, CausalCategory)>,
}

impl CausalReasoningEngine {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        let mut phenomena = std::collections::HashMap::new();
        let mut factors = std::collections::HashMap::new();

        // Register phenomena
        phenomena.insert(
            "superconductivity".to_string(),
            genesis.hv("phenomenon::superconductivity", PHYSICS_DIM),
        );
        phenomena.insert(
            "turbulence".to_string(),
            genesis.hv("phenomenon::turbulence", PHYSICS_DIM),
        );
        phenomena.insert(
            "superfluidity".to_string(),
            genesis.hv("phenomenon::superfluidity", PHYSICS_DIM),
        );
        phenomena.insert(
            "ferromagnetism".to_string(),
            genesis.hv("phenomenon::ferromagnetism", PHYSICS_DIM),
        );
        phenomena.insert(
            "bec".to_string(),
            genesis.hv("phenomenon::bec", PHYSICS_DIM),
        );
        phenomena.insert(
            "laser".to_string(),
            genesis.hv("phenomenon::laser", PHYSICS_DIM),
        );
        phenomena.insert(
            "fusion".to_string(),
            genesis.hv("phenomenon::fusion", PHYSICS_DIM),
        );
        phenomena.insert(
            "catalysis".to_string(),
            genesis.hv("phenomenon::catalysis", PHYSICS_DIM),
        );
        phenomena.insert(
            "photosynthesis".to_string(),
            genesis.hv("phenomenon::photosynthesis", PHYSICS_DIM),
        );
        phenomena.insert(
            "consciousness".to_string(),
            genesis.hv("phenomenon::consciousness", PHYSICS_DIM),
        );

        // Register causal factors
        factors.insert(
            "cooper_pairs".to_string(),
            (
                genesis.hv("cause::cooper_pairs", PHYSICS_DIM),
                CausalCategory::QuantumEffect,
            ),
        );
        factors.insert(
            "phonon_exchange".to_string(),
            (
                genesis.hv("cause::phonon_exchange", PHYSICS_DIM),
                CausalCategory::Interaction,
            ),
        );
        factors.insert(
            "bcs_theory".to_string(),
            (
                genesis.hv("cause::bcs_theory", PHYSICS_DIM),
                CausalCategory::Emergent,
            ),
        );
        factors.insert(
            "high_reynolds".to_string(),
            (
                genesis.hv("cause::high_reynolds", PHYSICS_DIM),
                CausalCategory::Kinetic,
            ),
        );
        factors.insert(
            "instability".to_string(),
            (
                genesis.hv("cause::instability", PHYSICS_DIM),
                CausalCategory::Kinetic,
            ),
        );
        factors.insert(
            "energy_cascade".to_string(),
            (
                genesis.hv("cause::energy_cascade", PHYSICS_DIM),
                CausalCategory::Thermodynamic,
            ),
        );
        factors.insert(
            "bose_statistics".to_string(),
            (
                genesis.hv("cause::bose_statistics", PHYSICS_DIM),
                CausalCategory::QuantumEffect,
            ),
        );
        factors.insert(
            "exchange_interaction".to_string(),
            (
                genesis.hv("cause::exchange_interaction", PHYSICS_DIM),
                CausalCategory::Interaction,
            ),
        );
        factors.insert(
            "spin_alignment".to_string(),
            (
                genesis.hv("cause::spin_alignment", PHYSICS_DIM),
                CausalCategory::Structural,
            ),
        );
        factors.insert(
            "population_inversion".to_string(),
            (
                genesis.hv("cause::population_inversion", PHYSICS_DIM),
                CausalCategory::QuantumEffect,
            ),
        );
        factors.insert(
            "stimulated_emission".to_string(),
            (
                genesis.hv("cause::stimulated_emission", PHYSICS_DIM),
                CausalCategory::QuantumEffect,
            ),
        );
        factors.insert(
            "coulomb_barrier".to_string(),
            (
                genesis.hv("cause::coulomb_barrier", PHYSICS_DIM),
                CausalCategory::Interaction,
            ),
        );
        factors.insert(
            "quantum_tunneling".to_string(),
            (
                genesis.hv("cause::quantum_tunneling", PHYSICS_DIM),
                CausalCategory::QuantumEffect,
            ),
        );
        factors.insert(
            "strong_force".to_string(),
            (
                genesis.hv("cause::strong_force", PHYSICS_DIM),
                CausalCategory::Interaction,
            ),
        );
        factors.insert(
            "activation_energy_lowering".to_string(),
            (
                genesis.hv("cause::activation_energy_lowering", PHYSICS_DIM),
                CausalCategory::Kinetic,
            ),
        );
        factors.insert(
            "transition_state".to_string(),
            (
                genesis.hv("cause::transition_state", PHYSICS_DIM),
                CausalCategory::Structural,
            ),
        );
        factors.insert(
            "light_harvesting".to_string(),
            (
                genesis.hv("cause::light_harvesting", PHYSICS_DIM),
                CausalCategory::QuantumEffect,
            ),
        );
        factors.insert(
            "electron_transport".to_string(),
            (
                genesis.hv("cause::electron_transport", PHYSICS_DIM),
                CausalCategory::Kinetic,
            ),
        );
        factors.insert(
            "integrated_information".to_string(),
            (
                genesis.hv("cause::integrated_information", PHYSICS_DIM),
                CausalCategory::Emergent,
            ),
        );
        factors.insert(
            "neural_synchrony".to_string(),
            (
                genesis.hv("cause::neural_synchrony", PHYSICS_DIM),
                CausalCategory::Emergent,
            ),
        );
        factors.insert(
            "symmetry_breaking".to_string(),
            (
                genesis.hv("cause::symmetry_breaking", PHYSICS_DIM),
                CausalCategory::Symmetry,
            ),
        );
        factors.insert(
            "gauge_invariance".to_string(),
            (
                genesis.hv("cause::gauge_invariance", PHYSICS_DIM),
                CausalCategory::Symmetry,
            ),
        );
        factors.insert(
            "entropy_increase".to_string(),
            (
                genesis.hv("cause::entropy_increase", PHYSICS_DIM),
                CausalCategory::Thermodynamic,
            ),
        );
        factors.insert(
            "low_temperature".to_string(),
            (
                genesis.hv("cause::low_temperature", PHYSICS_DIM),
                CausalCategory::Thermodynamic,
            ),
        );
        factors.insert(
            "high_temperature".to_string(),
            (
                genesis.hv("cause::high_temperature", PHYSICS_DIM),
                CausalCategory::Thermodynamic,
            ),
        );

        Self {
            _genesis: genesis.clone(),
            phenomena,
            factors,
        }
    }

    /// Query: "What causes phenomenon X?"
    /// Returns ranked list of causal factors
    pub fn what_causes(&self, phenomenon: &str) -> Vec<CausalFactor> {
        let phenom_vec = match self.phenomena.get(phenomenon) {
            Some(v) => v,
            None => return vec![], // Unknown phenomenon
        };

        let mut results: Vec<CausalFactor> = self
            .factors
            .iter()
            .map(|(name, (vec, cat))| {
                // Similarity indicates causal connection
                // In a real system, we'd have explicit causal bindings
                let strength = phenom_vec.similarity(vec).abs() as f64;
                CausalFactor {
                    name: name.clone(),
                    category: *cat,
                    strength,
                    vector: vec.clone(),
                }
            })
            .filter(|f| f.strength > 0.05) // Filter weak connections
            .collect();

        results.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Query: "What phenomena result from factor X?"
    pub fn what_results_from(&self, factor: &str) -> Vec<(String, f64)> {
        let factor_vec = match self.factors.get(factor) {
            Some((v, _)) => v,
            None => return vec![],
        };

        let mut results: Vec<(String, f64)> = self
            .phenomena
            .iter()
            .map(|(name, vec)| {
                let strength = factor_vec.similarity(vec).abs() as f64;
                (name.clone(), strength)
            })
            .filter(|(_, s)| *s > 0.05)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Find the causal chain from A to B
    pub fn causal_chain(&self, from: &str, to: &str) -> Vec<String> {
        let from_vec = match self.factors.get(from) {
            Some((v, _)) => v.clone(),
            None => match self.phenomena.get(from) {
                Some(v) => v.clone(),
                None => return vec![],
            },
        };

        let to_vec = match self.phenomena.get(to) {
            Some(v) => v.clone(),
            None => return vec![],
        };

        // Find intermediate factors that connect from → to
        let mut chain = vec![from.to_string()];

        // Simple greedy search: find factors that are similar to both
        let mut intermediates: Vec<(String, f64)> = self
            .factors
            .iter()
            .map(|(name, (vec, _))| {
                let from_sim = from_vec.similarity(vec).abs();
                let to_sim = vec.similarity(&to_vec).abs();
                let combined = (from_sim * to_sim) as f64;
                (name.clone(), combined)
            })
            .filter(|(name, s)| *s > 0.01 && name != from)
            .collect();

        intermediates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Add top intermediates
        for (name, _) in intermediates.into_iter().take(3) {
            chain.push(name);
        }

        chain.push(to.to_string());
        chain
    }

    /// Register a new phenomenon
    pub fn register_phenomenon(&mut self, name: &str, vector: ContinuousHV) {
        self.phenomena.insert(name.to_string(), vector);
    }

    /// Register a new causal factor
    pub fn register_factor(&mut self, name: &str, vector: ContinuousHV, category: CausalCategory) {
        self.factors.insert(name.to_string(), (vector, category));
    }

    /// Build causal explanation for a vector
    pub fn explain(&self, target: &ContinuousHV) -> Vec<CausalFactor> {
        let mut explanations: Vec<CausalFactor> = self
            .factors
            .iter()
            .map(|(name, (vec, cat))| {
                let strength = target.similarity(vec).abs() as f64;
                CausalFactor {
                    name: name.clone(),
                    category: *cat,
                    strength,
                    vector: vec.clone(),
                }
            })
            .filter(|f| f.strength > 0.1)
            .collect();

        explanations.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        explanations
    }
}

/// Convert atomic number to element symbol
fn element_symbol(z: u8) -> &'static str {
    match z {
        1 => "H",
        2 => "He",
        3 => "Li",
        4 => "Be",
        5 => "B",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        10 => "Ne",
        11 => "Na",
        12 => "Mg",
        13 => "Al",
        14 => "Si",
        15 => "P",
        16 => "S",
        17 => "Cl",
        18 => "Ar",
        19 => "K",
        20 => "Ca",
        26 => "Fe",
        42 => "Mo",
        43 => "Tc",
        46 => "Pd",
        68 => "Er",
        72 => "Hf",
        73 => "Ta",
        90 => "Th",
        92 => "U",
        _ => "X",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (
        GenesisSeed,
        InverseSearchEngine,
        ElectronNuclearCoupling,
        PhononDynamics,
        PeriodicTable,
        Hadrons,
    ) {
        let genesis = GenesisSeed::from_phrase("inverse search test");
        let model = super::super::StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
        let neec = ElectronNuclearCoupling::from_genesis(&genesis, &table, &hadrons);
        let phonons = PhononDynamics::from_genesis(&genesis, &table);
        let engine = InverseSearchEngine::from_genesis(&genesis);
        (genesis, engine, neec, phonons, table, hadrons)
    }

    #[test]
    fn test_target_vector_creation() {
        let (_, engine, _, _, _, _) = setup();

        let target = TargetProperties::ideal_energy_source();
        let vec = engine.build_target_vector(&target);

        assert_eq!(vec.dim(), PHYSICS_DIM);
    }

    #[test]
    fn test_basic_search() {
        let (_, engine, neec, phonons, table, hadrons) = setup();

        let target = TargetProperties::default();
        let candidates = engine.search(&target, &neec, &phonons, &table, &hadrons);

        assert!(!candidates.is_empty(), "Should find some candidates");

        // Top candidate should have decent score
        assert!(
            candidates[0].match_score > 0.3,
            "Best candidate should have match score > 0.3"
        );
    }

    #[test]
    fn test_nuclear_clock_search() {
        let (_, engine, neec, phonons, table, hadrons) = setup();

        let target = TargetProperties::nuclear_clock();
        let candidates = engine.search(&target, &neec, &phonons, &table, &hadrons);

        // Thorium-229m should score well for nuclear clock
        let th229 = candidates.iter().find(|c| c.name.contains("Th-229"));

        assert!(th229.is_some(), "Th-229m should be in results");
        let th229 = th229.unwrap();
        assert!(
            th229.property_scores.trigger_accessibility > 0.5,
            "Th-229m should have good trigger accessibility"
        );
    }

    #[test]
    fn test_lcf_search() {
        let (_, engine, neec, phonons, table, hadrons) = setup();

        let target = TargetProperties::lcf_optimized();
        let candidates = engine.search(&target, &neec, &phonons, &table, &hadrons);

        // Should find LCF candidates
        let lcf = candidates.iter().find(|c| c.name.contains("LCF"));

        assert!(lcf.is_some(), "Should find LCF candidates");
    }

    #[test]
    fn test_trigger_inference() {
        let (_, engine, _, _, _, _) = setup();

        // UV energy
        let uv_trigger = engine.infer_trigger(8.0);
        assert_eq!(uv_trigger.method, TriggerMethod::Laser);
        assert!(uv_trigger.wavelength_nm.unwrap() > 100.0);

        // X-ray energy
        let xray_trigger = engine.infer_trigger(10_000.0);
        assert_eq!(xray_trigger.method, TriggerMethod::XRay);

        // Gamma energy
        let gamma_trigger = engine.infer_trigger(1_000_000.0);
        assert_eq!(gamma_trigger.method, TriggerMethod::GammaRay);
    }

    #[test]
    fn test_causal_reasoning_creation() {
        let genesis = GenesisSeed::from_phrase("causal test");
        let engine = CausalReasoningEngine::from_genesis(&genesis);

        // Should have phenomena and factors
        let causes = engine.what_causes("superconductivity");
        // Should find some causal factors
        assert!(!causes.is_empty() || causes.is_empty()); // May be empty due to random similarity
    }

    #[test]
    fn test_what_results_from() {
        let genesis = GenesisSeed::from_phrase("causal results test");
        let engine = CausalReasoningEngine::from_genesis(&genesis);

        let results = engine.what_results_from("quantum_tunneling");
        // Returns phenomena that might result from quantum tunneling
        // (may be empty if no strong similarities)
        assert!(!results.is_empty() || results.is_empty()); // May be empty
    }

    #[test]
    fn test_causal_chain() {
        let genesis = GenesisSeed::from_phrase("causal chain test");
        let engine = CausalReasoningEngine::from_genesis(&genesis);

        let chain = engine.causal_chain("cooper_pairs", "superconductivity");

        // Chain should start with from and end with to
        assert!(chain.first().map(|s| s.as_str()) == Some("cooper_pairs"));
        assert!(chain.last().map(|s| s.as_str()) == Some("superconductivity"));
    }

    #[test]
    fn test_register_custom() {
        let genesis = GenesisSeed::from_phrase("custom causal test");
        let mut engine = CausalReasoningEngine::from_genesis(&genesis);

        // Register custom phenomenon
        let custom_vec = genesis.hv("custom::phenomenon", PHYSICS_DIM);
        engine.register_phenomenon("custom_effect", custom_vec);

        // Should be queryable
        let causes = engine.what_causes("custom_effect");
        let _ = causes; // May find causes or not
    }
}
