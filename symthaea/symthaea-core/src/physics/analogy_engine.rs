// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cross-Domain Analogy Engine
//!
//! **Status**: Exploratory module — HDC encodings for cross-domain analogy discovery via HDC vector similarity.
//!
//! This module discovers unexpected connections between physics domains
//! by finding concepts with similar HDC vector representations.
//!
//! ## Key Insight
//!
//! In HDC, similarity between vectors indicates conceptual similarity.
//! When concepts from different domains have high similarity, it suggests
//! an underlying structural or mathematical analogy.
//!
//! ## Examples
//!
//! - **Vortex shedding ↔ Action potential**: Both are threshold-triggered oscillations
//! - **BEC ↔ Laser**: Both involve macroscopic quantum coherence
//! - **Phase transition ↔ Market crash**: Both exhibit critical phenomena
//!
//! ## Applications
//!
//! - Transfer learning between physics domains
//! - Discovery of universal principles
//! - Creative problem-solving via analogy

use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A concept from a physics domain
#[derive(Debug, Clone)]
pub struct PhysicsConcept {
    /// Name of the concept
    pub name: String,
    /// Domain it belongs to
    pub domain: String,
    /// Brief description
    pub description: String,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// An analogy between two concepts
#[derive(Debug, Clone)]
pub struct Analogy {
    /// First concept
    pub concept_a: String,
    /// Second concept
    pub concept_b: String,
    /// Domain of first concept
    pub domain_a: String,
    /// Domain of second concept
    pub domain_b: String,
    /// Similarity score
    pub similarity: f64,
    /// Explanation of the analogy
    pub explanation: String,
}

/// Category of structural analogy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalogyType {
    Mathematical,   // Same equations
    Symmetry,       // Similar symmetries
    ScaleInvariant, // Same at different scales
    Dynamical,      // Similar dynamics
    Topological,    // Same topology
    Emergent,       // Similar emergence patterns
}

/// Cross-Domain Analogy Engine
pub struct AnalogyEngine {
    genesis: GenesisSeed,
    /// Registered concepts by name
    concepts: HashMap<String, PhysicsConcept>,
    /// Known analogy patterns
    patterns: Vec<AnalogyPattern>,
}

/// A known analogy pattern
#[derive(Debug, Clone)]
pub struct AnalogyPattern {
    pub name: String,
    pub analogy_type: AnalogyType,
    pub marker: ContinuousHV,
    pub examples: Vec<(String, String)>,
}

impl AnalogyEngine {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        let mut engine = Self {
            genesis: genesis.clone(),
            concepts: HashMap::new(),
            patterns: Vec::new(),
        };

        // Register known analogy patterns
        engine.register_patterns();

        // Pre-populate with concepts from various domains
        engine.populate_concepts();

        engine
    }

    fn register_patterns(&mut self) {
        // Wave-like phenomena
        self.patterns.push(AnalogyPattern {
            name: "Wave Equation".to_string(),
            analogy_type: AnalogyType::Mathematical,
            marker: self.genesis.hv("pattern::wave_equation", PHYSICS_DIM),
            examples: vec![
                (
                    "electromagnetic_wave".to_string(),
                    "acoustic_wave".to_string(),
                ),
                ("gravitational_wave".to_string(), "seismic_wave".to_string()),
            ],
        });

        // Diffusion-like phenomena
        self.patterns.push(AnalogyPattern {
            name: "Diffusion".to_string(),
            analogy_type: AnalogyType::Mathematical,
            marker: self.genesis.hv("pattern::diffusion", PHYSICS_DIM),
            examples: vec![
                (
                    "heat_conduction".to_string(),
                    "molecular_diffusion".to_string(),
                ),
                ("charge_diffusion".to_string(), "neuron_signal".to_string()),
            ],
        });

        // Oscillation/instability
        self.patterns.push(AnalogyPattern {
            name: "Threshold Oscillation".to_string(),
            analogy_type: AnalogyType::Dynamical,
            marker: self
                .genesis
                .hv("pattern::threshold_oscillation", PHYSICS_DIM),
            examples: vec![
                (
                    "vortex_shedding".to_string(),
                    "action_potential".to_string(),
                ),
                ("laser_mode_locking".to_string(), "heartbeat".to_string()),
            ],
        });

        // Symmetry breaking
        self.patterns.push(AnalogyPattern {
            name: "Spontaneous Symmetry Breaking".to_string(),
            analogy_type: AnalogyType::Symmetry,
            marker: self.genesis.hv("pattern::ssb", PHYSICS_DIM),
            examples: vec![
                ("ferromagnetism".to_string(), "higgs_mechanism".to_string()),
                ("superconductivity".to_string(), "superfluidity".to_string()),
            ],
        });

        // Macroscopic quantum coherence
        self.patterns.push(AnalogyPattern {
            name: "Quantum Coherence".to_string(),
            analogy_type: AnalogyType::Emergent,
            marker: self
                .genesis
                .hv("pattern::macroscopic_coherence", PHYSICS_DIM),
            examples: vec![
                ("bec".to_string(), "laser".to_string()),
                ("superconductor".to_string(), "superfluid".to_string()),
            ],
        });

        // Scale invariance / power laws
        self.patterns.push(AnalogyPattern {
            name: "Scale Invariance".to_string(),
            analogy_type: AnalogyType::ScaleInvariant,
            marker: self.genesis.hv("pattern::scale_invariance", PHYSICS_DIM),
            examples: vec![
                (
                    "critical_point".to_string(),
                    "turbulence_cascade".to_string(),
                ),
                ("fractal".to_string(), "renormalization".to_string()),
            ],
        });

        // Topological protection
        self.patterns.push(AnalogyPattern {
            name: "Topological Protection".to_string(),
            analogy_type: AnalogyType::Topological,
            marker: self.genesis.hv("pattern::topological", PHYSICS_DIM),
            examples: vec![
                (
                    "quantum_hall".to_string(),
                    "topological_insulator".to_string(),
                ),
                ("vortex".to_string(), "magnetic_monopole".to_string()),
            ],
        });
    }

    fn populate_concepts(&mut self) {
        // Fluid dynamics concepts
        self.register_concept(
            "vortex_shedding",
            "fluid_dynamics",
            "Periodic vortex formation behind bluff body",
            self.genesis.hv("fluid::vortex_shedding", PHYSICS_DIM),
        );

        self.register_concept(
            "turbulence_cascade",
            "fluid_dynamics",
            "Energy transfer from large to small scales",
            self.genesis.hv("fluid::turbulence_cascade", PHYSICS_DIM),
        );

        self.register_concept(
            "bernoulli_flow",
            "fluid_dynamics",
            "Pressure decreases with velocity",
            self.genesis.hv("fluid::bernoulli", PHYSICS_DIM),
        );

        // Electromagnetism concepts
        self.register_concept(
            "electromagnetic_wave",
            "electromagnetism",
            "Self-propagating EM disturbance",
            self.genesis.hv("em::wave", PHYSICS_DIM),
        );

        self.register_concept(
            "faraday_induction",
            "electromagnetism",
            "Changing magnetic flux induces EMF",
            self.genesis.hv("em::faraday", PHYSICS_DIM),
        );

        // Thermodynamics concepts
        self.register_concept(
            "heat_conduction",
            "thermodynamics",
            "Energy transfer via temperature gradient",
            self.genesis.hv("thermo::conduction", PHYSICS_DIM),
        );

        self.register_concept(
            "entropy_production",
            "thermodynamics",
            "Irreversible increase of entropy",
            self.genesis.hv("thermo::entropy_production", PHYSICS_DIM),
        );

        self.register_concept(
            "critical_point",
            "thermodynamics",
            "End of phase coexistence curve",
            self.genesis.hv("thermo::critical_point", PHYSICS_DIM),
        );

        // Quantum concepts
        self.register_concept(
            "bec",
            "quantum",
            "Bose-Einstein condensation",
            self.genesis.hv("quantum::bec", PHYSICS_DIM),
        );

        self.register_concept(
            "superconductivity",
            "quantum",
            "Zero resistance below Tc",
            self.genesis.hv("quantum::superconductivity", PHYSICS_DIM),
        );

        self.register_concept(
            "superfluidity",
            "quantum",
            "Frictionless flow below Tc",
            self.genesis.hv("quantum::superfluidity", PHYSICS_DIM),
        );

        self.register_concept(
            "quantum_tunneling",
            "quantum",
            "Particle traverses classically forbidden barrier",
            self.genesis.hv("quantum::tunneling", PHYSICS_DIM),
        );

        self.register_concept(
            "quantum_hall",
            "quantum",
            "Quantized Hall conductance",
            self.genesis.hv("quantum::hall", PHYSICS_DIM),
        );

        // Particle physics concepts
        self.register_concept(
            "higgs_mechanism",
            "particle",
            "Symmetry breaking gives mass",
            self.genesis.hv("particle::higgs", PHYSICS_DIM),
        );

        self.register_concept(
            "asymptotic_freedom",
            "particle",
            "Coupling decreases at high energy",
            self.genesis.hv("particle::asymptotic_freedom", PHYSICS_DIM),
        );

        // Condensed matter concepts
        self.register_concept(
            "ferromagnetism",
            "condensed_matter",
            "Spontaneous magnetic ordering",
            self.genesis.hv("cm::ferromagnetism", PHYSICS_DIM),
        );

        self.register_concept(
            "topological_insulator",
            "condensed_matter",
            "Insulating bulk with conducting surface",
            self.genesis.hv("cm::topological_insulator", PHYSICS_DIM),
        );

        // Neuroscience concepts
        self.register_concept(
            "action_potential",
            "neuroscience",
            "All-or-none neural signal",
            self.genesis.hv("neuro::action_potential", PHYSICS_DIM),
        );

        self.register_concept(
            "neural_synchrony",
            "neuroscience",
            "Coordinated firing of neurons",
            self.genesis.hv("neuro::synchrony", PHYSICS_DIM),
        );

        // Optics concepts
        self.register_concept(
            "laser",
            "optics",
            "Coherent light amplification",
            self.genesis.hv("optics::laser", PHYSICS_DIM),
        );

        self.register_concept(
            "soliton",
            "optics",
            "Self-reinforcing solitary wave",
            self.genesis.hv("optics::soliton", PHYSICS_DIM),
        );

        // Acoustics concepts
        self.register_concept(
            "acoustic_wave",
            "acoustics",
            "Mechanical wave in medium",
            self.genesis.hv("acoustics::wave", PHYSICS_DIM),
        );

        self.register_concept(
            "resonance",
            "acoustics",
            "Frequency matching amplification",
            self.genesis.hv("acoustics::resonance", PHYSICS_DIM),
        );

        // Geophysics concepts
        self.register_concept(
            "seismic_wave",
            "geophysics",
            "Elastic wave in Earth",
            self.genesis.hv("geo::seismic", PHYSICS_DIM),
        );

        self.register_concept(
            "mantle_convection",
            "geophysics",
            "Thermal convection in Earth's mantle",
            self.genesis.hv("geo::convection", PHYSICS_DIM),
        );

        // Cosmology concepts
        self.register_concept(
            "inflation",
            "cosmology",
            "Exponential early universe expansion",
            self.genesis.hv("cosmo::inflation", PHYSICS_DIM),
        );

        self.register_concept(
            "gravitational_wave",
            "cosmology",
            "Ripples in spacetime",
            self.genesis.hv("cosmo::gw", PHYSICS_DIM),
        );

        // Biophysics concepts
        self.register_concept(
            "protein_folding",
            "biophysics",
            "Polypeptide finding native state",
            self.genesis.hv("bio::folding", PHYSICS_DIM),
        );

        self.register_concept(
            "molecular_diffusion",
            "biophysics",
            "Random walk of molecules",
            self.genesis.hv("bio::diffusion", PHYSICS_DIM),
        );
    }

    /// Register a physics concept
    pub fn register_concept(
        &mut self,
        name: &str,
        domain: &str,
        description: &str,
        vector: ContinuousHV,
    ) {
        self.concepts.insert(
            name.to_string(),
            PhysicsConcept {
                name: name.to_string(),
                domain: domain.to_string(),
                description: description.to_string(),
                vector,
            },
        );
    }

    /// Find analogies for a given concept
    pub fn find_analogies(&self, concept_name: &str, min_similarity: f64) -> Vec<Analogy> {
        let concept = match self.concepts.get(concept_name) {
            Some(c) => c,
            None => return vec![],
        };

        let mut analogies: Vec<Analogy> = self
            .concepts
            .iter()
            .filter(|(name, other)| *name != concept_name && other.domain != concept.domain)
            .map(|(_, other)| {
                let sim = concept.vector.similarity(&other.vector) as f64;
                let explanation = self.explain_analogy(concept, other, sim);
                Analogy {
                    concept_a: concept.name.clone(),
                    concept_b: other.name.clone(),
                    domain_a: concept.domain.clone(),
                    domain_b: other.domain.clone(),
                    similarity: sim,
                    explanation,
                }
            })
            .filter(|a| a.similarity.abs() >= min_similarity)
            .collect();

        analogies.sort_by(|a, b| {
            b.similarity
                .abs()
                .partial_cmp(&a.similarity.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        analogies
    }

    /// Find all cross-domain analogies above threshold
    pub fn discover_all_analogies(&self, min_similarity: f64) -> Vec<Analogy> {
        let mut analogies = Vec::new();
        let names: Vec<_> = self.concepts.keys().cloned().collect();

        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let c1 = &self.concepts[&names[i]];
                let c2 = &self.concepts[&names[j]];

                // Only cross-domain analogies
                if c1.domain == c2.domain {
                    continue;
                }

                let sim = c1.vector.similarity(&c2.vector) as f64;
                if sim.abs() >= min_similarity {
                    let explanation = self.explain_analogy(c1, c2, sim);
                    analogies.push(Analogy {
                        concept_a: c1.name.clone(),
                        concept_b: c2.name.clone(),
                        domain_a: c1.domain.clone(),
                        domain_b: c2.domain.clone(),
                        similarity: sim,
                        explanation,
                    });
                }
            }
        }

        analogies.sort_by(|a, b| {
            b.similarity
                .abs()
                .partial_cmp(&a.similarity.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        analogies
    }

    /// Generate explanation for an analogy
    fn explain_analogy(&self, c1: &PhysicsConcept, c2: &PhysicsConcept, similarity: f64) -> String {
        // Check against known patterns
        for pattern in &self.patterns {
            let p1_sim = c1.vector.similarity(&pattern.marker).abs();
            let p2_sim = c2.vector.similarity(&pattern.marker).abs();

            if p1_sim > 0.1 && p2_sim > 0.1 {
                return format!(
                    "Both exhibit {} pattern ({:?}): {} and {} share underlying structure",
                    pattern.name, pattern.analogy_type, c1.name, c2.name
                );
            }
        }

        // Generic explanation
        if similarity > 0.3 {
            format!(
                "Strong structural similarity between {} and {}",
                c1.name, c2.name
            )
        } else if similarity > 0.1 {
            format!("Moderate analogy between {} and {}", c1.name, c2.name)
        } else {
            format!("Weak connection between {} and {}", c1.name, c2.name)
        }
    }

    /// Find concepts matching a pattern
    pub fn find_by_pattern(&self, pattern_name: &str) -> Vec<&PhysicsConcept> {
        let pattern = match self.patterns.iter().find(|p| p.name == pattern_name) {
            Some(p) => p,
            None => return vec![],
        };

        let mut matches: Vec<_> = self
            .concepts
            .values()
            .filter(|c| c.vector.similarity(&pattern.marker).abs() > 0.1)
            .collect();

        matches.sort_by(|a, b| {
            let sim_a = a.vector.similarity(&pattern.marker).abs();
            let sim_b = b.vector.similarity(&pattern.marker).abs();
            sim_b
                .partial_cmp(&sim_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches
    }

    /// Compose concepts to explore hypothetical phenomena
    pub fn compose_concepts(&self, names: &[&str]) -> Option<ContinuousHV> {
        let vectors: Vec<_> = names
            .iter()
            .filter_map(|n| self.concepts.get(*n))
            .map(|c| &c.vector)
            .collect();

        if vectors.is_empty() {
            None
        } else if vectors.len() == 1 {
            Some(vectors[0].clone())
        } else {
            let refs: Vec<_> = vectors.to_vec();
            Some(ContinuousHV::bundle(&refs))
        }
    }

    /// Find the nearest existing concept to a composed vector
    pub fn nearest_concept(&self, target: &ContinuousHV) -> Option<(&PhysicsConcept, f64)> {
        self.concepts
            .values()
            .map(|c| (c, c.vector.similarity(target) as f64))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// List all registered patterns
    pub fn list_patterns(&self) -> Vec<&str> {
        self.patterns.iter().map(|p| p.name.as_str()).collect()
    }

    /// List all concepts in a domain
    pub fn list_domain(&self, domain: &str) -> Vec<&PhysicsConcept> {
        self.concepts
            .values()
            .filter(|c| c.domain == domain)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> AnalogyEngine {
        let genesis = GenesisSeed::from_phrase("analogy test");
        AnalogyEngine::from_genesis(&genesis)
    }

    #[test]
    fn test_engine_creation() {
        let engine = setup();
        assert!(!engine.concepts.is_empty());
        assert!(!engine.patterns.is_empty());
    }

    #[test]
    fn test_concept_registration() {
        let mut engine = setup();

        engine.register_concept(
            "test_concept",
            "test_domain",
            "A test concept",
            engine.genesis.hv("test::concept", PHYSICS_DIM),
        );

        assert!(engine.concepts.contains_key("test_concept"));
    }

    #[test]
    fn test_find_analogies() {
        let engine = setup();

        let analogies = engine.find_analogies("bec", 0.0);

        // Should find some analogies (may vary by genesis seed)
        // At minimum, should run without error
        assert!(analogies.iter().all(|a| a.concept_a == "bec"));
    }

    #[test]
    fn test_discover_all() {
        let engine = setup();

        let all_analogies = engine.discover_all_analogies(0.05);

        // Should find some cross-domain analogies
        // All should be between different domains
        for analogy in &all_analogies {
            assert_ne!(analogy.domain_a, analogy.domain_b);
        }
    }

    #[test]
    fn test_compose_concepts() {
        let engine = setup();

        // Compose two concepts
        let composed = engine.compose_concepts(&["bec", "laser"]);

        assert!(composed.is_some());
        assert!(composed.unwrap().norm() > 0.0);
    }

    #[test]
    fn test_nearest_concept() {
        let engine = setup();

        // The laser concept should be nearest to itself
        if let Some(laser) = engine.concepts.get("laser") {
            let (nearest, sim) = engine.nearest_concept(&laser.vector).unwrap();
            assert_eq!(nearest.name, "laser");
            assert!((sim - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_list_patterns() {
        let engine = setup();

        let patterns = engine.list_patterns();

        assert!(patterns.contains(&"Wave Equation"));
        assert!(patterns.contains(&"Spontaneous Symmetry Breaking"));
    }

    #[test]
    fn test_list_domain() {
        let engine = setup();

        let quantum = engine.list_domain("quantum");

        // Should have quantum concepts
        assert!(!quantum.is_empty());
        for concept in quantum {
            assert_eq!(concept.domain, "quantum");
        }
    }

    #[test]
    fn test_find_by_pattern() {
        let engine = setup();

        let wave_concepts = engine.find_by_pattern("Wave Equation");

        // May find wave-related concepts (depends on similarity)
        // At minimum should not crash
        let _ = wave_concepts; // May be empty
    }
}
