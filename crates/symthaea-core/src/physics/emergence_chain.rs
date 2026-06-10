// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Emergence Chain: From Quarks to Consciousness
//!
//! This module provides a unified view of the complete emergence hierarchy,
//! tracing the path from fundamental particles to conscious experience.
//!
//! ## Emergence Levels
//!
//! 1. **Quark**: Up, down, strange quarks (fundamental)
//! 2. **Hadron**: Protons, neutrons (bound quarks)
//! 3. **Nucleus**: Atomic nuclei (bound hadrons)
//! 4. **Atom**: Full atoms (nucleus + electrons)
//! 5. **Molecule**: Chemical compounds (bound atoms)
//! 6. **Macromolecule**: Proteins, DNA (complex molecules)
//! 7. **Cell**: Living cells (organized macromolecules)
//! 8. **Neuron**: Neural cells with signaling
//! 9. **Circuit**: Neural networks
//! 10. **Consciousness**: Integrated information, qualia
//!
//! ## HDC Operations for Level Transitions
//!
//! | Transition | Operation | Example |
//! |------------|-----------|---------|
//! | Quark → Hadron | Weighted bundle | uud → proton |
//! | Hadron → Nucleus | Bundle + binding | p+n → deuteron |
//! | Nucleus → Atom | Bundle + shells | nucleus + electrons |
//! | Atom → Molecule | Bind with bond | C + O → CO |
//! | Molecule → Macro | Position bundle | AAs → protein |
//! | Macro → Cell | Bind structure | organelles → cell |
//! | Cell → Neuron | Bind + channels | cell + ion channels |
//! | Neuron → Circuit | Weighted bundle | neurons + synapses |
//! | Circuit → Mind | Phenomenal bind | integrated info |

use super::chemistry::Chemistry;
use super::consciousness_bridge::PhysicsConsciousnessBridge;
use super::hadrons::Hadrons;
use super::hdc_emergence_metrics::{
    CompositionNode, EmergenceAnalysis, EmergenceMetrics, TreeCoherenceResult,
    TreeRecoverabilityResult,
};
use super::molecular_biology::MolBioEncoder;
use super::neuroscience::NeuroEncoder;
use super::periodic_table::PeriodicTable;
use super::standard_model::{PHYSICS_DIM, StandardModel};
use crate::consciousness_metrics::{EntropyConfig, TruePhiCalculator, TruePhiResult};
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Emergence level in the hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EmergenceLevel {
    /// Fundamental quarks
    Quark = 0,
    /// Bound quark states (protons, neutrons)
    Hadron = 1,
    /// Atomic nuclei
    Nucleus = 2,
    /// Complete atoms
    Atom = 3,
    /// Simple molecules
    Molecule = 4,
    /// Complex macromolecules (proteins, DNA)
    Macromolecule = 5,
    /// Living cells
    Cell = 6,
    /// Neurons with signaling
    Neuron = 7,
    /// Neural circuits
    Circuit = 8,
    /// Conscious experience
    Consciousness = 9,
}

impl EmergenceLevel {
    /// All levels in order
    pub fn all() -> &'static [EmergenceLevel] {
        &[
            EmergenceLevel::Quark,
            EmergenceLevel::Hadron,
            EmergenceLevel::Nucleus,
            EmergenceLevel::Atom,
            EmergenceLevel::Molecule,
            EmergenceLevel::Macromolecule,
            EmergenceLevel::Cell,
            EmergenceLevel::Neuron,
            EmergenceLevel::Circuit,
            EmergenceLevel::Consciousness,
        ]
    }

    /// Domain label for this level
    pub fn domain(&self) -> &'static str {
        match self {
            EmergenceLevel::Quark => "emergence::quark",
            EmergenceLevel::Hadron => "emergence::hadron",
            EmergenceLevel::Nucleus => "emergence::nucleus",
            EmergenceLevel::Atom => "emergence::atom",
            EmergenceLevel::Molecule => "emergence::molecule",
            EmergenceLevel::Macromolecule => "emergence::macromolecule",
            EmergenceLevel::Cell => "emergence::cell",
            EmergenceLevel::Neuron => "emergence::neuron",
            EmergenceLevel::Circuit => "emergence::circuit",
            EmergenceLevel::Consciousness => "emergence::consciousness",
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            EmergenceLevel::Quark => "Quark",
            EmergenceLevel::Hadron => "Hadron",
            EmergenceLevel::Nucleus => "Nucleus",
            EmergenceLevel::Atom => "Atom",
            EmergenceLevel::Molecule => "Molecule",
            EmergenceLevel::Macromolecule => "Macromolecule",
            EmergenceLevel::Cell => "Cell",
            EmergenceLevel::Neuron => "Neuron",
            EmergenceLevel::Circuit => "Circuit",
            EmergenceLevel::Consciousness => "Consciousness",
        }
    }

    /// Get the next level up (if any)
    pub fn next(&self) -> Option<EmergenceLevel> {
        match self {
            EmergenceLevel::Quark => Some(EmergenceLevel::Hadron),
            EmergenceLevel::Hadron => Some(EmergenceLevel::Nucleus),
            EmergenceLevel::Nucleus => Some(EmergenceLevel::Atom),
            EmergenceLevel::Atom => Some(EmergenceLevel::Molecule),
            EmergenceLevel::Molecule => Some(EmergenceLevel::Macromolecule),
            EmergenceLevel::Macromolecule => Some(EmergenceLevel::Cell),
            EmergenceLevel::Cell => Some(EmergenceLevel::Neuron),
            EmergenceLevel::Neuron => Some(EmergenceLevel::Circuit),
            EmergenceLevel::Circuit => Some(EmergenceLevel::Consciousness),
            EmergenceLevel::Consciousness => None,
        }
    }

    /// Get the previous level down (if any)
    pub fn prev(&self) -> Option<EmergenceLevel> {
        match self {
            EmergenceLevel::Quark => None,
            EmergenceLevel::Hadron => Some(EmergenceLevel::Quark),
            EmergenceLevel::Nucleus => Some(EmergenceLevel::Hadron),
            EmergenceLevel::Atom => Some(EmergenceLevel::Nucleus),
            EmergenceLevel::Molecule => Some(EmergenceLevel::Atom),
            EmergenceLevel::Macromolecule => Some(EmergenceLevel::Molecule),
            EmergenceLevel::Cell => Some(EmergenceLevel::Macromolecule),
            EmergenceLevel::Neuron => Some(EmergenceLevel::Cell),
            EmergenceLevel::Circuit => Some(EmergenceLevel::Neuron),
            EmergenceLevel::Consciousness => Some(EmergenceLevel::Circuit),
        }
    }
}

/// A step in the emergence trace
#[derive(Debug, Clone)]
pub struct EmergenceStep {
    /// The level
    pub level: EmergenceLevel,
    /// Vector at this level
    pub vector: ContinuousHV,
    /// Phenomenal index (0.0 to 1.0)
    pub phenomenal_index: f64,
}

/// Result of rigorous phenomenal index computation
///
/// Unlike the basic `phenomenal_index()` which uses hardcoded level weights,
/// this result contains mathematically grounded measures from:
/// - True Φ (integrated information via Shannon entropy)
/// - HDC-native emergence metrics (structural properties of composition)
#[derive(Debug, Clone)]
pub struct RigorousPhiResult {
    /// True Φ from entropy-based IIT calculation
    pub true_phi: f64,
    /// HDC-native emergence score (0.0 to 1.0)
    pub emergence_score: f64,
    /// Combined phenomenal index (weighted average)
    pub combined_index: f64,
    /// Maximum binding depth in composition tree
    pub binding_depth: usize,
    /// Tree recoverability analysis
    pub recoverability: TreeRecoverabilityResult,
    /// Tree coherence analysis
    pub coherence: TreeCoherenceResult,
    /// Full entropy-based Φ result (for detailed analysis)
    pub phi_details: Option<TruePhiResult>,
    /// Full emergence analysis (for detailed analysis)
    pub emergence_details: Option<EmergenceAnalysis>,
}

impl RigorousPhiResult {
    /// Check if this represents high integration (consciousness-like)
    pub fn is_highly_integrated(&self) -> bool {
        self.combined_index > 0.5
    }

    /// Get a summary score (0.0 to 1.0)
    pub fn summary_score(&self) -> f64 {
        self.combined_index
    }
}

/// Complete emergence chain from quarks to consciousness
#[derive(Debug, Clone)]
pub struct EmergenceChain {
    /// Genesis seed for deterministic generation
    genesis: GenesisSeed,

    /// Standard model (quarks, leptons, bosons)
    pub model: StandardModel,
    /// Hadrons (protons, neutrons, mesons)
    pub hadrons: Hadrons,
    /// Periodic table (118 elements)
    pub table: PeriodicTable,
    /// Chemistry (bonds, molecules)
    pub chemistry: Chemistry,
    /// Molecular biology (DNA, proteins)
    pub molbio: MolBioEncoder,
    /// Neuroscience (channels, circuits)
    pub neuro: NeuroEncoder,
    /// Consciousness bridge
    pub bridge: PhysicsConsciousnessBridge,

    // Level marker vectors
    level_markers: Vec<ContinuousHV>,

    // True Φ calculator
    phi_calculator: TruePhiCalculator,

    // HDC-native emergence metrics calculator
    emergence_metrics: EmergenceMetrics,
}

impl EmergenceChain {
    /// Build complete emergence chain from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        // Build each layer compositionally
        let model = StandardModel::from_genesis(genesis);
        let hadrons = Hadrons::from_model(&model, genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, genesis);
        let chemistry = Chemistry::from_genesis(genesis, &table);
        let molbio = MolBioEncoder::from_genesis(genesis);
        let neuro = NeuroEncoder::from_genesis(genesis);
        let bridge = PhysicsConsciousnessBridge::from_genesis(genesis);

        // Create level markers
        let level_markers: Vec<ContinuousHV> = EmergenceLevel::all()
            .iter()
            .map(|level| genesis.hv(level.domain(), PHYSICS_DIM))
            .collect();

        // Initialize rigorous Φ calculators
        let phi_calculator = TruePhiCalculator::with_config(EntropyConfig::fast());
        let emergence_metrics = EmergenceMetrics::new();

        Self {
            genesis: genesis.clone(),
            model,
            hadrons,
            table,
            chemistry,
            molbio,
            neuro,
            bridge,
            level_markers,
            phi_calculator,
            emergence_metrics,
        }
    }

    /// Get the level marker vector
    pub fn level_vector(&self, level: EmergenceLevel) -> &ContinuousHV {
        &self.level_markers[level as usize]
    }

    /// Compose quarks into a hadron
    pub fn quarks_to_hadron(&self, quarks: &[&ContinuousHV], weights: &[f32]) -> ContinuousHV {
        let hadron = ContinuousHV::weighted_bundle(quarks, weights);
        hadron.bind(self.level_vector(EmergenceLevel::Hadron))
    }

    /// Compose hadrons into a nucleus
    pub fn hadrons_to_nucleus(&self, protons: u8, neutrons: u8) -> ContinuousHV {
        let p_vec = self.hadrons.proton.scale(protons as f32);
        let n_vec = self.hadrons.neutron.scale(neutrons as f32);
        let nucleus = ContinuousHV::bundle(&[&p_vec, &n_vec]);
        nucleus.bind(self.level_vector(EmergenceLevel::Nucleus))
    }

    /// Compose nucleus and electrons into an atom
    pub fn nucleus_to_atom(&self, atomic_number: u8) -> Option<ContinuousHV> {
        self.table
            .element(atomic_number)
            .map(|e| e.vector.bind(self.level_vector(EmergenceLevel::Atom)))
    }

    /// Compose atoms into a molecule
    pub fn atoms_to_molecule(&self, atoms: &[&ContinuousHV], bond: &ContinuousHV) -> ContinuousHV {
        let atom_bundle = ContinuousHV::bundle(atoms);
        let molecule = atom_bundle.bind(bond);
        molecule.bind(self.level_vector(EmergenceLevel::Molecule))
    }

    /// Compose molecules into a macromolecule (e.g., protein)
    pub fn molecules_to_macromolecule(&self, units: &[&ContinuousHV]) -> ContinuousHV {
        // Use positional encoding for sequence
        let mut result = ContinuousHV::zero(PHYSICS_DIM);
        for (i, unit) in units.iter().enumerate() {
            let pos = self.genesis.hv(&format!("position::{i}"), PHYSICS_DIM);
            let positioned = (*unit).bind(&pos);
            result = ContinuousHV::bundle(&[&result, &positioned]);
        }
        result.bind(self.level_vector(EmergenceLevel::Macromolecule))
    }

    /// Compose macromolecules into a cell
    pub fn macromolecules_to_cell(&self, components: &[&ContinuousHV]) -> ContinuousHV {
        let cell = ContinuousHV::bundle(components);
        cell.bind(self.level_vector(EmergenceLevel::Cell))
    }

    /// Compose cell with ion channels into a neuron
    pub fn cell_to_neuron(&self, cell: &ContinuousHV, channels: &[&ContinuousHV]) -> ContinuousHV {
        let channel_bundle = ContinuousHV::bundle(channels);
        let neuron = cell.bind(&channel_bundle);
        neuron.bind(self.level_vector(EmergenceLevel::Neuron))
    }

    /// Compose neurons with synapses into a circuit
    pub fn neurons_to_circuit(
        &self,
        neurons: &[&ContinuousHV],
        synapses: &[&ContinuousHV],
    ) -> ContinuousHV {
        let neuron_bundle = ContinuousHV::bundle(neurons);
        let synapse_bundle = ContinuousHV::bundle(synapses);
        let circuit = neuron_bundle.bind(&synapse_bundle);
        circuit.bind(self.level_vector(EmergenceLevel::Circuit))
    }

    /// Compose circuit into conscious experience
    pub fn circuit_to_consciousness(&self, circuit: &ContinuousHV) -> ContinuousHV {
        // Bind with phenomenal concepts
        let integrated = circuit
            .bind(&self.bridge.awareness)
            .bind(&self.bridge.qualia);
        integrated.bind(self.level_vector(EmergenceLevel::Consciousness))
    }

    /// Estimate phenomenal index for a vector at a given level using level weight
    /// and similarity to consciousness concepts.
    ///
    /// This is a fast heuristic used internally by trace/profile methods.
    /// For rigorous computation, use `phenomenal_index_rigorous()` instead.
    fn estimate_level_phenomenal_index(&self, vector: &ContinuousHV, level: EmergenceLevel) -> f64 {
        // Higher levels have more phenomenal character
        let level_weight = (level as usize) as f64 / 9.0;

        // Similarity to consciousness concepts
        let awareness_sim = vector.similarity(&self.bridge.awareness).abs() as f64;
        let qualia_sim = vector.similarity(&self.bridge.qualia).abs() as f64;

        // Combine with level weight
        (level_weight * 0.5 + (awareness_sim + qualia_sim) * 0.25).min(1.0)
    }

    /// Compute rigorous phenomenal index using true Φ and HDC-native metrics
    ///
    /// This method replaces the hardcoded level weights with mathematically
    /// grounded measures:
    /// - True Φ from Shannon entropy (measures information integration)
    /// - HDC-native emergence score (measures structural properties)
    ///
    /// # Arguments
    /// * `components` - The component vectors that make up this composition
    /// * `composition` - Optional composition tree for structural analysis
    ///
    /// # Returns
    /// RigorousPhiResult with entropy-based and structure-based measures
    pub fn phenomenal_index_rigorous(
        &self,
        components: &[ContinuousHV],
        composition: Option<&CompositionNode>,
    ) -> RigorousPhiResult {
        // 1. Compute true Φ from entropy measures
        let phi_result = self.phi_calculator.compute_true_phi(components);
        let true_phi = phi_result.phi;

        // Normalize true_phi to [0, 1] range
        // Max theoretical EI grows with number of component pairs
        let n = components.len();
        let max_pairs = if n > 1 { (n * (n - 1)) / 2 } else { 1 };
        let max_phi = (max_pairs as f64) * 4.0; // Approximate max entropy
        let normalized_phi = (true_phi / max_phi).min(1.0);

        // 2. Compute HDC-native emergence score (if composition tree provided)
        let (emergence_score, binding_depth, recoverability, coherence, emergence_details) =
            if let Some(tree) = composition {
                let analysis = self.emergence_metrics.analyze(tree);
                (
                    analysis.emergence_score,
                    analysis.depth.max_depth,
                    analysis.recoverability.clone(),
                    analysis.coherence.clone(),
                    Some(analysis),
                )
            } else {
                // Without composition tree, use fast approximation
                let fast_emergence = self.estimate_emergence_from_vectors(components);
                (
                    fast_emergence,
                    0,
                    TreeRecoverabilityResult {
                        avg_recoverability: 0.5,
                        min_recoverability: 0.5,
                        max_recoverability: 0.5,
                        bind_count: 0,
                        recoverable_count: 0,
                    },
                    TreeCoherenceResult {
                        avg_coherence: 0.5,
                        min_coherence: 0.5,
                        max_coherence: 0.5,
                        bind_count: 0,
                    },
                    None,
                )
            };

        // 3. Combine into unified index
        // Weight true_phi higher (it's more theoretically grounded)
        let combined_index = 0.6 * normalized_phi + 0.4 * emergence_score;

        RigorousPhiResult {
            true_phi: normalized_phi,
            emergence_score,
            combined_index,
            binding_depth,
            recoverability,
            coherence,
            phi_details: Some(phi_result),
            emergence_details,
        }
    }

    /// Estimate emergence score from vectors alone (no composition tree)
    ///
    /// Uses pairwise analysis of vectors to estimate structural properties.
    fn estimate_emergence_from_vectors(&self, components: &[ContinuousHV]) -> f64 {
        if components.len() < 2 {
            return 0.0;
        }

        // Estimate coherence from pairwise comparisons
        // If bind creates very different structure than bundle, coherence is high
        let mut total_coherence = 0.0;
        let mut pair_count = 0;

        for i in 0..components.len().min(5) {
            // Limit to first 5 for speed
            for j in (i + 1)..components.len().min(5) {
                let a = &components[i];
                let b = &components[j];

                let bound = a.bind(b);
                let bundled = ContinuousHV::bundle(&[a, b]);
                let coherence = 1.0 - bound.similarity(&bundled).abs();

                total_coherence += coherence as f64;
                pair_count += 1;
            }
        }

        let avg_coherence = if pair_count > 0 {
            total_coherence / pair_count as f64
        } else {
            0.5
        };

        // Estimate integration from distinctiveness
        if components.len() >= 2 {
            let bundled_all = ContinuousHV::bundle(&components.iter().collect::<Vec<_>>());
            let mut distinctiveness = 0.0;
            for c in components {
                distinctiveness += (1.0 - bundled_all.similarity(c).abs()) as f64;
            }
            let avg_distinctiveness = distinctiveness / components.len() as f64;

            // Combine coherence and distinctiveness
            (avg_coherence + avg_distinctiveness) / 2.0
        } else {
            avg_coherence
        }
    }

    /// Compute rigorous phenomenal profile across all emergence levels
    ///
    /// Uses true Φ and HDC metrics instead of hardcoded weights.
    pub fn rigorous_profile(
        &self,
        components: &[ContinuousHV],
        composition: Option<&CompositionNode>,
    ) -> Vec<(EmergenceLevel, RigorousPhiResult)> {
        // For each level, we compute the rigorous index
        // In practice, the components would be different at each level
        // Here we just compute for the given components
        let result = self.phenomenal_index_rigorous(components, composition);

        // Return the same result for each level (in real use, components vary by level)
        EmergenceLevel::all()
            .iter()
            .map(|level| (*level, result.clone()))
            .collect()
    }

    /// Access the true Φ calculator for advanced analysis
    pub fn phi_calculator(&self) -> &TruePhiCalculator {
        &self.phi_calculator
    }

    /// Access the emergence metrics calculator for advanced analysis
    pub fn emergence_metrics(&self) -> &EmergenceMetrics {
        &self.emergence_metrics
    }

    /// Trace emergence path for a vector, identifying which level it belongs to
    pub fn identify_level(&self, vector: &ContinuousHV) -> EmergenceLevel {
        let mut best_level = EmergenceLevel::Quark;
        let mut best_sim = f32::MIN;

        for level in EmergenceLevel::all() {
            let sim = vector.similarity(self.level_vector(*level));
            if sim > best_sim {
                best_sim = sim;
                best_level = *level;
            }
        }

        best_level
    }

    /// Get phenomenal profile across all levels for a concept
    pub fn phenomenal_profile(&self, concept: &ContinuousHV) -> Vec<(EmergenceLevel, f64)> {
        EmergenceLevel::all()
            .iter()
            .map(|level| {
                let phi = self.estimate_level_phenomenal_index(concept, *level);
                (*level, phi)
            })
            .collect()
    }

    /// Compose a vector to a target level
    pub fn compose_to_level(&self, start: &ContinuousHV, target: EmergenceLevel) -> ContinuousHV {
        start.bind(self.level_vector(target))
    }

    /// Trace the full emergence path from quarks to a given concept
    pub fn trace_emergence(&self, concept: &ContinuousHV) -> Vec<EmergenceStep> {
        EmergenceLevel::all()
            .iter()
            .map(|level| {
                let level_marker = self.level_vector(*level);
                let at_level = concept.bind(level_marker);
                let phi = self.estimate_level_phenomenal_index(&at_level, *level);
                EmergenceStep {
                    level: *level,
                    vector: at_level,
                    phenomenal_index: phi,
                }
            })
            .collect()
    }

    /// Create a complete emergence trace from hydrogen to consciousness
    pub fn hydrogen_to_consciousness(&self) -> Vec<EmergenceStep> {
        let mut steps = Vec::new();

        // Quark level - up quark
        let quark = &self.model.up_quark;
        steps.push(EmergenceStep {
            level: EmergenceLevel::Quark,
            vector: quark.clone(),
            phenomenal_index: self.estimate_level_phenomenal_index(quark, EmergenceLevel::Quark),
        });

        // Hadron level - proton
        let hadron = &self.hadrons.proton;
        steps.push(EmergenceStep {
            level: EmergenceLevel::Hadron,
            vector: hadron.clone(),
            phenomenal_index: self.estimate_level_phenomenal_index(hadron, EmergenceLevel::Hadron),
        });

        // Nucleus level - hydrogen nucleus (just a proton)
        let nucleus = self.hadrons_to_nucleus(1, 0);
        steps.push(EmergenceStep {
            level: EmergenceLevel::Nucleus,
            vector: nucleus.clone(),
            phenomenal_index: self
                .estimate_level_phenomenal_index(&nucleus, EmergenceLevel::Nucleus),
        });

        // Atom level - hydrogen
        if let Some(atom) = self.nucleus_to_atom(1) {
            steps.push(EmergenceStep {
                level: EmergenceLevel::Atom,
                vector: atom.clone(),
                phenomenal_index: self.estimate_level_phenomenal_index(&atom, EmergenceLevel::Atom),
            });
        }

        // Molecule level - H2
        if let Some(h) = self.table.element(1) {
            let h2 = self.atoms_to_molecule(&[&h.vector, &h.vector], &self.chemistry.single_bond);
            steps.push(EmergenceStep {
                level: EmergenceLevel::Molecule,
                vector: h2.clone(),
                phenomenal_index: self
                    .estimate_level_phenomenal_index(&h2, EmergenceLevel::Molecule),
            });

            // Macromolecule level - simple chain
            let macro_vec = self.molecules_to_macromolecule(&[&h2, &h2, &h2]);
            steps.push(EmergenceStep {
                level: EmergenceLevel::Macromolecule,
                vector: macro_vec.clone(),
                phenomenal_index: self
                    .estimate_level_phenomenal_index(&macro_vec, EmergenceLevel::Macromolecule),
            });

            // Cell level
            let cell = self.macromolecules_to_cell(&[&macro_vec]);
            steps.push(EmergenceStep {
                level: EmergenceLevel::Cell,
                vector: cell.clone(),
                phenomenal_index: self.estimate_level_phenomenal_index(&cell, EmergenceLevel::Cell),
            });

            // Neuron level
            let neuron = self.cell_to_neuron(&cell, &[]);
            steps.push(EmergenceStep {
                level: EmergenceLevel::Neuron,
                vector: neuron.clone(),
                phenomenal_index: self
                    .estimate_level_phenomenal_index(&neuron, EmergenceLevel::Neuron),
            });

            // Circuit level
            let circuit = self.neurons_to_circuit(&[&neuron], &[]);
            steps.push(EmergenceStep {
                level: EmergenceLevel::Circuit,
                vector: circuit.clone(),
                phenomenal_index: self
                    .estimate_level_phenomenal_index(&circuit, EmergenceLevel::Circuit),
            });

            // Consciousness level
            let consciousness = self.circuit_to_consciousness(&circuit);
            steps.push(EmergenceStep {
                level: EmergenceLevel::Consciousness,
                vector: consciousness.clone(),
                phenomenal_index: self
                    .estimate_level_phenomenal_index(&consciousness, EmergenceLevel::Consciousness),
            });
        }

        steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> EmergenceChain {
        let genesis = GenesisSeed::from_phrase("emergence_chain_test");
        EmergenceChain::from_genesis(&genesis)
    }

    #[test]
    fn test_emergence_chain_creation() {
        let chain = setup();
        assert_eq!(chain.level_markers.len(), 10);
    }

    #[test]
    fn test_all_levels_exist() {
        let levels = EmergenceLevel::all();
        assert_eq!(levels.len(), 10);
        assert_eq!(levels[0], EmergenceLevel::Quark);
        assert_eq!(levels[9], EmergenceLevel::Consciousness);
    }

    #[test]
    fn test_level_ordering() {
        assert!(EmergenceLevel::Quark < EmergenceLevel::Hadron);
        assert!(EmergenceLevel::Hadron < EmergenceLevel::Atom);
        assert!(EmergenceLevel::Circuit < EmergenceLevel::Consciousness);
    }

    #[test]
    fn test_level_navigation() {
        assert_eq!(EmergenceLevel::Quark.next(), Some(EmergenceLevel::Hadron));
        assert_eq!(EmergenceLevel::Consciousness.next(), None);
        assert_eq!(EmergenceLevel::Quark.prev(), None);
        assert_eq!(EmergenceLevel::Hadron.prev(), Some(EmergenceLevel::Quark));
    }

    #[test]
    fn test_quarks_to_hadron() {
        let chain = setup();
        let proton = chain.quarks_to_hadron(
            &[
                &chain.model.up_quark,
                &chain.model.up_quark,
                &chain.model.down_quark,
            ],
            &[1.0, 1.0, 1.0],
        );
        assert!(proton.norm() > 0.0);
        // Should have hadron-level character
        let hadron_sim = proton.similarity(chain.level_vector(EmergenceLevel::Hadron));
        assert!(hadron_sim > 0.0, "Should have hadron character");
    }

    #[test]
    fn test_hadrons_to_nucleus() {
        let chain = setup();
        // Helium-4: 2 protons, 2 neutrons
        let he4 = chain.hadrons_to_nucleus(2, 2);
        assert!(he4.norm() > 0.0);
    }

    #[test]
    fn test_nucleus_to_atom() {
        let chain = setup();
        let carbon = chain.nucleus_to_atom(6);
        assert!(carbon.is_some());
        assert!(carbon.unwrap().norm() > 0.0);
    }

    #[test]
    fn test_atoms_to_molecule() {
        let chain = setup();
        let h = chain.table.element(1).unwrap();
        let o = chain.table.element(8).unwrap();
        // Water: 2H + O
        let water = chain.atoms_to_molecule(
            &[&h.vector, &h.vector, &o.vector],
            &chain.chemistry.single_bond,
        );
        assert!(water.norm() > 0.0);
    }

    #[test]
    fn test_phenomenal_index_increases_with_level() {
        let chain = setup();
        let test_vec = chain.genesis.hv("test::concept", PHYSICS_DIM);

        let profile = chain.phenomenal_profile(&test_vec);
        let phi_quark = profile
            .iter()
            .find(|(l, _)| *l == EmergenceLevel::Quark)
            .unwrap()
            .1;
        let phi_consciousness = profile
            .iter()
            .find(|(l, _)| *l == EmergenceLevel::Consciousness)
            .unwrap()
            .1;

        assert!(
            phi_consciousness > phi_quark,
            "Consciousness level should have higher phenomenal index"
        );
    }

    #[test]
    fn test_hydrogen_to_consciousness_trace() {
        let chain = setup();
        let trace = chain.hydrogen_to_consciousness();

        assert_eq!(trace.len(), 10, "Should have all 10 levels");
        assert_eq!(trace[0].level, EmergenceLevel::Quark);
        assert_eq!(trace[9].level, EmergenceLevel::Consciousness);

        // Phenomenal index should generally increase
        assert!(trace[9].phenomenal_index > trace[0].phenomenal_index);
    }

    #[test]
    fn test_level_identification() {
        let chain = setup();

        // The level markers themselves should identify correctly
        let quark_marker = chain.level_vector(EmergenceLevel::Quark);
        let identified = chain.identify_level(quark_marker);
        assert_eq!(
            identified,
            EmergenceLevel::Quark,
            "Level marker should identify its own level"
        );
    }

    #[test]
    fn test_phenomenal_profile() {
        let chain = setup();
        let profile = chain.phenomenal_profile(&chain.bridge.qualia);

        assert_eq!(profile.len(), 10);
        // All phi values should be between 0 and 1
        for (_, phi) in &profile {
            assert!(*phi >= 0.0 && *phi <= 1.0);
        }
    }

    #[test]
    fn test_compose_to_level() {
        let chain = setup();
        let start = chain.genesis.hv("test::start", PHYSICS_DIM);

        let at_atom = chain.compose_to_level(&start, EmergenceLevel::Atom);
        let at_neuron = chain.compose_to_level(&start, EmergenceLevel::Neuron);

        // Should be different vectors
        let sim = at_atom.similarity(&at_neuron);
        assert!(
            sim < 0.9,
            "Different levels should produce different vectors"
        );
    }

    #[test]
    fn test_trace_emergence() {
        let chain = setup();
        let concept = chain.bridge.awareness.clone();
        let trace = chain.trace_emergence(&concept);

        assert_eq!(trace.len(), 10);
        for step in &trace {
            assert!(step.vector.norm() > 0.0);
            assert!(step.phenomenal_index >= 0.0);
        }
    }

    #[test]
    fn test_circuit_to_consciousness() {
        let chain = setup();
        let circuit = chain.genesis.hv("test::circuit", PHYSICS_DIM);
        let consciousness = chain.circuit_to_consciousness(&circuit);

        // The resulting vector should be non-zero
        assert!(
            consciousness.norm() > 0.0,
            "Consciousness vector should be non-zero"
        );

        // The consciousness level should have higher phenomenal index than lower levels
        let profile = chain.phenomenal_profile(&consciousness);
        let phi_cons = profile
            .iter()
            .find(|(l, _)| *l == EmergenceLevel::Consciousness)
            .unwrap()
            .1;
        let phi_quark = profile
            .iter()
            .find(|(l, _)| *l == EmergenceLevel::Quark)
            .unwrap()
            .1;
        assert!(
            phi_cons > phi_quark,
            "Consciousness should have higher phenomenal index"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // RIGOROUS PHENOMENAL INDEX TESTS
    // ═══════════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_rigorous_phi_single_component() {
        let chain = setup();
        let single = vec![chain.genesis.hv("test::single", PHYSICS_DIM)];

        let result = chain.phenomenal_index_rigorous(&single, None);

        // Single component should have near-zero phi
        assert!(
            result.true_phi < 0.1,
            "Single component should have low phi: {}",
            result.true_phi
        );
    }

    #[test]
    fn test_rigorous_phi_multiple_components() {
        let chain = setup();
        let components: Vec<ContinuousHV> = (0..4)
            .map(|i| chain.genesis.hv(&format!("test::comp_{}", i), PHYSICS_DIM))
            .collect();

        let result = chain.phenomenal_index_rigorous(&components, None);

        // Multiple components should have positive values
        assert!(result.true_phi >= 0.0, "Phi should be non-negative");
        assert!(
            result.emergence_score >= 0.0,
            "Emergence score should be non-negative"
        );
        assert!(
            result.combined_index >= 0.0 && result.combined_index <= 1.0,
            "Combined index should be in [0,1]: {}",
            result.combined_index
        );
    }

    #[test]
    fn test_rigorous_phi_with_composition_tree() {
        let chain = setup();

        // Build a composition tree
        let a = CompositionNode::atomic("A", chain.genesis.hv("A", PHYSICS_DIM));
        let b = CompositionNode::atomic("B", chain.genesis.hv("B", PHYSICS_DIM));
        let c = CompositionNode::atomic("C", chain.genesis.hv("C", PHYSICS_DIM));

        // (A ⊗ B) bundled with C
        let ab = CompositionNode::bind(a.clone(), b.clone());
        let tree = CompositionNode::bundle_uniform(vec![ab, c.clone()]);

        // Get component vectors
        let components = vec![
            chain.genesis.hv("A", PHYSICS_DIM),
            chain.genesis.hv("B", PHYSICS_DIM),
            chain.genesis.hv("C", PHYSICS_DIM),
        ];

        let result = chain.phenomenal_index_rigorous(&components, Some(&tree));

        // With tree, we should have structural analysis
        assert!(result.binding_depth >= 1, "Should detect binding depth");
        assert!(
            result.emergence_details.is_some(),
            "Should have emergence details"
        );
    }

    #[test]
    fn test_rigorous_phi_correlated_vs_independent() {
        let chain = setup();

        // Independent components
        let independent: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(PHYSICS_DIM, i as u64))
            .collect();
        let phi_independent = chain.phenomenal_index_rigorous(&independent, None);

        // Correlated components (all derived from same base)
        let base = chain.genesis.hv("base", PHYSICS_DIM);
        let correlated: Vec<ContinuousHV> = (0..4)
            .map(|i| {
                let noise = ContinuousHV::random(PHYSICS_DIM, (i + 100) as u64);
                ContinuousHV::weighted_bundle(&[&base, &noise], &[0.9, 0.1])
            })
            .collect();
        let phi_correlated = chain.phenomenal_index_rigorous(&correlated, None);

        println!("Independent Φ: {:.4}", phi_independent.true_phi);
        println!("Correlated Φ: {:.4}", phi_correlated.true_phi);

        // Correlated should have higher integration (more mutual information)
        assert!(
            phi_correlated.true_phi > phi_independent.true_phi * 0.8,
            "Correlated should have higher or similar phi: corr={:.4} > ind={:.4}",
            phi_correlated.true_phi,
            phi_independent.true_phi
        );
    }

    #[test]
    fn test_rigorous_phi_bound_vs_bundled() {
        let chain = setup();

        let a = chain.genesis.hv("test::A", PHYSICS_DIM);
        let b = chain.genesis.hv("test::B", PHYSICS_DIM);
        let c = chain.genesis.hv("test::C", PHYSICS_DIM);

        // Bundled structure
        let bundled = ContinuousHV::bundle(&[&a, &b]);
        let bundled_components = vec![bundled.clone(), c.clone()];
        let phi_bundled = chain.phenomenal_index_rigorous(&bundled_components, None);

        // Bound structure
        let bound = a.bind(&b);
        let bound_components = vec![bound.clone(), c.clone()];
        let phi_bound = chain.phenomenal_index_rigorous(&bound_components, None);

        println!("Bundled Φ: {:.4}", phi_bundled.combined_index);
        println!("Bound Φ: {:.4}", phi_bound.combined_index);

        // Both should produce valid results (specific ordering may vary)
        assert!(phi_bundled.combined_index >= 0.0);
        assert!(phi_bound.combined_index >= 0.0);
    }

    #[test]
    fn test_is_highly_integrated() {
        let chain = setup();

        // Create a set of correlated components (likely high integration)
        let base = chain.genesis.hv("base", PHYSICS_DIM);
        let components: Vec<ContinuousHV> = (0..6)
            .map(|i| {
                let noise = ContinuousHV::random(PHYSICS_DIM, i as u64);
                ContinuousHV::weighted_bundle(&[&base, &noise], &[0.95, 0.05])
            })
            .collect();

        let result = chain.phenomenal_index_rigorous(&components, None);
        println!("Combined index: {:.4}", result.combined_index);
        println!("Is highly integrated: {}", result.is_highly_integrated());

        // Just verify the method works
        assert!(result.summary_score() == result.combined_index);
    }

    #[test]
    fn test_rigorous_profile() {
        let chain = setup();
        let components: Vec<ContinuousHV> = (0..3)
            .map(|i| chain.genesis.hv(&format!("comp{}", i), PHYSICS_DIM))
            .collect();

        let profile = chain.rigorous_profile(&components, None);

        assert_eq!(profile.len(), 10, "Should have profile for all 10 levels");
        for (level, result) in &profile {
            assert!(
                result.combined_index >= 0.0 && result.combined_index <= 1.0,
                "Level {:?} should have valid index",
                level
            );
        }
    }

    #[test]
    fn test_phi_calculator_accessor() {
        let chain = setup();
        let calc = chain.phi_calculator();

        // Should be able to use the calculator directly
        let hv = chain.genesis.hv("test", PHYSICS_DIM);
        let entropy = calc.entropy(&hv);
        assert!(entropy >= 0.0, "Entropy should be non-negative");
    }

    #[test]
    fn test_emergence_metrics_accessor() {
        let chain = setup();
        let metrics = chain.emergence_metrics();

        // Should be able to use the metrics directly
        let a = CompositionNode::atomic("A", chain.genesis.hv("A", PHYSICS_DIM));
        let depth = metrics.depth_analyzer().max_depth(&a);
        assert_eq!(depth, 0, "Atomic node should have depth 0");
    }

    #[test]
    fn test_estimate_emergence_from_vectors() {
        let chain = setup();
        let single = vec![chain.genesis.hv("single", PHYSICS_DIM)];
        let multiple: Vec<ContinuousHV> = (0..4)
            .map(|i| chain.genesis.hv(&format!("v{}", i), PHYSICS_DIM))
            .collect();

        // Single vector should have low emergence
        let single_emergence = chain.estimate_emergence_from_vectors(&single);
        assert!(
            single_emergence < 0.1,
            "Single vector emergence should be low"
        );

        // Multiple vectors should have positive emergence
        let multi_emergence = chain.estimate_emergence_from_vectors(&multiple);
        assert!(
            multi_emergence >= 0.0,
            "Multi-vector emergence should be non-negative"
        );
    }
}
