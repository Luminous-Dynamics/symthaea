//! # Physics → Consciousness Bridge
//!
//! This module connects the Genesis Physics hierarchy to consciousness research,
//! testing whether HDC binding of physics concepts exhibits phenomenal integration.
//!
//! ## Core Hypothesis
//!
//! If binding creates phenomenal unity (as suggested by our HDC research),
//! then physics concepts that are bound together should show:
//! 1. Higher integrated information (Φ)
//! 2. Unity scores above bundled alternatives
//! 3. Distinctive topological structure
//!
//! ## Applications
//!
//! 1. **Qualia from Physics**: Does binding quark → hadron → atom create
//!    "proto-phenomenal" structure?
//!
//! 2. **Energy-Consciousness Relationship**: Is there a vector space
//!    relationship between energy and awareness concepts?
//!
//! 3. **Compositional Consciousness**: Can consciousness primitives
//!    be composed like physics primitives?

use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use super::standard_model::{StandardModel, PHYSICS_DIM};
use super::hadrons::Hadrons;
use super::periodic_table::PeriodicTable;
use super::chemistry::Chemistry;
use serde::{Deserialize, Serialize};

/// Consciousness concept categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsciousnessConcept {
    // Phenomenal concepts
    Qualia,
    Awareness,
    Subjective,
    Experience,
    Unity,

    // Functional concepts
    Computation,
    Information,
    Processing,
    Algorithm,
    Function,
}

impl ConsciousnessConcept {
    fn domain(&self) -> &'static str {
        match self {
            ConsciousnessConcept::Qualia => "consciousness::qualia",
            ConsciousnessConcept::Awareness => "consciousness::awareness",
            ConsciousnessConcept::Subjective => "consciousness::subjective",
            ConsciousnessConcept::Experience => "consciousness::experience",
            ConsciousnessConcept::Unity => "consciousness::unity",
            ConsciousnessConcept::Computation => "functional::computation",
            ConsciousnessConcept::Information => "functional::information",
            ConsciousnessConcept::Processing => "functional::processing",
            ConsciousnessConcept::Algorithm => "functional::algorithm",
            ConsciousnessConcept::Function => "functional::function",
        }
    }

    pub fn is_phenomenal(&self) -> bool {
        matches!(
            self,
            ConsciousnessConcept::Qualia
                | ConsciousnessConcept::Awareness
                | ConsciousnessConcept::Subjective
                | ConsciousnessConcept::Experience
                | ConsciousnessConcept::Unity
        )
    }
}

/// Physics-consciousness binding analysis
#[derive(Debug, Clone)]
pub struct PhysicsConsciousnessBridge {
    // Consciousness concept vectors
    pub qualia: ContinuousHV,
    pub awareness: ContinuousHV,
    pub subjective: ContinuousHV,
    pub experience: ContinuousHV,
    pub unity: ContinuousHV,

    // Functional concept vectors
    pub computation: ContinuousHV,
    pub information: ContinuousHV,
    pub processing: ContinuousHV,
    pub algorithm: ContinuousHV,
    pub function: ContinuousHV,

    // Bridge concepts
    pub embodiment: ContinuousHV,  // Link between physical and phenomenal
    pub emergence: ContinuousHV,   // How macro arises from micro
    pub integration: ContinuousHV, // IIT-style integration
}

impl PhysicsConsciousnessBridge {
    /// Create bridge from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            qualia: genesis.hv(ConsciousnessConcept::Qualia.domain(), PHYSICS_DIM),
            awareness: genesis.hv(ConsciousnessConcept::Awareness.domain(), PHYSICS_DIM),
            subjective: genesis.hv(ConsciousnessConcept::Subjective.domain(), PHYSICS_DIM),
            experience: genesis.hv(ConsciousnessConcept::Experience.domain(), PHYSICS_DIM),
            unity: genesis.hv(ConsciousnessConcept::Unity.domain(), PHYSICS_DIM),
            computation: genesis.hv(ConsciousnessConcept::Computation.domain(), PHYSICS_DIM),
            information: genesis.hv(ConsciousnessConcept::Information.domain(), PHYSICS_DIM),
            processing: genesis.hv(ConsciousnessConcept::Processing.domain(), PHYSICS_DIM),
            algorithm: genesis.hv(ConsciousnessConcept::Algorithm.domain(), PHYSICS_DIM),
            function: genesis.hv(ConsciousnessConcept::Function.domain(), PHYSICS_DIM),
            embodiment: genesis.hv("bridge::embodiment", PHYSICS_DIM),
            emergence: genesis.hv("bridge::emergence", PHYSICS_DIM),
            integration: genesis.hv("bridge::integration", PHYSICS_DIM),
        }
    }

    /// Measure phenomenal character of a vector
    ///
    /// How similar is this vector to phenomenal vs functional concepts?
    pub fn phenomenal_index(&self, vector: &ContinuousHV) -> f32 {
        let phenomenal_sim = (
            vector.similarity(&self.qualia)
            + vector.similarity(&self.awareness)
            + vector.similarity(&self.subjective)
            + vector.similarity(&self.experience)
            + vector.similarity(&self.unity)
        ) / 5.0;

        let functional_sim = (
            vector.similarity(&self.computation)
            + vector.similarity(&self.information)
            + vector.similarity(&self.processing)
            + vector.similarity(&self.algorithm)
            + vector.similarity(&self.function)
        ) / 5.0;

        // Range: -1 (fully functional) to +1 (fully phenomenal)
        phenomenal_sim - functional_sim
    }

    /// Test if binding increases unity score
    ///
    /// Compare:
    /// - bind(a, b): Creates new orthogonal composite
    /// - bundle(a, b): Creates superposition
    ///
    /// Unity hypothesis: bind creates more "unified" representation
    pub fn binding_unity_test(
        &self,
        a: &ContinuousHV,
        b: &ContinuousHV,
    ) -> BindingUnityResult {
        let bound = a.bind(b);
        let bundled = ContinuousHV::bundle(&[a, b]);

        // Measure unity (similarity to unity concept)
        let bound_unity = bound.similarity(&self.unity);
        let bundled_unity = bundled.similarity(&self.unity);

        // Measure integration (similarity to integration concept)
        let bound_integration = bound.similarity(&self.integration);
        let bundled_integration = bundled.similarity(&self.integration);

        BindingUnityResult {
            bound_unity,
            bundled_unity,
            bound_integration,
            bundled_integration,
            binding_advantage: bound_unity - bundled_unity,
        }
    }

    /// Analyze compositional hierarchy for phenomenal structure
    ///
    /// Tests: quark → hadron → atom → molecule
    /// Does each level of binding accumulate phenomenal character?
    pub fn compositional_phenomenal_analysis(
        &self,
        model: &StandardModel,
        hadrons: &Hadrons,
        table: &PeriodicTable,
        chemistry: &Chemistry,
    ) -> CompositionalAnalysis {
        // Level 0: Fundamental particles
        let quark_phenomenal = self.phenomenal_index(&model.up_quark);
        let electron_phenomenal = self.phenomenal_index(&model.electron);

        // Level 1: Hadrons (composed from quarks)
        let proton_phenomenal = self.phenomenal_index(&hadrons.proton);
        let neutron_phenomenal = self.phenomenal_index(&hadrons.neutron);

        // Level 2: Atoms (composed from hadrons + electrons)
        let hydrogen_phenomenal = table.element(1)
            .map(|e| self.phenomenal_index(&e.vector))
            .unwrap_or(0.0);
        let carbon_phenomenal = table.element(6)
            .map(|e| self.phenomenal_index(&e.vector))
            .unwrap_or(0.0);

        // Level 3: Molecules (composed from atoms)
        let water = chemistry.water(table);
        let water_phenomenal = self.phenomenal_index(&water.vector);

        CompositionalAnalysis {
            level_0_quarks: (quark_phenomenal + electron_phenomenal) / 2.0,
            level_1_hadrons: (proton_phenomenal + neutron_phenomenal) / 2.0,
            level_2_atoms: (hydrogen_phenomenal + carbon_phenomenal) / 2.0,
            level_3_molecules: water_phenomenal,
        }
    }

    /// Create "embodied qualia" by bundling physical substrate with phenomenal concepts
    ///
    /// Uses weighted bundle to add phenomenal character while preserving structure.
    /// Example: bundle(neuron_vector, qualia, embodiment) → embodied qualia
    pub fn embody_qualia(&self, physical_substrate: &ContinuousHV) -> ContinuousHV {
        // Bundle physical with phenomenal concepts, weighting phenomenal higher
        ContinuousHV::weighted_bundle(
            &[physical_substrate, &self.qualia, &self.embodiment, &self.experience],
            &[1.0, 2.0, 1.5, 1.0],
        )
    }

    /// Compute discrimination between phenomenal and functional for a set of vectors
    ///
    /// Returns Fisher's criterion for the given vectors
    pub fn compute_discrimination(&self, vectors: &[(&ContinuousHV, bool)]) -> f32 {
        if vectors.is_empty() {
            return 0.0;
        }

        // Split by class
        let phenomenal: Vec<&ContinuousHV> = vectors.iter()
            .filter(|(_, is_phen)| *is_phen)
            .map(|(v, _)| *v)
            .collect();

        let functional: Vec<&ContinuousHV> = vectors.iter()
            .filter(|(_, is_phen)| !*is_phen)
            .map(|(v, _)| *v)
            .collect();

        if phenomenal.is_empty() || functional.is_empty() {
            return 0.0;
        }

        // Compute centroids
        let phen_centroid = ContinuousHV::bundle(&phenomenal);
        let func_centroid = ContinuousHV::bundle(&functional);

        // Compute centroid distance
        let diff = phen_centroid.subtract(&func_centroid);
        let centroid_dist = diff.norm();

        // Compute within-class variance
        let phen_var: f32 = phenomenal.iter()
            .map(|v| v.subtract(&phen_centroid).norm())
            .sum::<f32>() / phenomenal.len() as f32;

        let func_var: f32 = functional.iter()
            .map(|v| v.subtract(&func_centroid).norm())
            .sum::<f32>() / functional.len() as f32;

        let avg_within = (phen_var + func_var) / 2.0;

        if avg_within < 1e-10 {
            return centroid_dist;
        }

        // Fisher's criterion
        centroid_dist / avg_within
    }
}

/// Result of binding unity test
#[derive(Debug, Clone)]
pub struct BindingUnityResult {
    pub bound_unity: f32,
    pub bundled_unity: f32,
    pub bound_integration: f32,
    pub bundled_integration: f32,
    pub binding_advantage: f32,
}

/// Compositional phenomenal analysis result
#[derive(Debug, Clone)]
pub struct CompositionalAnalysis {
    pub level_0_quarks: f32,
    pub level_1_hadrons: f32,
    pub level_2_atoms: f32,
    pub level_3_molecules: f32,
}

impl CompositionalAnalysis {
    /// Does phenomenal character increase with composition?
    pub fn shows_emergence(&self) -> bool {
        // Check if higher levels have more phenomenal character
        self.level_3_molecules > self.level_0_quarks
    }

    /// Compute emergence gradient (change per level)
    pub fn emergence_gradient(&self) -> f32 {
        let levels = [
            self.level_0_quarks,
            self.level_1_hadrons,
            self.level_2_atoms,
            self.level_3_molecules,
        ];

        let mut gradient = 0.0;
        for i in 1..levels.len() {
            gradient += levels[i] - levels[i - 1];
        }
        gradient / (levels.len() - 1) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (StandardModel, Hadrons, PeriodicTable, Chemistry, PhysicsConsciousnessBridge, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("consciousness bridge test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
        let chemistry = Chemistry::from_genesis(&genesis, &table);
        let bridge = PhysicsConsciousnessBridge::from_genesis(&genesis);
        (model, hadrons, table, chemistry, bridge, genesis)
    }

    #[test]
    fn test_bridge_creation() {
        let (_, _, _, _, bridge, _) = setup();

        assert_eq!(bridge.qualia.dim(), PHYSICS_DIM);
        assert_eq!(bridge.computation.dim(), PHYSICS_DIM);
    }

    #[test]
    fn test_phenomenal_index() {
        let (_, _, _, _, bridge, _) = setup();

        // Qualia should have high phenomenal index
        let qualia_index = bridge.phenomenal_index(&bridge.qualia);

        // Computation should have low phenomenal index
        let comp_index = bridge.phenomenal_index(&bridge.computation);

        assert!(
            qualia_index > comp_index,
            "Qualia should be more phenomenal: {} vs {}",
            qualia_index, comp_index
        );
    }

    #[test]
    fn test_binding_unity() {
        let (model, _, _, _, bridge, _) = setup();

        // Test binding of two particles
        let result = bridge.binding_unity_test(&model.up_quark, &model.down_quark);

        // Both should have some unity measure
        assert!(!result.bound_unity.is_nan());
        assert!(!result.bundled_unity.is_nan());
    }

    #[test]
    fn test_compositional_analysis() {
        let (model, hadrons, table, chemistry, bridge, _) = setup();

        let analysis = bridge.compositional_phenomenal_analysis(
            &model, &hadrons, &table, &chemistry
        );

        // All levels should have valid values
        assert!(!analysis.level_0_quarks.is_nan());
        assert!(!analysis.level_1_hadrons.is_nan());
        assert!(!analysis.level_2_atoms.is_nan());
        assert!(!analysis.level_3_molecules.is_nan());

        // Compute emergence gradient
        let gradient = analysis.emergence_gradient();
        println!(
            "Emergence analysis: quarks={:.4}, hadrons={:.4}, atoms={:.4}, molecules={:.4}, gradient={:.4}",
            analysis.level_0_quarks,
            analysis.level_1_hadrons,
            analysis.level_2_atoms,
            analysis.level_3_molecules,
            gradient
        );
    }

    #[test]
    fn test_embody_qualia() {
        let (model, _, _, _, bridge, _) = setup();

        let embodied = bridge.embody_qualia(&model.electron);

        // Embodied qualia should have different structure than bare electron
        let sim = embodied.similarity(&model.electron);
        assert!(
            sim < 0.5,
            "Embodied qualia should differ from bare physical: {}",
            sim
        );

        // Should have some phenomenal character
        let phen_index = bridge.phenomenal_index(&embodied);
        let electron_index = bridge.phenomenal_index(&model.electron);

        // Embodiment should increase phenomenal index
        assert!(
            phen_index > electron_index,
            "Embodiment should increase phenomenal index: {} vs {}",
            phen_index, electron_index
        );
    }

    #[test]
    fn test_discrimination() {
        let (_, _, _, _, bridge, _) = setup();

        // Create test vectors
        let vectors: Vec<(&ContinuousHV, bool)> = vec![
            (&bridge.qualia, true),
            (&bridge.awareness, true),
            (&bridge.experience, true),
            (&bridge.computation, false),
            (&bridge.algorithm, false),
            (&bridge.function, false),
        ];

        let fisher = bridge.compute_discrimination(&vectors);

        assert!(
            fisher > 0.0,
            "Should have positive discrimination: {}",
            fisher
        );
    }
}
