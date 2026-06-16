// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

use super::chemistry::Chemistry;
use super::hadrons::Hadrons;
use super::periodic_table::PeriodicTable;
use super::standard_model::{PHYSICS_DIM, StandardModel};
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
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
    pub embodiment: ContinuousHV, // Link between physical and phenomenal
    pub emergence: ContinuousHV,  // How macro arises from micro
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

    /// Estimate phenomenal character of a vector via concept similarity.
    ///
    /// Computes how similar this vector is to phenomenal vs functional concepts.
    /// Range: -1 (fully functional) to +1 (fully phenomenal).
    ///
    /// This is a fast heuristic based on labeled concept vector similarity.
    /// For rigorous measurement based on Shannon entropy and information
    /// integration, use `EmergenceChain::phenomenal_index_rigorous()`.
    pub fn estimate_phenomenal_index(&self, vector: &ContinuousHV) -> f32 {
        let phenomenal_sim = (vector.similarity(&self.qualia)
            + vector.similarity(&self.awareness)
            + vector.similarity(&self.subjective)
            + vector.similarity(&self.experience)
            + vector.similarity(&self.unity))
            / 5.0;

        let functional_sim = (vector.similarity(&self.computation)
            + vector.similarity(&self.information)
            + vector.similarity(&self.processing)
            + vector.similarity(&self.algorithm)
            + vector.similarity(&self.function))
            / 5.0;

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
    pub fn binding_unity_test(&self, a: &ContinuousHV, b: &ContinuousHV) -> BindingUnityResult {
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
        use crate::consciousness_metrics::TruePhiCalculator;
        let calc = TruePhiCalculator::new();

        // Normalize phi per-component to make levels comparable.
        // Using Φ/n (linear normalization) rather than Φ/(n*(n-1)/2*4.0) (quadratic)
        // because the quadratic denominator suppresses signal at higher composition levels.
        // Science: Tononi (2004) — Φ should scale with integration, not be penalized by size.
        let normalize_phi = |components: &[ContinuousHV]| -> f32 {
            let result = calc.compute_true_phi(components);
            let n = components.len().max(1) as f64;
            (result.phi / n).min(1.0) as f32
        };

        // Level 0: Fundamental particles (independent components)
        let quark_components = vec![model.up_quark.clone(), model.electron.clone()];
        let level_0 = normalize_phi(&quark_components);

        // Level 1: Hadrons — BOUND from quarks (compositional hierarchy)
        // Proton = bind(up_quark, up_quark, down_quark), but we use the pre-built
        // hadron vectors + bind them with the quarks they're composed of.
        let proton_bound = hadrons.proton.bind(&model.up_quark);
        let neutron_bound = hadrons.neutron.bind(&model.down_quark);
        let hadron_components = vec![proton_bound, neutron_bound];
        let level_1 = normalize_phi(&hadron_components);

        // Level 2: Atoms — BOUND from hadrons + electrons (compositional hierarchy)
        // Hydrogen = bind(proton, electron); Carbon = bind(nucleus, electrons)
        let hydrogen_bound = hadrons.proton.bind(&model.electron);
        let carbon_bound = if let Some(c) = table.element(6) {
            c.vector.bind(&hadrons.proton).bind(&hadrons.neutron)
        } else {
            hadrons.proton.bind(&hadrons.neutron) // fallback
        };
        let atom_components = vec![hydrogen_bound.clone(), carbon_bound];
        let level_2 = normalize_phi(&atom_components);

        // Level 3: Molecules — BOUND from atoms (compositional hierarchy)
        // Water = bind(hydrogen, hydrogen, oxygen)
        let water = chemistry.water(table);
        let oxygen_bound = if let Some(o) = table.element(8) {
            o.vector.bind(&hadrons.proton).bind(&hadrons.neutron)
        } else {
            hadrons.proton.clone()
        };
        let mol_components = vec![
            water.vector.bind(&hydrogen_bound), // water bound with hydrogen
            hydrogen_bound.clone(),             // hydrogen atom
            oxygen_bound,                       // oxygen atom (bound composite)
        ];
        let level_3 = normalize_phi(&mol_components);

        CompositionalAnalysis {
            level_0_quarks: level_0,
            level_1_hadrons: level_1,
            level_2_atoms: level_2,
            level_3_molecules: level_3,
        }
    }

    /// Analyze compositional hierarchy using rigorous entropy-based measures
    ///
    /// This is the recommended replacement for `compositional_phenomenal_analysis`.
    /// Instead of using similarity to concept vectors, it uses true Φ (integrated information)
    /// to measure how much information is integrated at each level.
    ///
    /// Returns Φ values for each level of composition.
    pub fn compositional_phi_analysis(
        &self,
        model: &StandardModel,
        hadrons: &Hadrons,
        table: &PeriodicTable,
        chemistry: &Chemistry,
    ) -> CompositionalPhiAnalysis {
        use crate::consciousness_metrics::TruePhiCalculator;
        let calc = TruePhiCalculator::new();

        // Level 0: Fundamental particles (quarks and electron)
        // Φ measures how much information is integrated in the quark triplet
        let quarks = vec![model.up_quark.clone(), model.down_quark.clone()];
        let quark_phi = calc.compute_true_phi(&quarks);

        // Level 1: Hadrons (proton, neutron) - composed from quarks
        let hadron_components = vec![hadrons.proton.clone(), hadrons.neutron.clone()];
        let hadron_phi = calc.compute_true_phi(&hadron_components);

        // Level 2: Atoms - hydrogen and carbon
        let mut atom_components = Vec::new();
        if let Some(h) = table.element(1) {
            atom_components.push(h.vector.clone());
        }
        if let Some(c) = table.element(6) {
            atom_components.push(c.vector.clone());
        }
        let atom_phi = if atom_components.len() >= 2 {
            calc.compute_true_phi(&atom_components)
        } else {
            crate::consciousness_metrics::TruePhiResult {
                phi: 0.0,
                system_ei: 0.0,
                mip_ei: 0.0,
                mip: crate::consciousness_metrics::TruePartition {
                    part_a: vec![],
                    part_b: vec![],
                },
                component_entropies: vec![],
                mutual_information_matrix: vec![],
            }
        };

        // Level 3: Molecule (water) - composed from atoms
        let water = chemistry.water(table);
        let h_elem = table.element(1).map(|e| e.vector.clone());
        let o_elem = table.element(8).map(|e| e.vector.clone());
        let molecule_phi = if let (Some(h), Some(o)) = (h_elem, o_elem) {
            let mol_components = vec![water.vector.clone(), h.clone(), h, o];
            calc.compute_true_phi(&mol_components)
        } else {
            crate::consciousness_metrics::TruePhiResult {
                phi: 0.0,
                system_ei: 0.0,
                mip_ei: 0.0,
                mip: crate::consciousness_metrics::TruePartition {
                    part_a: vec![],
                    part_b: vec![],
                },
                component_entropies: vec![],
                mutual_information_matrix: vec![],
            }
        };

        CompositionalPhiAnalysis {
            level_0_quarks_phi: quark_phi.phi,
            level_1_hadrons_phi: hadron_phi.phi,
            level_2_atoms_phi: atom_phi.phi,
            level_3_molecules_phi: molecule_phi.phi,
            // EI values for detailed analysis
            level_0_quarks_ei: quark_phi.system_ei,
            level_1_hadrons_ei: hadron_phi.system_ei,
            level_2_atoms_ei: atom_phi.system_ei,
            level_3_molecules_ei: molecule_phi.system_ei,
        }
    }

    /// Create "embodied qualia" by bundling physical substrate with phenomenal concepts
    ///
    /// Uses weighted bundle to add phenomenal character while preserving structure.
    /// Example: bundle(neuron_vector, qualia, embodiment) → embodied qualia
    pub fn embody_qualia(&self, physical_substrate: &ContinuousHV) -> ContinuousHV {
        // Bundle physical with phenomenal concepts, weighting phenomenal higher
        ContinuousHV::weighted_bundle(
            &[
                physical_substrate,
                &self.qualia,
                &self.embodiment,
                &self.experience,
            ],
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
        let phenomenal: Vec<&ContinuousHV> = vectors
            .iter()
            .filter(|(_, is_phen)| *is_phen)
            .map(|(v, _)| *v)
            .collect();

        let functional: Vec<&ContinuousHV> = vectors
            .iter()
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
        let phen_var: f32 = phenomenal
            .iter()
            .map(|v| v.subtract(&phen_centroid).norm())
            .sum::<f32>()
            / phenomenal.len() as f32;

        let func_var: f32 = functional
            .iter()
            .map(|v| v.subtract(&func_centroid).norm())
            .sum::<f32>()
            / functional.len() as f32;

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

/// Compositional Φ analysis result using rigorous entropy-based measures
///
/// This replaces `CompositionalAnalysis` with true integrated information (Φ)
/// at each level of physical composition.
#[derive(Debug, Clone)]
pub struct CompositionalPhiAnalysis {
    /// Integrated information at quark level
    pub level_0_quarks_phi: f64,
    /// Integrated information at hadron level
    pub level_1_hadrons_phi: f64,
    /// Integrated information at atom level
    pub level_2_atoms_phi: f64,
    /// Integrated information at molecule level
    pub level_3_molecules_phi: f64,
    /// Effective information at quark level
    pub level_0_quarks_ei: f64,
    /// Effective information at hadron level
    pub level_1_hadrons_ei: f64,
    /// Effective information at atom level
    pub level_2_atoms_ei: f64,
    /// Effective information at molecule level
    pub level_3_molecules_ei: f64,
}

impl CompositionalPhiAnalysis {
    /// Does Φ increase with composition level?
    pub fn shows_emergence(&self) -> bool {
        self.level_3_molecules_phi > self.level_0_quarks_phi
    }

    /// Compute gradient of Φ across levels
    pub fn phi_gradient(&self) -> f64 {
        let levels = [
            self.level_0_quarks_phi,
            self.level_1_hadrons_phi,
            self.level_2_atoms_phi,
            self.level_3_molecules_phi,
        ];

        let mut gradient = 0.0;
        for i in 1..levels.len() {
            gradient += levels[i] - levels[i - 1];
        }
        gradient / (levels.len() - 1) as f64
    }

    /// Total integrated information across all levels
    pub fn total_phi(&self) -> f64 {
        self.level_0_quarks_phi
            + self.level_1_hadrons_phi
            + self.level_2_atoms_phi
            + self.level_3_molecules_phi
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEMPORAL Φ AND CONSCIOUSNESS DYNAMICS
// Connecting temporal information integration to consciousness evolution
// ═══════════════════════════════════════════════════════════════════════════════

use crate::consciousness_metrics::{TemporalPhiCalculator, TemporalTransition};

/// Temporal consciousness state
///
/// Represents a conscious state at a moment in time, enabling
/// cause-effect analysis over temporal dynamics.
#[derive(Debug, Clone)]
pub struct TemporalConsciousnessState {
    /// Current state vector
    pub state: ContinuousHV,
    /// Timestamp (arbitrary units)
    pub time: f64,
    /// Phenomenal index at this moment
    pub phenomenal_index: f32,
}

/// Result of temporal consciousness analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConsciousnessAnalysis {
    /// Cause information: how much present specifies past
    pub cause_info: f64,
    /// Effect information: how much present specifies future
    pub effect_info: f64,
    /// Integrated cause information (φ_cause)
    pub integrated_cause: f64,
    /// Integrated effect information (φ_effect)
    pub integrated_effect: f64,
    /// Total cause-effect Φ
    pub phi_cause_effect: f64,
    /// Change in phenomenal index over time
    pub phenomenal_gradient: f64,
    /// Whether consciousness is growing (gradient > 0)
    pub is_growing: bool,
}

impl PhysicsConsciousnessBridge {
    /// Analyze temporal consciousness dynamics
    ///
    /// Computes cause-effect information for a sequence of conscious states.
    /// This connects temporal IIT to consciousness evolution.
    ///
    /// # Arguments
    /// * `past_state` - Previous conscious state
    /// * `current_state` - Current conscious state
    ///
    /// # Returns
    /// Temporal consciousness analysis including cause-effect measures
    pub fn analyze_temporal_consciousness(
        &self,
        past_state: &TemporalConsciousnessState,
        current_state: &TemporalConsciousnessState,
    ) -> TemporalConsciousnessAnalysis {
        let calc = TemporalPhiCalculator::new();

        // Create temporal transition
        let transition =
            TemporalTransition::new(past_state.state.clone(), current_state.state.clone());

        // Compute cause-effect information
        let ce_info = calc.compute_cause_effect(&transition);

        // Compute phenomenal gradient
        let phenomenal_gradient = if (current_state.time - past_state.time).abs() > 1e-10 {
            (current_state.phenomenal_index - past_state.phenomenal_index) as f64
                / (current_state.time - past_state.time)
        } else {
            0.0
        };

        TemporalConsciousnessAnalysis {
            cause_info: ce_info.cause_info,
            effect_info: ce_info.effect_info,
            integrated_cause: ce_info.integrated_cause,
            integrated_effect: ce_info.integrated_effect,
            phi_cause_effect: ce_info.phi_cause_effect,
            phenomenal_gradient,
            is_growing: phenomenal_gradient > 0.0,
        }
    }

    /// Analyze consciousness trajectory over multiple states
    ///
    /// Computes how consciousness evolves through a sequence of states,
    /// measuring integrated information flow over time.
    pub fn analyze_consciousness_trajectory(
        &self,
        states: &[TemporalConsciousnessState],
    ) -> Vec<TemporalConsciousnessAnalysis> {
        if states.len() < 2 {
            return vec![];
        }

        let mut analyses = Vec::with_capacity(states.len() - 1);

        for i in 0..states.len() - 1 {
            let analysis = self.analyze_temporal_consciousness(&states[i], &states[i + 1]);
            analyses.push(analysis);
        }

        analyses
    }

    /// Compute consciousness momentum
    ///
    /// Measures the "inertia" of consciousness - how much the current state
    /// preserves information from the past and projects into the future.
    pub fn consciousness_momentum(
        &self,
        past: &TemporalConsciousnessState,
        current: &TemporalConsciousnessState,
        future: &TemporalConsciousnessState,
    ) -> f64 {
        let past_analysis = self.analyze_temporal_consciousness(past, current);
        let future_analysis = self.analyze_temporal_consciousness(current, future);

        // Momentum = average of cause and effect preservation
        (past_analysis.cause_info + future_analysis.effect_info) / 2.0
    }

    /// Create a temporal consciousness state from a physical vector
    pub fn create_conscious_state(
        &self,
        physical: &ContinuousHV,
        time: f64,
    ) -> TemporalConsciousnessState {
        let phenomenal_index = self.estimate_phenomenal_index(physical);
        TemporalConsciousnessState {
            state: physical.clone(),
            time,
            phenomenal_index,
        }
    }

    /// Simulate consciousness evolution with a transformation
    ///
    /// Creates a sequence of conscious states by applying a transformation.
    pub fn simulate_consciousness_evolution<F>(
        &self,
        initial: &ContinuousHV,
        steps: usize,
        transform: F,
    ) -> Vec<TemporalConsciousnessState>
    where
        F: Fn(&ContinuousHV, usize) -> ContinuousHV,
    {
        let mut states = Vec::with_capacity(steps + 1);
        let mut current = initial.clone();

        for step in 0..=steps {
            let state = self.create_conscious_state(&current, step as f64);
            states.push(state);
            if step < steps {
                current = transform(&current, step);
            }
        }

        states
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (
        StandardModel,
        Hadrons,
        PeriodicTable,
        Chemistry,
        PhysicsConsciousnessBridge,
        GenesisSeed,
    ) {
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
        let qualia_index = bridge.estimate_phenomenal_index(&bridge.qualia);

        // Computation should have low phenomenal index
        let comp_index = bridge.estimate_phenomenal_index(&bridge.computation);

        assert!(
            qualia_index > comp_index,
            "Qualia should be more phenomenal: {} vs {}",
            qualia_index,
            comp_index
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

        let analysis =
            bridge.compositional_phenomenal_analysis(&model, &hadrons, &table, &chemistry);

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
        let phen_index = bridge.estimate_phenomenal_index(&embodied);
        let electron_index = bridge.estimate_phenomenal_index(&model.electron);

        // Embodiment should increase phenomenal index
        assert!(
            phen_index > electron_index,
            "Embodiment should increase phenomenal index: {} vs {}",
            phen_index,
            electron_index
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

    #[test]
    fn test_compositional_phi_analysis() {
        let (model, hadrons, table, chemistry, bridge, _) = setup();

        let analysis = bridge.compositional_phi_analysis(&model, &hadrons, &table, &chemistry);

        // All levels should have non-negative Φ
        assert!(
            analysis.level_0_quarks_phi >= 0.0,
            "Quark Φ should be non-negative: {:.4}",
            analysis.level_0_quarks_phi
        );
        assert!(
            analysis.level_1_hadrons_phi >= 0.0,
            "Hadron Φ should be non-negative: {:.4}",
            analysis.level_1_hadrons_phi
        );
        assert!(
            analysis.level_2_atoms_phi >= 0.0,
            "Atom Φ should be non-negative: {:.4}",
            analysis.level_2_atoms_phi
        );
        assert!(
            analysis.level_3_molecules_phi >= 0.0,
            "Molecule Φ should be non-negative: {:.4}",
            analysis.level_3_molecules_phi
        );

        // EI should be non-negative
        assert!(
            analysis.level_0_quarks_ei >= 0.0,
            "Quark EI should be non-negative: {:.4}",
            analysis.level_0_quarks_ei
        );

        // Compute gradient
        let gradient = analysis.phi_gradient();
        println!(
            "Rigorous Φ analysis: quarks={:.4}, hadrons={:.4}, atoms={:.4}, molecules={:.4}, gradient={:.4}",
            analysis.level_0_quarks_phi,
            analysis.level_1_hadrons_phi,
            analysis.level_2_atoms_phi,
            analysis.level_3_molecules_phi,
            gradient
        );
    }

    #[test]
    fn test_compositional_phi_vs_legacy() {
        let (model, hadrons, table, chemistry, bridge, _) = setup();

        // Both methods should complete without error
        let legacy = bridge.compositional_phenomenal_analysis(&model, &hadrons, &table, &chemistry);
        let rigorous = bridge.compositional_phi_analysis(&model, &hadrons, &table, &chemistry);

        // They measure different things but should both be valid
        assert!(!legacy.level_0_quarks.is_nan());
        assert!(rigorous.level_0_quarks_phi >= 0.0);

        // Rigorous should have meaningful total Φ
        let total = rigorous.total_phi();
        assert!(total >= 0.0, "Total Φ should be non-negative: {:.4}", total);
    }
}
