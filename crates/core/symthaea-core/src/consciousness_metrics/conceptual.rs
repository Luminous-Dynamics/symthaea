// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conceptual Structure (IIT 3.0/4.0)
//!
//! Constellation of concepts (mechanisms with φ > 0).

use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

use super::{ContinuousEntropyEstimator, IIT4Calculator, TruePhiCalculator};

/// A concept (mechanism with irreducible cause-effect)
///
/// In IIT, a concept is a mechanism that has integrated cause-effect
/// information (φ > 0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// Indices of elements that form this mechanism
    pub mechanism: Vec<usize>,
    /// Small phi - irreducibility of this mechanism
    pub phi: f64,
    /// Cause information
    pub cause_info: f64,
    /// Effect information
    pub effect_info: f64,
    /// Cause repertoire entropy
    pub cause_entropy: f64,
    /// Effect repertoire entropy
    pub effect_entropy: f64,
}

/// The conceptual structure (constellation of concepts)
///
/// The constellation is the complete set of concepts (mechanisms with φ > 0)
/// that specify the integrated cause-effect structure of the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptualStructure {
    /// All concepts (mechanisms with φ > 0)
    pub concepts: Vec<Concept>,
    /// Big Phi - integrated information of the whole system
    pub big_phi: f64,
    /// Total conceptual information (sum of all φ values)
    pub total_phi: f64,
    /// Number of mechanisms considered
    pub mechanisms_considered: usize,
    /// Fraction of mechanisms that are concepts (φ > 0)
    pub concept_fraction: f64,
}

/// Conceptual structure calculator
///
/// Computes the complete conceptual structure (constellation of concepts)
/// for a system of HDC components.
#[derive(Debug, Clone)]
pub struct ConceptualStructureCalculator {
    /// Threshold for considering a mechanism a concept
    phi_threshold: f64,
    /// Maximum mechanism size to consider (for computational tractability)
    max_mechanism_size: usize,
    /// Base estimator
    estimator: ContinuousEntropyEstimator,
}

impl Default for ConceptualStructureCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConceptualStructureCalculator {
    /// Create a new calculator with default parameters
    pub fn new() -> Self {
        Self {
            phi_threshold: 0.001,
            max_mechanism_size: 4,
            estimator: ContinuousEntropyEstimator::fast(),
        }
    }

    /// Create with custom parameters
    pub fn with_params(phi_threshold: f64, max_mechanism_size: usize) -> Self {
        Self {
            phi_threshold,
            max_mechanism_size,
            estimator: ContinuousEntropyEstimator::fast(),
        }
    }

    /// Compute the conceptual structure for a system
    ///
    /// Enumerates all possible mechanisms up to max_mechanism_size and
    /// computes their integrated cause-effect information.
    pub fn compute(&self, components: &[ContinuousHV]) -> ConceptualStructure {
        let n = components.len();
        if n == 0 {
            return ConceptualStructure {
                concepts: vec![],
                big_phi: 0.0,
                total_phi: 0.0,
                mechanisms_considered: 0,
                concept_fraction: 0.0,
            };
        }

        let mut concepts = Vec::new();
        let mut mechanisms_considered = 0;

        // Enumerate all possible mechanisms (subsets of components)
        let max_size = self.max_mechanism_size.min(n);
        for size in 1..=max_size {
            for mechanism in self.combinations(n, size) {
                mechanisms_considered += 1;

                // Get the mechanism components
                let mech_components: Vec<&ContinuousHV> =
                    mechanism.iter().map(|&i| &components[i]).collect();

                // Compute cause-effect information for this mechanism
                let concept = self.compute_concept(&mechanism, &mech_components, components);

                if concept.phi > self.phi_threshold {
                    concepts.push(concept);
                }
            }
        }

        // Compute big Phi using TruePhiCalculator
        let phi_calc = TruePhiCalculator::new();
        let big_phi_result = phi_calc.compute_true_phi(components);

        let total_phi: f64 = concepts.iter().map(|c| c.phi).sum();
        let concept_fraction = if mechanisms_considered > 0 {
            concepts.len() as f64 / mechanisms_considered as f64
        } else {
            0.0
        };

        ConceptualStructure {
            concepts,
            big_phi: big_phi_result.phi,
            total_phi,
            mechanisms_considered,
            concept_fraction,
        }
    }

    /// Compute a single concept for a mechanism
    fn compute_concept(
        &self,
        mechanism: &[usize],
        mech_components: &[&ContinuousHV],
        all_components: &[ContinuousHV],
    ) -> Concept {
        if mech_components.is_empty() {
            return Concept {
                mechanism: mechanism.to_vec(),
                phi: 0.0,
                cause_info: 0.0,
                effect_info: 0.0,
                cause_entropy: 0.0,
                effect_entropy: 0.0,
            };
        }

        // Bundle mechanism components
        let mech_bundle = ContinuousHV::bundle(mech_components);

        // Get purview (context) - all components not in mechanism
        let purview: Vec<ContinuousHV> = all_components
            .iter()
            .enumerate()
            .filter(|(i, _)| !mechanism.contains(i))
            .map(|(_, c)| c.clone())
            .collect();

        // If no purview, use mechanism itself
        let context = if purview.is_empty() {
            mech_components.iter().map(|&c| c.clone()).collect()
        } else {
            purview
        };

        // Compute integrated information
        let iit4 = IIT4Calculator::new();
        let phi = iit4.small_phi(&mech_bundle, &context);

        // Compute cause-effect entropies
        let cause_entropy = self.estimator.entropy(&mech_bundle);
        let effect_entropy = if !context.is_empty() {
            let refs: Vec<&ContinuousHV> = context.iter().collect();
            let context_bundle = ContinuousHV::bundle(&refs);
            self.estimator.entropy(&context_bundle)
        } else {
            0.0
        };

        // Cause information: MI between mechanism and context (mechanism → context)
        let context_bundle = if !context.is_empty() {
            let refs: Vec<&ContinuousHV> = context.iter().collect();
            ContinuousHV::bundle(&refs)
        } else {
            mech_bundle.clone()
        };
        let cause_info = self
            .estimator
            .mutual_information_fast(&mech_bundle, &context_bundle);

        // Effect information: how much the mechanism constrains the context
        // (effect repertoire). Computed separately from cause to satisfy
        // IIT 4.0 (Albantakis et al. 2023) requirement for asymmetric
        // cause-effect structure. Uses intrinsic information of the context
        // as a proxy: higher intrinsic info = mechanism constrains context more.
        // NOTE: Full IIT 4.0 compliance requires temporal transition data
        // (see temporal.rs for the complete implementation with separate
        // cause_information / effect_information using transition matrices).
        let effect_info = {
            let iit4_calc = IIT4Calculator::new();
            // Effect = how much the mechanism specifies the context's future state
            // Proxy: intrinsic information (specificity) of the context given
            // that it's coupled to this mechanism
            iit4_calc.intrinsic_information(&context_bundle)
        };

        Concept {
            mechanism: mechanism.to_vec(),
            phi,
            cause_info,
            effect_info,
            cause_entropy,
            effect_entropy,
        }
    }

    /// Generate all combinations of size k from n elements
    fn combinations(&self, n: usize, k: usize) -> Vec<Vec<usize>> {
        if k == 0 || k > n {
            return vec![];
        }

        let mut result = Vec::new();
        let mut current = Vec::with_capacity(k);
        self.generate_combinations(0, n, k, &mut current, &mut result);
        result
    }

    fn generate_combinations(
        &self,
        start: usize,
        n: usize,
        k: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }

        for i in start..n {
            current.push(i);
            self.generate_combinations(i + 1, n, k, current, result);
            current.pop();
        }
    }

    /// Get concepts sorted by phi (highest first)
    pub fn top_concepts<'a>(
        &self,
        structure: &'a ConceptualStructure,
        limit: usize,
    ) -> Vec<&'a Concept> {
        let mut sorted: Vec<&'a Concept> = structure.concepts.iter().collect();
        sorted.sort_by(|a, b| {
            b.phi
                .partial_cmp(&a.phi)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(limit).collect()
    }

    /// Compute conceptual distance between two structures
    ///
    /// Uses Earth Mover's Distance-like metric over concept constellations.
    pub fn conceptual_distance(&self, s1: &ConceptualStructure, s2: &ConceptualStructure) -> f64 {
        // Simple metric: absolute difference in total φ
        // (A more rigorous implementation would use EMD over concept space)
        let phi_diff = (s1.total_phi - s2.total_phi).abs();
        let concept_diff = (s1.concepts.len() as f64 - s2.concepts.len() as f64).abs();
        let big_phi_diff = (s1.big_phi - s2.big_phi).abs();

        // Weighted combination
        0.5 * big_phi_diff + 0.3 * phi_diff + 0.2 * concept_diff
    }
}
