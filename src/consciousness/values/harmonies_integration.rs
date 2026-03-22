// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Harmonies Integration: Eight Harmonies Value Framework
//!
//! Integrates the Eight Harmonies value system into consciousness operations.
//! This provides ethical and value-aligned decision making.

use super::eight_harmonies::{AlignmentResult, EightHarmonies, Harmony};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use symthaea_core::hdc::ContinuousHV;

use crate::hdc::harmony_basis::{HarmonyBasis, MoralFreeEnergy};
use symthaea_types::N_HARMONIES;

/// Configuration for harmonies integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmoniesIntegrationConfig {
    /// Minimum alignment threshold for actions
    pub alignment_threshold: f32,
    /// Weights for each harmony
    pub harmony_weights: HashMap<Harmony, f32>,
    /// Whether to enforce alignment
    pub enforce_alignment: bool,
    /// Learning rate for value updates
    pub learning_rate: f32,
    /// Embedding dimension for harmony vectors (must match input embeddings)
    pub dimension: usize,
}

impl Default for HarmoniesIntegrationConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert(Harmony::ResonantCoherence, 1.0);
        weights.insert(Harmony::PanSentientFlourishing, 1.0);
        weights.insert(Harmony::IntegralWisdom, 1.0);
        weights.insert(Harmony::InfinitePlay, 1.0);
        weights.insert(Harmony::UniversalInterconnectedness, 1.0);
        weights.insert(Harmony::SacredReciprocity, 1.0);
        weights.insert(Harmony::EvolutionaryProgression, 1.0);
        weights.insert(Harmony::SacredStillness, 1.0);

        Self {
            alignment_threshold: 0.7,
            harmony_weights: weights,
            enforce_alignment: true,
            learning_rate: 0.1,
            dimension: 512,
        }
    }
}

/// An action to be evaluated for value alignment
#[derive(Debug, Clone)]
pub struct ValuedAction {
    /// Action identifier
    pub id: String,
    /// Action description
    pub description: String,
    /// Action embedding
    pub embedding: ContinuousHV,
    /// Expected outcomes
    pub expected_outcomes: Vec<String>,
    /// Affected entities
    pub affected_entities: Vec<String>,
}

impl ValuedAction {
    /// Create a new valued action
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        embedding: ContinuousHV,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            embedding,
            expected_outcomes: Vec::new(),
            affected_entities: Vec::new(),
        }
    }

    /// Add an expected outcome
    pub fn with_outcome(mut self, outcome: impl Into<String>) -> Self {
        self.expected_outcomes.push(outcome.into());
        self
    }

    /// Add an affected entity
    pub fn with_entity(mut self, entity: impl Into<String>) -> Self {
        self.affected_entities.push(entity.into());
        self
    }
}

/// Result of value evaluation
#[derive(Debug, Clone)]
pub struct ValueEvaluation {
    /// Overall alignment score (0.0-1.0)
    pub overall_alignment: f32,
    /// Per-harmony scores
    pub harmony_scores: HashMap<Harmony, f32>,
    /// Whether action is approved
    pub approved: bool,
    /// Reasoning for the decision
    pub reasoning: String,
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
    /// 8D harmony coordinates (projection onto moral manifold)
    pub harmony_coordinates: [f64; N_HARMONIES],
    /// Moral free energy decomposition (FEP)
    pub moral_free_energy: MoralFreeEnergy,
}

/// The harmonies integrator
#[derive(Debug)]
pub struct HarmoniesIntegrator {
    /// Configuration
    config: HarmoniesIntegrationConfig,
    /// The eight harmonies system
    harmonies: EightHarmonies,
    /// Harmony embeddings (semantically grounded via HarmonyBasis)
    harmony_embeddings: HashMap<Harmony, ContinuousHV>,
    /// Shared semantic basis for harmony projection and free energy
    harmony_basis: Arc<HarmonyBasis>,
    /// Running mean of harmony coordinates (prior for moral free energy)
    harmony_prior: [f64; N_HARMONIES],
    /// Number of evaluations contributing to the prior (for EMA)
    prior_count: u64,
    /// Evaluation history
    history: Vec<ValueEvaluation>,
    /// Statistics
    stats: IntegratorStats,
}

/// Statistics for the integrator
#[derive(Debug, Clone, Default)]
pub struct IntegratorStats {
    /// Total evaluations
    pub total_evaluations: u64,
    /// Approved actions
    pub approved: u64,
    /// Rejected actions
    pub rejected: u64,
    /// Average alignment
    pub avg_alignment: f32,
}

impl HarmoniesIntegrator {
    /// Create a new integrator with its own `HarmonyBasis`.
    ///
    /// Harmony basis vectors are built by encoding keyword sets through
    /// `TextHdcEncoder`, so cosine similarity with an action's embedding
    /// reflects genuine semantic alignment rather than random projection.
    pub fn new(config: HarmoniesIntegrationConfig) -> Self {
        let basis = Arc::new(HarmonyBasis::new(config.dimension));
        Self::with_basis(config, basis)
    }

    /// Create a new integrator with a shared `HarmonyBasis`.
    pub fn with_basis(
        config: HarmoniesIntegrationConfig,
        harmony_basis: Arc<HarmonyBasis>,
    ) -> Self {
        let harmonies = EightHarmonies::default();

        // Build harmony embeddings from the semantic basis vectors
        let all_harmonies = Harmony::all();
        let mut harmony_embeddings = HashMap::new();
        for (i, &harmony) in all_harmonies.iter().enumerate() {
            harmony_embeddings.insert(harmony, harmony_basis.vectors[i].clone());
        }

        Self {
            config,
            harmonies,
            harmony_embeddings,
            harmony_basis,
            harmony_prior: [0.0; N_HARMONIES], // uniform prior until data arrives
            prior_count: 0,
            history: Vec::new(),
            stats: IntegratorStats::default(),
        }
    }

    /// Evaluate an action for value alignment.
    ///
    /// Projects the action's embedding onto the 8D harmony manifold, computes
    /// semantic alignment with each harmony, and measures moral free energy
    /// relative to the running prior (lower F → more consistent moral stance).
    pub fn evaluate(&mut self, action: &ValuedAction) -> ValueEvaluation {
        self.stats.total_evaluations += 1;

        let mut harmony_scores = HashMap::new();
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        // Calculate alignment with each harmony via semantic basis
        for (harmony, embedding) in &self.harmony_embeddings {
            let similarity = action.embedding.similarity(embedding);
            let weight = self
                .config
                .harmony_weights
                .get(harmony)
                .copied()
                .unwrap_or(1.0);

            let similarity = if similarity.is_finite() {
                similarity
            } else {
                0.0
            };
            harmony_scores.insert(*harmony, similarity);
            weighted_sum += similarity * weight;
            weight_total += weight;
        }

        let overall_alignment = if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            0.5
        };

        // Project onto harmony manifold and compute moral free energy
        let coords = self.harmony_basis.project(&action.embedding);
        let moral_fe = self.harmony_basis.moral_free_energy(
            &coords,
            &self.harmony_prior,
            1.0, // temperature
        );

        // Update running prior via EMA (alpha=0.1) — skip if coords contain NaN
        let coords_finite = coords.iter().all(|c| c.is_finite());
        if coords_finite {
            let alpha = if self.prior_count == 0 { 1.0 } else { 0.1 };
            for i in 0..N_HARMONIES {
                self.harmony_prior[i] = self.harmony_prior[i] * (1.0 - alpha) + coords[i] * alpha;
            }
        }
        self.prior_count += 1;

        // Determine approval
        let approved = overall_alignment >= self.config.alignment_threshold;

        if approved {
            self.stats.approved += 1;
        } else {
            self.stats.rejected += 1;
        }

        // Update average alignment
        let n = self.stats.total_evaluations as f32;
        self.stats.avg_alignment = (self.stats.avg_alignment * (n - 1.0) + overall_alignment) / n;

        // Generate reasoning
        let reasoning = self.generate_reasoning(&harmony_scores, overall_alignment, approved);

        // Generate suggestions if not approved
        let suggestions = if !approved {
            self.generate_suggestions(&harmony_scores)
        } else {
            Vec::new()
        };

        let evaluation = ValueEvaluation {
            overall_alignment,
            harmony_scores,
            approved,
            reasoning,
            suggestions,
            harmony_coordinates: coords,
            moral_free_energy: moral_fe,
        };

        self.history.push(evaluation.clone());
        evaluation
    }

    /// Generate reasoning for the evaluation
    fn generate_reasoning(
        &self,
        scores: &HashMap<Harmony, f32>,
        overall: f32,
        approved: bool,
    ) -> String {
        let mut reasoning = String::new();

        if approved {
            reasoning.push_str(&format!(
                "Action approved with {:.1}% alignment. ",
                overall * 100.0
            ));
        } else {
            reasoning.push_str(&format!(
                "Action not approved ({:.1}% < {:.1}% threshold). ",
                overall * 100.0,
                self.config.alignment_threshold * 100.0
            ));
        }

        // Find strongest and weakest harmonies
        let mut sorted: Vec<_> = scores.iter().filter(|(_, s)| s.is_finite()).collect();
        sorted.sort_by(|a, b| b.1.total_cmp(a.1));

        if let Some((harmony, score)) = sorted.first() {
            reasoning.push_str(&format!(
                "Strongest alignment: {:?} ({:.1}%). ",
                harmony,
                *score * 100.0
            ));
        }

        if let Some((harmony, score)) = sorted.last() {
            reasoning.push_str(&format!(
                "Weakest alignment: {:?} ({:.1}%).",
                harmony,
                *score * 100.0
            ));
        }

        reasoning
    }

    /// Generate suggestions for improving alignment
    fn generate_suggestions(&self, scores: &HashMap<Harmony, f32>) -> Vec<String> {
        let mut suggestions = Vec::new();

        for (harmony, &score) in scores {
            if score < 0.5 {
                let suggestion = match harmony {
                    Harmony::ResonantCoherence => {
                        "Consider how this action contributes to harmonious integration."
                    }
                    Harmony::PanSentientFlourishing => {
                        "Consider the impact on all sentient beings affected."
                    }
                    Harmony::IntegralWisdom => "Reflect on whether this embodies integral wisdom.",
                    Harmony::InfinitePlay => {
                        "Consider the playful, creative aspects of this action."
                    }
                    Harmony::UniversalInterconnectedness => {
                        "Consider the connections and relationships involved."
                    }
                    Harmony::SacredReciprocity => "Ensure mutual benefit and generative exchange.",
                    Harmony::EvolutionaryProgression => {
                        "Consider how this contributes to evolutionary growth."
                    }
                    Harmony::SacredStillness => {
                        "Honor the need for rest, release, and apophatic not-knowing."
                    }
                };
                suggestions.push(suggestion.to_string());
            }
        }

        suggestions
    }

    /// Update harmony embeddings based on feedback
    pub fn learn(&mut self, action: &ValuedAction, feedback: f32) {
        let learning_rate = self.config.learning_rate * feedback.signum();

        for (harmony, embedding) in self.harmony_embeddings.iter_mut() {
            let weight = self
                .config
                .harmony_weights
                .get(harmony)
                .copied()
                .unwrap_or(1.0);
            let adjustment = action.embedding.clone().scale(learning_rate * weight);
            *embedding = ContinuousHV::bundle_owned(&[embedding.clone(), adjustment]);
        }
    }

    /// Get alignment for the Eight Harmonies
    pub fn check_alignment(&mut self, description: &str) -> AlignmentResult {
        self.harmonies.evaluate(description)
    }

    /// Access the configuration.
    pub fn config(&self) -> &HarmoniesIntegrationConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &IntegratorStats {
        &self.stats
    }

    /// Get approval rate
    pub fn approval_rate(&self) -> f32 {
        if self.stats.total_evaluations == 0 {
            1.0
        } else {
            self.stats.approved as f32 / self.stats.total_evaluations as f32
        }
    }

    /// Access the current harmony prior (running mean of coordinates).
    pub fn harmony_prior(&self) -> &[f64; N_HARMONIES] {
        &self.harmony_prior
    }

    /// Update harmony weights from topology feedback.
    ///
    /// When `MoralTopology::analyze()` reveals that certain harmony dimensions
    /// have low variance (moral blind spots), increase their weight to encourage
    /// broader moral exploration. When a harmony is dominant, slightly reduce
    /// its weight to prevent moral fixation. High KL divergence indicates the
    /// system's moral stance is drifting from its prior — boost the dominant
    /// harmony to stabilize.
    pub fn apply_topology_feedback(
        &mut self,
        harmony_variance: &[f64; N_HARMONIES],
        dominant_harmony_idx: u8,
        completeness: f64,
    ) {
        self.apply_topology_feedback_with_kl(
            harmony_variance,
            dominant_harmony_idx,
            completeness,
            0.0,
        );
    }

    /// Extended topology feedback that also considers KL divergence.
    pub fn apply_topology_feedback_with_kl(
        &mut self,
        harmony_variance: &[f64; N_HARMONIES],
        dominant_harmony_idx: u8,
        completeness: f64,
        kl_divergence: f64,
    ) {
        let all = Harmony::all();

        // Boost weights for under-represented harmonies (low variance = blind spot)
        for i in 0..N_HARMONIES {
            let harmony = all[i];
            let weight = self
                .config
                .harmony_weights
                .get(&harmony)
                .copied()
                .unwrap_or(1.0);

            let variance = harmony_variance[i];
            let adjusted = if variance < 1e-4 {
                // Blind spot: boost weight to encourage exploration
                (weight * 1.15).min(2.0)
            } else if i == dominant_harmony_idx as usize
                && completeness < 0.8
                && kl_divergence <= 0.5
            {
                // Dominant axis with poor completeness and low KL: slight attenuation
                (weight * 0.95).max(0.5)
            } else if i == dominant_harmony_idx as usize && kl_divergence > 0.5 {
                // High KL divergence: moral stance is drifting from prior.
                // Boost the dominant harmony to anchor/stabilize.
                (weight * 1.1).min(2.0)
            } else {
                weight
            };

            self.config.harmony_weights.insert(harmony, adjusted);
        }
    }
}

impl Default for HarmoniesIntegrator {
    fn default() -> Self {
        Self::new(HarmoniesIntegrationConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrator_creation() {
        let integrator = HarmoniesIntegrator::default();
        assert_eq!(integrator.stats.total_evaluations, 0);
        assert_eq!(integrator.prior_count, 0);
    }

    #[test]
    fn test_action_evaluation() {
        let mut integrator = HarmoniesIntegrator::default();

        let action = ValuedAction::new(
            "help_user",
            "Assist user with their request",
            ContinuousHV::random(512, 42),
        );

        let evaluation = integrator.evaluate(&action);
        assert!(evaluation.overall_alignment >= 0.0);
        assert!(evaluation.overall_alignment <= 1.0);
        // Harmony coordinates should be finite 7D
        for c in &evaluation.harmony_coordinates {
            assert!(c.is_finite());
        }
        // Moral free energy should be finite
        assert!(evaluation.moral_free_energy.free_energy.is_finite());
    }

    #[test]
    fn test_moral_free_energy_decreases_with_consistency() {
        let mut integrator = HarmoniesIntegrator::default();

        // Feed the same action repeatedly — free energy should decrease as
        // the prior converges to the observed distribution.
        let action = ValuedAction::new(
            "help",
            "helping others learn and grow",
            ContinuousHV::random(512, 42),
        );

        let mut fe_values = Vec::new();
        for _ in 0..10 {
            let eval = integrator.evaluate(&action);
            fe_values.push(eval.moral_free_energy.kl_divergence);
        }

        // After the first evaluation, KL divergence should trend downward
        // (prior converges via EMA)
        let first_kl = fe_values[1]; // skip 0 (prior starts at zero)
        let last_kl = fe_values[9];
        assert!(
            last_kl <= first_kl + 0.01, // allow small numerical noise
            "KL divergence should decrease with consistent actions: first={first_kl}, last={last_kl}"
        );
    }

    #[test]
    fn test_topology_feedback_adjusts_weights() {
        let mut integrator = HarmoniesIntegrator::default();

        // All harmonies at 1.0 initially
        for h in Harmony::all() {
            assert!((*integrator.config.harmony_weights.get(&h).unwrap() - 1.0).abs() < 0.01,);
        }

        // Simulate low variance on ResonantCoherence (idx 0) = blind spot
        let mut variance = [0.1; N_HARMONIES];
        variance[0] = 0.0; // blind spot

        integrator.apply_topology_feedback(&variance, 3, 0.5);

        // ResonantCoherence weight should have increased
        let rc_weight = *integrator
            .config
            .harmony_weights
            .get(&Harmony::ResonantCoherence)
            .unwrap();
        assert!(
            rc_weight > 1.0,
            "Blind spot harmony should get boosted weight, got {rc_weight}"
        );
    }

    #[test]
    fn test_topology_feedback_to_harmonies() {
        let mut integrator = HarmoniesIntegrator::default();

        // ── Path A: Blind spot boost ──
        // Harmony 0 has zero variance → blind spot
        let mut variance = [0.1; N_HARMONIES];
        variance[0] = 0.0;

        integrator.apply_topology_feedback_with_kl(&variance, 3, 0.5, 0.3);

        let rc_weight = *integrator
            .config
            .harmony_weights
            .get(&Harmony::ResonantCoherence)
            .unwrap();
        assert!(
            rc_weight > 1.0,
            "Blind spot (variance=0) should boost weight, got {rc_weight}"
        );

        // ── Path B: Dominant attenuation (low KL, low completeness) ──
        for h in Harmony::all() {
            integrator.config.harmony_weights.insert(h, 1.0);
        }
        let uniform_variance = [0.1; N_HARMONIES];
        // dominant=2, completeness=0.5 (<0.8), kl=0.3 (<=0.5) → attenuate
        integrator.apply_topology_feedback_with_kl(&uniform_variance, 2, 0.5, 0.3);

        let dominant_weight = *integrator
            .config
            .harmony_weights
            .get(&Harmony::all()[2])
            .unwrap();
        assert!(
            dominant_weight < 1.0,
            "Dominant axis with low completeness & low KL should be attenuated, got {dominant_weight}"
        );
        // Non-dominant harmonies should be unchanged
        let other_weight = *integrator
            .config
            .harmony_weights
            .get(&Harmony::all()[5])
            .unwrap();
        assert!(
            (other_weight - 1.0_f32).abs() < f32::EPSILON,
            "Non-dominant, non-blind-spot harmony should be unchanged, got {other_weight}"
        );

        // ── Path C: High KL anchoring ──
        for h in Harmony::all() {
            integrator.config.harmony_weights.insert(h, 1.0);
        }
        // dominant=4, completeness=0.5 (doesn't matter), kl=0.8 (>0.5) → anchor/boost
        integrator.apply_topology_feedback_with_kl(&uniform_variance, 4, 0.5, 0.8);

        let anchored_weight = *integrator
            .config
            .harmony_weights
            .get(&Harmony::all()[4])
            .unwrap();
        assert!(
            anchored_weight > 1.0,
            "Dominant axis with high KL should be anchored (boosted), got {anchored_weight}"
        );

        // ── Path D: Multi-step accumulation — blind spot boost compounds ──
        for h in Harmony::all() {
            integrator.config.harmony_weights.insert(h, 1.0);
        }
        let mut blind_spot_variance = [0.1; N_HARMONIES];
        blind_spot_variance[6] = 0.0; // harmony 6 is a persistent blind spot

        for _ in 0..10 {
            integrator.apply_topology_feedback_with_kl(&blind_spot_variance, 0, 0.9, 0.1);
        }

        let boosted = *integrator
            .config
            .harmony_weights
            .get(&Harmony::all()[6])
            .unwrap();
        assert!(
            boosted > 1.5,
            "10 rounds of blind spot boost should compound significantly, got {boosted}"
        );
        assert!(
            boosted <= 2.0,
            "Weight should be clamped at 2.0, got {boosted}"
        );
    }

    #[test]
    fn test_valued_action_builder() {
        let action = ValuedAction::new("test", "description", ContinuousHV::random(512, 42))
            .with_outcome("positive outcome")
            .with_entity("user");

        assert_eq!(action.expected_outcomes.len(), 1);
        assert_eq!(action.affected_entities.len(), 1);
    }
}
