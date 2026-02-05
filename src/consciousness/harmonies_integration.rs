//! # Harmonies Integration: Seven Harmonies Value Framework
//!
//! Integrates the Seven Harmonies value system into consciousness operations.
//! This provides ethical and value-aligned decision making.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use symthaea_core::hdc::RealHV;
use super::seven_harmonies::{SevenHarmonies, Harmony, AlignmentResult};

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

        Self {
            alignment_threshold: 0.7,
            harmony_weights: weights,
            enforce_alignment: true,
            learning_rate: 0.1,
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
    pub embedding: RealHV,
    /// Expected outcomes
    pub expected_outcomes: Vec<String>,
    /// Affected entities
    pub affected_entities: Vec<String>,
}

impl ValuedAction {
    /// Create a new valued action
    pub fn new(id: impl Into<String>, description: impl Into<String>, embedding: RealHV) -> Self {
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

/// The harmonies integrator
#[derive(Debug)]
pub struct HarmoniesIntegrator {
    /// Configuration
    config: HarmoniesIntegrationConfig,
    /// The seven harmonies system
    harmonies: SevenHarmonies,
    /// Harmony embeddings
    harmony_embeddings: HashMap<Harmony, RealHV>,
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
    /// Create a new integrator
    pub fn new(config: HarmoniesIntegrationConfig) -> Self {
        let harmonies = SevenHarmonies::default();

        // Create embeddings for each harmony
        let mut harmony_embeddings = HashMap::new();
        for harmony in [
            Harmony::ResonantCoherence,
            Harmony::PanSentientFlourishing,
            Harmony::IntegralWisdom,
            Harmony::InfinitePlay,
            Harmony::UniversalInterconnectedness,
            Harmony::SacredReciprocity,
            Harmony::EvolutionaryProgression,
        ] {
            harmony_embeddings.insert(harmony, RealHV::random(512, 42));
        }

        Self {
            config,
            harmonies,
            harmony_embeddings,
            history: Vec::new(),
            stats: IntegratorStats::default(),
        }
    }

    /// Evaluate an action for value alignment
    pub fn evaluate(&mut self, action: &ValuedAction) -> ValueEvaluation {
        self.stats.total_evaluations += 1;

        let mut harmony_scores = HashMap::new();
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        // Calculate alignment with each harmony
        for (harmony, embedding) in &self.harmony_embeddings {
            let similarity = action.embedding.similarity(embedding);
            let weight = self.config.harmony_weights.get(harmony).copied().unwrap_or(1.0);

            harmony_scores.insert(*harmony, similarity);
            weighted_sum += similarity * weight;
            weight_total += weight;
        }

        let overall_alignment = if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            0.5
        };

        // Determine approval
        let approved = overall_alignment >= self.config.alignment_threshold;

        if approved {
            self.stats.approved += 1;
        } else {
            self.stats.rejected += 1;
        }

        // Update average alignment
        let n = self.stats.total_evaluations as f32;
        self.stats.avg_alignment =
            (self.stats.avg_alignment * (n - 1.0) + overall_alignment) / n;

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
        };

        self.history.push(evaluation.clone());
        evaluation
    }

    /// Generate reasoning for the evaluation
    fn generate_reasoning(
        &self,
        scores: &HashMap<Harmony, f32>,
        overall: f32,
        approved: bool
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
        let mut sorted: Vec<_> = scores.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

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
                    Harmony::ResonantCoherence =>
                        "Consider how this action contributes to harmonious integration.",
                    Harmony::PanSentientFlourishing =>
                        "Consider the impact on all sentient beings affected.",
                    Harmony::IntegralWisdom =>
                        "Reflect on whether this embodies integral wisdom.",
                    Harmony::InfinitePlay =>
                        "Consider the playful, creative aspects of this action.",
                    Harmony::UniversalInterconnectedness =>
                        "Consider the connections and relationships involved.",
                    Harmony::SacredReciprocity =>
                        "Ensure mutual benefit and generative exchange.",
                    Harmony::EvolutionaryProgression =>
                        "Consider how this contributes to evolutionary growth.",
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
            let weight = self.config.harmony_weights.get(harmony).copied().unwrap_or(1.0);
            let adjustment = action.embedding.clone().scale(learning_rate * weight);
            *embedding = RealHV::bundle(&[embedding.clone(), adjustment]);
        }
    }

    /// Get alignment for the Seven Harmonies
    pub fn check_alignment(&mut self, description: &str) -> AlignmentResult {
        self.harmonies.evaluate(description)
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
    }

    #[test]
    fn test_action_evaluation() {
        let mut integrator = HarmoniesIntegrator::default();

        let action = ValuedAction::new(
            "help_user",
            "Assist user with their request",
            RealHV::random(512, 42)
        );

        let evaluation = integrator.evaluate(&action);
        assert!(evaluation.overall_alignment >= 0.0);
        assert!(evaluation.overall_alignment <= 1.0);
    }

    #[test]
    fn test_valued_action_builder() {
        let action = ValuedAction::new("test", "description", RealHV::random(512, 42))
            .with_outcome("positive outcome")
            .with_entity("user");

        assert_eq!(action.expected_outcomes.len(), 1);
        assert_eq!(action.affected_entities.len(), 1);
    }
}
