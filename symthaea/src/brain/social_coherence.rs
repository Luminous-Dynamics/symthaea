//! # Social Coherence: Theory of Mind and Social Reasoning
//!
//! Implements social cognition capabilities including:
//! - Theory of Mind (ToM) - modeling others' mental states
//! - Social relationship tracking
//! - Cooperative reasoning
//! - Trust and reputation modeling

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use symthaea_core::hdc::RealHV;

/// Configuration for social coherence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialCoherenceConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Maximum agents to model
    pub max_agents: usize,
    /// Trust decay rate per interaction
    pub trust_decay: f32,
    /// Minimum trust threshold for cooperation
    pub cooperation_threshold: f32,
    /// Memory for past interactions
    pub interaction_memory: usize,
}

impl Default for SocialCoherenceConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            max_agents: 100,
            trust_decay: 0.01,
            cooperation_threshold: 0.3,
            interaction_memory: 50,
        }
    }
}

/// Mental state model of another agent
#[derive(Debug, Clone)]
pub struct MentalModel {
    /// Agent identifier
    pub agent_id: String,
    /// Estimated beliefs (as embedding)
    pub beliefs: RealHV,
    /// Estimated desires/goals
    pub desires: RealHV,
    /// Estimated intentions
    pub intentions: RealHV,
    /// Estimated emotional state
    pub emotional_state: EmotionalState,
    /// Confidence in this model
    pub confidence: f32,
    /// Last update timestamp
    pub last_updated: u64,
    /// Number of observations
    pub observation_count: u64,
}

/// Simplified emotional state for modeling others
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EmotionalState {
    /// Positive/negative valence (-1 to 1)
    pub valence: f32,
    /// Low/high arousal (0 to 1)
    pub arousal: f32,
    /// Trust level toward self (0 to 1)
    pub trust_toward_us: f32,
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            trust_toward_us: 0.5,
        }
    }
}

/// A social relationship between agents
#[derive(Debug, Clone)]
pub struct Relationship {
    /// Agent we have relationship with
    pub agent_id: String,
    /// Relationship type
    pub relationship_type: RelationshipType,
    /// Trust level (0 to 1)
    pub trust: f32,
    /// Familiarity level (0 to 1)
    pub familiarity: f32,
    /// Reciprocity balance (-1 to 1)
    pub reciprocity: f32,
    /// History of interactions
    pub interaction_history: Vec<Interaction>,
    /// Total positive interactions
    pub positive_interactions: u64,
    /// Total negative interactions
    pub negative_interactions: u64,
}

/// Type of social relationship
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Unknown/new agent
    Unknown,
    /// Acquaintance
    Acquaintance,
    /// Cooperative partner
    Ally,
    /// Competitive relationship
    Rival,
    /// Trusted collaborator
    Friend,
    /// Authoritative relationship
    Authority,
    /// Subordinate relationship
    Subordinate,
}

/// Record of a social interaction
#[derive(Debug, Clone)]
pub struct Interaction {
    /// Interaction timestamp
    pub timestamp: u64,
    /// Type of interaction
    pub interaction_type: InteractionType,
    /// Outcome valence (-1 to 1)
    pub outcome: f32,
    /// Context embedding
    pub context: RealHV,
    /// Our action
    pub our_action: String,
    /// Their response
    pub their_response: String,
}

/// Type of social interaction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionType {
    /// Cooperative exchange
    Cooperation,
    /// Competitive interaction
    Competition,
    /// Information sharing
    Communication,
    /// Negotiation
    Negotiation,
    /// Conflict
    Conflict,
    /// Help given/received
    Help,
    /// Resource exchange
    Exchange,
}

/// Result of social reasoning
#[derive(Debug, Clone)]
pub struct SocialReasoningResult {
    /// Recommended action
    pub recommended_action: String,
    /// Predicted response from other agent
    pub predicted_response: String,
    /// Confidence in prediction
    pub confidence: f32,
    /// Risk assessment
    pub risk: f32,
    /// Potential reward
    pub potential_reward: f32,
    /// Reasoning explanation
    pub reasoning: Vec<String>,
}

/// The social coherence system
#[derive(Debug)]
pub struct SocialCoherence {
    /// Configuration
    config: SocialCoherenceConfig,
    /// Mental models of other agents
    mental_models: HashMap<String, MentalModel>,
    /// Relationships with other agents
    relationships: HashMap<String, Relationship>,
    /// Self-model for comparison
    self_model: MentalModel,
    /// Current timestamp
    timestamp: u64,
    /// Statistics
    stats: SocialCoherenceStats,
}

/// Statistics for social coherence
#[derive(Debug, Clone, Default)]
pub struct SocialCoherenceStats {
    /// Total agents modeled
    pub agents_modeled: u64,
    /// Total interactions recorded
    pub interactions_recorded: u64,
    /// Successful predictions
    pub successful_predictions: u64,
    /// Total predictions made
    pub total_predictions: u64,
    /// Average trust level
    pub avg_trust: f32,
    /// Cooperation rate
    pub cooperation_rate: f32,
}

impl SocialCoherence {
    /// Create a new social coherence system
    pub fn new(config: SocialCoherenceConfig) -> Self {
        let dim = config.dimension;

        // Initialize self-model with deterministic seeds
        let self_model = MentalModel {
            agent_id: "self".to_string(),
            beliefs: RealHV::random(dim, 0x5E1F_BE1F), // "SELF_BEL"
            desires: RealHV::random(dim, 0x5E1F_DE51), // "SELF_DES"
            intentions: RealHV::random(dim, 0x5E1F_1714), // "SELF_INT"
            emotional_state: EmotionalState::default(),
            confidence: 1.0,
            last_updated: 0,
            observation_count: 0,
        };

        Self {
            config,
            mental_models: HashMap::new(),
            relationships: HashMap::new(),
            self_model,
            timestamp: 0,
            stats: SocialCoherenceStats::default(),
        }
    }

    /// Observe an agent and update mental model
    pub fn observe_agent(
        &mut self,
        agent_id: &str,
        observed_behavior: &RealHV,
        context: &RealHV,
    ) {
        self.timestamp += 1;

        // Hash agent_id for deterministic seeds
        let mut seed: u64 = 0x0B5E_BEE7; // "OBSERVE"
        for byte in agent_id.bytes() {
            seed = seed.wrapping_mul(31).wrapping_add(byte as u64);
        }
        let dim = self.config.dimension;
        let ts = self.timestamp;
        let model = self.mental_models.entry(agent_id.to_string()).or_insert_with(|| {
            MentalModel {
                agent_id: agent_id.to_string(),
                beliefs: RealHV::random(dim, seed),
                desires: RealHV::random(dim, seed.wrapping_add(1)),
                intentions: RealHV::random(dim, seed.wrapping_add(2)),
                emotional_state: EmotionalState::default(),
                confidence: 0.1,
                last_updated: ts,
                observation_count: 0,
            }
        });

        // Update model based on observation
        model.intentions = RealHV::bundle(&[model.intentions.clone(), observed_behavior.clone().scale(0.3)]);
        model.beliefs = RealHV::bundle(&[model.beliefs.clone(), context.clone().scale(0.1)]);
        model.observation_count += 1;
        model.last_updated = self.timestamp;

        // Increase confidence with more observations
        model.confidence = (model.confidence + 0.05).min(0.95);

        // Update stats after model borrow is done
        self.stats.agents_modeled = self.mental_models.len() as u64;
    }

    /// Record an interaction with an agent
    pub fn record_interaction(
        &mut self,
        agent_id: &str,
        interaction_type: InteractionType,
        outcome: f32,
        context: RealHV,
        our_action: impl Into<String>,
        their_response: impl Into<String>,
    ) {
        self.timestamp += 1;
        self.stats.interactions_recorded += 1;

        let relationship = self.relationships.entry(agent_id.to_string()).or_insert_with(|| {
            Relationship {
                agent_id: agent_id.to_string(),
                relationship_type: RelationshipType::Unknown,
                trust: 0.5,
                familiarity: 0.0,
                reciprocity: 0.0,
                interaction_history: Vec::new(),
                positive_interactions: 0,
                negative_interactions: 0,
            }
        });

        // Record interaction
        let interaction = Interaction {
            timestamp: self.timestamp,
            interaction_type,
            outcome,
            context,
            our_action: our_action.into(),
            their_response: their_response.into(),
        };

        relationship.interaction_history.push(interaction);
        if relationship.interaction_history.len() > self.config.interaction_memory {
            relationship.interaction_history.remove(0);
        }

        // Update trust based on outcome
        if outcome > 0.0 {
            relationship.positive_interactions += 1;
            relationship.trust = (relationship.trust + outcome * 0.1).min(1.0);
        } else {
            relationship.negative_interactions += 1;
            relationship.trust = (relationship.trust + outcome * 0.1).max(0.0);
        }

        // Update familiarity
        relationship.familiarity = (relationship.familiarity + 0.05).min(1.0);

        // Update reciprocity
        if interaction_type == InteractionType::Cooperation || interaction_type == InteractionType::Help {
            relationship.reciprocity = (relationship.reciprocity + 0.1).min(1.0);
        }

        // Save trust value before releasing borrow (NLL ends borrow here automatically)
        let current_trust = relationship.trust;

        // Classify relationship type (needs &mut self)
        self.classify_relationship(agent_id);

        // Update agent's emotional model
        if let Some(model) = self.mental_models.get_mut(agent_id) {
            model.emotional_state.trust_toward_us = current_trust;
            if outcome > 0.0 {
                model.emotional_state.valence = (model.emotional_state.valence + 0.1).min(1.0);
            } else {
                model.emotional_state.valence = (model.emotional_state.valence - 0.1).max(-1.0);
            }
        }

        // Update average trust
        let total_trust: f32 = self.relationships.values().map(|r| r.trust).sum();
        self.stats.avg_trust = total_trust / self.relationships.len().max(1) as f32;
    }

    /// Classify relationship type based on history
    fn classify_relationship(&mut self, agent_id: &str) {
        if let Some(rel) = self.relationships.get_mut(agent_id) {
            let total = (rel.positive_interactions + rel.negative_interactions) as f32;
            if total < 3.0 {
                rel.relationship_type = RelationshipType::Unknown;
            } else {
                let positive_ratio = rel.positive_interactions as f32 / total;
                rel.relationship_type = if positive_ratio > 0.8 && rel.trust > 0.7 {
                    RelationshipType::Friend
                } else if positive_ratio > 0.6 {
                    RelationshipType::Ally
                } else if positive_ratio > 0.4 {
                    RelationshipType::Acquaintance
                } else {
                    RelationshipType::Rival
                };
            }
        }
    }

    /// Predict how an agent will respond to an action
    pub fn predict_response(
        &self,
        agent_id: &str,
        proposed_action: &RealHV,
    ) -> Option<SocialReasoningResult> {
        let model = self.mental_models.get(agent_id)?;
        let relationship = self.relationships.get(agent_id);

        let trust = relationship.map(|r| r.trust).unwrap_or(0.5);

        // Estimate alignment between action and agent's intentions
        let alignment = proposed_action.similarity(&model.intentions);

        // Calculate predicted response characteristics
        let predicted_valence = if alignment > 0.5 {
            "positive"
        } else if alignment < -0.5 {
            "negative"
        } else {
            "neutral"
        };

        let confidence = model.confidence * (0.5 + trust * 0.5);

        let mut reasoning = Vec::new();
        reasoning.push(format!("Agent {} has {} confidence model", agent_id, model.confidence));
        reasoning.push(format!("Action-intention alignment: {:.2}", alignment));
        reasoning.push(format!("Trust level: {:.2}", trust));

        let risk = if alignment < 0.0 { (-alignment * 0.5).min(1.0) } else { 0.1 };
        let potential_reward = if alignment > 0.0 { alignment * trust } else { 0.0 };

        Some(SocialReasoningResult {
            recommended_action: format!("Proceed with {} caution", if risk > 0.5 { "high" } else { "low" }),
            predicted_response: format!("{} response expected", predicted_valence),
            confidence,
            risk,
            potential_reward,
            reasoning,
        })
    }

    /// Decide whether to cooperate with an agent
    pub fn should_cooperate(&self, agent_id: &str) -> bool {
        if let Some(relationship) = self.relationships.get(agent_id) {
            relationship.trust >= self.config.cooperation_threshold
        } else {
            // Default to cautious cooperation with unknown agents
            true
        }
    }

    /// Get mental model for an agent
    pub fn get_mental_model(&self, agent_id: &str) -> Option<&MentalModel> {
        self.mental_models.get(agent_id)
    }

    /// Get relationship with an agent
    pub fn get_relationship(&self, agent_id: &str) -> Option<&Relationship> {
        self.relationships.get(agent_id)
    }

    /// Get all allies (trusted cooperative agents)
    pub fn get_allies(&self) -> Vec<&Relationship> {
        self.relationships.values()
            .filter(|r| matches!(r.relationship_type, RelationshipType::Ally | RelationshipType::Friend))
            .collect()
    }

    /// Get all rivals
    pub fn get_rivals(&self) -> Vec<&Relationship> {
        self.relationships.values()
            .filter(|r| r.relationship_type == RelationshipType::Rival)
            .collect()
    }

    /// Get self model
    pub fn self_model(&self) -> &MentalModel {
        &self.self_model
    }

    /// Update self model
    pub fn update_self_model(&mut self, beliefs: RealHV, desires: RealHV, intentions: RealHV) {
        self.self_model.beliefs = beliefs;
        self.self_model.desires = desires;
        self.self_model.intentions = intentions;
        self.self_model.last_updated = self.timestamp;
    }

    /// Get statistics
    pub fn stats(&self) -> &SocialCoherenceStats {
        &self.stats
    }

    /// Decay trust for all relationships over time
    pub fn decay_trust(&mut self) {
        for relationship in self.relationships.values_mut() {
            // Decay toward neutral (0.5)
            let decay = self.config.trust_decay;
            if relationship.trust > 0.5 {
                relationship.trust = (relationship.trust - decay).max(0.5);
            } else if relationship.trust < 0.5 {
                relationship.trust = (relationship.trust + decay).min(0.5);
            }
        }
    }
}

impl Default for SocialCoherence {
    fn default() -> Self {
        Self::new(SocialCoherenceConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_social_coherence_creation() {
        let sc = SocialCoherence::default();
        assert_eq!(sc.mental_models.len(), 0);
        assert_eq!(sc.relationships.len(), 0);
    }

    #[test]
    fn test_agent_observation() {
        let mut sc = SocialCoherence::default();

        let behavior = RealHV::random(512, 0xBEEF_0001);
        let context = RealHV::random(512, 0xBEEF_0002);

        sc.observe_agent("agent1", &behavior, &context);

        assert!(sc.get_mental_model("agent1").is_some());
    }

    #[test]
    fn test_interaction_recording() {
        let mut sc = SocialCoherence::default();

        sc.record_interaction(
            "agent1",
            InteractionType::Cooperation,
            0.8,
            RealHV::random(512, 0xBEEF_0003),
            "helped",
            "thanked",
        );

        let rel = sc.get_relationship("agent1").unwrap();
        assert!(rel.trust > 0.5);
        assert_eq!(rel.positive_interactions, 1);
    }

    #[test]
    fn test_cooperation_decision() {
        let mut sc = SocialCoherence::default();

        // Build trust through positive interactions
        for i in 0..5 {
            sc.record_interaction(
                "trusted_agent",
                InteractionType::Cooperation,
                0.9,
                RealHV::random(512, 0xBEEF_0004 + i as u64),
                "cooperated",
                "cooperated back",
            );
        }

        assert!(sc.should_cooperate("trusted_agent"));

        // Build distrust through negative interactions
        for i in 0..10 {
            sc.record_interaction(
                "untrusted_agent",
                InteractionType::Conflict,
                -0.8,
                RealHV::random(512, 0xBEEF_0010 + i as u64),
                "offered",
                "betrayed",
            );
        }

        assert!(!sc.should_cooperate("untrusted_agent"));
    }

    #[test]
    fn test_relationship_classification() {
        let mut sc = SocialCoherence::default();

        // Many positive interactions should create ally/friend
        for i in 0..10 {
            sc.record_interaction(
                "friend",
                InteractionType::Help,
                0.9,
                RealHV::random(512, 0xBEEF_0020 + i as u64),
                "helped",
                "helped back",
            );
        }

        let rel = sc.get_relationship("friend").unwrap();
        assert!(matches!(rel.relationship_type, RelationshipType::Friend | RelationshipType::Ally));
    }
}
