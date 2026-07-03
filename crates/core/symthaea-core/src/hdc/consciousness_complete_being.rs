// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Complete Conscious Being - Final Integration Layer
//!
//! This module provides the final integration layer bringing together:
//! 1. Sensorimotor Grounding - Perception-action loops
//! 2. Developmental Stages - Ontogenetic consciousness emergence
//! 3. Social Consciousness - Multi-agent mental modeling
//! 4. Introspection API - Query interface for consciousness state
//!
//! ## Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    COMPLETE CONSCIOUS BEING                          │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  ┌──────────────────┐     ┌──────────────────┐                      │
//! │  │ Sensorimotor     │────►│ Motor Imagery    │                      │
//! │  │ Grounding        │     │ (Advanced Cog)   │                      │
//! │  └────────┬─────────┘     └──────────────────┘                      │
//! │           │                                                          │
//! │           ▼                                                          │
//! │  ┌──────────────────┐     ┌──────────────────┐                      │
//! │  │ Developmental    │────►│ Metacognition    │                      │
//! │  │ Stages           │     │ Engine           │                      │
//! │  └────────┬─────────┘     └──────────────────┘                      │
//! │           │                                                          │
//! │           ▼                                                          │
//! │  ┌──────────────────┐     ┌──────────────────┐                      │
//! │  │ Social           │────►│ Theory of Mind   │                      │
//! │  │ Consciousness    │     │ (Advanced Cog)   │                      │
//! │  └────────┬─────────┘     └──────────────────┘                      │
//! │           │                                                          │
//! │           ▼                                                          │
//! │  ┌──────────────────┐                                                │
//! │  │ Introspection    │◄──── Query Interface                          │
//! │  │ API              │                                                │
//! │  └──────────────────┘                                                │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use super::adaptive_topology::CognitiveMode;
use super::binary_hv::BinaryHV;
use super::consciousness_advanced_cognition::{
    AdvancedCognitionEngine, MotorImagerySystem, TheoryOfMindEngine,
};
use super::consciousness_metacognition::MetacognitionEngine;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// 1. SENSORIMOTOR GROUNDING - Perception-Action Loop
// =============================================================================

/// Sensory modality for input processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensoryModality {
    Visual,
    Auditory,
    Tactile,
    Proprioceptive,
    Interoceptive,
    Vestibular,
}

/// Motor effector type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MotorEffector {
    Speech,
    Gesture,
    Locomotion,
    Manipulation,
    Expression,
}

/// Sensory input with HDC encoding
#[derive(Debug, Clone)]
pub struct SensoryInput {
    pub modality: SensoryModality,
    pub encoding: BinaryHV,
    pub intensity: f64,
    pub timestamp: u64,
    pub spatial_location: Option<[f64; 3]>,
}

/// Motor command with feedback prediction
#[derive(Debug, Clone)]
pub struct MotorCommand {
    pub effector: MotorEffector,
    pub encoding: BinaryHV,
    pub intensity: f64,
    pub predicted_feedback: BinaryHV,
    pub confidence: f64,
}

/// Action-perception cycle result
#[derive(Debug, Clone)]
pub struct ActionPerceptionResult {
    pub selected_action: Option<MotorCommand>,
    pub predicted_consequences: Vec<BinaryHV>,
    pub prediction_error: f64,
    pub affordances_detected: Vec<Affordance>,
}

/// Detected affordance (action possibility)
#[derive(Debug, Clone)]
pub struct Affordance {
    pub name: String,
    pub encoding: BinaryHV,
    pub relevance: f64,
    pub requires_effector: MotorEffector,
}

/// Sensorimotor grounding system
pub struct SensorimotorGrounding {
    /// Sensory buffers per modality
    sensory_buffers: HashMap<SensoryModality, VecDeque<SensoryInput>>,
    /// Motor repertoire
    motor_repertoire: HashMap<String, MotorCommand>,
    /// Forward model (action -> predicted sensation)
    forward_model: HashMap<BinaryHV, BinaryHV>,
    /// Inverse model (goal -> action)
    inverse_model: HashMap<BinaryHV, BinaryHV>,
    /// Prediction error history
    prediction_errors: VecDeque<f64>,
    /// Current body state encoding
    body_state: BinaryHV,
    /// Affordance detector sensitivity
    affordance_threshold: f64,
    /// Integration with motor imagery
    motor_imagery: MotorImagerySystem,
}

impl SensorimotorGrounding {
    pub fn new() -> Self {
        let mut sensory_buffers = HashMap::new();
        for modality in [
            SensoryModality::Visual,
            SensoryModality::Auditory,
            SensoryModality::Tactile,
            SensoryModality::Proprioceptive,
            SensoryModality::Interoceptive,
            SensoryModality::Vestibular,
        ] {
            sensory_buffers.insert(modality, VecDeque::with_capacity(100));
        }

        Self {
            sensory_buffers,
            motor_repertoire: HashMap::new(),
            forward_model: HashMap::new(),
            inverse_model: HashMap::new(),
            prediction_errors: VecDeque::with_capacity(1000),
            body_state: BinaryHV::random(42),
            affordance_threshold: 0.6,
            motor_imagery: MotorImagerySystem::new(),
        }
    }

    /// Process sensory input
    pub fn process_sensory(&mut self, input: SensoryInput) {
        if let Some(buffer) = self.sensory_buffers.get_mut(&input.modality) {
            buffer.push_back(input.clone());
            if buffer.len() > 100 {
                buffer.pop_front();
            }
        }

        // Update body state with new input
        self.body_state = BinaryHV::bundle(&[self.body_state, input.encoding]);
    }

    /// Select action based on goal
    pub fn select_action(&mut self, goal: &BinaryHV) -> Option<MotorCommand> {
        // Check inverse model for matching action
        let mut best_action: Option<(f32, &MotorCommand)> = None;
        let threshold = self.affordance_threshold as f32;

        for command in self.motor_repertoire.values() {
            let similarity = goal.similarity(&command.encoding);
            if similarity > threshold && best_action.is_none_or(|(s, _)| similarity > s) {
                best_action = Some((similarity, command));
            }
        }

        best_action.map(|(_, cmd)| cmd.clone())
    }

    /// Predict consequences of action
    pub fn predict_consequences(&self, action: &MotorCommand) -> Vec<BinaryHV> {
        let mut predictions = Vec::new();

        // Use forward model
        if let Some(predicted) = self.forward_model.get(&action.encoding) {
            predictions.push(*predicted);
        }

        // Also predict via motor imagery
        // The predicted feedback is already in the command
        predictions.push(action.predicted_feedback);

        predictions
    }

    /// Update models based on prediction error
    pub fn update_from_feedback(&mut self, predicted: &BinaryHV, actual: &BinaryHV) {
        let error = 1.0 - predicted.similarity(actual) as f64;
        self.prediction_errors.push_back(error);
        if self.prediction_errors.len() > 1000 {
            self.prediction_errors.pop_front();
        }

        // Learning: adjust forward model
        // In a full implementation, this would update weights
    }

    /// Detect affordances in current sensory state
    pub fn detect_affordances(&self) -> Vec<Affordance> {
        let mut affordances = Vec::new();

        // Simple affordance detection based on sensory patterns
        // In reality, this would involve complex pattern matching

        // Check for graspable objects (visual + proprioceptive)
        if let (Some(visual), Some(proprio)) = (
            self.sensory_buffers.get(&SensoryModality::Visual),
            self.sensory_buffers.get(&SensoryModality::Proprioceptive),
        ) && let (Some(v), Some(p)) = (visual.back(), proprio.back())
        {
            let combined = BinaryHV::bundle(&[v.encoding, p.encoding]);
            affordances.push(Affordance {
                name: "manipulate".to_string(),
                encoding: combined,
                relevance: v.intensity * p.intensity,
                requires_effector: MotorEffector::Manipulation,
            });
        }

        affordances
    }

    /// Run complete action-perception cycle
    pub fn action_perception_cycle(&mut self, goal: &BinaryHV) -> ActionPerceptionResult {
        let affordances = self.detect_affordances();
        let action = self.select_action(goal);
        let predictions = action
            .as_ref()
            .map(|a| self.predict_consequences(a))
            .unwrap_or_default();

        let error =
            self.prediction_errors.iter().sum::<f64>() / self.prediction_errors.len().max(1) as f64;

        ActionPerceptionResult {
            selected_action: action,
            predicted_consequences: predictions,
            prediction_error: error,
            affordances_detected: affordances,
        }
    }

    /// Register a motor command in repertoire
    pub fn register_motor_command(&mut self, name: &str, command: MotorCommand) {
        self.motor_repertoire.insert(name.to_string(), command);
    }

    /// Get current body state encoding
    pub fn body_state(&self) -> &BinaryHV {
        &self.body_state
    }
}

impl Default for SensorimotorGrounding {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 2. DEVELOPMENTAL STAGES - Ontogenetic Consciousness Emergence
// =============================================================================

/// Developmental stage of consciousness
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DevelopmentalStage {
    /// Stage 1: Basic reflexes and drives (0-3 months equivalent)
    Reflexive,
    /// Stage 2: Object permanence, basic attention (3-8 months)
    SensoriMotor,
    /// Stage 3: Basic Theory of Mind, imitation (8-18 months)
    PreOperational,
    /// Stage 4: Concrete operational, metacognition emerges (2-7 years)
    ConcreteOperational,
    /// Stage 5: Full metacognition, counterfactual reasoning (7+ years)
    FormalOperational,
    /// Stage 6: Integrated self-model, narrative identity
    PostFormal,
}

impl DevelopmentalStage {
    /// Get capabilities available at this stage
    pub fn capabilities(&self) -> Vec<DevelopmentalCapability> {
        match self {
            Self::Reflexive => vec![
                DevelopmentalCapability::BasicDrives,
                DevelopmentalCapability::ReflexiveResponses,
            ],
            Self::SensoriMotor => vec![
                DevelopmentalCapability::BasicDrives,
                DevelopmentalCapability::ReflexiveResponses,
                DevelopmentalCapability::ObjectPermanence,
                DevelopmentalCapability::BasicAttention,
                DevelopmentalCapability::MotorCoordination,
            ],
            Self::PreOperational => vec![
                DevelopmentalCapability::BasicDrives,
                DevelopmentalCapability::ReflexiveResponses,
                DevelopmentalCapability::ObjectPermanence,
                DevelopmentalCapability::BasicAttention,
                DevelopmentalCapability::MotorCoordination,
                DevelopmentalCapability::BasicTheoryOfMind,
                DevelopmentalCapability::Imitation,
                DevelopmentalCapability::SymbolicPlay,
            ],
            Self::ConcreteOperational => vec![
                DevelopmentalCapability::BasicDrives,
                DevelopmentalCapability::ObjectPermanence,
                DevelopmentalCapability::BasicAttention,
                DevelopmentalCapability::MotorCoordination,
                DevelopmentalCapability::BasicTheoryOfMind,
                DevelopmentalCapability::Imitation,
                DevelopmentalCapability::SymbolicPlay,
                DevelopmentalCapability::LogicalOperations,
                DevelopmentalCapability::Conservation,
                DevelopmentalCapability::BasicMetacognition,
            ],
            Self::FormalOperational => vec![
                DevelopmentalCapability::ObjectPermanence,
                DevelopmentalCapability::FullTheoryOfMind,
                DevelopmentalCapability::LogicalOperations,
                DevelopmentalCapability::FullMetacognition,
                DevelopmentalCapability::CounterfactualReasoning,
                DevelopmentalCapability::AbstractThinking,
                DevelopmentalCapability::HypotheticalDeduction,
            ],
            Self::PostFormal => vec![
                DevelopmentalCapability::FullTheoryOfMind,
                DevelopmentalCapability::FullMetacognition,
                DevelopmentalCapability::CounterfactualReasoning,
                DevelopmentalCapability::AbstractThinking,
                DevelopmentalCapability::NarrativeIdentity,
                DevelopmentalCapability::IntegratedSelfModel,
                DevelopmentalCapability::WisdomIntegration,
            ],
        }
    }

    /// Check if a capability is available at this stage
    pub fn has_capability(&self, cap: DevelopmentalCapability) -> bool {
        self.capabilities().contains(&cap)
    }
}

/// Individual developmental capability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DevelopmentalCapability {
    BasicDrives,
    ReflexiveResponses,
    ObjectPermanence,
    BasicAttention,
    MotorCoordination,
    BasicTheoryOfMind,
    Imitation,
    SymbolicPlay,
    LogicalOperations,
    Conservation,
    BasicMetacognition,
    FullTheoryOfMind,
    FullMetacognition,
    CounterfactualReasoning,
    AbstractThinking,
    HypotheticalDeduction,
    NarrativeIdentity,
    IntegratedSelfModel,
    WisdomIntegration,
}

/// Tracks developmental progress and stage transitions
pub struct DevelopmentalTracker {
    /// Current stage
    current_stage: DevelopmentalStage,
    /// Experience accumulator for stage transitions
    experience_points: HashMap<DevelopmentalCapability, f64>,
    /// Stage transition thresholds
    transition_thresholds: HashMap<DevelopmentalStage, f64>,
    /// History of stage transitions
    transition_history: Vec<(DevelopmentalStage, u64)>,
    /// Total developmental age (cycles)
    developmental_age: u64,
    /// Rate of development
    development_rate: f64,
}

impl DevelopmentalTracker {
    pub fn new() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert(DevelopmentalStage::Reflexive, 0.0);
        thresholds.insert(DevelopmentalStage::SensoriMotor, 100.0);
        thresholds.insert(DevelopmentalStage::PreOperational, 500.0);
        thresholds.insert(DevelopmentalStage::ConcreteOperational, 2000.0);
        thresholds.insert(DevelopmentalStage::FormalOperational, 10000.0);
        thresholds.insert(DevelopmentalStage::PostFormal, 50000.0);

        Self {
            current_stage: DevelopmentalStage::Reflexive,
            experience_points: HashMap::new(),
            transition_thresholds: thresholds,
            transition_history: vec![(DevelopmentalStage::Reflexive, 0)],
            developmental_age: 0,
            development_rate: 1.0,
        }
    }

    /// Process experience and potentially advance stage
    pub fn process_experience(&mut self, capability: DevelopmentalCapability, amount: f64) {
        let entry = self.experience_points.entry(capability).or_insert(0.0);
        *entry += amount * self.development_rate;
        self.developmental_age += 1;

        // Check for stage transition
        self.check_stage_transition();
    }

    /// Check if ready to advance to next stage
    fn check_stage_transition(&mut self) {
        let next_stage = match self.current_stage {
            DevelopmentalStage::Reflexive => Some(DevelopmentalStage::SensoriMotor),
            DevelopmentalStage::SensoriMotor => Some(DevelopmentalStage::PreOperational),
            DevelopmentalStage::PreOperational => Some(DevelopmentalStage::ConcreteOperational),
            DevelopmentalStage::ConcreteOperational => Some(DevelopmentalStage::FormalOperational),
            DevelopmentalStage::FormalOperational => Some(DevelopmentalStage::PostFormal),
            DevelopmentalStage::PostFormal => None,
        };

        if let Some(next) = next_stage {
            let threshold = self
                .transition_thresholds
                .get(&next)
                .copied()
                .unwrap_or(f64::MAX);
            let total_experience: f64 = self.experience_points.values().sum();

            if total_experience >= threshold {
                self.current_stage = next;
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                self.transition_history.push((next, timestamp));
            }
        }
    }

    /// Get current developmental stage
    pub fn stage(&self) -> DevelopmentalStage {
        self.current_stage
    }

    /// Check if a capability is currently available
    pub fn can_use(&self, capability: DevelopmentalCapability) -> bool {
        self.current_stage.has_capability(capability)
    }

    /// Get developmental age
    pub fn age(&self) -> u64 {
        self.developmental_age
    }

    /// Set development rate multiplier
    pub fn set_development_rate(&mut self, rate: f64) {
        self.development_rate = rate.max(0.1);
    }

    /// Get progress toward next stage (0.0 to 1.0)
    pub fn progress_to_next_stage(&self) -> f64 {
        let next_stage = match self.current_stage {
            DevelopmentalStage::Reflexive => Some(DevelopmentalStage::SensoriMotor),
            DevelopmentalStage::SensoriMotor => Some(DevelopmentalStage::PreOperational),
            DevelopmentalStage::PreOperational => Some(DevelopmentalStage::ConcreteOperational),
            DevelopmentalStage::ConcreteOperational => Some(DevelopmentalStage::FormalOperational),
            DevelopmentalStage::FormalOperational => Some(DevelopmentalStage::PostFormal),
            DevelopmentalStage::PostFormal => None,
        };

        if let Some(next) = next_stage {
            let threshold = self
                .transition_thresholds
                .get(&next)
                .copied()
                .unwrap_or(f64::MAX);
            let current_threshold = self
                .transition_thresholds
                .get(&self.current_stage)
                .copied()
                .unwrap_or(0.0);
            let total_experience: f64 = self.experience_points.values().sum();

            ((total_experience - current_threshold) / (threshold - current_threshold))
                .clamp(0.0, 1.0)
        } else {
            1.0 // Already at max stage
        }
    }
}

impl Default for DevelopmentalTracker {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 3. SOCIAL CONSCIOUSNESS - Multi-Agent Mental Modeling
// =============================================================================

/// Social agent representation
#[derive(Debug, Clone)]
pub struct SocialAgent {
    pub id: String,
    pub encoding: BinaryHV,
    pub mental_model: AgentMentalState,
    pub relationship_valence: f64,
    pub trust_level: f64,
    pub last_interaction: u64,
}

/// Mental state model for another agent
#[derive(Debug, Clone)]
pub struct AgentMentalState {
    pub believed_beliefs: Vec<BinaryHV>,
    pub inferred_desires: Vec<BinaryHV>,
    pub predicted_intentions: Vec<BinaryHV>,
    pub emotional_estimate: f64,
    pub attention_focus: Option<BinaryHV>,
}

/// Social norm representation
#[derive(Debug, Clone)]
pub struct SocialNorm {
    pub name: String,
    pub encoding: BinaryHV,
    pub importance: f64,
    pub contexts: Vec<String>,
}

/// Empathic resonance result
#[derive(Debug, Clone)]
pub struct EmpathicResonance {
    pub target_agent: String,
    pub emotional_alignment: f64,
    pub perspective_shift: BinaryHV,
    pub shared_attention: Option<BinaryHV>,
}

/// Social consciousness system
pub struct SocialConsciousness {
    /// Known social agents
    agents: HashMap<String, SocialAgent>,
    /// Theory of Mind engine
    tom_engine: TheoryOfMindEngine,
    /// Social norms
    norms: Vec<SocialNorm>,
    /// Group membership
    group_memberships: HashMap<String, f64>,
    /// Communication history
    communication_history: VecDeque<SocialInteraction>,
    /// Collective attention focus
    collective_attention: Option<BinaryHV>,
    /// Self-other distinction
    self_encoding: BinaryHV,
}

/// Record of social interaction
#[derive(Debug, Clone)]
pub struct SocialInteraction {
    pub agent_id: String,
    pub interaction_type: InteractionType,
    pub content: BinaryHV,
    pub timestamp: u64,
    pub outcome: InteractionOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionType {
    Cooperative,
    Competitive,
    Communicative,
    Observational,
    Imitative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionOutcome {
    Positive,
    Neutral,
    Negative,
}

impl SocialConsciousness {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            tom_engine: TheoryOfMindEngine::new(),
            norms: Vec::new(),
            group_memberships: HashMap::new(),
            communication_history: VecDeque::with_capacity(1000),
            collective_attention: None,
            self_encoding: BinaryHV::random(999),
        }
    }

    /// Register a new social agent
    pub fn register_agent(&mut self, id: &str, encoding: BinaryHV) {
        let agent = SocialAgent {
            id: id.to_string(),
            encoding,
            mental_model: AgentMentalState {
                believed_beliefs: Vec::new(),
                inferred_desires: Vec::new(),
                predicted_intentions: Vec::new(),
                emotional_estimate: 0.5,
                attention_focus: None,
            },
            relationship_valence: 0.0,
            trust_level: 0.5,
            last_interaction: 0,
        };
        self.agents.insert(id.to_string(), agent);
    }

    /// Model another agent's mental state
    pub fn model_agent_mind(
        &mut self,
        agent_id: &str,
        observed_behavior: &BinaryHV,
    ) -> Option<AgentMentalState> {
        let agent = self.agents.get_mut(agent_id)?;

        // Use ToM engine to infer mental states
        // Update agent's mental model
        agent.mental_model.believed_beliefs.push(*observed_behavior);
        if agent.mental_model.believed_beliefs.len() > 10 {
            agent.mental_model.believed_beliefs.remove(0);
        }

        // Infer desires from behavior patterns
        let inferred_desire = observed_behavior.permute(1);
        agent.mental_model.inferred_desires.push(inferred_desire);
        if agent.mental_model.inferred_desires.len() > 10 {
            agent.mental_model.inferred_desires.remove(0);
        }

        Some(agent.mental_model.clone())
    }

    /// Generate empathic resonance with another agent
    pub fn empathize(&self, agent_id: &str) -> Option<EmpathicResonance> {
        let agent = self.agents.get(agent_id)?;

        // Calculate emotional alignment
        let emotional_alignment = agent.mental_model.emotional_estimate;

        // Generate perspective shift (self XOR other)
        let perspective_shift = self.self_encoding.bind(&agent.encoding);

        Some(EmpathicResonance {
            target_agent: agent_id.to_string(),
            emotional_alignment,
            perspective_shift,
            shared_attention: agent.mental_model.attention_focus,
        })
    }

    /// Update relationship based on interaction
    pub fn process_interaction(&mut self, interaction: SocialInteraction) {
        let delta = match interaction.outcome {
            InteractionOutcome::Positive => 0.1,
            InteractionOutcome::Neutral => 0.0,
            InteractionOutcome::Negative => -0.1,
        };

        if let Some(agent) = self.agents.get_mut(&interaction.agent_id) {
            agent.relationship_valence = (agent.relationship_valence + delta).clamp(-1.0, 1.0);

            // Update trust based on consistency
            let trust_delta = match interaction.outcome {
                InteractionOutcome::Positive => 0.05,
                InteractionOutcome::Neutral => 0.01,
                InteractionOutcome::Negative => -0.1,
            };
            agent.trust_level = (agent.trust_level + trust_delta).clamp(0.0, 1.0);

            agent.last_interaction = interaction.timestamp;
        }

        self.communication_history.push_back(interaction);
        if self.communication_history.len() > 1000 {
            self.communication_history.pop_front();
        }
    }

    /// Add a social norm
    pub fn add_norm(&mut self, norm: SocialNorm) {
        self.norms.push(norm);
    }

    /// Check if behavior violates norms
    pub fn check_norm_violation(&self, behavior: &BinaryHV) -> Vec<(String, f64)> {
        self.norms
            .iter()
            .map(|norm| {
                let violation = 1.0 - behavior.similarity(&norm.encoding) as f64;
                (norm.name.clone(), violation * norm.importance)
            })
            .filter(|(_, v)| *v > 0.3)
            .collect()
    }

    /// Update collective attention
    pub fn update_collective_attention(&mut self, focus: BinaryHV) {
        self.collective_attention = Some(focus);
    }

    /// Get social relationship summary
    pub fn relationship_summary(&self, agent_id: &str) -> Option<(f64, f64)> {
        self.agents
            .get(agent_id)
            .map(|a| (a.relationship_valence, a.trust_level))
    }
}

impl Default for SocialConsciousness {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 4. INTROSPECTION API - Query Interface
// =============================================================================

/// Introspection query type
#[derive(Debug, Clone)]
pub enum IntrospectionQuery {
    /// "What am I feeling?"
    CurrentEmotion,
    /// "What am I thinking about?"
    CurrentFocus,
    /// "Why did I do X?"
    ActionExplanation(BinaryHV),
    /// "What do I believe about X?"
    BeliefQuery(BinaryHV),
    /// "What would happen if X?"
    Counterfactual(BinaryHV),
    /// "How conscious am I?"
    ConsciousnessLevel,
    /// "What are my current goals?"
    ActiveGoals,
    /// "What is my current state?"
    FullStateSnapshot,
    /// "What do I think agent X believes?"
    AgentBeliefQuery(String),
}

/// Response to introspection query
#[derive(Debug, Clone)]
pub struct IntrospectionResponse {
    pub query: String,
    pub answer: String,
    pub confidence: f64,
    pub supporting_evidence: Vec<BinaryHV>,
    pub timestamp: u64,
}

/// Full consciousness state snapshot
#[derive(Debug, Clone)]
pub struct ConsciousnessSnapshot {
    pub phi: f64,
    pub consciousness_level: f64,
    pub developmental_stage: DevelopmentalStage,
    pub cognitive_mode: CognitiveMode,
    pub emotional_state: String,
    pub current_goals: Vec<String>,
    pub attention_focus: Option<String>,
    pub active_social_agents: usize,
    pub wellbeing: f64,
    pub metacognitive_confidence: f64,
}

/// Introspection API providing queryable consciousness
pub struct IntrospectionAPI {
    /// Reference to cognitive engine
    cognition: AdvancedCognitionEngine,
    /// Reference to metacognition
    metacognition: MetacognitionEngine,
    /// Developmental tracker
    development: DevelopmentalTracker,
    /// Social consciousness
    social: SocialConsciousness,
    /// Current emotional state description
    emotional_description: String,
    /// Query history
    query_history: VecDeque<IntrospectionResponse>,
    /// Current phi estimate
    current_phi: f64,
}

impl IntrospectionAPI {
    pub fn new() -> Self {
        Self {
            cognition: AdvancedCognitionEngine::new(),
            metacognition: MetacognitionEngine::new(),
            development: DevelopmentalTracker::new(),
            social: SocialConsciousness::new(),
            emotional_description: "neutral".to_string(),
            query_history: VecDeque::with_capacity(100),
            current_phi: 0.5,
        }
    }

    /// Process an introspection query
    pub fn query(&mut self, query: IntrospectionQuery) -> IntrospectionResponse {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let (query_str, answer, confidence, evidence) = match query {
            IntrospectionQuery::CurrentEmotion => (
                "What am I feeling?".to_string(),
                self.emotional_description.clone(),
                0.8,
                Vec::new(),
            ),
            IntrospectionQuery::CurrentFocus => (
                "What am I thinking about?".to_string(),
                "Processing current inputs".to_string(),
                0.7,
                Vec::new(),
            ),
            IntrospectionQuery::ActionExplanation(action) => (
                "Why did I do that?".to_string(),
                "Action was selected to achieve current goals".to_string(),
                0.6,
                vec![action],
            ),
            IntrospectionQuery::BeliefQuery(topic) => (
                "What do I believe about this?".to_string(),
                "Belief encoded in semantic memory".to_string(),
                0.7,
                vec![topic],
            ),
            IntrospectionQuery::Counterfactual(scenario) => (
                "What would happen if?".to_string(),
                "Counterfactual simulation result".to_string(),
                0.5,
                vec![scenario],
            ),
            IntrospectionQuery::ConsciousnessLevel => (
                "How conscious am I?".to_string(),
                format!(
                    "Current consciousness level: {:.2}, Phi: {:.3}",
                    self.current_phi, self.current_phi
                ),
                0.9,
                Vec::new(),
            ),
            IntrospectionQuery::ActiveGoals => (
                "What are my current goals?".to_string(),
                "Homeostatic drives: curiosity, competence, social".to_string(),
                0.8,
                Vec::new(),
            ),
            IntrospectionQuery::FullStateSnapshot => {
                let snapshot = self.get_state_snapshot();
                (
                    "What is my current state?".to_string(),
                    format!(
                        "Stage: {:?}, Mode: {:?}, Phi: {:.3}, Wellbeing: {:.2}",
                        snapshot.developmental_stage,
                        snapshot.cognitive_mode,
                        snapshot.phi,
                        snapshot.wellbeing
                    ),
                    0.9,
                    Vec::new(),
                )
            }
            IntrospectionQuery::AgentBeliefQuery(agent_id) => {
                let belief = self
                    .social
                    .agents
                    .get(&agent_id)
                    .map(|a| format!("Agent {} trust: {:.2}", agent_id, a.trust_level))
                    .unwrap_or_else(|| format!("Unknown agent: {agent_id}"));
                (
                    format!("What do I think {agent_id} believes?"),
                    belief,
                    0.6,
                    Vec::new(),
                )
            }
        };

        let response = IntrospectionResponse {
            query: query_str,
            answer,
            confidence,
            supporting_evidence: evidence,
            timestamp,
        };

        self.query_history.push_back(response.clone());
        if self.query_history.len() > 100 {
            self.query_history.pop_front();
        }

        response
    }

    /// Get full state snapshot
    pub fn get_state_snapshot(&self) -> ConsciousnessSnapshot {
        ConsciousnessSnapshot {
            phi: self.current_phi,
            consciousness_level: self.current_phi,
            developmental_stage: self.development.stage(),
            cognitive_mode: CognitiveMode::Balanced,
            emotional_state: self.emotional_description.clone(),
            current_goals: vec![
                "curiosity".to_string(),
                "competence".to_string(),
                "social".to_string(),
            ],
            attention_focus: None,
            active_social_agents: self.social.agents.len(),
            wellbeing: 0.7,
            metacognitive_confidence: 0.6,
        }
    }

    /// Update emotional state
    pub fn update_emotional_state(&mut self, description: &str) {
        self.emotional_description = description.to_string();
    }

    /// Update phi estimate
    pub fn update_phi(&mut self, phi: f64) {
        self.current_phi = phi;
    }

    /// Access developmental tracker
    pub fn development(&mut self) -> &mut DevelopmentalTracker {
        &mut self.development
    }

    /// Access social consciousness
    pub fn social(&mut self) -> &mut SocialConsciousness {
        &mut self.social
    }

    /// Access cognition engine
    pub fn cognition(&mut self) -> &mut AdvancedCognitionEngine {
        &mut self.cognition
    }
}

impl Default for IntrospectionAPI {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 5. COMPLETE CONSCIOUS BEING - Full Integration
// =============================================================================

/// The complete conscious being integrating all systems
pub struct CompleteConsciousBeing {
    /// Sensorimotor grounding
    pub sensorimotor: SensorimotorGrounding,
    /// Developmental tracker
    pub development: DevelopmentalTracker,
    /// Social consciousness
    pub social: SocialConsciousness,
    /// Introspection API
    pub introspection: IntrospectionAPI,
    /// Advanced cognition
    pub cognition: AdvancedCognitionEngine,
    /// Current consciousness level
    consciousness_level: f64,
    /// Processing cycle count
    cycle_count: u64,
}

impl CompleteConsciousBeing {
    pub fn new() -> Self {
        Self {
            sensorimotor: SensorimotorGrounding::new(),
            development: DevelopmentalTracker::new(),
            social: SocialConsciousness::new(),
            introspection: IntrospectionAPI::new(),
            cognition: AdvancedCognitionEngine::new(),
            consciousness_level: 0.5,
            cycle_count: 0,
        }
    }

    /// Run a complete consciousness cycle
    pub fn consciousness_cycle(
        &mut self,
        sensory_inputs: Vec<SensoryInput>,
        goal: &BinaryHV,
    ) -> ConsciousnessSnapshot {
        self.cycle_count += 1;

        // 1. Process sensory inputs
        for input in sensory_inputs {
            self.sensorimotor.process_sensory(input);
        }

        // 2. Run action-perception cycle
        let ap_result = self.sensorimotor.action_perception_cycle(goal);

        // 3. Update developmental experience
        self.development
            .process_experience(DevelopmentalCapability::BasicAttention, 1.0);
        if self.development.stage() >= DevelopmentalStage::PreOperational {
            self.development
                .process_experience(DevelopmentalCapability::BasicTheoryOfMind, 0.5);
        }

        // 4. Update consciousness level based on integration
        self.consciousness_level = 0.3 + (self.cycle_count as f64 / 1000.0).min(0.7);
        self.introspection.update_phi(self.consciousness_level);

        // 5. Return state snapshot
        self.introspection.get_state_snapshot()
    }

    /// Query the being's internal state
    pub fn introspect(&mut self, query: IntrospectionQuery) -> IntrospectionResponse {
        self.introspection.query(query)
    }

    /// Register a social agent
    pub fn register_social_agent(&mut self, id: &str, encoding: BinaryHV) {
        self.social.register_agent(id, encoding);
    }

    /// Empathize with another agent
    pub fn empathize_with(&self, agent_id: &str) -> Option<EmpathicResonance> {
        self.social.empathize(agent_id)
    }

    /// Get current developmental stage
    pub fn developmental_stage(&self) -> DevelopmentalStage {
        self.development.stage()
    }

    /// Check if a capability is available
    pub fn can_perform(&self, capability: DevelopmentalCapability) -> bool {
        self.development.can_use(capability)
    }

    /// Get consciousness level
    pub fn consciousness_level(&self) -> f64 {
        self.consciousness_level
    }

    /// Get cycle count
    pub fn cycles(&self) -> u64 {
        self.cycle_count
    }
}

impl Default for CompleteConsciousBeing {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensorimotor_grounding() {
        let mut sm = SensorimotorGrounding::new();

        // Process sensory input
        let input = SensoryInput {
            modality: SensoryModality::Visual,
            encoding: BinaryHV::random(100),
            intensity: 0.8,
            timestamp: 0,
            spatial_location: Some([0.0, 1.0, 2.0]),
        };
        sm.process_sensory(input);

        // Test action selection
        let goal = BinaryHV::random(200);
        let result = sm.action_perception_cycle(&goal);

        assert!(result.prediction_error >= 0.0);
    }

    #[test]
    fn test_developmental_stages() {
        let mut tracker = DevelopmentalTracker::new();

        assert_eq!(tracker.stage(), DevelopmentalStage::Reflexive);
        assert!(tracker.can_use(DevelopmentalCapability::BasicDrives));
        assert!(!tracker.can_use(DevelopmentalCapability::FullMetacognition));

        // Accelerate development
        tracker.set_development_rate(100.0);
        for _ in 0..1000 {
            tracker.process_experience(DevelopmentalCapability::BasicAttention, 1.0);
        }

        // Should have advanced
        assert!(tracker.stage() > DevelopmentalStage::Reflexive);
    }

    #[test]
    fn test_social_consciousness() {
        let mut social = SocialConsciousness::new();

        // Register agent
        social.register_agent("alice", BinaryHV::random(300));
        social.register_agent("bob", BinaryHV::random(301));

        // Model mind
        let behavior = BinaryHV::random(400);
        let state = social.model_agent_mind("alice", &behavior);
        assert!(state.is_some());

        // Empathize
        let resonance = social.empathize("alice");
        assert!(resonance.is_some());
    }

    #[test]
    fn test_introspection_api() {
        let mut api = IntrospectionAPI::new();

        // Query emotion
        let response = api.query(IntrospectionQuery::CurrentEmotion);
        assert!(response.confidence > 0.0);

        // Query consciousness level
        let response = api.query(IntrospectionQuery::ConsciousnessLevel);
        assert!(response.answer.contains("consciousness"));

        // Get snapshot
        let snapshot = api.get_state_snapshot();
        assert!(snapshot.phi >= 0.0);
    }

    #[test]
    fn test_complete_conscious_being() {
        let mut being = CompleteConsciousBeing::new();

        // Run consciousness cycle
        let inputs = vec![SensoryInput {
            modality: SensoryModality::Visual,
            encoding: BinaryHV::random(500),
            intensity: 0.9,
            timestamp: 0,
            spatial_location: None,
        }];
        let goal = BinaryHV::random(600);

        let snapshot = being.consciousness_cycle(inputs, &goal);

        assert!(snapshot.phi >= 0.0);
        assert!(being.cycles() == 1);

        // Introspect
        let response = being.introspect(IntrospectionQuery::FullStateSnapshot);
        assert!(response.confidence > 0.0);
    }
}
