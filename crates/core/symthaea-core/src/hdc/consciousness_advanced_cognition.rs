// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Advanced Cognition Integration Module
//!
//! This module integrates and enhances 6 advanced cognitive capabilities:
//!
//! 1. **Embodied Cognition & Motor Imagery** - Enhanced motor simulation
//! 2. **Theory of Mind Engine** - Mental state attribution for other agents
//! 3. **Imagination & Creative Synthesis** - Novel concept generation
//! 4. **Predictive Processing Framework** - Hierarchical prediction integration
//! 5. **Differentiated Memory Systems** - Semantic, procedural, prospective memory
//! 6. **Homeostatic Drive System** - Curiosity, competence, social drives
//!
//! ## Integration Philosophy
//!
//! Rather than duplicating existing systems, this module:
//! - Extends existing infrastructure with new capabilities
//! - Provides unified access through `AdvancedCognitionEngine`
//! - Creates cross-system interactions (e.g., drives → attention → memory)

use super::binary_hv::BinaryHV;
use super::emotional_depth::EmotionalBlend;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// =============================================================================
// 1. MOTOR IMAGERY & EMBODIED SIMULATION
// =============================================================================

/// Motor imagery system for action simulation without execution
#[derive(Debug, Clone)]
pub struct MotorImagerySystem {
    /// Current imagined action sequence
    imagined_sequence: VecDeque<MotorCommand>,
    /// Body schema state for simulation
    simulated_body_state: BodyState,
    /// Motor programs (learned action patterns)
    motor_programs: HashMap<String, MotorProgram>,
    /// Imagery vividness (0.0-1.0, affects simulation fidelity)
    imagery_vividness: f64,
    /// Motor prediction errors from imagined vs actual
    prediction_errors: VecDeque<MotorPredictionError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorCommand {
    /// Target effector (hand, foot, eye, etc.)
    pub effector: String,
    /// Action type
    pub action: MotorAction,
    /// Intended force/speed (0.0-1.0)
    pub intensity: f64,
    /// Duration in simulated ms
    pub duration_ms: u32,
    /// Expected outcome HV
    pub expected_outcome: BinaryHV,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum MotorAction {
    Reach,
    Grasp,
    Release,
    Push,
    Pull,
    Point,
    Wave,
    Walk,
    Turn,
    Speak,
    Look,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyState {
    /// Position of body parts (simplified 3D coords)
    pub positions: HashMap<String, (f64, f64, f64)>,
    /// Current posture encoding
    pub posture_hv: BinaryHV,
    /// Energy level (0.0-1.0)
    pub energy: f64,
    /// Tension level (0.0-1.0)
    pub tension: f64,
}

#[derive(Debug, Clone)]
pub struct MotorProgram {
    /// Name of the program
    pub name: String,
    /// Sequence of motor commands
    pub commands: Vec<MotorCommand>,
    /// Practice count (affects automaticity)
    pub practice_count: u32,
    /// Success rate (0.0-1.0)
    pub success_rate: f64,
}

#[derive(Debug, Clone)]
pub struct MotorPredictionError {
    /// What was predicted
    pub predicted: BinaryHV,
    /// What happened
    pub actual: BinaryHV,
    /// Error magnitude
    pub magnitude: f64,
    /// Which effector
    pub effector: String,
}

impl MotorImagerySystem {
    pub fn new() -> Self {
        let mut positions = HashMap::new();
        positions.insert("head".to_string(), (0.0, 1.7, 0.0));
        positions.insert("left_hand".to_string(), (-0.3, 1.0, 0.2));
        positions.insert("right_hand".to_string(), (0.3, 1.0, 0.2));

        Self {
            imagined_sequence: VecDeque::new(),
            simulated_body_state: BodyState {
                positions,
                posture_hv: BinaryHV::random(42),
                energy: 0.8,
                tension: 0.2,
            },
            motor_programs: HashMap::new(),
            imagery_vividness: 0.7,
            prediction_errors: VecDeque::with_capacity(50),
        }
    }

    /// Imagine an action without executing it
    pub fn imagine_action(&mut self, command: MotorCommand) -> MotorImageryResult {
        // Add to imagined sequence
        self.imagined_sequence.push_back(command.clone());

        // Simulate the action mentally
        let simulated_outcome = self.simulate_motor_command(&command);

        // Calculate similarity to expected outcome
        let match_quality = command.expected_outcome.similarity(&simulated_outcome);

        // Update body state prediction
        self.update_simulated_body_state(&command);

        MotorImageryResult {
            simulated_outcome,
            vividness: self.imagery_vividness,
            match_quality: match_quality as f64,
            energy_cost_estimate: self.estimate_energy_cost(&command),
        }
    }

    /// Simulate a motor command and predict outcome
    fn simulate_motor_command(&self, command: &MotorCommand) -> BinaryHV {
        // Use motor program if available, otherwise generate novel simulation
        if let Some(program) = self.motor_programs.get(&command.effector) {
            // Familiar action - use learned pattern
            program
                .commands
                .first()
                .map(|c| c.expected_outcome)
                .unwrap_or_else(|| BinaryHV::random(command.duration_ms as u64))
        } else {
            // Novel action - combine effector, action, intensity
            let effector_hv = BinaryHV::random(hash_string(&command.effector));
            let action_hv = BinaryHV::random(command.action as u64);
            let intensity_hv = BinaryHV::random((command.intensity * 1000.0) as u64);
            BinaryHV::bundle(&[effector_hv, action_hv, intensity_hv])
        }
    }

    fn update_simulated_body_state(&mut self, command: &MotorCommand) {
        // Simulate energy expenditure
        self.simulated_body_state.energy -= command.intensity * 0.01;
        self.simulated_body_state.energy = self.simulated_body_state.energy.max(0.0);

        // Update position estimate for effector
        if let Some(pos) = self
            .simulated_body_state
            .positions
            .get_mut(&command.effector)
        {
            match command.action {
                MotorAction::Reach => pos.2 += 0.1 * command.intensity,
                MotorAction::Point => pos.2 += 0.2 * command.intensity,
                _ => {}
            }
        }
    }

    fn estimate_energy_cost(&self, command: &MotorCommand) -> f64 {
        command.intensity * (command.duration_ms as f64 / 1000.0) * 0.1
    }

    /// Learn a motor program through practice
    pub fn learn_motor_program(&mut self, name: &str, commands: Vec<MotorCommand>) {
        self.motor_programs.insert(
            name.to_string(),
            MotorProgram {
                name: name.to_string(),
                commands,
                practice_count: 1,
                success_rate: 0.5,
            },
        );
    }

    /// Practice a motor program (increases automaticity)
    pub fn practice_program(&mut self, name: &str, success: bool) {
        if let Some(program) = self.motor_programs.get_mut(name) {
            program.practice_count += 1;
            // Update success rate with exponential moving average
            let alpha = 0.1;
            program.success_rate =
                program.success_rate * (1.0 - alpha) + if success { 1.0 } else { 0.0 } * alpha;
        }
    }

    /// Get body state encoding for cross-modal binding
    pub fn body_state_hv(&self) -> BinaryHV {
        self.simulated_body_state.posture_hv
    }

    /// Set imagery vividness
    pub fn set_vividness(&mut self, vividness: f64) {
        self.imagery_vividness = vividness.clamp(0.0, 1.0);
    }
}

#[derive(Debug, Clone)]
pub struct MotorImageryResult {
    pub simulated_outcome: BinaryHV,
    pub vividness: f64,
    pub match_quality: f64,
    pub energy_cost_estimate: f64,
}

impl Default for MotorImagerySystem {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 2. THEORY OF MIND ENGINE
// =============================================================================

/// Theory of Mind engine for modeling other agents' mental states
#[derive(Debug, Clone)]
pub struct TheoryOfMindEngine {
    /// Mental models of known agents
    agent_models: HashMap<String, AgentMentalModel>,
    /// Perspective-taking depth (1 = first-order, 2 = "I think you think", etc.)
    max_recursion_depth: u32,
    /// Cache of recent mental state inferences
    inference_cache: VecDeque<MentalStateInference>,
    /// Shared knowledge base (common ground)
    shared_knowledge: Vec<BinaryHV>,
}

#[derive(Debug, Clone)]
pub struct AgentMentalModel {
    /// Agent identifier
    pub agent_id: String,
    /// Inferred beliefs (what do they think is true?)
    pub beliefs: Vec<Belief>,
    /// Inferred desires (what do they want?)
    pub desires: Vec<Desire>,
    /// Inferred intentions (what will they do?)
    pub intentions: Vec<Intention>,
    /// Emotional state estimate
    pub emotional_state: CoreAffectEstimate,
    /// Confidence in this model (0.0-1.0)
    pub confidence: f64,
    /// Last updated timestamp
    pub last_updated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Belief {
    /// Content of belief
    pub content: BinaryHV,
    /// Description for introspection
    pub description: String,
    /// Strength of belief (0.0-1.0)
    pub strength: f64,
    /// Is this a false belief? (for false-belief reasoning)
    pub is_false_belief: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Desire {
    /// Content of desire
    pub content: BinaryHV,
    /// Description
    pub description: String,
    /// Urgency (0.0-1.0)
    pub urgency: f64,
    /// Is this fulfilled?
    pub fulfilled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intention {
    /// Intended action
    pub action: BinaryHV,
    /// Target of action
    pub target: Option<BinaryHV>,
    /// Description
    pub description: String,
    /// Commitment strength (0.0-1.0)
    pub commitment: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CoreAffectEstimate {
    pub valence: f64,
    pub arousal: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct MentalStateInference {
    /// Which agent
    pub agent_id: String,
    /// What was inferred
    pub inference_type: InferenceType,
    /// Evidence used
    pub evidence: Vec<BinaryHV>,
    /// Confidence
    pub confidence: f64,
    /// Timestamp
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InferenceType {
    BeliefAttribution,
    DesireAttribution,
    IntentionAttribution,
    EmotionAttribution,
    FalseBeliefDetection,
    DeceptionDetection,
}

impl TheoryOfMindEngine {
    pub fn new() -> Self {
        Self {
            agent_models: HashMap::new(),
            max_recursion_depth: 2,
            inference_cache: VecDeque::with_capacity(100),
            shared_knowledge: Vec::new(),
        }
    }

    /// Register a new agent to model
    pub fn register_agent(&mut self, agent_id: &str) {
        self.agent_models.insert(
            agent_id.to_string(),
            AgentMentalModel {
                agent_id: agent_id.to_string(),
                beliefs: Vec::new(),
                desires: Vec::new(),
                intentions: Vec::new(),
                emotional_state: CoreAffectEstimate {
                    valence: 0.0,
                    arousal: 0.5,
                    confidence: 0.3,
                },
                confidence: 0.3,
                last_updated: current_timestamp(),
            },
        );
    }

    /// Attribute a belief to an agent
    pub fn attribute_belief(&mut self, agent_id: &str, belief: Belief) {
        if let Some(model) = self.agent_models.get_mut(agent_id) {
            model.beliefs.push(belief.clone());
            model.last_updated = current_timestamp();

            self.inference_cache.push_back(MentalStateInference {
                agent_id: agent_id.to_string(),
                inference_type: InferenceType::BeliefAttribution,
                evidence: vec![belief.content],
                confidence: belief.strength,
                timestamp: current_timestamp(),
            });
        }
    }

    /// Attribute a desire to an agent
    pub fn attribute_desire(&mut self, agent_id: &str, desire: Desire) {
        if let Some(model) = self.agent_models.get_mut(agent_id) {
            model.desires.push(desire.clone());
            model.last_updated = current_timestamp();
        }
    }

    /// Attribute an intention to an agent
    pub fn attribute_intention(&mut self, agent_id: &str, intention: Intention) {
        if let Some(model) = self.agent_models.get_mut(agent_id) {
            model.intentions.push(intention.clone());
            model.last_updated = current_timestamp();
        }
    }

    /// Infer emotional state from behavioral cues
    pub fn infer_emotion(
        &mut self,
        agent_id: &str,
        behavioral_cues: &[BinaryHV],
    ) -> CoreAffectEstimate {
        // Simple inference based on cue patterns
        let valence = if !behavioral_cues.is_empty() {
            // Use first bit pattern as rough valence estimate
            let first = &behavioral_cues[0];
            (first.popcount() as f64 / 8192.0) - 0.5
        } else {
            0.0
        };

        let estimate = CoreAffectEstimate {
            valence: valence.clamp(-1.0, 1.0),
            arousal: 0.5 + behavioral_cues.len() as f64 * 0.1,
            confidence: 0.3 + behavioral_cues.len() as f64 * 0.1,
        };

        if let Some(model) = self.agent_models.get_mut(agent_id) {
            model.emotional_state = estimate;
            model.last_updated = current_timestamp();
        }

        estimate
    }

    /// Perform perspective-taking: simulate agent's viewpoint
    pub fn take_perspective(&self, agent_id: &str, situation: &BinaryHV) -> PerspectiveResult {
        let model = match self.agent_models.get(agent_id) {
            Some(m) => m,
            None => return PerspectiveResult::unknown(),
        };

        // Combine agent's beliefs with situation
        let mut perspective_hvs: Vec<BinaryHV> = model.beliefs.iter().map(|b| b.content).collect();
        perspective_hvs.push(*situation);

        let perspective = if perspective_hvs.is_empty() {
            *situation
        } else {
            BinaryHV::bundle(&perspective_hvs)
        };

        // What would they perceive?
        let perceived = perspective.bind(&self.rotate_for_perspective(situation));

        // What would they conclude?
        let conclusion = if !model.desires.is_empty() {
            // Bias toward fulfilling desires
            let desire_hv = &model.desires[0].content;
            perceived.bind(desire_hv)
        } else {
            perceived
        };

        PerspectiveResult {
            perceived_situation: perceived,
            likely_conclusion: conclusion,
            emotional_response: model.emotional_state,
            confidence: model.confidence,
        }
    }

    /// Rotate HV to simulate different perspective
    fn rotate_for_perspective(&self, hv: &BinaryHV) -> BinaryHV {
        hv.permute(42) // Perspective shift through permutation
    }

    /// Detect if agent has a false belief
    pub fn detect_false_belief(
        &self,
        agent_id: &str,
        reality: &BinaryHV,
    ) -> Option<FalseBeliefDetection> {
        let model = self.agent_models.get(agent_id)?;

        for belief in &model.beliefs {
            let similarity = belief.content.similarity(reality);
            if similarity < 0.3 && belief.strength > 0.5 {
                return Some(FalseBeliefDetection {
                    agent_id: agent_id.to_string(),
                    false_belief: belief.clone(),
                    reality: *reality,
                    divergence: 1.0 - similarity as f64,
                });
            }
        }

        None
    }

    /// Predict agent's next action based on BDI model
    pub fn predict_action(&self, agent_id: &str) -> Option<ActionPrediction> {
        let model = self.agent_models.get(agent_id)?;

        // Find strongest intention
        let strongest_intention = model
            .intentions
            .iter()
            .max_by(|a, b| a.commitment.total_cmp(&b.commitment))?;

        // Find most urgent unfulfilled desire
        let urgent_desire = model
            .desires
            .iter()
            .filter(|d| !d.fulfilled)
            .max_by(|a, b| a.urgency.total_cmp(&b.urgency));

        // Combine intention with desire
        let predicted_action = if let Some(desire) = urgent_desire {
            strongest_intention.action.bind(&desire.content)
        } else {
            strongest_intention.action
        };

        Some(ActionPrediction {
            predicted_action,
            based_on_intention: strongest_intention.description.clone(),
            confidence: strongest_intention.commitment * model.confidence,
        })
    }

    /// Add to shared knowledge (common ground)
    pub fn add_shared_knowledge(&mut self, knowledge: BinaryHV) {
        self.shared_knowledge.push(knowledge);
    }

    /// Get mental model for agent
    pub fn get_model(&self, agent_id: &str) -> Option<&AgentMentalModel> {
        self.agent_models.get(agent_id)
    }
}

#[derive(Debug, Clone)]
pub struct PerspectiveResult {
    pub perceived_situation: BinaryHV,
    pub likely_conclusion: BinaryHV,
    pub emotional_response: CoreAffectEstimate,
    pub confidence: f64,
}

impl PerspectiveResult {
    fn unknown() -> Self {
        Self {
            perceived_situation: BinaryHV::random(0),
            likely_conclusion: BinaryHV::random(0),
            emotional_response: CoreAffectEstimate {
                valence: 0.0,
                arousal: 0.5,
                confidence: 0.0,
            },
            confidence: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FalseBeliefDetection {
    pub agent_id: String,
    pub false_belief: Belief,
    pub reality: BinaryHV,
    pub divergence: f64,
}

#[derive(Debug, Clone)]
pub struct ActionPrediction {
    pub predicted_action: BinaryHV,
    pub based_on_intention: String,
    pub confidence: f64,
}

impl Default for TheoryOfMindEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 3. IMAGINATION & CREATIVE SYNTHESIS ENGINE
// =============================================================================

/// Imagination engine for mental simulation and creative synthesis
#[derive(Debug, Clone)]
pub struct ImaginationEngine {
    /// Current imagination workspace
    workspace: Vec<ImaginaryConstruct>,
    /// Constraint relaxation level (0.0 = realistic, 1.0 = fantastical)
    constraint_relaxation: f64,
    /// Analogical mapping store
    analogies: Vec<AnalogicalMapping>,
    /// Creative combinations history
    combinations_history: VecDeque<CreativeCombination>,
    /// Novelty threshold for accepting new ideas
    novelty_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct ImaginaryConstruct {
    /// The imagined content
    pub content: BinaryHV,
    /// Description
    pub description: String,
    /// Source concepts (what was combined)
    pub sources: Vec<BinaryHV>,
    /// Novelty score (how different from known concepts)
    pub novelty: f64,
    /// Coherence score (how internally consistent)
    pub coherence: f64,
    /// Utility estimate (how useful could this be)
    pub utility: f64,
}

#[derive(Debug, Clone)]
pub struct AnalogicalMapping {
    /// Source domain
    pub source_domain: BinaryHV,
    pub source_label: String,
    /// Target domain
    pub target_domain: BinaryHV,
    pub target_label: String,
    /// Mapping strength (structural similarity)
    pub strength: f64,
    /// Inferences made via this analogy
    pub inferences: Vec<BinaryHV>,
}

#[derive(Debug, Clone)]
pub struct CreativeCombination {
    /// Input concepts
    pub inputs: Vec<BinaryHV>,
    /// Output creation
    pub output: BinaryHV,
    /// Combination method used
    pub method: CombinationMethod,
    /// Quality assessment
    pub quality: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CombinationMethod {
    /// Merge properties (HV bundle)
    PropertyMerge,
    /// Transfer structure (analogy)
    StructuralTransfer,
    /// Invert/negate properties
    Negation,
    /// Blend with randomness
    RandomBlend,
    /// Constraint relaxation
    ConstraintRelax,
    /// Counterfactual simulation
    Counterfactual,
}

impl ImaginationEngine {
    pub fn new() -> Self {
        Self {
            workspace: Vec::new(),
            constraint_relaxation: 0.3,
            analogies: Vec::new(),
            combinations_history: VecDeque::with_capacity(100),
            novelty_threshold: 0.4,
        }
    }

    /// Imagine something new by combining concepts
    pub fn imagine(
        &mut self,
        concepts: &[BinaryHV],
        method: CombinationMethod,
    ) -> ImaginaryConstruct {
        let combined = match method {
            CombinationMethod::PropertyMerge => BinaryHV::bundle(concepts),
            CombinationMethod::StructuralTransfer => {
                if concepts.len() >= 2 {
                    // Transfer structure from first to second
                    concepts[0].bind(&concepts[1])
                } else {
                    concepts
                        .first()
                        .cloned()
                        .unwrap_or_else(|| BinaryHV::random(42))
                }
            }
            CombinationMethod::Negation => {
                if let Some(first) = concepts.first() {
                    first.invert()
                } else {
                    BinaryHV::random(42)
                }
            }
            CombinationMethod::RandomBlend => {
                let base = BinaryHV::bundle(concepts);
                let noise = BinaryHV::random(current_timestamp());
                BinaryHV::bundle(&[base, noise])
            }
            CombinationMethod::ConstraintRelax => {
                let base = BinaryHV::bundle(concepts);
                base.add_noise(self.constraint_relaxation as f32, current_timestamp())
            }
            CombinationMethod::Counterfactual => {
                // Negate first, bind with rest
                if let Some(first) = concepts.first() {
                    let negated = first.invert();
                    if concepts.len() > 1 {
                        negated.bind(&BinaryHV::bundle(&concepts[1..]))
                    } else {
                        negated
                    }
                } else {
                    BinaryHV::random(42)
                }
            }
        };

        // Calculate novelty against existing concepts
        let novelty = if concepts.is_empty() {
            1.0
        } else {
            let avg_similarity: f32 = concepts.iter().map(|c| combined.similarity(c)).sum::<f32>()
                / concepts.len() as f32;
            1.0 - avg_similarity as f64
        };

        // Calculate coherence (how self-consistent)
        let coherence = 1.0 - (self.constraint_relaxation * 0.5);

        // Estimate utility (placeholder - would need domain knowledge)
        let utility = novelty * coherence;

        let construct = ImaginaryConstruct {
            content: combined,
            description: format!("Imagined via {method:?}"),
            sources: concepts.to_vec(),
            novelty,
            coherence,
            utility,
        };

        // Record combination
        self.combinations_history.push_back(CreativeCombination {
            inputs: concepts.to_vec(),
            output: construct.content,
            method,
            quality: utility,
        });

        while self.combinations_history.len() > 100 {
            self.combinations_history.pop_front();
        }

        // Add to workspace if novel enough
        if novelty >= self.novelty_threshold {
            self.workspace.push(construct.clone());
        }

        construct
    }

    /// Create an analogy between two domains
    pub fn create_analogy(
        &mut self,
        source: &BinaryHV,
        source_label: &str,
        target: &BinaryHV,
        target_label: &str,
    ) -> AnalogicalMapping {
        let strength = source.similarity(target) as f64;

        // Generate inferences by binding source patterns to target
        let inference = source.bind(target);

        let mapping = AnalogicalMapping {
            source_domain: *source,
            source_label: source_label.to_string(),
            target_domain: *target,
            target_label: target_label.to_string(),
            strength,
            inferences: vec![inference],
        };

        self.analogies.push(mapping.clone());
        mapping
    }

    /// Use analogy to infer about target domain
    pub fn analogical_inference(
        &self,
        source_property: &BinaryHV,
        analogy_index: usize,
    ) -> Option<BinaryHV> {
        let analogy = self.analogies.get(analogy_index)?;

        // Map source property to target domain
        let mapped = source_property.bind(&analogy.source_domain);
        let inference = mapped.bind(&analogy.target_domain);

        Some(inference)
    }

    /// Simulate a scenario mentally
    pub fn simulate_scenario(&mut self, initial_state: &BinaryHV, steps: u32) -> Vec<BinaryHV> {
        let mut states = vec![*initial_state];

        for i in 0..steps {
            let current = states
                .last()
                .expect("states initialized with initial_state and only grows");
            // Evolve state through imagination (add temporal drift)
            let next = current.permute(i as usize + 1);
            // Add some creativity
            let creative_next = self.imagine(&[next], CombinationMethod::ConstraintRelax);
            states.push(creative_next.content);
        }

        states
    }

    /// Generate novel concept by exploring semantic space
    pub fn explore_semantic_space(
        &mut self,
        seed: &BinaryHV,
        exploration_radius: f64,
    ) -> Vec<ImaginaryConstruct> {
        let mut explorations = Vec::new();

        // Generate variations through different methods
        for method in &[
            CombinationMethod::ConstraintRelax,
            CombinationMethod::Negation,
            CombinationMethod::RandomBlend,
        ] {
            let noise_level = exploration_radius as f32;
            let varied = seed.add_noise(noise_level, current_timestamp() + *method as u64);
            let construct = self.imagine(&[*seed, varied], *method);
            explorations.push(construct);
        }

        explorations
    }

    /// Set constraint relaxation level
    pub fn set_constraint_relaxation(&mut self, level: f64) {
        self.constraint_relaxation = level.clamp(0.0, 1.0);
    }

    /// Get workspace contents
    pub fn workspace(&self) -> &[ImaginaryConstruct] {
        &self.workspace
    }

    /// Clear imagination workspace
    pub fn clear_workspace(&mut self) {
        self.workspace.clear();
    }
}

impl Default for ImaginationEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 4. PREDICTIVE PROCESSING INTEGRATION
// =============================================================================

/// Enhanced predictive processing with active inference
#[derive(Debug, Clone)]
pub struct PredictiveProcessor {
    /// Hierarchical prediction layers
    layers: Vec<PredictionLayer>,
    /// Active inference goals
    goals: Vec<ActiveGoal>,
    /// Prediction error history
    error_history: VecDeque<WeightedError>,
    /// Precision regulation
    precision_modulator: PrecisionModulator,
}

#[derive(Debug, Clone)]
pub struct PredictionLayer {
    /// Layer level (0 = sensory)
    pub level: u32,
    /// Current belief state
    pub belief: BinaryHV,
    /// Prediction for level below
    pub downward_prediction: BinaryHV,
    /// Precision (confidence)
    pub precision: f64,
    /// Learning rate
    pub learning_rate: f64,
}

#[derive(Debug, Clone)]
pub struct ActiveGoal {
    /// Goal state
    pub target: BinaryHV,
    /// Priority
    pub priority: f64,
    /// Expected path to goal
    pub expected_path: Vec<BinaryHV>,
}

#[derive(Debug, Clone)]
pub struct WeightedError {
    /// Prediction
    pub prediction: BinaryHV,
    /// Actual
    pub actual: BinaryHV,
    /// Raw error magnitude
    pub magnitude: f64,
    /// Precision-weighted error
    pub weighted: f64,
    /// Layer source
    pub layer: u32,
}

#[derive(Debug, Clone)]
pub struct PrecisionModulator {
    /// Base precision
    pub base_precision: f64,
    /// Attention modulation
    pub attention_boost: f64,
    /// Arousal modulation
    pub arousal_effect: f64,
}

impl PredictiveProcessor {
    pub fn new(num_layers: u32) -> Self {
        let layers = (0..num_layers)
            .map(|i| PredictionLayer {
                level: i,
                belief: BinaryHV::random(i as u64),
                downward_prediction: BinaryHV::random(i as u64 + 1000),
                precision: 0.8 - (i as f64 * 0.1),
                learning_rate: 0.1,
            })
            .collect();

        Self {
            layers,
            goals: Vec::new(),
            error_history: VecDeque::with_capacity(100),
            precision_modulator: PrecisionModulator {
                base_precision: 0.7,
                attention_boost: 0.0,
                arousal_effect: 0.0,
            },
        }
    }

    /// Process input through hierarchy
    pub fn process(&mut self, sensory_input: &BinaryHV) -> PredictionResult {
        let mut current_signal = *sensory_input;
        let mut total_error = 0.0;
        let mut layer_errors = Vec::new();

        for layer in &mut self.layers {
            // Calculate prediction error
            let error_magnitude =
                1.0 - layer.downward_prediction.similarity(&current_signal) as f64;
            let effective_precision =
                layer.precision * self.precision_modulator.effective_precision();
            let weighted_error = error_magnitude * effective_precision;

            layer_errors.push(WeightedError {
                prediction: layer.downward_prediction,
                actual: current_signal,
                magnitude: error_magnitude,
                weighted: weighted_error,
                layer: layer.level,
            });

            total_error += weighted_error;

            // Update belief based on error
            if weighted_error > 0.1 {
                // Significant error - update belief
                let update = current_signal.bind(&layer.belief);
                layer.belief = BinaryHV::bundle(&[layer.belief, update]);
            }

            // Generate prediction for next layer
            layer.downward_prediction = layer.belief.permute(layer.level as usize);

            // Pass belief up to next layer
            current_signal = layer.belief;
        }

        // Record errors
        for error in &layer_errors {
            self.error_history.push_back(error.clone());
        }
        while self.error_history.len() > 100 {
            self.error_history.pop_front();
        }

        PredictionResult {
            integrated_belief: self
                .layers
                .last()
                .map(|l| l.belief)
                .unwrap_or_else(|| *sensory_input),
            total_prediction_error: total_error,
            layer_errors,
            surprise: total_error / self.layers.len() as f64,
        }
    }

    /// Set an active inference goal
    pub fn set_goal(&mut self, target: BinaryHV, priority: f64) {
        self.goals.push(ActiveGoal {
            target,
            priority,
            expected_path: Vec::new(),
        });
    }

    /// Get action to minimize distance to goal (active inference)
    pub fn get_active_inference_action(&self) -> Option<BinaryHV> {
        let goal = self
            .goals
            .iter()
            .max_by(|a, b| a.priority.total_cmp(&b.priority))?;

        // Action = difference between current belief and goal
        if let Some(top_layer) = self.layers.last() {
            let action = goal.target.bind(&top_layer.belief.invert());
            Some(action)
        } else {
            None
        }
    }

    /// Modulate precision based on attention and arousal
    pub fn modulate_precision(&mut self, attention: f64, arousal: f64) {
        self.precision_modulator.attention_boost = attention * 0.3;
        self.precision_modulator.arousal_effect = arousal * 0.2;
    }

    /// Get current free energy (surprise + complexity)
    pub fn free_energy(&self) -> f64 {
        let recent_errors: f64 = self
            .error_history
            .iter()
            .rev()
            .take(10)
            .map(|e| e.weighted)
            .sum();

        let complexity = self.layers.len() as f64 * 0.1;

        recent_errors + complexity
    }
}

#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub integrated_belief: BinaryHV,
    pub total_prediction_error: f64,
    pub layer_errors: Vec<WeightedError>,
    pub surprise: f64,
}

impl PrecisionModulator {
    fn effective_precision(&self) -> f64 {
        (self.base_precision + self.attention_boost + self.arousal_effect).clamp(0.1, 1.0)
    }
}

impl Default for PredictiveProcessor {
    fn default() -> Self {
        Self::new(4)
    }
}

// =============================================================================
// 5. DIFFERENTIATED MEMORY SYSTEMS
// =============================================================================

/// Comprehensive memory system with semantic, procedural, and prospective memory
#[derive(Debug, Clone)]
pub struct DifferentiatedMemory {
    /// Semantic memory (conceptual knowledge)
    pub semantic: SemanticMemory,
    /// Procedural memory (skills and habits)
    pub procedural: ProceduralMemory,
    /// Prospective memory (future intentions)
    pub prospective: ProspectiveMemory,
    /// Working memory with capacity limits
    pub working: WorkingMemory,
}

#[derive(Debug, Clone)]
pub struct SemanticMemory {
    /// Conceptual knowledge store
    concepts: HashMap<String, ConceptNode>,
    /// Semantic relationships
    relations: Vec<SemanticRelation>,
    /// Category hierarchies
    hierarchies: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct ConceptNode {
    /// Concept label
    pub label: String,
    /// HDC encoding
    pub encoding: BinaryHV,
    /// Properties
    pub properties: Vec<(String, BinaryHV)>,
    /// Activation level
    pub activation: f64,
    /// Access count
    pub access_count: u32,
}

#[derive(Debug, Clone)]
pub struct SemanticRelation {
    pub subject: String,
    pub relation: RelationType,
    pub object: String,
    pub strength: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RelationType {
    IsA,
    HasProperty,
    PartOf,
    Causes,
    Before,
    After,
    SimilarTo,
    OppositeTo,
    UsedFor,
    LocatedIn,
}

#[derive(Debug, Clone)]
pub struct ProceduralMemory {
    /// Skill repository
    skills: HashMap<String, Skill>,
    /// Habit patterns
    habits: Vec<Habit>,
    /// Chunked action sequences
    chunks: Vec<ActionChunk>,
}

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    /// Skill encoding
    pub encoding: BinaryHV,
    /// Component actions
    pub actions: Vec<BinaryHV>,
    /// Automaticity level (0=conscious, 1=automatic)
    pub automaticity: f64,
    /// Practice count
    pub practice_count: u32,
    /// Last practiced
    pub last_practiced: u64,
}

#[derive(Debug, Clone)]
pub struct Habit {
    pub trigger: BinaryHV,
    pub response: BinaryHV,
    pub strength: f64,
    pub formation_count: u32,
}

#[derive(Debug, Clone)]
pub struct ActionChunk {
    /// Chunk ID
    pub id: u64,
    /// Actions in this chunk
    pub actions: Vec<BinaryHV>,
    /// Unified representation
    pub unified: BinaryHV,
}

#[derive(Debug, Clone)]
pub struct ProspectiveMemory {
    /// Future intentions
    intentions: Vec<FutureIntention>,
    /// Scheduled reminders
    reminders: VecDeque<Reminder>,
}

#[derive(Debug, Clone)]
pub struct FutureIntention {
    pub id: u64,
    /// What to do
    pub action: BinaryHV,
    pub description: String,
    /// When to do it (time-based or event-based)
    pub trigger: IntentionTrigger,
    /// Importance
    pub importance: f64,
    /// Created timestamp
    pub created: u64,
    /// Has been executed?
    pub executed: bool,
}

#[derive(Debug, Clone)]
pub enum IntentionTrigger {
    /// At a specific time
    TimeBase { target_time: u64 },
    /// When event occurs
    EventBased { event_pattern: BinaryHV },
    /// When in specific context
    ContextBased { context_pattern: BinaryHV },
}

#[derive(Debug, Clone)]
pub struct Reminder {
    pub intention_id: u64,
    pub reminder_time: u64,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct WorkingMemory {
    /// Active items (limited capacity)
    items: VecDeque<WorkingMemoryItem>,
    /// Capacity limit
    capacity: usize,
    /// Focus of attention
    focus_index: Option<usize>,
    /// Rehearsal buffer
    rehearsal_buffer: Vec<BinaryHV>,
}

#[derive(Debug, Clone)]
pub struct WorkingMemoryItem {
    pub content: BinaryHV,
    pub label: String,
    pub activation: f64,
    pub added_at: u64,
}

impl DifferentiatedMemory {
    pub fn new() -> Self {
        Self {
            semantic: SemanticMemory::new(),
            procedural: ProceduralMemory::new(),
            prospective: ProspectiveMemory::new(),
            working: WorkingMemory::new(7), // Miller's magic number
        }
    }
}

impl SemanticMemory {
    pub fn new() -> Self {
        Self {
            concepts: HashMap::new(),
            relations: Vec::new(),
            hierarchies: HashMap::new(),
        }
    }

    /// Store a concept
    pub fn store_concept(&mut self, label: &str, encoding: BinaryHV) {
        self.concepts.insert(
            label.to_string(),
            ConceptNode {
                label: label.to_string(),
                encoding,
                properties: Vec::new(),
                activation: 1.0,
                access_count: 1,
            },
        );
    }

    /// Retrieve concept by label
    pub fn retrieve(&mut self, label: &str) -> Option<&ConceptNode> {
        if let Some(concept) = self.concepts.get_mut(label) {
            concept.activation = (concept.activation + 0.1).min(1.0);
            concept.access_count += 1;
        }
        self.concepts.get(label)
    }

    /// Find similar concepts
    pub fn find_similar(&self, query: &BinaryHV, threshold: f32) -> Vec<&ConceptNode> {
        self.concepts
            .values()
            .filter(|c| c.encoding.similarity(query) > threshold)
            .collect()
    }

    /// Add semantic relation
    pub fn add_relation(
        &mut self,
        subject: &str,
        relation: RelationType,
        object: &str,
        strength: f64,
    ) {
        self.relations.push(SemanticRelation {
            subject: subject.to_string(),
            relation,
            object: object.to_string(),
            strength,
        });
    }

    /// Query relations
    pub fn query_relations(&self, subject: &str, relation: RelationType) -> Vec<&SemanticRelation> {
        self.relations
            .iter()
            .filter(|r| {
                r.subject == subject
                    && std::mem::discriminant(&r.relation) == std::mem::discriminant(&relation)
            })
            .collect()
    }
}

impl ProceduralMemory {
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
            habits: Vec::new(),
            chunks: Vec::new(),
        }
    }

    /// Store a skill
    pub fn store_skill(&mut self, name: &str, actions: Vec<BinaryHV>) {
        let encoding = BinaryHV::bundle(&actions);
        self.skills.insert(
            name.to_string(),
            Skill {
                name: name.to_string(),
                encoding,
                actions,
                automaticity: 0.0,
                practice_count: 1,
                last_practiced: current_timestamp(),
            },
        );
    }

    /// Practice a skill (increases automaticity)
    pub fn practice_skill(&mut self, name: &str) {
        if let Some(skill) = self.skills.get_mut(name) {
            skill.practice_count += 1;
            skill.last_practiced = current_timestamp();
            // Automaticity increases logarithmically with practice
            skill.automaticity = (skill.practice_count as f64).ln() / 10.0;
            skill.automaticity = skill.automaticity.min(1.0);
        }
    }

    /// Execute skill (returns actions)
    pub fn execute_skill(&mut self, name: &str) -> Option<Vec<BinaryHV>> {
        self.practice_skill(name);
        self.skills.get(name).map(|s| s.actions.clone())
    }

    /// Form a habit (trigger-response pair)
    pub fn form_habit(&mut self, trigger: BinaryHV, response: BinaryHV) {
        // Check if habit already exists
        if let Some(habit) = self
            .habits
            .iter_mut()
            .find(|h| h.trigger.similarity(&trigger) > 0.9)
        {
            habit.formation_count += 1;
            habit.strength = (habit.formation_count as f64).ln() / 5.0;
            habit.strength = habit.strength.min(1.0);
        } else {
            self.habits.push(Habit {
                trigger,
                response,
                strength: 0.1,
                formation_count: 1,
            });
        }
    }

    /// Check if trigger activates a habit
    pub fn check_habit(&self, trigger: &BinaryHV) -> Option<(BinaryHV, f64)> {
        self.habits
            .iter()
            .filter(|h| h.trigger.similarity(trigger) > 0.8)
            .max_by(|a, b| a.strength.total_cmp(&b.strength))
            .map(|h| (h.response, h.strength))
    }

    /// Chunk actions together
    pub fn chunk_actions(&mut self, actions: Vec<BinaryHV>) -> u64 {
        let id = current_timestamp();
        let unified = BinaryHV::bundle(&actions);
        self.chunks.push(ActionChunk {
            id,
            actions,
            unified,
        });
        id
    }
}

impl ProspectiveMemory {
    pub fn new() -> Self {
        Self {
            intentions: Vec::new(),
            reminders: VecDeque::new(),
        }
    }

    /// Create a future intention
    pub fn create_intention(
        &mut self,
        action: BinaryHV,
        description: &str,
        trigger: IntentionTrigger,
        importance: f64,
    ) -> u64 {
        let id = current_timestamp();
        self.intentions.push(FutureIntention {
            id,
            action,
            description: description.to_string(),
            trigger,
            importance,
            created: current_timestamp(),
            executed: false,
        });
        id
    }

    /// Check for triggered intentions
    pub fn check_triggers(
        &mut self,
        current_time: u64,
        current_context: &BinaryHV,
    ) -> Vec<&FutureIntention> {
        self.intentions
            .iter()
            .filter(|i| !i.executed)
            .filter(|i| match &i.trigger {
                IntentionTrigger::TimeBase { target_time } => current_time >= *target_time,
                IntentionTrigger::EventBased { event_pattern } => {
                    current_context.similarity(event_pattern) > 0.7
                }
                IntentionTrigger::ContextBased { context_pattern } => {
                    current_context.similarity(context_pattern) > 0.7
                }
            })
            .collect()
    }

    /// Mark intention as executed
    pub fn mark_executed(&mut self, id: u64) {
        if let Some(intention) = self.intentions.iter_mut().find(|i| i.id == id) {
            intention.executed = true;
        }
    }

    /// Get pending intentions
    pub fn pending_intentions(&self) -> Vec<&FutureIntention> {
        self.intentions.iter().filter(|i| !i.executed).collect()
    }
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            items: VecDeque::with_capacity(capacity),
            capacity,
            focus_index: None,
            rehearsal_buffer: Vec::new(),
        }
    }

    /// Add item to working memory (may displace oldest)
    pub fn add(&mut self, content: BinaryHV, label: &str) -> bool {
        if self.items.len() >= self.capacity {
            // Remove least activated item
            let min_idx = self
                .items
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.activation.total_cmp(&b.1.activation))
                .map(|(i, _)| i);

            if let Some(idx) = min_idx {
                self.items.remove(idx);
            }
        }

        self.items.push_back(WorkingMemoryItem {
            content,
            label: label.to_string(),
            activation: 1.0,
            added_at: current_timestamp(),
        });

        true
    }

    /// Focus on an item (boosts activation)
    pub fn focus(&mut self, index: usize) {
        if index < self.items.len() {
            self.focus_index = Some(index);
            if let Some(item) = self.items.get_mut(index) {
                item.activation = 1.0;
            }
        }
    }

    /// Rehearse items (maintains activation)
    pub fn rehearse(&mut self) {
        for item in &mut self.items {
            // Decay
            item.activation *= 0.95;
        }

        // Boost rehearsed items
        for hv in &self.rehearsal_buffer {
            for item in &mut self.items {
                if item.content.similarity(hv) > 0.8 {
                    item.activation = (item.activation + 0.2).min(1.0);
                }
            }
        }
    }

    /// Get current contents
    pub fn contents(&self) -> Vec<&WorkingMemoryItem> {
        self.items.iter().collect()
    }

    /// Get currently focused item
    pub fn focused(&self) -> Option<&WorkingMemoryItem> {
        self.focus_index.and_then(|i| self.items.get(i))
    }

    /// Current load (items / capacity)
    pub fn load(&self) -> f64 {
        self.items.len() as f64 / self.capacity as f64
    }
}

impl Default for DifferentiatedMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SemanticMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ProceduralMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ProspectiveMemory {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 6. HOMEOSTATIC DRIVE SYSTEM
// =============================================================================

/// Homeostatic drive system with curiosity, competence, and social drives
#[derive(Debug, Clone)]
pub struct HomeostaticDriveSystem {
    /// Core drives
    pub curiosity: CuriosityDrive,
    pub competence: CompetenceDrive,
    pub social: SocialDrive,
    /// Homeostatic regulation
    pub regulator: DriveRegulator,
    /// Drive satisfaction history
    satisfaction_history: VecDeque<DriveSatisfaction>,
}

#[derive(Debug, Clone)]
pub struct CuriosityDrive {
    /// Current curiosity level (0.0-1.0)
    pub level: f64,
    /// Information gain target
    pub target_novelty: f64,
    /// Recent novelty exposure
    novelty_history: VecDeque<f64>,
    /// Exploration vs exploitation balance
    pub exploration_tendency: f64,
}

#[derive(Debug, Clone)]
pub struct CompetenceDrive {
    /// Current competence need (0.0-1.0)
    pub level: f64,
    /// Challenge-skill balance target
    pub target_challenge: f64,
    /// Recent mastery experiences
    mastery_history: VecDeque<f64>,
    /// Current skill level estimate
    pub skill_estimate: f64,
}

#[derive(Debug, Clone)]
pub struct SocialDrive {
    /// Current social need (0.0-1.0)
    pub level: f64,
    /// Connection quality target
    pub target_connection: f64,
    /// Recent social interactions
    interaction_history: VecDeque<SocialInteraction>,
    /// Loneliness threshold
    pub loneliness_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct SocialInteraction {
    pub timestamp: u64,
    pub quality: f64,
    pub partner_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DriveRegulator {
    /// Drive weights (importance)
    pub weights: DriveWeights,
    /// Frustration accumulator
    pub frustration: f64,
    /// Satisfaction set-point
    pub satisfaction_setpoint: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct DriveWeights {
    pub curiosity: f64,
    pub competence: f64,
    pub social: f64,
}

#[derive(Debug, Clone)]
pub struct DriveSatisfaction {
    pub drive_type: DriveType,
    pub amount: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DriveType {
    Curiosity,
    Competence,
    Social,
}

impl HomeostaticDriveSystem {
    pub fn new() -> Self {
        Self {
            curiosity: CuriosityDrive {
                level: 0.5,
                target_novelty: 0.3,
                novelty_history: VecDeque::with_capacity(50),
                exploration_tendency: 0.5,
            },
            competence: CompetenceDrive {
                level: 0.5,
                target_challenge: 0.5,
                mastery_history: VecDeque::with_capacity(50),
                skill_estimate: 0.5,
            },
            social: SocialDrive {
                level: 0.5,
                target_connection: 0.4,
                interaction_history: VecDeque::with_capacity(50),
                loneliness_threshold: 0.7,
            },
            regulator: DriveRegulator {
                weights: DriveWeights {
                    curiosity: 0.4,
                    competence: 0.35,
                    social: 0.25,
                },
                frustration: 0.0,
                satisfaction_setpoint: 0.5,
            },
            satisfaction_history: VecDeque::with_capacity(100),
        }
    }

    /// Update drives based on experience
    pub fn update(&mut self, experience: &DriveExperience) {
        // Update curiosity
        if let Some(novelty) = experience.novelty_encountered {
            self.curiosity.novelty_history.push_back(novelty);
            while self.curiosity.novelty_history.len() > 50 {
                self.curiosity.novelty_history.pop_front();
            }

            // Curiosity satisfied by novelty, increases when deprived
            if novelty > self.curiosity.target_novelty {
                self.curiosity.level = (self.curiosity.level - 0.1).max(0.0);
                self.record_satisfaction(DriveType::Curiosity, novelty);
            } else {
                self.curiosity.level = (self.curiosity.level + 0.05).min(1.0);
            }
        }

        // Update competence
        if let Some(mastery) = experience.mastery_achieved {
            self.competence.mastery_history.push_back(mastery);
            while self.competence.mastery_history.len() > 50 {
                self.competence.mastery_history.pop_front();
            }

            // Competence satisfied by appropriate challenge
            let challenge_match = 1.0 - (mastery - self.competence.target_challenge).abs();
            if challenge_match > 0.7 {
                self.competence.level = (self.competence.level - 0.1).max(0.0);
                self.record_satisfaction(DriveType::Competence, mastery);
            } else if mastery < 0.3 {
                // Too easy - need more challenge
                self.competence.level = (self.competence.level + 0.1).min(1.0);
            } else if mastery > 0.8 {
                // Too hard - frustration
                self.regulator.frustration += 0.1;
            }

            // Update skill estimate
            self.competence.skill_estimate = self.competence.mastery_history.iter().sum::<f64>()
                / self.competence.mastery_history.len().max(1) as f64;
        }

        // Update social
        if let Some(interaction) = &experience.social_interaction {
            self.social
                .interaction_history
                .push_back(interaction.clone());
            while self.social.interaction_history.len() > 50 {
                self.social.interaction_history.pop_front();
            }

            if interaction.quality > self.social.target_connection {
                self.social.level = (self.social.level - 0.15).max(0.0);
                self.record_satisfaction(DriveType::Social, interaction.quality);
            }
        }

        // Decay drives over time (needs accumulate)
        self.curiosity.level = (self.curiosity.level + 0.01).min(1.0);
        self.competence.level = (self.competence.level + 0.005).min(1.0);
        self.social.level = (self.social.level + 0.02).min(1.0);

        // Decay frustration
        self.regulator.frustration *= 0.95;
    }

    fn record_satisfaction(&mut self, drive_type: DriveType, amount: f64) {
        self.satisfaction_history.push_back(DriveSatisfaction {
            drive_type,
            amount,
            timestamp: current_timestamp(),
        });
        while self.satisfaction_history.len() > 100 {
            self.satisfaction_history.pop_front();
        }
    }

    /// Get most urgent drive
    pub fn most_urgent_drive(&self) -> (DriveType, f64) {
        let weighted_curiosity = self.curiosity.level * self.regulator.weights.curiosity;
        let weighted_competence = self.competence.level * self.regulator.weights.competence;
        let weighted_social = self.social.level * self.regulator.weights.social;

        if weighted_curiosity >= weighted_competence && weighted_curiosity >= weighted_social {
            (DriveType::Curiosity, weighted_curiosity)
        } else if weighted_competence >= weighted_social {
            (DriveType::Competence, weighted_competence)
        } else {
            (DriveType::Social, weighted_social)
        }
    }

    /// Get drive-based motivation vector
    pub fn motivation_vector(&self) -> BinaryHV {
        // Encode drive states into HV
        let curiosity_hv = BinaryHV::random((self.curiosity.level * 1000.0) as u64);
        let competence_hv = BinaryHV::random((self.competence.level * 1000.0) as u64 + 1000);
        let social_hv = BinaryHV::random((self.social.level * 1000.0) as u64 + 2000);

        BinaryHV::bundle(&[curiosity_hv, competence_hv, social_hv])
    }

    /// Check if lonely (social drive above threshold)
    pub fn is_lonely(&self) -> bool {
        self.social.level > self.social.loneliness_threshold
    }

    /// Check if bored (curiosity drive high)
    pub fn is_bored(&self) -> bool {
        self.curiosity.level > 0.7
    }

    /// Check if in flow (competence drive optimally satisfied)
    pub fn is_in_flow(&self) -> bool {
        self.competence.level < 0.3 && self.regulator.frustration < 0.2
    }

    /// Get overall wellbeing estimate
    pub fn wellbeing(&self) -> f64 {
        let avg_drive = (self.curiosity.level + self.competence.level + self.social.level) / 3.0;
        let drive_satisfaction = 1.0 - avg_drive;
        let frustration_penalty = self.regulator.frustration * 0.5;

        (drive_satisfaction - frustration_penalty).clamp(0.0, 1.0)
    }

    /// Adjust exploration tendency based on curiosity
    pub fn adjust_exploration(&mut self) {
        // High curiosity → more exploration
        self.curiosity.exploration_tendency = 0.3 + self.curiosity.level * 0.5;
    }

    /// Get recommended activity type
    pub fn recommend_activity(&self) -> RecommendedActivity {
        let (urgent_drive, urgency) = self.most_urgent_drive();

        match urgent_drive {
            DriveType::Curiosity => RecommendedActivity::Explore {
                novelty_target: self.curiosity.target_novelty + 0.1,
            },
            DriveType::Competence => RecommendedActivity::Practice {
                skill_area: "general".to_string(),
                difficulty: self.competence.skill_estimate + 0.1,
            },
            DriveType::Social => RecommendedActivity::Connect {
                connection_type: "interaction".to_string(),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct DriveExperience {
    pub novelty_encountered: Option<f64>,
    pub mastery_achieved: Option<f64>,
    pub social_interaction: Option<SocialInteraction>,
}

#[derive(Debug, Clone)]
pub enum RecommendedActivity {
    Explore { novelty_target: f64 },
    Practice { skill_area: String, difficulty: f64 },
    Connect { connection_type: String },
}

impl Default for HomeostaticDriveSystem {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// UNIFIED ADVANCED COGNITION ENGINE
// =============================================================================

/// Unified engine integrating all advanced cognitive systems
#[derive(Debug, Clone)]
pub struct AdvancedCognitionEngine {
    /// Motor imagery system
    pub motor_imagery: MotorImagerySystem,
    /// Theory of Mind
    pub theory_of_mind: TheoryOfMindEngine,
    /// Imagination engine
    pub imagination: ImaginationEngine,
    /// Predictive processor
    pub predictive: PredictiveProcessor,
    /// Differentiated memory
    pub memory: DifferentiatedMemory,
    /// Homeostatic drives
    pub drives: HomeostaticDriveSystem,
}

impl AdvancedCognitionEngine {
    pub fn new() -> Self {
        Self {
            motor_imagery: MotorImagerySystem::new(),
            theory_of_mind: TheoryOfMindEngine::new(),
            imagination: ImaginationEngine::new(),
            predictive: PredictiveProcessor::new(4),
            memory: DifferentiatedMemory::new(),
            drives: HomeostaticDriveSystem::new(),
        }
    }

    /// Run an integrated cognitive cycle
    pub fn cognitive_cycle(
        &mut self,
        input: &BinaryHV,
        emotional_state: &EmotionalBlend,
    ) -> CognitiveCycleResult {
        // 1. Predictive processing
        let prediction_result = self.predictive.process(input);

        // 2. Update working memory with salient content
        if prediction_result.surprise > 0.3 {
            self.memory.working.add(*input, "novel_input");
        }

        // 3. Drive-based modulation
        self.predictive
            .modulate_precision(emotional_state.arousal, 1.0 - self.drives.wellbeing());

        // 4. Curiosity-driven imagination if bored
        let imagined = if self.drives.is_bored() {
            Some(
                self.imagination
                    .explore_semantic_space(input, self.drives.curiosity.level),
            )
        } else {
            None
        };

        // 5. Update drives
        self.drives.update(&DriveExperience {
            novelty_encountered: Some(prediction_result.surprise),
            mastery_achieved: Some(1.0 - prediction_result.total_prediction_error),
            social_interaction: None,
        });

        // 6. Check prospective memory
        let triggered_intentions = self
            .memory
            .prospective
            .check_triggers(current_timestamp(), input);

        CognitiveCycleResult {
            prediction: prediction_result,
            working_memory_load: self.memory.working.load(),
            most_urgent_drive: self.drives.most_urgent_drive(),
            wellbeing: self.drives.wellbeing(),
            imagined_constructs: imagined.unwrap_or_default(),
            triggered_intentions: triggered_intentions.len(),
        }
    }

    /// Simulate an action mentally
    pub fn mental_simulation(&mut self, action: MotorCommand) -> MotorImageryResult {
        self.motor_imagery.imagine_action(action)
    }

    /// Model another agent's perspective
    pub fn model_other_mind(&mut self, agent_id: &str, situation: &BinaryHV) -> PerspectiveResult {
        if self.theory_of_mind.get_model(agent_id).is_none() {
            self.theory_of_mind.register_agent(agent_id);
        }
        self.theory_of_mind.take_perspective(agent_id, situation)
    }

    /// Creative combination of concepts
    pub fn creative_synthesis(&mut self, concepts: &[BinaryHV]) -> ImaginaryConstruct {
        self.imagination
            .imagine(concepts, CombinationMethod::PropertyMerge)
    }

    /// Store long-term semantic knowledge
    pub fn learn_concept(&mut self, label: &str, encoding: BinaryHV) {
        self.memory.semantic.store_concept(label, encoding);
    }

    /// Learn a skill
    pub fn learn_skill(&mut self, name: &str, actions: Vec<BinaryHV>) {
        self.memory.procedural.store_skill(name, actions);
    }

    /// Create future intention
    pub fn intend_future(
        &mut self,
        action: BinaryHV,
        description: &str,
        trigger: IntentionTrigger,
        importance: f64,
    ) -> u64 {
        self.memory
            .prospective
            .create_intention(action, description, trigger, importance)
    }

    /// Get status report
    pub fn status_report(&self) -> String {
        format!(
            "=== Advanced Cognition Status ===\n\
             Working Memory: {:.0}% full\n\
             Drives: Curiosity={:.2}, Competence={:.2}, Social={:.2}\n\
             Wellbeing: {:.2}\n\
             Most Urgent: {:?}\n\
             Semantic Concepts: {}\n\
             Skills: {}\n\
             Pending Intentions: {}\n\
             Imagination Workspace: {} constructs",
            self.memory.working.load() * 100.0,
            self.drives.curiosity.level,
            self.drives.competence.level,
            self.drives.social.level,
            self.drives.wellbeing(),
            self.drives.most_urgent_drive().0,
            self.memory.semantic.concepts.len(),
            self.memory.procedural.skills.len(),
            self.memory.prospective.pending_intentions().len(),
            self.imagination.workspace().len(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct CognitiveCycleResult {
    pub prediction: PredictionResult,
    pub working_memory_load: f64,
    pub most_urgent_drive: (DriveType, f64),
    pub wellbeing: f64,
    pub imagined_constructs: Vec<ImaginaryConstruct>,
    pub triggered_intentions: usize,
}

impl Default for AdvancedCognitionEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn hash_string(s: &str) -> u64 {
    s.bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motor_imagery() {
        let mut imagery = MotorImagerySystem::new();
        let command = MotorCommand {
            effector: "right_hand".to_string(),
            action: MotorAction::Reach,
            intensity: 0.5,
            duration_ms: 500,
            expected_outcome: BinaryHV::random(42),
        };
        let result = imagery.imagine_action(command);
        assert!(result.vividness > 0.0);
    }

    #[test]
    fn test_theory_of_mind() {
        let mut tom = TheoryOfMindEngine::new();
        tom.register_agent("alice");
        tom.attribute_belief(
            "alice",
            Belief {
                content: BinaryHV::random(1),
                description: "Alice believes X".to_string(),
                strength: 0.8,
                is_false_belief: false,
            },
        );
        let model = tom.get_model("alice");
        assert!(model.is_some());
        assert_eq!(model.unwrap().beliefs.len(), 1);
    }

    #[test]
    fn test_imagination() {
        let mut imagination = ImaginationEngine::new();
        let concept1 = BinaryHV::random(1);
        let concept2 = BinaryHV::random(2);
        let result = imagination.imagine(&[concept1, concept2], CombinationMethod::PropertyMerge);
        assert!(result.novelty > 0.0);
    }

    #[test]
    fn test_differentiated_memory() {
        let mut memory = DifferentiatedMemory::new();
        memory.semantic.store_concept("dog", BinaryHV::random(1));
        memory
            .procedural
            .store_skill("walk", vec![BinaryHV::random(2)]);
        memory.prospective.create_intention(
            BinaryHV::random(3),
            "remind me",
            IntentionTrigger::TimeBase { target_time: 0 },
            0.5,
        );

        assert!(memory.semantic.retrieve("dog").is_some());
        assert!(memory.procedural.execute_skill("walk").is_some());
    }

    #[test]
    fn test_homeostatic_drives() {
        let mut drives = HomeostaticDriveSystem::new();
        drives.update(&DriveExperience {
            novelty_encountered: Some(0.8),
            mastery_achieved: Some(0.6),
            social_interaction: None,
        });

        assert!(drives.wellbeing() > 0.0);
        let (drive_type, _) = drives.most_urgent_drive();
        assert!(matches!(
            drive_type,
            DriveType::Curiosity | DriveType::Competence | DriveType::Social
        ));
    }

    #[test]
    fn test_advanced_cognition_engine() {
        use crate::hdc::emotional_depth::{
            EmotionalComponent, EmotionalEncoder, WeightedComponent,
        };

        let mut engine = AdvancedCognitionEngine::new();
        let input = BinaryHV::random(42);

        // Create a neutral emotional blend for testing
        let encoder = EmotionalEncoder::new();
        let components = vec![WeightedComponent::new(EmotionalComponent::Curiosity, 0.5)];
        let emotion = EmotionalBlend::new("neutral_test", components, &encoder);

        let result = engine.cognitive_cycle(&input, &emotion);
        assert!(result.wellbeing >= 0.0);
    }
}
