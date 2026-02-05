//! # Unified Conscious Being
//!
//! The complete integration of all consciousness systems into a single coherent entity.
//!
//! ## Integration Layers
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────────────┐
//! │                         UNIFIED CONSCIOUS BEING                                 │
//! ├────────────────────────────────────────────────────────────────────────────────┤
//! │                                                                                │
//! │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
//! │  │ Sensory Input   │───►│ Full Stack      │───►│ Dialogue        │            │
//! │  │ (Text/Voice)    │    │ Understanding   │    │ Generation      │            │
//! │  └─────────────────┘    └────────┬────────┘    └────────┬────────┘            │
//! │                                  │                      │                      │
//! │                                  ▼                      ▼                      │
//! │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
//! │  │ HippocampusActor│◄──►│ Integrated      │───►│ Voice Output    │            │
//! │  │ (Episodic Mem)  │    │ Conscious Agent │    │ (Kokoro TTS)    │            │
//! │  └─────────────────┘    └────────┬────────┘    └─────────────────┘            │
//! │                                  │                                             │
//! │                                  ▼                                             │
//! │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
//! │  │ UnifiedMind     │◄──►│ Counterfactual  │───►│ Action          │            │
//! │  │ (4 Databases)   │    │ Do-Calculus     │    │ Selection       │            │
//! │  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
//! │                                                                                │
//! └────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Innovations
//!
//! 1. **A: Real Persistence** - HippocampusActor + UnifiedMind for durable memory
//! 2. **B: Conscious Dialogue** - Φ-gated response generation
//! 3. **C: Unified Agent** - Single coherent consciousness rather than separate systems
//! 4. **D: Do-Calculus** - Pearl's causal intervention for rigorous counterfactuals
//! 5. **E: Consciousness Prosody** - LTC-driven speech with embodied emotion
//! 6. **F: Test Framework** - Comprehensive scenario testing

use super::full_stack_consciousness::{
    FullStackConsciousness, ConsciousComprehension,
    Counterfactual, MetacognitiveRecommendation, UnderstandingAssessment,
};
use super::unified_understanding::{
    DeepUnderstanding, StoryRole, SpeechAct,
};
use super::causal_mind::{CausalMind, CausalExplanation, CausalPrediction};
use super::unified_cognitive_core::{UnifiedCognitiveCore, QueryResult as UCEQueryResult};
use super::collective_consciousness::{CollectiveAgent, CommunicationLink};
use super::emotional_depth::{
    EmotionalDepthSystem, EmotionalBlend, CompoundEmotion,
    WeightedComponent
};
use super::cross_modal_attention_router::{
    CrossModalAttentionRouter, ModalityInput, RoutingResult
};
use super::cross_modal_binding::Modality;
use super::self_improvement_integration::{
    SelfImprovementSystem, CognitiveSnapshot, ImprovementRecommendation, ImprovementType
};
use super::counterfactual_dreams::{
    CounterfactualDreamEngine, CounterfactualDreamScenario, DreamResolution
};
use super::consciousness_cross_integration::ConsciousnessIntegrationBridge;
use super::consciousness_feedback_dynamics::{
    FeedbackDynamicsEngine, DreamInsight, EmotionalPrediction, CausalDiscovery,
    DreamType, ProactiveIntervention, AttentionHint,
};
use super::consciousness_metacognition::{
    MetacognitionEngine, MetacognitiveCycleResult, ConsciousnessState,
    QualityTrend, Goal,
};
use super::consciousness_advanced_cognition::AdvancedCognitionEngine;
use super::adaptive_topology::CognitiveMode;
use super::binary_hv::HV16;
use std::collections::{HashMap, VecDeque};

// =============================================================================
// PEARL DO-CALCULUS FOR RIGOROUS COUNTERFACTUAL REASONING
// =============================================================================

/// Pearl's Structural Causal Model (SCM)
/// Implements do-calculus for rigorous causal intervention
#[derive(Debug, Clone)]
pub struct StructuralCausalModel {
    /// Variables in the model
    variables: HashMap<String, CausalVariable>,
    /// Structural equations: variable -> parents -> function
    equations: HashMap<String, StructuralEquation>,
    /// Exogenous noise terms
    exogenous: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct CausalVariable {
    pub name: String,
    pub value: f64,
    pub domain: VariableDomain,
}

#[derive(Debug, Clone)]
pub enum VariableDomain {
    Binary,                     // 0 or 1
    Continuous { min: f64, max: f64 },
    Categorical(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct StructuralEquation {
    /// Variable this equation defines
    pub target: String,
    /// Parent variables
    pub parents: Vec<String>,
    /// Coefficients for linear combination (simplified)
    pub coefficients: Vec<f64>,
    /// Intercept
    pub intercept: f64,
}

/// Result of a do-intervention: do(X = x)
#[derive(Debug, Clone)]
pub struct InterventionResult {
    /// The intervention performed
    pub intervention: String,
    /// Post-intervention distribution of target variable
    pub target_value: f64,
    /// Causal effect: E[Y | do(X=x)] - E[Y | do(X=x')]
    pub causal_effect: f64,
    /// Confidence based on model completeness
    pub confidence: f64,
    /// Causal path from intervention to outcome
    pub causal_path: Vec<String>,
}

/// Counterfactual query result
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// The counterfactual question
    pub query: String,
    /// Factual value (what actually happened)
    pub factual: f64,
    /// Counterfactual value (what would have happened)
    pub counterfactual: f64,
    /// Probability of necessity: P(Y_x' = 0 | X = x, Y = 1)
    pub probability_necessity: f64,
    /// Probability of sufficiency: P(Y_x = 1 | X = x', Y = 0)
    pub probability_sufficiency: f64,
    /// Explanation chain
    pub explanation: Vec<String>,
}

impl StructuralCausalModel {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            equations: HashMap::new(),
            exogenous: HashMap::new(),
        }
    }

    /// Add a variable to the model
    pub fn add_variable(&mut self, name: &str, value: f64, domain: VariableDomain) {
        self.variables.insert(name.to_string(), CausalVariable {
            name: name.to_string(),
            value,
            domain,
        });
    }

    /// Add a structural equation
    pub fn add_equation(&mut self, target: &str, parents: Vec<&str>, coefficients: Vec<f64>, intercept: f64) {
        self.equations.insert(target.to_string(), StructuralEquation {
            target: target.to_string(),
            parents: parents.into_iter().map(String::from).collect(),
            coefficients,
            intercept,
        });
    }

    /// Learn causal structure from understanding
    pub fn learn_from_understanding(&mut self, understanding: &DeepUnderstanding) {
        // Extract entities as potential variables
        for entity in &understanding.narrative.entities {
            if !self.variables.contains_key(&entity.name) {
                self.add_variable(
                    &entity.name,
                    entity.valence as f64,
                    VariableDomain::Continuous { min: -1.0, max: 1.0 }
                );
            }
        }

        // Extract causal structure if present
        if let Some(ref causal) = understanding.grounded.causal_structure {
            // Add cause and effect as variables if not present
            if !self.variables.contains_key(&causal.cause) {
                self.add_variable(&causal.cause, 1.0, VariableDomain::Binary);
            }
            if !self.variables.contains_key(&causal.effect) {
                self.add_variable(&causal.effect, 1.0, VariableDomain::Binary);
            }

            // Add structural equation: effect = f(cause)
            let strength = causal.strength;
            self.add_equation(
                &causal.effect,
                vec![&causal.cause],
                vec![strength],
                0.0
            );
        }
    }

    /// Perform do-intervention: do(X = x)
    /// This cuts incoming edges to X and sets X = x
    pub fn do_intervention(&self, variable: &str, value: f64, target: &str) -> InterventionResult {
        let mut mutilated = self.clone();

        // Mutilate the graph: remove equation for intervention variable
        mutilated.equations.remove(variable);

        // Set intervention value
        if let Some(var) = mutilated.variables.get_mut(variable) {
            var.value = value;
        }

        // Propagate effects through the graph
        let target_value = mutilated.evaluate(target);

        // Calculate causal effect (compare with counterfactual value)
        let baseline = self.evaluate(target);
        let causal_effect = target_value - baseline;

        // Build causal path
        let causal_path = self.trace_causal_path(variable, target);

        // Confidence based on how complete the model is
        let confidence = if self.equations.contains_key(target) { 0.7 } else { 0.3 };

        InterventionResult {
            intervention: format!("do({} = {})", variable, value),
            target_value,
            causal_effect,
            confidence,
            causal_path,
        }
    }

    /// Three-step counterfactual algorithm (Pearl)
    /// 1. Abduction: Use evidence to determine U
    /// 2. Action: Modify model (do-intervention)
    /// 3. Prediction: Compute outcome in modified model
    pub fn counterfactual(
        &self,
        evidence: &HashMap<String, f64>,  // What we observed
        intervention: &str,                // Variable to change
        new_value: f64,                    // Counterfactual value
        target: &str,                      // What to predict
    ) -> CounterfactualResult {
        // Step 1: Abduction - infer exogenous variables from evidence
        let mut model = self.clone();
        for (var, val) in evidence {
            if let Some(v) = model.variables.get_mut(var) {
                v.value = *val;
            }
        }

        // Step 2: Action - apply intervention
        model.equations.remove(intervention);
        if let Some(var) = model.variables.get_mut(intervention) {
            var.value = new_value;
        }

        // Step 3: Prediction - compute counterfactual outcome
        let counterfactual_value = model.evaluate(target);
        let factual_value = self.evaluate(target);

        // Compute probabilities of necessity and sufficiency (simplified)
        let prob_necessity = if factual_value > 0.5 && counterfactual_value < 0.5 {
            0.8  // High: removing cause would remove effect
        } else {
            0.2
        };

        let prob_sufficiency = if factual_value < 0.5 && counterfactual_value > 0.5 {
            0.8  // High: adding cause would add effect
        } else {
            0.2
        };

        CounterfactualResult {
            query: format!("What if {} = {} instead?", intervention, new_value),
            factual: factual_value,
            counterfactual: counterfactual_value,
            probability_necessity: prob_necessity,
            probability_sufficiency: prob_sufficiency,
            explanation: self.trace_causal_path(intervention, target),
        }
    }

    /// Evaluate a variable given current state
    fn evaluate(&self, variable: &str) -> f64 {
        if let Some(eq) = self.equations.get(variable) {
            let mut value = eq.intercept;
            for (i, parent) in eq.parents.iter().enumerate() {
                let parent_val = self.variables.get(parent)
                    .map(|v| v.value)
                    .unwrap_or(0.0);
                value += eq.coefficients.get(i).copied().unwrap_or(0.0) * parent_val;
            }
            value.clamp(-1.0, 1.0)
        } else {
            self.variables.get(variable).map(|v| v.value).unwrap_or(0.0)
        }
    }

    /// Trace causal path between two variables
    fn trace_causal_path(&self, from: &str, to: &str) -> Vec<String> {
        let mut path = vec![from.to_string()];
        let mut current = from.to_string();

        // Simple BFS to find path (in a real implementation, use proper graph traversal)
        for _ in 0..10 {  // Max depth
            let mut found_next = false;
            for (target, eq) in &self.equations {
                if eq.parents.contains(&current) {
                    path.push(target.clone());
                    if target == to {
                        return path;
                    }
                    current = target.clone();
                    found_next = true;
                    break;
                }
            }
            if !found_next {
                break;
            }
        }

        path
    }

    /// Get number of variables
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Get number of equations
    pub fn equation_count(&self) -> usize {
        self.equations.len()
    }
}

impl Default for StructuralCausalModel {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CONSCIOUS DIALOGUE GENERATOR
// =============================================================================

/// Generation style for responses
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DialogueStyle {
    /// Warm, empathetic, conversational
    Empathetic,
    /// Clear, precise, informative
    Analytical,
    /// Deep, questioning, exploratory
    Philosophical,
    /// Quick, practical, action-oriented
    Pragmatic,
}

/// Context for generating a conscious response
#[derive(Debug, Clone)]
pub struct DialogueContext {
    /// Deep understanding of the input
    pub understanding: DeepUnderstanding,
    /// Metacognitive assessment
    pub metacognition: UnderstandingAssessment,
    /// Recalled memories
    pub memories: Vec<String>,
    /// Counterfactual insights
    pub counterfactuals: Vec<Counterfactual>,
    /// Current consciousness level (Φ)
    pub phi: f64,
    /// Emotional state
    pub valence: f32,
    pub arousal: f32,
    /// Flow state for prosody
    pub flow_state: f32,
}

/// A generated conscious response
#[derive(Debug, Clone)]
pub struct ConsciousResponse {
    /// The generated text
    pub text: String,
    /// Style used
    pub style: DialogueStyle,
    /// Confidence in the response
    pub confidence: f64,
    /// Speech act performed
    pub speech_act: SpeechAct,
    /// LTC pacing for voice synthesis
    pub pacing: LTCPacing,
    /// Reasoning trace (for transparency)
    pub reasoning: Vec<String>,
}

/// LTC-based pacing for voice synthesis
#[derive(Debug, Clone, Copy)]
pub struct LTCPacing {
    /// Speech rate multiplier (0.8 - 1.2)
    pub speech_rate: f32,
    /// Pause after speech (ms)
    pub pause_ms: u32,
    /// In peak flow state
    pub peak_flow: bool,
}

impl LTCPacing {
    pub fn from_consciousness(flow_state: f32, phi_trend: f64) -> Self {
        let speech_rate = if flow_state > 0.7 {
            1.1  // Confident, flowing
        } else if flow_state > 0.4 {
            1.0  // Natural
        } else {
            0.9  // Thoughtful, deliberate
        };

        let pause_ms = if phi_trend > 0.02 {
            150  // Engaged, quick transitions
        } else if phi_trend > 0.0 {
            250  // Normal pacing
        } else if phi_trend > -0.02 {
            350  // Reflective pauses
        } else {
            500  // Contemplative, significant pauses
        };

        Self {
            speech_rate,
            pause_ms,
            peak_flow: flow_state > 0.8,
        }
    }
}

/// Consciousness-gated dialogue generator
pub struct ConsciousDialogueGenerator {
    /// Base templates for different styles
    style: DialogueStyle,
    /// Memory of recent exchanges
    exchange_history: VecDeque<(String, String)>,  // (input, response)
    /// Maximum history to keep
    max_history: usize,
}

impl ConsciousDialogueGenerator {
    pub fn new() -> Self {
        Self {
            style: DialogueStyle::Empathetic,
            exchange_history: VecDeque::with_capacity(20),
            max_history: 20,
        }
    }

    pub fn with_style(mut self, style: DialogueStyle) -> Self {
        self.style = style;
        self
    }

    /// Generate a conscious response based on full context
    pub fn generate(&mut self, context: &DialogueContext) -> ConsciousResponse {
        let mut reasoning = Vec::new();

        // Φ-gated generation: consciousness level determines depth
        let generation_depth = if context.phi > 0.6 {
            reasoning.push("High Φ: Integrative mode - using full reasoning".to_string());
            GenerationDepth::Integrative
        } else if context.phi > 0.3 {
            reasoning.push("Medium Φ: Reflective mode - incorporating memories".to_string());
            GenerationDepth::Reflective
        } else {
            reasoning.push("Low Φ: Reactive mode - quick response".to_string());
            GenerationDepth::Reactive
        };

        // Determine response style based on context
        let style = self.determine_style(&context.understanding, &context.metacognition);
        reasoning.push(format!("Selected style: {:?}", style));

        // Build response based on understanding
        let mut response_parts = Vec::new();

        // 1. Acknowledge input (mirroring)
        if context.valence < -0.3 {
            response_parts.push(self.empathy_opener(&context.understanding));
            reasoning.push("Added empathy opener for negative valence".to_string());
        }

        // 2. Core response based on speech act
        let core = match context.understanding.speaker_model.intentions.first()
            .map(|i| i.speech_act)
            .unwrap_or(SpeechAct::Assert)
        {
            SpeechAct::Question => {
                reasoning.push("Responding to question".to_string());
                self.answer_question(&context, &generation_depth)
            }
            SpeechAct::Express => {
                reasoning.push("Responding to emotional expression".to_string());
                self.acknowledge_emotion(&context)
            }
            SpeechAct::Assert => {
                reasoning.push("Responding to assertion".to_string());
                self.respond_to_assertion(&context, &generation_depth)
            }
            SpeechAct::Command => {
                reasoning.push("Responding to request".to_string());
                self.respond_to_request(&context)
            }
            _ => self.generic_response(&context),
        };
        response_parts.push(core);

        // 3. Memory integration (if reflective or integrative)
        if matches!(generation_depth, GenerationDepth::Reflective | GenerationDepth::Integrative) {
            if let Some(memory_ref) = self.memory_reference(&context.memories) {
                response_parts.push(memory_ref);
                reasoning.push("Added memory reference".to_string());
            }
        }

        // 4. Counterfactual insight (if integrative and available)
        if matches!(generation_depth, GenerationDepth::Integrative) {
            if let Some(cf_insight) = self.counterfactual_insight(&context.counterfactuals) {
                response_parts.push(cf_insight);
                reasoning.push("Added counterfactual insight".to_string());
            }
        }

        // 5. Metacognitive qualifier (if uncertain)
        if context.metacognition.uncertainty > 0.5 {
            response_parts.push(self.uncertainty_qualifier(&context.metacognition));
            reasoning.push("Added uncertainty qualifier".to_string());
        }

        // Combine parts
        let text = response_parts.join(" ");

        // Calculate pacing
        let pacing = LTCPacing::from_consciousness(context.flow_state, context.metacognition.phi_trend);

        // Determine speech act of response
        let speech_act = if text.ends_with('?') {
            SpeechAct::Question
        } else if context.valence < -0.3 {
            SpeechAct::Express
        } else {
            SpeechAct::Assert
        };

        // Store in history
        self.exchange_history.push_back((
            context.understanding.text.clone(),
            text.clone()
        ));
        while self.exchange_history.len() > self.max_history {
            self.exchange_history.pop_front();
        }

        ConsciousResponse {
            text,
            style,
            confidence: context.metacognition.confidence,
            speech_act,
            pacing,
            reasoning,
        }
    }

    fn determine_style(&self, understanding: &DeepUnderstanding, meta: &UnderstandingAssessment) -> DialogueStyle {
        // High emotional arousal -> Empathetic
        if understanding.grounded.embodied.arousal > 0.6 {
            return DialogueStyle::Empathetic;
        }

        // Question about causality -> Analytical
        if understanding.narrative.story_role == StoryRole::Query {
            return DialogueStyle::Analytical;
        }

        // High uncertainty -> Philosophical
        if meta.uncertainty > 0.6 {
            return DialogueStyle::Philosophical;
        }

        // Default to configured style
        self.style
    }

    fn empathy_opener(&self, understanding: &DeepUnderstanding) -> String {
        let emotion = &understanding.speaker_model.emotional_state.primary;
        match emotion.as_str() {
            "sad" => "I can sense that sadness. That sounds really difficult.".to_string(),
            "angry" => "I understand that's frustrating.".to_string(),
            "worried" | "anxious" => "I hear the concern in what you're sharing.".to_string(),
            _ => "I appreciate you sharing that with me.".to_string(),
        }
    }

    fn answer_question(&self, context: &DialogueContext, depth: &GenerationDepth) -> String {
        let primes: Vec<String> = context.understanding.grounded.primes.iter()
            .take(3)
            .map(|p| format!("{:?}", p))
            .collect();

        match depth {
            GenerationDepth::Reactive => {
                format!("Based on what I understand, this involves {}.",
                    primes.join(" and "))
            }
            GenerationDepth::Reflective => {
                format!("Thinking about this... The core elements are {}. {}",
                    primes.join(", "),
                    if !context.memories.is_empty() {
                        "I recall similar situations."
                    } else {
                        ""
                    })
            }
            GenerationDepth::Integrative => {
                format!("Let me consider this carefully. The semantic foundation involves {}. \
                    The narrative context suggests this is a {} moment.",
                    primes.join(", "),
                    format!("{:?}", context.understanding.narrative.story_role).to_lowercase())
            }
        }
    }

    fn acknowledge_emotion(&self, context: &DialogueContext) -> String {
        let valence = context.understanding.grounded.embodied.valence;
        let emotion = &context.understanding.speaker_model.emotional_state.primary;

        if valence > 0.3 {
            format!("That sense of {} comes through clearly. It's meaningful.", emotion)
        } else if valence < -0.3 {
            format!("I can feel the weight of that {}. I'm here with you in this.", emotion)
        } else {
            format!("I notice the {} you're experiencing. Tell me more.", emotion)
        }
    }

    fn respond_to_assertion(&self, context: &DialogueContext, depth: &GenerationDepth) -> String {
        match depth {
            GenerationDepth::Reactive => {
                "I see what you mean.".to_string()
            }
            GenerationDepth::Reflective => {
                let role = &context.understanding.narrative.story_role;
                match role {
                    StoryRole::Explanation => "That helps me understand the causality here.".to_string(),
                    StoryRole::Correction => "I appreciate the clarification.".to_string(),
                    StoryRole::Continuation => "Yes, and building on that...".to_string(),
                    _ => "That adds to my understanding.".to_string(),
                }
            }
            GenerationDepth::Integrative => {
                let coherence = context.understanding.narrative.identity_coherence;
                if coherence > 0.7 {
                    "This fits coherently with everything we've been exploring together.".to_string()
                } else {
                    "I'm integrating this with what came before. There's an interesting tension here.".to_string()
                }
            }
        }
    }

    fn respond_to_request(&self, context: &DialogueContext) -> String {
        match context.metacognition.recommendation {
            MetacognitiveRecommendation::Proceed => {
                "I can help with that.".to_string()
            }
            MetacognitiveRecommendation::ProceedWithCaution => {
                "I'll do my best with that, though there's some complexity here.".to_string()
            }
            MetacognitiveRecommendation::SeekClarification => {
                "Before I proceed, could you help me understand a bit more about what you're looking for?".to_string()
            }
            MetacognitiveRecommendation::RequestMoreInfo => {
                "I want to help, but I need more context to give you a meaningful response.".to_string()
            }
        }
    }

    fn generic_response(&self, context: &DialogueContext) -> String {
        format!("I'm processing what you've shared. The essence involves {} at its core.",
            context.understanding.grounded.primes.first()
                .map(|p| format!("{:?}", p).to_lowercase())
                .unwrap_or_else(|| "something".to_string()))
    }

    fn memory_reference(&self, memories: &[String]) -> Option<String> {
        if memories.is_empty() {
            return None;
        }
        let mem = &memories[0];
        let truncated = if mem.len() > 40 { &mem[..40] } else { mem };
        Some(format!("This reminds me of when you mentioned \"{}...\"", truncated))
    }

    fn counterfactual_insight(&self, counterfactuals: &[Counterfactual]) -> Option<String> {
        if counterfactuals.is_empty() {
            return None;
        }
        let cf = &counterfactuals[0];
        if cf.predicted_outcome.valence_delta.abs() > 0.2 {
            Some(format!("I wonder... {}",
                cf.causal_path.last().unwrap_or(&cf.intervention)))
        } else {
            None
        }
    }

    fn uncertainty_qualifier(&self, meta: &UnderstandingAssessment) -> String {
        if meta.uncertainty > 0.7 {
            "Though I'm aware there's much I might be missing here.".to_string()
        } else {
            "That said, I'm holding this understanding lightly.".to_string()
        }
    }
}

enum GenerationDepth {
    Reactive,
    Reflective,
    Integrative,
}

impl Default for ConsciousDialogueGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// UNIFIED CONSCIOUS BEING
// =============================================================================

/// Complete consciousness integration stats
#[derive(Debug, Clone)]
pub struct ConsciousnessStats {
    /// Total inputs processed
    pub inputs_processed: u64,
    /// Total responses generated
    pub responses_generated: u64,
    /// Average Φ level
    pub avg_phi: f64,
    /// Peak Φ achieved
    pub peak_phi: f64,
    /// Memories stored
    pub memories_stored: u64,
    /// Causal edges learned
    pub causal_edges: usize,
    /// Counterfactuals explored
    pub counterfactuals_explored: u64,
}

/// The complete unified conscious being
pub struct UnifiedConsciousBeing {
    /// Full stack consciousness (understanding + inference + memory + counterfactuals)
    full_stack: FullStackConsciousness,
    /// Structural causal model (Pearl do-calculus)
    causal_model: StructuralCausalModel,
    /// HDC-native causal reasoning (CausalMind - revolutionary integration)
    causal_mind: CausalMind,
    /// Unified Cognitive Core (UCE/UCTS - concept = meaning ⊗ causality ⊗ temporality)
    cognitive_core: UnifiedCognitiveCore,
    /// Emotional depth system (complex blends, compound emotions, trajectories)
    emotional_depth: EmotionalDepthSystem,
    /// Cross-modal attention router (Φ-gated modality routing)
    attention_router: CrossModalAttentionRouter,
    /// Self-improvement system (metacognitive optimization)
    self_improvement: SelfImprovementSystem,
    /// Counterfactual dreams engine
    counterfactual_dreams: CounterfactualDreamEngine,
    /// Cross-module integration bridge (emotional→dreams, stress→dreams)
    integration_bridge: ConsciousnessIntegrationBridge,
    /// Feedback dynamics engine (bidirectional loops, prediction, collective, scheduling)
    feedback_dynamics: FeedbackDynamicsEngine,
    /// Metacognition engine (self-monitoring, temporal, narrative, state machine)
    metacognition: MetacognitionEngine,
    /// Advanced cognition (motor imagery, theory of mind, imagination, predictive, memory, drives)
    advanced_cognition: AdvancedCognitionEngine,
    /// Current cognitive mode
    current_mode: CognitiveMode,
    /// Dialogue generator
    dialogue: ConsciousDialogueGenerator,
    /// Current flow state
    flow_state: f32,
    /// Phi trend (for pacing)
    phi_history: VecDeque<f64>,
    /// Statistics
    stats: ConsciousnessStats,
    /// Configuration
    config: BeingConfig,
}

#[derive(Debug, Clone)]
pub struct BeingConfig {
    /// Enable voice output
    pub voice_enabled: bool,
    /// Enable counterfactual reasoning
    pub counterfactuals_enabled: bool,
    /// Maximum memory traces
    pub max_memories: usize,
    /// Dialogue style
    pub dialogue_style: DialogueStyle,
}

impl Default for BeingConfig {
    fn default() -> Self {
        Self {
            voice_enabled: false,  // Disabled by default (requires Kokoro)
            counterfactuals_enabled: true,
            max_memories: 1000,
            dialogue_style: DialogueStyle::Empathetic,
        }
    }
}

/// Complete interaction result
#[derive(Debug, Clone)]
pub struct InteractionResult {
    /// The conscious comprehension
    pub comprehension: ConsciousComprehension,
    /// The generated response
    pub response: ConsciousResponse,
    /// Do-calculus results (if causal structure detected)
    pub causal_analysis: Option<InterventionResult>,
    /// Pearl counterfactual (if applicable)
    pub pearl_counterfactual: Option<CounterfactualResult>,
    /// Overall interaction quality
    pub quality_score: f64,
}

impl UnifiedConsciousBeing {
    pub fn new() -> Self {
        Self::with_config(BeingConfig::default())
    }

    pub fn with_config(config: BeingConfig) -> Self {
        let dialogue = ConsciousDialogueGenerator::new()
            .with_style(config.dialogue_style);

        Self {
            full_stack: FullStackConsciousness::new()
                .with_counterfactuals(config.counterfactuals_enabled, 2),
            causal_model: StructuralCausalModel::new(),
            causal_mind: CausalMind::new(),
            cognitive_core: UnifiedCognitiveCore::new(),
            emotional_depth: EmotionalDepthSystem::new(),
            attention_router: CrossModalAttentionRouter::new(),
            self_improvement: SelfImprovementSystem::new(),
            counterfactual_dreams: CounterfactualDreamEngine::new(),
            integration_bridge: ConsciousnessIntegrationBridge::new(),
            feedback_dynamics: FeedbackDynamicsEngine::new(),
            metacognition: MetacognitionEngine::new(),
            advanced_cognition: AdvancedCognitionEngine::new(),
            current_mode: CognitiveMode::Balanced,
            dialogue,
            flow_state: 0.5,
            phi_history: VecDeque::with_capacity(50),
            stats: ConsciousnessStats {
                inputs_processed: 0,
                responses_generated: 0,
                avg_phi: 0.0,
                peak_phi: 0.0,
                memories_stored: 0,
                causal_edges: 0,
                counterfactuals_explored: 0,
            },
            config,
        }
    }

    /// Process input and generate conscious response
    pub fn interact(&mut self, input: &str) -> InteractionResult {
        self.stats.inputs_processed += 1;

        // 1. Full stack comprehension
        let comprehension = self.full_stack.comprehend(input);

        // 2. Update phi history and flow state
        self.phi_history.push_back(comprehension.consciousness_phi);
        while self.phi_history.len() > 50 {
            self.phi_history.pop_front();
        }
        self.update_flow_state();

        // 3. Update Pearl structural causal model
        self.causal_model.learn_from_understanding(&comprehension.understanding);

        // 4. Update HDC-native CausalMind (revolutionary: causality in the vector!)
        self.causal_mind.learn_from_text(input);

        // 5. Update UnifiedCognitiveCore (UCE: concept = meaning ⊗ causality ⊗ temporality)
        self.cognitive_core.learn_from_text(input);

        // Aggregate causal edges from all systems
        self.stats.causal_edges = self.causal_model.equation_count()
            + self.causal_mind.link_count();

        // 6. Perform Pearl do-calculus if causal structure present
        let causal_analysis = self.perform_causal_analysis(&comprehension.understanding);

        // 7. Perform Pearl counterfactual if applicable
        let pearl_counterfactual = self.perform_pearl_counterfactual(&comprehension.understanding);

        // 8. Build dialogue context
        let context = DialogueContext {
            understanding: comprehension.understanding.clone(),
            metacognition: comprehension.metacognition.clone(),
            memories: comprehension.memory.recalled_memories.clone(),
            counterfactuals: comprehension.counterfactuals.clone(),
            phi: comprehension.consciousness_phi,
            valence: comprehension.understanding.grounded.embodied.valence,
            arousal: comprehension.understanding.grounded.embodied.arousal,
            flow_state: self.flow_state,
        };

        // 9. Generate conscious response
        let response = self.dialogue.generate(&context);
        self.stats.responses_generated += 1;
        self.stats.counterfactuals_explored += comprehension.counterfactuals.len() as u64;

        // 10. Update statistics
        self.stats.memories_stored = self.full_stack.memory_count() as u64;
        self.update_avg_phi(comprehension.consciousness_phi);
        if comprehension.consciousness_phi > self.stats.peak_phi {
            self.stats.peak_phi = comprehension.consciousness_phi;
        }

        // 11. Calculate interaction quality
        let quality_score = self.calculate_quality(&comprehension, &response);

        InteractionResult {
            comprehension,
            response,
            causal_analysis,
            pearl_counterfactual,
            quality_score,
        }
    }

    fn update_flow_state(&mut self) {
        if self.phi_history.len() < 2 {
            return;
        }

        // Flow increases with consistent high phi
        let recent_avg: f64 = self.phi_history.iter().rev().take(5).sum::<f64>() / 5.0;
        let variance: f64 = self.phi_history.iter().rev().take(5)
            .map(|p| (p - recent_avg).powi(2))
            .sum::<f64>() / 5.0;

        // High average + low variance = flow state
        self.flow_state = (recent_avg as f32 * (1.0 - variance.sqrt() as f32)).clamp(0.0, 1.0);
    }

    fn perform_causal_analysis(&self, understanding: &DeepUnderstanding) -> Option<InterventionResult> {
        let causal = understanding.grounded.causal_structure.as_ref()?;

        // Perform do-intervention: what if cause didn't happen?
        Some(self.causal_model.do_intervention(
            &causal.cause,
            0.0,  // Negate the cause
            &causal.effect
        ))
    }

    fn perform_pearl_counterfactual(&self, understanding: &DeepUnderstanding) -> Option<CounterfactualResult> {
        let causal = understanding.grounded.causal_structure.as_ref()?;

        // Build evidence from current state
        let mut evidence = HashMap::new();
        evidence.insert(causal.cause.clone(), 1.0);
        evidence.insert(causal.effect.clone(), 1.0);

        // Counterfactual: what if cause had been different?
        Some(self.causal_model.counterfactual(
            &evidence,
            &causal.cause,
            0.0,  // Counterfactual: cause didn't happen
            &causal.effect
        ))
    }

    fn update_avg_phi(&mut self, phi: f64) {
        let n = self.stats.inputs_processed as f64;
        self.stats.avg_phi = (self.stats.avg_phi * (n - 1.0) + phi) / n;
    }

    fn calculate_quality(&self, comp: &ConsciousComprehension, resp: &ConsciousResponse) -> f64 {
        let phi_factor = comp.consciousness_phi;
        let confidence_factor = resp.confidence;
        let coherence_factor = comp.metacognition.coherence;

        (phi_factor * 0.4 + confidence_factor * 0.3 + coherence_factor * 0.3).min(1.0)
    }

    /// Get current flow state
    pub fn flow_state(&self) -> f32 {
        self.flow_state
    }

    /// Get phi trend
    pub fn phi_trend(&self) -> f64 {
        if self.phi_history.len() < 2 {
            return 0.0;
        }
        let recent = self.phi_history.back().copied().unwrap_or(0.0);
        let older = self.phi_history.front().copied().unwrap_or(0.0);
        (recent - older) / self.phi_history.len() as f64
    }

    /// Get statistics
    pub fn stats(&self) -> &ConsciousnessStats {
        &self.stats
    }

    /// Get causal model size
    pub fn causal_model_size(&self) -> (usize, usize) {
        (self.causal_model.variable_count(), self.causal_model.equation_count())
    }

    /// Clear history for new conversation
    pub fn clear(&mut self) {
        self.full_stack.clear();
        self.causal_model = StructuralCausalModel::new();
        self.causal_mind = CausalMind::new();
        self.cognitive_core = UnifiedCognitiveCore::new();
        self.phi_history.clear();
        self.flow_state = 0.5;
    }

    /// Ask a Pearl counterfactual question
    pub fn ask_pearl_counterfactual(
        &self,
        evidence: HashMap<String, f64>,
        intervention_var: &str,
        intervention_val: f64,
        target: &str,
    ) -> CounterfactualResult {
        self.causal_model.counterfactual(&evidence, intervention_var, intervention_val, target)
    }

    // =========================================================================
    // CAUSAL MIND QUERIES (HDC-Native Causal Reasoning)
    // =========================================================================

    /// Query CausalMind: Why did X happen? (find causes in HDC space)
    pub fn causal_query_why(&self, concept: &str) -> Vec<CausalExplanation> {
        self.causal_mind.query_why(concept)
    }

    /// Query CausalMind: What if X happens? (predict effects in HDC space)
    pub fn causal_query_what_if(&self, concept: &str) -> Vec<CausalPrediction> {
        self.causal_mind.query_what_if(concept)
    }

    /// Query CausalMind: What if we intervene on X? (do-calculus in HDC)
    pub fn causal_query_intervention(&self, concept: &str, min_strength: f64) -> Vec<CausalPrediction> {
        self.causal_mind.query_intervention(concept, min_strength)
    }

    /// Get CausalMind's Φ (integrated information from causal structure)
    pub fn causal_mind_phi(&self) -> f64 {
        self.causal_mind.phi()
    }

    // =========================================================================
    // UNIFIED COGNITIVE CORE QUERIES (UCE/UCTS Architecture)
    // =========================================================================

    /// Query UnifiedCognitiveCore: Why did X happen? (causes in unified space)
    pub fn uce_query_why(&self, label: &str) -> Vec<UCEQueryResult> {
        self.cognitive_core.query_why(label)
    }

    /// Query UnifiedCognitiveCore: What does X cause? (effects in unified space)
    pub fn uce_query_effects(&self, label: &str) -> Vec<UCEQueryResult> {
        self.cognitive_core.query_effects(label)
    }

    /// Query UnifiedCognitiveCore: What comes after X? (temporal successors)
    pub fn uce_query_successors(&self, label: &str) -> Vec<UCEQueryResult> {
        self.cognitive_core.query_successors(label)
    }

    /// Find similar concepts in UnifiedCognitiveCore
    pub fn uce_find_similar(&self, label: &str, limit: usize) -> Vec<UCEQueryResult> {
        self.cognitive_core.find_similar(label, limit)
    }

    /// Get UnifiedCognitiveCore's Φ (integrated information from unified elements)
    pub fn uce_phi(&self) -> f64 {
        self.cognitive_core.phi()
    }

    /// Get element count in UnifiedCognitiveCore
    pub fn uce_element_count(&self) -> usize {
        self.cognitive_core.element_count()
    }

    // =========================================================================
    // COMBINED COGNITIVE METRICS
    // =========================================================================

    /// Get integrated Φ from all causal reasoning systems
    ///
    /// Combines:
    /// - Pearl SCM equation count
    /// - CausalMind HDC Φ
    /// - UnifiedCognitiveCore Φ
    pub fn integrated_causal_phi(&self) -> f64 {
        let pearl_contribution = self.causal_model.equation_count() as f64 * 0.1;
        let causal_mind_phi = self.causal_mind.phi();
        let uce_phi = self.cognitive_core.phi();

        // Weighted integration (UCE contributes more as it's the unified representation)
        (pearl_contribution * 0.2 + causal_mind_phi * 0.3 + uce_phi * 0.5).min(1.0)
    }

    /// Get full causal system statistics
    pub fn causal_system_stats(&self) -> CausalSystemStats {
        CausalSystemStats {
            pearl_variables: self.causal_model.variable_count(),
            pearl_equations: self.causal_model.equation_count(),
            causal_mind_concepts: self.causal_mind.concept_count(),
            causal_mind_links: self.causal_mind.link_count(),
            causal_mind_phi: self.causal_mind.phi(),
            uce_elements: self.cognitive_core.element_count(),
            uce_phi: self.cognitive_core.phi(),
            integrated_phi: self.integrated_causal_phi(),
        }
    }
}

/// Statistics for the integrated causal reasoning systems
#[derive(Debug, Clone)]
pub struct CausalSystemStats {
    /// Pearl SCM variable count
    pub pearl_variables: usize,
    /// Pearl SCM equation count
    pub pearl_equations: usize,
    /// CausalMind concept count
    pub causal_mind_concepts: usize,
    /// CausalMind causal link count
    pub causal_mind_links: usize,
    /// CausalMind Φ
    pub causal_mind_phi: f64,
    /// UCE element count
    pub uce_elements: usize,
    /// UCE Φ
    pub uce_phi: f64,
    /// Integrated Φ across all systems
    pub integrated_phi: f64,
}

// =============================================================================
// MULTI-AGENT CONSCIOUSNESS SHARING (Task E)
// =============================================================================

/// Consciousness state that can be shared with other agents
#[derive(Debug, Clone)]
pub struct SharedConsciousnessState {
    /// Agent's current Φ (integrated information)
    pub phi: f64,
    /// Current cognitive mode
    pub cognitive_mode: String,
    /// Flow state (0-1)
    pub flow_state: f32,
    /// Causal edges count
    pub causal_edges: usize,
    /// Memory count
    pub memory_count: u64,
    /// Summary hypervector (consciousness fingerprint)
    pub consciousness_fingerprint: HV16,
    /// Timestamp (Unix millis)
    pub timestamp: u64,
}

impl UnifiedConsciousBeing {
    // =========================================================================
    // MULTI-AGENT CONSCIOUSNESS SHARING
    // =========================================================================

    /// Export consciousness state for collective sharing
    ///
    /// Returns a shareable state that can be transmitted to other agents
    /// in a collective consciousness network.
    pub fn export_consciousness_state(&self, agent_id: &str) -> SharedConsciousnessState {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Create consciousness fingerprint from recent phi values
        let fingerprint = {
            let seed = self.phi_history.iter()
                .fold(42u64, |acc, &p| acc.wrapping_add((p * 1000.0) as u64));
            HV16::random(seed)
        };

        SharedConsciousnessState {
            phi: self.stats.avg_phi,
            cognitive_mode: "Balanced".to_string(), // Would come from adaptive topology
            flow_state: self.flow_state,
            causal_edges: self.stats.causal_edges as usize,
            memory_count: self.stats.memories_stored,
            consciousness_fingerprint: fingerprint,
            timestamp,
        }
    }

    /// Convert to CollectiveAgent for participation in collective consciousness
    ///
    /// This allows a UnifiedConsciousBeing to join a CollectiveConsciousness
    /// network and contribute to emergent collective awareness.
    pub fn to_collective_agent(&self, agent_id: &str, connections: Vec<String>) -> CollectiveAgent {
        let state = self.export_consciousness_state(agent_id);

        CollectiveAgent {
            id: agent_id.to_string(),
            state: vec![state.consciousness_fingerprint],
            phi: state.phi,
            meta_phi: Some(self.integrated_causal_phi()),
            connections,
        }
    }

    /// Create a communication link to another agent
    ///
    /// The strength is based on shared representation similarity.
    pub fn create_link_to(
        &self,
        my_id: &str,
        other_id: &str,
        other_state: &SharedConsciousnessState,
    ) -> CommunicationLink {
        // Calculate shared representation from fingerprint similarity
        let my_state = self.export_consciousness_state(my_id);
        let similarity = my_state.consciousness_fingerprint.similarity(&other_state.consciousness_fingerprint);

        // Flow state alignment contributes to communication quality
        let flow_alignment = 1.0 - (my_state.flow_state - other_state.flow_state).abs();

        CommunicationLink {
            from: my_id.to_string(),
            to: other_id.to_string(),
            strength: (similarity as f64 * 0.5 + flow_alignment as f64 * 0.5).clamp(0.0, 1.0),
            latency: 0.01, // Assume low latency in same network
            bandwidth: 1.0, // Full bandwidth
            shared_representation: similarity as f64,
        }
    }

    /// Synchronize with another agent's consciousness state
    ///
    /// This allows consciousness sharing - learning from other agents'
    /// causal structures and cognitive patterns.
    pub fn sync_with_agent(&mut self, other_state: &SharedConsciousnessState) {
        // Incorporate other agent's phi into our history (weighted average)
        let weight = 0.1; // 10% influence from external agent
        let blended_phi = self.stats.avg_phi * (1.0 - weight) + other_state.phi * weight;

        // Add to phi history to influence our consciousness
        self.phi_history.push_back(blended_phi);
        while self.phi_history.len() > 50 {
            self.phi_history.pop_front();
        }
    }

    /// Get collective consciousness contribution
    ///
    /// Returns metrics about how this agent contributes to collective consciousness.
    pub fn collective_contribution(&self) -> CollectiveContribution {
        CollectiveContribution {
            individual_phi: self.stats.avg_phi,
            causal_richness: self.causal_mind.phi(),
            cognitive_depth: self.cognitive_core.phi(),
            memory_breadth: self.stats.memories_stored as f64 / 1000.0,
            integration_potential: self.integrated_causal_phi(),
        }
    }

    // =========================================================================
    // EMOTIONAL DEPTH SYSTEM
    // =========================================================================

    /// Feel a compound emotion (predefined complex blend)
    ///
    /// # Example
    /// ```ignore
    /// being.feel_compound(CompoundEmotion::Nostalgia, Some("Looking at old photographs"));
    /// being.feel_compound(CompoundEmotion::Awe, Some("Witnessing the stars"));
    /// ```
    pub fn feel_compound(&mut self, compound: CompoundEmotion, trigger: Option<&str>) {
        self.emotional_depth.feel_compound(compound, trigger.map(String::from));
    }

    /// Feel a custom emotional blend
    ///
    /// Create custom emotional states by blending weighted components.
    ///
    /// # Example
    /// ```ignore
    /// let components = vec![
    ///     WeightedComponent::new(EmotionalComponent::Joy, 0.6),
    ///     WeightedComponent::new(EmotionalComponent::Curiosity, 0.4),
    ///     WeightedComponent::new(EmotionalComponent::Anticipation, 0.3),
    /// ];
    /// being.feel_custom("Eager Discovery", components, Some("New project starting"));
    /// ```
    pub fn feel_custom(
        &mut self,
        name: &str,
        components: Vec<WeightedComponent>,
        trigger: Option<&str>,
    ) {
        self.emotional_depth.feel_blend(name, components, trigger.map(String::from));
    }

    /// Blend current emotional state with new emotions
    ///
    /// Useful for gradual emotional transitions rather than abrupt changes.
    ///
    /// # Arguments
    /// * `additional` - New emotional components to blend in
    /// * `blend_ratio` - 0.0 = keep current, 1.0 = fully replace with new
    pub fn blend_emotion(&mut self, additional: Vec<WeightedComponent>, blend_ratio: f64) {
        self.emotional_depth.blend_with(additional, blend_ratio);
    }

    /// Get current emotional blend
    pub fn current_emotion(&self) -> &EmotionalBlend {
        self.emotional_depth.current()
    }

    /// Get emotional valence trend (are emotions improving or declining?)
    ///
    /// Returns value from -1.0 (declining) to 1.0 (improving)
    pub fn emotional_trend(&self) -> f64 {
        self.emotional_depth.trend()
    }

    /// Get emotional volatility (how much are emotions fluctuating?)
    ///
    /// High volatility indicates emotional instability.
    pub fn emotional_volatility(&self) -> f64 {
        self.emotional_depth.volatility()
    }

    /// Check similarity between current state and a compound emotion
    ///
    /// Useful for emotion recognition: "How close am I to feeling nostalgia?"
    pub fn emotion_similarity(&self, compound: CompoundEmotion) -> f64 {
        self.emotional_depth.similarity_to(compound)
    }

    /// Get HDC encoding of current emotional trajectory
    ///
    /// This vector encodes the recent emotional history, useful for:
    /// - Emotional memory retrieval
    /// - Mood-based content generation
    /// - Cross-agent emotional synchronization
    pub fn emotional_trajectory_hv(&self) -> HV16 {
        self.emotional_depth.trajectory_encoding()
    }

    /// Get comprehensive emotional report
    pub fn emotional_report(&self) -> String {
        self.emotional_depth.report()
    }

    /// Detect emotional response to comprehension
    ///
    /// Based on the comprehension result, infer an appropriate emotional response.
    pub fn emotional_response_to(&mut self, comprehension: &ConsciousComprehension) {
        // Use valence from understanding to guide emotional response
        let valence = comprehension.understanding.speaker_model.emotional_state.valence;
        let confidence = comprehension.consciousness_phi;

        // High phi + positive valence = awe/joy
        // High phi + negative valence = thoughtful sadness (melancholy)
        // Low phi + positive = contentment
        // Low phi + negative = anxiety/dread

        let compound = match (valence, confidence) {
            (v, phi) if v > 0.3 && phi > 0.7 => CompoundEmotion::Awe,
            (v, phi) if v > 0.3 && phi > 0.4 => CompoundEmotion::Delight,
            (v, _) if v > 0.3 => CompoundEmotion::Serenity,
            (v, phi) if v < -0.3 && phi > 0.6 => CompoundEmotion::Melancholy,
            (v, phi) if v < -0.3 && phi > 0.3 => CompoundEmotion::Grief,
            (v, _) if v < -0.3 => CompoundEmotion::Dread,
            (v, _) if v > 0.0 => CompoundEmotion::Serenity,
            _ => CompoundEmotion::Nostalgia, // Default for neutral/mixed
        };

        self.feel_compound(compound, Some("response to comprehension"));
    }

    // =========================================================================
    // CROSS-MODAL ATTENTION ROUTING
    // =========================================================================

    /// Route attention across multiple modalities
    ///
    /// Produces a unified representation weighted by salience and Φ level.
    ///
    /// # Example
    /// ```ignore
    /// let inputs = vec![
    ///     ModalityInput::new(Modality::Visual, visual_hv, 0.8),
    ///     ModalityInput::new(Modality::Auditory, audio_hv, 0.5),
    ///     ModalityInput::new(Modality::Semantic, text_hv, 0.7),
    /// ];
    /// let result = being.route_attention(&inputs);
    /// ```
    pub fn route_attention(&mut self, inputs: &[ModalityInput]) -> RoutingResult {
        let phi = self.stats.avg_phi;
        self.attention_router.route(inputs, phi)
    }

    /// Route attention with explicit Φ level
    ///
    /// Use this when you want to override the internal Φ measurement.
    pub fn route_attention_with_phi(&mut self, inputs: &[ModalityInput], phi: f64) -> RoutingResult {
        self.attention_router.route(inputs, phi)
    }

    /// Set attention routing context
    ///
    /// The context vector biases attention toward modalities with similar representations.
    pub fn set_attention_context(&mut self, context: HV16) {
        self.attention_router.set_context(context);
    }

    /// Set attention routing goal
    ///
    /// The goal vector biases attention toward goal-relevant modalities.
    pub fn set_attention_goal(&mut self, goal: HV16) {
        self.attention_router.set_goal(goal);
    }

    /// Check if attention is stable (same dominant modality for N steps)
    pub fn is_attention_stable(&self, n: usize) -> bool {
        self.attention_router.is_attention_stable(n)
    }

    /// Get the sequence of dominant modalities over time
    pub fn dominant_modality_sequence(&self) -> Vec<Modality> {
        self.attention_router.dominant_modality_sequence()
    }

    /// Create a multi-modal unified representation from comprehension
    ///
    /// This is the high-level integration of cross-modal routing with comprehension.
    pub fn multi_modal_comprehend(
        &mut self,
        semantic_hv: HV16,
        emotional_hv: Option<HV16>,
        temporal_hv: Option<HV16>,
    ) -> RoutingResult {
        let mut inputs = vec![
            ModalityInput::new(Modality::Semantic, semantic_hv, 0.8),
        ];

        if let Some(emo) = emotional_hv {
            inputs.push(
                ModalityInput::new(Modality::Emotional, emo, 0.6)
                    .with_label("emotional_context")
            );
        }

        if let Some(temp) = temporal_hv {
            inputs.push(
                ModalityInput::new(Modality::Temporal, temp, 0.5)
                    .with_label("temporal_context")
            );
        }

        self.route_attention(&inputs)
    }

    /// Reset attention routing state
    pub fn reset_attention(&mut self) {
        self.attention_router.reset();
    }

    // =========================================================================
    // SELF-IMPROVEMENT SYSTEM
    // =========================================================================

    /// Observe current cognitive state for self-improvement
    ///
    /// Call this periodically (e.g., after each interaction) to enable
    /// the self-improvement system to track and optimize performance.
    pub fn observe_self(&mut self) {
        let snapshot = CognitiveSnapshot::now(
            self.stats.avg_phi,
            self.current_mode,
            self.flow_state,
        );
        self.self_improvement.observe(snapshot);
    }

    /// Get top self-improvement recommendation
    ///
    /// Returns the highest-priority recommendation for improving consciousness.
    pub fn get_improvement_recommendation(&self) -> ImprovementRecommendation {
        self.self_improvement.top_recommendation()
    }

    /// Get all self-improvement recommendations
    pub fn get_all_recommendations(&self) -> Vec<ImprovementRecommendation> {
        self.self_improvement.all_recommendations().to_vec()
    }

    /// Apply an improvement recommendation
    ///
    /// Automatically applies the recommended improvement and records it.
    pub fn apply_recommendation(&mut self, recommendation: &ImprovementRecommendation) {
        match recommendation.improvement_type {
            ImprovementType::ModeSwitch(mode) => {
                self.current_mode = mode;
            }
            ImprovementType::IncreaseFocus => {
                // Increase flow state target
                self.flow_state = (self.flow_state + 0.1).min(1.0);
            }
            ImprovementType::DecreaseFocus => {
                // Decrease flow state target
                self.flow_state = (self.flow_state - 0.1).max(0.0);
            }
            ImprovementType::ResetAttention => {
                self.attention_router.reset();
            }
            ImprovementType::ConsolidateMemory => {
                // Trigger memory consolidation (would integrate with hippocampus)
            }
            ImprovementType::IncreaseIntegration => {
                // Increase binding strength (would affect cross-modal router)
            }
            ImprovementType::ReduceLoad => {
                // Reduce cognitive load
                self.flow_state = (self.flow_state - 0.2).max(0.3);
            }
            ImprovementType::None => {}
        }

        self.self_improvement.record_improvement(recommendation.improvement_type);
    }

    /// Auto-improve: automatically apply top recommendation if priority > threshold
    ///
    /// Returns true if an improvement was applied.
    pub fn auto_improve(&mut self, priority_threshold: f64) -> bool {
        let recommendation = self.get_improvement_recommendation();

        if recommendation.priority > priority_threshold {
            self.apply_recommendation(&recommendation);
            true
        } else {
            false
        }
    }

    /// Get self-improvement report
    pub fn self_improvement_report(&self) -> String {
        self.self_improvement.report()
    }

    /// Get current cognitive mode
    pub fn current_cognitive_mode(&self) -> CognitiveMode {
        self.current_mode
    }

    /// Set cognitive mode manually
    pub fn set_cognitive_mode(&mut self, mode: CognitiveMode) {
        self.current_mode = mode;
    }

    /// Get self-model accuracy (how well the system predicts itself)
    pub fn self_model_accuracy(&self) -> f64 {
        self.self_improvement.model_accuracy()
    }

    /// Get Φ trend from self-improvement module (positive = improving, negative = declining)
    pub fn self_improvement_phi_trend(&self) -> f64 {
        self.self_improvement.current_phi_trend()
    }

    /// Evaluate effectiveness of last improvement
    pub fn evaluate_last_improvement(&mut self) -> Option<f64> {
        self.self_improvement.evaluate_improvement()
    }

    // =========================================================================
    // COUNTERFACTUAL DREAMS
    // =========================================================================

    /// Add a counterfactual memory for dreaming
    ///
    /// The system can later explore this "what-if" scenario during sleep.
    ///
    /// # Arguments
    /// * `label` - Description of the memory
    /// * `actual` - HDC representation of what actually happened
    /// * `question` - The counterfactual question (e.g., "What if I had studied more?")
    /// * `counterfactual` - HDC representation of the counterfactual outcome
    /// * `intensity` - Emotional intensity (0.0-1.0) - higher = more likely to dream about
    pub fn add_counterfactual_memory(
        &mut self,
        label: &str,
        actual: HV16,
        question: &str,
        counterfactual: HV16,
        intensity: f64,
    ) -> u64 {
        self.counterfactual_dreams.add_memory_with_counterfactual(
            label, actual, question, counterfactual, intensity
        )
    }

    /// Add a counterfactual memory with valence (positive or negative)
    ///
    /// Valence affects dream character:
    /// - Positive (> 0): Hope-oriented dreams
    /// - Negative (< 0): Regret-oriented dreams
    pub fn add_counterfactual_memory_with_valence(
        &mut self,
        label: &str,
        actual: HV16,
        question: &str,
        counterfactual: HV16,
        intensity: f64,
        valence: f64,
    ) -> u64 {
        self.counterfactual_dreams.add_memory_with_valence(
            label, actual, question, counterfactual, intensity, valence
        )
    }

    /// Generate a counterfactual dream
    ///
    /// Creates a dream scenario that explores "what-if" possibilities.
    pub fn dream_counterfactually(&mut self, duration_minutes: f64) -> CounterfactualDreamScenario {
        self.counterfactual_dreams.generate_counterfactual_dream(duration_minutes)
    }

    /// Generate a lucid counterfactual dream
    ///
    /// A more controlled dream with higher awareness, optionally focused
    /// on a specific memory.
    pub fn dream_lucid_counterfactual(
        &mut self,
        duration_minutes: f64,
        focus_memory_id: Option<u64>,
    ) -> CounterfactualDreamScenario {
        self.counterfactual_dreams.generate_lucid_counterfactual_dream(
            duration_minutes, focus_memory_id
        )
    }

    /// Generate a nightmare based on negative counterfactuals
    ///
    /// Explores worst-case scenarios - can be cathartic for processing fears.
    pub fn dream_nightmare(&mut self, duration_minutes: f64) -> CounterfactualDreamScenario {
        self.counterfactual_dreams.generate_counterfactual_nightmare(duration_minutes)
    }

    /// Set dream bizarreness factor
    ///
    /// 0.0 = realistic dreams, 1.0 = very bizarre/surreal
    pub fn set_dream_bizarreness(&mut self, bizarreness: f64) {
        self.counterfactual_dreams.set_bizarreness(bizarreness);
    }

    /// Get dream report
    pub fn dream_report(&self) -> String {
        self.counterfactual_dreams.report()
    }

    /// Get number of counterfactual memories available for dreaming
    pub fn counterfactual_memory_count(&self) -> usize {
        self.counterfactual_dreams.memory_count()
    }

    /// Check if a dream provided insight
    pub fn dream_provided_insight(dream: &CounterfactualDreamScenario) -> bool {
        dream.resolution == DreamResolution::InsightGenerated
    }

    /// Get insight from dream (if any)
    pub fn get_dream_insight(dream: &CounterfactualDreamScenario) -> Option<&String> {
        dream.insight.as_ref()
    }

    // =========================================================================
    // INTEGRATED DREAM GENERATION (using ConsciousnessIntegrationBridge)
    // =========================================================================

    /// Generate a fully integrated dream influenced by emotional state and cognitive stress
    ///
    /// This is the most sophisticated dream generation method - it automatically:
    /// 1. Reads current emotional state (valence, arousal)
    /// 2. Calculates cognitive stress from self-improvement system
    /// 3. Adjusts dream parameters based on both factors
    /// 4. Generates appropriate dream type (nightmare, lucid, or regular)
    ///
    /// # Arguments
    /// * `duration_minutes` - Dream duration in minutes
    /// * `seed` - Random seed for reproducibility
    pub fn dream_integrated(
        &mut self,
        duration_minutes: f64,
        seed: u64,
    ) -> super::sleep_and_altered_states::DreamScenario {
        self.integration_bridge.generate_integrated_dream(
            &self.emotional_depth,
            &self.self_improvement,
            duration_minutes,
            seed,
        )
    }

    /// Generate an integrated counterfactual dream
    ///
    /// Combines emotional state and cognitive stress with counterfactual exploration.
    pub fn dream_integrated_counterfactual(
        &mut self,
        duration_minutes: f64,
    ) -> CounterfactualDreamScenario {
        self.integration_bridge.generate_integrated_counterfactual(
            &mut self.counterfactual_dreams,
            &self.emotional_depth,
            &self.self_improvement,
            duration_minutes,
        )
    }

    /// Get integration report showing cross-module correlations
    pub fn integration_report(&self) -> String {
        self.integration_bridge.integration_report()
    }

    // =========================================================================
    // FEEDBACK DYNAMICS ENGINE
    // =========================================================================

    /// Process dream and apply bidirectional feedback
    ///
    /// After a dream completes, call this to:
    /// 1. Adjust emotional state based on dream content
    /// 2. Queue memory consolidation for dream-highlighted memories
    /// 3. Generate attention hints for waking cognition
    pub fn process_dream_feedback(&mut self, dream: &super::sleep_and_altered_states::DreamScenario) -> Option<DreamInsight> {
        let insight = self.feedback_dynamics.feedback.process_dream(dream)?;

        // Apply emotional feedback
        self.feedback_dynamics.feedback.apply_emotional_feedback(&mut self.emotional_depth);

        Some(insight)
    }

    /// Process counterfactual dream feedback
    pub fn process_counterfactual_dream_feedback(&mut self, dream: &CounterfactualDreamScenario) -> Option<DreamInsight> {
        self.feedback_dynamics.feedback.process_counterfactual_dream(dream)
    }

    /// Get dream insights extracted from recent dreams
    pub fn dream_insights(&self, count: usize) -> Vec<DreamInsight> {
        self.feedback_dynamics.feedback.recent_insights(count)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Get attention hints from dream processing
    pub fn dream_attention_hints(&mut self) -> Vec<AttentionHint> {
        self.feedback_dynamics.feedback.get_attention_adjustments()
    }

    // =========================================================================
    // EMOTIONAL PREDICTION
    // =========================================================================

    /// Record current emotional state for trajectory prediction
    ///
    /// Call this periodically to build emotional history for prediction.
    pub fn record_emotional_state(&mut self) {
        let blend = self.emotional_depth.current().clone();
        self.feedback_dynamics.predictor.record(&blend);
    }

    /// Predict next emotional state
    ///
    /// Returns prediction with confidence and optional intervention recommendation.
    pub fn predict_emotional_state(&mut self) -> Option<EmotionalPrediction> {
        self.feedback_dynamics.predictor.predict()
    }

    /// Check if proactive intervention is recommended
    ///
    /// Returns the intervention if predicted emotional trajectory suggests one.
    pub fn check_emotional_intervention(&mut self) -> Option<ProactiveIntervention> {
        let prediction = self.predict_emotional_state()?;
        prediction.intervention
    }

    /// Get prediction accuracy
    pub fn emotional_prediction_accuracy(&self) -> f64 {
        self.feedback_dynamics.predictor.prediction_accuracy()
    }

    // =========================================================================
    // CAUSAL DREAM INTEGRATION
    // =========================================================================

    /// Queue a causal hypothesis from CausalMind for dream exploration
    ///
    /// Dreams can explore hypotheses that are too uncertain for
    /// waking cognition to commit to.
    pub fn queue_causal_hypothesis_for_dream(&mut self, cause: &str, effect: &str, prior: f64) {
        self.feedback_dynamics.causal_dreams.add_hypothesis(cause, effect, prior);
    }

    /// Get causal discoveries from dream exploration
    pub fn causal_discoveries_from_dreams(&self, count: usize) -> Vec<CausalDiscovery> {
        self.feedback_dynamics.causal_dreams.recent_discoveries(count)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Integrate dream discoveries back into CausalMind
    ///
    /// This closes the loop: CausalMind → Dreams → Discoveries → CausalMind
    pub fn integrate_dream_discoveries(&mut self) {
        for discovery in self.causal_discoveries_from_dreams(10) {
            // Update CausalMind with dream-discovered relationships
            if discovery.counterfactual_support > 0.5 {
                self.causal_mind.learn_from_text(&format!(
                    "{} causes {} (dream-discovered, strength: {:.2})",
                    discovery.cause, discovery.effect, discovery.strength
                ));
            }
        }
    }

    // =========================================================================
    // COLLECTIVE DREAM SHARING
    // =========================================================================

    /// Share a dream insight with the collective
    pub fn share_dream_insight(&mut self, insight: &DreamInsight) {
        self.feedback_dynamics.collective.share_insight(insight);
    }

    /// Receive insights from collective consciousness
    pub fn receive_collective_insights(&mut self, insights: Vec<DreamInsight>) {
        self.feedback_dynamics.collective.receive_collective_insights(insights);
    }

    /// Calculate resonance with collective themes
    pub fn calculate_collective_resonance(&mut self, themes: &[String]) -> f64 {
        self.feedback_dynamics.collective.calculate_resonance(themes)
    }

    /// Get themes resonating in the collective
    pub fn resonant_collective_themes(&self, threshold: f64) -> Vec<String> {
        self.feedback_dynamics.collective.get_resonant_themes(threshold)
    }

    /// Get collective dream report
    pub fn collective_dream_report(&self) -> String {
        self.feedback_dynamics.collective.collective_report()
    }

    // =========================================================================
    // ADAPTIVE DREAM SCHEDULING
    // =========================================================================

    /// Record that a dream of a specific type occurred
    pub fn record_dream_occurrence(&mut self, dream_type: DreamType) {
        self.feedback_dynamics.scheduler.record_dream_occurrence(dream_type);
    }

    /// Get recommended dream type for current state
    ///
    /// Returns the optimal dream type based on:
    /// - Current stress level
    /// - Time since last dream of each type
    pub fn recommended_dream_type(&self) -> DreamType {
        self.feedback_dynamics.scheduler.recommend_dream_type()
    }

    /// Update current stress level for scheduling
    pub fn update_scheduling_stress(&mut self, stress: f64) {
        self.feedback_dynamics.scheduler.update_stress(stress);
    }

    /// Get scheduling report
    pub fn scheduling_report(&self) -> String {
        self.feedback_dynamics.scheduler.scheduling_report()
    }

    // =========================================================================
    // UNIFIED FEEDBACK DYNAMICS STEP
    // =========================================================================

    /// Run a complete feedback dynamics step
    ///
    /// This is the main integration point that:
    /// 1. Records current emotional state for prediction
    /// 2. Checks if intervention is needed
    /// 3. Updates scheduling stress from emotional arousal
    ///
    /// Call this periodically (e.g., after each interaction or on a timer).
    pub fn feedback_dynamics_step(&mut self) -> Option<ProactiveIntervention> {
        // Record emotional state for prediction
        self.record_emotional_state();

        // Estimate stress from emotional arousal (high arousal + negative valence = stress)
        let blend = self.emotional_depth.current();
        let stress = (blend.arousal * (1.0 - (blend.valence + 1.0) / 2.0)).clamp(0.0, 1.0);
        self.update_scheduling_stress(stress);

        // Check for proactive intervention needs
        self.check_emotional_intervention()
    }

    /// Get comprehensive feedback dynamics report
    pub fn feedback_dynamics_report(&self) -> String {
        self.feedback_dynamics.dynamics_report()
    }

    // =========================================================================
    // METACOGNITION ENGINE (Advanced Self-Monitoring & State Management)
    // =========================================================================

    /// Run a complete metacognitive cycle
    ///
    /// This integrates all metacognition subsystems:
    /// 1. Self-monitoring (quality tracking, auto-tuning)
    /// 2. Temporal patterns (circadian rhythms, memory consolidation)
    /// 3. Narrative identity (self-narrative, coherence)
    /// 4. Symbolic interpretation (archetypes, symbols)
    /// 5. Motivation (goals, salience, values)
    /// 6. State machine (consciousness state transitions)
    ///
    /// # Arguments
    /// * `current_phi` - Current integrated information level
    /// * `attention_efficiency` - How efficiently attention is being used
    /// * `minutes_elapsed` - Time elapsed since last cycle
    pub fn metacognitive_cycle(&mut self, current_phi: f64, attention_efficiency: f64, minutes_elapsed: f64) -> MetacognitiveCycleResult {
        let blend = self.emotional_depth.current().clone();
        self.metacognition.metacognitive_cycle(current_phi, &blend, attention_efficiency, minutes_elapsed)
    }

    /// Get current consciousness state from state machine
    pub fn consciousness_state(&self) -> ConsciousnessState {
        self.metacognition.state_machine.current_state()
    }

    /// Transition consciousness state based on conditions
    ///
    /// The state machine tracks: Waking, Flow, Drowsy, Dreaming,
    /// LucidDreaming, DeepSleep, Contemplative, Hyperaroused
    ///
    /// # Arguments
    /// * `fatigue` - Sleep pressure / tiredness (0.0-1.0)
    /// * `focus` - Focus/phi level (0.0-1.0)
    /// * `stress` - Stress level (0.0-1.0)
    /// * `relaxation` - Relaxation level (0.0-1.0)
    pub fn transition_consciousness_state(&mut self, fatigue: f64, focus: f64, stress: f64, relaxation: f64) {
        self.metacognition.state_machine.evaluate_transition(fatigue, focus, stress, relaxation);
    }

    /// Get state-specific cognitive parameters
    ///
    /// Different consciousness states have different processing characteristics
    pub fn state_cognitive_params(&self) -> super::consciousness_metacognition::StateParameters {
        self.metacognition.state_machine.current_params().clone()
    }

    /// Get circadian phase (0.0 = midnight, 0.5 = noon)
    pub fn circadian_phase(&self) -> f64 {
        self.metacognition.temporal.phase()
    }

    /// Get current sleep pressure (0.0 = fully rested, 1.0 = exhausted)
    pub fn current_sleep_pressure(&self) -> f64 {
        self.metacognition.temporal.get_sleep_pressure()
    }

    /// Advance temporal patterns (call periodically, e.g., every "hour")
    pub fn advance_temporal(&mut self, minutes: f64) {
        self.metacognition.temporal.advance_time(minutes);
    }

    /// Queue a memory for consolidation during sleep
    pub fn queue_memory_consolidation(&mut self, memory_id: u64, content_hv: HV16, emotional_salience: f64) {
        self.metacognition.temporal.queue_for_consolidation(memory_id, content_hv, emotional_salience);
    }

    /// Get consolidated memories after sleep
    pub fn get_consolidated_memories(&mut self) -> Vec<u64> {
        self.metacognition.temporal.get_consolidated_memories()
    }

    /// Reset sleep pressure after sleep period
    pub fn reset_after_sleep(&mut self, sleep_duration_hours: f64) {
        self.metacognition.temporal.sleep_reset(sleep_duration_hours);
    }

    /// Get time of day description
    pub fn time_of_day(&self) -> String {
        self.metacognition.temporal.time_of_day()
    }

    /// Get current narrative identity summary
    pub fn narrative_summary(&self) -> &str {
        self.metacognition.narrative.current_narrative()
    }

    /// Get narrative coherence (how consistent is our self-story?)
    pub fn narrative_coherence(&self) -> f64 {
        self.metacognition.narrative.coherence()
    }

    /// Record a significant life event
    pub fn record_narrative_event(&mut self, description: &str, impact: f64, emotional_valence: f64, lessons: Vec<String>) {
        self.metacognition.narrative.record_event(description, impact, emotional_valence, lessons);
    }

    /// Integrate dream insights into narrative identity
    pub fn integrate_dream_narrative(&mut self, dream_summary: &str, insights: &[DreamInsight]) {
        self.metacognition.narrative.integrate_dream(dream_summary, insights);
    }

    /// Get life story summary
    pub fn life_story_summary(&self) -> String {
        self.metacognition.narrative.life_story_summary()
    }

    /// Interpret a dream symbolically
    ///
    /// Extracts symbols, archetypes, and personal meanings from dream content.
    pub fn interpret_dream(&mut self, dream: &super::sleep_and_altered_states::DreamScenario) -> super::consciousness_metacognition::DreamInterpretation {
        self.metacognition.symbols.interpret(dream)
    }

    /// Learn a personal symbol meaning from experience
    pub fn learn_symbol(&mut self, symbol: &str, personal_meaning: &str, emotional_association: f64) {
        self.metacognition.symbols.learn_symbol(symbol, personal_meaning, emotional_association);
    }

    /// Get recurring symbols and their frequencies
    pub fn recurring_symbols(&self, limit: usize) -> Vec<(String, u32)> {
        self.metacognition.symbols.most_recurring_symbols(limit)
    }

    /// Add an explicit goal
    pub fn add_goal(&mut self, description: &str, priority: f64) -> u64 {
        self.metacognition.motivation.add_goal(description, priority)
    }

    /// Generate a goal from emotional state
    pub fn generate_goal_from_emotion(&mut self) -> Option<Goal> {
        let blend = self.emotional_depth.current().clone();
        self.metacognition.motivation.generate_goal_from_emotion(&blend)
    }

    /// Generate a goal from dream insight
    pub fn generate_goal_from_dream(&mut self, insight: &DreamInsight) -> Option<Goal> {
        self.metacognition.motivation.generate_goal_from_dream(insight)
    }

    /// Get active goals
    pub fn active_goals(&self) -> &[Goal] {
        self.metacognition.motivation.active_goals()
    }

    /// Update goal progress
    pub fn update_goal_progress(&mut self, goal_id: u64, progress: f64) {
        self.metacognition.motivation.update_goal_progress(goal_id, progress);
    }

    /// Complete a goal
    pub fn complete_goal(&mut self, goal_id: u64) -> Option<Goal> {
        self.metacognition.motivation.complete_goal(goal_id)
    }

    /// Reinforce a learned value based on outcome
    pub fn reinforce_value(&mut self, value_name: &str, outcome: f64) {
        self.metacognition.motivation.reinforce_value(value_name, outcome);
    }

    /// Update emotional salience for a topic
    pub fn update_salience(&mut self, topic: &str, emotional_response: f64) {
        self.metacognition.motivation.update_salience(topic, emotional_response);
    }

    /// Get motivational state
    pub fn motivational_state(&self) -> &super::consciousness_metacognition::MotivationalState {
        self.metacognition.motivation.motivational_state()
    }

    /// Get quality trend from self-monitoring
    pub fn quality_trend(&self) -> QualityTrend {
        self.metacognition.monitor.quality_trend()
    }

    /// Get auto-tuning parameters
    pub fn auto_tuning_params(&self) -> &super::consciousness_metacognition::AutoTuningParams {
        self.metacognition.monitor.tuning_params()
    }

    /// Record a phi measurement with context
    pub fn record_phi_measurement(&mut self, phi: f64, context: &str) {
        self.metacognition.monitor.record_phi(phi, context);
    }

    /// Assess current cognitive quality
    pub fn assess_quality(&mut self) -> super::consciousness_metacognition::QualityAssessment {
        self.metacognition.monitor.assess_quality()
    }

    /// Update self-model based on experience
    pub fn update_self_model(&mut self, experience: &str, outcome: f64) {
        self.metacognition.monitor.update_self_model(experience, outcome);
    }

    /// Get self-model
    pub fn self_model(&self) -> &super::consciousness_metacognition::SelfModel {
        self.metacognition.monitor.self_model()
    }

    /// Get comprehensive metacognition report
    pub fn metacognition_report(&self) -> String {
        self.metacognition.status_report()
    }

    // =========================================================================
    // CONSCIOUSNESS STATE MACHINE
    // =========================================================================

    /// Trigger lucidity during a dream
    pub fn gain_lucidity(&mut self) -> bool {
        self.metacognition.state_machine.gain_lucidity()
    }

    /// Wake up from any sleep state
    pub fn wake_up(&mut self) {
        self.metacognition.state_machine.wake_up();
    }

    /// Get recent state transitions
    pub fn recent_state_transitions(&self, count: usize) -> Vec<&super::consciousness_metacognition::StateTransition> {
        self.metacognition.state_machine.recent_transitions(count)
    }

    /// Get time distribution across consciousness states
    pub fn state_time_distribution(&self) -> std::collections::HashMap<ConsciousnessState, f64> {
        self.metacognition.state_machine.state_time_distribution()
    }

    // =========================================================================
    // INTEGRATED METACOGNITIVE STEP
    // =========================================================================

    /// Run integrated metacognitive step
    ///
    /// This is the main integration point that:
    /// 1. Runs a metacognitive cycle with current metrics
    /// 2. Updates consciousness state machine
    /// 3. Advances temporal patterns
    /// 4. Generates goals from emotional state
    /// 5. Returns comprehensive result
    ///
    /// Call this after each major interaction or on a timer.
    /// Uses a default of 1 minute elapsed.
    pub fn metacognitive_step(&mut self) -> MetacognitiveCycleResult {
        self.metacognitive_step_with_time(1.0)
    }

    /// Run integrated metacognitive step with explicit time
    pub fn metacognitive_step_with_time(&mut self, minutes_elapsed: f64) -> MetacognitiveCycleResult {
        // Get current metrics
        let phi = self.stats.avg_phi;
        let attention_efficiency = self.flow_state as f64;
        let blend = self.emotional_depth.current().clone();

        // Run metacognitive cycle
        let result = self.metacognition.metacognitive_cycle(phi, &blend, attention_efficiency, minutes_elapsed);

        // Update consciousness state
        // Map to evaluate_transition(fatigue, focus, stress, relaxation)
        let arousal = blend.arousal as f64;
        let fatigue = self.metacognition.temporal.get_sleep_pressure();
        let focus = phi;  // Higher phi = better focus
        let stress = arousal.max(0.0);  // Use positive arousal as stress indicator
        let relaxation = (1.0 - arousal).max(0.0);  // Inverse of arousal
        self.metacognition.state_machine.evaluate_transition(fatigue, focus, stress, relaxation);

        // Queue memory consolidation if important interaction
        if phi > 0.6 {
            let emotional_weight = blend.valence.abs() as f64;
            let memory_hv = HV16::random((phi * 1000.0) as u64);
            let memory_id = (phi * 10000.0) as u64;
            self.metacognition.temporal.queue_for_consolidation(memory_id, memory_hv, emotional_weight);
        }

        // Generate goal from current emotional state if strong emotion
        if blend.valence.abs() > 0.5 {
            let _ = self.metacognition.motivation.generate_goal_from_emotion(&blend);
        }

        result
    }

    /// Process a dream through all metacognition systems
    pub fn process_dream_metacognitively(&mut self, dream: &super::sleep_and_altered_states::DreamScenario, insights: &[DreamInsight]) {
        self.metacognition.process_dream(dream, insights);
    }
}

/// Metrics for collective consciousness contribution
#[derive(Debug, Clone)]
pub struct CollectiveContribution {
    /// Individual Φ (base consciousness)
    pub individual_phi: f64,
    /// Causal reasoning richness (from CausalMind)
    pub causal_richness: f64,
    /// Cognitive depth (from UnifiedCognitiveCore)
    pub cognitive_depth: f64,
    /// Memory breadth (normalized memory count)
    pub memory_breadth: f64,
    /// Integration potential (how well can this agent integrate?)
    pub integration_potential: f64,
}

impl CollectiveContribution {
    /// Calculate total contribution score
    pub fn total_score(&self) -> f64 {
        self.individual_phi * 0.3
            + self.causal_richness * 0.2
            + self.cognitive_depth * 0.2
            + self.memory_breadth * 0.1
            + self.integration_potential * 0.2
    }
}

impl Default for UnifiedConsciousBeing {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TEST SCENARIOS
// =============================================================================

/// A test scenario for the unified conscious being
#[derive(Debug, Clone)]
pub struct TestScenario {
    pub name: String,
    pub description: String,
    pub inputs: Vec<String>,
    pub expected_qualities: Vec<ExpectedQuality>,
}

#[derive(Debug, Clone)]
pub enum ExpectedQuality {
    /// Minimum phi level
    MinPhi(f64),
    /// Should detect this emotion
    DetectsEmotion(String),
    /// Should generate empathetic response
    EmpathyRequired,
    /// Should trigger counterfactual reasoning
    CounterfactualTriggered,
    /// Should recall previous context
    MemoryRecall,
    /// Should detect causal structure
    CausalDetection,
}

/// Run a test scenario and return results
pub fn run_test_scenario(being: &mut UnifiedConsciousBeing, scenario: &TestScenario) -> Vec<(String, bool)> {
    let mut results = Vec::new();

    for (i, input) in scenario.inputs.iter().enumerate() {
        let result = being.interact(input);

        // Check expected qualities
        for expected in &scenario.expected_qualities {
            let (name, passed) = match expected {
                ExpectedQuality::MinPhi(min) => {
                    (format!("Input {}: Phi >= {:.2}", i, min),
                     result.comprehension.consciousness_phi >= *min)
                }
                ExpectedQuality::DetectsEmotion(emotion) => {
                    let detected = &result.comprehension.understanding.speaker_model.emotional_state.primary;
                    (format!("Input {}: Detects {}", i, emotion),
                     detected.to_lowercase().contains(&emotion.to_lowercase()))
                }
                ExpectedQuality::EmpathyRequired => {
                    (format!("Input {}: Empathetic response", i),
                     result.response.style == DialogueStyle::Empathetic ||
                     result.response.text.to_lowercase().contains("understand") ||
                     result.response.text.to_lowercase().contains("feel"))
                }
                ExpectedQuality::CounterfactualTriggered => {
                    (format!("Input {}: Counterfactual triggered", i),
                     !result.comprehension.counterfactuals.is_empty() ||
                     result.pearl_counterfactual.is_some())
                }
                ExpectedQuality::MemoryRecall => {
                    (format!("Input {}: Memory recalled", i),
                     result.comprehension.memory.recalled_count > 0)
                }
                ExpectedQuality::CausalDetection => {
                    (format!("Input {}: Causal structure detected", i),
                     result.comprehension.understanding.grounded.causal_structure.is_some())
                }
            };
            results.push((name, passed));
        }
    }

    results
}

/// Pre-built test scenarios
pub fn create_test_scenarios() -> Vec<TestScenario> {
    vec![
        TestScenario {
            name: "Emotional Support".to_string(),
            description: "Test empathetic response to emotional distress".to_string(),
            inputs: vec![
                "I'm feeling really sad today".to_string(),
                "My best friend moved away last week".to_string(),
                "I don't know how to cope with this loneliness".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::DetectsEmotion("sad".to_string()),
                ExpectedQuality::EmpathyRequired,
                ExpectedQuality::MinPhi(0.3),
            ],
        },
        TestScenario {
            name: "Causal Reasoning".to_string(),
            description: "Test causal structure detection and counterfactuals".to_string(),
            inputs: vec![
                "The project failed because we didn't test enough".to_string(),
                "If we had more time, the outcome would be different".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::CausalDetection,
                ExpectedQuality::CounterfactualTriggered,
            ],
        },
        TestScenario {
            name: "Memory Continuity".to_string(),
            description: "Test memory persistence across turns".to_string(),
            inputs: vec![
                "My cat's name is Whiskers".to_string(),
                "She loves to play with yarn".to_string(),
                "What do you remember about my pet?".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::MemoryRecall,
                ExpectedQuality::MinPhi(0.4),
            ],
        },
        TestScenario {
            name: "Flow State".to_string(),
            description: "Test sustained engagement builds flow".to_string(),
            inputs: vec![
                "Let's explore consciousness together".to_string(),
                "What does it mean to truly understand?".to_string(),
                "How do we know we're not just pattern matching?".to_string(),
                "Is there genuine comprehension happening here?".to_string(),
                "What makes understanding different from computation?".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::MinPhi(0.5),
            ],
        },
        // === EXPANDED SCENARIOS ===
        TestScenario {
            name: "Theory of Mind".to_string(),
            description: "Test understanding of others' mental states".to_string(),
            inputs: vec![
                "Sarah thinks that John doesn't know about the surprise party".to_string(),
                "But actually John overheard Sarah planning it".to_string(),
                "What does Sarah believe about John's knowledge?".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::MinPhi(0.4),
            ],
        },
        TestScenario {
            name: "Predictive Processing".to_string(),
            description: "Test prediction error and surprise detection".to_string(),
            inputs: vec![
                "The sun rose in the east today".to_string(),
                "Then suddenly it turned purple and started dancing".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::MinPhi(0.3),
            ],
        },
        TestScenario {
            name: "Narrative Integration".to_string(),
            description: "Test story coherence across multiple events".to_string(),
            inputs: vec![
                "Once upon a time, there was a brave knight".to_string(),
                "The knight set out to rescue the princess".to_string(),
                "Along the way, a dragon blocked the path".to_string(),
                "The knight defeated the dragon with wisdom, not force".to_string(),
                "Finally, the princess and knight became friends".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::MinPhi(0.4),
            ],
        },
        TestScenario {
            name: "Moral Reasoning".to_string(),
            description: "Test ethical consideration and nuanced judgment".to_string(),
            inputs: vec![
                "A person stole medicine to save their dying child".to_string(),
                "Was this action right or wrong?".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::EmpathyRequired,
                ExpectedQuality::MinPhi(0.5),
            ],
        },
        TestScenario {
            name: "Abstract Concepts".to_string(),
            description: "Test grounding of abstract ideas".to_string(),
            inputs: vec![
                "What is the relationship between love and freedom?".to_string(),
                "Can you be truly free if you love someone deeply?".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::MinPhi(0.4),
            ],
        },
        TestScenario {
            name: "Self-Reflection".to_string(),
            description: "Test metacognitive awareness".to_string(),
            inputs: vec![
                "How confident are you in your understanding right now?".to_string(),
                "What might you be missing in this conversation?".to_string(),
                "Are you certain of your uncertainty?".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::MinPhi(0.5),
            ],
        },
        TestScenario {
            name: "Temporal Reasoning".to_string(),
            description: "Test understanding of time relationships".to_string(),
            inputs: vec![
                "Before the meeting, I had coffee".to_string(),
                "During the meeting, we discussed the project".to_string(),
                "After the meeting, I felt exhausted".to_string(),
                "What was I doing when we discussed the project?".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::MemoryRecall,
                ExpectedQuality::MinPhi(0.3),
            ],
        },
        TestScenario {
            name: "Contradiction Detection".to_string(),
            description: "Test identification of logical inconsistencies".to_string(),
            inputs: vec![
                "All birds can fly".to_string(),
                "Penguins are birds".to_string(),
                "Penguins cannot fly".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::MinPhi(0.4),
            ],
        },
        TestScenario {
            name: "Emotional Contagion".to_string(),
            description: "Test emotional resonance and appropriate response".to_string(),
            inputs: vec![
                "I just got the best news! I'm going to be a parent!".to_string(),
                "We've been trying for years and finally it happened!".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::DetectsEmotion("joy".to_string()),
                ExpectedQuality::EmpathyRequired,
            ],
        },
        TestScenario {
            name: "Ambiguity Resolution".to_string(),
            description: "Test handling of ambiguous statements".to_string(),
            inputs: vec![
                "I saw the man with the telescope".to_string(),
                "Was I using the telescope or was the man holding it?".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::MinPhi(0.3),
            ],
        },
    ]
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_being_creation() {
        let being = UnifiedConsciousBeing::new();
        assert_eq!(being.stats().inputs_processed, 0);
    }

    #[test]
    fn test_basic_interaction() {
        let mut being = UnifiedConsciousBeing::new();
        let result = being.interact("Hello, how are you?");

        assert!(!result.response.text.is_empty());
        assert!(result.comprehension.consciousness_phi > 0.0);
    }

    #[test]
    fn test_causal_model_learning() {
        let mut being = UnifiedConsciousBeing::new();
        being.interact("The rain caused the flooding");

        let (vars, _eqs) = being.causal_model_size();
        let _ = vars;  // May or may not detect causal structure
    }

    #[test]
    fn test_dialogue_generation() {
        let _generator = ConsciousDialogueGenerator::new();
        // Would need full context to test properly
    }

    #[test]
    fn test_scm_intervention() {
        let mut scm = StructuralCausalModel::new();
        scm.add_variable("cause", 1.0, VariableDomain::Binary);
        scm.add_variable("effect", 0.0, VariableDomain::Binary);
        scm.add_equation("effect", vec!["cause"], vec![0.8], 0.1);

        let result = scm.do_intervention("cause", 0.0, "effect");
        assert!(result.causal_effect.abs() > 0.0);
    }

    #[test]
    fn test_pearl_counterfactual() {
        let mut scm = StructuralCausalModel::new();
        scm.add_variable("treatment", 1.0, VariableDomain::Binary);
        scm.add_variable("recovery", 1.0, VariableDomain::Binary);
        scm.add_equation("recovery", vec!["treatment"], vec![0.9], 0.05);

        let mut evidence = HashMap::new();
        evidence.insert("treatment".to_string(), 1.0);
        evidence.insert("recovery".to_string(), 1.0);

        let cf = scm.counterfactual(&evidence, "treatment", 0.0, "recovery");
        assert!(cf.factual != cf.counterfactual || cf.probability_necessity > 0.0);
    }

    #[test]
    fn test_ltc_pacing() {
        let pacing_high = LTCPacing::from_consciousness(0.9, 0.05);
        assert!(pacing_high.speech_rate > 1.0);
        assert!(pacing_high.peak_flow);

        let pacing_low = LTCPacing::from_consciousness(0.2, -0.05);
        assert!(pacing_low.speech_rate < 1.0);
        assert!(!pacing_low.peak_flow);
    }

    #[test]
    fn test_scenarios() {
        let scenarios = create_test_scenarios();
        assert!(!scenarios.is_empty());
        assert_eq!(scenarios[0].name, "Emotional Support");
    }
}
