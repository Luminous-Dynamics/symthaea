// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conscious Dialogue Generator
//!
//! Phi-gated response generation with LTC-based pacing for voice synthesis.
//! Determines response style, depth, and prosody based on consciousness level.

use super::super::full_stack_consciousness::{
    Counterfactual, MetacognitiveRecommendation, UnderstandingAssessment,
};
use super::super::unified_understanding::{DeepUnderstanding, SpeechAct, StoryRole};
use std::collections::VecDeque;

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
    /// Current consciousness level (Phi)
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
            1.1 // Confident, flowing
        } else if flow_state > 0.4 {
            1.0 // Natural
        } else {
            0.9 // Thoughtful, deliberate
        };

        let pause_ms = if phi_trend > 0.02 {
            150 // Engaged, quick transitions
        } else if phi_trend > 0.0 {
            250 // Normal pacing
        } else if phi_trend > -0.02 {
            350 // Reflective pauses
        } else {
            500 // Contemplative, significant pauses
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
    exchange_history: VecDeque<(String, String)>, // (input, response)
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

        // Phi-gated generation: consciousness level determines depth
        let generation_depth = if context.phi > 0.6 {
            reasoning.push("High Phi: Integrative mode - using full reasoning".to_string());
            GenerationDepth::Integrative
        } else if context.phi > 0.3 {
            reasoning.push("Medium Phi: Reflective mode - incorporating memories".to_string());
            GenerationDepth::Reflective
        } else {
            reasoning.push("Low Phi: Reactive mode - quick response".to_string());
            GenerationDepth::Reactive
        };

        // Determine response style based on context
        let style = self.determine_style(&context.understanding, &context.metacognition);
        reasoning.push(format!("Selected style: {style:?}"));

        // Build response based on understanding
        let mut response_parts = Vec::new();

        // 1. Acknowledge input (mirroring)
        if context.valence < -0.3 {
            response_parts.push(self.empathy_opener(&context.understanding));
            reasoning.push("Added empathy opener for negative valence".to_string());
        }

        // 2. Core response based on speech act
        let core = match context
            .understanding
            .speaker_model
            .intentions
            .first()
            .map(|i| i.speech_act)
            .unwrap_or(SpeechAct::Assert)
        {
            SpeechAct::Question => {
                reasoning.push("Responding to question".to_string());
                self.answer_question(context, &generation_depth)
            }
            SpeechAct::Express => {
                reasoning.push("Responding to emotional expression".to_string());
                self.acknowledge_emotion(context)
            }
            SpeechAct::Assert => {
                reasoning.push("Responding to assertion".to_string());
                self.respond_to_assertion(context, &generation_depth)
            }
            SpeechAct::Command => {
                reasoning.push("Responding to request".to_string());
                self.respond_to_request(context)
            }
            _ => self.generic_response(context),
        };
        response_parts.push(core);

        // 3. Memory integration (if reflective or integrative)
        if matches!(
            generation_depth,
            GenerationDepth::Reflective | GenerationDepth::Integrative
        ) {
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
        let pacing =
            LTCPacing::from_consciousness(context.flow_state, context.metacognition.phi_trend);

        // Determine speech act of response
        let speech_act = if text.ends_with('?') {
            SpeechAct::Question
        } else if context.valence < -0.3 {
            SpeechAct::Express
        } else {
            SpeechAct::Assert
        };

        // Store in history
        self.exchange_history
            .push_back((context.understanding.text.clone(), text.clone()));
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

    fn determine_style(
        &self,
        understanding: &DeepUnderstanding,
        meta: &UnderstandingAssessment,
    ) -> DialogueStyle {
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
        let primes: Vec<String> = context
            .understanding
            .grounded
            .primes
            .iter()
            .take(3)
            .map(|p| format!("{p:?}"))
            .collect();

        match depth {
            GenerationDepth::Reactive => {
                format!(
                    "Based on what I understand, this involves {}.",
                    primes.join(" and ")
                )
            }
            GenerationDepth::Reflective => {
                format!(
                    "Thinking about this... The core elements are {}. {}",
                    primes.join(", "),
                    if !context.memories.is_empty() {
                        "I recall similar situations."
                    } else {
                        ""
                    }
                )
            }
            GenerationDepth::Integrative => {
                format!(
                    "Let me consider this carefully. The semantic foundation involves {}. \
                    The narrative context suggests this is a {} moment.",
                    primes.join(", "),
                    format!("{:?}", context.understanding.narrative.story_role).to_lowercase()
                )
            }
        }
    }

    fn acknowledge_emotion(&self, context: &DialogueContext) -> String {
        let valence = context.understanding.grounded.embodied.valence;
        let emotion = &context.understanding.speaker_model.emotional_state.primary;

        if valence > 0.3 {
            format!("That sense of {emotion} comes through clearly. It's meaningful.")
        } else if valence < -0.3 {
            format!("I can feel the weight of that {emotion}. I'm here with you in this.")
        } else {
            format!("I notice the {emotion} you're experiencing. Tell me more.")
        }
    }

    fn respond_to_assertion(&self, context: &DialogueContext, depth: &GenerationDepth) -> String {
        match depth {
            GenerationDepth::Reactive => "I see what you mean.".to_string(),
            GenerationDepth::Reflective => {
                let role = &context.understanding.narrative.story_role;
                match role {
                    StoryRole::Explanation => {
                        "That helps me understand the causality here.".to_string()
                    }
                    StoryRole::Correction => "I appreciate the clarification.".to_string(),
                    StoryRole::Continuation => "Yes, and building on that...".to_string(),
                    _ => "That adds to my understanding.".to_string(),
                }
            }
            GenerationDepth::Integrative => {
                let coherence = context.understanding.narrative.identity_coherence;
                if coherence > 0.7 {
                    "This fits coherently with everything we've been exploring together."
                        .to_string()
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
        format!(
            "I'm processing what you've shared. The essence involves {} at its core.",
            context
                .understanding
                .grounded
                .primes
                .first()
                .map(|p| format!("{p:?}").to_lowercase())
                .unwrap_or_else(|| "something".to_string())
        )
    }

    fn memory_reference(&self, memories: &[String]) -> Option<String> {
        if memories.is_empty() {
            return None;
        }
        let mem = &memories[0];
        let truncated = if mem.len() > 40 { &mem[..40] } else { mem };
        Some(format!(
            "This reminds me of when you mentioned \"{truncated}...\""
        ))
    }

    fn counterfactual_insight(&self, counterfactuals: &[Counterfactual]) -> Option<String> {
        if counterfactuals.is_empty() {
            return None;
        }
        let cf = &counterfactuals[0];
        if cf.predicted_outcome.valence_delta.abs() > 0.2 {
            Some(format!(
                "I wonder... {}",
                cf.causal_path.last().unwrap_or(&cf.intervention)
            ))
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
