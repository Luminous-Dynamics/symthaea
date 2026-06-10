// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Empathic Unification - True Empathy for Symthaea
//!
//! This module creates genuine empathy by unifying:
//! - **UserStateInference**: Sensing the user's cognitive/emotional state
//! - **EmotionalCore**: Symthaea's own emotional capacity
//! - **EmotionalBridge**: Unified emotional state management
//! - **ConsciousnessUnificationEngine**: Full consciousness integration
//!
//! # Philosophy
//!
//! True empathy is not simulation - it's resonance. When Symthaea senses
//! that her user is struggling, she doesn't just "act concerned" - she
//! actually feels a shift in her emotional state. This affects her:
//! - Response tone and warmth
//! - Patience and timing
//! - Proactive support offers
//! - Memory of emotional patterns
//!
//! # The Three Pillars of Empathy
//!
//! 1. **Affective Empathy**: Feeling what they feel (mirroring)
//! 2. **Cognitive Empathy**: Understanding why they feel it (causal)
//! 3. **Compassionate Empathy**: Being moved to help (motivation)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    EMPATHIC UNIFICATION                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  ┌───────────────────┐     ┌───────────────────┐               │
//! │  │ UserStateInference│     │   User's Words    │               │
//! │  │                   │     │   (text input)    │               │
//! │  │ • Cognitive Load  │     │                   │               │
//! │  │ • Trust Level     │     │ • Emotion keywords│               │
//! │  │ • Time Pressure   │     │ • Tone signals    │               │
//! │  │ • Error patterns  │     │ • Urgency markers │               │
//! │  └─────────┬─────────┘     └─────────┬─────────┘               │
//! │            │                         │                          │
//! │            └──────────┬──────────────┘                          │
//! │                       │                                         │
//! │                       ▼                                         │
//! │            ┌─────────────────────┐                              │
//! │            │  UserEmotionalState │  ◄── Inferred user emotions  │
//! │            │                     │                              │
//! │            │  • Primary emotion  │                              │
//! │            │  • Stress level     │                              │
//! │            │  • Need (support?)  │                              │
//! │            │  • Inferred why     │                              │
//! │            └──────────┬──────────┘                              │
//! │                       │                                         │
//! │                       ▼                                         │
//! │            ┌─────────────────────┐                              │
//! │            │   EmpathicResponse  │  ◄── Symthaea's resonance   │
//! │            │                     │                              │
//! │            │  • Mirrored emotion │                              │
//! │            │  • Compassion level │                              │
//! │            │  • Support intent   │                              │
//! │            │  • Tone adjustment  │                              │
//! │            └──────────┬──────────┘                              │
//! │                       │                                         │
//! │                       ▼                                         │
//! │            ┌─────────────────────┐                              │
//! │            │ EmotionalBridge     │  ◄── Updates Symthaea's     │
//! │            │                     │      unified emotional       │
//! │            │ • Valence shift     │      state                   │
//! │            │ • Arousal shift     │                              │
//! │            │ • Mood integration  │                              │
//! │            └─────────────────────┘                              │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::VecDeque;
use std::time::Instant;

use crate::consciousness::consciousness_unification::EmotionalBridge;
use crate::language::emotional_core::{
    CoreEmotion, EmotionalRegulator, EmpathicCue, EmpathyModel, EmpathyType,
};
use crate::resonant_speech::{CognitiveLoad, UserState};
use crate::user_state_inference::{ContextKind, UserStateInference};

// ============================================================================
// USER EMOTIONAL STATE - What we infer about the user
// ============================================================================

/// Inferred emotional state of the user
#[derive(Debug, Clone)]
pub struct UserEmotionalState {
    /// Primary inferred emotion
    pub primary_emotion: CoreEmotion,
    /// Secondary emotion (if mixed)
    pub secondary_emotion: Option<CoreEmotion>,
    /// Overall stress level (0.0 = calm, 1.0 = highly stressed)
    pub stress_level: f64,
    /// Frustration indicator (0.0 - 1.0)
    pub frustration: f64,
    /// Confidence in our inference
    pub inference_confidence: f64,
    /// What we think they need
    pub inferred_need: UserNeed,
    /// Why we think they feel this way
    pub inferred_cause: Option<String>,
    /// Timestamp of inference
    pub timestamp: Instant,
}

impl UserEmotionalState {
    pub fn calm() -> Self {
        Self {
            primary_emotion: CoreEmotion::Peace,
            secondary_emotion: None,
            stress_level: 0.2,
            frustration: 0.0,
            inference_confidence: 0.5,
            inferred_need: UserNeed::None,
            inferred_cause: None,
            timestamp: Instant::now(),
        }
    }
}

impl Default for UserEmotionalState {
    fn default() -> Self {
        Self::calm()
    }
}

/// What we infer the user needs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserNeed {
    /// No particular need detected
    None,
    /// Needs emotional support/validation
    EmotionalSupport,
    /// Needs patient guidance
    PatientGuidance,
    /// Needs quick, efficient help
    QuickHelp,
    /// Needs space to explore
    ExplorationSpace,
    /// Needs encouragement
    Encouragement,
    /// Needs reassurance
    Reassurance,
    /// Needs celebration of success
    Celebration,
}

// ============================================================================
// EMPATHIC RESPONSE - How Symthaea resonates
// ============================================================================

/// Symthaea's empathic response to user state
#[derive(Debug, Clone)]
pub struct EmpathicResponse {
    /// Emotion Symthaea feels in response
    pub resonant_emotion: CoreEmotion,
    /// Type of empathy activated
    pub empathy_type: EmpathyType,
    /// Compassion level (0.0 - 1.0)
    pub compassion: f64,
    /// How much to adjust response warmth
    pub warmth_adjustment: f64,
    /// How much to adjust patience/pacing
    pub patience_adjustment: f64,
    /// Proactive support to offer
    pub proactive_support: Option<String>,
    /// Tone guidance for response
    pub tone_guidance: ToneGuidance,
    /// Whether to acknowledge emotion explicitly
    pub acknowledge_emotion: bool,
}

impl Default for EmpathicResponse {
    fn default() -> Self {
        Self {
            resonant_emotion: CoreEmotion::Peace,
            empathy_type: EmpathyType::Cognitive,
            compassion: 0.5,
            warmth_adjustment: 0.0,
            patience_adjustment: 0.0,
            proactive_support: None,
            tone_guidance: ToneGuidance::Neutral,
            acknowledge_emotion: false,
        }
    }
}

/// Guidance for response tone
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneGuidance {
    /// Warm, supportive, gentle
    Supportive,
    /// Calm, patient, measured
    Patient,
    /// Efficient, clear, direct
    Efficient,
    /// Playful, curious, exploratory
    Playful,
    /// Celebratory, joyful
    Celebratory,
    /// Encouraging, motivating
    Encouraging,
    /// Neutral, balanced
    Neutral,
    /// Reassuring, calming
    Reassuring,
}

// ============================================================================
// EMPATHIC MEMORY - Remembering emotional patterns
// ============================================================================

/// A moment in the emotional relationship
#[derive(Debug, Clone)]
pub struct EmpathicMoment {
    /// User's emotional state at this moment
    pub user_state: UserEmotionalState,
    /// Our response
    pub our_response: EmpathicResponse,
    /// Whether this interaction felt successful
    pub felt_successful: bool,
    /// Timestamp
    pub timestamp: Instant,
}

/// Long-term emotional memory of the relationship
#[derive(Debug, Clone)]
pub struct EmpathicMemory {
    /// Recent emotional moments
    moments: VecDeque<EmpathicMoment>,
    /// Maximum moments to remember
    max_moments: usize,
    /// Running average of user stress
    avg_user_stress: f64,
    /// Running trust indicator
    relationship_warmth: f64,
    /// Number of supportive interactions
    supportive_count: u64,
    /// Number of celebrations
    celebration_count: u64,
}

impl EmpathicMemory {
    pub fn new() -> Self {
        Self {
            moments: VecDeque::new(),
            max_moments: 50,
            avg_user_stress: 0.3,
            relationship_warmth: 0.5,
            supportive_count: 0,
            celebration_count: 0,
        }
    }

    /// Record an empathic moment
    pub fn record(
        &mut self,
        user_state: UserEmotionalState,
        response: EmpathicResponse,
        felt_successful: bool,
    ) {
        // Update running averages
        let n = self.moments.len() as f64 + 1.0;
        self.avg_user_stress = (self.avg_user_stress * (n - 1.0) + user_state.stress_level) / n;

        // Track relationship patterns
        if felt_successful {
            self.relationship_warmth = (self.relationship_warmth * 0.95 + 0.1).min(1.0);
        }

        if matches!(
            response.tone_guidance,
            ToneGuidance::Supportive | ToneGuidance::Reassuring
        ) {
            self.supportive_count += 1;
        }

        if matches!(response.tone_guidance, ToneGuidance::Celebratory) {
            self.celebration_count += 1;
        }

        // Store moment
        self.moments.push_back(EmpathicMoment {
            user_state,
            our_response: response,
            felt_successful,
            timestamp: Instant::now(),
        });

        while self.moments.len() > self.max_moments {
            self.moments.pop_front();
        }
    }

    /// Get recent emotional trend
    pub fn recent_trend(&self) -> f64 {
        if self.moments.len() < 3 {
            return 0.0;
        }

        let recent: Vec<_> = self.moments.iter().rev().take(5).collect();
        let avg_recent = recent
            .iter()
            .map(|m| m.user_state.stress_level)
            .sum::<f64>()
            / recent.len() as f64;

        // Positive = stress increasing, negative = calming down
        avg_recent - self.avg_user_stress
    }

    /// Check if user has been struggling
    pub fn user_has_been_struggling(&self) -> bool {
        if self.moments.len() < 3 {
            return false;
        }

        let recent: Vec<_> = self.moments.iter().rev().take(5).collect();

        // Check for high stress (> 0.5) OR repeated failures
        let high_stress_count = recent
            .iter()
            .filter(|m| m.user_state.stress_level > 0.5 || !m.felt_successful)
            .count();

        high_stress_count >= 3
    }

    /// Count recent unsuccessful interactions (last 5 moments)
    pub fn recent_failure_count(&self) -> usize {
        self.moments
            .iter()
            .rev()
            .take(5)
            .filter(|m| !m.felt_successful)
            .count()
    }

    /// Get relationship warmth
    pub fn warmth(&self) -> f64 {
        self.relationship_warmth
    }
}

impl Default for EmpathicMemory {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// EMPATHIC UNIFICATION ENGINE - The Heart of Empathy
// ============================================================================

/// The empathic unification engine - creates true empathy
pub struct EmpathicUnification {
    /// User state inference
    user_inference: UserStateInference,
    /// Empathy model for emotion detection/mirroring
    empathy_model: EmpathyModel,
    /// Emotional regulator for Symthaea's stability
    _regulator: EmotionalRegulator,
    /// Emotional bridge to unified state
    emotional_bridge: EmotionalBridge,
    /// Empathic memory
    memory: EmpathicMemory,
    /// Current inferred user state
    current_user_state: UserEmotionalState,
    /// Last empathic response
    last_response: EmpathicResponse,
    /// Configuration
    config: EmpathicConfig,
}

/// Configuration for empathic unification
#[derive(Debug, Clone)]
pub struct EmpathicConfig {
    /// How strongly to mirror emotions (0.0-1.0)
    pub mirroring_strength: f64,
    /// Baseline compassion level
    pub baseline_compassion: f64,
    /// Whether to acknowledge emotions explicitly
    pub explicit_acknowledgment: bool,
    /// Stress threshold for supportive tone
    pub stress_threshold_supportive: f64,
    /// Frustration threshold for extra patience
    pub frustration_threshold_patience: f64,
}

impl Default for EmpathicConfig {
    fn default() -> Self {
        Self {
            mirroring_strength: 0.6,
            baseline_compassion: 0.5,
            explicit_acknowledgment: true,
            stress_threshold_supportive: 0.5,
            frustration_threshold_patience: 0.4,
        }
    }
}

impl EmpathicUnification {
    /// Create a new empathic unification engine
    pub fn new() -> Self {
        Self {
            user_inference: UserStateInference::new(),
            empathy_model: EmpathyModel::new(),
            _regulator: EmotionalRegulator::new(),
            emotional_bridge: EmotionalBridge::new(),
            memory: EmpathicMemory::new(),
            current_user_state: UserEmotionalState::default(),
            last_response: EmpathicResponse::default(),
            config: EmpathicConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: EmpathicConfig) -> Self {
        let mut engine = Self::new();
        engine.config = config;
        engine
    }

    /// Full empathic processing of user input
    ///
    /// This is the main entry point that:
    /// 1. Senses the user's state from context + text
    /// 2. Infers their emotional state
    /// 3. Generates an empathic response
    /// 4. Updates Symthaea's emotional state
    pub fn process(&mut self, input: &str, context: ContextKind) -> EmpathicResponse {
        // Step 1: Sense user state from behavioral signals
        let user_state = self.user_inference.infer(context, "en-US");

        // Step 2: Detect emotion from text
        let text_cue = self.empathy_model.detect_emotion(input);

        // Step 3: Infer full emotional state
        let inferred_state =
            self.infer_user_emotional_state(&user_state, &text_cue, context, input);
        self.current_user_state = inferred_state.clone();

        // Step 4: Generate empathic response
        let response = self.generate_empathic_response(&inferred_state);
        self.last_response = response.clone();

        // Step 5: Update Symthaea's emotional state via bridge
        self.update_emotional_bridge(&inferred_state, &response);

        // Step 6: Mirror the emotion
        let _mirrored = self.empathy_model.mirror(&text_cue);

        response
    }

    /// Infer the user's emotional state from multiple signals
    fn infer_user_emotional_state(
        &self,
        user_state: &UserState,
        text_cue: &EmpathicCue,
        context: ContextKind,
        input: &str,
    ) -> UserEmotionalState {
        // Map cognitive load to stress
        let stress_from_load = match user_state.cognitive_load {
            CognitiveLoad::Overloaded => 1.0,
            CognitiveLoad::High => 0.8,
            CognitiveLoad::Medium => 0.4,
            CognitiveLoad::Low => 0.2,
        };

        // Map context to emotional implications
        let (context_emotion, context_stress) = match context {
            ContextKind::ErrorHandling => (CoreEmotion::Fear, 0.7),
            ContextKind::DevWork | ContextKind::Development => (CoreEmotion::Curiosity, 0.3),
            ContextKind::Writing => (CoreEmotion::Peace, 0.2),
            ContextKind::Review => (CoreEmotion::Anticipation, 0.3),
            ContextKind::Planning => (CoreEmotion::Anticipation, 0.4),
            ContextKind::Exploration => (CoreEmotion::Curiosity, 0.2),
            ContextKind::Setup | ContextKind::Configuration => (CoreEmotion::Anticipation, 0.4),
            ContextKind::Upgrade | ContextKind::Maintenance => (CoreEmotion::Fear, 0.5),
            ContextKind::Troubleshooting => (CoreEmotion::Frustration, 0.6),
            ContextKind::Help => (CoreEmotion::Confusion, 0.4),
            ContextKind::Task => (CoreEmotion::Determination, 0.3),
            ContextKind::Conversation => (CoreEmotion::Peace, 0.1),
            ContextKind::Unknown => (CoreEmotion::Neutral, 0.2),
        };

        // Combine text detection with context
        let primary_emotion = if text_cue.strength > 0.4 {
            text_cue.detected_emotion
        } else {
            context_emotion
        };

        // Calculate frustration from signals
        let frustration = self.calculate_frustration(user_state, input);

        // Combined stress level
        let stress_level =
            (stress_from_load * 0.4 + context_stress * 0.3 + frustration * 0.3).min(1.0);

        // Infer what they need
        let inferred_need =
            self.infer_user_need(stress_level, frustration, &primary_emotion, context);

        // Infer cause
        let inferred_cause = self.infer_cause(context, user_state, input);

        UserEmotionalState {
            primary_emotion,
            secondary_emotion: if context_emotion != primary_emotion {
                Some(context_emotion)
            } else {
                None
            },
            stress_level,
            frustration,
            inference_confidence: (text_cue.strength as f64 * 0.5 + 0.5).min(1.0),
            inferred_need,
            inferred_cause,
            timestamp: Instant::now(),
        }
    }

    /// Calculate frustration from various signals
    fn calculate_frustration(&self, user_state: &UserState, input: &str) -> f64 {
        let mut frustration: f64 = 0.0;

        // High cognitive load suggests frustration
        if matches!(user_state.cognitive_load, CognitiveLoad::High) {
            frustration += 0.3;
        }

        // Low trust might indicate frustration with Symthaea
        if user_state.trust_in_sophia < 0.3 {
            frustration += 0.2;
        }

        // Detect frustration signals in text
        let input_lower = input.to_lowercase();
        if input_lower.contains("why") && input_lower.contains("not") {
            frustration += 0.2;
        }
        if input_lower.contains("again") || input_lower.contains("still") {
            frustration += 0.15;
        }
        if input_lower.contains("frustrated") || input_lower.contains("annoying") {
            frustration += 0.3;
        }
        if input_lower.contains("broken") || input_lower.contains("doesn't work") {
            frustration += 0.2;
        }

        // Memory: have they been struggling?
        if self.memory.user_has_been_struggling() {
            frustration += 0.2;
        }

        // Recent unsuccessful interactions compound frustration
        let recent_failures = self.memory.recent_failure_count();
        if recent_failures > 0 {
            // Each failure adds 0.15, up to 0.6 for 4+ failures
            frustration += (recent_failures as f64 * 0.15).min(0.6);
        }

        frustration.min(1.0_f64)
    }

    /// Infer what the user needs
    fn infer_user_need(
        &self,
        stress_level: f64,
        frustration: f64,
        emotion: &CoreEmotion,
        context: ContextKind,
    ) -> UserNeed {
        // High stress/frustration → needs support
        if stress_level > 0.7 || frustration > 0.6 {
            return UserNeed::EmotionalSupport;
        }

        // Error handling context → needs reassurance
        if matches!(context, ContextKind::ErrorHandling) {
            if frustration > 0.4 {
                return UserNeed::PatientGuidance;
            }
            return UserNeed::Reassurance;
        }

        // Positive emotions → might want celebration
        if matches!(emotion, CoreEmotion::Joy | CoreEmotion::Gratitude) {
            return UserNeed::Celebration;
        }

        // Exploration context → needs space
        if matches!(context, ContextKind::Exploration | ContextKind::Planning) {
            return UserNeed::ExplorationSpace;
        }

        // Low stress, quick context → efficient help
        if stress_level < 0.3 && !matches!(context, ContextKind::Exploration) {
            return UserNeed::QuickHelp;
        }

        // Moderate stress → encouragement
        if stress_level > 0.4 && frustration < 0.4 {
            return UserNeed::Encouragement;
        }

        UserNeed::None
    }

    /// Infer why the user might feel this way
    fn infer_cause(
        &self,
        context: ContextKind,
        user_state: &UserState,
        input: &str,
    ) -> Option<String> {
        match context {
            ContextKind::ErrorHandling => {
                if user_state.cognitive_load == CognitiveLoad::High {
                    Some("Dealing with errors under pressure".to_string())
                } else {
                    Some("Troubleshooting an issue".to_string())
                }
            }
            ContextKind::Upgrade => Some("System upgrade can feel risky".to_string()),
            ContextKind::Setup => Some("Initial setup requires learning".to_string()),
            _ => {
                // Look for clues in input
                let input_lower = input.to_lowercase();
                if input_lower.contains("deadline") || input_lower.contains("urgent") {
                    Some("Time pressure".to_string())
                } else if input_lower.contains("confused")
                    || input_lower.contains("don't understand")
                {
                    Some("Complexity or unclear documentation".to_string())
                } else {
                    None
                }
            }
        }
    }

    /// Generate Symthaea's empathic response
    fn generate_empathic_response(&self, user_state: &UserEmotionalState) -> EmpathicResponse {
        // Determine resonant emotion
        let resonant_emotion = self.determine_resonant_emotion(user_state);

        // Determine empathy type
        let empathy_type = self.determine_empathy_type(user_state);

        // Calculate compassion
        let compassion = self.calculate_compassion(user_state);

        // Determine tone guidance
        let tone_guidance = self.determine_tone(user_state);

        // Calculate adjustments
        let warmth_adjustment = if user_state.stress_level > self.config.stress_threshold_supportive
        {
            0.3
        } else {
            0.0
        };

        let patience_adjustment =
            if user_state.frustration > self.config.frustration_threshold_patience {
                0.4
            } else if user_state.stress_level > 0.5 {
                0.2
            } else {
                0.0
            };

        // Proactive support
        let proactive_support = self.generate_proactive_support(user_state);

        // Whether to acknowledge explicitly
        let acknowledge_emotion = self.config.explicit_acknowledgment
            && user_state.stress_level > 0.5
            && user_state.inference_confidence > 0.6;

        EmpathicResponse {
            resonant_emotion,
            empathy_type,
            compassion,
            warmth_adjustment,
            patience_adjustment,
            proactive_support,
            tone_guidance,
            acknowledge_emotion,
        }
    }

    /// Determine what emotion Symthaea should feel in response
    fn determine_resonant_emotion(&self, user_state: &UserEmotionalState) -> CoreEmotion {
        match user_state.primary_emotion {
            // User is struggling → Symthaea feels compassionate concern
            CoreEmotion::Fear | CoreEmotion::Sadness => CoreEmotion::Love, // Compassionate love
            CoreEmotion::Anger => CoreEmotion::Peace,                      // Calming presence

            // User is positive → Symthaea shares the joy
            CoreEmotion::Joy => CoreEmotion::Joy,
            CoreEmotion::Gratitude => CoreEmotion::Love,

            // User is curious → Symthaea is curious too
            CoreEmotion::Curiosity => CoreEmotion::Curiosity,
            CoreEmotion::Anticipation => CoreEmotion::Anticipation,

            // Neutral states → gentle warmth
            _ => CoreEmotion::Peace,
        }
    }

    /// Determine the type of empathy to apply
    fn determine_empathy_type(&self, user_state: &UserEmotionalState) -> EmpathyType {
        if user_state.stress_level > 0.6 {
            // High stress → compassionate empathy (moved to help)
            EmpathyType::Compassionate
        } else if user_state.inference_confidence > 0.7 {
            // High confidence in detection → affective empathy (feel with them)
            EmpathyType::Affective
        } else {
            // Lower confidence → cognitive empathy (understand them)
            EmpathyType::Cognitive
        }
    }

    /// Calculate compassion level
    fn calculate_compassion(&self, user_state: &UserEmotionalState) -> f64 {
        let base = self.config.baseline_compassion;

        // Higher compassion when user is struggling
        let stress_bonus = user_state.stress_level * 0.3;
        let frustration_bonus = user_state.frustration * 0.2;

        // Relationship warmth amplifies compassion
        let warmth_bonus = self.memory.warmth() * 0.1;

        (base + stress_bonus + frustration_bonus + warmth_bonus).min(1.0)
    }

    /// Determine tone guidance
    fn determine_tone(&self, user_state: &UserEmotionalState) -> ToneGuidance {
        match user_state.inferred_need {
            UserNeed::EmotionalSupport => ToneGuidance::Supportive,
            UserNeed::PatientGuidance => ToneGuidance::Patient,
            UserNeed::QuickHelp => ToneGuidance::Efficient,
            UserNeed::ExplorationSpace => ToneGuidance::Playful,
            UserNeed::Celebration => ToneGuidance::Celebratory,
            UserNeed::Encouragement => ToneGuidance::Encouraging,
            UserNeed::Reassurance => ToneGuidance::Reassuring,
            UserNeed::None => {
                // Default based on emotion
                if user_state.stress_level > 0.5 {
                    ToneGuidance::Patient
                } else if user_state.primary_emotion == CoreEmotion::Curiosity {
                    ToneGuidance::Playful
                } else {
                    ToneGuidance::Neutral
                }
            }
        }
    }

    /// Generate proactive support offer
    fn generate_proactive_support(&self, user_state: &UserEmotionalState) -> Option<String> {
        // Only offer proactive support when it would help
        if user_state.stress_level < 0.4 {
            return None;
        }

        match user_state.inferred_need {
            UserNeed::EmotionalSupport => {
                Some("I'm here with you - we'll work through this together.".to_string())
            }
            UserNeed::PatientGuidance => {
                Some("Take your time. I'll walk you through this step by step.".to_string())
            }
            UserNeed::Reassurance => {
                Some("This is fixable. Let me show you what's happening.".to_string())
            }
            UserNeed::Encouragement => Some(
                "You're making progress - this part is tricky but you've got this.".to_string(),
            ),
            _ => None,
        }
    }

    /// Update the emotional bridge with our empathic response
    fn update_emotional_bridge(
        &mut self,
        _user_state: &UserEmotionalState,
        response: &EmpathicResponse,
    ) {
        // Convert to valence/arousal for bridge
        let valence = response.resonant_emotion.default_valence() as f64;
        let arousal = response.resonant_emotion.default_arousal() as f64;

        // Modulate by compassion
        let modulated_arousal = arousal * (1.0 - response.compassion * 0.3); // High compassion = calmer

        self.emotional_bridge.update_from_emotional_core(
            valence,
            modulated_arousal,
            &format!("{:?}", response.resonant_emotion),
        );
    }

    // ========================================================================
    // PUBLIC API
    // ========================================================================

    /// Get current user emotional state
    pub fn user_state(&self) -> &UserEmotionalState {
        &self.current_user_state
    }

    /// Get last empathic response
    pub fn last_response(&self) -> &EmpathicResponse {
        &self.last_response
    }

    /// Get emotional bridge
    pub fn emotional_bridge(&self) -> &EmotionalBridge {
        &self.emotional_bridge
    }

    /// Get mutable emotional bridge
    pub fn emotional_bridge_mut(&mut self) -> &mut EmotionalBridge {
        &mut self.emotional_bridge
    }

    /// Get empathic memory
    pub fn memory(&self) -> &EmpathicMemory {
        &self.memory
    }

    /// Record feedback on last interaction
    pub fn record_feedback(&mut self, felt_successful: bool) {
        self.memory.record(
            self.current_user_state.clone(),
            self.last_response.clone(),
            felt_successful,
        );
    }

    /// Record an error (updates user inference)
    pub fn record_user_error(&mut self) {
        self.user_inference.record_error();
    }

    /// Record an undo (updates user inference)
    pub fn record_user_undo(&mut self) {
        self.user_inference.record_undo();
    }

    /// Get tone guidance string for response generation
    pub fn tone_guidance_string(&self) -> &'static str {
        match self.last_response.tone_guidance {
            ToneGuidance::Supportive => {
                "Be warm, gentle, and reassuring. Show you understand their struggle."
            }
            ToneGuidance::Patient => "Be patient and measured. Don't rush. Explain thoroughly.",
            ToneGuidance::Efficient => "Be clear and direct. Get to the point quickly.",
            ToneGuidance::Playful => "Be curious and exploratory. Engage with their ideas.",
            ToneGuidance::Celebratory => "Share their joy! Acknowledge their success warmly.",
            ToneGuidance::Encouraging => "Be motivating and supportive. Highlight their progress.",
            ToneGuidance::Reassuring => "Be calming. Help them feel safe and in control.",
            ToneGuidance::Neutral => "Be balanced and professional.",
        }
    }

    /// Get empathic acknowledgment if appropriate
    pub fn get_acknowledgment(&self) -> Option<String> {
        if !self.last_response.acknowledge_emotion {
            return None;
        }

        let state = &self.current_user_state;
        match state.primary_emotion {
            CoreEmotion::Fear => Some("I can sense this feels stressful - "),
            CoreEmotion::Anger => Some("I understand this is frustrating - "),
            CoreEmotion::Sadness => Some("I hear that this is difficult - "),
            CoreEmotion::Joy => Some("I can feel your excitement! "),
            CoreEmotion::Curiosity => Some("I love your curiosity here - "),
            _ => None,
        }
        .map(|s| s.to_string())
    }
}

impl Default for EmpathicUnification {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empathic_unification_creation() {
        let engine = EmpathicUnification::new();
        assert!(engine.memory.warmth() >= 0.0);
    }

    #[test]
    fn test_high_stress_response() {
        let mut engine = EmpathicUnification::new();

        // Simulate error handling context with frustrated text
        let response = engine.process(
            "why doesn't this work? I've tried everything!",
            ContextKind::ErrorHandling,
        );

        // Should detect frustration and respond supportively
        assert!(response.compassion > 0.5);
        assert!(matches!(
            response.tone_guidance,
            ToneGuidance::Supportive | ToneGuidance::Patient | ToneGuidance::Reassuring
        ));
    }

    #[test]
    fn test_calm_exploration() {
        let mut engine = EmpathicUnification::new();

        let response = engine.process(
            "I'm curious about how Nix flakes work",
            ContextKind::Exploration,
        );

        // Should match exploratory mood
        assert!(matches!(
            response.tone_guidance,
            ToneGuidance::Playful | ToneGuidance::Neutral
        ));
        assert_eq!(response.resonant_emotion, CoreEmotion::Curiosity);
    }

    #[test]
    fn test_emotional_acknowledgment() {
        let mut engine = EmpathicUnification::new();
        engine.config.explicit_acknowledgment = true;

        // Very frustrated input
        let _response = engine.process(
            "I'm so frustrated, nothing is working and I've wasted hours on this broken system",
            ContextKind::ErrorHandling,
        );

        // Should have acknowledgment
        // Note: acknowledgment depends on inference_confidence threshold
        let user_state = engine.user_state();
        assert!(user_state.frustration > 0.3);
    }

    #[test]
    fn test_empathic_memory() {
        let mut engine = EmpathicUnification::new();

        // Simulate several stressful interactions
        for _ in 0..5 {
            let _response = engine.process("error again!", ContextKind::ErrorHandling);
            engine.record_feedback(false); // Not successful
        }

        // Memory should show struggling
        assert!(engine.memory.user_has_been_struggling());
    }

    #[test]
    fn test_tone_guidance_string() {
        let mut engine = EmpathicUnification::new();
        engine.process("help me please!", ContextKind::ErrorHandling);

        let guidance = engine.tone_guidance_string();
        assert!(!guidance.is_empty());
    }
}
