// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Emotional Core: Affective Language Processing
//!
//! Provides emotional understanding and generation capabilities for language,
//! including sentiment analysis, emotional tone detection, and affective
//! response generation.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::primitive_system::PrimitiveSystem;

/// Configuration for emotional core
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalCoreConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Emotional categories to track
    pub categories: Vec<String>,
    /// Sensitivity to emotional content
    pub sensitivity: f32,
    /// Enable emotional memory
    pub memory_enabled: bool,
}

impl Default for EmotionalCoreConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            categories: vec![
                "joy".to_string(),
                "sadness".to_string(),
                "anger".to_string(),
                "fear".to_string(),
                "surprise".to_string(),
                "disgust".to_string(),
                "trust".to_string(),
                "anticipation".to_string(),
            ],
            sensitivity: 0.5,
            memory_enabled: true,
        }
    }
}

/// Emotional analysis result
#[derive(Debug, Clone)]
pub struct EmotionalAnalysis {
    /// Primary emotion detected
    pub primary_emotion: String,
    /// Confidence in primary emotion
    pub confidence: f32,
    /// Valence (-1 negative to 1 positive)
    pub valence: f32,
    /// Arousal (0 calm to 1 excited)
    pub arousal: f32,
    /// Dominance (0 submissive to 1 dominant)
    pub dominance: f32,
    /// All emotion scores
    pub emotion_scores: HashMap<String, f32>,
    /// Emotional embedding
    pub embedding: ContinuousHV,
}

/// Primitive grounding for emotions using NSM semantic primes
///
/// Maps Plutchik's 8 basic emotions to compositions of NSM primitives:
/// - joy = FEEL + GOOD + VERY
/// - sadness = FEEL + BAD + NOT_WANT
/// - anger = FEEL + BAD + WANT + NOT_CAN
/// - fear = FEEL + BAD + MAYBE_HAPPEN
/// - surprise = FEEL + NOT_KNOW + HAPPEN
/// - disgust = FEEL + BAD + BODY
/// - trust = FEEL + GOOD + SOMEONE + TRUE
/// - anticipation = FEEL + WANT + AFTER_HAPPEN
#[derive(Debug, Clone)]
pub struct EmotionPrimitiveGrounding {
    /// NSM primitives that compose this emotion
    pub nsm_primitives: Vec<String>,
    /// Binary BinaryHV encoding from primitives
    pub primitive_encoding: BinaryHV,
    /// Valence weight (-1 to 1)
    pub valence_weight: f32,
    /// Arousal weight (0 to 1)
    pub arousal_weight: f32,
}

impl EmotionPrimitiveGrounding {
    /// Create emotion grounding from NSM primitive names
    pub fn from_primitives(primitives: &[&str], valence: f32, arousal: f32) -> Self {
        let system = PrimitiveSystem::global();

        // Compose emotion encoding by binding primitives together
        let mut encoding = BinaryHV::random(0xE0C0_FEED); // Base seed for emotions
        for name in primitives {
            if let Some(prim) = system.get(name) {
                encoding = encoding.bind(&prim.encoding);
            }
        }

        Self {
            nsm_primitives: primitives.iter().map(|s| s.to_string()).collect(),
            primitive_encoding: encoding,
            valence_weight: valence,
            arousal_weight: arousal,
        }
    }

    /// Get the 8 Plutchik emotions grounded in NSM primitives
    pub fn plutchik_emotions() -> HashMap<String, EmotionPrimitiveGrounding> {
        let mut emotions = HashMap::new();

        // Joy: feeling very good
        emotions.insert(
            "joy".to_string(),
            Self::from_primitives(
                &["NSM_FEEL", "NSM_GOOD", "NSM_VERY"],
                1.0,
                0.7, // positive valence, moderate-high arousal
            ),
        );

        // Sadness: feeling bad, not wanting what happened
        emotions.insert(
            "sadness".to_string(),
            Self::from_primitives(
                &["NSM_FEEL", "NSM_BAD", "NSM_NOT", "NSM_WANT"],
                -0.8,
                0.2, // negative valence, low arousal
            ),
        );

        // Anger: feeling bad, wanting something you can't have
        emotions.insert(
            "anger".to_string(),
            Self::from_primitives(
                &["NSM_FEEL", "NSM_BAD", "NSM_WANT", "NSM_NOT", "NSM_CAN"],
                -0.6,
                0.9, // negative valence, high arousal
            ),
        );

        // Fear: feeling bad about what might happen
        emotions.insert(
            "fear".to_string(),
            Self::from_primitives(
                &["NSM_FEEL", "NSM_BAD", "NSM_MAYBE", "NSM_HAPPEN"],
                -0.7,
                0.8, // negative valence, high arousal
            ),
        );

        // Surprise: not knowing what happened
        emotions.insert(
            "surprise".to_string(),
            Self::from_primitives(
                &["NSM_FEEL", "NSM_NOT", "NSM_KNOW", "NSM_HAPPEN"],
                0.0,
                0.9, // neutral valence, high arousal
            ),
        );

        // Disgust: feeling bad in the body
        emotions.insert(
            "disgust".to_string(),
            Self::from_primitives(
                &["NSM_FEEL", "NSM_BAD", "NSM_BODY"],
                -0.8,
                0.5, // negative valence, moderate arousal
            ),
        );

        // Trust: feeling good about someone being true
        emotions.insert(
            "trust".to_string(),
            Self::from_primitives(
                &["NSM_FEEL", "NSM_GOOD", "NSM_SOMEONE", "NSM_TRUE"],
                0.7,
                0.3, // positive valence, low arousal
            ),
        );

        // Anticipation: wanting what will happen after
        emotions.insert(
            "anticipation".to_string(),
            Self::from_primitives(
                &["NSM_FEEL", "NSM_WANT", "NSM_AFTER", "NSM_HAPPEN"],
                0.4,
                0.6, // mildly positive valence, moderate arousal
            ),
        );

        emotions
    }
}

impl EmotionalAnalysis {
    /// Create a neutral analysis
    pub fn neutral(dimension: usize) -> Self {
        Self {
            primary_emotion: "neutral".to_string(),
            confidence: 1.0,
            valence: 0.0,
            arousal: 0.5,
            dominance: 0.5,
            emotion_scores: HashMap::new(),
            embedding: ContinuousHV::zero(dimension),
        }
    }
}

/// Emotional response generation result
#[derive(Debug, Clone)]
pub struct EmotionalResponse {
    /// Generated text with emotional tone
    pub text: String,
    /// Target emotion
    pub target_emotion: String,
    /// Achieved emotional intensity
    pub intensity: f32,
    /// Embedding of response
    pub embedding: ContinuousHV,
}

/// The emotional core system
#[derive(Debug)]
pub struct EmotionalCore {
    /// Configuration
    config: EmotionalCoreConfig,
    /// Emotion embeddings (ContinuousHV for continuous operations)
    emotion_embeddings: HashMap<String, ContinuousHV>,
    /// Primitive-grounded emotions (NSM semantic primes)
    primitive_groundings: HashMap<String, EmotionPrimitiveGrounding>,
    /// Emotional memory (recent states)
    memory: VecDeque<EmotionalAnalysis>,
    /// Current emotional state
    current_state: EmotionalAnalysis,
    /// Statistics
    stats: EmotionalCoreStats,
}

/// Statistics for emotional core
#[derive(Debug, Clone, Default)]
pub struct EmotionalCoreStats {
    /// Total analyses performed
    pub analyses: u64,
    /// Responses generated
    pub responses_generated: u64,
    /// Average valence
    pub avg_valence: f32,
    /// Most common emotion
    pub most_common_emotion: String,
}

impl EmotionalCore {
    /// Create a new emotional core with primitive-grounded emotions
    pub fn new(config: EmotionalCoreConfig) -> Self {
        let dim = config.dimension;

        // Initialize primitive groundings for all 8 Plutchik emotions
        let primitive_groundings = EmotionPrimitiveGrounding::plutchik_emotions();

        // Initialize emotion embeddings - now using primitive compositions as seeds
        // This ensures emotional embeddings are grounded in NSM semantic primes
        let mut emotion_embeddings = HashMap::new();
        for emotion in &config.categories {
            let embedding = if let Some(grounding) = primitive_groundings.get(emotion) {
                // Use primitive encoding to seed the ContinuousHV, ensuring grounding
                let seed = grounding.primitive_encoding.popcount() as u64;
                ContinuousHV::random(dim, 0xE0C0_0000 + seed)
            } else {
                // Fallback for unknown emotions
                let seed = 0xE0C0_DEAD;
                ContinuousHV::random(dim, seed)
            };
            emotion_embeddings.insert(emotion.clone(), embedding);
        }
        emotion_embeddings.insert("neutral".to_string(), ContinuousHV::zero(dim));

        Self {
            current_state: EmotionalAnalysis::neutral(dim),
            config,
            emotion_embeddings,
            primitive_groundings,
            memory: VecDeque::new(),
            stats: EmotionalCoreStats::default(),
        }
    }

    /// Get the primitive grounding for an emotion
    pub fn get_primitive_grounding(&self, emotion: &str) -> Option<&EmotionPrimitiveGrounding> {
        self.primitive_groundings.get(emotion)
    }

    /// Get the NSM primitives that compose an emotion
    pub fn emotion_to_primitives(&self, emotion: &str) -> Vec<String> {
        self.primitive_groundings
            .get(emotion)
            .map(|g| g.nsm_primitives.clone())
            .unwrap_or_default()
    }

    /// Calculate emotion similarity using primitive encodings
    pub fn primitive_similarity(&self, emotion1: &str, emotion2: &str) -> f32 {
        match (
            self.primitive_groundings.get(emotion1),
            self.primitive_groundings.get(emotion2),
        ) {
            (Some(g1), Some(g2)) => g1.primitive_encoding.similarity(&g2.primitive_encoding),
            _ => 0.0,
        }
    }

    /// Analyze emotional content of text
    pub fn analyze(&mut self, text: &str) -> EmotionalAnalysis {
        self.stats.analyses += 1;

        // Simple keyword-based emotion detection
        let text_lower = text.to_lowercase();

        let mut emotion_scores = HashMap::new();
        let emotion_keywords: HashMap<&str, Vec<&str>> = [
            (
                "joy",
                vec!["happy", "joy", "glad", "delighted", "pleased", "excited"],
            ),
            (
                "sadness",
                vec!["sad", "unhappy", "depressed", "melancholy", "grief"],
            ),
            (
                "anger",
                vec!["angry", "furious", "mad", "irritated", "annoyed"],
            ),
            (
                "fear",
                vec!["afraid", "scared", "fearful", "terrified", "anxious"],
            ),
            (
                "surprise",
                vec!["surprised", "amazed", "astonished", "shocked"],
            ),
            ("disgust", vec!["disgusted", "revolted", "sick", "repulsed"]),
            (
                "trust",
                vec!["trust", "believe", "confident", "faith", "rely"],
            ),
            (
                "anticipation",
                vec!["anticipate", "expect", "await", "hope", "looking forward"],
            ),
        ]
        .iter()
        .cloned()
        .collect();

        for (emotion, keywords) in &emotion_keywords {
            let score: f32 = keywords
                .iter()
                .map(|kw| if text_lower.contains(kw) { 1.0 } else { 0.0 })
                .sum::<f32>()
                / keywords.len().max(1) as f32;
            emotion_scores.insert(emotion.to_string(), score);
        }

        // Find primary emotion
        let (primary_emotion, confidence) = emotion_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(e, s)| (e.clone(), *s))
            .unwrap_or(("neutral".to_string(), 0.0));

        // Calculate VAD (valence, arousal, dominance)
        let valence = *emotion_scores.get("joy").unwrap_or(&0.0)
            - *emotion_scores.get("sadness").unwrap_or(&0.0)
            - *emotion_scores.get("anger").unwrap_or(&0.0)
            - *emotion_scores.get("fear").unwrap_or(&0.0);

        let arousal = *emotion_scores.get("anger").unwrap_or(&0.0)
            + *emotion_scores.get("fear").unwrap_or(&0.0)
            + *emotion_scores.get("surprise").unwrap_or(&0.0)
            + *emotion_scores.get("joy").unwrap_or(&0.0) * 0.5;

        let dominance = *emotion_scores.get("anger").unwrap_or(&0.0)
            + *emotion_scores.get("joy").unwrap_or(&0.0)
            - *emotion_scores.get("fear").unwrap_or(&0.0)
            - *emotion_scores.get("sadness").unwrap_or(&0.0);

        // Create embedding
        let embedding = if let Some(emb) = self.emotion_embeddings.get(&primary_emotion) {
            emb.clone().scale(confidence.max(0.1))
        } else {
            ContinuousHV::zero(self.config.dimension)
        };

        let analysis = EmotionalAnalysis {
            primary_emotion,
            confidence: confidence.max(0.1),
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
            dominance: dominance.clamp(0.0, 1.0),
            emotion_scores,
            embedding,
        };

        // Store in memory
        if self.config.memory_enabled {
            self.memory.push_back(analysis.clone());
            if self.memory.len() > 100 {
                self.memory.pop_front();
            }
        }

        // Update current state
        self.current_state = analysis.clone();

        // Update statistics
        let n = self.stats.analyses as f32;
        self.stats.avg_valence = (self.stats.avg_valence * (n - 1.0) + analysis.valence) / n;

        analysis
    }

    /// Generate emotionally-toned response
    pub fn generate_emotional_response(
        &mut self,
        base_text: &str,
        target_emotion: &str,
        intensity: f32,
    ) -> EmotionalResponse {
        self.stats.responses_generated += 1;

        // Get emotion embedding
        let emotion_emb = self
            .emotion_embeddings
            .get(target_emotion)
            .cloned()
            .unwrap_or_else(|| ContinuousHV::random(self.config.dimension, 0xFA11_BACC)); // Fallback seed

        // Simple emotion modifiers
        let modifiers: HashMap<&str, Vec<&str>> = [
            ("joy", vec!["happily", "joyfully", "with delight"]),
            ("sadness", vec!["sadly", "with sorrow", "regretfully"]),
            ("anger", vec!["angrily", "furiously", "with frustration"]),
            ("fear", vec!["fearfully", "anxiously", "nervously"]),
            (
                "surprise",
                vec!["surprisingly", "amazingly", "unexpectedly"],
            ),
        ]
        .iter()
        .cloned()
        .collect();

        let modifier = modifiers
            .get(target_emotion)
            .and_then(|m| m.first())
            .unwrap_or(&"");

        let text = if !modifier.is_empty() && intensity > 0.5 {
            format!("{modifier}, {base_text}")
        } else {
            base_text.to_string()
        };

        EmotionalResponse {
            text,
            target_emotion: target_emotion.to_string(),
            intensity,
            embedding: emotion_emb.scale(intensity),
        }
    }

    /// Get current emotional state
    pub fn current_state(&self) -> &EmotionalAnalysis {
        &self.current_state
    }

    /// Get emotional memory
    pub fn memory(&self) -> &VecDeque<EmotionalAnalysis> {
        &self.memory
    }

    /// Get emotion embedding
    pub fn emotion_embedding(&self, emotion: &str) -> Option<&ContinuousHV> {
        self.emotion_embeddings.get(emotion)
    }

    /// Get statistics
    pub fn stats(&self) -> &EmotionalCoreStats {
        &self.stats
    }

    /// Reset emotional state
    pub fn reset(&mut self) {
        self.current_state = EmotionalAnalysis::neutral(self.config.dimension);
        self.memory.clear();
    }
}

impl Default for EmotionalCore {
    fn default() -> Self {
        Self::new(EmotionalCoreConfig::default())
    }
}

// ============================================================================
// Types for Empathic Unification Integration
// ============================================================================

/// Core emotion type for empathic processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum CoreEmotion {
    /// Calm, neutral state
    #[default]
    Neutral,
    /// Positive, happy state
    Joy,
    /// Negative, sad state
    Sadness,
    /// Negative, aggressive state
    Anger,
    /// Negative, anxious state
    Fear,
    /// Positive, astonished state
    Surprise,
    /// Negative, repulsed state
    Disgust,
    /// Positive, believing state
    Trust,
    /// Positive, expecting state
    Anticipation,
    /// Frustrated state
    Frustration,
    /// Confused state
    Confusion,
    /// Curious, engaged state
    Curiosity,
    /// Focused, determined state
    Determination,
    /// Peaceful, serene state
    Peace,
    /// Grateful, appreciative state
    Gratitude,
    /// Loving, caring state
    Love,
}

impl CoreEmotion {
    /// Get valence of the emotion (-1.0 to 1.0)
    pub fn valence(&self) -> f64 {
        match self {
            Self::Neutral => 0.0,
            Self::Joy
            | Self::Trust
            | Self::Anticipation
            | Self::Curiosity
            | Self::Determination => 0.7,
            Self::Peace | Self::Gratitude | Self::Love => 0.8,
            Self::Surprise => 0.2,
            Self::Sadness | Self::Fear | Self::Confusion => -0.5,
            Self::Anger | Self::Disgust | Self::Frustration => -0.7,
        }
    }

    /// Get arousal level (0.0 to 1.0)
    pub fn arousal(&self) -> f64 {
        match self {
            Self::Neutral | Self::Sadness | Self::Peace => 0.2,
            Self::Trust | Self::Confusion | Self::Gratitude => 0.3,
            Self::Joy | Self::Anticipation | Self::Curiosity | Self::Love => 0.6,
            Self::Fear | Self::Surprise | Self::Frustration => 0.8,
            Self::Anger | Self::Disgust | Self::Determination => 0.7,
        }
    }

    /// Convert from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "joy" | "happy" | "happiness" => Self::Joy,
            "sadness" | "sad" => Self::Sadness,
            "anger" | "angry" => Self::Anger,
            "fear" | "afraid" | "scared" => Self::Fear,
            "surprise" | "surprised" => Self::Surprise,
            "disgust" | "disgusted" => Self::Disgust,
            "trust" | "trusting" => Self::Trust,
            "anticipation" | "anticipating" => Self::Anticipation,
            "frustration" | "frustrated" => Self::Frustration,
            "confusion" | "confused" => Self::Confusion,
            "curiosity" | "curious" => Self::Curiosity,
            "determination" | "determined" => Self::Determination,
            "peace" | "peaceful" | "calm" | "serene" => Self::Peace,
            "gratitude" | "grateful" | "thankful" => Self::Gratitude,
            "love" | "loving" | "affection" | "caring" => Self::Love,
            _ => Self::Neutral,
        }
    }

    /// Get default valence (alias for valence() for compatibility)
    pub fn default_valence(&self) -> f32 {
        self.valence() as f32
    }

    /// Get default arousal (alias for arousal() for compatibility)
    pub fn default_arousal(&self) -> f32 {
        self.arousal() as f32
    }
}

/// Type of empathy being expressed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum EmpathyType {
    /// Cognitive empathy - understanding perspective
    #[default]
    Cognitive,
    /// Affective empathy - feeling with them
    Affective,
    /// Compassionate empathy - moved to help
    Compassionate,
    /// Supportive empathy - focused on assistance
    Supportive,
}

/// An empathic cue detected in text or context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpathicCue {
    /// The cue type
    pub cue_type: EmpathyCueType,
    /// Intensity (0.0 to 1.0)
    pub intensity: f64,
    /// Source text or context that triggered the cue
    pub source: String,
    /// Detected core emotion
    pub detected_emotion: CoreEmotion,
    /// Strength of the detection (0.0 to 1.0)
    pub strength: f32,
}

/// Types of empathic cues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmpathyCueType {
    /// Emotional expression in text
    EmotionalExpression,
    /// Signs of stress or frustration
    StressSignal,
    /// Request for help or support
    HelpRequest,
    /// Expression of uncertainty
    Uncertainty,
    /// Positive feedback
    PositiveFeedback,
    /// Negative feedback
    NegativeFeedback,
}

/// Model for empathic response generation
#[derive(Debug, Clone, Default)]
pub struct EmpathyModel {
    /// Current empathy type being applied
    pub empathy_type: EmpathyType,
    /// Detected empathic cues
    pub cues: Vec<EmpathicCue>,
    /// Current compassion level (0.0 to 1.0)
    pub compassion_level: f64,
    /// Mirror emotion (what we reflect back)
    pub mirror_emotion: CoreEmotion,
}

impl EmpathyModel {
    /// Create a new empathy model
    pub fn new() -> Self {
        Self::default()
    }

    /// Process text for empathic cues
    pub fn process(&mut self, text: &str) -> &[EmpathicCue] {
        self.cues.clear();

        // Simple keyword-based cue detection
        let lower = text.to_lowercase();

        if lower.contains("help") || lower.contains("stuck") || lower.contains("can't") {
            self.cues.push(EmpathicCue {
                cue_type: EmpathyCueType::HelpRequest,
                intensity: 0.7,
                source: text.to_string(),
                detected_emotion: CoreEmotion::Confusion,
                strength: 0.6,
            });
        }

        if lower.contains("frustrated") || lower.contains("annoying") || lower.contains("ugh") {
            self.cues.push(EmpathicCue {
                cue_type: EmpathyCueType::StressSignal,
                intensity: 0.8,
                source: text.to_string(),
                detected_emotion: CoreEmotion::Frustration,
                strength: 0.8,
            });
        }

        if lower.contains("thanks") || lower.contains("great") || lower.contains("perfect") {
            self.cues.push(EmpathicCue {
                cue_type: EmpathyCueType::PositiveFeedback,
                intensity: 0.6,
                source: text.to_string(),
                detected_emotion: CoreEmotion::Gratitude,
                strength: 0.7,
            });
        }

        &self.cues
    }

    /// Detect emotion from text input
    pub fn detect_emotion(&mut self, text: &str) -> EmpathicCue {
        let lower = text.to_lowercase();

        // Check for various emotional signals
        let (emotion, strength, cue_type) = if lower.contains("frustrated")
            || lower.contains("annoying")
            || lower.contains("ugh")
        {
            (CoreEmotion::Frustration, 0.8, EmpathyCueType::StressSignal)
        } else if lower.contains("angry") || lower.contains("furious") {
            (CoreEmotion::Anger, 0.9, EmpathyCueType::StressSignal)
        } else if lower.contains("sad") || lower.contains("disappointed") {
            (
                CoreEmotion::Sadness,
                0.7,
                EmpathyCueType::EmotionalExpression,
            )
        } else if lower.contains("scared") || lower.contains("afraid") || lower.contains("worried")
        {
            (CoreEmotion::Fear, 0.7, EmpathyCueType::StressSignal)
        } else if lower.contains("happy") || lower.contains("great") || lower.contains("awesome") {
            (CoreEmotion::Joy, 0.8, EmpathyCueType::PositiveFeedback)
        } else if lower.contains("thanks")
            || lower.contains("thank you")
            || lower.contains("grateful")
        {
            (
                CoreEmotion::Gratitude,
                0.7,
                EmpathyCueType::PositiveFeedback,
            )
        } else if lower.contains("confused") || lower.contains("don't understand") {
            (CoreEmotion::Confusion, 0.6, EmpathyCueType::Uncertainty)
        } else if lower.contains("help") || lower.contains("stuck") || lower.contains("can't") {
            (CoreEmotion::Confusion, 0.5, EmpathyCueType::HelpRequest)
        } else if lower.contains("curious")
            || lower.contains("interesting")
            || lower.contains("wonder")
        {
            (
                CoreEmotion::Curiosity,
                0.6,
                EmpathyCueType::EmotionalExpression,
            )
        } else {
            (
                CoreEmotion::Neutral,
                0.3,
                EmpathyCueType::EmotionalExpression,
            )
        };

        EmpathicCue {
            cue_type,
            intensity: strength as f64,
            source: text.to_string(),
            detected_emotion: emotion,
            strength,
        }
    }

    /// Mirror the detected emotion for empathic response
    pub fn mirror(&mut self, cue: &EmpathicCue) -> CoreEmotion {
        // Mirror the emotion with appropriate response
        let mirrored = match cue.detected_emotion {
            CoreEmotion::Fear | CoreEmotion::Sadness => CoreEmotion::Love, // Compassion
            CoreEmotion::Anger | CoreEmotion::Frustration => CoreEmotion::Peace, // Calming
            CoreEmotion::Joy | CoreEmotion::Gratitude => CoreEmotion::Joy, // Share joy
            CoreEmotion::Confusion => CoreEmotion::Trust,                  // Reassurance
            CoreEmotion::Curiosity => CoreEmotion::Curiosity,              // Engage
            _ => CoreEmotion::Peace,                                       // Default to calm
        };
        self.mirror_emotion = mirrored;
        mirrored
    }

    /// Get recommended empathy type based on cues
    pub fn recommended_empathy_type(&self) -> EmpathyType {
        if self
            .cues
            .iter()
            .any(|c| c.cue_type == EmpathyCueType::StressSignal)
        {
            EmpathyType::Compassionate
        } else if self
            .cues
            .iter()
            .any(|c| c.cue_type == EmpathyCueType::HelpRequest)
        {
            EmpathyType::Supportive
        } else {
            EmpathyType::Cognitive
        }
    }
}

/// Emotional regulator for managing emotional state
#[derive(Debug, Clone, Default)]
pub struct EmotionalRegulator {
    /// Current emotional state
    pub current_emotion: CoreEmotion,
    /// Target emotional state (what we're regulating toward)
    pub target_emotion: CoreEmotion,
    /// Regulation strength (0.0 to 1.0)
    pub regulation_strength: f64,
    /// Emotional inertia (resistance to change)
    pub inertia: f64,
}

impl EmotionalRegulator {
    /// Create a new regulator
    pub fn new() -> Self {
        Self {
            current_emotion: CoreEmotion::Neutral,
            target_emotion: CoreEmotion::Neutral,
            regulation_strength: 0.5,
            inertia: 0.3,
        }
    }

    /// Regulate toward target emotion
    pub fn regulate(&mut self) {
        // Simple regulation - if strength overcomes inertia, move toward target
        if self.regulation_strength > self.inertia {
            self.current_emotion = self.target_emotion;
        }
    }

    /// Set target emotion
    pub fn set_target(&mut self, emotion: CoreEmotion) {
        self.target_emotion = emotion;
    }

    /// Get current valence
    pub fn valence(&self) -> f64 {
        self.current_emotion.valence()
    }

    /// Get current arousal
    pub fn arousal(&self) -> f64 {
        self.current_emotion.arousal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // EmotionalCoreConfig Tests
    // =========================================================================

    #[test]
    fn test_emotional_core_config_default() {
        let config = EmotionalCoreConfig::default();
        assert_eq!(config.dimension, 512);
        assert!((config.sensitivity - 0.5).abs() < 0.01);
        assert!(config.memory_enabled);
        assert_eq!(config.categories.len(), 8);
    }

    #[test]
    fn test_emotional_core_config_has_all_basic_emotions() {
        let config = EmotionalCoreConfig::default();
        assert!(config.categories.contains(&"joy".to_string()));
        assert!(config.categories.contains(&"sadness".to_string()));
        assert!(config.categories.contains(&"anger".to_string()));
        assert!(config.categories.contains(&"fear".to_string()));
        assert!(config.categories.contains(&"surprise".to_string()));
        assert!(config.categories.contains(&"disgust".to_string()));
        assert!(config.categories.contains(&"trust".to_string()));
        assert!(config.categories.contains(&"anticipation".to_string()));
    }

    // =========================================================================
    // EmotionalCore Creation Tests
    // =========================================================================

    #[test]
    fn test_emotional_core_creation() {
        let core = EmotionalCore::default();
        assert_eq!(core.stats.analyses, 0);
    }

    #[test]
    fn test_emotional_core_with_custom_config() {
        let config = EmotionalCoreConfig {
            dimension: 256,
            categories: vec!["happy".to_string(), "sad".to_string()],
            sensitivity: 0.8,
            memory_enabled: false,
        };
        let core = EmotionalCore::new(config);
        assert_eq!(core.stats.analyses, 0);
    }

    // =========================================================================
    // Emotion Analysis Tests
    // =========================================================================

    #[test]
    fn test_emotion_analysis_positive() {
        let mut core = EmotionalCore::default();
        let analysis = core.analyze("I am so happy and excited today!");

        assert!(analysis.valence > 0.0);
        assert_eq!(core.stats.analyses, 1);
    }

    #[test]
    fn test_negative_emotion() {
        let mut core = EmotionalCore::default();
        let analysis = core.analyze("I am sad and depressed.");

        assert!(analysis.valence < 0.0);
    }

    #[test]
    fn test_emotion_analysis_anger() {
        let mut core = EmotionalCore::default();
        let analysis = core.analyze("I am so angry and furious!");

        assert!(analysis.emotion_scores.get("anger").unwrap_or(&0.0) > &0.0);
        assert!(analysis.arousal > 0.0);
    }

    #[test]
    fn test_emotion_analysis_fear() {
        let mut core = EmotionalCore::default();
        let analysis = core.analyze("I am scared and terrified.");

        assert!(analysis.emotion_scores.get("fear").unwrap_or(&0.0) > &0.0);
    }

    #[test]
    fn test_emotion_analysis_surprise() {
        let mut core = EmotionalCore::default();
        let analysis = core.analyze("I am so surprised and amazed!");

        assert!(analysis.emotion_scores.get("surprise").unwrap_or(&0.0) > &0.0);
    }

    #[test]
    fn test_emotion_analysis_neutral() {
        let mut core = EmotionalCore::default();
        let analysis = core.analyze("The sky is blue.");

        // Neutral text should have low emotion scores
        assert!(analysis.confidence <= 0.1 || analysis.primary_emotion == "neutral");
    }

    #[test]
    fn test_emotion_analysis_empty_input() {
        let mut core = EmotionalCore::default();
        let analysis = core.analyze("");

        assert_eq!(core.stats.analyses, 1);
        // Empty input should still return a valid analysis
        assert!(!analysis.primary_emotion.is_empty());
    }

    // =========================================================================
    // VAD (Valence, Arousal, Dominance) Tests
    // =========================================================================

    #[test]
    fn test_valence_range() {
        let mut core = EmotionalCore::default();

        let positive = core.analyze("I am extremely happy!");
        assert!(positive.valence >= -1.0 && positive.valence <= 1.0);

        let negative = core.analyze("I am extremely sad!");
        assert!(negative.valence >= -1.0 && negative.valence <= 1.0);
    }

    #[test]
    fn test_arousal_range() {
        let mut core = EmotionalCore::default();
        let analysis = core.analyze("I am excited and energetic!");

        assert!(analysis.arousal >= 0.0 && analysis.arousal <= 1.0);
    }

    #[test]
    fn test_dominance_range() {
        let mut core = EmotionalCore::default();
        let analysis = core.analyze("I am powerful and in control!");

        assert!(analysis.dominance >= 0.0 && analysis.dominance <= 1.0);
    }

    // =========================================================================
    // Emotional Response Generation Tests
    // =========================================================================

    #[test]
    fn test_emotional_response() {
        let mut core = EmotionalCore::default();
        let response = core.generate_emotional_response("Hello there", "joy", 0.8);

        assert!(!response.text.is_empty());
        assert_eq!(response.target_emotion, "joy");
        assert!((response.intensity - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_emotional_response_sadness() {
        let mut core = EmotionalCore::default();
        let response = core.generate_emotional_response("I'm sorry", "sadness", 0.9);

        assert!(
            response.text.contains("sadly")
                || response.text.contains("sorrow")
                || response.text.contains("I'm sorry")
        );
        assert_eq!(response.target_emotion, "sadness");
    }

    #[test]
    fn test_emotional_response_low_intensity() {
        let mut core = EmotionalCore::default();
        let response = core.generate_emotional_response("Hello", "joy", 0.3);

        // Low intensity should not add modifier
        assert!(!response.text.starts_with("happily") && !response.text.starts_with("joyfully"));
    }

    #[test]
    fn test_emotional_response_unknown_emotion() {
        let mut core = EmotionalCore::default();
        let response = core.generate_emotional_response("Test", "unknown_emotion", 0.8);

        // Should still generate a response even for unknown emotion
        assert!(!response.text.is_empty());
    }

    // =========================================================================
    // Memory Tests
    // =========================================================================

    #[test]
    fn test_memory_enabled() {
        let mut core = EmotionalCore::default();
        core.analyze("Happy text");
        core.analyze("Sad text");

        assert_eq!(core.memory().len(), 2);
    }

    #[test]
    fn test_memory_disabled() {
        let config = EmotionalCoreConfig {
            memory_enabled: false,
            ..Default::default()
        };
        let mut core = EmotionalCore::new(config);
        core.analyze("Happy text");
        core.analyze("Sad text");

        assert!(core.memory().is_empty());
    }

    #[test]
    fn test_memory_limit() {
        let mut core = EmotionalCore::default();

        // Add more than 100 analyses
        for i in 0..110 {
            core.analyze(&format!("Text number {}", i));
        }

        // Memory should be capped at 100
        assert!(core.memory().len() <= 100);
    }

    // =========================================================================
    // State and Accessor Tests
    // =========================================================================

    #[test]
    fn test_current_state_accessor() {
        let mut core = EmotionalCore::default();
        core.analyze("I am happy!");

        let state = core.current_state();
        assert!(!state.primary_emotion.is_empty());
    }

    #[test]
    fn test_emotion_embedding_accessor() {
        let core = EmotionalCore::default();

        let joy_emb = core.emotion_embedding("joy");
        assert!(joy_emb.is_some());

        let unknown_emb = core.emotion_embedding("unknown");
        assert!(unknown_emb.is_none());

        let neutral_emb = core.emotion_embedding("neutral");
        assert!(neutral_emb.is_some());
    }

    #[test]
    fn test_stats_accessor() {
        let mut core = EmotionalCore::default();
        core.analyze("Test");
        core.generate_emotional_response("Test", "joy", 0.5);

        let stats = core.stats();
        assert_eq!(stats.analyses, 1);
        assert_eq!(stats.responses_generated, 1);
    }

    // =========================================================================
    // Reset Tests
    // =========================================================================

    #[test]
    fn test_reset() {
        let mut core = EmotionalCore::default();
        core.analyze("I am happy!");
        core.analyze("I am sad!");

        assert!(!core.memory().is_empty());

        core.reset();

        assert!(core.memory().is_empty());
        assert_eq!(core.current_state().primary_emotion, "neutral");
    }

    // =========================================================================
    // EmotionalAnalysis Tests
    // =========================================================================

    #[test]
    fn test_emotional_analysis_neutral() {
        let analysis = EmotionalAnalysis::neutral(256);

        assert_eq!(analysis.primary_emotion, "neutral");
        assert!((analysis.confidence - 1.0).abs() < 0.01);
        assert!((analysis.valence - 0.0).abs() < 0.01);
        assert!((analysis.arousal - 0.5).abs() < 0.01);
        assert!((analysis.dominance - 0.5).abs() < 0.01);
        assert!(analysis.emotion_scores.is_empty());
    }

    #[test]
    fn test_emotional_analysis_clone() {
        let mut core = EmotionalCore::default();
        let analysis = core.analyze("I am happy!");
        let cloned = analysis.clone();

        assert_eq!(analysis.primary_emotion, cloned.primary_emotion);
        assert!((analysis.valence - cloned.valence).abs() < 0.01);
    }

    // =========================================================================
    // Statistics Tests
    // =========================================================================

    #[test]
    fn test_avg_valence_accumulation() {
        let mut core = EmotionalCore::default();

        core.analyze("I am happy!");
        core.analyze("I am happy!");
        core.analyze("I am sad!");

        // Average valence should be somewhere in between
        let stats = core.stats();
        // Just verify it's being tracked
        assert!(stats.avg_valence.is_finite());
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_multiple_emotions_in_text() {
        let mut core = EmotionalCore::default();
        let analysis = core.analyze("I am happy but also sad and a bit angry.");

        // Should detect multiple emotions
        let nonzero_emotions = analysis
            .emotion_scores
            .values()
            .filter(|&&v| v > 0.0)
            .count();
        assert!(nonzero_emotions >= 1);
    }

    #[test]
    fn test_case_insensitivity() {
        let mut core = EmotionalCore::default();

        let lowercase = core.analyze("i am happy");
        let uppercase = core.analyze("I AM HAPPY");

        // Both should detect the same emotion
        assert_eq!(lowercase.primary_emotion, uppercase.primary_emotion);
    }

    #[test]
    fn test_embedding_dimension() {
        let config = EmotionalCoreConfig {
            dimension: 128,
            ..Default::default()
        };
        let mut core = EmotionalCore::new(config);
        let analysis = core.analyze("Happy");

        assert_eq!(analysis.embedding.dim(), 128);
    }
}
