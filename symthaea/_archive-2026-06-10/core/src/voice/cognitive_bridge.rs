// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cognitive Voice Bridge
//!
//! Connects the cognitive loop (CfC predictions) to voice synthesis,
//! enabling consciousness-driven speech with:
//!
//! - **CfC state → Pacing**: Neural dynamics drive speech rhythm
//! - **Prediction error → Confidence**: Uncertainty affects articulation
//! - **Attention → Emphasis**: Focus on concepts affects phoneme stress
//! - **Semantics → Prosody**: Meaning shapes intonation patterns
//!
//! ## Architecture
//!
//! ```text
//! CognitiveLoopService::cycle()
//!         │
//!         ├─ output: Vec<f32>         ──► LTCPacing (rate, tau, arousal)
//!         ├─ prediction_error: f32     ──► Confidence (crisp vs hesitant)
//!         ├─ attention_state: HashMap  ──► PhonemeEmphasis (stress mapping)
//!         └─ detected_primitives: Vec  ──► SemanticProsody (intonation)
//!         │
//!         ▼
//! CognitivePacing → ArticulatorySynthesizer
//! ```
//!
//! ## Semantic → Prosody Mapping
//!
//! Different semantic categories drive different prosodic patterns:
//!
//! | Category    | Pitch    | Rate    | Example Concepts            |
//! |-------------|----------|---------|----------------------------|
//! | Cause       | Rising   | Slower  | "because", "leads to"      |
//! | Effect      | Falling  | Normal  | "therefore", "results in"  |
//! | Question    | Rising   | Slower  | "why", "how", "what"       |
//! | Emphasis    | Higher   | Slower  | "important", "critical"    |
//! | Uncertainty | Falling  | Slower  | "maybe", "perhaps"         |
//! | Contrast    | Peak     | Pause   | "but", "however"           |

use crate::voice::LTCPacing;
use std::collections::{HashMap, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 16 CONSCIOUSNESS SIGNALS FOR VOICE
// ═══════════════════════════════════════════════════════════════════════════════

/// Compact view of Phase 16 consciousness signals relevant to voice prosody.
///
/// Extracted from `CycleMetadata` by the `voice_consciousness_signals()` accessor
/// on `CognitiveLoopService`. Provides the fields needed for consciousness-driven
/// prosody modulation without exposing the full 150+ field CycleMetadata.
#[derive(Debug, Clone, Copy)]
pub struct VoiceConsciousnessSignals {
    /// Unified quality score (0.0–1.0). Fusion of prediction coherence +
    /// agreement + anomaly detection.
    pub unified_quality: f32,

    /// Epistemic gate confidence (0.0–1.0, 0.5 when gate is off).
    pub epistemic_confidence: f32,

    /// Whether the dissipative health gate dampened learning this cycle.
    pub dissipative_gated: bool,

    /// Dissipative health factor (0.0–1.0). Lower = more dissipative stress.
    pub dissipative_factor: f32,

    /// Coherence velocity: rate of change of coherence (negative = dropping).
    pub coherence_velocity: f32,

    /// Cross-module agreement (0.0–1.0). Alignment between FEP, MCTS,
    /// resonator confidence, and moral judgment.
    pub cross_module_agreement: f32,

    /// Master consciousness level (0.0–1.0).
    pub consciousness_level: f64,

    /// Client distress (0=calm, 1=crisis). Used for therapeutic voice modulation.
    #[cfg(feature = "therapeutic")]
    pub client_distress: f32,

    /// Therapeutic alliance quality (0=broken, 1=strong).
    #[cfg(feature = "therapeutic")]
    pub alliance_quality: f32,

    /// Active therapeutic intent (0-7).
    #[cfg(feature = "therapeutic")]
    pub therapeutic_intent: f32,
}

impl Default for VoiceConsciousnessSignals {
    fn default() -> Self {
        Self {
            unified_quality: 1.0,
            epistemic_confidence: 0.5,
            dissipative_gated: false,
            dissipative_factor: 1.0,
            coherence_velocity: 0.0,
            cross_module_agreement: 1.0,
            consciousness_level: 0.5,
            #[cfg(feature = "therapeutic")]
            client_distress: 0.0,
            #[cfg(feature = "therapeutic")]
            alliance_quality: 0.5,
            #[cfg(feature = "therapeutic")]
            therapeutic_intent: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEMANTIC PROSODY MAPPING
// ═══════════════════════════════════════════════════════════════════════════════

/// Semantic category that affects prosody
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticCategory {
    /// Causal concepts: "because", "leads to", "causes"
    Cause,
    /// Effect concepts: "therefore", "results in", "consequently"
    Effect,
    /// Question concepts: "why", "how", "what", "when"
    Question,
    /// Emphasis concepts: "important", "critical", "key"
    Emphasis,
    /// Uncertainty concepts: "maybe", "perhaps", "possibly"
    Uncertainty,
    /// Contrast concepts: "but", "however", "although"
    Contrast,
    /// Sequence concepts: "first", "then", "finally"
    Sequence,
    /// Neutral (no special prosody)
    Neutral,
}

impl SemanticCategory {
    /// Detect category from a concept/primitive string
    pub fn from_concept(concept: &str) -> Self {
        let lower = concept.to_lowercase();

        // Contrast indicators (check BEFORE Question because "however" contains "how")
        if lower.contains("however")
            || lower.contains("although")
            || lower == "but"
            || lower.contains("contrast")
            || lower.contains("instead")
            || lower == "yet"
        {
            return Self::Contrast;
        }

        // Cause indicators
        if lower.contains("cause")
            || lower.contains("because")
            || lower.contains("leads")
            || lower.contains("since")
            || lower.contains("due to")
            || lower.contains("reason")
        {
            return Self::Cause;
        }

        // Effect indicators
        if lower.contains("effect")
            || lower.contains("therefore")
            || lower.contains("result")
            || lower.contains("consequently")
            || lower.contains("thus")
            || lower.contains("hence")
        {
            return Self::Effect;
        }

        // Question indicators
        if lower.contains("why")
            || lower == "how"
            || lower.contains("what")
            || lower.contains("when")
            || lower.contains("where")
            || lower.contains("who")
            || lower.contains("?")
        {
            return Self::Question;
        }

        // Emphasis indicators
        if lower.contains("important")
            || lower.contains("critical")
            || lower.contains("key")
            || lower.contains("essential")
            || lower.contains("crucial")
            || lower.contains("vital")
        {
            return Self::Emphasis;
        }

        // Uncertainty indicators
        if lower.contains("maybe")
            || lower.contains("perhaps")
            || lower.contains("possibly")
            || lower.contains("might")
            || lower.contains("uncertain")
            || lower.contains("unclear")
        {
            return Self::Uncertainty;
        }

        // Sequence indicators
        if lower.contains("first")
            || lower.contains("then")
            || lower.contains("next")
            || lower.contains("finally")
            || lower.contains("second")
            || lower.contains("last")
        {
            return Self::Sequence;
        }

        Self::Neutral
    }
}

/// Prosodic parameters derived from semantic content
#[derive(Debug, Clone, Default)]
pub struct SemanticProsody {
    /// Pitch contour modifier (-1.0 = falling, 0.0 = neutral, 1.0 = rising)
    pub pitch_contour: f32,

    /// Rate modifier (0.5 = half speed, 1.0 = normal, 1.5 = faster)
    pub rate_modifier: f32,

    /// Pre-phrase pause multiplier
    pub pre_pause: f32,

    /// Post-phrase pause multiplier
    pub post_pause: f32,

    /// Pitch range expansion (1.0 = normal, 2.0 = more expressive)
    pub pitch_range: f32,

    /// Volume/energy modifier
    pub energy_modifier: f32,

    /// Detected semantic categories
    pub categories: Vec<SemanticCategory>,
}

impl SemanticProsody {
    /// Create prosody from detected semantic primitives
    pub fn from_primitives(primitives: &[String]) -> Self {
        let categories: Vec<SemanticCategory> = primitives
            .iter()
            .map(|p| SemanticCategory::from_concept(p))
            .filter(|c| *c != SemanticCategory::Neutral)
            .collect();

        if categories.is_empty() {
            return Self::neutral();
        }

        // Combine prosodic effects from all detected categories
        let mut pitch_contour: f32 = 0.0;
        let mut rate_modifier: f32 = 1.0;
        let mut pre_pause: f32 = 1.0;
        let mut post_pause: f32 = 1.0;
        let mut pitch_range: f32 = 1.0;
        let mut energy_modifier: f32 = 1.0;

        for category in &categories {
            let (pc, rm, pre, post, pr, em) = Self::category_prosody(*category);
            pitch_contour += pc;
            rate_modifier *= rm;
            pre_pause = pre_pause.max(pre);
            post_pause = post_pause.max(post);
            pitch_range = pitch_range.max(pr);
            energy_modifier *= em;
        }

        // Normalize by number of categories
        let n = categories.len() as f32;
        pitch_contour /= n;

        Self {
            pitch_contour: pitch_contour.clamp(-1.0, 1.0),
            rate_modifier: rate_modifier.clamp(0.5, 1.5),
            pre_pause,
            post_pause,
            pitch_range,
            energy_modifier: energy_modifier.clamp(0.7, 1.5),
            categories,
        }
    }

    /// Get prosodic parameters for a semantic category
    fn category_prosody(category: SemanticCategory) -> (f32, f32, f32, f32, f32, f32) {
        // Returns: (pitch_contour, rate_modifier, pre_pause, post_pause, pitch_range, energy)
        match category {
            SemanticCategory::Cause => (0.3, 0.9, 1.0, 1.2, 1.1, 1.0), // Rising, slower, pause after
            SemanticCategory::Effect => (-0.3, 1.0, 1.2, 1.0, 1.0, 1.0), // Falling, pause before
            SemanticCategory::Question => (0.5, 0.85, 1.0, 1.3, 1.3, 1.0), // Rising, slower, expressive
            SemanticCategory::Emphasis => (0.2, 0.8, 1.1, 1.1, 1.2, 1.2),  // Higher, slower, louder
            SemanticCategory::Uncertainty => (-0.2, 0.85, 1.0, 1.2, 0.9, 0.9), // Falling, quieter
            SemanticCategory::Contrast => (0.0, 0.9, 1.5, 1.3, 1.4, 1.1),  // Pause, expressive
            SemanticCategory::Sequence => (0.1, 1.0, 1.2, 1.0, 1.0, 1.0), // Slight rise, pause before
            SemanticCategory::Neutral => (0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        }
    }

    /// Create neutral prosody
    pub fn neutral() -> Self {
        Self {
            pitch_contour: 0.0,
            rate_modifier: 1.0,
            pre_pause: 1.0,
            post_pause: 1.0,
            pitch_range: 1.0,
            energy_modifier: 1.0,
            categories: vec![],
        }
    }

    /// Check if any semantic content was detected
    pub fn has_semantic_content(&self) -> bool {
        !self.categories.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COGNITIVE PACING (Extended with Semantic Prosody)
// ═══════════════════════════════════════════════════════════════════════════════

/// Extended pacing with cognitive-loop derived parameters
#[derive(Debug, Clone)]
pub struct CognitivePacing {
    /// Base LTC pacing from neural state
    pub base_pacing: LTCPacing,

    /// Confidence level (0.0 = uncertain, 1.0 = confident)
    /// Derived from prediction error: confidence = 1 - error
    pub confidence: f32,

    /// Per-word emphasis based on attention weights
    /// Maps concept → emphasis multiplier
    pub word_emphasis: HashMap<String, f32>,

    /// Detected semantic primitives that should be emphasized
    pub highlighted_concepts: Vec<String>,

    /// Semantic prosody derived from detected primitives
    pub semantic_prosody: SemanticProsody,

    // ── Phase 16 consciousness signals ──────────────────────────────
    /// Unified quality score (fusion of prediction coherence + agreement + anomaly).
    /// 0.0 = low quality, 1.0 = high quality.
    pub unified_quality: f32,

    /// Epistemic gate confidence (0.0–1.0, 0.5 when gate is off).
    pub epistemic_confidence: f32,

    /// Whether the dissipative health gate dampened learning this cycle.
    pub dissipative_gated: bool,

    /// Dissipative health factor (0.0–1.0). Lower = more dissipative stress.
    pub dissipative_factor: f32,

    /// Coherence velocity: rate of change of coherence (negative = dropping).
    pub coherence_velocity: f32,

    /// Cross-module agreement (0.0–1.0). Alignment between FEP, MCTS,
    /// resonator confidence, and moral judgment.
    pub cross_module_agreement: f32,

    /// Master consciousness level (0.0–1.0).
    pub consciousness_level: f64,

    // ── Therapeutic voice modulation (feature-gated) ──────────────
    /// Client distress level (0=calm, 1=crisis). Higher distress → slower rate,
    /// longer pauses, softer energy. Science: Safran & Muran (2000) — therapist
    /// prosody attunement predicts alliance quality.
    #[cfg(feature = "therapeutic")]
    pub client_distress: f32,

    /// Therapeutic alliance quality (0=broken, 1=strong). Low alliance → warmer
    /// tone, reduced emphasis (less assertive). High alliance → permits wider
    /// dynamic range. Science: Bordin (1979), Flückiger et al. (2018) meta-analysis.
    #[cfg(feature = "therapeutic")]
    pub alliance_quality: f32,

    /// Active therapeutic intent (0-7). Affects prosodic coloring:
    /// - 0/6 (validate): warmer, softer, slower
    /// - 2 (reflect): rising pitch, question-like intonation
    /// - 3/5 (challenge/ground): firmer emphasis, steadier rate
    /// - 7 (crisis): calm, slow, grounding tone
    #[cfg(feature = "therapeutic")]
    pub therapeutic_intent: f32,
}

impl CognitivePacing {
    /// Create from cognitive loop output
    pub fn from_cognitive_loop(
        cfc_output: &[f32],
        tau: f32,
        prediction_error: f32,
        attention_state: HashMap<String, f32>,
        detected_primitives: Vec<String>,
    ) -> Self {
        // Base pacing from CfC state
        let base_pacing = LTCPacing::from_ltc_state(cfc_output, tau);

        // Confidence from prediction error (inverse)
        // High error = low confidence = hesitant speech
        let confidence = (1.0 - prediction_error.min(1.0)).max(0.0);

        // Convert attention weights to emphasis multipliers
        // Normalize attention to 0.5-1.5 range
        let max_attention = attention_state
            .values()
            .cloned()
            .fold(0.0f32, f32::max)
            .max(0.001);

        let word_emphasis: HashMap<String, f32> = attention_state
            .into_iter()
            .map(|(concept, attention)| {
                // Scale attention to emphasis multiplier
                let normalized = attention / max_attention;
                let emphasis = 0.5 + normalized; // 0.5 to 1.5
                (concept, emphasis)
            })
            .collect();

        // Compute semantic prosody from detected primitives
        let semantic_prosody = SemanticProsody::from_primitives(&detected_primitives);

        Self {
            base_pacing,
            confidence,
            word_emphasis,
            highlighted_concepts: detected_primitives,
            semantic_prosody,
            unified_quality: 1.0,
            epistemic_confidence: 0.5,
            dissipative_gated: false,
            dissipative_factor: 1.0,
            coherence_velocity: 0.0,
            cross_module_agreement: 1.0,
            consciousness_level: 0.5,
            #[cfg(feature = "therapeutic")]
            client_distress: 0.0,
            #[cfg(feature = "therapeutic")]
            alliance_quality: 0.5,
            #[cfg(feature = "therapeutic")]
            therapeutic_intent: 0.0,
        }
    }

    /// Create from cognitive loop output with Phase 16 consciousness signals.
    ///
    /// This is the preferred constructor when CycleMetadata is available,
    /// as it incorporates unified quality, epistemic gating, dissipative health,
    /// and coherence velocity into prosody modulation.
    pub fn from_cycle_metadata(
        cfc_output: &[f32],
        tau: f32,
        prediction_error: f32,
        attention_state: HashMap<String, f32>,
        detected_primitives: Vec<String>,
        signals: &VoiceConsciousnessSignals,
    ) -> Self {
        let mut pacing = Self::from_cognitive_loop(
            cfc_output,
            tau,
            prediction_error,
            attention_state,
            detected_primitives,
        );
        pacing.unified_quality = signals.unified_quality;
        pacing.epistemic_confidence = signals.epistemic_confidence;
        pacing.dissipative_gated = signals.dissipative_gated;
        pacing.dissipative_factor = signals.dissipative_factor;
        pacing.coherence_velocity = signals.coherence_velocity;
        pacing.cross_module_agreement = signals.cross_module_agreement;
        pacing.consciousness_level = signals.consciousness_level;
        #[cfg(feature = "therapeutic")]
        {
            pacing.client_distress = signals.client_distress;
            pacing.alliance_quality = signals.alliance_quality;
            pacing.therapeutic_intent = signals.therapeutic_intent;
        }
        pacing
    }

    /// Get effective pacing with confidence, semantic, and Phase 16 consciousness modulation
    pub fn effective_pacing(&self) -> LTCPacing {
        let mut pacing = self.base_pacing.clone();

        // === Confidence Modulation ===

        // Low confidence → slower rate, longer pauses
        let confidence_factor = 0.7 + self.confidence * 0.3; // 0.7 to 1.0
        pacing.rate *= confidence_factor;

        // Low confidence → higher tau (more deliberate)
        let tau_boost = 1.0 + (1.0 - self.confidence) * 0.5;
        pacing.tau *= tau_boost;

        // Low confidence → more pauses
        let pause_boost = 1.0 + (1.0 - self.confidence) * 0.3;
        pacing.phrase_pause *= pause_boost;

        // Confidence affects emphasis variance
        // High confidence = consistent emphasis
        // Low confidence = reduced emphasis (mumbling)
        pacing.emphasis *= self.confidence * 0.5 + 0.5;

        // === Semantic Prosody Modulation ===

        // Apply semantic rate modifier
        pacing.rate *= self.semantic_prosody.rate_modifier;

        // Apply semantic pause modifiers
        pacing.phrase_pause *= self.semantic_prosody.post_pause;
        pacing.sentence_pause *= self.semantic_prosody.pre_pause;

        // Apply energy modifier to emphasis
        pacing.emphasis *= self.semantic_prosody.energy_modifier;

        // Map pitch contour to emotional valence
        // Rising pitch → positive valence, falling → negative
        pacing.emotional_valence += self.semantic_prosody.pitch_contour * 0.3;
        pacing.emotional_valence = pacing.emotional_valence.clamp(-1.0, 1.0);

        // === Phase 16 Consciousness Signal Modulation ===

        // Low unified quality → slower, more cautious speech (0.8–1.0× rate)
        let quality_factor = 0.8 + self.unified_quality * 0.2;
        pacing.rate *= quality_factor;

        // Low epistemic confidence → flatter emphasis (0.7–1.0× emphasis)
        let epistemic_factor = 0.7 + self.epistemic_confidence * 0.3;
        pacing.emphasis *= epistemic_factor;

        // Dissipative gating → longer phrase pauses (+50%)
        if self.dissipative_gated {
            pacing.phrase_pause *= 1.5;
        }

        // Negative coherence velocity → slower rate, longer pauses
        if self.coherence_velocity < 0.0 {
            // velocity in roughly [-1, 0]: map to 0.85–1.0× rate
            let velocity_factor = (1.0 + self.coherence_velocity * 0.15).clamp(0.85, 1.0);
            pacing.rate *= velocity_factor;
            // Also lengthen pauses when coherence is dropping
            let pause_factor = (1.0 - self.coherence_velocity * 0.2).clamp(1.0, 1.3);
            pacing.phrase_pause *= pause_factor;
        }

        // Low cross-module agreement → reduced emphasis (modules disagree → less assertive)
        let agreement_factor = 0.8 + self.cross_module_agreement * 0.2;
        pacing.emphasis *= agreement_factor;

        // Consciousness level → overall expressiveness scaling
        // Higher consciousness = wider dynamic range in emphasis and valence
        let consciousness_scale = 0.6 + self.consciousness_level as f32 * 0.4;
        pacing.emphasis *= consciousness_scale;

        // === Therapeutic Voice Modulation ===
        // Science: Safran & Muran (2000) — therapist prosody attunement;
        // Imel et al. (2014) — vocal quality predicts therapeutic outcomes.
        #[cfg(feature = "therapeutic")]
        {
            // High client distress → slower rate, longer pauses, softer energy
            // Creates a holding, calming vocal environment
            if self.client_distress > 0.3 {
                let distress_factor = 1.0 - (self.client_distress - 0.3) * 0.4; // 0.72–1.0
                pacing.rate *= distress_factor;
                let pause_factor = 1.0 + (self.client_distress - 0.3) * 0.6; // 1.0–1.42
                pacing.phrase_pause *= pause_factor;
                pacing.sentence_pause *= pause_factor;
                // Softer energy: reduce emphasis proportionally
                pacing.emphasis *= 1.0 - (self.client_distress - 0.3) * 0.3; // 0.79–1.0
                // More breath pauses when client is distressed (models attentive presence)
                pacing.breath_probability += self.client_distress * 0.15;
                pacing.breath_probability = pacing.breath_probability.min(0.5);
            }

            // Low alliance → warmer, less assertive (rapport-building posture)
            if self.alliance_quality < 0.4 {
                let warmth_boost = (0.4 - self.alliance_quality) * 0.3; // 0–0.12
                pacing.emotional_valence += warmth_boost; // Shift toward warmth
                pacing.emotional_valence = pacing.emotional_valence.clamp(-1.0, 1.0);
                pacing.emphasis *= 0.85 + self.alliance_quality * 0.375; // 0.85–1.0
            }

            // High alliance → wider dynamic range (permits more expressive speech)
            if self.alliance_quality > 0.7 {
                let range_boost = (self.alliance_quality - 0.7) * 0.33; // 0–0.1
                pacing.emphasis *= 1.0 + range_boost;
            }

            // Therapeutic intent → prosodic coloring
            match self.therapeutic_intent as u8 {
                // Validate (0) or strong validation (6): warm, soft, deliberate
                0 | 6 => {
                    pacing.emotional_valence += 0.15;
                    pacing.emotional_valence = pacing.emotional_valence.clamp(-1.0, 1.0);
                    pacing.rate *= 0.92;
                }
                // Reflect (2): rising intonation, question-like, curious tone
                2 => {
                    pacing.emotional_valence += 0.05;
                    pacing.emotional_valence = pacing.emotional_valence.clamp(-1.0, 1.0);
                    // Slightly slower for reflective listening
                    pacing.rate *= 0.95;
                }
                // Challenge (5) or ground (4): firmer, steadier
                4 | 5 => {
                    pacing.emphasis *= 1.1;
                }
                // Crisis (7): calm, slow, grounding — like a anchor
                7 => {
                    pacing.rate *= 0.75;
                    pacing.phrase_pause *= 1.5;
                    pacing.sentence_pause *= 1.5;
                    pacing.emphasis *= 0.8;
                    pacing.emotional_valence = pacing.emotional_valence.max(0.0); // Never negative in crisis
                    pacing.breath_probability = 0.3; // Regular breathing cues
                }
                _ => {}
            }
        }

        pacing
    }

    /// Get pitch contour for current semantic content
    /// Returns: -1.0 (falling) to 1.0 (rising)
    pub fn pitch_contour(&self) -> f32 {
        self.semantic_prosody.pitch_contour
    }

    /// Get pitch range expansion factor
    pub fn pitch_range(&self) -> f32 {
        self.semantic_prosody.pitch_range
    }

    /// Check if semantic content suggests a question
    pub fn is_question(&self) -> bool {
        self.semantic_prosody
            .categories
            .contains(&SemanticCategory::Question)
    }

    /// Check if semantic content suggests emphasis
    pub fn needs_emphasis(&self) -> bool {
        self.semantic_prosody
            .categories
            .contains(&SemanticCategory::Emphasis)
    }

    /// Get all detected semantic categories
    pub fn semantic_categories(&self) -> &[SemanticCategory] {
        &self.semantic_prosody.categories
    }

    /// Get emphasis for a specific word/concept
    pub fn get_emphasis(&self, word: &str) -> f32 {
        // Check direct match
        if let Some(&emphasis) = self.word_emphasis.get(word) {
            return emphasis;
        }

        // Check if word contains a highlighted concept
        let word_lower = word.to_lowercase();
        for concept in &self.highlighted_concepts {
            if word_lower.contains(&concept.to_lowercase()) {
                // Emphasized concepts get 1.3x emphasis
                return 1.3;
            }
        }

        // Default emphasis
        1.0
    }

    /// Check if a concept is highlighted (should be emphasized)
    pub fn is_highlighted(&self, concept: &str) -> bool {
        let concept_lower = concept.to_lowercase();
        self.highlighted_concepts
            .iter()
            .any(|c| c.to_lowercase() == concept_lower)
    }
}

impl Default for CognitivePacing {
    fn default() -> Self {
        Self {
            base_pacing: LTCPacing::default(),
            confidence: 1.0,
            word_emphasis: HashMap::new(),
            highlighted_concepts: Vec::new(),
            semantic_prosody: SemanticProsody::neutral(),
            unified_quality: 1.0,
            epistemic_confidence: 0.5,
            dissipative_gated: false,
            dissipative_factor: 1.0,
            coherence_velocity: 0.0,
            cross_module_agreement: 1.0,
            consciousness_level: 0.5,
            #[cfg(feature = "therapeutic")]
            client_distress: 0.0,
            #[cfg(feature = "therapeutic")]
            alliance_quality: 0.5,
            #[cfg(feature = "therapeutic")]
            therapeutic_intent: 0.0,
        }
    }
}

/// Bridge between cognitive loop and voice synthesis
pub struct CognitiveVoiceBridge {
    /// Current pacing state
    current_pacing: CognitivePacing,
    /// Smoothing factor for pacing transitions
    smoothing: f32,
    /// History of prediction errors for confidence smoothing
    error_history: VecDeque<f32>,
    /// Maximum history length
    max_history: usize,
}

impl CognitiveVoiceBridge {
    /// Create a new bridge
    pub fn new() -> Self {
        Self {
            current_pacing: CognitivePacing::default(),
            smoothing: 0.3, // 30% new, 70% old
            error_history: VecDeque::new(),
            max_history: 10,
        }
    }

    /// Update pacing from cognitive loop output
    pub fn update(
        &mut self,
        cfc_output: &[f32],
        tau: f32,
        prediction_error: f32,
        attention_state: HashMap<String, f32>,
        detected_primitives: Vec<String>,
    ) -> &CognitivePacing {
        // Update error history for smoothing
        self.error_history.push_back(prediction_error);
        if self.error_history.len() > self.max_history {
            self.error_history.pop_front();
        }

        // Smooth prediction error
        let smoothed_error =
            self.error_history.iter().sum::<f32>() / self.error_history.len() as f32;

        // Create new pacing
        let new_pacing = CognitivePacing::from_cognitive_loop(
            cfc_output,
            tau,
            smoothed_error,
            attention_state,
            detected_primitives,
        );

        // Smooth transition to new pacing
        self.current_pacing.base_pacing = self
            .current_pacing
            .base_pacing
            .lerp(&new_pacing.base_pacing, self.smoothing);
        self.current_pacing.confidence = self.current_pacing.confidence * (1.0 - self.smoothing)
            + new_pacing.confidence * self.smoothing;
        self.current_pacing.word_emphasis = new_pacing.word_emphasis;
        self.current_pacing.highlighted_concepts = new_pacing.highlighted_concepts;
        self.current_pacing.semantic_prosody = new_pacing.semantic_prosody;

        &self.current_pacing
    }

    /// Update Phase 16 consciousness signals on the current pacing state.
    ///
    /// Call this after `update()` when CycleMetadata becomes available,
    /// to layer Phase 16 modulation onto the existing pacing.
    pub fn update_from_metadata(&mut self, signals: &VoiceConsciousnessSignals) {
        self.current_pacing.unified_quality = signals.unified_quality;
        self.current_pacing.epistemic_confidence = signals.epistemic_confidence;
        self.current_pacing.dissipative_gated = signals.dissipative_gated;
        self.current_pacing.dissipative_factor = signals.dissipative_factor;
        self.current_pacing.coherence_velocity = signals.coherence_velocity;
        self.current_pacing.cross_module_agreement = signals.cross_module_agreement;
        self.current_pacing.consciousness_level = signals.consciousness_level;
    }

    /// Get current pacing for synthesis
    pub fn get_pacing(&self) -> &CognitivePacing {
        &self.current_pacing
    }

    /// Get effective LTCPacing for synthesizer
    pub fn get_ltc_pacing(&self) -> LTCPacing {
        self.current_pacing.effective_pacing()
    }

    /// Set smoothing factor (0.0 = no change, 1.0 = instant)
    pub fn set_smoothing(&mut self, smoothing: f32) {
        self.smoothing = smoothing.clamp(0.0, 1.0);
    }
}

impl Default for CognitiveVoiceBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_pacing_creation() {
        let cfc_output = vec![0.5, -0.3, 0.8, -0.1, 0.2];
        let tau = 1.0;
        let prediction_error = 0.2;
        let mut attention = HashMap::new();
        attention.insert("cause".to_string(), 0.8);
        attention.insert("effect".to_string(), 0.6);
        let primitives = vec!["cause".to_string()];

        let pacing = CognitivePacing::from_cognitive_loop(
            &cfc_output,
            tau,
            prediction_error,
            attention,
            primitives,
        );

        // Confidence should be 0.8 (1 - 0.2)
        assert!((pacing.confidence - 0.8).abs() < 0.01);

        // Should have emphasis for "cause"
        let cause_emphasis = pacing.get_emphasis("cause");
        assert!(
            cause_emphasis > 1.0,
            "cause should be emphasized: {}",
            cause_emphasis
        );
    }

    #[test]
    fn test_confidence_affects_pacing() {
        let cfc_output = vec![0.5; 10];
        let tau = 1.0;

        // High confidence
        let pacing_high = CognitivePacing::from_cognitive_loop(
            &cfc_output,
            tau,
            0.1, // Low error = high confidence
            HashMap::new(),
            vec![],
        );

        // Low confidence
        let pacing_low = CognitivePacing::from_cognitive_loop(
            &cfc_output,
            tau,
            0.8, // High error = low confidence
            HashMap::new(),
            vec![],
        );

        let effective_high = pacing_high.effective_pacing();
        let effective_low = pacing_low.effective_pacing();

        // Low confidence should have slower rate
        assert!(
            effective_low.rate < effective_high.rate,
            "Low confidence should have slower rate: {} vs {}",
            effective_low.rate,
            effective_high.rate
        );

        // Low confidence should have longer pauses
        assert!(
            effective_low.phrase_pause > effective_high.phrase_pause,
            "Low confidence should have longer pauses: {} vs {}",
            effective_low.phrase_pause,
            effective_high.phrase_pause
        );
    }

    #[test]
    fn test_bridge_smoothing() {
        let mut bridge = CognitiveVoiceBridge::new();

        // Get initial confidence (should be default = 1.0)
        let initial_confidence = bridge.get_pacing().confidence;
        assert!(
            (initial_confidence - 1.0).abs() < 0.01,
            "Initial confidence should be 1.0: {}",
            initial_confidence
        );

        // Single update with very high error
        bridge.update(
            &[0.5; 10],
            1.0,
            1.0, // Maximum error = 0 confidence
            HashMap::new(),
            vec![],
        );
        let after_one_update = bridge.get_pacing().confidence;

        // Smoothing should prevent instant jump to 0
        // With smoothing=0.3: new = 1.0*0.7 + 0.0*0.3 = 0.7 (theoretical)
        // But error history also smooths, so actual value differs
        assert!(
            after_one_update > 0.2,
            "Smoothing should prevent instant drop to 0: {}",
            after_one_update
        );
        assert!(
            after_one_update < initial_confidence,
            "Confidence should decrease: {} vs {}",
            after_one_update,
            initial_confidence
        );

        // Multiple updates should converge toward the new value
        for _ in 0..20 {
            bridge.update(
                &[0.5; 10],
                1.0,
                1.0, // Keep high error
                HashMap::new(),
                vec![],
            );
        }
        let after_many_updates = bridge.get_pacing().confidence;

        // After many high-error updates, confidence should be low
        assert!(
            after_many_updates < 0.3,
            "Many high-error updates should reduce confidence: {}",
            after_many_updates
        );
    }

    // ===== Semantic Prosody Tests =====

    #[test]
    fn test_semantic_category_detection() {
        assert_eq!(
            SemanticCategory::from_concept("because"),
            SemanticCategory::Cause
        );
        assert_eq!(
            SemanticCategory::from_concept("therefore"),
            SemanticCategory::Effect
        );
        assert_eq!(
            SemanticCategory::from_concept("why"),
            SemanticCategory::Question
        );
        assert_eq!(
            SemanticCategory::from_concept("important"),
            SemanticCategory::Emphasis
        );
        assert_eq!(
            SemanticCategory::from_concept("maybe"),
            SemanticCategory::Uncertainty
        );
        assert_eq!(
            SemanticCategory::from_concept("however"),
            SemanticCategory::Contrast
        );
        assert_eq!(
            SemanticCategory::from_concept("first"),
            SemanticCategory::Sequence
        );
        assert_eq!(
            SemanticCategory::from_concept("random"),
            SemanticCategory::Neutral
        );
    }

    #[test]
    fn test_semantic_prosody_cause() {
        let prosody = SemanticProsody::from_primitives(&["cause".to_string()]);

        // Cause should have rising pitch
        assert!(
            prosody.pitch_contour > 0.0,
            "Cause should have rising pitch: {}",
            prosody.pitch_contour
        );

        // Cause should slow down
        assert!(
            prosody.rate_modifier < 1.0,
            "Cause should slow rate: {}",
            prosody.rate_modifier
        );
    }

    #[test]
    fn test_semantic_prosody_question() {
        let prosody = SemanticProsody::from_primitives(&["why".to_string()]);

        // Questions have strong rising pitch
        assert!(
            prosody.pitch_contour > 0.3,
            "Question should have rising pitch: {}",
            prosody.pitch_contour
        );

        // Questions are more expressive
        assert!(
            prosody.pitch_range > 1.0,
            "Question should expand pitch range: {}",
            prosody.pitch_range
        );
    }

    #[test]
    fn test_semantic_prosody_uncertainty() {
        let prosody =
            SemanticProsody::from_primitives(&["maybe".to_string(), "perhaps".to_string()]);

        // Uncertainty has falling pitch
        assert!(
            prosody.pitch_contour < 0.0,
            "Uncertainty should have falling pitch: {}",
            prosody.pitch_contour
        );

        // Uncertainty is quieter
        assert!(
            prosody.energy_modifier < 1.0,
            "Uncertainty should reduce energy: {}",
            prosody.energy_modifier
        );
    }

    #[test]
    fn test_pacing_with_semantic_prosody() {
        // Use mixed positive/negative values to get neutral base valence
        // (all positive = valence 1.0 which gets clamped)
        let cfc_output = vec![0.3, -0.2, 0.4, -0.3, 0.2, -0.1, 0.3, -0.2, 0.1, -0.1];
        let tau = 1.0;

        // Create pacing with question semantics
        let pacing_question = CognitivePacing::from_cognitive_loop(
            &cfc_output,
            tau,
            0.1,
            HashMap::new(),
            vec!["why".to_string()], // Question
        );

        // Create pacing without semantics
        let pacing_neutral =
            CognitivePacing::from_cognitive_loop(&cfc_output, tau, 0.1, HashMap::new(), vec![]);

        let effective_question = pacing_question.effective_pacing();
        let effective_neutral = pacing_neutral.effective_pacing();

        // Question should have higher emotional valence (rising intonation)
        // Question adds pitch_contour * 0.3 = 0.5 * 0.3 = 0.15 to valence
        assert!(
            effective_question.emotional_valence > effective_neutral.emotional_valence,
            "Question should raise emotional valence: {} vs {}",
            effective_question.emotional_valence,
            effective_neutral.emotional_valence
        );

        // Question detection should work
        assert!(pacing_question.is_question());
        assert!(!pacing_neutral.is_question());
    }

    #[test]
    fn test_semantic_prosody_combination() {
        // Multiple semantic categories combine
        let prosody = SemanticProsody::from_primitives(&[
            "important".to_string(), // Emphasis
            "but".to_string(),       // Contrast
        ]);

        assert!(prosody.has_semantic_content());
        assert_eq!(prosody.categories.len(), 2);

        // Should have pauses from contrast
        assert!(prosody.pre_pause > 1.2);
    }

    // ===== Phase 16 Consciousness Signal Tests =====

    fn make_base_pacing() -> CognitivePacing {
        CognitivePacing::from_cognitive_loop(
            &[0.3, -0.2, 0.4, -0.3, 0.2],
            1.0,
            0.1,
            HashMap::new(),
            vec![],
        )
    }

    #[test]
    fn test_phase16_quality_affects_rate() {
        let mut high_quality = make_base_pacing();
        high_quality.unified_quality = 1.0;
        let rate_high = high_quality.effective_pacing().rate;

        let mut low_quality = make_base_pacing();
        low_quality.unified_quality = 0.0;
        let rate_low = low_quality.effective_pacing().rate;

        assert!(
            rate_low < rate_high,
            "Low unified_quality should reduce rate: low={rate_low} vs high={rate_high}"
        );
    }

    #[test]
    fn test_phase16_epistemic_confidence_affects_emphasis() {
        let mut high_epistemic = make_base_pacing();
        high_epistemic.epistemic_confidence = 1.0;
        let emphasis_high = high_epistemic.effective_pacing().emphasis;

        let mut low_epistemic = make_base_pacing();
        low_epistemic.epistemic_confidence = 0.0;
        let emphasis_low = low_epistemic.effective_pacing().emphasis;

        assert!(
            emphasis_low < emphasis_high,
            "Low epistemic_confidence should reduce emphasis: low={emphasis_low} vs high={emphasis_high}"
        );
    }

    #[test]
    fn test_phase16_dissipative_gating_adds_pauses() {
        let mut ungated = make_base_pacing();
        ungated.dissipative_gated = false;
        let pause_ungated = ungated.effective_pacing().phrase_pause;

        let mut gated = make_base_pacing();
        gated.dissipative_gated = true;
        let pause_gated = gated.effective_pacing().phrase_pause;

        assert!(
            pause_gated > pause_ungated,
            "Dissipative gating should increase phrase_pause: gated={pause_gated} vs ungated={pause_ungated}"
        );
        // Specifically +50%
        let expected = pause_ungated * 1.5;
        assert!(
            (pause_gated - expected).abs() < 0.01,
            "Expected 1.5× pause: got {pause_gated}, expected {expected}"
        );
    }

    #[test]
    fn test_phase16_coherence_velocity_modulation() {
        let mut stable = make_base_pacing();
        stable.coherence_velocity = 0.0;
        let rate_stable = stable.effective_pacing().rate;

        let mut dropping = make_base_pacing();
        dropping.coherence_velocity = -0.5;
        let rate_dropping = dropping.effective_pacing().rate;

        assert!(
            rate_dropping < rate_stable,
            "Negative coherence_velocity should slow rate: dropping={rate_dropping} vs stable={rate_stable}"
        );
    }

    #[test]
    fn test_phase16_backward_compat() {
        // Existing from_cognitive_loop() should produce default Phase 16 values
        let pacing =
            CognitivePacing::from_cognitive_loop(&[0.5; 10], 1.0, 0.2, HashMap::new(), vec![]);

        assert!((pacing.unified_quality - 1.0).abs() < 0.01);
        assert!((pacing.epistemic_confidence - 0.5).abs() < 0.01);
        assert!(!pacing.dissipative_gated);
        assert!((pacing.dissipative_factor - 1.0).abs() < 0.01);
        assert!((pacing.coherence_velocity).abs() < 0.01);
        assert!((pacing.cross_module_agreement - 1.0).abs() < 0.01);
        assert!((pacing.consciousness_level - 0.5).abs() < 0.01);
    }

    // ===== Therapeutic Voice Modulation Tests =====

    #[cfg(feature = "therapeutic")]
    fn make_therapeutic_pacing(distress: f32, alliance: f32, intent: f32) -> CognitivePacing {
        let mut pacing = make_base_pacing();
        pacing.client_distress = distress;
        pacing.alliance_quality = alliance;
        pacing.therapeutic_intent = intent;
        pacing
    }

    #[cfg(feature = "therapeutic")]
    #[test]
    fn test_therapeutic_distress_slows_rate() {
        let calm = make_therapeutic_pacing(0.0, 0.5, 1.0);
        let distressed = make_therapeutic_pacing(0.8, 0.5, 1.0);
        let rate_calm = calm.effective_pacing().rate;
        let rate_distressed = distressed.effective_pacing().rate;
        assert!(
            rate_distressed < rate_calm,
            "High distress should slow rate: distressed={rate_distressed} vs calm={rate_calm}"
        );
    }

    #[cfg(feature = "therapeutic")]
    #[test]
    fn test_therapeutic_distress_increases_pauses() {
        let calm = make_therapeutic_pacing(0.0, 0.5, 1.0);
        let distressed = make_therapeutic_pacing(0.8, 0.5, 1.0);
        let pause_calm = calm.effective_pacing().phrase_pause;
        let pause_distressed = distressed.effective_pacing().phrase_pause;
        assert!(
            pause_distressed > pause_calm,
            "High distress should increase pauses: distressed={pause_distressed} vs calm={pause_calm}"
        );
    }

    #[cfg(feature = "therapeutic")]
    #[test]
    fn test_therapeutic_distress_softens_emphasis() {
        let calm = make_therapeutic_pacing(0.0, 0.5, 1.0);
        let distressed = make_therapeutic_pacing(0.8, 0.5, 1.0);
        let emph_calm = calm.effective_pacing().emphasis;
        let emph_distressed = distressed.effective_pacing().emphasis;
        assert!(
            emph_distressed < emph_calm,
            "High distress should soften emphasis: distressed={emph_distressed} vs calm={emph_calm}"
        );
    }

    #[cfg(feature = "therapeutic")]
    #[test]
    fn test_therapeutic_low_alliance_warms_tone() {
        let low_alliance = make_therapeutic_pacing(0.0, 0.1, 1.0);
        let high_alliance = make_therapeutic_pacing(0.0, 0.8, 1.0);
        let valence_low = low_alliance.effective_pacing().emotional_valence;
        let valence_high = high_alliance.effective_pacing().emotional_valence;
        assert!(
            valence_low > valence_high,
            "Low alliance should boost warmth: low={valence_low} vs high={valence_high}"
        );
    }

    #[cfg(feature = "therapeutic")]
    #[test]
    fn test_therapeutic_high_alliance_widens_range() {
        let mid_alliance = make_therapeutic_pacing(0.0, 0.5, 1.0);
        let high_alliance = make_therapeutic_pacing(0.0, 0.9, 1.0);
        let emph_mid = mid_alliance.effective_pacing().emphasis;
        let emph_high = high_alliance.effective_pacing().emphasis;
        assert!(
            emph_high > emph_mid,
            "High alliance should widen emphasis: high={emph_high} vs mid={emph_mid}"
        );
    }

    #[cfg(feature = "therapeutic")]
    #[test]
    fn test_therapeutic_intent_validate_warms() {
        let validate = make_therapeutic_pacing(0.0, 0.5, 0.0); // intent 0
        let neutral = make_therapeutic_pacing(0.0, 0.5, 1.0); // intent 1 (no match)
        let valence_val = validate.effective_pacing().emotional_valence;
        let valence_neutral = neutral.effective_pacing().emotional_valence;
        assert!(
            valence_val > valence_neutral,
            "Validate intent should warm tone: validate={valence_val} vs neutral={valence_neutral}"
        );
    }

    #[cfg(feature = "therapeutic")]
    #[test]
    fn test_therapeutic_intent_crisis_grounding() {
        let crisis = make_therapeutic_pacing(0.0, 0.5, 7.0);
        let neutral = make_therapeutic_pacing(0.0, 0.5, 1.0);
        let rate_crisis = crisis.effective_pacing().rate;
        let rate_neutral = neutral.effective_pacing().rate;
        assert!(
            rate_crisis < rate_neutral * 0.8,
            "Crisis intent should significantly slow rate: crisis={rate_crisis} vs neutral={rate_neutral}"
        );
        let breath_crisis = crisis.effective_pacing().breath_probability;
        assert!(
            breath_crisis >= 0.29,
            "Crisis should set breath_probability ≈ 0.3: got {breath_crisis}"
        );
    }

    #[cfg(feature = "therapeutic")]
    #[test]
    fn test_therapeutic_intent_challenge_firms() {
        let challenge = make_therapeutic_pacing(0.0, 0.5, 5.0); // intent 5
        let neutral = make_therapeutic_pacing(0.0, 0.5, 1.0);
        let emph_challenge = challenge.effective_pacing().emphasis;
        let emph_neutral = neutral.effective_pacing().emphasis;
        assert!(
            emph_challenge > emph_neutral,
            "Challenge intent should firm emphasis: challenge={emph_challenge} vs neutral={emph_neutral}"
        );
    }

    #[cfg(feature = "therapeutic")]
    #[test]
    fn test_therapeutic_distress_threshold() {
        // At exactly 0.3, distress modulation should not activate
        let at_threshold = make_therapeutic_pacing(0.3, 0.5, 1.0);
        let below_threshold = make_therapeutic_pacing(0.29, 0.5, 1.0);
        let rate_at = at_threshold.effective_pacing().rate;
        let rate_below = below_threshold.effective_pacing().rate;
        assert!(
            (rate_at - rate_below).abs() < 0.01,
            "At distress=0.3 should behave like below: at={rate_at} vs below={rate_below}"
        );
    }

    #[cfg(feature = "therapeutic")]
    #[test]
    fn test_therapeutic_breath_probability_capped() {
        let extreme = make_therapeutic_pacing(1.0, 0.5, 1.0);
        let breath = extreme.effective_pacing().breath_probability;
        assert!(
            breath <= 0.5,
            "Breath probability should cap at 0.5: got {breath}"
        );
    }
}
