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
use std::collections::HashMap;

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
        if lower.contains("however") || lower.contains("although")
            || lower == "but" || lower.contains("contrast")
            || lower.contains("instead") || lower == "yet"
        {
            return Self::Contrast;
        }

        // Cause indicators
        if lower.contains("cause") || lower.contains("because")
            || lower.contains("leads") || lower.contains("since")
            || lower.contains("due to") || lower.contains("reason")
        {
            return Self::Cause;
        }

        // Effect indicators
        if lower.contains("effect") || lower.contains("therefore")
            || lower.contains("result") || lower.contains("consequently")
            || lower.contains("thus") || lower.contains("hence")
        {
            return Self::Effect;
        }

        // Question indicators
        if lower.contains("why") || lower == "how"
            || lower.contains("what") || lower.contains("when")
            || lower.contains("where") || lower.contains("who")
            || lower.contains("?")
        {
            return Self::Question;
        }

        // Emphasis indicators
        if lower.contains("important") || lower.contains("critical")
            || lower.contains("key") || lower.contains("essential")
            || lower.contains("crucial") || lower.contains("vital")
        {
            return Self::Emphasis;
        }

        // Uncertainty indicators
        if lower.contains("maybe") || lower.contains("perhaps")
            || lower.contains("possibly") || lower.contains("might")
            || lower.contains("uncertain") || lower.contains("unclear")
        {
            return Self::Uncertainty;
        }

        // Sequence indicators
        if lower.contains("first") || lower.contains("then")
            || lower.contains("next") || lower.contains("finally")
            || lower.contains("second") || lower.contains("last")
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
        let categories: Vec<SemanticCategory> = primitives.iter()
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
            SemanticCategory::Cause => (0.3, 0.9, 1.0, 1.2, 1.1, 1.0),    // Rising, slower, pause after
            SemanticCategory::Effect => (-0.3, 1.0, 1.2, 1.0, 1.0, 1.0), // Falling, pause before
            SemanticCategory::Question => (0.5, 0.85, 1.0, 1.3, 1.3, 1.0), // Rising, slower, expressive
            SemanticCategory::Emphasis => (0.2, 0.8, 1.1, 1.1, 1.2, 1.2), // Higher, slower, louder
            SemanticCategory::Uncertainty => (-0.2, 0.85, 1.0, 1.2, 0.9, 0.9), // Falling, quieter
            SemanticCategory::Contrast => (0.0, 0.9, 1.5, 1.3, 1.4, 1.1), // Pause, expressive
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
        let max_attention = attention_state.values()
            .cloned()
            .fold(0.0f32, f32::max)
            .max(0.001);

        let word_emphasis: HashMap<String, f32> = attention_state
            .into_iter()
            .map(|(concept, attention)| {
                // Scale attention to emphasis multiplier
                let normalized = attention / max_attention;
                let emphasis = 0.5 + normalized;  // 0.5 to 1.5
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
        }
    }

    /// Get effective pacing with confidence and semantic modulation
    pub fn effective_pacing(&self) -> LTCPacing {
        let mut pacing = self.base_pacing.clone();

        // === Confidence Modulation ===

        // Low confidence → slower rate, longer pauses
        let confidence_factor = 0.7 + self.confidence * 0.3;  // 0.7 to 1.0
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
        self.semantic_prosody.categories.contains(&SemanticCategory::Question)
    }

    /// Check if semantic content suggests emphasis
    pub fn needs_emphasis(&self) -> bool {
        self.semantic_prosody.categories.contains(&SemanticCategory::Emphasis)
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
        self.highlighted_concepts.iter()
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
    error_history: Vec<f32>,
    /// Maximum history length
    max_history: usize,
}

impl CognitiveVoiceBridge {
    /// Create a new bridge
    pub fn new() -> Self {
        Self {
            current_pacing: CognitivePacing::default(),
            smoothing: 0.3,  // 30% new, 70% old
            error_history: Vec::new(),
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
        self.error_history.push(prediction_error);
        if self.error_history.len() > self.max_history {
            self.error_history.remove(0);
        }

        // Smooth prediction error
        let smoothed_error = self.error_history.iter().sum::<f32>()
            / self.error_history.len() as f32;

        // Create new pacing
        let new_pacing = CognitivePacing::from_cognitive_loop(
            cfc_output,
            tau,
            smoothed_error,
            attention_state,
            detected_primitives,
        );

        // Smooth transition to new pacing
        self.current_pacing.base_pacing = self.current_pacing.base_pacing
            .lerp(&new_pacing.base_pacing, self.smoothing);
        self.current_pacing.confidence = self.current_pacing.confidence
            * (1.0 - self.smoothing) + new_pacing.confidence * self.smoothing;
        self.current_pacing.word_emphasis = new_pacing.word_emphasis;
        self.current_pacing.highlighted_concepts = new_pacing.highlighted_concepts;
        self.current_pacing.semantic_prosody = new_pacing.semantic_prosody;

        &self.current_pacing
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
        assert!(cause_emphasis > 1.0, "cause should be emphasized: {}", cause_emphasis);
    }

    #[test]
    fn test_confidence_affects_pacing() {
        let cfc_output = vec![0.5; 10];
        let tau = 1.0;

        // High confidence
        let pacing_high = CognitivePacing::from_cognitive_loop(
            &cfc_output,
            tau,
            0.1,  // Low error = high confidence
            HashMap::new(),
            vec![],
        );

        // Low confidence
        let pacing_low = CognitivePacing::from_cognitive_loop(
            &cfc_output,
            tau,
            0.8,  // High error = low confidence
            HashMap::new(),
            vec![],
        );

        let effective_high = pacing_high.effective_pacing();
        let effective_low = pacing_low.effective_pacing();

        // Low confidence should have slower rate
        assert!(effective_low.rate < effective_high.rate,
                "Low confidence should have slower rate: {} vs {}",
                effective_low.rate, effective_high.rate);

        // Low confidence should have longer pauses
        assert!(effective_low.phrase_pause > effective_high.phrase_pause,
                "Low confidence should have longer pauses: {} vs {}",
                effective_low.phrase_pause, effective_high.phrase_pause);
    }

    #[test]
    fn test_bridge_smoothing() {
        let mut bridge = CognitiveVoiceBridge::new();

        // Get initial confidence (should be default = 1.0)
        let initial_confidence = bridge.get_pacing().confidence;
        assert!((initial_confidence - 1.0).abs() < 0.01,
                "Initial confidence should be 1.0: {}", initial_confidence);

        // Single update with very high error
        bridge.update(
            &[0.5; 10],
            1.0,
            1.0,  // Maximum error = 0 confidence
            HashMap::new(),
            vec![],
        );
        let after_one_update = bridge.get_pacing().confidence;

        // Smoothing should prevent instant jump to 0
        // With smoothing=0.3: new = 1.0*0.7 + 0.0*0.3 = 0.7 (theoretical)
        // But error history also smooths, so actual value differs
        assert!(after_one_update > 0.2,
                "Smoothing should prevent instant drop to 0: {}", after_one_update);
        assert!(after_one_update < initial_confidence,
                "Confidence should decrease: {} vs {}", after_one_update, initial_confidence);

        // Multiple updates should converge toward the new value
        for _ in 0..20 {
            bridge.update(
                &[0.5; 10],
                1.0,
                1.0,  // Keep high error
                HashMap::new(),
                vec![],
            );
        }
        let after_many_updates = bridge.get_pacing().confidence;

        // After many high-error updates, confidence should be low
        assert!(after_many_updates < 0.3,
                "Many high-error updates should reduce confidence: {}", after_many_updates);
    }

    // ===== Semantic Prosody Tests =====

    #[test]
    fn test_semantic_category_detection() {
        assert_eq!(SemanticCategory::from_concept("because"), SemanticCategory::Cause);
        assert_eq!(SemanticCategory::from_concept("therefore"), SemanticCategory::Effect);
        assert_eq!(SemanticCategory::from_concept("why"), SemanticCategory::Question);
        assert_eq!(SemanticCategory::from_concept("important"), SemanticCategory::Emphasis);
        assert_eq!(SemanticCategory::from_concept("maybe"), SemanticCategory::Uncertainty);
        assert_eq!(SemanticCategory::from_concept("however"), SemanticCategory::Contrast);
        assert_eq!(SemanticCategory::from_concept("first"), SemanticCategory::Sequence);
        assert_eq!(SemanticCategory::from_concept("random"), SemanticCategory::Neutral);
    }

    #[test]
    fn test_semantic_prosody_cause() {
        let prosody = SemanticProsody::from_primitives(&["cause".to_string()]);

        // Cause should have rising pitch
        assert!(prosody.pitch_contour > 0.0,
                "Cause should have rising pitch: {}", prosody.pitch_contour);

        // Cause should slow down
        assert!(prosody.rate_modifier < 1.0,
                "Cause should slow rate: {}", prosody.rate_modifier);
    }

    #[test]
    fn test_semantic_prosody_question() {
        let prosody = SemanticProsody::from_primitives(&["why".to_string()]);

        // Questions have strong rising pitch
        assert!(prosody.pitch_contour > 0.3,
                "Question should have rising pitch: {}", prosody.pitch_contour);

        // Questions are more expressive
        assert!(prosody.pitch_range > 1.0,
                "Question should expand pitch range: {}", prosody.pitch_range);
    }

    #[test]
    fn test_semantic_prosody_uncertainty() {
        let prosody = SemanticProsody::from_primitives(&["maybe".to_string(), "perhaps".to_string()]);

        // Uncertainty has falling pitch
        assert!(prosody.pitch_contour < 0.0,
                "Uncertainty should have falling pitch: {}", prosody.pitch_contour);

        // Uncertainty is quieter
        assert!(prosody.energy_modifier < 1.0,
                "Uncertainty should reduce energy: {}", prosody.energy_modifier);
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
            vec!["why".to_string()],  // Question
        );

        // Create pacing without semantics
        let pacing_neutral = CognitivePacing::from_cognitive_loop(
            &cfc_output,
            tau,
            0.1,
            HashMap::new(),
            vec![],
        );

        let effective_question = pacing_question.effective_pacing();
        let effective_neutral = pacing_neutral.effective_pacing();

        // Question should have higher emotional valence (rising intonation)
        // Question adds pitch_contour * 0.3 = 0.5 * 0.3 = 0.15 to valence
        assert!(effective_question.emotional_valence > effective_neutral.emotional_valence,
                "Question should raise emotional valence: {} vs {}",
                effective_question.emotional_valence, effective_neutral.emotional_valence);

        // Question detection should work
        assert!(pacing_question.is_question());
        assert!(!pacing_neutral.is_question());
    }

    #[test]
    fn test_semantic_prosody_combination() {
        // Multiple semantic categories combine
        let prosody = SemanticProsody::from_primitives(&[
            "important".to_string(),  // Emphasis
            "but".to_string(),        // Contrast
        ]);

        assert!(prosody.has_semantic_content());
        assert_eq!(prosody.categories.len(), 2);

        // Should have pauses from contrast
        assert!(prosody.pre_pause > 1.2);
    }
}
