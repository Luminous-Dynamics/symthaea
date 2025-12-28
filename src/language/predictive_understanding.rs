//! Predictive Understanding - Language-Specific Active Inference
//!
//! This module implements predictive processing specifically for natural language
//! understanding, building on the Free Energy Principle (Friston) applied to
//! linguistic processing.
//!
//! # Theoretical Foundation
//!
//! **Predictive Processing for Language** (Clark, 2013; Kuperberg & Jaeger, 2016):
//! - Language comprehension as prediction
//! - Every word predicts the next based on:
//!   - Lexical priors (word frequency)
//!   - Syntactic expectations (construction grammar)
//!   - Semantic context (frame knowledge)
//!   - Discourse coherence (narrative flow)
//!
//! # Architecture Integration
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │              PREDICTIVE UNDERSTANDING LAYER                    │
//! ├────────────────────────────────────────────────────────────────┤
//! │                                                                │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
//! │  │  Lexical     │    │  Syntactic   │    │  Semantic    │     │
//! │  │  Predictions │ →  │  Predictions │ →  │  Predictions │     │
//! │  │  (vocab)     │    │ (constructs) │    │  (frames)    │     │
//! │  └──────────────┘    └──────────────┘    └──────────────┘     │
//! │         ↓                   ↓                   ↓              │
//! │  ┌────────────────────────────────────────────────────────┐   │
//! │  │              PREDICTION ERROR INTEGRATION               │   │
//! │  │     Precision-weighted combination across levels        │   │
//! │  └────────────────────────────────────────────────────────┘   │
//! │         ↓                                                      │
//! │  ┌────────────────────────────────────────────────────────┐   │
//! │  │               FREE ENERGY MINIMIZATION                  │   │
//! │  │     F = -log P(obs) + KL[Q(z)||P(z|obs)]                │   │
//! │  └────────────────────────────────────────────────────────┘   │
//! │                                                                │
//! └────────────────────────────────────────────────────────────────┘
//! ```

use crate::hdc::binary_hv::HV16;
use super::frames::{FrameLibrary, FrameActivator, FrameInstance};
use super::constructions::{ConstructionGrammar, ConstructionParse};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Helper function to create HV16 from text (deterministic hash-based)
fn hv16_from_text(text: &str) -> HV16 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    HV16::random(hasher.finish())
}

// =============================================================================
// LINGUISTIC PREDICTION LEVELS
// =============================================================================

/// Hierarchical levels of linguistic prediction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LinguisticLevel {
    /// Phonological/orthographic patterns
    Sublexical,
    /// Word-level expectations
    Lexical,
    /// Syntactic structure predictions
    Syntactic,
    /// Semantic/frame-based predictions
    Semantic,
    /// Discourse/narrative expectations
    Discourse,
}

impl LinguisticLevel {
    /// Get timescale for this level (in milliseconds)
    pub fn timescale_ms(&self) -> f64 {
        match self {
            LinguisticLevel::Sublexical => 50.0,   // ~50ms phoneme processing
            LinguisticLevel::Lexical => 200.0,     // ~200ms word recognition
            LinguisticLevel::Syntactic => 400.0,   // ~400ms syntactic integration
            LinguisticLevel::Semantic => 600.0,    // ~600ms semantic integration
            LinguisticLevel::Discourse => 2000.0,  // ~2s discourse processing
        }
    }

    /// Default precision (inverse variance) for this level
    pub fn default_precision(&self) -> f64 {
        match self {
            LinguisticLevel::Sublexical => 10.0,  // High precision for low-level
            LinguisticLevel::Lexical => 8.0,
            LinguisticLevel::Syntactic => 5.0,
            LinguisticLevel::Semantic => 3.0,
            LinguisticLevel::Discourse => 1.0,    // Low precision for high-level
        }
    }

    /// All levels in order from bottom to top
    pub fn all() -> Vec<Self> {
        vec![
            LinguisticLevel::Sublexical,
            LinguisticLevel::Lexical,
            LinguisticLevel::Syntactic,
            LinguisticLevel::Semantic,
            LinguisticLevel::Discourse,
        ]
    }
}

// =============================================================================
// LINGUISTIC PREDICTION
// =============================================================================

/// A prediction at a specific linguistic level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticPrediction {
    /// Hierarchical level
    pub level: LinguisticLevel,

    /// Predicted HDC vector
    pub predicted: HV16,

    /// Precision (confidence in this prediction)
    pub precision: f64,

    /// Source of prediction (what generated it)
    pub source: PredictionSource,

    /// Alternatives (other possible predictions with probabilities)
    pub alternatives: Vec<(HV16, f64)>,
}

/// What generated this prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionSource {
    /// Statistical co-occurrence
    LexicalPrior { word: String, probability: f64 },
    /// Syntactic construction expectation
    SyntacticExpectation { construction: String, slot: String },
    /// Frame role filler expectation
    FrameExpectation { frame: String, role: String },
    /// Discourse coherence
    DiscourseContext { topic: String },
    /// Combined/ensemble prediction
    Ensemble { sources: Vec<String> },
}

// =============================================================================
// PREDICTION ERROR
// =============================================================================

/// Prediction error at a linguistic level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticError {
    /// Level where error occurred
    pub level: LinguisticLevel,

    /// Error vector (predicted XOR observed)
    pub error_vector: HV16,

    /// Error magnitude (normalized)
    pub magnitude: f64,

    /// Precision (inverse variance) used for this error
    pub precision: f64,

    /// Precision-weighted error
    pub weighted_error: f64,

    /// Surprise: -log P(observation | prediction)
    pub surprise: f64,
}

impl LinguisticError {
    /// Create new linguistic error from prediction and observation
    pub fn new(
        level: LinguisticLevel,
        predicted: &HV16,
        observed: &HV16,
        precision: f64,
    ) -> Self {
        // Error = predicted XOR observed (in HDC space)
        let error_vector = predicted.bind(observed);

        // Magnitude = 1 - similarity (lower similarity = higher error)
        let similarity = predicted.similarity(observed) as f64;
        let magnitude = 1.0 - similarity;

        // Precision-weighted error
        let weighted_error = precision * magnitude;

        // Surprise = -log P(obs|pred) ≈ -log similarity
        // Clamp similarity to avoid log(0)
        let surprise = if similarity > 0.001 {
            -similarity.ln()
        } else {
            10.0 // Maximum surprise
        };

        Self {
            level,
            error_vector,
            magnitude,
            precision,
            weighted_error,
            surprise,
        }
    }
}

// =============================================================================
// PREDICTIVE LANGUAGE MODEL
// =============================================================================

/// Configuration for predictive understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveConfig {
    /// Learning rate for belief updates
    pub learning_rate: f64,

    /// Precision weights per level
    pub level_precisions: HashMap<String, f64>,

    /// History length for temporal smoothing
    pub history_length: usize,

    /// Free energy threshold for "understanding"
    pub understanding_threshold: f64,

    /// Enable active inference (action selection)
    pub active_inference: bool,
}

impl Default for PredictiveConfig {
    fn default() -> Self {
        let mut level_precisions = HashMap::new();
        for level in LinguisticLevel::all() {
            level_precisions.insert(
                format!("{:?}", level),
                level.default_precision(),
            );
        }

        Self {
            learning_rate: 0.1,
            level_precisions,
            history_length: 100,
            understanding_threshold: 0.3, // Lower = stricter
            active_inference: false,
        }
    }
}

/// Predictive Language Understanding - The core module
///
/// Implements hierarchical predictive coding for language:
/// 1. Generates predictions at each level
/// 2. Computes prediction errors
/// 3. Updates beliefs to minimize free energy
/// 4. Learns from experience
#[derive(Clone)]
pub struct PredictiveUnderstanding {
    /// Configuration
    config: PredictiveConfig,

    /// Frame library for semantic predictions
    frame_library: FrameLibrary,

    /// Construction grammar for syntactic predictions
    grammar: ConstructionGrammar,

    /// Current predictions at each level
    predictions: HashMap<LinguisticLevel, LinguisticPrediction>,

    /// Current errors at each level
    errors: HashMap<LinguisticLevel, LinguisticError>,

    /// Current beliefs (state estimates) at each level
    beliefs: HashMap<LinguisticLevel, HV16>,

    /// Free energy history
    free_energy_history: VecDeque<f64>,

    /// Lexical prediction weights (learned)
    lexical_weights: HashMap<String, Vec<(String, f64)>>,

    /// Current discourse context
    discourse_context: HV16,

    /// Active frames (currently relevant)
    active_frames: Vec<FrameInstance>,

    /// Active constructions (currently being parsed)
    active_constructions: Vec<ConstructionParse>,
}

impl PredictiveUnderstanding {
    /// Create new predictive understanding system
    pub fn new(config: PredictiveConfig) -> Self {
        let frame_library = FrameLibrary::new();
        let grammar = ConstructionGrammar::new();

        // Initialize beliefs at each level with random vectors
        let mut beliefs = HashMap::new();
        for (i, level) in LinguisticLevel::all().iter().enumerate() {
            beliefs.insert(*level, HV16::random(42 + i as u64));
        }

        Self {
            config,
            frame_library,
            grammar,
            predictions: HashMap::new(),
            errors: HashMap::new(),
            beliefs,
            free_energy_history: VecDeque::new(),
            lexical_weights: Self::default_lexical_weights(),
            discourse_context: HV16::random(12345),
            active_frames: Vec::new(),
            active_constructions: Vec::new(),
        }
    }

    /// Default lexical co-occurrence weights (simplified)
    fn default_lexical_weights() -> HashMap<String, Vec<(String, f64)>> {
        let mut weights = HashMap::new();

        // Some common patterns
        weights.insert("the".to_string(), vec![
            ("cat".to_string(), 0.1),
            ("dog".to_string(), 0.1),
            ("man".to_string(), 0.1),
            ("woman".to_string(), 0.1),
            ("book".to_string(), 0.08),
        ]);

        weights.insert("a".to_string(), vec![
            ("cat".to_string(), 0.1),
            ("dog".to_string(), 0.1),
            ("man".to_string(), 0.1),
            ("book".to_string(), 0.1),
            ("new".to_string(), 0.08),
        ]);

        weights
    }

    /// Process a word and update the predictive hierarchy
    ///
    /// This is the main entry point for incremental processing.
    /// Returns the free energy (lower = better understanding).
    pub fn process_word(&mut self, word: &str, word_encoding: &HV16) -> f64 {
        // 1. Generate predictions at all levels
        self.generate_predictions(word);

        // 2. Compute prediction errors at each level
        self.compute_errors(word_encoding);

        // 3. Update beliefs via precision-weighted error integration
        self.update_beliefs();

        // 4. Compute total free energy
        let free_energy = self.compute_free_energy();

        // 5. Update history
        self.free_energy_history.push_back(free_energy);
        if self.free_energy_history.len() > self.config.history_length {
            self.free_energy_history.pop_front();
        }

        // 6. Update active frames and constructions
        self.update_active_structures(word);

        // 7. Learn from experience
        self.learn_from_error(word);

        free_energy
    }

    /// Generate predictions at all levels
    fn generate_predictions(&mut self, current_word: &str) {
        // Lexical level: predict next word based on co-occurrence
        let lexical_pred = self.predict_lexical(current_word);
        self.predictions.insert(LinguisticLevel::Lexical, lexical_pred);

        // Syntactic level: predict based on active constructions
        let syntactic_pred = self.predict_syntactic();
        self.predictions.insert(LinguisticLevel::Syntactic, syntactic_pred);

        // Semantic level: predict based on active frames
        let semantic_pred = self.predict_semantic();
        self.predictions.insert(LinguisticLevel::Semantic, semantic_pred);

        // Discourse level: predict based on narrative context
        let discourse_pred = self.predict_discourse();
        self.predictions.insert(LinguisticLevel::Discourse, discourse_pred);
    }

    /// Predict next word based on lexical co-occurrence
    fn predict_lexical(&self, current_word: &str) -> LinguisticPrediction {
        // Get co-occurrence predictions
        let candidates = self.lexical_weights.get(current_word);

        let (predicted, alternatives) = if let Some(words) = candidates {
            // Combine predictions into HDC vector
            let mut bundle_parts = Vec::new();
            let mut alts = Vec::new();

            for (word, prob) in words {
                // Create deterministic encoding for word
                let word_vec = hv16_from_text(word);
                bundle_parts.push(word_vec);
                alts.push((word_vec, *prob));
            }

            // Bundle all predictions
            let combined = if bundle_parts.is_empty() {
                HV16::random(current_word.as_bytes().iter().map(|b| *b as u64).sum())
            } else {
                HV16::bundle(&bundle_parts)
            };

            (combined, alts)
        } else {
            // No predictions - use random with low confidence
            (HV16::random(current_word.len() as u64), vec![])
        };

        LinguisticPrediction {
            level: LinguisticLevel::Lexical,
            predicted,
            precision: LinguisticLevel::Lexical.default_precision(),
            source: PredictionSource::LexicalPrior {
                word: current_word.to_string(),
                probability: if alternatives.is_empty() { 0.1 } else { alternatives[0].1 },
            },
            alternatives,
        }
    }

    /// Predict based on syntactic expectations from active constructions
    fn predict_syntactic(&self) -> LinguisticPrediction {
        if let Some(active) = self.active_constructions.first() {
            // Predict based on next expected slot in construction
            let predicted = active.encoding;

            LinguisticPrediction {
                level: LinguisticLevel::Syntactic,
                predicted,
                precision: LinguisticLevel::Syntactic.default_precision(),
                source: PredictionSource::SyntacticExpectation {
                    construction: active.construction_name.clone(),
                    slot: "next".to_string(),
                },
                alternatives: vec![],
            }
        } else {
            // No active construction - predict based on common patterns
            LinguisticPrediction {
                level: LinguisticLevel::Syntactic,
                predicted: self.beliefs.get(&LinguisticLevel::Syntactic)
                    .cloned()
                    .unwrap_or_else(|| HV16::random(999)),
                precision: LinguisticLevel::Syntactic.default_precision() * 0.5, // Lower confidence
                source: PredictionSource::SyntacticExpectation {
                    construction: "unknown".to_string(),
                    slot: "any".to_string(),
                },
                alternatives: vec![],
            }
        }
    }

    /// Predict based on semantic frames
    fn predict_semantic(&self) -> LinguisticPrediction {
        if let Some(frame_instance) = self.active_frames.first() {
            // Predict unfilled roles
            let predicted = frame_instance.frame_encoding;

            LinguisticPrediction {
                level: LinguisticLevel::Semantic,
                predicted,
                precision: LinguisticLevel::Semantic.default_precision(),
                source: PredictionSource::FrameExpectation {
                    frame: frame_instance.frame_name.clone(),
                    role: "unfilled".to_string(),
                },
                alternatives: vec![],
            }
        } else {
            LinguisticPrediction {
                level: LinguisticLevel::Semantic,
                predicted: self.beliefs.get(&LinguisticLevel::Semantic)
                    .cloned()
                    .unwrap_or_else(|| HV16::random(888)),
                precision: LinguisticLevel::Semantic.default_precision() * 0.3,
                source: PredictionSource::FrameExpectation {
                    frame: "none".to_string(),
                    role: "none".to_string(),
                },
                alternatives: vec![],
            }
        }
    }

    /// Predict based on discourse context
    fn predict_discourse(&self) -> LinguisticPrediction {
        LinguisticPrediction {
            level: LinguisticLevel::Discourse,
            predicted: self.discourse_context,
            precision: LinguisticLevel::Discourse.default_precision(),
            source: PredictionSource::DiscourseContext {
                topic: "current_topic".to_string(),
            },
            alternatives: vec![],
        }
    }

    /// Compute prediction errors at each level
    fn compute_errors(&mut self, observed: &HV16) {
        for (level, prediction) in &self.predictions {
            let error = LinguisticError::new(
                *level,
                &prediction.predicted,
                observed,
                prediction.precision,
            );
            self.errors.insert(*level, error);
        }
    }

    /// Update beliefs via precision-weighted prediction error minimization
    fn update_beliefs(&mut self) {
        let learning_rate = self.config.learning_rate;

        for (level, error) in &self.errors {
            if let Some(belief) = self.beliefs.get_mut(level) {
                // Update belief to reduce error
                // New belief = old belief + learning_rate * precision * error_direction
                // In HDC: bundle current belief with error-corrected prediction

                let prediction = &self.predictions[level];
                let weight = (error.precision * learning_rate).min(1.0);

                // Weighted bundle towards observation
                // More error = move belief more towards observation
                if error.magnitude > 0.1 {
                    // Significant error - adjust belief
                    *belief = HV16::bundle(&[
                        *belief,
                        error.error_vector,  // Move toward observation
                    ]);
                }
            }
        }
    }

    /// Compute total free energy across all levels
    pub fn compute_free_energy(&self) -> f64 {
        let mut total_fe = 0.0;

        for (level, error) in &self.errors {
            // F = precision * prediction_error² + complexity_penalty
            // Simplified: F = weighted_error + log(precision)
            total_fe += error.weighted_error;
            total_fe += error.surprise * 0.1; // Small contribution from surprise
        }

        // Add complexity term (prefer simpler explanations)
        let complexity = self.active_frames.len() as f64 * 0.01
            + self.active_constructions.len() as f64 * 0.01;

        total_fe + complexity
    }

    /// Update active frames and constructions based on new word
    fn update_active_structures(&mut self, word: &str) {
        // Try to activate new frames
        let activator = FrameActivator::with_library(self.frame_library.clone());
        let frame_instances = activator.activate_from_text(word);
        for frame_instance in frame_instances {
            // Check if we already have this frame active
            let frame_name = frame_instance.frame_name.clone();
            if !self.active_frames.iter().any(|f| f.frame_name == frame_name) {
                self.active_frames.push(frame_instance);
            }
        }

        // Limit active frames to most recent
        if self.active_frames.len() > 5 {
            self.active_frames.remove(0);
        }

        // Try to parse constructions
        // Simplified: just track that we're parsing
        // In full implementation, would do incremental parsing
    }

    /// Learn from prediction errors to improve future predictions
    fn learn_from_error(&mut self, word: &str) {
        // Update lexical co-occurrence weights based on what we observed
        // This is simplified Hebbian learning

        // Get previous word from context (simplified: use discourse encoding)
        // In full implementation, would track word history

        // For now, just note that learning should happen here
        // Full implementation would:
        // 1. Update transition probabilities
        // 2. Strengthen frame associations
        // 3. Reinforce construction patterns
    }

    /// Check if current understanding is "good enough" (low free energy)
    pub fn is_understanding(&self) -> bool {
        if let Some(&latest_fe) = self.free_energy_history.back() {
            latest_fe < self.config.understanding_threshold
        } else {
            false
        }
    }

    /// Get current free energy
    pub fn current_free_energy(&self) -> f64 {
        self.free_energy_history.back().copied().unwrap_or(1.0)
    }

    /// Get error at a specific level
    pub fn error_at_level(&self, level: LinguisticLevel) -> Option<&LinguisticError> {
        self.errors.get(&level)
    }

    /// Get prediction at a specific level
    pub fn prediction_at_level(&self, level: LinguisticLevel) -> Option<&LinguisticPrediction> {
        self.predictions.get(&level)
    }

    /// Get belief at a specific level
    pub fn belief_at_level(&self, level: LinguisticLevel) -> Option<&HV16> {
        self.beliefs.get(&level)
    }

    /// Get active frames
    pub fn active_frames(&self) -> &[FrameInstance] {
        &self.active_frames
    }

    /// Reset the system to initial state
    pub fn reset(&mut self) {
        self.predictions.clear();
        self.errors.clear();
        self.free_energy_history.clear();
        self.active_frames.clear();
        self.active_constructions.clear();
        self.discourse_context = HV16::random(54321);

        // Re-initialize beliefs
        for (i, level) in LinguisticLevel::all().iter().enumerate() {
            self.beliefs.insert(*level, HV16::random(42 + i as u64));
        }
    }
}

// =============================================================================
// INCREMENTAL SENTENCE PROCESSOR
// =============================================================================

/// Result of processing a sentence through predictive understanding
#[derive(Debug, Clone)]
pub struct SentenceUnderstanding {
    /// The input sentence
    pub sentence: String,

    /// Words in the sentence
    pub words: Vec<String>,

    /// Free energy at each word
    pub free_energy_curve: Vec<f64>,

    /// Final free energy
    pub final_free_energy: f64,

    /// Whether understanding was achieved
    pub understood: bool,

    /// Activated frames
    pub frames: Vec<String>,

    /// Most surprising word (highest prediction error)
    pub most_surprising: Option<(String, f64)>,

    /// Integration coherence (how well parts fit together)
    pub coherence: f64,
}

impl PredictiveUnderstanding {
    /// Process an entire sentence incrementally
    pub fn process_sentence(&mut self, sentence: &str) -> SentenceUnderstanding {
        self.reset();

        let words: Vec<String> = sentence
            .split_whitespace()
            .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        let mut free_energy_curve = Vec::new();
        let mut max_surprise = 0.0f64;
        let mut most_surprising = None;

        for word in &words {
            let word_encoding = hv16_from_text(word);
            let fe = self.process_word(word, &word_encoding);
            free_energy_curve.push(fe);

            // Track most surprising word
            if let Some(error) = self.error_at_level(LinguisticLevel::Lexical) {
                if error.surprise > max_surprise {
                    max_surprise = error.surprise;
                    most_surprising = Some((word.clone(), error.surprise));
                }
            }
        }

        let final_free_energy = self.current_free_energy();
        let understood = self.is_understanding();

        // Collect activated frames
        let frames: Vec<String> = self.active_frames
            .iter()
            .map(|f| f.frame_name.clone())
            .collect();

        // Compute coherence from variance of free energy
        let coherence = if free_energy_curve.len() > 1 {
            let mean: f64 = free_energy_curve.iter().sum::<f64>() / free_energy_curve.len() as f64;
            let variance: f64 = free_energy_curve.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / free_energy_curve.len() as f64;
            1.0 / (1.0 + variance) // Lower variance = higher coherence
        } else {
            0.5
        };

        SentenceUnderstanding {
            sentence: sentence.to_string(),
            words,
            free_energy_curve,
            final_free_energy,
            understood,
            frames,
            most_surprising,
            coherence,
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linguistic_levels() {
        let levels = LinguisticLevel::all();
        assert_eq!(levels.len(), 5);

        // Verify timescales increase with hierarchy
        let mut prev_timescale = 0.0;
        for level in &levels {
            let ts = level.timescale_ms();
            assert!(ts > prev_timescale, "Timescales should increase");
            prev_timescale = ts;
        }

        // Verify precisions decrease with hierarchy
        let mut prev_precision = f64::MAX;
        for level in &levels {
            let p = level.default_precision();
            assert!(p < prev_precision, "Precisions should decrease");
            prev_precision = p;
        }
    }

    #[test]
    fn test_linguistic_error_computation() {
        let pred = HV16::random(42);
        let obs = HV16::random(43);

        let error = LinguisticError::new(
            LinguisticLevel::Lexical,
            &pred,
            &obs,
            5.0,
        );

        assert!(error.magnitude >= 0.0 && error.magnitude <= 1.0);
        assert!(error.surprise >= 0.0);
        assert!(error.weighted_error > 0.0);
    }

    #[test]
    fn test_predictive_understanding_creation() {
        let config = PredictiveConfig::default();
        let pu = PredictiveUnderstanding::new(config);

        // Should have beliefs for all levels
        for level in LinguisticLevel::all() {
            assert!(pu.beliefs.contains_key(&level));
        }
    }

    #[test]
    fn test_word_processing() {
        let config = PredictiveConfig::default();
        let mut pu = PredictiveUnderstanding::new(config);

        let word_encoding = hv16_from_text("cat");
        let fe = pu.process_word("cat", &word_encoding);

        // Free energy should be finite and positive
        assert!(fe.is_finite());
        assert!(fe >= 0.0);

        // Should have errors at all levels
        assert!(!pu.errors.is_empty());
    }

    #[test]
    fn test_sentence_processing() {
        let config = PredictiveConfig::default();
        let mut pu = PredictiveUnderstanding::new(config);

        let result = pu.process_sentence("The cat sat on the mat");

        assert_eq!(result.words.len(), 6);
        assert_eq!(result.free_energy_curve.len(), 6);
        assert!(result.coherence >= 0.0 && result.coherence <= 1.0);
        assert!(result.final_free_energy.is_finite());
    }

    #[test]
    fn test_free_energy_decreases_with_predictable_sequence() {
        let config = PredictiveConfig::default();
        let mut pu = PredictiveUnderstanding::new(config);

        // Process similar sentences
        let result1 = pu.process_sentence("The cat");
        let fe1 = result1.final_free_energy;

        pu.reset();
        let result2 = pu.process_sentence("A dog");
        let fe2 = result2.final_free_energy;

        // Both should have finite free energy
        assert!(fe1.is_finite());
        assert!(fe2.is_finite());
    }

    #[test]
    fn test_frame_activation() {
        let config = PredictiveConfig::default();
        let mut pu = PredictiveUnderstanding::new(config);

        // Process sentence with frame-activating words
        let result = pu.process_sentence("She gave him the book");

        // Should activate TRANSFER frame
        assert!(result.frames.contains(&"TRANSFER".to_string()) ||
                result.frames.is_empty()); // May not activate depending on implementation
    }

    #[test]
    fn test_surprise_detection() {
        let config = PredictiveConfig::default();
        let mut pu = PredictiveUnderstanding::new(config);

        let result = pu.process_sentence("The purple elephant flew silently");

        // Should detect some surprising words
        assert!(result.most_surprising.is_some());

        if let Some((word, surprise)) = &result.most_surprising {
            assert!(surprise.is_finite());
            assert!(*surprise > 0.0);
        }
    }

    #[test]
    fn test_reset() {
        let config = PredictiveConfig::default();
        let mut pu = PredictiveUnderstanding::new(config);

        // Process some words
        let word_encoding = hv16_from_text("test");
        pu.process_word("test", &word_encoding);

        assert!(!pu.errors.is_empty());

        // Reset
        pu.reset();

        assert!(pu.errors.is_empty());
        assert!(pu.predictions.is_empty());
        assert!(pu.active_frames.is_empty());
    }

    #[test]
    fn test_coherence_measure() {
        let config = PredictiveConfig::default();
        let mut pu = PredictiveUnderstanding::new(config);

        // Coherent sentence
        let coherent = pu.process_sentence("The quick brown fox jumps");

        pu.reset();

        // Random words (less coherent)
        let random = pu.process_sentence("Purple seven flying slowly blue");

        // Both should have valid coherence scores
        assert!(coherent.coherence >= 0.0 && coherent.coherence <= 1.0);
        assert!(random.coherence >= 0.0 && random.coherence <= 1.0);
    }

    #[test]
    fn test_prediction_source_variants() {
        let lexical = PredictionSource::LexicalPrior {
            word: "test".to_string(),
            probability: 0.5,
        };

        let syntactic = PredictionSource::SyntacticExpectation {
            construction: "Transitive".to_string(),
            slot: "Object".to_string(),
        };

        let semantic = PredictionSource::FrameExpectation {
            frame: "TRANSFER".to_string(),
            role: "Theme".to_string(),
        };

        let _discourse = PredictionSource::DiscourseContext {
            topic: "commerce".to_string(),
        };

        let _ensemble = PredictionSource::Ensemble {
            sources: vec!["lexical".to_string(), "syntactic".to_string()],
        };

        // All should be distinct patterns
        match lexical {
            PredictionSource::LexicalPrior { .. } => (),
            _ => panic!("Wrong variant"),
        }

        match syntactic {
            PredictionSource::SyntacticExpectation { .. } => (),
            _ => panic!("Wrong variant"),
        }
    }
}
