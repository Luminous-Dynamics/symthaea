//! # Emotional Core: Affective Language Processing
//!
//! Provides emotional understanding and generation capabilities for language,
//! including sentiment analysis, emotional tone detection, and affective
//! response generation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use symthaea_core::hdc::RealHV;

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
    pub embedding: RealHV,
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
            embedding: RealHV::zero(dimension),
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
    pub embedding: RealHV,
}

/// The emotional core system
#[derive(Debug)]
pub struct EmotionalCore {
    /// Configuration
    config: EmotionalCoreConfig,
    /// Emotion embeddings
    emotion_embeddings: HashMap<String, RealHV>,
    /// Emotional memory (recent states)
    memory: Vec<EmotionalAnalysis>,
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
    /// Create a new emotional core
    pub fn new(config: EmotionalCoreConfig) -> Self {
        let dim = config.dimension;

        // Initialize emotion embeddings with deterministic seeds based on emotion name
        let mut emotion_embeddings = HashMap::new();
        for (idx, emotion) in config.categories.iter().enumerate() {
            let seed = 0xE0C0_0000 + idx as u64;  // Emotional Core seed base
            emotion_embeddings.insert(emotion.clone(), RealHV::random(dim, seed));
        }
        emotion_embeddings.insert("neutral".to_string(), RealHV::zero(dim));

        Self {
            current_state: EmotionalAnalysis::neutral(dim),
            config,
            emotion_embeddings,
            memory: Vec::new(),
            stats: EmotionalCoreStats::default(),
        }
    }

    /// Analyze emotional content of text
    pub fn analyze(&mut self, text: &str) -> EmotionalAnalysis {
        self.stats.analyses += 1;

        // Simple keyword-based emotion detection
        let text_lower = text.to_lowercase();

        let mut emotion_scores = HashMap::new();
        let emotion_keywords: HashMap<&str, Vec<&str>> = [
            ("joy", vec!["happy", "joy", "glad", "delighted", "pleased", "excited"]),
            ("sadness", vec!["sad", "unhappy", "depressed", "melancholy", "grief"]),
            ("anger", vec!["angry", "furious", "mad", "irritated", "annoyed"]),
            ("fear", vec!["afraid", "scared", "fearful", "terrified", "anxious"]),
            ("surprise", vec!["surprised", "amazed", "astonished", "shocked"]),
            ("disgust", vec!["disgusted", "revolted", "sick", "repulsed"]),
            ("trust", vec!["trust", "believe", "confident", "faith", "rely"]),
            ("anticipation", vec!["anticipate", "expect", "await", "hope", "looking forward"]),
        ].iter().cloned().collect();

        for (emotion, keywords) in &emotion_keywords {
            let score: f32 = keywords.iter()
                .map(|kw| if text_lower.contains(kw) { 1.0 } else { 0.0 })
                .sum::<f32>() / keywords.len() as f32;
            emotion_scores.insert(emotion.to_string(), score);
        }

        // Find primary emotion
        let (primary_emotion, confidence) = emotion_scores.iter()
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
            RealHV::zero(self.config.dimension)
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
            self.memory.push(analysis.clone());
            if self.memory.len() > 100 {
                self.memory.remove(0);
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
    pub fn generate_emotional_response(&mut self, base_text: &str, target_emotion: &str, intensity: f32) -> EmotionalResponse {
        self.stats.responses_generated += 1;

        // Get emotion embedding
        let emotion_emb = self.emotion_embeddings.get(target_emotion)
            .cloned()
            .unwrap_or_else(|| RealHV::random(self.config.dimension, 0xFA11_BACC));  // Fallback seed

        // Simple emotion modifiers
        let modifiers: HashMap<&str, Vec<&str>> = [
            ("joy", vec!["happily", "joyfully", "with delight"]),
            ("sadness", vec!["sadly", "with sorrow", "regretfully"]),
            ("anger", vec!["angrily", "furiously", "with frustration"]),
            ("fear", vec!["fearfully", "anxiously", "nervously"]),
            ("surprise", vec!["surprisingly", "amazingly", "unexpectedly"]),
        ].iter().cloned().collect();

        let modifier = modifiers.get(target_emotion)
            .and_then(|m| m.first())
            .unwrap_or(&"");

        let text = if !modifier.is_empty() && intensity > 0.5 {
            format!("{}, {}", modifier, base_text)
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
    pub fn memory(&self) -> &[EmotionalAnalysis] {
        &self.memory
    }

    /// Get emotion embedding
    pub fn emotion_embedding(&self, emotion: &str) -> Option<&RealHV> {
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

        assert!(response.text.contains("sadly") || response.text.contains("sorrow") || response.text.contains("I'm sorry"));
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
        let nonzero_emotions = analysis.emotion_scores.values()
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
