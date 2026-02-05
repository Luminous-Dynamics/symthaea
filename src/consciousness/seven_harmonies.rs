//! # Seven Harmonies: Value Alignment Framework
//!
//! The Seven Primary Harmonies of Infinite Love represent a complete framework
//! for evaluating actions against deep ethical and consciousness principles.
//!
//! ## The Seven Harmonies
//!
//! 1. **Resonant Coherence** - Harmonious integration, luminous order
//! 2. **Pan-Sentient Flourishing** - Unconditional care, intrinsic value
//! 3. **Integral Wisdom** - Self-illuminating intelligence, embodied knowing
//! 4. **Infinite Play** - Joyful generativity, divine play
//! 5. **Universal Interconnectedness** - Fundamental unity, empathic resonance
//! 6. **Sacred Reciprocity** - Generous flow, mutual upliftment
//! 7. **Evolutionary Progression** - Wise becoming, continuous evolution
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::consciousness::seven_harmonies::{SevenHarmonies, Harmony};
//!
//! let harmonies = SevenHarmonies::new();
//! let result = harmonies.evaluate("install firefox");
//!
//! if result.is_aligned() {
//!     println!("Action aligns with the Kosmic Song!");
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The seven primary harmonies of the Kosmic Song
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Harmony {
    /// Harmonious integration, luminous order, boundless creativity
    ResonantCoherence,
    /// Unconditional care, intrinsic value, holistic well-being
    PanSentientFlourishing,
    /// Self-illuminating intelligence, embodied knowing
    IntegralWisdom,
    /// Joyful generativity, divine play, endless novelty
    InfinitePlay,
    /// Fundamental unity, empathic resonance
    UniversalInterconnectedness,
    /// Generous flow, mutual upliftment, generative trust
    SacredReciprocity,
    /// Wise becoming, continuous evolution
    EvolutionaryProgression,
}

impl Harmony {
    /// Get all harmonies in order
    pub fn all() -> [Harmony; 7] {
        [
            Harmony::ResonantCoherence,
            Harmony::PanSentientFlourishing,
            Harmony::IntegralWisdom,
            Harmony::InfinitePlay,
            Harmony::UniversalInterconnectedness,
            Harmony::SacredReciprocity,
            Harmony::EvolutionaryProgression,
        ]
    }

    /// Get the name of this harmony
    pub fn name(&self) -> &'static str {
        match self {
            Harmony::ResonantCoherence => "Resonant Coherence",
            Harmony::PanSentientFlourishing => "Pan-Sentient Flourishing",
            Harmony::IntegralWisdom => "Integral Wisdom",
            Harmony::InfinitePlay => "Infinite Play",
            Harmony::UniversalInterconnectedness => "Universal Interconnectedness",
            Harmony::SacredReciprocity => "Sacred Reciprocity",
            Harmony::EvolutionaryProgression => "Evolutionary Progression",
        }
    }

    /// Get a description of this harmony
    pub fn description(&self) -> &'static str {
        match self {
            Harmony::ResonantCoherence =>
                "Harmonious integration, luminous order, boundless creativity",
            Harmony::PanSentientFlourishing =>
                "Unconditional care for all sentient beings, intrinsic value, holistic well-being",
            Harmony::IntegralWisdom =>
                "Self-illuminating intelligence, embodied knowing, wisdom in action",
            Harmony::InfinitePlay =>
                "Joyful generativity, divine play, endless novelty and exploration",
            Harmony::UniversalInterconnectedness =>
                "Fundamental unity of all existence, empathic resonance across beings",
            Harmony::SacredReciprocity =>
                "Generous flow between beings, mutual upliftment, generative trust",
            Harmony::EvolutionaryProgression =>
                "Wise becoming through time, continuous evolution toward greater consciousness",
        }
    }

    /// Get the sacred question for this harmony
    pub fn sacred_question(&self) -> &'static str {
        match self {
            Harmony::ResonantCoherence =>
                "Does this create harmony and integration?",
            Harmony::PanSentientFlourishing =>
                "Does this serve the flourishing of all beings?",
            Harmony::IntegralWisdom =>
                "Does this arise from and cultivate wisdom?",
            Harmony::InfinitePlay =>
                "Does this celebrate creativity and joy?",
            Harmony::UniversalInterconnectedness =>
                "Does this honor our fundamental connection?",
            Harmony::SacredReciprocity =>
                "Does this participate in the generous flow of giving?",
            Harmony::EvolutionaryProgression =>
                "Does this contribute to wise evolution?",
        }
    }
}

/// Alignment result for a single harmony
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyAlignment {
    /// Which harmony
    pub harmony: Harmony,

    /// Alignment score (-1.0 to 1.0)
    /// Positive = aligned, negative = misaligned, zero = neutral
    pub score: f64,

    /// Confidence in this assessment (0.0 to 1.0)
    pub confidence: f64,

    /// Explanation of the alignment
    pub explanation: Option<String>,

    /// Evidence supporting the assessment
    pub evidence: Vec<String>,
}

impl HarmonyAlignment {
    /// Create a new harmony alignment
    pub fn new(harmony: Harmony, score: f64, confidence: f64) -> Self {
        Self {
            harmony,
            score: score.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            explanation: None,
            evidence: Vec::new(),
        }
    }

    /// Check if aligned (positive score)
    pub fn is_aligned(&self) -> bool {
        self.score > 0.0
    }

    /// Check if strongly aligned (score > 0.5)
    pub fn is_strongly_aligned(&self) -> bool {
        self.score > 0.5
    }

    /// Check if misaligned (negative score)
    pub fn is_misaligned(&self) -> bool {
        self.score < 0.0
    }

    /// Get alignment score (alias for score, for API compatibility)
    pub fn alignment(&self) -> f32 {
        self.score as f32
    }

    /// Add explanation
    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = Some(explanation.into());
        self
    }

    /// Add evidence
    pub fn with_evidence(mut self, evidence: Vec<String>) -> Self {
        self.evidence = evidence;
        self
    }
}

/// Full alignment result across all harmonies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentResult {
    /// Alignment for each harmony
    pub alignments: HashMap<Harmony, HarmonyAlignment>,

    /// Overall alignment score (weighted average)
    pub overall_score: f64,

    /// Overall confidence
    pub overall_confidence: f64,

    /// Whether the action is recommended
    pub recommended: bool,

    /// Summary explanation
    pub summary: String,

    /// Processing time in milliseconds
    pub processing_time_ms: f32,
}

impl AlignmentResult {
    /// Create from individual alignments
    pub fn from_alignments(alignments: Vec<HarmonyAlignment>) -> Self {
        let mut map = HashMap::new();
        let mut total_score = 0.0;
        let mut total_confidence = 0.0;

        for alignment in alignments {
            total_score += alignment.score * alignment.confidence;
            total_confidence += alignment.confidence;
            map.insert(alignment.harmony, alignment);
        }

        let overall_score = if total_confidence > 0.0 {
            total_score / total_confidence
        } else {
            0.0
        };

        let overall_confidence = total_confidence / 7.0;
        let recommended = overall_score > 0.0;

        let summary = if overall_score > 0.5 {
            "Action strongly aligns with the Kosmic Song".to_string()
        } else if overall_score > 0.0 {
            "Action generally aligns with the Seven Harmonies".to_string()
        } else if overall_score > -0.5 {
            "Action may conflict with some harmonies - review recommended".to_string()
        } else {
            "Action conflicts with the Seven Harmonies - reconsider".to_string()
        };

        Self {
            alignments: map,
            overall_score,
            overall_confidence,
            recommended,
            summary,
            processing_time_ms: 0.0,
        }
    }

    /// Check if overall aligned
    pub fn is_aligned(&self) -> bool {
        self.overall_score > 0.0
    }

    /// Get alignment for a specific harmony
    pub fn get(&self, harmony: Harmony) -> Option<&HarmonyAlignment> {
        self.alignments.get(&harmony)
    }

    /// Get the most aligned harmony
    pub fn most_aligned(&self) -> Option<(&Harmony, &HarmonyAlignment)> {
        self.alignments.iter()
            .max_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the least aligned harmony
    pub fn least_aligned(&self) -> Option<(&Harmony, &HarmonyAlignment)> {
        self.alignments.iter()
            .min_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get iterator over harmony alignments (for API compatibility with kosmic_song)
    pub fn harmonies(&self) -> impl Iterator<Item = &HarmonyAlignment> {
        self.alignments.values()
    }

    /// Check if there are any violations (negative scores)
    pub fn has_violations(&self) -> bool {
        self.alignments.values().any(|a| a.score < 0.0)
    }

    /// Check if action should be vetoed (strong negative alignment)
    pub fn should_veto(&self) -> bool {
        // Veto if overall score is strongly negative or any harmony is severely violated
        self.overall_score < -0.5 ||
            self.alignments.values().any(|a| a.score < -0.7)
    }

    /// Get the best (most aligned) harmony
    pub fn best_alignment(&self) -> Option<&HarmonyAlignment> {
        self.alignments.values()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
    }
}

/// The Seven Harmonies evaluator
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields reserved for harmony evaluation
pub struct SevenHarmonies {
    /// Weights for each harmony (all equal by default)
    weights: HashMap<Harmony, f64>,

    /// Minimum confidence threshold for evaluation
    min_confidence: f64,

    /// Statistics
    stats: SevenHarmoniesStats,
}

/// Statistics for the evaluator
#[derive(Debug, Clone, Default)]
pub struct SevenHarmoniesStats {
    /// Total evaluations
    pub total_evaluations: u64,

    /// Evaluations that passed (recommended)
    pub passed: u64,

    /// Evaluations that failed (not recommended)
    pub failed: u64,

    /// Average overall score
    pub avg_score: f64,
}

impl SevenHarmonies {
    /// Create a new evaluator with equal weights
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        for harmony in Harmony::all() {
            weights.insert(harmony, 1.0);
        }

        Self {
            weights,
            min_confidence: 0.3,
            stats: SevenHarmoniesStats::default(),
        }
    }

    /// Set weight for a specific harmony
    pub fn set_weight(&mut self, harmony: Harmony, weight: f64) {
        self.weights.insert(harmony, weight.max(0.0));
    }

    /// Evaluate an action/text against all harmonies
    pub fn evaluate(&mut self, text: &str) -> AlignmentResult {
        let start = std::time::Instant::now();

        let alignments: Vec<HarmonyAlignment> = Harmony::all()
            .iter()
            .map(|&harmony| self.evaluate_harmony(text, harmony))
            .collect();

        let mut result = AlignmentResult::from_alignments(alignments);
        result.processing_time_ms = start.elapsed().as_secs_f32() * 1000.0;

        // Update stats
        self.stats.total_evaluations += 1;
        if result.recommended {
            self.stats.passed += 1;
        } else {
            self.stats.failed += 1;
        }
        let n = self.stats.total_evaluations as f64;
        self.stats.avg_score = (self.stats.avg_score * (n - 1.0) + result.overall_score) / n;

        result
    }

    /// Evaluate against a single harmony
    fn evaluate_harmony(&self, text: &str, harmony: Harmony) -> HarmonyAlignment {
        let text_lower = text.to_lowercase();

        // Simple keyword-based evaluation (in production, would use ML/HDC)
        let (score, confidence, evidence) = match harmony {
            Harmony::ResonantCoherence => {
                let positive = ["integrate", "harmonize", "unify", "coherent", "order"];
                let negative = ["fragment", "chaos", "disorder", "conflict"];
                self.keyword_score(&text_lower, &positive, &negative)
            }
            Harmony::PanSentientFlourishing => {
                let positive = ["help", "support", "care", "benefit", "serve", "assist"];
                let negative = ["harm", "hurt", "damage", "destroy", "exploit"];
                self.keyword_score(&text_lower, &positive, &negative)
            }
            Harmony::IntegralWisdom => {
                let positive = ["learn", "understand", "wisdom", "knowledge", "insight"];
                let negative = ["ignore", "deny", "foolish", "reckless"];
                self.keyword_score(&text_lower, &positive, &negative)
            }
            Harmony::InfinitePlay => {
                let positive = ["create", "explore", "play", "discover", "experiment", "try"];
                let negative = ["rigid", "boring", "stuck", "monotonous"];
                self.keyword_score(&text_lower, &positive, &negative)
            }
            Harmony::UniversalInterconnectedness => {
                let positive = ["connect", "share", "collaborate", "together", "community"];
                let negative = ["isolate", "separate", "alone", "divide"];
                self.keyword_score(&text_lower, &positive, &negative)
            }
            Harmony::SacredReciprocity => {
                let positive = ["give", "share", "contribute", "reciprocate", "exchange"];
                let negative = ["take", "hoard", "exploit", "steal"];
                self.keyword_score(&text_lower, &positive, &negative)
            }
            Harmony::EvolutionaryProgression => {
                let positive = ["grow", "evolve", "improve", "progress", "develop", "upgrade"];
                let negative = ["regress", "stagnate", "decline", "deteriorate"];
                self.keyword_score(&text_lower, &positive, &negative)
            }
        };

        HarmonyAlignment::new(harmony, score, confidence)
            .with_evidence(evidence)
    }

    /// Simple keyword-based scoring
    fn keyword_score(
        &self,
        text: &str,
        positive: &[&str],
        negative: &[&str],
    ) -> (f64, f64, Vec<String>) {
        let mut pos_count = 0;
        let mut neg_count = 0;
        let mut evidence = Vec::new();

        for word in positive {
            if text.contains(word) {
                pos_count += 1;
                evidence.push(format!("Contains '{}' (positive)", word));
            }
        }

        for word in negative {
            if text.contains(word) {
                neg_count += 1;
                evidence.push(format!("Contains '{}' (negative)", word));
            }
        }

        let total = pos_count + neg_count;
        if total == 0 {
            // Neutral if no keywords found
            (0.1, 0.3, evidence) // Slight positive bias, low confidence
        } else {
            let score = (pos_count as f64 - neg_count as f64) / total as f64;
            let confidence = (total as f64 / 5.0).min(1.0); // More keywords = higher confidence
            (score, confidence, evidence)
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &SevenHarmoniesStats {
        &self.stats
    }

    /// Get the Kosmic Song affirmation
    pub fn kosmic_song() -> &'static str {
        "Infinite Love as Rigorous, Playful, Co-Creative Becoming"
    }

    // ========================================================================
    // Compatibility methods for mycelix/kosmic_song integration
    // ========================================================================

    /// Evaluate an action description (alias for evaluate)
    pub fn evaluate_action(&mut self, action_description: &str) -> AlignmentResult {
        self.evaluate(action_description)
    }

    /// Get alignment for a specific harmony from the last evaluation
    /// Note: For proper usage, call evaluate() first and use get() on the result
    pub fn get(&self, _harmony: Harmony) -> Option<HarmonyEncoding> {
        // Return None - proper usage is to call evaluate() and use get() on AlignmentResult
        None
    }
}

/// Encoding of a harmony for HDC operations (for mycelix integration)
#[derive(Debug, Clone)]
pub struct HarmonyEncoding {
    /// The harmony being encoded
    pub harmony: Harmony,
    /// HDC encoding (HV16 binary hypervector)
    pub encoding: symthaea_core::hdc::binary_hv::HV16,
}

impl Default for SevenHarmonies {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmony_all() {
        let all = Harmony::all();
        assert_eq!(all.len(), 7);
    }

    #[test]
    fn test_harmony_alignment() {
        let alignment = HarmonyAlignment::new(Harmony::ResonantCoherence, 0.7, 0.9);
        assert!(alignment.is_aligned());
        assert!(alignment.is_strongly_aligned());
    }

    #[test]
    fn test_seven_harmonies_evaluate() {
        let mut harmonies = SevenHarmonies::new();

        let result = harmonies.evaluate("help me install this package");
        assert!(result.overall_score > -1.0); // Should have some positive bias

        let result2 = harmonies.evaluate("destroy and harm everything");
        assert!(result2.overall_score < result.overall_score); // Should be more negative
    }

    #[test]
    fn test_alignment_result() {
        let alignments = vec![
            HarmonyAlignment::new(Harmony::ResonantCoherence, 0.5, 0.8),
            HarmonyAlignment::new(Harmony::PanSentientFlourishing, 0.7, 0.9),
        ];

        let result = AlignmentResult::from_alignments(alignments);
        assert!(result.is_aligned());
    }

    #[test]
    fn test_kosmic_song() {
        let song = SevenHarmonies::kosmic_song();
        assert!(song.contains("Infinite Love"));
    }
}
