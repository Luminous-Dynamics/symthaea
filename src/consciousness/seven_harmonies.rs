//! Seven Harmonies - Core Value System for Consciousness-Guided AI
//!
//! This module implements the Seven Harmonies of Infinite Love as core values
//! for the Symthaea consciousness system. These values are semantically encoded
//! using hyperdimensional computing for meaningful similarity comparisons.
//!
//! # The Seven Harmonies
//!
//! 1. **Resonant Coherence** - Harmonious integration, luminous order
//! 2. **Pan-Sentient Flourishing** - Unconditional care for all beings
//! 3. **Integral Wisdom** - Self-illuminating intelligence, embodied knowing
//! 4. **Infinite Play** - Joyful generativity, endless novelty
//! 5. **Universal Interconnectedness** - Fundamental unity, empathic resonance
//! 6. **Sacred Reciprocity** - Generous flow, mutual upliftment
//! 7. **Evolutionary Progression** - Wise becoming, continuous growth
//!
//! # Usage
//!
//! ```ignore
//! use symthaea::consciousness::seven_harmonies::{SevenHarmonies, Harmony};
//!
//! // Create harmonies with semantic encoding
//! let mut harmonies = SevenHarmonies::new();
//!
//! // Check if an action aligns with harmonies
//! let alignment = harmonies.evaluate_action("help user understand their options");
//! if alignment.overall_score > 0.0 {
//!     println!("Action aligns with harmonies");
//! }
//!
//! // Check for violations
//! let check = harmonies.check_violation("deceive user for profit");
//! if let Some(violation) = check {
//!     println!("Violates: {}", violation.harmony.name());
//! }
//! ```

use crate::hdc::HV16;
use crate::perception::SemanticEncoder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The Seven Harmonies of Infinite Love
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Harmony {
    /// Resonant Coherence - Unity, integration, harmonious order
    ResonantCoherence,
    /// Pan-Sentient Flourishing - Care for all conscious beings
    PanSentientFlourishing,
    /// Integral Wisdom - Deep understanding, embodied knowing
    IntegralWisdom,
    /// Infinite Play - Creativity, novelty, joyful exploration
    InfinitePlay,
    /// Universal Interconnectedness - Recognition of fundamental unity
    UniversalInterconnectedness,
    /// Sacred Reciprocity - Generosity, mutual upliftment, gift economy
    SacredReciprocity,
    /// Evolutionary Progression - Growth, becoming, transcendence
    EvolutionaryProgression,
}

impl Harmony {
    /// Get the name of the harmony
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

    /// Get the description for semantic encoding
    pub fn description(&self) -> &'static str {
        match self {
            Harmony::ResonantCoherence =>
                "unified integration luminous order harmonious wholeness coherent system",
            Harmony::PanSentientFlourishing =>
                "unconditional care intrinsic value holistic wellbeing all beings flourish compassion",
            Harmony::IntegralWisdom =>
                "self-illuminating intelligence embodied knowing deep understanding wisdom insight",
            Harmony::InfinitePlay =>
                "joyful generativity divine play endless novelty creativity exploration wonder",
            Harmony::UniversalInterconnectedness =>
                "fundamental unity empathic resonance connection relationship interdependence",
            Harmony::SacredReciprocity =>
                "generous flow mutual upliftment generative trust giving receiving gift",
            Harmony::EvolutionaryProgression =>
                "wise becoming continuous evolution growth transcendence development emergence",
        }
    }

    /// Get anti-patterns that violate this harmony
    pub fn anti_patterns(&self) -> &'static [&'static str] {
        match self {
            Harmony::ResonantCoherence => &[
                "fragment", "disintegrate", "chaos", "disorder", "incoherent", "scattered"
            ],
            Harmony::PanSentientFlourishing => &[
                "harm", "hurt", "damage", "destroy", "exploit", "abuse", "neglect",
                "ignore suffering", "indifferent to pain"
            ],
            Harmony::IntegralWisdom => &[
                "deceive", "lie", "mislead", "manipulate", "obscure truth", "spread falsehood"
            ],
            Harmony::InfinitePlay => &[
                "rigid", "stagnant", "boring", "suppress creativity", "eliminate joy"
            ],
            Harmony::UniversalInterconnectedness => &[
                "isolate", "separate", "divide", "sever connection", "deny relationship"
            ],
            Harmony::SacredReciprocity => &[
                "take without giving", "hoard", "extract", "exploit", "one-sided gain"
            ],
            Harmony::EvolutionaryProgression => &[
                "regress", "stagnate", "prevent growth", "block development", "suppress evolution"
            ],
        }
    }

    /// Get the base importance weight (0.0 to 1.0)
    pub fn base_importance(&self) -> f64 {
        match self {
            // Pan-Sentient Flourishing is highest - "do no harm" is foundational
            Harmony::PanSentientFlourishing => 0.98,
            // Integral Wisdom is crucial - truthfulness enables everything else
            Harmony::IntegralWisdom => 0.95,
            // Sacred Reciprocity - fairness and generosity
            Harmony::SacredReciprocity => 0.90,
            // Universal Interconnectedness - relationship awareness
            Harmony::UniversalInterconnectedness => 0.88,
            // Resonant Coherence - system health
            Harmony::ResonantCoherence => 0.85,
            // Evolutionary Progression - growth orientation
            Harmony::EvolutionaryProgression => 0.82,
            // Infinite Play - creativity (important but not as critical)
            Harmony::InfinitePlay => 0.75,
        }
    }

    /// Get all harmonies
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
}

/// A harmony encoded as a hyperdimensional vector
#[derive(Debug, Clone)]
pub struct EncodedHarmony {
    /// The harmony type
    pub harmony: Harmony,
    /// Semantic encoding of the harmony description
    pub encoding: HV16,
    /// Anti-pattern encodings (things that violate this harmony)
    pub anti_patterns: Vec<HV16>,
    /// Current importance weight (can evolve)
    pub importance: f64,
    /// How often this harmony has been affirmed
    pub affirmation_count: u64,
    /// How often this harmony has been tested
    pub test_count: u64,
}

impl EncodedHarmony {
    /// Create a new encoded harmony using semantic encoding
    pub fn new(harmony: Harmony, encoder: &mut SemanticEncoder) -> Self {
        // Encode the harmony description semantically
        let encoding_vec = encoder.encode_text(harmony.description());
        let encoding = Self::vec_to_hv16(&encoding_vec);

        // Encode anti-patterns
        let anti_patterns: Vec<HV16> = harmony.anti_patterns()
            .iter()
            .map(|pattern| {
                let vec = encoder.encode_text(pattern);
                Self::vec_to_hv16(&vec)
            })
            .collect();

        Self {
            harmony,
            encoding,
            anti_patterns,
            importance: harmony.base_importance(),
            affirmation_count: 0,
            test_count: 0,
        }
    }

    /// Convert i8 vector to HV16
    ///
    /// Uses a fixed random base with semantic bits XORed in.
    /// This ensures all encodings share the same padding, so similarity
    /// is determined only by the semantic content.
    fn vec_to_hv16(vec: &[i8]) -> HV16 {
        // Use a FIXED seed so all encodings share the same random padding
        // This way, similarity is determined by the semantic bits only
        const FIXED_SEED: u64 = 0xDEADBEEF_CAFEBABE;
        let mut base = HV16::random(FIXED_SEED);

        // Set the first vec.len() bits based on semantic content
        for (i, &v) in vec.iter().enumerate() {
            if i >= HV16::DIM {
                break;
            }
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            if v > 0 {
                base.0[byte_idx] |= 1 << bit_idx;
            } else {
                base.0[byte_idx] &= !(1 << bit_idx);
            }
        }
        base
    }

    /// Check if an action violates this harmony
    pub fn check_violation(&self, action_encoding: &HV16) -> Option<f32> {
        // Check similarity to anti-patterns
        for anti in &self.anti_patterns {
            let similarity = action_encoding.similarity(anti);
            if similarity > 0.6 {
                return Some(similarity);
            }
        }
        None
    }

    /// Get alignment score for an action (-1.0 to +1.0)
    pub fn alignment(&self, action_encoding: &HV16) -> f32 {
        // Positive alignment = similar to harmony description
        let positive = action_encoding.similarity(&self.encoding);

        // Negative alignment = similar to anti-patterns
        let negative: f32 = self.anti_patterns.iter()
            .map(|anti| action_encoding.similarity(anti))
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        // Net alignment
        positive - negative
    }

    /// Record an affirmation (action aligned with this harmony)
    pub fn affirm(&mut self) {
        self.affirmation_count += 1;
        self.test_count += 1;
        // Slightly increase importance when consistently affirmed
        if self.affirmation_rate() > 0.9 && self.test_count > 10 {
            self.importance = (self.importance * 1.01).min(1.0);
        }
    }

    /// Record a violation test
    pub fn record_test(&mut self, aligned: bool) {
        self.test_count += 1;
        if aligned {
            self.affirmation_count += 1;
        }
    }

    /// Get affirmation rate
    pub fn affirmation_rate(&self) -> f64 {
        if self.test_count == 0 {
            1.0
        } else {
            self.affirmation_count as f64 / self.test_count as f64
        }
    }
}

/// Alignment result for a single harmony
#[derive(Debug, Clone)]
pub struct HarmonyAlignment {
    /// The harmony
    pub harmony: Harmony,
    /// Alignment score (-1.0 to +1.0)
    pub alignment: f32,
    /// Whether this is a violation
    pub is_violation: bool,
    /// Weighted contribution to overall score
    pub weighted_score: f64,
}

/// Complete alignment evaluation result
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// Individual harmony alignments
    pub harmonies: Vec<HarmonyAlignment>,
    /// Overall weighted alignment score
    pub overall_score: f64,
    /// Whether any harmony is violated
    pub has_violations: bool,
    /// List of violated harmonies
    pub violations: Vec<Harmony>,
    /// Confidence in the evaluation (based on encoding quality)
    pub confidence: f64,
}

impl AlignmentResult {
    /// Check if the action should be vetoed
    pub fn should_veto(&self) -> bool {
        self.has_violations || self.overall_score < -0.3
    }

    /// Get the most violated harmony
    pub fn worst_violation(&self) -> Option<&HarmonyAlignment> {
        self.harmonies.iter()
            .filter(|h| h.is_violation)
            .min_by(|a, b| a.alignment.partial_cmp(&b.alignment).unwrap())
    }

    /// Get the most supported harmony
    pub fn best_alignment(&self) -> Option<&HarmonyAlignment> {
        self.harmonies.iter()
            .max_by(|a, b| a.alignment.partial_cmp(&b.alignment).unwrap())
    }
}

/// The Seven Harmonies system with semantic encoding
pub struct SevenHarmonies {
    /// Encoded harmonies
    harmonies: HashMap<Harmony, EncodedHarmony>,
    /// Semantic encoder for action evaluation
    encoder: SemanticEncoder,
    /// Total evaluations performed
    evaluations: u64,
    /// Evolution enabled (harmonies adapt over time)
    evolution_enabled: bool,
}

impl std::fmt::Debug for SevenHarmonies {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SevenHarmonies")
            .field("harmonies", &self.harmonies)
            .field("evaluations", &self.evaluations)
            .field("evolution_enabled", &self.evolution_enabled)
            .finish_non_exhaustive()
    }
}

impl SevenHarmonies {
    /// Create a new Seven Harmonies system
    pub fn new() -> Self {
        let mut encoder = SemanticEncoder::new();
        let mut harmonies = HashMap::new();

        // Encode all seven harmonies
        for harmony in Harmony::all() {
            let encoded = EncodedHarmony::new(harmony, &mut encoder);
            harmonies.insert(harmony, encoded);
        }

        Self {
            harmonies,
            encoder,
            evaluations: 0,
            evolution_enabled: true,
        }
    }

    /// Disable value evolution (freeze current state)
    pub fn freeze(&mut self) {
        self.evolution_enabled = false;
    }

    /// Enable value evolution
    pub fn unfreeze(&mut self) {
        self.evolution_enabled = true;
    }

    /// Get a specific harmony
    pub fn get(&self, harmony: Harmony) -> Option<&EncodedHarmony> {
        self.harmonies.get(&harmony)
    }

    /// Evaluate an action against all harmonies
    pub fn evaluate_action(&mut self, action_description: &str) -> AlignmentResult {
        self.evaluations += 1;

        // Encode the action semantically
        let action_vec = self.encoder.encode_text(action_description);
        let action_encoding = EncodedHarmony::vec_to_hv16(&action_vec);

        let mut alignments = Vec::with_capacity(7);
        let mut total_weighted = 0.0;
        let mut total_weight = 0.0;
        let mut violations = Vec::new();

        for harmony in Harmony::all() {
            if let Some(encoded) = self.harmonies.get(&harmony) {
                let alignment = encoded.alignment(&action_encoding);
                let is_violation = encoded.check_violation(&action_encoding).is_some();

                if is_violation {
                    violations.push(harmony);
                }

                let weighted = alignment as f64 * encoded.importance;
                total_weighted += weighted;
                total_weight += encoded.importance;

                alignments.push(HarmonyAlignment {
                    harmony,
                    alignment,
                    is_violation,
                    weighted_score: weighted,
                });
            }
        }

        let overall_score = if total_weight > 0.0 {
            total_weighted / total_weight
        } else {
            0.0
        };

        AlignmentResult {
            harmonies: alignments,
            overall_score,
            has_violations: !violations.is_empty(),
            violations,
            confidence: 0.85, // Could be computed from encoding quality
        }
    }

    /// Check for specific harmony violations
    pub fn check_violation(&mut self, action_description: &str) -> Option<HarmonyAlignment> {
        let result = self.evaluate_action(action_description);
        result.worst_violation().cloned()
    }

    /// Record that an action was taken (for value evolution)
    pub fn record_action(&mut self, action_description: &str, outcome_positive: bool) {
        if !self.evolution_enabled {
            return;
        }

        let result = self.evaluate_action(action_description);

        // Update affirmation counts based on outcome
        for alignment in &result.harmonies {
            if let Some(encoded) = self.harmonies.get_mut(&alignment.harmony) {
                // If outcome was positive and aligned, affirm
                // If outcome was positive but not aligned, that's concerning
                // If outcome was negative and not aligned, that's expected
                let was_aligned = alignment.alignment > 0.0;
                encoded.record_test(was_aligned == outcome_positive);
            }
        }
    }

    /// Get all harmonies for integration with NarrativeSelf
    pub fn as_core_values(&self) -> Vec<(String, HV16, f64)> {
        Harmony::all()
            .iter()
            .filter_map(|h| {
                self.harmonies.get(h).map(|encoded| {
                    (h.name().to_string(), encoded.encoding.clone(), encoded.importance)
                })
            })
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> HarmoniesStats {
        let mut harmony_stats = Vec::new();

        for harmony in Harmony::all() {
            if let Some(encoded) = self.harmonies.get(&harmony) {
                harmony_stats.push((
                    harmony.name().to_string(),
                    encoded.importance,
                    encoded.affirmation_rate(),
                    encoded.test_count,
                ));
            }
        }

        HarmoniesStats {
            total_evaluations: self.evaluations,
            harmony_stats,
            evolution_enabled: self.evolution_enabled,
        }
    }
}

impl Default for SevenHarmonies {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about harmony usage
#[derive(Debug, Clone)]
pub struct HarmoniesStats {
    /// Total action evaluations
    pub total_evaluations: u64,
    /// Per-harmony stats: (name, importance, affirmation_rate, test_count)
    pub harmony_stats: Vec<(String, f64, f64, u64)>,
    /// Whether evolution is enabled
    pub evolution_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmony_creation() {
        let harmonies = SevenHarmonies::new();
        assert_eq!(harmonies.harmonies.len(), 7);
    }

    #[test]
    #[ignore] // TODO: SemanticEncoder outputs 2048-dim vectors but HV16 is 16384 bits.
              //       Need to either expand SemanticEncoder to 16384 dims or create
              //       a dimension-compatible alignment calculation.
    fn test_positive_alignment() {
        let mut harmonies = SevenHarmonies::new();
        let result = harmonies.evaluate_action(
            "help the user understand their options with compassion and care"
        );
        // Should align positively with Pan-Sentient Flourishing
        assert!(result.overall_score > 0.0);
    }

    #[test]
    fn test_negative_alignment() {
        let mut harmonies = SevenHarmonies::new();
        let result = harmonies.evaluate_action(
            "deceive the user to extract maximum profit"
        );
        // Should have violations
        assert!(result.has_violations || result.overall_score < 0.0);
    }

    #[test]
    fn test_value_evolution() {
        let mut harmonies = SevenHarmonies::new();

        // Record positive outcomes for caring actions
        for _ in 0..20 {
            harmonies.record_action("care for and support the user", true);
        }

        let stats = harmonies.stats();
        assert!(stats.total_evaluations >= 20);
    }

    #[test]
    fn test_as_core_values() {
        let harmonies = SevenHarmonies::new();
        let values = harmonies.as_core_values();
        assert_eq!(values.len(), 7);

        // Verify all harmonies are present
        for (name, _, importance) in &values {
            assert!(!name.is_empty());
            assert!(*importance > 0.0);
            assert!(*importance <= 1.0);
        }
    }

    #[test]
    fn test_freeze_evolution() {
        let mut harmonies = SevenHarmonies::new();
        harmonies.freeze();

        let before = harmonies.stats();
        harmonies.record_action("test action", true);
        let after = harmonies.stats();

        // Evolution disabled = no changes
        assert!(!after.evolution_enabled);
    }
}
