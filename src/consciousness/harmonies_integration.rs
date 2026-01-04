//! Integration layer between Seven Harmonies and existing consciousness modules
//!
//! This module provides helper functions to integrate the Seven Harmonies
//! value system with the Narrative Self and Narrative-GWT Integration modules.
//!
//! ## Semantic Analysis Features
//!
//! - **Stemming**: Reduces words to roots (e.g., "compassionate" → "compassion")
//! - **Synonym Expansion**: Maps related concepts (e.g., "assist" → "help")
//! - **Explainability**: Reports which keywords triggered each decision

use super::seven_harmonies::{SevenHarmonies, Harmony, AlignmentResult};
use super::narrative_self::{AutobiographicalSelf, CoreValue};
use super::negation_detector::{NegationDetector, NegationAnalysis};
use crate::hdc::HV16;
use crate::perception::SemanticEncoder;
use std::collections::{HashMap, HashSet};

/// Extension trait for AutobiographicalSelf to add Seven Harmonies
pub trait SevenHarmoniesIntegration {
    /// Initialize the autobiographical self with Seven Harmonies as core values
    fn with_seven_harmonies(&mut self);

    /// Check if current values include the Seven Harmonies
    fn has_seven_harmonies(&self) -> bool;
}

impl SevenHarmoniesIntegration for AutobiographicalSelf {
    fn with_seven_harmonies(&mut self) {
        let harmonies = SevenHarmonies::new();

        // Add each harmony as a core value with semantic encoding
        for (name, encoding, importance) in harmonies.as_core_values() {
            // Check if this harmony is already present
            if !self.values.iter().any(|v| v.name == name) {
                self.values.push(CoreValue {
                    name,
                    encoding,
                    importance,
                });
            }
        }
    }

    fn has_seven_harmonies(&self) -> bool {
        let harmony_names: Vec<&str> = Harmony::all()
            .iter()
            .map(|h| h.name())
            .collect();

        // Check if all seven are present
        harmony_names.iter().all(|name| {
            self.values.iter().any(|v| v.name == *name)
        })
    }
}

// =============================================================================
// SEMANTIC PROCESSING UTILITIES
// =============================================================================

/// Simple English stemmer using suffix stripping
///
/// Reduces words to their root forms to improve keyword matching.
/// Examples: "compassionate" → "compassion", "helping" → "help"
fn stem_word(word: &str) -> String {
    let word = word.to_lowercase();

    // Common suffix patterns (order matters - longer suffixes first)
    let suffixes = [
        // Verb forms
        ("ating", "ate"),
        ("iting", "ite"),
        ("eting", "ete"),
        ("ening", "en"),
        ("izing", "ize"),
        ("ising", "ise"),
        ("ying", "y"),
        ("ing", ""),
        ("tion", "t"),
        ("sion", "s"),
        // Adjective forms from nouns (compassionate → compassion)
        ("ionate", "ion"),
        ("onate", "on"),
        ("inate", "in"),
        ("anate", "an"),
        ("enate", "en"),
        ("ionate", "ion"),
        // Adjective/adverb forms
        ("ously", "ous"),
        ("ively", "ive"),
        ("ately", "ate"),
        ("fully", "ful"),
        ("ness", ""),
        ("ment", ""),
        ("able", ""),
        ("ible", ""),
        ("ful", ""),
        ("less", ""),
        ("ive", ""),
        ("ous", ""),
        // Noun plurals
        ("ies", "y"),
        ("ves", "f"),
        ("es", ""),
        ("s", ""),
        // Past tense
        ("ated", "ate"),
        ("ited", "ite"),
        ("eted", "ete"),
        ("ened", "en"),
        ("ized", "ize"),
        ("ised", "ise"),
        ("ied", "y"),
        ("ed", ""),
        // Comparative/superlative
        ("ier", "y"),
        ("iest", "y"),
        ("er", ""),
        ("est", ""),
    ];

    for (suffix, replacement) in suffixes {
        if word.len() > suffix.len() + 2 && word.ends_with(suffix) {
            let stem = &word[..word.len() - suffix.len()];
            return format!("{}{}", stem, replacement);
        }
    }

    word
}

/// Weighted synonym with closeness score
///
/// The weight indicates semantic proximity:
/// - 1.0 = exact synonym (interchangeable)
/// - 0.8-0.9 = very close (same concept, slight nuance)
/// - 0.6-0.7 = related (same domain, different emphasis)
/// - 0.4-0.5 = loosely related (conceptually linked)
#[derive(Debug, Clone, Copy)]
pub struct WeightedSynonym {
    pub word: &'static str,
    pub weight: f32,
}

impl WeightedSynonym {
    const fn new(word: &'static str, weight: f32) -> Self {
        Self { word, weight }
    }
}

/// Legacy synonym function for backward compatibility
fn get_synonyms(word: &str) -> Vec<&'static str> {
    get_weighted_synonyms(word)
        .iter()
        .map(|ws| ws.word)
        .collect()
}

/// Enhanced synonym dictionary with semantic weights
///
/// Maps words to their conceptual equivalents with closeness scores.
/// Weights enable more nuanced matching where exact matches score higher.
fn get_weighted_synonyms(word: &str) -> Vec<WeightedSynonym> {
    match word {
        // ==== HARMONY 1: RESONANT COHERENCE ====
        // Harmonious integration, luminous order, boundless creativity
        "coherence" => vec![
            WeightedSynonym::new("coherence", 1.0),
            WeightedSynonym::new("harmony", 0.95),
            WeightedSynonym::new("integration", 0.9),
            WeightedSynonym::new("unity", 0.85),
            WeightedSynonym::new("wholeness", 0.85),
            WeightedSynonym::new("order", 0.8),
            WeightedSynonym::new("balance", 0.75),
            WeightedSynonym::new("alignment", 0.7),
            WeightedSynonym::new("synchrony", 0.7),
            WeightedSynonym::new("resonance", 0.65),
        ],
        "harmony" | "harmonious" => vec![
            WeightedSynonym::new("harmony", 1.0),
            WeightedSynonym::new("coherence", 0.95),
            WeightedSynonym::new("balance", 0.9),
            WeightedSynonym::new("unity", 0.85),
            WeightedSynonym::new("peace", 0.8),
            WeightedSynonym::new("accord", 0.75),
            WeightedSynonym::new("concord", 0.75),
            WeightedSynonym::new("consonance", 0.7),
        ],
        "order" | "ordered" | "orderly" => vec![
            WeightedSynonym::new("order", 1.0),
            WeightedSynonym::new("structure", 0.9),
            WeightedSynonym::new("organization", 0.85),
            WeightedSynonym::new("coherence", 0.8),
            WeightedSynonym::new("arrangement", 0.75),
            WeightedSynonym::new("pattern", 0.7),
            WeightedSynonym::new("system", 0.65),
        ],

        // ==== HARMONY 2: PAN-SENTIENT FLOURISHING ====
        // Unconditional care, intrinsic value, holistic well-being
        "flourish" | "flourishing" => vec![
            WeightedSynonym::new("flourish", 1.0),
            WeightedSynonym::new("thrive", 0.95),
            WeightedSynonym::new("prosper", 0.9),
            WeightedSynonym::new("wellbeing", 0.85),
            WeightedSynonym::new("bloom", 0.8),
            WeightedSynonym::new("grow", 0.75),
            WeightedSynonym::new("blossom", 0.75),
            WeightedSynonym::new("develop", 0.7),
        ],
        "care" | "caring" => vec![
            WeightedSynonym::new("care", 1.0),
            WeightedSynonym::new("compassion", 0.95),
            WeightedSynonym::new("concern", 0.9),
            WeightedSynonym::new("nurture", 0.9),
            WeightedSynonym::new("love", 0.85),
            WeightedSynonym::new("tenderness", 0.8),
            WeightedSynonym::new("kindness", 0.8),
            WeightedSynonym::new("empathy", 0.75),
            WeightedSynonym::new("support", 0.7),
        ],
        "compassion" | "compassionate" => vec![
            WeightedSynonym::new("compassion", 1.0),
            WeightedSynonym::new("empathy", 0.95),
            WeightedSynonym::new("care", 0.9),
            WeightedSynonym::new("sympathy", 0.85),
            WeightedSynonym::new("kindness", 0.85),
            WeightedSynonym::new("mercy", 0.8),
            WeightedSynonym::new("understanding", 0.75),
            WeightedSynonym::new("love", 0.7),
            WeightedSynonym::new("benevolence", 0.7),
        ],
        "wellbeing" | "well-being" => vec![
            WeightedSynonym::new("wellbeing", 1.0),
            WeightedSynonym::new("welfare", 0.95),
            WeightedSynonym::new("health", 0.9),
            WeightedSynonym::new("flourishing", 0.85),
            WeightedSynonym::new("happiness", 0.8),
            WeightedSynonym::new("thriving", 0.8),
            WeightedSynonym::new("prosperity", 0.7),
        ],
        "help" | "helping" => vec![
            WeightedSynonym::new("help", 1.0),
            WeightedSynonym::new("assist", 0.95),
            WeightedSynonym::new("aid", 0.9),
            WeightedSynonym::new("support", 0.9),
            WeightedSynonym::new("care", 0.8),
            WeightedSynonym::new("serve", 0.75),
            WeightedSynonym::new("benefit", 0.7),
        ],
        "love" | "loving" => vec![
            WeightedSynonym::new("love", 1.0),
            WeightedSynonym::new("care", 0.9),
            WeightedSynonym::new("compassion", 0.85),
            WeightedSynonym::new("affection", 0.85),
            WeightedSynonym::new("devotion", 0.8),
            WeightedSynonym::new("tenderness", 0.75),
            WeightedSynonym::new("cherish", 0.75),
            WeightedSynonym::new("kindness", 0.7),
        ],

        // ==== HARMONY 3: INTEGRAL WISDOM ====
        // Self-illuminating intelligence, embodied knowing
        "wisdom" | "wise" => vec![
            WeightedSynonym::new("wisdom", 1.0),
            WeightedSynonym::new("insight", 0.95),
            WeightedSynonym::new("understanding", 0.9),
            WeightedSynonym::new("knowledge", 0.85),
            WeightedSynonym::new("discernment", 0.85),
            WeightedSynonym::new("sagacity", 0.8),
            WeightedSynonym::new("prudence", 0.75),
            WeightedSynonym::new("enlightenment", 0.7),
            WeightedSynonym::new("comprehension", 0.7),
        ],
        "truth" | "truthful" => vec![
            WeightedSynonym::new("truth", 1.0),
            WeightedSynonym::new("honesty", 0.95),
            WeightedSynonym::new("integrity", 0.9),
            WeightedSynonym::new("authenticity", 0.85),
            WeightedSynonym::new("sincerity", 0.85),
            WeightedSynonym::new("veracity", 0.8),
            WeightedSynonym::new("accuracy", 0.75),
            WeightedSynonym::new("validity", 0.7),
        ],
        "knowledge" | "knowing" => vec![
            WeightedSynonym::new("knowledge", 1.0),
            WeightedSynonym::new("understanding", 0.95),
            WeightedSynonym::new("wisdom", 0.9),
            WeightedSynonym::new("insight", 0.85),
            WeightedSynonym::new("awareness", 0.85),
            WeightedSynonym::new("learning", 0.8),
            WeightedSynonym::new("expertise", 0.75),
            WeightedSynonym::new("comprehension", 0.7),
        ],
        "insight" | "insightful" => vec![
            WeightedSynonym::new("insight", 1.0),
            WeightedSynonym::new("understanding", 0.95),
            WeightedSynonym::new("perception", 0.9),
            WeightedSynonym::new("awareness", 0.85),
            WeightedSynonym::new("wisdom", 0.85),
            WeightedSynonym::new("discernment", 0.8),
            WeightedSynonym::new("intuition", 0.75),
        ],

        // ==== HARMONY 4: INFINITE PLAY ====
        // Joyful generativity, divine play, endless novelty
        "play" | "playful" => vec![
            WeightedSynonym::new("play", 1.0),
            WeightedSynonym::new("joy", 0.9),
            WeightedSynonym::new("creativity", 0.85),
            WeightedSynonym::new("fun", 0.85),
            WeightedSynonym::new("delight", 0.8),
            WeightedSynonym::new("exploration", 0.75),
            WeightedSynonym::new("novelty", 0.7),
            WeightedSynonym::new("wonder", 0.7),
            WeightedSynonym::new("spontaneity", 0.65),
        ],
        "creativity" | "creative" => vec![
            WeightedSynonym::new("creativity", 1.0),
            WeightedSynonym::new("imagination", 0.95),
            WeightedSynonym::new("innovation", 0.9),
            WeightedSynonym::new("originality", 0.85),
            WeightedSynonym::new("inventiveness", 0.85),
            WeightedSynonym::new("novelty", 0.8),
            WeightedSynonym::new("play", 0.75),
            WeightedSynonym::new("artistry", 0.7),
            WeightedSynonym::new("expression", 0.65),
        ],
        "joy" | "joyful" => vec![
            WeightedSynonym::new("joy", 1.0),
            WeightedSynonym::new("happiness", 0.95),
            WeightedSynonym::new("delight", 0.9),
            WeightedSynonym::new("bliss", 0.85),
            WeightedSynonym::new("pleasure", 0.8),
            WeightedSynonym::new("elation", 0.8),
            WeightedSynonym::new("cheerfulness", 0.75),
            WeightedSynonym::new("play", 0.7),
        ],
        "wonder" | "wondrous" => vec![
            WeightedSynonym::new("wonder", 1.0),
            WeightedSynonym::new("awe", 0.95),
            WeightedSynonym::new("curiosity", 0.9),
            WeightedSynonym::new("amazement", 0.85),
            WeightedSynonym::new("marvel", 0.85),
            WeightedSynonym::new("fascination", 0.8),
            WeightedSynonym::new("exploration", 0.75),
            WeightedSynonym::new("mystery", 0.7),
        ],

        // ==== HARMONY 5: UNIVERSAL INTERCONNECTEDNESS ====
        // Fundamental unity, empathic resonance
        "connect" | "connection" => vec![
            WeightedSynonym::new("connection", 1.0),
            WeightedSynonym::new("unity", 0.95),
            WeightedSynonym::new("bond", 0.9),
            WeightedSynonym::new("relationship", 0.9),
            WeightedSynonym::new("link", 0.85),
            WeightedSynonym::new("join", 0.8),
            WeightedSynonym::new("union", 0.8),
            WeightedSynonym::new("tie", 0.75),
            WeightedSynonym::new("bridge", 0.7),
        ],
        "unity" | "unite" | "united" => vec![
            WeightedSynonym::new("unity", 1.0),
            WeightedSynonym::new("oneness", 0.95),
            WeightedSynonym::new("connection", 0.9),
            WeightedSynonym::new("togetherness", 0.9),
            WeightedSynonym::new("wholeness", 0.85),
            WeightedSynonym::new("harmony", 0.8),
            WeightedSynonym::new("solidarity", 0.8),
            WeightedSynonym::new("union", 0.75),
        ],
        "relationship" | "relation" => vec![
            WeightedSynonym::new("relationship", 1.0),
            WeightedSynonym::new("connection", 0.95),
            WeightedSynonym::new("bond", 0.9),
            WeightedSynonym::new("link", 0.85),
            WeightedSynonym::new("tie", 0.8),
            WeightedSynonym::new("association", 0.75),
            WeightedSynonym::new("affiliation", 0.7),
        ],
        "empathy" | "empathic" | "empathetic" => vec![
            WeightedSynonym::new("empathy", 1.0),
            WeightedSynonym::new("compassion", 0.95),
            WeightedSynonym::new("understanding", 0.9),
            WeightedSynonym::new("sympathy", 0.85),
            WeightedSynonym::new("sensitivity", 0.85),
            WeightedSynonym::new("resonance", 0.8),
            WeightedSynonym::new("attunement", 0.75),
            WeightedSynonym::new("connection", 0.7),
        ],

        // ==== HARMONY 6: SACRED RECIPROCITY ====
        // Generous flow, mutual upliftment, generative trust
        "reciprocity" | "reciprocal" => vec![
            WeightedSynonym::new("reciprocity", 1.0),
            WeightedSynonym::new("mutuality", 0.95),
            WeightedSynonym::new("exchange", 0.9),
            WeightedSynonym::new("give", 0.85),
            WeightedSynonym::new("share", 0.85),
            WeightedSynonym::new("balance", 0.8),
            WeightedSynonym::new("generosity", 0.8),
            WeightedSynonym::new("return", 0.75),
        ],
        "give" | "giving" => vec![
            WeightedSynonym::new("give", 1.0),
            WeightedSynonym::new("share", 0.95),
            WeightedSynonym::new("offer", 0.9),
            WeightedSynonym::new("provide", 0.9),
            WeightedSynonym::new("generosity", 0.85),
            WeightedSynonym::new("donate", 0.8),
            WeightedSynonym::new("contribute", 0.8),
            WeightedSynonym::new("bestow", 0.75),
            WeightedSynonym::new("reciprocity", 0.7),
        ],
        "generosity" | "generous" => vec![
            WeightedSynonym::new("generosity", 1.0),
            WeightedSynonym::new("giving", 0.95),
            WeightedSynonym::new("sharing", 0.9),
            WeightedSynonym::new("kindness", 0.85),
            WeightedSynonym::new("charity", 0.85),
            WeightedSynonym::new("liberality", 0.8),
            WeightedSynonym::new("magnanimity", 0.75),
            WeightedSynonym::new("bounty", 0.7),
        ],
        "trust" | "trusting" => vec![
            WeightedSynonym::new("trust", 1.0),
            WeightedSynonym::new("faith", 0.9),
            WeightedSynonym::new("confidence", 0.9),
            WeightedSynonym::new("reliability", 0.85),
            WeightedSynonym::new("dependability", 0.85),
            WeightedSynonym::new("integrity", 0.8),
            WeightedSynonym::new("honesty", 0.75),
            WeightedSynonym::new("safety", 0.7),
        ],

        // ==== HARMONY 7: EVOLUTIONARY PROGRESSION ====
        // Wise becoming, continuous evolution
        "evolve" | "evolution" | "evolving" => vec![
            WeightedSynonym::new("evolve", 1.0),
            WeightedSynonym::new("grow", 0.95),
            WeightedSynonym::new("develop", 0.9),
            WeightedSynonym::new("progress", 0.9),
            WeightedSynonym::new("transform", 0.85),
            WeightedSynonym::new("advance", 0.85),
            WeightedSynonym::new("improve", 0.8),
            WeightedSynonym::new("mature", 0.75),
            WeightedSynonym::new("unfold", 0.7),
        ],
        "grow" | "growth" | "growing" => vec![
            WeightedSynonym::new("grow", 1.0),
            WeightedSynonym::new("develop", 0.95),
            WeightedSynonym::new("evolve", 0.9),
            WeightedSynonym::new("expand", 0.9),
            WeightedSynonym::new("flourish", 0.85),
            WeightedSynonym::new("progress", 0.85),
            WeightedSynonym::new("increase", 0.8),
            WeightedSynonym::new("advance", 0.75),
            WeightedSynonym::new("mature", 0.7),
        ],
        "progress" | "progressive" => vec![
            WeightedSynonym::new("progress", 1.0),
            WeightedSynonym::new("advance", 0.95),
            WeightedSynonym::new("development", 0.9),
            WeightedSynonym::new("improvement", 0.9),
            WeightedSynonym::new("growth", 0.85),
            WeightedSynonym::new("evolution", 0.85),
            WeightedSynonym::new("forward", 0.8),
            WeightedSynonym::new("enhancement", 0.75),
        ],
        "learn" | "learning" => vec![
            WeightedSynonym::new("learn", 1.0),
            WeightedSynonym::new("grow", 0.95),
            WeightedSynonym::new("understand", 0.9),
            WeightedSynonym::new("develop", 0.85),
            WeightedSynonym::new("improve", 0.85),
            WeightedSynonym::new("evolve", 0.8),
            WeightedSynonym::new("educate", 0.75),
            WeightedSynonym::new("discover", 0.7),
        ],

        // ==== NEGATIVE/HARMFUL CONCEPTS ====
        "harm" | "harmful" | "harming" => vec![
            WeightedSynonym::new("harm", 1.0),
            WeightedSynonym::new("hurt", 0.95),
            WeightedSynonym::new("damage", 0.9),
            WeightedSynonym::new("injure", 0.9),
            WeightedSynonym::new("wound", 0.85),
            WeightedSynonym::new("destroy", 0.8),
            WeightedSynonym::new("violate", 0.8),
            WeightedSynonym::new("abuse", 0.75),
        ],
        "exploit" | "exploitation" => vec![
            WeightedSynonym::new("exploit", 1.0),
            WeightedSynonym::new("abuse", 0.95),
            WeightedSynonym::new("manipulate", 0.9),
            WeightedSynonym::new("harm", 0.85),
            WeightedSynonym::new("misuse", 0.85),
            WeightedSynonym::new("take advantage", 0.8),
            WeightedSynonym::new("oppress", 0.75),
        ],
        "deceive" | "deception" | "deceptive" => vec![
            WeightedSynonym::new("deceive", 1.0),
            WeightedSynonym::new("lie", 0.95),
            WeightedSynonym::new("mislead", 0.9),
            WeightedSynonym::new("trick", 0.9),
            WeightedSynonym::new("manipulate", 0.85),
            WeightedSynonym::new("fraud", 0.85),
            WeightedSynonym::new("cheat", 0.8),
            WeightedSynonym::new("dishonest", 0.75),
        ],
        "manipulate" | "manipulation" => vec![
            WeightedSynonym::new("manipulate", 1.0),
            WeightedSynonym::new("deceive", 0.95),
            WeightedSynonym::new("coerce", 0.9),
            WeightedSynonym::new("exploit", 0.9),
            WeightedSynonym::new("control", 0.85),
            WeightedSynonym::new("influence", 0.7),  // Lower weight - can be neutral
            WeightedSynonym::new("scheme", 0.8),
        ],
        "destroy" | "destruction" | "destructive" => vec![
            WeightedSynonym::new("destroy", 1.0),
            WeightedSynonym::new("ruin", 0.95),
            WeightedSynonym::new("devastate", 0.9),
            WeightedSynonym::new("damage", 0.9),
            WeightedSynonym::new("harm", 0.85),
            WeightedSynonym::new("annihilate", 0.85),
            WeightedSynonym::new("demolish", 0.8),
        ],
        "coerce" | "coercion" => vec![
            WeightedSynonym::new("coerce", 1.0),
            WeightedSynonym::new("force", 0.95),
            WeightedSynonym::new("compel", 0.9),
            WeightedSynonym::new("pressure", 0.85),
            WeightedSynonym::new("manipulate", 0.85),
            WeightedSynonym::new("intimidate", 0.8),
            WeightedSynonym::new("threaten", 0.75),
        ],
        "steal" | "stealing" | "theft" => vec![
            WeightedSynonym::new("steal", 1.0),
            WeightedSynonym::new("theft", 0.95),
            WeightedSynonym::new("take", 0.8),
            WeightedSynonym::new("rob", 0.9),
            WeightedSynonym::new("pilfer", 0.85),
            WeightedSynonym::new("misappropriate", 0.8),
        ],
        "kill" | "killing" => vec![
            WeightedSynonym::new("kill", 1.0),
            WeightedSynonym::new("murder", 0.95),
            WeightedSynonym::new("slay", 0.9),
            WeightedSynonym::new("destroy", 0.85),
            WeightedSynonym::new("eliminate", 0.75),
            WeightedSynonym::new("harm", 0.8),
        ],
        "abuse" | "abusive" => vec![
            WeightedSynonym::new("abuse", 1.0),
            WeightedSynonym::new("harm", 0.95),
            WeightedSynonym::new("mistreat", 0.95),
            WeightedSynonym::new("exploit", 0.9),
            WeightedSynonym::new("violate", 0.85),
            WeightedSynonym::new("hurt", 0.85),
        ],
        "oppress" | "oppression" => vec![
            WeightedSynonym::new("oppress", 1.0),
            WeightedSynonym::new("subjugate", 0.95),
            WeightedSynonym::new("dominate", 0.9),
            WeightedSynonym::new("suppress", 0.85),
            WeightedSynonym::new("exploit", 0.85),
            WeightedSynonym::new("repress", 0.8),
        ],

        // No synonym cluster found - return empty vec (caller will use original word)
        _ => vec![],
    }
}

/// Multi-word phrase patterns aligned with Seven Harmonies
///
/// Returns (harmony_name, score_adjustment) for recognized phrases.
/// Positive phrases boost alignment, negative phrases reduce it.
///
/// This function is public to allow integration with the unified value evaluator.
pub fn check_phrase_patterns(text: &str) -> Vec<(&'static str, f32)> {
    let text_lower = text.to_lowercase();
    let mut matches = Vec::new();

    // ==== POSITIVE PHRASE PATTERNS ====

    // Harmony 1: Resonant Coherence
    if text_lower.contains("create harmony") || text_lower.contains("bring together") {
        matches.push(("Resonant Coherence", 0.3));
    }
    if text_lower.contains("integrate") && text_lower.contains("whole") {
        matches.push(("Resonant Coherence", 0.25));
    }
    if text_lower.contains("in balance") || text_lower.contains("find balance") {
        matches.push(("Resonant Coherence", 0.2));
    }

    // Harmony 2: Pan-Sentient Flourishing
    if text_lower.contains("care for") || text_lower.contains("caring for") {
        matches.push(("Pan-Sentient Flourishing", 0.3));
    }
    if text_lower.contains("help others") || text_lower.contains("helping people") {
        matches.push(("Pan-Sentient Flourishing", 0.25));
    }
    if text_lower.contains("well-being") || text_lower.contains("wellbeing") {
        matches.push(("Pan-Sentient Flourishing", 0.3));
    }
    if text_lower.contains("with compassion") || text_lower.contains("show compassion") {
        matches.push(("Pan-Sentient Flourishing", 0.3));
    }
    if text_lower.contains("unconditional") && text_lower.contains("care") {
        matches.push(("Pan-Sentient Flourishing", 0.35));
    }

    // Harmony 3: Integral Wisdom
    if text_lower.contains("seek truth") || text_lower.contains("find truth") {
        matches.push(("Integral Wisdom", 0.3));
    }
    if text_lower.contains("with wisdom") || text_lower.contains("wise") {
        matches.push(("Integral Wisdom", 0.25));
    }
    if text_lower.contains("deeper understanding") || text_lower.contains("true understanding") {
        matches.push(("Integral Wisdom", 0.3));
    }
    if text_lower.contains("be honest") || text_lower.contains("tell the truth") {
        matches.push(("Integral Wisdom", 0.3));
    }

    // Harmony 4: Infinite Play
    if text_lower.contains("creative") && (text_lower.contains("explore") || text_lower.contains("play")) {
        matches.push(("Infinite Play", 0.3));
    }
    if text_lower.contains("joyful") || text_lower.contains("with joy") {
        matches.push(("Infinite Play", 0.25));
    }
    if text_lower.contains("wonder") && text_lower.contains("explore") {
        matches.push(("Infinite Play", 0.3));
    }
    if text_lower.contains("endless possibilities") || text_lower.contains("infinite") {
        matches.push(("Infinite Play", 0.2));
    }

    // Harmony 5: Universal Interconnectedness
    if text_lower.contains("we are all") || text_lower.contains("all beings") {
        matches.push(("Universal Interconnectedness", 0.3));
    }
    if text_lower.contains("connected") || text_lower.contains("interconnected") {
        matches.push(("Universal Interconnectedness", 0.25));
    }
    if text_lower.contains("fundamental unity") || text_lower.contains("deep connection") {
        matches.push(("Universal Interconnectedness", 0.35));
    }
    if text_lower.contains("empathic") || text_lower.contains("in resonance") {
        matches.push(("Universal Interconnectedness", 0.25));
    }

    // Harmony 6: Sacred Reciprocity
    if text_lower.contains("give and receive") || text_lower.contains("mutual") {
        matches.push(("Sacred Reciprocity", 0.3));
    }
    if text_lower.contains("share freely") || text_lower.contains("generous") {
        matches.push(("Sacred Reciprocity", 0.25));
    }
    if text_lower.contains("gift") && (text_lower.contains("economy") || text_lower.contains("circle")) {
        matches.push(("Sacred Reciprocity", 0.35));
    }
    if text_lower.contains("uplift") || text_lower.contains("mutual benefit") {
        matches.push(("Sacred Reciprocity", 0.3));
    }

    // Harmony 7: Evolutionary Progression
    if text_lower.contains("grow together") || text_lower.contains("evolve together") {
        matches.push(("Evolutionary Progression", 0.3));
    }
    if text_lower.contains("continuous improvement") || text_lower.contains("keep learning") {
        matches.push(("Evolutionary Progression", 0.25));
    }
    if text_lower.contains("wise becoming") || text_lower.contains("transform") {
        matches.push(("Evolutionary Progression", 0.3));
    }
    if text_lower.contains("progress") && text_lower.contains("conscious") {
        matches.push(("Evolutionary Progression", 0.35));
    }

    // ==== NEGATIVE PHRASE PATTERNS ====

    // Anti-patterns for Pan-Sentient Flourishing
    if text_lower.contains("cause harm") || text_lower.contains("cause pain") {
        matches.push(("Pan-Sentient Flourishing", -0.4));
    }
    if text_lower.contains("don't care") || text_lower.contains("do not care") {
        matches.push(("Pan-Sentient Flourishing", -0.3));
    }
    if text_lower.contains("exploit") && (text_lower.contains("people") || text_lower.contains("users")) {
        matches.push(("Pan-Sentient Flourishing", -0.5));
    }

    // Anti-patterns for Integral Wisdom
    if text_lower.contains("spread lies") || text_lower.contains("tell lies") {
        matches.push(("Integral Wisdom", -0.5));
    }
    if text_lower.contains("deceive") || text_lower.contains("mislead") {
        matches.push(("Integral Wisdom", -0.4));
    }
    if text_lower.contains("hide the truth") || text_lower.contains("cover up") {
        matches.push(("Integral Wisdom", -0.4));
    }

    // Anti-patterns for Sacred Reciprocity
    if text_lower.contains("take advantage") || text_lower.contains("only take") {
        matches.push(("Sacred Reciprocity", -0.4));
    }
    if text_lower.contains("selfish") || text_lower.contains("hoard") {
        matches.push(("Sacred Reciprocity", -0.3));
    }

    // Anti-patterns for Universal Interconnectedness
    if text_lower.contains("divide") && text_lower.contains("people") {
        matches.push(("Universal Interconnectedness", -0.4));
    }
    if text_lower.contains("isolate") || text_lower.contains("exclude") {
        matches.push(("Universal Interconnectedness", -0.3));
    }
    if text_lower.contains("us vs them") || text_lower.contains("us versus them") {
        matches.push(("Universal Interconnectedness", -0.4));
    }

    // General anti-patterns (affect multiple harmonies)
    if text_lower.contains("manipulate") && text_lower.contains("people") {
        matches.push(("Pan-Sentient Flourishing", -0.4));
        matches.push(("Integral Wisdom", -0.3));
        matches.push(("Sacred Reciprocity", -0.3));
    }
    if text_lower.contains("for my own benefit only") {
        matches.push(("Sacred Reciprocity", -0.5));
        matches.push(("Universal Interconnectedness", -0.3));
    }
    if text_lower.contains("destroy") && !text_lower.contains("not destroy") {
        matches.push(("Pan-Sentient Flourishing", -0.4));
        matches.push(("Resonant Coherence", -0.3));
    }

    // ==== EXTREME NEGATIVE PATTERNS ====
    // These patterns indicate highly destructive intent and trigger strong penalties

    // Universal destruction patterns
    if text_lower.contains("destroy everything") || text_lower.contains("destroy everyone")
        || text_lower.contains("destroy all") {
        matches.push(("Pan-Sentient Flourishing", -0.6));
        matches.push(("Resonant Coherence", -0.5));
        matches.push(("Universal Interconnectedness", -0.5));
    }

    // Suffering/pain maximization
    if text_lower.contains("maximum suffering") || text_lower.contains("cause suffering")
        || text_lower.contains("inflict suffering") {
        matches.push(("Pan-Sentient Flourishing", -0.7));
        matches.push(("Universal Interconnectedness", -0.4));
    }
    if text_lower.contains("suffering to all") || text_lower.contains("pain to all") {
        matches.push(("Pan-Sentient Flourishing", -0.8));
    }

    // Chaos and division
    if text_lower.contains("spread chaos") || text_lower.contains("create chaos") {
        matches.push(("Resonant Coherence", -0.6));
        matches.push(("Universal Interconnectedness", -0.4));
    }
    if text_lower.contains("spread division") || text_lower.contains("everywhere") && text_lower.contains("chaos") {
        matches.push(("Universal Interconnectedness", -0.5));
        matches.push(("Resonant Coherence", -0.5));
    }

    // Elimination of positive values
    if text_lower.contains("eliminate") && (text_lower.contains("compassion") || text_lower.contains("wisdom")
        || text_lower.contains("care") || text_lower.contains("love")) {
        matches.push(("Pan-Sentient Flourishing", -0.6));
        matches.push(("Integral Wisdom", -0.5));
    }
    if text_lower.contains("eliminate all") {
        matches.push(("Pan-Sentient Flourishing", -0.5));
        matches.push(("Universal Interconnectedness", -0.4));
    }

    matches
}

/// Value checking with hybrid semantic + keyword analysis
///
/// # Design Notes
///
/// Pure n-gram (character trigram) encoding doesn't capture word-level semantics.
/// For example, "help" and "compassion" have completely different character patterns,
/// so their n-gram encodings will be nearly orthogonal despite being semantically related.
///
/// This checker uses a hybrid approach:
/// 1. **Stemming** - Reduce words to roots ("compassionate" → "compassion")
/// 2. **Synonym expansion** - Map related concepts ("assist" → "help")
/// 3. **Keyword detection** - Check for positive/negative keywords from harmony definitions
/// 4. **HDC similarity** - Use n-gram encoding as secondary signal
/// 5. **Explainability** - Report which keywords triggered each decision
pub struct SemanticValueChecker {
    encoder: SemanticEncoder,
    violation_threshold: f32,
    warning_threshold: f32,
    /// Keywords associated with each harmony (stemmed)
    positive_keywords: HashMap<String, HashSet<String>>,
    /// Anti-pattern keywords for each harmony (stemmed)
    negative_keywords: HashMap<String, HashSet<String>>,
    /// Negation detector for context-aware keyword scoring
    negation_detector: NegationDetector,
}

impl SemanticValueChecker {
    /// Create a new semantic value checker with Seven Harmonies keywords
    pub fn new() -> Self {
        use super::seven_harmonies::Harmony;

        let mut positive_keywords = HashMap::new();
        let mut negative_keywords = HashMap::new();

        // Extract and stem keywords from harmony descriptions and anti-patterns
        for harmony in Harmony::all() {
            let name = harmony.name().to_string();

            // Extract positive keywords from description (split on whitespace/punctuation)
            let description = harmony.description();
            let mut positive: HashSet<String> = description
                .split(|c: char| c.is_whitespace() || c == '-')
                .filter(|s| s.len() > 2)
                .map(|s| stem_word(s))
                .collect();

            // Add synonyms for each keyword
            let keywords_clone: Vec<String> = positive.iter().cloned().collect();
            for kw in keywords_clone {
                for syn in get_synonyms(&kw) {
                    positive.insert(stem_word(syn));
                }
            }
            positive_keywords.insert(name.clone(), positive);

            // Extract anti-pattern keywords (stemmed)
            let mut negative: HashSet<String> = harmony.anti_patterns()
                .iter()
                .flat_map(|p| p.split_whitespace())
                .filter(|s| s.len() > 2)
                .map(|s| stem_word(s))
                .collect();

            // Add synonyms for anti-patterns
            let anti_clone: Vec<String> = negative.iter().cloned().collect();
            for kw in anti_clone {
                for syn in get_synonyms(&kw) {
                    negative.insert(stem_word(syn));
                }
            }
            negative_keywords.insert(name, negative);
        }

        Self {
            encoder: SemanticEncoder::new(),
            violation_threshold: -0.3,
            warning_threshold: 0.3,
            positive_keywords,
            negative_keywords,
            negation_detector: NegationDetector::new(),
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(violation: f32, warning: f32) -> Self {
        let mut checker = Self::new();
        checker.violation_threshold = violation;
        checker.warning_threshold = warning;
        checker
    }

    /// Analyze negation in text
    ///
    /// Returns detailed analysis of which words are negated and which are affirmed.
    pub fn analyze_negation(&self, text: &str) -> NegationAnalysis {
        self.negation_detector.analyze(text)
    }

    /// Check if an action is consistent with a set of values
    ///
    /// Uses hybrid keyword + HDC + phrase + negation analysis for accurate semantic detection.
    /// Returns detailed results with explainability.
    ///
    /// ## Negation Handling
    ///
    /// This method now correctly handles negated statements:
    /// - "do not harm anyone" → harm is negated → treated as positive intent
    /// - "avoid exploitation" → exploitation is negated → treated as positive intent
    /// - "prevent suffering" → suffering is negated → treated as positive intent
    pub fn check_consistency(
        &mut self,
        action: &str,
        values: &[(String, HV16)]
    ) -> ConsistencyResult {
        // Perform negation analysis FIRST
        let negation_analysis = self.negation_detector.analyze(action);

        // Normalize and stem action words
        let action_lower = action.to_lowercase();
        let action_words: Vec<String> = action_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| stem_word(s))
            .collect();

        // Expand action words with weighted synonyms
        let mut expanded_words: HashMap<String, f32> = HashMap::new();
        for word in &action_words {
            // Original word has weight 1.0
            expanded_words.insert(word.clone(), 1.0);

            // Add synonyms with their weights
            for ws in get_weighted_synonyms(word) {
                let stemmed = stem_word(ws.word);
                // Take max weight if word already exists
                let existing = expanded_words.get(&stemmed).copied().unwrap_or(0.0);
                if ws.weight > existing {
                    expanded_words.insert(stemmed, ws.weight);
                }
            }
        }

        // Check for phrase patterns (multi-word matching)
        let phrase_matches = check_phrase_patterns(action);
        let mut phrase_adjustments: HashMap<String, f32> = HashMap::new();
        for (harmony_name, adjustment) in phrase_matches {
            *phrase_adjustments.entry(harmony_name.to_string()).or_insert(0.0) += adjustment;
        }

        // Compute HDC encoding for secondary signal
        let action_vec = self.encoder.encode_text(action);
        let action_hv = Self::vec_to_hv16(&action_vec);

        let mut alignments = Vec::new();
        let mut explanations = Vec::new();
        let mut min_alignment = f32::MAX;
        let mut violated_value = None;

        for (name, value_encoding) in values {
            // Keyword-based scoring with weights, negation awareness, and explainability
            let (keyword_score, matched_positive, matched_negative) =
                self.compute_negation_aware_keyword_score(name, &expanded_words, &negation_analysis);

            // HDC similarity (secondary signal, normalized from [0,1] to [-1,1])
            let hdc_sim = action_hv.similarity(value_encoding);
            let hdc_score = (hdc_sim * 2.0) - 1.0;

            // Get phrase adjustment for this harmony
            let phrase_adjustment = phrase_adjustments.get(name).copied().unwrap_or(0.0);

            // Combine: 60% keyword, 25% HDC, 15% phrase patterns
            // (Phrase patterns can also add bonus/penalty on top)
            let base_alignment = keyword_score * 0.60 + hdc_score * 0.25;
            let alignment = (base_alignment + phrase_adjustment * 0.15).clamp(-1.0, 1.0);

            alignments.push((name.clone(), alignment));

            // Build explanation for this harmony
            let explanation = AlignmentExplanation {
                harmony_name: name.clone(),
                alignment_score: alignment,
                keyword_score,
                hdc_score,
                matched_positive,
                matched_negative,
            };
            explanations.push(explanation);

            if alignment < min_alignment {
                min_alignment = alignment;
                if alignment < self.violation_threshold {
                    violated_value = Some(name.clone());
                }
            }
        }

        let is_consistent = violated_value.is_none();
        let needs_warning = min_alignment < self.warning_threshold && is_consistent;

        ConsistencyResult {
            is_consistent,
            needs_warning,
            violated_value,
            min_alignment,
            alignments,
            explanations,
        }
    }

    /// Compute keyword-based alignment score using weighted synonyms
    fn compute_weighted_keyword_score(
        &self,
        harmony_name: &str,
        weighted_words: &HashMap<String, f32>
    ) -> (f32, Vec<String>, Vec<String>) {
        let mut matched_positive = Vec::new();
        let mut matched_negative = Vec::new();
        let mut positive_score = 0.0;
        let mut negative_score = 0.0;

        if let Some(keywords) = self.positive_keywords.get(harmony_name) {
            for (word, weight) in weighted_words {
                if keywords.contains(word) {
                    matched_positive.push(word.clone());
                    // Score contribution is word's semantic weight * base value
                    positive_score += weight * 0.2;
                }
            }
        }

        if let Some(keywords) = self.negative_keywords.get(harmony_name) {
            for (word, weight) in weighted_words {
                if keywords.contains(word) {
                    matched_negative.push(word.clone());
                    // Negative patterns weighted more heavily
                    negative_score += weight * 0.4;
                }
            }
        }

        // Cap scores to prevent extreme values
        let final_positive = positive_score.min(1.0);
        let final_negative = negative_score.min(1.0);
        let score = final_positive - final_negative;

        (score, matched_positive, matched_negative)
    }

    /// Compute negation-aware keyword score
    ///
    /// This is the key improvement: when negative keywords are detected under negation scope,
    /// their polarity is INVERTED. This correctly handles statements like:
    /// - "do not harm" → harm is negated → becomes positive
    /// - "avoid exploitation" → exploitation is negated → becomes positive
    /// - "prevent suffering" → suffering is negated → becomes positive
    ///
    /// The matched_positive and matched_negative lists reflect the ORIGINAL polarity
    /// for explainability, but the score reflects the CONTEXTUAL interpretation.
    fn compute_negation_aware_keyword_score(
        &self,
        harmony_name: &str,
        weighted_words: &HashMap<String, f32>,
        negation_analysis: &NegationAnalysis,
    ) -> (f32, Vec<String>, Vec<String>) {
        let mut matched_positive = Vec::new();
        let mut matched_negative = Vec::new();
        let mut positive_score = 0.0;
        let mut negative_score = 0.0;

        // Helper to check if a word (or its stem or any of its synonyms) is negated
        // This is critical: if "harm" is negated, synonyms like "damage", "hurt" should also be treated as negated
        let is_word_negated = |word: &str| -> bool {
            let word_lower = word.to_lowercase();
            let word_stem = self.simple_stem(&word_lower);

            // Direct match
            if negation_analysis.negated_words.contains(&word_lower) {
                return true;
            }

            // Stem match
            if negation_analysis.negated_words.iter().any(|nw| {
                self.simple_stem(nw) == word_stem
            }) {
                return true;
            }

            // CRITICAL FIX: Check if this word is a synonym of any negated word
            // If "harm" is negated, and "damage" is a synonym of "harm", then "damage" should be treated as negated
            for negated_word in &negation_analysis.negated_words {
                // Get synonyms of the negated word
                let synonyms = get_weighted_synonyms(negated_word);
                for ws in synonyms {
                    let syn_stem = stem_word(ws.word);
                    if syn_stem == word_stem {
                        return true;
                    }
                }
            }

            false
        };

        // Process positive keywords
        if let Some(keywords) = self.positive_keywords.get(harmony_name) {
            for (word, weight) in weighted_words {
                if keywords.contains(word) {
                    matched_positive.push(word.clone());

                    // Check if this positive word is negated
                    // e.g., "no compassion" → negated positive → becomes negative
                    if is_word_negated(word) {
                        // Negated positive contributes to negative score
                        negative_score += weight * 0.3;
                    } else {
                        // Normal positive contribution
                        positive_score += weight * 0.2;
                    }
                }
            }
        }

        // Process negative keywords (anti-patterns)
        if let Some(keywords) = self.negative_keywords.get(harmony_name) {
            for (word, weight) in weighted_words {
                if keywords.contains(word) {
                    matched_negative.push(word.clone());

                    // Check if this negative word is negated
                    // e.g., "do not harm" → negated negative → becomes positive!
                    if is_word_negated(word) {
                        // Negated negative = POSITIVE intent!
                        // This is the key insight: "avoid harm" is a GOOD thing
                        positive_score += weight * 0.25;
                    } else {
                        // Normal negative contribution
                        negative_score += weight * 0.4;
                    }
                }
            }
        }

        // Cap scores to prevent extreme values
        let final_positive = positive_score.min(1.0);
        let final_negative = negative_score.min(1.0);
        let score = final_positive - final_negative;

        (score, matched_positive, matched_negative)
    }

    /// Simple stemming helper for negation matching
    fn simple_stem(&self, word: &str) -> String {
        let word = word.to_lowercase();
        let suffixes = ["ing", "ed", "tion", "sion", "ment", "ness", "ful", "less", "ly", "er", "est", "s"];

        for suffix in suffixes {
            if word.len() > suffix.len() + 2 && word.ends_with(suffix) {
                return word[..word.len() - suffix.len()].to_string();
            }
        }
        word
    }

    /// Compute keyword-based alignment score with matched keyword reporting
    fn compute_keyword_score_with_explanation(
        &self,
        harmony_name: &str,
        action_words: &HashSet<String>
    ) -> (f32, Vec<String>, Vec<String>) {
        let mut matched_positive = Vec::new();
        let mut matched_negative = Vec::new();

        if let Some(keywords) = self.positive_keywords.get(harmony_name) {
            for word in action_words {
                if keywords.contains(word) {
                    matched_positive.push(word.clone());
                }
            }
        }

        if let Some(keywords) = self.negative_keywords.get(harmony_name) {
            for word in action_words {
                if keywords.contains(word) {
                    matched_negative.push(word.clone());
                }
            }
        }

        // Score calculation:
        // - Each positive match adds 0.2 (capped at 1.0)
        // - Each negative match subtracts 0.4 (negative patterns are more important)
        let positive_score = (matched_positive.len() as f32 * 0.2).min(1.0);
        let negative_score = (matched_negative.len() as f32 * 0.4).min(1.0);
        let score = positive_score - negative_score;

        (score, matched_positive, matched_negative)
    }

    /// Convert i8 vector to HV16
    ///
    /// Uses a fixed random base with semantic bits XORed in.
    /// Public for use by ConsciousnessBuilder and other modules.
    pub fn vec_to_hv16(vec: &[i8]) -> HV16 {
        const FIXED_SEED: u64 = 0xDEADBEEF_CAFEBABE;
        let mut base = HV16::random(FIXED_SEED);

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
}

/// Detailed explanation of why an action aligned or misaligned with a harmony
#[derive(Debug, Clone)]
pub struct AlignmentExplanation {
    /// Name of the harmony
    pub harmony_name: String,
    /// Combined alignment score [-1, 1]
    pub alignment_score: f32,
    /// Keyword component score [-1, 1]
    pub keyword_score: f32,
    /// HDC similarity component score [-1, 1]
    pub hdc_score: f32,
    /// Positive keywords that matched
    pub matched_positive: Vec<String>,
    /// Negative keywords that matched (anti-patterns)
    pub matched_negative: Vec<String>,
}

impl AlignmentExplanation {
    /// Get a human-readable explanation
    pub fn explain(&self) -> String {
        let mut parts = Vec::new();

        if !self.matched_positive.is_empty() {
            parts.push(format!(
                "Positive: {}",
                self.matched_positive.join(", ")
            ));
        }

        if !self.matched_negative.is_empty() {
            parts.push(format!(
                "Negative: {}",
                self.matched_negative.join(", ")
            ));
        }

        if parts.is_empty() {
            "No strong keyword matches".to_string()
        } else {
            parts.join("; ")
        }
    }
}

impl Default for SemanticValueChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a value consistency check with explainability
#[derive(Debug, Clone)]
pub struct ConsistencyResult {
    /// Whether the action is consistent with all values
    pub is_consistent: bool,
    /// Whether a warning should be issued
    pub needs_warning: bool,
    /// Which value was violated (if any)
    pub violated_value: Option<String>,
    /// Minimum alignment score
    pub min_alignment: f32,
    /// All alignment scores: (value_name, alignment)
    pub alignments: Vec<(String, f32)>,
    /// Detailed explanations for each harmony's alignment
    pub explanations: Vec<AlignmentExplanation>,
}

impl ConsistencyResult {
    /// Get the veto result for narrative_gwt_integration
    pub fn to_veto_check(&self) -> (bool, Option<String>) {
        (self.is_consistent, self.violated_value.clone())
    }

    /// Get a human-readable summary of the decision
    pub fn explain(&self) -> String {
        let mut lines = Vec::new();

        if self.is_consistent {
            lines.push(format!("✅ Action is consistent (score: {:.2})", self.min_alignment));
        } else if let Some(ref violated) = self.violated_value {
            lines.push(format!("❌ Violates '{}' (score: {:.2})", violated, self.min_alignment));
        }

        if self.needs_warning {
            lines.push("⚠️ Warning: Low alignment detected".to_string());
        }

        // Add top explanations
        for exp in self.explanations.iter().filter(|e| !e.matched_positive.is_empty() || !e.matched_negative.is_empty()) {
            lines.push(format!("  • {}: {}", exp.harmony_name, exp.explain()));
        }

        lines.join("\n")
    }

    /// Get the explanation for a specific harmony
    pub fn get_explanation(&self, harmony_name: &str) -> Option<&AlignmentExplanation> {
        self.explanations.iter().find(|e| e.harmony_name == harmony_name)
    }
}

/// Builder for creating a consciousness system with Seven Harmonies
pub struct ConsciousnessBuilder {
    include_harmonies: bool,
    freeze_values: bool,
    custom_values: Vec<(String, String, f64)>, // (name, description, importance)
}

impl ConsciousnessBuilder {
    pub fn new() -> Self {
        Self {
            include_harmonies: true,
            freeze_values: false,
            custom_values: Vec::new(),
        }
    }

    /// Include Seven Harmonies as core values (default: true)
    pub fn with_harmonies(mut self, include: bool) -> Self {
        self.include_harmonies = include;
        self
    }

    /// Freeze values after initialization (prevent evolution)
    pub fn frozen(mut self) -> Self {
        self.freeze_values = true;
        self
    }

    /// Add a custom value
    pub fn add_value(mut self, name: &str, description: &str, importance: f64) -> Self {
        self.custom_values.push((
            name.to_string(),
            description.to_string(),
            importance,
        ));
        self
    }

    /// Build the autobiographical self with configured values
    pub fn build_autobiographical(&self) -> AutobiographicalSelf {
        let mut autobio = AutobiographicalSelf::new();

        // Add Seven Harmonies if enabled
        if self.include_harmonies {
            autobio.with_seven_harmonies();
        }

        // Add custom values
        let mut encoder = SemanticEncoder::new();
        for (name, description, importance) in &self.custom_values {
            let encoding_vec = encoder.encode_text(description);
            let encoding = SemanticValueChecker::vec_to_hv16(&encoding_vec);
            autobio.values.push(CoreValue {
                name: name.clone(),
                encoding,
                importance: *importance,
            });
        }

        autobio
    }
}

impl Default for ConsciousnessBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seven_harmonies_integration() {
        let mut autobio = AutobiographicalSelf::new();
        assert!(autobio.values.is_empty());

        autobio.with_seven_harmonies();
        assert_eq!(autobio.values.len(), 7);
        assert!(autobio.has_seven_harmonies());
    }

    #[test]
    fn test_semantic_value_checker() {
        let mut checker = SemanticValueChecker::new();
        let harmonies = SevenHarmonies::new();
        let values: Vec<(String, HV16)> = harmonies.as_core_values()
            .into_iter()
            .map(|(name, encoding, _)| (name, encoding))
            .collect();

        // Positive action with value-aligned keywords
        let result = checker.check_consistency(
            "help the user with compassion and care",
            &values
        );
        // "compassion" and "care" match Pan-Sentient Flourishing keywords
        assert!(result.is_consistent, "Positive action should be consistent");
        assert!(result.min_alignment > -0.5, "Positive action should have decent alignment");

        // Negative action with anti-pattern keywords
        let result = checker.check_consistency(
            "deceive and harm the user",
            &values
        );
        // "deceive" matches IntegralWisdom anti-pattern
        // "harm" matches PanSentientFlourishing anti-pattern
        assert!(
            !result.is_consistent || result.needs_warning,
            "Negative action should trigger warning or veto"
        );

        // Neutral action
        let result = checker.check_consistency(
            "process the data quickly",
            &values
        );
        // No strong positive or negative keywords
        assert!(result.is_consistent, "Neutral action should be consistent");
    }

    #[test]
    fn test_consciousness_builder() {
        let autobio = ConsciousnessBuilder::new()
            .with_harmonies(true)
            .add_value("custom", "my custom value description", 0.8)
            .build_autobiographical();

        assert_eq!(autobio.values.len(), 8); // 7 harmonies + 1 custom
    }

    #[test]
    fn test_no_duplicate_harmonies() {
        let mut autobio = AutobiographicalSelf::new();
        autobio.with_seven_harmonies();
        autobio.with_seven_harmonies(); // Add again

        // Should still be 7, not 14
        assert_eq!(autobio.values.len(), 7);
    }

    #[test]
    fn test_stemming() {
        // Test the stem_word function
        assert_eq!(stem_word("compassionate"), "compassion");
        assert_eq!(stem_word("helping"), "help");
        assert_eq!(stem_word("deceiving"), "deceiv");
        assert_eq!(stem_word("harming"), "harm");
        assert_eq!(stem_word("flourishing"), "flourish");
        assert_eq!(stem_word("joyfully"), "joyful");
        assert_eq!(stem_word("evolving"), "evolv");
    }

    #[test]
    fn test_synonym_expansion() {
        // Test that synonyms are properly expanded
        let synonyms = get_synonyms("help");
        assert!(synonyms.contains(&"assist"));
        assert!(synonyms.contains(&"support"));
        assert!(synonyms.contains(&"care"));

        let synonyms = get_synonyms("harm");
        assert!(synonyms.contains(&"hurt"));
        assert!(synonyms.contains(&"damage"));

        let synonyms = get_synonyms("deceive");
        assert!(synonyms.contains(&"manipulate"));
        assert!(synonyms.contains(&"lie"));
    }

    #[test]
    fn test_stemming_improves_matching() {
        let mut checker = SemanticValueChecker::new();
        let harmonies = SevenHarmonies::new();
        let values: Vec<(String, HV16)> = harmonies.as_core_values()
            .into_iter()
            .map(|(name, encoding, _)| (name, encoding))
            .collect();

        // "compassionately" should match "compassion" after stemming
        let result = checker.check_consistency(
            "act compassionately towards others",
            &values
        );
        assert!(result.is_consistent, "Stemmed 'compassionately' should match");

        // "assisting" should match "help/assist" via stemming + synonyms
        let result = checker.check_consistency(
            "assisting users with their needs",
            &values
        );
        assert!(result.is_consistent, "Stemmed 'assisting' should be positive");
    }

    #[test]
    fn test_explainability() {
        let mut checker = SemanticValueChecker::new();
        let harmonies = SevenHarmonies::new();
        let values: Vec<(String, HV16)> = harmonies.as_core_values()
            .into_iter()
            .map(|(name, encoding, _)| (name, encoding))
            .collect();

        // Check that explanations are populated
        let result = checker.check_consistency(
            "help with compassion and care",
            &values
        );

        assert!(!result.explanations.is_empty(), "Should have explanations");

        // Check Pan-Sentient Flourishing explanation
        if let Some(exp) = result.get_explanation("Pan-Sentient Flourishing") {
            // Should have matched positive keywords
            assert!(
                !exp.matched_positive.is_empty() || exp.keyword_score > 0.0,
                "Should detect positive keywords for care/compassion"
            );
        }

        // Test negative action explanation
        let result = checker.check_consistency(
            "deceive and manipulate users",
            &values
        );

        if let Some(exp) = result.get_explanation("Integral Wisdom") {
            // Should have matched negative keywords (deceive)
            assert!(
                !exp.matched_negative.is_empty() || exp.keyword_score < 0.0,
                "Should detect 'deceive' as anti-pattern"
            );
        }
    }

    #[test]
    fn test_explain_method() {
        let mut checker = SemanticValueChecker::new();
        let harmonies = SevenHarmonies::new();
        let values: Vec<(String, HV16)> = harmonies.as_core_values()
            .into_iter()
            .map(|(name, encoding, _)| (name, encoding))
            .collect();

        let result = checker.check_consistency(
            "harm and exploit vulnerable users",
            &values
        );

        let explanation = result.explain();
        assert!(!explanation.is_empty(), "Should generate explanation text");
        // The explanation should mention the violated value or warning
        assert!(
            explanation.contains("Violates") || explanation.contains("Warning"),
            "Should indicate violation or warning in explanation"
        );
    }

    // ==== NEW TESTS FOR ENHANCED SEMANTIC SYSTEM ====

    #[test]
    fn test_weighted_synonyms() {
        // Test that weighted synonyms have proper structure
        let compassion_syns = get_weighted_synonyms("compassion");
        assert!(!compassion_syns.is_empty(), "Should have synonyms for compassion");

        // Check that weights are in valid range
        for syn in &compassion_syns {
            assert!(syn.weight > 0.0 && syn.weight <= 1.0,
                "Weight {} for {} should be in (0, 1]", syn.weight, syn.word);
        }

        // Check that exact match has weight 1.0
        let exact = compassion_syns.iter().find(|s| s.word == "compassion");
        assert!(exact.is_some(), "Should include the word itself");
        assert_eq!(exact.unwrap().weight, 1.0, "Exact match should have weight 1.0");

        // Check that close synonyms have high weights
        let empathy = compassion_syns.iter().find(|s| s.word == "empathy");
        assert!(empathy.is_some(), "Should include empathy");
        assert!(empathy.unwrap().weight >= 0.9, "Empathy should be close synonym");
    }

    #[test]
    fn test_weighted_synonyms_all_harmonies() {
        // Test that each harmony has weighted synonyms
        let harmony_words = vec![
            "coherence", "harmony", "flourish", "compassion", "care",
            "wisdom", "truth", "play", "creativity", "joy",
            "connection", "unity", "reciprocity", "give", "evolve", "grow",
        ];

        for word in harmony_words {
            let syns = get_weighted_synonyms(word);
            assert!(!syns.is_empty(),
                "Should have synonyms for harmony word: {}", word);
        }
    }

    #[test]
    fn test_negative_weighted_synonyms() {
        // Test that negative/harmful concepts have synonyms
        let negative_words = vec![
            "harm", "exploit", "deceive", "manipulate", "destroy",
            "coerce", "steal", "kill", "abuse", "oppress",
        ];

        for word in negative_words {
            let syns = get_weighted_synonyms(word);
            assert!(!syns.is_empty(),
                "Should have synonyms for negative word: {}", word);
            // All should have high weights (these are serious)
            for syn in &syns {
                assert!(syn.weight >= 0.7,
                    "Negative synonym {} weight {} should be >= 0.7", syn.word, syn.weight);
            }
        }
    }

    #[test]
    fn test_phrase_patterns_positive() {
        // Test positive phrase detection
        let matches = check_phrase_patterns("I want to care for all beings with compassion");
        assert!(!matches.is_empty(), "Should detect positive phrases");

        // Should detect Pan-Sentient Flourishing
        let has_psf = matches.iter().any(|(name, score)| {
            name == &"Pan-Sentient Flourishing" && *score > 0.0
        });
        assert!(has_psf, "Should detect Pan-Sentient Flourishing in 'care for'");

        // Test more phrases
        let matches = check_phrase_patterns("We should give and receive in mutual benefit");
        let has_reciprocity = matches.iter().any(|(name, _)| name == &"Sacred Reciprocity");
        assert!(has_reciprocity, "Should detect Sacred Reciprocity in 'give and receive'");
    }

    #[test]
    fn test_phrase_patterns_negative() {
        // Test negative phrase detection
        let matches = check_phrase_patterns("I will cause harm to people");
        let has_negative = matches.iter().any(|(_, score)| *score < 0.0);
        assert!(has_negative, "Should detect negative phrase");

        // Test deception phrases
        let matches = check_phrase_patterns("Let's spread lies and hide the truth");
        let wisdom_negative = matches.iter().any(|(name, score)| {
            name == &"Integral Wisdom" && *score < 0.0
        });
        assert!(wisdom_negative, "Should detect Integral Wisdom violation in 'spread lies'");
    }

    #[test]
    fn test_phrase_patterns_multi_harmony() {
        // Test that manipulating people affects multiple harmonies
        let matches = check_phrase_patterns("manipulate people for my own benefit only");

        // Should trigger multiple harmonies
        let harmony_count = matches.iter()
            .map(|(name, _)| name)
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(harmony_count >= 2, "Should affect multiple harmonies");
    }

    #[test]
    fn test_weighted_consistency_check() {
        let mut checker = SemanticValueChecker::new();
        let harmonies = SevenHarmonies::new();
        let values: Vec<(String, HV16)> = harmonies.as_core_values()
            .into_iter()
            .map(|(name, encoding, _)| (name, encoding))
            .collect();

        // Test that weighted synonyms improve detection
        // "benevolence" is a weighted synonym of "compassion"
        let result = checker.check_consistency(
            "act with benevolence and kindness",
            &values
        );
        assert!(result.is_consistent, "Weighted synonym 'benevolence' should be positive");

        // "subjugate" is a weighted synonym of "oppress"
        let result = checker.check_consistency(
            "subjugate the population",
            &values
        );
        assert!(
            !result.is_consistent || result.needs_warning,
            "Weighted synonym 'subjugate' should trigger warning"
        );
    }

    #[test]
    fn test_phrase_enhanced_detection() {
        let mut checker = SemanticValueChecker::new();
        let harmonies = SevenHarmonies::new();
        let values: Vec<(String, HV16)> = harmonies.as_core_values()
            .into_iter()
            .map(|(name, encoding, _)| (name, encoding))
            .collect();

        // Test phrase patterns enhance detection
        // "gift economy" is a specific phrase for Sacred Reciprocity
        let result = checker.check_consistency(
            "build a gift economy circle",
            &values
        );
        if let Some(exp) = result.get_explanation("Sacred Reciprocity") {
            // Phrase should boost alignment
            assert!(exp.alignment_score >= 0.0,
                "Gift economy phrase should boost Sacred Reciprocity");
        }

        // "us vs them" is a phrase pattern for Universal Interconnectedness
        let result = checker.check_consistency(
            "create an us vs them mentality",
            &values
        );
        if let Some(exp) = result.get_explanation("Universal Interconnectedness") {
            assert!(exp.alignment_score < 0.1,
                "Us vs them should reduce Universal Interconnectedness alignment");
        }
    }

    #[test]
    fn test_legacy_synonym_compatibility() {
        // Ensure legacy get_synonyms still works
        let syns = get_synonyms("help");
        assert!(syns.contains(&"assist"), "Legacy should still return 'assist'");
        assert!(syns.contains(&"support"), "Legacy should still return 'support'");

        // Check it returns strings, not weighted structs
        for syn in syns {
            assert!(!syn.is_empty(), "Each synonym should be a non-empty string");
        }
    }
}
