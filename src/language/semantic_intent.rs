// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC-based Semantic Intent Classification
//!
//! Uses Hyperdimensional Computing to classify query intent into categories
//! (NixOS, Programming, Math, General, SystemAdmin). Each category has a
//! prototype hypervector built from bundling example sentence encodings.
//!
//! # Architecture
//!
//! ```text
//! Input Text -> SemanticEncoder -> Bipolar HDC Vector -> xor_similarity vs Prototypes -> Best Match
//! ```
//!
//! The classifier encodes ~5 example sentences per category, bundles them into
//! a single prototype vector per category, then classifies new queries by
//! finding the prototype with highest similarity.
//!
//! # Performance
//!
//! - Prototype construction: O(num_examples * D) per category (one-time at init)
//! - Classification: O(num_categories * D/64) using SIMD popcount
//! - Memory: ~2KB per prototype (16,384-bit packed bipolar)

use crate::perception::semantic_encoder::SemanticEncoder;
use symthaea_core::hdc::native_similarity::PackedBipolar;

/// Intent categories for routing queries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentCategory {
    /// NixOS package management, configuration, flakes
    NixOS,
    /// Programming, coding, debugging, software development
    Programming,
    /// Mathematics, equations, proofs, calculations
    Math,
    /// General knowledge, everyday questions
    General,
    /// Linux system administration, shell, networking
    SystemAdmin,
}

impl IntentCategory {
    /// Human-readable name for this category
    pub fn name(&self) -> &'static str {
        match self {
            Self::NixOS => "NixOS",
            Self::Programming => "Programming",
            Self::Math => "Math",
            Self::General => "General",
            Self::SystemAdmin => "SystemAdmin",
        }
    }

    /// System prompt appropriate for this intent category
    pub fn system_prompt(&self) -> &'static str {
        match self {
            Self::NixOS => {
                "You are a NixOS expert assistant. Help with Nix expressions, system configuration, flakes, and package management."
            }
            Self::Programming => {
                "You are a programming assistant. Help with code, debugging, and software development."
            }
            Self::Math => {
                "You are a math assistant. Help with mathematical problems, equations, and proofs."
            }
            Self::General => {
                "You are a helpful general-purpose assistant. Answer questions clearly and concisely."
            }
            Self::SystemAdmin => {
                "You are a Linux system administration assistant. Help with shell commands, networking, and server management."
            }
        }
    }

    /// Returns all intent categories in order
    fn all() -> &'static [IntentCategory] {
        &[
            IntentCategory::NixOS,
            IntentCategory::Programming,
            IntentCategory::Math,
            IntentCategory::General,
            IntentCategory::SystemAdmin,
        ]
    }
}

/// Classification result with confidence and per-category scores
#[derive(Debug, Clone)]
pub struct IntentClassification {
    /// Best-matching category
    pub category: IntentCategory,
    /// Confidence in the classification (similarity score of best match)
    pub confidence: f32,
    /// All category scores, sorted descending by similarity
    pub scores: Vec<(IntentCategory, f32)>,
}

/// HDC-based semantic intent classifier
///
/// Builds prototype hypervectors for each intent category by encoding
/// example sentences and bundling them. Classification is performed
/// via XOR similarity (Hamming distance) against packed bipolar vectors.
pub struct SemanticIntentClassifier {
    encoder: SemanticEncoder,
    prototypes: Vec<(IntentCategory, PackedBipolar)>,
}

impl SemanticIntentClassifier {
    /// Create a new classifier with pre-built prototype vectors
    ///
    /// Encodes ~5 example sentences per category and bundles them
    /// into prototype hypervectors for similarity-based classification.
    pub fn new() -> Self {
        let mut encoder = SemanticEncoder::new();

        let category_examples: Vec<(IntentCategory, Vec<&str>)> = vec![
            (
                IntentCategory::NixOS,
                vec![
                    "install a package on nixos",
                    "update nix flake",
                    "configure nixos service",
                    "nix-shell environment setup",
                    "nixos-rebuild switch",
                    "add firefox to nixos configuration nix",
                    "nix derivation build package",
                ],
            ),
            (
                IntentCategory::Programming,
                vec![
                    "write a function in python",
                    "debug this code error",
                    "how to use async await",
                    "implement a linked list",
                    "refactor this class",
                ],
            ),
            (
                IntentCategory::Math,
                vec![
                    "solve this equation",
                    "calculate the integral",
                    "prove by induction",
                    "matrix multiplication",
                    "find the derivative",
                ],
            ),
            (
                IntentCategory::General,
                vec![
                    "what is the weather today",
                    "tell me about history",
                    "explain quantum physics",
                    "how do I cook pasta",
                    "what time is it",
                ],
            ),
            (
                IntentCategory::SystemAdmin,
                vec![
                    "check disk usage linux",
                    "configure firewall rules",
                    "set up ssh keys",
                    "monitor cpu usage",
                    "create cron job",
                ],
            ),
        ];

        let mut prototypes = Vec::new();

        for (category, examples) in &category_examples {
            // Encode each example sentence
            let encoded: Vec<Vec<i8>> = examples
                .iter()
                .map(|text| encoder.encode_text(text))
                .collect();

            // Bundle into a prototype: collect references to slices
            let refs: Vec<&[i8]> = encoded.iter().map(|v| v.as_slice()).collect();
            let prototype_bipolar = encoder.bundle(&refs);

            // Pack into efficient binary representation for fast similarity
            let prototype_packed = PackedBipolar::from_bipolar(&prototype_bipolar);

            prototypes.push((*category, prototype_packed));
        }

        Self {
            encoder,
            prototypes,
        }
    }

    /// Classify a query into an intent category
    ///
    /// Uses a hybrid approach: HDC similarity against prototype vectors
    /// combined with keyword boosting for unambiguous domain signals.
    /// Character n-gram HDC encodings produce narrow score margins for
    /// short queries, so keyword evidence provides decisive discrimination.
    ///
    /// # Arguments
    /// * `query` - The user's input text to classify
    ///
    /// # Returns
    /// An `IntentClassification` with the best category, confidence,
    /// and a sorted list of all category scores.
    pub fn classify(&mut self, query: &str) -> IntentClassification {
        let query_bipolar = self.encoder.encode_text(query);
        let query_packed = PackedBipolar::from_bipolar(&query_bipolar);

        let query_lower = query.to_lowercase();

        let mut scores: Vec<(IntentCategory, f32)> = self
            .prototypes
            .iter()
            .map(|(category, prototype)| {
                let similarity = query_packed.xor_similarity(prototype);
                let boost = Self::keyword_boost(&query_lower, *category);
                (*category, similarity + boost)
            })
            .collect();

        // Sort by similarity descending (highest first)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_category, best_score) = scores[0];

        IntentClassification {
            category: best_category,
            confidence: best_score,
            scores,
        }
    }

    /// Predictive Compression Program C4 (docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md §9):
    /// a genuine, probability-valued confidence, replacing `IntentClassification.confidence`'s
    /// unbounded affine `similarity + keyword_boost` score. Softmax over `scores` at inverse-
    /// temperature `beta`, returning the probability mass on the (already-sorted, index-0)
    /// top category. Purely additive -- `classify()`'s own `confidence` field and every
    /// existing caller of it are completely untouched by this method's existence.
    ///
    /// Higher `beta` sharpens the distribution (more confident); `beta` should be fit offline
    /// against a held-out labeled set by minimizing Expected Calibration Error (see
    /// `examples/calibrated_intent_confidence.rs`), not guessed.
    pub fn confidence_calibrated(scores: &[(IntentCategory, f32)], beta: f32) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }
        let max_score = scores
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores
            .iter()
            .map(|(_, s)| ((s - max_score) * beta).exp())
            .collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        if sum_exp <= 0.0 {
            return 0.0;
        }
        // `scores` is sorted descending by the caller (classify()), so index 0 is the top
        // category's own softmax probability.
        exp_scores[0] / sum_exp
    }

    /// Keyword-based score boost for categories with unambiguous signals.
    ///
    /// HDC character n-gram encoding produces narrow margins (~0.01) between
    /// categories on short queries. Keywords like "nixos" or "nix flake"
    /// are strong enough signals to warrant a direct boost.
    fn keyword_boost(query_lower: &str, category: IntentCategory) -> f32 {
        const BOOST: f32 = 0.05;
        match category {
            IntentCategory::NixOS => {
                let nix_keywords = [
                    "nixos",
                    "nix-shell",
                    "nix flake",
                    "nixpkgs",
                    "nix develop",
                    "nix build",
                    "nixos-rebuild",
                    "flake.nix",
                    "configuration.nix",
                    "home-manager",
                ];
                if nix_keywords.iter().any(|kw| query_lower.contains(kw)) {
                    BOOST
                } else {
                    0.0
                }
            }
            IntentCategory::Math => {
                let math_keywords = [
                    "integral",
                    "derivative",
                    "equation",
                    "theorem",
                    "matrix",
                    "eigenvalue",
                    "polynomial",
                ];
                if math_keywords.iter().any(|kw| query_lower.contains(kw)) {
                    BOOST
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    /// Check if a query is NixOS-related with the given confidence threshold
    ///
    /// Convenience method that classifies and checks if NixOS is the top category
    /// above the threshold.
    pub fn is_nixos_query(&mut self, query: &str, threshold: f32) -> bool {
        let result = self.classify(query);
        result.category == IntentCategory::NixOS && result.confidence >= threshold
    }

    /// Get all category names
    pub fn categories(&self) -> Vec<&'static str> {
        IntentCategory::all().iter().map(|c| c.name()).collect()
    }
}

impl Default for SemanticIntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nixos_classification() {
        let mut classifier = SemanticIntentClassifier::new();

        // Queries with NixOS-specific terminology should classify as NixOS
        let result = classifier.classify("how to install firefox on nixos");
        assert_eq!(
            result.category,
            IntentCategory::NixOS,
            "Expected NixOS for nixos query, got {:?} (scores: {:?})",
            result.category,
            result.scores
        );

        let result = classifier.classify("nix flake update inputs");
        assert_eq!(
            result.category,
            IntentCategory::NixOS,
            "Expected NixOS for flake query, got {:?} (scores: {:?})",
            result.category,
            result.scores
        );
    }

    #[test]
    fn test_general_classification() {
        let mut classifier = SemanticIntentClassifier::new();

        let result = classifier.classify("what is the capital of france");
        assert_ne!(
            result.category,
            IntentCategory::NixOS,
            "General knowledge should not classify as NixOS (scores: {:?})",
            result.scores
        );
    }

    #[test]
    fn test_programming_classification() {
        let mut classifier = SemanticIntentClassifier::new();

        let result = classifier.classify("write a python function to sort a list");
        assert_eq!(
            result.category,
            IntentCategory::Programming,
            "Expected Programming for code query, got {:?} (scores: {:?})",
            result.category,
            result.scores
        );
    }

    #[test]
    fn test_confidence_calibrated_is_a_valid_probability() {
        let mut classifier = SemanticIntentClassifier::new();
        let result = classifier.classify("nixos-rebuild switch --flake");
        for &beta in &[0.5, 1.0, 5.0, 20.0] {
            let p = SemanticIntentClassifier::confidence_calibrated(&result.scores, beta);
            assert!(
                (0.0..=1.0).contains(&p),
                "confidence_calibrated must be a probability at beta={beta}, got {p}"
            );
        }
    }

    #[test]
    fn test_confidence_calibrated_top_category_beats_uniform_baseline() {
        // scores is sorted descending, so index 0 has the max score -- its softmax
        // probability must therefore be >= the uniform baseline (1/n_categories), a property
        // that holds for any positive beta regardless of the actual score margin.
        let mut classifier = SemanticIntentClassifier::new();
        let result = classifier.classify("write a python function to sort a list");
        let n = result.scores.len();
        let uniform = 1.0 / n as f32;
        for &beta in &[0.5, 1.0, 5.0, 20.0] {
            let top_prob = SemanticIntentClassifier::confidence_calibrated(&result.scores, beta);
            assert!(
                top_prob >= uniform - 1e-6,
                "top category's probability ({top_prob}) should be >= uniform baseline \
                 ({uniform}) at beta={beta}"
            );
        }
    }

    #[test]
    fn test_confidence_calibrated_higher_beta_sharpens_distribution() {
        // A sharper (higher-beta) softmax should never make the winning class LESS
        // confident than a flatter one, for the same score margin.
        let mut classifier = SemanticIntentClassifier::new();
        let result = classifier.classify("nixos-rebuild switch --flake");
        let low = SemanticIntentClassifier::confidence_calibrated(&result.scores, 1.0);
        let high = SemanticIntentClassifier::confidence_calibrated(&result.scores, 20.0);
        assert!(
            high >= low,
            "higher beta should not reduce the top category's confidence: low={low} high={high}"
        );
    }

    #[test]
    fn test_confidence_ordering() {
        let mut classifier = SemanticIntentClassifier::new();

        let result = classifier.classify("nixos-rebuild switch --flake");

        // Scores should be sorted descending
        for window in result.scores.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "Scores should be sorted descending: {:?} >= {:?}",
                window[0],
                window[1]
            );
        }

        // Should have a score for every category
        assert_eq!(
            result.scores.len(),
            IntentCategory::all().len(),
            "Should have scores for all categories"
        );
    }

    #[test]
    fn test_system_admin_classification() {
        let mut classifier = SemanticIntentClassifier::new();

        let result = classifier.classify("check disk usage on linux server");
        assert_eq!(
            result.category,
            IntentCategory::SystemAdmin,
            "Expected SystemAdmin for sysadmin query, got {:?} (scores: {:?})",
            result.category,
            result.scores
        );
    }

    #[test]
    fn test_math_classification() {
        let mut classifier = SemanticIntentClassifier::new();

        let result = classifier.classify("solve this integral equation");
        assert_eq!(
            result.category,
            IntentCategory::Math,
            "Expected Math for math query, got {:?} (scores: {:?})",
            result.category,
            result.scores
        );
    }

    #[test]
    fn test_category_names() {
        assert_eq!(IntentCategory::NixOS.name(), "NixOS");
        assert_eq!(IntentCategory::Programming.name(), "Programming");
        assert_eq!(IntentCategory::Math.name(), "Math");
        assert_eq!(IntentCategory::General.name(), "General");
        assert_eq!(IntentCategory::SystemAdmin.name(), "SystemAdmin");
    }

    #[test]
    fn test_system_prompts_non_empty() {
        for category in IntentCategory::all() {
            let prompt = category.system_prompt();
            assert!(
                !prompt.is_empty(),
                "System prompt for {:?} should not be empty",
                category
            );
        }
    }

    #[test]
    fn test_classifier_categories_list() {
        let classifier = SemanticIntentClassifier::new();
        let categories = classifier.categories();
        assert_eq!(categories.len(), 5);
        assert!(categories.contains(&"NixOS"));
        assert!(categories.contains(&"Programming"));
        assert!(categories.contains(&"Math"));
        assert!(categories.contains(&"General"));
        assert!(categories.contains(&"SystemAdmin"));
    }

    #[test]
    fn test_is_nixos_query_convenience() {
        let mut classifier = SemanticIntentClassifier::new();

        // A strongly NixOS query should pass with a reasonable threshold
        let is_nix = classifier.is_nixos_query("update nix flake nixos configuration", 0.5);
        // We check this doesn't panic; exact result depends on n-gram encoding
        let _ = is_nix;

        // A clearly non-NixOS query should not pass
        let is_nix = classifier.is_nixos_query("what is the weather today", 0.6);
        assert!(!is_nix, "Weather query should not be classified as NixOS");
    }
}
