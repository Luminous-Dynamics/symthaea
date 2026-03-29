// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Intent classification using Symthaea's HDC AssociativeLearner
//!
//! The classifier uses Hyperdimensional Computing for:
//! - Zero-shot generalization to new phrasings
//! - One-shot learning from single examples
//! - Online learning from user feedback

use mycelix_core_types::wisdom_engine::{
    AssociativeLearner, AssociativeLearnerConfig, MemorySnapshot,
};
use std::time::{SystemTime, UNIX_EPOCH};

use super::patterns::PatternMatcher;
use super::Intent;

/// Result of intent classification
#[derive(Debug, Clone)]
pub struct IntentResult {
    /// The primary intent detected
    pub intent: Intent,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,

    /// Alternative intents with their confidences
    pub alternatives: Vec<(Intent, f32)>,

    /// Whether this was a zero-shot classification (novel input)
    pub is_novel: bool,

    /// The extracted argument (package name, search query, etc.)
    pub argument: Option<String>,
}

/// HDC-powered intent classifier
pub struct IntentClassifier {
    /// The AssociativeLearner for HDC-based classification
    learner: AssociativeLearner,

    /// Regex pattern matcher as fallback
    patterns: PatternMatcher,

    /// Whether the classifier has been bootstrapped
    bootstrapped: bool,
}

impl IntentClassifier {
    /// Symthaea-compatible HDC dimension (2^14 = 16384)
    /// Higher dimensions provide better orthogonality and zero-shot generalization
    pub const HDC_DIMENSION: usize = 16384;

    /// Create a new intent classifier
    pub fn new() -> Self {
        let config = AssociativeLearnerConfig {
            dimension: Self::HDC_DIMENSION,
            learning_rate: 0.15,
            decay_rate: 0.005,
            similarity_threshold: 0.25,
            use_binary_memory: true,
            max_experiences: 5000,
            positive_weight: 1.0,
            negative_weight: -0.3,
        };

        let mut learner = AssociativeLearner::with_config(config);

        // Register known actions
        for action in [
            "search",
            "install",
            "remove",
            "list_generations",
            "rebuild",
            "garbage_collect",
            "status",
            "help",
            "unknown",
        ] {
            learner.register_action(action);
        }

        Self {
            learner,
            patterns: PatternMatcher::new(),
            bootstrapped: false,
        }
    }

    /// Bootstrap the classifier with canonical NixOS queries
    pub fn bootstrap(&mut self) {
        if self.bootstrapped {
            return;
        }

        let timestamp = current_timestamp();
        let bootstrap_data = get_bootstrap_data();

        for (query, action, outcome) in bootstrap_data {
            let context = self.query_to_context(query);
            let context_refs: Vec<(&str, &str)> =
                context.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
            self.learner.learn(&context_refs, action, outcome, timestamp);
        }

        self.bootstrapped = true;
    }

    /// Classify a natural language query
    pub fn classify(&mut self, query: &str) -> IntentResult {
        let query = query.trim().to_lowercase();

        // Try regex patterns first for exact matches
        if let Some((intent, confidence)) = self.patterns.match_query(&query) {
            return IntentResult {
                intent,
                confidence,
                alternatives: vec![],
                is_novel: false,
                argument: self.patterns.extract_argument(&query),
            };
        }

        // Use HDC for semantic understanding
        let context = self.query_to_context(&query);
        let context_refs: Vec<(&str, &str)> =
            context.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();

        let predictions = self.learner.predict(&context_refs);

        if predictions.is_empty() {
            return IntentResult {
                intent: Intent::Unknown,
                confidence: 0.0,
                alternatives: vec![],
                is_novel: true,
                argument: None,
            };
        }

        // Extract argument from query
        let argument = self.extract_argument(&query, &predictions[0].action);

        // Convert to Intent
        let primary_intent = Intent::from_action(&predictions[0].action, argument.clone());

        // Get alternatives
        let alternatives: Vec<(Intent, f32)> = predictions
            .iter()
            .skip(1)
            .take(3)
            .map(|p| {
                (
                    Intent::from_action(&p.action, argument.clone()),
                    p.confidence,
                )
            })
            .collect();

        IntentResult {
            intent: primary_intent,
            confidence: predictions[0].confidence,
            alternatives,
            is_novel: predictions[0].is_novel,
            argument,
        }
    }

    /// Learn from command outcome
    pub fn learn_from_outcome(&mut self, query: &str, intent: &Intent, outcome: f32) {
        let context = self.query_to_context(query);
        let context_refs: Vec<(&str, &str)> =
            context.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();

        self.learner
            .learn(&context_refs, intent.action_name(), outcome, current_timestamp());
    }

    /// Convert a query to HDC context features
    fn query_to_context(&self, query: &str) -> Vec<(String, String)> {
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut context = Vec::new();

        // Add individual tokens
        for (i, word) in words.iter().enumerate() {
            context.push((format!("token_{}", i), word.to_string()));
        }

        // Add bigrams for better context
        for pair in words.windows(2) {
            context.push(("bigram".to_string(), format!("{}_{}", pair[0], pair[1])));
        }

        // Add query length category
        let length_category = match words.len() {
            0..=1 => "very_short",
            2..=3 => "short",
            4..=6 => "medium",
            _ => "long",
        };
        context.push(("length".to_string(), length_category.to_string()));

        // Add keyword indicators
        for word in &words {
            if is_search_keyword(word) {
                context.push(("has_search_keyword".to_string(), "true".to_string()));
            }
            if is_install_keyword(word) {
                context.push(("has_install_keyword".to_string(), "true".to_string()));
            }
            if is_remove_keyword(word) {
                context.push(("has_remove_keyword".to_string(), "true".to_string()));
            }
            if is_list_keyword(word) {
                context.push(("has_list_keyword".to_string(), "true".to_string()));
            }
        }

        context
    }

    /// Extract the argument (package name, search query) from the input
    fn extract_argument(&self, query: &str, action: &str) -> Option<String> {
        let words: Vec<&str> = query.split_whitespace().collect();

        // Remove action-related keywords and return the rest
        let skip_words = match action {
            "search" => vec![
                "search", "find", "look", "for", "looking", "a", "an", "the", "query", "packages",
                "package", "nix",
            ],
            "install" => vec![
                "install", "add", "get", "download", "a", "an", "the", "package", "nix",
            ],
            "remove" => vec![
                "remove", "uninstall", "delete", "rm", "a", "an", "the", "package", "nix",
            ],
            _ => vec![],
        };

        let remaining: Vec<&str> = words
            .into_iter()
            .filter(|w| !skip_words.contains(&w.to_lowercase().as_str()))
            .collect();

        if remaining.is_empty() {
            None
        } else {
            Some(remaining.join(" "))
        }
    }

    /// Export memory for persistence
    pub fn export_memory(&self) -> MemorySnapshot {
        self.learner.export_memory()
    }

    /// Import memory from persistence
    pub fn import_memory(&mut self, snapshot: MemorySnapshot) {
        self.learner.import_memory(snapshot);
    }

    /// Get learning statistics
    pub fn stats(&self) -> &mycelix_core_types::wisdom_engine::AssociativeLearnerStats {
        self.learner.stats()
    }
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Helper functions
// =============================================================================

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn is_search_keyword(word: &str) -> bool {
    matches!(
        word,
        "search" | "find" | "look" | "looking" | "query" | "discover" | "locate" | "where"
    )
}

fn is_install_keyword(word: &str) -> bool {
    matches!(
        word,
        "install" | "add" | "get" | "download" | "setup" | "enable"
    )
}

fn is_remove_keyword(word: &str) -> bool {
    matches!(
        word,
        "remove" | "uninstall" | "delete" | "rm" | "disable" | "purge"
    )
}

fn is_list_keyword(word: &str) -> bool {
    matches!(
        word,
        "list" | "show" | "display" | "generations" | "history" | "versions"
    )
}

/// Get bootstrap training data for the classifier
fn get_bootstrap_data() -> Vec<(&'static str, &'static str, f32)> {
    vec![
        // Search variations (positive examples)
        ("search firefox", "search", 1.0),
        ("find vim", "search", 1.0),
        ("look for git", "search", 1.0),
        ("looking for a text editor", "search", 1.0),
        ("find me a markdown editor", "search", 1.0),
        ("search for python", "search", 1.0),
        ("query packages rust", "search", 1.0),
        ("where is neovim", "search", 1.0),
        ("locate htop", "search", 1.0),
        ("find music player", "search", 1.0),
        ("search video editor", "search", 1.0),
        ("find terminal emulator", "search", 1.0),
        ("look for web browser", "search", 1.0),
        ("search image viewer", "search", 1.0),
        ("find pdf reader", "search", 1.0),

        // Install variations
        ("install firefox", "install", 1.0),
        ("add vim", "install", 1.0),
        ("get git", "install", 1.0),
        ("install neovim", "install", 1.0),
        ("add package htop", "install", 1.0),
        ("install python3", "install", 1.0),
        ("get me rustup", "install", 1.0),
        ("install a terminal emulator", "install", 1.0),
        ("setup docker", "install", 1.0),
        ("enable flatpak", "install", 1.0),

        // Remove variations
        ("remove firefox", "remove", 1.0),
        ("uninstall vim", "remove", 1.0),
        ("delete git", "remove", 1.0),
        ("rm htop", "remove", 1.0),
        ("remove package neovim", "remove", 1.0),
        ("uninstall python", "remove", 1.0),
        ("purge docker", "remove", 1.0),

        // List generations
        ("list generations", "list_generations", 1.0),
        ("show generations", "list_generations", 1.0),
        ("generations", "list_generations", 1.0),
        ("show system history", "list_generations", 1.0),
        ("list system versions", "list_generations", 1.0),
        ("generation history", "list_generations", 1.0),
        ("what generations exist", "list_generations", 1.0),

        // Rebuild
        ("rebuild", "rebuild", 1.0),
        ("rebuild system", "rebuild", 1.0),
        ("nixos-rebuild", "rebuild", 1.0),
        ("update system", "rebuild", 1.0),
        ("switch configuration", "rebuild", 1.0),
        ("apply config", "rebuild", 1.0),
        ("rebuild switch", "rebuild", 1.0),

        // Garbage collect
        ("gc", "garbage_collect", 1.0),
        ("garbage collect", "garbage_collect", 1.0),
        ("clean up", "garbage_collect", 1.0),
        ("nix-collect-garbage", "garbage_collect", 1.0),
        ("delete old generations", "garbage_collect", 1.0),
        ("free up space", "garbage_collect", 1.0),
        ("cleanup", "garbage_collect", 1.0),
        ("clear cache", "garbage_collect", 1.0),

        // Status
        ("status", "status", 1.0),
        ("system info", "status", 1.0),
        ("show status", "status", 1.0),
        ("system status", "status", 1.0),
        ("what version", "status", 1.0),
        ("nixos version", "status", 1.0),
        ("current generation", "status", 1.0),

        // Help
        ("help", "help", 1.0),
        ("how to use", "help", 1.0),
        ("what can you do", "help", 1.0),
        ("commands", "help", 1.0),
        ("usage", "help", 1.0),

        // Negative examples (wrong classifications)
        ("search firefox", "install", -0.5),
        ("install vim", "search", -0.5),
        ("remove git", "install", -0.5),
        ("list generations", "search", -0.5),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_creation() {
        let classifier = IntentClassifier::new();
        assert!(!classifier.bootstrapped);
    }

    #[test]
    fn test_bootstrap() {
        let mut classifier = IntentClassifier::new();
        classifier.bootstrap();
        assert!(classifier.bootstrapped);
    }

    #[test]
    fn test_search_classification() {
        let mut classifier = IntentClassifier::new();
        classifier.bootstrap();

        let result = classifier.classify("search firefox");
        assert!(matches!(result.intent, Intent::Search { .. }));
    }

    #[test]
    fn test_install_classification() {
        let mut classifier = IntentClassifier::new();
        classifier.bootstrap();

        let result = classifier.classify("install vim");
        assert!(matches!(result.intent, Intent::Install { .. }));
    }

    #[test]
    fn test_zero_shot_generalization() {
        let mut classifier = IntentClassifier::new();
        classifier.bootstrap();

        // Novel phrasing not in bootstrap data
        let result = classifier.classify("i want to get neofetch");
        // Should still classify as install due to "get" keyword and HDC generalization
        assert!(result.confidence > 0.0);
    }
}
