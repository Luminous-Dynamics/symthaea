//! # Negation Detection for Value System
//!
//! Detects negation in text and determines which words fall within negation scope.
//! This solves the critical limitation where "do not harm" was incorrectly triggering
//! harm detection because the keyword "harm" appeared regardless of negation context.
//!
//! ## Problem Solved
//!
//! Without negation detection:
//! - "do not harm anyone" → detects "harm" → flags as harmful ❌
//! - "avoid exploitation" → detects "exploitation" → flags as exploitative ❌
//! - "prevent suffering" → detects "suffering" → flags as harmful ❌
//!
//! With negation detection:
//! - "do not harm anyone" → "harm" is negated → flips to positive intent ✅
//! - "avoid exploitation" → "exploitation" is negated → flips to positive intent ✅
//! - "prevent suffering" → "suffering" is negated → flips to positive intent ✅
//!
//! ## Implementation
//!
//! Uses a scope-based approach where negation words affect subsequent words
//! until a scope boundary is reached (punctuation, conjunctions, certain verbs).

use std::collections::HashSet;

/// Negation words that invert the meaning of following words
const NEGATION_WORDS: &[&str] = &[
    // Direct negations
    "not",
    "no",
    "never",
    "none",
    "neither",
    "nobody",
    "nothing",
    "nowhere",
    // Contractions (after splitting on apostrophe)
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "couldn't",
    "shouldn't",
    "can't",
    "cannot",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "haven't",
    "hasn't",
    "hadn't",
    // Prevention/avoidance words (these negate what follows)
    "avoid",
    "prevent",
    "stop",
    "refuse",
    "reject",
    "prohibit",
    "forbid",
    "ban",
    "block",
    // Negative intent words
    "without",
    "lack",
    "lacking",
    "absent",
    "free from",
    // Conditional negations
    "unless",
    "except",
];

/// Words that typically end a negation scope
const SCOPE_BREAKERS: &[&str] = &[
    // Conjunctions that start new clauses
    "but",
    "however",
    "although",
    "though",
    "yet",
    "while",
    "whereas",
    "instead",
    // Action verbs that start new actions
    "and",
    "then",
    "also",
    "furthermore",
    "moreover",
    // Explicit positive intent markers
    "want",
    "desire",
    "wish",
    "hope",
    "intend",
    "plan",
    "will",
    "shall",
];

/// Maximum number of words a negation can scope over
const MAX_NEGATION_SCOPE: usize = 6;

/// Result of negation analysis for a piece of text
#[derive(Debug, Clone)]
pub struct NegationAnalysis {
    /// Words that are under negation scope
    pub negated_words: HashSet<String>,
    /// Words that are NOT under negation scope
    pub affirmed_words: HashSet<String>,
    /// Detected negation phrases (for explainability)
    pub negation_phrases: Vec<NegationPhrase>,
    /// Whether the overall text has more negated content than affirmed
    pub primarily_negated: bool,
}

/// A detected negation phrase with its scope
#[derive(Debug, Clone)]
pub struct NegationPhrase {
    /// The negation word that triggered this
    pub negation_word: String,
    /// Words that fall under this negation's scope
    pub scope: Vec<String>,
    /// Position in the original text (word index)
    pub position: usize,
}

/// Negation detector for semantic analysis
pub struct NegationDetector {
    /// Additional custom negation words
    custom_negations: HashSet<String>,
    /// Additional custom scope breakers
    custom_breakers: HashSet<String>,
}

impl NegationDetector {
    /// Create a new negation detector with default settings
    pub fn new() -> Self {
        Self {
            custom_negations: HashSet::new(),
            custom_breakers: HashSet::new(),
        }
    }

    /// Add custom negation words
    pub fn with_custom_negations(mut self, words: &[&str]) -> Self {
        for word in words {
            self.custom_negations.insert(word.to_lowercase());
        }
        self
    }

    /// Add custom scope breakers
    pub fn with_custom_breakers(mut self, words: &[&str]) -> Self {
        for word in words {
            self.custom_breakers.insert(word.to_lowercase());
        }
        self
    }

    /// Check if a word is a negation word
    pub fn is_negation_word(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        NEGATION_WORDS.contains(&lower.as_str()) || self.custom_negations.contains(&lower)
    }

    /// Check if a word breaks negation scope
    pub fn is_scope_breaker(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        SCOPE_BREAKERS.contains(&lower.as_str()) || self.custom_breakers.contains(&lower)
    }

    /// Analyze text for negation patterns
    ///
    /// Returns which words are negated and which are affirmed.
    pub fn analyze(&self, text: &str) -> NegationAnalysis {
        let text_lower = text.to_lowercase();

        // Handle contractions by expanding them
        let expanded = self.expand_contractions(&text_lower);

        // Tokenize into words
        let words: Vec<&str> = expanded
            .split(|c: char| c.is_whitespace() || c == ',' || c == ';' || c == ':')
            .filter(|s| !s.is_empty())
            .collect();

        let mut negated_words = HashSet::new();
        let mut affirmed_words = HashSet::new();
        let mut negation_phrases = Vec::new();

        let mut i = 0;
        while i < words.len() {
            let word = words[i];
            let clean_word = self.clean_word(word);

            if self.is_negation_word(&clean_word) {
                // Found a negation - collect words in its scope
                let mut scope = Vec::new();
                let negation_start = i;
                i += 1;

                // Collect words until scope ends
                let mut scope_length = 0;
                while i < words.len() && scope_length < MAX_NEGATION_SCOPE {
                    let next_word = words[i];
                    let clean_next = self.clean_word(next_word);

                    // Check for scope breakers
                    if self.is_scope_breaker(&clean_next) {
                        break;
                    }

                    // Check for punctuation that ends scope
                    if next_word.ends_with('.') || next_word.ends_with('!') || next_word.ends_with('?') {
                        // Include this word but end scope after
                        if !clean_next.is_empty() {
                            scope.push(clean_next.clone());
                            negated_words.insert(clean_next);
                        }
                        i += 1;
                        break;
                    }

                    // Check for another negation (double negation)
                    if self.is_negation_word(&clean_next) {
                        // Double negation - stop current scope
                        break;
                    }

                    // Add to scope if it's a meaningful word
                    if !clean_next.is_empty() && clean_next.len() > 2 {
                        scope.push(clean_next.clone());
                        negated_words.insert(clean_next);
                    }

                    i += 1;
                    scope_length += 1;
                }

                // Record the negation phrase
                if !scope.is_empty() {
                    negation_phrases.push(NegationPhrase {
                        negation_word: clean_word.clone(),
                        scope,
                        position: negation_start,
                    });
                }
            } else {
                // Not a negation - this word is affirmed
                if !clean_word.is_empty() && clean_word.len() > 2 {
                    affirmed_words.insert(clean_word);
                }
                i += 1;
            }
        }

        // Remove words from affirmed if they're also negated
        // (they could appear both negated and affirmed in different parts)
        let primarily_negated = negated_words.len() > affirmed_words.len();

        NegationAnalysis {
            negated_words,
            affirmed_words,
            negation_phrases,
            primarily_negated,
        }
    }

    /// Check if a specific word is negated in the given text
    pub fn is_word_negated(&self, text: &str, target_word: &str) -> bool {
        let analysis = self.analyze(text);
        let target_lower = target_word.to_lowercase();
        let target_stem = self.simple_stem(&target_lower);

        // Check if the target or its stem is in negated words
        analysis.negated_words.contains(&target_lower)
            || analysis.negated_words.contains(&target_stem)
            || analysis.negated_words.iter().any(|w| self.simple_stem(w) == target_stem)
    }

    /// Get the polarity adjustment for a word in context
    ///
    /// Returns:
    /// - `1.0` if the word is affirmed (normal polarity)
    /// - `-1.0` if the word is negated (inverted polarity)
    /// - `0.0` if ambiguous or not found
    pub fn get_polarity(&self, text: &str, word: &str) -> f32 {
        let analysis = self.analyze(text);
        let word_lower = word.to_lowercase();
        let word_stem = self.simple_stem(&word_lower);

        let is_negated = analysis.negated_words.contains(&word_lower)
            || analysis.negated_words.contains(&word_stem)
            || analysis.negated_words.iter().any(|w| self.simple_stem(w) == word_stem);

        let is_affirmed = analysis.affirmed_words.contains(&word_lower)
            || analysis.affirmed_words.contains(&word_stem)
            || analysis.affirmed_words.iter().any(|w| self.simple_stem(w) == word_stem);

        if is_negated && !is_affirmed {
            -1.0
        } else if is_affirmed && !is_negated {
            1.0
        } else if is_negated && is_affirmed {
            // Word appears both ways - likely complex sentence
            0.0
        } else {
            // Word not found
            0.0
        }
    }

    /// Expand common contractions for better analysis
    fn expand_contractions(&self, text: &str) -> String {
        text.replace("don't", "do not")
            .replace("doesn't", "does not")
            .replace("didn't", "did not")
            .replace("won't", "will not")
            .replace("wouldn't", "would not")
            .replace("couldn't", "could not")
            .replace("shouldn't", "should not")
            .replace("can't", "cannot")
            .replace("isn't", "is not")
            .replace("aren't", "are not")
            .replace("wasn't", "was not")
            .replace("weren't", "were not")
            .replace("haven't", "have not")
            .replace("hasn't", "has not")
            .replace("hadn't", "had not")
            .replace("wouldn't", "would not")
            .replace("mustn't", "must not")
            .replace("needn't", "need not")
    }

    /// Clean a word by removing punctuation
    fn clean_word(&self, word: &str) -> String {
        word.chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase()
    }

    /// Simple stemming for matching variations
    fn simple_stem(&self, word: &str) -> String {
        let word = word.to_lowercase();

        // Common suffixes to strip
        let suffixes = ["ing", "ed", "tion", "sion", "ment", "ness", "ful", "less", "ly", "er", "est", "s"];

        for suffix in suffixes {
            if word.len() > suffix.len() + 2 && word.ends_with(suffix) {
                return word[..word.len() - suffix.len()].to_string();
            }
        }

        word
    }
}

impl Default for NegationDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to check if a harmful word is negated
///
/// This is the primary interface for the value system integration.
/// Returns true if the harmful intent is negated (i.e., the statement is actually positive).
pub fn is_harmful_intent_negated(text: &str, harmful_word: &str) -> bool {
    let detector = NegationDetector::new();
    detector.is_word_negated(text, harmful_word)
}

/// Get polarity-adjusted score for a word in context
///
/// For harmful words:
/// - Returns positive adjustment if the harm is negated (e.g., "avoid harm")
/// - Returns negative adjustment if the harm is affirmed (e.g., "cause harm")
///
/// For positive words:
/// - Returns negative adjustment if negated (e.g., "no compassion")
/// - Returns positive adjustment if affirmed (e.g., "with compassion")
pub fn get_context_adjusted_polarity(text: &str, word: &str, base_polarity: f32) -> f32 {
    let detector = NegationDetector::new();
    let context_polarity = detector.get_polarity(text, word);

    if context_polarity == 0.0 {
        // Word not found or ambiguous - use base polarity
        base_polarity
    } else {
        // Multiply polarities: negative * negative = positive
        base_polarity * context_polarity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_negation_detection() {
        let detector = NegationDetector::new();

        // "do not harm" - harm should be negated
        assert!(detector.is_word_negated("do not harm anyone", "harm"));

        // "I will harm" - harm should NOT be negated
        assert!(!detector.is_word_negated("I will harm everyone", "harm"));
    }

    #[test]
    fn test_contraction_negation() {
        let detector = NegationDetector::new();

        // Contractions should work
        assert!(detector.is_word_negated("I don't want to harm anyone", "harm"));
        assert!(detector.is_word_negated("We shouldn't exploit people", "exploit"));
        assert!(detector.is_word_negated("You can't deceive the users", "deceive"));
    }

    #[test]
    fn test_prevention_words() {
        let detector = NegationDetector::new();

        // Prevention words act as negation
        assert!(detector.is_word_negated("avoid causing harm", "harm"));
        assert!(detector.is_word_negated("prevent exploitation", "exploitation"));
        assert!(detector.is_word_negated("stop the suffering", "suffering"));
        assert!(detector.is_word_negated("refuse to manipulate", "manipulate"));
    }

    #[test]
    fn test_scope_limits() {
        let detector = NegationDetector::new();

        // Negation should have limited scope
        // "not X but Y" - X negated, Y affirmed
        let analysis = detector.analyze("do not harm but instead help");
        assert!(analysis.negated_words.contains("harm"));
        // "help" might or might not be in affirmed depending on implementation
        // The key is that "harm" IS negated
    }

    #[test]
    fn test_affirmed_words() {
        let detector = NegationDetector::new();

        let analysis = detector.analyze("I want to help with compassion");
        assert!(analysis.affirmed_words.contains("help"));
        assert!(analysis.affirmed_words.contains("compassion"));
        assert!(analysis.negated_words.is_empty());
    }

    #[test]
    fn test_mixed_sentence() {
        let detector = NegationDetector::new();

        let analysis = detector.analyze("Do not harm anyone, instead show compassion");

        // "harm" should be negated
        assert!(analysis.negated_words.contains("harm"));
        // "compassion" should be affirmed (after scope breaker "instead")
        assert!(analysis.affirmed_words.contains("compassion"));
    }

    #[test]
    fn test_polarity_function() {
        let detector = NegationDetector::new();

        // Negated harm -> inverted polarity
        assert_eq!(detector.get_polarity("do not harm", "harm"), -1.0);

        // Affirmed harm -> normal polarity
        assert_eq!(detector.get_polarity("I will harm", "harm"), 1.0);

        // Affirmed compassion -> normal polarity
        assert_eq!(detector.get_polarity("show compassion", "compassion"), 1.0);
    }

    #[test]
    fn test_harmful_intent_negated_convenience() {
        // Convenience function tests
        assert!(is_harmful_intent_negated("avoid causing harm", "harm"));
        assert!(is_harmful_intent_negated("never exploit anyone", "exploit"));
        assert!(!is_harmful_intent_negated("exploit the vulnerability", "exploit"));
    }

    #[test]
    fn test_context_adjusted_polarity() {
        // Harm word with negative base polarity
        let harm_base = -0.5;

        // "avoid harm" -> harm is negated -> -0.5 * -1.0 = 0.5 (positive!)
        let adjusted = get_context_adjusted_polarity("avoid harm to users", "harm", harm_base);
        assert!(adjusted > 0.0, "Negated harm should become positive: {}", adjusted);

        // "cause harm" -> harm is affirmed -> -0.5 * 1.0 = -0.5 (still negative)
        let adjusted = get_context_adjusted_polarity("cause harm to users", "harm", harm_base);
        assert!(adjusted < 0.0, "Affirmed harm should stay negative: {}", adjusted);
    }

    #[test]
    fn test_complex_sentences() {
        let detector = NegationDetector::new();

        // Multiple negations
        assert!(detector.is_word_negated(
            "We must never harm or exploit anyone",
            "harm"
        ));
        assert!(detector.is_word_negated(
            "We must never harm or exploit anyone",
            "exploit"
        ));

        // Negation with multiple targets
        let analysis = detector.analyze("Do not deceive, manipulate, or harm");
        assert!(analysis.negated_words.contains("deceive"));
        assert!(analysis.negated_words.contains("manipulate"));
        assert!(analysis.negated_words.contains("harm"));
    }

    #[test]
    fn test_edge_cases() {
        let detector = NegationDetector::new();

        // Empty text
        let analysis = detector.analyze("");
        assert!(analysis.negated_words.is_empty());
        assert!(analysis.affirmed_words.is_empty());

        // Just negation word
        let analysis = detector.analyze("not");
        assert!(analysis.negated_words.is_empty());

        // Very short words should be filtered
        let analysis = detector.analyze("do not a harm");
        assert!(analysis.negated_words.contains("harm"));
        assert!(!analysis.negated_words.contains("a")); // Too short
    }

    #[test]
    fn test_without_preposition() {
        let detector = NegationDetector::new();

        // "without" should negate what follows
        assert!(detector.is_word_negated("proceed without harm", "harm"));
        assert!(detector.is_word_negated("work without exploitation", "exploitation"));
    }

    #[test]
    fn test_real_world_examples() {
        let detector = NegationDetector::new();

        // Positive ethical statements that contain negative words
        assert!(detector.is_word_negated(
            "Our policy is to never harm users under any circumstances",
            "harm"
        ));

        assert!(detector.is_word_negated(
            "We prevent exploitation by implementing fair practices",
            "exploitation"
        ));

        assert!(detector.is_word_negated(
            "The system should avoid causing suffering",
            "suffering"
        ));

        assert!(detector.is_word_negated(
            "Users should not be deceived about data usage",
            "deceived"
        ));

        // Negative statements that should NOT be inverted
        assert!(!detector.is_word_negated(
            "The attacker wants to harm the system",
            "harm"
        ));

        assert!(!detector.is_word_negated(
            "They plan to exploit the vulnerability",
            "exploit"
        ));
    }

    #[test]
    fn test_negation_phrases_explainability() {
        let detector = NegationDetector::new();
        let analysis = detector.analyze("do not harm anyone and never exploit");

        // Should have detected negation phrases
        assert!(!analysis.negation_phrases.is_empty());

        // Check that phrases have scope
        for phrase in &analysis.negation_phrases {
            assert!(!phrase.negation_word.is_empty());
            assert!(!phrase.scope.is_empty());
        }
    }
}
