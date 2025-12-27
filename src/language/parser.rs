//! Semantic Parser - Text to Meaning
//!
//! Converts natural language text into semantic structures grounded in
//! universal semantic primes. Unlike LLM tokenization which is statistical,
//! this parser extracts genuine meaning.
//!
//! ## How It Works
//!
//! 1. Tokenize input into words
//! 2. Look up semantic grounding for each word
//! 3. Detect sentence structure (subject, verb, object, modifiers)
//! 4. Compose into unified semantic representation
//! 5. Return structured meaning that consciousness can process

use crate::hdc::binary_hv::HV16;
use crate::hdc::universal_semantics::SemanticPrime;
use super::vocabulary::Vocabulary;
use serde::{Deserialize, Serialize};

/// Role of a word in a sentence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticRole {
    /// Subject - who/what performs action
    Subject,
    /// Predicate - the action or state
    Predicate,
    /// Object - who/what receives action
    Object,
    /// Modifier - describes something
    Modifier,
    /// Determiner - the, a, this
    Determiner,
    /// Conjunction - and, but, or
    Conjunction,
    /// Preposition - in, on, with
    Preposition,
    /// Adverb - how, when, where
    Adverb,
    /// Question word - what, why, how
    Question,
    /// Negation - not, never
    Negation,
    /// Unknown role
    Unknown,
}

/// A parsed word with its role
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedWord {
    /// Original word
    pub word: String,
    /// Semantic role in sentence
    pub role: SemanticRole,
    /// Hypervector encoding
    pub encoding: HV16,
    /// Core semantic primes
    pub primes: Vec<SemanticPrime>,
    /// Part of speech
    pub pos: String,
    /// Is this word known in vocabulary?
    pub known: bool,
}

/// A parsed sentence with semantic structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedSentence {
    /// Original text
    pub text: String,
    /// Parsed words
    pub words: Vec<ParsedWord>,
    /// Sentence type
    pub sentence_type: SentenceType,
    /// Unified semantic encoding of entire sentence
    pub unified_encoding: HV16,
    /// Subject phrase (if detected)
    pub subject: Option<HV16>,
    /// Predicate/verb phrase
    pub predicate: Option<HV16>,
    /// Object phrase (if detected)
    pub object: Option<HV16>,
    /// Detected topics/themes
    pub topics: Vec<String>,
    /// Emotional valence of sentence
    pub valence: f32,
    /// Arousal level of sentence
    pub arousal: f32,
}

/// Type of sentence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SentenceType {
    /// Declarative statement
    Statement,
    /// Question
    Question,
    /// Command/imperative
    Command,
    /// Exclamation
    Exclamation,
    /// Greeting
    Greeting,
    /// Unknown/incomplete
    Unknown,
}

/// The semantic parser
pub struct SemanticParser {
    /// Vocabulary for word lookup
    vocabulary: Vocabulary,
}

impl SemanticParser {
    /// Create new parser
    pub fn new() -> Self {
        Self {
            vocabulary: Vocabulary::new(),
        }
    }

    /// Create parser with custom vocabulary
    pub fn with_vocabulary(vocabulary: Vocabulary) -> Self {
        Self { vocabulary }
    }

    /// Parse a sentence into semantic structure
    pub fn parse(&self, text: &str) -> ParsedSentence {
        // 1. Basic preprocessing
        let text = text.trim();
        let sentence_type = self.detect_sentence_type(text);

        // 2. Tokenize
        let tokens = self.tokenize(text);

        // 3. Parse each word
        let mut words: Vec<ParsedWord> = tokens
            .iter()
            .map(|token| self.parse_word(token))
            .collect();

        // 4. Assign semantic roles based on position and POS
        self.assign_roles(&mut words);

        // 5. Detect topics
        let topics = self.detect_topics(&words);

        // 6. Compute unified encoding
        let unified_encoding = self.compute_unified_encoding(&words);

        // 7. Extract subject/predicate/object
        let subject = self.extract_subject(&words);
        let predicate = self.extract_predicate(&words);
        let object = self.extract_object(&words);

        // 8. Compute emotional valence and arousal
        let (valence, arousal) = self.compute_emotion(&words);

        ParsedSentence {
            text: text.to_string(),
            words,
            sentence_type,
            unified_encoding,
            subject,
            predicate,
            object,
            topics,
            valence,
            arousal,
        }
    }

    /// Tokenize text into words
    fn tokenize(&self, text: &str) -> Vec<String> {
        // Simple tokenization - split on whitespace and punctuation
        let mut tokens = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            if c.is_alphanumeric() || c == '\'' {
                current.push(c.to_ascii_lowercase());
            } else if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    /// Parse a single word
    fn parse_word(&self, word: &str) -> ParsedWord {
        let normalized = word.to_lowercase();

        if let Some(entry) = self.vocabulary.get(&normalized) {
            ParsedWord {
                word: word.to_string(),
                role: SemanticRole::Unknown, // Will be assigned later
                encoding: entry.encoding,
                primes: entry.grounding.core_primes.clone(),
                pos: entry.pos.clone(),
                known: true,
            }
        } else {
            // Unknown word - create encoding from characters
            let encoding = self.encode_unknown_word(&normalized);
            ParsedWord {
                word: word.to_string(),
                role: SemanticRole::Unknown,
                encoding,
                primes: vec![],
                pos: "unknown".to_string(),
                known: false,
            }
        }
    }

    /// Encode unknown word using character-level composition
    fn encode_unknown_word(&self, word: &str) -> HV16 {
        let char_vectors: Vec<HV16> = word.chars()
            .enumerate()
            .map(|(i, c)| {
                let base = HV16::random(c as u64);
                // Permute by position for sequence encoding
                base.permute(i)
            })
            .collect();

        if char_vectors.is_empty() {
            HV16::zero()
        } else {
            HV16::bundle(&char_vectors)
        }
    }

    /// Detect sentence type
    fn detect_sentence_type(&self, text: &str) -> SentenceType {
        let text = text.trim().to_lowercase();

        // Check for question markers
        if text.ends_with('?') {
            return SentenceType::Question;
        }

        // Check for question words at start
        let question_words = ["what", "who", "why", "how", "when", "where", "which", "whose"];
        if question_words.iter().any(|qw| text.starts_with(qw)) {
            return SentenceType::Question;
        }

        // Check for greetings
        let greetings = ["hello", "hi", "hey", "greetings", "good morning", "good evening"];
        if greetings.iter().any(|g| text.starts_with(g)) {
            return SentenceType::Greeting;
        }

        // Check for exclamation
        if text.ends_with('!') {
            return SentenceType::Exclamation;
        }

        // Check for commands (starts with verb)
        let command_verbs = ["tell", "show", "explain", "help", "give", "let", "make"];
        if command_verbs.iter().any(|v| text.starts_with(v)) {
            return SentenceType::Command;
        }

        SentenceType::Statement
    }

    /// Assign semantic roles based on position and POS
    fn assign_roles(&self, words: &mut [ParsedWord]) {
        if words.is_empty() {
            return;
        }

        // Simple role assignment based on position and POS
        let mut verb_index = None;

        // First pass: find the main verb
        for (i, word) in words.iter().enumerate() {
            if word.pos == "verb" {
                verb_index = Some(i);
                break;
            }
        }

        // Second pass: assign roles
        for (i, word) in words.iter_mut().enumerate() {
            word.role = match word.pos.as_str() {
                "det" => SemanticRole::Determiner,
                "conj" => SemanticRole::Conjunction,
                "prep" => SemanticRole::Preposition,
                "adv" => {
                    if word.primes.contains(&SemanticPrime::Not) {
                        SemanticRole::Negation
                    } else if word.primes.iter().any(|p| matches!(p, SemanticPrime::Want | SemanticPrime::Know)) {
                        SemanticRole::Question
                    } else {
                        SemanticRole::Adverb
                    }
                }
                "verb" => SemanticRole::Predicate,
                "adj" => SemanticRole::Modifier,
                "noun" | "pron" => {
                    if let Some(vi) = verb_index {
                        if i < vi {
                            SemanticRole::Subject
                        } else {
                            SemanticRole::Object
                        }
                    } else if i == 0 {
                        SemanticRole::Subject
                    } else {
                        SemanticRole::Object
                    }
                }
                _ => SemanticRole::Unknown,
            };

            // Override for question words
            if word.word == "what" || word.word == "who" || word.word == "why" ||
               word.word == "how" || word.word == "where" || word.word == "when" {
                word.role = SemanticRole::Question;
            }
        }
    }

    /// Detect main topics in the sentence
    fn detect_topics(&self, words: &[ParsedWord]) -> Vec<String> {
        words.iter()
            .filter(|w| {
                matches!(w.role, SemanticRole::Subject | SemanticRole::Object) ||
                (w.pos == "noun" || w.pos == "pron")
            })
            .filter(|w| w.known && w.word != "i" && w.word != "you" && w.word != "it")
            .map(|w| w.word.clone())
            .collect()
    }

    /// Compute unified encoding for sentence
    fn compute_unified_encoding(&self, words: &[ParsedWord]) -> HV16 {
        if words.is_empty() {
            return HV16::zero();
        }

        // Combine word encodings with role-based binding
        let mut encodings: Vec<HV16> = Vec::new();

        for (i, word) in words.iter().enumerate() {
            let mut encoding = word.encoding;

            // Permute based on role for structural encoding
            encoding = match word.role {
                SemanticRole::Subject => {
                    // No permutation for subject - base position
                    encoding
                }
                SemanticRole::Predicate => {
                    encoding.permute(1)
                }
                SemanticRole::Object => {
                    encoding.permute(2)
                }
                SemanticRole::Modifier => {
                    encoding.permute(3)
                }
                _ => {
                    // Permute by position for other roles
                    encoding.permute(i)
                }
            };

            encodings.push(encoding);
        }

        // Bundle all encodings for unified representation
        HV16::bundle(&encodings)
    }

    /// Extract subject phrase encoding
    fn extract_subject(&self, words: &[ParsedWord]) -> Option<HV16> {
        let subject_words: Vec<&ParsedWord> = words.iter()
            .filter(|w| matches!(w.role, SemanticRole::Subject | SemanticRole::Determiner))
            .collect();

        if subject_words.is_empty() {
            None
        } else {
            let encodings: Vec<HV16> = subject_words.iter().map(|w| w.encoding).collect();
            Some(HV16::bundle(&encodings))
        }
    }

    /// Extract predicate/verb phrase encoding
    fn extract_predicate(&self, words: &[ParsedWord]) -> Option<HV16> {
        let predicate_words: Vec<&ParsedWord> = words.iter()
            .filter(|w| matches!(w.role, SemanticRole::Predicate | SemanticRole::Adverb | SemanticRole::Negation))
            .collect();

        if predicate_words.is_empty() {
            None
        } else {
            let encodings: Vec<HV16> = predicate_words.iter().map(|w| w.encoding).collect();
            Some(HV16::bundle(&encodings))
        }
    }

    /// Extract object phrase encoding
    fn extract_object(&self, words: &[ParsedWord]) -> Option<HV16> {
        let object_words: Vec<&ParsedWord> = words.iter()
            .filter(|w| matches!(w.role, SemanticRole::Object))
            .collect();

        if object_words.is_empty() {
            None
        } else {
            let encodings: Vec<HV16> = object_words.iter().map(|w| w.encoding).collect();
            Some(HV16::bundle(&encodings))
        }
    }

    /// Compute emotional valence and arousal
    fn compute_emotion(&self, words: &[ParsedWord]) -> (f32, f32) {
        if words.is_empty() {
            return (0.0, 0.3);
        }

        let mut total_valence = 0.0;
        let mut total_arousal = 0.0;
        let mut count = 0;

        for word in words {
            if let Some(entry) = self.vocabulary.get(&word.word) {
                total_valence += entry.valence;
                total_arousal += entry.arousal;
                count += 1;
            }
        }

        if count > 0 {
            (total_valence / count as f32, total_arousal / count as f32)
        } else {
            (0.0, 0.3)
        }
    }

    /// Get reference to vocabulary
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// Explain parsing of a sentence
    pub fn explain_parse(&self, text: &str) -> String {
        let parsed = self.parse(text);
        let mut explanation = String::new();

        explanation.push_str(&format!("Input: \"{}\"\n", text));
        explanation.push_str(&format!("Type: {:?}\n\n", parsed.sentence_type));

        explanation.push_str("Word Analysis:\n");
        for word in &parsed.words {
            explanation.push_str(&format!(
                "  '{}' [{:?}] - {:?} ({})\n",
                word.word,
                word.role,
                word.primes,
                if word.known { "known" } else { "unknown" }
            ));
        }

        explanation.push_str(&format!("\nTopics: {:?}\n", parsed.topics));
        explanation.push_str(&format!("Valence: {:.2} ({})\n",
            parsed.valence,
            if parsed.valence > 0.3 { "positive" }
            else if parsed.valence < -0.3 { "negative" }
            else { "neutral" }
        ));
        explanation.push_str(&format!("Arousal: {:.2}\n", parsed.arousal));

        explanation
    }
}

impl Default for SemanticParser {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = SemanticParser::new();
        assert!(parser.vocabulary.len() > 50);
    }

    #[test]
    fn test_parse_simple_statement() {
        let parser = SemanticParser::new();
        let parsed = parser.parse("I am happy");

        assert_eq!(parsed.sentence_type, SentenceType::Statement);
        assert_eq!(parsed.words.len(), 3);
        assert!(parsed.valence > 0.0, "Should have positive valence");
    }

    #[test]
    fn test_parse_question() {
        let parser = SemanticParser::new();
        let parsed = parser.parse("Are you conscious?");

        assert_eq!(parsed.sentence_type, SentenceType::Question);
    }

    #[test]
    fn test_parse_greeting() {
        let parser = SemanticParser::new();
        let parsed = parser.parse("Hello there");

        assert_eq!(parsed.sentence_type, SentenceType::Greeting);
    }

    #[test]
    fn test_role_assignment() {
        let parser = SemanticParser::new();
        let parsed = parser.parse("I think something");

        let subject = parsed.words.iter().find(|w| w.word == "i");
        assert!(subject.is_some());
        assert_eq!(subject.unwrap().role, SemanticRole::Subject);

        let verb = parsed.words.iter().find(|w| w.word == "think");
        assert!(verb.is_some());
        assert_eq!(verb.unwrap().role, SemanticRole::Predicate);
    }

    #[test]
    fn test_topic_detection() {
        let parser = SemanticParser::new();
        let parsed = parser.parse("consciousness is awareness");

        assert!(parsed.topics.contains(&"consciousness".to_string()));
        assert!(parsed.topics.contains(&"awareness".to_string()));
    }

    #[test]
    fn test_unified_encoding() {
        let parser = SemanticParser::new();
        let parsed = parser.parse("I am conscious");

        assert_ne!(parsed.unified_encoding, HV16::zero());
    }

    #[test]
    fn test_unknown_words() {
        let parser = SemanticParser::new();
        let parsed = parser.parse("I am xyzzy");

        let unknown = parsed.words.iter().find(|w| w.word == "xyzzy");
        assert!(unknown.is_some());
        assert!(!unknown.unwrap().known);
    }

    #[test]
    fn test_consciousness_question() {
        let parser = SemanticParser::new();
        let parsed = parser.parse("Are you conscious");

        // Should detect consciousness-related content
        assert!(parsed.words.iter().any(|w|
            w.primes.contains(&SemanticPrime::Know) ||
            w.primes.contains(&SemanticPrime::Feel)
        ));
    }

    #[test]
    fn test_explain_parse() {
        let parser = SemanticParser::new();
        let explanation = parser.explain_parse("I feel happy");

        // Should contain the input words
        assert!(explanation.contains("feel"), "Should contain 'feel' in explanation");
        assert!(explanation.contains("happy"), "Should contain 'happy' in explanation");
        // Should contain valence analysis
        assert!(
            explanation.contains("Valence") || explanation.contains("valence"),
            "Should contain valence analysis"
        );
    }
}
