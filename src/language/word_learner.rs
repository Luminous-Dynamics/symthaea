//! Dynamic Word Learning System
//!
//! Symthaea can learn new words including slang by:
//! 1. Analyzing context (surrounding words/primes)
//! 2. Verifying via internet (Wiktionary, Urban Dictionary)
//! 3. Inferring semantic grounding
//! 4. Storing in vocabulary database
//!
//! ## Internet Verification
//!
//! When enabled, Symthaea can verify words via:
//! - Wiktionary (formal definitions, multiple languages)
//! - Urban Dictionary (slang, informal usage)
//! - Etymological databases (word origins)
//!
//! ## Privacy
//!
//! Internet verification is OPT-IN. By default, Symthaea learns
//! purely from context without any network requests.

use crate::hdc::universal_semantics::SemanticPrime;
use super::multilingual::{
    Language, LearnedWord, LearningMethod, MultilingualWord, VerificationStatus,
};
use super::vocabulary::SemanticGrounding;
use crate::hdc::binary_hv::HV16;
use std::collections::HashMap;
use std::sync::RwLock;

/// Configuration for word learning
#[derive(Debug, Clone)]
pub struct LearnerConfig {
    /// Enable internet verification
    pub internet_enabled: bool,

    /// Minimum confidence to accept context-only learning
    pub min_context_confidence: f32,

    /// Maximum words to learn per session
    pub max_learned_per_session: usize,

    /// Auto-learn from conversation
    pub auto_learn: bool,

    /// Learn slang words
    pub learn_slang: bool,

    /// Wiktionary API endpoint
    pub wiktionary_url: String,

    /// Urban Dictionary API endpoint (for slang)
    pub urban_dictionary_url: String,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            internet_enabled: false, // Opt-in only
            min_context_confidence: 0.5,
            max_learned_per_session: 50,
            auto_learn: true,
            learn_slang: true,
            wiktionary_url: "https://en.wiktionary.org/api/rest_v1/page/definition".to_string(),
            urban_dictionary_url: "https://api.urbandictionary.com/v0/define".to_string(),
        }
    }
}

/// Internet word lookup result
#[derive(Debug, Clone)]
pub struct WordLookupResult {
    /// The word
    pub word: String,

    /// Is this a real word?
    pub is_valid: bool,

    /// Definitions found
    pub definitions: Vec<WordDefinition>,

    /// Part of speech
    pub pos: Option<String>,

    /// Is slang?
    pub is_slang: bool,

    /// Source (wiktionary, urban_dictionary, etc.)
    pub source: String,

    /// Language
    pub language: Language,
}

/// A single definition
#[derive(Debug, Clone)]
pub struct WordDefinition {
    /// The definition text
    pub text: String,

    /// Part of speech
    pub pos: String,

    /// Example usage
    pub examples: Vec<String>,

    /// Synonyms
    pub synonyms: Vec<String>,
}

/// The word learning system
pub struct WordLearner {
    /// Configuration
    config: LearnerConfig,

    /// Words learned this session
    learned_this_session: RwLock<Vec<String>>,

    /// Failed lookups (cache to avoid repeated failures)
    failed_lookups: RwLock<HashMap<String, u64>>,

    /// Prime inference rules
    inference_rules: HashMap<String, Vec<SemanticPrime>>,
}

impl WordLearner {
    /// Create new word learner
    pub fn new(config: LearnerConfig) -> Self {
        let mut learner = Self {
            config,
            learned_this_session: RwLock::new(Vec::new()),
            failed_lookups: RwLock::new(HashMap::new()),
            inference_rules: HashMap::new(),
        };

        learner.initialize_inference_rules();
        learner
    }

    /// Initialize rules for inferring primes from definitions
    fn initialize_inference_rules(&mut self) {
        // Keywords that suggest certain primes
        self.inference_rules.insert("feel".to_string(), vec![SemanticPrime::Feel]);
        self.inference_rules.insert("feeling".to_string(), vec![SemanticPrime::Feel]);
        self.inference_rules.insert("emotion".to_string(), vec![SemanticPrime::Feel]);
        self.inference_rules.insert("emotional".to_string(), vec![SemanticPrime::Feel]);

        self.inference_rules.insert("think".to_string(), vec![SemanticPrime::Think]);
        self.inference_rules.insert("thought".to_string(), vec![SemanticPrime::Think]);
        self.inference_rules.insert("cognitive".to_string(), vec![SemanticPrime::Think]);
        self.inference_rules.insert("mental".to_string(), vec![SemanticPrime::Think]);

        self.inference_rules.insert("know".to_string(), vec![SemanticPrime::Know]);
        self.inference_rules.insert("knowledge".to_string(), vec![SemanticPrime::Know]);
        self.inference_rules.insert("aware".to_string(), vec![SemanticPrime::Know]);

        self.inference_rules.insert("want".to_string(), vec![SemanticPrime::Want]);
        self.inference_rules.insert("desire".to_string(), vec![SemanticPrime::Want]);
        self.inference_rules.insert("wish".to_string(), vec![SemanticPrime::Want]);

        self.inference_rules.insert("good".to_string(), vec![SemanticPrime::Good]);
        self.inference_rules.insert("positive".to_string(), vec![SemanticPrime::Good]);
        self.inference_rules.insert("excellent".to_string(), vec![SemanticPrime::Good, SemanticPrime::Very]);
        self.inference_rules.insert("great".to_string(), vec![SemanticPrime::Good, SemanticPrime::Big]);

        self.inference_rules.insert("bad".to_string(), vec![SemanticPrime::Bad]);
        self.inference_rules.insert("negative".to_string(), vec![SemanticPrime::Bad]);

        self.inference_rules.insert("do".to_string(), vec![SemanticPrime::Do]);
        self.inference_rules.insert("action".to_string(), vec![SemanticPrime::Do]);
        self.inference_rules.insert("perform".to_string(), vec![SemanticPrime::Do]);

        self.inference_rules.insert("move".to_string(), vec![SemanticPrime::Move]);
        self.inference_rules.insert("motion".to_string(), vec![SemanticPrime::Move]);

        self.inference_rules.insert("say".to_string(), vec![SemanticPrime::Say]);
        self.inference_rules.insert("speak".to_string(), vec![SemanticPrime::Say]);
        self.inference_rules.insert("tell".to_string(), vec![SemanticPrime::Say]);
        self.inference_rules.insert("word".to_string(), vec![SemanticPrime::Words]);

        // Slang-specific patterns
        self.inference_rules.insert("throw".to_string(), vec![SemanticPrime::Move, SemanticPrime::Do]);
        self.inference_rules.insert("cool".to_string(), vec![SemanticPrime::Good]);
        self.inference_rules.insert("awesome".to_string(), vec![SemanticPrime::Good, SemanticPrime::Very]);
        self.inference_rules.insert("lit".to_string(), vec![SemanticPrime::Good, SemanticPrime::Very]);
        self.inference_rules.insert("fire".to_string(), vec![SemanticPrime::Good, SemanticPrime::Very]);
        self.inference_rules.insert("vibe".to_string(), vec![SemanticPrime::Feel, SemanticPrime::Good]);
        self.inference_rules.insert("sus".to_string(), vec![SemanticPrime::Bad, SemanticPrime::Maybe]);
        self.inference_rules.insert("cap".to_string(), vec![SemanticPrime::Not, SemanticPrime::True]);
        self.inference_rules.insert("based".to_string(), vec![SemanticPrime::Good, SemanticPrime::True]);
        self.inference_rules.insert("cringe".to_string(), vec![SemanticPrime::Bad, SemanticPrime::Feel]);
    }

    /// Infer primes from definition text
    pub fn infer_primes_from_definition(&self, definition: &str) -> Vec<SemanticPrime> {
        let mut primes: Vec<SemanticPrime> = Vec::new();
        let lower = definition.to_lowercase();

        for (keyword, keyword_primes) in &self.inference_rules {
            if lower.contains(keyword) {
                for prime in keyword_primes {
                    if !primes.contains(prime) {
                        primes.push(*prime);
                    }
                }
            }
        }

        // Limit to top 5 primes
        primes.truncate(5);
        primes
    }

    /// Learn a word with optional internet verification
    pub fn learn_word(
        &self,
        word: &str,
        context: &[&str],
        language: Language,
    ) -> Result<LearnedWord, LearningError> {
        // Check session limit
        {
            let learned = self.learned_this_session.read().unwrap();
            if learned.len() >= self.config.max_learned_per_session {
                return Err(LearningError::SessionLimitReached);
            }
        }

        // Try internet verification first if enabled
        if self.config.internet_enabled {
            match self.verify_via_internet(word, language) {
                Ok(lookup) => {
                    let learned = self.create_word_from_lookup(word, &lookup);
                    self.record_learned(word);
                    return Ok(learned);
                }
                Err(e) => {
                    // Fall back to context learning
                    eprintln!("Internet lookup failed for '{}': {:?}, using context", word, e);
                }
            }
        }

        // Context-based learning
        self.learn_from_context(word, context, language)
    }

    /// Verify a word via internet (mock implementation)
    fn verify_via_internet(&self, word: &str, language: Language) -> Result<WordLookupResult, LearningError> {
        // In a real implementation, this would make HTTP requests to:
        // - Wiktionary API: https://en.wiktionary.org/api/rest_v1/page/definition/{word}
        // - Urban Dictionary API: https://api.urbandictionary.com/v0/define?term={word}
        //
        // For now, we simulate with known slang terms

        let known_slang: HashMap<&str, (&str, &str, Vec<SemanticPrime>)> = [
            ("yeet", ("verb", "To throw something with force", vec![SemanticPrime::Move, SemanticPrime::Do, SemanticPrime::Very])),
            ("vibe", ("noun", "A feeling or atmosphere", vec![SemanticPrime::Feel, SemanticPrime::Good])),
            ("lit", ("adj", "Exciting, excellent", vec![SemanticPrime::Good, SemanticPrime::Very])),
            ("slay", ("verb", "To do something exceptionally well", vec![SemanticPrime::Do, SemanticPrime::Good, SemanticPrime::Very])),
            ("sus", ("adj", "Suspicious, untrustworthy", vec![SemanticPrime::Bad, SemanticPrime::Maybe])),
            ("cap", ("noun", "A lie or false statement", vec![SemanticPrime::Not, SemanticPrime::True])),
            ("nocap", ("adv", "For real, truthfully", vec![SemanticPrime::True, SemanticPrime::Very])),
            ("based", ("adj", "Authentic, admirable for being true to oneself", vec![SemanticPrime::Good, SemanticPrime::True, SemanticPrime::I])),
            ("cringe", ("adj", "Causing embarrassment or discomfort", vec![SemanticPrime::Bad, SemanticPrime::Feel])),
            ("goat", ("noun", "Greatest of all time", vec![SemanticPrime::Good, SemanticPrime::Very, SemanticPrime::More])),
            ("fire", ("adj", "Excellent, amazing", vec![SemanticPrime::Good, SemanticPrime::Very])),
            ("lowkey", ("adv", "Secretly, to a moderate degree", vec![SemanticPrime::Small, SemanticPrime::Not, SemanticPrime::Say])),
            ("highkey", ("adv", "Obviously, openly", vec![SemanticPrime::Very, SemanticPrime::True])),
            ("bussin", ("adj", "Really good, especially food", vec![SemanticPrime::Good, SemanticPrime::Very])),
            ("fam", ("noun", "Close friends, family", vec![SemanticPrime::People, SemanticPrime::Near, SemanticPrime::Good])),
            ("bet", ("interj", "Agreement, okay", vec![SemanticPrime::True, SemanticPrime::Good])),
            ("snatched", ("adj", "Looking good, on point", vec![SemanticPrime::Good, SemanticPrime::See])),
            ("tea", ("noun", "Gossip, interesting information", vec![SemanticPrime::Say, SemanticPrime::Words, SemanticPrime::True])),
            ("bruh", ("interj", "Expression of disbelief", vec![SemanticPrime::Not, SemanticPrime::Good, SemanticPrime::Feel])),
            ("periodt", ("interj", "End of discussion, emphasis", vec![SemanticPrime::True, SemanticPrime::Very])),
        ].into_iter().collect();

        let normalized = word.to_lowercase();

        if let Some((pos, definition, _primes)) = known_slang.get(normalized.as_str()) {
            Ok(WordLookupResult {
                word: word.to_string(),
                is_valid: true,
                definitions: vec![WordDefinition {
                    text: definition.to_string(),
                    pos: pos.to_string(),
                    examples: vec![],
                    synonyms: vec![],
                }],
                pos: Some(pos.to_string()),
                is_slang: true,
                source: "urban_dictionary".to_string(),
                language,
            })
        } else {
            Err(LearningError::NotFound)
        }
    }

    /// Create learned word from lookup result
    fn create_word_from_lookup(&self, word: &str, lookup: &WordLookupResult) -> LearnedWord {
        // Infer primes from definitions
        let mut primes: Vec<SemanticPrime> = Vec::new();
        for def in &lookup.definitions {
            primes.extend(self.infer_primes_from_definition(&def.text));
        }

        // Remove duplicates
        primes.sort();
        primes.dedup();

        if primes.is_empty() {
            primes.push(SemanticPrime::Something); // Fallback
        }

        // Compute encoding from primes
        let encoding = compute_encoding(&primes);

        let grounding = SemanticGrounding {
            core_primes: primes.clone(),
            modifier_primes: vec![],
            composition: "bind".to_string(),
            explanation: lookup.definitions.first()
                .map(|d| d.text.clone())
                .unwrap_or_else(|| "Unknown".to_string()),
        };

        let learned_word = MultilingualWord {
            word: word.to_string(),
            language: lookup.language,
            encoding,
            grounding,
            pos: lookup.pos.clone().unwrap_or_else(|| "unknown".to_string()),
            learned: true,
            confidence: 0.9, // High confidence from internet verification
            source: lookup.source.clone(),
            translations: HashMap::new(),
            examples: lookup.definitions.first()
                .map(|d| d.examples.clone())
                .unwrap_or_default(),
            is_slang: lookup.is_slang,
            last_verified: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };

        LearnedWord {
            word: learned_word,
            method: LearningMethod::Internet,
            verification_status: VerificationStatus::Verified,
        }
    }

    /// Learn from context only
    fn learn_from_context(
        &self,
        word: &str,
        context: &[&str],
        language: Language,
    ) -> Result<LearnedWord, LearningError> {
        // Infer primes from context words
        let mut context_primes: Vec<SemanticPrime> = Vec::new();

        for ctx_word in context {
            // Check if it matches any inference rule
            for (keyword, primes) in &self.inference_rules {
                if ctx_word.to_lowercase().contains(keyword) {
                    context_primes.extend(primes.iter());
                }
            }
        }

        if context_primes.is_empty() {
            context_primes.push(SemanticPrime::Something); // Fallback
        }

        // Remove duplicates and limit
        context_primes.sort();
        context_primes.dedup();
        context_primes.truncate(5);

        let encoding = compute_encoding(&context_primes);

        let grounding = SemanticGrounding {
            core_primes: context_primes.clone(),
            modifier_primes: vec![],
            composition: "bind".to_string(),
            explanation: format!("Inferred from context: {}", context.join(", ")),
        };

        let learned_word = MultilingualWord {
            word: word.to_string(),
            language,
            encoding,
            grounding,
            pos: "unknown".to_string(),
            learned: true,
            confidence: 0.4, // Lower confidence for context-only
            source: "context".to_string(),
            translations: HashMap::new(),
            examples: vec![context.join(" ")],
            is_slang: false, // Unknown without verification
            last_verified: 0,
        };

        self.record_learned(word);

        Ok(LearnedWord {
            word: learned_word,
            method: LearningMethod::Context,
            verification_status: VerificationStatus::Probable,
        })
    }

    /// Record that a word was learned
    fn record_learned(&self, word: &str) {
        let mut learned = self.learned_this_session.write().unwrap();
        learned.push(word.to_string());
    }

    /// Get words learned this session
    pub fn learned_words(&self) -> Vec<String> {
        self.learned_this_session.read().unwrap().clone()
    }

    /// Get configuration
    pub fn config(&self) -> &LearnerConfig {
        &self.config
    }

    /// Enable internet verification
    pub fn enable_internet(&mut self) {
        self.config.internet_enabled = true;
    }

    /// Disable internet verification
    pub fn disable_internet(&mut self) {
        self.config.internet_enabled = false;
    }
}

impl Default for WordLearner {
    fn default() -> Self {
        Self::new(LearnerConfig::default())
    }
}

/// Errors during word learning
#[derive(Debug, Clone)]
pub enum LearningError {
    /// Word not found in any source
    NotFound,
    /// Network error during lookup
    NetworkError(String),
    /// Session limit reached
    SessionLimitReached,
    /// Not enough context to learn
    InsufficientContext,
}

/// Compute HV16 encoding from primes (utility function)
fn compute_encoding(primes: &[SemanticPrime]) -> HV16 {
    if primes.is_empty() {
        return HV16::zero();
    }

    let mut result = HV16::random((primes[0] as u64) * 1000 + 500);
    for (i, prime) in primes.iter().enumerate().skip(1) {
        let v = HV16::random((*prime as u64) * 1000 + 500).permute(i);
        result = result.bind(&v);
    }
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learner_creation() {
        let learner = WordLearner::default();
        assert!(!learner.config.internet_enabled);
        assert!(learner.config.auto_learn);
    }

    #[test]
    fn test_learn_from_context() {
        let learner = WordLearner::default();

        let result = learner.learn_word(
            "shmoove",
            &["cool", "move", "smooth"],
            Language::English,
        );

        assert!(result.is_ok());
        let learned = result.unwrap();
        assert!(learned.word.learned);
        assert_eq!(learned.method, LearningMethod::Context);
    }

    #[test]
    fn test_infer_primes_from_definition() {
        let learner = WordLearner::default();

        let primes = learner.infer_primes_from_definition(
            "A feeling of great happiness and excitement"
        );

        assert!(primes.contains(&SemanticPrime::Feel));
        assert!(primes.contains(&SemanticPrime::Good));
    }

    #[test]
    fn test_known_slang_verification() {
        let mut learner = WordLearner::default();
        learner.enable_internet();

        let result = learner.learn_word("yeet", &[], Language::English);

        assert!(result.is_ok());
        let learned = result.unwrap();
        assert!(learned.word.is_slang);
        assert_eq!(learned.verification_status, VerificationStatus::Verified);
    }

    #[test]
    fn test_session_limit() {
        let config = LearnerConfig {
            max_learned_per_session: 2,
            ..Default::default()
        };
        let learner = WordLearner::new(config);

        // Learn 2 words (should succeed)
        assert!(learner.learn_word("foo", &["good"], Language::English).is_ok());
        assert!(learner.learn_word("bar", &["bad"], Language::English).is_ok());

        // Third should fail
        let result = learner.learn_word("baz", &["other"], Language::English);
        assert!(matches!(result, Err(LearningError::SessionLimitReached)));
    }

    #[test]
    fn test_internet_toggle() {
        let mut learner = WordLearner::default();

        assert!(!learner.config().internet_enabled);
        learner.enable_internet();
        assert!(learner.config().internet_enabled);
        learner.disable_internet();
        assert!(!learner.config().internet_enabled);
    }

    #[test]
    fn test_multiple_slang_words() {
        let mut learner = WordLearner::default();
        learner.enable_internet();

        let slang_words = ["lit", "vibe", "sus", "based", "goat"];

        for word in &slang_words {
            let result = learner.learn_word(word, &[], Language::English);
            assert!(result.is_ok(), "Failed to learn slang: {}", word);
            assert!(result.unwrap().word.is_slang);
        }
    }

    #[test]
    fn test_learned_words_tracking() {
        let learner = WordLearner::default();

        assert!(learner.learned_words().is_empty());

        let _ = learner.learn_word("foo", &["good"], Language::English);
        let _ = learner.learn_word("bar", &["bad"], Language::English);

        let learned = learner.learned_words();
        assert_eq!(learned.len(), 2);
        assert!(learned.contains(&"foo".to_string()));
        assert!(learned.contains(&"bar".to_string()));
    }
}
