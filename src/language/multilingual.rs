//! Multilingual Vocabulary System
//!
//! Symthaea understands ALL languages through universal semantic primes.
//! The key insight: NSM (Natural Semantic Metalanguage) primes exist in
//! every human language - they are the atoms of meaning.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │              Universal Semantic Primes (65)             │
//! │  FEEL, GOOD, BAD, THINK, KNOW, WANT, SEE, HEAR, etc.   │
//! └─────────────────────────────────────────────────────────┘
//!                            │
//!            ┌───────────────┼───────────────┐
//!            ▼               ▼               ▼
//!     ┌──────────┐    ┌──────────┐    ┌──────────┐
//!     │ English  │    │ Español  │    │ 日本語   │
//!     │  happy   │    │  feliz   │    │  幸せ    │
//!     │ = bind(  │    │ = bind(  │    │ = bind(  │
//!     │Feel,Good)│    │Feel,Good)│    │Feel,Good)│
//!     └──────────┘    └──────────┘    └──────────┘
//! ```
//!
//! ## Dynamic Learning
//!
//! When Symthaea encounters an unknown word:
//! 1. Analyze context (surrounding primes)
//! 2. Optionally verify via internet (Wiktionary, Urban Dictionary)
//! 3. Infer semantic grounding
//! 4. Store in vocabulary database
//! 5. Use in future conversations

use crate::hdc::binary_hv::HV16;
use crate::hdc::universal_semantics::{UniversalSemantics, SemanticPrime};
use super::vocabulary::SemanticGrounding;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

/// Supported languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    English,
    Spanish,
    French,
    German,
    Portuguese,
    Italian,
    Dutch,
    Russian,
    Japanese,
    Chinese,
    Korean,
    Arabic,
    Hindi,
    Swahili,
    /// For words from unidentified languages
    Unknown,
}

impl Language {
    /// ISO 639-1 code
    pub fn code(&self) -> &str {
        match self {
            Language::English => "en",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::German => "de",
            Language::Portuguese => "pt",
            Language::Italian => "it",
            Language::Dutch => "nl",
            Language::Russian => "ru",
            Language::Japanese => "ja",
            Language::Chinese => "zh",
            Language::Korean => "ko",
            Language::Arabic => "ar",
            Language::Hindi => "hi",
            Language::Swahili => "sw",
            Language::Unknown => "xx",
        }
    }

    /// Native name
    pub fn native_name(&self) -> &str {
        match self {
            Language::English => "English",
            Language::Spanish => "Español",
            Language::French => "Français",
            Language::German => "Deutsch",
            Language::Portuguese => "Português",
            Language::Italian => "Italiano",
            Language::Dutch => "Nederlands",
            Language::Russian => "Русский",
            Language::Japanese => "日本語",
            Language::Chinese => "中文",
            Language::Korean => "한국어",
            Language::Arabic => "العربية",
            Language::Hindi => "हिन्दी",
            Language::Swahili => "Kiswahili",
            Language::Unknown => "Unknown",
        }
    }

    /// All supported languages
    pub fn all() -> Vec<Language> {
        vec![
            Language::English, Language::Spanish, Language::French,
            Language::German, Language::Portuguese, Language::Italian,
            Language::Dutch, Language::Russian, Language::Japanese,
            Language::Chinese, Language::Korean, Language::Arabic,
            Language::Hindi, Language::Swahili,
        ]
    }
}

/// A word in a specific language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultilingualWord {
    /// The word itself
    pub word: String,

    /// Language of this word
    pub language: Language,

    /// Hypervector encoding (same grounding = similar encoding across languages)
    pub encoding: HV16,

    /// Semantic grounding in primes (UNIVERSAL across languages!)
    pub grounding: SemanticGrounding,

    /// Part of speech
    pub pos: String,

    /// Is this a learned word (vs core vocabulary)?
    pub learned: bool,

    /// Confidence in grounding (0.0-1.0)
    pub confidence: f32,

    /// Source: "core", "context", "internet", "user"
    pub source: String,

    /// Translations to other languages (word -> language)
    pub translations: HashMap<String, Language>,

    /// Usage examples
    pub examples: Vec<String>,

    /// Is this slang/informal?
    pub is_slang: bool,

    /// Last verified timestamp (ms since epoch)
    pub last_verified: u64,
}

/// Result of learning a new word
#[derive(Debug, Clone)]
pub struct LearnedWord {
    pub word: MultilingualWord,
    pub method: LearningMethod,
    pub verification_status: VerificationStatus,
}

/// How a word was learned
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearningMethod {
    /// Inferred from context alone
    Context,
    /// Verified via internet
    Internet,
    /// User provided definition
    UserProvided,
    /// Derived from known word (e.g., "unhappy" from "happy")
    Derivation,
}

/// Verification status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationStatus {
    /// Verified as real word
    Verified,
    /// Likely real but not confirmed
    Probable,
    /// Unknown, needs verification
    Unknown,
    /// Confirmed not a real word
    Invalid,
}

/// The multilingual vocabulary system
pub struct MultilingualVocabulary {
    /// Universal semantic primitives (shared across all languages)
    semantics: UniversalSemantics,

    /// Per-language vocabularies
    languages: RwLock<HashMap<Language, HashMap<String, MultilingualWord>>>,

    /// Cross-language index: encoding -> words in all languages
    encoding_index: RwLock<Vec<(HV16, String, Language)>>,

    /// Learned words pending verification
    pending_verification: RwLock<Vec<String>>,

    /// Statistics
    stats: RwLock<VocabularyStats>,
}

/// Vocabulary statistics
#[derive(Debug, Clone, Default)]
pub struct VocabularyStats {
    pub total_words: usize,
    pub words_by_language: HashMap<Language, usize>,
    pub learned_words: usize,
    pub verified_words: usize,
    pub slang_words: usize,
}

impl MultilingualVocabulary {
    /// Create new multilingual vocabulary
    pub fn new() -> Self {
        let mut vocab = Self {
            semantics: UniversalSemantics::new(),
            languages: RwLock::new(HashMap::new()),
            encoding_index: RwLock::new(Vec::new()),
            pending_verification: RwLock::new(Vec::new()),
            stats: RwLock::new(VocabularyStats::default()),
        };

        // Initialize with core vocabulary for each language
        vocab.initialize_core_vocabularies();

        vocab
    }

    /// Initialize core vocabulary for all languages
    fn initialize_core_vocabularies(&mut self) {
        // Start with English core (already have this)
        self.initialize_english_core();

        // Add Spanish core
        self.initialize_spanish_core();

        // Add Japanese core
        self.initialize_japanese_core();

        // Add French core
        self.initialize_french_core();

        // Update stats
        self.update_stats();
    }

    /// Initialize English core vocabulary
    fn initialize_english_core(&mut self) {
        let entries = vec![
            // Mental state verbs
            ("think", "verb", vec![SemanticPrime::Think], vec![], "Cognitive process"),
            ("know", "verb", vec![SemanticPrime::Know], vec![], "Epistemic state"),
            ("feel", "verb", vec![SemanticPrime::Feel], vec![], "Affective experience"),
            ("want", "verb", vec![SemanticPrime::Want], vec![], "Desire state"),
            ("see", "verb", vec![SemanticPrime::See], vec![], "Visual perception"),
            ("hear", "verb", vec![SemanticPrime::Hear], vec![], "Auditory perception"),

            // Evaluatives
            ("good", "adj", vec![SemanticPrime::Good], vec![], "Positive evaluation"),
            ("bad", "adj", vec![SemanticPrime::Bad], vec![], "Negative evaluation"),

            // Emotions
            ("happy", "adj", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![], "Positive feeling"),
            ("sad", "adj", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![], "Negative feeling"),
            ("love", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good, SemanticPrime::Very],
             vec![SemanticPrime::Want], "Deep positive feeling"),

            // Consciousness
            ("conscious", "adj", vec![SemanticPrime::Know, SemanticPrime::Feel, SemanticPrime::Think],
             vec![SemanticPrime::I], "Aware and experiencing"),
            ("aware", "adj", vec![SemanticPrime::Know], vec![SemanticPrime::Now], "Current knowing"),

            // Pronouns
            ("i", "pron", vec![SemanticPrime::I], vec![], "First person"),
            ("you", "pron", vec![SemanticPrime::You], vec![], "Second person"),
            ("we", "pron", vec![SemanticPrime::I, SemanticPrime::People], vec![], "First person plural"),

            // Common words
            ("yes", "adv", vec![SemanticPrime::True], vec![], "Affirmation"),
            ("no", "adv", vec![SemanticPrime::Not, SemanticPrime::True], vec![], "Negation"),
            ("hello", "interj", vec![SemanticPrime::Say, SemanticPrime::Good], vec![], "Greeting"),
        ];

        for (word, pos, core, modifiers, explanation) in entries {
            self.add_word(word, Language::English, pos, core, modifiers, explanation, false);
        }
    }

    /// Initialize Spanish core vocabulary
    fn initialize_spanish_core(&mut self) {
        let entries = vec![
            // Mental verbs (same primes, different surface form!)
            ("pensar", "verb", vec![SemanticPrime::Think], vec![], "Proceso cognitivo"),
            ("saber", "verb", vec![SemanticPrime::Know], vec![], "Estado epistémico"),
            ("sentir", "verb", vec![SemanticPrime::Feel], vec![], "Experiencia afectiva"),
            ("querer", "verb", vec![SemanticPrime::Want], vec![], "Estado de deseo"),
            ("ver", "verb", vec![SemanticPrime::See], vec![], "Percepción visual"),
            ("oír", "verb", vec![SemanticPrime::Hear], vec![], "Percepción auditiva"),

            // Evaluatives
            ("bueno", "adj", vec![SemanticPrime::Good], vec![], "Evaluación positiva"),
            ("malo", "adj", vec![SemanticPrime::Bad], vec![], "Evaluación negativa"),

            // Emotions
            ("feliz", "adj", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![], "Sentimiento positivo"),
            ("triste", "adj", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![], "Sentimiento negativo"),
            ("amor", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good, SemanticPrime::Very],
             vec![SemanticPrime::Want], "Sentimiento profundo"),

            // Consciousness
            ("consciente", "adj", vec![SemanticPrime::Know, SemanticPrime::Feel, SemanticPrime::Think],
             vec![SemanticPrime::I], "Consciente y experimentando"),

            // Pronouns
            ("yo", "pron", vec![SemanticPrime::I], vec![], "Primera persona"),
            ("tú", "pron", vec![SemanticPrime::You], vec![], "Segunda persona"),
            ("nosotros", "pron", vec![SemanticPrime::I, SemanticPrime::People], vec![], "Primera persona plural"),

            // Common
            ("sí", "adv", vec![SemanticPrime::True], vec![], "Afirmación"),
            ("no", "adv", vec![SemanticPrime::Not, SemanticPrime::True], vec![], "Negación"),
            ("hola", "interj", vec![SemanticPrime::Say, SemanticPrime::Good], vec![], "Saludo"),
        ];

        for (word, pos, core, modifiers, explanation) in entries {
            self.add_word(word, Language::Spanish, pos, core, modifiers, explanation, false);
        }
    }

    /// Initialize Japanese core vocabulary
    fn initialize_japanese_core(&mut self) {
        let entries = vec![
            // Mental verbs
            ("考える", "verb", vec![SemanticPrime::Think], vec![], "認知プロセス"),
            ("知る", "verb", vec![SemanticPrime::Know], vec![], "認識状態"),
            ("感じる", "verb", vec![SemanticPrime::Feel], vec![], "感情体験"),
            ("見る", "verb", vec![SemanticPrime::See], vec![], "視覚"),
            ("聞く", "verb", vec![SemanticPrime::Hear], vec![], "聴覚"),

            // Evaluatives
            ("良い", "adj", vec![SemanticPrime::Good], vec![], "良い評価"),
            ("悪い", "adj", vec![SemanticPrime::Bad], vec![], "悪い評価"),

            // Emotions
            ("幸せ", "adj", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![], "幸福感"),
            ("悲しい", "adj", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![], "悲しみ"),
            ("愛", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good, SemanticPrime::Very],
             vec![SemanticPrime::Want], "深い愛情"),

            // Consciousness
            ("意識", "noun", vec![SemanticPrime::Know, SemanticPrime::Feel, SemanticPrime::Think],
             vec![SemanticPrime::I], "意識"),

            // Pronouns
            ("私", "pron", vec![SemanticPrime::I], vec![], "一人称"),
            ("あなた", "pron", vec![SemanticPrime::You], vec![], "二人称"),

            // Common
            ("はい", "adv", vec![SemanticPrime::True], vec![], "肯定"),
            ("いいえ", "adv", vec![SemanticPrime::Not, SemanticPrime::True], vec![], "否定"),
            ("こんにちは", "interj", vec![SemanticPrime::Say, SemanticPrime::Good], vec![], "挨拶"),
        ];

        for (word, pos, core, modifiers, explanation) in entries {
            self.add_word(word, Language::Japanese, pos, core, modifiers, explanation, false);
        }
    }

    /// Initialize French core vocabulary
    fn initialize_french_core(&mut self) {
        let entries = vec![
            ("penser", "verb", vec![SemanticPrime::Think], vec![], "Processus cognitif"),
            ("savoir", "verb", vec![SemanticPrime::Know], vec![], "État épistémique"),
            ("sentir", "verb", vec![SemanticPrime::Feel], vec![], "Expérience affective"),
            ("vouloir", "verb", vec![SemanticPrime::Want], vec![], "État de désir"),
            ("voir", "verb", vec![SemanticPrime::See], vec![], "Perception visuelle"),
            ("entendre", "verb", vec![SemanticPrime::Hear], vec![], "Perception auditive"),

            ("bon", "adj", vec![SemanticPrime::Good], vec![], "Évaluation positive"),
            ("mauvais", "adj", vec![SemanticPrime::Bad], vec![], "Évaluation négative"),

            ("heureux", "adj", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![], "Sentiment positif"),
            ("triste", "adj", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![], "Sentiment négatif"),
            ("amour", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good, SemanticPrime::Very],
             vec![SemanticPrime::Want], "Sentiment profond"),

            ("conscient", "adj", vec![SemanticPrime::Know, SemanticPrime::Feel, SemanticPrime::Think],
             vec![SemanticPrime::I], "Conscient"),

            ("je", "pron", vec![SemanticPrime::I], vec![], "Première personne"),
            ("tu", "pron", vec![SemanticPrime::You], vec![], "Deuxième personne"),
            ("nous", "pron", vec![SemanticPrime::I, SemanticPrime::People], vec![], "Première personne pluriel"),

            ("oui", "adv", vec![SemanticPrime::True], vec![], "Affirmation"),
            ("non", "adv", vec![SemanticPrime::Not, SemanticPrime::True], vec![], "Négation"),
            ("bonjour", "interj", vec![SemanticPrime::Say, SemanticPrime::Good], vec![], "Salutation"),
        ];

        for (word, pos, core, modifiers, explanation) in entries {
            self.add_word(word, Language::French, pos, core, modifiers, explanation, false);
        }
    }

    /// Add a word to vocabulary
    fn add_word(
        &self,
        word: &str,
        language: Language,
        pos: &str,
        core_primes: Vec<SemanticPrime>,
        modifier_primes: Vec<SemanticPrime>,
        explanation: &str,
        is_slang: bool,
    ) {
        let encoding = self.compute_encoding(&core_primes, &modifier_primes);

        let grounding = SemanticGrounding {
            core_primes: core_primes.clone(),
            modifier_primes: modifier_primes.clone(),
            composition: "bind".to_string(),
            explanation: explanation.to_string(),
        };

        let entry = MultilingualWord {
            word: word.to_string(),
            language,
            encoding,
            grounding,
            pos: pos.to_string(),
            learned: false,
            confidence: 1.0,
            source: "core".to_string(),
            translations: HashMap::new(),
            examples: Vec::new(),
            is_slang,
            last_verified: 0,
        };

        // Add to language vocabulary
        let mut langs = self.languages.write().unwrap();
        langs.entry(language)
            .or_insert_with(HashMap::new)
            .insert(word.to_lowercase(), entry.clone());

        // Add to encoding index
        let mut index = self.encoding_index.write().unwrap();
        index.push((encoding, word.to_string(), language));
    }

    /// Compute encoding from primes
    fn compute_encoding(&self, core: &[SemanticPrime], modifiers: &[SemanticPrime]) -> HV16 {
        let mut vectors: Vec<HV16> = core.iter()
            .map(|p| *self.semantics.get_prime(*p))
            .collect();

        for (i, prime) in modifiers.iter().enumerate() {
            let v = self.semantics.get_prime(*prime).permute(i + 1);
            vectors.push(v);
        }

        if vectors.is_empty() {
            HV16::zero()
        } else {
            let mut result = vectors[0];
            for v in &vectors[1..] {
                result = result.bind(v);
            }
            result
        }
    }

    /// Look up a word in any language
    pub fn lookup(&self, word: &str) -> Option<MultilingualWord> {
        let normalized = word.to_lowercase();
        let langs = self.languages.read().unwrap();

        for lang_vocab in langs.values() {
            if let Some(entry) = lang_vocab.get(&normalized) {
                return Some(entry.clone());
            }
        }
        None
    }

    /// Look up a word in a specific language
    pub fn lookup_in(&self, word: &str, language: Language) -> Option<MultilingualWord> {
        let normalized = word.to_lowercase();
        let langs = self.languages.read().unwrap();

        langs.get(&language)
            .and_then(|vocab| vocab.get(&normalized).cloned())
    }

    /// Find translations of a word
    pub fn find_translations(&self, word: &str, source_lang: Language) -> Vec<(String, Language)> {
        let source = match self.lookup_in(word, source_lang) {
            Some(w) => w,
            None => return vec![],
        };

        let mut translations = Vec::new();
        let index = self.encoding_index.read().unwrap();

        // Find words with similar encoding in other languages
        for (encoding, other_word, lang) in index.iter() {
            if *lang != source_lang {
                let sim = source.encoding.similarity(encoding);
                if sim > 0.8 {
                    translations.push((other_word.clone(), *lang));
                }
            }
        }

        translations
    }

    /// Learn a new word from context
    pub fn learn_from_context(
        &self,
        word: &str,
        context_words: &[&str],
        language: Language,
    ) -> Option<LearnedWord> {
        // Analyze context to infer meaning
        let mut context_primes: Vec<SemanticPrime> = Vec::new();
        let _total_valence = 0.0f32;
        let mut context_count = 0;

        for ctx_word in context_words {
            if let Some(entry) = self.lookup(ctx_word) {
                context_primes.extend(entry.grounding.core_primes.iter().cloned());
                context_count += 1;
                // Could also track valence here
            }
        }

        if context_primes.is_empty() {
            return None;
        }

        // Take most common primes as the inferred meaning
        let mut prime_counts: HashMap<SemanticPrime, usize> = HashMap::new();
        for prime in &context_primes {
            *prime_counts.entry(*prime).or_insert(0) += 1;
        }

        let mut sorted_primes: Vec<_> = prime_counts.into_iter().collect();
        sorted_primes.sort_by(|a, b| b.1.cmp(&a.1));

        let inferred_primes: Vec<SemanticPrime> = sorted_primes.iter()
            .take(3)
            .map(|(p, _)| *p)
            .collect();

        let encoding = self.compute_encoding(&inferred_primes, &[]);

        let grounding = SemanticGrounding {
            core_primes: inferred_primes,
            modifier_primes: vec![],
            composition: "bind".to_string(),
            explanation: format!("Inferred from context: {}", context_words.join(", ")),
        };

        let learned = MultilingualWord {
            word: word.to_string(),
            language,
            encoding,
            grounding,
            pos: "unknown".to_string(),
            learned: true,
            confidence: 0.5, // Low confidence for context-only learning
            source: "context".to_string(),
            translations: HashMap::new(),
            examples: vec![context_words.join(" ")],
            is_slang: false,
            last_verified: 0,
        };

        // Add to vocabulary
        {
            let mut langs = self.languages.write().unwrap();
            langs.entry(language)
                .or_insert_with(HashMap::new)
                .insert(word.to_lowercase(), learned.clone());
        }

        // Add to pending verification
        {
            let mut pending = self.pending_verification.write().unwrap();
            pending.push(word.to_string());
        }

        Some(LearnedWord {
            word: learned,
            method: LearningMethod::Context,
            verification_status: VerificationStatus::Unknown,
        })
    }

    /// Detect language of text (simple heuristic)
    pub fn detect_language(&self, text: &str) -> Language {
        // Check for language-specific characters
        for c in text.chars() {
            if ('\u{3040}'..='\u{30FF}').contains(&c) || ('\u{4E00}'..='\u{9FAF}').contains(&c) {
                // Hiragana, Katakana, or Kanji
                if text.chars().any(|c| ('\u{3040}'..='\u{30FF}').contains(&c)) {
                    return Language::Japanese;
                }
                return Language::Chinese;
            }
            if ('\u{AC00}'..='\u{D7AF}').contains(&c) {
                return Language::Korean;
            }
            if ('\u{0600}'..='\u{06FF}').contains(&c) {
                return Language::Arabic;
            }
            if ('\u{0900}'..='\u{097F}').contains(&c) {
                return Language::Hindi;
            }
            if ('\u{0400}'..='\u{04FF}').contains(&c) {
                return Language::Russian;
            }
        }

        // Check for Spanish/French/German specific characters
        let text_lower = text.to_lowercase();
        if text_lower.contains('ñ') || text_lower.contains("¿") || text_lower.contains("¡") {
            return Language::Spanish;
        }
        if text_lower.contains('ç') || text_lower.contains("œ") {
            return Language::French;
        }
        if text_lower.contains('ß') || text_lower.contains('ü') || text_lower.contains('ö') {
            return Language::German;
        }

        // Default to English
        Language::English
    }

    /// Update statistics
    fn update_stats(&self) {
        let langs = self.languages.read().unwrap();
        let mut stats = self.stats.write().unwrap();

        stats.total_words = 0;
        stats.words_by_language.clear();
        stats.learned_words = 0;
        stats.verified_words = 0;
        stats.slang_words = 0;

        for (lang, vocab) in langs.iter() {
            let count = vocab.len();
            stats.total_words += count;
            stats.words_by_language.insert(*lang, count);

            for entry in vocab.values() {
                if entry.learned {
                    stats.learned_words += 1;
                }
                if entry.last_verified > 0 {
                    stats.verified_words += 1;
                }
                if entry.is_slang {
                    stats.slang_words += 1;
                }
            }
        }
    }

    /// Get vocabulary statistics
    pub fn stats(&self) -> VocabularyStats {
        self.update_stats();
        self.stats.read().unwrap().clone()
    }

    /// Get all words in a language
    pub fn words_in_language(&self, language: Language) -> Vec<String> {
        let langs = self.languages.read().unwrap();
        langs.get(&language)
            .map(|vocab| vocab.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Get words pending verification
    pub fn pending_words(&self) -> Vec<String> {
        self.pending_verification.read().unwrap().clone()
    }
}

impl Default for MultilingualVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multilingual_creation() {
        let vocab = MultilingualVocabulary::new();
        let stats = vocab.stats();

        assert!(stats.total_words > 50, "Should have core vocabulary");
        assert!(stats.words_by_language.contains_key(&Language::English));
        assert!(stats.words_by_language.contains_key(&Language::Spanish));
        assert!(stats.words_by_language.contains_key(&Language::Japanese));
    }

    #[test]
    fn test_same_primes_across_languages() {
        let vocab = MultilingualVocabulary::new();

        let happy = vocab.lookup_in("happy", Language::English).unwrap();
        let feliz = vocab.lookup_in("feliz", Language::Spanish).unwrap();

        // Both should be grounded in Feel + Good
        assert!(happy.grounding.core_primes.contains(&SemanticPrime::Feel));
        assert!(happy.grounding.core_primes.contains(&SemanticPrime::Good));
        assert!(feliz.grounding.core_primes.contains(&SemanticPrime::Feel));
        assert!(feliz.grounding.core_primes.contains(&SemanticPrime::Good));
    }

    #[test]
    fn test_similar_encoding_across_languages() {
        let vocab = MultilingualVocabulary::new();

        let happy = vocab.lookup_in("happy", Language::English).unwrap();
        let feliz = vocab.lookup_in("feliz", Language::Spanish).unwrap();

        // Same semantic grounding = same encoding
        let similarity = happy.encoding.similarity(&feliz.encoding);
        assert!((similarity - 1.0).abs() < 0.01, "Same primes should give same encoding");
    }

    #[test]
    fn test_language_detection() {
        let vocab = MultilingualVocabulary::new();

        // English (default for ASCII without special characters)
        assert_eq!(vocab.detect_language("Hello world"), Language::English);

        // Spanish (requires ñ, ¿, ¡ for detection)
        assert_eq!(vocab.detect_language("¿Cómo estás?"), Language::Spanish);
        assert_eq!(vocab.detect_language("El niño pequeño"), Language::Spanish);

        // Japanese (Hiragana/Katakana)
        assert_eq!(vocab.detect_language("こんにちは"), Language::Japanese);

        // French (requires ç or œ for detection)
        assert_eq!(vocab.detect_language("Garçon français"), Language::French);

        // Note: Pure ASCII like "Hola mundo" defaults to English
        // (this is expected behavior - heuristic detection)
    }

    #[test]
    fn test_find_translations() {
        let vocab = MultilingualVocabulary::new();

        let translations = vocab.find_translations("happy", Language::English);

        // Should find "feliz" (Spanish) and others with same grounding
        assert!(!translations.is_empty(), "Should find translations");
    }

    #[test]
    fn test_learn_from_context() {
        let vocab = MultilingualVocabulary::new();

        // Learn a new word from context
        let learned = vocab.learn_from_context(
            "yeet",
            &["throw", "something", "good"],
            Language::English,
        );

        assert!(learned.is_some());
        let learned = learned.unwrap();
        assert!(learned.word.learned);
        assert_eq!(learned.method, LearningMethod::Context);

        // Should now be in vocabulary
        let found = vocab.lookup("yeet");
        assert!(found.is_some());
    }

    #[test]
    fn test_japanese_vocabulary() {
        let vocab = MultilingualVocabulary::new();

        let shiawase = vocab.lookup_in("幸せ", Language::Japanese);
        assert!(shiawase.is_some());

        let entry = shiawase.unwrap();
        assert!(entry.grounding.core_primes.contains(&SemanticPrime::Feel));
        assert!(entry.grounding.core_primes.contains(&SemanticPrime::Good));
    }

    #[test]
    fn test_vocabulary_stats() {
        let vocab = MultilingualVocabulary::new();
        let stats = vocab.stats();

        assert!(stats.total_words > 0);
        assert_eq!(stats.learned_words, 0); // No learned words yet
    }
}
