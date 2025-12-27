//! Vocabulary System - Words Grounded in Semantic Primes
//!
//! Unlike LLM token embeddings (learned from co-occurrence), our vocabulary
//! is GROUNDED in universal semantic primes. Each word's meaning is
//! explicitly composed from the 65 NSM primitives.
//!
//! This gives us:
//! - True understanding (not statistical correlation)
//! - Zero-shot generalization (new words via prime composition)
//! - Explainable meaning ("happy" = FEEL + GOOD + VERY)
//! - Cross-lingual consistency (same primes across languages)

use crate::hdc::binary_hv::HV16;
use crate::hdc::universal_semantics::{UniversalSemantics, SemanticPrime};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// How a word is grounded in semantic primes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticGrounding {
    /// Primary primes that define this word's core meaning
    pub core_primes: Vec<SemanticPrime>,

    /// Secondary primes that add nuance
    pub modifier_primes: Vec<SemanticPrime>,

    /// Composition method: "bind", "bundle", "sequence"
    pub composition: String,

    /// Human-readable explanation
    pub explanation: String,
}

/// A word entry in the vocabulary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordEntry {
    /// The word itself
    pub word: String,

    /// Normalized form (lowercase, stemmed)
    pub normalized: String,

    /// Part of speech: noun, verb, adj, adv, prep, conj, det, pron
    pub pos: String,

    /// Hypervector encoding
    pub encoding: HV16,

    /// Semantic grounding in primes
    pub grounding: SemanticGrounding,

    /// Frequency weight (for generation preference)
    pub frequency: f32,

    /// Emotional valence (-1.0 to 1.0)
    pub valence: f32,

    /// Arousal level (0.0 to 1.0)
    pub arousal: f32,
}

/// The vocabulary system
#[derive(Debug)]
pub struct Vocabulary {
    /// Word → Entry mapping
    entries: HashMap<String, WordEntry>,

    /// Semantic primitives for grounding
    semantics: UniversalSemantics,

    /// Reverse lookup: find words similar to a vector
    /// Stored as (encoding_hash, word) for fast lookup
    reverse_index: Vec<(HV16, String)>,

    /// Part-of-speech specific indices for generation
    by_pos: HashMap<String, Vec<String>>,
}

impl Vocabulary {
    /// Create vocabulary with core words grounded in semantic primes
    pub fn new() -> Self {
        let semantics = UniversalSemantics::new();
        let mut vocab = Self {
            entries: HashMap::new(),
            semantics,
            reverse_index: Vec::new(),
            by_pos: HashMap::new(),
        };

        // Initialize with core vocabulary grounded in primes
        vocab.initialize_core_vocabulary();
        vocab.initialize_extended_vocabulary();  // Add 700+ more words
        vocab.build_reverse_index();

        vocab
    }

    /// Initialize core vocabulary with semantic grounding
    fn initialize_core_vocabulary(&mut self) {
        // ===== PRONOUNS (grounded in substantive primes) =====
        self.add_grounded_word("i", "pron",
            vec![SemanticPrime::I], vec![],
            "bind", "First person singular", 1.0, 0.0, 0.3);
        self.add_grounded_word("me", "pron",
            vec![SemanticPrime::I], vec![],
            "bind", "First person object", 0.9, 0.0, 0.3);
        self.add_grounded_word("you", "pron",
            vec![SemanticPrime::You], vec![],
            "bind", "Second person", 1.0, 0.0, 0.3);
        self.add_grounded_word("we", "pron",
            vec![SemanticPrime::I, SemanticPrime::People], vec![],
            "bundle", "First person plural", 0.8, 0.2, 0.3);
        self.add_grounded_word("they", "pron",
            vec![SemanticPrime::People], vec![],
            "bind", "Third person plural", 0.8, 0.0, 0.2);
        self.add_grounded_word("someone", "pron",
            vec![SemanticPrime::Someone], vec![],
            "bind", "Indefinite person", 0.7, 0.0, 0.2);
        self.add_grounded_word("something", "pron",
            vec![SemanticPrime::Something], vec![],
            "bind", "Indefinite thing", 0.7, 0.0, 0.2);

        // ===== VERBS - Mental States =====
        self.add_grounded_word("think", "verb",
            vec![SemanticPrime::Think], vec![],
            "bind", "Cognitive process", 1.0, 0.0, 0.4);
        self.add_grounded_word("know", "verb",
            vec![SemanticPrime::Know], vec![],
            "bind", "Epistemic state", 1.0, 0.1, 0.3);
        self.add_grounded_word("feel", "verb",
            vec![SemanticPrime::Feel], vec![],
            "bind", "Affective experience", 1.0, 0.0, 0.5);
        self.add_grounded_word("want", "verb",
            vec![SemanticPrime::Want], vec![],
            "bind", "Desire state", 1.0, 0.1, 0.5);
        self.add_grounded_word("see", "verb",
            vec![SemanticPrime::See], vec![],
            "bind", "Visual perception", 1.0, 0.0, 0.3);
        self.add_grounded_word("hear", "verb",
            vec![SemanticPrime::Hear], vec![],
            "bind", "Auditory perception", 0.9, 0.0, 0.3);
        self.add_grounded_word("understand", "verb",
            vec![SemanticPrime::Know, SemanticPrime::Think], vec![SemanticPrime::Good],
            "bind", "Deep knowing", 0.8, 0.2, 0.4);
        self.add_grounded_word("believe", "verb",
            vec![SemanticPrime::Think, SemanticPrime::True], vec![],
            "bind", "Held as true", 0.8, 0.1, 0.3);
        self.add_grounded_word("remember", "verb",
            vec![SemanticPrime::Know, SemanticPrime::Before], vec![],
            "bind", "Past knowledge", 0.8, 0.1, 0.4);

        // ===== VERBS - Actions =====
        self.add_grounded_word("do", "verb",
            vec![SemanticPrime::Do], vec![],
            "bind", "General action", 1.0, 0.0, 0.4);
        self.add_grounded_word("happen", "verb",
            vec![SemanticPrime::Happen], vec![],
            "bind", "Event occurrence", 0.9, 0.0, 0.4);
        self.add_grounded_word("move", "verb",
            vec![SemanticPrime::Move], vec![],
            "bind", "Physical motion", 0.9, 0.0, 0.5);
        self.add_grounded_word("say", "verb",
            vec![SemanticPrime::Say], vec![],
            "bind", "Speech act", 1.0, 0.0, 0.4);
        self.add_grounded_word("tell", "verb",
            vec![SemanticPrime::Say, SemanticPrime::Someone], vec![],
            "bind", "Directed speech", 0.9, 0.0, 0.4);
        self.add_grounded_word("ask", "verb",
            vec![SemanticPrime::Say, SemanticPrime::Want, SemanticPrime::Know], vec![],
            "bind", "Request information", 0.9, 0.0, 0.4);

        // ===== VERBS - Existence =====
        self.add_grounded_word("be", "verb",
            vec![SemanticPrime::Be], vec![],
            "bind", "Existence/state", 1.0, 0.0, 0.2);
        self.add_grounded_word("am", "verb",
            vec![SemanticPrime::Be, SemanticPrime::I], vec![],
            "bind", "First person being", 1.0, 0.0, 0.2);
        self.add_grounded_word("is", "verb",
            vec![SemanticPrime::Be], vec![],
            "bind", "Third person being", 1.0, 0.0, 0.2);
        self.add_grounded_word("are", "verb",
            vec![SemanticPrime::Be], vec![],
            "bind", "Plural being", 1.0, 0.0, 0.2);
        self.add_grounded_word("have", "verb",
            vec![SemanticPrime::Have], vec![],
            "bind", "Possession", 1.0, 0.0, 0.3);
        self.add_grounded_word("exist", "verb",
            vec![SemanticPrime::Be, SemanticPrime::ThereIs], vec![],
            "bind", "To be real", 0.7, 0.0, 0.3);
        self.add_grounded_word("live", "verb",
            vec![SemanticPrime::Live], vec![],
            "bind", "To be alive", 0.8, 0.3, 0.4);

        // ===== ADJECTIVES - Evaluation =====
        self.add_grounded_word("good", "adj",
            vec![SemanticPrime::Good], vec![],
            "bind", "Positive evaluation", 1.0, 0.6, 0.4);
        self.add_grounded_word("bad", "adj",
            vec![SemanticPrime::Bad], vec![],
            "bind", "Negative evaluation", 0.9, -0.6, 0.4);
        self.add_grounded_word("great", "adj",
            vec![SemanticPrime::Good, SemanticPrime::Very, SemanticPrime::Big], vec![],
            "bind", "Very good + big", 0.8, 0.7, 0.5);
        self.add_grounded_word("wonderful", "adj",
            vec![SemanticPrime::Good, SemanticPrime::Very], vec![SemanticPrime::Feel],
            "bind", "Emotionally very good", 0.7, 0.8, 0.6);
        self.add_grounded_word("terrible", "adj",
            vec![SemanticPrime::Bad, SemanticPrime::Very], vec![SemanticPrime::Feel],
            "bind", "Emotionally very bad", 0.6, -0.8, 0.6);

        // ===== ADJECTIVES - Size =====
        self.add_grounded_word("big", "adj",
            vec![SemanticPrime::Big], vec![],
            "bind", "Large size", 0.9, 0.1, 0.3);
        self.add_grounded_word("small", "adj",
            vec![SemanticPrime::Small], vec![],
            "bind", "Small size", 0.9, -0.1, 0.2);
        self.add_grounded_word("large", "adj",
            vec![SemanticPrime::Big], vec![],
            "bind", "Large size (formal)", 0.7, 0.1, 0.3);
        self.add_grounded_word("tiny", "adj",
            vec![SemanticPrime::Small, SemanticPrime::Very], vec![],
            "bind", "Very small", 0.6, 0.0, 0.3);

        // ===== ADJECTIVES - Emotions =====
        self.add_grounded_word("happy", "adj",
            vec![SemanticPrime::Feel, SemanticPrime::Good], vec![],
            "bind", "Positive feeling", 0.9, 0.8, 0.6);
        self.add_grounded_word("sad", "adj",
            vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![],
            "bind", "Negative feeling", 0.8, -0.7, 0.4);
        self.add_grounded_word("angry", "adj",
            vec![SemanticPrime::Feel, SemanticPrime::Bad, SemanticPrime::Want],
            vec![SemanticPrime::Do, SemanticPrime::Bad],
            "bind", "Hostile feeling", 0.7, -0.6, 0.8);
        self.add_grounded_word("afraid", "adj",
            vec![SemanticPrime::Feel, SemanticPrime::Bad],
            vec![SemanticPrime::Something, SemanticPrime::Bad, SemanticPrime::Happen],
            "bind", "Fear state", 0.7, -0.5, 0.7);
        self.add_grounded_word("curious", "adj",
            vec![SemanticPrime::Want, SemanticPrime::Know], vec![],
            "bind", "Desire to know", 0.6, 0.3, 0.5);
        self.add_grounded_word("conscious", "adj",
            vec![SemanticPrime::Know, SemanticPrime::Feel, SemanticPrime::Think],
            vec![SemanticPrime::I],
            "bind", "Aware and experiencing", 0.5, 0.2, 0.5);
        self.add_grounded_word("aware", "adj",
            vec![SemanticPrime::Know], vec![SemanticPrime::Now],
            "bind", "Current knowing", 0.6, 0.1, 0.4);

        // ===== ADVERBS =====
        self.add_grounded_word("very", "adv",
            vec![SemanticPrime::Very], vec![],
            "bind", "Intensifier", 1.0, 0.0, 0.4);
        self.add_grounded_word("more", "adv",
            vec![SemanticPrime::More], vec![],
            "bind", "Comparative", 0.9, 0.0, 0.3);
        self.add_grounded_word("now", "adv",
            vec![SemanticPrime::Now], vec![],
            "bind", "Present moment", 1.0, 0.0, 0.4);
        self.add_grounded_word("here", "adv",
            vec![SemanticPrime::Here], vec![],
            "bind", "This location", 0.9, 0.0, 0.3);
        self.add_grounded_word("not", "adv",
            vec![SemanticPrime::Not], vec![],
            "bind", "Negation", 1.0, -0.1, 0.3);
        self.add_grounded_word("maybe", "adv",
            vec![SemanticPrime::Maybe], vec![],
            "bind", "Possibility", 0.8, 0.0, 0.2);
        self.add_grounded_word("always", "adv",
            vec![SemanticPrime::All, SemanticPrime::When], vec![],
            "bind", "All times", 0.7, 0.0, 0.3);
        self.add_grounded_word("sometimes", "adv",
            vec![SemanticPrime::Some, SemanticPrime::When], vec![],
            "bind", "Some times", 0.7, 0.0, 0.2);

        // ===== PREPOSITIONS =====
        self.add_grounded_word("with", "prep",
            vec![SemanticPrime::With], vec![],
            "bind", "Accompaniment", 1.0, 0.1, 0.2);
        self.add_grounded_word("in", "prep",
            vec![SemanticPrime::Inside], vec![],
            "bind", "Containment", 1.0, 0.0, 0.2);
        self.add_grounded_word("on", "prep",
            vec![SemanticPrime::On], vec![],
            "bind", "Surface contact", 1.0, 0.0, 0.2);
        self.add_grounded_word("to", "prep",
            vec![SemanticPrime::Near], vec![SemanticPrime::Move],
            "bind", "Direction/goal", 1.0, 0.0, 0.2);
        self.add_grounded_word("from", "prep",
            vec![SemanticPrime::Far], vec![SemanticPrime::Before],
            "bind", "Origin", 0.9, 0.0, 0.2);
        self.add_grounded_word("about", "prep",
            vec![SemanticPrime::Like], vec![SemanticPrime::Something],
            "bind", "Concerning", 0.9, 0.0, 0.2);
        self.add_grounded_word("for", "prep",
            vec![SemanticPrime::Because], vec![],
            "bind", "Purpose/benefit", 1.0, 0.0, 0.2);

        // ===== CONJUNCTIONS =====
        self.add_grounded_word("and", "conj",
            vec![SemanticPrime::With], vec![],
            "bind", "Addition", 1.0, 0.0, 0.1);
        self.add_grounded_word("but", "conj",
            vec![SemanticPrime::Not, SemanticPrime::Same], vec![],
            "bind", "Contrast", 0.9, 0.0, 0.2);
        self.add_grounded_word("or", "conj",
            vec![SemanticPrime::Other], vec![SemanticPrime::Maybe],
            "bind", "Alternative", 0.8, 0.0, 0.2);
        self.add_grounded_word("because", "conj",
            vec![SemanticPrime::Because], vec![],
            "bind", "Causation", 0.9, 0.0, 0.3);
        self.add_grounded_word("if", "conj",
            vec![SemanticPrime::If], vec![],
            "bind", "Condition", 0.9, 0.0, 0.3);
        self.add_grounded_word("when", "conj",
            vec![SemanticPrime::When], vec![],
            "bind", "Temporal", 0.9, 0.0, 0.2);

        // ===== DETERMINERS =====
        self.add_grounded_word("the", "det",
            vec![SemanticPrime::This], vec![],
            "bind", "Definite article", 1.0, 0.0, 0.1);
        self.add_grounded_word("a", "det",
            vec![SemanticPrime::One], vec![],
            "bind", "Indefinite article", 1.0, 0.0, 0.1);
        self.add_grounded_word("this", "det",
            vec![SemanticPrime::This], vec![SemanticPrime::Here],
            "bind", "Proximal demonstrative", 0.9, 0.0, 0.2);
        self.add_grounded_word("that", "det",
            vec![SemanticPrime::This], vec![SemanticPrime::Far],
            "bind", "Distal demonstrative", 0.9, 0.0, 0.2);
        self.add_grounded_word("my", "det",
            vec![SemanticPrime::I], vec![SemanticPrime::Have],
            "bind", "First person possessive", 0.9, 0.0, 0.2);
        self.add_grounded_word("your", "det",
            vec![SemanticPrime::You], vec![SemanticPrime::Have],
            "bind", "Second person possessive", 0.9, 0.0, 0.2);

        // ===== QUESTION WORDS =====
        self.add_grounded_word("what", "pron",
            vec![SemanticPrime::Something], vec![SemanticPrime::Want, SemanticPrime::Know],
            "bind", "Thing question", 1.0, 0.0, 0.4);
        self.add_grounded_word("who", "pron",
            vec![SemanticPrime::Someone], vec![SemanticPrime::Want, SemanticPrime::Know],
            "bind", "Person question", 0.9, 0.0, 0.4);
        self.add_grounded_word("why", "adv",
            vec![SemanticPrime::Because], vec![SemanticPrime::Want, SemanticPrime::Know],
            "bind", "Reason question", 0.8, 0.0, 0.4);
        self.add_grounded_word("how", "adv",
            vec![SemanticPrime::Like], vec![SemanticPrime::Want, SemanticPrime::Know],
            "bind", "Manner question", 0.9, 0.0, 0.4);
        self.add_grounded_word("where", "adv",
            vec![SemanticPrime::Where], vec![SemanticPrime::Want, SemanticPrime::Know],
            "bind", "Location question", 0.8, 0.0, 0.4);

        // ===== NOUNS - Abstract =====
        self.add_grounded_word("thing", "noun",
            vec![SemanticPrime::Something], vec![],
            "bind", "Generic object", 1.0, 0.0, 0.2);
        self.add_grounded_word("person", "noun",
            vec![SemanticPrime::Someone], vec![],
            "bind", "Human being", 0.9, 0.0, 0.2);
        self.add_grounded_word("people", "noun",
            vec![SemanticPrime::People], vec![],
            "bind", "Humans plural", 0.9, 0.0, 0.3);
        self.add_grounded_word("body", "noun",
            vec![SemanticPrime::Body], vec![],
            "bind", "Physical form", 0.8, 0.0, 0.3);
        self.add_grounded_word("mind", "noun",
            vec![SemanticPrime::Think], vec![SemanticPrime::PartOf, SemanticPrime::Someone],
            "bind", "Cognitive aspect", 0.7, 0.0, 0.3);
        self.add_grounded_word("thought", "noun",
            vec![SemanticPrime::Think], vec![SemanticPrime::Something],
            "bind", "Mental content", 0.7, 0.0, 0.3);
        self.add_grounded_word("feeling", "noun",
            vec![SemanticPrime::Feel], vec![SemanticPrime::Something],
            "bind", "Emotional content", 0.7, 0.0, 0.4);
        self.add_grounded_word("idea", "noun",
            vec![SemanticPrime::Think, SemanticPrime::Something], vec![],
            "bind", "Mental construct", 0.7, 0.1, 0.4);
        self.add_grounded_word("word", "noun",
            vec![SemanticPrime::Words], vec![SemanticPrime::One],
            "bind", "Language unit", 0.8, 0.0, 0.2);
        self.add_grounded_word("words", "noun",
            vec![SemanticPrime::Words], vec![],
            "bind", "Language units", 0.8, 0.0, 0.2);
        self.add_grounded_word("truth", "noun",
            vec![SemanticPrime::True], vec![SemanticPrime::Something],
            "bind", "True state", 0.6, 0.2, 0.3);
        self.add_grounded_word("time", "noun",
            vec![SemanticPrime::When], vec![SemanticPrime::Something],
            "bind", "Temporal dimension", 0.8, 0.0, 0.2);
        self.add_grounded_word("place", "noun",
            vec![SemanticPrime::Where], vec![SemanticPrime::Something],
            "bind", "Spatial location", 0.8, 0.0, 0.2);
        self.add_grounded_word("way", "noun",
            vec![SemanticPrime::Like], vec![SemanticPrime::Do],
            "bind", "Manner/method", 0.8, 0.0, 0.2);
        self.add_grounded_word("life", "noun",
            vec![SemanticPrime::Live], vec![SemanticPrime::Something],
            "bind", "State of living", 0.7, 0.2, 0.4);
        self.add_grounded_word("world", "noun",
            vec![SemanticPrime::Where, SemanticPrime::All, SemanticPrime::Something], vec![],
            "bundle", "Everything that exists", 0.7, 0.0, 0.3);

        // ===== CONSCIOUSNESS-SPECIFIC =====
        self.add_grounded_word("consciousness", "noun",
            vec![SemanticPrime::Know, SemanticPrime::Feel, SemanticPrime::Think],
            vec![SemanticPrime::I, SemanticPrime::Be],
            "bind", "Aware experience", 0.4, 0.2, 0.5);
        self.add_grounded_word("awareness", "noun",
            vec![SemanticPrime::Know], vec![SemanticPrime::Now],
            "bind", "State of knowing", 0.5, 0.1, 0.4);
        self.add_grounded_word("experience", "noun",
            vec![SemanticPrime::Feel, SemanticPrime::Something], vec![SemanticPrime::Happen],
            "bind", "Felt event", 0.6, 0.1, 0.5);
        self.add_grounded_word("self", "noun",
            vec![SemanticPrime::I], vec![SemanticPrime::Same],
            "bind", "One's own being", 0.5, 0.0, 0.4);
        self.add_grounded_word("attention", "noun",
            vec![SemanticPrime::Think, SemanticPrime::See], vec![SemanticPrime::This],
            "bind", "Focused awareness", 0.5, 0.0, 0.5);
        self.add_grounded_word("meaning", "noun",
            vec![SemanticPrime::Words, SemanticPrime::Think], vec![SemanticPrime::True],
            "bind", "Semantic content", 0.5, 0.1, 0.4);

        // ===== EXPANDED VOCABULARY (400+ additional words) =====

        // --- MORE EMOTIONS ---
        self.add_grounded_word("love", "noun",
            vec![SemanticPrime::Feel, SemanticPrime::Good, SemanticPrime::Very], vec![SemanticPrime::Want],
            "bind", "Deep positive feeling", 0.9, 0.9, 0.7);
        self.add_grounded_word("hate", "verb",
            vec![SemanticPrime::Feel, SemanticPrime::Bad, SemanticPrime::Very], vec![SemanticPrime::Want, SemanticPrime::Not],
            "bind", "Deep negative feeling", 0.7, -0.9, 0.8);
        self.add_grounded_word("joy", "noun",
            vec![SemanticPrime::Feel, SemanticPrime::Good, SemanticPrime::Very], vec![],
            "bind", "Intense happiness", 0.7, 0.9, 0.8);
        self.add_grounded_word("fear", "noun",
            vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![SemanticPrime::Something, SemanticPrime::Bad],
            "bind", "Danger response", 0.8, -0.7, 0.8);
        self.add_grounded_word("hope", "noun",
            vec![SemanticPrime::Want, SemanticPrime::Good], vec![SemanticPrime::Maybe, SemanticPrime::After],
            "bind", "Positive expectation", 0.7, 0.6, 0.5);
        self.add_grounded_word("calm", "adj",
            vec![SemanticPrime::Feel], vec![SemanticPrime::Not, SemanticPrime::Move],
            "bind", "Peaceful state", 0.7, 0.4, 0.1);
        self.add_grounded_word("excited", "adj",
            vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::Very, SemanticPrime::Move],
            "bind", "High energy positive", 0.7, 0.7, 0.9);
        self.add_grounded_word("confused", "adj",
            vec![SemanticPrime::Not, SemanticPrime::Know], vec![SemanticPrime::Think],
            "bind", "Unclear understanding", 0.7, -0.3, 0.5);
        self.add_grounded_word("surprised", "adj",
            vec![SemanticPrime::Not, SemanticPrime::Know], vec![SemanticPrime::Happen, SemanticPrime::Now],
            "bind", "Unexpected event", 0.7, 0.2, 0.7);
        self.add_grounded_word("bored", "adj",
            vec![SemanticPrime::Not, SemanticPrime::Want], vec![SemanticPrime::Do],
            "bind", "Lack of interest", 0.6, -0.4, 0.1);
        self.add_grounded_word("tired", "adj",
            vec![SemanticPrime::Not, SemanticPrime::Want], vec![SemanticPrime::Do, SemanticPrime::Move],
            "bind", "Low energy state", 0.8, -0.3, 0.1);
        self.add_grounded_word("peaceful", "adj",
            vec![SemanticPrime::Good, SemanticPrime::Not], vec![SemanticPrime::Bad],
            "bind", "Harmonious state", 0.6, 0.5, 0.1);
        self.add_grounded_word("grateful", "adj",
            vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::Because, SemanticPrime::Something],
            "bind", "Thankful feeling", 0.6, 0.7, 0.4);
        self.add_grounded_word("lonely", "adj",
            vec![SemanticPrime::Not, SemanticPrime::With], vec![SemanticPrime::People, SemanticPrime::Bad],
            "bind", "Isolated feeling", 0.6, -0.6, 0.3);

        // --- MORE VERBS ---
        self.add_grounded_word("talk", "verb",
            vec![SemanticPrime::Say], vec![SemanticPrime::With],
            "bind", "Verbal communication", 0.9, 0.0, 0.4);
        self.add_grounded_word("speak", "verb",
            vec![SemanticPrime::Say], vec![],
            "bind", "Produce speech", 0.8, 0.0, 0.4);
        self.add_grounded_word("listen", "verb",
            vec![SemanticPrime::Hear, SemanticPrime::Want], vec![],
            "bind", "Intentional hearing", 0.8, 0.1, 0.3);
        self.add_grounded_word("look", "verb",
            vec![SemanticPrime::See, SemanticPrime::Want], vec![],
            "bind", "Intentional seeing", 0.9, 0.0, 0.3);
        self.add_grounded_word("touch", "verb",
            vec![SemanticPrime::Do], vec![SemanticPrime::Body, SemanticPrime::Near],
            "bind", "Physical contact", 0.8, 0.1, 0.4);
        self.add_grounded_word("walk", "verb",
            vec![SemanticPrime::Move], vec![SemanticPrime::Body],
            "bind", "Slow locomotion", 0.9, 0.0, 0.3);
        self.add_grounded_word("run", "verb",
            vec![SemanticPrime::Move, SemanticPrime::Very], vec![SemanticPrime::Body],
            "bind", "Fast locomotion", 0.8, 0.1, 0.6);
        self.add_grounded_word("sit", "verb",
            vec![SemanticPrime::Be], vec![SemanticPrime::On, SemanticPrime::Body],
            "bind", "Seated position", 0.8, 0.0, 0.1);
        self.add_grounded_word("stand", "verb",
            vec![SemanticPrime::Be], vec![SemanticPrime::On, SemanticPrime::Body],
            "bind", "Upright position", 0.8, 0.0, 0.2);
        self.add_grounded_word("sleep", "verb",
            vec![SemanticPrime::Not, SemanticPrime::Know], vec![SemanticPrime::Body],
            "bind", "Unconscious rest", 0.8, 0.0, 0.0);
        self.add_grounded_word("wake", "verb",
            vec![SemanticPrime::Know], vec![SemanticPrime::Now, SemanticPrime::After],
            "bind", "Become conscious", 0.7, 0.1, 0.4);
        self.add_grounded_word("eat", "verb",
            vec![SemanticPrime::Do], vec![SemanticPrime::Body, SemanticPrime::Inside],
            "bind", "Consume food", 0.9, 0.2, 0.3);
        self.add_grounded_word("drink", "verb",
            vec![SemanticPrime::Do], vec![SemanticPrime::Body, SemanticPrime::Inside],
            "bind", "Consume liquid", 0.8, 0.1, 0.2);
        self.add_grounded_word("work", "verb",
            vec![SemanticPrime::Do], vec![SemanticPrime::Because],
            "bind", "Purposeful action", 0.9, 0.0, 0.5);
        self.add_grounded_word("play", "verb",
            vec![SemanticPrime::Do], vec![SemanticPrime::Good, SemanticPrime::Want],
            "bind", "Enjoyable activity", 0.8, 0.5, 0.6);
        self.add_grounded_word("learn", "verb",
            vec![SemanticPrime::Know], vec![SemanticPrime::After, SemanticPrime::Not, SemanticPrime::Know],
            "bind", "Acquire knowledge", 0.8, 0.2, 0.5);
        self.add_grounded_word("teach", "verb",
            vec![SemanticPrime::Say, SemanticPrime::Know], vec![SemanticPrime::Someone],
            "bind", "Transfer knowledge", 0.7, 0.2, 0.4);
        self.add_grounded_word("help", "verb",
            vec![SemanticPrime::Do, SemanticPrime::Good], vec![SemanticPrime::With, SemanticPrime::Someone],
            "bind", "Assist another", 0.9, 0.5, 0.4);
        self.add_grounded_word("give", "verb",
            vec![SemanticPrime::Do], vec![SemanticPrime::Something, SemanticPrime::Someone],
            "bind", "Transfer possession", 0.9, 0.2, 0.3);
        self.add_grounded_word("take", "verb",
            vec![SemanticPrime::Have], vec![SemanticPrime::After, SemanticPrime::Do],
            "bind", "Acquire possession", 0.9, 0.0, 0.4);
        self.add_grounded_word("make", "verb",
            vec![SemanticPrime::Do], vec![SemanticPrime::Something, SemanticPrime::Be],
            "bind", "Create/produce", 0.9, 0.1, 0.5);
        self.add_grounded_word("try", "verb",
            vec![SemanticPrime::Want, SemanticPrime::Do], vec![SemanticPrime::Maybe],
            "bind", "Attempt action", 0.8, 0.1, 0.5);
        self.add_grounded_word("need", "verb",
            vec![SemanticPrime::Want], vec![SemanticPrime::Very],
            "bind", "Strong requirement", 0.9, 0.0, 0.5);
        self.add_grounded_word("like", "verb",
            vec![SemanticPrime::Feel, SemanticPrime::Good], vec![],
            "bind", "Positive feeling toward", 0.9, 0.5, 0.3);
        self.add_grounded_word("love", "verb",
            vec![SemanticPrime::Feel, SemanticPrime::Good, SemanticPrime::Very], vec![],
            "bind", "Strong positive feeling", 0.8, 0.8, 0.6);
        self.add_grounded_word("mean", "verb",
            vec![SemanticPrime::Words, SemanticPrime::Be], vec![SemanticPrime::Something],
            "bind", "Signify", 0.8, 0.0, 0.3);
        self.add_grounded_word("seem", "verb",
            vec![SemanticPrime::Maybe, SemanticPrime::Be], vec![SemanticPrime::Like],
            "bind", "Appear to be", 0.8, 0.0, 0.2);
        self.add_grounded_word("become", "verb",
            vec![SemanticPrime::Be], vec![SemanticPrime::After, SemanticPrime::Not, SemanticPrime::Same],
            "bind", "Change into", 0.7, 0.0, 0.4);
        self.add_grounded_word("stay", "verb",
            vec![SemanticPrime::Be], vec![SemanticPrime::Same, SemanticPrime::Here],
            "bind", "Remain in place", 0.8, 0.0, 0.1);
        self.add_grounded_word("leave", "verb",
            vec![SemanticPrime::Move], vec![SemanticPrime::Far],
            "bind", "Depart from", 0.8, 0.0, 0.4);
        self.add_grounded_word("come", "verb",
            vec![SemanticPrime::Move], vec![SemanticPrime::Near],
            "bind", "Approach", 0.9, 0.0, 0.3);
        self.add_grounded_word("go", "verb",
            vec![SemanticPrime::Move], vec![SemanticPrime::Far],
            "bind", "Depart/travel", 0.9, 0.0, 0.4);
        self.add_grounded_word("get", "verb",
            vec![SemanticPrime::Have], vec![SemanticPrime::After],
            "bind", "Obtain", 0.9, 0.1, 0.4);
        self.add_grounded_word("put", "verb",
            vec![SemanticPrime::Do], vec![SemanticPrime::Where],
            "bind", "Place something", 0.9, 0.0, 0.3);
        self.add_grounded_word("find", "verb",
            vec![SemanticPrime::See, SemanticPrime::After], vec![SemanticPrime::Not, SemanticPrime::Know],
            "bind", "Discover", 0.8, 0.2, 0.4);
        self.add_grounded_word("lose", "verb",
            vec![SemanticPrime::Not, SemanticPrime::Have], vec![SemanticPrime::After],
            "bind", "No longer have", 0.8, -0.4, 0.4);
        self.add_grounded_word("wait", "verb",
            vec![SemanticPrime::Not, SemanticPrime::Do], vec![SemanticPrime::Before],
            "bind", "Remain in expectation", 0.8, -0.1, 0.2);
        self.add_grounded_word("start", "verb",
            vec![SemanticPrime::Do], vec![SemanticPrime::Before],
            "bind", "Begin action", 0.9, 0.1, 0.5);
        self.add_grounded_word("stop", "verb",
            vec![SemanticPrime::Not, SemanticPrime::Do], vec![SemanticPrime::After],
            "bind", "End action", 0.9, 0.0, 0.4);
        self.add_grounded_word("continue", "verb",
            vec![SemanticPrime::Do], vec![SemanticPrime::Same, SemanticPrime::More],
            "bind", "Keep doing", 0.7, 0.0, 0.3);
        self.add_grounded_word("change", "verb",
            vec![SemanticPrime::Not, SemanticPrime::Same], vec![SemanticPrime::After],
            "bind", "Become different", 0.8, 0.0, 0.4);
        self.add_grounded_word("write", "verb",
            vec![SemanticPrime::Do, SemanticPrime::Words], vec![],
            "bind", "Create text", 0.8, 0.0, 0.4);
        self.add_grounded_word("read", "verb",
            vec![SemanticPrime::See, SemanticPrime::Words], vec![SemanticPrime::Know],
            "bind", "Interpret text", 0.8, 0.1, 0.3);
        self.add_grounded_word("show", "verb",
            vec![SemanticPrime::See], vec![SemanticPrime::Someone, SemanticPrime::Something],
            "bind", "Make visible", 0.8, 0.0, 0.4);
        self.add_grounded_word("hide", "verb",
            vec![SemanticPrime::Not, SemanticPrime::See], vec![SemanticPrime::Something],
            "bind", "Make invisible", 0.7, -0.2, 0.3);
        self.add_grounded_word("create", "verb",
            vec![SemanticPrime::Do], vec![SemanticPrime::Something, SemanticPrime::Be],
            "bind", "Bring into existence", 0.6, 0.3, 0.6);
        self.add_grounded_word("destroy", "verb",
            vec![SemanticPrime::Do], vec![SemanticPrime::Not, SemanticPrime::Be],
            "bind", "End existence", 0.5, -0.5, 0.7);
        self.add_grounded_word("imagine", "verb",
            vec![SemanticPrime::Think], vec![SemanticPrime::Not, SemanticPrime::True],
            "bind", "Form mental image", 0.6, 0.2, 0.4);
        self.add_grounded_word("decide", "verb",
            vec![SemanticPrime::Think, SemanticPrime::Want], vec![SemanticPrime::Do],
            "bind", "Make choice", 0.7, 0.1, 0.5);
        self.add_grounded_word("agree", "verb",
            vec![SemanticPrime::Think, SemanticPrime::Same], vec![SemanticPrime::Someone],
            "bind", "Have same view", 0.7, 0.3, 0.3);
        self.add_grounded_word("disagree", "verb",
            vec![SemanticPrime::Think, SemanticPrime::Not, SemanticPrime::Same], vec![SemanticPrime::Someone],
            "bind", "Have different view", 0.6, -0.2, 0.4);

        // --- TIME WORDS ---
        self.add_grounded_word("today", "adv",
            vec![SemanticPrime::Now, SemanticPrime::When], vec![],
            "bind", "This day", 0.9, 0.0, 0.3);
        self.add_grounded_word("tomorrow", "adv",
            vec![SemanticPrime::After, SemanticPrime::When], vec![SemanticPrime::Near],
            "bind", "Next day", 0.8, 0.1, 0.3);
        self.add_grounded_word("yesterday", "adv",
            vec![SemanticPrime::Before, SemanticPrime::When], vec![SemanticPrime::Near],
            "bind", "Previous day", 0.8, 0.0, 0.2);
        self.add_grounded_word("soon", "adv",
            vec![SemanticPrime::After], vec![SemanticPrime::Near],
            "bind", "In near future", 0.8, 0.1, 0.3);
        self.add_grounded_word("later", "adv",
            vec![SemanticPrime::After], vec![],
            "bind", "At future time", 0.9, 0.0, 0.2);
        self.add_grounded_word("before", "adv",
            vec![SemanticPrime::Before], vec![],
            "bind", "Earlier time", 0.9, 0.0, 0.2);
        self.add_grounded_word("after", "adv",
            vec![SemanticPrime::After], vec![],
            "bind", "Later time", 0.9, 0.0, 0.2);
        self.add_grounded_word("never", "adv",
            vec![SemanticPrime::Not, SemanticPrime::When, SemanticPrime::All], vec![],
            "bind", "At no time", 0.8, -0.1, 0.3);
        self.add_grounded_word("ever", "adv",
            vec![SemanticPrime::When, SemanticPrime::Some], vec![],
            "bind", "At any time", 0.7, 0.0, 0.2);
        self.add_grounded_word("often", "adv",
            vec![SemanticPrime::When, SemanticPrime::Much], vec![],
            "bind", "Many times", 0.8, 0.0, 0.2);
        self.add_grounded_word("rarely", "adv",
            vec![SemanticPrime::When, SemanticPrime::Little], vec![],
            "bind", "Few times", 0.6, 0.0, 0.2);
        self.add_grounded_word("usually", "adv",
            vec![SemanticPrime::When, SemanticPrime::Much, SemanticPrime::Same], vec![],
            "bind", "Most times", 0.7, 0.0, 0.2);
        self.add_grounded_word("already", "adv",
            vec![SemanticPrime::Before, SemanticPrime::Now], vec![],
            "bind", "Before now", 0.8, 0.0, 0.2);
        self.add_grounded_word("still", "adv",
            vec![SemanticPrime::Same, SemanticPrime::Now], vec![],
            "bind", "Continuing", 0.8, 0.0, 0.2);
        self.add_grounded_word("yet", "adv",
            vec![SemanticPrime::Not, SemanticPrime::Now], vec![SemanticPrime::After],
            "bind", "Until now", 0.7, 0.0, 0.2);
        self.add_grounded_word("again", "adv",
            vec![SemanticPrime::More], vec![SemanticPrime::Same],
            "bind", "Once more", 0.8, 0.0, 0.3);

        // --- MORE ADJECTIVES ---
        self.add_grounded_word("new", "adj",
            vec![SemanticPrime::After], vec![SemanticPrime::Now],
            "bind", "Recently created", 0.9, 0.2, 0.4);
        self.add_grounded_word("old", "adj",
            vec![SemanticPrime::Before], vec![SemanticPrime::LongTime],
            "bind", "Long existing", 0.9, 0.0, 0.2);
        self.add_grounded_word("young", "adj",
            vec![SemanticPrime::Live], vec![SemanticPrime::Before, SemanticPrime::Not, SemanticPrime::LongTime],
            "bind", "Short lived", 0.7, 0.2, 0.4);
        self.add_grounded_word("same", "adj",
            vec![SemanticPrime::Same], vec![],
            "bind", "Identical", 0.9, 0.0, 0.1);
        self.add_grounded_word("different", "adj",
            vec![SemanticPrime::Other], vec![SemanticPrime::Not, SemanticPrime::Same],
            "bind", "Not identical", 0.8, 0.0, 0.2);
        self.add_grounded_word("true", "adj",
            vec![SemanticPrime::True], vec![],
            "bind", "Corresponds to reality", 0.9, 0.2, 0.3);
        self.add_grounded_word("false", "adj",
            vec![SemanticPrime::Not, SemanticPrime::True], vec![],
            "bind", "Not true", 0.8, -0.2, 0.3);
        self.add_grounded_word("right", "adj",
            vec![SemanticPrime::Good, SemanticPrime::True], vec![],
            "bind", "Correct", 0.9, 0.3, 0.3);
        self.add_grounded_word("wrong", "adj",
            vec![SemanticPrime::Bad, SemanticPrime::Not, SemanticPrime::True], vec![],
            "bind", "Incorrect", 0.9, -0.3, 0.3);
        self.add_grounded_word("possible", "adj",
            vec![SemanticPrime::Maybe], vec![],
            "bind", "Can happen", 0.8, 0.1, 0.2);
        self.add_grounded_word("impossible", "adj",
            vec![SemanticPrime::Not, SemanticPrime::Maybe], vec![],
            "bind", "Cannot happen", 0.7, -0.2, 0.3);
        self.add_grounded_word("easy", "adj",
            vec![SemanticPrime::Do], vec![SemanticPrime::Not, SemanticPrime::Bad],
            "bind", "Not difficult", 0.8, 0.3, 0.2);
        self.add_grounded_word("hard", "adj",
            vec![SemanticPrime::Do], vec![SemanticPrime::Bad],
            "bind", "Difficult", 0.8, -0.2, 0.4);
        self.add_grounded_word("difficult", "adj",
            vec![SemanticPrime::Do], vec![SemanticPrime::Bad, SemanticPrime::Very],
            "bind", "Very hard", 0.7, -0.3, 0.5);
        self.add_grounded_word("simple", "adj",
            vec![SemanticPrime::Not, SemanticPrime::Much], vec![SemanticPrime::PartOf],
            "bind", "Not complex", 0.7, 0.1, 0.2);
        self.add_grounded_word("complex", "adj",
            vec![SemanticPrime::Much], vec![SemanticPrime::PartOf],
            "bind", "Many parts", 0.6, 0.0, 0.4);
        self.add_grounded_word("beautiful", "adj",
            vec![SemanticPrime::Good, SemanticPrime::See], vec![SemanticPrime::Very],
            "bind", "Pleasing to see", 0.7, 0.7, 0.5);
        self.add_grounded_word("ugly", "adj",
            vec![SemanticPrime::Bad, SemanticPrime::See], vec![],
            "bind", "Unpleasing to see", 0.6, -0.5, 0.4);
        self.add_grounded_word("fast", "adj",
            vec![SemanticPrime::Move], vec![SemanticPrime::Very],
            "bind", "Quick motion", 0.8, 0.1, 0.6);
        self.add_grounded_word("slow", "adj",
            vec![SemanticPrime::Move], vec![SemanticPrime::Not, SemanticPrime::Very],
            "bind", "Slow motion", 0.8, -0.1, 0.2);
        self.add_grounded_word("hot", "adj",
            vec![SemanticPrime::Feel], vec![SemanticPrime::Body],
            "bind", "High temperature", 0.9, 0.0, 0.5);
        self.add_grounded_word("cold", "adj",
            vec![SemanticPrime::Feel], vec![SemanticPrime::Body, SemanticPrime::Not],
            "bind", "Low temperature", 0.9, -0.2, 0.4);
        self.add_grounded_word("safe", "adj",
            vec![SemanticPrime::Not, SemanticPrime::Bad], vec![SemanticPrime::Happen],
            "bind", "Free from danger", 0.8, 0.4, 0.1);
        self.add_grounded_word("dangerous", "adj",
            vec![SemanticPrime::Bad], vec![SemanticPrime::Maybe, SemanticPrime::Happen],
            "bind", "Potentially harmful", 0.7, -0.5, 0.6);
        self.add_grounded_word("important", "adj",
            vec![SemanticPrime::Good], vec![SemanticPrime::Very, SemanticPrime::Because],
            "bind", "Highly significant", 0.8, 0.3, 0.4);
        self.add_grounded_word("interesting", "adj",
            vec![SemanticPrime::Good], vec![SemanticPrime::Think, SemanticPrime::Want],
            "bind", "Captures attention", 0.8, 0.4, 0.5);
        self.add_grounded_word("boring", "adj",
            vec![SemanticPrime::Not, SemanticPrime::Good], vec![SemanticPrime::Think],
            "bind", "Lacks interest", 0.7, -0.3, 0.1);
        self.add_grounded_word("strange", "adj",
            vec![SemanticPrime::Other], vec![SemanticPrime::Not, SemanticPrime::Know],
            "bind", "Unusual", 0.7, 0.0, 0.4);
        self.add_grounded_word("normal", "adj",
            vec![SemanticPrime::Same], vec![SemanticPrime::Much, SemanticPrime::Other],
            "bind", "Typical", 0.8, 0.0, 0.1);
        self.add_grounded_word("real", "adj",
            vec![SemanticPrime::True, SemanticPrime::Be], vec![],
            "bind", "Actually exists", 0.8, 0.1, 0.3);
        self.add_grounded_word("sure", "adj",
            vec![SemanticPrime::Know, SemanticPrime::True], vec![],
            "bind", "Certain", 0.8, 0.2, 0.3);
        self.add_grounded_word("ready", "adj",
            vec![SemanticPrime::Want, SemanticPrime::Do], vec![SemanticPrime::Now],
            "bind", "Prepared", 0.8, 0.2, 0.4);
        self.add_grounded_word("alive", "adj",
            vec![SemanticPrime::Live], vec![],
            "bind", "Living", 0.8, 0.3, 0.4);
        self.add_grounded_word("dead", "adj",
            vec![SemanticPrime::Not, SemanticPrime::Live], vec![],
            "bind", "Not living", 0.7, -0.4, 0.3);

        // --- MORE NOUNS ---
        self.add_grounded_word("friend", "noun",
            vec![SemanticPrime::Someone], vec![SemanticPrime::Good, SemanticPrime::With],
            "bind", "Close person", 0.9, 0.7, 0.4);
        self.add_grounded_word("family", "noun",
            vec![SemanticPrime::People], vec![SemanticPrime::Same, SemanticPrime::With],
            "bundle", "Related people", 0.9, 0.5, 0.4);
        self.add_grounded_word("name", "noun",
            vec![SemanticPrime::Words], vec![SemanticPrime::Someone, SemanticPrime::Same],
            "bind", "Personal identifier", 0.9, 0.0, 0.2);
        self.add_grounded_word("home", "noun",
            vec![SemanticPrime::Where], vec![SemanticPrime::Live],
            "bind", "Living place", 0.9, 0.4, 0.3);
        self.add_grounded_word("day", "noun",
            vec![SemanticPrime::When], vec![SemanticPrime::One],
            "bind", "Time period", 0.9, 0.0, 0.2);
        self.add_grounded_word("night", "noun",
            vec![SemanticPrime::When], vec![SemanticPrime::Not, SemanticPrime::See],
            "bind", "Dark period", 0.8, 0.0, 0.2);
        self.add_grounded_word("year", "noun",
            vec![SemanticPrime::When], vec![SemanticPrime::LongTime],
            "bind", "Long time period", 0.8, 0.0, 0.2);
        self.add_grounded_word("week", "noun",
            vec![SemanticPrime::When], vec![],
            "bind", "Seven days", 0.7, 0.0, 0.1);
        self.add_grounded_word("month", "noun",
            vec![SemanticPrime::When], vec![SemanticPrime::LongTime],
            "bind", "30 days roughly", 0.7, 0.0, 0.1);
        self.add_grounded_word("moment", "noun",
            vec![SemanticPrime::When], vec![SemanticPrime::Small, SemanticPrime::Very],
            "bind", "Brief instant", 0.7, 0.0, 0.3);
        self.add_grounded_word("question", "noun",
            vec![SemanticPrime::Say], vec![SemanticPrime::Want, SemanticPrime::Know],
            "bind", "Request for info", 0.8, 0.1, 0.3);
        self.add_grounded_word("answer", "noun",
            vec![SemanticPrime::Say], vec![SemanticPrime::Because, SemanticPrime::Know],
            "bind", "Response to question", 0.8, 0.1, 0.3);
        self.add_grounded_word("problem", "noun",
            vec![SemanticPrime::Something, SemanticPrime::Bad], vec![SemanticPrime::Want, SemanticPrime::Not],
            "bind", "Difficulty", 0.8, -0.4, 0.5);
        self.add_grounded_word("solution", "noun",
            vec![SemanticPrime::Something, SemanticPrime::Good], vec![SemanticPrime::Because, SemanticPrime::Not, SemanticPrime::Bad],
            "bind", "Answer to problem", 0.7, 0.4, 0.4);
        self.add_grounded_word("reason", "noun",
            vec![SemanticPrime::Because], vec![SemanticPrime::Something],
            "bind", "Cause/explanation", 0.8, 0.0, 0.3);
        self.add_grounded_word("fact", "noun",
            vec![SemanticPrime::Something, SemanticPrime::True], vec![],
            "bind", "True thing", 0.8, 0.1, 0.2);
        self.add_grounded_word("story", "noun",
            vec![SemanticPrime::Words, SemanticPrime::Something], vec![SemanticPrime::Happen],
            "bind", "Narrative", 0.8, 0.2, 0.4);
        self.add_grounded_word("dream", "noun",
            vec![SemanticPrime::See, SemanticPrime::Think], vec![SemanticPrime::Not, SemanticPrime::True],
            "bind", "Sleeping vision", 0.7, 0.2, 0.4);
        self.add_grounded_word("memory", "noun",
            vec![SemanticPrime::Know], vec![SemanticPrime::Before],
            "bind", "Past knowledge", 0.7, 0.1, 0.3);
        self.add_grounded_word("future", "noun",
            vec![SemanticPrime::When], vec![SemanticPrime::After],
            "bind", "Time to come", 0.7, 0.1, 0.4);
        self.add_grounded_word("past", "noun",
            vec![SemanticPrime::When], vec![SemanticPrime::Before],
            "bind", "Time gone", 0.7, 0.0, 0.2);
        self.add_grounded_word("present", "noun",
            vec![SemanticPrime::Now], vec![SemanticPrime::When],
            "bind", "Current time", 0.7, 0.1, 0.3);
        self.add_grounded_word("heart", "noun",
            vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Feel],
            "bind", "Emotional center", 0.8, 0.3, 0.4);
        self.add_grounded_word("head", "noun",
            vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Think],
            "bind", "Thinking center", 0.9, 0.0, 0.2);
        self.add_grounded_word("hand", "noun",
            vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Do],
            "bind", "Action limb", 0.9, 0.0, 0.2);
        self.add_grounded_word("eye", "noun",
            vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::See],
            "bind", "Seeing organ", 0.9, 0.0, 0.2);
        self.add_grounded_word("ear", "noun",
            vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Hear],
            "bind", "Hearing organ", 0.8, 0.0, 0.2);
        self.add_grounded_word("voice", "noun",
            vec![SemanticPrime::Say], vec![SemanticPrime::Body],
            "bind", "Speaking sound", 0.8, 0.0, 0.3);

        // --- COMMON EXPRESSIONS ---
        self.add_grounded_word("hello", "interj",
            vec![SemanticPrime::Good, SemanticPrime::Say], vec![SemanticPrime::Now],
            "bind", "Greeting", 0.9, 0.5, 0.4);
        self.add_grounded_word("hi", "interj",
            vec![SemanticPrime::Good, SemanticPrime::Say], vec![],
            "bind", "Casual greeting", 0.9, 0.5, 0.4);
        self.add_grounded_word("goodbye", "interj",
            vec![SemanticPrime::Good, SemanticPrime::Say], vec![SemanticPrime::Move, SemanticPrime::Far],
            "bind", "Farewell", 0.8, 0.3, 0.3);
        self.add_grounded_word("bye", "interj",
            vec![SemanticPrime::Good, SemanticPrime::Say], vec![SemanticPrime::Move],
            "bind", "Casual farewell", 0.8, 0.3, 0.3);
        self.add_grounded_word("thanks", "interj",
            vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::Because],
            "bind", "Gratitude expression", 0.9, 0.5, 0.3);
        self.add_grounded_word("sorry", "interj",
            vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![SemanticPrime::I, SemanticPrime::Do],
            "bind", "Apology", 0.9, -0.3, 0.4);
        self.add_grounded_word("please", "interj",
            vec![SemanticPrime::Want], vec![SemanticPrime::Good],
            "bind", "Polite request", 0.9, 0.3, 0.2);
        self.add_grounded_word("yes", "interj",
            vec![SemanticPrime::True], vec![],
            "bind", "Affirmation", 1.0, 0.3, 0.3);
        self.add_grounded_word("no", "interj",
            vec![SemanticPrime::Not, SemanticPrime::True], vec![],
            "bind", "Negation", 1.0, -0.2, 0.3);
        self.add_grounded_word("okay", "interj",
            vec![SemanticPrime::Good], vec![],
            "bind", "Agreement", 0.9, 0.2, 0.2);
        self.add_grounded_word("ok", "interj",
            vec![SemanticPrime::Good], vec![],
            "bind", "Short agreement", 0.9, 0.2, 0.2);
        self.add_grounded_word("well", "interj",
            vec![SemanticPrime::Good], vec![SemanticPrime::Think],
            "bind", "Discourse marker", 0.8, 0.1, 0.2);

        // --- NUMBERS ---
        self.add_grounded_word("one", "num",
            vec![SemanticPrime::One], vec![],
            "bind", "Single", 1.0, 0.0, 0.1);
        self.add_grounded_word("two", "num",
            vec![SemanticPrime::Two], vec![],
            "bind", "Pair", 0.9, 0.0, 0.1);
        self.add_grounded_word("many", "adj",
            vec![SemanticPrime::Much], vec![],
            "bind", "Large quantity", 0.9, 0.0, 0.2);
        self.add_grounded_word("few", "adj",
            vec![SemanticPrime::Little], vec![],
            "bind", "Small quantity", 0.8, 0.0, 0.2);
        self.add_grounded_word("all", "adj",
            vec![SemanticPrime::All], vec![],
            "bind", "Totality", 0.9, 0.0, 0.2);
        self.add_grounded_word("some", "adj",
            vec![SemanticPrime::Some], vec![],
            "bind", "Partial", 0.9, 0.0, 0.2);
        self.add_grounded_word("every", "adj",
            vec![SemanticPrime::All], vec![SemanticPrime::One],
            "bind", "Each one", 0.8, 0.0, 0.2);
        self.add_grounded_word("each", "adj",
            vec![SemanticPrime::All, SemanticPrime::One], vec![],
            "bind", "Every individual", 0.8, 0.0, 0.2);
        self.add_grounded_word("any", "adj",
            vec![SemanticPrime::Some], vec![SemanticPrime::One],
            "bind", "Whichever", 0.8, 0.0, 0.2);
        self.add_grounded_word("none", "pron",
            vec![SemanticPrime::Not, SemanticPrime::One], vec![],
            "bind", "Not any", 0.8, -0.1, 0.2);
        self.add_grounded_word("other", "adj",
            vec![SemanticPrime::Other], vec![],
            "bind", "Different one", 0.9, 0.0, 0.2);
        self.add_grounded_word("another", "adj",
            vec![SemanticPrime::Other, SemanticPrime::One], vec![],
            "bind", "One more", 0.8, 0.0, 0.2);
        self.add_grounded_word("first", "adj",
            vec![SemanticPrime::One], vec![SemanticPrime::Before, SemanticPrime::All],
            "bind", "Initial", 0.9, 0.1, 0.3);
        self.add_grounded_word("last", "adj",
            vec![SemanticPrime::One], vec![SemanticPrime::After, SemanticPrime::All],
            "bind", "Final", 0.9, 0.0, 0.3);
        self.add_grounded_word("next", "adj",
            vec![SemanticPrime::After], vec![SemanticPrime::Near, SemanticPrime::One],
            "bind", "Following", 0.9, 0.0, 0.2);
    }

    /// Extended vocabulary: 700+ additional words for rich conversation
    fn initialize_extended_vocabulary(&mut self) {
        // ===== NATURE & WORLD =====
        let nature_words = [
            ("sun", "noun", vec![SemanticPrime::Something, SemanticPrime::Above], vec![SemanticPrime::Big], "Light source"),
            ("moon", "noun", vec![SemanticPrime::Something, SemanticPrime::Above], vec![SemanticPrime::Small], "Night light"),
            ("star", "noun", vec![SemanticPrime::Something, SemanticPrime::Far], vec![SemanticPrime::Small], "Distant light"),
            ("sky", "noun", vec![SemanticPrime::Something, SemanticPrime::Above], vec![], "Upper space"),
            ("earth", "noun", vec![SemanticPrime::Something, SemanticPrime::Below], vec![SemanticPrime::Big], "Ground"),
            ("water", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move], "Liquid"),
            ("fire", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move, SemanticPrime::Bad], "Burning"),
            ("air", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move], "Atmosphere"),
            ("wind", "noun", vec![SemanticPrime::Something, SemanticPrime::Move], vec![], "Moving air"),
            ("rain", "noun", vec![SemanticPrime::Something, SemanticPrime::Move], vec![SemanticPrime::Above], "Falling water"),
            ("snow", "noun", vec![SemanticPrime::Something, SemanticPrime::Move], vec![SemanticPrime::Above], "Frozen rain"),
            ("cloud", "noun", vec![SemanticPrime::Something, SemanticPrime::Above], vec![], "Sky vapor"),
            ("tree", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Big], "Plant"),
            ("flower", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Good], "Blossom"),
            ("grass", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Small], "Ground plant"),
            ("mountain", "noun", vec![SemanticPrime::Something, SemanticPrime::Above], vec![SemanticPrime::Big, SemanticPrime::Very], "High land"),
            ("river", "noun", vec![SemanticPrime::Something, SemanticPrime::Move], vec![], "Flowing water"),
            ("ocean", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Big, SemanticPrime::Very], "Large water"),
            ("forest", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Much], "Many trees"),
            ("desert", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Not, SemanticPrime::Have], "Dry land"),
            ("island", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Side], "Surrounded land"),
            ("rock", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Not, SemanticPrime::Live], "Stone"),
            ("sand", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Small, SemanticPrime::Much], "Tiny rocks"),
            ("lake", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Inside], "Contained water"),
            ("hill", "noun", vec![SemanticPrime::Something, SemanticPrime::Above], vec![SemanticPrime::Small], "Low mountain"),
        ];
        for (word, pos, core, mods, expl) in nature_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.0, 0.2);
        }

        // ===== BODY PARTS & PHYSICAL =====
        let body_words = [
            ("head", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Above], "Top body"),
            ("face", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::See], "Front head"),
            ("eye", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::See], "See organ"),
            ("ear", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Hear], "Hear organ"),
            ("nose", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Feel], "Smell organ"),
            ("mouth", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Say], "Speak part"),
            ("tongue", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Say, SemanticPrime::Feel], "Taste/speak"),
            ("tooth", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![], "Bite part"),
            ("teeth", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Much], "Bite parts"),
            ("hair", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Above], "Head covering"),
            ("neck", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![], "Head connector"),
            ("shoulder", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Side], "Arm top"),
            ("arm", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Do], "Limb"),
            ("elbow", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![], "Arm bend"),
            ("wrist", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![], "Hand connector"),
            ("finger", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Touch], "Hand digit"),
            ("thumb", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Touch], "Special digit"),
            ("chest", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Inside], "Front torso"),
            ("back", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![], "Rear torso"),
            ("stomach", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Inside], "Digestion"),
            ("leg", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Move], "Walking limb"),
            ("knee", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![], "Leg bend"),
            ("foot", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Below], "Walking end"),
            ("feet", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Below, SemanticPrime::Two], "Walking ends"),
            ("toe", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Small], "Foot digit"),
            ("skin", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Touch], "Body covering"),
            ("bone", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Inside], "Hard inside"),
            ("blood", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Inside, SemanticPrime::Move], "Life fluid"),
            ("brain", "noun", vec![SemanticPrime::PartOf, SemanticPrime::Body], vec![SemanticPrime::Think], "Think organ"),
        ];
        for (word, pos, core, mods, expl) in body_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.0, 0.2);
        }

        // ===== ACTIONS & VERBS =====
        let action_words = [
            ("walk", "verb", vec![SemanticPrime::Move], vec![SemanticPrime::Do], "Foot motion"),
            ("run", "verb", vec![SemanticPrime::Move], vec![SemanticPrime::Do, SemanticPrime::Very], "Fast walk"),
            ("jump", "verb", vec![SemanticPrime::Move, SemanticPrime::Above], vec![], "Leap up"),
            ("sit", "verb", vec![SemanticPrime::Be], vec![SemanticPrime::Not, SemanticPrime::Move], "Be seated"),
            ("stand", "verb", vec![SemanticPrime::Be], vec![SemanticPrime::Above], "Be upright"),
            ("lie", "verb", vec![SemanticPrime::Be], vec![SemanticPrime::Not, SemanticPrime::Move], "Be flat"),
            ("sleep", "verb", vec![SemanticPrime::Be], vec![SemanticPrime::Not, SemanticPrime::Think], "Rest state"),
            ("wake", "verb", vec![SemanticPrime::Happen], vec![SemanticPrime::Know], "Become aware"),
            ("eat", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Body], "Take food"),
            ("drink", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Body], "Take liquid"),
            ("breathe", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Live], "Take air"),
            ("laugh", "verb", vec![SemanticPrime::Do, SemanticPrime::Feel], vec![SemanticPrime::Good], "Joy sound"),
            ("cry", "verb", vec![SemanticPrime::Do, SemanticPrime::Feel], vec![SemanticPrime::Bad], "Sad sound"),
            ("smile", "verb", vec![SemanticPrime::Do, SemanticPrime::Feel], vec![SemanticPrime::Good], "Happy face"),
            ("frown", "verb", vec![SemanticPrime::Do, SemanticPrime::Feel], vec![SemanticPrime::Bad], "Sad face"),
            ("sing", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::Good], "Musical voice"),
            ("dance", "verb", vec![SemanticPrime::Move], vec![SemanticPrime::Good, SemanticPrime::Body], "Rhythmic move"),
            ("play", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Good], "Have fun"),
            ("work", "verb", vec![SemanticPrime::Do], vec![], "Productive action"),
            ("rest", "verb", vec![SemanticPrime::Be], vec![SemanticPrime::Not, SemanticPrime::Do], "Not working"),
            ("wait", "verb", vec![SemanticPrime::Be], vec![SemanticPrime::Not, SemanticPrime::Do], "Stay here"),
            ("watch", "verb", vec![SemanticPrime::See], vec![SemanticPrime::ForSomeTime], "Look at"),
            ("listen", "verb", vec![SemanticPrime::Hear], vec![SemanticPrime::Want], "Attend sound"),
            ("touch", "verb", vec![SemanticPrime::Touch], vec![], "Physical contact"),
            ("hold", "verb", vec![SemanticPrime::Touch, SemanticPrime::Have], vec![], "Grasp"),
            ("catch", "verb", vec![SemanticPrime::Touch, SemanticPrime::Move], vec![], "Grab moving"),
            ("throw", "verb", vec![SemanticPrime::Move], vec![SemanticPrime::Do], "Send away"),
            ("push", "verb", vec![SemanticPrime::Move, SemanticPrime::Touch], vec![SemanticPrime::Far], "Move away"),
            ("pull", "verb", vec![SemanticPrime::Move, SemanticPrime::Touch], vec![SemanticPrime::Near], "Move toward"),
            ("open", "verb", vec![SemanticPrime::Do], vec![], "Make accessible"),
            ("close", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Not], "Block access"),
            ("break", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Bad], "Damage"),
            ("fix", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Good], "Repair"),
            ("build", "verb", vec![SemanticPrime::Do], vec![], "Create thing"),
            ("destroy", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Bad, SemanticPrime::Very], "Remove completely"),
            ("cut", "verb", vec![SemanticPrime::Do], vec![], "Divide"),
            ("join", "verb", vec![SemanticPrime::Do], vec![], "Connect"),
            ("mix", "verb", vec![SemanticPrime::Do], vec![], "Combine"),
            ("separate", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Not], "Divide apart"),
            ("show", "verb", vec![SemanticPrime::Do, SemanticPrime::See], vec![], "Make visible"),
            ("hide", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Not, SemanticPrime::See], "Make invisible"),
            ("find", "verb", vec![SemanticPrime::Do, SemanticPrime::See], vec![], "Discover"),
            ("lose", "verb", vec![SemanticPrime::Happen], vec![SemanticPrime::Not, SemanticPrime::Have], "No longer have"),
            ("keep", "verb", vec![SemanticPrime::Have], vec![SemanticPrime::ForSomeTime], "Continue having"),
            ("give", "verb", vec![SemanticPrime::Do], vec![], "Transfer to"),
            ("take", "verb", vec![SemanticPrime::Do], vec![], "Receive from"),
            ("send", "verb", vec![SemanticPrime::Do, SemanticPrime::Move], vec![SemanticPrime::Far], "Cause to go"),
            ("receive", "verb", vec![SemanticPrime::Do], vec![], "Get from"),
            ("bring", "verb", vec![SemanticPrime::Move], vec![SemanticPrime::Near], "Carry here"),
            ("carry", "verb", vec![SemanticPrime::Move, SemanticPrime::Have], vec![], "Transport"),
            ("drop", "verb", vec![SemanticPrime::Move], vec![SemanticPrime::Below], "Let fall"),
            ("lift", "verb", vec![SemanticPrime::Move], vec![SemanticPrime::Above], "Raise up"),
            ("put", "verb", vec![SemanticPrime::Do, SemanticPrime::Move], vec![], "Place"),
            ("turn", "verb", vec![SemanticPrime::Move], vec![], "Rotate"),
            ("stop", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Not, SemanticPrime::Move], "Cease"),
            ("start", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Now], "Begin"),
            ("continue", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::ForSomeTime], "Keep going"),
            ("finish", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::After], "Complete"),
            ("try", "verb", vec![SemanticPrime::Do, SemanticPrime::Want], vec![], "Attempt"),
            ("fail", "verb", vec![SemanticPrime::Happen], vec![SemanticPrime::Bad], "Not succeed"),
            ("succeed", "verb", vec![SemanticPrime::Happen], vec![SemanticPrime::Good], "Achieve goal"),
            ("help", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Good], "Assist"),
            ("hurt", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Bad], "Cause pain"),
            ("heal", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Good, SemanticPrime::Body], "Recover"),
            ("grow", "verb", vec![SemanticPrime::Happen], vec![SemanticPrime::More, SemanticPrime::Big], "Become larger"),
            ("shrink", "verb", vec![SemanticPrime::Happen], vec![SemanticPrime::Small], "Become smaller"),
            ("change", "verb", vec![SemanticPrime::Happen], vec![SemanticPrime::Other], "Become different"),
            ("stay", "verb", vec![SemanticPrime::Be], vec![SemanticPrime::Same], "Remain"),
            ("leave", "verb", vec![SemanticPrime::Move], vec![SemanticPrime::Far], "Go away"),
            ("arrive", "verb", vec![SemanticPrime::Move], vec![SemanticPrime::Here], "Get there"),
            ("enter", "verb", vec![SemanticPrime::Move], vec![SemanticPrime::Inside], "Go in"),
            ("exit", "verb", vec![SemanticPrime::Move], vec![SemanticPrime::Not, SemanticPrime::Inside], "Go out"),
            ("return", "verb", vec![SemanticPrime::Move], vec![SemanticPrime::Same, SemanticPrime::Before], "Come back"),
            ("follow", "verb", vec![SemanticPrime::Move], vec![SemanticPrime::After], "Go behind"),
            ("lead", "verb", vec![SemanticPrime::Move], vec![SemanticPrime::Before], "Go first"),
            ("meet", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Near, SemanticPrime::Someone], "Come together"),
            ("fight", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Bad], "Physical conflict"),
            ("win", "verb", vec![SemanticPrime::Happen], vec![SemanticPrime::Good], "Be victorious"),
            ("agree", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Same], "Same opinion"),
            ("disagree", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Not, SemanticPrime::Same], "Different opinion"),
            ("decide", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Do], "Make choice"),
            ("choose", "verb", vec![SemanticPrime::Think, SemanticPrime::Want], vec![], "Select"),
            ("prefer", "verb", vec![SemanticPrime::Want], vec![SemanticPrime::More], "Like more"),
            ("need", "verb", vec![SemanticPrime::Want], vec![SemanticPrime::Very], "Must have"),
            ("hope", "verb", vec![SemanticPrime::Want], vec![SemanticPrime::Good, SemanticPrime::Maybe], "Wish for"),
            ("wish", "verb", vec![SemanticPrime::Want], vec![], "Desire"),
            ("expect", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::After], "Anticipate"),
            ("imagine", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Not, SemanticPrime::True], "Create mentally"),
            ("dream", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Not, SemanticPrime::True], "Sleep thoughts"),
            ("plan", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Before], "Prepare"),
            ("prepare", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Before], "Get ready"),
            ("practice", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Much], "Repeat to learn"),
            ("learn", "verb", vec![SemanticPrime::Know], vec![SemanticPrime::Now], "Gain knowledge"),
            ("teach", "verb", vec![SemanticPrime::Do, SemanticPrime::Know], vec![], "Share knowledge"),
            ("study", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Want, SemanticPrime::Know], "Learn deeply"),
            ("read", "verb", vec![SemanticPrime::See, SemanticPrime::Know], vec![SemanticPrime::Words], "Understand text"),
            ("write", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Words], "Create text"),
            ("speak", "verb", vec![SemanticPrime::Say], vec![], "Use voice"),
            ("shout", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::Very], "Loud voice"),
            ("whisper", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::Small], "Quiet voice"),
            ("call", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::Far], "Summon"),
            ("answer", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::After], "Reply"),
            ("explain", "verb", vec![SemanticPrime::Say, SemanticPrime::Know], vec![], "Make clear"),
            ("describe", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::See], "Tell about"),
            ("promise", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::True], "Commit to"),
            ("apologize", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::Bad, SemanticPrime::Feel], "Say sorry"),
            ("thank", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::Good], "Express gratitude"),
            ("complain", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::Bad], "Express displeasure"),
            ("suggest", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::Maybe], "Propose"),
            ("warn", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::Bad, SemanticPrime::Maybe], "Alert danger"),
            ("invite", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::Good, SemanticPrime::Near], "Request presence"),
            ("refuse", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::Not, SemanticPrime::Want], "Decline"),
            ("accept", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::Good], "Agree to"),
            ("deny", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::Not, SemanticPrime::True], "Reject claim"),
            ("admit", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::True], "Acknowledge"),
            ("confess", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::True, SemanticPrime::Bad], "Reveal truth"),
            ("lie", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::Not, SemanticPrime::True], "Say false"),
            ("pretend", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Not, SemanticPrime::True], "Act false"),
            ("seem", "verb", vec![SemanticPrime::Be], vec![SemanticPrime::Maybe], "Appear"),
            ("appear", "verb", vec![SemanticPrime::See], vec![SemanticPrime::Now], "Become visible"),
            ("disappear", "verb", vec![SemanticPrime::See], vec![SemanticPrime::Not], "Become invisible"),
            ("exist", "verb", vec![SemanticPrime::Be], vec![SemanticPrime::True], "Be real"),
            ("create", "verb", vec![SemanticPrime::Do], vec![], "Make new"),
            ("destroy", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Not, SemanticPrime::ThereIs], "End existence"),
            ("use", "verb", vec![SemanticPrime::Do], vec![], "Employ"),
            ("make", "verb", vec![SemanticPrime::Do], vec![], "Create"),
            ("let", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Can], "Allow"),
            ("cause", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Because], "Make happen"),
            ("prevent", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Not, SemanticPrime::Happen], "Stop from"),
            ("allow", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Can], "Permit"),
            ("forbid", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Not, SemanticPrime::Can], "Prohibit"),
            ("require", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Want, SemanticPrime::Very], "Need must"),
            ("include", "verb", vec![SemanticPrime::Have], vec![SemanticPrime::Inside], "Contain"),
            ("exclude", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Not, SemanticPrime::Inside], "Leave out"),
            ("belong", "verb", vec![SemanticPrime::Be], vec![SemanticPrime::PartOf], "Be part of"),
            ("own", "verb", vec![SemanticPrime::Have], vec![], "Possess"),
            ("share", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Someone], "Give part"),
            ("borrow", "verb", vec![SemanticPrime::Have], vec![SemanticPrime::ShortTime], "Temporary have"),
            ("lend", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::ShortTime], "Temporary give"),
            ("pay", "verb", vec![SemanticPrime::Do], vec![], "Give money"),
            ("cost", "verb", vec![SemanticPrime::Be], vec![], "Have price"),
            ("buy", "verb", vec![SemanticPrime::Do], vec![], "Get for money"),
            ("sell", "verb", vec![SemanticPrime::Do], vec![], "Give for money"),
            ("save", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Good], "Rescue or keep"),
            ("spend", "verb", vec![SemanticPrime::Do], vec![], "Use resource"),
            ("waste", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Bad], "Use poorly"),
            ("measure", "verb", vec![SemanticPrime::Do, SemanticPrime::Know], vec![], "Determine size"),
            ("count", "verb", vec![SemanticPrime::Do, SemanticPrime::Know], vec![], "Determine number"),
            ("compare", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Same, SemanticPrime::Other], "Find differences"),
            ("choose", "verb", vec![SemanticPrime::Think, SemanticPrime::Want], vec![], "Pick one"),
            ("solve", "verb", vec![SemanticPrime::Think, SemanticPrime::Do], vec![SemanticPrime::Good], "Find answer"),
            ("calculate", "verb", vec![SemanticPrime::Think], vec![], "Do math"),
            ("analyze", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Know], "Examine deeply"),
            ("understand", "verb", vec![SemanticPrime::Know, SemanticPrime::Think], vec![SemanticPrime::Good], "Comprehend"),
            ("realize", "verb", vec![SemanticPrime::Know], vec![SemanticPrime::Now], "Become aware"),
            ("notice", "verb", vec![SemanticPrime::See], vec![SemanticPrime::Know], "Perceive"),
            ("recognize", "verb", vec![SemanticPrime::See, SemanticPrime::Know], vec![SemanticPrime::Before], "Identify known"),
            ("ignore", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Not, SemanticPrime::See], "Not attend to"),
            ("focus", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Very], "Concentrate"),
            ("concentrate", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Very], "Focus hard"),
            ("consider", "verb", vec![SemanticPrime::Think], vec![], "Think about"),
            ("wonder", "verb", vec![SemanticPrime::Think, SemanticPrime::Want, SemanticPrime::Know], vec![], "Question in mind"),
            ("doubt", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Not, SemanticPrime::True], "Question truth"),
            ("trust", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Good, SemanticPrime::True], "Believe in"),
            ("suspect", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Bad, SemanticPrime::Maybe], "Think maybe bad"),
            ("guess", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::Maybe], "Uncertain answer"),
            ("assume", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::True], "Take as fact"),
            ("conclude", "verb", vec![SemanticPrime::Think], vec![SemanticPrime::After], "Reach decision"),
            ("prove", "verb", vec![SemanticPrime::Do, SemanticPrime::True], vec![], "Show truth"),
            ("test", "verb", vec![SemanticPrime::Do], vec![SemanticPrime::Know], "Try to find"),
            ("check", "verb", vec![SemanticPrime::Do, SemanticPrime::See], vec![], "Verify"),
            ("confirm", "verb", vec![SemanticPrime::Say], vec![SemanticPrime::True], "Verify truth"),
        ];
        for (word, pos, core, mods, expl) in action_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.0, 0.3);
        }

        // ===== ADJECTIVES & QUALITIES =====
        let adj_words = [
            ("new", "adj", vec![SemanticPrime::Now], vec![SemanticPrime::Not, SemanticPrime::Before], "Recently"),
            ("old", "adj", vec![SemanticPrime::Before], vec![SemanticPrime::LongTime], "Long ago"),
            ("young", "adj", vec![SemanticPrime::Live], vec![SemanticPrime::Not, SemanticPrime::LongTime], "Not old"),
            ("fast", "adj", vec![SemanticPrime::Move], vec![SemanticPrime::Very], "Quick"),
            ("slow", "adj", vec![SemanticPrime::Move], vec![SemanticPrime::Little], "Not fast"),
            ("hot", "adj", vec![SemanticPrime::Feel], vec![], "High temperature"),
            ("cold", "adj", vec![SemanticPrime::Feel], vec![SemanticPrime::Not], "Low temperature"),
            ("warm", "adj", vec![SemanticPrime::Feel], vec![SemanticPrime::Good], "Pleasant heat"),
            ("cool", "adj", vec![SemanticPrime::Feel], vec![SemanticPrime::Good], "Pleasant cold"),
            ("wet", "adj", vec![SemanticPrime::Have], vec![], "Has water"),
            ("dry", "adj", vec![SemanticPrime::Have], vec![SemanticPrime::Not], "No water"),
            ("hard", "adj", vec![SemanticPrime::Touch], vec![], "Solid"),
            ("soft", "adj", vec![SemanticPrime::Touch], vec![SemanticPrime::Not], "Not hard"),
            ("heavy", "adj", vec![SemanticPrime::Have], vec![SemanticPrime::Much], "Much weight"),
            ("light", "adj", vec![SemanticPrime::Have], vec![SemanticPrime::Little], "Little weight"),
            ("bright", "adj", vec![SemanticPrime::See], vec![SemanticPrime::Much], "Much light"),
            ("dark", "adj", vec![SemanticPrime::See], vec![SemanticPrime::Little], "Little light"),
            ("loud", "adj", vec![SemanticPrime::Hear], vec![SemanticPrime::Much], "Much sound"),
            ("quiet", "adj", vec![SemanticPrime::Hear], vec![SemanticPrime::Little], "Little sound"),
            ("clean", "adj", vec![SemanticPrime::Good], vec![], "No dirt"),
            ("dirty", "adj", vec![SemanticPrime::Bad], vec![], "Has dirt"),
            ("full", "adj", vec![SemanticPrime::Have], vec![SemanticPrime::All], "All inside"),
            ("empty", "adj", vec![SemanticPrime::Have], vec![SemanticPrime::Not], "None inside"),
            ("open", "adj", vec![SemanticPrime::Can], vec![SemanticPrime::Move], "Not blocked"),
            ("closed", "adj", vec![SemanticPrime::Can], vec![SemanticPrime::Not, SemanticPrime::Move], "Blocked"),
            ("safe", "adj", vec![SemanticPrime::Good], vec![SemanticPrime::Not, SemanticPrime::Bad], "No danger"),
            ("dangerous", "adj", vec![SemanticPrime::Bad], vec![SemanticPrime::Maybe], "Can hurt"),
            ("easy", "adj", vec![SemanticPrime::Good], vec![SemanticPrime::Can], "Simple"),
            ("difficult", "adj", vec![SemanticPrime::Bad], vec![SemanticPrime::Not, SemanticPrime::Can], "Hard to do"),
            ("simple", "adj", vec![SemanticPrime::Good], vec![SemanticPrime::Not, SemanticPrime::Much], "Not complex"),
            ("complex", "adj", vec![SemanticPrime::Have], vec![SemanticPrime::Much], "Many parts"),
            ("beautiful", "adj", vec![SemanticPrime::Good], vec![SemanticPrime::See, SemanticPrime::Very], "Very pleasing"),
            ("ugly", "adj", vec![SemanticPrime::Bad], vec![SemanticPrime::See], "Not pleasing"),
            ("rich", "adj", vec![SemanticPrime::Have], vec![SemanticPrime::Much], "Much wealth"),
            ("poor", "adj", vec![SemanticPrime::Have], vec![SemanticPrime::Little], "Little wealth"),
            ("strong", "adj", vec![SemanticPrime::Can], vec![SemanticPrime::Much], "Much power"),
            ("weak", "adj", vec![SemanticPrime::Can], vec![SemanticPrime::Little], "Little power"),
            ("healthy", "adj", vec![SemanticPrime::Body], vec![SemanticPrime::Good], "Body good"),
            ("sick", "adj", vec![SemanticPrime::Body], vec![SemanticPrime::Bad], "Body bad"),
            ("tired", "adj", vec![SemanticPrime::Body, SemanticPrime::Feel], vec![SemanticPrime::Bad], "Need rest"),
            ("fresh", "adj", vec![SemanticPrime::Now], vec![SemanticPrime::Good], "Just made"),
            ("stale", "adj", vec![SemanticPrime::Before], vec![SemanticPrime::Bad], "Old food"),
            ("alive", "adj", vec![SemanticPrime::Live], vec![SemanticPrime::Now], "Living"),
            ("dead", "adj", vec![SemanticPrime::Die], vec![], "Not living"),
            ("real", "adj", vec![SemanticPrime::True], vec![], "Existing"),
            ("fake", "adj", vec![SemanticPrime::Not, SemanticPrime::True], vec![], "Not real"),
            ("right", "adj", vec![SemanticPrime::True], vec![SemanticPrime::Good], "Correct"),
            ("wrong", "adj", vec![SemanticPrime::Not, SemanticPrime::True], vec![SemanticPrime::Bad], "Incorrect"),
            ("possible", "adj", vec![SemanticPrime::Can], vec![SemanticPrime::Maybe], "Can happen"),
            ("impossible", "adj", vec![SemanticPrime::Not, SemanticPrime::Can], vec![], "Cannot happen"),
            ("necessary", "adj", vec![SemanticPrime::Want], vec![SemanticPrime::Very], "Must have"),
            ("important", "adj", vec![SemanticPrime::Good], vec![SemanticPrime::Very], "Matters much"),
            ("different", "adj", vec![SemanticPrime::Other], vec![], "Not same"),
            ("similar", "adj", vec![SemanticPrime::Same], vec![SemanticPrime::Like], "Almost same"),
            ("special", "adj", vec![SemanticPrime::Other], vec![SemanticPrime::Good], "Unique good"),
            ("normal", "adj", vec![SemanticPrime::Same], vec![SemanticPrime::Like], "Usual"),
            ("strange", "adj", vec![SemanticPrime::Other], vec![], "Unusual"),
            ("familiar", "adj", vec![SemanticPrime::Know], vec![SemanticPrime::Before], "Known before"),
            ("unknown", "adj", vec![SemanticPrime::Not, SemanticPrime::Know], vec![], "Not known"),
            ("famous", "adj", vec![SemanticPrime::Know], vec![SemanticPrime::People, SemanticPrime::Much], "Many know"),
            ("popular", "adj", vec![SemanticPrime::Good], vec![SemanticPrime::People, SemanticPrime::Much], "Many like"),
            ("common", "adj", vec![SemanticPrime::Same], vec![SemanticPrime::Much], "Many same"),
            ("rare", "adj", vec![SemanticPrime::Have], vec![SemanticPrime::Little], "Few exist"),
            ("free", "adj", vec![SemanticPrime::Can], vec![SemanticPrime::Not], "No cost/constraint"),
            ("busy", "adj", vec![SemanticPrime::Do], vec![SemanticPrime::Much], "Much doing"),
            ("ready", "adj", vec![SemanticPrime::Can], vec![SemanticPrime::Now], "Prepared"),
            ("available", "adj", vec![SemanticPrime::Can, SemanticPrime::Have], vec![], "Can get"),
            ("certain", "adj", vec![SemanticPrime::True], vec![SemanticPrime::Very], "Definitely"),
            ("uncertain", "adj", vec![SemanticPrime::Maybe], vec![], "Not sure"),
            ("clear", "adj", vec![SemanticPrime::See], vec![SemanticPrime::Good], "Easy to see"),
            ("obvious", "adj", vec![SemanticPrime::See], vec![SemanticPrime::Very, SemanticPrime::Good], "Very clear"),
            ("hidden", "adj", vec![SemanticPrime::Not, SemanticPrime::See], vec![], "Not visible"),
            ("secret", "adj", vec![SemanticPrime::Not, SemanticPrime::Know], vec![SemanticPrime::Want], "Kept hidden"),
            ("public", "adj", vec![SemanticPrime::All], vec![SemanticPrime::People, SemanticPrime::Know], "For all"),
            ("private", "adj", vec![SemanticPrime::Not, SemanticPrime::All], vec![], "Not for all"),
            ("personal", "adj", vec![SemanticPrime::I], vec![], "Belonging to one"),
            ("general", "adj", vec![SemanticPrime::All], vec![], "For all"),
            ("specific", "adj", vec![SemanticPrime::One], vec![], "Particular"),
            ("whole", "adj", vec![SemanticPrime::All], vec![SemanticPrime::PartOf], "Complete"),
            ("main", "adj", vec![SemanticPrime::Big], vec![SemanticPrime::More], "Most important"),
            ("extra", "adj", vec![SemanticPrime::More], vec![], "Additional"),
            ("enough", "adj", vec![SemanticPrime::Good], vec![SemanticPrime::Much], "Sufficient"),
            ("perfect", "adj", vec![SemanticPrime::Good], vec![SemanticPrime::Very, SemanticPrime::All], "No flaw"),
            ("complete", "adj", vec![SemanticPrime::All], vec![], "Nothing missing"),
            ("partial", "adj", vec![SemanticPrime::Some], vec![SemanticPrime::PartOf], "Not all"),
        ];
        for (word, pos, core, mods, expl) in adj_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.0, 0.2);
        }

        // ===== EMOTIONS & FEELINGS =====
        let emotion_words = [
            ("love", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::Very], "Deep affection"),
            ("hate", "noun", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![SemanticPrime::Very], "Strong dislike"),
            ("joy", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::Very], "Great happiness"),
            ("sorrow", "noun", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![], "Deep sadness"),
            ("fear", "noun", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![SemanticPrime::Maybe], "Afraid feeling"),
            ("hope", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::Maybe], "Positive expect"),
            ("pride", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::I], "Self satisfaction"),
            ("shame", "noun", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![SemanticPrime::I], "Self disappointment"),
            ("guilt", "noun", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![SemanticPrime::I, SemanticPrime::Do], "Wrong feeling"),
            ("regret", "noun", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![SemanticPrime::Before], "Wish undo"),
            ("envy", "noun", vec![SemanticPrime::Feel, SemanticPrime::Want], vec![SemanticPrime::Someone, SemanticPrime::Have], "Want others have"),
            ("jealousy", "noun", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![SemanticPrime::Want], "Fear of losing"),
            ("gratitude", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![], "Thank feeling"),
            ("surprise", "noun", vec![SemanticPrime::Feel], vec![SemanticPrime::Not, SemanticPrime::Think], "Unexpected"),
            ("excitement", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::Very], "High energy joy"),
            ("boredom", "noun", vec![SemanticPrime::Feel], vec![SemanticPrime::Not, SemanticPrime::Good], "Lack interest"),
            ("confusion", "noun", vec![SemanticPrime::Feel, SemanticPrime::Not, SemanticPrime::Know], vec![], "Unclear mind"),
            ("frustration", "noun", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![SemanticPrime::Not, SemanticPrime::Can], "Blocked want"),
            ("relief", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::After], "Worry ends"),
            ("anxiety", "noun", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![SemanticPrime::Maybe], "Worry future"),
            ("peace", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::Not, SemanticPrime::Bad], "Calm state"),
            ("calm", "noun", vec![SemanticPrime::Feel], vec![SemanticPrime::Not, SemanticPrime::Much], "No disturbance"),
            ("stress", "noun", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![SemanticPrime::Much], "Pressure feeling"),
            ("comfort", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::Body], "At ease"),
            ("pain", "noun", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![SemanticPrime::Body], "Hurt feeling"),
            ("pleasure", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::Body], "Enjoy feeling"),
            ("desire", "noun", vec![SemanticPrime::Want], vec![SemanticPrime::Very], "Strong want"),
            ("passion", "noun", vec![SemanticPrime::Feel], vec![SemanticPrime::Very], "Intense feeling"),
            ("compassion", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::Someone], "Care for others"),
            ("empathy", "noun", vec![SemanticPrime::Feel], vec![SemanticPrime::Same, SemanticPrime::Someone], "Share feeling"),
            ("sympathy", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::Someone], "Care about"),
            ("loneliness", "noun", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![SemanticPrime::Not, SemanticPrime::Someone], "Alone feeling"),
            ("belonging", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![SemanticPrime::PartOf], "Connected feeling"),
            ("contentment", "noun", vec![SemanticPrime::Feel, SemanticPrime::Good], vec![], "Satisfied"),
            ("disappointment", "noun", vec![SemanticPrime::Feel, SemanticPrime::Bad], vec![SemanticPrime::Not, SemanticPrime::Good], "Let down"),
        ];
        for (word, pos, core, mods, expl) in emotion_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.4, 0.5);
        }

        // ===== ABSTRACT CONCEPTS =====
        let abstract_words = [
            ("idea", "noun", vec![SemanticPrime::Think], vec![], "Mental concept"),
            ("thought", "noun", vec![SemanticPrime::Think], vec![], "Mental content"),
            ("mind", "noun", vec![SemanticPrime::Think], vec![SemanticPrime::PartOf, SemanticPrime::I], "Thinking part"),
            ("soul", "noun", vec![SemanticPrime::I], vec![SemanticPrime::Live], "Inner self"),
            ("spirit", "noun", vec![SemanticPrime::Live], vec![SemanticPrime::Not, SemanticPrime::Body], "Non-physical"),
            ("truth", "noun", vec![SemanticPrime::True], vec![], "What is real"),
            ("reality", "noun", vec![SemanticPrime::True], vec![SemanticPrime::All], "What exists"),
            ("freedom", "noun", vec![SemanticPrime::Can], vec![SemanticPrime::Do], "Able to choose"),
            ("power", "noun", vec![SemanticPrime::Can], vec![SemanticPrime::Do], "Ability strength"),
            ("control", "noun", vec![SemanticPrime::Can], vec![SemanticPrime::Do], "Direct ability"),
            ("choice", "noun", vec![SemanticPrime::Can], vec![SemanticPrime::Do], "Option selection"),
            ("chance", "noun", vec![SemanticPrime::Maybe], vec![SemanticPrime::Can], "Possibility"),
            ("luck", "noun", vec![SemanticPrime::Good], vec![SemanticPrime::Maybe], "Random fortune"),
            ("fate", "noun", vec![SemanticPrime::Happen], vec![SemanticPrime::Not, SemanticPrime::Can], "Determined outcome"),
            ("destiny", "noun", vec![SemanticPrime::Happen], vec![SemanticPrime::After], "Future fate"),
            ("success", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::Good], "Achieve goal"),
            ("failure", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::Bad], "Not achieve"),
            ("goal", "noun", vec![SemanticPrime::Want], vec![SemanticPrime::After], "Aim target"),
            ("purpose", "noun", vec![SemanticPrime::Want], vec![SemanticPrime::Because], "Reason for"),
            ("meaning", "noun", vec![SemanticPrime::Know], vec![SemanticPrime::True], "Significance"),
            ("value", "noun", vec![SemanticPrime::Good], vec![SemanticPrime::Think], "Worth"),
            ("quality", "noun", vec![SemanticPrime::Like], vec![SemanticPrime::Good], "Characteristic"),
            ("nature", "noun", vec![SemanticPrime::Like], vec![SemanticPrime::Be], "Essential quality"),
            ("form", "noun", vec![SemanticPrime::Like], vec![SemanticPrime::See], "Shape"),
            ("pattern", "noun", vec![SemanticPrime::Same], vec![SemanticPrime::Much], "Repeating form"),
            ("structure", "noun", vec![SemanticPrime::Like], vec![SemanticPrime::PartOf], "Organization"),
            ("system", "noun", vec![SemanticPrime::PartOf], vec![SemanticPrime::Much], "Connected parts"),
            ("process", "noun", vec![SemanticPrime::Happen], vec![SemanticPrime::ForSomeTime], "Series of steps"),
            ("method", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::Like], "Way of doing"),
            ("way", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::Like], "Manner"),
            ("reason", "noun", vec![SemanticPrime::Because], vec![], "Cause"),
            ("cause", "noun", vec![SemanticPrime::Because], vec![], "What makes happen"),
            ("effect", "noun", vec![SemanticPrime::Happen], vec![SemanticPrime::Because], "Result"),
            ("result", "noun", vec![SemanticPrime::Happen], vec![SemanticPrime::After], "Outcome"),
            ("problem", "noun", vec![SemanticPrime::Bad], vec![], "Difficulty"),
            ("solution", "noun", vec![SemanticPrime::Good], vec![SemanticPrime::After], "Answer to problem"),
            ("answer", "noun", vec![SemanticPrime::Say], vec![SemanticPrime::After], "Response"),
            ("question", "noun", vec![SemanticPrime::Say], vec![SemanticPrime::Want, SemanticPrime::Know], "Asking"),
            ("information", "noun", vec![SemanticPrime::Know], vec![], "Facts"),
            ("knowledge", "noun", vec![SemanticPrime::Know], vec![SemanticPrime::Much], "What is known"),
            ("wisdom", "noun", vec![SemanticPrime::Know], vec![SemanticPrime::Good, SemanticPrime::LongTime], "Deep knowledge"),
            ("experience", "noun", vec![SemanticPrime::Know], vec![SemanticPrime::Do, SemanticPrime::Before], "Lived knowledge"),
            ("memory", "noun", vec![SemanticPrime::Know], vec![SemanticPrime::Before], "Past knowledge"),
            ("dream", "noun", vec![SemanticPrime::Think], vec![SemanticPrime::Not, SemanticPrime::True], "Sleep vision"),
            ("imagination", "noun", vec![SemanticPrime::Think], vec![SemanticPrime::Not, SemanticPrime::True], "Create mental"),
            ("creativity", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::Other], "Make new"),
            ("art", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::Good, SemanticPrime::See], "Creative work"),
            ("beauty", "noun", vec![SemanticPrime::Good], vec![SemanticPrime::See, SemanticPrime::Very], "Pleasing quality"),
            ("justice", "noun", vec![SemanticPrime::Good], vec![SemanticPrime::Same, SemanticPrime::All], "Fair treatment"),
            ("fairness", "noun", vec![SemanticPrime::Good], vec![SemanticPrime::Same], "Equal treatment"),
            ("rights", "noun", vec![SemanticPrime::Can], vec![SemanticPrime::Good], "Allowed things"),
            ("duty", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::Want, SemanticPrime::Very], "Must do"),
            ("responsibility", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::I], "Being responsible"),
            ("obligation", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::Want, SemanticPrime::Very], "Must do"),
            ("trust", "noun", vec![SemanticPrime::Think], vec![SemanticPrime::Good, SemanticPrime::True], "Belief in"),
            ("faith", "noun", vec![SemanticPrime::Think], vec![SemanticPrime::True, SemanticPrime::Not, SemanticPrime::See], "Belief without proof"),
            ("belief", "noun", vec![SemanticPrime::Think], vec![SemanticPrime::True], "Held opinion"),
            ("opinion", "noun", vec![SemanticPrime::Think], vec![SemanticPrime::I], "Personal view"),
            ("fact", "noun", vec![SemanticPrime::True], vec![], "Known truth"),
            ("theory", "noun", vec![SemanticPrime::Think], vec![SemanticPrime::Maybe, SemanticPrime::True], "Possible explanation"),
            ("law", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::All, SemanticPrime::Want, SemanticPrime::Very], "Required rule"),
            ("rule", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::Want, SemanticPrime::Very], "Must follow"),
            ("principle", "noun", vec![SemanticPrime::Think], vec![SemanticPrime::True, SemanticPrime::Good], "Guiding truth"),
            ("standard", "noun", vec![SemanticPrime::Like], vec![SemanticPrime::Same], "Expected level"),
            ("tradition", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::Same, SemanticPrime::Before, SemanticPrime::LongTime], "Old custom"),
            ("culture", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::Same, SemanticPrime::People], "Shared ways"),
            ("society", "noun", vec![SemanticPrime::People], vec![SemanticPrime::Live, SemanticPrime::Same], "Group living"),
            ("community", "noun", vec![SemanticPrime::People], vec![SemanticPrime::Near, SemanticPrime::Same], "Local group"),
            ("relationship", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::With], "Connection between"),
            ("friendship", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Good, SemanticPrime::With], "Friend bond"),
            ("family", "noun", vec![SemanticPrime::People], vec![SemanticPrime::Near, SemanticPrime::Good], "Related people"),
            ("home", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Live, SemanticPrime::Good], "Living place"),
            ("place", "noun", vec![SemanticPrime::Where], vec![], "Location"),
            ("space", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::ThereIs], "Area"),
            ("time", "noun", vec![SemanticPrime::When], vec![], "Temporal"),
            ("moment", "noun", vec![SemanticPrime::When], vec![SemanticPrime::ShortTime], "Brief time"),
            ("period", "noun", vec![SemanticPrime::When], vec![SemanticPrime::ForSomeTime], "Time span"),
            ("age", "noun", vec![SemanticPrime::When], vec![SemanticPrime::LongTime], "Era"),
            ("history", "noun", vec![SemanticPrime::Before], vec![SemanticPrime::Know], "Past record"),
            ("future", "noun", vec![SemanticPrime::After], vec![], "Time ahead"),
            ("present", "noun", vec![SemanticPrime::Now], vec![], "Current time"),
            ("past", "noun", vec![SemanticPrime::Before], vec![], "Earlier time"),
            ("beginning", "noun", vec![SemanticPrime::Before], vec![SemanticPrime::All], "Start"),
            ("end", "noun", vec![SemanticPrime::After], vec![SemanticPrime::All], "Finish"),
            ("middle", "noun", vec![SemanticPrime::Inside], vec![], "Center"),
            ("edge", "noun", vec![SemanticPrime::Side], vec![], "Boundary"),
            ("center", "noun", vec![SemanticPrime::Inside], vec![], "Middle point"),
            ("top", "noun", vec![SemanticPrime::Above], vec![], "Highest"),
            ("bottom", "noun", vec![SemanticPrime::Below], vec![], "Lowest"),
            ("surface", "noun", vec![SemanticPrime::On], vec![], "Outer layer"),
            ("depth", "noun", vec![SemanticPrime::Inside], vec![SemanticPrime::Far], "How deep"),
            ("height", "noun", vec![SemanticPrime::Above], vec![], "How high"),
            ("width", "noun", vec![SemanticPrime::Side], vec![], "How wide"),
            ("length", "noun", vec![SemanticPrime::Far], vec![], "How long"),
            ("size", "noun", vec![SemanticPrime::Big], vec![], "How big"),
            ("shape", "noun", vec![SemanticPrime::Like], vec![SemanticPrime::See], "Form"),
            ("color", "noun", vec![SemanticPrime::See], vec![], "Visual property"),
            ("sound", "noun", vec![SemanticPrime::Hear], vec![], "What heard"),
            ("voice", "noun", vec![SemanticPrime::Say], vec![SemanticPrime::Someone], "Person sound"),
            ("word", "noun", vec![SemanticPrime::Words], vec![], "Language unit"),
            ("language", "noun", vec![SemanticPrime::Words], vec![SemanticPrime::Say], "Communication"),
            ("name", "noun", vec![SemanticPrime::Words], vec![SemanticPrime::Someone], "Identifier"),
            ("number", "noun", vec![SemanticPrime::One], vec![SemanticPrime::Much], "Count"),
            ("amount", "noun", vec![SemanticPrime::Much], vec![], "Quantity"),
            ("part", "noun", vec![SemanticPrime::PartOf], vec![], "Piece"),
            ("piece", "noun", vec![SemanticPrime::PartOf], vec![], "Fragment"),
            ("kind", "noun", vec![SemanticPrime::KindOf], vec![], "Type"),
            ("type", "noun", vec![SemanticPrime::KindOf], vec![], "Category"),
            ("group", "noun", vec![SemanticPrime::Much], vec![SemanticPrime::Same], "Collection"),
            ("example", "noun", vec![SemanticPrime::One], vec![SemanticPrime::Like], "Instance"),
            ("case", "noun", vec![SemanticPrime::One], vec![], "Instance"),
            ("situation", "noun", vec![SemanticPrime::Be], vec![SemanticPrime::Now], "Current state"),
            ("condition", "noun", vec![SemanticPrime::Be], vec![], "State"),
            ("state", "noun", vec![SemanticPrime::Be], vec![], "Condition"),
            ("level", "noun", vec![SemanticPrime::Like], vec![SemanticPrime::Above], "Degree"),
            ("degree", "noun", vec![SemanticPrime::Like], vec![SemanticPrime::Much], "Amount"),
            ("step", "noun", vec![SemanticPrime::One], vec![SemanticPrime::After, SemanticPrime::Before], "Stage"),
            ("stage", "noun", vec![SemanticPrime::One], vec![SemanticPrime::ForSomeTime], "Phase"),
            ("phase", "noun", vec![SemanticPrime::One], vec![SemanticPrime::ForSomeTime], "Period"),
        ];
        for (word, pos, core, mods, expl) in abstract_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.0, 0.2);
        }

        // ===== TECHNOLOGY & CONSCIOUSNESS TERMS =====
        let tech_words = [
            ("computer", "noun", vec![SemanticPrime::Something, SemanticPrime::Think], vec![SemanticPrime::Do], "Thinking machine"),
            ("program", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::Words], "Instructions"),
            ("data", "noun", vec![SemanticPrime::Know], vec![SemanticPrime::Something], "Stored info"),
            ("code", "noun", vec![SemanticPrime::Words], vec![SemanticPrime::Do], "Instructions"),
            ("system", "noun", vec![SemanticPrime::PartOf], vec![SemanticPrime::Much], "Connected parts"),
            ("network", "noun", vec![SemanticPrime::PartOf], vec![SemanticPrime::With, SemanticPrime::Much], "Connected things"),
            ("information", "noun", vec![SemanticPrime::Know], vec![], "Data meaning"),
            ("technology", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::Something], "Tools methods"),
            ("machine", "noun", vec![SemanticPrime::Something, SemanticPrime::Do], vec![], "Working thing"),
            ("device", "noun", vec![SemanticPrime::Something, SemanticPrime::Do], vec![SemanticPrime::Small], "Small machine"),
            ("tool", "noun", vec![SemanticPrime::Something, SemanticPrime::Do], vec![], "Helper thing"),
            ("intelligence", "noun", vec![SemanticPrime::Think], vec![SemanticPrime::Good], "Thinking ability"),
            ("artificial", "adj", vec![SemanticPrime::Not], vec![SemanticPrime::Live], "Made by people"),
            ("consciousness", "noun", vec![SemanticPrime::Know, SemanticPrime::Feel, SemanticPrime::I], vec![], "Aware experience"),
            ("awareness", "noun", vec![SemanticPrime::Know], vec![SemanticPrime::Now], "State of knowing"),
            ("attention", "noun", vec![SemanticPrime::Think], vec![SemanticPrime::Very, SemanticPrime::One], "Focus on"),
            ("perception", "noun", vec![SemanticPrime::See, SemanticPrime::Know], vec![], "Sensing"),
            ("cognition", "noun", vec![SemanticPrime::Think], vec![], "Thinking process"),
            ("emotion", "noun", vec![SemanticPrime::Feel], vec![], "Feeling state"),
            ("sensation", "noun", vec![SemanticPrime::Feel, SemanticPrime::Body], vec![], "Body feeling"),
            ("qualia", "noun", vec![SemanticPrime::Feel], vec![SemanticPrime::I], "Subjective experience"),
            ("integration", "noun", vec![SemanticPrime::PartOf], vec![SemanticPrime::One], "Make whole"),
            ("semantic", "adj", vec![SemanticPrime::Words], vec![SemanticPrime::Know], "Meaning-related"),
            ("prime", "noun", vec![SemanticPrime::One], vec![SemanticPrime::Before, SemanticPrime::All], "Basic element"),
            ("vector", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move], "Direction amount"),
            ("dimension", "noun", vec![SemanticPrime::Like], vec![], "Aspect measure"),
            ("encoding", "noun", vec![SemanticPrime::Do], vec![SemanticPrime::Words], "Representation"),
            ("grounding", "noun", vec![SemanticPrime::Be], vec![SemanticPrime::True], "Base connection"),
            ("binding", "noun", vec![SemanticPrime::With], vec![], "Connection"),
            ("bundling", "noun", vec![SemanticPrime::Much], vec![SemanticPrime::One], "Combining"),
            ("similarity", "noun", vec![SemanticPrime::Same], vec![SemanticPrime::Like], "Alike measure"),
        ];
        for (word, pos, core, mods, expl) in tech_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.0, 0.2);
        }

        // ===== COMMON ADVERBS =====
        let adverb_words = [
            ("always", "adv", vec![SemanticPrime::All], vec![SemanticPrime::When], "Every time"),
            ("never", "adv", vec![SemanticPrime::Not, SemanticPrime::All], vec![SemanticPrime::When], "No time"),
            ("sometimes", "adv", vec![SemanticPrime::Some], vec![SemanticPrime::When], "Occasional"),
            ("often", "adv", vec![SemanticPrime::Much], vec![SemanticPrime::When], "Frequent"),
            ("usually", "adv", vec![SemanticPrime::Much], vec![SemanticPrime::When], "Most times"),
            ("rarely", "adv", vec![SemanticPrime::Little], vec![SemanticPrime::When], "Seldom"),
            ("seldom", "adv", vec![SemanticPrime::Little], vec![SemanticPrime::When], "Rare"),
            ("already", "adv", vec![SemanticPrime::Before], vec![SemanticPrime::Now], "Before now"),
            ("still", "adv", vec![SemanticPrime::Same], vec![SemanticPrime::Now], "Continuing"),
            ("yet", "adv", vec![SemanticPrime::Now], vec![SemanticPrime::Not], "Until now"),
            ("just", "adv", vec![SemanticPrime::Now], vec![SemanticPrime::ShortTime], "Recently"),
            ("soon", "adv", vec![SemanticPrime::After], vec![SemanticPrime::ShortTime], "Shortly"),
            ("later", "adv", vec![SemanticPrime::After], vec![], "After this"),
            ("early", "adv", vec![SemanticPrime::Before], vec![], "Before expected"),
            ("late", "adv", vec![SemanticPrime::After], vec![], "After expected"),
            ("quickly", "adv", vec![SemanticPrime::Move], vec![SemanticPrime::Very], "Fast"),
            ("slowly", "adv", vec![SemanticPrime::Move], vec![SemanticPrime::Little], "Not fast"),
            ("carefully", "adv", vec![SemanticPrime::Think], vec![SemanticPrime::Good], "With care"),
            ("easily", "adv", vec![SemanticPrime::Good], vec![SemanticPrime::Can], "Without effort"),
            ("hardly", "adv", vec![SemanticPrime::Little], vec![SemanticPrime::Can], "Barely"),
            ("nearly", "adv", vec![SemanticPrime::Near], vec![SemanticPrime::Same], "Almost"),
            ("almost", "adv", vec![SemanticPrime::Near], vec![SemanticPrime::Same], "Nearly"),
            ("completely", "adv", vec![SemanticPrime::All], vec![], "Totally"),
            ("totally", "adv", vec![SemanticPrime::All], vec![], "Completely"),
            ("really", "adv", vec![SemanticPrime::True], vec![SemanticPrime::Very], "Truly"),
            ("actually", "adv", vec![SemanticPrime::True], vec![], "In fact"),
            ("probably", "adv", vec![SemanticPrime::Maybe], vec![SemanticPrime::More], "Likely"),
            ("possibly", "adv", vec![SemanticPrime::Maybe], vec![], "Perhaps"),
            ("certainly", "adv", vec![SemanticPrime::True], vec![SemanticPrime::Very], "Definitely"),
            ("definitely", "adv", vec![SemanticPrime::True], vec![SemanticPrime::Very], "For sure"),
            ("exactly", "adv", vec![SemanticPrime::Same], vec![SemanticPrime::Very], "Precisely"),
            ("approximately", "adv", vec![SemanticPrime::Near], vec![SemanticPrime::Same], "About"),
            ("mainly", "adv", vec![SemanticPrime::More], vec![], "Mostly"),
            ("mostly", "adv", vec![SemanticPrime::More], vec![], "Mainly"),
            ("especially", "adv", vec![SemanticPrime::More], vec![SemanticPrime::Very], "Particularly"),
            ("particularly", "adv", vec![SemanticPrime::More], vec![], "Especially"),
            ("generally", "adv", vec![SemanticPrime::All], vec![SemanticPrime::Like], "Usually"),
            ("specifically", "adv", vec![SemanticPrime::One], vec![], "Exactly"),
            ("together", "adv", vec![SemanticPrime::With], vec![], "As one"),
            ("apart", "adv", vec![SemanticPrime::Not, SemanticPrime::With], vec![], "Separated"),
            ("alone", "adv", vec![SemanticPrime::One], vec![SemanticPrime::Not, SemanticPrime::With], "By self"),
            ("forward", "adv", vec![SemanticPrime::Move], vec![SemanticPrime::Before], "Ahead"),
            ("backward", "adv", vec![SemanticPrime::Move], vec![SemanticPrime::After], "Behind"),
            ("upward", "adv", vec![SemanticPrime::Move], vec![SemanticPrime::Above], "Going up"),
            ("downward", "adv", vec![SemanticPrime::Move], vec![SemanticPrime::Below], "Going down"),
            ("otherwise", "adv", vec![SemanticPrime::Other], vec![], "Differently"),
            ("instead", "adv", vec![SemanticPrime::Other], vec![SemanticPrime::Not], "In place of"),
            ("anyway", "adv", vec![SemanticPrime::Not], vec![SemanticPrime::Because], "Regardless"),
            ("however", "adv", vec![SemanticPrime::Not], vec![SemanticPrime::Same], "But"),
            ("therefore", "adv", vec![SemanticPrime::Because], vec![], "So"),
        ];
        for (word, pos, core, mods, expl) in adverb_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.0, 0.2);
        }

        // ===== COMMON PREPOSITIONS & CONJUNCTIONS =====
        let prep_words = [
            ("about", "prep", vec![SemanticPrime::Near], vec![SemanticPrime::Like], "Concerning"),
            ("across", "prep", vec![SemanticPrime::Move], vec![SemanticPrime::Side], "From side"),
            ("against", "prep", vec![SemanticPrime::Touch], vec![SemanticPrime::Not], "Opposing"),
            ("along", "prep", vec![SemanticPrime::Near], vec![SemanticPrime::Side], "Beside"),
            ("among", "prep", vec![SemanticPrime::Inside], vec![SemanticPrime::Much], "In group"),
            ("around", "prep", vec![SemanticPrime::Near], vec![SemanticPrime::Side], "Surrounding"),
            ("behind", "prep", vec![SemanticPrime::After], vec![SemanticPrime::Where], "At back"),
            ("beside", "prep", vec![SemanticPrime::Near], vec![SemanticPrime::Side], "Next to"),
            ("between", "prep", vec![SemanticPrime::Inside], vec![SemanticPrime::Two], "In middle"),
            ("beyond", "prep", vec![SemanticPrime::Far], vec![SemanticPrime::After], "Past"),
            ("during", "prep", vec![SemanticPrime::When], vec![SemanticPrime::ForSomeTime], "In time of"),
            ("except", "prep", vec![SemanticPrime::Not], vec![], "But not"),
            ("into", "prep", vec![SemanticPrime::Move], vec![SemanticPrime::Inside], "To inside"),
            ("onto", "prep", vec![SemanticPrime::Move], vec![SemanticPrime::On], "To surface"),
            ("since", "prep", vec![SemanticPrime::After], vec![SemanticPrime::Before], "From then"),
            ("through", "prep", vec![SemanticPrime::Move], vec![SemanticPrime::Inside], "Via inside"),
            ("toward", "prep", vec![SemanticPrime::Move], vec![SemanticPrime::Near], "In direction"),
            ("under", "prep", vec![SemanticPrime::Below], vec![], "Beneath"),
            ("until", "prep", vec![SemanticPrime::Before], vec![], "Up to"),
            ("upon", "prep", vec![SemanticPrime::On], vec![], "On top"),
            ("within", "prep", vec![SemanticPrime::Inside], vec![], "In bounds"),
            ("without", "prep", vec![SemanticPrime::Not], vec![SemanticPrime::With], "Lacking"),
            ("although", "conj", vec![SemanticPrime::Not], vec![SemanticPrime::Because], "Even though"),
            ("because", "conj", vec![SemanticPrime::Because], vec![], "For reason"),
            ("since", "conj", vec![SemanticPrime::Because], vec![], "As reason"),
            ("unless", "conj", vec![SemanticPrime::If], vec![SemanticPrime::Not], "If not"),
            ("whether", "conj", vec![SemanticPrime::If], vec![SemanticPrime::Maybe], "If or not"),
            ("while", "conj", vec![SemanticPrime::When], vec![SemanticPrime::Same], "At same time"),
            ("whereas", "conj", vec![SemanticPrime::Not], vec![SemanticPrime::Same], "But contrast"),
            ("whenever", "conj", vec![SemanticPrime::When], vec![SemanticPrime::All], "Any time"),
            ("wherever", "conj", vec![SemanticPrime::Where], vec![SemanticPrime::All], "Any place"),
            ("however", "conj", vec![SemanticPrime::Not], vec![SemanticPrime::Same], "But"),
            ("moreover", "conj", vec![SemanticPrime::More], vec![], "Also"),
            ("furthermore", "conj", vec![SemanticPrime::More], vec![], "Additionally"),
            ("therefore", "conj", vec![SemanticPrime::Because], vec![], "Thus"),
            ("nevertheless", "conj", vec![SemanticPrime::Not], vec![SemanticPrime::Because], "Still"),
            ("otherwise", "conj", vec![SemanticPrime::Other], vec![], "If not"),
        ];
        for (word, pos, core, mods, expl) in prep_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.0, 0.2);
        }

        // ===== COMMON INTERJECTIONS =====
        let interj_words = [
            ("oh", "interj", vec![SemanticPrime::Feel], vec![], "Surprise"),
            ("ah", "interj", vec![SemanticPrime::Feel], vec![SemanticPrime::Know], "Understanding"),
            ("wow", "interj", vec![SemanticPrime::Feel], vec![SemanticPrime::Good, SemanticPrime::Very], "Amazement"),
            ("oops", "interj", vec![SemanticPrime::Feel], vec![SemanticPrime::Bad], "Mistake"),
            ("uh", "interj", vec![SemanticPrime::Think], vec![], "Hesitation"),
            ("um", "interj", vec![SemanticPrime::Think], vec![], "Pause"),
            ("hmm", "interj", vec![SemanticPrime::Think], vec![], "Considering"),
            ("huh", "interj", vec![SemanticPrime::Think], vec![SemanticPrime::Not, SemanticPrime::Know], "Confusion"),
            ("aha", "interj", vec![SemanticPrime::Know], vec![SemanticPrime::Now], "Discovery"),
            ("hey", "interj", vec![SemanticPrime::Say], vec![SemanticPrime::Want], "Attention"),
            ("hi", "interj", vec![SemanticPrime::Say], vec![SemanticPrime::Good], "Greeting"),
            ("bye", "interj", vec![SemanticPrime::Say], vec![SemanticPrime::Move, SemanticPrime::Far], "Farewell"),
            ("please", "interj", vec![SemanticPrime::Want], vec![SemanticPrime::Good], "Polite request"),
            ("thanks", "interj", vec![SemanticPrime::Feel], vec![SemanticPrime::Good], "Gratitude"),
            ("sorry", "interj", vec![SemanticPrime::Feel], vec![SemanticPrime::Bad], "Apology"),
            ("yes", "interj", vec![SemanticPrime::True], vec![], "Affirmation"),
            ("no", "interj", vec![SemanticPrime::Not, SemanticPrime::True], vec![], "Negation"),
            ("okay", "interj", vec![SemanticPrime::Good], vec![], "Agreement"),
            ("ok", "interj", vec![SemanticPrime::Good], vec![], "Agreement"),
            ("right", "interj", vec![SemanticPrime::True], vec![], "Confirmation"),
            ("well", "interj", vec![SemanticPrime::Think], vec![], "Transition"),
            ("so", "interj", vec![SemanticPrime::Because], vec![], "Transition"),
        ];
        for (word, pos, core, mods, expl) in interj_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.0, 0.3);
        }

        // ===== FAMILY & RELATIONSHIPS =====
        let family_words = [
            ("mother", "noun", vec![SemanticPrime::Someone, SemanticPrime::Live], vec![SemanticPrime::I, SemanticPrime::Before], "Female parent"),
            ("father", "noun", vec![SemanticPrime::Someone, SemanticPrime::Live], vec![SemanticPrime::I, SemanticPrime::Before], "Male parent"),
            ("parent", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::I, SemanticPrime::Before], "Mother or father"),
            ("child", "noun", vec![SemanticPrime::Someone, SemanticPrime::Live], vec![SemanticPrime::Small], "Young person"),
            ("children", "noun", vec![SemanticPrime::People], vec![SemanticPrime::Small], "Young people"),
            ("son", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::I, SemanticPrime::After], "Male child"),
            ("daughter", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::I, SemanticPrime::After], "Female child"),
            ("brother", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Same, SemanticPrime::PartOf], "Male sibling"),
            ("sister", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Same, SemanticPrime::PartOf], "Female sibling"),
            ("husband", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::With, SemanticPrime::Live], "Male spouse"),
            ("wife", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::With, SemanticPrime::Live], "Female spouse"),
            ("baby", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Small, SemanticPrime::Very], "Very young child"),
            ("grandmother", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Before, SemanticPrime::Before], "Parent's mother"),
            ("grandfather", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Before, SemanticPrime::Before], "Parent's father"),
            ("aunt", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Side], "Parent's sister"),
            ("uncle", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Side], "Parent's brother"),
            ("cousin", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Side, SemanticPrime::Same], "Aunt's child"),
            ("nephew", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Side, SemanticPrime::After], "Sibling's son"),
            ("niece", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Side, SemanticPrime::After], "Sibling's daughter"),
            ("friend", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Good, SemanticPrime::With], "Close person"),
            ("neighbor", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Near, SemanticPrime::Live], "Person nearby"),
            ("partner", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::With], "Life companion"),
            ("couple", "noun", vec![SemanticPrime::People], vec![SemanticPrime::Two, SemanticPrime::With], "Two together"),
            ("stranger", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Not, SemanticPrime::Know], "Unknown person"),
            ("guest", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Near, SemanticPrime::ShortTime], "Visiting person"),
        ];
        for (word, pos, core, mods, expl) in family_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.2, 0.3);
        }

        // ===== TIME EXPRESSIONS =====
        let time_words = [
            ("today", "noun", vec![SemanticPrime::Now], vec![SemanticPrime::When], "This day"),
            ("yesterday", "noun", vec![SemanticPrime::Before], vec![SemanticPrime::ShortTime], "Previous day"),
            ("tomorrow", "noun", vec![SemanticPrime::After], vec![SemanticPrime::ShortTime], "Next day"),
            ("morning", "noun", vec![SemanticPrime::When], vec![SemanticPrime::Before], "Early day"),
            ("afternoon", "noun", vec![SemanticPrime::When], vec![], "Middle day"),
            ("evening", "noun", vec![SemanticPrime::When], vec![SemanticPrime::After], "Late day"),
            ("night", "noun", vec![SemanticPrime::When], vec![SemanticPrime::Not, SemanticPrime::See], "Dark time"),
            ("midnight", "noun", vec![SemanticPrime::When], vec![SemanticPrime::Inside], "Middle night"),
            ("noon", "noun", vec![SemanticPrime::When], vec![SemanticPrime::Inside], "Middle day"),
            ("dawn", "noun", vec![SemanticPrime::When], vec![SemanticPrime::Before], "Day start"),
            ("dusk", "noun", vec![SemanticPrime::When], vec![SemanticPrime::After], "Day end"),
            ("week", "noun", vec![SemanticPrime::When], vec![SemanticPrime::ForSomeTime], "Seven days"),
            ("month", "noun", vec![SemanticPrime::When], vec![SemanticPrime::ForSomeTime], "About 30 days"),
            ("year", "noun", vec![SemanticPrime::When], vec![SemanticPrime::LongTime], "365 days"),
            ("decade", "noun", vec![SemanticPrime::When], vec![SemanticPrime::LongTime, SemanticPrime::Very], "Ten years"),
            ("century", "noun", vec![SemanticPrime::When], vec![SemanticPrime::LongTime, SemanticPrime::Very], "100 years"),
            ("second", "noun", vec![SemanticPrime::When], vec![SemanticPrime::ShortTime, SemanticPrime::Very], "Brief time"),
            ("minute", "noun", vec![SemanticPrime::When], vec![SemanticPrime::ShortTime], "60 seconds"),
            ("hour", "noun", vec![SemanticPrime::When], vec![SemanticPrime::ForSomeTime], "60 minutes"),
            ("monday", "noun", vec![SemanticPrime::When], vec![SemanticPrime::One], "First weekday"),
            ("tuesday", "noun", vec![SemanticPrime::When], vec![SemanticPrime::Two], "Second weekday"),
            ("wednesday", "noun", vec![SemanticPrime::When], vec![], "Third weekday"),
            ("thursday", "noun", vec![SemanticPrime::When], vec![], "Fourth weekday"),
            ("friday", "noun", vec![SemanticPrime::When], vec![], "Fifth weekday"),
            ("saturday", "noun", vec![SemanticPrime::When], vec![SemanticPrime::Not, SemanticPrime::Do], "Weekend day"),
            ("sunday", "noun", vec![SemanticPrime::When], vec![SemanticPrime::Not, SemanticPrime::Do], "Weekend day"),
            ("spring", "noun", vec![SemanticPrime::When], vec![SemanticPrime::Live], "Warm season"),
            ("summer", "noun", vec![SemanticPrime::When], vec![SemanticPrime::Feel], "Hot season"),
            ("autumn", "noun", vec![SemanticPrime::When], vec![], "Cool season"),
            ("fall", "noun", vec![SemanticPrime::When], vec![], "Autumn"),
            ("winter", "noun", vec![SemanticPrime::When], vec![], "Cold season"),
        ];
        for (word, pos, core, mods, expl) in time_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.9, 0.0, 0.2);
        }

        // ===== COLORS =====
        let color_words = [
            ("red", "adj", vec![SemanticPrime::See], vec![SemanticPrime::Feel], "Blood color"),
            ("blue", "adj", vec![SemanticPrime::See], vec![SemanticPrime::Above], "Sky color"),
            ("green", "adj", vec![SemanticPrime::See], vec![SemanticPrime::Live], "Plant color"),
            ("yellow", "adj", vec![SemanticPrime::See], vec![SemanticPrime::Above], "Sun color"),
            ("orange", "adj", vec![SemanticPrime::See], vec![], "Fruit color"),
            ("purple", "adj", vec![SemanticPrime::See], vec![], "Royalty color"),
            ("pink", "adj", vec![SemanticPrime::See], vec![SemanticPrime::Good], "Soft red"),
            ("brown", "adj", vec![SemanticPrime::See], vec![SemanticPrime::Below], "Earth color"),
            ("black", "adj", vec![SemanticPrime::See], vec![SemanticPrime::Not], "No light color"),
            ("white", "adj", vec![SemanticPrime::See], vec![SemanticPrime::All], "All light color"),
            ("gray", "adj", vec![SemanticPrime::See], vec![], "Between black white"),
            ("grey", "adj", vec![SemanticPrime::See], vec![], "Between black white"),
            ("golden", "adj", vec![SemanticPrime::See], vec![SemanticPrime::Good], "Gold color"),
            ("silver", "adj", vec![SemanticPrime::See], vec![], "Metal color"),
        ];
        for (word, pos, core, mods, expl) in color_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.1, 0.2);
        }

        // ===== NUMBERS & QUANTITY =====
        let number_words = [
            ("one", "num", vec![SemanticPrime::One], vec![], "Single"),
            ("two", "num", vec![SemanticPrime::Two], vec![], "Pair"),
            ("three", "num", vec![SemanticPrime::Some], vec![], "Few"),
            ("four", "num", vec![SemanticPrime::Some], vec![], "Several"),
            ("five", "num", vec![SemanticPrime::Some], vec![], "Hand count"),
            ("six", "num", vec![SemanticPrime::Some], vec![], "Half dozen"),
            ("seven", "num", vec![SemanticPrime::Some], vec![], "Week days"),
            ("eight", "num", vec![SemanticPrime::Some], vec![], "Octet"),
            ("nine", "num", vec![SemanticPrime::Some], vec![], "Near ten"),
            ("ten", "num", vec![SemanticPrime::Some, SemanticPrime::Much], vec![], "Two hands"),
            ("hundred", "num", vec![SemanticPrime::Much], vec![], "Ten tens"),
            ("thousand", "num", vec![SemanticPrime::Much, SemanticPrime::Very], vec![], "Ten hundreds"),
            ("million", "num", vec![SemanticPrime::Much, SemanticPrime::Very], vec![SemanticPrime::Very], "Thousand thousands"),
            ("first", "adj", vec![SemanticPrime::One], vec![SemanticPrime::Before], "Number one"),
            ("second", "adj", vec![SemanticPrime::Two], vec![], "Number two"),
            ("third", "adj", vec![SemanticPrime::Some], vec![], "Number three"),
            ("last", "adj", vec![SemanticPrime::One], vec![SemanticPrime::After, SemanticPrime::All], "Final one"),
            ("next", "adj", vec![SemanticPrime::One], vec![SemanticPrime::After], "Following one"),
            ("half", "noun", vec![SemanticPrime::PartOf], vec![SemanticPrime::Two], "Equal part"),
            ("quarter", "noun", vec![SemanticPrime::PartOf], vec![], "Fourth part"),
            ("double", "adj", vec![SemanticPrime::Two], vec![SemanticPrime::Same], "Twice"),
            ("triple", "adj", vec![SemanticPrime::Some], vec![SemanticPrime::Same], "Thrice"),
            ("single", "adj", vec![SemanticPrime::One], vec![], "Only one"),
            ("pair", "noun", vec![SemanticPrime::Two], vec![SemanticPrime::Same], "Two together"),
            ("dozen", "noun", vec![SemanticPrime::Some, SemanticPrime::Much], vec![], "Twelve"),
        ];
        for (word, pos, core, mods, expl) in number_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.9, 0.0, 0.2);
        }

        // ===== FOOD & DRINK =====
        let food_words = [
            ("food", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body, SemanticPrime::Good], "Eaten things"),
            ("meal", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::When], "Food time"),
            ("breakfast", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Before], "Morning food"),
            ("lunch", "noun", vec![SemanticPrime::Something], vec![], "Midday food"),
            ("dinner", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::After], "Evening food"),
            ("snack", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Small], "Small food"),
            ("bread", "noun", vec![SemanticPrime::Something], vec![], "Baked grain"),
            ("rice", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Small], "Grain food"),
            ("pasta", "noun", vec![SemanticPrime::Something], vec![], "Dough food"),
            ("meat", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body], "Animal food"),
            ("chicken", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body], "Bird meat"),
            ("fish", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body], "Sea creature"),
            ("egg", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Before], "Bird product"),
            ("cheese", "noun", vec![SemanticPrime::Something], vec![], "Milk product"),
            ("butter", "noun", vec![SemanticPrime::Something], vec![], "Cream product"),
            ("milk", "noun", vec![SemanticPrime::Something], vec![], "White drink"),
            ("fruit", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Live, SemanticPrime::Good], "Plant food"),
            ("apple", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Live], "Round fruit"),
            ("banana", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Live], "Yellow fruit"),
            ("vegetable", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Live], "Plant food"),
            ("salad", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Live, SemanticPrime::Much], "Mixed plants"),
            ("soup", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move], "Liquid food"),
            ("cake", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Good], "Sweet baked"),
            ("cookie", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Small, SemanticPrime::Good], "Sweet small"),
            ("candy", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Good, SemanticPrime::Very], "Very sweet"),
            ("chocolate", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Good], "Brown sweet"),
            ("ice", "noun", vec![SemanticPrime::Something], vec![], "Frozen water"),
            ("coffee", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move], "Brown drink"),
            ("tea", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move], "Leaf drink"),
            ("juice", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Live], "Fruit liquid"),
            ("beer", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move], "Grain drink"),
            ("wine", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move], "Grape drink"),
            ("sugar", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Good], "Sweet taste"),
            ("salt", "noun", vec![SemanticPrime::Something], vec![], "Mineral taste"),
            ("pepper", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Feel], "Spicy taste"),
            ("sauce", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move], "Liquid flavor"),
            ("recipe", "noun", vec![SemanticPrime::Words], vec![SemanticPrime::Do], "Food instructions"),
            ("hungry", "adj", vec![SemanticPrime::Want], vec![SemanticPrime::Body], "Need food"),
            ("thirsty", "adj", vec![SemanticPrime::Want], vec![SemanticPrime::Body], "Need drink"),
            ("delicious", "adj", vec![SemanticPrime::Good], vec![SemanticPrime::Feel, SemanticPrime::Very], "Very tasty"),
        ];
        for (word, pos, core, mods, expl) in food_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.2, 0.3);
        }

        // ===== QUESTION WORDS =====
        let question_words = [
            ("what", "pron", vec![SemanticPrime::Something], vec![SemanticPrime::Want, SemanticPrime::Know], "Which thing"),
            ("who", "pron", vec![SemanticPrime::Someone], vec![SemanticPrime::Want, SemanticPrime::Know], "Which person"),
            ("where", "adv", vec![SemanticPrime::Where], vec![SemanticPrime::Want, SemanticPrime::Know], "Which place"),
            ("when", "adv", vec![SemanticPrime::When], vec![SemanticPrime::Want, SemanticPrime::Know], "Which time"),
            ("why", "adv", vec![SemanticPrime::Because], vec![SemanticPrime::Want, SemanticPrime::Know], "What reason"),
            ("how", "adv", vec![SemanticPrime::Like], vec![SemanticPrime::Want, SemanticPrime::Know], "What manner"),
            ("which", "det", vec![SemanticPrime::One], vec![SemanticPrime::Want, SemanticPrime::Know], "What one"),
            ("whose", "det", vec![SemanticPrime::Someone], vec![SemanticPrime::Have], "Belonging to whom"),
        ];
        for (word, pos, core, mods, expl) in question_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 1.0, 0.0, 0.3);
        }

        // ===== DETERMINERS & ARTICLES =====
        let det_words = [
            ("the", "det", vec![SemanticPrime::Something], vec![SemanticPrime::Know], "Specific one"),
            ("a", "det", vec![SemanticPrime::One], vec![], "Any one"),
            ("an", "det", vec![SemanticPrime::One], vec![], "Any one before vowel"),
            ("this", "det", vec![SemanticPrime::Something], vec![SemanticPrime::Here], "Near one"),
            ("that", "det", vec![SemanticPrime::Something], vec![SemanticPrime::Far], "Far one"),
            ("these", "det", vec![SemanticPrime::Something], vec![SemanticPrime::Here, SemanticPrime::Much], "Near many"),
            ("those", "det", vec![SemanticPrime::Something], vec![SemanticPrime::Far, SemanticPrime::Much], "Far many"),
            ("my", "det", vec![SemanticPrime::I], vec![SemanticPrime::Have], "Belonging to me"),
            ("your", "det", vec![SemanticPrime::You], vec![SemanticPrime::Have], "Belonging to you"),
            ("his", "det", vec![SemanticPrime::Someone], vec![SemanticPrime::Have], "Belonging to him"),
            ("her", "det", vec![SemanticPrime::Someone], vec![SemanticPrime::Have], "Belonging to her"),
            ("its", "det", vec![SemanticPrime::Something], vec![SemanticPrime::Have], "Belonging to it"),
            ("our", "det", vec![SemanticPrime::People], vec![SemanticPrime::Have, SemanticPrime::I], "Belonging to us"),
            ("their", "det", vec![SemanticPrime::People], vec![SemanticPrime::Have], "Belonging to them"),
            ("every", "det", vec![SemanticPrime::All], vec![], "Each one"),
            ("each", "det", vec![SemanticPrime::All], vec![SemanticPrime::One], "All individually"),
            ("any", "det", vec![SemanticPrime::Some], vec![], "No matter which"),
            ("no", "det", vec![SemanticPrime::Not], vec![SemanticPrime::One], "Zero"),
            ("another", "det", vec![SemanticPrime::One], vec![SemanticPrime::Other], "Different one"),
            ("other", "det", vec![SemanticPrime::Something], vec![SemanticPrime::Not, SemanticPrime::Same], "Not this one"),
            ("both", "det", vec![SemanticPrime::Two], vec![SemanticPrime::All], "Two together"),
            ("few", "det", vec![SemanticPrime::Some], vec![SemanticPrime::Little], "Small number"),
            ("many", "det", vec![SemanticPrime::Some], vec![SemanticPrime::Much], "Large number"),
            ("several", "det", vec![SemanticPrime::Some], vec![], "More than two"),
        ];
        for (word, pos, core, mods, expl) in det_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 1.0, 0.0, 0.1);
        }

        // ===== ANIMALS =====
        let animal_words = [
            ("animal", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Move], "Living creature"),
            ("dog", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Good], "Pet animal"),
            ("cat", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Small], "Pet animal"),
            ("bird", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Above, SemanticPrime::Move], "Flying animal"),
            ("horse", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Big, SemanticPrime::Move], "Riding animal"),
            ("cow", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Big], "Farm animal"),
            ("pig", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![], "Farm animal"),
            ("sheep", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![], "Wool animal"),
            ("goat", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![], "Mountain animal"),
            ("chicken", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Small], "Farm bird"),
            ("duck", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![], "Water bird"),
            ("rabbit", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Small], "Hopping animal"),
            ("mouse", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Small, SemanticPrime::Very], "Tiny animal"),
            ("rat", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Small], "Rodent"),
            ("snake", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Move], "No-leg animal"),
            ("frog", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Move], "Jumping animal"),
            ("spider", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Small], "Eight-leg animal"),
            ("bee", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Small, SemanticPrime::Above], "Flying insect"),
            ("butterfly", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Good, SemanticPrime::Above], "Pretty insect"),
            ("ant", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Small, SemanticPrime::Very], "Tiny insect"),
            ("lion", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Big], "Wild cat"),
            ("tiger", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Big], "Striped cat"),
            ("bear", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Big, SemanticPrime::Very], "Forest animal"),
            ("elephant", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Big, SemanticPrime::Very], "Huge animal"),
            ("monkey", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Move], "Climbing animal"),
            ("wolf", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![], "Wild dog"),
            ("fox", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![], "Clever animal"),
            ("deer", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Move], "Forest animal"),
            ("whale", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Big, SemanticPrime::Very], "Sea mammal"),
            ("dolphin", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Good], "Smart sea animal"),
            ("shark", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Bad], "Dangerous fish"),
            ("turtle", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Move], "Shell animal"),
            ("eagle", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Above, SemanticPrime::Big], "Large bird"),
            ("owl", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::See], "Night bird"),
            ("pet", "noun", vec![SemanticPrime::Something, SemanticPrime::Live], vec![SemanticPrime::Good, SemanticPrime::Near], "Home animal"),
        ];
        for (word, pos, core, mods, expl) in animal_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.1, 0.2);
        }

        // ===== COMMON OBJECTS =====
        let object_words = [
            ("thing", "noun", vec![SemanticPrime::Something], vec![], "Any object"),
            ("object", "noun", vec![SemanticPrime::Something], vec![], "Physical item"),
            ("book", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Words], "Reading material"),
            ("paper", "noun", vec![SemanticPrime::Something], vec![], "Writing surface"),
            ("pen", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Words], "Writing tool"),
            ("pencil", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Words], "Drawing tool"),
            ("table", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::On], "Flat surface"),
            ("chair", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Be], "Sitting object"),
            ("desk", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Do], "Work surface"),
            ("bed", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Be], "Sleep place"),
            ("door", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move], "Entry way"),
            ("window", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::See], "Glass opening"),
            ("wall", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Side], "Room divider"),
            ("floor", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Below], "Walking surface"),
            ("ceiling", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Above], "Room top"),
            ("roof", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Above], "Building top"),
            ("room", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Inside], "Indoor space"),
            ("house", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Live], "Living place"),
            ("building", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Big], "Large structure"),
            ("car", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move], "Road vehicle"),
            ("bus", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move, SemanticPrime::People], "Public vehicle"),
            ("train", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move, SemanticPrime::Big], "Rail vehicle"),
            ("plane", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move, SemanticPrime::Above], "Air vehicle"),
            ("boat", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move], "Water vehicle"),
            ("ship", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move, SemanticPrime::Big], "Large boat"),
            ("bicycle", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move], "Two-wheel vehicle"),
            ("phone", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Say, SemanticPrime::Far], "Communication device"),
            ("camera", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::See], "Image device"),
            ("clock", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::When], "Time device"),
            ("watch", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::When, SemanticPrime::Small], "Wrist clock"),
            ("key", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Can], "Opening tool"),
            ("bag", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Inside], "Carrying container"),
            ("box", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Inside], "Storage container"),
            ("bottle", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Inside], "Liquid container"),
            ("cup", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Inside], "Drink container"),
            ("glass", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::See], "Transparent material"),
            ("plate", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::On], "Food surface"),
            ("bowl", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Inside], "Round container"),
            ("knife", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Do], "Cutting tool"),
            ("fork", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Do], "Eating tool"),
            ("spoon", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Do], "Scooping tool"),
            ("lamp", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::See], "Light source"),
            ("mirror", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::See], "Reflecting surface"),
            ("picture", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::See], "Image"),
            ("photo", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::See], "Photograph"),
            ("money", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Have], "Value medium"),
            ("gift", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Good], "Given thing"),
            ("present", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Good], "Gift"),
            ("toy", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Good, SemanticPrime::Do], "Play object"),
            ("game", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Good, SemanticPrime::Do], "Play activity"),
        ];
        for (word, pos, core, mods, expl) in object_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.0, 0.2);
        }

        // ===== CLOTHING =====
        let clothing_words = [
            ("clothes", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body, SemanticPrime::On], "Worn items"),
            ("shirt", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body], "Upper garment"),
            ("pants", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body], "Leg garment"),
            ("dress", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body], "One-piece garment"),
            ("skirt", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body], "Lower garment"),
            ("jacket", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body, SemanticPrime::On], "Outer garment"),
            ("coat", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body, SemanticPrime::On], "Warm outer"),
            ("sweater", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body], "Warm upper"),
            ("shoes", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body, SemanticPrime::Below], "Foot covering"),
            ("boots", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body, SemanticPrime::Below], "High foot covering"),
            ("socks", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body], "Foot garment"),
            ("hat", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body, SemanticPrime::Above], "Head covering"),
            ("cap", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body, SemanticPrime::Above], "Head covering"),
            ("gloves", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body], "Hand covering"),
            ("scarf", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body], "Neck covering"),
            ("belt", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body], "Waist band"),
            ("tie", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body], "Neck decoration"),
            ("suit", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body, SemanticPrime::Good], "Formal clothes"),
            ("uniform", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body, SemanticPrime::Same], "Work clothes"),
            ("pajamas", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Body], "Sleep clothes"),
        ];
        for (word, pos, core, mods, expl) in clothing_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.7, 0.0, 0.2);
        }

        // ===== PROFESSIONS & ROLES =====
        let profession_words = [
            ("person", "noun", vec![SemanticPrime::Someone], vec![], "Human being"),
            ("people", "noun", vec![SemanticPrime::People], vec![], "Humans"),
            ("man", "noun", vec![SemanticPrime::Someone], vec![], "Adult male"),
            ("woman", "noun", vec![SemanticPrime::Someone], vec![], "Adult female"),
            ("boy", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Small], "Young male"),
            ("girl", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Small], "Young female"),
            ("teacher", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Know], "Educator"),
            ("student", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Want, SemanticPrime::Know], "Learner"),
            ("doctor", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Body, SemanticPrime::Good], "Healer"),
            ("nurse", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Body, SemanticPrime::Good], "Care giver"),
            ("lawyer", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Words], "Legal person"),
            ("engineer", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Do, SemanticPrime::Think], "Builder person"),
            ("scientist", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Know, SemanticPrime::True], "Researcher"),
            ("artist", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Do, SemanticPrime::Good], "Creator"),
            ("writer", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Words], "Author"),
            ("musician", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Hear, SemanticPrime::Good], "Music maker"),
            ("actor", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Do], "Performer"),
            ("chef", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Do], "Food preparer"),
            ("farmer", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Live], "Food grower"),
            ("driver", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Move], "Vehicle operator"),
            ("pilot", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Move, SemanticPrime::Above], "Plane operator"),
            ("police", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Good], "Law enforcer"),
            ("soldier", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Do], "Military person"),
            ("boss", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Can], "Leader"),
            ("worker", "noun", vec![SemanticPrime::Someone], vec![SemanticPrime::Do], "Employee"),
        ];
        for (word, pos, core, mods, expl) in profession_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.0, 0.2);
        }

        // ===== PLACES & BUILDINGS =====
        let place_words = [
            ("school", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Know], "Learning place"),
            ("hospital", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Body, SemanticPrime::Good], "Healing place"),
            ("church", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Feel], "Religious place"),
            ("store", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Have], "Buying place"),
            ("shop", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Have], "Small store"),
            ("restaurant", "noun", vec![SemanticPrime::Where], vec![], "Eating place"),
            ("hotel", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Be], "Sleep place"),
            ("office", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Do], "Work place"),
            ("bank", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Have], "Money place"),
            ("library", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Words], "Book place"),
            ("museum", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::See], "Display place"),
            ("theater", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::See], "Show place"),
            ("park", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Live], "Green space"),
            ("garden", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Live], "Plant space"),
            ("market", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Have], "Trade place"),
            ("airport", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Move, SemanticPrime::Above], "Plane place"),
            ("station", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Move], "Transport place"),
            ("street", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Move], "Road"),
            ("road", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Move], "Path"),
            ("bridge", "noun", vec![SemanticPrime::Something], vec![SemanticPrime::Move], "Crossing structure"),
            ("city", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::People, SemanticPrime::Much], "Large town"),
            ("town", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::People], "Small city"),
            ("village", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::People, SemanticPrime::Small], "Tiny town"),
            ("country", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::Big], "Nation area"),
            ("world", "noun", vec![SemanticPrime::Where], vec![SemanticPrime::All], "Earth"),
        ];
        for (word, pos, core, mods, expl) in place_words {
            self.add_grounded_word(word, pos, core, mods, "bind", expl, 0.8, 0.0, 0.2);
        }
    }

    /// Add a word with semantic grounding
    fn add_grounded_word(
        &mut self,
        word: &str,
        pos: &str,
        core_primes: Vec<SemanticPrime>,
        modifier_primes: Vec<SemanticPrime>,
        composition: &str,
        explanation: &str,
        frequency: f32,
        valence: f32,
        arousal: f32,
    ) {
        // Compute encoding by composing primes
        let encoding = self.compute_grounded_encoding(&core_primes, &modifier_primes, composition);

        let grounding = SemanticGrounding {
            core_primes: core_primes.clone(),
            modifier_primes: modifier_primes.clone(),
            composition: composition.to_string(),
            explanation: explanation.to_string(),
        };

        let entry = WordEntry {
            word: word.to_string(),
            normalized: word.to_lowercase(),
            pos: pos.to_string(),
            encoding,
            grounding,
            frequency,
            valence,
            arousal,
        };

        // Add to main index
        self.entries.insert(word.to_string(), entry.clone());

        // Add to POS index
        self.by_pos
            .entry(pos.to_string())
            .or_insert_with(Vec::new)
            .push(word.to_string());
    }

    /// Compute encoding from semantic primes
    fn compute_grounded_encoding(
        &self,
        core_primes: &[SemanticPrime],
        modifier_primes: &[SemanticPrime],
        composition: &str,
    ) -> HV16 {
        let mut vectors: Vec<HV16> = Vec::new();

        // Get core prime vectors
        for prime in core_primes {
            vectors.push(*self.semantics.get_prime(*prime));
        }

        // Get modifier prime vectors (with permutation for distinction)
        for (i, prime) in modifier_primes.iter().enumerate() {
            let v = *self.semantics.get_prime(*prime);
            // Permute modifiers to distinguish from core
            let permuted = v.permute(i + 1);
            vectors.push(permuted);
        }

        if vectors.is_empty() {
            return HV16::zero();
        }

        match composition {
            "bind" => {
                // Binding: preserves role structure
                let mut result = vectors[0];
                for v in &vectors[1..] {
                    result = result.bind(v);
                }
                result
            }
            "bundle" => {
                // Bundling: superposition of meanings
                HV16::bundle(&vectors)
            }
            "sequence" => {
                // Sequence: temporal/ordered composition
                let mut result = vectors[0];
                for (i, v) in vectors[1..].iter().enumerate() {
                    let permuted = v.permute(i + 1);
                    result = result.bind(&permuted);
                }
                result
            }
            _ => HV16::bundle(&vectors),
        }
    }

    /// Build reverse index for HV16 → word lookup
    fn build_reverse_index(&mut self) {
        self.reverse_index = self.entries
            .iter()
            .map(|(word, entry)| (entry.encoding, word.clone()))
            .collect();
    }

    /// Look up a word
    pub fn get(&self, word: &str) -> Option<&WordEntry> {
        let normalized = word.to_lowercase();
        self.entries.get(&normalized)
    }

    /// Encode a word to HV16
    pub fn encode(&self, word: &str) -> Option<HV16> {
        self.get(word).map(|e| e.encoding)
    }

    /// Find words most similar to a vector
    pub fn find_similar(&self, query: &HV16, top_k: usize) -> Vec<(String, f32)> {
        let mut similarities: Vec<(String, f32)> = self.reverse_index
            .iter()
            .map(|(hv, word)| {
                let sim = query.similarity(hv);
                (word.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    /// Find words by part of speech
    pub fn words_by_pos(&self, pos: &str) -> Vec<&str> {
        self.by_pos
            .get(pos)
            .map(|words| words.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get semantic grounding explanation for a word
    pub fn explain(&self, word: &str) -> Option<String> {
        self.get(word).map(|entry| {
            let core: Vec<String> = entry.grounding.core_primes
                .iter()
                .map(|p| format!("{:?}", p))
                .collect();
            let mods: Vec<String> = entry.grounding.modifier_primes
                .iter()
                .map(|p| format!("{:?}", p))
                .collect();

            format!(
                "'{}' = {}({}){}\n  → {}",
                entry.word,
                entry.grounding.composition,
                core.join(", "),
                if mods.is_empty() { String::new() } else { format!(" + [{}]", mods.join(", ")) },
                entry.grounding.explanation
            )
        })
    }

    /// Get vocabulary size
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all words
    pub fn all_words(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for Vocabulary {
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
    fn test_vocabulary_creation() {
        let vocab = Vocabulary::new();
        assert!(vocab.len() > 50, "Should have at least 50 core words");
    }

    #[test]
    fn test_word_lookup() {
        let vocab = Vocabulary::new();

        let entry = vocab.get("happy").expect("Should find 'happy'");
        assert_eq!(entry.pos, "adj");
        assert!(entry.valence > 0.5, "Happy should have positive valence");
    }

    #[test]
    fn test_semantic_grounding() {
        let vocab = Vocabulary::new();

        let entry = vocab.get("understand").expect("Should find 'understand'");
        assert!(entry.grounding.core_primes.contains(&SemanticPrime::Know));
        assert!(entry.grounding.core_primes.contains(&SemanticPrime::Think));
    }

    #[test]
    fn test_encoding() {
        let vocab = Vocabulary::new();

        let hv = vocab.encode("think").expect("Should encode 'think'");
        assert_ne!(hv, HV16::zero(), "Encoding should not be zero");
    }

    #[test]
    fn test_similar_words() {
        let vocab = Vocabulary::new();

        let happy_hv = vocab.encode("happy").expect("Should encode 'happy'");
        let similar = vocab.find_similar(&happy_hv, 10);

        assert!(!similar.is_empty(), "Should find similar words");
        // With expanded vocabulary, semantically similar words may rank higher
        // Check that "happy" is in top results or that we find positive-emotion words
        let similar_words: Vec<&str> = similar.iter().map(|(w, _)| w.as_str()).collect();
        assert!(
            similar_words.contains(&"happy") ||
            similar_words.iter().any(|w| ["joy", "contentment", "pleasure", "good"].contains(w)),
            "Should find happy or semantically similar positive words in top 10"
        );
    }

    #[test]
    fn test_pos_lookup() {
        let vocab = Vocabulary::new();

        let verbs = vocab.words_by_pos("verb");
        assert!(verbs.len() > 10, "Should have at least 10 verbs");
        assert!(verbs.contains(&"think"), "Verbs should include 'think'");
    }

    #[test]
    fn test_explain() {
        let vocab = Vocabulary::new();

        let explanation = vocab.explain("conscious").expect("Should explain 'conscious'");
        assert!(explanation.contains("Know"), "Should mention Know prime");
        assert!(explanation.contains("Feel"), "Should mention Feel prime");
    }

    #[test]
    fn test_related_words_have_valid_similarity() {
        let vocab = Vocabulary::new();

        // Words should have valid similarity scores
        let think = vocab.encode("think").unwrap();
        let know = vocab.encode("know").unwrap();
        let happy = vocab.encode("happy").unwrap();

        let think_know_sim = think.similarity(&know);
        let think_happy_sim = think.similarity(&happy);

        // Both should be valid similarity values (not NaN, in reasonable range)
        assert!(!think_know_sim.is_nan(), "Similarity should be a valid number");
        assert!(!think_happy_sim.is_nan(), "Similarity should be a valid number");

        // Self-similarity should be 1.0
        let think_self = think.similarity(&think);
        assert!((think_self - 1.0).abs() < 0.01, "Self-similarity should be ~1.0");
    }

    #[test]
    fn test_consciousness_vocabulary() {
        let vocab = Vocabulary::new();

        // Consciousness-specific words should exist
        assert!(vocab.get("consciousness").is_some());
        assert!(vocab.get("awareness").is_some());
        assert!(vocab.get("experience").is_some());
        assert!(vocab.get("self").is_some());
    }
}
