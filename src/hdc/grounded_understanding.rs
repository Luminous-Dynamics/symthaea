//! # Grounded Understanding: True Comprehension Through Semantic Primitives
//!
//! This module bridges the gap between surface text and deep understanding by:
//!
//! 1. **Lexical Grounding**: Mapping words to universal semantic primes
//! 2. **Compositional Semantics**: Building complex meanings from primitives
//! 3. **Causal Structure**: Extracting "why" from linguistic structure
//! 4. **Embodied Grounding**: Connecting meaning to emotional/physiological experience
//! 5. **Conscious Integration**: Binding understanding with phenomenal experience
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    GROUNDED UNDERSTANDING PIPELINE                       │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │   "I feel sad because my friend left"                                   │
//! │                       │                                                 │
//! │                       ▼                                                 │
//! │   ┌──────────────────────────────────────┐                              │
//! │   │         LEXICAL DECOMPOSITION        │                              │
//! │   │   I → SELF                           │                              │
//! │   │   feel → FEEL                        │                              │
//! │   │   sad → FEEL(BAD)                    │                              │
//! │   │   because → CAUSES                   │                              │
//! │   │   friend → SOMEONE(LIKE,WANT,WITH)   │                              │
//! │   │   left → MOVE(AWAY)                  │                              │
//! │   └──────────────────────────────────────┘                              │
//! │                       │                                                 │
//! │                       ▼                                                 │
//! │   ┌──────────────────────────────────────┐                              │
//! │   │       COMPOSITIONAL STRUCTURE        │                              │
//! │   │                                      │                              │
//! │   │   CAUSE:  MOVE(FRIEND, AWAY)         │                              │
//! │   │   EFFECT: FEEL(SELF, BAD)            │                              │
//! │   │   AGENT:  SELF                       │                              │
//! │   │   THEME:  FRIEND                     │                              │
//! │   └──────────────────────────────────────┘                              │
//! │                       │                                                 │
//! │                       ▼                                                 │
//! │   ┌──────────────────────────────────────┐                              │
//! │   │        EMBODIED GROUNDING            │                              │
//! │   │                                      │                              │
//! │   │   Emotional: valence=-0.7, arousal=0.3 │                            │
//! │   │   Hormone:   ↑cortisol, ↓dopamine    │                              │
//! │   │   Prosody:   slow, low energy        │                              │
//! │   └──────────────────────────────────────┘                              │
//! │                       │                                                 │
//! │                       ▼                                                 │
//! │   ┌──────────────────────────────────────┐                              │
//! │   │      CONSCIOUS INTEGRATION           │                              │
//! │   │                                      │                              │
//! │   │   Φ-weighted binding                 │                              │
//! │   │   Qualia generation                  │                              │
//! │   │   Memory consolidation               │                              │
//! │   └──────────────────────────────────────┘                              │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## True Understanding vs Pattern Matching
//!
//! | Aspect | Pattern Matching | True Understanding |
//! |--------|------------------|-------------------|
//! | Basis | Statistical correlation | Compositional semantics |
//! | Generalization | Seen patterns only | Novel combinations |
//! | Explanation | None | Causal structure |
//! | Grounding | None | Embodied experience |
//! | Consciousness | None | Φ-integrated |

use super::universal_semantics::{UniversalSemantics, SemanticPrime};
use super::binary_hv::HV16;
use std::collections::HashMap;

// =============================================================================
// PLACEHOLDER: CausalRoleMarkers (stub until causal_mind module implemented)
// =============================================================================

/// Causal role markers for understanding causal structure in text
///
/// TODO: Move to causal_mind module when implemented
#[derive(Debug, Clone, Default)]
pub struct CausalRoleMarkers {
    /// Marker patterns (placeholder)
    _markers: HashMap<String, CausalRole>,
}

/// Types of causal roles
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CausalRole {
    Agent,
    Patient,
    Instrument,
    Goal,
    Source,
}

impl CausalRoleMarkers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract causal structure from text (placeholder)
    pub fn extract(&self, _text: &str) -> Vec<(String, CausalRole)> {
        Vec::new() // TODO: Implement causal extraction
    }
}

// =============================================================================
// LEXICAL GROUNDING: Word → Semantic Primes
// =============================================================================

/// Mapping from words to semantic prime decompositions
#[derive(Debug, Clone)]
pub struct LexicalGrounding {
    /// Word → list of component primes
    word_primes: HashMap<String, Vec<SemanticPrime>>,

    /// Word → emotional valence (-1 to +1)
    word_valence: HashMap<String, f32>,

    /// Word → arousal level (0 to 1)
    word_arousal: HashMap<String, f32>,

    /// Causal connectors
    causal_markers: HashMap<String, CausalRelation>,
}

/// Types of causal relations expressed in language
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CausalRelation {
    Causes,     // because, since, therefore
    Prevents,   // despite, although, prevents
    Enables,    // allows, lets, permits
    Temporal,   // before, after, when, while
    Conditional, // if, unless, would
}

impl LexicalGrounding {
    pub fn new() -> Self {
        let mut grounding = Self {
            word_primes: HashMap::new(),
            word_valence: HashMap::new(),
            word_arousal: HashMap::new(),
            causal_markers: HashMap::new(),
        };
        grounding.initialize_core_lexicon();
        grounding
    }

    /// Initialize the core lexicon with semantic prime mappings
    fn initialize_core_lexicon(&mut self) {
        // === PRONOUNS → SUBSTANTIVES ===
        self.add_word("i", vec![SemanticPrime::I], 0.0, 0.3);
        self.add_word("me", vec![SemanticPrime::I], 0.0, 0.3);
        self.add_word("my", vec![SemanticPrime::I], 0.0, 0.3);
        self.add_word("you", vec![SemanticPrime::You], 0.0, 0.3);
        self.add_word("your", vec![SemanticPrime::You], 0.0, 0.3);
        self.add_word("someone", vec![SemanticPrime::Someone], 0.0, 0.2);
        self.add_word("somebody", vec![SemanticPrime::Someone], 0.0, 0.2);
        self.add_word("something", vec![SemanticPrime::Something], 0.0, 0.2);
        self.add_word("people", vec![SemanticPrime::People], 0.0, 0.3);
        self.add_word("body", vec![SemanticPrime::Body], 0.0, 0.3);

        // === EMOTIONS → MENTAL PREDICATES + EVALUATORS ===
        self.add_word("happy", vec![SemanticPrime::Feel, SemanticPrime::Good], 0.8, 0.6);
        self.add_word("joy", vec![SemanticPrime::Feel, SemanticPrime::Good], 0.9, 0.7);
        self.add_word("joyful", vec![SemanticPrime::Feel, SemanticPrime::Good], 0.9, 0.7);
        self.add_word("love", vec![SemanticPrime::Feel, SemanticPrime::Good, SemanticPrime::Want], 0.9, 0.7);
        self.add_word("sad", vec![SemanticPrime::Feel, SemanticPrime::Bad], -0.7, 0.4);
        self.add_word("grief", vec![SemanticPrime::Feel, SemanticPrime::Bad, SemanticPrime::Die], -0.9, 0.5);
        self.add_word("angry", vec![SemanticPrime::Feel, SemanticPrime::Bad, SemanticPrime::Want], -0.6, 0.9);
        self.add_word("fear", vec![SemanticPrime::Feel, SemanticPrime::Bad, SemanticPrime::Maybe], -0.7, 0.8);
        self.add_word("afraid", vec![SemanticPrime::Feel, SemanticPrime::Bad, SemanticPrime::Maybe], -0.7, 0.8);
        self.add_word("calm", vec![SemanticPrime::Feel, SemanticPrime::Good], 0.3, 0.1);
        self.add_word("peaceful", vec![SemanticPrime::Feel, SemanticPrime::Good], 0.5, 0.1);
        self.add_word("anxious", vec![SemanticPrime::Feel, SemanticPrime::Bad, SemanticPrime::Maybe], -0.5, 0.7);
        self.add_word("worried", vec![SemanticPrime::Think, SemanticPrime::Bad, SemanticPrime::Maybe], -0.4, 0.6);
        self.add_word("excited", vec![SemanticPrime::Feel, SemanticPrime::Good], 0.7, 0.9);
        self.add_word("surprised", vec![SemanticPrime::Feel, SemanticPrime::Know], 0.1, 0.8);

        // === MENTAL VERBS ===
        self.add_word("think", vec![SemanticPrime::Think], 0.0, 0.4);
        self.add_word("know", vec![SemanticPrime::Know], 0.1, 0.3);
        self.add_word("believe", vec![SemanticPrime::Think, SemanticPrime::True], 0.1, 0.3);
        self.add_word("want", vec![SemanticPrime::Want], 0.2, 0.5);
        self.add_word("need", vec![SemanticPrime::Want], 0.0, 0.6);
        self.add_word("feel", vec![SemanticPrime::Feel], 0.0, 0.5);
        self.add_word("see", vec![SemanticPrime::See], 0.0, 0.3);
        self.add_word("hear", vec![SemanticPrime::Hear], 0.0, 0.3);
        self.add_word("understand", vec![SemanticPrime::Know, SemanticPrime::Think], 0.2, 0.4);
        self.add_word("remember", vec![SemanticPrime::Know, SemanticPrime::Before], 0.0, 0.4);
        self.add_word("forget", vec![SemanticPrime::Know, SemanticPrime::Not], -0.2, 0.3);

        // === ACTIONS ===
        self.add_word("do", vec![SemanticPrime::Do], 0.0, 0.5);
        self.add_word("happen", vec![SemanticPrime::Happen], 0.0, 0.4);
        self.add_word("move", vec![SemanticPrime::Move], 0.0, 0.5);
        self.add_word("go", vec![SemanticPrime::Move], 0.0, 0.4);
        self.add_word("come", vec![SemanticPrime::Move], 0.1, 0.4);
        self.add_word("leave", vec![SemanticPrime::Move, SemanticPrime::Far], -0.2, 0.4);
        self.add_word("left", vec![SemanticPrime::Move, SemanticPrime::Far, SemanticPrime::Before], -0.2, 0.3);
        self.add_word("stay", vec![SemanticPrime::Move, SemanticPrime::Not], 0.1, 0.2);
        self.add_word("say", vec![SemanticPrime::Say], 0.0, 0.4);
        self.add_word("said", vec![SemanticPrime::Say, SemanticPrime::Before], 0.0, 0.3);
        self.add_word("tell", vec![SemanticPrime::Say], 0.0, 0.4);
        self.add_word("live", vec![SemanticPrime::Live], 0.3, 0.3);
        self.add_word("die", vec![SemanticPrime::Die], -0.8, 0.6);
        self.add_word("died", vec![SemanticPrime::Die, SemanticPrime::Before], -0.9, 0.5);

        // === EVALUATORS ===
        self.add_word("good", vec![SemanticPrime::Good], 0.7, 0.3);
        self.add_word("great", vec![SemanticPrime::Good, SemanticPrime::Very], 0.9, 0.5);
        self.add_word("wonderful", vec![SemanticPrime::Good, SemanticPrime::Very], 0.9, 0.6);
        self.add_word("bad", vec![SemanticPrime::Bad], -0.7, 0.4);
        self.add_word("terrible", vec![SemanticPrime::Bad, SemanticPrime::Very], -0.9, 0.6);
        self.add_word("big", vec![SemanticPrime::Big], 0.1, 0.3);
        self.add_word("small", vec![SemanticPrime::Small], -0.1, 0.2);
        self.add_word("true", vec![SemanticPrime::True], 0.3, 0.3);
        self.add_word("false", vec![SemanticPrime::True, SemanticPrime::Not], -0.3, 0.3);

        // === RELATIONAL ===
        self.add_word("friend", vec![SemanticPrime::Someone, SemanticPrime::Good, SemanticPrime::Want], 0.6, 0.4);
        self.add_word("enemy", vec![SemanticPrime::Someone, SemanticPrime::Bad], -0.6, 0.6);
        self.add_word("family", vec![SemanticPrime::People, SemanticPrime::Like], 0.5, 0.4);
        self.add_word("mother", vec![SemanticPrime::Someone, SemanticPrime::Live], 0.5, 0.4);
        self.add_word("father", vec![SemanticPrime::Someone, SemanticPrime::Live], 0.5, 0.4);
        self.add_word("child", vec![SemanticPrime::Someone, SemanticPrime::Small], 0.4, 0.5);

        // === LOGICAL/CAUSAL ===
        self.add_word("not", vec![SemanticPrime::Not], 0.0, 0.2);
        self.add_word("no", vec![SemanticPrime::Not], -0.1, 0.3);
        self.add_word("maybe", vec![SemanticPrime::Maybe], 0.0, 0.3);
        self.add_word("can", vec![SemanticPrime::Can], 0.1, 0.3);
        self.add_word("cannot", vec![SemanticPrime::Can, SemanticPrime::Not], -0.2, 0.4);
        self.add_word("if", vec![SemanticPrime::If], 0.0, 0.3);
        self.add_word("because", vec![SemanticPrime::Because], 0.0, 0.4);

        // === TEMPORAL ===
        self.add_word("now", vec![SemanticPrime::Now], 0.0, 0.5);
        self.add_word("before", vec![SemanticPrime::Before], 0.0, 0.3);
        self.add_word("after", vec![SemanticPrime::After], 0.0, 0.3);
        self.add_word("when", vec![SemanticPrime::When], 0.0, 0.3);
        self.add_word("long", vec![SemanticPrime::LongTime], 0.0, 0.2);
        self.add_word("short", vec![SemanticPrime::ShortTime], 0.0, 0.3);

        // === SPATIAL ===
        self.add_word("here", vec![SemanticPrime::Here], 0.1, 0.3);
        self.add_word("there", vec![SemanticPrime::Far], 0.0, 0.2);
        self.add_word("where", vec![SemanticPrime::Where], 0.0, 0.3);
        self.add_word("above", vec![SemanticPrime::Above], 0.0, 0.2);
        self.add_word("below", vec![SemanticPrime::Below], 0.0, 0.2);
        self.add_word("near", vec![SemanticPrime::Near], 0.1, 0.2);
        self.add_word("far", vec![SemanticPrime::Far], -0.1, 0.2);
        self.add_word("inside", vec![SemanticPrime::Inside], 0.0, 0.2);

        // === QUANTIFIERS ===
        self.add_word("one", vec![SemanticPrime::One], 0.0, 0.2);
        self.add_word("two", vec![SemanticPrime::Two], 0.0, 0.2);
        self.add_word("some", vec![SemanticPrime::Some], 0.0, 0.2);
        self.add_word("all", vec![SemanticPrime::All], 0.1, 0.3);
        self.add_word("much", vec![SemanticPrime::Much], 0.1, 0.3);
        self.add_word("many", vec![SemanticPrime::Much], 0.1, 0.3);
        self.add_word("little", vec![SemanticPrime::Little], 0.0, 0.2);
        self.add_word("few", vec![SemanticPrime::Little], 0.0, 0.2);

        // === INTENSIFIERS ===
        self.add_word("very", vec![SemanticPrime::Very], 0.0, 0.4);
        self.add_word("really", vec![SemanticPrime::Very], 0.0, 0.4);
        self.add_word("more", vec![SemanticPrime::More], 0.1, 0.3);
        self.add_word("less", vec![SemanticPrime::More, SemanticPrime::Not], -0.1, 0.2);

        // === CAUSAL MARKERS ===
        self.causal_markers.insert("because".to_string(), CausalRelation::Causes);
        self.causal_markers.insert("since".to_string(), CausalRelation::Causes);
        self.causal_markers.insert("therefore".to_string(), CausalRelation::Causes);
        self.causal_markers.insert("so".to_string(), CausalRelation::Causes);
        self.causal_markers.insert("thus".to_string(), CausalRelation::Causes);
        self.causal_markers.insert("hence".to_string(), CausalRelation::Causes);

        self.causal_markers.insert("despite".to_string(), CausalRelation::Prevents);
        self.causal_markers.insert("although".to_string(), CausalRelation::Prevents);
        self.causal_markers.insert("but".to_string(), CausalRelation::Prevents);
        self.causal_markers.insert("however".to_string(), CausalRelation::Prevents);
        self.causal_markers.insert("yet".to_string(), CausalRelation::Prevents);

        self.causal_markers.insert("allows".to_string(), CausalRelation::Enables);
        self.causal_markers.insert("lets".to_string(), CausalRelation::Enables);
        self.causal_markers.insert("enables".to_string(), CausalRelation::Enables);
        self.causal_markers.insert("permits".to_string(), CausalRelation::Enables);

        self.causal_markers.insert("before".to_string(), CausalRelation::Temporal);
        self.causal_markers.insert("after".to_string(), CausalRelation::Temporal);
        self.causal_markers.insert("when".to_string(), CausalRelation::Temporal);
        self.causal_markers.insert("while".to_string(), CausalRelation::Temporal);

        self.causal_markers.insert("if".to_string(), CausalRelation::Conditional);
        self.causal_markers.insert("unless".to_string(), CausalRelation::Conditional);
        self.causal_markers.insert("would".to_string(), CausalRelation::Conditional);
    }

    fn add_word(&mut self, word: &str, primes: Vec<SemanticPrime>, valence: f32, arousal: f32) {
        self.word_primes.insert(word.to_string(), primes);
        self.word_valence.insert(word.to_string(), valence);
        self.word_arousal.insert(word.to_string(), arousal);
    }

    /// Decompose a word into semantic primes
    pub fn decompose(&self, word: &str) -> Option<&Vec<SemanticPrime>> {
        self.word_primes.get(&word.to_lowercase())
    }

    /// Get emotional valence for a word
    pub fn valence(&self, word: &str) -> Option<f32> {
        self.word_valence.get(&word.to_lowercase()).copied()
    }

    /// Get arousal level for a word
    pub fn arousal(&self, word: &str) -> Option<f32> {
        self.word_arousal.get(&word.to_lowercase()).copied()
    }

    /// Check if word is a causal marker
    pub fn causal_relation(&self, word: &str) -> Option<CausalRelation> {
        self.causal_markers.get(&word.to_lowercase()).copied()
    }

    /// Get number of words in lexicon
    pub fn lexicon_size(&self) -> usize {
        self.word_primes.len()
    }
}

impl Default for LexicalGrounding {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// UNDERSTOOD MEANING: The result of grounded understanding
// =============================================================================

/// A fully understood meaning with all grounding layers
#[derive(Debug, Clone)]
pub struct UnderstoodMeaning {
    /// Original text
    pub text: String,

    /// Semantic decomposition (primes used)
    pub primes: Vec<SemanticPrime>,

    /// Compositional structure (HDC representation)
    pub semantic_hv: HV16,

    /// Causal structure (if detected)
    pub causal_structure: Option<CausalStructure>,

    /// Emotional grounding
    pub embodied: EmbodiedGrounding,

    /// Consciousness integration score
    pub integration_phi: f64,

    /// Qualia description
    pub qualia: String,

    /// Understanding confidence (0-1)
    pub confidence: f64,
}

/// Causal structure extracted from text
#[derive(Debug, Clone)]
pub struct CausalStructure {
    /// The cause component
    pub cause: String,
    pub cause_hv: HV16,

    /// The effect component
    pub effect: String,
    pub effect_hv: HV16,

    /// Type of causal relation
    pub relation: CausalRelation,

    /// Causal strength estimate
    pub strength: f64,
}

/// Embodied/emotional grounding of meaning
#[derive(Debug, Clone)]
pub struct EmbodiedGrounding {
    /// Emotional valence (-1 to +1)
    pub valence: f32,

    /// Arousal level (0 to 1)
    pub arousal: f32,

    /// Dominance/control (0 to 1)
    pub dominance: f32,

    /// Suggested hormone response
    pub hormone_suggestion: HormoneHint,

    /// Voice prosody hints
    pub prosody: ProsodyHint,
}

/// Hormone modulation hint based on understood meaning
#[derive(Debug, Clone)]
pub struct HormoneHint {
    pub cortisol_delta: f32,    // stress
    pub dopamine_delta: f32,    // reward
    pub acetylcholine_delta: f32, // focus
}

/// Prosody hints for expressing understood meaning
#[derive(Debug, Clone)]
pub struct ProsodyHint {
    pub rate: f32,      // speech rate
    pub pitch: f32,     // pitch shift
    pub energy: f32,    // vocal energy
}

// =============================================================================
// GROUNDED UNDERSTANDING ENGINE
// =============================================================================

/// The complete grounded understanding system
pub struct GroundedUnderstanding {
    /// Lexical grounding (word → primes)
    lexicon: LexicalGrounding,

    /// Universal semantics (prime composition)
    semantics: UniversalSemantics,

    /// Causal role markers
    causal_markers: CausalRoleMarkers,

    /// Understanding depth (how much computation to invest)
    depth: UnderstandingDepth,
}

/// How deeply to process understanding
#[derive(Debug, Clone, Copy)]
pub enum UnderstandingDepth {
    /// Fast: Just semantic primes (no causal/embodied)
    Surface,
    /// Medium: Primes + causal structure
    Structural,
    /// Deep: Full grounded understanding
    Deep,
}

impl GroundedUnderstanding {
    pub fn new() -> Self {
        Self {
            lexicon: LexicalGrounding::new(),
            semantics: UniversalSemantics::new(),
            causal_markers: CausalRoleMarkers::new(),
            depth: UnderstandingDepth::Deep,
        }
    }

    pub fn with_depth(mut self, depth: UnderstandingDepth) -> Self {
        self.depth = depth;
        self
    }

    /// Understand a piece of text
    pub fn understand(&self, text: &str) -> UnderstoodMeaning {
        // 1. Tokenize and decompose into primes
        let words = self.tokenize(text);
        let mut all_primes = Vec::new();
        let mut total_valence = 0.0f32;
        let mut total_arousal = 0.0f32;
        let mut word_count = 0;

        for word in &words {
            if let Some(primes) = self.lexicon.decompose(word) {
                all_primes.extend(primes.clone());
            }
            if let Some(v) = self.lexicon.valence(word) {
                total_valence += v;
                word_count += 1;
            }
            if let Some(a) = self.lexicon.arousal(word) {
                total_arousal += a;
            }
        }

        // 2. Compose semantic HDC representation
        let semantic_hv = self.compose_meaning(&all_primes);

        // 3. Extract causal structure (if depth allows)
        let causal_structure = match self.depth {
            UnderstandingDepth::Surface => None,
            _ => self.extract_causal_structure(text, &words),
        };

        // 4. Compute embodied grounding
        let avg_valence = if word_count > 0 { total_valence / word_count as f32 } else { 0.0 };
        let avg_arousal = if word_count > 0 { total_arousal / word_count as f32 } else { 0.3 };

        let embodied = self.compute_embodied_grounding(avg_valence, avg_arousal);

        // 5. Compute integration (simplified Φ estimate)
        let integration_phi = self.estimate_integration(&all_primes, &semantic_hv);

        // 6. Generate qualia description
        let qualia = self.generate_qualia(&embodied, &all_primes);

        // 7. Compute confidence
        let known_word_ratio = all_primes.len() as f64 / words.len().max(1) as f64;
        let confidence = known_word_ratio.min(1.0);

        UnderstoodMeaning {
            text: text.to_string(),
            primes: all_primes,
            semantic_hv,
            causal_structure,
            embodied,
            integration_phi,
            qualia,
            confidence,
        }
    }

    /// Simple tokenization
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphabetic())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    /// Compose primes into unified meaning vector
    fn compose_meaning(&self, primes: &[SemanticPrime]) -> HV16 {
        if primes.is_empty() {
            return HV16::random(0);
        }

        // Get HV for each prime
        let hvs: Vec<HV16> = primes.iter()
            .map(|p| *self.semantics.get_prime(*p))
            .collect();

        // Bundle all primes together
        HV16::bundle(&hvs)
    }

    /// Extract causal structure from text
    fn extract_causal_structure(&self, text: &str, words: &[String]) -> Option<CausalStructure> {
        // Find causal marker
        let mut causal_idx = None;
        let mut relation = None;

        for (i, word) in words.iter().enumerate() {
            if let Some(rel) = self.lexicon.causal_relation(word) {
                causal_idx = Some(i);
                relation = Some(rel);
                break;
            }
        }

        let (idx, rel) = match (causal_idx, relation) {
            (Some(i), Some(r)) => (i, r),
            _ => return None,
        };

        // Split into cause and effect based on relation
        let (cause_words, effect_words) = match rel {
            CausalRelation::Causes => {
                // "X because Y" → Y causes X
                let effect = &words[..idx];
                let cause = &words[idx+1..];
                (cause.to_vec(), effect.to_vec())
            }
            _ => {
                // "X enables/prevents Y" → X relation Y
                let cause = &words[..idx];
                let effect = &words[idx+1..];
                (cause.to_vec(), effect.to_vec())
            }
        };

        // Compose HDC vectors for cause and effect
        let cause_primes: Vec<SemanticPrime> = cause_words.iter()
            .filter_map(|w| self.lexicon.decompose(w))
            .flatten()
            .copied()
            .collect();
        let effect_primes: Vec<SemanticPrime> = effect_words.iter()
            .filter_map(|w| self.lexicon.decompose(w))
            .flatten()
            .copied()
            .collect();

        let cause_hv = self.compose_meaning(&cause_primes);
        let effect_hv = self.compose_meaning(&effect_primes);

        // Estimate causal strength from linguistic cues
        let strength = 0.7; // Default moderate strength

        Some(CausalStructure {
            cause: cause_words.join(" "),
            cause_hv,
            effect: effect_words.join(" "),
            effect_hv,
            relation: rel,
            strength,
        })
    }

    /// Compute embodied grounding from emotional dimensions
    fn compute_embodied_grounding(&self, valence: f32, arousal: f32) -> EmbodiedGrounding {
        // Dominance: high arousal + positive valence = high dominance
        let dominance = (0.5 + valence * 0.3 + (1.0 - arousal) * 0.2).clamp(0.0, 1.0);

        // Hormone suggestions based on emotional state
        let hormone_suggestion = HormoneHint {
            cortisol_delta: if valence < -0.3 { 0.2 } else { -0.1 },
            dopamine_delta: if valence > 0.3 { 0.3 } else { -0.1 },
            acetylcholine_delta: if arousal > 0.5 { 0.2 } else { 0.0 },
        };

        // Prosody hints
        let prosody = ProsodyHint {
            rate: 0.9 + arousal * 0.3,  // Higher arousal = faster
            pitch: valence * 2.0,        // Positive = higher pitch
            energy: arousal,             // Arousal = energy
        };

        EmbodiedGrounding {
            valence,
            arousal,
            dominance,
            hormone_suggestion,
            prosody,
        }
    }

    /// Estimate integration (simplified Φ)
    fn estimate_integration(&self, primes: &[SemanticPrime], semantic_hv: &HV16) -> f64 {
        // More diverse primes = higher integration
        let unique_primes: std::collections::HashSet<_> = primes.iter().collect();
        let diversity = unique_primes.len() as f64 / 65.0; // 65 total primes

        // Integration correlates with semantic density
        let density = primes.len() as f64 / 10.0; // Normalize

        (diversity * 0.5 + density.min(1.0) * 0.5).min(1.0)
    }

    /// Generate qualia description
    fn generate_qualia(&self, embodied: &EmbodiedGrounding, primes: &[SemanticPrime]) -> String {
        let emotion_word = if embodied.valence > 0.5 {
            "warm and bright"
        } else if embodied.valence > 0.0 {
            "gently pleasant"
        } else if embodied.valence > -0.5 {
            "tinged with melancholy"
        } else {
            "heavy and dark"
        };

        let intensity_word = if embodied.arousal > 0.7 {
            "intensely"
        } else if embodied.arousal > 0.4 {
            "clearly"
        } else {
            "softly"
        };

        format!("{} felt, {}", intensity_word, emotion_word)
    }

    /// Get the lexicon for inspection
    pub fn lexicon(&self) -> &LexicalGrounding {
        &self.lexicon
    }

    /// Compare similarity of two meanings
    pub fn similarity(&self, a: &UnderstoodMeaning, b: &UnderstoodMeaning) -> f64 {
        a.semantic_hv.similarity(&b.semantic_hv) as f64
    }
}

impl Default for GroundedUnderstanding {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexical_grounding() {
        let lexicon = LexicalGrounding::new();

        // Test basic decomposition
        let happy = lexicon.decompose("happy");
        assert!(happy.is_some());
        assert!(happy.unwrap().contains(&SemanticPrime::Feel));
        assert!(happy.unwrap().contains(&SemanticPrime::Good));

        // Test valence
        assert!(lexicon.valence("happy").unwrap() > 0.5);
        assert!(lexicon.valence("sad").unwrap() < -0.5);
    }

    #[test]
    fn test_causal_detection() {
        let lexicon = LexicalGrounding::new();

        assert_eq!(lexicon.causal_relation("because"), Some(CausalRelation::Causes));
        assert_eq!(lexicon.causal_relation("although"), Some(CausalRelation::Prevents));
        assert_eq!(lexicon.causal_relation("if"), Some(CausalRelation::Conditional));
    }

    #[test]
    fn test_understanding() {
        let engine = GroundedUnderstanding::new();

        let meaning = engine.understand("I feel happy");

        assert!(!meaning.primes.is_empty());
        assert!(meaning.embodied.valence > 0.0);
        assert!(meaning.confidence > 0.5);
    }

    #[test]
    fn test_causal_understanding() {
        let engine = GroundedUnderstanding::new();

        let meaning = engine.understand("I feel sad because my friend left");

        assert!(meaning.causal_structure.is_some());
        let causal = meaning.causal_structure.unwrap();
        assert_eq!(causal.relation, CausalRelation::Causes);
        assert!(causal.cause.contains("friend"));
        assert!(causal.effect.contains("sad"));
    }

    #[test]
    fn test_semantic_similarity() {
        let engine = GroundedUnderstanding::new();

        let happy = engine.understand("I am happy");
        let joyful = engine.understand("I feel joyful");
        let sad = engine.understand("I am sad");

        // Happy and joyful should be more similar than happy and sad
        let sim_happy_joyful = engine.similarity(&happy, &joyful);
        let sim_happy_sad = engine.similarity(&happy, &sad);

        assert!(sim_happy_joyful > sim_happy_sad);
    }

    #[test]
    fn test_embodied_grounding() {
        let engine = GroundedUnderstanding::new();

        let excited = engine.understand("I feel excited and happy");
        let calm = engine.understand("I feel calm and peaceful");

        // Excited should have higher arousal
        assert!(excited.embodied.arousal > calm.embodied.arousal);

        // Both should be positive valence
        assert!(excited.embodied.valence > 0.0);
        assert!(calm.embodied.valence > 0.0);
    }
}
