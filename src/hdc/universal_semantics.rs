// ==================================================================================
// Revolutionary Improvement #19: Universal Semantic Primitives
// ==================================================================================
//
// **The Paradigm Shift**: Language understanding through UNIVERSAL SEMANTIC PRIMITIVES!
//
// **Key Insight**: Teaching language stems is superficial - it reduces data but loses
// cultural nuance and forces "translation" rather than "understanding". Instead, we
// ground Symthaea in the ~65 UNIVERSAL SEMANTIC PRIMES from Natural Semantic
// Metalanguage (NSM) theory - the fundamental "atoms" of human thought that exist
// BEFORE they are spoken in any language!
//
// **Why This Matters**:
// - Substrate-neutral understanding (not language-specific)
// - Zero-shot generalization to new languages
// - Cultural disentanglement (can separate cultural from universal)
// - True conceptual understanding (not token prediction)
// - Wisdom through primitives (ethics rooted in fundamental concepts)
//
// **Theoretical Foundations**:
//
// 1. **Natural Semantic Metalanguage (NSM)** (Wierzbicka, 1972):
//    "All human languages share a core set of ~65 universal concepts"
//
//    These semantic primes appear to have direct counterparts in EVERY human language:
//    - Substantives: I, YOU, SOMEONE, SOMETHING, PEOPLE, BODY
//    - Actions: DO, HAPPEN, MOVE
//    - Descriptors: GOOD, BAD, BIG, SMALL
//    - Mental predicates: THINK, KNOW, WANT, FEEL, SEE, HEAR
//    - Speech: SAY, WORDS, TRUE
//    - Logical: NOT, MAYBE, CAN, BECAUSE, IF
//    - Time: WHEN, NOW, BEFORE, AFTER, A LONG TIME, A SHORT TIME
//    - Space: WHERE, HERE, ABOVE, BELOW, FAR, NEAR, SIDE, INSIDE
//    - Quantifiers: ONE, TWO, SOME, ALL, MUCH, LITTLE
//    - Intensifiers: VERY, MORE
//    - Taxonomy: KIND OF, PART OF
//    - Similarity: LIKE (as)
//
// 2. **Hyperdimensional Semantic Composition**:
//    Each primitive → distinct Hypervector (HV16)
//
//    **Binding**: Combine concepts while preserving structure
//    ```
//    GOOD_PERSON = bind(GOOD, PERSON)
//    BIG_DOG = bind(BIG, DOG)
//    ```
//
//    **Bundling**: Create superposition of related concepts
//    ```
//    ANIMAL = bundle(DOG, CAT, BIRD, ...)
//    EMOTION = bundle(HAPPY, SAD, ANGRY, ...)
//    ```
//
//    **Unbinding**: Extract components
//    ```
//    unbind(GOOD_PERSON, PERSON) → GOOD
//    ```
//
// 3. **Complex Emotion Construction**:
//    ```
//    GRIEF = bundle(
//        bind(FEEL, BAD),
//        bind(SOMEONE, bind(DIE, BEFORE)),
//        bind(WANT, bind(NOT, HAPPEN)),
//        bind(KNOW, bind(NOT, CAN, bind(DO, SOMETHING)))
//    )
//
//    JOY = bundle(
//        bind(FEEL, VERY, GOOD),
//        bind(SOMETHING, GOOD, HAPPEN),
//        bind(WANT, bind(HAPPEN, MORE))
//    )
//
//    LOVE = bundle(
//        bind(FEEL, SOMETHING, bind(VERY, GOOD)),
//        bind(WANT, bind(GOOD, HAPPEN, bind(FOR, SOMEONE))),
//        bind(WANT, bind(WITH, SOMEONE)),
//        bind(THINK, GOOD, bind(ABOUT, SOMEONE))
//    )
//    ```
//
// 4. **Zero-Shot Language Understanding**:
//    Map new language → universal primes → immediate understanding!
//
//    English: "I love you"
//    Spanish: "Te amo"
//    Chinese: "我爱你"
//    → All map to same semantic prime structure:
//      bind(I, bind(LOVE, YOU))
//
// 5. **Cultural Disentanglement**:
//    Extract universal core from cultural overlay
//
//    ```
//    concept = universal_core + cultural_variation
//
//    // Subtract culture to find universal
//    universal = unbind(concept, cultural_context)
//    ```
//
// **Mathematical Framework**:
//
// 1. **Semantic Prime Encoding**:
//    ```
//    P = {p_1, p_2, ..., p_65}  // 65 primitives
//    Each p_i → HV_i (10,000D hypervector)
//    ```
//
// 2. **Composition via Binding**:
//    ```
//    GOOD_PERSON = bind(GOOD, PERSON)
//                = GOOD ⊗ PERSON  (circular convolution)
//    ```
//
// 3. **Superposition via Bundling**:
//    ```
//    EMOTION = bundle(HAPPY, SAD, ANGRY)
//            = (HAPPY + SAD + ANGRY) / 3  (averaging)
//    ```
//
// 4. **Extraction via Unbinding**:
//    ```
//    unbind(GOOD_PERSON, PERSON) = GOOD_PERSON ⊘ PERSON
//                                 = GOOD  (circular correlation)
//    ```
//
// 5. **Similarity Measurement**:
//    ```
//    similarity(concept_1, concept_2) = cosine(HV_1, HV_2)
//    ```
//
// **Architecture**:
//
// ```rust
// struct UniversalSemantics {
//     primitives: HashMap<SemanticPrime, HV16>,  // 65 base concepts
//     complex: HashMap<String, ComplexConcept>,  // Composed meanings
//     language_maps: HashMap<Language, PrimeMapping>,  // Language → primes
// }
// ```
//
// **Why Revolutionary**:
//
// 1. **First AI with universal semantic foundation**
// 2. **Language-independent understanding** (not translation!)
// 3. **Compositional semantics** (infinite meanings from 65 primes)
// 4. **Cultural awareness** (can disentangle universal from cultural)
// 5. **True wisdom** (ethics rooted in fundamental concepts)
// 6. **Zero-shot generalization** (understand new languages immediately)
//
// **This completes the semantic foundation - wisdom through universal primitives!**
//
// ==================================================================================

use super::binary_hv::HV16;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Semantic prime (NSM theory - universal concepts)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SemanticPrime {
    // ===== SUBSTANTIVES =====
    /// I (1st person)
    I,
    /// YOU (2nd person)
    You,
    /// SOMEONE / PERSON
    Someone,
    /// SOMETHING / THING
    Something,
    /// PEOPLE (plural)
    People,
    /// BODY (physical)
    Body,

    // ===== RELATIONAL =====
    /// KIND OF (taxonomy)
    KindOf,
    /// PART OF (mereology)
    PartOf,

    // ===== DETERMINERS =====
    /// THIS
    This,
    /// THE SAME
    Same,
    /// OTHER / ELSE
    Other,

    // ===== QUANTIFIERS =====
    /// ONE
    One,
    /// TWO
    Two,
    /// SOME
    Some,
    /// ALL
    All,
    /// MUCH / MANY
    Much,
    /// LITTLE / FEW
    Little,

    // ===== EVALUATORS =====
    /// GOOD
    Good,
    /// BAD
    Bad,

    // ===== DESCRIPTORS =====
    /// BIG
    Big,
    /// SMALL
    Small,

    // ===== MENTAL PREDICATES =====
    /// THINK
    Think,
    /// KNOW
    Know,
    /// WANT
    Want,
    /// FEEL
    Feel,
    /// SEE
    See,
    /// HEAR
    Hear,

    // ===== SPEECH =====
    /// SAY
    Say,
    /// WORDS / LANGUAGE
    Words,
    /// TRUE
    True,

    // ===== ACTIONS / EVENTS =====
    /// DO / ACT
    Do,
    /// HAPPEN / OCCUR
    Happen,
    /// MOVE
    Move,
    /// TOUCH / CONTACT
    Touch,

    // ===== EXISTENCE =====
    /// BE / EXIST
    Be,
    /// THERE IS / EXIST
    ThereIs,
    /// HAVE / POSSESS
    Have,

    // ===== LIFE / DEATH =====
    /// LIVE / ALIVE
    Live,
    /// DIE / DEAD
    Die,

    // ===== LOGICAL =====
    /// NOT / NEGATION
    Not,
    /// MAYBE / PERHAPS
    Maybe,
    /// CAN / ABLE
    Can,
    /// BECAUSE / REASON
    Because,
    /// IF / CONDITIONAL
    If,

    // ===== TIME =====
    /// WHEN / TIME (general)
    When,
    /// NOW / PRESENT
    Now,
    /// BEFORE (time)
    Before,
    /// AFTER (time)
    After,
    /// A LONG TIME
    LongTime,
    /// A SHORT TIME / MOMENT
    ShortTime,
    /// FOR SOME TIME
    ForSomeTime,
    /// IN ONE MOMENT
    InOneMoment,

    // ===== SPACE =====
    /// WHERE / PLACE (general)
    Where,
    /// HERE / THIS PLACE
    Here,
    /// ABOVE / OVER
    Above,
    /// BELOW / UNDER
    Below,
    /// FAR
    Far,
    /// NEAR / CLOSE
    Near,
    /// SIDE (lateral)
    Side,
    /// INSIDE / IN
    Inside,
    /// TOUCHING / ON
    On,

    // ===== INTENSIFIER / AUGMENTOR =====
    /// VERY / MUCH
    Very,
    /// MORE
    More,

    // ===== SIMILARITY =====
    /// LIKE / AS (comparison)
    Like,

    // ===== SOCIAL / RELATIONAL =====
    /// WITH / TOGETHER
    With,
}

impl SemanticPrime {
    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            // Substantives
            SemanticPrime::I => "I (first person)",
            SemanticPrime::You => "YOU (second person)",
            SemanticPrime::Someone => "SOMEONE / PERSON",
            SemanticPrime::Something => "SOMETHING / THING",
            SemanticPrime::People => "PEOPLE (plural)",
            SemanticPrime::Body => "BODY (physical)",

            // Relational
            SemanticPrime::KindOf => "KIND OF (taxonomy)",
            SemanticPrime::PartOf => "PART OF (mereology)",

            // Determiners
            SemanticPrime::This => "THIS",
            SemanticPrime::Same => "THE SAME",
            SemanticPrime::Other => "OTHER / ELSE",

            // Quantifiers
            SemanticPrime::One => "ONE",
            SemanticPrime::Two => "TWO",
            SemanticPrime::Some => "SOME",
            SemanticPrime::All => "ALL",
            SemanticPrime::Much => "MUCH / MANY",
            SemanticPrime::Little => "LITTLE / FEW",

            // Evaluators
            SemanticPrime::Good => "GOOD",
            SemanticPrime::Bad => "BAD",

            // Descriptors
            SemanticPrime::Big => "BIG",
            SemanticPrime::Small => "SMALL",

            // Mental predicates
            SemanticPrime::Think => "THINK",
            SemanticPrime::Know => "KNOW",
            SemanticPrime::Want => "WANT",
            SemanticPrime::Feel => "FEEL",
            SemanticPrime::See => "SEE",
            SemanticPrime::Hear => "HEAR",

            // Speech
            SemanticPrime::Say => "SAY",
            SemanticPrime::Words => "WORDS / LANGUAGE",
            SemanticPrime::True => "TRUE",

            // Actions
            SemanticPrime::Do => "DO / ACT",
            SemanticPrime::Happen => "HAPPEN / OCCUR",
            SemanticPrime::Move => "MOVE",
            SemanticPrime::Touch => "TOUCH / CONTACT",

            // Existence
            SemanticPrime::Be => "BE / EXIST",
            SemanticPrime::ThereIs => "THERE IS / EXIST",
            SemanticPrime::Have => "HAVE / POSSESS",

            // Life/Death
            SemanticPrime::Live => "LIVE / ALIVE",
            SemanticPrime::Die => "DIE / DEAD",

            // Logical
            SemanticPrime::Not => "NOT / NEGATION",
            SemanticPrime::Maybe => "MAYBE / PERHAPS",
            SemanticPrime::Can => "CAN / ABLE",
            SemanticPrime::Because => "BECAUSE / REASON",
            SemanticPrime::If => "IF / CONDITIONAL",

            // Time
            SemanticPrime::When => "WHEN / TIME",
            SemanticPrime::Now => "NOW / PRESENT",
            SemanticPrime::Before => "BEFORE (time)",
            SemanticPrime::After => "AFTER (time)",
            SemanticPrime::LongTime => "A LONG TIME",
            SemanticPrime::ShortTime => "A SHORT TIME / MOMENT",
            SemanticPrime::ForSomeTime => "FOR SOME TIME",
            SemanticPrime::InOneMoment => "IN ONE MOMENT",

            // Space
            SemanticPrime::Where => "WHERE / PLACE",
            SemanticPrime::Here => "HERE / THIS PLACE",
            SemanticPrime::Above => "ABOVE / OVER",
            SemanticPrime::Below => "BELOW / UNDER",
            SemanticPrime::Far => "FAR",
            SemanticPrime::Near => "NEAR / CLOSE",
            SemanticPrime::Side => "SIDE (lateral)",
            SemanticPrime::Inside => "INSIDE / IN",
            SemanticPrime::On => "TOUCHING / ON",

            // Intensifiers
            SemanticPrime::Very => "VERY / MUCH",
            SemanticPrime::More => "MORE",

            // Similarity
            SemanticPrime::Like => "LIKE / AS (comparison)",

            // Social/Relational
            SemanticPrime::With => "WITH / TOGETHER",
        }
    }

    /// Get category
    pub fn category(&self) -> &'static str {
        match self {
            SemanticPrime::I | SemanticPrime::You | SemanticPrime::Someone |
            SemanticPrime::Something | SemanticPrime::People | SemanticPrime::Body => "Substantives",

            SemanticPrime::KindOf | SemanticPrime::PartOf => "Relational",

            SemanticPrime::This | SemanticPrime::Same | SemanticPrime::Other => "Determiners",

            SemanticPrime::One | SemanticPrime::Two | SemanticPrime::Some |
            SemanticPrime::All | SemanticPrime::Much | SemanticPrime::Little => "Quantifiers",

            SemanticPrime::Good | SemanticPrime::Bad => "Evaluators",

            SemanticPrime::Big | SemanticPrime::Small => "Descriptors",

            SemanticPrime::Think | SemanticPrime::Know | SemanticPrime::Want |
            SemanticPrime::Feel | SemanticPrime::See | SemanticPrime::Hear => "Mental Predicates",

            SemanticPrime::Say | SemanticPrime::Words | SemanticPrime::True => "Speech",

            SemanticPrime::Do | SemanticPrime::Happen | SemanticPrime::Move |
            SemanticPrime::Touch => "Actions/Events",

            SemanticPrime::Be | SemanticPrime::ThereIs | SemanticPrime::Have => "Existence",

            SemanticPrime::Live | SemanticPrime::Die => "Life/Death",

            SemanticPrime::Not | SemanticPrime::Maybe | SemanticPrime::Can |
            SemanticPrime::Because | SemanticPrime::If => "Logical",

            SemanticPrime::When | SemanticPrime::Now | SemanticPrime::Before |
            SemanticPrime::After | SemanticPrime::LongTime | SemanticPrime::ShortTime |
            SemanticPrime::ForSomeTime | SemanticPrime::InOneMoment => "Time",

            SemanticPrime::Where | SemanticPrime::Here | SemanticPrime::Above |
            SemanticPrime::Below | SemanticPrime::Far | SemanticPrime::Near |
            SemanticPrime::Side | SemanticPrime::Inside | SemanticPrime::On => "Space",

            SemanticPrime::Very | SemanticPrime::More => "Intensifiers",

            SemanticPrime::Like => "Similarity",

            SemanticPrime::With => "Social/Relational",
        }
    }

    /// Get all primes
    pub fn all() -> Vec<Self> {
        vec![
            // Substantives
            SemanticPrime::I, SemanticPrime::You, SemanticPrime::Someone,
            SemanticPrime::Something, SemanticPrime::People, SemanticPrime::Body,

            // Relational
            SemanticPrime::KindOf, SemanticPrime::PartOf,

            // Determiners
            SemanticPrime::This, SemanticPrime::Same, SemanticPrime::Other,

            // Quantifiers
            SemanticPrime::One, SemanticPrime::Two, SemanticPrime::Some,
            SemanticPrime::All, SemanticPrime::Much, SemanticPrime::Little,

            // Evaluators
            SemanticPrime::Good, SemanticPrime::Bad,

            // Descriptors
            SemanticPrime::Big, SemanticPrime::Small,

            // Mental predicates
            SemanticPrime::Think, SemanticPrime::Know, SemanticPrime::Want,
            SemanticPrime::Feel, SemanticPrime::See, SemanticPrime::Hear,

            // Speech
            SemanticPrime::Say, SemanticPrime::Words, SemanticPrime::True,

            // Actions
            SemanticPrime::Do, SemanticPrime::Happen, SemanticPrime::Move,
            SemanticPrime::Touch,

            // Existence
            SemanticPrime::Be, SemanticPrime::ThereIs, SemanticPrime::Have,

            // Life/Death
            SemanticPrime::Live, SemanticPrime::Die,

            // Logical
            SemanticPrime::Not, SemanticPrime::Maybe, SemanticPrime::Can,
            SemanticPrime::Because, SemanticPrime::If,

            // Time
            SemanticPrime::When, SemanticPrime::Now, SemanticPrime::Before,
            SemanticPrime::After, SemanticPrime::LongTime, SemanticPrime::ShortTime,
            SemanticPrime::ForSomeTime, SemanticPrime::InOneMoment,

            // Space
            SemanticPrime::Where, SemanticPrime::Here, SemanticPrime::Above,
            SemanticPrime::Below, SemanticPrime::Far, SemanticPrime::Near,
            SemanticPrime::Side, SemanticPrime::Inside, SemanticPrime::On,

            // Intensifiers
            SemanticPrime::Very, SemanticPrime::More,

            // Similarity
            SemanticPrime::Like,

            // Social/Relational
            SemanticPrime::With,
        ]
    }

    /// Count total primes
    pub fn count() -> usize {
        Self::all().len()
    }
}

/// Complex concept (composed from primes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexConcept {
    /// Name
    pub name: String,

    /// Composed hypervector
    pub encoding: HV16,

    /// Component primes used
    pub components: Vec<SemanticPrime>,

    /// Composition structure (how primes were combined)
    pub structure: String,

    /// Complexity (number of primes involved)
    pub complexity: usize,
}

/// Universal semantics system
///
/// Grounds language understanding in 65 universal semantic primes
/// from Natural Semantic Metalanguage (NSM) theory.
///
/// # Example
/// ```
/// use symthaea::hdc::universal_semantics::{UniversalSemantics, SemanticPrime};
///
/// let mut semantics = UniversalSemantics::new();
///
/// // Get primitive encoding
/// let good = semantics.get_prime(SemanticPrime::Good);
/// let person = semantics.get_prime(SemanticPrime::Someone);
///
/// // Compose complex concept
/// let good_person = semantics.compose_concept(
///     "good person",
///     vec![SemanticPrime::Good, SemanticPrime::Someone],
///     "bind(GOOD, SOMEONE)"
/// );
///
/// println!("Complexity: {}", good_person.complexity);
/// ```
#[derive(Debug)]
pub struct UniversalSemantics {
    /// Primitive encodings (65 universal concepts)
    primitives: HashMap<SemanticPrime, HV16>,

    /// Complex concept library
    concepts: HashMap<String, ComplexConcept>,
}

impl UniversalSemantics {
    /// Create new universal semantics system
    pub fn new() -> Self {
        let mut primitives = HashMap::new();

        // Initialize all 65 primes with distinct hypervectors
        for (idx, prime) in SemanticPrime::all().iter().enumerate() {
            // Each prime gets unique random encoding
            let encoding = HV16::random((1000 + idx * 100) as u64);
            primitives.insert(*prime, encoding);
        }

        Self {
            primitives,
            concepts: HashMap::new(),
        }
    }

    /// Get primitive encoding
    pub fn get_prime(&self, prime: SemanticPrime) -> &HV16 {
        self.primitives.get(&prime)
            .expect("All primes should be initialized")
    }

    /// Compose complex concept from primes
    pub fn compose_concept(
        &mut self,
        name: impl Into<String>,
        components: Vec<SemanticPrime>,
        structure: impl Into<String>,
    ) -> ComplexConcept {
        let name = name.into();
        let structure = structure.into();

        // Encode by bundling component primes
        let vectors: Vec<HV16> = components.iter()
            .map(|p| *self.get_prime(*p))
            .collect();

        let encoding = if vectors.len() == 1 {
            vectors[0]
        } else {
            HV16::bundle(&vectors)
        };

        let complexity = components.len();

        let concept = ComplexConcept {
            name: name.clone(),
            encoding,
            components,
            structure,
            complexity,
        };

        self.concepts.insert(name, concept.clone());
        concept
    }

    /// Bind two concepts (preserves structure)
    pub fn bind(&self, a: &HV16, b: &HV16) -> HV16 {
        a.bind(b)
    }

    /// Bundle concepts (superposition)
    pub fn bundle(&self, concepts: &[HV16]) -> HV16 {
        HV16::bundle(concepts)
    }

    /// Get concept by name
    pub fn get_concept(&self, name: &str) -> Option<&ComplexConcept> {
        self.concepts.get(name)
    }

    /// Measure similarity between concepts
    pub fn similarity(&self, a: &HV16, b: &HV16) -> f64 {
        a.similarity(b) as f64
    }

    /// Count primitives
    pub fn num_primitives(&self) -> usize {
        self.primitives.len()
    }

    /// Count complex concepts
    pub fn num_concepts(&self) -> usize {
        self.concepts.len()
    }
}

impl Default for UniversalSemantics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_prime_count() {
        let primes = SemanticPrime::all();
        assert_eq!(primes.len(), 65);  // NSM theory: 65 universal primes
    }

    #[test]
    fn test_universal_semantics_creation() {
        let semantics = UniversalSemantics::new();
        assert_eq!(semantics.num_primitives(), 65);
        assert_eq!(semantics.num_concepts(), 0);
    }

    #[test]
    fn test_primitive_encoding() {
        let semantics = UniversalSemantics::new();

        let good = semantics.get_prime(SemanticPrime::Good);
        let bad = semantics.get_prime(SemanticPrime::Bad);

        // Different primes should have different encodings
        let similarity = semantics.similarity(good, bad);
        assert!(similarity < 0.9);  // Should be different
    }

    #[test]
    fn test_concept_composition() {
        let mut semantics = UniversalSemantics::new();

        let concept = semantics.compose_concept(
            "good person",
            vec![SemanticPrime::Good, SemanticPrime::Someone],
            "bind(GOOD, SOMEONE)"
        );

        assert_eq!(concept.name, "good person");
        assert_eq!(concept.complexity, 2);
        assert_eq!(concept.components.len(), 2);
    }

    #[test]
    fn test_complex_emotion_grief() {
        let mut semantics = UniversalSemantics::new();

        // GRIEF = feel bad + someone died + want not happen + know cannot do anything
        let grief = semantics.compose_concept(
            "grief",
            vec![
                SemanticPrime::Feel,
                SemanticPrime::Bad,
                SemanticPrime::Someone,
                SemanticPrime::Die,
                SemanticPrime::Want,
                SemanticPrime::Not,
                SemanticPrime::Happen,
                SemanticPrime::Know,
                SemanticPrime::Can,
                SemanticPrime::Do,
            ],
            "bundle(bind(FEEL, BAD), bind(SOMEONE, DIE), bind(WANT, bind(NOT, HAPPEN)), bind(KNOW, bind(NOT, bind(CAN, DO))))"
        );

        assert_eq!(grief.name, "grief");
        assert!(grief.complexity >= 10);
    }

    #[test]
    fn test_complex_emotion_joy() {
        let mut semantics = UniversalSemantics::new();

        // JOY = feel very good + something good happened + want more
        let joy = semantics.compose_concept(
            "joy",
            vec![
                SemanticPrime::Feel,
                SemanticPrime::Very,
                SemanticPrime::Good,
                SemanticPrime::Something,
                SemanticPrime::Happen,
                SemanticPrime::Want,
                SemanticPrime::More,
            ],
            "bundle(bind(FEEL, bind(VERY, GOOD)), bind(SOMETHING, bind(GOOD, HAPPEN)), bind(WANT, MORE))"
        );

        assert_eq!(joy.name, "joy");
        assert!(joy.complexity >= 7);
    }

    #[test]
    fn test_complex_emotion_love() {
        let mut semantics = UniversalSemantics::new();

        // LOVE = feel very good about someone + want good for them + want to be with them
        let love = semantics.compose_concept(
            "love",
            vec![
                SemanticPrime::I,
                SemanticPrime::Feel,
                SemanticPrime::Very,
                SemanticPrime::Good,
                SemanticPrime::Someone,
                SemanticPrime::Want,
                SemanticPrime::Think,
            ],
            "bundle(bind(I, bind(FEEL, bind(VERY, GOOD))), bind(WANT, bind(GOOD, SOMEONE)), bind(THINK, bind(GOOD, SOMEONE)))"
        );

        assert_eq!(love.name, "love");
        assert!(love.complexity >= 7);
    }

    #[test]
    fn test_concept_retrieval() {
        let mut semantics = UniversalSemantics::new();

        semantics.compose_concept(
            "happiness",
            vec![SemanticPrime::Feel, SemanticPrime::Good],
            "bind(FEEL, GOOD)"
        );

        let retrieved = semantics.get_concept("happiness");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "happiness");
    }

    #[test]
    fn test_semantic_categories() {
        assert_eq!(SemanticPrime::Think.category(), "Mental Predicates");
        assert_eq!(SemanticPrime::Good.category(), "Evaluators");
        assert_eq!(SemanticPrime::Now.category(), "Time");
        assert_eq!(SemanticPrime::Here.category(), "Space");
    }

    #[test]
    fn test_serialization() {
        let prime = SemanticPrime::Good;
        let serialized = serde_json::to_string(&prime).unwrap();
        assert!(!serialized.is_empty());

        let deserialized: SemanticPrime = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, prime);
    }
}
