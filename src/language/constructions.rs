//! Construction Grammar - Meaningful Grammatical Patterns
//!
//! **Theoretical Foundation**: Construction Grammar (Goldberg, 1995) proposes that
//! grammatical patterns (constructions) carry meaning independent of the words that
//! fill them. A construction is a form-meaning pair.
//!
//! **Why This Matters**:
//! - "She sneezed the napkin off the table" is understood via the CAUSED_MOTION construction
//! - Even nonsense words get meaning from construction: "She blorped him the fleem"
//! - Constructions are compositional primitives at the syntactic level
//!
//! **Integration with NSM + HDC + Frames**:
//! - Constructions select and constrain frame activation
//! - Construction meaning maps to NSM prime compositions
//! - Slot fillers bind to frame elements
//!
//! # Core Constructions
//!
//! | Construction | Form | Meaning | Example |
//! |--------------|------|---------|---------|
//! | Ditransitive | [SUBJ V IOBJ DOBJ] | X causes Y to receive Z | "She gave him a book" |
//! | Caused-Motion | [SUBJ V OBJ OBL] | X causes Y to move Z | "She sneezed the napkin off" |
//! | Resultative | [SUBJ V OBJ RESULT] | X causes Y to become Z | "She hammered the metal flat" |
//! | Way-Construction | [SUBJ V POSS way OBL] | X moves with difficulty along Z | "She made her way up" |
//! | Passive | [OBJ be V-ed (by SUBJ)] | Y was affected by X | "The book was read" |

use crate::hdc::binary_hv::HV16;
use crate::hdc::universal_semantics::{SemanticPrime, UniversalSemantics};
use super::frames::{SemanticFrame, FrameElement, FrameLibrary};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// SYNTACTIC SLOTS
// =============================================================================

/// A slot in a syntactic pattern
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyntacticSlot {
    /// Subject position
    Subject,
    /// Verb position
    Verb,
    /// Direct object position
    DirectObject,
    /// Indirect object position
    IndirectObject,
    /// Oblique (prepositional phrase)
    Oblique(String),  // Preposition
    /// Resultative predicate
    Result,
    /// Possessive
    Possessive,
    /// Fixed word
    Fixed(String),
    /// Optional slot
    Optional(Box<SyntacticSlot>),
}

impl SyntacticSlot {
    /// Check if this slot can be filled by a given POS tag
    pub fn accepts(&self, pos: &str) -> bool {
        match self {
            SyntacticSlot::Subject => matches!(pos, "NP" | "NOUN" | "PRON"),
            SyntacticSlot::Verb => matches!(pos, "VP" | "VERB"),
            SyntacticSlot::DirectObject => matches!(pos, "NP" | "NOUN" | "PRON"),
            SyntacticSlot::IndirectObject => matches!(pos, "NP" | "NOUN" | "PRON"),
            SyntacticSlot::Oblique(_) => matches!(pos, "PP" | "PREP"),
            SyntacticSlot::Result => matches!(pos, "AP" | "ADJ" | "NP"),
            SyntacticSlot::Possessive => matches!(pos, "POSS" | "PRON"),
            SyntacticSlot::Fixed(word) => pos == word,
            SyntacticSlot::Optional(inner) => inner.accepts(pos),
        }
    }
}

// =============================================================================
// SYNTACTIC PATTERN
// =============================================================================

/// A syntactic pattern (sequence of slots)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntacticPattern {
    /// Name of the pattern
    pub name: String,

    /// Ordered slots
    pub slots: Vec<SyntacticSlot>,

    /// Pattern encoding
    pub encoding: HV16,
}

impl SyntacticPattern {
    /// Create new syntactic pattern
    pub fn new(name: impl Into<String>, slots: Vec<SyntacticSlot>) -> Self {
        let name = name.into();

        // Create encoding from slot sequence
        let seed = name.bytes().fold(100u64, |acc, b| acc.wrapping_add(b as u64).wrapping_mul(37));
        let encoding = HV16::random(seed);

        Self { name, slots, encoding }
    }

    /// Check if a token sequence matches this pattern
    pub fn matches(&self, tokens: &[&str], pos_tags: &[&str]) -> Option<Vec<(SyntacticSlot, String)>> {
        if pos_tags.len() < self.required_slots() {
            return None;
        }

        let mut bindings = Vec::new();
        let mut pos_idx = 0;

        for slot in &self.slots {
            if pos_idx >= pos_tags.len() {
                if matches!(slot, SyntacticSlot::Optional(_)) {
                    continue;
                }
                return None;
            }

            match slot {
                SyntacticSlot::Optional(inner) => {
                    if inner.accepts(pos_tags[pos_idx]) {
                        bindings.push((slot.clone(), tokens[pos_idx].to_string()));
                        pos_idx += 1;
                    }
                }
                _ => {
                    if slot.accepts(pos_tags[pos_idx]) {
                        bindings.push((slot.clone(), tokens[pos_idx].to_string()));
                        pos_idx += 1;
                    } else {
                        return None;
                    }
                }
            }
        }

        Some(bindings)
    }

    /// Count required (non-optional) slots
    pub fn required_slots(&self) -> usize {
        self.slots.iter()
            .filter(|s| !matches!(s, SyntacticSlot::Optional(_)))
            .count()
    }
}

// =============================================================================
// SEMANTIC STRUCTURE (Construction Meaning)
// =============================================================================

/// Semantic structure contributed by a construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticStructure {
    /// Primary frame evoked
    pub primary_frame: String,

    /// Role mappings (slot → frame element)
    pub role_mappings: HashMap<String, String>,

    /// Additional semantic constraints
    pub constraints: Vec<SemanticConstraint>,

    /// NSM prime structure of the construction meaning
    pub prime_structure: Vec<(SemanticPrime, SemanticPrime)>,

    /// HDC encoding
    pub encoding: HV16,
}

impl SemanticStructure {
    /// Create new semantic structure
    pub fn new(
        primary_frame: impl Into<String>,
        role_mappings: HashMap<String, String>,
        prime_structure: Vec<(SemanticPrime, SemanticPrime)>,
    ) -> Self {
        let primary_frame = primary_frame.into();

        // Create encoding from frame and primes
        let mut semantics = UniversalSemantics::new();
        let mut vectors = Vec::new();

        for (p1, p2) in &prime_structure {
            let bound = semantics.get_prime(*p1).bind(semantics.get_prime(*p2));
            vectors.push(bound);
        }

        let encoding = if vectors.is_empty() {
            let seed = primary_frame.bytes().fold(200u64, |acc, b| acc.wrapping_add(b as u64).wrapping_mul(41));
            HV16::random(seed)
        } else {
            HV16::bundle(&vectors)
        };

        Self {
            primary_frame,
            role_mappings,
            constraints: Vec::new(),
            prime_structure,
            encoding,
        }
    }

    /// Add a semantic constraint
    pub fn add_constraint(&mut self, constraint: SemanticConstraint) {
        self.constraints.push(constraint);
    }

    /// Map a syntactic slot to a frame element
    pub fn get_frame_element(&self, slot_name: &str) -> Option<&String> {
        self.role_mappings.get(slot_name)
    }
}

/// Semantic constraints on constructions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SemanticConstraint {
    /// Slot must be animate
    RequireAnimate(String),
    /// Slot must be inanimate
    RequireInanimate(String),
    /// Slot must be a location
    RequireLocation(String),
    /// Verb must be in a certain class
    VerbClass(String),
    /// Custom constraint
    Custom(String),
}

// =============================================================================
// CONSTRUCTION
// =============================================================================

/// A grammatical construction (form-meaning pair)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Construction {
    /// Construction name
    pub name: String,

    /// Description
    pub description: String,

    /// Syntactic form
    pub form: SyntacticPattern,

    /// Semantic meaning
    pub meaning: SemanticStructure,

    /// Example sentences
    pub examples: Vec<String>,

    /// Frequency/entrenchment score
    pub entrenchment: f32,

    /// Combined encoding
    pub encoding: HV16,
}

impl Construction {
    /// Create a new construction
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        form: SyntacticPattern,
        meaning: SemanticStructure,
    ) -> Self {
        let name = name.into();
        let description = description.into();

        // Combine form and meaning encodings
        let encoding = form.encoding.bind(&meaning.encoding);

        Self {
            name,
            description,
            form,
            meaning,
            examples: Vec::new(),
            entrenchment: 1.0,
            encoding,
        }
    }

    /// Add an example
    pub fn add_example(&mut self, example: impl Into<String>) -> &mut Self {
        self.examples.push(example.into());
        self
    }

    /// Set entrenchment score
    pub fn set_entrenchment(&mut self, score: f32) -> &mut Self {
        self.entrenchment = score;
        self
    }

    /// Attempt to parse with this construction
    pub fn parse(&self, tokens: &[&str], pos_tags: &[&str]) -> Option<ConstructionParse> {
        self.form.matches(tokens, pos_tags).map(|bindings| {
            ConstructionParse {
                construction_name: self.name.clone(),
                slot_fillers: bindings.into_iter()
                    .map(|(slot, filler)| (format!("{:?}", slot), filler))
                    .collect(),
                frame: self.meaning.primary_frame.clone(),
                encoding: self.encoding,
            }
        })
    }
}

/// Result of parsing with a construction
#[derive(Debug, Clone)]
pub struct ConstructionParse {
    /// Name of the matched construction
    pub construction_name: String,

    /// Slot fillers
    pub slot_fillers: HashMap<String, String>,

    /// Activated frame
    pub frame: String,

    /// Parse encoding
    pub encoding: HV16,
}

impl ConstructionParse {
    /// Get filler for a slot
    pub fn get_filler(&self, slot: &str) -> Option<&String> {
        self.slot_fillers.get(slot)
    }

    /// Get frame element binding
    pub fn get_frame_binding(&self, frame_element: &str, meaning: &SemanticStructure) -> Option<&String> {
        // Find slot that maps to this frame element
        for (slot, element) in &meaning.role_mappings {
            if element == frame_element {
                return self.slot_fillers.get(slot);
            }
        }
        None
    }
}

// =============================================================================
// CONSTRUCTION GRAMMAR
// =============================================================================

/// A collection of constructions forming a grammar
#[derive(Debug, Clone)]
pub struct ConstructionGrammar {
    /// All constructions indexed by name
    constructions: HashMap<String, Construction>,

    /// Constructions sorted by complexity (for greedy parsing)
    by_complexity: Vec<String>,
}

impl ConstructionGrammar {
    /// Create new construction grammar
    pub fn new() -> Self {
        let mut grammar = Self {
            constructions: HashMap::new(),
            by_complexity: Vec::new(),
        };

        // Initialize with core English constructions
        grammar.initialize_core_constructions();

        grammar
    }

    /// Initialize core constructions
    fn initialize_core_constructions(&mut self) {
        // =================================================================
        // TRANSITIVE Construction
        // =================================================================
        let transitive_form = SyntacticPattern::new(
            "Transitive",
            vec![
                SyntacticSlot::Subject,
                SyntacticSlot::Verb,
                SyntacticSlot::DirectObject,
            ],
        );

        let mut transitive_meaning = SemanticStructure::new(
            "CAUSATION",
            [
                ("Subject".to_string(), "Cause".to_string()),
                ("DirectObject".to_string(), "Effect".to_string()),
            ].into_iter().collect(),
            vec![
                (SemanticPrime::Someone, SemanticPrime::Do),
                (SemanticPrime::Something, SemanticPrime::Happen),
            ],
        );
        transitive_meaning.add_constraint(SemanticConstraint::RequireAnimate("Subject".into()));

        let mut transitive = Construction::new(
            "Transitive",
            "Basic transitive construction: X does something to Y",
            transitive_form,
            transitive_meaning,
        );
        transitive.add_example("The cat chased the mouse");
        transitive.add_example("She read the book");

        self.add_construction(transitive);

        // =================================================================
        // DITRANSITIVE Construction
        // =================================================================
        let ditransitive_form = SyntacticPattern::new(
            "Ditransitive",
            vec![
                SyntacticSlot::Subject,
                SyntacticSlot::Verb,
                SyntacticSlot::IndirectObject,
                SyntacticSlot::DirectObject,
            ],
        );

        let ditransitive_meaning = SemanticStructure::new(
            "TRANSFER",
            [
                ("Subject".to_string(), "Donor".to_string()),
                ("IndirectObject".to_string(), "Recipient".to_string()),
                ("DirectObject".to_string(), "Theme".to_string()),
            ].into_iter().collect(),
            vec![
                (SemanticPrime::Someone, SemanticPrime::Do),
                (SemanticPrime::Someone, SemanticPrime::Have),
                (SemanticPrime::Something, SemanticPrime::Move),
            ],
        );

        let mut ditransitive = Construction::new(
            "Ditransitive",
            "Transfer construction: X causes Y to receive Z",
            ditransitive_form,
            ditransitive_meaning,
        );
        ditransitive.add_example("She gave him a book");
        ditransitive.add_example("He told her a story");
        ditransitive.set_entrenchment(0.9);

        self.add_construction(ditransitive);

        // =================================================================
        // CAUSED-MOTION Construction
        // =================================================================
        let caused_motion_form = SyntacticPattern::new(
            "CausedMotion",
            vec![
                SyntacticSlot::Subject,
                SyntacticSlot::Verb,
                SyntacticSlot::DirectObject,
                SyntacticSlot::Oblique("to/into/off".into()),
            ],
        );

        let caused_motion_meaning = SemanticStructure::new(
            "MOTION",
            [
                ("Subject".to_string(), "Cause".to_string()),
                ("DirectObject".to_string(), "Mover".to_string()),
                ("Oblique".to_string(), "Goal".to_string()),
            ].into_iter().collect(),
            vec![
                (SemanticPrime::Someone, SemanticPrime::Do),
                (SemanticPrime::Something, SemanticPrime::Move),
                (SemanticPrime::Where, SemanticPrime::After),
            ],
        );

        let mut caused_motion = Construction::new(
            "CausedMotion",
            "X causes Y to move to Z",
            caused_motion_form,
            caused_motion_meaning,
        );
        caused_motion.add_example("She put the book on the table");
        caused_motion.add_example("He sneezed the napkin off the table");
        caused_motion.set_entrenchment(0.8);

        self.add_construction(caused_motion);

        // =================================================================
        // RESULTATIVE Construction
        // =================================================================
        let resultative_form = SyntacticPattern::new(
            "Resultative",
            vec![
                SyntacticSlot::Subject,
                SyntacticSlot::Verb,
                SyntacticSlot::DirectObject,
                SyntacticSlot::Result,
            ],
        );

        let resultative_meaning = SemanticStructure::new(
            "CAUSATION",
            [
                ("Subject".to_string(), "Cause".to_string()),
                ("DirectObject".to_string(), "Effect".to_string()),
                ("Result".to_string(), "Result_state".to_string()),
            ].into_iter().collect(),
            vec![
                (SemanticPrime::Someone, SemanticPrime::Do),
                (SemanticPrime::Something, SemanticPrime::Happen),
                (SemanticPrime::Something, SemanticPrime::Be),
            ],
        );

        let mut resultative = Construction::new(
            "Resultative",
            "X causes Y to become Z (state)",
            resultative_form,
            resultative_meaning,
        );
        resultative.add_example("She hammered the metal flat");
        resultative.add_example("He painted the house red");
        resultative.set_entrenchment(0.7);

        self.add_construction(resultative);

        // =================================================================
        // INTRANSITIVE MOTION Construction
        // =================================================================
        let intrans_motion_form = SyntacticPattern::new(
            "IntransitiveMotion",
            vec![
                SyntacticSlot::Subject,
                SyntacticSlot::Verb,
                SyntacticSlot::Optional(Box::new(SyntacticSlot::Oblique("to/from".into()))),
            ],
        );

        let intrans_motion_meaning = SemanticStructure::new(
            "MOTION",
            [
                ("Subject".to_string(), "Mover".to_string()),
                ("Oblique".to_string(), "Goal".to_string()),
            ].into_iter().collect(),
            vec![
                (SemanticPrime::Something, SemanticPrime::Move),
            ],
        );

        let mut intrans_motion = Construction::new(
            "IntransitiveMotion",
            "X moves (to Z)",
            intrans_motion_form,
            intrans_motion_meaning,
        );
        intrans_motion.add_example("She walked to the store");
        intrans_motion.add_example("The ball rolled");

        self.add_construction(intrans_motion);

        // =================================================================
        // COPULAR Construction
        // =================================================================
        let copular_form = SyntacticPattern::new(
            "Copular",
            vec![
                SyntacticSlot::Subject,
                SyntacticSlot::Fixed("be".into()),
                SyntacticSlot::Result,  // Predicate nominal/adjectival
            ],
        );

        let copular_meaning = SemanticStructure::new(
            "BEING_LOCATED",
            [
                ("Subject".to_string(), "Theme".to_string()),
                ("Result".to_string(), "Location".to_string()),
            ].into_iter().collect(),
            vec![
                (SemanticPrime::Something, SemanticPrime::Be),
            ],
        );

        let mut copular = Construction::new(
            "Copular",
            "X is Y (state/identity)",
            copular_form,
            copular_meaning,
        );
        copular.add_example("She is happy");
        copular.add_example("He is a doctor");
        copular.set_entrenchment(1.0);

        self.add_construction(copular);

        // =================================================================
        // WAY Construction
        // =================================================================
        let way_form = SyntacticPattern::new(
            "Way",
            vec![
                SyntacticSlot::Subject,
                SyntacticSlot::Verb,
                SyntacticSlot::Possessive,
                SyntacticSlot::Fixed("way".into()),
                SyntacticSlot::Oblique("through/into/out".into()),
            ],
        );

        let way_meaning = SemanticStructure::new(
            "MOTION",
            [
                ("Subject".to_string(), "Mover".to_string()),
                ("Oblique".to_string(), "Path".to_string()),
            ].into_iter().collect(),
            vec![
                (SemanticPrime::Someone, SemanticPrime::Move),
                (SemanticPrime::Do, SemanticPrime::Something),  // With effort
            ],
        );

        let mut way = Construction::new(
            "Way",
            "X moves with effort/difficulty along Z",
            way_form,
            way_meaning,
        );
        way.add_example("She made her way through the crowd");
        way.add_example("He elbowed his way to the front");
        way.set_entrenchment(0.6);

        self.add_construction(way);

        // =================================================================
        // UNIVERSAL CONCEPTUAL LANGUAGE (UCL) CONSTRUCTIONS
        // Revolutionary: Cross-domain grammatical patterns
        // =================================================================

        // =================================================================
        // OPPOSITION Construction - Maps to CONFLICT frame
        // =================================================================
        let opposition_form = SyntacticPattern::new(
            "Opposition",
            vec![
                SyntacticSlot::Subject,                    // Party_1
                SyntacticSlot::Verb,                       // opposes/fights/resists
                SyntacticSlot::DirectObject,               // Party_2
                SyntacticSlot::Optional(Box::new(
                    SyntacticSlot::Oblique("over/about".into())  // Issue
                )),
            ],
        );

        let opposition_meaning = SemanticStructure::new(
            "CONFLICT",
            [
                ("Subject".to_string(), "Party_1".to_string()),
                ("DirectObject".to_string(), "Party_2".to_string()),
                ("Oblique".to_string(), "Issue".to_string()),
            ].into_iter().collect(),
            vec![
                (SemanticPrime::Someone, SemanticPrime::Want),
                (SemanticPrime::Someone, SemanticPrime::Not),   // Opposing wants
                (SemanticPrime::Bad, SemanticPrime::Happen),    // Potential harm
            ],
        );

        let mut opposition = Construction::new(
            "Opposition",
            "X is in conflict with Y (over Z)",
            opposition_form,
            opposition_meaning,
        );
        opposition.add_example("The company opposes the regulation");
        opposition.add_example("They fought each other over resources");
        opposition.add_example("Workers resisted the change");
        opposition.set_entrenchment(0.7);

        self.add_construction(opposition);

        // =================================================================
        // COLLABORATION Construction - Maps to COOPERATION frame
        // =================================================================
        let collaboration_form = SyntacticPattern::new(
            "Collaboration",
            vec![
                SyntacticSlot::Subject,                    // Cooperators (compound)
                SyntacticSlot::Verb,                       // cooperate/collaborate/work
                SyntacticSlot::Oblique("on/with/toward".into()),  // Goal/Activity
            ],
        );

        let collaboration_meaning = SemanticStructure::new(
            "COOPERATION",
            [
                ("Subject".to_string(), "Cooperators".to_string()),
                ("Oblique".to_string(), "Goal".to_string()),
            ].into_iter().collect(),
            vec![
                (SemanticPrime::People, SemanticPrime::Do),
                (SemanticPrime::Same, SemanticPrime::Want),
                (SemanticPrime::Good, SemanticPrime::All),
            ],
        );

        let mut collaboration = Construction::new(
            "Collaboration",
            "X and Y work together toward Z",
            collaboration_form,
            collaboration_meaning,
        );
        collaboration.add_example("The teams collaborate on the project");
        collaboration.add_example("They worked together toward a solution");
        collaboration.add_example("Nations cooperate on climate action");
        collaboration.set_entrenchment(0.75);

        self.add_construction(collaboration);

        // =================================================================
        // EXCHANGE Construction - Maps to TRADE frame
        // =================================================================
        let exchange_form = SyntacticPattern::new(
            "Exchange",
            vec![
                SyntacticSlot::Subject,                    // Trader_1
                SyntacticSlot::Verb,                       // trade/exchange/swap
                SyntacticSlot::DirectObject,               // Item_1
                SyntacticSlot::Oblique("with".into()),     // Trader_2
                SyntacticSlot::Optional(Box::new(
                    SyntacticSlot::Oblique("for".into())   // Item_2
                )),
            ],
        );

        let exchange_meaning = SemanticStructure::new(
            "TRADE",
            [
                ("Subject".to_string(), "Trader_1".to_string()),
                ("DirectObject".to_string(), "Item_1".to_string()),
                ("Oblique_with".to_string(), "Trader_2".to_string()),
                ("Oblique_for".to_string(), "Item_2".to_string()),
            ].into_iter().collect(),
            vec![
                (SemanticPrime::Someone, SemanticPrime::Have),
                (SemanticPrime::Something, SemanticPrime::Move),
                (SemanticPrime::Good, SemanticPrime::All),
            ],
        );

        let mut exchange = Construction::new(
            "Exchange",
            "X trades A with Y for B",
            exchange_form,
            exchange_meaning,
        );
        exchange.add_example("She traded her car with him for a boat");
        exchange.add_example("They exchanged gifts");
        exchange.add_example("Countries swap resources with allies");
        exchange.set_entrenchment(0.7);

        self.add_construction(exchange);

        // =================================================================
        // CYCLIC_CAUSATION Construction - Maps to FEEDBACK_LOOP frame
        // =================================================================
        let cyclic_form = SyntacticPattern::new(
            "CyclicCausation",
            vec![
                SyntacticSlot::Subject,                    // System/Output
                SyntacticSlot::Verb,                       // affects/influences/reinforces
                SyntacticSlot::DirectObject,               // Target
                SyntacticSlot::Fixed("which".into()),
                SyntacticSlot::Verb,                       // affects back
                SyntacticSlot::DirectObject,               // Original (referring back)
            ],
        );

        let cyclic_meaning = SemanticStructure::new(
            "FEEDBACK_LOOP",
            [
                ("Subject".to_string(), "Output".to_string()),
                ("DirectObject".to_string(), "Effect_on_Input".to_string()),
            ].into_iter().collect(),
            vec![
                (SemanticPrime::Something, SemanticPrime::Happen),
                (SemanticPrime::Because, SemanticPrime::Before),
                (SemanticPrime::After, SemanticPrime::Same),
            ],
        );

        let mut cyclic = Construction::new(
            "CyclicCausation",
            "X affects Y which affects X (feedback loop)",
            cyclic_form,
            cyclic_meaning,
        );
        cyclic.add_example("Growth drives investment which drives growth");
        cyclic.add_example("Fear causes panic which causes more fear");
        cyclic.add_example("Success breeds confidence which breeds success");
        cyclic.set_entrenchment(0.6);

        self.add_construction(cyclic);

        // =================================================================
        // REGULATION Construction - Maps to NORM_ENFORCEMENT frame
        // =================================================================
        let regulation_form = SyntacticPattern::new(
            "Regulation",
            vec![
                SyntacticSlot::Subject,                    // Enforcer
                SyntacticSlot::Verb,                       // enforce/regulate/require
                SyntacticSlot::DirectObject,               // Norm/Rule
                SyntacticSlot::Oblique("on/upon".into()),  // Subject (regulated entity)
            ],
        );

        let regulation_meaning = SemanticStructure::new(
            "NORM_ENFORCEMENT",
            [
                ("Subject".to_string(), "Enforcer".to_string()),
                ("DirectObject".to_string(), "Norm".to_string()),
                ("Oblique".to_string(), "Subject".to_string()),
            ].into_iter().collect(),
            vec![
                (SemanticPrime::Someone, SemanticPrime::Can),
                (SemanticPrime::True, SemanticPrime::Good),
                (SemanticPrime::Do, SemanticPrime::Do),
            ],
        );

        let mut regulation = Construction::new(
            "Regulation",
            "X enforces norm Y on Z",
            regulation_form,
            regulation_meaning,
        );
        regulation.add_example("The government enforces safety standards on manufacturers");
        regulation.add_example("Society regulates behavior through norms");
        regulation.add_example("Parents impose rules on children");
        regulation.set_entrenchment(0.65);

        self.add_construction(regulation);

        // =================================================================
        // ADJUSTMENT Construction - Maps to ADAPTATION frame
        // =================================================================
        let adjustment_form = SyntacticPattern::new(
            "Adjustment",
            vec![
                SyntacticSlot::Subject,                    // Adapter
                SyntacticSlot::Verb,                       // adapt/adjust/evolve
                SyntacticSlot::Oblique("to".into()),       // Environment
            ],
        );

        let adjustment_meaning = SemanticStructure::new(
            "ADAPTATION",
            [
                ("Subject".to_string(), "Adapter".to_string()),
                ("Oblique".to_string(), "Environment".to_string()),
            ].into_iter().collect(),
            vec![
                (SemanticPrime::Something, SemanticPrime::Live),
                (SemanticPrime::Other, SemanticPrime::Happen),
                (SemanticPrime::Something, SemanticPrime::Other),
            ],
        );

        let mut adjustment = Construction::new(
            "Adjustment",
            "X adapts/adjusts to Y",
            adjustment_form,
            adjustment_meaning,
        );
        adjustment.add_example("Species adapt to their environment");
        adjustment.add_example("The company adjusted to market changes");
        adjustment.add_example("We evolved to survive in harsh conditions");
        adjustment.set_entrenchment(0.7);

        self.add_construction(adjustment);

        // Update complexity ordering
        self.update_complexity_order();
    }

    /// Add a construction
    pub fn add_construction(&mut self, construction: Construction) {
        let name = construction.name.clone();
        self.constructions.insert(name.clone(), construction);
        self.by_complexity.push(name);
    }

    /// Update complexity ordering (more complex first for greedy parsing)
    fn update_complexity_order(&mut self) {
        self.by_complexity.sort_by(|a, b| {
            let ca = self.constructions.get(a).map(|c| c.form.slots.len()).unwrap_or(0);
            let cb = self.constructions.get(b).map(|c| c.form.slots.len()).unwrap_or(0);
            cb.cmp(&ca)  // Descending order
        });
    }

    /// Get construction by name
    pub fn get(&self, name: &str) -> Option<&Construction> {
        self.constructions.get(name)
    }

    /// Parse tokens with grammar (returns all matching constructions)
    pub fn parse_all(&self, tokens: &[&str], pos_tags: &[&str]) -> Vec<ConstructionParse> {
        let mut parses = Vec::new();

        for name in &self.by_complexity {
            if let Some(construction) = self.constructions.get(name) {
                if let Some(parse) = construction.parse(tokens, pos_tags) {
                    parses.push(parse);
                }
            }
        }

        parses
    }

    /// Parse with best (most complex) matching construction
    pub fn parse_best(&self, tokens: &[&str], pos_tags: &[&str]) -> Option<ConstructionParse> {
        for name in &self.by_complexity {
            if let Some(construction) = self.constructions.get(name) {
                if let Some(parse) = construction.parse(tokens, pos_tags) {
                    return Some(parse);
                }
            }
        }
        None
    }

    /// Simple matching without POS tags (uses word form heuristics)
    ///
    /// This is a convenience method that infers simple POS tags from word forms.
    /// For full accuracy, use `parse_all` with proper POS tags.
    pub fn match_constructions(&self, words: &[String]) -> Vec<ConstructionParse> {
        // Simple POS inference from word forms
        let pos_tags: Vec<&str> = words.iter().map(|w| {
            let w_lower = w.to_lowercase();
            if ["the", "a", "an", "this", "that", "my", "your"].contains(&w_lower.as_str()) {
                "DET"
            } else if ["is", "are", "was", "were", "be", "been", "being",
                       "have", "has", "had", "do", "does", "did",
                       "give", "gave", "take", "took", "put", "make", "made",
                       "see", "saw", "know", "knew", "think", "thought",
                       "say", "said", "go", "went", "come", "came",
                       "want", "wanted", "walk", "walked", "run", "ran"].contains(&w_lower.as_str()) {
                "V"
            } else if w_lower.ends_with("ly") {
                "ADV"
            } else if w_lower.ends_with("ing") {
                "VBG"  // Could also be noun, but simplify
            } else if w_lower.ends_with("ed") {
                "VBD"
            } else if ["i", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they", "them"].contains(&w_lower.as_str()) {
                "PRON"
            } else if ["on", "in", "at", "to", "from", "with", "by", "for", "about", "through"].contains(&w_lower.as_str()) {
                "PREP"
            } else {
                "N"  // Default to noun
            }
        }).collect();

        let tokens: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        self.parse_all(&tokens, &pos_tags)
    }

    /// Find constructions compatible with a frame
    pub fn for_frame(&self, frame_name: &str) -> Vec<&Construction> {
        self.constructions.values()
            .filter(|c| c.meaning.primary_frame == frame_name)
            .collect()
    }

    /// Count constructions
    pub fn len(&self) -> usize {
        self.constructions.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.constructions.is_empty()
    }

    /// Get all construction names
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.constructions.keys()
    }
}

impl Default for ConstructionGrammar {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CONSTRUCTION-FRAME INTEGRATOR
// =============================================================================

/// Integrates constructions with frame semantics
pub struct ConstructionFrameIntegrator {
    /// Construction grammar
    grammar: ConstructionGrammar,

    /// Frame library
    frames: FrameLibrary,
}

impl ConstructionFrameIntegrator {
    /// Create new integrator
    pub fn new() -> Self {
        Self {
            grammar: ConstructionGrammar::new(),
            frames: FrameLibrary::new(),
        }
    }

    /// Parse sentence and bind to frames
    pub fn parse_and_bind(
        &self,
        tokens: &[&str],
        pos_tags: &[&str],
    ) -> Option<IntegratedParse> {
        // Get construction parse
        let construction_parse = self.grammar.parse_best(tokens, pos_tags)?;

        // Get the frame
        let frame = self.frames.get(&construction_parse.frame)?;

        // Create bindings
        let mut frame_bindings = HashMap::new();
        let construction = self.grammar.get(&construction_parse.construction_name)?;

        for (slot, filler) in &construction_parse.slot_fillers {
            if let Some(frame_element) = construction.meaning.get_frame_element(slot) {
                // Create HV16 encoding for filler
                let filler_seed = filler.bytes().fold(300u64, |acc, b| acc.wrapping_add(b as u64).wrapping_mul(43));
                let filler_hv = HV16::random(filler_seed);
                frame_bindings.insert(frame_element.clone(), (filler.clone(), filler_hv));
            }
        }

        Some(IntegratedParse {
            construction: construction_parse.construction_name.clone(),
            frame: construction_parse.frame.clone(),
            bindings: frame_bindings,
            encoding: construction_parse.encoding.bind(&frame.encoding),
        })
    }

    /// Get grammar reference
    pub fn grammar(&self) -> &ConstructionGrammar {
        &self.grammar
    }

    /// Get frames reference
    pub fn frames(&self) -> &FrameLibrary {
        &self.frames
    }
}

impl Default for ConstructionFrameIntegrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Integrated parse result
#[derive(Debug, Clone)]
pub struct IntegratedParse {
    /// Construction used
    pub construction: String,

    /// Frame activated
    pub frame: String,

    /// Frame element bindings (element → (text, encoding))
    pub bindings: HashMap<String, (String, HV16)>,

    /// Combined encoding
    pub encoding: HV16,
}

impl IntegratedParse {
    /// Get text filler for frame element
    pub fn get_text(&self, element: &str) -> Option<&str> {
        self.bindings.get(element).map(|(text, _)| text.as_str())
    }

    /// Get encoding for frame element
    pub fn get_encoding(&self, element: &str) -> Option<&HV16> {
        self.bindings.get(element).map(|(_, enc)| enc)
    }

    /// Check if all core elements of a frame are bound
    pub fn is_complete(&self, frame: &SemanticFrame) -> bool {
        frame.core_elements.iter().all(|elem| self.bindings.contains_key(&elem.name))
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntactic_pattern_creation() {
        let pattern = SyntacticPattern::new(
            "Test",
            vec![SyntacticSlot::Subject, SyntacticSlot::Verb, SyntacticSlot::DirectObject],
        );

        assert_eq!(pattern.name, "Test");
        assert_eq!(pattern.slots.len(), 3);
        assert_eq!(pattern.required_slots(), 3);
    }

    #[test]
    fn test_construction_grammar_initialization() {
        let grammar = ConstructionGrammar::new();

        assert!(!grammar.is_empty());
        assert!(grammar.len() >= 6);  // At least 6 core constructions

        // Check specific constructions exist
        assert!(grammar.get("Transitive").is_some());
        assert!(grammar.get("Ditransitive").is_some());
        assert!(grammar.get("CausedMotion").is_some());
        assert!(grammar.get("Resultative").is_some());
    }

    #[test]
    fn test_construction_examples() {
        let grammar = ConstructionGrammar::new();

        let ditransitive = grammar.get("Ditransitive").unwrap();
        assert!(!ditransitive.examples.is_empty());
        assert!(ditransitive.examples.iter().any(|e| e.contains("gave")));
    }

    #[test]
    fn test_semantic_structure_role_mapping() {
        let meaning = SemanticStructure::new(
            "TRANSFER",
            [
                ("Subject".to_string(), "Donor".to_string()),
                ("DirectObject".to_string(), "Theme".to_string()),
            ].into_iter().collect(),
            vec![(SemanticPrime::Someone, SemanticPrime::Do)],
        );

        assert_eq!(meaning.get_frame_element("Subject"), Some(&"Donor".to_string()));
        assert_eq!(meaning.get_frame_element("DirectObject"), Some(&"Theme".to_string()));
        assert_eq!(meaning.get_frame_element("Nonexistent"), None);
    }

    #[test]
    fn test_frame_integration() {
        let grammar = ConstructionGrammar::new();

        // Check constructions map to frames
        let ditransitive = grammar.get("Ditransitive").unwrap();
        assert_eq!(ditransitive.meaning.primary_frame, "TRANSFER");

        let caused_motion = grammar.get("CausedMotion").unwrap();
        assert_eq!(caused_motion.meaning.primary_frame, "MOTION");
    }

    #[test]
    fn test_constructions_for_frame() {
        let grammar = ConstructionGrammar::new();

        let motion_constructions = grammar.for_frame("MOTION");
        assert!(!motion_constructions.is_empty());
        assert!(motion_constructions.iter().any(|c| c.name == "CausedMotion"));
    }

    #[test]
    fn test_construction_frame_integrator() {
        let integrator = ConstructionFrameIntegrator::new();

        // Check both grammar and frames are initialized
        assert!(!integrator.grammar().is_empty());
        assert!(!integrator.frames().is_empty());

        // Check frames reference exists
        assert!(integrator.frames().get("TRANSFER").is_some());
    }

    #[test]
    fn test_entrenchment_scores() {
        let grammar = ConstructionGrammar::new();

        let ditransitive = grammar.get("Ditransitive").unwrap();
        assert!(ditransitive.entrenchment > 0.0);
        assert!(ditransitive.entrenchment <= 1.0);

        let way = grammar.get("Way").unwrap();
        // Way construction should be less entrenched than Ditransitive
        assert!(way.entrenchment < ditransitive.entrenchment);
    }

    #[test]
    fn test_construction_encoding() {
        let grammar = ConstructionGrammar::new();

        let trans = grammar.get("Transitive").unwrap();
        let ditrans = grammar.get("Ditransitive").unwrap();

        // Different constructions should have different encodings
        let similarity = trans.encoding.similarity(&ditrans.encoding);
        assert!(similarity < 0.9);  // Should be different
    }

    #[test]
    fn test_syntactic_slot_accepts() {
        let subject = SyntacticSlot::Subject;
        assert!(subject.accepts("NP"));
        assert!(subject.accepts("NOUN"));
        assert!(subject.accepts("PRON"));
        assert!(!subject.accepts("VERB"));

        let verb = SyntacticSlot::Verb;
        assert!(verb.accepts("VP"));
        assert!(verb.accepts("VERB"));
        assert!(!verb.accepts("NP"));
    }

    #[test]
    fn test_complexity_ordering() {
        let grammar = ConstructionGrammar::new();

        // More complex constructions should come first in by_complexity
        let first_few: Vec<_> = grammar.by_complexity.iter().take(3).collect();

        // Ditransitive (4 slots) should come before Transitive (3 slots)
        let ditrans_pos = grammar.by_complexity.iter().position(|n| n == "Ditransitive");
        let trans_pos = grammar.by_complexity.iter().position(|n| n == "Transitive");

        if let (Some(d), Some(t)) = (ditrans_pos, trans_pos) {
            assert!(d < t, "Ditransitive should be parsed before Transitive");
        }
    }
}
