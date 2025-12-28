//! Frame Semantics - Structured Situation Representations
//!
//! **Theoretical Foundation**: Frame Semantics (Fillmore, 1982) proposes that words
//! evoke structured representations of situations called "frames". Each frame has
//! defined roles (frame elements) and can be compositionally combined.
//!
//! **Why This Matters**:
//! - Words don't just have definitions; they evoke entire situation structures
//! - "Buy" and "sell" describe the same event from different perspectives
//! - Frames enable inference: if someone buys, someone sold
//!
//! **Integration with NSM + HDC**:
//! - Frame elements map to semantic prime compositions
//! - Frames are encoded as HDC bundles of bound role-filler pairs
//! - Frame activation uses similarity matching against HDC encodings
//!
//! # Architecture
//!
//! ```text
//! Lexical Item → Frame Activation → Role Binding → HDC Encoding
//!      ↓              ↓                   ↓            ↓
//!   "gave"    TRANSFER frame     Agent=child    bind(AGENT, child_hv)
//!                                 Recipient=dog   bind(RECIPIENT, dog_hv)
//!                                 Theme=bone      bind(THEME, bone_hv)
//! ```

use crate::hdc::binary_hv::HV16;
use crate::hdc::universal_semantics::{SemanticPrime, UniversalSemantics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// FRAME ELEMENTS (Roles within Frames)
// =============================================================================

/// A frame element is a role that can be filled in a frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameElement {
    /// Role name (e.g., "Buyer", "Seller", "Goods")
    pub name: String,

    /// Semantic type constraint (e.g., "Sentient", "Physical_object")
    pub semantic_type: String,

    /// Whether this element is core (required) or peripheral
    pub is_core: bool,

    /// NSM prime decomposition of this role
    pub prime_structure: Vec<SemanticPrime>,

    /// HDC encoding of this element
    pub encoding: HV16,

    /// Current filler (if bound)
    pub filler: Option<HV16>,
}

impl FrameElement {
    /// Create a new frame element
    pub fn new(
        name: impl Into<String>,
        semantic_type: impl Into<String>,
        is_core: bool,
        prime_structure: Vec<SemanticPrime>,
    ) -> Self {
        let name = name.into();
        let semantic_type = semantic_type.into();

        // Create encoding from prime structure
        let encoding = Self::encode_from_primes(&prime_structure, &name);

        Self {
            name,
            semantic_type,
            is_core,
            prime_structure,
            encoding,
            filler: None,
        }
    }

    /// Encode from semantic primes
    fn encode_from_primes(primes: &[SemanticPrime], name: &str) -> HV16 {
        if primes.is_empty() {
            // Use name-based random encoding
            let seed = name.bytes().fold(0u64, |acc, b| acc.wrapping_add(b as u64) * 31);
            return HV16::random(seed);
        }

        // Bundle prime encodings
        let semantics = UniversalSemantics::new();
        let vectors: Vec<HV16> = primes.iter()
            .map(|p| *semantics.get_prime(*p))
            .collect();

        if vectors.len() == 1 {
            vectors[0]
        } else {
            HV16::bundle(&vectors)
        }
    }

    /// Bind a filler to this element
    pub fn bind_filler(&mut self, filler: HV16) {
        self.filler = Some(filler);
    }

    /// Get bound encoding (role + filler)
    pub fn bound_encoding(&self) -> Option<HV16> {
        self.filler.map(|f| self.encoding.bind(&f))
    }

    /// Check if this element can accept a given filler
    pub fn can_accept(&self, filler_type: &str) -> bool {
        // Basic semantic type checking
        match self.semantic_type.as_str() {
            "Sentient" => matches!(filler_type, "person" | "animal" | "agent"),
            "Physical_object" => matches!(filler_type, "object" | "thing" | "substance"),
            "Location" => matches!(filler_type, "place" | "location" | "position"),
            "Event" => matches!(filler_type, "event" | "action" | "happening"),
            "Time" => matches!(filler_type, "time" | "duration" | "instant"),
            "State" => matches!(filler_type, "state" | "condition" | "quality"),
            _ => true,  // Accept by default if type unknown
        }
    }
}

// =============================================================================
// FRAME RELATIONS
// =============================================================================

/// Relations between frames
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameRelation {
    /// Parent frame (more general)
    InheritsFrom,
    /// Subframe (specialized version)
    SubframeOf,
    /// Perspective on same situation
    PerspectiveOf,
    /// Uses another frame
    Uses,
    /// Precedes in time
    Precedes,
    /// Caused by another frame
    CausedBy,
    /// Inchoative (beginning of)
    InchoativeOf,
    /// Causative (causes something to enter state)
    CausativeOf,
}

// =============================================================================
// SEMANTIC FRAME
// =============================================================================

/// A semantic frame - structured representation of a type of situation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFrame {
    /// Frame name (e.g., "COMMERCIAL_TRANSACTION")
    pub name: String,

    /// Definition/description
    pub definition: String,

    /// Core frame elements (required roles)
    pub core_elements: Vec<FrameElement>,

    /// Non-core frame elements (optional roles)
    pub non_core_elements: Vec<FrameElement>,

    /// Lexical units that evoke this frame
    pub lexical_units: Vec<String>,

    /// Relations to other frames
    pub relations: Vec<(String, FrameRelation)>,

    /// HDC encoding of the entire frame
    pub encoding: HV16,

    /// NSM-based definition using primes
    pub prime_definition: Vec<(SemanticPrime, SemanticPrime)>,
}

impl SemanticFrame {
    /// Create a new semantic frame
    pub fn new(
        name: impl Into<String>,
        definition: impl Into<String>,
    ) -> Self {
        let name = name.into();
        let definition = definition.into();

        // Generate initial encoding from name
        let seed = name.bytes().fold(42u64, |acc, b| acc.wrapping_add(b as u64).wrapping_mul(31));
        let encoding = HV16::random(seed);

        Self {
            name,
            definition,
            core_elements: Vec::new(),
            non_core_elements: Vec::new(),
            lexical_units: Vec::new(),
            relations: Vec::new(),
            encoding,
            prime_definition: Vec::new(),
        }
    }

    /// Add a core frame element
    pub fn add_core_element(&mut self, element: FrameElement) -> &mut Self {
        self.core_elements.push(element);
        self.update_encoding();
        self
    }

    /// Add a non-core frame element
    pub fn add_non_core_element(&mut self, element: FrameElement) -> &mut Self {
        self.non_core_elements.push(element);
        self
    }

    /// Add a lexical unit that evokes this frame
    pub fn add_lexical_unit(&mut self, word: impl Into<String>) -> &mut Self {
        self.lexical_units.push(word.into());
        self
    }

    /// Add a relation to another frame
    pub fn add_relation(&mut self, target: impl Into<String>, relation: FrameRelation) -> &mut Self {
        self.relations.push((target.into(), relation));
        self
    }

    /// Set NSM prime definition
    pub fn set_prime_definition(&mut self, primes: Vec<(SemanticPrime, SemanticPrime)>) -> &mut Self {
        self.prime_definition = primes;
        self.update_encoding();
        self
    }

    /// Update encoding based on elements and primes
    fn update_encoding(&mut self) {
        let mut vectors = Vec::new();

        // Add core element encodings
        for elem in &self.core_elements {
            vectors.push(elem.encoding);
        }

        // Add prime definition encodings
        for (p1, p2) in &self.prime_definition {
            let semantics = UniversalSemantics::new();
            let bound = semantics.get_prime(*p1).bind(semantics.get_prime(*p2));
            vectors.push(bound);
        }

        if !vectors.is_empty() {
            self.encoding = HV16::bundle(&vectors);
        }
    }

    /// Check if a word evokes this frame
    pub fn is_evoked_by(&self, word: &str) -> bool {
        let word_lower = word.to_lowercase();
        self.lexical_units.iter().any(|lu| lu.to_lowercase() == word_lower)
    }

    /// Get all elements (core + non-core)
    pub fn all_elements(&self) -> impl Iterator<Item = &FrameElement> {
        self.core_elements.iter().chain(self.non_core_elements.iter())
    }

    /// Clone with bound elements
    pub fn instantiate(&self) -> FrameInstance {
        FrameInstance {
            frame_name: self.name.clone(),
            core_bindings: self.core_elements.iter()
                .map(|e| (e.name.clone(), e.filler))
                .collect(),
            non_core_bindings: self.non_core_elements.iter()
                .map(|e| (e.name.clone(), e.filler))
                .collect(),
            frame_encoding: self.encoding,
        }
    }

    /// Compute instantiated encoding with fillers
    pub fn instantiated_encoding(&self) -> HV16 {
        let mut vectors = vec![self.encoding];

        for elem in self.core_elements.iter().chain(self.non_core_elements.iter()) {
            if let Some(bound) = elem.bound_encoding() {
                vectors.push(bound);
            }
        }

        HV16::bundle(&vectors)
    }
}

/// An instantiated frame (with role bindings)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameInstance {
    /// Name of the frame
    pub frame_name: String,

    /// Core element bindings
    pub core_bindings: HashMap<String, Option<HV16>>,

    /// Non-core element bindings
    pub non_core_bindings: HashMap<String, Option<HV16>>,

    /// Frame encoding
    pub frame_encoding: HV16,
}

impl FrameInstance {
    /// Check if all core elements are bound
    pub fn is_complete(&self) -> bool {
        self.core_bindings.values().all(|v| v.is_some())
    }

    /// Compute completeness ratio
    pub fn completeness(&self) -> f32 {
        if self.core_bindings.is_empty() {
            return 1.0;
        }
        let bound = self.core_bindings.values().filter(|v| v.is_some()).count();
        bound as f32 / self.core_bindings.len() as f32
    }

    /// Get combined encoding
    pub fn encoding(&self) -> HV16 {
        let mut vectors = vec![self.frame_encoding];

        for filler in self.core_bindings.values().flatten() {
            vectors.push(*filler);
        }

        HV16::bundle(&vectors)
    }
}

// =============================================================================
// FRAME LIBRARY
// =============================================================================

/// Library of semantic frames
#[derive(Debug, Clone)]
pub struct FrameLibrary {
    /// All frames indexed by name
    frames: HashMap<String, SemanticFrame>,

    /// Lexical unit to frame mapping
    lu_to_frame: HashMap<String, Vec<String>>,

    /// Frame encodings for similarity search
    frame_encodings: Vec<(String, HV16)>,
}

impl FrameLibrary {
    /// Create new frame library
    pub fn new() -> Self {
        let mut lib = Self {
            frames: HashMap::new(),
            lu_to_frame: HashMap::new(),
            frame_encodings: Vec::new(),
        };

        // Initialize with core frames
        lib.initialize_core_frames();

        lib
    }

    /// Initialize core semantic frames
    fn initialize_core_frames(&mut self) {
        // =====================================================================
        // TRANSFER Frame
        // =====================================================================
        let mut transfer = SemanticFrame::new(
            "TRANSFER",
            "An event in which a Theme is transferred from a Donor to a Recipient"
        );

        transfer.add_core_element(FrameElement::new(
            "Donor",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Have, SemanticPrime::Before],
        ));

        transfer.add_core_element(FrameElement::new(
            "Recipient",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Have, SemanticPrime::After],
        ));

        transfer.add_core_element(FrameElement::new(
            "Theme",
            "Physical_object",
            true,
            vec![SemanticPrime::Something, SemanticPrime::Move],
        ));

        transfer.add_non_core_element(FrameElement::new(
            "Manner", "State", false, vec![SemanticPrime::Like],
        ));

        transfer.add_non_core_element(FrameElement::new(
            "Time", "Time", false, vec![SemanticPrime::When],
        ));

        transfer.add_non_core_element(FrameElement::new(
            "Place", "Location", false, vec![SemanticPrime::Where],
        ));

        transfer
            .add_lexical_unit("give")
            .add_lexical_unit("gave")
            .add_lexical_unit("given")
            .add_lexical_unit("hand")
            .add_lexical_unit("pass")
            .add_lexical_unit("transfer")
            .add_lexical_unit("donate")
            .add_lexical_unit("deliver");

        transfer.set_prime_definition(vec![
            (SemanticPrime::Someone, SemanticPrime::Do),
            (SemanticPrime::Something, SemanticPrime::Move),
            (SemanticPrime::Someone, SemanticPrime::Have),
        ]);

        self.add_frame(transfer);

        // =====================================================================
        // COMMERCIAL_TRANSACTION Frame
        // =====================================================================
        let mut commercial = SemanticFrame::new(
            "COMMERCIAL_TRANSACTION",
            "A transfer of goods from Seller to Buyer in exchange for money"
        );

        commercial.add_core_element(FrameElement::new(
            "Buyer",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Want, SemanticPrime::Something],
        ));

        commercial.add_core_element(FrameElement::new(
            "Seller",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Have, SemanticPrime::Something],
        ));

        commercial.add_core_element(FrameElement::new(
            "Goods",
            "Physical_object",
            true,
            vec![SemanticPrime::Something, SemanticPrime::Good],
        ));

        commercial.add_core_element(FrameElement::new(
            "Money",
            "Physical_object",
            true,
            vec![SemanticPrime::Something, SemanticPrime::Much],
        ));

        commercial
            .add_lexical_unit("buy")
            .add_lexical_unit("bought")
            .add_lexical_unit("purchase")
            .add_lexical_unit("sell")
            .add_lexical_unit("sold")
            .add_lexical_unit("trade")
            .add_lexical_unit("pay");

        commercial.add_relation("TRANSFER", FrameRelation::InheritsFrom);

        self.add_frame(commercial);

        // =====================================================================
        // MOTION Frame
        // =====================================================================
        let mut motion = SemanticFrame::new(
            "MOTION",
            "A Mover moves from a Source along a Path to a Goal"
        );

        motion.add_core_element(FrameElement::new(
            "Mover",
            "Physical_object",
            true,
            vec![SemanticPrime::Something, SemanticPrime::Move],
        ));

        motion.add_non_core_element(FrameElement::new(
            "Source",
            "Location",
            false,
            vec![SemanticPrime::Where, SemanticPrime::Before],
        ));

        motion.add_non_core_element(FrameElement::new(
            "Goal",
            "Location",
            false,
            vec![SemanticPrime::Where, SemanticPrime::After],
        ));

        motion.add_non_core_element(FrameElement::new(
            "Path",
            "Location",
            false,
            vec![SemanticPrime::Where, SemanticPrime::Side],
        ));

        motion.add_non_core_element(FrameElement::new(
            "Manner",
            "State",
            false,
            vec![SemanticPrime::Like],
        ));

        motion
            .add_lexical_unit("go")
            .add_lexical_unit("went")
            .add_lexical_unit("gone")
            .add_lexical_unit("move")
            .add_lexical_unit("travel")
            .add_lexical_unit("walk")
            .add_lexical_unit("run")
            .add_lexical_unit("fly")
            .add_lexical_unit("come")
            .add_lexical_unit("arrive")
            .add_lexical_unit("leave");

        self.add_frame(motion);

        // =====================================================================
        // COMMUNICATION Frame
        // =====================================================================
        let mut communication = SemanticFrame::new(
            "COMMUNICATION",
            "A Speaker conveys a Message to an Addressee about a Topic"
        );

        communication.add_core_element(FrameElement::new(
            "Speaker",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Say],
        ));

        communication.add_core_element(FrameElement::new(
            "Addressee",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Hear],
        ));

        communication.add_core_element(FrameElement::new(
            "Message",
            "State",
            true,
            vec![SemanticPrime::Words, SemanticPrime::Something],
        ));

        communication.add_non_core_element(FrameElement::new(
            "Topic",
            "State",
            false,
            vec![SemanticPrime::Something, SemanticPrime::Think],
        ));

        communication
            .add_lexical_unit("tell")
            .add_lexical_unit("told")
            .add_lexical_unit("say")
            .add_lexical_unit("said")
            .add_lexical_unit("speak")
            .add_lexical_unit("talk")
            .add_lexical_unit("ask")
            .add_lexical_unit("answer")
            .add_lexical_unit("explain")
            .add_lexical_unit("describe");

        self.add_frame(communication);

        // =====================================================================
        // PERCEPTION Frame
        // =====================================================================
        let mut perception = SemanticFrame::new(
            "PERCEPTION",
            "A Perceiver becomes aware of a Stimulus through sensory experience"
        );

        perception.add_core_element(FrameElement::new(
            "Perceiver",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::See],
        ));

        perception.add_core_element(FrameElement::new(
            "Stimulus",
            "Physical_object",
            true,
            vec![SemanticPrime::Something, SemanticPrime::ThereIs],
        ));

        perception
            .add_lexical_unit("see")
            .add_lexical_unit("saw")
            .add_lexical_unit("seen")
            .add_lexical_unit("hear")
            .add_lexical_unit("heard")
            .add_lexical_unit("feel")
            .add_lexical_unit("felt")
            .add_lexical_unit("smell")
            .add_lexical_unit("taste")
            .add_lexical_unit("notice")
            .add_lexical_unit("perceive")
            .add_lexical_unit("observe")
            .add_lexical_unit("watch");

        self.add_frame(perception);

        // =====================================================================
        // CAUSATION Frame
        // =====================================================================
        let mut causation = SemanticFrame::new(
            "CAUSATION",
            "A Cause brings about an Effect"
        );

        causation.add_core_element(FrameElement::new(
            "Cause",
            "Event",
            true,
            vec![SemanticPrime::Something, SemanticPrime::Because],
        ));

        causation.add_core_element(FrameElement::new(
            "Effect",
            "Event",
            true,
            vec![SemanticPrime::Something, SemanticPrime::Happen],
        ));

        causation
            .add_lexical_unit("cause")
            .add_lexical_unit("caused")
            .add_lexical_unit("make")
            .add_lexical_unit("made")
            .add_lexical_unit("result")
            .add_lexical_unit("lead")
            .add_lexical_unit("bring");

        self.add_frame(causation);

        // =====================================================================
        // COGNITION Frame
        // =====================================================================
        let mut cognition = SemanticFrame::new(
            "COGNITION",
            "A Cognizer has a mental state about some Content"
        );

        cognition.add_core_element(FrameElement::new(
            "Cognizer",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Think],
        ));

        cognition.add_core_element(FrameElement::new(
            "Content",
            "State",
            true,
            vec![SemanticPrime::Something, SemanticPrime::Know],
        ));

        cognition
            .add_lexical_unit("think")
            .add_lexical_unit("thought")
            .add_lexical_unit("know")
            .add_lexical_unit("knew")
            .add_lexical_unit("believe")
            .add_lexical_unit("understand")
            .add_lexical_unit("realize")
            .add_lexical_unit("remember")
            .add_lexical_unit("forget")
            .add_lexical_unit("consider");

        self.add_frame(cognition);

        // =====================================================================
        // DESIRING Frame
        // =====================================================================
        let mut desiring = SemanticFrame::new(
            "DESIRING",
            "An Experiencer wants an Event to occur or to obtain a Focal_participant"
        );

        desiring.add_core_element(FrameElement::new(
            "Experiencer",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Want],
        ));

        desiring.add_core_element(FrameElement::new(
            "Event",
            "Event",
            true,
            vec![SemanticPrime::Something, SemanticPrime::Happen],
        ));

        desiring
            .add_lexical_unit("want")
            .add_lexical_unit("wanted")
            .add_lexical_unit("wish")
            .add_lexical_unit("desire")
            .add_lexical_unit("hope")
            .add_lexical_unit("long")
            .add_lexical_unit("yearn")
            .add_lexical_unit("need");

        self.add_frame(desiring);

        // =====================================================================
        // JUDGMENT Frame
        // =====================================================================
        let mut judgment = SemanticFrame::new(
            "JUDGMENT",
            "A Judge evaluates an Evaluee based on some Reason"
        );

        judgment.add_core_element(FrameElement::new(
            "Judge",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Think],
        ));

        judgment.add_core_element(FrameElement::new(
            "Evaluee",
            "Physical_object",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Do],
        ));

        judgment.add_non_core_element(FrameElement::new(
            "Reason",
            "State",
            false,
            vec![SemanticPrime::Because, SemanticPrime::Something],
        ));

        judgment
            .add_lexical_unit("judge")
            .add_lexical_unit("judged")
            .add_lexical_unit("praise")
            .add_lexical_unit("blame")
            .add_lexical_unit("criticize")
            .add_lexical_unit("approve")
            .add_lexical_unit("condemn")
            .add_lexical_unit("admire");

        self.add_frame(judgment);

        // =====================================================================
        // BEING_LOCATED Frame
        // =====================================================================
        let mut being_located = SemanticFrame::new(
            "BEING_LOCATED",
            "A Theme is located at a Location"
        );

        being_located.add_core_element(FrameElement::new(
            "Theme",
            "Physical_object",
            true,
            vec![SemanticPrime::Something, SemanticPrime::ThereIs],
        ));

        being_located.add_core_element(FrameElement::new(
            "Location",
            "Location",
            true,
            vec![SemanticPrime::Where, SemanticPrime::Here],
        ));

        being_located
            .add_lexical_unit("be")
            .add_lexical_unit("is")
            .add_lexical_unit("was")
            .add_lexical_unit("were")
            .add_lexical_unit("stand")
            .add_lexical_unit("sit")
            .add_lexical_unit("lie")
            .add_lexical_unit("located");

        self.add_frame(being_located);

        // =====================================================================
        // UNIVERSAL CONCEPTUAL LANGUAGE (UCL) FRAMES
        // Revolutionary Improvement: Cross-domain semantic structures
        // =====================================================================

        // =====================================================================
        // CONFLICT Frame - Opposing Goals
        // =====================================================================
        let mut conflict = SemanticFrame::new(
            "CONFLICT",
            "Two or more parties with incompatible goals competing for resources or outcomes"
        );

        conflict.add_core_element(FrameElement::new(
            "Party_1",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Want, SemanticPrime::Something],
        ));

        conflict.add_core_element(FrameElement::new(
            "Party_2",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Want, SemanticPrime::Something],
        ));

        conflict.add_core_element(FrameElement::new(
            "Issue",
            "Physical_object",
            true,
            vec![SemanticPrime::Something, SemanticPrime::Bad],
        ));

        conflict.add_non_core_element(FrameElement::new(
            "Resolution",
            "Event",
            false,
            vec![SemanticPrime::Happen, SemanticPrime::Good],
        ));

        conflict.add_non_core_element(FrameElement::new(
            "Outcome",
            "State",
            false,
            vec![SemanticPrime::Be, SemanticPrime::After],
        ));

        conflict
            .add_lexical_unit("conflict")
            .add_lexical_unit("fight")
            .add_lexical_unit("dispute")
            .add_lexical_unit("struggle")
            .add_lexical_unit("battle")
            .add_lexical_unit("compete")
            .add_lexical_unit("clash")
            .add_lexical_unit("oppose");

        conflict.set_prime_definition(vec![
            (SemanticPrime::Someone, SemanticPrime::Want),
            (SemanticPrime::Someone, SemanticPrime::Not),  // Opposing
            (SemanticPrime::Can, SemanticPrime::Not),      // Incompatibility
        ]);

        self.add_frame(conflict);

        // =====================================================================
        // COOPERATION Frame - Shared Goals
        // =====================================================================
        let mut cooperation = SemanticFrame::new(
            "COOPERATION",
            "Multiple agents working together toward a shared goal"
        );

        cooperation.add_core_element(FrameElement::new(
            "Cooperators",
            "Sentient",
            true,
            vec![SemanticPrime::People, SemanticPrime::Do, SemanticPrime::Same],
        ));

        cooperation.add_core_element(FrameElement::new(
            "Goal",
            "State",
            true,
            vec![SemanticPrime::Something, SemanticPrime::Good, SemanticPrime::Want],
        ));

        cooperation.add_core_element(FrameElement::new(
            "Activity",
            "Event",
            true,
            vec![SemanticPrime::Do, SemanticPrime::Something],
        ));

        cooperation.add_non_core_element(FrameElement::new(
            "Benefit",
            "State",
            false,
            vec![SemanticPrime::Good, SemanticPrime::All],
        ));

        cooperation
            .add_lexical_unit("cooperate")
            .add_lexical_unit("collaborate")
            .add_lexical_unit("work together")
            .add_lexical_unit("team")
            .add_lexical_unit("partner")
            .add_lexical_unit("join")
            .add_lexical_unit("unite")
            .add_lexical_unit("coordinate");

        cooperation.set_prime_definition(vec![
            (SemanticPrime::People, SemanticPrime::Do),
            (SemanticPrime::Same, SemanticPrime::Want),   // Shared goals
            (SemanticPrime::Good, SemanticPrime::All),    // Mutual benefit
        ]);

        self.add_frame(cooperation);

        // =====================================================================
        // FEEDBACK_LOOP Frame - Cyclical Causation
        // =====================================================================
        let mut feedback = SemanticFrame::new(
            "FEEDBACK_LOOP",
            "A cyclical process where output influences subsequent input"
        );

        feedback.add_core_element(FrameElement::new(
            "System",
            "Physical_object",
            true,
            vec![SemanticPrime::Something, SemanticPrime::Do],
        ));

        feedback.add_core_element(FrameElement::new(
            "Output",
            "Event",
            true,
            vec![SemanticPrime::Happen, SemanticPrime::After],
        ));

        feedback.add_core_element(FrameElement::new(
            "Effect_on_Input",
            "Event",
            true,
            vec![SemanticPrime::Happen, SemanticPrime::Because],
        ));

        feedback.add_non_core_element(FrameElement::new(
            "Polarity",
            "State",
            false,
            vec![SemanticPrime::More, SemanticPrime::Little],  // Positive or negative
        ));

        feedback.add_non_core_element(FrameElement::new(
            "Stability",
            "State",
            false,
            vec![SemanticPrime::Same, SemanticPrime::LongTime],
        ));

        feedback
            .add_lexical_unit("feedback")
            .add_lexical_unit("loop")
            .add_lexical_unit("cycle")
            .add_lexical_unit("iteration")
            .add_lexical_unit("recursion")
            .add_lexical_unit("reinforce")
            .add_lexical_unit("amplify")
            .add_lexical_unit("regulate");

        feedback.set_prime_definition(vec![
            (SemanticPrime::Something, SemanticPrime::Happen),
            (SemanticPrime::Because, SemanticPrime::Before),  // Caused by prior output
            (SemanticPrime::After, SemanticPrime::Same),      // Cyclical
        ]);

        self.add_frame(feedback);

        // =====================================================================
        // NORM_ENFORCEMENT Frame - Social Rules and Sanctions
        // =====================================================================
        let mut norm_enforcement = SemanticFrame::new(
            "NORM_ENFORCEMENT",
            "A social norm is maintained through sanctions or rewards"
        );

        norm_enforcement.add_core_element(FrameElement::new(
            "Norm",
            "State",
            true,
            vec![SemanticPrime::True, SemanticPrime::Good, SemanticPrime::Do],
        ));

        norm_enforcement.add_core_element(FrameElement::new(
            "Enforcer",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Can, SemanticPrime::Do],
        ));

        norm_enforcement.add_core_element(FrameElement::new(
            "Subject",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Do],
        ));

        norm_enforcement.add_non_core_element(FrameElement::new(
            "Sanction",
            "Event",
            false,
            vec![SemanticPrime::Bad, SemanticPrime::Happen],
        ));

        norm_enforcement.add_non_core_element(FrameElement::new(
            "Reward",
            "Event",
            false,
            vec![SemanticPrime::Good, SemanticPrime::Happen],
        ));

        norm_enforcement
            .add_lexical_unit("enforce")
            .add_lexical_unit("rule")
            .add_lexical_unit("punish")
            .add_lexical_unit("reward")
            .add_lexical_unit("sanction")
            .add_lexical_unit("regulate")
            .add_lexical_unit("govern")
            .add_lexical_unit("comply");

        norm_enforcement.set_prime_definition(vec![
            (SemanticPrime::Someone, SemanticPrime::Do),
            (SemanticPrime::Not, SemanticPrime::Good),     // Violation
            (SemanticPrime::Bad, SemanticPrime::Happen),   // Consequence
        ]);

        self.add_frame(norm_enforcement);

        // =====================================================================
        // ADAPTATION Frame - System Adjustment
        // =====================================================================
        let mut adaptation = SemanticFrame::new(
            "ADAPTATION",
            "A system changes its behavior in response to environmental conditions"
        );

        adaptation.add_core_element(FrameElement::new(
            "Adapter",
            "Physical_object",
            true,
            vec![SemanticPrime::Something, SemanticPrime::Live],
        ));

        adaptation.add_core_element(FrameElement::new(
            "Environment",
            "Location",
            true,
            vec![SemanticPrime::Where, SemanticPrime::Something],
        ));

        adaptation.add_core_element(FrameElement::new(
            "Change",
            "Event",
            true,
            vec![SemanticPrime::Happen, SemanticPrime::Other],
        ));

        adaptation.add_non_core_element(FrameElement::new(
            "Fitness",
            "State",
            false,
            vec![SemanticPrime::Good, SemanticPrime::More],
        ));

        adaptation.add_non_core_element(FrameElement::new(
            "Pressure",
            "Event",
            false,
            vec![SemanticPrime::Because, SemanticPrime::Happen],
        ));

        adaptation
            .add_lexical_unit("adapt")
            .add_lexical_unit("adjust")
            .add_lexical_unit("evolve")
            .add_lexical_unit("change")
            .add_lexical_unit("modify")
            .add_lexical_unit("accommodate")
            .add_lexical_unit("respond")
            .add_lexical_unit("fit");

        adaptation.set_prime_definition(vec![
            (SemanticPrime::Something, SemanticPrime::Live),
            (SemanticPrime::Other, SemanticPrime::Happen),   // Environment changes
            (SemanticPrime::Something, SemanticPrime::Other), // Adapter changes
        ]);

        self.add_frame(adaptation);

        // =====================================================================
        // TRADE Frame - Mutual Exchange (extends COMMERCIAL_TRANSACTION)
        // =====================================================================
        let mut trade = SemanticFrame::new(
            "TRADE",
            "Mutual exchange of goods, services, or value between parties"
        );

        trade.add_core_element(FrameElement::new(
            "Trader_1",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Have, SemanticPrime::Something],
        ));

        trade.add_core_element(FrameElement::new(
            "Trader_2",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Have, SemanticPrime::Something],
        ));

        trade.add_core_element(FrameElement::new(
            "Item_1",
            "Physical_object",
            true,
            vec![SemanticPrime::Something, SemanticPrime::Good],
        ));

        trade.add_core_element(FrameElement::new(
            "Item_2",
            "Physical_object",
            true,
            vec![SemanticPrime::Something, SemanticPrime::Good],
        ));

        trade.add_non_core_element(FrameElement::new(
            "Value_Equivalence",
            "State",
            false,
            vec![SemanticPrime::Same, SemanticPrime::Much],
        ));

        trade.add_non_core_element(FrameElement::new(
            "Agreement",
            "Event",
            false,
            vec![SemanticPrime::Say, SemanticPrime::True],
        ));

        trade
            .add_lexical_unit("trade")
            .add_lexical_unit("exchange")
            .add_lexical_unit("swap")
            .add_lexical_unit("barter")
            .add_lexical_unit("reciprocate");

        trade.add_relation("COMMERCIAL_TRANSACTION", FrameRelation::InheritsFrom);
        trade.add_relation("TRANSFER", FrameRelation::Uses);

        trade.set_prime_definition(vec![
            (SemanticPrime::Someone, SemanticPrime::Have),
            (SemanticPrime::Something, SemanticPrime::Move),   // Transfer
            (SemanticPrime::Good, SemanticPrime::All),         // Mutual benefit
        ]);

        self.add_frame(trade);
    }

    /// Add a frame to the library
    pub fn add_frame(&mut self, frame: SemanticFrame) {
        // Index lexical units
        for lu in &frame.lexical_units {
            self.lu_to_frame
                .entry(lu.to_lowercase())
                .or_default()
                .push(frame.name.clone());
        }

        // Add encoding for similarity search
        self.frame_encodings.push((frame.name.clone(), frame.encoding));

        // Store frame
        self.frames.insert(frame.name.clone(), frame);
    }

    /// Get frame by name
    pub fn get(&self, name: &str) -> Option<&SemanticFrame> {
        self.frames.get(name)
    }

    /// Get mutable frame by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut SemanticFrame> {
        self.frames.get_mut(name)
    }

    /// Get frames evoked by a word
    pub fn evoked_by(&self, word: &str) -> Vec<&SemanticFrame> {
        let word_lower = word.to_lowercase();
        self.lu_to_frame
            .get(&word_lower)
            .map(|names| {
                names.iter()
                    .filter_map(|n| self.frames.get(n))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find most similar frame to encoding
    pub fn find_similar(&self, encoding: &HV16, top_k: usize) -> Vec<(&SemanticFrame, f32)> {
        let mut similarities: Vec<_> = self.frame_encodings.iter()
            .map(|(name, enc)| {
                let sim = encoding.similarity(enc);
                (name.as_str(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        similarities.iter()
            .take(top_k)
            .filter_map(|(name, sim)| {
                self.frames.get(*name).map(|f| (f, *sim))
            })
            .collect()
    }

    /// Count frames
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Get all frame names
    pub fn frame_names(&self) -> impl Iterator<Item = &String> {
        self.frames.keys()
    }
}

impl Default for FrameLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// FRAME ACTIVATOR
// =============================================================================

/// Activates frames from text input
pub struct FrameActivator {
    /// Frame library
    library: FrameLibrary,
}

impl FrameActivator {
    /// Create new frame activator
    pub fn new() -> Self {
        Self {
            library: FrameLibrary::new(),
        }
    }

    /// Create with custom library
    pub fn with_library(library: FrameLibrary) -> Self {
        Self { library }
    }

    /// Activate frames from a list of words
    pub fn activate(&self, words: &[String]) -> Vec<FrameInstance> {
        let mut instances = Vec::new();

        for word in words {
            let frames = self.library.evoked_by(word);
            for frame in frames {
                instances.push(frame.instantiate());
            }
        }

        instances
    }

    /// Activate frames from text (simple tokenization)
    pub fn activate_from_text(&self, text: &str) -> Vec<FrameInstance> {
        let words: Vec<String> = text
            .to_lowercase()
            .split(|c: char| !c.is_alphabetic())
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        self.activate(&words)
    }

    /// Get library reference
    pub fn library(&self) -> &FrameLibrary {
        &self.library
    }

    /// Get mutable library reference
    pub fn library_mut(&mut self) -> &mut FrameLibrary {
        &mut self.library
    }
}

impl Default for FrameActivator {
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
    fn test_frame_element_creation() {
        let elem = FrameElement::new(
            "Buyer",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Want],
        );

        assert_eq!(elem.name, "Buyer");
        assert_eq!(elem.semantic_type, "Sentient");
        assert!(elem.is_core);
        assert_eq!(elem.prime_structure.len(), 2);
    }

    #[test]
    fn test_frame_library_initialization() {
        let lib = FrameLibrary::new();

        assert!(!lib.is_empty());
        assert!(lib.len() >= 10);  // At least 10 core frames

        // Check specific frames exist
        assert!(lib.get("TRANSFER").is_some());
        assert!(lib.get("COMMERCIAL_TRANSACTION").is_some());
        assert!(lib.get("MOTION").is_some());
        assert!(lib.get("COMMUNICATION").is_some());
        assert!(lib.get("PERCEPTION").is_some());
    }

    #[test]
    fn test_lexical_unit_evocation() {
        let lib = FrameLibrary::new();

        // "give" should evoke TRANSFER
        let frames = lib.evoked_by("give");
        assert!(!frames.is_empty());
        assert!(frames.iter().any(|f| f.name == "TRANSFER"));

        // "buy" should evoke COMMERCIAL_TRANSACTION
        let frames = lib.evoked_by("buy");
        assert!(!frames.is_empty());
        assert!(frames.iter().any(|f| f.name == "COMMERCIAL_TRANSACTION"));

        // "think" should evoke COGNITION
        let frames = lib.evoked_by("think");
        assert!(!frames.is_empty());
        assert!(frames.iter().any(|f| f.name == "COGNITION"));
    }

    #[test]
    fn test_frame_instance_completeness() {
        let lib = FrameLibrary::new();

        let transfer = lib.get("TRANSFER").unwrap();
        let instance = transfer.instantiate();

        // Should not be complete (no fillers bound)
        assert!(!instance.is_complete());
        assert!(instance.completeness() < 1.0);
    }

    #[test]
    fn test_frame_element_binding() {
        let mut elem = FrameElement::new(
            "Agent",
            "Sentient",
            true,
            vec![SemanticPrime::Someone, SemanticPrime::Do],
        );

        let filler = HV16::random(42);
        elem.bind_filler(filler);

        assert!(elem.filler.is_some());
        assert!(elem.bound_encoding().is_some());
    }

    #[test]
    fn test_frame_activator() {
        let activator = FrameActivator::new();

        let instances = activator.activate_from_text("The child gave the dog a bone");

        // Should activate TRANSFER (from "gave")
        assert!(!instances.is_empty());
        assert!(instances.iter().any(|i| i.frame_name == "TRANSFER"));
    }

    #[test]
    fn test_frame_similarity_search() {
        let lib = FrameLibrary::new();

        // Get TRANSFER encoding
        let transfer = lib.get("TRANSFER").unwrap();

        // Find similar frames
        let similar = lib.find_similar(&transfer.encoding, 3);

        // TRANSFER should be most similar to itself
        assert!(!similar.is_empty());
        assert_eq!(similar[0].0.name, "TRANSFER");
    }

    #[test]
    fn test_frame_relations() {
        let lib = FrameLibrary::new();

        let commercial = lib.get("COMMERCIAL_TRANSACTION").unwrap();

        // Should have relation to TRANSFER
        assert!(!commercial.relations.is_empty());
        assert!(commercial.relations.iter().any(|(target, rel)| {
            target == "TRANSFER" && *rel == FrameRelation::InheritsFrom
        }));
    }

    #[test]
    fn test_semantic_type_checking() {
        let elem = FrameElement::new(
            "Buyer",
            "Sentient",
            true,
            vec![SemanticPrime::Someone],
        );

        assert!(elem.can_accept("person"));
        assert!(elem.can_accept("animal"));
        assert!(!elem.can_accept("place"));
    }

    #[test]
    fn test_multiple_lexical_units() {
        let lib = FrameLibrary::new();

        // All tense forms should evoke the same frame
        let go_frames = lib.evoked_by("go");
        let went_frames = lib.evoked_by("went");
        let gone_frames = lib.evoked_by("gone");

        assert!(!go_frames.is_empty());
        assert!(!went_frames.is_empty());
        assert!(!gone_frames.is_empty());

        // All should be MOTION
        assert!(go_frames.iter().any(|f| f.name == "MOTION"));
        assert!(went_frames.iter().any(|f| f.name == "MOTION"));
        assert!(gone_frames.iter().any(|f| f.name == "MOTION"));
    }

    #[test]
    fn test_nsm_prime_integration() {
        let lib = FrameLibrary::new();

        let transfer = lib.get("TRANSFER").unwrap();

        // Should have prime definition
        assert!(!transfer.prime_definition.is_empty());

        // Core elements should have prime structures
        for elem in &transfer.core_elements {
            assert!(!elem.prime_structure.is_empty());
        }
    }
}
