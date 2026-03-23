// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Structured Knowledge Extraction
//!
//! Extracts (entity, relation, event) tuples from raw text using
//! semantic role decomposition. Extends the counterfactual module's
//! 7-role system (Agent, Patient, Instrument, Context, Goal, Source,
//! Destination) to parse natural language into structured facts.
//!
//! Science: Fillmore (1968) case grammar, Palmer et al. (2005) PropBank

use std::collections::HashMap;

// ── Types ──────────────────────────────────────────────────────────────────

/// Semantic roles for fact decomposition (extends counterfactual/semantic_roles.rs)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticRole {
    /// Who performs the action
    Agent,
    /// Who/what is affected
    Patient,
    /// Tool or means used
    Instrument,
    /// Surrounding circumstances
    Context,
    /// Purpose or end-state
    Goal,
    /// Origin point
    Source,
    /// End point
    Destination,
    /// When the event occurs
    Temporal,
    /// Where the event occurs
    Location,
    /// Why the event occurs
    Cause,
    /// What results from the event
    Result,
    /// Subject calls/invokes Object (function invocation)
    Calls,
    /// Subject implements Object (trait/interface implementation)
    Implements,
    /// Subject depends on Object (import/use dependency)
    DependsOn,
    /// Subject returns Object (function → return type)
    ReturnsType,
    /// Subject error was fixed by Object strategy
    FixedBy,
    #[cfg(feature = "therapeutic")]
    TherapeuticTarget,
    #[cfg(feature = "therapeutic")]
    ProtectiveFactor,
    #[cfg(feature = "therapeutic")]
    RiskFactor,
}

/// Entity types for knowledge graph nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Organization,
    Place,
    Event,
    Concept,
    Quantity,
    Temporal,
    Artifact,
    /// A physical process or phenomenon
    Process,
    /// An abstract relation or property
    Property,
    /// A function, method, or closure
    Function,
    /// A struct, enum, trait, class, or type alias
    Type,
    /// A module, crate, package, or namespace
    Module,
    /// A compiler error code or pattern (e.g., E0308)
    ErrorPattern,
    /// A fragment of source code
    CodeSnippet,
    #[cfg(feature = "therapeutic")]
    ClinicalConcept,
    #[cfg(feature = "therapeutic")]
    Symptom,
    #[cfg(feature = "therapeutic")]
    Intervention,
}

/// An extracted entity from text
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    /// Surface text span
    pub text: String,
    /// Entity type classification
    pub entity_type: EntityType,
    /// Confidence in extraction (0.0–1.0)
    pub confidence: f32,
    /// Character offset in source text
    pub offset: usize,
}

/// An extracted relation between entities
#[derive(Debug, Clone)]
pub struct ExtractedRelation {
    /// Subject entity text
    pub subject: String,
    /// Relation verb/predicate
    pub predicate: String,
    /// Object entity text
    pub object: String,
    /// Semantic role of the subject
    pub subject_role: SemanticRole,
    /// Semantic role of the object
    pub object_role: SemanticRole,
    /// Whether this relation is causal (implies DAG edge)
    pub is_causal: bool,
    /// Whether this is negated ("X did NOT do Y")
    pub is_negated: bool,
    /// Confidence in extraction (0.0–1.0)
    pub confidence: f32,
}

/// A structured fact extracted from text — the atomic unit of knowledge
#[derive(Debug, Clone)]
pub struct ExtractedFact {
    /// The entities involved
    pub entities: Vec<ExtractedEntity>,
    /// The relations between them
    pub relations: Vec<ExtractedRelation>,
    /// Role assignments: entity_text → semantic_role
    pub role_map: HashMap<String, SemanticRole>,
    /// Source text this fact was extracted from
    pub source_text: String,
    /// Overall extraction confidence
    pub confidence: f32,
}

// ── Extraction Engine ──────────────────────────────────────────────────────

/// Pattern-based knowledge extractor
///
/// Uses a cascade of heuristic matchers to identify entities, relations,
/// and causal structures in natural language. This is NOT full NLP — it's
/// a pragmatic extraction layer that leverages Symthaea's HDC similarity
/// for disambiguation.
pub struct KnowledgeExtractor {
    /// Causal verb patterns (sorted by frequency for early exit)
    causal_verbs: Vec<&'static str>,
    /// Negation markers
    negation_markers: Vec<&'static str>,
    /// Temporal markers for date/time extraction
    temporal_markers: Vec<&'static str>,
    /// Known entity cache: text → EntityType (grows over time)
    entity_cache: HashMap<String, EntityType>,
    /// Extraction statistics
    total_extractions: u64,
}

impl Default for KnowledgeExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeExtractor {
    pub fn new() -> Self {
        Self {
            causal_verbs: vec![
                "cause",
                "causes",
                "caused",
                "leads to",
                "led to",
                "results in",
                "resulted in",
                "trigger",
                "triggers",
                "triggered",
                "produce",
                "produces",
                "produced",
                "enable",
                "enables",
                "enabled",
                "prevent",
                "prevents",
                "prevented",
                "block",
                "blocks",
                "blocked",
                "disrupt",
                "disrupts",
                "disrupted",
                "destroy",
                "destroys",
                "destroyed",
                "create",
                "creates",
                "created",
                "force",
                "forces",
                "forced",
                "compels",
                "increase",
                "increases",
                "increased",
                "decrease",
                "decreases",
                "decreased",
                "accelerates",
                "decelerates",
                "amplifies",
                "dampens",
                "constrain",
                "constrains",
                "constrained",
                "require",
                "requires",
                "required",
                "implies",
                "implied",
                "entails",
                "entailed",
            ],
            negation_markers: vec![
                "not",
                "no",
                "never",
                "neither",
                "nor",
                "cannot",
                "can't",
                "won't",
                "wouldn't",
                "shouldn't",
                "doesn't",
                "didn't",
                "isn't",
                "aren't",
                "wasn't",
                "weren't",
                "without",
                "fails to",
                "failed to",
                "unable to",
            ],
            temporal_markers: vec![
                "before",
                "after",
                "during",
                "while",
                "when",
                "until",
                "since",
                "then",
                "now",
                "currently",
                "recently",
                "previously",
                "tomorrow",
                "yesterday",
                "today",
                "already",
                "soon",
            ],
            entity_cache: HashMap::new(),
            total_extractions: 0,
        }
    }

    /// Extract structured facts from a text input.
    ///
    /// Returns a list of extracted facts, each containing entities, relations,
    /// and role assignments. The extractor operates sentence-by-sentence.
    pub fn extract(&mut self, text: &str) -> Vec<ExtractedFact> {
        let sentences = self.split_sentences(text);
        let mut facts = Vec::new();

        for sentence in &sentences {
            if sentence.len() < 5 {
                continue; // Skip trivially short fragments
            }

            let entities = self.extract_entities(sentence);
            let relations = self.extract_relations(sentence, &entities);
            let role_map = self.assign_roles(sentence, &entities);
            let confidence = self.estimate_confidence(&entities, &relations);

            if !entities.is_empty() || !relations.is_empty() {
                facts.push(ExtractedFact {
                    entities,
                    relations,
                    role_map,
                    source_text: sentence.to_string(),
                    confidence,
                });
            }
        }

        self.total_extractions += facts.len() as u64;
        facts
    }

    /// Register a known entity for improved extraction
    pub fn register_entity(&mut self, text: &str, entity_type: EntityType) {
        self.entity_cache.insert(text.to_lowercase(), entity_type);
    }

    /// Total facts extracted since creation
    pub fn total_extractions(&self) -> u64 {
        self.total_extractions
    }

    // ── Code-Specific Extraction ──────────────────────────────────────────

    /// Extract code-specific entities from text containing source code or
    /// code-adjacent natural language (e.g., documentation, error messages).
    ///
    /// Detects function signatures, type declarations, module references,
    /// compiler error codes, and crate/package references.
    pub fn extract_code_entities(&self, text: &str) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();

        // Function signatures: "fn foo(", "def foo(", "function foo("
        for prefix in &["fn ", "def ", "function "] {
            let mut search_from = 0;
            while let Some(pos) = text[search_from..].find(prefix) {
                let abs_pos = search_from + pos;
                let after = &text[abs_pos + prefix.len()..];
                if let Some(name) = after.split('(').next() {
                    let name = name.trim();
                    if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                        entities.push(ExtractedEntity {
                            text: name.to_string(),
                            entity_type: EntityType::Function,
                            confidence: 0.9,
                            offset: abs_pos,
                        });
                    }
                }
                search_from = abs_pos + prefix.len();
            }
        }

        // Type declarations: "struct Foo", "enum Foo", "trait Foo", "class Foo", "type Foo"
        for keyword in &["struct ", "enum ", "trait ", "class ", "type "] {
            let mut search_from = 0;
            while let Some(pos) = text[search_from..].find(keyword) {
                let abs_pos = search_from + pos;
                let after = &text[abs_pos + keyword.len()..];
                if let Some(name) = after
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                {
                    let name = name.trim();
                    if !name.is_empty() && name.chars().next().map_or(false, |c| c.is_uppercase()) {
                        entities.push(ExtractedEntity {
                            text: name.to_string(),
                            entity_type: EntityType::Type,
                            confidence: 0.9,
                            offset: abs_pos,
                        });
                    }
                }
                search_from = abs_pos + keyword.len();
            }
        }

        // Module references: "mod foo", "use foo::", "import foo", "from foo import"
        {
            // "mod name"
            let mut search_from = 0;
            while let Some(pos) = text[search_from..].find("mod ") {
                let abs_pos = search_from + pos;
                let after = &text[abs_pos + 4..];
                if let Some(name) = after
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                {
                    let name = name.trim();
                    if !name.is_empty() {
                        entities.push(ExtractedEntity {
                            text: name.to_string(),
                            entity_type: EntityType::Module,
                            confidence: 0.85,
                            offset: abs_pos,
                        });
                    }
                }
                search_from = abs_pos + 4;
            }

            // "use foo::" — extract the root module
            let mut search_from = 0;
            while let Some(pos) = text[search_from..].find("use ") {
                let abs_pos = search_from + pos;
                let after = &text[abs_pos + 4..];
                if let Some(root) = after.split("::").next() {
                    let root = root.trim();
                    if !root.is_empty() && root.chars().all(|c| c.is_alphanumeric() || c == '_') {
                        entities.push(ExtractedEntity {
                            text: root.to_string(),
                            entity_type: EntityType::Module,
                            confidence: 0.85,
                            offset: abs_pos,
                        });
                    }
                }
                search_from = abs_pos + 4;
            }

            // "import foo"
            let mut search_from = 0;
            while let Some(pos) = text[search_from..].find("import ") {
                let abs_pos = search_from + pos;
                let after = &text[abs_pos + 7..];
                if let Some(name) = after
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                {
                    let name = name.trim();
                    if !name.is_empty() {
                        entities.push(ExtractedEntity {
                            text: name.to_string(),
                            entity_type: EntityType::Module,
                            confidence: 0.85,
                            offset: abs_pos,
                        });
                    }
                }
                search_from = abs_pos + 7;
            }

            // "from foo import"
            let mut search_from = 0;
            while let Some(pos) = text[search_from..].find("from ") {
                let abs_pos = search_from + pos;
                let after = &text[abs_pos + 5..];
                if after.contains(" import") {
                    if let Some(name) = after
                        .split(|c: char| !c.is_alphanumeric() && c != '_')
                        .next()
                    {
                        let name = name.trim();
                        if !name.is_empty() {
                            entities.push(ExtractedEntity {
                                text: name.to_string(),
                                entity_type: EntityType::Module,
                                confidence: 0.85,
                                offset: abs_pos,
                            });
                        }
                    }
                }
                search_from = abs_pos + 5;
            }
        }

        // Error codes: E followed by 4 digits (Rust compiler error pattern)
        {
            let bytes = text.as_bytes();
            for i in 0..bytes.len() {
                if bytes[i] == b'E'
                    && i + 4 < bytes.len()
                    && bytes[i + 1..=i + 4].iter().all(|b| b.is_ascii_digit())
                {
                    // Ensure it's not part of a longer identifier
                    let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
                    let after_ok = i + 5 >= bytes.len() || !bytes[i + 5].is_ascii_alphanumeric();
                    if before_ok && after_ok {
                        let code = &text[i..i + 5];
                        entities.push(ExtractedEntity {
                            text: code.to_string(),
                            entity_type: EntityType::ErrorPattern,
                            confidence: 0.9,
                            offset: i,
                        });
                    }
                }
            }
        }

        // Crate/package references: "crate::", "pub(crate)"
        {
            let mut search_from = 0;
            while let Some(pos) = text[search_from..].find("crate::") {
                let abs_pos = search_from + pos;
                let after = &text[abs_pos + 7..];
                if let Some(name) = after
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                {
                    let name = name.trim();
                    if !name.is_empty() {
                        entities.push(ExtractedEntity {
                            text: format!("crate::{}", name),
                            entity_type: EntityType::CodeSnippet,
                            confidence: 0.8,
                            offset: abs_pos,
                        });
                    }
                }
                search_from = abs_pos + 7;
            }
        }

        entities
    }

    /// Extract code-specific semantic relations from text.
    ///
    /// Detects call/invoke patterns, trait implementations, dependency
    /// relationships, return types, and fix/resolution patterns.
    pub fn extract_code_relations(&self, text: &str) -> Vec<ExtractedRelation> {
        let mut relations = Vec::new();
        let lower = text.to_lowercase();

        // "calls" / "invokes" patterns → Calls role
        for pattern in &["calls ", "invokes ", "calling "] {
            if let Some(pos) = lower.find(pattern) {
                let before = lower[..pos].trim();
                let after = lower[pos + pattern.len()..].trim();
                let subject = before.split_whitespace().last().unwrap_or("").to_string();
                let object = after
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                    .unwrap_or("")
                    .to_string();
                if !subject.is_empty() && !object.is_empty() {
                    relations.push(ExtractedRelation {
                        subject,
                        predicate: pattern.trim().to_string(),
                        object,
                        subject_role: SemanticRole::Agent,
                        object_role: SemanticRole::Calls,
                        is_causal: false,
                        is_negated: false,
                        confidence: 0.8,
                    });
                }
            }
        }

        // "impl Trait for Type" → Implements role
        {
            let mut search_from = 0;
            while let Some(pos) = text[search_from..].find("impl ") {
                let abs_pos = search_from + pos;
                let after = &text[abs_pos + 5..];
                if let Some(for_pos) = after.find(" for ") {
                    let trait_name = after[..for_pos].trim();
                    let type_after = after[for_pos + 5..].trim();
                    let type_name = type_after
                        .split(|c: char| !c.is_alphanumeric() && c != '_')
                        .next()
                        .unwrap_or("");
                    if !trait_name.is_empty() && !type_name.is_empty() {
                        relations.push(ExtractedRelation {
                            subject: type_name.to_string(),
                            predicate: "implements".to_string(),
                            object: trait_name.to_string(),
                            subject_role: SemanticRole::Agent,
                            object_role: SemanticRole::Implements,
                            is_causal: false,
                            is_negated: false,
                            confidence: 0.9,
                        });
                    }
                }
                search_from = abs_pos + 5;
            }
        }

        // "implements" in natural language → Implements role
        if let Some(pos) = lower.find("implements ") {
            // Skip if this is from "impl ... for" already handled above
            if !text[..pos].ends_with("impl ") {
                let before = lower[..pos].trim();
                let after = lower[pos + 11..].trim();
                let subject = before.split_whitespace().last().unwrap_or("").to_string();
                let object = after
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                    .unwrap_or("")
                    .to_string();
                if !subject.is_empty() && !object.is_empty() {
                    relations.push(ExtractedRelation {
                        subject,
                        predicate: "implements".to_string(),
                        object,
                        subject_role: SemanticRole::Agent,
                        object_role: SemanticRole::Implements,
                        is_causal: false,
                        is_negated: false,
                        confidence: 0.85,
                    });
                }
            }
        }

        // "use" / "import" / "depends on" → DependsOn role
        for pattern in &["depends on ", "use ", "import "] {
            if let Some(pos) = lower.find(pattern) {
                let before = lower[..pos].trim();
                let after = lower[pos + pattern.len()..].trim();
                let subject = before.split_whitespace().last().unwrap_or("").to_string();
                let object = after
                    .split(|c: char| !c.is_alphanumeric() && c != '_' && c != ':')
                    .next()
                    .unwrap_or("")
                    .to_string();
                if !subject.is_empty() && !object.is_empty() {
                    relations.push(ExtractedRelation {
                        subject,
                        predicate: pattern.trim().to_string(),
                        object,
                        subject_role: SemanticRole::Agent,
                        object_role: SemanticRole::DependsOn,
                        is_causal: false,
                        is_negated: false,
                        confidence: 0.8,
                    });
                }
            }
        }

        // "returns" / "->" / "→" → ReturnsType role
        for pattern in &["returns ", "-> ", "→ "] {
            if let Some(pos) = lower.find(pattern) {
                let before = lower[..pos].trim();
                let after_offset = pos + pattern.len();
                let after = if after_offset <= text.len() {
                    text[after_offset..].trim()
                } else {
                    ""
                };
                let subject = before.split_whitespace().last().unwrap_or("").to_string();
                let object = after
                    .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '<' && c != '>')
                    .next()
                    .unwrap_or("")
                    .to_string();
                if !subject.is_empty() && !object.is_empty() {
                    relations.push(ExtractedRelation {
                        subject,
                        predicate: pattern.trim().to_string(),
                        object,
                        subject_role: SemanticRole::Agent,
                        object_role: SemanticRole::ReturnsType,
                        is_causal: false,
                        is_negated: false,
                        confidence: 0.85,
                    });
                }
            }
        }

        // "fixed by" / "resolved by" → FixedBy role
        for pattern in &["fixed by ", "resolved by "] {
            if let Some(pos) = lower.find(pattern) {
                let before = lower[..pos].trim();
                let after = lower[pos + pattern.len()..].trim();
                let subject = before.split_whitespace().last().unwrap_or("").to_string();
                let object = after
                    .split(|c: char| c == '.' || c == ',' || c == ';')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !subject.is_empty() && !object.is_empty() {
                    relations.push(ExtractedRelation {
                        subject,
                        predicate: pattern.trim().to_string(),
                        object,
                        subject_role: SemanticRole::Agent,
                        object_role: SemanticRole::FixedBy,
                        is_causal: true,
                        is_negated: false,
                        confidence: 0.8,
                    });
                }
            }
        }

        relations
    }

    // ── Internal Methods ────────────────────────────────────────────────

    fn split_sentences<'a>(&self, text: &'a str) -> Vec<&'a str> {
        // Split on sentence boundaries: . ! ? followed by whitespace or EOL
        let mut sentences = Vec::new();
        let mut start = 0;

        for (i, c) in text.char_indices() {
            if (c == '.' || c == '!' || c == '?') && i + 1 < text.len() {
                let next = text[i + 1..].chars().next();
                if next == Some(' ') || next == Some('\n') || next == Some('\t') {
                    let sentence = text[start..=i].trim();
                    if !sentence.is_empty() {
                        sentences.push(sentence);
                    }
                    start = i + 1;
                }
            }
        }

        // Capture trailing text
        let remainder = text[start..].trim();
        if !remainder.is_empty() {
            sentences.push(remainder);
        }

        sentences
    }

    fn extract_entities(&self, sentence: &str) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();
        let lower = sentence.to_lowercase();

        // 1. Check entity cache for known entities
        for (known, entity_type) in &self.entity_cache {
            if let Some(pos) = lower.find(known.as_str()) {
                entities.push(ExtractedEntity {
                    text: sentence[pos..pos + known.len()].to_string(),
                    entity_type: *entity_type,
                    confidence: 0.9,
                    offset: pos,
                });
            }
        }

        // 2. Capitalized noun phrases (heuristic: sequences of capitalized words)
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let mut i = 0;
        let mut offset = 0;
        while i < words.len() {
            let word = words[i];
            // Track character offset
            if let Some(pos) = sentence[offset..].find(word) {
                offset += pos;
            }

            if word.len() > 1
                && word.chars().next().map_or(false, |c| c.is_uppercase())
                && !is_sentence_start(i, &words)
            {
                // Collect consecutive capitalized words
                let start_idx = i;
                let start_offset = offset;
                while i < words.len() && words[i].chars().next().map_or(false, |c| c.is_uppercase())
                {
                    i += 1;
                }
                let phrase: String = words[start_idx..i].join(" ");
                let entity_type = self.classify_entity(&phrase);
                entities.push(ExtractedEntity {
                    text: phrase,
                    entity_type,
                    confidence: 0.6,
                    offset: start_offset,
                });
                continue;
            }
            i += 1;
        }

        // 3. Temporal expressions
        for marker in &self.temporal_markers {
            if let Some(pos) = lower.find(marker) {
                // Extract the temporal phrase (marker + up to 4 following words)
                let after = &sentence[pos..];
                let temporal_phrase: String = after
                    .split_whitespace()
                    .take(4)
                    .collect::<Vec<_>>()
                    .join(" ");
                // Avoid duplicates
                if !entities.iter().any(|e| e.offset == pos) {
                    entities.push(ExtractedEntity {
                        text: temporal_phrase,
                        entity_type: EntityType::Temporal,
                        confidence: 0.5,
                        offset: pos,
                    });
                }
            }
        }

        entities
    }

    fn classify_entity(&self, phrase: &str) -> EntityType {
        let lower = phrase.to_lowercase();

        // Check cache first
        if let Some(etype) = self.entity_cache.get(&lower) {
            return *etype;
        }

        // Heuristic classification
        if lower.ends_with("university")
            || lower.ends_with("institute")
            || lower.ends_with("corporation")
            || lower.ends_with("inc")
            || lower.ends_with("ltd")
            || lower.ends_with("organization")
            || lower.ends_with("agency")
            || lower.ends_with("ministry")
            || lower.ends_with("government")
            || lower.ends_with("council")
            || lower.ends_with("commission")
            || lower.ends_with("committee")
            || lower.ends_with("nations")
        {
            EntityType::Organization
        } else if lower.contains("city")
            || lower.contains("country")
            || lower.contains("river")
            || lower.contains("mountain")
            || lower.contains("ocean")
            || lower.contains("sea")
            || lower.contains("strait")
            || lower.contains("gulf")
            || lower.contains("island")
            || lower.contains("peninsula")
            || lower.contains("region")
            || lower.contains("province")
            || lower.contains("state of")
        {
            EntityType::Place
        } else if lower.contains("war")
            || lower.contains("battle")
            || lower.contains("operation")
            || lower.contains("crisis")
            || lower.contains("summit")
            || lower.contains("election")
            || lower.contains("revolution")
        {
            EntityType::Event
        } else {
            // Clinical entity classification (therapeutic feature)
            #[cfg(feature = "therapeutic")]
            {
                if lower.contains("disorder")
                    || lower.contains("syndrome")
                    || lower.contains("diagnosis")
                    || lower.contains("dsm")
                    || lower.contains("icd")
                    || lower.contains("comorbid")
                    || lower.contains("pathology")
                    || lower.contains("etiology")
                    || lower.contains("prognosis")
                    || lower.contains("psychopathology")
                {
                    return EntityType::ClinicalConcept;
                } else if lower.contains("symptom")
                    || lower.contains("insomnia")
                    || lower.contains("anhedonia")
                    || lower.contains("rumination")
                    || lower.contains("dissociation")
                    || lower.contains("flashback")
                    || lower.contains("hallucination")
                    || lower.contains("delusion")
                    || lower.contains("ideation")
                    || lower.contains("dysphoria")
                    || lower.contains("hypervigilance")
                    || lower.contains("agoraphobia")
                {
                    return EntityType::Symptom;
                } else if lower.contains("therapy")
                    || lower.contains("intervention")
                    || lower.contains("treatment")
                    || lower.contains("medication")
                    || lower.contains("cbt")
                    || lower.contains("dbt")
                    || lower.contains("emdr")
                    || lower.contains("psychotherapy")
                    || lower.contains("counseling")
                    || lower.contains("rehabilitation")
                {
                    return EntityType::Intervention;
                }
            }
            // Default: if it looks like a proper noun, likely Person or Organization
            if phrase.split_whitespace().count() <= 3 {
                EntityType::Person // Best guess for short proper nouns
            } else {
                EntityType::Concept
            }
        }
    }

    fn extract_relations(
        &self,
        sentence: &str,
        entities: &[ExtractedEntity],
    ) -> Vec<ExtractedRelation> {
        let mut relations = Vec::new();
        let lower = sentence.to_lowercase();

        if entities.len() < 2 {
            return relations;
        }

        // Check for causal verbs
        for verb in &self.causal_verbs {
            if let Some(_verb_pos) = lower.find(verb) {
                // Find entities before and after the verb
                let before: Vec<_> = entities
                    .iter()
                    .filter(|e| {
                        lower.find(&e.text.to_lowercase()).unwrap_or(usize::MAX) < _verb_pos
                    })
                    .collect();
                let after: Vec<_> = entities
                    .iter()
                    .filter(|e| lower.find(&e.text.to_lowercase()).unwrap_or(0) > _verb_pos)
                    .collect();

                if let (Some(subj), Some(obj)) = (before.last(), after.first()) {
                    let is_negated = self.check_negation(&lower, _verb_pos);
                    relations.push(ExtractedRelation {
                        subject: subj.text.clone(),
                        predicate: verb.to_string(),
                        object: obj.text.clone(),
                        subject_role: SemanticRole::Agent,
                        object_role: if is_causal_verb(verb) {
                            SemanticRole::Result
                        } else {
                            SemanticRole::Patient
                        },
                        is_causal: is_causal_verb(verb),
                        is_negated,
                        confidence: if is_negated { 0.5 } else { 0.7 },
                    });
                }
            }
        }

        // Fallback: SVO pattern (first entity = subject, last = object)
        // Only if no relations found via verbs
        if relations.is_empty() && entities.len() >= 2 {
            // Extract the verb phrase between first and last entity
            let first = &entities[0];
            let last = &entities[entities.len() - 1];
            let first_end = first.offset + first.text.len();
            let last_start = last.offset;

            if first_end < last_start && last_start <= sentence.len() {
                let between = sentence[first_end..last_start].trim();
                if !between.is_empty() && between.len() < 100 {
                    let is_negated = self
                        .negation_markers
                        .iter()
                        .any(|n| between.to_lowercase().contains(n));
                    relations.push(ExtractedRelation {
                        subject: first.text.clone(),
                        predicate: between.to_string(),
                        object: last.text.clone(),
                        subject_role: SemanticRole::Agent,
                        object_role: SemanticRole::Patient,
                        is_causal: false,
                        is_negated,
                        confidence: 0.4,
                    });
                }
            }
        }

        relations
    }

    fn assign_roles(
        &self,
        sentence: &str,
        entities: &[ExtractedEntity],
    ) -> HashMap<String, SemanticRole> {
        let mut roles = HashMap::new();
        let lower = sentence.to_lowercase();

        for (i, entity) in entities.iter().enumerate() {
            let role = if i == 0 {
                SemanticRole::Agent
            } else if entity.entity_type == EntityType::Temporal {
                SemanticRole::Temporal
            } else if entity.entity_type == EntityType::Place {
                SemanticRole::Location
            } else if lower.contains("because") || lower.contains("due to") {
                // If there's a causal marker, later entities might be causes
                if i > entities.len() / 2 {
                    SemanticRole::Cause
                } else {
                    SemanticRole::Patient
                }
            } else {
                // Clinical role assignment (therapeutic feature)
                #[cfg(feature = "therapeutic")]
                {
                    if entity.entity_type == EntityType::Symptom
                        || entity.entity_type == EntityType::ClinicalConcept
                    {
                        if lower.contains("protective")
                            || lower.contains("resilience")
                            || lower.contains("strength")
                        {
                            SemanticRole::ProtectiveFactor
                        } else if lower.contains("risk")
                            || lower.contains("vulnerability")
                            || lower.contains("predispos")
                        {
                            SemanticRole::RiskFactor
                        } else {
                            SemanticRole::TherapeuticTarget
                        }
                    } else {
                        SemanticRole::Patient
                    }
                }
                #[cfg(not(feature = "therapeutic"))]
                {
                    SemanticRole::Patient
                }
            };
            roles.insert(entity.text.clone(), role);
        }

        roles
    }

    fn check_negation(&self, lower: &str, verb_pos: usize) -> bool {
        // Check for negation markers within 30 chars before the verb
        let window_start = verb_pos.saturating_sub(30);
        let window = &lower[window_start..verb_pos];
        self.negation_markers.iter().any(|n| window.contains(n))
    }

    fn estimate_confidence(
        &self,
        entities: &[ExtractedEntity],
        relations: &[ExtractedRelation],
    ) -> f32 {
        if entities.is_empty() {
            return 0.0;
        }

        let entity_conf: f32 =
            entities.iter().map(|e| e.confidence).sum::<f32>() / entities.len() as f32;
        let relation_conf: f32 = if relations.is_empty() {
            0.3 // Penalty for no relations
        } else {
            relations.iter().map(|r| r.confidence).sum::<f32>() / relations.len() as f32
        };

        (entity_conf * 0.4 + relation_conf * 0.6).clamp(0.0, 1.0)
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn is_sentence_start(word_idx: usize, words: &[&str]) -> bool {
    if word_idx == 0 {
        return true;
    }
    // After sentence-ending punctuation
    if let Some(prev) = words.get(word_idx - 1) {
        prev.ends_with('.') || prev.ends_with('!') || prev.ends_with('?')
    } else {
        false
    }
}

fn is_causal_verb(verb: &str) -> bool {
    matches!(
        verb,
        "cause"
            | "causes"
            | "caused"
            | "leads to"
            | "led to"
            | "results in"
            | "resulted in"
            | "trigger"
            | "triggers"
            | "triggered"
            | "produce"
            | "produces"
            | "produced"
            | "enable"
            | "enables"
            | "enabled"
            | "prevent"
            | "prevents"
            | "prevented"
            | "block"
            | "blocks"
            | "blocked"
            | "disrupt"
            | "disrupts"
            | "disrupted"
            | "destroy"
            | "destroys"
            | "destroyed"
            | "create"
            | "creates"
            | "created"
            | "force"
            | "forces"
            | "forced"
            | "compels"
            | "increase"
            | "increases"
            | "increased"
            | "decrease"
            | "decreases"
            | "decreased"
            | "accelerates"
            | "decelerates"
            | "amplifies"
            | "dampens"
            | "constrain"
            | "constrains"
            | "constrained"
            | "require"
            | "requires"
            | "required"
            | "implies"
            | "implied"
            | "entails"
            | "entailed"
    )
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_extraction() {
        let mut extractor = KnowledgeExtractor::new();
        extractor.register_entity("iran", EntityType::Place);
        extractor.register_entity("united states", EntityType::Organization);

        let facts = extractor.extract(
            "The United States launched airstrikes against Iran. \
             Iran retaliated with missile strikes.",
        );
        assert!(!facts.is_empty());
    }

    #[test]
    fn test_causal_extraction() {
        let mut extractor = KnowledgeExtractor::new();
        extractor.register_entity("sanctions", EntityType::Concept);
        extractor.register_entity("oil prices", EntityType::Concept);

        let facts = extractor.extract("Sanctions caused oil prices to increase dramatically.");
        assert!(!facts.is_empty());
        // Should find a causal relation
        let has_causal = facts
            .iter()
            .any(|f| f.relations.iter().any(|r| r.is_causal));
        assert!(has_causal);
    }

    #[test]
    fn test_negation_detection() {
        let mut extractor = KnowledgeExtractor::new();
        extractor.register_entity("diplomacy", EntityType::Concept);
        extractor.register_entity("conflict", EntityType::Concept);

        let facts = extractor.extract("Diplomacy did not prevent the conflict from escalating.");
        assert!(!facts.is_empty());
        let has_negated = facts
            .iter()
            .any(|f| f.relations.iter().any(|r| r.is_negated));
        assert!(has_negated);
    }

    #[test]
    fn test_temporal_extraction() {
        let mut extractor = KnowledgeExtractor::new();
        let facts = extractor.extract("Before the summit, tensions had been rising for months.");
        assert!(!facts.is_empty());
        let has_temporal = facts.iter().any(|f| {
            f.entities
                .iter()
                .any(|e| e.entity_type == EntityType::Temporal)
        });
        assert!(has_temporal);
    }

    #[test]
    fn test_empty_input() {
        let mut extractor = KnowledgeExtractor::new();
        let facts = extractor.extract("");
        assert!(facts.is_empty());
    }

    #[test]
    fn test_entity_cache_grows() {
        let mut extractor = KnowledgeExtractor::new();
        assert_eq!(extractor.total_extractions(), 0);
        extractor.register_entity("nato", EntityType::Organization);
        let _ = extractor.extract("NATO expanded its eastern flank.");
        assert!(extractor.total_extractions() > 0);
    }

    #[test]
    fn test_sentence_splitting() {
        let extractor = KnowledgeExtractor::new();
        let sentences =
            extractor.split_sentences("First sentence. Second sentence! Third? And more.");
        assert_eq!(sentences.len(), 4);
    }

    #[test]
    fn test_multiple_relations() {
        let mut extractor = KnowledgeExtractor::new();
        extractor.register_entity("blockade", EntityType::Event);
        extractor.register_entity("trade", EntityType::Concept);
        extractor.register_entity("inflation", EntityType::Concept);

        let facts = extractor.extract("The blockade disrupted trade, which led to inflation.");
        assert!(!facts.is_empty());
    }

    // ── Code-Specific Extraction Tests ──────────────────────────────────

    #[test]
    fn test_extract_code_entities_rust() {
        let extractor = KnowledgeExtractor::new();
        let code = r#"
            fn process_gradients(data: &[f32]) -> Vec<f32> { todo!() }
            struct CognitiveLoop { phi: f32 }
            enum SubstrateType { Silicon, Quantum }
            trait Conscious { fn phi(&self) -> f32; }
            mod extraction;
            use std::collections::HashMap;
        "#;

        let entities = extractor.extract_code_entities(code);

        // Check function detection
        let functions: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Function)
            .collect();
        assert!(
            functions.iter().any(|e| e.text == "process_gradients"),
            "should detect fn process_gradients"
        );
        assert!(
            functions.iter().any(|e| e.text == "phi"),
            "should detect fn phi inside trait"
        );

        // Check type detection
        let types: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Type)
            .collect();
        assert!(
            types.iter().any(|e| e.text == "CognitiveLoop"),
            "should detect struct CognitiveLoop"
        );
        assert!(
            types.iter().any(|e| e.text == "SubstrateType"),
            "should detect enum SubstrateType"
        );
        assert!(
            types.iter().any(|e| e.text == "Conscious"),
            "should detect trait Conscious"
        );

        // Check module detection
        let modules: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Module)
            .collect();
        assert!(
            modules.iter().any(|e| e.text == "extraction"),
            "should detect mod extraction"
        );
        assert!(
            modules.iter().any(|e| e.text == "std"),
            "should detect use std::"
        );

        // Confidence should be high (structural patterns)
        for entity in &entities {
            assert!(
                entity.confidence >= 0.8,
                "code entity confidence should be >= 0.8, got {} for '{}'",
                entity.confidence,
                entity.text
            );
        }
    }

    #[test]
    fn test_extract_code_entities_python() {
        let extractor = KnowledgeExtractor::new();
        let code = r#"
            def train_model(epochs: int) -> float:
                pass
            class NeuralNetwork:
                pass
            import torch
            from numpy import array
        "#;

        let entities = extractor.extract_code_entities(code);

        // Check function detection
        let functions: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Function)
            .collect();
        assert!(
            functions.iter().any(|e| e.text == "train_model"),
            "should detect def train_model"
        );

        // Check type detection
        let types: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Type)
            .collect();
        assert!(
            types.iter().any(|e| e.text == "NeuralNetwork"),
            "should detect class NeuralNetwork"
        );

        // Check module detection
        let modules: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Module)
            .collect();
        assert!(
            modules.iter().any(|e| e.text == "torch"),
            "should detect import torch"
        );
        assert!(
            modules.iter().any(|e| e.text == "numpy"),
            "should detect from numpy import"
        );
    }

    #[test]
    fn test_extract_code_relations() {
        let extractor = KnowledgeExtractor::new();

        // Calls pattern
        let rels = extractor.extract_code_relations("process_gradients calls aggregate_weights");
        assert!(
            rels.iter().any(|r| r.object_role == SemanticRole::Calls),
            "should detect 'calls' relation"
        );

        // Implements pattern (Rust syntax)
        let rels = extractor.extract_code_relations("impl Display for CognitiveLoop");
        assert!(
            rels.iter()
                .any(|r| r.object_role == SemanticRole::Implements),
            "should detect 'impl Trait for Type' relation"
        );
        let impl_rel = rels
            .iter()
            .find(|r| r.object_role == SemanticRole::Implements)
            .unwrap();
        assert_eq!(impl_rel.subject, "CognitiveLoop");
        assert_eq!(impl_rel.object, "Display");

        // DependsOn pattern
        let rels = extractor.extract_code_relations("extraction depends on std::collections");
        assert!(
            rels.iter()
                .any(|r| r.object_role == SemanticRole::DependsOn),
            "should detect 'depends on' relation"
        );

        // ReturnsType pattern
        let rels = extractor.extract_code_relations("compute_phi returns f32");
        assert!(
            rels.iter()
                .any(|r| r.object_role == SemanticRole::ReturnsType),
            "should detect 'returns' relation"
        );

        // FixedBy pattern
        let rels =
            extractor.extract_code_relations("E0308 fixed by adding explicit type annotation");
        assert!(
            rels.iter().any(|r| r.object_role == SemanticRole::FixedBy),
            "should detect 'fixed by' relation"
        );
        let fix_rel = rels
            .iter()
            .find(|r| r.object_role == SemanticRole::FixedBy)
            .unwrap();
        assert!(fix_rel.is_causal, "fix relations should be causal");
    }

    #[test]
    fn test_extract_error_codes() {
        let extractor = KnowledgeExtractor::new();
        let text = "Compiler error E0308: mismatched types. Also see E0277 and E0382.";

        let entities = extractor.extract_code_entities(text);
        let errors: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::ErrorPattern)
            .collect();

        assert_eq!(errors.len(), 3, "should detect three error codes");
        let codes: Vec<&str> = errors.iter().map(|e| e.text.as_str()).collect();
        assert!(codes.contains(&"E0308"), "should detect E0308");
        assert!(codes.contains(&"E0277"), "should detect E0277");
        assert!(codes.contains(&"E0382"), "should detect E0382");

        // Confidence should be high
        for error in &errors {
            assert!(
                error.confidence >= 0.9,
                "error code confidence should be >= 0.9"
            );
        }

        // Should NOT match partial patterns
        let text2 = "The variable EXTRA1234 is not an error code";
        let entities2 = extractor.extract_code_entities(text2);
        let false_errors: Vec<_> = entities2
            .iter()
            .filter(|e| e.entity_type == EntityType::ErrorPattern)
            .collect();
        assert!(
            false_errors.is_empty(),
            "should not match E within longer identifiers"
        );
    }
}
