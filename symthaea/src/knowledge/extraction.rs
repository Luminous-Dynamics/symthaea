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
}
