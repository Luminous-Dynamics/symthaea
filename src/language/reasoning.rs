//! Reasoning Engine - Multi-step Thought Chains
//!
//! This module provides explicit reasoning capabilities that LLMs lack:
//! - Working memory for active concepts
//! - Inference rules (if-then patterns)
//! - Goal-directed reasoning chains
//! - Explainable reasoning traces
//!
//! Unlike LLMs which are black boxes, every reasoning step is transparent.

use crate::hdc::binary_hv::HV16;
use std::collections::{HashMap, VecDeque};

// ============================================================================
// Core Types
// ============================================================================

/// A concept in working memory
#[derive(Debug, Clone, PartialEq)]
pub struct Concept {
    /// Unique identifier
    pub id: ConceptId,
    /// Human-readable name
    pub name: String,
    /// HDC encoding for similarity
    pub encoding: HV16,
    /// Activation level (0.0 - 1.0)
    pub activation: f32,
    /// Semantic type
    pub concept_type: ConceptType,
    /// Properties/attributes
    pub properties: HashMap<String, String>,
}

/// Concept identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConceptId(pub u64);

impl ConceptId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Types of concepts
#[derive(Debug, Clone, PartialEq)]
pub enum ConceptType {
    Entity,      // Things: person, place, object
    Action,      // Verbs: run, think, feel
    Property,    // Adjectives: big, red, happy
    Relation,    // Between things: loves, contains, causes
    Abstract,    // Ideas: freedom, consciousness, love
    State,       // Conditions: alive, aware, sleeping
    Event,       // Occurrences: birth, meeting, storm
}

/// A relation between concepts
#[derive(Debug, Clone, PartialEq)]
pub struct Relation {
    pub from: ConceptId,
    pub relation_type: RelationType,
    pub to: ConceptId,
    pub strength: f32,  // Confidence 0.0 - 1.0
}

/// Types of relations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RelationType {
    IsA,         // Cat IS_A Animal
    HasPart,     // Car HAS_PART Wheel
    HasProperty, // Sky HAS_PROPERTY Blue
    Causes,      // Rain CAUSES Wet
    Precedes,    // Spring PRECEDES Summer
    Contains,    // Box CONTAINS Gift
    LocatedIn,   // Paris LOCATED_IN France
    Implies,     // Smoke IMPLIES Fire
    Opposes,     // Hot OPPOSES Cold
    Similar,     // Happy SIMILAR Joyful
    InstanceOf,  // Fido INSTANCE_OF Dog
    CapableOf,   // Bird CAPABLE_OF Fly
    UsedFor,     // Hammer USED_FOR Nail
    MotivatedBy, // Study MOTIVATED_BY Learn
    ResultsIn,   // Exercise RESULTS_IN Health
}

impl RelationType {
    /// Get inverse relation if applicable
    pub fn inverse(&self) -> Option<RelationType> {
        match self {
            RelationType::IsA => None,  // No simple inverse
            RelationType::HasPart => Some(RelationType::Contains),
            RelationType::Contains => Some(RelationType::HasPart),
            RelationType::Causes => Some(RelationType::ResultsIn),
            RelationType::ResultsIn => Some(RelationType::Causes),
            RelationType::Precedes => None,  // Temporal, no inverse
            RelationType::LocatedIn => Some(RelationType::Contains),
            RelationType::Implies => None,
            RelationType::Opposes => Some(RelationType::Opposes),  // Symmetric
            RelationType::Similar => Some(RelationType::Similar),  // Symmetric
            _ => None,
        }
    }
}

/// An inference rule
#[derive(Debug, Clone)]
pub struct InferenceRule {
    pub id: u64,
    pub name: String,
    /// Conditions that must be true
    pub conditions: Vec<Condition>,
    /// Conclusions to draw
    pub conclusions: Vec<Conclusion>,
    /// Confidence in this rule
    pub confidence: f32,
}

/// A condition in a rule
#[derive(Debug, Clone)]
pub enum Condition {
    /// Concept exists with property
    HasProperty { concept: ConceptPattern, property: String, value: Option<String> },
    /// Relation exists between concepts
    RelationExists { from: ConceptPattern, relation: RelationType, to: ConceptPattern },
    /// Concept is of type
    IsType { concept: ConceptPattern, concept_type: ConceptType },
    /// Activation above threshold
    ActiveAbove { concept: ConceptPattern, threshold: f32 },
    /// Negation
    Not(Box<Condition>),
}

/// Pattern for matching concepts
#[derive(Debug, Clone)]
pub enum ConceptPattern {
    /// Match specific concept by ID
    Specific(ConceptId),
    /// Match by name
    ByName(String),
    /// Match any concept of type
    AnyOfType(ConceptType),
    /// Variable to bind
    Variable(String),
}

/// A conclusion from a rule
#[derive(Debug, Clone)]
pub enum Conclusion {
    /// Assert a new relation
    AssertRelation { from: ConceptPattern, relation: RelationType, to: ConceptPattern },
    /// Set activation level
    SetActivation { concept: ConceptPattern, activation: f32 },
    /// Add property
    AddProperty { concept: ConceptPattern, property: String, value: String },
    /// Create derived concept
    CreateConcept { name: String, concept_type: ConceptType },
}

/// A single step in a reasoning chain
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    pub step_number: usize,
    pub rule_applied: Option<String>,
    pub premises: Vec<String>,
    pub conclusion: String,
    pub confidence: f32,
}

/// Result of reasoning
#[derive(Debug, Clone)]
pub struct ReasoningResult {
    pub success: bool,
    pub answer: Option<String>,
    pub trace: Vec<ReasoningStep>,
    pub final_confidence: f32,
    pub concepts_activated: Vec<String>,
    pub inferences_made: usize,
}

/// A goal to achieve through reasoning
#[derive(Debug, Clone)]
pub struct Goal {
    pub description: String,
    pub target_concept: Option<ConceptPattern>,
    pub target_relation: Option<(ConceptPattern, RelationType, ConceptPattern)>,
    pub max_depth: usize,
}

// ============================================================================
// Reasoning Engine
// ============================================================================

/// The main reasoning engine
pub struct ReasoningEngine {
    /// Working memory - currently active concepts
    working_memory: HashMap<ConceptId, Concept>,
    /// Known relations
    relations: Vec<Relation>,
    /// Inference rules
    rules: Vec<InferenceRule>,
    /// Goal stack
    goals: VecDeque<Goal>,
    /// Reasoning trace
    trace: Vec<ReasoningStep>,
    /// Next concept ID
    next_id: u64,
    /// Max reasoning depth
    max_depth: usize,
    /// Activation decay per step
    activation_decay: f32,
}

impl ReasoningEngine {
    /// Create a new reasoning engine
    pub fn new() -> Self {
        let mut engine = Self {
            working_memory: HashMap::new(),
            relations: Vec::new(),
            rules: Vec::new(),
            goals: VecDeque::new(),
            trace: Vec::new(),
            next_id: 1,
            max_depth: 10,
            activation_decay: 0.1,
        };
        engine.initialize_default_rules();
        engine
    }

    /// Initialize with common-sense inference rules
    fn initialize_default_rules(&mut self) {
        // Transitivity of IS_A
        self.add_rule(InferenceRule {
            id: 1,
            name: "IS_A Transitivity".to_string(),
            conditions: vec![
                Condition::RelationExists {
                    from: ConceptPattern::Variable("X".to_string()),
                    relation: RelationType::IsA,
                    to: ConceptPattern::Variable("Y".to_string()),
                },
                Condition::RelationExists {
                    from: ConceptPattern::Variable("Y".to_string()),
                    relation: RelationType::IsA,
                    to: ConceptPattern::Variable("Z".to_string()),
                },
            ],
            conclusions: vec![
                Conclusion::AssertRelation {
                    from: ConceptPattern::Variable("X".to_string()),
                    relation: RelationType::IsA,
                    to: ConceptPattern::Variable("Z".to_string()),
                },
            ],
            confidence: 0.95,
        });

        // Property inheritance
        self.add_rule(InferenceRule {
            id: 2,
            name: "Property Inheritance".to_string(),
            conditions: vec![
                Condition::RelationExists {
                    from: ConceptPattern::Variable("X".to_string()),
                    relation: RelationType::IsA,
                    to: ConceptPattern::Variable("Y".to_string()),
                },
                Condition::HasProperty {
                    concept: ConceptPattern::Variable("Y".to_string()),
                    property: "default_property".to_string(),
                    value: None,
                },
            ],
            conclusions: vec![
                Conclusion::AddProperty {
                    concept: ConceptPattern::Variable("X".to_string()),
                    property: "inherited".to_string(),
                    value: "true".to_string(),
                },
            ],
            confidence: 0.85,
        });

        // Causation chain
        self.add_rule(InferenceRule {
            id: 3,
            name: "Causal Chain".to_string(),
            conditions: vec![
                Condition::RelationExists {
                    from: ConceptPattern::Variable("A".to_string()),
                    relation: RelationType::Causes,
                    to: ConceptPattern::Variable("B".to_string()),
                },
                Condition::RelationExists {
                    from: ConceptPattern::Variable("B".to_string()),
                    relation: RelationType::Causes,
                    to: ConceptPattern::Variable("C".to_string()),
                },
            ],
            conclusions: vec![
                Conclusion::AssertRelation {
                    from: ConceptPattern::Variable("A".to_string()),
                    relation: RelationType::Causes,
                    to: ConceptPattern::Variable("C".to_string()),
                },
            ],
            confidence: 0.75,
        });

        // Implication to causation
        self.add_rule(InferenceRule {
            id: 4,
            name: "Implication Suggests Causation".to_string(),
            conditions: vec![
                Condition::RelationExists {
                    from: ConceptPattern::Variable("X".to_string()),
                    relation: RelationType::Implies,
                    to: ConceptPattern::Variable("Y".to_string()),
                },
            ],
            conclusions: vec![
                Conclusion::SetActivation {
                    concept: ConceptPattern::Variable("Y".to_string()),
                    activation: 0.7,
                },
            ],
            confidence: 0.80,
        });

        // Opposition spreads negative activation
        self.add_rule(InferenceRule {
            id: 5,
            name: "Opposition Inhibition".to_string(),
            conditions: vec![
                Condition::RelationExists {
                    from: ConceptPattern::Variable("X".to_string()),
                    relation: RelationType::Opposes,
                    to: ConceptPattern::Variable("Y".to_string()),
                },
                Condition::ActiveAbove {
                    concept: ConceptPattern::Variable("X".to_string()),
                    threshold: 0.5,
                },
            ],
            conclusions: vec![
                Conclusion::SetActivation {
                    concept: ConceptPattern::Variable("Y".to_string()),
                    activation: 0.2,
                },
            ],
            confidence: 0.70,
        });
    }

    /// Add a concept to working memory
    pub fn add_concept(&mut self, name: &str, concept_type: ConceptType) -> ConceptId {
        let id = ConceptId::new(self.next_id);
        self.next_id += 1;

        // Generate seed from name hash + id for reproducible encodings
        let seed = name.bytes().fold(self.next_id, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));

        let concept = Concept {
            id,
            name: name.to_string(),
            encoding: HV16::random(seed),
            activation: 1.0,
            concept_type,
            properties: HashMap::new(),
        };

        self.working_memory.insert(id, concept);
        id
    }

    /// Add a concept with specific encoding
    pub fn add_concept_with_encoding(&mut self, name: &str, concept_type: ConceptType, encoding: HV16) -> ConceptId {
        let id = ConceptId::new(self.next_id);
        self.next_id += 1;

        let concept = Concept {
            id,
            name: name.to_string(),
            encoding,
            activation: 1.0,
            concept_type,
            properties: HashMap::new(),
        };

        self.working_memory.insert(id, concept);
        id
    }

    /// Get a concept by ID
    pub fn get_concept(&self, id: ConceptId) -> Option<&Concept> {
        self.working_memory.get(&id)
    }

    /// Get a concept by name
    pub fn get_concept_by_name(&self, name: &str) -> Option<&Concept> {
        self.working_memory.values().find(|c| c.name.eq_ignore_ascii_case(name))
    }

    /// Add a relation between concepts
    pub fn add_relation(&mut self, from: ConceptId, relation_type: RelationType, to: ConceptId, strength: f32) {
        self.relations.push(Relation {
            from,
            relation_type,
            to,
            strength,
        });
    }

    /// Add an inference rule
    pub fn add_rule(&mut self, rule: InferenceRule) {
        self.rules.push(rule);
    }

    /// Set activation level of a concept
    pub fn activate(&mut self, id: ConceptId, level: f32) {
        if let Some(concept) = self.working_memory.get_mut(&id) {
            concept.activation = level.clamp(0.0, 1.0);
        }
    }

    /// Activate a concept by name
    pub fn activate_by_name(&mut self, name: &str, level: f32) {
        if let Some(concept) = self.working_memory.values_mut().find(|c| c.name.eq_ignore_ascii_case(name)) {
            concept.activation = level.clamp(0.0, 1.0);
        }
    }

    /// Decay all activations
    pub fn decay_activations(&mut self) {
        for concept in self.working_memory.values_mut() {
            concept.activation = (concept.activation - self.activation_decay).max(0.0);
        }
    }

    /// Get all active concepts (activation > threshold)
    pub fn active_concepts(&self, threshold: f32) -> Vec<&Concept> {
        self.working_memory.values()
            .filter(|c| c.activation > threshold)
            .collect()
    }

    /// Find relations from a concept
    pub fn relations_from(&self, id: ConceptId) -> Vec<&Relation> {
        self.relations.iter().filter(|r| r.from == id).collect()
    }

    /// Find relations to a concept
    pub fn relations_to(&self, id: ConceptId) -> Vec<&Relation> {
        self.relations.iter().filter(|r| r.to == id).collect()
    }

    /// Check if a relation exists
    pub fn has_relation(&self, from: ConceptId, relation_type: &RelationType, to: ConceptId) -> bool {
        self.relations.iter().any(|r| r.from == from && &r.relation_type == relation_type && r.to == to)
    }

    /// Reason about a query
    pub fn reason(&mut self, query: &str) -> ReasoningResult {
        self.trace.clear();

        // Parse query to find relevant concepts
        let query_lower = query.to_lowercase();
        let mut activated = Vec::new();

        // Activate concepts mentioned in query
        for concept in self.working_memory.values_mut() {
            if query_lower.contains(&concept.name.to_lowercase()) {
                concept.activation = 1.0;
                activated.push(concept.name.clone());
            }
        }

        // Run inference cycles
        let mut inferences_made = 0;
        for depth in 0..self.max_depth {
            let new_inferences = self.inference_cycle(depth);
            inferences_made += new_inferences;

            if new_inferences == 0 {
                break;  // Fixed point reached
            }
        }

        // Generate answer from trace
        let (success, answer) = self.generate_answer(query);

        let final_confidence = if self.trace.is_empty() {
            0.5
        } else {
            self.trace.iter().map(|s| s.confidence).sum::<f32>() / self.trace.len() as f32
        };

        ReasoningResult {
            success,
            answer,
            trace: self.trace.clone(),
            final_confidence,
            concepts_activated: activated,
            inferences_made,
        }
    }

    /// Run one cycle of inference
    fn inference_cycle(&mut self, _cycle: usize) -> usize {
        let mut new_inferences = 0;

        // Collect active concepts for rule matching
        let active: Vec<ConceptId> = self.working_memory.values()
            .filter(|c| c.activation > 0.3)
            .map(|c| c.id)
            .collect();

        // Try each rule
        for rule in &self.rules.clone() {
            if let Some(bindings) = self.try_match_rule(rule, &active) {
                // Apply conclusions
                for conclusion in &rule.conclusions {
                    if self.apply_conclusion(conclusion, &bindings) {
                        new_inferences += 1;

                        // Record reasoning step
                        self.trace.push(ReasoningStep {
                            step_number: self.trace.len() + 1,
                            rule_applied: Some(rule.name.clone()),
                            premises: self.format_premises(rule, &bindings),
                            conclusion: self.format_conclusion(conclusion, &bindings),
                            confidence: rule.confidence,
                        });
                    }
                }
            }
        }

        new_inferences
    }

    /// Try to match a rule against current state
    fn try_match_rule(&self, rule: &InferenceRule, active: &[ConceptId]) -> Option<HashMap<String, ConceptId>> {
        let mut bindings: HashMap<String, ConceptId> = HashMap::new();

        for condition in &rule.conditions {
            if !self.check_condition(condition, &mut bindings, active) {
                return None;
            }
        }

        Some(bindings)
    }

    /// Check if a condition is satisfied
    fn check_condition(&self, condition: &Condition, bindings: &mut HashMap<String, ConceptId>, active: &[ConceptId]) -> bool {
        match condition {
            Condition::RelationExists { from, relation, to } => {
                for rel in &self.relations {
                    if &rel.relation_type == relation {
                        if self.pattern_matches(from, rel.from, bindings) &&
                           self.pattern_matches(to, rel.to, bindings) {
                            return true;
                        }
                    }
                }
                false
            }
            Condition::HasProperty { concept, property, value } => {
                for id in active {
                    if self.pattern_matches(concept, *id, bindings) {
                        if let Some(c) = self.working_memory.get(id) {
                            if let Some(prop_val) = c.properties.get(property) {
                                if value.is_none() || value.as_ref() == Some(prop_val) {
                                    return true;
                                }
                            }
                        }
                    }
                }
                false
            }
            Condition::IsType { concept, concept_type } => {
                for id in active {
                    if self.pattern_matches(concept, *id, bindings) {
                        if let Some(c) = self.working_memory.get(id) {
                            if &c.concept_type == concept_type {
                                return true;
                            }
                        }
                    }
                }
                false
            }
            Condition::ActiveAbove { concept, threshold } => {
                for id in active {
                    if self.pattern_matches(concept, *id, bindings) {
                        if let Some(c) = self.working_memory.get(id) {
                            if c.activation > *threshold {
                                return true;
                            }
                        }
                    }
                }
                false
            }
            Condition::Not(inner) => {
                !self.check_condition(inner, bindings, active)
            }
        }
    }

    /// Check if a pattern matches a concept
    fn pattern_matches(&self, pattern: &ConceptPattern, id: ConceptId, bindings: &mut HashMap<String, ConceptId>) -> bool {
        match pattern {
            ConceptPattern::Specific(target_id) => *target_id == id,
            ConceptPattern::ByName(name) => {
                self.working_memory.get(&id)
                    .map(|c| c.name.eq_ignore_ascii_case(name))
                    .unwrap_or(false)
            }
            ConceptPattern::AnyOfType(concept_type) => {
                self.working_memory.get(&id)
                    .map(|c| &c.concept_type == concept_type)
                    .unwrap_or(false)
            }
            ConceptPattern::Variable(var) => {
                if let Some(&bound_id) = bindings.get(var) {
                    bound_id == id
                } else {
                    bindings.insert(var.clone(), id);
                    true
                }
            }
        }
    }

    /// Apply a conclusion
    fn apply_conclusion(&mut self, conclusion: &Conclusion, bindings: &HashMap<String, ConceptId>) -> bool {
        match conclusion {
            Conclusion::AssertRelation { from, relation, to } => {
                let from_id = self.resolve_pattern(from, bindings);
                let to_id = self.resolve_pattern(to, bindings);

                if let (Some(f), Some(t)) = (from_id, to_id) {
                    if !self.has_relation(f, relation, t) {
                        self.add_relation(f, relation.clone(), t, 0.8);
                        return true;
                    }
                }
                false
            }
            Conclusion::SetActivation { concept, activation } => {
                if let Some(id) = self.resolve_pattern(concept, bindings) {
                    self.activate(id, *activation);
                    return true;
                }
                false
            }
            Conclusion::AddProperty { concept, property, value } => {
                if let Some(id) = self.resolve_pattern(concept, bindings) {
                    if let Some(c) = self.working_memory.get_mut(&id) {
                        c.properties.insert(property.clone(), value.clone());
                        return true;
                    }
                }
                false
            }
            Conclusion::CreateConcept { name, concept_type } => {
                if self.get_concept_by_name(name).is_none() {
                    self.add_concept(name, concept_type.clone());
                    return true;
                }
                false
            }
        }
    }

    /// Resolve a pattern to a concept ID
    fn resolve_pattern(&self, pattern: &ConceptPattern, bindings: &HashMap<String, ConceptId>) -> Option<ConceptId> {
        match pattern {
            ConceptPattern::Specific(id) => Some(*id),
            ConceptPattern::ByName(name) => self.get_concept_by_name(name).map(|c| c.id),
            ConceptPattern::Variable(var) => bindings.get(var).copied(),
            ConceptPattern::AnyOfType(_) => None,  // Can't resolve to single ID
        }
    }

    /// Format premises for trace
    fn format_premises(&self, rule: &InferenceRule, bindings: &HashMap<String, ConceptId>) -> Vec<String> {
        let mut premises = Vec::new();
        for condition in &rule.conditions {
            premises.push(self.format_condition(condition, bindings));
        }
        premises
    }

    /// Format a condition as string
    fn format_condition(&self, condition: &Condition, bindings: &HashMap<String, ConceptId>) -> String {
        match condition {
            Condition::RelationExists { from, relation, to } => {
                let from_name = self.pattern_name(from, bindings);
                let to_name = self.pattern_name(to, bindings);
                format!("{} {:?} {}", from_name, relation, to_name)
            }
            Condition::HasProperty { concept, property, value } => {
                let name = self.pattern_name(concept, bindings);
                if let Some(v) = value {
                    format!("{} has {}={}", name, property, v)
                } else {
                    format!("{} has {}", name, property)
                }
            }
            Condition::IsType { concept, concept_type } => {
                let name = self.pattern_name(concept, bindings);
                format!("{} is {:?}", name, concept_type)
            }
            Condition::ActiveAbove { concept, threshold } => {
                let name = self.pattern_name(concept, bindings);
                format!("{} activation > {}", name, threshold)
            }
            Condition::Not(inner) => {
                format!("NOT ({})", self.format_condition(inner, bindings))
            }
        }
    }

    /// Format a conclusion as string
    fn format_conclusion(&self, conclusion: &Conclusion, bindings: &HashMap<String, ConceptId>) -> String {
        match conclusion {
            Conclusion::AssertRelation { from, relation, to } => {
                let from_name = self.pattern_name(from, bindings);
                let to_name = self.pattern_name(to, bindings);
                format!("Therefore: {} {:?} {}", from_name, relation, to_name)
            }
            Conclusion::SetActivation { concept, activation } => {
                let name = self.pattern_name(concept, bindings);
                format!("Activate {} to {}", name, activation)
            }
            Conclusion::AddProperty { concept, property, value } => {
                let name = self.pattern_name(concept, bindings);
                format!("Add {}.{} = {}", name, property, value)
            }
            Conclusion::CreateConcept { name, concept_type } => {
                format!("Create {:?} '{}'", concept_type, name)
            }
        }
    }

    /// Get name for a pattern
    fn pattern_name(&self, pattern: &ConceptPattern, bindings: &HashMap<String, ConceptId>) -> String {
        match pattern {
            ConceptPattern::Specific(id) => {
                self.working_memory.get(id).map(|c| c.name.clone()).unwrap_or_else(|| format!("#{}", id.0))
            }
            ConceptPattern::ByName(name) => name.clone(),
            ConceptPattern::Variable(var) => {
                if let Some(id) = bindings.get(var) {
                    self.working_memory.get(id).map(|c| c.name.clone()).unwrap_or_else(|| var.clone())
                } else {
                    var.clone()
                }
            }
            ConceptPattern::AnyOfType(t) => format!("any {:?}", t),
        }
    }

    /// Generate answer from reasoning trace
    fn generate_answer(&self, query: &str) -> (bool, Option<String>) {
        if self.trace.is_empty() {
            return (false, Some("I don't have enough information to reason about this.".to_string()));
        }

        // Build explanation from trace
        let mut explanation = String::new();

        for step in &self.trace {
            explanation.push_str(&format!("Step {}: ", step.step_number));
            if let Some(ref rule) = step.rule_applied {
                explanation.push_str(&format!("Using '{}': ", rule));
            }
            explanation.push_str(&step.conclusion);
            explanation.push_str(". ");
        }

        let final_step = self.trace.last().unwrap();
        let success = final_step.confidence > 0.5;

        (success, Some(explanation))
    }

    /// Explain reasoning about a specific relation
    pub fn explain_relation(&self, from_name: &str, to_name: &str) -> Option<String> {
        let from = self.get_concept_by_name(from_name)?;
        let to = self.get_concept_by_name(to_name)?;

        let direct_relations: Vec<_> = self.relations.iter()
            .filter(|r| r.from == from.id && r.to == to.id)
            .collect();

        if direct_relations.is_empty() {
            return Some(format!("I don't know of any direct relationship between {} and {}.", from_name, to_name));
        }

        let mut explanation = format!("{} and {} are related: ", from_name, to_name);
        for rel in direct_relations {
            explanation.push_str(&format!("{:?} (confidence: {:.0}%), ", rel.relation_type, rel.strength * 100.0));
        }

        Some(explanation)
    }

    /// Get reasoning trace as human-readable string
    pub fn trace_string(&self) -> String {
        if self.trace.is_empty() {
            return "No reasoning performed yet.".to_string();
        }

        let mut result = String::new();
        result.push_str("=== Reasoning Trace ===\n");

        for step in &self.trace {
            result.push_str(&format!("\nStep {}:\n", step.step_number));
            if let Some(ref rule) = step.rule_applied {
                result.push_str(&format!("  Rule: {}\n", rule));
            }
            result.push_str("  Premises:\n");
            for premise in &step.premises {
                result.push_str(&format!("    - {}\n", premise));
            }
            result.push_str(&format!("  Conclusion: {}\n", step.conclusion));
            result.push_str(&format!("  Confidence: {:.0}%\n", step.confidence * 100.0));
        }

        result
    }

    /// Clear working memory but keep rules
    pub fn clear_memory(&mut self) {
        self.working_memory.clear();
        self.relations.clear();
        self.trace.clear();
        self.next_id = 1;
    }

    /// Get stats
    pub fn stats(&self) -> ReasoningStats {
        ReasoningStats {
            concepts: self.working_memory.len(),
            relations: self.relations.len(),
            rules: self.rules.len(),
            trace_steps: self.trace.len(),
        }
    }
}

impl Default for ReasoningEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about reasoning state
#[derive(Debug, Clone)]
pub struct ReasoningStats {
    pub concepts: usize,
    pub relations: usize,
    pub rules: usize,
    pub trace_steps: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = ReasoningEngine::new();
        assert!(engine.rules.len() >= 5);  // Default rules
        assert_eq!(engine.working_memory.len(), 0);
    }

    #[test]
    fn test_add_concept() {
        let mut engine = ReasoningEngine::new();
        let id = engine.add_concept("cat", ConceptType::Entity);
        assert!(engine.get_concept(id).is_some());
        assert_eq!(engine.get_concept(id).unwrap().name, "cat");
    }

    #[test]
    fn test_add_relation() {
        let mut engine = ReasoningEngine::new();
        let cat = engine.add_concept("cat", ConceptType::Entity);
        let animal = engine.add_concept("animal", ConceptType::Entity);
        engine.add_relation(cat, RelationType::IsA, animal, 1.0);
        assert!(engine.has_relation(cat, &RelationType::IsA, animal));
    }

    #[test]
    fn test_is_a_transitivity() {
        let mut engine = ReasoningEngine::new();

        // Cat IS_A Mammal IS_A Animal
        let cat = engine.add_concept("cat", ConceptType::Entity);
        let mammal = engine.add_concept("mammal", ConceptType::Entity);
        let animal = engine.add_concept("animal", ConceptType::Entity);

        engine.add_relation(cat, RelationType::IsA, mammal, 1.0);
        engine.add_relation(mammal, RelationType::IsA, animal, 1.0);

        // Reason about cats
        let result = engine.reason("Is a cat an animal?");

        // Should infer cat IS_A animal
        assert!(result.inferences_made > 0);
        assert!(engine.has_relation(cat, &RelationType::IsA, animal));
    }

    #[test]
    fn test_causal_chain() {
        let mut engine = ReasoningEngine::new();

        // Rain CAUSES Wet, Wet CAUSES Slippery
        let rain = engine.add_concept("rain", ConceptType::Event);
        let wet = engine.add_concept("wet", ConceptType::State);
        let slippery = engine.add_concept("slippery", ConceptType::State);

        engine.add_relation(rain, RelationType::Causes, wet, 0.9);
        engine.add_relation(wet, RelationType::Causes, slippery, 0.8);

        let result = engine.reason("Does rain cause slippery conditions?");

        // Should infer rain CAUSES slippery
        assert!(result.inferences_made > 0);
        assert!(engine.has_relation(rain, &RelationType::Causes, slippery));
    }

    #[test]
    fn test_activation_decay() {
        let mut engine = ReasoningEngine::new();
        let id = engine.add_concept("test", ConceptType::Abstract);

        engine.activate(id, 1.0);
        assert_eq!(engine.get_concept(id).unwrap().activation, 1.0);

        engine.decay_activations();
        assert!(engine.get_concept(id).unwrap().activation < 1.0);
    }

    #[test]
    fn test_active_concepts() {
        let mut engine = ReasoningEngine::new();
        let a = engine.add_concept("active", ConceptType::Abstract);
        let b = engine.add_concept("inactive", ConceptType::Abstract);

        engine.activate(a, 0.8);
        engine.activate(b, 0.2);

        let active = engine.active_concepts(0.5);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].name, "active");
    }

    #[test]
    fn test_explain_relation() {
        let mut engine = ReasoningEngine::new();
        let paris = engine.add_concept("paris", ConceptType::Entity);
        let france = engine.add_concept("france", ConceptType::Entity);
        engine.add_relation(paris, RelationType::LocatedIn, france, 1.0);

        let explanation = engine.explain_relation("paris", "france");
        assert!(explanation.is_some());
        assert!(explanation.unwrap().contains("LocatedIn"));
    }

    #[test]
    fn test_reasoning_trace() {
        let mut engine = ReasoningEngine::new();
        let a = engine.add_concept("a", ConceptType::Entity);
        let b = engine.add_concept("b", ConceptType::Entity);
        let c = engine.add_concept("c", ConceptType::Entity);

        engine.add_relation(a, RelationType::IsA, b, 1.0);
        engine.add_relation(b, RelationType::IsA, c, 1.0);

        let result = engine.reason("What is a?");
        let trace = engine.trace_string();

        assert!(trace.contains("Reasoning Trace"));
    }

    #[test]
    fn test_stats() {
        let mut engine = ReasoningEngine::new();
        engine.add_concept("test", ConceptType::Entity);
        engine.add_concept("other", ConceptType::Entity);

        let stats = engine.stats();
        assert_eq!(stats.concepts, 2);
        assert!(stats.rules >= 5);
    }

    #[test]
    fn test_concept_by_name() {
        let mut engine = ReasoningEngine::new();
        engine.add_concept("Consciousness", ConceptType::Abstract);

        assert!(engine.get_concept_by_name("consciousness").is_some());
        assert!(engine.get_concept_by_name("CONSCIOUSNESS").is_some());
        assert!(engine.get_concept_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_relation_types() {
        assert_eq!(RelationType::Opposes.inverse(), Some(RelationType::Opposes));
        assert_eq!(RelationType::Similar.inverse(), Some(RelationType::Similar));
        assert!(RelationType::IsA.inverse().is_none());
    }
}
