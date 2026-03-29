// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LLM-Assisted Knowledge Extraction
//!
//! Uses the LLMOrgan to extract structured facts from complex text
//! that the pattern-based extractor can't handle (coreference,
//! multi-clause sentences, implicit relations).
//!
//! The pattern extractor remains the fast path (~0.01ms per sentence);
//! LLM extraction is the slow path (~50-500ms) used for:
//! - Sentences where pattern extraction finds 0 relations
//! - Explicit "deep extract" requests from the reasoning engine
//! - Novel domains with no registered entities
//!
//! Science: Etzioni et al. (2011) Open IE, Wei et al. (2022) chain-of-thought

use super::extraction::{
    EntityType, ExtractedEntity, ExtractedFact, ExtractedRelation, SemanticRole,
};
use std::collections::HashMap;

/// Prompt template for structured extraction
const EXTRACTION_PROMPT: &str = r#"Extract structured facts from this text. For each fact, list:
- ENTITIES: name (type: Person/Organization/Place/Event/Concept/Quantity)
- RELATIONS: subject -> predicate -> object (causal: yes/no, negated: yes/no)
- ROLES: entity=role (Agent/Patient/Instrument/Context/Goal/Source/Destination/Temporal/Location/Cause/Result)

Text: {INPUT}

Respond in this exact format:
FACT:
  ENTITY: name | type | confidence
  ENTITY: name | type | confidence
  RELATION: subject | predicate | object | causal | negated | confidence
  ROLE: name = role
"#;

/// Result of an LLM extraction attempt
#[derive(Debug)]
pub enum LlmExtractionResult {
    /// Successfully extracted facts via LLM
    Success(Vec<ExtractedFact>),
    /// LLM unavailable; use pattern extractor fallback
    Unavailable,
    /// LLM returned unparseable response
    ParseError(String),
}

/// Parse an LLM response into structured facts.
///
/// This is a best-effort parser that handles the structured format
/// defined in EXTRACTION_PROMPT. Tolerant of whitespace and case variations.
pub fn parse_llm_response(response: &str) -> LlmExtractionResult {
    let mut facts = Vec::new();
    let mut current_entities = Vec::new();
    let mut current_relations = Vec::new();
    let mut current_roles: HashMap<String, SemanticRole> = HashMap::new();
    let mut in_fact = false;

    for line in response.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("FACT:") || trimmed == "FACT" {
            // Save previous fact if any
            if in_fact && (!current_entities.is_empty() || !current_relations.is_empty()) {
                facts.push(ExtractedFact {
                    entities: std::mem::take(&mut current_entities),
                    relations: std::mem::take(&mut current_relations),
                    role_map: std::mem::take(&mut current_roles),
                    source_text: String::new(), // Filled by caller
                    confidence: 0.7,            // LLM extraction base confidence
                });
            }
            in_fact = true;
            current_entities.clear();
            current_relations.clear();
            current_roles.clear();
            continue;
        }

        if !in_fact {
            continue;
        }

        if trimmed.starts_with("ENTITY:") {
            if let Some(entity) = parse_entity_line(trimmed) {
                current_entities.push(entity);
            }
        } else if trimmed.starts_with("RELATION:") {
            if let Some(relation) = parse_relation_line(trimmed) {
                current_relations.push(relation);
            }
        } else if trimmed.starts_with("ROLE:") {
            if let Some((name, role)) = parse_role_line(trimmed) {
                current_roles.insert(name, role);
            }
        }
    }

    // Save last fact
    if in_fact && (!current_entities.is_empty() || !current_relations.is_empty()) {
        facts.push(ExtractedFact {
            entities: current_entities,
            relations: current_relations,
            role_map: current_roles,
            source_text: String::new(),
            confidence: 0.7,
        });
    }

    if facts.is_empty() {
        LlmExtractionResult::ParseError("No facts parsed from LLM response".into())
    } else {
        LlmExtractionResult::Success(facts)
    }
}

/// Format the extraction prompt with the given input text
pub fn format_extraction_prompt(input: &str) -> String {
    EXTRACTION_PROMPT.replace("{INPUT}", input)
}

// ── Parsers ──────────────────────────────────────────────────────────────

fn parse_entity_line(line: &str) -> Option<ExtractedEntity> {
    let content = line.strip_prefix("ENTITY:")?.trim();
    let parts: Vec<&str> = content.split('|').map(str::trim).collect();
    if parts.len() < 2 {
        return None;
    }

    let name = parts[0].to_string();
    let entity_type = parse_entity_type(parts[1]);
    let confidence = parts
        .get(2)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.7);

    Some(ExtractedEntity {
        text: name,
        entity_type,
        confidence,
        offset: 0, // LLM doesn't provide character offsets
    })
}

fn parse_relation_line(line: &str) -> Option<ExtractedRelation> {
    let content = line.strip_prefix("RELATION:")?.trim();
    let parts: Vec<&str> = content.split('|').map(str::trim).collect();
    if parts.len() < 3 {
        return None;
    }

    let subject = parts[0].to_string();
    let predicate = parts[1].to_string();
    let object = parts[2].to_string();
    let is_causal = parts
        .get(3)
        .map_or(false, |s| s.trim().eq_ignore_ascii_case("yes"));
    let is_negated = parts
        .get(4)
        .map_or(false, |s| s.trim().eq_ignore_ascii_case("yes"));
    let confidence = parts
        .get(5)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.6);

    Some(ExtractedRelation {
        subject,
        predicate,
        object,
        subject_role: SemanticRole::Agent,
        object_role: if is_causal {
            SemanticRole::Result
        } else {
            SemanticRole::Patient
        },
        is_causal,
        is_negated,
        confidence,
    })
}

fn parse_role_line(line: &str) -> Option<(String, SemanticRole)> {
    let content = line.strip_prefix("ROLE:")?.trim();
    let parts: Vec<&str> = content.split('=').map(str::trim).collect();
    if parts.len() != 2 {
        return None;
    }

    let name = parts[0].to_string();
    let role = parse_semantic_role(parts[1])?;
    Some((name, role))
}

fn parse_entity_type(s: &str) -> EntityType {
    match s.trim().to_lowercase().as_str() {
        "person" => EntityType::Person,
        "organization" | "org" => EntityType::Organization,
        "place" | "location" => EntityType::Place,
        "event" => EntityType::Event,
        "concept" => EntityType::Concept,
        "quantity" | "number" => EntityType::Quantity,
        "temporal" | "time" | "date" => EntityType::Temporal,
        "artifact" | "object" | "thing" => EntityType::Artifact,
        "process" => EntityType::Process,
        "property" => EntityType::Property,
        _ => EntityType::Concept, // Default fallback
    }
}

fn parse_semantic_role(s: &str) -> Option<SemanticRole> {
    match s.trim().to_lowercase().as_str() {
        "agent" => Some(SemanticRole::Agent),
        "patient" => Some(SemanticRole::Patient),
        "instrument" => Some(SemanticRole::Instrument),
        "context" => Some(SemanticRole::Context),
        "goal" => Some(SemanticRole::Goal),
        "source" => Some(SemanticRole::Source),
        "destination" => Some(SemanticRole::Destination),
        "temporal" | "time" => Some(SemanticRole::Temporal),
        "location" | "place" => Some(SemanticRole::Location),
        "cause" => Some(SemanticRole::Cause),
        "result" | "effect" => Some(SemanticRole::Result),
        _ => None,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_complete_response() {
        let response = r#"FACT:
  ENTITY: United States | Organization | 0.95
  ENTITY: Iran | Organization | 0.90
  RELATION: United States | sanctioned | Iran | yes | no | 0.85
  ROLE: United States = Agent
  ROLE: Iran = Patient
"#;
        match parse_llm_response(response) {
            LlmExtractionResult::Success(facts) => {
                assert_eq!(facts.len(), 1);
                assert_eq!(facts[0].entities.len(), 2);
                assert_eq!(facts[0].relations.len(), 1);
                assert!(facts[0].relations[0].is_causal);
                assert!(!facts[0].relations[0].is_negated);
                assert_eq!(facts[0].role_map.len(), 2);
            }
            other => panic!("Expected Success, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_multiple_facts() {
        let response = r#"FACT:
  ENTITY: sanctions | Concept | 0.9
  ENTITY: oil prices | Concept | 0.8
  RELATION: sanctions | caused | oil prices | yes | no | 0.7
FACT:
  ENTITY: inflation | Concept | 0.8
  ENTITY: recession | Concept | 0.7
  RELATION: inflation | leads to | recession | yes | no | 0.6
"#;
        match parse_llm_response(response) {
            LlmExtractionResult::Success(facts) => {
                assert_eq!(facts.len(), 2);
            }
            other => panic!("Expected Success, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_negated_relation() {
        let response = r#"FACT:
  ENTITY: diplomacy | Concept | 0.8
  ENTITY: war | Event | 0.9
  RELATION: diplomacy | prevented | war | yes | yes | 0.7
"#;
        match parse_llm_response(response) {
            LlmExtractionResult::Success(facts) => {
                assert!(facts[0].relations[0].is_negated);
                assert!(facts[0].relations[0].is_causal);
            }
            other => panic!("Expected Success, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_empty_response() {
        let response = "I don't know how to extract facts from this.";
        match parse_llm_response(response) {
            LlmExtractionResult::ParseError(_) => {} // Expected
            other => panic!("Expected ParseError, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_entity_types() {
        assert_eq!(parse_entity_type("Person"), EntityType::Person);
        assert_eq!(parse_entity_type("org"), EntityType::Organization);
        assert_eq!(parse_entity_type("PLACE"), EntityType::Place);
        assert_eq!(parse_entity_type("unknown"), EntityType::Concept); // Fallback
    }

    #[test]
    fn test_parse_semantic_roles() {
        assert_eq!(parse_semantic_role("Agent"), Some(SemanticRole::Agent));
        assert_eq!(parse_semantic_role("effect"), Some(SemanticRole::Result));
        assert_eq!(parse_semantic_role("time"), Some(SemanticRole::Temporal));
        assert_eq!(parse_semantic_role("nonsense"), None);
    }

    #[test]
    fn test_format_prompt() {
        let prompt = format_extraction_prompt("The US sanctioned Iran.");
        assert!(prompt.contains("The US sanctioned Iran."));
        assert!(prompt.contains("Extract structured facts"));
    }

    #[test]
    fn test_partial_entity_line() {
        // Only name and type, no confidence
        let entity = parse_entity_line("ENTITY: Iran | Organization");
        assert!(entity.is_some());
        let e = entity.unwrap();
        assert_eq!(e.text, "Iran");
        assert_eq!(e.entity_type, EntityType::Organization);
        assert_eq!(e.confidence, 0.7); // Default
    }
}
