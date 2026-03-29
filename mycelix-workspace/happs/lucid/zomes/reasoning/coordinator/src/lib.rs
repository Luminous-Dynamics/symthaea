// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LUCID Reasoning Coordinator Zome

use hdk::prelude::*;
use reasoning_integrity::*;

fn anchor_hash(s: &str) -> ExternResult<EntryHash> {
    hash_entry(&EntryTypes::Anchor(Anchor(s.to_string())))
}

fn create_anchor(s: &str) -> ExternResult<EntryHash> {
    create_entry(&EntryTypes::Anchor(Anchor(s.to_string())))?;
    anchor_hash(s)
}

fn generate_uuid() -> String {
    let mut bytes = [0u8; 16];
    getrandom_03::fill(&mut bytes).expect("Failed to generate random bytes");
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

/// Record a detected contradiction
#[hdk_extern]
pub fn record_contradiction(input: RecordContradictionInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let contradiction = Contradiction {
        id: generate_uuid(),
        thought_a_id: input.thought_a_id.clone(),
        thought_b_id: input.thought_b_id.clone(),
        contradiction_type: input.contradiction_type,
        severity: input.severity,
        explanation: input.explanation,
        suggested_resolution: input.suggested_resolution,
        resolved: false,
        resolution_notes: None,
        detected_at: now,
        resolved_at: None,
        phi_at_detection: None,
        semantic_similarity: None,
        coherence_at_detection: None,
    };

    let action_hash = create_entry(&EntryTypes::Contradiction(contradiction))?;

    // Link from both thoughts
    for thought_id in [&input.thought_a_id, &input.thought_b_id] {
        let thought_anchor = format!("thought:{}", thought_id);
        create_anchor(&thought_anchor)?;
        create_link(
            anchor_hash(&thought_anchor)?,
            action_hash.clone(),
            LinkTypes::ThoughtToContradictions,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Contradiction not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordContradictionInput {
    pub thought_a_id: String,
    pub thought_b_id: String,
    pub contradiction_type: ContradictionType,
    pub severity: f64,
    pub explanation: String,
    pub suggested_resolution: Option<String>,
}

/// Record a contradiction with semantic analysis from Symthaea
///
/// This is the enhanced version that includes phi, coherence, and semantic
/// similarity data computed by Symthaea in the Tauri layer.
#[hdk_extern]
pub fn record_contradiction_analyzed(input: RecordContradictionAnalyzedInput) -> ExternResult<Record> {
    let now = sys_time()?;

    // Classify contradiction type based on semantic analysis
    let contradiction_type = classify_contradiction_type(&input);

    // Calculate severity based on phi, coherence, and similarity
    let severity = calculate_semantic_severity(&input);

    let suggested_resolution = generate_resolution_suggestion(&input, &contradiction_type);

    let contradiction = Contradiction {
        id: generate_uuid(),
        thought_a_id: input.thought_a_id.clone(),
        thought_b_id: input.thought_b_id.clone(),
        contradiction_type,
        severity,
        explanation: input.explanation,
        suggested_resolution: Some(suggested_resolution),
        resolved: false,
        resolution_notes: None,
        detected_at: now,
        resolved_at: None,
        phi_at_detection: Some(input.phi),
        semantic_similarity: Some(input.semantic_similarity),
        coherence_at_detection: Some(input.coherence),
    };

    let action_hash = create_entry(&EntryTypes::Contradiction(contradiction))?;

    // Link from both thoughts
    for thought_id in [&input.thought_a_id, &input.thought_b_id] {
        let thought_anchor = format!("thought:{}", thought_id);
        create_anchor(&thought_anchor)?;
        create_link(
            anchor_hash(&thought_anchor)?,
            action_hash.clone(),
            LinkTypes::ThoughtToContradictions,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Contradiction not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordContradictionAnalyzedInput {
    pub thought_a_id: String,
    pub thought_b_id: String,
    /// Content of thought A (for analysis)
    pub thought_a_content: String,
    /// Content of thought B (for analysis)
    pub thought_b_content: String,
    /// Description of the contradiction
    pub explanation: String,
    /// Phi (integrated information) from Symthaea coherence check
    pub phi: f64,
    /// Overall coherence from Symthaea
    pub coherence: f64,
    /// Semantic similarity between the two thoughts (from HDC embeddings)
    pub semantic_similarity: f64,
    /// Logical coherence score
    pub logical_coherence: f64,
}

/// Classify contradiction type using semantic analysis
fn classify_contradiction_type(input: &RecordContradictionAnalyzedInput) -> ContradictionType {
    // Fundamental: very low phi indicates deep structural conflict
    if input.phi < 0.3 {
        return ContradictionType::Fundamental;
    }

    // Low logical coherence with moderate phi = logical contradiction
    if input.logical_coherence < 0.5 && input.phi >= 0.3 {
        return ContradictionType::Logical;
    }

    // High semantic similarity but low coherence = semantic divergence
    // (They're talking about the same thing but reaching opposite conclusions)
    if input.semantic_similarity > 0.6 && input.coherence < 0.4 {
        return ContradictionType::SemanticDivergence;
    }

    // Moderate everything = potentially apparent/surface-level contradiction
    if input.phi >= 0.5 && input.coherence >= 0.4 {
        return ContradictionType::Apparent;
    }

    // Default to factual inconsistency
    ContradictionType::Factual
}

/// Calculate severity based on semantic metrics
fn calculate_semantic_severity(input: &RecordContradictionAnalyzedInput) -> f64 {
    // Severity increases with:
    // - Lower phi (more fundamental conflict)
    // - Lower coherence
    // - Higher semantic similarity (they're about the same thing)
    // - Lower logical coherence

    let phi_factor = 1.0 - input.phi; // Inverted: low phi = high severity
    let coherence_factor = 1.0 - input.coherence;
    let similarity_factor = input.semantic_similarity; // High similarity = high severity
    let logical_factor = 1.0 - input.logical_coherence;

    // Weighted combination
    let raw_severity = phi_factor * 0.3
        + coherence_factor * 0.25
        + similarity_factor * 0.25
        + logical_factor * 0.2;

    raw_severity.clamp(0.0, 1.0)
}

/// Generate a resolution suggestion based on contradiction type
fn generate_resolution_suggestion(
    _input: &RecordContradictionAnalyzedInput,
    contradiction_type: &ContradictionType,
) -> String {
    match contradiction_type {
        ContradictionType::Fundamental => {
            "This appears to be a fundamental belief conflict. Consider examining the core assumptions underlying both thoughts. You may need to revise one or both beliefs, or identify a higher-level principle that resolves the tension.".to_string()
        }
        ContradictionType::Logical => {
            "This is a logical inconsistency. Review the reasoning chain that led to each conclusion. One or both may contain a logical error or unstated premise.".to_string()
        }
        ContradictionType::SemanticDivergence => {
            "These thoughts discuss the same topic but reach different conclusions. Consider whether they might both be partially true under different conditions, or whether new evidence could resolve the disagreement.".to_string()
        }
        ContradictionType::Apparent => {
            "This may be a surface-level contradiction that can be resolved with clarification. The thoughts might not actually conflict when context is considered. Review the specific claims in each.".to_string()
        }
        ContradictionType::Factual => {
            "There appears to be a factual inconsistency. Verify the evidence for each claim. One may be based on outdated or incorrect information.".to_string()
        }
        ContradictionType::Temporal => {
            "This contradiction may be time-dependent. Check if one belief supersedes the other, or if both are valid at different points in time.".to_string()
        }
        ContradictionType::SourceConflict => {
            "The sources for these beliefs conflict. Evaluate the credibility and currency of each source to determine which is more reliable.".to_string()
        }
        ContradictionType::ConfidenceInconsistency => {
            "Your confidence levels in related beliefs are inconsistent. Consider whether your certainty in one should affect your certainty in the other.".to_string()
        }
    }
}

/// Get contradictions for a thought
#[hdk_extern]
pub fn get_thought_contradictions(thought_id: String) -> ExternResult<Vec<Record>> {
    let thought_anchor = format!("thought:{}", thought_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&thought_anchor)?, LinkTypes::ThoughtToContradictions)?,
        GetStrategy::default(),
    )?;

    let mut contradictions = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            contradictions.push(record);
        }
    }

    Ok(contradictions)
}

/// Create a coherence report
#[hdk_extern]
pub fn create_coherence_report(input: CreateReportInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let report = CoherenceReport {
        id: generate_uuid(),
        thought_ids: input.thought_ids,
        coherence_score: input.coherence_score,
        phi_estimate: input.phi_estimate,
        contradiction_count: input.contradiction_count,
        knowledge_gaps: input.knowledge_gaps,
        suggestions: input.suggestions,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::CoherenceReport(report))?;

    let agent_anchor = format!("agent_reports:{}", agent);
    create_anchor(&agent_anchor)?;
    create_link(
        anchor_hash(&agent_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToReports,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Report not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateReportInput {
    pub thought_ids: Vec<String>,
    pub coherence_score: f64,
    pub phi_estimate: Option<f64>,
    pub contradiction_count: u32,
    pub knowledge_gaps: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Record an inference
#[hdk_extern]
pub fn record_inference(input: RecordInferenceInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let inference = Inference {
        id: generate_uuid(),
        content: input.content,
        premise_ids: input.premise_ids.clone(),
        inference_type: input.inference_type,
        confidence: input.confidence,
        accepted: false,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::Inference(inference))?;

    let agent_anchor = format!("agent_inferences:{}", agent);
    create_anchor(&agent_anchor)?;
    create_link(
        anchor_hash(&agent_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToInferences,
        (),
    )?;

    // Link to premises
    for premise_id in input.premise_ids {
        let premise_anchor = format!("thought:{}", premise_id);
        create_link(
            action_hash.clone(),
            anchor_hash(&premise_anchor)?,
            LinkTypes::InferenceToPremises,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Inference not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordInferenceInput {
    pub content: String,
    pub premise_ids: Vec<String>,
    pub inference_type: InferenceType,
    pub confidence: f64,
}

/// Get all inferences
#[hdk_extern]
pub fn get_my_inferences(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_anchor = format!("agent_inferences:{}", agent);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&agent_anchor)?, LinkTypes::AgentToInferences)?,
        GetStrategy::default(),
    )?;

    let mut inferences = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            inferences.push(record);
        }
    }

    Ok(inferences)
}
