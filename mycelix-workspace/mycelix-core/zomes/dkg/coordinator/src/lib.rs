// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DKG (Distributed Knowledge Graph) Coordinator Zome
//!
//! Public API for the Truth Ledger.
//!
//! Key operations:
//! - `add_knowledge`: Create a new verifiable triple
//! - `attest_knowledge`: Reinforce or dispute a triple
//! - `query_knowledge`: Search with confidence filtering
//! - `get_confidence`: Get the confidence level of a specific triple

use hdk::prelude::*;
use dkg_integrity::{
    Attestation, EntryTypes, KnowledgeReputation, LinkTypes, VerifiableTriple,
    MIN_INITIAL_CONFIDENCE, MAX_CONFIDENCE, MIN_ATTESTATION_REPUTATION,
};

// ============================================================================
// INPUT/OUTPUT TYPES
// ============================================================================

/// Input for creating a new knowledge triple
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddKnowledgeInput {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub initial_confidence: f32,
    pub evidence_hash: Option<String>,
    pub empirical_level: u8,
    pub normative_level: u8,
    pub materiality_level: u8,
}

/// Output from adding knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddKnowledgeOutput {
    pub triple_hash: ActionHash,
    pub initial_confidence: f32,
    pub epistemic_classification: String,
}

/// Input for attesting to a triple
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestInput {
    pub triple_hash: ActionHash,
    pub agreement: bool,  // true = confirm, false = dispute
    pub evidence_hash: Option<String>,
}

/// Result of a confidence query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceResult {
    pub triple_hash: ActionHash,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub raw_confidence: f32,
    pub effective_confidence: f32,
    pub epistemic_classification: String,
    pub attestation_count: u32,
    pub dispute_count: u32,
}

/// Input for querying knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryInput {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub min_confidence: Option<f32>,
    pub limit: Option<usize>,
}

/// Agent's knowledge stats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentKnowledgeStats {
    pub agent: String,
    pub reputation_score: f32,
    pub total_claims: u64,
    pub confirmed_claims: u64,
    pub disputed_claims: u64,
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Get current timestamp as microseconds
fn current_timestamp_micros() -> ExternResult<i64> {
    Ok(sys_time()?.0 as i64)
}

/// Deserialize entry from record (HDK 0.6 pattern)
fn deserialize_entry<T: TryFrom<SerializedBytes>>(record: &Record) -> Option<T> {
    if let Some(entry) = record.entry().as_option() {
        if let Entry::App(bytes) = entry {
            if let Ok(value) = T::try_from(
                SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
            ) {
                return Some(value);
            }
        }
    }
    None
}

/// Get agent's reputation score from their reputation record
fn get_agent_reputation_score(agent_str: &str) -> ExternResult<f32> {
    let path = Path::from(format!("reputation.{}", agent_str));
    let typed_path = path.typed(LinkTypes::AgentToReputation)?;
    let path_hash = typed_path.path_entry_hash()?;

    let links = get_links(
        LinkQuery::new(
            AnyLinkableHash::from(path_hash),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AgentToReputation as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.into_iter().next() {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                if let Some(rep) = deserialize_entry::<KnowledgeReputation>(&record) {
                    return Ok(rep.reputation_score);
                }
            }
        }
    }

    // Default reputation for new agents
    Ok(0.5)
}

/// Count attestations and disputes for a triple
fn count_attestations_for_triple(triple_hash: &ActionHash) -> ExternResult<(u32, u32)> {
    let links = get_links(
        LinkQuery::new(
            AnyLinkableHash::from(triple_hash.clone()),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::TripleToAttestations as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut attestations = 0u32;
    let mut disputes = 0u32;

    for link in links {
        if let Some(att_hash) = link.target.into_action_hash() {
            if let Some(record) = get(att_hash, GetOptions::default())? {
                if let Some(att) = deserialize_entry::<Attestation>(&record) {
                    if att.agreement {
                        attestations += 1;
                    } else {
                        disputes += 1;
                    }
                }
            }
        }
    }

    Ok((attestations, disputes))
}

// ============================================================================
// PUBLIC FUNCTIONS
// ============================================================================

/// Add a new piece of knowledge to the graph.
///
/// Creates a VerifiableTriple with the given epistemic classification.
#[hdk_extern]
pub fn add_knowledge(input: AddKnowledgeInput) -> ExternResult<AddKnowledgeOutput> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = current_timestamp_micros()?;

    // Validate minimum confidence for empirical level
    let min_confidence = match input.empirical_level {
        0 => MIN_INITIAL_CONFIDENCE,  // E0
        1 => 0.15,                     // E1
        2 => 0.25,                     // E2
        3 => 0.4,                      // E3
        4 => 0.5,                      // E4
        _ => MIN_INITIAL_CONFIDENCE,
    };

    if input.initial_confidence < min_confidence {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Initial confidence {} below minimum {} for E{}",
                input.initial_confidence, min_confidence, input.empirical_level)
        )));
    }

    let triple = VerifiableTriple {
        subject: input.subject.clone(),
        predicate: input.predicate.clone(),
        object: input.object.clone(),
        confidence: input.initial_confidence,
        evidence_hash: input.evidence_hash,
        empirical_level: input.empirical_level,
        normative_level: input.normative_level,
        materiality_level: input.materiality_level,
        creator: agent.to_string(),
        created_at: now,
        last_reinforced_at: now,
        version: 1,
    };

    let epistemic_class = triple.epistemic_classification();

    // Create the entry
    let action_hash = create_entry(&EntryTypes::VerifiableTriple(triple))?;

    // Create index links using typed paths
    let subject_path = Path::from(format!("subject.{}", input.subject));
    let subject_typed = subject_path.typed(LinkTypes::SubjectToTriples)?;
    subject_typed.ensure()?;
    create_link(
        subject_typed.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::SubjectToTriples,
        ()
    )?;

    let predicate_path = Path::from(format!("predicate.{}", input.predicate));
    let predicate_typed = predicate_path.typed(LinkTypes::PredicateToTriples)?;
    predicate_typed.ensure()?;
    create_link(
        predicate_typed.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::PredicateToTriples,
        ()
    )?;

    let object_path = Path::from(format!("object.{}", input.object));
    let object_typed = object_path.typed(LinkTypes::ObjectToTriples)?;
    object_typed.ensure()?;
    create_link(
        object_typed.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::ObjectToTriples,
        ()
    )?;

    let agent_path = Path::from(format!("agent.{}", agent));
    let agent_typed = agent_path.typed(LinkTypes::AgentToTriples)?;
    agent_typed.ensure()?;
    create_link(
        agent_typed.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AgentToTriples,
        ()
    )?;

    Ok(AddKnowledgeOutput {
        triple_hash: action_hash,
        initial_confidence: input.initial_confidence,
        epistemic_classification: epistemic_class,
    })
}

/// Attest to an existing triple (reinforce or dispute).
///
/// Attestations from higher-reputation agents have more impact.
#[hdk_extern]
pub fn attest_knowledge(input: AttestInput) -> ExternResult<ConfidenceResult> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_str = agent.to_string();
    let now = current_timestamp_micros()?;

    // Get caller's reputation
    let caller_reputation = get_agent_reputation_score(&agent_str)?;

    if caller_reputation < MIN_ATTESTATION_REPUTATION {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Reputation {} below attestation threshold {}",
                caller_reputation, MIN_ATTESTATION_REPUTATION)
        )));
    }

    // Get the current triple
    let record = get(input.triple_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Triple not found".to_string()
        )))?;

    let triple: VerifiableTriple = deserialize_entry(&record)
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not deserialize triple".to_string()
        )))?;

    // Check caller isn't attesting their own claim
    if triple.creator == agent_str {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot attest to your own claims".to_string()
        )));
    }

    // Create attestation record
    let attestation = Attestation {
        triple_hash: input.triple_hash.to_string(),
        attester: agent_str,
        attester_reputation: caller_reputation,
        agreement: input.agreement,
        evidence_hash: input.evidence_hash,
        attested_at: now,
    };

    let attestation_hash = create_entry(&EntryTypes::Attestation(attestation))?;

    // Link attestation to triple
    create_link(
        input.triple_hash.clone(),
        attestation_hash,
        LinkTypes::TripleToAttestations,
        ()
    )?;

    // Calculate new confidence
    let confidence_delta = if input.agreement {
        VerifiableTriple::reinforcement_delta(caller_reputation, triple.confidence)
    } else {
        -VerifiableTriple::dispute_delta(caller_reputation, triple.confidence)
    };

    let new_confidence = (triple.confidence + confidence_delta)
        .clamp(MIN_INITIAL_CONFIDENCE, MAX_CONFIDENCE);

    // Update the triple with new confidence
    let updated_triple = VerifiableTriple {
        confidence: new_confidence,
        last_reinforced_at: now,
        version: triple.version + 1,
        ..triple.clone()
    };

    // Calculate effective confidence before moving
    let effective_conf = updated_triple.effective_confidence(now);
    let epist_class = updated_triple.epistemic_classification();

    update_entry(input.triple_hash.clone(), &EntryTypes::VerifiableTriple(updated_triple.clone()))?;

    // Get attestation counts
    let (attestation_count, dispute_count) = count_attestations_for_triple(&input.triple_hash)?;

    Ok(ConfidenceResult {
        triple_hash: input.triple_hash,
        subject: updated_triple.subject,
        predicate: updated_triple.predicate,
        object: updated_triple.object,
        raw_confidence: new_confidence,
        effective_confidence: effective_conf,
        epistemic_classification: epist_class,
        attestation_count,
        dispute_count,
    })
}

/// Get the confidence level of a specific triple.
///
/// Returns both raw confidence and effective confidence (after decay).
#[hdk_extern]
pub fn get_confidence(triple_hash: ActionHash) -> ExternResult<ConfidenceResult> {
    let now = current_timestamp_micros()?;

    let record = get(triple_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Triple not found".to_string()
        )))?;

    let triple: VerifiableTriple = deserialize_entry(&record)
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not deserialize triple".to_string()
        )))?;

    let (attestation_count, dispute_count) = count_attestations_for_triple(&triple_hash)?;

    // Calculate before moving fields
    let effective_conf = triple.effective_confidence(now);
    let epist_class = triple.epistemic_classification();

    Ok(ConfidenceResult {
        triple_hash,
        subject: triple.subject,
        predicate: triple.predicate,
        object: triple.object,
        raw_confidence: triple.confidence,
        effective_confidence: effective_conf,
        epistemic_classification: epist_class,
        attestation_count,
        dispute_count,
    })
}

/// Query knowledge by subject.
#[hdk_extern]
pub fn query_by_subject(subject: String) -> ExternResult<Vec<ConfidenceResult>> {
    let now = current_timestamp_micros()?;
    let path = Path::from(format!("subject.{}", subject));
    let typed_path = path.typed(LinkTypes::SubjectToTriples)?;
    let path_hash = typed_path.path_entry_hash()?;

    let links = get_links(
        LinkQuery::new(
            AnyLinkableHash::from(path_hash),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::SubjectToTriples as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut results = Vec::new();

    for link in links.into_iter().take(100) {
        if let Some(triple_hash) = link.target.into_action_hash() {
            if let Some(record) = get(triple_hash.clone(), GetOptions::default())? {
                if let Some(triple) = deserialize_entry::<VerifiableTriple>(&record) {
                    let (attestation_count, dispute_count) = count_attestations_for_triple(&triple_hash)?;

                    // Calculate before moving fields
                    let effective_conf = triple.effective_confidence(now);
                    let epist_class = triple.epistemic_classification();

                    results.push(ConfidenceResult {
                        triple_hash,
                        subject: triple.subject,
                        predicate: triple.predicate,
                        object: triple.object,
                        raw_confidence: triple.confidence,
                        effective_confidence: effective_conf,
                        epistemic_classification: epist_class,
                        attestation_count,
                        dispute_count,
                    });
                }
            }
        }
    }

    // Sort by effective confidence (highest first)
    results.sort_by(|a, b| b.effective_confidence.partial_cmp(&a.effective_confidence)
        .unwrap_or(std::cmp::Ordering::Equal));

    Ok(results)
}

/// Query knowledge by predicate.
#[hdk_extern]
pub fn query_by_predicate(predicate: String) -> ExternResult<Vec<ConfidenceResult>> {
    let now = current_timestamp_micros()?;
    let path = Path::from(format!("predicate.{}", predicate));
    let typed_path = path.typed(LinkTypes::PredicateToTriples)?;
    let path_hash = typed_path.path_entry_hash()?;

    let links = get_links(
        LinkQuery::new(
            AnyLinkableHash::from(path_hash),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::PredicateToTriples as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut results = Vec::new();

    for link in links.into_iter().take(100) {
        if let Some(triple_hash) = link.target.into_action_hash() {
            if let Some(record) = get(triple_hash.clone(), GetOptions::default())? {
                if let Some(triple) = deserialize_entry::<VerifiableTriple>(&record) {
                    let (attestation_count, dispute_count) = count_attestations_for_triple(&triple_hash)?;

                    // Calculate before moving fields
                    let effective_conf = triple.effective_confidence(now);
                    let epist_class = triple.epistemic_classification();

                    results.push(ConfidenceResult {
                        triple_hash,
                        subject: triple.subject,
                        predicate: triple.predicate,
                        object: triple.object,
                        raw_confidence: triple.confidence,
                        effective_confidence: effective_conf,
                        epistemic_classification: epist_class,
                        attestation_count,
                        dispute_count,
                    });
                }
            }
        }
    }

    results.sort_by(|a, b| b.effective_confidence.partial_cmp(&a.effective_confidence)
        .unwrap_or(std::cmp::Ordering::Equal));

    Ok(results)
}

/// Get my knowledge stats.
#[hdk_extern]
pub fn get_my_knowledge_stats(_: ()) -> ExternResult<AgentKnowledgeStats> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_str = agent.to_string();

    let reputation_score = get_agent_reputation_score(&agent_str)?;

    // Count claims via links
    let agent_path = Path::from(format!("agent.{}", agent));
    let typed_path = agent_path.typed(LinkTypes::AgentToTriples)?;
    let path_hash = typed_path.path_entry_hash()?;

    let links = get_links(
        LinkQuery::new(
            AnyLinkableHash::from(path_hash),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AgentToTriples as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    Ok(AgentKnowledgeStats {
        agent: agent_str,
        reputation_score,
        total_claims: links.len() as u64,
        confirmed_claims: 0,  // Would need to iterate and check
        disputed_claims: 0,
    })
}
