// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! DKG Coordinator Zome - Truth Engine Interface
//!
//! Exposes the Distributed Knowledge Graph to Holochain's DHT.
//! Uses the mycelix-sdk for confidence calculation and integrates
//! with MATL for reputation-weighted attestations.
//!
//! # Zome Functions
//!
//! - `submit_claim`: Post a verifiable claim to the DHT
//! - `attest_claim`: Endorse or challenge an existing claim
//! - `get_claims`: Query claims by subject
//! - `get_truth`: Get confidence-weighted facts (Truth Engine)
//! - `get_agent_reputation`: Query an agent's reputation score
//! - `list_subjects`: Discover all subjects with claims

use dkg_integrity::{AnchorEntry, AttestationEntry, ClaimEntry, EntryTypes, LinkTypes};
use hdk::prelude::*;
use mycelix_sdk::dkg::{
    ConfidenceInput, EpistemicType, TripleValue, VerifiableTriple, calculate_confidence,
    meets_threshold,
};
use mycelix_sdk::matl::{GovernanceTier, KVector};

// ============================================================================
// Input/Output Types
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitClaimInput {
    pub subject: String,
    pub predicate: String,
    /// The object value as a string (JSON-encoded for complex types)
    pub object: String,
    /// Object type hint: "text", "number", "integer", "boolean"
    #[serde(default = "default_object_type")]
    pub object_type: String,
    #[serde(default)]
    pub epistemic_type: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
}

fn default_object_type() -> String {
    "text".to_string()
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AttestClaimInput {
    pub claim_hash: ActionHash,
    pub attestation_type: String,
    #[serde(default)]
    pub evidence: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WeightedClaim {
    pub claim_hash: ActionHash,
    pub subject: String,
    pub predicate: String,
    /// The object value as a string
    pub object: String,
    /// Object type hint: "text", "number", "integer", "boolean"
    pub object_type: String,
    pub author: AgentPubKey,
    pub confidence: f64,
    pub attestation_count: usize,
    pub endorsements: usize,
    pub challenges: usize,
    pub created_at: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AgentReputationInfo {
    pub agent: AgentPubKey,
    pub claim_count: usize,
    pub attestation_count: usize,
    pub endorsements_received: usize,
    pub challenges_received: usize,
    pub reputation_score: f64,
}

// ============================================================================
// Zome Functions
// ============================================================================

/// Submit a new claim to the DHT
#[hdk_extern]
pub fn submit_claim(input: SubmitClaimInput) -> ExternResult<ActionHash> {
    let agent_info = agent_info()?;
    let now = sys_time()?.as_micros() / 1_000_000;

    let claim = ClaimEntry {
        subject: input.subject.clone(),
        predicate: input.predicate,
        object: input.object,
        object_type: input.object_type,
        epistemic_type: input
            .epistemic_type
            .unwrap_or_else(|| "empirical".to_string()),
        domain: input.domain,
        created_at: now as u64,
    };

    // Validate
    match claim.validate()? {
        ValidateCallbackResult::Valid => {}
        ValidateCallbackResult::Invalid(reason) => {
            return Err(wasm_error!(WasmErrorInner::Guest(reason)));
        }
        _ => {}
    }

    // Create the claim entry
    let claim_hash = create_entry(EntryTypes::Claim(claim.clone()))?;

    // Create links for indexing
    let subject_anchor_hash = create_anchor(&input.subject)?;
    create_link(
        subject_anchor_hash.clone(),
        claim_hash.clone(),
        LinkTypes::SubjectToClaim,
        (),
    )?;

    // Agent -> Claim link for reputation tracking
    create_link(
        agent_info.agent_initial_pubkey,
        claim_hash.clone(),
        LinkTypes::AgentToClaim,
        (),
    )?;

    // Register subject in global index
    let all_subjects_anchor = create_anchor("__all_subjects__")?;
    create_link(
        all_subjects_anchor,
        subject_anchor_hash,
        LinkTypes::AllSubjects,
        input.subject.as_bytes().to_vec(),
    )?;

    Ok(claim_hash)
}

/// Attest to an existing claim (endorse, challenge, or acknowledge)
#[hdk_extern]
pub fn attest_claim(input: AttestClaimInput) -> ExternResult<ActionHash> {
    let agent_info = agent_info()?;
    let now = sys_time()?.as_micros() / 1_000_000;

    // Verify the claim exists
    let _claim_record = get(input.claim_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Claim not found".to_string())))?;

    let attestation = AttestationEntry {
        claim_hash: input.claim_hash.clone(),
        attestation_type: input.attestation_type,
        evidence: input.evidence,
        created_at: now as u64,
    };

    // Validate
    match attestation.validate()? {
        ValidateCallbackResult::Valid => {}
        ValidateCallbackResult::Invalid(reason) => {
            return Err(wasm_error!(WasmErrorInner::Guest(reason)));
        }
        _ => {}
    }

    // Create the attestation entry
    let attestation_hash = create_entry(EntryTypes::Attestation(attestation))?;

    // Link claim -> attestation
    create_link(
        input.claim_hash,
        attestation_hash.clone(),
        LinkTypes::ClaimToAttestation,
        (),
    )?;

    // Link agent -> attestation
    create_link(
        agent_info.agent_initial_pubkey,
        attestation_hash.clone(),
        LinkTypes::AgentToAttestation,
        (),
    )?;

    Ok(attestation_hash)
}

/// Get all claims about a subject
#[hdk_extern]
pub fn get_claims(subject: String) -> ExternResult<Vec<WeightedClaim>> {
    let subject_anchor_hash = create_anchor(&subject)?;
    let now = sys_time()?.as_micros() / 1_000_000;

    let links = get_links(
        LinkQuery::try_new(subject_anchor_hash, LinkTypes::SubjectToClaim)?,
        GetStrategy::default(),
    )?;

    let mut claims = Vec::new();

    for link in links {
        let claim_hash = link
            .target
            .into_action_hash()
            .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string())))?;

        if let Some(weighted) = get_weighted_claim(&claim_hash, now as u64)? {
            claims.push(weighted);
        }
    }

    // Sort by confidence (highest first)
    claims.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(claims)
}

/// Get the "truth" about a subject - confidence-weighted facts
///
/// This is the core Truth Engine function. Returns claims sorted by
/// confidence, filtered to only include those meeting minimum threshold.
#[hdk_extern]
pub fn get_truth(subject: String) -> ExternResult<Vec<WeightedClaim>> {
    let claims = get_claims(subject)?;

    // Filter to only include claims meeting minimum confidence threshold
    let verified: Vec<WeightedClaim> = claims
        .into_iter()
        .filter(|c| meets_threshold(c.confidence, "low"))
        .collect();

    Ok(verified)
}

/// Get an agent's reputation information
#[hdk_extern]
pub fn get_agent_reputation(agent: AgentPubKey) -> ExternResult<AgentReputationInfo> {
    let claim_links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::AgentToClaim)?,
        GetStrategy::default(),
    )?;

    let attestation_links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::AgentToAttestation)?,
        GetStrategy::default(),
    )?;

    // Count endorsements and challenges received
    let mut endorsements_received = 0;
    let mut challenges_received = 0;

    for link in &claim_links {
        if let Some(claim_hash) = link.target.clone().into_action_hash() {
            let attestations = get_claim_attestations(&claim_hash)?;
            for att in attestations {
                if att.is_endorsement() {
                    endorsements_received += 1;
                } else if att.is_challenge() {
                    challenges_received += 1;
                }
            }
        }
    }

    let reputation_score = calculate_agent_reputation(&agent)?;

    Ok(AgentReputationInfo {
        agent,
        claim_count: claim_links.len(),
        attestation_count: attestation_links.len(),
        endorsements_received,
        challenges_received,
        reputation_score,
    })
}

/// List all subjects that have claims
#[hdk_extern]
pub fn list_subjects(_: ()) -> ExternResult<Vec<String>> {
    let anchor = create_anchor("__all_subjects__")?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllSubjects)?,
        GetStrategy::default(),
    )?;

    let subjects: Vec<String> = links
        .into_iter()
        .filter_map(|link| String::from_utf8(link.tag.into_inner()).ok())
        .collect();

    // Deduplicate
    let mut unique: Vec<String> = subjects
        .into_iter()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    unique.sort();

    Ok(unique)
}

/// Health check
#[hdk_extern]
pub fn ping(_: ()) -> ExternResult<String> {
    Ok("pong".to_string())
}

// ============================================================================
// Internal Helpers
// ============================================================================

/// Create an anchor entry and return its hash
fn create_anchor(anchor: &str) -> ExternResult<EntryHash> {
    let anchor_entry = AnchorEntry {
        anchor: anchor.to_string(),
    };
    create_entry(EntryTypes::Anchor(anchor_entry.clone()))?;
    hash_entry(&anchor_entry)
}

/// Get a weighted claim with confidence calculation
fn get_weighted_claim(
    claim_hash: &ActionHash,
    current_time: u64,
) -> ExternResult<Option<WeightedClaim>> {
    let record = match get(claim_hash.clone(), GetOptions::default())? {
        Some(r) => r,
        None => return Ok(None),
    };

    let claim: ClaimEntry = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid claim entry".to_string())))?;

    let author = record.action().author().clone();

    // Get attestations
    let attestations = get_claim_attestations(claim_hash)?;

    let mut endorsements = 0;
    let mut challenges = 0;
    let mut attester_reputations = Vec::new();
    let mut contradiction_weight = 0.0;

    for att in &attestations {
        // Get attester's reputation
        let att_record = get(att.claim_hash.clone(), GetOptions::default())?;
        if let Some(rec) = att_record {
            let attester = rec.action().author();
            let rep = calculate_agent_reputation(attester)?;

            if att.is_endorsement() {
                endorsements += 1;
                attester_reputations.push(rep);
            } else if att.is_challenge() {
                challenges += 1;
                contradiction_weight += rep;
            }
        }
    }

    // Build VerifiableTriple for SDK confidence calculation
    let object = match claim.object_type.as_str() {
        "number" => claim
            .object
            .parse::<f64>()
            .map(TripleValue::Float)
            .unwrap_or_else(|_| TripleValue::String(claim.object.clone())),
        "integer" => claim
            .object
            .parse::<i64>()
            .map(TripleValue::Integer)
            .unwrap_or_else(|_| TripleValue::String(claim.object.clone())),
        "boolean" => claim
            .object
            .parse::<bool>()
            .map(TripleValue::Boolean)
            .unwrap_or_else(|_| TripleValue::String(claim.object.clone())),
        _ => TripleValue::String(claim.object.clone()),
    };

    let epistemic_type = match claim.epistemic_type.as_str() {
        "normative" => EpistemicType::Normative,
        "metaphysical" => EpistemicType::Metaphysical,
        _ => EpistemicType::Empirical,
    };

    let triple = VerifiableTriple::new(claim.subject.clone(), claim.predicate.as_str(), object)
        .with_epistemic_type(epistemic_type)
        .with_timestamp(claim.created_at);

    let input = ConfidenceInput {
        triple: &triple,
        attestation_count: attestations.len(),
        attester_reputations: &attester_reputations,
        contradiction_weights: contradiction_weight,
        current_time,
    };

    let confidence_score = calculate_confidence(&input);

    Ok(Some(WeightedClaim {
        claim_hash: claim_hash.clone(),
        subject: claim.subject,
        predicate: claim.predicate,
        object: claim.object,
        object_type: claim.object_type,
        author,
        confidence: confidence_score.score,
        attestation_count: attestations.len(),
        endorsements,
        challenges,
        created_at: claim.created_at,
    }))
}

/// Get all attestations for a claim
fn get_claim_attestations(claim_hash: &ActionHash) -> ExternResult<Vec<AttestationEntry>> {
    let links = get_links(
        LinkQuery::try_new(claim_hash.clone(), LinkTypes::ClaimToAttestation)?,
        GetStrategy::default(),
    )?;

    let mut attestations = Vec::new();

    for link in links {
        let att_hash = link.target.into_action_hash().ok_or_else(|| {
            wasm_error!(WasmErrorInner::Guest(
                "Invalid attestation link".to_string()
            ))
        })?;

        if let Some(record) = get(att_hash, GetOptions::default())? {
            let att: AttestationEntry = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest("Invalid attestation".to_string()))
                })?;

            attestations.push(att);
        }
    }

    Ok(attestations)
}

/// Calculate agent reputation from their claim history
/// Integrates with MATL K-Vector for multi-dimensional trust scoring
fn calculate_agent_reputation(agent: &AgentPubKey) -> ExternResult<f64> {
    let claim_links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::AgentToClaim)?,
        GetStrategy::default(),
    )?;

    if claim_links.is_empty() {
        return Ok(0.5); // Neutral reputation for new agents
    }

    let mut total_endorsements = 0;
    let mut total_challenges = 0;

    for link in &claim_links {
        if let Some(claim_hash) = link.target.clone().into_action_hash() {
            let attestations = get_claim_attestations(&claim_hash)?;
            for att in attestations {
                if att.is_endorsement() {
                    total_endorsements += 1;
                } else if att.is_challenge() {
                    total_challenges += 1;
                }
            }
        }
    }

    // Build K-Vector from on-chain activity
    let kvector =
        build_kvector_from_activity(claim_links.len(), total_endorsements, total_challenges);

    // Use MATL trust score (weighted K-Vector)
    Ok(kvector.trust_score() as f64)
}

/// Build a K-Vector from on-chain activity metrics
/// This integrates DKG with MATL's 8-dimensional trust model
fn build_kvector_from_activity(
    claim_count: usize,
    endorsements: usize,
    challenges: usize,
) -> KVector {
    let total_feedback = endorsements + challenges;

    // k_r (Reputation): Based on endorsement ratio
    let k_r = if total_feedback > 0 {
        (endorsements as f32 / total_feedback as f32).clamp(0.0, 1.0)
    } else {
        0.5 // Neutral for no feedback
    };

    // k_a (Activity): Based on claim count (log scale, max at 100 claims)
    let k_a = ((claim_count as f32).ln_1p() / 100_f32.ln_1p()).clamp(0.0, 1.0);

    // k_i (Integrity): High if few challenges relative to endorsements
    let k_i = if total_feedback > 0 {
        let challenge_ratio = challenges as f32 / total_feedback as f32;
        (1.0 - challenge_ratio * 2.0).clamp(0.0, 1.0)
    } else {
        0.5
    };

    // k_p (Performance): Based on feedback quality
    let k_p = if total_feedback > 5 {
        k_r * 1.1 // Boost for established track record
    } else {
        0.5
    }
    .clamp(0.0, 1.0);

    // k_m (Membership): Based on activity level
    let k_m = if claim_count > 10 {
        0.8
    } else if claim_count > 0 {
        0.5
    } else {
        0.2
    };

    // k_s (Stake): Not tracked on-chain in DKG, use neutral
    let k_s = 0.5;

    // k_h (Historical): Based on consistent positive feedback
    let k_h = if endorsements > 10 && challenges < endorsements / 5 {
        0.9
    } else if endorsements > challenges {
        0.6
    } else {
        0.3
    };

    // k_topo (Topology): Not tracked in DKG, use neutral
    let k_topo = 0.5;

    KVector::new(k_r, k_a, k_i, k_p, k_m, k_s, k_h, k_topo)
}

/// Compute governance tier from trust score
fn compute_governance_tier(trust_score: f32) -> GovernanceTier {
    if trust_score >= 0.6 {
        GovernanceTier::Constitutional
    } else if trust_score >= 0.4 {
        GovernanceTier::Major
    } else if trust_score >= 0.3 {
        GovernanceTier::Basic
    } else {
        GovernanceTier::Observer
    }
}

/// Get an agent's governance tier based on their K-Vector
#[hdk_extern]
pub fn get_agent_governance_tier(agent: AgentPubKey) -> ExternResult<String> {
    let claim_links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::AgentToClaim)?,
        GetStrategy::default(),
    )?;

    let mut total_endorsements = 0;
    let mut total_challenges = 0;

    for link in &claim_links {
        if let Some(claim_hash) = link.target.clone().into_action_hash() {
            let attestations = get_claim_attestations(&claim_hash)?;
            for att in attestations {
                if att.is_endorsement() {
                    total_endorsements += 1;
                } else if att.is_challenge() {
                    total_challenges += 1;
                }
            }
        }
    }

    let kvector =
        build_kvector_from_activity(claim_links.len(), total_endorsements, total_challenges);

    let tier = compute_governance_tier(kvector.trust_score());
    Ok(format!("{:?}", tier))
}
