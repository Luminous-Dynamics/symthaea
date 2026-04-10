// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DKG Coordinator Zome - Truth Engine Interface
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

use dkg_integrity::{
    AnchorEntry, AttestationEntry, ClaimEntry, ClaimStatus, ConsensusSnapshot, DisputeEntry,
    DisputeStatus, EntryTypes, LinkTypes,
};
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
// Consensus Mechanism
// ============================================================================

/// Minimum endorsements required for quorum
const MIN_ENDORSEMENTS: u32 = 3;
/// Minimum aggregate K-vector trust score for quorum
const MIN_AGGREGATE_TRUST: f64 = 2.0;
/// Maximum challenge ratio before claim is contested
const MAX_CHALLENGE_RATIO: f64 = 0.3;
/// Days after which temporal decay begins
const DECAY_START_DAYS: u64 = 90;
/// Confidence reduction per 30-day period after decay starts
const DECAY_PER_PERIOD: f64 = 0.10;

#[derive(Serialize, Deserialize, Debug)]
pub struct EvaluateConsensusResult {
    pub claim_hash: ActionHash,
    pub status: ClaimStatus,
    pub endorsement_count: u32,
    pub challenge_count: u32,
    pub aggregate_reputation: f64,
    pub confidence: f64,
    pub snapshot_hash: ActionHash,
}

/// Evaluate consensus for a claim.
///
/// Computes quorum: requires aggregate K-vector score of endorsers > 2.0
/// AND endorsement count >= 3 AND challenge ratio < 0.3.
/// Creates a ConsensusSnapshot entry recording the evaluation.
#[hdk_extern]
pub fn evaluate_consensus(claim_hash: ActionHash) -> ExternResult<EvaluateConsensusResult> {
    let now = sys_time()?.as_micros() / 1_000_000;

    // Verify the claim exists
    let _claim_record = get(claim_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Claim not found".to_string())))?;

    // Get attestations
    let attestations = get_claim_attestations(&claim_hash)?;

    let mut endorsement_count: u32 = 0;
    let mut challenge_count: u32 = 0;
    let mut aggregate_reputation: f64 = 0.0;

    for att in &attestations {
        // Get the attestation's action record to find the attester
        let att_links = get_links(
            LinkQuery::try_new(claim_hash.clone(), LinkTypes::ClaimToAttestation)?,
            GetStrategy::default(),
        )?;

        // We already have the attestation entry; compute trust from the claim author pattern
        if att.is_endorsement() {
            endorsement_count += 1;
            // Approximate: use a default trust contribution per endorser
            // In production, we'd look up each attester's K-vector
            aggregate_reputation += 0.5; // Conservative default per endorser
        } else if att.is_challenge() {
            challenge_count += 1;
        }
    }

    // Refine aggregate_reputation by looking up actual attesters
    // Walk attestation links to find attester agents
    let att_links = get_links(
        LinkQuery::try_new(claim_hash.clone(), LinkTypes::ClaimToAttestation)?,
        GetStrategy::default(),
    )?;
    let mut real_aggregate: f64 = 0.0;
    let mut real_endorsements: u32 = 0;
    let mut real_challenges: u32 = 0;

    for link in &att_links {
        let att_hash = link.target.clone().into_action_hash().ok_or_else(|| {
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
            let attester = record.action().author();
            let rep = calculate_agent_reputation(attester)?;

            if att.is_endorsement() {
                real_endorsements += 1;
                real_aggregate += rep;
            } else if att.is_challenge() {
                real_challenges += 1;
            }
        }
    }

    // Use the real computed values
    endorsement_count = real_endorsements;
    challenge_count = real_challenges;
    aggregate_reputation = real_aggregate;

    // Check for active disputes
    let dispute_links = get_links(
        LinkQuery::try_new(claim_hash.clone(), LinkTypes::ClaimToDispute)?,
        GetStrategy::default(),
    )?;
    let has_open_dispute = dispute_links.iter().any(|link| {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Ok(Some(record)) = get(hash, GetOptions::default()) {
                if let Ok(Some(dispute)) = record.entry().to_app_option::<DisputeEntry>() {
                    return dispute.status == DisputeStatus::Open;
                }
            }
        }
        false
    });

    // Determine status
    let total_attestations = endorsement_count + challenge_count;
    let challenge_ratio = if total_attestations > 0 {
        challenge_count as f64 / total_attestations as f64
    } else {
        0.0
    };

    // Get weighted claim for confidence score
    let weighted = get_weighted_claim(&claim_hash, now as u64)?;
    let confidence = weighted.map(|w| w.confidence).unwrap_or(0.0);

    let status = if has_open_dispute {
        ClaimStatus::Contested
    } else if challenge_ratio > MAX_CHALLENGE_RATIO {
        ClaimStatus::Contested
    } else if endorsement_count >= MIN_ENDORSEMENTS
        && aggregate_reputation >= MIN_AGGREGATE_TRUST
        && challenge_ratio <= MAX_CHALLENGE_RATIO
    {
        ClaimStatus::Established
    } else if endorsement_count > 0 {
        ClaimStatus::Attested
    } else {
        ClaimStatus::Proposed
    };

    // Create consensus snapshot
    let snapshot = ConsensusSnapshot {
        claim_hash: claim_hash.clone(),
        status: status.clone(),
        endorsement_count,
        challenge_count,
        aggregate_reputation,
        confidence,
        evaluated_at: now as u64,
    };

    let snapshot_hash = create_entry(EntryTypes::ConsensusSnapshot(snapshot))?;

    // Link claim to snapshot
    create_link(
        claim_hash.clone(),
        snapshot_hash.clone(),
        LinkTypes::ClaimToConsensus,
        (),
    )?;

    Ok(EvaluateConsensusResult {
        claim_hash,
        status,
        endorsement_count,
        challenge_count,
        aggregate_reputation,
        confidence,
        snapshot_hash,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FileDisputeInput {
    pub claim_hash: ActionHash,
    pub reason: String,
    pub evidence: Vec<String>,
}

/// File a dispute against a claim.
///
/// Creates a DisputeEntry, transitions claim to Contested status.
/// Requires challenger to have governance tier >= Basic.
#[hdk_extern]
pub fn file_dispute(input: FileDisputeInput) -> ExternResult<ActionHash> {
    let agent_info = agent_info()?;
    let now = sys_time()?.as_micros() / 1_000_000;

    // Verify claim exists
    let _claim_record = get(input.claim_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Claim not found".to_string())))?;

    // Check challenger's governance tier
    let rep = calculate_agent_reputation(&agent_info.agent_initial_pubkey)?;
    let claim_links = get_links(
        LinkQuery::try_new(
            agent_info.agent_initial_pubkey.clone(),
            LinkTypes::AgentToClaim,
        )?,
        GetStrategy::default(),
    )?;
    let mut total_endorsements = 0;
    let mut total_challenges = 0;
    for link in &claim_links {
        if let Some(ch) = link.target.clone().into_action_hash() {
            let atts = get_claim_attestations(&ch)?;
            for att in atts {
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
    match tier {
        GovernanceTier::Observer => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Challenger must have governance tier >= Basic to file disputes".to_string()
            )));
        }
        _ => {} // Basic, Major, Constitutional are all allowed
    }

    // Validate evidence count
    if input.evidence.len() > 5 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Maximum 5 evidence URIs allowed".to_string()
        )));
    }

    let dispute = DisputeEntry {
        claim_hash: input.claim_hash.clone(),
        challenger: agent_info.agent_initial_pubkey.clone(),
        reason: input.reason,
        evidence: input.evidence,
        status: DisputeStatus::Open,
        resolution: None,
        created_at: now as u64,
    };

    // Validate
    match dispute.validate()? {
        ValidateCallbackResult::Valid => {}
        ValidateCallbackResult::Invalid(reason) => {
            return Err(wasm_error!(WasmErrorInner::Guest(reason)));
        }
        _ => {}
    }

    let dispute_hash = create_entry(EntryTypes::Dispute(dispute))?;

    // Link claim to dispute
    create_link(
        input.claim_hash.clone(),
        dispute_hash.clone(),
        LinkTypes::ClaimToDispute,
        (),
    )?;

    // Link agent to dispute
    create_link(
        agent_info.agent_initial_pubkey,
        dispute_hash.clone(),
        LinkTypes::AgentToDispute,
        (),
    )?;

    Ok(dispute_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResolveDisputeInput {
    pub dispute_hash: ActionHash,
    pub resolution: String,
    pub upheld: bool,
}

/// Resolve a dispute. Only callable by agents with Constitutional tier.
///
/// If upheld, claim stays Contested→Resolved with reduced confidence.
/// If dismissed, claim returns to Attested.
#[hdk_extern]
pub fn resolve_dispute(input: ResolveDisputeInput) -> ExternResult<ActionHash> {
    let agent_info = agent_info()?;

    // Check resolver's governance tier — must be Constitutional
    let claim_links = get_links(
        LinkQuery::try_new(
            agent_info.agent_initial_pubkey.clone(),
            LinkTypes::AgentToClaim,
        )?,
        GetStrategy::default(),
    )?;
    let mut total_endorsements = 0;
    let mut total_challenges = 0;
    for link in &claim_links {
        if let Some(ch) = link.target.clone().into_action_hash() {
            let atts = get_claim_attestations(&ch)?;
            for att in atts {
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
    if tier != GovernanceTier::Constitutional {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only Constitutional-tier agents can resolve disputes".to_string()
        )));
    }

    // Get the original dispute
    let record = get(input.dispute_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Dispute not found".to_string())))?;

    let dispute: DisputeEntry = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid dispute entry".to_string())))?;

    if dispute.status != DisputeStatus::Open {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Dispute is not open".to_string()
        )));
    }

    // Update dispute status
    let resolved_dispute = DisputeEntry {
        claim_hash: dispute.claim_hash.clone(),
        challenger: dispute.challenger,
        reason: dispute.reason,
        evidence: dispute.evidence,
        status: if input.upheld {
            DisputeStatus::Resolved
        } else {
            DisputeStatus::Dismissed
        },
        resolution: Some(input.resolution),
        created_at: dispute.created_at,
    };

    let updated_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::Dispute(resolved_dispute),
    )?;

    // Re-evaluate consensus after dispute resolution
    let _ = evaluate_consensus(dispute.claim_hash)?;

    Ok(updated_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TemporalDecayResult {
    pub claim_hash: ActionHash,
    pub original_confidence: f64,
    pub decayed_confidence: f64,
    pub status: ClaimStatus,
    pub snapshot_hash: ActionHash,
}

/// Apply temporal decay to a claim's confidence.
///
/// Claims older than 90 days without re-attestation lose 10% confidence
/// per 30-day period. Claims below "low" threshold transition to Decayed.
#[hdk_extern]
pub fn apply_temporal_decay(claim_hash: ActionHash) -> ExternResult<TemporalDecayResult> {
    let now = sys_time()?.as_micros() / 1_000_000;

    // Get the claim to find its age
    let record = get(claim_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Claim not found".to_string())))?;

    let claim: ClaimEntry = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid claim entry".to_string())))?;

    let age_seconds = (now as u64).saturating_sub(claim.created_at);
    let age_days = age_seconds / 86400;

    // Get current confidence from weighted claim
    let weighted = get_weighted_claim(&claim_hash, now as u64)?;
    let original_confidence = weighted.map(|w| w.confidence).unwrap_or(0.0);

    // Apply decay if older than threshold
    let decayed_confidence = if age_days > DECAY_START_DAYS {
        let decay_periods = (age_days - DECAY_START_DAYS) / 30;
        let decay_factor = 1.0 - (decay_periods as f64 * DECAY_PER_PERIOD);
        (original_confidence * decay_factor.max(0.0)).max(0.0)
    } else {
        original_confidence
    };

    let status = if decayed_confidence < 0.1 && age_days > DECAY_START_DAYS {
        ClaimStatus::Decayed
    } else if decayed_confidence < original_confidence {
        // Re-evaluate with decay
        ClaimStatus::Attested
    } else {
        ClaimStatus::Attested
    };

    // Get attestation counts for snapshot
    let att_links = get_links(
        LinkQuery::try_new(claim_hash.clone(), LinkTypes::ClaimToAttestation)?,
        GetStrategy::default(),
    )?;
    let mut endorsement_count: u32 = 0;
    let mut challenge_count: u32 = 0;
    let mut aggregate_rep: f64 = 0.0;

    for link in &att_links {
        let att_hash = link
            .target
            .clone()
            .into_action_hash()
            .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link".to_string())))?;
        if let Some(rec) = get(att_hash, GetOptions::default())? {
            let att: AttestationEntry = rec
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest("Invalid attestation".to_string()))
                })?;
            let attester = rec.action().author();
            let rep = calculate_agent_reputation(attester)?;
            if att.is_endorsement() {
                endorsement_count += 1;
                aggregate_rep += rep;
            } else if att.is_challenge() {
                challenge_count += 1;
            }
        }
    }

    let snapshot = ConsensusSnapshot {
        claim_hash: claim_hash.clone(),
        status: status.clone(),
        endorsement_count,
        challenge_count,
        aggregate_reputation: aggregate_rep,
        confidence: decayed_confidence,
        evaluated_at: now as u64,
    };

    let snapshot_hash = create_entry(EntryTypes::ConsensusSnapshot(snapshot))?;
    create_link(
        claim_hash.clone(),
        snapshot_hash.clone(),
        LinkTypes::ClaimToConsensus,
        (),
    )?;

    Ok(TemporalDecayResult {
        claim_hash,
        original_confidence,
        decayed_confidence,
        status,
        snapshot_hash,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MarketResolvedInput {
    pub market_id: String,
    pub claim_hash: ActionHash,
    pub outcome: String,
    pub confidence: f64,
}

/// Callback from markets_integration when a verification market resolves.
///
/// If market confidence > 0.7, auto-transitions claim to Resolved/Established.
#[hdk_extern]
pub fn on_market_resolved(input: MarketResolvedInput) -> ExternResult<EvaluateConsensusResult> {
    let now = sys_time()?.as_micros() / 1_000_000;

    // Verify claim exists
    let _record = get(input.claim_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Claim not found".to_string())))?;

    // Determine status based on market outcome
    let market_verified = input.outcome == "Yes" && input.confidence > 0.7;

    // Get current attestation state
    let att_links = get_links(
        LinkQuery::try_new(input.claim_hash.clone(), LinkTypes::ClaimToAttestation)?,
        GetStrategy::default(),
    )?;
    let mut endorsement_count: u32 = 0;
    let mut challenge_count: u32 = 0;
    let mut aggregate_rep: f64 = 0.0;

    for link in &att_links {
        let att_hash = link
            .target
            .clone()
            .into_action_hash()
            .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link".to_string())))?;
        if let Some(rec) = get(att_hash, GetOptions::default())? {
            let att: AttestationEntry = rec
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest("Invalid attestation".to_string()))
                })?;
            let attester = rec.action().author();
            let rep = calculate_agent_reputation(attester)?;
            if att.is_endorsement() {
                endorsement_count += 1;
                aggregate_rep += rep;
            } else if att.is_challenge() {
                challenge_count += 1;
            }
        }
    }

    let status = if market_verified {
        ClaimStatus::Established
    } else {
        ClaimStatus::Resolved
    };

    let snapshot = ConsensusSnapshot {
        claim_hash: input.claim_hash.clone(),
        status: status.clone(),
        endorsement_count,
        challenge_count,
        aggregate_reputation: aggregate_rep,
        confidence: input.confidence,
        evaluated_at: now as u64,
    };

    let snapshot_hash = create_entry(EntryTypes::ConsensusSnapshot(snapshot))?;
    create_link(
        input.claim_hash.clone(),
        snapshot_hash.clone(),
        LinkTypes::ClaimToConsensus,
        (),
    )?;

    Ok(EvaluateConsensusResult {
        claim_hash: input.claim_hash,
        status,
        endorsement_count,
        challenge_count,
        aggregate_reputation: aggregate_rep,
        confidence: input.confidence,
        snapshot_hash,
    })
}

/// Get the latest consensus snapshot for a claim
#[hdk_extern]
pub fn get_claim_consensus(claim_hash: ActionHash) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(claim_hash, LinkTypes::ClaimToConsensus)?,
        GetStrategy::default(),
    )?;

    // Return the most recent snapshot (last link)
    if let Some(link) = links.last() {
        let hash =
            link.target.clone().into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?;
        get(hash, GetOptions::default())
    } else {
        Ok(None)
    }
}

/// Get all disputes for a claim
#[hdk_extern]
pub fn get_claim_disputes(claim_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(claim_hash, LinkTypes::ClaimToDispute)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        let hash = link
            .target
            .into_action_hash()
            .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
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
        consciousness: None,
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

    KVector::new_legacy(k_r, k_a, k_i, k_p, k_m, k_s, k_h, k_topo)
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

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // build_kvector_from_activity tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_kvector_no_activity() {
        // No claims, no feedback -> neutral/low values
        let kv = build_kvector_from_activity(0, 0, 0);

        // k_r: no feedback -> neutral 0.5
        assert!(
            (kv.k_r - 0.5).abs() < 0.001,
            "k_r should be 0.5 (neutral), got {}",
            kv.k_r
        );

        // k_a: ln(1)/ln(101) = 0.0
        assert!(
            kv.k_a < 0.01,
            "k_a should be ~0.0 for zero claims, got {}",
            kv.k_a
        );

        // k_i: no feedback -> neutral 0.5
        assert!(
            (kv.k_i - 0.5).abs() < 0.001,
            "k_i should be 0.5 (neutral), got {}",
            kv.k_i
        );

        // k_p: total_feedback <= 5 -> 0.5
        assert!(
            (kv.k_p - 0.5).abs() < 0.001,
            "k_p should be 0.5, got {}",
            kv.k_p
        );

        // k_m: claim_count == 0 -> 0.2
        assert!(
            (kv.k_m - 0.2).abs() < 0.001,
            "k_m should be 0.2, got {}",
            kv.k_m
        );

        // k_s: always 0.5 in DKG
        assert!(
            (kv.k_s - 0.5).abs() < 0.001,
            "k_s should be 0.5, got {}",
            kv.k_s
        );

        // k_h: endorsements=0, challenges=0, neither condition met -> 0.3
        assert!(
            (kv.k_h - 0.3).abs() < 0.001,
            "k_h should be 0.3, got {}",
            kv.k_h
        );

        // k_topo: always 0.5 in DKG
        assert!(
            (kv.k_topo - 0.5).abs() < 0.001,
            "k_topo should be 0.5, got {}",
            kv.k_topo
        );

        // Trust score should be moderate-low (around 0.36)
        let score = kv.trust_score();
        assert!(
            score > 0.2 && score < 0.5,
            "No-activity trust score should be moderate-low, got {}",
            score
        );
    }

    #[test]
    fn test_build_kvector_high_endorsements() {
        // Many claims, high endorsements, low challenges -> high reputation
        let kv = build_kvector_from_activity(50, 100, 5);

        // k_r: 100/105 ~ 0.952
        assert!(
            kv.k_r > 0.9,
            "k_r should be high with strong endorsements, got {}",
            kv.k_r
        );

        // k_a: ln(51)/ln(101) ~ 0.852
        assert!(
            kv.k_a > 0.8,
            "k_a should be high for 50 claims, got {}",
            kv.k_a
        );

        // k_i: 1.0 - (5/105)*2.0 ~ 0.905
        assert!(
            kv.k_i > 0.85,
            "k_i should be high with few challenges, got {}",
            kv.k_i
        );

        // k_p: total_feedback > 5, k_r * 1.1 clamped to 1.0
        assert!(
            (kv.k_p - 1.0).abs() < 0.001,
            "k_p should be 1.0 (clamped), got {}",
            kv.k_p
        );

        // k_m: claim_count > 10 -> 0.8
        assert!(
            (kv.k_m - 0.8).abs() < 0.001,
            "k_m should be 0.8, got {}",
            kv.k_m
        );

        // k_h: endorsements > 10 && challenges < endorsements/5 (5 < 20) -> 0.9
        assert!(
            (kv.k_h - 0.9).abs() < 0.001,
            "k_h should be 0.9, got {}",
            kv.k_h
        );

        // Trust score should be high (Constitutional tier)
        let score = kv.trust_score();
        assert!(
            score >= 0.6,
            "High-endorsement trust score should be Constitutional (>= 0.6), got {}",
            score
        );
    }

    #[test]
    fn test_build_kvector_mixed_feedback() {
        // Equal endorsements and challenges -> moderate reputation
        let kv = build_kvector_from_activity(10, 10, 10);

        // k_r: 10/20 = 0.5
        assert!(
            (kv.k_r - 0.5).abs() < 0.001,
            "k_r should be 0.5 for equal feedback, got {}",
            kv.k_r
        );

        // k_i: 1.0 - (10/20)*2.0 = 0.0
        assert!(
            kv.k_i < 0.01,
            "k_i should be ~0.0 with equal challenges, got {}",
            kv.k_i
        );

        // k_p: total_feedback > 5, k_r * 1.1 = 0.55
        assert!(
            (kv.k_p - 0.55).abs() < 0.01,
            "k_p should be ~0.55, got {}",
            kv.k_p
        );

        // k_m: claim_count=10, 10 > 10 is false, 10 > 0 -> 0.5
        assert!(
            (kv.k_m - 0.5).abs() < 0.001,
            "k_m should be 0.5, got {}",
            kv.k_m
        );

        // k_h: endorsements=10, not > 10. endorsements > challenges: 10 > 10 is false -> 0.3
        assert!(
            (kv.k_h - 0.3).abs() < 0.001,
            "k_h should be 0.3, got {}",
            kv.k_h
        );

        // Trust score should be in Basic range
        let score = kv.trust_score();
        assert!(
            score >= 0.3 && score < 0.4,
            "Mixed-feedback trust score should be Basic tier (0.3-0.4), got {}",
            score
        );
    }

    #[test]
    fn test_build_kvector_all_challenges() {
        // Only challenges, no endorsements -> low reputation
        let kv = build_kvector_from_activity(5, 0, 20);

        // k_r: 0/20 = 0.0
        assert!(
            kv.k_r < 0.01,
            "k_r should be ~0.0 with no endorsements, got {}",
            kv.k_r
        );

        // k_i: 1.0 - (20/20)*2.0 = -1.0 clamped to 0.0
        assert!(kv.k_i < 0.01, "k_i should be 0.0 (clamped), got {}", kv.k_i);

        // k_p: total_feedback > 5, k_r * 1.1 = 0.0
        assert!(kv.k_p < 0.01, "k_p should be ~0.0, got {}", kv.k_p);

        // k_h: endorsements=0 -> falls through to 0.3
        assert!(
            (kv.k_h - 0.3).abs() < 0.001,
            "k_h should be 0.3, got {}",
            kv.k_h
        );

        // Trust score should be low (Observer tier)
        let score = kv.trust_score();
        assert!(
            score < 0.3,
            "All-challenges trust score should be Observer (< 0.3), got {}",
            score
        );
    }

    // -----------------------------------------------------------------------
    // compute_governance_tier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_governance_tier_constitutional() {
        let tier = compute_governance_tier(0.8);
        assert_eq!(
            tier,
            GovernanceTier::Constitutional,
            "Trust score 0.8 should be Constitutional"
        );
    }

    #[test]
    fn test_governance_tier_major() {
        let tier = compute_governance_tier(0.5);
        assert_eq!(
            tier,
            GovernanceTier::Major,
            "Trust score 0.5 should be Major"
        );
    }

    #[test]
    fn test_governance_tier_basic() {
        let tier = compute_governance_tier(0.35);
        assert_eq!(
            tier,
            GovernanceTier::Basic,
            "Trust score 0.35 should be Basic"
        );
    }

    #[test]
    fn test_governance_tier_observer() {
        let tier = compute_governance_tier(0.1);
        assert_eq!(
            tier,
            GovernanceTier::Observer,
            "Trust score 0.1 should be Observer"
        );
    }

    #[test]
    fn test_governance_tier_boundary_values() {
        // Exact boundary: 0.6 -> Constitutional
        assert_eq!(
            compute_governance_tier(0.6),
            GovernanceTier::Constitutional,
            "0.6 is the Constitutional boundary (inclusive)"
        );

        // Exact boundary: 0.4 -> Major
        assert_eq!(
            compute_governance_tier(0.4),
            GovernanceTier::Major,
            "0.4 is the Major boundary (inclusive)"
        );

        // Exact boundary: 0.3 -> Basic
        assert_eq!(
            compute_governance_tier(0.3),
            GovernanceTier::Basic,
            "0.3 is the Basic boundary (inclusive)"
        );

        // Just below each boundary
        assert_eq!(
            compute_governance_tier(0.5999),
            GovernanceTier::Major,
            "Just below 0.6 should be Major"
        );
        assert_eq!(
            compute_governance_tier(0.3999),
            GovernanceTier::Basic,
            "Just below 0.4 should be Basic"
        );
        assert_eq!(
            compute_governance_tier(0.2999),
            GovernanceTier::Observer,
            "Just below 0.3 should be Observer"
        );

        // Zero trust score
        assert_eq!(
            compute_governance_tier(0.0),
            GovernanceTier::Observer,
            "Zero trust score should be Observer"
        );

        // Maximum trust score
        assert_eq!(
            compute_governance_tier(1.0),
            GovernanceTier::Constitutional,
            "Maximum trust score should be Constitutional"
        );
    }

    // -----------------------------------------------------------------------
    // Integration: kvector -> tier
    // -----------------------------------------------------------------------

    #[test]
    fn test_kvector_to_tier_integration() {
        // No activity -> Observer or Basic (moderate-low trust)
        let kv_none = build_kvector_from_activity(0, 0, 0);
        let tier_none = compute_governance_tier(kv_none.trust_score());
        assert!(
            tier_none == GovernanceTier::Basic || tier_none == GovernanceTier::Observer,
            "No-activity agent should be Observer or Basic, got {:?} (score={})",
            tier_none,
            kv_none.trust_score()
        );

        // High endorsements -> Constitutional
        let kv_high = build_kvector_from_activity(50, 100, 5);
        let tier_high = compute_governance_tier(kv_high.trust_score());
        assert_eq!(
            tier_high,
            GovernanceTier::Constitutional,
            "Highly endorsed agent should be Constitutional (score={})",
            kv_high.trust_score()
        );

        // Mixed feedback -> Basic
        let kv_mixed = build_kvector_from_activity(10, 10, 10);
        let tier_mixed = compute_governance_tier(kv_mixed.trust_score());
        assert_eq!(
            tier_mixed,
            GovernanceTier::Basic,
            "Mixed-feedback agent should be Basic (score={})",
            kv_mixed.trust_score()
        );

        // All challenges -> Observer
        let kv_bad = build_kvector_from_activity(5, 0, 20);
        let tier_bad = compute_governance_tier(kv_bad.trust_score());
        assert_eq!(
            tier_bad,
            GovernanceTier::Observer,
            "All-challenges agent should be Observer (score={})",
            kv_bad.trust_score()
        );

        // Moderate good standing -> Major
        let kv_moderate = build_kvector_from_activity(15, 20, 3);
        let tier_moderate = compute_governance_tier(kv_moderate.trust_score());
        assert!(
            tier_moderate == GovernanceTier::Major
                || tier_moderate == GovernanceTier::Constitutional,
            "Moderate good-standing agent should be Major or Constitutional, got {:?} (score={})",
            tier_moderate,
            kv_moderate.trust_score()
        );
    }

    // -----------------------------------------------------------------------
    // Constants verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_consensus_constants() {
        assert_eq!(
            MIN_ENDORSEMENTS, 3,
            "Quorum requires minimum 3 endorsements"
        );
        assert!(
            (MIN_AGGREGATE_TRUST - 2.0).abs() < f64::EPSILON,
            "Quorum requires minimum 2.0 aggregate trust"
        );
        assert!(
            (MAX_CHALLENGE_RATIO - 0.3).abs() < f64::EPSILON,
            "Maximum challenge ratio is 0.3"
        );
        assert_eq!(DECAY_START_DAYS, 90, "Temporal decay starts after 90 days");
        assert!(
            (DECAY_PER_PERIOD - 0.10).abs() < f64::EPSILON,
            "Decay rate is 10% per 30-day period"
        );
    }
}
