// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Bridge Coordinator Zome - Cross-hApp verification and MATL integration
//!
//! This zome provides the coordination logic for cross-hApp queries,
//! reputation management, and verification records in the mutual aid network.

use bridge_integrity::{
    AidQuery, AidResult, Anchor as BridgeAnchor, BridgeConfig, EntryTypes, LinkTypes,
    MatlReputation, QueryPurpose, QueryStatus, VerificationRecord, VerificationResult,
    VerificationType,
};
use hdk::prelude::*;

/// Input for creating a cross-hApp query
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateQueryInput {
    pub subject_did: String,
    pub requester_did: String,
    pub purpose: QueryPurpose,
    pub source_happ: String,
    pub target_happ: String,
    pub parameters: Option<String>,
    pub ttl_seconds: u64,
}

/// Input for submitting a query result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmitResultInput {
    pub query_hash: ActionHash,
    pub query_id: String,
    pub success: bool,
    pub data: Option<String>,
    pub error: Option<String>,
    pub provider_trust_score: f64,
}

/// Input for updating MATL reputation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateReputationInput {
    pub member_did: String,
    pub contribution_score: Option<f64>,
    pub fulfillment_score: Option<f64>,
    pub engagement_score: Option<f64>,
    pub pools_count: Option<u32>,
    pub total_contributed: Option<u64>,
    pub total_received: Option<u64>,
    pub successful_offers: Option<u32>,
    pub requests_made: Option<u32>,
}

/// Input for creating a verification record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateVerificationInput {
    pub subject_did: String,
    pub verification_type: VerificationType,
    pub verifier_happ: String,
    pub result: VerificationResult,
    pub evidence: Option<String>,
    pub valid_for_seconds: u64,
}

/// Output containing a query with its action hash
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryWithHash {
    pub hash: ActionHash,
    pub query: AidQuery,
}

/// Output containing a result with its action hash
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResultWithHash {
    pub hash: ActionHash,
    pub result: AidResult,
}

/// Output containing a reputation with its action hash
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReputationWithHash {
    pub hash: ActionHash,
    pub reputation: MatlReputation,
}

/// Output containing a verification with its action hash
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationWithHash {
    pub hash: ActionHash,
    pub verification: VerificationRecord,
}

/// Input for checking verification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckVerificationInput {
    pub member_did: String,
    pub verification_type: VerificationType,
}

/// Input for checking trust threshold
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrustThresholdInput {
    pub member_did: String,
    pub threshold: f64,
}

/// Generate a unique ID based on timestamp and agent
fn generate_id(prefix: &str) -> ExternResult<String> {
    let now = sys_time()?;
    let agent = agent_info()?.agent_initial_pubkey;
    // Create unique ID from timestamp and agent pubkey truncated hash
    let agent_str = format!("{:?}", agent);
    let short_hash = &agent_str[agent_str.len().saturating_sub(8)..];
    Ok(format!(
        "{}_{:x}_{}",
        prefix,
        now.as_micros() as u64,
        short_hash
    ))
}

/// Get or create the main queries anchor
fn get_queries_anchor() -> ExternResult<EntryHash> {
    let anchor = BridgeAnchor::new("all_queries");
    let entry_hash = hash_entry(&anchor)?;
    create_entry(EntryTypes::Anchor(anchor))?;
    Ok(entry_hash)
}

/// Get or create the pending queries anchor
fn get_pending_anchor() -> ExternResult<EntryHash> {
    let anchor = BridgeAnchor::new("pending_queries");
    let entry_hash = hash_entry(&anchor)?;
    create_entry(EntryTypes::Anchor(anchor))?;
    Ok(entry_hash)
}

/// Get or create a member-specific anchor
fn get_member_anchor(did: &str) -> ExternResult<EntryHash> {
    let anchor = BridgeAnchor::new(format!("member_{}", did));
    let entry_hash = hash_entry(&anchor)?;
    create_entry(EntryTypes::Anchor(anchor))?;
    Ok(entry_hash)
}

/// Get or create a source hApp anchor
fn get_source_happ_anchor(happ: &str) -> ExternResult<EntryHash> {
    let anchor = BridgeAnchor::new(format!("source_happ_{}", happ));
    let entry_hash = hash_entry(&anchor)?;
    create_entry(EntryTypes::Anchor(anchor))?;
    Ok(entry_hash)
}

/// Get or create a target hApp anchor
fn get_target_happ_anchor(happ: &str) -> ExternResult<EntryHash> {
    let anchor = BridgeAnchor::new(format!("target_happ_{}", happ));
    let entry_hash = hash_entry(&anchor)?;
    create_entry(EntryTypes::Anchor(anchor))?;
    Ok(entry_hash)
}

/// Create a cross-hApp query
#[hdk_extern]
pub fn create_query(input: CreateQueryInput) -> ExternResult<QueryWithHash> {
    let now = sys_time()?;
    let id = generate_id("query")?;

    // Calculate expiration
    let expires_at =
        Timestamp::from_micros(now.as_micros() + (input.ttl_seconds as i64 * 1_000_000));

    let query = AidQuery {
        id: id.clone(),
        subject_did: input.subject_did,
        requester_did: input.requester_did,
        purpose: input.purpose,
        source_happ: input.source_happ.clone(),
        target_happ: input.target_happ.clone(),
        parameters: input.parameters,
        status: QueryStatus::Pending,
        created_at: now,
        expires_at,
    };

    // Create the entry
    let action_hash = create_entry(EntryTypes::AidQuery(query.clone()))?;

    // Link from main anchor
    let queries_anchor = get_queries_anchor()?;
    create_link(
        queries_anchor,
        action_hash.clone(),
        LinkTypes::AnchorToQuery,
        id.as_bytes().to_vec(),
    )?;

    // Link from pending anchor
    let pending_anchor = get_pending_anchor()?;
    create_link(
        pending_anchor,
        action_hash.clone(),
        LinkTypes::AnchorToPendingQuery,
        (),
    )?;

    // Link from source hApp
    let source_anchor = get_source_happ_anchor(&input.source_happ)?;
    create_link(
        source_anchor,
        action_hash.clone(),
        LinkTypes::SourceHappToQuery,
        (),
    )?;

    // Link from target hApp
    let target_anchor = get_target_happ_anchor(&input.target_happ)?;
    create_link(
        target_anchor,
        action_hash.clone(),
        LinkTypes::TargetHappToQuery,
        (),
    )?;

    Ok(QueryWithHash {
        hash: action_hash,
        query,
    })
}

/// Get a query by its action hash
#[hdk_extern]
pub fn get_query(action_hash: ActionHash) -> ExternResult<Option<QueryWithHash>> {
    match get(action_hash.clone(), GetOptions::default())? {
        Some(record) => {
            let query: AidQuery = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest("No AidQuery found".to_string()))
                })?;
            Ok(Some(QueryWithHash {
                hash: action_hash,
                query,
            }))
        }
        None => Ok(None),
    }
}

/// Submit a result for a query
#[hdk_extern]
pub fn submit_result(input: SubmitResultInput) -> ExternResult<ResultWithHash> {
    let now = sys_time()?;

    // Update query status
    let record = get(input.query_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Query not found".to_string())))?;

    let mut query: AidQuery = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No AidQuery found".to_string())))?;

    // Check if query is expired
    if now > query.expires_at {
        query.status = QueryStatus::Expired;
        update_entry(input.query_hash.clone(), EntryTypes::AidQuery(query))?;
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Query has expired".to_string()
        )));
    }

    // Update status
    query.status = if input.success {
        QueryStatus::Completed
    } else {
        QueryStatus::Failed
    };
    let new_query_hash = update_entry(input.query_hash.clone(), EntryTypes::AidQuery(query))?;

    // Remove from pending
    let pending_anchor = get_pending_anchor()?;
    let pending_links = get_links(
        LinkQuery::try_new(pending_anchor, LinkTypes::AnchorToPendingQuery)?,
        GetStrategy::default(),
    )?;
    for link in pending_links {
        if link.target == input.query_hash.clone().into() {
            delete_link(link.create_link_hash, GetOptions::default())?;
        }
    }

    // Create result
    let result = AidResult {
        query_id: input.query_id,
        success: input.success,
        data: input.data,
        error: input.error,
        provider_trust_score: input.provider_trust_score,
        created_at: now,
    };
    let result_hash = create_entry(EntryTypes::AidResult(result.clone()))?;

    // Link query to result
    create_link(
        new_query_hash,
        result_hash.clone(),
        LinkTypes::QueryToResult,
        (),
    )?;

    Ok(ResultWithHash {
        hash: result_hash,
        result,
    })
}

/// Get result for a query
#[hdk_extern]
pub fn get_query_result(query_hash: ActionHash) -> ExternResult<Option<ResultWithHash>> {
    let links = get_links(
        LinkQuery::try_new(query_hash, LinkTypes::QueryToResult)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Ok(Some(result)) = record.entry().to_app_option::<AidResult>() {
                    return Ok(Some(ResultWithHash {
                        hash: action_hash,
                        result,
                    }));
                }
            }
        }
    }

    Ok(None)
}

/// Get all pending queries
#[hdk_extern]
pub fn get_pending_queries(_: ()) -> ExternResult<Vec<QueryWithHash>> {
    let pending_anchor = get_pending_anchor()?;
    let links = get_links(
        LinkQuery::try_new(pending_anchor, LinkTypes::AnchorToPendingQuery)?,
        GetStrategy::default(),
    )?;

    let mut queries = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(query_with_hash) = get_query(action_hash)? {
                // Only include non-expired queries
                let now = sys_time()?;
                if now <= query_with_hash.query.expires_at {
                    queries.push(query_with_hash);
                }
            }
        }
    }

    Ok(queries)
}

/// Get queries for a target hApp
#[hdk_extern]
pub fn get_queries_for_happ(happ: String) -> ExternResult<Vec<QueryWithHash>> {
    let target_anchor = get_target_happ_anchor(&happ)?;
    let links = get_links(
        LinkQuery::try_new(target_anchor, LinkTypes::TargetHappToQuery)?,
        GetStrategy::default(),
    )?;

    let mut queries = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(query_with_hash) = get_query(action_hash)? {
                queries.push(query_with_hash);
            }
        }
    }

    Ok(queries)
}

/// Calculate and update MATL reputation for a member
#[hdk_extern]
pub fn update_reputation(input: UpdateReputationInput) -> ExternResult<ReputationWithHash> {
    let now = sys_time()?;

    // Try to get existing reputation
    let member_anchor = get_member_anchor(&input.member_did)?;
    let links = get_links(
        LinkQuery::try_new(member_anchor.clone(), LinkTypes::MemberToReputation)?,
        GetStrategy::default(),
    )?;

    let (existing_hash, mut reputation) = if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Ok(Some(rep)) = record.entry().to_app_option::<MatlReputation>() {
                    (Some(action_hash), rep)
                } else {
                    (None, create_default_reputation(&input.member_did, now))
                }
            } else {
                (None, create_default_reputation(&input.member_did, now))
            }
        } else {
            (None, create_default_reputation(&input.member_did, now))
        }
    } else {
        (None, create_default_reputation(&input.member_did, now))
    };

    // Apply updates
    if let Some(score) = input.contribution_score {
        reputation.contribution_score = score;
    }
    if let Some(score) = input.fulfillment_score {
        reputation.fulfillment_score = score;
    }
    if let Some(score) = input.engagement_score {
        reputation.engagement_score = score;
    }
    if let Some(count) = input.pools_count {
        reputation.pools_count = count;
    }
    if let Some(amount) = input.total_contributed {
        reputation.total_contributed = amount;
    }
    if let Some(amount) = input.total_received {
        reputation.total_received = amount;
    }
    if let Some(count) = input.successful_offers {
        reputation.successful_offers = count;
    }
    if let Some(count) = input.requests_made {
        reputation.requests_made = count;
    }

    // Recalculate overall trust score
    reputation.trust_score = calculate_trust_score(&reputation);
    reputation.last_activity = now;
    reputation.calculated_at = now;

    // Create or update entry
    let hash = if let Some(old_hash) = existing_hash {
        update_entry(old_hash, EntryTypes::MatlReputation(reputation.clone()))?
    } else {
        let new_hash = create_entry(EntryTypes::MatlReputation(reputation.clone()))?;

        // Link member to reputation
        create_link(
            member_anchor,
            new_hash.clone(),
            LinkTypes::MemberToReputation,
            (),
        )?;

        new_hash
    };

    Ok(ReputationWithHash { hash, reputation })
}

/// Create a default reputation for a new member
fn create_default_reputation(member_did: &str, now: Timestamp) -> MatlReputation {
    MatlReputation {
        member_did: member_did.to_string(),
        trust_score: 0.5, // Neutral starting score
        contribution_score: 0.5,
        fulfillment_score: 0.5,
        engagement_score: 0.5,
        pools_count: 0,
        total_contributed: 0,
        total_received: 0,
        successful_offers: 0,
        requests_made: 0,
        last_activity: now,
        calculated_at: now,
    }
}

/// Calculate overall trust score from component scores
fn calculate_trust_score(reputation: &MatlReputation) -> f64 {
    // Weighted average of component scores
    let contribution_weight = 0.35;
    let fulfillment_weight = 0.35;
    let engagement_weight = 0.20;
    let activity_weight = 0.10;

    // Activity bonus based on participation
    let activity_score = if reputation.pools_count > 0 {
        let participation =
            (reputation.successful_offers as f64 + reputation.requests_made as f64) / 10.0;
        (participation).min(1.0)
    } else {
        0.0
    };

    let weighted = (reputation.contribution_score * contribution_weight)
        + (reputation.fulfillment_score * fulfillment_weight)
        + (reputation.engagement_score * engagement_weight)
        + (activity_score * activity_weight);

    // Clamp to valid range
    weighted.max(0.0).min(1.0)
}

/// Get reputation for a member
#[hdk_extern]
pub fn get_reputation(member_did: String) -> ExternResult<Option<ReputationWithHash>> {
    let member_anchor = get_member_anchor(&member_did)?;
    let links = get_links(
        LinkQuery::try_new(member_anchor, LinkTypes::MemberToReputation)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Ok(Some(reputation)) = record.entry().to_app_option::<MatlReputation>() {
                    return Ok(Some(ReputationWithHash {
                        hash: action_hash,
                        reputation,
                    }));
                }
            }
        }
    }

    Ok(None)
}

/// Create a verification record
#[hdk_extern]
pub fn create_verification(input: CreateVerificationInput) -> ExternResult<VerificationWithHash> {
    let now = sys_time()?;
    let id = generate_id("verif")?;

    let verification = VerificationRecord {
        id,
        subject_did: input.subject_did.clone(),
        verification_type: input.verification_type,
        verifier_happ: input.verifier_happ,
        result: input.result,
        evidence: input.evidence,
        verified_at: now,
        valid_for_seconds: input.valid_for_seconds,
    };

    let hash = create_entry(EntryTypes::VerificationRecord(verification.clone()))?;

    // Link member to verification
    let member_anchor = get_member_anchor(&input.subject_did)?;
    create_link(
        member_anchor,
        hash.clone(),
        LinkTypes::MemberToVerification,
        (),
    )?;

    Ok(VerificationWithHash { hash, verification })
}

/// Get verifications for a member
#[hdk_extern]
pub fn get_verifications(member_did: String) -> ExternResult<Vec<VerificationWithHash>> {
    let member_anchor = get_member_anchor(&member_did)?;
    let links = get_links(
        LinkQuery::try_new(member_anchor, LinkTypes::MemberToVerification)?,
        GetStrategy::default(),
    )?;

    let now = sys_time()?;
    let mut verifications = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Ok(Some(verification)) = record.entry().to_app_option::<VerificationRecord>()
                {
                    // Check if still valid
                    let expires_micros = verification.verified_at.as_micros()
                        + (verification.valid_for_seconds as i64 * 1_000_000);
                    if now.as_micros() <= expires_micros {
                        verifications.push(VerificationWithHash {
                            hash: action_hash,
                            verification,
                        });
                    }
                }
            }
        }
    }

    Ok(verifications)
}

/// Quick verification check for a specific type
#[hdk_extern]
pub fn check_verification(
    input: CheckVerificationInput,
) -> ExternResult<Option<VerificationWithHash>> {
    let verifications = get_verifications(input.member_did)?;

    for v in verifications {
        if v.verification.verification_type == input.verification_type
            && v.verification.result == VerificationResult::Verified
        {
            return Ok(Some(v));
        }
    }

    Ok(None)
}

/// Query member verification for cross-hApp use
#[hdk_extern]
pub fn query_member_verification(input: CreateQueryInput) -> ExternResult<QueryWithHash> {
    // Create the query
    let query = create_query(input)?;

    // In a real implementation, this would trigger cross-hApp communication
    // For now, we just create the query and let the target hApp respond

    Ok(query)
}

/// Get trust score for a member (convenience function)
#[hdk_extern]
pub fn get_trust_score(member_did: String) -> ExternResult<f64> {
    match get_reputation(member_did)? {
        Some(rep) => Ok(rep.reputation.trust_score),
        None => Ok(0.5), // Neutral score for unknown members
    }
}

/// Check if a member meets a trust threshold
#[hdk_extern]
pub fn meets_trust_threshold(input: TrustThresholdInput) -> ExternResult<bool> {
    let score = get_trust_score(input.member_did)?;
    Ok(score >= input.threshold)
}
