// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use mycelix_mail_integrity::*;

// Mycelix SDK imports for MATL integration
use mycelix_sdk::matl::{
    ReputationScore as MATLReputationScore,
    DEFAULT_BYZANTINE_THRESHOLD,
};
use mycelix_sdk::bridge::{
    CrossHappReputation,
    HappReputationScore,
};

/// Input for spam reporting
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpamReportInput {
    pub message_hash: ActionHash,
    pub reason: String,
}

// === Path Helper for HDK 0.6.0 ===
fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.clone().typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

/// Input for querying stored spam reports
#[derive(Serialize, Deserialize, Debug)]
pub struct GetSpamReportsInput {
    pub since: Timestamp,
}

/// Check the trust score for a specific DID
/// This queries the local MATL trust scores stored on the DHT
#[hdk_extern]
pub fn check_sender_trust(did: String) -> ExternResult<f64> {
    debug!("Checking trust for DID: {}", did);

    // Use a Path for DID-based lookup
    let path = Path::from(format!("trust.{}", did));
    let path_hash = ensure_path(path, LinkTypes::TrustByDid)?;

    // Query trust scores linked to this DID path (HDK 0.6.0 API)
    let links = get_links(
        LinkQuery::new(
            path_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::TrustByDid as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    // If we have a trust score, return it
    if let Some(link) = links.first() {
        // Convert the link target to the appropriate hash type
        let hash_any_dht: AnyDhtHash =
            ActionHash::from_raw_39(link.target.get_raw_39().to_vec()).into();
        let record = get(hash_any_dht, GetOptions::default())?;

        if let Some(record) = record {
            let trust_score: TrustScore = record
                .entry()
                .to_app_option()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Deserialization error: {:?}",
                        e
                    )))
                })?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid trust score".into()
                )))?;

            debug!("Found trust score for {}: {}", did, trust_score.score);
            return Ok(trust_score.score);
        }
    }

    // Default: neutral score for new/unknown users (0.5)
    // This allows new users to send mail, but they start with low reputation
    debug!("No trust score found for {}, returning default 0.5", did);
    Ok(0.5)
}

/// Update or create a trust score for a DID
/// This is called by the MATL bridge to sync trust scores from the MATL system
#[hdk_extern]
pub fn update_trust_score(trust_score: TrustScore) -> ExternResult<ActionHash> {
    debug!(
        "Updating trust score for {}: {}",
        trust_score.did, trust_score.score
    );

    // Create the trust score entry
    let score_hash = create_entry(EntryTypes::TrustScore(trust_score.clone()))?;

    // Use a Path for DID-based lookup (HDK 0.6.0 typed paths)
    let path = Path::from(format!("trust.{}", trust_score.did));
    let path_hash = ensure_path(path, LinkTypes::TrustByDid)?;

    // Link from DID path to trust score
    create_link(path_hash, score_hash.clone(), LinkTypes::TrustByDid, ())?;

    // Maintain an index path for enumeration
    let index_path = Path::from("trust_index");
    let index_hash = ensure_path(index_path, LinkTypes::TrustIndex)?;
    create_link(index_hash, score_hash.clone(), LinkTypes::TrustIndex, ())?;

    Ok(score_hash)
}

/// Get filtered inbox messages based on minimum trust threshold
/// This is the KEY SPAM FILTER - uses MATL scores to filter messages
#[hdk_extern]
pub fn filter_inbox(min_trust: f64) -> ExternResult<Vec<MailMessage>> {
    debug!("Filtering inbox with min_trust: {}", min_trust);

    // Call the mail_messages zome to get all inbox messages
    let response: ZomeCallResponse = call(
        CallTargetCell::Local,
        "mail_messages",
        "get_inbox".into(),
        None,
        (),
    )?;

    // Decode the response
    let all_messages: Vec<MailMessage> = match response {
        ZomeCallResponse::Ok(result) => decode(&result.into_vec()).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to decode response: {:?}",
                e
            )))
        })?,
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Zome call failed".into()
            )))
        }
    };

    debug!("Got {} total messages", all_messages.len());

    // Filter by trust score
    let mut trusted_messages = Vec::new();

    for message in all_messages {
        // Check sender's trust score
        let trust_score = check_sender_trust(message.from_did.clone())?;

        if trust_score >= min_trust {
            trusted_messages.push(message);
        } else {
            debug!(
                "Filtered out message from {} (trust: {} < {})",
                message.from_did, trust_score, min_trust
            );
        }
    }

    debug!("Returning {} trusted messages", trusted_messages.len());
    Ok(trusted_messages)
}

/// Get all trust scores (for admin/debugging)
#[hdk_extern]
pub fn get_all_trust_scores(_: ()) -> ExternResult<Vec<TrustScore>> {
    let index_path = Path::from("trust_index");
    let index_hash = match ensure_path(index_path, LinkTypes::TrustIndex) {
        Ok(hash) => hash,
        Err(_) => return Ok(Vec::new()),
    };

    let links = get_links(
        LinkQuery::new(
            index_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::TrustIndex as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut scores = Vec::new();
    for link in links {
        let hash_any_dht: AnyDhtHash =
            ActionHash::from_raw_39(link.target.get_raw_39().to_vec()).into();
        if let Some(record) = get(hash_any_dht, GetOptions::default())? {
            if let Some(score) = record.entry().to_app_option::<TrustScore>().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Deserialization error: {:?}",
                    e
                )))
            })? {
                scores.push(score);
            }
        }
    }

    Ok(scores)
}

/// Report spam/malicious message
/// This creates a negative report that feeds back into MATL
#[hdk_extern]
pub fn report_spam(input: SpamReportInput) -> ExternResult<()> {
    debug!(
        "Spam reported for message {:?}: {}",
        input.message_hash, input.reason
    );

    // Clone message_hash before move
    let message_hash = input.message_hash.clone();

    // Get the message to identify the sender
    let record = get(input.message_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Message not found".into())
    ))?;

    let message: MailMessage = record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid message entry".into()
        )))?;

    let reporter = agent_info()?.agent_initial_pubkey;
    let report = SpamReport {
        reporter,
        spammer_did: message.from_did.clone(),
        message_hash,
        reason: input.reason,
        reported_at: sys_time()?,
    };

    let report_hash = create_entry(EntryTypes::SpamReport(report))?;

    let path = Path::from("spam_reports");
    let path_hash = ensure_path(path, LinkTypes::SpamReports)?;
    create_link(path_hash, report_hash, LinkTypes::SpamReports, ())?;

    debug!("Spam report registered for sender: {}", message.from_did);
    Ok(())
}

/// Retrieve spam reports recorded since a given timestamp
#[hdk_extern]
pub fn get_spam_reports(input: GetSpamReportsInput) -> ExternResult<Vec<SpamReport>> {
    let path = Path::from("spam_reports");
    let path_hash = match ensure_path(path, LinkTypes::SpamReports) {
        Ok(hash) => hash,
        Err(_) => return Ok(Vec::new()),
    };

    let links = get_links(
        LinkQuery::new(
            path_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::SpamReports as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut reports = Vec::new();
    for link in links {
        let hash_any_dht: AnyDhtHash =
            ActionHash::from_raw_39(link.target.get_raw_39().to_vec()).into();
        if let Some(record) = get(hash_any_dht, GetOptions::default())? {
            if let Some(report) = record.entry().to_app_option::<SpamReport>().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Deserialization error: {:?}",
                    e
                )))
            })? {
                if report.reported_at > input.since {
                    reports.push(report);
                }
            }
        }
    }

    reports.sort_by(|a, b| a.reported_at.cmp(&b.reported_at));
    Ok(reports)
}

// =============================================================================
// MATL SDK Integration - Enhanced Trust Scoring
// =============================================================================

/// Input for updating trust with MATL
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MATLTrustUpdateInput {
    pub did: String,
    pub is_positive: bool,
    pub source: String,
}

/// Result of MATL trust evaluation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MATLTrustResult {
    pub did: String,
    pub score: f64,
    pub is_byzantine: bool,
    pub total_interactions: u64,
    pub is_new_user: bool,
}

/// Update trust using MATL SDK's Bayesian reputation scoring
///
/// This uses the SDK's ReputationScore for proper Bayesian smoothing
#[hdk_extern]
pub fn update_trust_with_matl(input: MATLTrustUpdateInput) -> ExternResult<ActionHash> {
    debug!("MATL trust update for {}: positive={}", input.did, input.is_positive);

    // Get existing trust score to preserve counts (for future use when we track counts)
    let _existing_score = check_sender_trust(input.did.clone())?;

    // Create SDK reputation tracker
    let mut matl_rep = MATLReputationScore::new(&input.did, &input.source);

    // If we have existing data, we need to reconstruct the counts
    // For now, we'll use the current score and add the new interaction
    if input.is_positive {
        matl_rep.record_positive();
    } else {
        matl_rep.record_negative();
    }

    // Get the MATL-computed score (with Bayesian smoothing)
    let new_score = matl_rep.score;

    // Create the trust score entry
    let trust_score = TrustScore {
        did: input.did.clone(),
        score: new_score,
        source: input.source.clone(),
        last_updated: sys_time()?,
    };

    // Use update_trust_score to store it
    update_trust_score(trust_score)
}

/// Get comprehensive MATL trust evaluation
///
/// Returns detailed trust information including Byzantine detection
#[hdk_extern]
pub fn evaluate_trust_matl(did: String) -> ExternResult<MATLTrustResult> {
    let score = check_sender_trust(did.clone())?;

    // Check for Byzantine behavior (score below threshold)
    let is_byzantine = score < DEFAULT_BYZANTINE_THRESHOLD;

    // Get spam reports to estimate interactions
    let since = Timestamp::from_micros(0);
    let spam_reports = get_spam_reports(GetSpamReportsInput { since })?;
    let negative_count = spam_reports.iter()
        .filter(|r| r.spammer_did == did)
        .count() as u64;

    // Estimate total interactions (rough heuristic)
    // In production, would track actual message counts
    let estimated_total = if negative_count > 0 {
        // If we have spam reports, estimate based on score
        ((negative_count as f64) / (1.0 - score.max(0.01))) as u64
    } else {
        0
    };

    let is_new_user = estimated_total < 10;

    Ok(MATLTrustResult {
        did,
        score,
        is_byzantine,
        total_interactions: estimated_total + negative_count,
        is_new_user,
    })
}

/// Filter inbox with MATL-enhanced Byzantine detection
///
/// Filters messages and also returns Byzantine senders for blocking
#[hdk_extern]
pub fn filter_inbox_matl(min_trust: f64) -> ExternResult<FilteredInboxResult> {
    let all_messages = filter_inbox(min_trust)?;

    // Analyze each sender for Byzantine behavior
    let mut trusted_messages = Vec::new();
    let mut byzantine_senders = Vec::new();

    // Collect unique senders
    let mut seen_senders: std::collections::HashSet<String> = std::collections::HashSet::new();

    for message in all_messages {
        let trust_result = evaluate_trust_matl(message.from_did.clone())?;

        if trust_result.is_byzantine && !seen_senders.contains(&message.from_did) {
            byzantine_senders.push(message.from_did.clone());
            seen_senders.insert(message.from_did.clone());
        }

        if !trust_result.is_byzantine {
            trusted_messages.push(message);
        }
    }

    Ok(FilteredInboxResult {
        messages: trusted_messages,
        byzantine_senders,
        filter_threshold: min_trust,
    })
}

/// Result of MATL-filtered inbox
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FilteredInboxResult {
    pub messages: Vec<MailMessage>,
    pub byzantine_senders: Vec<String>,
    pub filter_threshold: f64,
}

/// Record a positive interaction (e.g., reply, mark as important)
#[hdk_extern]
pub fn record_positive_interaction(did: String) -> ExternResult<ActionHash> {
    update_trust_with_matl(MATLTrustUpdateInput {
        did,
        is_positive: true,
        source: "mail".to_string(),
    })
}

/// Record a negative interaction (wrapper for report_spam with MATL update)
#[hdk_extern]
pub fn record_negative_interaction(input: SpamReportInput) -> ExternResult<()> {
    // First record the spam report
    report_spam(input.clone())?;

    // Then update MATL trust score
    // Get the message to find the sender
    let record = get(input.message_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Message not found".into())
    ))?;

    let message: MailMessage = record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid message entry".into()
        )))?;

    update_trust_with_matl(MATLTrustUpdateInput {
        did: message.from_did,
        is_positive: false,
        source: "mail".to_string(),
    })?;

    Ok(())
}

/// Get cross-hApp reputation for a DID
///
/// Creates a Bridge message for querying reputation across hApps
#[hdk_extern]
pub fn get_cross_happ_reputation(did: String) -> ExternResult<CrossHappReputationResult> {
    // Get local mail reputation
    let local_score = check_sender_trust(did.clone())?;

    // Create HappReputationScore for mail
    let mail_score = HappReputationScore {
        happ_id: "mail".to_string(),
        happ_name: "Mycelix Mail".to_string(),
        score: local_score,
        interactions: 0, // Would track actual interactions in production
        last_updated: sys_time()?.0 as u64 / 1_000_000,
    };

    // Create cross-hApp reputation using SDK's from_scores
    // In production, this would include scores from other hApps via bridge queries
    let cross_happ = CrossHappReputation::from_scores(did.clone(), vec![mail_score]);

    // Confidence: 0 for unknown users (0.5 score), higher for known users
    let confidence = if local_score == 0.5 { 0.0 } else { 0.8 };

    Ok(CrossHappReputationResult {
        did,
        local_score,
        cross_happ_score: cross_happ.aggregate,
        happ_count: cross_happ.scores.len(),
        confidence,
    })
}

/// Result of cross-hApp reputation query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrossHappReputationResult {
    pub did: String,
    pub local_score: f64,
    pub cross_happ_score: f64,
    pub happ_count: usize,
    pub confidence: f64,
}

// === Helper Functions ===

/// Calculate a simple local trust score based on message history
/// This is a fallback when MATL scores are not available
#[allow(dead_code)]
fn calculate_local_trust(did: &str) -> ExternResult<f64> {
    // In production, this would:
    // 1. Count total messages sent by this DID
    // 2. Count spam reports
    // 3. Calculate score = 1.0 - (spam_reports / total_messages)

    // For MVP, return neutral
    debug!("Calculating local trust for {} (MVP: returning 0.5)", did);
    Ok(0.5)
}
