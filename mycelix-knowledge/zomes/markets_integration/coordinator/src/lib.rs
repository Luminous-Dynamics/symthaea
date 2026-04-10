// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Markets Integration Coordinator Zome
//!
//! Implements bidirectional integration between Knowledge and Epistemic Markets:
//!
//! ## Knowledge → Markets
//! - `request_verification_market`: Spawn a market to verify a claim's epistemic level
//! - `calculate_market_value`: Assess if creating a market is worthwhile
//!
//! ## Markets → Knowledge
//! - `on_market_created`: Handle market creation callback
//! - `on_verification_market_resolved`: Update claim when market resolves
//! - `register_claim_in_prediction`: Track claim usage as evidence
//!
//! ## Queries
//! - `get_claim_markets`: Get all markets related to a claim
//! - `get_market_claims`: Get all claims referenced by a market
//! - `get_pending_verifications`: Get claims awaiting verification
//!
//! Updated to use HDK 0.6 patterns (LinkQuery, GetStrategy, Anchor pattern)

use hdk::prelude::*;
use markets_integration_integrity::*;

/// Helper to get an anchor entry hash for link bases
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

// ============================================================================
// INPUT/OUTPUT TYPES
// ============================================================================

/// Input for requesting a verification market
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RequestVerificationMarketInput {
    /// The claim to verify
    pub claim_id: String,

    /// Target epistemic position
    pub target_epistemic: EpistemicTarget,

    /// Bounty for market participation
    pub bounty: u64,

    /// Deadline in days from now
    pub deadline_days: u32,

    /// Supporting evidence URIs
    pub supporting_evidence: Vec<String>,

    /// Counter-evidence to challenge
    pub counter_evidence: Vec<String>,
}

/// Input when market is created
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MarketCreatedInput {
    /// The original request ID
    pub request_id: String,

    /// The created market ID
    pub market_id: String,

    /// The market question
    pub question: String,
}

/// Input when market resolves
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MarketResolvedInput {
    /// The market that resolved
    pub market_id: String,

    /// The claim being verified
    pub claim_id: String,

    /// Resolution outcome ("Yes", "No", etc.)
    pub outcome: String,

    /// Confidence level (0.0-1.0)
    pub confidence: f64,

    /// Number of oracles that voted
    pub oracle_count: u32,

    /// MATL-weighted confidence
    pub matl_weighted_confidence: f64,

    /// Total stake in the market
    pub total_stake: u64,
}

/// Input for registering claim in prediction
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RegisterClaimInput {
    /// The claim being referenced
    pub claim_id: String,

    /// The market containing the prediction
    pub market_id: String,

    /// The specific prediction
    pub prediction_id: String,

    /// How the claim is used
    pub usage: ClaimUsageType,

    /// Relevance score (0.0-1.0)
    pub relevance_score: f64,
}

/// Summary of a market related to a claim
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MarketSummary {
    /// Market identifier
    pub market_id: String,

    /// Market question
    pub question: String,

    /// Relationship type
    pub relationship: MarketRelationship,

    /// Current status
    pub status: String,

    /// Current probability (if active)
    pub current_probability: Option<f64>,

    /// Resolution outcome (if resolved)
    pub resolution_outcome: Option<String>,
}

/// Type of relationship between claim and market
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MarketRelationship {
    /// Market was created to verify this claim
    VerificationMarket,

    /// Claim is used as evidence in predictions
    ClaimUsedAsEvidence,
}

/// Result of cascade update after verification
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CascadeUpdateResult {
    /// Claims that were updated
    pub updated_claims: Vec<String>,

    /// Claims that triggered new verification requests
    pub triggered_verifications: Vec<String>,

    /// Errors encountered
    pub errors: Vec<String>,
}

// ============================================================================
// VERIFICATION MARKET REQUEST (Knowledge → Markets)
// ============================================================================

/// Request a verification market for a claim
///
/// This initiates the process of creating an Epistemic Market to verify
/// whether a claim can achieve a higher epistemic level.
#[hdk_extern]
pub fn request_verification_market(
    input: RequestVerificationMarketInput,
) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Calculate deadline
    let deadline_micros = input.deadline_days as i64 * 24 * 60 * 60 * 1_000_000;
    let deadline = Timestamp::from_micros(now.as_micros() + deadline_micros);

    // Generate question based on target
    let question = generate_verification_question(&input.claim_id, &input.target_epistemic);

    // Create the request entry
    let request = VerificationMarketRequest {
        id: format!("vmr_{}_{}", input.claim_id, now.as_micros()),
        claim_id: input.claim_id.clone(),
        question,
        current_epistemic: EpistemicPosition {
            empirical: 0.5, // Will be fetched from claims zome
            normative: 0.5,
            mythic: 0.5,
        },
        target_epistemic: input.target_epistemic.clone(),
        bounty: input.bounty,
        deadline,
        requester: agent.clone(),
        status: MarketRequestStatus::Pending,
        supporting_evidence: input.supporting_evidence,
        counter_evidence: input.counter_evidence,
        created_at: now,
        updated_at: now,
    };

    // Create the entry
    let action_hash = create_entry(EntryTypes::VerificationMarketRequest(request.clone()))?;

    // Create links for indexing using anchor pattern
    let claim_anchor = format!("claim:{}", input.claim_id);
    create_entry(&EntryTypes::Anchor(Anchor(claim_anchor.clone())))?;
    create_link(
        anchor_hash(&claim_anchor)?,
        action_hash.clone(),
        LinkTypes::ClaimToMarketRequest,
        (),
    )?;

    create_link(agent, action_hash.clone(), LinkTypes::AgentToRequest, ())?;

    // Send bridge event to Epistemic Markets via signal
    let event = MarketsIntegrationEvent::VerificationRequested {
        request_id: request.id.clone(),
        claim_id: input.claim_id.clone(),
        question: request.question.clone(),
        target_epistemic: input.target_epistemic.clone(),
        bounty: input.bounty,
        deadline,
    };

    emit_signal(serde_json::to_value(&event).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to serialize event: {}",
            e
        )))
    })?)?;

    debug!(
        "Emitted VerificationRequested event for claim {}",
        input.claim_id
    );

    Ok(action_hash)
}

/// Generate a verification question from the target epistemic level
fn generate_verification_question(claim_id: &str, target: &EpistemicTarget) -> String {
    let target_level = if let Some(e) = target.empirical {
        if e >= 0.9 {
            "E4 (Measurable/Reproducible)"
        } else if e >= 0.7 {
            "E3 (Cryptographic)"
        } else if e >= 0.5 {
            "E2 (Private Verify)"
        } else {
            "E1 (Testimonial)"
        }
    } else {
        "higher epistemic level"
    };

    format!(
        "Will claim {} achieve {} verification within the deadline?",
        claim_id, target_level
    )
}

/// Get pending verification requests
#[hdk_extern]
pub fn get_pending_verifications(_: ()) -> ExternResult<Vec<Record>> {
    // In a full implementation, we'd query by status
    // For now, return empty vec
    Ok(vec![])
}

// ============================================================================
// MARKET CALLBACKS (Markets → Knowledge)
// ============================================================================

/// Handle market creation callback from Epistemic Markets
///
/// Updates the request status and creates bidirectional links.
#[hdk_extern]
pub fn on_market_created(input: MarketCreatedInput) -> ExternResult<ActionHash> {
    let now = sys_time()?;

    // Find the original request by ID
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::VerificationMarketRequest,
        )?))
        .include_entries(true);

    let records = query(filter)?;

    let mut found_record: Option<Record> = None;
    for record in records {
        if let Some(request) = record
            .entry()
            .to_app_option::<VerificationMarketRequest>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if request.id == input.request_id {
                found_record = Some(record);
                break;
            }
        }
    }

    let original_record = found_record.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Request not found: {}",
        input.request_id
    ))))?;

    let original_request: VerificationMarketRequest = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid request entry".into()
        )))?;

    // Update request status to MarketCreated
    let updated_request = VerificationMarketRequest {
        id: original_request.id,
        claim_id: original_request.claim_id,
        question: original_request.question,
        current_epistemic: original_request.current_epistemic,
        target_epistemic: original_request.target_epistemic,
        bounty: original_request.bounty,
        deadline: original_request.deadline,
        requester: original_request.requester,
        status: MarketRequestStatus::MarketCreated {
            market_id: input.market_id.clone(),
            created_at: now,
        },
        supporting_evidence: original_request.supporting_evidence,
        counter_evidence: original_request.counter_evidence,
        created_at: original_request.created_at,
        updated_at: now,
    };

    let updated_hash = update_entry(
        original_record.action_address().clone(),
        &EntryTypes::VerificationMarketRequest(updated_request),
    )?;

    // Create link from request to market using anchor pattern
    let market_anchor = format!("market:{}", input.market_id);
    create_entry(&EntryTypes::Anchor(Anchor(market_anchor.clone())))?;
    create_link(
        updated_hash.clone(),
        anchor_hash(&market_anchor)?,
        LinkTypes::RequestToMarket,
        (),
    )?;

    debug!(
        "Market created: {} for request {}",
        input.market_id, input.request_id
    );

    // Emit event for other listeners
    let event = MarketsIntegrationEvent::MarketCreated {
        request_id: input.request_id,
        market_id: input.market_id,
        question: input.question,
    };

    emit_signal(serde_json::to_value(&event).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to serialize event: {}",
            e
        )))
    })?)?;

    Ok(updated_hash)
}

/// Handle market resolution - update claim epistemic position
///
/// This is the key callback that closes the loop: when an Epistemic Market
/// resolves, this function updates the knowledge claim's epistemic classification.
#[hdk_extern]
pub fn on_verification_market_resolved(input: MarketResolvedInput) -> ExternResult<ActionHash> {
    let now = sys_time()?;

    // Determine if verification was successful
    let verification_successful = input.outcome == "Yes" && input.matl_weighted_confidence >= 0.67;

    // Create market evidence entry
    let evidence = MarketEvidence {
        id: format!("me_{}_{}", input.market_id, now.as_micros()),
        claim_id: input.claim_id.clone(),
        market_id: input.market_id.clone(),
        evidence_type: MarketEvidenceType::PredictionResolved {
            outcome: input.outcome.clone(),
            winning_probability: input.confidence,
        },
        confidence: input.confidence,
        oracle_count: input.oracle_count,
        matl_weighted_confidence: input.matl_weighted_confidence,
        total_stake: input.total_stake,
        timestamp: now,
    };

    let evidence_hash = create_entry(EntryTypes::MarketEvidence(evidence))?;

    // Create link from claim to evidence using anchor pattern
    let claim_anchor = format!("claim:{}", input.claim_id);
    create_entry(&EntryTypes::Anchor(Anchor(claim_anchor.clone())))?;
    create_link(
        anchor_hash(&claim_anchor)?,
        evidence_hash.clone(),
        LinkTypes::ClaimToMarketEvidence,
        (),
    )?;

    // If verification successful, update the claim's epistemic position
    if verification_successful {
        // Calculate the new epistemic position based on verification outcome
        let confidence_boost = input.matl_weighted_confidence * 0.3;

        // Update the claim via the claims zome
        #[derive(Serialize, Deserialize, Debug)]
        struct UpdateClaimInput {
            claim_id: String,
            content: Option<String>,
            classification: Option<EpistemicPosition>,
            sources: Option<Vec<String>>,
            tags: Option<Vec<String>>,
            confidence: Option<f64>,
            expires: Option<Timestamp>,
        }

        // First get the current claim to read its epistemic position
        let claim_response = call(
            CallTargetCell::Local,
            ZomeName::from("claims"),
            FunctionName::from("get_claim"),
            None,
            input.claim_id.clone(),
        )?;

        let claim_record: Option<Record> = match claim_response {
            ZomeCallResponse::Ok(bytes) => bytes.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode claim {} from claims zome: {}",
                    input.claim_id, e
                )))
            })?,
            ZomeCallResponse::NetworkError(e) => {
                debug!(
                    "Cross-zome call to claims::get_claim failed (network) for claim {}: {:?}",
                    input.claim_id, e
                );
                None
            }
            ZomeCallResponse::Unauthorized(_, _, _, _) => {
                debug!(
                    "Unauthorized to call claims::get_claim for claim {}",
                    input.claim_id
                );
                None
            }
            other => {
                debug!(
                    "Unexpected response from claims::get_claim for claim {}: {:?}",
                    input.claim_id, other
                );
                None
            }
        };

        if let Some(claim_record) = claim_record {
            if let Some(Entry::App(app_entry)) = claim_record.entry().as_option() {
                let entry_bytes = app_entry.bytes();
                if let Ok(claim_data) = serde_json::from_slice::<serde_json::Value>(&entry_bytes) {
                    // Extract current epistemic values
                    let classification = claim_data
                        .get("classification")
                        .cloned()
                        .unwrap_or_default();
                    let current_e = classification
                        .get("empirical")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.5);
                    let current_n = classification
                        .get("normative")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.5);
                    let current_m = classification
                        .get("mythic")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.5);

                    // Boost empirical position based on successful verification
                    let new_epistemic = EpistemicPosition {
                        empirical: (current_e + confidence_boost).min(1.0),
                        normative: current_n,
                        mythic: current_m,
                    };

                    let update_input = UpdateClaimInput {
                        claim_id: input.claim_id.clone(),
                        content: None,
                        classification: Some(new_epistemic.clone()),
                        sources: None,
                        tags: None,
                        confidence: Some((input.confidence + input.matl_weighted_confidence) / 2.0),
                        expires: None,
                    };

                    let update_response = call(
                        CallTargetCell::Local,
                        ZomeName::from("claims"),
                        FunctionName::from("update_claim"),
                        None,
                        update_input,
                    )?;

                    match update_response {
                        ZomeCallResponse::Ok(bytes) => {
                            debug!(
                                "Claim {} epistemic position updated via market {} verification",
                                input.claim_id, input.market_id
                            );
                            let _ = bytes; // Acknowledge the response

                            // Trigger cascade update for dependent claims
                            match call(
                                CallTargetCell::Local,
                                ZomeName::from("claims"),
                                FunctionName::from("cascade_update"),
                                None,
                                input.claim_id.clone(),
                            ) {
                                Ok(ZomeCallResponse::Ok(_)) => {
                                    debug!("Cascade update completed for claim {}", input.claim_id);
                                }
                                Ok(ZomeCallResponse::NetworkError(e)) => {
                                    debug!(
                                        "Cascade update failed (network) for claim {}: {:?}",
                                        input.claim_id, e
                                    );
                                }
                                Ok(ZomeCallResponse::Unauthorized(_, _, _, _)) => {
                                    debug!(
                                        "Cascade update unauthorized for claim {} - check capabilities",
                                        input.claim_id
                                    );
                                }
                                Ok(other) => {
                                    debug!(
                                        "Cascade update unexpected response for claim {}: {:?}",
                                        input.claim_id, other
                                    );
                                }
                                Err(e) => {
                                    debug!(
                                        "Cascade update call failed for claim {}: {:?}",
                                        input.claim_id, e
                                    );
                                }
                            }

                            // Emit claim updated event
                            let event = MarketsIntegrationEvent::ClaimUpdated {
                                claim_id: input.claim_id.clone(),
                                market_id: input.market_id.clone(),
                                new_epistemic,
                            };

                            emit_signal(serde_json::to_value(&event).map_err(|e| {
                                wasm_error!(WasmErrorInner::Guest(format!(
                                    "Failed to serialize event: {}",
                                    e
                                )))
                            })?)?;
                        }
                        ZomeCallResponse::NetworkError(e) => {
                            debug!(
                                "Failed to update claim {} epistemic position (network error): {:?}",
                                input.claim_id, e
                            );
                        }
                        ZomeCallResponse::Unauthorized(_, _, _, _) => {
                            debug!(
                                "Unauthorized to update claim {} - check zome call capabilities",
                                input.claim_id
                            );
                        }
                        other => {
                            debug!(
                                "Unexpected response updating claim {}: {:?}",
                                input.claim_id, other
                            );
                        }
                    }
                }
            }
        }

        debug!(
            "Claim {} verified successfully via market {}",
            input.claim_id, input.market_id
        );
    }

    // Notify DKG consensus mechanism of market resolution
    #[derive(Serialize, Deserialize, Debug)]
    struct DkgMarketResolvedInput {
        market_id: String,
        claim_hash: ActionHash,
        outcome: String,
        confidence: f64,
    }

    // Try to parse claim_id as an encoded ActionHash for DKG callback.
    // Not every claim_id is an action hash, so skip the callback when it is not.
    if let Ok(claim_action_hash_b64) = ActionHashB64::from_b64_str(&input.claim_id) {
        let claim_action_hash = ActionHash::from(claim_action_hash_b64);
        let dkg_input = DkgMarketResolvedInput {
            market_id: input.market_id.clone(),
            claim_hash: claim_action_hash,
            outcome: input.outcome.clone(),
            confidence: input.matl_weighted_confidence,
        };

        match call(
            CallTargetCell::Local,
            ZomeName::from("dkg"),
            FunctionName::from("on_market_resolved"),
            None,
            dkg_input,
        ) {
            Ok(ZomeCallResponse::Ok(_)) => {
                debug!(
                    "DKG consensus updated for claim {} via market {}",
                    input.claim_id, input.market_id
                );
            }
            Ok(other) => {
                debug!(
                    "DKG on_market_resolved returned non-Ok for claim {}: {:?}",
                    input.claim_id, other
                );
            }
            Err(e) => {
                debug!(
                    "DKG on_market_resolved call failed for claim {}: {:?}",
                    input.claim_id, e
                );
            }
        }
    }

    Ok(evidence_hash)
}

/// Register a claim being used as evidence in a prediction
///
/// When a predictor references a knowledge claim in their reasoning,
/// this creates a link that can affect claim credibility based on
/// prediction outcomes.
#[hdk_extern]
pub fn register_claim_in_prediction(input: RegisterClaimInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let reference = ClaimAsMarketEvidence {
        id: format!("came_{}_{}", input.claim_id, now.as_micros()),
        claim_id: input.claim_id.clone(),
        market_id: input.market_id,
        prediction_id: input.prediction_id,
        usage: input.usage,
        relevance_score: input.relevance_score,
        referenced_by: agent,
        timestamp: now,
    };

    let action_hash = create_entry(EntryTypes::ClaimAsMarketEvidence(reference))?;

    // Create links using anchor pattern
    let claim_anchor = format!("claim:{}", input.claim_id);
    create_entry(&EntryTypes::Anchor(Anchor(claim_anchor.clone())))?;
    create_link(
        anchor_hash(&claim_anchor)?,
        action_hash.clone(),
        LinkTypes::ClaimUsedInPrediction,
        (),
    )?;

    Ok(action_hash)
}

// ============================================================================
// QUERIES
// ============================================================================

/// Get all markets related to a claim
///
/// Returns both verification markets (created FOR this claim) and markets
/// where this claim is used as evidence.
#[hdk_extern]
pub fn get_claim_markets(claim_id: String) -> ExternResult<Vec<MarketSummary>> {
    let claim_anchor = format!("claim:{}", claim_id);
    let mut summaries = Vec::new();

    // Get verification market requests
    let request_links = get_links(
        LinkQuery::try_new(anchor_hash(&claim_anchor)?, LinkTypes::ClaimToMarketRequest)?,
        GetStrategy::default(),
    )?;

    for link in request_links {
        if let Ok(target) = ActionHash::try_from(link.target) {
            if let Some(record) = get(target, GetOptions::default())? {
                if let Some(request) = record
                    .entry()
                    .to_app_option::<VerificationMarketRequest>()
                    .ok()
                    .flatten()
                {
                    summaries.push(MarketSummary {
                        market_id: match &request.status {
                            MarketRequestStatus::MarketCreated { market_id, .. } => {
                                market_id.clone()
                            }
                            MarketRequestStatus::MarketActive { market_id, .. } => {
                                market_id.clone()
                            }
                            MarketRequestStatus::MarketResolved { market_id, .. } => {
                                market_id.clone()
                            }
                            MarketRequestStatus::ClaimUpdated { market_id, .. } => {
                                market_id.clone()
                            }
                            _ => request.id.clone(),
                        },
                        question: request.question,
                        relationship: MarketRelationship::VerificationMarket,
                        status: format!("{:?}", request.status),
                        current_probability: match &request.status {
                            MarketRequestStatus::MarketActive {
                                current_probability,
                                ..
                            } => Some(*current_probability),
                            _ => None,
                        },
                        resolution_outcome: match &request.status {
                            MarketRequestStatus::MarketResolved { outcome, .. } => {
                                Some(outcome.clone())
                            }
                            _ => None,
                        },
                    });
                }
            }
        }
    }

    // Get claims used as evidence
    let evidence_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&claim_anchor)?,
            LinkTypes::ClaimUsedInPrediction,
        )?,
        GetStrategy::default(),
    )?;

    for link in evidence_links {
        if let Ok(target) = ActionHash::try_from(link.target) {
            if let Some(record) = get(target, GetOptions::default())? {
                if let Some(reference) = record
                    .entry()
                    .to_app_option::<ClaimAsMarketEvidence>()
                    .ok()
                    .flatten()
                {
                    summaries.push(MarketSummary {
                        market_id: reference.market_id,
                        question: format!(
                            "Prediction {} references this claim",
                            reference.prediction_id
                        ),
                        relationship: MarketRelationship::ClaimUsedAsEvidence,
                        status: "Active".to_string(),
                        current_probability: None,
                        resolution_outcome: None,
                    });
                }
            }
        }
    }

    Ok(summaries)
}

/// Get market evidence for a claim
#[hdk_extern]
pub fn get_claim_evidence(claim_id: String) -> ExternResult<Vec<Record>> {
    let claim_anchor = format!("claim:{}", claim_id);

    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&claim_anchor)?,
            LinkTypes::ClaimToMarketEvidence,
        )?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Ok(target) = ActionHash::try_from(link.target) {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

// ============================================================================
// MARKET VALUE ASSESSMENT
// ============================================================================

/// Calculate the value of creating a verification market for a claim
///
/// Analyzes:
/// - Information value: How much would verification teach us?
/// - Decision impact: How many decisions depend on this claim?
/// - Current uncertainty: How uncertain is the current classification?
/// - Dependency count: How many other claims depend on this?
#[hdk_extern]
pub fn calculate_market_value(claim_id: String) -> ExternResult<ActionHash> {
    let now = sys_time()?;

    // In a full implementation, we'd:
    // 1. Query the graph zome for dependencies
    // 2. Analyze cross-hApp references
    // 3. Calculate information value using entropy
    // 4. Estimate market participation from similar markets

    // For now, create a placeholder assessment
    let assessment = MarketValueAssessment {
        id: format!("mva_{}_{}", claim_id, now.as_micros()),
        claim_id: claim_id.clone(),
        information_value: 0.7,
        decision_impact: 0.5,
        uncertainty: 0.6,
        dependency_count: 3,
        estimated_participation: 10,
        verification_cost: 50.0,
        recommendation: MarketRecommendation::CreateMarket {
            reason: "Moderate information value with reasonable cost".to_string(),
            suggested_bounty: 100,
            suggested_deadline_days: 7,
        },
        assessed_at: now,
        expires_at: Timestamp::from_micros(now.as_micros() + 7 * 24 * 60 * 60 * 1_000_000),
    };

    let action_hash = create_entry(EntryTypes::MarketValueAssessment(assessment))?;

    // Link to claim using anchor pattern
    let claim_anchor = format!("claim:{}", claim_id);
    create_entry(&EntryTypes::Anchor(Anchor(claim_anchor.clone())))?;
    create_link(
        anchor_hash(&claim_anchor)?,
        action_hash.clone(),
        LinkTypes::ClaimToValueAssessment,
        (),
    )?;

    Ok(action_hash)
}

/// Get the latest market value assessment for a claim
#[hdk_extern]
pub fn get_market_value_assessment(claim_id: String) -> ExternResult<Option<Record>> {
    let claim_anchor = format!("claim:{}", claim_id);

    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&claim_anchor)?,
            LinkTypes::ClaimToValueAssessment,
        )?,
        GetStrategy::default(),
    )?;

    // Get the most recent assessment
    if let Some(link) = links.last() {
        if let Ok(target) = ActionHash::try_from(link.target.clone()) {
            return get(target, GetOptions::default());
        }
    }

    Ok(None)
}

// ============================================================================
// CASCADE UPDATES
// ============================================================================

/// Trigger cascade update when a claim's epistemic position changes
///
/// When a claim is verified through a market, this propagates the update
/// to all dependent claims, potentially triggering new verification requests.
#[hdk_extern]
pub fn trigger_cascade_update(claim_id: String) -> ExternResult<CascadeUpdateResult> {
    // In a full implementation, we'd:
    // 1. Query the graph zome for all dependents
    // 2. Recalculate belief strength for each dependent
    // 3. Check if any dependents should spawn verification markets
    // 4. Emit events for the updates

    let result = CascadeUpdateResult {
        updated_claims: vec![],
        triggered_verifications: vec![],
        errors: vec![],
    };

    debug!("Cascade update triggered for claim {}", claim_id);

    Ok(result)
}
