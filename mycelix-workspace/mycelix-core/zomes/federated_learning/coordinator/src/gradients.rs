// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gradient submission, role management, reputation queries.

use hdk::prelude::*;
use federated_learning_integrity::*;

use crate::config::*;
use crate::auth::*;
use crate::detection::*;
use crate::signals::Signal;
use crate::bridge::*;
use crate::pipeline::get_or_create_reputation;
use super::ensure_path;

/// Grant a role to an agent (coordinator only)
#[hdk_extern]
pub fn grant_role(input: GrantRoleInput) -> ExternResult<()> {
    require_coordinator_role()?;
    validate_node_id(&input.agent)?;

    let role_path = Path::from(format!("roles.{}.{}", input.role, input.agent));
    let typed = role_path.typed(LinkTypes::RoundToGradients)?;
    typed.ensure()?;

    Ok(())
}

/// Revoke a role from an agent (coordinator only)
#[hdk_extern]
pub fn revoke_role(input: GrantRoleInput) -> ExternResult<()> {
    require_coordinator_role()?;

    // Cannot revoke own coordinator role
    let agent = agent_info()?.agent_initial_pubkey.to_string();
    if input.role == COORDINATOR_ROLE && input.agent == agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot revoke your own coordinator role".to_string()
        )));
    }

    // Create revocation path to track that this role was revoked
    // (Paths cannot be directly deleted in Holochain, so we track revocations separately)
    let revoke_path = Path::from(format!("roles_revoked.{}.{}", input.role, input.agent));
    let typed_revoke = revoke_path.typed(LinkTypes::RoundToGradients)?;
    typed_revoke.ensure()?;

    Ok(())
}

/// Input for role management
#[derive(Serialize, Deserialize, Debug)]
pub struct GrantRoleInput {
    pub agent: String,
    pub role: String,
}

/// Input for submitting a gradient
#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitGradientInput {
    pub node_id: String,
    pub round: u32,
    pub gradient_hash: String,
    pub cpu_usage: f32,
    pub memory_mb: f32,
    pub network_latency_ms: f32,
    pub trust_score: Option<f32>,
    /// Optional model architecture hash for version validation
    #[serde(default)]
    pub architecture_hash: Option<String>,
}

/// Submit a gradient to the DHT
///
/// This function performs Byzantine detection BEFORE storing the gradient.
/// Malicious gradients are REJECTED at the zome level, not just recorded.
#[hdk_extern]
pub fn submit_gradient(input: SubmitGradientInput) -> ExternResult<ActionHash> {
    // F-01: Validate node_id
    validate_node_id(&input.node_id)?;

    // H-02: Bind node_id to caller's AgentPubKey
    // This prevents impersonation and Sybil attacks
    let agent = agent_info()?.agent_initial_pubkey;
    let caller_id = agent.to_string();
    if input.node_id != caller_id {
        // H-02: node_id must match caller's AgentPubKey
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "node_id must match caller's agent pubkey. Expected: {}, got: {}",
            caller_id, input.node_id
        ))));
    }

    // H-01: Ignore self-reported trust_score — on-chain reputation is authoritative.
    // Reputation includes time-based decay computed lazily at retrieval.
    let on_chain_trust = match get_or_create_reputation(&input.node_id) {
        Ok(rep) => {
            // QUARANTINE: Reject gradients from nodes below minimum reputation
            if rep.reputation_score < crate::config::MIN_REPUTATION_FOR_SUBMISSION {
                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Node {} is quarantined: reputation {:.3} below minimum {:.3}. \
                     Participate in successful rounds to recover.",
                    input.node_id, rep.reputation_score,
                    crate::config::MIN_REPUTATION_FOR_SUBMISSION
                ))));
            }
            Some(rep.reputation_score)
        }
        Err(_) => Some(0.5), // Default for new/unknown nodes
    };

    // F-06: Check rate limit
    check_rate_limit("submit_gradient")?;

    // ==========================================================================
    // BYZANTINE DETECTION: Run before storing gradient
    // ==========================================================================

    // Extract features for Byzantine detection
    let gradient_features = extract_gradient_features(&input)?;

    // Run hierarchical detection
    let (is_byzantine, detection_confidence) = run_byzantine_detection(&gradient_features);

    // If Byzantine detected with sufficient confidence, REJECT the gradient
    if is_byzantine && detection_confidence >= BYZANTINE_REJECTION_THRESHOLD {
        // Compute evidence hash for audit trail
        let evidence_hash = compute_evidence_hash(&gradient_features, detection_confidence);

        // Record Byzantine behavior (creates permanent record)
        record_byzantine_internal(
            &input.node_id,
            input.round,
            "hierarchical_submit",
            detection_confidence,
            &evidence_hash,
        )?;

        // REJECT: Do not store the gradient
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Byzantine gradient detected and rejected: node={}, round={}, confidence={:.2}",
                input.node_id, input.round, detection_confidence
            )
        )));
    }

    // ==========================================================================
    // GRADIENT ACCEPTED: Store as normal
    // ==========================================================================

    let timestamp = sys_time()?;
    let gradient = ModelGradient {
        node_id: input.node_id.clone(),
        round: input.round,
        gradient_hash: input.gradient_hash,
        timestamp: timestamp.0 as i64 / 1_000_000, // Convert to seconds
        cpu_usage: input.cpu_usage,
        memory_mb: input.memory_mb,
        network_latency_ms: input.network_latency_ms,
        trust_score: on_chain_trust, // H-01: Always use on-chain trust, never self-reported
    };

    let action_hash = create_entry(&EntryTypes::ModelGradient(gradient.clone()))?;

    // Create link from round anchor to this gradient
    let round_path = Path::from(format!("round.{}", input.round));
    let round_entry_hash = ensure_path(round_path, LinkTypes::RoundToGradients)?;
    create_link(
        round_entry_hash,
        action_hash.clone(),
        LinkTypes::RoundToGradients,
        vec![],
    )?;

    // Create link from node to this gradient
    let node_path = Path::from(format!("node.{}", input.node_id.clone()));
    let node_entry_hash = ensure_path(node_path, LinkTypes::NodeToGradients)?;
    create_link(
        node_entry_hash,
        action_hash.clone(),
        LinkTypes::NodeToGradients,
        vec![],
    )?;

    // Emit signal for other nodes
    // NOTE: signature is None for locally emitted signals
    // Remote recipients should use send_signed_signal for cryptographic authentication
    emit_signal(Signal::GradientSubmitted {
        node_id: input.node_id,
        round: input.round,
        action_hash: action_hash.clone(),
        source: None,
        signature: None,
    })?;

    Ok(action_hash)
}

/// Get all gradients for a specific round
#[hdk_extern]
pub fn get_round_gradients(round: u32) -> ExternResult<Vec<(ActionHash, ModelGradient)>> {
    let round_path = Path::from(format!("round.{}", round));
    let round_entry_hash = ensure_path(round_path, LinkTypes::RoundToGradients)?;

    let links = get_links(
        LinkQuery::new(
            round_entry_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::RoundToGradients as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut gradients = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(gradient) = record
                    .entry()
                    .as_option()
                    .and_then(|entry| match entry {
                        Entry::App(bytes) => ModelGradient::try_from(
                            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                        )
                        .ok(),
                        _ => None,
                    })
                {
                    gradients.push((action_hash, gradient));
                }
            }
        }
    }

    Ok(gradients)
}

/// Get all gradients from a specific node
#[hdk_extern]
pub fn get_node_gradients(node_id: String) -> ExternResult<Vec<(ActionHash, ModelGradient)>> {
    let node_path = Path::from(format!("node.{}", node_id));
    let node_entry_hash = ensure_path(node_path, LinkTypes::NodeToGradients)?;

    let links = get_links(
        LinkQuery::new(
            node_entry_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::NodeToGradients as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut gradients = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(gradient) = record
                    .entry()
                    .as_option()
                    .and_then(|entry| match entry {
                        Entry::App(bytes) => ModelGradient::try_from(
                            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                        )
                        .ok(),
                        _ => None,
                    })
                {
                    gradients.push((action_hash, gradient));
                }
            }
        }
    }

    Ok(gradients)
}

/// Input for completing a training round
#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteRoundInput {
    pub round: u32,
    pub selected_gradient: ActionHash,
    pub participant_count: u32,
    pub byzantine_detected: bool,
    pub byzantine_count: u32,
    pub accuracy: f32,
}

/// Complete a training round with Byzantine defense results
/// SECURITY (F-03): Requires coordinator role
#[hdk_extern]
pub fn complete_round(input: CompleteRoundInput) -> ExternResult<ActionHash> {
    // F-03: Authorization check - only coordinators can complete rounds
    require_coordinator_role()?;

    let timestamp = sys_time()?;
    let round_result = TrainingRound {
        round: input.round,
        selected_gradient: input.selected_gradient,
        participant_count: input.participant_count,
        byzantine_detected: input.byzantine_detected,
        byzantine_count: input.byzantine_count,
        accuracy: input.accuracy,
        completed_at: timestamp.0 as i64 / 1_000_000,
    };

    let action_hash = create_entry(&EntryTypes::TrainingRound(round_result.clone()))?;

    // Emit signal for round completion
    emit_signal(Signal::RoundCompleted {
        round: input.round,
        accuracy: input.accuracy,
        byzantine_count: input.byzantine_count,
        source: None,
        signature: None,
    })?;

    Ok(action_hash)
}

/// Record a Byzantine detection
/// SECURITY (F-03): Requires detector role
///
/// This function records Byzantine behavior locally and also reports it to the
/// Bridge zome for cross-hApp reputation tracking. The Bridge call is non-blocking -
/// if it fails, the local recording still succeeds.
#[hdk_extern]
pub fn record_byzantine(input: ByzantineRecordInput) -> ExternResult<ActionHash> {
    // F-03: Authorization check - only detectors can record Byzantine behavior
    require_detector_role()?;
    // F-01: Validate node_id
    validate_node_id(&input.node_id)?;
    // F-06: Rate limit
    check_rate_limit("record_byzantine")?;

    let timestamp = sys_time()?;
    let detected_at = timestamp.0 as i64 / 1_000_000;
    let record = ByzantineRecord {
        node_id: input.node_id.clone(),
        round: input.round,
        detection_method: input.detection_method.clone(),
        confidence: input.confidence,
        evidence_hash: input.evidence_hash.clone(),
        detected_at,
    };

    let action_hash = create_entry(&EntryTypes::ByzantineRecord(record))?;

    // Link to round
    let round_path = Path::from(format!("round_byzantine.{}", input.round));
    let round_entry_hash = ensure_path(round_path, LinkTypes::RoundToByzantine)?;
    create_link(
        round_entry_hash,
        action_hash.clone(),
        LinkTypes::RoundToByzantine,
        vec![],
    )?;

    // === BRIDGE INTEGRATION: Report negative reputation ===
    // Calculate negative score based on confidence (higher confidence = lower reputation)
    // Score is inverted: confidence 0.9 -> reputation 0.1
    let negative_score = (1.0 - input.confidence).max(0.0).min(1.0) as f64;

    // Get current reputation to track interaction counts
    let current_rep = get_or_create_reputation(&input.node_id).unwrap_or(NodeReputation {
        node_id: input.node_id.clone(),
        successful_rounds: 0,
        failed_rounds: 0,
        reputation_score: 0.5,
        last_updated: detected_at,
    });

    // Report to Bridge zome with synchronous confirmation for Byzantine detection
    // This is security-critical so we require confirmation with retries
    let reputation_result = call_bridge_record_reputation_sync(
        &input.node_id,
        negative_score,
        current_rep.successful_rounds as u64,
        current_rep.failed_rounds as u64 + 1, // Increment negative interactions
        Some(input.evidence_hash.clone()),
    )?;

    if !reputation_result.confirmed {
        debug!(
            "Warning: Byzantine reputation update not confirmed after {} attempts: {:?}",
            reputation_result.attempts, reputation_result.error
        );
    }

    // === BRIDGE INTEGRATION: Broadcast Byzantine detection event ===
    let event_payload = serde_json::json!({
        "node_id": input.node_id,
        "round": input.round,
        "detection_method": input.detection_method,
        "confidence": input.confidence,
        "evidence_hash": input.evidence_hash,
        "detected_at": detected_at,
        "reputation_confirmed": reputation_result.confirmed,
    });

    // Broadcast with high priority and synchronous confirmation (Byzantine detection is security-critical)
    let broadcast_result = call_bridge_broadcast_event_sync(
        EVENT_BYZANTINE_DETECTED,
        &event_payload,
        2, // Critical priority for Byzantine events
    )?;

    if !broadcast_result.confirmed {
        debug!(
            "Warning: Byzantine event broadcast not confirmed after {} attempts: {:?}",
            broadcast_result.attempts, broadcast_result.error
        );
    }

    // Emit local signal
    emit_signal(Signal::ByzantineDetected {
        node_id: input.node_id,
        round: input.round,
        confidence: input.confidence,
        source: None,
        signature: None,
    })?;

    Ok(action_hash)
}

/// Input for Byzantine record
#[derive(Serialize, Deserialize, Debug)]
pub struct ByzantineRecordInput {
    pub node_id: String,
    pub round: u32,
    pub detection_method: String,
    pub confidence: f32,
    pub evidence_hash: String,
}

/// Get Byzantine records for a round
#[hdk_extern]
pub fn get_round_byzantine_records(round: u32) -> ExternResult<Vec<ByzantineRecord>> {
    let round_path = Path::from(format!("round_byzantine.{}", round));
    let round_entry_hash = ensure_path(round_path, LinkTypes::RoundToByzantine)?;

    let links = get_links(
        LinkQuery::new(
            round_entry_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::RoundToByzantine as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(byzantine_record) = record
                    .entry()
                    .as_option()
                    .and_then(|entry| match entry {
                        Entry::App(bytes) => ByzantineRecord::try_from(
                            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                        )
                        .ok(),
                        _ => None,
                    })
                {
                    records.push(byzantine_record);
                }
            }
        }
    }

    Ok(records)
}

/// Update or create node reputation
/// SECURITY (F-03): Requires coordinator role
#[hdk_extern]
pub fn update_reputation(input: UpdateReputationInput) -> ExternResult<ActionHash> {
    // F-03: Authorization check - only coordinators can update reputation
    require_coordinator_role()?;
    // F-01: Validate node_id
    validate_node_id(&input.node_id)?;

    let timestamp = sys_time()?;
    let node_path = Path::from(format!("node_reputation.{}", input.node_id.clone()));
    let node_entry_hash = ensure_path(node_path.clone(), LinkTypes::NodeToReputation)?;

    // Check if reputation exists
    let existing_links = get_links(
        LinkQuery::new(
            node_entry_hash.clone(),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::NodeToReputation as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let reputation = NodeReputation {
        node_id: input.node_id.clone(),
        successful_rounds: input.successful_rounds,
        failed_rounds: input.failed_rounds,
        reputation_score: input.reputation_score,
        last_updated: timestamp.0 as i64 / 1_000_000,
    };

    let action_hash = if let Some(link) = existing_links.first() {
        // Update existing
        if let Some(old_hash) = link.target.clone().into_action_hash() {
            update_entry(old_hash, &EntryTypes::NodeReputation(reputation))?
        } else {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Invalid link target".to_string()
            )));
        }
    } else {
        // Create new
        let hash = create_entry(&EntryTypes::NodeReputation(reputation))?;
        create_link(node_entry_hash, hash.clone(), LinkTypes::NodeToReputation, vec![])?;
        hash
    };

    Ok(action_hash)
}

/// Input for reputation update
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateReputationInput {
    pub node_id: String,
    pub successful_rounds: u32,
    pub failed_rounds: u32,
    pub reputation_score: f32,
}

/// Get node reputation with time-based decay applied.
/// Returns the decay-adjusted reputation score (not the raw stored value).
#[hdk_extern]
pub fn get_reputation(node_id: String) -> ExternResult<Option<NodeReputation>> {
    let node_path = Path::from(format!("node_reputation.{}", node_id));
    let node_entry_hash = ensure_path(node_path, LinkTypes::NodeToReputation)?;

    let links = get_links(
        LinkQuery::new(
            node_entry_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::NodeToReputation as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(mut rep) = record
                    .entry()
                    .as_option()
                    .and_then(|entry| match entry {
                        Entry::App(bytes) => NodeReputation::try_from(
                            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                        )
                        .ok(),
                        _ => None,
                    })
                {
                    // Apply time-based decay to returned score
                    rep.reputation_score = crate::pipeline::apply_reputation_decay(&rep)?;
                    return Ok(Some(rep));
                }
            }
        }
    }

    Ok(None)
}
