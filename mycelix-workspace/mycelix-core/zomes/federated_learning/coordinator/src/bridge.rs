// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-zome bridge integration for reputation and event broadcasting.

use hdk::prelude::*;

use crate::auth::{validate_node_id, require_coordinator_role};
use super::ensure_path;
use crate::pipeline::get_or_create_reputation;
use federated_learning_integrity::*;

/// The Bridge zome name for cross-zome calls
pub(crate) const BRIDGE_ZOME_NAME: &str = "bridge";

/// FL hApp identifier for Bridge registration
pub(crate) const FL_HAPP_ID: &str = "federated_learning";
pub(crate) const FL_HAPP_NAME: &str = "Mycelix Federated Learning";

/// Bridge event types
pub(crate) const EVENT_BYZANTINE_DETECTED: &str = "fl_byzantine_detected";
pub(crate) const EVENT_ROUND_COMPLETED: &str = "fl_round_completed";
pub(crate) const EVENT_COORDINATOR_ELECTED: &str = "fl_coordinator_elected";

// ============================================================================
// CROSS-ZOME RELIABILITY LAYER
// ============================================================================

/// Configuration for cross-zome call retry behavior
#[derive(Clone, Debug)]
pub struct CrossZomeRetryConfig {
    pub max_retries: u32,
    pub base_delay_us: u64,
    pub max_total_time_us: u64,
    pub require_confirmation: bool,
}

impl Default for CrossZomeRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_us: 100_000,
            max_total_time_us: 5_000_000,
            require_confirmation: true,
        }
    }
}

impl CrossZomeRetryConfig {
    #[allow(dead_code)] // Part of the designed retry API; will be used when fire-and-forget broadcasts are wired
    pub fn fire_and_forget() -> Self {
        Self {
            max_retries: 0,
            base_delay_us: 0,
            max_total_time_us: 0,
            require_confirmation: false,
        }
    }

    pub fn critical() -> Self {
        Self {
            max_retries: 5,
            base_delay_us: 200_000,
            max_total_time_us: 15_000_000,
            require_confirmation: true,
        }
    }
}

/// Result of a cross-zome call with confirmation status
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrossZomeCallResult {
    pub action_hash: Option<ActionHash>,
    pub attempts: u32,
    pub confirmed: bool,
    pub error: Option<String>,
    pub elapsed_us: u64,
}

impl CrossZomeCallResult {
    pub(crate) fn success(hash: ActionHash, attempts: u32, elapsed_us: u64) -> Self {
        Self { action_hash: Some(hash), attempts, confirmed: true, error: None, elapsed_us }
    }

    pub(crate) fn failure(error: String, attempts: u32, elapsed_us: u64) -> Self {
        Self { action_hash: None, attempts, confirmed: false, error: Some(error), elapsed_us }
    }

    pub(crate) fn skipped() -> Self {
        Self { action_hash: None, attempts: 0, confirmed: false, error: Some("Fire-and-forget mode - no confirmation".to_string()), elapsed_us: 0 }
    }
}

/// Error type for cross-zome calls
#[allow(dead_code)] // Part of the designed cross-zome reliability API
#[derive(Debug, Clone)]
pub enum CrossZomeError {
    CallFailed(String),
    Unauthorized,
    DecodeError(String),
    MaxRetriesExceeded { attempts: u32, last_error: String },
    TimeoutExceeded { elapsed_us: u64 },
}

impl std::fmt::Display for CrossZomeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CallFailed(e) => write!(f, "Cross-zome call failed: {}", e),
            Self::Unauthorized => write!(f, "Unauthorized to call target zome"),
            Self::DecodeError(e) => write!(f, "Failed to decode response: {}", e),
            Self::MaxRetriesExceeded { attempts, last_error } => {
                write!(f, "Max retries ({}) exceeded, last error: {}", attempts, last_error)
            }
            Self::TimeoutExceeded { elapsed_us } => {
                write!(f, "Timeout exceeded after {}us", elapsed_us)
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeRecordReputationInput {
    pub agent: String,
    pub happ_id: String,
    pub happ_name: String,
    pub score: f64,
    pub interactions: u64,
    pub negative_interactions: u64,
    pub evidence_hash: Option<String>,
}

/// Input for Bridge zome's broadcast_event function
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeBroadcastEventInput {
    pub event_type: String,
    pub payload: Vec<u8>,
    pub targets: Vec<String>,
    pub priority: u8,
}

/// Call the Bridge zome to record reputation with retry support
pub(crate) fn call_bridge_record_reputation_with_retry(
    agent: &str,
    score: f64,
    positive_interactions: u64,
    negative_interactions: u64,
    evidence_hash: Option<String>,
    config: &CrossZomeRetryConfig,
) -> CrossZomeCallResult {
    let start_time = std::time::Instant::now();

    if !config.require_confirmation && config.max_retries == 0 {
        let input = BridgeRecordReputationInput {
            agent: agent.to_string(),
            happ_id: FL_HAPP_ID.to_string(),
            happ_name: FL_HAPP_NAME.to_string(),
            score,
            interactions: positive_interactions,
            negative_interactions,
            evidence_hash,
        };

        let _ = call(
            CallTargetCell::Local,
            ZomeName::from(BRIDGE_ZOME_NAME),
            FunctionName::from("record_reputation"),
            None,
            input,
        );
        return CrossZomeCallResult::skipped();
    }

    let mut attempts = 0u32;
    #[allow(unused_assignments)]
    let mut last_error = String::new();
    let mut current_delay = config.base_delay_us;

    loop {
        attempts += 1;
        let elapsed = start_time.elapsed().as_micros() as u64;

        if elapsed > config.max_total_time_us && config.max_total_time_us > 0 {
            return CrossZomeCallResult::failure(
                format!("Timeout exceeded: {}us", elapsed),
                attempts,
                elapsed,
            );
        }

        let input = BridgeRecordReputationInput {
            agent: agent.to_string(),
            happ_id: FL_HAPP_ID.to_string(),
            happ_name: FL_HAPP_NAME.to_string(),
            score,
            interactions: positive_interactions,
            negative_interactions,
            evidence_hash: evidence_hash.clone(),
        };

        match call(
            CallTargetCell::Local,
            ZomeName::from(BRIDGE_ZOME_NAME),
            FunctionName::from("record_reputation"),
            None,
            input,
        ) {
            Ok(response) => {
                match response {
                    ZomeCallResponse::Ok(extern_io) => {
                        match extern_io.decode::<ActionHash>() {
                            Ok(action_hash) => {
                                debug!(
                                    "Successfully recorded reputation for {} via Bridge after {} attempts: {:?}",
                                    agent, attempts, action_hash
                                );
                                return CrossZomeCallResult::success(
                                    action_hash,
                                    attempts,
                                    start_time.elapsed().as_micros() as u64,
                                );
                            }
                            Err(e) => {
                                last_error = format!("Decode error: {:?}", e);
                            }
                        }
                    }
                    ZomeCallResponse::Unauthorized(_, _, _, _) => {
                        return CrossZomeCallResult::failure(
                            "Unauthorized to call Bridge zome".to_string(),
                            attempts,
                            start_time.elapsed().as_micros() as u64,
                        );
                    }
                    _ => {
                        last_error = "Unexpected ZomeCallResponse variant".to_string();
                    }
                }
            }
            Err(e) => {
                last_error = format!("Call failed: {:?}", e);
            }
        }

        if attempts > config.max_retries {
            return CrossZomeCallResult::failure(
                format!("Max retries exceeded: {}", last_error),
                attempts,
                start_time.elapsed().as_micros() as u64,
            );
        }

        let delay_until = start_time.elapsed().as_micros() as u64 + current_delay;
        while (start_time.elapsed().as_micros() as u64) < delay_until {
            std::hint::spin_loop();
        }
        current_delay *= 2;
    }
}

/// Legacy wrapper for fire-and-forget reputation recording
pub(crate) fn call_bridge_record_reputation(
    agent: &str,
    score: f64,
    positive_interactions: u64,
    negative_interactions: u64,
    evidence_hash: Option<String>,
) -> ExternResult<Option<ActionHash>> {
    let result = call_bridge_record_reputation_with_retry(
        agent, score, positive_interactions, negative_interactions, evidence_hash,
        &CrossZomeRetryConfig::default(),
    );
    Ok(result.action_hash)
}

/// Record reputation with synchronous confirmation (blocking)
pub(crate) fn call_bridge_record_reputation_sync(
    agent: &str,
    score: f64,
    positive_interactions: u64,
    negative_interactions: u64,
    evidence_hash: Option<String>,
) -> ExternResult<CrossZomeCallResult> {
    Ok(call_bridge_record_reputation_with_retry(
        agent, score, positive_interactions, negative_interactions, evidence_hash,
        &CrossZomeRetryConfig::critical(),
    ))
}

/// Call the Bridge zome to broadcast an event with retry support
pub(crate) fn call_bridge_broadcast_event_with_retry(
    event_type: &str,
    payload: &serde_json::Value,
    priority: u8,
    config: &CrossZomeRetryConfig,
) -> CrossZomeCallResult {
    let start_time = std::time::Instant::now();
    let payload_bytes = serde_json::to_vec(payload).unwrap_or_default();

    if !config.require_confirmation && config.max_retries == 0 {
        let input = BridgeBroadcastEventInput {
            event_type: event_type.to_string(),
            payload: payload_bytes,
            targets: vec![],
            priority,
        };

        let _ = call(
            CallTargetCell::Local,
            ZomeName::from(BRIDGE_ZOME_NAME),
            FunctionName::from("broadcast_event"),
            None,
            input,
        );
        return CrossZomeCallResult::skipped();
    }

    let mut attempts = 0u32;
    #[allow(unused_assignments)]
    let mut last_error = String::new();
    let mut current_delay = config.base_delay_us;

    loop {
        attempts += 1;
        let elapsed = start_time.elapsed().as_micros() as u64;

        if elapsed > config.max_total_time_us && config.max_total_time_us > 0 {
            return CrossZomeCallResult::failure(
                format!("Timeout exceeded: {}us", elapsed),
                attempts,
                elapsed,
            );
        }

        let input = BridgeBroadcastEventInput {
            event_type: event_type.to_string(),
            payload: payload_bytes.clone(),
            targets: vec![],
            priority,
        };

        match call(
            CallTargetCell::Local,
            ZomeName::from(BRIDGE_ZOME_NAME),
            FunctionName::from("broadcast_event"),
            None,
            input,
        ) {
            Ok(response) => {
                match response {
                    ZomeCallResponse::Ok(extern_io) => {
                        match extern_io.decode::<ActionHash>() {
                            Ok(action_hash) => {
                                debug!(
                                    "Successfully broadcast {} event via Bridge after {} attempts: {:?}",
                                    event_type, attempts, action_hash
                                );
                                return CrossZomeCallResult::success(
                                    action_hash,
                                    attempts,
                                    start_time.elapsed().as_micros() as u64,
                                );
                            }
                            Err(e) => {
                                last_error = format!("Decode error: {:?}", e);
                            }
                        }
                    }
                    ZomeCallResponse::Unauthorized(_, _, _, _) => {
                        return CrossZomeCallResult::failure(
                            "Unauthorized to call Bridge zome for broadcast".to_string(),
                            attempts,
                            start_time.elapsed().as_micros() as u64,
                        );
                    }
                    _ => {
                        last_error = "Unexpected ZomeCallResponse variant from broadcast".to_string();
                    }
                }
            }
            Err(e) => {
                last_error = format!("Broadcast call failed: {:?}", e);
            }
        }

        if attempts > config.max_retries {
            return CrossZomeCallResult::failure(
                format!("Max retries exceeded: {}", last_error),
                attempts,
                start_time.elapsed().as_micros() as u64,
            );
        }

        let delay_until = start_time.elapsed().as_micros() as u64 + current_delay;
        while (start_time.elapsed().as_micros() as u64) < delay_until {
            std::hint::spin_loop();
        }
        current_delay *= 2;
    }
}

/// Legacy wrapper for backward compatibility
pub(crate) fn call_bridge_broadcast_event(
    event_type: &str,
    payload: &serde_json::Value,
    priority: u8,
) -> ExternResult<Option<ActionHash>> {
    let config = CrossZomeRetryConfig::default();
    let result = call_bridge_broadcast_event_with_retry(event_type, payload, priority, &config);
    Ok(result.action_hash)
}

/// Broadcast event with synchronous confirmation (blocking)
pub(crate) fn call_bridge_broadcast_event_sync(
    event_type: &str,
    payload: &serde_json::Value,
    priority: u8,
) -> ExternResult<CrossZomeCallResult> {
    let config = CrossZomeRetryConfig::critical();
    Ok(call_bridge_broadcast_event_with_retry(event_type, payload, priority, &config))
}

/// Input for updating positive reputation after successful round participation
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdatePositiveReputationInput {
    pub node_id: String,
    pub round: u32,
    pub contribution_quality: f64,
    pub evidence_hash: Option<String>,
}

/// Result of positive reputation update
#[derive(Serialize, Deserialize, Debug)]
pub struct PositiveReputationResult {
    pub local_reputation_hash: Option<ActionHash>,
    pub bridge_reputation_hash: Option<ActionHash>,
    pub bridge_event_hash: Option<ActionHash>,
    pub new_reputation_score: f32,
}

/// Update node reputation positively after successful round participation
#[hdk_extern]
pub fn update_node_reputation_positive(input: UpdatePositiveReputationInput) -> ExternResult<PositiveReputationResult> {
    // SECURITY: Only coordinators can positively update reputation
    require_coordinator_role()?;

    validate_node_id(&input.node_id)?;

    let current_rep = get_or_create_reputation(&input.node_id)?;
    // Note: current_rep.reputation_score already has decay applied (lazy evaluation).
    // The positive update is computed on the decayed value, then persisted with fresh timestamp.

    let quality_weight = input.contribution_quality.clamp(0.0, 1.0);
    let prior_weight = (current_rep.successful_rounds + current_rep.failed_rounds) as f64;
    let new_score = if prior_weight > 0.0 {
        let weighted_prior = current_rep.reputation_score as f64 * prior_weight;
        let weighted_new = quality_weight;
        ((weighted_prior + weighted_new) / (prior_weight + 1.0)) as f32
    } else {
        quality_weight as f32
    };

    let new_successful = current_rep.successful_rounds + 1;
    let timestamp = sys_time()?;

    let reputation = NodeReputation {
        node_id: input.node_id.clone(),
        successful_rounds: new_successful,
        failed_rounds: current_rep.failed_rounds,
        reputation_score: new_score,
        last_updated: timestamp.0 as i64 / 1_000_000,
    };

    let node_path = Path::from(format!("node_reputation.{}", input.node_id.clone()));
    let node_entry_hash = ensure_path(node_path, LinkTypes::NodeToReputation)?;

    let existing_links = get_links(
        LinkQuery::new(
            node_entry_hash.clone(),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::NodeToReputation as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let local_hash = if let Some(link) = existing_links.first() {
        if let Some(old_hash) = link.target.clone().into_action_hash() {
            Some(update_entry(old_hash, &EntryTypes::NodeReputation(reputation))?)
        } else {
            None
        }
    } else {
        let hash = create_entry(&EntryTypes::NodeReputation(NodeReputation {
            node_id: input.node_id.clone(),
            successful_rounds: new_successful,
            failed_rounds: current_rep.failed_rounds,
            reputation_score: new_score,
            last_updated: timestamp.0 as i64 / 1_000_000,
        }))?;
        create_link(node_entry_hash, hash.clone(), LinkTypes::NodeToReputation, vec![])?;
        Some(hash)
    };

    let bridge_rep_hash = call_bridge_record_reputation(
        &input.node_id,
        new_score as f64,
        new_successful as u64,
        current_rep.failed_rounds as u64,
        input.evidence_hash,
    )?;

    let event_payload = serde_json::json!({
        "node_id": input.node_id,
        "round": input.round,
        "contribution_quality": input.contribution_quality,
        "new_reputation": new_score,
        "successful_rounds": new_successful,
    });
    let bridge_event_hash = call_bridge_broadcast_event(
        EVENT_ROUND_COMPLETED,
        &event_payload,
        0,
    )?;

    Ok(PositiveReputationResult {
        local_reputation_hash: local_hash,
        bridge_reputation_hash: bridge_rep_hash,
        bridge_event_hash,
        new_reputation_score: new_score,
    })
}

/// Broadcast coordinator election event to Bridge
pub(crate) fn broadcast_coordinator_elected(coordinator_pubkey: &str, election_type: &str) -> ExternResult<Option<ActionHash>> {
    let event_payload = serde_json::json!({
        "coordinator_pubkey": coordinator_pubkey,
        "election_type": election_type,
        "timestamp": sys_time()?.0 as i64 / 1_000_000,
    });

    call_bridge_broadcast_event(
        EVENT_COORDINATOR_ELECTED,
        &event_payload,
        1,
    )
}
