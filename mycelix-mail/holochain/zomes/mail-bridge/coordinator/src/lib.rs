#![deny(unsafe_code)]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mail Bridge Coordinator Zome
//!
//! Cross-cluster dispatch for the unified Mycelix hApp.
//! Enables identity resolution, trust propagation, and consciousness gating
//! between mail and other Mycelix clusters.

use hdk::prelude::*;
use mail_bridge_integrity::*;

/// Maximum bridge calls per minute per agent
const BRIDGE_RATE_LIMIT: u32 = 100;

/// Consciousness tier requirements for mail operations
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MailTier {
    /// Can receive email only
    Observer,
    /// Can send to contacts
    Participant,
    /// Can send to anyone, report spam
    Citizen,
    /// Can manage shared mailboxes
    Steward,
    /// Federation administration
    Guardian,
}

/// Dispatch a cross-cluster query
#[hdk_extern]
pub fn dispatch_query(input: MailBridgeQuery) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let query = MailBridgeQuery {
        requester: agent.clone(),
        queried_at: now,
        ..input
    };

    let hash = create_entry(EntryTypes::MailBridgeQuery(query))?;

    create_link(agent, hash.clone(), LinkTypes::AgentToQueries, ())?;

    Ok(hash)
}

/// Resolve a DID to an agent public key via the identity cluster
#[hdk_extern]
pub fn resolve_identity(did: String) -> ExternResult<Option<AgentPubKey>> {
    // Call the identity cluster via OtherRole
    let result = call(
        CallTargetCell::OtherRole("identity".into()),
        "identity_registry".into(),
        "resolve_did".into(),
        None,
        did,
    );

    match result {
        Ok(response) => {
            let agent: Option<AgentPubKey> = response
                .decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {}", e))))?;
            Ok(agent)
        }
        Err(_) => {
            // Identity cluster not available (standalone mode)
            Ok(None)
        }
    }
}

/// Query trust score from the governance/identity cluster
#[hdk_extern]
pub fn query_cross_cluster_trust(agent: AgentPubKey) -> ExternResult<Option<f64>> {
    let result = call(
        CallTargetCell::OtherRole("identity".into()),
        "trust_credentials".into(),
        "get_trust_score".into(),
        None,
        agent,
    );

    match result {
        Ok(response) => {
            let score: Option<f64> = response
                .decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {}", e))))?;
            Ok(score)
        }
        Err(_) => Ok(None),
    }
}

/// Check consciousness tier for a mail operation
#[hdk_extern]
pub fn check_mail_tier(agent: AgentPubKey) -> ExternResult<MailTier> {
    // Try to get consciousness profile from identity cluster
    let trust = query_cross_cluster_trust(agent)?;

    match trust {
        Some(score) if score >= 0.8 => Ok(MailTier::Guardian),
        Some(score) if score >= 0.6 => Ok(MailTier::Steward),
        Some(score) if score >= 0.4 => Ok(MailTier::Citizen),
        Some(score) if score >= 0.3 => Ok(MailTier::Participant),
        _ => Ok(MailTier::Observer),
    }
}

/// Broadcast an event to other clusters
#[hdk_extern]
pub fn broadcast_event(event: MailBridgeEvent) -> ExternResult<ActionHash> {
    let hash = create_entry(EntryTypes::MailBridgeEvent(event))?;
    Ok(hash)
}

// ==================== INIT ====================

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

// ==================== SIGNAL HANDLING ====================

#[hdk_extern]
pub fn recv_remote_signal(_signal: ExternIO) -> ExternResult<()> {
    Ok(())
}
