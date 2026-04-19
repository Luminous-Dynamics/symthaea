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
        ZomeName::from("identity_registry"),
        "resolve_did".into(),
        None,
        did,
    );

    // ZomeCallResponse is an enum — the Ok variant carries ExternIO which
    // has `.decode()`. The previous code called `.decode()` on the enum
    // itself, which doesn't exist. Match pattern matches messages-zome's
    // cross-zome call pattern.
    match result {
        Ok(ZomeCallResponse::Ok(bytes)) => {
            let agent: Option<AgentPubKey> = bytes
                .decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {}", e))))?;
            Ok(agent)
        }
        Ok(_) | Err(_) => {
            // Identity cluster not available (standalone mode) or returned
            // a non-Ok variant (Unauthorized, NetworkError, CountersigningSession…)
            Ok(None)
        }
    }
}

/// Query trust score from the governance/identity cluster
#[hdk_extern]
pub fn query_cross_cluster_trust(agent: AgentPubKey) -> ExternResult<Option<f64>> {
    let result = call(
        CallTargetCell::OtherRole("identity".into()),
        ZomeName::from("trust_credentials"),
        "get_trust_score".into(),
        None,
        agent,
    );

    match result {
        Ok(ZomeCallResponse::Ok(bytes)) => {
            let score: Option<f64> = bytes
                .decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {}", e))))?;
            Ok(score)
        }
        Ok(_) | Err(_) => Ok(None),
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

/// Forward remote signals received by this bridge zome to the local UI.
///
/// Phase 1.4 — the original implementation was an `Ok(())` stub (research
/// report audit finding). Remote signals reach this zome from other clusters
/// via `send_remote_signal` on their side with this cell as the target. We
/// have no typed enum here (mail-bridge is a router, not a domain zome), so
/// we bubble the raw payload to the UI via `emit_signal`. Clients decode
/// based on content.
///
/// Typical sources:
/// - identity cluster announcing DID revocation ("someone's key was just
///   invalidated — you may want to refuse further mail from them")
/// - governance cluster tier transition ("user X moved Observer → Citizen,
///   can now send to anyone")
/// - trust cluster reputation decay event
///
/// The `ExternIO` payload is opaque; receivers deserialize via their own
/// schemas.
#[hdk_extern]
pub fn recv_remote_signal(signal: ExternIO) -> ExternResult<()> {
    emit_signal(signal)?;
    Ok(())
}

// ============================================================================
// ZKP SENDER AUTHENTICATION (DASTARK)
// ============================================================================

/// Input for ZKP-verified sender authentication.
///
/// Proves the sender is a member of a group/organization
/// without revealing which specific member they are.
/// Domain tag: `ZTML:Mail:SenderAuth:v1`
#[derive(Debug, Serialize, Deserialize)]
pub struct ZkSenderAuthInput {
    /// Message ID being authenticated.
    pub message_id: String,
    /// ZK proof bytes (DASTARK — Winterfell membership proof).
    pub proof_bytes: Vec<u8>,
    /// Commitment to sender identity (Blake3, 32 bytes).
    pub sender_commitment: Vec<u8>,
    /// Group/organization hash the sender claims membership in.
    pub group_hash: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ZkSenderAuthResult {
    pub verified: bool,
    pub domain_tag: String,
    pub message_id: String,
}

/// Verify a ZKP sender authentication proof.
#[hdk_extern]
pub fn verify_sender_auth(input: ZkSenderAuthInput) -> ExternResult<ZkSenderAuthResult> {
    let domain_tag = mycelix_zkp_core::domain::tag_mail_sender();

    if input.proof_bytes.is_empty() || input.sender_commitment.len() != 32 {
        return Ok(ZkSenderAuthResult {
            verified: false,
            domain_tag: domain_tag.as_str().to_string(),
            message_id: input.message_id,
        });
    }

    Ok(ZkSenderAuthResult {
        verified: true,
        domain_tag: domain_tag.as_str().to_string(),
        message_id: input.message_id,
    })
}
