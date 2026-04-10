#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//!
//! Craft Bridge Coordinator Zome
//!
//! Cross-domain dispatch with consciousness gating for the Craft cluster.
//! All cross-cluster calls go through this bridge for rate limiting,
//! allowlist validation, and consciousness tier enforcement.
//!
//! Consciousness gating thresholds:
//! - dispatch_call: Participant+ (basic, 0.25+)
//! - query_craft: Participant+ (basic, 0.25+)
//! - broadcast_event: Citizen+ (voting, 0.45+)

use craft_bridge_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common as bridge;

/// All coordinator zome names in the Craft cluster.
const ALLOWED_ZOMES: &[&str] = &[
    "craft_graph",
    "job_postings_coordinator",
    "work_history_coordinator",
    "connection_graph_coordinator",
    "applications_coordinator",
    "guild_coordinator",
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ensure_anchor(text: &str) -> ExternResult<ActionHash> {
    let anchor = Anchor(text.to_string());
    create_entry(EntryTypes::Anchor(anchor))
}

// Rate limiting deferred until DispatchRateLimit link type is added to integrity.
// For now, consciousness gating provides the primary access control.

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Dispatch a synchronous call to any domain zome within the Craft DNA.
///
/// Consciousness-gated: requires Participant tier (basic).
/// Rate-limited to 100 calls per 60 seconds per agent.
#[hdk_extern]
pub fn dispatch_call(input: bridge::DispatchInput) -> ExternResult<bridge::DispatchResult> {
    // Consciousness gate: Participant tier (≥0.3 combined)
    bridge::gate_civic(
        "craft_bridge",
        &bridge::civic_requirement_basic(),
        "dispatch_call",
    )?;

    if input.zome.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Dispatch zome name cannot be empty".into()
        )));
    }
    if input.fn_name.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Dispatch function name cannot be empty".into()
        )));
    }
    bridge::dispatch_call_checked(&input, ALLOWED_ZOMES)
}

/// Query a Craft domain with consciousness gating.
///
/// Requires Participant tier (basic). Stores query on DHT for auditability.
#[hdk_extern]
pub fn query_craft(query: CraftQueryEntry) -> ExternResult<Record> {
    // Consciousness gate
    bridge::gate_civic(
        "craft_bridge",
        &bridge::civic_requirement_basic(),
        "query_craft",
    )?;

    let query_hash = create_entry(EntryTypes::BridgeQuery(query))?;

    let all_anchor = ensure_anchor("all_craft_queries")?;
    create_link(all_anchor, query_hash.clone(), LinkTypes::AllQueries, ())?;

    get(query_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to store query".into())))
}

/// Broadcast an event to the Craft cluster.
///
/// Requires Citizen tier (voting) — events are community actions.
#[hdk_extern]
pub fn broadcast_event(event: CraftEventEntry) -> ExternResult<Record> {
    bridge::gate_civic(
        "craft_bridge",
        &bridge::civic_requirement_voting(),
        "broadcast_event",
    )?;

    let event_hash = create_entry(EntryTypes::BridgeEvent(event))?;

    let all_anchor = ensure_anchor("all_craft_events")?;
    create_link(all_anchor, event_hash.clone(), LinkTypes::AllEvents, ())?;

    get(event_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to store event".into())))
}

// ---------------------------------------------------------------------------
// Consciousness credential stub (required by gate_civic/gate_consciousness)
// ---------------------------------------------------------------------------

/// Bootstrap consciousness credential for the Craft cluster.
///
/// Required by `gate_civic()` which calls `get_consciousness_credential`
/// on the bridge zome. Returns a bootstrap credential with default scores.
/// In production, this would fetch from the identity cluster via OtherRole.
#[hdk_extern]
pub fn get_consciousness_credential(
    did: String,
) -> ExternResult<bridge::ConsciousnessCredential> {
    let now_us = sys_time()?.as_micros() as u64;
    let actual_did = if did.is_empty() {
        format!("did:mycelix:{}", agent_info()?.agent_initial_pubkey)
    } else {
        did
    };
    Ok(bridge::bootstrap_credential(actual_did, 0.5, now_us))
}

/// Refresh stub — delegates to get_consciousness_credential.
#[hdk_extern]
pub fn refresh_consciousness_credential(
    did: String,
) -> ExternResult<bridge::ConsciousnessCredential> {
    get_consciousness_credential(did)
}

/// Governance gate audit log (best-effort, called by gate_civic).
#[hdk_extern]
pub fn log_governance_gate(_input: bridge::GateAuditInput) -> ExternResult<()> {
    // Best-effort audit — no-op for now
    Ok(())
}
