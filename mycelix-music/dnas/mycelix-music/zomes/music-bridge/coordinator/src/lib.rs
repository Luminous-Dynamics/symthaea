// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Music Bridge Coordinator Zome
//!
//! Unified cross-domain dispatch for the Music cluster.
//! Provides three integration patterns:
//!
//! 1. **dispatch_call** — synchronous RPC to any music zome via
//!    `call(CallTargetCell::Local, ...)`.
//! 2. **query_music** — audited async query/response with auto-dispatch
//! 3. **broadcast_event** — pub-sub event distribution across domains

use hdk::prelude::*;
use music_bridge_integrity::*;
use mycelix_bridge_common::{
    self as bridge, check_rate_limit_count, BridgeHealth, DispatchInput, DispatchResult,
    GateAuditInput, ResolveQueryInput, RATE_LIMIT_WINDOW_SECS,
};

// ============================================================================
// Allowed zome names — security boundary for dispatch
// ============================================================================

const ALLOWED_ZOMES: &[&str] = &[
    "catalog",
    "plays",
    "balances",
    "trust",
];

// ============================================================================
// Helpers
// ============================================================================

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
}

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

// ============================================================================
// Rate Limiting
// ============================================================================

fn enforce_rate_limit(target_zome: &str) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    let anchor = ensure_anchor("dispatch_rate_limit")?;

    let links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::DispatchRateLimit)?,
        GetStrategy::Local,
    )?;

    let now = sys_time()?;
    let window_start_micros = now.as_micros() - (RATE_LIMIT_WINDOW_SECS * 1_000_000);
    let window_start = Timestamp::from_micros(window_start_micros);

    let recent_count = links.iter().filter(|l| l.timestamp >= window_start).count();

    check_rate_limit_count(recent_count).map_err(|msg| wasm_error!(WasmErrorInner::Guest(msg)))?;

    create_link(
        agent,
        anchor,
        LinkTypes::DispatchRateLimit,
        target_zome.as_bytes().to_vec(),
    )?;

    Ok(())
}

// ============================================================================
// Cross-Domain Dispatch (synchronous RPC)
// ============================================================================

/// Dispatch a synchronous call to any domain zome within the Music DNA.
///
/// Rate-limited to 100 calls per 60 seconds per agent. Validates the target
/// zome against an allowlist, then uses `call(CallTargetCell::Local, ...)`.
#[hdk_extern]
pub fn dispatch_call(input: DispatchInput) -> ExternResult<DispatchResult> {
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
    enforce_rate_limit(&input.zome)?;
    bridge::dispatch_call_checked(&input, ALLOWED_ZOMES)
}

// ============================================================================
// Audited Query/Response
// ============================================================================

/// Submit a cross-domain music query.
///
/// Stores the query on the DHT for auditability, then attempts to auto-dispatch
/// to the target domain zome.
#[hdk_extern]
pub fn query_music(query: MusicQueryEntry) -> ExternResult<Record> {
    mycelix_bridge_common::gate_consciousness(
        "music_bridge",
        &mycelix_bridge_common::requirement_for_basic(),
        "query_music",
    )?;

    if query.domain.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Query domain cannot be empty or whitespace-only".into()
        )));
    }
    if query.query_type.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Query type cannot be empty or whitespace-only".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::Query(query.clone()))?;

    let all_anchor = ensure_anchor("all_music_queries")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllQueries, ())?;

    let agent_anchor = ensure_anchor(&format!("agent_queries:{}", query.requester))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToQuery,
        (),
    )?;

    let domain_anchor = ensure_anchor(&format!("domain_queries:{}", query.domain))?;
    create_link(
        domain_anchor,
        action_hash.clone(),
        LinkTypes::DomainToQuery,
        (),
    )?;

    // Attempt auto-dispatch: if domain matches an allowed zome and query_type
    // matches a known function, dispatch directly.
    if ALLOWED_ZOMES.contains(&query.domain.as_str()) {
        let payload_bytes = ExternIO::encode(query.params.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0;
        let dispatch = DispatchInput {
            zome: query.domain.clone(),
            fn_name: query.query_type.clone(),
            payload: payload_bytes,
        };
        if let Ok(result) = dispatch_call(dispatch) {
            if result.success {
                let result_str = result
                    .response
                    .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
                    .unwrap_or_else(|| "null".to_string());
                let _ = resolve_query(ResolveQueryInput {
                    query_hash: action_hash.clone(),
                    result: result_str,
                    success: true,
                });
            }
        }
    }

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created query".into()
    )))
}

/// Resolve a pending query with a result
#[hdk_extern]
pub fn resolve_query(input: ResolveQueryInput) -> ExternResult<Record> {
    mycelix_bridge_common::gate_consciousness(
        "music_bridge",
        &mycelix_bridge_common::requirement_for_voting(),
        "resolve_query",
    )?;

    let record = get(input.query_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Query not found".into())))?;

    let mut query: MusicQueryEntry = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid query entry".into()
        )))?;

    let now = sys_time()?;
    query.result = Some(input.result);
    query.resolved_at = Some(now);
    query.success = Some(input.success);

    let updated_hash = update_entry(input.query_hash, &EntryTypes::Query(query))?;
    get_latest_record(updated_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated query".into()
    )))
}

// ============================================================================
// Event Broadcasting
// ============================================================================

/// Signal payload emitted to connected UI clients when a bridge event is created
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeEventSignal {
    pub signal_type: String,
    pub domain: String,
    pub event_type: String,
    pub payload: String,
    pub action_hash: ActionHash,
}

/// Broadcast a cross-domain event and emit a signal to connected clients
#[hdk_extern]
pub fn broadcast_event(event: MusicEventEntry) -> ExternResult<Record> {
    mycelix_bridge_common::gate_consciousness(
        "music_bridge",
        &mycelix_bridge_common::requirement_for_basic(),
        "broadcast_event",
    )?;

    if event.payload.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Event payload cannot be empty or whitespace-only".into()
        )));
    }
    if event.domain.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Event domain cannot be empty or whitespace-only".into()
        )));
    }
    if event.event_type.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Event type cannot be empty or whitespace-only".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::Event(event.clone()))?;

    let all_anchor = ensure_anchor("all_music_events")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllEvents, ())?;

    let type_anchor = ensure_anchor(&format!("event_type:{}:{}", event.domain, event.event_type))?;
    create_link(
        type_anchor,
        action_hash.clone(),
        LinkTypes::EventTypeToEvent,
        (),
    )?;

    let agent_anchor = ensure_anchor(&format!("agent_events:{}", event.source_agent))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToEvent,
        (),
    )?;

    let domain_anchor = ensure_anchor(&format!("domain_events:{}", event.domain))?;
    create_link(
        domain_anchor,
        action_hash.clone(),
        LinkTypes::DomainToEvent,
        (),
    )?;

    let signal = BridgeEventSignal {
        signal_type: "music_bridge_event".to_string(),
        domain: event.domain.clone(),
        event_type: event.event_type.clone(),
        payload: event.payload.clone(),
        action_hash: action_hash.clone(),
    };
    emit_signal(&signal)?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created event".into()
    )))
}

// ============================================================================
// Governance Gate Audit
// ============================================================================

/// Log a governance gate decision as an auditable event.
#[hdk_extern]
pub fn log_governance_gate(input: GateAuditInput) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    let event = MusicEventEntry {
        schema_version: 1,
        domain: "governance_gate".to_string(),
        event_type: input.action_name.clone(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&input).unwrap_or_default(),
        created_at: sys_time()?,
        related_hashes: vec![],
    };
    let action_hash = create_entry(&EntryTypes::Event(event))?;

    let all_anchor = ensure_anchor("all_music_events")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllEvents, ())?;

    let type_anchor = ensure_anchor(&format!("event_type:governance_gate:{}", input.action_name))?;
    create_link(
        type_anchor,
        action_hash.clone(),
        LinkTypes::EventTypeToEvent,
        (),
    )?;

    let agent_anchor = ensure_anchor(&format!("agent_events:{}", agent))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToEvent,
        (),
    )?;

    let domain_anchor = ensure_anchor("domain_events:governance_gate")?;
    create_link(domain_anchor, action_hash, LinkTypes::DomainToEvent, ())?;

    Ok(())
}

// ============================================================================
// Health Check
// ============================================================================

/// Standard bridge health endpoint
#[hdk_extern]
pub fn health_check(_: ()) -> ExternResult<BridgeHealth> {
    let agent = agent_info()?.agent_initial_pubkey;
    Ok(BridgeHealth {
        cluster: "music".to_string(),
        agent: agent.to_string(),
        zome_count: ALLOWED_ZOMES.len() as u32,
        healthy: true,
    })
}
