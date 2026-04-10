// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Praxis Bridge Coordinator Zome
//!
//! Unified cross-domain dispatch for the Praxis cluster.
//! Provides three integration patterns:
//!
//! 1. **dispatch_call** — synchronous RPC to any domain zome via
//!    `call(CallTargetCell::Local, ...)`. The core value of clustering.
//! 2. **query_praxis** — audited async query/response with auto-dispatch
//! 3. **broadcast_event** — pub-sub event distribution across domains

#![deny(unsafe_code)]

use praxis_bridge_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{
    self as bridge, check_rate_limit_count, DispatchInput, DispatchResult, EventTypeQuery,
    GateAuditInput, GovernanceAuditFilter, GovernanceAuditResult, ResolveQueryInput,
    RATE_LIMIT_WINDOW_SECS,
};

// ============================================================================
// Allowed zome names — security boundary for dispatch
// ============================================================================

/// All 10 coordinator zome names in the Praxis cluster.
const ALLOWED_ZOMES: &[&str] = &[
    "learning_coordinator",
    "fl_coordinator",
    "credential_coordinator",
    "dao_coordinator",
    "srs_coordinator",
    "gamification_coordinator",
    "adaptive_coordinator",
    "pods_coordinator",
    "knowledge_coordinator",
    "integration_coordinator",
];

// ============================================================================
// Helpers (use zome-specific EntryTypes for anchors)
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

// ============================================================================
// Rate Limiting
// ============================================================================

/// Check rate limit and log the dispatch. Returns error if limit exceeded.
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

    // Log this dispatch
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

/// Dispatch a synchronous call to any domain zome within the Praxis DNA.
///
/// Rate-limited to 100 calls per 60 seconds per agent. Validates the target
/// zome against an allowlist, then uses `call(CallTargetCell::Local, ...)`.

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

#[hdk_extern]
pub fn dispatch_call(input: DispatchInput) -> ExternResult<DispatchResult> {
    // Validate zome and fn_name are not empty
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
// Audited Query/Response (with trust tier gating)
// ============================================================================

/// Submit a cross-domain praxis query.
///
/// Stores the query on the DHT for auditability. Requires at least Participant
/// trust tier (consciousness gating via bridge-common).
#[hdk_extern]
pub fn query_praxis(query: EdunetQueryEntry) -> ExternResult<Record> {
    // Require at least Participant tier to submit queries
    mycelix_bridge_common::gate_civic(
        "praxis_bridge",
        &mycelix_bridge_common::civic_requirement_basic(),
        "query_praxis",
    )?;

    // Reject empty or whitespace-only domain
    if query.domain.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Query domain cannot be empty or whitespace-only".into()
        )));
    }
    // Reject empty or whitespace-only query_type
    if query.query_type.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Query type cannot be empty or whitespace-only".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::Query(query.clone()))?;

    let all_anchor = ensure_anchor("all_praxis_queries")?;
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

    // Attempt auto-dispatch to the target domain zome
    let target_zome = match query.domain.as_str() {
        "learning" => Some("learning_coordinator"),
        "fl" => Some("fl_coordinator"),
        "credential" => Some("credential_coordinator"),
        "dao" => Some("dao_coordinator"),
        "srs" => Some("srs_coordinator"),
        "gamification" => Some("gamification_coordinator"),
        "adaptive" => Some("adaptive_coordinator"),
        "pods" => Some("pods_coordinator"),
        "knowledge" => Some("knowledge_coordinator"),
        "integration" => Some("integration_coordinator"),
        _ => None,
    };

    if let Some(zome_name) = target_zome {
        let payload_bytes = ExternIO::encode(query.params.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0;
        let dispatch = DispatchInput {
            zome: zome_name.to_string(),
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

// ============================================================================
// Query Resolution
// ============================================================================

/// Resolve a pending query with a result
#[hdk_extern]
pub fn resolve_query(input: ResolveQueryInput) -> ExternResult<Record> {
    // Require Citizen tier to resolve queries (modifies existing data)
    mycelix_bridge_common::gate_civic(
        "praxis_bridge",
        &mycelix_bridge_common::civic_requirement_voting(),
        "resolve_query",
    )?;

    let record = get(input.query_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Query not found".into())))?;

    let mut query: EdunetQueryEntry = record
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
    /// Signal type identifier for client-side routing
    pub signal_type: String,
    /// The domain that produced the event
    pub domain: String,
    /// Type of event within the domain
    pub event_type: String,
    /// Serialized event payload
    pub payload: String,
    /// Action hash of the created event entry
    pub action_hash: ActionHash,
}

/// Broadcast a cross-domain event and emit a signal to connected clients
#[hdk_extern]
pub fn broadcast_event(event: EdunetEventEntry) -> ExternResult<Record> {
    // Require at least Participant tier to broadcast events
    mycelix_bridge_common::gate_civic(
        "praxis_bridge",
        &mycelix_bridge_common::civic_requirement_basic(),
        "broadcast_event",
    )?;

    // Reject empty or whitespace-only payloads
    if event.payload.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Event payload cannot be empty or whitespace-only".into()
        )));
    }
    // Reject empty or whitespace-only domain
    if event.domain.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Event domain cannot be empty or whitespace-only".into()
        )));
    }
    // Reject empty or whitespace-only event_type
    if event.event_type.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Event type cannot be empty or whitespace-only".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::Event(event.clone()))?;

    let all_anchor = ensure_anchor("all_praxis_events")?;
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

    // Emit signal to connected UI clients
    let signal = BridgeEventSignal {
        signal_type: "praxis_bridge_event".to_string(),
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
///
/// Called fire-and-forget by each coordinator's `require_consciousness()`.
/// Stores the decision as an `EdunetEventEntry` with `domain: "governance_gate"`.
#[hdk_extern]
pub fn log_governance_gate(input: GateAuditInput) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    let event = EdunetEventEntry {
        schema_version: 1,
        domain: "governance_gate".to_string(),
        event_type: input.action_name.clone(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&input).unwrap_or_default(),
        created_at: sys_time()?,
        related_hashes: vec![],
    };
    let action_hash = create_entry(&EntryTypes::Event(event))?;

    // All events index
    let all_anchor = ensure_anchor("all_praxis_events")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllEvents, ())?;

    // Event type index
    let type_anchor = ensure_anchor(&format!("event_type:governance_gate:{}", input.action_name))?;
    create_link(
        type_anchor,
        action_hash.clone(),
        LinkTypes::EventTypeToEvent,
        (),
    )?;

    // Agent index
    let agent_anchor = ensure_anchor(&format!("agent_events:{}", agent))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToEvent,
        (),
    )?;

    // Domain index
    let domain_anchor = ensure_anchor("domain_events:governance_gate")?;
    create_link(domain_anchor, action_hash, LinkTypes::DomainToEvent, ())?;

    Ok(())
}

/// Query governance gate audit events with filtering.
#[hdk_extern]
pub fn get_governance_audit_trail(
    filter: GovernanceAuditFilter,
) -> ExternResult<GovernanceAuditResult> {
    let domain_anchor = anchor_hash("domain_events:governance_gate")?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToEvent)?,
        GetStrategy::default(),
    )?;
    let records = bridge::records_from_links(links)?;
    let mut entries = Vec::new();
    for record in &records {
        if let Some(_entry) = record.entry().as_option() {
            if let Ok(Some(event)) = record.entry().to_app_option::<EdunetEventEntry>() {
                if let Ok(audit) = serde_json::from_str::<GateAuditInput>(&event.payload) {
                    if let Some(ref action) = filter.action_name {
                        if &audit.action_name != action {
                            continue;
                        }
                    }
                    if let Some(ref zome) = filter.zome_name {
                        if &audit.zome_name != zome {
                            continue;
                        }
                    }
                    if let Some(eligible) = filter.eligible {
                        if audit.eligible != eligible {
                            continue;
                        }
                    }
                    if let Some(from_us) = filter.from_us {
                        let event_us = event.created_at.as_micros();
                        if event_us < from_us {
                            continue;
                        }
                    }
                    if let Some(to_us) = filter.to_us {
                        let event_us = event.created_at.as_micros();
                        if event_us > to_us {
                            continue;
                        }
                    }
                    entries.push(audit);
                }
            }
        }
    }
    let total = entries.len() as u32;
    Ok(GovernanceAuditResult {
        entries,
        total_matched: total,
    })
}

// ============================================================================
// Query helpers
// ============================================================================

/// Get events for a specific domain
#[hdk_extern]
pub fn get_domain_events(domain: String) -> ExternResult<Vec<Record>> {
    let domain_anchor = anchor_hash(&format!("domain_events:{}", domain))?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToEvent)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

/// Get all events
#[hdk_extern]
pub fn get_all_events(_: ()) -> ExternResult<Vec<Record>> {
    let all_anchor = anchor_hash("all_praxis_events")?;
    let links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllEvents)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

/// Get events by type within a domain
#[hdk_extern]
pub fn get_events_by_type(query: EventTypeQuery) -> ExternResult<Vec<Record>> {
    let type_anchor = anchor_hash(&format!("event_type:{}:{}", query.domain, query.event_type))?;
    let links = get_links(
        LinkQuery::try_new(type_anchor, LinkTypes::EventTypeToEvent)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

/// Get all queries for a specific domain
#[hdk_extern]
pub fn get_domain_queries(domain: String) -> ExternResult<Vec<Record>> {
    let domain_anchor = anchor_hash(&format!("domain_queries:{}", domain))?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToQuery)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

/// Get my queries
#[hdk_extern]
pub fn get_my_queries(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let agent_anchor = anchor_hash(&format!("agent_queries:{}", caller))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToQuery)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

/// Get my events
#[hdk_extern]
pub fn get_my_events(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let agent_anchor = anchor_hash(&format!("agent_events:{}", caller))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToEvent)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

// ============================================================================
// Bridge Health
// ============================================================================

/// Return bridge health status.
#[hdk_extern]
pub fn get_bridge_health(_: ()) -> ExternResult<bridge::BridgeHealth> {
    let agent = agent_info()?.agent_initial_pubkey;
    // Count events and queries from their respective anchors
    let event_links = get_links(
        LinkQuery::try_new(anchor_hash("all_praxis_events")?, LinkTypes::AllEvents)?,
        GetStrategy::Local,
    )
    .unwrap_or_default();
    let query_links = get_links(
        LinkQuery::try_new(anchor_hash("all_praxis_queries")?, LinkTypes::AllQueries)?,
        GetStrategy::Local,
    )
    .unwrap_or_default();

    Ok(bridge::BridgeHealth {
        healthy: true,
        agent: agent.to_string(),
        total_events: event_links.len() as u32,
        total_queries: query_links.len() as u32,
        domains: vec![
            "learning".into(),
            "fl".into(),
            "credential".into(),
            "dao".into(),
            "srs".into(),
            "gamification".into(),
            "adaptive".into(),
            "pods".into(),
            "knowledge".into(),
            "integration".into(),
        ],
    })
}

// ============================================================================
// TEND Payout Bridge — Learning Becomes Economic Activity
// ============================================================================
//
// Converts high-quality learning events into TEND mutual credit.
//
// Formula: P_tend = T_hours × (B_rate + ΔPoL × reputation_factor)
// - B_rate = 0.25 TEND/hr base
// - Quality bonus scaled by quality_permille
// - Reputation factor from consciousness profile (Reputational Burn guardrail)
//
// GUARDRAILS:
// - TEND formula in coordinator only (NOT integrity — clock determinism)
// - Rate limited: max 4 TEND events per day per agent
// - Non-blocking: learning event succeeds even if TEND dispatch fails
// - Cap: max 2.0 TEND per event

/// Maximum TEND credits per single learning event
const MAX_TEND_PER_EVENT: f32 = 2.0;

/// Base rate: TEND per hour of study
const BASE_TEND_RATE: f32 = 0.25;

/// Maximum TEND-generating events per day per agent
const MAX_TEND_EVENTS_PER_DAY: u32 = 4;

/// Minimum quality permille for TEND eligibility
const MIN_QUALITY_FOR_TEND: u16 = 600;

/// Minimum duration in seconds (15 minutes)
const MIN_DURATION_FOR_TEND: u32 = 900;

/// Input for triggering a learning TEND credit.
#[derive(Serialize, Deserialize, Debug)]
pub struct LearningTendInput {
    /// Quality of the learning session (0-1000 permille)
    pub quality_permille: u16,
    /// Duration in seconds
    pub duration_seconds: u32,
    /// Consciousness phi level (0-1000 permille, optional — from frontend)
    pub phi_permille: Option<u16>,
    /// Agent's reputation permille (0-1000, from consciousness profile)
    pub reputation_permille: Option<u16>,
}

/// Result of TEND credit computation.
#[derive(Serialize, Deserialize, Debug)]
pub struct LearningTendResult {
    /// TEND credits earned (0.0 if not eligible)
    pub tend_credits: f32,
    /// Whether the agent hit the daily rate limit
    pub rate_limited: bool,
    /// Breakdown of the computation
    pub breakdown: String,
}

/// Compute TEND credits for a learning event.
///
/// Does NOT create the actual TEND exchange (that requires cross-hApp call
/// to the Finance cluster). Returns the computed amount for the frontend
/// to display and/or dispatch.
#[hdk_extern]
pub fn compute_learning_tend(input: LearningTendInput) -> ExternResult<LearningTendResult> {
    // Check minimums
    if input.quality_permille < MIN_QUALITY_FOR_TEND {
        return Ok(LearningTendResult {
            tend_credits: 0.0,
            rate_limited: false,
            breakdown: format!(
                "Quality {} < minimum {} permille",
                input.quality_permille, MIN_QUALITY_FOR_TEND
            ),
        });
    }

    if input.duration_seconds < MIN_DURATION_FOR_TEND {
        return Ok(LearningTendResult {
            tend_credits: 0.0,
            rate_limited: false,
            breakdown: format!(
                "Duration {}s < minimum {}s",
                input.duration_seconds, MIN_DURATION_FOR_TEND
            ),
        });
    }

    // Rate limit check: count today's TEND events via links
    let agent = agent_info()?.agent_initial_pubkey;
    let today_anchor = {
        let now = sys_time()?;
        let day = now.as_micros() / (86_400 * 1_000_000); // Day number
        format!("tend_daily.{}.{}", agent, day)
    };

    // Check existing daily count via simple link counting
    let tend_path = Path::from(today_anchor.clone());
    let typed = tend_path.typed(LinkTypes::DispatchRateLimit)?;
    typed.ensure()?;
    let daily_anchor = typed.path.path_entry_hash()?;

    let daily_links = get_links(
        LinkQuery::try_new(daily_anchor.clone(), LinkTypes::DispatchRateLimit)?,
        GetStrategy::Local,
    )?;

    if daily_links.len() as u32 >= MAX_TEND_EVENTS_PER_DAY {
        return Ok(LearningTendResult {
            tend_credits: 0.0,
            rate_limited: true,
            breakdown: format!(
                "Daily limit reached ({}/{})",
                daily_links.len(),
                MAX_TEND_EVENTS_PER_DAY
            ),
        });
    }

    // Compute TEND credits
    let hours = input.duration_seconds as f32 / 3600.0;
    let quality_factor = input.quality_permille as f32 / 1000.0;

    // Base credit: hours × base rate
    let base_credit = hours * BASE_TEND_RATE;

    // Quality bonus: hours × quality × 0.5
    let quality_bonus = hours * quality_factor * 0.5;

    // Reputation factor (Reputational Burn guardrail):
    // High reputation = more economic breath. Low/new = reduced rate.
    let reputation = input.reputation_permille.unwrap_or(500) as f32 / 1000.0;
    let reputation_multiplier = 0.5 + reputation * 0.5; // Range: 0.5x to 1.0x

    // Phi bonus (optional): focused consciousness adds small bonus
    let phi_bonus = input
        .phi_permille
        .map(|p| (p as f32 / 1000.0) * 0.1 * hours)
        .unwrap_or(0.0);

    let total = ((base_credit + quality_bonus + phi_bonus) * reputation_multiplier)
        .min(MAX_TEND_PER_EVENT);

    // Record the daily count (link to anchor for rate limiting)
    let agent_hash: AnyDhtHash = agent.into();
    create_link(
        daily_anchor,
        agent_hash,
        LinkTypes::DispatchRateLimit,
        vec![],
    )?;

    Ok(LearningTendResult {
        tend_credits: total,
        rate_limited: false,
        breakdown: format!(
            "hours={:.2} base={:.3} quality_bonus={:.3} phi={:.3} rep_mult={:.2} total={:.3}",
            hours, base_credit, quality_bonus, phi_bonus, reputation_multiplier, total
        ),
    })
}
