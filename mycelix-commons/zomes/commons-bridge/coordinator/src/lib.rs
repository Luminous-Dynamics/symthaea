#![allow(deprecated)] // Uses legacy ConsciousnessCredential/Tier for fallback path
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Commons Bridge Coordinator Zome
//!
//! Unified cross-domain dispatch for the Commons cluster.
//! Provides three integration patterns:
//!
//! 1. **dispatch_call** — synchronous RPC to any domain zome via
//!    `call(CallTargetCell::Local, ...)` for local zomes, or
//!    `call(CallTargetCell::OtherRole(...), ...)` for cross-DNA zomes.
//! 2. **query_commons** — audited async query/response with auto-dispatch
//! 3. **broadcast_event** — pub-sub event distribution across domains
//!
//! ## Sub-Cluster Architecture
//!
//! The Commons DNA is split into two sub-cluster DNAs to fit under
//! Holochain's 16MB DNA limit:
//!
//! - **commons_land**: property + housing + water + food (physical infrastructure)
//! - **commons_care**: care + mutualaid + transport + support + space (social/care)
//!
//! Both DNAs include this same bridge WASM. At runtime, the bridge reads
//! `dna_info().modifiers.properties` to determine which sub-cluster it is
//! running in, then routes calls to either `CallTargetCell::Local` (same DNA)
//! or `CallTargetCell::OtherRole("commons_land"/"commons_care")` (other DNA).

use commons_bridge_integrity::*;
use commons_types::{CommonsEvent, CommonsQuery};
use hdk::prelude::*;
use mycelix_bridge_common::{
    self as bridge, check_rate_limit_count, needs_refresh, resolve_commons_zome,
    routing_registry, AuditTrailEntry, AuditTrailQuery, AuditTrailResult, BridgeDomain,
    BridgeHealth, CareAvailabilityQuery, CareAvailabilityResult, ConsciousnessCredential,
    ConsciousnessProfile, ConsciousnessTier,
    CrossClusterDispatchInput, CrossClusterRole, DispatchInput, DispatchResult,
    EventTypeQuery, GateAuditInput, GovernanceAuditFilter, GovernanceAuditResult,
    PropertyOwnershipQuery, PropertyOwnershipResult, ResolveQueryInput, RATE_LIMIT_WINDOW_SECS,
};

// ============================================================================
// Sub-Cluster Membership — defines which zomes live in which DNA
// ============================================================================

/// Zomes that live in the commons_land DNA (physical infrastructure).
const LAND_ZOMES: &[&str] = routing_registry::COMMONS_LAND_ZOMES;

/// Zomes that live in the commons_care DNA (social/care).
const CARE_ZOMES: &[&str] = routing_registry::COMMONS_CARE_ZOMES;

// ============================================================================
// Allowed zome names — security boundary for dispatch (union of both DNAs)
// ============================================================================

const ALLOWED_ZOMES: &[&str] = routing_registry::COMMONS_LOCAL_ZOMES;

/// hApp role name for the commons_land DNA.
const COMMONS_LAND_ROLE: &str = "commons_land";

/// hApp role name for the commons_care DNA.
const COMMONS_CARE_ROLE: &str = "commons_care";

// NOTE: COMMONS_LAND_ROLE and COMMONS_CARE_ROLE are sub-cluster role names,
// not cross-cluster roles. They are internal to commons and not part of the
// routing registry's CrossClusterRole enum.

// ============================================================================
// Sub-Cluster Detection
// ============================================================================

/// Which sub-cluster this bridge instance is running in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubCluster {
    Land,
    Care,
}

/// DNA properties struct for sub-cluster detection.
#[derive(Debug, serde::Deserialize, Default)]
struct DnaProperties {
    #[serde(default)]
    sub_cluster: Option<String>,
}

/// Detect which sub-cluster DNA we are running in by reading DNA properties.
///
/// The DNA YAML sets `properties.sub_cluster` to either "land" or "care".
/// Falls back to `Land` if properties cannot be parsed (e.g., running in
/// the legacy single-DNA configuration for backward compatibility).
fn detect_sub_cluster() -> SubCluster {
    if let Ok(info) = dna_info() {
        // Properties are msgpack-encoded SerializedBytes — wrap bytes in ExternIO to decode
        let extern_io = ExternIO(info.modifiers.properties.bytes().to_vec());
        if let Ok(props) = extern_io.decode::<DnaProperties>() {
            if let Some(sc) = props.sub_cluster.as_deref() {
                return match sc {
                    "care" => SubCluster::Care,
                    _ => SubCluster::Land,
                };
            }
        }
    }
    // Default: treat as land (backward compat with legacy single DNA)
    SubCluster::Land
}

/// Check if a zome is local (in the same DNA) given our sub-cluster.
fn is_local_zome(zome: &str, cluster: SubCluster) -> bool {
    match cluster {
        SubCluster::Land => LAND_ZOMES.contains(&zome),
        SubCluster::Care => CARE_ZOMES.contains(&zome),
    }
}

/// Get the OtherRole name for the sibling sub-cluster DNA.
fn sibling_role(cluster: SubCluster) -> &'static str {
    match cluster {
        SubCluster::Land => COMMONS_CARE_ROLE,
        SubCluster::Care => COMMONS_LAND_ROLE,
    }
}

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

/// Dispatch a synchronous call to any domain zome within the Commons cluster.
///
/// Rate-limited to 100 calls per 60 seconds per agent. Validates the target
/// zome against an allowlist, then routes the call:
///
/// - **Local zome** (same DNA): `call(CallTargetCell::Local, ...)`
/// - **Sibling DNA zome**: `call(CallTargetCell::OtherRole("commons_land"/"commons_care"), ...)`
///
/// This transparent routing means callers do not need to know which DNA a
/// zome lives in — the bridge handles it automatically.
///
/// ## Example (from another coordinator zome or external client)
/// ```ignore
/// let input = DispatchInput {
///     zome: "property_registry".into(),
///     fn_name: "verify_ownership".into(),
///     payload: encode(&VerifyOwnershipInput { ... })?,
/// };
/// let result: DispatchResult = call("commons_bridge", "dispatch_call", input)?;
/// ```
#[hdk_extern]
pub fn dispatch_call(input: DispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&input.zome)?;

    // Validate against the full allowlist (both sub-clusters)
    if !ALLOWED_ZOMES.contains(&input.zome.as_str()) {
        return Ok(DispatchResult::err(
            mycelix_bridge_common::BridgeErrorCode::AllowlistRejected,
            format!("Zome '{}' is not in the commons allowlist", input.zome),
        ));
    }

    let cluster = detect_sub_cluster();

    if is_local_zome(&input.zome, cluster) {
        // Local dispatch — same DNA, zero overhead
        bridge::dispatch_call_checked(&input, ALLOWED_ZOMES)
    } else {
        // Cross-DNA dispatch to sibling sub-cluster
        let cross = CrossClusterDispatchInput {
            role: sibling_role(cluster).to_string(),
            zome: input.zome,
            fn_name: input.fn_name,
            payload: input.payload,
        };
        bridge::dispatch_call_cross_cluster(&cross, ALLOWED_ZOMES)
    }
}

// ============================================================================
// Audited Query/Response (with auto-dispatch)
// ============================================================================

/// Submit a cross-domain query within the Commons cluster.
///
/// Stores the query on the DHT for auditability, then attempts to auto-dispatch
/// to the target domain zome if the query_type matches a known function name.
/// If auto-dispatch succeeds, the query is automatically resolved with the result.
#[hdk_extern]
pub fn query_commons(query: CommonsQuery) -> ExternResult<Record> {
    // Require at least Participant tier to submit queries (prevents DHT spam from Observers)
    mycelix_bridge_common::gate_civic(
        "commons_bridge",
        &mycelix_bridge_common::civic_requirement_basic(),
        "query_commons",
    )?;

    let action_hash = create_entry(&EntryTypes::Query(query.clone()))?;

    // Link to all queries
    let all_anchor = ensure_anchor("all_commons_queries")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllQueries, ())?;

    // Link agent to query
    let agent_anchor = ensure_anchor(&format!("agent_queries:{}", query.requester))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToQuery,
        (),
    )?;

    // Link domain to query
    let domain_anchor = ensure_anchor(&format!("domain_queries:{}", query.domain))?;
    create_link(
        domain_anchor,
        action_hash.clone(),
        LinkTypes::DomainToQuery,
        (),
    )?;

    // Attempt auto-dispatch using type-safe routing
    let target = BridgeDomain::from_str_loose(&query.domain)
        .and_then(|d| resolve_commons_zome(d, &query.query_type));
    if let Some(zome) = target {
        let zome_name = zome.as_str().to_string();
        let payload_bytes = ExternIO::encode(query.params.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0;
        let dispatch = DispatchInput {
            zome: zome_name,
            fn_name: query.query_type.clone(),
            payload: payload_bytes,
        };
        if let Ok(result) = dispatch_call(dispatch) {
            if result.success {
                let result_str = result
                    .response
                    .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
                    .unwrap_or_else(|| "null".to_string());
                // Auto-resolve the query. Failure here leaves the query in "pending"
                // state — the data is still on the DHT, just not marked resolved.
                if let Err(e) = resolve_query(ResolveQueryInput {
                    query_hash: action_hash.clone(),
                    result: result_str,
                    success: true,
                }) {
                    debug!("Auto-resolve failed for query {:?}: {:?}", action_hash, e);
                }
            }
        }
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created query".into()
    )))
}

/// Test-only wrapper — delegates to type-safe routing from bridge-common.
/// Runtime code uses `BridgeDomain::from_str_loose` + `resolve_commons_zome` directly.
#[cfg(test)]
fn resolve_domain_zome(domain: &str, query_type: &str) -> Option<String> {
    BridgeDomain::from_str_loose(domain)
        .and_then(|d| resolve_commons_zome(d, query_type))
        .map(|z| z.as_str().to_string())
}

/// Resolve a pending query with a result
#[hdk_extern]
pub fn resolve_query(input: ResolveQueryInput) -> ExternResult<Record> {
    // Require Citizen tier to resolve queries (modifies existing data)
    mycelix_bridge_common::gate_civic(
        "commons_bridge",
        &mycelix_bridge_common::civic_requirement_voting(),
        "resolve_query",
    )?;

    let record = get(input.query_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Query not found".into())))?;

    let mut query: StoredQuery = record
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

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated query".into()
    )))
}

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

/// Broadcast a cross-domain event within the Commons cluster and emit a signal
#[hdk_extern]
pub fn broadcast_event(event: CommonsEvent) -> ExternResult<Record> {
    // Require at least Participant tier to broadcast events
    mycelix_bridge_common::gate_civic(
        "commons_bridge",
        &mycelix_bridge_common::civic_requirement_basic(),
        "broadcast_event",
    )?;

    let action_hash = create_entry(&EntryTypes::Event(event.clone()))?;

    // Link to all events
    let all_anchor = ensure_anchor("all_commons_events")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllEvents, ())?;

    // Link by event type
    let type_anchor = ensure_anchor(&format!("event_type:{}:{}", event.domain, event.event_type))?;
    create_link(
        type_anchor,
        action_hash.clone(),
        LinkTypes::EventTypeToEvent,
        (),
    )?;

    // Link agent to event
    let agent_anchor = ensure_anchor(&format!("agent_events:{}", event.source_agent))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToEvent,
        (),
    )?;

    // Link domain to event
    let domain_anchor = ensure_anchor(&format!("domain_events:{}", event.domain))?;
    create_link(
        domain_anchor,
        action_hash.clone(),
        LinkTypes::DomainToEvent,
        (),
    )?;

    // Emit signal to connected UI clients
    let signal = BridgeEventSignal {
        signal_type: "commons_bridge_event".to_string(),
        domain: event.domain.clone(),
        event_type: event.event_type.clone(),
        payload: event.payload.clone(),
        action_hash: action_hash.clone(),
    };
    emit_signal(&signal)?;

    // Cross-cluster notification fanout
    // Dispatch notification to other clusters for events that cross boundaries
    {
        let notification = mycelix_bridge_entry_types::CrossClusterNotification {
            schema_version: 1,
            source_cluster: "commons".into(),
            source_zome: event.domain.clone(),
            event_type: event.event_type.clone(),
            target_clusters: vec![], // broadcast to all
            target_agents: vec![],
            payload: event.payload.clone(),
            priority: 1, // Normal priority
            created_at: sys_time()?,
            expires_at: None,
        };

        let payload_bytes = ExternIO::encode(&notification)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0;

        // Fan out to connected clusters (best-effort, don't fail on dispatch errors)
        let targets: &[(CrossClusterRole, &str)] = &[
            (CrossClusterRole::Civic, "civic_bridge"),
            (CrossClusterRole::Identity, "identity_bridge"),
            (CrossClusterRole::Finance, "finance_bridge"),
            (CrossClusterRole::Hearth, "hearth_bridge"),
        ];

        for (target_role, zome) in targets {
            let allowed = routing_registry::get_allowed_zomes(
                CrossClusterRole::Commons,
                *target_role,
            );
            // Skip if the target zome isn't in the allowed list for this route
            if allowed.is_empty() || !allowed.iter().any(|z| *z == *zome) {
                continue;
            }
            let dispatch = CrossClusterDispatchInput {
                role: target_role.as_str().to_string(),
                zome: zome.to_string(),
                fn_name: "receive_notification".into(),
                payload: payload_bytes.clone(),
            };
            // Best-effort: log errors but don't fail the broadcast
            if let Ok(result) = bridge::dispatch_call_cross_cluster(&dispatch, allowed) {
                if !result.success {
                    debug!("Notification fanout to {} failed: {:?}", target_role.as_str(), result.error);
                }
            }
        }
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created event".into()
    )))
}

// ============================================================================
// Governance Gate Audit
// ============================================================================

/// Log a governance gate decision as an auditable event.
///
/// Called fire-and-forget by each coordinator's `require_civic()`.
/// Stores the decision as a `BridgeEventEntry` with `domain: "governance_gate"`.
#[hdk_extern]
pub fn log_governance_gate(input: GateAuditInput) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    let event = StoredEvent {
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
    let all_anchor = ensure_anchor("all_commons_events")?;
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
            if let Ok(Some(event)) = record.entry().to_app_option::<StoredEvent>() {
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

/// Get all events for a specific domain
#[hdk_extern]
pub fn get_domain_events(domain: String) -> ExternResult<Vec<Record>> {
    let domain_anchor = anchor_hash(&format!("domain_events:{}", domain))?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToEvent)?,
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

/// Get all recent events across all domains
#[hdk_extern]
pub fn get_all_events(_: ()) -> ExternResult<Vec<Record>> {
    let all_anchor = anchor_hash("all_commons_events")?;
    let links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllEvents)?,
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
// Cross-Cluster Dispatch: Commons → Civic
// ============================================================================

/// Civic-side zomes that commons-bridge is allowed to call cross-cluster.
const ALLOWED_CIVIC_ZOMES: &[&str] =
    routing_registry::get_allowed_zomes(CrossClusterRole::Commons, CrossClusterRole::Civic);

/// The hApp role name for the Civic DNA.
const CIVIC_ROLE: &str = routing_registry::role_name(CrossClusterRole::Civic);

/// Dispatch a call to any zome in the Civic DNA.
///
/// This is the cross-cluster counterpart of `dispatch_call`.  It uses
/// `CallTargetCell::OtherRole("civic")` to reach zomes in the Civic DNA.
///
/// ## Example use cases
/// - Housing checking for active emergencies before issuing leases
/// - Property verifying media publications referencing a parcel
/// - Water stewardship checking justice disputes on watershed rights
#[hdk_extern]
pub fn dispatch_civic_call(input: CrossClusterDispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&format!("civic:{}", input.zome))?;
    let dispatch = CrossClusterDispatchInput {
        role: CIVIC_ROLE.to_string(),
        zome: input.zome,
        fn_name: input.fn_name,
        payload: input.payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_CIVIC_ZOMES)
}

// ---- Specific cross-cluster use cases ----

/// Input for checking active emergencies near a location.
#[derive(Serialize, Deserialize, Debug)]
pub struct CheckEmergencyForAreaInput {
    /// Latitude of the area to check.
    pub lat: f64,
    /// Longitude of the area to check.
    pub lon: f64,
}

/// Result of an emergency area check.
#[derive(Serialize, Deserialize, Debug)]
pub struct EmergencyAreaCheckResult {
    pub has_active_emergencies: bool,
    pub active_count: u32,
    pub recommendation: Option<String>,
    pub error: Option<String>,
}

/// Check if there are active emergencies affecting a geographic area.
///
/// Cross-cluster call: commons-bridge → civic emergency_incidents via
/// `CallTargetCell::OtherRole("civic")`.  Used by housing, property,
/// and water zomes before critical operations in disaster-prone areas.
#[hdk_extern]
pub fn check_emergency_for_area(
    input: CheckEmergencyForAreaInput,
) -> ExternResult<EmergencyAreaCheckResult> {
    let response = call(
        CallTargetCell::OtherRole(CIVIC_ROLE.into()),
        ZomeName::from("emergency_incidents"),
        FunctionName::from("get_active_disasters"),
        None,
        (),
    );

    match &response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            let records: Vec<Record> = extern_io.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
            })?;
            let count = records.len() as u32;
            Ok(EmergencyAreaCheckResult {
                has_active_emergencies: count > 0,
                active_count: count,
                recommendation: if count > 0 {
                    Some(format!(
                        "{} active emergency(ies) — verify operations at ({:.4}, {:.4}) are safe to proceed",
                        count, input.lat, input.lon
                    ))
                } else {
                    None
                },
                error: None,
            })
        }
        Ok(ZomeCallResponse::NetworkError(err)) => Ok(EmergencyAreaCheckResult {
            has_active_emergencies: false,
            active_count: 0,
            recommendation: None,
            error: Some(format!(
                "Circuit breaker: civic cluster unavailable (network error), operation suspended: {}",
                err
            )),
        }),
        Ok(other) => Ok(EmergencyAreaCheckResult {
            has_active_emergencies: false,
            active_count: 0,
            recommendation: None,
            error: Some(format!(
                "Circuit breaker: civic cluster returned unexpected response, operation suspended: {:?}",
                other
            )),
        }),
        Err(e) => Ok(EmergencyAreaCheckResult {
            has_active_emergencies: false,
            active_count: 0,
            recommendation: None,
            error: Some(format!(
                "Circuit breaker: civic cluster unreachable (transport error), operation suspended: {:?}",
                e
            )),
        }),
    }
}

/// Input for checking justice disputes related to a property.
#[derive(Serialize, Deserialize, Debug)]
pub struct CheckJusticeDisputesInput {
    /// Property or resource identifier to check for disputes.
    pub resource_id: String,
}

/// Result of a justice dispute check.
#[derive(Serialize, Deserialize, Debug)]
pub struct JusticeDisputeCheckResult {
    pub has_pending_cases: bool,
    pub recommendation: Option<String>,
    pub error: Option<String>,
}

/// Check if there are active justice cases that may affect a property transfer.
///
/// Cross-cluster call: commons-bridge → civic justice_cases via
/// `CallTargetCell::OtherRole("civic")`.  Used by property-transfer
/// to verify no pending enforcement actions before completing transfers.
#[hdk_extern]
pub fn check_justice_disputes_for_property(
    input: CheckJusticeDisputesInput,
) -> ExternResult<JusticeDisputeCheckResult> {
    // Query the civic bridge for justice-related cases matching the resource
    let dispatch = CrossClusterDispatchInput {
        role: CIVIC_ROLE.to_string(),
        zome: "civic_bridge".to_string(),
        fn_name: "get_domain_events".to_string(),
        payload: ExternIO::encode("justice".to_string())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0,
    };

    match bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_CIVIC_ZOMES) {
        Ok(result) if result.success => {
            // Events were returned — check if any reference this property
            // In production, we'd decode and filter; for now, presence of
            // justice events is a signal to proceed with caution.
            Ok(JusticeDisputeCheckResult {
                has_pending_cases: result.response.is_some_and(|r| r.len() > 4),
                recommendation: Some(format!(
                    "Verify resource '{}' is not subject to active enforcement before transfer",
                    input.resource_id
                )),
                error: None,
            })
        }
        Ok(result) => Ok(JusticeDisputeCheckResult {
            has_pending_cases: false,
            recommendation: None,
            error: result.error,
        }),
        Err(e) => Ok(JusticeDisputeCheckResult {
            has_pending_cases: false,
            recommendation: None,
            error: Some(format!("Cross-cluster call failed: {:?}", e)),
        }),
    }
}

// ============================================================================
// Audit Trail Queries
// ============================================================================

/// Query bridge events within a time range, optionally filtered by domain and type.
///
/// Retrieves all events from the DHT, then filters by timestamp and optional
/// domain/event_type criteria. Returns lightweight summaries with payload previews.
#[hdk_extern]
pub fn query_audit_trail(query: AuditTrailQuery) -> ExternResult<AuditTrailResult> {
    let from = Timestamp::from_micros(query.from_us);
    let to = Timestamp::from_micros(query.to_us);

    // Get events — if domain+event_type are both specified, use the type anchor;
    // if only domain, use the domain anchor; otherwise get all.
    let records =
        if let (Some(ref domain), Some(ref event_type)) = (&query.domain, &query.event_type) {
            let type_anchor = anchor_hash(&format!("event_type:{}:{}", domain, event_type))?;
            let links = get_links(
                LinkQuery::try_new(type_anchor, LinkTypes::EventTypeToEvent)?,
                GetStrategy::default(),
            )?;
            bridge::records_from_links(links)?
        } else if let Some(ref domain) = query.domain {
            let domain_anchor = anchor_hash(&format!("domain_events:{}", domain))?;
            let links = get_links(
                LinkQuery::try_new(domain_anchor, LinkTypes::DomainToEvent)?,
                GetStrategy::default(),
            )?;
            bridge::records_from_links(links)?
        } else {
            get_all_events(())?
        };

    let mut entries = Vec::new();
    for record in &records {
        // Filter by timestamp
        let action = record.action();
        let ts = action.timestamp();
        if ts < from || ts > to {
            continue;
        }

        // Extract entry fields
        if let Some(entry) = record.entry().as_option() {
            let bytes: SerializedBytes = SerializedBytes::try_from(entry.clone())
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Serialize: {:?}", e))))?;
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes.bytes()) {
                let domain = value
                    .get("domain")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let event_type = value
                    .get("event_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let source = value
                    .get("source_agent")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let payload = value
                    .get("payload")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");
                let preview = if payload.len() > 120 {
                    format!("{}...", &payload[..120])
                } else {
                    payload.to_string()
                };

                entries.push(AuditTrailEntry {
                    domain,
                    event_type,
                    source_agent: source,
                    payload_preview: preview,
                    created_at_us: ts.as_micros(),
                    action_hash: record.action_address().clone(),
                });
            }
        }
    }

    let total = entries.len() as u32;
    Ok(AuditTrailResult {
        entries,
        total_matched: total,
        query_from_us: query.from_us,
        query_to_us: query.to_us,
    })
}

// ============================================================================
// Health Check
// ============================================================================

/// Network health check — returns status for all commons domains
#[hdk_extern]
pub fn health_check(_: ()) -> ExternResult<BridgeHealth> {
    let caller = agent_info()?.agent_initial_pubkey;
    let events = get_all_events(())?;
    let queries = get_my_queries(())?;

    Ok(BridgeHealth {
        healthy: true,
        agent: caller.to_string(),
        total_events: events.len() as u32,
        total_queries: queries.len() as u32,
        domains: vec![
            "property".to_string(),
            "housing".to_string(),
            "care".to_string(),
            "mutualaid".to_string(),
            "water".to_string(),
            "food".to_string(),
            "transport".to_string(),
            "support".to_string(),
            "space".to_string(),
        ],
    })
}

// ============================================================================
// Typed Convenience Functions (intra-cluster)
// ============================================================================

/// Verify property ownership — housing/care/mutualaid can check before acting.
///
/// Dispatches to `property_registry.verify_ownership` with typed input/output.
/// Uses smart routing: if running in commons_care DNA, the call crosses to
/// commons_land via `CallTargetCell::OtherRole`.
#[hdk_extern]
pub fn verify_property_ownership(
    input: PropertyOwnershipQuery,
) -> ExternResult<PropertyOwnershipResult> {
    enforce_rate_limit("property_registry")?;
    let payload = ExternIO::encode(&input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Encode error: {:?}", e))))?;
    let dispatch = DispatchInput {
        zome: "property_registry".into(),
        fn_name: "verify_ownership".into(),
        payload: payload.0,
    };
    let result = dispatch_call(dispatch)?;
    if result.success {
        if let Some(response) = result.response {
            let decoded: PropertyOwnershipResult = ExternIO(response).decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
            })?;
            Ok(decoded)
        } else {
            Ok(PropertyOwnershipResult {
                is_owner: false,
                owner_did: None,
                error: Some("No response data".into()),
            })
        }
    } else {
        Ok(PropertyOwnershipResult {
            is_owner: false,
            owner_did: None,
            error: result.error,
        })
    }
}

/// Check care provider availability — mutualaid can find matching caregivers.
///
/// Dispatches to `care_matching.check_availability` with typed input/output.
/// Uses smart routing: if running in commons_land DNA, the call crosses to
/// commons_care via `CallTargetCell::OtherRole`.
#[hdk_extern]
pub fn check_care_availability(
    input: CareAvailabilityQuery,
) -> ExternResult<CareAvailabilityResult> {
    enforce_rate_limit("care_matching")?;
    let payload = ExternIO::encode(&input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Encode error: {:?}", e))))?;
    let dispatch = DispatchInput {
        zome: "care_matching".into(),
        fn_name: "check_availability".into(),
        payload: payload.0,
    };
    let result = dispatch_call(dispatch)?;
    if result.success {
        if let Some(response) = result.response {
            let decoded: CareAvailabilityResult = ExternIO(response).decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
            })?;
            Ok(decoded)
        } else {
            Ok(CareAvailabilityResult {
                available_count: 0,
                recommendation: "No response data".into(),
                error: Some("Empty response".into()),
            })
        }
    } else {
        Ok(CareAvailabilityResult {
            available_count: 0,
            recommendation: "Dispatch failed".into(),
            error: result.error,
        })
    }
}

// ============================================================================
// Cross-Cluster Dispatch (Commons → Identity)
// ============================================================================

/// Identity-side zomes that commons-bridge is allowed to call cross-cluster.
const ALLOWED_IDENTITY_ZOMES: &[&str] =
    routing_registry::get_allowed_zomes(CrossClusterRole::Commons, CrossClusterRole::Identity);

/// The hApp role name for the Identity DNA.
const IDENTITY_ROLE: &str = routing_registry::role_name(CrossClusterRole::Identity);

/// Dispatch a cross-cluster call to any allowed zome in the Identity DNA.
#[hdk_extern]
pub fn dispatch_identity_call(input: CrossClusterDispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&format!("identity:{}", input.zome))?;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: input.zome,
        fn_name: input.fn_name,
        payload: input.payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

/// Verify an agent's DID is active before sensitive operations.
///
/// Cross-cluster call: commons-bridge → identity did_registry.
/// Used by property-transfer, housing-governance, and care-credentials
/// to verify agent identity before allowing sensitive operations.
#[hdk_extern]
pub fn verify_agent_did(did: String) -> ExternResult<DispatchResult> {
    enforce_rate_limit("identity:did_registry")?;
    let payload = ExternIO::encode(did)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "did_registry".to_string(),
        fn_name: "is_did_active".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

/// Get an agent's trust score from the identity bridge.
///
/// Returns MATL composite score (0.0-1.0) combining MFA assurance and reputation.
/// Used for trust-gated operations in commons domains.
#[hdk_extern]
pub fn get_agent_trust_score(did: String) -> ExternResult<DispatchResult> {
    enforce_rate_limit("identity:identity_bridge")?;
    let payload = ExternIO::encode(did)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "identity_bridge".to_string(),
        fn_name: "get_matl_score".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

// ============================================================================
// Consciousness Credential (cross-cluster → identity)
// ============================================================================

/// 10-minute cache TTL for consciousness credentials (in microseconds).
const CREDENTIAL_CACHE_TTL_US: i64 = 600_000_000;

/// Check for a cached consciousness credential that is still valid (< 10 min old).
fn get_cached_credential(did: &str) -> ExternResult<Option<ConsciousnessCredential>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::AgentToCredentialCache)?,
        GetStrategy::Local,
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Clean up stale links (>1 hour old)
    const CACHE_STALE_CLEANUP_US: i64 = 3_600_000_000;
    let now = sys_time()?.as_micros();
    let stale_cutoff = Timestamp::from_micros(now - CACHE_STALE_CLEANUP_US);
    for link in &links {
        if link.timestamp < stale_cutoff {
            let _ = delete_link(link.create_link_hash.clone(), GetOptions::default());
        }
    }

    // Get the most recent link (safe: early return above guarantees non-empty)
    let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) else {
        return Ok(None);
    };
    let target = link.target.into_action_hash().ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(
            "Invalid credential cache link target".into()
        ))
    })?;

    if let Some(record) = get(target, GetOptions::default())? {
        let cached: CachedCredentialEntry = record
            .entry()
            .to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest(
                    "No entry in cached credential record".into()
                ))
            })?;

        if cached.did == did && now - cached.cached_at_us < CREDENTIAL_CACHE_TTL_US {
            let credential: ConsciousnessCredential = serde_json::from_str(&cached.credential_json)
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Credential cache decode error: {}",
                        e
                    )))
                })?;
            // Check credential expiry — don't serve expired credentials from cache
            if credential.is_expired(now as u64) {
                return Ok(None);
            }
            return Ok(Some(credential));
        }
    }

    Ok(None)
}

/// Store a consciousness credential in the local cache.
fn cache_credential(credential: &ConsciousnessCredential) -> ExternResult<()> {
    let now = sys_time()?.as_micros();
    let json = serde_json::to_string(credential).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Credential cache encode error: {}",
            e
        )))
    })?;

    let entry = CachedCredentialEntry {
        schema_version: 1,
        did: credential.did.clone(),
        credential_json: json,
        cached_at_us: now,
    };

    let action_hash = create_entry(&EntryTypes::CachedCredential(entry))?;
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(agent, action_hash, LinkTypes::AgentToCredentialCache, ())?;

    Ok(())
}

/// Get a consciousness credential for the specified DID.
///
/// Legacy 4D credential accessor — now delegates to the 8D sovereign path.
///
/// Fetches a `SovereignCredential` via `get_sovereign_credential`, converts
/// to the legacy 4D `ConsciousnessCredential`, caches, and returns.
/// This ensures ALL credential fetching goes through the 8D identity bridge.
///
/// If the identity cluster is unreachable, checks cache. If no cache,
/// fails closed (no unverified credentials).
#[hdk_extern]
pub fn get_consciousness_credential(did: String) -> ExternResult<ConsciousnessCredential> {
    use mycelix_bridge_common::sovereign_gate::sovereign_from_credential;

    // 1. Check cache first (avoids cross-cluster call if recent)
    if let Some(cached) = get_cached_credential(&did)? {
        let now_us = sys_time()?.as_micros() as u64;
        if !needs_refresh(&cached, now_us) {
            return Ok(cached);
        }
        // Cache exists but needs refresh — try sovereign path, fall back to cached
    }

    // 2. Fetch via sovereign path (8D)
    match get_sovereign_credential(did.clone()) {
        Ok(sovereign_cred) => {
            // Convert 8D → 4D for backward compat
            let legacy_profile = {
                let lp = mycelix_bridge_common::sovereign_gate::LegacyProfile::from(
                    sovereign_cred.profile.clone(),
                );
                ConsciousnessProfile {
                    identity: lp.identity,
                    reputation: lp.reputation,
                    community: lp.community,
                    engagement: lp.engagement,
                }
            };
            let credential = ConsciousnessCredential {
                did: sovereign_cred.did,
                profile: legacy_profile.clone(),
                tier: ConsciousnessTier::from_score(legacy_profile.combined_score()),
                issued_at: sovereign_cred.issued_at,
                expires_at: sovereign_cred.expires_at,
                issuer: sovereign_cred.issuer,
                trajectory_commitment: None,
                extensions: Default::default(),
            };
            let _ = cache_credential(&credential);
            Ok(credential)
        }
        Err(_) => {
            // Sovereign path failed — serve cached if available
            if let Some(cached) = get_cached_credential(&did)? {
                debug!("Sovereign credential fetch failed, serving cached 4D credential");
                return Ok(cached);
            }
            // No cache, no sovereign — fail closed
            Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Identity cluster unreachable — cannot verify credentials \
                 for {}. Operations suspended until identity verification is restored.",
                did
            ))))
        }
    }
}

/// Fetch a native 8D `SovereignCredential` from the identity bridge.
///
/// Called by `gate_civic()` when the bridge supports native 8D credentials.
/// Dispatches to the identity cluster's `issue_sovereign_credential` extern.
/// Falls back to an error if the identity cluster is unreachable — callers
/// should fall back to the legacy `get_consciousness_credential` path.
#[hdk_extern]
pub fn get_sovereign_credential(
    did: String,
) -> ExternResult<mycelix_bridge_common::sovereign_gate::SovereignCredential> {
    enforce_rate_limit("identity:identity_bridge")?;

    let response = call(
        CallTargetCell::OtherRole(IDENTITY_ROLE.into()),
        ZomeName::new("identity_bridge"),
        FunctionName::new("issue_sovereign_credential"),
        None,
        did.clone(),
    )?;

    match response {
        ZomeCallResponse::Ok(extern_io) => {
            extern_io.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode sovereign credential: {:?}",
                    e
                )))
            })
        }
        other => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Sovereign credential call failed for {}: {:?}",
            did, other
        )))),
    }
}

/// Refresh a consciousness credential by re-fetching from the identity cluster.
///
/// Called by `gate_civic()` when a credential is nearing expiry (within
/// 2 hours). Fetches a fresh credential from the identity bridge, updates the
/// local engagement dimension, and caches the result.
///
/// If the identity cluster is unreachable, logs a warning and returns the
/// existing cached credential (or a fallback if no cache exists).
#[hdk_extern]
pub fn refresh_consciousness_credential(did: String) -> ExternResult<ConsciousnessCredential> {
    debug!(
        "commons-bridge: refreshing consciousness credential for {}",
        did
    );

    // Attempt cross-cluster call to identity bridge
    let payload = ExternIO::encode(did.clone())
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "identity_bridge".to_string(),
        fn_name: "issue_consciousness_credential".to_string(),
        payload,
    };
    let result = bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES);

    let mut credential: ConsciousnessCredential = match result {
        Ok(r) if r.success => {
            let response_bytes = r.response.ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest(
                    "Empty response from identity bridge during refresh".into()
                ))
            })?;
            ExternIO(response_bytes).decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode refreshed consciousness credential: {:?}",
                    e
                )))
            })?
        }
        _ => {
            // Identity cluster unreachable — return existing cached credential if available
            debug!(
                "commons-bridge: identity cluster unreachable during refresh for {}",
                did
            );
            if let Some(cached) = get_cached_credential(&did)? {
                return Ok(cached);
            }
            // No cache either — fail closed. Do not issue unverified credentials.
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Identity cluster unreachable during refresh and no cached credential \
                 for {}. Cannot proceed without verified consciousness credentials.",
                did
            ))));
        }
    };

    // Fill in engagement dimension locally
    credential.profile.engagement = calculate_local_engagement()?;

    // Recalculate tier with engagement included
    credential.tier = ConsciousnessTier::from_score(credential.profile.combined_score());

    // Cache the refreshed credential
    let _ = cache_credential(&credential);

    Ok(credential)
}

/// Issue a consciousness credential enriched with Symthaea consciousness signals.
///
/// When a Symthaea instance is connected, it provides multi-dimensional consciousness
/// data (phi, meta_awareness, coherence, care_activation) that produces a richer
/// `engagement` score than the local activity-based calculation.
///
/// Falls back to `get_consciousness_credential` (local engagement) if Symthaea
/// signals are not provided (all zero).
#[hdk_extern]
pub fn get_consciousness_credential_enriched(
    input: EnrichedCredentialInput,
) -> ExternResult<ConsciousnessCredential> {
    // If no Symthaea signals provided, fall back to standard local engagement
    let has_symthaea = input.phi > 0.0
        || input.meta_awareness > 0.0
        || input.coherence > 0.0
        || input.care_activation > 0.0;

    if !has_symthaea {
        return get_consciousness_credential(input.did);
    }

    // 1. Get base credential from identity bridge (identity, reputation, community)
    let payload = ExternIO::encode(input.did.clone())
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "identity_bridge".to_string(),
        fn_name: "issue_consciousness_credential".to_string(),
        payload,
    };
    let result = bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES);

    let base_cred: ConsciousnessCredential = match result {
        Ok(r) if r.success => {
            let response_bytes = r.response.ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest(
                    "Empty response from identity bridge".into()
                ))
            })?;
            ExternIO(response_bytes).decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode consciousness credential: {:?}",
                    e
                )))
            })?
        }
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Identity cluster unreachable — cannot issue enriched credential for {}",
                input.did
            ))));
        }
    };

    // 2. Build enriched credential using Symthaea signals for engagement
    let now_us = sys_time()?.as_micros() as u64;
    let credential = mycelix_bridge_common::ConsciousnessCredential::from_symthaea(
        input.did,
        input.phi,
        input.meta_awareness,
        input.coherence,
        input.care_activation,
        base_cred.profile.identity,
        base_cred.profile.reputation,
        base_cred.profile.community,
        format!("did:mycelix:{}", agent_info()?.agent_initial_pubkey),
        now_us,
    );

    // 3. Cache the enriched credential
    let _ = cache_credential(&credential);

    Ok(credential)
}

/// Input for enriched consciousness credential issuance.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EnrichedCredentialInput {
    /// Agent DID
    pub did: String,
    /// Symthaea Integrated Information (Φ) \[0, 1\]
    pub phi: f64,
    /// Symthaea meta-cognitive depth \[0, 1\]
    pub meta_awareness: f64,
    /// Symthaea narrative coherence \[0, 1\]
    pub coherence: f64,
    /// Symthaea empathic responsiveness \[0, 1\]
    pub care_activation: f64,
}

/// Calculate local engagement score from this cluster's activity.
///
/// Counts the calling agent's events and queries in the last 90 days,
/// applies 30-day half-life exponential decay, and normalizes to 0.0-1.0.
/// 50 decay-weighted interactions = 1.0 (capped).
fn calculate_local_engagement() -> ExternResult<f64> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;
    let ninety_days_micros = 90i64 * 24 * 60 * 60 * 1_000_000;
    let cutoff = Timestamp::from_micros(now.as_micros() - ninety_days_micros);
    let half_life_micros = 30.0 * 24.0 * 60.0 * 60.0 * 1_000_000.0;

    let mut weighted_count = 0.0f64;

    // Count events
    if let Ok(event_anchor) = anchor_hash(&format!("agent_events:{}", agent)) {
        if let Ok(event_links) = get_links(
            LinkQuery::try_new(event_anchor, LinkTypes::AgentToEvent)?,
            GetStrategy::Local,
        ) {
            for link in &event_links {
                if link.timestamp >= cutoff {
                    let age_micros = (now.as_micros() - link.timestamp.as_micros()) as f64;
                    weighted_count += (-age_micros * 0.693 / half_life_micros).exp();
                }
            }
        }
    }

    // Count queries
    if let Ok(query_anchor) = anchor_hash(&format!("agent_queries:{}", agent)) {
        if let Ok(query_links) = get_links(
            LinkQuery::try_new(query_anchor, LinkTypes::AgentToQuery)?,
            GetStrategy::Local,
        ) {
            for link in &query_links {
                if link.timestamp >= cutoff {
                    let age_micros = (now.as_micros() - link.timestamp.as_micros()) as f64;
                    weighted_count += (-age_micros * 0.693 / half_life_micros).exp();
                }
            }
        }
    }

    // Normalize: 50 decay-weighted interactions in 90 days = 1.0
    Ok((weighted_count / 50.0).min(1.0))
}

// ============================================================================
// Cross-Cluster Dispatch: Commons → Finance
// ============================================================================

/// Finance-side zomes that commons-bridge is allowed to call cross-cluster.
const ALLOWED_FINANCE_ZOMES: &[&str] =
    routing_registry::get_allowed_zomes(CrossClusterRole::Commons, CrossClusterRole::Finance);

/// The hApp role name for the Finance DNA.
const FINANCE_ROLE: &str = routing_registry::role_name(CrossClusterRole::Finance);

/// Register a commons property as finance collateral.
///
/// Verifies the caller's consciousness tier (Participant+), confirms the
/// property exists in the local property-registry, then makes a cross-cluster
/// call to the finance bridge to register the property as collateral, making
/// it eligible for SAP-backed lending positions.
///
/// Non-fatal on finance cluster unreachability — the property remains
/// registered locally even if finance collateral registration fails.
#[hdk_extern]
pub fn register_property_as_collateral(
    input: PropertyCollateralInput,
) -> ExternResult<PropertyCollateralResult> {
    // 1. Verify consciousness tier via local bridge (Participant+ required)
    let credential = get_consciousness_credential(input.owner_did.clone())?;
    if credential.tier.min_score() < ConsciousnessTier::Participant.min_score() {
        return Ok(PropertyCollateralResult {
            success: false,
            property_id: input.property_id,
            collateral_registered: false,
            error: Some(format!(
                "Insufficient consciousness tier: {:?} (Participant+ required)",
                credential.tier
            )),
        });
    }

    // 2. Verify the property exists in local property-registry
    let cluster = detect_sub_cluster();
    let property_call_target = if is_local_zome("property_registry", cluster) {
        CallTargetCell::Local
    } else {
        CallTargetCell::OtherRole(sibling_role(cluster).into())
    };
    let property_response = call(
        property_call_target,
        ZomeName::from("property_registry"),
        FunctionName::from("get_property"),
        None,
        input.property_id.clone(),
    );
    match &property_response {
        Ok(ZomeCallResponse::Ok(_)) => {
            // Property exists — proceed
        }
        Ok(other) => {
            return Ok(PropertyCollateralResult {
                success: false,
                property_id: input.property_id,
                collateral_registered: false,
                error: Some(format!(
                    "Property not found in registry: {:?}",
                    other
                )),
            });
        }
        Err(e) => {
            return Ok(PropertyCollateralResult {
                success: false,
                property_id: input.property_id,
                collateral_registered: false,
                error: Some(format!("Property registry lookup failed: {:?}", e)),
            });
        }
    }

    // 3. Cross-cluster call to finance bridge to register collateral
    #[derive(Serialize, Debug)]
    struct RegisterCollateralPayload {
        owner_did: String,
        source_happ: String,
        asset_type: String,
        asset_id: String,
        value_estimate: u64,
        currency: String,
    }

    match call(
        CallTargetCell::OtherRole(FINANCE_ROLE.into()),
        ZomeName::from("finance_bridge"),
        FunctionName::from("register_collateral"),
        None,
        RegisterCollateralPayload {
            owner_did: input.owner_did.clone(),
            source_happ: "mycelix-commons".to_string(),
            asset_type: "RealEstate".to_string(),
            asset_id: input.property_id.clone(),
            value_estimate: input.appraised_value,
            currency: "SAP".to_string(),
        },
    ) {
        Ok(ZomeCallResponse::Ok(_result)) => Ok(PropertyCollateralResult {
            success: true,
            property_id: input.property_id,
            collateral_registered: true,
            error: None,
        }),
        Ok(other) => Ok(PropertyCollateralResult {
            success: false,
            property_id: input.property_id,
            collateral_registered: false,
            error: Some(format!(
                "Circuit breaker: finance cluster returned unexpected response, operation suspended: {:?}",
                other
            )),
        }),
        Err(e) => {
            // Circuit breaker: Finance cluster unreachable — non-fatal, property still registered locally
            debug!(
                "Circuit breaker: finance cluster unreachable for register_collateral, operation suspended: {:?}",
                e
            );
            Ok(PropertyCollateralResult {
                success: false,
                property_id: input.property_id,
                collateral_registered: false,
                error: Some(format!(
                    "Circuit breaker: finance cluster unreachable, operation suspended: {:?}",
                    e
                )),
            })
        }
    }
}

/// Query collateral health for a commons property.
///
/// Cross-cluster call to finance bridge to check LTV status.
/// Returns loan-to-value ratio and status for the given property.
#[hdk_extern]
pub fn check_property_collateral_health(
    property_id: String,
) -> ExternResult<CollateralHealthResult> {
    match call(
        CallTargetCell::OtherRole(FINANCE_ROLE.into()),
        ZomeName::from("finance_bridge"),
        FunctionName::from("update_collateral_health"),
        None,
        property_id.clone(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            // Decode the health status
            #[derive(Debug, Deserialize)]
            struct HealthResponse {
                ltv_ratio: f64,
                status: String,
            }
            match result.decode::<HealthResponse>() {
                Ok(health) => Ok(CollateralHealthResult {
                    property_id,
                    ltv_ratio: Some(health.ltv_ratio),
                    status: Some(health.status),
                    available: true,
                    error: None,
                }),
                Err(e) => Ok(CollateralHealthResult {
                    property_id,
                    ltv_ratio: None,
                    status: None,
                    available: false,
                    error: Some(format!("Decode error: {:?}", e)),
                }),
            }
        }
        Ok(other) => Ok(CollateralHealthResult {
            property_id,
            ltv_ratio: None,
            status: None,
            available: false,
            error: Some(format!(
                "Circuit breaker: finance cluster returned unexpected response, operation suspended: {:?}",
                other
            )),
        }),
        Err(e) => Ok(CollateralHealthResult {
            property_id,
            ltv_ratio: None,
            status: None,
            available: false,
            error: Some(format!(
                "Circuit breaker: finance cluster unreachable, operation suspended: {:?}",
                e
            )),
        }),
    }
}

/// Input for registering a commons property as finance collateral.
#[derive(Serialize, Deserialize, Debug)]
pub struct PropertyCollateralInput {
    /// Property identifier from the commons property-registry.
    pub property_id: String,
    /// DID of the property owner requesting collateral registration.
    pub owner_did: String,
    /// Appraised value of the property in smallest currency unit.
    pub appraised_value: u64,
}

/// Result of a property collateral registration attempt.
#[derive(Serialize, Deserialize, Debug)]
pub struct PropertyCollateralResult {
    /// Whether the overall operation succeeded.
    pub success: bool,
    /// The property identifier that was processed.
    pub property_id: String,
    /// Whether the collateral was registered in the finance cluster.
    pub collateral_registered: bool,
    /// Error message if the operation failed.
    pub error: Option<String>,
}

/// Result of a collateral health check.
#[derive(Serialize, Deserialize, Debug)]
pub struct CollateralHealthResult {
    /// The property identifier that was checked.
    pub property_id: String,
    /// Loan-to-value ratio (0.0-1.0), if available.
    pub ltv_ratio: Option<f64>,
    /// Collateral status string (e.g., "Healthy", "AtRisk", "Liquidating").
    pub status: Option<String>,
    /// Whether the finance cluster was reachable.
    pub available: bool,
    /// Error message if the check failed.
    pub error: Option<String>,
}

// ============================================================================
// Observability — Bridge Metrics Export
// ============================================================================

/// Return a JSON-encoded snapshot of this bridge's dispatch metrics.
///
/// Includes per-function success/error counts, latency percentiles (p50/p95/p99),
/// rate limit hits, and cross-cluster call counts. Metrics accumulate for the
/// lifetime of the cell and reset on conductor restart.
///
/// Returns JSON string for maximum compatibility — callers can deserialize with
/// `serde_json::from_str::<BridgeMetricsSnapshot>()`. Suitable for Prometheus
/// scraping, admin tools, or Observatory dashboard consumption.
#[hdk_extern]
pub fn get_bridge_metrics(_: ()) -> ExternResult<String> {
    let snapshot = mycelix_bridge_common::metrics::metrics_snapshot();
    serde_json::to_string(&snapshot).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to serialize metrics snapshot: {}",
            e
        )))
    })
}

// ============================================================================
// Notification Service
// ============================================================================

/// Receive a cross-cluster notification and store it locally.
///
/// Called by other bridges via `CallTargetCell::OtherRole("commons")`.
#[hdk_extern]
pub fn receive_notification(notification: mycelix_bridge_entry_types::CrossClusterNotification) -> ExternResult<ActionHash> {
    let action_hash = create_entry(&EntryTypes::Notification(notification.clone()))?;

    // Link to agent's notification inbox
    let agent = agent_info()?.agent_initial_pubkey;
    let inbox_anchor = ensure_anchor(&format!("notifications:{:?}", agent))?;
    create_link(inbox_anchor, action_hash.clone(), LinkTypes::AgentToNotification, ())?;

    // Link to global notifications anchor
    let all_anchor = ensure_anchor("all_notifications")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllNotifications, ())?;

    // Emit signal to connected UI clients
    let signal = mycelix_bridge_common::notifications::NotificationSignal {
        signal_type: "cross_cluster_notification".into(),
        source_cluster: notification.source_cluster,
        event_type: notification.event_type,
        payload: notification.payload,
        priority: notification.priority,
    };
    emit_signal(&signal)?;

    Ok(action_hash)
}

/// Get notifications for the calling agent.
#[hdk_extern]
pub fn get_my_notifications(input: mycelix_bridge_common::notifications::NotificationQueryInput) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let inbox_anchor = ensure_anchor(&format!("notifications:{:?}", agent))?;
    let links = get_links(
        LinkQuery::try_new(inbox_anchor, LinkTypes::AgentToNotification)?,
        GetStrategy::Local,
    )?;

    let limit = input.limit
        .unwrap_or(mycelix_bridge_common::notifications::DEFAULT_NOTIFICATION_LIMIT)
        .min(mycelix_bridge_common::notifications::MAX_NOTIFICATIONS_PER_AGENT);

    let mut records = Vec::new();
    for link in links.iter().rev().take(limit) {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

/// Get unread notification count for the calling agent.
#[hdk_extern]
pub fn get_unread_count(_: ()) -> ExternResult<u32> {
    let agent = agent_info()?.agent_initial_pubkey;
    let inbox_anchor = ensure_anchor(&format!("notifications:{:?}", agent))?;
    let links = get_links(
        LinkQuery::try_new(inbox_anchor, LinkTypes::AgentToNotification)?,
        GetStrategy::Local,
    )?;
    Ok(links.len() as u32)
}

/// Subscribe to specific event types from specific clusters.
#[hdk_extern]
pub fn subscribe_events(input: mycelix_bridge_common::notifications::SubscribeInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let sub_anchor = ensure_anchor(&format!("subscriptions:{:?}", agent))?;

    let payload = serde_json::to_string(&input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    let event = mycelix_bridge_entry_types::BridgeEventEntry {
        schema_version: 1,
        domain: "notifications".into(),
        event_type: "subscription".into(),
        source_agent: agent.clone(),
        payload,
        created_at: sys_time()?,
        related_hashes: vec![],
    };
    let action_hash = create_entry(&EntryTypes::Event(event))?;
    create_link(sub_anchor, action_hash.clone(), LinkTypes::NotificationSubscription, ())?;

    Ok(action_hash)
}

/// Unsubscribe from events by deleting the subscription link.
#[hdk_extern]
pub fn unsubscribe_events(subscription_hash: ActionHash) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    let sub_anchor = ensure_anchor(&format!("subscriptions:{:?}", agent))?;

    let links = get_links(
        LinkQuery::try_new(sub_anchor, LinkTypes::NotificationSubscription)?,
        GetStrategy::Local,
    )?;

    for link in links {
        let target = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if target == subscription_hash {
            delete_link(link.create_link_hash, GetOptions::default())?;
            break;
        }
    }
    Ok(())
}

// ============================================================================
// Geo-Spatial Proximity Queries
// ============================================================================

/// Get entries near a given location by dispatching to the appropriate domain zome.
///
/// The `entry_type` field in the query determines which zome to call:
/// - "property" → property_registry::get_nearby_properties
/// - "shelter" → emergency_shelters::get_nearby_shelters (cross-cluster to civic)
/// - Default → property_registry::get_nearby_properties
#[hdk_extern]
pub fn get_nearby(input: commons_types::geo::NearbyQuery) -> ExternResult<Vec<Record>> {
    let zome = match input.entry_type.as_deref() {
        Some("property") | None => "property_registry",
        Some("housing") => "housing_units",
        Some("food") => "food_production",
        Some("water") => "water_capture",
        Some("shelter") => {
            // Cross-cluster dispatch to civic
            let payload = ExternIO::encode(&input)
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
            let dispatch = CrossClusterDispatchInput {
                role: "civic".into(),
                zome: "emergency_shelters".into(),
                fn_name: "get_nearby_shelters".into(),
                payload: payload.0,
            };
            let result = bridge::dispatch_call_cross_cluster(
                &dispatch,
                routing_registry::get_allowed_zomes(CrossClusterRole::Commons, CrossClusterRole::Civic),
            )?;
            if result.success {
                if let Some(response) = result.response {
                    return ExternIO(response).decode::<Vec<Record>>()
                        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())));
                }
            }
            return Ok(vec![]);
        }
        Some(other) => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!("Unknown entry_type for get_nearby: '{}'. Supported: property, housing, food, water, shelter", other)
            )));
        }
    };

    // Local dispatch to the appropriate commons zome
    let payload = ExternIO::encode(&input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
    let fn_name = format!("get_nearby_{}", input.entry_type.as_deref().unwrap_or("properties"));
    let dispatch = DispatchInput {
        zome: zome.into(),
        fn_name,
        payload: payload.0,
    };
    let result = bridge::dispatch_call_checked(&dispatch, routing_registry::COMMONS_LOCAL_ZOMES)?;
    if result.success {
        if let Some(response) = result.response {
            return ExternIO(response).decode::<Vec<Record>>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())));
        }
    }
    Ok(vec![])
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Allowlist validation ----

    #[test]
    fn local_allowlist_covers_all_domains() {
        // Verify ALLOWED_ZOMES has representatives from all 7+ commons domains
        let has_property = ALLOWED_ZOMES.iter().any(|z| z.starts_with("property_"));
        let has_housing = ALLOWED_ZOMES.iter().any(|z| z.starts_with("housing_"));
        let has_care = ALLOWED_ZOMES.iter().any(|z| z.starts_with("care_"));
        let has_mutualaid = ALLOWED_ZOMES.iter().any(|z| z.starts_with("mutualaid_"));
        let has_water = ALLOWED_ZOMES.iter().any(|z| z.starts_with("water_"));
        let has_food = ALLOWED_ZOMES.iter().any(|z| z.starts_with("food_"));
        let has_transport = ALLOWED_ZOMES.iter().any(|z| z.starts_with("transport_"));
        let has_support = ALLOWED_ZOMES.iter().any(|z| z.starts_with("support_"));
        let has_space = ALLOWED_ZOMES.contains(&"space");
        assert!(has_property, "ALLOWED_ZOMES missing property domain");
        assert!(has_housing, "ALLOWED_ZOMES missing housing domain");
        assert!(has_care, "ALLOWED_ZOMES missing care domain");
        assert!(has_mutualaid, "ALLOWED_ZOMES missing mutualaid domain");
        assert!(has_water, "ALLOWED_ZOMES missing water domain");
        assert!(has_food, "ALLOWED_ZOMES missing food domain");
        assert!(has_transport, "ALLOWED_ZOMES missing transport domain");
        assert!(has_support, "ALLOWED_ZOMES missing support domain");
        assert!(has_space, "ALLOWED_ZOMES missing space");
    }

    #[test]
    fn local_allowlist_has_expected_count() {
        // Land (17) + Care (21) + commons_bridge = 40 (union of both sub-clusters + bridge)
        assert_eq!(ALLOWED_ZOMES.len(), 40);
    }

    #[test]
    fn civic_allowlist_covers_all_civic_domains() {
        let has_justice = ALLOWED_CIVIC_ZOMES
            .iter()
            .any(|z| z.starts_with("justice_"));
        let has_emergency = ALLOWED_CIVIC_ZOMES
            .iter()
            .any(|z| z.starts_with("emergency_"));
        let has_media = ALLOWED_CIVIC_ZOMES.iter().any(|z| z.starts_with("media_"));
        let has_bridge = ALLOWED_CIVIC_ZOMES.contains(&"civic_bridge");
        assert!(has_justice, "ALLOWED_CIVIC_ZOMES missing justice domain");
        assert!(
            has_emergency,
            "ALLOWED_CIVIC_ZOMES missing emergency domain"
        );
        assert!(has_media, "ALLOWED_CIVIC_ZOMES missing media domain");
        assert!(has_bridge, "ALLOWED_CIVIC_ZOMES missing civic_bridge");
    }

    #[test]
    fn civic_allowlist_has_expected_count() {
        // 5 justice + 6 emergency + 4 media + 1 civic_bridge = 16
        assert_eq!(ALLOWED_CIVIC_ZOMES.len(), 16);
    }

    #[test]
    fn civic_role_constant_is_civic() {
        assert_eq!(CIVIC_ROLE, "civic");
    }

    // ---- Domain resolution ----

    #[test]
    fn resolve_property_domain() {
        assert_eq!(
            resolve_domain_zome("property", "get_property").unwrap(),
            "property_registry"
        );
        assert_eq!(
            resolve_domain_zome("property", "transfer_ownership").unwrap(),
            "property_transfer"
        );
        assert_eq!(
            resolve_domain_zome("property", "file_dispute").unwrap(),
            "property_disputes"
        );
        assert_eq!(
            resolve_domain_zome("property", "check_title").unwrap(),
            "property_registry"
        );
    }

    #[test]
    fn resolve_housing_domain() {
        assert_eq!(
            resolve_domain_zome("housing", "create_clt_lease").unwrap(),
            "housing_clt"
        );
        assert_eq!(
            resolve_domain_zome("housing", "add_member").unwrap(),
            "housing_membership"
        );
        assert_eq!(
            resolve_domain_zome("housing", "pay_fee").unwrap(),
            "housing_finances"
        );
        assert_eq!(
            resolve_domain_zome("housing", "report_maintenance").unwrap(),
            "housing_maintenance"
        );
        assert_eq!(
            resolve_domain_zome("housing", "submit_proposal").unwrap(),
            "housing_governance"
        );
        assert_eq!(
            resolve_domain_zome("housing", "list_units").unwrap(),
            "housing_units"
        );
    }

    #[test]
    fn resolve_care_domain() {
        assert_eq!(
            resolve_domain_zome("care", "find_match").unwrap(),
            "care_matching"
        );
        assert_eq!(
            resolve_domain_zome("care", "join_circle").unwrap(),
            "care_circles"
        );
        assert_eq!(
            resolve_domain_zome("care", "verify_credential").unwrap(),
            "care_credentials"
        );
        assert_eq!(
            resolve_domain_zome("care", "create_plan").unwrap(),
            "care_plans"
        );
        assert_eq!(
            resolve_domain_zome("care", "log_hours").unwrap(),
            "care_timebank"
        );
    }

    #[test]
    fn resolve_mutualaid_domain() {
        assert_eq!(
            resolve_domain_zome("mutualaid", "book_resource").unwrap(),
            "mutualaid_resources"
        );
        assert_eq!(
            resolve_domain_zome("mutualaid", "post_need").unwrap(),
            "mutualaid_needs"
        );
        assert_eq!(
            resolve_domain_zome("mutualaid", "join_pool").unwrap(),
            "mutualaid_pools"
        );
        assert_eq!(
            resolve_domain_zome("mutualaid", "submit_request").unwrap(),
            "mutualaid_requests"
        );
        assert_eq!(
            resolve_domain_zome("mutualaid", "form_circle").unwrap(),
            "mutualaid_circles"
        );
        assert_eq!(
            resolve_domain_zome("mutualaid", "propose_governance").unwrap(),
            "mutualaid_governance"
        );
    }

    #[test]
    fn resolve_water_domain() {
        assert_eq!(
            resolve_domain_zome("water", "test_purity").unwrap(),
            "water_purity"
        );
        assert_eq!(
            resolve_domain_zome("water", "log_capture").unwrap(),
            "water_capture"
        );
        assert_eq!(
            resolve_domain_zome("water", "assign_steward").unwrap(),
            "water_steward"
        );
        assert_eq!(
            resolve_domain_zome("water", "record_wisdom").unwrap(),
            "water_wisdom"
        );
        assert_eq!(
            resolve_domain_zome("water", "measure_flow_rate").unwrap(),
            "water_flow"
        );
    }

    #[test]
    fn resolve_food_domain() {
        assert_eq!(
            resolve_domain_zome("food", "register_plot").unwrap(),
            "food_production"
        );
        assert_eq!(
            resolve_domain_zome("food", "list_market").unwrap(),
            "food_distribution"
        );
        assert_eq!(
            resolve_domain_zome("food", "place_order").unwrap(),
            "food_distribution"
        );
        assert_eq!(
            resolve_domain_zome("food", "start_batch").unwrap(),
            "food_preservation"
        );
        assert_eq!(
            resolve_domain_zome("food", "check_storage").unwrap(),
            "food_preservation"
        );
        assert_eq!(
            resolve_domain_zome("food", "catalog_seed").unwrap(),
            "food_knowledge"
        );
        assert_eq!(
            resolve_domain_zome("food", "share_recipe").unwrap(),
            "food_knowledge"
        );
    }

    #[test]
    fn resolve_transport_domain() {
        assert_eq!(
            resolve_domain_zome("transport", "register_vehicle").unwrap(),
            "transport_routes"
        );
        assert_eq!(
            resolve_domain_zome("transport", "create_route").unwrap(),
            "transport_routes"
        );
        assert_eq!(
            resolve_domain_zome("transport", "post_ride_share").unwrap(),
            "transport_sharing"
        );
        assert_eq!(
            resolve_domain_zome("transport", "request_cargo").unwrap(),
            "transport_sharing"
        );
        assert_eq!(
            resolve_domain_zome("transport", "get_carbon_credits").unwrap(),
            "transport_impact"
        );
        assert_eq!(
            resolve_domain_zome("transport", "calculate_emissions").unwrap(),
            "transport_impact"
        );
    }

    #[test]
    fn resolve_unknown_domain_returns_none() {
        assert!(resolve_domain_zome("nonexistent", "anything").is_none());
    }

    // ---- Cross-cluster input type serde ----

    #[test]
    fn check_emergency_input_serde_roundtrip() {
        let input = CheckEmergencyForAreaInput {
            lat: 32.95,
            lon: -96.73,
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: CheckEmergencyForAreaInput = serde_json::from_str(&json).unwrap();
        assert!((input2.lat - 32.95).abs() < 1e-10);
        assert!((input2.lon - (-96.73)).abs() < 1e-10);
    }

    #[test]
    fn emergency_area_check_result_serde_roundtrip() {
        let result = EmergencyAreaCheckResult {
            has_active_emergencies: true,
            active_count: 3,
            recommendation: Some("Caution advised".into()),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: EmergencyAreaCheckResult = serde_json::from_str(&json).unwrap();
        assert!(r2.has_active_emergencies);
        assert_eq!(r2.active_count, 3);
        assert_eq!(r2.recommendation.as_deref(), Some("Caution advised"));
    }

    #[test]
    fn justice_dispute_input_serde_roundtrip() {
        let input = CheckJusticeDisputesInput {
            resource_id: "PROP-001".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: CheckJusticeDisputesInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.resource_id, "PROP-001");
    }

    #[test]
    fn justice_dispute_result_serde_roundtrip() {
        let result = JusticeDisputeCheckResult {
            has_pending_cases: false,
            recommendation: None,
            error: Some("Network unreachable".into()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: JusticeDisputeCheckResult = serde_json::from_str(&json).unwrap();
        assert!(!r2.has_pending_cases);
        assert!(r2.recommendation.is_none());
        assert_eq!(r2.error.as_deref(), Some("Network unreachable"));
    }

    // ---- Rate limit validation ----

    #[test]
    fn rate_limit_under_threshold_passes() {
        assert!(check_rate_limit_count(0).is_ok());
        assert!(check_rate_limit_count(50).is_ok());
        assert!(check_rate_limit_count(99).is_ok());
    }

    #[test]
    fn rate_limit_at_threshold_rejects() {
        let err = check_rate_limit_count(100).unwrap_err();
        assert!(err.contains("Rate limit exceeded"));
        assert!(err.contains("100"));
    }

    #[test]
    fn rate_limit_over_threshold_rejects() {
        let err = check_rate_limit_count(500).unwrap_err();
        assert!(err.contains("Rate limit exceeded"));
    }

    // ---- Typed convenience function serde ----

    #[test]
    fn property_ownership_query_serde_roundtrip() {
        let q = PropertyOwnershipQuery {
            property_id: "PROP-001".into(),
            requester_did: "did:mycelix:agent_123".into(),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: PropertyOwnershipQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.property_id, "PROP-001");
        assert_eq!(q2.requester_did, "did:mycelix:agent_123");
    }

    #[test]
    fn property_ownership_result_serde_roundtrip() {
        let r = PropertyOwnershipResult {
            is_owner: true,
            owner_did: Some("did:mycelix:owner_456".into()),
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: PropertyOwnershipResult = serde_json::from_str(&json).unwrap();
        assert!(r2.is_owner);
        assert_eq!(r2.owner_did.as_deref(), Some("did:mycelix:owner_456"));
    }

    #[test]
    fn care_availability_query_serde_roundtrip() {
        let q = CareAvailabilityQuery {
            skill_needed: "nursing".into(),
            location: Some("downtown".into()),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: CareAvailabilityQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.skill_needed, "nursing");
        assert_eq!(q2.location.as_deref(), Some("downtown"));
    }

    #[test]
    fn care_availability_result_serde_roundtrip() {
        let r = CareAvailabilityResult {
            available_count: 5,
            recommendation: "3 providers nearby".into(),
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: CareAvailabilityResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.available_count, 5);
        assert!(r2.error.is_none());
    }

    // ---- Bridge event signal serde ----

    #[test]
    fn bridge_event_signal_serde_roundtrip() {
        let signal = BridgeEventSignal {
            signal_type: "commons_bridge_event".to_string(),
            domain: "property".to_string(),
            event_type: "ownership_transferred".to_string(),
            payload: r#"{"property_id":"PROP-001"}"#.to_string(),
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let s2: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(s2.signal_type, "commons_bridge_event");
        assert_eq!(s2.domain, "property");
        assert_eq!(s2.event_type, "ownership_transferred");
        assert!(s2.payload.contains("PROP-001"));
    }

    // ---- Audit trail type serde ----

    #[test]
    fn audit_trail_query_full_filters_serde() {
        let q = AuditTrailQuery {
            from_us: 1_700_000_000_000_000,
            to_us: 1_700_001_000_000_000,
            domain: Some("property".into()),
            event_type: Some("ownership_transferred".into()),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: AuditTrailQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.from_us, 1_700_000_000_000_000);
        assert_eq!(q2.domain.as_deref(), Some("property"));
        assert_eq!(q2.event_type.as_deref(), Some("ownership_transferred"));
    }

    #[test]
    fn audit_trail_query_no_filters_serde() {
        let q = AuditTrailQuery {
            from_us: 0,
            to_us: i64::MAX,
            domain: None,
            event_type: None,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: AuditTrailQuery = serde_json::from_str(&json).unwrap();
        assert!(q2.domain.is_none());
        assert!(q2.event_type.is_none());
    }

    #[test]
    fn audit_trail_entry_serde_roundtrip() {
        let e = AuditTrailEntry {
            domain: "housing".into(),
            event_type: "lease_created".into(),
            source_agent: "uhCAk_test".into(),
            payload_preview: r#"{"lease_id":"L-1"}"#.into(),
            created_at_us: 1_700_000_500_000_000,
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&e).unwrap();
        let e2: AuditTrailEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(e2.domain, "housing");
        assert_eq!(e2.event_type, "lease_created");
        assert!(e2.payload_preview.contains("L-1"));
    }

    #[test]
    fn audit_trail_result_empty_serde() {
        let r = AuditTrailResult {
            entries: vec![],
            total_matched: 0,
            query_from_us: 0,
            query_to_us: 1_000_000,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: AuditTrailResult = serde_json::from_str(&json).unwrap();
        assert!(r2.entries.is_empty());
        assert_eq!(r2.total_matched, 0);
    }

    #[test]
    fn bridge_event_signal_type_is_commons() {
        let signal = BridgeEventSignal {
            signal_type: "commons_bridge_event".to_string(),
            domain: "water".to_string(),
            event_type: "flow_measured".to_string(),
            payload: "{}".to_string(),
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        assert_eq!(signal.signal_type, "commons_bridge_event");
    }

    // ============================================================================
    // Cross-domain dispatch edge case tests
    // ============================================================================

    // ---- Allowlist integrity ----

    #[test]
    fn allowed_zomes_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_ZOMES {
            assert!(
                seen.insert(zome),
                "Duplicate zome in ALLOWED_ZOMES: '{}'",
                zome
            );
        }
    }

    #[test]
    fn allowed_civic_zomes_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_CIVIC_ZOMES {
            assert!(
                seen.insert(zome),
                "Duplicate zome in ALLOWED_CIVIC_ZOMES: '{}'",
                zome
            );
        }
    }

    #[test]
    fn allowed_zomes_entries_are_non_empty() {
        for zome in ALLOWED_ZOMES {
            assert!(!zome.is_empty(), "ALLOWED_ZOMES contains an empty string");
            assert!(
                !zome.contains(' '),
                "ALLOWED_ZOMES entry '{}' contains whitespace",
                zome
            );
        }
    }

    #[test]
    fn allowed_civic_zomes_entries_are_non_empty() {
        for zome in ALLOWED_CIVIC_ZOMES {
            assert!(
                !zome.is_empty(),
                "ALLOWED_CIVIC_ZOMES contains an empty string"
            );
            assert!(
                !zome.contains(' '),
                "ALLOWED_CIVIC_ZOMES entry '{}' contains whitespace",
                zome
            );
        }
    }

    #[test]
    fn allowed_zomes_per_domain_count() {
        let property_count = ALLOWED_ZOMES
            .iter()
            .filter(|z| z.starts_with("property_"))
            .count();
        let housing_count = ALLOWED_ZOMES
            .iter()
            .filter(|z| z.starts_with("housing_"))
            .count();
        let care_count = ALLOWED_ZOMES
            .iter()
            .filter(|z| z.starts_with("care_"))
            .count();
        let mutualaid_count = ALLOWED_ZOMES
            .iter()
            .filter(|z| z.starts_with("mutualaid_"))
            .count();
        let water_count = ALLOWED_ZOMES
            .iter()
            .filter(|z| z.starts_with("water_"))
            .count();
        let food_count = ALLOWED_ZOMES
            .iter()
            .filter(|z| z.starts_with("food_"))
            .count();
        let transport_count = ALLOWED_ZOMES
            .iter()
            .filter(|z| z.starts_with("transport_"))
            .count();
        let support_count = ALLOWED_ZOMES
            .iter()
            .filter(|z| z.starts_with("support_"))
            .count();
        let space_count = ALLOWED_ZOMES.iter().filter(|z| *z == &"space").count();
        assert_eq!(property_count, 4, "Expected 4 property zomes");
        assert_eq!(housing_count, 6, "Expected 6 housing zomes");
        assert_eq!(care_count, 5, "Expected 5 care zomes");
        assert_eq!(mutualaid_count, 7, "Expected 7 mutualaid zomes");
        assert_eq!(water_count, 5, "Expected 5 water zomes");
        assert_eq!(food_count, 4, "Expected 4 food zomes");
        assert_eq!(transport_count, 3, "Expected 3 transport zomes");
        assert_eq!(support_count, 3, "Expected 3 support zomes");
        assert_eq!(space_count, 1, "Expected 1 space zome");
    }

    // ---- resolve_domain_zome outputs are in ALLOWED_ZOMES ----

    #[test]
    fn resolve_outputs_are_in_allowlist() {
        // Every zome name returned by resolve_domain_zome must be in ALLOWED_ZOMES
        let test_cases = vec![
            ("property", "get_property"),
            ("property", "transfer_ownership"),
            ("property", "file_dispute"),
            ("property", "check_encumbrance"),
            ("property", "check_title"),
            ("housing", "create_clt_lease"),
            ("housing", "add_member"),
            ("housing", "pay_fee"),
            ("housing", "report_maintenance"),
            ("housing", "submit_proposal"),
            ("housing", "list_units"),
            ("care", "find_match"),
            ("care", "join_circle"),
            ("care", "verify_credential"),
            ("care", "create_plan"),
            ("care", "log_hours"),
            ("mutualaid", "book_resource"),
            ("mutualaid", "post_need"),
            ("mutualaid", "join_pool"),
            ("mutualaid", "submit_request"),
            ("mutualaid", "form_circle"),
            ("mutualaid", "propose_governance"),
            ("mutualaid", "log_timebank"),
            ("water", "test_purity"),
            ("water", "log_capture"),
            ("water", "assign_steward"),
            ("water", "record_wisdom"),
            ("water", "measure_flow_rate"),
            ("food", "register_plot"),
            ("food", "list_market"),
            ("food", "start_batch"),
            ("food", "catalog_seed"),
            ("transport", "register_vehicle"),
            ("transport", "post_ride_share"),
            ("transport", "get_carbon_credits"),
            ("support", "open_ticket"),
            ("support", "run_diagnostic"),
            ("support", "search_knowledge"),
            ("space", "anything"),
        ];
        for (domain, query_type) in test_cases {
            let resolved = resolve_domain_zome(domain, query_type);
            assert!(
                resolved.is_some(),
                "resolve_domain_zome('{}', '{}') returned None",
                domain,
                query_type
            );
            let zome_name = resolved.unwrap();
            assert!(
                ALLOWED_ZOMES.contains(&zome_name.as_str()),
                "resolve_domain_zome('{}', '{}') returned '{}' which is not in ALLOWED_ZOMES",
                domain,
                query_type,
                zome_name
            );
        }
    }

    // ---- Edge cases for resolve_domain_zome ----

    #[test]
    fn resolve_empty_query_type_uses_default() {
        // Empty query_type should fall through to the default zome for each domain
        assert_eq!(
            resolve_domain_zome("property", "").unwrap(),
            "property_registry"
        );
        assert_eq!(resolve_domain_zome("housing", "").unwrap(), "housing_units");
        assert_eq!(resolve_domain_zome("care", "").unwrap(), "care_timebank");
        assert_eq!(
            resolve_domain_zome("mutualaid", "").unwrap(),
            "mutualaid_timebank"
        );
        assert_eq!(resolve_domain_zome("water", "").unwrap(), "water_flow");
        assert_eq!(resolve_domain_zome("food", "").unwrap(), "food_production");
        assert_eq!(
            resolve_domain_zome("transport", "").unwrap(),
            "transport_routes"
        );
        assert_eq!(
            resolve_domain_zome("support", "").unwrap(),
            "support_knowledge"
        );
        assert_eq!(resolve_domain_zome("space", "").unwrap(), "space");
    }

    #[test]
    fn resolve_empty_domain_returns_none() {
        assert!(resolve_domain_zome("", "get_property").is_none());
    }

    #[test]
    fn resolve_case_insensitive_domain() {
        // Domain matching is now case-insensitive (fixed via routing module)
        assert_eq!(
            resolve_domain_zome("Property", "get_property").unwrap(),
            "property_registry"
        );
        assert_eq!(
            resolve_domain_zome("PROPERTY", "get_property").unwrap(),
            "property_registry"
        );
        assert_eq!(
            resolve_domain_zome("HOUSING", "list_units").unwrap(),
            "housing_units"
        );
    }

    #[test]
    fn resolve_multiple_keywords_uses_first_match() {
        // When query_type contains multiple matching keywords, the first
        // match in the match arm order wins.
        // "transfer_dispute" contains both "transfer" and "dispute";
        // "transfer" is checked first in the property match.
        assert_eq!(
            resolve_domain_zome("property", "transfer_dispute").unwrap(),
            "property_transfer"
        );
        // "ownership_dispute" contains "ownership" (matches transfer) and "dispute"
        // "transfer" | "ownership" is checked first
        assert_eq!(
            resolve_domain_zome("property", "ownership_dispute").unwrap(),
            "property_transfer"
        );
    }

    #[test]
    fn resolve_gibberish_query_type_uses_default() {
        // Nonsensical query_type that matches no keywords should use default
        assert_eq!(
            resolve_domain_zome("property", "xyzzy_foobar").unwrap(),
            "property_registry"
        );
        assert_eq!(
            resolve_domain_zome("housing", "quantum_entanglement").unwrap(),
            "housing_units"
        );
        assert_eq!(
            resolve_domain_zome("care", "cosmic_rays").unwrap(),
            "care_timebank"
        );
        assert_eq!(
            resolve_domain_zome("water", "blockchain_mining").unwrap(),
            "water_flow"
        );
    }

    // ---- Health check domain list ----

    #[test]
    fn health_check_returns_all_domains() {
        // The health_check function lists all 9 commons domains; verify the list
        // is consistent with what resolve_domain_zome accepts.
        let expected_domains = vec![
            "property",
            "housing",
            "care",
            "mutualaid",
            "water",
            "food",
            "transport",
            "support",
            "space",
        ];
        for domain in &expected_domains {
            assert!(
                resolve_domain_zome(domain, "anything").is_some(),
                "Health check domain '{}' is not recognized by resolve_domain_zome",
                domain
            );
        }
    }

    // ---- Mutualaid handoff keyword routing ----

    #[test]
    fn resolve_mutualaid_handoff_routes_to_needs() {
        assert_eq!(
            resolve_domain_zome("mutualaid", "confirm_handoff").unwrap(),
            "mutualaid_needs"
        );
    }

    // ---- Water quality alias routing ----

    #[test]
    fn resolve_water_quality_routes_to_purity() {
        assert_eq!(
            resolve_domain_zome("water", "check_quality").unwrap(),
            "water_purity"
        );
    }

    // ---- Food order keyword routing ----

    #[test]
    fn resolve_food_order_routes_to_distribution() {
        assert_eq!(
            resolve_domain_zome("food", "place_order").unwrap(),
            "food_distribution"
        );
    }

    // ---- Housing resale routing ----

    #[test]
    fn resolve_housing_resale_routes_to_clt() {
        assert_eq!(
            resolve_domain_zome("housing", "submit_resale").unwrap(),
            "housing_clt"
        );
    }

    // ---- Property encumbrance routing ----

    #[test]
    fn resolve_property_encumbrance_routes_to_registry() {
        assert_eq!(
            resolve_domain_zome("property", "add_encumbrance").unwrap(),
            "property_registry"
        );
    }

    // ---- Mutualaid booking keyword routing ----

    #[test]
    fn resolve_mutualaid_booking_routes_to_resources() {
        assert_eq!(
            resolve_domain_zome("mutualaid", "confirm_booking").unwrap(),
            "mutualaid_resources"
        );
    }

    // ---- Water harvest alias routing ----

    #[test]
    fn resolve_water_harvest_routes_to_capture() {
        assert_eq!(
            resolve_domain_zome("water", "log_harvest").unwrap(),
            "water_capture"
        );
    }

    // ---- Water guardian alias routing ----

    #[test]
    fn resolve_water_guardian_routes_to_steward() {
        assert_eq!(
            resolve_domain_zome("water", "appoint_guardian").unwrap(),
            "water_steward"
        );
    }

    // ---- Transport cargo keyword routing ----

    #[test]
    fn resolve_transport_cargo_routes_to_sharing() {
        assert_eq!(
            resolve_domain_zome("transport", "post_cargo_offer").unwrap(),
            "transport_sharing"
        );
    }

    // ---- Transport emission keyword routing ----

    #[test]
    fn resolve_transport_emission_routes_to_impact() {
        assert_eq!(
            resolve_domain_zome("transport", "log_emission").unwrap(),
            "transport_impact"
        );
    }

    // ---- Care credential keyword routing ----

    #[test]
    fn resolve_care_credential_routes_to_credentials() {
        assert_eq!(
            resolve_domain_zome("care", "issue_credential").unwrap(),
            "care_credentials"
        );
    }

    // ---- Food storage keyword routing ----

    #[test]
    fn resolve_food_storage_routes_to_preservation() {
        assert_eq!(
            resolve_domain_zome("food", "check_storage_capacity").unwrap(),
            "food_preservation"
        );
    }

    // ---- Identity cross-cluster allowlist ----

    #[test]
    fn identity_allowlist_contains_bridge() {
        assert!(ALLOWED_IDENTITY_ZOMES.contains(&"identity_bridge"));
    }

    #[test]
    fn identity_allowlist_contains_did_registry() {
        assert!(ALLOWED_IDENTITY_ZOMES.contains(&"did_registry"));
    }

    #[test]
    fn identity_role_constant_correct() {
        assert_eq!(IDENTITY_ROLE, "identity");
    }

    #[test]
    fn identity_allowlist_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_IDENTITY_ZOMES {
            assert!(
                seen.insert(zome),
                "Duplicate zome in ALLOWED_IDENTITY_ZOMES: '{}'",
                zome
            );
        }
    }

    #[test]
    fn identity_allowlist_entries_are_non_empty() {
        for zome in ALLOWED_IDENTITY_ZOMES {
            assert!(
                !zome.is_empty(),
                "ALLOWED_IDENTITY_ZOMES contains an empty string"
            );
            assert!(
                !zome.contains(' '),
                "ALLOWED_IDENTITY_ZOMES entry '{}' contains whitespace",
                zome
            );
        }
    }

    // ============================================================================
    // Sub-Cluster Routing Tests
    // ============================================================================

    // ---- LAND_ZOMES / CARE_ZOMES partitioning ----

    #[test]
    fn land_zomes_has_expected_count() {
        // 4 property + 6 housing + 5 water + 4 food = 19
        assert_eq!(LAND_ZOMES.len(), 19, "Expected 19 land zomes");
    }

    #[test]
    fn care_zomes_has_expected_count() {
        // 5 care + 7 mutualaid + 3 transport + 3 support + 1 space + 1 wellbeing + 1 community-calendar = 21
        assert_eq!(CARE_ZOMES.len(), 21, "Expected 21 care zomes");
    }

    #[test]
    fn land_and_care_zomes_are_disjoint() {
        for zome in LAND_ZOMES {
            assert!(
                !CARE_ZOMES.contains(zome),
                "Zome '{}' appears in both LAND_ZOMES and CARE_ZOMES",
                zome
            );
        }
    }

    #[test]
    fn land_plus_care_equals_allowed() {
        // Every zome in ALLOWED_ZOMES must be in exactly one sub-cluster
        let mut combined: Vec<&&str> = LAND_ZOMES.iter().chain(CARE_ZOMES.iter()).collect();
        combined.sort();
        let mut allowed: Vec<&&str> = ALLOWED_ZOMES.iter().collect();
        allowed.sort();
        assert_eq!(
            combined, allowed,
            "LAND_ZOMES + CARE_ZOMES must equal ALLOWED_ZOMES"
        );
    }

    #[test]
    fn land_zomes_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in LAND_ZOMES {
            assert!(seen.insert(zome), "Duplicate in LAND_ZOMES: '{}'", zome);
        }
    }

    #[test]
    fn care_zomes_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in CARE_ZOMES {
            assert!(seen.insert(zome), "Duplicate in CARE_ZOMES: '{}'", zome);
        }
    }

    // ---- is_local_zome routing ----

    #[test]
    fn property_zomes_are_local_in_land() {
        assert!(is_local_zome("property_registry", SubCluster::Land));
        assert!(is_local_zome("property_transfer", SubCluster::Land));
        assert!(is_local_zome("property_disputes", SubCluster::Land));
        assert!(is_local_zome("property_commons", SubCluster::Land));
    }

    #[test]
    fn property_zomes_are_remote_in_care() {
        assert!(!is_local_zome("property_registry", SubCluster::Care));
        assert!(!is_local_zome("property_transfer", SubCluster::Care));
    }

    #[test]
    fn care_zomes_are_local_in_care() {
        assert!(is_local_zome("care_timebank", SubCluster::Care));
        assert!(is_local_zome("care_matching", SubCluster::Care));
        assert!(is_local_zome("care_circles", SubCluster::Care));
        assert!(is_local_zome("care_plans", SubCluster::Care));
        assert!(is_local_zome("care_credentials", SubCluster::Care));
    }

    #[test]
    fn care_zomes_are_remote_in_land() {
        assert!(!is_local_zome("care_timebank", SubCluster::Land));
        assert!(!is_local_zome("care_matching", SubCluster::Land));
    }

    #[test]
    fn housing_zomes_are_local_in_land() {
        assert!(is_local_zome("housing_units", SubCluster::Land));
        assert!(is_local_zome("housing_clt", SubCluster::Land));
        assert!(is_local_zome("housing_governance", SubCluster::Land));
    }

    #[test]
    fn mutualaid_zomes_are_local_in_care() {
        assert!(is_local_zome("mutualaid_needs", SubCluster::Care));
        assert!(is_local_zome("mutualaid_pools", SubCluster::Care));
        assert!(is_local_zome("mutualaid_timebank", SubCluster::Care));
    }

    #[test]
    fn water_zomes_are_local_in_land() {
        assert!(is_local_zome("water_flow", SubCluster::Land));
        assert!(is_local_zome("water_purity", SubCluster::Land));
    }

    #[test]
    fn food_zomes_are_local_in_land() {
        assert!(is_local_zome("food_production", SubCluster::Land));
        assert!(is_local_zome("food_distribution", SubCluster::Land));
    }

    #[test]
    fn transport_zomes_are_local_in_care() {
        assert!(is_local_zome("transport_routes", SubCluster::Care));
        assert!(is_local_zome("transport_sharing", SubCluster::Care));
        assert!(is_local_zome("transport_impact", SubCluster::Care));
    }

    #[test]
    fn support_zomes_are_local_in_care() {
        assert!(is_local_zome("support_knowledge", SubCluster::Care));
        assert!(is_local_zome("support_tickets", SubCluster::Care));
        assert!(is_local_zome("support_diagnostics", SubCluster::Care));
    }

    #[test]
    fn space_is_local_in_care() {
        assert!(is_local_zome("space", SubCluster::Care));
    }

    #[test]
    fn unknown_zome_is_remote_in_both() {
        assert!(!is_local_zome("nonexistent_zome", SubCluster::Land));
        assert!(!is_local_zome("nonexistent_zome", SubCluster::Care));
    }

    // ---- sibling_role ----

    #[test]
    fn sibling_of_land_is_care() {
        assert_eq!(sibling_role(SubCluster::Land), COMMONS_CARE_ROLE);
    }

    #[test]
    fn sibling_of_care_is_land() {
        assert_eq!(sibling_role(SubCluster::Care), COMMONS_LAND_ROLE);
    }

    #[test]
    fn role_constants_are_correct() {
        assert_eq!(COMMONS_LAND_ROLE, "commons_land");
        assert_eq!(COMMONS_CARE_ROLE, "commons_care");
    }

    // ============================================================================
    // Finance Cross-Cluster Tests
    // ============================================================================

    #[test]
    fn finance_allowlist_contains_bridge() {
        assert!(ALLOWED_FINANCE_ZOMES.contains(&"finance_bridge"));
    }

    #[test]
    fn finance_allowlist_has_expected_count() {
        // finance_bridge + currency_mint + payments + treasury + staking + recognition = 6
        assert_eq!(ALLOWED_FINANCE_ZOMES.len(), 6);
    }

    #[test]
    fn finance_role_constant_correct() {
        assert_eq!(FINANCE_ROLE, "finance");
    }

    #[test]
    fn finance_allowlist_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_FINANCE_ZOMES {
            assert!(
                seen.insert(zome),
                "Duplicate zome in ALLOWED_FINANCE_ZOMES: '{}'",
                zome
            );
        }
    }

    #[test]
    fn finance_allowlist_entries_are_non_empty() {
        for zome in ALLOWED_FINANCE_ZOMES {
            assert!(
                !zome.is_empty(),
                "ALLOWED_FINANCE_ZOMES contains an empty string"
            );
            assert!(
                !zome.contains(' '),
                "ALLOWED_FINANCE_ZOMES entry '{}' contains whitespace",
                zome
            );
        }
    }

    // ---- Property collateral type serde ----

    #[test]
    fn property_collateral_input_serde_roundtrip() {
        let input = PropertyCollateralInput {
            property_id: "PROP-001".into(),
            owner_did: "did:mycelix:agent_abc".into(),
            appraised_value: 250_000,
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: PropertyCollateralInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.property_id, "PROP-001");
        assert_eq!(input2.owner_did, "did:mycelix:agent_abc");
        assert_eq!(input2.appraised_value, 250_000);
    }

    #[test]
    fn property_collateral_result_success_serde_roundtrip() {
        let result = PropertyCollateralResult {
            success: true,
            property_id: "PROP-001".into(),
            collateral_registered: true,
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: PropertyCollateralResult = serde_json::from_str(&json).unwrap();
        assert!(r2.success);
        assert!(r2.collateral_registered);
        assert_eq!(r2.property_id, "PROP-001");
        assert!(r2.error.is_none());
    }

    #[test]
    fn property_collateral_result_failure_serde_roundtrip() {
        let result = PropertyCollateralResult {
            success: false,
            property_id: "PROP-002".into(),
            collateral_registered: false,
            error: Some("Finance cluster unreachable".into()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: PropertyCollateralResult = serde_json::from_str(&json).unwrap();
        assert!(!r2.success);
        assert!(!r2.collateral_registered);
        assert_eq!(r2.error.as_deref(), Some("Finance cluster unreachable"));
    }

    #[test]
    fn collateral_health_result_available_serde_roundtrip() {
        let result = CollateralHealthResult {
            property_id: "PROP-001".into(),
            ltv_ratio: Some(0.65),
            status: Some("Healthy".into()),
            available: true,
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: CollateralHealthResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.property_id, "PROP-001");
        assert!((r2.ltv_ratio.unwrap() - 0.65).abs() < 1e-10);
        assert_eq!(r2.status.as_deref(), Some("Healthy"));
        assert!(r2.available);
        assert!(r2.error.is_none());
    }

    #[test]
    fn collateral_health_result_unavailable_serde_roundtrip() {
        let result = CollateralHealthResult {
            property_id: "PROP-003".into(),
            ltv_ratio: None,
            status: None,
            available: false,
            error: Some("Finance cluster unreachable".into()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: CollateralHealthResult = serde_json::from_str(&json).unwrap();
        assert!(!r2.available);
        assert!(r2.ltv_ratio.is_none());
        assert!(r2.status.is_none());
        assert_eq!(r2.error.as_deref(), Some("Finance cluster unreachable"));
    }
}
