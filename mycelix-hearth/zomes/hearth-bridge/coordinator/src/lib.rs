//! Hearth Bridge Coordinator Zome
//!
//! Unified gateway for cross-domain and cross-cluster communication in the
//! Hearth (Family/Household) cluster. Provides three integration patterns:
//!
//! 1. **dispatch_call** -- synchronous RPC to any hearth domain zome
//! 2. **query_hearth** -- audited async query/response with DHT persistence
//! 3. **broadcast_event** -- event distribution to connected clients
//!
//! Cross-cluster callers (Personal, Identity, Commons, Civic) reach this
//! bridge via `CallTargetCell::OtherRole("hearth")`.

use hdk::prelude::*;
use hearth_bridge_integrity::*;
use hearth_coordinator_common::{decode_zome_response, get_latest_record};
use hearth_types::{
    BondUpdate, CareSummary, DigestEpochInput, GratitudeSummary, RhythmSummary, SeveranceInput,
    SeveranceSummaryData, WeeklyDigest,
};
use mycelix_bridge_common::{
    self as bridge, check_rate_limit_count, BridgeHealth, CrossClusterDispatchInput, DispatchInput,
    DispatchResult, EventTypeQuery, ResolveQueryInput, RATE_LIMIT_WINDOW_SECS,
    ConsciousnessCredential, ConsciousnessTier, GateAuditInput,
    GovernanceAuditFilter, GovernanceAuditResult,
    needs_refresh,
};

// ============================================================================
// Allowed zome names -- security boundary for dispatch
// ============================================================================

const ALLOWED_ZOMES: &[&str] = &[
    "hearth_kinship",
    "hearth_gratitude",
    "hearth_stories",
    "hearth_care",
    "hearth_autonomy",
    "hearth_emergency",
    "hearth_decisions",
    "hearth_resources",
    "hearth_milestones",
    "hearth_rhythms",
];

// Cross-cluster dispatch targets (what hearth bridge can call in other clusters)
const ALLOWED_PERSONAL_ZOMES: &[&str] = &["personal_bridge"];
const ALLOWED_IDENTITY_ZOMES: &[&str] = &["identity_bridge", "did_registry", "recovery"];
const ALLOWED_COMMONS_ZOMES: &[&str] = &["commons_bridge"];
const ALLOWED_CIVIC_ZOMES: &[&str] = &["civic_bridge"];

// ============================================================================
// Domain-to-zome resolution
// ============================================================================

fn resolve_zome(domain: &str) -> Option<&'static str> {
    match domain {
        "kinship" => Some("hearth_kinship"),
        "gratitude" => Some("hearth_gratitude"),
        "stories" => Some("hearth_stories"),
        "care" => Some("hearth_care"),
        "autonomy" => Some("hearth_autonomy"),
        "emergency" => Some("hearth_emergency"),
        "decisions" => Some("hearth_decisions"),
        "resources" => Some("hearth_resources"),
        "milestones" => Some("hearth_milestones"),
        "rhythms" => Some("hearth_rhythms"),
        _ => None,
    }
}

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

/// Dispatch a synchronous call to any domain zome within the Hearth DNA.
#[hdk_extern]
pub fn dispatch_call(input: DispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&input.zome)?;
    bridge::dispatch_call_checked(&input, ALLOWED_ZOMES)
}

// ============================================================================
// Audited Query/Response
// ============================================================================

/// Submit a cross-domain hearth query.
/// Creates a BridgeQuery entry on the DHT, links it for indexing,
/// and attempts auto-dispatch to the resolved zome.
#[hdk_extern]
pub fn query_hearth(input: DispatchInput) -> ExternResult<Record> {
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Determine domain from zome name or use zome as domain
    let domain = ALLOWED_ZOMES
        .iter()
        .find(|z| **z == input.zome)
        .map(|z| z.replace("hearth_", ""))
        .unwrap_or_else(|| input.zome.clone());

    let query = HearthQueryEntry {
        domain: domain.clone(),
        query_type: input.fn_name.clone(),
        requester: caller.clone(),
        params: String::from_utf8_lossy(&input.payload).to_string(),
        result: None,
        created_at: now,
        resolved_at: None,
        success: None,
    };

    let action_hash = create_entry(&EntryTypes::BridgeQuery(query))?;

    let all_anchor = ensure_anchor("all_hearth_queries")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllQueries, ())?;

    let agent_anchor = ensure_anchor(&format!("agent_queries:{}", caller))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToQuery,
        (),
    )?;

    let domain_anchor = ensure_anchor(&format!("domain_queries:{}", domain))?;
    create_link(
        domain_anchor,
        action_hash.clone(),
        LinkTypes::DomainToQuery,
        (),
    )?;

    // Attempt auto-dispatch
    if let Some(zome_name) = resolve_zome(&domain) {
        let dispatch = DispatchInput {
            zome: zome_name.to_string(),
            fn_name: input.fn_name,
            payload: input.payload,
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created query".into()
    )))
}

// ============================================================================
// Query Resolution
// ============================================================================

/// Resolve a pending query with a result.
#[hdk_extern]
pub fn resolve_query(input: ResolveQueryInput) -> ExternResult<Record> {
    let record = get(input.query_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Query not found".into())))?;

    let mut query: HearthQueryEntry = record
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

    let updated_hash = update_entry(input.query_hash, &EntryTypes::BridgeQuery(query))?;
    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated query".into()
    )))
}

// ============================================================================
// Event Broadcasting
// ============================================================================

/// Signal payload emitted to connected UI clients.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeEventSignal {
    pub signal_type: String,
    pub domain: String,
    pub event_type: String,
    pub payload: String,
    pub action_hash: ActionHash,
}

/// Broadcast a cross-domain event and emit a signal to connected clients.
#[hdk_extern]
pub fn broadcast_event(input: DispatchInput) -> ExternResult<Record> {
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let domain = ALLOWED_ZOMES
        .iter()
        .find(|z| **z == input.zome)
        .map(|z| z.replace("hearth_", ""))
        .unwrap_or_else(|| input.zome.clone());

    let event = HearthEventEntry {
        domain: domain.clone(),
        event_type: input.fn_name.clone(),
        source_agent: caller.clone(),
        payload: String::from_utf8_lossy(&input.payload).to_string(),
        created_at: now,
        related_hashes: vec![],
    };

    let action_hash = create_entry(&EntryTypes::BridgeEvent(event.clone()))?;

    let all_anchor = ensure_anchor("all_hearth_events")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllEvents, ())?;

    let agent_anchor = ensure_anchor(&format!("agent_events:{}", caller))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToEvent,
        (),
    )?;

    let domain_anchor = ensure_anchor(&format!("domain_events:{}", domain))?;
    create_link(
        domain_anchor,
        action_hash.clone(),
        LinkTypes::DomainToEvent,
        (),
    )?;

    let signal = BridgeEventSignal {
        signal_type: "hearth_bridge_event".to_string(),
        domain: event.domain,
        event_type: event.event_type,
        payload: event.payload,
        action_hash: action_hash.clone(),
    };
    emit_signal(&signal)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created event".into()
    )))
}

/// Log a governance gate decision as an auditable event.
///
/// Called fire-and-forget by each coordinator's `require_consciousness()`.
/// Stores the decision as a `HearthEventEntry` with `domain: "governance_gate"`.
#[hdk_extern]
pub fn log_governance_gate(input: GateAuditInput) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    let event = HearthEventEntry {
        domain: "governance_gate".to_string(),
        event_type: input.action_name.clone(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&input).unwrap_or_default(),
        created_at: sys_time()?,
        related_hashes: vec![],
    };
    let action_hash = create_entry(&EntryTypes::BridgeEvent(event))?;

    // All events index
    let all_anchor = ensure_anchor("all_hearth_events")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllEvents, ())?;

    // Event type index
    let type_anchor = ensure_anchor(&format!("event_type:governance_gate:{}", input.action_name))?;
    create_link(type_anchor, action_hash.clone(), LinkTypes::EventTypeToEvent, ())?;

    // Agent index
    let agent_anchor = ensure_anchor(&format!("agent_events:{}", agent))?;
    create_link(agent_anchor, action_hash.clone(), LinkTypes::AgentToEvent, ())?;

    // Domain index
    let domain_anchor = ensure_anchor("domain_events:governance_gate")?;
    create_link(domain_anchor, action_hash, LinkTypes::DomainToEvent, ())?;

    Ok(())
}

/// Query governance gate audit events with filtering.
#[hdk_extern]
pub fn get_governance_audit_trail(filter: GovernanceAuditFilter) -> ExternResult<GovernanceAuditResult> {
    let domain_anchor = anchor_hash("domain_events:governance_gate")?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToEvent)?,
        GetStrategy::default(),
    )?;
    let records = bridge::records_from_links(links)?;
    let mut entries = Vec::new();
    for record in &records {
        if let Some(_entry) = record.entry().as_option() {
            if let Ok(Some(event)) = record.entry().to_app_option::<HearthEventEntry>() {
                if let Ok(audit) = serde_json::from_str::<GateAuditInput>(&event.payload) {
                    if let Some(ref action) = filter.action_name {
                        if &audit.action_name != action { continue; }
                    }
                    if let Some(ref zome) = filter.zome_name {
                        if &audit.zome_name != zome { continue; }
                    }
                    if let Some(eligible) = filter.eligible {
                        if audit.eligible != eligible { continue; }
                    }
                    if let Some(from_us) = filter.from_us {
                        let event_us = event.created_at.as_micros();
                        if event_us < from_us { continue; }
                    }
                    if let Some(to_us) = filter.to_us {
                        let event_us = event.created_at.as_micros();
                        if event_us > to_us { continue; }
                    }
                    entries.push(audit);
                }
            }
        }
    }
    let total = entries.len() as u32;
    Ok(GovernanceAuditResult { entries, total_matched: total })
}

// ============================================================================
// Query Helpers
// ============================================================================

#[hdk_extern]
pub fn get_domain_events(domain: String) -> ExternResult<Vec<Record>> {
    let domain_anchor = anchor_hash(&format!("domain_events:{}", domain))?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToEvent)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

#[hdk_extern]
pub fn get_all_events(_: ()) -> ExternResult<Vec<Record>> {
    let all_anchor = anchor_hash("all_hearth_events")?;
    let links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllEvents)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

#[hdk_extern]
pub fn get_events_by_type(query: EventTypeQuery) -> ExternResult<Vec<Record>> {
    // Reuse domain-based indexing since we store by domain
    let domain_anchor = anchor_hash(&format!("domain_events:{}", query.domain))?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToEvent)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

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

// ============================================================================
// Cross-Cluster Dispatch
// ============================================================================

const PERSONAL_ROLE: &str = "personal";
const IDENTITY_ROLE: &str = "identity";
const COMMONS_ROLE: &str = "commons";
const CIVIC_ROLE: &str = "civic";

/// Dispatch a call to the Personal cluster.
#[hdk_extern]
pub fn dispatch_personal_call(input: CrossClusterDispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&format!("personal:{}", input.zome))?;
    let dispatch = CrossClusterDispatchInput {
        role: PERSONAL_ROLE.to_string(),
        zome: input.zome,
        fn_name: input.fn_name,
        payload: input.payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_PERSONAL_ZOMES)
}

/// Dispatch a call to the Identity cluster.
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

/// Dispatch a call to the Commons cluster.
#[hdk_extern]
pub fn dispatch_commons_call(input: CrossClusterDispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&format!("commons:{}", input.zome))?;
    let dispatch = CrossClusterDispatchInput {
        role: String::new(), // resolved by dispatch_call_cross_cluster_commons
        zome: input.zome,
        fn_name: input.fn_name,
        payload: input.payload,
    };
    bridge::dispatch_call_cross_cluster_commons(&dispatch, ALLOWED_COMMONS_ZOMES)
}

/// Dispatch a call to the Civic cluster.
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

// ============================================================================
// Typed Convenience Functions
// ============================================================================

/// Verify a member's identity via the Identity cluster.
#[hdk_extern]
pub fn verify_member_identity(agent: AgentPubKey) -> ExternResult<DispatchResult> {
    enforce_rate_limit("identity:identity_bridge")?;
    let payload = ExternIO::encode(agent)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "identity_bridge".to_string(),
        fn_name: "verify_identity".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

/// Escalate an emergency alert to the Civic cluster's emergency system.
#[hdk_extern]
pub fn escalate_emergency(alert_data: String) -> ExternResult<DispatchResult> {
    enforce_rate_limit("civic:civic_bridge")?;
    let payload = ExternIO::encode(alert_data)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: CIVIC_ROLE.to_string(),
        zome: "civic_bridge".to_string(),
        fn_name: "escalate_emergency".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_CIVIC_ZOMES)
}

/// Query the timebank balance for an agent from the Commons cluster.
#[hdk_extern]
pub fn query_timebank_balance(agent: AgentPubKey) -> ExternResult<DispatchResult> {
    enforce_rate_limit("commons:commons_bridge")?;
    let payload = ExternIO::encode(agent)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: COMMONS_ROLE.to_string(),
        zome: "commons_bridge".to_string(),
        fn_name: "query_timebank_balance".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_COMMONS_ZOMES)
}

// ============================================================================
// Severance (H3: Coming-of-age data migration)
// ============================================================================

/// Initiate a severance process for a member leaving or coming of age.
/// Records a SeveranceSummaryData event on the DHT and dispatches
/// a personal vault export via the Personal cluster.
#[hdk_extern]
pub fn initiate_severance(input: SeveranceInput) -> ExternResult<Record> {
    enforce_rate_limit("kinship:severance")?;
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Build the severance summary
    let summary = SeveranceSummaryData {
        hearth_hash: input.hearth_hash.clone(),
        member: caller.clone(),
        milestones_exported: if input.export_milestones { 1 } else { 0 },
        care_records_exported: if input.export_care_history { 1 } else { 0 },
        bond_snapshot_exported: input.export_bond_snapshot,
        new_role: input.new_role.clone(),
        completed_at: now,
    };

    // Record as a bridge event for audit trail
    let event_payload = serde_json::to_string(&summary)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    let event = HearthEventEntry {
        domain: "kinship".to_string(),
        event_type: "severance_initiated".to_string(),
        source_agent: caller.clone(),
        payload: event_payload,
        created_at: now,
        related_hashes: vec![],
    };

    let action_hash = create_entry(&EntryTypes::BridgeEvent(event))?;

    let all_anchor = ensure_anchor("all_hearth_events")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllEvents, ())?;

    let domain_anchor = ensure_anchor("domain_events:kinship")?;
    create_link(
        domain_anchor,
        action_hash.clone(),
        LinkTypes::DomainToEvent,
        (),
    )?;

    // Dispatch personal vault export via cross-cluster call
    let export_payload = ExternIO::encode(&input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let personal_dispatch = CrossClusterDispatchInput {
        role: PERSONAL_ROLE.to_string(),
        zome: "personal_bridge".to_string(),
        fn_name: "export_member_data".to_string(),
        payload: export_payload,
    };
    // Best-effort -- don't fail the severance if the personal export fails
    let _ = bridge::dispatch_call_cross_cluster(&personal_dispatch, ALLOWED_PERSONAL_ZOMES);

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created severance event".into()
    )))
}

// ============================================================================
// Epoch Sync (H2: Weekly digest rollup)
// ============================================================================

/// Input for hearth epoch sync.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HearthSyncInput {
    pub hearth_hash: ActionHash,
    pub epoch_start: Timestamp,
    pub epoch_end: Timestamp,
}

/// Decode a typed response from a DispatchResult.
fn decode_dispatch_response<T: serde::de::DeserializeOwned + std::fmt::Debug>(
    result: &DispatchResult,
    context: &str,
) -> ExternResult<T> {
    if !result.success {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "{} dispatch failed: {}",
            context,
            result.error.as_deref().unwrap_or("unknown error")
        ))));
    }
    let response_bytes = result.response.as_ref().ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "{} dispatch returned no response",
            context
        )))
    })?;
    let extern_io = ExternIO(response_bytes.clone());
    extern_io.decode().map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to decode {} response: {}",
            context, e
        )))
    })
}

/// Trigger epoch sync for a hearth: calls digest creation on gratitude,
/// care, and rhythm zomes, queries bond snapshots, assembles a WeeklyDigest,
/// and stores it via kinship.
#[hdk_extern]
pub fn hearth_sync(input: HearthSyncInput) -> ExternResult<Record> {
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let epoch_input = DigestEpochInput {
        hearth_hash: input.hearth_hash.clone(),
        epoch_start: input.epoch_start,
        epoch_end: input.epoch_end,
    };
    let epoch_payload = ExternIO::encode(epoch_input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;

    // Call create_gratitude_digest
    let gratitude_dispatch = DispatchInput {
        zome: "hearth_gratitude".to_string(),
        fn_name: "create_gratitude_digest".to_string(),
        payload: epoch_payload.clone(),
    };
    let gratitude_result = bridge::dispatch_call_checked(&gratitude_dispatch, ALLOWED_ZOMES)?;
    let gratitude_summary: Vec<GratitudeSummary> =
        decode_dispatch_response(&gratitude_result, "gratitude_digest")?;

    // Call create_care_digest
    let care_dispatch = DispatchInput {
        zome: "hearth_care".to_string(),
        fn_name: "create_care_digest".to_string(),
        payload: epoch_payload.clone(),
    };
    let care_result = bridge::dispatch_call_checked(&care_dispatch, ALLOWED_ZOMES)?;
    let care_summary: Vec<CareSummary> = decode_dispatch_response(&care_result, "care_digest")?;

    // Call create_rhythm_digest
    let rhythm_dispatch = DispatchInput {
        zome: "hearth_rhythms".to_string(),
        fn_name: "create_rhythm_digest".to_string(),
        payload: epoch_payload,
    };
    let rhythm_result = bridge::dispatch_call_checked(&rhythm_dispatch, ALLOWED_ZOMES)?;
    let rhythm_summary: Vec<RhythmSummary> =
        decode_dispatch_response(&rhythm_result, "rhythm_digest")?;

    // Get bond snapshots via cross-zome call to kinship
    let bond_response = call(
        CallTargetCell::Local,
        ZomeName::new("hearth_kinship"),
        FunctionName::new("get_bond_snapshots"),
        None,
        input.hearth_hash.clone(),
    )?;
    let bond_updates: Vec<BondUpdate> = decode_zome_response(bond_response, "get_bond_snapshots")?;

    // Assemble the WeeklyDigest
    let digest = WeeklyDigest {
        hearth_hash: input.hearth_hash,
        epoch_start: input.epoch_start,
        epoch_end: input.epoch_end,
        bond_updates,
        care_summary,
        gratitude_summary,
        rhythm_summary,
        created_by: caller,
        created_at: now,
    };

    // Store via kinship's create_weekly_digest
    let digest_response = call(
        CallTargetCell::Local,
        ZomeName::new("hearth_kinship"),
        FunctionName::new("create_weekly_digest"),
        None,
        digest,
    )?;
    decode_zome_response(digest_response, "create_weekly_digest")
}

/// Get all weekly digests for a hearth (delegates to kinship).
#[hdk_extern]
pub fn get_weekly_digests(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("hearth_kinship"),
        FunctionName::new("get_weekly_digests"),
        None,
        hearth_hash,
    )?;
    decode_zome_response(response, "get_weekly_digests")
}

// ============================================================================
// Admin
// ============================================================================

/// Health check for the hearth bridge.
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
            "kinship".to_string(),
            "gratitude".to_string(),
            "stories".to_string(),
            "care".to_string(),
            "autonomy".to_string(),
            "emergency".to_string(),
            "decisions".to_string(),
            "resources".to_string(),
            "milestones".to_string(),
            "rhythms".to_string(),
        ],
    })
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
    let now_cleanup = sys_time()?.as_micros();
    let stale_cutoff = Timestamp::from_micros(now_cleanup - CACHE_STALE_CLEANUP_US);
    for link in &links {
        if link.timestamp < stale_cutoff {
            let _ = delete_link(link.create_link_hash.clone(), GetOptions::default());
        }
    }

    let link = links.into_iter().max_by_key(|l| l.timestamp)
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No credential cache links found".into())))?;
    let target = link.target.into_action_hash().ok_or_else(||
        wasm_error!(WasmErrorInner::Guest("Invalid credential cache link target".into())))?;

    if let Some(record) = get_latest_record(target)? {
        let cached: CachedCredentialEntry = record.entry()
            .to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No entry in cached credential record".into())))?;

        if cached.did == did {
            let now = sys_time()?.as_micros();
            if now - cached.cached_at_us < CREDENTIAL_CACHE_TTL_US {
                let credential: ConsciousnessCredential = serde_json::from_str(&cached.credential_json)
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!(
                        "Credential cache decode error: {}", e
                    ))))?;
                // Check credential expiry — don't serve expired credentials from cache
                if credential.is_expired(now as u64) {
                    return Ok(None);
                }
                return Ok(Some(credential));
            }
        }
    }

    Ok(None)
}

/// Store a consciousness credential in the local cache.
fn cache_credential(credential: &ConsciousnessCredential) -> ExternResult<()> {
    let now = sys_time()?.as_micros();
    let json = serde_json::to_string(credential)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!(
            "Credential cache encode error: {}", e
        ))))?;

    let entry = CachedCredentialEntry {
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
/// First checks the local cache (10-minute TTL). On cache miss, makes a
/// cross-cluster call to `identity_bridge.issue_consciousness_credential` to
/// obtain identity, reputation, and community dimensions, then fills in the
/// engagement dimension locally from this cluster's activity data.
/// The result is cached for subsequent calls.
#[hdk_extern]
pub fn get_consciousness_credential(did: String) -> ExternResult<ConsciousnessCredential> {
    // 1. Check cache first (avoids cross-cluster call if recent)
    if let Some(cached) = get_cached_credential(&did)? {
        // Proactive refresh — attempt inline refresh, fall back to cached if it fails
        let now_us = sys_time()?.as_micros() as u64;
        if needs_refresh(&cached, now_us) {
            debug!("Credential nearing expiry, attempting proactive refresh for {}", cached.did);
            if let Ok(ZomeCallResponse::Ok(response)) = call(
                CallTargetCell::OtherRole(IDENTITY_ROLE.into()),
                ZomeName::new("identity_bridge"),
                FunctionName::new("refresh_consciousness_credential"),
                None,
                cached.did.clone(),
            ) {
                if let Ok(refreshed) = response.decode::<ConsciousnessCredential>() {
                    let _ = cache_credential(&refreshed);
                    return Ok(refreshed);
                }
            }
            // Refresh failed — serve cached credential (still valid, just nearing expiry)
        }
        return Ok(cached);
    }

    enforce_rate_limit("identity:identity_bridge")?;

    // 2. Cross-cluster call to identity: issue_consciousness_credential
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
            // Identity role unavailable (single-DNA mode or network partition).
            // Return a permissive fallback credential so single-cluster operations
            // can proceed. In production, the identity role will provide real scores.
            debug!("Identity role unavailable, using fallback consciousness credential for {}", did);
            let now_us = sys_time()?.as_micros() as u64;
            ConsciousnessCredential::from_unified_consciousness(
                did.clone(),
                0.5,   // unified_consciousness — mid-range default
                0.5,   // identity — above Participant threshold (0.25)
                0.5,   // reputation
                0.5,   // community
                "did:mycelix:hearth-bridge-fallback".to_string(),
                now_us,
            )
        }
    };

    // 3. Fill in engagement dimension locally
    credential.profile.engagement = calculate_local_engagement()?;

    // 4. Recalculate tier with engagement included
    credential.tier = ConsciousnessTier::from_score(credential.profile.combined_score());

    // 5. Cache the result for subsequent calls
    let _ = cache_credential(&credential);

    // Proactive refresh check (covers edge case: identity issued a short-lived credential)
    let now_us = sys_time()?.as_micros() as u64;
    if needs_refresh(&credential, now_us) {
        debug!("Freshly-issued credential nearing expiry, attempting proactive refresh for {}", credential.did);
        if let Ok(ZomeCallResponse::Ok(response)) = call(
            CallTargetCell::OtherRole(IDENTITY_ROLE.into()),
            ZomeName::new("identity_bridge"),
            FunctionName::new("refresh_consciousness_credential"),
            None,
            credential.did.clone(),
        ) {
            if let Ok(refreshed) = response.decode::<ConsciousnessCredential>() {
                let _ = cache_credential(&refreshed);
                return Ok(refreshed);
            }
        }
    }

    Ok(credential)
}

/// Refresh a consciousness credential by re-fetching from the identity cluster.
///
/// Called by `gate_consciousness()` when a credential is nearing expiry (within
/// 2 hours). Fetches a fresh credential from the identity bridge, updates the
/// local engagement dimension, and caches the result.
///
/// If the identity cluster is unreachable, logs a warning and returns the
/// existing cached credential (or a fallback if no cache exists).
#[hdk_extern]
pub fn refresh_consciousness_credential(did: String) -> ExternResult<ConsciousnessCredential> {
    debug!("hearth-bridge: refreshing consciousness credential for {}", did);

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
            debug!("hearth-bridge: identity cluster unreachable during refresh for {}", did);
            if let Some(cached) = get_cached_credential(&did)? {
                return Ok(cached);
            }
            // No cache either — return Observer-tier fallback
            let now_us = sys_time()?.as_micros() as u64;
            ConsciousnessCredential::from_unified_consciousness(
                did.clone(),
                0.0, 0.0, 0.0, 0.0,
                "did:mycelix:hearth-bridge-fallback".to_string(),
                now_us,
            )
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

/// Calculate local engagement score from this cluster's activity.
fn calculate_local_engagement() -> ExternResult<f64> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;
    let ninety_days_micros = 90i64 * 24 * 60 * 60 * 1_000_000;
    let cutoff = Timestamp::from_micros(now.as_micros() - ninety_days_micros);
    let half_life_micros = 30.0 * 24.0 * 60.0 * 60.0 * 1_000_000.0;

    let mut weighted_count = 0.0f64;

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

    Ok((weighted_count / 50.0).min(1.0))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use hearth_types::MemberRole;

    // ---- Allowlist validation ----

    #[test]
    fn local_allowlist_covers_all_hearth_domains() {
        assert!(ALLOWED_ZOMES.contains(&"hearth_kinship"));
        assert!(ALLOWED_ZOMES.contains(&"hearth_gratitude"));
        assert!(ALLOWED_ZOMES.contains(&"hearth_stories"));
        assert!(ALLOWED_ZOMES.contains(&"hearth_care"));
        assert!(ALLOWED_ZOMES.contains(&"hearth_autonomy"));
        assert!(ALLOWED_ZOMES.contains(&"hearth_emergency"));
        assert!(ALLOWED_ZOMES.contains(&"hearth_decisions"));
        assert!(ALLOWED_ZOMES.contains(&"hearth_resources"));
        assert!(ALLOWED_ZOMES.contains(&"hearth_milestones"));
        assert!(ALLOWED_ZOMES.contains(&"hearth_rhythms"));
    }

    #[test]
    fn local_allowlist_has_expected_count() {
        assert_eq!(ALLOWED_ZOMES.len(), 10);
    }

    #[test]
    fn allowed_zomes_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_ZOMES {
            assert!(seen.insert(zome), "Duplicate in ALLOWED_ZOMES: '{}'", zome);
        }
    }

    #[test]
    fn allowed_zomes_entries_are_non_empty() {
        for zome in ALLOWED_ZOMES {
            assert!(!zome.is_empty());
            assert!(!zome.contains(' '));
        }
    }

    // ---- Cross-cluster allowlists ----

    #[test]
    fn personal_allowlist_contains_bridge() {
        assert!(ALLOWED_PERSONAL_ZOMES.contains(&"personal_bridge"));
    }

    #[test]
    fn identity_allowlist_contains_expected() {
        assert!(ALLOWED_IDENTITY_ZOMES.contains(&"identity_bridge"));
        assert!(ALLOWED_IDENTITY_ZOMES.contains(&"did_registry"));
        assert!(ALLOWED_IDENTITY_ZOMES.contains(&"recovery"));
    }

    #[test]
    fn commons_allowlist_contains_bridge() {
        assert!(ALLOWED_COMMONS_ZOMES.contains(&"commons_bridge"));
    }

    #[test]
    fn civic_allowlist_contains_bridge() {
        assert!(ALLOWED_CIVIC_ZOMES.contains(&"civic_bridge"));
    }

    #[test]
    fn role_constants_correct() {
        assert_eq!(PERSONAL_ROLE, "personal");
        assert_eq!(IDENTITY_ROLE, "identity");
        assert_eq!(COMMONS_ROLE, "commons");
        assert_eq!(CIVIC_ROLE, "civic");
    }

    // ---- Domain resolution ----

    #[test]
    fn resolve_all_hearth_domains() {
        assert_eq!(resolve_zome("kinship"), Some("hearth_kinship"));
        assert_eq!(resolve_zome("gratitude"), Some("hearth_gratitude"));
        assert_eq!(resolve_zome("stories"), Some("hearth_stories"));
        assert_eq!(resolve_zome("care"), Some("hearth_care"));
        assert_eq!(resolve_zome("autonomy"), Some("hearth_autonomy"));
        assert_eq!(resolve_zome("emergency"), Some("hearth_emergency"));
        assert_eq!(resolve_zome("decisions"), Some("hearth_decisions"));
        assert_eq!(resolve_zome("resources"), Some("hearth_resources"));
        assert_eq!(resolve_zome("milestones"), Some("hearth_milestones"));
        assert_eq!(resolve_zome("rhythms"), Some("hearth_rhythms"));
    }

    #[test]
    fn resolve_unknown_domain_returns_none() {
        assert!(resolve_zome("justice").is_none());
        assert!(resolve_zome("property").is_none());
        assert!(resolve_zome("").is_none());
        assert!(resolve_zome("housing").is_none());
    }

    #[test]
    fn resolve_outputs_are_in_allowlist() {
        let domains = [
            "kinship",
            "gratitude",
            "stories",
            "care",
            "autonomy",
            "emergency",
            "decisions",
            "resources",
            "milestones",
            "rhythms",
        ];
        for domain in &domains {
            let resolved = resolve_zome(domain).unwrap();
            assert!(
                ALLOWED_ZOMES.contains(&resolved),
                "resolve('{}') = '{}' not in ALLOWED_ZOMES",
                domain,
                resolved
            );
        }
    }

    // ---- Signal type ----

    #[test]
    fn bridge_event_signal_type_is_hearth() {
        let signal = BridgeEventSignal {
            signal_type: "hearth_bridge_event".to_string(),
            domain: "kinship".to_string(),
            event_type: "member_joined".to_string(),
            payload: "{}".to_string(),
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        assert_eq!(signal.signal_type, "hearth_bridge_event");
    }

    #[test]
    fn bridge_event_signal_serde_roundtrip() {
        let signal = BridgeEventSignal {
            signal_type: "hearth_bridge_event".to_string(),
            domain: "milestones".to_string(),
            event_type: "milestone_recorded".to_string(),
            payload: r#"{"type":"graduation"}"#.to_string(),
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let back: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.domain, "milestones");
        assert_eq!(back.event_type, "milestone_recorded");
    }

    // ---- Rate limit ----

    #[test]
    fn rate_limit_under_threshold_passes() {
        assert!(check_rate_limit_count(0).is_ok());
        assert!(check_rate_limit_count(99).is_ok());
    }

    #[test]
    fn rate_limit_at_threshold_rejects() {
        let err = check_rate_limit_count(100).unwrap_err();
        assert!(err.contains("Rate limit exceeded"));
    }

    // ---- Health check domain list ----

    #[test]
    fn health_check_domains_match_resolution() {
        let domains = [
            "kinship",
            "gratitude",
            "stories",
            "care",
            "autonomy",
            "emergency",
            "decisions",
            "resources",
            "milestones",
            "rhythms",
        ];
        for domain in &domains {
            assert!(resolve_zome(domain).is_some());
        }
    }

    // ---- Cross-cluster allowlist no-duplicate checks ----

    #[test]
    fn personal_allowlist_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_PERSONAL_ZOMES {
            assert!(seen.insert(zome), "Duplicate: '{}'", zome);
        }
    }

    #[test]
    fn identity_allowlist_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_IDENTITY_ZOMES {
            assert!(seen.insert(zome), "Duplicate: '{}'", zome);
        }
    }

    #[test]
    fn commons_allowlist_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_COMMONS_ZOMES {
            assert!(seen.insert(zome), "Duplicate: '{}'", zome);
        }
    }

    #[test]
    fn civic_allowlist_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_CIVIC_ZOMES {
            assert!(seen.insert(zome), "Duplicate: '{}'", zome);
        }
    }

    // ---- HearthSyncInput serde ----

    #[test]
    fn hearth_sync_input_serde_roundtrip() {
        let input = HearthSyncInput {
            hearth_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            epoch_start: Timestamp::from_micros(0),
            epoch_end: Timestamp::from_micros(604_800_000_000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: HearthSyncInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hearth_hash, input.hearth_hash);
    }

    // ---- HearthSyncInput epoch fields ----

    #[test]
    fn hearth_sync_input_preserves_epoch_fields() {
        let input = HearthSyncInput {
            hearth_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            epoch_start: Timestamp::from_micros(1_000_000),
            epoch_end: Timestamp::from_micros(604_801_000_000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: HearthSyncInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.epoch_start, Timestamp::from_micros(1_000_000));
        assert_eq!(back.epoch_end, Timestamp::from_micros(604_801_000_000));
    }

    // ---- decode_dispatch_response ----

    #[test]
    fn decode_dispatch_response_success() {
        let payload: Vec<u32> = vec![1, 2, 3];
        let encoded = ExternIO::encode(payload.clone()).unwrap();
        let result = DispatchResult {
            success: true,
            response: Some(encoded.0),
            error: None,
        };
        let decoded: Vec<u32> = decode_dispatch_response(&result, "test").unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn decode_dispatch_response_failure_returns_error() {
        let result = DispatchResult {
            success: false,
            response: None,
            error: Some("something broke".to_string()),
        };
        let err = decode_dispatch_response::<Vec<u32>>(&result, "test_ctx").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("test_ctx"), "error should contain context");
        assert!(
            msg.contains("something broke"),
            "error should contain original message"
        );
    }

    #[test]
    fn decode_dispatch_response_failure_no_error_message() {
        let result = DispatchResult {
            success: false,
            response: None,
            error: None,
        };
        let err = decode_dispatch_response::<Vec<u32>>(&result, "ctx").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("unknown error"));
    }

    #[test]
    fn decode_dispatch_response_success_but_no_response() {
        let result = DispatchResult {
            success: true,
            response: None,
            error: None,
        };
        let err = decode_dispatch_response::<Vec<u32>>(&result, "ctx").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("no response"));
    }

    #[test]
    fn decode_dispatch_response_malformed_bytes() {
        let result = DispatchResult {
            success: true,
            response: Some(vec![0xFF, 0xFE, 0xFD]),
            error: None,
        };
        let err = decode_dispatch_response::<Vec<u32>>(&result, "ctx").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("decode"));
    }

    // ---- SeveranceInput serde ----

    #[test]
    fn severance_input_serde_roundtrip() {
        let input = SeveranceInput {
            hearth_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            member_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            export_milestones: true,
            export_care_history: true,
            export_bond_snapshot: false,
            new_role: MemberRole::Adult,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SeveranceInput = serde_json::from_str(&json).unwrap();
        assert!(back.export_milestones);
        assert_eq!(back.new_role, MemberRole::Adult);
    }
}
