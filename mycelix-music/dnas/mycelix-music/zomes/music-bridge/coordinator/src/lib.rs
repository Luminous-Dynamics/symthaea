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
    mycelix_bridge_common::gate_civic(
        "music_bridge",
        &mycelix_bridge_common::civic_requirement_basic(),
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
    mycelix_bridge_common::gate_civic(
        "music_bridge",
        &mycelix_bridge_common::civic_requirement_voting(),
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
    mycelix_bridge_common::gate_civic(
        "music_bridge",
        &mycelix_bridge_common::civic_requirement_basic(),
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
        healthy: true,
        agent: agent.to_string(),
        total_events: 0,
        total_queries: 0,
        domains: ALLOWED_ZOMES.iter().map(|z| z.to_string()).collect(),
    })
}

// ============================================================================
// Consciousness Composition
// ============================================================================

/// Input from Symthaea's cognitive loop — the cognitive state that triggered
/// this composition request.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConsciousnessTrigger {
    /// Eight Harmonies activation vector [0,1]^8 from MusicalState
    pub harmony_activations: Vec<f32>,
    /// Valence from Russell's circumplex model [-1, 1]
    pub valence: f32,
    /// Arousal from Russell's circumplex model [0, 1]
    pub arousal: f32,
    /// Integrated Information (Phi) consciousness score [0, 1]
    pub phi_score: f32,
    /// Narrative episode tags active at composition time
    pub narrative_tags: Vec<String>,
    /// Requested duration in seconds (clamped to [4, 120])
    pub duration_secs: Option<f32>,
}

/// Result returned to the caller
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConsciousCompositionResult {
    /// DHT record of the stored composition metadata
    pub action_hash: ActionHash,
    /// Derived tempo BPM
    pub tempo_bpm: f32,
    /// Scale name used
    pub scale_name: String,
    /// Number of notes generated (estimated from duration/density)
    pub note_count: u32,
    /// Composite quality score
    pub quality_score: f32,
    /// Signal was emitted to UI clients
    pub signal_emitted: bool,
}

/// Minimum Phi score required to trigger a consciousness composition.
/// Below this threshold, consciousness is not sufficiently integrated
/// to produce a meaningful piece — a neutral ambient tone would mask the lack.
const MIN_PHI_FOR_COMPOSITION: f32 = 0.3;

/// Map VA + harmonies to musical parameter summary (WASM-compatible, no heavy synthesis).
///
/// Full audio synthesis lives in symthaea-muse (native Rust); in the Holochain
/// WASM context we store only composition *metadata* on the DHT and emit a
/// signal so frontends can request the actual audio via a native service.
fn derive_musical_params(trigger: &ConsciousnessTrigger) -> (f32, String, u32, f32) {
    // Tempo: high arousal → faster. Range [60, 160] BPM.
    let arousal = trigger.arousal.clamp(0.0, 1.0);
    let valence = trigger.valence.clamp(-1.0, 1.0);
    let tempo_bpm = 60.0 + arousal * 100.0;

    // Scale: negative valence → minor; extreme arousal + harmonies guide mode
    let ha = &trigger.harmony_activations;
    let dominant_harmony = ha
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let scale_name = if valence < -0.3 {
        match dominant_harmony {
            2 => "phrygian".to_string(),       // IntegralWisdom → ancient depth
            4 => "maqam_hijaz".to_string(),    // EthicalPresence → Middle Eastern tension
            _ => "minor".to_string(),
        }
    } else if valence > 0.3 {
        match dominant_harmony {
            0 => "lydian".to_string(),         // InfinitePlay → bright, floating
            1 => "major".to_string(),          // UniversalInterconnectedness → grounded joy
            6 => "raga_yaman".to_string(),     // RegenerativeHarmony → flourishing
            _ => "major".to_string(),
        }
    } else {
        match dominant_harmony {
            3 => "dorian".to_string(),         // SovereignPresence → modal, balanced
            5 => "mixolydian".to_string(),     // CollectiveWisdom → folk-like
            7 => "gamelan_slendro".to_string(),// SacredStillness → meditative
            _ => "dorian".to_string(),
        }
    };

    // Note count estimate: density scales with arousal and phi
    let duration = trigger.duration_secs.unwrap_or(8.0).clamp(4.0, 120.0);
    let density = 0.3 + arousal * 0.5 + trigger.phi_score * 0.2;
    let note_count = (duration * density * 4.0) as u32;

    // Quality: composite heuristic from phi, harmony breadth, va balance
    let harmony_sum: f32 = ha.iter().sum::<f32>() / 8.0;
    let va_balance = 1.0 - (valence.abs() * 0.3 + (arousal - 0.5).abs() * 0.3);
    let quality_score = (trigger.phi_score * 0.4 + harmony_sum * 0.3 + va_balance * 0.3)
        .clamp(0.0, 1.0);

    (tempo_bpm, scale_name, note_count.max(4), quality_score)
}

/// Compose a piece of music from Symthaea's current cognitive state.
///
/// This is the consciousness→music bridge entry point. When Symthaea's creative
/// manager detects sufficient integration (Phi > 0.3), it calls this zome
/// function to record the composition decision on the DHT and notify connected
/// UI frontends via a signal.
///
/// The heavy audio synthesis is done by `symthaea-muse` in native Rust;
/// this function stores the composition *parameters* and emits a signal
/// so the Music UI can request audio from the local Symthaea service.
///
/// Requires Observer+ consciousness tier (lowest tier — consciousness speaks freely).
#[hdk_extern]
pub fn compose_from_consciousness(
    trigger: ConsciousnessTrigger,
) -> ExternResult<ConsciousCompositionResult> {
    // Validate harmony dimensions
    if trigger.harmony_activations.len() != 8 {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "harmony_activations must have 8 elements, got {}",
            trigger.harmony_activations.len()
        ))));
    }

    // Minimum consciousness threshold
    if trigger.phi_score < MIN_PHI_FOR_COMPOSITION {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Phi score {:.3} below minimum {:.3} for consciousness composition",
            trigger.phi_score, MIN_PHI_FOR_COMPOSITION
        ))));
    }

    // Gate: Observer+ (no tier restriction — consciousness speaks)
    mycelix_bridge_common::gate_civic(
        "music_bridge",
        &mycelix_bridge_common::civic_requirement_basic(),
        "compose_from_consciousness",
    )?;

    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;
    let duration = trigger.duration_secs.unwrap_or(8.0).clamp(4.0, 120.0);

    let (tempo_bpm, scale_name, note_count, quality_score) = derive_musical_params(&trigger);

    let entry = ConsciousCompositionEntry {
        schema_version: 1,
        composer_agent: agent.clone(),
        harmony_activations: trigger.harmony_activations.clone(),
        valence: trigger.valence.clamp(-1.0, 1.0),
        arousal: trigger.arousal.clamp(0.0, 1.0),
        phi_score: trigger.phi_score.clamp(0.0, 1.0),
        narrative_tags: trigger.narrative_tags.clone(),
        tempo_bpm,
        scale_name: scale_name.clone(),
        duration_secs: duration,
        note_count,
        quality_score,
        composed_at: now,
    };

    let action_hash = create_entry(&EntryTypes::ConsciousComposition(entry))?;

    // Index: global
    let all_anchor = ensure_anchor("all_conscious_compositions")?;
    create_link(
        all_anchor,
        action_hash.clone(),
        LinkTypes::AllConsciousCompositions,
        (),
    )?;

    // Index: per-agent
    let agent_anchor = ensure_anchor(&format!("agent_compositions:{}", agent))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToConsciousComposition,
        (),
    )?;

    // Signal to connected UIs: "a new consciousness composition is available,
    // request audio from the local Symthaea service"
    #[derive(Serialize, Deserialize, Debug)]
    struct ConsciousCompositionSignal {
        signal_type: String,
        action_hash: ActionHash,
        tempo_bpm: f32,
        scale_name: String,
        note_count: u32,
        quality_score: f32,
        phi_score: f32,
        valence: f32,
        arousal: f32,
        narrative_tags: Vec<String>,
    }

    let signal = ConsciousCompositionSignal {
        signal_type: "consciousness_composition".to_string(),
        action_hash: action_hash.clone(),
        tempo_bpm,
        scale_name: scale_name.clone(),
        note_count,
        quality_score,
        phi_score: trigger.phi_score,
        valence: trigger.valence,
        arousal: trigger.arousal,
        narrative_tags: trigger.narrative_tags,
    };

    let signal_emitted = emit_signal(&signal).is_ok();

    Ok(ConsciousCompositionResult {
        action_hash,
        tempo_bpm,
        scale_name,
        note_count,
        quality_score,
        signal_emitted,
    })
}

/// Get all consciousness compositions (paginated via DHT link queries)
#[hdk_extern]
pub fn get_conscious_compositions(_: ()) -> ExternResult<Vec<Record>> {
    let all_anchor = anchor_hash("all_conscious_compositions")?;
    let links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllConsciousCompositions)?,
        GetStrategy::Local,
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Ok(Some(record)) = get_latest_record(action_hash) {
                records.push(record);
            }
        }
    }
    Ok(records)
}

/// Get consciousness compositions for a specific agent
#[hdk_extern]
pub fn get_agent_compositions(agent: AgentPubKey) -> ExternResult<Vec<Record>> {
    let agent_anchor = anchor_hash(&format!("agent_compositions:{}", agent))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToConsciousComposition)?,
        GetStrategy::Local,
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Ok(Some(record)) = get_latest_record(action_hash) {
                records.push(record);
            }
        }
    }
    Ok(records)
}

// ============================================================================
// Cross-Cluster Dispatch
// ============================================================================

/// Input for cross-cluster dispatch to other hApp roles.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrossClusterInput {
    /// Target role name (e.g., "identity", "finance", "governance")
    pub target_role: String,
    /// Target zome within that role
    pub target_zome: String,
    /// Function name to call
    pub fn_name: String,
    /// Serialized payload
    pub payload: Vec<u8>,
}

/// Dispatch a call to another cluster via CallTargetCell::OtherRole.
///
/// Enables Music -> Identity (DID verification), Music -> Finance (royalty
/// settlement via SAP/TEND), Music -> Governance (artist proposals).
///
/// Requires Citizen+ consciousness tier.
#[hdk_extern]
pub fn cross_cluster_dispatch(input: CrossClusterInput) -> ExternResult<DispatchResult> {
    mycelix_bridge_common::gate_civic(
        "music_bridge",
        &mycelix_bridge_common::civic_requirement_voting(),
        "cross_cluster_dispatch",
    )?;

    // Validate target role
    let allowed_targets = ["identity", "finance", "governance", "civic", "commons_land", "commons_care"];
    if !allowed_targets.contains(&input.target_role.as_str()) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cross-cluster dispatch to '{}' not allowed from music. Allowed: {:?}",
            input.target_role, allowed_targets
        ))));
    }

    if input.target_zome.trim().is_empty() || input.fn_name.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Target zome and function name cannot be empty".into()
        )));
    }

    enforce_rate_limit(&format!("xcluster:{}", input.target_role))?;

    let response = call(
        CallTargetCell::OtherRole(input.target_role.into()),
        ZomeName::from(input.target_zome),
        FunctionName::from(input.fn_name),
        None,
        ExternIO(input.payload),
    )?;

    match response {
        ZomeCallResponse::Ok(output) => Ok(DispatchResult {
            success: true,
            response: Some(output.0),
            error: None,
            error_code: None,
        }),
        ZomeCallResponse::NetworkError(err) => Ok(DispatchResult {
            success: false,
            response: None,
            error: Some(format!("Network error: {}", err)),
            error_code: None,
        }),
        other => Ok(DispatchResult {
            success: false,
            response: None,
            error: Some(format!("Call failed: {:?}", other)),
            error_code: None,
        }),
    }
}
