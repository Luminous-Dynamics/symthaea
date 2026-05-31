// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Justice Bridge Coordinator Zome
//!
//! Cross-hApp communication for dispute filing, enforcement execution,
//! and dispute history across the Mycelix ecosystem.

use hdk::prelude::*;
use justice_bridge_integrity::*;

const JUSTICE_HAPP_ID: &str = "mycelix-justice";

/// Create an anchor path for linking
fn anchor_path(anchor: &str) -> ExternResult<TypedPath> {
    Path::from(anchor).typed(LinkTypes::ActiveDisputes)
}

/// Create a DID anchor path for linking disputes to DIDs
fn did_anchor(did: &str) -> ExternResult<TypedPath> {
    Path::from(format!("did/{}", did)).typed(LinkTypes::DidToDisputes)
}

/// Create a hApp anchor path for enforcement actions
fn happ_anchor(happ: &str) -> ExternResult<TypedPath> {
    Path::from(format!("happ/{}", happ)).typed(LinkTypes::HappToEnforcements)
}

/// File a dispute from another hApp
#[hdk_extern]
pub fn file_cross_happ_dispute(input: FileDisputeInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let dispute = CrossHappDispute {
        id: format!(
            "dispute:{}:{}:{}",
            input.source_happ,
            input.complainant_did,
            now.as_micros()
        ),
        source_happ: input.source_happ.clone(),
        complainant_did: input.complainant_did.clone(),
        respondent_did: input.respondent_did.clone(),
        dispute_type: input.dispute_type,
        description: input.description,
        evidence_refs: input.evidence_refs,
        status: DisputeStatus::Filed,
        created_at: now,
    };

    let hash = create_entry(&EntryTypes::CrossHappDispute(dispute.clone()))?;

    // Link from complainant
    let complainant_path = did_anchor(&input.complainant_did)?;
    complainant_path.ensure()?;
    create_link(
        complainant_path.path_entry_hash()?,
        hash.clone(),
        LinkTypes::DidToDisputes,
        (),
    )?;

    // Link from respondent
    let respondent_path = did_anchor(&input.respondent_did)?;
    respondent_path.ensure()?;
    create_link(
        respondent_path.path_entry_hash()?,
        hash.clone(),
        LinkTypes::DidToDisputes,
        (),
    )?;

    // Link to active disputes
    let active_path = anchor_path("active_disputes")?;
    active_path.ensure()?;
    create_link(
        active_path.path_entry_hash()?,
        hash.clone(),
        LinkTypes::ActiveDisputes,
        (),
    )?;

    // Broadcast event
    broadcast_justice_event(BroadcastJusticeEventInput {
        event_type: JusticeEventType::DisputeFiled,
        dispute_id: Some(dispute.id),
        subject_did: input.complainant_did,
        payload: serde_json::json!({ "source_happ": input.source_happ }).to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Dispute not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FileDisputeInput {
    pub source_happ: String,
    pub complainant_did: String,
    pub respondent_did: String,
    pub dispute_type: DisputeType,
    pub description: String,
    pub evidence_refs: Vec<String>,
}

/// Request enforcement on another hApp
#[hdk_extern]
pub fn request_enforcement(input: RequestEnforcementInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let action = EnforcementAction {
        id: format!(
            "enforce:{}:{}:{}",
            input.dispute_id,
            input.target_happ,
            now.as_micros()
        ),
        dispute_id: input.dispute_id.clone(),
        target_happ: input.target_happ.clone(),
        target_did: input.target_did.clone(),
        action_type: input.action_type.clone(),
        parameters: serde_json::to_string(&input.parameters).unwrap_or_default(),
        status: EnforcementStatus::Ordered,
        ordered_at: now,
        executed_at: None,
    };

    let hash = create_entry(&EntryTypes::EnforcementAction(action))?;

    let happ_path = happ_anchor(&input.target_happ)?;
    happ_path.ensure()?;
    create_link(
        happ_path.path_entry_hash()?,
        hash.clone(),
        LinkTypes::HappToEnforcements,
        (),
    )?;

    broadcast_justice_event(BroadcastJusticeEventInput {
        event_type: JusticeEventType::EnforcementOrdered,
        dispute_id: Some(input.dispute_id),
        subject_did: input.target_did,
        payload: serde_json::json!({
            "target_happ": input.target_happ,
            "action_type": format!("{:?}", input.action_type),
        })
        .to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Enforcement not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RequestEnforcementInput {
    pub dispute_id: String,
    pub target_happ: String,
    pub target_did: String,
    pub action_type: EnforcementType,
    pub parameters: serde_json::Value,
}

/// Get pending enforcements for a hApp
#[hdk_extern]
pub fn get_pending_enforcements(target_happ: String) -> ExternResult<Vec<Record>> {
    let happ_path = happ_anchor(&target_happ)?;
    let links = get_links(
        LinkQuery::try_new(happ_path.path_entry_hash()?, LinkTypes::HappToEnforcements)?,
        GetStrategy::default(),
    )?;

    let mut enforcements = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            enforcements.push(record);
        }
    }

    Ok(enforcements)
}

/// Acknowledge enforcement execution
#[hdk_extern]
pub fn acknowledge_enforcement(input: AcknowledgeEnforcementInput) -> ExternResult<bool> {
    broadcast_justice_event(BroadcastJusticeEventInput {
        event_type: JusticeEventType::EnforcementExecuted,
        dispute_id: None,
        subject_did: input.target_did,
        payload: serde_json::json!({
            "enforcement_id": input.enforcement_id,
            "status": format!("{:?}", input.status),
        })
        .to_string(),
    })?;
    Ok(true)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AcknowledgeEnforcementInput {
    pub enforcement_id: String,
    pub target_did: String,
    pub status: EnforcementStatus,
    pub result: Option<String>,
}

/// Get dispute history for a DID
#[hdk_extern]
pub fn get_dispute_history(input: GetDisputeHistoryInput) -> ExternResult<DisputeHistoryResult> {
    let now = sys_time()?;

    // Record the query
    let query = DisputeHistoryQuery {
        id: format!("history:{}:{}", input.did, now.as_micros()),
        did: input.did.clone(),
        source_happ: input.source_happ.clone(),
        queried_at: now,
    };
    create_entry(&EntryTypes::DisputeHistoryQuery(query))?;

    // Get disputes involving this DID
    let did_path = did_anchor(&input.did)?;
    let links = get_links(
        LinkQuery::try_new(did_path.path_entry_hash()?, LinkTypes::DidToDisputes)?,
        GetStrategy::default(),
    )?;

    let mut disputes = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            disputes.push(record);
        }
    }

    Ok(DisputeHistoryResult {
        did: input.did,
        total_disputes: disputes.len() as u32,
        as_complainant: 0, // Would count from records
        as_respondent: 0,
        active_disputes: 0,
        disputes,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetDisputeHistoryInput {
    pub did: String,
    pub source_happ: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DisputeHistoryResult {
    pub did: String,
    pub total_disputes: u32,
    pub as_complainant: u32,
    pub as_respondent: u32,
    pub active_disputes: u32,
    pub disputes: Vec<Record>,
}

/// Broadcast justice event
#[hdk_extern]
pub fn broadcast_justice_event(input: BroadcastJusticeEventInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let event = JusticeBridgeEvent {
        id: format!("event:{:?}:{}", input.event_type, now.as_micros()),
        event_type: input.event_type,
        dispute_id: input.dispute_id,
        subject_did: input.subject_did,
        payload: input.payload,
        source_happ: JUSTICE_HAPP_ID.to_string(),
        timestamp: now,
    };

    let hash = create_entry(&EntryTypes::JusticeBridgeEvent(event))?;

    let events_path: TypedPath = Path::from("recent_events").typed(LinkTypes::RecentEvents)?;
    events_path.ensure()?;
    create_link(
        events_path.path_entry_hash()?,
        hash.clone(),
        LinkTypes::RecentEvents,
        (),
    )?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Event not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BroadcastJusticeEventInput {
    pub event_type: JusticeEventType,
    pub dispute_id: Option<String>,
    pub subject_did: String,
    pub payload: String,
}
