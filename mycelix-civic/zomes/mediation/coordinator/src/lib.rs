// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mediation Coordinator Zome
//!
//! Community mediation for dispute resolution before formal justice.
//! Supports request → acceptance → mediator assignment → session recording →
//! agreement → resolution, with escalation to the justice-cases zome.

use hdk::prelude::*;
use mediation_integrity::*;
use mycelix_bridge_common::{
    civic_requirement_basic, civic_requirement_proposal, civic_requirement_voting,
    GovernanceEligibility,
};

/// Helper to get an anchor entry hash.
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    hash_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))
}

/// Helper to ensure an anchor entry exists and return its hash.
fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    create_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))?;
    anchor_hash(anchor_str)
}

// ============================================================================
// REQUEST MANAGEMENT
// ============================================================================

/// Request mediation for a dispute.
#[hdk_extern]
pub fn request_mediation(request: MediationRequest) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "request_mediation")?;

    let action_hash = create_entry(&EntryTypes::MediationRequest(request.clone()))?;

    // Link from all requests anchor
    let all_anchor = ensure_anchor("all_mediation_requests")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllRequests, ())?;

    // Link from requester
    let requester_anchor = ensure_anchor(&format!("agent/{}/mediation", request.requester_did))?;
    create_link(requester_anchor, action_hash.clone(), LinkTypes::AgentToRequest, ())?;

    // Link from respondent
    let respondent_anchor = ensure_anchor(&format!("agent/{}/mediation", request.respondent_did))?;
    create_link(respondent_anchor, action_hash.clone(), LinkTypes::AgentToRequest, ())?;

    // Link by status
    let status_str = format!("{:?}", request.status);
    let status_anchor = ensure_anchor(&format!("mediation/status/{}", status_str))?;
    create_link(status_anchor, action_hash.clone(), LinkTypes::RequestsByStatus, ())?;

    Ok(action_hash)
}

/// Get a mediation request by its action hash.
#[hdk_extern]
pub fn get_request(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Get all mediation requests involving the calling agent.
#[hdk_extern]
pub fn get_my_requests(_: ()) -> ExternResult<Vec<Record>> {
    let agent_info = agent_info()?;
    let agent_did = format!("{}", agent_info.agent_initial_pubkey);
    let base = anchor_hash(&format!("agent/{}/mediation", agent_did))?;
    let links = get_links(
        LinkQuery::try_new(base, LinkTypes::AgentToRequest)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links.into_iter().rev().take(100) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

/// Respondent accepts a mediation request.
#[hdk_extern]
pub fn accept_mediation(request_hash: ActionHash) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "accept_mediation")?;

    let record = get(request_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Request not found".into())))?;
    let mut request: MediationRequest = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid request entry".into())))?;

    if request.status != MediationStatus::Requested {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request must be in Requested status to accept".into()
        )));
    }

    request.status = MediationStatus::Accepted;
    let updated = update_entry(request_hash, &request)?;
    Ok(updated)
}

// ============================================================================
// MEDIATOR MANAGEMENT
// ============================================================================

/// Input for assigning a mediator to a request.
#[derive(Serialize, Deserialize, Debug)]
pub struct AssignMediatorInput {
    pub request_hash: ActionHash,
    pub mediator_did: String,
}

/// Assign a mediator to a mediation request.
#[hdk_extern]
pub fn assign_mediator(input: AssignMediatorInput) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_voting(), "assign_mediator")?;

    let record = get(input.request_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Request not found".into())))?;
    let mut request: MediationRequest = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid request entry".into())))?;

    if request.status != MediationStatus::Accepted {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request must be Accepted before assigning a mediator".into()
        )));
    }

    request.status = MediationStatus::InProgress;
    let updated = update_entry(input.request_hash, &request)?;
    Ok(updated)
}

// ============================================================================
// SESSION RECORDING
// ============================================================================

/// Record a mediation session.
#[hdk_extern]
pub fn record_session(session: MediationSession) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "record_session")?;

    let action_hash = create_entry(&EntryTypes::MediationSession(session.clone()))?;

    // Link from request
    let request_anchor = ensure_anchor(&format!("mediation/request/{}/sessions", session.request_id))?;
    create_link(request_anchor, action_hash.clone(), LinkTypes::RequestToSession, ())?;

    // Link from mediator
    let mediator_anchor = ensure_anchor(&format!("mediator/{}/sessions", session.mediator_did))?;
    create_link(mediator_anchor, action_hash.clone(), LinkTypes::MediatorToSession, ())?;

    Ok(action_hash)
}

/// Get all sessions for a mediation request.
#[hdk_extern]
pub fn get_request_sessions(request_id: String) -> ExternResult<Vec<Record>> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "get_request_sessions")?;

    let base = anchor_hash(&format!("mediation/request/{}/sessions", request_id))?;
    let links = get_links(
        LinkQuery::try_new(base, LinkTypes::RequestToSession)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links.into_iter().rev().take(100) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

// ============================================================================
// AGREEMENTS
// ============================================================================

/// Propose a mediation agreement.
#[hdk_extern]
pub fn propose_agreement(agreement: MediationAgreement) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "propose_agreement")?;

    let action_hash = create_entry(&EntryTypes::MediationAgreement(agreement.clone()))?;

    // Link from request
    let request_anchor = ensure_anchor(&format!("mediation/request/{}/agreements", agreement.request_id))?;
    create_link(request_anchor, action_hash.clone(), LinkTypes::RequestToAgreement, ())?;

    Ok(action_hash)
}

/// Accept an existing agreement (add the calling agent to agreed_by).
#[hdk_extern]
pub fn accept_agreement(agreement_hash: ActionHash) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "accept_agreement")?;

    let record = get(agreement_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Agreement not found".into())))?;
    let mut agreement: MediationAgreement = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid agreement entry".into())))?;

    let agent_info = agent_info()?;
    let agent_did = format!("{}", agent_info.agent_initial_pubkey);

    if !agreement.agreed_by.contains(&agent_did) {
        agreement.agreed_by.push(agent_did);
    }

    let updated = update_entry(agreement_hash, &agreement)?;
    Ok(updated)
}

// ============================================================================
// RESOLUTION & ESCALATION
// ============================================================================

/// Mark a mediation as resolved.
#[hdk_extern]
pub fn resolve_mediation(request_hash: ActionHash) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "resolve_mediation")?;

    let record = get(request_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Request not found".into())))?;
    let mut request: MediationRequest = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid request entry".into())))?;

    if request.status != MediationStatus::InProgress {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only resolve mediation that is InProgress".into()
        )));
    }

    request.status = MediationStatus::Resolved;
    let updated = update_entry(request_hash, &request)?;
    Ok(updated)
}

/// Escalate a failed mediation to the formal justice system.
///
/// Dispatches to the justice_cases zome via `call(CallTargetCell::Local, ...)`.
#[hdk_extern]
pub fn escalate_to_justice(request_hash: ActionHash) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "escalate_to_justice")?;

    let record = get(request_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Request not found".into())))?;
    let mut request: MediationRequest = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid request entry".into())))?;

    if request.status != MediationStatus::InProgress && request.status != MediationStatus::Accepted {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only escalate mediation that is Accepted or InProgress".into()
        )));
    }

    // Build a justice case payload for cross-zome dispatch
    let case_payload = serde_json::json!({
        "id": format!("mediation-escalation-{}", request.request_id),
        "title": format!("Escalated mediation: {}", request.subject),
        "description": format!(
            "Mediation failed for dispute between {} and {}. Category: {:?}. Original description: {}",
            request.requester_did, request.respondent_did, request.category, request.description
        ),
        "case_type": "ConductViolation",
        "complainant": request.requester_did,
        "respondent": request.respondent_did,
        "parties": [],
        "phase": "Filed",
        "status": "Active",
        "severity": match request.urgency {
            MediationUrgency::High => "Serious",
            MediationUrgency::Medium => "Moderate",
            MediationUrgency::Low => "Minor",
        },
        "context": {
            "happ": "mediation",
            "reference_id": request.request_id,
            "community": null,
            "jurisdiction": null
        },
        "created_at": sys_time()?.as_micros(),
        "updated_at": sys_time()?.as_micros(),
        "phase_deadline": null
    });

    // Dispatch to justice_cases zome locally
    let _response = call(
        CallTargetCell::Local,
        "justice_cases",
        "file_case".into(),
        None,
        &case_payload,
    )?;

    // Mark the mediation request as escalated
    request.status = MediationStatus::Escalated;
    let updated = update_entry(request_hash, &request)?;
    Ok(updated)
}

// ============================================================================
// QUERIES
// ============================================================================

/// Get all open (non-resolved, non-withdrawn) mediation requests.
#[hdk_extern]
pub fn get_open_requests(_: ()) -> ExternResult<Vec<Record>> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "get_open_requests")?;

    let base = anchor_hash("all_mediation_requests")?;
    let links = get_links(
        LinkQuery::try_new(base, LinkTypes::AllRequests)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links.into_iter().rev().take(200) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                // Filter to open statuses only
                if let Some(req) = record
                    .entry()
                    .to_app_option::<MediationRequest>()
                    .ok()
                    .flatten()
                {
                    match req.status {
                        MediationStatus::Requested
                        | MediationStatus::Accepted
                        | MediationStatus::InProgress => {
                            records.push(record);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    Ok(records)
}

/// Get all sessions for a specific mediator.
#[hdk_extern]
pub fn get_mediator_sessions(mediator_did: String) -> ExternResult<Vec<Record>> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "get_mediator_sessions")?;

    let base = anchor_hash(&format!("mediator/{}/sessions", mediator_did))?;
    let links = get_links(
        LinkQuery::try_new(base, LinkTypes::MediatorToSession)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links.into_iter().rev().take(100) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_mediation_status_flow() {
        use mediation_integrity::MediationStatus;
        // Verify all status variants serialize/deserialize
        let statuses = vec![
            MediationStatus::Requested,
            MediationStatus::Accepted,
            MediationStatus::InProgress,
            MediationStatus::Resolved,
            MediationStatus::Escalated,
            MediationStatus::Withdrawn,
        ];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let back: MediationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*s, back);
        }
        // Verify valid state transitions are representable
        assert_ne!(MediationStatus::Requested, MediationStatus::Accepted);
        assert_ne!(MediationStatus::InProgress, MediationStatus::Resolved);
        assert_ne!(MediationStatus::InProgress, MediationStatus::Escalated);
    }

    #[test]
    fn test_urgency_levels() {
        use mediation_integrity::MediationUrgency;
        let levels = vec![
            MediationUrgency::Low,
            MediationUrgency::Medium,
            MediationUrgency::High,
        ];
        for level in &levels {
            let json = serde_json::to_string(level).unwrap();
            let back: MediationUrgency = serde_json::from_str(&json).unwrap();
            assert_eq!(*level, back);
        }
        // Verify all three levels are distinct
        assert_ne!(MediationUrgency::Low, MediationUrgency::Medium);
        assert_ne!(MediationUrgency::Medium, MediationUrgency::High);
        assert_ne!(MediationUrgency::Low, MediationUrgency::High);
    }

    #[test]
    fn test_category_variants() {
        use mediation_integrity::MediationCategory;
        let categories = vec![
            MediationCategory::Neighbor,
            MediationCategory::Property,
            MediationCategory::Noise,
            MediationCategory::SharedSpace,
            MediationCategory::Financial,
            MediationCategory::Other,
        ];
        assert_eq!(categories.len(), 6);
        for c in &categories {
            let json = serde_json::to_string(c).unwrap();
            let back: MediationCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(*c, back);
        }
    }
}
