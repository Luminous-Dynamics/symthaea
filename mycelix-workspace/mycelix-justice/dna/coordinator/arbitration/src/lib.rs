// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Arbitration Coordinator Zome
//!
//! Manages arbitration panels, deliberations, decisions, and appeals.
//! Implements Tier 2 (arbitration) and Tier 3 (appeal) of the justice system.

use hdk::prelude::*;
use mycelix_justice_integrity::*;

/// Create an arbitration panel for a case
#[hdk_extern]
pub fn create_arbitration(arbitration: Arbitration) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Arbitration(arbitration.clone()))?;
    let record = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not get created arbitration".into())
    ))?;

    // Link from case
    let case_path = Path::from(format!("cases/{}/arbitration", arbitration.case_id));
    create_link(
        case_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::CaseToArbitration,
        (),
    )?;

    // Link from each arbitrator
    for arb in &arbitration.arbitrators {
        let arb_path = Path::from(format!("arbitrators/{}/cases", arb.did));
        create_link(
            arb_path.path_entry_hash()?,
            action_hash.clone(),
            LinkTypes::ArbitratorToCases,
            (),
        )?;
    }

    Ok(record)
}

/// Get arbitration for a case
#[hdk_extern]
pub fn get_case_arbitration(case_id: String) -> ExternResult<Option<Record>> {
    let case_path = Path::from(format!("cases/{}/arbitration", case_id));
    let links = get_links(
        LinkQuery::try_new(case_path.path_entry_hash()?, LinkTypes::CaseToArbitration)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            return get(action_hash, GetOptions::default());
        }
    }

    Ok(None)
}

/// Update arbitration status
#[hdk_extern]
pub fn update_arbitration_status(input: UpdateArbStatusInput) -> ExternResult<Record> {
    let record = get(input.arbitration_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Arbitration not found".into())
    ))?;

    let mut arb: Arbitration = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid arbitration entry".into()
        )))?;

    arb.status = input.new_status;

    let action_hash = update_entry(input.arbitration_hash, &arb)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated arbitration".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateArbStatusInput {
    pub arbitration_hash: ActionHash,
    pub new_status: ArbitrationStatus,
}

/// Record an arbitrator's acceptance or recusal
#[hdk_extern]
pub fn record_arbitrator_response(input: ArbitratorResponseInput) -> ExternResult<Record> {
    let record = get(input.arbitration_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Arbitration not found".into())
    ))?;

    let mut arb: Arbitration = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid arbitration entry".into()
        )))?;

    // Find and update the arbitrator
    for a in &mut arb.arbitrators {
        if a.did == input.arbitrator_did {
            a.accepted = input.accepted;
            a.recused = input.recused;
            a.recusal_reason = input.recusal_reason.clone();
        }
    }

    let action_hash = update_entry(input.arbitration_hash, &arb)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated arbitration".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ArbitratorResponseInput {
    pub arbitration_hash: ActionHash,
    pub arbitrator_did: String,
    pub accepted: bool,
    pub recused: bool,
    pub recusal_reason: Option<String>,
}

/// Render a decision
#[hdk_extern]
pub fn render_decision(decision: Decision) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Decision(decision.clone()))?;
    let record = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not get created decision".into())
    ))?;

    // Link from case
    let case_path = Path::from(format!("cases/{}/decisions", decision.case_id));
    create_link(
        case_path.path_entry_hash()?,
        action_hash,
        LinkTypes::CaseToDecisions,
        (),
    )?;

    Ok(record)
}

/// Get decisions for a case
#[hdk_extern]
pub fn get_case_decisions(case_id: String) -> ExternResult<Vec<Record>> {
    let case_path = Path::from(format!("cases/{}/decisions", case_id));
    let links = get_links(
        LinkQuery::try_new(case_path.path_entry_hash()?, LinkTypes::CaseToDecisions)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

/// File an appeal
#[hdk_extern]
pub fn file_appeal(appeal: Appeal) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Appeal(appeal.clone()))?;
    let record = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not get created appeal".into())
    ))?;

    // Link from decision
    let decision_path = Path::from(format!("decisions/{}/appeals", appeal.decision_id));
    create_link(
        decision_path.path_entry_hash()?,
        action_hash,
        LinkTypes::DecisionToAppeals,
        (),
    )?;

    Ok(record)
}

/// Get appeals for a decision
#[hdk_extern]
pub fn get_decision_appeals(decision_id: String) -> ExternResult<Vec<Record>> {
    let decision_path = Path::from(format!("decisions/{}/appeals", decision_id));
    let links = get_links(
        LinkQuery::try_new(
            decision_path.path_entry_hash()?,
            LinkTypes::DecisionToAppeals,
        )?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

/// Update appeal status
#[hdk_extern]
pub fn update_appeal_status(input: UpdateAppealStatusInput) -> ExternResult<Record> {
    let record = get(input.appeal_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Appeal not found".into())
    ))?;

    let mut appeal: Appeal = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid appeal entry".into()
        )))?;

    appeal.status = input.new_status;

    let action_hash = update_entry(input.appeal_hash, &appeal)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated appeal".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateAppealStatusInput {
    pub appeal_hash: ActionHash,
    pub new_status: AppealStatus,
}

/// Finalize a decision (no more appeals allowed)
#[hdk_extern]
pub fn finalize_decision(input: FinalizeDecisionInput) -> ExternResult<Record> {
    let record = get(input.decision_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Decision not found".into())
    ))?;

    let mut decision: Decision = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid decision entry".into()
        )))?;

    decision.finalized = true;

    let action_hash = update_entry(input.decision_hash, &decision)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated decision".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FinalizeDecisionInput {
    pub decision_hash: ActionHash,
}

/// Get cases for an arbitrator
#[hdk_extern]
pub fn get_arbitrator_cases(arbitrator_did: String) -> ExternResult<Vec<Record>> {
    let arb_path = Path::from(format!("arbitrators/{}/cases", arbitrator_did));
    let links = get_links(
        LinkQuery::try_new(arb_path.path_entry_hash()?, LinkTypes::ArbitratorToCases)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }

    Ok(records)
}
