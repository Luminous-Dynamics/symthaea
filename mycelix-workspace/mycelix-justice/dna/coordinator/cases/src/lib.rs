// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Cases Coordinator Zome
//!
//! Manages the full lifecycle of dispute cases including filing,
//! mediation, escalation, and closure.

use hdk::prelude::*;
use mycelix_justice_integrity::*;

/// File a new case
#[hdk_extern]
pub fn file_case(case: Case) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Case(case.clone()))?;
    let record = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not get created case".into())
    ))?;

    // Link from complainant
    let complainant_path = Path::from(format!("users/{}/cases", case.complainant));
    create_link(
        complainant_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::ComplainantToCases,
        (),
    )?;

    // Link from respondent
    let respondent_path = Path::from(format!("users/{}/cases", case.respondent));
    create_link(
        respondent_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::RespondentToCases,
        (),
    )?;

    // Link to all cases
    let all_cases_path = Path::from("cases/all");
    create_link(
        all_cases_path.path_entry_hash()?,
        action_hash,
        LinkTypes::AllCases,
        (),
    )?;

    Ok(record)
}

/// Get a case by its action hash
#[hdk_extern]
pub fn get_case(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Update case phase
#[hdk_extern]
pub fn update_case_phase(input: UpdatePhaseInput) -> ExternResult<Record> {
    let record = get(input.case_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Case not found".into())))?;

    let mut case: Case = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid case entry".into()
        )))?;

    case.phase = input.new_phase;
    case.updated_at = sys_time()?;
    case.phase_deadline = input.deadline;

    let action_hash = update_entry(input.case_hash, &case)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated case".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdatePhaseInput {
    pub case_hash: ActionHash,
    pub new_phase: CasePhase,
    pub deadline: Option<Timestamp>,
}

/// Update case status
#[hdk_extern]
pub fn update_case_status(input: UpdateStatusInput) -> ExternResult<Record> {
    let record = get(input.case_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Case not found".into())))?;

    let mut case: Case = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid case entry".into()
        )))?;

    case.status = input.new_status;
    case.updated_at = sys_time()?;

    let action_hash = update_entry(input.case_hash, &case)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated case".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateStatusInput {
    pub case_hash: ActionHash,
    pub new_status: CaseStatus,
}

/// Add a party to a case
#[hdk_extern]
pub fn add_party(input: AddPartyInput) -> ExternResult<Record> {
    let record = get(input.case_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Case not found".into())))?;

    let mut case: Case = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid case entry".into()
        )))?;

    let party = CaseParty {
        did: input.party_did,
        role: input.role,
        joined_at: sys_time()?,
    };

    case.parties.push(party);
    case.updated_at = sys_time()?;

    let action_hash = update_entry(input.case_hash, &case)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated case".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddPartyInput {
    pub case_hash: ActionHash,
    pub party_did: String,
    pub role: PartyRole,
}

/// Submit evidence for a case
#[hdk_extern]
pub fn submit_evidence(evidence: Evidence) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Evidence(evidence.clone()))?;
    let record = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not get created evidence".into())
    ))?;

    // Link from case
    let case_path = Path::from(format!("cases/{}/evidence", evidence.case_id));
    create_link(
        case_path.path_entry_hash()?,
        action_hash,
        LinkTypes::CaseToEvidence,
        (),
    )?;

    Ok(record)
}

/// Get all evidence for a case
#[hdk_extern]
pub fn get_case_evidence(case_id: String) -> ExternResult<Vec<Record>> {
    let case_path = Path::from(format!("cases/{}/evidence", case_id));
    let links = get_links(
        LinkQuery::try_new(case_path.path_entry_hash()?, LinkTypes::CaseToEvidence)?,
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

/// Initiate mediation for a case
#[hdk_extern]
pub fn initiate_mediation(mediation: Mediation) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Mediation(mediation.clone()))?;
    let record = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not get created mediation".into())
    ))?;

    // Link from case
    let case_path = Path::from(format!("cases/{}/mediation", mediation.case_id));
    create_link(
        case_path.path_entry_hash()?,
        action_hash,
        LinkTypes::CaseToMediation,
        (),
    )?;

    Ok(record)
}

/// Get cases for a user (as complainant or respondent)
#[hdk_extern]
pub fn get_my_cases(did: String) -> ExternResult<Vec<Record>> {
    let user_path = Path::from(format!("users/{}/cases", did));
    let links = get_links(
        LinkQuery::try_new(user_path.path_entry_hash()?, LinkTypes::ComplainantToCases)?,
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

    // Also get cases where user is respondent
    let respondent_links = get_links(
        LinkQuery::try_new(user_path.path_entry_hash()?, LinkTypes::RespondentToCases)?,
        GetStrategy::default(),
    )?;

    for link in respondent_links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                // Avoid duplicates
                if !records
                    .iter()
                    .any(|r| r.action_address() == record.action_address())
                {
                    records.push(record);
                }
            }
        }
    }

    Ok(records)
}

/// Get all open cases
#[hdk_extern]
pub fn get_all_cases(_: ()) -> ExternResult<Vec<Record>> {
    let all_path = Path::from("cases/all");
    let links = get_links(
        LinkQuery::try_new(all_path.path_entry_hash()?, LinkTypes::AllCases)?,
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
