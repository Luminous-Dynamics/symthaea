// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Complaints Coordinator Zome
use complaints_integrity::*;
use hdk::prelude::*;

/// Create a deterministic anchor hash from a string
fn anchor_hash(s: &str) -> ExternResult<EntryHash> {
    let hash = holo_hash::blake2b_256(s.as_bytes());
    Ok(EntryHash::from_raw_32(hash.to_vec()))
}

/// Helper to get records from links
fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

#[hdk_extern]
pub fn file_complaint(complaint: Complaint) -> ExternResult<Record> {
    if complaint.complainant.is_empty() || complaint.complainant.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Complainant must be 1-256 characters".into()
        )));
    }
    if complaint.respondent.is_empty() || complaint.respondent.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Respondent must be 1-256 characters".into()
        )));
    }
    if complaint.title.is_empty() || complaint.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Title must be 1-256 characters".into()
        )));
    }
    if complaint.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be under 4096 characters".into()
        )));
    }
    if complaint.evidence_ids.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Maximum 100 evidence items".into()
        )));
    }
    let action_hash = create_entry(&EntryTypes::Complaint(complaint.clone()))?;
    create_link(
        anchor_hash(&format!("complainant:{}", complaint.complainant))?,
        action_hash.clone(),
        LinkTypes::ComplainantToComplaint,
        (),
    )?;
    create_link(
        anchor_hash(&format!("respondent:{}", complaint.respondent))?,
        action_hash.clone(),
        LinkTypes::RespondentToComplaint,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[hdk_extern]
pub fn get_complaint(complaint_id: String) -> ExternResult<Option<Record>> {
    if complaint_id.is_empty() || complaint_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Complaint ID must be 1-256 characters".into()
        )));
    }
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Complaint,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(c) = record.entry().to_app_option::<Complaint>().ok().flatten() {
            if c.id == complaint_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

#[hdk_extern]
pub fn get_complaints_by_party(party_did: String) -> ExternResult<Vec<Record>> {
    if party_did.is_empty() || party_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Party DID must be 1-256 characters".into()
        )));
    }
    let mut complaints = Vec::new();

    let complainant_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("complainant:{}", party_did))?,
            LinkTypes::ComplainantToComplaint,
        )?,
        GetStrategy::default(),
    )?;
    complaints.extend(records_from_links(complainant_links)?);

    let respondent_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("respondent:{}", party_did))?,
            LinkTypes::RespondentToComplaint,
        )?,
        GetStrategy::default(),
    )?;
    complaints.extend(records_from_links(respondent_links)?);

    Ok(complaints)
}

/// Update complaint status (e.g., from Open to UnderReview, Resolved, Dismissed)
#[hdk_extern]
pub fn update_complaint_status(input: UpdateComplaintStatusInput) -> ExternResult<Record> {
    if input.complaint_id.is_empty() || input.complaint_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Complaint ID must be 1-256 characters".into()
        )));
    }
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Complaint,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(complaint) = record.entry().to_app_option::<Complaint>().ok().flatten() {
            if complaint.id == input.complaint_id {
                let updated = Complaint {
                    status: input.new_status,
                    ..complaint
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Complaint(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Complaint not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateComplaintStatusInput {
    pub complaint_id: String,
    pub new_status: ComplaintStatus,
}

/// Submit a response to a complaint (by the respondent)
#[hdk_extern]
pub fn submit_response(input: SubmitResponseInput) -> ExternResult<Record> {
    if input.complaint_id.is_empty() || input.complaint_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Complaint ID must be 1-256 characters".into()
        )));
    }
    if input.respondent.is_empty() || input.respondent.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Respondent must be 1-256 characters".into()
        )));
    }
    if input.content.is_empty() || input.content.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Content must be 1-4096 characters".into()
        )));
    }
    let response = Response {
        id: format!(
            "response:{}:{}",
            input.complaint_id,
            sys_time()?.as_micros()
        ),
        complaint_id: input.complaint_id.clone(),
        respondent: input.respondent.clone(),
        content: input.content,
        submitted: sys_time()?,
    };
    let action_hash = create_entry(&EntryTypes::Response(response))?;
    create_link(
        anchor_hash(&format!("complaint:{}", input.complaint_id))?,
        action_hash.clone(),
        LinkTypes::ComplaintToResponse,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitResponseInput {
    pub complaint_id: String,
    pub respondent: String,
    pub content: String,
}

/// Get all open complaints in the system (for arbitration panel)
#[hdk_extern]
pub fn get_open_complaints(_: ()) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Complaint,
        )?))
        .include_entries(true);

    let mut open_complaints = Vec::new();
    for record in query(filter)? {
        if let Some(complaint) = record.entry().to_app_option::<Complaint>().ok().flatten() {
            if matches!(
                complaint.status,
                ComplaintStatus::Open | ComplaintStatus::UnderReview
            ) {
                open_complaints.push(record);
            }
        }
    }
    Ok(open_complaints)
}

/// Get responses to a complaint
#[hdk_extern]
pub fn get_complaint_responses(complaint_id: String) -> ExternResult<Vec<Record>> {
    if complaint_id.is_empty() || complaint_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Complaint ID must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("complaint:{}", complaint_id))?,
            LinkTypes::ComplaintToResponse,
        )?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}
