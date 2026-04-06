#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Applications Coordinator Zome
//!
//! Business logic for job application lifecycle management.
//! State transitions are validated at the integrity layer.

use hdk::prelude::*;
use applications_integrity::{
    ApplicationStatus, EntryTypes, JobApplication, LinkTypes,
};

// ============== Helpers ==============

fn deserialize_application(record: &Record) -> ExternResult<JobApplication> {
    match record.entry().as_option() {
        Some(Entry::App(bytes)) => JobApplication::try_from(
            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
        )
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {:?}", e)))),
        _ => Err(wasm_error!(WasmErrorInner::Guest("Not an app entry".into()))),
    }
}

fn load_applications(links: Vec<Link>) -> ExternResult<Vec<(ActionHash, JobApplication)>> {
    let mut items = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Ok(app) = deserialize_application(&record) {
                    items.push((hash, app));
                }
            }
        }
    }
    Ok(items)
}

// ============== Input Types ==============

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateApplicationInput {
    pub job_posting_hash: ActionHash,
    pub cover_message: Option<String>,
    pub resume_credential_hashes: Vec<ActionHash>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AdvanceApplicationInput {
    pub application_hash: ActionHash,
    pub new_status: ApplicationStatus,
}

// ============== Extern Functions ==============

/// Create a new job application in Draft status.
#[hdk_extern]
pub fn create_application(input: CreateApplicationInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let application = JobApplication {
        job_posting_hash: input.job_posting_hash.clone(),
        applicant: agent.clone(),
        cover_message: input.cover_message,
        resume_credential_hashes: input.resume_credential_hashes,
        status: ApplicationStatus::Draft,
        submitted_at: now,
        updated_at: now,
    };

    let action_hash = create_entry(EntryTypes::JobApplication(application))?;

    // Link: applicant -> application
    let agent_hash: AnyDhtHash = agent.into();
    create_link(agent_hash, action_hash.clone(), LinkTypes::AgentToApplication, vec![])?;

    // Link: job posting -> application
    create_link(
        input.job_posting_hash,
        action_hash.clone(),
        LinkTypes::JobPostingToApplication,
        vec![],
    )?;

    Ok(action_hash)
}

/// Submit a draft application (Draft -> Submitted).
#[hdk_extern]
pub fn submit_application(application_hash: ActionHash) -> ExternResult<ActionHash> {
    let record = get(application_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Application not found".into())))?;

    let mut app = deserialize_application(&record)?;

    if app.status != ApplicationStatus::Draft {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only submit applications in Draft status".into()
        )));
    }

    app.status = ApplicationStatus::Submitted;
    app.updated_at = sys_time()?;

    // State transition validated at integrity layer
    update_entry(application_hash, &app)
}

/// Advance an application to a new status (employer action).
/// State machine transitions are validated at the integrity layer.
#[hdk_extern]
pub fn advance_application(input: AdvanceApplicationInput) -> ExternResult<ActionHash> {
    let record = get(input.application_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Application not found".into())))?;

    let mut app = deserialize_application(&record)?;

    // Pre-check at coordinator level for better error messages
    if !app.status.can_transition_to(&input.new_status) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot transition from {:?} to {:?}. Valid: {:?}",
            app.status,
            input.new_status,
            app.status.valid_transitions()
        ))));
    }

    app.status = input.new_status;
    app.updated_at = sys_time()?;

    update_entry(input.application_hash, &app)
}

/// Withdraw an application (applicant action, available from any non-terminal state).
#[hdk_extern]
pub fn withdraw_application(application_hash: ActionHash) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let record = get(application_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Application not found".into())))?;

    let mut app = deserialize_application(&record)?;

    // Only the applicant can withdraw
    if app.applicant != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the applicant can withdraw their application".into()
        )));
    }

    if app.status.is_terminal() {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot withdraw: application is already in terminal state {:?}",
            app.status
        ))));
    }

    app.status = ApplicationStatus::Withdrawn;
    app.updated_at = sys_time()?;

    update_entry(application_hash, &app)
}

/// List all applications submitted by the calling agent.
#[hdk_extern]
pub fn list_my_applications(_: ()) -> ExternResult<Vec<(ActionHash, JobApplication)>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_hash: AnyDhtHash = agent.into();

    let links = get_links(
        LinkQuery::try_new(agent_hash, LinkTypes::AgentToApplication)?,
        GetStrategy::Local,
    )?;

    load_applications(links)
}

/// List all applications for a specific job posting (employer view).
/// Only the posting author should call this — access control is at the app layer.
#[hdk_extern]
pub fn list_applications_for_posting(posting_hash: ActionHash) -> ExternResult<Vec<(ActionHash, JobApplication)>> {
    let links = get_links(
        LinkQuery::try_new(posting_hash, LinkTypes::JobPostingToApplication)?,
        GetStrategy::Local,
    )?;

    load_applications(links)
}

/// Get a single application by hash.
#[hdk_extern]
pub fn get_application(application_hash: ActionHash) -> ExternResult<Option<JobApplication>> {
    let Some(record) = get(application_hash, GetOptions::default())? else {
        return Ok(None);
    };
    Ok(Some(deserialize_application(&record)?))
}
