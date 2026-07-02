// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Projects Coordinator Zome
//!
//! Provides functions for climate project management and milestone tracking.
//! Uses HDK 0.6.0-dev.1 with LinkQuery::try_new() pattern.

use hdk::prelude::*;
use projects_integrity::*;

/// Get or create an anchor for the given string
fn get_or_create_anchor(anchor_text: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_text.to_string());
    let entry_hash = hash_entry(&anchor)?;

    // Check if anchor exists
    if get(entry_hash.clone(), GetOptions::default())?.is_none() {
        create_entry(&EntryTypes::Anchor(anchor))?;
    }

    Ok(entry_hash)
}

// ============================================================================
// Climate Project Management
// ============================================================================

/// Input for creating a climate project
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateProjectInput {
    pub id: String,
    pub name: String,
    pub project_type: ProjectType,
    pub location: Location,
    pub expected_credits: f64,
    pub start_date: i64,
}

/// Create a new climate project
#[hdk_extern]
pub fn create_climate_project(input: CreateProjectInput) -> ExternResult<Record> {
    let project = ClimateProject {
        id: input.id.clone(),
        name: input.name,
        project_type: input.project_type,
        location: input.location,
        expected_credits: input.expected_credits,
        start_date: input.start_date,
        verifier_did: None,
        status: ProjectStatus::Proposed,
    };

    let action_hash = create_entry(&EntryTypes::ClimateProject(project))?;

    // Link from all projects anchor
    let all_anchor = get_or_create_anchor("all_projects")?;
    create_link(
        all_anchor,
        action_hash.clone(),
        LinkTypes::AnchorToProjects,
        (),
    )?;

    // Link from project type anchor
    let type_anchor = get_or_create_anchor(&format!("type:{:?}", input.project_type))?;
    create_link(
        type_anchor,
        action_hash.clone(),
        LinkTypes::TypeToProjects,
        (),
    )?;

    // Link from status anchor
    let status_anchor = get_or_create_anchor("status:Proposed")?;
    create_link(
        status_anchor,
        action_hash.clone(),
        LinkTypes::StatusToProjects,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created project".into())))
}

/// Get a specific project by action hash
#[hdk_extern]
pub fn get_climate_project(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Get all climate projects
#[hdk_extern]
pub fn get_all_projects(_: ()) -> ExternResult<Vec<Record>> {
    let anchor_hash = get_or_create_anchor("all_projects")?;
    let query = LinkQuery::try_new(anchor_hash, LinkTypes::AnchorToProjects)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Get projects by type
#[hdk_extern]
pub fn get_projects_by_type(project_type: ProjectType) -> ExternResult<Vec<Record>> {
    let anchor_hash = get_or_create_anchor(&format!("type:{:?}", project_type))?;
    let query = LinkQuery::try_new(anchor_hash, LinkTypes::TypeToProjects)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Get projects by status
#[hdk_extern]
pub fn get_projects_by_status(status: ProjectStatus) -> ExternResult<Vec<Record>> {
    let anchor_hash = get_or_create_anchor(&format!("status:{:?}", status))?;
    let query = LinkQuery::try_new(anchor_hash, LinkTypes::StatusToProjects)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Input for verifying a project
#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyProjectInput {
    pub project_action_hash: ActionHash,
    pub verifier_did: String,
}

/// Verify a project (transitions from Proposed to Verified)
#[hdk_extern]
pub fn verify_project(input: VerifyProjectInput) -> ExternResult<Record> {
    let record = get(input.project_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Project not found".into())))?;

    let project: ClimateProject = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid project entry".into())))?;

    // Verify project is in Proposed status
    if project.status != ProjectStatus::Proposed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only proposed projects can be verified".into()
        )));
    }

    let updated = ClimateProject {
        verifier_did: Some(input.verifier_did.clone()),
        status: ProjectStatus::Verified,
        ..project
    };

    let action_hash = update_entry(input.project_action_hash.clone(), &EntryTypes::ClimateProject(updated))?;

    // Create update link
    create_link(
        input.project_action_hash,
        action_hash.clone(),
        LinkTypes::ProjectUpdates,
        (),
    )?;

    // Link from verifier
    let verifier_anchor = get_or_create_anchor(&format!("verifier:{}", input.verifier_did))?;
    create_link(
        verifier_anchor,
        action_hash.clone(),
        LinkTypes::VerifierToProjects,
        (),
    )?;

    // Update status links
    let status_anchor = get_or_create_anchor("status:Verified")?;
    create_link(
        status_anchor,
        action_hash.clone(),
        LinkTypes::StatusToProjects,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get verified project".into())))
}

/// Input for activating a project
#[derive(Serialize, Deserialize, Debug)]
pub struct ActivateProjectInput {
    pub project_action_hash: ActionHash,
}

/// Activate a project (transitions from Verified to Active)
#[hdk_extern]
pub fn activate_project(input: ActivateProjectInput) -> ExternResult<Record> {
    let record = get(input.project_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Project not found".into())))?;

    let project: ClimateProject = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid project entry".into())))?;

    // Verify project is in Verified status
    if project.status != ProjectStatus::Verified {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only verified projects can be activated".into()
        )));
    }

    let updated = ClimateProject {
        status: ProjectStatus::Active,
        ..project
    };

    let action_hash = update_entry(input.project_action_hash.clone(), &EntryTypes::ClimateProject(updated))?;

    // Create update link
    create_link(
        input.project_action_hash,
        action_hash.clone(),
        LinkTypes::ProjectUpdates,
        (),
    )?;

    // Update status links
    let status_anchor = get_or_create_anchor("status:Active")?;
    create_link(
        status_anchor,
        action_hash.clone(),
        LinkTypes::StatusToProjects,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get activated project".into())))
}

/// Input for completing a project
#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteProjectInput {
    pub project_action_hash: ActionHash,
}

/// Complete a project (transitions from Active to Completed)
#[hdk_extern]
pub fn complete_project(input: CompleteProjectInput) -> ExternResult<Record> {
    let record = get(input.project_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Project not found".into())))?;

    let project: ClimateProject = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid project entry".into())))?;

    // Verify project is in Active status
    if project.status != ProjectStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only active projects can be completed".into()
        )));
    }

    let updated = ClimateProject {
        status: ProjectStatus::Completed,
        ..project
    };

    let action_hash = update_entry(input.project_action_hash.clone(), &EntryTypes::ClimateProject(updated))?;

    // Create update link
    create_link(
        input.project_action_hash,
        action_hash.clone(),
        LinkTypes::ProjectUpdates,
        (),
    )?;

    // Update status links
    let status_anchor = get_or_create_anchor("status:Completed")?;
    create_link(
        status_anchor,
        action_hash.clone(),
        LinkTypes::StatusToProjects,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get completed project".into())))
}

/// Get projects verified by a specific verifier
#[hdk_extern]
pub fn get_projects_by_verifier(verifier_did: String) -> ExternResult<Vec<Record>> {
    let anchor_hash = get_or_create_anchor(&format!("verifier:{}", verifier_did))?;
    let query = LinkQuery::try_new(anchor_hash, LinkTypes::VerifierToProjects)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

// ============================================================================
// Milestone Management
// ============================================================================

/// Input for creating a milestone
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateMilestoneInput {
    pub project_action_hash: ActionHash,
    pub project_id: String,
    pub title: String,
    pub description: String,
    pub target_date: i64,
}

/// Create a new milestone for a project
#[hdk_extern]
pub fn create_milestone(input: CreateMilestoneInput) -> ExternResult<Record> {
    // Verify project exists
    let _project_record = get(input.project_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Project not found".into())))?;

    let milestone = ProjectMilestone {
        project_id: input.project_id,
        title: input.title,
        description: input.description,
        target_date: input.target_date,
        completed_at: None,
        credits_issued: None,
        verified_by: None,
    };

    let action_hash = create_entry(&EntryTypes::ProjectMilestone(milestone))?;

    // Link from project to milestone
    create_link(
        input.project_action_hash,
        action_hash.clone(),
        LinkTypes::ProjectToMilestones,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created milestone".into())))
}

/// Get all milestones for a project
#[hdk_extern]
pub fn get_project_milestones(project_action_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let query = LinkQuery::try_new(project_action_hash, LinkTypes::ProjectToMilestones)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Input for completing a milestone
#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteMilestoneInput {
    pub milestone_action_hash: ActionHash,
    pub credits_issued: Option<f64>,
    pub verifier_did: String,
}

/// Complete a milestone and optionally issue credits
#[hdk_extern]
pub fn complete_milestone(input: CompleteMilestoneInput) -> ExternResult<Record> {
    let record = get(input.milestone_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Milestone not found".into())))?;

    let milestone: ProjectMilestone = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid milestone entry".into())))?;

    // Verify milestone is not already completed
    if milestone.completed_at.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Milestone is already completed".into()
        )));
    }

    let now = sys_time()?;
    let completed_at = now.as_micros() / 1_000_000; // Convert to seconds

    let updated = ProjectMilestone {
        completed_at: Some(completed_at),
        credits_issued: input.credits_issued,
        verified_by: Some(input.verifier_did),
        ..milestone
    };

    let action_hash = update_entry(input.milestone_action_hash, &EntryTypes::ProjectMilestone(updated))?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get completed milestone".into())))
}

// ============================================================================
// Statistics
// ============================================================================

/// Summary of projects
#[derive(Serialize, Deserialize, Debug)]
pub struct ProjectsSummary {
    pub total_projects: u64,
    pub proposed_count: u64,
    pub verified_count: u64,
    pub active_count: u64,
    pub completed_count: u64,
    pub total_expected_credits: f64,
    pub reforestation_count: u64,
    pub renewable_energy_count: u64,
    pub methane_capture_count: u64,
    pub ocean_restoration_count: u64,
    pub direct_air_capture_count: u64,
}

/// Get a summary of all projects
#[hdk_extern]
pub fn get_projects_summary(_: ()) -> ExternResult<ProjectsSummary> {
    let records = get_all_projects(())?;

    let mut summary = ProjectsSummary {
        total_projects: 0,
        proposed_count: 0,
        verified_count: 0,
        active_count: 0,
        completed_count: 0,
        total_expected_credits: 0.0,
        reforestation_count: 0,
        renewable_energy_count: 0,
        methane_capture_count: 0,
        ocean_restoration_count: 0,
        direct_air_capture_count: 0,
    };

    for record in records {
        if let Some(project) = record
            .entry()
            .to_app_option::<ClimateProject>()
            .ok()
            .flatten()
        {
            summary.total_projects += 1;
            summary.total_expected_credits += project.expected_credits;

            match project.status {
                ProjectStatus::Proposed => summary.proposed_count += 1,
                ProjectStatus::Verified => summary.verified_count += 1,
                ProjectStatus::Active => summary.active_count += 1,
                ProjectStatus::Completed => summary.completed_count += 1,
            }

            match project.project_type {
                ProjectType::Reforestation => summary.reforestation_count += 1,
                ProjectType::RenewableEnergy => summary.renewable_energy_count += 1,
                ProjectType::MethaneCapture => summary.methane_capture_count += 1,
                ProjectType::OceanRestoration => summary.ocean_restoration_count += 1,
                ProjectType::DirectAirCapture => summary.direct_air_capture_count += 1,
            }
        }
    }

    Ok(summary)
}
