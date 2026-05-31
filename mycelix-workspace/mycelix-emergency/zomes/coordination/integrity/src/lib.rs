// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Coordination Integrity Zome
//! Teams, zones, SITREPs, and agent check-ins

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A response team
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Team {
    pub id: String,
    pub name: String,
    pub team_type: TeamType,
    pub members: Vec<AgentPubKey>,
    pub lead: AgentPubKey,
    pub assigned_zone: Option<ActionHash>,
    pub status: TeamStatus,
}

/// Types of response teams
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TeamType {
    SearchAndRescue,
    Medical,
    Logistics,
    Communications,
    Shelter,
    Assessment,
    HazMat,
    Volunteer,
}

/// Team operational status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TeamStatus {
    Forming,
    Active,
    OnBreak,
    Disbanded,
}

/// A zone assignment for a team
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Assignment {
    pub team_hash: ActionHash,
    pub zone_hash: ActionHash,
    pub objective: String,
    pub assigned_at: Timestamp,
    pub assigned_by: AgentPubKey,
    pub status: AssignmentStatus,
}

/// Assignment status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AssignmentStatus {
    Active,
    Completed,
    Cancelled,
    Reassigned,
}

/// A situation report from the field
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SituationReport {
    pub team_hash: ActionHash,
    pub zone_hash: ActionHash,
    pub timestamp: Timestamp,
    pub conditions: String,
    pub casualties_found: u32,
    pub resources_needed: Vec<String>,
    pub hazards: Vec<String>,
    pub access_status: AccessStatus,
    pub synced: bool,
}

/// Access status to a zone
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AccessStatus {
    Open,
    Restricted,
    Blocked,
    Hazardous,
    Flooded,
    Collapsed,
}

/// An agent location check-in
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Checkpoint {
    pub agent: AgentPubKey,
    pub lat: f64,
    pub lon: f64,
    pub timestamp: Timestamp,
    pub status: AgentStatus,
    pub battery_level: Option<u8>,
    pub connectivity: ConnectivityStatus,
}

/// Connectivity status of a field agent
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ConnectivityStatus {
    Online,
    Intermittent,
    Offline,
}

/// Status of a field agent
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AgentStatus {
    Active,
    NeedsRelief,
    Injured,
    Evacuating,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Team(Team),
    Assignment(Assignment),
    SituationReport(SituationReport),
    Checkpoint(Checkpoint),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllTeams,
    ActiveTeams,
    TeamToAssignment,
    TeamToSitrep,
    ZoneToTeam,
    AgentToCheckpoint,
    AgentToTeam,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Team(team) => validate_create_team(action, team),
                EntryTypes::Assignment(assignment) => {
                    validate_create_assignment(action, assignment)
                }
                EntryTypes::SituationReport(sitrep) => validate_create_sitrep(action, sitrep),
                EntryTypes::Checkpoint(checkpoint) => {
                    validate_create_checkpoint(action, checkpoint)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Team(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Assignment(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SituationReport(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Checkpoint(_) => Ok(ValidateCallbackResult::Invalid(
                    "Checkpoints are immutable; create a new one".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AllTeams => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ActiveTeams => Ok(ValidateCallbackResult::Valid),
            LinkTypes::TeamToAssignment => Ok(ValidateCallbackResult::Valid),
            LinkTypes::TeamToSitrep => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ZoneToTeam => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToCheckpoint => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToTeam => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::ActiveTeams => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ZoneToTeam => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_team(_action: Create, team: Team) -> ExternResult<ValidateCallbackResult> {
    if team.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Team ID cannot be empty".into(),
        ));
    }
    if team.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Team name cannot be empty".into(),
        ));
    }
    if team.members.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Team must have at least one member".into(),
        ));
    }
    if !team.members.contains(&team.lead) {
        return Ok(ValidateCallbackResult::Invalid(
            "Team lead must be a member".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_assignment(
    _action: Create,
    assignment: Assignment,
) -> ExternResult<ValidateCallbackResult> {
    if assignment.objective.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Assignment objective cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_sitrep(
    _action: Create,
    sitrep: SituationReport,
) -> ExternResult<ValidateCallbackResult> {
    if sitrep.conditions.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "SITREP conditions cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_checkpoint(
    _action: Create,
    checkpoint: Checkpoint,
) -> ExternResult<ValidateCallbackResult> {
    if checkpoint.lat < -90.0 || checkpoint.lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if checkpoint.lon < -180.0 || checkpoint.lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    if let Some(battery) = checkpoint.battery_level {
        if battery > 100 {
            return Ok(ValidateCallbackResult::Invalid(
                "Battery level cannot exceed 100".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}
