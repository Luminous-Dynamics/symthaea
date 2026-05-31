// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Coordination Coordinator Zome
//! Team management, zone assignments, SITREPs, and agent check-ins

use coordination_integrity::*;
use hdk::prelude::*;

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Form a new response team
#[hdk_extern]
pub fn form_team(input: FormTeamInput) -> ExternResult<Record> {
    if input.name.is_empty() || input.name.len() > 128 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Team name must be 1-128 characters".into()
        )));
    }
    if input.members.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Team must have at least one member".into()
        )));
    }
    if !input.members.contains(&input.lead) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Team lead must be a member".into()
        )));
    }

    let team = Team {
        id: input.id,
        name: input.name,
        team_type: input.team_type,
        members: input.members.clone(),
        lead: input.lead,
        assigned_zone: None,
        status: TeamStatus::Forming,
    };

    let action_hash = create_entry(&EntryTypes::Team(team))?;

    // Link to all teams
    create_entry(&EntryTypes::Anchor(Anchor("all_teams".to_string())))?;
    create_link(
        anchor_hash("all_teams")?,
        action_hash.clone(),
        LinkTypes::AllTeams,
        (),
    )?;

    // Link to active teams
    create_entry(&EntryTypes::Anchor(Anchor("active_teams".to_string())))?;
    create_link(
        anchor_hash("active_teams")?,
        action_hash.clone(),
        LinkTypes::ActiveTeams,
        (),
    )?;

    // Link each member to the team
    for member in &input.members {
        create_link(
            member.clone(),
            action_hash.clone(),
            LinkTypes::AgentToTeam,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created team".into()
    )))
}

/// Input for forming a team
#[derive(Serialize, Deserialize, Debug)]
pub struct FormTeamInput {
    pub id: String,
    pub name: String,
    pub team_type: TeamType,
    pub members: Vec<AgentPubKey>,
    pub lead: AgentPubKey,
}

/// Assign a team to an operational zone
#[hdk_extern]
pub fn assign_to_zone(input: AssignToZoneInput) -> ExternResult<Record> {
    if input.objective.is_empty() || input.objective.len() > 1024 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Objective must be 1-1024 characters".into()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?;

    let assignment = Assignment {
        team_hash: input.team_hash.clone(),
        zone_hash: input.zone_hash.clone(),
        objective: input.objective,
        assigned_at: now,
        assigned_by: agent_info.agent_initial_pubkey,
        status: AssignmentStatus::Active,
    };

    let action_hash = create_entry(&EntryTypes::Assignment(assignment))?;

    // Link team to assignment
    create_link(
        input.team_hash.clone(),
        action_hash.clone(),
        LinkTypes::TeamToAssignment,
        (),
    )?;

    // Link zone to team
    create_link(
        input.zone_hash,
        input.team_hash.clone(),
        LinkTypes::ZoneToTeam,
        (),
    )?;

    // Update team with assigned zone
    let team_record = get(input.team_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Team not found".into())))?;

    let current_team: Team = team_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid team entry".into()
        )))?;

    let updated_team = Team {
        assigned_zone: Some(input.zone_hash_for_team),
        status: TeamStatus::Active,
        ..current_team
    };

    update_entry(
        team_record.action_address().clone(),
        &EntryTypes::Team(updated_team),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created assignment".into()
    )))
}

/// Input for assigning a team to a zone
#[derive(Serialize, Deserialize, Debug)]
pub struct AssignToZoneInput {
    pub team_hash: ActionHash,
    pub zone_hash: ActionHash,
    pub zone_hash_for_team: ActionHash,
    pub objective: String,
}

/// Submit a situation report from the field
#[hdk_extern]
pub fn submit_sitrep(input: SubmitSitrepInput) -> ExternResult<Record> {
    if input.conditions.is_empty() || input.conditions.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Conditions must be 1-4096 characters".into()
        )));
    }

    let now = sys_time()?;

    let sitrep = SituationReport {
        team_hash: input.team_hash.clone(),
        zone_hash: input.zone_hash.clone(),
        timestamp: now,
        conditions: input.conditions,
        casualties_found: input.casualties_found,
        resources_needed: input.resources_needed,
        hazards: input.hazards,
        access_status: input.access_status,
        synced: input.synced,
    };

    let action_hash = create_entry(&EntryTypes::SituationReport(sitrep))?;

    // Link team to sitrep
    create_link(
        input.team_hash,
        action_hash.clone(),
        LinkTypes::TeamToSitrep,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created SITREP".into()
    )))
}

/// Input for submitting a SITREP
#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitSitrepInput {
    pub team_hash: ActionHash,
    pub zone_hash: ActionHash,
    pub conditions: String,
    pub casualties_found: u32,
    pub resources_needed: Vec<String>,
    pub hazards: Vec<String>,
    pub access_status: AccessStatus,
    pub synced: bool,
}

/// Agent location check-in
#[hdk_extern]
pub fn checkin(input: CheckinInput) -> ExternResult<Record> {
    if input.lat < -90.0 || input.lat > 90.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Latitude must be between -90 and 90".into()
        )));
    }
    if input.lon < -180.0 || input.lon > 180.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Longitude must be between -180 and 180".into()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?;

    let checkpoint = Checkpoint {
        agent: agent_info.agent_initial_pubkey.clone(),
        lat: input.lat,
        lon: input.lon,
        timestamp: now,
        status: input.status,
        battery_level: input.battery_level,
        connectivity: input.connectivity,
    };

    let action_hash = create_entry(&EntryTypes::Checkpoint(checkpoint))?;

    // Link agent to checkpoint
    create_link(
        agent_info.agent_initial_pubkey,
        action_hash.clone(),
        LinkTypes::AgentToCheckpoint,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created checkpoint".into()
    )))
}

/// Input for agent check-in
#[derive(Serialize, Deserialize, Debug)]
pub struct CheckinInput {
    pub lat: f64,
    pub lon: f64,
    pub status: AgentStatus,
    pub battery_level: Option<u8>,
    pub connectivity: ConnectivityStatus,
}

/// Get all SITREPs for a team
#[hdk_extern]
pub fn get_team_sitreps(team_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(team_hash, LinkTypes::TeamToSitrep)?,
        GetStrategy::default(),
    )?;

    let mut sitreps = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            sitreps.push(record);
        }
    }

    sitreps.sort_by(|a, b| a.action().timestamp().cmp(&b.action().timestamp()));
    Ok(sitreps)
}

/// Get teams assigned to a zone
#[hdk_extern]
pub fn get_zone_teams(zone_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(zone_hash, LinkTypes::ZoneToTeam)?,
        GetStrategy::default(),
    )?;

    let mut teams = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            teams.push(record);
        }
    }

    Ok(teams)
}

/// Get the latest location for an agent
#[hdk_extern]
pub fn get_agent_location(agent: AgentPubKey) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToCheckpoint)?,
        GetStrategy::default(),
    )?;

    let mut latest: Option<Record> = None;
    let mut latest_ts = Timestamp::from_micros(0);

    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            let ts = record.action().timestamp();
            if ts > latest_ts {
                latest_ts = ts;
                latest = Some(record);
            }
        }
    }

    Ok(latest)
}
