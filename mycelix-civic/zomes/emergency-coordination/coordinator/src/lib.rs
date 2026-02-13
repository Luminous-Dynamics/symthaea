//! Coordination Coordinator Zome
//! Team management, zone assignments, SITREPs, and agent check-ins

use emergency_coordination_integrity::*;
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

    sitreps.sort_by_key(|a| a.action().timestamp());
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

// =============================================================================
// CROSS-DOMAIN: emergency-coordination → emergency-incidents
// =============================================================================

/// Context about active disasters for a zone assignment
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DisasterContextResult {
    pub active_disaster_count: u32,
    pub disasters: Vec<DisasterSummary>,
    pub highest_severity: Option<String>,
    pub error: Option<String>,
}

/// Summary of an active disaster
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DisasterSummary {
    pub id: String,
    pub title: String,
    pub severity: String,
    pub disaster_type: String,
}

/// Get disaster context for zone assignment.
///
/// Cross-domain call: emergency-coordination → emergency-incidents via CallTargetCell::Local.
/// Provides active disaster information so teams can be assigned with proper context.
#[hdk_extern]
pub fn get_disaster_context(_: ()) -> ExternResult<DisasterContextResult> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("emergency_incidents"),
        FunctionName::from("get_active_disasters"),
        None,
        (),
    );

    let disaster_records: Vec<Record> = match &response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            extern_io.decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?
        }
        Ok(other) => {
            return Ok(DisasterContextResult {
                active_disaster_count: 0,
                disasters: vec![],
                highest_severity: None,
                error: Some(format!("Unexpected response from emergency_incidents: {:?}", other)),
            });
        }
        Err(e) => {
            return Ok(DisasterContextResult {
                active_disaster_count: 0,
                disasters: vec![],
                highest_severity: None,
                error: Some(format!("Failed to call emergency_incidents: {:?}", e)),
            });
        }
    };

    let mut summaries = Vec::new();
    let severity_order = ["Critical", "Severe", "Moderate", "Minor"];
    let mut highest_idx = severity_order.len();

    for record in &disaster_records {
        if let Some(entry) = record.entry().as_option() {
            let bytes: SerializedBytes = SerializedBytes::try_from(entry.clone())
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Serialize error: {:?}", e))))?;
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes.bytes()) {
                let id = value.get("id").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                let title = value.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let severity = value.get("severity").and_then(|v| v.as_str()).unwrap_or("Unknown").to_string();
                let disaster_type = value.get("disaster_type").and_then(|v| v.as_str()).unwrap_or("Unknown").to_string();

                // Track highest severity
                if let Some(idx) = severity_order.iter().position(|s| *s == severity) {
                    if idx < highest_idx {
                        highest_idx = idx;
                    }
                }

                summaries.push(DisasterSummary {
                    id,
                    title,
                    severity,
                    disaster_type,
                });
            }
        }
    }

    let highest_severity = if highest_idx < severity_order.len() {
        Some(severity_order[highest_idx].to_string())
    } else if !summaries.is_empty() {
        Some("Unknown".to_string())
    } else {
        None
    };

    Ok(DisasterContextResult {
        active_disaster_count: summaries.len() as u32,
        disasters: summaries,
        highest_severity,
        error: None,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disaster_context_result_serde_roundtrip() {
        let r = DisasterContextResult {
            active_disaster_count: 2,
            disasters: vec![
                DisasterSummary {
                    id: "DIS-1".into(),
                    title: "Winter Storm".into(),
                    severity: "Severe".into(),
                    disaster_type: "Weather".into(),
                },
                DisasterSummary {
                    id: "DIS-2".into(),
                    title: "Flood".into(),
                    severity: "Critical".into(),
                    disaster_type: "Flood".into(),
                },
            ],
            highest_severity: Some("Critical".into()),
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: DisasterContextResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.active_disaster_count, 2);
        assert_eq!(r2.disasters.len(), 2);
        assert_eq!(r2.highest_severity.as_deref(), Some("Critical"));
        assert!(r2.error.is_none());
    }

    #[test]
    fn disaster_context_empty_no_disasters() {
        let r = DisasterContextResult {
            active_disaster_count: 0,
            disasters: vec![],
            highest_severity: None,
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: DisasterContextResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.active_disaster_count, 0);
        assert!(r2.disasters.is_empty());
        assert!(r2.highest_severity.is_none());
    }

    #[test]
    fn disaster_summary_serde_roundtrip() {
        let s = DisasterSummary {
            id: "DIS-5".into(),
            title: "Earthquake".into(),
            severity: "Critical".into(),
            disaster_type: "Seismic".into(),
        };
        let json = serde_json::to_string(&s).unwrap();
        let s2: DisasterSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(s2.id, "DIS-5");
        assert_eq!(s2.title, "Earthquake");
        assert_eq!(s2.severity, "Critical");
        assert_eq!(s2.disaster_type, "Seismic");
    }

    #[test]
    fn disaster_context_error_state() {
        let r = DisasterContextResult {
            active_disaster_count: 0,
            disasters: vec![],
            highest_severity: None,
            error: Some("Failed to call emergency_incidents: timeout".into()),
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: DisasterContextResult = serde_json::from_str(&json).unwrap();
        assert!(r2.error.as_ref().unwrap().contains("timeout"));
    }
}
