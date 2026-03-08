//! Coordination Coordinator Zome
//! Team management, zone assignments, SITREPs, and agent check-ins

use emergency_coordination_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
};

/// Summary of a single active disaster
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct DisasterSummary {
    pub id: String,
    pub title: String,
    pub severity: String,
    pub disaster_type: String,
}

/// Result of querying active disaster context
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct DisasterContextResult {
    pub active_disaster_count: u32,
    pub disasters: Vec<DisasterSummary>,
    pub highest_severity: Option<String>,
    pub error: Option<String>,
}

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("civic_bridge", requirement, action_name)
}

/// Form a new response team

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

#[hdk_extern]
pub fn form_team(input: FormTeamInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "form_team")?;
    if input.name.trim().is_empty() || input.name.len() > 128 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Team name must be 1-128 non-whitespace characters".into()
        )));
    }
    if input.id.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Team ID cannot be empty or whitespace-only".into()
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
    require_consciousness(&requirement_for_proposal(), "assign_to_zone")?;
    if input.objective.trim().is_empty() || input.objective.len() > 1024 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Objective must be 1-1024 non-whitespace characters".into()
        )));
    }

    // Deduplication: check if this team is already assigned to this zone
    let existing_zone_teams = get_links(
        LinkQuery::try_new(input.zone_hash.clone(), LinkTypes::ZoneToTeam)?,
        GetStrategy::default(),
    )?;
    for link in &existing_zone_teams {
        if let Ok(target_hash) = ActionHash::try_from(link.target.clone()) {
            if target_hash == input.team_hash {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Team is already assigned to this zone".into()
                )));
            }
        }
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
    require_consciousness(&requirement_for_basic(), "submit_sitrep")?;
    if input.conditions.trim().is_empty() || input.conditions.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Conditions must be 1-4096 non-whitespace characters".into()
        )));
    }
    // Reject resources_needed entries that are empty or whitespace-only
    for resource in &input.resources_needed {
        if resource.trim().is_empty() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Resource needed entries cannot be empty or whitespace-only".into()
            )));
        }
    }
    // Reject hazard entries that are empty or whitespace-only
    for hazard in &input.hazards {
        if hazard.trim().is_empty() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Hazard entries cannot be empty or whitespace-only".into()
            )));
        }
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
    require_consciousness(&requirement_for_basic(), "checkin")?;
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
        if let Some(record) = get_latest_record(action_hash)? {
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
        if let Some(record) = get_latest_record(action_hash)? {
            teams.push(record);
        }
    }

    Ok(teams)
}

// =============================================================================
// CROSS-DOMAIN: emergency-coordination → emergency-incidents
// =============================================================================

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
        Ok(ZomeCallResponse::Ok(extern_io)) => extern_io
            .decode()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?,
        Ok(other) => {
            return Ok(DisasterContextResult {
                active_disaster_count: 0,
                disasters: vec![],
                highest_severity: None,
                error: Some(format!(
                    "Unexpected response from emergency_incidents: {:?}",
                    other
                )),
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

    for record in &disaster_records {
        if let Some(entry) = record.entry().as_option() {
            let bytes: SerializedBytes = SerializedBytes::try_from(entry.clone()).map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Serialize error: {:?}", e)))
            })?;
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes.bytes()) {
                let id = value
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let title = value
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let severity = value
                    .get("severity")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
                    .to_string();
                let disaster_type = value
                    .get("disaster_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
                    .to_string();

                summaries.push(DisasterSummary {
                    id,
                    title,
                    severity,
                    disaster_type,
                });
            }
        }
    }

    let severity_refs: Vec<&str> = summaries.iter().map(|s| s.severity.as_str()).collect();
    let highest_severity = determine_highest_severity(&severity_refs);

    Ok(DisasterContextResult {
        active_disaster_count: summaries.len() as u32,
        disasters: summaries,
        highest_severity,
        error: None,
    })
}

/// Determine the highest severity from a list of severity strings.
///
/// Returns the most critical severity present, using the ordering:
/// Critical > Severe > Moderate > Minor.
/// Returns `Some("Unknown")` if severities are non-empty but contain no recognized values.
/// Returns `None` if the input slice is empty.
pub fn determine_highest_severity(severities: &[&str]) -> Option<String> {
    if severities.is_empty() {
        return None;
    }

    let severity_order = ["Critical", "Severe", "Moderate", "Minor"];
    let mut highest_idx = severity_order.len();

    for s in severities {
        if let Some(idx) = severity_order.iter().position(|known| known == s) {
            if idx < highest_idx {
                highest_idx = idx;
            }
        }
    }

    if highest_idx < severity_order.len() {
        Some(severity_order[highest_idx].to_string())
    } else {
        Some("Unknown".to_string())
    }
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
        if let Some(record) = get_latest_record(action_hash)? {
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

    // ========================================================================
    // ENUM VARIANT SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn team_type_all_variants_serde() {
        let variants = vec![
            TeamType::SearchAndRescue,
            TeamType::Medical,
            TeamType::Logistics,
            TeamType::Communications,
            TeamType::Shelter,
            TeamType::Assessment,
            TeamType::HazMat,
            TeamType::Volunteer,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: TeamType = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn team_status_all_variants_serde() {
        let variants = vec![
            TeamStatus::Forming,
            TeamStatus::Active,
            TeamStatus::OnBreak,
            TeamStatus::Disbanded,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: TeamStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn agent_status_all_variants_serde() {
        let variants = vec![
            AgentStatus::Active,
            AgentStatus::NeedsRelief,
            AgentStatus::Injured,
            AgentStatus::Evacuating,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: AgentStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn connectivity_status_all_variants_serde() {
        let variants = vec![
            ConnectivityStatus::Online,
            ConnectivityStatus::Intermittent,
            ConnectivityStatus::Offline,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: ConnectivityStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn access_status_all_variants_serde() {
        let variants = vec![
            AccessStatus::Open,
            AccessStatus::Restricted,
            AccessStatus::Blocked,
            AccessStatus::Hazardous,
            AccessStatus::Flooded,
            AccessStatus::Collapsed,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: AccessStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn assignment_status_all_variants_serde() {
        let variants = vec![
            AssignmentStatus::Active,
            AssignmentStatus::Completed,
            AssignmentStatus::Cancelled,
            AssignmentStatus::Reassigned,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: AssignmentStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    // ========================================================================
    // INPUT STRUCT SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn form_team_input_serde_roundtrip() {
        let agent_a = AgentPubKey::from_raw_36(vec![0xab; 36]);
        let agent_b = AgentPubKey::from_raw_36(vec![0xcd; 36]);
        let input = FormTeamInput {
            id: "team-alpha".to_string(),
            name: "Search & Rescue Alpha".to_string(),
            team_type: TeamType::SearchAndRescue,
            members: vec![agent_a.clone(), agent_b],
            lead: agent_a,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: FormTeamInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "team-alpha");
        assert_eq!(back.name, "Search & Rescue Alpha");
        assert_eq!(back.members.len(), 2);
    }

    #[test]
    fn assign_to_zone_input_serde_roundtrip() {
        let input = AssignToZoneInput {
            team_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            zone_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            zone_hash_for_team: ActionHash::from_raw_36(vec![0xcd; 36]),
            objective: "Clear sector 7 for survivors".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: AssignToZoneInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.objective, "Clear sector 7 for survivors");
    }

    #[test]
    fn submit_sitrep_input_serde_roundtrip() {
        let input = SubmitSitrepInput {
            team_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            zone_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            conditions: "Heavy flooding, partial building collapse".to_string(),
            casualties_found: 3,
            resources_needed: vec!["medical supplies".into(), "boats".into()],
            hazards: vec!["downed power lines".into()],
            access_status: AccessStatus::Restricted,
            synced: false,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SubmitSitrepInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.conditions, "Heavy flooding, partial building collapse");
        assert_eq!(back.casualties_found, 3);
        assert_eq!(back.resources_needed.len(), 2);
        assert_eq!(back.hazards.len(), 1);
        assert_eq!(back.access_status, AccessStatus::Restricted);
        assert!(!back.synced);
    }

    #[test]
    fn submit_sitrep_input_empty_vecs_serde() {
        let input = SubmitSitrepInput {
            team_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            zone_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            conditions: "All clear".to_string(),
            casualties_found: 0,
            resources_needed: vec![],
            hazards: vec![],
            access_status: AccessStatus::Open,
            synced: true,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SubmitSitrepInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.casualties_found, 0);
        assert!(back.resources_needed.is_empty());
        assert!(back.hazards.is_empty());
        assert!(back.synced);
    }

    #[test]
    fn checkin_input_serde_roundtrip() {
        let input = CheckinInput {
            lat: 32.9483,
            lon: -96.7299,
            status: AgentStatus::Active,
            battery_level: Some(85),
            connectivity: ConnectivityStatus::Online,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CheckinInput = serde_json::from_str(&json).unwrap();
        assert!((back.lat - 32.9483).abs() < f64::EPSILON);
        assert!((back.lon - (-96.7299)).abs() < f64::EPSILON);
        assert_eq!(back.status, AgentStatus::Active);
        assert_eq!(back.battery_level, Some(85));
        assert_eq!(back.connectivity, ConnectivityStatus::Online);
    }

    #[test]
    fn checkin_input_no_battery_serde() {
        let input = CheckinInput {
            lat: 0.0,
            lon: 0.0,
            status: AgentStatus::Injured,
            battery_level: None,
            connectivity: ConnectivityStatus::Offline,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CheckinInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.battery_level, None);
        assert_eq!(back.status, AgentStatus::Injured);
        assert_eq!(back.connectivity, ConnectivityStatus::Offline);
    }

    // ========================================================================
    // SEVERITY ORDERING LOGIC TESTS
    // ========================================================================

    #[test]
    fn severity_empty_returns_none() {
        assert_eq!(determine_highest_severity(&[]), None);
    }

    #[test]
    fn severity_single_critical() {
        assert_eq!(
            determine_highest_severity(&["Critical"]),
            Some("Critical".to_string())
        );
    }

    #[test]
    fn severity_single_minor() {
        assert_eq!(
            determine_highest_severity(&["Minor"]),
            Some("Minor".to_string())
        );
    }

    #[test]
    fn severity_critical_beats_severe() {
        assert_eq!(
            determine_highest_severity(&["Severe", "Critical"]),
            Some("Critical".to_string())
        );
    }

    #[test]
    fn severity_critical_beats_all() {
        assert_eq!(
            determine_highest_severity(&["Minor", "Moderate", "Severe", "Critical"]),
            Some("Critical".to_string())
        );
    }

    #[test]
    fn severity_severe_beats_moderate() {
        assert_eq!(
            determine_highest_severity(&["Moderate", "Severe"]),
            Some("Severe".to_string())
        );
    }

    #[test]
    fn severity_moderate_beats_minor() {
        assert_eq!(
            determine_highest_severity(&["Minor", "Moderate"]),
            Some("Moderate".to_string())
        );
    }

    #[test]
    fn severity_unknown_when_no_recognized_values() {
        assert_eq!(
            determine_highest_severity(&["LowPriority", "Advisory"]),
            Some("Unknown".to_string())
        );
    }

    #[test]
    fn severity_recognized_beats_unknown() {
        assert_eq!(
            determine_highest_severity(&["Advisory", "Minor", "LowPriority"]),
            Some("Minor".to_string())
        );
    }

    #[test]
    fn severity_case_sensitive() {
        // "critical" (lowercase) is not recognized
        assert_eq!(
            determine_highest_severity(&["critical"]),
            Some("Unknown".to_string())
        );
    }

    #[test]
    fn severity_duplicates_handled() {
        assert_eq!(
            determine_highest_severity(&["Severe", "Severe", "Severe"]),
            Some("Severe".to_string())
        );
    }

    #[test]
    fn severity_single_unknown_returns_unknown() {
        assert_eq!(
            determine_highest_severity(&["Informational"]),
            Some("Unknown".to_string())
        );
    }

    // ========================================================================
    // HARDENING: Whitespace-only validation tests
    // ========================================================================

    #[test]
    fn form_team_input_whitespace_only_name_serde() {
        // Whitespace-only name should serialize fine (validation is coordinator-side)
        let agent_a = AgentPubKey::from_raw_36(vec![0xab; 36]);
        let input = FormTeamInput {
            id: "team-ws".to_string(),
            name: "   \t  ".to_string(),
            team_type: TeamType::SearchAndRescue,
            members: vec![agent_a.clone()],
            lead: agent_a,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: FormTeamInput = serde_json::from_str(&json).unwrap();
        // Whitespace-only should be caught by coordinator trim() check
        assert!(back.name.trim().is_empty());
    }

    #[test]
    fn form_team_input_whitespace_only_id_serde() {
        let agent_a = AgentPubKey::from_raw_36(vec![0xab; 36]);
        let input = FormTeamInput {
            id: "   ".to_string(),
            name: "Valid Name".to_string(),
            team_type: TeamType::Medical,
            members: vec![agent_a.clone()],
            lead: agent_a,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: FormTeamInput = serde_json::from_str(&json).unwrap();
        assert!(back.id.trim().is_empty());
    }

    #[test]
    fn assign_to_zone_whitespace_only_objective_serde() {
        let input = AssignToZoneInput {
            team_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            zone_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            zone_hash_for_team: ActionHash::from_raw_36(vec![0xcd; 36]),
            objective: "  \n\t  ".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: AssignToZoneInput = serde_json::from_str(&json).unwrap();
        assert!(back.objective.trim().is_empty());
    }

    #[test]
    fn submit_sitrep_whitespace_only_conditions_serde() {
        let input = SubmitSitrepInput {
            team_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            zone_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            conditions: "   \t\n   ".to_string(),
            casualties_found: 0,
            resources_needed: vec![],
            hazards: vec![],
            access_status: AccessStatus::Open,
            synced: false,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SubmitSitrepInput = serde_json::from_str(&json).unwrap();
        assert!(back.conditions.trim().is_empty());
    }

    #[test]
    fn submit_sitrep_whitespace_only_resource_detected() {
        let input = SubmitSitrepInput {
            team_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            zone_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            conditions: "Valid conditions".to_string(),
            casualties_found: 0,
            resources_needed: vec!["   ".to_string(), "valid resource".to_string()],
            hazards: vec![],
            access_status: AccessStatus::Open,
            synced: false,
        };
        // The first resource entry is whitespace-only
        assert!(input.resources_needed[0].trim().is_empty());
        assert!(!input.resources_needed[1].trim().is_empty());
    }

    #[test]
    fn submit_sitrep_whitespace_only_hazard_detected() {
        let input = SubmitSitrepInput {
            team_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            zone_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            conditions: "Valid conditions".to_string(),
            casualties_found: 0,
            resources_needed: vec![],
            hazards: vec!["\t".to_string()],
            access_status: AccessStatus::Open,
            synced: false,
        };
        assert!(input.hazards[0].trim().is_empty());
    }

    #[test]
    fn submit_sitrep_empty_resource_entry_detected() {
        let input = SubmitSitrepInput {
            team_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            zone_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            conditions: "Valid conditions".to_string(),
            casualties_found: 0,
            resources_needed: vec!["".to_string()],
            hazards: vec![],
            access_status: AccessStatus::Open,
            synced: false,
        };
        assert!(input.resources_needed[0].trim().is_empty());
    }

    // ========================================================================
    // HARDENING: Severity determinism validation
    // ========================================================================

    #[test]
    fn severity_determinism_same_inputs_same_output() {
        // Verify that the same inputs always produce the same output
        // regardless of ordering in the severity list
        let inputs = &["Minor", "Severe", "Moderate", "Critical"];
        let result1 = determine_highest_severity(inputs);
        let result2 = determine_highest_severity(inputs);
        let result3 = determine_highest_severity(inputs);
        assert_eq!(result1, result2);
        assert_eq!(result2, result3);
        assert_eq!(result1, Some("Critical".to_string()));
    }

    #[test]
    fn severity_determinism_different_orderings() {
        let ordering_a = &["Minor", "Severe", "Moderate", "Critical"];
        let ordering_b = &["Critical", "Moderate", "Severe", "Minor"];
        let ordering_c = &["Moderate", "Critical", "Minor", "Severe"];
        let result_a = determine_highest_severity(ordering_a);
        let result_b = determine_highest_severity(ordering_b);
        let result_c = determine_highest_severity(ordering_c);
        assert_eq!(result_a, result_b);
        assert_eq!(result_b, result_c);
        assert_eq!(result_a, Some("Critical".to_string()));
    }

    #[test]
    fn severity_determinism_with_unknowns_mixed() {
        let a = &["Unknown", "Advisory", "Moderate"];
        let b = &["Moderate", "Unknown", "Advisory"];
        assert_eq!(determine_highest_severity(a), determine_highest_severity(b));
        assert_eq!(determine_highest_severity(a), Some("Moderate".to_string()));
    }

    #[test]
    fn severity_determinism_all_unknown() {
        let a = &["Foo", "Bar", "Baz"];
        let b = &["Baz", "Foo", "Bar"];
        assert_eq!(determine_highest_severity(a), determine_highest_severity(b));
        assert_eq!(determine_highest_severity(a), Some("Unknown".to_string()));
    }

    // ========================================================================
    // HARDENING: Checkin input edge cases
    // ========================================================================

    #[test]
    fn checkin_input_boundary_lat_lon_serde() {
        // Boundary values at exact limits
        let input = CheckinInput {
            lat: 90.0,
            lon: -180.0,
            status: AgentStatus::Active,
            battery_level: Some(100),
            connectivity: ConnectivityStatus::Online,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CheckinInput = serde_json::from_str(&json).unwrap();
        assert!((back.lat - 90.0).abs() < f64::EPSILON);
        assert!((back.lon - (-180.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn checkin_input_negative_boundary_serde() {
        let input = CheckinInput {
            lat: -90.0,
            lon: 180.0,
            status: AgentStatus::Evacuating,
            battery_level: Some(0),
            connectivity: ConnectivityStatus::Offline,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CheckinInput = serde_json::from_str(&json).unwrap();
        assert!((back.lat - (-90.0)).abs() < f64::EPSILON);
        assert!((back.lon - 180.0).abs() < f64::EPSILON);
        assert_eq!(back.battery_level, Some(0));
    }
}
