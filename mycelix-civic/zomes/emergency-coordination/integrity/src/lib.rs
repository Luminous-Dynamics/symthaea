//! Coordination Integrity Zome
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

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // RESULT HELPERS
    // ========================================================================

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn invalid_msg(result: &ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => msg.clone(),
            _ => panic!("Expected Invalid, got {:?}", result),
        }
    }

    // ========================================================================
    // CONSTRUCTION HELPERS
    // ========================================================================

    fn fake_create() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: EntryRateWeight::default(),
        }
    }

    fn agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn agent2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![2u8; 36])
    }

    fn agent3() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![3u8; 36])
    }

    fn ts() -> Timestamp {
        Timestamp::from_micros(0)
    }

    fn ah() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn make_team() -> Team {
        let lead = agent();
        Team {
            id: "team-1".into(),
            name: "Search & Rescue Alpha".into(),
            team_type: TeamType::SearchAndRescue,
            members: vec![lead.clone(), agent2()],
            lead,
            assigned_zone: None,
            status: TeamStatus::Active,
        }
    }

    fn make_assignment() -> Assignment {
        Assignment {
            team_hash: ah(),
            zone_hash: ah(),
            objective: "Clear sector 7 for survivors".into(),
            assigned_at: ts(),
            assigned_by: agent(),
            status: AssignmentStatus::Active,
        }
    }

    fn make_sitrep() -> SituationReport {
        SituationReport {
            team_hash: ah(),
            zone_hash: ah(),
            timestamp: ts(),
            conditions: "Heavy flooding, partial building collapse".into(),
            casualties_found: 3,
            resources_needed: vec!["medical supplies".into(), "boats".into()],
            hazards: vec!["downed power lines".into()],
            access_status: AccessStatus::Restricted,
            synced: false,
        }
    }

    fn make_checkpoint() -> Checkpoint {
        Checkpoint {
            agent: agent(),
            lat: 32.9483,
            lon: -96.7299,
            timestamp: ts(),
            status: AgentStatus::Active,
            battery_level: Some(85),
            connectivity: ConnectivityStatus::Online,
        }
    }

    // ========================================================================
    // TEAM VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_team_passes() {
        let result = validate_create_team(fake_create(), make_team());
        assert!(is_valid(&result));
    }

    #[test]
    fn team_empty_id_rejected() {
        let mut team = make_team();
        team.id = "".into();
        let result = validate_create_team(fake_create(), team);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Team ID cannot be empty");
    }

    #[test]
    fn team_empty_name_rejected() {
        let mut team = make_team();
        team.name = "".into();
        let result = validate_create_team(fake_create(), team);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Team name cannot be empty");
    }

    #[test]
    fn team_empty_members_rejected() {
        let mut team = make_team();
        team.members = vec![];
        let result = validate_create_team(fake_create(), team);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Team must have at least one member"
        );
    }

    #[test]
    fn team_lead_not_in_members_rejected() {
        let mut team = make_team();
        team.lead = agent3();
        // members still contains agent() and agent2(), not agent3()
        let result = validate_create_team(fake_create(), team);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Team lead must be a member");
    }

    #[test]
    fn team_lead_is_sole_member_passes() {
        let sole = agent();
        let team = Team {
            id: "team-solo".into(),
            name: "Solo Recon".into(),
            team_type: TeamType::Assessment,
            members: vec![sole.clone()],
            lead: sole,
            assigned_zone: None,
            status: TeamStatus::Forming,
        };
        let result = validate_create_team(fake_create(), team);
        assert!(is_valid(&result));
    }

    #[test]
    fn team_all_fields_valid_with_assigned_zone_passes() {
        let mut team = make_team();
        team.assigned_zone = Some(ah());
        let result = validate_create_team(fake_create(), team);
        assert!(is_valid(&result));
    }

    #[test]
    fn team_every_type_variant_passes() {
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
        for variant in variants {
            let mut team = make_team();
            team.team_type = variant;
            let result = validate_create_team(fake_create(), team);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn team_every_status_variant_passes() {
        let variants = vec![
            TeamStatus::Forming,
            TeamStatus::Active,
            TeamStatus::OnBreak,
            TeamStatus::Disbanded,
        ];
        for variant in variants {
            let mut team = make_team();
            team.status = variant;
            let result = validate_create_team(fake_create(), team);
            assert!(is_valid(&result));
        }
    }

    // ========================================================================
    // ASSIGNMENT VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_assignment_passes() {
        let result = validate_create_assignment(fake_create(), make_assignment());
        assert!(is_valid(&result));
    }

    #[test]
    fn assignment_empty_objective_rejected() {
        let mut assignment = make_assignment();
        assignment.objective = "".into();
        let result = validate_create_assignment(fake_create(), assignment);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Assignment objective cannot be empty"
        );
    }

    #[test]
    fn assignment_whitespace_only_objective_passes() {
        // The validation uses is_empty(), not trim().is_empty()
        let mut assignment = make_assignment();
        assignment.objective = "   ".into();
        let result = validate_create_assignment(fake_create(), assignment);
        assert!(is_valid(&result));
    }

    #[test]
    fn assignment_every_status_variant_passes() {
        let variants = vec![
            AssignmentStatus::Active,
            AssignmentStatus::Completed,
            AssignmentStatus::Cancelled,
            AssignmentStatus::Reassigned,
        ];
        for variant in variants {
            let mut assignment = make_assignment();
            assignment.status = variant;
            let result = validate_create_assignment(fake_create(), assignment);
            assert!(is_valid(&result));
        }
    }

    // ========================================================================
    // SITREP VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_sitrep_passes() {
        let result = validate_create_sitrep(fake_create(), make_sitrep());
        assert!(is_valid(&result));
    }

    #[test]
    fn sitrep_empty_conditions_rejected() {
        let mut sitrep = make_sitrep();
        sitrep.conditions = "".into();
        let result = validate_create_sitrep(fake_create(), sitrep);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "SITREP conditions cannot be empty"
        );
    }

    #[test]
    fn sitrep_whitespace_only_conditions_passes() {
        // The validation uses is_empty(), not trim().is_empty()
        let mut sitrep = make_sitrep();
        sitrep.conditions = "  \t  ".into();
        let result = validate_create_sitrep(fake_create(), sitrep);
        assert!(is_valid(&result));
    }

    #[test]
    fn sitrep_zero_casualties_passes() {
        let mut sitrep = make_sitrep();
        sitrep.casualties_found = 0;
        let result = validate_create_sitrep(fake_create(), sitrep);
        assert!(is_valid(&result));
    }

    #[test]
    fn sitrep_empty_resources_and_hazards_passes() {
        let mut sitrep = make_sitrep();
        sitrep.resources_needed = vec![];
        sitrep.hazards = vec![];
        let result = validate_create_sitrep(fake_create(), sitrep);
        assert!(is_valid(&result));
    }

    #[test]
    fn sitrep_every_access_status_passes() {
        let variants = vec![
            AccessStatus::Open,
            AccessStatus::Restricted,
            AccessStatus::Blocked,
            AccessStatus::Hazardous,
            AccessStatus::Flooded,
            AccessStatus::Collapsed,
        ];
        for variant in variants {
            let mut sitrep = make_sitrep();
            sitrep.access_status = variant;
            let result = validate_create_sitrep(fake_create(), sitrep);
            assert!(is_valid(&result));
        }
    }

    // ========================================================================
    // CHECKPOINT VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_checkpoint_passes() {
        let result = validate_create_checkpoint(fake_create(), make_checkpoint());
        assert!(is_valid(&result));
    }

    #[test]
    fn checkpoint_lat_too_low_rejected() {
        let mut cp = make_checkpoint();
        cp.lat = -90.1;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Latitude must be between -90 and 90"
        );
    }

    #[test]
    fn checkpoint_lat_too_high_rejected() {
        let mut cp = make_checkpoint();
        cp.lat = 90.1;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Latitude must be between -90 and 90"
        );
    }

    #[test]
    fn checkpoint_lat_exactly_negative_90_passes() {
        let mut cp = make_checkpoint();
        cp.lat = -90.0;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn checkpoint_lat_exactly_90_passes() {
        let mut cp = make_checkpoint();
        cp.lat = 90.0;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn checkpoint_lat_zero_passes() {
        let mut cp = make_checkpoint();
        cp.lat = 0.0;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn checkpoint_lon_too_low_rejected() {
        let mut cp = make_checkpoint();
        cp.lon = -180.1;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    #[test]
    fn checkpoint_lon_too_high_rejected() {
        let mut cp = make_checkpoint();
        cp.lon = 180.1;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    #[test]
    fn checkpoint_lon_exactly_negative_180_passes() {
        let mut cp = make_checkpoint();
        cp.lon = -180.0;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn checkpoint_lon_exactly_180_passes() {
        let mut cp = make_checkpoint();
        cp.lon = 180.0;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn checkpoint_lon_zero_passes() {
        let mut cp = make_checkpoint();
        cp.lon = 0.0;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn checkpoint_battery_over_100_rejected() {
        let mut cp = make_checkpoint();
        cp.battery_level = Some(101);
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Battery level cannot exceed 100"
        );
    }

    #[test]
    fn checkpoint_battery_exactly_100_passes() {
        let mut cp = make_checkpoint();
        cp.battery_level = Some(100);
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn checkpoint_battery_zero_passes() {
        let mut cp = make_checkpoint();
        cp.battery_level = Some(0);
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn checkpoint_battery_none_passes() {
        let mut cp = make_checkpoint();
        cp.battery_level = None;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn checkpoint_battery_max_u8_rejected() {
        let mut cp = make_checkpoint();
        cp.battery_level = Some(u8::MAX); // 255
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
    }

    #[test]
    fn checkpoint_extreme_lat_lon_both_out_of_range() {
        let mut cp = make_checkpoint();
        cp.lat = 999.0;
        cp.lon = 999.0;
        let result = validate_create_checkpoint(fake_create(), cp);
        // Lat check comes first
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Latitude must be between -90 and 90"
        );
    }

    #[test]
    fn checkpoint_every_agent_status_passes() {
        let variants = vec![
            AgentStatus::Active,
            AgentStatus::NeedsRelief,
            AgentStatus::Injured,
            AgentStatus::Evacuating,
        ];
        for variant in variants {
            let mut cp = make_checkpoint();
            cp.status = variant;
            let result = validate_create_checkpoint(fake_create(), cp);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn checkpoint_every_connectivity_status_passes() {
        let variants = vec![
            ConnectivityStatus::Online,
            ConnectivityStatus::Intermittent,
            ConnectivityStatus::Offline,
        ];
        for variant in variants {
            let mut cp = make_checkpoint();
            cp.connectivity = variant;
            let result = validate_create_checkpoint(fake_create(), cp);
            assert!(is_valid(&result));
        }
    }
}
