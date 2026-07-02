// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Coordination Integrity Zome
//! Teams, zones, SITREPs, and agent check-ins

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

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
            tag,
            action: _,
        } => {
            let tag_len = tag.0.len();
            match link_type {
                LinkTypes::AllTeams => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::ActiveTeams => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::TeamToAssignment => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::TeamToSitrep => {
                    // SITREP links may carry condition metadata in tags
                    if tag_len > 512 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 512 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::ZoneToTeam => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::AgentToCheckpoint => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::AgentToTeam => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink { tag, action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            let result = check_link_author_match(
                original_action.action().author(),
                &action.author,
            );
            if result != ValidateCallbackResult::Valid {
                return Ok(result);
            }
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Delete link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}

fn validate_create_team(_action: Create, team: Team) -> ExternResult<ValidateCallbackResult> {
    if team.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Team ID cannot be empty".into(),
        ));
    }
    if team.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Team ID must be 256 characters or fewer".into(),
        ));
    }
    if team.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Team name cannot be empty".into(),
        ));
    }
    if team.name.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Team name must be 512 characters or fewer".into(),
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
    if assignment.objective.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Assignment objective cannot be empty".into(),
        ));
    }
    if assignment.objective.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Assignment objective must be 4096 characters or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_sitrep(
    _action: Create,
    sitrep: SituationReport,
) -> ExternResult<ValidateCallbackResult> {
    if sitrep.conditions.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "SITREP conditions cannot be empty".into(),
        ));
    }
    if sitrep.conditions.len() > 8192 {
        return Ok(ValidateCallbackResult::Invalid(
            "SITREP conditions must be 8192 characters or fewer".into(),
        ));
    }
    if sitrep.resources_needed.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many resources_needed entries (max 200)".into(),
        ));
    }
    for r in &sitrep.resources_needed {
        if r.len() > 512 {
            return Ok(ValidateCallbackResult::Invalid(
                "Individual resource_needed entry too long (max 512)".into(),
            ));
        }
    }
    if sitrep.hazards.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many hazards entries (max 200)".into(),
        ));
    }
    for h in &sitrep.hazards {
        if h.len() > 512 {
            return Ok(ValidateCallbackResult::Invalid(
                "Individual hazard entry too long (max 512)".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_checkpoint(
    _action: Create,
    checkpoint: Checkpoint,
) -> ExternResult<ValidateCallbackResult> {
    // NaN/Infinity guard: NaN comparisons silently pass range checks
    if !checkpoint.lat.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be a finite number".into(),
        ));
    }
    if !checkpoint.lon.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be a finite number".into(),
        ));
    }
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
        assert_eq!(invalid_msg(&result), "Team must have at least one member");
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
        assert_eq!(invalid_msg(&result), "Assignment objective cannot be empty");
    }

    #[test]
    fn assignment_whitespace_only_objective_rejected() {
        let mut assignment = make_assignment();
        assignment.objective = "   ".into();
        let result = validate_create_assignment(fake_create(), assignment);
        assert!(!is_valid(&result));
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
        assert_eq!(invalid_msg(&result), "SITREP conditions cannot be empty");
    }

    #[test]
    fn sitrep_whitespace_only_conditions_rejected() {
        let mut sitrep = make_sitrep();
        sitrep.conditions = "  \t  ".into();
        let result = validate_create_sitrep(fake_create(), sitrep);
        assert!(!is_valid(&result));
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
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    #[test]
    fn checkpoint_lat_too_high_rejected() {
        let mut cp = make_checkpoint();
        cp.lat = 90.1;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
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
        assert_eq!(invalid_msg(&result), "Battery level cannot exceed 100");
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
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
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

    // ========================================================================
    // SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn serde_roundtrip_team_type() {
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
    fn serde_roundtrip_team_status() {
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
    fn serde_roundtrip_assignment_status() {
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

    #[test]
    fn serde_roundtrip_access_status() {
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
    fn serde_roundtrip_agent_status() {
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
    fn serde_roundtrip_connectivity_status() {
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
    fn serde_roundtrip_anchor() {
        let anchor = Anchor("all_teams".into());
        let bytes = holochain_serialized_bytes::encode(&anchor).unwrap();
        let back: Anchor = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(anchor, back);
    }

    #[test]
    fn serde_roundtrip_team() {
        let team = make_team();
        let bytes = holochain_serialized_bytes::encode(&team).unwrap();
        let back: Team = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(team, back);
    }

    #[test]
    fn serde_roundtrip_assignment() {
        let assignment = make_assignment();
        let bytes = holochain_serialized_bytes::encode(&assignment).unwrap();
        let back: Assignment = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(assignment, back);
    }

    #[test]
    fn serde_roundtrip_situation_report() {
        let sitrep = make_sitrep();
        let bytes = holochain_serialized_bytes::encode(&sitrep).unwrap();
        let back: SituationReport = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(sitrep, back);
    }

    #[test]
    fn serde_roundtrip_checkpoint() {
        let cp = make_checkpoint();
        let bytes = holochain_serialized_bytes::encode(&cp).unwrap();
        let back: Checkpoint = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(cp, back);
    }

    // ========================================================================
    // UNICODE AND SPECIAL CHARACTER EDGE CASES
    // ========================================================================

    #[test]
    fn team_unicode_id_and_name_passes() {
        let lead = agent();
        let team = Team {
            id: "\u{1F525}\u{6551}\u{63F4}\u{968A}".into(), // fire emoji + CJK
            name: "\u{041A}\u{043E}\u{043C}\u{0430}\u{043D}\u{0434}\u{0430} \u{0410}\u{043B}\u{044C}\u{0444}\u{0430}".into(), // Cyrillic
            team_type: TeamType::SearchAndRescue,
            members: vec![lead.clone()],
            lead,
            assigned_zone: None,
            status: TeamStatus::Active,
        };
        let result = validate_create_team(fake_create(), team);
        assert!(is_valid(&result));
    }

    #[test]
    fn assignment_unicode_objective_passes() {
        let mut assignment = make_assignment();
        assignment.objective =
            "\u{7DCA}\u{6025}\u{907F}\u{96E3}\u{6240}\u{3092}\u{78BA}\u{4FDD}\u{3059}\u{308B}"
                .into(); // Japanese
        let result = validate_create_assignment(fake_create(), assignment);
        assert!(is_valid(&result));
    }

    #[test]
    fn sitrep_unicode_conditions_passes() {
        let mut sitrep = make_sitrep();
        sitrep.conditions = "\u{00C1}rea inundada, peligro de derrumbe \u{26A0}\u{FE0F}".into(); // Spanish + warning sign
        let result = validate_create_sitrep(fake_create(), sitrep);
        assert!(is_valid(&result));
    }

    #[test]
    fn sitrep_unicode_resources_and_hazards_passes() {
        let mut sitrep = make_sitrep();
        sitrep.resources_needed = vec![
            "\u{6551}\u{6025}\u{7BB1}".into(), // Japanese: first aid kit
            "\u{98DF}\u{7CE7}".into(),         // Chinese: food rations
        ];
        sitrep.hazards = vec![
            "\u{653E}\u{5C04}\u{7DDA}".into(), // Japanese: radiation
        ];
        let result = validate_create_sitrep(fake_create(), sitrep);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // MAX VALUE AND LARGE INPUT EDGE CASES
    // ========================================================================

    #[test]
    fn sitrep_max_casualties_passes() {
        let mut sitrep = make_sitrep();
        sitrep.casualties_found = u32::MAX;
        let result = validate_create_sitrep(fake_create(), sitrep);
        assert!(is_valid(&result));
    }

    #[test]
    fn team_many_members_passes() {
        let lead = agent();
        let mut members: Vec<AgentPubKey> = (0..100u8)
            .map(|i| AgentPubKey::from_raw_36(vec![i; 36]))
            .collect();
        // Ensure lead is in members
        members.push(lead.clone());
        let team = Team {
            id: "large-team".into(),
            name: "Mass Volunteer Corps".into(),
            team_type: TeamType::Volunteer,
            members,
            lead,
            assigned_zone: None,
            status: TeamStatus::Active,
        };
        let result = validate_create_team(fake_create(), team);
        assert!(is_valid(&result));
    }

    #[test]
    fn team_lead_duplicated_in_members_passes() {
        let lead = agent();
        let team = Team {
            id: "dup-test".into(),
            name: "Dup Lead Team".into(),
            team_type: TeamType::Medical,
            members: vec![lead.clone(), lead.clone(), agent2()],
            lead,
            assigned_zone: None,
            status: TeamStatus::Forming,
        };
        let result = validate_create_team(fake_create(), team);
        assert!(is_valid(&result));
    }

    #[test]
    fn sitrep_many_resources_and_hazards_passes() {
        let mut sitrep = make_sitrep();
        sitrep.resources_needed = (0..50).map(|i| format!("resource-{}", i)).collect();
        sitrep.hazards = (0..50).map(|i| format!("hazard-{}", i)).collect();
        let result = validate_create_sitrep(fake_create(), sitrep);
        assert!(is_valid(&result));
    }

    #[test]
    fn assignment_long_objective_rejected() {
        let mut assignment = make_assignment();
        assignment.objective = "A".repeat(10_000);
        let result = validate_create_assignment(fake_create(), assignment);
        assert!(is_invalid(&result));
    }

    #[test]
    fn sitrep_long_conditions_rejected() {
        let mut sitrep = make_sitrep();
        sitrep.conditions = "Flooding ".repeat(1_000); // 9,000 chars > 8192 limit
        let result = validate_create_sitrep(fake_create(), sitrep);
        assert!(is_invalid(&result));
    }

    // ========================================================================
    // FLOATING POINT EDGE CASES (CHECKPOINT)
    // ========================================================================

    #[test]
    fn checkpoint_negative_zero_lat_lon_passes() {
        let mut cp = make_checkpoint();
        cp.lat = -0.0;
        cp.lon = -0.0;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn checkpoint_nan_lat_rejected() {
        let mut cp = make_checkpoint();
        cp.lat = f64::NAN;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be a finite number");
    }

    #[test]
    fn checkpoint_nan_lon_rejected() {
        let mut cp = make_checkpoint();
        cp.lon = f64::NAN;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Longitude must be a finite number");
    }

    #[test]
    fn checkpoint_positive_infinity_lat_rejected() {
        let mut cp = make_checkpoint();
        cp.lat = f64::INFINITY;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be a finite number");
    }

    #[test]
    fn checkpoint_negative_infinity_lat_rejected() {
        let mut cp = make_checkpoint();
        cp.lat = f64::NEG_INFINITY;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be a finite number");
    }

    #[test]
    fn checkpoint_positive_infinity_lon_rejected() {
        let mut cp = make_checkpoint();
        cp.lon = f64::INFINITY;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Longitude must be a finite number");
    }

    #[test]
    fn checkpoint_negative_infinity_lon_rejected() {
        let mut cp = make_checkpoint();
        cp.lon = f64::NEG_INFINITY;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Longitude must be a finite number");
    }

    #[test]
    fn checkpoint_subnormal_lat_lon_passes() {
        let mut cp = make_checkpoint();
        cp.lat = f64::MIN_POSITIVE; // smallest positive normal
        cp.lon = -f64::MIN_POSITIVE;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // ANCHOR TESTS
    // ========================================================================

    #[test]
    fn anchor_serde_roundtrip_empty_string() {
        let anchor = Anchor("".into());
        let bytes = holochain_serialized_bytes::encode(&anchor).unwrap();
        let back: Anchor = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(anchor, back);
    }

    #[test]
    fn anchor_serde_roundtrip_unicode() {
        let anchor = Anchor("\u{1F30D}\u{7DCA}\u{6025}".into());
        let bytes = holochain_serialized_bytes::encode(&anchor).unwrap();
        let back: Anchor = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(anchor, back);
    }

    // ========================================================================
    // CLONE / EQUALITY TRAIT TESTS
    // ========================================================================

    #[test]
    fn team_clone_is_equal() {
        let team = make_team();
        let cloned = team.clone();
        assert_eq!(team, cloned);
    }

    #[test]
    fn assignment_clone_is_equal() {
        let assignment = make_assignment();
        let cloned = assignment.clone();
        assert_eq!(assignment, cloned);
    }

    #[test]
    fn sitrep_clone_is_equal() {
        let sitrep = make_sitrep();
        let cloned = sitrep.clone();
        assert_eq!(sitrep, cloned);
    }

    #[test]
    fn checkpoint_clone_is_equal() {
        let cp = make_checkpoint();
        let cloned = cp.clone();
        assert_eq!(cp, cloned);
    }

    // ========================================================================
    // COMBINED INVALID FIELD PRIORITY TESTS
    // ========================================================================

    #[test]
    fn team_empty_id_takes_priority_over_empty_name() {
        let lead = agent();
        let team = Team {
            id: "".into(),
            name: "".into(),
            team_type: TeamType::Medical,
            members: vec![lead.clone()],
            lead,
            assigned_zone: None,
            status: TeamStatus::Active,
        };
        let result = validate_create_team(fake_create(), team);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Team ID cannot be empty");
    }

    #[test]
    fn team_empty_name_takes_priority_over_empty_members() {
        let lead = agent();
        let team = Team {
            id: "t-1".into(),
            name: "".into(),
            team_type: TeamType::Logistics,
            members: vec![],
            lead,
            assigned_zone: None,
            status: TeamStatus::Active,
        };
        let result = validate_create_team(fake_create(), team);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Team name cannot be empty");
    }

    #[test]
    fn team_empty_members_takes_priority_over_lead_not_member() {
        let team = Team {
            id: "t-1".into(),
            name: "Team Name".into(),
            team_type: TeamType::Shelter,
            members: vec![],
            lead: agent3(),
            assigned_zone: None,
            status: TeamStatus::Active,
        };
        let result = validate_create_team(fake_create(), team);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Team must have at least one member");
    }

    #[test]
    fn checkpoint_lat_invalid_takes_priority_over_lon_invalid() {
        let mut cp = make_checkpoint();
        cp.lat = -91.0;
        cp.lon = -181.0;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    #[test]
    fn checkpoint_lon_invalid_only_when_lat_valid() {
        let mut cp = make_checkpoint();
        cp.lat = 45.0;
        cp.lon = 200.0;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    // ========================================================================
    // SITREP SYNCED FLAG
    // ========================================================================

    #[test]
    fn sitrep_synced_true_passes() {
        let mut sitrep = make_sitrep();
        sitrep.synced = true;
        let result = validate_create_sitrep(fake_create(), sitrep);
        assert!(is_valid(&result));
    }

    #[test]
    fn sitrep_synced_false_passes() {
        let mut sitrep = make_sitrep();
        sitrep.synced = false;
        let result = validate_create_sitrep(fake_create(), sitrep);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // LINK TAG VALIDATION TESTS
    // ========================================================================

    fn validate_create_link_tag(
        link_type: LinkTypes,
        tag: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag_len = tag.len();
        match link_type {
            LinkTypes::TeamToSitrep => {
                if tag_len > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            _ => {
                if tag_len > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    fn validate_delete_link_tag(tag: Vec<u8>) -> ExternResult<ValidateCallbackResult> {
        if tag.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Delete link tag too long (max 256 bytes)".into(),
            ));
        }
        Ok(ValidateCallbackResult::Valid)
    }

    // -- AllTeams (256 limit) --

    #[test]
    fn all_teams_link_tag_valid() {
        let result = validate_create_link_tag(LinkTypes::AllTeams, vec![0u8; 100]);
        assert!(is_valid(&result));
    }

    #[test]
    fn all_teams_link_tag_at_limit() {
        let result = validate_create_link_tag(LinkTypes::AllTeams, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn all_teams_link_tag_over_limit() {
        let result = validate_create_link_tag(LinkTypes::AllTeams, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    // -- TeamToSitrep (512 limit) --

    #[test]
    fn team_to_sitrep_link_tag_valid() {
        let result = validate_create_link_tag(LinkTypes::TeamToSitrep, vec![0u8; 400]);
        assert!(is_valid(&result));
    }

    #[test]
    fn team_to_sitrep_link_tag_at_limit() {
        let result = validate_create_link_tag(LinkTypes::TeamToSitrep, vec![0u8; 512]);
        assert!(is_valid(&result));
    }

    #[test]
    fn team_to_sitrep_link_tag_over_limit() {
        let result = validate_create_link_tag(LinkTypes::TeamToSitrep, vec![0u8; 513]);
        assert!(is_invalid(&result));
    }

    // -- AgentToCheckpoint (256 limit) --

    #[test]
    fn agent_to_checkpoint_link_tag_at_limit() {
        let result = validate_create_link_tag(LinkTypes::AgentToCheckpoint, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn agent_to_checkpoint_link_tag_over_limit() {
        let result = validate_create_link_tag(LinkTypes::AgentToCheckpoint, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    // -- Large tag DoS prevention --

    #[test]
    fn massive_link_tag_rejected_all_types() {
        let huge_tag = vec![0xFFu8; 10_000];
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::AllTeams,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::ActiveTeams,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::TeamToAssignment,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::TeamToSitrep,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::ZoneToTeam,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::AgentToCheckpoint,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::AgentToTeam,
            huge_tag.clone()
        )));
    }

    // -- Delete link tag tests --

    #[test]
    fn delete_link_tag_valid() {
        let result = validate_delete_link_tag(vec![0u8; 128]);
        assert!(is_valid(&result));
    }

    #[test]
    fn delete_link_tag_at_limit() {
        let result = validate_delete_link_tag(vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn delete_link_tag_over_limit() {
        let result = validate_delete_link_tag(vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    // ========================================================================
    // HARDENING: STRING LENGTH LIMITS
    // ========================================================================

    #[test]
    fn team_id_too_long_rejected() {
        let mut team = make_team();
        team.id = "T".repeat(257);
        let result = validate_create_team(fake_create(), team);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Team ID must be 256 characters or fewer"
        );
    }

    #[test]
    fn team_id_at_limit_accepted() {
        let mut team = make_team();
        team.id = "T".repeat(256);
        let result = validate_create_team(fake_create(), team);
        assert!(is_valid(&result));
    }

    #[test]
    fn team_name_too_long_rejected() {
        let mut team = make_team();
        team.name = "N".repeat(513);
        let result = validate_create_team(fake_create(), team);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Team name must be 512 characters or fewer"
        );
    }

    #[test]
    fn team_name_at_limit_accepted() {
        let mut team = make_team();
        team.name = "N".repeat(512);
        let result = validate_create_team(fake_create(), team);
        assert!(is_valid(&result));
    }

    #[test]
    fn assignment_objective_too_long_rejected() {
        let mut assignment = make_assignment();
        assignment.objective = "O".repeat(4097);
        let result = validate_create_assignment(fake_create(), assignment);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Assignment objective must be 4096 characters or fewer"
        );
    }

    #[test]
    fn assignment_objective_at_limit_accepted() {
        let mut assignment = make_assignment();
        assignment.objective = "O".repeat(4096);
        let result = validate_create_assignment(fake_create(), assignment);
        assert!(is_valid(&result));
    }

    #[test]
    fn sitrep_conditions_too_long_rejected() {
        let mut sitrep = make_sitrep();
        sitrep.conditions = "C".repeat(8193);
        let result = validate_create_sitrep(fake_create(), sitrep);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "SITREP conditions must be 8192 characters or fewer"
        );
    }

    #[test]
    fn sitrep_conditions_at_limit_accepted() {
        let mut sitrep = make_sitrep();
        sitrep.conditions = "C".repeat(8192);
        let result = validate_create_sitrep(fake_create(), sitrep);
        assert!(is_valid(&result));
    }

    #[test]
    fn checkpoint_both_nan_rejected() {
        let mut cp = make_checkpoint();
        cp.lat = f64::NAN;
        cp.lon = f64::NAN;
        let result = validate_create_checkpoint(fake_create(), cp);
        assert!(is_invalid(&result));
        // Lat check comes first
        assert_eq!(invalid_msg(&result), "Latitude must be a finite number");
    }
}
