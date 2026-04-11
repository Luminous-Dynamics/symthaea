// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Incidents Coordinator Zome
//! Business logic for disaster declaration and lifecycle management

use emergency_incidents_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{
    civic_requirement_basic, civic_requirement_proposal, civic_requirement_voting,
    GovernanceEligibility,
};
use mycelix_zome_helpers::{get_latest_record};


/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Declare a new disaster

#[hdk_extern]
pub fn declare_disaster(input: DeclareDisasterInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "declare_disaster")?;
    if input.title.is_empty() || input.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Title must be 1-256 characters".into()
        )));
    }
    if input.description.is_empty() || input.description.len() > 8192 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be 1-8192 characters".into()
        )));
    }
    if input.affected_area.radius_km <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Affected area radius must be positive".into()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?;

    // Capture geo coordinates before affected_area is moved
    let disaster_center_lat = input.affected_area.center_lat;
    let disaster_center_lon = input.affected_area.center_lon;

    let disaster = Disaster {
        id: input.id.clone(),
        disaster_type: input.disaster_type.clone(),
        title: input.title,
        description: input.description,
        severity: input.severity,
        declared_by: agent_info.agent_initial_pubkey.clone(),
        declared_at: now,
        affected_area: input.affected_area,
        status: DisasterStatus::Declared,
        estimated_affected: input.estimated_affected,
        coordination_lead: input.coordination_lead,
    };

    let action_hash = create_entry(&EntryTypes::Disaster(disaster))?;

    // Link to all disasters anchor
    create_entry(&EntryTypes::Anchor(Anchor("all_disasters".to_string())))?;
    create_link(
        anchor_hash("all_disasters")?,
        action_hash.clone(),
        LinkTypes::AllDisasters,
        (),
    )?;

    // Link to active disasters anchor
    create_entry(&EntryTypes::Anchor(Anchor("active_disasters".to_string())))?;
    create_link(
        anchor_hash("active_disasters")?,
        action_hash.clone(),
        LinkTypes::ActiveDisasters,
        (),
    )?;

    // Link by disaster type
    let type_anchor = format!("disaster_type:{:?}", input.disaster_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::DisasterByType,
        (),
    )?;

    // Link agent to disaster
    create_link(
        agent_info.agent_initial_pubkey,
        action_hash.clone(),
        LinkTypes::AgentToDisaster,
        (),
    )?;

    // Geo-spatial index for disaster center
    // Coordinates captured before affected_area was moved into the disaster entry
    let geo_hash = commons_types::geo::geohash_encode(
        disaster_center_lat,
        disaster_center_lon,
        6,
    );
    let geo_anchor_str = format!("geo:{}", geo_hash);
    create_entry(&EntryTypes::Anchor(Anchor(geo_anchor_str.clone())))?;
    create_link(
        anchor_hash(&geo_anchor_str)?,
        action_hash.clone(),
        LinkTypes::GeoIndex,
        geo_hash.as_bytes().to_vec(),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created disaster".into()
    )))
}

/// Input for declaring a disaster
#[derive(Serialize, Deserialize, Debug)]
pub struct DeclareDisasterInput {
    pub id: String,
    pub disaster_type: DisasterType,
    pub title: String,
    pub description: String,
    pub severity: SeverityLevel,
    pub affected_area: AffectedArea,
    pub estimated_affected: u32,
    pub coordination_lead: Option<AgentPubKey>,
}

/// Get all active disasters
#[hdk_extern]
pub fn get_active_disasters(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("active_disasters")?, LinkTypes::ActiveDisasters)?,
        GetStrategy::default(),
    )?;

    let mut disasters = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            disasters.push(record);
        }
    }

    Ok(disasters)
}

/// Update a disaster's status
#[hdk_extern]
pub fn update_disaster_status(input: UpdateDisasterStatusInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "update_disaster_status")?;
    let current_record = get(input.disaster_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Disaster not found".into())),
    )?;

    let current_disaster: Disaster = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid disaster entry".into()
        )))?;

    let updated_disaster = Disaster {
        status: input.new_status.clone(),
        ..current_disaster
    };

    let new_action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::Disaster(updated_disaster),
    )?;

    // If closing or in recovery, remove from active disasters
    if matches!(
        input.new_status,
        DisasterStatus::Closed | DisasterStatus::Recovery
    ) {
        let links = get_links(
            LinkQuery::try_new(anchor_hash("active_disasters")?, LinkTypes::ActiveDisasters)?,
            GetStrategy::default(),
        )?;
        for link in links {
            let target = ActionHash::try_from(link.target.clone());
            if let Ok(target_hash) = target {
                if target_hash == input.disaster_hash {
                    delete_link(link.create_link_hash, GetOptions::default())?;
                }
            }
        }
    }

    get_latest_record(new_action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated disaster".into()
    )))
}

/// Input for updating disaster status
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateDisasterStatusInput {
    pub disaster_hash: ActionHash,
    pub new_status: DisasterStatus,
}

/// Add an incident update to a disaster
#[hdk_extern]
pub fn add_incident_update(input: AddIncidentUpdateInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "add_incident_update")?;
    if input.content.is_empty() || input.content.len() > 8192 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Content must be 1-8192 characters".into()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?;

    let update = IncidentUpdate {
        disaster_hash: input.disaster_hash.clone(),
        author: agent_info.agent_initial_pubkey,
        timestamp: now,
        update_type: input.update_type,
        content: input.content,
    };

    let action_hash = create_entry(&EntryTypes::IncidentUpdate(update))?;

    // Link disaster to update
    create_link(
        input.disaster_hash,
        action_hash.clone(),
        LinkTypes::DisasterToUpdate,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created update".into()
    )))
}

/// Input for adding an incident update
#[derive(Serialize, Deserialize, Debug)]
pub struct AddIncidentUpdateInput {
    pub disaster_hash: ActionHash,
    pub update_type: UpdateType,
    pub content: String,
}

/// End a disaster (set status to Closed)
#[hdk_extern]
pub fn end_disaster(disaster_hash: ActionHash) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_voting(), "end_disaster")?;
    update_disaster_status(UpdateDisasterStatusInput {
        disaster_hash,
        new_status: DisasterStatus::Closed,
    })
}

/// Get the timeline of updates for a disaster
#[hdk_extern]
pub fn get_disaster_timeline(disaster_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(disaster_hash, LinkTypes::DisasterToUpdate)?,
        GetStrategy::default(),
    )?;

    let mut updates = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            updates.push(record);
        }
    }

    updates.sort_by(|a, b| {
        let ts_a = a.action().timestamp();
        let ts_b = b.action().timestamp();
        ts_a.cmp(&ts_b)
    });

    Ok(updates)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Coordinator input struct serde roundtrip tests
    // ========================================================================

    #[test]
    fn declare_disaster_input_serde_roundtrip() {
        let input = DeclareDisasterInput {
            id: "disaster-1".to_string(),
            disaster_type: DisasterType::Hurricane,
            title: "Hurricane Alpha".to_string(),
            description: "Category 4 hurricane approaching coast".to_string(),
            severity: SeverityLevel::Level4,
            affected_area: AffectedArea {
                center_lat: 29.76,
                center_lon: -95.37,
                radius_km: 150.0,
                boundary: None,
                zones: vec![],
            },
            estimated_affected: 500000,
            coordination_lead: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: DeclareDisasterInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "disaster-1");
        assert_eq!(decoded.title, "Hurricane Alpha");
        assert_eq!(decoded.estimated_affected, 500000);
        assert!(decoded.coordination_lead.is_none());
    }

    #[test]
    fn update_disaster_status_input_serde_roundtrip() {
        let input = UpdateDisasterStatusInput {
            disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            new_status: DisasterStatus::Recovery,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateDisasterStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, DisasterStatus::Recovery);
    }

    #[test]
    fn add_incident_update_input_serde_roundtrip() {
        let input = AddIncidentUpdateInput {
            disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            update_type: UpdateType::CasualtyReport,
            content: "15 confirmed injuries, 0 fatalities".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddIncidentUpdateInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.content, "15 confirmed injuries, 0 fatalities");
    }

    // ========================================================================
    // Integrity enum serde tests (all variants)
    // ========================================================================

    #[test]
    fn disaster_type_all_variants_serde() {
        let variants = vec![
            DisasterType::Hurricane,
            DisasterType::Earthquake,
            DisasterType::Wildfire,
            DisasterType::Flood,
            DisasterType::Tornado,
            DisasterType::Pandemic,
            DisasterType::Industrial,
            DisasterType::MassCasualty,
            DisasterType::CyberAttack,
            DisasterType::Infrastructure,
            DisasterType::Other("Chemical Spill".to_string()),
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: DisasterType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn severity_level_all_variants_serde() {
        let variants = vec![
            SeverityLevel::Level1,
            SeverityLevel::Level2,
            SeverityLevel::Level3,
            SeverityLevel::Level4,
            SeverityLevel::Level5,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: SeverityLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn disaster_status_all_variants_serde() {
        let variants = vec![
            DisasterStatus::Declared,
            DisasterStatus::Active,
            DisasterStatus::Recovery,
            DisasterStatus::Closed,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: DisasterStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn update_type_all_variants_serde() {
        let variants = vec![
            UpdateType::StatusChange,
            UpdateType::SeverityChange,
            UpdateType::AreaExpansion,
            UpdateType::AreaContraction,
            UpdateType::CasualtyReport,
            UpdateType::ResourceUpdate,
            UpdateType::WeatherUpdate,
            UpdateType::InfrastructureUpdate,
            UpdateType::EvacuationOrder,
            UpdateType::AllClear,
            UpdateType::General,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: UpdateType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn zone_priority_all_variants_serde() {
        let variants = vec![
            ZonePriority::Critical,
            ZonePriority::High,
            ZonePriority::Medium,
            ZonePriority::Low,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: ZonePriority = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn zone_status_all_variants_serde() {
        let variants = vec![
            ZoneStatus::Unassessed,
            ZoneStatus::Active,
            ZoneStatus::Cleared,
            ZoneStatus::Hazardous,
            ZoneStatus::Evacuated,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: ZoneStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // AffectedArea struct serde roundtrip tests
    // ========================================================================

    #[test]
    fn affected_area_serde_roundtrip() {
        let area = AffectedArea {
            center_lat: 34.05,
            center_lon: -118.25,
            radius_km: 50.0,
            boundary: Some(vec![(34.0, -118.0), (34.1, -118.5)]),
            zones: vec![OperationalZone {
                id: "zone-1".to_string(),
                name: "Downtown".to_string(),
                boundary: vec![(34.05, -118.25), (34.06, -118.26)],
                priority: ZonePriority::Critical,
                status: ZoneStatus::Active,
            }],
        };
        let json = serde_json::to_string(&area).unwrap();
        let decoded: AffectedArea = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.center_lat, 34.05);
        assert_eq!(decoded.center_lon, -118.25);
        assert_eq!(decoded.radius_km, 50.0);
        assert_eq!(decoded.zones.len(), 1);
        assert_eq!(decoded.zones[0].name, "Downtown");
        assert_eq!(decoded.zones[0].priority, ZonePriority::Critical);
    }

    #[test]
    fn affected_area_no_boundary_serde() {
        let area = AffectedArea {
            center_lat: 0.0,
            center_lon: 0.0,
            radius_km: 10.0,
            boundary: None,
            zones: vec![],
        };
        let json = serde_json::to_string(&area).unwrap();
        let decoded: AffectedArea = serde_json::from_str(&json).unwrap();
        assert!(decoded.boundary.is_none());
        assert!(decoded.zones.is_empty());
    }

    // ========================================================================
    // Disaster struct full serde roundtrip
    // ========================================================================

    #[test]
    fn disaster_full_struct_serde_roundtrip() {
        let disaster = Disaster {
            id: "dis-42".to_string(),
            disaster_type: DisasterType::Earthquake,
            title: "Major Quake".to_string(),
            description: "7.2 magnitude earthquake".to_string(),
            severity: SeverityLevel::Level5,
            declared_by: AgentPubKey::from_raw_36(vec![1u8; 36]),
            declared_at: Timestamp::from_micros(1000000),
            affected_area: AffectedArea {
                center_lat: 37.77,
                center_lon: -122.42,
                radius_km: 200.0,
                boundary: Some(vec![(37.0, -123.0), (38.0, -122.0)]),
                zones: vec![],
            },
            status: DisasterStatus::Active,
            estimated_affected: 2_000_000,
            coordination_lead: Some(AgentPubKey::from_raw_36(vec![2u8; 36])),
        };
        let json = serde_json::to_string(&disaster).unwrap();
        let decoded: Disaster = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "dis-42");
        assert_eq!(decoded.disaster_type, DisasterType::Earthquake);
        assert_eq!(decoded.severity, SeverityLevel::Level5);
        assert_eq!(decoded.status, DisasterStatus::Active);
        assert_eq!(decoded.estimated_affected, 2_000_000);
        assert!(decoded.coordination_lead.is_some());
    }

    #[test]
    fn disaster_no_coordination_lead_serde() {
        let disaster = Disaster {
            id: "dis-43".to_string(),
            disaster_type: DisasterType::Wildfire,
            title: "Forest Fire".to_string(),
            description: "Spreading fire".to_string(),
            severity: SeverityLevel::Level3,
            declared_by: AgentPubKey::from_raw_36(vec![1u8; 36]),
            declared_at: Timestamp::from_micros(0),
            affected_area: AffectedArea {
                center_lat: 0.0,
                center_lon: 0.0,
                radius_km: 1.0,
                boundary: None,
                zones: vec![],
            },
            status: DisasterStatus::Declared,
            estimated_affected: 0,
            coordination_lead: None,
        };
        let json = serde_json::to_string(&disaster).unwrap();
        let decoded: Disaster = serde_json::from_str(&json).unwrap();
        assert!(decoded.coordination_lead.is_none());
        assert_eq!(decoded.estimated_affected, 0);
    }

    // ========================================================================
    // IncidentUpdate struct serde roundtrip
    // ========================================================================

    #[test]
    fn incident_update_full_struct_serde_roundtrip() {
        let update = IncidentUpdate {
            disaster_hash: ActionHash::from_raw_36(vec![3u8; 36]),
            author: AgentPubKey::from_raw_36(vec![4u8; 36]),
            timestamp: Timestamp::from_micros(999999),
            update_type: UpdateType::EvacuationOrder,
            content: "Mandatory evacuation for Zone A".to_string(),
        };
        let json = serde_json::to_string(&update).unwrap();
        let decoded: IncidentUpdate = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.update_type, UpdateType::EvacuationOrder);
        assert_eq!(decoded.content, "Mandatory evacuation for Zone A");
    }

    // ========================================================================
    // OperationalZone struct serde roundtrip
    // ========================================================================

    #[test]
    fn operational_zone_serde_roundtrip() {
        let zone = OperationalZone {
            id: "zone-alpha".to_string(),
            name: "Alpha Sector".to_string(),
            boundary: vec![(30.0, -90.0), (31.0, -91.0), (30.5, -90.5)],
            priority: ZonePriority::High,
            status: ZoneStatus::Hazardous,
        };
        let json = serde_json::to_string(&zone).unwrap();
        let decoded: OperationalZone = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "zone-alpha");
        assert_eq!(decoded.name, "Alpha Sector");
        assert_eq!(decoded.boundary.len(), 3);
        assert_eq!(decoded.priority, ZonePriority::High);
        assert_eq!(decoded.status, ZoneStatus::Hazardous);
    }

    #[test]
    fn operational_zone_empty_boundary_serde() {
        let zone = OperationalZone {
            id: "zone-empty".to_string(),
            name: "Empty Zone".to_string(),
            boundary: vec![],
            priority: ZonePriority::Low,
            status: ZoneStatus::Unassessed,
        };
        let json = serde_json::to_string(&zone).unwrap();
        let decoded: OperationalZone = serde_json::from_str(&json).unwrap();
        assert!(decoded.boundary.is_empty());
    }

    // ========================================================================
    // Boundary and edge-case tests for input validation logic
    // ========================================================================

    #[test]
    fn declare_disaster_input_empty_title_boundary() {
        let input = DeclareDisasterInput {
            id: "d-1".to_string(),
            disaster_type: DisasterType::Flood,
            title: String::new(),
            description: "Valid desc".to_string(),
            severity: SeverityLevel::Level1,
            affected_area: AffectedArea {
                center_lat: 0.0,
                center_lon: 0.0,
                radius_km: 1.0,
                boundary: None,
                zones: vec![],
            },
            estimated_affected: 1,
            coordination_lead: None,
        };
        // Empty title should serialize/deserialize fine (validation is coordinator-side)
        let json = serde_json::to_string(&input).unwrap();
        let decoded: DeclareDisasterInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.title.is_empty());
    }

    #[test]
    fn declare_disaster_input_max_length_title() {
        let input = DeclareDisasterInput {
            id: "d-2".to_string(),
            disaster_type: DisasterType::Tornado,
            title: "A".repeat(256),
            description: "Desc".to_string(),
            severity: SeverityLevel::Level2,
            affected_area: AffectedArea {
                center_lat: 45.0,
                center_lon: -90.0,
                radius_km: 100.0,
                boundary: None,
                zones: vec![],
            },
            estimated_affected: 100,
            coordination_lead: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: DeclareDisasterInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.title.len(), 256);
    }

    #[test]
    fn declare_disaster_input_max_description() {
        let input = DeclareDisasterInput {
            id: "d-3".to_string(),
            disaster_type: DisasterType::Pandemic,
            title: "Test".to_string(),
            description: "B".repeat(8192),
            severity: SeverityLevel::Level3,
            affected_area: AffectedArea {
                center_lat: 0.0,
                center_lon: 0.0,
                radius_km: 10.0,
                boundary: None,
                zones: vec![],
            },
            estimated_affected: 0,
            coordination_lead: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: DeclareDisasterInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.description.len(), 8192);
    }

    #[test]
    fn declare_disaster_input_zero_estimated_affected() {
        let input = DeclareDisasterInput {
            id: "d-4".to_string(),
            disaster_type: DisasterType::CyberAttack,
            title: "Cyber Incident".to_string(),
            description: "System breach".to_string(),
            severity: SeverityLevel::Level1,
            affected_area: AffectedArea {
                center_lat: 40.0,
                center_lon: -74.0,
                radius_km: 5.0,
                boundary: None,
                zones: vec![],
            },
            estimated_affected: 0,
            coordination_lead: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: DeclareDisasterInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.estimated_affected, 0);
    }

    #[test]
    fn declare_disaster_input_max_estimated_affected() {
        let input = DeclareDisasterInput {
            id: "d-5".to_string(),
            disaster_type: DisasterType::Infrastructure,
            title: "Grid Failure".to_string(),
            description: "Total grid failure".to_string(),
            severity: SeverityLevel::Level5,
            affected_area: AffectedArea {
                center_lat: 0.0,
                center_lon: 0.0,
                radius_km: 1000.0,
                boundary: None,
                zones: vec![],
            },
            estimated_affected: u32::MAX,
            coordination_lead: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: DeclareDisasterInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.estimated_affected, u32::MAX);
    }

    // ========================================================================
    // DisasterType::Other with various strings
    // ========================================================================

    #[test]
    fn disaster_type_other_empty_string_serde() {
        let variant = DisasterType::Other(String::new());
        let json = serde_json::to_string(&variant).unwrap();
        let decoded: DisasterType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, DisasterType::Other(String::new()));
    }

    #[test]
    fn disaster_type_other_unicode_string_serde() {
        let variant = DisasterType::Other("\u{6D2A}\u{6C34}".to_string()); // Chinese for "flood"
        let json = serde_json::to_string(&variant).unwrap();
        let decoded: DisasterType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, variant);
    }

    // ========================================================================
    // AffectedArea with multiple zones
    // ========================================================================

    #[test]
    fn affected_area_multiple_zones_serde() {
        let area = AffectedArea {
            center_lat: 29.76,
            center_lon: -95.37,
            radius_km: 300.0,
            boundary: Some(vec![(29.0, -96.0), (30.0, -95.0), (29.5, -94.5)]),
            zones: vec![
                OperationalZone {
                    id: "z-1".to_string(),
                    name: "Central".to_string(),
                    boundary: vec![(29.7, -95.3)],
                    priority: ZonePriority::Critical,
                    status: ZoneStatus::Active,
                },
                OperationalZone {
                    id: "z-2".to_string(),
                    name: "Outer Ring".to_string(),
                    boundary: vec![(29.8, -95.4), (29.9, -95.5)],
                    priority: ZonePriority::Medium,
                    status: ZoneStatus::Evacuated,
                },
                OperationalZone {
                    id: "z-3".to_string(),
                    name: "Cleared Area".to_string(),
                    boundary: vec![],
                    priority: ZonePriority::Low,
                    status: ZoneStatus::Cleared,
                },
            ],
        };
        let json = serde_json::to_string(&area).unwrap();
        let decoded: AffectedArea = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.zones.len(), 3);
        assert_eq!(decoded.zones[0].priority, ZonePriority::Critical);
        assert_eq!(decoded.zones[1].status, ZoneStatus::Evacuated);
        assert_eq!(decoded.zones[2].status, ZoneStatus::Cleared);
    }

    // ========================================================================
    // Add incident update input edge cases
    // ========================================================================

    #[test]
    fn add_incident_update_input_all_update_types() {
        let update_types = vec![
            UpdateType::StatusChange,
            UpdateType::SeverityChange,
            UpdateType::AreaExpansion,
            UpdateType::AreaContraction,
            UpdateType::CasualtyReport,
            UpdateType::ResourceUpdate,
            UpdateType::WeatherUpdate,
            UpdateType::InfrastructureUpdate,
            UpdateType::EvacuationOrder,
            UpdateType::AllClear,
            UpdateType::General,
        ];
        for ut in update_types {
            let input = AddIncidentUpdateInput {
                disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
                update_type: ut.clone(),
                content: format!("Content for {:?}", ut),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: AddIncidentUpdateInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.update_type, ut);
        }
    }

    #[test]
    fn add_incident_update_input_max_content() {
        let input = AddIncidentUpdateInput {
            disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            update_type: UpdateType::General,
            content: "X".repeat(8192),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddIncidentUpdateInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.content.len(), 8192);
    }

    // ========================================================================
    // Affected area with extreme coordinate values
    // ========================================================================

    #[test]
    fn affected_area_extreme_lat_lon_serde() {
        let area = AffectedArea {
            center_lat: -90.0,
            center_lon: 180.0,
            radius_km: 0.001,
            boundary: None,
            zones: vec![],
        };
        let json = serde_json::to_string(&area).unwrap();
        let decoded: AffectedArea = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.center_lat, -90.0);
        assert_eq!(decoded.center_lon, 180.0);
        assert!((decoded.radius_km - 0.001).abs() < f32::EPSILON);
    }

    #[test]
    fn affected_area_large_radius_serde() {
        let area = AffectedArea {
            center_lat: 0.0,
            center_lon: 0.0,
            radius_km: 20015.0, // half Earth circumference
            boundary: None,
            zones: vec![],
        };
        let json = serde_json::to_string(&area).unwrap();
        let decoded: AffectedArea = serde_json::from_str(&json).unwrap();
        assert!((decoded.radius_km - 20015.0).abs() < 1.0);
    }
}
